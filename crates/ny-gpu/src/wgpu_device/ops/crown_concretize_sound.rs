// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound GPU CROWN concretize (increment 1 of the sound GPU-resident backward).
//!
//! Computes per-spec `(lower, upper)` from a coefficient pair `(a_lower, a_upper)`,
//! their accumulated error `(a_lower_err, a_upper_err)`, the input box, and the
//! bias — widening each bound OUTWARD by the certified penalty
//! `Σ_j (err[j] + γ_n·|a[j]|)·max(|x_l[j]|,|x_u[j]|) + additive` so the result is
//! a SOUND enclosure under round-to-nearest f32. This is the on-device form of
//! the CPU `γ_n·S` certified-error concretization — the verdict-deciding step the
//! `sound_gpu_crown_required` gate currently forces onto the CPU.

use ny_core::{f32_to_f64_exact, NyError, Result};

use super::super::WgpuDevice;
use crate::wgpu_device::sound_consts::{combine_slack_f32, eft_r_slack_f32, gamma_k_f32};

/// Round a finite, non-negative f64 result toward +∞.
///
/// Every input to the fallback-radius proof originates as f32, so its products
/// remain far inside f64's normal range. Stepping each rounded result one f64 ULP
/// upward therefore gives a simple conservative enclosure without depending on
/// the host rounding mode or an error-prone tolerance.
fn round_up_nonnegative_f64(value: f64) -> Option<f64> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    if value == 0.0 {
        return Some(0.0);
    }
    let outward = value.next_up();
    outward.is_finite().then_some(outward)
}

fn add_up_nonnegative_f64(lhs: f64, rhs: f64) -> Option<f64> {
    if !lhs.is_finite() || !rhs.is_finite() || lhs < 0.0 || rhs < 0.0 {
        return None;
    }
    if lhs == 0.0 {
        return Some(rhs);
    }
    if rhs == 0.0 {
        return Some(lhs);
    }
    round_up_nonnegative_f64(lhs + rhs)
}

fn mul_up_nonnegative_f64(lhs: f64, rhs: f64) -> Option<f64> {
    if !lhs.is_finite() || !rhs.is_finite() || lhs < 0.0 || rhs < 0.0 {
        return None;
    }
    if lhs == 0.0 || rhs == 0.0 {
        return Some(0.0);
    }
    round_up_nonnegative_f64(lhs * rhs)
}

fn affine_radius_step_up(
    radius: f64,
    coefficient: f32,
    coefficient_error: f32,
    xmax: f64,
) -> Option<f64> {
    let coefficient_radius = add_up_nonnegative_f64(
        f32_to_f64_exact(coefficient).abs(),
        f32_to_f64_exact(coefficient_error),
    )?;
    let term = mul_up_nonnegative_f64(coefficient_radius, xmax)?;
    add_up_nonnegative_f64(radius, term)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoundConcretizeParams {
    num_specs: u32,
    input_dim: u32,
    gamma_n: f32,
    additive: f32,
    slack: f32,
    /// #batched-bab: per-domain spec-row count (reuses a padding slot). Each domain
    /// concretizes against its OWN input box; `== num_specs` (single domain) →
    /// domain index 0 → byte-identical.
    num_specs_per_dom: u32,
    /// #eft-err (former padding): 1 ⇒ the shader's barrier-fma EFT sequence with
    /// the MEASURED residual charge (·`eft_r_slack`) replaces the a-priori
    /// `γ_n·|a|` term. 0 ⇒ byte-identical legacy behavior.
    eft_mode: u32,
    eft_r_slack: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SoundConcretizeShape {
    num_specs_u32: u32,
    input_dim_u32: u32,
    num_specs_per_dom_u32: u32,
    coeff: usize,
    box_len: usize,
    output_bytes: u64,
}

fn concretize_checked_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        NyError::InvalidSpec(format!(
            "concretize_sound_gpu: {label}={value} exceeds WGSL u32 indexing"
        ))
    })
}

fn concretize_checked_bytes(elements: usize, label: &str) -> Result<u64> {
    u64::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<f32>() as u64))
        .ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "concretize_sound_gpu: {label} byte count overflows"
            ))
        })
}

fn sound_concretize_shape(
    num_specs: usize,
    num_specs_per_dom: usize,
    input_dim: usize,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> Result<SoundConcretizeShape> {
    if num_specs == 0 {
        return Err(NyError::InvalidSpec(
            "concretize_sound_gpu: shape preflight requires nonzero num_specs".into(),
        ));
    }
    if num_specs_per_dom == 0
        || num_specs_per_dom > num_specs
        || !num_specs.is_multiple_of(num_specs_per_dom)
    {
        return Err(NyError::InvalidSpec(format!(
            "concretize_sound_gpu: num_specs_per_dom={num_specs_per_dom} must be nonzero \
             and divide num_specs={num_specs}"
        )));
    }

    let n_domains = num_specs / num_specs_per_dom;
    let coeff = num_specs.checked_mul(input_dim).ok_or_else(|| {
        NyError::InvalidSpec("concretize_sound_gpu: num_specs*input_dim overflows usize".into())
    })?;
    let box_len = n_domains.checked_mul(input_dim).ok_or_else(|| {
        NyError::InvalidSpec("concretize_sound_gpu: n_domains*input_dim overflows usize".into())
    })?;
    let packed_err = coeff.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec("concretize_sound_gpu: packed error length overflows usize".into())
    })?;
    let packed_bias = num_specs.checked_mul(2).ok_or_else(|| {
        NyError::InvalidSpec("concretize_sound_gpu: packed bias length overflows usize".into())
    })?;

    let num_specs_u32 = concretize_checked_u32(num_specs, "num_specs")?;
    let input_dim_u32 = concretize_checked_u32(input_dim, "input_dim")?;
    let num_specs_per_dom_u32 = concretize_checked_u32(num_specs_per_dom, "num_specs_per_dom")?;
    // These products are formed directly in WGSL (`spec*input_dim`,
    // `domain*input_dim`, `coeff+idx`, and `num_specs+spec`). Checking only the
    // factors would allow modulo-u32 aliasing.
    concretize_checked_u32(coeff, "num_specs*input_dim")?;
    concretize_checked_u32(box_len, "n_domains*input_dim")?;
    concretize_checked_u32(packed_err, "2*num_specs*input_dim")?;
    concretize_checked_u32(packed_bias, "2*num_specs")?;

    let output_bytes = concretize_checked_bytes(num_specs.max(1), "output")?;
    for (label, elements) in [
        ("coefficient", coeff.max(1)),
        ("input box", box_len.max(1)),
        ("packed error", packed_err.max(1)),
        ("packed bias", packed_bias.max(1)),
        ("output", num_specs.max(1)),
    ] {
        let bytes = concretize_checked_bytes(elements, label)?;
        if bytes > max_buffer_size || bytes > max_storage_buffer_binding_size {
            return Err(NyError::UnsupportedOp(format!(
                "concretize_sound_gpu: {label} buffer needs {bytes} bytes, but \
                 max_buffer_size={max_buffer_size} and \
                 max_storage_buffer_binding_size={max_storage_buffer_binding_size}"
            )));
        }
    }

    Ok(SoundConcretizeShape {
        num_specs_u32,
        input_dim_u32,
        num_specs_per_dom_u32,
        coeff,
        box_len,
        output_bytes,
    })
}

fn concretize_max_workgroups(limit: u32) -> Result<usize> {
    if limit == 0 {
        return Err(NyError::UnsupportedOp(
            "concretize_sound_gpu: device reports zero \
             max_compute_workgroups_per_dimension"
                .into(),
        ));
    }
    Ok(limit as usize)
}

/// #u4 C1 (TAINT_GUARD_AUDIT.md §4): per-spec-row out-of-band taint words for
/// the host-preflight consult. One `u32` per spec row (`len == num_specs`),
/// pre-OR'd by the CALLER from every taint buffer feeding that row — the
/// coefficient words of the last taint-twin kernel in the chain, the err words
/// of the combine twin, and the host bias folds' words — under the canon rule
/// `taint_out = OR over inputs of (taint_in AND (partner_value != 0 OR
/// partner_taint != 0)) OR (op itself saturated/degraded)`. A clean exact-zero
/// partner authenticates annihilation; a tainted stored zero does not. Nonzero
/// = tainted. Reducing to one word per row keeps the consult a pure host check
/// next to the affine-radius proof: a tainted row is refused BEFORE dispatch,
/// with no extra shader binding.
pub(crate) type SpecRowTaint = [u32];

/// The C1 consult body, run by the `concretize_sound_gpu_batched` preflight
/// ONLY under `sentinel_taint_selfcheck::PRODUCTION_GUARDS_CONSULT_TAINT_WORD`
/// (the const gates WHETHER it runs; the arms below are what an ARMED build
/// enforces). Fail-closed on every arm: absent words refuse (a caller that
/// never ran the taint twins gets no verdict use), a wrong-length slice
/// refuses (row/word misalignment could hide a tainted row), and any nonzero
/// word refuses with the row named. Same typed `NyError::InvalidSpec` shape as
/// the affine-radius refusal, so ny-propagate's existing `Err` fallback
/// (gpu_suffix.rs) routes to the CPU sound backward unchanged.
pub(crate) fn consult_spec_row_taint(taint: Option<&SpecRowTaint>, num_specs: usize) -> Result<()> {
    let Some(rows) = taint else {
        return Err(NyError::InvalidSpec(
            "concretize_sound_gpu: taint words absent — refusing verdict use (fail-closed)".into(),
        ));
    };
    if rows.len() != num_specs {
        return Err(NyError::InvalidSpec(format!(
            "concretize_sound_gpu: taint.len()={} != num_specs={num_specs} — \
             refusing verdict use (fail-closed)",
            rows.len()
        )));
    }
    if let Some((spec, word)) = rows
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, word)| word != 0)
    {
        return Err(NyError::InvalidSpec(format!(
            "concretize_sound_gpu: spec row {spec} carries taint word {word:#010x} — \
             refusing verdict use (fail-closed)"
        )));
    }
    Ok(())
}

impl WgpuDevice {
    /// Allocation-free capacity preflight used by the typed intermediate sweep
    /// before it accepts a request and dispatches the resident walk.
    pub(super) fn intermediate_sweep_concretize_preflight(
        &self,
        num_specs: usize,
        input_dim: usize,
        input_lower: &[f32],
        input_upper: &[f32],
    ) -> Result<()> {
        let limits = self.device.limits();
        let _ = sound_concretize_shape(
            num_specs,
            num_specs,
            input_dim,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        )?;
        let max_wg = concretize_max_workgroups(limits.max_compute_workgroups_per_dimension)?;
        if num_specs > max_wg {
            return Err(NyError::UnsupportedOp(format!(
                "concretize_sound_gpu: num_specs {num_specs} exceeds \
                 max_compute_workgroups_per_dimension {max_wg}"
            )));
        }
        // Exercise every dimension-dependent certified constant before the
        // resident dispatch. This turns an unsupported reduction width into a
        // clean pre-dispatch decline rather than a post-walk failure.
        gamma_k_f32(input_dim)?;
        combine_slack_f32(input_dim)?;
        let _ = crate::wgpu_device::sound_consts::rung3_flush_safe_additive(
            u32::try_from(input_dim).map_err(|_| {
                NyError::InvalidSpec("concretize input dimension exceeds u32".into())
            })?,
        )?;
        if self
            .charged_flush_authority_cached()
            .is_some_and(|policy| policy.refuse_subnormal_inputs)
            && input_lower
                .iter()
                .chain(input_upper)
                .any(|value| *value != 0.0 && value.abs() < f32::MIN_POSITIVE)
        {
            return Err(NyError::UnsupportedOp(
                "#flush-charge: intermediate sweep input contains a subnormal endpoint".into(),
            ));
        }
        Ok(())
    }

    /// Sound concretization on the GPU. All slices are row-major `(num_specs ×
    /// input_dim)` for the coefficient/err matrices, length `input_dim` for the
    /// input box, length `num_specs` for the biases. Returns `(lower, upper)`,
    /// each length `num_specs`, a sound enclosure of the network output range.
    /// `taint` is the per-spec-row [`SpecRowTaint`] word slice for the armed #u4
    /// C1 consult (see [`concretize_sound_gpu_batched`] for the contract). The
    /// Rust type remains `Option` so missing words can be refused explicitly.
    ///
    /// Increment 1 of the sound GPU-resident CROWN backward (task #15). Currently
    /// a standalone, soundness-tested primitive; the `crown_backward_gpu`
    /// integration follows once per-layer error tracking (increments 2–7) lands.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn concretize_sound_gpu(
        &self,
        num_specs: usize,
        input_dim: usize,
        a_lower: &[f32],
        a_upper: &[f32],
        a_lower_err: &[f32],
        a_upper_err: &[f32],
        input_lower: &[f32],
        input_upper: &[f32],
        bias_lower: &[f32],
        bias_upper: &[f32],
        taint: Option<&SpecRowTaint>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // #batched-bab: single domain (per-domain count == total spec count).
        self.concretize_sound_gpu_batched(
            num_specs,
            num_specs,
            input_dim,
            a_lower,
            a_upper,
            a_lower_err,
            a_upper_err,
            input_lower,
            input_upper,
            bias_lower,
            bias_upper,
            taint,
        )
    }

    /// #batched-bab: domain-block form of [`concretize_sound_gpu`]. Rows are stacked in
    /// `n_domains = num_specs / num_specs_per_dom` blocks of `num_specs_per_dom` rows,
    /// and the input box is `n_domains * input_dim` wide — row `s` concretizes against
    /// its OWN domain's box `[dom*input_dim .. )`, `dom = s / num_specs_per_dom`
    /// (CROWN_CONCRETIZE_SOUND_SHADER dbase, HOLE 3). With `num_specs_per_dom ==
    /// num_specs` (single domain) this is byte-identical to the pre-batch path.
    ///
    /// #u4 C1: `taint` carries one [`SpecRowTaint`] word per spec row (`len ==
    /// num_specs`; the caller pre-ORs its coefficient/err/bias taint buffers down
    /// to one word per row). The reviewed production gate is currently armed:
    /// `None`, a wrong-length slice, or any nonzero word yields a typed refusal
    /// (fail-closed), with the same `Err` shape as the affine-radius proof.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn concretize_sound_gpu_batched(
        &self,
        num_specs: usize,
        num_specs_per_dom: usize,
        input_dim: usize,
        a_lower: &[f32],
        a_upper: &[f32],
        a_lower_err: &[f32],
        a_upper_err: &[f32],
        input_lower: &[f32],
        input_upper: &[f32],
        bias_lower: &[f32],
        bias_upper: &[f32],
        taint: Option<&SpecRowTaint>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let coeff = num_specs.checked_mul(input_dim).ok_or_else(|| {
            NyError::InvalidSpec("concretize_sound_gpu: num_specs*input_dim overflows usize".into())
        })?;
        for (name, len) in [
            ("a_lower", a_lower.len()),
            ("a_upper", a_upper.len()),
            ("a_lower_err", a_lower_err.len()),
            ("a_upper_err", a_upper_err.len()),
        ] {
            if len != coeff {
                return Err(NyError::InvalidSpec(format!(
                    "concretize_sound_gpu: {name}.len()={len} != num_specs*input_dim={coeff}"
                )));
            }
        }
        if bias_lower.len() != num_specs || bias_upper.len() != num_specs {
            return Err(NyError::shape_mismatch(
                vec![num_specs],
                vec![bias_lower.len()],
            ));
        }
        if num_specs == 0 {
            return Ok((vec![], vec![]));
        }
        let limits = self.device.limits();
        let shape = sound_concretize_shape(
            num_specs,
            num_specs_per_dom,
            input_dim,
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        )?;
        debug_assert_eq!(shape.coeff, coeff);
        // Per-domain input boxes are stacked n_domains-wide; single domain → input_dim.
        let box_len = shape.box_len;
        if input_lower.len() != box_len || input_upper.len() != box_len {
            return Err(NyError::shape_mismatch(
                vec![box_len],
                vec![input_lower.len()],
            ));
        }

        for (index, (&lower, &upper)) in input_lower.iter().zip(input_upper).enumerate() {
            let lower_bits = lower.to_bits();
            let upper_bits = upper.to_bits();
            let lower_finite = lower_bits & 0x7f80_0000 != 0x7f80_0000;
            let upper_finite = upper_bits & 0x7f80_0000 != 0x7f80_0000;
            if !lower_finite || !upper_finite || f32_to_f64_exact(lower) > f32_to_f64_exact(upper) {
                return Err(NyError::InvalidSpec(format!(
                    "concretize_sound_gpu: invalid input interval at {index}: [{lower}, {upper}]"
                )));
            }
        }
        for (name, values) in [
            ("a_lower", a_lower),
            ("a_upper", a_upper),
            ("bias_lower", bias_lower),
            ("bias_upper", bias_upper),
        ] {
            if let Some((index, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| value.to_bits() & 0x7f80_0000 == 0x7f80_0000)
            {
                return Err(NyError::InvalidSpec(format!(
                    "concretize_sound_gpu: {name}[{index}] is non-finite ({value})"
                )));
            }
        }
        for (name, values) in [("a_lower_err", a_lower_err), ("a_upper_err", a_upper_err)] {
            if let Some((index, value)) = values.iter().copied().enumerate().find(|(_, value)| {
                let bits = value.to_bits();
                let negative_nonzero = bits & 0x8000_0000 != 0 && bits & 0x7fff_ffff != 0;
                bits & 0x7f80_0000 == 0x7f80_0000 || negative_nonzero
            }) {
                return Err(NyError::InvalidSpec(format!(
                    "concretize_sound_gpu: {name}[{index}] is not a finite nonnegative \
                     error bound ({value})"
                )));
            }
        }

        // The shader's ±FALLBACK_BOUND repair is sound only when it encloses the
        // exact affine range.  Prove that precondition in f64 before dispatch,
        // including coefficient uncertainty, rather than treating a finite
        // sentinel as mathematical infinity. Resident GEMM overflow is tainted
        // through the coefficient-error channel (1e30); a value-kernel
        // FALLBACK_BOUND or that error taint necessarily fails this same proof
        // unless a later exact zero map has legitimately removed its effect.
        let fallback = f64::from(crate::FALLBACK_BOUND);
        for spec in 0..num_specs {
            let domain = spec / num_specs_per_dom;
            let mut lower_radius = Some(f32_to_f64_exact(bias_lower[spec]).abs());
            let mut upper_radius = Some(f32_to_f64_exact(bias_upper[spec]).abs());
            for j in 0..input_dim {
                let coeff_index = spec * input_dim + j;
                let box_index = domain * input_dim + j;
                let xmax = f32_to_f64_exact(input_lower[box_index])
                    .abs()
                    .max(f32_to_f64_exact(input_upper[box_index]).abs());
                lower_radius = lower_radius.and_then(|radius| {
                    affine_radius_step_up(
                        radius,
                        a_lower[coeff_index],
                        a_lower_err[coeff_index],
                        xmax,
                    )
                });
                upper_radius = upper_radius.and_then(|radius| {
                    affine_radius_step_up(
                        radius,
                        a_upper[coeff_index],
                        a_upper_err[coeff_index],
                        xmax,
                    )
                });
            }
            let (Some(lower_radius), Some(upper_radius)) = (lower_radius, upper_radius) else {
                return Err(NyError::InvalidSpec(format!(
                    "concretize_sound_gpu: spec row {spec} affine radius \
                     overflowed while proving enclosure by FALLBACK_BOUND={fallback}"
                )));
            };
            if lower_radius >= fallback || upper_radius >= fallback {
                return Err(NyError::InvalidSpec(format!(
                    "concretize_sound_gpu: spec row {spec} outward affine radius \
                     ({lower_radius}, {upper_radius}) is not enclosed by \
                     FALLBACK_BOUND={fallback}"
                )));
            }
        }
        // #u4 C1 (TAINT_GUARD_AUDIT.md §4): the out-of-band taint-word consult,
        // alongside the affine-radius proof above. That proof is MAGNITUDE-only
        // — a downscaled (laundered) sentinel passes it trivially — while the
        // OR-carried word survives laundering by construction. The compile-time
        // gate is currently armed by the reviewed source change in
        // ops/sentinel_taint_selfcheck.rs; it is never a runtime decision.
        if super::sentinel_taint_selfcheck::PRODUCTION_GUARDS_CONSULT_TAINT_WORD {
            consult_spec_row_taint(taint, num_specs)?;
        }
        // #wg-limit-guard (SOUNDNESS, fail-closed): this shader dispatches ONE workgroup
        // per spec row (`wg_id.x = spec_row`), so `dispatch_workgroups(num_specs)`
        // overruns the wgpu `max_compute_workgroups_per_dimension` cap (default 65535,
        // NOT raised by `NY_GPU_BIG_BINDINGS`) once `num_specs > max_wg`. An over-limit
        // dispatch is UB on some drivers (silently wrong — closer-to-zero, UNSOUND —
        // bound, or a crash), so fail closed and let the caller sub-chunk / fall back to
        // the sound CPU concretize. Value-neutral for every in-range call.
        let max_wg = concretize_max_workgroups(limits.max_compute_workgroups_per_dimension)?;
        if num_specs > max_wg {
            return Err(NyError::UnsupportedOp(format!(
                "concretize_sound_gpu: num_specs {num_specs} exceeds \
                 max_compute_workgroups_per_dimension {max_wg} — sub-chunk the batch"
            )));
        }

        // γ_n = n·u/(1−n·u) (u = 2⁻²⁴) bounds the concretize dot's f32 rounding.
        // `additive` = weight-INDEPENDENT normal-range underflow floor (survives Metal
        // FTZ, unlike the old 8·n·η subnormal one which flushed to 0); the on-device
        // `flushacc·slack·F32_MIN_NORMAL` term (shader §0) adds the amplified
        // operand-flush cover a reduction over huge-dynamic-range a·x actually needs.
        // #eft-err: the measured-residual concretize. Cached-only gate read (this
        // op runs inside a GPU-checked section — see the deadlock guard note in
        // eft_selfcheck.rs); uninitialized/refused ⇒ legacy γ_n, byte-identical.
        let eft_on = std::env::var("NY_EFT_ERR").ok().as_deref() == Some("1")
            && self.eft_primitives_cached();
        // #flush-charge: on a charged-flush device the legacy concretize slack
        // is widened by the oracle-derived factor (the shader carries ONE
        // per-tap max charge against up to four first-order DAZ channels), the
        // EFT arm is refused belt-and-braces (`eft_primitives_cached()` is
        // already false on a flushing adapter), and subnormal input-box
        // endpoints are refused (a DAZ-zeroed `x` under a large accumulated
        // coefficient error `e` loses `e·2^-126`, which no flushacc term
        // bounds). `None` on every other device ⇒ byte-identical.
        let charged_policy = self.charged_flush_authority_cached();
        if let Some(policy) = charged_policy {
            if policy.eft_forbidden && eft_on {
                return Err(NyError::UnsupportedOp(
                    "#flush-charge: the EFT concretize arm is refused under \
                     charged-flush authority (fail-closed)"
                        .into(),
                ));
            }
            if policy.refuse_subnormal_inputs
                && input_lower
                    .iter()
                    .chain(input_upper.iter())
                    .any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE)
            {
                return Err(NyError::UnsupportedOp(
                    "#flush-charge: the input box contains a SUBNORMAL \
                     endpoint; its DAZ loss under an accumulated coefficient \
                     error is un-chargeable — refusing (fail-closed)"
                        .into(),
                ));
            }
        }
        let gamma_n = gamma_k_f32(input_dim)?;
        let slack = match charged_policy {
            Some(policy) => crate::wgpu_device::sound_consts::charged_concretize_slack(
                combine_slack_f32(input_dim)?,
                policy.concretize_slack_factor,
            )?,
            None => combine_slack_f32(input_dim)?,
        };
        let eft_r_slack = if eft_on {
            let residual_terms = input_dim.checked_mul(2).ok_or_else(|| {
                NyError::UnsupportedOp(
                    "concretize_sound_gpu: EFT residual term count overflows usize".into(),
                )
            })?;
            eft_r_slack_f32(residual_terms)?
        } else {
            0.0
        };
        // The admitted rung-3 loss occurs inside the residual lane, before the
        // shader multiplies that lane by eft_r_slack. Scale the floor by the
        // identical outward multiplier; adding only the base afterward would
        // under-charge every r_slack > 1.
        let additive = if eft_on {
            super::super::sound_consts::rung3_flush_safe_additive_scaled(
                shape.input_dim_u32,
                eft_r_slack,
            )?
        } else {
            super::super::sound_consts::rung3_flush_safe_additive(shape.input_dim_u32)?
        };
        let params = SoundConcretizeParams {
            num_specs: shape.num_specs_u32,
            input_dim: shape.input_dim_u32,
            gamma_n,
            additive,
            slack,
            num_specs_per_dom: shape.num_specs_per_dom_u32,
            eft_mode: u32::from(eft_on),
            // 4 residual terms per tap over the strided dot + the tree
            // reduction's captured adds + final-assembly headroom: the
            // γ_{2·(2n)+2} cover of eft_r_slack_f32(2n) dominates the count.
            eft_r_slack,
        };
        self.run_gpu_checked_with_crown_deadline("concretize_sound_gpu", || {
            // Ordinary callers perform first materialization inside these
            // checked scopes. An authoritative sweep requires the same cache
            // to exist before acceptance, so this lookup cannot compile there.
            let (pipeline, layout) = self.sound_concretize_pipeline();
            // A requested DenormPreserve creation may have failed while the
            // cached adapter probe had authorized EFT. Refuse instead of
            // dispatching through a loading path the receipt did not attest.
            if eft_on && !self.denorm_preserve_contract_intact() {
                return Err(NyError::UnsupportedOp(
                    "sound GPU concretize EFT refused: a requested DenormPreserve \
                     shader module fell back to plain WGSL, so the cached rung-3 \
                     qualification no longer covers this production pipeline"
                        .into(),
                ));
            }

            let storage = |data: &[f32], label: &str| -> wgpu::Buffer {
                let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (data.len().max(1) * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if !data.is_empty() {
                    self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
                    super::intermediate_sweep::note_host_to_device(
                        data.len().saturating_mul(size_of::<f32>()),
                    );
                }
                buf
            };
            let out_buf = |label: &str| -> wgpu::Buffer {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: shape.output_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };

            let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sound_concretize_params"),
                size: size_of::<SoundConcretizeParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));
            super::intermediate_sweep::note_host_to_device(size_of::<SoundConcretizeParams>());

            // Pack lower|upper into single buffers to stay within the 8-storage-
            // buffer compute-stage limit (Metal default).
            let mut bias_packed = Vec::with_capacity(2 * num_specs);
            bias_packed.extend_from_slice(bias_lower);
            bias_packed.extend_from_slice(bias_upper);
            let mut err_packed = Vec::with_capacity(2 * coeff);
            err_packed.extend_from_slice(a_lower_err);
            err_packed.extend_from_slice(a_upper_err);

            let b_al = storage(a_lower, "sc_a_lower");
            let b_au = storage(a_upper, "sc_a_upper");
            let b_xl = storage(input_lower, "sc_input_lower");
            let b_xu = storage(input_upper, "sc_input_upper");
            let b_bias = storage(&bias_packed, "sc_bias");
            let b_ol = out_buf("sc_out_lower");
            let b_ou = out_buf("sc_out_upper");
            let b_err = storage(&err_packed, "sc_a_err");

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sound_concretize_bind_group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_al.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_au.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: b_xl.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: b_xu.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: b_bias.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: b_ol.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: b_ou.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: b_err.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sound_concretize_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("sound_concretize_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(shape.num_specs_u32, 1, 1);
                super::intermediate_sweep::note_dispatches(1);
            }
            // One staging buffer per output, copied after the pass.
            let mut stage = |src: &wgpu::Buffer, label: &str| -> wgpu::Buffer {
                let s = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: shape.output_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(src, 0, &s, 0, shape.output_bytes);
                s
            };
            let st_l = stage(&b_ol, "sc_stage_lower");
            let st_u = stage(&b_ou, "sc_stage_upper");
            self.queue.submit(Some(encoder.finish()));
            super::intermediate_sweep::note_submits(1);

            let lower = Self::read_buffer(&self.device, &st_l, num_specs)?;
            let upper = Self::read_buffer(&self.device, &st_u, num_specs)?;
            Ok((lower, upper))
        })
    }
}

#[cfg(test)]
mod host_soundness_tests {
    use super::{
        add_up_nonnegative_f64, affine_radius_step_up, concretize_max_workgroups,
        consult_spec_row_taint, sound_concretize_shape,
    };

    #[test]
    fn outward_add_does_not_lose_terms_that_cross_fallback_threshold() {
        let fallback = f64::from(crate::FALLBACK_BOUND);
        let start = fallback.next_down();
        let tiny = 2.0f64.powi(-21);

        // At this magnitude one f64 ULP is 2^-19. Each 2^-21 contribution is
        // positive but is lost by naïve round-to-nearest accumulation. Five such
        // terms exceed the exact one-ULP deficit, so a conservative guard must
        // reject even though the naïve accumulator remains below the sentinel.
        assert_eq!(fallback - start, 2.0f64.powi(-19));
        assert!(5.0 * tiny > fallback - start);

        let mut naive = start;
        let mut outward = Some(start);
        for _ in 0..5 {
            naive += tiny;
            outward = outward.and_then(|radius| add_up_nonnegative_f64(radius, tiny));
        }

        assert!(naive < fallback, "regression setup must fool a naïve sum");
        assert!(
            outward.expect("finite outward sum") >= fallback,
            "outward accumulation must not authorize an unsafe finite fallback"
        );
    }

    #[test]
    fn affine_radius_step_is_outward_and_zero_terms_are_exact() {
        let radius = 3.0f64;
        assert_eq!(
            affine_radius_step_up(radius, 0.0, 0.0, 17.0),
            Some(radius),
            "zero coefficients must not accumulate artificial ULPs"
        );

        let coefficient = 1.25f32;
        let coefficient_error = 0.5f32;
        let xmax = 3.0f64;
        let exact = radius + (f64::from(coefficient).abs() + f64::from(coefficient_error)) * xmax;
        let outward = affine_radius_step_up(radius, coefficient, coefficient_error, xmax).unwrap();
        assert!(outward >= exact);

        // The smallest binary32 subnormal must remain nonzero before a large
        // input magnitude amplifies it. A hardware f32→f64 conversion under
        // DAZ could otherwise erase this entire radius contribution.
        let subnormal = f32::from_bits(1);
        let huge_x = 2.0f64.powi(100);
        let exact_subnormal_term = ny_core::f32_to_f64_exact(subnormal) * huge_x;
        let outward_subnormal = affine_radius_step_up(0.0, subnormal, 0.0, huge_x).unwrap();
        assert!(exact_subnormal_term > 0.0);
        assert!(outward_subnormal >= exact_subnormal_term);

        // Every resident value-kernel FALLBACK_BOUND sentinel reaches this same
        // host guard before the verdict-facing concretize dispatch. A sentinel
        // coefficient (or a propagated 1e30 error taint) must therefore make the
        // proven radius at least FALLBACK_BOUND and be refused.
        let fallback = crate::FALLBACK_BOUND;
        assert!(affine_radius_step_up(0.0, fallback, 0.0, 1.0).unwrap() >= f64::from(fallback));
        assert!(affine_radius_step_up(0.0, 0.0, 1.0e30, 1.0).unwrap() >= f64::from(fallback));
    }

    /// #u4 C1: production taint plumbing and the reviewed arming change have
    /// landed. Pin the gate open so silently dropping back to magnitude-only
    /// guards is a test failure. The separately reviewed raw-device authority
    /// gate in sound_authority.rs is now open but remains independently
    /// qualified by its explicit request and full live-probe conjunction.
    #[test]
    fn taint_word_gate_is_armed_per_the_2026_08_11_review() {
        const {
            assert!(
                super::super::sentinel_taint_selfcheck::PRODUCTION_GUARDS_CONSULT_TAINT_WORD,
                "PRODUCTION_GUARDS_CONSULT_TAINT_WORD was ARMED by the 2026-08-11 UTC \
                 source review (production plumbing + segment words + on-device \
                 transports all landed and measured; see the const's doc). \
                 Closing it again is itself a source-review event."
            );
        }
    }

    /// #u4 C1: what the armed preflight enforces. Every arm is fail-closed:
    /// absent words refuse, misaligned lengths refuse, and any nonzero word
    /// refuses while naming the row; only exactly `num_specs` zero words pass.
    #[test]
    fn taint_consult_is_fail_closed_on_every_arm() {
        let absent = consult_spec_row_taint(None, 3).expect_err("absent words must refuse");
        assert!(
            format!("{absent}").contains("taint words absent — refusing verdict use"),
            "absent-words refusal must say so: {absent}"
        );

        let short = consult_spec_row_taint(Some(&[0, 0]), 3)
            .expect_err("a wrong-length word slice must refuse");
        assert!(
            format!("{short}").contains("refusing verdict use"),
            "length-mismatch refusal must be fail-closed: {short}"
        );

        let tainted = consult_spec_row_taint(Some(&[0, 4, 0]), 3)
            .expect_err("a nonzero word must refuse the whole batch");
        let msg = format!("{tainted}");
        assert!(
            msg.contains("spec row 1") && msg.contains("refusing verdict use"),
            "taint refusal must name the row and be fail-closed: {msg}"
        );

        consult_spec_row_taint(Some(&[0, 0, 0]), 3).expect("all-zero words must pass");
    }

    #[test]
    fn shape_preflight_rejects_wgsl_index_wrap_and_oversized_bindings() {
        assert!(concretize_max_workgroups(0).is_err());
        assert_eq!(concretize_max_workgroups(1).unwrap(), 1);

        let valid = sound_concretize_shape(2, 1, 3, u64::MAX, u64::MAX)
            .expect("two domains with three inputs fit");
        assert_eq!(valid.coeff, 6);
        assert_eq!(valid.box_len, 6);

        assert!(
            sound_concretize_shape(2, 1, 3, 47, 47).is_err(),
            "the packed 12-f32 error binding needs 48 bytes"
        );

        if usize::BITS > 32 {
            let wraps_u32 = (u32::MAX as usize / 2) + 1;
            assert!(
                sound_concretize_shape(wraps_u32, wraps_u32, 1, u64::MAX, u64::MAX).is_err(),
                "packed WGSL indexing must not wrap even when each factor fits u32"
            );
            assert!(
                sound_concretize_shape(65_536, 1, 65_536, u64::MAX, u64::MAX).is_err(),
                "num_specs*input_dim must fit the WGSL index type"
            );
        }
    }
}

#[cfg(all(test, feature = "gpu-tests"))]
mod tests {
    use crate::wgpu_device::test_support::{gpu_test_serial_guard, require_device};

    /// #wg-limit-guard (SOUNDNESS, fail-closed): a spec-row count over
    /// `max_compute_workgroups_per_dimension` (the shader dispatches ONE workgroup per
    /// row) must return a clean `Err` — never a silently over-tight (unsound) bound or a
    /// crash from an over-limit dispatch. Proves MY guard fires (descriptive message)
    /// BEFORE any GPU work, and that a batch exactly AT the limit is accepted.
    #[test]
    fn concretize_over_workgroup_limit_fails_closed_not_corrupt() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        let max_wg = device
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .max(1) as usize;
        // input_dim = 1 keeps buffers tiny (num_specs f32 each).
        let input_dim = 1usize;
        let over = max_wg + 1;
        let mk = |n: usize| {
            (
                vec![0.5f32; n * input_dim], // a_lower
                vec![0.5f32; n * input_dim], // a_upper
                vec![0.0f32; n * input_dim], // a_lower_err
                vec![0.0f32; n * input_dim], // a_upper_err
                vec![0.0f32; n],             // input_lower (n_domains=n, input_dim=1)
                vec![1.0f32; n],             // input_upper
                vec![0.0f32; n],             // bias_lower
                vec![0.0f32; n],             // bias_upper
            )
        };
        let (al, au, ale, aue, xl, xu, bl, bu) = mk(over);
        let res = device.concretize_sound_gpu_batched(
            over,
            1,
            input_dim,
            &al,
            &au,
            &ale,
            &aue,
            &xl,
            &xu,
            &bl,
            &bu,
            Some(&vec![0u32; over]),
        );
        let err = res.expect_err("over-limit concretize must fail closed, not return a bound");
        let msg = format!("{err}");
        assert!(
            msg.contains("max_compute_workgroups_per_dimension"),
            "expected the fail-closed workgroup-limit guard message, got: {msg}"
        );

        // A batch exactly at the limit passes the guard (and returns finite bounds).
        let (al, au, ale, aue, xl, xu, bl, bu) = mk(max_wg);
        let (lo, hi) = device
            .concretize_sound_gpu_batched(
                max_wg,
                1,
                input_dim,
                &al,
                &au,
                &ale,
                &aue,
                &xl,
                &xu,
                &bl,
                &bu,
                Some(&vec![0u32; max_wg]),
            )
            .expect("at-limit concretize must succeed");
        assert_eq!(lo.len(), max_wg);
        assert_eq!(hi.len(), max_wg);
        assert!(lo.iter().chain(hi.iter()).all(|v| v.is_finite()));
    }

    /// #u4 C1: the armed device preflight refuses absent or nonzero words and
    /// accepts clean words deterministically. The host-side test above pins the
    /// remaining wrong-length arm without requiring a GPU allocation.
    #[test]
    fn taint_consult_is_armed_and_fail_closed_on_device() {
        let _g = gpu_test_serial_guard();
        let device = require_device();
        const {
            assert!(
                super::super::sentinel_taint_selfcheck::PRODUCTION_GUARDS_CONSULT_TAINT_WORD,
                "this pin tests the ARMED consult (2026-08-11 UTC review)"
            );
        }

        let num_specs = 2usize;
        let input_dim = 3usize;
        let a_l = [0.5f32, -1.25, 2.0, 0.75, -0.5, 1.5];
        let a_u = [0.75f32, -1.0, 2.25, 1.0, -0.25, 1.75];
        let e_l = [0.01f32; 6];
        let e_u = [0.02f32; 6];
        let x_l = [-1.0f32, 0.0, 0.25];
        let x_u = [1.0f32, 0.5, 0.75];
        let b_l = [0.1f32, -0.2];
        let b_u = [0.2f32, -0.1];
        let run = |taint: Option<&super::SpecRowTaint>| {
            device.concretize_sound_gpu_batched(
                num_specs, num_specs, input_dim, &a_l, &a_u, &e_l, &e_u, &x_l, &x_u, &b_l, &b_u,
                taint,
            )
        };

        // Armed: ABSENT words refuse fail-closed (the un-worded chain can
        // never reach a verdict), and a tainted row refuses naming the row.
        let msg = run(None)
            .expect_err("armed consult must refuse taint=None")
            .to_string();
        assert!(
            msg.contains("taint words absent"),
            "refusal must be the typed fail-closed message, got: {msg}"
        );
        let msg = run(Some(&[0u32, 0xDEAD_BEEF]))
            .expect_err("armed consult must refuse a tainted row")
            .to_string();
        assert!(
            msg.contains("row 1"),
            "refusal must name the row, got: {msg}"
        );

        // Clean words pass, deterministically.
        let (l1, u1) = run(Some(&[0u32, 0])).expect("clean words must pass");
        let (l2, u2) = run(Some(&[0u32, 0])).expect("clean words must pass (repeat)");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        assert_eq!(bits(&l1), bits(&l2));
        assert_eq!(bits(&u1), bits(&u2));
    }

    /// Exact (f64 corner) lower/upper of the network output. CROWN's `lower_a` is
    /// the *lower linear bound* (output ≥ lower_a·x + lower_b), so the sound lower
    /// bound is the min over `lower_a' ∈ [lower_a − err_l, lower_a + err_l]` and
    /// `x ∈ box` — and symmetrically the upper uses ONLY `upper_a ± err_u`. (The
    /// two sides are independent; they must NOT be mixed into one interval.)
    /// `f32·f32` is exact in `f64`, so this is a faithful oracle.
    #[allow(clippy::too_many_arguments)]
    fn oracle(
        num_specs: usize,
        input_dim: usize,
        a_lower: &[f32],
        a_upper: &[f32],
        a_lower_err: &[f32],
        a_upper_err: &[f32],
        x_l: &[f32],
        x_u: &[f32],
        b_l: &[f32],
        b_u: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let mut lo = vec![0.0f64; num_specs];
        let mut hi = vec![0.0f64; num_specs];
        for s in 0..num_specs {
            let mut l = f64::from(b_l[s]);
            let mut h = f64::from(b_u[s]);
            for j in 0..input_dim {
                let idx = s * input_dim + j;
                let xl = f64::from(x_l[j]);
                let xu = f64::from(x_u[j]);
                // Lower bound: lower_a ± err_l only.
                let lmin = f64::from(a_lower[idx]) - f64::from(a_lower_err[idx]);
                let lmax = f64::from(a_lower[idx]) + f64::from(a_lower_err[idx]);
                let lc = [lmin * xl, lmin * xu, lmax * xl, lmax * xu];
                l += lc.iter().copied().fold(f64::INFINITY, f64::min);
                // Upper bound: upper_a ± err_u only.
                let umin = f64::from(a_upper[idx]) - f64::from(a_upper_err[idx]);
                let umax = f64::from(a_upper[idx]) + f64::from(a_upper_err[idx]);
                let uc = [umin * xl, umin * xu, umax * xl, umax * xu];
                h += uc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            }
            lo[s] = l;
            hi[s] = h;
        }
        (lo, hi)
    }

    #[test]
    fn sound_concretize_encloses_true_range_on_gpu() {
        let _g = gpu_test_serial_guard();
        let device = require_device();

        let mut state: u64 = 0xC0DE_F00D;
        let mut rng = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };

        for &(num_specs, input_dim) in &[(3usize, 5usize), (8, 257), (2, 1024)] {
            let coeff = num_specs * input_dim;
            let a_mid: Vec<f32> = (0..coeff).map(|_| rng() * 2.0).collect();
            // Relaxation gap (a_lower <= a_upper) plus a small rounding error.
            let mut a_lower = vec![0.0f32; coeff];
            let mut a_upper = vec![0.0f32; coeff];
            let mut a_lower_err = vec![0.0f32; coeff];
            let mut a_upper_err = vec![0.0f32; coeff];
            for i in 0..coeff {
                let gap = (rng() * 0.1).abs();
                a_lower[i] = a_mid[i] - gap;
                a_upper[i] = a_mid[i] + gap;
                a_lower_err[i] = (rng() * 0.05).abs();
                a_upper_err[i] = (rng() * 0.05).abs();
            }
            let x_l: Vec<f32> = (0..input_dim).map(|_| rng()).collect();
            let x_u: Vec<f32> = (0..input_dim)
                .map(|i| x_l[i] + (rng() * 0.5).abs())
                .collect();
            let b_l: Vec<f32> = (0..num_specs).map(|_| rng()).collect();
            let b_u: Vec<f32> = (0..num_specs)
                .map(|i| b_l[i] + (rng() * 0.5).abs())
                .collect();

            let (lo, hi) = device
                .concretize_sound_gpu(
                    num_specs,
                    input_dim,
                    &a_lower,
                    &a_upper,
                    &a_lower_err,
                    &a_upper_err,
                    &x_l,
                    &x_u,
                    &b_l,
                    &b_u,
                    Some(&vec![0u32; num_specs]),
                )
                .expect("sound concretize");
            let (o_lo, o_hi) = oracle(
                num_specs,
                input_dim,
                &a_lower,
                &a_upper,
                &a_lower_err,
                &a_upper_err,
                &x_l,
                &x_u,
                &b_l,
                &b_u,
            );
            for s in 0..num_specs {
                assert!(lo[s].is_finite() && hi[s].is_finite() && lo[s] <= hi[s]);
                assert!(
                    f64::from(lo[s]) <= o_lo[s] + 1e-4,
                    "({num_specs}x{input_dim}) spec {s}: UNSOUND lower {} > true min {}",
                    lo[s],
                    o_lo[s]
                );
                assert!(
                    f64::from(hi[s]) >= o_hi[s] - 1e-4,
                    "({num_specs}x{input_dim}) spec {s}: UNSOUND upper {} < true max {}",
                    hi[s],
                    o_hi[s]
                );
            }
        }
    }

    /// By-construction check independent of the adapter's measured subnormal mode:
    /// the §0 amplified operand-flush term must widen each bound by at least
    /// `xmax·FLT_MIN` even for a subnormal coefficient the GPU computes exactly here.
    /// Under Metal FTZ that subnormal `a` flushes to 0 and the true product `a·x` (a
    /// NORMAL f32) can be silently dropped on a flushing path; the amplified floor
    /// certifies it back.
    /// The OLD weight-independent `8·n·η` floor emitted a widening ~90 binary orders
    /// of magnitude too tight here — a false-VERIFIED break. This is the concretize
    /// twin of `sound_gpu_ibp_flush_radius_amplified_by_weight_t1_1`.
    #[test]
    fn sound_concretize_amplified_flush_covers_subnormal_times_huge() {
        let _g = gpu_test_serial_guard();
        let device = require_device();

        // One spec, one input. a·x = 2^-130 · 2^100 = 2^-30 (a NORMAL f32).
        let a = 2.0f32.powi(-130); // subnormal: behavior is adapter/loading-path specific
        let x = 2.0f32.powi(100); // huge magnitude
        let (lo, hi) = device
            .concretize_sound_gpu(
                1,
                1,
                &[a],
                &[a],
                &[0.0],
                &[0.0],
                &[x],
                &[x],
                &[0.0],
                &[0.0],
                Some(&[0u32; 1]),
            )
            .expect("sound concretize");
        let (lo, hi) = (f64::from(lo[0]), f64::from(hi[0]));

        // Amplified-flush budget: xmax·FLT_MIN = 2^100·2^-126 = 2^-26.
        let flt_min = f64::from(f32::from_bits(0x0080_0000)); // 2^-126
        let amplified = f64::from(x) * flt_min; // 2^-26
        let y = f64::from(a) * f64::from(x); // 2^-30, exact

        assert!(
            lo <= y && y <= hi,
            "interval [{lo:e}, {hi:e}] must enclose true a·x = {y:e}"
        );
        assert!(
            hi - y >= 0.5 * amplified,
            "upper widening {:e} must cover amplified flush {amplified:e} (Metal FTZ soundness)",
            hi - y
        );
        assert!(
            y - lo >= 0.5 * amplified,
            "lower widening {:e} must cover amplified flush {amplified:e} (Metal FTZ soundness)",
            y - lo
        );
    }

    /// Regression for final assembly rounding: the lower endpoint is small while
    /// the propagated coefficient-error penalty is about 1024. Charging only the
    /// endpoint and bias misses the rounding of the dominant final subtraction
    /// and can publish a lower bound above the exact affine minimum.
    #[test]
    fn sound_concretize_final_assembly_includes_dominant_penalty() {
        let _g = gpu_test_serial_guard();
        let device = require_device();

        let a = -0.015625_f32;
        let bias = 0.000_122_070_3_f32;
        let err = 1_023.999_9_f32;
        let (lo, hi) = device
            .concretize_sound_gpu(
                1,
                1,
                &[a],
                &[a],
                &[err],
                &[err],
                &[1.0],
                &[1.0],
                &[bias],
                &[bias],
                Some(&[0u32; 1]),
            )
            .expect("sound concretize");

        let exact_lower = f64::from(bias) + f64::from(a) - f64::from(err);
        let exact_upper = f64::from(bias) + f64::from(a) + f64::from(err);
        assert!(
            f64::from(lo[0]) <= exact_lower,
            "lower {} exceeds exact minimum {exact_lower}",
            lo[0]
        );
        assert!(
            f64::from(hi[0]) >= exact_upper,
            "upper {} is below exact maximum {exact_upper}",
            hi[0]
        );
    }

    /// RAII guard for the forced EFT-primitive failure: released on drop, so a
    /// failing assertion cannot leak a forced refusal into later tests in this
    /// process (the serial guard orders them, it does not isolate them).
    struct ForcedEftFail;
    impl ForcedEftFail {
        fn arm() -> Self {
            super::super::eft_selfcheck::set_force_eft_selfcheck_fail(true);
            ForcedEftFail
        }
    }
    impl Drop for ForcedEftFail {
        fn drop(&mut self) {
            super::super::eft_selfcheck::set_force_eft_selfcheck_fail(false);
        }
    }

    /// #u6 (sound_authority.rs obligation U6; `docs/METAL_EFT_VIABLE_2026-08-04.md`
    /// §U6): eft-mode concretize is NOT value-neutral — BY DESIGN. Legacy folds
    /// `local + a_pos*x + a_neg*x'` in one expression and charges the a-priori
    /// `γ_n·|a|` dot-rounding term; EFT mode runs the barrier-fma sequence as
    /// two separate accumulating adds and charges the MEASURED residual sum
    /// (·`eft_r_slack`). Different op sequence ⇒ different bits, and the
    /// measured charge is usually far tighter than γ_n. What soundness requires
    /// is therefore NOT equality but:
    ///
    /// 1. BOTH modes independently enclose the exact f64 corner oracle
    ///    (enclosure is the assert; equality is NEVER asserted);
    /// 2. where the outputs differ is REPORTED (printed), and on an
    ///    EFT-authorized adapter they must actually differ somewhere (the
    ///    "not value-neutral" half of the claim — with `input_dim = 1024` the
    ///    γ_n charge is ~6e-5·Σ|a|·xmax while the measured residuals are
    ///    u-scale, orders apart in f32);
    /// 3. the FAIL-CLOSED IDENTITY: with the EFT lane REFUSED (forced
    ///    primitive failure — checked per call by `eft_primitives_cached`, or
    ///    a gate that never authorized), `NY_EFT_ERR=1` output is
    ///    BIT-IDENTICAL to legacy; and the armed C1 taint consult refuses a
    ///    tainted row in EFT mode exactly as it does in legacy mode.
    #[test]
    fn u6_concretize_eft_vs_legacy_enclose_and_fail_closed_identity() {
        use ny_test_utils::env::ScopedEnvVar;
        let _g = gpu_test_serial_guard();
        let device = require_device();
        // Hardware fact, not an assertion: decides which half of claim 2/3 is
        // checkable here (the eft_err_channel_ab precedent).
        let eft_authorized = device.verify_eft_primitives();
        println!(
            "[u6] adapter={} backend={:?} verify_eft_primitives={eft_authorized}",
            device.adapter_info.name, device.adapter_info.backend,
        );

        fn lcg(state: &mut u64) -> u64 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *state
        }
        fn frac(state: &mut u64) -> f32 {
            ((lcg(state) >> 40) as f32) / (1u64 << 24) as f32
        }
        // `± 2^e·(1+frac)`, e ∈ [lo, hi] — powi is exact at the 2^-126 edges.
        fn banded(state: &mut u64, lo: i32, hi: i32) -> f32 {
            let e = lo + (lcg(state) % (hi - lo + 1) as u64) as i32;
            let s = if lcg(state) & 1 == 0 { 1.0f32 } else { -1.0 };
            s * 2.0f32.powi(e) * (1.0 + frac(state))
        }
        let mut state: u64 = 0x0066_C0DE_2026_0810;

        let mut n_diff = 0usize;
        let mut n_printed = 0usize;
        let mut n_total = 0usize;
        for &(num_specs, input_dim) in &[(3usize, 5usize), (8, 257), (2, 1024), (4, 511)] {
            let coeff = num_specs * input_dim;
            let mut a_lower = vec![0.0f32; coeff];
            let mut a_upper = vec![0.0f32; coeff];
            let mut a_lower_err = vec![0.0f32; coeff];
            let mut a_upper_err = vec![0.0f32; coeff];
            for i in 0..coeff {
                // Mixed-magnitude bands (2^-30..2^6 keeps the affine-radius
                // preflight below FALLBACK_BOUND at dim 1024), ~1% exact
                // zeros, and a scattered 2^-126 subnormal-edge lane.
                let mid = if frac(&mut state) < 0.01 {
                    0.0
                } else if i % 89 == 0 {
                    banded(&mut state, -127, -120)
                } else {
                    banded(&mut state, -30, 6)
                };
                let gap = banded(&mut state, -30, -4).abs() * mid.abs().max(2.0f32.powi(-20));
                a_lower[i] = mid - gap;
                a_upper[i] = mid + gap;
                a_lower_err[i] = if i % 3 == 0 {
                    0.0
                } else {
                    banded(&mut state, -30, -8).abs() * mid.abs().max(1.0)
                };
                a_upper_err[i] = if i % 5 == 0 {
                    0.0
                } else {
                    banded(&mut state, -30, -8).abs() * mid.abs().max(1.0)
                };
            }
            let mut x_l = vec![0.0f32; input_dim];
            let mut x_u = vec![0.0f32; input_dim];
            for j in 0..input_dim {
                let c = banded(&mut state, -10, 3);
                let w = banded(&mut state, -12, 1).abs();
                x_l[j] = c - w;
                x_u[j] = c + w;
            }
            let b_l: Vec<f32> = (0..num_specs).map(|_| banded(&mut state, -8, 2)).collect();
            let b_u: Vec<f32> = b_l
                .iter()
                .map(|&b| b + banded(&mut state, -10, 0).abs())
                .collect();
            let clean = vec![0u32; num_specs];

            let run = |taint: Option<&super::SpecRowTaint>| {
                device.concretize_sound_gpu(
                    num_specs,
                    input_dim,
                    &a_lower,
                    &a_upper,
                    &a_lower_err,
                    &a_upper_err,
                    &x_l,
                    &x_u,
                    &b_l,
                    &b_u,
                    taint,
                )
            };

            // Legacy (explicit UNSET, so an outer NY_EFT_ERR=1 cannot collapse
            // the A/B) then EFT (env ON; the capability half of the gate is the
            // adapter's — that is the point of the branch below).
            let (lo_leg, hi_leg) = {
                let _e = ScopedEnvVar::unset("NY_EFT_ERR");
                run(Some(&clean)).expect("legacy concretize")
            };
            let (lo_eft, hi_eft) = {
                let _e = ScopedEnvVar::set("NY_EFT_ERR", "1");
                run(Some(&clean)).expect("eft concretize")
            };

            // Claim 1 — BOTH modes enclose the exact corner oracle. The
            // tolerance covers only the oracle's own f64 summation noise
            // (~dim·2^-53·Σ|terms| ≈ 1.1e-13·scale at dim=1024), with ~1000x
            // headroom at 1e-10 — three orders BELOW the f32 dot's own
            // rounding (~6e-7·scale), so a dropped or halved eft residual
            // charge cannot hide under it (review defect 2: the original
            // 1e-6 exceeded the entire f32 rounding error and neutered the
            // assert on exactly the claim's risk surface).
            let (o_lo, o_hi) = oracle(
                num_specs,
                input_dim,
                &a_lower,
                &a_upper,
                &a_lower_err,
                &a_upper_err,
                &x_l,
                &x_u,
                &b_l,
                &b_u,
            );
            for (mode, lo, hi) in [("legacy", &lo_leg, &hi_leg), ("eft", &lo_eft, &hi_eft)] {
                for s in 0..num_specs {
                    let tol = 1e-10 * (1.0 + o_lo[s].abs().max(o_hi[s].abs()));
                    assert!(
                        lo[s].is_finite() && hi[s].is_finite() && lo[s] <= hi[s],
                        "({num_specs}x{input_dim}) {mode} spec {s}: malformed [{}, {}]",
                        lo[s],
                        hi[s]
                    );
                    assert!(
                        f64::from(lo[s]) <= o_lo[s] + tol,
                        "({num_specs}x{input_dim}) {mode} spec {s}: UNSOUND lower {} > \
                         true min {}",
                        lo[s],
                        o_lo[s]
                    );
                    assert!(
                        f64::from(hi[s]) >= o_hi[s] - tol,
                        "({num_specs}x{input_dim}) {mode} spec {s}: UNSOUND upper {} < \
                         true max {}",
                        hi[s],
                        o_hi[s]
                    );
                }
            }

            // Claim 2 — report where the modes differ (never assert equality).
            for s in 0..num_specs {
                n_total += 1;
                if lo_leg[s].to_bits() != lo_eft[s].to_bits()
                    || hi_leg[s].to_bits() != hi_eft[s].to_bits()
                {
                    n_diff += 1;
                    if n_printed < 8 {
                        println!(
                            "[u6] ({num_specs}x{input_dim}) spec {s} differs: \
                             legacy=[{:e}, {:e}] eft=[{:e}, {:e}]",
                            lo_leg[s], hi_leg[s], lo_eft[s], hi_eft[s],
                        );
                        n_printed += 1;
                    }
                }
            }

            // Claim 3a — the fail-closed identity, constructible on EVERY
            // adapter: force the primitive gate to refuse (consulted per call)
            // and the NY_EFT_ERR=1 output must be BIT-IDENTICAL to legacy.
            {
                let _forced = ForcedEftFail::arm();
                let _e = ScopedEnvVar::set("NY_EFT_ERR", "1");
                let (lo_f, hi_f) = run(Some(&clean)).expect("forced-refusal concretize");
                for s in 0..num_specs {
                    assert!(
                        lo_f[s].to_bits() == lo_leg[s].to_bits()
                            && hi_f[s].to_bits() == hi_leg[s].to_bits(),
                        "({num_specs}x{input_dim}) spec {s}: EFT lane REFUSED but \
                         NY_EFT_ERR=1 still changed the output \
                         (legacy=[{:e}, {:e}] refused=[{:e}, {:e}]) — part of the \
                         compensated concretize is reachable behind a closed gate",
                        lo_leg[s],
                        hi_leg[s],
                        lo_f[s],
                        hi_f[s],
                    );
                }
            }
        }

        // Claim 2, resolved by the adapter: authorized ⇒ the modes MUST have
        // differed somewhere (γ_n vs measured residuals are orders apart at
        // dim 1024); refused ⇒ the gate keeps EFT dark ⇒ bit-identical
        // everywhere is the fail-closed identity itself.
        println!("[u6] authorized={eft_authorized} rows={n_total} rows_differing={n_diff}");
        if eft_authorized {
            assert!(
                n_diff > 0,
                "EFT authorized but concretize was value-neutral across {n_total} rows — \
                 the U6 claim (min-combine-style tightening changes the bits) did not \
                 manifest; either the eft_mode uniform is not reaching the shader or \
                 the measured-residual charge silently equals γ_n"
            );
        } else {
            assert_eq!(
                n_diff, 0,
                "EFT gate REFUSED on this adapter but NY_EFT_ERR=1 changed {n_diff} \
                 of {n_total} rows — the refusal is not byte-identical"
            );
        }

        // Claim 3b — the armed C1 consult is mode-independent: a tainted row
        // refuses under NY_EFT_ERR=1 exactly as it does in legacy mode.
        {
            let _e = ScopedEnvVar::set("NY_EFT_ERR", "1");
            let msg = device
                .concretize_sound_gpu(
                    2,
                    1,
                    &[0.5f32, 0.5],
                    &[0.5f32, 0.5],
                    &[0.0f32, 0.0],
                    &[0.0f32, 0.0],
                    &[0.0f32],
                    &[1.0f32],
                    &[0.0f32, 0.0],
                    &[0.0f32, 0.0],
                    Some(&[0u32, 0xDEAD_BEEF]),
                )
                .expect_err("armed consult must refuse a tainted row in EFT mode too")
                .to_string();
            assert!(
                msg.contains("row 1"),
                "EFT-mode taint refusal must name the row, got: {msg}"
            );
        }
    }
}
