// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared host-side sound-arithmetic constants for the GPU verdict paths.
//!
//! `gamma_k_f32`, `combine_slack_f32`, and `up_f32` were originally private in
//! `ops/crown_backward_sound_resident.rs`. The sound GPU IBP forward
//! (`docs/SOUND_GPU_IBP_PLAN.md` §2.1) needs the same host-side error sizing, so
//! they are hoisted here to ONE `pub(crate)` home — CROWN backward, the sound
//! concretize, and the sound IBP forward all share this single copy instead of
//! carrying divergent duplicates. All three are HOST-side f64 helpers rounded
//! OUTWARD to f32 uniforms; no f64 ever enters a WGSL body.

use ny_core::{f32_to_f64_exact, f64_to_f32_up, NyError, Result};

/// f32 unit roundoff `u = 2^-24`.
const U: f64 = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24

/// γ_k = k·u/(1−k·u) (the f32 dot-product backward-error factor for a length-k
/// reduction).
///
/// Higham's finite bound requires `k·u < 1`. There is no safe finite substitute
/// outside that regime, so this fails closed before a caller can build a GPU
/// uniform or dispatch a verdict-sensitive shader.
pub(crate) fn gamma_k_f32(k: usize) -> Result<f32> {
    let ku = (k as f64) * U;
    if !ku.is_finite() || ku >= 1.0 {
        return Err(NyError::UnsupportedOp(format!(
            "sound f32 reduction length {k} has k*u >= 1, so Higham gamma_k \
             has no finite bound"
        )));
    }
    // Narrow OUTWARD (#gamma-outward). A plain `as f32` rounds to NEAREST, so the
    // certified factor could land up to 0.5 f32 ULP BELOW exact γ — the unsound
    // direction, since γ multiplies an error budget. Every other γ constructor in
    // the tree already rounds outward (`ny_core::dd::gamma_n_at` steps `next_up_f64`
    // twice; `margin_row/rounding.rs` uses `next_up`); this was the one place that
    // did not.
    Ok(up_f32(ku / (1.0 - ku)))
}

/// SOUNDNESS slack for the AW-error combine on the f32 error GEMMs.
///
/// The combine reads `s_prod = fl(|A|@|W|)` and `prop = fl(err@|W|)`, both f32
/// dot products over the length-`k` contraction. By Higham each is ≥
/// `(1−γ_k)·(exact sum)`, so the exact sum ≤ `f32_result/(1−γ_k)`. To turn the
/// on-device `(γ_k·s_prod + prop)` (which uses the UNDER-reported f32 products)
/// into an OUTWARD bound on `γ_k·S_exact + prop_exact`, scale by
/// `slack ≥ 1/(1−γ_k)`. We add four ULPs of headroom `(1+u)^4` for the combine's
/// own f32 ops (γ·s multiply, the add, the slack multiply, the +additive) and
/// round the f32 cast UP. `γ_k` here is `gamma_k_f32(k)` (the SAME factor the
/// reductions incur). For the small `k` where the old fixed 1.000001 was already
/// adequate this evaluates to ~1.0000xx as well, but it now SCALES with k so wide
/// contractions are covered.
pub(crate) fn combine_slack_f32(k: usize) -> Result<f32> {
    let g = f32_to_f64_exact(gamma_k_f32(k)?);
    // Recovery requires γ_k < 1 (equivalently k·u < 1/2). Once γ_k reaches one,
    // the rounded non-negative reduction has no positive lower factor to invert.
    // Infinity is the only conservative response; a finite fallback would
    // under-state the exact sum for some reduction orders.
    if !g.is_finite() || g >= 1.0 {
        return Err(NyError::UnsupportedOp(format!(
            "sound f32 reduction length {k} has gamma_k >= 1, so its \
             non-negative reduction has no finite recovery slack"
        )));
    }
    let inv = 1.0 / (1.0 - g);
    let headroom = (1.0 + U).powi(4); // 4 combine f32 ops, each ≤ (1+u) growth
    Ok(up_f32(inv * headroom))
}

/// SOUNDNESS slack for the EFT residual channel (#eft-err).
///
/// The EFT twin's per-element residual sum `R = Σ|ep| + Σ|es|` is accumulated
/// in f32 over `2k` non-negative terms (plain adds), so by Higham the f32
/// result ≥ `(1−γ_{2k})·R_exact` — recovering the outward bound needs
/// `1/(1−γ_{2k})`. Headroom `(1+u)^6` covers the min-combine's own f32 ops:
/// the `|V−value|` subtraction/abs, the `R+d` add, the `·r_slack` multiply,
/// the `prop·slack` product, the cross add, and the `+flush`. Rounded UP.
/// Extra f32 adds the RESIDUAL lane absorbs in the workgroup tree reduction,
/// beyond the `2k` dot terms (#eft-err U3, measured 2026-08-06).
///
/// `CROWN_CONCRETIZE_SOUND_SHADER`'s reduction does, per level:
///
/// ```wgsl
/// sh_rl[local_id] = sh_rl[local_id] + sh_rl[local_id + stride] + rl_add;
/// ```
///
/// — TWO adds, over `stride = 128, 64, ... 1`, i.e. 8 levels. None of those 16
/// adds were in the `2k + 2` count, so the recovery factor was computed for a
/// shorter reduction than the kernel actually performs.
///
/// `METAL_EFT_VIABLE_2026-08-04.md` §5 raised this as U3 and offered two
/// settlements: an exact count, or bumping the headroom. This is the exact
/// count. Charging it unconditionally slightly OVER-charges the flat GEMM twin,
/// which has no tree — over-charging is the sound direction, and one constant is
/// worth more than a second code path that could be wired to the wrong kernel.
const TREE_REDUCTION_RESIDUAL_ADDS: usize = 16; // 2 adds x 8 levels (256-lane workgroup)

/// #rung3-fma-floor: base additive covering fma SUBNORMAL-RESULT flushes in the
/// EFT residual lanes under DenormPreserve (authority-ladder rung 3,
/// GB10-measured 2026-08-10 America/Los_Angeles: injected `DenormPreserve`
/// preserves core add/mul subnormals but fma still flushes subnormal RESULTS —
/// hardware, not toolchain).
///
/// Every fma-barrier TwoSum in the shipped kernels computes 4 fma results
/// (`bb`, `sb`, `da`, `db`) plus the residual add; each flushable output loses
/// at most `2^-126` when its true value is subnormal. The
/// `TWO_PROD_EXACT_FLOOR_F32` guard covers small product values; a subnormal
/// TwoProduct residual above that guard is still possible (and is pinned by
/// `flush_charge_oracle`), so the conservative aggregate below covers those fma
/// residual losses too. Per dispatch element the reduction surface includes
/// `k` chain TwoSums + 2·256 tree-level residual updates + the final running-sum
/// TwoSum + per-element activation pairs — over-counted uniformly as
/// `9·k + 4800` flush points
/// (each point = one possible `2^-126` loss), folded into the SAME
/// [`ny_core::ftz_safe_underflow_floor`] mechanism as the §0 value-lane flush
/// cover (the helper over-bounds `points·2^-126` with a normal-range f32).
/// This is the loss in the residual lane *before* its recovery multiplier. An
/// EFT consumer that later multiplies the residual by `r_slack` must publish
/// [`rung3_flush_safe_additive_scaled`] instead; adding this unscaled base after
/// that multiplication would under-charge the admitted zero flush whenever
/// `r_slack > 1`.
///
/// Charged UNCONDITIONALLY (also in the FTZ world and in legacy m0 mode):
/// widening-only, and at `k = 10^5` the floor is still `< 2^-100`. That is tiny
/// at the measured CROWN scales, but no minimum certified-margin invariant is
/// assumed: as with any sound widening, it may turn a sufficiently small proof
/// margin into unknown. One always-on constant avoids a mode-conditional that
/// could be wired to the wrong path.
pub(crate) fn rung3_flush_safe_additive(reduction_k: u32) -> Result<f32> {
    let flush_points = reduction_k
        .checked_mul(9)
        .and_then(|points| points.checked_add(4800))
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "sound rung-3 flush-point count 9*{reduction_k}+4800 overflows u32"
            ))
        })?;
    Ok(ny_core::ftz_safe_underflow_floor(flush_points))
}

/// Scale the rung-3 fma-residual flush floor by the exact downstream recovery
/// multiplier that the shader applies to its residual lane.
///
/// Both operands are f32 values, so their product is exactly representable in
/// f64 before [`up_f32`] rounds it outward for the uniform. Invalid multipliers
/// and any future non-finite outcome fail closed rather than publishing a
/// finite under-charge.
pub(crate) fn rung3_flush_safe_additive_scaled(
    reduction_k: u32,
    residual_slack: f32,
) -> Result<f32> {
    if !residual_slack.is_finite() || residual_slack < 1.0 {
        return Err(NyError::UnsupportedOp(format!(
            "sound EFT residual slack must be finite and >= 1, got {residual_slack}"
        )));
    }
    let base = rung3_flush_safe_additive(reduction_k)?;
    let scaled = up_f32(f64::from(base) * f64::from(residual_slack));
    if !scaled.is_finite() {
        return Err(NyError::UnsupportedOp(format!(
            "sound rung-3 residual flush floor overflows for k={reduction_k}, \
             residual_slack={residual_slack}"
        )));
    }
    Ok(scaled)
}

pub(crate) fn eft_r_slack_f32(k: usize) -> Result<f32> {
    let Some(residual_terms) = k
        .checked_mul(2)
        .and_then(|terms| terms.checked_add(2))
        .and_then(|terms| terms.checked_add(TREE_REDUCTION_RESIDUAL_ADDS))
    else {
        return Err(NyError::UnsupportedOp(format!(
            "sound EFT residual reduction length overflows for k={k}"
        )));
    };
    let g = f32_to_f64_exact(gamma_k_f32(residual_terms)?);
    if !g.is_finite() || g >= 1.0 {
        return Err(NyError::UnsupportedOp(format!(
            "sound EFT residual reduction for k={k} has gamma >= 1, so it \
             has no finite recovery slack"
        )));
    }
    let inv = 1.0 / (1.0 - g);
    let headroom = (1.0 + U).powi(6);
    Ok(up_f32(inv * headroom))
}

/// Round an `f64` UP to `f32` (outward, toward +∞ in magnitude for positive `x`;
/// for negative `x` steps toward the value that is `>= x`).
pub(crate) fn up_f32(x: f64) -> f32 {
    f64_to_f32_up(x)
}

// ---------------------------------------------------------------------------
// #daz-flush-cover-v2 — the DAZ operand-flush cover of the two AW-error combines
// ---------------------------------------------------------------------------

/// Process gate for the corrected DAZ operand-flush cover (`#daz-flush-cover-v2`).
/// DARK, default OFF ⇒ every uniform is byte-identical to the shipped build.
///
/// # Why the correction exists
///
/// Both AW-error combines (`CROWN_AW_ERROR_COMBINE_SHADER` and
/// `CROWN_EFT_MIN_COMBINE_SHADER`) charge the same §0 operand-flush floor
/// `flushacc·slack·F32_MIN_NORMAL` with
/// `flushacc = 1 + k + ‖a_i‖₁ + max_j‖w_j‖₁`. That term carries `‖w_j‖₁`
/// EXACTLY ONCE, but on a DAZ adapter there are TWO independent
/// `μ·‖w_j‖₁`-sized operand-flush channels feeding one output element:
///
/// 1. the coefficient GEMM `A@W` (and its EFT twin): a subnormal `a_il` is
///    zeroed before the multiply, losing up to `μ·|w_lj|` per tap;
/// 2. the PROPAGATED-error GEMM `prop = fl(err@|W|)`: a subnormal `err_il` is
///    zeroed the same way, losing another `μ·|w_lj|` per tap.
///
/// and a THIRD channel with no term at all:
///
/// 3. a subnormal WEIGHT `|w_lj|` is zeroed in that same `prop` GEMM, losing up
///    to `μ·err_il` per tap — i.e. `μ·‖err_i‖₁`, which nothing in `flushacc`
///    scales with.
///
/// Both are exact-rational-oracle counterexamples, not conjecture; see
/// `wgpu_device::flush_charge_oracle`. This gate applies the widening for (2)
/// (`2·max_j‖w_j‖₁`, plus a 2× engineering margin on the whole `flushacc`) and
/// FAILS CLOSED on (3) via [`refuse_subnormal_weight_under_daz_cover`].
///
/// # This gate does not, on its own, make Metal armable
///
/// The correction is a PRECONDITION for relaxing the rung-3 subnormal policy,
/// not a substitute for it. On a non-flushing adapter every term here is
/// multiplied by `2^-126` and changes nothing; an adapter whose core/DAZ lanes
/// flush is still refused. Only the separately charged exact-zero subnormal
/// fma-residual exception is admitted. Arming remains a human decision.
pub(crate) fn daz_flush_cover_v2_enabled() -> bool {
    std::env::var("NY_GPU_DAZ_FLUSH_COVER_V2").ok().as_deref() == Some("1")
}

/// #flush-charge: the audited `‖w‖₁` widening factor of `#daz-flush-cover-v2`.
/// `2×` for the second `μ‖w‖₁` operand-flush channel (the `prop` GEMM's `err`
/// operand) and `2×` engineering margin. Pinned by
/// `flush_charge_oracle::derived_cover_plus_subnormal_weight_refusal_encloses`
/// (worst measured consumption 0.4978 at this factor).
pub(crate) const CHARGED_W_L1_FACTOR: f32 = 4.0;

/// #flush-charge: concretize legacy-branch `slack` widening under charged-flush
/// authority. The shader carries ONE `max(|a_l|,|a_u|,xmax,1)` charge per tap
/// while a pure-flush adapter has up to FOUR first-order `μ·max` DAZ channels
/// per tap per side (two value products sharing one max term, the `e·xmax`
/// penalty product, and the `γ·|a|·xmax` term). `8 = 4 channels × 2 margin`.
/// Derived and pinned by
/// `flush_charge_oracle::charged_concretize_factor_covers_the_multi_channel_daz_demand`
/// (which also proves the UNwidened uniform does NOT enclose there).
pub(crate) const CHARGED_CONCRETIZE_SLACK_FACTOR: f32 = 8.0;

/// #flush-charge: bias-combine `slack` widening under charged-flush authority.
/// Two independent `μ·|b_j|` DAZ channels per tap (the value product's
/// subnormal `a_j` and the propagated term's subnormal `err_j`) against ONE
/// `max(|a_j|,|b_j|,1)` charge: `4 = 2 channels × 2 margin`. The third channel
/// (subnormal BIAS, `err_j·μ` with no covering term) is refused by
/// `FlushChargePolicy::refuse_subnormal_bias`. Derived and pinned by
/// `flush_charge_oracle::charged_bias_combine_factor_covers_the_double_daz_demand`.
pub(crate) const CHARGED_BIAS_COMBINE_SLACK_FACTOR: f32 = 4.0;

/// #flush-charge §E: activation intercept-bias `slack` widening
/// (`ActBiasParams.slack`, `CROWN_ACTIVATION_INTERCEPT_BIAS_SHADER` legacy arm)
/// under charged-flush authority. Per tap the flushacc carries ONE
/// `max(|a_j|,|li_j|,|ui_j|,1)` charge while a pure-flush adapter routes up to
/// TWO first-order `μ·max` demands through it:
///
/// 1. value-DAZ of a subnormal coefficient `a_j` in `a_j·sel_int`
///    (loss ≤ `μ·|sel| ≤ μ·max(|li|,|ui|)`), and
/// 2. DAZ of a subnormal incoming radius `err_j` in `err_j·(|li|+|ui|)`, which
///    vaporizes the propagated cover; the DEMAND it was covering is only
///    `err_j·Lip ≤ μ·max(|li|,|ui|)` (the `|li|+|ui|` form is 2× over-provisioned
///    against the Lipschitz constant of `v ↦ v·int(v)` — the ADDITIVE1
///    one-max-per-tap lemma, `act_bias_per_tap_demand_is_one_max_beyond_exact_covers`).
///
/// `4 = 2 channels × 2 margin`. Result-FTZ events on the same multiplies are
/// mutually exclusive with the operand-DAZ losses (a DAZ-zeroed multiplicand
/// yields an exact ±0 product) and are `< μ` each, covered by the rung-3
/// additive. The third channel (subnormal INTERCEPT, `err_j·μ` with no covering
/// term) is UNCHARGEABLE and stays refused by
/// `FlushChargePolicy::refuse_subnormal_slopes` (which covers intercepts);
/// negative pin: `charged_act_bias_cannot_cover_subnormal_intercepts`.
/// Derived and pinned by
/// `flush_charge_oracle::charged_act_bias_factor_covers_the_double_daz_demand`.
pub(crate) const CHARGED_ACT_BIAS_SLACK_FACTOR: f32 = 4.0;

/// Widen the `w_l1_max` uniform of the AW-error combines to the derived cover.
///
/// Returns `w_l1_max` unchanged when `armed` is false (byte-identical). When
/// armed, returns `4·w_l1_max`: `2×` for the second `μ‖w‖₁` channel (the `prop`
/// GEMM's `err` operand) and `2×` engineering margin, since the shader's
/// `flushacc` chain spends its `(1+u)^4` slack headroom on its own rounding and
/// the measured budget consumption of the 2×-only form reaches 0.9956 (exact
/// oracle, 300 000 adversarial rows). At `4×` the measured worst consumption is
/// 0.4978. The term is `·F32_MIN_NORMAL`, so the cost is ≤ `2^-124·‖w‖₁`.
///
/// `armed` is the OR of the env experiment gate
/// ([`daz_flush_cover_v2_enabled`]) and the device's charged-flush authority
/// (`charged_flush_authority_cached().is_some()`), computed at the call site so
/// the arming input is explicit and testable rather than ambient.
///
/// Multiplying an `f32` by 4 is EXACT, so no rounding is introduced; a
/// non-finite result fails closed rather than publishing a saturated floor.
pub(crate) fn daz_flush_cover_w_l1(w_l1_max: f32, armed: bool) -> Result<f32> {
    if !armed {
        return Ok(w_l1_max);
    }
    let widened = w_l1_max * CHARGED_W_L1_FACTOR;
    if !widened.is_finite() {
        return Err(NyError::UnsupportedOp(format!(
            "#daz-flush-cover-v2: weight column L1 {w_l1_max:e} overflows the \
             widened operand-flush cover; refusing rather than publishing a \
             saturated floor"
        )));
    }
    Ok(widened)
}

/// #flush-charge: shared outward slack widening for the charged concretize /
/// bias-combine uniforms. `up_f32(slack · factor)` with fail-closed validation:
/// the exact product of two f32 values is representable in f64, so the only
/// rounding is the final outward cast.
fn charged_widened_slack(slack: f32, factor: f32, what: &str) -> Result<f32> {
    if !slack.is_finite() || slack < 1.0 || !factor.is_finite() || factor < 1.0 {
        return Err(NyError::UnsupportedOp(format!(
            "#flush-charge: {what} widening needs finite slack >= 1 and \
             factor >= 1, got slack={slack:e}, factor={factor:e}"
        )));
    }
    let widened = up_f32(f64::from(slack) * f64::from(factor));
    if !widened.is_finite() {
        return Err(NyError::UnsupportedOp(format!(
            "#flush-charge: {what} widened slack overflows for slack={slack:e},\
             factor={factor:e}; refusing rather than publishing a saturated \
             uniform"
        )));
    }
    Ok(widened)
}

/// #flush-charge: widen the concretize `slack` uniform by the policy's
/// concretize factor (outward, fail-closed). See
/// [`CHARGED_CONCRETIZE_SLACK_FACTOR`].
pub(crate) fn charged_concretize_slack(slack: f32, factor: f32) -> Result<f32> {
    charged_widened_slack(slack, factor, "concretize slack")
}

/// #flush-charge: widen a bias-combine `slack` uniform by the policy's bias
/// factor (outward, fail-closed). See [`CHARGED_BIAS_COMBINE_SLACK_FACTOR`].
pub(crate) fn charged_bias_slack(slack: f32, factor: f32) -> Result<f32> {
    charged_widened_slack(slack, factor, "bias-combine slack")
}

/// #flush-charge §E: widen the activation intercept-bias `slack` uniform
/// (`ActBiasParams.slack`) by the policy's act-bias factor (outward,
/// fail-closed). See [`CHARGED_ACT_BIAS_SLACK_FACTOR`].
pub(crate) fn charged_act_bias_slack(slack: f32, factor: f32) -> Result<f32> {
    charged_widened_slack(slack, factor, "activation intercept-bias slack")
}

/// Fail-closed precondition for the derived cover: the `prop = fl(err@|W|)`
/// GEMM's `|W|` operand must not be DAZ-flushable, i.e. the weight tensor must
/// contain no subnormal. The loss `Σ_l err_il·|w_lj|` over subnormal weights is
/// bounded by `μ·‖err_i‖₁`, and `‖err_i‖₁` is a DEVICE-resident quantity that no
/// uniform in either combine carries — so the only sound response without a new
/// per-row reduction is refusal. The host already walks the whole weight tensor
/// to build `w_l1_max`, so this costs one extra predicate on that pass.
///
/// No-op when `armed` is false (the whole channel is unreachable there because
/// the sound-GPU authority ladder still refuses flushing adapters). `armed` is
/// the same explicit OR as [`daz_flush_cover_w_l1`]'s.
pub(crate) fn refuse_subnormal_weight_under_daz_cover(
    weight: &[f32],
    what: &str,
    armed: bool,
) -> Result<()> {
    if !armed {
        return Ok(());
    }
    if weight
        .iter()
        .any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE)
    {
        return Err(NyError::UnsupportedOp(format!(
            "#daz-flush-cover-v2: {what} contains a SUBNORMAL weight. On a \
             DAZ adapter that operand is zeroed inside the propagated-error \
             GEMM fl(err@|W|), losing up to ‖err_i‖₁·2^-126 which no combine \
             uniform bounds. Refusing the sound GPU path for this layer."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        charged_act_bias_slack, charged_bias_slack, charged_concretize_slack, combine_slack_f32,
        daz_flush_cover_v2_enabled, daz_flush_cover_w_l1, eft_r_slack_f32, gamma_k_f32,
        refuse_subnormal_weight_under_daz_cover, rung3_flush_safe_additive,
        rung3_flush_safe_additive_scaled, CHARGED_ACT_BIAS_SLACK_FACTOR,
        CHARGED_BIAS_COMBINE_SLACK_FACTOR, CHARGED_CONCRETIZE_SLACK_FACTOR, CHARGED_W_L1_FACTOR, U,
    };

    /// #daz-flush-cover-v2 is DARK: with the gate unset (the default, and the
    /// state of every shipped run) both helpers are the identity when unarmed,
    /// so every uniform byte the fold uploads is unchanged — including for a
    /// weight tensor full of subnormals, which must NOT start refusing on the
    /// default path. `armed` on the production call sites is
    /// `env gate || charged authority`, both false by default (the charged
    /// authority source gate is compile-time closed).
    #[test]
    fn daz_flush_cover_v2_is_byte_identical_while_dark() {
        assert!(
            !daz_flush_cover_v2_enabled(),
            "NY_GPU_DAZ_FLUSH_COVER_V2 must be OFF by default"
        );
        let armed = daz_flush_cover_v2_enabled();
        for w in [0.0f32, 1.0, 1.0e30, f32::MIN_POSITIVE, f32::MAX] {
            assert_eq!(
                daz_flush_cover_w_l1(w, armed)
                    .expect("dark path never refuses")
                    .to_bits(),
                w.to_bits(),
                "w_l1_max {w:e} was perturbed while the gate is dark"
            );
        }
        let subnormal_weights = [f32::from_bits(0x007f_ffff), 1.0, 0.0];
        assert!(
            refuse_subnormal_weight_under_daz_cover(&subnormal_weights, "test", armed).is_ok(),
            "the dark path must not introduce a new refusal"
        );
    }

    /// The widened cover is a strict WIDENING (never below the shipped value)
    /// and fails closed rather than saturating. Checked on the arithmetic
    /// directly so it holds regardless of the process gate.
    #[test]
    fn daz_flush_cover_widening_is_outward_and_fails_closed_on_overflow() {
        for w in [0.0f32, 1.0, 1.0e30, f32::MIN_POSITIVE] {
            let widened = daz_flush_cover_w_l1(w, true).expect("finite widening");
            assert!(widened.is_finite() && widened >= w, "w={w:e}");
            // doubling twice is EXACT for f32, so no rounding is introduced
            assert_eq!(f64::from(widened), f64::from(w) * 4.0, "w={w:e}");
        }
        assert!(
            daz_flush_cover_w_l1(f32::MAX, true).is_err(),
            "the armed cover must fail closed on overflow, not saturate"
        );
        let subnormal_weights = [f32::from_bits(0x007f_ffff), 1.0, 0.0];
        assert!(
            refuse_subnormal_weight_under_daz_cover(&subnormal_weights, "test", true).is_err(),
            "the armed cover must refuse a subnormal weight"
        );
    }

    /// #flush-charge: the charged slack widenings are outward (never below the
    /// exact product) and fail closed on invalid/overflowing inputs. The factor
    /// constants themselves are exact powers of two, so widening by them
    /// introduces no rounding at all.
    #[test]
    fn charged_slack_widenings_are_outward_and_fail_closed() {
        for factor in [
            CHARGED_CONCRETIZE_SLACK_FACTOR,
            CHARGED_BIAS_COMBINE_SLACK_FACTOR,
            CHARGED_ACT_BIAS_SLACK_FACTOR,
        ] {
            assert!(factor >= 1.0 && factor.log2().fract() == 0.0);
            for slack in [1.0f32, 1.000_001, 1.5, 1.0e30] {
                let widened = charged_concretize_slack(slack, factor).expect("finite widening");
                assert!(
                    f64::from(widened) >= f64::from(slack) * f64::from(factor),
                    "slack={slack:e} factor={factor:e}: widened {widened:e} \
                     fell below the exact product"
                );
                assert_eq!(
                    charged_bias_slack(slack, factor).unwrap().to_bits(),
                    widened.to_bits(),
                    "the two widenings share one audited arithmetic path"
                );
                assert_eq!(
                    charged_act_bias_slack(slack, factor).unwrap().to_bits(),
                    widened.to_bits(),
                    "the act-bias widening shares the same audited arithmetic path"
                );
            }
        }
        // exact-power-of-two factors keep the widening rounding-free
        assert_eq!(
            charged_concretize_slack(1.5, 8.0).unwrap().to_bits(),
            12.0f32.to_bits()
        );
        for invalid in [
            (0.5f32, 4.0f32),
            (f32::NAN, 4.0),
            (1.0, 0.5),
            (1.0, f32::INFINITY),
            (f32::MAX, 8.0),
        ] {
            assert!(
                charged_concretize_slack(invalid.0, invalid.1).is_err(),
                "invalid ({}, {}) was accepted",
                invalid.0,
                invalid.1
            );
        }
        // the w_l1 factor is the audited #daz-flush-cover-v2 4x
        assert_eq!(CHARGED_W_L1_FACTOR, 4.0);
    }

    #[test]
    fn gamma_k_is_never_narrowed_below_exact_factor() {
        for k in [
            0usize,
            1,
            2,
            3,
            255,
            256,
            257,
            1024,
            1 << 20,
            1 << 23,
            3 << 22,
            (1 << 24) - 1,
        ] {
            let ku = (k as f64) * U;
            let exact = ku / (1.0 - ku);
            let published = f64::from(gamma_k_f32(k).expect("finite Higham regime"));
            assert!(
                published >= exact,
                "k={k}: published gamma {published:e} is below exact {exact:e}"
            );
        }
    }

    #[test]
    fn gamma_k_has_no_finite_fallback_outside_higham_regime() {
        assert_eq!(gamma_k_f32(3 << 22).unwrap(), 3.0);
        assert!(gamma_k_f32(1 << 24).is_err());
        assert!(gamma_k_f32((1 << 24) + 1).is_err());
        assert!(gamma_k_f32(usize::MAX).is_err());
    }

    #[test]
    fn reduction_recovery_slacks_fail_closed_when_gamma_reaches_one() {
        assert!(combine_slack_f32((1 << 23) - 1).unwrap().is_finite());
        assert!(combine_slack_f32(1 << 23).is_err());
        assert!(combine_slack_f32(3 << 22).is_err());

        // eft_r_slack_f32(k) recovers a `2k + 2 + TREE_REDUCTION_RESIDUAL_ADDS`
        // term reduction (#eft-err U3). Counting the tree's 16 adds moved this
        // boundary DOWN by 8 — that is the fix working: the kernel really does
        // perform those adds, so the length at which recovery becomes impossible
        // is genuinely lower than the old `2k + 2` count implied.
        const K_MAX: usize = ((1 << 23) - 20) / 2; // largest k with 2k+18 < 2^23
        assert!(eft_r_slack_f32(K_MAX).unwrap().is_finite());
        assert!(eft_r_slack_f32(K_MAX + 1).is_err());
        assert!(eft_r_slack_f32(usize::MAX).is_err());
    }

    #[test]
    fn rung3_residual_flush_floor_is_scaled_outward_and_fails_closed() {
        const K_POINTS_MAX: u32 = (u32::MAX - 4800) / 9;
        for k in [0u32, 1, 256, 100_000, K_POINTS_MAX] {
            let base = rung3_flush_safe_additive(k).expect("test flush-point count fits u32");
            assert_eq!(
                rung3_flush_safe_additive_scaled(k, 1.0)
                    .expect("identity residual slack is valid")
                    .to_bits(),
                base.to_bits(),
                "k={k}: identity scaling changed the base floor"
            );
            for slack in [1.000_001f32, 1.5, 16.0, f32::from_bits(0x4b00_0001)] {
                let scaled = rung3_flush_safe_additive_scaled(k, slack)
                    .expect("finite test multiplier must have a finite floor");
                let exact = f64::from(base) * f64::from(slack);
                assert!(
                    f64::from(scaled) >= exact,
                    "k={k}, slack={slack}: scaled floor {scaled:e} < exact {exact:e}"
                );
            }
        }

        for invalid in [0.0, 0.999_999_94, f32::NAN, f32::INFINITY] {
            assert!(
                rung3_flush_safe_additive_scaled(1, invalid).is_err(),
                "invalid residual slack {invalid:?} was accepted"
            );
        }
        const K_MAX: usize = ((1 << 23) - 20) / 2;
        let boundary_slack =
            eft_r_slack_f32(K_MAX).expect("largest admitted EFT reduction has finite slack");
        let boundary_base = rung3_flush_safe_additive(u32::try_from(K_MAX).unwrap()).unwrap();
        let boundary_scaled =
            rung3_flush_safe_additive_scaled(u32::try_from(K_MAX).unwrap(), boundary_slack)
                .expect("largest admitted EFT reduction has a finite scaled floor");
        assert!(
            f64::from(boundary_scaled) >= f64::from(boundary_base) * f64::from(boundary_slack),
            "largest admitted EFT slack was not charged outward"
        );
        assert!(
            rung3_flush_safe_additive(K_POINTS_MAX).is_ok(),
            "largest representable 9k+4800 count must be accepted"
        );
        assert!(
            rung3_flush_safe_additive(K_POINTS_MAX + 1).is_err(),
            "an overflowing 9k+4800 count must fail closed, not saturate"
        );
    }
}
