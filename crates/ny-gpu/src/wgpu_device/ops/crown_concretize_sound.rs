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

use ny_core::{NyError, Result};

use super::super::WgpuDevice;
use crate::wgpu_device::sound_consts::combine_slack_f32;

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

impl WgpuDevice {
    /// Sound concretization on the GPU. All slices are row-major `(num_specs ×
    /// input_dim)` for the coefficient/err matrices, length `input_dim` for the
    /// input box, length `num_specs` for the biases. Returns `(lower, upper)`,
    /// each length `num_specs`, a sound enclosure of the network output range.
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
        )
    }

    /// #batched-bab: domain-block form of [`concretize_sound_gpu`]. Rows are stacked in
    /// `n_domains = num_specs / num_specs_per_dom` blocks of `num_specs_per_dom` rows,
    /// and the input box is `n_domains * input_dim` wide — row `s` concretizes against
    /// its OWN domain's box `[dom*input_dim .. )`, `dom = s / num_specs_per_dom`
    /// (CROWN_CONCRETIZE_SOUND_SHADER dbase, HOLE 3). With `num_specs_per_dom ==
    /// num_specs` (single domain) this is byte-identical to the pre-batch path.
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
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let coeff = num_specs * input_dim;
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
        // Per-domain input boxes are stacked n_domains-wide; single domain → input_dim.
        let n_domains = num_specs.checked_div(num_specs_per_dom).unwrap_or(1);
        let box_len = n_domains * input_dim;
        if input_lower.len() != box_len || input_upper.len() != box_len {
            return Err(NyError::shape_mismatch(
                vec![box_len],
                vec![input_lower.len()],
            ));
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
        // #wg-limit-guard (SOUNDNESS, fail-closed): this shader dispatches ONE workgroup
        // per spec row (`wg_id.x = spec_row`), so `dispatch_workgroups(num_specs)`
        // overruns the wgpu `max_compute_workgroups_per_dimension` cap (default 65535,
        // NOT raised by `NY_GPU_BIG_BINDINGS`) once `num_specs > max_wg`. An over-limit
        // dispatch is UB on some drivers (silently wrong — closer-to-zero, UNSOUND —
        // bound, or a crash), so fail closed and let the caller sub-chunk / fall back to
        // the sound CPU concretize. Value-neutral for every in-range call.
        let max_wg = self
            .device
            .limits()
            .max_compute_workgroups_per_dimension
            .max(1) as usize;
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
        const U: f64 = f64::from_bits(0x3E70_0000_0000_0000); // 2^-24
        let nu = (input_dim as f64) * U;
        let gamma_n = if nu < 0.5 { nu / (1.0 - nu) } else { 2.0 * nu };
        // #eft-err: the measured-residual concretize. Cached-only gate read (this
        // op runs inside a GPU-checked section — see the deadlock guard note in
        // eft_selfcheck.rs); uninitialized/refused ⇒ legacy γ_n, byte-identical.
        let eft_on = std::env::var("NY_EFT_ERR").ok().as_deref() == Some("1")
            && self.eft_primitives_cached();
        let params = SoundConcretizeParams {
            num_specs: num_specs as u32,
            input_dim: input_dim as u32,
            gamma_n: gamma_n as f32,
            additive: ny_core::ftz_safe_underflow_floor(input_dim as u32),
            slack: combine_slack_f32(input_dim),
            num_specs_per_dom: num_specs_per_dom as u32,
            eft_mode: u32::from(eft_on),
            eft_r_slack: if eft_on {
                // 4 residual terms per tap over the strided dot + the tree
                // reduction's captured adds + final-assembly headroom: the
                // γ_{2·(2n)+2} cover of eft_r_slack_f32(2n) dominates the count.
                super::super::sound_consts::eft_r_slack_f32(2 * input_dim)
            } else {
                0.0
            },
        };

        self.run_gpu_checked("concretize_sound_gpu", || {
            let (pipeline, layout) = Self::create_crown_concretize_sound_pipeline(&self.device);

            let storage = |data: &[f32], label: &str| -> wgpu::Buffer {
                let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (data.len().max(1) * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if !data.is_empty() {
                    self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
                }
                buf
            };
            let out_buf = |label: &str| -> wgpu::Buffer {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (num_specs * size_of::<f32>()) as u64,
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
                layout: &layout,
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
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(num_specs as u32, 1, 1);
            }
            // One staging buffer per output, copied after the pass.
            let mut stage = |src: &wgpu::Buffer, label: &str| -> wgpu::Buffer {
                let s = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (num_specs * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(src, 0, &s, 0, (num_specs * size_of::<f32>()) as u64);
                s
            };
            let st_l = stage(&b_ol, "sc_stage_lower");
            let st_u = stage(&b_ou, "sc_stage_upper");
            self.queue.submit(Some(encoder.finish()));

            let lower = Self::read_buffer(&self.device, &st_l, num_specs)?;
            let upper = Self::read_buffer(&self.device, &st_u, num_specs)?;
            Ok((lower, upper))
        })
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
            over, 1, input_dim, &al, &au, &ale, &aue, &xl, &xu, &bl, &bu,
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
                max_wg, 1, input_dim, &al, &au, &ale, &aue, &xl, &xu, &bl, &bu,
            )
            .expect("at-limit concretize must succeed");
        assert_eq!(lo.len(), max_wg);
        assert_eq!(hi.len(), max_wg);
        assert!(lo.iter().chain(hi.iter()).all(|v| v.is_finite()));
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

    /// By-construction check the Vulkan oracle CANNOT do (Vulkan keeps subnormals):
    /// the §0 amplified operand-flush term must widen each bound by at least
    /// `xmax·FLT_MIN` even for a subnormal coefficient the GPU computes exactly here.
    /// Under Metal FTZ that subnormal `a` flushes to 0 and the true product `a·x` (a
    /// NORMAL f32) would be silently dropped; the amplified floor certifies it back.
    /// The OLD weight-independent `8·n·η` floor emitted a widening ~90 binary orders
    /// of magnitude too tight here — a false-VERIFIED break. This is the concretize
    /// twin of `sound_gpu_ibp_flush_radius_amplified_by_weight_t1_1`.
    #[test]
    fn sound_concretize_amplified_flush_covers_subnormal_times_huge() {
        let _g = gpu_test_serial_guard();
        let device = require_device();

        // One spec, one input. a·x = 2^-130 · 2^100 = 2^-30 (a NORMAL f32).
        let a = 2.0f32.powi(-130); // subnormal: Vulkan preserves it, Metal flushes to 0
        let x = 2.0f32.powi(100); // huge magnitude
        let (lo, hi) = device
            .concretize_sound_gpu(1, 1, &[a], &[a], &[0.0], &[0.0], &[x], &[x], &[0.0], &[0.0])
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
}
