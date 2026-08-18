// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! SOUND (verdict-legal) MaxPool2d CROWN-backward on the GPU (`docs/SOUND_GPU_IBP_PLAN.md`
//! T1.2). Transposes an incoming linear frontier on the maxpool OUTPUT into a
//! frontier on the INPUT, mirroring the PROVEN CPU winner/i* relaxation
//! (`ny-propagate` `layers/pooling/max.rs::propagate_linear_with_bounds`) so the
//! returned coefficient interval + directed bias is a SUPERSET of the CPU one — and
//! therefore of the true reachable range.
//!
//! Split GPU / host:
//! - The per-window `i*`+definite metadata (packed `window_meta[w]`) and each
//!   window's `max_upper` are computed ONCE on the host in f32 (independent of the
//!   spec/output rows).
//! - The COEFFICIENT gather (the O(num_outputs·input_size) work) runs on the GPU
//!   (`MAXPOOL_CROWN_SOUND` shader): definite winner → route both rows through `i*`;
//!   else route the lower row iff `la>0`, upper iff `ua<0`; per-coefficient error
//!   `3·γ_k·S·slack + additive` (coefficient-1 accumulation, NORMAL FTZ-safe floor).
//! - The BIAS (the `la<0`→`la·max_upper` / `ua>0`→`ua·max_upper` CONSTANT arms over
//!   non-definite windows) is folded on the host with exact bit-level f32→f64 lifts
//!   and a directed f64 step after every addition, then published with a DAZ-safe
//!   directed f64→f32 conversion. This encloses the exact host sum even when many
//!   additions round in the same direction.
//!
//! Any wgpu error → `Err` (the shared `run_gpu_checked`) so a verdict is never
//! decided by a failed op — the caller keeps the proven-sound CPU relaxation.

use ny_core::dd::{next_down_f64, next_up_f64};
use ny_core::{
    f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up, ftz_safe_underflow_floor, NyError, Result,
};

use crate::wgpu_device::params::MaxpoolCrownSoundParams;
use crate::wgpu_device::sound_consts::{combine_slack_f32, gamma_k_f32};
use crate::wgpu_device::WgpuDevice;

use super::gpu_checked_u32;
use super::ibp_forward::create_buffer;

fn accumulate_lower_bias_outward(accumulator: f64, coefficient: f32, max_upper: f32) -> f64 {
    let coefficient = f32_to_f64_exact(coefficient);
    if coefficient < 0.0 {
        // A product of two finite binary32 values is exact in binary64: at most
        // 48 significand bits. Only the addition needs a directed rounding step.
        next_down_f64(accumulator + coefficient * f32_to_f64_exact(max_upper))
    } else {
        accumulator
    }
}

fn accumulate_upper_bias_outward(accumulator: f64, coefficient: f32, max_upper: f32) -> f64 {
    let coefficient = f32_to_f64_exact(coefficient);
    if coefficient > 0.0 {
        next_up_f64(accumulator + coefficient * f32_to_f64_exact(max_upper))
    } else {
        accumulator
    }
}

fn max_f32_by_exact_lift(lhs: f32, rhs: f32) -> f32 {
    let lhs_bits = lhs.to_bits();
    let rhs_bits = rhs.to_bits();
    let lhs_nan = lhs_bits & 0x7f80_0000 == 0x7f80_0000 && lhs_bits & 0x007f_ffff != 0;
    let rhs_nan = rhs_bits & 0x7f80_0000 == 0x7f80_0000 && rhs_bits & 0x007f_ffff != 0;
    if lhs_nan || rhs_nan {
        return f32::NAN;
    }
    if f32_to_f64_exact(rhs) > f32_to_f64_exact(lhs) {
        rhs
    } else {
        lhs
    }
}

/// Result of a sound MaxPool2d CROWN backward: the transposed frontier on the maxpool
/// INPUT — coefficients + their certified per-coefficient error + directed bias.
// Fields are read by the gpu-tests enclosure oracle only until T1.2 is wired into a
// production dispatch path (see `maxpool_crown_backward_gpu_sound`).
#[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
pub(crate) struct MaxpoolCrownResult {
    /// `[num_outputs · input_size]` lower/upper coefficient rows.
    pub(crate) lower_a: Vec<f32>,
    pub(crate) upper_a: Vec<f32>,
    /// `[num_outputs · input_size]` certified per-coefficient error (`≥ 0`).
    pub(crate) lower_a_err: Vec<f32>,
    pub(crate) upper_a_err: Vec<f32>,
    /// `[num_outputs]` directed-rounded bias.
    pub(crate) lower_b: Vec<f32>,
    pub(crate) upper_b: Vec<f32>,
}

impl WgpuDevice {
    /// SOUND MaxPool2d CROWN backward (T1.2). All inputs are row-major f32:
    /// `lower_a`/`upper_a` are `[num_outputs · output_size]`, `lower_b`/`upper_b` are
    /// `[num_outputs]`, `pre_lower`/`pre_upper` are the maxpool INPUT bounds
    /// `[input_size]`. `input_size = channels·in_h·in_w`, `output_size =
    /// channels·out_h·out_w`. Returns the transposed frontier on the input.
    // T1.2 landed + oracle-tested (gpu-tests enclosure test) but not yet wired into a
    // production dispatch path; kept compiled so the wiring step is a pure call-site add.
    #[cfg_attr(not(all(test, feature = "gpu-tests")), allow(dead_code))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn maxpool_crown_backward_gpu_sound(
        &self,
        lower_a: &[f32],
        upper_a: &[f32],
        lower_b: &[f32],
        upper_b: &[f32],
        pre_lower: &[f32],
        pre_upper: &[f32],
        num_outputs: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> Result<MaxpoolCrownResult> {
        if sh == 0 || sw == 0 {
            return Err(NyError::InvalidSpec("maxpool crown: zero stride".into()));
        }
        let input_size = channels * in_h * in_w;
        let output_size = channels * out_h * out_w;
        if lower_a.len() != num_outputs * output_size || upper_a.len() != num_outputs * output_size
        {
            return Err(NyError::shape_mismatch(
                vec![num_outputs * output_size],
                vec![lower_a.len()],
            ));
        }
        if lower_b.len() != num_outputs || upper_b.len() != num_outputs {
            return Err(NyError::shape_mismatch(
                vec![num_outputs],
                vec![lower_b.len()],
            ));
        }
        if pre_lower.len() != input_size || pre_upper.len() != input_size {
            return Err(NyError::shape_mismatch(
                vec![input_size],
                vec![pre_lower.len()],
            ));
        }
        for (index, (&lower, &upper)) in pre_lower.iter().zip(pre_upper).enumerate() {
            let lower_bits = lower.to_bits();
            let upper_bits = upper.to_bits();
            if lower_bits & 0x7f80_0000 == 0x7f80_0000
                || upper_bits & 0x7f80_0000 == 0x7f80_0000
                || f32_to_f64_exact(lower) > f32_to_f64_exact(upper)
            {
                return Err(NyError::InvalidSpec(format!(
                    "maxpool crown: invalid pre-activation interval at {index}: \
                     [{lower}, {upper}]"
                )));
            }
        }

        // --- Host: per-window i*+definite metadata (packed) + max_upper. ---
        let num_windows = output_size;
        let mut window_meta = vec![0xFFFF_FFFFu32; num_windows]; // sentinel = empty window
        let mut win_max_upper = vec![0.0f32; num_windows];
        let in_hw = in_h * in_w;
        let out_hw = out_h * out_w;
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let w = c * out_hw + oh * out_w + ow;
                    // Collect the valid (non-padding) taps of this window.
                    let mut taps: Vec<(usize, f32, f32)> = Vec::with_capacity(kh * kw);
                    for kh_i in 0..kh {
                        let ih = (oh * sh + kh_i) as isize - ph as isize;
                        if ih < 0 || ih >= in_h as isize {
                            continue;
                        }
                        for kw_i in 0..kw {
                            let iw = (ow * sw + kw_i) as isize - pw as isize;
                            if iw < 0 || iw >= in_w as isize {
                                continue;
                            }
                            let flat = c * in_hw + (ih as usize) * in_w + (iw as usize);
                            taps.push((flat, pre_lower[flat], pre_upper[flat]));
                        }
                    }
                    if taps.is_empty() {
                        // All positions are padding: the output is max over an
                        // empty set (-inf), which no finite frontier row can
                        // bound. `MaxPool2dLayer::output_size` rejects the
                        // padding >= kernel geometry that creates such windows;
                        // refuse here too rather than leave the sentinel to
                        // silently drop the window's coefficients.
                        return Err(NyError::InvalidSpec(format!(
                            "maxpool crown: pooling window at output ({oh},{ow}) \
                             contains no input positions: kernel=({kh},{kw}), \
                             padding=({ph},{pw})"
                        )));
                    }
                    // i* = argmax lower. `l > istar_lower` is false for a NaN `l`
                    // (NaN never wins) — matching the CPU `max_by(partial_cmp)` which
                    // keeps the current max on an incomparable (NaN) element. The exact
                    // tie-break is irrelevant to soundness (any max-lower input is a
                    // valid `y ≥ x_{i*}` witness).
                    let mut istar = taps[0].0;
                    let mut istar_lower = taps[0].1;
                    let mut mu = f32::NEG_INFINITY;
                    for &(flat, l, u) in &taps {
                        if f32_to_f64_exact(l) > f32_to_f64_exact(istar_lower) {
                            istar = flat;
                            istar_lower = l;
                        }
                        mu = max_f32_by_exact_lift(mu, u);
                    }
                    // is_definite: l_{i*} ≥ max over taps≠i* of u.
                    let mut max_upper_excl = f32::NEG_INFINITY;
                    for &(flat, _l, u) in &taps {
                        if flat != istar {
                            max_upper_excl = max_f32_by_exact_lift(max_upper_excl, u);
                        }
                    }
                    let is_definite =
                        f32_to_f64_exact(istar_lower) >= f32_to_f64_exact(max_upper_excl);
                    let packed = (istar as u32) | (u32::from(is_definite) << 31);
                    window_meta[w] = packed;
                    win_max_upper[w] = mu;
                }
            }
        }

        // --- GPU: coefficient gather. ---
        // k = max windows that can route to one input + 3 (conservative γ count).
        let max_cover = kh
            .div_ceil(sh)
            .checked_mul(kw.div_ceil(sw))
            .ok_or_else(|| NyError::InvalidSpec("maxpool crown cover overflow".into()))?;
        let k = max_cover.checked_add(3).ok_or_else(|| {
            NyError::InvalidSpec("maxpool crown reduction length overflow".into())
        })?;
        let k_u32 = gpu_checked_u32(k, "maxpool crown k")?;
        let total = num_outputs
            .checked_mul(input_size)
            .ok_or_else(|| NyError::InvalidSpec("maxpool crown output overflow".into()))?;
        let params = MaxpoolCrownSoundParams {
            num_outputs: gpu_checked_u32(num_outputs, "mp num_outputs")?,
            input_size: gpu_checked_u32(input_size, "mp input_size")?,
            output_size: gpu_checked_u32(output_size, "mp output_size")?,
            channels: gpu_checked_u32(channels, "mp channels")?,
            in_h: gpu_checked_u32(in_h, "mp in_h")?,
            in_w: gpu_checked_u32(in_w, "mp in_w")?,
            out_h: gpu_checked_u32(out_h, "mp out_h")?,
            out_w: gpu_checked_u32(out_w, "mp out_w")?,
            kh: gpu_checked_u32(kh, "mp kh")?,
            kw: gpu_checked_u32(kw, "mp kw")?,
            sh: gpu_checked_u32(sh, "mp sh")?,
            sw: gpu_checked_u32(sw, "mp sw")?,
            ph: gpu_checked_u32(ph, "mp ph")?,
            pw: gpu_checked_u32(pw, "mp pw")?,
            gamma_k: gamma_k_f32(k)?,
            slack: combine_slack_f32(k)?,
            additive: ftz_safe_underflow_floor(k_u32),
            total: gpu_checked_u32(total, "mp total")?,
            _p0: 0,
            _p1: 0,
        };

        let (new_lower_a, new_upper_a, err_comb) =
            self.run_maxpool_crown_gather(&params, lower_a, upper_a, &window_meta, total)?;
        let (lower_a_err, upper_a_err) = err_comb.split_at(total);

        // --- Host: directed bias (bit-exact lifts + outward f64 accumulation). ---
        let mut nlb: Vec<f64> = lower_b
            .iter()
            .map(|&value| f32_to_f64_exact(value))
            .collect();
        let mut nub: Vec<f64> = upper_b
            .iter()
            .map(|&value| f32_to_f64_exact(value))
            .collect();
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let w = c * out_hw + oh * out_w + ow;
                    let meta = window_meta[w];
                    if meta == 0xFFFF_FFFF || (meta >> 31) & 1 == 1 {
                        continue; // empty or definite-winner window ⇒ no bias
                    }
                    for out in 0..num_outputs {
                        let index = out * output_size + w;
                        nlb[out] = accumulate_lower_bias_outward(
                            nlb[out],
                            lower_a[index],
                            win_max_upper[w],
                        );
                        nub[out] = accumulate_upper_bias_outward(
                            nub[out],
                            upper_a[index],
                            win_max_upper[w],
                        );
                    }
                }
            }
        }
        let lower_b_out: Vec<f32> = nlb.iter().map(|&value| f64_to_f32_down(value)).collect();
        let upper_b_out: Vec<f32> = nub.iter().map(|&value| f64_to_f32_up(value)).collect();

        Ok(MaxpoolCrownResult {
            lower_a: new_lower_a,
            upper_a: new_upper_a,
            lower_a_err: lower_a_err.to_vec(),
            upper_a_err: upper_a_err.to_vec(),
            lower_b: lower_b_out,
            upper_b: upper_b_out,
        })
    }

    /// Dispatch the `MAXPOOL_CROWN_SOUND` coefficient gather. Bindings: 0 params
    /// (uniform), 1 lower_a, 2 upper_a, 3 window_meta (u32) RO; 4 new_lower_a,
    /// 5 new_upper_a, 6 err_comb (2·total) RW. Returns `(new_lower_a, new_upper_a,
    /// err_comb)`. Wrapped in `run_gpu_checked`.
    fn run_maxpool_crown_gather(
        &self,
        params: &MaxpoolCrownSoundParams,
        lower_a: &[f32],
        upper_a: &[f32],
        window_meta: &[u32],
        total: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let f32_size = size_of::<f32>() as u64;
        self.run_gpu_checked("maxpool_crown_sound", || {
            let params_buf = create_buffer(
                &self.device,
                "mp_crown_params",
                size_of::<MaxpoolCrownSoundParams>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            self.queue
                .write_buffer(&params_buf, 0, bytemuck::bytes_of(params));

            let mk_ro = |label: &'static str, data: &[u8]| {
                let buf = create_buffer(
                    &self.device,
                    label,
                    data.len().max(4) as u64,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                );
                self.queue.write_buffer(&buf, 0, data);
                buf
            };
            let la_buf = mk_ro("mp_crown_la", bytemuck::cast_slice(lower_a));
            let ua_buf = mk_ro("mp_crown_ua", bytemuck::cast_slice(upper_a));
            let wm_buf = mk_ro("mp_crown_meta", bytemuck::cast_slice(window_meta));

            let out_usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let nla_buf = create_buffer(
                &self.device,
                "mp_crown_nla",
                (total.max(1) as u64) * f32_size,
                out_usage,
            );
            let nua_buf = create_buffer(
                &self.device,
                "mp_crown_nua",
                (total.max(1) as u64) * f32_size,
                out_usage,
            );
            let err_buf = create_buffer(
                &self.device,
                "mp_crown_err",
                (total.max(1) as u64) * 2 * f32_size,
                out_usage,
            );

            let pipe = &self.ibp_sound_pipelines().maxpool_crown;
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mp_crown_bg"),
                layout: &pipe.1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: la_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ua_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wm_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: nla_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: nua_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: err_buf.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mp_crown_enc"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mp_crown_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipe.0);
                pass.set_bind_group(0, &bind_group, &[]);
                let wg = gpu_checked_u32(total, "mp crown dispatch")?.div_ceil(64);
                pass.dispatch_workgroups(wg.max(1), 1, 1);
            }

            let mk_stg = |label: &'static str, len: usize| {
                create_buffer(
                    &self.device,
                    label,
                    (len.max(1) as u64) * f32_size,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                )
            };
            let stg_nla = mk_stg("mp_stg_nla", total);
            let stg_nua = mk_stg("mp_stg_nua", total);
            let stg_err = mk_stg("mp_stg_err", 2 * total);
            encoder.copy_buffer_to_buffer(&nla_buf, 0, &stg_nla, 0, (total as u64) * f32_size);
            encoder.copy_buffer_to_buffer(&nua_buf, 0, &stg_nua, 0, (total as u64) * f32_size);
            encoder.copy_buffer_to_buffer(&err_buf, 0, &stg_err, 0, (2 * total as u64) * f32_size);
            self.queue.submit(std::iter::once(encoder.finish()));

            let mut out = WgpuDevice::read_buffers_batched(
                &self.device,
                &[(&stg_nla, total), (&stg_nua, total), (&stg_err, 2 * total)],
            )?;
            let err = out.pop().expect("3 readbacks");
            let nua = out.pop().expect("3 readbacks");
            let nla = out.pop().expect("3 readbacks");
            Ok((nla, nua, err))
        })
    }
}

#[cfg(test)]
mod directed_bias_tests {
    use ny_core::{f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up};

    use super::{
        accumulate_lower_bias_outward, accumulate_upper_bias_outward, max_f32_by_exact_lift,
    };

    #[test]
    fn directed_bias_accumulation_preserves_subnormal_terms_lost_by_nearest_add() {
        let positive_tiny = f32::from_bits(1);
        let negative_tiny = f32::from_bits(0x8000_0001);
        assert!(f32_to_f64_exact(positive_tiny) > 0.0);
        assert!(f32_to_f64_exact(negative_tiny) < 0.0);

        let lower = accumulate_lower_bias_outward(1.0, negative_tiny, 1.0);
        let upper = accumulate_upper_bias_outward(-1.0, positive_tiny, 1.0);
        assert!(lower < 1.0, "lower step must not lose a negative term");
        assert!(upper > -1.0, "upper step must not lose a positive term");
        assert!(f64_to_f32_down(lower) <= 1.0);
        assert!(f64_to_f32_up(upper) >= -1.0);
    }

    #[test]
    fn directed_publication_never_emits_subnormal_endpoints() {
        let tiny = f32_to_f64_exact(f32::from_bits(1));
        assert_eq!(f64_to_f32_down(tiny).to_bits(), 0);
        assert_eq!(f64_to_f32_up(tiny), f32::MIN_POSITIVE);
        assert_eq!(f64_to_f32_down(-tiny), -f32::MIN_POSITIVE);
        assert_eq!(f64_to_f32_up(-tiny).to_bits(), 0x8000_0000);
    }

    #[test]
    fn metadata_ordering_does_not_treat_negative_subnormal_as_zero() {
        let negative_tiny = f32::from_bits(0x8000_0001);
        assert_eq!(max_f32_by_exact_lift(negative_tiny, 0.0).to_bits(), 0);
        assert!(
            f32_to_f64_exact(negative_tiny) < f32_to_f64_exact(0.0),
            "definite-winner comparisons must preserve this ordering under DAZ"
        );
    }
}
