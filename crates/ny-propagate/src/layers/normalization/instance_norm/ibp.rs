// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Interval bound propagation for InstanceNorm1d.
//!
//! InstanceNorm1d: y[c, t] = ny[c] * (x[c, t] - mean_c) / sqrt(var_c + eps) + beta[c]
//!
//! where mean_c and var_c are computed over the time dimension T for each channel c.
//!
//! IBP strategy (per channel):
//! 1. Bound mean(x[c,:]) via interval arithmetic on [l[c,t], u[c,t]] per timestep
//! 2. Bound var from interval on centered values
//! 3. Bound std = sqrt(var + eps) (monotone sqrt)
//! 4. For y[c,t] = ny[c] * (x[c,t] - mean_c) / std_c + beta[c]:
//!    bound using sign analysis and interval division
//!
//! Conservative strategy: treat each channel as an independent LayerNorm over T.

use std::borrow::Cow;

use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_dim_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::super::math_common::{
    compute_batch_prefix, outward_midpoint_radius, square_interval_bounds,
};

use super::types::InstanceNorm1dLayer;
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

impl InstanceNorm1dLayer {
    /// Fallback output bounds when input bounds are non-finite.
    ///
    /// Uses the fact that after normalization, the maximum z-value for any
    /// element in a T-dimensional vector is bounded by sqrt(T-1) (when one
    /// element is the outlier and all others are identical).
    fn fallback_output_bounds(&self, shape: &[usize]) -> Result<BoundedTensor> {
        let ndim = shape.len();
        if ndim < 2 {
            return Err(NyError::InvalidSpec(
                "InstanceNorm1d requires at least 2D input [C, T]".to_string(),
            ));
        }

        let num_channels = shape[ndim - 2];
        let time_len = shape[ndim - 1];

        if self.num_channels() != num_channels {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.num_channels()],
                got: vec![num_channels],
            });
        }

        if time_len == 0 {
            return BoundedTensor::new(ArrayD::zeros(IxDyn(shape)), ArrayD::zeros(IxDyn(shape)));
        }

        // Max |z| = sqrt(T-1), rounded UP for soundness (#3344).
        let z_max = if time_len <= 1 {
            0.0
        } else {
            next_up_f32(((time_len as f32) - 1.0).sqrt())
        };

        let mut out_lower = ArrayD::<f32>::zeros(IxDyn(shape));
        let mut out_upper = ArrayD::<f32>::zeros(IxDyn(shape));

        // Fill each channel with its ny/beta-scaled bounds
        for c in 0..num_channels {
            let g = self.ny[c];
            let b = self.beta[c];

            // Directed rounding on fallback bounds (#3344).
            let (ch_lower, ch_upper) = if !g.is_finite() || !b.is_finite() {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else if g >= 0.0 {
                (next_down_f32(b + g * (-z_max)), next_up_f32(b + g * z_max))
            } else {
                (next_down_f32(b + g * z_max), next_up_f32(b + g * (-z_max)))
            };

            // Fill all time positions for this channel across all batch dims
            fill_channel_bounds(&mut out_lower, &mut out_upper, shape, c, ch_lower, ch_upper)?;
        }

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// Forward-mode IBP for InstanceNorm1d using Jacobian-based propagation.
    ///
    /// Computes the Jacobian at the center point (midpoint of bounds), then
    /// propagates input radii through the absolute Jacobian and adds a
    /// second-order remainder term for curvature.
    ///
    /// This implementation is admitted only as heuristic analysis, not as
    /// proof authority. The outward arithmetic below hardens its enclosure
    /// behavior but does not change that provenance classification.
    ///
    /// ## Math (per-channel, time dimension T)
    ///
    /// `y[c,t] = ny[c] * (x[c,t] - mean_c) / std_c + beta[c]`
    ///
    /// Jacobian at center `c` with `mean_c`, `std_c`, `z_t = (c_t - mean_c)/std_c`:
    /// ```text
    /// J[s,t] = ny[c] / std_c * (delta_st - 1/T - z_s * z_t / T)
    /// ```
    ///
    /// Same formula as LayerNorm Jacobian but with a single ny per channel.
    ///
    /// Reference: Jacobian from `math.rs` (verified against finite differences).
    /// Fix for #3098: replaces the unsound `max_radius / T` coupling correction.
    fn propagate_ibp_forward_mode(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        let num_channels = shape[ndim - 2];
        let time_len = shape[ndim - 1];

        let Some((center, radius)) = outward_midpoint_radius(input.lower(), input.upper()) else {
            return self.fallback_output_bounds(shape);
        };

        let mut out_lower = input.lower().clone();
        let mut out_upper = input.upper().clone();

        let batch_size: usize =
            checked_dim_product(&shape[..ndim - 2], "InstanceNorm IBP batch dimensions")?;
        let nf = time_len as f32;

        for batch_idx in 0..batch_size.max(1) {
            let batch_prefix = compute_batch_prefix(shape, ndim, batch_idx);

            for c in 0..num_channels {
                // Collect center and radius for this channel
                let mut center_vals = Vec::with_capacity(time_len);
                let mut radius_vals = Vec::with_capacity(time_len);
                for t in 0..time_len {
                    let mut idx = batch_prefix.clone();
                    idx.push(c);
                    idx.push(t);
                    center_vals.push(center[idx.as_slice()]);
                    radius_vals.push(radius[idx.as_slice()]);
                }

                self.forward_mode_channel_1d_slice(
                    &center_vals,
                    &radius_vals,
                    c,
                    nf,
                    &mut |t, lo, hi| {
                        let mut idx = batch_prefix.clone();
                        idx.push(c);
                        idx.push(t);
                        out_lower[idx.as_slice()] = lo;
                        out_upper[idx.as_slice()] = hi;
                    },
                );
            }
        }

        // Post-check: reject NaN in forward-mode output bounds.
        // Consistent with the conservative path (line ~398) which also rejects NaN.
        // NaN can arise if std_c computation produces pathological values despite
        // the finite center guard above.
        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "InstanceNorm1d forward-mode IBP: NaN in computed bounds".to_string(),
            ));
        }

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// Core forward-mode computation for a single channel's time dimension.
    ///
    /// Computes Jacobian-based output bounds for one channel's 1D time slice.
    /// Calls `emit(t, lower_t, upper_t)` for each output time position.
    ///
    /// InstanceNorm per channel: `y_t = ny[c] * (x_t - mean) / std + beta[c]`
    ///
    /// Jacobian at center:
    /// ```text
    /// J[s,t] = ny[c] / std_c * (delta_st - 1/T - z_s * z_t / T)
    /// ```
    /// where `z_t = (c_t - mean_c) / std_c`.
    ///
    /// Fix for #3098: replaces the unsound `max_radius / T` coupling correction
    /// with exact Jacobian propagation plus a conservative second-order remainder.
    #[inline]
    fn forward_mode_channel_1d_slice(
        &self,
        center: &[f32],
        radius: &[f32],
        channel: usize,
        nf: f32,
        emit: &mut impl FnMut(usize, f32, f32),
    ) {
        let time_len = center.len();
        let nf_64 = time_len as f64;

        let g = self.ny[channel];
        let b = self.beta[channel];

        // Compute mean and std at center using f64 accumulation.
        // f32 accumulation loses ~log2(T) bits of precision for T=768+.
        // Ported from LayerNorm #2423.
        let mean_c_f64: f64 = center.iter().map(|&x| x as f64).sum::<f64>() / nf_64;
        let mean_c = mean_c_f64 as f32;
        let var_c: f32 = (center
            .iter()
            .map(|&xi| {
                let d = xi as f64 - mean_c_f64;
                d * d
            })
            .sum::<f64>()
            / nf_64) as f32;
        let std_c = (var_c + self.eps).sqrt();

        // Normalized values at center, computed in f64 to prevent cancellation
        // in z_s * z_t terms used by the Jacobian. Ported from LayerNorm #2423.
        let std_c_f64 = std_c as f64;
        let z_f64: Vec<f64> = center
            .iter()
            .map(|&ci| (ci as f64 - mean_c_f64) / std_c_f64)
            .collect();
        let z: Vec<f32> = z_f64.iter().map(|&zi| zi as f32).collect();

        // Precompute: sum of radius^2 for the second-order remainder (f64 accumulation).
        // next_up_f32: radius_sq_sum is in the NUMERATOR of the second-order bound
        // R₂ = 3.5·|γ|·||r||²/(√T·σ²). Rounding up ensures R₂ is an upper bound.
        // Ported from LayerNorm #3270.
        let radius_sq_sum: f32 =
            next_up_f32((radius.iter().map(|&r| (r as f64) * (r as f64)).sum::<f64>()) as f32);

        // Compute minimum std over the input box using interval arithmetic.
        // Same approach as LayerNorm: bound the minimum variance achievable
        // within the input box to get a sound denominator for the Hessian bound.
        // next_up_f32: mean_radius widens the centered intervals [a_lower, a_upper].
        // Rounding UP makes intervals wider → var_lower smaller → std_min smaller →
        // second-order remainder larger (conservative, sound).
        // Ported from LayerNorm #3270.
        let mean_radius: f32 =
            next_up_f32((radius.iter().map(|&r| r as f64).sum::<f64>() / nf_64) as f32);
        let mut var_lower_f64: f64 = 0.0;
        for t in 0..time_len {
            let a_c = center[t] - mean_c;
            let a_lower = a_c - radius[t] - mean_radius;
            let a_upper = a_c + radius[t] + mean_radius;
            let sq_lower = if a_lower > 0.0 {
                (a_lower as f64) * (a_lower as f64)
            } else if a_upper < 0.0 {
                (a_upper as f64) * (a_upper as f64)
            } else {
                0.0 // interval straddles zero
            };
            var_lower_f64 += sq_lower;
        }
        // Directed rounding: var_lower is a lower bound on variance → round down.
        // Smaller variance → smaller std denominator → tighter (unsound) bounds
        // if not rounded conservatively. Ported from LayerNorm #3270.
        let var_lower = next_down_f32((var_lower_f64 / nf_64) as f32);
        // Directed rounding on sqrt: std_min is a denominator → round DOWN
        // (smaller denominator = larger R₂ = sound). f64 intermediate for
        // precision. Pattern from GroupNorm #3327, fix for #3332.
        let std_min = next_down_f32(((var_lower as f64 + self.eps as f64).sqrt()) as f32);

        // Sound affine-envelope ceiling for the forward-mode (Jacobian + 2nd-order)
        // output interval. InstanceNorm1d is mean-subtract (centered) over the time axis:
        // |z_t| <= sqrt(T-1) in exact reals, and the downstream f32 program overshoots by
        // at most
        //   delta = (T+2)*ulp(max_abs_x)/std_min + base*EPSILON + 0.5*ulp(sqrt(T-1)).
        // A spec-compliant naive f32 per-channel mean is a sequential sum of T terms
        // of magnitude ~M=max_abs_x, accumulating ~(T-1)*0.5*ulp(T*M) rounding ->
        // ~0.5*(T-1)*ulp(M) after /T, plus the (x - mean) subtraction (~ulp(M)), so
        // the numerator error is ~0.5*(T+1)*ulp(M): a bare ulp(max_abs_x) (no T
        // factor) UNDER-counts it and is UNSOUND. The (T+2) factor dominates;
        // base*EPSILON covers the final divide's relative error at |z|<=base. The
        // term is amplified by the smallest sound std (std_min rounded DOWN so
        // 1/std_min over-estimates 1/std). All steps round OUTWARD, so
        // max_norm_safe >= the true reachable |z| ceiling.
        //
        // The Jacobian first-order + 2nd-order remainder interval is a sound enclosure,
        // and the affine envelope [g*(-Z)+b, g*(+Z)+b] (Z = max_norm_safe, y = g*z + b
        // affine in z) is ALSO a sound enclosure. Intersecting two sound enclosures is
        // still sound and never looser, so it tightens without excluding a reachable
        // output. max_abs_x over the box = max_t(|center_t| + radius_t).
        let max_norm_safe = if time_len > 1 {
            let max_abs_x = center
                .iter()
                .zip(radius.iter())
                .map(|(&c, &r)| c.abs() + r.abs())
                .fold(0.0_f32, f32::max);
            let base = (((time_len as f64) - 1.0).sqrt()) as f32;
            if max_abs_x.is_finite() && std_min > 0.0 && std_min.is_finite() {
                let ulp_x = next_up_f32(max_abs_x) - max_abs_x;
                let ulp_b = next_up_f32(base) - base;
                // (n+2) ULP(max|x|): a spec-compliant naive f32 per-channel mean
                // is a sequential sum of n=time_len terms of magnitude ~M,
                // accumulating ~(n-1)*0.5*ulp(n*M) error; after /n that is
                // ~0.5*(n-1)*ulp(M), plus the (x_t - mean) subtraction (~ulp(M)).
                // A bare ulp(max|x|) (no n factor) UNDER-counts this and is
                // unsound; (n+2) dominates the (n-1)-term summation + subtraction.
                // base*EPSILON covers the final divide's relative error at
                // |z|<=base; 0.5*ulp(base) the sqrt rounding.
                let delta =
                    next_up_f32((nf + 2.0) * ulp_x / std_min + base * f32::EPSILON + 0.5 * ulp_b);
                next_up_f32(base + delta)
            } else {
                f32::INFINITY
            }
        } else {
            // T == 1: z is identically 0; envelope collapses to {b}.
            0.0
        };

        let g_f64 = g as f64;
        for s in 0..time_len {
            // Center-point output
            let y_center = g * z[s] + b;

            // First-order: output_radius_s = sum_t |J[s,t]| * radius_t
            // J[s,t] = ny[c] / std_c * (delta_st - 1/T - z_s * z_t / T)
            // f64 for both individual Jacobian entries and the sum to prevent
            // cancellation in (delta_st - 1/T - z_s*z_t/T). Ported from LayerNorm #2423.
            let mut first_order_radius_f64 = 0.0_f64;
            for t in 0..time_len {
                let delta_st: f64 = if s == t { 1.0 } else { 0.0 };
                let j_st_f64 =
                    g_f64 / std_c_f64 * (delta_st - 1.0 / nf_64 - z_f64[s] * z_f64[t] / nf_64);
                first_order_radius_f64 += j_st_f64.abs() * radius[t] as f64;
            }
            // Directed rounding: radius widens output bounds → round up. #3270.
            let first_order_radius = next_up_f32(first_order_radius_f64 as f32);

            // Second-order remainder: R₂(s) ≤ 7|γ|·||r||² / (2√T·σ_min²)
            //
            // InstanceNorm per-channel Hessian has the same structure as
            // LayerNorm (mean-subtraction normalization), so the same bound
            // applies with n=T (time dimension). See LayerNorm
            // forward_mode_standard_1d_slice for full derivation.
            // Guard: Inf σ_min² → denominator Inf → x/Inf=0 (unsound).
            // See LayerNorm forward_mode_standard_1d_slice for full rationale.
            let second_order_denom = nf.sqrt() * std_min * std_min;
            let second_order = if second_order_denom.is_finite() && second_order_denom > 0.0 {
                3.5 * g.abs() * radius_sq_sum / second_order_denom
            } else {
                f32::INFINITY
            };

            let output_radius = first_order_radius + second_order;

            // Directed rounding on final emit: y_center has f64→f32 rounding
            // error up to 0.5 ULP, and f32 subtraction/addition adds another
            // 0.5 ULP. next_down/next_up compensates both. Part of #3344.
            let mut jac_lower = next_down_f32(y_center - output_radius);
            let mut jac_upper = next_up_f32(y_center + output_radius);

            // Intersect with the sign-aware affine envelope [g*(-Z)+b, g*(+Z)+b],
            // Z = max_norm_safe. y = g*z + b is affine in z; for g >= 0 the endpoints are
            // (b - g*Z, b + g*Z), for g < 0 they swap. Endpoints rounded OUTWARD so the
            // envelope stays a sound enclosure; NaN-propagating max/min so a NaN Jacobian
            // bound survives intersection and fires downstream guards.
            if max_norm_safe.is_finite() {
                let (env_lower, env_upper) = if g >= 0.0 {
                    (
                        next_down_f32(b - g * max_norm_safe),
                        next_up_f32(b + g * max_norm_safe),
                    )
                } else {
                    (
                        next_down_f32(b + g * max_norm_safe),
                        next_up_f32(b - g * max_norm_safe),
                    )
                };
                jac_lower = nan_propagating_max(jac_lower, env_lower);
                jac_upper = nan_propagating_min(jac_upper, env_upper);
            }

            emit(s, jac_lower, jac_upper);
        }
    }
}

impl InstanceNorm1dLayer {
    /// Conservative IBP for one channel within a batch position.
    ///
    /// Bounds mean, variance, std via interval arithmetic, then bounds each
    /// output position y[t] = ny * (x[t] - mean) / std + beta.
    fn ibp_conservative_channel(
        &self,
        x_lowers: &[f32],
        x_uppers: &[f32],
        channel: usize,
        out_lower: &mut [f32],
        out_upper: &mut [f32],
    ) {
        let time_len = x_lowers.len();
        let nf_64 = time_len as f64;

        // Step 1: Bound mean using f64 accumulation with directed rounding.
        // f32 accumulation loses ~log2(T) bits of precision for T=768+.
        // Directed rounding: mean_lower rounds down (next_down_f32), mean_upper
        // rounds up (next_up_f32) so the interval encloses the true mean.
        // Ported from LayerNorm #2423/#3270.
        let mean_lower: f32 =
            next_down_f32((x_lowers.iter().map(|&x| x as f64).sum::<f64>() / nf_64) as f32);
        let mean_upper: f32 =
            next_up_f32((x_uppers.iter().map(|&x| x as f64).sum::<f64>() / nf_64) as f32);

        // Mean overflow guard: if sum overflows to Inf (many large finite inputs),
        // fall back to ny-scaled sqrt(T-1) bounds for this channel. Matches
        // LayerNorm's has_nonfinite_mean → fallback_output_bounds pattern.
        if !mean_lower.is_finite() || !mean_upper.is_finite() {
            let g = self.ny[channel];
            let b = self.beta[channel];
            let z_max = if time_len > 1 {
                // #3344: round UP
                next_up_f32(((time_len as f32) - 1.0).sqrt())
            } else {
                0.0
            };
            let (ch_lower, ch_upper) = if !g.is_finite() || !b.is_finite() {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else if g >= 0.0 {
                (next_down_f32(b + g * (-z_max)), next_up_f32(b + g * z_max))
            } else {
                (next_down_f32(b + g * z_max), next_up_f32(b + g * (-z_max)))
            };
            for t in 0..time_len {
                out_lower[t] = ch_lower;
                out_upper[t] = ch_upper;
            }
            return;
        }

        // Step 2-3: Bound variance and std using f64 accumulation.
        // Catastrophic cancellation in (xi - mean)^2 when xi ≈ mean is the
        // primary precision concern. Ported from LayerNorm #2423.
        let mut sq_sum_lower_f64 = 0.0_f64;
        let mut sq_sum_upper_f64 = 0.0_f64;
        for t in 0..time_len {
            // Directed rounding on intermediate deviation: feeds variance via
            // square_interval_bounds — too-narrow domain → too-narrow variance → unsound.
            // Part of #3344.
            let centered_lower = next_down_f32(x_lowers[t] - mean_upper);
            let centered_upper = next_up_f32(x_uppers[t] - mean_lower);
            let (sq_l, sq_u) = square_interval_bounds(centered_lower, centered_upper);
            sq_sum_lower_f64 += sq_l as f64;
            sq_sum_upper_f64 += sq_u as f64;
        }

        // Directed rounding: var_lower is denominator bound → round down,
        // var_upper is range bound → round up. Ported from LayerNorm #3270.
        let var_lower = next_down_f32((sq_sum_lower_f64 / nf_64) as f32);
        let var_upper = next_up_f32((sq_sum_upper_f64 / nf_64) as f32);
        // Directed rounding on sqrt: std_lower is denominator → round DOWN,
        // std_upper is range bound → round UP (wider interval = sound).
        // f64 intermediate for precision. Pattern from GroupNorm #3327, fix for #3332.
        let std_lower = next_down_f32(((var_lower as f64 + self.eps as f64).sqrt()) as f32);
        let std_upper = next_up_f32(((var_upper as f64 + self.eps as f64).sqrt()) as f32);

        let g = self.ny[channel];
        let b = self.beta[channel];

        // Theoretical bound: normalized values are in [-sqrt(T-1), sqrt(T-1)]
        // because sum of centered-normalized deviations = 0. The interval arithmetic
        // over-approximation (treating x_t, mean, std as independent) can exceed this.
        // Max |z| = sqrt(T-1) for clamping (#3196), rounded UP (#3344).
        //
        // Sound f32 float-error margin for the sqrt(T-1) z-clamp.
        //
        // InstanceNorm1d is a MEAN-SUBTRACT (centered) norm over the time axis:
        // z_t = (x_t - mean)/std with sum_t (x_t - mean) = 0, so in exact reals the
        // tight Cauchy-Schwarz ceiling is |z_t| <= sqrt(T-1). But the downstream f32
        // inference program the verifier must cover evaluates z in finite precision;
        // catastrophic cancellation in fl(x_t - mean) for large |x_t| can push the
        // realized |z_t| slightly above sqrt(T-1). Clamping at next_up(sqrt(T-1))
        // (~1 ULP above the real bound) could then chop a reachable output => UNSOUND.
        //
        // Worst-case f32 error of z_t = fl(fl(x_t - mean)/std): a spec-compliant
        // naive f32 mean is a sequential sum of T terms of magnitude ~M=max|x|,
        // accumulating ~(T-1)*0.5*ulp(T*M) error; after /T that is ~0.5*(T-1)*ulp(M),
        // and the (x_t - mean) subtraction adds another ~ulp(M). So the numerator
        // absolute error is ~0.5*(T+1)*ulp(M), NOT ulp(M) — a bare ulp(max|x|) (no
        // T factor) UNDER-counts it and is UNSOUND (validated counterexample at
        // T=10 realized 0.14375 vs budget ulp(max)=0.0625). The (T+2) factor below
        // dominates this. Amplified by 1/std_lower (std_lower rounded DOWN, so
        // 1/std_lower over-estimates 1/std). Hence
        //   delta = (T+2)*ulp(max_abs_x)/std_lower + base*EPSILON + 0.5*ulp(sqrt(T-1)),
        // where base*EPSILON covers the final divide's relative error at |z|<=base.
        // and |z_t| <= sqrt(T-1) + delta over the reachable f32 set. The mean-subtract
        // (T+2)*ulp/std numerator term is required for centered norms (RMSNorm,
        // lacking numerator cancellation, uses a denominator-relative margin instead).
        //
        // Every step rounds strictly OUTWARD (std_lower DOWN; ulp_x, ulp_b >= true ULPs;
        // sum next_up'd; base+delta next_up'd), so max_norm >= the true f32 ceiling and
        // the clamp [-max_norm, max_norm] is a sound SUPERSET of the reachable set,
        // still tighter than the unclamped 4-corner interval in the high-magnitude /
        // low-variance regime.
        //
        // Non-finite guard: if max_abs_x or std_lower is not finite/positive, fall through
        // to max_norm = +INF so the unclamped interval is kept and repaired, never tightened.
        let max_norm = if time_len > 1 {
            let max_abs_x = x_lowers
                .iter()
                .zip(x_uppers.iter())
                .map(|(&l, &u)| l.abs().max(u.abs()))
                .fold(0.0_f32, f32::max);
            let base = (((time_len as f64) - 1.0).sqrt()) as f32;
            if max_abs_x.is_finite() && std_lower > 0.0 && std_lower.is_finite() {
                let ulp_x = next_up_f32(max_abs_x) - max_abs_x;
                let ulp_b = next_up_f32(base) - base;
                // (n+2) ULP(max|x|): a spec-compliant naive f32 per-channel mean
                // is a sequential sum of n=time_len terms of magnitude ~M,
                // accumulating ~(n-1)*0.5*ulp(n*M) error; after /n that is
                // ~0.5*(n-1)*ulp(M), plus the (x_t - mean) subtraction (~ulp(M)).
                // A bare ulp(max|x|) (no n factor) UNDER-counts this and is
                // unsound; (n+2) dominates the (n-1)-term summation + subtraction.
                // base*EPSILON covers the final divide's relative error at
                // |z|<=base; 0.5*ulp(base) the sqrt rounding.
                let delta = next_up_f32(
                    (time_len as f32 + 2.0) * ulp_x / std_lower + base * f32::EPSILON + 0.5 * ulp_b,
                );
                next_up_f32(base + delta)
            } else {
                f32::INFINITY
            }
        } else {
            0.0
        };

        // Step 4: Bound y[t] = g * (x[t] - mean) / std + b
        for t in 0..time_len {
            // Directed rounding on intermediate deviation: if centered_lower
            // rounds UP, the 4-corner domain is too narrow → unsound. Part of #3344.
            let centered_lower = next_down_f32(x_lowers[t] - mean_upper);
            let centered_upper = next_up_f32(x_uppers[t] - mean_lower);

            let corners = [
                centered_lower / std_upper,
                centered_lower / std_lower,
                centered_upper / std_upper,
                centered_upper / std_lower,
            ];

            let ratio_l = corners
                .iter()
                .fold(f32::INFINITY, |a, &b| nan_propagating_min(a, b));
            let ratio_u = corners
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| nan_propagating_max(a, b));

            // Clamp to [-sqrt(T-1), sqrt(T-1)]. NaN must propagate through the clamp
            // so downstream guards fire correctly.
            let ratio_l = nan_propagating_max(ratio_l, -max_norm);
            let ratio_u = nan_propagating_min(ratio_u, max_norm);

            // Directed rounding on final bounds: bare f32 mul+add from
            // corner division results. Part of #3344.
            let (y_l, y_u) = if g >= 0.0 {
                (next_down_f32(g * ratio_l + b), next_up_f32(g * ratio_u + b))
            } else {
                (next_down_f32(g * ratio_u + b), next_up_f32(g * ratio_l + b))
            };

            out_lower[t] = y_l;
            out_upper[t] = y_u;
        }
    }
}

impl InstanceNorm1dLayer {
    /// Validate IBP input and return (num_channels, time_len).
    fn validate_ibp_input(&self, input: &BoundedTensor) -> Result<(usize, usize)> {
        let shape = input.shape();
        let ndim = shape.len();

        if ndim < 2 {
            return Err(NyError::InvalidSpec(
                "InstanceNorm1d requires at least 2D input [C, T]".to_string(),
            ));
        }
        if shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "InstanceNorm1d: zero-valued dimension in input shape".to_string(),
            ));
        }
        // Reject non-finite input bounds (Category B per domain validation policy).
        if input.lower().iter().any(|x| !x.is_finite())
            || input.upper().iter().any(|x| !x.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "InstanceNorm1d IBP: non-finite input bounds".to_string(),
            ));
        }

        let num_channels = shape[ndim - 2];
        let time_len = shape[ndim - 1];

        if time_len > (1 << 24) {
            return Err(NyError::InternalError(format!(
                "InstanceNorm1d time dimension {time_len} exceeds f32 exact integer range"
            )));
        }
        if self.num_channels() != num_channels {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.num_channels()],
                got: vec![num_channels],
            });
        }

        Ok((num_channels, time_len))
    }

    /// Conservative IBP over all batch positions and channels.
    fn propagate_ibp_conservative(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        let (num_channels, time_len) = self.validate_ibp_input(input)?;

        let mut out_lower = input.lower().clone();
        let mut out_upper = input.upper().clone();
        let batch_size: usize =
            checked_dim_product(&shape[..ndim - 2], "InstanceNorm IBP batch dimensions")?;

        for batch_idx in 0..batch_size.max(1) {
            let batch_prefix = compute_batch_prefix(shape, ndim, batch_idx);

            for c in 0..num_channels {
                let mut x_lowers = Vec::with_capacity(time_len);
                let mut x_uppers = Vec::with_capacity(time_len);
                for t in 0..time_len {
                    let mut idx = batch_prefix.clone();
                    idx.push(c);
                    idx.push(t);
                    x_lowers.push(input.lower()[idx.as_slice()]);
                    x_uppers.push(input.upper()[idx.as_slice()]);
                }

                let mut ch_lower = vec![0.0_f32; time_len];
                let mut ch_upper = vec![0.0_f32; time_len];
                self.ibp_conservative_channel(
                    &x_lowers,
                    &x_uppers,
                    c,
                    &mut ch_lower,
                    &mut ch_upper,
                );

                for t in 0..time_len {
                    let mut idx = batch_prefix.clone();
                    idx.push(c);
                    idx.push(t);
                    out_lower[idx.as_slice()] = ch_lower[t];
                    out_upper[idx.as_slice()] = ch_upper[t];
                }
            }
        }

        // Post-check: reject NaN
        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "InstanceNorm1d IBP: NaN in computed bounds".to_string(),
            ));
        }

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}

impl BoundPropagation for InstanceNorm1dLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.validate_ibp_input(input)?;

        if self.forward_mode {
            self.propagate_ibp_forward_mode(input)
        } else {
            self.propagate_ibp_conservative(input)
        }
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "InstanceNorm1d is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        InstanceNorm1dLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Fill channel bounds across all batch dimensions and time positions.
fn fill_channel_bounds(
    out_lower: &mut ArrayD<f32>,
    out_upper: &mut ArrayD<f32>,
    shape: &[usize],
    channel: usize,
    lower_val: f32,
    upper_val: f32,
) -> Result<()> {
    let ndim = shape.len();
    let time_len = shape[ndim - 1];
    let batch_size: usize =
        checked_dim_product(&shape[..ndim - 2], "InstanceNorm IBP batch dimensions")?;

    for batch_idx in 0..batch_size.max(1) {
        let batch_prefix = compute_batch_prefix(shape, ndim, batch_idx);
        for t in 0..time_len {
            let mut idx = batch_prefix.clone();
            idx.push(channel);
            idx.push(t);
            out_lower[idx.as_slice()] = lower_val;
            out_upper[idx.as_slice()] = upper_val;
        }
    }
    Ok(())
}
