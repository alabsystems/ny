// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Interval bound propagation for RMSNorm.
//!
//! RMSNorm: y_i = ny_i * x_i / rms, where rms = sqrt(mean(x^2) + eps)
//!
//! IBP strategy:
//! 1. Bound mean(x^2) via interval arithmetic on [l_i^2, u_i^2] per coordinate
//!    (accounting for 0-crossings: if l_i <= 0 <= u_i, then x_i^2 lower bound is 0)
//! 2. Bound rms from interval on mean(x^2) + eps (monotone sqrt)
//! 3. For each y_i = ny_i * x_i / rms: bound using sign analysis of ny_i
//!    and the interval arithmetic of x_i / rms

use std::borrow::Cow;

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{checked_dim_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, L2Constraint, RepairStrategy};

use super::super::math_common::square_interval_bounds;

use super::types::RmsNormLayer;
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

impl RmsNormLayer {
    /// Fallback output bounds when input bounds are non-finite.
    ///
    /// Uses the fact that RMSNorm output magnitude is bounded by
    /// |ny_i| * sqrt(n) (the maximum possible |z_i| for unit-RMS vector).
    fn fallback_output_bounds(&self, shape: &[usize]) -> Result<BoundedTensor> {
        let ndim = shape.len();
        if ndim == 0 {
            return Err(NyError::InvalidSpec(
                "RMSNorm requires at least 1D input".to_string(),
            ));
        }

        let norm_size = shape[ndim - 1];
        if self.ny.len() != norm_size {
            return Err(NyError::ShapeMismatch {
                expected: vec![norm_size],
                got: vec![self.ny.len()],
            });
        }

        if norm_size == 0 {
            return BoundedTensor::new(ArrayD::zeros(IxDyn(shape)), ArrayD::zeros(IxDyn(shape)));
        }

        let z_max = next_up_f32((norm_size as f32).sqrt()); // #3344: round UP
        let (per_dim_lower, per_dim_upper) = self.fallback_per_dim_bounds(norm_size, z_max);

        let mut out_lower = ArrayD::<f32>::zeros(IxDyn(shape));
        let mut out_upper = ArrayD::<f32>::zeros(IxDyn(shape));
        for mut lane in out_lower.lanes_mut(ndarray::Axis(ndim - 1)) {
            lane.assign(&per_dim_lower);
        }
        for mut lane in out_upper.lanes_mut(ndarray::Axis(ndim - 1)) {
            lane.assign(&per_dim_upper);
        }

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// Per-dimension fallback bounds from ny and z_max = sqrt(n).
    fn fallback_per_dim_bounds(&self, norm_size: usize, z_max: f32) -> (Array1<f32>, Array1<f32>) {
        let mut per_dim_lower = Array1::<f32>::zeros(norm_size);
        let mut per_dim_upper = Array1::<f32>::zeros(norm_size);
        for i in 0..norm_size {
            let g = self.ny[i];
            if !g.is_finite() {
                per_dim_lower.fill(f32::NEG_INFINITY);
                per_dim_upper.fill(f32::INFINITY);
                break;
            }
            if g >= 0.0 {
                per_dim_lower[i] = next_down_f32(-g * z_max); // #3344
                per_dim_upper[i] = next_up_f32(g * z_max);
            } else {
                per_dim_lower[i] = next_down_f32(g * z_max);
                per_dim_upper[i] = next_up_f32(-g * z_max);
            }
        }
        (per_dim_lower, per_dim_upper)
    }

    /// Build the per-slice L2 (Euclidean-ball) annotation for the RMSNorm output.
    ///
    /// THE LEVER. RMSNorm output `z_i = x_i / rms` (rms = sqrt(mean(x²) + eps))
    /// satisfies the EXACT joint bound, in real arithmetic:
    ///   Σ_i z_i² = Σ x_i² / (mean(x²) + eps)
    ///           = n·mean(x²) / (mean(x²) + eps)  ≤  n   (eps > 0 ⇒ strict),
    /// so ‖z‖₂ ≤ √n — a SPHERE. After the per-coordinate gain g_i the output is
    /// y_i = g_i·z_i, and
    ///   ‖y‖₂² = Σ g_i²·z_i² ≤ (max_i g_i²)·Σ z_i² ≤ (max_i g_i²)·n,
    /// hence ‖y‖₂ ≤ (max_i|g_i|)·√n. The ball is centred at the ORIGIN (RMSNorm
    /// has no bias; b = 0), which is the natural centre of the symmetric output.
    ///
    /// FLOAT MARGIN (outward). A spec-compliant naive f32 rms accumulates an
    /// n-term sum-of-squares with relative error ≲ 0.5·n·EPSILON, so the realized
    /// fl(rms) ≥ rms_true·(1 − δ), δ ≈ 0.5·n·EPSILON. Then each realized
    /// (x_i/fl(rms))² ≤ (x_i/rms_true)²/(1−δ)², so Σ ≤ n/(1−δ)² and
    /// ‖z‖₂ ≤ √n/(1−δ) ≤ √n·(1 + 2δ). We fold in a generous RELATIVE margin
    /// `rel = (n + 4)·EPSILON ≥ 2δ` (also covering the per-coordinate divide, the
    /// squaring, and the final √· rounding) and round every step OUTWARD via
    /// `next_up_f32`. The resulting radius is therefore a PROVEN upper bound on
    /// the true ‖y_slice − 0‖₂ over the whole input box.
    ///
    /// Returns `None` (drop the annotation — sound, just no tightening) for
    /// rank-0, an empty / huge norm axis, or a non-finite gain.
    fn compute_l2_constraint(&self, shape: &[usize]) -> Option<L2Constraint> {
        // THE GATE: only attach the sphere in a top-level plain IBP pass. Inside
        // iterative CROWN bound recomputation the gate is OFF, so we skip the
        // (per-pass) center allocation entirely — byte-identical to pre-lever and
        // sound (the box bound is unchanged). See `crate::l2_lever_gate`.
        if !crate::l2_lever_gate::l2_lever_active() {
            return None;
        }
        let ndim = shape.len();
        if ndim == 0 {
            return None;
        }
        let axis = ndim - 1;
        let norm_size = shape[axis];
        if norm_size == 0 || shape.contains(&0) {
            return None;
        }
        if self.ny.len() != norm_size {
            return None;
        }
        // max_i |g_i| (gain); bail on any non-finite gain.
        let mut max_abs_g = 0.0_f32;
        for &g in self.ny.iter() {
            if !g.is_finite() {
                return None;
            }
            max_abs_g = max_abs_g.max(g.abs());
        }

        // radius = max|g| · √n · (1 + rel), rel = (n + 4)·EPSILON, rounded OUTWARD.
        let nf = norm_size as f32;
        let sqrt_n = next_up_f32(nf.sqrt());
        let rel = next_up_f32((nf + 4.0) * f32::EPSILON);
        let one_plus_rel = next_up_f32(1.0 + rel);
        let radius = next_up_f32(next_up_f32(max_abs_g * sqrt_n) * one_plus_rel);
        if !radius.is_finite() {
            return None;
        }

        // Center at the origin (RMSNorm has no bias). Radius is per-slice;
        // shape = tensor shape with the last (norm) axis removed.
        let center = ArrayD::<f32>::zeros(IxDyn(shape));
        let radius_shape: Vec<usize> = shape[..axis].to_vec();
        let radius_arr = ArrayD::<f32>::from_elem(IxDyn(&radius_shape), radius);

        L2Constraint::new(center, radius_arr, axis, shape)
    }

    /// Forward-mode IBP for RMSNorm using Jacobian-based propagation.
    ///
    /// Computes the Jacobian at the center point, propagates input radii through
    /// the absolute Jacobian, then adds a second-order remainder for soundness.
    ///
    /// Reference: Jacobian from `math.rs` (verified against finite differences).
    /// Fix for #3098: replaces the unsound `max_radius / n` coupling correction.
    #[inline]
    fn propagate_ibp_forward_mode(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        if shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "RMSNorm: zero-valued dimension in input shape".to_string(),
            ));
        }
        let norm_size = shape[ndim - 1];
        let center = (input.lower() + input.upper()) * 0.5;
        let radius = (input.upper() - input.lower()) * 0.5;

        if center.iter().chain(radius.iter()).any(|&v| !v.is_finite()) {
            return self.fallback_output_bounds(shape);
        }

        let mut out_lower = input.lower().clone();
        let mut out_upper = input.upper().clone();
        let nf = norm_size as f32;

        for_each_norm_slice(shape, |prefix| {
            let prefix_len = prefix.len();
            let mut full_idx = [0usize; 8];
            full_idx[..prefix_len].copy_from_slice(prefix);
            let mut center_slice = Vec::with_capacity(norm_size);
            let mut radius_slice = Vec::with_capacity(norm_size);
            for i in 0..norm_size {
                full_idx[prefix_len] = i;
                let idx = &full_idx[..=prefix_len];
                center_slice.push(center[idx]);
                radius_slice.push(radius[idx]);
            }
            let prefix_snapshot: [usize; 8] = full_idx;
            self.forward_mode_1d_slice(&center_slice, &radius_slice, nf, &mut |i, lo, hi| {
                let mut emit_idx = prefix_snapshot;
                emit_idx[prefix_len] = i;
                out_lower[&emit_idx[..=prefix_len]] = lo;
                out_upper[&emit_idx[..=prefix_len]] = hi;
            });
        })?;

        // Post-check: reject NaN in forward-mode output bounds.
        // Consistent with InstanceNorm forward-mode and the conservative path.
        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "RmsNorm forward-mode IBP: NaN in computed bounds".to_string(),
            ));
        }

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    /// Core forward-mode computation for a single normalization slice.
    ///
    /// Computes Jacobian-based output bounds for one 1D slice of the input.
    /// Calls `emit(i, lower_i, upper_i)` for each output coordinate.
    #[inline]
    fn forward_mode_1d_slice(
        &self,
        center: &[f32],
        radius: &[f32],
        nf: f32,
        emit: &mut impl FnMut(usize, f32, f32),
    ) {
        let norm_size = center.len();
        // f64 accumulation + directed rounding (#3327, pattern from #3324)
        let nf_64 = norm_size as f64;

        // Sound z-saturation envelope for the forward (Jacobian) mode.
        //
        // The Jacobian-based output bounds [lo, hi] over-approximate the true
        // output via a 1st-order linearization + 2nd-order remainder; in the
        // high-magnitude / low-variance regime that remainder can blow [lo, hi]
        // well past the analytic ceiling. The true output is y_i = g_i * z_i
        // with the SAME exact bound |z_i| ≤ sqrt(n) proved in
        // `conservative_ibp_1d_slice` (RMSNorm uses raw x_i, no mean
        // cancellation, so the constant is sqrt(n) — NOT sqrt(n-1)).
        //
        // FLOAT MARGIN (delta > 0, denominator-relative — NOT the LayerNorm
        // numerator-cancellation term). RMSNorm has no (x - mean) cancellation in
        // the numerator (z_i = x_i / rms uses x_i directly), so there is NO
        // ulp(max|x|)/std term. But the denominator rms = sqrt(mean(x^2) + eps)
        // sums n squares; a spec-compliant naive f32 evaluation accumulates a
        // relative error ~0.5*n*EPSILON in that sum (and sqrt halves it again at
        // most), so the realized |z_i| = |x_i / fl(rms)| can exceed sqrt(n) by a
        // RELATIVE ~0.5*n*EPSILON, i.e. by ~base*0.5*n*EPSILON in absolute terms.
        // delta = base*(n*0.5 + 1)*EPSILON + 0.5*ulp(base): the n*0.5 term is the
        // sum-of-squares relative error, the +1 the final x_i/rms divide's
        // relative error at |z|<=base, and 0.5*ulp(base) the sqrt(n) rounding.
        //
        // So every reachable output lies in the affine envelope [g·(-Z), g·(+Z)]
        // (RMSNorm has no bias term, b = 0), Z = max_norm_safe. Intersecting the
        // Jacobian interval with this sound envelope can only TIGHTEN it and
        // never excludes a reachable point, so the result stays a sound superset.
        // Every step rounds OUTWARD (next_up into delta, next_up into Z); the
        // envelope endpoints below are directed-rounded OUTWARD (next_down for the
        // low end, next_up for the high end) and the intersection is NaN-propagating.
        let max_norm_safe = {
            let base = (norm_size as f32).sqrt();
            let ulp_b = next_up_f32(base) - base; // 1 ULP at sqrt(n)
            let delta =
                next_up_f32(base * ((norm_size as f32) * 0.5 + 1.0) * f32::EPSILON + 0.5 * ulp_b);
            next_up_f32(base + delta) // #3344
        };

        // Compute rms at center in f64
        let mean_sq_f64: f64 = center.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / nf_64;
        let rms_f64 = (mean_sq_f64 + self.eps as f64).sqrt();
        let rms_cubed_f64 = rms_f64 * rms_f64 * rms_f64;

        // next_up_f32: radius_sq_sum is in the NUMERATOR of second-order bound
        let radius_sq_sum: f32 =
            next_up_f32((radius.iter().map(|&r| (r as f64) * (r as f64)).sum::<f64>()) as f32);

        // Compute minimum rms over the input box using interval arithmetic.
        // rms = sqrt(mean(x²) + eps). The minimum rms occurs at the minimum
        // of mean(x²). For each x_i in [c_i - r_i, c_i + r_i], x_i² is
        // minimized at the point closest to 0 within the interval.
        let mut mean_sq_lower_f64: f64 = 0.0;
        for i in 0..norm_size {
            let lo = center[i] - radius[i];
            let hi = center[i] + radius[i];
            let sq_lower = if lo > 0.0 {
                (lo as f64) * (lo as f64)
            } else if hi < 0.0 {
                (hi as f64) * (hi as f64)
            } else {
                0.0 // interval contains 0
            };
            mean_sq_lower_f64 += sq_lower;
        }
        // next_down: mean_sq_lower feeds into rms_min denominator → round DOWN for soundness
        let mean_sq_lower = next_down_f32((mean_sq_lower_f64 / nf_64) as f32);
        // next_down: rms_min is a denominator → smaller = larger R2 = sound. #3327.
        let rms_min = next_down_f32(((mean_sq_lower as f64 + self.eps as f64).sqrt()) as f32);
        let rms_min_cubed = rms_min * rms_min * rms_min;
        let rms_min_5th = rms_min_cubed * rms_min * rms_min;

        // Compute position-dependent bounds for the second-order remainder.
        //
        // RmsNorm Hessian: H_i[j,k] = γ_i * [-δ_ij·x_k/(n·rms³)
        //   - δ_ik·x_j/(n·rms³) - x_i·δ_jk/(n·rms³) + 3·x_i·x_j·x_k/(n²·rms⁵)]
        //
        // Unlike LayerNorm, the Hessian involves x_i (not centered x_i - μ), so
        // the bound depends on the absolute position |c_i| + r_i, not just radii.
        //
        // At any ξ in the box: |ξ_k| ≤ X_k := |c_k| + r_k and rms(ξ) ≥ rms_min.
        // The second-order remainder for output dimension i is:
        //   R₂(i) ≤ (|γ_i|/2) · [2·r_i·S_Xr/(n·rms_min³)
        //     + X_i·‖r‖²/(n·rms_min³) + 3·X_i·S_Xr²/(n²·rms_min⁵)]
        // where S_Xr = Σ_k X_k · r_k.
        // next_up_f32: s_xr is in NUMERATOR of R2 → round up (sound). #3327.
        let s_xr: f32 = next_up_f32(
            (0..norm_size)
                .map(|k| {
                    let x_k = (center[k].abs() as f64) + (radius[k] as f64);
                    x_k * (radius[k] as f64)
                })
                .sum::<f64>() as f32,
        );

        for i in 0..norm_size {
            let c_i = center[i];
            let g_i = self.ny[i];
            let g_i_f64 = g_i as f64;
            let c_i_f64 = c_i as f64;

            // Center-point output (f64 for precision, not directed-rounded)
            let y_center = (g_i_f64 * c_i_f64 / rms_f64) as f32;

            // First-order: output_radius_i = sum_j |J[i,j]| * radius_j
            // J[i,j] = ny_i * (delta_ij / rms - c_i * c_j / (n * rms^3))
            // Jacobian entries and sum computed in f64 to avoid cancellation (#3327)
            let mut first_order_radius_f64 = 0.0_f64;
            for j in 0..norm_size {
                let delta_ij: f64 = if i == j { 1.0 } else { 0.0 };
                let j_ij = g_i_f64
                    * (delta_ij / rms_f64 - c_i_f64 * (center[j] as f64) / (nf_64 * rms_cubed_f64));
                first_order_radius_f64 += j_ij.abs() * radius[j] as f64;
            }
            // next_up: first_order_radius widens output bounds (sound)
            let first_order_radius = next_up_f32(first_order_radius_f64 as f32);

            // Second-order remainder with position-dependent Hessian bound.
            // Guard: if rms_min powers overflow to Inf, denominators become Inf
            // and second_order vanishes via IEEE 754 x/Inf=0 (unsound).
            let x_i = c_i.abs() + radius[i];
            let second_order =
                if rms_min_cubed.is_finite() && rms_min_5th.is_finite() && rms_min_cubed > 0.0 {
                    g_i.abs()
                        * 0.5
                        * (2.0 * radius[i] * s_xr / (nf * rms_min_cubed)
                            + x_i * radius_sq_sum / (nf * rms_min_cubed)
                            + 3.0 * x_i * s_xr * s_xr / (nf * nf * rms_min_5th))
                } else {
                    f32::INFINITY
                };

            let output_radius = first_order_radius + second_order;
            let lo = next_down_f32(y_center - output_radius); // #3344
            let hi = next_up_f32(y_center + output_radius);

            // Intersect the Jacobian interval with the sound affine z-saturation
            // envelope [g_i·(-Z), g_i·(+Z)] (b = 0 for RMSNorm), Z = max_norm_safe.
            // Sign-aware: g_i ≥ 0 ⇒ [g_i·(-Z), g_i·(+Z)]; g_i < 0 flips the ends.
            // Endpoints rounded OUTWARD; intersection via NaN-propagating max/min
            // (a NaN lo/hi from the Jacobian path survives and trips the post-loop
            // NaN guard, exactly as before).
            let (env_lo, env_hi) = if g_i >= 0.0 {
                (
                    next_down_f32(g_i * -max_norm_safe),
                    next_up_f32(g_i * max_norm_safe),
                )
            } else {
                (
                    next_down_f32(g_i * max_norm_safe),
                    next_up_f32(g_i * -max_norm_safe),
                )
            };
            let lo = nan_propagating_max(lo, env_lo);
            let hi = nan_propagating_min(hi, env_hi);
            emit(i, lo, hi);
        }
    }

    /// Conservative IBP for one normalization slice.
    /// Bounds mean(x²) via interval arithmetic, computes rms interval,
    /// then bounds y_i = ny_i * x_i / rms using 4-corner division.
    fn conservative_ibp_1d_slice(
        &self,
        lower: &[f32],
        upper: &[f32],
        emit: &mut impl FnMut(usize, f32, f32),
    ) {
        let norm_size = lower.len();
        // f64 accumulation + directed rounding for sq_sum (#3327, pattern from #3324)
        let nf_64 = norm_size as f64;

        // Bound mean(x^2) via interval arithmetic on x_i^2 per coordinate
        let mut sq_sum_lower_f64 = 0.0_f64;
        let mut sq_sum_upper_f64 = 0.0_f64;
        for i in 0..norm_size {
            let (sq_l, sq_u) = square_interval_bounds(lower[i], upper[i]);
            sq_sum_lower_f64 += sq_l as f64;
            sq_sum_upper_f64 += sq_u as f64;
        }

        // rms bounds (sqrt is monotone increasing)
        // next_down: rms_lower is denominator upper-bound side → round DOWN for soundness
        // next_up: rms_upper is denominator lower-bound side → round UP for soundness
        let rms_lower = next_down_f32(((sq_sum_lower_f64 / nf_64 + self.eps as f64).sqrt()) as f32);
        let rms_upper = next_up_f32(((sq_sum_upper_f64 / nf_64 + self.eps as f64).sqrt()) as f32);

        // Sound z-saturation clamp: |x_i / rms| ≤ max_norm_safe.
        //
        // PROOF OF MARGIN (denominator-relative — NO mean-cancellation term).
        // RMSNorm normalizes the RAW x_i (NOT a centered (x_i - mean)), so the
        // exact-arithmetic ceiling comes from Cauchy-Schwarz alone:
        //   z_i^2 = x_i^2 / (mean(x^2) + eps),  mean(x^2) = (Σ_k x_k^2)/n.
        //   Since x_i^2 ≤ Σ_k x_k^2 and rms^2 = (Σ x^2)/n + eps ≥ (Σ x^2)/n,
        //   z_i^2 ≤ Σ x^2 / ((Σ x^2)/n) = n  ⇒  |z_i| ≤ sqrt(n).
        // (eps > 0 makes this strict; sqrt(n) is the tight, not-improvable
        // constant. Unlike LayerNorm there is NO Σ(x_k - mean) = 0 cancellation
        // in the NUMERATOR, so the LayerNorm sqrt(n-1) constant and its
        // (n+2)*ulp(max_abs_x)/std_lower NUMERATOR margin do NOT apply here —
        // using sqrt(n-1) would be UNSOUND for RMSNorm.)
        //
        // But the DENOMINATOR rms = sqrt(mean(x^2) + eps) is an n-term
        // sum-of-squares; a spec-compliant naive f32 evaluation accumulates a
        // RELATIVE error ~0.5*n*EPSILON in that sum, so the realized
        // |z_i| = |x_i / fl(rms)| can exceed sqrt(n) by a relative ~0.5*n*EPSILON,
        // i.e. by ~base*0.5*n*EPSILON absolute. Hence (base = sqrt(n)):
        //   delta = base*(n*0.5 + 1)*EPSILON + 0.5*ulp(base)
        // the n*0.5 term is the sum-of-squares relative error, the +1 the final
        // x_i/rms divide's relative error at |z|<=base, 0.5*ulp(base) the sqrt(n)
        // rounding. No max_abs_x term: this margin is purely relative. Every step
        // rounds OUTWARD (next_up into delta, next_up into max_norm).
        //
        // SOUNDNESS OF THE CLAMP: the clamp is applied to the 4-corner ratios
        // x/rms_lower, where rms_lower (next_down-rounded, box-minimum mean(x^2))
        // satisfies rms_lower ≤ rms_true, so every corner xu/rms_lower ≥
        // x_i/rms_true. The clamp can therefore only OVER-approximate a feasible
        // value, never cut below a genuinely reachable |z_i|, for ANY
        // max_norm ≥ true sqrt(n) + delta. With max_norm_safe ≥ the realized f32
        // |z_i| ceiling the clamped IBP interval is a sound SUPERSET of the
        // reachable set => SOUND. It is still strictly tighter than the unclamped
        // 4-corner interval in the high-magnitude / low-variance regime,
        // retaining the #3196 benefit.
        let max_norm = {
            let base = (norm_size as f32).sqrt();
            let ulp_b = next_up_f32(base) - base; // 1 ULP at sqrt(n)
            let delta =
                next_up_f32(base * ((norm_size as f32) * 0.5 + 1.0) * f32::EPSILON + 0.5 * ulp_b);
            next_up_f32(base + delta) // #3344, max_norm_safe
        };

        // For each output y_i = ny_i * x_i / rms
        for i in 0..norm_size {
            let (lo, hi) = bound_ny_x_over_rms(
                self.ny[i], lower[i], upper[i], rms_lower, rms_upper, max_norm,
            );
            emit(i, lo, hi);
        }
    }
}

impl BoundPropagation for RmsNormLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();

        validate_ibp_input(shape, &self.ny, self.forward_mode, input)?;

        if self.forward_mode {
            let out = self.propagate_ibp_forward_mode(input)?;
            // Same proven sphere applies (the ‖z‖₂ ≤ √n bound is independent of
            // the box relaxation mode). Drop on failure (sound).
            return Ok(match self.compute_l2_constraint(shape) {
                Some(c) => out.with_l2_constraint(c),
                None => out,
            });
        }

        let norm_size = shape[ndim - 1];
        let mut out_lower = input.lower().clone();
        let mut out_upper = input.upper().clone();

        for_each_norm_slice(shape, |prefix| {
            let prefix_len = prefix.len();
            let mut full_idx = [0usize; 8];
            full_idx[..prefix_len].copy_from_slice(prefix);
            let mut lower_slice = Vec::with_capacity(norm_size);
            let mut upper_slice = Vec::with_capacity(norm_size);
            for i in 0..norm_size {
                full_idx[prefix_len] = i;
                let idx = &full_idx[..=prefix_len];
                lower_slice.push(input.lower()[idx]);
                upper_slice.push(input.upper()[idx]);
            }
            let prefix_snapshot: [usize; 8] = full_idx;
            self.conservative_ibp_1d_slice(&lower_slice, &upper_slice, &mut |i, lo, hi| {
                let mut emit_idx = prefix_snapshot;
                emit_idx[prefix_len] = i;
                out_lower[&emit_idx[..=prefix_len]] = lo;
                out_upper[&emit_idx[..=prefix_len]] = hi;
            });
        })?;

        // Post-check: reject NaN
        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "RMSNorm IBP: NaN in computed bounds".to_string(),
            ));
        }

        let out = BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)?;
        // THE LEVER: attach the proven ‖y‖₂ ≤ max|g|·√n sphere so the
        // immediately-downstream Linear can replace its decorrelated box bound
        // (‖w‖₁·√n) with the exact Cauchy–Schwarz bound (‖w‖₂·√n). Intersection
        // only tightens; if the constraint cannot be built it is dropped (sound).
        Ok(match self.compute_l2_constraint(shape) {
            Some(c) => out.with_l2_constraint(c),
            None => out,
        })
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "RMSNorm is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        RmsNormLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

// ── Free-standing helpers ──────────────────────────────────────────────────

/// Validate input shape and parameters for IBP propagation.
fn validate_ibp_input(
    shape: &[usize],
    ny: &Array1<f32>,
    forward_mode: bool,
    input: &BoundedTensor,
) -> Result<()> {
    let ndim = shape.len();
    if ndim == 0 {
        return Err(NyError::InvalidSpec(
            "RMSNorm requires at least 1D input".to_string(),
        ));
    }
    // The per-slice index buffer is a fixed `[0usize; 8]` (`full_idx`), so a
    // rank > 8 input would make `full_idx[prefix_len]` / `full_idx[..=prefix_len]`
    // index out of bounds and PANIC. Fail closed on higher-rank tensors (e.g. a
    // malformed/adversarial ONNX model) instead. (Trust verifier: closes the
    // index_out_of_bounds obligations in the RMSNorm IBP path.)
    if ndim > 8 {
        return Err(NyError::InvalidSpec(format!(
            "RMSNorm IBP supports tensors up to rank 8, got rank {ndim}"
        )));
    }
    if shape.contains(&0) {
        return Err(NyError::InvalidSpec(
            "RMSNorm: zero-valued dimension in input shape".to_string(),
        ));
    }
    // Reject non-finite input bounds (Category B per domain validation policy).
    if !forward_mode
        && (input.lower().iter().any(|x| !x.is_finite())
            || input.upper().iter().any(|x| !x.is_finite()))
    {
        return Err(NyError::NumericalInstability(
            "RMSNorm IBP: non-finite input bounds".to_string(),
        ));
    }
    let norm_size = shape[ndim - 1];
    if norm_size > (1 << 24) {
        return Err(NyError::InternalError(format!(
            "RMSNorm dimension {norm_size} exceeds f32 exact integer range"
        )));
    }
    if ny.len() != norm_size {
        return Err(NyError::ShapeMismatch {
            expected: vec![norm_size],
            got: vec![ny.len()],
        });
    }
    Ok(())
}

/// Iterate over normalization slices along the last axis.
///
/// Calls `process` for each batch position with the batch prefix slice.
/// The caller builds full N-D indices on the stack by appending the
/// feature index to the prefix.
///
/// Part of #2237: replaced `Fn(usize) -> Vec<usize>` callback with
/// direct prefix passing to eliminate per-element heap allocations.
fn for_each_norm_slice(shape: &[usize], mut process: impl FnMut(&[usize])) -> Result<()> {
    let ndim = shape.len();
    let prefix_len = ndim - 1;
    let batch_size: usize =
        checked_dim_product(&shape[..prefix_len], "RmsNorm IBP batch dimensions")?;

    let mut batch_buf = [0usize; 8];
    for batch_idx in 0..batch_size {
        let mut remaining = batch_idx;
        for d in (0..prefix_len).rev() {
            batch_buf[d] = remaining % shape[d];
            remaining /= shape[d];
        }
        process(&batch_buf[..prefix_len]);
    }
    Ok(())
}

/// Bound y = ny * x / rms via 4-corner interval arithmetic.
///
/// Given x in [xl, xu] and rms in [rms_lower, rms_upper] (both positive),
/// computes [lo, hi] for ny * x / rms using sign analysis of ny.
/// Clamps the ratio x/rms to [-max_norm, max_norm] for tightness (Cauchy-Schwarz).
#[inline]
fn bound_ny_x_over_rms(
    ny: f32,
    xl: f32,
    xu: f32,
    rms_lower: f32,
    rms_upper: f32,
    max_norm: f32,
) -> (f32, f32) {
    let corners = [
        xl / rms_upper,
        xl / rms_lower,
        xu / rms_upper,
        xu / rms_lower,
    ];

    let ratio_l = corners
        .iter()
        .fold(f32::INFINITY, |a, &b| nan_propagating_min(a, b));
    let ratio_u = corners
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| nan_propagating_max(a, b));

    // Clamp to [-sqrt(n), sqrt(n)]. NaN must propagate through the clamp
    // so downstream guards fire correctly.
    let ratio_l = nan_propagating_max(ratio_l, -max_norm);
    let ratio_u = nan_propagating_min(ratio_u, max_norm);

    if ny >= 0.0 {
        // directed rounding (#3344)
        (next_down_f32(ny * ratio_l), next_up_f32(ny * ratio_u))
    } else {
        (next_down_f32(ny * ratio_u), next_up_f32(ny * ratio_l))
    }
}
