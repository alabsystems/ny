// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Forward-mode IBP for LayerNorm using Jacobian-based propagation.
//!
//! Contains `propagate_ibp_forward_mode` and the core
//! `forward_mode_standard_1d_slice` mathematical kernel.

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::super::types::{LayerNormLayer, LayerNormMode};
use super::slices;
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::normalization::math_common::outward_midpoint_radius;

impl LayerNormLayer {
    /// Forward-mode IBP for LayerNorm using Jacobian-based propagation.
    ///
    /// Computes the Jacobian at the center point (midpoint of bounds), then
    /// propagates input radii through the absolute Jacobian, and adds a
    /// second-order remainder term for curvature.
    ///
    /// This implementation is admitted only as heuristic analysis, not as
    /// proof authority. The outward arithmetic below hardens its enclosure
    /// behavior but does not change that provenance classification.
    ///
    /// ## Math (LayerNorm Standard)
    ///
    /// `y_i = ny_i * (x_i - mean(x)) / std(x) + beta_i`
    ///
    /// Jacobian at center `c` with `mean_c`, `std_c`, `z_i = (c_i - mean_c)/std_c`:
    /// ```text
    /// J[i,j] = ny_i / std_c * (delta_ij - 1/n - z_i * z_j / n)
    /// ```
    ///
    /// First-order output radius:
    /// ```text
    /// output_radius_i = sum_j |J[i,j]| * radius_j
    /// ```
    ///
    /// Reference: Jacobian from `math.rs` (verified against finite differences).
    /// Fix for #3098: replaces the unsound `max_radius / n` coupling correction.
    #[inline]
    pub(crate) fn propagate_ibp_forward_mode(
        &self,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        // Guard: zero-valued dimensions cause division-by-zero in batch index
        // decoding. (#2806)
        if shape.contains(&0) {
            return Err(NyError::InvalidSpec(
                "LayerNorm: zero-valued dimension in input shape".to_string(),
            ));
        }
        let norm_size = shape[ndim - 1];

        // Non-finite inputs already rejected by propagate_ibp() caller.

        let Some((center, radius)) = outward_midpoint_radius(input.lower(), input.upper()) else {
            return self.fallback_output_bounds(shape);
        };

        match self.mode {
            LayerNormMode::Standard => {
                let mut out_lower = input.lower().clone();
                let mut out_upper = input.upper().clone();
                let nf = norm_size as f32;

                if ndim == 1 {
                    let center_slice: Vec<f32> = (0..norm_size).map(|i| center[[i]]).collect();
                    let radius_slice: Vec<f32> = (0..norm_size).map(|i| radius[[i]]).collect();
                    self.forward_mode_standard_1d_slice(
                        &center_slice,
                        &radius_slice,
                        nf,
                        &mut |i, lo, hi| {
                            out_lower[[i]] = lo;
                            out_upper[[i]] = hi;
                        },
                    );
                } else {
                    let bs = slices::batch_size(shape)?;
                    let prefix_len = ndim - 1;
                    let mut full_idx = [0usize; 8]; // stack buffer, part of #2237

                    for batch_idx in 0..bs {
                        slices::decode_batch_prefix_into(
                            shape,
                            batch_idx,
                            &mut full_idx[..prefix_len],
                        );
                        let center_slice = slices::collect_last_axis_slice(
                            &center,
                            &full_idx[..prefix_len],
                            norm_size,
                        );
                        let radius_slice = slices::collect_last_axis_slice(
                            &radius,
                            &full_idx[..prefix_len],
                            norm_size,
                        );

                        let prefix_snapshot: [usize; 8] = full_idx;
                        self.forward_mode_standard_1d_slice(
                            &center_slice,
                            &radius_slice,
                            nf,
                            &mut |i, lo, hi| {
                                let mut emit_idx = prefix_snapshot;
                                emit_idx[prefix_len] = i;
                                out_lower[&emit_idx[..=prefix_len]] = lo;
                                out_upper[&emit_idx[..=prefix_len]] = hi;
                            },
                        );
                    }
                }

                // Post-check: reject NaN in forward-mode output bounds.
                // Consistent with InstanceNorm forward-mode and the conservative
                // path which also rejects NaN.
                if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
                    return Err(NyError::NumericalInstability(
                        "LayerNorm forward-mode IBP: NaN in computed bounds".to_string(),
                    ));
                }

                // Repair non-finite outputs: sensitivity * radius can produce Inf
                // for large ny or small std. Consistent with IBP overflow
                // strategy (#3030, #3060).
                BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
            }
            LayerNormMode::MeanOnly => {
                // MeanOnly: y_i = ny_i * (x_i - mean(X)) + beta_i
                // Jacobian J[i,j] = ny_i * (delta_{ij} - 1/n) is CONSTANT
                // (independent of x), so forward-mode provides no accuracy benefit
                // over conservative interval arithmetic. Use interval mean bounds
                // for soundness: diff_l = xl - mean_upper, diff_u = xu - mean_lower.
                // Fix for #3142: was using center-point mean, producing unsound bounds.
                let mut out_lower = input.lower().clone();
                let mut out_upper = input.upper().clone();

                if ndim == 1 {
                    // f64 accumulation for mean to prevent precision loss. Part of #2423.
                    if norm_size == 0 {
                        return Err(NyError::InvalidSpec(
                            "LayerNorm MeanOnly: zero norm_size".to_string(),
                        ));
                    }
                    let nf_64 = norm_size as f64;
                    // Directed rounding: mean_l is a lower bound -> round down (#3270).
                    let mean_l: f32 = next_down_f32(
                        (input.lower().iter().map(|&x| x as f64).sum::<f64>() / nf_64) as f32,
                    );
                    // Directed rounding: mean_u is an upper bound -> round up (#3270).
                    let mean_u: f32 = next_up_f32(
                        (input.upper().iter().map(|&x| x as f64).sum::<f64>() / nf_64) as f32,
                    );
                    for i in 0..norm_size {
                        let xl = input.lower()[[i]];
                        let xu = input.upper()[[i]];
                        let g = self.ny[i];
                        let b = self.beta[i];
                        // Directed rounding on intermediate deviation. Part of #3344.
                        let diff_l = next_down_f32(xl - mean_u);
                        let diff_u = next_up_f32(xu - mean_l);

                        // Directed rounding on final bounds: bare f32 mul+add
                        // can lose up to 1 ULP per operation. Part of #3344.
                        if g >= 0.0 {
                            out_lower[[i]] = next_down_f32(g * diff_l + b);
                            out_upper[[i]] = next_up_f32(g * diff_u + b);
                        } else {
                            out_lower[[i]] = next_down_f32(g * diff_u + b);
                            out_upper[[i]] = next_up_f32(g * diff_l + b);
                        }
                    }
                } else {
                    let bs = slices::batch_size(shape)?;
                    let prefix_len = ndim - 1;
                    let mut full_idx = [0usize; 8]; // stack buffer, part of #2237

                    for batch_idx in 0..bs {
                        slices::decode_batch_prefix_into(
                            shape,
                            batch_idx,
                            &mut full_idx[..prefix_len],
                        );

                        // f64 accumulation for mean computation. Part of #2423.
                        let mut lower_sum: f64 = 0.0;
                        let mut upper_sum: f64 = 0.0;
                        for i in 0..norm_size {
                            full_idx[prefix_len] = i;
                            let idx = &full_idx[..=prefix_len];
                            lower_sum += input.lower()[idx] as f64;
                            upper_sum += input.upper()[idx] as f64;
                        }
                        // Directed rounding: mean_l is a lower bound -> round down (#3270).
                        let mean_l = next_down_f32((lower_sum / norm_size as f64) as f32);
                        // Directed rounding: mean_u is an upper bound -> round up (#3270).
                        let mean_u = next_up_f32((upper_sum / norm_size as f64) as f32);

                        for i in 0..norm_size {
                            full_idx[prefix_len] = i;
                            let idx = &full_idx[..=prefix_len];

                            let xl = input.lower()[idx];
                            let xu = input.upper()[idx];
                            let g = self.ny[i];
                            let b = self.beta[i];
                            // Directed rounding on intermediate deviation. Part of #3344.
                            let diff_l = next_down_f32(xl - mean_u);
                            let diff_u = next_up_f32(xu - mean_l);

                            // Directed rounding on final bounds. Part of #3344.
                            if g >= 0.0 {
                                out_lower[idx] = next_down_f32(g * diff_l + b);
                                out_upper[idx] = next_up_f32(g * diff_u + b);
                            } else {
                                out_lower[idx] = next_down_f32(g * diff_u + b);
                                out_upper[idx] = next_up_f32(g * diff_l + b);
                            }
                        }
                    }
                }

                // Repair non-finite outputs: g * diff can produce Inf for large
                // ny or wide input intervals. Consistent with IBP overflow
                // strategy (#3030, #3060).
                BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
            }
        }
    }

    /// Core forward-mode computation for a single Standard-mode normalization slice.
    ///
    /// Computes Jacobian-based output bounds for one 1D slice of the input.
    /// Calls `emit(i, lower_i, upper_i)` for each output coordinate.
    ///
    /// LayerNorm Standard: `y_i = ny_i * (x_i - mean) / std + beta_i`
    ///
    /// Jacobian at center `c`:
    /// ```text
    /// J[i,j] = ny_i / std_c * (delta_ij - 1/n - z_i * z_j / n)
    /// ```
    /// where `z_i = (c_i - mean_c) / std_c`.
    ///
    /// Fix for #3098: replaces the unsound `max_radius / n` coupling correction
    /// with exact Jacobian propagation plus a conservative second-order remainder.
    #[inline]
    fn forward_mode_standard_1d_slice(
        &self,
        center: &[f32],
        radius: &[f32],
        nf: f32,
        emit: &mut impl FnMut(usize, f32, f32),
    ) {
        let norm_size = center.len();
        let nf_64 = nf as f64;

        // f64 accumulation for mean and variance to prevent catastrophic cancellation
        // for large norm_size (768+). Part of #2423.
        // Reference: Higham, "Accuracy and Stability of Numerical Algorithms", S4.2.
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
        // in z_i * z_j terms used by the Jacobian.
        let std_c_f64 = std_c as f64;
        let z_f64: Vec<f64> = center
            .iter()
            .map(|&ci| (ci as f64 - mean_c_f64) / std_c_f64)
            .collect();
        let z: Vec<f32> = z_f64.iter().map(|&zi| zi as f32).collect();

        // Precompute: sum of radius^2 for the second-order remainder (f64 accumulation).
        // next_up_f32: radius_sq_sum is in the NUMERATOR of the second-order bound
        // R2 = 3.5*|ny|*||r||^2/(sqrt(n)*sigma^2). Rounding up ensures R2 is an upper bound.
        let radius_sq_sum: f32 =
            next_up_f32((radius.iter().map(|&r| (r as f64) * (r as f64)).sum::<f64>()) as f32);

        // Compute minimum std over the input box using interval arithmetic.
        //
        // The second-order remainder depends on 1/sigma^2 (Hessian scale factor),
        // so we need the MINIMUM std over the box for a sound bound. Using center-point std
        // is unsound when the box reaches low-variance regions (e.g., when all
        // elements can be equal within the box, variance -> 0, Hessian -> inf).
        //
        // Interval arithmetic: mean in [mean_c - mean_r, mean_c + mean_r] where
        // mean_r = sum(radius)/n. For each element, the centered interval is
        // [c_i - mean_c - r_i - mean_r, c_i - mean_c + r_i + mean_r]. The lower
        // bound on the squared centered value is 0 when the interval straddles 0.
        // next_up_f32: mean_radius widens the centered intervals [a_lower, a_upper].
        // Rounding UP makes intervals wider -> var_lower smaller -> std_min smaller ->
        // second-order remainder larger (conservative, sound).
        let mean_radius: f32 =
            next_up_f32((radius.iter().map(|&r| r as f64).sum::<f64>() / nf_64) as f32);
        let mut var_lower_f64: f64 = 0.0;
        for i in 0..norm_size {
            let a_c = center[i] - mean_c; // centered value at center
            let a_lower = a_c - radius[i] - mean_radius;
            let a_upper = a_c + radius[i] + mean_radius;
            let sq_lower = if a_lower > 0.0 {
                (a_lower as f64) * (a_lower as f64)
            } else if a_upper < 0.0 {
                (a_upper as f64) * (a_upper as f64)
            } else {
                0.0 // interval straddles zero
            };
            var_lower_f64 += sq_lower;
        }
        // Directed rounding: var_lower is a lower bound on variance -> round down.
        // Smaller variance -> smaller std denominator -> tighter (unsound) bounds
        // if not rounded conservatively (#3270).
        let var_lower = next_down_f32((var_lower_f64 / nf_64) as f32);
        // Directed rounding on sqrt: std_min is a denominator -> round DOWN
        // (smaller denominator = larger R2 = sound). f64 intermediate for
        // precision. Pattern from GroupNorm #3327, fix for #3332.
        let std_min = next_down_f32(((var_lower as f64 + self.eps as f64).sqrt()) as f32);

        // Sound affine envelope on the normalized value z, to INTERSECT with the
        // Jacobian (first-order + remainder) interval below. In exact real
        // arithmetic the centered-normalized z satisfies |z_i| <= sqrt(n-1)
        // (mean-subtraction removes one DOF). The downstream f32 kernel computes
        // z_i = fl(fl(x_i - mean)/std), whose absolute deviation from exact z is
        // bounded by (n+2)*ulp(max_abs_x)/std + base*EPSILON + 0.5*ulp(sqrt(n-1)):
        // a spec-compliant naive f32 mean is a sequential sum of n terms of
        // magnitude ~M=max_abs_x, accumulating ~(n-1)*0.5*ulp(n*M) rounding ->
        // ~0.5*(n-1)*ulp(M) after /n, plus the (x_i - mean) subtraction (~ulp(M)),
        // so the numerator error is ~0.5*(n+1)*ulp(M), NOT ulp(M): a bare
        // ulp(max_abs_x) (no n factor) UNDER-counts it and is UNSOUND. The (n+2)
        // factor dominates. Amplified by 1/std (std >= std_min, std_min next_down'd
        // so 1/std_min over-estimates), base*EPSILON covers the final divide's
        // relative error at |z|<=base, plus the divide rounding <= 0.5*ulp(sqrt(n-1)).
        // Every step rounds strictly OUTWARD (next_up), so Z = max_norm_safe >= the
        // true f32 ceiling; the envelope [g*(-Z)+b, g*(+Z)+b] is therefore a sound
        // SUPERSET of every reachable g*z+b, and intersecting it with the
        // Jacobian interval can only tighten -- never exclude a reachable output.
        //
        // This is the LayerNorm (mean-subtract) margin: the (n+2)*ulp(max_abs_x)/std
        // term is REQUIRED here precisely because of (x-mean) cancellation;
        // RMSNorm (no centering) uses sqrt(n) with a denominator-relative margin.
        //
        // max_abs_x over the box: max_i (|center_i| + radius_i). Non-finite guard:
        // if it is Inf/NaN (Inf center/radius), set Z = +INF so the envelope is
        // vacuous (no tightening) and the existing repair paths handle it.
        let max_abs_x = center
            .iter()
            .zip(radius.iter())
            .map(|(&c, &r)| next_up_f32(c.abs() + r.abs()))
            .fold(0.0_f32, f32::max);
        let max_norm = if norm_size > 1 {
            if max_abs_x.is_finite() && std_min > 0.0 {
                let base = ((nf_64 - 1.0).sqrt()) as f32; // norm_size > 1
                let ulp_x = next_up_f32(max_abs_x) - max_abs_x; // 1 ULP at max_abs_x
                let ulp_b = next_up_f32(base) - base; // 1 ULP at sqrt(n-1)
                                                      // (n+2) ULP(max|x|): a spec-compliant naive f32 mean is a
                                                      // sequential sum of n terms of magnitude ~M, accumulating
                                                      // ~(n-1)*0.5*ulp(n*M) error; after /n that is ~0.5*(n-1)*ulp(M),
                                                      // plus the (x_i - mean) subtraction (~ulp(M)). A bare ulp(max|x|)
                                                      // (no n factor) UNDER-counts this and is unsound; (n+2) dominates
                                                      // the (n-1)-term summation + subtraction. base*EPSILON covers the
                                                      // final divide's relative error at |z|<=base; 0.5*ulp(base) sqrt.
                let delta =
                    next_up_f32((nf + 2.0) * ulp_x / std_min + base * f32::EPSILON + 0.5 * ulp_b);
                next_up_f32(base + delta)
            } else {
                f32::INFINITY
            }
        } else {
            // n == 1: exact z is identically 0; the f32 numerator (x - mean) is
            // also exactly 0 (single-element mean), so the envelope is {b}.
            0.0
        };

        for i in 0..norm_size {
            let g_i = self.ny[i];
            let b_i = self.beta[i];

            // Center-point output
            let y_center = g_i * z[i] + b_i;

            // First-order: output_radius_i = sum_j |J[i,j]| * radius_j
            // J[i,j] = ny_i / std_c * (delta_ij - 1/n - z_i * z_j / n)
            // f64 for both individual Jacobian entries and the sum to prevent
            // cancellation in (delta_ij - 1/n - z_i*z_j/n). Part of #2423.
            let g_i_f64 = g_i as f64;
            let mut first_order_radius_f64 = 0.0_f64;
            for j in 0..norm_size {
                let delta_ij: f64 = if i == j { 1.0 } else { 0.0 };
                let j_ij_f64 =
                    g_i_f64 / std_c_f64 * (delta_ij - 1.0 / nf_64 - z_f64[i] * z_f64[j] / nf_64);
                first_order_radius_f64 += j_ij_f64.abs() * radius[j] as f64;
            }
            // Directed rounding: radius widens output bounds -> round up (#3270).
            let first_order_radius = next_up_f32(first_order_radius_f64 as f32);

            // Second-order remainder: conservative bound on Hessian contribution.
            //
            // The Hessian of z_i w.r.t. x has the form:
            //   H_i[j,k] = (1/(n*sigma^2)) * bracket(z_i, z_j, z_k, delta)
            // where the bracket contains 4 terms bounded using ||z||^2 = n:
            //   T1,T2: Sum|delta_{ij}-1/n|*|z_k|*r_j*r_k <= sqrt(n)*||r||^2 (Cauchy-Schwarz)
            //   T3: |z_i|*Sum|delta_{jk}-1/n|*r_j*r_k <= 2*sqrt(n)*||r||^2 (eigenvalue bound)
            //   T4: 3|z_i|/(n)*(Sum|z_j|*r_j)^2 <= 3*sqrt(n)*||r||^2 (Cauchy-Schwarz)
            // Total: Sum_{j,k}|bracket|*r_j*r_k <= 7*sqrt(n)*||r||^2
            //
            // Therefore R2(i) <= (1/2)|ny_i| * 7*sqrt(n)*||r||^2 / (n*sigma_min^2)
            //                  = 7|ny_i|*||r||^2 / (2*sqrt(n)*sigma_min^2)
            //
            // Uses sigma_min (minimum sigma over the input box) for soundness since
            // 1/sigma^2 is decreasing.
            //
            // Guard: if sigma_min^2 overflows to Inf (extreme center values ~1e19+),
            // the denominator becomes Inf and second_order would vanish via
            // IEEE 754 x/Inf=0, silently dropping the curvature correction
            // (unsound). Emit Inf to trigger new_repaired fallback (#3423).
            let second_order_denom = nf.sqrt() * std_min * std_min;
            let second_order = if second_order_denom.is_finite() && second_order_denom > 0.0 {
                3.5 * g_i.abs() * radius_sq_sum / second_order_denom
            } else {
                f32::INFINITY
            };

            let output_radius = first_order_radius + second_order;

            // Directed rounding on final emit: y_center has f64->f32 rounding
            // error up to 0.5 ULP, and f32 subtraction/addition adds another
            // 0.5 ULP. next_down/next_up compensates both. Part of #3344.
            let jac_l = next_down_f32(y_center - output_radius);
            let jac_u = next_up_f32(y_center + output_radius);

            // Intersect with the sound affine envelope [g*(-Z)+b, g*(+Z)+b]
            // (Z = max_norm). Sign-aware: g >= 0 maps z in [-Z, Z] to
            // [g*(-Z)+b, g*(+Z)+b]; g < 0 flips the endpoints. Endpoints are
            // directed-rounded OUTWARD (next_down lower / next_up upper) so the
            // envelope stays a sound superset of every reachable g*z+b. When
            // Z = +INF (non-finite guard) the envelope endpoints are +/-INF and
            // the intersection is a no-op. nan_propagating_max/min keep NaN
            // flowing so the downstream NaN post-check fires.
            let (env_l, env_u) = if g_i >= 0.0 {
                (
                    next_down_f32(g_i * (-max_norm) + b_i),
                    next_up_f32(g_i * max_norm + b_i),
                )
            } else {
                (
                    next_down_f32(g_i * max_norm + b_i),
                    next_up_f32(g_i * (-max_norm) + b_i),
                )
            };
            let out_l = nan_propagating_max(jac_l, env_l);
            let out_u = nan_propagating_min(jac_u, env_u);

            emit(i, out_l, out_u);
        }
    }
}
