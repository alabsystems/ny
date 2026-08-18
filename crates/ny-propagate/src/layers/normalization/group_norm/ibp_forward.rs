// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Forward-mode IBP for GroupNorm using Jacobian-based propagation.
//!
//! Split from ibp.rs to stay under the 500-line file limit.
//! Part of #3205.

use ny_core::{checked_dim_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor, RepairStrategy};

use super::super::math_common::{compute_batch_prefix, outward_midpoint_radius};
use super::types::GroupNormLayer;
use crate::bounds::{nan_propagating_max, nan_propagating_min};

impl GroupNormLayer {
    /// Forward-mode IBP for one group: compute per-element output bounds.
    ///
    /// Uses first-order Taylor expansion around center point with second-order
    /// remainder bound. This path is admitted only as heuristic analysis, not
    /// proof authority. Returns (lower, upper) vectors of length group_size.
    fn ibp_forward_mode_group(
        &self,
        center_vals: &[f32],
        radius_vals: &[f32],
        group_idx: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let cpg = self.channels_per_group();
        let group_size = center_vals.len();
        let time_len = group_size / cpg;
        let nf = group_size as f32;
        // f64 accumulation + directed rounding (#3327, pattern from #3324)
        let nf_64 = group_size as f64;
        let eps_64 = self.eps as f64;
        let group_start_ch = group_idx * cpg;

        // Group mean and std at center — fully in f64 to prevent cancellation (#3327)
        let mean_c_f64: f64 = center_vals.iter().map(|&x| x as f64).sum::<f64>() / nf_64;
        let mean_c = mean_c_f64 as f32;
        let var_c_f64: f64 = center_vals
            .iter()
            .map(|&xi| {
                let d = xi as f64 - mean_c_f64;
                d * d
            })
            .sum::<f64>()
            / nf_64;
        let std_c_f64 = (var_c_f64 + eps_64).sqrt();
        // z values in f64 for Jacobian cancellation resistance
        let z_f64: Vec<f64> = center_vals
            .iter()
            .map(|&ci| (ci as f64 - mean_c_f64) / std_c_f64)
            .collect();

        // next_up: radius_sq_sum is in the NUMERATOR of second-order bound
        let radius_sq_sum: f32 = next_up_f32(
            (radius_vals
                .iter()
                .map(|&r| (r as f64) * (r as f64))
                .sum::<f64>()) as f32,
        );

        // Minimum std bound for second-order remainder
        // next_up: mean_radius widens centered intervals (conservative/sound)
        let mean_radius: f32 =
            next_up_f32((radius_vals.iter().map(|&r| r as f64).sum::<f64>() / nf_64) as f32);
        let std_min =
            self.ibp_forward_mode_std_lower(center_vals, radius_vals, mean_c, mean_radius);

        // Sound affine-envelope ceiling for the forward-mode (Jacobian + 2nd-order)
        // output interval. GroupNorm is mean-subtract (centered): the normalized value
        // satisfies |z_i| <= sqrt(n-1) in exact reals, and the downstream f32 program
        // overshoots by at most
        //   delta = (n+2)*ulp(max_abs_x)/std_min + base*EPSILON + 0.5*ulp(sqrt(n-1)).
        // A spec-compliant naive f32 group mean is a sequential sum of n=group_size
        // terms of magnitude ~M=max_abs_x, accumulating ~(n-1)*0.5*ulp(n*M) rounding
        // -> ~0.5*(n-1)*ulp(M) after /n, plus the (x - mean) subtraction (~ulp(M)),
        // so the numerator error is ~0.5*(n+1)*ulp(M): a bare ulp(max_abs_x) (no n
        // factor) UNDER-counts it and is UNSOUND. The (n+2) factor dominates;
        // base*EPSILON covers the final divide's relative error at |z|<=base. The
        // term is amplified by the smallest sound std (std_min rounded DOWN so
        // 1/std_min over-estimates 1/std). Every step rounds OUTWARD (next_up), so
        // max_norm_safe >= the true reachable |z| ceiling.
        //
        // The Jacobian first-order + second-order remainder interval is itself a sound
        // enclosure of the reachable f32 output, and the affine envelope
        // [g*(-Z)+b, g*(+Z)+b] (Z = max_norm_safe) is ALSO a sound enclosure (since every
        // reachable z lies in [-Z, Z], and y = g*z + b is affine/monotone in z). The
        // intersection of two sound enclosures is still a sound enclosure, and never
        // looser than either — so intersecting tightens without ever excluding a
        // reachable output. max_abs_x over the box = max_t(|center_t| + radius_t).
        let max_norm_safe = if group_size > 1 {
            let max_abs_x = center_vals
                .iter()
                .zip(radius_vals.iter())
                .map(|(&c, &r)| c.abs() + r.abs())
                .fold(0.0_f32, f32::max);
            let base = (((group_size as f64) - 1.0).sqrt()) as f32;
            if max_abs_x.is_finite() && std_min > 0.0 && std_min.is_finite() {
                let ulp_x = next_up_f32(max_abs_x) - max_abs_x;
                let ulp_b = next_up_f32(base) - base;
                // (n+2) ULP(max|x|): a spec-compliant naive f32 group mean is a
                // sequential sum of n=group_size terms of magnitude ~M, accumulating
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
            // n == 1: z is identically 0; envelope collapses to {b} (handled per-element).
            0.0
        };

        let mut grp_lower = vec![0.0_f32; group_size];
        let mut grp_upper = vec![0.0_f32; group_size];

        for s in 0..group_size {
            let c = group_start_ch + (s / time_len);
            let ny_s = self.ny[c];
            let beta_s = self.beta[c];
            // Center-point output in f64 for precision (#3327)
            let y_center = (ny_s as f64 * z_f64[s] + beta_s as f64) as f32;

            // First-order radius: sum of |J_sj| * radius_j
            // Jacobian entries and sum in f64 to avoid cancellation (#3327)
            let g_f64 = ny_s as f64;
            let mut first_order_radius_f64 = 0.0_f64;
            for j in 0..group_size {
                let delta_sj: f64 = if s == j { 1.0 } else { 0.0 };
                let j_sj =
                    g_f64 / std_c_f64 * (delta_sj - 1.0 / nf_64 - z_f64[s] * z_f64[j] / nf_64);
                first_order_radius_f64 += j_sj.abs() * radius_vals[j] as f64;
            }
            // next_up: first_order_radius widens output bounds (sound)
            let first_order_radius = next_up_f32(first_order_radius_f64 as f32);

            // Second-order remainder bound
            let second_order_denom = nf.sqrt() * std_min * std_min;
            let second_order = if second_order_denom.is_finite() && second_order_denom > 0.0 {
                3.5 * ny_s.abs() * radius_sq_sum / second_order_denom
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
            // Z = max_norm_safe. y = ny_s*z + beta_s is affine in z; for ny_s >= 0 the
            // envelope endpoints are (beta_s - ny_s*Z, beta_s + ny_s*Z), for ny_s < 0 they
            // are swapped. Endpoints rounded OUTWARD (down for lower, up for upper) so the
            // envelope stays a sound enclosure. NaN-propagating max/min so a NaN Jacobian
            // bound (degenerate std) survives the intersection and fires downstream guards.
            if max_norm_safe.is_finite() {
                let (env_lower, env_upper) = if ny_s >= 0.0 {
                    (
                        next_down_f32(beta_s - ny_s * max_norm_safe),
                        next_up_f32(beta_s + ny_s * max_norm_safe),
                    )
                } else {
                    (
                        next_down_f32(beta_s + ny_s * max_norm_safe),
                        next_up_f32(beta_s - ny_s * max_norm_safe),
                    )
                };
                jac_lower = nan_propagating_max(jac_lower, env_lower);
                jac_upper = nan_propagating_min(jac_upper, env_upper);
            }

            grp_lower[s] = jac_lower;
            grp_upper[s] = jac_upper;
        }

        (grp_lower, grp_upper)
    }

    /// Lower bound on std for forward-mode second-order remainder.
    fn ibp_forward_mode_std_lower(
        &self,
        center_vals: &[f32],
        radius_vals: &[f32],
        mean_c: f32,
        mean_radius: f32,
    ) -> f32 {
        // f64 accumulation for var_lower (#3327, pattern from #3324)
        let nf_64 = center_vals.len() as f64;
        let mut var_lower_f64: f64 = 0.0;
        for i in 0..center_vals.len() {
            let a_c = center_vals[i] - mean_c;
            let a_lower = a_c - radius_vals[i] - mean_radius;
            let a_upper = a_c + radius_vals[i] + mean_radius;
            let sq_lower = if a_lower > 0.0 {
                (a_lower as f64) * (a_lower as f64)
            } else if a_upper < 0.0 {
                (a_upper as f64) * (a_upper as f64)
            } else {
                0.0
            };
            var_lower_f64 += sq_lower;
        }
        // next_down: var_lower feeds into std_min denominator → round DOWN for soundness
        let var_lower = next_down_f32((var_lower_f64 / nf_64) as f32);
        // next_down: std_min is a denominator → smaller = larger R2 = sound. #3327.
        next_down_f32(((var_lower as f64 + self.eps as f64).sqrt()) as f32)
    }

    /// Forward-mode IBP for GroupNorm using Jacobian-based propagation.
    ///
    /// Computes the Jacobian at the center point, then propagates input radii
    /// through the absolute Jacobian plus a second-order remainder term.
    pub(super) fn propagate_ibp_forward_mode(
        &self,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let shape = input.shape();
        let ndim = shape.len();
        let (_num_channels, time_len) = self.validate_ibp_input(input)?;
        let cpg = self.channels_per_group();
        let group_size = cpg * time_len;

        let Some((center, radius)) = outward_midpoint_radius(input.lower(), input.upper()) else {
            return self.fallback_output_bounds(shape);
        };

        let mut out_lower = input.lower().clone();
        let mut out_upper = input.upper().clone();

        let batch_size: usize =
            checked_dim_product(&shape[..ndim - 2], "GroupNorm IBP batch dimensions")?;

        for batch_idx in 0..batch_size.max(1) {
            let batch_prefix = compute_batch_prefix(shape, ndim, batch_idx);

            for g_idx in 0..self.num_groups {
                let group_start_ch = g_idx * cpg;

                let mut center_vals = Vec::with_capacity(group_size);
                let mut radius_vals = Vec::with_capacity(group_size);
                for c_offset in 0..cpg {
                    let c = group_start_ch + c_offset;
                    for t in 0..time_len {
                        let mut idx = batch_prefix.clone();
                        idx.push(c);
                        idx.push(t);
                        center_vals.push(center[idx.as_slice()]);
                        radius_vals.push(radius[idx.as_slice()]);
                    }
                }

                let (grp_lower, grp_upper) =
                    self.ibp_forward_mode_group(&center_vals, &radius_vals, g_idx);

                for s in 0..group_size {
                    let c = group_start_ch + (s / time_len);
                    let t = s % time_len;
                    let mut idx = batch_prefix.clone();
                    idx.push(c);
                    idx.push(t);
                    out_lower[idx.as_slice()] = grp_lower[s];
                    out_upper[idx.as_slice()] = grp_upper[s];
                }
            }
        }

        if out_lower.iter().any(|v| v.is_nan()) || out_upper.iter().any(|v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "GroupNorm forward-mode IBP: NaN in computed bounds".to_string(),
            ));
        }

        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }
}
