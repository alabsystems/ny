// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concrete evaluation and Jacobian for GroupNorm.
//!
//! GroupNorm normalizes each group of channels independently.
//! For a group with `cpg` channels and `T` time steps, the group has
//! `n = cpg * T` elements. Normalization:
//!   y[c_offset, t] = ny[c] * (x[c_offset, t] - mean_g) / sqrt(var_g + eps) + beta[c]
//! where mean_g and var_g are over all n elements in the group.
//!
//! The Jacobian has the same structure as LayerNorm applied to the group:
//!   J[i, j] = ny[c_i] / std * (delta_ij - 1/n - z_i * z_j / n)
//! where c_i is the channel of element i and z_i = (x_i - mean)/std.

use ndarray::Array2;
use ny_core::{NyError, Result};

use super::types::GroupNormLayer;

impl GroupNormLayer {
    /// Evaluate GroupNorm for one group.
    ///
    /// `group_vals` has length `cpg * time_len`, laid out as
    /// [c0_t0, c0_t1, ..., c0_tT, c1_t0, ...].
    /// Returns output of same length with per-element ny/beta applied.
    pub(crate) fn eval_group(
        &self,
        group_vals: &[f32],
        group_idx: usize,
        cpg: usize,
        time_len: usize,
    ) -> Result<Vec<f32>> {
        let n = group_vals.len();
        if n == 0 {
            return Err(NyError::InternalError(
                "GroupNorm eval: empty group".to_string(),
            ));
        }
        if n > (1 << 24) {
            return Err(NyError::InternalError(format!(
                "GroupNorm group size {n} exceeds f32 exact integer range"
            )));
        }

        // f64 accumulation to avoid catastrophic cancellation (#3325)
        let nf = n as f64;
        let mean = group_vals.iter().map(|&xi| xi as f64).sum::<f64>() / nf;
        let var = group_vals
            .iter()
            .map(|&xi| {
                let d = xi as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / nf;
        let std = (var + self.eps as f64).sqrt();

        let group_start_ch = group_idx * cpg;
        let mut output = vec![0.0_f32; n];
        for c_offset in 0..cpg {
            let c = group_start_ch + c_offset;
            let g = self.ny[c] as f64;
            let b = self.beta[c] as f64;
            for t in 0..time_len {
                let idx = c_offset * time_len + t;
                output[idx] = (g * (group_vals[idx] as f64 - mean) / std + b) as f32;
            }
        }
        Ok(output)
    }

    /// Compute the Jacobian of GroupNorm for one group.
    ///
    /// For a group with n = cpg * T elements:
    ///   J[i, j] = ny[c_i] / std * (delta_ij - 1/n - z_i * z_j / n)
    /// where z_i = (x_i - mean) / std.
    ///
    /// This is the LayerNorm Jacobian formula applied to the group, but with
    /// per-element ny based on the channel index of element i.
    pub(crate) fn jacobian_group(
        &self,
        group_vals: &[f32],
        group_idx: usize,
        cpg: usize,
        time_len: usize,
    ) -> Result<Array2<f32>> {
        let n = group_vals.len();
        if n == 0 {
            return Err(NyError::InternalError(
                "GroupNorm jacobian: empty group".to_string(),
            ));
        }

        // f64 accumulation to avoid catastrophic cancellation in Jacobian (#3325)
        let nf = n as f64;
        let mean = group_vals.iter().map(|&xi| xi as f64).sum::<f64>() / nf;
        let var = group_vals
            .iter()
            .map(|&xi| {
                let d = xi as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / nf;
        let std = (var + self.eps as f64).sqrt();

        let z: Vec<f64> = group_vals
            .iter()
            .map(|&xi| (xi as f64 - mean) / std)
            .collect();

        let group_start_ch = group_idx * cpg;
        let mut jacobian = Array2::<f32>::zeros((n, n));

        for i in 0..n {
            let c_i = group_start_ch + i / time_len;
            let g_i = self.ny[c_i] as f64;
            for j in 0..n {
                let delta_ij: f64 = if i == j { 1.0 } else { 0.0 };
                // J[i,j] = ny[c_i] / std * (delta_ij - 1/n - z_i * z_j / n)
                jacobian[[i, j]] = (g_i / std * (delta_ij - 1.0 / nf - z[i] * z[j] / nf)) as f32;
            }
        }

        Ok(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;
    use crate::layers::normalization::trait_norm::NormLayer;

    fn default_gn(c: usize, g: usize) -> GroupNormLayer {
        GroupNormLayer::new_default(c, g, 1e-5).unwrap()
    }

    #[test]
    fn test_eval_group_constant_input_gives_beta() {
        let layer = GroupNormLayer::new(
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]),
            Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0]),
            2, // 2 groups of 2 channels
            1e-5,
        )
        .unwrap();
        // Group 0 has channels [0, 1], time_len=3
        // Constant input → (x - mean) = 0 → y = beta[c]
        let group_vals = vec![5.0; 6]; // 2 channels * 3 time = 6 elements, all 5.0
        let y = layer.eval_group(&group_vals, 0, 2, 3).unwrap();
        // Channel 0: beta[0] = 10.0
        for (t, &val) in y[..3].iter().enumerate() {
            assert!(
                (val - 10.0).abs() < 1e-3,
                "Expected 10.0 at t={t}, got {val}"
            );
        }
        // Channel 1: beta[1] = 20.0
        for (t, &val) in y[3..6].iter().enumerate() {
            assert!(
                (val - 20.0).abs() < 1e-3,
                "Expected 20.0 at t={t}, got {val}"
            );
        }
    }

    #[test]
    fn test_eval_group_zero_mean() {
        let layer = default_gn(4, 2);
        let group_vals = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]; // 2ch * 3t
        let y = layer.eval_group(&group_vals, 0, 2, 3).unwrap();
        let y_mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
        assert!(
            y_mean.abs() < 1e-4,
            "Expected group output mean ≈ 0, got {y_mean}"
        );
    }

    #[test]
    fn test_jacobian_group_shape() {
        let layer = default_gn(4, 2);
        let group_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2ch * 3t = 6
        let j = layer.jacobian_group(&group_vals, 0, 2, 3).unwrap();
        assert_eq!(j.shape(), &[6, 6]);
    }

    #[test]
    fn test_jacobian_group_rows_sum_to_zero() {
        // Since ny is uniform (1.0), each row of the Jacobian sums to 0:
        // sum_j J[i,j] = ny/std * sum_j (delta_ij - 1/n - z_i*z_j/n)
        //              = ny/std * (1 - 1 - z_i * sum_j z_j / n)
        //              = ny/std * (- z_i * 0) = 0  (since sum z_j = 0)
        let layer = default_gn(4, 2);
        let group_vals = vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0]; // 2ch * 3t
        let j = layer.jacobian_group(&group_vals, 0, 2, 3).unwrap();
        for i in 0..6 {
            let row_sum: f32 = (0..6).map(|c| j[[i, c]]).sum();
            assert!(row_sum.abs() < 1e-5, "Row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_jacobian_group_finite_differences() {
        let layer = default_gn(4, 2);
        let group_vals = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 2ch * 3t
        let j = layer.jacobian_group(&group_vals, 0, 2, 3).unwrap();
        let h = 1e-3;

        for jj in 0..6 {
            let mut xp = group_vals.clone();
            let mut xm = group_vals.clone();
            xp[jj] += h;
            xm[jj] -= h;
            let yp = layer.eval_group(&xp, 0, 2, 3).unwrap();
            let ym = layer.eval_group(&xm, 0, 2, 3).unwrap();

            for i in 0..6 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (j[[i, jj]] - fd).abs() < 1e-3,
                    "J[{i},{jj}] = {}, fd = {fd}",
                    j[[i, jj]]
                );
            }
        }
    }

    #[test]
    fn test_instance_norm_is_groupnorm_c_groups() {
        // InstanceNorm is the special case where num_groups = C.
        // With C=3, T=4, num_groups=3: each group is one channel.
        let layer = default_gn(3, 3); // 3 groups of 1 channel = InstanceNorm
        let x: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5 + 1.0).collect();

        let y = layer.eval(&Array1::from_vec(x.clone())).unwrap();
        let j = layer.jacobian(&Array1::from_vec(x)).unwrap();

        // Each channel should independently have mean ≈ 0
        for c in 0..3 {
            let ch_mean: f32 = (0..4).map(|t| y[c * 4 + t]).sum::<f32>() / 4.0;
            assert!(ch_mean.abs() < 1e-4, "Channel {c} mean = {ch_mean}");
        }

        // Jacobian should be block-diagonal: off-block entries should be 0
        for c1 in 0..3 {
            for c2 in 0..3 {
                if c1 != c2 {
                    for t1 in 0..4 {
                        for t2 in 0..4 {
                            assert!(
                                j[[c1 * 4 + t1, c2 * 4 + t2]].abs() < 1e-10,
                                "Off-block J[{},{},{}{}] = {}",
                                c1,
                                t1,
                                c2,
                                t2,
                                j[[c1 * 4 + t1, c2 * 4 + t2]]
                            );
                        }
                    }
                }
            }
        }
    }
}
