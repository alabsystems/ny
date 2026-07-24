// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concrete evaluation and Jacobian for RMSNorm.
//!
//! RMSNorm: y_i = ny_i * x_i / sqrt(mean(x^2) + eps)
//!
//! Jacobian derivation (from designs/2026-02-26-p0-normalization-layers-rmsnorm-instancenorm-adain.md):
//!   Let s = sum(x^2) / n + eps, rms = sqrt(s)
//!   drms/dx_j = x_j / (n * rms)
//!   dy_i/dx_j = ny_i * [delta_ij * rms - x_i * x_j / (n * rms)] / rms^2
//!             = ny_i * [delta_ij / rms - x_i * x_j / (n * rms^3)]

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};

use super::types::RmsNormLayer;

impl RmsNormLayer {
    /// Evaluate RMSNorm at a concrete point.
    ///
    /// Returns ny * x / sqrt(mean(x^2) + eps)
    pub fn eval(&self, x: &Array1<f32>) -> Result<Array1<f32>> {
        if x.is_empty() {
            return Err(NyError::InternalError(
                "RMSNorm eval: empty tensor input".to_string(),
            ));
        }
        if x.len() > (1 << 24) {
            // f32 precision guard (#2136)
            return Err(NyError::InternalError(format!(
                "RMSNorm dimension {} exceeds f32 exact integer range",
                x.len()
            )));
        }
        // f64 accumulation to avoid catastrophic cancellation (#3325)
        let n = x.len() as f64;
        let mean_sq = x.iter().map(|&xi| (xi as f64) * (xi as f64)).sum::<f64>() / n;
        let rms = (mean_sq + self.eps as f64).sqrt();

        Ok(x.iter()
            .zip(self.ny.iter())
            .map(|(&xi, &g)| (g as f64 * xi as f64 / rms) as f32)
            .collect())
    }

    /// Compute the Jacobian of RMSNorm at a point.
    ///
    /// For RMSNorm: y_i = ny_i * x_i / rms
    /// where rms = sqrt(mean(x^2) + eps)
    ///
    /// J[i,j] = ny_i * [delta_ij / rms - x_i * x_j / (n * rms^3)]
    ///
    /// This is simpler than LayerNorm's Jacobian because there is no
    /// mean-subtraction term (-1/n from the mean gradient).
    pub fn jacobian(&self, x: &Array1<f32>) -> Result<Array2<f32>> {
        if x.is_empty() {
            return Err(NyError::InternalError(
                "RMSNorm jacobian: empty tensor input".to_string(),
            ));
        }
        let n = x.len();
        // f64 accumulation to avoid catastrophic cancellation in Jacobian (#3325)
        let nf = n as f64;

        let mean_sq = x.iter().map(|&xi| (xi as f64) * (xi as f64)).sum::<f64>() / nf;
        let rms = (mean_sq + self.eps as f64).sqrt();
        let rms_cubed = rms * rms * rms;

        let mut jacobian = Array2::<f32>::zeros((n, n));

        for i in 0..n {
            let xi = x[i] as f64;
            let gi = self.ny[i] as f64;
            for j in 0..n {
                let xj = x[j] as f64;
                let delta_ij: f64 = if i == j { 1.0 } else { 0.0 };
                // J[i,j] = ny_i * [delta_ij / rms - x_i * x_j / (n * rms^3)]
                jacobian[[i, j]] = (gi * (delta_ij / rms - xi * xj / (nf * rms_cubed))) as f32;
            }
        }

        Ok(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn default_rn(n: usize) -> RmsNormLayer {
        RmsNormLayer::new_default(n, 1e-5).unwrap()
    }

    fn custom_rn(ny: &[f32]) -> RmsNormLayer {
        RmsNormLayer::new(Array1::from_vec(ny.to_vec()), 1e-5).unwrap()
    }

    // -- eval tests --

    #[test]
    fn test_eval_known_values() {
        // x = [3, 4], mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5 + 1e-5) ~ 3.5355
        // ny = 1: y_i = x_i / rms
        let rn = default_rn(2);
        let x = arr1(&[3.0, 4.0]);
        let y = rn.eval(&x).unwrap();
        let rms = (12.5_f32 + 1e-5).sqrt();
        assert!((y[0] - 3.0 / rms).abs() < 1e-5);
        assert!((y[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn test_eval_zero_input() {
        // x = [0, 0, 0] -> rms = sqrt(eps), y_i = 0
        let rn = default_rn(3);
        let x = arr1(&[0.0, 0.0, 0.0]);
        let y = rn.eval(&x).unwrap();
        for &yi in y.iter() {
            assert!(yi.abs() < 1e-3, "Expected ~0, got {}", yi);
        }
    }

    #[test]
    fn test_eval_ny_scales_output() {
        let rn2 = custom_rn(&[2.0, 2.0, 2.0]);
        let rn1 = default_rn(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let y2 = rn2.eval(&x).unwrap();
        let y1 = rn1.eval(&x).unwrap();
        for i in 0..3 {
            assert!((y2[i] - 2.0 * y1[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_eval_empty_input_returns_error() {
        let rn = default_rn(0);
        let x = arr1(&[] as &[f32]);
        assert!(rn.eval(&x).is_err());
    }

    // -- jacobian tests --

    #[test]
    fn test_jacobian_shape() {
        let rn = default_rn(4);
        let x = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let j = rn.jacobian(&x).unwrap();
        assert_eq!(j.shape(), &[4, 4]);
    }

    #[test]
    fn test_jacobian_finite_differences() {
        let rn = default_rn(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let j = rn.jacobian(&x).unwrap();
        let h = 1e-3;

        for jj in 0..3 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[jj] += h;
            xm[jj] -= h;
            let yp = rn.eval(&xp).unwrap();
            let ym = rn.eval(&xm).unwrap();

            for i in 0..3 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (j[[i, jj]] - fd).abs() < 1e-3,
                    "J[{},{}] = {}, fd = {}",
                    i,
                    jj,
                    j[[i, jj]],
                    fd
                );
            }
        }
    }

    #[test]
    fn test_jacobian_ny_scaling() {
        let rn2 = custom_rn(&[2.0, 2.0, 2.0]);
        let rn1 = default_rn(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let j2 = rn2.jacobian(&x).unwrap();
        let j1 = rn1.jacobian(&x).unwrap();
        for i in 0..3 {
            for jj in 0..3 {
                assert!((j2[[i, jj]] - 2.0 * j1[[i, jj]]).abs() < 1e-5);
            }
        }
    }

    /// Unlike LayerNorm, RMSNorm rows do NOT sum to zero (no mean subtraction).
    /// Instead, sum of row i = ny_i / rms - ny_i * x_i * sum(x) / (n * rms^3)
    #[test]
    fn test_jacobian_rows_do_not_sum_to_zero() {
        let rn = default_rn(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let j = rn.jacobian(&x).unwrap();
        // At least one row should have non-zero sum (unlike LayerNorm)
        let any_nonzero = (0..3).any(|i| {
            let row_sum: f32 = (0..3).map(|c| j[[i, c]]).sum();
            row_sum.abs() > 1e-5
        });
        assert!(
            any_nonzero,
            "RMSNorm Jacobian rows should generally not sum to zero"
        );
    }
}
