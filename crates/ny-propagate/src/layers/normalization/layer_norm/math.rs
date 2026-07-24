// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concrete evaluation and Jacobian for LayerNorm.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};

use super::types::{LayerNormLayer, LayerNormMode};

impl LayerNormLayer {
    /// Evaluate LayerNorm at a concrete point.
    ///
    /// Returns ny * (x - mean(x)) / std(x) + beta
    pub fn eval(&self, x: &Array1<f32>) -> Result<Array1<f32>> {
        if x.is_empty() {
            return Err(NyError::InternalError(
                "LayerNorm eval: empty tensor input".to_string(),
            ));
        }
        if x.len() > (1 << 24) {
            // f32 precision guard (#2136)
            return Err(NyError::InternalError(format!(
                "LayerNorm dimension {} exceeds f32 exact integer range",
                x.len()
            )));
        }
        // f64 accumulation to avoid catastrophic cancellation (#3325)
        let n = x.len() as f64;
        let mean = x.iter().map(|&xi| xi as f64).sum::<f64>() / n;
        Ok(match self.mode {
            LayerNormMode::Standard => {
                let var = x
                    .iter()
                    .map(|&xi| {
                        let d = xi as f64 - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / n;
                let std = (var + self.eps as f64).sqrt();
                x.iter()
                    .zip(self.ny.iter())
                    .zip(self.beta.iter())
                    .map(|((&xi, &g), &b)| (g as f64 * (xi as f64 - mean) / std + b as f64) as f32)
                    .collect()
            }
            LayerNormMode::MeanOnly => x
                .iter()
                .zip(self.ny.iter())
                .zip(self.beta.iter())
                .map(|((&xi, &g), &b)| (g as f64 * (xi as f64 - mean) + b as f64) as f32)
                .collect(),
        })
    }

    /// Compute the Jacobian of LayerNorm at a point.
    ///
    /// For LayerNorm: y_i = ny_i * (x_i - mean(x)) / std(x) + beta_i
    /// The Jacobian entry `J[i,j]` = ∂y_i/∂x_j:
    ///   `J[i,j]` = ny_i / std * \[δ_ij - 1/n - z_i * z_j / n\]
    /// where z_i = (x_i - mean) / std is the normalized value.
    pub fn jacobian(&self, x: &Array1<f32>) -> Result<Array2<f32>> {
        if x.is_empty() {
            return Err(NyError::InternalError(
                "LayerNorm jacobian: empty tensor input".to_string(),
            ));
        }
        let n = x.len();
        // f64 accumulation to avoid catastrophic cancellation in Jacobian (#3325)
        let nf = n as f64;

        let mean = x.iter().map(|&xi| xi as f64).sum::<f64>() / nf;
        Ok(match self.mode {
            LayerNormMode::Standard => {
                let var = x
                    .iter()
                    .map(|&xi| {
                        let d = xi as f64 - mean;
                        d * d
                    })
                    .sum::<f64>()
                    / nf;
                let std = (var + self.eps as f64).sqrt();

                // Compute normalized values z_i in f64
                let z: Vec<f64> = x.iter().map(|&xi| (xi as f64 - mean) / std).collect();

                // Build Jacobian matrix
                let mut jacobian = Array2::<f32>::zeros((n, n));

                for i in 0..n {
                    let gi = self.ny[i] as f64;
                    for j in 0..n {
                        let delta_ij: f64 = if i == j { 1.0 } else { 0.0 };
                        // J[i,j] = ny_i / std * [δ_ij - 1/n - z_i * z_j / n]
                        jacobian[[i, j]] =
                            (gi / std * (delta_ij - 1.0 / nf - z[i] * z[j] / nf)) as f32;
                    }
                }

                jacobian
            }
            LayerNormMode::MeanOnly => {
                let mut jacobian = Array2::<f32>::zeros((n, n));
                for i in 0..n {
                    let gi = self.ny[i] as f64;
                    for j in 0..n {
                        let delta_ij: f64 = if i == j { 1.0 } else { 0.0 };
                        // J[i,j] = ny_i * (δ_ij - 1/n)
                        jacobian[[i, j]] = (gi * (delta_ij - 1.0 / nf)) as f32;
                    }
                }
                jacobian
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    fn default_ln(n: usize) -> LayerNormLayer {
        LayerNormLayer::new_default(n, 1e-5).unwrap()
    }

    fn custom_ln(ny: &[f32], beta: &[f32]) -> LayerNormLayer {
        LayerNormLayer::new(
            Array1::from_vec(ny.to_vec()),
            Array1::from_vec(beta.to_vec()),
            1e-5,
        )
        .unwrap()
    }

    // ── eval: standard mode ────────────────────────────────────────────

    #[test]
    fn test_eval_constant_input_gives_beta() {
        let ln = custom_ln(&[2.0, 3.0, 4.0], &[10.0, 20.0, 30.0]);
        let x = arr1(&[5.0, 5.0, 5.0]);
        let y = ln.eval(&x).unwrap();
        // All inputs equal → (x - mean) = 0 → y = beta
        assert!((y[0] - 10.0).abs() < 1e-3);
        assert!((y[1] - 20.0).abs() < 1e-3);
        assert!((y[2] - 30.0).abs() < 1e-3);
    }

    #[test]
    fn test_eval_output_mean_approx_zero() {
        // ny=1, beta=0 → output mean ≈ 0
        let ln = default_ln(3);
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = ln.eval(&x).unwrap();
        let y_mean = y.mean().unwrap();
        assert!(y_mean.abs() < 1e-5, "Expected mean ≈ 0, got {}", y_mean);
    }

    #[test]
    fn test_eval_output_unit_variance() {
        // ny=1, beta=0 → output variance ≈ 1
        let ln = default_ln(4);
        let x = arr1(&[1.0, 3.0, 5.0, 7.0]);
        let y = ln.eval(&x).unwrap();
        let n = y.len() as f32;
        let y_mean = y.mean().unwrap();
        let y_var = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f32>() / n;
        assert!(
            (y_var - 1.0).abs() < 1e-5,
            "Expected var ≈ 1, got {}",
            y_var
        );
    }

    #[test]
    fn test_eval_ny_scales_output() {
        let ln2 = custom_ln(&[2.0, 2.0, 2.0], &[0.0, 0.0, 0.0]);
        let ln1 = default_ln(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let y2 = ln2.eval(&x).unwrap();
        let y1 = ln1.eval(&x).unwrap();
        for i in 0..3 {
            assert!((y2[i] - 2.0 * y1[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_eval_beta_shifts_output() {
        let ln = custom_ln(&[1.0, 1.0, 1.0], &[10.0, 20.0, 30.0]);
        let ln0 = default_ln(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let y = ln.eval(&x).unwrap();
        let y0 = ln0.eval(&x).unwrap();
        assert!((y[0] - y0[0] - 10.0).abs() < 1e-5);
        assert!((y[1] - y0[1] - 20.0).abs() < 1e-5);
        assert!((y[2] - y0[2] - 30.0).abs() < 1e-5);
    }

    // ── eval: mean-only mode ───────────────────────────────────────────

    #[test]
    fn test_eval_mean_only_mode() {
        let ln = default_ln(3).with_mode(LayerNormMode::MeanOnly);
        let x = arr1(&[1.0, 2.0, 3.0]);
        let y = ln.eval(&x).unwrap();
        // mean = 2.0, ny=1, beta=0: y_i = x_i - 2
        assert!((y[0] - (-1.0)).abs() < 1e-5);
        assert!((y[1] - 0.0).abs() < 1e-5);
        assert!((y[2] - 1.0).abs() < 1e-5);
    }

    // ── jacobian: standard mode ────────────────────────────────────────

    #[test]
    fn test_jacobian_shape() {
        let ln = default_ln(4);
        let x = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let j = ln.jacobian(&x).unwrap();
        assert_eq!(j.shape(), &[4, 4]);
    }

    #[test]
    fn test_jacobian_rows_sum_to_zero() {
        // LayerNorm is shift-invariant → each Jacobian row sums to 0.
        let ln = default_ln(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let j = ln.jacobian(&x).unwrap();
        for i in 0..3 {
            let row_sum: f32 = (0..3).map(|c| j[[i, c]]).sum();
            assert!(row_sum.abs() < 1e-5, "Row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_jacobian_finite_differences() {
        let ln = default_ln(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let j = ln.jacobian(&x).unwrap();
        let h = 1e-3;

        for jj in 0..3 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[jj] += h;
            xm[jj] -= h;
            let yp = ln.eval(&xp).unwrap();
            let ym = ln.eval(&xm).unwrap();

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
        let ln2 = custom_ln(&[2.0, 2.0, 2.0], &[0.0, 0.0, 0.0]);
        let ln1 = default_ln(3);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let j2 = ln2.jacobian(&x).unwrap();
        let j1 = ln1.jacobian(&x).unwrap();
        for i in 0..3 {
            for jj in 0..3 {
                assert!((j2[[i, jj]] - 2.0 * j1[[i, jj]]).abs() < 1e-5);
            }
        }
    }

    // ── jacobian: mean-only mode ───────────────────────────────────────

    #[test]
    fn test_jacobian_mean_only_known_values() {
        let ln = default_ln(3).with_mode(LayerNormMode::MeanOnly);
        let x = arr1(&[1.0, 2.0, 3.0]);
        let j = ln.jacobian(&x).unwrap();
        // ny=1: J[i,j] = δ_ij - 1/3
        for i in 0..3 {
            for jj in 0..3 {
                let expected = if i == jj { 2.0 / 3.0 } else { -1.0 / 3.0 };
                assert!(
                    (j[[i, jj]] - expected).abs() < 1e-5,
                    "J[{},{}] = {}, expected {}",
                    i,
                    jj,
                    j[[i, jj]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_jacobian_mean_only_finite_differences() {
        let ln = custom_ln(&[2.0, 1.5, 0.5], &[1.0, 2.0, 3.0]).with_mode(LayerNormMode::MeanOnly);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let j = ln.jacobian(&x).unwrap();
        let h = 1e-3;

        for jj in 0..3 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[jj] += h;
            xm[jj] -= h;
            let yp = ln.eval(&xp).unwrap();
            let ym = ln.eval(&xm).unwrap();

            for i in 0..3 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (j[[i, jj]] - fd).abs() < 1e-3,
                    "Mean-only J[{},{}] = {}, fd = {}",
                    i,
                    jj,
                    j[[i, jj]],
                    fd
                );
            }
        }
    }
}
