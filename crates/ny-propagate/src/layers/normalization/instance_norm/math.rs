// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concrete evaluation and Jacobian for InstanceNorm1d.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};

use super::types::InstanceNorm1dLayer;

impl InstanceNorm1dLayer {
    /// Evaluate InstanceNorm1d for a single channel (1D slice of length T).
    ///
    /// Returns `ny[c] * (x - mean(x)) / sqrt(var(x) + eps) + beta[c]`
    /// for one channel.
    pub fn eval_channel(&self, x: &Array1<f32>, channel: usize) -> Result<Array1<f32>> {
        let t = x.len();
        if t == 0 {
            return Err(NyError::InternalError(
                "InstanceNorm1d eval: empty time dimension".to_string(),
            ));
        }
        if t > (1 << 24) {
            return Err(NyError::InternalError(format!(
                "InstanceNorm1d time dimension {t} exceeds f32 exact integer range"
            )));
        }
        if channel >= self.num_channels() {
            return Err(NyError::InvalidSpec(format!(
                "InstanceNorm1d channel {channel} >= num_channels {}",
                self.num_channels()
            )));
        }

        // f64 accumulation to avoid catastrophic cancellation (#3325)
        let n = t as f64;
        let mean = x.iter().map(|&xi| xi as f64).sum::<f64>() / n;
        let var = x
            .iter()
            .map(|&xi| {
                let d = xi as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = (var + self.eps as f64).sqrt();

        let g = self.ny[channel] as f64;
        let b = self.beta[channel] as f64;

        Ok(x.mapv(|xi| (g * (xi as f64 - mean) / std + b) as f32))
    }

    /// Evaluate InstanceNorm1d for a 2D input [C, T].
    ///
    /// Each row (channel) is independently normalized over the time dimension.
    pub fn eval_2d(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        let (c, _t) = (x.nrows(), x.ncols());
        if c != self.num_channels() {
            return Err(NyError::ShapeMismatch {
                expected: vec![self.num_channels()],
                got: vec![c],
            });
        }

        let mut result = x.clone();
        for ch in 0..c {
            let channel_data = x.row(ch).to_owned();
            let normed = self.eval_channel(&channel_data, ch)?;
            result.row_mut(ch).assign(&normed);
        }
        Ok(result)
    }

    /// Compute the Jacobian of InstanceNorm1d for a single channel.
    ///
    /// For channel c with time dimension T:
    /// y_t = ny[c] * (x_t - mean(x)) / std(x) + beta[c]
    ///
    /// The Jacobian `J[s, t]` = ∂y_s/∂x_t for the same channel:
    ///   `J[s, t]` = ny[c] / std * [δ_st - 1/T - z_s * z_t / T]
    /// where z_t = (x_t - mean) / std.
    ///
    /// This is the same formula as LayerNorm's Jacobian but applied within a channel.
    pub fn jacobian_channel(&self, x: &Array1<f32>, channel: usize) -> Result<Array2<f32>> {
        let t = x.len();
        if t == 0 {
            return Err(NyError::InternalError(
                "InstanceNorm1d jacobian: empty time dimension".to_string(),
            ));
        }
        if channel >= self.num_channels() {
            return Err(NyError::InvalidSpec(format!(
                "InstanceNorm1d channel {channel} >= num_channels {}",
                self.num_channels()
            )));
        }

        // f64 accumulation to avoid catastrophic cancellation in Jacobian (#3325)
        let nf = t as f64;
        let mean = x.iter().map(|&xi| xi as f64).sum::<f64>() / nf;
        let var = x
            .iter()
            .map(|&xi| {
                let d = xi as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / nf;
        let std = (var + self.eps as f64).sqrt();

        let z: Vec<f64> = x.iter().map(|&xi| (xi as f64 - mean) / std).collect();

        let g = self.ny[channel] as f64;
        let mut jacobian = Array2::<f32>::zeros((t, t));

        for s in 0..t {
            for r in 0..t {
                let delta_st: f64 = if s == r { 1.0 } else { 0.0 };
                // J[s,t] = ny[c] / std * [δ_st - 1/T - z_s * z_t / T]
                jacobian[[s, r]] = (g / std * (delta_st - 1.0 / nf - z[s] * z[r] / nf)) as f32;
            }
        }

        Ok(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn default_in(c: usize) -> InstanceNorm1dLayer {
        InstanceNorm1dLayer::new_default(c, 1e-5).unwrap()
    }

    fn custom_in(ny: &[f32], beta: &[f32]) -> InstanceNorm1dLayer {
        InstanceNorm1dLayer::new(
            Array1::from_vec(ny.to_vec()),
            Array1::from_vec(beta.to_vec()),
            1e-5,
        )
        .unwrap()
    }

    #[test]
    fn test_eval_channel_constant_input_gives_beta() {
        let layer = custom_in(&[2.0, 3.0], &[10.0, 20.0]);
        let x = arr1(&[5.0, 5.0, 5.0, 5.0]); // constant time series
        let y = layer.eval_channel(&x, 0).unwrap();
        // All inputs equal → (x - mean) = 0 → y = beta[0] = 10.0
        for &val in y.iter() {
            assert!((val - 10.0).abs() < 1e-3, "Expected 10.0, got {val}");
        }
    }

    #[test]
    fn test_eval_channel_zero_mean_output() {
        let layer = default_in(1);
        let x = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let y = layer.eval_channel(&x, 0).unwrap();
        let y_mean = y.mean().unwrap();
        assert!(y_mean.abs() < 1e-5, "Expected mean ≈ 0, got {y_mean}");
    }

    #[test]
    fn test_eval_2d_multi_channel() {
        let layer = default_in(2);
        let x = arr2(&[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]);
        let y = layer.eval_2d(&x).unwrap();
        // Both channels should have zero-mean output
        for ch in 0..2 {
            let ch_mean: f32 = y.row(ch).mean().unwrap();
            assert!(ch_mean.abs() < 1e-4, "Channel {ch} mean = {ch_mean}");
        }
    }

    #[test]
    fn test_jacobian_channel_shape() {
        let layer = default_in(1);
        let x = arr1(&[1.0, 2.0, 3.0]);
        let j = layer.jacobian_channel(&x, 0).unwrap();
        assert_eq!(j.shape(), &[3, 3]);
    }

    #[test]
    fn test_jacobian_channel_rows_sum_to_zero() {
        let layer = default_in(1);
        let x = arr1(&[1.0, 3.0, 5.0, 7.0]);
        let j = layer.jacobian_channel(&x, 0).unwrap();
        for i in 0..4 {
            let row_sum: f32 = (0..4).map(|c| j[[i, c]]).sum();
            assert!(row_sum.abs() < 1e-5, "Row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn test_jacobian_channel_finite_differences() {
        let layer = default_in(1);
        let x = arr1(&[1.0, 3.0, 5.0]);
        let j = layer.jacobian_channel(&x, 0).unwrap();
        let h = 1e-3;

        for jj in 0..3 {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[jj] += h;
            xm[jj] -= h;
            let yp = layer.eval_channel(&xp, 0).unwrap();
            let ym = layer.eval_channel(&xm, 0).unwrap();

            for i in 0..3 {
                let fd = (yp[i] - ym[i]) / (2.0 * h);
                assert!(
                    (j[[i, jj]] - fd).abs() < 1e-3,
                    "J[{i},{jj}] = {}, fd = {fd}",
                    j[[i, jj]]
                );
            }
        }
    }
}
