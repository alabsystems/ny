// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concrete evaluation for AdaIN1d.
//!
//! AdaIN(x) = style_gamma * InstanceNorm(x) + style_beta
//!
//! The Jacobian of AdaIN w.r.t. x is:
//!   J_adain[c,s; c,t] = style_gamma[c] * J_instnorm[c,s; c,t]
//!
//! (The style parameters are constants, so they simply scale the InstanceNorm Jacobian.)

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};

use super::types::AdaIN1dLayer;

impl AdaIN1dLayer {
    /// Evaluate AdaIN1d for a single channel (1D slice of length T).
    ///
    /// Returns `style_gamma[c] * InstanceNorm(x, c) + style_beta[c]`.
    /// Only valid for fixed-style AdaIN.
    pub fn eval_channel(&self, x: &Array1<f32>, channel: usize) -> Result<Array1<f32>> {
        if channel >= self.num_channels() {
            return Err(NyError::InvalidSpec(format!(
                "AdaIN1d channel {channel} >= num_channels {}",
                self.num_channels()
            )));
        }

        let style_gamma = self.style_gamma()?;
        let style_beta = self.style_beta()?;

        // Step 1: InstanceNorm the channel
        let normed = self.instance_norm.eval_channel(x, channel)?;

        // Step 2: Apply style affine transform
        let sg = style_gamma[channel];
        let sb = style_beta[channel];

        Ok(normed.mapv(|z| sg * z + sb))
    }

    /// Evaluate AdaIN1d for a 2D input [C, T].
    ///
    /// Each row (channel) is independently normalized and then styled.
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
            let styled = self.eval_channel(&channel_data, ch)?;
            result.row_mut(ch).assign(&styled);
        }
        Ok(result)
    }

    /// Compute the Jacobian of AdaIN1d for a single channel.
    ///
    /// The AdaIN Jacobian is simply the InstanceNorm Jacobian scaled by style_gamma:
    ///   `J_adain[s, t]` = style_gamma[c] * J_instnorm[s, t]
    ///
    /// This follows because AdaIN is an affine function of InstanceNorm's output.
    /// Only valid for fixed-style AdaIN.
    pub fn jacobian_channel(&self, x: &Array1<f32>, channel: usize) -> Result<Array2<f32>> {
        if channel >= self.num_channels() {
            return Err(NyError::InvalidSpec(format!(
                "AdaIN1d channel {channel} >= num_channels {}",
                self.num_channels()
            )));
        }

        let style_gamma = self.style_gamma()?;
        let mut j = self.instance_norm.jacobian_channel(x, channel)?;
        let sg = style_gamma[channel];
        j.mapv_inplace(|v| sg * v);
        Ok(j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    use crate::layers::normalization::InstanceNorm1dLayer;

    fn default_adain(c: usize) -> AdaIN1dLayer {
        let inn = InstanceNorm1dLayer::new_default(c, 1e-5).unwrap();
        AdaIN1dLayer::new_identity_style(inn).unwrap()
    }

    fn styled_adain(ny: &[f32], beta: &[f32], sg: &[f32], sb: &[f32]) -> AdaIN1dLayer {
        let inn = InstanceNorm1dLayer::new(
            Array1::from_vec(ny.to_vec()),
            Array1::from_vec(beta.to_vec()),
            1e-5,
        )
        .unwrap();
        AdaIN1dLayer::new(
            inn,
            Array1::from_vec(sg.to_vec()),
            Array1::from_vec(sb.to_vec()),
        )
        .unwrap()
    }

    #[test]
    fn test_identity_style_matches_instance_norm() {
        let inn = InstanceNorm1dLayer::new_default(2, 1e-5).unwrap();
        let adain = AdaIN1dLayer::new_identity_style(inn.clone()).unwrap();
        let x = arr2(&[[1.0, 2.0, 3.0, 4.0], [5.0, 10.0, 15.0, 20.0]]);

        let inn_result = inn.eval_2d(&x).unwrap();
        let adain_result = adain.eval_2d(&x).unwrap();

        for (a, b) in inn_result.iter().zip(adain_result.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Identity style should match InstanceNorm: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_style_scaling() {
        // InstanceNorm with ny=1, beta=0; style_gamma=2, style_beta=10
        let layer = styled_adain(&[1.0], &[0.0], &[2.0], &[10.0]);
        let x = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let y = layer.eval_channel(&x, 0).unwrap();
        // InstanceNorm(x) has zero mean and unit-ish scale
        // AdaIN = 2 * InstanceNorm(x) + 10
        let y_mean = y.mean().unwrap();
        // Mean of 2*z + 10 where mean(z) ≈ 0 should be ≈ 10
        assert!(
            (y_mean - 10.0).abs() < 1e-4,
            "Mean should be ≈ 10 (style_beta), got {y_mean}"
        );
    }

    #[test]
    fn test_jacobian_finite_differences() {
        let layer = styled_adain(&[1.5], &[0.5], &[2.0], &[3.0]);
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

    #[test]
    fn test_jacobian_is_scaled_instance_norm_jacobian() {
        let inn_gamma = &[1.5];
        let inn_beta = &[0.5];
        let sg = &[3.0];
        let sb = &[7.0]; // beta doesn't affect Jacobian

        let inn = InstanceNorm1dLayer::new(
            Array1::from_vec(inn_gamma.to_vec()),
            Array1::from_vec(inn_beta.to_vec()),
            1e-5,
        )
        .unwrap();
        let adain = styled_adain(inn_gamma, inn_beta, sg, sb);

        let x = arr1(&[2.0, 4.0, 6.0, 8.0]);
        let j_inn = inn.jacobian_channel(&x, 0).unwrap();
        let j_adain = adain.jacobian_channel(&x, 0).unwrap();

        for i in 0..4 {
            for j in 0..4 {
                let expected = sg[0] * j_inn[[i, j]];
                assert!(
                    (j_adain[[i, j]] - expected).abs() < 1e-6,
                    "J_adain[{i},{j}] = {} != sg * J_inn = {}",
                    j_adain[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_eval_channel_out_of_range() {
        let layer = default_adain(2);
        let x = arr1(&[1.0, 2.0]);
        assert!(layer.eval_channel(&x, 2).is_err());
    }

    #[test]
    fn test_jacobian_channel_out_of_range() {
        let layer = default_adain(2);
        let x = arr1(&[1.0, 2.0]);
        assert!(layer.jacobian_channel(&x, 2).is_err());
    }
}
