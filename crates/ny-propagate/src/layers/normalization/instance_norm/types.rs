// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Type definitions and constructors for InstanceNorm1d.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};

use super::super::layer_norm::types::LayerNormCrownMode;
use super::super::trait_norm::NormLayer;
use super::super::validate::validate_norm_eps;

/// InstanceNorm1d layer: y[c, t] = ny[c] * (x[c, t] - mean_c) / sqrt(var_c + eps) + beta[c]
///
/// Instance normalization normalizes each channel independently across the time/spatial
/// dimension. For input shape `[C, T]` (or `[B, C, T]`), computes per-channel
/// mean and variance over the T dimension.
///
/// Reference: Ulyanov et al., "Instance Normalization: The Missing Ingredient for
/// Fast Stylization," 2016.
///
/// Used in avoice kernels K2 (standalone), K3 (AdaIN), K4 (Snake+InstanceNorm).
#[derive(Debug, Clone)]
pub struct InstanceNorm1dLayer {
    /// Scale parameter per channel (ny), shape [C]
    pub ny: Array1<f32>,
    /// Shift parameter per channel (beta), shape [C]
    pub beta: Array1<f32>,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Use forward mode for IBP: compute mean/std from center point (midpoint of bounds)
    /// instead of computing uncertain bounds on mean/std.
    ///
    /// This is a heuristic analysis mode and is not admitted as proof
    /// authority; use conservative IBP for certified verification.
    /// Default: false (use conservative IBP)
    pub forward_mode: bool,
    /// CROWN linearization mode.
    /// Default: IbpValidated (sound Jacobian-based linearization with IBP validation).
    /// Reuses LayerNormCrownMode since the mode semantics are identical.
    pub crown_mode: LayerNormCrownMode,
}

impl InstanceNorm1dLayer {
    /// Create a new InstanceNorm1d layer.
    ///
    /// `ny` and `beta` have shape `[num_channels]`.
    /// Returns an error if eps is non-finite or below the supported minimum.
    pub fn new(ny: Array1<f32>, beta: Array1<f32>, eps: f32) -> Result<Self> {
        if ny.len() != beta.len() {
            return Err(NyError::ShapeMismatch {
                expected: vec![ny.len()],
                got: vec![beta.len()],
            });
        }
        Ok(Self {
            ny,
            beta,
            eps: validate_norm_eps(eps, "InstanceNorm1d")?,
            forward_mode: false,
            crown_mode: LayerNormCrownMode::default(),
        })
    }

    /// Create an InstanceNorm1d layer with default ny=1 and beta=0.
    ///
    /// Returns an error if eps is non-finite or below the supported minimum.
    pub fn new_default(num_channels: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            ny: Array1::ones(num_channels),
            beta: Array1::zeros(num_channels),
            eps: validate_norm_eps(eps, "InstanceNorm1d")?,
            forward_mode: false,
            crown_mode: LayerNormCrownMode::default(),
        })
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.ny.len()
    }

    /// Enable or disable forward mode.
    pub fn with_forward_mode(mut self, enabled: bool) -> Self {
        self.forward_mode = enabled;
        self
    }

    /// Set the CROWN linearization mode.
    pub fn with_crown_mode(mut self, mode: LayerNormCrownMode) -> Self {
        self.crown_mode = mode;
        self
    }
}

impl NormLayer for InstanceNorm1dLayer {
    fn layer_name(&self) -> &'static str {
        "InstanceNorm1d"
    }

    fn eval(&self, x: &Array1<f32>) -> Result<Array1<f32>> {
        // Flat eval: split into channels, eval each, concatenate.
        let num_channels = self.num_channels();
        // Explicit zero-channel guard: reject a malformed 0-channel layer
        // cleanly rather than panic on `% 0`.
        if num_channels == 0 || !x.len().is_multiple_of(num_channels) {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_channels],
                got: vec![x.len()],
            });
        }
        let time_len = x.len() / num_channels;
        let mut y = Array1::<f32>::zeros(x.len());
        for c in 0..num_channels {
            let start = c * time_len;
            let end = start + time_len;
            let x_channel = x.slice(ndarray::s![start..end]).to_owned();
            let y_channel = self.eval_channel(&x_channel, c)?;
            for t in 0..time_len {
                y[start + t] = y_channel[t];
            }
        }
        Ok(y)
    }

    fn jacobian(&self, x: &Array1<f32>) -> Result<Array2<f32>> {
        // Flat Jacobian: block-diagonal, one T×T block per channel.
        let num_channels = self.num_channels();
        // Explicit zero-channel guard (see `eval`): reject rather than panic on `% 0`.
        if num_channels == 0 || !x.len().is_multiple_of(num_channels) {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_channels],
                got: vec![x.len()],
            });
        }
        let time_len = x.len() / num_channels;
        let total = num_channels * time_len;
        let mut jacobian = Array2::<f32>::zeros((total, total));
        for c in 0..num_channels {
            let start = c * time_len;
            let end = start + time_len;
            let x_channel = x.slice(ndarray::s![start..end]).to_owned();
            let j_channel = self.jacobian_channel(&x_channel, c)?;
            for s in 0..time_len {
                for t in 0..time_len {
                    jacobian[[start + s, start + t]] = j_channel[[s, t]];
                }
            }
        }
        Ok(jacobian)
    }

    fn crown_mode(&self) -> LayerNormCrownMode {
        self.crown_mode
    }
}
