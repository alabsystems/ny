// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Type definitions and constructors for RMSNorm.

use ndarray::{Array1, Array2};
use ny_core::Result;

use super::super::layer_norm::types::LayerNormCrownMode;
use super::super::trait_norm::NormLayer;
use super::super::validate::validate_norm_eps;

/// RMSNorm layer: y_i = ny_i * x_i / sqrt(mean(x^2) + eps)
///
/// Root Mean Square Layer Normalization. Unlike LayerNorm, RMSNorm does not
/// subtract the mean. It normalizes by the root mean square of the input.
/// No beta (bias) parameter by default.
///
/// Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019.
/// Used in modern transformers: LLaMA, Qwen, Mistral.
#[derive(Debug, Clone)]
pub struct RmsNormLayer {
    /// Scale parameter (ny)
    pub ny: Array1<f32>,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Use forward mode for IBP: compute rms from center point (midpoint of bounds)
    /// instead of computing uncertain bounds on rms. This dramatically reduces
    /// bound explosion but may not be perfectly sound for large perturbations.
    /// Default: false (use conservative IBP)
    pub forward_mode: bool,
    /// CROWN linearization mode.
    /// Default: IbpValidated (shared decomposed primitive-chain CROWN matching
    /// alpha-beta-CROWN decomposition). Reuses LayerNormCrownMode since the mode
    /// semantics are identical.
    pub crown_mode: LayerNormCrownMode,
}

impl RmsNormLayer {
    /// Create a new RMSNorm layer.
    ///
    /// Returns an error if eps is negative, NaN, or infinite.
    pub fn new(ny: Array1<f32>, eps: f32) -> Result<Self> {
        Ok(Self {
            ny,
            eps: validate_norm_eps(eps, "RMSNorm")?,
            forward_mode: false,
            crown_mode: LayerNormCrownMode::default(),
        })
    }

    /// Create an RMSNorm layer with default ny=1.
    ///
    /// Returns an error if eps is negative, NaN, or infinite.
    pub fn new_default(size: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            ny: Array1::ones(size),
            eps: validate_norm_eps(eps, "RMSNorm")?,
            forward_mode: false,
            crown_mode: LayerNormCrownMode::default(),
        })
    }

    /// Enable or disable forward mode.
    ///
    /// Forward mode uses rms computed from the center (midpoint) of input bounds
    /// instead of computing uncertain bounds. This dramatically reduces bound explosion
    /// but may not be perfectly sound for large perturbations.
    pub fn with_forward_mode(mut self, enabled: bool) -> Self {
        self.forward_mode = enabled;
        self
    }

    /// Set the CROWN linearization mode.
    ///
    /// - `IbpValidated` (default): shared decomposed primitive-chain CROWN (sound)
    /// - `Sound`: Return error if CROWN linearization is attempted
    /// - `Cut`: Use identity relaxation (sound but loses correlations)
    /// - `Sampling`: Use heuristic sampling-based linearization (NOT provably sound)
    pub fn with_crown_mode(mut self, mode: LayerNormCrownMode) -> Self {
        self.crown_mode = mode;
        self
    }
}

impl NormLayer for RmsNormLayer {
    fn layer_name(&self) -> &'static str {
        "RMSNorm"
    }

    fn eval(&self, x: &Array1<f32>) -> Result<Array1<f32>> {
        // Delegate to the existing eval() on the concrete type.
        self.eval(x)
    }

    fn jacobian(&self, x: &Array1<f32>) -> Result<Array2<f32>> {
        // Delegate to the existing jacobian() on the concrete type.
        self.jacobian(x)
    }

    fn crown_mode(&self) -> LayerNormCrownMode {
        self.crown_mode
    }
}
