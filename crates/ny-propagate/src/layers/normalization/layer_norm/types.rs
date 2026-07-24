// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Type definitions and constructors for LayerNorm.

use ndarray::{Array1, Array2};
use ny_core::Result;

use super::super::trait_norm::NormLayer;
use super::super::validate::validate_norm_eps;

/// Shared CROWN mode for normalization-layer linearization.
///
/// This enum is reused by `LayerNorm`, `RmsNorm`, `GroupNorm`,
/// `InstanceNorm1d`, and `AdaIN1d`. `BatchNorm` does not use this enum because
/// its affine form has a separate exact CROWN path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerNormCrownMode {
    /// Sound normalization CROWN for LayerNorm-family layers.
    ///
    /// Used by `LayerNorm`, `RmsNorm`, `GroupNorm`, `InstanceNorm1d`, and
    /// `AdaIN1d`. This is the default mode for those layers.
    ///
    /// For `LayerNorm`, this is the shared decomposed primitive-chain CROWN
    /// path (`x -> mean -> d -> d^2 -> var -> sqrt -> reciprocal -> d*inv_std
    /// -> ny*norm + beta`) with rowwise validation against fused LayerNorm
    /// IBP. This matches the alpha-beta-CROWN-style decomposition strategy and
    /// is the default sound LayerNorm route.
    ///
    /// Other normalization layers may still use their existing crate-specific
    /// `IbpValidated` implementations until they are migrated to the shared
    /// decomposed path.
    #[default]
    IbpValidated,
    /// Refuse normalization CROWN and require another strategy.
    ///
    /// For `LayerNorm`, `RmsNorm`, `GroupNorm`, `InstanceNorm1d`, and
    /// `AdaIN1d`, this returns `SoundnessRefusal` instead of attempting a
    /// normalization-layer linearization. Use this to force IBP-only or
    /// explicit `Cut` boundaries through those layers.
    Sound,
    /// Use identity relaxation for LayerNorm-family layers.
    ///
    /// Supported by `LayerNorm`, `RmsNorm`, `GroupNorm`, `InstanceNorm1d`, and
    /// `AdaIN1d`. This preserves upstream CROWN correlations but skips the
    /// normalization transform itself.
    Cut,
    /// Use heuristic sampling-based normalization CROWN.
    ///
    /// Supported by `LayerNorm`, `RmsNorm`, `GroupNorm`, `InstanceNorm1d`, and
    /// `AdaIN1d`. Uses local linearization verified by sampling, with optional
    /// IBP validation for tighter margins, but is not provably sound.
    Sampling,
}

/// Mode for LayerNorm normalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerNormMode {
    /// Standard LayerNorm: subtract mean and divide by std.
    #[default]
    Standard,
    /// Mean-only LayerNorm: subtract mean, skip variance normalization.
    MeanOnly,
}

impl LayerNormMode {
    /// Parse the supported string aliases for LayerNorm mode selection.
    #[must_use]
    pub fn parse_alias(value: &str) -> Option<Self> {
        match value.to_ascii_lowercase().as_str() {
            "mean_only" | "mean-only" | "meanonly" | "mean" | "deept" => Some(Self::MeanOnly),
            "standard" | "full" | "default" => Some(Self::Standard),
            _ => None,
        }
    }
}

/// LayerNorm layer: y = ny * (x - mean) / sqrt(var + eps) + beta
///
/// Normalizes inputs across the last dimension (or specified normalized_shape).
#[derive(Debug, Clone)]
pub struct LayerNormLayer {
    /// Scale parameter (ny)
    pub ny: Array1<f32>,
    /// Shift parameter (beta)
    pub beta: Array1<f32>,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Use forward mode for IBP: compute mean/std from center point (midpoint of bounds)
    /// instead of computing uncertain bounds on mean/std. This dramatically reduces
    /// bound explosion but may not be perfectly sound for large perturbations.
    /// Default: false (use conservative IBP)
    pub forward_mode: bool,
    /// CROWN linearization mode.
    /// Default: IbpValidated (for LayerNorm, shared decomposed CROWN with fused-IBP validation).
    pub crown_mode: LayerNormCrownMode,
    /// LayerNorm mode: Standard or MeanOnly.
    pub mode: LayerNormMode,
}

impl LayerNormLayer {
    /// Create a new LayerNorm layer.
    ///
    /// Returns an error if eps is negative, NaN, or infinite.
    pub fn new(ny: Array1<f32>, beta: Array1<f32>, eps: f32) -> Result<Self> {
        Ok(Self {
            ny,
            beta,
            eps: validate_norm_eps(eps, "LayerNorm")?,
            forward_mode: false,
            crown_mode: LayerNormCrownMode::default(),
            mode: LayerNormMode::Standard,
        })
    }

    /// Create a LayerNorm layer with default ny=1 and beta=0.
    ///
    /// Returns an error if eps is negative, NaN, or infinite.
    pub fn new_default(size: usize, eps: f32) -> Result<Self> {
        Ok(Self {
            ny: Array1::ones(size),
            beta: Array1::zeros(size),
            eps: validate_norm_eps(eps, "LayerNorm")?,
            forward_mode: false,
            crown_mode: LayerNormCrownMode::default(),
            mode: LayerNormMode::Standard,
        })
    }

    /// Create a LayerNorm layer with forward mode enabled (tighter but approximate bounds).
    ///
    /// Returns an error if eps is negative, NaN, or infinite.
    pub fn new_forward_mode(ny: Array1<f32>, beta: Array1<f32>, eps: f32) -> Result<Self> {
        Ok(Self {
            ny,
            beta,
            eps: validate_norm_eps(eps, "LayerNorm")?,
            forward_mode: true,
            crown_mode: LayerNormCrownMode::default(),
            mode: LayerNormMode::Standard,
        })
    }

    /// Enable or disable forward mode.
    ///
    /// Forward mode uses mean/std computed from the center (midpoint) of input bounds
    /// instead of computing uncertain bounds. This dramatically reduces bound explosion
    /// but may not be perfectly sound for large perturbations.
    pub fn with_forward_mode(mut self, enabled: bool) -> Self {
        self.forward_mode = enabled;
        self
    }

    /// Set the CROWN linearization mode.
    ///
    /// `LayerNormCrownMode` is shared across `LayerNorm`, `RmsNorm`,
    /// `GroupNorm`, `InstanceNorm1d`, and `AdaIN1d`.
    ///
    /// - `IbpValidated` (default): for LayerNorm, shared decomposed primitive-chain CROWN
    ///   with fused-IBP row fallback
    /// - `Sound`: Return error if normalization CROWN is attempted
    /// - `Cut`: Identity relaxation (preserves upstream correlations, skips norm)
    /// - `Sampling`: Heuristic sampling-based linearization (optional IBP validation)
    pub fn with_crown_mode(mut self, mode: LayerNormCrownMode) -> Self {
        self.crown_mode = mode;
        self
    }

    /// Set the LayerNorm mode (standard or mean-only).
    pub fn with_mode(mut self, mode: LayerNormMode) -> Self {
        self.mode = mode;
        self
    }
}

impl NormLayer for LayerNormLayer {
    fn layer_name(&self) -> &'static str {
        "LayerNorm"
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
