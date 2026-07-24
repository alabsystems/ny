// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Type definitions and constructors for AdaIN1d.

use ndarray::{Array1, Array2};
use ny_core::{NyError, Result};

use super::super::instance_norm::InstanceNorm1dLayer;
use super::super::layer_norm::types::LayerNormCrownMode;
use super::super::trait_norm::NormLayer;

/// Style parameter source for AdaIN1d.
///
/// - `Fixed`: style_gamma and style_beta are embedded constants (from ONNX initializers).
///   This is the original `#3912` unary surface.
/// - `Variable`: style_gamma and style_beta come from preceding graph layers as bounded
///   intervals. The AdaIN node becomes ternary: `(x, style_gamma, style_beta)`.
#[derive(Debug, Clone)]
pub enum AdaINStyleMode {
    /// Style parameters are known constants, shape `[C]` each.
    /// Boxed to reduce enum size (clippy::large_enum_variant).
    Fixed(Box<FixedStyleParams>),
    /// Style parameters arrive as graph inputs (bounded intervals at propagation time).
    Variable,
}

/// Embedded style parameters for fixed-style AdaIN1d.
#[derive(Debug, Clone)]
pub struct FixedStyleParams {
    pub style_gamma: Array1<f32>,
    pub style_beta: Array1<f32>,
}

/// AdaIN1d layer: y = style_gamma * InstanceNorm(x) + style_beta
///
/// Adaptive Instance Normalization applies per-channel style conditioning
/// on top of instance normalization. For input shape `[C, T]` (or `[B, C, T]`):
///
/// 1. Normalize x per-channel: z = InstanceNorm(x)
/// 2. Scale by style: y = style_gamma[c] * z[c, t] + style_beta[c]
///
/// At inference time with fixed style parameters, this is equivalent to
/// InstanceNorm with modified ny/beta: `effective_gamma = style_gamma * IN.ny`,
/// `effective_beta = style_gamma * IN.beta + style_beta`.
///
/// When style parameters come from preceding graph layers (variable-style),
/// the layer is ternary and computes bounds over the joint `(x, g, b)` surface.
///
/// Reference: Huang & Belongie, "Arbitrary Style Transfer in Real-time with
/// Adaptive Instance Normalization," ICCV 2017.
///
/// Used in avoice kernels K3 (AdaIN vocoder), K4 (Snake+AdaIN pipeline).
#[derive(Debug, Clone)]
pub struct AdaIN1dLayer {
    /// The underlying InstanceNorm layer (handles normalization)
    pub instance_norm: InstanceNorm1dLayer,
    /// Style parameter source: fixed constants or variable graph inputs.
    pub style_mode: AdaINStyleMode,
}

impl AdaIN1dLayer {
    /// Create a new fixed-style AdaIN1d layer.
    ///
    /// `instance_norm` is the underlying InstanceNorm layer (with its own ny/beta).
    /// `style_gamma` and `style_beta` are per-channel style parameters, shape `[num_channels]`.
    pub fn new(
        instance_norm: InstanceNorm1dLayer,
        style_gamma: Array1<f32>,
        style_beta: Array1<f32>,
    ) -> Result<Self> {
        let num_channels = instance_norm.num_channels();
        if style_gamma.len() != num_channels {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_channels],
                got: vec![style_gamma.len()],
            });
        }
        if style_beta.len() != num_channels {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_channels],
                got: vec![style_beta.len()],
            });
        }
        if style_gamma.iter().any(|v| !v.is_finite()) {
            return Err(NyError::InvalidSpec(
                "AdaIN1d style_gamma contains non-finite values".to_string(),
            ));
        }
        if style_beta.iter().any(|v| !v.is_finite()) {
            return Err(NyError::InvalidSpec(
                "AdaIN1d style_beta contains non-finite values".to_string(),
            ));
        }
        Ok(Self {
            instance_norm,
            style_mode: AdaINStyleMode::Fixed(Box::new(FixedStyleParams {
                style_gamma,
                style_beta,
            })),
        })
    }

    /// Create an AdaIN1d layer with identity style (ny=1, beta=0).
    ///
    /// This is equivalent to plain InstanceNorm — useful for testing.
    pub fn new_identity_style(instance_norm: InstanceNorm1dLayer) -> Result<Self> {
        let num_channels = instance_norm.num_channels();
        Self::new(
            instance_norm,
            Array1::ones(num_channels),
            Array1::zeros(num_channels),
        )
    }

    /// Create a variable-style AdaIN1d layer.
    ///
    /// Style parameters come from preceding graph layers as bounded intervals.
    /// The layer becomes ternary: `(x, style_gamma, style_beta)`.
    pub fn variable_style(instance_norm: InstanceNorm1dLayer) -> Result<Self> {
        Ok(Self {
            instance_norm,
            style_mode: AdaINStyleMode::Variable,
        })
    }

    /// Whether this layer requires style inputs from the graph (variable-style).
    ///
    /// When true, the layer is ternary and expects 3 graph inputs.
    /// When false, style parameters are embedded constants (unary).
    pub fn requires_style_inputs(&self) -> bool {
        matches!(self.style_mode, AdaINStyleMode::Variable)
    }

    /// Per-channel style scale (fixed-style only).
    pub fn style_gamma(&self) -> Result<&Array1<f32>> {
        match &self.style_mode {
            AdaINStyleMode::Fixed(params) => Ok(&params.style_gamma),
            AdaINStyleMode::Variable => Err(NyError::UnsupportedOp(
                "AdaIN1d variable-style has no embedded style_gamma".to_string(),
            )),
        }
    }

    /// Per-channel style shift (fixed-style only).
    pub fn style_beta(&self) -> Result<&Array1<f32>> {
        match &self.style_mode {
            AdaINStyleMode::Fixed(params) => Ok(&params.style_beta),
            AdaINStyleMode::Variable => Err(NyError::UnsupportedOp(
                "AdaIN1d variable-style has no embedded style_beta".to_string(),
            )),
        }
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.instance_norm.num_channels()
    }

    /// Enable or disable forward mode on the inner InstanceNorm.
    pub fn with_forward_mode(mut self, enabled: bool) -> Self {
        self.instance_norm = self.instance_norm.with_forward_mode(enabled);
        self
    }

    /// Set the CROWN linearization mode on the inner InstanceNorm.
    pub fn with_crown_mode(mut self, mode: LayerNormCrownMode) -> Self {
        self.instance_norm = self.instance_norm.with_crown_mode(mode);
        self
    }

    /// Materialize the fixed-style `InstanceNorm1d` equivalent used by
    /// `IbpValidated` CROWN routing.
    ///
    /// Only valid for fixed-style AdaIN. Returns error for variable-style.
    pub(crate) fn effective_instance_norm(&self) -> Result<InstanceNorm1dLayer> {
        let params = match &self.style_mode {
            AdaINStyleMode::Fixed(params) => params,
            AdaINStyleMode::Variable => {
                return Err(NyError::UnsupportedOp(
                    "AdaIN1d variable-style cannot collapse to effective InstanceNorm".to_string(),
                ));
            }
        };
        let (style_gamma, style_beta) = (&params.style_gamma, &params.style_beta);

        let effective_gamma = style_gamma * &self.instance_norm.ny;
        let effective_beta = (style_gamma * &self.instance_norm.beta) + style_beta;

        if effective_gamma.iter().any(|value| !value.is_finite()) {
            return Err(NyError::InvalidSpec(
                "AdaIN1d effective InstanceNorm ny contains non-finite values".to_string(),
            ));
        }
        if effective_beta.iter().any(|value| !value.is_finite()) {
            return Err(NyError::InvalidSpec(
                "AdaIN1d effective InstanceNorm beta contains non-finite values".to_string(),
            ));
        }

        Ok(
            InstanceNorm1dLayer::new(effective_gamma, effective_beta, self.instance_norm.eps)?
                .with_forward_mode(self.instance_norm.forward_mode)
                .with_crown_mode(self.instance_norm.crown_mode),
        )
    }
}

impl NormLayer for AdaIN1dLayer {
    fn layer_name(&self) -> &'static str {
        "AdaIN1d"
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
        self.instance_norm.crown_mode
    }
}
