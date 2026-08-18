// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Type definitions and constructors for BatchNorm.

use ndarray::ArrayD;
use ny_core::dd::{next_down_f64, next_up_f64};
use ny_core::{
    f32_to_f64_exact, f64_to_f32_up, require_f64_interval_proof_environment, NyError, Result,
};

/// A finite binary64 enclosure used only while authenticating the exact-real
/// BatchNorm affine encoded by the authored binary32 parameters.
#[derive(Clone, Copy, Debug)]
struct RealInterval {
    lo: f64,
    hi: f64,
}

#[inline]
fn down2(value: f64) -> f64 {
    next_down_f64(next_down_f64(value))
}

#[inline]
fn up2(value: f64) -> f64 {
    next_up_f64(next_up_f64(value))
}

impl RealInterval {
    #[inline]
    fn point(value: f32) -> Self {
        let value = f32_to_f64_exact(value);
        Self {
            lo: value,
            hi: value,
        }
    }

    #[inline]
    fn checked(lo: f64, hi: f64) -> Option<Self> {
        (lo.is_finite() && hi.is_finite() && lo <= hi).then_some(Self { lo, hi })
    }

    #[inline]
    fn add(self, other: Self) -> Option<Self> {
        Self::checked(down2(self.lo + other.lo), up2(self.hi + other.hi))
    }

    #[inline]
    fn sub(self, other: Self) -> Option<Self> {
        Self::checked(down2(self.lo - other.hi), up2(self.hi - other.lo))
    }

    fn mul(self, other: Self) -> Option<Self> {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::checked(down2(lo), up2(hi))
    }

    fn div(self, other: Self) -> Option<Self> {
        if other.lo <= 0.0 || !other.lo.is_finite() || !other.hi.is_finite() {
            return None;
        }
        let quotients = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ];
        let lo = quotients.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = quotients.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::checked(down2(lo), up2(hi))
    }

    fn sqrt(self) -> Option<Self> {
        if self.lo <= 0.0 || !self.hi.is_finite() {
            return None;
        }
        Self::checked(down2(self.lo.sqrt()), up2(self.hi.sqrt()))
    }
}

/// Publish a finite binary32 radius enclosing a stored coefficient around an
/// exact-real binary64 interval. Every subtraction and the final conversion is
/// rounded outward; subnormal binary32 radii widen to `MIN_POSITIVE`.
fn interval_radius_f32(center: f32, interval: RealInterval) -> Option<f32> {
    let center = f32_to_f64_exact(center);
    let lower_gap = up2((center - interval.lo).abs());
    let upper_gap = up2((interval.hi - center).abs());
    let radius = f64_to_f32_up(up2(lower_gap.max(upper_gap)));
    (radius.is_finite() && radius >= 0.0).then_some(radius)
}

/// Authenticated interpretation of BatchNorm's channel dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchNormChannelAxisHint {
    /// The propagated tensor always uses this channel axis.
    Fixed(usize),
    /// Standard ONNX `[N,C,...]` provenance.  NY has two propagation surfaces:
    /// sequential networks retain the authored rank/channel axis 1, while DAG
    /// networks strip the leading singleton batch and use axis 0.  Recording
    /// the authored rank distinguishes those layouts even when extents collide.
    OnnxNchw { authored_rank: usize },
}

/// BatchNorm layer: y = (x - mean) / sqrt(var + eps) * ny + beta
///
/// During inference, mean and variance are fixed (running statistics).
/// This can be simplified to: y = x * scale + bias where:
///   scale = ny / sqrt(var + eps)
///   bias = beta - mean * ny / sqrt(var + eps)
#[derive(Debug, Clone)]
pub struct BatchNormLayer {
    /// Pre-computed scale: ny / sqrt(var + eps)
    pub scale: ArrayD<f32>,
    /// Pre-computed bias: beta - mean * scale
    pub bias: ArrayD<f32>,
    /// Per-channel certified bound on `|scale[c] - scale_real[c]|`, i.e. the f32
    /// rounding error baked into the precomputed `scale` versus the exact real
    /// coefficient `ny / sqrt(var + eps)`. Always non-negative and finite:
    /// non-finite or non-positive-denominator channels are refused at
    /// construction. Consumed by IBP and CROWN to fold the precompute error
    /// *outward* so bounds stay sound against the real affine, not merely the
    /// f32-rounded one (#batchnorm-ibp-directed-rounding).
    pub scale_err: ArrayD<f32>,
    /// Per-channel certified bound on `|bias[c] - bias_real[c]|` (mirror of
    /// [`scale_err`](Self::scale_err) for the precomputed `bias`; captures
    /// catastrophic cancellation in `beta - mean*scale` as an absolute error).
    pub bias_err: ArrayD<f32>,
    /// Number of channels (for proper broadcasting)
    pub num_channels: usize,
    /// Authenticated channel axis for the tensor shape seen by propagation.
    ///
    /// `None` preserves the legacy shape-based inference used by manually
    /// constructed layers and tests.  ONNX conversion records the authored
    /// rank so both retained `[N,C,...]` and batch-stripped `[C,...]` execution
    /// surfaces remain exact.  This provenance is necessary for ambiguous
    /// shapes such as `[C,L]` with `C == L`, where dimensions alone cannot
    /// distinguish channels from a manually batched `[N,C]` tensor.
    pub channel_axis_hint: Option<BatchNormChannelAxisHint>,
}

impl BatchNormLayer {
    /// Validate the public affine representation before it participates in a
    /// proof. Constructors establish this invariant, but the fields remain
    /// public for compatibility with existing integrations, so propagation
    /// surfaces must also fail closed on malformed struct literals.
    pub(crate) fn validate_affine_parameters(&self) -> Result<()> {
        let expected_shape = [self.num_channels];
        if self.num_channels == 0
            || self.scale.shape() != expected_shape
            || self.bias.shape() != expected_shape
            || self.scale_err.shape() != expected_shape
            || self.bias_err.shape() != expected_shape
        {
            return Err(NyError::InvalidSpec(format!(
                "BatchNorm: affine vectors must all have shape [{0}] with {0} > 0",
                self.num_channels
            )));
        }
        if self.scale.iter().any(|value| !value.is_finite())
            || self.bias.iter().any(|value| !value.is_finite())
        {
            return Err(NyError::InvalidSpec(
                "BatchNorm: affine scale and bias must be finite".to_string(),
            ));
        }
        if self
            .scale_err
            .iter()
            .chain(self.bias_err.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(NyError::InvalidSpec(
                "BatchNorm: affine error radii must be finite and non-negative".to_string(),
            ));
        }
        match self.channel_axis_hint {
            Some(BatchNormChannelAxisHint::Fixed(axis)) if axis > 1 => {
                return Err(NyError::InvalidSpec(format!(
                    "BatchNorm: unsupported channel-axis hint {axis}; expected 0 or 1"
                )));
            }
            Some(BatchNormChannelAxisHint::OnnxNchw { authored_rank }) if authored_rank < 2 => {
                return Err(NyError::InvalidSpec(format!(
                    "BatchNorm: ONNX NCHW provenance requires authored rank at least 2, got {authored_rank}"
                )));
            }
            _ => {}
        }
        Ok(())
    }

    /// Create a new BatchNorm layer from ONNX parameters.
    ///
    /// ONNX BatchNormalization inputs:
    /// - scale (ny): per-channel scale
    /// - B (beta): per-channel bias
    /// - mean: running mean per channel
    /// - var: running variance per channel
    /// - epsilon: small constant (default 1e-5)
    pub fn new(
        ny: &ArrayD<f32>,
        beta: &ArrayD<f32>,
        mean: &ArrayD<f32>,
        var: &ArrayD<f32>,
        epsilon: f32,
    ) -> Result<Self> {
        if ny.ndim() != 1 || ny.is_empty() {
            return Err(NyError::InvalidSpec(
                "BatchNorm: scale, bias, mean, and variance must be non-empty 1-D channel vectors"
                    .to_string(),
            ));
        }
        for (name, parameter) in [("bias", beta), ("mean", mean), ("variance", var)] {
            if parameter.shape() != ny.shape() {
                return Err(NyError::InvalidSpec(format!(
                    "BatchNorm: {name} shape {:?} does not match scale shape {:?}",
                    parameter.shape(),
                    ny.shape()
                )));
            }
        }
        if !epsilon.is_finite() {
            return Err(NyError::InvalidSpec(
                "BatchNorm: epsilon must be finite".to_string(),
            ));
        }
        for (name, parameter) in [
            ("scale", ny),
            ("bias", beta),
            ("mean", mean),
            ("variance", var),
        ] {
            if parameter.iter().any(|value| !value.is_finite()) {
                return Err(NyError::InvalidSpec(format!(
                    "BatchNorm: {name} contains a non-finite value"
                )));
            }
        }

        // Adjacent-binary64 interval arithmetic is authoritative only under the
        // probed IEEE environment. Refuse construction rather than silently
        // treating a rounded f64 recomputation as the exact-real coefficient.
        require_f64_interval_proof_environment()?;

        let epsilon_interval = RealInterval::point(epsilon);
        let mut scale_values = Vec::with_capacity(ny.len());
        let mut bias_values = Vec::with_capacity(ny.len());
        let mut scale_error_values = Vec::with_capacity(ny.len());
        let mut bias_error_values = Vec::with_capacity(ny.len());

        for (((&gamma, &beta_value), &mean_value), &variance) in
            ny.iter().zip(beta.iter()).zip(mean.iter()).zip(var.iter())
        {
            // The executable f32 affine must itself be defined and finite. In
            // particular, zero is not an admissible denominator: ONNX's real
            // expression is undefined at 0/sqrt(0), and any nonzero
            // neighbourhood is unbounded.
            let variance_sum_f32 = variance + epsilon;
            if !variance_sum_f32.is_finite() || variance_sum_f32 <= 0.0 {
                return Err(NyError::InvalidSpec(
                    "BatchNorm: variance + epsilon must be finite and strictly positive"
                        .to_string(),
                ));
            }
            let denominator_f32 = variance_sum_f32.sqrt();
            let scale_f32 = gamma / denominator_f32;
            let bias_f32 = beta_value - mean_value * scale_f32;
            if !scale_f32.is_finite() || !bias_f32.is_finite() {
                return Err(NyError::InvalidSpec(
                    "BatchNorm: derived scale and bias must be finite".to_string(),
                ));
            }

            // Enclose the exact-real coefficients. Widen every primitive, not
            // merely the final result: this covers rounding in var+eps, sqrt,
            // division, multiplication, and subtraction independently.
            let denominator = RealInterval::point(variance)
                .add(epsilon_interval)
                .and_then(RealInterval::sqrt)
                .ok_or_else(|| {
                    NyError::SoundnessRefusal(
                        "BatchNorm: exact-real denominator could not be enclosed as strictly positive"
                            .to_string(),
                    )
                })?;
            let scale_interval = RealInterval::point(gamma).div(denominator).ok_or_else(|| {
                NyError::SoundnessRefusal(
                    "BatchNorm: exact-real scale interval overflowed".to_string(),
                )
            })?;
            let bias_interval = RealInterval::point(beta_value)
                .sub(
                    RealInterval::point(mean_value)
                        .mul(scale_interval)
                        .ok_or_else(|| {
                            NyError::SoundnessRefusal(
                                "BatchNorm: exact-real mean-scale interval overflowed".to_string(),
                            )
                        })?,
                )
                .ok_or_else(|| {
                    NyError::SoundnessRefusal(
                        "BatchNorm: exact-real bias interval overflowed".to_string(),
                    )
                })?;

            let scale_error = interval_radius_f32(scale_f32, scale_interval).ok_or_else(|| {
                NyError::SoundnessRefusal(
                    "BatchNorm: scale precompute-error radius is not finitely representable"
                        .to_string(),
                )
            })?;
            let bias_error = interval_radius_f32(bias_f32, bias_interval).ok_or_else(|| {
                NyError::SoundnessRefusal(
                    "BatchNorm: bias precompute-error radius is not finitely representable"
                        .to_string(),
                )
            })?;

            scale_values.push(scale_f32);
            bias_values.push(bias_f32);
            scale_error_values.push(scale_error);
            bias_error_values.push(bias_error);
        }

        let build_array = |values: Vec<f32>, label: &str| {
            ArrayD::from_shape_vec(ny.raw_dim(), values).map_err(|error| {
                NyError::InternalError(format!(
                    "BatchNorm: failed to materialize {label} vector: {error}"
                ))
            })
        };
        let scale = build_array(scale_values, "scale")?;
        let bias = build_array(bias_values, "bias")?;
        let scale_err = build_array(scale_error_values, "scale error")?;
        let bias_err = build_array(bias_error_values, "bias error")?;
        let num_channels = scale.len();

        Ok(Self {
            scale,
            bias,
            scale_err,
            bias_err,
            num_channels,
            channel_axis_hint: None,
        })
    }

    /// Create from pre-computed scale and bias.
    ///
    /// Validates that scale and bias have matching shapes and contain only
    /// finite values (NaN/Inf would silently poison CROWN backward propagation).
    pub fn from_scale_bias(scale: ArrayD<f32>, bias: ArrayD<f32>) -> Result<Self> {
        if scale.shape() != bias.shape() {
            return Err(NyError::ShapeMismatch {
                expected: scale.shape().to_vec(),
                got: bias.shape().to_vec(),
            });
        }
        if scale.ndim() != 1 || scale.is_empty() {
            return Err(NyError::InvalidSpec(
                "BatchNorm: pre-computed scale and bias must be non-empty 1-D channel vectors"
                    .to_string(),
            ));
        }
        if scale.iter().any(|v| !v.is_finite()) {
            return Err(NyError::InvalidSpec(
                "BatchNorm: non-finite value in pre-computed scale — would poison CROWN backward"
                    .to_string(),
            ));
        }
        if bias.iter().any(|v| !v.is_finite()) {
            return Err(NyError::InvalidSpec(
                "BatchNorm: non-finite value in pre-computed bias — would poison CROWN backward"
                    .to_string(),
            ));
        }
        let num_channels = scale.len();
        // The caller supplies the exact f32 coefficients it intends to use, so
        // there is no precompute error to account for here (error = 0). A caller
        // that derives scale/bias from raw statistics should go through `new`,
        // which populates real error bounds.
        let scale_err = ArrayD::<f32>::zeros(scale.raw_dim());
        let bias_err = ArrayD::<f32>::zeros(bias.raw_dim());
        Ok(Self {
            scale,
            bias,
            scale_err,
            bias_err,
            num_channels,
            channel_axis_hint: None,
        })
    }

    /// Attach authenticated channel-axis provenance.
    ///
    /// BatchNorm propagation supports only the leading channel axis of an
    /// unbatched tensor (`0`) or axis `1` of a tensor that still carries a
    /// leading batch dimension.
    pub fn with_channel_axis(mut self, channel_axis: usize) -> Result<Self> {
        if channel_axis > 1 {
            return Err(NyError::InvalidSpec(format!(
                "BatchNorm: unsupported channel-axis hint {channel_axis}; expected 0 or 1"
            )));
        }
        self.channel_axis_hint = Some(BatchNormChannelAxisHint::Fixed(channel_axis));
        Ok(self)
    }

    /// Attach standard ONNX `[N,C,...]` provenance at the authored input rank.
    pub fn with_onnx_nchw_rank(mut self, authored_rank: usize) -> Result<Self> {
        if authored_rank < 2 {
            return Err(NyError::InvalidSpec(format!(
                "BatchNorm: ONNX NCHW provenance requires authored rank at least 2, got {authored_rank}"
            )));
        }
        self.channel_axis_hint = Some(BatchNormChannelAxisHint::OnnxNchw { authored_rank });
        Ok(self)
    }
}
