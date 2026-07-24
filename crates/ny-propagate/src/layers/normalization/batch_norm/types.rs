// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Type definitions and constructors for BatchNorm.

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::next_up_f32;

use crate::bounds::safe_mul_for_bounds;

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
    /// coefficient `ny / sqrt(var + eps)`. Non-negative and finite for
    /// finite-scale channels; `+∞` for degenerate (var+eps→0) channels whose
    /// scale is already ±∞. Consumed by IBP (and, where wired, CROWN) to fold the
    /// precompute error *outward* so bounds stay sound against the real affine,
    /// not merely the f32-rounded one (#batchnorm-ibp-directed-rounding).
    pub scale_err: ArrayD<f32>,
    /// Per-channel certified bound on `|bias[c] - bias_real[c]|` (mirror of
    /// [`scale_err`](Self::scale_err) for the precomputed `bias`; captures
    /// catastrophic cancellation in `beta - mean*scale` as an absolute error).
    pub bias_err: ArrayD<f32>,
    /// Number of channels (for proper broadcasting)
    pub num_channels: usize,
}

impl BatchNormLayer {
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
        // Guard: negative variance causes NaN via sqrt(negative), which silently
        // poisons the entire CROWN backward chain. (#2814)
        if var.iter().any(|&v| v + epsilon < 0.0) {
            return Err(NyError::InvalidSpec(
                "BatchNorm: variance + epsilon is negative — would produce NaN scale".to_string(),
            ));
        }

        // Compute scale = ny / sqrt(var + eps).
        //
        // NOTE: the guard above only rejects var+eps < 0. A channel with
        // var + eps == 0 (e.g. var = -eps, or var underflowing to 0 in f32)
        // yields sqrt(0) = 0 and scale = ny / 0 = ±inf. We deliberately keep
        // that ±inf scale: an Inf scale is a sound (if maximally imprecise)
        // affine coefficient that the IBP/CROWN paths widen to ±inf downstream.
        let std = var.mapv(|v| (v + epsilon).sqrt());
        let scale = ny / &std;

        // Compute bias = beta - mean * scale.
        //
        // Use safe_mul_for_bounds (0 * inf = 0) for the mean*scale product.
        // With a degenerate Inf scale and a finite mean, a plain multiply gives
        // mean * inf, and crucially 0 * inf = NaN when mean == 0. A NaN bias
        // poisons every downstream firewall (it cannot be recovered to a precise
        // bound and forces whole-matrix conservative fallback / strict-construction
        // aborts). safe_mul keeps the product NaN-free: mean == 0 composes to 0
        // (bias = beta), and a nonzero mean times Inf yields a ±inf bias, which is
        // sound and handled by new_or_conservative / concretize. Finite scales are
        // unaffected (ordinary multiplication).
        let mean_scale = {
            let mut ms = mean.clone();
            ndarray::Zip::from(&mut ms)
                .and(&scale)
                .for_each(|m, &s| *m = safe_mul_for_bounds(*m, s));
            ms
        };
        let bias = beta - &mean_scale;

        // Certified error bounds on the f32-precomputed `scale`/`bias` versus the
        // exact real coefficients. Recompute the affine in f64 (the f32 raw params
        // are exact in f64; f64 sqrt/div add only ~2^-53) and bound the gap. The
        // IBP path folds these outward so the bound stays sound against the *real*
        // affine, covering the ~ulp(scale)·|x| precompute error that a single
        // final next_down/next_up does not (#batchnorm-ibp-directed-rounding).
        let eps64 = epsilon as f64;
        let std_f64 = var.mapv(|v| ((v as f64) + eps64).sqrt());
        let mut scale_f64 = ny.mapv(|g| g as f64);
        ndarray::Zip::from(&mut scale_f64)
            .and(&std_f64)
            .for_each(|s, &sd| *s /= sd);
        let mut bias_f64 = beta.mapv(|b| b as f64);
        ndarray::Zip::from(&mut bias_f64)
            .and(mean)
            .and(&scale_f64)
            .for_each(|b, &m, &s| *b -= (m as f64) * s);

        // For a degenerate channel (var+eps→0 → scale = ±∞) the unboundedness is
        // carried *in the coefficient* (incoming·∞ = ±∞, sound at concretize) — and
        // the bias is either finite (mean=0) or already ±∞. There is no FINITE
        // precompute error to fold into the bias, so err = 0 (folding ∞ here would
        // wrongly push an otherwise-finite bias to ±∞; the coefficient handles it).
        let mut scale_err = ArrayD::<f32>::zeros(scale.raw_dim());
        ndarray::Zip::from(&mut scale_err)
            .and(&scale)
            .and(&scale_f64)
            .for_each(|e, &s32, &s64| {
                *e = if s32.is_finite() {
                    next_up_f32((((s32 as f64) - s64).abs()) as f32)
                } else {
                    0.0
                };
            });
        let mut bias_err = ArrayD::<f32>::zeros(bias.raw_dim());
        ndarray::Zip::from(&mut bias_err)
            .and(&scale)
            .and(&bias)
            .and(&bias_f64)
            .for_each(|e, &s32, &b32, &b64| {
                *e = if s32.is_finite() && b32.is_finite() && b64.is_finite() {
                    next_up_f32((((b32 as f64) - b64).abs()) as f32)
                } else {
                    0.0
                };
            });

        let num_channels = scale.len();

        Ok(Self {
            scale,
            bias,
            scale_err,
            bias_err,
            num_channels,
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
        })
    }
}
