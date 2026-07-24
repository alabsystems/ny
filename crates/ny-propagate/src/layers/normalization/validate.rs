// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared validation helpers for normalization layers.
//!
//! Extracted from per-layer `validated_eps()` and `MIN_EPS` constants
//! that were duplicated 4x (LayerNorm, RmsNorm, InstanceNorm, AdaIN).
//!
//! Reference: designs/2026-02-27-normalization-trait-dedup.md

use ny_core::{NyError, Result};

/// Minimum eps for normalization layers to prevent division-by-zero NaN.
///
/// When all inputs are identical (LayerNorm) or zero (RmsNorm), the
/// denominator is `sqrt(var + eps)` or `sqrt(mean_sq + eps)`. With eps=0,
/// the denominator is 0 and normalization produces 0/0 = NaN. This floor
/// prevents that.
///
/// Value 1e-12 is well below any practical eps (PyTorch default: 1e-5,
/// ONNX default: 1e-5) while still preventing division-by-zero.
pub(crate) const NORM_MIN_EPS: f32 = 1e-12;

/// Validate eps: must be finite and non-negative. Returns clamped value
/// (at least `NORM_MIN_EPS`) or error for invalid inputs (NaN, Inf, negative).
///
/// `layer_name` is used in the error message (e.g., "LayerNorm", "RMSNorm").
#[inline]
pub(crate) fn validate_norm_eps(eps: f32, layer_name: &str) -> Result<f32> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "{layer_name} eps invalid: {eps} (must be finite and non-negative)"
        )));
    }
    Ok(eps.max(NORM_MIN_EPS))
}
