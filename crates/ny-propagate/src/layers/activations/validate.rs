// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared validation helpers for activation layer constructor parameters.

use ny_core::{NyError, Result};

#[inline]
pub(crate) fn validate_positive_finite(value: f32, layer: &str, param: &str) -> Result<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "{layer} {param} invalid: {value} (must be finite and > 0)"
        )));
    }
    Ok(value)
}

#[inline]
pub(crate) fn validate_nonnegative_finite(value: f32, layer: &str, param: &str) -> Result<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "{layer} {param} invalid: {value} (must be finite and >= 0)"
        )));
    }
    Ok(value)
}

#[inline]
pub(crate) fn validate_finite(value: f32, layer: &str, param: &str) -> Result<f32> {
    if !value.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "{layer} {param} invalid: {value} (must be finite)"
        )));
    }
    Ok(value)
}

#[inline]
pub(crate) fn validate_clip_bounds(min: f32, max: f32) -> Result<(f32, f32)> {
    if min.is_nan() || max.is_nan() {
        return Err(NyError::InvalidSpec(format!(
            "Clip bounds invalid: min={min}, max={max} (NaN not allowed)"
        )));
    }
    if min > max {
        return Err(NyError::InvalidSpec(format!(
            "Clip bounds invalid: min={min}, max={max} (min must be <= max)"
        )));
    }
    Ok((min, max))
}
