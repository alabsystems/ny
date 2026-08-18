// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX axis resolution utilities.
//!
//! Provides validated axis resolution for negative-index handling per the ONNX spec.
//! Use these instead of inline `(ndim as i32 + axis) as usize` which silently wraps
//! on out-of-range negative values.

use crate::{NyError, Result};

/// Resolve a single ONNX axis index (i64) to a positive `usize` with bounds validation.
///
/// Handles negative indexing (e.g., -1 → last axis) per ONNX spec. Returns `Err` if the
/// resolved axis is out of range `[0, ndim)`.
pub fn resolve_axis(axis: i64, ndim: usize, layer_name: &str) -> Result<usize> {
    if axis < 0 {
        // Work in unsigned magnitudes so an unusually large `usize` rank does
        // not wrap through `i64`, and so `i64::MIN` is handled without negation
        // overflow.
        let magnitude = usize::try_from(axis.unsigned_abs()).map_err(|_| {
            NyError::InvalidSpec(format!(
                "{layer_name}: axis {axis} magnitude cannot be represented on this platform"
            ))
        })?;
        if magnitude > ndim {
            return Err(NyError::InvalidSpec(format!(
                "{layer_name}: axis {axis} out of range for {ndim}D tensor"
            )));
        }
        Ok(ndim - magnitude)
    } else {
        let a = usize::try_from(axis).map_err(|_| {
            NyError::InvalidSpec(format!(
                "{layer_name}: axis {axis} cannot be represented on this platform"
            ))
        })?;
        if a >= ndim {
            return Err(NyError::InvalidSpec(format!(
                "{layer_name}: axis {axis} out of range for {ndim}D tensor"
            )));
        }
        Ok(a)
    }
}

/// Resolve a single ONNX axis (i32) to a positive `usize` with bounds validation.
///
/// Convenience wrapper for layers that store axis as `i32`.
pub fn resolve_axis_i32(axis: i32, ndim: usize, layer_name: &str) -> Result<usize> {
    resolve_axis(i64::from(axis), ndim, layer_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_positive_axis() {
        assert_eq!(resolve_axis(0, 3, "Test").unwrap(), 0);
        assert_eq!(resolve_axis(2, 3, "Test").unwrap(), 2);
    }

    #[test]
    fn test_resolve_negative_axis() {
        assert_eq!(resolve_axis(-1, 3, "Test").unwrap(), 2);
        assert_eq!(resolve_axis(-3, 3, "Test").unwrap(), 0);
    }

    #[test]
    fn test_positive_out_of_range() {
        assert!(
            resolve_axis(3, 3, "Test").is_err(),
            "axis 3 out of range for 3D"
        );
        assert!(
            resolve_axis(100, 3, "Test").is_err(),
            "axis 100 out of range for 3D"
        );
    }

    #[test]
    fn test_negative_out_of_range() {
        assert!(
            resolve_axis(-4, 3, "Test").is_err(),
            "axis -4 out of range for 3D"
        );
        assert!(
            resolve_axis(-100, 3, "Test").is_err(),
            "axis -100 out of range for 3D"
        );
    }

    #[test]
    fn test_zero_ndim() {
        assert!(
            resolve_axis(0, 0, "Test").is_err(),
            "axis 0 invalid for 0D tensor"
        );
        assert!(
            resolve_axis(-1, 0, "Test").is_err(),
            "axis -1 invalid for 0D tensor"
        );
    }

    #[test]
    fn test_i32_wrapper() {
        assert_eq!(resolve_axis_i32(-1, 3, "Test").unwrap(), 2);
        assert_eq!(resolve_axis_i32(1, 3, "Test").unwrap(), 1);
        assert!(
            resolve_axis_i32(3, 3, "Test").is_err(),
            "i32 axis 3 out of range for 3D"
        );
    }

    #[test]
    fn test_i64_min_axis_is_rejected_without_overflow() {
        assert!(resolve_axis(i64::MIN, 3, "Test").is_err());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn test_negative_axis_with_rank_above_i64_max_does_not_wrap() {
        assert_eq!(
            resolve_axis(-1, usize::MAX, "Test").unwrap(),
            usize::MAX - 1
        );
    }
}
