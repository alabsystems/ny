// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared per-channel spatial stride and flat-index-to-channel helpers.
//!
//! Used by activation layers with per-channel parameters (Snake, PReLU)
//! to map a flat neuron index to the correct channel without duplicating
//! the stride validation or division math.
//!
//! Part of #4169.

use ny_core::{NyError, Result};

/// Compute per-channel spatial stride: number of spatial elements per channel.
///
/// For input shape `[C, T]` with `C == channel_count`, returns `T`.
/// For scalar parameters (`channel_count <= 1`), returns `total_elements`
/// so every element maps to the single parameter value.
///
/// Returns `ShapeMismatch` if `total_elements` is zero or not evenly
/// divisible by `channel_count`.
pub(crate) fn per_channel_spatial_stride(
    total_elements: usize,
    channel_count: usize,
    _layer_name: &str,
) -> Result<usize> {
    if channel_count <= 1 {
        return Ok(total_elements);
    }
    if total_elements == 0 || !total_elements.is_multiple_of(channel_count) {
        return Err(NyError::ShapeMismatch {
            expected: vec![channel_count],
            got: vec![total_elements],
        });
    }
    Ok(total_elements / channel_count)
}

/// Map a flat element index to a channel index given the per-channel stride.
///
/// For row-major layout `[C, T]`, elements within a channel are contiguous,
/// so `channel = flat_idx / stride` where `stride = T`.
#[inline]
pub(crate) fn channel_index_for_flat(flat_idx: usize, stride: usize) -> usize {
    flat_idx / stride
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_broadcast_returns_total_elements() {
        // channel_count == 1 → stride == total_elements (scalar broadcast)
        assert_eq!(per_channel_spatial_stride(12, 1, "Test").unwrap(), 12);
        assert_eq!(per_channel_spatial_stride(0, 1, "Test").unwrap(), 0);
    }

    #[test]
    fn test_valid_multi_channel_stride() {
        // [C=3, T=4] → stride = 4
        assert_eq!(per_channel_spatial_stride(12, 3, "Test").unwrap(), 4);
        // [C=2, T=3] → stride = 3
        assert_eq!(per_channel_spatial_stride(6, 2, "Test").unwrap(), 3);
    }

    #[test]
    fn test_indivisible_total_returns_shape_mismatch() {
        let err = per_channel_spatial_stride(7, 3, "Test").unwrap_err();
        assert!(
            matches!(err, NyError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}"
        );
    }

    #[test]
    fn test_zero_total_with_multi_channel_returns_shape_mismatch() {
        let err = per_channel_spatial_stride(0, 3, "Test").unwrap_err();
        assert!(
            matches!(err, NyError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}"
        );
    }

    #[test]
    fn test_channel_index_for_flat_two_channels() {
        // [C=2, T=3]: flat indices 0,1,2 → channel 0; 3,4,5 → channel 1
        let stride = 3;
        assert_eq!(channel_index_for_flat(0, stride), 0);
        assert_eq!(channel_index_for_flat(1, stride), 0);
        assert_eq!(channel_index_for_flat(2, stride), 0);
        assert_eq!(channel_index_for_flat(3, stride), 1);
        assert_eq!(channel_index_for_flat(4, stride), 1);
        assert_eq!(channel_index_for_flat(5, stride), 1);
    }
}
