// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Backward layout resolution for linear CROWN propagation.
//!
//! Centralizes the shape compatibility logic shared by CPU CROWN, GEMM CROWN,
//! and batched CROWN backward paths. Previously duplicated in three call sites.

use ny_core::{NyError, Result};

/// Resolved layout for a linear CROWN backward pass.
///
/// When CROWN bounds have more columns than the weight matrix has rows,
/// the extra columns represent repeated position blocks (e.g., from
/// ReduceMean expansion over a sequence dimension). Each block is
/// processed independently through the same weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LinearBackwardLayout {
    /// Number of output rows in the incoming CROWN bounds.
    pub num_outputs: usize,
    /// Feature dimension of each position block (= weight matrix rows).
    pub out_features: usize,
    /// Number of position blocks (1 for standard case, >1 for sequence dim).
    pub num_positions: usize,
    /// Input features per position (= weight matrix columns).
    pub in_features: usize,
    /// Total input features across all positions (= num_positions * in_features).
    pub total_in_features: usize,
}

/// Resolve the backward layout for a linear CROWN pass.
///
/// Checks that `bounds_inputs` (columns of the incoming A matrix) is compatible
/// with `weight_rows` (rows of the weight matrix, i.e. out_features of the layer).
///
/// # Compatibility rules
///
/// - **Exact match:** `bounds_inputs == weight_rows` → single position block.
/// - **Divisible:** `bounds_inputs % weight_rows == 0` → multiple position blocks
///   (sequence dimension from ReduceMean expansion).
/// - **Mismatch:** returns `NyError::ShapeMismatch`.
pub(crate) fn resolve_backward_layout(
    num_outputs: usize,
    bounds_inputs: usize,
    weight_rows: usize,
    in_features: usize,
) -> Result<LinearBackwardLayout> {
    let (out_features, num_positions) = if bounds_inputs == weight_rows {
        (bounds_inputs, 1usize)
    } else if weight_rows > 0 && bounds_inputs.is_multiple_of(weight_rows) {
        (weight_rows, bounds_inputs / weight_rows)
    } else {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_outputs, weight_rows],
            got: vec![num_outputs, bounds_inputs],
        });
    };

    Ok(LinearBackwardLayout {
        num_outputs,
        out_features,
        num_positions,
        in_features,
        total_in_features: num_positions * in_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_exact_match() {
        let layout = resolve_backward_layout(4, 8, 8, 3).unwrap();
        assert_eq!(layout.num_outputs, 4);
        assert_eq!(layout.out_features, 8);
        assert_eq!(layout.num_positions, 1);
        assert_eq!(layout.in_features, 3);
        assert_eq!(layout.total_in_features, 3);
    }

    #[test]
    fn test_resolve_divisible_sequence() {
        // bounds_inputs=12, weight_rows=4 → 3 positions of 4 features each
        let layout = resolve_backward_layout(2, 12, 4, 5).unwrap();
        assert_eq!(layout.num_outputs, 2);
        assert_eq!(layout.out_features, 4);
        assert_eq!(layout.num_positions, 3);
        assert_eq!(layout.in_features, 5);
        assert_eq!(layout.total_in_features, 15);
    }

    #[test]
    fn test_resolve_mismatch() {
        let err = resolve_backward_layout(2, 7, 4, 3).unwrap_err();
        assert!(
            matches!(err, NyError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got: {err:?}"
        );
    }
}
