// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for beta-CROWN backward pass through Transpose layers.
//!
//! Prior to the fix in #1969, `propagate_layer_backward_with_alpha_beta` and
//! `propagate_layer_backward_with_beta` treated Transpose as a passthrough
//! (identical to Flatten/Reshape), returning `output_bounds.clone()`. This is
//! unsound because Transpose permutes elements — the CROWN coefficient columns
//! must be permuted by the inverse of the transpose permutation.
//!
//! These tests verify that the beta-CROWN backward pass correctly calls
//! `TransposeLayer::propagate_linear` instead of the identity passthrough.

use super::prelude::*;
use crate::layers::transform::TransposeLayer;

/// Verify that beta-CROWN backward through Transpose permutes CROWN coefficients.
///
/// Uses a [2,3] -> [3,2] transpose and checks that identity CROWN bounds
/// get their columns permuted (not passed through unchanged).
///
/// Reference: TransposeLayer::propagate_linear in layers/transform/mod.rs
#[ntest::timeout(10000)]
#[test]
fn test_transpose_backward_permutes_coefficients_alpha_beta() {
    // Construct pre_bounds with 2D shape [2,3] — this is what the Transpose
    // layer sees as its input shape.
    let pre_lower = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let pre_upper = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        .unwrap()
        .into_dyn();
    let pre_bounds = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity CROWN bounds for 6 outputs (flattened [3,2] = 6 elements)
    let output_bounds = LinearBounds::identity(6);

    // Construct the Transpose layer as a Layer enum variant
    let transpose = TransposeLayer::new(vec![1, 0]);
    let transpose_layer = Layer::Transpose(transpose);

    // Minimal network (just needs to exist for DomainAlphaState)
    let network = Network::new();
    let layer_bounds: Vec<Arc<BoundedTensor>> = vec![];
    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Call propagate_layer_backward_with_alpha_beta for the Transpose layer
    let result = verifier.propagate_layer_backward_with_alpha_beta(
        &transpose_layer,
        &output_bounds,
        &pre_bounds,
        None, // no constraints
        &beta_state,
        &alpha_state,
        0,    // layer index (doesn't matter for Transpose)
        None, // no GEMM engine
    );

    let new_bounds = result.expect("Transpose backward should succeed");

    // Verify the EXACT permutation mapping for [2,3] -> [3,2] with perm [1,0].
    //
    // Input shape [2,3], output shape [3,2]. For CROWN backward with identity A:
    //   mapping[out_flat] = in_flat: [0, 3, 1, 4, 2, 5]
    //   inv_mapping[in_flat] = out_flat: [0, 2, 4, 1, 3, 5]
    //   new_A[row, in_col] = identity[row, inv_mapping[in_col]]
    //                       = 1.0 iff row == inv_mapping[in_col]
    //
    // Expected 1.0 positions (row, col): (0,0), (2,1), (4,2), (1,3), (3,4), (5,5)
    let inv_mapping = [0usize, 2, 4, 1, 3, 5];
    for (in_col, &expected_row) in inv_mapping.iter().enumerate() {
        for row in 0..6 {
            let expected_val: f32 = if row == expected_row { 1.0 } else { 0.0 };
            assert!(
                (new_bounds.lower_a[[row, in_col]] - expected_val).abs() < 1e-10,
                "lower_a[{},{}] should be {}, got {}",
                row,
                in_col,
                expected_val,
                new_bounds.lower_a[[row, in_col]]
            );
            assert!(
                (new_bounds.upper_a[[row, in_col]] - expected_val).abs() < 1e-10,
                "upper_a[{},{}] should be {}, got {}",
                row,
                in_col,
                expected_val,
                new_bounds.upper_a[[row, in_col]]
            );
        }
    }

    // Bias should remain zero for both lower and upper
    for i in 0..6 {
        assert!(
            new_bounds.lower_b[i].abs() < 1e-10,
            "Lower bias[{}] should be 0, got {}",
            i,
            new_bounds.lower_b[i]
        );
        assert!(
            new_bounds.upper_b[i].abs() < 1e-10,
            "Upper bias[{}] should be 0, got {}",
            i,
            new_bounds.upper_b[i]
        );
    }
}

/// Verify that the beta-only backward path also permutes correctly.
#[ntest::timeout(10000)]
#[test]
fn test_transpose_backward_permutes_coefficients_beta_only() {
    let pre_lower = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let pre_upper = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        .unwrap()
        .into_dyn();
    let pre_bounds = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let output_bounds = LinearBounds::identity(6);
    let transpose = TransposeLayer::new(vec![1, 0]);
    let transpose_layer = Layer::Transpose(transpose);

    let beta_state = BetaState::empty();
    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let result = verifier.propagate_layer_backward_with_beta(
        &transpose_layer,
        &output_bounds,
        &pre_bounds,
        None,
        &beta_state,
        0,
        None,
    );

    let new_bounds = result.expect("Transpose backward (beta-only) should succeed");

    // Verify exact permutation mapping (same as alpha-beta test).
    // inv_mapping for [2,3] -> [3,2] with perm [1,0]: [0, 2, 4, 1, 3, 5]
    let inv_mapping = [0usize, 2, 4, 1, 3, 5];
    for (in_col, &expected_row) in inv_mapping.iter().enumerate() {
        for row in 0..6 {
            let expected_val: f32 = if row == expected_row { 1.0 } else { 0.0 };
            assert!(
                (new_bounds.lower_a[[row, in_col]] - expected_val).abs() < 1e-10,
                "lower_a[{},{}] should be {}, got {}",
                row,
                in_col,
                expected_val,
                new_bounds.lower_a[[row, in_col]]
            );
            assert!(
                (new_bounds.upper_a[[row, in_col]] - expected_val).abs() < 1e-10,
                "upper_a[{},{}] should be {}, got {}",
                row,
                in_col,
                expected_val,
                new_bounds.upper_a[[row, in_col]]
            );
        }
    }

    // Bias should remain zero for both lower and upper
    for i in 0..6 {
        assert!(
            new_bounds.lower_b[i].abs() < 1e-10,
            "Lower bias[{}] should be 0, got {}",
            i,
            new_bounds.lower_b[i]
        );
        assert!(
            new_bounds.upper_b[i].abs() < 1e-10,
            "Upper bias[{}] should be 0, got {}",
            i,
            new_bounds.upper_b[i]
        );
    }
}
