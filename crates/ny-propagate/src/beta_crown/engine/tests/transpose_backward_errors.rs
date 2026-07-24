// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Error-path regression tests for transpose backward propagation.

use super::prelude::*;
use crate::layers::transform::TransposeLayer;

/// Regression: alpha+beta transpose backward must propagate TransposeLayer errors.
#[ntest::timeout(10000)]
#[test]
fn test_transpose_backward_alpha_beta_propagates_transpose_error() {
    let pre_lower = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let pre_upper = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        .unwrap()
        .into_dyn();
    let pre_bounds = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let output_bounds = LinearBounds::identity(6);
    // Invalid axes length for 2D input: resolve_perm should return ShapeMismatch.
    let transpose_layer = Layer::Transpose(TransposeLayer::new(vec![0]));

    let network = Network::new();
    let layer_bounds: Vec<Arc<BoundedTensor>> = vec![];
    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let result = verifier.propagate_layer_backward_with_alpha_beta(
        &transpose_layer,
        &output_bounds,
        &pre_bounds,
        None,
        &beta_state,
        &alpha_state,
        None,
        0,
        None,
    );

    match result {
        Err(ny_core::NyError::ShapeMismatch { expected, got }) => {
            assert_eq!(expected, vec![2], "Expected rank-2 permutation length");
            assert_eq!(got, vec![1], "Got invalid rank-1 permutation");
        }
        other => panic!("Expected ShapeMismatch from invalid transpose axes, got {other:?}"),
    }
}

/// Regression: beta-only transpose backward must propagate TransposeLayer errors.
#[ntest::timeout(10000)]
#[test]
fn test_transpose_backward_beta_only_propagates_transpose_error() {
    let pre_lower = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .into_dyn();
    let pre_upper = Array2::from_shape_vec((2, 3), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        .unwrap()
        .into_dyn();
    let pre_bounds = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let output_bounds = LinearBounds::identity(6);
    let transpose_layer = Layer::Transpose(TransposeLayer::new(vec![0]));

    let beta_state = BetaState::empty();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let result = verifier.propagate_layer_backward_with_beta(
        &transpose_layer,
        &output_bounds,
        &pre_bounds,
        None,
        &beta_state,
        0,
        None,
    );

    match result {
        Err(ny_core::NyError::ShapeMismatch { expected, got }) => {
            assert_eq!(expected, vec![2], "Expected rank-2 permutation length");
            assert_eq!(got, vec![1], "Got invalid rank-1 permutation");
        }
        other => panic!("Expected ShapeMismatch from invalid transpose axes, got {other:?}"),
    }
}
