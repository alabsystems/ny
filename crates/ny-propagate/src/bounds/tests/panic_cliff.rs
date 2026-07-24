// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #2907: CROWN backward panic cliff elimination.
//!
//! These tests verify that malformed LinearBounds produce conservative
//! fallbacks (±Inf) instead of panicking.

use crate::bounds::LinearBounds;
use ndarray::{Array1, Array2};
use ny_tensor::BoundedTensor;

/// #2907 Site 1: concretize with mismatched bias length returns conservative
/// fallback instead of panicking.
#[ntest::timeout(10000)]
#[test]
fn test_concretize_malformed_bias_length_returns_conservative_2907() {
    // Create LinearBounds with mismatched bias length (3 outputs in A, 2 in lower_b).
    let malformed = LinearBounds {
        lower_a: Array2::zeros((3, 2)),
        lower_b: Array1::zeros(2), // should be 3 — deliberately mismatched
        upper_a: Array2::zeros((3, 2)),
        upper_b: Array1::zeros(3),
        lower_a_err: None,
        upper_a_err: None,
    };

    let input = BoundedTensor::new(
        ndarray::arr1(&[0.0f32, 1.0]).into_dyn(),
        ndarray::arr1(&[1.0f32, 2.0]).into_dyn(),
    )
    .expect("valid input bounds");

    // Should NOT panic — should return conservative [-inf, +inf] fallback.
    let result = malformed.concretize(&input);
    assert_eq!(result.len(), 3, "output should have 3 elements");
    assert!(
        result.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "malformed bounds should produce -inf lower"
    );
    assert!(
        result.upper().iter().all(|&v| v == f32::INFINITY),
        "malformed bounds should produce +inf upper"
    );
}

/// #2907 Site 1: concretize_sound with mismatched A-matrix rows returns
/// conservative fallback instead of panicking.
#[ntest::timeout(10000)]
#[test]
fn test_concretize_sound_malformed_a_matrix_returns_conservative_2907() {
    // Mismatched A-matrix dimensions (lower_a rows != upper_a rows).
    let malformed = LinearBounds {
        lower_a: Array2::zeros((2, 3)),
        lower_b: Array1::zeros(2),
        upper_a: Array2::zeros((3, 3)), // row mismatch with lower_a!
        upper_b: Array1::zeros(3),
        lower_a_err: None,
        upper_a_err: None,
    };

    let input = BoundedTensor::new(
        ndarray::arr1(&[0.0f32, 1.0, 2.0]).into_dyn(),
        ndarray::arr1(&[1.0f32, 2.0, 3.0]).into_dyn(),
    )
    .expect("valid input bounds");

    // Should NOT panic — should return conservative [-inf, +inf] fallback.
    let result = malformed.concretize_sound(&input);
    assert!(
        result.lower().iter().all(|&v| v == f32::NEG_INFINITY),
        "malformed bounds should produce -inf lower"
    );
    assert!(
        result.upper().iter().all(|&v| v == f32::INFINITY),
        "malformed bounds should produce +inf upper"
    );
}
