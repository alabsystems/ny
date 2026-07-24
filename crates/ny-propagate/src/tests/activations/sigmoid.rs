// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Sigmoid tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_sigmoid_ibp_basic() {
    // Test sigmoid on interval that straddles zero
    let lower = ArrayD::from_elem(IxDyn(&[4]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sigmoid_layer = SigmoidLayer::new();
    let output = sigmoid_layer.propagate_ibp(&input).unwrap();

    // sigmoid is monotonic
    let expected_lower = 1.0 / (1.0 + 2.0_f32.exp()); // sigmoid(-2)
    let expected_upper = 1.0 / (1.0 + (-2.0_f32).exp()); // sigmoid(2)

    for i in 0..4 {
        assert!(
            (output.lower()[[i]] - expected_lower).abs() < 1e-5,
            "sigmoid(-2) should be {}, got {}",
            expected_lower,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - expected_upper).abs() < 1e-5,
            "sigmoid(2) should be {}, got {}",
            expected_upper,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sigmoid_range() {
    // Sigmoid output should always be in [0, 1] (inclusive at limits due to float precision)
    let lower = ArrayD::from_elem(IxDyn(&[2]), -10.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sigmoid_layer = SigmoidLayer::new();
    let output = sigmoid_layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[i]] >= 0.0 && output.lower()[[i]] <= 1.0,
            "sigmoid lower bound should be in [0, 1], got {}",
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]] >= 0.0 && output.upper()[[i]] <= 1.0,
            "sigmoid upper bound should be in [0, 1], got {}",
            output.upper()[[i]]
        );
        // Also check monotonicity: lower input gives lower sigmoid
        assert!(
            output.lower()[[i]] < output.upper()[[i]],
            "sigmoid bounds should be ordered"
        );
    }
}

/// Regression test for #3316: sigmoid IBP bounds must stay in [0, 1] even for
/// extreme inputs where directed rounding could push past the range boundary.
/// sigmoid(-1000) → 0 via f64 underflow, next_down_f32(0.0) = -1e-45.
/// sigmoid(1000) → 1 via f64 saturation, next_up_f32(1.0) > 1.
#[ntest::timeout(10000)]
#[test]
fn test_sigmoid_ibp_extreme_range_clamp_3316() {
    let lower = ArrayD::from_elem(IxDyn(&[2]), -1000.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1000.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sigmoid_layer = SigmoidLayer::new();
    let output = sigmoid_layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[i]] >= 0.0,
            "sigmoid lower bound must be >= 0, got {}",
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]] <= 1.0,
            "sigmoid upper bound must be <= 1, got {}",
            output.upper()[[i]]
        );
    }
}
