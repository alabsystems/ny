// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Softplus tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_softplus_ibp_basic() {
    // Test softplus on interval
    let lower = ArrayD::from_elem(IxDyn(&[3]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let softplus_layer = SoftplusLayer::new();
    let output = softplus_layer.propagate_ibp(&input).unwrap();

    // softplus is monotonic
    let expected_lower = (1.0 + (-2.0_f32).exp()).ln();
    let expected_upper = (1.0 + 2.0_f32.exp()).ln();

    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - expected_lower).abs() < 1e-5,
            "softplus(-2) should be {}, got {}",
            expected_lower,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - expected_upper).abs() < 1e-5,
            "softplus(2) should be {}, got {}",
            expected_upper,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softplus_always_positive() {
    // Softplus output should always be positive
    let lower = ArrayD::from_elem(IxDyn(&[2]), -100.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -50.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let softplus_layer = SoftplusLayer::new();
    let output = softplus_layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[i]] >= 0.0,
            "softplus should always be non-negative"
        );
    }
}

/// Regression test for #3316: softplus IBP lower bound must be >= 0 even for
/// extreme negative inputs where directed rounding could push past zero.
/// softplus(-1000) → 0 via f64 underflow, next_down_f32(0.0) = -1e-45.
#[ntest::timeout(10000)]
#[test]
fn test_softplus_ibp_extreme_range_clamp_3316() {
    let lower = ArrayD::from_elem(IxDyn(&[2]), -1000.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -500.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let softplus_layer = SoftplusLayer::new();
    let output = softplus_layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[i]] >= 0.0,
            "softplus lower bound must be >= 0 for extreme negative inputs, got {}",
            output.lower()[[i]]
        );
    }
}
