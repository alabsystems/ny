// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Tanh tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_tanh_ibp_basic() {
    // Test tanh on interval that straddles zero
    let lower = ArrayD::from_elem(IxDyn(&[4]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let tanh_layer = TanhLayer::new();
    let output = tanh_layer.propagate_ibp(&input).unwrap();

    // tanh is monotonic: tanh(-1) ≈ -0.7616, tanh(1) ≈ 0.7616
    let expected_lower = (-1.0_f32).tanh();
    let expected_upper = (1.0_f32).tanh();

    for i in 0..4 {
        assert!(
            (output.lower()[[i]] - expected_lower).abs() < 1e-5,
            "tanh(-1) should be {}, got {}",
            expected_lower,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - expected_upper).abs() < 1e-5,
            "tanh(1) should be {}, got {}",
            expected_upper,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_tanh_ibp_soundness() {
    // Test that IBP bounds are sound (contain actual function values)
    // Test multiple intervals
    let test_cases = vec![
        (-5.0_f32, 5.0_f32),
        (-1.0, 1.0),
        (0.0, 2.0),
        (-2.0, 0.0),
        (-10.0, 10.0),
    ];

    for (l, u) in test_cases {
        let lower = ArrayD::from_elem(IxDyn(&[1]), l);
        let upper = ArrayD::from_elem(IxDyn(&[1]), u);
        let input = BoundedTensor::new(lower, upper).unwrap();

        let tanh_layer = TanhLayer::new();
        let output = tanh_layer.propagate_ibp(&input).unwrap();

        // Test several points in the interval
        for i in 0..=10 {
            let x = l + (u - l) * (i as f32 / 10.0);
            let y = x.tanh();
            assert!(
                output.lower()[[0]] <= y && y <= output.upper()[[0]],
                "tanh({}) = {} should be in [{}, {}]",
                x,
                y,
                output.lower()[[0]],
                output.upper()[[0]]
            );
        }
    }
}

/// Regression test for #3316: tanh IBP bounds must stay in [-1, 1] even for
/// extreme inputs where directed rounding could push past the range boundary.
/// tanh(1000) → 1 via f64 saturation, next_up_f32(1.0) > 1.
/// tanh(-1000) → -1 via f64 saturation, next_down_f32(-1.0) < -1.
#[ntest::timeout(10000)]
#[test]
fn test_tanh_ibp_extreme_range_clamp_3316() {
    let lower = ArrayD::from_elem(IxDyn(&[2]), -1000.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1000.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let tanh_layer = TanhLayer::new();
    let output = tanh_layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[i]] >= -1.0,
            "tanh lower bound must be >= -1, got {}",
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]] <= 1.0,
            "tanh upper bound must be <= 1, got {}",
            output.upper()[[i]]
        );
    }
}
