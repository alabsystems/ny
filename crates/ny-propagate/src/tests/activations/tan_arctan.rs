// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Tan/Arctan tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_tan_ibp_monotonic_interval() {
    // tan is monotonic on (-pi/2, pi/2)
    let lower = ArrayD::from_elem(IxDyn(&[2]), -0.5f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 0.5f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let tan_layer = TanLayer::new();
    let output = tan_layer.propagate_ibp(&input).unwrap();

    let expected_lower = (-0.5f32).tan();
    let expected_upper = (0.5f32).tan();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - expected_lower).abs() < 1e-5,
            "tan(-0.5) should be {}, got {}",
            expected_lower,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - expected_upper).abs() < 1e-5,
            "tan(0.5) should be {}, got {}",
            expected_upper,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_tan_ibp_asymptote_interval() {
    use std::f32::consts::PI;
    // Interval crosses pi/2 asymptote
    let lower = ArrayD::from_elem(IxDyn(&[1]), PI / 2.0 - 0.2);
    let upper = ArrayD::from_elem(IxDyn(&[1]), PI / 2.0 + 0.2);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let tan_layer = TanLayer::new();
    let output = tan_layer.propagate_ibp(&input).unwrap();

    assert!(output.lower()[[0]].is_infinite() && output.lower()[[0]].is_sign_negative());
    assert!(output.upper()[[0]].is_infinite() && output.upper()[[0]].is_sign_positive());
}

#[ntest::timeout(10000)]
#[test]
fn test_arctan_ibp_monotonic_interval() {
    let lower = ArrayD::from_elem(IxDyn(&[2]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let arctan_layer = ArctanLayer::new();
    let output = arctan_layer.propagate_ibp(&input).unwrap();

    let expected_lower = (-1.0f32).atan();
    let expected_upper = (1.0f32).atan();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - expected_lower).abs() < 1e-5,
            "atan(-1) should be {}, got {}",
            expected_lower,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - expected_upper).abs() < 1e-5,
            "atan(1) should be {}, got {}",
            expected_upper,
            output.upper()[[i]]
        );
    }
}

/// Regression test for #3316: arctan IBP bounds must stay in [-π/2, π/2] even for
/// extreme inputs where directed rounding could push past the range boundary.
/// arctan(1e10) → π/2 via f64 saturation, next_up_f32(π/2) > π/2.
#[ntest::timeout(10000)]
#[test]
fn test_arctan_ibp_extreme_range_clamp_3316() {
    use std::f32::consts::FRAC_PI_2;
    let lower = ArrayD::from_elem(IxDyn(&[2]), -1e10f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1e10f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let arctan_layer = ArctanLayer::new();
    let output = arctan_layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[i]] >= -FRAC_PI_2,
            "arctan lower bound must be >= -π/2, got {}",
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]] <= FRAC_PI_2,
            "arctan upper bound must be <= π/2, got {}",
            output.upper()[[i]]
        );
    }
}
