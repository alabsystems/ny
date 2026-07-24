// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Sin/Cos tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_sin_ibp_no_extrema() {
    use std::f32::consts::PI;
    // Test sin on interval that doesn't contain extrema
    let lower = ArrayD::from_elem(IxDyn(&[2]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), PI / 4.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sin_layer = SinLayer::new();
    let output = sin_layer.propagate_ibp(&input).unwrap();

    // sin is monotonically increasing on [0, π/4]
    let expected_lower = 0.0_f32.sin(); // = 0
    let expected_upper = (PI / 4.0).sin(); // ≈ 0.707

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - expected_lower).abs() < 1e-5,
            "sin(0) should be {}, got {}",
            expected_lower,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - expected_upper).abs() < 1e-5,
            "sin(π/4) should be {}, got {}",
            expected_upper,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sin_ibp_contains_maximum() {
    use std::f32::consts::PI;
    // Test sin on interval that contains π/2 (maximum)
    let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), PI);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sin_layer = SinLayer::new();
    let output = sin_layer.propagate_ibp(&input).unwrap();

    // Interval contains π/2 where sin=1
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-5,
        "sin max should be 1 when interval contains π/2, got {}",
        output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sin_sound_mode_uses_ibp_constant_bounds() {
    let sin_layer = SinLayer::new().with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.5_f32, 0.0, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5_f32, 0.25, 1.5]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let result = sin_layer
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();

    assert!(
        result.lower_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero lower_a in sound-mode constant bounds"
    );
    assert!(
        result.upper_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero upper_a in sound-mode constant bounds"
    );

    let ibp_bounds = sin_layer.propagate_ibp(&input).unwrap();
    let ibp_flat = ibp_bounds.flatten();
    let ibp_lower = ibp_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let ibp_upper = ibp_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    for i in 0..3 {
        assert!(
            (result.lower_b[i] - ibp_lower[i]).abs() <= 1e-6,
            "Sound-mode lower_b mismatch at {}: {} vs {}",
            i,
            result.lower_b[i],
            ibp_lower[i]
        );
        assert!(
            (result.upper_b[i] - ibp_upper[i]).abs() <= 1e-6,
            "Sound-mode upper_b mismatch at {}: {} vs {}",
            i,
            result.upper_b[i],
            ibp_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sin_sound_mode_constant_bounds_contain_sampled_outputs() {
    let sin_layer = SinLayer::new().with_sound_mode(true);

    let interval_lower = -2.7f32;
    let interval_upper = 2.3f32;
    let lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![interval_lower]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![interval_upper]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let result = sin_layer
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();

    assert!(
        result.lower_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero lower_a in sound-mode constant bounds"
    );
    assert!(
        result.upper_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero upper_a in sound-mode constant bounds"
    );

    let samples = 200usize;
    assert!(
        samples + 1 >= 100,
        "sin soundness checks require at least 100 sampled points"
    );
    for i in 0..=samples {
        let x = interval_lower + (interval_upper - interval_lower) * (i as f32) / (samples as f32);
        let y = x.sin();
        assert!(
            result.lower_b[0] <= y + 1e-5,
            "sound-mode Sin lower bound {} > sin({}) = {}",
            result.lower_b[0],
            x,
            y
        );
        assert!(
            result.upper_b[0] >= y - 1e-5,
            "sound-mode Sin upper bound {} < sin({}) = {}",
            result.upper_b[0],
            x,
            y
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_ibp_no_extrema() {
    use std::f32::consts::PI;
    // Test cos on interval that doesn't contain extrema
    let lower = ArrayD::from_elem(IxDyn(&[2]), PI / 4.0);
    let upper = ArrayD::from_elem(IxDyn(&[2]), PI / 2.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let cos_layer = CosLayer::new();
    let output = cos_layer.propagate_ibp(&input).unwrap();

    // cos is monotonically decreasing on [π/4, π/2]
    let expected_upper = (PI / 4.0).cos(); // ≈ 0.707
    let expected_lower = (PI / 2.0).cos(); // ≈ 0

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - expected_lower).abs() < 1e-5,
            "cos(π/2) should be {}, got {}",
            expected_lower,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - expected_upper).abs() < 1e-5,
            "cos(π/4) should be {}, got {}",
            expected_upper,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_ibp_contains_minimum() {
    use std::f32::consts::PI;
    // Test cos on interval that contains π (minimum)
    let lower = ArrayD::from_elem(IxDyn(&[1]), PI / 2.0);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 3.0 * PI / 2.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let cos_layer = CosLayer::new();
    let output = cos_layer.propagate_ibp(&input).unwrap();

    // Interval contains π where cos=-1
    assert!(
        (output.lower()[[0]] - (-1.0)).abs() < 1e-5,
        "cos min should be -1 when interval contains π, got {}",
        output.lower()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_cos_sound_mode_uses_ibp_constant_bounds() {
    use std::f32::consts::PI;
    let cos_layer = CosLayer::new().with_sound_mode(true);

    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, PI / 6.0, PI]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![PI / 4.0, PI / 3.0, 4.0 * PI / 3.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let result = cos_layer
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();

    assert!(
        result.lower_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero lower_a in sound-mode constant bounds"
    );
    assert!(
        result.upper_a.iter().all(|v| v.abs() <= 1e-8),
        "Expected zero upper_a in sound-mode constant bounds"
    );

    let ibp_bounds = cos_layer.propagate_ibp(&input).unwrap();
    let ibp_flat = ibp_bounds.flatten();
    let ibp_lower = ibp_flat
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let ibp_upper = ibp_flat
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    for i in 0..3 {
        assert!(
            (result.lower_b[i] - ibp_lower[i]).abs() <= 1e-6,
            "Sound-mode lower_b mismatch at {}: {} vs {}",
            i,
            result.lower_b[i],
            ibp_lower[i]
        );
        assert!(
            (result.upper_b[i] - ibp_upper[i]).abs() <= 1e-6,
            "Sound-mode upper_b mismatch at {}: {} vs {}",
            i,
            result.upper_b[i],
            ibp_upper[i]
        );
    }
}

/// Regression test for #3316: sin IBP bounds must stay in [-1, 1] even when
/// directed rounding pushes endpoint evaluations past the range boundary.
/// sin(π/2 - ε) ≈ 1, next_up_f32(≈1.0) could exceed 1.0.
#[ntest::timeout(10000)]
#[test]
fn test_sin_ibp_near_extrema_range_clamp_3316() {
    use std::f32::consts::PI;
    // Interval near π/2 where sin is close to 1 but doesn't contain the extremum
    let lower = ArrayD::from_elem(IxDyn(&[1]), PI / 2.0 - 0.001);
    let upper = ArrayD::from_elem(IxDyn(&[1]), PI / 2.0 - 0.0001);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let sin_layer = SinLayer::new();
    let output = sin_layer.propagate_ibp(&input).unwrap();

    assert!(
        output.lower()[[0]] >= -1.0,
        "sin lower bound must be >= -1, got {}",
        output.lower()[[0]]
    );
    assert!(
        output.upper()[[0]] <= 1.0,
        "sin upper bound must be <= 1, got {}",
        output.upper()[[0]]
    );
}

/// Regression test for #3316: cos IBP bounds must stay in [-1, 1] even when
/// directed rounding pushes endpoint evaluations past the range boundary.
#[ntest::timeout(10000)]
#[test]
fn test_cos_ibp_near_extrema_range_clamp_3316() {
    // Interval near 0 where cos is close to 1 but doesn't contain the extremum
    let lower = ArrayD::from_elem(IxDyn(&[1]), 0.0001f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 0.001f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let cos_layer = CosLayer::new();
    let output = cos_layer.propagate_ibp(&input).unwrap();

    assert!(
        output.lower()[[0]] >= -1.0,
        "cos lower bound must be >= -1, got {}",
        output.lower()[[0]]
    );
    assert!(
        output.upper()[[0]] <= 1.0,
        "cos upper bound must be <= 1, got {}",
        output.upper()[[0]]
    );
}
