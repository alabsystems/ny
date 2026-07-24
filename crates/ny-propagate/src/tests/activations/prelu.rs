// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{Array1, ArrayD, IxDyn};

// ==================== PReLU tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_prelu_ibp_all_positive() {
    // When input is all positive, PReLU is identity regardless of slope.
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 0.5]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 4.0, 1.5]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let prelu = PReluLayer::from_scalar(0.25);
    let output = prelu.propagate_ibp(&input).unwrap();

    // All positive: output = input
    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - input.lower()[[i]]).abs() < 1e-6,
            "PReLU positive lower[{}]: expected {}, got {}",
            i,
            input.lower()[[i]],
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - input.upper()[[i]]).abs() < 1e-6,
            "PReLU positive upper[{}]: expected {}, got {}",
            i,
            input.upper()[[i]],
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_prelu_ibp_all_negative() {
    // When input is all negative, PReLU scales by slope.
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-4.0, -3.0, -2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, -0.5, -0.1]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slope = 0.25;
    let prelu = PReluLayer::from_scalar(slope);
    let output = prelu.propagate_ibp(&input).unwrap();

    // All negative with positive slope: output = slope * input (monotonically increasing)
    for i in 0..3 {
        let expected_lower = slope * input.lower()[[i]];
        let expected_upper = slope * input.upper()[[i]];
        assert!(
            (output.lower()[[i]] - expected_lower).abs() < 1e-6,
            "PReLU negative lower[{}]: expected {}, got {}",
            i,
            expected_lower,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - expected_upper).abs() < 1e-6,
            "PReLU negative upper[{}]: expected {}, got {}",
            i,
            expected_upper,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_prelu_ibp_crossing_zero() {
    // Input crosses zero: [-2, 3]. With slope 0.25:
    // prelu(-2) = 0.25 * -2 = -0.5, prelu(3) = 3
    // So output bounds = [-0.5, 3]
    let lower = ArrayD::from_elem(IxDyn(&[1]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slope = 0.25;
    let prelu = PReluLayer::from_scalar(slope);
    let output = prelu.propagate_ibp(&input).unwrap();

    let expected_lower = slope * (-2.0); // -0.5
    let expected_upper = 3.0;
    assert!(
        (output.lower()[[0]] - expected_lower).abs() < 1e-6,
        "PReLU crossing lower: expected {}, got {}",
        expected_lower,
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - expected_upper).abs() < 1e-6,
        "PReLU crossing upper: expected {}, got {}",
        expected_upper,
        output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_prelu_ibp_negative_slope_crossing() {
    // Negative slope (slope < 0) changes behavior in negative region.
    // For slope=-0.5 and input [-2, 3]:
    //   prelu(-2) = -0.5 * -2 = 1.0
    //   prelu(-1) = -0.5 * -1 = 0.5
    //   prelu(0) = 0
    //   prelu(3) = 3
    // With negative slope and crossing: min at x=0 (PReLU(0) = 0),
    // max at endpoints: max(slope*l, u) = max(1.0, 3.0) = 3.0.
    // So correct bounds = [0.0, 3.0].
    // Part of #1914: previously computed lower = -1.5 (applied slope to positive u).
    let lower = ArrayD::from_elem(IxDyn(&[1]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 3.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slope = -0.5;
    let prelu = PReluLayer::from_scalar(slope);
    let output = prelu.propagate_ibp(&input).unwrap();

    // Exact bounds after #1914 fix
    assert!(
        (output.lower()[[0]] - 0.0).abs() < 1e-6,
        "PReLU neg slope crossing: lower should be 0.0, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 3.0).abs() < 1e-6,
        "PReLU neg slope crossing: upper should be 3.0, got {}",
        output.upper()[[0]]
    );

    // Also verify bounds are sound at sample points
    let test_points = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    for x in &test_points {
        let y = if *x >= 0.0 { *x } else { slope * x };
        assert!(
            output.lower()[[0]] <= y + 1e-6,
            "PReLU neg slope: lower {} > f({}) = {}",
            output.lower()[[0]],
            x,
            y
        );
        assert!(
            output.upper()[[0]] >= y - 1e-6,
            "PReLU neg slope: upper {} < f({}) = {}",
            output.upper()[[0]],
            x,
            y
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_prelu_ibp_per_channel_slopes() {
    // Per-channel slopes: different slope per neuron
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -1.0, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 1.0, 3.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slopes = Array1::from_vec(vec![0.1, 0.5, 2.0]);
    let prelu = PReluLayer::new(slopes).expect("invariant: non-empty slope");
    let output = prelu.propagate_ibp(&input).unwrap();

    // Channel 0: all negative, slope 0.1 → [-0.2, -0.1]
    assert!((output.lower()[[0]] - (-0.2)).abs() < 1e-6);
    assert!((output.upper()[[0]] - (-0.1)).abs() < 1e-6);

    // Channel 1: crossing, slope 0.5 → [-0.5, 1.0]
    assert!((output.lower()[[1]] - (-0.5)).abs() < 1e-6);
    assert!((output.upper()[[1]] - 1.0).abs() < 1e-6);

    // Channel 2: all positive → identity [1.0, 3.0]
    assert!((output.lower()[[2]] - 1.0).abs() < 1e-6);
    assert!((output.upper()[[2]] - 3.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_prelu_linear_requires_preactivation_bounds() {
    let bounds = LinearBounds::identity(4);
    let prelu = PReluLayer::from_scalar(0.25);
    let result = prelu.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "PReLU::propagate_linear should error without pre-activation bounds"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_prelu_crown_soundness() {
    // Test CROWN backward bounds for PReLU with slope=0.25.
    // Pre-activation intervals: [-2, 3] (crossing), [-1, -0.5] (negative), [1, 2] (positive)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -1.0, 1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, -0.5, 2.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let slope = 0.25;
    let prelu = PReluLayer::from_scalar(slope);

    let result = prelu
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let intervals = [(-2.0f32, 3.0f32), (-1.0, -0.5), (1.0, 2.0)];
    let tol = 1e-4;

    for (j, (l, u)) in intervals.iter().enumerate() {
        // Sample 21 points in [l, u]
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let x = l + (u - l) * t;
            let fx = if x >= 0.0 { x } else { slope * x };

            // CROWN bounds: lower_a[j,j] * x + lower_b[j] and upper_a[j,j] * x + upper_b[j]
            // Since we used identity, only diagonal terms matter
            let lb = result.lower_a[[j, j]] * x + result.lower_b[j];
            let ub = result.upper_a[[j, j]] * x + result.upper_b[j];

            assert!(
                lb <= fx + tol,
                "PReLU CROWN lower violated at x={}: lb={} > f(x)={}",
                x,
                lb,
                fx
            );
            assert!(
                ub >= fx - tol,
                "PReLU CROWN upper violated at x={}: ub={} < f(x)={}",
                x,
                ub,
                fx
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_prelu_crown_negative_slope() {
    // Test CROWN with negative slope (slope=-0.5) on crossing interval [-3, 2]
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), -3.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 2.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let slope = -0.5;
    let prelu = PReluLayer::from_scalar(slope);

    let result = prelu
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let tol = 1e-4;
    for i in 0..=30 {
        let t = i as f32 / 30.0;
        let x = -3.0 + 5.0 * t;
        let fx = if x >= 0.0 { x } else { slope * x };

        let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];

        assert!(
            lb <= fx + tol,
            "PReLU neg slope CROWN lower violated at x={}: lb={} > f(x)={}",
            x,
            lb,
            fx
        );
        assert!(
            ub >= fx - tol,
            "PReLU neg slope CROWN upper violated at x={}: ub={} < f(x)={}",
            x,
            ub,
            fx
        );
    }
}
