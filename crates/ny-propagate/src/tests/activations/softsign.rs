// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

// ==================== Softsign tests ====================

/// Softsign: x / (1 + |x|)
fn softsign_eval(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_ibp_basic() {
    // Softsign(0) = 0, Softsign(1) = 0.5
    let lower = ArrayD::from_elem(IxDyn(&[3]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = SoftsignLayer::new();
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!(
            output.lower()[[i]].abs() < 1e-6,
            "softsign(0) should be 0, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 0.5).abs() < 1e-6,
            "softsign(1) should be 0.5, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_ibp_negative() {
    // Softsign(-3) = -3/4 = -0.75, Softsign(-1) = -1/2 = -0.5
    let lower = ArrayD::from_elem(IxDyn(&[2]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = SoftsignLayer::new();
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - (-0.75)).abs() < 1e-6,
            "softsign(-3) should be -0.75, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - (-0.5)).abs() < 1e-6,
            "softsign(-1) should be -0.5, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_ibp_crossing_zero() {
    // Softsign(-2) = -2/3, Softsign(4) = 4/5 = 0.8
    let lower = ArrayD::from_elem(IxDyn(&[1]), -2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = SoftsignLayer::new();
    let output = layer.propagate_ibp(&input).unwrap();

    let expected_lower = -2.0 / 3.0;
    let expected_upper = 4.0 / 5.0;
    assert!(
        (output.lower()[[0]] - expected_lower).abs() < 1e-6,
        "softsign(-2) should be {}, got {}",
        expected_lower,
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - expected_upper).abs() < 1e-6,
        "softsign(4) should be {}, got {}",
        expected_upper,
        output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_ibp_large_values() {
    // Softsign approaches ±1 asymptotically
    let lower = ArrayD::from_elem(IxDyn(&[1]), -100.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 100.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = SoftsignLayer::new();
    let output = layer.propagate_ibp(&input).unwrap();

    // softsign(-100) = -100/101 ≈ -0.9901
    // softsign(100) = 100/101 ≈ 0.9901
    assert!(output.lower()[[0]] > -1.0, "softsign lower should be > -1");
    assert!(
        output.lower()[[0]] < -0.99,
        "softsign(-100) should be close to -1"
    );
    assert!(output.upper()[[0]] < 1.0, "softsign upper should be < 1");
    assert!(
        output.upper()[[0]] > 0.99,
        "softsign(100) should be close to 1"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_ibp_soundness_sampling() {
    // Verify IBP bounds contain all concrete evaluations
    let lower = ArrayD::from_elem(IxDyn(&[1]), -3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = SoftsignLayer::new();
    let output = layer.propagate_ibp(&input).unwrap();

    let tol = 1e-6;
    for i in 0..=50 {
        let t = i as f32 / 50.0;
        let x = -3.0 + 5.0 * t;
        let fx = softsign_eval(x);
        assert!(
            output.lower()[[0]] <= fx + tol,
            "IBP lower {} > softsign({}) = {}",
            output.lower()[[0]],
            x,
            fx
        );
        assert!(
            output.upper()[[0]] >= fx - tol,
            "IBP upper {} < softsign({}) = {}",
            output.upper()[[0]],
            x,
            fx
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_linear_requires_preactivation_bounds() {
    let bounds = LinearBounds::identity(4);
    let layer = SoftsignLayer::new();
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "Softsign::propagate_linear should error without pre-activation bounds"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_crown_soundness() {
    // Test CROWN backward bounds contain true softsign(x) for all sample points.
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-3.0, 0.0, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 2.0, 5.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let layer = SoftsignLayer::new();

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let intervals = [(-3.0f32, -1.0f32), (0.0, 2.0), (-1.0, 5.0)];
    let tol = 1e-4;

    for (j, (l, u)) in intervals.iter().enumerate() {
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let x = l + (u - l) * t;
            let fx = softsign_eval(x);

            let lb = result.lower_a[[j, j]] * x + result.lower_b[j];
            let ub = result.upper_a[[j, j]] * x + result.upper_b[j];

            assert!(
                lb <= fx + tol,
                "Softsign CROWN lower violated at x={}: lb={} > f(x)={}",
                x,
                lb,
                fx
            );
            assert!(
                ub >= fx - tol,
                "Softsign CROWN upper violated at x={}: ub={} < f(x)={}",
                x,
                ub,
                fx
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_crown_wide_interval() {
    // Wide interval: [-10, 10]
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), -10.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 10.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let layer = SoftsignLayer::new();

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let tol = 1e-3; // Wider tolerance for wide interval (sampling-based relaxation)
    for i in 0..=40 {
        let t = i as f32 / 40.0;
        let x = -10.0 + 20.0 * t;
        let fx = softsign_eval(x);

        let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];

        assert!(
            lb <= fx + tol,
            "Softsign CROWN lower violated at x={}: lb={} > f(x)={}",
            x,
            lb,
            fx
        );
        assert!(
            ub >= fx - tol,
            "Softsign CROWN upper violated at x={}: ub={} < f(x)={}",
            x,
            ub,
            fx
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_softsign_crown_nan_inf_guard() {
    // Updated for #2977: CROWN backward domain_guard rejects non-finite
    // pre-activation with NumericalInstability.
    let layer = SoftsignLayer::new();
    let linear_bounds = LinearBounds::identity(1);

    // Test with Inf lower bound
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
    )
    .unwrap();
    let result = layer.propagate_linear_with_bounds(&linear_bounds, &pre);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "Softsign with -Inf lower should trigger domain_guard: got {:?}",
        result
    );

    // Test with NaN upper bound
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), f32::NAN),
    )
    .unwrap();
    let result = layer.propagate_linear_with_bounds(&linear_bounds, &pre);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "Softsign with NaN upper should trigger domain_guard: got {:?}",
        result
    );
}
