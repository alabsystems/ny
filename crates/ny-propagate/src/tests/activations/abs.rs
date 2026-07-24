// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::activations::LinearRelaxation;
use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Abs tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_abs_ibp_positive_interval() {
    let lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let abs = AbsLayer;
    let output = abs.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!((output.lower()[[i]] - 2.0).abs() < 1e-6);
        assert!((output.upper()[[i]] - 5.0).abs() < 1e-6);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_abs_ibp_negative_interval() {
    let lower = ArrayD::from_elem(IxDyn(&[2]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let abs = AbsLayer;
    let output = abs.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!((output.lower()[[i]] - 2.0).abs() < 1e-6);
        assert!((output.upper()[[i]] - 5.0).abs() < 1e-6);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_abs_ibp_crosses_zero() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-3.0, -2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0, 1.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let abs = AbsLayer;
    let output = abs.propagate_ibp(&input).unwrap();

    assert!((output.lower()[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 4.0).abs() < 1e-6);
    assert!((output.lower()[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[1]] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_abs_linear_requires_preactivation_bounds() {
    // Without pre-activation bounds, Abs must return an error (nonlinear layer).
    // Callers must use propagate_linear_with_bounds instead.
    let bounds = LinearBounds::identity(4);
    let abs = AbsLayer;
    let result = abs.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "Abs::propagate_linear should error without pre-activation bounds"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_abs_crown_with_bounds_positive_region() {
    // All positive pre-activation bounds: |x| = x (identity)
    let pre_lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let abs = AbsLayer;

    let result = abs
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // In positive region, should be identity (slope=1, intercept=0)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 1.0).abs() < 1e-6,
            "Positive region should have slope 1"
        );
        assert!(
            (result.upper_a[[i, i]] - 1.0).abs() < 1e-6,
            "Positive region should have slope 1"
        );
    }
    assert!(result.lower_b.iter().all(|&x| x.abs() < 1e-6));
    assert!(result.upper_b.iter().all(|&x| x.abs() < 1e-6));
}

#[ntest::timeout(10000)]
#[test]
fn test_abs_crown_with_bounds_negative_region() {
    // All negative pre-activation bounds: |x| = -x (negation)
    let pre_lower = ArrayD::from_elem(IxDyn(&[3]), -5.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[3]), -2.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let abs = AbsLayer;

    let result = abs
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // In negative region, should be negation (slope=-1, intercept=0)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] + 1.0).abs() < 1e-6,
            "Negative region should have slope -1"
        );
        assert!(
            (result.upper_a[[i, i]] + 1.0).abs() < 1e-6,
            "Negative region should have slope -1"
        );
    }
    assert!(result.lower_b.iter().all(|&x| x.abs() < 1e-6));
    assert!(result.upper_b.iter().all(|&x| x.abs() < 1e-6));
}

#[ntest::timeout(10000)]
#[test]
fn test_abs_crown_soundness() {
    // Test that CROWN bounds are sound (contain true outputs)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, 0.5, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 2.0, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let abs = AbsLayer;

    let result = abs
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points in the input range and verify bounds hold
    let test_points: [Vec<f32>; 4] = [
        vec![-2.0, 0.5, -1.0], // lower
        vec![3.0, 2.0, 4.0],   // upper
        vec![0.0, 1.0, 0.0],   // middle
        vec![-1.0, 1.5, 2.0],  // random
    ];

    for point in &test_points {
        let abs_output: Vec<f32> = point.iter().map(|x| x.abs()).collect();

        // Check each output dimension
        for (j, &abs_val) in abs_output.iter().enumerate() {
            // Lower bound: lower_a * x + lower_b should be <= abs(point)
            let lb_val: f32 = (0..3)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            // Upper bound: upper_a * x + upper_b should be >= abs(point)
            let ub_val: f32 = (0..3)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= abs_val + tol,
                "Lower bound violated at point {:?}: lb {} > abs {}",
                point,
                lb_val,
                abs_val
            );
            assert!(
                ub_val >= abs_val - tol,
                "Upper bound violated at point {:?}: ub {} < abs {}",
                point,
                ub_val,
                abs_val
            );
        }
    }
}

/// Regression test for #1780: subnormal crossing intervals must keep a sound upper CROWN line.
#[ntest::timeout(10000)]
#[test]
fn test_abs_crown_subnormal_upper_soundness_1780() {
    // Counterexample region from proof/audit notes.
    let l = -1.31e-33f32;
    let u = 4.27e-34f32;
    assert!(l < 0.0 && u > 0.0);

    let LinearRelaxation {
        upper_slope,
        upper_intercept,
        ..
    } = abs_linear_relaxation(l, u);

    // Validate endpoints and interior samples in the exact interval.
    for i in 0..=256 {
        let t = i as f32 / 256.0;
        let x = l + (u - l) * t;
        let fx = x.abs();
        let ub = upper_slope * x + upper_intercept;
        assert!(
            ub >= fx,
            "Abs upper relaxation unsound at x={x:e} in [{l:e}, {u:e}]: ub={ub:e}, |x|={fx:e}, slope={upper_slope:e}, intercept={upper_intercept:e}"
        );
    }
}
