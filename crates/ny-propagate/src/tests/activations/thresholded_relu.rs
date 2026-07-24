// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== ThresholdedRelu CROWN tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_thresholded_relu_crown_backward_identity() {
    // When all pre-activations are above threshold, should be identity
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let threshold_relu = ThresholdedReluLayer::new(1.0); // alpha = 1.0

    let result = threshold_relu
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All above threshold, should be identity (slope=1, intercept=0)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 1.0).abs() < 1e-6,
            "Active region should have slope 1"
        );
        assert!(
            (result.upper_a[[i, i]] - 1.0).abs() < 1e-6,
            "Active region should have slope 1"
        );
    }
    assert!(result.lower_b.iter().all(|&x| x.abs() < 1e-6));
    assert!(result.upper_b.iter().all(|&x| x.abs() < 1e-6));
}

#[ntest::timeout(10000)]
#[test]
fn test_thresholded_relu_crown_backward_zero() {
    // When all pre-activations are at or below threshold, should be zero
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -1.0, 0.5]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.5, 0.5, 1.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let threshold_relu = ThresholdedReluLayer::new(1.0); // alpha = 1.0

    let result = threshold_relu
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All at or below threshold, should be zero
    for i in 0..3 {
        assert!(
            result.lower_a[[i, i]].abs() < 1e-6,
            "Inactive region should have slope 0"
        );
        assert!(
            result.upper_a[[i, i]].abs() < 1e-6,
            "Inactive region should have slope 0"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_thresholded_relu_crown_soundness() {
    // Test that CROWN bounds are sound (contain true outputs)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.5, 2.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 1.5, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let alpha = 1.0;
    let threshold_relu = ThresholdedReluLayer::new(alpha);

    let result = threshold_relu
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points in the input range and verify bounds hold
    let test_points: [Vec<f32>; 5] = [
        vec![-1.0, 0.5, 2.0], // lower
        vec![2.0, 1.5, 4.0],  // upper
        vec![0.0, 1.0, 3.0],  // middle
        vec![1.0, 1.0, 2.5],  // on threshold
        vec![1.5, 1.2, 3.5],  // above threshold
    ];

    for point in &test_points {
        // ThresholdedRelu: y = x if x > alpha, else 0
        let tr_output: Vec<f32> = point
            .iter()
            .map(|&x| if x > alpha { x } else { 0.0 })
            .collect();

        // Check each output dimension
        for (j, &tr_val) in tr_output.iter().enumerate() {
            let lb_val: f32 = (0..3)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..3)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= tr_val + tol,
                "Lower bound violated at point {:?}: lb {} > tr {}",
                point,
                lb_val,
                tr_val
            );
            assert!(
                ub_val >= tr_val - tol,
                "Upper bound violated at point {:?}: ub {} < tr {}",
                point,
                ub_val,
                tr_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_thresholded_relu_crown_l_eq_alpha_regression_1759() {
    // Regression for #1759: l == alpha used to trigger alpha/(alpha-l) division by zero.
    // Validate both positive and negative alpha boundaries produce finite sound bounds.
    for &alpha in &[1.0, -1.0] {
        let pre_lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![alpha]).unwrap();
        let pre_upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![alpha + 1.0]).unwrap();
        let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

        let linear_bounds = LinearBounds::identity(1);
        let threshold_relu = ThresholdedReluLayer::new(alpha);

        let result = threshold_relu
            .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
            .unwrap();

        let has_non_finite = result
            .lower_a
            .iter()
            .chain(result.upper_a.iter())
            .chain(result.lower_b.iter())
            .chain(result.upper_b.iter())
            .any(|v| !v.is_finite());
        assert!(
            !has_non_finite,
            "CROWN bounds must remain finite when l == alpha: alpha={alpha}"
        );

        for &x in &[alpha, alpha + 0.1, alpha + 0.5, alpha + 1.0] {
            let y = if x > alpha { x } else { 0.0 };
            let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
            let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];
            assert!(lb.is_finite(), "lower bound must be finite at x={x}");
            assert!(ub.is_finite(), "upper bound must be finite at x={x}");
            assert!(
                lb <= y + 1e-5,
                "lower bound violated at x={x}: lb={lb}, y={y}, alpha={alpha}"
            );
            assert!(
                ub >= y - 1e-5,
                "upper bound violated at x={x}: ub={ub}, y={y}, alpha={alpha}"
            );
        }
    }
}
