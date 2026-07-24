// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Shrink CROWN tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_shrink_crown_backward_dead_zone() {
    // When all pre-activations are in dead zone, should be zero
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.3, -0.2, 0.1]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.2, 0.3, 0.4]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let shrink = ShrinkLayer::new(0.0, 0.5); // bias=0, lambd=0.5

    let result = shrink
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All in dead zone [-0.5, 0.5], should be zero
    for i in 0..3 {
        assert!(
            result.lower_a[[i, i]].abs() < 1e-6,
            "Dead zone should have slope 0"
        );
        assert!(
            result.upper_a[[i, i]].abs() < 1e-6,
            "Dead zone should have slope 0"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_shrink_crown_backward_positive_piece() {
    // When all pre-activations are in positive piece (x > lambd)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 4.0, 5.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let bias = 0.1;
    let lambd = 0.5;
    let shrink = ShrinkLayer::new(bias, lambd);

    let result = shrink
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All in positive piece, should be y = x - bias (slope=1, intercept=-bias)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 1.0).abs() < 1e-6,
            "Positive piece should have slope 1"
        );
        assert!(
            (result.upper_a[[i, i]] - 1.0).abs() < 1e-6,
            "Positive piece should have slope 1"
        );
    }
    // Intercept should be -bias
    for &b in result.lower_b.iter() {
        assert!((b - (-bias)).abs() < 1e-6, "Intercept should be -bias");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_shrink_crown_backward_negative_piece() {
    // When all pre-activations are in negative piece (x < -lambd)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-4.0, -3.0, -2.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, -0.8, -0.7]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let bias = 0.2;
    let lambd = 0.5;
    let shrink = ShrinkLayer::new(bias, lambd);

    let result = shrink
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // All in negative piece, should be y = x + bias (slope=1, intercept=+bias)
    for i in 0..3 {
        assert!(
            (result.lower_a[[i, i]] - 1.0).abs() < 1e-6,
            "Negative piece should have slope 1"
        );
        assert!(
            (result.upper_a[[i, i]] - 1.0).abs() < 1e-6,
            "Negative piece should have slope 1"
        );
    }
    // Intercept should be +bias
    for &b in result.lower_b.iter() {
        assert!((b - bias).abs() < 1e-6, "Intercept should be +bias");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_shrink_crown_soundness() {
    // Test that CROWN bounds are sound (contain true outputs)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -0.3, 0.2, 1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.8, 0.4, 1.5, 3.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let bias = 0.1;
    let lambd = 0.5;
    let shrink = ShrinkLayer::new(bias, lambd);

    let result = shrink
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points in the input range and verify bounds hold
    // Pre-activation bounds: dim0=[-2.0,-0.8], dim1=[-0.3,0.4], dim2=[0.2,1.5], dim3=[1.0,3.0]
    let test_points: [Vec<f32>; 5] = [
        vec![-2.0, -0.3, 0.2, 1.0], // lower bounds
        vec![-0.8, 0.4, 1.5, 3.0],  // upper bounds
        vec![-1.5, 0.0, 0.8, 2.0],  // middle
        vec![-0.9, 0.3, 0.5, 1.5],  // within bounds (dim0 in neg piece, dim1 in dead zone, etc)
        vec![-1.0, 0.1, 1.0, 1.5],  // mixed
    ];

    for point in &test_points {
        // Shrink: y = x - bias if x > lambd, x + bias if x < -lambd, else 0
        let shrink_output: Vec<f32> = point
            .iter()
            .map(|&x| {
                if x > lambd {
                    x - bias
                } else if x < -lambd {
                    x + bias
                } else {
                    0.0
                }
            })
            .collect();

        // Check each output dimension
        for (j, &shrink_val) in shrink_output.iter().enumerate() {
            let lb_val: f32 = (0..4)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..4)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= shrink_val + tol,
                "Lower bound violated at point {:?}: lb {} > shrink {}",
                point,
                lb_val,
                shrink_val
            );
            assert!(
                ub_val >= shrink_val - tol,
                "Upper bound violated at point {:?}: ub {} < shrink {}",
                point,
                ub_val,
                shrink_val
            );
        }
    }
}
