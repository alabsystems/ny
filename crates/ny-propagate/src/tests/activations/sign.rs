// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Sign CROWN tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_sign_crown_constant_bounds() {
    // Sign is piecewise constant with values in {-1, 0, 1}
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0, -2.0, -0.5, 0.0, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[5]), vec![2.0, -1.0, 0.5, 0.5, 0.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(5);
    let sign = SignLayer;

    let result = sign
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // CROWN for Sign: slopes are 0 everywhere except for boundary cases
    // [0, u] and [l, 0] which have non-zero slopes on the diagonal.
    // Reference: sign_crown_relaxation in piecewise_constant.rs, Part of #3769.
    //
    // Test elements:
    //   0: [1.0, 2.0] → positive → slopes 0
    //   1: [-2.0, -1.0] → negative → slopes 0
    //   2: [-0.5, 0.5] → spans zero → slopes 0
    //   3: [0.0, 0.5] → boundary l=0,u>0 → lower_slope=1/0.5=2.0
    //   4: [-1.0, 0.0] → boundary l<0,u=0 → upper_slope=1/1.0=1.0
    let expected_lower_slope = [0.0, 0.0, 0.0, 1.0 / 0.5, 0.0];
    let expected_upper_slope = [0.0, 0.0, 0.0, 0.0, 1.0 / 1.0];
    for i in 0..5 {
        for j in 0..5 {
            let expected_l = if i == j { expected_lower_slope[i] } else { 0.0 };
            let expected_u = if i == j { expected_upper_slope[i] } else { 0.0 };
            assert!(
                (result.lower_a[[i, j]] - expected_l).abs() < 1e-6,
                "Sign CROWN lower slope at [{i},{j}]: got {}, expected {expected_l}",
                result.lower_a[[i, j]],
            );
            assert!(
                (result.upper_a[[i, j]] - expected_u).abs() < 1e-6,
                "Sign CROWN upper slope at [{i},{j}]: got {}, expected {expected_u}",
                result.upper_a[[i, j]],
            );
        }
    }

    // Check intercepts:
    // [1, 2]: positive -> sign = 1
    // [-2, -1]: negative -> sign = -1
    // [-0.5, 0.5]: crosses zero -> sign in [-1, 1]
    // [0, 0.5]: zero and positive -> sign in [0, 1]
    // [-1, 0]: negative and zero -> sign in [-1, 0]
    let expected_lower = [1.0, -1.0, -1.0, 0.0, -1.0];
    let expected_upper = [1.0, -1.0, 1.0, 1.0, 0.0];

    for i in 0..5 {
        assert!(
            (result.lower_b[i] - expected_lower[i]).abs() < 1e-6,
            "Sign CROWN lower intercept mismatch at {}: got {}, expected {}",
            i,
            result.lower_b[i],
            expected_lower[i]
        );
        assert!(
            (result.upper_b[i] - expected_upper[i]).abs() < 1e-6,
            "Sign CROWN upper intercept mismatch at {}: got {}, expected {}",
            i,
            result.upper_b[i],
            expected_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sign_crown_soundness() {
    // Test that CROWN bounds are sound
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-2.0, -0.5, 0.5, -1.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![-0.5, 1.0, 2.0, 0.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let sign = SignLayer;

    let result = sign
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample test points
    let test_points: [Vec<f32>; 3] = [
        vec![-2.0, -0.5, 0.5, -1.0], // lower
        vec![-0.5, 1.0, 2.0, 0.5],   // upper
        vec![-1.0, 0.0, 1.0, 0.0],   // middle with zero
    ];

    for point in &test_points {
        let sign_output: Vec<f32> = point.iter().map(|x| x.signum()).collect();

        for (j, &sign_val) in sign_output.iter().enumerate() {
            let lower_bound = result.lower_b[j];
            let upper_bound = result.upper_b[j];

            assert!(
                sign_val >= lower_bound - 1e-6,
                "Sign output {} should be >= lower bound {}",
                sign_val,
                lower_bound
            );
            assert!(
                sign_val <= upper_bound + 1e-6,
                "Sign output {} should be <= upper bound {}",
                sign_val,
                upper_bound
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_sign_crown_network_integration() {
    // Test Sign in a simple network with CROWN propagation
    let weight =
        ndarray::Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            .unwrap();
    let bias = Some(ndarray::Array1::from_vec(vec![0.0, 0.0, 0.0]));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::Sign(SignLayer));

    let input_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-2.0, -0.5, 1.0]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.5, 0.5, 2.0]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    let crown_result = network.propagate_crown(&input).unwrap();
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // CROWN bounds should equal IBP bounds for sign (constant relaxation)
    for i in 0..3 {
        assert!(
            (crown_result.lower()[[i]] - ibp_result.lower()[[i]]).abs() < 1e-4,
            "Sign CROWN lower should match IBP: {} vs {}",
            crown_result.lower()[[i]],
            ibp_result.lower()[[i]]
        );
        assert!(
            (crown_result.upper()[[i]] - ibp_result.upper()[[i]]).abs() < 1e-4,
            "Sign CROWN upper should match IBP: {} vs {}",
            crown_result.upper()[[i]],
            ibp_result.upper()[[i]]
        );
    }
}
