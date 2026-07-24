// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Floor CROWN tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_floor_crown_constant_bounds() {
    // Floor is piecewise constant, so CROWN should produce slope=0 bounds
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.2, -0.8, 2.9, -2.1]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.8, 0.3, 3.5, -1.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let floor = FloorLayer;

    let result = floor
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // CROWN for discontinuous functions: slope = 0, intercept = f(bound)
    // All coefficients should be zero (constant bounds)
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                result.lower_a[[i, j]].abs() < 1e-6,
                "Floor CROWN lower slope should be 0, got {} at [{},{}]",
                result.lower_a[[i, j]],
                i,
                j
            );
            assert!(
                result.upper_a[[i, j]].abs() < 1e-6,
                "Floor CROWN upper slope should be 0, got {} at [{},{}]",
                result.upper_a[[i, j]],
                i,
                j
            );
        }
    }

    // Check intercepts match IBP bounds
    // floor([1.2, 1.8]) = [1, 1], floor([-0.8, 0.3]) = [-1, 0],
    // floor([2.9, 3.5]) = [2, 3], floor([-2.1, -1.5]) = [-3, -2]
    let expected_lower = [1.0, -1.0, 2.0, -3.0];
    let expected_upper = [1.0, 0.0, 3.0, -2.0];

    for i in 0..4 {
        assert!(
            (result.lower_b[i] - expected_lower[i]).abs() < 1e-6,
            "Floor CROWN lower intercept mismatch at {}: got {}, expected {}",
            i,
            result.lower_b[i],
            expected_lower[i]
        );
        assert!(
            (result.upper_b[i] - expected_upper[i]).abs() < 1e-6,
            "Floor CROWN upper intercept mismatch at {}: got {}, expected {}",
            i,
            result.upper_b[i],
            expected_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_floor_crown_soundness() {
    // Test that CROWN bounds are sound (contain true outputs)
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.5, 0.2, 2.7]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.8, 3.3]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let floor = FloorLayer;

    let result = floor
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample test points within the input bounds
    let test_points: [Vec<f32>; 3] = [
        vec![-1.5, 0.2, 2.7], // lower
        vec![0.5, 1.8, 3.3],  // upper
        vec![-0.5, 1.0, 3.0], // middle
    ];

    for point in &test_points {
        let floor_output: Vec<f32> = point.iter().map(|x| x.floor()).collect();

        for (j, &floor_val) in floor_output.iter().enumerate() {
            // Since slope=0, bound = intercept (constant)
            let lower_bound = result.lower_b[j];
            let upper_bound = result.upper_b[j];

            assert!(
                floor_val >= lower_bound - 1e-6,
                "Floor output {} should be >= lower bound {}",
                floor_val,
                lower_bound
            );
            assert!(
                floor_val <= upper_bound + 1e-6,
                "Floor output {} should be <= upper bound {}",
                floor_val,
                upper_bound
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_floor_crown_network_integration() {
    // Test Floor in a simple network with CROWN propagation
    let weight = ndarray::Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    let bias = Some(ndarray::Array1::from_vec(vec![0.5, -0.5, 0.0, 0.0]));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::Floor(FloorLayer));

    let input_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0, 0.0, 1.0, -1.0]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 1.0, 2.0, 0.0]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    let crown_result = network.propagate_crown(&input).unwrap();
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // CROWN bounds should equal IBP bounds for floor (constant relaxation)
    for i in 0..4 {
        assert!(
            (crown_result.lower()[[i]] - ibp_result.lower()[[i]]).abs() < 1e-4,
            "Floor CROWN lower should match IBP: {} vs {}",
            crown_result.lower()[[i]],
            ibp_result.lower()[[i]]
        );
        assert!(
            (crown_result.upper()[[i]] - ibp_result.upper()[[i]]).abs() < 1e-4,
            "Floor CROWN upper should match IBP: {} vs {}",
            crown_result.upper()[[i]],
            ibp_result.upper()[[i]]
        );
    }
}
