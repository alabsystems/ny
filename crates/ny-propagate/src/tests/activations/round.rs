// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Round CROWN tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_round_crown_constant_bounds() {
    // Round is piecewise constant, so CROWN should produce slope=0 bounds
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.2, -0.8, 2.9, -2.1]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.8, 0.3, 3.5, -1.5]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(4);
    let round = RoundLayer;

    let result = round
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // CROWN for discontinuous functions: slope = 0
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                result.lower_a[[i, j]].abs() < 1e-6,
                "Round CROWN lower slope should be 0"
            );
            assert!(
                result.upper_a[[i, j]].abs() < 1e-6,
                "Round CROWN upper slope should be 0"
            );
        }
    }

    // Check intercepts match IBP bounds
    // round([1.2, 1.8]) = [1, 2], round([-0.8, 0.3]) = [-1, 0],
    // round([2.9, 3.5]) = [3, 4], round([-2.1, -1.5]) = [-2, -2]
    let expected_lower = [1.0, -1.0, 3.0, -2.0];
    let expected_upper = [2.0, 0.0, 4.0, -2.0];

    for i in 0..4 {
        assert!(
            (result.lower_b[i] - expected_lower[i]).abs() < 1e-6,
            "Round CROWN lower intercept mismatch at {}: got {}, expected {}",
            i,
            result.lower_b[i],
            expected_lower[i]
        );
        assert!(
            (result.upper_b[i] - expected_upper[i]).abs() < 1e-6,
            "Round CROWN upper intercept mismatch at {}: got {}, expected {}",
            i,
            result.upper_b[i],
            expected_upper[i]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_round_crown_soundness() {
    // Test that CROWN bounds are sound
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.5, 0.2, 2.7]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.5, 1.8, 3.3]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let round = RoundLayer;

    let result = round
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample test points
    let test_points: [Vec<f32>; 3] = [
        vec![-1.5, 0.2, 2.7],
        vec![0.5, 1.8, 3.3],
        vec![-0.5, 1.0, 3.0],
    ];

    for point in &test_points {
        let round_output: Vec<f32> = point.iter().map(|x| x.round()).collect();

        for (j, &round_val) in round_output.iter().enumerate() {
            let lower_bound = result.lower_b[j];
            let upper_bound = result.upper_b[j];

            assert!(
                round_val >= lower_bound - 1e-6,
                "Round output {} should be >= lower bound {}",
                round_val,
                lower_bound
            );
            assert!(
                round_val <= upper_bound + 1e-6,
                "Round output {} should be <= upper bound {}",
                round_val,
                upper_bound
            );
        }
    }
}
