// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, Array1, ArrayD};

// ============================================================
// AVERAGE POOL CROWN TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_average_pool_crown_backward_basic() {
    // Test AveragePool CROWN backward propagation
    // Input: [1, 3, 3] (1 channel, 3x3 spatial)
    // Kernel: 2x2, Stride: 1, Padding: 0
    // Output: [1, 2, 2] (1 channel, 2x2 spatial)
    use crate::layers::AveragePoolLayer;

    let pre_lower = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 3, 3]),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 3, 3]),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let avg_pool = AveragePoolLayer::new((2, 2), (1, 1), (0, 0), false);

    // Output size is 2x2 = 4, Input size is 3x3 = 9
    let linear_bounds = LinearBounds::identity(4);

    let result = avg_pool
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Check dimensions
    assert_eq!(result.lower_a.shape(), &[4, 9]);
    assert_eq!(result.upper_a.shape(), &[4, 9]);
    assert_eq!(result.lower_b.len(), 4);
    assert_eq!(result.upper_b.len(), 4);

    // Check structure: each output averages 4 inputs
    // Output [0] = avg(input[0], input[1], input[3], input[4]) = avg of positions (0,0), (0,1), (1,0), (1,1)
    let weight = 0.25_f32; // 1/4 for 2x2 kernel
    let tol = 1e-5;

    // Lower_a for output 0 should have weight 0.25 for inputs 0, 1, 3, 4
    assert!(
        (result.lower_a[[0, 0]] - weight).abs() < tol,
        "Expected weight {} at [0,0], got {}",
        weight,
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_a[[0, 1]] - weight).abs() < tol,
        "Expected weight {} at [0,1], got {}",
        weight,
        result.lower_a[[0, 1]]
    );
    assert!(
        (result.lower_a[[0, 3]] - weight).abs() < tol,
        "Expected weight {} at [0,3], got {}",
        weight,
        result.lower_a[[0, 3]]
    );
    assert!(
        (result.lower_a[[0, 4]] - weight).abs() < tol,
        "Expected weight {} at [0,4], got {}",
        weight,
        result.lower_a[[0, 4]]
    );

    // Inputs not in the window should have weight 0
    assert!(
        result.lower_a[[0, 2]].abs() < tol,
        "Expected 0 at [0,2], got {}",
        result.lower_a[[0, 2]]
    );
    assert!(
        result.lower_a[[0, 5]].abs() < tol,
        "Expected 0 at [0,5], got {}",
        result.lower_a[[0, 5]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_average_pool_ibp_rejects_rank5_input() {
    use crate::layers::AveragePoolLayer;

    let avg_pool = AveragePoolLayer::new((2, 2), (1, 1), (0, 0), false);
    let lower = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 1, 1, 1]));
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[1, 1, 1, 1, 1]), 1.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let err = avg_pool.propagate_ibp(&input).unwrap_err();
    assert!(
        err.to_string()
            .contains("AveragePool IBP requires 3D or 4D input"),
        "unexpected error: {}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_average_pool_crown_soundness() {
    // Test that CROWN bounds are sound (contain the actual function values)
    use crate::layers::AveragePoolLayer;

    let pre_lower = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 4, 4]),
        (0..16).map(|i| i as f32).collect::<Vec<_>>(),
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 4, 4]),
        (0..16).map(|i| i as f32 + 1.0).collect::<Vec<_>>(),
    )
    .unwrap();
    let pre_activation = BoundedTensor::new(pre_lower.clone(), pre_upper.clone()).unwrap();

    let avg_pool = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);

    // Output: [1, 2, 2] = 4 elements, Input: [1, 4, 4] = 16 elements
    let linear_bounds = LinearBounds::identity(4);

    let result = avg_pool
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Sample points and verify bounds contain actual values
    for sample in 0..10 {
        // Generate a random point in the interval
        let point: Vec<f32> = (0..16)
            .map(|i| {
                let t = ((sample as u32).wrapping_mul(2654435761) ^ (i as u32)) as f32
                    / u32::MAX as f32;
                let pre_l = pre_lower.as_slice().unwrap()[i];
                let pre_u = pre_upper.as_slice().unwrap()[i];
                pre_l + (pre_u - pre_l) * t
            })
            .collect();

        // Compute actual average pool output
        // Output [0,0,0] = avg(input[0], input[1], input[4], input[5])
        // Output [0,0,1] = avg(input[2], input[3], input[6], input[7])
        // Output [0,1,0] = avg(input[8], input[9], input[12], input[13])
        // Output [0,1,1] = avg(input[10], input[11], input[14], input[15])
        let actual_output = [
            (point[0] + point[1] + point[4] + point[5]) / 4.0,
            (point[2] + point[3] + point[6] + point[7]) / 4.0,
            (point[8] + point[9] + point[12] + point[13]) / 4.0,
            (point[10] + point[11] + point[14] + point[15]) / 4.0,
        ];

        // Check each output dimension
        for (j, &actual_val) in actual_output.iter().enumerate() {
            let lb_val: f32 = (0..16)
                .map(|i| result.lower_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.lower_b[j];

            let ub_val: f32 = (0..16)
                .map(|i| result.upper_a[[j, i]] * point[i])
                .sum::<f32>()
                + result.upper_b[j];

            let tol = 1e-4;
            assert!(
                lb_val <= actual_val + tol,
                "CROWN lower bound violated at sample {}, dim {}: lb {} > actual {}",
                sample,
                j,
                lb_val,
                actual_val
            );
            assert!(
                ub_val >= actual_val - tol,
                "CROWN upper bound violated at sample {}, dim {}: ub {} < actual {}",
                sample,
                j,
                ub_val,
                actual_val
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_average_pool_global_crown() {
    // Test global average pooling CROWN
    use crate::layers::AveragePoolLayer;

    let pre_lower = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3, 3]),
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // channel 0
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // channel 1
        ],
    )
    .unwrap();
    let pre_upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[2, 3, 3]),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // channel 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // channel 1
        ],
    )
    .unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Global pooling: kernel_size (0, 0)
    let avg_pool = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);

    // Output: [2, 1, 1] = 2 elements (one per channel)
    // Input: [2, 3, 3] = 18 elements
    let linear_bounds = LinearBounds::identity(2);

    let result = avg_pool
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Check dimensions
    assert_eq!(result.lower_a.shape(), &[2, 18]);
    assert_eq!(result.upper_a.shape(), &[2, 18]);

    // Global pool averages all 9 elements per channel
    let weight = 1.0 / 9.0;
    let tol = 1e-5;

    // Channel 0 output (index 0) should have weight 1/9 for inputs 0-8
    for i in 0..9 {
        assert!(
            (result.lower_a[[0, i]] - weight).abs() < tol,
            "Expected weight {} at [0,{}], got {}",
            weight,
            i,
            result.lower_a[[0, i]]
        );
    }
    // Channel 0 output should have 0 weight for channel 1 inputs (9-17)
    for i in 9..18 {
        assert!(
            result.lower_a[[0, i]].abs() < tol,
            "Expected 0 at [0,{}], got {}",
            i,
            result.lower_a[[0, i]]
        );
    }

    // Channel 1 output (index 1) should have weight 1/9 for inputs 9-17
    for i in 9..18 {
        assert!(
            (result.lower_a[[1, i]] - weight).abs() < tol,
            "Expected weight {} at [1,{}], got {}",
            weight,
            i,
            result.lower_a[[1, i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_average_pool_crown_network_integration() {
    // Test AveragePool CROWN in a network context
    use crate::layers::{AveragePoolLayer, LinearLayer, ReshapeLayer};
    use crate::network::Network;
    use ndarray::Array2;

    // Create a simple network: Reshape -> AveragePool -> Flatten -> Linear
    // Input: flat 9 elements -> reshape to [1, 3, 3] -> avgpool 2x2 -> [1, 2, 2] -> flatten -> linear

    let avg_pool = AveragePoolLayer::new((2, 2), (1, 1), (0, 0), false);

    let weight =
        Array2::from_shape_vec((2, 4), vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let bias: Option<Array1<f32>> = Some(arr1(&[0.0, 0.0]));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 3, 3])));
    network.add_layer(Layer::AveragePool(avg_pool));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    // Create input bounds
    let input_lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[9]), vec![0.0; 9]).unwrap();
    let input_upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[9]), vec![1.0; 9]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Test CROWN propagation
    let crown_result = network.propagate_crown(&input).unwrap();

    // Test IBP propagation for comparison
    let ibp_result = network.propagate_ibp(&input).unwrap();

    // Flatten the results for comparison
    let crown_lower = crown_result.lower().as_slice().unwrap();
    let crown_upper = crown_result.upper().as_slice().unwrap();
    let ibp_lower = ibp_result.lower().as_slice().unwrap();
    let ibp_upper = ibp_result.upper().as_slice().unwrap();

    // CROWN bounds should be tighter than or equal to IBP bounds
    for i in 0..crown_lower.len() {
        assert!(
            crown_lower[i] >= ibp_lower[i] - 1e-4,
            "CROWN lower bound {} should be >= IBP lower bound {}",
            crown_lower[i],
            ibp_lower[i]
        );
        assert!(
            crown_upper[i] <= ibp_upper[i] + 1e-4,
            "CROWN upper bound {} should be <= IBP upper bound {}",
            crown_upper[i],
            ibp_upper[i]
        );
    }
}
