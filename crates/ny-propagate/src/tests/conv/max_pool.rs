// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, Array1, ArrayD};

// ============================================================
// MAXPOOL2D TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_maxpool2d_ibp_concrete() {
    // Test max pooling with concrete (non-interval) input
    let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // Input: 4x4 with values 1-16
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 4, 4]));
    for h in 0..4 {
        for w in 0..4 {
            input_data[[0, h, w]] = (h * 4 + w + 1) as f32;
        }
    }
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape: (1, 2, 2)
    assert_eq!(output.shape(), &[1, 2, 2]);

    // MaxPool 2x2 stride 2 on 4x4:
    // [1,2,5,6] -> max=6, [3,4,7,8] -> max=8
    // [9,10,13,14] -> max=14, [11,12,15,16] -> max=16
    assert!((output.lower()[[0, 0, 0]] - 6.0).abs() < 1e-6);
    assert!((output.lower()[[0, 0, 1]] - 8.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1, 0]] - 14.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1, 1]] - 16.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool2d_ibp_rejects_rank5_input() {
    let maxpool = MaxPool2dLayer::new((2, 2), (1, 1), (0, 0));
    let lower = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 1, 1, 1]));
    let upper = ArrayD::from_elem(ndarray::IxDyn(&[1, 1, 1, 1, 1]), 1.0);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let err = maxpool.propagate_ibp(&input).unwrap_err();
    assert!(
        err.to_string()
            .contains("MaxPool2d IBP requires 3D or 4D input"),
        "unexpected error: {}",
        err
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool2d_ibp_interval() {
    // Test max pooling with interval input
    let maxpool = MaxPool2dLayer::new((2, 2), (1, 1), (0, 0));

    // Input: 3x3 with bounds [i, i+1] for each position i
    let mut lower_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    let mut upper_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    for h in 0..3 {
        for w in 0..3 {
            let i = (h * 3 + w) as f32;
            lower_data[[0, h, w]] = i;
            upper_data[[0, h, w]] = i + 1.0;
        }
    }
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape: (1, 2, 2) for 3x3 input with 2x2 kernel stride 1
    assert_eq!(output.shape(), &[1, 2, 2]);

    // Position (0,0) pools from (0,0), (0,1), (1,0), (1,1)
    // Lower bounds: 0, 1, 3, 4 -> max = 4
    // Upper bounds: 1, 2, 4, 5 -> max = 5
    assert!((output.lower()[[0, 0, 0]] - 4.0).abs() < 1e-6);
    assert!((output.upper()[[0, 0, 0]] - 5.0).abs() < 1e-6);

    // Position (0,1) pools from (0,1), (0,2), (1,1), (1,2)
    // Lower bounds: 1, 2, 4, 5 -> max = 5
    // Upper bounds: 2, 3, 5, 6 -> max = 6
    assert!((output.lower()[[0, 0, 1]] - 5.0).abs() < 1e-6);
    assert!((output.upper()[[0, 0, 1]] - 6.0).abs() < 1e-6);

    // Position (1,0) pools from (1,0), (1,1), (2,0), (2,1)
    // Lower bounds: 3, 4, 6, 7 -> max = 7
    // Upper bounds: 4, 5, 7, 8 -> max = 8
    assert!((output.lower()[[0, 1, 0]] - 7.0).abs() < 1e-6);
    assert!((output.upper()[[0, 1, 0]] - 8.0).abs() < 1e-6);

    // Position (1,1) pools from (1,1), (1,2), (2,1), (2,2)
    // Lower bounds: 4, 5, 7, 8 -> max = 8
    // Upper bounds: 5, 6, 8, 9 -> max = 9
    assert!((output.lower()[[0, 1, 1]] - 8.0).abs() < 1e-6);
    assert!((output.upper()[[0, 1, 1]] - 9.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool2d_soundness() {
    // Soundness test: verify concrete outputs are within bounds
    let maxpool = MaxPool2dLayer::new((2, 2), (1, 1), (0, 0));

    // Input with perturbation
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::from_epsilon(center.clone(), 0.1).unwrap();

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Test several concrete points within input bounds
    for offset in [-0.1, -0.05, 0.0, 0.05, 0.1] {
        let concrete_input = center.clone().mapv(|v| v + offset);

        // Manually compute max pool
        for oh in 0..3 {
            for ow in 0..3 {
                let mut max_val = f32::NEG_INFINITY;
                for kh in 0..2 {
                    for kw in 0..2 {
                        max_val = max_val.max(concrete_input[[0, oh + kh, ow + kw]]);
                    }
                }

                // Verify concrete output is within bounds
                assert!(
                    max_val >= output.lower()[[0, oh, ow]] - 1e-6,
                    "Concrete {} should be >= lower {}",
                    max_val,
                    output.lower()[[0, oh, ow]]
                );
                assert!(
                    max_val <= output.upper()[[0, oh, ow]] + 1e-6,
                    "Concrete {} should be <= upper {}",
                    max_val,
                    output.upper()[[0, oh, ow]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool2d_stride() {
    // Test max pooling with stride
    let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // 6x6 input -> 3x3 output with stride 2
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 6, 6]));
    for h in 0..6 {
        for w in 0..6 {
            input_data[[0, h, w]] = (h * 6 + w) as f32;
        }
    }
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape: (1, 3, 3)
    assert_eq!(output.shape(), &[1, 3, 3]);

    // Position (0,0) pools from (0,0), (0,1), (1,0), (1,1): 0,1,6,7 -> max=7
    assert!((output.lower()[[0, 0, 0]] - 7.0).abs() < 1e-6);

    // Position (0,1) pools from (0,2), (0,3), (1,2), (1,3): 2,3,8,9 -> max=9
    assert!((output.lower()[[0, 0, 1]] - 9.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool2d_padding() {
    // Test max pooling with padding
    let maxpool = MaxPool2dLayer::new((3, 3), (1, 1), (1, 1));

    // 3x3 input with padding 1 -> 3x3 output
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    for h in 0..3 {
        for w in 0..3 {
            input_data[[0, h, w]] = (h * 3 + w + 1) as f32;
        }
    }
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape should be (1, 3, 3)
    assert_eq!(output.shape(), &[1, 3, 3]);

    // Center position (1,1) sees all 9 values -> max=9
    assert!((output.lower()[[0, 1, 1]] - 9.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool2d_multi_channel() {
    // Test max pooling with multiple channels
    let maxpool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // 2 channels, 4x4 each
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[2, 4, 4]));
    for c in 0..2 {
        for h in 0..4 {
            for w in 0..4 {
                input_data[[c, h, w]] = ((c + 1) * 100 + h * 4 + w) as f32;
            }
        }
    }
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = maxpool.propagate_ibp(&input).unwrap();

    // Output shape: (2, 2, 2)
    assert_eq!(output.shape(), &[2, 2, 2]);

    // Channel 0: values 100-115, max of first 2x2 block = 105 (100+1+4)
    // Wait, let me recalculate: positions (0,0), (0,1), (1,0), (1,1)
    // Values: 100, 101, 104, 105 -> max = 105
    assert!((output.lower()[[0, 0, 0]] - 105.0).abs() < 1e-6);

    // Channel 1: values 200-215, same pattern
    assert!((output.lower()[[1, 0, 0]] - 205.0).abs() < 1e-6);
}

// =============================================================================
// MaxPool2d CROWN Tests
// =============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_max_pool_crown_backward_basic() {
    // Test MaxPool2d CROWN backward propagation coefficient structure
    use crate::layers::MaxPool2dLayer;
    use crate::LinearBounds;

    // Create a 2x2 max pool with stride 2 (non-overlapping)
    // Input: 1 channel, 4x4 -> Output: 1 channel, 2x2
    let max_pool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // Input shape: [1, 4, 4] = 16 elements
    // Output shape: [1, 2, 2] = 4 elements
    let input_size = 16;
    let output_size = 4;

    // Create identity linear bounds at the output
    let bounds = LinearBounds::identity(output_size);

    // Create input bounds where one element in each window is clearly larger
    // Window 0 (positions 0,1,4,5): element 0 has highest range
    // Window 1 (positions 2,3,6,7): element 2 has highest range
    // etc.
    let mut lower = vec![0.0f32; input_size];
    let mut upper = vec![0.5f32; input_size];

    // Make element 0 the clear winner in window 0 (top-left)
    lower[0] = 1.0;
    upper[0] = 2.0;

    // Make element 3 the clear winner in window 1 (top-right)
    lower[3] = 1.0;
    upper[3] = 2.0;

    // Make element 8 the clear winner in window 2 (bottom-left)
    lower[8] = 1.0;
    upper[8] = 2.0;

    // Make element 15 the clear winner in window 3 (bottom-right)
    lower[15] = 1.0;
    upper[15] = 2.0;

    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 4]), lower).unwrap(),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 4]), upper).unwrap(),
    )
    .unwrap();

    let result = max_pool
        .propagate_linear_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // With clear winners, gradient should flow through winners only
    // Output 0 (window 0) should have gradient through input 0 only
    let tol = 1e-5;
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < tol,
        "Expected 1.0 at [0,0], got {}",
        result.lower_a[[0, 0]]
    );
    // Other inputs in window 0 should have 0 gradient
    assert!(result.lower_a[[0, 1]].abs() < tol);
    assert!(result.lower_a[[0, 4]].abs() < tol);
    assert!(result.lower_a[[0, 5]].abs() < tol);

    // Output 1 (window 1) should have gradient through input 3 only
    assert!(
        (result.lower_a[[1, 3]] - 1.0).abs() < tol,
        "Expected 1.0 at [1,3], got {}",
        result.lower_a[[1, 3]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_max_pool_crown_soundness() {
    // Test that MaxPool2d CROWN bounds are sound (contain actual values)
    use crate::layers::MaxPool2dLayer;
    use ndarray::ArrayD;

    let max_pool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // Create input with some variation
    let input_lower = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 4, 4]),
        vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
        ],
    )
    .unwrap();
    let input_upper = ArrayD::from_shape_vec(
        ndarray::IxDyn(&[1, 4, 4]),
        vec![
            0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        ],
    )
    .unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    // Get IBP bounds (exact for max pool)
    let ibp_result = max_pool.propagate_ibp(&input).unwrap();

    // Test multiple concrete points within input bounds
    let test_points = [
        // Point at lower bounds
        vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
        ],
        // Point at upper bounds
        vec![
            0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
        ],
        // Point at midpoint
        vec![
            0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55,
            1.65, 1.75,
        ],
    ];

    let ibp_lower = ibp_result.lower().as_slice().unwrap();
    let ibp_upper = ibp_result.upper().as_slice().unwrap();

    for point in test_points.iter() {
        let concrete_input = BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 4]), point.clone()).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4, 4]), point.clone()).unwrap(),
        )
        .unwrap();
        let concrete_output = max_pool.propagate_ibp(&concrete_input).unwrap();
        let concrete_vals = concrete_output.lower().as_slice().unwrap();

        // Check that concrete values are within IBP bounds
        for (i, &val) in concrete_vals.iter().enumerate() {
            assert!(
                val >= ibp_lower[i] - 1e-5 && val <= ibp_upper[i] + 1e-5,
                "Concrete value {} at index {} not in IBP bounds [{}, {}]",
                val,
                i,
                ibp_lower[i],
                ibp_upper[i]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_max_pool_crown_uncertain_case() {
    // Test MaxPool2d CROWN when there's no clear winner (uses constant IBP bounds)
    use crate::layers::MaxPool2dLayer;
    use crate::LinearBounds;

    let max_pool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    // Input shape: [1, 2, 2] = 4 elements (single pooling window)
    // Output shape: [1, 1, 1] = 1 element
    let _input_size = 4;
    let output_size = 1;

    let bounds = LinearBounds::identity(output_size);

    // All inputs have overlapping intervals - no clear winner
    let lower = vec![0.0f32, 0.1, 0.2, 0.3];
    let upper = vec![1.0f32, 1.1, 1.2, 1.3];

    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 2]), lower).unwrap(),
        ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2, 2]), upper).unwrap(),
    )
    .unwrap();

    let result = max_pool
        .propagate_linear_with_bounds(&bounds, &pre_activation)
        .unwrap();

    // No definite winner. SOUND dense lower relaxation routes the LOWER row
    // linearly through i* = argmax_i l_i = index 3 (l_3 = 0.3), since
    // y = max(x) >= x_3 pointwise. The UPPER row (ua>0) stays constant.
    let tol = 1e-5;

    // Lower row: coeff 1.0 at i*=3, zero elsewhere; lower_b absorbs no constant.
    assert!(
        (result.lower_a[[0, 3]] - 1.0).abs() < tol,
        "lower row should route through i*=3 with coeff 1.0, got {}",
        result.lower_a[[0, 3]]
    );
    for i in 0..3 {
        assert!(
            result.lower_a[[0, i]].abs() < tol,
            "Expected 0 at lower_a[0,{}], got {}",
            i,
            result.lower_a[[0, i]]
        );
    }

    // Upper row UNCHANGED: no gradient flows (constant max_upper).
    for i in 0..4 {
        assert!(
            result.upper_a[[0, i]].abs() < tol,
            "Expected 0 at upper_a[0,{}], got {}",
            i,
            result.upper_a[[0, i]]
        );
    }

    // Lower bias is now 0 (routed linearly, no constant); upper bias is the
    // constant IBP max_upper = 1.3 (unchanged).
    let tol_bias = 0.05;
    assert!(
        result.lower_b[0].abs() < tol_bias,
        "Lower bias {} should be ~0 (routed linearly through x_3)",
        result.lower_b[0]
    );
    assert!(
        (result.upper_b[0] - 1.3).abs() < tol_bias,
        "Upper bias {} should be ~1.3 (max_upper, unchanged)",
        result.upper_b[0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_max_pool_crown_network_integration() {
    // Test MaxPool2d CROWN in a network context
    use crate::layers::{LinearLayer, MaxPool2dLayer, ReshapeLayer};
    use crate::network::Network;
    use ndarray::Array2;

    // Create a simple network: Reshape -> MaxPool -> Flatten -> Linear
    // Input: flat 16 elements -> reshape to [1, 4, 4] -> maxpool 2x2 -> [1, 2, 2] -> flatten -> linear

    let max_pool = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));

    let weight =
        Array2::from_shape_vec((2, 4), vec![1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
    let bias: Option<Array1<f32>> = Some(arr1(&[0.0, 0.0]));
    let linear = LinearLayer::new(weight, bias).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 4, 4])));
    network.add_layer(Layer::MaxPool2d(max_pool));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    // Create input bounds
    let input_lower = ArrayD::from_shape_vec(ndarray::IxDyn(&[16]), vec![0.0; 16]).unwrap();
    let input_upper = ArrayD::from_shape_vec(ndarray::IxDyn(&[16]), vec![1.0; 16]).unwrap();
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

    // CROWN bounds should be close to IBP bounds (might not be tighter due to approximation)
    // But they should be sound (contain the true values)
    for i in 0..crown_lower.len() {
        // Allow some slack due to approximation error in max pool CROWN
        assert!(
            crown_lower[i] >= ibp_lower[i] - 1.0,
            "CROWN lower bound {} should be >= IBP lower bound {} - 1.0",
            crown_lower[i],
            ibp_lower[i]
        );
        assert!(
            crown_upper[i] <= ibp_upper[i] + 1.0,
            "CROWN upper bound {} should be <= IBP upper bound {} + 1.0",
            crown_upper[i],
            ibp_upper[i]
        );
    }

    // Verify soundness by testing concrete points
    let test_values = vec![
        vec![0.5; 16], // midpoint
        vec![0.0; 16], // lower
        vec![1.0; 16], // upper
    ];

    for vals in test_values {
        let concrete_input = BoundedTensor::new(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[16]), vals.clone()).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[16]), vals).unwrap(),
        )
        .unwrap();
        let concrete_output = network.propagate_ibp(&concrete_input).unwrap();
        let concrete_vals = concrete_output.lower().as_slice().unwrap();

        for (i, &val) in concrete_vals.iter().enumerate() {
            assert!(
                val >= crown_lower[i] - 1e-4 && val <= crown_upper[i] + 1e-4,
                "Concrete value {} at index {} not in CROWN bounds [{}, {}]",
                val,
                i,
                crown_lower[i],
                crown_upper[i]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_crown_maxpool2d_network_does_not_panic() {
    // Regression test: β-CROWN should support MaxPool2d in sequential networks.
    //
    // This is required for CLI property reductions that introduce pooling-based max objectives.
    use crate::beta_crown::{
        BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic,
    };
    use crate::layers::{LinearLayer, MaxPool2dLayer, ReshapeLayer};
    use crate::network::Network;
    use crate::Layer;
    use ndarray::{Array2, ArrayD, IxDyn};
    use std::time::Duration;

    // Network: Reshape [16] -> [1,4,4] -> MaxPool2d -> [1,2,2] -> Reshape [4] -> Linear -> [1]
    let mut network = Network::new();
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![1, 4, 4])));
    network.add_layer(Layer::MaxPool2d(MaxPool2dLayer::new(
        (2, 2),
        (2, 2),
        (0, 0),
    )));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![4])));

    let weight = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    network.add_layer(Layer::Linear(LinearLayer::new(weight, None).unwrap()));

    let input_lower = ArrayD::from_shape_vec(IxDyn(&[16]), vec![0.0; 16]).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[16]), vec![1.0; 16]).unwrap();
    let input = BoundedTensor::new(input_lower, input_upper).unwrap();

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(5),
        max_domains: 1_000,
        max_depth: 10,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        branching_heuristic: BranchingHeuristic::LargestBoundWidth,
        batch_size: 1,
        parallel_children: false,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -0.1).unwrap();
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "Expected Verified, got {:?}",
        result.result
    );
}
