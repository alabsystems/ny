// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::network::relu_crown_relaxation;
use ny_test_utils::CountingGemmEngine;

#[ntest::timeout(10000)]
#[test]
fn test_block_progress_fraction_complete() {
    let p = BlockProgress {
        block_index: 0,
        total_blocks: 10,
        block_name: "layer0".to_string(),
        elapsed: std::time::Duration::from_secs(2),
        current_max_sensitivity: 1.0,
        degraded_so_far: 0,
    };
    assert!((p.fraction() - 0.1).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_layer_progress_fraction_complete() {
    let p = LayerProgress {
        node_index: 4,
        total_nodes: 10,
        node_name: "n4".to_string(),
        layer_type: "Linear".to_string(),
        elapsed: std::time::Duration::from_secs(2),
        current_max_sensitivity: 1.0,
        degraded_so_far: 0,
    };
    assert!((p.fraction() - 0.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_relaxation_positive() {
    let (ls, li, us, ui) = relu_crown_relaxation(1.0, 2.0);
    assert_eq!((ls, li, us, ui), (1.0, 0.0, 1.0, 0.0));
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_relaxation_negative() {
    let (ls, li, us, ui) = relu_crown_relaxation(-2.0, -1.0);
    assert_eq!((ls, li, us, ui), (0.0, 0.0, 0.0, 0.0));
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_relaxation_crossing() {
    let (_ls, _li, us, _ui) = relu_crown_relaxation(-1.0, 2.0);
    // Upper slope should be 2/(2-(-1)) = 2/3
    assert!((us - 2.0 / 3.0).abs() < 1e-6);
}

// ============================================================
// IBP TESTS FOR LINEAR LAYER
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_identity() {
    // Identity matrix: output should equal input bounds
    // W = [[1, 0], [0, 1]], b = [0, 0]
    let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let bias = arr1(&[0.0, 0.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    // Identity: output bounds should equal input bounds
    assert!((output.lower()[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 2.0).abs() < 1e-6);
    assert!((output.lower()[[1]] - 1.0).abs() < 1e-6);
    assert!((output.upper()[[1]] - 3.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_positive_weights() {
    // Simple positive weight matrix
    // W = [[1, 2], [3, 4]], b = [0, 0]
    // x in [[0, 1], [0, 1]]
    //
    // Hand calculation:
    // W+ = W (all positive), W- = 0
    // lower_y = W @ x_lower = [[1,2],[3,4]] @ [0,0] = [0, 0]
    // upper_y = W @ x_upper = [[1,2],[3,4]] @ [1,1] = [3, 7]
    let weight = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!((output.lower()[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 3.0).abs() < 1e-6);
    assert!((output.lower()[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[1]] - 7.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_mixed_weights() {
    // Mixed positive/negative weights
    // W = [[1, -1], [-2, 3]], b = [0, 0]
    // x in [[0, 1], [0, 1]]
    //
    // Hand calculation:
    // W+ = [[1, 0], [0, 3]], W- = [[0, -1], [-2, 0]]
    // lower_y = W+ @ [0,0] + W- @ [1,1] = [0,0] + [-1,-2] = [-1, -2]
    // upper_y = W+ @ [1,1] + W- @ [0,0] = [1,3] + [0,0] = [1, 3]
    let weight = arr2(&[[1.0, -1.0], [-2.0, 3.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!(
        (output.lower()[[0]] - (-1.0)).abs() < 1e-6,
        "lower[0] = {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-6,
        "upper[0] = {}",
        output.upper()[[0]]
    );
    assert!(
        (output.lower()[[1]] - (-2.0)).abs() < 1e-6,
        "lower[1] = {}",
        output.lower()[[1]]
    );
    assert!(
        (output.upper()[[1]] - 3.0).abs() < 1e-6,
        "upper[1] = {}",
        output.upper()[[1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_with_bias() {
    // W = [[1, 0], [0, 1]], b = [1, -1]
    // x in [[0, 0], [1, 1]]
    // output = W @ x + b = x + b
    // lower = [0, 0] + [1, -1] = [1, -1]
    // upper = [1, 1] + [1, -1] = [2, 0]
    let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let bias = arr1(&[1.0, -1.0]);
    let linear = LinearLayer::new(weight, Some(bias)).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!((output.lower()[[0]] - 1.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 2.0).abs() < 1e-6);
    assert!((output.lower()[[1]] - (-1.0)).abs() < 1e-6);
    assert!((output.upper()[[1]] - 0.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_asymmetric_bounds() {
    // x in [[-1, 2], [1, 3]] (non-symmetric, non-zero)
    // W = [[1, 2]], b = [0]
    // W+ = [[1, 2]], W- = [[0, 0]]
    // lower_y = W+ @ [-1, 1] + W- @ [2, 3] = [1*(-1) + 2*1] = [1]
    // upper_y = W+ @ [2, 3] + W- @ [-1, 1] = [1*2 + 2*3] = [8]
    let weight = arr2(&[[1.0, 2.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input =
        BoundedTensor::new(arr1(&[-1.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!(
        (output.lower()[[0]] - 1.0).abs() < 1e-6,
        "lower = {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 8.0).abs() < 1e-6,
        "upper = {}",
        output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_all_negative_weights() {
    // W = [[-1, -2]], b = [0]
    // x in [[0, 1], [0, 1]]
    // W+ = [[0, 0]], W- = [[-1, -2]]
    // lower_y = W+ @ [0, 0] + W- @ [1, 1] = [0] + [-1 + -2] = [-3]
    // upper_y = W+ @ [1, 1] + W- @ [0, 0] = [0] + [0] = [0]
    let weight = arr2(&[[-1.0, -2.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = linear.propagate_ibp(&input).unwrap();

    assert!((output.lower()[[0]] - (-3.0)).abs() < 1e-6);
    assert!((output.upper()[[0]] - 0.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_with_engine_threads_1d_linear_path_3954() {
    let linear = LinearLayer::new(
        arr2(&[[1.0_f32, -0.5, 2.0], [-1.0, 0.25, 0.5]]),
        Some(arr1(&[0.1_f32, -0.2])),
    )
    .unwrap();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5, -0.25]).into_dyn(),
        arr1(&[2.0_f32, 1.5, 0.75]).into_dyn(),
    )
    .unwrap();

    let baseline = linear.propagate_ibp(&input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = linear
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        "linear ibp with engine lower",
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        "linear ibp with engine upper",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3954 regression: LinearLayer::propagate_ibp_with_engine should hit GemmEngine"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_with_engine_uses_single_gemm_for_concrete_input_3954() {
    let linear = LinearLayer::new(
        arr2(&[[1.0_f32, -0.5, 2.0], [-1.0, 0.25, 0.5]]),
        Some(arr1(&[0.1_f32, -0.2])),
    )
    .unwrap();
    let input = BoundedTensor::concrete(arr1(&[2.0_f32, 1.5, 0.75]).into_dyn()).unwrap();

    let baseline = linear.propagate_ibp(&input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = linear
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        "linear ibp concrete with engine lower",
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        "linear ibp concrete with engine upper",
    );
    assert_eq!(
        engine.gemm_calls(),
        1,
        "#3954 regression: concrete linear IBP should use a single GEMM"
    );
}

// ============================================================
// IBP TESTS FOR RELU
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_relu_ibp_all_positive() {
    let input =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();

    let output = ReLULayer.propagate_ibp(&input).unwrap();

    // All positive: ReLU is identity
    assert!((output.lower()[[0]] - 1.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 3.0).abs() < 1e-6);
    assert!((output.lower()[[1]] - 2.0).abs() < 1e-6);
    assert!((output.upper()[[1]] - 4.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_ibp_all_negative() {
    let input = BoundedTensor::new(
        arr1(&[-4.0, -3.0]).into_dyn(),
        arr1(&[-2.0, -1.0]).into_dyn(),
    )
    .unwrap();

    let output = ReLULayer.propagate_ibp(&input).unwrap();

    // All negative: ReLU outputs zero
    assert!((output.lower()[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 0.0).abs() < 1e-6);
    assert!((output.lower()[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[1]] - 0.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_ibp_crossing_zero() {
    let input =
        BoundedTensor::new(arr1(&[-1.0, -2.0]).into_dyn(), arr1(&[2.0, 1.0]).into_dyn()).unwrap();

    let output = ReLULayer.propagate_ibp(&input).unwrap();

    // Crossing zero: lower = max(-1, 0) = 0, upper = max(2, 0) = 2
    assert!((output.lower()[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 2.0).abs() < 1e-6);
    assert!((output.lower()[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[1]] - 1.0).abs() < 1e-6);
}

// ============================================================
// NETWORK (MULTI-LAYER) IBP TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_network_linear_relu() {
    // Simple 2-layer network: Linear(2->2) -> ReLU
    // W = [[1, -1], [-1, 1]], b = [0, 0]
    // x in [[-1, 1], [-1, 1]]
    //
    // After Linear:
    // W+ = [[1, 0], [0, 1]], W- = [[0, -1], [-1, 0]]
    // lower = W+ @ [-1, -1] + W- @ [1, 1] = [-1, -1] + [-1, -1] = [-2, -2]
    // upper = W+ @ [1, 1] + W- @ [-1, -1] = [1, 1] + [1, 1] = [2, 2]
    //
    // After ReLU: [max(-2,0), max(-2,0)] to [max(2,0), max(2,0)] = [0, 0] to [2, 2]
    let weight = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = network.propagate_ibp(&input).unwrap();

    assert!((output.lower()[[0]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 2.0).abs() < 1e-6);
    assert!((output.lower()[[1]] - 0.0).abs() < 1e-6);
    assert!((output.upper()[[1]] - 2.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_network_two_linear_layers() {
    // Linear(2->2) -> ReLU -> Linear(2->1)
    // First linear: identity, second linear: sum
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let w2 = arr2(&[[1.0, 1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, 2.0]).into_dyn(), arr1(&[1.0, 3.0]).into_dyn()).unwrap();

    let output = network.propagate_ibp(&input).unwrap();

    // After identity: [-1, 2] to [1, 3]
    // After ReLU: [0, 2] to [1, 3]
    // After sum: [0+2, 1+3] = [2, 4]
    assert!((output.lower()[[0]] - 2.0).abs() < 1e-6);
    assert!((output.upper()[[0]] - 4.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_ibp_with_engine_threads_linear_layers_3954() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32, -1.0], [0.5, 2.0]]), None).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0_f32, 0.25], [-0.5, 1.5]]),
            Some(arr1(&[0.1, -0.3])),
        )
        .unwrap(),
    ));

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5]).into_dyn(),
        arr1(&[2.0_f32, 3.0]).into_dyn(),
    )
    .unwrap();

    let baseline = network.propagate_ibp(&input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = network
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        "network ibp with engine lower",
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        "network ibp with engine upper",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3954 regression: Network::propagate_ibp_with_engine should thread GemmEngine to Linear layers"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_soundness() {
    // Soundness test: verify that for any concrete input within bounds,
    // the concrete output is within computed bounds.
    //
    // W = [[2, -1], [1, 3]], b = [1, -2]
    // x in [[0, 2], [1, 3]]
    let weight = arr2(&[[2.0, -1.0], [1.0, 3.0]]);
    let bias = arr1(&[1.0, -2.0]);
    let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output_bounds = linear.propagate_ibp(&input).unwrap();

    // Test several concrete points within input bounds
    let test_points = [
        arr1(&[0.0, 1.0]), // lower corner
        arr1(&[2.0, 3.0]), // upper corner
        arr1(&[1.0, 2.0]), // center
        arr1(&[0.5, 1.5]), // random point 1
        arr1(&[1.5, 2.5]), // random point 2
    ];

    for x in &test_points {
        let y = weight.dot(x) + &bias;

        // Verify y[i] is within [lower[i], upper[i]] for all i
        for i in 0..y.len() {
            assert!(
                y[i] >= output_bounds.lower()[[i]] - 1e-6,
                "Soundness violation: y[{}] = {} < lower = {}",
                i,
                y[i],
                output_bounds.lower()[[i]]
            );
            assert!(
                y[i] <= output_bounds.upper()[[i]] + 1e-6,
                "Soundness violation: y[{}] = {} > upper = {}",
                i,
                y[i],
                output_bounds.upper()[[i]]
            );
        }
    }
}

// ============================================================
// DIRECTED ROUNDING / SOUND IBP TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_propagate_ibp_sound_widens_bounds() {
    // propagate_ibp_sound should produce bounds that are slightly wider
    // than propagate_ibp due to directed rounding
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let w2 = arr2(&[[1.0, 1.0]]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, 2.0]).into_dyn(), arr1(&[1.0, 3.0]).into_dyn()).unwrap();

    let normal_output = network.propagate_ibp(&input).unwrap();
    let sound_output = network.propagate_ibp_sound(&input).unwrap();

    // Sound bounds should be at least as wide (lower <= lower, upper >= upper)
    assert!(
        sound_output.lower()[[0]] <= normal_output.lower()[[0]],
        "Sound lower bound should be <= normal: {} <= {}",
        sound_output.lower()[[0]],
        normal_output.lower()[[0]]
    );
    assert!(
        sound_output.upper()[[0]] >= normal_output.upper()[[0]],
        "Sound upper bound should be >= normal: {} >= {}",
        sound_output.upper()[[0]],
        normal_output.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_propagate_ibp_sound_preserves_soundness() {
    // Sound propagation should still satisfy the soundness property:
    // for any x in input bounds, f(x) is in output bounds
    let weight = arr2(&[[2.0, -1.0], [1.0, 3.0]]);
    let bias = arr1(&[1.0, -2.0]);
    let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn()).unwrap();

    let output_bounds = network.propagate_ibp_sound(&input).unwrap();

    // Test concrete points
    let test_points = [arr1(&[0.0, 1.0]), arr1(&[2.0, 3.0]), arr1(&[1.0, 2.0])];

    for x in &test_points {
        // Compute concrete output: ReLU(Wx + b)
        let linear_out = weight.dot(x) + &bias;
        let relu_out = linear_out.mapv(|v| v.max(0.0));

        for i in 0..relu_out.len() {
            assert!(
                relu_out[i] >= output_bounds.lower()[[i]] - 1e-6,
                "Sound propagation violation: out[{}] = {} < lower = {}",
                i,
                relu_out[i],
                output_bounds.lower()[[i]]
            );
            assert!(
                relu_out[i] <= output_bounds.upper()[[i]] + 1e-6,
                "Sound propagation violation: out[{}] = {} > upper = {}",
                i,
                relu_out[i],
                output_bounds.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_layer_shape_validation() {
    // Bias shape mismatch should error
    let weight = arr2(&[[1.0, 2.0], [3.0, 4.0]]); // 2x2
    let bad_bias = arr1(&[1.0, 2.0, 3.0]); // wrong size

    let result = LinearLayer::new(weight, Some(bad_bias));
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_input_shape_validation() {
    // Input dimension mismatch should error
    let weight = arr2(&[[1.0, 2.0]]);
    let linear = LinearLayer::new(weight, None).unwrap();

    let input = BoundedTensor::new(
        arr1(&[0.0, 0.0, 0.0]).into_dyn(), // 3 elements, expected 2
        arr1(&[1.0, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();

    let result = linear.propagate_ibp(&input);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_linear_ibp_depth_14_preserves_large_finite_bounds_2549() {
    // #2549 regression: finite IBP bounds above 1e10 must not be narrowed.
    // Build a depth-14 linear network with gain 10 each layer:
    // output range should be [-1e14, 1e14] for input [-1, 1].
    let mut network = Network::new();
    for _ in 0..14 {
        let weight = arr2(&[[10.0_f32]]);
        let bias = arr1(&[0.0_f32]);
        network.add_layer(Layer::Linear(LinearLayer::new(weight, Some(bias)).unwrap()));
    }

    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let output = network.propagate_ibp(&input).unwrap();
    let expected = 10.0_f32.powi(14);
    let lo = output.lower()[[0]];
    let hi = output.upper()[[0]];

    assert!(
        lo.is_finite() && hi.is_finite(),
        "bounds must be finite: [{lo}, {hi}]"
    );
    assert!(lo <= -expected * 0.99, "lower unexpectedly narrowed: {lo}");
    assert!(hi >= expected * 0.99, "upper unexpectedly narrowed: {hi}");
}

/// Quantify accumulated rounding error for networks of increasing depth.
///
/// This test builds deterministic networks of varying depth (5, 10, 20, 50 layers)
/// with typical weight magnitudes, then compares the output width of `propagate_ibp`
/// vs `propagate_ibp_sound` to measure how much the n-ULP directed rounding widens
/// bounds. This is AC #3 of issue #1690.
///
/// Expected result: width difference is proportional to `depth * in_features` ULPs,
/// but negligible compared to the overall bound width (which grows exponentially
/// with depth due to relaxation errors).
#[ntest::timeout(60000)]
#[test]
fn test_quantify_rounding_error_vs_depth() {
    use ndarray::Array1;

    // Build a network of alternating Linear(dim->dim) + ReLU layers
    fn build_network(depth: usize, dim: usize) -> Network {
        use ndarray::Array2;
        let mut network = Network::new();
        for d in 0..depth {
            // Deterministic weight pattern: scaled rotation-like matrix
            let scale = 0.5_f32; // keep weights < 1 to avoid explosion
            let mut w = Array2::<f32>::zeros((dim, dim));
            for i in 0..dim {
                for j in 0..dim {
                    // Mix of positive and negative weights
                    let base = if (i + j + d) % 3 == 0 {
                        scale
                    } else if (i + j + d) % 3 == 1 {
                        -scale * 0.5
                    } else {
                        scale * 0.3
                    };
                    w[[i, j]] = base;
                }
            }
            let bias = Array1::<f32>::from_elem(dim, 0.1);
            network.add_layer(Layer::Linear(LinearLayer::new(w, Some(bias)).unwrap()));
            network.add_layer(Layer::ReLU(ReLULayer));
        }
        network
    }

    let dim = 64; // typical small network width
                  // Note: at scale=0.5 and dim=64, bounds grow ~32x per layer pair.
                  // Very deep settings can produce non-finite values from overflow,
                  // so we include depth 50 to validate non-finite handling paths.
    let depths = [3, 5, 10, 15, 50];
    let input = BoundedTensor::new(
        Array1::<f32>::from_elem(dim, -1.0).into_dyn(),
        Array1::<f32>::from_elem(dim, 1.0).into_dyn(),
    )
    .unwrap();

    for &depth in &depths {
        let network = build_network(depth, dim);

        let normal = network.propagate_ibp(&input).unwrap();
        let sound = network.propagate_ibp_sound(&input).unwrap();

        // Check for NaN/Inf in outputs
        let normal_has_nan = (0..normal.len())
            .any(|i| !normal.lower()[[i]].is_finite() || !normal.upper()[[i]].is_finite());
        let sound_has_nan = (0..sound.len())
            .any(|i| !sound.lower()[[i]].is_finite() || !sound.upper()[[i]].is_finite());

        if normal_has_nan || sound_has_nan {
            eprintln!(
                "depth={depth}: skipping width comparison (non-finite bounds: \
                 normal_nan={normal_has_nan}, sound_nan={sound_has_nan})"
            );
            continue;
        }

        // Verify element-wise: sound lower <= normal lower, sound upper >= normal upper
        let mut max_lower_diff: f32 = 0.0;
        let mut max_upper_diff: f32 = 0.0;
        for i in 0..normal.len() {
            let lower_diff = normal.lower()[[i]] - sound.lower()[[i]]; // should be >= 0
            let upper_diff = sound.upper()[[i]] - normal.upper()[[i]]; // should be >= 0
            assert!(
                lower_diff >= 0.0,
                "depth={depth}, i={i}: sound lower {} > normal lower {} (diff={lower_diff})",
                sound.lower()[[i]],
                normal.lower()[[i]]
            );
            assert!(
                upper_diff >= 0.0,
                "depth={depth}, i={i}: sound upper {} < normal upper {} (diff={upper_diff})",
                sound.upper()[[i]],
                normal.upper()[[i]]
            );
            max_lower_diff = max_lower_diff.max(lower_diff);
            max_upper_diff = max_upper_diff.max(upper_diff);
        }

        // Compute total width for each (only for non-saturated elements)
        let normal_width: f32 = (0..normal.len())
            .map(|i| normal.upper()[[i]] - normal.lower()[[i]])
            .sum();
        let sound_width: f32 = (0..sound.len())
            .map(|i| sound.upper()[[i]] - sound.lower()[[i]])
            .sum();
        let width_diff = sound_width - normal_width;

        let relative_diff = if normal_width.is_finite() && normal_width > 0.0 {
            width_diff / normal_width
        } else {
            0.0 // Saturated bounds — skip relative comparison
        };
        eprintln!(
            "Rounding quantification depth={depth} dim={dim}: \
             normal_width={normal_width:.6e} sound_width={sound_width:.6e} \
             diff={width_diff:.6e} relative={relative_diff:.6e} \
             max_lower_diff={max_lower_diff:.6e} max_upper_diff={max_upper_diff:.6e}"
        );

        // Key assertion: relative rounding overhead should be small (<1%)
        // For practical networks, rounding adds much less than relaxation error.
        // Skip this check when bounds have saturated (width is Inf/NaN).
        if normal_width.is_finite() && normal_width > 1e-10 {
            assert!(
                relative_diff < 0.01,
                "depth={depth}: rounding overhead {relative_diff:.6e} exceeds 1% threshold"
            );
        }
    }
}
