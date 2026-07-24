// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use approx::assert_relative_eq;
use ndarray::arr1;
use ny_tensor::BoundedTensor;

#[ntest::timeout(10000)]
#[test]
fn test_convert_to_propagate_network() {
    let path = require_test_model("single_linear.onnx");

    let model = load_onnx(&path).expect("Failed to load model");
    let network = model.to_propagate_network().expect("Failed to convert");

    assert_eq!(network.num_layers(), 1);

    // Test IBP propagation
    let input =
        BoundedTensor::new(arr1(&[1.0, 1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = network.propagate_ibp(&input).unwrap();

    // Expected: y = x @ W.T + b
    // x = [1, 1]
    // W = [[1, 2], [3, -1], [-2, 1]]
    // W @ x = [1*1 + 2*1, 3*1 + (-1)*1, (-2)*1 + 1*1] = [3, 2, -1]
    // + bias [0.5, -0.5, 1.0] = [3.5, 1.5, 0.0]
    assert_relative_eq!(output.lower()[[0]], 3.5, epsilon = 1e-5);
    assert_relative_eq!(output.lower()[[1]], 1.5, epsilon = 1e-5);
    assert_relative_eq!(output.lower()[[2]], 0.0, epsilon = 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_bounded_input() {
    let path = require_test_model("single_linear.onnx");

    let model = load_onnx(&path).expect("Failed to load model");
    let network = model.to_propagate_network().expect("Failed to convert");

    // Test with bounded input (interval)
    let input = BoundedTensor::new(
        arr1(&[0.0, 0.0]).into_dyn(), // lower bound
        arr1(&[1.0, 1.0]).into_dyn(), // upper bound
    )
    .unwrap();

    let output = network.propagate_ibp(&input).unwrap();

    // For W = [[1, 2], [3, -1], [-2, 1]], b = [0.5, -0.5, 1.0]
    // W+ = [[1, 2], [3, 0], [0, 1]], W- = [[0, 0], [0, -1], [-2, 0]]
    // lower = W+ @ [0,0] + W- @ [1,1] + b = [0, -1, -2] + [0.5, -0.5, 1.0] = [0.5, -1.5, -1.0]
    // upper = W+ @ [1,1] + W- @ [0,0] + b = [3, 3, 1] + [0.5, -0.5, 1.0] = [3.5, 2.5, 2.0]
    assert_relative_eq!(output.lower()[[0]], 0.5, epsilon = 1e-5);
    assert_relative_eq!(output.lower()[[1]], -1.5, epsilon = 1e-5);
    assert_relative_eq!(output.lower()[[2]], -1.0, epsilon = 1e-5);
    assert_relative_eq!(output.upper()[[0]], 3.5, epsilon = 1e-5);
    assert_relative_eq!(output.upper()[[1]], 2.5, epsilon = 1e-5);
    assert_relative_eq!(output.upper()[[2]], 2.0, epsilon = 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_softmax_ibp_soundness() {
    let path = require_test_model_with_hint("softmax.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load model");
    let network = model.to_propagate_network().expect("Failed to convert");

    // Test soundness: sample points in input interval, verify outputs are within bounds
    let input = BoundedTensor::new(
        arr1(&[-1.0, 0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[0.0, 1.0, 2.0, 3.0]).into_dyn(),
    )
    .unwrap();

    let bounds = network.propagate_ibp(&input).expect("IBP failed");

    // Sample corners and midpoint
    let test_points = vec![
        arr1(&[-1.0, 0.0, 1.0, 2.0]), // lower corner
        arr1(&[0.0, 1.0, 2.0, 3.0]),  // upper corner
        arr1(&[-0.5, 0.5, 1.5, 2.5]), // midpoint
    ];

    for point in test_points {
        // Compute actual softmax
        let exp_vals: Vec<f32> = point.iter().map(|&x: &f32| x.exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let softmax: Vec<f32> = exp_vals.iter().map(|&e| e / sum).collect();

        // Verify each output is within bounds
        for (i, &s) in softmax.iter().enumerate() {
            assert!(
                s >= bounds.lower()[[i]] - 1e-5,
                "Softmax output {} = {} below lower bound {}",
                i,
                s,
                bounds.lower()[[i]]
            );
            assert!(
                s <= bounds.upper()[[i]] + 1e-5,
                "Softmax output {} = {} above upper bound {}",
                i,
                s,
                bounds.upper()[[i]]
            );
        }
    }
}
