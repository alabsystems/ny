// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_basic() {
    // Simple 1D convolution: sum of 3 adjacent elements
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    kernel[[0, 0, 2]] = 1.0;
    let conv = Conv1dLayer::new(kernel, None, 1, 0).unwrap();

    // Input: [1, 2, 3, 4, 5]
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 5]));
    for i in 0..5 {
        input_data[[0, i]] = (i + 1) as f32;
    }
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // Output: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    assert_eq!(output.shape(), &[1, 3]);
    assert!((output.lower()[[0, 0]] - 6.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1]] - 9.0).abs() < 1e-6);
    assert!((output.lower()[[0, 2]] - 12.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_with_bias() {
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 2]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    kernel[[1, 0, 0]] = 2.0;
    kernel[[1, 0, 1]] = -1.0;
    let bias = arr1(&[0.5, -0.5]);
    let conv = Conv1dLayer::new(kernel, Some(bias), 1, 0).unwrap();

    // Input: [1, 2]
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 2]));
    input_data[[0, 0]] = 1.0;
    input_data[[0, 1]] = 2.0;
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // Channel 0: 1+2 + 0.5 = 3.5
    // Channel 1: 2*1 + (-1)*2 - 0.5 = 2 - 2 - 0.5 = -0.5
    assert_eq!(output.shape(), &[2, 1]);
    assert!((output.lower()[[0, 0]] - 3.5).abs() < 1e-6);
    assert!((output.lower()[[1, 0]] - (-0.5)).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_ibp_soundness() {
    // Soundness test: verify concrete outputs are within bounds
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 3]));
    kernel[[0, 0, 0]] = 2.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    let conv = Conv1dLayer::new(kernel.clone(), None, 1, 0).unwrap();

    // Input in [-1, 1]
    let lower_data = ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), -1.0f32);
    let upper_data = ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), 1.0f32);
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output_bounds = conv.propagate_ibp(&input).unwrap();

    // Test several concrete inputs
    let test_inputs = [
        ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), -1.0f32), // all lower
        ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), 1.0f32),  // all upper
        ArrayD::from_elem(ndarray::IxDyn(&[1, 5]), 0.0f32),  // center
    ];

    for test_input in &test_inputs {
        let concrete_output = conv1d_single(test_input, &kernel, 1, 0, 1, 1).unwrap();

        for oc in 0..1 {
            for ol in 0..3 {
                let val = concrete_output[[oc, ol]];
                assert!(
                    val >= output_bounds.lower()[[oc, ol]] - 1e-6,
                    "Soundness: val {} < lower {}",
                    val,
                    output_bounds.lower()[[oc, ol]]
                );
                assert!(
                    val <= output_bounds.upper()[[oc, ol]] + 1e-6,
                    "Soundness: val {} > upper {}",
                    val,
                    output_bounds.upper()[[oc, ol]]
                );
            }
        }
    }
}
