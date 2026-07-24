// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_identity_kernel() {
    // 1x1 identity kernel: output equals input (center pixel only due to valid conv)
    // Actually, for 2x2 kernel on 3x3 input with stride=1, padding=0, output is 2x2
    // Let's use a simpler test: all-ones kernel with concrete input

    // Kernel: [[1, 1], [1, 1]] sums the 2x2 region
    let kernel = kernel_4d(&[[[[1.0, 1.0], [1.0, 1.0]]]]);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // Concrete input (lower == upper)
    let input_data = input_3d(&[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]);
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // Output shape: (1, 2, 2) - 2x2 output from 3x3 input with 2x2 kernel
    assert_eq!(output.shape(), &[1, 2, 2]);

    // For concrete input, lower == upper
    // output[0,0,0] = 1+2+4+5 = 12
    // output[0,0,1] = 2+3+5+6 = 16
    // output[0,1,0] = 4+5+7+8 = 24
    // output[0,1,1] = 5+6+8+9 = 28
    assert!((output.lower()[[0, 0, 0]] - 12.0).abs() < 1e-6);
    assert!((output.upper()[[0, 0, 0]] - 12.0).abs() < 1e-6);
    assert!((output.lower()[[0, 0, 1]] - 16.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1, 0]] - 24.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1, 1]] - 28.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_batched_input() {
    // Verify Conv2d IBP supports (batch, channels, height, width) inputs.
    let kernel = kernel_4d(&[[[[1.0, 1.0], [1.0, 1.0]]]]);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3, 3]));
    for h in 0..3 {
        for w in 0..3 {
            input_data[[0, 0, h, w]] = (h * 3 + w + 1) as f32; // 1..9
            input_data[[1, 0, h, w]] = (h * 3 + w + 2) as f32; // 2..10
        }
    }
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 1, 2, 2]);
    // Batch 0: same as test_conv2d_ibp_identity_kernel
    assert!((output.lower()[[0, 0, 0, 0]] - 12.0).abs() < 1e-6);
    assert!((output.lower()[[0, 0, 0, 1]] - 16.0).abs() < 1e-6);
    assert!((output.lower()[[0, 0, 1, 0]] - 24.0).abs() < 1e-6);
    assert!((output.lower()[[0, 0, 1, 1]] - 28.0).abs() < 1e-6);

    // Batch 1: each input entry is +1, so each 2x2 sum is +4.
    assert!((output.lower()[[1, 0, 0, 0]] - 16.0).abs() < 1e-6);
    assert!((output.lower()[[1, 0, 0, 1]] - 20.0).abs() < 1e-6);
    assert!((output.lower()[[1, 0, 1, 0]] - 28.0).abs() < 1e-6);
    assert!((output.lower()[[1, 0, 1, 1]] - 32.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_positive_kernel() {
    // All positive kernel with bounded input
    let kernel = kernel_4d(&[[[[1.0, 2.0], [3.0, 4.0]]]]);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // Input: all zeros to all ones
    let lower_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    let upper_data = ArrayD::ones(ndarray::IxDyn(&[1, 3, 3]));
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // For all-positive kernel and input in [0, 1]:
    // lower = conv([0,0,0,0], kernel) = 0
    // upper = conv([1,1,1,1], kernel) = 1+2+3+4 = 10
    assert_eq!(output.shape(), &[1, 2, 2]);
    for idx in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)] {
        assert!(
            (output.lower()[[idx.0, idx.1, idx.2]] - 0.0).abs() < 1e-6,
            "lower at {:?} = {}",
            idx,
            output.lower()[[idx.0, idx.1, idx.2]]
        );
        assert!(
            (output.upper()[[idx.0, idx.1, idx.2]] - 10.0).abs() < 1e-6,
            "upper at {:?} = {}",
            idx,
            output.upper()[[idx.0, idx.1, idx.2]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_mixed_kernel() {
    // Mixed positive/negative kernel
    // Kernel: [[1, -1], [-1, 1]] (edge detector style)
    let kernel = kernel_4d(&[[[[1.0, -1.0], [-1.0, 1.0]]]]);
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // Input in [0, 1]
    let lower_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    let upper_data = ArrayD::ones(ndarray::IxDyn(&[1, 3, 3]));
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // For mixed kernel, bounds should consider pos/neg contributions
    // For each position, max is sum of positive weights (2), min is sum of negative weights (-2)
    assert_eq!(output.shape(), &[1, 2, 2]);
    for idx in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)] {
        assert!(
            (output.lower()[[idx.0, idx.1, idx.2]] - (-2.0)).abs() < 1e-6,
            "lower at {:?} = {}",
            idx,
            output.lower()[[idx.0, idx.1, idx.2]]
        );
        assert!(
            (output.upper()[[idx.0, idx.1, idx.2]] - 2.0).abs() < 1e-6,
            "upper at {:?} = {}",
            idx,
            output.upper()[[idx.0, idx.1, idx.2]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_soundness() {
    // Soundness test: verify concrete outputs are within bounds
    let kernel = kernel_4d(&[[[[2.0, -1.0], [1.0, -2.0]]]]);
    let conv = Conv2dLayer::new(kernel.clone(), None, (1, 1), (0, 0)).unwrap();

    // Input in [-1, 1]
    let lower_data = ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), -1.0f32);
    let upper_data = ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), 1.0f32);
    let input = BoundedTensor::new(lower_data, upper_data).unwrap();

    let output_bounds = conv.propagate_ibp(&input).unwrap();

    // Test several concrete inputs
    let test_inputs = [
        ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), -1.0f32), // all lower
        ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), 1.0f32),  // all upper
        ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), 0.0f32),  // center
    ];

    for test_input in &test_inputs {
        let concrete_output = conv2d_single(test_input, &kernel, (1, 1), (0, 0), (1, 1)).unwrap();

        for oc in 0..1 {
            for oh in 0..2 {
                for ow in 0..2 {
                    let val = concrete_output[[oc, oh, ow]];
                    assert!(
                        val >= output_bounds.lower()[[oc, oh, ow]] - 1e-6,
                        "Soundness: val {} < lower {}",
                        val,
                        output_bounds.lower()[[oc, oh, ow]]
                    );
                    assert!(
                        val <= output_bounds.upper()[[oc, oh, ow]] + 1e-6,
                        "Soundness: val {} > upper {}",
                        val,
                        output_bounds.upper()[[oc, oh, ow]]
                    );
                }
            }
        }
    }
}
