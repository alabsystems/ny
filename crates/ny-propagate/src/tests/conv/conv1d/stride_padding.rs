// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_with_stride() {
    // Conv1d with stride=2
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    let conv = Conv1dLayer::new(kernel, None, 2, 0).unwrap();

    // Input: [1, 2, 3, 4] with stride 2 -> output length = (4-2)/2 + 1 = 2
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 4]));
    for i in 0..4 {
        input_data[[0, i]] = (i + 1) as f32;
    }
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // Output: [1+2, 3+4] = [3, 7]
    assert_eq!(output.shape(), &[1, 2]);
    assert!((output.lower()[[0, 0]] - 3.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1]] - 7.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_with_padding() {
    // Conv1d with padding=1
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 3]));
    let conv = Conv1dLayer::new(kernel, None, 1, 1).unwrap();

    // Input: [1, 1, 1] with padding 1 -> output length = (3+2-3)/1 + 1 = 3
    let input_data = ArrayD::ones(ndarray::IxDyn(&[1, 3]));
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // With padding: [0,1,1], [1,1,1], [1,1,0] -> sums: 2, 3, 2
    assert_eq!(output.shape(), &[1, 3]);
    assert!(
        (output.lower()[[0, 0]] - 2.0).abs() < 1e-6,
        "left edge = {}",
        output.lower()[[0, 0]]
    );
    assert!(
        (output.lower()[[0, 1]] - 3.0).abs() < 1e-6,
        "center = {}",
        output.lower()[[0, 1]]
    );
    assert!(
        (output.lower()[[0, 2]] - 2.0).abs() < 1e-6,
        "right edge = {}",
        output.lower()[[0, 2]]
    );
}
