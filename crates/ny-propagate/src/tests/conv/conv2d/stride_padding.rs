// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_with_stride() {
    // 2x2 kernel with stride 2 on 4x4 input -> 2x2 output
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 2, 2]));
    let conv = Conv2dLayer::new(kernel, None, (2, 2), (0, 0)).unwrap();

    // Input: 4x4 matrix with values 1..16
    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 4, 4]));
    for i in 0..4 {
        for j in 0..4 {
            input_data[[0, i, j]] = (i * 4 + j + 1) as f32;
        }
    }
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // Output shape: 2x2
    assert_eq!(output.shape(), &[1, 2, 2]);

    // Each output is sum of 2x2 blocks:
    // top-left: 1+2+5+6 = 14
    // top-right: 3+4+7+8 = 22
    // bottom-left: 9+10+13+14 = 46
    // bottom-right: 11+12+15+16 = 54
    assert!((output.lower()[[0, 0, 0]] - 14.0).abs() < 1e-6);
    assert!((output.lower()[[0, 0, 1]] - 22.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1, 0]] - 46.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1, 1]] - 54.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_with_padding() {
    // 3x3 kernel with padding 1 on 3x3 input -> 3x3 output
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 3, 3]));
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (1, 1)).unwrap();

    // 3x3 input of all ones
    let input_data = ArrayD::ones(ndarray::IxDyn(&[1, 3, 3]));
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();

    // With padding=1, output is 3x3
    // Corner: sees 4 ones (2x2 valid region) = 4
    // Edge: sees 6 ones (2x3 valid region) = 6
    // Center: sees 9 ones (3x3 valid region) = 9
    assert_eq!(output.shape(), &[1, 3, 3]);
    assert!(
        (output.lower()[[0, 0, 0]] - 4.0).abs() < 1e-6,
        "corner = {}",
        output.lower()[[0, 0, 0]]
    );
    assert!(
        (output.lower()[[0, 0, 1]] - 6.0).abs() < 1e-6,
        "edge = {}",
        output.lower()[[0, 0, 1]]
    );
    assert!(
        (output.lower()[[0, 1, 1]] - 9.0).abs() < 1e-6,
        "center = {}",
        output.lower()[[0, 1, 1]]
    );
}
