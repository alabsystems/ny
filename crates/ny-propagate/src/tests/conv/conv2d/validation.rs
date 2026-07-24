// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_shape_validation() {
    // Kernel must be 4D
    let bad_kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 2, 2])); // 3D, not 4D
    let result = Conv2dLayer::new(bad_kernel, None, (1, 1), (0, 0));
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_input_channels_validation() {
    // Input channels must match kernel
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 3, 2, 2])); // 3 input channels
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let input = BoundedTensor::new(
        ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3])), // 1 channel, expected 3
        ArrayD::ones(ndarray::IxDyn(&[1, 3, 3])),
    )
    .unwrap();

    let result = conv.propagate_ibp(&input);
    assert!(result.is_err());
}
