// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_shape_validation() {
    // Kernel must be 3D
    let bad_kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 2])); // 2D, not 3D
    let result = Conv1dLayer::new(bad_kernel, None, 1, 0);
    assert!(result.is_err());
}
