// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_transpose_basic() {
    // Test that conv2d_transpose is the inverse of conv2d in the gradient sense
    // Input: [1, 3, 3], Kernel: [1, 1, 2, 2], Output: [1, 2, 2]
    let mut kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = 2.0;
    kernel[[0, 0, 1, 0]] = 3.0;
    kernel[[0, 0, 1, 1]] = 4.0;

    // Forward conv input
    let mut input = ArrayD::zeros(ndarray::IxDyn(&[1, 3, 3]));
    for i in 0..3 {
        for j in 0..3 {
            input[[0, i, j]] = (i * 3 + j + 1) as f32;
        }
    }

    // Gradient at output (identity for this test)
    let mut grad_out = ArrayD::zeros(ndarray::IxDyn(&[1, 2, 2]));
    grad_out[[0, 0, 0]] = 1.0;
    grad_out[[0, 0, 1]] = 1.0;
    grad_out[[0, 1, 0]] = 1.0;
    grad_out[[0, 1, 1]] = 1.0;

    // Compute transposed conv
    let grad_in = conv2d_transpose(&grad_out, &kernel, (1, 1), (0, 0), (1, 1), (3, 3)).unwrap();

    // Expected: scatter of grad * kernel at each input position
    // Position (0,0) receives grad[0,0] * kernel[0,0] = 1 * 1 = 1
    // Position (0,1) receives grad[0,0] * kernel[0,1] + grad[0,1] * kernel[0,0] = 1*2 + 1*1 = 3
    // etc.
    assert_eq!(grad_in.shape(), &[1, 3, 3]);
    assert!((grad_in[[0, 0, 0]] - 1.0).abs() < 1e-6);
    assert!((grad_in[[0, 0, 1]] - 3.0).abs() < 1e-6);
    assert!((grad_in[[0, 0, 2]] - 2.0).abs() < 1e-6);
    assert!((grad_in[[0, 1, 0]] - 4.0).abs() < 1e-6);
    assert!((grad_in[[0, 1, 1]] - 10.0).abs() < 1e-6);
    assert!((grad_in[[0, 1, 2]] - 6.0).abs() < 1e-6);
    assert!((grad_in[[0, 2, 0]] - 3.0).abs() < 1e-6);
    assert!((grad_in[[0, 2, 1]] - 7.0).abs() < 1e-6);
    assert!((grad_in[[0, 2, 2]] - 4.0).abs() < 1e-6);
}
