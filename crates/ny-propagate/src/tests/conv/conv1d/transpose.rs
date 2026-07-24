// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_transpose_basic() {
    // Test that conv1d_transpose is the inverse of conv1d in the gradient sense
    // For a 1x1 conv with identity kernel, transpose should also be identity
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 1]));
    kernel[[0, 0, 0]] = 1.0;

    // "Gradient" at output
    let mut grad_out = ArrayD::zeros(ndarray::IxDyn(&[1, 5]));
    grad_out[[0, 0]] = 1.0;
    grad_out[[0, 2]] = 3.0;
    grad_out[[0, 4]] = 5.0;

    let grad_in = conv1d_transpose(&grad_out, &kernel, 1, 0, 1, 1, 5).unwrap();

    // With identity 1x1 kernel and no stride/padding, should match
    assert_eq!(grad_in.shape(), &[1, 5]);
    assert!((grad_in[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((grad_in[[0, 2]] - 3.0).abs() < 1e-6);
    assert!((grad_in[[0, 4]] - 5.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_transpose_multi_channel() {
    // Test transpose with multiple input/output channels
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 3, 2])); // out_c=2, in_c=3, k=2
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 0.5;
    kernel[[1, 1, 0]] = -1.0;
    kernel[[1, 2, 1]] = 2.0;

    // Gradient at conv output: [2, 3] (out_c=2, out_len=3)
    let mut grad_out = ArrayD::zeros(ndarray::IxDyn(&[2, 3]));
    grad_out[[0, 0]] = 1.0;
    grad_out[[1, 1]] = 1.0;

    let grad_in = conv1d_transpose(&grad_out, &kernel, 1, 0, 1, 1, 4).unwrap();

    // Output should be [3, 4] (in_c=3, in_len=4)
    assert_eq!(grad_in.shape(), &[3, 4]);

    // Channel 0: kernel[0,0,:] = [1.0, 0.5], grad[0,:] = [1, 0, 0]
    // Transpose scatters: pos 0 gets 1.0*1.0=1.0, pos 1 gets 1.0*0.5=0.5
    assert!((grad_in[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((grad_in[[0, 1]] - 0.5).abs() < 1e-6);

    // Channel 1: kernel[1,1,:] = [-1.0, 0.0], grad[1,:] = [0, 1, 0]
    // Transpose scatters: pos 1 gets -1.0*1.0=-1.0
    assert!((grad_in[[1, 1]] - (-1.0)).abs() < 1e-6);

    // Channel 2: kernel[1,2,:] = [0.0, 2.0], grad[1,:] = [0, 1, 0]
    // Transpose scatters: pos 2 gets 2.0*1.0=2.0
    assert!((grad_in[[2, 2]] - 2.0).abs() < 1e-6);
}
