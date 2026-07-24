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

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_identity_bounds() {
    // Test CROWN with identity linear bounds (output = conv_output)
    let mut kernel = ArrayD::ones(ndarray::IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = 0.0;
    kernel[[0, 0, 1, 0]] = 0.0;
    kernel[[0, 0, 1, 1]] = 0.0;
    kernel[[1, 0, 0, 0]] = 0.0;
    kernel[[1, 0, 0, 1]] = 1.0;
    kernel[[1, 0, 1, 0]] = 0.0;
    kernel[[1, 0, 1, 1]] = 0.0;

    // With input [1, 3, 3] and kernel [2, 1, 2, 2], output is [2, 2, 2] = 8 elements
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 3, 3).unwrap();

    // Identity bounds: A = I, b = 0
    let identity_bounds = LinearBounds::identity(8);

    // Propagate backward through conv
    let result = conv
        .propagate_linear(&identity_bounds)
        .unwrap()
        .into_owned();

    // Should have shape [8, 9] (8 outputs, 9 = 1*3*3 inputs)
    assert_eq!(result.lower_a.shape(), &[8, 9]);
    assert_eq!(result.upper_a.shape(), &[8, 9]);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_simple_backward() {
    // Simple test: 1x1 conv (essentially a linear layer)
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 1, 1]));
    kernel[[0, 0, 0, 0]] = 2.0; // Output channel 0: 2x input
    kernel[[1, 0, 0, 0]] = 3.0; // Output channel 1: 3x input

    // Input [1, 2, 2] -> Output [2, 2, 2]
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2).unwrap();

    // Identity bounds on output
    let identity_bounds = LinearBounds::identity(8); // 2 * 2 * 2 = 8

    let result = conv
        .propagate_linear(&identity_bounds)
        .unwrap()
        .into_owned();

    // For 1x1 conv, backward pass through a single output position should give:
    // A @ W where W is the 1x1 kernel value
    // Output channel 0 position (0,0) -> A[0, :] should have kernel[0,0,0,0]=2.0 at input position 0
    assert!((result.lower_a[[0, 0]] - 2.0).abs() < 1e-6);
    assert!((result.lower_a[[0, 1]] - 0.0).abs() < 1e-6);

    // Output channel 1 position (0,0) -> A[4, :] should have kernel[1,0,0,0]=3.0 at input position 0
    assert!((result.lower_a[[4, 0]] - 3.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_with_bias() {
    // Test CROWN bias handling
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 2, 2]));
    let bias = Array1::from_vec(vec![0.5]);

    // Input [1, 3, 3] -> Output [1, 2, 2]
    let conv = Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 3, 3).unwrap();

    // Identity bounds on output
    let identity_bounds = LinearBounds::identity(4); // 1 * 2 * 2 = 4

    let result = conv
        .propagate_linear(&identity_bounds)
        .unwrap()
        .into_owned();

    // Bias contribution: each output position contributes 0.5 to its bound
    // Identity bounds sum over one position each, so bias contrib is 0.5
    assert!((result.lower_b[0] - 0.5).abs() < 1e-6);
    assert!((result.lower_b[1] - 0.5).abs() < 1e-6);
    assert!((result.lower_b[2] - 0.5).abs() < 1e-6);
    assert!((result.lower_b[3] - 0.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_vs_ibp_tightness() {
    // Compare CROWN vs IBP - CROWN should be at least as tight
    // Use a small network: Conv2d -> (flatten for bounds computation)
    let kernel = ArrayD::from_elem(ndarray::IxDyn(&[2, 1, 2, 2]), 0.5);
    let bias = Array1::from_vec(vec![0.1, -0.1]);

    let conv =
        Conv2dLayer::with_input_shape(kernel.clone(), Some(bias.clone()), (1, 1), (0, 0), 3, 3)
            .unwrap();
    let conv_ibp = Conv2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).unwrap();

    // Input with perturbation
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 3, 3]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();

    // IBP bounds
    let ibp_output = conv_ibp.propagate_ibp(&input).unwrap();
    let ibp_flat = ibp_output.flatten();

    // CROWN bounds
    let identity = LinearBounds::identity(8); // 2 * 2 * 2 = 8
    let crown_bounds = conv.propagate_linear(&identity).unwrap().into_owned();

    // Concretize CROWN bounds with input bounds
    let flat_input = input.flatten();
    let x_l = flat_input
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();
    let x_u = flat_input
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .unwrap();

    // CROWN lower: A+ @ x_l + A- @ x_u + b
    // CROWN upper: A+ @ x_u + A- @ x_l + b
    let a_pos_l = crown_bounds.lower_a.mapv(|v| v.max(0.0));
    let a_neg_l = crown_bounds.lower_a.mapv(|v| v.min(0.0));
    let a_pos_u = crown_bounds.upper_a.mapv(|v| v.max(0.0));
    let a_neg_u = crown_bounds.upper_a.mapv(|v| v.min(0.0));

    let crown_lower = a_pos_l.dot(&x_l) + a_neg_l.dot(&x_u) + &crown_bounds.lower_b;
    let crown_upper = a_pos_u.dot(&x_u) + a_neg_u.dot(&x_l) + &crown_bounds.upper_b;

    // For linear layers (conv is linear), CROWN should match IBP exactly
    // since there's no non-linearity to relax
    for i in 0..8 {
        let ibp_l = ibp_flat.lower().as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper().as_slice().unwrap()[i];
        let crown_l = crown_lower[i];
        let crown_u = crown_upper[i];

        // CROWN should be at least as tight as IBP (or same for linear)
        assert!(
            crown_l >= ibp_l - 1e-5,
            "CROWN lower {} should be >= IBP lower {}",
            crown_l,
            ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-5,
            "CROWN upper {} should be <= IBP upper {}",
            crown_u,
            ibp_u
        );

        // For pure linear, should be nearly identical
        assert!(
            (crown_l - ibp_l).abs() < 1e-4,
            "CROWN ({}) and IBP ({}) should match for linear",
            crown_l,
            ibp_l
        );
        assert!(
            (crown_u - ibp_u).abs() < 1e-4,
            "CROWN ({}) and IBP ({}) should match for linear",
            crown_u,
            ibp_u
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_requires_input_shape() {
    // Test that CROWN fails without input_shape set
    let kernel = ArrayD::ones(ndarray::IxDyn(&[1, 1, 2, 2]));
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let identity_bounds = LinearBounds::identity(4);
    let result = conv.propagate_linear(&identity_bounds);

    assert!(result.is_err());
}
