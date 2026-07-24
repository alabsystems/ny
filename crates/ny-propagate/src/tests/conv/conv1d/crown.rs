// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_identity_bounds() {
    // Test shape handling: verify dimensions work correctly
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3])); // out_c=2, in_c=1, k=3
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.0;
    kernel[[1, 0, 2]] = 0.5;

    // Input: [1, 8] -> Output: [2, 6] = 12 elements
    let conv = Conv1dLayer::with_input_length(kernel, None, 1, 0, 8).unwrap();

    // Identity bounds on output
    let output_size = 12;
    let identity = LinearBounds::identity(output_size);

    let new_bounds = conv
        .propagate_linear(&identity)
        .expect("Conv1d CROWN backward failed");

    // New bounds should be on flattened input: 1 * 8 = 8
    assert_eq!(new_bounds.num_inputs(), 8);
    assert_eq!(new_bounds.num_outputs(), 12);

    // Kernel[0] = [1, 1, 1] sums 3 consecutive inputs per output position.
    // With identity downstream, the first output row should have exactly 3 non-zero
    // entries (1.0) in the coefficient matrix at the corresponding input positions.
    let row0 = new_bounds.lower_a.row(0);
    let nnz: usize = row0.iter().filter(|&&v| v != 0.0).count();
    assert_eq!(
        nnz, 3,
        "First output should depend on exactly 3 inputs, got {nnz}"
    );
    assert!(
        (row0.iter().filter(|&&v| v != 0.0).copied().sum::<f32>() - 3.0).abs() < 1e-6,
        "Sum of non-zero entries in row 0 should be 3.0 (three 1.0 kernel weights)"
    );

    // Biases should be zero (no bias in the conv layer).
    assert!(
        new_bounds.lower_b.iter().all(|&v| v == 0.0),
        "Lower biases should be zero for unbiased conv"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_simple_backward() {
    // Test CROWN backward pass through a simple 1x1 conv
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 1]));
    kernel[[0, 0, 0]] = 2.0; // Just scales by 2

    let conv = Conv1dLayer::with_input_length(kernel, None, 1, 0, 4).unwrap();

    // Identity bounds on output [1, 4] = 4 elements
    let identity = LinearBounds::identity(4);

    let result = conv.propagate_linear(&identity).unwrap().into_owned();

    // Backward through 2x scaling should give A with 2.0 entries
    for i in 0..4 {
        assert!(
            (result.lower_a[[i, i]] - 2.0).abs() < 1e-6,
            "Expected 2.0 at [{}, {}], got {}",
            i,
            i,
            result.lower_a[[i, i]]
        );
        assert!((result.upper_a[[i, i]] - 2.0).abs() < 1e-6);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_with_bias() {
    // Test that bias is handled correctly in CROWN
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 1]));
    kernel[[0, 0, 0]] = 1.0;

    let bias = arr1(&[3.0]);

    let conv = Conv1dLayer::with_input_length(kernel, Some(bias), 1, 0, 4).unwrap();

    // Identity bounds on output
    let identity = LinearBounds::identity(4);

    let result = conv.propagate_linear(&identity).unwrap().into_owned();

    // Bias contribution: each output gets +3.0
    // For identity A matrix, bias_contrib[i] = sum over spatial * bias = 1 * 3.0 = 3.0
    for i in 0..4 {
        assert!(
            (result.lower_b[i] - 3.0).abs() < 1e-6,
            "Expected bias 3.0 at [{}], got {}",
            i,
            result.lower_b[i]
        );
        assert!((result.upper_b[i] - 3.0).abs() < 1e-6);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_vs_ibp_tightness() {
    // For pure Conv1d (linear operation), CROWN should match IBP exactly
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.5;
    kernel[[1, 0, 2]] = 0.5;

    let bias = arr1(&[1.0, -0.5]);

    let in_len = 6;
    let conv_crown =
        Conv1dLayer::with_input_length(kernel.clone(), Some(bias.clone()), 1, 0, in_len).unwrap();
    let conv_ibp = Conv1dLayer::new(kernel, Some(bias), 1, 0).unwrap();

    // Input bounds
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, in_len]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();

    // IBP bounds
    let ibp_output = conv_ibp.propagate_ibp(&input).unwrap();

    // CROWN bounds
    let out_len = conv_crown.output_length(in_len).unwrap();
    let output_size = 2 * out_len; // out_c=2
    let identity = LinearBounds::identity(output_size);

    let crown_bounds = conv_crown.propagate_linear(&identity).unwrap().into_owned();

    // Concretize CROWN bounds
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

    let a_pos_l = crown_bounds.lower_a.mapv(|v| v.max(0.0));
    let a_neg_l = crown_bounds.lower_a.mapv(|v| v.min(0.0));
    let a_pos_u = crown_bounds.upper_a.mapv(|v| v.max(0.0));
    let a_neg_u = crown_bounds.upper_a.mapv(|v| v.min(0.0));

    let crown_lower = a_pos_l.dot(&x_l) + a_neg_l.dot(&x_u) + &crown_bounds.lower_b;
    let crown_upper = a_pos_u.dot(&x_u) + a_neg_u.dot(&x_l) + &crown_bounds.upper_b;

    // Compare with IBP
    let ibp_flat = ibp_output.flatten();
    for i in 0..output_size {
        let ibp_l = ibp_flat.lower().as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper().as_slice().unwrap()[i];

        // For linear layers, CROWN and IBP should match
        assert!(
            (crown_lower[i] - ibp_l).abs() < 1e-4,
            "Output {}: CROWN lower {} vs IBP lower {}",
            i,
            crown_lower[i],
            ibp_l
        );
        assert!(
            (crown_upper[i] - ibp_u).abs() < 1e-4,
            "Output {}: CROWN upper {} vs IBP upper {}",
            i,
            crown_upper[i],
            ibp_u
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_requires_input_length() {
    // CROWN should fail if input_length is not set
    let kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 3]));
    let conv = Conv1dLayer::new(kernel, None, 1, 0).unwrap();

    let identity = LinearBounds::identity(4);
    let result = conv.propagate_linear(&identity);
    assert!(result.is_err());
}
