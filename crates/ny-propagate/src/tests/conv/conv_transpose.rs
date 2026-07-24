// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{arr1, ArrayD};

// =============================================================================
// ConvTranspose IBP/CROWN Tests
// =============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_conv_transpose1d_ibp_basic() {
    // ConvTranspose1d with kernel [1, 2] on input [1, 2, 3].
    // Output length = (3 - 1) * 1 + 2 - 0 = 4.
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = 2.0;
    let conv = ConvTranspose1dLayer::new(kernel, None, 1, 0).unwrap();

    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 3]));
    input_data[[0, 0]] = 1.0;
    input_data[[0, 1]] = 2.0;
    input_data[[0, 2]] = 3.0;
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();
    assert_eq!(output.shape(), &[1, 4]);

    let expected = [1.0, 4.0, 7.0, 6.0];
    for (idx, &value) in expected.iter().enumerate() {
        assert!(
            (output.lower()[[0, idx]] - value).abs() < 1e-6,
            "lower[{}] = {}",
            idx,
            output.lower()[[0, idx]]
        );
        assert!(
            (output.upper()[[0, idx]] - value).abs() < 1e-6,
            "upper[{}] = {}",
            idx,
            output.upper()[[0, idx]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv_transpose2d_ibp_basic() {
    // ConvTranspose2d with 2x2 kernel of ones on 2x2 input.
    // Output shape = (2 - 1) * 1 + 2 - 0 = 3 in each dim.
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = 1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = 1.0;
    let conv = ConvTranspose2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let mut input_data = ArrayD::zeros(ndarray::IxDyn(&[1, 2, 2]));
    input_data[[0, 0, 0]] = 1.0;
    input_data[[0, 0, 1]] = 2.0;
    input_data[[0, 1, 0]] = 3.0;
    input_data[[0, 1, 1]] = 4.0;
    let input = BoundedTensor::concrete(input_data).unwrap();

    let output = conv.propagate_ibp(&input).unwrap();
    assert_eq!(output.shape(), &[1, 3, 3]);

    let expected = [[1.0, 3.0, 2.0], [4.0, 10.0, 6.0], [3.0, 7.0, 4.0]];
    for (h, expected_row) in expected.iter().enumerate() {
        for (w, &value) in expected_row.iter().enumerate() {
            assert!(
                (output.lower()[[0, h, w]] - value).abs() < 1e-6,
                "lower[{},{}] = {}",
                h,
                w,
                output.lower()[[0, h, w]]
            );
            assert!(
                (output.upper()[[0, h, w]] - value).abs() < 1e-6,
                "upper[{},{}] = {}",
                h,
                w,
                output.upper()[[0, h, w]]
            );
        }
    }
}

// ============================================================
// CONVTRANSPOSE CROWN TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_conv_transpose1d_crown_vs_ibp() {
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;

    let bias = arr1(&[0.25]);

    let in_len = 4;
    let conv_crown =
        ConvTranspose1dLayer::with_input_length(kernel.clone(), Some(bias.clone()), 1, 0, in_len)
            .unwrap();
    let conv_ibp = ConvTranspose1dLayer::new(kernel, Some(bias), 1, 0).unwrap();

    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, in_len]), 0.1);
    let input = BoundedTensor::from_epsilon(center, 0.2).unwrap();

    let ibp_output = conv_ibp.propagate_ibp(&input).unwrap();

    let out_len = conv_crown.output_length(in_len).unwrap();
    let output_size = conv_crown.out_channels() * out_len;
    let identity = LinearBounds::identity(output_size);
    let crown_bounds = conv_crown.propagate_linear(&identity).unwrap().into_owned();
    let crown_output = crown_bounds
        .concretize(&input)
        .reshape(&[conv_crown.out_channels(), out_len])
        .unwrap();

    for (crown, ibp) in crown_output.lower().iter().zip(ibp_output.lower().iter()) {
        assert!(
            (crown - ibp).abs() < 1e-4,
            "ConvTranspose1d lower mismatch: {} vs {}",
            crown,
            ibp
        );
    }
    for (crown, ibp) in crown_output.upper().iter().zip(ibp_output.upper().iter()) {
        assert!(
            (crown - ibp).abs() < 1e-4,
            "ConvTranspose1d upper mismatch: {} vs {}",
            crown,
            ibp
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv_transpose2d_crown_vs_ibp() {
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[1, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = -0.5;

    let bias = arr1(&[0.1]);

    let in_h = 2;
    let in_w = 2;
    let conv_crown = ConvTranspose2dLayer::with_input_shape(
        kernel.clone(),
        Some(bias.clone()),
        (1, 1),
        (0, 0),
        in_h,
        in_w,
    )
    .unwrap();
    let conv_ibp = ConvTranspose2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).unwrap();

    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, in_h, in_w]), 0.2);
    let input = BoundedTensor::from_epsilon(center, 0.15).unwrap();

    let ibp_output = conv_ibp.propagate_ibp(&input).unwrap();

    let (out_h, out_w) = conv_crown.output_size(in_h, in_w).unwrap();
    let output_size = conv_crown.out_channels() * out_h * out_w;
    let identity = LinearBounds::identity(output_size);
    let crown_bounds = conv_crown.propagate_linear(&identity).unwrap().into_owned();
    let crown_output = crown_bounds
        .concretize(&input)
        .reshape(&[conv_crown.out_channels(), out_h, out_w])
        .unwrap();

    for (crown, ibp) in crown_output.lower().iter().zip(ibp_output.lower().iter()) {
        assert!(
            (crown - ibp).abs() < 1e-4,
            "ConvTranspose2d lower mismatch: {} vs {}",
            crown,
            ibp
        );
    }
    for (crown, ibp) in crown_output.upper().iter().zip(ibp_output.upper().iter()) {
        assert!(
            (crown - ibp).abs() < 1e-4,
            "ConvTranspose2d upper mismatch: {} vs {}",
            crown,
            ibp
        );
    }
}
