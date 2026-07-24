// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_network_integration() {
    // Test CROWN through a Conv1d -> ReLU network
    // This verifies the full backward pass works with non-linearities

    // Input: [1, 8] -> Conv [2, 1, 3] -> [2, 6] -> ReLU -> [2, 6]
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.5;
    kernel[[1, 0, 2]] = 0.5;

    let conv = Conv1dLayer::with_input_length(kernel.clone(), None, 1, 0, 8).unwrap();

    // Input with perturbation around 0.5
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 8]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();

    // Get IBP bounds for pre-activation (conv output)
    let conv_ibp = Conv1dLayer::new(kernel, None, 1, 0).unwrap();
    let pre_activation = conv_ibp.propagate_ibp(&input).unwrap();

    // ReLU layer
    let relu = ReLULayer;

    // IBP through ReLU
    let ibp_output = relu.propagate_ibp(&pre_activation).unwrap();

    // Now do CROWN backward:
    // Start with identity on ReLU output
    let relu_output_size = 2 * 6; // 12
    let identity = LinearBounds::identity(relu_output_size);

    // Backward through ReLU
    let relu_bounds = relu
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Backward through Conv
    let conv_bounds = conv.propagate_linear(&relu_bounds).unwrap().into_owned();

    // Concretize
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

    let a_pos_l = conv_bounds.lower_a.mapv(|v| v.max(0.0));
    let a_neg_l = conv_bounds.lower_a.mapv(|v| v.min(0.0));
    let a_pos_u = conv_bounds.upper_a.mapv(|v| v.max(0.0));
    let a_neg_u = conv_bounds.upper_a.mapv(|v| v.min(0.0));

    let crown_lower = a_pos_l.dot(&x_l) + a_neg_l.dot(&x_u) + &conv_bounds.lower_b;
    let crown_upper = a_pos_u.dot(&x_u) + a_neg_u.dot(&x_l) + &conv_bounds.upper_b;

    // CROWN should be tighter than or equal to IBP
    let ibp_flat = ibp_output.flatten();
    for i in 0..relu_output_size {
        let ibp_l = ibp_flat.lower().as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper().as_slice().unwrap()[i];
        let crown_l = crown_lower[i];
        let crown_u = crown_upper[i];

        // CROWN should be at least as tight
        assert!(
            crown_l >= ibp_l - 1e-4,
            "Output {}: CROWN lower {} should be >= IBP lower {}",
            i,
            crown_l,
            ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-4,
            "Output {}: CROWN upper {} should be <= IBP upper {}",
            i,
            crown_u,
            ibp_u
        );
    }
}
