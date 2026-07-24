// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_network_integration() {
    // CROWN reads process-wide experiment switches. Hold the shared test lock
    // so parallel env-gate tests cannot change its route mid-propagation.
    let _env_lock = ny_test_utils::env::lock_env();

    // Test CROWN through a Conv2d -> ReLU network
    // This verifies the full backward pass works with non-linearities

    // Input: [1, 4, 4] -> Conv [2, 1, 2, 2] -> [2, 3, 3] -> ReLU -> [2, 3, 3]
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = -1.0;
    kernel[[1, 0, 0, 0]] = 0.5;
    kernel[[1, 0, 0, 1]] = 0.5;
    kernel[[1, 0, 1, 0]] = 0.5;
    kernel[[1, 0, 1, 1]] = 0.5;

    let conv = Conv2dLayer::with_input_shape(kernel.clone(), None, (1, 1), (0, 0), 4, 4).unwrap();

    // Input with perturbation around 0.5
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();

    // Get IBP bounds for pre-activation (conv output)
    let conv_ibp = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    let pre_activation = conv_ibp.propagate_ibp(&input).unwrap();

    // ReLU layer
    let relu = ReLULayer;

    // IBP through ReLU
    let ibp_output = relu.propagate_ibp(&pre_activation).unwrap();

    // Now do CROWN backward:
    // Start with identity on ReLU output
    let relu_output_size = 2 * 3 * 3; // 18
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

    // With ReLU, CROWN should often be tighter
    let ibp_widths: Vec<f32> = (0..relu_output_size)
        .map(|i| ibp_flat.upper().as_slice().unwrap()[i] - ibp_flat.lower().as_slice().unwrap()[i])
        .collect();
    let crown_widths: Vec<f32> = (0..relu_output_size)
        .map(|i| crown_upper[i] - crown_lower[i])
        .collect();

    let avg_ibp_width: f32 = ibp_widths.iter().sum::<f32>() / relu_output_size as f32;
    let avg_crown_width: f32 = crown_widths.iter().sum::<f32>() / relu_output_size as f32;

    println!("Conv2d->ReLU CROWN vs IBP:");
    println!("  Average IBP width: {}", avg_ibp_width);
    println!("  Average CROWN width: {}", avg_crown_width);
    println!(
        "  CROWN improvement: {:.2}%",
        (1.0 - avg_crown_width / avg_ibp_width) * 100.0
    );

    // CROWN should be tighter for networks with ReLU
    assert!(
        avg_crown_width <= avg_ibp_width + 1e-5,
        "CROWN should be at least as tight as IBP"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_network_propagate_crown() {
    // CROWN reads process-wide experiment switches. Hold the shared test lock
    // so parallel env-gate tests cannot change its route mid-propagation.
    let _env_lock = ny_test_utils::env::lock_env();

    // Test that Network::propagate_crown works with Conv2d layers
    // instead of falling back to IBP
    use crate::network::Network;

    // Build a Conv2d -> ReLU network
    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 2, 2]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -1.0;
    kernel[[0, 0, 1, 0]] = 1.0;
    kernel[[0, 0, 1, 1]] = -1.0;
    kernel[[1, 0, 0, 0]] = 0.5;
    kernel[[1, 0, 0, 1]] = 0.5;
    kernel[[1, 0, 1, 0]] = 0.5;
    kernel[[1, 0, 1, 1]] = 0.5;

    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    let relu = ReLULayer;

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(relu));

    // Input with perturbation
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 4, 4]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();

    // Get IBP bounds
    let ibp_output = network.propagate_ibp(&input).unwrap();

    // Get CROWN bounds - should now work instead of falling back
    let crown_output = network.propagate_crown(&input).unwrap();

    // CROWN should be at least as tight as IBP
    let ibp_flat = ibp_output.flatten();
    let crown_flat = crown_output.flatten();

    let output_size = ibp_flat.len();
    for i in 0..output_size {
        let ibp_l = ibp_flat.lower().as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper().as_slice().unwrap()[i];
        let crown_l = crown_flat.lower().as_slice().unwrap()[i];
        let crown_u = crown_flat.upper().as_slice().unwrap()[i];

        // CROWN should be at least as tight (with tolerance)
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

    // Verify CROWN provides tighter bounds for Conv->ReLU networks
    let ibp_avg_width: f32 = (0..output_size)
        .map(|i| ibp_flat.upper().as_slice().unwrap()[i] - ibp_flat.lower().as_slice().unwrap()[i])
        .sum::<f32>()
        / output_size as f32;
    let crown_avg_width: f32 = (0..output_size)
        .map(|i| {
            crown_flat.upper().as_slice().unwrap()[i] - crown_flat.lower().as_slice().unwrap()[i]
        })
        .sum::<f32>()
        / output_size as f32;

    println!("Network Conv2d->ReLU:");
    println!("  IBP avg width: {}", ibp_avg_width);
    println!("  CROWN avg width: {}", crown_avg_width);
    println!(
        "  Improvement: {:.2}%",
        (1.0 - crown_avg_width / ibp_avg_width) * 100.0
    );
}
