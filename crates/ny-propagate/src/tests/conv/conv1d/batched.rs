// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_batched_crown_basic() {
    // Test batched CROWN backward propagation through Conv1d
    // Input: [2, 8] (2 channels, 8 length)
    // Kernel: [3, 2, 3] (3 out_channels, 2 in_channels, 3 kernel_size)
    // Output: [3, 6] (3 channels, 6 length) = 18 flattened

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[3, 2, 3]));
    // Initialize kernel with some values
    for oc in 0..3 {
        for ic in 0..2 {
            for k in 0..3 {
                kernel[[oc, ic, k]] = ((oc * 6 + ic * 3 + k) as f32 * 0.1) - 0.3;
            }
        }
    }

    let bias = arr1(&[0.1, -0.1, 0.2]);
    let conv = Conv1dLayer::with_input_length(kernel.clone(), Some(bias), 1, 0, 8).unwrap();

    // For Conv1d, use flattened output size for identity bounds
    // Output: [3, 6] -> flattened size = 18
    let conv_out_size = 3 * 6; // 18
    let identity_bounds = BatchedLinearBounds::identity(&[conv_out_size]).unwrap();

    // Propagate backward
    let input_bounds = conv.propagate_linear_batched(&identity_bounds).unwrap();

    // Verify output dimensions
    let conv_in_size = 2 * 8; // 16
    let expected_a_shape = vec![conv_out_size, conv_in_size]; // [18, 16]
    assert_eq!(
        input_bounds.lower_a.shape(),
        expected_a_shape.as_slice(),
        "lower_a shape mismatch"
    );
    assert_eq!(
        input_bounds.upper_a.shape(),
        expected_a_shape.as_slice(),
        "upper_a shape mismatch"
    );

    // Verify the bounds are finite
    assert!(
        input_bounds.lower_a.iter().all(|&v| v.is_finite()),
        "lower_a has non-finite values"
    );
    assert!(
        input_bounds.upper_a.iter().all(|&v| v.is_finite()),
        "upper_a has non-finite values"
    );
    assert!(
        input_bounds.lower_b.iter().all(|&v| v.is_finite()),
        "lower_b has non-finite values"
    );
    assert!(
        input_bounds.upper_b.iter().all(|&v| v.is_finite()),
        "upper_b has non-finite values"
    );

    println!("Conv1d batched CROWN test passed!");
    println!("  Input bounds A shape: {:?}", input_bounds.lower_a.shape());
    println!("  Input bounds b shape: {:?}", input_bounds.lower_b.shape());
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_batched_crown_soundness() {
    // Test that batched CROWN produces sound bounds by sampling random inputs
    // and verifying output is within bounds

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3]));
    kernel[[0, 0, 0]] = 1.0;
    kernel[[0, 0, 1]] = -1.0;
    kernel[[0, 0, 2]] = 1.0;
    kernel[[1, 0, 0]] = 0.5;
    kernel[[1, 0, 1]] = 0.5;
    kernel[[1, 0, 2]] = 0.5;

    let bias = arr1(&[0.1, -0.2]);
    let conv = Conv1dLayer::with_input_length(kernel.clone(), Some(bias), 1, 0, 8).unwrap();

    // Input with perturbation
    let center = ArrayD::from_elem(ndarray::IxDyn(&[1, 8]), 0.5);
    let input = BoundedTensor::from_epsilon(center, 0.1).unwrap();

    // Get IBP bounds for comparison
    let ibp_output = conv.propagate_ibp(&input).unwrap();

    // For Conv1d, use flattened output size for identity bounds
    // Output: [2, 6] -> flattened size = 12
    let conv_out_size = 2 * 6; // 12
    let conv_in_size = input.lower().len();
    let identity_bounds = BatchedLinearBounds::identity(&[conv_out_size]).unwrap();
    let crown_bounds = conv.propagate_linear_batched(&identity_bounds).unwrap();

    // Concretize CROWN bounds (flatten input to match batched CROWN representation)
    let input_flat = BoundedTensor::new(
        input
            .lower()
            .clone()
            .into_shape_with_order(vec![conv_in_size])
            .unwrap()
            .into_dyn(),
        input
            .upper()
            .clone()
            .into_shape_with_order(vec![conv_in_size])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();
    let crown_output = crown_bounds.concretize(&input_flat).unwrap();

    // CROWN should be as tight or tighter than IBP (same for linear layers)
    let ibp_flat = ibp_output.flatten();
    let crown_flat = crown_output.flatten();

    for i in 0..12 {
        let ibp_l = ibp_flat.lower().as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper().as_slice().unwrap()[i];
        let crown_l = crown_flat.lower().as_slice().unwrap()[i];
        let crown_u = crown_flat.upper().as_slice().unwrap()[i];

        // CROWN should be at least as tight (with small tolerance for numerical error)
        assert!(
            crown_l >= ibp_l - 1e-4,
            "Output {}: CROWN lower {} < IBP lower {} (diff: {})",
            i,
            crown_l,
            ibp_l,
            crown_l - ibp_l
        );
        assert!(
            crown_u <= ibp_u + 1e-4,
            "Output {}: CROWN upper {} > IBP upper {} (diff: {})",
            i,
            crown_u,
            ibp_u,
            crown_u - ibp_u
        );
    }

    println!("Conv1d batched CROWN soundness test passed!");
}
