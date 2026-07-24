// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Verify soundness by sampling concrete inputs within a perturbation region
/// and asserting all outputs fall within computed bounds.
fn assert_crown_bounds_contain_concrete_outputs(
    conv: &Conv2dLayer,
    in_shape: &[usize],
    lower_vals: &[f32],
    upper_vals: &[f32],
    crown_lower: &[f32],
    crown_upper: &[f32],
) {
    let num_interior = 10;
    for s in 0..num_interior + 3 {
        let sample: Vec<f32> = if s == num_interior {
            lower_vals.to_vec()
        } else if s == num_interior + 1 {
            upper_vals.to_vec()
        } else if s == num_interior + 2 {
            lower_vals
                .iter()
                .zip(upper_vals.iter())
                .map(|(&l, &u)| f32::midpoint(l, u))
                .collect()
        } else {
            let t = s as f32 / (num_interior - 1).max(1) as f32;
            lower_vals
                .iter()
                .zip(upper_vals.iter())
                .enumerate()
                .map(|(i, (&l, &u))| {
                    let phase = (i as f32 * 0.618_034 + t) % 1.0;
                    l + (u - l) * phase
                })
                .collect()
        };

        let input_nd = ArrayD::from_shape_vec(ndarray::IxDyn(in_shape), sample).unwrap();
        let point = BoundedTensor::new(input_nd.clone(), input_nd).unwrap();
        let output = conv.propagate_ibp(&point).unwrap();
        let out_vals: Vec<f32> = output.lower().iter().copied().collect();

        for (i, &y) in out_vals.iter().enumerate() {
            assert!(
                crown_lower[i] - 1e-5 <= y && y <= crown_upper[i] + 1e-5,
                "Soundness violation at output {}: concrete={}, CROWN=[{}, {}], sample #{}",
                i,
                y,
                crown_lower[i],
                crown_upper[i],
                s
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_batched_crown_basic() {
    // Test batched CROWN backward propagation through Conv2d
    // Input: [2, 4, 4] (2 channels, 4x4 spatial)
    // Kernel: [3, 2, 3, 3] (3 out_channels, 2 in_channels, 3x3 kernel)
    // Output: [3, 2, 2] (3 channels, 2x2 spatial) = 12 flattened

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[3, 2, 3, 3]));
    // Initialize kernel with some values
    for oc in 0..3 {
        for ic in 0..2 {
            for kh in 0..3 {
                for kw in 0..3 {
                    kernel[[oc, ic, kh, kw]] =
                        ((oc * 18 + ic * 9 + kh * 3 + kw) as f32 * 0.05) - 0.4;
                }
            }
        }
    }

    let bias = arr1(&[0.1, -0.1, 0.2]);
    let conv =
        Conv2dLayer::with_input_shape(kernel.clone(), Some(bias), (1, 1), (0, 0), 4, 4).unwrap();

    // For Conv2d, use flattened output size for identity bounds
    // Output: [3, 2, 2] -> flattened size = 12
    let conv_out_size = 3 * 2 * 2; // 12
    let identity_bounds = BatchedLinearBounds::identity(&[conv_out_size]).unwrap();

    // Propagate backward
    let input_bounds = conv
        .propagate_linear_batched(&identity_bounds, None)
        .unwrap();

    // Verify output dimensions
    let conv_in_size = 2 * 4 * 4; // 32
    let expected_a_shape = vec![conv_out_size, conv_in_size]; // [12, 32]
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

    println!("Conv2d batched CROWN test passed!");
    println!("  Input bounds A shape: {:?}", input_bounds.lower_a.shape());
    println!("  Input bounds b shape: {:?}", input_bounds.lower_b.shape());
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_batched_crown_preserves_batch_dims_in_input_shape() {
    // Regression: batched Conv2d backward should preserve batch dims when updating input_shape.
    let kernel = ArrayD::from_elem(ndarray::IxDyn(&[1, 1, 1, 1]), 1.0);
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 4, 4).unwrap();

    let conv_out_size = 4 * 4; // channels=1, 16 flattened outputs
    let conv_in_size = 4 * 4; // channels=1, 16 flattened inputs
    let identity_bounds = BatchedLinearBounds::identity(&[2, conv_out_size]).unwrap();

    let input_bounds = conv
        .propagate_linear_batched(&identity_bounds, None)
        .unwrap();

    assert_eq!(
        input_bounds.input_shape,
        vec![2, conv_in_size],
        "input_shape should preserve batch dims"
    );
    assert_eq!(
        input_bounds.output_shape,
        vec![2, conv_out_size],
        "output_shape should remain flattened"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_batched_crown_soundness() {
    // Test that batched CROWN produces sound bounds: concrete outputs from
    // sampled inputs must fall within the computed CROWN bounds.
    // Fixed: previously only checked CROWN-vs-IBP tightness without sampling
    // concrete inputs, so false-tight bugs would pass silently.

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 1, 3, 3]));
    kernel[[0, 0, 0, 0]] = 1.0;
    kernel[[0, 0, 0, 1]] = -1.0;
    kernel[[0, 0, 0, 2]] = 1.0;
    kernel[[0, 0, 1, 0]] = 0.5;
    kernel[[0, 0, 1, 1]] = 0.5;
    kernel[[0, 0, 1, 2]] = 0.5;
    kernel[[0, 0, 2, 0]] = -0.5;
    kernel[[0, 0, 2, 1]] = -0.5;
    kernel[[0, 0, 2, 2]] = -0.5;
    kernel[[1, 0, 0, 0]] = 0.3;
    kernel[[1, 0, 1, 1]] = 0.3;
    kernel[[1, 0, 2, 2]] = 0.3;

    let bias = arr1(&[0.1, -0.2]);
    let conv =
        Conv2dLayer::with_input_shape(kernel.clone(), Some(bias), (1, 1), (0, 0), 6, 6).unwrap();

    let in_shape = [1_usize, 6, 6];
    let center_3d = ArrayD::from_elem(ndarray::IxDyn(&in_shape), 0.5);
    let input_3d = BoundedTensor::from_epsilon(center_3d, 0.1).unwrap();

    // IBP bounds for tightness comparison
    let ibp_output = conv.propagate_ibp(&input_3d).unwrap();

    let conv_out_size = 2 * 4 * 4; // 32
    let identity_bounds = BatchedLinearBounds::identity(&[conv_out_size]).unwrap();
    let crown_bounds = conv
        .propagate_linear_batched(&identity_bounds, None)
        .unwrap();

    let input_flat = input_3d.flatten();
    let crown_output = crown_bounds.concretize(&input_flat).unwrap();

    let ibp_flat = ibp_output.flatten();
    let crown_flat = crown_output.flatten();

    // Check 1: CROWN should be as tight or tighter than IBP (same for linear layers)
    for i in 0..32 {
        let ibp_l = ibp_flat.lower().as_slice().unwrap()[i];
        let ibp_u = ibp_flat.upper().as_slice().unwrap()[i];
        let crown_l = crown_flat.lower().as_slice().unwrap()[i];
        let crown_u = crown_flat.upper().as_slice().unwrap()[i];

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

    // Check 2: Actual soundness — sample concrete inputs within the perturbation
    // region and verify each output falls within CROWN bounds.
    assert_crown_bounds_contain_concrete_outputs(
        &conv,
        &in_shape,
        input_flat.lower().as_slice().unwrap(),
        input_flat.upper().as_slice().unwrap(),
        crown_flat.lower().as_slice().unwrap(),
        crown_flat.upper().as_slice().unwrap(),
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_batched_crown_vs_regular_crown() {
    // Verify that batched CROWN produces the same results as regular CROWN
    // when batch size is 1

    let mut kernel = ArrayD::zeros(ndarray::IxDyn(&[2, 2, 3, 3]));
    for i in 0..36 {
        let oc = i / 18;
        let ic = (i % 18) / 9;
        let kh = (i % 9) / 3;
        let kw = i % 3;
        kernel[[oc, ic, kh, kw]] = (i as f32 * 0.05) - 0.4;
    }

    let bias = arr1(&[0.1, -0.1]);
    let conv =
        Conv2dLayer::with_input_shape(kernel.clone(), Some(bias), (1, 1), (0, 0), 5, 5).unwrap();

    // Output: [2, 3, 3] = 18 elements
    let conv_out_size = 2 * 3 * 3;

    // Regular CROWN with LinearBounds
    let regular_bounds = LinearBounds::identity(conv_out_size);
    let regular_result = conv.propagate_linear(&regular_bounds).unwrap();

    // Batched CROWN with BatchedLinearBounds
    let batched_bounds = BatchedLinearBounds::identity(&[conv_out_size]).unwrap();
    let batched_result = conv
        .propagate_linear_batched(&batched_bounds, None)
        .unwrap();

    // Results should match
    let regular_la = regular_result.lower_a.as_slice().unwrap();
    let batched_la = batched_result.lower_a.as_slice().unwrap();

    assert_eq!(regular_la.len(), batched_la.len(), "lower_a size mismatch");
    for (i, (&r, &b)) in regular_la.iter().zip(batched_la.iter()).enumerate() {
        assert!(
            (r - b).abs() < 1e-5,
            "lower_a[{}] mismatch: regular={}, batched={}, diff={}",
            i,
            r,
            b,
            (r - b).abs()
        );
    }

    let regular_ua = regular_result.upper_a.as_slice().unwrap();
    let batched_ua = batched_result.upper_a.as_slice().unwrap();

    for (i, (&r, &b)) in regular_ua.iter().zip(batched_ua.iter()).enumerate() {
        assert!(
            (r - b).abs() < 1e-5,
            "upper_a[{}] mismatch: regular={}, batched={}, diff={}",
            i,
            r,
            b,
            (r - b).abs()
        );
    }

    let regular_lb = regular_result.lower_b.as_slice().unwrap();
    let batched_lb = batched_result.lower_b.as_slice().unwrap();

    for (i, (&r, &b)) in regular_lb.iter().zip(batched_lb.iter()).enumerate() {
        assert!(
            (r - b).abs() < 1e-5,
            "lower_b[{}] mismatch: regular={}, batched={}, diff={}",
            i,
            r,
            b,
            (r - b).abs()
        );
    }

    println!("Conv2d batched CROWN matches regular CROWN!");
}

// Note: Conv2d::propagate_linear_batched is implemented and works at the layer level
// (see tests above). Network::propagate_crown_batched now supports Conv2d by
// flattening spatial dimensions into a single feature dim in batched bounds.
