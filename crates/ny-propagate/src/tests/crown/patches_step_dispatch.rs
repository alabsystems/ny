// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for Patches-mode CROWN dispatch through the public
//! `Network::propagate_crown()` API.
//!
//! When a sequential network contains Conv2d, the CROWN backward loop uses
//! `crown_backward_step_patches` internally. These tests exercise that path
//! by constructing CNN networks and verifying CROWN soundness.
//!
//! Part of #3463

use super::*;
use crate::layers::convolution::conv2d::Conv2dLayer;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

// ============================================================
// Helpers
// ============================================================

/// Build a BoundedTensor input region for a CNN with given spatial shape.
fn cnn_input_region(shape: &[usize]) -> BoundedTensor {
    let n: usize = shape.iter().product();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), vec![0.0_f32; n]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), vec![1.0_f32; n]).unwrap(),
    )
    .unwrap()
}

/// Assert CROWN bounds are sound by sampling concrete points.
fn assert_patches_crown_sound(crown: &BoundedTensor, network: &Network, in_shape: &[usize]) {
    let n: usize = in_shape.iter().product();
    let tol = 1e-3;
    let crown_flat = crown.flatten();

    let test_points: Vec<Vec<f32>> = vec![
        vec![0.5; n],
        vec![0.0; n],
        vec![1.0; n],
        (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
        (0..n).map(|i| i as f32 / n.max(2) as f32).collect(),
    ];

    for (idx, point) in test_points.iter().enumerate() {
        let arr = ArrayD::from_shape_vec(IxDyn(in_shape), point.clone()).unwrap();
        let pt = BoundedTensor::new(arr.clone(), arr).unwrap();
        let output = network.propagate_ibp(&pt).unwrap();
        let out_flat = output.flatten();

        for i in 0..out_flat.lower().len() {
            let val = out_flat.lower()[[i]];
            assert!(
                crown_flat.lower()[[i]] <= val + tol,
                "sample {idx} output[{i}]={val} < crown_lower={}",
                crown_flat.lower()[[i]]
            );
            assert!(
                crown_flat.upper()[[i]] >= val - tol,
                "sample {idx} output[{i}]={val} > crown_upper={}",
                crown_flat.upper()[[i]]
            );
        }
    }
}

// ============================================================
// Conv2d → Flatten → Linear (no activation — CROWN should be exact)
// ============================================================

/// Linear CNN without activation: CROWN Patches dispatch is exercised and
/// should produce exact bounds (no relaxation error).
#[ntest::timeout(10000)]
#[test]
fn test_patches_dispatch_linear_cnn_exact() {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, 0.3, 0.2, 0.4]).unwrap();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // 9 inputs (1*3*3 = conv output) → 2 outputs
    let w = arr2(&[
        [0.1_f32, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1],
    ]);
    let linear = LinearLayer::new(w, Some(arr1(&[0.0_f32, 0.0]))).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    let in_shape = [1, 4, 4];
    let input = cnn_input_region(&in_shape);
    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    assert_patches_crown_sound(&crown, &network, &in_shape);

    // For a linear network, CROWN should match IBP exactly
    let tol = 1e-3;
    let cf = crown.flatten();
    let if_ = ibp.flatten();
    for i in 0..cf.lower().len() {
        assert!(
            (cf.lower()[[i]] - if_.lower()[[i]]).abs() < tol,
            "CROWN lower[{i}]={} != IBP lower[{i}]={}",
            cf.lower()[[i]],
            if_.lower()[[i]]
        );
    }
}

// ============================================================
// Conv2d → ReLU → Flatten → Linear (Patches dispatch + activation)
// ============================================================

/// CNN with ReLU: exercises Patches dispatch for both Conv2d backward and
/// ReLU element-wise activation backward in Patches mode.
///
/// Soundness: sampled concrete outputs lie within CROWN bounds.
/// Tightness: CROWN at least as tight as IBP.
#[ntest::timeout(10000)]
#[test]
fn test_patches_dispatch_conv_relu_soundness() {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, 0.3, 0.2, 0.4]).unwrap();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    let w = arr2(&[
        [0.1_f32, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1],
    ]);
    let linear = LinearLayer::new(w, Some(arr1(&[0.0_f32, 0.0]))).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    let in_shape = [1, 4, 4];
    let input = cnn_input_region(&in_shape);
    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    assert_patches_crown_sound(&crown, &network, &in_shape);

    // CROWN should be at least as tight as IBP
    let tol = 1e-3;
    let cf = crown.flatten();
    let if_ = ibp.flatten();
    for i in 0..cf.lower().len() {
        assert!(
            cf.lower()[[i]] >= if_.lower()[[i]] - tol,
            "CROWN lower[{i}]={} looser than IBP lower[{i}]={}",
            cf.lower()[[i]],
            if_.lower()[[i]]
        );
        assert!(
            cf.upper()[[i]] <= if_.upper()[[i]] + tol,
            "CROWN upper[{i}]={} looser than IBP upper[{i}]={}",
            cf.upper()[[i]],
            if_.upper()[[i]]
        );
    }
}

// ============================================================
// Multi-channel Conv2d: wider network exercises Patches dispatch
// ============================================================

/// Multi-channel Conv2d (2 output channels) with ReLU.
///
/// Wider networks stress the Patches backward more because
/// the patches tensor has shape (out_c, out_h, out_w, in_c, kH, kW).
#[ntest::timeout(10000)]
#[test]
fn test_patches_dispatch_multichannel_conv2d() {
    // 1→2 channels, 2x2 kernel
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 2, 2]),
        vec![0.5, 0.3, 0.2, 0.4, -0.1, 0.2, 0.3, -0.2],
    )
    .unwrap();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // Flatten 2*3*3 = 18 → 2 outputs
    let w_data: Vec<f32> = (0..36).map(|i| (i as f32 - 18.0) / 36.0).collect();
    let w = ndarray::Array2::from_shape_vec((2, 18), w_data).unwrap();
    let linear = LinearLayer::new(w, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    let in_shape = [1, 4, 4];
    let input = cnn_input_region(&in_shape);
    let crown = network.propagate_crown(&input).unwrap();

    assert_patches_crown_sound(&crown, &network, &in_shape);
}
