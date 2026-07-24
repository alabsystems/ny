// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `crown_backward_step` dispatch (crown.rs:60).
//!
//! `crown_backward_step` is module-private, so we test through the public
//! `Network::propagate_crown()` API. Each test constructs a minimal sequential
//! network where the target layer is the dispatch-determining layer, then
//! verifies CROWN backward produces sound bounds.
//!
//! Soundness check: sample concrete points in the input region, evaluate the
//! network, and verify all outputs fall within the CROWN bounds.
//!
//! Part of #3463 (Medium item: crown_backward_step dispatch coverage).

use super::*;
use crate::layers::{Conv1dLayer, ConvTranspose1dLayer, SliceLayer, TileLayer, TransposeLayer};
use ndarray::{arr1, arr2, ArrayD, IxDyn};

/// Helper: evaluate a sequential network at a single concrete point with given shape.
fn evaluate_network_shaped(network: &Network, point: &[f32], shape: &[usize]) -> Vec<f32> {
    let arr = ArrayD::from_shape_vec(IxDyn(shape), point.to_vec()).unwrap();
    let input = BoundedTensor::new(arr.clone(), arr).unwrap();
    let output = network.propagate_ibp(&input).unwrap();
    output.lower().iter().copied().collect()
}

/// Helper: sample corners and midpoints of input region, return concrete outputs.
/// `shape` is the shape of the input tensor (e.g., [1, 4] for Conv1d).
fn sample_outputs_shaped(
    network: &Network,
    lower: &[f32],
    upper: &[f32],
    shape: &[usize],
) -> Vec<Vec<f32>> {
    let dim = lower.len();
    let mut outputs = Vec::new();

    // Midpoint
    let mid: Vec<f32> = lower
        .iter()
        .zip(upper)
        .map(|(l, u)| (l + u) / 2.0)
        .collect();
    outputs.push(evaluate_network_shaped(network, &mid, shape));

    // All corners (2^dim for small dim)
    if dim <= 6 {
        for mask in 0..(1u32 << dim) {
            let point: Vec<f32> = (0..dim)
                .map(|i| {
                    if mask & (1 << i) != 0 {
                        upper[i]
                    } else {
                        lower[i]
                    }
                })
                .collect();
            outputs.push(evaluate_network_shaped(network, &point, shape));
        }
    }

    // Intermediate points
    for k in 1..=3 {
        let t = k as f32 / 4.0;
        let point: Vec<f32> = lower
            .iter()
            .zip(upper)
            .map(|(l, u)| l + (u - l) * t)
            .collect();
        outputs.push(evaluate_network_shaped(network, &point, shape));
    }

    outputs
}

/// Assert CROWN bounds are sound: all sampled concrete outputs lie within bounds.
fn assert_crown_sound_shaped(
    crown: &BoundedTensor,
    network: &Network,
    lower: &[f32],
    upper: &[f32],
    shape: &[usize],
    label: &str,
) {
    let tol = 1e-3;
    let outputs = sample_outputs_shaped(network, lower, upper, shape);
    let crown_flat = crown.flatten();
    let cl = crown_flat.lower();
    let cu = crown_flat.upper();

    for (sample_idx, output) in outputs.iter().enumerate() {
        for (i, &val) in output.iter().enumerate() {
            assert!(
                cl[[i]] <= val + tol,
                "{label}: sample {sample_idx} output[{i}]={val} < crown_lower[{i}]={}",
                cl[[i]]
            );
            assert!(
                cu[[i]] >= val - tol,
                "{label}: sample {sample_idx} output[{i}]={val} > crown_upper[{i}]={}",
                cu[[i]]
            );
        }
    }
}

/// Flat (1D) variant of assert_crown_sound for networks with 1D inputs.
fn assert_crown_sound(
    crown: &BoundedTensor,
    network: &Network,
    lower: &[f32],
    upper: &[f32],
    label: &str,
) {
    let shape = vec![lower.len()];
    assert_crown_sound_shaped(crown, network, lower, upper, &shape, label);
}

// ============================================================
// Conv1d dispatch arm (crown.rs:75-91)
// ============================================================

/// Conv1d in a sequential network exercises the Conv1d dispatch arm.
/// Network: Conv1d(1→1, kernel_size=2, stride=1, padding=0).
/// Input shape: [1, 4] (channels=1, length=4).
#[ntest::timeout(10000)]
#[test]
fn test_crown_step_dispatch_conv1d_normal() {
    // Kernel: [out_ch=1, in_ch=1, kernel_size=2]
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0f32, 0.5]).unwrap();
    let bias = Some(arr1(&[0.1f32]));
    let conv = Conv1dLayer::new(kernel, bias, 1, 0).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv1d(conv));

    // Input: [1, 4] — 1 channel, 4 timesteps
    let lower = vec![0.0f32; 4];
    let upper = vec![1.0f32; 4];
    let shape = vec![1, 4];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape), lower.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&shape), upper.clone()).unwrap(),
    )
    .unwrap();

    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    // CROWN should be sound
    assert_crown_sound_shaped(&crown, &network, &lower, &upper, &shape, "Conv1d");

    // CROWN should be at least as tight as IBP (for linear network, equal)
    let crown_flat = crown.flatten();
    let ibp_flat = ibp.flatten();
    for i in 0..ibp_flat.lower().len() {
        assert!(
            crown_flat.lower()[[i]] >= ibp_flat.lower()[[i]] - 1e-5,
            "Conv1d: CROWN lower[{i}] looser than IBP"
        );
        assert!(
            crown_flat.upper()[[i]] <= ibp_flat.upper()[[i]] + 1e-5,
            "Conv1d: CROWN upper[{i}] looser than IBP"
        );
    }
}

/// Conv1d followed by ReLU: CROWN backward propagates through ReLU (catch-all)
/// then through Conv1d (Conv1d dispatch arm). Tests the composition.
#[ntest::timeout(10000)]
#[test]
fn test_crown_step_dispatch_conv1d_relu_soundness() {
    // Conv1d(1→1, k=2) → ReLU — crossing region for some outputs
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0f32, -1.0]).unwrap();
    let bias = Some(arr1(&[0.0f32]));
    let conv = Conv1dLayer::new(kernel, bias, 1, 0).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv1d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    let lower = vec![0.0f32; 4];
    let upper = vec![1.0f32; 4];
    let shape = vec![1, 4];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape), lower.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&shape), upper.clone()).unwrap(),
    )
    .unwrap();

    let crown = network.propagate_crown(&input).unwrap();

    // Soundness only (CROWN may differ from IBP due to ReLU relaxation)
    assert_crown_sound_shaped(&crown, &network, &lower, &upper, &shape, "Conv1d+ReLU");
}

// ============================================================
// ConvTranspose1d dispatch arm (crown.rs:93-109)
// ============================================================

/// ConvTranspose1d exercises the ConvTranspose1d dispatch arm.
/// Network: ConvTranspose1d(1→1, kernel_size=2, stride=1, padding=0).
#[ntest::timeout(10000)]
#[test]
fn test_crown_step_dispatch_conv_transpose1d() {
    // Kernel: [in_ch=1, out_ch=1, kernel_size=2]
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0f32, 0.5]).unwrap();
    let bias = Some(arr1(&[0.2f32]));
    let conv = ConvTranspose1dLayer::new(kernel, bias, 1, 0).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::ConvTranspose1d(conv));

    // Input: [1, 3] — 1 channel, 3 timesteps
    let lower = vec![0.0f32; 3];
    let upper = vec![1.0f32; 3];
    let shape = vec![1, 3];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape), lower.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&shape), upper.clone()).unwrap(),
    )
    .unwrap();

    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    // Soundness: CROWN bounds contain all concrete outputs
    assert_crown_sound_shaped(&crown, &network, &lower, &upper, &shape, "ConvTranspose1d");

    // CROWN at least as tight as IBP
    let crown_flat = crown.flatten();
    let ibp_flat = ibp.flatten();
    for i in 0..ibp_flat.lower().len() {
        assert!(
            crown_flat.lower()[[i]] >= ibp_flat.lower()[[i]] - 1e-5,
            "ConvTranspose1d: CROWN lower[{i}] looser than IBP"
        );
        assert!(
            crown_flat.upper()[[i]] <= ibp_flat.upper()[[i]] + 1e-5,
            "ConvTranspose1d: CROWN upper[{i}] looser than IBP"
        );
    }
}

// ============================================================
// Transpose dispatch arm (crown.rs:153-161)
// ============================================================

/// Transpose exercises the set_input_shape + propagate_linear path.
/// Network: Linear(2→4) → Reshape([2,2]) → Transpose([1,0]).
/// Transpose swaps the 2×2 matrix to produce 2×2 with swapped axes.
#[ntest::timeout(10000)]
#[test]
fn test_crown_step_dispatch_transpose() {
    let w = arr2(&[[1.0f32, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 1.0]]);
    let b = arr1(&[0.0f32, 0.0, 0.0, 0.0]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![2, 2])));
    network.add_layer(Layer::Transpose(TransposeLayer::new(vec![1, 0])));

    let lower = vec![0.0f32, 0.0];
    let upper = vec![1.0f32, 1.0];
    let input = BoundedTensor::new(arr1(&lower).into_dyn(), arr1(&upper).into_dyn()).unwrap();

    let crown = network.propagate_crown(&input).unwrap();
    assert_crown_sound(&crown, &network, &lower, &upper, "Transpose");
}

// ============================================================
// Tile dispatch arm (crown.rs:162-175)
// ============================================================

/// Tile exercises the clone + set_input_shape + propagate_linear path.
/// Network: Linear(2→2) → Tile(axis=0, reps=2).
/// Output is [x1, x2, x1, x2] (doubled along axis 0).
#[ntest::timeout(10000)]
#[test]
fn test_crown_step_dispatch_tile() {
    let w = arr2(&[[1.0f32, 0.0], [0.0, 1.0]]);
    let b = arr1(&[0.0f32, 0.0]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::Tile(TileLayer::new(0, 2)));

    let lower = vec![0.0f32, 0.0];
    let upper = vec![1.0f32, 1.0];
    let input = BoundedTensor::new(arr1(&lower).into_dyn(), arr1(&upper).into_dyn()).unwrap();

    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    // Soundness
    assert_crown_sound(&crown, &network, &lower, &upper, "Tile");

    // For a pure-linear network (Linear + Tile), CROWN should be exact (= IBP)
    let crown_flat = crown.flatten();
    let ibp_flat = ibp.flatten();
    for i in 0..ibp_flat.lower().len() {
        assert!(
            (crown_flat.lower()[[i]] - ibp_flat.lower()[[i]]).abs() < 1e-5,
            "Tile: CROWN lower[{i}]={} != IBP lower[{i}]={}",
            crown_flat.lower()[[i]],
            ibp_flat.lower()[[i]]
        );
    }
}

// ============================================================
// Slice dispatch arm (crown.rs:176-190)
// ============================================================

/// Slice exercises the clone + set_input_shape + propagate_linear path.
/// Network: Linear(3→4) → Slice(axis=0, start=1, end=3).
/// Output selects indices [1, 2] from the 4-element Linear output.
#[ntest::timeout(10000)]
#[test]
fn test_crown_step_dispatch_slice() {
    let w = arr2(&[
        [1.0f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]);
    let b = arr1(&[0.0f32, 0.0, 0.0, 0.0]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::Slice(SliceLayer::new(0, 1, 3)));

    let lower = vec![0.0f32, 0.0, 0.0];
    let upper = vec![1.0f32, 1.0, 1.0];
    let input = BoundedTensor::new(arr1(&lower).into_dyn(), arr1(&upper).into_dyn()).unwrap();

    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    // Soundness
    assert_crown_sound(&crown, &network, &lower, &upper, "Slice");

    // For a pure-linear network, CROWN should match IBP exactly
    let crown_flat = crown.flatten();
    let ibp_flat = ibp.flatten();
    assert_eq!(
        crown_flat.lower().len(),
        2,
        "Slice should select 2 elements"
    );
    for i in 0..ibp_flat.lower().len() {
        assert!(
            (crown_flat.lower()[[i]] - ibp_flat.lower()[[i]]).abs() < 1e-5,
            "Slice: CROWN lower[{i}]={} != IBP lower[{i}]={}",
            crown_flat.lower()[[i]],
            ibp_flat.lower()[[i]]
        );
    }
}

// ============================================================
// Unary catch-all trait dispatch (crown.rs:224-276)
// with activation: ReLU crossing + CROWN tighter than IBP
// ============================================================

/// The catch-all arm dispatches non-linear activations via propagate_crown_backward.
/// Linear → ReLU crossing: CROWN should be tighter than IBP.
/// This tests the catch-all match arm path, not a specific layer arm.
#[ntest::timeout(10000)]
#[test]
fn test_crown_step_dispatch_catchall_activation_tighter() {
    // Linear → ReLU with crossing region (pre-ReLU spans negative and positive)
    let w = arr2(&[[1.0f32, -1.0], [-1.0, 1.0]]);
    let b = arr1(&[0.0f32, 0.0]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network.add_layer(Layer::ReLU(ReLULayer));
    // Add a final linear to combine ReLU outputs
    let w2 = arr2(&[[1.0f32, 1.0]]);
    let b2 = arr1(&[0.0f32]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let lower = vec![0.0f32, 0.0];
    let upper = vec![1.0f32, 1.0];
    let input = BoundedTensor::new(arr1(&lower).into_dyn(), arr1(&upper).into_dyn()).unwrap();

    let crown = network.propagate_crown(&input).unwrap();
    let ibp = network.propagate_ibp(&input).unwrap();

    // CROWN soundness
    assert_crown_sound(&crown, &network, &lower, &upper, "ReLU-catchall");

    // CROWN should be strictly tighter than IBP for crossing ReLU
    let crown_flat = crown.flatten();
    let ibp_flat = ibp.flatten();
    let crown_width = crown_flat.upper()[[0]] - crown_flat.lower()[[0]];
    let ibp_width = ibp_flat.upper()[[0]] - ibp_flat.lower()[[0]];
    assert!(
        crown_width < ibp_width + 1e-6,
        "CROWN width ({crown_width}) should be <= IBP width ({ibp_width})"
    );
}

// ============================================================
// SkipMerge no-op dispatch arm (crown.rs:202)
// ============================================================

/// SkipMerge is a no-op match arm. In a sequential network, it should be
/// transparent — CROWN bounds should be identical to the network without it.
#[ntest::timeout(10000)]
#[test]
fn test_crown_step_dispatch_skip_merge_noop() {
    let w = arr2(&[[2.0f32, -1.0], [1.0, 3.0]]);
    let b = arr1(&[0.5f32, -0.5]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    // Network WITH SkipMerge
    let mut network_skip = Network::new();
    network_skip.add_layer(Layer::Linear(linear.clone()));
    network_skip.add_layer(Layer::SkipMerge(SkipMergeLayer));

    // Network WITHOUT SkipMerge
    let mut network_no_skip = Network::new();
    network_no_skip.add_layer(Layer::Linear(linear));

    let lower = vec![0.0f32, 0.0];
    let upper = vec![1.0f32, 1.0];
    let input = BoundedTensor::new(arr1(&lower).into_dyn(), arr1(&upper).into_dyn()).unwrap();

    let crown_skip = network_skip.propagate_crown(&input).unwrap();
    let crown_no_skip = network_no_skip.propagate_crown(&input).unwrap();

    // SkipMerge is a no-op: bounds should be identical
    let skip_flat = crown_skip.flatten();
    let no_skip_flat = crown_no_skip.flatten();
    for i in 0..no_skip_flat.lower().len() {
        assert!(
            (skip_flat.lower()[[i]] - no_skip_flat.lower()[[i]]).abs() < 1e-6,
            "SkipMerge should be transparent: lower[{i}] differs"
        );
        assert!(
            (skip_flat.upper()[[i]] - no_skip_flat.upper()[[i]]).abs() < 1e-6,
            "SkipMerge should be transparent: upper[{i}] differs"
        );
    }
}
