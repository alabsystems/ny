// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork integration tests for the RoPE (Rotary Position Embedding) layer.
//!
//! Verifies that RoPE composes correctly with other layers in a GraphNetwork,
//! including both IBP forward and CROWN backward propagation.
//!
//! Part of #3155.

use crate::*;
use ndarray::{arr1, arr2};

/// Compute the RoPE forward pass for a concrete 1D input.
///
/// Given `x` of length `head_dim` and cos/sin frequencies of length `head_dim/2`:
///   y[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
///   y[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]
fn rope_forward(x: &[f32], cos_freqs: &[f32], sin_freqs: &[f32]) -> Vec<f32> {
    let num_pairs = cos_freqs.len();
    let mut y = vec![0.0f32; x.len()];
    for i in 0..num_pairs {
        let c = cos_freqs[i];
        let s = sin_freqs[i];
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        y[2 * i] = x0 * c - x1 * s;
        y[2 * i + 1] = x0 * s + x1 * c;
    }
    y
}

/// Assert that `output` lies within `bounds` at every position, with tolerance `tol`.
fn assert_within_bounds(output: &[f32], bounds: &BoundedTensor, label: &str, tol: f32) {
    for (j, &val) in output.iter().enumerate() {
        assert!(
            val >= bounds.lower()[[j]] - tol,
            "{label} lower violation: output[{j}]={val} < lower={}",
            bounds.lower()[[j]]
        );
        assert!(
            val <= bounds.upper()[[j]] + tol,
            "{label} upper violation: output[{j}]={val} > upper={}",
            bounds.upper()[[j]]
        );
    }
}

/// Assert CROWN bounds are at least as tight as IBP bounds for `dim` positions.
fn assert_crown_tighter_than_ibp(crown: &BoundedTensor, ibp: &BoundedTensor, dim: usize) {
    for j in 0..dim {
        assert!(
            crown.lower()[[j]] >= ibp.lower()[[j]] - 1e-4,
            "CROWN lower[{j}]={} looser than IBP lower={}",
            crown.lower()[[j]],
            ibp.lower()[[j]]
        );
        assert!(
            crown.upper()[[j]] <= ibp.upper()[[j]] + 1e-4,
            "CROWN upper[{j}]={} looser than IBP upper={}",
            crown.upper()[[j]],
            ibp.upper()[[j]]
        );
    }
}

/// Single RoPE layer in a GraphNetwork: IBP bounds are sound.
///
/// Constructs a graph with one RoPE node fed directly from input. Samples
/// concrete points at box corners and interior, and verifies all outputs
/// lie within the computed IBP bounds.
#[ntest::timeout(10000)]
#[test]
fn test_rope_in_network_ibp() {
    let angle = std::f32::consts::FRAC_PI_4; // 45 degrees
    let cos_freqs = vec![angle.cos(), (2.0 * angle).cos()];
    let sin_freqs = vec![angle.sin(), (2.0 * angle).sin()];

    let rope = RopeLayer::new(cos_freqs.clone(), sin_freqs.clone()).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("rope", Layer::RoPE(rope)));
    graph.set_output("rope");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.5, 0.0, -1.5]).into_dyn(),
        arr1(&[1.0_f32, 0.5, 2.0, 0.5]).into_dyn(),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    assert_eq!(ibp_bounds.shape(), &[4]);

    let test_points: Vec<[f32; 4]> = vec![
        [-1.0, -0.5, 0.0, -1.5], // lower corner
        [1.0, 0.5, 2.0, 0.5],    // upper corner
        [0.0, 0.0, 1.0, -0.5],   // interior
        [-1.0, 0.5, 2.0, -1.5],  // mixed corners
        [1.0, -0.5, 0.0, 0.5],   // mixed corners
    ];

    for point in &test_points {
        let output = rope_forward(point, &cos_freqs, &sin_freqs);
        assert_within_bounds(&output, &ibp_bounds, "IBP", 1e-5);
    }
}

/// Single RoPE layer in a GraphNetwork: CROWN bounds are sound and at least as
/// tight as IBP (RoPE is linear, so CROWN should be exact).
#[ntest::timeout(10000)]
#[test]
fn test_rope_in_network_crown() {
    let angle = std::f32::consts::FRAC_PI_3; // 60 degrees
    let cos_freqs = vec![angle.cos()];
    let sin_freqs = vec![angle.sin()];

    let rope = RopeLayer::new(cos_freqs.clone(), sin_freqs.clone()).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("rope", Layer::RoPE(rope)));
    graph.set_output("rope");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    let test_points: Vec<[f32; 2]> = vec![
        [-1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [0.0, 0.0],
        [0.5, -0.3],
    ];

    for point in &test_points {
        let output = rope_forward(point, &cos_freqs, &sin_freqs);
        assert_within_bounds(&output, &crown_bounds, "CROWN", 1e-5);
    }

    assert_crown_tighter_than_ibp(&crown_bounds, &ibp_bounds, 2);
}

/// Linear -> RoPE sequential composition: CROWN backward propagates correctly
/// through both layers.
///
/// Tests the avoice use case where a linear projection (Q/K) is followed
/// by RoPE rotation. Both layers are linear, so CROWN should be exact.
#[ntest::timeout(10000)]
#[test]
fn test_rope_after_linear_crown() {
    let weight = arr2(&[
        [1.0_f32, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.5, 0.0, 0.0, 1.0],
    ]);
    let bias = arr1(&[0.1_f32, -0.1, 0.2, -0.2]);
    let linear = LinearLayer::new(weight.clone(), Some(bias.clone())).unwrap();

    let rope = RopeLayer::from_position(3, 4, 10000.0).unwrap();
    let cos_freqs: Vec<f32> = rope.cos_freqs.clone();
    let sin_freqs: Vec<f32> = rope.sin_freqs.clone();

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "rope",
        Layer::RoPE(rope),
        vec!["linear".to_string()],
    ));
    graph.set_output("rope");

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5, -0.5, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    let ibp_bounds = graph.propagate_ibp(&input).unwrap();
    let crown_bounds = graph.propagate_crown(&input).unwrap();

    let test_points: Vec<[f32; 4]> = vec![
        [-0.5, -0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0],
        [-0.5, 0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.3, -0.2, 0.1, -0.4],
    ];

    for point in &test_points {
        let x = arr1(point);
        let linear_out: Vec<f32> = (0..4).map(|i| weight.row(i).dot(&x) + bias[i]).collect();
        let rope_out = rope_forward(&linear_out, &cos_freqs, &sin_freqs);
        assert_within_bounds(&rope_out, &ibp_bounds, "IBP", 1e-5);
        assert_within_bounds(&rope_out, &crown_bounds, "CROWN", 1e-5);
    }

    assert_crown_tighter_than_ibp(&crown_bounds, &ibp_bounds, 4);
}
