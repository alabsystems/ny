// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for Tile layer IBP and CROWN propagation.
//!
//! This module tests:
//! - TileLayer for GQA (Grouped Query Attention) key/value head expansion
//! - Basic tiling along different axes
//! - Negative axis indexing
//! - CROWN backward propagation (coefficient structure and soundness)
//! - GraphNetwork.propagate_ibp_detailed for layer-by-layer verification

use super::*;
use crate::LinearBounds;
use ndarray::{Array2, ArrayD, IxDyn};

// ============================================================
// TILE LAYER TESTS (for GQA support)
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_tile_layer_basic() {
    // Test basic tile operation: repeat [1, 2] twice along axis 0
    // Input: [[1], [2]] (2D with shape [2, 1])
    // After tile axis=0, reps=3: [[1], [2], [1], [2], [1], [2]] (shape [6, 1])
    use ndarray::array;

    let input = BoundedTensor::new(
        array![[1.0], [2.0]].into_dyn(),
        array![[1.5], [2.5]].into_dyn(),
    )
    .unwrap();

    let tile = TileLayer::new(0, 3);
    let output = tile.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[6, 1]);
    // Lower bounds: [1, 2, 1, 2, 1, 2]
    assert!((output.lower()[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((output.lower()[[1, 0]] - 2.0).abs() < 1e-6);
    assert!((output.lower()[[2, 0]] - 1.0).abs() < 1e-6);
    assert!((output.lower()[[3, 0]] - 2.0).abs() < 1e-6);
    // Upper bounds: [1.5, 2.5, 1.5, 2.5, 1.5, 2.5]
    assert!((output.upper()[[0, 0]] - 1.5).abs() < 1e-6);
    assert!((output.upper()[[5, 0]] - 2.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_tile_layer_gqa_pattern() {
    // Test GQA-style tiling: expand KV heads to match Q heads
    // Input: [seq=2, num_kv_heads=2, head_dim=3]
    // Tile axis=1, reps=4: [seq=2, num_kv_heads*4=8, head_dim=3]
    use ndarray::Array3;

    // Create input with shape [2, 2, 3]
    let lower = Array3::from_shape_fn((2, 2, 3), |(s, h, d)| (s * 10 + h * 3 + d) as f32);
    let upper = lower.clone() + 0.5;

    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();

    let tile = TileLayer::new(1, 4); // Tile along head dimension
    let output = tile.propagate_ibp(&input).unwrap();

    // Output shape should be [2, 8, 3]
    assert_eq!(output.shape(), &[2, 8, 3]);

    // The heads should be repeated: [head0, head1, head0, head1, head0, head1, head0, head1]
    // head0 values at seq=0: [0, 1, 2]
    // head1 values at seq=0: [3, 4, 5]
    assert!((output.lower()[[0, 0, 0]] - 0.0).abs() < 1e-6); // seq0, head0, d0
    assert!((output.lower()[[0, 1, 0]] - 3.0).abs() < 1e-6); // seq0, head1, d0
    assert!((output.lower()[[0, 2, 0]] - 0.0).abs() < 1e-6); // seq0, head0 (repeat)
    assert!((output.lower()[[0, 3, 0]] - 3.0).abs() < 1e-6); // seq0, head1 (repeat)
}

#[ntest::timeout(10000)]
#[test]
fn test_tile_layer_reps_one_is_noop() {
    // Tile with reps=1 should be a no-op
    use ndarray::array;

    let lower = array![[1.0, 2.0], [3.0, 4.0]];
    let upper = array![[1.5, 2.5], [3.5, 4.5]];
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();

    let tile = TileLayer::new(0, 1);
    let output = tile.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    assert!((output.lower()[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((output.upper()[[1, 1]] - 4.5).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_tile_layer_negative_axis() {
    // Test negative axis indexing
    use ndarray::array;

    let input = BoundedTensor::new(
        array![[1.0, 2.0], [3.0, 4.0]].into_dyn(),
        array![[1.5, 2.5], [3.5, 4.5]].into_dyn(),
    )
    .unwrap();

    let tile = TileLayer::new(-1, 2); // Tile along last axis (axis 1)
    let output = tile.propagate_ibp(&input).unwrap();

    // Shape [2, 2] -> [2, 4] when tiling axis 1 by 2
    assert_eq!(output.shape(), &[2, 4]);
    // Row 0: [1, 2, 1, 2]
    assert!((output.lower()[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((output.lower()[[0, 1]] - 2.0).abs() < 1e-6);
    assert!((output.lower()[[0, 2]] - 1.0).abs() < 1e-6);
    assert!((output.lower()[[0, 3]] - 2.0).abs() < 1e-6);
}

// ============================================================
// TILE LAYER CROWN TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_tile_crown_backward_basic() {
    // Test CROWN backward pass for Tile: input [2, 3] -> tile axis=0, reps=2 -> [4, 3]
    // In the backward pass, coefficients from replicated outputs should sum to their input.
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds on the output (4*3=12 elements after tiling)
    let linear_bounds = LinearBounds::identity(12);
    let tile = TileLayer::new(0, 2);

    let result = tile
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Output should have shape (12, 6) - 12 outputs, 6 inputs
    assert_eq!(result.lower_a.nrows(), 12);
    assert_eq!(result.lower_a.ncols(), 6);

    // For tile axis=0, reps=2:
    // Output layout: [row0_input, row1_input, row0_input, row1_input]
    // Output indices 0-2 map to input indices 0-2 (row 0)
    // Output indices 3-5 map to input indices 3-5 (row 1)
    // Output indices 6-8 map to input indices 0-2 (row 0, replica)
    // Output indices 9-11 map to input indices 3-5 (row 1, replica)

    // Input index 0 receives from output indices 0 and 6
    // With identity bounds, coefficient at [0,0] should be 1 (from output 0)
    // and coefficient at [6,0] should also be 1 (from output 6)
    // After backward: new_A[:, 0] = A[:, 0] + A[:, 6] for identity gives [1,0,...0] + [0,...1,...0] per row

    // Check that each input index gets sum of coefficients from all its replicas
    // For identity matrix input, backward pass should give us summed columns
    // Input 0 contributes to outputs 0 and 6 in forward pass
    // In backward, row i of result.lower_a tells us coefficients for output i
    // Since we have identity, row 0 has 1 at position 0 (from output 0 -> input 0)
    // and row 6 has 1 at position 0 (from output 6 -> input 0)

    // Actually, the backward pass sums contributions from all outputs that map to each input
    // So result.lower_a[[row, input_j]] = sum over outputs k that map to input_j of bounds.lower_a[[row, k]]
    // With identity bounds: result.lower_a[[i, j]] = 1 if outputs that map to j include i, else 0

    // Check first row (output 0): should have coefficient at input 0
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-6,
        "Expected 1.0 for [0,0], got {}",
        result.lower_a[[0, 0]]
    );

    // Check row 6: should also have coefficient at input 0 (since output 6 maps to input 0)
    assert!(
        (result.lower_a[[6, 0]] - 1.0).abs() < 1e-6,
        "Expected 1.0 for [6,0], got {}",
        result.lower_a[[6, 0]]
    );

    // Output 3 maps to input 3, output 9 also maps to input 3
    assert!(
        (result.lower_a[[3, 3]] - 1.0).abs() < 1e-6,
        "Expected 1.0 for [3,3], got {}",
        result.lower_a[[3, 3]]
    );
    assert!(
        (result.lower_a[[9, 3]] - 1.0).abs() < 1e-6,
        "Expected 1.0 for [9,3], got {}",
        result.lower_a[[9, 3]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_tile_crown_soundness() {
    // Test that CROWN bounds are sound (equal to IBP for linear operations)
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(12); // 4*3=12 outputs
    let tile = TileLayer::new(0, 2);

    let result = tile
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Concretize CROWN bounds
    let concrete = result.concretize(&pre_activation);

    // IBP result for comparison
    let ibp_result = tile.propagate_ibp(&pre_activation).unwrap();
    let ibp_flat_lower = ibp_result.lower().iter().cloned().collect::<Vec<f32>>();
    let ibp_flat_upper = ibp_result.upper().iter().cloned().collect::<Vec<f32>>();

    // For a linear operation like Tile, CROWN bounds should exactly match IBP bounds
    for i in 0..12 {
        assert!(
            (concrete.lower()[[i]] - ibp_flat_lower[i]).abs() < 1e-5,
            "CROWN lower {} should equal IBP lower {} at index {}",
            concrete.lower()[[i]],
            ibp_flat_lower[i],
            i
        );
        assert!(
            (concrete.upper()[[i]] - ibp_flat_upper[i]).abs() < 1e-5,
            "CROWN upper {} should equal IBP upper {} at index {}",
            concrete.upper()[[i]],
            ibp_flat_upper[i],
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_tile_crown_gqa_pattern() {
    // Test CROWN backward pass for GQA-style tiling (axis=1)
    // Input: [2, 2, 3] (seq, kv_heads, head_dim)
    // Tile axis=1, reps=4 -> [2, 8, 3]
    use ndarray::Array3;

    let lower = Array3::from_shape_fn((2, 2, 3), |(s, h, d)| (s * 10 + h * 3 + d) as f32);
    let upper = lower.clone() + 1.0;
    let pre_activation = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();

    // Identity linear bounds on output (2*8*3=48 elements)
    let linear_bounds = LinearBounds::identity(48);
    let tile = TileLayer::new(1, 4);

    let result = tile
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Output: (48, 12) - 48 outputs, 12 inputs
    assert_eq!(result.lower_a.nrows(), 48);
    assert_eq!(result.lower_a.ncols(), 12);

    // Verify soundness by comparing with IBP
    let concrete = result.concretize(&pre_activation);
    let ibp_result = tile.propagate_ibp(&pre_activation).unwrap();
    let ibp_flat_lower = ibp_result.lower().iter().cloned().collect::<Vec<f32>>();
    let ibp_flat_upper = ibp_result.upper().iter().cloned().collect::<Vec<f32>>();

    for i in 0..48 {
        assert!(
            (concrete.lower()[[i]] - ibp_flat_lower[i]).abs() < 1e-5,
            "CROWN lower {} should equal IBP lower {} at index {} for GQA pattern",
            concrete.lower()[[i]],
            ibp_flat_lower[i],
            i
        );
        assert!(
            (concrete.upper()[[i]] - ibp_flat_upper[i]).abs() < 1e-5,
            "CROWN upper {} should equal IBP upper {} at index {} for GQA pattern",
            concrete.upper()[[i]],
            ibp_flat_upper[i],
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_tile_crown_reps_one() {
    // Test CROWN backward pass when reps=1 (should be identity/no-op)
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(6);
    let tile = TileLayer::new(0, 1); // reps=1, no-op

    let result = tile
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Should be identity matrix (6x6)
    assert_eq!(result.lower_a.nrows(), 6);
    assert_eq!(result.lower_a.ncols(), 6);

    for i in 0..6 {
        for j in 0..6 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (result.lower_a[[i, j]] - expected).abs() < 1e-6,
                "Expected {} at [{},{}], got {}",
                expected,
                i,
                j,
                result.lower_a[[i, j]]
            );
        }
    }
}

// ============================================================
// GRAPH NETWORK DETAILED PROPAGATION TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_propagate_ibp_detailed() {
    // Test layer-by-layer verification (propagate_ibp_detailed)
    // Build a simple 3-layer graph: input -> Linear1 -> ReLU -> Linear2
    use ndarray::Array1;

    let mut graph = GraphNetwork::new();

    // Linear1: 4 -> 3
    let w1 = Array2::from_shape_fn((3, 4), |(i, j)| if i == j { 1.0 } else { 0.0 });
    let b1 = Array1::zeros(3);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();
    graph.add_node(GraphNode::new(
        "linear1".to_string(),
        Layer::Linear(linear1),
        vec!["_input".to_string()],
    ));

    // ReLU
    graph.add_node(GraphNode::new(
        "relu".to_string(),
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    // Linear2: 3 -> 2
    let w2 = Array2::from_shape_fn((2, 3), |(i, j)| if i == j { 2.0 } else { 0.0 });
    let b2 = Array1::from_vec(vec![0.1, 0.2]);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();
    graph.add_node(GraphNode::new(
        "linear2".to_string(),
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));

    graph.set_output("linear2");

    // Create input with epsilon = 0.1
    let epsilon = 0.1;
    let input_data = ArrayD::zeros(IxDyn(&[4]));
    let input = BoundedTensor::from_epsilon(input_data, epsilon).unwrap();

    // Run layer-by-layer verification
    let result = graph.propagate_ibp_detailed(&input, epsilon).unwrap();

    // Check result structure
    assert_eq!(result.total_nodes, 3);
    assert_eq!(result.nodes.len(), 3);
    assert!((result.input_epsilon - epsilon).abs() < 1e-6);

    // Check node names and order
    assert_eq!(result.nodes[0].name, "linear1");
    assert_eq!(result.nodes[1].name, "relu");
    assert_eq!(result.nodes[2].name, "linear2");

    // Check layer types
    assert_eq!(result.nodes[0].layer_type, "Linear");
    assert_eq!(result.nodes[1].layer_type, "ReLU");
    assert_eq!(result.nodes[2].layer_type, "Linear");

    // Check that bounds are reasonable (not degraded)
    for node in &result.nodes {
        assert!(
            !node.has_degraded(),
            "Node {} should not be degraded",
            node.name
        );
        assert!(
            node.sensitivity > 0.0,
            "Sensitivity should be positive for {}",
            node.name
        );
        assert!(
            node.output_width > 0.0,
            "Output width should be positive for {}",
            node.name
        );
    }

    // Check output shapes
    assert_eq!(result.nodes[0].output_shape, vec![3]);
    assert_eq!(result.nodes[1].output_shape, vec![3]);
    assert_eq!(result.nodes[2].output_shape, vec![2]);

    // Verify final width is set
    assert!(result.final_width > 0.0);
    assert!(result.final_width.is_finite());
}
