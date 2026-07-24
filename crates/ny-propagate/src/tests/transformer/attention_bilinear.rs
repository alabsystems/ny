// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BilinearCrown attention tests for #286: probs@V through broadcast McCormick.

use super::prelude::*;

/// Build an attention graph with BilinearCrown for both Q@K^T and probs@V.
fn build_bilinear_attention_graph(seq: usize, dim: usize) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    for name in ["q", "k", "v"] {
        graph.add_node(GraphNode::from_input(
            name,
            Layer::GELU(GELULayer::default()),
        ));
    }
    let scale = 1.0 / (dim as f32).sqrt();
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1).with_heuristic_sampling(true)),
        vec!["scores".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "out",
        Layer::BilinearCrown(BilinearCrownLayer::new(false, None)),
        "probs",
        "v",
    ));
    graph.set_output("out");
    let input = BoundedTensor::new(
        ArrayD::from_elem(vec![1, 1, seq, dim], -0.5_f32),
        ArrayD::from_elem(vec![1, 1, seq, dim], 0.5_f32),
    )
    .unwrap();
    (graph, input)
}

#[ntest::timeout(60000)]
#[test]
fn test_bilinear_crown_probs_v_large_seq_fallback() {
    // Part of #286: BilinearCrown probs@V at output node gracefully falls back
    // to partial CROWN (ShapeMismatch: N-D identity last dim != m*n).
    let (graph, input) = build_bilinear_attention_graph(65, 4);
    let ibp = graph.propagate_ibp(&input).unwrap();
    let crown = graph.propagate_crown_batched(&input).unwrap();
    assert_eq!(crown.shape(), &[1, 1, 65, 4]);
    for ((&cl, &cu), (&il, &iu)) in crown
        .lower()
        .iter()
        .zip(crown.upper().iter())
        .zip(ibp.lower().iter().zip(ibp.upper().iter()))
    {
        assert!(cl.is_finite() && cu.is_finite(), "Non-finite CROWN bounds");
        assert!(cl <= cu + 1e-5, "Invalid interval: {cl} > {cu}");
        assert!(cl >= il - 1e-4, "CROWN lower {cl} < IBP lower {il}");
        assert!(cu <= iu + 1e-4, "CROWN upper {cu} > IBP upper {iu}");
    }
}

/// Build a simple graph with BilinearCrown as the output node:
/// Input → GELU (Q path) + GELU (K path) → BilinearCrown(Q, K^T) (output)
///
/// Uses self-attention pattern (Q and K share shape [1, 1, seq, dim])
/// with transpose_b=true so output is [1, 1, seq, seq] and z_size = seq^2.
///
/// This isolates the flat identity optimization (#286) without the
/// complexity of a full attention graph (no Softmax → BilinearCrown chain).
fn build_bilinear_output_graph(seq: usize, dim: usize) -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::binary(
        "out",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("out");

    let shape = vec![1, 1, seq, dim];
    let input = BoundedTensor::new(
        ArrayD::from_elem(shape.clone(), -0.5_f32),
        ArrayD::from_elem(shape, 0.5_f32),
    )
    .unwrap();
    (graph, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_bilinear_output_flat_identity_small() {
    // Part of #286: When BilinearCrown is the output node, McCormick linearization
    // at the output node is inherently no tighter than IBP (the McCormick lower plane
    // is below the actual product curve). The N-D compose correctly skips the output
    // node (identity downstream) and falls through to partial CROWN = IBP.
    //
    // Self-attention: seq=8, dim=4 → output [1,1,8,8], z_size = 8*8 = 64
    let (graph, input) = build_bilinear_output_graph(8, 4);
    let ibp = graph.propagate_ibp(&input).unwrap();
    let crown = graph.propagate_crown_batched(&input).unwrap();
    assert_eq!(crown.shape(), &[1, 1, 8, 8]);

    // Soundness check: CROWN bounds must be at least as tight as IBP.
    // At the output node with identity downstream, CROWN ≈ IBP (partial CROWN fallback).
    for ((&cl, &cu), (&il, &iu)) in crown
        .lower()
        .iter()
        .zip(crown.upper().iter())
        .zip(ibp.lower().iter().zip(ibp.upper().iter()))
    {
        assert!(cl.is_finite() && cu.is_finite(), "Non-finite CROWN bounds");
        assert!(cl <= cu + 1e-5, "Invalid interval: {cl} > {cu}");
        assert!(cl >= il - 1e-4, "CROWN lower {cl} < IBP lower {il}");
        assert!(cu <= iu + 1e-4, "CROWN upper {cu} > IBP upper {iu}");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_bilinear_output_flat_identity_at_threshold() {
    // Part of #286: z_size = 64 * 64 = 4096 — exactly at the threshold.
    // Self-attention: seq=64, dim=1. Output [1,1,64,64], z_size = 64*64 = 4096 = threshold.
    // Should use flat identity (z_size <= 4096), not fall back to partial CROWN.
    let (graph, input) = build_bilinear_output_graph(64, 1);
    let ibp = graph.propagate_ibp(&input).unwrap();
    let crown = graph.propagate_crown_batched(&input).unwrap();
    assert_eq!(crown.shape(), &[1, 1, 64, 64]);

    for ((&cl, &cu), (&il, &iu)) in crown
        .lower()
        .iter()
        .zip(crown.upper().iter())
        .zip(ibp.lower().iter().zip(ibp.upper().iter()))
    {
        assert!(cl.is_finite() && cu.is_finite(), "Non-finite CROWN bounds");
        assert!(cl <= cu + 1e-5, "Invalid interval: {cl} > {cu}");
        assert!(cl >= il - 1e-4, "CROWN lower {cl} < IBP lower {il}");
        assert!(cu <= iu + 1e-4, "CROWN upper {cu} > IBP upper {iu}");
    }
}
