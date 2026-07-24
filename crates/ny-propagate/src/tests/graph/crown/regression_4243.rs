// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression test for #4243 ViT-style attention context reshape bounds.

use crate::types::BoundsProvenance;
use crate::*;
use ndarray::{ArrayD, IxDyn};

fn build_vit_context_reshape_graph_4243() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    // head_dim must equal seq_len so that the context MatMul (probs @ v)
    // produces a square output [batch, heads, seq, seq].  identity_for_attention()
    // requires the last two dimensions to be equal; otherwise it returns None and
    // the full-composition retry never fires.  With heads=4, head_dim=4, seq=4 the
    // final reshape still produces [1, 4, 16] as required by the #4243 contract.
    let head_dim = 4_usize;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    graph.add_node(GraphNode::from_input(
        "q",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "k",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::from_input(
        "v",
        Layer::GELU(GELULayer::default()),
    ));
    graph.add_node(GraphNode::binary(
        "scores",
        Layer::MatMul(MatMulLayer::new(true, Some(scale))),
        "q",
        "k",
    ));
    graph.add_node(GraphNode::new(
        "probs",
        Layer::Softmax(SoftmaxLayer::new(-1).with_heuristic_sampling(true)),
        vec!["scores".into()],
    ));
    graph.add_node(GraphNode::binary(
        "context",
        Layer::MatMul(MatMulLayer::new(false, None)),
        "probs",
        "v",
    ));
    graph.add_node(GraphNode::new(
        "context_bshd",
        Layer::Transpose(TransposeLayer::new(vec![0, 2, 1, 3])),
        vec!["context".into()],
    ));
    graph.add_node(GraphNode::new(
        "flat_context",
        Layer::Reshape(ReshapeLayer::new(vec![0, 0, -1])),
        vec!["context_bshd".into()],
    ));
    graph.set_output("flat_context");

    graph
}

fn assert_finite_ordered_4243(bounds: &BoundedTensor, label: &str) {
    for (idx, (&lower, &upper)) in bounds.lower().iter().zip(bounds.upper().iter()).enumerate() {
        assert!(
            lower.is_finite() && upper.is_finite(),
            "{label}: non-finite bounds at flat index {idx}: lower={lower} upper={upper}"
        );
        assert!(
            lower <= upper + 1e-5,
            "{label}: inverted interval at flat index {idx}: lower={lower} upper={upper}"
        );
    }

    let max_width = bounds.max_width();
    assert!(max_width.is_finite(), "{label}: max width must stay finite");
}

fn max_bound_delta_4243(left: &BoundedTensor, right: &BoundedTensor) -> f32 {
    let lower_delta = left
        .lower()
        .iter()
        .zip(right.lower().iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);
    let upper_delta = left
        .upper()
        .iter()
        .zip(right.upper().iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);

    lower_delta.max(upper_delta)
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_attention_context_reshape_stays_finite_4243() {
    tests::with_crown_dense_budget_mb("2048", || {
        let graph = build_vit_context_reshape_graph_4243();
        let expected_shape = [1_usize, 4, 16];
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1, 4, 4, 4]), -0.5_f32),
            ArrayD::from_elem(IxDyn(&[1, 4, 4, 4]), 0.5_f32),
        )
        .unwrap();

        let ibp_bounds = graph.propagate_ibp(&input).unwrap();
        let public_result = graph
            .propagate_crown_batched_with_attention_full_composition(&input)
            .expect("#4243 public full-composition entrypoint should succeed on ViT reshape graph");
        let (diagnostic_result, used_attention_full_composition) = graph
            .propagate_crown_batched_with_attention_full_composition_diagnostic(&input)
            .expect("#4243 diagnostic helper should succeed on the same ViT reshape graph");

        assert_eq!(ibp_bounds.shape(), &expected_shape[..]);
        assert_eq!(public_result.bounds.shape(), &expected_shape[..]);
        assert_eq!(diagnostic_result.bounds.shape(), &expected_shape[..]);
        assert!(
            used_attention_full_composition,
            "#4243 diagnostic helper must confirm the full-composition branch executed"
        );
        // Provenance is Crown OR Ibp: the verifier returns the tighter of the two
        // sound bounds. Since the MatMul McCormick CROWN-backward now carries its
        // certified f32 coefficient error (#matmul-batched-mccormick), CROWN is
        // honestly slightly looser and IBP can win on this tiny ViT graph — both are
        // sound. What matters is the bound is finite/ordered (checked below) and the
        // public and diagnostic entrypoints agree.
        assert!(
            matches!(
                public_result.provenance,
                BoundsProvenance::Crown | BoundsProvenance::ForwardFallback(_)
            ),
            "#4243 provenance should be Crown or ForwardFallback, got {:?}",
            public_result.provenance
        );
        assert_eq!(public_result.provenance, diagnostic_result.provenance);

        assert_finite_ordered_4243(&public_result.bounds, "#4243 public full composition");
        assert_finite_ordered_4243(
            &diagnostic_result.bounds,
            "#4243 diagnostic full composition",
        );

        let max_delta = max_bound_delta_4243(&public_result.bounds, &diagnostic_result.bounds);
        assert!(
            max_delta <= 1e-6,
            "#4243 public and diagnostic full-composition results diverged: max_delta={max_delta:.6e}"
        );
    });
}
