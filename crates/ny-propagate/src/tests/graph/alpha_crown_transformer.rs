// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for CROWN-IBP intermediate tightening on transformer-style
//! graphs containing unary ops (GELU, SiLU, Softmax, LayerNorm, RmsNorm).
//!
//! These tests verify the Phase 3 acceptance criteria from
//! `designs/2026-03-13-issue-3628-transformer-intermediate-bounds-architecture.md`:
//! - Narrowed blocklist allows CROWN-IBP collection for unary transformer ops
//! - Demand-driven selection produces strictly tighter intermediates than IBP
//! - Demand-driven provenance correctly skips non-demand nodes

use crate::*;
use ndarray::{arr1, arr2};

fn total_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .upper()
        .iter()
        .zip(bounds.lower().iter())
        .map(|(&u, &l)| u - l)
        .sum()
}

/// Build: Input(2) → Linear(3) → ReLU → Linear(3) → GELU → Linear(3) → ReLU → Linear(2)
///
/// Has GELU (formerly blocklisted unary transformer op) plus ReLU nodes.
/// Exercises the demand-driven path where output-only nodes are skipped.
fn build_relu_gelu_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let linear1 = LinearLayer::new(w1, Some(arr1(&[0.0, 0.1, -0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4], [-0.3, 0.7, 0.1]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "gelu1",
        Layer::GELU(GELULayer::new(GeluApproximation::Tanh)),
        vec!["linear2".to_string()],
    ));

    let w3 = arr2(&[[0.4_f32, 0.3, -0.5], [-0.2, 0.8, 0.1], [0.6, -0.1, 0.3]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(linear3),
        vec!["gelu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));

    let w4 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    let linear4 = LinearLayer::new(w4, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear4",
        Layer::Linear(linear4),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear4");
    graph
}

/// #3628/#3775: Verify the narrowed blocklist allows CROWN-IBP collection
/// and that demand-driven tightening produces strictly tighter intermediates.
///
/// Reference: designs/2026-03-13-issue-3628-transformer-intermediate-bounds-architecture.md
#[ntest::timeout(60000)]
#[test]
fn test_gelu_graph_crown_ibp_tightening_3628() {
    use crate::types::{BoundsProvenance, CrownIbpFallbackReason};

    let graph = build_relu_gelu_graph();
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .unwrap();

    // Blocklist must allow CROWN-IBP (GELU no longer blocked).
    assert!(
        graph.should_use_crown_ibp_intermediates(),
        "#3628/#3775: GELU-containing graph must allow CROWN-IBP intermediates"
    );

    let ibp_bounds = graph.collect_node_bounds(&input).unwrap();
    let result = graph
        .collect_crown_ibp_bounds_dag_with_status_and_engine(&input, None)
        .unwrap();

    // At least one activation-input must be strictly tighter with CROWN-IBP.
    let targets = ["linear1", "linear2", "linear3"];
    let mut any_tighter = false;
    for name in &targets {
        let ibp_w = total_width(ibp_bounds.get(*name).unwrap());
        let crown_w = total_width(result.bounds.get(*name).unwrap());
        assert!(crown_w <= ibp_w + 1e-5, "CROWN-IBP wider at {name}");
        if crown_w + 1e-6 < ibp_w {
            any_tighter = true;
        }
    }
    assert!(any_tighter, "no activation input tightened by CROWN-IBP");

    // Demand-driven skip must fire for at least one non-demand node.
    let skip_count = result
        .provenance
        .values()
        .filter(|p| {
            matches!(
                p,
                BoundsProvenance::ForwardFallback(CrownIbpFallbackReason::DemandDrivenSkip)
            )
        })
        .count();
    assert!(skip_count > 0, "demand-driven skip should fire");

    // Activation inputs must have CROWN provenance (not fallback).
    for name in &targets {
        let p = result.provenance_for_node(name).unwrap();
        assert!(
            matches!(p, BoundsProvenance::Crown),
            "{name} should have CROWN provenance, got {p:?}"
        );
    }
}
