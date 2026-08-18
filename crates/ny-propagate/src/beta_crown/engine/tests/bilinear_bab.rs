// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GenBaB integration tests for BilinearCrown BaB splitting (#286 Phase 5).
//!
//! Split from `genbab.rs` for the 500-line limit.

use super::prelude::*;

/// Build a small attention graph for BaB integration testing:
/// Input([1,1,2,2]) → GELU (Q) + GELU (K) → BilinearCrown(Q@K^T)
///
/// Q and K share input bounds, so QK^T = QQ^T ≥ 0 concretely. Interval
/// analysis treats Q and K as independent → McCormick overapproximation.
fn build_bilinear_bab_graph() -> (GraphNetwork, BoundedTensor) {
    use crate::layers::{BilinearCrownLayer, GELULayer};

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
        "scores",
        Layer::BilinearCrown(BilinearCrownLayer::new(true, None)),
        "q",
        "k",
    ));
    graph.set_output("scores");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), 1.0_f32),
    )
    .unwrap();
    (graph, input)
}

/// Assert CROWN bounds are at least as tight as IBP (soundness).
fn assert_crown_at_least_as_tight_as_ibp(crown: &BoundedTensor, ibp: &BoundedTensor) {
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

/// Compute total interval width across all elements.
fn total_bound_width(bounds: &BoundedTensor) -> f32 {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .map(|(&l, &u)| u - l)
        .sum()
}

/// BaB verification works end-to-end on a graph containing BilinearCrown
/// (#286 Phase 5). GenBaB splits GELU and BilinearCrown inputs to tighten
/// McCormick relaxation and verify sum(QK^T) ≥ -0.5 (true min is 0).
///
/// Source: designs/2026-03-04-286-attention-bilinear-alternative.md (Approach C)
/// Reference: auto_LiRPA BoundMatMul.splittable (linear.py:948)
#[ntest::timeout(60000)]
#[test]
fn test_bilinear_crown_bab_splitting_integration() {
    use crate::beta_crown::nonlinear_branching::NonlinearBranchingConfig;

    let (graph, input) = build_bilinear_bab_graph();
    let ibp = graph.propagate_ibp(&input).unwrap();
    let crown = graph.propagate_crown_batched(&input).unwrap();
    assert_crown_at_least_as_tight_as_ibp(&crown, &ibp);

    eprintln!(
        "Phase 5: IBP width={:.4}, CROWN width={:.4}",
        total_bound_width(&ibp),
        total_bound_width(&crown)
    );

    // BaB with GenBaB: property sum(QK^T) ≥ -0.5. True min is 0 (QQ^T PSD).
    let objective: Vec<f32> = vec![1.0; crown.lower().len()];
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(NonlinearBranchingConfig {
            num_candidates: 4,
            ..Default::default()
        }),
        max_domains: 500,
        timeout: Duration::from_secs(30),
        ..Default::default()
    };
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split(&graph, &input, &objective, -0.5)
        .unwrap();

    eprintln!(
        "BaB: {:?}, explored={}, verified={}, depth={}",
        result.result, result.domains_explored, result.domains_verified, result.max_depth_reached
    );

    // Must not report violation (true min is 0 > -0.5).
    assert!(
        !matches!(
            result.result,
            BabVerificationStatus::Violated { .. }
                | BabVerificationStatus::PotentialViolation { .. }
        ),
        "sum(QK^T) >= -0.5 should hold (true min 0): {:?}",
        result.result
    );
    assert!(
        result.domains_explored >= 1,
        "BaB should explore at least 1 domain"
    );
}

/// McCormick CROWN is at least as tight as IBP for BilinearCrown (#286).
#[ntest::timeout(10000)]
#[test]
fn test_bilinear_crown_vs_ibp_measurement() {
    let (graph, input) = build_bilinear_bab_graph();
    let ibp = graph.propagate_ibp(&input).unwrap();
    let crown = graph.propagate_crown_batched(&input).unwrap();

    let n = ibp.lower().len() as f32;
    let ibp_avg = total_bound_width(&ibp) / n;
    let crown_avg = total_bound_width(&crown) / n;
    eprintln!("IBP avg width={ibp_avg:.4}, CROWN avg width={crown_avg:.4}");

    assert!(
        crown_avg <= ibp_avg + 1e-4,
        "CROWN average width ({crown_avg}) should be <= IBP ({ibp_avg})"
    );
}
