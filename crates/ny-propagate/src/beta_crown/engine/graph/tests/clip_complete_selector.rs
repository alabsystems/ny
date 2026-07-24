// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use crate::beta_crown::config::InputClipType;
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic};
use crate::{BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer};
use ndarray::{arr1, arr2};

fn selector_test_input() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("selector-test input should build")
}

fn selector_test_graph() -> GraphNetwork {
    // Two contradictory sum constraints on the same output space:
    //   y0 = x0 + x1 - 0.8   (wants x0 + x1 <= 0.8)
    //   y1 = -x0 - x1 + 1.2  (wants x0 + x1 >= 1.2)
    //
    // On the split children, one pass of per-row relaxed clipping leaves a
    // non-empty interval hull (for example x0 in [0.4, 0.5], x1 in [0.7, 0.8])
    // even though the joint polytope is infeasible. The full-spec complete
    // path can prove that infeasibility via the combined constraint set.
    let linear = LinearLayer::new(
        arr2(&[[1.0_f32, 1.0_f32], [-1.0_f32, -1.0_f32]]),
        Some(arr1(&[-0.8_f32, 1.2_f32])),
    )
    .expect("selector-test linear layer should build");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("out", Layer::Linear(linear)));
    graph.set_output("out");
    graph
}

fn selector_test_config(input_clip_type: InputClipType) -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        input_clip_type,
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 8,
        max_depth: 1,
        timeout: Duration::from_secs(1),
        ..Default::default()
    }
}

/// Behavioral seam for #3878.
///
/// If the multi-objective selector ever regresses to the relaxed single-row
/// path while `input_clip_type == Complete`, this test flips from
/// `Verified` to `Unknown`: the relaxed route cannot prove the contradictory
/// sum constraints on the first split, but the full-spec complete route can.
#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_complete_selector_is_behaviorally_distinct_3878() {
    let input = selector_test_input();
    let graph = selector_test_graph();
    let objectives = vec![vec![1.0_f32, 0.0], vec![0.0_f32, 1.0]];
    let thresholds = vec![0.0_f32, 0.0];
    let relaxed = BetaCrownVerifier::new(selector_test_config(InputClipType::Relaxed))
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("relaxed selector regression should not error");
    let complete = BetaCrownVerifier::new(selector_test_config(InputClipType::Complete))
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("complete selector regression should not error");

    assert!(
        matches!(relaxed.result, BabVerificationStatus::Unknown { .. }),
        "relaxed single-row clipping should leave the contradictory-sum child unresolved, got {:?}",
        relaxed.result
    );
    assert!(
        matches!(complete.result, BabVerificationStatus::Verified),
        "complete full-spec clipping should verify the contradictory-sum child, got {:?}",
        complete.result
    );
    assert!(
        complete.domains_explored < relaxed.domains_explored,
        "complete selector should discharge the split children earlier than relaxed (complete explored {}, relaxed explored {})",
        complete.domains_explored,
        relaxed.domains_explored
    );
}
