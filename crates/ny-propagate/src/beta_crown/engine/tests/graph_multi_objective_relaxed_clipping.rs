// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;

fn single_input_identity_graph() -> GraphNetwork {
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity layer should build");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("id", Layer::Linear(linear)));
    graph.set_output("id");
    graph
}

fn multi_objective_clip_case() -> (GraphNetwork, BoundedTensor, Vec<Vec<f32>>, [f32; 2]) {
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    )
    .unwrap();
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = [0.2_f32, -0.8_f32];
    (graph, input, objectives, thresholds)
}

fn multi_objective_clip_verifier(enable_relaxed_clip: bool) -> BetaCrownVerifier {
    let base_config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        enable_relaxed_clip,
        relaxed_clip_iterations: 1,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 8,
        max_depth: 1,
        // Needs >2s so per-node CROWN budget (SPEC_CROWN_MIN_NODE_BUDGET_SECS=2.0)
        // doesn't bail to IBP, which discards LinearBounds and breaks clip verification.
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    BetaCrownVerifier::new(base_config)
}

fn run_multi_objective_clip_case(enable_relaxed_clip: bool) -> crate::BetaCrownResult {
    let (graph, input, objectives, thresholds) = multi_objective_clip_case();
    multi_objective_clip_verifier(enable_relaxed_clip)
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("multi-objective graph input split should not error")
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_graph_input_split_multi_objective_relaxed_clip_verifies_clip_only_child_1579() {
    let with_clip = run_multi_objective_clip_case(true);
    let without_clip = run_multi_objective_clip_case(false);

    assert!(
        matches!(with_clip.result, BabVerificationStatus::Verified),
        "relaxed clipping should discharge the left child after the root split, got {:?}",
        with_clip.result
    );
    assert_eq!(
        with_clip.domains_explored, 1,
        "clip-enabled path should resolve both children from the root split"
    );
    assert_eq!(
        with_clip.domains_verified, 2,
        "expected one direct verification and one clip-only verification"
    );

    assert!(
        matches!(without_clip.result, BabVerificationStatus::Unknown { .. }),
        "without relaxed clipping the left child should remain unresolved at depth 1, got {:?}",
        without_clip.result
    );
    assert_eq!(
        without_clip.domains_explored, 2,
        "clip-disabled path should need to pop the unresolved left child"
    );
    assert_eq!(
        without_clip.max_depth_reached, 1,
        "the unresolved child should hit the configured depth limit"
    );
    assert_eq!(
        without_clip.domains_verified, 1,
        "without clipping only the right child should verify directly"
    );
}
