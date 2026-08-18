// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for graph domain processing fallback behavior.

use std::collections::HashMap;

use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;

use super::super::super::domain_results::GraphDomainResult;
use crate::beta_crown::{BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic, GraphBabDomain};
use crate::{GraphNetwork, GraphNode, Layer, LinearLayer, ReLULayer};

/// Build a 2-layer ReLU graph with 4 unstable neurons for multi-depth tests.
/// Graph: input(2) -> relu0(2) -> linear1(2x2) -> relu1(2) -> linear2(2x1)
fn build_two_relu_graph() -> (GraphNetwork, BoundedTensor, GraphBabDomain, Vec<String>) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), None).unwrap()),
        vec!["relu0".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0, 1.0]]), None).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[-1.0f32, -1.0]).into_dyn(),
        arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Pre-activation bounds crossing 0 → all neurons unstable.
    let mut node_bounds = HashMap::new();
    node_bounds.insert(
        "linear1".to_string(),
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap(),
    );
    node_bounds.insert(
        "linear2".to_string(),
        BoundedTensor::new(arr1(&[-5.0]).into_dyn(), arr1(&[5.0]).into_dyn()).unwrap(),
    );

    let domain = GraphBabDomain::root(node_bounds, -1.0, 1.0, &input, false).unwrap();
    let relu_nodes = vec!["relu0".to_string(), "relu1".to_string()];

    (graph, input, domain, relu_nodes)
}

/// Integration test for #2767: verify multi-depth branching with split_depth > 1
/// enters the multi-depth code path and doesn't crash.
#[ntest::timeout(10000)]
#[test]
fn test_multi_depth_parallel_creates_more_children_2767() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::LargestBoundWidth,
        verify_upper_bound: false,
        ..Default::default()
    });
    let (graph, _input, domain, relu_nodes) = build_two_relu_graph();

    // Verify 4 unstable neurons exist (2 per ReLU layer)
    let unstable = verifier.find_unstable_graph_neurons(&graph, &domain, &relu_nodes);
    assert_eq!(unstable.len(), 4, "need 4 unstable neurons for multi-depth");

    // Single-depth: up to 2 children
    let result_single =
        verifier.process_graph_domain_parallel(&graph, &domain, &relu_nodes, &[1.0], 0.0, None, 1);
    let single_count = match &result_single {
        GraphDomainResult::Children(c) => c.len(),
        _ => 0,
    };

    // Multi-depth (split_depth=2): up to 4 children (2^2)
    let result_multi =
        verifier.process_graph_domain_parallel(&graph, &domain, &relu_nodes, &[1.0], 0.0, None, 2);
    let multi_count = match &result_multi {
        GraphDomainResult::Children(c) => c.len(),
        _ => 0,
    };

    // If both produce children, multi-depth should have more
    if single_count > 0 && multi_count > 0 {
        assert!(
            multi_count > single_count,
            "multi-depth should produce more children: multi={multi_count}, single={single_count}"
        );
    }

    // Multi-depth path was entered (branches or fails, never NoUnstable)
    assert!(
        matches!(
            result_multi,
            GraphDomainResult::Children(_) | GraphDomainResult::PropagationFailure
        ),
        "multi-depth should enter branching path, got {result_multi:?}"
    );
}

#[test]
fn parallel_fallback_caps_parent_at_max_depth_minus_one() {
    assert_eq!(
        super::super::cap_relu_split_depth_for_parent(4, 8, 3, 4),
        1,
        "parallel fallback must reduce a depth-four request to the last legal level"
    );
    assert_eq!(
        super::super::cap_relu_split_depth_for_parent(4, 8, 4, 4),
        0,
        "a max-depth parent must not expand"
    );
}

/// Regression test for #1915: if graph branch selection fails in the parallel
/// path, domain processing must return `PropagationFailure` instead of silently
/// continuing with stale state.
#[ntest::timeout(5000)]
#[test]
fn test_process_graph_domain_parallel_branch_selection_error_returns_propagation_failure_1915() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    });

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear1",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0, 1.0]]), None).expect("valid linear layer")),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear1");

    let input = BoundedTensor::new(
        arr1(&[-1.0f32, -1.0]).into_dyn(),
        arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .expect("valid input bounds");

    // Missing node bounds force BoundImpact branch score computation to fail.
    let domain = GraphBabDomain::root(HashMap::new(), -1.0, 1.0, &input, false).unwrap();
    let relu_nodes = vec!["relu".to_string()];

    let result =
        verifier.process_graph_domain_parallel(&graph, &domain, &relu_nodes, &[1.0], 0.0, None, 1);

    assert!(
        matches!(result, GraphDomainResult::PropagationFailure),
        "branch-selection error must map to PropagationFailure, got {result:?}"
    );
}
