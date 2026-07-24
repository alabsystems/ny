// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2};
use ny_core::NaiveCpuGemmEngine;
use ny_propagate::beta_crown::{BatchedDomains, GraphBabDomain};
use ny_propagate::layers::{LinearLayer, SigmoidLayer, SkipMergeLayer};
use ny_propagate::{BetaCrownConfig, BetaCrownVerifier, GraphNetwork, GraphNode, Layer};
use ny_tensor::BoundedTensor;

fn build_single_domain(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
) -> (GraphBabDomain, BatchedDomains) {
    let mut initial_bounds = std::collections::HashMap::new();
    for node_name in graph
        .topological_sort()
        .expect("graph should have valid execution order")
    {
        initial_bounds.insert(node_name, input_bounds.clone());
    }
    let root = GraphBabDomain::root(initial_bounds, -10.0, 10.0, input_bounds, false).unwrap();
    let domains: Vec<&GraphBabDomain> = vec![&root];
    let layer_names: Vec<String> = vec![];
    let batched = BatchedDomains::from_graph_domains(&domains, &layer_names)
        .expect("should build BatchedDomains");
    (root, batched)
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_crown_batched_rejects_multi_input_sigmoid_1840() {
    let mut graph = GraphNetwork::new();
    let a = LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap();
    let b = LinearLayer::new(arr2(&[[0.5_f32]]), None).unwrap();
    graph.add_node(GraphNode::from_input("a", Layer::Linear(a)));
    graph.add_node(GraphNode::from_input("b", Layer::Linear(b)));
    graph.add_node(GraphNode::new(
        "bad_sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["a".to_string(), "b".to_string()],
    ));
    graph.set_output("bad_sigmoid");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let (root, batched) = build_single_domain(&graph, &input_bounds);
    let domains: Vec<&GraphBabDomain> = vec![&root];

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let err = verifier
        .propagate_crown_with_batched_domains_full(
            &graph,
            &domains,
            &batched,
            &[1.0_f32],
            &NaiveCpuGemmEngine,
        )
        .expect_err("multi-input Sigmoid must fail arity validation");

    let msg = err.to_string();
    assert!(
        msg.contains("expects exactly 1 input"),
        "expected unary-input arity error, got: {}",
        msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_crown_batched_rejects_multi_input_skip_merge_1840() {
    let mut graph = GraphNetwork::new();
    let a = LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap();
    let b = LinearLayer::new(arr2(&[[0.5_f32]]), None).unwrap();
    graph.add_node(GraphNode::from_input("a", Layer::Linear(a)));
    graph.add_node(GraphNode::from_input("b", Layer::Linear(b)));
    graph.add_node(GraphNode::new(
        "bad_skip",
        Layer::SkipMerge(SkipMergeLayer::new()),
        vec!["a".to_string(), "b".to_string()],
    ));
    graph.set_output("bad_skip");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let (root, batched) = build_single_domain(&graph, &input_bounds);
    let domains: Vec<&GraphBabDomain> = vec![&root];

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let err = verifier
        .propagate_crown_with_batched_domains_full(
            &graph,
            &domains,
            &batched,
            &[1.0_f32],
            &NaiveCpuGemmEngine,
        )
        .expect_err("multi-input SkipMerge must fail arity validation");

    let msg = err.to_string();
    assert!(
        msg.contains("SkipMerge node bad_skip expects exactly 1 input"),
        "expected SkipMerge arity error, got: {}",
        msg
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_crown_batched_keeps_valid_unary_sigmoid_path_1840() {
    let mut graph = GraphNetwork::new();
    let proj = LinearLayer::new(arr2(&[[1.25_f32]]), None).unwrap();
    graph.add_node(GraphNode::from_input("proj", Layer::Linear(proj)));
    graph.add_node(GraphNode::new(
        "sigmoid",
        Layer::Sigmoid(SigmoidLayer),
        vec!["proj".to_string()],
    ));
    graph.set_output("sigmoid");

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let (root, batched) = build_single_domain(&graph, &input_bounds);
    let domains: Vec<&GraphBabDomain> = vec![&root];

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let results = verifier
        .propagate_crown_with_batched_domains_full(
            &graph,
            &domains,
            &batched,
            &[1.0_f32],
            &NaiveCpuGemmEngine,
        )
        .expect("valid unary Sigmoid graph should succeed");
    let (actual, _) = results[0]
        .as_ref()
        .expect("single domain should produce a bound");
    let actual_lower = actual.flatten().lower()[[0]];
    let actual_upper = actual.flatten().upper()[[0]];
    assert!(
        actual_lower.is_finite(),
        "lower bound should remain finite, got {}",
        actual_lower,
    );
    assert!(
        actual_upper.is_finite(),
        "upper bound should remain finite, got {}",
        actual_upper,
    );
    assert!(
        actual_lower <= actual_upper + 1e-6,
        "output bounds should be ordered: lower {} > upper {}",
        actual_lower,
        actual_upper
    );
}
