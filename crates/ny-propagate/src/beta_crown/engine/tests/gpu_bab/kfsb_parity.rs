// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn graph_gpu_kfsb_intercept_only_fixture_4300() -> (GraphNetwork, BoundedTensor, Vec<f32>, f32) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 1.0, 1.0, 1.0]]), None)
                .expect("linear layer should build"),
        ),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear");

    let input = BoundedTensor::new(
        arr1(&[-0.1, -1.0, -2.0, -3.0]).into_dyn(),
        arr1(&[0.2, 2.0, 1.0, 3.0]).into_dyn(),
    )
    .expect("input bounds should build");

    (graph, input, vec![1.0], 0.5)
}

fn graph_gpu_kfsb_zero_candidates_fixture_4300() -> (GraphNetwork, BoundedTensor, Vec<f32>, f32) {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[1.0, 10.0])))
                .expect("linear layer should build"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0, 1.0]]), None).expect("linear layer should build"),
        ),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        arr1(&[-2.0, -11.0]).into_dyn(),
        arr1(&[0.0, -9.0]).into_dyn(),
    )
    .expect("input bounds should build");

    (graph, input, vec![1.0], 0.5)
}

fn assert_graph_gpu_kfsb_verdict_parity_4300(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objective: &[f32],
    threshold: f32,
    config: BetaCrownConfig,
) {
    let sequential = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split(graph, input, objective, threshold)
        .expect("sequential graph relu-split path should not error");
    let no_engine = BetaCrownVerifier::new(config.clone())
        .verify_graph_gpu_domain_list(graph, input, objective, threshold, None, None)
        .expect("engine=None graph gpu-bab path should not error");
    let engine = NaiveCpuGemmEngine;
    let with_engine = BetaCrownVerifier::new(config)
        .verify_graph_gpu_domain_list(graph, input, objective, threshold, Some(&engine), None)
        .expect("engine-backed graph gpu-bab path should not error");

    assert_eq!(
        std::mem::discriminant(&sequential.result),
        std::mem::discriminant(&no_engine.result),
        "engine=None graph gpu-bab should preserve sequential selector semantics"
    );
    assert_eq!(
        std::mem::discriminant(&no_engine.result),
        std::mem::discriminant(&with_engine.result),
        "engine-backed graph gpu-bab should preserve engine=None selector semantics"
    );
    assert!(
        no_engine.domains_explored >= 1,
        "engine=None graph gpu-bab should execute at least one post-root domain; got {} explored domains",
        no_engine.domains_explored
    );
    assert!(
        with_engine.domains_explored >= 1,
        "engine-backed graph gpu-bab should execute at least one post-root domain; got {} explored domains",
        with_engine.domains_explored
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_graph_gpu_kfsb_intercept_only_engine_parity_4300() {
    let (graph, input, objective, threshold) = graph_gpu_kfsb_intercept_only_fixture_4300();
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        max_domains: 32,
        max_depth: 8,
        batch_size: 4,
        beta_iterations: 0,
        branching_heuristic: BranchingHeuristic::KfsbInterceptOnly,
        fsb_candidates: 1,
        ..Default::default()
    };

    assert_graph_gpu_kfsb_verdict_parity_4300(&graph, &input, &objective, threshold, config);
}

#[ntest::timeout(60000)]
#[test]
fn test_graph_gpu_kfsb_zero_candidates_engine_parity_4300() {
    let (graph, input, objective, threshold) = graph_gpu_kfsb_zero_candidates_fixture_4300();
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(10),
        max_domains: 32,
        max_depth: 8,
        batch_size: 4,
        beta_iterations: 0,
        branching_heuristic: BranchingHeuristic::KfsbInterceptOnly,
        fsb_candidates: 0,
        ..Default::default()
    };

    assert_graph_gpu_kfsb_verdict_parity_4300(&graph, &input, &objective, threshold, config);
}
