// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use super::prelude::*;

fn assert_output_bounds_match(label: &str, expected: &BoundedTensor, actual: &BoundedTensor) {
    assert_eq!(
        actual.lower().iter().copied().collect::<Vec<_>>(),
        expected.lower().iter().copied().collect::<Vec<_>>(),
        "{label}: lower output bounds changed"
    );
    assert_eq!(
        actual.upper().iter().copied().collect::<Vec<_>>(),
        expected.upper().iter().copied().collect::<Vec<_>>(),
        "{label}: upper output bounds changed"
    );
}

fn assert_results_match(
    label: &str,
    expected: &crate::beta_crown::BetaCrownResult,
    actual: &crate::beta_crown::BetaCrownResult,
) {
    assert_eq!(
        actual.result, expected.result,
        "{label}: verification status changed"
    );
    assert_eq!(
        actual.domains_explored, expected.domains_explored,
        "{label}: domains_explored changed"
    );
    assert_eq!(
        actual.domains_verified, expected.domains_verified,
        "{label}: domains_verified changed"
    );
    assert_eq!(
        actual.max_depth_reached, expected.max_depth_reached,
        "{label}: max_depth_reached changed"
    );
    assert_eq!(
        actual.cuts_generated, expected.cuts_generated,
        "{label}: cuts_generated changed"
    );

    match (&expected.output_bounds, &actual.output_bounds) {
        (Some(expected_bounds), Some(actual_bounds)) => {
            assert_output_bounds_match(label, expected_bounds, actual_bounds);
        }
        (None, None) => {}
        _ => panic!(
            "{label}: output-bounds presence changed: expected={:?} actual={:?}",
            expected.output_bounds.is_some(),
            actual.output_bounds.is_some()
        ),
    }
}

fn equivalent_deadline(timeout: Duration) -> Option<Instant> {
    Some(Instant::now() + timeout)
}

/// Verify that a short deadline caps the BaB engine at approximately
/// `deadline - now`, not `config.timeout`. Uses a 300s configured timeout
/// with a 100ms deadline; the engine must finish within 2s (generous margin
/// for test overhead). If the deadline were ignored, the 300s timeout would
/// blow past the limit. Part of #4321 acceptance criterion 5.
#[ntest::timeout(5000)]
#[test]
fn test_short_deadline_caps_engine_relu_split_4321() {
    let graph = simple_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid input");
    let objective = vec![1.0_f32];
    let threshold = 0.5_f32;
    let config = BetaCrownConfig {
        timeout: Duration::from_mins(5),
        max_domains: 10_000,
        max_depth: 100,
        batch_size: 1,
        use_alpha_crown: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        ..Default::default()
    };

    let start = Instant::now();
    let short_deadline = Some(Instant::now() + Duration::from_millis(100));
    let _result = BetaCrownVerifier::new(config).verify_graph_relu_split_with_engine_gpu(
        &graph,
        &input,
        &objective,
        threshold,
        None,
        short_deadline,
    );
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(2),
        "Engine should have terminated in <2s with 100ms deadline, took {elapsed:?}"
    );
}

/// A deadline that expires during the alpha bootstrap is a normal verifier
/// timeout, not an API error. This also covers a loaded shared builder where a
/// short configured budget can expire before the first graph node is visited.
#[test]
fn test_expired_deadline_maps_relu_split_bootstrap_to_timeout() {
    let graph = simple_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid input");
    let config = BetaCrownConfig {
        use_alpha_crown: true,
        enable_pgd_attack: false,
        ..Default::default()
    };
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("system uptime exceeds one millisecond");

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_with_engine_gpu(
            &graph,
            &input,
            &[1.0_f32],
            0.5,
            None,
            Some(expired),
        )
        .expect("deadline expiry should produce a verifier status");

    assert_eq!(result.result, BabVerificationStatus::Timeout);
}

fn simple_relu_graph() -> GraphNetwork {
    let linear1 =
        LinearLayer::new(arr2(&[[1.0_f32, -1.0], [-1.0, 1.0]]), None).expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0_f32, 1.0]]), None).expect("valid linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

fn anti_correlated_multi_objective_graph() -> (GraphNetwork, BoundedTensor) {
    let linear1 = LinearLayer::new(Array2::eye(1), None).expect("valid linear1");
    let linear2 = LinearLayer::new(
        arr2(&[[1.0_f32], [-1.0_f32]]),
        Some(arr1(&[0.5_f32, 0.5_f32])),
    )
    .expect("valid linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");
    (graph, input)
}

fn identity_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity linear")),
    ));
    graph.set_output("out");
    graph
}

fn branchy_sequential_config() -> BetaCrownConfig {
    BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        timeout: Duration::from_secs(5),
        max_domains: 2,
        max_depth: 10,
        batch_size: 1,
        ..Default::default()
    }
}

fn branchy_graph_config() -> BetaCrownConfig {
    BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        timeout: Duration::from_secs(5),
        max_domains: 2,
        max_depth: 10,
        batch_size: 1,
        ..Default::default()
    }
}

fn input_split_config() -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        enable_pgd_attack: false,
        input_split_ibp_enhancement: false,
        max_domains: 64,
        max_depth: 1,
        batch_size: 4,
        timeout: Duration::from_secs(5),
        reorder_bab: false,
        ..Default::default()
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_with_engine_none_matches_verify_4321() {
    let network = simple_network();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid input");
    let threshold = 0.5_f32;
    let config = branchy_sequential_config();

    let expected = BetaCrownVerifier::new(config.clone())
        .verify(&network, &input, threshold)
        .expect("legacy sequential verify should succeed");
    let actual = BetaCrownVerifier::new(config)
        .verify_with_engine(&network, &input, threshold, None, None)
        .expect("explicit None deadline sequential verify should succeed");

    assert_results_match("sequential verify", &expected, &actual);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_relu_split_none_matches_wrapper_4321() {
    let graph = simple_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid input");
    let objective = vec![1.0_f32];
    let threshold = 0.5_f32;
    let config = branchy_graph_config();

    let expected = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split(&graph, &input, &objective, threshold)
        .expect("legacy graph relu split should succeed");
    let actual = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_with_engine_gpu(&graph, &input, &objective, threshold, None, None)
        .expect("explicit None deadline graph relu split should succeed");

    assert_results_match("graph relu split", &expected, &actual);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_input_split_none_matches_wrapper_4321() {
    let graph = identity_graph();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid input");
    let objective = vec![1.0_f32];
    let threshold = 0.4_f32;
    let config = input_split_config();

    let expected = BetaCrownVerifier::new(config.clone())
        .verify_graph_input_split(&graph, &input, &objective, threshold)
        .expect("legacy graph input split should succeed");
    let actual = BetaCrownVerifier::new(config)
        .verify_graph_input_split_with_engine(&graph, &input, &objective, threshold, None, None)
        .expect("explicit None deadline graph input split should succeed");

    assert_results_match("graph input split", &expected, &actual);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_multi_objective_disjunctive_none_matches_wrapper_4321() {
    let (graph, input) = anti_correlated_multi_objective_graph();
    let objectives = vec![vec![-1.0_f32, 0.0_f32], vec![0.0_f32, -1.0_f32]];
    let thresholds = vec![-0.55_f32, -0.55_f32];
    let config = BetaCrownConfig {
        use_alpha_crown: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        timeout: Duration::from_secs(5),
        max_domains: 100,
        max_depth: 10,
        batch_size: 1,
        ..Default::default()
    };

    let expected = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
        .expect("legacy multi-objective relu split should succeed");
    let actual = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("explicit None deadline multi-objective relu split should succeed");

    assert_results_match("multi-objective relu split disjunctive", &expected, &actual);
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_multi_objective_conjunctive_none_matches_equivalent_deadline_4321() {
    let (graph, input) = anti_correlated_multi_objective_graph();
    let objectives = vec![vec![-1.0_f32, 0.0_f32], vec![0.0_f32, -1.0_f32]];
    let thresholds = vec![-0.55_f32, -0.55_f32];
    let config = BetaCrownConfig {
        use_alpha_crown: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        timeout: Duration::from_secs(5),
        max_domains: 100,
        max_depth: 10,
        batch_size: 1,
        ..Default::default()
    };

    let none_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split_multi_objective_conjunctive_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("None-deadline conjunctive multi-objective relu split should succeed");
    let equivalent_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split_multi_objective_conjunctive_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            equivalent_deadline(config.timeout),
        )
        .expect(
            "equivalent explicit deadline conjunctive multi-objective relu split should succeed",
        );

    assert_results_match(
        "multi-objective relu split conjunctive",
        &equivalent_result,
        &none_result,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_input_split_multi_objective_none_matches_equivalent_deadline_4321() {
    let graph = identity_graph();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid input");
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = vec![0.4_f32, 0.4_f32];
    let config = input_split_config();

    let none_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("None-deadline multi-objective input split should succeed");
    let equivalent_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            equivalent_deadline(config.timeout),
        )
        .expect("equivalent explicit deadline multi-objective input split should succeed");

    assert_results_match(
        "multi-objective input split conjunctive",
        &equivalent_result,
        &none_result,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_input_split_multi_clause_none_matches_equivalent_deadline_4321() {
    let graph = identity_graph();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid input");
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = vec![0.4_f32, 0.4_f32];
    let clause_sizes = vec![1usize, 1usize];
    let config = input_split_config();

    let none_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_input_split_multi_clause_disjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            None,
        )
        .expect("None-deadline grouped input split should succeed");
    let equivalent_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_input_split_multi_clause_disjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            equivalent_deadline(config.timeout),
        )
        .expect("equivalent explicit deadline grouped input split should succeed");

    assert_results_match(
        "multi-clause disjunctive input split",
        &equivalent_result,
        &none_result,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_none_matches_equivalent_deadline_4321() {
    let graph = simple_relu_graph();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid input");
    let objective = vec![1.0_f32];
    let threshold = 0.5_f32;
    let config = branchy_graph_config();

    let none_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .expect("None-deadline GPU BaB should succeed");
    let equivalent_result = BetaCrownVerifier::new(config.clone())
        .verify_graph_gpu_domain_list(
            &graph,
            &input,
            &objective,
            threshold,
            None,
            equivalent_deadline(config.timeout),
        )
        .expect("equivalent explicit deadline GPU BaB should succeed");

    assert_results_match("gpu domain-list BaB", &equivalent_result, &none_result);
}
