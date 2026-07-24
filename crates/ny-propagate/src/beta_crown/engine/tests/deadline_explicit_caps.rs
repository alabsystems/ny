// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use super::prelude::*;

fn expired_deadline() -> Option<Instant> {
    Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    )
}

fn assert_expired_deadline_halts_before_branching(
    label: &str,
    result: &crate::beta_crown::BetaCrownResult,
    elapsed: Duration,
) {
    assert!(
        elapsed < Duration::from_secs(2),
        "{label}: expired deadline should terminate quickly, took {elapsed:?}"
    );
    assert!(
        result.domains_explored <= 1,
        "{label}: expired deadline should stop before branching, explored {} domains with {:?}",
        result.domains_explored,
        result.result
    );
    assert!(
        !matches!(result.result, BabVerificationStatus::Verified),
        "{label}: expired deadline should not report Verified on a branchy fixture"
    );
}

fn multi_objective_timeout_graph() -> (GraphNetwork, BoundedTensor) {
    let w1 = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.5, -0.3, 0.2, 0.1, -0.4, 0.6, -0.1, 0.3, 0.3, 0.2, -0.5, 0.4, -0.1, 0.4, 0.3, -0.6,
        ],
    )
    .expect("w1");
    let w2 = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.4, -0.2, 0.3, -0.1, -0.3, 0.5, 0.1, 0.2, 0.2, -0.4, 0.6, -0.3, -0.1, 0.3, -0.2, 0.5,
        ],
    )
    .expect("w2");
    let w3 = Array2::from_shape_vec((2, 4), vec![0.3, -0.2, 0.4, 0.1, -0.1, 0.5, -0.3, 0.2])
        .expect("w3");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, None).expect("linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, None).expect("linear2")),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, None).expect("linear3")),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0, 1.0]).into_dyn(),
    )
    .expect("valid bounded input");
    (graph, input)
}

fn single_output_identity_graph() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "out",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("identity linear")),
    ));
    graph.set_output("out");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_with_engine_expired_deadline_stops_before_branching_4321() {
    let network = simple_network();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid input");
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        timeout: Duration::from_mins(5),
        max_domains: 10_000,
        max_depth: 100,
        batch_size: 1,
        ..Default::default()
    };

    let start = Instant::now();
    let result = BetaCrownVerifier::new(config)
        .verify_with_engine(&network, &input, 0.5, None, expired_deadline())
        .expect("expired-deadline sequential verify should return cleanly");

    assert_expired_deadline_halts_before_branching(
        "sequential verify_with_engine",
        &result,
        start.elapsed(),
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_multi_objective_expired_deadline_stops_before_branching_4321() {
    let (graph, input) = multi_objective_timeout_graph();
    let objectives = vec![vec![1.0_f32, 0.0_f32], vec![0.0_f32, 1.0_f32]];
    let thresholds = vec![100.0_f32, 100.0_f32];
    let config = BetaCrownConfig {
        use_alpha_crown: false,
        enable_cuts: false,
        enable_pgd_attack: false,
        timeout: Duration::from_mins(5),
        max_domains: 100_000,
        max_depth: 100,
        batch_size: 64,
        ..Default::default()
    };

    let start = Instant::now();
    let disjunctive = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            expired_deadline(),
        )
        .expect("expired-deadline disjunctive multi-objective verify should return cleanly");
    assert_expired_deadline_halts_before_branching(
        "graph multi-objective disjunctive",
        &disjunctive,
        start.elapsed(),
    );

    let start = Instant::now();
    let conjunctive = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_conjunctive_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            expired_deadline(),
        )
        .expect("expired-deadline conjunctive multi-objective verify should return cleanly");
    assert_expired_deadline_halts_before_branching(
        "graph multi-objective conjunctive",
        &conjunctive,
        start.elapsed(),
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_input_split_expired_deadline_stops_before_branching_4321() {
    let graph = single_output_identity_graph();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid bounded input");
    let objectives = vec![vec![1.0_f32], vec![-1.0_f32]];
    let thresholds = vec![0.4_f32, 0.4_f32];
    let clause_sizes = vec![1usize, 1usize];
    let config = BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::InputSplit,
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        enable_relaxed_clip: false,
        input_split_ibp_enhancement: false,
        timeout: Duration::from_mins(5),
        max_domains: 64,
        max_depth: 1,
        batch_size: 4,
        ..Default::default()
    };

    let start = Instant::now();
    let multi_objective = BetaCrownVerifier::new(config.clone())
        .verify_graph_input_split_multi_objective_conjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            expired_deadline(),
        )
        .expect("expired-deadline multi-objective input split should return cleanly");
    assert_expired_deadline_halts_before_branching(
        "graph input split multi-objective",
        &multi_objective,
        start.elapsed(),
    );

    let start = Instant::now();
    let grouped = BetaCrownVerifier::new(config)
        .verify_graph_input_split_multi_clause_disjunctive(
            &graph,
            &input,
            &objectives,
            &thresholds,
            &clause_sizes,
            None,
            expired_deadline(),
        )
        .expect("expired-deadline grouped input split should return cleanly");
    assert_expired_deadline_halts_before_branching(
        "graph input split multi-clause disjunctive",
        &grouped,
        start.elapsed(),
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_gpu_bab_expired_deadline_stops_before_branching_4321() {
    let graph = super::gpu_bab::simple_graph_network();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid input");
    let config = BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        enable_cuts: false,
        timeout: Duration::from_mins(5),
        max_domains: 20,
        max_depth: 5,
        batch_size: 2,
        ..Default::default()
    };

    let start = Instant::now();
    let result = BetaCrownVerifier::new(config)
        .verify_graph_gpu_domain_list(&graph, &input, &[1.0_f32], 0.5, None, expired_deadline())
        .expect("expired-deadline GPU BaB should return cleanly");

    assert_expired_deadline_halts_before_branching("gpu domain-list BaB", &result, start.elapsed());
}
