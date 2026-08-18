// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DAG alpha-CROWN engine parity regressions.
//!
//! `#3499` threads a `GemmEngine` into the ECAPA stage-local alpha helper via
//! `collect_alpha_crown_bounds_dag_with_engine(...)`. These tests lock down the
//! lower-level contract directly: using an explicit GEMM engine must preserve
//! the collected DAG alpha bounds and alpha state relative to `engine=None`,
//! including the deadline-carrying path used by the avoice runner.

use crate::bounds::{AlphaCrownConfig, GradientMethod, GraphAlphaState};
use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_test_utils::assert_bounded_tensor_close;
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn build_diamond_dag() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let b1 = arr1(&[0.1_f32, -0.1]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid Linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2a = arr2(&[[0.8_f32, -0.3], [-0.2, 0.6]]);
    graph.add_node(GraphNode::new(
        "linear2a",
        Layer::Linear(LinearLayer::new(w2a, None).expect("valid Linear2a")),
        vec!["relu1".to_string()],
    ));

    let w2b = arr2(&[[-0.4_f32, 0.7], [0.5, -0.1]]);
    graph.add_node(GraphNode::new(
        "linear2b",
        Layer::Linear(LinearLayer::new(w2b, None).expect("valid Linear2b")),
        vec!["relu1".to_string()],
    ));

    graph.add_node(GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["linear2a".to_string(), "linear2b".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["add".to_string()],
    ));
    graph.set_output("relu2");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.5_f32),
    )
    .expect("valid input bounds");

    (graph, input)
}

fn build_relu_chain_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let b1 = arr1(&[0.0_f32, 0.1, -0.1]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid Linear1")),
    ));

    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.5_f32, -0.3, 0.8], [0.2, 0.6, -0.4]]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, None).expect("valid Linear2")),
        vec!["relu1".to_string()],
    ));
    graph.set_output("linear2");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -0.5_f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.5_f32),
    )
    .expect("valid input bounds");

    (graph, input)
}

fn build_two_relu_chain_graph() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();

    let w1 = arr2(&[[1.0_f32, -0.6], [0.4, 0.8], [-0.9, 0.5]]);
    let b1 = arr1(&[0.0_f32, 0.05, -0.15]);
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).expect("valid Linear1")),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));

    let w2 = arr2(&[[0.7_f32, -0.2, 0.5], [-0.4, 0.9, -0.3], [0.3, 0.4, 0.8]]);
    let b2 = arr1(&[0.1_f32, -0.2, 0.05]);
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).expect("valid Linear2")),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));

    let w3 = arr2(&[[0.6_f32, -0.7, 0.9]]);
    let b3 = arr1(&[0.15_f32]);
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, Some(b3)).expect("valid Linear3")),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[2]), 1.0_f32),
    )
    .expect("valid input bounds");

    (graph, input)
}

fn alpha_engine_config(deadline_secs: Option<u64>) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations: 2,
        spsa_samples: 1,
        adaptive_skip: false,
        fix_interm_bounds: false,
        deadline: deadline_secs.map(|secs| Instant::now() + Duration::from_secs(secs)),
        ..AlphaCrownConfig::default()
    }
}
fn non_spsa_collect_config(method: GradientMethod) -> AlphaCrownConfig {
    AlphaCrownConfig {
        iterations: 6,
        gradient_method: method,
        fix_interm_bounds: false,
        adaptive_skip: false,
        adaptive_skip_pilot: false,
        spsa_samples: 1,
        ..AlphaCrownConfig::default()
    }
}
fn diamond_forward(x0: f32, x1: f32) -> [f32; 2] {
    let relu1_0 = (1.0 * x0 - 0.5 * x1 + 0.1).max(0.0);
    let relu1_1 = (0.5 * x0 + x1 - 0.1).max(0.0);
    let branch_a_0 = 0.8 * relu1_0 - 0.3 * relu1_1;
    let branch_a_1 = -0.2 * relu1_0 + 0.6 * relu1_1;
    let branch_b_0 = -0.4 * relu1_0 + 0.7 * relu1_1;
    let branch_b_1 = 0.5 * relu1_0 - 0.1 * relu1_1;
    [
        (branch_a_0 + branch_b_0).max(0.0),
        (branch_a_1 + branch_b_1).max(0.0),
    ]
}
fn assert_diamond_output_sound(bounds: &BoundedTensor, label: &str) {
    for i in 0..=10 {
        for j in 0..=10 {
            let x0 = -0.5 + i as f32 * 0.1;
            let x1 = -0.5 + j as f32 * 0.1;
            let output = diamond_forward(x0, x1);
            for (dim, actual) in output.into_iter().enumerate() {
                assert!(
                    actual >= bounds.lower()[[dim]] - 1e-5
                        && actual <= bounds.upper()[[dim]] + 1e-5,
                    "{label}: output[{dim}]={actual} outside [{}, {}] at ({x0}, {x1})",
                    bounds.lower()[[dim]],
                    bounds.upper()[[dim]],
                );
            }
        }
    }
}

fn assert_node_bounds_parity(
    baseline: &HashMap<String, BoundedTensor>,
    with_engine: &HashMap<String, BoundedTensor>,
    tol: f32,
    label: &str,
) {
    assert_eq!(
        baseline.len(),
        with_engine.len(),
        "{label}: node-count mismatch: baseline={}, with_engine={}",
        baseline.len(),
        with_engine.len()
    );

    for (name, baseline_bounds) in baseline {
        let engine_bounds = with_engine
            .get(name)
            .unwrap_or_else(|| panic!("{label}: missing node '{name}' in engine result"));
        assert_bounded_tensor_close(
            engine_bounds,
            baseline_bounds,
            tol,
            &format!("{label} node '{name}'"),
        );
    }
}

fn assert_node_bounds_finite(bounds: &HashMap<String, BoundedTensor>, label: &str) {
    for (name, node_bounds) in bounds {
        assert_bounds_finite(node_bounds, &format!("{label} node '{name}'"));
    }
}

fn assert_alpha_state_parity(
    baseline: &GraphAlphaState,
    with_engine: &GraphAlphaState,
    tol: f32,
    label: &str,
) {
    let baseline_relu_nodes: Vec<&str> = baseline.relu_nodes().collect();
    let engine_relu_nodes: Vec<&str> = with_engine.relu_nodes().collect();
    assert_eq!(
        baseline_relu_nodes, engine_relu_nodes,
        "{label}: ReLU alpha-state nodes diverged"
    );
    assert_eq!(
        baseline.num_unstable(),
        with_engine.num_unstable(),
        "{label}: unstable-neuron counts diverged"
    );

    for node_name in baseline_relu_nodes {
        let baseline_alpha = baseline
            .alpha(node_name)
            .unwrap_or_else(|| panic!("{label}: missing lower alpha for '{node_name}'"));
        let engine_alpha = with_engine
            .alpha(node_name)
            .unwrap_or_else(|| panic!("{label}: missing engine lower alpha for '{node_name}'"));
        assert_eq!(
            baseline_alpha.len(),
            engine_alpha.len(),
            "{label}: lower alpha length mismatch for '{node_name}'"
        );
        for (idx, (&actual, &expected)) in
            engine_alpha.iter().zip(baseline_alpha.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() <= tol,
                "{label}: lower alpha mismatch for '{node_name}' index {idx}: actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }

        let baseline_upper = baseline
            .alpha_upper(node_name)
            .unwrap_or_else(|| panic!("{label}: missing upper alpha for '{node_name}'"));
        let engine_upper = with_engine
            .alpha_upper(node_name)
            .unwrap_or_else(|| panic!("{label}: missing engine upper alpha for '{node_name}'"));
        assert_eq!(
            baseline_upper.len(),
            engine_upper.len(),
            "{label}: upper alpha length mismatch for '{node_name}'"
        );
        for (idx, (&actual, &expected)) in
            engine_upper.iter().zip(baseline_upper.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() <= tol,
                "{label}: upper alpha mismatch for '{node_name}' index {idx}: actual={actual}, expected={expected}, diff={}",
                (actual - expected).abs()
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_engine_parity_diamond_3499() {
    let (graph, input) = build_diamond_dag();

    let baseline_config = alpha_engine_config(None);
    let (baseline_bounds, baseline_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &baseline_config)
        .expect("baseline DAG alpha-CROWN should succeed");

    let engine_config = alpha_engine_config(None);
    let engine = CountingGemmEngine::new();
    let (with_engine_bounds, with_engine_state) = graph
        .collect_alpha_crown_bounds_dag_with_engine(&input, &engine_config, Some(&engine))
        .expect("engine DAG alpha-CROWN should succeed");

    assert_node_bounds_finite(
        &with_engine_bounds,
        "#3499 DAG alpha-CROWN with engine output",
    );
    assert_node_bounds_parity(
        &baseline_bounds,
        &with_engine_bounds,
        1e-4,
        "#3499 DAG alpha-CROWN engine parity",
    );
    assert_alpha_state_parity(
        &baseline_state,
        &with_engine_state,
        1e-4,
        "#3499 DAG alpha-CROWN engine parity",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3499 regression: collect_alpha_crown_bounds_dag_with_engine should hit GemmEngine"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_deadline_engine_parity_diamond_3499() {
    let (graph, input) = build_diamond_dag();

    let baseline_config = alpha_engine_config(Some(30));
    let (baseline_bounds, baseline_state) = graph
        .collect_alpha_crown_bounds_dag(&input, &baseline_config)
        .expect("baseline DAG alpha-CROWN with deadline should succeed");

    let engine_config = alpha_engine_config(Some(30));
    let engine = CountingGemmEngine::new();
    let (with_engine_bounds, with_engine_state) = graph
        .collect_alpha_crown_bounds_dag_with_engine(&input, &engine_config, Some(&engine))
        .expect("engine DAG alpha-CROWN with deadline should succeed");

    assert_node_bounds_finite(
        &with_engine_bounds,
        "#3499 deadline DAG alpha-CROWN with engine output",
    );
    assert_node_bounds_parity(
        &baseline_bounds,
        &with_engine_bounds,
        1e-4,
        "#3499 deadline DAG alpha-CROWN engine parity",
    );
    assert_alpha_state_parity(
        &baseline_state,
        &with_engine_state,
        1e-4,
        "#3499 deadline DAG alpha-CROWN engine parity",
    );
    assert_eq!(
        engine.gemm_calls(),
        0,
        "finite DAG alpha-CROWN must decline opaque dense GemmEngine kernels until they expose cooperative deadline polling"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_alpha_crown_sound_with_engine_matches_baseline_3772() {
    let (graph, input) = build_relu_chain_graph();

    let baseline = graph
        .propagate_alpha_crown_sound(&input)
        .expect("baseline graph alpha-CROWN sound path should succeed");

    let engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_alpha_crown_sound_with_engine(&input, Some(&engine))
        .expect("engine-aware graph alpha-CROWN sound path should succeed");

    assert_bounds_finite(
        &with_engine,
        "#3772 graph alpha-CROWN sound with engine output",
    );
    assert_bounded_tensor_close(
        &with_engine,
        &baseline,
        1e-6,
        "#3772 graph alpha-CROWN sound wrapper parity",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3772 regression: graph propagate_alpha_crown_sound_with_engine should hit GemmEngine"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_matches_sequential_non_spsa_methods_4036() {
    let (graph, input) = build_two_relu_chain_graph();
    for (method, label) in [
        (GradientMethod::FiniteDifferences, "FiniteDifferences"),
        (GradientMethod::Analytic, "Analytic"),
        (GradientMethod::AnalyticChain, "AnalyticChain"),
    ] {
        let config = non_spsa_collect_config(method);
        let (collected_bounds, alpha_state) = graph
            .collect_alpha_crown_bounds_dag(&input, &config)
            .unwrap_or_else(|_| panic!("#4036 {label} sequential alpha collection should succeed"));
        let sequential_bounds = graph
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap_or_else(|_| panic!("#4036 {label} sequential alpha-CROWN should succeed"));

        let collected_output = collected_bounds
            .get(graph.output_name())
            .unwrap_or_else(|| {
                panic!("#4036 {label} collected bounds should include the output node")
            });
        assert_bounded_tensor_close(
            collected_output,
            &sequential_bounds,
            1e-3,
            &format!("#4036 {label} sequential collection should stay near direct alpha-CROWN"),
        );
        let collected_width = collected_output.upper()[[0]] - collected_output.lower()[[0]];
        let sequential_width = sequential_bounds.upper()[[0]] - sequential_bounds.lower()[[0]];
        assert!(
            collected_width <= sequential_width + 1e-5,
            "#4036 {label} collected sequential output should not be looser than direct alpha-CROWN: collected={collected_width}, direct={sequential_width}"
        );

        let alpha_nodes: Vec<&str> = alpha_state.relu_nodes().collect();
        assert_eq!(
            alpha_nodes,
            vec!["relu1", "relu2"],
            "#4036 {label} returned warm-start state should preserve sequential ReLU order"
        );
        assert!(
            alpha_state.num_unstable() > 0,
            "#4036 {label} regression graph should expose optimizable ReLU alpha state"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_alpha_crown_bounds_dag_true_dag_non_spsa_methods_stay_sound_4036() {
    let (graph, input) = build_diamond_dag();

    for (method, label) in [
        (GradientMethod::FiniteDifferences, "FiniteDifferences"),
        (GradientMethod::Analytic, "Analytic"),
        (GradientMethod::AnalyticChain, "AnalyticChain"),
    ] {
        let config = non_spsa_collect_config(method);
        let (collected_bounds, alpha_state) = graph
            .collect_alpha_crown_bounds_dag(&input, &config)
            .unwrap_or_else(|_| panic!("#4036 {label} true DAG collection should succeed"));
        let direct_bounds = graph
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap_or_else(|_| panic!("#4036 {label} true DAG direct alpha-CROWN should succeed"));
        let collected_output = collected_bounds
            .get(graph.output_name())
            .unwrap_or_else(|| panic!("#4036 {label} true DAG output node should be collected"));

        assert_bounds_finite(collected_output, &format!("#4036 {label} true DAG output"));
        assert_diamond_output_sound(
            collected_output,
            &format!("#4036 {label} true DAG soundness"),
        );

        let collected_width: f32 = collected_output
            .upper()
            .iter()
            .zip(collected_output.lower().iter())
            .map(|(upper, lower)| upper - lower)
            .sum();
        let direct_width: f32 = direct_bounds
            .upper()
            .iter()
            .zip(direct_bounds.lower().iter())
            .map(|(upper, lower)| upper - lower)
            .sum();
        assert!(
            collected_width <= direct_width + 1e-5,
            "#4036 {label} true DAG collected output should not be looser than direct alpha-CROWN: collected={collected_width}, direct={direct_width}"
        );
        assert!(
            alpha_state.num_unstable() > 0,
            "#4036 {label} diamond DAG should expose optimizable ReLU alpha state"
        );
    }
}
