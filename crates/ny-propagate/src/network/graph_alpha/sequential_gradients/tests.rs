// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::{Layer, LinearLayer, ReLULayer};
use crate::network::core::GraphNode;
use crate::NETWORK_INPUT;
use ndarray::{arr1, arr2};

type ChainRuleFixture = (
    GraphNetwork,
    BoundedTensor,
    HashMap<String, BoundedTensor>,
    AlphaState,
    Vec<String>,
    HashMap<String, usize>,
    Vec<Array1<f32>>,
);

fn build_error_fixture() -> (GraphNetwork, BoundedTensor, AlphaState, Vec<String>) {
    // _input -> linear -> relu, but tests provide empty node_bounds to trigger
    // a deterministic CROWN error after alpha perturbation is applied.
    let mut graph = GraphNetwork::new();
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("linear should construct");
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear".to_string()],
    ));
    graph.set_output("relu");

    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("input bounds should construct");
    let pre_activation =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("pre-activation bounds should construct");
    let mut alpha_state =
        AlphaState::from_preactivation_bounds(&[pre_activation], &[0]).expect("alpha init");
    alpha_state.alphas[0][0] = 0.5;
    alpha_state.unstable_mask[0][0] = true;

    let exec_order = graph
        .topological_sort()
        .expect("topological_sort should succeed");
    (graph, input, alpha_state, exec_order)
}

fn build_chain_rule_fixture() -> ChainRuleFixture {
    let mut graph = GraphNetwork::new();
    let linear1 = LinearLayer::new(arr2(&[[1.0_f32, 0.0], [0.0, 1.0]]), None)
        .expect("linear1 should construct");
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear1".to_string()],
    ));

    let linear2 = LinearLayer::new(
        arr2(&[[1.0_f32, -1.0], [0.5, 0.5]]),
        Some(arr1(&[0.0_f32, -0.25])),
    )
    .expect("linear2 should construct");
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer::new()),
        vec!["linear2".to_string()],
    ));
    graph.set_output("relu2");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("input bounds should construct");

    let mut node_bounds = graph
        .collect_node_bounds(&input)
        .expect("node bounds should collect");
    node_bounds.insert(NETWORK_INPUT.to_string(), input.clone());

    let pre_activation_bounds = vec![
        node_bounds
            .get("linear1")
            .expect("linear1 bounds should exist")
            .clone(),
        node_bounds
            .get("linear2")
            .expect("linear2 bounds should exist")
            .clone(),
    ];
    let alpha_state = AlphaState::from_preactivation_bounds(&pre_activation_bounds, &[0, 1])
        .expect("alpha state should initialize");
    let exec_order = graph
        .topological_sort()
        .expect("topological_sort should succeed");
    let relu_name_to_idx = HashMap::from([
        (String::from("relu1"), 0_usize),
        (String::from("relu2"), 1_usize),
    ]);
    let analytic_gradients = alpha_state
        .alphas
        .iter()
        .enumerate()
        .map(|(relu_idx, alpha)| Array1::from_elem(alpha.len(), 17.0 + relu_idx as f32))
        .collect();

    (
        graph,
        input,
        node_bounds,
        alpha_state,
        exec_order,
        relu_name_to_idx,
        analytic_gradients,
    )
}

#[test]
fn spsa_restores_alphas_when_crown_fails() {
    let (graph, input, mut alpha_state, exec_order) = build_error_fixture();
    let original = alpha_state.alphas.clone();
    let config = AlphaCrownConfig {
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 1,
        ..AlphaCrownConfig::default()
    };
    let node_bounds = HashMap::new();
    let relu_name_to_idx = HashMap::from([(String::from("relu"), 0_usize)]);

    let result = spsa_gradients(
        &graph,
        &config,
        &mut alpha_state,
        &input,
        &node_bounds,
        &exec_order,
        1,
        &relu_name_to_idx,
        None,
    );

    assert!(
        result.is_err(),
        "expected propagation error with missing bounds"
    );
    assert_eq!(
        alpha_state.alphas, original,
        "SPSA must restore alpha state on early error return"
    );
}

#[test]
fn finite_difference_restores_alphas_when_crown_fails() {
    let (graph, input, mut alpha_state, exec_order) = build_error_fixture();
    let original = alpha_state.alphas.clone();
    let node_bounds = HashMap::new();
    let relu_name_to_idx = HashMap::from([(String::from("relu"), 0_usize)]);

    let result = finite_difference_gradients(
        &graph,
        &mut alpha_state,
        &input,
        &node_bounds,
        &exec_order,
        1,
        &relu_name_to_idx,
        None,
    );

    assert!(
        result.is_err(),
        "expected propagation error with missing bounds"
    );
    assert_eq!(
        alpha_state.alphas, original,
        "finite differences must restore alpha state on early error return"
    );
}

#[test]
fn analytic_chain_valid_order_does_not_silently_fallback_to_local_gradients() {
    reset_sequential_gradient_diagnostics();
    let (graph, input, node_bounds, alpha_state, exec_order, relu_name_to_idx, local_grads) =
        build_chain_rule_fixture();

    let chain_grads = analytic_chain_gradients(
        &graph,
        &alpha_state,
        &input,
        &node_bounds,
        &exec_order,
        2,
        &relu_name_to_idx,
        None,
        &local_grads,
        0,
    )
    .expect("analytic chain should succeed on valid contiguous ReLU order");

    assert!(
        chain_grads
            .iter()
            .zip(local_grads.iter())
            .any(|(chain, local)| chain != local),
        "valid AnalyticChain execution must not fall back to the caller-provided local \
         gradients; identical output indicates the silent fallback that #2550 guards"
    );
    assert_eq!(
        sequential_chain_fallbacks(),
        0,
        "successful AnalyticChain execution should not increment fallback diagnostics"
    );
}

#[test]
fn analytic_chain_invalid_order_falls_back_to_local_gradients() {
    reset_sequential_gradient_diagnostics();
    let (graph, input, node_bounds, alpha_state, exec_order, _relu_name_to_idx, local_grads) =
        build_chain_rule_fixture();
    let invalid_order = HashMap::from([
        (String::from("relu1"), 0_usize),
        (String::from("relu2"), 3_usize),
    ]);

    let chain_grads = analytic_chain_gradients(
        &graph,
        &alpha_state,
        &input,
        &node_bounds,
        &exec_order,
        2,
        &invalid_order,
        None,
        &local_grads,
        0,
    )
    .expect("invalid order should return the documented local-gradient fallback");

    assert_eq!(
        chain_grads, local_grads,
        "non-contiguous ReLU indices must use the explicit local-gradient fallback"
    );
    assert_eq!(
        sequential_chain_fallbacks(),
        1,
        "invalid alpha order should record a fallback even after iteration zero (#2544)"
    );
}

#[test]
fn compute_sequential_gradients_analytic_zeroes_non_finite_entries_2544() {
    reset_sequential_gradient_diagnostics();
    let (graph, input, node_bounds, mut alpha_state, exec_order, relu_name_to_idx, _local_grads) =
        build_chain_rule_fixture();
    let config = AlphaCrownConfig {
        gradient_method: GradientMethod::Analytic,
        ..AlphaCrownConfig::default()
    };
    let analytic_gradients = vec![
        arr1(&[f32::NAN, 2.0_f32]),
        arr1(&[f32::INFINITY, f32::NEG_INFINITY]),
    ];

    let gradients = compute_sequential_gradients(
        &graph,
        &config,
        &mut alpha_state,
        &input,
        &node_bounds,
        &exec_order,
        2,
        &relu_name_to_idx,
        None,
        &analytic_gradients,
        3,
    )
    .expect("analytic gradients should sanitize instead of propagating NaN/Inf");

    assert_eq!(gradients[0], arr1(&[0.0_f32, 2.0]));
    assert_eq!(gradients[1], arr1(&[0.0_f32, 0.0]));
    assert_eq!(
        sequential_gradient_sanitized_values(),
        3,
        "all non-finite analytic entries should be counted by diagnostics"
    );
}

#[test]
fn analytic_chain_backward_error_records_fallback_after_iter_zero_2544() {
    reset_sequential_gradient_diagnostics();
    let (graph, input, alpha_state, exec_order) = build_error_fixture();
    let node_bounds = HashMap::new();
    let relu_name_to_idx = HashMap::from([(String::from("relu"), 0_usize)]);
    let local_grads = vec![arr1(&[3.0_f32])];

    let gradients = analytic_chain_gradients(
        &graph,
        &alpha_state,
        &input,
        &node_bounds,
        &exec_order,
        1,
        &relu_name_to_idx,
        None,
        &local_grads,
        4,
    )
    .expect("AnalyticChain should fall back to local gradients on backward errors");

    assert_eq!(gradients, local_grads);
    assert_eq!(
        sequential_chain_fallbacks(),
        1,
        "backward-pass errors must record a fallback on every iteration, not only iter 0 (#2544)"
    );
}
