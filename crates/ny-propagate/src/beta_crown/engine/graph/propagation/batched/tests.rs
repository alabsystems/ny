// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{arr1, arr2, Array1, Array2};
use ny_core::NaiveCpuGemmEngine;

use super::{
    indexed_pending::IndexedPendingLinearBounds, BatchedBackwardContext, BetaCrownVerifier,
};
use crate::batched_domain::{BatchedDomains, CachedLinearBounds};
use crate::beta_crown::{BetaCrownConfig, GraphBabDomain, GraphNeuronConstraint};
use crate::network::{backward_div_to_numerator, DivBackwardResult};
use crate::{
    AddLayer, BoundedTensor, ConcatLayer, DivLayer, GraphNetwork, GraphNode, Layer, LinearBounds,
    LinearLayer, OpaqueSkipLayer, ReLULayer, SigmoidLayer, NETWORK_INPUT,
};

fn build_single_relu_graph_for_batched_mode_tests() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).expect("valid linear2");

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

fn zero_network_input_accumulator(num_outputs: usize, input_dim: usize) -> LinearBounds {
    LinearBounds::new(
        Array2::zeros((num_outputs, input_dim)),
        Array1::zeros(num_outputs),
        Array2::zeros((num_outputs, input_dim)),
        Array1::zeros(num_outputs),
    )
    .expect("existing network-input accumulator")
}

fn assert_network_input_bias_merge(
    input_lb: &LinearBounds,
    expected_shape: &[usize],
    expected_lower_b: Array1<f32>,
    expected_upper_b: Array1<f32>,
    context: &str,
) {
    assert_eq!(
        input_lb.lower_a.shape(),
        expected_shape,
        "{context}: lower_a shape mismatch"
    );
    assert_eq!(
        input_lb.upper_a.shape(),
        expected_shape,
        "{context}: upper_a shape mismatch"
    );
    assert!(
        input_lb.lower_a.iter().all(|&value| value == 0.0),
        "{context}: lower_a should remain zero at NETWORK_INPUT, got {:?}",
        input_lb.lower_a
    );
    assert!(
        input_lb.upper_a.iter().all(|&value| value == 0.0),
        "{context}: upper_a should remain zero at NETWORK_INPUT, got {:?}",
        input_lb.upper_a
    );
    assert_eq!(
        input_lb.lower_b, expected_lower_b,
        "{context}: lower bias should merge without widening"
    );
    assert_eq!(
        input_lb.upper_b, expected_upper_b,
        "{context}: upper bias should merge without widening"
    );
}

fn build_two_output_identity_graph_for_dense_spec_fallback_4403() -> GraphNetwork {
    let output = LinearLayer::new(
        arr2(&[[1.0_f32, 0.0_f32], [0.0_f32, 1.0_f32]]),
        Some(arr1(&[0.0_f32, 0.0_f32])),
    )
    .expect("valid output layer");
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("out", Layer::Linear(output)));
    graph.set_output("out");
    graph
}

fn build_branch_add_graph_for_indexed_pending_4417() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "left",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).expect("left branch"),
        ),
    ));
    graph.add_node(GraphNode::from_input(
        "right",
        Layer::Linear(
            LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).expect("right branch"),
        ),
    ));
    graph.add_node(GraphNode::new(
        "sum",
        Layer::Add(AddLayer),
        vec!["left".to_string(), "right".to_string()],
    ));
    graph.set_output("sum");
    graph
}

fn make_indexed_pending(names: &[&str], n_domains: usize) -> IndexedPendingLinearBounds {
    let mut graph = GraphNetwork::new();
    let mut output_name = None;
    for &name in names {
        if name == NETWORK_INPUT {
            continue;
        }
        graph.add_node(GraphNode::from_input(
            name,
            Layer::Linear(
                LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[0.0_f32]))).expect("linear layer"),
            ),
        ));
        output_name = Some(name);
    }
    graph.set_output(output_name.expect("indexed pending fixture needs at least one node"));
    let plan = graph.dispatch_plan().expect("dispatch plan should build");
    IndexedPendingLinearBounds::new(plan, n_domains)
}

fn assert_flat_bounds_eq_4403(actual: &BoundedTensor, expected: &BoundedTensor, context: &str) {
    let actual = actual.flatten();
    let expected = expected.flatten();
    assert_eq!(
        actual.lower(),
        expected.lower(),
        "{context}: lower bounds mismatch"
    );
    assert_eq!(
        actual.upper(),
        expected.upper(),
        "{context}: upper bounds mismatch"
    );
}

#[test]
fn test_wildcard_dispatch_rejects_multi_input_unary_1840() {
    let layer = Layer::Sigmoid(SigmoidLayer);
    let node_inputs = vec!["left".to_string(), "right".to_string()];
    let lb = LinearBounds::identity(1);
    let constrained_input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let mut node_linear_bounds: HashMap<String, Vec<Option<LinearBounds>>> = HashMap::new();
    let mut input_accumulated = false;

    let err = BetaCrownVerifier::propagate_wildcard_crown_backward_batched(
        "bad_sigmoid",
        &layer,
        &node_inputs,
        lb,
        &constrained_input,
        &HashMap::new(),
        &mut node_linear_bounds,
        &mut input_accumulated,
        0,
        1,
    )
    .expect_err("multi-input unary wildcard dispatch must be rejected");

    let msg = err.to_string();
    assert!(
        msg.contains("expects exactly 1 input"),
        "expected unary-input arity error, got: {}",
        msg
    );
}

#[test]
fn test_wildcard_dispatch_opaque_skip_keeps_multi_input_dependencies_1840() {
    let layer = Layer::OpaqueSkip(OpaqueSkipLayer::new());
    let node_inputs = vec!["left".to_string(), "right".to_string()];
    let lb = LinearBounds::identity(1);
    let constrained_input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let mut node_linear_bounds: HashMap<String, Vec<Option<LinearBounds>>> = HashMap::new();
    let mut input_accumulated = false;

    BetaCrownVerifier::propagate_wildcard_crown_backward_batched(
        "opaque",
        &layer,
        &node_inputs,
        lb,
        &constrained_input,
        &HashMap::new(),
        &mut node_linear_bounds,
        &mut input_accumulated,
        0,
        1,
    )
    .expect("opaque skip wildcard dispatch should succeed");

    for input_name in &node_inputs {
        let bound = node_linear_bounds
            .get(input_name)
            .and_then(|per_domain| per_domain.first())
            .and_then(|opt| opt.as_ref())
            .unwrap_or_else(|| panic!("expected accumulated bounds for input {}", input_name));
        assert!(
            bound.lower_b[0].is_infinite() && bound.lower_b[0].is_sign_negative(),
            "expected -inf lower bias for input {}",
            input_name
        );
        assert!(
            bound.upper_b[0].is_infinite() && bound.upper_b[0].is_sign_positive(),
            "expected +inf upper bias for input {}",
            input_name
        );
    }
    assert!(
        !input_accumulated,
        "opaque skip test should not mark _input as directly accumulated"
    );
}

#[test]
fn test_dispatch_node_backward_add_splits_bias_with_shared_dispatch_1949() {
    let node = GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["_input".to_string(), "residual".to_string()],
    );
    let mut node_lb = LinearBounds::identity(1);
    node_lb.lower_b[[0]] = 2.0;
    node_lb.upper_b[[0]] = 4.0;

    let constrained_input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let constrained_inputs = vec![constrained_input];
    let bounds_caches = [HashMap::new()];
    let beta_states = vec![None];
    let alpha_states = vec![None];
    let mut node_linear_bounds = make_indexed_pending(&["residual"], 1);

    super::backward_core::dispatch_node_backward(
        "add",
        &node,
        vec![Some(node_lb)],
        &constrained_inputs,
        &bounds_caches.iter().collect::<Vec<_>>(),
        &beta_states,
        &alpha_states,
        &mut node_linear_bounds,
        1,
        constrained_inputs[0].len(),
        &NaiveCpuGemmEngine,
        None,
        None,  // mul_binary_alphas
        false, // stack_domains
    )
    .expect("Add node dispatch should succeed via shared backward dispatch");

    let input_lb = node_linear_bounds
        .get_name(NETWORK_INPUT)
        .and_then(|per_domain| per_domain.first())
        .and_then(|opt| opt.as_ref())
        .expect("expected propagated bounds for _input");
    let residual_lb = node_linear_bounds
        .get_name("residual")
        .and_then(|per_domain| per_domain.first())
        .and_then(|opt| opt.as_ref())
        .expect("expected propagated bounds for residual input");

    // After #2617: bias is in separate channel accumulated to NETWORK_INPUT ("_input").
    // Add dispatch pre-extracts incoming bias and passes zero-bias to propagation.
    // Total bias (bias_lower=2.0, bias_upper=4.0) goes via the separate channel.
    //
    // _input IS NETWORK_INPUT, so it receives both the bias channel (2.0, 4.0) AND
    // bounds_a (identity A-matrix, zero bias). Final: lower_b ≈ 2.0, upper_b ≈ 4.0.
    // residual gets only bounds_b (identity A-matrix, zero bias): lower_b = 0.0.
    assert!(
        (input_lb.lower_b[[0]] - 2.0).abs() < 1e-6,
        "input_lb lower_b should be ~2.0 (bias channel + bounds_a), got {}",
        input_lb.lower_b[[0]]
    );
    assert!(
        (input_lb.upper_b[[0]] - 4.0).abs() < 1e-6,
        "input_lb upper_b should be ~4.0 (bias channel + bounds_a), got {}",
        input_lb.upper_b[[0]]
    );
    assert!(
        residual_lb.lower_b[[0]].abs() < 1e-6,
        "residual_lb lower_b should be ~0 (no bias for non-input node), got {}",
        residual_lb.lower_b[[0]]
    );
    assert!(
        residual_lb.upper_b[[0]].abs() < 1e-6,
        "residual_lb upper_b should be ~0 (no bias for non-input node), got {}",
        residual_lb.upper_b[[0]]
    );
    assert!(
        node_linear_bounds.input_accumulated()[0],
        "propagating to _input should mark input_accumulated"
    );
}

#[test]
fn test_dispatch_node_backward_concat_constant_first_input_uses_shared_dispatch_1949() {
    let constant = BoundedTensor::new(arr1(&[0.25]).into_dyn(), arr1(&[0.25]).into_dyn())
        .expect("constant bounded tensor");
    let concat = ConcatLayer::with_constants(0, vec![vec![1], vec![2]], vec![Some(constant), None]);
    let node = GraphNode::new(
        "concat",
        Layer::Concat(concat),
        vec!["const_token".to_string(), "_input".to_string()],
    );
    let node_lb = LinearBounds::identity(3);

    let constrained_input =
        BoundedTensor::new(arr1(&[-1.0, -2.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn())
            .expect("input bounds");
    let constrained_inputs = vec![constrained_input];
    let bounds_caches = [HashMap::new()];
    let beta_states = vec![None];
    let alpha_states = vec![None];
    let mut node_linear_bounds = make_indexed_pending(&["concat"], 1);

    super::backward_core::dispatch_node_backward(
        "concat",
        &node,
        vec![Some(node_lb)],
        &constrained_inputs,
        &bounds_caches.iter().collect::<Vec<_>>(),
        &beta_states,
        &alpha_states,
        &mut node_linear_bounds,
        1,
        constrained_inputs[0].len(),
        &NaiveCpuGemmEngine,
        None,
        None,  // mul_binary_alphas
        false, // stack_domains
    )
    .expect("concat dispatch should not require cache lookup for constant first input");

    assert!(
        node_linear_bounds.get_name("const_token").is_none(),
        "constant concat input should not receive propagated bounds"
    );
    let input_lb = node_linear_bounds
        .get_name(NETWORK_INPUT)
        .and_then(|per_domain| per_domain.first())
        .and_then(|opt| opt.as_ref())
        .expect("expected propagated bounds for dynamic _input");
    assert_eq!(
        input_lb.num_inputs(),
        2,
        "concat split should preserve dynamic input width"
    );
    assert!(
        node_linear_bounds.input_accumulated()[0],
        "propagating concat split to _input should mark input_accumulated"
    );
}

#[test]
fn test_dispatch_node_backward_binary_bias_carrier_uses_network_input_width_4302() {
    let node = GraphNode::new(
        "add",
        Layer::Add(AddLayer),
        vec!["left".to_string(), "right".to_string()],
    );
    let mut node_lb = LinearBounds::identity(2);
    node_lb.lower_b = arr1(&[1.0_f32, -0.5]);
    node_lb.upper_b = arr1(&[1.5_f32, 0.25]);

    let constrained_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -2.0, -3.0, -4.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0, 3.0, 4.0]).into_dyn(),
    )
    .expect("input bounds");
    let constrained_inputs = vec![constrained_input];
    let bounds_caches = [HashMap::new()];
    let beta_states = vec![None];
    let alpha_states = vec![None];
    let mut node_linear_bounds = make_indexed_pending(&["left", "right"], 1);
    node_linear_bounds
        .seed_name(
            NETWORK_INPUT,
            0,
            zero_network_input_accumulator(2, constrained_inputs[0].len()),
        )
        .expect("network-input accumulator should seed");

    super::backward_core::dispatch_node_backward(
        "add",
        &node,
        vec![Some(node_lb)],
        &constrained_inputs,
        &bounds_caches.iter().collect::<Vec<_>>(),
        &beta_states,
        &alpha_states,
        &mut node_linear_bounds,
        1,
        constrained_inputs[0].len(),
        &NaiveCpuGemmEngine,
        None,
        None,
        false, // stack_domains
    )
    .expect("binary dispatch should merge bias carrier at full input width");

    let input_lb = node_linear_bounds
        .get_name(NETWORK_INPUT)
        .and_then(|per_domain| per_domain.first())
        .and_then(|opt| opt.as_ref())
        .expect("expected merged network-input bounds");

    assert_network_input_bias_merge(
        input_lb,
        &[2, 4],
        arr1(&[1.0_f32, -0.5]),
        arr1(&[1.5_f32, 0.25]),
        "binary bias carrier",
    );
}

#[test]
fn test_dispatch_node_backward_concat_bias_carrier_uses_network_input_width_4302() {
    let concat = ConcatLayer::with_input_shapes(0, vec![vec![2], vec![1]]);
    let node = GraphNode::new(
        "concat",
        Layer::Concat(concat),
        vec!["left".to_string(), "right".to_string()],
    );
    let mut node_lb = LinearBounds::identity(3);
    node_lb.lower_b = arr1(&[-0.25_f32, 0.5, 1.0]);
    node_lb.upper_b = arr1(&[0.75_f32, 1.5, 2.0]);

    let constrained_input = BoundedTensor::new(
        arr1(&[-2.0_f32, -1.0, 0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[2.0_f32, 3.0, 4.0, 5.0, 6.0]).into_dyn(),
    )
    .expect("input bounds");
    let constrained_inputs = vec![constrained_input];
    let bounds_caches = [HashMap::new()];
    let beta_states = vec![None];
    let alpha_states = vec![None];
    let mut node_linear_bounds = make_indexed_pending(&["left", "right"], 1);
    node_linear_bounds
        .seed_name(
            NETWORK_INPUT,
            0,
            zero_network_input_accumulator(3, constrained_inputs[0].len()),
        )
        .expect("network-input accumulator should seed");

    super::backward_core::dispatch_node_backward(
        "concat",
        &node,
        vec![Some(node_lb)],
        &constrained_inputs,
        &bounds_caches.iter().collect::<Vec<_>>(),
        &beta_states,
        &alpha_states,
        &mut node_linear_bounds,
        1,
        constrained_inputs[0].len(),
        &NaiveCpuGemmEngine,
        None,
        None,
        false, // stack_domains
    )
    .expect("concat dispatch should merge bias carrier at full input width");

    let input_lb = node_linear_bounds
        .get_name(NETWORK_INPUT)
        .and_then(|per_domain| per_domain.first())
        .and_then(|opt| opt.as_ref())
        .expect("expected merged network-input bounds");

    assert_network_input_bias_merge(
        input_lb,
        &[3, 5],
        arr1(&[-0.25_f32, 0.5, 1.0]),
        arr1(&[0.75_f32, 1.5, 2.0]),
        "concat bias carrier",
    );
}

#[test]
fn test_dispatch_node_backward_div_reuses_positive_denominator_helper_4354() {
    let node = GraphNode::new(
        "div",
        Layer::Div(DivLayer),
        vec!["_input".to_string(), "den".to_string()],
    );
    let node_lb = LinearBounds::new(
        arr2(&[[1.0_f32, -2.0]]),
        arr1(&[0.5_f32]),
        arr2(&[[1.5_f32, -1.0]]),
        arr1(&[0.75_f32]),
    )
    .expect("node linear bounds");

    let constrained_input = BoundedTensor::new(
        arr1(&[-2.0_f32, 1.0]).into_dyn(),
        arr1(&[4.0_f32, 3.0]).into_dyn(),
    )
    .expect("input bounds");
    let denominator_bounds =
        BoundedTensor::new(arr1(&[2.0_f32]).into_dyn(), arr1(&[4.0_f32]).into_dyn())
            .expect("positive denominator bounds");
    let div_output_bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[2.0_f32, 2.0]).into_dyn(),
    )
    .expect("div output bounds");

    let DivBackwardResult::PropagateNumerator(expected) = backward_div_to_numerator(
        &node_lb,
        &constrained_input,
        &denominator_bounds,
        &div_output_bounds,
    )
    .expect("direct positive-denominator Div helper should propagate the numerator") else {
        panic!("direct positive-denominator Div helper unexpectedly concretized the node");
    };

    let constrained_inputs = vec![constrained_input];
    let bounds_caches = [HashMap::from([
        ("den".to_string(), Arc::new(denominator_bounds)),
        ("div".to_string(), Arc::new(div_output_bounds)),
    ])];
    let beta_states = vec![None];
    let alpha_states = vec![None];
    let mut node_linear_bounds = make_indexed_pending(&["div"], 1);

    super::backward_core::dispatch_node_backward(
        "div",
        &node,
        vec![Some(node_lb)],
        &constrained_inputs,
        &bounds_caches.iter().collect::<Vec<_>>(),
        &beta_states,
        &alpha_states,
        &mut node_linear_bounds,
        1,
        constrained_inputs[0].len(),
        &NaiveCpuGemmEngine,
        None,
        None,
        false, // stack_domains
    )
    .expect("positive-denominator Div should use the reciprocal-scaling helper");
    assert!(
        node_linear_bounds.get_name("den").is_none(),
        "Div helper should not propagate backward bounds into the denominator input",
    );
    let input_lb = node_linear_bounds
        .get_name(NETWORK_INPUT)
        .and_then(|per_domain| per_domain.first())
        .and_then(|opt| opt.as_ref())
        .expect("expected propagated bounds for numerator/_input");

    assert_eq!(input_lb.lower_a, expected.lower_a);
    assert_eq!(input_lb.upper_a, expected.upper_a);
    assert_eq!(input_lb.lower_b, expected.lower_b);
    assert_eq!(input_lb.upper_b, expected.upper_b);
    assert_eq!(input_lb.lower_a_err, expected.lower_a_err);
    assert_eq!(input_lb.upper_a_err, expected.upper_a_err);
    assert!(
        node_linear_bounds.input_accumulated()[0],
        "propagating Div numerator to _input should mark input_accumulated",
    );
}

#[test]
fn test_batched_reverse_traversal_merge_graph_uses_indexed_pending_4417() {
    let graph = build_branch_add_graph_for_indexed_pending_4417();
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid input");
    let initial_bounds = graph
        .collect_node_bounds(&input)
        .expect("graph bounds should collect");
    let root = GraphBabDomain::root(initial_bounds, -10.0, 10.0, &input, false)
        .expect("root domain should build");
    let domains = vec![&root];
    let layer_names = vec!["left".to_string(), "right".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("valid context");
    let objective = vec![1.0_f32];

    let results = BetaCrownVerifier::new(BetaCrownConfig::default())
        .propagate_crown_batched_with_context(&graph, &ctx, &objective, &NaiveCpuGemmEngine)
        .expect("batched propagation should succeed");
    let output = results[0].0.flatten();

    assert!(
        (output.lower()[[0]] - -2.0_f32).abs() < 1e-6,
        "merge graph lower bound should accumulate both branches at network input, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 2.0_f32).abs() < 1e-6,
        "merge graph upper bound should accumulate both branches at network input, got {}",
        output.upper()[[0]]
    );
}

#[test]
fn test_batched_standard_mode_ignores_warm_start_cache_1813() {
    let graph = build_single_relu_graph_for_batched_mode_tests();
    let input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).expect("valid input");
    let initial_bounds = graph
        .collect_node_bounds(&input)
        .expect("graph bounds should collect");
    let root = GraphBabDomain::root(initial_bounds, -10.0, 10.0, &input, false)
        .expect("root domain with finite bounds should not fail");

    let mut child = root
        .with_constraint(
            &graph,
            GraphNeuronConstraint {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                is_active: true,
                score: 0.0,
            },
            false,
        )
        .expect("with_constraint should not fail on valid test domain")
        .expect("active child should be feasible");

    // Inject cached lA at the branch point. Standard mode must ignore this cache.
    let mut cached = HashMap::new();
    cached.insert("relu1".to_string(), LinearBounds::identity(1));
    child.cached_la = Some(Arc::new(CachedLinearBounds::from_linear_bounds_map(cached)));

    let domains = vec![&child];
    let layer_names = vec!["relu1".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("valid context");
    let objective = vec![1.0_f32];

    let warm = BetaCrownVerifier::new(BetaCrownConfig {
        enable_la_warm_start: true,
        ..Default::default()
    })
    .propagate_crown_batched_with_context(&graph, &ctx, &objective, &NaiveCpuGemmEngine)
    .expect("standard batched path (warm-start enabled) should succeed");

    let cold = BetaCrownVerifier::new(BetaCrownConfig {
        enable_la_warm_start: false,
        ..Default::default()
    })
    .propagate_crown_batched_with_context(&graph, &ctx, &objective, &NaiveCpuGemmEngine)
    .expect("standard batched path (warm-start disabled) should succeed");

    assert_eq!(warm.len(), 1, "expected one domain result");
    assert_eq!(cold.len(), 1, "expected one domain result");

    let warm_output = warm[0].0.flatten();
    let cold_output = cold[0].0.flatten();
    let warm_lb = warm_output.lower()[[0]];
    let warm_ub = warm_output.upper()[[0]];
    let cold_lb = cold_output.lower()[[0]];
    let cold_ub = cold_output.upper()[[0]];

    assert!(
        (warm_lb - cold_lb).abs() < 1e-6 && (warm_ub - cold_ub).abs() < 1e-6,
        "standard mode should ignore warm-start cache/config. warm=[{}, {}], cold=[{}, {}]",
        warm_lb,
        warm_ub,
        cold_lb,
        cold_ub
    );
}

#[test]
fn test_batched_with_la_capture_gates_intermediate_storage_1813() {
    let graph = build_single_relu_graph_for_batched_mode_tests();
    let input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).expect("valid input");
    let initial_bounds = graph
        .collect_node_bounds(&input)
        .expect("graph bounds should collect");
    let root = GraphBabDomain::root(initial_bounds, -10.0, 10.0, &input, false)
        .expect("root domain with finite bounds should not fail");

    let domains = vec![&root];
    let layer_names = vec!["relu1".to_string()];
    let batched =
        BatchedDomains::from_graph_domains(&domains, &layer_names).expect("batched domains");
    let ctx = BatchedBackwardContext::from_domains(&domains, &batched).expect("valid context");
    let objective = vec![1.0_f32];

    let no_capture = BetaCrownVerifier::new(BetaCrownConfig {
        enable_la_warm_start: false,
        ..Default::default()
    })
    .propagate_crown_batched_with_context_capture_la(&graph, &ctx, &objective, &NaiveCpuGemmEngine)
    .expect("capture-la path should succeed when warm-start disabled");
    assert!(
        no_capture.intermediate_la.is_none(),
        "capture-la path must skip intermediate storage when warm-start is disabled"
    );

    let with_capture = BetaCrownVerifier::new(BetaCrownConfig {
        enable_la_warm_start: true,
        ..Default::default()
    })
    .propagate_crown_batched_with_context_capture_la(&graph, &ctx, &objective, &NaiveCpuGemmEngine)
    .expect("capture-la path should succeed when warm-start enabled");

    let captured = with_capture
        .intermediate_la
        .expect("intermediate lA should be captured when warm-start is enabled");
    assert_eq!(captured.len(), 1, "expected one domain's captured lA map");
    assert!(
        !captured[0].is_empty(),
        "captured lA map should contain at least the output node"
    );
    assert!(
        captured[0].contains_key("linear2"),
        "captured lA should include output node 'linear2', got keys: {:?}",
        captured[0].keys().collect::<Vec<_>>()
    );
}

#[test]
fn test_concretize_dense_specs_without_input_accumulation_uses_spec_ibp_fallback_4403() {
    let graph = build_two_output_identity_graph_for_dense_spec_fallback_4403();

    let constrained_input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .expect("valid constrained input");
    let spec_matrix = arr2(&[[1.0_f32, -1.0_f32], [0.5_f32, 1.5_f32]]);
    let raw_output_bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, 2.0_f32]).into_dyn(),
        arr1(&[3.0_f32, 4.0_f32]).into_dyn(),
    )
    .expect("valid raw output bounds");

    let mut cache = HashMap::new();
    cache.insert("out".to_string(), Arc::new(raw_output_bounds.clone()));
    let expected = graph
        .propagate_crown_with_specs_fallback_ibp(&constrained_input, &spec_matrix, &cache, "out")
        .expect("spec-space IBP fallback should succeed");

    let mut node_linear_bounds =
        IndexedPendingLinearBounds::new(graph.dispatch_plan().expect("plan"), 1);
    let input_accumulated = node_linear_bounds.input_accumulated().to_vec();
    let input_bounds_vec = node_linear_bounds.take_network_input();
    let results = BetaCrownVerifier::concretize_batched_results_specs(
        &graph,
        &spec_matrix,
        std::slice::from_ref(&constrained_input),
        &[&cache],
        &input_accumulated,
        input_bounds_vec,
        "out",
        &[spec_matrix.nrows()],
        1,
        false,
    )
    .expect("dense-spec concretization should succeed");

    assert_eq!(
        results.len(),
        1,
        "single-domain fallback should return one result"
    );
    assert!(
        results[0].input_linear.is_none(),
        "IBP fallback branch must not report input linear bounds",
    );

    assert_flat_bounds_eq_4403(
        &results[0].output_bounds,
        &expected,
        "dense-spec fallback should stay in spec space",
    );

    let raw = raw_output_bounds.flatten();
    assert_eq!(
        results[0].output_bounds.flatten().lower().len(),
        raw.lower().len(),
        "spec-space and raw output bounds must share element count for this fixture",
    );
    assert_ne!(
        results[0].output_bounds.flatten().lower(),
        raw.lower(),
        "regression fixture must differ from raw output-node lower bounds",
    );
    assert_ne!(
        results[0].output_bounds.flatten().upper(),
        raw.upper(),
        "regression fixture must differ from raw output-node upper bounds",
    );
}

// =========================================================================
// Regression test for #1996: UnsupportedOp identity fallback removed
// =========================================================================

/// Regression test #1996: layers returning UnsupportedOp from propagate_crown_backward
/// must cause the batched wildcard dispatch to return error, not silently pass bounds
/// through unchanged (identity fallback).
///
/// Before #1996, the Err(UnsupportedOp) arm in batched.rs:427 accumulated identity
/// bounds on all inputs — unsound because the layer's transformation was skipped.
/// After #1996, the error propagates up so the BaB loop can fall back to sequential.
#[test]
fn test_unsupported_op_returns_error_not_identity_1996() {
    use crate::GatherLayer;

    // GatherLayer::propagate_linear returns UnsupportedOp because Gather requires
    // index information that isn't available in the CROWN backward path.
    let gather = GatherLayer::new(0, None, vec![1]);
    let layer = Layer::Gather(gather);
    let node_inputs = vec!["input".to_string()];
    let lb = LinearBounds::identity(1);
    let constrained_input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let mut node_linear_bounds: HashMap<String, Vec<Option<LinearBounds>>> = HashMap::new();
    let mut input_accumulated = false;

    let result = BetaCrownVerifier::propagate_wildcard_crown_backward_batched(
        "gather_node",
        &layer,
        &node_inputs,
        lb,
        &constrained_input,
        &HashMap::new(),
        &mut node_linear_bounds,
        &mut input_accumulated,
        0,
        1,
    );

    assert!(
        result.is_err(),
        "UnsupportedOp from propagate_crown_backward must return error, \
         not silently pass bounds through unchanged (identity fallback). \
         Before #1996, this would Ok(()) with unsound identity bounds."
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Gather"),
        "Error should mention the layer type, got: {}",
        err_msg
    );
}

// ---- #refold-guard unit tests (pure selection + comparison) ----

fn gcr(lower: Vec<f32>, upper: Vec<f32>) -> ny_core::GpuCrownResult {
    ny_core::GpuCrownResult {
        lower_bounds: lower,
        upper_bounds: upper,
    }
}

/// The guard always checks domain 0, plus the domain with the LARGEST minimum
/// lower bound (the most verified-looking row), dedup'd.
#[test]
fn refold_guard_indices_pick_anchor_and_most_verified() {
    let results = vec![
        gcr(vec![-5.0, -2.0], vec![1.0, 1.0]),
        gcr(vec![0.9, 0.4], vec![2.0, 2.0]), // min lower 0.4 — most verified-looking
        gcr(vec![-1.0, 0.8], vec![2.0, 2.0]),
    ];
    assert_eq!(super::refold_guard_indices(&results), vec![0, 1]);

    // Argmax == 0 → dedup to just the anchor.
    let results = vec![gcr(vec![3.0], vec![4.0]), gcr(vec![-1.0], vec![0.0])];
    assert_eq!(super::refold_guard_indices(&results), vec![0]);

    // A fully-NaN row scores -inf and never wins the argmax.
    let results = vec![
        gcr(vec![-9.0], vec![1.0]),
        gcr(vec![f32::NAN], vec![1.0]),
        gcr(vec![-3.0], vec![1.0]),
    ];
    assert_eq!(super::refold_guard_indices(&results), vec![0, 2]);
}

/// The comparison contract mirrors the kernel differential oracles: two-sided
/// relative 1e-3 closeness per row, both bounds; reorder-scale noise passes,
/// cross-domain misassignment-scale drift fails, non-finite fails closed.
#[test]
fn refold_rows_match_contract() {
    let wide = gcr(vec![1.0, -2.0], vec![3.0, 4.0]);
    // Bitwise-identical (internal stacker fallback case) passes.
    assert!(super::refold_rows_match(
        &wide,
        &gcr(vec![1.0, -2.0], vec![3.0, 4.0])
    ));
    // Reorder-scale noise (well inside 1e-3 relative) passes.
    assert!(super::refold_rows_match(
        &wide,
        &gcr(vec![1.0005, -2.0005], vec![3.001, 4.001])
    ));
    // Misassignment-scale drift (another domain's row) fails.
    assert!(!super::refold_rows_match(
        &wide,
        &gcr(vec![1.5, -2.0], vec![3.0, 4.0])
    ));
    // Non-finite on either side fails closed.
    assert!(!super::refold_rows_match(
        &wide,
        &gcr(vec![f32::NAN, -2.0], vec![3.0, 4.0])
    ));
    assert!(!super::refold_rows_match(
        &gcr(vec![f32::INFINITY, -2.0], vec![3.0, 4.0]),
        &gcr(vec![1.0, -2.0], vec![3.0, 4.0])
    ));
    // Length mismatch fails closed.
    assert!(!super::refold_rows_match(&wide, &gcr(vec![1.0], vec![3.0])));
}

/// #wide-decline-tally WIRING PIN (CPU-only).
///
/// The tally is worthless if a refusal can slip through unlabelled: an entry that
/// returns `None` without naming a reason reads, in the `[wide-lane]` report, as
/// "there were simply no candidate batches" — the exact ambiguity this work
/// exists to remove. So pin the invariant directly on the production entry:
/// a declined candidate ALWAYS bumps the candidate counter AND at least one
/// decline reason.
///
/// The fixture is a ReLU-only (non-conv) graph, so the entry cannot reach the
/// GPU seam. Which reason wins depends on whether this host has a sound GPU
/// backend registered (`entry_no_sound_backend` vs `entry_graph_not_conv`), so
/// the assertion is on "some reason was recorded", not on a host-specific one.
#[test]
fn wide_entry_records_a_reason_for_every_declined_candidate() {
    use ny_core::wide_lane_telemetry::{
        reset_wide_lane_telemetry_for_tests, wide_lane_candidate_count, wide_lane_decline_tally,
    };

    let graph = build_single_relu_graph_for_batched_mode_tests();
    let input =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).expect("valid input");
    let node_bounds = graph
        .collect_node_bounds(&input)
        .expect("graph bounds should collect");
    let cache: HashMap<String, Arc<BoundedTensor>> = node_bounds
        .into_iter()
        .map(|(name, bt)| (name, Arc::new(bt)))
        .collect();

    let n_domains = 2usize;
    let cache_refs: Vec<&HashMap<String, Arc<BoundedTensor>>> = vec![&cache; n_domains];
    let inputs: Vec<BoundedTensor> = vec![input; n_domains];
    let betas: Vec<Option<&crate::beta_crown::state::GraphBetaState>> = vec![None; n_domains];
    let alphas: Vec<Option<&crate::beta_crown::state::GraphDomainAlphaState>> =
        vec![None; n_domains];

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    reset_wide_lane_telemetry_for_tests();
    let declined = verifier.try_gpu_beta_batched_resnet_opt(
        &graph,
        "linear2",
        1,
        &[1.0_f32],
        1,
        n_domains,
        &cache_refs,
        &inputs,
        &betas,
        &alphas,
        &NaiveCpuGemmEngine,
        "tally-wiring-pin",
        None,
        false,
    );

    assert!(
        declined.is_none(),
        "a ReLU-only graph has no conv suffix, so the resnet wide entry must decline"
    );
    // `>=` not `==`: the counters are process-global and cargo runs this crate's
    // tests in parallel, so a sibling test may also have entered the lane.
    assert!(
        wide_lane_candidate_count() >= 1,
        "every call must count itself as a candidate — the report's denominator"
    );
    let tally = wide_lane_decline_tally();
    assert!(
        !tally.is_empty(),
        "a declined candidate must name a reason; an unlabelled `None` is exactly \
         the blind spot this tally removes"
    );
    reset_wide_lane_telemetry_for_tests();
}
