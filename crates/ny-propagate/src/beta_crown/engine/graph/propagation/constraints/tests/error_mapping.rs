// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Infeasible-domain and propagation-failure regressions (#1978, #2926).

use std::collections::HashMap;

use ndarray::{arr1, arr2};

use crate::beta_crown::{GraphCrownContext, GraphNeuronConstraint, GraphSplitHistory};
use crate::{
    BetaCrownConfig, BetaCrownVerifier, BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer,
    ReLULayer,
};

use super::two_neuron::{build_two_neuron_input_bounds, build_two_neuron_relu_graph};
// =========================================================================
// Regression tests for #1978: propagation error → PropagationFailure mapping
// =========================================================================

/// Build a graph where pre-activation bounds at relu1 are always positive:
/// linear1(w=1, b=2) maps [0, 1] to [2, 3]. An "inactive" constraint on
/// relu1 neuron 0 is infeasible because pre_l=2 > 0.
fn build_always_positive_relu_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[2.0]))).expect("valid linear1");
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

fn positive_input_bounds() -> BoundedTensor {
    BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn())
        .expect("valid input bounds")
}

/// Infeasible: relu1 marked inactive but pre-activation is [2.0, 3.0].
fn infeasible_inactive_history() -> GraphSplitHistory {
    GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 0.0,
    })
}

/// Build a graph where pre-activation bounds at relu1 are always negative:
/// linear1(w=1, b=-2) maps [0, 1] to [-2, -1]. An "active" constraint on
/// relu1 neuron 0 is infeasible because pre_u=-1 < 0.
fn build_always_negative_relu_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[-2.0]))).expect("valid linear1");
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

/// Infeasible: relu1 marked active but pre-activation is [-2.0, -1.0].
fn infeasible_active_history() -> GraphSplitHistory {
    GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    })
}

/// Regression test for #1978: `propagate_crown_with_graph_constraints` must return
/// `Err` (not silently produce stale bounds) when given an infeasible inactive
/// ReLU constraint. This is the prerequisite for the BaB loop mapping errors to
/// `GraphDomainResult::PropagationFailure`.
#[test]
fn test_infeasible_inactive_constraint_returns_error_1978() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_always_positive_relu_graph();
    let input = positive_input_bounds();
    let history = infeasible_inactive_history();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(
        result.is_err(),
        "Infeasible inactive constraint should produce Err, got Ok: pre-activation is [2, 3] \
         but constraint says neuron is inactive (<=0)"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Infeasible"),
        "Error should mention 'Infeasible', got: {}",
        err_msg
    );
}

/// Regression test for #1978: same as above but for an infeasible active constraint
/// where pre-activation upper bound is strictly negative.
#[test]
fn test_infeasible_active_constraint_returns_error_1978() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_always_negative_relu_graph();
    let input = positive_input_bounds();
    let history = infeasible_active_history();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    assert!(
        result.is_err(),
        "Infeasible active constraint should produce Err, got Ok: pre-activation is [-2, -1] \
         but constraint says neuron is active (>=0)"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Infeasible"),
        "Error should mention 'Infeasible', got: {}",
        err_msg
    );
}

/// Regression test for #1978: `propagate_crown_with_graph_beta` (the wrapper
/// called by the BaB NoUnstable path) must also propagate the error from
/// infeasible constraints, not swallow it.
#[test]
fn test_propagate_crown_with_graph_beta_propagates_infeasible_error_1978() {
    use crate::beta_crown::state::GraphBetaState;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_always_positive_relu_graph();
    let input = positive_input_bounds();
    let history = infeasible_inactive_history();
    let context = GraphCrownContext::for_history(&history);
    let beta_state = GraphBetaState::from_history(&history).expect("finite history");

    let result = verifier.propagate_crown_with_graph_beta(
        &graph,
        &input,
        &context,
        &beta_state,
        Some(&[1.0]),
    );
    assert!(
        result.is_err(),
        "propagate_crown_with_graph_beta must propagate infeasible constraint error, \
         got Ok — this would cause the BaB loop to return stale NoUnstable bounds"
    );
}

// ---------------------------------------------------------------------------
// #2926: InfeasibleDomain error variant tests
// ---------------------------------------------------------------------------

/// Verify that single-constraint infeasibility (inactive on always-positive)
/// returns an `InfeasibleDomain` error variant, not generic `InvalidSpec`.
#[test]
fn test_infeasible_inactive_returns_infeasible_domain_variant_2926() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_always_positive_relu_graph();
    let input = positive_input_bounds();
    let history = infeasible_inactive_history();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    let err = result.expect_err("infeasible inactive constraint should produce Err");
    assert!(
        err.is_infeasible_domain(),
        "Error should be InfeasibleDomain variant, got: {err}"
    );
}

/// Verify that single-constraint infeasibility (active on always-negative)
/// returns an `InfeasibleDomain` error variant.
#[test]
fn test_infeasible_active_returns_infeasible_domain_variant_2926() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_always_negative_relu_graph();
    let input = positive_input_bounds();
    let history = infeasible_active_history();
    let context = GraphCrownContext::for_history(&history);

    let result =
        verifier.propagate_crown_with_graph_constraints(&graph, &input, &context, None, None);
    let err = result.expect_err("infeasible active constraint should produce Err");
    assert!(
        err.is_infeasible_domain(),
        "Error should be InfeasibleDomain variant, got: {err}"
    );
}

/// Regression for `#4354`: inherited node caches must not turn a sound child
/// into a false infeasible domain.
///
/// Setup: two-neuron ReLU graph, neuron 0 constrained active.
/// We inject stale inherited bounds where the unconstrained neuron's cached
/// ReLU interval is disjoint from the fresh child-domain ReLU repropagation.
/// The forward pass must keep the fresh child bounds instead of treating the
/// child as infeasible.
#[test]
fn test_inherited_relu_cache_conflict_falls_back_to_fresh_child_bounds_4354() {
    use std::sync::Arc;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds(); // x ∈ [-1, 1]²

    // Constrain neuron 0 as active. Neuron 1 is unconstrained.
    let history = GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    // Inject base_bounds where relu1's CROWN-IBP bound for neuron 1 has
    // crown_l = 5.0, crown_u = 6.0. The actual relu output for neuron 1
    // (pre-activation [-1, 1], relu → [0, 1]) has relu_upper = 1.0.
    // Intersection: lower = max(0.0, 5.0) = 5.0, upper = min(1.0, 6.0) = 1.0.
    // Fresh child-domain bounds are still sound ([0, 1]), so the inherited
    // cache must be dropped instead of marking the child infeasible.
    let mut base_bounds: HashMap<String, Arc<BoundedTensor>> = HashMap::new();
    // linear1 output: identity mapping, so bounds = input = [-1, 1]²
    base_bounds.insert(
        "linear1".to_string(),
        Arc::new(
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .expect("valid linear1 bounds"),
        ),
    );
    // relu1 CROWN-IBP bounds: neuron 0 = [0, 1] (fine), neuron 1 = [5, 6] (stale/conflicting)
    base_bounds.insert(
        "relu1".to_string(),
        Arc::new(
            BoundedTensor::new(arr1(&[0.0, 5.0]).into_dyn(), arr1(&[1.0, 6.0]).into_dyn())
                .expect("valid relu1 bounds"),
        ),
    );
    // linear2 output: sum of relu neurons
    base_bounds.insert(
        "linear2".to_string(),
        Arc::new(
            BoundedTensor::new(arr1(&[5.0]).into_dyn(), arr1(&[7.0]).into_dyn())
                .expect("valid linear2 bounds"),
        ),
    );

    let (cache, constrained_input) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, Some(&base_bounds), None)
        .expect("conflicting inherited cache should fall back to fresh child bounds");
    assert_eq!(
        constrained_input.lower(),
        input.lower(),
        "non-input constraint should keep the input box unchanged"
    );
    assert!(
        cache.contains_key("relu1"),
        "forward cache should include relu1 after fallback"
    );
    let relu_bounds = cache.get("relu1").expect("relu1 bounds should exist");
    assert!(
        relu_bounds.lower()[[0]] == 0.0 && relu_bounds.upper()[[0]] == 1.0,
        "constrained neuron should stay active after fallback, got [{}, {}]",
        relu_bounds.lower()[[0]],
        relu_bounds.upper()[[0]]
    );
    assert!(
        relu_bounds.lower()[[1]] == 0.0 && relu_bounds.upper()[[1]] == 1.0,
        "unconstrained neuron should use fresh child-domain ReLU bounds after dropping the stale inherited cache, got [{}, {}]",
        relu_bounds.lower()[[1]],
        relu_bounds.upper()[[1]]
    );
    let linear2_bounds = cache.get("linear2").expect("linear2 bounds should exist");
    let linear2_lower = linear2_bounds.lower()[[0]];
    let linear2_upper = linear2_bounds.upper()[[0]];
    assert!(
        linear2_lower.abs() <= 1e-6 && (linear2_upper - 7.0).abs() <= 1e-6,
        "downstream linear node should stay finite after the ReLU fallback, got [{linear2_lower}, {linear2_upper}]"
    );
}

/// Pre-constraint infeasibility (from lookups.rs) should also be InfeasibleDomain.
#[test]
fn test_pre_constraint_infeasibility_returns_infeasible_domain_2926() {
    use super::super::lookups::apply_pre_constraints;

    // Bounds: neuron 0 ∈ [2.0, 3.0] (always positive)
    let bounds =
        BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[3.0]).into_dyn()).expect("valid");

    // Inactive constraint on neuron 0: clamp upper to min(3.0, 0.0) = 0.0
    // But lower is 2.0 > 0.0, so infeasible.
    let constraints = vec![(0usize, false, "test_relu".to_string())];
    let result = apply_pre_constraints(&bounds, &constraints);
    let err = result.expect_err("pre-constraint on always-positive should be InfeasibleDomain");
    assert!(
        err.is_infeasible_domain(),
        "Error should be InfeasibleDomain variant, got: {err}"
    );
}
