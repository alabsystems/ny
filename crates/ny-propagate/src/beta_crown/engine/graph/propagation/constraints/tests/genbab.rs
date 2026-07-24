// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GenBaB constraint integration regressions (#2399).

use ndarray::arr1;
use std::collections::HashMap;

use crate::beta_crown::{GraphCrownContext, GraphNeuronConstraint, GraphSplitHistory};
use crate::{BetaCrownConfig, BetaCrownVerifier, BoundedTensor};

use super::operator_dispatch::build_sigmoid_graph;
use super::support::scalar_interval;
use super::TOL;
// =========================================================================
// #2399: GenBaB constraint integration in forward pass
// =========================================================================

/// Test that `apply_genbab_pre_constraints` correctly tightens bounds at
/// arbitrary split points for general nonlinearities.
///
/// Mathematical basis: GenBaB branching constrains pre-activation x to either
/// x >= split_point (upper branch) or x <= split_point (lower branch).
/// Upper branch: lower = max(lower, split_point).
/// Lower branch: upper = min(upper, split_point).
#[test]
fn test_apply_genbab_pre_constraints_tightens_at_split_point_2399() {
    use super::super::lookups::apply_genbab_pre_constraints;

    // Bounds: neuron 0 ∈ [-2.0, 2.0]
    let bounds = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn())
        .expect("valid bounds");

    // Upper branch at split_point=0.5: x >= 0.5 → lower = max(-2.0, 0.5) = 0.5
    let upper_branch = vec![(0usize, 0.5f32, true, "gelu1".to_string())];
    let result = apply_genbab_pre_constraints(&bounds, &upper_branch).expect("should succeed");
    let flat = result.flatten();
    assert!(
        (flat.lower()[[0]] - 0.5).abs() < TOL,
        "upper branch: lower should be 0.5, got {}",
        flat.lower()[[0]]
    );
    assert!(
        (flat.upper()[[0]] - 2.0).abs() < TOL,
        "upper branch: upper should remain 2.0, got {}",
        flat.upper()[[0]]
    );

    // Lower branch at split_point=-0.3: x <= -0.3 → upper = min(2.0, -0.3) = -0.3
    let lower_branch = vec![(0usize, -0.3f32, false, "gelu1".to_string())];
    let result = apply_genbab_pre_constraints(&bounds, &lower_branch).expect("should succeed");
    let flat = result.flatten();
    assert!(
        (flat.lower()[[0]] - (-2.0)).abs() < TOL,
        "lower branch: lower should remain -2.0, got {}",
        flat.lower()[[0]]
    );
    assert!(
        (flat.upper()[[0]] - (-0.3)).abs() < TOL,
        "lower branch: upper should be -0.3, got {}",
        flat.upper()[[0]]
    );
}

/// Test that GenBaB pre-constraint returns InfeasibleDomain when the split point
/// makes the interval empty.
#[test]
fn test_genbab_pre_constraint_infeasibility_returns_infeasible_domain_2399() {
    use super::super::lookups::apply_genbab_pre_constraints;

    // Bounds: neuron 0 ∈ [-2.0, -1.0] (always negative)
    let bounds = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[-1.0]).into_dyn())
        .expect("valid bounds");

    // Upper branch at split_point=0.0: x >= 0.0, but upper = -1.0 < 0.0
    // After clamping: lower = max(-2.0, 0.0) = 0.0, upper = -1.0 → infeasible
    let constraints = vec![(0usize, 0.0f32, true, "gelu1".to_string())];
    let result = apply_genbab_pre_constraints(&bounds, &constraints);
    let err = result.expect_err("infeasible genbab constraint should produce InfeasibleDomain");
    assert!(
        err.is_infeasible_domain(),
        "Error should be InfeasibleDomain variant, got: {err}"
    );
}

/// Build forward bounds for the sigmoid graph with an optional GenBaB constraint.
/// Returns (bounds_cache, constrained_input) from `compute_constrained_forward_bounds`.
fn sigmoid_forward_with_genbab(
    split_point: Option<(f32, bool)>,
) -> (
    HashMap<String, std::sync::Arc<BoundedTensor>>,
    BoundedTensor,
) {
    use crate::beta_crown::GenBabConstraint;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_sigmoid_graph();
    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn())
        .expect("valid input bounds");

    let mut history = GraphSplitHistory::new();
    if let Some((sp, is_upper)) = split_point {
        history.add_genbab_constraint(
            GenBabConstraint::new("sigmoid1".to_string(), 0, sp, is_upper, 1.0)
                .expect("valid genbab constraint"),
        );
    }
    verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("forward should succeed")
}

/// Test that the forward pass tightens pre-activation bounds when GenBaB constraints
/// are present on a Sigmoid node.
///
/// GenBaB constraint: sigmoid1, upper branch, split_point=0.0
///   → pre-activation constrained to [max(lower, 0.0), upper] = [0.0, 2.0]
#[test]
fn test_genbab_forward_tightens_preactivation_2399() {
    let (uc_cache, _) = sigmoid_forward_with_genbab(None);
    let (c_cache, _) = sigmoid_forward_with_genbab(Some((0.0, true)));

    let uc_linear1 = uc_cache
        .get("linear1")
        .expect("linear1 in unconstrained cache");
    let c_linear1 = c_cache
        .get("linear1")
        .expect("linear1 in constrained cache");

    let (uc_l, uc_u) = scalar_interval(uc_linear1);
    let (c_l, c_u) = scalar_interval(c_linear1);

    // Constraint x >= 0.0 should tighten lower from -2.0 to 0.0
    assert!(
        c_l >= uc_l - TOL,
        "constrained lower {} should be >= unconstrained {}",
        c_l,
        uc_l
    );
    assert!(
        (c_l - 0.0).abs() < TOL,
        "constrained lower should be 0.0, got {}",
        c_l
    );
    assert!(
        (c_u - uc_u).abs() < TOL,
        "upper should remain unchanged: {} vs {}",
        c_u,
        uc_u
    );
}

/// Test that GenBaB forward tightening propagates through the sigmoid to produce
/// tighter output bounds.
///
/// With x ≥ 0, sigmoid(x) ≥ sigmoid(0) = 0.5, vs unconstrained ≈ 0.119.
#[test]
fn test_genbab_forward_tightens_sigmoid_output_2399() {
    let (uc_cache, _) = sigmoid_forward_with_genbab(None);
    let (c_cache, _) = sigmoid_forward_with_genbab(Some((0.0, true)));

    let uc_sigmoid = uc_cache
        .get("sigmoid1")
        .expect("sigmoid1 in unconstrained cache");
    let c_sigmoid = c_cache
        .get("sigmoid1")
        .expect("sigmoid1 in constrained cache");

    let (uc_sig_l, _) = scalar_interval(uc_sigmoid);
    let (c_sig_l, _) = scalar_interval(c_sigmoid);

    // Constrained sigmoid lower should be strictly above unconstrained.
    // unconstrained ≈ sigmoid(-2) ≈ 0.119, constrained ≈ sigmoid(0) = 0.5
    assert!(
        c_sig_l > uc_sig_l + 0.01,
        "constrained sigmoid lower {} should be significantly above unconstrained {} \
         (GenBaB forward tightening propagates through nonlinearity)",
        c_sig_l,
        uc_sig_l
    );
}

/// Test that GenBaB constraints produce tighter CROWN output bounds
/// (end-to-end through both forward and backward passes).
#[test]
fn test_genbab_crown_output_tighter_than_unconstrained_2399() {
    use crate::beta_crown::GenBabConstraint;

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_sigmoid_graph();
    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn())
        .expect("valid input bounds");

    // Unconstrained CROWN
    let unconstrained_history = GraphSplitHistory::new();
    let uc_context = GraphCrownContext::for_history(&unconstrained_history);
    let (uc_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &uc_context, None, None)
        .expect("unconstrained CROWN should succeed");

    // Constrained CROWN: sigmoid1 upper branch at split_point=0.0
    let mut constrained_history = GraphSplitHistory::new();
    constrained_history.add_genbab_constraint(
        GenBabConstraint::new("sigmoid1".to_string(), 0, 0.0, true, 1.0)
            .expect("valid genbab constraint"),
    );
    let c_context = GraphCrownContext::for_history(&constrained_history);
    let (c_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &c_context, None, None)
        .expect("constrained CROWN should succeed");

    let (uc_lower, uc_upper) = scalar_interval(&uc_output);
    let (c_lower, c_upper) = scalar_interval(&c_output);

    // Constrained bounds should be at least as tight (lower >= unconstrained, upper <= unconstrained)
    assert!(
        c_lower >= uc_lower - TOL,
        "constrained lower {} should be >= unconstrained lower {}",
        c_lower,
        uc_lower
    );
    assert!(
        c_upper <= uc_upper + TOL,
        "constrained upper {} should be <= unconstrained upper {}",
        c_upper,
        uc_upper
    );

    // Soundness: constrained bounds must still contain the true output range
    // for x ∈ [0, 2] (the constrained input domain).
    // sigmoid(0) = 0.5, sigmoid(2) ≈ 0.881
    let true_lower = 0.5;
    let true_upper = 1.0 / (1.0 + (-2.0f32).exp()); // sigmoid(2)
    assert!(
        c_lower <= true_lower + TOL,
        "CROWN lower {} must be <= true min {} for soundness",
        c_lower,
        true_lower
    );
    assert!(
        c_upper >= true_upper - TOL,
        "CROWN upper {} must be >= true max {} for soundness",
        c_upper,
        true_upper
    );
}

/// Test `is_any_constrained()` detects both ReLU and GenBaB constraints.
#[test]
fn test_is_any_constrained_detects_both_types_2399() {
    use crate::beta_crown::GenBabConstraint;

    let mut history = GraphSplitHistory::new();

    // No constraints yet
    assert!(!history.is_any_constrained("relu1", 0));
    assert!(!history.is_any_constrained("gelu1", 0));

    // Add ReLU constraint
    history.add_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    assert!(history.is_any_constrained("relu1", 0));
    assert!(!history.is_any_constrained("relu1", 1)); // different neuron
    assert!(!history.is_any_constrained("gelu1", 0)); // different node

    // Add GenBaB constraint
    history.add_genbab_constraint(
        GenBabConstraint::new("gelu1".to_string(), 0, 0.5, true, 1.0)
            .expect("valid genbab constraint"),
    );
    assert!(history.is_any_constrained("gelu1", 0));
    assert!(!history.is_any_constrained("gelu1", 1)); // different neuron

    // Both types present
    assert!(history.is_any_constrained("relu1", 0)); // still detected
    assert!(history.is_any_constrained("gelu1", 0)); // still detected
}

/// Test `build_constraint_lookups` correctly builds pre_genbab map from GenBaB constraints.
#[test]
fn test_build_constraint_lookups_includes_genbab_2399() {
    use super::super::lookups::build_constraint_lookups;
    use crate::beta_crown::GenBabConstraint;

    let graph = build_sigmoid_graph();

    let relu_constraints = vec![];
    let genbab_constraints = vec![
        GenBabConstraint::new("sigmoid1".to_string(), 0, 0.5, true, 1.0)
            .expect("valid genbab constraint"),
    ];

    let lookups = build_constraint_lookups(&relu_constraints, &genbab_constraints, &graph)
        .expect("should succeed");

    // pre_genbab should map "linear1" (sigmoid1's input) → constraint info
    assert!(
        lookups.pre_genbab.contains_key("linear1"),
        "pre_genbab should map pre-activation node 'linear1'"
    );
    let constraints = &lookups.pre_genbab["linear1"];
    assert_eq!(constraints.len(), 1, "should have one GenBaB constraint");
    assert_eq!(constraints[0].0, 0, "neuron_idx should be 0");
    assert!(
        (constraints[0].1 - 0.5).abs() < TOL,
        "split_point should be 0.5"
    );
    assert!(constraints[0].2, "is_upper_branch should be true");
    assert_eq!(constraints[0].3, "sigmoid1", "node name should be sigmoid1");

    // by_relu should be empty (no ReLU constraints)
    assert!(lookups.by_relu.is_empty(), "by_relu should be empty");
    // pre should be empty (no ReLU constraints)
    assert!(lookups.pre.is_empty(), "pre should be empty");
}

/// Test dual-sided GenBaB constraints: both upper-branch and lower-branch on the
/// same neuron pinch bounds from both sides.
///
/// In BaB, successive splits can produce domains where a neuron has both
/// x >= split_low (upper branch) AND x <= split_high (lower branch), yielding
/// [split_low, split_high]. This test verifies sequential application produces
/// the correct pinched interval.
#[test]
fn test_genbab_dual_sided_constraint_pinches_bounds_2399() {
    use super::super::lookups::apply_genbab_pre_constraints;

    // Bounds: neuron 0 in [-3.0, 3.0]
    let bounds = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[3.0]).into_dyn())
        .expect("valid bounds");

    // Two constraints: upper branch at -1.0 (x >= -1.0) + lower branch at 1.5 (x <= 1.5)
    // Expected: [-1.0, 1.5]
    let constraints = vec![
        (0usize, -1.0f32, true, "sigmoid1".to_string()),
        (0usize, 1.5f32, false, "sigmoid1".to_string()),
    ];
    let result = apply_genbab_pre_constraints(&bounds, &constraints).expect("should succeed");
    let flat = result.flatten();
    assert!(
        (flat.lower()[[0]] - (-1.0)).abs() < TOL,
        "dual-sided: lower should be -1.0, got {}",
        flat.lower()[[0]]
    );
    assert!(
        (flat.upper()[[0]] - 1.5).abs() < TOL,
        "dual-sided: upper should be 1.5, got {}",
        flat.upper()[[0]]
    );

    // Reversed order should give the same result (commutativity)
    let constraints_rev = vec![
        (0usize, 1.5f32, false, "sigmoid1".to_string()),
        (0usize, -1.0f32, true, "sigmoid1".to_string()),
    ];
    let result_rev =
        apply_genbab_pre_constraints(&bounds, &constraints_rev).expect("should succeed");
    let flat_rev = result_rev.flatten();
    assert!(
        (flat_rev.lower()[[0]] - (-1.0)).abs() < TOL,
        "reversed: lower should be -1.0, got {}",
        flat_rev.lower()[[0]]
    );
    assert!(
        (flat_rev.upper()[[0]] - 1.5).abs() < TOL,
        "reversed: upper should be 1.5, got {}",
        flat_rev.upper()[[0]]
    );
}

/// A GenBaB constraint on a binary McCormick op (MulBinary z = x·y) with
/// `input_index = Some(1)` must map to the SECOND input node, not the first.
///
/// Regression for the GenBaB MulBinary blocker (#mul-genbab): `build_constraint_lookups`
/// previously hard-coded `inputs.first()`, so a second-input split clamped the wrong
/// input — a hard index error (different-length inputs) or a silent clamp of the
/// wrong neuron (unsound). The constraint now carries `input_index` and the lookup
/// resolves `node.inputs[input_index]`.
#[test]
fn test_build_constraint_lookups_genbab_input_index_routes_to_second_input() {
    use super::super::lookups::build_constraint_lookups;
    use crate::beta_crown::GenBabConstraint;
    use crate::layers::binary_ops::MulBinaryLayer;
    use crate::layers::LinearLayer;
    use crate::{GraphNetwork, GraphNode, Layer};
    use ndarray::arr2;

    // gate (input 0) and up (input 1) feed an element-wise MulBinary.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "gate",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
    ));
    graph.add_node(GraphNode::from_input(
        "up",
        Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap()),
    ));
    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        "gate",
        "up",
    ));
    graph.set_output("mul");

    // Split on input_index = 1 (the "up" input).
    let c1 = GenBabConstraint::new("mul".to_string(), 0, 0.5, true, 1.0)
        .unwrap()
        .with_input_index(1);
    let lookups = build_constraint_lookups(&[], std::slice::from_ref(&c1), &graph)
        .expect("input_index=1 lookup should succeed");
    assert!(
        lookups.pre_genbab.contains_key("up"),
        "input_index=1 must route to the second input node 'up', got keys {:?}",
        lookups.pre_genbab.keys().collect::<Vec<_>>()
    );
    assert!(
        !lookups.pre_genbab.contains_key("gate"),
        "input_index=1 must NOT route to the first input node 'gate'"
    );

    // Split on input_index = 0 (default) routes to "gate".
    let c0 = GenBabConstraint::new("mul".to_string(), 0, 0.5, true, 1.0)
        .unwrap()
        .with_input_index(0);
    let lookups0 = build_constraint_lookups(&[], std::slice::from_ref(&c0), &graph)
        .expect("input_index=0 lookup should succeed");
    assert!(
        lookups0.pre_genbab.contains_key("gate"),
        "input_index=0 must route to the first input node 'gate'"
    );
}

/// Test that dual-sided constraints detect infeasibility when split points cross.
///
/// If upper branch says x >= 2.0 and lower branch says x <= -1.0, the domain
/// is empty (no x satisfies both). The function must return InfeasibleDomain.
#[test]
fn test_genbab_dual_sided_crossing_splits_infeasible_2399() {
    use super::super::lookups::apply_genbab_pre_constraints;

    let bounds = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[3.0]).into_dyn())
        .expect("valid bounds");

    // Upper branch at 2.0 (x >= 2.0) then lower branch at -1.0 (x <= -1.0)
    // After first: lower = max(-3, 2) = 2.0, upper = 3.0
    // After second: upper = min(3, -1) = -1.0 → lower(2.0) > upper(-1.0) → infeasible
    let constraints = vec![
        (0usize, 2.0f32, true, "gelu1".to_string()),
        (0usize, -1.0f32, false, "gelu1".to_string()),
    ];
    let err = apply_genbab_pre_constraints(&bounds, &constraints)
        .expect_err("crossing splits should be infeasible");
    assert!(
        err.is_infeasible_domain(),
        "Error should be InfeasibleDomain, got: {err}"
    );
}
