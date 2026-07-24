// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-neuron graph regressions (#1813, #2826).

use ndarray::{arr1, arr2};

use crate::beta_crown::{GraphCrownContext, GraphNeuronConstraint, GraphSplitHistory};
use crate::{
    BetaCrownConfig, BetaCrownVerifier, BoundedTensor, GraphNetwork, GraphNode, Layer, LinearLayer,
    ReLULayer,
};

use super::support::{assert_cache_bounds_close, assert_scalar_bounds, scalar_interval};
use super::TOL;

use ny_test_utils::assert_bounded_tensor_close;
// ====================================================================
// Multi-neuron graph tests — exercise backward.rs with non-trivial networks.
// Part of proof_coverage phase: backward.rs extracted in #1813 with zero tests.
// ====================================================================

/// Build a 2-input, 2-neuron graph: linear1(2→2) → relu1 → linear2(2→1)
///
/// linear1: W = [[1,0],[0,1]], b = [0,0] (identity)
/// linear2: W = [[1,1]], b = [0] (sum)
///
/// For input [l1,l2]..[u1,u2]:
///   after relu1: [max(l1,0), max(l2,0)] .. [max(u1,0), max(u2,0)]
///   after linear2: [max(l1,0)+max(l2,0)] .. [max(u1,0)+max(u2,0)]
pub(super) fn build_two_neuron_relu_graph() -> GraphNetwork {
    let linear1 = LinearLayer::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), Some(arr1(&[0.0, 0.0])))
        .expect("valid linear1");
    let linear2 = LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).expect("valid linear2");

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

pub(super) fn build_two_neuron_input_bounds() -> BoundedTensor {
    // Input x ∈ [-1, 1]², so both neurons span the unstable region.
    BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
        .expect("valid input bounds")
}

/// Constrain neuron 0 inactive (x0 ≤ 0) and leave neuron 1 unconstrained.
fn mixed_constraint_history() -> GraphSplitHistory {
    GraphSplitHistory::new().with_constraint(GraphNeuronConstraint {
        node_name: "relu1".to_string(),
        neuron_idx: 0,
        is_active: false,
        score: 0.0,
    })
}

/// Constrain both neurons: neuron 0 inactive, neuron 1 active.
fn both_constrained_history() -> GraphSplitHistory {
    GraphSplitHistory::new()
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: false,
            score: 0.0,
        })
        .with_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 1,
            is_active: true,
            score: 0.0,
        })
}

#[test]
fn test_multi_neuron_mixed_constraint_tightens_bounds() {
    // Constrain neuron 0 inactive → relu1[0] = 0. Neuron 1 is unconstrained.
    // Unconstrained: output ∈ [0, 2] (sum of two ReLU outputs on [-1,1]).
    // Constrained: relu1[0] = 0, relu1[1] ∈ [0,1], so output ∈ [0, 1].
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds();

    let unconstrained_history = GraphSplitHistory::new();
    let constrained_history = mixed_constraint_history();
    let unconstrained_ctx = GraphCrownContext::for_history(&unconstrained_history);
    let constrained_ctx = GraphCrownContext::for_history(&constrained_history);

    let (unconstrained_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &unconstrained_ctx, None, None)
        .expect("unconstrained propagation should succeed");
    let (constrained_output, constrained_cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &constrained_ctx, None, None)
        .expect("constrained propagation should succeed");

    let (u_lower, u_upper) = scalar_interval(&unconstrained_output);
    let (c_lower, c_upper) = scalar_interval(&constrained_output);

    // Constrained upper bound must be strictly tighter.
    assert!(
        c_upper < u_upper - 1e-4,
        "mixed constraint should tighten upper: unconstrained={}, constrained={}",
        u_upper,
        c_upper
    );
    // Constrained lower bound must not be looser.
    assert!(
        c_lower >= u_lower - TOL,
        "mixed constraint must not loosen lower: unconstrained={}, constrained={}",
        u_lower,
        c_lower
    );

    // Verify relu1 cache shows neuron 0 is dead (both bounds = 0).
    let relu1_bounds = constrained_cache
        .get("relu1")
        .expect("relu1 must be in cache");
    let relu1_flat = relu1_bounds.flatten();
    assert!(
        relu1_flat.lower()[[0]].abs() < TOL,
        "inactive neuron 0 lower should be 0, got {}",
        relu1_flat.lower()[[0]]
    );
    assert!(
        relu1_flat.upper()[[0]].abs() < TOL,
        "inactive neuron 0 upper should be 0, got {}",
        relu1_flat.upper()[[0]]
    );
}

#[test]
fn test_multi_neuron_both_constrained_soundness_and_cache() {
    // Constrain neuron 0 inactive (→ 0), neuron 1 active (→ identity for x₁ ≥ 0).
    //
    // Forward pass with constraints gives relu1 = [0,0] × [0,1], linear2 output = [0,1].
    //
    // CROWN backward concretizes at the *input* domain [-1,1]². With neuron 0 dead
    // (slope=0) and neuron 1 active (slope=1), the backward linear function is
    // f(x) = 0·x₁ + 1·x₂, giving concretized bounds [-1, 1]. This is sound
    // (contains the true range [0, 1]) but looser than the constrained IBP bounds.
    //
    // This is expected CROWN behavior: backward concretization at the input may
    // not benefit from constraint-induced domain narrowing at internal nodes.
    // The forward cache bounds correctly reflect constraints.
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds();

    let constrained_history = both_constrained_history();
    let constrained_ctx = GraphCrownContext::for_history(&constrained_history);

    let (constrained_output, constrained_cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &constrained_ctx, None, None)
        .expect("both-constrained should succeed");

    let (c_lower, c_upper) = scalar_interval(&constrained_output);

    // Soundness: bounds must contain the true output range [0, 1].
    assert!(
        c_lower <= 0.0 + TOL,
        "constrained lower must contain true min 0: got {}",
        c_lower
    );
    assert!(
        c_upper >= 1.0 - TOL,
        "constrained upper must contain true max 1: got {}",
        c_upper
    );

    // Forward cache should reflect constraints exactly.
    let relu1 = constrained_cache
        .get("relu1")
        .expect("relu1 must be in cache");
    let relu1_flat = relu1.flatten();
    // Neuron 0 inactive → dead.
    assert!(
        relu1_flat.lower()[[0]].abs() < TOL && relu1_flat.upper()[[0]].abs() < TOL,
        "relu1[0] should be dead: [{}, {}]",
        relu1_flat.lower()[[0]],
        relu1_flat.upper()[[0]]
    );
    // Neuron 1 active → lower ≥ 0.
    assert!(
        relu1_flat.lower()[[1]] >= -TOL,
        "relu1[1] active: lower should be ≥ 0, got {}",
        relu1_flat.lower()[[1]]
    );
    assert!(
        (relu1_flat.upper()[[1]] - 1.0).abs() < TOL,
        "relu1[1] active: upper should be 1.0, got {}",
        relu1_flat.upper()[[1]]
    );

    // Forward cache output node should have the tight IBP bounds [0, 1].
    let linear2 = constrained_cache
        .get("linear2")
        .expect("linear2 must be in cache");
    assert_scalar_bounds(
        linear2,
        0.0,
        1.0,
        "linear2 (forward cache, both constrained)",
    );
}

#[test]
fn test_multi_neuron_standard_vs_storing_intermediates_parity() {
    // Verify that Standard and StoringIntermediates modes produce identical bounds
    // on the multi-neuron graph with mixed constraints.
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds();

    let history = mixed_constraint_history();
    let context = GraphCrownContext::for_history(&history);

    let (standard_output, standard_cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("standard propagation should succeed");
    let (intermediate_output, intermediate_cache, intermediate) = verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect("intermediate propagation should succeed");

    assert_bounded_tensor_close(
        &standard_output,
        &intermediate_output,
        TOL,
        "multi-neuron output parity",
    );
    assert_cache_bounds_close(
        &standard_cache,
        &intermediate_cache,
        "multi-neuron cache parity",
    );

    // StoringIntermediates should capture A matrix at relu1 (which is constrained).
    assert!(
        intermediate.a_at_relu.contains_key("relu1"),
        "intermediate pass must capture A matrix for constrained relu1"
    );
    // A matrix should have shape [output_dim, relu_dim] = [1, 2]
    let a_relu = &intermediate.a_at_relu["relu1"];
    assert_eq!(
        a_relu.shape(),
        &[1, 2],
        "A at relu1 should be [1, 2], got {:?}",
        a_relu.shape()
    );
}

#[test]
fn test_beta_contribution_widens_bounds_relative_to_zero_beta() {
    // β-CROWN adds ±β·sign to A matrices. With β=0 (default from_history), bounds
    // should equal standard CROWN. With non-zero β, bounds should differ.
    //
    // Mathematical basis: The Lagrangian augmented bound includes
    //   sum_i(β_i · sign_i · a_j,i) added to lower_a and subtracted from upper_a
    // (See backward.rs lines 342-367)
    //
    // With β > 0, the A matrix coefficients change, producing different concretized bounds.
    use crate::beta_crown::state::{GraphBetaEntry, GraphBetaState};

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds();

    let history = mixed_constraint_history();
    let context = GraphCrownContext::for_history(&history);

    // Zero beta baseline
    let (zero_beta_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("zero-beta propagation should succeed");

    // Non-zero beta: constrain neuron 0 inactive with β=0.5
    let beta_state = GraphBetaState {
        entries: vec![GraphBetaEntry {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            split_point: 0.0,
            value: 0.5,
            sign: -1.0, // inactive constraint
            grad: 0.0,
            m: 0.0,
            v: 0.0,
            v_max: 0.0,
        }],
        ..GraphBetaState::empty()
    };

    let (beta_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, Some(&beta_state), None)
        .expect("nonzero-beta propagation should succeed");

    let (z_lower, z_upper) = scalar_interval(&zero_beta_output);
    let (b_lower, b_upper) = scalar_interval(&beta_output);

    // With non-zero β, bounds should differ from zero-β baseline.
    let lower_changed = (b_lower - z_lower).abs() > 1e-6;
    let upper_changed = (b_upper - z_upper).abs() > 1e-6;
    assert!(
        lower_changed || upper_changed,
        "non-zero β must change at least one bound: zero=({}, {}), beta=({}, {})",
        z_lower,
        z_upper,
        b_lower,
        b_upper
    );

    // Soundness: lower must still be ≤ upper after β modification.
    assert!(
        b_lower <= b_upper + TOL,
        "beta bounds must be valid: lower={} > upper={}",
        b_lower,
        b_upper
    );
}

#[test]
fn test_graph_constraints_skip_non_finite_beta_contributions_2826() {
    use crate::beta_crown::state::{GraphBetaEntry, GraphBetaState};

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds();

    let history = mixed_constraint_history();
    let context = GraphCrownContext::for_history(&history);

    let (zero_beta_output, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("zero-beta propagation should succeed");

    for sign in [1.0_f32, -1.0_f32] {
        let beta_state = GraphBetaState {
            entries: vec![GraphBetaEntry {
                node_name: "relu1".to_string(),
                neuron_idx: 0,
                split_point: 0.0,
                value: f32::INFINITY,
                sign,
                grad: 0.0,
                m: 0.0,
                v: 0.0,
                v_max: 0.0,
            }],
            ..GraphBetaState::empty()
        };

        let (inf_beta_output, _) = verifier
            .propagate_crown_with_graph_constraints(
                &graph,
                &input,
                &context,
                Some(&beta_state),
                None,
            )
            .expect("non-finite beta should be skipped, not fail");

        assert_bounded_tensor_close(
            &inf_beta_output,
            &zero_beta_output,
            TOL,
            &format!("non-finite beta should match zero-beta baseline (sign={sign})"),
        );
    }
}

#[test]
fn test_objective_coefficients_produce_scalar_output() {
    // With a custom objective vector [1, -1], the output should be a scalar
    // representing the weighted combination of the 2-neuron output.
    //
    // For the two-neuron graph with identity linear1 and sum linear2 (2→1),
    // the network output is already 1D. But the objective path exercises a
    // different initialization in backward.rs (lines 107-131): building A from
    // the objective vector rather than identity.
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_two_neuron_relu_graph();
    let input = build_two_neuron_input_bounds();

    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    // Network output is 1D (from the sum layer), so objective must be length 1.
    let objective = [1.0f32];

    let (output_with_obj, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, Some(&objective))
        .expect("objective propagation should succeed");

    let (output_no_obj, _) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("no-objective propagation should succeed");

    // With objective [1.0] on a 1D output, result should match identity (no-objective).
    assert_bounded_tensor_close(
        &output_with_obj,
        &output_no_obj,
        TOL,
        "objective=[1.0] should match identity on 1D output",
    );
}
