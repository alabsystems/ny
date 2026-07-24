// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constrained forward-tightening regressions (#1901).

use crate::beta_crown::{GraphCrownContext, GraphSplitHistory};
use crate::{BetaCrownConfig, BetaCrownVerifier};

use super::support::{
    active_relu_history, assert_cache_bounds_close, assert_scalar_bounds, build_input_bounds,
    build_single_relu_graph, inactive_relu_history, scalar_interval,
};
use super::TOL;

use ny_test_utils::assert_bounded_tensor_close;
#[test]
fn test_compute_constrained_forward_bounds_tightens_relu_pipeline_1901() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_single_relu_graph();
    let input = build_input_bounds();
    let history = inactive_relu_history();

    let (cache, constrained_input) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("forward constrained bounds should succeed");

    assert_bounded_tensor_close(&constrained_input, &input, TOL, "constrained_input");
    assert_scalar_bounds(
        cache
            .get("linear1")
            .expect("linear1 bounds should exist in forward cache"),
        -1.0,
        0.0,
        "linear1",
    );
    assert_scalar_bounds(
        cache
            .get("relu1")
            .expect("relu1 bounds should exist in forward cache"),
        0.0,
        0.0,
        "relu1",
    );
    assert_scalar_bounds(
        cache
            .get("linear2")
            .expect("linear2 bounds should exist in forward cache"),
        0.0,
        0.0,
        "linear2",
    );
}

#[test]
fn test_propagate_crown_with_graph_constraints_is_tighter_than_unconstrained_1901() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_single_relu_graph();
    let input = build_input_bounds();
    let unconstrained_history = GraphSplitHistory::new();
    let constrained_history = inactive_relu_history();
    let unconstrained_context = GraphCrownContext::for_history(&unconstrained_history);
    let constrained_context = GraphCrownContext::for_history(&constrained_history);

    let (unconstrained_output, unconstrained_cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &unconstrained_context, None, None)
        .expect("unconstrained propagation should succeed");
    let (constrained_output, constrained_cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &constrained_context, None, None)
        .expect("constrained propagation should succeed");

    let (unconstrained_lower, unconstrained_upper) = scalar_interval(&unconstrained_output);
    let (constrained_lower, constrained_upper) = scalar_interval(&constrained_output);

    assert!(
        constrained_lower >= unconstrained_lower - TOL,
        "constrained lower bound must not be looser: unconstrained={}, constrained={}",
        unconstrained_lower,
        constrained_lower
    );
    assert!(
        constrained_upper <= unconstrained_upper + TOL,
        "constrained upper bound must not be looser: unconstrained={}, constrained={}",
        unconstrained_upper,
        constrained_upper
    );
    assert!(
        constrained_upper < unconstrained_upper - 1e-4,
        "inactive ReLU split should strictly tighten upper bound: unconstrained={}, constrained={}",
        unconstrained_upper,
        constrained_upper
    );

    let (_, unconstrained_relu_upper) = scalar_interval(
        unconstrained_cache
            .get("relu1")
            .expect("relu1 bounds should exist in unconstrained cache"),
    );
    let (_, constrained_relu_upper) = scalar_interval(
        constrained_cache
            .get("relu1")
            .expect("relu1 bounds should exist in constrained cache"),
    );
    assert!(
        constrained_relu_upper < unconstrained_relu_upper - 1e-4,
        "relu1 upper bound should tighten under inactive constraint: unconstrained={}, constrained={}",
        unconstrained_relu_upper,
        constrained_relu_upper
    );
}

#[test]
fn test_storing_intermediates_matches_standard_and_forward_cache_1901() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_single_relu_graph();
    let input = build_input_bounds();
    let history = inactive_relu_history();
    let context = GraphCrownContext::for_history(&history);

    let (forward_cache, _constrained_input) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("forward constrained bounds should succeed");
    let (standard_output, standard_cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("standard constrained propagation should succeed");
    let (intermediate_output, intermediate_cache, intermediate) = verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect("intermediate constrained propagation should succeed");

    assert_bounded_tensor_close(&standard_output, &intermediate_output, TOL, "output parity");
    assert_cache_bounds_close(&forward_cache, &standard_cache, "forward_vs_standard");
    assert_cache_bounds_close(
        &forward_cache,
        &intermediate_cache,
        "forward_vs_intermediate",
    );

    assert!(
        intermediate.a_at_relu.contains_key("relu1"),
        "intermediate pass must capture A matrix for constrained relu1"
    );
    let pre_relu_bounds = intermediate
        .pre_relu_bounds
        .get("relu1")
        .expect("intermediate pass must capture pre-ReLU bounds for relu1");
    assert!(
        (pre_relu_bounds.0[0] - (-1.0)).abs() <= TOL,
        "captured relu1 pre-lower should match constrained forward bounds, got {}",
        pre_relu_bounds.0[0]
    );
    assert!(
        (pre_relu_bounds.1[0] - 0.0).abs() <= TOL,
        "captured relu1 pre-upper should match constrained forward bounds, got {}",
        pre_relu_bounds.1[0]
    );
}

/// Verify active constraint tightening: active ReLU constraint forces pre-activation
/// lower bound to max(l, 0) = 0.0 and keeps upper bound unchanged. Output should be
/// strictly tighter than the unconstrained case (lower bound moves from -1.0 to 0.0).
///
/// This complements the inactive-only tests above by covering the other branch of the
/// constraint match (Some(true) vs Some(false)).
#[test]
fn test_active_constraint_tightens_forward_bounds_1901() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_single_relu_graph();
    let input = build_input_bounds();
    let history = active_relu_history();

    let (cache, _constrained_input) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("active-constrained forward bounds should succeed");

    // linear1 produces input passed through (weight=I, bias=0): bounds = [-1, 1].
    // Active constraint on relu1 forces pre-activation lower to max(-1, 0) = 0.
    // So relu1 output = [0, 1] (the identity in the active region).
    assert_scalar_bounds(
        cache
            .get("relu1")
            .expect("relu1 bounds should exist in forward cache"),
        0.0,
        1.0,
        "relu1 (active)",
    );

    // linear2 (weight=I, bias=0) outputs relu1 bounds unchanged: [0, 1].
    assert_scalar_bounds(
        cache
            .get("linear2")
            .expect("linear2 bounds should exist in forward cache"),
        0.0,
        1.0,
        "linear2 (active)",
    );

    // Compare with unconstrained: unconstrained relu1 has bounds [0, 1] (same as active
    // constraint, since ReLU naturally clips negative pre-activation). But the unconstrained
    // forward pass intersects with CROWN-IBP, which may give different intermediate bounds.
    // The key check: active constraint is at least as tight as unconstrained.
    let unconstrained_history = GraphSplitHistory::new();
    let (unconstrained_cache, _) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &unconstrained_history, None, None)
        .expect("unconstrained forward bounds should succeed");
    let (_, unconstrained_relu_upper) = scalar_interval(
        unconstrained_cache
            .get("relu1")
            .expect("relu1 bounds should exist in unconstrained cache"),
    );
    let (constrained_relu_lower, constrained_relu_upper) = scalar_interval(
        cache
            .get("relu1")
            .expect("relu1 bounds should exist in constrained cache"),
    );
    assert!(
        constrained_relu_lower >= 0.0 - TOL,
        "active-constrained relu1 lower must be >= 0: got {}",
        constrained_relu_lower
    );
    assert!(
        constrained_relu_upper <= unconstrained_relu_upper + TOL,
        "active-constrained relu1 upper must not be looser than unconstrained: \
         constrained={}, unconstrained={}",
        constrained_relu_upper,
        unconstrained_relu_upper
    );
}
