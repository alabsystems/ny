// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Clip-in-alpha CROWN and forward-linear-bounds regressions (#3776, #3438, #3813 warm-start).

use std::collections::HashMap;

use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::{GraphCrownContext, GraphSplitHistory};
use crate::{BetaCrownConfig, BetaCrownVerifier, BoundedTensor};

use ndarray::arr1;

use super::support::{
    active_relu_history, assert_cache_bounds_close, assert_scalar_bounds, build_input_bounds,
    build_single_relu_graph, build_two_relu_clip_graph, clip_test_history, scalar_interval,
};
use super::TOL;

use ny_test_utils::assert_bounded_tensor_close;
#[test]
fn test_clip_in_alpha_crown_tightens_graph_split_cache_3776() {
    let graph = build_two_relu_clip_graph();
    let input = build_input_bounds();
    let history = clip_test_history();
    let context = GraphCrownContext::for_history(&history);

    let baseline_verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let (baseline_output, baseline_cache) = baseline_verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("baseline constrained propagation should succeed");

    let clip_config = BetaCrownConfig {
        clip_in_alpha_crown: true,
        clip_interm_topk: 1,
        ..BetaCrownConfig::default()
    };
    let clip_verifier = BetaCrownVerifier::new(clip_config);
    let (clipped_output, clipped_cache) = clip_verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("clip_in_alpha_crown propagation should succeed");

    assert_bounded_tensor_close(&baseline_output, &clipped_output, TOL, "output parity");

    assert_scalar_bounds(
        baseline_cache
            .get("linear1")
            .expect("baseline linear1 bounds should exist"),
        0.0,
        1.0,
        "baseline linear1",
    );
    assert_scalar_bounds(
        clipped_cache
            .get("linear1")
            .expect("clipped linear1 bounds should exist"),
        0.0,
        0.2,
        "clipped linear1",
    );
    assert_scalar_bounds(
        baseline_cache
            .get("relu1")
            .expect("baseline relu1 bounds should exist"),
        0.0,
        1.0,
        "baseline relu1",
    );
    assert_scalar_bounds(
        clipped_cache
            .get("relu1")
            .expect("clipped relu1 bounds should exist"),
        0.0,
        0.2,
        "clipped relu1",
    );
    assert_scalar_bounds(
        clipped_cache
            .get("linear2")
            .expect("clipped linear2 bounds should exist"),
        -0.2,
        0.0,
        "clipped linear2",
    );
}

#[test]
fn test_clip_in_alpha_crown_storing_intermediates_matches_standard_3776() {
    let graph = build_two_relu_clip_graph();
    let input = build_input_bounds();
    let history = clip_test_history();
    let context = GraphCrownContext::for_history(&history);

    let config = BetaCrownConfig {
        clip_in_alpha_crown: true,
        clip_interm_topk: 1,
        ..BetaCrownConfig::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (standard_output, standard_cache) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("standard clip_in_alpha_crown propagation should succeed");
    let (intermediate_output, intermediate_cache, intermediate) = verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect("intermediate clip_in_alpha_crown propagation should succeed");

    assert_bounded_tensor_close(
        &standard_output,
        &intermediate_output,
        TOL,
        "clip_in_alpha_crown output parity",
    );
    assert_cache_bounds_close(
        &standard_cache,
        &intermediate_cache,
        "clip_in_alpha_crown cache parity",
    );
    assert!(
        intermediate.a_at_relu.contains_key("relu1"),
        "storing-intermediates clip path must still capture relu1 A"
    );
    assert!(
        intermediate.a_at_relu.contains_key("relu2"),
        "storing-intermediates clip path must still capture relu2 A"
    );
}

#[test]
fn test_clip_in_alpha_crown_is_noop_without_split_history_3776() {
    let graph = build_two_relu_clip_graph();
    let input = build_input_bounds();
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let baseline_verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let (baseline_output, baseline_cache) = baseline_verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("baseline propagation without split history should succeed");

    let clip_config = BetaCrownConfig {
        clip_in_alpha_crown: true,
        clip_interm_topk: 1,
        ..BetaCrownConfig::default()
    };
    let clip_verifier = BetaCrownVerifier::new(clip_config);
    let (clipped_output, clipped_cache) = clip_verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, None)
        .expect("clip_in_alpha_crown should be a no-op without split history");

    assert_bounded_tensor_close(
        &baseline_output,
        &clipped_output,
        TOL,
        "clip_in_alpha_crown no-op output parity",
    );
    assert_cache_bounds_close(
        &baseline_cache,
        &clipped_cache,
        "clip_in_alpha_crown no-op cache parity",
    );
}

#[test]
fn test_clip_in_alpha_crown_storing_intermediates_is_noop_without_split_history_3776() {
    let graph = build_two_relu_clip_graph();
    let input = build_input_bounds();
    let history = GraphSplitHistory::new();
    let context = GraphCrownContext::for_history(&history);

    let baseline_verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let (baseline_output, baseline_cache, baseline_intermediate) = baseline_verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect("baseline storing-intermediates without split history should succeed");

    let clip_config = BetaCrownConfig {
        clip_in_alpha_crown: true,
        clip_interm_topk: 1,
        ..BetaCrownConfig::default()
    };
    let clip_verifier = BetaCrownVerifier::new(clip_config);
    let (clipped_output, clipped_cache, clipped_intermediate) = clip_verifier
        .propagate_crown_with_graph_constraints_storing_intermediates(
            &graph, &input, &context, None, None,
        )
        .expect(
            "clip_in_alpha_crown storing-intermediates should be a no-op without split history",
        );

    assert_bounded_tensor_close(
        &baseline_output,
        &clipped_output,
        TOL,
        "clip_in_alpha_crown no-op storing output parity",
    );
    assert_cache_bounds_close(
        &baseline_cache,
        &clipped_cache,
        "clip_in_alpha_crown no-op storing cache parity",
    );
    assert!(
        baseline_intermediate.a_at_relu.is_empty()
            && baseline_intermediate.pre_relu_bounds.is_empty(),
        "baseline no-history storing path should not capture constrained intermediates",
    );
    assert!(
        clipped_intermediate.a_at_relu.is_empty()
            && clipped_intermediate.pre_relu_bounds.is_empty(),
        "clip_in_alpha_crown no-history storing path should stay a no-op",
    );
}

/// Assert forward linear bounds for a single 1-D node: lA[0,0], lb[0], uA[0,0], ub[0].
fn assert_forward_bounds_1d(
    fwd: &crate::batched_domain::CachedLinearBounds,
    node: &str,
    expected_la: f32,
    expected_lb: f32,
    expected_ua: f32,
    expected_ub: f32,
) {
    let lb = fwd
        .linear_bounds(node)
        .unwrap_or_else(|| panic!("{node} forward bounds missing"));
    assert!(
        (lb.lower_a()[[0, 0]] - expected_la).abs() <= TOL,
        "{node} lA={}",
        lb.lower_a()[[0, 0]]
    );
    assert!(
        (lb.lower_b()[0] - expected_lb).abs() <= TOL,
        "{node} lb={}",
        lb.lower_b()[0]
    );
    assert!(
        (lb.upper_a()[[0, 0]] - expected_ua).abs() <= TOL,
        "{node} uA={}",
        lb.upper_a()[[0, 0]]
    );
    assert!(
        (lb.upper_b()[0] - expected_ub).abs() <= TOL,
        "{node} ub={}",
        lb.upper_b()[0]
    );
}

#[test]
fn test_forward_linear_bounds_computation_3776() {
    use crate::beta_crown::engine::graph::clip_alpha::compute_forward_linear_bounds;

    let graph = build_two_relu_clip_graph();
    let input = build_input_bounds();
    let history = clip_test_history();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let (bounds_cache, constrained_input) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("constrained forward bounds should succeed");
    let exec_order = graph.exec_order().expect("topological sort should succeed");

    let fwd = compute_forward_linear_bounds(
        &graph,
        &history,
        exec_order,
        &bounds_cache,
        &constrained_input,
    )
    .expect("forward linear bounds should succeed");

    // linear1: identity through W=1,b=0
    assert_forward_bounds_1d(&fwd, "linear1", 1.0, 0.0, 1.0, 0.0);
    // relu1 (constrained active): identity
    assert_forward_bounds_1d(&fwd, "relu1", 1.0, 0.0, 1.0, 0.0);
    // linear2: W=1, b=-0.2
    assert_forward_bounds_1d(&fwd, "linear2", 1.0, -0.2, 1.0, -0.2);
    // relu2 (constrained inactive): zero
    assert_forward_bounds_1d(&fwd, "relu2", 0.0, 0.0, 0.0, 0.0);
}

#[test]
fn test_forward_linear_bounds_unconstrained_relu_uses_triangle_relaxation_3776() {
    use crate::beta_crown::engine::graph::clip_alpha::compute_forward_linear_bounds;

    let graph = build_single_relu_graph();
    let input = build_input_bounds();
    let history = GraphSplitHistory::new();

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let (bounds_cache, constrained_input) = verifier
        .compute_constrained_forward_bounds(&graph, &input, &history, None, None)
        .expect("unconstrained forward bounds should succeed");
    let exec_order = graph.exec_order().expect("topological sort should succeed");

    let fwd = compute_forward_linear_bounds(
        &graph,
        &history,
        exec_order,
        &bounds_cache,
        &constrained_input,
    )
    .expect("forward linear bounds should succeed");

    // linear1: identity through W=1,b=0
    assert_forward_bounds_1d(&fwd, "linear1", 1.0, 0.0, 1.0, 0.0);
    // relu1 on [-1, 1] uses the triangle relaxation:
    // lower = 0, upper = 0.5 * x + 0.5.
    assert_forward_bounds_1d(&fwd, "relu1", 0.0, 0.0, 0.5, 0.5);
    // linear2 is identity after the relaxed relu1 bounds.
    assert_forward_bounds_1d(&fwd, "linear2", 0.0, 0.0, 0.5, 0.5);
}

#[test]
fn test_forward_linear_bounds_relu_nonfinite_falls_back_3438() {
    use crate::beta_crown::engine::graph::clip_alpha::compute_forward_linear_bounds;

    let graph = build_single_relu_graph();
    let input = build_input_bounds();
    let history = GraphSplitHistory::new();
    let exec_order = graph.exec_order().expect("topological sort should succeed");
    let bounds_cache = HashMap::from([(
        "linear1".to_string(),
        BoundedTensor::new_unchecked(
            arr1(&[-1.0_f32]).into_dyn(),
            arr1(&[f32::INFINITY]).into_dyn(),
        )
        .expect("pre-activation bounds should have matching shape"),
    )]);

    let fwd = compute_forward_linear_bounds(&graph, &history, exec_order, &bounds_cache, &input)
        .expect("forward linear bounds should succeed");

    assert_forward_bounds_1d(&fwd, "linear1", 1.0, 0.0, 1.0, 0.0);

    let relu_bounds = fwd
        .linear_bounds("relu1")
        .expect("relu1 forward bounds should exist");
    assert_eq!(
        relu_bounds.lower_a()[[0, 0]],
        0.0,
        "relu1 lower A should fall back to conservative zero coefficients"
    );
    assert_eq!(
        relu_bounds.upper_a()[[0, 0]],
        0.0,
        "relu1 upper A should fall back to conservative zero coefficients"
    );
    assert!(
        relu_bounds.lower_b()[0].is_infinite() && relu_bounds.lower_b()[0].is_sign_negative(),
        "relu1 lower bias should fall back to -Inf, got {}",
        relu_bounds.lower_b()[0]
    );
    assert!(
        relu_bounds.upper_b()[0].is_infinite() && relu_bounds.upper_b()[0].is_sign_positive(),
        "relu1 upper bias should fall back to +Inf, got {}",
        relu_bounds.upper_b()[0]
    );
}

#[test]
fn test_constrained_backward_warm_start_seeds_last_branch_node_3813() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let graph = build_single_relu_graph();
    let input = build_input_bounds();
    let history = active_relu_history();
    let context = GraphCrownContext::for_history(&history);
    let objective = [1.0_f32];

    let (_baseline_output, _baseline_cache, baseline_la) = verifier
        .propagate_crown_with_graph_constraints_with_cache(
            &graph,
            &input,
            &context,
            None,
            Some(&objective),
            None,
            true,
        )
        .expect("baseline constrained backward with capture should succeed");
    let baseline_la = baseline_la.expect("baseline constrained backward should capture lA");
    assert!(
        baseline_la.linear_bounds("linear2").is_some(),
        "baseline capture should include nodes above the branch point"
    );

    let seed_cache = crate::batched_domain::CachedLinearBounds::from_linear_bounds_map(
        HashMap::from([("relu1".to_string(), crate::LinearBounds::identity(1))]),
    );
    let (warm_output, _warm_cache, warm_la) = verifier
        .propagate_crown_with_graph_constraints_with_cache(
            &graph,
            &input,
            &context,
            None,
            Some(&objective),
            Some(&seed_cache),
            true,
        )
        .expect("warm-start constrained backward with capture should succeed");

    assert!(
        warm_output.lower_scalar().is_finite() && warm_output.upper_scalar().is_finite(),
        "warm-start constrained backward should still produce finite objective bounds"
    );

    let warm_la = warm_la.expect("warm-start constrained backward should capture lA");
    assert!(
        warm_la.linear_bounds("relu1").is_some(),
        "warm-start capture should still include the branch node"
    );
    assert!(
        warm_la.linear_bounds("linear2").is_none(),
        "warm-start capture should skip nodes above the last branch node"
    );
}

/// Multi-objective disjunctive regression: clip_in_alpha_crown tightens
/// intermediate bounds identically regardless of objective vector.
///
/// In joint multi-objective verification (cora-style), all objectives share the
/// same graph topology and split constraints. clip_in_alpha_crown operates on
/// intermediate nodes (objective-independent), so the tightened cache must be
/// identical across different objectives. This proves the existing joint
/// multi-objective route is preserved — if per-clause decomposition were used,
/// each clause would have a different constraint set and different tightening.
///
/// Part of #3776.
#[test]
fn test_clip_in_alpha_crown_multi_objective_shared_tightening_3776() {
    let graph = build_two_relu_clip_graph();
    let input = build_input_bounds();
    let history = clip_test_history();
    let context = GraphCrownContext::for_history(&history);

    let config = BetaCrownConfig {
        clip_in_alpha_crown: true,
        clip_interm_topk: 1,
        ..BetaCrownConfig::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // Objective 1: identity (output > threshold)
    let (output_pos, cache_pos) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, Some(&[1.0]))
        .expect("clip_in_alpha_crown with positive objective should succeed");

    // Objective 2: negated (−output > threshold)
    let (output_neg, cache_neg) = verifier
        .propagate_crown_with_graph_constraints(&graph, &input, &context, None, Some(&[-1.0]))
        .expect("clip_in_alpha_crown with negative objective should succeed");

    // Intermediate bounds must be identical: clip_in_alpha_crown operates on
    // node bounds independently of the objective vector. Shared tightening
    // confirms joint multi-objective processing (not per-clause decomposition).
    assert_cache_bounds_close(
        &cache_pos,
        &cache_neg,
        "clip_in_alpha_crown multi-objective cache parity",
    );

    // Verify output bounds are valid (non-NaN, lower <= upper).
    // With clip_test_history constraining relu2 inactive, both objectives produce
    // near-zero output — that's expected for a fully-constrained graph.
    let (pos_l, pos_u) = scalar_interval(&output_pos);
    let (neg_l, neg_u) = scalar_interval(&output_neg);
    assert!(
        pos_l.is_finite() && pos_u.is_finite(),
        "positive obj output must be finite"
    );
    assert!(
        neg_l.is_finite() && neg_u.is_finite(),
        "negative obj output must be finite"
    );
    assert!(
        pos_l <= pos_u + TOL,
        "positive obj: lower({pos_l}) <= upper({pos_u})"
    );
    assert!(
        neg_l <= neg_u + TOL,
        "negative obj: lower({neg_l}) <= upper({neg_u})"
    );
}
