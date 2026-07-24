// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// NOTE: split from tests.rs for maintainability.

use super::prelude::*;

// Relaxed Clipping Tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_relaxed_clipping_integration() {
    // Test that relaxed clipping is wired correctly into input split child creation.
    // Uses a simple network where clipping should tighten bounds.
    use crate::beta_crown::branching::BranchingHeuristic;

    // Network: Linear(2->1) with weights [1, 1] and bias -5
    // Output = x1 + x2 - 5
    // Constraint: x1 + x2 <= 5
    let w = arr2(&[[1.0, 1.0]]);
    let b = arr1(&[-5.0]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    // Input: x in [0, 10]^2
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 10.0]).unwrap(),
    )
    .unwrap();

    // Config with relaxed clipping enabled
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 10,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    // Create a parent domain with initial bounds
    let parent = BabDomain {
        history: SplitHistory::new(),
        lower_bound: -5.0, // Output range: [0+0-5, 10+10-5] = [-5, 15]
        upper_bound: 15.0,
        priority: -5.0,
        layer_bounds: vec![Arc::new(input.clone())],
        alpha_state: None,
        domain_alpha_state: DomainAlphaState::empty(),
        beta_state: BetaState::empty(),
        input_bounds: Some(Arc::new(input.clone())),
        input_split_count: 0,
        intermediate_bounds: IntermediateLinearBounds::empty(),
    };

    // Call create_input_split_child with relaxed clipping enabled
    // Split dimension 0 at midpoint: [0, 5] for left child
    let child = verifier
        .create_input_split_child(&network, &input, &parent, 0, 0.0, 5.0, 0.0, None, None)
        .unwrap();

    // Verify child was created
    assert!(child.is_some(), "Child domain should be created");

    let child = child.unwrap();

    // Verify input bounds are stored
    assert!(
        child.input_bounds.is_some(),
        "Child should have input bounds"
    );

    // Check that bounds were tightened by relaxed clipping
    // The linear constraint x1 + x2 - 5 <= 0 means x1 + x2 <= 5
    // With x1 in [0, 5], x2 should be clipped to at most 5 (not 10)
    let child_bounds = child.input_bounds.as_ref().unwrap();
    let flat = child_bounds.flatten();

    // x1 should be in [0, 5] (from split)
    assert!(
        flat.lower()[[0]] >= -0.01 && flat.lower()[[0]] <= 0.01,
        "x1 lower should be ~0, got {}",
        flat.lower()[[0]]
    );
    assert!(
        flat.upper()[[0]] >= 4.99 && flat.upper()[[0]] <= 5.01,
        "x1 upper should be ~5, got {}",
        flat.upper()[[0]]
    );

    // x2 may be tightened by relaxed clipping (depends on constraint)
    // With positive coefficient on x2, relaxed clipping should shrink upper bound
    // to satisfy x1 + x2 <= 5 across the child domain.
    assert!(
        flat.lower()[[1]] <= flat.upper()[[1]],
        "x2 bounds should be valid"
    );
    assert!(
        flat.upper()[[1]] <= 5.01,
        "x2 upper should be <= 5 after clipping, got {}",
        flat.upper()[[1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxed_clipping_skips_verified_child() {
    // Test that relaxed clipping short-circuits when a child is already verified.
    use crate::beta_crown::branching::BranchingHeuristic;

    // Network: Linear(2->1) with weights [1, 1], bias 0.
    let w = arr2(&[[1.0, 1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    // Input: x in [0, 1]^2
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    // Config with relaxed clipping enabled
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 10,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    // Parent domain (bounds are already verified for threshold = -1.0).
    let parent = BabDomain {
        history: SplitHistory::new(),
        lower_bound: 0.0,
        upper_bound: 2.0,
        priority: 0.0,
        layer_bounds: vec![Arc::new(input.clone())],
        alpha_state: None,
        domain_alpha_state: DomainAlphaState::empty(),
        beta_state: BetaState::empty(),
        input_bounds: Some(Arc::new(input.clone())),
        input_split_count: 0,
        intermediate_bounds: IntermediateLinearBounds::empty(),
    };

    let child = verifier
        .create_input_split_child(&network, &input, &parent, 0, 0.0, 0.5, -1.0, None, None)
        .unwrap();

    assert!(child.is_none(), "Verified child should be skipped");
}

#[ntest::timeout(10000)]
#[test]
fn test_relaxed_clipping_disabled_preserves_bounds() {
    // Test that with relaxed clipping disabled, bounds are not changed.
    use crate::beta_crown::branching::BranchingHeuristic;

    let w = arr2(&[[1.0, 1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 10.0]).unwrap(),
    )
    .unwrap();

    // Config with relaxed clipping DISABLED
    let config = BetaCrownConfig {
        enable_relaxed_clip: false,
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 10,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    let parent = BabDomain {
        history: SplitHistory::new(),
        lower_bound: 0.0,
        upper_bound: 20.0,
        priority: 0.0,
        layer_bounds: vec![Arc::new(input.clone())],
        alpha_state: None,
        domain_alpha_state: DomainAlphaState::empty(),
        beta_state: BetaState::empty(),
        input_bounds: Some(Arc::new(input.clone())),
        input_split_count: 0,
        intermediate_bounds: IntermediateLinearBounds::empty(),
    };

    let child = verifier
        .create_input_split_child(&network, &input, &parent, 0, 0.0, 5.0, 0.0, None, None)
        .unwrap()
        .unwrap();

    let child_bounds = child.input_bounds.as_ref().unwrap();
    let flat = child_bounds.flatten();

    // With clipping disabled, x2 should remain [0, 10] (not tightened)
    assert!(
        (flat.lower()[[1]] - 0.0).abs() < 0.01,
        "x2 lower should be 0, got {}",
        flat.lower()[[1]]
    );
    assert!(
        (flat.upper()[[1]] - 10.0).abs() < 0.01,
        "x2 upper should be 10, got {}",
        flat.upper()[[1]]
    );
}

fn single_input_identity_network() -> Network {
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));
    network
}

fn single_input_identity_graph() -> GraphNetwork {
    let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).unwrap();
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("id", Layer::Linear(linear)));
    graph.set_output("id");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_relaxed_clipping_verify_upper_bound_preverified_uses_upper_mode() {
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        verify_upper_bound: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let network = single_input_identity_network();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap(),
    )
    .unwrap();

    let outcome = verifier
        .apply_relaxed_clipping(&network, input, &[1], 0.2, None)
        .unwrap();

    assert!(
        outcome.verified,
        "verify_upper_bound must normalize upper(x) < threshold into lower-bound form; reusing lower > threshold would fail on x in [0, 0.1]"
    );
    let flat = outcome.bounds.flatten();
    assert_eq!(flat.lower()[[0]], 0.0);
    assert_eq!(flat.upper()[[0]], 0.1);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_relaxed_clipping_graph_verify_upper_bound_preverified_uses_upper_mode() {
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        verify_upper_bound: true,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap(),
    )
    .unwrap();

    let outcome = verifier
        .apply_relaxed_clipping_graph(&graph, &input, &[1], &[1.0], 0.2, None)
        .unwrap();

    assert!(
        outcome.verified,
        "graph relaxed clipping must use upper-bound normalization when verify_upper_bound=true"
    );
    let flat = outcome.bounds.flatten();
    assert_eq!(flat.lower()[[0]], 0.0);
    assert_eq!(flat.upper()[[0]], 0.1);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_relaxed_clipping_graph_empty_objective_is_noop() {
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let graph = single_input_identity_graph();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    let outcome = verifier
        .apply_relaxed_clipping_graph(&graph, &input, &[1], &[], 0.0, None)
        .unwrap();

    assert!(
        !outcome.verified,
        "empty objectives should bypass graph relaxed clipping instead of claiming verification"
    );
    let flat = outcome.bounds.flatten();
    assert_eq!(flat.lower()[[0]], 0.0);
    assert_eq!(flat.upper()[[0]], 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_any_verified_shape_mismatch_returns_false() {
    let dm_lb = arr2(&[[0.3_f32, -0.1_f32]]);
    let thresholds = arr2(&[[0.0_f32]]);

    assert!(
        !BetaCrownVerifier::any_verified(&dm_lb, &thresholds),
        "shape mismatches must not claim verification"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_precomputed_clip_verifies_infeasible_domain() {
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    let linear_bounds = LinearBounds::new(
        arr2(&[[1.0], [-1.0]]),
        arr1(&[-0.2, 0.8]),
        arr2(&[[1.0], [-1.0]]),
        arr1(&[-0.2, 0.8]),
    )
    .unwrap();

    let outcome = verifier
        .clip_multi_objective_with_precomputed_linear(&input, &[1], &linear_bounds, &[0.0, 0.0])
        .unwrap();

    assert!(
        outcome.verified,
        "joint multi-spec clipping should verify an infeasible child domain"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_precomputed_clip_threshold_len_mismatch_errors() {
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    let linear_bounds = LinearBounds::new(
        arr2(&[[1.0], [-1.0]]),
        arr1(&[-0.2, 0.8]),
        arr2(&[[1.0], [-1.0]]),
        arr1(&[-0.2, 0.8]),
    )
    .unwrap();

    let err = match verifier.clip_multi_objective_with_precomputed_linear(
        &input,
        &[1],
        &linear_bounds,
        &[0.0],
    ) {
        Err(err) => err,
        Ok(_) => panic!("threshold/row count mismatch should return an error"),
    };
    let msg = err.to_string().to_lowercase();

    assert!(
        msg.contains("shape") || msg.contains("mismatch"),
        "threshold/row count mismatch should surface as a shape error, got: {msg}"
    );
    assert!(
        msg.contains('2') && msg.contains('1'),
        "error should mention both expected row count and provided threshold count, got: {msg}"
    );
}

// Grouped-safe Multi-Spec Clipping Tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_grouped_safe_clip_returns_per_row_lower_bounds() {
    // Two spec rows with the same infeasible domain as the conjunctive test.
    // Verify that grouped_safe returns per-row lower bounds instead of a scalar.
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // 1D input [0, 1]
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    // Row 0: x >= 0.2 (coeff=1, bias=-0.2) → lower bound = x_min - 0.2
    // Row 1: -x >= -0.8 (coeff=-1, bias=0.8) → lower bound = -x_max + 0.8
    // Together these make x in [0.2, 0.8], which after clipping makes the box empty.
    let linear_bounds = LinearBounds::new(
        arr2(&[[1.0], [-1.0]]),
        arr1(&[-0.2, 0.8]),
        arr2(&[[1.0], [-1.0]]),
        arr1(&[-0.2, 0.8]),
    )
    .unwrap();

    let outcome = verifier
        .clip_multi_objective_grouped_safe(&input, &[1], &linear_bounds, &[0.0, 0.0])
        .unwrap();

    // Joint clipping should detect infeasibility.
    assert!(
        outcome.infeasible_after_clip,
        "grouped-safe clip should detect infeasible child domain"
    );
    // Per-row lower bounds should be returned (length matches spec rows).
    assert_eq!(
        outcome.postclip_lower_bounds.len(),
        2,
        "should return one lower bound per spec row"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_safe_clip_non_infeasible_returns_tightened_bounds() {
    // Wide domain where clipping tightens but doesn't make infeasible.
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // 1D input [0, 10]
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).unwrap(),
    )
    .unwrap();

    // Single row: x >= 3 (coeff=1, bias=-3) → should tighten lower to ~3
    let linear_bounds =
        LinearBounds::new(arr2(&[[1.0]]), arr1(&[-3.0]), arr2(&[[1.0]]), arr1(&[-3.0])).unwrap();

    let outcome = verifier
        .clip_multi_objective_grouped_safe(&input, &[1], &linear_bounds, &[0.0])
        .unwrap();

    assert!(
        !outcome.infeasible_after_clip,
        "wide domain should not be infeasible after single-row clip"
    );
    assert_eq!(outcome.postclip_lower_bounds.len(), 1);
    // Per-row lower bounds must be finite.
    assert!(
        outcome.postclip_lower_bounds[0].is_finite(),
        "postclip lower bound should be finite, got {}",
        outcome.postclip_lower_bounds[0]
    );
    // Clipped bounds should remain valid (lower <= upper).
    let flat = outcome.bounds.flatten();
    assert!(
        flat.lower()[[0]] <= flat.upper()[[0]],
        "clipped bounds should be valid: lower {} <= upper {}",
        flat.lower()[[0]],
        flat.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_grouped_safe_clip_empty_rows_returns_unchanged() {
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        relaxed_clip_iterations: 1,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    // Zero rows in linear bounds
    let linear_bounds = LinearBounds::new(
        arr2(&[[] as [f32; 0]]).t().to_owned(),
        arr1(&[] as &[f32]),
        arr2(&[[] as [f32; 0]]).t().to_owned(),
        arr1(&[] as &[f32]),
    )
    .unwrap();

    let outcome = verifier
        .clip_multi_objective_grouped_safe(&input, &[1], &linear_bounds, &[])
        .unwrap();

    assert!(!outcome.infeasible_after_clip);
    assert!(outcome.postclip_lower_bounds.is_empty());
}

// Complete Clipping Dispatch Tests
// ============================================================================

#[ntest::timeout(10000)]
#[test]
fn test_complete_clipping_dispatch_creates_valid_child() {
    // Test that InputClipType::Complete dispatches correctly and produces
    // valid (non-NaN, non-empty) child domains.
    use crate::beta_crown::branching::BranchingHeuristic;
    use crate::beta_crown::config::InputClipType;

    // Network: Linear(2->1) with weights [1, 1] and bias -5
    let w = arr2(&[[1.0, 1.0]]);
    let b = arr1(&[-5.0]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 10.0]).unwrap(),
    )
    .unwrap();

    // Config with complete clipping enabled
    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Complete,
        relaxed_clip_iterations: 1,
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 10,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    let parent = BabDomain {
        history: SplitHistory::new(),
        lower_bound: -5.0,
        upper_bound: 15.0,
        priority: -5.0,
        layer_bounds: vec![Arc::new(input.clone())],
        alpha_state: None,
        domain_alpha_state: DomainAlphaState::empty(),
        beta_state: BetaState::empty(),
        input_bounds: Some(Arc::new(input.clone())),
        input_split_count: 0,
        intermediate_bounds: IntermediateLinearBounds::empty(),
    };

    let child = verifier
        .create_input_split_child(&network, &input, &parent, 0, 0.0, 5.0, 0.0, None, None)
        .unwrap();

    // Child should be created (either Some with valid bounds or None if verified)
    if let Some(ref child) = child {
        let child_bounds = child.input_bounds.as_ref().unwrap();
        let flat = child_bounds.flatten();

        // Bounds should be valid (lower <= upper, no NaN)
        assert!(
            flat.lower()[[0]] <= flat.upper()[[0]],
            "x1 bounds should be valid: {} <= {}",
            flat.lower()[[0]],
            flat.upper()[[0]]
        );
        assert!(
            flat.lower()[[1]] <= flat.upper()[[1]],
            "x2 bounds should be valid: {} <= {}",
            flat.lower()[[1]],
            flat.upper()[[1]]
        );
        assert!(
            !flat.lower()[[0]].is_nan() && !flat.upper()[[0]].is_nan(),
            "x1 bounds should not be NaN"
        );
        assert!(
            !flat.lower()[[1]].is_nan() && !flat.upper()[[1]].is_nan(),
            "x2 bounds should not be NaN"
        );

        // Relaxed clipping should have tightened x2 (same as relaxed test)
        assert!(
            flat.upper()[[1]] <= 5.01,
            "x2 upper should be <= 5 after complete clipping, got {}",
            flat.upper()[[1]]
        );
    }
    // If child is None, the domain was verified by clipping — also valid
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clipping_multi_spec_exercises_lp_solver() {
    // Regression test for Prover finding: previous tests used single-spec
    // Linear(2->1) networks, so n_spec == 1 and the LP solver path
    // (guarded by `if n_spec > 1`) was never exercised. This test uses
    // Linear(2->2) to force n_spec == 2.
    use crate::beta_crown::branching::BranchingHeuristic;
    use crate::beta_crown::config::InputClipType;

    // Network: Linear(2->2) with weights [[1, 1], [-1, 1]] and bias [-3, 0]
    // Output spec 0: x1 + x2 - 3
    // Output spec 1: -x1 + x2
    let w = arr2(&[[1.0, 1.0], [-1.0, 1.0]]);
    let b = arr1(&[-3.0, 0.0]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    // Input: x in [0, 5]^2
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0, 5.0]).unwrap(),
    )
    .unwrap();

    let config = BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Complete,
        relaxed_clip_iterations: 1,
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 10,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);

    let parent = BabDomain {
        history: SplitHistory::new(),
        lower_bound: -3.0,
        upper_bound: 7.0,
        priority: -3.0,
        layer_bounds: vec![Arc::new(input.clone())],
        alpha_state: None,
        domain_alpha_state: DomainAlphaState::empty(),
        beta_state: BetaState::empty(),
        input_bounds: Some(Arc::new(input.clone())),
        input_split_count: 0,
        intermediate_bounds: IntermediateLinearBounds::empty(),
    };

    let child = verifier
        .create_input_split_child(&network, &input, &parent, 0, 0.0, 2.5, 0.0, None, None)
        .unwrap();

    // Child should be created with valid bounds (or verified via LP solver)
    if let Some(ref child) = child {
        let child_bounds = child.input_bounds.as_ref().unwrap();
        let flat = child_bounds.flatten();

        // Bounds should be valid: no NaN, lower <= upper
        for i in 0..2 {
            assert!(
                !flat.lower()[[i]].is_nan() && !flat.upper()[[i]].is_nan(),
                "bounds at dim {} must not be NaN",
                i
            );
            assert!(
                flat.lower()[[i]] <= flat.upper()[[i]],
                "bounds at dim {} must be valid: {} <= {}",
                i,
                flat.lower()[[i]],
                flat.upper()[[i]]
            );
        }
    }
    // If child is None, domain was verified by complete clipping — also valid
}

#[ntest::timeout(10000)]
#[test]
fn test_complete_clipping_no_worse_than_relaxed() {
    // Complete clipping should produce bounds at least as tight as relaxed.
    // For single-spec networks, they should be equivalent.
    use crate::beta_crown::branching::BranchingHeuristic;
    use crate::beta_crown::config::InputClipType;

    let w = arr2(&[[1.0, 1.0]]);
    let b = arr1(&[-5.0]);
    let linear = LinearLayer::new(w, Some(b)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![10.0, 10.0]).unwrap(),
    )
    .unwrap();

    let parent = BabDomain {
        history: SplitHistory::new(),
        lower_bound: -5.0,
        upper_bound: 15.0,
        priority: -5.0,
        layer_bounds: vec![Arc::new(input.clone())],
        alpha_state: None,
        domain_alpha_state: DomainAlphaState::empty(),
        beta_state: BetaState::empty(),
        input_bounds: Some(Arc::new(input.clone())),
        input_split_count: 0,
        intermediate_bounds: IntermediateLinearBounds::empty(),
    };

    // Relaxed
    let relaxed_config = BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Relaxed,
        relaxed_clip_iterations: 1,
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 10,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    let relaxed_verifier = BetaCrownVerifier::new(relaxed_config);
    let relaxed_child = relaxed_verifier
        .create_input_split_child(&network, &input, &parent, 0, 0.0, 5.0, 0.0, None, None)
        .unwrap();

    // Complete
    let complete_config = BetaCrownConfig {
        enable_relaxed_clip: true,
        input_clip_type: InputClipType::Complete,
        relaxed_clip_iterations: 1,
        branching_heuristic: BranchingHeuristic::InputSplit,
        max_domains: 10,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    let complete_verifier = BetaCrownVerifier::new(complete_config);
    let complete_child = complete_verifier
        .create_input_split_child(&network, &input, &parent, 0, 0.0, 5.0, 0.0, None, None)
        .unwrap();

    // Both should produce children (this network has loose bounds at threshold=0)
    match (relaxed_child, complete_child) {
        (Some(r), Some(c)) => {
            let r_flat = r.input_bounds.as_ref().unwrap().flatten();
            let c_flat = c.input_bounds.as_ref().unwrap().flatten();

            // Complete bounds should be no looser than relaxed
            // (lower >= relaxed lower, upper <= relaxed upper)
            let r_x2_upper = r_flat.upper()[[1]];
            let c_x2_upper = c_flat.upper()[[1]];
            assert!(
                c_x2_upper <= r_x2_upper + 0.01,
                "complete x2 upper ({}) should be <= relaxed x2 upper ({})",
                c_x2_upper,
                r_x2_upper
            );
        }
        (None, None) => {
            // Both verified — equivalent
        }
        (Some(_), None) => {
            // Complete verified but relaxed didn't — complete is strictly better
        }
        (None, Some(_)) => {
            panic!("Relaxed verified but complete didn't — this should not happen");
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_with_precomputed_linear_coeff_err_blocks_verified() {
    // A row whose raw stored coefficients clear the threshold, but whose
    // certified coefficient-error envelope closes the margin: the clip must
    // not claim `verified` from a bound the true coefficients do not entail
    // (#vnncomp-aw-soundness).
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    // Objective lower bound: 1.0 * x, so the minimum over [0.5, 1.0] is 0.5.
    let mut linear_bounds =
        LinearBounds::new(arr2(&[[1.0]]), arr1(&[0.0]), arr2(&[[1.0]]), arr1(&[0.0])).unwrap();

    // Exact coefficients clear the threshold: 0.5 > 0.4.
    let outcome = verifier
        .clip_with_precomputed_linear(&input, &[1], &linear_bounds, 0, 0.4)
        .unwrap();
    assert!(
        outcome.verified,
        "exact coefficients clear the threshold and must verify"
    );

    // A certified coefficient error of 0.2 admits a true bound as low as
    // 0.5 - 0.2 * max(|x_l|, |x_u|) = 0.3 < 0.4: not provably verified.
    linear_bounds.set_coeff_err(arr2(&[[0.2]]), arr2(&[[0.2]]));
    let outcome = verifier
        .clip_with_precomputed_linear(&input, &[1], &linear_bounds, 0, 0.4)
        .unwrap();
    assert!(
        !outcome.verified,
        "coefficient-error envelope closes the margin; must not claim verified"
    );
}
