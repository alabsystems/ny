// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for `verify_graph_relu_split_with_bounds` — the pre-computed bounds
//! variant of the BaB verification loop.
//!
//! This module has ZERO test coverage in the existing suite. These tests
//! establish basic correctness by cross-validating against the direct
//! `verify_graph_relu_split` path, which is already tested in cutting_planes.rs.
//!
//! Issue: #1892

use super::prelude::*;
use crate::beta_crown::domain::GraphPrecomputedBounds;
use crate::beta_crown::engine::graph::test_non_finite_domain_result_in_relu_split_bounds;
use crate::beta_crown::state::GraphDomainAlphaState;
use ny_test_utils::CountingGemmEngine;

// ============================================================
// Parity tests: with_bounds should match direct path
// ============================================================

/// Test that verify_graph_relu_split_with_bounds returns Verified for a
/// trivially-true property, matching the direct path.
///
/// Graph: y = relu(x), x ∈ [-1, 1]
/// Property: y >= -0.5 (true since relu output ∈ [0, 1])
#[ntest::timeout(10000)]
#[test]
fn test_relu_split_with_bounds_verified_matches_direct() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 32,
        max_depth: 8,
        timeout: Duration::from_secs(1),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // Pre-compute bounds
    let (node_bounds, output_bounds) = verifier
        .compute_initial_graph_bounds(&graph, &input, None)
        .unwrap();
    let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);

    // Direct path
    let result_direct = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], -0.5)
        .unwrap();

    // Pre-computed bounds path
    let result_precomputed = verifier
        .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], -0.5, &precomputed)
        .unwrap();

    assert!(
        matches!(result_direct.result, BabVerificationStatus::Verified),
        "direct path: expected Verified, got {:?}",
        result_direct.result
    );
    assert!(
        matches!(result_precomputed.result, BabVerificationStatus::Verified),
        "precomputed path: expected Verified, got {:?}",
        result_precomputed.result
    );
}

/// Caller-supplied bounds can prove the root before the normal bootstrap.
/// Both public wrappers must still reject a quarantined cut-authority request.
#[ntest::timeout(10000)]
#[test]
fn relu_split_with_precomputed_early_verified_rejects_cut_authority() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let safe = BetaCrownVerifier::new(BetaCrownConfig {
        use_alpha_crown: false,
        use_crown_ibp: false,
        ..Default::default()
    });
    let (node_bounds, output_bounds) = safe
        .compute_initial_graph_bounds(&graph, &input, None)
        .expect("safe precomputation");
    let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig {
        enable_cuts: true,
        ..Default::default()
    });
    let direct_error = verifier
        .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], -0.5, &precomputed)
        .expect_err("early-Verified precomputed path must reject cut authority");
    let engine_error = verifier
        .verify_graph_relu_split_with_bounds_with_engine(
            &graph,
            &input,
            &[1.0],
            -0.5,
            &precomputed,
            None,
            None,
        )
        .expect_err("engine wrapper must share the same cut-authority quarantine");

    for error in [direct_error, engine_error] {
        assert!(
            error
                .to_string()
                .contains("cut proof authority is quarantined"),
            "expected quarantine error, got {error}"
        );
    }
}

/// Test that verify_graph_relu_split_with_bounds detects a violation,
/// matching the direct path.
///
/// Graph: y = relu(x), x ∈ [-1, 1]
/// Property: y > 0.5 (false since relu(x) can be 0)
#[ntest::timeout(10000)]
#[test]
fn test_relu_split_with_bounds_violation_matches_direct() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 64,
        max_depth: 8,
        timeout: Duration::from_secs(2),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (node_bounds, output_bounds) = verifier
        .compute_initial_graph_bounds(&graph, &input, None)
        .unwrap();
    let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);

    let result_direct = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], 0.5)
        .unwrap();
    let result_precomputed = verifier
        .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], 0.5, &precomputed)
        .unwrap();

    assert!(
        matches!(
            result_direct.result,
            BabVerificationStatus::PotentialViolation { .. }
        ),
        "direct path: expected PotentialViolation, got {:?}",
        result_direct.result
    );
    assert!(
        matches!(
            result_precomputed.result,
            BabVerificationStatus::PotentialViolation { .. }
        ),
        "precomputed path: expected PotentialViolation, got {:?}",
        result_precomputed.result
    );
}

// ============================================================
// Multi-output graph tests
// ============================================================

/// Test relu_split_with_bounds on a multi-layer graph: Linear -> ReLU -> Linear.
///
/// Uses the simple_graph_network (2->2->ReLU->2->1) with objective [1.0].
/// Pre-computes bounds once, then verifies two different thresholds.
/// This exercises the "pre-compute once, verify multiple properties" pattern
/// that is the main use case for this API.
#[ntest::timeout(10000)]
#[test]
fn test_relu_split_with_bounds_multi_layer_graph() {
    use super::gpu_bab::simple_graph_network;

    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 100,
        max_depth: 10,
        timeout: Duration::from_secs(2),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    // Pre-compute bounds once
    let (node_bounds, output_bounds) = verifier
        .compute_initial_graph_bounds(&graph, &input, None)
        .unwrap();
    let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);

    // Property 1: easy threshold (should verify)
    let result1 = verifier
        .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], -10.0, &precomputed)
        .unwrap();
    assert!(
        matches!(result1.result, BabVerificationStatus::Verified),
        "easy threshold (-10.0): expected Verified, got {:?}",
        result1.result
    );
    assert_eq!(
        result1.domains_explored, 1,
        "easy threshold should verify at root without branching"
    );

    // Property 2: same precomputed bounds, harder threshold
    // Run direct path too for comparison
    let result_precomputed = verifier
        .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], -0.5, &precomputed)
        .unwrap();
    let result_direct = verifier
        .verify_graph_relu_split(&graph, &input, &[1.0], -0.5)
        .unwrap();

    // Both paths should agree on verification status
    assert_eq!(
        std::mem::discriminant(&result_precomputed.result),
        std::mem::discriminant(&result_direct.result),
        "precomputed ({:?}) and direct ({:?}) should agree on verification status",
        result_precomputed.result,
        result_direct.result
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_compute_initial_graph_bounds_with_stored_engine_threads_gemm() {
    use super::gpu_bab::simple_graph_network;

    let graph = simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let engine = Arc::new(CountingGemmEngine::new());
    let verifier = BetaCrownVerifier::new_with_engine(
        BetaCrownConfig {
            use_alpha_crown: false,
            timeout: Duration::from_secs(1),
            ..Default::default()
        },
        engine.clone(),
    );

    let (_node_bounds, output_bounds) = verifier
        .compute_initial_graph_bounds(&graph, &input, None)
        .unwrap();

    assert_eq!(output_bounds.lower().len(), 1);
    assert!(
        engine.gemm_calls() > 0,
        "compute_initial_graph_bounds should use the stored GemmEngine for graph CROWN precomputation"
    );
}

/// Regression test for #3707: the loop-top non-finite guard must drop
/// internally constructed NaN/Inf domains before they reach verification or
/// branching checks.
///
/// The public `verify_graph_relu_split_with_bounds*` entry points now reject
/// non-finite root/child bounds earlier via `GraphBabDomain::root()` and
/// `domain_priority()`, so this crate-local regression uses the graph module's
/// test hook to exercise the exact loop-top guard and final `Unknown` result
/// surface directly.
#[test]
fn test_relu_split_with_bounds_non_finite_domain_guard_3707() {
    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("finite test bounds");
    let domain = GraphBabDomain {
        history: GraphSplitHistory::new(),
        node_bounds: std::collections::HashMap::new(),
        lower_bound: f32::NAN,
        upper_bound: 1.0,
        depth: 3,
        priority: 0.0,
        input_bounds: Arc::new(input),
        beta_state: GraphBetaState::empty(),
        alpha_state: GraphDomainAlphaState::empty(),
        cached_la: None,
        delta_pre_nodes: Vec::new(),
    };

    let result = test_non_finite_domain_result_in_relu_split_bounds(&domain);
    assert!(
        matches!(
            result.result,
            BabVerificationStatus::Unknown { ref reason }
                if reason.contains("Child propagation failed for some domains")
        ),
        "dropping a non-finite domain must return Unknown with a propagation-failure reason, got {:?}",
        result.result
    );
    assert_eq!(
        result.domains_explored, 1,
        "the non-finite domain should be dropped immediately at the loop top"
    );
    assert_eq!(
        result.max_depth_reached, domain.depth,
        "the loop result should preserve the dropped domain depth"
    );
}

/// Structural regression for #3707: both child branches in
/// `relu_split_with_bounds` must guard non-finite child bounds before calling
/// `domain_priority(...)`, and they must map the drop to
/// `unresolved_due_to_propagation_failure`.
#[test]
fn test_relu_split_with_bounds_non_finite_child_guard_3707() {
    let source = include_str!("../graph/relu_split_bounds.rs");
    let active_drop = source
        .find("relu_split_with_bounds: active child dropped")
        .expect("active child non-finite guard must remain present");
    let active_priority = source[active_drop..]
        .find("active_child.priority = self.config.domain_priority(l, u)?;")
        .expect("active child should still assign priority after the guard");
    let active_failure = source[active_drop..]
        .find("unresolved_due_to_propagation_failure = true;")
        .expect("active child drop must map to propagation failure");
    assert!(
        active_failure < active_priority,
        "active child must set propagation failure before any priority assignment"
    );

    let inactive_drop = source
        .find("relu_split_with_bounds: inactive child dropped")
        .expect("inactive child non-finite guard must remain present");
    let inactive_priority = source[inactive_drop..]
        .find("inactive_child.priority = self.config.domain_priority(l, u)?;")
        .expect("inactive child should still assign priority after the guard");
    let inactive_failure = source[inactive_drop..]
        .find("unresolved_due_to_propagation_failure = true;")
        .expect("inactive child drop must map to propagation failure");
    assert!(
        inactive_failure < inactive_priority,
        "inactive child must set propagation failure before any priority assignment"
    );
}

// ============================================================
// verify_upper_bound mode tests
// ============================================================

/// Test relu_split_with_bounds in verify_upper_bound mode.
///
/// When verify_upper_bound=true, the property is ub < threshold.
/// Graph: y = relu(x), x ∈ [-1, 1], output ∈ [0, 1]
/// Property: ub < 2.0 (true since max output is 1.0)
#[ntest::timeout(10000)]
#[test]
fn test_relu_split_with_bounds_verify_upper_bound_mode() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        use_alpha_crown: false,
        use_crown_ibp: false,
        enable_cuts: false,
        max_domains: 32,
        max_depth: 8,
        timeout: Duration::from_secs(1),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (node_bounds, output_bounds) = verifier
        .compute_initial_graph_bounds(&graph, &input, None)
        .unwrap();
    let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);

    // ub < 2.0 should verify (max ReLU output is 1.0)
    let result = verifier
        .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], 2.0, &precomputed)
        .unwrap();
    assert!(
        matches!(result.result, BabVerificationStatus::Verified),
        "expected Verified for ub < 2.0, got {:?}",
        result.result
    );

    // ub < 0.3: the property is false (max ReLU output is 1.0 > 0.3), but
    // with a single-node ReLU graph BaB may not find a concrete violation —
    // it returns either PotentialViolation or Unknown. The key invariant is
    // it must NOT return Verified.
    let result_not_verified = verifier
        .verify_graph_relu_split_with_bounds(&graph, &input, &[1.0], 0.3, &precomputed)
        .unwrap();
    assert!(
        !matches!(result_not_verified.result, BabVerificationStatus::Verified),
        "property ub < 0.3 is false — must not return Verified, got {:?}",
        result_not_verified.result
    );
}

// ============================================================
// Objective shape mismatch
// ============================================================

/// Test that objective shape mismatch returns an error, not a panic.
#[ntest::timeout(5000)]
#[test]
fn test_relu_split_with_bounds_shape_mismatch_error() {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        verify_upper_bound: false,
        use_alpha_crown: false,
        enable_cuts: false,
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);

    let (node_bounds, output_bounds) = verifier
        .compute_initial_graph_bounds(&graph, &input, None)
        .unwrap();
    let precomputed = GraphPrecomputedBounds::new(&node_bounds, &output_bounds);

    // Output is 1D but objective has 3 elements — shape mismatch
    let result = verifier.verify_graph_relu_split_with_bounds(
        &graph,
        &input,
        &[1.0, 2.0, 3.0],
        0.0,
        &precomputed,
    );
    assert!(
        result.is_err(),
        "expected error for objective/output shape mismatch"
    );
}

// ============================================================
// Warmup deadline tests (#4260)
// ============================================================

/// `compute_initial_graph_bounds` returns bounds rather than a verifier status,
/// so an already-expired request with no precollected map must preserve the
/// typed deadline instead of starting a fresh uncapped IBP sweep.
#[ntest::timeout(10000)]
#[test]
fn test_compute_initial_graph_bounds_expired_deadline_refuses_fresh_fallback_4260() {
    use std::time::Instant;

    // Multi-layer graph so CROWN can produce tighter bounds than IBP.
    let graph = super::gpu_bab::simple_graph_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let mut config = BetaCrownConfig {
        use_alpha_crown: false,
        timeout: Duration::from_secs(5),
        ..Default::default()
    };
    // Keep the bootstrap on pure IBP so this regression isolates the root-output
    // call that used to drop the caller deadline.
    config.alpha_config.fix_interm_bounds = true;
    let verifier = BetaCrownVerifier::new(config);
    // Test: expired deadline (1ms in the past).
    let expired = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    let error = verifier
        .compute_initial_graph_bounds(&graph, &input, expired)
        .expect_err("expired initial graph bounds must not launch fresh fallback work");
    assert!(
        matches!(error, ny_core::NyError::DeadlineExceeded(_)),
        "expected typed deadline refusal, got {error:?}"
    );
}
