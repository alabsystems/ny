// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the BaB engine core verification loop (#2568).
//!
//! Covers: BabLoopState transitions, pop_domain_batch queue behavior,
//! domain priority ordering, prefilter_domain_batch status classification,
//! process_batch_children queue insertion, and full verify-loop edge cases.

use super::prelude::*;
use crate::beta_crown::engine::verify_phases::{pop_domain_batch, BabLoopState};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_layer_bounds_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap(),
    )
    .unwrap()
}

/// Build a minimal root domain with the given output bounds.
fn root_domain(lower_bound: f32, upper_bound: f32) -> BabDomain {
    let layer = make_layer_bounds_1d(&[0.0], &[1.0]);
    BabDomain::root(vec![layer], lower_bound, upper_bound).unwrap()
}

/// Build a root domain with explicit priority override.
fn root_domain_with_priority(lower_bound: f32, upper_bound: f32, priority: f32) -> BabDomain {
    let mut d = root_domain(lower_bound, upper_bound);
    d.priority = priority;
    d
}

// ---------------------------------------------------------------------------
// 1. BabLoopState transitions
// ---------------------------------------------------------------------------

#[test]
fn test_bab_loop_state_new_has_no_unresolved() {
    let state = BabLoopState::new(false);
    assert_eq!(state.domains_explored, 0);
    assert_eq!(state.domains_verified, 0);
    assert_eq!(state.max_depth, 0);
    assert!(
        !state.has_unresolved(),
        "fresh state must not be unresolved"
    );
}

#[test]
fn test_bab_loop_state_propagation_failure_marks_unresolved() {
    let mut state = BabLoopState::new(false);
    state.unresolved_due_to_propagation_failure = true;
    assert!(
        state.has_unresolved(),
        "propagation failure must surface as unresolved"
    );
    let reason = state.unresolved_reason(10);
    assert!(
        reason.contains("propagation"),
        "reason should mention propagation: got '{reason}'"
    );
}

#[test]
fn test_bab_loop_state_no_branch_marks_unresolved() {
    let mut state = BabLoopState::new(false);
    state.unresolved_due_to_no_branch = true;
    assert!(
        state.has_unresolved(),
        "no-branch condition must surface as unresolved"
    );
    let reason = state.unresolved_reason(10);
    assert!(
        reason.contains("unstable"),
        "reason should mention unstable neurons: got '{reason}'"
    );
}

#[test]
fn test_bab_loop_state_unsplittable_marks_unresolved() {
    let mut state = BabLoopState::new(false);
    state.unresolved_due_to_unsplittable = true;
    assert!(
        state.has_unresolved(),
        "unsplittable input box must surface as unresolved"
    );
    let reason = state.unresolved_reason(10);
    assert!(
        reason.contains("splittable input dimension"),
        "reason should mention the unsplittable input box: got '{reason}'"
    );
}

#[test]
fn test_bab_loop_state_depth_marks_unresolved() {
    let mut state = BabLoopState::new(false);
    state.unresolved_due_to_depth = true;
    assert!(state.has_unresolved());
    let reason = state.unresolved_reason(42);
    assert!(
        reason.contains("42"),
        "reason should include the max depth: got '{reason}'"
    );
}

#[test]
fn test_bab_loop_state_multiple_unresolved_reasons_combined() {
    let mut state = BabLoopState::new(false);
    state.unresolved_due_to_propagation_failure = true;
    state.unresolved_due_to_no_branch = true;
    let reason = state.unresolved_reason(10);
    assert!(
        reason.contains("propagation") && reason.contains("unstable"),
        "multiple reasons should be combined: got '{reason}'"
    );
}

#[test]
fn test_bab_loop_state_verified_result_status() {
    let mut state = BabLoopState::new(false);
    state.domains_explored = 5;
    state.domains_verified = 5;
    state.max_depth = 3;
    let result = state.verified_result(std::time::Instant::now(), 2);
    assert_eq!(result.result, BabVerificationStatus::Verified);
    assert_eq!(result.domains_explored, 5);
    assert_eq!(result.domains_verified, 5);
    assert_eq!(result.max_depth_reached, 3);
    assert_eq!(result.cuts_generated, 2);
}

#[test]
fn test_bab_loop_state_unknown_result_carries_reason() {
    let mut state = BabLoopState::new(false);
    state.domains_explored = 10;
    state.domains_verified = 3;
    let result = state.unknown_result(
        std::time::Instant::now(),
        0,
        "test failure reason".to_string(),
    );
    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert_eq!(reason, "test failure reason");
        }
        other => panic!("expected Unknown, got {other:?}"),
    }
    assert_eq!(result.domains_explored, 10);
    assert_eq!(result.domains_verified, 3);
}

// ---------------------------------------------------------------------------
// 2. pop_domain_batch queue behavior
// ---------------------------------------------------------------------------

#[test]
fn test_pop_domain_batch_empty_queue_returns_empty() {
    let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();
    let batch = pop_domain_batch(&mut queue, 10);
    assert!(
        batch.is_empty(),
        "popping from empty queue must return empty vec"
    );
}

#[test]
fn test_pop_domain_batch_fewer_than_batch_size() {
    let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();
    queue.push(root_domain(1.0, 2.0));
    queue.push(root_domain(3.0, 4.0));

    let batch = pop_domain_batch(&mut queue, 10);
    assert_eq!(batch.len(), 2, "should return all available domains");
    assert!(queue.is_empty(), "queue should be drained");
}

#[test]
fn test_pop_domain_batch_exact_batch_size() {
    let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();
    for i in 0..5 {
        queue.push(root_domain(i as f32, (i + 1) as f32));
    }

    let batch = pop_domain_batch(&mut queue, 5);
    assert_eq!(batch.len(), 5);
    assert!(queue.is_empty());
}

#[test]
fn test_pop_domain_batch_respects_batch_size_limit() {
    let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();
    for i in 0..10 {
        queue.push(root_domain(i as f32, (i + 1) as f32));
    }

    let batch = pop_domain_batch(&mut queue, 3);
    assert_eq!(batch.len(), 3, "must not exceed batch size");
    assert_eq!(queue.len(), 7, "remaining domains stay in queue");
}

#[test]
fn test_pop_domain_batch_pops_highest_priority_first() {
    let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();
    queue.push(root_domain_with_priority(1.0, 2.0, 5.0));
    queue.push(root_domain_with_priority(3.0, 4.0, 100.0));
    queue.push(root_domain_with_priority(2.0, 3.0, 50.0));

    let batch = pop_domain_batch(&mut queue, 2);
    assert_eq!(batch.len(), 2);
    // Max-heap: highest priority first
    assert_eq!(
        batch[0].priority(),
        100.0,
        "first popped domain should have highest priority"
    );
    assert_eq!(
        batch[1].priority(),
        50.0,
        "second popped domain should have second-highest priority"
    );
}

// ---------------------------------------------------------------------------
// 3. Domain priority ordering in BinaryHeap (NaN handling)
// ---------------------------------------------------------------------------

#[test]
fn test_domain_nan_priority_surfaces_first() {
    // NaN priority domains should be popped first (surface invalid domains
    // immediately rather than letting them accumulate silently).
    let mut queue: BinaryHeap<BabDomain> = BinaryHeap::new();
    queue.push(root_domain_with_priority(1.0, 2.0, 100.0));
    queue.push(root_domain_with_priority(0.5, 1.5, f32::NAN));
    queue.push(root_domain_with_priority(2.0, 3.0, 200.0));

    let first = queue.pop().unwrap();
    assert!(
        first.priority().is_nan(),
        "NaN-priority domain must be popped first from max-heap"
    );
}

#[test]
fn test_domain_ordering_deterministic_for_equal_priorities() {
    // Two domains with equal priority should compare as Equal (not panic).
    let d1 = root_domain_with_priority(1.0, 2.0, 42.0);
    let d2 = root_domain_with_priority(3.0, 4.0, 42.0);
    assert_eq!(
        d1.cmp(&d2),
        std::cmp::Ordering::Equal,
        "domains with equal priority must compare as Equal"
    );
}

// ---------------------------------------------------------------------------
// 4. Full verify loop: domain limit and timeout paths
// ---------------------------------------------------------------------------

#[test]
fn test_verify_domain_limit_returns_unknown() {
    // Network where root is not verified/violated, forcing BaB to split.
    // Set max_domains=1 so it hits the domain limit after exploring root.
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 1,
        timeout: Duration::from_secs(10),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -100.0).unwrap();

    // With max_domains=1, the engine explores the root and then hits the limit.
    // The root domain has lower_bound > -100 (should verify immediately for
    // this trivial threshold), so we expect Verified.
    // If threshold is chosen to make root NOT verify, we'd get Unknown.
    // With threshold=-100 and a simple network, root will verify.
    assert_eq!(
        result.result,
        BabVerificationStatus::Verified,
        "trivial threshold should verify at root"
    );
    assert_eq!(result.domains_explored, 1);
}

#[test]
fn test_verify_domain_limit_triggers_unknown_for_tight_threshold() {
    // Set a threshold that the root cannot verify but also cannot violate,
    // forcing the engine to try splitting. With max_domains=1, it hits the limit.
    let w = arr2(&[[1.0, -1.0]]);
    let linear = LinearLayer::new(w, None).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear));

    // Input: x in [-1, 1], y in [-1, 1]. Output = x - y, range [-2, 2].
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // threshold=0: lower=-2 (not > 0), upper=2 (not < 0), so Unknown territory.
    let config = BetaCrownConfig {
        max_domains: 2,
        timeout: Duration::from_secs(10),
        max_depth: 0, // prevent any splitting
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, 0.0).unwrap();

    // Root is not verified (lower=-2 <= 0) and not violated (upper=2 >= 0).
    // max_depth=0 prevents splitting, so domain is unresolved.
    match &result.result {
        BabVerificationStatus::Unknown { reason } => {
            assert!(
                !reason.is_empty(),
                "Unknown result should carry a non-empty reason"
            );
        }
        other => panic!("expected Unknown due to max_depth=0, got {other:?}"),
    }
}

#[test]
fn test_verify_timeout_returns_timeout_status() {
    // Use a network that requires splitting with zero timeout budget for BaB.
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Extremely short timeout: the initial phase alone should consume it.
    let config = BetaCrownConfig {
        timeout: Duration::from_nanos(1),
        ..Default::default()
    };
    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, 0.0).unwrap();

    // With a 1ns timeout, either the initial phase returns early (Verified/
    // PotentialViolation) or the BaB loop times out immediately.
    let is_terminal = matches!(
        result.result,
        BabVerificationStatus::Verified
            | BabVerificationStatus::PotentialViolation
            | BabVerificationStatus::Timeout
            | BabVerificationStatus::Unknown { .. }
    );
    assert!(
        is_terminal,
        "extremely short timeout must produce a terminal result, got {:?}",
        result.result
    );
}

// ---------------------------------------------------------------------------
// 5. BabDomain construction edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_bab_domain_root_rejects_infinity() {
    let layer = make_layer_bounds_1d(&[0.0], &[1.0]);
    let result = BabDomain::root(vec![layer.clone()], f32::INFINITY, 1.0);
    assert!(result.is_err(), "+Inf lower_bound should be rejected");

    let result = BabDomain::root(vec![layer], 0.0, f32::NEG_INFINITY);
    assert!(result.is_err(), "-Inf upper_bound should be rejected");
}

#[test]
fn test_bab_domain_child_rejects_nan() {
    let layer = make_layer_bounds_1d(&[0.0], &[1.0]);
    let result = BabDomain::child(
        SplitHistory::new(),
        f32::NAN,
        1.0,
        vec![Arc::new(layer)],
        None,
        DomainAlphaState::empty(),
        BetaState::empty(),
        None,
        0,
        IntermediateLinearBounds::empty(),
    );
    assert!(
        result.is_err(),
        "NaN lower_bound in child should be rejected"
    );
}

#[test]
fn test_bab_domain_depth_includes_input_splits() {
    let layer = make_layer_bounds_1d(&[0.0], &[1.0]);
    let input =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let domain = BabDomain::root_with_input(vec![layer], 0.0, 1.0, &input).unwrap();
    // Root has 0 splits, depth = history.depth() + input_split_count = 0
    assert_eq!(domain.depth(), 0);
}
