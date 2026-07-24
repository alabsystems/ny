// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::BinaryHeap;
use std::time::Instant;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::domain_results::GraphDomainResult;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::BabVerificationStatus;
use crate::beta_crown::BetaCrownVerifier;

use super::test_domain;

/// AlreadyVerified -> domains_verified incremented.
#[ntest::timeout(5000)]
#[test]
fn test_aggregate_already_verified() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
    let mut cut_pool = GraphCutPool::default();
    let priority_fn = |l: f32, _u: f32| -> ny_core::Result<f32> { Ok(l) };

    let results = vec![GraphDomainResult::AlreadyVerified];

    let violation = verifier
        .aggregate_bab_results(
            results,
            0.0,
            &priority_fn,
            &mut queue,
            &mut lifecycle,
            &mut cut_pool,
        )
        .expect("should not error");

    assert!(
        violation.is_none(),
        "AlreadyVerified should not be violation"
    );
    assert_eq!(lifecycle.domains_verified, 1);
}

/// Violation -> returns Some(PotentialViolation) immediately.
#[ntest::timeout(5000)]
#[test]
fn test_aggregate_violation_returns_early() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
    let mut cut_pool = GraphCutPool::default();
    let priority_fn = |l: f32, _u: f32| -> ny_core::Result<f32> { Ok(l) };

    let results = vec![
        GraphDomainResult::AlreadyVerified,
        GraphDomainResult::Violation,
        GraphDomainResult::AlreadyVerified,
    ];

    let violation = verifier
        .aggregate_bab_results(
            results,
            0.0,
            &priority_fn,
            &mut queue,
            &mut lifecycle,
            &mut cut_pool,
        )
        .expect("should not error");

    let r = violation.expect("Violation should return Some");
    assert!(
        matches!(r.result, BabVerificationStatus::PotentialViolation),
        "should be PotentialViolation, got {:?}",
        r.result
    );
    assert_eq!(lifecycle.domains_verified, 1);
}

/// Children with verified=true -> domains_verified incremented.
#[ntest::timeout(5000)]
#[test]
fn test_aggregate_children_verified_counted() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
    let mut cut_pool = GraphCutPool::default();
    let priority_fn = |l: f32, _u: f32| -> ny_core::Result<f32> { Ok(l) };

    let child = test_domain(5.0, 10.0);
    let results = vec![GraphDomainResult::Children(vec![(child, true)])];

    let violation = verifier
        .aggregate_bab_results(
            results,
            0.0,
            &priority_fn,
            &mut queue,
            &mut lifecycle,
            &mut cut_pool,
        )
        .expect("should not error");

    assert!(violation.is_none());
    assert_eq!(lifecycle.domains_verified, 1);
    assert!(queue.is_empty(), "verified child should not be enqueued");
}

/// Children with verified=false -> enqueued with priority.
#[ntest::timeout(5000)]
#[test]
fn test_aggregate_children_unverified_enqueued() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
    let mut cut_pool = GraphCutPool::default();
    let priority_fn = |l: f32, _u: f32| -> ny_core::Result<f32> { Ok(l) };

    let child = test_domain(1.0, 5.0);
    let results = vec![GraphDomainResult::Children(vec![(child, false)])];

    let violation = verifier
        .aggregate_bab_results(
            results,
            3.0,
            &priority_fn,
            &mut queue,
            &mut lifecycle,
            &mut cut_pool,
        )
        .expect("should not error");

    assert!(violation.is_none());
    assert_eq!(lifecycle.domains_verified, 0);
    assert_eq!(queue.len(), 1, "unverified child should be enqueued");
}

/// NoUnstable with verified=true -> domains_verified incremented.
#[ntest::timeout(5000)]
#[test]
fn test_aggregate_no_unstable_verified() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
    let mut cut_pool = GraphCutPool::default();
    let priority_fn = |l: f32, _u: f32| -> ny_core::Result<f32> { Ok(l) };

    let results = vec![GraphDomainResult::NoUnstable {
        lower: 5.0,
        upper: 10.0,
        verified: true,
    }];

    let violation = verifier
        .aggregate_bab_results(
            results,
            3.0,
            &priority_fn,
            &mut queue,
            &mut lifecycle,
            &mut cut_pool,
        )
        .expect("should not error");

    assert!(violation.is_none());
    assert_eq!(lifecycle.domains_verified, 1);
}

/// NoUnstable with verified=false and violation -> returns PotentialViolation.
#[ntest::timeout(5000)]
#[test]
fn test_aggregate_no_unstable_violation() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
    let mut cut_pool = GraphCutPool::default();
    let priority_fn = |l: f32, _u: f32| -> ny_core::Result<f32> { Ok(l) };

    let results = vec![GraphDomainResult::NoUnstable {
        lower: -5.0,
        upper: -1.0,
        verified: false,
    }];

    let violation = verifier
        .aggregate_bab_results(
            results,
            3.0,
            &priority_fn,
            &mut queue,
            &mut lifecycle,
            &mut cut_pool,
        )
        .expect("should not error");

    let r = violation.expect("should return violation");
    assert!(
        matches!(r.result, BabVerificationStatus::PotentialViolation),
        "NoUnstable with violation bounds should be PotentialViolation, got {:?}",
        r.result
    );
}

/// NoUnstable with verified=false and no violation -> unresolved_due_to_no_branch.
#[ntest::timeout(5000)]
#[test]
fn test_aggregate_no_unstable_unresolved() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
    let mut cut_pool = GraphCutPool::default();
    let priority_fn = |l: f32, _u: f32| -> ny_core::Result<f32> { Ok(l) };

    let results = vec![GraphDomainResult::NoUnstable {
        lower: 1.0,
        upper: 5.0,
        verified: false,
    }];

    let violation = verifier
        .aggregate_bab_results(
            results,
            3.0,
            &priority_fn,
            &mut queue,
            &mut lifecycle,
            &mut cut_pool,
        )
        .expect("should not error");

    assert!(violation.is_none());
    assert!(
        lifecycle.unresolved_due_to_no_branch,
        "NoUnstable + not-verified + not-violation must set unresolved_due_to_no_branch"
    );
}

/// PropagationFailure -> unresolved_due_to_propagation_failure.
#[ntest::timeout(5000)]
#[test]
fn test_aggregate_propagation_failure() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let mut lifecycle = GraphBabLifecycle::new(Instant::now());
    let mut queue: BinaryHeap<GraphBabDomain> = BinaryHeap::new();
    let mut cut_pool = GraphCutPool::default();
    let priority_fn = |l: f32, _u: f32| -> ny_core::Result<f32> { Ok(l) };

    let results = vec![GraphDomainResult::PropagationFailure];

    let violation = verifier
        .aggregate_bab_results(
            results,
            0.0,
            &priority_fn,
            &mut queue,
            &mut lifecycle,
            &mut cut_pool,
        )
        .expect("should not error");

    assert!(violation.is_none());
    assert!(
        lifecycle.unresolved_due_to_propagation_failure,
        "PropagationFailure must set unresolved_due_to_propagation_failure"
    );
}
