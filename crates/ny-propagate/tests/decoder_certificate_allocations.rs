// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
//! Allocator requests for the actual public certificate query/replay routes.
//! This independent test executable owns its allocator; the propagation library
//! retains its unconditional unsafe-code denial.

#[path = "support/certificate_allocations.rs"]
mod allocations;

use allocations::{measure, Counts};
use ny_propagate::certify::work::WorkBudget;
use ny_propagate::decoder::normalized_head::shared::{
    replay_shared_normalized_next_token, verify_shared_normalized_next_token,
};
use ny_propagate::decoder::normalized_head::{
    FinalNormalization, FinalNormalizationKind, NormalizedNextTokenOutcome,
};
use ny_propagate::decoder::{
    HeadArithmetic, HiddenBox, NextTokenOutcome, NextTokenProvenance, SharedAffineHead,
    VocabOptions,
};
use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

fn ledger() -> WorkBudget {
    WorkBudget::new(None, 32 << 20, 64 << 20, 1 << 34, 10_000, Duration::ZERO).unwrap()
}

fn fixture() -> (SharedAffineHead, VocabOptions) {
    let limits = VocabOptions {
        budget: Some(ledger()),
        ..VocabOptions::default()
    };
    let owner = SharedAffineHead::new(
        4,
        1,
        vec![0.0; 4],
        vec![2.0, 0.0, 0.0, 0.0],
        HeadArithmetic::ExactReal,
        &limits,
    )
    .unwrap();
    (owner, limits)
}

fn hidden() -> HiddenBox<'static> {
    HiddenBox {
        lo: &[-1.0],
        hi: &[1.0],
    }
}

fn assert_one_scope_owner(scope: &ny_propagate::certify::scope::CertificateScope, counts: &Counts) {
    assert!(!counts.overflowed);
    let owner = counts.allocation_containing(std::ptr::from_ref(scope).addr());
    assert!(
        owner.is_some(),
        "the minted scope must inhabit one measured allocation"
    );
    assert!(
        owner.unwrap().bytes >= size_of_val(scope) + 2 * size_of::<usize>(),
        "the enclosing owner includes its scope and the Arc control block"
    );
}

fn observe(
    phase: &str,
    report_bytes: usize,
    certificate_bytes: usize,
    scope: &ny_propagate::certify::scope::CertificateScope,
    reservation: usize,
    counts: &Counts,
) {
    assert_one_scope_owner(scope, counts);
    let owner = counts
        .allocation_containing(std::ptr::from_ref(scope).addr())
        .unwrap();
    println!(
        "{{\"phase\":\"{phase}\",\"report_bytes\":{report_bytes},\"certificate_bytes\":{certificate_bytes},\"owner_request_bytes\":{},\"report_plus_owner_bytes\":{},\"reserved_bytes\":{reservation},\"allocations\":{},\"reallocations\":{},\"deallocations\":{},\"requested_bytes\":{}}}",
        owner.bytes,
        report_bytes + owner.bytes,
        counts.allocations,
        counts.reallocations,
        counts.deallocations,
        counts.requested_bytes,
    );
}

/// This observation also runs against the unchanged representation before the
/// patch is applied. Compare its exact native rows across source-bound runs;
/// observing a row alone supplies no compatibility or retained-price claim.
#[test]
fn decoder_owner_native_storage_observation() {
    let (owner, limits) = fixture();
    let (query, counts) = measure(|| {
        owner
            .verify_next_token(hidden(), &[0.0], 0, &limits)
            .unwrap()
    })
    .unwrap();
    let NextTokenOutcome::Certified(proof) = &query.outcome else {
        panic!("checked constant query")
    };
    observe(
        "query",
        size_of_val(&query),
        size_of_val(proof),
        proof.scope(),
        query.reserved_bytes(),
        &counts,
    );
    let (replay, counts) = measure(|| {
        owner
            .replay_next_token(hidden(), &[0.0], 0, &query.evidence, &limits)
            .unwrap()
    })
    .unwrap();
    let NextTokenOutcome::Certified(proof) = &replay.outcome else {
        panic!("checked constant replay")
    };
    observe(
        "replay",
        size_of_val(&replay),
        size_of_val(proof),
        proof.scope(),
        replay.reserved_bytes(),
        &counts,
    );
    let normalization = FinalNormalization {
        kind: FinalNormalizationKind::RmsNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1e-5,
    };
    let (query, counts) = measure(|| {
        verify_shared_normalized_next_token(&owner, normalization, hidden(), 0, &limits).unwrap()
    })
    .unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &query.outcome else {
        panic!("checked constant normalized query")
    };
    observe(
        "normalized_query",
        size_of_val(&query),
        size_of_val(proof),
        proof.scope(),
        query.reserved_bytes(),
        &counts,
    );
    let (replay, counts) = measure(|| {
        replay_shared_normalized_next_token(
            &owner,
            normalization,
            hidden(),
            0,
            &query.evidence,
            &limits,
        )
        .unwrap()
    })
    .unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &replay.outcome else {
        panic!("checked constant normalized replay")
    };
    observe(
        "normalized_replay",
        size_of_val(&replay),
        size_of_val(proof),
        proof.scope(),
        replay.reserved_bytes(),
        &counts,
    );
}

#[test]
fn decoder_owner_query_replay_clone_scope_and_last_alias_use_actual_allocator_requests() {
    let (owner, limits) = fixture();
    let budget = owner.budget();
    let (mut query, query_allocations) = measure(|| {
        owner
            .verify_next_token(hidden(), &[0.0], 0, &limits)
            .unwrap()
    })
    .unwrap();
    let NextTokenOutcome::Certified(proof) = &query.outcome else {
        panic!("strict constant margin")
    };
    assert_eq!(size_of_val(proof), size_of::<Arc<()>>());
    assert_one_scope_owner(proof.scope(), &query_allocations);
    let (retained, clone_allocations) = measure(|| black_box(proof).clone()).unwrap();
    assert!(clone_allocations.is_empty());
    assert!(std::ptr::eq(proof.scope(), retained.scope()));
    let (replay, replay_allocations) = measure(|| {
        owner
            .replay_next_token(hidden(), &[0.0], 0, &query.evidence, &limits)
            .unwrap()
    })
    .unwrap();
    let NextTokenOutcome::Certified(replayed) = &replay.outcome else {
        panic!("strict original replay")
    };
    assert_one_scope_owner(replayed.scope(), &replay_allocations);
    assert_eq!(retained.scope(), replayed.scope());
    assert!(!std::ptr::eq(retained.scope(), replayed.scope()));
    assert_eq!(retained.provenance(), NextTokenProvenance::CheckedQuery);
    assert_eq!(replayed.provenance(), NextTokenProvenance::OriginalReplay);
    query.outcome = NextTokenOutcome::Certified(replayed.clone());
    assert_eq!(query.provenance(), NextTokenProvenance::OriginalReplay);
    assert_eq!(retained.provenance(), NextTokenProvenance::CheckedQuery);
    query.evidence.groups[0].upper = -99.0;
    assert!(owner
        .replay_next_token(hidden(), &[0.0], 0, &query.evidence, &limits)
        .is_err());
    let mut foreign = limits;
    foreign.budget = Some(ledger());
    let before = budget.receipt().unwrap();
    assert!(owner
        .verify_next_token(hidden(), &[0.0], 0, &foreign)
        .is_err());
    let after = budget.receipt().unwrap();
    assert_eq!(before.admitted, after.admitted);
    assert_eq!(before.work_reserved, after.work_reserved);
    assert_eq!(before.live_persistent_bytes, after.live_persistent_bytes);
    let expected = retained.reserved_bytes() + retained.shared_head_bytes();
    drop(query);
    drop(replay);
    drop(owner);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, expected);
    let (last, clone_allocations) = measure(|| black_box(&retained).clone()).unwrap();
    assert!(clone_allocations.is_empty());
    drop(retained);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, expected);
    drop(last);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().work_reserved, after.work_reserved);
}

#[test]
fn decoder_normalized_owner_query_replay_clone_scope_and_last_alias_are_measured() {
    let (owner, limits) = fixture();
    let budget = owner.budget();
    let normalization = FinalNormalization {
        kind: FinalNormalizationKind::RmsNorm,
        gamma: &[1.0],
        beta: None,
        epsilon: 1e-5,
    };
    let (mut query, query_allocations) = measure(|| {
        verify_shared_normalized_next_token(&owner, normalization, hidden(), 0, &limits).unwrap()
    })
    .unwrap();
    let NormalizedNextTokenOutcome::Certified(proof) = &query.outcome else {
        panic!("strict normalized constant margin")
    };
    assert_eq!(size_of_val(proof), size_of::<Arc<()>>());
    assert_one_scope_owner(proof.scope(), &query_allocations);
    let (retained, clone_allocations) = measure(|| black_box(proof).clone()).unwrap();
    assert!(clone_allocations.is_empty());
    assert!(std::ptr::eq(proof.scope(), retained.scope()));
    let checked = retained.checked_stats().clone();
    let (replay, replay_allocations) = measure(|| {
        replay_shared_normalized_next_token(
            &owner,
            normalization,
            hidden(),
            0,
            &query.evidence,
            &limits,
        )
        .unwrap()
    })
    .unwrap();
    let NormalizedNextTokenOutcome::Certified(replayed) = &replay.outcome else {
        panic!("strict normalized replay")
    };
    assert_one_scope_owner(replayed.scope(), &replay_allocations);
    assert_eq!(retained.scope(), replayed.scope());
    assert!(!std::ptr::eq(retained.scope(), replayed.scope()));
    query.outcome = NormalizedNextTokenOutcome::Certified(replayed.clone());
    assert_eq!(query.provenance(), NextTokenProvenance::OriginalReplay);
    query.stats.kernel_coordinate_visits = u64::MAX;
    query.evidence.rows[0].upper = -99.0;
    assert_eq!(retained.checked_stats(), &checked);
    assert_eq!(retained.provenance(), NextTokenProvenance::CheckedQuery);
    assert!(replay_shared_normalized_next_token(
        &owner,
        normalization,
        hidden(),
        0,
        &query.evidence,
        &limits,
    )
    .is_err());
    let mut foreign = limits;
    foreign.budget = Some(ledger());
    let before = budget.receipt().unwrap();
    assert!(
        verify_shared_normalized_next_token(&owner, normalization, hidden(), 0, &foreign,).is_err()
    );
    let after = budget.receipt().unwrap();
    assert_eq!(before.admitted, after.admitted);
    assert_eq!(before.work_reserved, after.work_reserved);
    assert_eq!(before.live_persistent_bytes, after.live_persistent_bytes);
    let expected = retained.reserved_bytes() + retained.shared_head_bytes();
    drop(query);
    drop(replay);
    drop(owner);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, expected);
    let (last, clone_allocations) = measure(|| black_box(&retained).clone()).unwrap();
    assert!(clone_allocations.is_empty());
    drop(retained);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, expected);
    drop(last);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().work_reserved, after.work_reserved);
}
