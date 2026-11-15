// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve};
use std::time::Duration;

fn budget(authority: &MemoryAuthority, cancel: &CancellationToken) -> WorkBudget {
    WorkBudget::new_in(
        authority,
        cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: 100,
            persistent_bytes: 100,
            max_work: 20,
            max_proposals: 5,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap()
}

fn request(scratch: usize, persistent: usize) -> WorkRequest {
    WorkRequest {
        scratch_bytes: scratch,
        persistent_bytes: persistent,
        work: 3,
        proposals: 1,
    }
}

#[test]
fn residency_distinct_queries_share_aggregate_admission_without_partial_local_debit() {
    let authority = MemoryAuthority::new(100);
    let first = budget(&authority, &CancellationToken::default());
    let second = budget(&authority, &CancellationToken::default());
    assert!(!first.shares_ledger(&second));
    assert!(first.shares_memory_authority(&second));
    let lease = first.lease(request(40, 30)).unwrap();
    assert_eq!(
        second.lease(request(20, 20)).unwrap_err(),
        WorkRefusal::MemoryLimit
    );
    let local = second.receipt().unwrap();
    assert_eq!(
        (
            local.live_scratch_bytes,
            local.live_persistent_bytes,
            local.work_reserved
        ),
        (0, 0, 0)
    );
    assert_eq!((local.attempted, local.refused, local.admitted), (1, 1, 0));
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 70);
    drop(lease);
    let retry = second.lease(request(20, 20)).unwrap();
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 40);
    drop(retry);
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn residency_concurrent_admission_is_atomic_across_independent_work_ledgers() {
    let authority = MemoryAuthority::new(100);
    let barrier = Arc::new(std::sync::Barrier::new(3));
    let release = Arc::new(std::sync::Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let authority = authority.clone();
        let barrier = barrier.clone();
        let release = release.clone();
        workers.push(std::thread::spawn(move || {
            let ledger = budget(&authority, &CancellationToken::default());
            barrier.wait();
            let claim = ledger.lease(request(40, 30));
            let admitted = claim.is_ok();
            release.wait();
            drop(claim);
            admitted
        }));
    }
    barrier.wait();
    release.wait();
    let admitted = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .filter(|admitted| *admitted)
        .count();
    assert_eq!(admitted, 1);
    let receipt = authority.receipt().unwrap();
    assert_eq!(
        (receipt.admitted, receipt.refused, receipt.peak_leased_bytes),
        (1, 1, 70)
    );
    assert_eq!(
        (receipt.live_pending_bytes, receipt.live_materialized_bytes),
        (0, 0)
    );
}

#[test]
fn residency_pending_materialized_and_last_clone_keep_one_capacity_claim() {
    let authority = MemoryAuthority::new(100);
    let ledger = budget(&authority, &CancellationToken::default());
    let mut lease = ledger
        .lease_for(LeaseClass::Replay, request(20, 30))
        .unwrap();
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 50);
    lease.mark_materialized().unwrap();
    assert_eq!(
        (
            authority.receipt().unwrap().live_pending_bytes,
            authority.receipt().unwrap().live_materialized_bytes
        ),
        (0, 50)
    );
    let retained = lease.commit_persistent().unwrap();
    let clone = retained.clone();
    assert_eq!(authority.receipt().unwrap().live_materialized_bytes, 30);
    assert_eq!(ledger.receipt().unwrap().live_scratch_bytes, 0);
    drop(retained);
    assert_eq!(authority.receipt().unwrap().live_materialized_bytes, 30);
    drop(clone);
    assert_eq!(authority.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().completed, 1);
}

#[test]
fn residency_cancelled_detached_worker_retains_bytes_until_actual_drop() {
    let authority = MemoryAuthority::new(100);
    let cancel = CancellationToken::default();
    let ledger = budget(&authority, &cancel);
    let lease = ledger.lease(request(20, 30)).unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        receive.recv().unwrap();
        assert_eq!(lease.checkpoint(), Err(WorkRefusal::Cancelled));
        drop(lease);
    });
    cancel.cancel();
    assert_eq!(
        ledger.checkpoint_for(LeaseClass::Replay),
        Err(WorkRefusal::Cancelled)
    );
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 50);
    assert_eq!(
        ledger.lease(request(0, 0)).unwrap_err(),
        WorkRefusal::Cancelled
    );
    send.send(()).unwrap();
    worker.join().unwrap();
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, 3);
}

#[test]
fn residency_cancelled_query_does_not_cancel_resident_or_an_explicit_new_query() {
    let authority = MemoryAuthority::new(100);
    let model_ledger = budget(&authority, &CancellationToken::default());
    let model = model_ledger
        .lease(request(0, 60))
        .unwrap()
        .commit_persistent()
        .unwrap();
    let cancel = CancellationToken::default();
    let query = budget(&authority, &cancel);
    cancel.cancel();
    assert_eq!(
        query.lease(request(10, 10)).unwrap_err(),
        WorkRefusal::Cancelled
    );
    let next = budget(&authority, &CancellationToken::default());
    let active = next.lease(request(10, 10)).unwrap();
    assert_eq!(authority.receipt().unwrap().live_materialized_bytes, 60);
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 20);
    assert!(!query.shares_ledger(&next));
    assert!(query.shares_memory_authority(&next));
    drop(active);
    drop(model);
    assert_eq!(authority.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn residency_equal_caps_do_not_substitute_authority_or_hide_overflow() {
    let authority = MemoryAuthority::new(usize::MAX);
    let other = MemoryAuthority::new(usize::MAX);
    assert!(!authority.shares_authority(&other));
    let mut claim = authority.reserve(usize::MAX).unwrap();
    assert_eq!(authority.reserve(1).unwrap_err(), WorkRefusal::Overflow);
    claim.materialize().unwrap();
    assert_eq!(authority.reserve(1).unwrap_err(), WorkRefusal::Overflow);
    claim.retain_only(1).unwrap();
    assert_eq!(authority.receipt().unwrap().live_materialized_bytes, 1);
    drop(claim);
    assert_eq!(authority.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn residency_completion_does_not_release_live_scratch_or_replenish_work() {
    let authority = MemoryAuthority::new(100);
    let first = budget(&authority, &CancellationToken::default());
    let other = budget(&authority, &CancellationToken::default());
    let mut lease = first.lease(request(40, 30)).unwrap();
    lease.complete().unwrap();
    lease.complete().unwrap();
    assert_eq!(first.receipt().unwrap().completed, 1);
    assert_eq!(first.receipt().unwrap().live_scratch_bytes, 40);
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 70);
    assert_eq!(
        other.lease(request(40, 0)).unwrap_err(),
        WorkRefusal::MemoryLimit
    );
    let retained = lease.commit_persistent().unwrap();
    assert_eq!(authority.receipt().unwrap().live_materialized_bytes, 30);
    let next = other.lease(request(40, 0)).unwrap();
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 40);
    drop(next);
    drop(retained);
    assert_eq!(first.receipt().unwrap().work_reserved, 3);
}

#[test]
fn residency_local_refusal_does_not_charge_global_and_cancellation_keeps_completed_claim() {
    let authority = MemoryAuthority::new(100);
    let cancel = CancellationToken::default();
    let ledger = budget(&authority, &cancel);
    assert_eq!(
        ledger.lease(request(101, 0)).unwrap_err(),
        WorkRefusal::ScratchLimit
    );
    assert_eq!(authority.receipt().unwrap().attempted, 0);
    let mut lease = ledger.lease(request(20, 30)).unwrap();
    lease.complete().unwrap();
    cancel.cancel();
    assert_eq!(
        lease.commit_persistent().unwrap_err(),
        WorkRefusal::Cancelled
    );
    assert_eq!(authority.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(ledger.receipt().unwrap().work_reserved, 3);
    for class in [
        LeaseClass::Optional,
        LeaseClass::Replay,
        LeaseClass::Fallback,
    ] {
        assert_eq!(ledger.checkpoint_for(class), Err(WorkRefusal::Cancelled));
    }
}

#[test]
fn residency_retained_owner_unavailable_claim_matches_original_commit_fail_closed() {
    for materialized in [false, true] {
        let mut results = Vec::new();
        for typed in [false, true] {
            let authority = MemoryAuthority::new(200);
            let cancel = CancellationToken::default();
            let ledger = budget(&authority, &cancel);
            let requested = request(40, 100);
            let mut lease = ledger.lease(requested).unwrap();
            if materialized {
                lease.mark_materialized().unwrap();
            }
            let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _guard = authority.0.lock().unwrap();
                panic!("isolated retained-owner claim refusal control");
            }));
            assert!(poisoned.is_err());
            let result = if typed {
                lease.commit_retained_with(|| [1_u8]).map(|_| ())
            } else {
                lease.commit_persistent().map(|_| ())
            };
            assert_eq!(result, Err(WorkRefusal::Unavailable));
            let local = ledger.receipt().unwrap();
            assert_eq!(local.completed, 1);
            assert_eq!(local.live_scratch_bytes, 0);
            assert_eq!(local.live_persistent_bytes, 0);
            assert_eq!(local.work_reserved, requested.work);
            assert_eq!(authority.receipt(), Err(WorkRefusal::Unavailable));
            let state = authority.0.lock().unwrap_err().into_inner();
            assert_eq!(
                state.receipt.live_pending_bytes,
                if materialized { 0 } else { 140 }
            );
            assert_eq!(
                state.receipt.live_materialized_bytes,
                if materialized { 140 } else { 0 }
            );
            results.push((local, state.receipt));
        }
        assert_eq!(results[0], results[1]);
    }
}
