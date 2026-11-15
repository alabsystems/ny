// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use std::mem::size_of;
use std::sync::{Arc, Barrier};

fn query(memory: &MemoryAuthority, cancel: &CancellationToken, cap: usize) -> WorkBudget {
    WorkBudget::new_in(
        memory,
        cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: cap,
            persistent_bytes: 0,
            max_work: 1000,
            max_proposals: 10,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap()
}

#[test]
fn actual_vector_capacity_is_reconciled_before_fill_and_sealed_at_materialization() {
    let memory = MemoryAuthority::new(4096);
    let budget = query(&memory, &CancellationToken::default(), 4096);
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: size_of::<u64>(),
            work: 17,
            ..WorkRequest::default()
        })
        .unwrap();
    let mut storage = Vec::<u64>::new();
    storage.try_reserve_exact(17).unwrap();
    let actual_bytes = storage.capacity() * size_of::<u64>();
    assert!(actual_bytes >= 17 * size_of::<u64>());
    assert_eq!((storage).len(), 0);
    lease
        .reserve_more_scratch(actual_bytes - size_of::<u64>())
        .unwrap();
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, actual_bytes);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, actual_bytes);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    let patterns = [0, 1 << 63, 1, 0x7ff8_1234_5678_9abc];
    for i in 0..17 {
        storage.push(patterns[i % patterns.len()]);
    }
    lease.mark_materialized().unwrap();
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        actual_bytes
    );
    assert_eq!(lease.reserve_more_scratch(1), Err(WorkRefusal::Unavailable));
    let receipt = budget.receipt().unwrap();
    assert_eq!(
        (
            receipt.attempted,
            receipt.admitted,
            receipt.completed,
            receipt.work_reserved
        ),
        (1, 1, 0, 17)
    );
    for (i, &value) in storage.iter().enumerate() {
        assert_eq!(value, patterns[i % patterns.len()]);
    }
    drop(storage);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        actual_bytes
    );
    drop(lease);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(budget.receipt().unwrap().work_reserved, 17);
}

#[test]
fn growth_preserves_protected_bytes_and_refusals_do_not_change_existing_claims() {
    let budget = WorkBudget::new_reserved(
        None,
        64,
        32,
        5,
        3,
        Duration::ZERO,
        WorkReserve {
            persistent_bytes: 16,
            work: 2,
            proposals: 1,
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(64).unwrap();
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: 8,
            persistent_bytes: 8,
            work: 1,
            proposals: 1,
        })
        .unwrap();
    lease.reserve_more_scratch(32).unwrap();
    let before = budget.receipt().unwrap();
    assert_eq!(
        (
            before.live_scratch_bytes,
            before.live_persistent_bytes,
            before.peak_leased_bytes
        ),
        (40, 8, 48)
    );
    assert_eq!(lease.reserve_more_scratch(1), Err(WorkRefusal::MemoryLimit));
    assert_eq!(budget.receipt().unwrap(), before);
    assert_eq!(
        lease.reserve_more_scratch(25),
        Err(WorkRefusal::ScratchLimit)
    );
    assert_eq!(budget.receipt().unwrap(), before);
    let fallback = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: 16,
                work: 2,
                proposals: 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes
            + budget.receipt().unwrap().live_persistent_bytes,
        64
    );
    assert_eq!(
        budget.tighten_total_byte_limit(63),
        Err(WorkRefusal::MemoryLimit)
    );
    drop(fallback);
    drop(lease);
    assert_eq!(
        (
            budget.receipt().unwrap().live_scratch_bytes,
            budget.receipt().unwrap().live_persistent_bytes
        ),
        (0, 0)
    );
    assert_eq!(
        (
            budget.receipt().unwrap().work_reserved,
            budget.receipt().unwrap().proposals_reserved
        ),
        (3, 2)
    );
}

#[test]
fn byte_growth_neither_spends_nor_replenishes_already_reserved_work_and_proposals() {
    let budget = WorkBudget::new_reserved(
        None,
        64,
        0,
        5,
        3,
        Duration::ZERO,
        WorkReserve {
            work: 2,
            proposals: 1,
            ..WorkReserve::default()
        },
    )
    .unwrap();
    let mut optional = budget
        .lease(WorkRequest {
            scratch_bytes: 8,
            work: 3,
            proposals: 2,
            ..WorkRequest::default()
        })
        .unwrap();
    let mut fallback = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 2,
                proposals: 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    fallback.complete().unwrap();
    drop(fallback);
    let before = budget.receipt().unwrap();
    optional.reserve_more_scratch(16).unwrap();
    optional.reserve_more_scratch(0).unwrap();
    let after = budget.receipt().unwrap();
    assert_eq!(after.live_scratch_bytes, 24);
    assert_eq!((after.work_reserved, after.proposals_reserved), (5, 3));
    assert_eq!(
        (after.attempted, after.admitted, after.completed),
        (before.attempted, before.admitted, before.completed)
    );
    assert_eq!(optional.reserve_more_work(1), Err(WorkRefusal::WorkLimit));
}

#[test]
fn resident_model_and_distinct_queries_share_the_same_capacity_during_growth() {
    let memory = MemoryAuthority::new(100);
    let mut model = memory.reserve(30).unwrap();
    model.materialize().unwrap();
    assert_eq!(model.grow_pending(1), Err(WorkRefusal::Unavailable));
    let first = query(&memory, &CancellationToken::default(), 100);
    let second = query(&memory, &CancellationToken::default(), 100);
    assert!(!first.shares_ledger(&second));
    assert!(first.shares_memory_authority(&second));
    let mut lease = first
        .lease(WorkRequest {
            scratch_bytes: 8,
            ..WorkRequest::default()
        })
        .unwrap();
    lease.reserve_more_scratch(62).unwrap();
    assert_eq!(
        (
            memory.receipt().unwrap().live_pending_bytes,
            memory.receipt().unwrap().live_materialized_bytes
        ),
        (70, 30)
    );
    let before = first.receipt().unwrap();
    assert_eq!(lease.reserve_more_scratch(1), Err(WorkRefusal::MemoryLimit));
    assert_eq!(first.receipt().unwrap(), before);
    assert_eq!(
        second
            .lease(WorkRequest {
                scratch_bytes: 1,
                ..WorkRequest::default()
            })
            .unwrap_err(),
        WorkRefusal::MemoryLimit
    );
    assert_eq!(second.receipt().unwrap().live_scratch_bytes, 0);
    drop(lease);
    assert_eq!(
        (
            memory.receipt().unwrap().live_pending_bytes,
            memory.receipt().unwrap().live_materialized_bytes
        ),
        (0, 30)
    );
    drop(model);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn original_cancellation_and_deadline_refuse_growth_without_releasing_live_owners() {
    let memory = MemoryAuthority::new(100);
    let cancel = CancellationToken::default();
    let budget = query(&memory, &cancel, 100);
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: 8,
            ..WorkRequest::default()
        })
        .unwrap();
    cancel.cancel();
    for additional in [0, 1] {
        assert_eq!(
            lease.reserve_more_scratch(additional),
            Err(WorkRefusal::Cancelled)
        );
    }
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 8);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 8);
    drop(lease);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);

    let deadline = Instant::now() + Duration::from_millis(500);
    let budget = WorkBudget::new(Some(deadline), 100, 0, 0, 0, Duration::ZERO).unwrap();
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: 8,
            ..WorkRequest::default()
        })
        .unwrap();
    assert_eq!(lease.deadline(), Some(deadline));
    std::thread::sleep(
        deadline.saturating_duration_since(Instant::now()) + Duration::from_millis(2),
    );
    assert_eq!(lease.reserve_more_scratch(1), Err(WorkRefusal::Deadline));
    assert_eq!(lease.deadline(), Some(deadline));
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 8);
    drop(lease);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn materialization_and_completion_seal_growth_and_persistent_commit_keeps_last_owner() {
    for joined in [false, true] {
        let memory = MemoryAuthority::new(100);
        let budget = if joined {
            WorkBudget::new_in(
                &memory,
                &CancellationToken::default(),
                WorkLimits {
                    deadline: None,
                    scratch_bytes: 50,
                    persistent_bytes: 50,
                    max_work: 10,
                    max_proposals: 0,
                    certificate_reserve: Duration::ZERO,
                    reserve: WorkReserve::default(),
                },
            )
            .unwrap()
        } else {
            WorkBudget::new(None, 50, 50, 10, 0, Duration::ZERO).unwrap()
        };
        let mut lease = budget
            .lease(WorkRequest {
                scratch_bytes: 8,
                persistent_bytes: 17,
                work: 1,
                ..WorkRequest::default()
            })
            .unwrap();
        lease.reserve_more_scratch(24).unwrap();
        lease.mark_materialized().unwrap();
        assert_eq!(lease.reserve_more_scratch(0), Err(WorkRefusal::Unavailable));
        lease.reserve_more_work(1).unwrap();
        let keep = lease.commit_persistent().unwrap();
        let last = keep.clone();
        assert_eq!(
            (
                budget.receipt().unwrap().live_scratch_bytes,
                budget.receipt().unwrap().live_persistent_bytes
            ),
            (0, 17)
        );
        if joined {
            assert_eq!(
                (
                    memory.receipt().unwrap().live_pending_bytes,
                    memory.receipt().unwrap().live_materialized_bytes
                ),
                (0, 17)
            );
        }
        drop(keep);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 17);
        drop(last);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
        let mut completed = budget
            .lease(WorkRequest {
                scratch_bytes: 8,
                ..WorkRequest::default()
            })
            .unwrap();
        completed.complete().unwrap();
        assert_eq!(
            completed.reserve_more_scratch(1),
            Err(WorkRefusal::Unavailable)
        );
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 8);
    }
}

#[ntest::timeout(60_000)]
#[test]
fn concurrent_growth_across_distinct_ledgers_cannot_double_spend_global_capacity() {
    let memory = MemoryAuthority::new(100);
    let start = Arc::new(Barrier::new(3));
    let observed = Arc::new(Barrier::new(3));
    let release = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let memory = memory.clone();
        let start = start.clone();
        let observed = observed.clone();
        let release = release.clone();
        workers.push(std::thread::spawn(move || {
            let budget = query(&memory, &CancellationToken::default(), 100);
            let mut lease = budget.lease(WorkRequest::default()).unwrap();
            start.wait();
            let result = lease.reserve_more_scratch(70);
            let receipt = budget.receipt().unwrap();
            observed.wait();
            release.wait();
            drop(lease);
            (result, receipt)
        }));
    }
    start.wait();
    observed.wait();
    let held = memory.receipt().unwrap();
    assert_eq!((held.live_pending_bytes, held.peak_leased_bytes), (70, 70));
    release.wait();
    let results: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|(r, _)| r.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|(r, _)| *r == Err(WorkRefusal::MemoryLimit))
            .count(),
        1
    );
    for (result, receipt) in results {
        assert_eq!(
            receipt.live_scratch_bytes,
            if result.is_ok() { 70 } else { 0 }
        );
        assert_eq!(
            (receipt.attempted, receipt.admitted, receipt.completed),
            (1, 1, 0)
        );
    }
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn overflow_and_actual_capacity_refusal_preserve_old_claims_before_any_fill() {
    let memory = MemoryAuthority::new(usize::MAX);
    let budget = query(&memory, &CancellationToken::default(), usize::MAX);
    let mut huge = budget
        .lease(WorkRequest {
            scratch_bytes: usize::MAX - 1,
            ..WorkRequest::default()
        })
        .unwrap();
    let before = budget.receipt().unwrap();
    assert_eq!(huge.reserve_more_scratch(2), Err(WorkRefusal::Overflow));
    assert_eq!(budget.receipt().unwrap(), before);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, usize::MAX - 1);
    drop(huge);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);

    let memory = MemoryAuthority::new(24);
    let model = memory.reserve(8).unwrap();
    let budget = query(&memory, &CancellationToken::default(), 100);
    let mut lease = budget
        .lease(WorkRequest {
            scratch_bytes: 8,
            ..WorkRequest::default()
        })
        .unwrap();
    let mut storage = Vec::<u8>::new();
    storage.try_reserve_exact(17).unwrap();
    let actual = storage.capacity();
    assert!(actual >= 17);
    let before = budget.receipt().unwrap();
    assert_eq!(
        lease.reserve_more_scratch(actual - 8),
        Err(WorkRefusal::MemoryLimit)
    );
    assert!(
        storage.is_empty(),
        "refused capacity never enters a getter/fill"
    );
    assert_eq!(budget.receipt().unwrap(), before);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 16);
    drop(storage);
    drop(lease);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 8);
    drop(model);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}
