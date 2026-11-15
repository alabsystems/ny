// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::super::indexed_pending::IndexedPendingLinearBounds;
use super::super::bounded_matrix::{BoundedArray, BoundedMat};
use super::super::{copy_into, BoundedBatchPending, PENDING_COMMIT};
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::layers::LinearLayer;
use crate::{GraphNetwork, GraphNode, Layer, LinearBounds};
use ndarray::{Array1, Array2};
use std::cell::Cell;
use std::sync::atomic::Ordering;
use std::time::Duration;

fn model(memory: &MemoryAuthority) -> (WorkBudget, PersistentReservation) {
    let budget = WorkBudget::new_in(
        memory,
        &CancellationToken::default(),
        WorkLimits {
            deadline: None,
            scratch_bytes: 0,
            persistent_bytes: 512,
            max_work: 7,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let retained = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: 512,
                work: 7,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    (budget, retained)
}

fn query(
    memory: &MemoryAuthority,
    cancel: &CancellationToken,
    work: u64,
    deadline: Option<Instant>,
    reserve: Duration,
) -> WorkBudget {
    let budget = WorkBudget::new_in(
        memory,
        cancel,
        WorkLimits {
            deadline,
            scratch_bytes: 1 << 20,
            persistent_bytes: 1 << 20,
            max_work: work,
            max_proposals: 0,
            certificate_reserve: reserve,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(1 << 20).unwrap();
    budget
}

fn used(owner: &SoaStorageBudget) -> usize {
    owner.reserved_bytes.load(Ordering::Acquire)
}
fn refused_as<T>(result: Result<T>, expected: &str) {
    match result {
        Err(NyError::UnsupportedOp(reason)) => assert!(reason.contains(expected), "{reason}"),
        Err(other) => panic!("wrong refusal: {other}"),
        Ok(_) => panic!("expected {expected}"),
    }
}
fn expired<T>(result: Result<T>) {
    match result {
        Err(error) => assert!(error.is_deadline_exceeded(), "{error}"),
        Ok(_) => panic!("expected original-authority expiry"),
    }
}

#[test]
fn retained_identity_joins_same_memory_without_renewing_producer_work() {
    let memory = MemoryAuthority::new(1 << 23);
    let (producer, retained) = model(&memory);
    let budget = query(
        &memory,
        &CancellationToken::default(),
        10000,
        None,
        Duration::ZERO,
    );
    assert!(retained.shares_ledger(&producer));
    assert!(!retained.shares_ledger(&budget));
    assert!(retained.shares_memory_authority(&budget));
    let wrong = query(
        &MemoryAuthority::new(1 << 23),
        &CancellationToken::default(),
        10000,
        None,
        Duration::ZERO,
    );
    assert!(!retained.shares_memory_authority(&wrong));
    assert!(
        SoaStorageBudget::from_query(&wrong, retained.clone(), LeaseClass::Fallback, None).is_err()
    );
    let unjoined = WorkBudget::new(None, 10000, 10000, 10000, 0, Duration::ZERO).unwrap();
    assert!(!retained.shares_memory_authority(&unjoined));
    assert!(
        SoaStorageBudget::from_query(&unjoined, retained.clone(), LeaseClass::Fallback, None)
            .is_err()
    );
    assert_eq!(wrong.receipt().unwrap().admitted, 0);
    assert_eq!(unjoined.receipt().unwrap().admitted, 0);
    let producer_before = producer.receipt().unwrap();
    let owner = SoaStorageBudget::from_query(&budget, retained.clone(), LeaseClass::Fallback, None)
        .unwrap();
    assert_eq!(producer.receipt().unwrap(), producer_before);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        512 + used(&owner)
    );
    drop(retained);
    drop(owner);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(producer.receipt().unwrap().work_reserved, 7);
}

#[test]
fn joined_arrays_and_aligned_matrices_keep_bits_actual_capacity_and_last_owner() {
    let memory = MemoryAuthority::new(1 << 23);
    let (producer, retained) = model(&memory);
    let budget = query(
        &memory,
        &CancellationToken::default(),
        1000000,
        None,
        Duration::ZERO,
    );
    let owner =
        SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap();
    let header = used(&owner);
    let bits = [0u32, 0x80000000, 1, 0x7f800000, 0x7fc01234, 0x3f800000];
    let array = BoundedArray::<f32>::try_from_fn(&owner, 20, 65, None, |i, j| {
        f32::from_bits(bits[(i * 65 + j) % bits.len()])
    })
    .unwrap();
    let after_array = used(&owner);
    let matrix = BoundedMat::<f64>::try_from_fn(&owner, 7, 5, None, |i, j| {
        f64::from_bits([0, 1 << 63, 1, 0x7ff8123456789abc, 0x3ff0000000000000][(i + j) % 5])
    })
    .unwrap();
    for i in 0..20 {
        for j in 0..65 {
            assert_eq!(
                array.array()[[i, j]].to_bits(),
                bits[(i * 65 + j) % bits.len()]
            );
        }
    }
    for i in 0..7 {
        for j in 0..5 {
            assert_eq!(
                matrix.matrix()[(i, j)].to_bits(),
                [0, 1 << 63, 1, 0x7ff8123456789abc, 0x3ff0000000000000][(i + j) % 5]
            );
        }
    }
    assert_eq!(matrix.matrix().row_stride(), 1);
    assert_eq!(matrix.matrix().col_stride(), 8);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, used(&owner));
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        512 + used(&owner)
    );
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        64 + 2 * 64 + 20 * 65 + 5 * 16
    );
    assert_eq!(
        (
            budget.receipt().unwrap().admitted,
            budget.receipt().unwrap().completed
        ),
        (3, 2)
    );
    let complete_bytes = used(&owner);
    drop(owner);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        512 + complete_bytes
    );
    drop(array);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        512 + complete_bytes - (after_array - header)
    );
    drop(matrix);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(
        (
            budget.receipt().unwrap().live_scratch_bytes,
            budget.receipt().unwrap().completed
        ),
        (0, 2)
    );
    assert_eq!(producer.receipt().unwrap().work_reserved, 7);
}

#[test]
fn work_and_global_byte_refusals_precede_getters_and_keep_other_claims() {
    for work_limit in [false, true] {
        let memory = MemoryAuthority::new(if work_limit { 1 << 23 } else { 8192 });
        let (_producer, retained) = model(&memory);
        let budget = query(
            &memory,
            &CancellationToken::default(),
            if work_limit { 128 } else { 1000000 },
            None,
            Duration::ZERO,
        );
        let owner =
            SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap();
        let header = used(&owner);
        let called = Cell::new(0);
        let result = BoundedArray::<f32>::try_from_fn(
            &owner,
            1,
            if work_limit { 8 } else { 4096 },
            None,
            |_, _| {
                called.set(called.get() + 1);
                1.0
            },
        );
        refused_as(
            result,
            if work_limit {
                "WorkLimit"
            } else {
                "MemoryLimit"
            },
        );
        assert_eq!(called.get(), 0);
        assert_eq!(used(&owner), header);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, header);
        assert_eq!(
            memory.receipt().unwrap().live_materialized_bytes,
            512 + header
        );
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, 128);
    }
}

#[test]
fn actual_capacity_overage_refusal_rolls_back_the_same_query_and_local_account() {
    let memory = MemoryAuthority::new(8192);
    let (_producer, retained) = model(&memory);
    let budget = query(
        &memory,
        &CancellationToken::default(),
        1000000,
        None,
        Duration::ZERO,
    );
    let owner =
        SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap();
    let header = used(&owner);
    let sibling = query(
        &memory,
        &CancellationToken::default(),
        1000000,
        None,
        Duration::ZERO,
    );
    let held = sibling
        .lease(WorkRequest {
            scratch_bytes: 8192 - 512 - header - 8,
            ..WorkRequest::default()
        })
        .unwrap();
    let mut poll = Poll::for_owner(&owner, None);
    let mut charge = owner.charge(1, &mut poll).unwrap();
    let mut values = Vec::<u8>::new();
    values.try_reserve_exact(17).unwrap();
    let capacity = values.capacity();
    assert!(capacity >= 17);
    refused_as(
        charge.reconcile::<u8>(1, capacity, &mut poll),
        "MemoryLimit",
    );
    assert_eq!((values).len(), 0);
    assert_eq!(used(&owner), header + 1);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, header + 1);
    drop(values);
    drop(charge);
    assert_eq!(used(&owner), header);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, header);
    drop(held);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn original_query_cancellation_aborts_after_real_weighted_work_without_publishing() {
    for matrix in [false, true] {
        let memory = MemoryAuthority::new(1 << 23);
        let (_producer, retained) = model(&memory);
        let cancel = CancellationToken::default();
        let budget = query(&memory, &cancel, 1000000, None, Duration::ZERO);
        let owner =
            SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap();
        let header = used(&owner);
        let calls = Cell::new(0);
        let get = |_: usize, _: usize| {
            calls.set(calls.get() + 1);
            if calls.get() == 1 {
                cancel.cancel();
            }
            1.0
        };
        if matrix {
            expired(BoundedMat::<f64>::try_from_fn(&owner, 8192, 1, None, get));
        } else {
            expired(BoundedArray::<f64>::try_from_fn(&owner, 8192, 1, None, get));
        }
        assert_eq!(
            calls.get(),
            2048,
            "4096 f32-equivalent units of real f64 fill"
        );
        assert_eq!(used(&owner), header);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, header);
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        assert!(budget.receipt().unwrap().work_reserved >= 64 + 64 + 8192 * 2);
        drop(owner);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    }
    let memory = MemoryAuthority::new(1 << 23);
    let (_producer, retained) = model(&memory);
    let cancel = CancellationToken::default();
    let budget = query(&memory, &cancel, 1000000, None, Duration::ZERO);
    let owner =
        SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap();
    let header = used(&owner);
    let calls = Cell::new(0);
    let mut poll = Poll::for_owner(&owner, None);
    let mut charge = owner.charge(8192 * 4, &mut poll).unwrap();
    let mut dst = super::super::filled(8192, 42.0, &mut charge, &mut poll).unwrap();
    charge.materialize().unwrap();
    let source = std::iter::from_fn(|| {
        calls.set(calls.get() + 1);
        if calls.get() == 1 {
            cancel.cancel();
        }
        Some(1.0)
    });
    expired(copy_into(&mut dst, None, source, &mut poll));
    assert_eq!(calls.get(), 4096);
    assert_eq!(dst.iter().filter(|&&v| v == 1.0).count(), 4096);
    drop(dst);
    drop(charge);
    assert_eq!(used(&owner), header);
}

#[test]
fn original_class_deadlines_only_narrow_and_expiry_precedes_getters() {
    let memory = MemoryAuthority::new(1 << 23);
    let (_producer, retained) = model(&memory);
    let final_at = Instant::now() + Duration::from_secs(60);
    let budget = query(
        &memory,
        &CancellationToken::default(),
        1000000,
        Some(final_at),
        Duration::from_secs(5),
    );
    let owner = SoaStorageBudget::from_query(&budget, retained.clone(), LeaseClass::Optional, None)
        .unwrap();
    assert_eq!(
        owner.deadline,
        Some(final_at.checked_sub(Duration::from_secs(5)).unwrap())
    );
    let array = BoundedArray::<f32>::try_from_fn(&owner, 1, 1, Some(final_at), |_, _| 1.0).unwrap();
    assert_eq!(array.deadline(), owner.deadline);
    let earlier = Instant::now() + Duration::from_secs(10);
    let matrix = BoundedMat::<f64>::try_from_fn(&owner, 1, 1, Some(earlier), |_, _| 1.0).unwrap();
    assert_eq!(matrix.deadline(), Some(earlier));
    let calls = Cell::new(0);
    expired(BoundedArray::<f32>::try_from_fn(
        &owner,
        1,
        1,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
        |_, _| {
            calls.set(1);
            1.0
        },
    ));
    assert_eq!(calls.get(), 0);
    let fallback =
        SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap();
    assert_eq!(fallback.deadline, Some(final_at));
}

fn plan() -> crate::network::CrownDispatchPlan {
    let mut graph = GraphNetwork::new();
    graph
        .try_add_node(GraphNode::from_input(
            "out",
            Layer::Linear(LinearLayer::new(Array2::eye(2), None).unwrap()),
        ))
        .unwrap();
    graph.set_output("out");
    crate::network::CrownDispatchPlan::build(&graph).unwrap()
}
fn bits(bounds: &LinearBounds) -> Vec<u32> {
    let mut out = Vec::new();
    for values in [bounds.lower_a(), bounds.upper_a()] {
        out.extend(values.iter().map(|v| v.to_bits()));
    }
    for values in [bounds.lower_b(), bounds.upper_b()] {
        out.extend(values.iter().map(|v| v.to_bits()));
    }
    for error in [bounds.lower_a_err(), bounds.upper_a_err()] {
        out.push(u32::from(error.is_some()));
        if let Some(values) = error {
            out.extend(values.iter().map(|v| v.to_bits()));
        }
    }
    out
}

#[test]
fn joined_pending_seed_take_merge_and_materialize_match_original_reference_bits() {
    let plan = plan();
    let idx = plan.output_node_idx;
    let seed = LinearBounds::new(
        Array2::from_elem((2, 2), 0.5),
        Array1::from_elem(2, -0.25),
        Array2::from_elem((2, 2), -0.5),
        Array1::from_elem(2, 0.75),
    )
    .unwrap();
    for deadline in [None, Some(Instant::now() + Duration::from_secs(60))] {
        let memory = MemoryAuthority::new(1 << 23);
        let (_producer, retained) = model(&memory);
        let budget = query(
            &memory,
            &CancellationToken::default(),
            1000000,
            deadline,
            Duration::ZERO,
        );
        let owner =
            SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap();
        let header = used(&owner);
        let pending = BoundedBatchPending::try_new(&plan, 2, &owner, None)
            .unwrap()
            .seed_broadcast(idx, &seed)
            .unwrap();
        let (pending, slot) = pending.take(idx).unwrap();
        let slot = slot.unwrap();
        let contribution = slot.try_materialize(0, None).unwrap().unwrap();
        let mut reference = IndexedPendingLinearBounds::new(&plan, 2);
        let mut pending = pending;
        for _ in 0..2 {
            reference
                .accumulate_idx(idx, contribution.bounds().clone(), 1)
                .unwrap();
            pending = pending.accumulate_domain(idx, 1, &contribution).unwrap();
        }
        let expected = reference.take_idx(idx).unwrap();
        let (pending, merged) = pending.take(idx).unwrap();
        let merged = merged.unwrap();
        let observed = merged.try_materialize(1, None).unwrap().unwrap();
        assert_eq!(bits(observed.bounds()), bits(expected[1].as_ref().unwrap()));
        drop(observed);
        drop(merged);
        drop(contribution);
        drop(slot);
        drop(pending);
        assert_eq!(used(&owner), header);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, header);
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        assert!(budget.receipt().unwrap().work_reserved > 64);
    }
}

#[test]
fn joined_pending_refusal_consumes_private_storage_and_keeps_external_owner() {
    let plan = plan();
    let idx = plan.output_node_idx;
    let seed = LinearBounds::identity(2);
    for final_commit in [false, true] {
        let memory = MemoryAuthority::new(1 << 23);
        let (_producer, retained) = model(&memory);
        let budget = query(
            &memory,
            &CancellationToken::default(),
            1000000,
            None,
            Duration::ZERO,
        );
        let owner =
            SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap();
        let header = used(&owner);
        let external = BoundedArray::<f32>::try_from_fn(&owner, 1, 5, None, |_, _| 3.0).unwrap();
        let external_bytes = used(&owner) - header;
        let pending = BoundedBatchPending::try_new(&plan, 2, &owner, None).unwrap();
        let before = budget.receipt().unwrap().work_reserved;
        if final_commit {
            let mut poll = Poll::for_owner(&owner, None);
            poll.forced_stage = Some(PENDING_COMMIT);
            expired(pending.seed_with_poll(idx, &seed, &mut poll));
            assert!(budget.receipt().unwrap().work_reserved > before);
        } else {
            let held = budget
                .lease(WorkRequest {
                    work: 1000000 - before - 1,
                    ..WorkRequest::default()
                })
                .unwrap();
            refused_as(pending.seed_broadcast(idx, &seed), "WorkLimit");
            drop(held);
        }
        assert_eq!(used(&owner), header + external_bytes);
        assert_eq!(
            budget.receipt().unwrap().live_scratch_bytes,
            header + external_bytes
        );
        assert!(external.array().iter().all(|&v| v == 3.0));
        drop(external);
        assert_eq!(used(&owner), header);
    }
}

#[test]
fn merged_contribution_deadline_persists_through_transaction_take_and_publication() {
    let plan = plan();
    let idx = plan.output_node_idx;
    let seed = LinearBounds::identity(2);
    for joined in [false, true] {
        let memory = MemoryAuthority::new(1 << 23);
        let (_producer, retained) = model(&memory);
        let budget = query(
            &memory,
            &CancellationToken::default(),
            1000000,
            None,
            Duration::ZERO,
        );
        let owner = if joined {
            SoaStorageBudget::from_query(&budget, retained, LeaseClass::Fallback, None).unwrap()
        } else {
            SoaStorageBudget::new(0, None).unwrap()
        };
        let pending = BoundedBatchPending::try_new(&plan, 2, &owner, None)
            .unwrap()
            .seed_broadcast(idx, &seed)
            .unwrap();
        let (pending, slot) = pending.take(idx).unwrap();
        let slot = slot.unwrap();
        let earlier = Instant::now() + Duration::from_secs(10);
        let contribution = slot.try_materialize(0, Some(earlier)).unwrap().unwrap();
        let pending = pending.accumulate_domain(idx, 1, &contribution).unwrap();
        assert_eq!(pending.deadline, Some(earlier));
        let (_pending, merged) = pending.take(idx).unwrap();
        let merged = merged.unwrap();
        assert_eq!(merged.deadline, Some(earlier));
        let later = earlier + Duration::from_secs(30);
        let published = merged.try_materialize(1, Some(later)).unwrap().unwrap();
        assert_eq!(published.deadline(), Some(earlier));
    }
}
