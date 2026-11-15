// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::super::super::{BoundedBatchPending, BoundedLinearBounds};
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, PersistentReservation, WorkBudget, WorkLimits, WorkRequest, WorkReserve,
};
use crate::layers::LinearLayer;
use crate::network::CrownDispatchPlan;
use crate::{GraphNetwork, GraphNode, Layer};
use ndarray::s;
use std::sync::atomic::Ordering;
use std::time::Duration;

const RETAINED: usize = 1 << 16;

fn joined(cancel: &CancellationToken) -> (MemoryAuthority, WorkBudget, Arc<SoaStorageBudget>) {
    let (memory, query, owner, _retained) = joined_retained(cancel);
    (memory, query, owner)
}

fn joined_retained(
    cancel: &CancellationToken,
) -> (
    MemoryAuthority,
    WorkBudget,
    Arc<SoaStorageBudget>,
    PersistentReservation,
) {
    let memory = MemoryAuthority::new(1 << 25);
    let producer = WorkBudget::new_in(
        &memory,
        &CancellationToken::default(),
        WorkLimits {
            deadline: None,
            scratch_bytes: 0,
            persistent_bytes: RETAINED,
            max_work: 7,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let retained = producer
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: RETAINED,
                work: 7,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    let query = WorkBudget::new_in(
        &memory,
        cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: 1 << 24,
            persistent_bytes: 1 << 24,
            max_work: 1 << 30,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let owner =
        SoaStorageBudget::from_query(&query, retained.clone(), LeaseClass::Fallback, None).unwrap();
    (memory, query, owner, retained)
}

fn used(owner: &SoaStorageBudget) -> usize {
    owner.reserved_bytes.load(Ordering::Acquire)
}
fn clean(
    memory: &MemoryAuthority,
    query: &WorkBudget,
    owner: Arc<SoaStorageBudget>,
    header: usize,
) {
    assert_eq!(used(&owner), header);
    assert_eq!(query.receipt().unwrap().live_scratch_bytes, header);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        RETAINED + header
    );
    drop(owner);
    assert_eq!(query.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

fn parts(
    owner: &Arc<SoaStorageBudget>,
    rows: usize,
    cols: usize,
    strided: bool,
) -> [BoundedArray<f32>; 6] {
    let values = [
        0.0,
        -0.0,
        1.25,
        -0.375,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
    ];
    std::array::from_fn(|role| {
        let width = if role == 1 || role == 3 { 1 } else { cols };
        let value = |i: usize, j: usize| values[(i * 3 + j * 5 + role) % values.len()];
        if strided {
            let mut out =
                BoundedArray::try_from_fn(owner, rows * 2 + 2, width * 2 + 2, None, |i, j| {
                    value(i / 2, j / 2)
                })
                .unwrap();
            out.storage = out.storage.slice_move(s![1..1+rows*2;2,1..1+width*2;2]);
            out
        } else {
            if rows == 0 || width == 0 {
                BoundedArray::try_from_fn_allow_empty(owner, rows, width, None, value).unwrap()
            } else {
                BoundedArray::try_from_fn(owner, rows, width, None, value).unwrap()
            }
        }
    })
}

fn reference(parts: &[BoundedArray<f32>; 6]) -> LinearBounds {
    LinearBounds::new_or_conservative_with_err(
        parts[0].array().clone(),
        parts[1].array().column(0).to_owned(),
        parts[2].array().clone(),
        parts[3].array().column(0).to_owned(),
        parts[4].array().clone(),
        parts[5].array().clone(),
    )
    .unwrap()
}

fn bound_bits(b: &LinearBounds) -> Vec<u32> {
    let mut bits = Vec::new();
    bits.extend(b.lower_a().iter().map(|v| v.to_bits()));
    bits.extend(b.lower_b().iter().map(|v| v.to_bits()));
    bits.extend(b.upper_a().iter().map(|v| v.to_bits()));
    bits.extend(b.upper_b().iter().map(|v| v.to_bits()));
    for error in [b.lower_a_err(), b.upper_a_err()] {
        bits.push(u32::from(error.is_some()));
        if let Some(error) = error {
            bits.extend(error.iter().map(|v| v.to_bits()));
        }
    }
    bits
}

#[test]
fn claimed_output_moves_all_six_owners_offsets_capacities_without_new_lease_or_bytes() {
    for (rows, cols) in [(0, 0), (0, 7), (3, 0), (1, 1), (5, 17)] {
        for strided in [false, true] {
            let (memory, query, owner) = joined(&CancellationToken::default());
            let header = used(&owner);
            let input = parts(&owner, rows, cols, strided);
            let expected = reference(&input);
            let pointers = input.each_ref().map(|p| p.storage.as_ptr() as usize);
            let capacities = input.each_ref().map(|p| p._charge.bytes);
            let bytes = used(&owner);
            let before = query.receipt().unwrap();
            query.tighten_total_byte_limit(bytes).unwrap();
            let mut output = ClaimedKernelOutput::from_parts(input, None).unwrap();
            assert_eq!(bound_bits(output.bounds()), bound_bits(&expected));
            assert_eq!(used(&owner), bytes);
            assert_eq!(query.receipt().unwrap().admitted, before.admitted);
            assert_eq!(output._claims.each_ref().map(|c| c.bytes), capacities);
            // Unwrap only in this descendant test while the wrapper still
            // holds ALL claims. Raw Vecs drop before the emptied wrapper.
            let b = &mut output.bounds;
            let raw = [
                std::mem::take(&mut b.lower_a).into_raw_vec_and_offset(),
                std::mem::take(&mut b.lower_b).into_raw_vec_and_offset(),
                std::mem::take(&mut b.upper_a).into_raw_vec_and_offset(),
                std::mem::take(&mut b.upper_b).into_raw_vec_and_offset(),
                b.lower_a_err.take().unwrap().into_raw_vec_and_offset(),
                b.upper_a_err.take().unwrap().into_raw_vec_and_offset(),
            ];
            for i in 0..6 {
                assert_eq!(raw[i].0.capacity() * 4, capacities[i]);
                // Empty ndarray owners need not retain a meaningful offset.
                if rows != 0 && (i == 1 || i == 3 || cols != 0) {
                    assert_eq!(
                        raw[i].0.as_ptr() as usize + raw[i].1.unwrap_or(0) * 4,
                        pointers[i]
                    );
                }
            }
            drop(raw);
            drop(output);
            clean(&memory, &query, owner, header);
        }
    }
}

#[test]
fn claimed_output_uses_original_numeric_firewall_and_keeps_every_claim_on_fallback() {
    for strided in [false, true] {
        for role in 0..6 {
            for value in [
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -1.0,
                f32::from_bits(1),
                -0.0,
            ] {
                let (memory, query, owner) = joined(&CancellationToken::default());
                let header = used(&owner);
                let mut input = parts(&owner, 3, 5, strided);
                input[role].storage[[1, 0]] = value;
                let expected = reference(&input);
                let bytes = used(&owner);
                let output = ClaimedKernelOutput::from_parts(input, None).unwrap();
                assert_eq!(bound_bits(output.bounds()), bound_bits(&expected));
                assert_eq!(used(&owner), bytes);
                drop(output);
                clean(&memory, &query, owner, header);
            }
        }
        let (memory, query, owner) = joined(&CancellationToken::default());
        let header = used(&owner);
        let mut input = parts(&owner, 3, 5, strided);
        input[4] = BoundedArray::try_from_fn(&owner, 3, 4, None, |_, _| 1.0).unwrap();
        let expected = reference(&input);
        let bytes = used(&owner);
        let output = ClaimedKernelOutput::from_parts(input, None).unwrap();
        assert_eq!(bound_bits(output.bounds()), bound_bits(&expected));
        assert!(output.bounds().lower_a_err().is_none() && output.bounds().upper_a_err().is_none());
        assert_eq!(
            used(&owner),
            bytes,
            "discarded error storage stays conservatively charged"
        );
        drop(output);
        clean(&memory, &query, owner, header);
    }
}

#[test]
fn claimed_output_rejects_every_foreign_plane_and_unjoined_authority() {
    for role in 0..6 {
        let (memory, query, owner) = joined(&CancellationToken::default());
        let header = used(&owner);
        let (other_memory, other_query, other) = joined(&CancellationToken::default());
        let other_header = used(&other);
        let mut input = parts(&owner, 3, 5, false);
        let cols = if role == 1 || role == 3 { 1 } else { 5 };
        input[role] = BoundedArray::try_from_fn(&other, 3, cols, None, |_, _| 1.0).unwrap();
        assert!(ClaimedKernelOutput::from_parts(input, None).is_err());
        clean(&other_memory, &other_query, other, other_header);
        clean(&memory, &query, owner, header);
    }
    // A shared ledger, class AND original retained reservation still cannot
    // substitute a different local payload account for any of the six planes.
    for role in 0..6 {
        let (memory, query, owner, retained) = joined_retained(&CancellationToken::default());
        let header = used(&owner);
        let other =
            SoaStorageBudget::from_query(&query, retained.clone(), LeaseClass::Fallback, None)
                .unwrap();
        let mut input = parts(&owner, 3, 5, false);
        input[role] = BoundedArray::try_from_fn(
            &other,
            3,
            if role == 1 || role == 3 { 1 } else { 5 },
            None,
            |_, _| 1.0,
        )
        .unwrap();
        assert!(ClaimedKernelOutput::from_parts(input, None).is_err());
        drop(other);
        drop(retained);
        clean(&memory, &query, owner, header);
    }
    let owner = SoaStorageBudget::new(0, None).unwrap();
    assert!(ClaimedKernelOutput::from_parts(parts(&owner, 3, 5, false), None).is_err());
    assert_eq!(used(&owner), 0);
}

#[test]
fn claimed_output_hard_shapes_and_bias_representation_refuse_without_claim_leaks() {
    for (role, width) in [(0, 5), (1, 1), (2, 5), (3, 1), (1, 2), (3, 2)] {
        let (memory, query, owner) = joined(&CancellationToken::default());
        let header = used(&owner);
        let mut input = parts(&owner, 3, 5, false);
        input[role] = BoundedArray::try_from_fn(&owner, 4, width, None, |_, _| 1.0).unwrap();
        assert!(ClaimedKernelOutput::from_parts(input, None).is_err());
        clean(&memory, &query, owner, header);
    }
}

#[test]
fn claimed_output_refusal_at_every_metadata_scan_repair_sanitize_and_final_boundary_discards_all() {
    for variant in 0..4 {
        let mut terminal = 0;
        for stop in 0..128 {
            let (memory, query, owner) = joined(&CancellationToken::default());
            let header = used(&owner);
            let (rows, cols) = if variant >= 2 { (3, 5) } else { (5, 257) };
            let mut input = parts(&owner, rows, cols, variant >= 2);
            if variant == 1 {
                input[0].storage[[2, 3]] = f32::NAN;
            }
            if variant == 2 {
                input[5] =
                    BoundedArray::try_from_fn(&owner, rows, cols - 1, None, |_, _| 1.0).unwrap();
            }
            let mut calls = 0;
            let result = ClaimedKernelOutput::observed(input, None, |n| {
                calls = n;
                if stop != 0 && n == stop {
                    return Err(ny_core::NyError::DeadlineExceeded(
                        "forced publication boundary".into(),
                    ));
                }
                Ok(())
            });
            if stop == 0 {
                terminal = calls;
                assert!(terminal > 6 && terminal < 128);
                drop(result.unwrap());
            } else {
                assert!(matches!(result,Err(e) if e.is_deadline_exceeded()));
                assert_eq!(calls, stop);
            }
            clean(&memory, &query, owner, header);
            if stop == terminal {
                break;
            }
        }
    }
}

#[test]
fn claimed_output_original_cancel_and_narrowed_deadline_remain_authoritative() {
    let (memory, query, owner) = joined(&CancellationToken::default());
    let header = used(&owner);
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut input = parts(&owner, 3, 5, false);
    input[4].deadline = Some(deadline);
    let output =
        ClaimedKernelOutput::from_parts(input, Some(deadline + Duration::from_secs(30))).unwrap();
    assert_eq!(output.deadline(), Some(deadline));
    drop(output);
    assert!(ClaimedKernelOutput::from_parts(
        parts(&owner, 3, 5, false),
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    clean(&memory, &query, owner, header);
    let mut final_boundary = 0;
    {
        let (memory, query, owner) = joined(&CancellationToken::default());
        let header = used(&owner);
        let output = ClaimedKernelOutput::observed(parts(&owner, 3, 5, false), None, |n| {
            final_boundary = n;
            Ok(())
        })
        .unwrap();
        drop(output);
        clean(&memory, &query, owner, header);
    }
    for stop in [1, final_boundary] {
        let cancel = CancellationToken::default();
        let (memory, query, owner) = joined(&cancel);
        let header = used(&owner);
        let result = ClaimedKernelOutput::observed(parts(&owner, 3, 5, false), None, |n| {
            if n == stop {
                cancel.cancel();
            }
            Ok(())
        });
        assert!(matches!(result,Err(e) if e.is_deadline_exceeded()));
        clean(&memory, &query, owner, header);
    }
}

#[test]
fn claimed_output_debits_exact_existing_query_work_without_new_phase_or_budget() {
    let (memory, query, owner) = joined(&CancellationToken::default());
    let header = used(&owner);
    let input = parts(&owner, 5, 257, false);
    let before = query.receipt().unwrap();
    let output = ClaimedKernelOutput::from_parts(input, None).unwrap();
    let after = query.receipt().unwrap();
    assert_eq!(
        after.work_reserved - before.work_reserved,
        448 + 8 * (4 * 5 * 257 + 2 * 5)
    );
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(
        std::ptr::from_ref(output.owner().as_ref()),
        std::ptr::from_ref(owner.as_ref())
    );
    drop(output);
    clean(&memory, &query, owner, header);
}

fn plan() -> CrownDispatchPlan {
    let mut graph = GraphNetwork::new();
    graph
        .try_add_node(GraphNode::from_input(
            "out",
            Layer::Linear(LinearLayer::new(Array2::eye(2), None).unwrap()),
        ))
        .unwrap();
    graph.set_output("out");
    CrownDispatchPlan::build(&graph).unwrap()
}

#[test]
fn claimed_output_and_legacy_materialization_use_one_borrowed_merge_and_preserve_sources_on_refusal(
) {
    let plan = plan();
    let idx = plan.output_node_idx;
    let (memory, query, owner) = joined(&CancellationToken::default());
    let header = used(&owner);
    let input = parts(&owner, 2, 5, true);
    let expected = reference(&input);
    let output = ClaimedKernelOutput::from_parts(input, None).unwrap();
    let original = bound_bits(output.bounds());
    let mut actual = BoundedBatchPending::try_new(&plan, 2, &owner, None).unwrap();
    let mut legacy = BoundedBatchPending::try_new(&plan, 2, &owner, None).unwrap();
    // The legacy fixture owns its original complete declared payload claim.
    let mut poll = Poll::for_owner(&owner, None);
    let mut charge = owner.charge(expected.memory_bytes(), &mut poll).unwrap();
    charge.materialize().unwrap();
    // The fixture poll owns a QueryAuthority Arc (including its header lease).
    // It is no longer used; release it before asserting terminal owner cleanup.
    drop(poll);
    let previous = BoundedLinearBounds {
        bounds: expected,
        _charge: charge,
        deadline: None,
    };
    for domain in [0, 1, 0] {
        actual = actual.accumulate_domain(idx, domain, &output).unwrap();
        legacy = legacy.accumulate_domain(idx, domain, &previous).unwrap();
    }
    let (actual, slot) = actual.take(idx).unwrap();
    let (legacy, old_slot) = legacy.take(idx).unwrap();
    let slot = slot.unwrap();
    let old_slot = old_slot.unwrap();
    for domain in 0..2 {
        let a = slot.try_materialize(domain, None).unwrap().unwrap();
        let b = old_slot.try_materialize(domain, None).unwrap().unwrap();
        assert_eq!(bound_bits(a.bounds()), bound_bits(b.bounds()));
    }
    assert_eq!(bound_bits(output.bounds()), original);
    let (foreign_memory, foreign_query, foreign) = joined(&CancellationToken::default());
    let foreign_header = used(&foreign);
    let pending = BoundedBatchPending::try_new(&plan, 2, &foreign, None).unwrap();
    assert!(pending.accumulate_domain(idx, 0, &output).is_err());
    assert_eq!(bound_bits(output.bounds()), original);
    clean(&foreign_memory, &foreign_query, foreign, foreign_header);
    drop(actual);
    drop(legacy);
    drop(slot);
    drop(old_slot);
    drop(previous);
    drop(output);
    clean(&memory, &query, owner, header);
}

#[test]
fn claimed_empty_array_explicit_zero_axes_keep_zero_capacity_and_legacy_refusals() {
    for (rows, cols) in [(0, 0), (0, 7), (3, 0), (0, 1 << 30), (1 << 30, 0)] {
        let (memory, query, owner) = joined(&CancellationToken::default());
        let header = used(&owner);
        let before = query.receipt().unwrap();
        let a = BoundedArray::<f64>::try_from_fn_allow_empty(&owner, rows, cols, None, |_, _| {
            panic!("empty getter executed")
        })
        .unwrap();
        assert_eq!(a.array().dim(), (rows, cols));
        assert_eq!(a.array().len(), 0);
        assert_eq!(a._charge.bytes, 0);
        assert_eq!(used(&owner), header);
        assert_eq!(
            query.receipt().unwrap().live_scratch_bytes,
            before.live_scratch_bytes
        );
        let BoundedArray {
            storage,
            _charge,
            deadline: _,
        } = a;
        let (raw, _) = storage.into_raw_vec_and_offset();
        assert_eq!(raw.capacity(), 0);
        assert!(
            BoundedArray::<f64>::try_from_fn(&owner, rows, cols, None, |_, _| panic!(
                "legacy empty getter executed"
            ))
            .is_err()
        );
        assert!(
            BoundedMat::<f64>::try_from_fn(&owner, rows, cols, None, |_, _| panic!(
                "native empty getter executed"
            ))
            .is_err()
        );
        drop(raw);
        drop(_charge);
        clean(&memory, &query, owner, header);
    }
}

#[test]
fn claimed_empty_array_checked_overflow_and_every_empty_publication_fault_drop_claims() {
    let (memory, query, owner) = joined(&CancellationToken::default());
    let header = used(&owner);
    for (rows, cols) in [(usize::MAX, 2), (usize::MAX, 1)] {
        assert!(BoundedArray::<f32>::try_from_fn_allow_empty(
            &owner,
            rows,
            cols,
            None,
            |_, _| panic!("overflow getter executed")
        )
        .is_err());
        assert_eq!(used(&owner), header);
    }
    for stage in [ENTRY, AFTER_RESERVE, COMMIT] {
        let mut poll = BufferPoll::for_owner(&owner, None);
        poll.shared.forced_stage = Some(stage);
        let result = BoundedArray::<f32>::with_poll_layout(
            &owner,
            1 << 30,
            0,
            None,
            true,
            |_, _| panic!("empty getter executed"),
            &mut poll,
        );
        assert!(matches!(result,Err(e) if e.is_deadline_exceeded()));
        assert_eq!(poll.observed_units, 0);
        assert_eq!(used(&owner), header);
        drop(poll);
    }
    clean(&memory, &query, owner, header);
}

#[test]
fn claimed_empty_array_keeps_original_query_cancellation_and_absolute_deadline() {
    let cancel = CancellationToken::default();
    let (memory, query, owner) = joined(&cancel);
    let header = used(&owner);
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(
        matches!(BoundedArray::<f32>::try_from_fn_allow_empty(&owner,0,7,Some(past),|_,_|panic!("expired empty getter")),Err(e) if e.is_deadline_exceeded())
    );
    assert_eq!(used(&owner), header);
    cancel.cancel();
    assert!(
        matches!(BoundedArray::<f64>::try_from_fn_allow_empty(&owner,3,0,None,|_,_|panic!("cancelled empty getter")),Err(e) if e.is_deadline_exceeded())
    );
    assert_eq!(used(&owner), header);
    clean(&memory, &query, owner, header);
}

#[test]
fn claimed_empty_array_fixture_poll_retains_query_header_until_its_explicit_drop() {
    let (memory, query, owner) = joined(&CancellationToken::default());
    let header = used(&owner);
    assert_eq!(Arc::strong_count(owner.query.as_ref().unwrap()), 1);
    let before = query.receipt().unwrap();
    let poll = Poll::for_owner(&owner, None);
    assert_eq!(Arc::strong_count(owner.query.as_ref().unwrap()), 2);
    assert_eq!(query.receipt().unwrap().admitted, before.admitted);
    drop(owner);
    assert_eq!(Arc::strong_count(poll.query.as_ref().unwrap()), 1);
    assert_eq!(query.receipt().unwrap().live_scratch_bytes, header);
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        RETAINED + header
    );
    drop(poll);
    assert_eq!(query.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}
