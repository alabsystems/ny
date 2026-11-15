// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve};
use crate::layers::linear::crown_single::{
    aw_f64_with_abssum_chunked_poll, aw_f64_with_abssum_chunked_poll_with_storage, WidenedWeightF64,
};
use faer::Mat;
use std::sync::atomic::Ordering;
use std::time::Duration;

const RETAINED: usize = 1 << 16;

fn joined(
    cancel: &CancellationToken,
    work: u64,
    deadline: Option<Instant>,
    class: LeaseClass,
    reserve: Duration,
) -> (MemoryAuthority, WorkBudget, Arc<SoaStorageBudget>) {
    let memory = MemoryAuthority::new(1 << 24);
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
            deadline,
            scratch_bytes: 1 << 22,
            persistent_bytes: 1 << 22,
            max_work: work,
            max_proposals: 0,
            certificate_reserve: reserve,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let owner = SoaStorageBudget::from_query(&query, retained, class, None).unwrap();
    (memory, query, owner)
}

fn fixture(m: usize, k: usize, p: usize) -> (Mat<f32>, Mat<f32>) {
    let values = [
        0.0,
        -0.0,
        1.25,
        -0.375,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
    ];
    (
        Mat::from_fn(m, k, |i, j| values[(i * 7 + j * 3) % values.len()]),
        Mat::from_fn(k, p, |i, j| values[(i * 5 + j * 7 + 1) % values.len()]),
    )
}
fn input_bits(a: &Mat<f32>, w: &Mat<f32>) -> Vec<u32> {
    [a, w]
        .into_iter()
        .flat_map(|matrix| {
            (0..matrix.ncols())
                .flat_map(move |j| (0..matrix.nrows()).map(move |i| matrix[(i, j)].to_bits()))
        })
        .collect()
}
fn used(owner: &SoaStorageBudget) -> usize {
    owner.reserved_bytes.load(Ordering::Acquire)
}
fn cleaned(
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

#[test]
fn leased_shared_aw_preserves_bits_layout_and_query_work_for_ragged_blocks() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        assert!(rayon::current_thread_index().is_some());
        for nested in [false, true] {
            let _guard = nested.then(crate::faer_parallelism::NestedFaerParGuard::new);
            assert_eq!(
                matches!(crate::faer_parallelism::current_par(), faer::Par::Seq),
                !nested
            );
            for (m, k, p) in [(10, 5, 7), (6, 3, 11), (66, 3, 5)] {
                let (a, w) = fixture(m, k, p);
                let original = input_bits(&a, &w);
                let quantum = 4 * k * p;
                let (expected_a, expected_s, blocks) =
                    aw_f64_with_abssum_chunked_poll(&a, &w, quantum, &mut || Ok(()))
                        .unwrap()
                        .unwrap();
                for requested in [None, Some(Instant::now() + Duration::from_secs(60))] {
                    let (memory, query, owner) = joined(
                        &CancellationToken::default(),
                        1 << 30,
                        None,
                        LeaseClass::Fallback,
                        Duration::ZERO,
                    );
                    let header = used(&owner);
                    let mut storage = LeasedAwStorage::new(&owner, requested).unwrap();
                    let (actual_a, actual_s, actual_blocks) =
                        aw_f64_with_abssum_chunked_poll_with_storage(
                            a.as_ref(),
                            w.as_ref(),
                            None,
                            quantum,
                            &mut || Ok(()),
                            &mut storage,
                        )
                        .unwrap()
                        .unwrap();
                    assert_eq!(actual_blocks, blocks);
                    assert_eq!(
                        actual_a.array().mapv(f64::to_bits),
                        expected_a.mapv(f64::to_bits)
                    );
                    assert_eq!(
                        actual_s.array().mapv(f64::to_bits),
                        expected_s.mapv(f64::to_bits)
                    );
                    assert_eq!(actual_a.deadline(), requested);
                    assert_eq!(actual_s.deadline(), requested);
                    assert!(query.receipt().unwrap().work_reserved >= (8 * m * k * p) as u64);
                    assert_eq!(query.receipt().unwrap().live_scratch_bytes, used(&owner));
                    assert_eq!(
                        memory.receipt().unwrap().live_materialized_bytes,
                        RETAINED + used(&owner)
                    );
                    assert_eq!(input_bits(&a, &w), original);
                    drop(storage);
                    drop(actual_a);
                    assert!(
                        used(&owner) > header,
                        "the other output must retain its claim"
                    );
                    drop(actual_s);
                    cleaned(&memory, &query, owner, header);
                }
            }
        }
    });
}

#[test]
fn leased_shared_aw_refuses_hidden_scratch_singleton_tail_and_unbound_cache_before_buffers() {
    let (memory, query, owner) = joined(
        &CancellationToken::default(),
        1 << 30,
        None,
        LeaseClass::Fallback,
        Duration::ZERO,
    );
    let header = used(&owner);
    let mut storage = LeasedAwStorage::new(&owner, None).unwrap();
    for (m, k, p, quantum) in [
        (4, 17, 17, 4 * 17 * 17),
        (5, 3, 5, 60),
        (1, 3, 5, 60),
        (4, 1, 5, 20),
        (4, 3, 1, 12),
    ] {
        let (a, w) = fixture(m, k, p);
        let original = input_bits(&a, &w);
        assert!(matches!(
            aw_f64_with_abssum_chunked_poll_with_storage(
                a.as_ref(),
                w.as_ref(),
                None,
                quantum,
                &mut || Ok(()),
                &mut storage
            ),
            Err(NyError::UnsupportedOp(_))
        ));
        assert_eq!(input_bits(&a, &w), original);
        assert_eq!(used(&owner), header);
        assert_eq!(query.receipt().unwrap().work_reserved, 64);
    }
    let (a, w) = fixture(4, 3, 5);
    let cached = WidenedWeightF64::new(&w).unwrap();
    assert!(matches!(
        aw_f64_with_abssum_chunked_poll_with_storage(
            a.as_ref(),
            w.as_ref(),
            Some(&cached),
            60,
            &mut || Ok(()),
            &mut storage
        ),
        Err(NyError::UnsupportedOp(_))
    ));
    assert!(LeasedAwStorage::product_units(usize::MAX, 2, 2).is_err());
    assert_eq!(query.receipt().unwrap().work_reserved, 64);
    drop(storage);
    cleaned(&memory, &query, owner, header);
}

#[test]
fn leased_shared_aw_work_and_byte_refusals_leave_inputs_and_existing_claims() {
    for work_refusal in [false, true] {
        let (memory, query, owner) = joined(
            &CancellationToken::default(),
            if work_refusal { 64 } else { 1 << 30 },
            None,
            LeaseClass::Fallback,
            Duration::ZERO,
        );
        let header = used(&owner);
        if !work_refusal {
            query.tighten_total_byte_limit(header).unwrap();
        }
        let (a, w) = fixture(10, 5, 7);
        let original = input_bits(&a, &w);
        let mut storage = LeasedAwStorage::new(&owner, None).unwrap();
        assert!(aw_f64_with_abssum_chunked_poll_with_storage(
            a.as_ref(),
            w.as_ref(),
            None,
            140,
            &mut || Ok(()),
            &mut storage
        )
        .is_err());
        assert_eq!(input_bits(&a, &w), original);
        drop(storage);
        cleaned(&memory, &query, owner, header);
    }
}

#[test]
fn leased_shared_aw_original_cancellation_after_first_real_product_discards_output() {
    let cancel = CancellationToken::default();
    let (memory, query, owner) =
        joined(&cancel, 1 << 30, None, LeaseClass::Fallback, Duration::ZERO);
    let header = used(&owner);
    let (a, w) = fixture(10, 5, 7);
    let original = input_bits(&a, &w);
    let mut calls = 0;
    let mut storage = LeasedAwStorage::new(&owner, None).unwrap();
    let result = aw_f64_with_abssum_chunked_poll_with_storage(
        a.as_ref(),
        w.as_ref(),
        None,
        140,
        &mut || {
            calls += 1;
            if calls == 6 {
                cancel.cancel();
            } // Original poll immediately after the first full-k GEMM.
            Ok(())
        },
        &mut storage,
    );
    assert!(matches!(result, Err(error) if error.is_deadline_exceeded()));
    assert_eq!(calls, 6);
    assert!(query.receipt().unwrap().work_reserved > 64 + 4 * 4 * 5 * 7);
    assert!(memory.receipt().unwrap().peak_leased_bytes > RETAINED + header);
    assert_eq!(input_bits(&a, &w), original);
    drop(storage);
    cleaned(&memory, &query, owner, header);
}

#[test]
fn leased_shared_aw_final_authority_refusal_keeps_private_results_unpublished() {
    let (a, w) = fixture(10, 5, 7);
    let original = input_bits(&a, &w);
    let mut terminal_call = 0;
    for refuse in [false, true] {
        let cancel = CancellationToken::default();
        let (memory, query, owner) =
            joined(&cancel, 1 << 30, None, LeaseClass::Fallback, Duration::ZERO);
        let header = used(&owner);
        let mut calls = 0;
        let mut storage = LeasedAwStorage::new(&owner, None).unwrap();
        let result = aw_f64_with_abssum_chunked_poll_with_storage(
            a.as_ref(),
            w.as_ref(),
            None,
            140,
            &mut || {
                calls += 1;
                if refuse && calls == terminal_call {
                    cancel.cancel();
                }
                Ok(())
            },
            &mut storage,
        );
        if refuse {
            assert!(matches!(result, Err(error) if error.is_deadline_exceeded()));
            assert_eq!(calls, terminal_call);
        } else {
            assert!(result.is_ok());
            drop(result);
            terminal_call = calls;
            assert!(terminal_call > 6);
        }
        assert_eq!(input_bits(&a, &w), original);
        drop(storage);
        cleaned(&memory, &query, owner, header);
    }
}

#[test]
fn leased_shared_aw_keeps_original_class_deadline_and_rejects_unjoined_storage() {
    let now = Instant::now();
    let original_deadline = now + Duration::from_secs(60);
    let (memory, query, owner) = joined(
        &CancellationToken::default(),
        1 << 30,
        Some(original_deadline),
        LeaseClass::Optional,
        Duration::from_secs(30),
    );
    let header = used(&owner);
    let mut storage = LeasedAwStorage::new(&owner, Some(original_deadline)).unwrap();
    assert_eq!(
        storage.deadline,
        Some(
            original_deadline
                .checked_sub(Duration::from_secs(30))
                .unwrap()
        )
    );
    let (a, w) = fixture(6, 3, 5);
    let (aw, abs, _) = aw_f64_with_abssum_chunked_poll_with_storage(
        a.as_ref(),
        w.as_ref(),
        None,
        60,
        &mut || Ok(()),
        &mut storage,
    )
    .unwrap()
    .unwrap();
    assert_eq!(aw.deadline(), storage.deadline);
    assert_eq!(abs.deadline(), storage.deadline);
    drop(aw);
    drop(abs);
    drop(storage);
    assert!(
        matches!(LeasedAwStorage::new(&owner, Some(now.checked_sub(Duration::from_secs(1)).unwrap())), Err(error) if error.is_deadline_exceeded())
    );
    let unjoined = SoaStorageBudget::new(0, None).unwrap();
    assert!(LeasedAwStorage::new(&unjoined, None).is_err());
    cleaned(&memory, &query, owner, header);
}
