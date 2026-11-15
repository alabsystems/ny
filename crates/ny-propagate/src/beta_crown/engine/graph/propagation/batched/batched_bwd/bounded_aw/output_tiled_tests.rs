// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve};
use crate::layers::linear::bias::{f32_to_f64_exact, publish_error_up_normal};
use crate::layers::linear::crown_single::{
    aw_f64_with_abssum_chunked_poll, aw_f64_with_abssum_chunked_poll_with_storage, gamma_n_f64,
};
use faer::Mat;
use num_rational::BigRational;
use num_traits::Signed;
use std::sync::atomic::Ordering;
use std::time::Duration;

const RETAINED: usize = 1 << 16;

fn joined(
    cancel: &CancellationToken,
    work: u64,
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
            deadline: None,
            scratch_bytes: 1 << 22,
            persistent_bytes: 1 << 22,
            max_work: work,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let owner = SoaStorageBudget::from_query(&query, retained, LeaseClass::Fallback, None).unwrap();
    (memory, query, owner)
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

fn rational(value: f64) -> BigRational {
    BigRational::from_float(value).expect("finite exact float")
}

fn fixture(m: usize, k: usize, p: usize) -> (Mat<f32>, Mat<f32>) {
    // Cancellation, mixed exponents, both zero signs and true f32 subnormals.
    // Products remain finite binary64 and published centers finite binary32.
    let bits = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x807f_ffff,
        0x0080_0000,
        0x3f80_0001,
        0xbf80_0000,
        0x5380_0001,
        0xd380_0000,
        0x3080_0123,
        0xb080_0000,
        0x4080_0001,
    ];
    (
        Mat::from_fn(m, k, |i, j| {
            f32::from_bits(bits[(i * 7 + j * 5) % bits.len()])
        }),
        Mat::from_fn(k, p, |i, j| {
            f32::from_bits(bits[(i * 5 + j * 7 + 1) % bits.len()])
        }),
    )
}

fn bits(a: &Mat<f32>, w: &Mat<f32>) -> Vec<u32> {
    [a, w]
        .into_iter()
        .flat_map(|matrix| {
            (0..matrix.ncols())
                .flat_map(move |j| (0..matrix.nrows()).map(move |i| matrix[(i, j)].to_bits()))
        })
        .collect()
}

#[test]
fn output_plan_covers_every_cell_once_with_literal4096_work_and_no_singletons() {
    assert_eq!(POLL_CHUNK_UNITS, 4096);
    for m in 2..18 {
        for k in 2..81 {
            for p in 2..26 {
                let possible = (2 + m % 2) * k * (2 + p % 2) <= 1024;
                let plan = OutputPlan::new(m, k, p);
                assert_eq!(plan.is_ok(), possible, "{m}x{k}x{p}");
                let Ok(plan) = plan else { continue };
                let mut coverage = vec![0u8; m * p];
                let mut row = 0;
                while row < m {
                    let rows = OutputPlan::block(m - row, plan.row_limit);
                    let mut col = 0;
                    while col < p {
                        let cols = OutputPlan::block(p - col, plan.col_limit);
                        assert!(rows >= 2 && cols >= 2);
                        assert!(4 * rows * k * cols <= 4096);
                        for i in row..row + rows {
                            for j in col..col + cols {
                                coverage[i * p + j] += 1;
                            }
                        }
                        col += cols;
                    }
                    row += rows;
                }
                assert!(coverage.into_iter().all(|n| n == 1));
            }
        }
    }
    assert!(OutputPlan::new(usize::MAX, 2, 2).is_err());
    assert!(OutputPlan::new(2, usize::MAX, 2).is_err());
    for shape in [
        (0, 50, 50),
        (1, 50, 50),
        (2, 1, 50),
        (2, 50, 1),
        (2, 257, 2),
    ] {
        assert!(OutputPlan::new(shape.0, shape.1, shape.2).is_err());
    }
}

#[test]
fn tiled_shared_aw_original_model_shapes_enclose_exact_rationals_in_both_parallel_scopes() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        for nested in [false, true] {
            let _guard = nested.then(crate::faer_parallelism::NestedFaerParGuard::new);
            assert_eq!(
                matches!(crate::faer_parallelism::current_par(), faer::Par::Seq),
                !nested
            );
            for (m, k, p, rows) in [
                (2, 50, 50, 2),
                (6, 50, 50, 2),
                (3, 50, 50, 3),
                (4, 50, 5, 4),
                (4, 5, 50, 4),
                (7, 17, 19, 4),
                (66, 3, 5, 66),
            ] {
                let (a, w) = fixture(m, k, p);
                let original = bits(&a, &w);
                let (twin_a, twin_s, twin_blocks) =
                    aw_f64_with_abssum_chunked_poll(&a, &w, rows * k * p, &mut || Ok(()))
                        .unwrap()
                        .unwrap();
                let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 30);
                let header = used(&owner);
                let mut storage = OutputTiledLeasedAwStorage::new(&owner, None).unwrap();
                let (actual_a, actual_s, blocks) = aw_f64_with_abssum_chunked_poll_with_storage(
                    a.as_ref(),
                    w.as_ref(),
                    None,
                    rows * k * p,
                    &mut || Ok(()),
                    &mut storage,
                )
                .unwrap()
                .unwrap();
                assert_eq!(blocks, twin_blocks);
                assert_eq!(actual_a.array().strides(), &[p as isize, 1]);
                assert_eq!(actual_s.array().strides(), &[p as isize, 1]);
                let gamma = gamma_n_f64(k);
                for i in 0..m {
                    for j in 0..p {
                        let mut exact = rational(0.0);
                        let mut exact_s = rational(0.0);
                        for kk in 0..k {
                            let term = rational(f32_to_f64_exact(a[(i, kk)]))
                                * rational(f32_to_f64_exact(w[(kk, j)]));
                            exact_s += term.abs();
                            exact += term;
                        }
                        let center = actual_a.array()[[i, j]];
                        let s = actual_s.array()[[i, j]];
                        assert!((rational(center) - &exact).abs() <= rational(gamma * s));
                        assert!((rational(s) - &exact_s).abs() <= rational(gamma * s));
                        let stored = center as f32;
                        let cast = (center - f64::from(stored)).abs();
                        let error = f64::from(publish_error_up_normal(cast + gamma * s));
                        assert!(rational(f64::from(stored)) - rational(error) <= exact);
                        assert!(exact <= rational(f64::from(stored)) + rational(error));
                        assert!(
                            (rational(center) - rational(twin_a[[i, j]])).abs()
                                <= rational(gamma * (s + twin_s[[i, j]]))
                        );
                    }
                }
                assert_eq!(bits(&a, &w), original);
                assert!(query.receipt().unwrap().work_reserved >= (8 * m * k * p) as u64);
                assert_eq!(query.receipt().unwrap().live_scratch_bytes, used(&owner));
                drop(storage);
                drop(actual_a);
                assert!(used(&owner) > header);
                drop(actual_s);
                cleaned(&memory, &query, owner, header);
            }
        }
    });
}

#[test]
fn tiled_product_preserves_strided_sources_and_column_major_padding() {
    let (m, k, p) = (3, 50, 51);
    let lhs_stride = m + 4;
    let rhs_stride = p + 5;
    let lhs = vec![2.0; lhs_stride * k];
    let rhs = vec![3.0; rhs_stride * k];
    let a = MatRef::from_column_major_slice_with_stride(&lhs, m, k, lhs_stride);
    let b = MatRef::from_row_major_slice_with_stride(&rhs, k, p, rhs_stride);
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 30);
    let header = used(&owner);
    let mut storage = OutputTiledLeasedAwStorage::new(&owner, None).unwrap();
    let dst = storage.product(a, b).unwrap();
    let view = dst.view();
    assert_eq!(view.row_stride(), 1);
    assert_eq!(view.col_stride(), 8);
    for j in 0..p {
        for i in 0..m {
            assert_eq!(view[(i, j)], 300.0);
        }
    }
    let raw: &[f64] = bytemuck::cast_slice(&dst.storage);
    for j in 0..p {
        for i in m..8 {
            assert_eq!(raw[j * 8 + i].to_bits(), 0);
        }
    }
    assert!(lhs.iter().all(|v| *v == 2.0));
    assert!(rhs.iter().all(|v| *v == 3.0));
    drop(dst);
    drop(storage);
    cleaned(&memory, &query, owner, header);
}

fn zero_cost() -> u64 {
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 30);
    let header = used(&owner);
    let before = query.receipt().unwrap().work_reserved;
    let mut storage = OutputTiledLeasedAwStorage::new(&owner, None).unwrap();
    let zero = storage.matrix_zeros(2, 50).unwrap();
    let cost = query.receipt().unwrap().work_reserved - before;
    drop(zero);
    drop(storage);
    cleaned(&memory, &query, owner, header);
    cost
}

#[test]
fn tiled_product_debits_literal20000_native_units_before_five_full_k_calls() {
    let a = Mat::from_fn(2, 50, |_, _| 2.0);
    let b = Mat::from_fn(50, 50, |_, _| 3.0);
    let fill = zero_cost();
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 30);
    let header = used(&owner);
    let before = query.receipt().unwrap().work_reserved;
    let mut storage = OutputTiledLeasedAwStorage::new(&owner, None).unwrap();
    let mut completed = 0;
    let result = storage
        .product_observed(a.as_ref(), b.as_ref(), |n| {
            completed = n;
            assert_eq!(
                query.receipt().unwrap().work_reserved - before,
                fill + 4000 * n as u64
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(completed, 5);
    assert_eq!(
        query.receipt().unwrap().work_reserved - before,
        fill + 20_000
    );
    drop(result);
    drop(storage);
    cleaned(&memory, &query, owner, header);
}

#[test]
fn tiled_product_original_cancellation_after_every_native_call_discards_private_output() {
    let a = Mat::from_fn(2, 50, |i, j| (i + j) as f64);
    let b = Mat::from_fn(50, 50, |i, j| (i * 3 + j) as f64);
    for stop in 1..=5 {
        let cancel = CancellationToken::default();
        let (memory, query, owner) = joined(&cancel, 1 << 30);
        let header = used(&owner);
        let mut storage = OutputTiledLeasedAwStorage::new(&owner, None).unwrap();
        let mut completed = 0;
        let result = storage.product_observed(a.as_ref(), b.as_ref(), |n| {
            completed = n;
            if n == stop {
                cancel.cancel();
            }
            Ok(())
        });
        assert!(matches!(result, Err(error) if error.is_deadline_exceeded()));
        assert_eq!(completed, stop);
        for i in 0..2 {
            for j in 0..50 {
                assert_eq!(a[(i, j)], (i + j) as f64);
            }
        }
        for i in 0..50 {
            for j in 0..50 {
                assert_eq!(b[(i, j)], (i * 3 + j) as f64);
            }
        }
        drop(storage);
        cleaned(&memory, &query, owner, header);
    }
}

#[test]
fn tiled_product_work_limit_refuses_before_every_unaffordable_native_call() {
    let fill = zero_cost();
    let a = Mat::from_fn(2, 50, |_, _| 2.0);
    let b = Mat::from_fn(50, 50, |_, _| 3.0);
    for affordable in 0..5 {
        let limit = 64 + fill + 4000 * affordable;
        let (memory, query, owner) = joined(&CancellationToken::default(), limit);
        let header = used(&owner);
        let mut storage = OutputTiledLeasedAwStorage::new(&owner, None).unwrap();
        let mut completed = 0;
        assert!(storage
            .product_observed(a.as_ref(), b.as_ref(), |n| {
                completed = n;
                Ok(())
            })
            .is_err());
        assert_eq!(completed as u64, affordable);
        assert_eq!(query.receipt().unwrap().work_reserved, limit);
        drop(storage);
        cleaned(&memory, &query, owner, header);
    }
}

#[test]
fn tiled_shared_aw_rejects_unbound_cache_and_impossible_tail_before_any_buffers() {
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 30);
    let header = used(&owner);
    let mut storage = OutputTiledLeasedAwStorage::new(&owner, None).unwrap();
    for (m, k, p, rows, cached) in [
        (5, 50, 50, 4, false),
        (1, 50, 50, 2, false),
        (2, 257, 2, 2, false),
        (3, 171, 3, 3, false),
        (2, 50, 50, 2, true),
        (2, 50, 50, 0, false),
    ] {
        assert!(storage.admit_products(m, k, p, rows, cached).is_err());
        assert_eq!(query.receipt().unwrap().work_reserved, 64);
        assert_eq!(used(&owner), header);
    }
    let (a, w) = fixture(5, 50, 50);
    let original = bits(&a, &w);
    assert!(aw_f64_with_abssum_chunked_poll_with_storage(
        a.as_ref(),
        w.as_ref(),
        None,
        4 * 50 * 50,
        &mut || Ok(()),
        &mut storage,
    )
    .is_err());
    assert_eq!(bits(&a, &w), original);
    assert_eq!(query.receipt().unwrap().work_reserved, 64);
    drop(storage);
    cleaned(&memory, &query, owner, header);
}

#[test]
fn tiled_shared_aw_every_outer_poll_refusal_publishes_nothing_and_releases_scratch() {
    let (a, w) = fixture(6, 50, 50);
    let original = bits(&a, &w);
    let mut count = 0;
    for stop in 0..32 {
        let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 30);
        let header = used(&owner);
        let mut storage = OutputTiledLeasedAwStorage::new(&owner, None).unwrap();
        let mut calls = 0;
        let result = aw_f64_with_abssum_chunked_poll_with_storage(
            a.as_ref(),
            w.as_ref(),
            None,
            2 * 50 * 50,
            &mut || {
                calls += 1;
                if stop != 0 && calls == stop {
                    return Err(NyError::DeadlineExceeded("forced outer seam".into()));
                }
                Ok(())
            },
            &mut storage,
        );
        if stop == 0 {
            count = calls;
            assert!(count > 6 && count < 32);
            drop(result.unwrap().unwrap());
        } else {
            assert!(matches!(result, Err(error) if error.is_deadline_exceeded()));
            assert_eq!(calls, stop);
        }
        assert_eq!(bits(&a, &w), original);
        drop(storage);
        cleaned(&memory, &query, owner, header);
        if stop == count {
            break;
        }
    }
}

#[test]
fn tiled_storage_keeps_original_deadline_cancellation_and_byte_authority() {
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 30);
    let header = used(&owner);
    assert!(OutputTiledLeasedAwStorage::new(
        &owner,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut storage = OutputTiledLeasedAwStorage::new(&owner, Some(deadline)).unwrap();
    let result = storage.matrix_zeros(2, 50).unwrap();
    assert_eq!(result.deadline(), Some(deadline));
    drop(result);
    query.tighten_total_byte_limit(header).unwrap();
    assert!(storage.matrix_zeros(2, 50).is_err());
    drop(storage);
    cleaned(&memory, &query, owner, header);

    let cancel = CancellationToken::default();
    let (memory, query, owner) = joined(&cancel, 1 << 30);
    let header = used(&owner);
    cancel.cancel();
    assert!(OutputTiledLeasedAwStorage::new(&owner, None).is_err());
    cleaned(&memory, &query, owner, header);
    let unjoined = SoaStorageBudget::new(0, None).unwrap();
    assert!(OutputTiledLeasedAwStorage::new(&unjoined, None).is_err());
}
