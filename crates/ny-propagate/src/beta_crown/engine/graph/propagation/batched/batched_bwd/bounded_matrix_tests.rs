// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use faer::Mat;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn budget(deadline: Option<Instant>) -> Arc<SoaStorageBudget> {
    Arc::new(SoaStorageBudget {
        retained_bytes: 0,
        ceiling_bytes: 1 << 20,
        reserved_bytes: AtomicUsize::new(0),
        deadline,
        query: None,
    })
}
fn used(owner: &SoaStorageBudget) -> usize {
    owner.reserved_bytes.load(Ordering::Acquire)
}
fn expired<T>(result: Result<T>) {
    match result {
        Err(error) => assert!(error.is_deadline_exceeded(), "{error}"),
        Ok(_) => panic!("expected typed expiry"),
    }
}
fn memory<T>(result: Result<T>) {
    match result {
        Err(error) => assert!(error.is_cpu_memory_exceeded(), "{error}"),
        Ok(_) => panic!("expected live-budget refusal"),
    }
}
fn f32_value(i: usize) -> f32 {
    f32::from_bits(
        [
            0x3f800000, 0x80000000, 1, 0x7fc01234, 0x7f800000, 0xff800000, 0x3e800000,
        ][i % 7],
    )
}
fn f64_value(i: usize) -> f64 {
    f64::from_bits(
        [
            0x3ff0000000000000,
            0x8000000000000000,
            1,
            0x7ff8000000001234,
            0x7ff0000000000000,
            0xfff0000000000000,
            0x3fd0000000000000,
        ][i % 7],
    )
}

#[test]
fn bounded_array_storage_keeps_f32_f64_bits_and_live_payload_leases() {
    let owner = budget(None);
    let mut a =
        BoundedArray::<f32>::try_from_fn(&owner, 3, 7, None, |r, c| f32_value(r * 7 + c)).unwrap();
    let b =
        BoundedArray::<f64>::try_from_fn(&owner, 5, 3, None, |r, c| f64_value(r * 3 + c)).unwrap();
    let expected_a = Array2::from_shape_fn((3, 7), |(r, c)| f32_value(r * 7 + c));
    let expected_b = Array2::from_shape_fn((5, 3), |(r, c)| f64_value(r * 3 + c));
    assert_eq!(a.array().mapv(f32::to_bits), expected_a.mapv(f32::to_bits));
    assert_eq!(b.array().mapv(f64::to_bits), expected_b.mapv(f64::to_bits));
    assert_eq!(used(&owner), a._charge.bytes + b._charge.bytes);
    let original_used = used(&owner);
    a.view_mut()[[0, 0]] = f32::from_bits(2);
    assert_eq!(a.view()[[0, 0]].to_bits(), 2);
    assert_eq!(used(&owner), original_used);
    drop(a);
    assert_eq!(used(&owner), b._charge.bytes);
    drop(b);
    assert_eq!(used(&owner), 0);
}

#[test]
fn bounded_faer_storage_matches_owned_layout_padding_and_scalar_bits() {
    let owner = budget(None);
    for request in [None, Some(Instant::now() + Duration::from_secs(30))] {
        for rows in [1, 7, 16, 65] {
            for cols in [1, 5, 50, 65, 120] {
                let mut a = BoundedMat::<f32>::try_from_fn(&owner, rows, cols, request, |r, c| {
                    f32_value(r + 3 * c)
                })
                .unwrap();
                let b = BoundedMat::<f64>::try_from_fn(&owner, rows, cols, request, |r, c| {
                    f64_value(r + 3 * c)
                })
                .unwrap();
                let reference_a = Mat::<f32>::from_fn(rows, cols, |r, c| f32_value(r + 3 * c));
                let reference_b = Mat::<f64>::from_fn(rows, cols, |r, c| f64_value(r + 3 * c));
                assert_eq!(a.matrix().shape(), reference_a.shape());
                assert_eq!(b.matrix().shape(), reference_b.shape());
                assert_eq!(a.matrix().col_stride(), reference_a.col_stride());
                assert_eq!(b.matrix().col_stride(), reference_b.col_stride());
                assert_eq!(
                    a._charge.bytes,
                    a.storage.capacity() * size_of::<F32Block>()
                );
                assert!(
                    a._charge.bytes >= a.matrix().col_stride() as usize * cols * size_of::<f32>()
                );
                assert_eq!(a.matrix().as_ptr() as usize % 64, 0);
                assert_eq!(
                    b._charge.bytes,
                    b.storage.capacity() * size_of::<F64Block>()
                );
                assert!(
                    b._charge.bytes >= b.matrix().col_stride() as usize * cols * size_of::<f64>()
                );
                assert_eq!(b.matrix().as_ptr() as usize % 64, 0);
                for c in 0..cols {
                    for r in 0..rows {
                        assert_eq!(a.matrix()[(r, c)].to_bits(), reference_a[(r, c)].to_bits());
                        assert_eq!(b.matrix()[(r, c)].to_bits(), reference_b[(r, c)].to_bits());
                    }
                }
                assert_eq!(used(&owner), a._charge.bytes + b._charge.bytes);
                let before = used(&owner);
                {
                    let mut view = a.view_mut();
                    view[(0, 0)] = f32::from_bits(2);
                }
                assert_eq!(a.view()[(0, 0)].to_bits(), 2);
                assert_eq!(used(&owner), before);
                drop(a);
                assert_eq!(used(&owner), b._charge.bytes);
                drop(b);
                assert_eq!(used(&owner), 0);
            }
        }
    }
}

#[test]
fn bounded_matrix_storage_never_regrants_original_deadline_or_runs_expired_getter() {
    let authority = Instant::now() + Duration::from_secs(30);
    let owner = budget(Some(authority));
    let earlier = authority.checked_sub(Duration::from_secs(10)).unwrap();
    for (request, expected) in [
        (None, authority),
        (Some(authority + Duration::from_secs(10)), authority),
        (Some(earlier), earlier),
    ] {
        let a = BoundedArray::<f32>::try_from_fn(&owner, 2, 3, request, |_, _| 1.0).unwrap();
        let b = BoundedMat::<f64>::try_from_fn(&owner, 2, 3, request, |_, _| 1.0).unwrap();
        assert_eq!(a.deadline(), Some(expected));
        assert_eq!(b.deadline(), Some(expected));
    }
    assert_eq!(used(&owner), 0);
    let calls = Cell::new(0);
    let past = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    expired(BoundedArray::<f32>::try_from_fn(
        &owner,
        2,
        3,
        past,
        |_, _| {
            calls.set(calls.get() + 1);
            1.0
        },
    ));
    expired(BoundedMat::<f64>::try_from_fn(
        &owner,
        2,
        3,
        past,
        |_, _| {
            calls.set(calls.get() + 1);
            1.0
        },
    ));
    let expired_owner = budget(past);
    expired(BoundedArray::<f64>::try_from_fn(
        &expired_owner,
        2,
        3,
        None,
        |_, _| {
            calls.set(calls.get() + 1);
            1.0
        },
    ));
    expired(BoundedMat::<f32>::try_from_fn(
        &expired_owner,
        2,
        3,
        None,
        |_, _| {
            calls.set(calls.get() + 1);
            1.0
        },
    ));
    assert_eq!(calls.get(), 0);
    assert_eq!(used(&owner), 0);
    assert_eq!(used(&expired_owner), 0);
}

#[test]
fn bounded_matrix_storage_discards_partial_fill_and_commit_faults() {
    let owner = budget(None);
    for is_f64 in [false, true] {
        for faer in [false, true] {
            let calls = Cell::new(0);
            let mut poll = BufferPoll::new(None);
            poll.fail_after_units = Some(4096);
            if is_f64 {
                let getter = |r, c| {
                    calls.set(calls.get() + 1);
                    f64_value(r + c)
                };
                if faer {
                    expired(BoundedMat::<f64>::with_poll(
                        &owner, 5000, 3, None, getter, &mut poll,
                    ));
                } else {
                    expired(BoundedArray::<f64>::with_poll(
                        &owner, 5000, 3, None, getter, &mut poll,
                    ));
                }
            } else {
                let getter = |r, c| {
                    calls.set(calls.get() + 1);
                    f32_value(r + c)
                };
                if faer {
                    expired(BoundedMat::<f32>::with_poll(
                        &owner, 5000, 3, None, getter, &mut poll,
                    ));
                } else {
                    expired(BoundedArray::<f32>::with_poll(
                        &owner, 5000, 3, None, getter, &mut poll,
                    ));
                }
            }
            assert_eq!(poll.observed_units, 4096);
            assert_eq!(calls.get(), if is_f64 { 2048 } else { 4096 });
            assert_eq!(used(&owner), 0);
        }
    }
    for stage in [AFTER_RESERVE, COMMIT] {
        for faer in [false, true] {
            let calls = Cell::new(0);
            let mut poll = BufferPoll::new(None);
            poll.shared.forced_stage = Some(stage);
            let getter = |r, c| {
                calls.set(calls.get() + 1);
                f64_value(r + c)
            };
            if faer {
                expired(BoundedMat::<f64>::with_poll(
                    &owner, 3, 5, None, getter, &mut poll,
                ));
            } else {
                expired(BoundedArray::<f64>::with_poll(
                    &owner, 3, 5, None, getter, &mut poll,
                ));
            }
            assert_eq!(calls.get(), if stage == AFTER_RESERVE { 0 } else { 15 });
            assert_eq!(used(&owner), 0);
        }
    }
}

#[test]
fn bounded_matrix_storage_checked_shapes_and_shared_quota_refuse_before_fill() {
    let owner = budget(None);
    let calls = Cell::new(0);
    for (rows, cols) in [(0, 3), (2, 0), (usize::MAX, 2), (usize::MAX, 1)] {
        assert!(
            BoundedMat::<f32>::try_from_fn(&owner, rows, cols, None, |_, _| {
                calls.set(calls.get() + 1);
                1.0
            })
            .is_err()
        );
        assert!(
            BoundedArray::<f64>::try_from_fn(&owner, rows, cols, None, |_, _| {
                calls.set(calls.get() + 1);
                1.0
            })
            .is_err()
        );
        assert_eq!(calls.get(), 0);
        assert_eq!(used(&owner), 0);
    }
    let a = BoundedArray::<f64>::try_from_fn(&owner, 5, 11, None, |_, _| 1.0).unwrap();
    let b = BoundedMat::<f32>::try_from_fn(&owner, 65, 7, None, |_, _| 2.0).unwrap();
    let held = owner
        .charge(owner.ceiling_bytes - used(&owner), &mut Poll::new(None))
        .unwrap();
    memory(BoundedArray::<f64>::try_from_fn(
        &owner,
        2,
        3,
        None,
        |_, _| {
            calls.set(calls.get() + 1);
            1.0
        },
    ));
    memory(BoundedMat::<f32>::try_from_fn(
        &owner,
        2,
        3,
        None,
        |_, _| {
            calls.set(calls.get() + 1);
            1.0
        },
    ));
    assert_eq!(calls.get(), 0);
    assert_eq!(used(&owner), a._charge.bytes + b._charge.bytes + held.bytes);
    drop(held);
    drop(a);
    drop(b);
    assert_eq!(used(&owner), 0);
}

#[test]
fn bounded_matrix_storage_keeps_strided_source_and_copy_lifetimes_separate() {
    let owner = budget(None);
    let source =
        BoundedArray::<f32>::try_from_fn(&owner, 3, 65, None, |r, c| f32_value(r + 3 * c)).unwrap();
    let source_bytes = source._charge.bytes;
    let expected = source.view().reversed_axes();
    assert!(!expected.is_standard_layout());
    let copy =
        BoundedMat::<f32>::try_from_fn(&owner, 65, 3, None, |r, c| expected[[r, c]]).unwrap();
    let traversal = Cell::new(0);
    let last = Cell::new((0, 0));
    let back = BoundedArray::<f32>::try_from_fn(&owner, 65, 3, None, |r, c| {
        assert_eq!((r, c), (traversal.get() / 3, traversal.get() % 3));
        traversal.set(traversal.get() + 1);
        last.set((r, c));
        copy.matrix()[(r, c)]
    })
    .unwrap();
    assert_eq!(traversal.get(), 195);
    assert_eq!(last.get(), (64, 2));
    for r in 0..65 {
        for c in 0..3 {
            assert_eq!(copy.matrix()[(r, c)].to_bits(), expected[[r, c]].to_bits());
            assert_eq!(back.array()[[r, c]].to_bits(), expected[[r, c]].to_bits());
        }
    }
    assert_eq!(
        used(&owner),
        source_bytes + copy._charge.bytes + back._charge.bytes
    );
    let remaining = copy._charge.bytes + back._charge.bytes;
    drop(source);
    assert_eq!(used(&owner), remaining);
    drop(copy);
    assert_eq!(used(&owner), back._charge.bytes);
    drop(back);
    assert_eq!(used(&owner), 0);
    let order = Cell::new(0);
    let matrix = BoundedMat::<f64>::try_from_fn(&owner, 5, 7, None, |r, c| {
        assert_eq!((r, c), (order.get() % 5, order.get() / 5));
        order.set(order.get() + 1);
        f64_value(r + c)
    })
    .unwrap();
    assert_eq!(order.get(), 35);
    drop(matrix);
    assert_eq!(used(&owner), 0);
}
