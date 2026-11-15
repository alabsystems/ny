// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::super::super::super::super::{BoundedBatchSlot, BoundedLinearBounds};
use super::*;
use crate::amortise::NoPoll;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve};
use crate::layers::linear::bias::f32_to_f64_exact;
use ndarray::Array1;
use num_rational::BigRational;
use std::sync::atomic::Ordering;
use std::time::Duration;

const RETAINED: usize = 1 << 20;
fn joined(
    cancel: &CancellationToken,
    work: u64,
) -> (MemoryAuthority, WorkBudget, Arc<SoaStorageBudget>) {
    let memory = MemoryAuthority::new(1 << 26);
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
            scratch_bytes: 1 << 25,
            persistent_bytes: 1 << 25,
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
fn fixture(
    rows: usize,
    k: usize,
    p: usize,
    positions: usize,
    bias: bool,
    errors: u8,
) -> (LinearLayer, LinearBounds) {
    let layer = LinearLayer::new(
        Array2::from_shape_fn((k, p), |(i, j)| {
            ((i * 3 + j * 5) % 11) as f32 * 0.125 - 0.625
        }),
        bias.then(|| Array1::from_shape_fn(k, |i| (i % 7) as f32 * 0.25 - 0.75)),
    )
    .unwrap();
    let a = Array2::from_shape_fn((rows, k * positions), |(i, j)| {
        ((i * 5 + j * 3) % 13) as f32 * 0.0625 - 0.375
    });
    let mut bounds = LinearBounds::new(
        a.clone(),
        Array1::from_shape_fn(rows, |i| i as f32 * 0.125),
        -a,
        Array1::from_shape_fn(rows, |i| i as f32 * 0.25),
    )
    .unwrap();
    bounds.lower_a_err =
        (errors & 1 != 0).then(|| Array2::from_elem((rows, k * positions), 0.03125));
    bounds.upper_a_err =
        (errors & 2 != 0).then(|| Array2::from_elem((rows, k * positions), 0.0625));
    (layer, bounds)
}
// cfg(test)-only general relation. The plain-spec broadcast seed API remains
// unchanged. Every source allocation keeps its real original capacity claim,
// and absent incoming error planes remain absent throughout these controls.
struct FixtureRelation {
    bounds: LinearBounds,
    claims: [Option<Charge>; 6],
    capacities: [usize; 6],
    deadline: Option<Instant>,
}
impl super::super::super::super::super::relation_sealed::Sealed for FixtureRelation {}
impl BoundedRelation for FixtureRelation {
    fn bounds(&self) -> &LinearBounds {
        &self.bounds
    }
    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    fn owner(&self) -> &Arc<SoaStorageBudget> {
        &self.claims[0].as_ref().unwrap().owner
    }
}
fn split_source(part: BoundedArray<f32>) -> (Array2<f32>, Charge, usize) {
    let shape = part.storage.dim();
    let pointer = part.storage.as_ptr();
    let (values, offset) = part.storage.into_raw_vec_and_offset();
    assert!(offset.is_none() || offset == Some(0));
    let capacity = values.capacity() * size_of::<f32>();
    assert_eq!(part._charge.bytes, capacity);
    let storage = Array2::from_shape_vec(shape, values).unwrap();
    assert_eq!(storage.as_ptr(), pointer);
    (storage, part._charge, capacity)
}
fn source(owner: &Arc<SoaStorageBudget>, input: &LinearBounds) -> FixtureRelation {
    let la = BoundedArray::try_from_fn_allow_empty(
        owner,
        input.lower_a().nrows(),
        input.lower_a().ncols(),
        None,
        |i, j| input.lower_a()[[i, j]],
    )
    .unwrap();
    let lb =
        BoundedArray::try_from_fn_allow_empty(owner, input.lower_b().len(), 1, None, |i, _| {
            input.lower_b()[i]
        })
        .unwrap();
    let ua = BoundedArray::try_from_fn_allow_empty(
        owner,
        input.upper_a().nrows(),
        input.upper_a().ncols(),
        None,
        |i, j| input.upper_a()[[i, j]],
    )
    .unwrap();
    let ub =
        BoundedArray::try_from_fn_allow_empty(owner, input.upper_b().len(), 1, None, |i, _| {
            input.upper_b()[i]
        })
        .unwrap();
    let le = input.lower_a_err().map(|e| {
        BoundedArray::try_from_fn_allow_empty(owner, e.nrows(), e.ncols(), None, |i, j| e[[i, j]])
            .unwrap()
    });
    let ue = input.upper_a_err().map(|e| {
        BoundedArray::try_from_fn_allow_empty(owner, e.nrows(), e.ncols(), None, |i, j| e[[i, j]])
            .unwrap()
    });
    let (la, cla, capla) = split_source(la);
    let (lb, clb, caplb) = split_source(lb);
    let (ua, cua, capua) = split_source(ua);
    let (ub, cub, capub) = split_source(ub);
    let (le, cle, caple) = match le {
        Some(a) => {
            let (a, c, n) = split_source(a);
            (Some(a), Some(c), n)
        }
        None => (None, None, 0),
    };
    let (ue, cue, capue) = match ue {
        Some(a) => {
            let (a, c, n) = split_source(a);
            (Some(a), Some(c), n)
        }
        None => (None, None, 0),
    };
    let claims = [Some(cla), Some(clb), Some(cua), Some(cub), cle, cue];
    let bounds = LinearBounds::from_prevalidated_parts_with_optional_err(
        la,
        lb.index_axis_move(Axis(1), 0),
        ua,
        ub.index_axis_move(Axis(1), 0),
        le,
        ue,
    )
    .unwrap();
    FixtureRelation {
        bounds,
        claims,
        capacities: [capla, caplb, capua, capub, caple, capue],
        deadline: None,
    }
}
fn bits(b: &LinearBounds) -> Vec<u32> {
    let mut v = Vec::new();
    for a in [b.lower_a(), b.upper_a()] {
        v.extend(a.iter().map(|x| x.to_bits()));
    }
    for a in [b.lower_b(), b.upper_b()] {
        v.extend(a.iter().map(|x| x.to_bits()));
    }
    for a in [b.lower_a_err(), b.upper_a_err()] {
        v.push(u32::from(a.is_some()));
        if let Some(a) = a {
            v.extend(a.iter().map(|x| x.to_bits()));
        }
    }
    v
}
fn owned(layer: &LinearLayer, bounds: &LinearBounds) -> LinearBounds {
    crown_single::propagate_linear_pollable_cpu(layer, bounds, None, None)
        .unwrap()
        .into_owned()
}

fn q(v: f32) -> BigRational {
    BigRational::from_float(f32_to_f64_exact(v)).unwrap()
}

#[test]
fn bounded_cpu_pure_dispatch_covers_native_singleton_and_wide_full_contraction_without_new_allowance(
) {
    assert_eq!(product_route(2, 1, 1).unwrap(), ProductRoute::Scalar);
    assert_eq!(product_route(2, 257, 2).unwrap(), ProductRoute::Scalar);
    assert_eq!(product_route(2, 1025, 3).unwrap(), ProductRoute::Scalar);
    assert_eq!(product_route(0, 3, 5).unwrap(), ProductRoute::Scalar);
    assert!(product_route(usize::MAX, usize::MAX, 3).is_err());
    if cfg!(target_arch = "aarch64") {
        assert!(matches!(
            product_route(2, 50, 50).unwrap(),
            ProductRoute::Native { .. }
        ));
        assert!(matches!(
            product_route(6, 50, 5).unwrap(),
            ProductRoute::Native { .. }
        ));
    }
}

#[test]
fn bounded_cpu_real_native_and_scalar_sequence_keep_legacy_dyadic_bits_and_original_output_owner() {
    for (rows, k, p, positions) in [
        (1, 50, 50, 2),
        (3, 50, 5, 3),
        (2, 1, 1, 2),
        (2, 257, 2, 2),
        (2, 1025, 3, 1),
    ] {
        for bias in [false, true] {
            for errors in [0, 1, 2, 3] {
                let (layer, input) = fixture(rows, k, p, positions, bias, errors);
                let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 29);
                let header = used(&owner);
                assert!(layer.retained_bytes().unwrap() + input.memory_bytes() <= RETAINED);
                let src = source(&owner, &input);
                let source_bytes = used(&owner);
                let saved = bits(src.bounds());
                let layer_bytes = layer.retained_bytes().unwrap();
                let output = propagate(&src, &layer, None).unwrap();
                assert!(Arc::ptr_eq(output.owner(), &owner));
                assert_eq!(
                    layer.retained_bytes().unwrap(),
                    layer_bytes,
                    "no lazy abs/widened cache may be populated"
                );
                assert_eq!(bits(src.bounds()), saved);
                assert_eq!(
                    bits(output.bounds()),
                    bits(&owned(&layer, src.bounds())),
                    "{rows}x{k}x{p} positions{positions} bias{bias} err{errors}"
                );
                assert!(used(&owner) > source_bytes);
                assert_eq!(query.receipt().unwrap().live_scratch_bytes, used(&owner));
                drop(output);
                assert_eq!(used(&owner), source_bytes);
                drop(src);
                clean(&memory, &query, owner, header);
            }
        }
    }
}

#[test]
fn bounded_cpu_all_position_incoming_planes_and_biases_enclose_exact_rational_uncertainty() {
    for (rows, k, p) in [(1, 50, 50), (2, 257, 2), (2, 1, 3)] {
        for errors in [1, 2, 3] {
            let positions = 3;
            let (layer, input) = fixture(rows, k, p, positions, true, errors);
            let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 29);
            let header = used(&owner);
            let src = source(&owner, &input);
            let output = propagate(&src, &layer, None).unwrap();
            let out = output.bounds();
            for (a, e, oa, oe, old_b, new_b, lower) in [
                (
                    input.lower_a(),
                    input.lower_a_err(),
                    out.lower_a(),
                    out.lower_a_err().unwrap(),
                    input.lower_b(),
                    out.lower_b(),
                    true,
                ),
                (
                    input.upper_a(),
                    input.upper_a_err(),
                    out.upper_a(),
                    out.upper_a_err().unwrap(),
                    input.upper_b(),
                    out.upper_b(),
                    false,
                ),
            ] {
                for i in 0..rows {
                    let mut exact_bias = q(old_b[i]);
                    let mut bias_error = q(0.0);
                    for pos in 0..positions {
                        for kk in 0..k {
                            let b = layer.bias().unwrap()[kk];
                            exact_bias += q(a[[i, pos * k + kk]]) * q(b);
                            if let Some(e) = e {
                                bias_error += q(e[[i, pos * k + kk]]) * q(b.abs());
                            }
                        }
                        for j in 0..p {
                            let mut exact = q(0.0);
                            let mut incoming = q(0.0);
                            for kk in 0..k {
                                let w = original_weight(&layer)[(kk, j)];
                                exact += q(a[[i, pos * k + kk]]) * q(w);
                                if let Some(e) = e {
                                    incoming += q(e[[i, pos * k + kk]]) * q(w.abs());
                                }
                            }
                            let stored = q(oa[[i, pos * p + j]]);
                            let radius = q(oe[[i, pos * p + j]]);
                            assert!(
                                stored.clone() - radius.clone() <= exact.clone() - incoming.clone()
                            );
                            assert!(exact + incoming <= stored + radius);
                        }
                    }
                    if lower {
                        assert!(q(new_b[i]) <= exact_bias - bias_error);
                    } else {
                        assert!(exact_bias + bias_error <= q(new_b[i]));
                    }
                }
            }
            drop(output);
            drop(src);
            clean(&memory, &query, owner, header);
        }
    }
}

#[test]
fn bounded_cpu_every_observed_refusal_including_post_publication_releases_private_capacity() {
    let (layer, input) = fixture(2, 3, 2, 2, true, 3);
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 27);
    let header = used(&owner);
    let src = source(&owner, &input);
    let initial = used(&owner);
    let saved = bits(src.bounds());
    let mut final_boundary = 0;
    let output = propagate_observed(&src, &layer, None, |n| {
        final_boundary = n;
        Ok(())
    })
    .unwrap();
    drop(output);
    assert!(final_boundary > 32);
    assert_eq!(used(&owner), initial);
    for stop in 1..=final_boundary {
        let mut calls = 0;
        let result = propagate_observed(&src, &layer, None, |n| {
            calls = n;
            if n == stop {
                Err(NyError::DeadlineExceeded("bounded CPU observer".into()))
            } else {
                Ok(())
            }
        });
        assert!(matches!(result,Err(e)if e.is_deadline_exceeded()));
        assert_eq!(calls, stop);
        assert_eq!(used(&owner), initial);
        assert_eq!(query.receipt().unwrap().live_scratch_bytes, initial);
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        assert_eq!(bits(src.bounds()), saved);
    }
    drop(src);
    clean(&memory, &query, owner, header);
}

#[test]
fn bounded_cpu_original_query_work_never_renews_or_falls_back_after_native_refusal() {
    let (layer, input) = fixture(1, 50, 50, 1, true, 3);
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 29);
    let header = used(&owner);
    let src = source(&owner, &input);
    let output = propagate(&src, &layer, None).unwrap();
    drop(output);
    let total = query.receipt().unwrap().work_reserved;
    drop(src);
    clean(&memory, &query, owner, header);
    for limit in [total - 1, total] {
        let (memory, query, owner) = joined(&CancellationToken::default(), limit);
        let header = used(&owner);
        let src = source(&owner, &input);
        let initial = used(&owner);
        let original = bits(src.bounds());
        let result = propagate(&src, &layer, None);
        assert_eq!(result.is_ok(), limit == total);
        if limit != total {
            assert!(matches!(result, Err(NyError::UnsupportedOp(_))));
        } else {
            drop(result.unwrap());
        }
        assert!(query.receipt().unwrap().work_reserved <= limit);
        assert_eq!(used(&owner), initial);
        assert_eq!(bits(src.bounds()), original);
        drop(src);
        clean(&memory, &query, owner, header);
    }
}

#[test]
fn bounded_cpu_original_cancel_and_narrowest_absolute_deadline_remain_authoritative() {
    let (layer, input) = fixture(2, 3, 2, 2, true, 3);
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 27);
    let header = used(&owner);
    let mut src = source(&owner, &input);
    let deadline = Instant::now() + Duration::from_secs(60);
    src.deadline = Some(deadline);
    let mut final_boundary = 0;
    let output = propagate_observed(
        &src,
        &layer,
        Some(deadline + Duration::from_secs(60)),
        |n| {
            final_boundary = n;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(output.deadline(), Some(deadline));
    drop(output);
    let initial = used(&owner);
    assert!(
        matches!(propagate(&src,&layer,Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())),Err(e)if e.is_deadline_exceeded())
    );
    assert_eq!(used(&owner), initial);
    drop(src);
    clean(&memory, &query, owner, header);
    for stop in [1, final_boundary] {
        let cancel = CancellationToken::default();
        let (memory, query, owner) = joined(&cancel, 1 << 27);
        let header = used(&owner);
        let src = source(&owner, &input);
        let initial = used(&owner);
        let saved = bits(src.bounds());
        let result = propagate_observed(&src, &layer, None, |n| {
            if n == stop {
                cancel.cancel();
            }
            Ok(())
        });
        assert!(matches!(result,Err(e)if e.is_deadline_exceeded()));
        assert_eq!(used(&owner), initial);
        assert_eq!(bits(src.bounds()), saved);
        drop(src);
        clean(&memory, &query, owner, header);
    }
}

#[test]
fn bounded_cpu_unadmitted_engine_stack_foreign_source_and_unjoined_account_precede_payloads() {
    let (layer, input) = fixture(2, 3, 2, 1, true, 3);
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 27);
    let header = used(&owner);
    let src = source(&owner, &input);
    let initial = used(&owner);
    let layout = CpuLayout {
        num_outputs: 2,
        out_features: 3,
        in_features: 2,
        num_positions: 1,
        total_in_features: 2,
    };
    let clone = src.bounds().clone();
    for (allow, stack, bounds) in [
        (true, false, src.bounds()),
        (false, true, src.bounds()),
        (false, false, &clone),
    ] {
        let mut storage = QueryCpuStorage {
            source: &src,
            layer: &layer,
            owner: &owner,
            deadline: None,
            poll: Poll::for_owner(&owner, None),
            route: None,
        };
        assert!(storage
            .admit(&layer, bounds, &layout, None, allow, stack, &mut NoPoll)
            .is_err());
        assert_eq!(used(&owner), initial);
    }
    drop(src);
    clean(&memory, &query, owner, header);
    let legacy = SoaStorageBudget::new(0, None).unwrap();
    let mut poll = Poll::for_owner(&legacy, None);
    let mut charge = legacy.charge(input.memory_bytes(), &mut poll).unwrap();
    charge.materialize().unwrap();
    drop(poll);
    let src = BoundedLinearBounds {
        bounds: input,
        _charge: charge,
        deadline: None,
    };
    let before = used(&legacy);
    assert!(propagate(&src, &layer, None).is_err());
    assert_eq!(used(&legacy), before);
    drop(src);
    assert_eq!(used(&legacy), 0);
}

#[test]
fn bounded_cpu_claimed_output_can_feed_same_original_query_without_materializing_a_bounds_copy() {
    let (layer, input) = fixture(2, 3, 2, 2, true, 3);
    let next = LinearLayer::new(
        Array2::from_shape_fn((2, 1), |(i, _)| i as f32 * 0.25 - 0.125),
        Some(Array1::from_vec(vec![0.5, -0.25])),
    )
    .unwrap();
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 27);
    let header = used(&owner);
    let src = source(&owner, &input);
    let first = propagate(&src, &layer, None).unwrap();
    let initial = used(&owner);
    let saved = bits(first.bounds());
    let second = propagate(&first, &next, None).unwrap();
    assert!(Arc::ptr_eq(second.owner(), first.owner()));
    assert_eq!(bits(first.bounds()), saved);
    assert_eq!(bits(second.bounds()), bits(&owned(&next, first.bounds())));
    drop(second);
    assert_eq!(used(&owner), initial);
    drop(first);
    drop(src);
    clean(&memory, &query, owner, header);
}

#[test]
fn bounded_cpu_row_flag_actual_capacity_and_every_fill_commit_fault_use_original_claim() {
    let (layer, input) = fixture(2, 3, 2, 1, true, 3);
    let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 27);
    let header = used(&owner);
    let src = source(&owner, &input);
    let initial = used(&owner);
    {
        let mut storage = QueryCpuStorage {
            source: &src,
            layer: &layer,
            owner: &owner,
            deadline: None,
            poll: Poll::for_owner(&owner, None),
            route: None,
        };
        for rows in [0, 1, 4097] {
            let flags = storage.row_flags_zeros(rows).unwrap();
            assert_eq!(flags.values.len(), rows);
            assert!(flags.values.iter().all(|&v| !v));
            assert_eq!(
                flags._charge.bytes,
                flags.values.capacity() * size_of::<bool>()
            );
            assert_eq!(used(&owner), initial + flags._charge.bytes);
            assert_eq!(query.receipt().unwrap().live_scratch_bytes, used(&owner));
            drop(flags);
            assert_eq!(used(&owner), initial);
        }
        for stage in [
            "bounded SoA storage entry",
            "bounded SoA storage after reserve",
            FLAGS_FILL,
            FLAGS_COMMIT,
        ] {
            storage.poll.forced_stage = Some(stage);
            assert!(matches!(storage.row_flags_zeros(4097),Err(e)if e.is_deadline_exceeded()));
            assert_eq!(used(&owner), initial);
            assert_eq!(query.receipt().unwrap().live_scratch_bytes, initial);
            assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        }
    }
    drop(src);
    clean(&memory, &query, owner, header);
}

#[test]
fn bounded_cpu_general_relation_fixture_keeps_optional_planes_capacities_and_plain_seed_refusal() {
    for mask in [0, 1, 2, 3] {
        let (_layer, input) = fixture(2, 3, 2, 2, true, mask);
        let (memory, query, owner) = joined(&CancellationToken::default(), 1 << 27);
        let header = used(&owner);
        let seed = BoundedBatchSlot::try_seeded(&owner, 1, &input, None);
        if mask == 0 {
            let seed = seed.unwrap();
            let materialized = seed.try_materialize(0, None).unwrap().unwrap();
            assert_eq!(bits(materialized.bounds()), bits(&input));
            drop(materialized);
            drop(seed);
        } else {
            assert!(matches!(seed, Err(NyError::InvalidSpec(ref reason))
                if reason == "bounded SoA storage: seed must retain the original plain-spec contract"));
        }
        assert_eq!(used(&owner), header);
        let saved = bits(&input);
        let src = source(&owner, &input);
        assert_eq!(bits(src.bounds()), saved);
        assert_eq!(src.bounds().lower_a_err().is_some(), mask & 1 != 0);
        assert_eq!(src.bounds().upper_a_err().is_some(), mask & 2 != 0);
        let capacities: usize = src.capacities.iter().sum();
        assert_eq!(used(&owner), header + capacities);
        assert_eq!(
            query.receipt().unwrap().live_scratch_bytes,
            header + capacities
        );
        for (claim, &capacity) in src.claims.iter().zip(&src.capacities) {
            if let Some(claim) = claim {
                assert!(Arc::ptr_eq(&claim.owner, &owner));
                assert_eq!(claim.bytes, capacity);
            } else {
                assert_eq!(capacity, 0);
            }
        }
        drop(src);
        clean(&memory, &query, owner, header);
    }
}
