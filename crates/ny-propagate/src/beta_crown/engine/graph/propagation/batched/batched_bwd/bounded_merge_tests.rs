// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::super::super::indexed_pending::IndexedPendingLinearBounds;
use super::super::{SoaStorageBudget, PENDING_COMMIT};
use super::*;
use crate::layers::LinearLayer;
use crate::network::CrownDispatchPlan;
use crate::{GraphNode, Layer};
use ndarray::{Array1, Array2};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

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

fn used(owner: &SoaStorageBudget) -> usize {
    owner.reserved_bytes.load(Ordering::Acquire)
}

fn fixture(
    rows: usize,
    cols: usize,
    mask: usize,
    stress: bool,
    strided: bool,
    salt: usize,
) -> LinearBounds {
    let values = if stress {
        vec![
            0x7fc01234, 0x7f800000, 0xff800000, 0x80000000, 1, 0x71800000, 0x3f800000,
        ]
    } else {
        vec![
            0x3f800000, 0x80000000, 1, 0x40000000, 0x3e800000, 0x40400000, 0xbf000000,
        ]
    };
    let matrix = |offset| {
        let value = |i, j| f32::from_bits(values[(i * cols + j + salt + offset) % values.len()]);
        if strided {
            let out = Array2::from_shape_fn((cols, rows), |(j, i)| value(i, j)).reversed_axes();
            if rows > 1 && cols > 1 {
                assert!(out.as_slice().is_none());
            }
            out
        } else {
            Array2::from_shape_fn((rows, cols), |(i, j)| value(i, j))
        }
    };
    let error = |offset| {
        let mut out = matrix(offset);
        if !stress {
            out.mapv_inplace(f32::abs);
        }
        out
    };
    // Test fixtures deliberately retain one-sided Options and nonfinite bits;
    // they are existing reference-storage states, not fresh seed constructors.
    LinearBounds {
        lower_a: matrix(0),
        upper_a: matrix(2),
        lower_b: Array1::from_shape_fn(rows, |i| {
            f32::from_bits(values[(i + salt + 3) % values.len()])
        }),
        upper_b: Array1::from_shape_fn(rows, |i| {
            f32::from_bits(values[(i + salt + 5) % values.len()])
        }),
        lower_a_err: (mask & 1 != 0).then(|| error(1)),
        upper_a_err: (mask & 2 != 0).then(|| error(4)),
    }
}

fn leased(
    owner: &Arc<SoaStorageBudget>,
    bounds: LinearBounds,
    deadline: Option<Instant>,
) -> BoundedLinearBounds {
    // Fixture construction is outside the operation under test. Its complete
    // payload is leased before the private operation can borrow any plane.
    let charge = owner
        .charge(bounds.memory_bytes(), &mut Poll::new(deadline))
        .unwrap();
    BoundedLinearBounds {
        bounds,
        _charge: charge,
        deadline,
    }
}

fn same(actual: &LinearBounds, expected: &LinearBounds) {
    let matrix = |a: &Array2<f32>, b: &Array2<f32>| {
        assert_eq!(a.dim(), b.dim());
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    };
    matrix(actual.lower_a(), expected.lower_a());
    matrix(actual.upper_a(), expected.upper_a());
    for (a, b) in [
        (actual.lower_b(), expected.lower_b()),
        (actual.upper_b(), expected.upper_b()),
    ] {
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    for (a, b) in [
        (actual.lower_a_err(), expected.lower_a_err()),
        (actual.upper_a_err(), expected.upper_a_err()),
    ] {
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            matrix(a, b);
        }
    }
}

fn expired<T>(result: Result<T>) {
    match result {
        Err(error) => assert!(error.is_deadline_exceeded(), "{error}"),
        Ok(_) => panic!("expected expiry"),
    }
}

#[test]
fn bounded_merge_matches_indexed_reference_finite_stress_and_strided_planes() {
    let plan = plan();
    let idx = plan.output_node_idx;
    for cols in [1, 3, 65, 120] {
        for stress in [false, true] {
            for strided in [false, true] {
                for first in 0..4 {
                    for second in 0..4 {
                        let owner = SoaStorageBudget::new(0, None).unwrap();
                        let mut actual =
                            BoundedBatchPending::try_new(&plan, 5, &owner, None).unwrap();
                        let mut reference = IndexedPendingLinearBounds::new(&plan, 5);
                        for (step, mask) in [first, second, 0].into_iter().enumerate() {
                            for domain in [0, 2, 4] {
                                let bounds = fixture(2, cols, mask, stress, strided, step + domain);
                                if !stress {
                                    assert!(bounds.lower_a().iter().all(|v| v.is_finite()));
                                    assert!(bounds.lower_a().iter().any(|v| *v != 0.0));
                                }
                                reference
                                    .accumulate_idx(idx, bounds.clone(), domain)
                                    .unwrap();
                                let contribution = leased(&owner, bounds, None);
                                actual = actual
                                    .accumulate_domain(idx, domain, &contribution)
                                    .unwrap();
                            }
                        }
                        let expected = reference.take_idx(idx).unwrap();
                        let (pending, slot) = actual.take(idx).unwrap();
                        let slot = slot.unwrap();
                        for (domain, expected) in expected.iter().enumerate() {
                            let observed = slot.try_materialize(domain, None).unwrap();
                            assert_eq!(observed.is_some(), expected.is_some());
                            if let (Some(a), Some(b)) = (observed, expected) {
                                if !stress {
                                    assert!(a
                                        .bounds()
                                        .lower_a()
                                        .iter()
                                        .chain(a.bounds().upper_a().iter())
                                        .chain(a.bounds().lower_b().iter())
                                        .chain(a.bounds().upper_b().iter())
                                        .all(|v| v.is_finite()));
                                }
                                same(a.bounds(), b);
                            }
                        }
                        drop(slot);
                        drop(pending);
                        assert_eq!(used(&owner), 0);
                    }
                }
            }
        }
    }
}

#[test]
fn bounded_merge_exact_zero_and_one_sided_error_pairing_match_reference() {
    let plan = plan();
    let idx = plan.output_node_idx;
    let owner = SoaStorageBudget::new(0, None).unwrap();
    let mut pending = BoundedBatchPending::try_new(&plan, 1, &owner, None).unwrap();
    let mut reference = IndexedPendingLinearBounds::new(&plan, 1);
    for (step, mask) in [0, 0, 1, 2].into_iter().enumerate() {
        let mut bounds = fixture(2, 3, mask, false, false, 0);
        bounds.lower_a_mut().fill(0.0);
        bounds.upper_a_mut().fill(-0.0);
        if let Some(e) = bounds.lower_a_err.as_mut() {
            e.fill(0.0);
        }
        if let Some(e) = bounds.upper_a_err.as_mut() {
            e.fill(0.0);
        }
        reference.accumulate_idx(idx, bounds.clone(), 0).unwrap();
        let contribution = leased(&owner, bounds, None);
        pending = pending.accumulate_domain(idx, 0, &contribution).unwrap();
        let observed = pending.slots[idx]
            .as_ref()
            .unwrap()
            .try_materialize(0, None)
            .unwrap()
            .unwrap();
        same(
            observed.bounds(),
            reference.get_name("out").unwrap()[0].as_ref().unwrap(),
        );
        if step < 2 {
            assert!(observed.bounds().lower_a_err().is_none());
            assert!(observed.bounds().upper_a_err().is_none());
        }
        if step == 2 {
            assert!(observed
                .bounds()
                .lower_a_err()
                .unwrap()
                .iter()
                .all(|v| *v > 0.0));
            assert!(observed
                .bounds()
                .upper_a_err()
                .unwrap()
                .iter()
                .all(|v| v.to_bits() == 0));
        }
    }
    drop(pending);
    assert_eq!(used(&owner), 0);
}

#[test]
fn bounded_merge_refuses_layout_indices_or_foreign_account_and_drops_pending() {
    let plan = plan();
    let idx = plan.output_node_idx;
    for mode in 0..6 {
        let owner = SoaStorageBudget::new(0, None).unwrap();
        let other = SoaStorageBudget::new(0, None).unwrap();
        let mut pending = BoundedBatchPending::try_new(&plan, 2, &owner, None).unwrap();
        let initial = leased(&owner, fixture(2, 3, 0, false, false, 0), None);
        pending = pending.accumulate_domain(idx, 0, &initial).unwrap();
        drop(initial);
        let mut bounds = fixture(2, 3, 3, false, false, 1);
        if mode == 2 {
            bounds = fixture(3, 3, 3, false, false, 1);
        }
        if mode == 3 {
            bounds.lower_a_err = Some(Array2::zeros((1, 3)));
        }
        if mode == 4 {
            bounds.upper_b = Array1::zeros(1);
        }
        let source_owner = if mode == 5 { &other } else { &owner };
        let contribution = leased(source_owner, bounds, None);
        let source_bytes = contribution._charge.bytes;
        let result = pending.accumulate_domain(
            if mode == 0 { usize::MAX } else { idx },
            if mode == 1 { 2 } else { 0 },
            &contribution,
        );
        assert!(result.is_err());
        assert_eq!(used(&owner), if mode == 5 { 0 } else { source_bytes });
        drop(contribution);
        assert_eq!(used(&owner), 0);
        assert_eq!(used(&other), 0);
    }
}

#[test]
fn bounded_merge_expiry_discards_partial_transaction_after_real_work() {
    let plan = plan();
    let idx = plan.output_node_idx;
    for stage in [
        ENTRY,
        COEFFICIENT,
        BIAS,
        PRESENCE,
        ERROR,
        DOMAIN_COMMIT,
        PENDING_COMMIT,
    ] {
        let owner = SoaStorageBudget::new(0, None).unwrap();
        let contribution = leased(&owner, fixture(2, 5000, 3, false, false, 0), None);
        let pending = BoundedBatchPending::try_new(&plan, 2, &owner, None)
            .unwrap()
            .accumulate_domain(idx, 0, &contribution)
            .unwrap();
        let mut poll = Poll::new(None);
        poll.forced_stage = Some(stage);
        expired(accumulate(pending, idx, 0, &contribution, &mut poll));
        assert_eq!(used(&owner), contribution._charge.bytes);
        drop(contribution);
        assert_eq!(used(&owner), 0);
    }
    for stage in [COEFFICIENT, BIAS, PRESENCE, ERROR] {
        let owner = SoaStorageBudget::new(0, None).unwrap();
        let mut bounds = fixture(
            5000,
            2,
            if stage == PRESENCE { 0 } else { 3 },
            false,
            false,
            0,
        );
        if stage == PRESENCE {
            bounds.lower_a_mut().fill(0.0);
            bounds.upper_a_mut().fill(0.0);
        }
        let contribution = leased(&owner, bounds, None);
        let pending = BoundedBatchPending::try_new(&plan, 2, &owner, None)
            .unwrap()
            .accumulate_domain(idx, 0, &contribution)
            .unwrap();
        FAIL_AFTER_WORK.with(|fault| fault.set(Some((stage, 4096))));
        OBSERVED_WORK.with(|count| count.set(0));
        let result = accumulate(pending, idx, 0, &contribution, &mut Poll::new(None));
        FAIL_AFTER_WORK.with(|fault| fault.set(None));
        expired(result);
        assert_eq!(OBSERVED_WORK.with(std::cell::Cell::get), 4096);
        assert_eq!(used(&owner), contribution._charge.bytes);
        drop(contribution);
        assert_eq!(used(&owner), 0);
    }
    let deadline = Instant::now() + Duration::from_secs(30);
    let owner = SoaStorageBudget::new(0, Some(deadline)).unwrap();
    let pending = BoundedBatchPending::try_new(&plan, 1, &owner, None).unwrap();
    assert_eq!(pending.deadline, Some(deadline));
    let contribution = leased(&owner, fixture(2, 3, 0, false, false, 0), None);
    let mut pending = pending;
    pending.deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    expired(pending.accumulate_domain(idx, 0, &contribution));
    assert_eq!(used(&owner), contribution._charge.bytes);
}

#[test]
fn bounded_merge_aggregate_budget_refusal_keeps_only_external_source_lease() {
    let plan = plan();
    let idx = plan.output_node_idx;
    let owner = SoaStorageBudget::new(0, None).unwrap();
    let contribution = leased(&owner, fixture(2, 3, 3, false, false, 0), None);
    let pending = BoundedBatchPending::try_new(&plan, 3, &owner, None).unwrap();
    let remaining = owner.ceiling_bytes - owner.retained_bytes - used(&owner);
    let held = owner.charge(remaining, &mut Poll::new(None)).unwrap();
    let result = pending.accumulate_domain(idx, 1, &contribution);
    match result {
        Err(error) => assert!(error.is_cpu_memory_exceeded(), "{error}"),
        Ok(_) => panic!("expected aggregate refusal"),
    }
    assert_eq!(used(&owner), held.bytes + contribution._charge.bytes);
    drop(held);
    drop(contribution);
    assert_eq!(used(&owner), 0);
}

#[test]
fn bounded_merge_keeps_domain_order_input_flags_and_original_authority() {
    let plan = plan();
    let owner = SoaStorageBudget::new(0, Some(Instant::now() + Duration::from_secs(30))).unwrap();
    let mut pending = BoundedBatchPending::try_new(&plan, 5, &owner, None).unwrap();
    let mut reference = IndexedPendingLinearBounds::new(&plan, 5);
    for (step, idx, domain) in [
        (0, plan.output_node_idx, 4),
        (1, plan.network_input_idx, 2),
        (2, plan.network_input_idx, 4),
        (3, plan.output_node_idx, 4),
    ] {
        let bounds = fixture(2, 3, step % 4, false, true, step);
        reference
            .accumulate_idx(idx, bounds.clone(), domain)
            .unwrap();
        let contribution = leased(&owner, bounds, None);
        pending = pending
            .accumulate_domain(idx, domain, &contribution)
            .unwrap();
        assert_eq!(pending.deadline, owner.deadline);
        assert_eq!(pending.input_accumulated(), reference.input_accumulated());
    }
    for idx in [plan.output_node_idx, plan.network_input_idx] {
        let expected = reference.take_idx(idx).unwrap();
        let (next, slot) = pending.take(idx).unwrap();
        pending = next;
        let slot = slot.unwrap();
        for (domain, expected) in expected.iter().enumerate() {
            let actual = slot.try_materialize(domain, None).unwrap();
            assert_eq!(actual.is_some(), expected.is_some());
            if let (Some(actual), Some(expected)) = (actual, expected) {
                same(actual.bounds(), expected);
            }
        }
    }
    drop(pending);
    assert_eq!(used(&owner), 0);
}
