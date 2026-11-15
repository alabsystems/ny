// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::BatchPending;
use super::*;
use crate::layers::LinearLayer;
use crate::{GraphNetwork, GraphNode, Layer};
use ndarray::{array, Array2};
use std::sync::{atomic::AtomicUsize, Barrier};
use std::time::Duration;

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

fn budget(
    ceiling_bytes: usize,
    retained_bytes: usize,
    deadline: Option<Instant>,
) -> Arc<SoaStorageBudget> {
    Arc::new(SoaStorageBudget {
        retained_bytes,
        ceiling_bytes: ceiling_bytes.min(cpu_crown_dense_budget_bytes()),
        reserved_bytes: AtomicUsize::new(0),
        deadline,
        query: None,
    })
}

fn used(owner: &SoaStorageBudget) -> usize {
    owner.reserved_bytes.load(Ordering::Acquire)
}

fn assert_memory<T>(result: Result<T>) {
    match result {
        Err(error) => assert!(error.is_cpu_memory_exceeded(), "{error}"),
        Ok(_) => panic!("expected a memory refusal"),
    }
}

fn assert_expiry<T>(result: Result<T>, stage: Option<&str>) {
    match result {
        Err(error) => {
            assert!(error.is_deadline_exceeded(), "{error}");
            if let Some(stage) = stage {
                assert!(error.to_string().contains(stage), "{error}");
            }
        }
        Ok(_) => panic!("expected typed expiry"),
    }
}

fn forced(stage: &'static str) -> Poll {
    let mut poll = Poll::new(None);
    poll.forced_stage = Some(stage);
    poll
}

fn same(actual: &LinearBounds, expected: &LinearBounds) {
    for (a, b) in [
        (actual.lower_a(), expected.lower_a()),
        (actual.upper_a(), expected.upper_a()),
    ] {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    for (a, b) in [
        (actual.lower_b(), expected.lower_b()),
        (actual.upper_b(), expected.upper_b()),
    ] {
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    for (a, b) in [
        (actual.lower_a_err(), expected.lower_a_err()),
        (actual.upper_a_err(), expected.upper_a_err()),
    ] {
        assert_eq!(a.is_some(), b.is_some());
        if let (Some(a), Some(b)) = (a, b) {
            assert_eq!(a.shape(), b.shape());
            assert_eq!(
                a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn bounded_seed_matches_legacy_contiguous_and_strided_specs_bits() {
    let plan = plan();
    let idx = plan.output_node_idx;
    for nonstandard in [false, true] {
        let values = if nonstandard {
            array![[1.0_f32, -0.0, f32::from_bits(1)], [0.25, 3.5, -2.0]].reversed_axes()
        } else {
            array![[1.0_f32, -0.0], [f32::from_bits(1), 0.25], [3.5, -2.0]]
        };
        if nonstandard {
            assert!(values.as_slice().is_none());
        }
        let mut upper = values.clone();
        upper[[0, 0]] = 5.0;
        let seed = LinearBounds::new(
            values,
            array![0.125_f32, -0.0, f32::from_bits(1)],
            upper,
            array![0.25_f32, 0.0, f32::from_bits(2)],
        )
        .unwrap();
        for domains in [1, 3, 8] {
            let owner = SoaStorageBudget::new(seed.memory_bytes(), None).unwrap();
            let mut legacy = BatchPending::new(&plan, domains);
            legacy.seed_broadcast(idx, &seed).unwrap();
            let legacy = legacy.take(idx).unwrap();
            let pending = BoundedBatchPending::try_new(&plan, domains, &owner, None)
                .unwrap()
                .seed_broadcast(idx, &seed)
                .unwrap();
            let (pending, slot) = pending.take(idx).unwrap();
            let slot = slot.unwrap();
            assert!(slot.slot().active.iter().all(|a| *a));
            assert!(slot
                .slot()
                .lower_err_present
                .iter()
                .chain(&slot.slot().upper_err_present)
                .all(|a| !*a));
            for domain in 0..domains {
                let actual = slot.try_materialize(domain, None).unwrap().unwrap();
                same(actual.bounds(), &legacy.materialize(domain).unwrap());
            }
            drop(slot);
            drop(pending);
            assert_eq!(used(&owner), 0);
        }
    }
}

#[test]
fn bounded_materialization_keeps_exact_markers_stress_bits_and_source_lifetime() {
    for stress in [false, true] {
        let owner = budget(1 << 20, 0, None);
        let mut slot = BoundedBatchSlot::try_zeroed(&owner, 5, 2, 3, None).unwrap();
        for d in 0..4 {
            slot.storage.active[d] = true;
            slot.storage.lower_err_present[d] = d & 1 != 0;
            slot.storage.upper_err_present[d] = d & 2 != 0;
        }
        let bits = if stress {
            vec![
                0x7fc01234_u32,
                0x7f800000,
                0xff800000,
                0x80000000,
                1,
                0x3f800000,
            ]
        } else {
            vec![
                0x3f800000_u32,
                0x80000000,
                1,
                0x40000000,
                0x3e800000,
                0x40400000,
            ]
        };
        for (plane_index, plane) in [
            &mut slot.storage.lower_a,
            &mut slot.storage.upper_a,
            &mut slot.storage.lower_b,
            &mut slot.storage.upper_b,
            &mut slot.storage.lower_a_err,
            &mut slot.storage.upper_a_err,
        ]
        .into_iter()
        .enumerate()
        {
            for (i, value) in plane.iter_mut().enumerate() {
                *value = f32::from_bits(bits[(i + plane_index) % bits.len()]);
            }
        }
        let baseline = used(&owner);
        for d in 0..4 {
            let expected = slot.storage.materialize(d).unwrap();
            let actual = slot.try_materialize(d, None).unwrap().unwrap();
            if !stress {
                assert!(actual
                    .bounds()
                    .lower_a()
                    .iter()
                    .chain(actual.bounds().upper_a())
                    .all(|x| x.is_finite()));
            }
            same(actual.bounds(), &expected);
            assert!(used(&owner) > baseline);
            drop(actual);
            assert_eq!(used(&owner), baseline);
        }
        assert!(slot.try_materialize(4, None).unwrap().is_none());
        assert_eq!(used(&owner), baseline);
        assert!(slot.try_materialize(5, None).is_err());
        assert_eq!(used(&owner), baseline);
        let surviving = slot.try_materialize(3, None).unwrap().unwrap();
        let expected = slot.storage.materialize(3).unwrap();
        drop(slot);
        assert!(used(&owner) > 0);
        same(surviving.bounds(), &expected);
        drop(surviving);
        assert_eq!(used(&owner), 0);
    }
}

#[test]
fn bounded_storage_charges_aggregate_capacity_and_releases_all_error_paths() {
    assert!(
        chunk_elements::<Option<BoundedBatchSlot>>() * element_units::<Option<BoundedBatchSlot>>()
            <= POLL_CHUNK_UNITS
    );
    assert_eq!(chunk_elements::<f32>(), POLL_CHUNK_UNITS);
    let owner = budget(96, 16, None);
    let mut poll = Poll::new(None);
    let mut first = owner.charge(64, &mut poll).unwrap();
    assert_eq!(used(&owner), 64);
    assert_memory(owner.charge(17, &mut poll));
    assert_eq!(used(&owner), 64);
    first.reconcile::<f32>(4, 8, &mut poll).unwrap();
    assert_eq!(used(&owner), 80);
    assert_memory(first.reconcile::<f32>(4, 5, &mut poll));
    assert_eq!(used(&owner), 80);
    assert_memory(first.reconcile::<f32>(0, usize::MAX, &mut poll));
    assert_eq!(used(&owner), 80);
    drop(first);
    assert_eq!(used(&owner), 0);
    let bytes = slot_layout(2, 3, 4).unwrap().2;
    let owner = budget(bytes * 2 - 1, 0, None);
    let first = BoundedBatchSlot::try_zeroed(&owner, 2, 3, 4, None).unwrap();
    let before = used(&owner);
    assert_memory(BoundedBatchSlot::try_zeroed(&owner, 2, 3, 4, None));
    assert_eq!(used(&owner), before);
    drop(first);
    assert_eq!(used(&owner), 0);
    assert!(BoundedBatchSlot::try_zeroed(&owner, usize::MAX, 2, 2, None).is_err());
    assert_eq!(used(&owner), 0);
    assert!(BoundedBatchSlot::try_zeroed(&owner, 1, 0, 2, None).is_err());
    assert_eq!(used(&owner), 0);
    let mut nonplain = LinearBounds::identity(2);
    nonplain.lower_a_err = Some(Array2::zeros((2, 2)));
    assert!(BoundedBatchSlot::try_seeded(&owner, 1, &nonplain, None).is_err());
    assert_eq!(used(&owner), 0);
}

#[test]
fn bounded_storage_never_regrants_deadline_and_forced_copy_faults_publish_nothing() {
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let later = Instant::now() + Duration::from_secs(60);
    assert_expiry(SoaStorageBudget::new(usize::MAX, Some(expired)), None);
    let owner = budget(1 << 20, 0, Some(expired));
    assert_expiry(
        BoundedBatchSlot::try_zeroed(&owner, usize::MAX, 2, 2, Some(later)),
        None,
    );
    assert_eq!(used(&owner), 0);
    assert_eq!(narrower(Some(expired), None), Some(expired));
    assert_eq!(narrower(Some(expired), Some(later)), Some(expired));
    let owner = budget(1 << 20, 0, Some(later));
    let slot = BoundedBatchSlot::try_seeded(&owner, 2, &LinearBounds::identity(2), None).unwrap();
    assert_eq!(slot.deadline, Some(later));
    let before = used(&owner);
    assert_expiry(slot.try_materialize(0, Some(expired)), None);
    assert_eq!(used(&owner), before);
    drop(slot);
    assert_eq!(used(&owner), 0);
    let owner = budget(8 << 20, 0, None);
    let seed = LinearBounds::identity(65);
    let mut fill_lease = owner
        .charge(
            (POLL_CHUNK_UNITS * 2 + 7) * size_of::<f32>(),
            &mut Poll::new(None),
        )
        .unwrap();
    assert_expiry(
        filled(
            POLL_CHUNK_UNITS * 2 + 7,
            0.0_f32,
            &mut fill_lease,
            &mut forced(FILL),
        ),
        Some(FILL),
    );
    drop(fill_lease);
    assert_eq!(used(&owner), 0);
    for stage in [ENTRY, AFTER_RESERVE, FILL, COPY, SLOT_COMMIT, SEED_COMMIT] {
        assert_expiry(
            BoundedBatchSlot::seeded_with_poll(&owner, 2, &seed, None, &mut forced(stage)),
            Some(stage),
        );
        assert_eq!(used(&owner), 0);
    }
    let slot = BoundedBatchSlot::try_seeded(&owner, 2, &seed, None).unwrap();
    let before = used(&owner);
    for stage in [ENTRY, AFTER_RESERVE, COPY, RELATION_COMMIT] {
        assert_expiry(
            slot.materialize_with_poll(0, None, &mut forced(stage)),
            Some(stage),
        );
        assert_eq!(used(&owner), before);
    }
    drop(slot);
    assert_eq!(used(&owner), 0);
}

#[test]
fn bounded_pending_failure_discards_the_complete_private_transaction() {
    let plan = plan();
    let seed = LinearBounds::identity(2);
    let owner = budget(1 << 20, 0, None);
    let pending = BoundedBatchPending::try_new(&plan, 3, &owner, None).unwrap();
    assert!(used(&owner) > 0);
    assert!(pending.seed_broadcast(usize::MAX, &seed).is_err());
    assert_eq!(used(&owner), 0);
    let pending = BoundedBatchPending::try_new(&plan, 3, &owner, None)
        .unwrap()
        .seed_broadcast(plan.output_node_idx, &seed)
        .unwrap();
    assert!(pending.seed_broadcast(plan.output_node_idx, &seed).is_err());
    assert_eq!(used(&owner), 0);
    let pending = BoundedBatchPending::try_new(&plan, 3, &owner, None).unwrap();
    assert_expiry(
        pending.seed_with_poll(plan.network_input_idx, &seed, &mut forced(PENDING_COMMIT)),
        Some(PENDING_COMMIT),
    );
    assert_eq!(used(&owner), 0);
    let pending = BoundedBatchPending::try_new(&plan, 3, &owner, None)
        .unwrap()
        .seed_broadcast(plan.network_input_idx, &seed)
        .unwrap();
    assert_eq!(pending.input_accumulated, vec![true; 3]);
    let (pending, slot) = pending.take(plan.network_input_idx).unwrap();
    let slot = slot.unwrap();
    assert!(used(&owner) > 0);
    drop(pending);
    assert!(used(&owner) > 0);
    drop(slot);
    assert_eq!(used(&owner), 0);
}

#[test]
fn bounded_storage_concurrent_reservations_cannot_each_spend_the_full_allowance() {
    let owner = budget(64, 0, None);
    let barrier = Barrier::new(8);
    let successes = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let owner = &owner;
            let barrier = &barrier;
            let successes = &successes;
            scope.spawn(move || {
                let lease = owner.charge(64, &mut Poll::new(None));
                barrier.wait();
                if lease.is_ok() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }
                drop(lease);
            });
        }
    });
    assert_eq!(successes.load(Ordering::Relaxed), 1);
    assert_eq!(used(&owner), 0);
}
