// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::layers::LinearLayer;
use crate::{GraphNetwork, GraphNode, Layer};
use ndarray::{s, Array, Array1, Array2, Axis, Dimension, Slice};
use std::time::Duration;

fn query(
    memory: &MemoryAuthority,
    cancel: &CancellationToken,
    work: u64,
    persistent: usize,
    deadline: Option<Instant>,
    reserve: Duration,
) -> WorkBudget {
    WorkBudget::new_in(
        memory,
        cancel,
        WorkLimits {
            deadline,
            scratch_bytes: 1 << 22,
            persistent_bytes: persistent,
            max_work: work,
            max_proposals: 0,
            certificate_reserve: reserve,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap()
}

fn spare(rows: usize, cols: usize, capacity: usize, adversarial: bool) -> (Array2<f32>, usize) {
    let bits = [0, 0x8000_0000, 1, 0x3f80_0000, 0x7fc0_1234, 0x7f80_0000];
    let mut raw = Vec::with_capacity(capacity);
    raw.push(91.0);
    for i in 0..rows * cols {
        raw.push(if adversarial {
            f32::from_bits(bits[i % bits.len()])
        } else {
            (i % 7) as f32 * 0.125
        });
    }
    raw.push(92.0);
    let actual = raw.capacity();
    let values = Array1::from_vec(raw).slice_axis_move(Axis(0), Slice::from(1..1 + rows * cols));
    (values.into_shape_with_order((rows, cols)).unwrap(), actual)
}
fn fixture(mask: [bool; 2], adversarial: bool) -> (LinearBounds, usize) {
    let (la, c0) = spare(2, 3, 64, adversarial);
    let (lb, c1) = spare(1, 2, 16, adversarial);
    let (ua, c2) = spare(2, 3, 95, adversarial);
    let (ub, c3) = spare(1, 2, 21, adversarial);
    let (le, c4) = spare(2, 3, 64, adversarial);
    let (ue, c5) = spare(2, 3, 77, adversarial);
    let seed = LinearBounds {
        lower_a: la,
        lower_b: lb.into_shape_with_order(2).unwrap(),
        upper_a: ua,
        upper_b: ub.into_shape_with_order(2).unwrap(),
        lower_a_err: mask[0].then_some(le),
        upper_a_err: mask[1].then_some(ue),
    };
    (
        seed,
        4 * (c0 + c1 + c2 + c3 + if mask[0] { c4 } else { 0 } + if mask[1] { c5 } else { 0 }),
    )
}
type ArrayIdentity = (usize, Vec<usize>, Vec<isize>, Vec<u32>);
fn array_identity<D: Dimension>(values: &Array<f32, D>) -> ArrayIdentity {
    (
        values.as_ptr() as usize,
        values.shape().to_vec(),
        values.strides().to_vec(),
        values.iter().map(|v| v.to_bits()).collect(),
    )
}
fn identity(seed: &LinearBounds) -> Vec<Option<ArrayIdentity>> {
    vec![
        Some(array_identity(&seed.lower_a)),
        Some(array_identity(&seed.lower_b)),
        Some(array_identity(&seed.upper_a)),
        Some(array_identity(&seed.upper_b)),
        seed.lower_a_err.as_ref().map(array_identity),
        seed.upper_a_err.as_ref().map(array_identity),
    ]
}
fn bits(seed: &LinearBounds) -> Vec<Option<Vec<u32>>> {
    identity(seed)
        .into_iter()
        .map(|a| a.map(|(_, _, _, bits)| bits))
        .collect()
}
fn plan() -> CrownDispatchPlan {
    let mut graph = GraphNetwork::new();
    graph
        .try_add_node(GraphNode::from_input(
            "out",
            Layer::Linear(LinearLayer::new(Array2::eye(3), None).unwrap()),
        ))
        .unwrap();
    graph.set_output("out");
    CrownDispatchPlan::build(&graph).unwrap()
}

#[test]
fn retained_seed_preserves_original_six_array_pointers_offsets_capacities_and_bits() {
    for mask in [[false, false], [true, false], [false, true], [true, true]] {
        let (original, payload) = fixture(mask, true);
        let before = identity(&original);
        let memory = MemoryAuthority::new(1 << 24);
        let budget = query(
            &memory,
            &CancellationToken::default(),
            1 << 30,
            1 << 22,
            None,
            Duration::ZERO,
        );
        let seed =
            RetainedSoaSeed::try_adopt(original, &budget, LeaseClass::Fallback, None).unwrap();
        assert_eq!(identity(seed.seed()), before);
        let bytes = size_of::<RetainedSoaSeed>()
            + PersistentReservation::allocation_request_bytes().unwrap()
            + payload;
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            PersistentReservation::allocation_request_bytes().unwrap(),
            56
        );
        assert_eq!(seed.retained_bytes(), bytes);
        assert_eq!(budget.receipt().unwrap().work_reserved, 64);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, bytes);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, bytes);
        let storage = seed.storage_budget(&budget, None).unwrap();
        let root_bytes = budget.receipt().unwrap().live_scratch_bytes;
        assert_eq!(
            memory.receipt().unwrap().live_materialized_bytes,
            bytes + root_bytes,
            "binding must not charge the same seed again"
        );
        let original = seed.into_original();
        assert_eq!(identity(&original), before);
        assert_eq!(
            budget.receipt().unwrap().live_persistent_bytes,
            bytes,
            "the storage claim keeps its original producer claim"
        );
        drop(storage);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    }
}

#[test]
fn spare_backing_capacity_controls_admission_and_refusal_returns_exact_original() {
    for admit in [false, true] {
        let (original, _) = fixture([true, true], false);
        let before = identity(&original);
        let (original, quote) = observe_seed(original);
        let bytes = quote.unwrap().unwrap();
        assert!(bytes > original.memory_bytes());
        let memory = MemoryAuthority::new(bytes);
        let budget = query(
            &memory,
            &CancellationToken::default(),
            64,
            if admit { bytes } else { bytes - 1 },
            None,
            Duration::ZERO,
        );
        let result = RetainedSoaSeed::try_adopt(original, &budget, LeaseClass::Fallback, None);
        let restored = if admit {
            let seed = result.unwrap();
            assert_eq!(seed.retained_bytes(), bytes);
            assert_eq!(memory.receipt().unwrap().live_materialized_bytes, bytes);
            seed.into_original()
        } else {
            let failure = match result {
                Err(failure) => failure,
                Ok(_) => panic!("one-byte short admission succeeded"),
            };
            let (restored, error, unsupported) = failure.into_parts();
            assert!(!unsupported && matches!(error, NyError::UnsupportedOp(_)));
            restored
        };
        assert_eq!(identity(&restored), before);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(
            budget.receipt().unwrap().work_reserved,
            64,
            "failed admission cannot refund or renew work"
        );
    }
}

#[test]
fn retained_seed_broadcast_uses_original_planes_and_keeps_last_query_claim() {
    let plan = plan();
    for mask in [[false, false], [true, false], [false, true], [true, true]] {
        for idx in [plan.output_node_idx, plan.network_input_idx] {
            let (original, _) = fixture(mask, false);
            let expected = bits(&original);
            let before = identity(&original);
            let memory = MemoryAuthority::new(1 << 24);
            let budget = query(
                &memory,
                &CancellationToken::default(),
                1 << 30,
                1 << 22,
                None,
                Duration::ZERO,
            );
            let seed =
                RetainedSoaSeed::try_adopt(original, &budget, LeaseClass::Fallback, None).unwrap();
            let bytes = seed.retained_bytes();
            let seeded = seed.seed_pending(&budget, &plan, 3, idx, None);
            if mask.iter().any(|&present| present) {
                // Custody preserves error planes; the original seed-broadcast
                // contract still accepts plain-spec seeds only. This refusal
                // must neither erase source errors nor leak a query claim.
                let error = match seeded {
                    Err(error) => error,
                    Ok(_) => panic!("original plain-spec seed contract was widened"),
                };
                assert!(matches!(&error, NyError::InvalidSpec(_)));
                assert!(error
                    .to_string()
                    .contains("seed must retain the original plain-spec contract"));
                assert_eq!(identity(seed.seed()), before);
                assert_eq!(bits(seed.seed()), expected);
                assert_eq!(budget.receipt().unwrap().live_persistent_bytes, bytes);
                assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
                drop(seed);
                assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
                assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
                continue;
            }
            let pending = seeded.unwrap();
            assert_eq!(identity(seed.seed()), before);
            assert_eq!(
                pending.input_accumulated(),
                &[idx == plan.network_input_idx; 3]
            );
            drop(seed);
            assert_eq!(budget.receipt().unwrap().live_persistent_bytes, bytes);
            let (pending, slot) = pending.take(idx).unwrap();
            let slot = slot.unwrap();
            for domain in 0..3 {
                let output = slot.try_materialize(domain, None).unwrap().unwrap();
                assert_eq!(bits(output.bounds()), expected);
            }
            drop(slot);
            drop(pending);
            assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
            assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
            assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
        }
    }
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn retained_seed_cannot_switch_query_or_extend_original_class_and_deadline() {
    let now = Instant::now();
    let end = now + Duration::from_secs(60);
    let memory = MemoryAuthority::new(1 << 24);
    let budget = query(
        &memory,
        &CancellationToken::default(),
        1 << 30,
        1 << 22,
        Some(end),
        Duration::from_secs(20),
    );
    // Exercise successful seeding with the original plain-spec seed class.
    // All nonplain masks remain checked as refusals in the broadcast test.
    let (original, _) = fixture([false, false], false);
    let seed = RetainedSoaSeed::try_adopt(
        original,
        &budget,
        LeaseClass::Optional,
        Some(end + Duration::from_secs(30)),
    )
    .unwrap();
    assert_eq!(
        seed.deadline,
        Some(end.checked_sub(Duration::from_secs(20)).unwrap())
    );
    let same_memory_other_query = query(
        &memory,
        &CancellationToken::default(),
        1 << 30,
        1 << 22,
        Some(end),
        Duration::ZERO,
    );
    let foreign_query = query(
        &MemoryAuthority::new(1 << 24),
        &CancellationToken::default(),
        1 << 30,
        1 << 22,
        Some(end),
        Duration::ZERO,
    );
    for other in [&same_memory_other_query, &foreign_query] {
        assert!(seed.storage_budget(other, None).is_err());
        assert_eq!(other.receipt().unwrap().admitted, 0);
    }
    let narrow = now + Duration::from_secs(10);
    let plan = plan();
    let pending = seed
        .seed_pending(
            &budget.clone(),
            &plan,
            1,
            plan.output_node_idx,
            Some(narrow),
        )
        .unwrap();
    let (pending, slot) = pending.take(plan.output_node_idx).unwrap();
    let output = slot
        .unwrap()
        .try_materialize(0, Some(end))
        .unwrap()
        .unwrap();
    assert_eq!(output.deadline(), Some(narrow));
    drop(output);
    drop(pending);
    drop(seed);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn original_work_and_global_byte_refusals_keep_prior_live_owners_and_seed() {
    for work_refusal in [false, true] {
        let (original, _) = fixture([true, true], false);
        let before = identity(&original);
        let (original, quote) = observe_seed(original);
        let bytes = quote.unwrap().unwrap();
        let memory = MemoryAuthority::new(bytes + 7);
        let producer = query(
            &memory,
            &CancellationToken::default(),
            7,
            8,
            None,
            Duration::ZERO,
        );
        let prior = producer
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: 8,
                    work: 7,
                    ..WorkRequest::default()
                },
            )
            .unwrap()
            .commit_persistent()
            .unwrap();
        let budget = query(
            &memory,
            &CancellationToken::default(),
            if work_refusal { 63 } else { 64 },
            1 << 22,
            None,
            Duration::ZERO,
        );
        let failure =
            match RetainedSoaSeed::try_adopt(original, &budget, LeaseClass::Fallback, None) {
                Err(failure) => failure,
                Ok(_) => panic!("expected original authority refusal"),
            };
        let (restored, _, unsupported) = failure.into_parts();
        assert!(!unsupported);
        assert_eq!(identity(&restored), before);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 8);
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(producer.receipt().unwrap().work_reserved, 7);
        drop(prior);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    }
}

#[test]
fn cancellation_and_expired_entry_never_renew_or_release_original_seed_claims() {
    let cancel = CancellationToken::default();
    let memory = MemoryAuthority::new(1 << 24);
    let budget = query(&memory, &cancel, 1 << 30, 1 << 22, None, Duration::ZERO);
    let (original, _) = fixture([true, true], false);
    let before = identity(&original);
    let seed = RetainedSoaSeed::try_adopt(original, &budget, LeaseClass::Fallback, None).unwrap();
    let bytes = seed.retained_bytes();
    cancel.cancel();
    assert!(
        matches!(seed.seed_pending(&budget, &plan(), 1, 0, None), Err(error) if error.is_deadline_exceeded())
    );
    assert_eq!(identity(seed.seed()), before);
    assert_eq!(budget.receipt().unwrap().work_reserved, 64);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, bytes);
    let restored = seed.into_original();
    assert_eq!(identity(&restored), before);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);

    let budget = query(
        &memory,
        &CancellationToken::default(),
        64,
        1 << 22,
        None,
        Duration::ZERO,
    );
    let failure = match RetainedSoaSeed::try_adopt(
        restored,
        &budget,
        LeaseClass::Fallback,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap()),
    ) {
        Err(failure) => failure,
        Ok(_) => panic!("expired seed admission succeeded"),
    };
    let (restored, error, unsupported) = failure.into_parts();
    assert!(error.is_deadline_exceeded() && !unsupported);
    assert_eq!(identity(&restored), before);
    assert_eq!(budget.receipt().unwrap().admitted, 0);
    let unjoined = WorkBudget::new(None, 1 << 22, 1 << 22, 64, 0, Duration::ZERO).unwrap();
    let failure = match RetainedSoaSeed::try_adopt(restored, &unjoined, LeaseClass::Fallback, None)
    {
        Err(failure) => failure,
        Ok(_) => panic!("unjoined seed admission succeeded"),
    };
    let (restored, _, unsupported) = failure.into_parts();
    assert!(!unsupported);
    assert_eq!(identity(&restored), before);
    assert_eq!(unjoined.receipt().unwrap().admitted, 0);
}

#[test]
fn each_noncanonical_seed_role_refuses_cleanly_and_returns_all_original_owners() {
    for role in 0..6 {
        let (mut original, _) = fixture([true, true], false);
        match role {
            0 => original.lower_a = original.lower_a.slice_move(s![.., ..;-1]),
            1 => original.lower_b = original.lower_b.slice_move(s![..;-1]),
            2 => original.upper_a = original.upper_a.slice_move(s![.., ..;-1]),
            3 => original.upper_b = original.upper_b.slice_move(s![..;-1]),
            4 => original.lower_a_err = original.lower_a_err.map(|a| a.slice_move(s![.., ..;-1])),
            5 => original.upper_a_err = original.upper_a_err.map(|a| a.slice_move(s![.., ..;-1])),
            _ => unreachable!(),
        }
        let before = identity(&original);
        let memory = MemoryAuthority::new(1 << 24);
        let budget = query(
            &memory,
            &CancellationToken::default(),
            64,
            1 << 22,
            None,
            Duration::ZERO,
        );
        let failure =
            match RetainedSoaSeed::try_adopt(original, &budget, LeaseClass::Fallback, None) {
                Err(failure) => failure,
                Ok(_) => panic!("unsupported role {role} admitted"),
            };
        let (restored, error, unsupported) = failure.into_parts();
        assert!(unsupported && matches!(error, NyError::UnsupportedOp(_)));
        assert_eq!(identity(&restored), before);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
        assert_eq!(budget.receipt().unwrap().work_reserved, 64);
    }
}

#[test]
fn hard_capacity_invariants_and_overflow_are_never_hidden_by_unsupported_roles() {
    let facts = [
        Ok(None),
        Ok(Some(4)),
        Err(invalid("injected hard provenance invariant")),
        Ok(Some(4)),
        Ok(Some(0)),
        Ok(Some(0)),
        Ok(Some(56)),
    ];
    assert!(
        matches!(quote_seed(facts), Err(NyError::InvalidSpec(reason)) if reason.contains("injected hard"))
    );
    let facts = [
        Ok(None),
        Ok(Some(usize::MAX)),
        Ok(Some(4)),
        Ok(Some(4)),
        Ok(Some(0)),
        Ok(Some(0)),
        Ok(Some(56)),
    ];
    assert!(
        matches!(quote_seed(facts), Err(NyError::InvalidSpec(reason)) if reason.contains("overflow"))
    );
}
