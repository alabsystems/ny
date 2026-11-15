// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkBudget as QueryBudget, WorkLimits, WorkRefusal, WorkReserve};
use crate::margin_row::rounding::RoundMode;
use std::time::Duration;

fn resources(cap: usize) -> (QueryBudget, MemoryAuthority, CancellationToken, Instant) {
    let at = Instant::now() + Duration::from_secs(60);
    let memory = MemoryAuthority::new(cap);
    let cancel = CancellationToken::default();
    let budget = QueryBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(at),
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: 10_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(cap).unwrap();
    (budget, memory, cancel, at)
}
fn path(index: u32, positive: bool) -> LeafPath {
    LeafPath::from_literals(vec![CoverLiteral {
        site: CoverSite::Head { index },
        positive,
    }])
    .unwrap()
}
fn evidence() -> Option<LeafEvidence> {
    crate::certify::margin_row::bound_evidence(RoundMode::Outward, 0.75)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}

#[test]
fn paid_cover_real_partition_preserves_legacy_cover_and_paid_classes() {
    let (budget, memory, _, at) = resources(1 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 2, at).unwrap();
    let off = path(1, false);
    let on = path(1, true);
    paid.leaf(Some(&off), evidence(), None).unwrap();
    paid.leaf(Some(&on), evidence(), None).unwrap();
    let cover = paid.prove(&[7, 2, 7, 4097], None).unwrap();
    let old = paid.ledger(None).unwrap().prove(&[7, 2, 7, 4097]).unwrap();
    assert_eq!(cover.cover(None).unwrap(), &old);
    assert_eq!(cover.cover(None).unwrap().classes(), [2, 7, 4097]);
    assert!(live(&memory) > 0);
    drop(paid);
    assert!(live(&memory) > 0);
    drop(cover);
    assert_eq!(live(&memory), 0);
}
#[test]
fn paid_cover_duplicates_missing_and_split_misidentity_still_refuse() {
    for (first, second, expected) in [
        (path(1, false), path(1, false), "Uncovered"),
        (path(1, false), path(2, true), "NoCommonSplit"),
        (LeafPath::root(), path(1, true), "Overlap"),
    ] {
        let (budget, _, _, at) = resources(1 << 20);
        let mut paid = PaidCoverLedger::new(&budget, 2, at).unwrap();
        paid.leaf(Some(&first), evidence(), None).unwrap();
        paid.leaf(Some(&second), evidence(), None).unwrap();
        let old = paid.ledger(None).unwrap().prove(&[1]).unwrap_err();
        let new = paid.prove(&[1], None).err().unwrap();
        assert!(format!("{old:?}").contains(expected));
        match new {
            PaidCoverError::Cover(actual) => assert_eq!(actual, old),
            _ => panic!("partition refusal changed"),
        }
    }
}
#[test]
fn paid_cover_missing_path_and_uncertified_leaf_preserve_first_refusal() {
    let (budget, _, _, at) = resources(1 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 2, at).unwrap();
    paid.leaf(None, evidence(), None).unwrap();
    paid.leaf(Some(&LeafPath::root()), None, None).unwrap();
    assert_eq!(paid.recorded(), 2);
    assert!(matches!(
        paid.prove(&[1], None),
        Err(PaidCoverError::Cover(CoverRefusal::BadPath { leaf: 0 }))
    ));
}
#[test]
fn paid_cover_capacity_refusal_preserves_actual_previous_leaf() {
    let (budget, _, _, at) = resources(1 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 1, at).unwrap();
    paid.leaf(Some(&path(1, false)), evidence(), None).unwrap();
    let before = budget.receipt().unwrap();
    assert!(matches!(
        paid.leaf(Some(&path(1, true)), evidence(), None),
        Err(PaidCoverError::Resource(WorkRefusal::PersistentLimit))
    ));
    assert_eq!(paid.recorded(), 1);
    assert_eq!(paid.ledger(None).unwrap().leaves.len(), 1);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
}
#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn paid_cover_header_and_paths_admit_before_allocation_on_same_nested_ledger() {
    let (budget, memory, _, at) = resources(8192);
    let first = PaidCoverLedger::new(&budget, 1, at).unwrap();
    let before = live(&memory);
    assert!(PaidCoverLedger::new(&budget.clone(), 1, at).is_err());
    assert_eq!(live(&memory), before);
    drop(first);
    assert_eq!(live(&memory), 0);
}
#[test]
fn paid_cover_cancelled_and_earlier_clock_cannot_publish_or_renew() {
    let (budget, memory, cancel, at) = resources(1 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 2, at).unwrap();
    let early = Instant::now() + Duration::from_secs(30);
    paid.leaf(Some(&path(1, false)), evidence(), Some(early))
        .unwrap();
    assert!(paid.deadline(None) <= early);
    let before = live(&memory);
    cancel.cancel();
    assert!(paid.leaf(Some(&path(1, true)), evidence(), None).is_err());
    assert_eq!(paid.recorded(), 1);
    assert_eq!(live(&memory), before);
    assert!(paid.prove(&[1], None).is_err());
    drop(paid);
    assert_eq!(live(&memory), 0);
}
#[test]
fn paid_cover_expired_constructor_has_no_actual_admission() {
    let (budget, memory, _, _) = resources(1 << 20);
    assert!(matches!(
        PaidCoverLedger::new(
            &budget,
            2,
            Instant::now().checked_sub(Duration::from_secs(1)).unwrap()
        ),
        Err(PaidCoverError::Resource(WorkRefusal::Deadline))
    ));
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().admitted, 0);
}
#[test]
fn paid_cover_deep_differently_ordered_real_partition_matches_legacy() {
    let (budget, _, _, at) = resources(1 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 4, at).unwrap();
    for (outer, inner) in [(false, false), (false, true), (true, false), (true, true)] {
        let p = LeafPath::from_literals(vec![
            CoverLiteral {
                site: CoverSite::Head { index: 9 },
                positive: inner,
            },
            CoverLiteral {
                site: CoverSite::Head { index: 1 },
                positive: outer,
            },
        ])
        .unwrap();
        paid.leaf(Some(&p), evidence(), None).unwrap();
    }
    let new = paid.prove(&[9, 7, 5, 3, 1], None).unwrap();
    assert_eq!(
        new.cover(None).unwrap(),
        &paid.ledger(None).unwrap().prove(&[9, 7, 5, 3, 1]).unwrap()
    );
}
#[test]
fn paid_cover_full_vocabulary_uses_actual_usize_classes_and_linear_storage() {
    let (budget, _, _, at) = resources(16 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 1, at).unwrap();
    paid.leaf(Some(&LeafPath::root()), evidence(), None)
        .unwrap();
    let classes: Vec<_> = (0..50_303usize).rev().collect();
    let cover = paid.prove(&classes, None).unwrap();
    assert_eq!(cover.cover(None).unwrap().classes().len(), 50_303);
    assert_eq!(cover.cover(None).unwrap().classes()[50_302], 50_302);
    assert!(cover.resident_bytes() < 2 * classes.len() * size_of::<usize>() + 8192);
}

#[test]
fn paid_cover_errors_preserve_original_cause_and_display() {
    use std::error::Error;
    for refusal in [
        CoverRefusal::NoLeaves,
        CoverRefusal::BadPath { leaf: 3 },
        CoverRefusal::Uncovered { depth: 7 },
    ] {
        let error = PaidCoverError::Cover(refusal.clone());
        assert_eq!(error.to_string(), format!("paid cover: Cover({refusal:?})"));
        let source = error.source().unwrap();
        let cause = source.downcast_ref::<CoverRefusal>().unwrap();
        assert_eq!(cause, &refusal);
        let PaidCoverError::Cover(original) = &error else {
            unreachable!()
        };
        assert!(std::ptr::eq(cause, original));
        assert!(source.source().is_none());
    }
    let error = PaidCoverError::Resource(WorkRefusal::Deadline);
    assert_eq!(error.to_string(), "paid cover: Resource(Deadline)");
    assert_eq!(
        error.source().unwrap().downcast_ref::<WorkRefusal>(),
        Some(&WorkRefusal::Deadline)
    );
    let allocation = PaidCoverError::Allocation;
    assert_eq!(allocation.to_string(), "paid cover: Allocation");
    assert!(allocation.source().is_none());
}

#[test]
fn paid_cover_retained_header_keeps_class_payload_price_and_last_alias() {
    let (budget, memory, _, at) = resources(1 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 1, at).unwrap();
    paid.leaf(Some(&LeafPath::root()), evidence(), None)
        .unwrap();
    let classes = [7, 2, 7, 4097];
    let cover = paid.prove(&classes, None).unwrap();
    let expected = PAID_COVER_META
        + size_of::<PaidTreeCover>()
        + size_of::<PaidTreeCoverData>()
        + 2 * classes.len() * size_of::<usize>();
    assert_eq!(cover.resident_bytes(), expected);
    assert_eq!(cover.cover(None).unwrap().classes(), [2, 7, 4097]);
    let last = PaidTreeCover(cover.0.clone());
    assert_eq!(
        std::ptr::from_ref(cover.cover(None).unwrap()),
        std::ptr::from_ref(last.cover(None).unwrap())
    );
    let held = cover.resident_bytes();
    drop(paid);
    drop(cover);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, held);
    assert_eq!(live(&memory), held);
    let spent = budget.receipt().unwrap().work_reserved;
    last.checkpoint(None).unwrap();
    drop(last);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().work_reserved, spent);
}

#[test]
fn paid_cover_retained_guard_refuses_foreign_data_ledger_and_expired_clock() {
    let (budget, memory, _, at) = resources(1 << 20);
    let (foreign, _, _, _) = resources(1 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 1, at).unwrap();
    paid.leaf(Some(&LeafPath::root()), evidence(), None)
        .unwrap();
    let mut cover = paid.prove(&[1], None).unwrap();
    let before = budget.receipt().unwrap();
    cover.0.get_mut().unwrap().budget = foreign;
    assert!(matches!(
        cover.checkpoint(None),
        Err(PaidCoverError::Resource(WorkRefusal::Unavailable))
    ));
    assert_eq!(budget.receipt().unwrap(), before);
    cover.0.get_mut().unwrap().budget = budget.clone();
    cover.checkpoint(None).unwrap();
    cover.0.get_mut().unwrap().deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock can represent the original expired deadline");
    assert!(matches!(
        cover.checkpoint(Some(at)),
        Err(PaidCoverError::Resource(WorkRefusal::Deadline))
    ));
    assert_eq!(budget.receipt().unwrap(), before);
    drop(cover);
    drop(paid);
    assert_eq!(live(&memory), 0);
}

#[test]
fn paid_cover_native_header_price_and_final_owner_observation() {
    use crate::certify::work::PersistentReservation;
    let (budget, memory, _, at) = resources(1 << 20);
    let mut paid = PaidCoverLedger::new(&budget, 1, at).unwrap();
    paid.leaf(Some(&LeafPath::root()), evidence(), None)
        .unwrap();
    let original = [7, 2, 7, 4097];
    let cover = paid.prove(&original, None).unwrap();
    let handle = size_of::<PaidTreeCover>();
    let data = size_of::<PaidTreeCoverData>();
    let header = handle.checked_add(data).unwrap();
    let classes_overlap = 2 * original.len() * size_of::<usize>();
    assert_eq!(
        cover.resident_bytes(),
        PAID_COVER_META + header + classes_overlap
    );
    assert_eq!(cover.cover(None).unwrap().classes(), [2, 7, 4097]);
    let resident = cover.resident_bytes();
    drop(paid);
    let held = budget.receipt().unwrap();
    let memory_held = memory.receipt().unwrap();
    assert_eq!(held.live_persistent_bytes, resident);
    assert_eq!(held.live_scratch_bytes, 0);
    assert_eq!(live(&memory), resident);
    eprintln!(
        "PAID_COVER_NATIVE_RETAINED handle={} data={} priced_header={} tree_cover={} reservation_handle={} classes={} classes_overlap={} meta={} resident={} work={} proposals={} memory_admitted={} memory_pending={} memory_materialized={}",
        handle, data, header, size_of::<TreeCover>(), size_of::<PersistentReservation>(),
        original.len(), classes_overlap, PAID_COVER_META, resident,
        held.work_reserved, held.proposals_reserved, memory_held.admitted,
        memory_held.live_pending_bytes, memory_held.live_materialized_bytes,
    );
    drop(cover);
    let released = budget.receipt().unwrap();
    assert_eq!(released.live_persistent_bytes, 0);
    assert_eq!(released.live_scratch_bytes, 0);
    assert_eq!(released.work_reserved, held.work_reserved);
    assert_eq!(released.proposals_reserved, held.proposals_reserved);
    assert_eq!(live(&memory), 0);
}
