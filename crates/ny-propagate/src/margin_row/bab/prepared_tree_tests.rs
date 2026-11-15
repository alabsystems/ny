// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::support_program::{
    PreparedRootOutcome, PreparedRootProofOptions, PreparedRootSession,
};
use crate::margin_row::{bab::MarginRowBab, RootPrecisionRuntimePolicy};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 32 << 20;
fn clock() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(clock()),
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(CAP).unwrap();
    (budget, memory, cancel)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn fixture(classes: usize, fold: bool) -> TwinSpec {
    let mut weights = vec![0.0; 2 * classes];
    weights[0] = 1.0;
    weights[1] = -1.0;
    let mut bias = vec![0.0; classes];
    bias[0] = 0.75;
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![if fold { 1.0 / 1024.0 } else { 0.0 }],
                weight_rel_err: if fold { 1.0 / 512.0 } else { 0.0 },
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![1.0, 1.0],
                bias: vec![-1.0, -1.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: weights,
                bias,
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (classes, 2),
            },
        ],
    }
}
fn original_open(budget: &WorkBudget, classes: usize, fold: bool) -> PreparedRootOpen {
    let session = PreparedRootSession::compile_declared_twin(
        fixture(classes, fold),
        &[0.0],
        &[3.0],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            prefer_patch: false,
            ..PreparedRootProofOptions::default()
        },
        budget,
        Some(clock()),
    )
    .unwrap();
    let order: Vec<_> = (1..classes).rev().collect();
    let PreparedRootOutcome::Open(open) = session.try_prove_root(0, &order, None).unwrap() else {
        panic!("original duplicate-head root must remain open")
    };
    assert!(open.receipt().ordinary_completed);
    assert!(!open.receipt().terminal_closed);
    open
}
fn options(schedule: PreparedTreeSchedule) -> PreparedTreeOptions {
    PreparedTreeOptions {
        max_expansions: 2,
        lru_capacity: 4,
        schedule,
    }
}
fn covered(outcome: PreparedTreeOutcome) -> PreparedTreeCertificate {
    let PreparedTreeOutcome::Covered(cert) = outcome else {
        panic!("two genuine head children must close")
    };
    cert
}
fn exact_margin(x: BigRational) -> BigRational {
    let trunk = x.max(BigRational::zero());
    let pre = trunk - BigRational::from_integer(1.into());
    let head = pre.max(BigRational::zero());
    BigRational::new(3.into(), 4.into()) + head.clone() - head
}
#[test]
fn prepared_tree_actual_marginrow_entry_open_root_two_paid_head_children_close_original_cover() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let open = original_open(&budget, 2, false);
    let cert = covered(
        MarginRowBab::run_prepared(open, options(PreparedTreeSchedule::Serial), None).unwrap(),
    );
    assert_eq!(cert.receipt().expansions, 1);
    assert_eq!(cert.receipt().bound_leaves, 2);
    assert_eq!(cert.cover(None).unwrap().classes(), [1]);
    assert_eq!(cert.cover(None).unwrap().leaves(), 2);
    assert_eq!(cert.leaves().count(), 2);
    for (evaluation, path) in cert.leaves() {
        assert!(path.domain().same_domain(evaluation.domain()));
        assert_eq!(path.path(None).unwrap().depth(), 1);
        assert_eq!(
            path.path(None).unwrap().literals()[0].site,
            crate::certify::CoverSite::Head { index: 1 }
        );
        let lower = evaluation.bounds(None).unwrap()[0];
        assert!(lower > 0.0);
        for x in [
            BigRational::zero(),
            BigRational::new(1.into(), 2.into()),
            BigRational::from_integer(1.into()),
            BigRational::from_integer(3.into()),
        ] {
            assert!(BigRational::from_float(lower).unwrap() <= exact_margin(x));
        }
    }
    assert!(live(&memory) > 0);
    drop(cert);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_tree_parallel_and_wave_actual_schedule_keep_same_original_children_and_scope() {
    let _env = crate::tests::lock_env_shared();
    for schedule in [
        PreparedTreeSchedule::Parallel { width: 2 },
        PreparedTreeSchedule::Wave { width: 2 },
    ] {
        let (budget, memory, _) = resources();
        let cert = covered(
            MarginRowBab::run_prepared(original_open(&budget, 2, false), options(schedule), None)
                .unwrap(),
        );
        assert_eq!(cert.receipt().expansions, 1);
        assert_eq!(cert.receipt().bound_leaves, 2);
        assert!(cert.root().budget().shares_ledger(&budget));
        match schedule {
            PreparedTreeSchedule::Parallel { .. } => assert_eq!(cert.receipt().parallel_batches, 1),
            PreparedTreeSchedule::Wave { .. } => assert_eq!(cert.receipt().wave_batches, 1),
            _ => unreachable!(),
        }
        cert.checkpoint(None).unwrap();
        drop(cert);
        assert_eq!(live(&memory), 0);
    }
}
#[test]
fn prepared_tree_full_original_order_above64_keeps_distinct_paid_leaf_bound_arrays() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let cert = covered(
        MarginRowBab::run_prepared(
            original_open(&budget, 66, false),
            options(PreparedTreeSchedule::Serial),
            None,
        )
        .unwrap(),
    );
    assert_eq!(cert.root().classes()[0], 65);
    assert_eq!(cert.cover(None).unwrap().classes().len(), 65);
    assert_eq!(cert.cover(None).unwrap().classes()[64], 65);
    let leaves: Vec<_> = cert.leaves().collect();
    let first = leaves[0].0.bounds(None).unwrap();
    let second = leaves[1].0.bounds(None).unwrap();
    assert_eq!(first.len(), 65);
    assert_eq!(second.len(), 65);
    assert_ne!(first.as_ptr(), second.as_ptr());
    assert_eq!(leaves[0].0.classes(), cert.root().classes());
    assert_eq!(leaves[1].0.classes(), cert.root().classes());
    assert!(first.iter().chain(second).all(|&v| v > 0.0 && v <= 0.75));
}
#[test]
fn prepared_tree_original_shared_fold_uncertainty_children_stay_below_independent_margin() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let cert = covered(
        MarginRowBab::run_prepared(
            original_open(&budget, 2, true),
            options(PreparedTreeSchedule::Serial),
            None,
        )
        .unwrap(),
    );
    for (leaf, _) in cert.leaves() {
        for &lower in leaf.bounds(None).unwrap() {
            assert!(lower > 0.0 && lower <= 0.75)
        }
    }
    // Both head coordinates are the same actual affine operation even when
    // shared original Conv parameters range over their fold uncertainty.
    for wc in [
        BigRational::new(511.into(), 512.into()),
        BigRational::new(513.into(), 512.into()),
    ] {
        for x in [BigRational::zero(), BigRational::from_integer(3.into())] {
            assert_eq!(
                exact_margin(wc.clone() * x),
                BigRational::new(3.into(), 4.into())
            );
        }
    }
}
#[test]
fn prepared_tree_expansion_limit_open_resume_keeps_actual_frontier_and_original_work() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let PreparedTreeOutcome::Open(open) = MarginRowBab::run_prepared(
        original_open(&budget, 2, false),
        PreparedTreeOptions {
            max_expansions: 0,
            lru_capacity: 1,
            schedule: PreparedTreeSchedule::Serial,
        },
        None,
    )
    .unwrap() else {
        panic!("zero expansions cannot claim root closure")
    };
    assert_eq!(open.reason(), PreparedTreeOpenReason::ExpansionLimit);
    assert_eq!(open.open_domains(), 1);
    assert_eq!(open.receipt().bound_leaves, 0);
    let before = budget.receipt().unwrap();
    let PreparedTreeOutcome::Open(open) = open.resume(None).unwrap() else {
        panic!("resume grants no expansion credit")
    };
    assert_eq!(open.receipt().expansions, 0);
    assert!(budget.receipt().unwrap().work_reserved >= before.work_reserved);
    drop(open);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_tree_exact_domain_lru_hit_shares_real_rows_and_spends_no_new_payload() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let mut state = PreparedTreeState::new(
        original_open(&budget, 2, false),
        options(PreparedTreeSchedule::Serial),
        None,
    )
    .unwrap();
    let domain = state.root.clone();
    let first = state.rows_for(&domain).unwrap();
    let before = live(&memory);
    let second = state.rows_for(&domain).unwrap();
    assert!(first.same_owner(&second));
    assert!(first.domain().same_domain(&domain));
    assert_eq!(live(&memory), before);
    assert_eq!(state.receipt.cache_hits, 1);
    drop(state);
    drop(first);
    assert!(live(&memory) > 0);
    drop(second);
    drop(domain);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_tree_second_leaf_cap_refusal_rolls_back_cover_and_restores_real_parent_then_recovers() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let mut state = PreparedTreeState::new(
        original_open(&budget, 2, false),
        options(PreparedTreeSchedule::Serial),
        None,
    )
    .unwrap();
    state.root_entry().unwrap();
    let mut held = None;
    let mut observer = |s: &mut PreparedTreeState, index: usize| {
        if index == 0 {
            let r = budget.receipt().unwrap();
            let available = CAP - r.live_scratch_bytes - r.live_persistent_bytes;
            let mut lease = budget
                .lease_for(
                    LeaseClass::Fallback,
                    WorkRequest {
                        persistent_bytes: available - 1,
                        ..WorkRequest::default()
                    },
                )
                .unwrap();
            lease.mark_materialized().unwrap();
            held = Some(lease.commit_persistent().unwrap());
            assert_eq!(s.ledger.recorded(), 1);
        }
    };
    let error = state.schedule_batch_inner(Some(&mut observer)).unwrap_err();
    assert!(recoverable(&error));
    state.restore_parents();
    assert_eq!(state.frontier.len(), 1);
    assert_eq!(state.ledger.recorded(), 0);
    assert_eq!(state.closed.len(), 0);
    assert_eq!(state.receipt.expansions, 0);
    assert_eq!(state.receipt.bound_leaves, 0);
    drop(held);
    let spent = budget.receipt().unwrap().work_reserved;
    assert_eq!(state.schedule_batch().unwrap(), None);
    assert!(budget.receipt().unwrap().work_reserved > spent);
    let cert = covered(
        state
            .finish(PreparedTreeOpenReason::ExpansionLimit)
            .unwrap(),
    );
    assert_eq!(cert.cover(None).unwrap().leaves(), 2);
}
#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Retain the original paid owner while a cloned attempt is refused."
)]
fn prepared_tree_cancelled_source_and_expired_inherited_clock_cannot_start_new_tree_claims() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let open = original_open(&budget, 2, false);
    let before = live(&memory);
    cancel.cancel();
    assert!(
        MarginRowBab::run_prepared(open.clone(), options(PreparedTreeSchedule::Serial), None)
            .is_err()
    );
    assert_eq!(live(&memory), before);
    drop(open);
    assert_eq!(live(&memory), 0);
    let (budget, memory, _) = resources();
    let open = original_open(&budget, 2, false);
    let before = live(&memory);
    assert!(MarginRowBab::run_prepared(
        open.clone(),
        options(PreparedTreeSchedule::Serial),
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    assert_eq!(live(&memory), before);
}
#[test]
fn prepared_tree_occupied_same_original_ledger_refuses_container_before_allocation_preserving_open()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let open = original_open(&budget, 2, false);
    let r = budget.receipt().unwrap();
    let remaining = CAP - r.live_persistent_bytes - r.live_scratch_bytes;
    let mut hold = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: remaining - 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    hold.mark_materialized().unwrap();
    let hold = hold.commit_persistent().unwrap();
    let before = live(&memory);
    assert!(
        MarginRowBab::run_prepared(open.clone(), options(PreparedTreeSchedule::Serial), None)
            .is_err()
    );
    assert_eq!(live(&memory), before);
    open.checkpoint(None).unwrap();
    drop(hold);
    drop(open);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_tree_successful_earlier_clock_survives_complete_cover_and_later_request() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let early = Instant::now() + Duration::from_secs(30);
    let cert = covered(
        MarginRowBab::run_prepared(
            original_open(&budget, 2, false),
            options(PreparedTreeSchedule::Serial),
            Some(early),
        )
        .unwrap(),
    );
    assert!(cert.deadline() <= early);
    for (evaluation, path) in cert.leaves() {
        assert!(evaluation.deadline(Some(clock())).unwrap() <= early);
        path.path(Some(clock())).unwrap();
    }
    cert.checkpoint(Some(clock())).unwrap();
    assert!(cert.cover.deadline() <= early);
}
#[test]
fn prepared_tree_cache_cannot_relabel_equal_numeric_foreign_original_root() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let mut state = PreparedTreeState::new(
        original_open(&budget, 2, false),
        options(PreparedTreeSchedule::Serial),
        None,
    )
    .unwrap();
    let other = PreparedDomain::from_open(original_open(&budget, 2, false), None).unwrap();
    let rows = build_prepared_domain_rows(&other, None).unwrap();
    assert!(state.cache_rows(&rows).is_err());
    assert_eq!(state.lru.entries.len(), 0);
    assert_eq!(state.ledger.recorded(), 0);
}

#[test]
fn prepared_tree_retained_cover_keeps_source_and_outcome_layout_bound() {
    let _env = crate::tests::lock_env_shared();
    assert!(size_of::<PreparedTreeCertificate>().abs_diff(size_of::<PreparedTreeOpen>()) <= 64);
    let (budget, memory, cancel) = resources();
    let cert = covered(
        MarginRowBab::run_prepared(
            original_open(&budget, 2, false),
            options(PreparedTreeSchedule::Serial),
            None,
        )
        .unwrap(),
    );
    assert_eq!(cert.cover(None).unwrap().classes(), [1]);
    assert_eq!(cert.cover(None).unwrap().leaves(), 2);
    cert.checkpoint(None).unwrap();
    let held = live(&memory);
    assert!(held > cert.cover.resident_bytes());
    eprintln!(
        "PREPARED_TREE_RETAINED_LAYOUT outcome={} covered={} open={} cover_handle={} cover_reserved={} live={}",
        size_of::<PreparedTreeOutcome>(), size_of::<PreparedTreeCertificate>(), size_of::<PreparedTreeOpen>(),
        size_of::<PaidTreeCover>(), cert.cover.resident_bytes(), held,
    );
    let spent = budget.receipt().unwrap().work_reserved;
    cancel.cancel();
    assert!(cert.checkpoint(None).is_err());
    assert_eq!(live(&memory), held);
    drop(cert);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().work_reserved, spent);
}

#[test]
fn prepared_tree_native_complete_header_observation() {
    let outcome = size_of::<PreparedTreeOutcome>();
    let covered = size_of::<PreparedTreeCertificate>();
    let open = size_of::<PreparedTreeOpen>();
    let state = size_of::<PreparedTreeState>();
    assert!(outcome <= META);
    assert!(covered <= META);
    assert!(open <= META);
    assert!(state <= META);
    eprintln!(
        "PREPARED_TREE_NATIVE_HEADERS outcome={} covered={} open={} state={} paid_cover={} entry={} cache_entry={} eval_seeded={} closed_leaf={} meta={}",
        outcome, covered, open, state, size_of::<PaidTreeCover>(),
        size_of::<PreparedDomainEntry>(), size_of::<CacheEntry>(),
        size_of::<PreparedEvalSeeded>(), size_of::<ClosedLeaf>(), META,
    );
}

#[test]
fn prepared_tree_debug_reports_distinct_live_claims_without_spending_or_detaching_owners() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let PreparedTreeOutcome::Open(open) = MarginRowBab::run_prepared(
        original_open(&budget, 2, false),
        PreparedTreeOptions {
            max_expansions: 0,
            ..options(PreparedTreeSchedule::Serial)
        },
        None,
    )
    .unwrap() else {
        panic!("zero expansions retain the real open root")
    };
    let ledger = open.state.ledger.resident_bytes().unwrap();
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let observed = format!("{open:?}");
    assert!(observed.contains(&format!("cover_ledger_reserved_bytes: Ok({ledger})")));
    let after = budget.receipt().unwrap();
    assert_eq!(after.work_reserved, before.work_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(live(&memory), bytes);
    open.checkpoint(None).unwrap();
    drop(open);
    assert_eq!(live(&memory), 0);
    let cert = covered(
        MarginRowBab::run_prepared(
            original_open(&budget, 2, false),
            options(PreparedTreeSchedule::Serial),
            None,
        )
        .unwrap(),
    );
    let ledger = cert.state.ledger.resident_bytes().unwrap();
    let cover = cert.cover.resident_bytes();
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let observed = format!("{cert:?}");
    assert!(observed.contains(&format!("cover_ledger_reserved_bytes: Ok({ledger})")));
    assert!(observed.contains(&format!("cover_reserved_bytes: {cover}")));
    let after = budget.receipt().unwrap();
    assert_eq!(after.work_reserved, before.work_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(live(&memory), bytes);
    cert.checkpoint(None).unwrap();
    drop(cert);
    assert_eq!(live(&memory), 0);
}
