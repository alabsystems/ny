// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::{
    residency::CancellationToken,
    work::{WorkLimits, WorkRefusal, WorkReserve},
};
use crate::decoder::{NextTokenOutcome, NextTokenProvenance};
use std::time::Duration;

fn ledger(memory: &MemoryAuthority, cancel: &CancellationToken, work: u64) -> WorkBudget {
    WorkBudget::new_in(
        memory,
        cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: 8 << 20,
            persistent_bytes: 8 << 20,
            max_work: work,
            max_proposals: 1024,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap()
}
fn options(budget: &WorkBudget) -> VocabOptions {
    VocabOptions {
        budget: Some(budget.clone()),
        max_vocab: 3,
        max_hidden: 2,
        max_persistent_bytes: 8 << 20,
        max_scratch_bytes: 8 << 20,
        max_terms: 16 << 20,
        ..VocabOptions::default()
    }
}
fn resident(memory: &MemoryAuthority) -> (ResidentAffineHead, WorkBudget, CancellationToken) {
    let cancel = CancellationToken::default();
    let budget = ledger(memory, &cancel, 16 << 20);
    let head = ResidentAffineHead::new(
        3,
        2,
        vec![0., 0., 1., -1., -2., 3.],
        vec![4., 0., 0.],
        HeadArithmetic::ExactReal,
        &options(&budget),
    )
    .unwrap();
    (head, budget, cancel)
}
fn hidden() -> HiddenBox<'static> {
    HiddenBox {
        lo: &[-0.125, -0.125],
        hi: &[0.125, 0.125],
    }
}

#[test]
fn resident_head_complete_query_wire_and_fresh_original_replay() {
    let memory = MemoryAuthority::new(64 << 20);
    let (head, _, _) = resident(&memory);
    let q1 = ledger(&memory, &CancellationToken::default(), 16 << 20);
    let ctx = head.bind_query(&q1).unwrap();
    let o = options(&q1);
    let checked = ctx
        .verify_dense_next_token(hidden(), &[0., 0.], 0, &o)
        .unwrap();
    assert!(checked.is_certified());
    assert_eq!(checked.provenance(), NextTokenProvenance::OriginalReplay);
    let NextTokenOutcome::Certified(proof) = &checked.outcome else {
        panic!("complete original cover")
    };
    assert_eq!(proof.covered_competitors(), 2);
    assert!(proof.maximum_competitor_upper().unwrap() < 0.);
    let wire = ctx.export_json(&checked.evidence, &o).unwrap();
    let q2 = ledger(&memory, &CancellationToken::default(), 16 << 20);
    let fresh = head.bind_query(&q2).unwrap();
    let replay = fresh
        .replay_next_token_json(hidden(), &[0., 0.], 0, wire.as_bytes(), &options(&q2))
        .unwrap();
    assert!(replay.is_certified());
    assert_eq!(replay.provenance(), NextTokenProvenance::OriginalReplay);
    assert!(!q1.shares_ledger(&q2));
    assert!(q1.shares_memory_authority(&q2));
    let mut bad = checked.evidence.clone();
    bad.groups.last_mut().unwrap().members.clear();
    assert!(fresh
        .replay_next_token(hidden(), &[0., 0.], 0, &bad, &options(&q2))
        .is_err());
}

#[test]
fn resident_head_separate_queries_cannot_replace_old_session_or_bound_ledger() {
    let memory = MemoryAuthority::new(64 << 20);
    let (head, model, _) = resident(&memory);
    let q = ledger(&memory, &CancellationToken::default(), 16 << 20);
    let ctx = head.bind_query(&q).unwrap();
    assert!(ctx.budget().shares_ledger(&q));
    let replacement = ledger(&memory, &CancellationToken::default(), 16 << 20);
    assert!(matches!(
        ctx.verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&replacement)),
        Err(DecoderRefusal::Resource(
            "shared head caller ledger mismatch"
        ))
    ));
    // The original session owner never receives a renewed work ledger.
    assert!(head.original.budget().shares_ledger(&model));
    assert!(matches!(
        head.original
            .verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&q)),
        Err(DecoderRefusal::Resource(
            "shared head caller ledger mismatch"
        ))
    ));
    assert!(ctx
        .verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&q))
        .unwrap()
        .is_certified());
}

#[test]
fn resident_head_cancelled_or_spent_query_never_refills_or_cancels_new_query() {
    let memory = MemoryAuthority::new(64 << 20);
    let (head, _, construction_cancel) = resident(&memory);
    construction_cancel.cancel();
    let cancel = CancellationToken::default();
    let q = ledger(&memory, &cancel, 16 << 20);
    let ctx = head.bind_query(&q).unwrap();
    cancel.cancel();
    assert!(matches!(
        ctx.verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&q)),
        Err(DecoderRefusal::Budget(WorkRefusal::Cancelled))
    ));
    assert!(head.bind_query(&q).is_err());
    let spent = ledger(&memory, &CancellationToken::default(), 1024);
    let spent_ctx = head.bind_query(&spent).unwrap();
    assert!(matches!(
        spent_ctx.verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&spent)),
        Err(DecoderRefusal::Budget(WorkRefusal::WorkLimit))
    ));
    assert!(head.bind_query(&spent).is_err());
    let fresh = ledger(&memory, &CancellationToken::default(), 16 << 20);
    assert!(head
        .bind_query(&fresh)
        .unwrap()
        .verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&fresh))
        .unwrap()
        .is_certified());
}

#[test]
fn resident_head_certificate_last_clone_retains_one_model_and_query_claim() {
    let memory = MemoryAuthority::new(64 << 20);
    let (head, model, _) = resident(&memory);
    let original = head.resident_bytes();
    let q = ledger(&memory, &CancellationToken::default(), 16 << 20);
    let ctx = head.bind_query(&q).unwrap();
    let report = ctx
        .verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&q))
        .unwrap();
    let NextTokenOutcome::Certified(proof) = &report.outcome else {
        panic!("strict original cover")
    };
    let kept = proof.clone();
    let last = kept.clone();
    let live = memory.receipt().unwrap().live_materialized_bytes;
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(model.receipt().unwrap().live_persistent_bytes, original);
    assert!(live > original + report.reserved_bytes());
    drop(report);
    drop(ctx);
    drop(head);
    drop(kept);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, live);
    assert_eq!(last.shared_head_bytes(), original);
    drop(last);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(q.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(model.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn resident_head_binding_capacity_is_counted_in_tight_local_options() {
    let memory = MemoryAuthority::new(64 << 20);
    let (head, _, _) = resident(&memory);
    let q = ledger(&memory, &CancellationToken::default(), 16 << 20);
    let ctx = head.bind_query(&q).unwrap();
    let checked = ctx
        .verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&q))
        .unwrap();
    let replay = ctx
        .replay_next_token(hidden(), &[0., 0.], 0, &checked.evidence, &options(&q))
        .unwrap();
    let mut tight = options(&q);
    tight.max_persistent_bytes = (head.resident_bytes() + replay.reserved_bytes()) as u64;
    drop(replay);
    assert!(matches!(
        ctx.replay_next_token(hidden(), &[0., 0.], 0, &checked.evidence, &tight),
        Err(DecoderRefusal::Resource(_))
    ));
}

#[test]
fn resident_head_requires_actual_shared_authority_and_explicit_construction() {
    let memory = MemoryAuthority::new(64 << 20);
    let (head, _, _) = resident(&memory);
    let other = MemoryAuthority::new(64 << 20);
    let foreign = ledger(&other, &CancellationToken::default(), 16 << 20);
    assert!(matches!(
        head.bind_query(&foreign),
        Err(DecoderRefusal::Resource(
            "resident query memory authority mismatch"
        ))
    ));
    let plain = WorkBudget::new(None, 8 << 20, 8 << 20, 16 << 20, 1024, Duration::ZERO).unwrap();
    assert!(head.bind_query(&plain).is_err());
    assert!(ResidentAffineHead::new(
        1,
        1,
        vec![1.],
        vec![0.],
        HeadArithmetic::ExactReal,
        &options(&plain)
    )
    .is_err());
    assert!(ResidentAffineHead::new(
        1,
        1,
        vec![1.],
        vec![0.],
        HeadArithmetic::ExactReal,
        &VocabOptions::default()
    )
    .is_err());
}

#[test]
fn resident_head_simultaneous_queries_share_global_claims_and_refused_ties_stay_unknown() {
    let memory = MemoryAuthority::new(64 << 20);
    let (head, _, _) = resident(&memory);
    std::thread::scope(|s| {
        let jobs: Vec<_> = (0..2)
            .map(|_| {
                let head = &head;
                let memory = &memory;
                s.spawn(move || {
                    let q = ledger(memory, &CancellationToken::default(), 16 << 20);
                    head.bind_query(&q)
                        .unwrap()
                        .verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&q))
                        .unwrap()
                })
            })
            .collect();
        let reports: Vec<_> = jobs.into_iter().map(|j| j.join().unwrap()).collect();
        assert!(reports.iter().all(NextTokenReport::is_certified));
        assert!(memory.receipt().unwrap().live_materialized_bytes > head.resident_bytes());
    });
    // An admitted complete original head with a strict tie supplies no proof.
    let q = ledger(&memory, &CancellationToken::default(), 16 << 20);
    let tie = ResidentAffineHead::new(
        3,
        2,
        vec![0.; 6],
        vec![0.; 3],
        HeadArithmetic::ExactReal,
        &options(&q),
    )
    .unwrap();
    let separate = ledger(&memory, &CancellationToken::default(), 16 << 20);
    assert!(!tie
        .bind_query(&separate)
        .unwrap()
        .verify_dense_next_token(hidden(), &[0., 0.], 0, &options(&separate))
        .unwrap()
        .is_certified());
    // Reproduce the original model's capacity exactly; binding must refuse.
    let exact = MemoryAuthority::new(head.resident_bytes() + 4096);
    let (full, _, _) = resident(&exact);
    let competing = ledger(&exact, &CancellationToken::default(), 16 << 20);
    let _live = competing
        .lease(crate::certify::work::WorkRequest {
            scratch_bytes: 4096,
            work: 1,
            ..Default::default()
        })
        .unwrap();
    let refused = ledger(&exact, &CancellationToken::default(), 16 << 20);
    assert!(matches!(
        full.bind_query(&refused),
        Err(DecoderRefusal::Budget(WorkRefusal::MemoryLimit))
    ));
    assert_eq!(refused.receipt().unwrap().work_reserved, 0);
}

#[test]
fn resident_head_normalized_complete_query_and_original_wire_replay_share_one_model() {
    use crate::decoder::normalized_head::{
        FinalNormalization, FinalNormalizationKind, NormalizedNextTokenOutcome,
    };
    let memory = MemoryAuthority::new(64 << 20);
    let (head, _, _) = resident(&memory);
    for kind in [
        FinalNormalizationKind::LayerNorm,
        FinalNormalizationKind::RmsNorm,
    ] {
        let norm = FinalNormalization {
            kind,
            gamma: &[1., 0.5],
            beta: Some(&[0.25, -0.25]),
            epsilon: 1e-5,
        };
        let pre = HiddenBox {
            lo: &[-1e-5; 2],
            hi: &[1e-5; 2],
        };
        let q = ledger(&memory, &CancellationToken::default(), 16 << 20);
        let ctx = head.bind_query(&q).unwrap();
        let checked = ctx
            .verify_normalized_next_token(norm, pre, 0, &options(&q))
            .unwrap();
        let NormalizedNextTokenOutcome::Certified(proof) = &checked.outcome else {
            panic!("original conditional norm/head cover")
        };
        assert_eq!(proof.covered_competitors(), 2);
        assert_eq!(proof.checked_stats().original_head_visits, 9);
        assert_eq!(proof.provenance(), NextTokenProvenance::CheckedQuery);
        let wire = ctx
            .export_normalized_json(&checked.evidence, &options(&q))
            .unwrap();
        let fresh = ledger(&memory, &CancellationToken::default(), 16 << 20);
        let replay_ctx = head.bind_query(&fresh).unwrap();
        let replay = replay_ctx
            .replay_normalized_json(norm, pre, 0, wire.as_bytes(), &options(&fresh))
            .unwrap();
        assert!(matches!(
            replay.report.outcome,
            NormalizedNextTokenOutcome::Certified(_)
        ));
        assert_eq!(
            replay.report.provenance(),
            NextTokenProvenance::OriginalReplay
        );
        assert_eq!(replay.report.stats.coverage_visits, 2);
        assert_eq!(
            replay.report.evidence.scope_digest,
            checked.evidence.scope_digest
        );
        let changed = FinalNormalization {
            gamma: &[1., 0.25],
            ..norm
        };
        assert!(replay_ctx
            .replay_normalized_json(changed, pre, 0, wire.as_bytes(), &options(&fresh))
            .is_err());
        let retained = proof.clone();
        let live = memory.receipt().unwrap().live_materialized_bytes;
        drop(checked);
        drop(ctx);
        assert_eq!(memory.receipt().unwrap().live_materialized_bytes, live);
        drop(retained);
    }
    assert_eq!(
        memory.receipt().unwrap().live_materialized_bytes,
        head.resident_bytes()
    );
}

#[test]
fn resident_head_normalized_local_and_wire_caps_include_separate_original_storage() {
    use crate::decoder::normalized_head::wire::normalized_wire_size_bound;
    use crate::decoder::normalized_head::{FinalNormalization, FinalNormalizationKind};
    let memory = MemoryAuthority::new(64 << 20);
    let (head, _, _) = resident(&memory);
    let q = ledger(&memory, &CancellationToken::default(), 16 << 20);
    let ctx = head.bind_query(&q).unwrap();
    let norm = FinalNormalization {
        kind: FinalNormalizationKind::LayerNorm,
        gamma: &[1., 1.],
        beta: None,
        epsilon: 1e-5,
    };
    let pre = HiddenBox {
        lo: &[0.; 2],
        hi: &[0.; 2],
    };
    let checked = ctx
        .verify_normalized_next_token(norm, pre, 0, &options(&q))
        .unwrap();
    let mut tight = options(&q);
    tight.max_persistent_bytes = (q.receipt().unwrap().live_persistent_bytes
        + normalized_wire_size_bound(3).unwrap()) as u64;
    assert!(
        ctx.export_normalized_json(&checked.evidence, &tight)
            .is_err(),
        "wire output and original separate-ledger owner coexist"
    );
    let query_bytes = checked.reserved_bytes();
    drop(checked);
    tight.max_persistent_bytes = (q.receipt().unwrap().live_persistent_bytes + query_bytes) as u64;
    assert!(
        ctx.verify_normalized_next_token(norm, pre, 0, &tight)
            .is_err(),
        "query-local cap includes original resident storage from construction ledger"
    );
    assert!(ctx
        .verify_normalized_next_token(norm, pre, 0, &options(&q))
        .is_ok());
}
