// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual original owners through terminal, ordinary RootEval and publication.
//! No control imports scalar bounds, mutable gates, RootEval or proof tokens.

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{PersistentReservation, WorkLimits, WorkReserve};
use crate::certify::{TokenScope, UnsatSource};
use crate::margin_row::spec::TwinOpSpec;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 128 << 20;

fn clock() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(2 * CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel)
}

fn live(memory: &MemoryAuthority) -> usize {
    let receipt = memory.receipt().unwrap();
    receipt.live_pending_bytes + receipt.live_materialized_bytes
}

fn fixture(fold: bool, positive: bool) -> TwinSpec {
    TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.25],
                bias: vec![-0.25],
                bias_err: vec![if fold { 1.0 / 256.0 } else { 0.0 }],
                weight_rel_err: if fold { 1.0 / 128.0 } else { 0.0 },
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
                weight: vec![0.5, -0.25],
                bias: vec![0.75, 0.5],
                bias_err: if fold { vec![1.0 / 512.0; 2] } else { vec![] },
                weight_rel_err: if fold { 1.0 / 256.0 } else { 0.0 },
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.75, 0.25, 0.5, -0.25, -0.25, 0.5],
                bias: vec![if positive { 2.0 } else { -2.0 }, 0.0, 0.125],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (3, 2),
            },
        ],
    }
}

fn session(spec: TwinSpec, budget: &WorkBudget, rows: bool) -> PreparedRootSession {
    PreparedRootSession::compile_declared_twin(
        spec,
        &[-1.0],
        &[1.0],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            keep_input_rows: rows,
            prefer_patch: false,
            ..PreparedRootProofOptions::default()
        },
        budget,
        Some(clock()),
    )
    .unwrap()
}

fn consumers() -> PreparedRootConsumerOptions {
    PreparedRootConsumerOptions {
        clip_rows: true,
        retain: Some(PreparedRootRetention {
            per_layer: 1,
            budget_bytes: 1 << 20,
        }),
        ..PreparedRootConsumerOptions::default()
    }
}

fn evaluation(session: &PreparedRootSession, classes: &[usize]) -> PreparedRootEvaluation {
    let adapter = project_prepared_root(
        session.source().clone(),
        session.policy(),
        BackwardIntermRuntimePolicy::default(),
        false,
        None,
        Some(session.deadline(None).unwrap()),
    )
    .unwrap();
    let rows = build_prepared_ordinary_rows(adapter, None).unwrap();
    prepared_root_eval_reuse(rows, 0, classes, None).unwrap()
}

fn r(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}

fn exact_margin(x: f64, bits: usize, class: usize, fold: bool, positive: bool) -> BigRational {
    let sign = |i| {
        if !fold {
            BigRational::zero()
        } else if bits & (1usize << i) == 0_usize {
            r(-1.0)
        } else {
            r(1.0)
        }
    };
    let conv = (r(1.25) * (r(1.0) + sign(0) * r(1.0 / 128.0)) * r(x)
        + r(-0.25)
        + sign(1) * r(1.0 / 256.0))
    .max(BigRational::zero());
    let y0 =
        (r(0.5) * (r(1.0) + sign(2) * r(1.0 / 256.0)) * &conv + r(0.75) + sign(3) * r(1.0 / 512.0))
            .max(BigRational::zero());
    let y1 =
        (r(-0.25) * (r(1.0) + sign(4) * r(1.0 / 256.0)) * conv + r(0.5) + sign(5) * r(1.0 / 512.0))
            .max(BigRational::zero());
    let target = r(if positive { 2.0 } else { -2.0 }) + r(0.75) * &y0 + r(0.25) * &y1;
    let other = match class {
        1 => r(0.5) * y0 - r(0.25) * y1,
        2 => r(-0.25) * y0 + r(0.5) * y1 + r(0.125),
        _ => panic!("authored original competitor"),
    };
    target - other
}

#[test]
fn prepared_proof_ordinary_closed_terminal_keeps_cheap_route_and_original_order() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(false, true), &budget, false);
    let PreparedRootOutcome::Proven(proof) = session.try_prove_root(0, &[2, 1], None).unwrap()
    else {
        panic!("positive original root");
    };
    assert_eq!(
        proof.attempt_receipt(),
        PreparedRootAttemptReceipt {
            reference: session.reference_receipt(),
            terminal_closed: true,
            ..PreparedRootAttemptReceipt::default()
        }
    );
    assert!(proof.publication.ordinary().is_none());
    assert!(proof.publication.receipt().ordinary.is_none());
    assert_eq!(proof.classes(), [2, 1]);
    assert_eq!(proof.publication.token().classes(), [1, 2]);
    assert!(proof.session().source().same_source(session.source()));
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop((proof, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_proof_ordinary_requested_products_execute_real_pack_and_selected_original_rational_fold_corners(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(true, true), &budget, true);
    let PreparedRootOutcome::Proven(proof) = session
        .try_prove_root_with_consumers(0, &[2, 1], consumers(), None)
        .unwrap()
    else {
        panic!("positive genuine ordinary root");
    };
    assert!(
        proof.attempt_receipt().ordinary_attempted && proof.attempt_receipt().ordinary_completed
    );
    assert!(!proof.attempt_receipt().ordinary_refused);
    let ordinary = proof.publication.ordinary().unwrap();
    let adapter = ordinary.adapter();
    assert!(adapter.source().same_source(session.source()));
    assert!(adapter.clip_rows().is_some() && adapter.retained_rows().is_some());
    assert_eq!(ordinary.receipt().classes, 2);
    assert!(ordinary.receipt().tiles > 0);
    assert_eq!(
        proof.publication.receipt().ordinary,
        Some(ordinary.receipt())
    );
    let rows = ordinary.rows(None).unwrap();
    assert_eq!(rows.al(None).unwrap().a.dim(), (1, 2));
    assert_eq!(rows.au(None).unwrap().e.as_ref().unwrap().dim(), (1, 2));
    assert_eq!(proof.publication.token().source(), UnsatSource::Authorized);
    assert_eq!(proof.publication.token().scope(), TokenScope::Instance);
    assert!(proof.publication.receipt().containment_checks > 0);
    let bounds = ordinary.bounds(None).unwrap();
    let mut minimum = f64::INFINITY;
    for (column, &class) in proof.classes().iter().enumerate() {
        let selected = proof.publication.terminal().lower()[column].max(bounds[column]);
        minimum = minimum.min(selected);
        for bits in 0..64 {
            for x in [-1.0, -0.25, 0.0, 0.2, 1.0] {
                let exact = exact_margin(x, bits, class, true, true);
                assert!(exact > BigRational::zero());
                assert!(r(selected) <= exact);
            }
        }
    }
    assert_eq!(proof.min_margin().to_bits(), minimum.to_bits());
    proof.validate_objective(0, &[2, 1], None).unwrap();
    assert!(proof.validate_objective(0, &[1, 2], None).is_err());
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(rows);
    drop((proof, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_proof_ordinary_unclosed_terminal_attempts_real_root_and_keeps_open_search_owners() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(false, false), &budget, false);
    let PreparedRootOutcome::Open(open) = session.try_prove_root(0, &[2, 1], None).unwrap() else {
        panic!("true negative margin stays open");
    };
    assert!(!open.receipt().terminal_closed);
    assert!(open.receipt().ordinary_attempted && open.receipt().ordinary_completed);
    assert!(!open.receipt().ordinary_refused);
    assert_eq!(open.classes(), [2, 1]);
    assert!(open.session().source().same_source(session.source()));
    assert!(open
        .ordinary_evaluation()
        .unwrap()
        .adapter()
        .source()
        .same_source(session.source()));
    assert!(open
        .ordinary_evaluation()
        .unwrap()
        .bounds(None)
        .unwrap()
        .iter()
        .any(|&b| b <= 0.0));
    assert!(open.terminal().lower().iter().any(|&b| b <= 0.0));
    open.checkpoint(None).unwrap();
    let alias = open.clone();
    let held = live(&memory);
    drop((open, session));
    assert_eq!(live(&memory), held);
    alias.checkpoint(None).unwrap();
    drop(alias);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_proof_ordinary_resource_refusal_preserves_complete_open_source_on_same_occupied_ledger()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(false, false), &budget, false);
    let mut blocker: Option<PersistentReservation> = None;
    let mut observed = false;
    let mut occupy = || {
        observed = true;
        let r = budget.receipt().unwrap();
        // Occupy this test's unchanged, separately declared persistent
        // allowance. The production source-size ceiling is not this ledger's
        // memory limit; scratch also has its own independently priced ceiling.
        let bytes = CAP - r.live_persistent_bytes;
        let mut lease = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: bytes,
                    work: 1,
                    ..WorkRequest::default()
                },
            )
            .unwrap();
        lease.mark_materialized().unwrap();
        blocker = Some(lease.commit_persistent().unwrap());
    };
    let PreparedRootOutcome::Open(open) = session
        .try_prove_inner(
            0,
            &[2, 1],
            PreparedRootConsumerOptions::default(),
            None,
            Some(&mut occupy),
        )
        .unwrap()
    else {
        panic!("occupied original authority remains open");
    };
    assert!(observed);
    assert!(open.receipt().ordinary_attempted && open.receipt().ordinary_refused);
    assert!(!open.receipt().ordinary_completed);
    assert!(open.ordinary_evaluation().is_none());
    assert!(open.session().budget().shares_ledger(&budget));
    open.checkpoint(None).unwrap();
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(blocker);
    drop((open, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_proof_ordinary_cancel_after_actual_terminal_publishes_no_root_owner_or_memo() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let session = session(fixture(false, true), &budget, true);
    let before = live(&memory);
    let mut observed = false;
    let mut stop = || {
        observed = true;
        cancel.cancel();
    };
    assert!(session
        .try_prove_inner(0, &[2, 1], consumers(), None, Some(&mut stop))
        .is_err());
    assert!(observed);
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert!(session.checkpoint(None).is_err());
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise the inherited deadline through a cloned open-session owner."
)]
fn prepared_proof_ordinary_open_session_and_aliases_preserve_earliest_producer_clock() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(false, false), &budget, false);
    let early = Instant::now() + Duration::from_secs(30);
    let PreparedRootOutcome::Open(open) = session.try_prove_root(0, &[2, 1], Some(early)).unwrap()
    else {
        panic!("open original root");
    };
    assert!(open.deadline(None).unwrap() <= early);
    assert!(open.session().deadline(None).unwrap() <= early);
    assert!(
        open.clone()
            .ordinary_evaluation()
            .unwrap()
            .deadline(None)
            .unwrap()
            <= early
    );
    let alias = open.session().restricted(Some(clock())).unwrap();
    assert!(alias.deadline(None).unwrap() <= early);
    assert!(matches!(
        open.checkpoint(Some(Instant::now())),
        Err(ny_core::NyError::DeadlineExceeded(_))
    ));
}

#[test]
fn prepared_proof_ordinary_foreign_compiler_ledger_cannot_create_session_even_with_equal_limits() {
    let _env = crate::tests::lock_env_shared();
    let (first, first_memory, _) = resources();
    let (other, other_memory, _) = resources();
    let model = OriginalModel::compile(fixture(false, true), &first, Some(clock())).unwrap();
    assert!(PreparedRootSession::from_compiled_model(
        model,
        super::super::ir::OriginalInputDomain {
            lower: &[-1.0],
            upper: &[1.0],
        },
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions::default(),
        &other,
        Some(clock())
    )
    .is_err());
    assert_eq!(live(&first_memory), 0);
    assert_eq!(live(&other_memory), 0);
}

#[test]
fn prepared_proof_ordinary_foreign_actual_evaluation_cannot_authorize_equal_original_bytes() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let first = session(fixture(false, true), &budget, false);
    let second = session(fixture(false, true), &budget, false);
    assert_eq!(first.source_digest(), second.source_digest());
    assert!(!first.source().same_source(second.source()));
    let terminal =
        build_prepared_terminal(first.source().construction(), 0, &[2, 1], None).unwrap();
    let evaluated = evaluation(&second, &[2, 1]);
    assert!(authorize_prepared_evaluation(
        first.source().clone(),
        terminal,
        evaluated,
        0,
        first.policy(),
        None
    )
    .is_err());
    first.checkpoint(None).unwrap();
    second.checkpoint(None).unwrap();
    drop((first, second));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_proof_ordinary_changed_original_class_order_target_and_partial_coverage_refuse() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(false, true), &budget, false);
    let terminal =
        build_prepared_terminal(session.source().construction(), 0, &[2, 1], None).unwrap();
    let evaluated = evaluation(&session, &[1, 2]);
    assert!(authorize_prepared_evaluation(
        session.source().clone(),
        terminal,
        evaluated,
        0,
        session.policy(),
        None
    )
    .is_err());
    let terminal =
        build_prepared_terminal(session.source().construction(), 0, &[2, 1], None).unwrap();
    let evaluated = evaluation(&session, &[2, 1]);
    assert!(authorize_prepared_evaluation(
        session.source().clone(),
        terminal,
        evaluated,
        1,
        session.policy(),
        None
    )
    .is_err());
    for classes in [&[1][..], &[1, 1][..], &[0, 1][..], &[1, usize::MAX][..]] {
        assert!(session.try_prove_root(0, classes, None).is_err());
    }
    session.checkpoint(None).unwrap();
}

#[test]
fn prepared_proof_ordinary_requested_missing_rows_or_enabled_bi_cannot_hide_behind_terminal_positive(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(false, true), &budget, false);
    assert!(session
        .try_prove_root_with_consumers(0, &[2, 1], consumers(), None)
        .is_err());
    let requested = PreparedRootConsumerOptions {
        backward_interm: BackwardIntermRuntimePolicy {
            enabled: true,
            ..BackwardIntermRuntimePolicy::default()
        },
        ..PreparedRootConsumerOptions::default()
    };
    assert!(session
        .try_prove_root_with_consumers(0, &[2, 1], requested, None)
        .is_err());
    let PreparedRootOutcome::Proven(proof) = session.try_prove_root(0, &[2, 1], None).unwrap()
    else {
        panic!("unchanged cheap source stays useful");
    };
    assert!(proof.publication.ordinary().is_none());
}

#[test]
fn prepared_proof_ordinary_open_numerical_bounds_cannot_enter_original_instance_authorizer() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let session = session(fixture(false, false), &budget, false);
    let terminal =
        build_prepared_terminal(session.source().construction(), 0, &[2, 1], None).unwrap();
    let evaluated = evaluation(&session, &[2, 1]);
    assert!(terminal.lower().iter().all(|&b| b < 0.0));
    assert!(evaluated.bounds(None).unwrap().iter().all(|&b| b < 0.0));
    assert!(authorize_prepared_evaluation(
        session.source().clone(),
        terminal,
        evaluated,
        0,
        session.policy(),
        None
    )
    .is_err());
    for x in [-1.0, 0.0, 1.0] {
        assert!(exact_margin(x, 0, 1, false, false) < BigRational::zero());
    }
}

#[test]
fn prepared_proof_ordinary_publication_and_session_clones_retain_actual_pack_until_last_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(false, true), &budget, true);
    let PreparedRootOutcome::Proven(proof) = session
        .try_prove_root_with_consumers(0, &[2, 1], consumers(), None)
        .unwrap()
    else {
        panic!("positive owned ordinary proof");
    };
    let alias = proof.clone();
    let shared = proof.session().clone();
    let rows = proof.publication.ordinary().unwrap().rows(None).unwrap();
    let held = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    drop((proof, session));
    assert_eq!(live(&memory), held);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    alias.checkpoint(None).unwrap();
    drop(alias);
    assert!(live(&memory) > shared.source().resident_bytes());
    rows.evaluation().checkpoint(None).unwrap();
    drop(rows);
    assert!(live(&memory) > 0);
    shared.checkpoint(None).unwrap();
    drop(shared);
    assert_eq!(live(&memory), 0);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn prepared_proof_ordinary_successful_ratification_retains_earlier_clock_in_all_returned_owners() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(false, true), &budget, true);
    let PreparedRootOutcome::Proven(proof) = session
        .try_prove_root_with_consumers(0, &[2, 1], consumers(), None)
        .unwrap()
    else {
        panic!("positive actual ordinary proof");
    };
    let original = proof.clone();
    let early = Instant::now() + Duration::from_secs(30);
    let kept = proof.ratified(&GuardReport::new(), Some(early)).unwrap();
    assert!(kept.deadline(None).unwrap() <= early);
    assert!(kept.clone().deadline(None).unwrap() <= early);
    assert!(kept.session().deadline(None).unwrap() <= early);
    assert!(kept.publication.deadline(None).unwrap() <= early);
    // Restriction is per alias; another previously owned proof keeps its own
    // original phase deadline and cannot renew the ratified alias.
    assert!(original.deadline(None).unwrap() > early);
    kept.checkpoint(None).unwrap();
    drop((original, kept, session));
    assert_eq!(live(&memory), 0);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "checkpoint a proof alias while retaining the original for foreign-owner refusal controls"
)]
fn prepared_proof_checkpoint_rejects_equal_bytes_foreign_session_owners_before_spending() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let original = session(fixture(false, true), &budget, false);
    let PreparedRootOutcome::Proven(proof) = original.try_prove_root(0, &[2, 1], None).unwrap()
    else {
        panic!("the real positive original must publish");
    };
    proof.checkpoint(None).unwrap();
    proof.clone().checkpoint(None).unwrap();
    let (other_budget, other_memory, _) = resources();
    let same_ledger = session(fixture(false, true), &budget, false);
    let other_ledger = session(fixture(false, true), &other_budget, false);
    for foreign in [&same_ledger, &other_ledger] {
        foreign.checkpoint(None).unwrap();
        assert_eq!(foreign.source_digest(), original.source_digest());
        assert_eq!(foreign.compiled_digest(), original.compiled_digest());
        assert_eq!(foreign.policy(), proof.policy());
        assert!(!foreign.source().same_source(original.source()));
        let mut forged = proof.clone();
        forged.session = foreign.clone();
        let before = budget.receipt().unwrap();
        let other_before = other_budget.receipt().unwrap();
        let held = live(&memory);
        let other_held = live(&other_memory);
        let error = forged.checkpoint(None).unwrap_err().to_string();
        assert!(error.contains("foreign original session source"));
        assert_eq!(budget.receipt().unwrap(), before);
        assert_eq!(other_budget.receipt().unwrap(), other_before);
        assert_eq!(live(&memory), held);
        assert_eq!(live(&other_memory), other_held);
        drop(forged);
        proof.checkpoint(None).unwrap();
    }
    drop((proof, original, same_ledger, other_ledger));
    assert_eq!(live(&memory), 0);
    assert_eq!(live(&other_memory), 0);
}

#[test]
fn prepared_proof_checkpoint_rejects_changed_session_precision_with_authentic_publication() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let original = session(fixture(false, true), &budget, false);
    let PreparedRootOutcome::Proven(proof) = original.try_prove_root(0, &[2, 1], None).unwrap()
    else {
        panic!("the real positive original must publish");
    };
    assert!(!proof.policy().root_f32 && !proof.policy().tableau_f32);
    for tableau in [false, true] {
        let mut forged = proof.clone();
        if tableau {
            forged.session.policy.tableau_f32 = true;
        } else {
            forged.session.policy.root_f32 = true;
        }
        assert!(forged.session.source().same_source(original.source()));
        assert_eq!(forged.publication.receipt().policy, proof.policy());
        let held = live(&memory);
        let error = forged.checkpoint(None).unwrap_err().to_string();
        assert!(error.contains("different frozen precision policy"));
        assert_eq!(live(&memory), held);
        drop(forged);
        proof.checkpoint(None).unwrap();
    }
    drop((proof, original));
    assert_eq!(live(&memory), 0);
}

fn storage_observation_fixture() -> TwinSpec {
    let mut spec = fixture(false, true);
    spec.n_in = 5;
    spec.ops[0] = TwinOpSpec::Conv {
        input: 0,
        weight: vec![1.0 / 3.0, -1.0 / 7.0, 2.0 / 9.0],
        bias: vec![-1.0 / 11.0],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 3),
        stride: (1, 1),
        pads: (0, 1, 0, 1),
        ishape: (1, 1, 5),
        oshape: (1, 1, 5),
    };
    spec.ops[3] = TwinOpSpec::Gemm {
        input: 3,
        weight: vec![0.5; 10],
        bias: vec![0.75, 0.5],
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (2, 5),
    };
    spec
}

#[test]
fn prepared_row_storage_observations_follow_actual_dense_and_patch_payloads_without_owning_them() {
    let _env = crate::tests::lock_env_shared();
    for prefer_patch in [false, true] {
        let (budget, memory, _) = resources();
        let session = PreparedRootSession::compile_declared_twin(
            storage_observation_fixture(),
            &[-1.0; 5],
            &[1.0; 5],
            RootPrecisionRuntimePolicy::default(),
            PreparedRootProofOptions {
                keep_input_rows: true,
                row_f32: true,
                prefer_patch,
                dense_tile: 1,
                patch_tile: 1,
                patch_ratio: 1.0,
                ..PreparedRootProofOptions::default()
            },
            &budget,
            Some(clock()),
        )
        .unwrap();
        let before = budget.receipt().unwrap();
        let observed = session.row_storage_receipt();
        assert_eq!(budget.receipt().unwrap(), before);
        let source = session.source().construction();
        let actual = source
            .layers()
            .iter()
            .flat_map(|layer| layer.tiles())
            .map(|tile| tile.rows().owned_payload_bytes().unwrap())
            .sum::<usize>();
        assert_eq!(observed.payload_bytes_after_total, actual);
        assert!(observed.tiles > 1);
        assert!(observed.coefficient_cells_stored > 0 && observed.bias_cells_stored > 0);
        assert!(observed.nominal_cells_changed > 0);
        assert!(
            observed.nominal_cells_changed
                <= observed.coefficient_cells_stored + observed.bias_cells_stored
        );
        assert!(observed.payload_bytes_before_total > observed.payload_bytes_after_total);
        assert_eq!(
            observed.nominal_allocation_bytes_total,
            (observed.coefficient_cells_stored + observed.bias_cells_stored) * size_of::<f32>()
        );
        assert!(observed.max_admitted_peak_bytes > 0);
        assert!(
            observed.max_admitted_peak_bytes <= crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP
        );
        if prefer_patch {
            assert!(source.stats().patch_tiles > 0);
        } else {
            assert!(source.stats().dense_tiles > 0);
        }
        let PreparedRootOutcome::Proven(proof) = session.try_prove_root(0, &[2, 1], None).unwrap()
        else {
            panic!("positive actual storage fixture");
        };
        assert_eq!(proof.session().row_storage_receipt(), observed);
        drop(proof);
        drop(session);
        assert_eq!(live(&memory), 0);
        // A copied historical value retains no source, arrays or byte claim.
        assert!(observed.tiles > 0);
    }
}

#[test]
fn prepared_row_storage_observations_stay_zero_for_binary32_reference_and_unstored_rows() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = PreparedRootSession::compile_declared_twin(
        fixture(false, true),
        &[-1.0],
        &[1.0],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            reference_arithmetic: super::super::PreparedReferenceArithmetic::Binary32Certified,
            ..PreparedRootProofOptions::default()
        },
        &budget,
        Some(clock()),
    )
    .unwrap();
    assert!(session.reference_receipt().binary32_conv_dot_ops > 0);
    assert_eq!(
        session.row_storage_receipt(),
        PreparedRowStorageReceipt::default()
    );
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_row_storage_observation_does_not_renew_expired_source_or_admit_invalid_delivery() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = PreparedRootSession::compile_declared_twin(
        fixture(false, true),
        &[-1.0],
        &[1.0],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            row_f32: true,
            ..PreparedRootProofOptions::default()
        },
        &budget,
        Some(clock()),
    );
    assert!(session.is_err());
    assert_eq!(live(&memory), 0);
    let session = self::session(fixture(false, true), &budget, false);
    let mut expired = session.clone();
    expired.deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an expired deadline");
    let before = budget.receipt().unwrap();
    assert_eq!(
        expired.row_storage_receipt(),
        PreparedRowStorageReceipt::default()
    );
    assert_eq!(budget.receipt().unwrap(), before);
    assert!(expired.try_prove_root(0, &[2, 1], None).is_err());
    drop(expired);
    drop(session);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_ordinary_projection_handoff_preserves_original_unique_payload_and_earliest_clock() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(false, false), &budget, false);
    let adapter = project_prepared_root(
        session.source().clone(),
        session.policy(),
        BackwardIntermRuntimePolicy::default(),
        false,
        None,
        Some(clock()),
    )
    .unwrap();
    let early = Instant::now() + Duration::from_secs(30);
    let rows = build_prepared_ordinary_rows(adapter.clone(), Some(early)).unwrap();
    let pointers = rows.storage_pointers();
    let result = evaluate_requested_projection(&adapter, rows, 0, &[2, 1], Some(clock())).unwrap();
    assert!(!result.receipt().cached_pack);
    assert!(!result.receipt().shared_payload_copy);
    assert!(result.deadline(None).unwrap() <= early);
    assert_eq!(result.adversaries(), [2, 1]);
    let moved = result.rows(None).unwrap();
    let lower = moved.al(None).unwrap();
    let upper = moved.au(None).unwrap();
    assert_eq!(lower.a.as_ptr() as usize, pointers[0]);
    assert_eq!(lower.e.as_ref().unwrap().as_ptr() as usize, pointers[1]);
    assert_eq!(lower.b.as_ptr() as usize, pointers[2]);
    assert_eq!(lower.eb.as_ptr() as usize, pointers[3]);
    assert_eq!(upper.a.as_ptr() as usize, pointers[4]);
    assert_eq!(upper.e.as_ref().unwrap().as_ptr() as usize, pointers[5]);
    assert_eq!(upper.b.as_ptr() as usize, pointers[6]);
    assert_eq!(upper.eb.as_ptr() as usize, pointers[7]);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop((moved, result, adapter, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_ordinary_projection_handoff_refuses_same_source_with_foreign_product_projection() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(false, false), &budget, true);
    let expected = project_prepared_root(
        session.source().clone(),
        session.policy(),
        BackwardIntermRuntimePolicy::default(),
        true,
        None,
        Some(clock()),
    )
    .unwrap();
    let foreign = project_prepared_root(
        session.source().clone(),
        session.policy(),
        BackwardIntermRuntimePolicy::default(),
        false,
        None,
        Some(clock()),
    )
    .unwrap();
    assert!(expected.source().same_source(foreign.source()));
    assert!(!expected.same_owner(&foreign));
    assert!(expected.clip_rows().is_some());
    assert!(foreign.clip_rows().is_none());
    let rows = build_prepared_ordinary_rows(foreign.clone(), None).unwrap();
    let row_claim = rows.resident_bytes();
    let before = budget.receipt().unwrap();
    let held = live(&memory);
    let peak = memory.receipt().unwrap().peak_leased_bytes;
    let error = evaluate_requested_projection(&expected, rows, 0, &[2, 1], None)
        .err()
        .expect("foreign product projection must refuse")
        .to_string();
    assert!(error.contains("ordinary memo foreign original projection"));
    let after = budget.receipt().unwrap();
    // Only the paid fixed binding attempt ran; no evaluation or payload copy
    // was admitted. The consumed foreign rows released their original claim.
    assert_eq!(after.admitted, before.admitted + 1);
    assert_eq!(after.work_reserved, before.work_reserved + 512);
    assert_eq!(
        after.live_persistent_bytes,
        before.live_persistent_bytes - row_claim
    );
    assert_eq!(live(&memory), held - row_claim);
    assert_eq!(memory.receipt().unwrap().peak_leased_bytes, peak);
    expected.checkpoint(None).unwrap();
    foreign.checkpoint(None).unwrap();
    drop((expected, foreign, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_ordinary_projection_handoff_refuses_expired_and_one_credit_short_binding_work() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let session = session(fixture(false, false), &budget, false);
    let adapter = project_prepared_root(
        session.source().clone(),
        session.policy(),
        BackwardIntermRuntimePolicy::default(),
        false,
        None,
        Some(clock()),
    )
    .unwrap();
    let rows = build_prepared_ordinary_rows(adapter.clone(), None).unwrap();
    let before = budget.receipt().unwrap();
    let held = live(&memory);
    assert!(evaluate_requested_projection(
        &adapter,
        rows.clone(),
        0,
        &[2, 1],
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .expect("test clock supports an expired deadline")
        )
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap(), before);
    assert_eq!(live(&memory), held);
    // Occupy the unchanged real test ledger's remaining work, leaving exactly
    // one credit less than the explicitly quoted 512-credit binding attempt.
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 100_000_000_000_000 - before.work_reserved - 511,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let occupied = budget.receipt().unwrap();
    assert!(evaluate_requested_projection(&adapter, rows.clone(), 0, &[2, 1], None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, occupied.admitted);
    assert_eq!(live(&memory), held);
    rows.checkpoint(None).unwrap();
    drop((blocker, rows, adapter, session));
    assert_eq!(live(&memory), 0);
}
