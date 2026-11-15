// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Genuine public sessions and original producer controls, without imported
//! row planes, supplied owner claims, callbacks or proof-token construction.

use super::{PreparedHeadPreactivationRow, OWNER_WORK};
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, WorkBudget, WorkLimits, WorkRefusal, WorkRequest, WorkReserve,
};
use crate::margin_row::{
    BackwardIntermRuntimePolicy, PreparedRootCompilation, PreparedRootProofOptions,
    PreparedRootSession, RootPrecisionRuntimePolicy, TwinOpSpec, TwinSpec,
};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::{Duration, Instant};

const CAP: usize = 128 << 20;
const MAX_WORK: u64 = 100_000_000_000_000;

fn at() -> Instant {
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
            max_work: MAX_WORK,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel)
}

fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}

fn fixture(fold: bool) -> TwinSpec {
    let rho = if fold { 1.0 / 256.0 } else { 0.0 };
    let error = if fold { 1.0 / 512.0 } else { 0.0 };
    TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![-0.25],
                bias_err: vec![error],
                weight_rel_err: rho,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 2),
                oshape: (1, 1, 2),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![0.5, -0.25, -1.0, 0.75, 0.0, 0.0],
                bias: vec![0.125, -0.5, 2.0],
                bias_err: if fold { vec![error; 3] } else { vec![] },
                weight_rel_err: rho,
                shape: (3, 2),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.25; 6],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 3),
            },
        ],
    }
}

fn compilation(budget: &WorkBudget, fold: bool) -> PreparedRootCompilation {
    PreparedRootCompilation::compile_declared_twin(fixture(fold), budget, Some(at())).unwrap()
}

fn domain(
    compiled: PreparedRootCompilation,
    budget: &WorkBudget,
    width: f64,
) -> PreparedRootSession {
    compiled
        .prepare_declared_domain(
            &[-width; 2],
            &[width; 2],
            RootPrecisionRuntimePolicy::default(),
            PreparedRootProofOptions {
                prefer_patch: false,
                dense_tile: 1,
                ..PreparedRootProofOptions::default()
            },
            budget,
            None,
        )
        .unwrap()
}

fn session(budget: &WorkBudget, fold: bool) -> PreparedRootSession {
    domain(compilation(budget, fold), budget, 1.0)
}

fn r(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

fn line(view: &PreparedHeadPreactivationRow<'_>, x: [f64; 2]) -> (BigRational, BigRational) {
    let mut nominal = r(view.bias());
    let mut error = r(view.bias_error());
    assert!(view.bias().is_finite());
    assert!(view.bias_error().is_finite() && view.bias_error() >= 0.0);
    for (expected, (i, a, e)) in view.entries().enumerate() {
        assert_eq!(i, expected);
        assert!(a.is_finite() && e.is_finite() && e >= 0.0);
        nominal += r(a) * r(x[i]);
        error += r(e) * r(x[i].abs());
    }
    (nominal, error)
}

fn exact(x: [f64; 2], output: usize, signs: usize, fold: bool) -> BigRational {
    let sign = |i: usize| r(if signs & (1 << i) == 0 { -1.0 } else { 1.0 });
    let rho = r(if fold { 1.0 / 256.0 } else { 0.0 });
    let eps = r(if fold { 1.0 / 512.0 } else { 0.0 });
    let h = x.map(|x| {
        ((r(1.0) + sign(0) * &rho) * r(x) - r(0.25) + sign(1) * &eps).max(BigRational::zero())
    });
    let w = [[0.5, -0.25], [-1.0, 0.75], [0.0, 0.0]][output];
    r(w[0]) * (r(1.0) + sign(2) * &rho) * &h[0]
        + r(w[1]) * (r(1.0) + sign(3) * &rho) * &h[1]
        + r([0.125, -0.5, 2.0][output])
        + sign(4) * &eps
}

#[test]
fn prepared_head_rows_original_coordinates_constants_and_fold_family_have_rational_support() {
    let _env = crate::tests::lock_env_shared();
    assert!(!PreparedRootProofOptions::default().keep_input_rows);
    assert!(!PreparedRootProofOptions::default().row_f32);
    for fold in [false, true] {
        let (budget, _, _) = resources();
        let source = session(&budget, fold);
        let rows = source.head_preactivation_rows(None).unwrap();
        rows.validate_session(&source, None).unwrap();
        assert_eq!(rows.n_in(), 2);
        assert_eq!(rows.columns(), 3);
        assert_eq!(rows.first_head_operation(), 3);
        // The model has only two final output classes. Column 2 is still a
        // genuine first-head preactivation with its own independent constant.
        assert_eq!(
            rows.declared_domain(None).unwrap(),
            (&[-1.0; 2][..], &[1.0; 2][..])
        );
        let family_corners = if fold { 32 } else { 1 };
        for x0 in [-1.0, 0.0, 1.0] {
            for x1 in [-1.0, 0.0, 1.0] {
                for signs in 0..family_corners {
                    for j in 0..3 {
                        let lower = rows.lower(j, None).unwrap();
                        let upper = rows.upper(j, None).unwrap();
                        assert_eq!(lower.original_output(), j);
                        assert_eq!(upper.original_output(), j);
                        assert_eq!(lower.entries().count(), 2);
                        assert_eq!(upper.entries().count(), 2);
                        let y = exact([x0, x1], j, signs, fold);
                        let (lv, le) = line(&lower, [x0, x1]);
                        let (uv, ue) = line(&upper, [x0, x1]);
                        assert!(lv - le <= y && y <= uv + ue);
                    }
                }
            }
        }
        assert!(rows.lower(3, None).is_err());
        assert!(rows.upper(usize::MAX, None).is_err());
        assert!(rows.lower(2, None).unwrap().bias().abs() > 1.0);
        assert!(rows.upper(2, None).unwrap().bias().abs() > 1.0);
    }
}

#[test]
fn prepared_head_row_views_are_bit_identical_to_original_directional_views() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget, true);
    let rows = source.head_preactivation_rows(None).unwrap();
    let adapter = super::project_prepared_root(
        source.source().clone(),
        RootPrecisionRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
        false,
        None,
        None,
    )
    .unwrap();
    let original = super::build_prepared_ordinary_rows(adapter, None).unwrap();
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    for j in 0..rows.columns() {
        let public = [rows.lower(j, None).unwrap(), rows.upper(j, None).unwrap()];
        let direct = [
            original.lower(j, None).unwrap(),
            original.upper(j, None).unwrap(),
        ];
        for (public, direct) in public.iter().zip(&direct) {
            assert_eq!(public.original_output(), direct.original_output());
            assert_eq!(public.bias().to_bits(), direct.bias().to_bits());
            assert_eq!(public.bias_error().to_bits(), direct.bias_error().to_bits());
            let public_bits: Vec<_> = public
                .entries()
                .map(|(i, a, e)| (i, a.to_bits(), e.to_bits()))
                .collect();
            let direct_bits: Vec<_> = direct
                .entries()
                .map(|(i, a, e)| (i, a.to_bits(), e.to_bits()))
                .collect();
            assert_eq!(public_bits, direct_bits);
        }
    }
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.work_reserved, before.work_reserved);
    assert_eq!(live(&memory), bytes);
}

#[test]
fn prepared_head_rows_require_exact_source_for_same_model_different_domains() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let compiled = compilation(&budget, false);
    let first = domain(compiled.clone(), &budget, 1.0);
    let other_domain = domain(compiled, &budget, 0.5);
    assert_eq!(first.compiled_digest(), other_domain.compiled_digest());
    assert!(!first.same_source(&other_domain));
    let rows = first.head_preactivation_rows(None).unwrap();
    let same_source = first.restricted(None).unwrap();
    rows.validate_session(&same_source, None).unwrap();
    assert!(rows.validate_session(&other_domain, None).is_err());
    let same_content = session(&budget, false);
    assert_eq!(first.compiled_digest(), same_content.compiled_digest());
    assert_eq!(first.declared_lower(), same_content.declared_lower());
    assert!(rows.validate_session(&same_content, None).is_err());
    let (foreign_budget, _, _) = resources();
    let foreign = session(&foreign_budget, false);
    assert_eq!(first.compiled_digest(), foreign.compiled_digest());
    assert!(!budget.shares_ledger(&foreign_budget));
    assert!(rows.validate_session(&foreign, None).is_err());
}

#[test]
fn prepared_head_rows_clones_share_paid_payload_and_keep_last_source_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    let rows = source.head_preactivation_rows(None).unwrap();
    let independent = source.head_preactivation_rows(None).unwrap();
    assert!(!rows.same_owner(&independent));
    rows.validate_session(&source, None).unwrap();
    independent.validate_session(&source, None).unwrap();
    drop(independent);
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    let work = budget.receipt().unwrap().work_reserved;
    let alias = rows.clone();
    assert!(rows.same_owner(&alias));
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(budget.receipt().unwrap().work_reserved, work);
    drop(source);
    // Numeric rows retain their original domain/source, not merely a model.
    assert_eq!(live(&memory), before);
    drop(rows);
    {
        let snapshot = alias.upper(2, None).unwrap();
        assert_eq!(snapshot.original_output(), 2);
        assert_eq!(snapshot.entries().count(), 2);
        assert!(live(&memory) > 0);
    }
    drop(alias);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn prepared_head_rows_preserve_earliest_clocks_and_shared_cancellation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let source = session(&budget, false);
    let session_at = source
        .deadline(None)
        .unwrap()
        .checked_sub(Duration::from_secs(10))
        .expect("test clock supports an earlier deadline");
    let restricted = source.restricted(Some(session_at)).unwrap();
    let rows = restricted
        .head_preactivation_rows(Some(session_at + Duration::from_secs(1)))
        .unwrap();
    assert_eq!(rows.deadline(None).unwrap(), session_at);
    assert_eq!(
        rows.deadline(Some(session_at + Duration::from_secs(2)))
            .unwrap(),
        session_at
    );
    let requested = session_at
        .checked_sub(Duration::from_secs(1))
        .expect("test clock supports an earlier deadline");
    let earlier = source.head_preactivation_rows(Some(requested)).unwrap();
    assert_eq!(earlier.deadline(None).unwrap(), requested);
    let expired = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("test clock supports an expired deadline"),
    );
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    assert!(source.head_preactivation_rows(expired).is_err());
    assert!(rows.lower(1, expired).is_err());
    assert!(rows.upper(1, expired).is_err());
    assert!(rows.declared_domain(expired).is_err());
    assert!(rows.validate_session(&source, expired).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(live(&memory), before);
    cancel.cancel();
    assert!(source.head_preactivation_rows(None).is_err());
    assert!(rows.lower(1, None).is_err());
    assert!(rows.upper(1, None).is_err());
    assert!(rows.declared_domain(None).is_err());
    assert!(rows.validate_session(&source, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(live(&memory), before);
}

#[test]
fn prepared_head_rows_reader_has_explicit_rows_only_projection_and_keeps_actual_precision() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let source = session(&budget, false);
    let rows = source.head_preactivation_rows(None).unwrap();
    let adapter = rows.0.rows.adapter();
    assert!(adapter.source().same_source(source.source()));
    assert_eq!(
        adapter.receipt().precision,
        source.source().construction().prefix().root().policy()
    );
    assert!(!adapter.receipt().clipping_requested);
    assert!(!adapter.receipt().retained_requested);
    assert!(!adapter.receipt().backward_interm.enabled);
    assert!(adapter.clip_rows().is_none());
    assert!(adapter.retained_rows().is_none());
    rows.validate_session(&source, None).unwrap();
    source.checkpoint(None).unwrap();
}

#[test]
fn prepared_head_row_header_refuses_occupied_capacity_before_projection() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    // This is a genuine completed source on its capped original ledger.
    // A smaller combined ceiling stays below the per-persistent limit so
    // this control specifically reaches the combined MemoryLimit guard.
    let cap = budget.tighten_total_byte_limit(CAP / 2).unwrap();
    let original = budget.receipt().unwrap();
    let header = super::owner_bytes().unwrap();
    let occupied = cap
        .checked_sub(original.live_persistent_bytes + original.live_scratch_bytes)
        .and_then(|remaining| remaining.checked_sub(header - 1))
        .unwrap();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: occupied,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let error = source.head_preactivation_rows(None).err().unwrap();
    assert!(error.to_string().contains("MemoryLimit"));
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.work_reserved, before.work_reserved);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), bytes);
    drop(blocker);
    let rows = source.head_preactivation_rows(None).unwrap();
    assert_eq!(rows.columns(), 3);
}

#[test]
fn prepared_head_row_admission_failure_drops_projection_and_header_without_refill() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    // Observe the real unchanged projection's admission cost on this exact
    // source, then keep only enough remaining work to repeat that projection
    // plus the new wrapper. The row builder must spend the same exhausted
    // ledger, not a fresh allowance or a memory-only retry.
    let before_projection = budget.receipt().unwrap();
    let projection = super::project_prepared_root(
        source.source().clone(),
        RootPrecisionRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
        false,
        None,
        None,
    )
    .unwrap();
    let after_projection = budget.receipt().unwrap();
    let projection_work = after_projection.work_reserved - before_projection.work_reserved;
    let projection_admissions = after_projection.admitted - before_projection.admitted;
    assert!(projection_work > 0 && projection_admissions > 0);
    drop(projection);
    let bytes = live(&memory);
    let before = budget.receipt().unwrap();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: MAX_WORK - before.work_reserved - OWNER_WORK - projection_work,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let occupied = budget.receipt().unwrap();
    let error = source.head_preactivation_rows(None).err().unwrap();
    assert!(error.to_string().contains("WorkLimit"));
    let refused = budget.receipt().unwrap();
    assert_eq!(
        refused.admitted,
        occupied.admitted + 1 + projection_admissions
    );
    assert_eq!(refused.work_reserved, MAX_WORK);
    assert_eq!(refused.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(refused.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), bytes);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    drop(blocker);
    assert!(matches!(
        budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 1,
                ..WorkRequest::default()
            }
        ),
        Err(WorkRefusal::WorkLimit)
    ));
    assert_eq!(budget.receipt().unwrap().work_reserved, MAX_WORK);
    source.checkpoint(None).unwrap();
}
