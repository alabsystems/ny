// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Public flows use genuine completed declared-source sessions. No test mints
//! row arrays, source seals, proof tokens or phase authority.

use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, WorkBudget, WorkLimits, WorkRefusal, WorkRequest, WorkReserve,
};
use crate::margin_row::clip::Halfspace;
use crate::margin_row::{
    PreparedInputSplit, PreparedInputSplitSide, PreparedRootProofOptions, PreparedRootSession,
    RootPrecisionRuntimePolicy, TwinOpSpec, TwinSpec,
};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::{Duration, Instant};

const CAP: usize = 128 << 20;

fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

fn resources(lane_bytes: usize) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(lane_bytes.checked_mul(2).unwrap());
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: lane_bytes,
            persistent_bytes: lane_bytes,
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

fn conv(input: usize, bias: f64) -> TwinOpSpec {
    TwinOpSpec::Conv {
        input,
        weight: vec![1.0],
        bias: vec![bias],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 2),
        oshape: (1, 1, 2),
    }
}

fn fixture(bias: f64) -> TwinSpec {
    TwinSpec {
        n_in: 2,
        ops: vec![
            conv(0, bias),
            TwinOpSpec::Relu { input: 1 },
            conv(2, -0.1),
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.25; 2],
                bias: vec![0.5],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 2),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![0.5, -0.25],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    }
}

fn session(budget: &WorkBudget, keep_input_rows: bool, bias: f64) -> PreparedRootSession {
    PreparedRootSession::compile_declared_twin(
        fixture(bias),
        &[-1.0; 2],
        &[1.0; 2],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            keep_input_rows,
            prefer_patch: false,
            dense_tile: 1,
            ..PreparedRootProofOptions::default()
        },
        budget,
        Some(at()),
    )
    .unwrap()
}

fn request(side: PreparedInputSplitSide) -> PreparedInputSplit {
    PreparedInputSplit {
        layer: 0,
        target: 0,
        side,
    }
}

fn r(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}

fn eval_halfspace(h: &Halfspace, x: &[f64]) -> BigRational {
    h.a.iter()
        .zip(x)
        .fold(r(h.d), |sum, (a, x)| sum + r(*a) * r(*x))
}

#[test]
fn prepared_public_input_rows_filter_and_necessary_halfspaces_keep_original_ids() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP);
    let source = session(&budget, true, -0.25);
    let rows = source.retained_input_rows(None).unwrap();
    rows.validate_session(&source, None).unwrap();
    let ids: Vec<_> = rows.ids(None).unwrap().collect();
    assert_eq!(ids, [(0, 0), (0, 1), (1, 0), (1, 1)]);
    assert_eq!(rows.len(), ids.len());
    assert!(!rows.is_empty());
    let filtered = rows.filter(&[(0, 0), (1, 1)], None).unwrap();
    assert!(!rows.same_owner(&filtered));
    assert_eq!(
        filtered.ids(None).unwrap().collect::<Vec<_>>(),
        [(0, 0), (1, 1)]
    );
    assert!(rows.filter(&[(0, 0), (0, 0)], None).is_err());
    assert!(rows.filter(&[(1, 1), (0, 0)], None).is_err());
    assert!(rows.filter(&[(0, 2)], None).is_err());
    let missing = [PreparedInputSplit {
        layer: 0,
        target: 1,
        side: PreparedInputSplitSide::Active,
    }];
    assert!(filtered.necessary_halfspaces(&missing, None).is_err());
    let requests = [
        request(PreparedInputSplitSide::Inactive),
        request(PreparedInputSplitSide::Active),
    ];
    let batch = filtered.necessary_halfspaces(&requests, None).unwrap();
    batch.validate_session(&source, None).unwrap();
    assert_eq!(batch.splits(None).unwrap(), requests);
    assert_eq!(
        batch.selection_fingerprint(),
        filtered.content_fingerprint()
    );
    let reversed = filtered
        .necessary_halfspaces(&[requests[1], requests[0]], None)
        .unwrap();
    assert_ne!(batch.content_fingerprint(), reversed.content_fingerprint());
    let hs = batch.halfspaces(None).unwrap();
    assert_eq!(hs.len(), 2);
    assert!(hs.iter().all(|h| h.a.len() == 2 && h.d.is_finite()));
    assert!(batch
        .charged_errors(None)
        .unwrap()
        .iter()
        .all(|error| error.is_finite() && *error >= 0.0));
    // The first original preactivation is x[0] - 1/4. These exact rational
    // checks use only points satisfying each declared closed phase.
    for x in [-1.0, 0.0, 0.25] {
        assert!(eval_halfspace(&hs[0], &[x, 0.0]) <= BigRational::zero());
    }
    for x in [0.25, 0.5, 1.0] {
        assert!(eval_halfspace(&hs[1], &[x, 0.0]) <= BigRational::zero());
    }
}

#[test]
fn prepared_public_input_row_batch_clones_keep_last_owner_and_refuse_foreign_sessions() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP);
    let first = session(&budget, true, -0.25);
    let second = session(&budget, true, -0.25);
    let rows = first.retained_input_rows(None).unwrap();
    let other = second.retained_input_rows(None).unwrap();
    assert_eq!(rows.content_fingerprint(), other.content_fingerprint());
    assert!(!first.same_source(&second));
    assert!(rows.validate_session(&second, None).is_err());
    let filtered = rows.filter(&[(0, 0)], None).unwrap();
    let batch = filtered
        .necessary_halfspaces(&[request(PreparedInputSplitSide::Active)], None)
        .unwrap();
    assert!(batch.validate_session(&second, None).is_err());
    let (foreign_budget, foreign_memory, _) = resources(CAP);
    let foreign = session(&foreign_budget, true, -0.25);
    assert_eq!(first.source_digest(), foreign.source_digest());
    assert!(!budget.shares_ledger(&foreign_budget));
    assert!(rows.validate_session(&foreign, None).is_err());
    assert!(batch.validate_session(&foreign, None).is_err());
    drop(foreign);
    assert_eq!(live(&foreign_memory), 0);
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    let rows_alias = rows.clone();
    let batch_alias = batch.clone();
    assert!(rows.same_owner(&rows_alias));
    assert!(batch.same_owner(&batch_alias));
    assert_eq!(
        batch.halfspaces(None).unwrap().as_ptr(),
        batch_alias.halfspaces(None).unwrap().as_ptr()
    );
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    drop((first, second, other, rows, rows_alias, filtered));
    batch.checkpoint(None).unwrap();
    assert_eq!(batch.halfspaces(None).unwrap().len(), 1);
    assert!(live(&memory) > batch.resident_bytes());
    drop(batch);
    assert!(live(&memory) > 0);
    drop(batch_alias);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn prepared_public_input_rows_preserve_restricted_deadlines_and_refuse_expiry() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources(CAP);
    let source = session(&budget, true, -0.25);
    let early = source
        .deadline(None)
        .unwrap()
        .checked_sub(Duration::from_secs(10))
        .expect("earlier test deadline");
    let restricted = source.restricted(Some(early)).unwrap();
    let rows = restricted
        .retained_input_rows(Some(early + Duration::from_secs(1)))
        .unwrap();
    assert_eq!(rows.deadline(None).unwrap(), early);
    let filtered = rows
        .filter(&[(0, 0)], Some(early + Duration::from_secs(2)))
        .unwrap();
    assert_eq!(filtered.deadline(None).unwrap(), early);
    let earliest = early
        .checked_sub(Duration::from_secs(1))
        .expect("earliest test deadline");
    let requests = [request(PreparedInputSplitSide::Inactive)];
    let batch = filtered
        .necessary_halfspaces(&requests, Some(earliest))
        .unwrap();
    assert_eq!(batch.deadline(Some(early)).unwrap(), earliest);
    let expired = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("expired test deadline"),
    );
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    assert!(restricted.retained_input_rows(expired).is_err());
    assert!(rows.ids(expired).is_err());
    assert!(rows.filter(&[(0, 0)], expired).is_err());
    assert!(rows.necessary_halfspaces(&requests, expired).is_err());
    assert!(batch.splits(expired).is_err());
    assert!(batch.halfspaces(expired).is_err());
    assert!(batch.charged_errors(expired).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(live(&memory), before);
    cancel.cancel();
    assert!(rows.ids(None).is_err());
    assert!(rows.filter(&[(0, 0)], None).is_err());
    assert!(rows.necessary_halfspaces(&requests, None).is_err());
    assert!(batch.splits(None).is_err());
    assert!(batch.halfspaces(None).is_err());
    assert!(batch.charged_errors(None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(live(&memory), before);
}

#[test]
fn prepared_public_input_rows_missing_retention_refuses_even_with_no_unstable_rows() {
    let _env = crate::tests::lock_env_shared();
    assert!(!PreparedRootProofOptions::default().keep_input_rows);
    assert!(!PreparedRootProofOptions::default().row_f32);
    let (budget, _, _) = resources(CAP);
    let missing = session(&budget, false, -100.0);
    assert!(missing.retained_input_rows(None).is_err());
    let delivered = session(&budget, true, -100.0);
    let rows = delivered.retained_input_rows(None).unwrap();
    assert!(rows.is_empty());
    assert_eq!(rows.ids(None).unwrap().count(), 0);
}

#[test]
fn prepared_public_input_row_filter_refuses_one_byte_below_observed_metadata_claim() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP);
    let source = session(&budget, true, -0.25);
    let rows = source.retained_input_rows(None).unwrap();
    // Observe a genuine admitted reservation under the generous unchanged
    // limits. This does not claim exact-cap acceptance or constant owner sizes.
    let reference = rows.filter(&[(0, 0)], None).unwrap();
    let claim = reference.resident_bytes();
    assert!(claim > 1);
    drop(reference);
    let receipt = budget.receipt().unwrap();
    let existing = receipt.live_persistent_bytes + receipt.live_scratch_bytes;
    let ceiling = existing.checked_add(claim - 1).unwrap();
    assert_eq!(budget.tighten_total_byte_limit(ceiling).unwrap(), ceiling);
    let before = live(&memory);
    let peak = memory.receipt().unwrap().peak_leased_bytes;
    assert!(rows.filter(&[(0, 0)], None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, receipt.admitted);
    assert_eq!(live(&memory), before);
    assert_eq!(memory.receipt().unwrap().peak_leased_bytes, peak);
    rows.checkpoint(None).unwrap();
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Keep explicit shared ledger aliases to verify clones cannot widen the inherited root byte ceiling."
)]
fn prepared_public_input_rows_inherit_existing_root_ceiling_on_shared_permissive_ledger() {
    let _env = crate::tests::lock_env_shared();
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let (budget, memory, _) = resources(cap.checked_add(CAP).unwrap());
    let source = session(&budget, true, -0.25);
    let receipt = budget.receipt().unwrap();
    let existing = receipt.live_persistent_bytes + receipt.live_scratch_bytes;
    assert!(existing < cap);
    // Construction has already restricted the initially permissive ledger.
    // A clone cannot lift that ceiling, even before the new public operation.
    assert_eq!(
        budget.clone().tighten_total_byte_limit(cap + CAP).unwrap(),
        cap
    );
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    assert!(matches!(
        budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: cap - existing + 1,
                ..WorkRequest::default()
            }
        ),
        Err(WorkRefusal::MemoryLimit)
    ));
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(live(&memory), before);
    let rows = source.retained_input_rows(None).unwrap();
    let shared = budget.clone();
    let receipt = shared.receipt().unwrap();
    let existing = receipt.live_persistent_bytes + receipt.live_scratch_bytes;
    assert!(matches!(
        shared.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: cap - existing + 1,
                ..WorkRequest::default()
            }
        ),
        Err(WorkRefusal::MemoryLimit)
    ));
    rows.checkpoint(None).unwrap();
}
