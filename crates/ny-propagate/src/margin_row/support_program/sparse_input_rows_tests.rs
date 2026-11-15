// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Authentic public sparse requests; existing producers are independent controls.
//! No supplied rows, fabricated source claims, timing hooks or FP-state mutation.

use super::{halfspace_owner_bytes, row_owner_bytes, row_owner_work};
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, WorkBudget, WorkLimits, WorkRefusal, WorkRequest, WorkReserve,
};
use crate::margin_row::{
    PreparedInputSplitSide, PreparedRootCompilation, PreparedRootProofOptions,
    PreparedRootRetention, PreparedRootSession, PreparedRowStorageReceipt,
    PreparedSparseInputHalfspace, PreparedSparseInputRow, PreparedSparseInputRows,
    RootPrecisionRuntimePolicy, TwinOpSpec, TwinSpec,
};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::{Duration, Instant};

const CAP: usize = 128 << 20;
const MAX_WORK: u64 = 100_000_000_000_000;
const LOWER: [f64; 4] = [-1.0, 0.5, -1.0, -1.0];
const UPPER: [f64; 4] = [1.0, 1.0, 0.0, 1.0];

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

fn conv(input: usize, weight: f64, bias: f64, fold: bool) -> TwinOpSpec {
    TwinOpSpec::Conv {
        input,
        weight: vec![weight],
        bias: vec![bias],
        bias_err: vec![if fold { 1.0 / 512.0 } else { 0.0 }],
        weight_rel_err: if fold { 1.0 / 256.0 } else { 0.0 },
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 4),
        oshape: (1, 1, 4),
    }
}

fn fixture(fold: bool) -> TwinSpec {
    TwinSpec {
        n_in: 4,
        ops: vec![
            conv(0, 1.0, -0.25, fold),
            TwinOpSpec::Relu { input: 1 },
            conv(2, 2.0, -0.125, fold),
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.25; 8],
                bias: vec![0.5; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 4),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![0.5, -0.25, -0.5, 0.25],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    }
}

fn compile(budget: &WorkBudget, fold: bool) -> PreparedRootCompilation {
    PreparedRootCompilation::compile_declared_twin(fixture(fold), budget, Some(at())).unwrap()
}

fn prepare(
    compiled: PreparedRootCompilation,
    budget: &WorkBudget,
    lower: &[f64],
    upper: &[f64],
    keep: bool,
    row_f32: bool,
) -> PreparedRootSession {
    compiled
        .prepare_declared_domain(
            lower,
            upper,
            RootPrecisionRuntimePolicy::default(),
            PreparedRootProofOptions {
                keep_input_rows: keep,
                row_f32,
                prefer_patch: false,
                dense_tile: 2,
                ..PreparedRootProofOptions::default()
            },
            budget,
            None,
        )
        .unwrap()
}

fn session(budget: &WorkBudget, fold: bool) -> PreparedRootSession {
    prepare(compile(budget, fold), budget, &LOWER, &UPPER, false, false)
}

fn bits(row: &PreparedSparseInputRow<'_>) -> (u64, u64, Vec<(usize, u64, u64)>) {
    (
        row.bias().to_bits(),
        row.bias_error().to_bits(),
        row.entries()
            .map(|(i, a, e)| (i, a.to_bits(), e.to_bits()))
            .collect(),
    )
}

fn q(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}

fn halfspace_value(h: &PreparedSparseInputHalfspace<'_>, x: &[f64]) -> BigRational {
    h.coefficients(None)
        .unwrap()
        .iter()
        .zip(x)
        .fold(q(h.constant(None).unwrap()), |sum, (a, x)| {
            sum + q(*a) * q(*x)
        })
}

#[test]
fn prepared_sparse_input_rows_keep_unsorted_stable_ids_and_match_paid_original_planes() {
    let _env = crate::tests::lock_env_shared();
    assert!(ny_core::has_f64_interval_proof_environment());
    let (budget, memory, _) = resources();
    let source = session(&budget, true);
    let requested = [3, 1, 2, 0];
    let gate = source.source().construction().prefix().gate(0).unwrap();
    assert!(gate.l[1] >= 0.0 && gate.u[2] <= 0.0);
    let mut different_constants = false;
    for layer in 0..2 {
        let rows: PreparedSparseInputRows =
            source.sparse_input_rows(layer, &requested, None).unwrap();
        rows.validate_session(&source, None).unwrap();
        assert_eq!(rows.original_layer(), layer);
        assert_eq!(rows.n_in(), 4);
        assert_eq!(rows.columns(), 4);
        assert_eq!(rows.targets(None).unwrap(), requested);
        assert_eq!(
            rows.declared_domain(None).unwrap(),
            (&LOWER[..], &UPPER[..])
        );
        let product = source
            .source()
            .construction()
            .layer_products(layer)
            .unwrap();
        let direct =
            super::run_prepared_prefix_input_rows(product.prefix_before(), &requested, None)
                .unwrap();
        for (column, &target) in requested.iter().enumerate() {
            let lower = rows.lower(column, None).unwrap();
            let upper = rows.upper(column, None).unwrap();
            assert_eq!(lower.original_target(), target);
            assert_eq!(upper.original_target(), target);
            for (public, direct) in [
                (&lower, direct.lower(column).unwrap()),
                (&upper, direct.upper(column).unwrap()),
            ] {
                assert_eq!(public.bias().to_bits(), direct.bias().to_bits());
                assert_eq!(public.bias_error().to_bits(), direct.bias_error().to_bits());
                let original: Vec<_> = direct
                    .entries()
                    .map(|(i, a, e)| (i, a.to_bits(), e.to_bits()))
                    .collect();
                assert_eq!(bits(public).2, original);
                assert_eq!(public.entries().count(), 4);
                assert!(public.bias_error() >= 0.0);
                assert!(public
                    .entries()
                    .all(|(_, a, e)| a.is_finite() && e.is_finite() && e >= 0.0));
            }
            if layer == 1 {
                different_constants |= lower.bias().to_bits() != upper.bias().to_bits();
            }
        }
        let before = budget.receipt().unwrap();
        let bytes = live(&memory);
        rows.targets(None).unwrap();
        rows.lower(1, None).unwrap();
        rows.upper(2, None).unwrap();
        assert!(rows.lower(4, None).is_err());
        assert!(rows.upper(usize::MAX, None).is_err());
        let after = budget.receipt().unwrap();
        assert_eq!(after.admitted, before.admitted);
        assert_eq!(after.work_reserved, before.work_reserved);
        assert_eq!(live(&memory), bytes);
    }
    assert!(
        different_constants,
        "independent directional constants must not be averaged"
    );
}

#[test]
fn prepared_sparse_halfspaces_match_original_helper_bits_and_closed_phase_boundary() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let source = session(&budget, false);
    let rows = source.sparse_input_rows(0, &[3, 1, 2, 0], None).unwrap();
    for side in [
        PreparedInputSplitSide::Inactive,
        PreparedInputSplitSide::Active,
    ] {
        let public = rows.necessary_halfspace(3, side, None).unwrap();
        let direct = super::input_halfspace_prepared(
            &rows.0.rows,
            rows.prefix().unwrap(),
            rows.0.op,
            rows.0.rows.targets(),
            3,
            side,
            rows.deadline(None).unwrap(),
        )
        .unwrap();
        public.validate_session(&source, None).unwrap();
        assert_eq!(public.original_layer(), 0);
        assert_eq!(public.original_target(), 0);
        assert_eq!(public.side(), side);
        assert_eq!(
            public
                .coefficients(None)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            direct
                .halfspace()
                .a
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            public.constant(None).unwrap().to_bits(),
            direct.halfspace().d.to_bits()
        );
        assert_eq!(
            public.charged_error(None).unwrap().to_bits(),
            direct.charged_error().to_bits()
        );
        let points: &[f64] = match side {
            PreparedInputSplitSide::Inactive => &[-1.0, 0.0, 0.25],
            PreparedInputSplitSide::Active => &[0.25, 0.5, 1.0],
        };
        for &x0 in points {
            assert!(halfspace_value(&public, &[x0, 0.75, -0.5, 0.0]) <= BigRational::zero());
        }
    }
    assert!(rows
        .necessary_halfspace(4, PreparedInputSplitSide::Active, None)
        .is_err());
}

#[test]
fn prepared_sparse_delivery_keeps_missing_retention_and_historical_storage_receipts_exact() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let source = session(&budget, false);
    let before = source.row_storage_receipt();
    assert_eq!(before, PreparedRowStorageReceipt::default());
    assert!(source.retained_input_rows(None).is_err());
    assert!(source
        .ranked_row_advice(
            PreparedRootRetention {
                per_layer: 1,
                budget_bytes: CAP
            },
            None
        )
        .is_err());
    let rows = source.sparse_input_rows(0, &[1, 2], None).unwrap();
    assert_eq!(source.row_storage_receipt(), before);
    assert!(source.retained_input_rows(None).is_err());
    assert!(source
        .ranked_row_advice(
            PreparedRootRetention {
                per_layer: 1,
                budget_bytes: CAP
            },
            None
        )
        .is_err());
    assert_eq!(
        rows.0.rows.backing(),
        crate::margin_row::input_rows::InputRowsBacking::F64
    );
    let historical = prepare(compile(&budget, true), &budget, &LOWER, &UPPER, true, true);
    let receipt = historical.row_storage_receipt();
    assert!(receipt.tiles > 0 && receipt.coefficient_cells_stored > 0);
    let fresh = historical.sparse_input_rows(0, &[3, 1], None).unwrap();
    assert_eq!(
        fresh.0.rows.backing(),
        crate::margin_row::input_rows::InputRowsBacking::F64
    );
    assert_eq!(historical.row_storage_receipt(), receipt);
}

#[test]
fn prepared_sparse_invalid_public_targets_release_new_claims_without_importing_trust() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    let baseline = live(&memory);
    for (layer, targets) in [
        (0, &[][..]),
        (0, &[3, 3][..]),
        (0, &[3, 4][..]),
        (0, &[0; 65][..]),
        (2, &[0][..]),
        (usize::MAX, &[0][..]),
    ] {
        assert!(source.sparse_input_rows(layer, targets, None).is_err());
        assert_eq!(live(&memory), baseline);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
        assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    }
    source.sparse_input_rows(0, &[2, 1, 3], None).unwrap();
}

#[test]
fn prepared_sparse_rows_and_halfspaces_refuse_equal_content_and_foreign_domains() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let compiled = compile(&budget, false);
    let source = prepare(compiled.clone(), &budget, &LOWER, &UPPER, false, false);
    let narrower = prepare(
        compiled,
        &budget,
        &[-0.5, 0.5, -0.5, -0.5],
        &UPPER,
        false,
        false,
    );
    let equal = session(&budget, false);
    let (foreign_budget, _, _) = resources();
    let foreign = session(&foreign_budget, false);
    assert_eq!(source.compiled_digest(), narrower.compiled_digest());
    assert_eq!(source.compiled_digest(), equal.compiled_digest());
    let rows = source.sparse_input_rows(0, &[3, 1], None).unwrap();
    let halfspace = rows
        .necessary_halfspace(0, PreparedInputSplitSide::Active, None)
        .unwrap();
    let alias = source.restricted(None).unwrap();
    rows.validate_session(&alias, None).unwrap();
    halfspace.validate_session(&alias, None).unwrap();
    for other in [&narrower, &equal, &foreign] {
        assert!(!source.same_source(other));
        assert!(rows.validate_session(other, None).is_err());
        assert!(halfspace.validate_session(other, None).is_err());
    }
}

#[test]
fn prepared_sparse_rows_and_halfspaces_keep_earliest_clocks_live_fp_and_cancellation() {
    let _env = crate::tests::lock_env_shared();
    assert!(ny_core::has_f64_interval_proof_environment());
    let (budget, memory, cancel) = resources();
    let source = session(&budget, false);
    let session_at = source
        .deadline(None)
        .unwrap()
        .checked_sub(Duration::from_secs(10))
        .expect("earlier test deadline");
    let restricted = source.restricted(Some(session_at)).unwrap();
    let rows = restricted
        .sparse_input_rows(0, &[3, 1], Some(session_at + Duration::from_secs(1)))
        .unwrap();
    assert_eq!(rows.deadline(None).unwrap(), session_at);
    let half_at = session_at
        .checked_sub(Duration::from_secs(1))
        .expect("earliest test deadline");
    let halfspace = rows
        .necessary_halfspace(0, PreparedInputSplitSide::Active, Some(half_at))
        .unwrap();
    assert_eq!(halfspace.deadline(Some(session_at)).unwrap(), half_at);
    rows.checkpoint(None).unwrap();
    halfspace.checkpoint(None).unwrap();
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let expired = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("expired test deadline"),
    );
    assert!(source.sparse_input_rows(0, &[0], expired).is_err());
    assert!(rows.targets(expired).is_err());
    assert!(rows.lower(0, expired).is_err());
    assert!(rows.upper(0, expired).is_err());
    assert!(rows.declared_domain(expired).is_err());
    assert!(rows
        .necessary_halfspace(0, PreparedInputSplitSide::Active, expired)
        .is_err());
    assert!(halfspace.coefficients(expired).is_err());
    assert!(halfspace.constant(expired).is_err());
    assert!(halfspace.charged_error(expired).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(live(&memory), bytes);
    cancel.cancel();
    assert!(source.sparse_input_rows(0, &[0], None).is_err());
    assert!(rows.lower(0, None).is_err());
    assert!(rows.upper(0, None).is_err());
    assert!(rows.targets(None).is_err());
    assert!(rows
        .necessary_halfspace(0, PreparedInputSplitSide::Active, None)
        .is_err());
    assert!(halfspace.coefficients(None).is_err());
    assert!(halfspace.constant(None).is_err());
    assert!(halfspace.charged_error(None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(live(&memory), bytes);
}

#[test]
fn prepared_sparse_current_instance_clock_is_checked_in_an_isolated_child() {
    const MARKER: &str = "__sparse_input_rows_global_clock_child__";
    const NAME: &str = "margin_row::support_program::sparse_input_rows::tests::prepared_sparse_current_instance_clock_is_checked_in_an_isolated_child";
    if !std::env::args().any(|arg| arg == MARKER) {
        // The process-global instance clock has unrelated parallel readers.
        // Reexecute only this exact control; --skip is an inert libtest marker,
        // not a new production/environment hook or a skipped real test.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                NAME,
                "--test-threads=1",
                "--color=never",
                "--skip",
                MARKER,
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("1 passed; 0 failed"), "{stdout}");
        assert!(
            stdout.contains(
                "prepared_sparse_current_instance_clock_is_checked_in_an_isolated_child ... ok"
            ),
            "{stdout}"
        );
        return;
    }
    struct Restore(Option<Instant>);
    impl Drop for Restore {
        fn drop(&mut self) {
            match self.0 {
                Some(at) => ny_core::instance_budget::publish(at),
                None => ny_core::instance_budget::clear(),
            }
        }
    }
    let _restore = Restore(ny_core::instance_budget::deadline());
    ny_core::instance_budget::clear();
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    let original = source.deadline(None).unwrap();
    let rows = source.sparse_input_rows(0, &[3, 1], None).unwrap();
    let halfspace = rows
        .necessary_halfspace(0, PreparedInputSplitSide::Active, None)
        .unwrap();
    let earlier = original
        .checked_sub(Duration::from_secs(10))
        .expect("earlier test deadline");
    ny_core::instance_budget::publish(earlier);
    assert_eq!(rows.deadline(None).unwrap(), earlier);
    assert_eq!(halfspace.deadline(None).unwrap(), earlier);
    let captured = source.sparse_input_rows(0, &[0], None).unwrap();
    ny_core::instance_budget::publish(original);
    assert_eq!(captured.deadline(None).unwrap(), earlier);
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    ny_core::instance_budget::publish(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .expect("expired test deadline"),
    );
    assert!(source.sparse_input_rows(0, &[0], None).is_err());
    assert!(rows.targets(None).is_err());
    assert!(rows.lower(0, None).is_err());
    assert!(rows.upper(0, None).is_err());
    assert!(rows
        .necessary_halfspace(0, PreparedInputSplitSide::Active, None)
        .is_err());
    assert!(halfspace.coefficients(None).is_err());
    assert!(halfspace.constant(None).is_err());
    assert!(halfspace.charged_error(None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(live(&memory), bytes);
}

#[test]
fn prepared_sparse_actual_headers_refuse_one_byte_short_occupied_shared_capacity() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    let cap = budget.tighten_total_byte_limit(CAP / 2).unwrap();
    let before = budget.receipt().unwrap();
    let header = row_owner_bytes().unwrap();
    let occupied = cap
        .checked_sub(before.live_persistent_bytes + before.live_scratch_bytes)
        .and_then(|n| n.checked_sub(header - 1))
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
    let error = source.sparse_input_rows(0, &[3, 1], None).err().unwrap();
    assert!(error.to_string().contains("MemoryLimit"), "{error}");
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        before.work_reserved
    );
    assert_eq!(live(&memory), bytes);
    drop(blocker);
    let rows = source.sparse_input_rows(0, &[3, 1], None).unwrap();
    let before = budget.receipt().unwrap();
    let header = halfspace_owner_bytes().unwrap();
    let occupied = cap
        .checked_sub(before.live_persistent_bytes + before.live_scratch_bytes)
        .and_then(|n| n.checked_sub(header - 1))
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
    let error = rows
        .necessary_halfspace(0, PreparedInputSplitSide::Active, None)
        .err()
        .unwrap();
    assert!(error.to_string().contains("MemoryLimit"), "{error}");
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        before.work_reserved
    );
    assert_eq!(live(&memory), bytes);
    drop(blocker);
    rows.necessary_halfspace(0, PreparedInputSplitSide::Active, None)
        .unwrap();
}

#[test]
fn prepared_sparse_work_refusal_precedes_untrusted_scan_and_releases_partial_owners() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    // Only enough work for this public request header. The next original
    // producer phase must refuse before the duplicate late ID is inspected.
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: MAX_WORK - before.work_reserved - row_owner_work(2).unwrap(),
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let error = source.sparse_input_rows(0, &[3, 3], None).err().unwrap();
    assert!(error.to_string().contains("WorkLimit"), "{error}");
    assert_eq!(live(&memory), bytes);
    assert_eq!(budget.receipt().unwrap().work_reserved, MAX_WORK);
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

    // Separate genuine source: measure the unchanged producer's actual cost,
    // then leave one unit less after the public header. No guessed census or
    // zero-admission assertion is used for the later partially admitted fail.
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    let product = source.source().construction().layer_products(1).unwrap();
    let before = budget.receipt().unwrap();
    let direct =
        super::run_prepared_prefix_input_rows(product.prefix_before(), &[3, 1], None).unwrap();
    let producer_work = budget.receipt().unwrap().work_reserved - before.work_reserved;
    assert!(producer_work > 1);
    drop(direct);
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: MAX_WORK
                    - before.work_reserved
                    - row_owner_work(2).unwrap()
                    - (producer_work - 1),
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let occupied = budget.receipt().unwrap();
    let error = source.sparse_input_rows(1, &[3, 1], None).err().unwrap();
    assert!(error.to_string().contains("WorkLimit"), "{error}");
    let refused = budget.receipt().unwrap();
    assert!(refused.admitted > occupied.admitted);
    assert!(refused.work_reserved >= occupied.work_reserved + row_owner_work(2).unwrap());
    assert_eq!(refused.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(refused.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(live(&memory), bytes);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    drop(blocker);
    let remaining = MAX_WORK - refused.work_reserved;
    assert!(matches!(
        budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: remaining + 1,
                ..WorkRequest::default()
            }
        ),
        Err(WorkRefusal::WorkLimit)
    ));
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        refused.work_reserved
    );
    source.checkpoint(None).unwrap();
}

#[test]
fn prepared_sparse_clones_and_borrowed_halfspaces_keep_last_actual_claims_alive() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(&budget, false);
    let rows = source.sparse_input_rows(0, &[3, 1], None).unwrap();
    let before = live(&memory);
    let receipt = budget.receipt().unwrap();
    let alias = rows.clone();
    assert_eq!(
        rows.targets(None).unwrap().as_ptr(),
        alias.targets(None).unwrap().as_ptr()
    );
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, receipt.admitted);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        receipt.work_reserved
    );
    drop(source);
    drop(rows);
    assert_eq!(live(&memory), before);
    let halfspace = alias
        .necessary_halfspace(0, PreparedInputSplitSide::Active, None)
        .unwrap();
    assert!(live(&memory) > before);
    halfspace.checkpoint(None).unwrap();
    alias.lower(1, None).unwrap();
    // The halfspace's real borrow prevents dropping alias before this drop.
    drop(halfspace);
    assert_eq!(live(&memory), before);
    drop(alias);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}
