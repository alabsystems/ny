// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Controls use actual completed prepared-row producers, never supplied rows.

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkBudget, WorkLimits, WorkReserve};
use crate::margin_row::engine::run_prepared_prefix_input_rows;
use crate::margin_row::input_rows::InputRowsBacking;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::support_program::{OriginalModel, OriginalRoot};
use crate::margin_row::RootPrecisionRuntimePolicy;
use num_rational::BigRational as Q;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 64 << 20;

fn q(value: f64) -> Q {
    Q::from_float(value).unwrap()
}

fn fixture(
    fold: bool,
) -> (
    PreparedPrefix,
    WorkBudget,
    MemoryAuthority,
    CancellationToken,
) {
    let memory = MemoryAuthority::new(CAP);
    let cancellation = CancellationToken::default();
    let deadline = Instant::now() + Duration::from_secs(30);
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: Some(deadline),
            scratch_bytes: CAP / 2,
            persistent_bytes: CAP / 2,
            max_work: 1_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let spec = TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![1.5],
                bias_err: vec![if fold { 0.001 } else { 0.0 }],
                weight_rel_err: if fold { 0.003 } else { 0.0 },
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 2),
                oshape: (1, 1, 2),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![-0.75],
                shift: vec![0.125],
                shift_err: vec![if fold { 0.002 } else { 0.0 }],
                scale_rel_err: if fold { 0.003 } else { 0.0 },
                shape: (1, 1, 2),
            },
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.75, -0.125],
                bias: vec![0.125],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (1, 2),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![0.5, -0.75],
                bias: vec![0.0, 0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    };
    let model = OriginalModel::compile(spec, &budget, Some(deadline)).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-2.0, -2.0],
        &[-1.0, -1.0],
        RootPrecisionRuntimePolicy::default(),
        Some(deadline),
    )
    .unwrap();
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let (prefix, _) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    (prefix, budget, memory, cancellation)
}

fn rows(prefix: &PreparedPrefix) -> InputRowPair<'static> {
    run_prepared_prefix_input_rows(prefix, &[1, 0], None).unwrap()
}

fn original(value: f64, fold: bool, signs: [f64; 4]) -> Q {
    let rel = if fold { 0.003 } else { 0.0 };
    let w = q(1.0) + q(signs[0]) * q(rel);
    let b = q(1.5) + q(signs[1]) * q(if fold { 0.001 } else { 0.0 });
    let h = (w * q(value) + b).max(Q::zero());
    let scale = q(-0.75) * (q(1.0) + q(signs[2]) * q(rel));
    scale * h + q(0.125) + q(signs[3]) * q(if fold { 0.002 } else { 0.0 })
}

fn check_original_phases(prefix: &PreparedPrefix, rows: &InputRowPair<'_>, fold: bool) {
    let values = [-2.0, -1.75, -1.5, -1.25, -1.0];
    for column in 0..2 {
        let inactive = input_halfspace_prepared(
            rows,
            prefix,
            3,
            &[1, 0],
            column,
            InputSplitSide::Inactive,
            rows.deadline(),
        )
        .unwrap();
        let active = input_halfspace_prepared(
            rows,
            prefix,
            3,
            &[1, 0],
            column,
            InputSplitSide::Active,
            rows.deadline(),
        )
        .unwrap();
        let target = [1, 0][column];
        assert_eq!(inactive.original_target(), target);
        assert_eq!(active.original_target(), target);
        assert_eq!(inactive.deadline(), rows.deadline());
        for x0 in values {
            for x1 in values {
                let input = [x0, x1];
                for bits in 0..16 {
                    let signs =
                        std::array::from_fn(|i| if bits & (1 << i) == 0 { -1.0 } else { 1.0 });
                    let truth = original(input[target], fold, signs);
                    let eval = |halfspace: &Halfspace| {
                        q(halfspace.d) + q(halfspace.a[0]) * q(x0) + q(halfspace.a[1]) * q(x1)
                    };
                    if truth <= Q::zero() {
                        assert!(eval(inactive.halfspace()) <= Q::zero());
                    }
                    if truth >= Q::zero() {
                        assert!(eval(active.halfspace()) <= Q::zero());
                    }
                }
            }
        }
        for (side, halfspace) in [
            (InputSplitSide::Inactive, &inactive),
            (InputSplitSide::Active, &active),
        ] {
            let row = match side {
                InputSplitSide::Inactive => rows.lower(column),
                InputSplitSide::Active => rows.upper(column),
            }
            .unwrap();
            let signed_bias = if side == InputSplitSide::Inactive {
                row.bias()
            } else {
                -row.bias()
            };
            let error = row
                .entries()
                .fold(q(row.bias_error()), |sum, (_, _, e)| sum + q(e) * q(2.0));
            assert!(q(halfspace.halfspace().d) <= q(signed_bias) - &error);
            assert!(q(halfspace.charged_error()) >= error);
        }
    }
}

#[test]
fn input_clip_prepared_actual_signed_rows_and_fold_corners_preserve_original_phase_members() {
    let _env = crate::tests::lock_env_shared();
    for fold in [false, true] {
        let (prefix, _, _, _) = fixture(fold);
        let rows = rows(&prefix);
        check_original_phases(&prefix, &rows, fold);
    }
}

#[test]
fn input_clip_prepared_real_f32_rows_pay_cast_gaps_in_both_phase_constraints() {
    let _env = crate::tests::lock_env_shared();
    let (prefix, _, _, _) = fixture(true);
    let rows = rows(&prefix);
    let required = rows.f32_storage_required_bytes(0, rows.deadline()).unwrap();
    let deadline = rows.deadline();
    let transferred = rows.into_f32(0, required, deadline).unwrap();
    assert_eq!(transferred.rows.backing(), InputRowsBacking::F32);
    assert!(transferred.receipt.coefficient_cells_stored > 0);
    check_original_phases(&prefix, &transferred.rows, true);
}

#[test]
fn input_clip_prepared_admits_output_on_original_ledger_and_releases_only_last_scope() {
    let _env = crate::tests::lock_env_shared();
    let (prefix, budget, memory, _) = fixture(false);
    let rows = rows(&prefix);
    let before = budget.receipt().unwrap();
    let halfspace = input_halfspace_prepared(
        &rows,
        &prefix,
        3,
        &[1, 0],
        0,
        InputSplitSide::Inactive,
        rows.deadline(),
    )
    .unwrap();
    let during = budget.receipt().unwrap();
    assert!(during.live_persistent_bytes > before.live_persistent_bytes);
    assert!(during.work_reserved > before.work_reserved);
    assert_eq!(during.live_scratch_bytes, before.live_scratch_bytes);
    assert!(halfspace._resident.as_ref().unwrap().shares_ledger(&budget));
    drop(halfspace);
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before.live_persistent_bytes
    );
    drop(rows);
    drop(prefix);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
}

#[test]
fn input_clip_prepared_foreign_prefix_operation_order_and_expiry_are_atomic_refusals() {
    let _env = crate::tests::lock_env_shared();
    let (prefix, budget, _, _) = fixture(false);
    let (foreign, _, _, _) = fixture(false);
    let rows = rows(&prefix);
    let before = budget.receipt().unwrap().live_persistent_bytes;
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    for (owner, op, targets, column, deadline) in [
        (&foreign, 3, [1, 0], 0, rows.deadline()),
        (&prefix, 1, [1, 0], 0, rows.deadline()),
        (&prefix, 3, [0, 1], 0, rows.deadline()),
        (&prefix, 3, [1, 0], 2, rows.deadline()),
        (&prefix, 3, [1, 0], 0, expired),
    ] {
        assert!(input_halfspace_prepared(
            &rows,
            owner,
            op,
            &targets,
            column,
            InputSplitSide::Active,
            deadline
        )
        .is_err());
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
    }
}

#[test]
fn input_clip_prepared_occupied_shared_capacity_and_cancellation_preserve_prior_evidence() {
    let _env = crate::tests::lock_env_shared();
    let (prefix, budget, _, cancellation) = fixture(false);
    let rows = rows(&prefix);
    let before = budget.receipt().unwrap().live_persistent_bytes;
    let occupied = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP / 2 - before,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    assert!(input_halfspace_prepared(
        &rows,
        &prefix,
        3,
        &[1, 0],
        0,
        InputSplitSide::Active,
        rows.deadline()
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, CAP / 2);
    drop(occupied);
    let halfspace = input_halfspace_prepared(
        &rows,
        &prefix,
        3,
        &[1, 0],
        0,
        InputSplitSide::Active,
        rows.deadline(),
    )
    .unwrap();
    let held = budget.receipt().unwrap().live_persistent_bytes;
    cancellation.cancel();
    assert!(input_halfspace_prepared(
        &rows,
        &prefix,
        3,
        &[1, 0],
        1,
        InputSplitSide::Inactive,
        rows.deadline()
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, held);
    drop(halfspace);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
}

#[test]
fn input_clip_prepared_work_refusal_precedes_malformed_late_target_contents() {
    let _env = crate::tests::lock_env_shared();
    let (prefix, budget, memory, _) = fixture(false);
    let rows = rows(&prefix);
    assert_eq!(rows.targets(), [1, 0]);
    // The last declaration differs. Shape/count/owner binding remains genuine.
    let malformed = [1, 1];
    let quote = u64::try_from((rows.n_in() + rows.columns()) * 32 + 1024).unwrap();
    let before = budget.receipt().unwrap();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 1_000_000_000 - before.work_reserved - (quote - 1),
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let occupied = budget.receipt().unwrap();
    let memory_before = memory.receipt().unwrap();
    let error = input_halfspace_prepared(
        &rows,
        &prefix,
        3,
        &malformed,
        0,
        InputSplitSide::Active,
        rows.deadline(),
    )
    .err()
    .expect("the full scan must be admitted before its malformed final element")
    .to_string();
    assert!(error.contains("input clipping original budget: WorkLimit"));
    assert!(!error.contains("changed original target order"));
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, occupied.admitted);
    assert_eq!(after.work_reserved, occupied.work_reserved);
    assert_eq!(after.live_persistent_bytes, occupied.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, occupied.live_scratch_bytes);
    let memory_after = memory.receipt().unwrap();
    assert_eq!(
        memory_after.live_pending_bytes,
        memory_before.live_pending_bytes
    );
    assert_eq!(
        memory_after.live_materialized_bytes,
        memory_before.live_materialized_bytes
    );
    assert_eq!(
        memory_after.peak_leased_bytes,
        memory_before.peak_leased_bytes
    );
    drop((blocker, rows, prefix));
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn input_clip_prepared_paid_scan_preserves_late_target_scope_refusal_and_releases_output() {
    let _env = crate::tests::lock_env_shared();
    let (prefix, budget, memory, _) = fixture(false);
    let rows = rows(&prefix);
    let quote = u64::try_from((rows.n_in() + rows.columns()) * 32 + 1024).unwrap();
    let before = budget.receipt().unwrap();
    let memory_before = memory.receipt().unwrap();
    let error = input_halfspace_prepared(
        &rows,
        &prefix,
        3,
        &[1, 1],
        0,
        InputSplitSide::Inactive,
        rows.deadline(),
    )
    .err()
    .expect("paid validation still rejects changed original target order")
    .to_string();
    assert!(error.contains("changed original target order"));
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted + 1);
    assert_eq!(after.work_reserved, before.work_reserved + quote);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    let memory_after = memory.receipt().unwrap();
    assert_eq!(
        memory_after.live_pending_bytes,
        memory_before.live_pending_bytes
    );
    assert_eq!(
        memory_after.live_materialized_bytes,
        memory_before.live_materialized_bytes
    );
    // The original signature still authenticates valid original order.
    rows.validate_prepared_prefix(&prefix, 3, &[1, 0]).unwrap();
    drop((rows, prefix));
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn input_clip_prepared_cheap_binding_refuses_before_exhausted_work_admission() {
    let _env = crate::tests::lock_env_shared();
    let (prefix, budget, _, _) = fixture(false);
    let (foreign, _, _, _) = fixture(false);
    let rows = rows(&prefix);
    let before = budget.receipt().unwrap();
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: 1_000_000_000 - before.work_reserved,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let occupied = budget.receipt().unwrap();
    for (owner, op, targets) in [
        (&foreign, 3, &[1, 0][..]),
        (&prefix, 1, &[1, 0][..]),
        (&prefix, 3, &[1][..]),
    ] {
        let error = input_halfspace_prepared(
            &rows,
            owner,
            op,
            targets,
            0,
            InputSplitSide::Active,
            rows.deadline(),
        )
        .err()
        .expect("cheap original binding must precede target-work admission")
        .to_string();
        assert!(error.contains("stale original prefix/operation/ordered targets"));
        assert_eq!(budget.receipt().unwrap(), occupied);
    }
    drop((blocker, rows, prefix, foreign));
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
}

#[test]
fn input_clip_prepared_requested_target_validation_clock_cannot_renew_producer() {
    let _env = crate::tests::lock_env_shared();
    let (prefix, budget, _, _) = fixture(false);
    let rows = rows(&prefix);
    let original = rows.deadline();
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .expect("test clock supports an expired deadline");
    assert!(original > Instant::now());
    let before = budget.receipt().unwrap();
    let error = rows
        .validate_prepared_prefix_with_deadline(&prefix, 3, &[1, 1], expired)
        .expect_err("requested earlier clock must beat malformed target contents");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
    assert_eq!(budget.receipt().unwrap(), before);
    assert_eq!(rows.deadline(), original);
    rows.validate_prepared_prefix(&prefix, 3, &[1, 0]).unwrap();
    assert!(input_halfspace_prepared(
        &rows,
        &prefix,
        3,
        &[1, 0],
        0,
        InputSplitSide::Active,
        expired,
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap(), before);
}
