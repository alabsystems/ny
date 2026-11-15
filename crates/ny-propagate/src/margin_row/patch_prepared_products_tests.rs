// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Actual original prepared-prefix patch production and refusal controls.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::engine::run_prepared_prefix_input_rows;
use crate::margin_row::input_rows::InputRowsBacking;
use crate::margin_row::support_program::{OriginalModel, OriginalRoot};
use crate::margin_row::{RootPrecisionRuntimePolicy, RoundMode};
use num_rational::BigRational;
use num_traits::Zero;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
const CAP: usize = 128 << 20;
fn authority() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let token = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &token,
        WorkLimits {
            deadline: None,
            scratch_bytes: CAP / 2,
            persistent_bytes: CAP / 2,
            max_work: 1_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, token)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_materialized_bytes + r.live_pending_bytes
}
fn at() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn fixture(n: usize, fold: bool, stable: bool) -> TwinSpec {
    let err = if fold { 0.003 } else { 0.0 };
    TwinSpec {
        n_in: n,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![-0.75, 0.5, 0.25],
                bias: vec![if stable { 2.0 } else { 0.125 }],
                bias_err: vec![if fold { 0.001 } else { 0.0 }],
                weight_rel_err: err,
                kernel: (1, 1, 1, 3),
                stride: (1, 1),
                pads: (0, 1, 0, 1),
                ishape: (1, 1, n),
                oshape: (1, 1, n),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![-1.75],
                shift: vec![0.25],
                scale_rel_err: err,
                shift_err: vec![if fold { 0.002 } else { 0.0 }],
                shape: (1, 1, n),
            },
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![0.25; n],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, n),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![0.5; 101],
                bias: vec![0.0; 101],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (101, 1),
            },
        ],
    }
}
fn prepare(n: usize, fold: bool, stable: bool, budget: &WorkBudget) -> PreparedPrefix {
    let model = OriginalModel::compile(fixture(n, fold, stable), budget, None).unwrap();
    let root = OriginalRoot::new(
        model,
        &vec![-0.5; n],
        &vec![0.75; n],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap();
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    root.empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap()
        .0
}
fn rat(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}
fn original(input: &[f64], target: usize, fold: bool, bits: usize) -> BigRational {
    let sign = |k| {
        if bits & (1_usize << k) == 0 {
            -1.0
        } else {
            1.0
        }
    };
    let rel = if fold { 0.003 } else { 0.0 };
    let mut h = rat(0.125) + rat(sign(3)) * rat(if fold { 0.001 } else { 0.0 });
    for (tap, w) in [-0.75, 0.5, 0.25].into_iter().enumerate() {
        if let Some(i) = target
            .checked_add(tap)
            .and_then(|i| i.checked_sub(1))
            .filter(|&i| i < input.len())
        {
            h += rat(w) * (rat(1.0) + rat(sign(tap)) * rat(rel)) * rat(input[i]);
        }
    }
    h = h.max(BigRational::zero());
    rat(-1.75) * (rat(1.0) + rat(sign(4)) * rat(rel)) * &h
        + rat(0.25)
        + rat(sign(5)) * rat(if fold { 0.002 } else { 0.0 })
        + h
}
fn rational_control(fold: bool) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority();
    let prefix = prepare(5, fold, false, &budget);
    let d = at();
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(d)).unwrap();
    let rows = run_prepared_prefix_patch_input_rows(&plan, &[4, 0, 2], Some(d)).unwrap();
    assert_eq!(rows.backing(), InputRowsBacking::F64);
    rows.validate_prepared_prefix(&prefix, 4, &[4, 0, 2])
        .unwrap();
    for input in [
        [-0.5; 5],
        [0.75; 5],
        [-0.5, 0.75, 0.25, -0.25, 0.5],
        [0.5, -0.5, 0.75, 0.0, -0.25],
    ] {
        for bits in 0..if fold { 64 } else { 1 } {
            for (column, target) in [4, 0, 2].into_iter().enumerate() {
                let exact = original(&input, target, fold, bits);
                for upper in [false, true] {
                    let row = if upper {
                        rows.upper(column)
                    } else {
                        rows.lower(column)
                    }
                    .unwrap();
                    let line = row.entries().fold(rat(row.bias()), |sum, (i, a, _)| {
                        sum + rat(a) * rat(input[i])
                    });
                    let eta = rat(row.pointwise_error(d).unwrap());
                    assert!(
                        if upper {
                            line + eta >= exact
                        } else {
                            line - eta <= exact
                        },
                        "target={target},fold={fold},bits={bits},upper={upper}"
                    );
                }
            }
        }
    }
}
#[test]
fn prepared_patch_signed_padded_residual_rows_contain_original_exact_rational_values() {
    rational_control(false);
}
#[test]
fn prepared_patch_rows_pay_all_independent_weight_bias_and_affine_fold_uncertainties() {
    rational_control(true);
}
#[test]
fn prepared_patch_nominal_rows_equal_dense_and_support_uses_absolute_ordered_original_inputs() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority();
    let prefix = prepare(5, false, true, &budget);
    let d = at();
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(d)).unwrap();
    let targets = [4, 0, 2];
    let patch = run_prepared_prefix_patch_input_rows(&plan, &targets, Some(d)).unwrap();
    let dense = run_prepared_prefix_input_rows(&prefix, &targets, Some(d)).unwrap();
    for col in 0..targets.len() {
        for upper in [false, true] {
            let p = if upper {
                patch.upper(col)
            } else {
                patch.lower(col)
            }
            .unwrap();
            let q = if upper {
                dense.upper(col)
            } else {
                dense.lower(col)
            }
            .unwrap();
            assert_eq!(p.bias().to_bits(), q.bias().to_bits());
            let actual: Vec<_> = p.entries().collect();
            let expected: Vec<_> = q.entries().collect();
            assert!(actual.windows(2).all(|w| w[0].0 < w[1].0));
            for (i, a, e) in &actual {
                let (_, da, de) = expected[*i];
                assert!(*i < 5);
                assert_eq!(
                    if *a == 0.0 { 0 } else { a.to_bits() },
                    if da == 0.0 { 0 } else { da.to_bits() }
                );
                assert!(*e <= de);
            }
            for (i, a, _) in expected {
                if !actual.iter().any(|v| v.0 == i) {
                    assert_eq!(a, 0.0);
                }
            }
        }
    }
}
#[test]
fn prepared_patch_plan_is_reusable_and_rows_retain_prefix_after_last_plan_owner_drops() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority();
    let prefix = prepare(5, false, false, &budget);
    let d = at();
    let before = live(&memory);
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(d)).unwrap();
    let paid = plan.resident_bytes().unwrap();
    assert!(live(&memory) >= before + paid);
    let rows = run_prepared_prefix_patch_input_rows(&plan, &[4, 0], Some(d)).unwrap();
    let other = run_prepared_prefix_patch_input_rows(&plan, &[2], Some(d)).unwrap();
    drop(other);
    drop(plan);
    drop(prefix);
    assert!(live(&memory) > 0);
    assert!(rows.lower(0).unwrap().pointwise_error(d).is_ok());
    drop(rows);
    assert_eq!(live(&memory), 0);
}
#[test]
fn prepared_patch_rejects_expired_duplicate_out_of_range_foreign_and_reordered_scopes_atomically() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority();
    let prefix = prepare(5, false, false, &budget);
    let other = prepare(5, false, false, &budget);
    let d = at();
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(d)).unwrap();
    let before = live(&memory);
    for targets in [&[][..], &[0, 0][..], &[5][..]] {
        assert!(run_prepared_prefix_patch_input_rows(&plan, targets, Some(d)).is_err());
        assert_eq!(live(&memory), before);
    }
    assert!(run_prepared_prefix_patch_input_rows(
        &plan,
        &[0],
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    assert_eq!(live(&memory), before);
    let rows = run_prepared_prefix_patch_input_rows(&plan, &[4, 0], Some(d)).unwrap();
    assert!(rows.validate_prepared_prefix(&other, 4, &[4, 0]).is_err());
    assert!(rows.validate_prepared_prefix(&prefix, 4, &[0, 4]).is_err());
}
#[test]
fn prepared_patch_plan_and_row_cap_refusals_preserve_other_original_owner_claims() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority();
    let prefix = prepare(5, false, false, &budget);
    let before = live(&memory);
    let limits = budget.receipt().unwrap();
    let remaining = CAP / 2 - limits.live_persistent_bytes;
    let mut blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: remaining - 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    blocker.mark_materialized().unwrap();
    let blocker = blocker.commit_persistent().unwrap();
    assert!(PreparedPatchPlan::build(&prefix, 1.0, Some(at())).is_err());
    assert_eq!(live(&memory), before + blocker.bytes());
    drop(blocker);
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(at())).unwrap();
    let request = prepared_patch_request(&plan, 4).unwrap();
    let scratch_cap = CAP / 2;
    let mut blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: scratch_cap - request.scratch_bytes + 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    blocker.mark_materialized().unwrap();
    let before = live(&memory);
    assert!(run_prepared_prefix_patch_input_rows(&plan, &[4, 0], Some(at())).is_err());
    assert_eq!(live(&memory), before);
}
#[test]
fn prepared_patch_preserves_original_deadline_and_shared_cancellation_without_renewal() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, token) = authority();
    let prefix = prepare(5, false, false, &budget);
    let d = at();
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(d)).unwrap();
    assert_eq!(
        plan.deadline(Some(d + Duration::from_secs(300))).unwrap(),
        d
    );
    let before = live(&memory);
    token.cancel();
    assert!(run_prepared_prefix_patch_input_rows(&plan, &[0], Some(d)).is_err());
    assert_eq!(live(&memory), before);
}
#[test]
fn prepared_patch_actual_original_input_and_class_counts_exceed_native_head_caps() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority();
    let prefix = prepare(5000, false, false, &budget);
    let d = at();
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(d)).unwrap();
    let rows = run_prepared_prefix_patch_input_rows(&plan, &[4999, 0], Some(d)).unwrap();
    assert_eq!(rows.net().n_in, 5000);
    assert_eq!(rows.net().n_out, 101);
    assert!(rows.upper(0).unwrap().entries().count() <= 3);
}
#[test]
fn prepared_patch_checked_geometry_rejects_overflow_without_entering_allocations() {
    let _env = crate::tests::lock_env_shared();
    let ops = [
        GeoOp::Conv {
            input: 0,
            kernel: (1, 1, 1, 3),
            stride: (1, usize::MAX),
            pads: (0, 0, 0, 0),
            ishape: (1, 1, 5),
            oshape: (1, 1, 5),
            transposed: false,
        },
        GeoOp::Relu { input: 1 },
    ];
    assert!(prepared_geometry(&ops, &[5, 5, 5], 1, 1.0, &|| Ok(())).is_err());
    let a = Frame {
        sy: 1,
        sx: 1,
        oy: 0,
        ox: isize::MAX,
        ph: 1,
        pw: 3,
        ..Frame::seed((1, 1, 5))
    };
    let mut frames = [Some(a)];
    assert!(prepared_union_into(&mut frames, 0, Frame::seed((1, 1, 5))).is_err());
}
struct CancelFrame<'a> {
    actual: &'a PreparedPrefixFrame<'a>,
    token: &'a CancellationToken,
    count: AtomicUsize,
    cancel_at: usize,
}
impl FrozenGateFrame for CancelFrame<'_> {
    fn mode(&self) -> RoundMode {
        self.actual.mode()
    }
    fn layer_count(&self) -> usize {
        self.actual.layers().len()
    }
    fn layer(&self, i: usize) -> Result<&super::super::root::LayerGates> {
        self.actual
            .gate(i)
            .ok_or_else(|| product_bad("missing actual gate"))
    }
    fn lo(&self) -> &[f64] {
        self.actual.root().lo()
    }
    fn hi(&self) -> &[f64] {
        self.actual.root().hi()
    }
    fn mid(&self) -> &[f64] {
        self.actual.root().mid()
    }
    fn rad(&self) -> &[f64] {
        self.actual.root().rad()
    }
    fn xabs(&self) -> &[f64] {
        self.actual.root().xabs()
    }
    fn cooperative(&self) -> bool {
        true
    }
    fn checkpoint(&self) -> Result<()> {
        if self.count.fetch_add(1, Ordering::SeqCst) == self.cancel_at {
            self.token.cancel();
        }
        self.actual.checkpoint()
    }
}
#[test]
fn prepared_patch_common_numerical_walk_polls_original_cancellation_inside_long_work() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, token) = authority();
    // An actual broad convolution makes the embedded/nominal/error frontier
    // large enough to exercise bounded checks inside numerical work.
    let n = 512;
    let mut spec = fixture(n, false, false);
    if let TwinOpSpec::Conv {
        weight,
        kernel,
        pads,
        ..
    } = &mut spec.ops[0]
    {
        *weight = (0..129)
            .map(|i| if i % 2 == 0 { 0.0078125 } else { -0.00390625 })
            .collect();
        *kernel = (1, 1, 1, 129);
        *pads = (0, 64, 0, 64);
    }
    let model = OriginalModel::compile(spec, &budget, None).unwrap();
    let root = OriginalRoot::new(
        model,
        &vec![-0.5; n],
        &vec![0.75; n],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap();
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let prefix = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap()
        .0;
    let d = at();
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(d)).unwrap();
    let request = prepared_patch_request(&plan, 64).unwrap();
    let _lease = budget.lease_for(LeaseClass::Fallback, request).unwrap();
    let frame = prefix.frame(Some(d)).unwrap();
    let cancel = CancelFrame {
        actual: &frame,
        token: &token,
        count: AtomicUsize::new(0),
        cancel_at: 80,
    };
    let targets: Vec<_> = (0..32).chain(0..32).collect();
    let lower: Vec<_> = (0..64).map(|i| i < 32).collect();
    let result = PatchKernel {
        source_net: prefix.root().model().net(),
        geo: &plan.geo,
        vops: &plan.vops,
    }
    .walk_inner(
        prefix.root().model().net(),
        &cancel,
        &targets,
        &lower,
        None,
        &GpuConvRoute::Off,
        Some(d),
    );
    assert!(matches!(result, Some(Err(_))));
    assert!(cancel.count.load(Ordering::SeqCst) > 80);
}

#[test]
fn prepared_patch_rejects_invalid_ratio_and_no_gain_geometry_without_releasing_prior_prefix() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority();
    let prefix = prepare(5, false, false, &budget);
    let before = live(&memory);
    for ratio in [f64::NAN, 0.0, 1.1] {
        assert!(PreparedPatchPlan::build(&prefix, ratio, Some(at())).is_err());
        assert_eq!(live(&memory), before);
    }
    let one = prepare(1, false, false, &budget);
    let before = live(&memory);
    assert!(PreparedPatchPlan::build(&one, 0.5, Some(at())).is_err());
    assert_eq!(live(&memory), before);
}
#[test]
fn prepared_patch_fixed_local_ceiling_counts_retained_same_ledger_claims_under_larger_global_cap() {
    let _env = crate::tests::lock_env_shared();
    let cap = super::super::root_box::ROOT_SOURCE_BYTES_CAP;
    let memory = MemoryAuthority::new(cap * 4);
    let token = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &token,
        WorkLimits {
            deadline: None,
            scratch_bytes: cap * 2,
            persistent_bytes: cap * 2,
            max_work: 1_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    let prefix = prepare(5, false, false, &budget);
    let plan = PreparedPatchPlan::build(&prefix, 1.0, Some(at())).unwrap();
    let current = budget.receipt().unwrap().live_persistent_bytes;
    let target_scope_bytes = 2 * size_of::<usize>() + size_of::<InputRowScope<'static>>() + 256;
    let mut blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: cap - current - target_scope_bytes + 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    blocker.mark_materialized().unwrap();
    let blocker = blocker.commit_persistent().unwrap();
    let before = live(&memory);
    assert!(matches!(
        PreparedPatchPlan::build(&prefix, 1.0, Some(at())),
        Err(NyError::CpuMemoryExceeded { .. })
    ));
    assert_eq!(live(&memory), before);
    // There is room for every current owner, but one byte too little for
    // the target owner alone. No target lease or temporary peak may occur.
    assert!(before < cap);
    let work_before = budget.receipt().unwrap();
    let memory_before = memory.receipt().unwrap();
    assert!(matches!(
        run_prepared_prefix_patch_input_rows(&plan, &[4, 0], Some(at())),
        Err(NyError::CpuMemoryExceeded { .. })
    ));
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, work_before.admitted);
    let memory_after = memory.receipt().unwrap();
    assert_eq!(memory_after.admitted, memory_before.admitted);
    assert_eq!(
        memory_after.peak_leased_bytes,
        memory_before.peak_leased_bytes
    );
    drop(blocker);
    assert!(run_prepared_prefix_patch_input_rows(&plan, &[4, 0], Some(at())).is_ok());
}
