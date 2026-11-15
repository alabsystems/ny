// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whole-original rational and resource controls; no terminal fixture may
//! supply gates, head boxes, seeds or a fabricated completed construction.

use super::super::{build_prepared_root, OriginalModel, OriginalRoot, PreparedConstructionOptions};
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::RootPrecisionRuntimePolicy;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 128 << 20;
fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources(
    cap: usize,
    original_deadline: Option<Instant>,
) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap.checked_mul(2).unwrap());
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: original_deadline,
            scratch_bytes: cap,
            persistent_bytes: cap,
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
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn source(
    spec: TwinSpec,
    budget: &WorkBudget,
    original_deadline: Option<Instant>,
) -> PreparedRootConstruction {
    let n = spec.n_in;
    let model = OriginalModel::compile(spec, budget, original_deadline).unwrap();
    let root = OriginalRoot::new(
        model,
        &vec![-1.0; n],
        &vec![1.0; n],
        RootPrecisionRuntimePolicy::default(),
        original_deadline,
    )
    .unwrap();
    build_prepared_root(root, PreparedConstructionOptions::default(), Some(at())).unwrap()
}

fn fixture(fold: bool) -> TwinSpec {
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
                weight: vec![1.5, -0.75, 0.25],
                bias: vec![0.125, 0.5, -0.25],
                bias_err: if fold { vec![1.0 / 512.0; 3] } else { vec![] },
                weight_rel_err: if fold { 1.0 / 256.0 } else { 0.0 },
                shape: (3, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, -0.5, 0.75, -1.0, 0.25, 0.5, 0.25, 1.0, -0.75],
                bias: vec![0.5, -0.125, 0.25],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (3, 3),
            },
        ],
    }
}
fn r(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

/// Independent authored-coordinate interpretation. Each admitted parameter
/// uncertainty is varied separately; no compiler, seed, gate or error formula
/// is reused. For each corner, all exact piecewise-linear breakpoints give the
/// COMPLETE one-dimensional minima, rather than a nominal random-point check.
fn exact_corners(spec: &TwinSpec, fold: bool, result: &PreparedTerminal) {
    let (cw, cb, cr, ce) = match &spec.ops[0] {
        TwinOpSpec::Conv {
            weight,
            bias,
            weight_rel_err,
            bias_err,
            ..
        } => (weight[0], bias[0], *weight_rel_err, bias_err[0]),
        _ => unreachable!(),
    };
    let (hw, hb, hr, he) = match &spec.ops[3] {
        TwinOpSpec::Gemm {
            weight,
            bias,
            weight_rel_err,
            bias_err,
            ..
        } => (weight, bias, *weight_rel_err, bias_err),
        _ => unreachable!(),
    };
    let (fw, fb) = match &spec.ops[5] {
        TwinOpSpec::Gemm { weight, bias, .. } => (weight, bias),
        _ => unreachable!(),
    };
    let mut minima: Vec<Option<BigRational>> = vec![None; 2];
    for bits in 0..if fold { 256 } else { 1 } {
        let sign = |bit: usize| {
            if bits & (1 << bit) == 0 {
                r(-1.0)
            } else {
                r(1.0)
            }
        };
        let w = r(cw) * (r(1.0) + sign(0) * r(cr));
        let b = r(cb) + sign(1) * r(ce);
        let wh: Vec<_> = (0..3)
            .map(|j| r(hw[j]) * (r(1.0) + sign(2 + j) * r(hr)))
            .collect();
        let bh: Vec<_> = (0..3)
            .map(|j| r(hb[j]) + sign(5 + j) * r(if he.is_empty() { 0.0 } else { he[j] }))
            .collect();
        let mut points = vec![r(-1.0), r(1.0)];
        let trunk_kink = -&b / &w;
        if trunk_kink >= r(-1.0) && trunk_kink <= r(1.0) {
            points.push(trunk_kink);
        }
        for j in 0..3 {
            let h = -&bh[j] / &wh[j];
            if h >= BigRational::zero() {
                let x = (h - &b) / &w;
                if x >= r(-1.0) && x <= r(1.0) {
                    points.push(x);
                }
            }
        }
        for x in points {
            let h = (&w * x + &b).max(BigRational::zero());
            let ys: Vec<_> = (0..3).map(|j| &wh[j] * &h + &bh[j]).collect();
            for j in 0..3 {
                assert!(
                    r(result.head_lo()[j]) <= ys[j],
                    "lower head {j},corner {bits}"
                );
                assert!(
                    r(result.head_hi()[j]) >= ys[j],
                    "upper head {j},corner {bits}"
                );
            }
            let outputs: Vec<_> = (0..3)
                .map(|k| {
                    (0..3).fold(r(fb[k]), |sum, j| {
                        sum + r(fw[k * 3 + j]) * ys[j].clone().max(BigRational::zero())
                    })
                })
                .collect();
            for (column, &class) in [2usize, 1].iter().enumerate() {
                let margin = &outputs[0] - &outputs[class];
                assert!(
                    r(result.lower()[column]) <= margin,
                    "margin class {class},corner {bits}"
                );
                if minima[column].as_ref().is_none_or(|old| margin < *old) {
                    minima[column] = Some(margin);
                }
            }
        }
    }
    for (column, minimum) in minima.iter().enumerate() {
        assert!(r(result.lower()[column]) <= *minimum.as_ref().unwrap());
    }
}

#[test]
fn prepared_terminal_whole_original_unfolded_head_and_every_piecewise_minimum_are_enclosed() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let spec = fixture(false);
    let src = source(spec.clone(), &budget, None);
    let result = build_prepared_terminal(&src, 0, &[2, 1], Some(at())).unwrap();
    result
        .validate_source(&src, 0, &[2, 1], Some(at()))
        .unwrap();
    exact_corners(&spec, false, &result);
    assert_eq!(result.head_input(), 3);
    assert_eq!(result.adversaries(), [2, 1]);
    assert_eq!(
        (
            result.receipt().head_lower_columns,
            result.receipt().head_upper_columns,
            result.receipt().margin_columns
        ),
        (3, 3, 2)
    );
    assert_eq!(result.receipt().backward_passes, 3);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn prepared_terminal_whole_original_independent_conv_head_weight_and_bias_fold_corners_are_enclosed(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let spec = fixture(true);
    let src = source(spec.clone(), &budget, None);
    let result = build_prepared_terminal(&src, 0, &[2, 1], Some(at())).unwrap();
    exact_corners(&spec, true, &result);
    assert!(result.receipt().seed_rounding_terms > 0);
}

fn many(heads: usize, classes: usize) -> TwinSpec {
    let mut spec = fixture(false);
    if let TwinOpSpec::Conv { bias, .. } = &mut spec.ops[0] {
        bias[0] = 2.0;
    }
    spec.ops[3] = TwinOpSpec::Gemm {
        input: 3,
        weight: vec![0.25; heads],
        bias: vec![0.5; heads],
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (heads, 1),
    };
    spec.ops[5] = TwinOpSpec::Gemm {
        input: 5,
        weight: vec![0.0; classes * heads],
        bias: (0..classes)
            .map(|i| if i == 0 { 2.0 } else { 0.0 })
            .collect(),
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (classes, heads),
    };
    spec
}

#[test]
fn prepared_terminal_more_than512_head_rows_and100_classes_use_complete_real_bounded_tiles() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let src = source(many(600, 130), &budget, None);
    let classes: Vec<_> = (1..130).rev().collect();
    let result = build_prepared_terminal(&src, 0, &classes, Some(at())).unwrap();
    assert_eq!(result.adversaries(), classes);
    assert_eq!(result.head_lo().len(), 600);
    assert_eq!(result.head_hi().len(), 600);
    assert_eq!(result.lower().len(), 129);
    // Original h=(5/4)x+2 is active. Every original y=(1/4)h+1/2
    // lies in [11/16,21/16]; original class margins are exactly2.
    for (&lo, &hi) in result.head_lo().iter().zip(result.head_hi()) {
        assert!(r(lo) <= r(11.0 / 16.0));
        assert!(r(hi) >= r(21.0 / 16.0));
    }
    assert!(result.lower().iter().all(|&x| x > 0.0 && r(x) <= r(2.0)));
    let receipt = result.receipt();
    assert_eq!(
        (
            receipt.head_lower_columns,
            receipt.head_upper_columns,
            receipt.margin_columns
        ),
        (600, 600, 129)
    );
    assert_eq!(
        (
            receipt.head_tiles,
            receipt.margin_tiles,
            receipt.backward_passes
        ),
        (10, 3, 23)
    );
    assert_eq!(receipt.max_tile_columns, 64);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

fn blocker(budget: &WorkBudget, bytes: usize) -> PersistentReservation {
    let mut lease = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: bytes,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    lease.mark_materialized().unwrap();
    lease.commit_persistent().unwrap()
}

#[test]
fn prepared_terminal_cap_adapts_before_seed_allocation_to_one_column_without_skipping_any_original_row(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(ROOT_SOURCE_BYTES_CAP + CAP, None);
    let src = source(many(8, 5), &budget, None);
    let limit = at();
    let p = plan(&src, 4, limit).unwrap();
    let old = budget.receipt().unwrap();
    let visible = old.live_scratch_bytes + old.live_persistent_bytes;
    let minimum_tile = p.walk_fixed_bytes + p.walk_bytes_per_column + p.frame_bytes;
    let held = blocker(
        &budget,
        ROOT_SOURCE_BYTES_CAP - visible - p.persistent - p.scratch - minimum_tile,
    );
    let result = build_prepared_terminal(&src, 0, &[4, 3, 2, 1], Some(limit)).unwrap();
    assert_eq!(result.receipt().max_tile_columns, 1);
    assert_eq!(
        (
            result.receipt().head_lower_columns,
            result.receipt().head_upper_columns,
            result.receipt().margin_columns
        ),
        (8, 8, 4)
    );
    assert!(
        result.receipt().tile_admission_attempts
            > result.receipt().head_tiles + result.receipt().margin_tiles
    );
    assert!(result.lower().iter().all(|&b| b > 0.0));
    assert!(memory.receipt().unwrap().peak_leased_bytes <= ROOT_SOURCE_BYTES_CAP);
    drop(result);
    drop(held);
    drop(src);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_terminal_whole_output_one_past_cap_refuses_before_output_claim_and_keeps_existing_source(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(ROOT_SOURCE_BYTES_CAP + CAP, None);
    let src = source(many(8, 5), &budget, None);
    let limit = at();
    let p = plan(&src, 4, limit).unwrap();
    let old = budget.receipt().unwrap();
    let visible = old.live_scratch_bytes + old.live_persistent_bytes;
    let held = blocker(
        &budget,
        ROOT_SOURCE_BYTES_CAP - visible - p.persistent - p.scratch + 1,
    );
    let before = budget.receipt().unwrap();
    let bytes = live(&memory);
    assert!(
        matches!(build_prepared_terminal(&src,0,&[4,3,2,1],Some(limit)),Err(NyError::CpuMemoryExceeded { required_bytes,budget_bytes,.. }) if required_bytes==ROOT_SOURCE_BYTES_CAP+1 && budget_bytes==ROOT_SOURCE_BYTES_CAP)
    );
    let after = budget.receipt().unwrap();
    // The explicitly priced geometry header completes; no output lease or
    // tile/seed allocation was admitted. Work is never refilled on refusal.
    assert_eq!(after.admitted, before.admitted + 1);
    assert_eq!(after.live_scratch_bytes, before.live_scratch_bytes);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert!(after.work_reserved > before.work_reserved);
    assert_eq!(live(&memory), bytes);
    drop(held);
    let result = build_prepared_terminal(&src, 0, &[4, 3, 2, 1], Some(limit)).unwrap();
    assert_eq!(result.lower().len(), 4);
}

#[test]
fn prepared_terminal_fewer_than_one_affordable_tile_refuses_and_releases_only_its_new_capacities() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(ROOT_SOURCE_BYTES_CAP + CAP, None);
    let src = source(many(8, 5), &budget, None);
    let limit = at();
    let p = plan(&src, 4, limit).unwrap();
    let old = budget.receipt().unwrap();
    let visible = old.live_scratch_bytes + old.live_persistent_bytes;
    let minimum_tile = p.walk_fixed_bytes + p.walk_bytes_per_column + p.frame_bytes;
    let held = blocker(
        &budget,
        ROOT_SOURCE_BYTES_CAP - visible - p.persistent - p.scratch - minimum_tile + 1,
    );
    let bytes = live(&memory);
    assert!(matches!(
        build_prepared_terminal(&src, 0, &[4, 3, 2, 1], Some(limit)),
        Err(NyError::CpuMemoryExceeded { .. })
    ));
    assert_eq!(live(&memory), bytes);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(held);
}

#[test]
fn prepared_terminal_exact_original_owner_mapping_and_order_refuse_foreign_equal_limit_sources() {
    let _env = crate::tests::lock_env_shared();
    let (first, _, _) = resources(CAP, None);
    let src = source(fixture(false), &first, None);
    let terminal = build_prepared_terminal(&src, 0, &[2, 1], Some(at())).unwrap();
    let (second, _, _) = resources(CAP, None);
    let foreign = source(fixture(false), &second, None);
    for (root, t, classes) in [
        (&foreign, 0, vec![2, 1]),
        (&src, 1, vec![2, 1]),
        (&src, 0, vec![1, 2]),
        (&src, 0, vec![2]),
    ] {
        assert!(terminal
            .validate_source(root, t, &classes, Some(at()))
            .is_err());
    }
    terminal
        .validate_source(&src, 0, &[2, 1], Some(at()))
        .unwrap();
    assert!(!terminal.0.resident.shares_ledger(&second));
}

#[test]
fn prepared_terminal_duplicate_missing_self_and_out_of_range_classes_cannot_publish_partial_products(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let src = source(fixture(false), &budget, None);
    let old = live(&memory);
    for (target, classes) in [
        (0, vec![]),
        (0, vec![1, 1]),
        (0, vec![0]),
        (0, vec![3]),
        (3, vec![1]),
    ] {
        assert!(build_prepared_terminal(&src, target, &classes, Some(at())).is_err());
        assert_eq!(live(&memory), old);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }
}

#[test]
fn prepared_terminal_earliest_root_deadline_survives_fresh_caller_clock_and_last_clone_releases_claim_once(
) {
    let _env = crate::tests::lock_env_shared();
    let limit = at();
    let (budget, memory, _) = resources(CAP, Some(limit));
    let src = source(fixture(false), &budget, Some(limit));
    let result =
        build_prepared_terminal(&src, 0, &[2, 1], Some(limit + Duration::from_secs(300))).unwrap();
    assert_eq!(
        result
            .deadline(Some(limit + Duration::from_secs(600)))
            .unwrap(),
        limit
    );
    assert!(result.deadline(Some(Instant::now())).is_err());
    let old = live(&memory);
    let clone = result.clone();
    assert_eq!(live(&memory), old);
    assert!(Arc::ptr_eq(&clone.0, &result.0));
    let claim = result.resident_bytes();
    drop(result);
    assert_eq!(live(&memory), old);
    drop(clone);
    assert_eq!(live(&memory), old - claim);
    drop(src);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_terminal_cancel_after_actual_head_tile_discards_every_new_result_and_keeps_spent_work()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources(CAP, None);
    let src = source(fixture(true), &budget, None);
    let old = live(&memory);
    let before = budget.receipt().unwrap();
    let mut observed = 0;
    let mut hook = |receipt: &PreparedTerminalReceipt| {
        assert!(receipt.head_lower_columns > 0);
        observed += 1;
        cancel.cancel();
    };
    assert!(build_inner(&src, 0, &[2, 1], Some(at()), Some(&mut hook)).is_err());
    assert_eq!(observed, 1);
    assert_eq!(live(&memory), old);
    let after = budget.receipt().unwrap();
    assert_eq!(after.live_scratch_bytes, 0);
    assert_eq!(after.live_persistent_bytes, before.live_persistent_bytes);
    assert!(after.work_reserved > before.work_reserved);
    assert!(build_prepared_terminal(&src, 0, &[2, 1], Some(at())).is_err());
}

#[test]
fn prepared_terminal_expired_before_entry_and_after_complete_margin_tile_cannot_publish_or_renew() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let src = source(fixture(false), &budget, None);
    let before = budget.receipt().unwrap();
    assert!(matches!(
        build_prepared_terminal(&src, 0, &[2, 1], Some(Instant::now())),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    let bytes = live(&memory);
    let mut after_margin = false;
    let limit = Instant::now() + Duration::from_secs(2);
    let mut hook = |receipt: &PreparedTerminalReceipt| {
        if receipt.margin_columns > 0 {
            after_margin = true;
            // A deliberately late private observer demonstrates the final
            // post-production checkpoint. No blocking wait is used.
            while Instant::now() < limit {
                std::hint::spin_loop();
            }
        }
    };
    assert!(matches!(
        build_inner(&src, 0, &[2, 1], Some(limit), Some(&mut hook)),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert!(after_margin);
    assert_eq!(live(&memory), bytes);
}

#[test]
fn prepared_terminal_complete_signed_difference_residuals_and_subnormal_seed_products_are_paid() {
    let _env = crate::tests::lock_env_shared();
    let (q, error) = difference(1.0, 2f64.powi(-60)).unwrap();
    assert_eq!(q, 1.0);
    assert_eq!(error, 2f64.powi(-60));
    assert_eq!(r(q) - r(error), r(1.0) - r(2f64.powi(-60)));
    let min = f64::from_bits(1);
    for a in [min, -min] {
        let (center, err) = product_center_error(a, 0.5).unwrap();
        let exact = r(a) * r(0.5);
        assert!(r(center) - r(err) <= exact && r(center) + r(err) >= exact);
        assert!(err > 0.0);
    }
    assert_eq!(
        product_center_error(-0.0, 0.5).unwrap().0.to_bits(),
        (-0.0f64).to_bits()
    );
    let (budget, _, _) = resources(CAP, None);
    let mut spec = fixture(false);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[5] {
        weight[0] = 1.0;
        weight[3] = 2f64.powi(-60);
        bias[0] = 1.0;
        bias[1] = 2f64.powi(-60);
    }
    let src = source(spec.clone(), &budget, None);
    let result = build_prepared_terminal(&src, 0, &[2, 1], Some(at())).unwrap();
    assert!(result.receipt().parameter_difference_residual_terms >= 2);
    exact_corners(&spec, false, &result);
}

#[test]
fn prepared_terminal_upper_head_lines_dominate_original_relu_at_rational_kinks_extreme_and_subnormal_endpoints(
) {
    let _env = crate::tests::lock_env_shared();
    for (lo, hi) in [
        (-2.0, 1.0),
        (-1.0, 2.0),
        (-f64::MAX, f64::MAX),
        (-f64::from_bits(1), f64::from_bits(1)),
        (-0.0, 1.0),
        (-1.0, 0.0),
    ] {
        let line = head_line(lo, hi).unwrap();
        for y in [lo, 0.0, hi] {
            let exact = r(y).max(BigRational::zero());
            assert!(r(line.slope) * r(y) + r(line.intercept) >= exact);
            assert!(r(line.alpha) * r(y) <= exact);
        }
    }
}

#[test]
fn prepared_terminal_nonfinite_exact_source_difference_and_nominal_overflow_refuse_without_leaking()
{
    let _env = crate::tests::lock_env_shared();
    assert!(difference(f64::MAX, -f64::MAX).is_err());
    assert!(product_center_error(f64::MAX, 2.0).is_err());
    assert!(head_line(f64::NAN, 1.0).is_err());
    let (budget, memory, _) = resources(CAP, None);
    let mut spec = fixture(false);
    if let TwinOpSpec::Gemm { bias, .. } = &mut spec.ops[5] {
        bias[0] = f64::MAX;
        bias[1] = -f64::MAX;
    }
    let src = source(spec, &budget, None);
    let bytes = live(&memory);
    assert!(build_prepared_terminal(&src, 0, &[2, 1], Some(at())).is_err());
    assert_eq!(live(&memory), bytes);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
struct DirectedRounding {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl DirectedRounding {
    #[allow(unsafe_code)]
    fn enter() -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr",saved=out(reg)saved,options(nomem,nostack));
            let directed = (saved & !(3_u64 << 22)) | (1_u64 << 22);
            std::arch::asm!("msr fpcr, {directed}","isb",directed=in(reg)directed,options(nomem,nostack));
            Self { saved }
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let saved = std::arch::x86_64::_mm_getcsr();
            std::arch::x86_64::_mm_setcsr((saved & !(3 << 13)) | (1 << 13));
            Self { saved }
        }
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl Drop for DirectedRounding {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            std::arch::asm!("msr fpcr, {saved}","isb",saved=in(reg)self.saved,options(nomem,nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn prepared_terminal_unqualified_environment_refuses_before_admission_and_live_owner_validation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let src = source(fixture(false), &budget, None);
    let terminal = build_prepared_terminal(&src, 0, &[2, 1], Some(at())).unwrap();
    let bytes = memory.receipt().unwrap();
    let before = budget.receipt().unwrap();
    {
        let _directed = DirectedRounding::enter();
        assert!(!ny_core::has_f64_interval_proof_environment());
        assert!(matches!(
            build_prepared_terminal(&src, 0, &[2, 1], Some(at())),
            Err(NyError::SoundnessRefusal(_))
        ));
        assert!(matches!(
            terminal.validate_source(&src, 0, &[2, 1], Some(at())),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    assert_eq!(memory.receipt().unwrap(), bytes);
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    terminal
        .validate_source(&src, 0, &[2, 1], Some(at()))
        .unwrap();
}
