// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual ordinary kernel, whole-original rational and shared-owner controls.

use super::super::{
    build_prepared_root, project_prepared_root, share_prepared_root, OriginalModel, OriginalRoot,
    PreparedConstructionOptions,
};
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::engine::BackwardEngine;
use crate::margin_row::root::RetainCfg;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::{BackwardIntermRuntimePolicy, RootPrecisionRuntimePolicy};
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 128 << 20;
fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources(
    cap: usize,
    original: Option<Instant>,
) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(2 * cap);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: original,
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
fn r(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}
fn bits(v: impl IntoIterator<Item = f64>) -> Vec<u64> {
    v.into_iter().map(f64::to_bits).collect()
}

fn fixture(fold: bool, heads: usize) -> TwinSpec {
    let rho = if fold { 1.0 / 256.0 } else { 0.0 };
    let error = if fold { 1.0 / 512.0 } else { 0.0 };
    let mut head_weight = Vec::with_capacity(2 * heads);
    let mut head_bias = Vec::with_capacity(heads);
    for j in 0..heads {
        head_weight.extend(if j % 2 == 0 {
            [0.5, -0.25]
        } else {
            [-0.125, 0.75]
        });
        head_bias.push(if j % 2 == 0 { 0.5 } else { -0.25 });
    }
    TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.25, -0.5, -0.25, 0.75],
                bias: vec![-0.25, 0.125],
                bias_err: vec![error; 2],
                weight_rel_err: rho,
                kernel: (2, 2, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (2, 1, 1),
                oshape: (2, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![-0.5, 0.25],
                shift: vec![0.125, -0.25],
                shift_err: vec![error; 2],
                scale_rel_err: rho,
                shape: (2, 1, 1),
            },
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: head_weight,
                bias: head_bias,
                bias_err: if fold { vec![error; heads] } else { vec![] },
                weight_rel_err: rho,
                shape: (heads, 2),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![0.125; 3 * heads],
                bias: vec![0.5, 0.0, -0.25],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (3, heads),
            },
        ],
    }
}
fn adapter(
    spec: TwinSpec,
    budget: &WorkBudget,
    original: Option<Instant>,
    products: bool,
) -> PreparedRootAdapter {
    let model = OriginalModel::compile(spec, budget, original.or(Some(at()))).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-1.0, -0.5],
        &[0.75, 1.0],
        RootPrecisionRuntimePolicy::default(),
        original.or(Some(at())),
    )
    .unwrap();
    let construction = build_prepared_root(
        root,
        PreparedConstructionOptions {
            keep_input_rows: products,
            prefer_patch: false,
            ..PreparedConstructionOptions::default()
        },
        Some(at()),
    )
    .unwrap();
    let source = share_prepared_root(construction, Some(at())).unwrap();
    project_prepared_root(
        source,
        RootPrecisionRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
        products,
        products.then(RetainCfg::default),
        Some(at()),
    )
    .unwrap()
}

fn exact(x: [f64; 2], signs: usize, fold: bool) -> [BigRational; 2] {
    let sign = |i: usize| {
        if signs & (1 << i) == 0 {
            r(-1.0)
        } else {
            r(1.0)
        }
    };
    let rho = r(if fold { 1.0 / 256.0 } else { 0.0 });
    let eps = r(if fold { 1.0 / 512.0 } else { 0.0 });
    let mut h = [BigRational::zero(), BigRational::zero()];
    let w = [[1.25, -0.5], [-0.25, 0.75]];
    let b = [-0.25, 0.125];
    for c in 0..2 {
        h[c] = (r(w[c][0]) * (r(1.0) + sign(0) * &rho) * r(x[0])
            + r(w[c][1]) * (r(1.0) + sign(1) * &rho) * r(x[1])
            + r(b[c])
            + sign(2) * &eps)
            .max(BigRational::zero());
    }
    let mut z = [BigRational::zero(), BigRational::zero()];
    for c in 0..2 {
        z[c] = (r([-0.5, 0.25][c]) * (r(1.0) + sign(3) * &rho) * &h[c]
            + r([0.125, -0.25][c])
            + sign(4) * &eps
            + &h[c])
            .max(BigRational::zero());
    }
    [
        r(0.5) * (r(1.0) + sign(5) * &rho) * &z[0]
            + r(-0.25) * (r(1.0) + sign(6) * &rho) * &z[1]
            + r(0.5)
            + sign(7) * &eps,
        r(-0.125) * (r(1.0) + sign(5) * &rho) * &z[0] + r(0.75) * (r(1.0) + sign(6) * &rho) * &z[1]
            - r(0.25)
            + sign(7) * &eps,
    ]
}

#[test]
fn prepared_ordinary_rows_whole_original_fold_family_and_all_input_corners_have_rational_support() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let original = adapter(fixture(true, 2), &budget, None, true);
    let rows = build_prepared_ordinary_rows(original.clone(), None).unwrap();
    rows.validate_adapter(&original, None).unwrap();
    let (lo, hi) = rows.box_bounds(None).unwrap();
    for x0 in [-1.0, 0.0, 0.75] {
        for x1 in [-0.5, 0.0, 1.0] {
            for signs in 0..256 {
                let y = exact([x0, x1], signs, true);
                for j in 0..2 {
                    assert!(r(lo[j]) <= y[j] && y[j] <= r(hi[j]));
                    let l = rows.lower(j, None).unwrap();
                    let u = rows.upper(j, None).unwrap();
                    let line = |view: &PreparedOrdinaryRowView<'_>| {
                        let mut value = r(view.bias());
                        let mut eta = r(view.bias_error());
                        for (i, a, e) in view.entries() {
                            value += r(a) * r([x0, x1][i]);
                            eta += r(e) * r([x0, x1][i].abs());
                        }
                        (value, eta)
                    };
                    let (lv, le) = line(&l);
                    let (uv, ue) = line(&u);
                    assert!(lv - le <= y[j] && y[j] <= uv + ue);
                }
            }
        }
    }
    assert!(original.clip_rows().is_some() && original.retained_rows().is_some());
    assert!(rows.adapter().same_owner(&original));
    assert_eq!(rows.receipt().lower_columns, 2);
    assert_eq!(rows.receipt().upper_columns, 2);
    assert_eq!(rows.receipt().stored_coefficient_cells, 8);
    assert_eq!(rows.receipt().stored_error_cells, 8);
    assert_eq!(rows.receipt().concretized_columns, 4);
}

#[test]
fn prepared_ordinary_rows_actual_ordinary_kernel_and_concretization_bits_are_identical() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let original = adapter(fixture(true, 7), &budget, None, false);
    let rows = build_prepared_ordinary_rows(original.clone(), None).unwrap();
    let p = plan(&original, at()).unwrap();
    let mut walk = admit(
        &budget,
        WorkRequest {
            scratch_bytes: p.tile_fixed + 7 * p.tile_per_column,
            work: 2 * p.work_per_column * 7,
            ..WorkRequest::default()
        },
        at(),
    )
    .unwrap();
    let engine = BackwardEngine::new(original.net(), original.gates());
    let (lower, upper) = engine.y_rows(None).unwrap();
    for (is_lower, actual) in [(true, &lower), (false, &upper)] {
        let (a, e, b, eb) = rows.matrices(is_lower, None).unwrap();
        assert_eq!(bits(a.iter().copied()), bits(actual.a.iter().copied()));
        assert_eq!(
            bits(e.iter().copied()),
            bits(actual.e.as_ref().unwrap().iter().copied())
        );
        assert_eq!(bits(b.iter().copied()), bits(actual.b.iter().copied()));
        assert_eq!(bits(eb.iter().copied()), bits(actual.eb.iter().copied()));
    }
    let (lo, hi) = rows.box_bounds(None).unwrap();
    assert_eq!(
        bits(lo.iter().copied()),
        bits(engine.concretize_lower(&lower))
    );
    assert_eq!(
        bits(hi.iter().copied()),
        bits(engine.concretize_upper(&upper))
    );
    drop((lower, upper));
    walk.complete().unwrap();
}

#[test]
fn prepared_ordinary_rows_complete_head_above_legacy_512_uses_actual_bounded_tiles() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let original = adapter(fixture(false, 513), &budget, None, false);
    let rows = build_prepared_ordinary_rows(original, None).unwrap();
    let receipt = rows.receipt();
    assert_eq!(receipt.lower_columns, 513);
    assert_eq!(receipt.upper_columns, 513);
    assert_eq!(receipt.tiles, 9);
    assert_eq!(receipt.backward_passes, 18);
    assert_eq!(receipt.max_tile_columns, 64);
    assert_eq!(receipt.concretized_columns, 1026);
    for j in 0..513 {
        assert_eq!(rows.lower(j, None).unwrap().original_output(), j);
        assert_eq!(
            rows.upper(j, None)
                .unwrap()
                .entries()
                .map(|(i, _, _)| i)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }
    let (lo, hi) = rows.box_bounds(None).unwrap();
    let y = exact([0.75, -0.5], 0, false);
    for j in 0..513 {
        assert!(r(lo[j]) <= y[j % 2] && y[j % 2] <= r(hi[j]));
    }
}

#[test]
fn prepared_ordinary_rows_tile_width_shrinks_before_buffers_and_preserves_complete_columns() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let original = adapter(fixture(false, 65), &budget, None, false);
    let p = plan(&original, at()).unwrap();
    let held = budget.receipt().unwrap().live_persistent_bytes;
    let keep = p.persistent + p.tile_fixed + 8 * p.tile_per_column;
    let mut occupied = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP - held - keep,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    occupied.mark_materialized().unwrap();
    let occupied = occupied.commit_persistent().unwrap();
    let before = live(&memory);
    // Original budget has separate persistent and scratch maxima, so use its
    // monotone combined ceiling to make the complete source overlap decisive.
    let current = budget.receipt().unwrap();
    budget
        .tighten_total_byte_limit(current.live_persistent_bytes + keep)
        .unwrap();
    let rows = build_prepared_ordinary_rows(original, None).unwrap();
    assert!(rows.receipt().tile_admission_attempts > rows.receipt().tiles);
    assert!(rows.receipt().max_tile_columns <= 8);
    assert_eq!(rows.receipt().lower_columns, 65);
    assert_eq!(rows.receipt().upper_columns, 65);
    assert_eq!(live(&memory), before + rows.resident_bytes());
    drop((rows, occupied));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_ordinary_rows_one_byte_short_owner_refuses_before_output_allocation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let original = adapter(fixture(false, 2), &budget, None, false);
    let current = budget.receipt().unwrap();
    let bytes = owner_bytes(2, 2).unwrap();
    assert!(bytes > META);
    budget
        .tighten_total_byte_limit(current.live_persistent_bytes + bytes - 1)
        .unwrap();
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    assert!(build_prepared_ordinary_rows(original.clone(), None).is_err());
    assert_eq!(live(&memory), before);
    // The paid header census succeeds; complete output owner never admits.
    assert_eq!(budget.receipt().unwrap().admitted, admitted + 1);
    original.checkpoint(None).unwrap();
}

#[test]
fn prepared_ordinary_rows_cancellation_after_actual_tile_publishes_no_partial_memo() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources(CAP, None);
    let original = adapter(fixture(false, 65), &budget, None, false);
    let before = live(&memory);
    let mut seen = 0;
    let mut observe = |tiles| {
        seen = tiles;
        cancel.cancel();
    };
    assert!(build_inner(original.clone(), None, Some(&mut observe)).is_err());
    assert_eq!(seen, 1);
    assert_eq!(live(&memory), before);
    assert!(budget.receipt().unwrap().work_reserved > 0);
    assert!(original.checkpoint(None).is_err());
}

#[test]
fn prepared_ordinary_rows_cancelled_original_and_expired_request_refuse_before_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources(CAP, None);
    let original = adapter(fixture(false, 2), &budget, None, false);
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    let expired = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    assert!(matches!(
        build_prepared_ordinary_rows(original.clone(), Some(expired)),
        Err(NyError::DeadlineExceeded(_))
    ));
    cancel.cancel();
    assert!(build_prepared_ordinary_rows(original, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    // The moved projection is dropped, so source claims correctly disappear.
    assert!(live(&memory) < before);
}

#[test]
fn prepared_ordinary_rows_same_numeric_model_or_equal_limits_cannot_rebind_memo() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let (foreign, _, _) = resources(CAP, None);
    let original = adapter(fixture(false, 2), &budget, None, false);
    let same_ledger_other = adapter(fixture(false, 2), &budget, None, false);
    let other_ledger = adapter(fixture(false, 2), &foreign, None, false);
    let rows = build_prepared_ordinary_rows(original.clone(), None).unwrap();
    rows.validate_adapter(&original, None).unwrap();
    assert!(rows.validate_adapter(&same_ledger_other, None).is_err());
    assert!(rows.validate_adapter(&other_ledger, None).is_err());
    assert!(rows.lower(2, None).is_err());
    assert!(rows.upper(usize::MAX, None).is_err());
}

#[test]
fn prepared_ordinary_rows_producer_request_minimum_is_retained_after_source_request_ends() {
    let _env = crate::tests::lock_env_shared();
    let original_at = at();
    let short = original_at.checked_sub(Duration::from_secs(20)).unwrap();
    let (budget, _, _) = resources(CAP, Some(original_at));
    let original = adapter(fixture(false, 2), &budget, Some(original_at), false);
    let rows = build_prepared_ordinary_rows(original, Some(short)).unwrap();
    assert_eq!(rows.deadline(None).unwrap(), short);
    assert_eq!(rows.deadline(Some(at())).unwrap(), short);
    let earlier = short.checked_sub(Duration::from_secs(5)).unwrap();
    assert_eq!(rows.deadline(Some(earlier)).unwrap(), earlier);
    assert_eq!(rows.deadline(None).unwrap(), short);
}

#[test]
fn prepared_ordinary_rows_clone_and_matrix_views_keep_one_paid_last_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let original = adapter(fixture(false, 2), &budget, None, true);
    let before = live(&memory);
    let rows = build_prepared_ordinary_rows(original.clone(), None).unwrap();
    assert_eq!(live(&memory), before + rows.resident_bytes());
    let after = budget.receipt().unwrap();
    let alias = rows.clone();
    assert!(rows.same_owner(&alias));
    assert_eq!(budget.receipt().unwrap(), after);
    {
        let (a, e, _, _) = alias.matrices(true, None).unwrap();
        let (a2, e2, _, _) = rows.matrices(true, None).unwrap();
        assert_eq!(a.as_ptr(), a2.as_ptr());
        assert_eq!(e.as_ptr(), e2.as_ptr());
    }
    drop((rows, original));
    assert!(live(&memory) > 0);
    alias.checkpoint(None).unwrap();
    drop(alias);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_ordinary_rows_work_refusal_is_not_retried_as_memory_or_given_fresh_budget() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let original = adapter(fixture(false, 2), &budget, None, false);
    let r0 = budget.receipt().unwrap();
    let remaining = 100_000_000_000_000 - r0.work_reserved;
    let mut hold = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                work: remaining,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    hold.complete().unwrap();
    drop(hold);
    let before = live(&memory);
    let attempted = budget.receipt().unwrap().attempted;
    assert!(build_prepared_ordinary_rows(original.clone(), None).is_err());
    assert_eq!(budget.receipt().unwrap().attempted, attempted + 1);
    assert_eq!(live(&memory), before);
    original.checkpoint(None).unwrap();
}

#[test]
fn prepared_ordinary_rows_zero_coefficients_errors_and_signed_zero_inputs_remain_finite() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let mut spec = fixture(false, 2);
    if let TwinOpSpec::Conv { weight, bias, .. } = &mut spec.ops[0] {
        weight.fill(-0.0);
        bias.fill(-0.0);
    }
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[6] {
        weight.fill(0.0);
        bias.copy_from_slice(&[-0.0, 0.25]);
    }
    let rows = build_prepared_ordinary_rows(adapter(spec, &budget, None, false), None).unwrap();
    let (lo, hi) = rows.box_bounds(None).unwrap();
    assert!(lo[0] <= 0.0 && 0.0 <= hi[0]);
    assert!(lo[1] <= 0.25 && 0.25 <= hi[1]);
    for is_lower in [true, false] {
        let (a, e, _, eb) = rows.matrices(is_lower, None).unwrap();
        assert!(a.iter().all(|v| *v == 0.0));
        assert!(e.iter().chain(eb).all(|v| v.is_finite() && *v >= 0.0));
    }
}

#[test]
fn prepared_ordinary_rows_numerical_subnormal_products_do_not_omit_original_support() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources(CAP, None);
    let mut spec = fixture(false, 2);
    let tiny = f64::from_bits(1);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[6] {
        weight.copy_from_slice(&[tiny, -tiny, -tiny, tiny]);
        bias.fill(0.0);
    }
    let original = adapter(spec, &budget, None, false);
    let rows = build_prepared_ordinary_rows(original, None).unwrap();
    let (lo, hi) = rows.box_bounds(None).unwrap();
    for x0 in [-1.0, 0.75] {
        for x1 in [-0.5, 1.0] {
            let h0 = (r(1.25) * r(x0) - r(0.5) * r(x1) - r(0.25)).max(BigRational::zero());
            let h1 = (r(-0.25) * r(x0) + r(0.75) * r(x1) + r(0.125)).max(BigRational::zero());
            let z0 = (r(0.5) * h0 + r(0.125)).max(BigRational::zero());
            let z1 = (r(1.25) * h1 - r(0.25)).max(BigRational::zero());
            let y = r(tiny) * (z0 - z1);
            assert!(r(lo[0]) <= y && y <= r(hi[0]));
            assert!(r(lo[1]) <= -&y && -&y <= r(hi[1]));
        }
    }
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
            std::arch::asm!("mrs {saved}, fpcr", saved = out(reg) saved, options(nomem, nostack));
            let directed = (saved & !(3 << 22)) | (1 << 22);
            std::arch::asm!("msr fpcr, {directed}", "isb", directed = in(reg) directed, options(nomem, nostack));
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
            std::arch::asm!("msr fpcr, {saved}", "isb", saved = in(reg) self.saved, options(nomem, nostack));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_setcsr(self.saved);
        }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
fn prepared_ordinary_rows_unqualified_environment_refuses_before_new_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources(CAP, None);
    let original = adapter(fixture(false, 2), &budget, None, false);
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    {
        let _rounding = DirectedRounding::enter();
        assert!(!ny_core::has_f64_interval_proof_environment());
        assert!(matches!(
            build_prepared_ordinary_rows(original.clone(), None),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    original.checkpoint(None).unwrap();
}
