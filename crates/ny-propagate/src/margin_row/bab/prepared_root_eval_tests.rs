// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Genuine original owners through actual YPack, RootEval and paid weak memo.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::support_program::{
    build_prepared_ordinary_rows, build_prepared_root, project_prepared_root, share_prepared_root,
    OriginalModel, OriginalRoot, PreparedConstructionOptions,
};
use crate::margin_row::{BackwardIntermRuntimePolicy, RootPrecisionRuntimePolicy};
use num_rational::BigRational;
use num_traits::Zero;

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
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn r(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}
fn fixture(heads: usize, classes: usize) -> TwinSpec {
    let mut hw = Vec::new();
    let mut hb = Vec::new();
    for k in 0..heads {
        hw.extend(if k % 2 == 0 {
            [1.0, -0.5]
        } else {
            [-0.25, 0.75]
        });
        hb.push(0.125);
    }
    let mut w = Vec::new();
    let mut b = Vec::new();
    for j in 0..classes {
        for k in 0..heads {
            w.push(if j == 0 {
                if k % 2 == 0 {
                    0.5
                } else {
                    -0.25
                }
            } else if j % 2 == 1 {
                if k % 2 == 0 {
                    -0.125
                } else {
                    0.5
                }
            } else {
                0.25
            });
        }
        b.push(if j == 0 { 1.0 } else { 0.0 });
    }
    TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0, 0.0, 0.0, 1.0],
                bias: vec![0.0; 2],
                bias_err: vec![0.0; 2],
                weight_rel_err: 0.0,
                kernel: (2, 2, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (2, 1, 1),
                oshape: (2, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: hw,
                bias: hb,
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (heads, 2),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: w,
                bias: b,
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (classes, heads),
            },
        ],
    }
}
fn adapter(spec: TwinSpec, budget: &WorkBudget, products: bool) -> PreparedRootAdapter {
    let at = clock();
    let model = OriginalModel::compile(spec, budget, Some(at)).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-1.0, -1.0],
        &[1.0, 1.0],
        RootPrecisionRuntimePolicy::default(),
        Some(at),
    )
    .unwrap();
    let source = build_prepared_root(
        root,
        PreparedConstructionOptions {
            keep_input_rows: products,
            prefer_patch: false,
            ..PreparedConstructionOptions::default()
        },
        Some(at),
    )
    .unwrap();
    project_prepared_root(
        share_prepared_root(source, Some(at)).unwrap(),
        RootPrecisionRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
        products,
        products.then(super::super::super::root::RetainCfg::default),
        Some(at),
    )
    .unwrap()
}
fn pointers(al: &PassOut, au: &PassOut, lo: &[f64], hi: &[f64]) -> [usize; 10] {
    [
        al.a.as_ptr() as usize,
        al.e.as_ref().unwrap().as_ptr() as usize,
        al.b.as_ptr() as usize,
        al.eb.as_ptr() as usize,
        au.a.as_ptr() as usize,
        au.e.as_ref().unwrap().as_ptr() as usize,
        au.b.as_ptr() as usize,
        au.eb.as_ptr() as usize,
        lo.as_ptr() as usize,
        hi.as_ptr() as usize,
    ]
}
fn exact_margin(x: [f64; 2], heads: usize, class: usize) -> BigRational {
    let h = [
        r(x[0]).max(BigRational::zero()),
        r(x[1]).max(BigRational::zero()),
    ];
    let mut result = r(1.0);
    for k in 0..heads {
        let y = if k % 2 == 0 {
            &h[0] - r(0.5) * &h[1] + r(0.125)
        } else {
            -r(0.25) * &h[0] + r(0.75) * &h[1] + r(0.125)
        };
        let z = y.max(BigRational::zero());
        let target = if k % 2 == 0 { 0.5 } else { -0.25 };
        let adv = if class % 2 == 1 {
            if k % 2 == 0 {
                -0.125
            } else {
                0.5
            }
        } else {
            0.25
        };
        result += (r(target) - r(adv)) * z;
    }
    result
}

#[test]
fn prepared_eval_unique_transport_moves_all_ten_actual_allocations_and_one_claim() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let ptrs = rows.storage_pointers();
    let before = live(&memory);
    let admitted = budget.receipt().unwrap().admitted;
    let moved = rows
        .with_root_rows(None, Ok, |_| panic!("unique producer"))
        .unwrap();
    let (al, au, lo, hi, owner) = moved.into_parts();
    assert_eq!(ptrs, pointers(&al, &au, &lo, &hi));
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    owner.checkpoint(None).unwrap();
    drop((al, au, lo, hi, owner, a));
    assert_eq!(live(&memory), 0);
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn prepared_eval_shared_transport_keeps_earlier_clock_and_admits_real_independent_copy() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let original = rows.clone();
    let ptrs = rows.storage_pointers();
    let earlier = Instant::now() + Duration::from_secs(20);
    let shared = rows
        .with_root_rows(Some(earlier), |_| panic!("shared source"), Ok)
        .unwrap();
    assert_eq!(shared.deadline(None).unwrap(), earlier);
    assert_eq!(shared.clone().deadline(Some(clock())).unwrap(), earlier);
    let before = live(&memory);
    let moved = shared.copy_into_root_rows(None).unwrap();
    let (al, au, lo, hi, owner) = moved.into_parts();
    let copies = pointers(&al, &au, &lo, &hi);
    for (old, new) in ptrs.into_iter().zip(copies) {
        assert_ne!(old, new);
    }
    assert_eq!(owner.deadline(None).unwrap(), earlier);
    assert!(live(&memory) > before);
    assert_eq!(original.box_bounds(None).unwrap().0, lo);
    assert_eq!(original.box_bounds(None).unwrap().1, hi);
    for j in 0..2 {
        for (i, v, e) in original.lower(j, None).unwrap().entries() {
            assert_eq!(v.to_bits(), al.a[[i, j]].to_bits());
            assert_eq!(e.to_bits(), al.e.as_ref().unwrap()[[i, j]].to_bits());
        }
    }
    drop((al, au, lo, hi, owner));
    assert_eq!(live(&memory), before);
    drop((original, a));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_eval_actual_three_component_max_is_below_signed_whole_model_rationals() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let a = adapter(fixture(2, 3), &budget, true);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let result = build_prepared_root_eval(rows, 0, &[2, 1], None).unwrap();
    assert!(result.adapter().same_owner(&a));
    assert!(result.adapter().clip_rows().is_some());
    assert!(result.adapter().retained_rows().is_some());
    assert_eq!(result.adversaries(), [2, 1]);
    assert_eq!(result.receipt().classes, 2);
    for x0 in [-1.0, 0.0, 0.25, 1.0] {
        for x1 in [-1.0, 0.0, 0.5, 1.0] {
            for (i, &class) in result.adversaries().iter().enumerate() {
                assert!(r(result.bounds(None).unwrap()[i]) <= exact_margin([x0, x1], 2, class));
            }
        }
    }
    let raw = result.legacy_view(None).unwrap();
    assert_eq!(raw.dj, result.bounds(None).unwrap());
    assert!(raw.pack.prepared_owner.is_some());
    assert!(raw.pack.head_group_source.is_none());
    assert_eq!(raw.pack.al.a.dim(), (2, 2));
    assert_eq!(raw.pack.au.e.as_ref().unwrap().dim(), (2, 2));
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn prepared_eval_actual_memo_hit_reuses_result_and_pack_and_keeps_exact_order() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let eng = BackwardEngine::new(a.net(), a.gates());
    let earlier = Instant::now() + Duration::from_secs(20);
    let hit = prepared_root_eval_lookup(&eng, 0, &[1, 2], Some(earlier))
        .unwrap()
        .unwrap();
    assert!(result.same_owner(&hit));
    assert!(Arc::ptr_eq(result.pack(), hit.pack()));
    assert_eq!(hit.deadline(None).unwrap(), earlier);
    assert_eq!(hit.clone().deadline(Some(clock())).unwrap(), earlier);
    assert!(prepared_root_eval_lookup(&eng, 0, &[2, 1], None)
        .unwrap()
        .is_none());
    assert!(prepared_root_eval_lookup(&eng, 1, &[0, 2], None)
        .unwrap()
        .is_none());
    let shared = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let held = shared.clone();
    let reused = prepared_root_eval_reuse(shared, 0, &[1, 2], None).unwrap();
    assert!(reused.same_owner(&result));
    assert!(held.checkpoint(None).is_ok());
}

#[test]
fn prepared_eval_new_objective_reuses_guarded_pack_without_copying_shared_payload() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let first = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let rows = build_prepared_ordinary_rows(a, None).unwrap();
    let held = rows.clone();
    let second = build_prepared_root_eval(rows, 0, &[2, 1], None).unwrap();
    assert!(Arc::ptr_eq(first.pack(), second.pack()));
    assert!(second.receipt().cached_pack);
    assert!(!second.receipt().shared_payload_copy);
    held.checkpoint(None).unwrap();
}

#[test]
fn prepared_eval_first_shared_ingress_has_paid_copy_and_keeps_old_owner_valid() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a, None).unwrap();
    let held = rows.clone();
    let ptrs = held.storage_pointers();
    let result = build_prepared_root_eval(rows, 0, &[1, 2], None).unwrap();
    assert!(result.receipt().shared_payload_copy);
    assert!(!result.receipt().cached_pack);
    assert_ne!(ptrs[0], result.pack().al.a.as_ptr() as usize);
    held.checkpoint(None).unwrap();
}

#[test]
#[expect(
    clippy::redundant_clone,
    reason = "Exercise clone propagation of deadline, cancellation and shared accounting ownership."
)]
fn prepared_eval_paid_ybox_and_memo_rows_retain_actual_last_source_and_earliest_clock() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let earlier = Instant::now() + Duration::from_secs(20);
    let ybox = result.copy_ybox(Some(earlier)).unwrap();
    let rows = result.rows(Some(earlier)).unwrap();
    assert_eq!(rows.evaluation().deadline(None).unwrap(), earlier);
    let expected = result.pack().ly0.clone();
    drop((result, a));
    assert_eq!(ybox.0.deadline, earlier);
    assert_eq!(ybox.clone().view(Some(clock())).unwrap().ly, expected);
    assert_eq!(rows.al(None).unwrap().a.ncols(), 2);
    assert!(live(&memory) > 0);
    drop((rows, ybox));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_eval_two_weak_slots_preserve_control_block_claims_without_source_cycle() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let base = live(&memory);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let rows = result.rows(None).unwrap();
    drop(result);
    assert!(live(&memory) > base);
    drop(rows);
    assert!(live(&memory) > base); // genuine weak Arc/header claims remain.
    a.gates().root_eval_memo.clear();
    assert_eq!(live(&memory), base);
    drop(a);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_eval_foreign_original_projection_and_nonpercol_engine_never_hit_memo() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let foreign = adapter(fixture(2, 3), &budget, false);
    let eng = BackwardEngine::new(foreign.net(), foreign.gates());
    assert!(prepared_root_eval_lookup(&eng, 0, &[1, 2], None)
        .unwrap()
        .is_none());
    let altered = BackwardEngine::new(a.net(), a.gates())
        .with_err_lane(super::super::super::errlane::ErrLane::Envelope);
    assert!(prepared_root_eval_lookup(&altered, 0, &[1, 2], None)
        .unwrap()
        .is_none());
    result.checkpoint(None).unwrap();
}

#[test]
fn prepared_eval_cancel_after_actual_completed_tile_publishes_no_result_or_memo() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let a = adapter(fixture(2, 66), &budget, false);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let base = live(&memory) - rows.resident_bytes();
    let mut observed = 0;
    let mut observer = |tile| {
        observed = tile;
        cancel.cancel();
    };
    assert!(build_inner(
        rows,
        0,
        &(1..66).collect::<Vec<_>>(),
        None,
        Some(&mut observer)
    )
    .is_err());
    assert_eq!(observed, 1);
    assert_eq!(live(&memory), base);
    assert!(a
        .gates()
        .root_eval_memo
        .1
        .lock()
        .unwrap()
        .slots
        .iter()
        .all(Option::is_none));
}

#[test]
fn prepared_eval_expired_original_handoff_refuses_before_new_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a, None).unwrap();
    let before = budget.receipt().unwrap().admitted;
    assert!(build_prepared_root_eval(
        rows,
        0,
        &[1, 2],
        Some(
            Instant::now()
                .checked_sub(Duration::from_millis(1))
                .unwrap()
        )
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before);
}

#[test]
fn prepared_eval_zero_additional_capacity_refuses_without_losing_shared_original_rows() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let held = rows.clone();
    let before = live(&memory);
    budget.tighten_total_byte_limit(before).unwrap();
    assert!(build_prepared_root_eval(rows, 0, &[1, 2], None).is_err());
    assert_eq!(live(&memory), before);
    held.checkpoint(None).unwrap();
    a.checkpoint(None).unwrap();
}

#[test]
fn prepared_eval_original_513_head_and_65_classes_use_complete_bounded_tiles() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let a = adapter(fixture(513, 66), &budget, false);
    let rows = build_prepared_ordinary_rows(a, None).unwrap();
    let classes = (1..66).collect::<Vec<_>>();
    let result = build_prepared_root_eval(rows, 0, &classes, None).unwrap();
    assert_eq!(result.bounds(None).unwrap().len(), 65);
    assert_eq!(result.adversaries(), classes);
    assert_eq!(result.receipt().tiles, 2);
    assert_eq!(result.rows(None).unwrap().al(None).unwrap().a.ncols(), 513);
    for (i, &j) in classes.iter().enumerate() {
        assert!(r(result.bounds(None).unwrap()[i]) <= exact_margin([1.0, 1.0], 513, j));
    }
}

#[test]
fn prepared_eval_original_difference_negative_twosum_gap_is_charged_after_component_max() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let mut spec = fixture(2, 2);
    if let TwinOpSpec::Gemm { weight, bias, .. } = spec.ops.last_mut().unwrap() {
        *weight = vec![1.0, 0.0, 2.0_f64.powi(-54), 0.0];
        *bias = vec![0.0, 0.0];
    }
    let a = adapter(spec, &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a, None).unwrap(),
        0,
        &[1],
        None,
    )
    .unwrap();
    assert_eq!(result.receipt().difference_residual_terms, 1); // one complete functional correction after the max, not one per component.
    let exact = (r(1.0) - r(2.0_f64.powi(-54))) * r(0.125);
    assert!(r(result.bounds(None).unwrap()[0]) <= exact);
}

#[test]
fn prepared_eval_genuine_constant_subnormal_head_max_contains_negative_rational_margin() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let mut spec = fixture(64, 2);
    let tiny = f64::from_bits(1);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[3] {
        weight.fill(0.0);
        bias.fill(tiny);
    }
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[5] {
        weight[..64].fill(0.0);
        weight[64..].fill(0.5);
        bias[0] = 8.0 * tiny;
        bias[1] = 0.0;
    }
    let a = adapter(spec, &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a, None).unwrap(),
        0,
        &[1],
        None,
    )
    .unwrap();
    let exact = -r(24.0) * r(tiny);
    assert!(r(result.bounds(None).unwrap()[0]) <= exact);
    assert_eq!(result.tree_classes(None).unwrap(), [1]);
}

#[test]
fn prepared_eval_zero_margin_never_leaves_strict_original_class_pending_set() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let mut spec = fixture(2, 3);
    if let TwinOpSpec::Gemm { weight, bias, .. } = spec.ops.last_mut().unwrap() {
        weight.fill(0.0);
        bias.fill(0.0);
    }
    let a = adapter(spec, &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a, None).unwrap(),
        0,
        &[2, 1],
        None,
    )
    .unwrap();
    assert_eq!(result.tree_classes(None).unwrap(), [2, 1]);
    assert!(result.bounds(None).unwrap().iter().all(|v| *v <= 0.0));
}

#[test]
fn prepared_eval_actual_fold_uncertainty_family_retains_valid_whole_original_max() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let mut spec = fixture(2, 3);
    if let TwinOpSpec::Conv {
        weight_rel_err,
        bias_err,
        ..
    } = &mut spec.ops[0]
    {
        *weight_rel_err = 1.0 / 256.0;
        *bias_err = vec![1.0 / 512.0; 2];
    }
    let a = adapter(spec, &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a, None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    for x0 in [-1.0, 0.0, 1.0] {
        for x1 in [-1.0, 0.0, 1.0] {
            for signs in 0..16 {
                let signed = |i| {
                    if signs & (1 << i) == 0 {
                        r(-1.0)
                    } else {
                        r(1.0)
                    }
                };
                let h = [
                    (r(x0) * (r(1.0) + signed(0) * r(1.0 / 256.0)) + signed(1) * r(1.0 / 512.0))
                        .max(BigRational::zero()),
                    (r(x1) * (r(1.0) + signed(2) * r(1.0 / 256.0)) + signed(3) * r(1.0 / 512.0))
                        .max(BigRational::zero()),
                ];
                let z = [
                    (&h[0] - r(0.5) * &h[1] + r(0.125)).max(BigRational::zero()),
                    (-r(0.25) * &h[0] + r(0.75) * &h[1] + r(0.125)).max(BigRational::zero()),
                ];
                let expected = [
                    r(1.0) + r(0.625) * &z[0] - r(0.75) * &z[1],
                    r(1.0) + r(0.25) * &z[0] - r(0.5) * &z[1],
                ];
                for j in 0..2 {
                    assert!(r(result.bounds(None).unwrap()[j]) <= expected[j]);
                }
            }
        }
    }
}

#[test]
fn prepared_eval_shared_copy_one_byte_short_refuses_before_any_new_payload_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a, None).unwrap();
    let held = rows.clone();
    let before = live(&memory);
    let required = rows.resident_bytes();
    budget
        .tighten_total_byte_limit(before + required - 1)
        .unwrap();
    let admitted = budget.receipt().unwrap().admitted;
    assert!(rows.copy_into_root_rows(None).is_err());
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    held.checkpoint(None).unwrap();
}

#[test]
fn prepared_eval_paid_ybox_no_capacity_refuses_without_plain_vector_escape() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a, None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let before = live(&memory);
    budget.tighten_total_byte_limit(before).unwrap();
    let admitted = budget.receipt().unwrap().admitted;
    assert!(result.copy_ybox(None).is_err());
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    assert_eq!(result.bounds(None).unwrap().len(), 2);
}

fn slot_ids(a: &PreparedRootAdapter) -> [Option<usize>; 2] {
    let slots = a.gates().root_eval_memo.1.lock().unwrap();
    std::array::from_fn(|i| {
        slots.slots[i]
            .as_ref()
            .map(|s| s.evaluation.as_ptr() as usize)
    })
}

#[test]
fn prepared_eval_busy_memo_lookup_and_store_refuse_cooperatively_without_replacing_old_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let before = live(&memory);
    let ids = slot_ids(&a);
    let guard = a.gates().root_eval_memo.1.lock().unwrap();
    let eng = BackwardEngine::new(a.net(), a.gates());
    assert!(matches!(
        prepared_root_eval_lookup(&eng, 0, &[1, 2], None),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert!(matches!(
        store(&result),
        Err(NyError::UnsupportedConfiguration(_))
    ));
    assert_eq!(live(&memory), before);
    drop(guard);
    assert_eq!(slot_ids(&a), ids);
    result.checkpoint(None).unwrap();
}

#[test]
fn prepared_eval_cancel_after_actual_memo_insert_rolls_back_both_old_slots_and_claims() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let first = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let second = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[2, 1],
        None,
    )
    .unwrap();
    let ids = slot_ids(&a);
    let before = live(&memory);
    let mut observed = false;
    let mut observer = || {
        observed = true;
        cancel.cancel();
    };
    assert!(store_inner(&first, Some(&mut observer)).is_err());
    assert!(observed);
    assert_eq!(slot_ids(&a), ids);
    assert_eq!(live(&memory), before);
    assert!(first.checkpoint(None).is_err() && second.checkpoint(None).is_err());
    drop((first, second, a));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_eval_linear_bitmap_preserves_all_4097_original_vocabulary_ids_and_order() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = resources();
    let a = adapter(fixture(2, 4098), &budget, false);
    let classes = (1..4098).rev().collect::<Vec<_>>();
    let result = build_prepared_root_eval(
        build_prepared_ordinary_rows(a, None).unwrap(),
        0,
        &classes,
        None,
    )
    .unwrap();
    assert_eq!(result.adversaries(), classes);
    assert_eq!(result.bounds(None).unwrap().len(), 4097);
    assert_eq!(result.receipt().classes, 4097);
    assert_eq!(result.receipt().tiles, 65);
    for i in [0, 1, 2048, 4096] {
        assert!(r(result.bounds(None).unwrap()[i]) <= exact_margin([1.0, 1.0], 2, classes[i]));
    }
}

#[test]
fn prepared_eval_bitmap_duplicate_foreign_and_target_ids_refuse_without_mutating_prior_memo() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let first = build_prepared_root_eval(
        build_prepared_ordinary_rows(a.clone(), None).unwrap(),
        0,
        &[1, 2],
        None,
    )
    .unwrap();
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let ids = slot_ids(&a);
    let before = live(&memory);
    for malformed in [[1, 1], [1, usize::MAX], [0, 1]] {
        assert!(build_prepared_root_eval(rows.clone(), 0, &malformed, None).is_err());
        assert_eq!(slot_ids(&a), ids);
        assert_eq!(live(&memory), before);
    }
    rows.checkpoint(None).unwrap();
    first.checkpoint(None).unwrap();
}

#[test]
fn prepared_eval_consuming_callback_cancelled_source_runs_neither_branch() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancellation) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let called = std::cell::Cell::new(0usize);
    let before = budget.receipt().unwrap();
    cancellation.cancel();
    assert!(rows
        .with_root_rows(
            None,
            |_| {
                called.set(1);
                Ok(())
            },
            |_| {
                called.set(2);
                Ok(())
            },
        )
        .is_err());
    assert_eq!(called.get(), 0);
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.work_reserved, before.work_reserved);
    assert_eq!(after.proposals_reserved, before.proposals_reserved);
    drop(a);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_eval_consuming_unique_callback_error_releases_original_claim_once() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let original_pointers = rows.storage_pointers();
    let bytes = rows.resident_bytes();
    let before = live(&memory);
    let spent = budget.receipt().unwrap();
    let result = rows.with_root_rows(
        None,
        |moved| -> Result<()> {
            let (al, au, lo, hi, owner) = moved.into_parts();
            assert_eq!(pointers(&al, &au, &lo, &hi), original_pointers);
            owner.checkpoint(None).unwrap();
            drop((al, au, lo, hi, owner));
            Err(NyError::InvalidSpec("consuming callback refusal".into()))
        },
        |_| panic!("unique callback must retain atomic unwrap result"),
    );
    assert!(matches!(result, Err(NyError::InvalidSpec(_))));
    assert_eq!(live(&memory), before - bytes);
    assert_eq!(budget.receipt().unwrap().work_reserved, spent.work_reserved);
    assert_eq!(
        budget.receipt().unwrap().proposals_reserved,
        spent.proposals_reserved
    );
    drop(a);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_eval_consuming_shared_callback_error_preserves_last_original_alias() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let original = rows.clone();
    let earlier = Instant::now() + Duration::from_secs(20);
    let before = live(&memory);
    let spent = budget.receipt().unwrap();
    let result = rows.with_root_rows(
        Some(earlier),
        |_| panic!("shared callback must retain atomic unwrap result"),
        |shared| -> Result<()> {
            assert!(shared.same_owner(&original));
            assert_eq!(shared.storage_pointers(), original.storage_pointers());
            assert_eq!(shared.deadline(None).unwrap(), earlier);
            Err(NyError::InvalidSpec(
                "shared consuming callback refusal".into(),
            ))
        },
    );
    assert!(matches!(result, Err(NyError::InvalidSpec(_))));
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().admitted, spent.admitted);
    assert_eq!(budget.receipt().unwrap().work_reserved, spent.work_reserved);
    assert_eq!(
        budget.receipt().unwrap().proposals_reserved,
        spent.proposals_reserved
    );
    original.checkpoint(None).unwrap();
    drop((original, a));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_eval_consuming_unique_callback_unwind_releases_original_payload() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let a = adapter(fixture(2, 3), &budget, false);
    let rows = build_prepared_ordinary_rows(a.clone(), None).unwrap();
    let bytes = rows.resident_bytes();
    let before = live(&memory);
    let spent = budget.receipt().unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rows.with_root_rows(
            None,
            |_moved| -> Result<()> { panic!("consuming callback unwind") },
            |_| panic!("unique callback must retain atomic unwrap result"),
        )
    }));
    assert!(result.is_err());
    assert_eq!(live(&memory), before - bytes);
    assert_eq!(budget.receipt().unwrap().work_reserved, spent.work_reserved);
    assert_eq!(
        budget.receipt().unwrap().proposals_reserved,
        spent.proposals_reserved
    );
    drop(a);
    assert_eq!(live(&memory), 0);
}
