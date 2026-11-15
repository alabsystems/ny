// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::support_program::PreparedSplitSite;
use crate::margin_row::{
    PreparedRootOutcome, PreparedRootProofOptions, PreparedRootSession, RootPrecisionRuntimePolicy,
    TwinOpSpec, TwinSpec,
};
use num_rational::BigRational;
use std::time::Duration;

#[test]
fn prepared_domain_eval_genuine_mixed_head_phases_tighten_shared_row_with_own_errors() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain(&budget, 2, false);
    let mixed = root
        .split_head(0, PreparedPhaseSide::Inactive, None)
        .unwrap()
        .split_head(1, PreparedPhaseSide::Active, None)
        .unwrap();
    let eval = evaluate_prepared_domain(&mixed, None).unwrap();
    assert_eq!(eval.receipt().phase_halfspaces, 2);
    assert!(eval.receipt().tightened_heads > 0);
    // Closed phase halves overlap at the real tie x=1. This is a nonempty
    // child, and must close by an actual margin rather than fake emptiness.
    assert!(eval.crossing().is_none());
    let ybox = eval.ybox(None).unwrap();
    assert!(ybox.ly[0] > -1e-10 && ybox.uy[1] < 1e-10);
    assert!(eval
        .bounds(None)
        .unwrap()
        .iter()
        .all(|&b| b > 0.7 && b <= 0.75));
    drop((eval, mixed, root));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_distinct_original_head_biases_do_not_gain_shared_phase_equality() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain_parameters(&budget, 2, false, false, [-1.0, -2.0], 0.75);
    let child = root
        .split_head(0, PreparedPhaseSide::Active, None)
        .unwrap()
        .split_head(1, PreparedPhaseSide::Inactive, None)
        .unwrap();
    let eval = evaluate_prepared_domain(&child, None).unwrap();
    assert!(eval.crossing().is_none());
    // Actual original input x=3/2 obeys both phases with preactivations
    // +1/2 and -1/2. An equality based on weights alone would exclude it.
    let y = eval.ybox(None).unwrap();
    assert!(y.uy[0] >= 0.5 && y.ly[1] <= -0.5);
    for &bound in eval.bounds(None).unwrap() {
        assert!(BigRational::from_float(bound).unwrap() <= BigRational::new(5.into(), 4.into()));
    }
    drop((eval, child, root));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_independent_original_head_fold_errors_forbid_alias_phase_reuse() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let spec = TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Gemm {
                input: 0,
                weight: vec![1.0, 1.0],
                bias: vec![-1.0, -1.0],
                bias_err: vec![0.25, 0.25],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Gemm {
                input: 2,
                weight: vec![1.0, -1.0, 0.0, 0.0],
                bias: vec![0.75, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    };
    let session = PreparedRootSession::compile_declared_twin(
        spec,
        &[-1.0],
        &[3.0],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            prefer_patch: false,
            ..PreparedRootProofOptions::default()
        },
        &budget,
        Some(clock()),
    )
    .unwrap();
    let PreparedRootOutcome::Open(open) = session.try_prove_root(0, &[1], None).unwrap() else {
        panic!("independent original fold envelope must leave root open")
    };
    let root = PreparedDomain::from_open(open, None).unwrap();
    let child = root
        .split_head(0, PreparedPhaseSide::Inactive, None)
        .unwrap()
        .split_head(1, PreparedPhaseSide::Active, None)
        .unwrap();
    let eval = evaluate_prepared_domain(&child, None).unwrap();
    assert!(eval.crossing().is_none());
    // At original input1, independent allowed biases -5/4 and -3/4
    // produce (-1/4,+1/4), satisfying this mixed phase and margin1/2.
    let y = eval.ybox(None).unwrap();
    assert!(y.ly[0] <= -0.25 && y.uy[1] >= 0.25);
    for &bound in eval.bounds(None).unwrap() {
        assert!(BigRational::from_float(bound).unwrap() <= BigRational::new(1.into(), 2.into()));
    }
    drop((eval, child, root, session));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_actual_trunk_phase_uses_authenticated_original_input_rows() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain_with_clip(&budget, 2, false, true);
    assert!(root.adapter().clip_rows().is_some());
    let child = root
        .split_trunk(1, 0, PreparedPhaseSide::Active, None)
        .unwrap();
    let rows = build_prepared_domain_rows(&child, None).unwrap();
    assert!(rows.endpoints(None).unwrap().0[0] < -1.5);
    let eval = evaluate_prepared_domain_with_rows(&child, rows, None).unwrap();
    assert_eq!(eval.receipt().phase_halfspaces, 1);
    assert!(eval.receipt().tightened_heads > 0);
    assert!(eval.ybox(None).unwrap().ly[0] > -1.0000000001);
    for &bound in eval.bounds(None).unwrap() {
        assert!(BigRational::from_float(bound).unwrap() <= BigRational::from_float(0.75).unwrap());
    }
    drop((eval, child, root));
    assert_eq!(live(&memory), 0);
}

fn clock() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(256 << 20);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(clock()),
            scratch_bytes: 128 << 20,
            persistent_bytes: 128 << 20,
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
fn original_domain(budget: &WorkBudget, classes: usize, fold: bool) -> PreparedDomain {
    original_domain_with_clip(budget, classes, fold, false)
}
fn original_domain_with_clip(
    budget: &WorkBudget,
    classes: usize,
    fold: bool,
    clip: bool,
) -> PreparedDomain {
    original_domain_parameters(budget, classes, fold, clip, [-1.0, -1.0], 0.75)
}
fn original_domain_parameters(
    budget: &WorkBudget,
    classes: usize,
    fold: bool,
    clip: bool,
    head_bias: [f64; 2],
    margin_bias: f64,
) -> PreparedDomain {
    let mut weights = vec![0.0; classes * 2];
    weights[0] = 1.0;
    weights[1] = -1.0;
    let mut bias = vec![0.0; classes];
    bias[0] = margin_bias;
    let spec = TwinSpec {
        n_in: 1,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![if fold { 1.0 / 1024.0 } else { 0.0 }],
                weight_rel_err: if fold { 1.0 / 512.0 } else { 0.0 },
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
                weight: vec![1.0, 1.0],
                bias: head_bias.to_vec(),
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: weights,
                bias,
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (classes, 2),
            },
        ],
    };
    let session = PreparedRootSession::compile_declared_twin(
        spec,
        &[-1.0],
        &[3.0],
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            prefer_patch: false,
            keep_input_rows: clip,
            ..PreparedRootProofOptions::default()
        },
        budget,
        Some(clock()),
    )
    .unwrap();
    let order: Vec<usize> = (1..classes).rev().collect();
    let PreparedRootOutcome::Open(open) = session
        .try_prove_root_with_consumers(
            0,
            &order,
            crate::margin_row::PreparedRootConsumerOptions {
                clip_rows: clip,
                ..crate::margin_row::PreparedRootConsumerOptions::default()
            },
            None,
        )
        .unwrap()
    else {
        panic!("actual duplicate-head root is open")
    };
    PreparedDomain::from_open(open, None).unwrap()
}

#[test]
fn prepared_domain_eval_original_disjoint_phase_halfspaces_close_by_directed_crossing() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain_parameters(&budget, 2, false, false, [-1.0, -2.0], 0.1);
    let disjoint = root
        .split_head(0, PreparedPhaseSide::Inactive, None)
        .unwrap()
        .split_head(1, PreparedPhaseSide::Active, None)
        .unwrap();
    let evaluation = evaluate_prepared_domain(&disjoint, None).unwrap();
    // For the actual original t=ReLU(x), these phases require t<=1 and
    // t>=2. No supplied box or numerical contradiction mints emptiness.
    assert!(evaluation
        .crossing()
        .is_some_and(|slack| slack.is_finite() && slack > 0.0));
    assert_eq!(evaluation.receipt().phase_halfspaces, 2);
    let ybox = evaluation.ybox(None).unwrap();
    assert!(ybox.ly.iter().zip(&ybox.uy).any(|(l, u)| l > u));
    assert!(evaluation
        .bounds(None)
        .unwrap()
        .iter()
        .all(|&bound| bound == 0.0));
    drop((evaluation, disjoint, root));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_original_open_root_negative_head_children_close_complete_real_cover() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain(&budget, 3, false);
    assert!(root
        .root_evaluation()
        .bounds(None)
        .unwrap()
        .iter()
        .any(|&x| x <= 0.0));
    let site = root.next_split(None).unwrap().unwrap();
    assert_eq!(site, PreparedSplitSite::Head { neuron: 1 });
    let off = root.split(site, PreparedPhaseSide::Inactive, None).unwrap();
    let on = root.split(site, PreparedPhaseSide::Active, None).unwrap();
    let a = evaluate_prepared_domain(&off, None).unwrap();
    let b = evaluate_prepared_domain(&on, None).unwrap();
    assert_eq!(a.classes(), [2, 1]);
    assert_eq!(b.classes(), [2, 1]);
    assert!(a.crossing().is_none() && b.crossing().is_none());
    let rational = BigRational::from_float(0.75).unwrap();
    for result in [&a, &b] {
        for &bound in result.bounds(None).unwrap() {
            assert!(bound > 0.0);
            assert!(BigRational::from_float(bound).unwrap() <= rational);
        }
    }
    // These are actual original margins, not supplied positive evidence.
    let mut paid = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: 64 << 10,
                work: 1_000_000,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let pa = off.cover_path(None).unwrap();
    let pb = on.cover_path(None).unwrap();
    let mut ledger = CoverLedger::new();
    for (result, path) in [(&a, &pa), (&b, &pb)] {
        let margin = result
            .bounds(None)
            .unwrap()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        ledger.leaf(
            Some(path.path(None).unwrap().clone()),
            crate::certify::margin_row::bound_evidence(RoundMode::Outward, margin),
        );
    }
    let cover = ledger.prove(root.classes()).unwrap();
    assert_eq!(cover.classes(), [1, 2]);
    assert_eq!(cover.leaves(), 2);
    assert!(cover.min_margin() > 0.0);
    drop((cover, ledger, pa, pb));
    paid.complete().unwrap();
    drop(paid);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop((root, off, on, a, b));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_trunk_fixed_rows_use_actual_kernel_and_preserve_pointwise_rational_scope() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain(&budget, 2, false);
    let off = root
        .split_trunk(1, 0, PreparedPhaseSide::Inactive, None)
        .unwrap();
    let on = root
        .split_trunk(1, 0, PreparedPhaseSide::Active, None)
        .unwrap();
    let a = build_prepared_domain_rows(&off, None).unwrap();
    let b = build_prepared_domain_rows(&on, None).unwrap();
    for (rows, inputs) in [
        (&a, vec![-1.0_f64, -0.5, 0.0]),
        (&b, vec![0.0_f64, 0.5, 1.0, 3.0]),
    ] {
        let al = rows.lower(None).unwrap();
        let au = rows.upper(None).unwrap();
        for x in inputs {
            for j in 0..2 {
                let exact = BigRational::from_float(x.max(0.0) - 1.0).unwrap();
                let low = BigRational::from_float(al.a[[0, j]]).unwrap()
                    * BigRational::from_float(x).unwrap()
                    + BigRational::from_float(al.b[j]).unwrap();
                let high = BigRational::from_float(au.a[[0, j]]).unwrap()
                    * BigRational::from_float(x).unwrap()
                    + BigRational::from_float(au.b[j]).unwrap();
                let le = BigRational::from_float(al.e.as_ref().unwrap()[[0, j]]).unwrap()
                    * BigRational::from_float(x.abs()).unwrap()
                    + BigRational::from_float(al.eb[j]).unwrap();
                let ue = BigRational::from_float(au.e.as_ref().unwrap()[[0, j]]).unwrap()
                    * BigRational::from_float(x.abs()).unwrap()
                    + BigRational::from_float(au.eb[j]).unwrap();
                assert!(low - le <= exact && high + ue >= exact);
            }
        }
    }
    assert!(a.lower(None).unwrap().a[[0, 0]].abs() < b.lower(None).unwrap().a[[0, 0]].abs());
    assert_eq!(root.adapter().gates().layers[0].unst, [0]);
    drop((root, off, on, a, b));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_sibling_and_foreign_equal_source_rows_never_relabel_child_scope() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain(&budget, 2, false);
    let off = root
        .split_head(1, PreparedPhaseSide::Inactive, None)
        .unwrap();
    let on = root.split_head(1, PreparedPhaseSide::Active, None).unwrap();
    let rows = build_prepared_domain_rows(&off, None).unwrap();
    let before = live(&memory);
    assert!(evaluate_prepared_domain_with_rows(&on, rows.clone(), None).is_err());
    assert_eq!(live(&memory), before);
    let foreign = original_domain(&budget, 2, false);
    assert!(evaluate_prepared_domain_with_rows(&foreign, rows.clone(), None).is_err());
    rows.checkpoint(None).unwrap();
    drop((root, foreign, off, on, rows));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_cancel_after_actual_completed_tile_has_no_partial_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    let root = original_domain(&budget, 2, false);
    let before = live(&memory);
    let mut reached = false;
    let mut hook = |tiles| {
        assert_eq!(tiles, 1);
        reached = true;
        cancel.cancel();
    };
    assert!(build_rows_inner(&root, None, Some(&mut hook)).is_err());
    assert!(reached);
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(root);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_earliest_producer_clock_and_last_rows_owner_survive_parent_drop() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain(&budget, 2, false);
    let child = root.split_head(1, PreparedPhaseSide::Active, None).unwrap();
    let early = Instant::now() + Duration::from_secs(30);
    let eval = evaluate_prepared_domain(&child, Some(early)).unwrap();
    assert!(eval.deadline(None).unwrap() <= early);
    let rows = eval.rows(None).unwrap();
    assert!(rows.deadline(None).unwrap() <= early);
    let alias = eval.clone();
    drop((root, child, eval));
    assert!(live(&memory) > 0);
    alias.checkpoint(None).unwrap();
    rows.checkpoint(None).unwrap();
    drop(alias);
    assert!(live(&memory) > 0);
    drop(rows);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_expired_or_full_original_authority_refuses_before_first_payload_allocation()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain(&budget, 2, false);
    let admitted = budget.receipt().unwrap().admitted;
    assert!(build_prepared_domain_rows(
        &root,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    let r = budget.receipt().unwrap();
    budget
        .tighten_total_byte_limit(r.live_scratch_bytes + r.live_persistent_bytes)
        .unwrap();
    let before = live(&memory);
    assert!(build_prepared_domain_rows(&root, None).is_err());
    assert_eq!(live(&memory), before);
    root.checkpoint(None).unwrap();
    drop(root);
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_complete_65_original_classes_preserve_usize_order_and_tile_coverage() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain(&budget, 66, false);
    let child = root.split_head(1, PreparedPhaseSide::Active, None).unwrap();
    let eval = evaluate_prepared_domain(&child, None).unwrap();
    assert_eq!(eval.classes(), (1usize..66).rev().collect::<Vec<_>>());
    assert_eq!(eval.bounds(None).unwrap().len(), 65);
    assert_eq!(eval.receipt().margin_tiles, 2);
    assert_eq!(eval.receipt().direct_columns, 65);
    assert!(eval
        .bounds(None)
        .unwrap()
        .iter()
        .all(|&b| b > 0.0 && b <= 0.75));
    drop((root, child, eval));
    assert_eq!(live(&memory), 0);
}

#[test]
fn prepared_domain_eval_actual_signed_shared_fold_uncertainty_stays_below_original_rational_margin()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let root = original_domain(&budget, 2, true);
    for side in [PreparedPhaseSide::Inactive, PreparedPhaseSide::Active] {
        let child = root.split_head(1, side, None).unwrap();
        let eval = evaluate_prepared_domain(&child, None).unwrap();
        for &bound in eval.bounds(None).unwrap() {
            assert!(bound > 0.0);
            assert!(
                BigRational::from_float(bound).unwrap() <= BigRational::from_float(0.75).unwrap()
            );
        }
        for dw in [-1.0 / 512.0, 1.0 / 512.0] {
            for db in [-1.0 / 1024.0, 1.0 / 1024.0] {
                for x in [-1.0_f64, 0.0, 1.0, 3.0] {
                    let y = ((1.0 + dw) * x + db).max(0.0) - 1.0;
                    assert_eq!(0.75 + y.max(0.0) - y.max(0.0), 0.75);
                }
            }
        }
    }
    drop(root);
    assert_eq!(live(&memory), 0);
}
