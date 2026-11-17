// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Cooperative arithmetic controls. These helpers do not mint proof owners.
//! The whole-original oracle uses a genuine compiled kernel; constructed row
//! carriers below test arithmetic premises only, never publication authority.

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{
    LeaseClass, WorkBudget, WorkLimits, WorkRefusal, WorkRequest, WorkReserve,
};
use crate::margin_row::engine::LaneDir;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use num_rational::BigRational;
use num_traits::Zero;
use std::cell::Cell;
use std::time::{Duration, Instant};

fn q(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}
fn check() -> Result<()> {
    Ok(())
}
fn fixture(fold: bool) -> TwinSpec {
    TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0, -0.5, -0.25, 0.75],
                bias: vec![0.125, -0.25],
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
                weight: vec![1.25, -0.5, -0.75, 0.25, 0.5, 0.75],
                bias: vec![-0.25, 0.125, -0.375],
                bias_err: if fold { vec![1.0 / 128.0; 3] } else { vec![] },
                weight_rel_err: if fold { 1.0 / 64.0 } else { 0.0 },
                shape: (3, 2),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![1.0, 0.5, -0.25, 0.0, 0.25, 0.5, -0.25, 0.75, 0.0],
                bias: vec![0.25, -0.125, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (3, 3),
            },
        ],
    }
}
fn root(net: &TwinNet, mode: RoundMode) -> RootGates {
    RootGates::build(net, &[-1.0, -0.5], &[0.75, 1.0], mode, None).unwrap()
}
fn pack(n: usize, r: usize, a: f64, e: f64) -> PassOut {
    PassOut {
        a: Array2::from_elem((n, r), a),
        e: Some(Array2::from_elem((n, r), e)),
        b: vec![0.0; r],
        eb: vec![0.0; r],
        coll: None,
        coll_rows: None,
        unst_rows: None,
    }
}
fn resource(
    scratch: usize,
    work: u64,
    deadline: Option<Instant>,
) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(scratch.saturating_mul(4).max(1));
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline,
            scratch_bytes: scratch,
            persistent_bytes: scratch,
            max_work: work,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel)
}
fn error(error: WorkRefusal) -> NyError {
    if error == WorkRefusal::Deadline {
        NyError::DeadlineExceeded("head original deadline".into())
    } else {
        NyError::UnsupportedConfiguration(format!("head original ledger: {error}"))
    }
}

#[test]
fn bounds_cooperative_nominal_batch_gate_seed_bits_preserve_original_order() {
    let net = TwinNet::compile(&fixture(false)).unwrap();
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let root = root(&net, mode);
        let eng = BackwardEngine::new(&net, &root);
        let (al, au) = eng.y_rows(None).unwrap();
        let ybox = YBox::from_rows(&eng, &al, &au);
        let legacy = MarginBatch::new(&net, 0, &[2, 1]).unwrap();
        let new = margin_batch_with_checkpoint(&net, 0, &[2, 1], &check).unwrap();
        assert_eq!(legacy.adv, new.adv);
        assert_eq!(bits(&legacy.wp), bits(&new.wp));
        assert_eq!(bits(&legacy.wn), bits(&new.wn));
        assert_eq!(bits(&legacy.cst), bits(&new.cst));
        let ha = [0.25, 0.75, 0.5];
        let old_gate = head_gates_with(&ybox, mode, Some(&ha));
        let new_gate = head_gates_with_checkpoint(&ybox, mode, Some(&ha), &check).unwrap();
        assert_eq!(bits(&old_gate.alpha), bits(&new_gate.alpha));
        assert_eq!(bits(&old_gate.s), bits(&new_gate.s));
        assert_eq!(bits(&old_gate.c), bits(&new_gate.c));
        let old_seed = margin_seed(&legacy, &old_gate, &ybox, mode);
        let new_seed = margin_seed_with_checkpoint(&new, &new_gate, &ybox, mode, &check).unwrap();
        assert_eq!(
            bits(old_seed.seed.s.as_slice().unwrap()),
            bits(new_seed.seed.s.as_slice().unwrap())
        );
        assert_eq!(bits(&old_seed.cst), bits(&new_seed.cst));
        assert_eq!(bits(&old_seed.zu), bits(&new_seed.zu));
        if mode.outward() {
            for (old, new) in old_seed
                .seed
                .e
                .as_ref()
                .unwrap()
                .iter()
                .zip(new_seed.seed.e.as_ref().unwrap())
            {
                assert!(*new >= *old);
            }
            for (old, new) in old_seed.m1.iter().zip(&new_seed.m1) {
                assert!(*new <= *old);
            }
        }
    }
}

#[test]
fn bounds_cooperative_parity_composition_and_direct_bits_match_legacy() {
    let net = TwinNet::compile(&fixture(false)).unwrap();
    let root = root(&net, RoundMode::Parity);
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).unwrap();
    let ybox = YBox::from_rows(&eng, &al, &au);
    let mb = MarginBatch::new(&net, 0, &[2, 1]).unwrap();
    let gates = head_gates(&ybox, RoundMode::Parity);
    let ms = margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Parity, &check).unwrap();
    let pass = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .unwrap();
    let ad = row_dots_with_checkpoint(&root, &al, &check).unwrap();
    let ud = row_dots_with_checkpoint(&root, &au, &check).unwrap();
    assert_eq!(bits(&ad.abs_dot), bits(&row_dots(&root, &al).abs_dot));
    assert_eq!(bits(&ad.err_dot), bits(&row_dots(&root, &al).err_dot));
    let old = compose_viay(&eng, &mb, &gates, &al, &au, &ad, &ud, RoundMode::Parity);
    let new = compose_viay_with_checkpoint(
        &eng,
        &mb,
        &gates,
        &al,
        &au,
        &ad,
        &ud,
        RoundMode::Parity,
        &check,
    )
    .unwrap();
    assert_eq!(bits(&old), bits(&new));
    let low = eng.concretize_lower(&pass);
    let direct =
        per_class_direct_from_lower_with_checkpoint(&low, &ms, 0..2, RoundMode::Parity, &check)
            .unwrap();
    assert_eq!(
        bits(&direct),
        bits(&per_class_direct(&eng, &pass, &ms, 0..2))
    );
}

#[test]
fn bounds_cooperative_whole_original_fold_corner_rational_containment() {
    let net = TwinNet::compile(&fixture(true)).unwrap();
    let root = root(&net, RoundMode::Outward);
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).unwrap();
    let ybox = YBox::from_rows(&eng, &al, &au);
    let mb = margin_batch_with_checkpoint(&net, 0, &[2, 1], &check).unwrap();
    let gates =
        head_gates_with_checkpoint(&ybox, RoundMode::Outward, Some(&[0.25, 0.75, 0.5]), &check)
            .unwrap();
    let ms = margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check).unwrap();
    let pass = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .unwrap();
    let low = eng.concretize_lower(&pass);
    let direct =
        per_class_direct_from_lower_with_checkpoint(&low, &ms, 0..2, RoundMode::Outward, &check)
            .unwrap();
    let ad = row_dots_with_checkpoint(&root, &al, &check).unwrap();
    let ud = row_dots_with_checkpoint(&root, &au, &check).unwrap();
    let via = compose_viay_with_checkpoint(
        &eng,
        &mb,
        &gates,
        &al,
        &au,
        &ad,
        &ud,
        RoundMode::Outward,
        &check,
    )
    .unwrap();
    let w = [[1.25, -0.5], [-0.75, 0.25], [0.5, 0.75]];
    // Exercise the production legacy composer with its actual RN dot caches
    // as well as the cooperative lane above.
    let legacy_via = compose_viay(
        &eng,
        &mb,
        &gates,
        &al,
        &au,
        &row_dots(&root, &al),
        &row_dots(&root, &au),
        RoundMode::Outward,
    );
    let b = [-0.25, 0.125, -0.375];
    let out = [[1.0, 0.5, -0.25], [0.0, 0.25, 0.5], [-0.25, 0.75, 0.0]];
    let ob = [0.25, -0.125, 0.0];
    for x0 in [-1.0, 0.75] {
        for x1 in [-0.5, 1.0] {
            let h = [
                (q(x0) - q(0.5) * q(x1) + q(0.125)).max(BigRational::zero()),
                (-q(0.25) * q(x0) + q(0.75) * q(x1) - q(0.25)).max(BigRational::zero()),
            ];
            for signs in 0..8 {
                let mut z = Vec::new();
                for j in 0..3 {
                    let nominal = q(w[j][0]) * &h[0] + q(w[j][1]) * &h[1] + q(b[j]);
                    let radius = q(1.0 / 64.0)
                        * (q(w[j][0].abs()) * &h[0] + q(w[j][1].abs()) * &h[1])
                        + q(1.0 / 128.0);
                    z.push(
                        (nominal
                            + if signs & (1 << j) == 0 {
                                -radius
                            } else {
                                radius
                            })
                        .max(BigRational::zero()),
                    );
                }
                for (r0, j) in [2, 1].into_iter().enumerate() {
                    let mut truth = q(ob[0]) - q(ob[j]);
                    for k in 0..3 {
                        truth += (q(out[0][k]) - q(out[j][k])) * &z[k];
                    }
                    let claimed = direct[r0].max(ms.m1[r0]).max(via[r0]).max(legacy_via[r0]);
                    assert!(
                        q(claimed) <= truth,
                        "row{r0} input{x0},{x1} fold{signs}: {claimed}"
                    );
                }
            }
        }
    }
}

#[test]
fn bounds_cooperative_m1_subnormal_ties_refuse_legacy_positive_max() {
    let tiny = f64::from_bits(1);
    let n = 64;
    let mb = MarginBatch {
        t: 0,
        adv: vec![1],
        wp: vec![0.0; n],
        wn: vec![-0.5; n],
        cst: vec![8.0 * tiny],
        n_y: n,
    };
    let ybox = YBox {
        ly: vec![tiny; n],
        uy: vec![tiny; n],
    };
    let gates = head_gates(&ybox, RoundMode::Outward);
    let old = relative_only_margin_seed_for_test(&mb, &gates, &ybox, RoundMode::Outward);
    let new = margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check).unwrap();
    let exact = q(-24.0) * q(tiny);
    assert!(q(old.m1[0]) > exact);
    assert!(old.m1[0] > 0.0);
    assert!(q(new.m1[0]) <= exact);
    assert!(new.m1[0] < 0.0);
}

#[test]
fn bounds_cooperative_seed_product_gap_keeps_subnormal_error_before_amplification() {
    let tiny = f64::from_bits(1);
    let mb = MarginBatch {
        t: 0,
        adv: vec![1],
        wp: vec![tiny],
        wn: vec![0.0],
        cst: vec![0.0],
        n_y: 1,
    };
    let gates = HeadGates {
        alpha: vec![0.5],
        s: vec![1.0],
        c: vec![1.0],
    };
    let ybox = YBox {
        ly: vec![-1.0],
        uy: vec![1.0],
    };
    let ms = margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check).unwrap();
    let exact = q(tiny) * q(0.5);
    let center = q(ms.seed.s[[0, 0]]);
    let error = q(ms.seed.e.as_ref().unwrap()[[0, 0]]);
    assert!(center.clone() - &error <= exact && exact <= center + error);
}

#[test]
fn bounds_cooperative_row_error_underflow_then_large_margin_is_paid() {
    let tiny = f64::from_bits(1);
    let net = TwinNet::compile(&fixture(false)).unwrap();
    let root = RootGates::build(&net, &[0.5, 0.0], &[0.5, 0.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let mut al = pack(2, 3, 0.0, 0.0);
    let mut au = pack(2, 3, 0.0, 0.0);
    al.e.as_mut().unwrap()[[0, 0]] = tiny;
    au.e.as_mut().unwrap()[[0, 0]] = tiny;
    let old = relative_only_row_dots_for_test(&root, &al);
    let ad = row_dots_with_checkpoint(&root, &al, &check).unwrap();
    let ud = row_dots_with_checkpoint(&root, &au, &check).unwrap();
    assert_eq!(old.err_dot[0], 0.0);
    assert!(q(ad.err_dot[0]) >= q(tiny) * q(0.5));
    let mb = MarginBatch {
        t: 0,
        adv: vec![1],
        wp: vec![1e300, 0.0, 0.0],
        wn: vec![0.0; 3],
        cst: vec![0.0],
        n_y: 3,
    };
    let gates = HeadGates {
        alpha: vec![1.0; 3],
        s: vec![1.0; 3],
        c: vec![0.0; 3],
    };
    let new = compose_viay_with_checkpoint(
        &eng,
        &mb,
        &gates,
        &al,
        &au,
        &ad,
        &ud,
        RoundMode::Outward,
        &check,
    )
    .unwrap();
    let exact = -q(1e300) * q(tiny) * q(0.5);
    assert!(q(new[0]) <= exact);
}

#[test]
fn bounds_cooperative_magnitude_dot_and_zero_error_lanes_are_directed() {
    let tiny = f64::from_bits(1);
    let net = TwinNet::compile(&fixture(false)).unwrap();
    let root = RootGates::build(&net, &[0.5, 0.0], &[0.5, 0.0], RoundMode::Outward, None).unwrap();
    let mut rows = pack(2, 3, 0.0, 0.0);
    rows.a[[0, 0]] = tiny;
    let dots = row_dots_with_checkpoint(&root, &rows, &check).unwrap();
    assert!(q(dots.abs_dot[0]) >= q(tiny) * q(0.5));
    assert_eq!(dots.err_dot, vec![0.0; 3]);
    assert_eq!(dots.abs_dot[1], 0.0);
}

#[test]
fn bounds_cooperative_cancellation_is_polled_inside_single_wide_input_row() {
    let net = TwinNet::compile(&fixture(false)).unwrap();
    let root = root(&net, RoundMode::Outward);
    let rows = pack(2, 1024, 1.0, 0.25);
    let calls = Cell::new(0);
    let cancel = || {
        let n = calls.get() + 1;
        calls.set(n);
        if n == 20 {
            Err(NyError::UnsupportedConfiguration(
                "cancel actual inner row scan".into(),
            ))
        } else {
            Ok(())
        }
    };
    assert!(row_dots_with_checkpoint(&root, &rows, &cancel).is_err());
    assert_eq!(calls.get(), 20);
}

#[test]
fn bounds_cooperative_gate_chunks_preserve_extreme_and_partial_block_bits() {
    let n = HEAD_POLL + 3;
    let mut ly = vec![-0.25; n];
    let mut uy = vec![0.75; n];
    ly[0] = -f64::MAX;
    uy[0] = f64::MAX;
    ly[1] = -0.0;
    uy[1] = 0.0;
    ly[2] = -2.0;
    uy[2] = -1.0;
    let ybox = YBox { ly, uy };
    let slopes = vec![0.375; n];
    let old = head_gates_with(&ybox, RoundMode::Outward, Some(&slopes));
    let new = head_gates_with_checkpoint(&ybox, RoundMode::Outward, Some(&slopes), &check).unwrap();
    assert_eq!(bits(&old.alpha), bits(&new.alpha));
    assert_eq!(bits(&old.s), bits(&new.s));
    assert_eq!(bits(&old.c), bits(&new.c));
}

#[test]
fn bounds_cooperative_shape_and_missing_error_planes_refuse() {
    let net = TwinNet::compile(&fixture(false)).unwrap();
    let root = root(&net, RoundMode::Outward);
    let mut rows = pack(2, 3, 0.0, 0.0);
    rows.e = None;
    assert!(row_dots_with_checkpoint(&root, &rows, &check).is_err());
    rows = pack(2, 3, 0.0, 0.0);
    rows.eb.pop();
    assert!(row_dots_with_checkpoint(&root, &rows, &check).is_err());
    assert!(head_gates_with_checkpoint(
        &YBox {
            ly: vec![0.0],
            uy: vec![]
        },
        RoundMode::Outward,
        None,
        &check
    )
    .is_err());
    assert!(head_gates_with_checkpoint(
        &YBox {
            ly: vec![f64::NAN],
            uy: vec![1.0]
        },
        RoundMode::Outward,
        None,
        &check
    )
    .is_err());
}

#[test]
fn bounds_cooperative_quotes_include_real_output_and_all_gate_chunk_overlap() {
    let n = HEAD_POLL + 3;
    let request = head_gates_request(n).unwrap();
    assert_eq!(request.scratch_bytes, HEAD_META + 3 * (n + HEAD_POLL) * 8);
    let new = head_gates_with_checkpoint(
        &YBox {
            ly: vec![-1.0; n],
            uy: vec![2.0; n],
        },
        RoundMode::Outward,
        None,
        &check,
    )
    .unwrap();
    assert_eq!(new.alpha.capacity(), n);
    assert_eq!(new.s.capacity(), n);
    assert_eq!(new.c.capacity(), n);
    assert!(margin_batch_request(usize::MAX, 2).is_err());
    assert!(compose_viay_request(usize::MAX, 1, 1).is_err());
    assert!(compose_viay_request(1, usize::MAX, 0).is_err());
}

#[test]
fn bounds_cooperative_original_cap_floor_refuses_before_buffer_callback() {
    let request = head_gates_request(513).unwrap();
    let (budget, _, _) = resource(request.scratch_bytes - 1, request.work, None);
    let before = budget.receipt().unwrap().admitted;
    assert!(matches!(
        budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: request.scratch_bytes,
                work: request.work,
                ..WorkRequest::default()
            }
        ),
        Err(WorkRefusal::ScratchLimit)
    ));
    assert_eq!(budget.receipt().unwrap().admitted, before);
}

#[test]
fn bounds_cooperative_nested_original_ledger_and_cancellation_survive_allocation() {
    let request = head_gates_request(257).unwrap();
    let (budget, memory, cancel) = resource(request.scratch_bytes, 2 * request.work, None);
    let nested = budget.clone();
    assert!(budget.shares_ledger(&nested));
    let mut lease = nested
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: request.scratch_bytes,
                work: request.work,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    lease.mark_materialized().unwrap();
    assert!(matches!(
        budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: 1,
                ..WorkRequest::default()
            }
        ),
        Err(WorkRefusal::ScratchLimit)
    ));
    let calls = Cell::new(0);
    let checkpoint = || {
        let count = calls.get() + 1;
        calls.set(count);
        if count == 6 {
            cancel.cancel();
        }
        nested.checkpoint_for(LeaseClass::Fallback).map_err(error)
    };
    assert!(head_gates_with_checkpoint(
        &YBox {
            ly: vec![-1.0; 257],
            uy: vec![1.0; 257]
        },
        RoundMode::Outward,
        None,
        &checkpoint
    )
    .is_err());
    drop(lease);
    let r = memory.receipt().unwrap();
    assert_eq!(r.live_pending_bytes + r.live_materialized_bytes, 0);
    assert!(budget.checkpoint_for(LeaseClass::Fallback).is_err());
}

#[test]
fn bounds_cooperative_earliest_original_deadline_refuses_without_renewal() {
    let request = head_gates_request(1).unwrap();
    let original = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let (budget, _, _) = resource(request.scratch_bytes, request.work, Some(original));
    assert!(matches!(
        budget.lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: request.scratch_bytes,
                work: request.work,
                ..WorkRequest::default()
            }
        ),
        Err(WorkRefusal::Deadline)
    ));
    let calls = Cell::new(0);
    let checkpoint = || {
        calls.set(calls.get() + 1);
        budget.checkpoint_for(LeaseClass::Fallback).map_err(error)
    };
    assert!(matches!(
        head_gates_with_checkpoint(
            &YBox {
                ly: vec![0.0],
                uy: vec![1.0]
            },
            RoundMode::Outward,
            None,
            &checkpoint
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(calls.get(), 1);
    assert_eq!(budget.receipt().unwrap().admitted, 0);
}

#[test]
fn bounds_cooperative_constant_product_underflow_full_gap_is_retained() {
    let tiny = f64::from_bits(1);
    let n = 64;
    let mb = MarginBatch {
        t: 0,
        adv: vec![1],
        wp: vec![0.0; n],
        wn: vec![-0.5; n],
        cst: vec![8.0 * tiny],
        n_y: n,
    };
    let gates = HeadGates {
        alpha: vec![0.0; n],
        s: vec![0.5; n],
        c: vec![tiny; n],
    };
    let ybox = YBox {
        ly: vec![-tiny; n],
        uy: vec![tiny; n],
    };
    let ms = margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check).unwrap();
    let exact = -q(24.0) * q(tiny);
    let center = q(ms.cst[0]);
    let error = q(ms.cst_err[0]);
    assert!(center.clone() - &error <= exact && exact <= center + error);
}

#[test]
fn bounds_cooperative_via_y_subnormal_products_cannot_preserve_legacy_positive() {
    let tiny = f64::from_bits(1);
    let n = 64;
    let net = TwinNet::compile(&fixture(false)).unwrap();
    let root = RootGates::build(&net, &[0.5, 0.0], &[0.5, 0.0], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let mut al = pack(2, n, 0.0, 0.0);
    let mut au = pack(2, n, 0.0, 0.0);
    for j in 0..n {
        al.a[[0, j]] = tiny;
        au.a[[0, j]] = tiny;
    }
    let mb = MarginBatch {
        t: 0,
        adv: vec![1],
        wp: vec![0.0; n],
        wn: vec![-0.5; n],
        cst: vec![8.0 * tiny],
        n_y: n,
    };
    let gates = HeadGates {
        alpha: vec![1.0; n],
        s: vec![1.0; n],
        c: vec![0.0; n],
    };
    let old_ad = relative_only_row_dots_for_test(&root, &al);
    let old_ud = relative_only_row_dots_for_test(&root, &au);
    let old = relative_only_compose_viay_for_test(
        &eng,
        &mb,
        &gates,
        &al,
        &au,
        &old_ad,
        &old_ud,
        RoundMode::Outward,
    );
    let ad = row_dots_with_checkpoint(&root, &al, &check).unwrap();
    let ud = row_dots_with_checkpoint(&root, &au, &check).unwrap();
    let new = compose_viay_with_checkpoint(
        &eng,
        &mb,
        &gates,
        &al,
        &au,
        &ad,
        &ud,
        RoundMode::Outward,
        &check,
    )
    .unwrap();
    let exact = -q(8.0) * q(tiny);
    assert!(q(old[0]) > exact && old[0] > 0.0);
    assert!(q(new[0]) <= exact && new[0] < 0.0);
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(super) struct HeadDirectedRounding {
    #[cfg(target_arch = "aarch64")]
    saved: u64,
    #[cfg(target_arch = "x86_64")]
    saved: u32,
}
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[allow(deprecated)]
impl HeadDirectedRounding {
    #[allow(unsafe_code)]
    pub(super) fn enter() -> Self {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let saved: u64;
            std::arch::asm!("mrs {saved}, fpcr",saved=out(reg)saved,options(nomem,nostack));
            let directed = (saved & !(3 << 22)) | (1 << 22);
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
impl Drop for HeadDirectedRounding {
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
fn bounds_cooperative_unqualified_environment_cannot_certify_head_scalars() {
    let _env = crate::tests::lock_env_shared();
    let ybox = YBox {
        ly: vec![-1.0],
        uy: vec![1.0],
    };
    let gates = HeadGates {
        alpha: vec![0.5],
        s: vec![0.5],
        c: vec![0.5],
    };
    let mb = MarginBatch {
        t: 0,
        adv: vec![1],
        wp: vec![1.0],
        wn: vec![0.0],
        cst: vec![0.0],
        n_y: 1,
    };
    let _rounding = HeadDirectedRounding::enter();
    assert!(!ny_core::has_f64_interval_proof_environment());
    assert!(matches!(
        head_gates_with_checkpoint(&ybox, RoundMode::Outward, None, &check),
        Err(NyError::SoundnessRefusal(_))
    ));
    assert!(matches!(
        margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check),
        Err(NyError::SoundnessRefusal(_))
    ));
}

#[test]
fn bounds_cooperative_directed_overflow_is_never_hidden_by_min_or_cancellation() {
    assert!(head_sum_hi(f64::INFINITY, f64::NEG_INFINITY).is_nan());
    let (lo, hi) = head_interval_product(f64::MAX, f64::MAX, 0.0, 2.0);
    assert!(lo.is_nan() && hi.is_nan());
    let n = 64;
    let mb = MarginBatch {
        t: 0,
        adv: vec![1],
        wp: vec![f64::MAX; n],
        wn: vec![0.0; n],
        cst: vec![0.0],
        n_y: n,
    };
    let ybox = YBox {
        ly: vec![2.0; n],
        uy: vec![2.0; n],
    };
    let gates = head_gates(&ybox, RoundMode::Outward);
    assert!(margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check).is_err());
}

#[test]
fn bounds_cooperative_original_compiled_constant_head_reaches_legacy_underflow() {
    let tiny = f64::from_bits(1);
    let n = 64;
    let spec = TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![0.0; 2],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 2, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (2, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![0.0; n],
                bias: vec![1024.0 * tiny; n],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (n, 1),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: [vec![0.0; n], vec![1.0 / 4096.0; n]].concat(),
                bias: vec![8.0 * tiny, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, n),
            },
        ],
    };
    let net = TwinNet::compile(&spec).unwrap();
    let root = RootGates::build(&net, &[0.0; 2], &[0.0; 2], RoundMode::Outward, None).unwrap();
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).unwrap();
    let ybox = YBox::from_rows(&eng, &al, &au);
    assert!(ybox.uy.iter().all(|&u| u < 2048.0 * tiny));
    let mb = MarginBatch::new(&net, 0, &[1]).unwrap();
    let gates = head_gates(&ybox, RoundMode::Outward);
    let old = relative_only_margin_seed_for_test(&mb, &gates, &ybox, RoundMode::Outward);
    let new = margin_seed_with_checkpoint(&mb, &gates, &ybox, RoundMode::Outward, &check).unwrap();
    let exact = q(8.0 * tiny) - q(64.0) * q(1.0 / 4096.0) * q(1024.0 * tiny);
    assert!(exact < q(0.0));
    assert!(q(old.m1[0]) > exact && old.m1[0] > 0.0);
    assert!(q(new.m1[0]) <= exact && new.m1[0] < 0.0);
    let repaired = margin_seed(&mb, &gates, &ybox, RoundMode::Outward);
    assert!(q(repaired.m1[0]) <= exact && repaired.m1[0] < 0.0);
    let evaluated = crate::margin_row::bab::root_eval(&eng, &net, 0, &[1]).unwrap();
    assert!(q(evaluated.dj[0]) <= exact);
    assert_eq!(evaluated.tree_classes, vec![1]);
    assert!(!crate::margin_row::bab::closed(
        RoundMode::Outward,
        evaluated.dj[0]
    ));
    // This exercises the actual serial eval_node and worker eval_with_pack
    // tree consumers, not just a supplied head box or component scalar.
    let visited =
        crate::margin_row::bab::MarginRowBab::diff_walk_serial_vs_worker(&net, &root, 0, &[1], 1);
    assert_ne!((visited).len(), 0);
    for (_, lower_bits, workers_identical) in visited {
        let bound = f64::from_bits(lower_bits);
        assert!(workers_identical && bound.is_finite());
        assert!(q(bound) <= exact);
        assert!(!crate::margin_row::bab::closed(RoundMode::Outward, bound));
    }
}

// Historical relative-only comparator: no production caller can invoke it.
fn relative_only_margin_seed_for_test(
    mb: &MarginBatch,
    gates: &HeadGates,
    ybox: &YBox,
    mode: RoundMode,
) -> MarginSeed {
    let nf = mb.nf();
    let n_y = mb.n_y;
    let mut s_mat = Array2::<f64>::zeros((n_y, nf));
    let mut e_mat = mode.outward().then(|| Array2::<f64>::zeros((n_y, nf)));
    for j in 0..n_y {
        for r0 in 0..nf {
            let v = mb.wp[r0 * n_y + j] * gates.alpha[j] + mb.wn[r0 * n_y + j] * gates.s[j];
            s_mat[[j, r0]] = v;
            if let Some(e) = e_mat.as_mut() {
                let mag = (mb.wp[r0 * n_y + j] * gates.alpha[j]).abs()
                    + (mb.wn[r0 * n_y + j] * gates.s[j]).abs();
                e[[j, r0]] = slack16(3.0 * UNIT * mag);
            }
        }
    }
    let gam = gamma_n(n_y + 8);
    let mut cst = vec![0.0; nf];
    let mut cst_err = vec![0.0; nf];
    let mut m1 = vec![0.0; nf];
    let zl: Vec<f64> = ybox.ly.iter().map(|v| v.max(0.0)).collect();
    let zu: Vec<f64> = ybox.uy.iter().map(|v| v.max(0.0)).collect();
    for r0 in 0..nf {
        let wp = &mb.wp[r0 * n_y..(r0 + 1) * n_y];
        let wn = &mb.wn[r0 * n_y..(r0 + 1) * n_y];
        let mut acc_c = 0.0;
        let mut abs_c = 0.0;
        let mut acc_m = 0.0;
        let mut abs_m = 0.0;
        for j in 0..n_y {
            acc_c += wn[j] * gates.c[j];
            abs_c += wn[j].abs() * gates.c[j];
            acc_m += wp[j] * zl[j] + wn[j] * zu[j];
            abs_m += wp[j] * zl[j] + wn[j].abs() * zu[j];
        }
        cst[r0] = acc_c + mb.cst[r0];
        m1[r0] = acc_m + mb.cst[r0];
        if mode.outward() {
            cst_err[r0] = slack16(gam * (abs_c + mb.cst[r0].abs()));
            let sl = next_up(gam * (abs_m + mb.cst[r0].abs()));
            m1[r0] = next_down(next_down(m1[r0] - sl));
        }
    }
    MarginSeed {
        seed: Seed { s: s_mat, e: e_mat },
        cst,
        cst_err,
        m1,
        zu,
    }
}

// Historical relative-only comparator: no production caller can invoke it.
fn relative_only_row_dots_for_test(root: &RootGates, pack: &PassOut) -> RowDots {
    let n_in = root.mid.len();
    let r = pack.a.ncols();
    let asl = pack.a.as_slice().expect("standard layout");
    let esl = pack.e.as_ref().map(|e| e.as_slice().expect("layout"));
    let mut abs_dot = vec![0.0; r];
    let mut err_dot = vec![0.0; r];
    for i in 0..n_in {
        let xa = root.xabs[i];
        let arow = &asl[i * r..(i + 1) * r];
        for (j, av) in arow.iter().enumerate() {
            abs_dot[j] += av.abs() * xa;
        }
        if let Some(es) = esl {
            let erow = &es[i * r..(i + 1) * r];
            for (j, ev) in erow.iter().enumerate() {
                err_dot[j] += ev * xa;
            }
        }
    }
    for j in 0..r {
        abs_dot[j] += pack.b[j].abs();
        err_dot[j] += pack.eb[j];
    }
    RowDots { abs_dot, err_dot }
}

// Historical relative-only comparator: no production caller can invoke it.
fn relative_only_compose_viay_for_test(
    eng: &BackwardEngine<'_>,
    mb: &MarginBatch,
    gates: &HeadGates,
    al: &PassOut,
    au: &PassOut,
    al_dots: &RowDots,
    au_dots: &RowDots,
    mode: RoundMode,
) -> Vec<f64> {
    let nf = mb.nf();
    let n_y = mb.n_y;
    let root = eng.root;
    let n_in = root.mid.len();
    let asl = al.a.as_slice().expect("standard layout");
    let usl = au.a.as_slice().expect("standard layout");
    let gam_row = gamma_n(2 * n_y + 8);
    let gam_con = gamma_n(n_in + 16);
    let mut out = vec![0.0; nf];
    let mut rowbuf = vec![0.0; n_in];
    for r0 in 0..nf {
        let wp = &mb.wp[r0 * n_y..(r0 + 1) * n_y];
        let wn = &mb.wn[r0 * n_y..(r0 + 1) * n_y];
        // Composition coefficients cA = wp*alpha (>=0 side on lower rows),
        // cB = wn*s (<=0 side on upper rows).
        let ca: Vec<f64> = (0..n_y).map(|j| wp[j] * gates.alpha[j]).collect();
        let cb: Vec<f64> = (0..n_y).map(|j| wn[j] * gates.s[j]).collect();
        // rows_r[k] = sum_j ca[j]*Al[k][j] + cb[j]*Au[k][j]
        for (k, rb) in rowbuf.iter_mut().enumerate() {
            let arow = &asl[k * n_y..(k + 1) * n_y];
            let urow = &usl[k * n_y..(k + 1) * n_y];
            let mut acc = 0.0;
            for j in 0..n_y {
                acc += ca[j] * arow[j] + cb[j] * urow[j];
            }
            *rb = acc;
        }
        let mut bias = 0.0;
        let mut cterm = 0.0;
        for j in 0..n_y {
            bias += ca[j] * al.b[j] + cb[j] * au.b[j];
            cterm += wn[j] * gates.c[j];
        }
        // Concretize rowbuf over the box.
        let mut v = 0.0;
        let mut rr = 0.0;
        for (k, rb) in rowbuf.iter().enumerate() {
            v += rb * root.mid[k];
            rr += rb.abs() * root.rad[k];
        }
        let raw = v - rr + bias + cterm + mb.cst[r0];
        if !mode.outward() {
            out[r0] = raw;
            continue;
        }
        // Certified penalty: pack errors + seed-coefficient rounding +
        // row-computation gamma + concretization gamma + cterm/cst rounding.
        let mut pen = 0.0;
        let mut tmag = 0.0;
        for j in 0..n_y {
            let ea = slack16(3.0 * UNIT * ca[j].abs());
            let ebc = slack16(3.0 * UNIT * cb[j].abs());
            pen += (ca[j].abs() + ea) * al_dots.err_dot[j]
                + ea * al_dots.abs_dot[j]
                + (cb[j].abs() + ebc) * au_dots.err_dot[j]
                + ebc * au_dots.abs_dot[j];
            tmag += ca[j].abs() * al_dots.abs_dot[j]
                + cb[j].abs() * au_dots.abs_dot[j]
                + wn[j].abs() * gates.c[j];
        }
        tmag += mb.cst[r0].abs();
        let slack = next_up(
            certify_up(pen, 1e-13) + next_up((gam_row + gam_con) * certify_up(tmag, 1e-13)),
        );
        out[r0] = next_down(next_down(raw - slack));
    }
    out
}

#[test]
fn bounds_cooperative_legacy_triggered_normal_and_parity_bits_preserved() {
    let net = TwinNet::compile(&fixture(false)).unwrap();
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let root = root(&net, mode);
        let eng = BackwardEngine::new(&net, &root);
        let ybox = YBox {
            ly: vec![-1.0; 3],
            uy: vec![2.0; 3],
        };
        let mb = MarginBatch::new(&net, 0, &[2, 1]).unwrap();
        let gates = head_gates(&ybox, mode);
        let old = relative_only_margin_seed_for_test(&mb, &gates, &ybox, mode);
        let new = margin_seed(&mb, &gates, &ybox, mode);
        assert_eq!(
            bits(old.seed.s.as_slice().unwrap()),
            bits(new.seed.s.as_slice().unwrap())
        );
        assert_eq!(bits(&old.cst), bits(&new.cst));
        assert_eq!(bits(&old.cst_err), bits(&new.cst_err));
        assert_eq!(bits(&old.m1), bits(&new.m1));
        if let (Some(a), Some(b)) = (&old.seed.e, &new.seed.e) {
            assert_eq!(bits(a.as_slice().unwrap()), bits(b.as_slice().unwrap()));
        }
        let al = pack(2, 3, 1.25, 0.125);
        let au = pack(2, 3, 1.5, 0.25);
        let old_ad = relative_only_row_dots_for_test(&root, &al);
        let old_ud = relative_only_row_dots_for_test(&root, &au);
        let ad = row_dots(&root, &al);
        let ud = row_dots(&root, &au);
        assert_eq!(bits(&old_ad.abs_dot), bits(&ad.abs_dot));
        assert_eq!(bits(&old_ad.err_dot), bits(&ad.err_dot));
        let old = relative_only_compose_viay_for_test(
            &eng, &mb, &gates, &al, &au, &old_ad, &old_ud, mode,
        );
        let new = compose_viay(&eng, &mb, &gates, &al, &au, &ad, &ud, mode);
        assert_eq!(bits(&old), bits(&new));
    }
}
