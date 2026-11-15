// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Native affine inputs stay in their original flat coordinates. These controls
//! use no convolution, synthetic flatten, supplied gate or numerical proof.

use super::super::cli_proof::{
    PreparedRootConsumerOptions, PreparedRootOutcome, PreparedRootProof, PreparedRootProofOptions,
    PreparedRootSession,
};
use super::super::terminal::build_prepared_terminal;
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::bab::{
    MarginRowBab, PreparedTreeOptions, PreparedTreeOutcome, PreparedTreeSchedule,
};
use crate::margin_row::RootPrecisionRuntimePolicy;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 32 << 20;
fn at() -> Instant {
    Instant::now() + Duration::from_secs(60)
}
fn resources() -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(CAP);
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: Some(at()),
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 100_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    budget.tighten_total_byte_limit(CAP).unwrap();
    (budget, memory, cancel)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn native(n: usize, h: usize, margin: f64) -> TwinSpec {
    TwinSpec {
        n_in: n,
        ops: vec![
            TwinOpSpec::Gemm {
                input: 0,
                weight: vec![0.0; n * h],
                bias: vec![0.0; h],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (h, n),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Gemm {
                input: 2,
                weight: vec![0.0; 2 * h],
                bias: vec![margin, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, h),
            },
        ],
    }
}
fn session(spec: TwinSpec, lo: &[f64], hi: &[f64], budget: &WorkBudget) -> PreparedRootSession {
    PreparedRootSession::compile_declared_twin(
        spec,
        lo,
        hi,
        RootPrecisionRuntimePolicy::default(),
        PreparedRootProofOptions {
            keep_input_rows: true,
            ..Default::default()
        },
        budget,
        Some(at()),
    )
    .unwrap()
}
fn r(value: f64) -> BigRational {
    BigRational::from_float(value).unwrap()
}
fn original_float(input: usize, output: usize) -> f32 {
    match input * 128 + output {
        0 => 0.0,
        1 => -0.0,
        2 => f32::from_bits(1),
        3 => f32::from_bits(0x007f_ffff),
        i => (i as f32 - 1900.0) / 4096.0,
    }
}

#[test]
fn native_affine_original30x128_coordinates_and_parameter_allocations_move_without_dummy_ops() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let mut spec = native(30, 128, 0.75);
    let (wptr, bptr, expected) = match &mut spec.ops[0] {
        TwinOpSpec::Gemm { weight, bias, .. } => {
            // Authored original MatMul coordinates [input,output], transposed
            // once into the extractor's genuine [output,input] declaration.
            for input in 0..30 {
                for output in 0..128 {
                    let original = original_float(input, output);
                    weight[output * 30 + input] = f64::from(original);
                }
            }
            for (j, value) in bias.iter_mut().enumerate() {
                *value = f64::from((j as f32 - 64.0) / 128.0);
            }
            (weight.as_ptr(), bias.as_ptr(), weight.clone())
        }
        _ => unreachable!(),
    };
    let model = OriginalModel::compile(spec, &budget, Some(at())).unwrap();
    let net = model.net();
    let (weight, bias, shape) = net.gemm1();
    assert_eq!(shape, (128, 30));
    assert_eq!((net.n_in, net.n_y, net.n_out, net.i_gemm1), (30, 128, 2, 0));
    assert_eq!((net.trunk_relus).len(), 0);
    assert_eq!(net.ops.len(), 3);
    assert!(matches!(&net.ops[0], TwinOp::Gemm { input: 0, .. }));
    assert!(matches!(&net.ops[1], TwinOp::Relu { input: 1, .. }));
    assert!(matches!(&net.ops[2], TwinOp::Gemm { input: 2, .. }));
    assert_eq!(weight.as_ptr(), wptr);
    assert_eq!(bias.as_ptr(), bptr);
    assert_eq!(weight, expected);
    for input in 0..30 {
        for output in 0..128 {
            let original = original_float(input, output);
            assert_eq!(
                weight[output * 30 + input].to_bits(),
                f64::from(original).to_bits()
            );
        }
    }
    assert!(model.budget().shares_ledger(&budget));
    drop(model);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_signed_original_fold_corners_bound_terminal_and_actual_ordinary_kernel() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let mut spec = native(2, 2, 3.0);
    let weights = [1.5, -0.75, -0.5, 1.25];
    let biases = [0.125, -0.25];
    let rho = 1.0 / 64.0;
    let eb = 1.0 / 128.0;
    if let TwinOpSpec::Gemm {
        weight,
        bias,
        bias_err,
        weight_rel_err,
        ..
    } = &mut spec.ops[0]
    {
        *weight = weights.to_vec();
        *bias = biases.to_vec();
        *bias_err = vec![eb; 2];
        *weight_rel_err = rho;
    }
    if let TwinOpSpec::Gemm { weight, .. } = &mut spec.ops[2] {
        *weight = vec![0.5, -0.25, -0.25, 0.5];
    }
    let lo = [-1.0, -0.5];
    let hi = [1.0, 1.5];
    let source = session(spec, &lo, &hi, &budget);
    let construction = source.source().construction();
    assert_eq!(construction.prefix().layer_count(), 0);
    assert_eq!(construction.trunk_output().tensor(), 0);
    assert!(construction.layers().is_empty());
    assert!(construction.coverage_complete(None).unwrap());
    let terminal = build_prepared_terminal(construction, 0, &[1], None).unwrap();
    assert_eq!(terminal.receipt().head_lower_columns, 2);
    assert_eq!(terminal.receipt().head_upper_columns, 2);
    let PreparedRootOutcome::Proven(proof) = source
        .try_prove_root_with_consumers(
            0,
            &[1],
            PreparedRootConsumerOptions {
                clip_rows: true,
                ..Default::default()
            },
            None,
        )
        .unwrap()
    else {
        panic!("signed original native margin must be proved")
    };
    assert!(proof.attempt_receipt().ordinary_attempted);
    assert!(proof.attempt_receipt().ordinary_completed);
    // Independent exact original coordinates; every input-box corner and
    // independent coefficient/bias uncertainty corner is interpreted directly.
    for input_mask in 0..4usize {
        let x = [
            r(if input_mask & 1 == 0 { lo[0] } else { hi[0] }),
            r(if input_mask & 2 == 0 { lo[1] } else { hi[1] }),
        ];
        for parameter_mask in 0..64usize {
            let mut z = Vec::new();
            for row in 0..2 {
                let sign = if parameter_mask & (1 << (4 + row)) == 0 {
                    -1.0
                } else {
                    1.0
                };
                let mut pre = r(biases[row]) + r(sign) * r(eb);
                for col in 0..2 {
                    let k = row * 2 + col;
                    let sign = if parameter_mask & (1 << k) == 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let w = r(weights[k]) * (r(1.0) + r(sign) * r(rho));
                    pre += w * x[col].clone();
                }
                assert!(r(terminal.head_lo()[row]) <= pre);
                assert!(r(terminal.head_hi()[row]) >= pre);
                z.push(pre.max(BigRational::zero()));
            }
            let exact = r(3.0) + r(0.75) * z[0].clone() - r(0.75) * z[1].clone();
            assert!(r(terminal.lower()[0]) <= exact);
            assert!(r(proof.min_margin()) <= exact);
        }
    }
    drop(source);
    drop(terminal);
    assert!(live(&memory) > 0);
    drop(proof);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_original30x128_open_root_real_head_children_close_complete_cover() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let mut spec = native(30, 128, 0.75);
    if let TwinOpSpec::Gemm { weight, bias, .. } = &mut spec.ops[0] {
        weight[0] = 1.0;
        weight[30] = 1.0;
        bias[0] = -1.0;
        bias[1] = -1.0;
    }
    if let TwinOpSpec::Gemm { weight, .. } = &mut spec.ops[2] {
        weight[0] = 1.0;
        weight[1] = -1.0;
    }
    let mut lo = vec![-1.0; 30];
    let mut hi = vec![1.0; 30];
    lo[0] = 0.0;
    hi[0] = 3.0;
    let source = session(spec, &lo, &hi, &budget);
    let PreparedRootOutcome::Open(open) = source.try_prove_root(0, &[1], None).unwrap() else {
        panic!("correlated duplicate native heads must reach the real open route")
    };
    assert!(open.receipt().ordinary_completed);
    let PreparedTreeOutcome::Covered(cert) = MarginRowBab::run_prepared(
        open,
        PreparedTreeOptions {
            max_expansions: 2,
            lru_capacity: 4,
            schedule: PreparedTreeSchedule::Serial,
        },
        None,
    )
    .unwrap() else {
        panic!("real conditional native heads must close")
    };
    assert_eq!(cert.receipt().bound_leaves, 2);
    let proof = PreparedRootProof::from_prepared_tree(cert, None).unwrap();
    assert!(proof.session().same_source(&source));
    assert!(proof.min_margin() > 0.0 && r(proof.min_margin()) <= r(0.75));
    assert_eq!(proof.classes(), &[1]);
    for value in [0.0, 0.5, 1.0, 1.5, 3.0] {
        let z = (r(value) - r(1.0)).max(BigRational::zero());
        let exact = r(0.75) + z.clone() - z;
        assert!(r(proof.min_margin()) <= exact);
    }
    drop(source);
    drop(proof);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_closed_terminal_keeps_cheap_route_and_negative_subnormal_stays_open() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(native(1, 1, 0.75), &[-1.0], &[1.0], &budget);
    let PreparedRootOutcome::Proven(proof) = source.try_prove_root(0, &[1], None).unwrap() else {
        panic!("constant original margin must close")
    };
    assert!(proof.attempt_receipt().terminal_closed);
    assert!(!proof.attempt_receipt().ordinary_attempted);
    assert!(r(proof.min_margin()) <= r(0.75));
    drop(source);
    drop(proof);
    let negative = session(native(1, 1, -f64::from_bits(1)), &[-1.0], &[1.0], &budget);
    let PreparedRootOutcome::Open(open) = negative.try_prove_root(0, &[1], None).unwrap() else {
        panic!("exact negative minimum subnormal must not authorize")
    };
    assert!(open.terminal().lower()[0] < 0.0);
    assert!(r(open.terminal().lower()[0]) <= -r(f64::from_bits(1)));
    drop(negative);
    drop(open);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_invalid_shape_tail_final_fold_and_nonfinite_still_refuse_without_owner() {
    let _env = crate::tests::lock_env_shared();
    for case in 0..5 {
        let (budget, memory, _) = resources();
        let mut spec = native(2, 3, 0.75);
        match case {
            0 => {
                if let TwinOpSpec::Gemm { shape, .. } = &mut spec.ops[0] {
                    shape.1 = 1;
                }
            }
            1 => {
                if let TwinOpSpec::Relu { input } = &mut spec.ops[1] {
                    *input = 0;
                }
            }
            2 => {
                if let TwinOpSpec::Gemm { weight_rel_err, .. } = &mut spec.ops[2] {
                    *weight_rel_err = 0.001;
                }
            }
            3 => {
                if let TwinOpSpec::Gemm { bias_err, .. } = &mut spec.ops[2] {
                    *bias_err = vec![0.0, 0.001];
                }
            }
            4 => {
                if let TwinOpSpec::Gemm { weight, .. } = &mut spec.ops[0] {
                    weight[0] = f64::NAN;
                }
            }
            _ => unreachable!(),
        }
        assert!(OriginalModel::compile(spec, &budget, Some(at())).is_err());
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn native_affine_expired_cancelled_and_occupied_original_cap_publish_no_partial_proof() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, cancel) = resources();
    assert!(OriginalModel::compile(
        native(30, 128, 0.75),
        &budget,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    assert_eq!(live(&memory), 0);
    let source = session(
        native(30, 128, 0.75),
        &vec![-1.0; 30],
        &vec![1.0; 30],
        &budget,
    );
    let baseline = live(&memory);
    let mut held = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: CAP - baseline - 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    held.mark_materialized().unwrap();
    let occupied = held.commit_persistent().unwrap();
    assert!(source.try_prove_root(0, &[1], None).is_err());
    assert_eq!(live(&memory), CAP - 1);
    drop(occupied);
    assert_eq!(live(&memory), baseline);
    source.checkpoint(None).unwrap();
    cancel.cancel();
    assert!(source.try_prove_root(0, &[1], None).is_err());
    assert_eq!(live(&memory), baseline);
    drop(source);
    assert_eq!(live(&memory), 0);
}

#[test]
fn native_affine_exact_owner_alias_deadline_and_last_release_are_preserved() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = resources();
    let source = session(
        native(30, 128, 0.75),
        &vec![-1.0; 30],
        &vec![1.0; 30],
        &budget,
    );
    let old = budget.receipt().unwrap();
    let early = Instant::now() + Duration::from_secs(5);
    let clone = source.restricted(Some(early)).unwrap();
    assert!(clone.same_source(&source));
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        old.live_persistent_bytes
    );
    assert!(clone.deadline(None).unwrap() <= early);
    let PreparedRootOutcome::Proven(proof) = clone.try_prove_root(0, &[1], None).unwrap() else {
        panic!("authentic native constant must close")
    };
    let last = proof.clone();
    assert!(proof.deadline(None).unwrap() <= early);
    let before_clone = budget.receipt().unwrap();
    let alias = last.clone();
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        before_clone.live_persistent_bytes
    );
    drop(source);
    drop(clone);
    drop(proof);
    drop(last);
    assert!(live(&memory) > 0);
    alias.checkpoint(None).unwrap();
    drop(alias);
    assert_eq!(live(&memory), 0);
}
