// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Genuine original owners, independent rational kernels and paid frontiers.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::support_program::OriginalModel;
use crate::margin_row::RootPrecisionRuntimePolicy;
use num_rational::BigRational;
use std::time::Duration;

const CAP: usize = 128 << 20;
fn authority(
    cap: usize,
    deadline: Option<Instant>,
) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap.checked_mul(2).unwrap());
    let token = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &token,
        WorkLimits {
            deadline,
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: 1_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, token)
}
fn at() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn rat(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}
fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|x| x.to_bits()).collect()
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_materialized_bytes + r.live_pending_bytes
}
fn head(ops: &mut Vec<TwinOpSpec>, n: usize) {
    let input = ops.len();
    ops.push(TwinOpSpec::Flatten { input });
    let input = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input,
        weight: vec![0.25; n],
        bias: vec![0.0],
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (1, n),
    });
    let input = ops.len();
    ops.push(TwinOpSpec::Relu { input });
    let input = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input,
        weight: vec![0.5; 2],
        bias: vec![0.0; 2],
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (2, 1),
    });
}
fn spec(conv: TwinOpSpec, shape: (usize, usize, usize), n_in: usize) -> TwinSpec {
    let mut ops = vec![
        conv,
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::ChannelAffine {
            input: 2,
            scale: vec![-1.1000000000000003; shape.0],
            shift: vec![0.10000000000000002; shape.0],
            scale_rel_err: 0.003,
            shift_err: vec![0.002; shape.0],
            shape,
        },
        TwinOpSpec::Relu { input: 3 },
    ];
    head(&mut ops, shape.0 * shape.1 * shape.2);
    TwinSpec { n_in, ops }
}
fn dense(weights: Vec<f64>, bias: Vec<f64>, n: usize, rho: f64, be: f64) -> TwinSpec {
    let co = bias.len();
    spec(
        TwinOpSpec::Conv {
            input: 0,
            weight: weights,
            bias,
            bias_err: vec![be; co],
            weight_rel_err: rho,
            kernel: (co, n, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (n, 1, 1),
            oshape: (co, 1, 1),
        },
        (co, 1, 1),
        n,
    )
}
fn make_root(
    spec: TwinSpec,
    lo: &[f64],
    hi: &[f64],
    budget: &WorkBudget,
    deadline: Option<Instant>,
) -> OriginalRoot {
    let model = OriginalModel::compile(spec, budget, deadline).unwrap();
    OriginalRoot::new(
        model,
        lo,
        hi,
        RootPrecisionRuntimePolicy::default(),
        deadline,
    )
    .unwrap()
}

/// Independent authored-coordinate evaluation. It never reads compiled gather,
/// nominal casts, gamma or the new error helpers. Each output independently
/// enumerates its original weight/bias uncertainty corners and input vertices.
fn check_original_conv(source: &TwinSpec, input: &CertifiedTensorBox, got: &ComputedReferenceF32) {
    let (weights, bias, be, rho, kernel, stride, pads, ishape, oshape, transposed) =
        match &source.ops[0] {
            TwinOpSpec::Conv {
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => (
                weight,
                bias,
                bias_err,
                *weight_rel_err,
                *kernel,
                *stride,
                *pads,
                *ishape,
                *oshape,
                false,
            ),
            TwinOpSpec::ConvTranspose {
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => (
                weight,
                bias,
                bias_err,
                *weight_rel_err,
                *kernel,
                *stride,
                *pads,
                *ishape,
                *oshape,
                true,
            ),
            _ => panic!("oracle requires the authored convolution"),
        };
    let (co, ci, kh, kw) = kernel;
    let (_, ih, iw) = ishape;
    let (_, oh, ow) = oshape;
    let k = ci * kh * kw;
    assert!(input.lo().len() <= 6 && k <= 6);
    for vertex in 0..1usize << input.lo().len() {
        let x: Vec<_> = (0..input.lo().len())
            .map(|i| {
                rat(if vertex & (1 << i) == 0 {
                    input.lo()[i]
                } else {
                    input.hi()[i]
                })
            })
            .collect();
        for fold in 0..1usize << (k + 1) {
            let sign = |bit: usize| {
                if fold & (1 << bit) == 0 {
                    rat(-1.0)
                } else {
                    rat(1.0)
                }
            };
            for o in 0..co {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut y = rat(bias[o]) + sign(k) * rat(be[o]);
                        for c in 0..ci {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let tap = (c * kh + ky) * kw + kx;
                                    let w = rat(weights[o * k + tap])
                                        * (rat(1.0) + sign(tap) * rat(rho));
                                    if transposed {
                                        // Authored transpose convolution is a forward scatter.
                                        for iy in 0..ih {
                                            for ix in 0..iw {
                                                let ycoord = iy as isize * stride.0 as isize
                                                    - pads.0 as isize
                                                    + ky as isize;
                                                let xcoord = ix as isize * stride.1 as isize
                                                    - pads.1 as isize
                                                    + kx as isize;
                                                if ycoord == oy as isize && xcoord == ox as isize {
                                                    y += &w * &x[(c * ih + iy) * iw + ix];
                                                }
                                            }
                                        }
                                    } else {
                                        let iy = oy as isize * stride.0 as isize - pads.0 as isize
                                            + ky as isize;
                                        let ix = ox as isize * stride.1 as isize - pads.1 as isize
                                            + kx as isize;
                                        if iy >= 0
                                            && ix >= 0
                                            && iy < ih as isize
                                            && ix < iw as isize
                                        {
                                            y += &w * &x[(c * ih + iy as usize) * iw + ix as usize];
                                        }
                                    }
                                }
                            }
                        }
                        let id = (o * oh + oy) * ow + ox;
                        assert!(
                            rat(got.lo()[id]) <= y && y <= rat(got.hi()[id]),
                            "output{id}, input{vertex}, fold{fold}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn reference_f32_original_conv_full_cast_and_fold_gaps_contain_independent_rational_corners() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP, None);
    let source = dense(
        vec![
            0.10000000000000003,
            -1.25,
            0.3333333333333333,
            -0.2,
            0.7,
            1.0000000000000002,
        ],
        vec![0.1, -0.3],
        3,
        0.003,
        0.001,
    );
    let root = make_root(
        source.clone(),
        &[-0.7000000000000001, -0.2, 0.10000000000000002],
        &[0.2, 0.9000000000000001, 0.8],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let before = live(&memory);
    let result = derive_reference_f32(&root, 0, &input, Some(at())).unwrap();
    assert_eq!(result.receipt().conv_ops, 1);
    assert_eq!(result.receipt().multiply_add_pairs, 12);
    assert_eq!(result.receipt().converted_cells, 14);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(live(&memory), before + result.resident_bytes());
    check_original_conv(&source, &input, &result);
    drop(result);
    assert_eq!(live(&memory), before);
}

#[test]
fn reference_f32_signed_original_padding_is_not_a_tensor_batch_or_property_axis() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP, None);
    let source = spec(
        TwinOpSpec::Conv {
            input: 0,
            weight: vec![-0.25, 0.1, 1.5, 0.0],
            bias: vec![0.125],
            bias_err: vec![0.001],
            weight_rel_err: 0.003,
            kernel: (1, 1, 2, 2),
            stride: (1, 1),
            pads: (1, 1, 0, 0),
            ishape: (1, 2, 3),
            oshape: (1, 2, 3),
        },
        (1, 2, 3),
        6,
    );
    let root = make_root(
        source.clone(),
        &[-0.8, -0.5, -0.3, -0.1, 0.2, 0.4],
        &[-0.2, 0.1, 0.4, 0.6, 0.8, 1.1],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let result = derive_reference_f32(&root, 0, &input, Some(at())).unwrap();
    assert_eq!(result.lo().len(), 6);
    assert_eq!(result.receipt().conv_ops, 1);
    assert!(result.receipt().multiply_add_pairs < 2 * 6 * 4);
    check_original_conv(&source, &input, &result);
}

#[test]
fn reference_f32_original_transposed_stride_padding_taps_match_rational_forward_scatter() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP, None);
    let source = spec(
        TwinOpSpec::ConvTranspose {
            input: 0,
            weight: vec![-0.25, 0.1, 1.5, 0.0],
            bias: vec![0.125],
            bias_err: vec![0.001],
            weight_rel_err: 0.003,
            kernel: (1, 1, 2, 2),
            stride: (2, 2),
            pads: (1, 0, 0, 1),
            ishape: (1, 2, 2),
            oshape: (1, 3, 3),
            out_pad: (0, 0),
        },
        (1, 3, 3),
        4,
    );
    let root = make_root(
        source.clone(),
        &[-0.8, -0.5, 0.2, 0.4],
        &[-0.2, 0.1, 0.8, 1.1],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let result = derive_reference_f32(&root, 0, &input, Some(at())).unwrap();
    assert_eq!(result.lo().len(), 9);
    assert!(root
        .model()
        .net()
        .ops
        .iter()
        .any(|op| matches!(op,TwinOp::Conv(c) if c.transposed)));
    check_original_conv(&source, &input, &result);
}

#[test]
fn reference_f32_original_affine_signed_scale_and_fold_have_independent_exact_support() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP, None);
    let root = make_root(
        dense(vec![-0.3, 0.75], vec![0.1, -0.2], 1, 0.003, 0.001),
        &[-0.8],
        &[1.1],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let (_, relu) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    let got = derive_reference_f32(&root, 2, &relu, Some(at())).unwrap();
    assert_eq!(got.receipt().affine_ops, 1);
    assert_eq!(got.receipt().conv_ops, 0);
    assert_eq!(got.receipt().multiply_add_pairs, 4);
    let TwinOp::ChannelAffine {
        scale,
        shift,
        scale_rel_err,
        shift_err,
        ..
    } = &root.model().net().ops[2]
    else {
        unreachable!()
    };
    for i in 0..2 {
        for x in [relu.lo()[i], relu.hi()[i]] {
            for ws in [-1.0, 1.0] {
                for bs in [-1.0, 1.0] {
                    let y = rat(scale[i]) * (rat(1.0) + rat(ws) * rat(*scale_rel_err)) * rat(x)
                        + rat(shift[i])
                        + rat(bs) * rat(shift_err[i]);
                    assert!(rat(got.lo()[i]) <= y && y <= rat(got.hi()[i]));
                }
            }
        }
    }
}

#[test]
fn reference_f32_cancelling_large_signed_dots_do_not_discard_accumulation_error() {
    let _env = crate::tests::lock_env_shared();
    qualify().unwrap();
    // A final cast of an f64 dot gives 1.0, while the actual f32 reduction
    // loses the middle unit and gives 0.0. This pins the production dot kernel.
    let mut actual = Dot::new(cast(0.0).unwrap(), 0.0).unwrap();
    for w in [16_777_216.0, 1.0, -16_777_216.0] {
        actual
            .step(w, cast(w).unwrap(), cast(1.0).unwrap(), 0.0, 1.0)
            .unwrap();
    }
    assert_eq!(actual.nominal.to_bits(), 0.0_f32.to_bits());
    assert_ne!(actual.nominal.to_bits(), (1.0_f64 as f32).to_bits());
    assert!(actual.finish(false).unwrap() >= 1.0);
    let (budget, _, _) = authority(CAP, None);
    let source = dense(
        vec![16_777_217.0, -16_777_216.0, 0.10000000000000003],
        vec![-0.1],
        3,
        0.0,
        0.0,
    );
    let root = make_root(
        source.clone(),
        &[1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let got = derive_reference_f32(&root, 0, &input, Some(at())).unwrap();
    check_original_conv(&source, &input, &got);
    assert!(got.lo()[0] < 1.0 && got.hi()[0] > 1.0);
}

#[test]
fn reference_f32_subnormal_operand_flush_is_charged_before_weight_amplification() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP, None);
    let tiny = f32_to_f64_exact(f32::from_bits(1));
    let source = dense(vec![2_f64.powi(100)], vec![-0.0], 1, 0.0, 0.0);
    let root = make_root(source.clone(), &[-tiny], &[tiny], &budget, None);
    let input = root.input_box().unwrap();
    let got = derive_reference_f32(&root, 0, &input, Some(at())).unwrap();
    assert_eq!(got.receipt().subnormal_casts_normalized, 2);
    assert!(got.hi()[0] > MIN_NORMAL32 * 1.0e10);
    check_original_conv(&source, &input, &got);
    let casted = cast(tiny).unwrap();
    assert_eq!(casted.converted.to_bits(), 1);
    assert_eq!(casted.operand.to_bits(), 0);
    assert!(rat(casted.gap) >= rat(tiny));
}

#[test]
fn reference_f32_product_underflow_signed_zero_and_full_subtraction_residual_are_outward() {
    let _env = crate::tests::lock_env_shared();
    qualify().unwrap();
    assert_eq!(cast(-0.0).unwrap().converted.to_bits(), 0x8000_0000);
    assert_eq!(cast(-0.0).unwrap().operand.to_bits(), 0x8000_0000);
    let residual = 2_f64.powi(-60);
    let gap = full_gap(1.0, -residual);
    assert!(rat(gap) >= rat(1.0) + rat(residual));
    assert!(gap > 1.0);
    let tiny64 = f64::from_bits(1);
    assert!(rat(cast(tiny64).unwrap().gap) >= rat(tiny64));
    assert!(rat(mul_up(tiny64, tiny64)) >= rat(tiny64) * rat(tiny64));
    let (budget, _, _) = authority(CAP, None);
    let source = dense(vec![2_f64.powi(-100)], vec![0.0], 1, 0.0, 0.0);
    let root = make_root(
        source.clone(),
        &[-2_f64.powi(-100)],
        &[2_f64.powi(-100)],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let got = derive_reference_f32(&root, 0, &input, Some(at())).unwrap();
    check_original_conv(&source, &input, &got);
    assert!(got.lo()[0] < 0.0 && got.hi()[0] > 0.0);
}

#[test]
fn reference_f32_constant_bias_only_result_reports_conversions_without_inventing_dot_ops() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP, None);
    let root = make_root(
        dense(vec![0.0, -0.0], vec![0.10000000000000002], 2, 0.0, 0.001),
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let got = derive_reference_f32(&root, 0, &input, Some(at())).unwrap();
    let r = got.receipt();
    assert_eq!(r.conv_ops, 0);
    assert_eq!(r.multiply_add_pairs, 0);
    assert_eq!(r.constant_outputs, 1);
    assert!(r.converted_cells > 0);
    assert!(rat(got.lo()[0]) <= rat(0.10000000000000002) - rat(0.001));
    assert!(rat(got.hi()[0]) >= rat(0.10000000000000002) + rat(0.001));
}

#[test]
fn reference_f32_original_current_ledger_exact_local_cap_is_admitted_one_past_is_not() {
    let _env = crate::tests::lock_env_shared();
    let cap = crate::margin_row::root_box::ROOT_SOURCE_BYTES_CAP;
    let (budget, memory, _) = authority(cap * 2, None);
    let root = make_root(
        dense(vec![0.1], vec![0.2], 1, 0.0, 0.0),
        &[-0.3],
        &[0.4],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let plan = Plan::new(&root, 0, &input).unwrap();
    let request = plan.request.scratch_bytes + plan.request.persistent_bytes;
    for over in [0, 1] {
        let current = live(&memory);
        let mut blocker = budget
            .lease_for(
                LeaseClass::Fallback,
                WorkRequest {
                    persistent_bytes: cap - current - request + over,
                    ..WorkRequest::default()
                },
            )
            .unwrap();
        blocker.mark_materialized().unwrap();
        let blocker = blocker.commit_persistent().unwrap();
        let before = memory.receipt().unwrap();
        let work_before = budget.receipt().unwrap();
        let result = derive_reference_f32(&root, 0, &input, Some(at()));
        if over == 0 {
            drop(result.unwrap());
        } else {
            assert!(matches!(result, Err(NyError::CpuMemoryExceeded { .. })));
            assert_eq!(memory.receipt().unwrap(), before);
            assert_eq!(budget.receipt().unwrap().admitted, work_before.admitted);
            assert_eq!(
                budget.receipt().unwrap().work_reserved,
                work_before.work_reserved
            );
        }
        drop(blocker);
    }
}

#[test]
fn reference_f32_same_ledger_claim_moves_once_and_earliest_source_deadline_cannot_renew() {
    let _env = crate::tests::lock_env_shared();
    let d = at();
    let (budget, memory, _) = authority(CAP, Some(d));
    let root = make_root(
        dense(vec![0.1], vec![0.2], 1, 0.0, 0.0),
        &[-0.3],
        &[0.4],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let got = derive_reference_f32(&root, 0, &input, Some(d + Duration::from_secs(300))).unwrap();
    assert_eq!(got.deadline().unwrap(), Some(d));
    let lp = got.lo.as_ptr();
    let up = got.hi.as_ptr();
    let bytes = got.resident_bytes();
    let (lo, hi, deadline, claim, receipt) = got.into_parts(&root, 0).unwrap();
    assert_eq!(lo.as_ptr(), lp);
    assert_eq!(hi.as_ptr(), up);
    assert_eq!(deadline, Some(d));
    assert_eq!(claim.bytes(), bytes);
    assert!(claim.shares_ledger(&budget));
    assert_eq!(receipt.conv_ops, 1);
    let foreign = make_root(
        dense(vec![0.1], vec![0.2], 1, 0.0, 0.0),
        &[-0.3],
        &[0.4],
        &budget,
        None,
    );
    let rejected = derive_reference_f32(&root, 0, &input, Some(d)).unwrap();
    assert!(rejected.into_parts(&foreign, 0).is_err());
    drop(foreign);
    drop(input);
    drop(root);
    assert_eq!(live(&memory), bytes);
    drop((lo, hi, claim));
    assert_eq!(live(&memory), 0);
}

#[test]
fn reference_f32_expiry_cancellation_and_foreign_source_cannot_publish_or_refill() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, token) = authority(CAP, None);
    let source = dense(vec![0.1, 0.2], vec![0.2, 0.3], 1, 0.0, 0.0);
    let root = make_root(source.clone(), &[-0.3], &[0.4], &budget, None);
    let input = root.input_box().unwrap();
    let foreign = make_root(source, &[-0.3], &[0.4], &budget, None);
    let foreign_input = foreign.input_box().unwrap();
    let before = memory.receipt().unwrap();
    let counts = budget.receipt().unwrap();
    assert!(derive_reference_f32(&root, 0, &foreign_input, Some(at())).is_err());
    assert!(matches!(
        derive_reference_f32(
            &root,
            0,
            &input,
            Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
        ),
        Err(NyError::DeadlineExceeded(_))
    ));
    assert_eq!(memory.receipt().unwrap(), before);
    assert_eq!(budget.receipt().unwrap().admitted, counts.admitted);
    let mut completed = 0;
    assert!(derive_inner(
        &root,
        0,
        &input,
        Some(at()),
        Some(&mut |_| {
            completed += 1;
            token.cancel();
        })
    )
    .is_err());
    assert_eq!(completed, 1);
    assert_eq!(
        live(&memory),
        before.live_pending_bytes + before.live_materialized_bytes
    );
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert!(budget.receipt().unwrap().work_reserved > counts.work_reserved);
    let cancelled = budget.receipt().unwrap();
    assert!(derive_reference_f32(&root, 0, &input, Some(at())).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, cancelled.admitted);
    assert_eq!(
        budget.receipt().unwrap().work_reserved,
        cancelled.work_reserved
    );
}

#[test]
fn reference_f32_nonfinite_conversion_nominal_overflow_and_gamma_limits_refuse() {
    let _env = crate::tests::lock_env_shared();
    qualify().unwrap();
    for x in [f64::NAN, f64::INFINITY, f64::MAX] {
        assert!(cast(x).is_err());
    }
    assert!(gamma((1 << 23) - 1).is_err());
    assert!(gamma(usize::MAX).is_err());
    let (budget, memory, _) = authority(CAP, None);
    let root = make_root(
        dense(vec![1.0e30], vec![0.0], 1, 0.0, 0.0),
        &[1.0e30],
        &[1.0e30],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let before = live(&memory);
    assert!(derive_reference_f32(&root, 0, &input, Some(at())).is_err());
    assert_eq!(live(&memory), before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn reference_f32_component_leaves_legacy_reference_bits_and_policy_unchanged() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP, None);
    let root = make_root(
        dense(vec![-0.1, 0.2], vec![0.125], 2, 0.003, 0.001),
        &[-0.3, 0.1],
        &[0.4, 0.7],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let before = root.derive(0, &input, None).unwrap();
    let legacy = (bits(before.lo()), bits(before.hi()));
    drop(derive_reference_f32(&root, 0, &input, Some(at())).unwrap());
    let after = root.derive(0, &input, None).unwrap();
    assert_eq!(legacy, (bits(after.lo()), bits(after.hi())));
    assert!(!root.policy().root_f32 && !root.policy().tableau_f32);
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
fn reference_f32_unqualified_certificate_environment_refuses_before_admission() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP, None);
    let root = make_root(
        dense(vec![0.1], vec![0.2], 1, 0.0, 0.0),
        &[-0.3],
        &[0.4],
        &budget,
        None,
    );
    let input = root.input_box().unwrap();
    let before = memory.receipt().unwrap();
    let counts = budget.receipt().unwrap();
    {
        let _directed = DirectedRounding::enter();
        assert!(!ny_core::has_f64_interval_proof_environment());
        assert!(matches!(
            derive_reference_f32(&root, 0, &input, Some(at())),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    assert_eq!(memory.receipt().unwrap(), before);
    assert_eq!(budget.receipt().unwrap().admitted, counts.admitted);
    qualify().unwrap();
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
#[test]
#[expect(
    clippy::redundant_clone,
    reason = "retain the source owner so refusal cannot release its charged storage before receipt comparison"
)]
fn binary32_reference_construction_refuses_unqualified_rounding_before_new_admission() {
    use super::super::{
        build_prepared_root, PreparedConstructionOptions, PreparedReferenceArithmetic,
    };
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP, None);
    let root = make_root(
        dense(vec![0.1], vec![0.2], 1, 0.0, 0.0),
        &[-0.3],
        &[0.4],
        &budget,
        None,
    );
    let before = memory.receipt().unwrap();
    let admitted = budget.receipt().unwrap().admitted;
    {
        let _rounding = DirectedRounding::enter();
        assert!(matches!(
            build_prepared_root(
                root.clone(),
                PreparedConstructionOptions {
                    reference_arithmetic: PreparedReferenceArithmetic::Binary32Certified,
                    ..Default::default()
                },
                Some(at())
            ),
            Err(NyError::SoundnessRefusal(_))
        ));
    }
    assert_eq!(memory.receipt().unwrap(), before);
    assert_eq!(budget.receipt().unwrap().admitted, admitted);
    qualify().unwrap();
}

#[test]
fn binary32_reference_construction_subnormal_flush_contains_independent_original_fold_corners() {
    use super::super::{
        build_prepared_root, PreparedConstructionOptions, PreparedReferenceArithmetic,
    };
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP, None);
    let tiny = f32_to_f64_exact(f32::from_bits(1));
    let weight = 2_f64.powi(100);
    let root = make_root(
        dense(vec![weight], vec![-0.0], 1, 0.0, 0.0),
        &[-tiny],
        &[tiny],
        &budget,
        None,
    );
    let source = build_prepared_root(
        root,
        PreparedConstructionOptions {
            reference_arithmetic: PreparedReferenceArithmetic::Binary32Certified,
            ..Default::default()
        },
        Some(at()),
    )
    .unwrap();
    let receipt = source.reference_receipt();
    assert!(receipt.subnormal_casts_normalized >= 2);
    assert_eq!(
        (receipt.completed_conv_ops, receipt.completed_affine_ops),
        (1, 1)
    );
    for x in [-tiny, 0.0, tiny] {
        for rho in [-0.003, 0.003] {
            for bias in [-0.002, 0.002] {
                let h = (rat(weight) * rat(x)).max(rat(0.0));
                let exact = (rat(-1.1000000000000003) * (rat(1.0) + rat(rho)) * h
                    + rat(0.10000000000000002)
                    + rat(bias))
                .max(rat(0.0));
                assert!(rat(source.trunk_output().lo()[0]) <= exact);
                assert!(exact <= rat(source.trunk_output().hi()[0]));
            }
        }
    }
    source.validate_reference(None).unwrap();
    drop(source);
    assert_eq!(live(&memory), 0);
}
