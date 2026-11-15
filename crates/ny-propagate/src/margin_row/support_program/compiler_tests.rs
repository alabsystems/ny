// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use std::time::Duration;

fn authority(
    cap: usize,
    deadline: Option<Instant>,
) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap);
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline,
            scratch_bytes: cap / 2,
            persistent_bytes: cap / 2,
            max_work: 1_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancellation)
}

fn fixture(transposed: bool) -> TwinSpec {
    let kernel = (2, 1, 2, 2);
    let ishape = (1, 2, 3);
    let oshape = if transposed { (2, 4, 5) } else { (2, 2, 2) };
    let n = oshape.0 * oshape.1 * oshape.2;
    let weight = vec![0.5, -1.0, 0.25, 0.75, -0.5, 1.5, 0.125, 1.0];
    let bias = vec![0.125, -0.25];
    let bias_err = vec![0.001, 0.002];
    let conv = if transposed {
        TwinOpSpec::ConvTranspose {
            input: 0,
            weight,
            bias,
            bias_err,
            weight_rel_err: 0.003,
            kernel,
            stride: (2, 2),
            pads: (1, 0, 0, 1),
            ishape,
            oshape,
            out_pad: (1, 0),
        }
    } else {
        TwinOpSpec::Conv {
            input: 0,
            weight,
            bias,
            bias_err,
            weight_rel_err: 0.003,
            kernel,
            stride: (1, 2),
            pads: (1, 0, 0, 1),
            ishape,
            oshape,
        }
    };
    TwinSpec {
        n_in: 6,
        ops: vec![
            conv,
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Add { lhs: 2, rhs: 2 },
            TwinOpSpec::ChannelAffine {
                input: 3,
                scale: vec![0.5, -0.75],
                shift: vec![0.125, -0.25],
                shift_err: vec![0.001, 0.002],
                scale_rel_err: 0.004,
                shape: oshape,
            },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: (0..3 * n).map(|i| (i as f64 - 4.0) / 16.0).collect(),
                bias: vec![0.125, -0.5, 0.25],
                bias_err: vec![0.003, 0.004, 0.005],
                weight_rel_err: 0.006,
                shape: (3, n),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![1.0, -0.5, 0.25, -0.75, 0.125, 0.5],
                bias: vec![0.0, 0.125],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 3),
            },
        ],
    }
}

fn wide(n: usize, head: usize, classes: usize) -> TwinSpec {
    TwinSpec {
        n_in: n,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![1.0],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, n),
                oshape: (1, 1, n),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Flatten { input: 2 },
            TwinOpSpec::Gemm {
                input: 3,
                weight: vec![0.25; n * head],
                bias: vec![0.0; head],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (head, n),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.5; classes * head],
                bias: vec![0.0; classes],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (classes, head),
            },
        ],
    }
}

fn no_live_claims(budget: &WorkBudget, memory: &MemoryAuthority) {
    let work = budget.receipt().unwrap();
    let bytes = memory.receipt().unwrap();
    assert_eq!(
        (work.live_scratch_bytes, work.live_persistent_bytes),
        (0, 0)
    );
    assert_eq!(
        (bytes.live_pending_bytes, bytes.live_materialized_bytes),
        (0, 0)
    );
}

#[test]
fn original_compilation_local_peak_counts_paid_source_claim_before_new_admission() {
    let cap = super::super::super::root_box::ROOT_SOURCE_BYTES_CAP;
    let (budget, memory, _) = authority(2 * cap, None);
    let spec = fixture(false);
    let (sizes, _) = validate_geometry(&spec, &budget, None).unwrap();
    let request = compile_prediction(&spec, &sizes, &budget, None).unwrap();
    assert!(request.persistent_bytes < cap);
    let held = cap - request.scratch_bytes - request.persistent_bytes + 1;
    // This genuine same-ledger source/geometry reservation conservatively
    // exercises the frontier without allocating hundreds of MiB of vectors.
    let source = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: held,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let global_only = budget.lease_for(LeaseClass::Fallback, request).unwrap();
    drop(global_only);
    let before = budget.receipt().unwrap();
    assert!(matches!(lease_compilation(&budget, held, request, None),
        Err(NyError::CpuMemoryExceeded { required_bytes, budget_bytes, site: "original support compilation total live" })
        if required_bytes == cap + 1 && budget_bytes == cap));
    let after = budget.receipt().unwrap();
    assert_eq!(after.admitted, before.admitted);
    assert_eq!(after.live_scratch_bytes, held);
    assert_eq!(after.live_persistent_bytes, 0);
    drop(source);
    no_live_claims(&budget, &memory);
}

#[test]
fn original_compilation_exact_local_peak_and_checked_overflow_preserve_source_claim() {
    let cap = super::super::super::root_box::ROOT_SOURCE_BYTES_CAP;
    let (budget, memory, _) = authority(2 * cap, None);
    let spec = fixture(false);
    let (sizes, _) = validate_geometry(&spec, &budget, None).unwrap();
    let request = compile_prediction(&spec, &sizes, &budget, None).unwrap();
    let held = cap - request.scratch_bytes - request.persistent_bytes;
    let source = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: held,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let mut admitted = lease_compilation(&budget, held, request, None).unwrap();
    assert_eq!(
        budget.receipt().unwrap().live_scratch_bytes
            + budget.receipt().unwrap().live_persistent_bytes,
        cap
    );
    admitted.complete().unwrap();
    drop(admitted);
    let before = budget.receipt().unwrap();
    assert!(lease_compilation(&budget, usize::MAX, request, None).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, held);
    drop(source);
    no_live_claims(&budget, &memory);
}

#[test]
fn original_virtual_conv_preserves_signed_bits_fold_budgets_and_independent_zero_padding_geometry()
{
    let (budget, memory, _) = authority(64 << 20, None);
    let mut spec = fixture(false);
    if let TwinOpSpec::Conv { weight, .. } = &mut spec.ops[0] {
        weight[0] = -0.0;
        weight[1] = f64::from_bits(1);
    }
    let model = OriginalModel::compile(spec, &budget, None).unwrap();
    let TwinOp::Conv(original) = &model.net().ops[0] else {
        panic!("original convolution");
    };
    let ishape = (1, 3, 4);
    let oshape = (2, 2, 2);
    let bytes = virtual_conv_allocation_bytes(original, ishape, oshape).unwrap();
    let mut paid = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: bytes,
                work: 1_000_000,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let built = virtual_conv_from_original(original, ishape, oshape, &budget, None).unwrap();
    paid.mark_materialized().unwrap();
    assert_eq!(
        built.gather,
        vec![0, 2, 4, 6, 1, 3, 5, 7, 4, 6, 8, 10, 5, 7, 9, 11]
    );
    assert_eq!(built.pads, (0, 0, 0, 0));
    assert_eq!(built.stride, original.stride);
    assert_eq!(built.k_bwd, original.k_bwd);
    assert_eq!(
        built.weight_rel_err.to_bits(),
        original.weight_rel_err.to_bits()
    );
    for (actual, expected) in built.wmat.iter().zip(&original.wmat) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for (actual, expected) in built.bias.iter().zip(&original.bias) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    for (actual, expected) in built.bias_err.iter().zip(&original.bias_err) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    let transposed: Vec<_> = [0, 4, 1, 5, 2, 6, 3, 7]
        .map(|i| original.wmat[i].to_bits())
        .into_iter()
        .collect();
    assert_eq!(
        built.wt.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        transposed
    );
    let actual = size_of::<ConvOp>()
        + (built.wmat.capacity()
            + built.wt.capacity()
            + built.bias.capacity()
            + built.bias_err.capacity())
            * size_of::<f64>()
        + built.gather.capacity() * size_of::<usize>()
        + built.back_taps.capacity() * size_of::<Vec<(usize, usize, usize)>>()
        + built
            .back_taps
            .iter()
            .map(|v| v.capacity() * size_of::<(usize, usize, usize)>())
            .sum::<usize>();
    assert!(actual <= bytes);
    paid.complete().unwrap();
    drop(built);
    drop(paid);
    drop(model);
    no_live_claims(&budget, &memory);
}

#[test]
fn original_virtual_conv_invalid_mapping_transpose_expiry_and_cancellation_refuse() {
    let (budget, memory, cancellation) = authority(64 << 20, None);
    let model = OriginalModel::compile(fixture(false), &budget, None).unwrap();
    let TwinOp::Conv(original) = &model.net().ops[0] else {
        panic!("original convolution");
    };
    let baseline = budget.receipt().unwrap().live_persistent_bytes;
    for (input, output) in [
        ((2, 3, 4), (2, 2, 2)),
        ((1, 2, 4), (2, 2, 2)),
        ((1, 3, 4), (1, 2, 2)),
        ((1, 0, 4), (2, 2, 2)),
        ((1, isize::MAX as usize, isize::MAX as usize), (2, 2, 2)),
    ] {
        assert!(virtual_conv_from_original(original, input, output, &budget, None).is_err());
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, baseline);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }
    assert!(virtual_conv_from_original(
        original,
        (1, 3, 4),
        (2, 2, 2),
        &budget,
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    let transposed = OriginalModel::compile(fixture(true), &budget, None).unwrap();
    let TwinOp::Conv(original_transposed) = &transposed.net().ops[0] else {
        panic!("original convolution");
    };
    assert!(
        virtual_conv_from_original(original_transposed, (1, 3, 4), (2, 2, 2), &budget, None)
            .is_err()
    );
    cancellation.cancel();
    assert!(virtual_conv_from_original(original, (1, 3, 4), (2, 2, 2), &budget, None).is_err());
    drop(transposed);
    drop(model);
    no_live_claims(&budget, &memory);
}

#[test]
fn original_test_repricing_requires_paid_same_ledger_sole_owner_and_preserves_numerical_identity() {
    let (budget, memory, _) = authority(64 << 20, None);
    let model = OriginalModel::compile(fixture(false), &budget, None).unwrap();
    let identity = model.identity().clone();
    let actual = model.actual_bytes();
    let net = std::ptr::from_ref::<TwinNet>(model.net());
    let bytes = model.resident_bytes() + 4096;
    let claim = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: bytes,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    let model = model.reprice_for_test(claim).unwrap();
    assert_eq!(model.identity(), &identity);
    assert_eq!(model.actual_bytes(), actual);
    assert_eq!(std::ptr::from_ref::<TwinNet>(model.net()), net);
    assert_eq!(model.resident_bytes(), bytes);
    let duplicate = model.clone();
    let claim = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: bytes,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    assert!(model.reprice_for_test(claim).is_err());
    drop(duplicate);
    no_live_claims(&budget, &memory);
    let model = OriginalModel::compile(fixture(false), &budget, None).unwrap();
    let (foreign, foreign_memory, _) = authority(64 << 20, None);
    let claim = foreign
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: bytes,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    assert!(model.reprice_for_test(claim).is_err());
    no_live_claims(&budget, &memory);
    no_live_claims(&foreign, &foreign_memory);
}

#[test]
fn original_owner_moves_parameters_preserves_fold_bits_and_shares_last_claim() {
    let (budget, memory, _) = authority(64 << 20, None);
    let spec = fixture(false);
    let conv_ptr = match &spec.ops[0] {
        TwinOpSpec::Conv { weight, .. } => weight.as_ptr(),
        _ => unreachable!(),
    };
    let (head_ptr, error_ptr) = match &spec.ops[5] {
        TwinOpSpec::Gemm {
            weight, bias_err, ..
        } => (weight.as_ptr(), bias_err.as_ptr()),
        _ => unreachable!(),
    };
    let model = OriginalModel::compile(spec, &budget, None).unwrap();
    let shared = model.clone();
    assert!(model.same_model(&shared));
    assert!(std::ptr::eq(model.net(), shared.net()));
    assert!(model.budget().shares_ledger(&budget));
    match &model.net().ops[0] {
        TwinOp::Conv(c) => {
            assert_eq!(c.wmat.as_ptr(), conv_ptr);
            assert_eq!(c.weight_rel_err.to_bits(), 0.003f64.to_bits());
        }
        _ => unreachable!(),
    }
    match &model.net().ops[5] {
        TwinOp::Gemm {
            weight, bias_err, ..
        } => {
            assert_eq!(weight.as_ptr(), head_ptr);
            assert_eq!(bias_err.as_ptr(), error_ptr);
        }
        _ => unreachable!(),
    }
    let claim = model.resident_bytes();
    assert!(model.actual_bytes() <= claim);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, claim);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, claim);
    drop(model);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, claim);
    drop(shared);
    no_live_claims(&budget, &memory);
}

#[test]
fn original_compilation_matches_legacy_padded_stride_and_transpose_geometry() {
    for transposed in [false, true] {
        let (budget, _, _) = authority(64 << 20, None);
        let spec = fixture(transposed);
        let oracle = TwinNet::compile(&spec).unwrap();
        let model = OriginalModel::compile(spec, &budget, None).unwrap();
        let net = model.net();
        assert_eq!(net.tsize, oracle.tsize);
        assert_eq!(net.trunk_relus, oracle.trunk_relus);
        assert_eq!(
            (net.n_in, net.n_y, net.n_out, net.i_gemm1),
            (oracle.n_in, oracle.n_y, oracle.n_out, oracle.i_gemm1)
        );
        match (&net.ops[0], &oracle.ops[0]) {
            (TwinOp::Conv(a), TwinOp::Conv(b)) => {
                assert_eq!(a.gather, b.gather);
                assert_eq!(a.back_taps, b.back_taps);
                assert_eq!(a.wmat, b.wmat);
                assert_eq!(a.wt, b.wt);
                assert_eq!(
                    (a.k_fwd, a.k_bwd, a.transposed),
                    (b.k_fwd, b.k_bwd, b.transposed)
                );
                assert_eq!(
                    (a.kernel, a.ishape, a.oshape, a.stride, a.pads),
                    (b.kernel, b.ishape, b.oshape, b.stride, b.pads)
                );
                assert!(a.gather.contains(&usize::MAX));
            }
            _ => unreachable!(),
        }
        match (&net.ops[3], &oracle.ops[3]) {
            (
                TwinOp::ChannelAffine {
                    scale: a,
                    shift: b,
                    shift_err: c,
                    scale_rel_err: d,
                    ..
                },
                TwinOp::ChannelAffine {
                    scale: e,
                    shift: f,
                    shift_err: g,
                    scale_rel_err: h,
                    ..
                },
            ) => {
                assert_eq!((a, b, c, d), (e, f, g, h));
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn original_owner_ignores_head_helper_input_class_and_width_caps() {
    let (budget, _, _) = authority(64 << 20, None);
    for (n, head, classes) in [(5000, 1, 101), (1, 513, 101)] {
        let model = OriginalModel::compile(wide(n, head, classes), &budget, None).unwrap();
        assert_eq!(
            (model.net().n_in, model.net().n_y, model.net().n_out),
            (n, head, classes)
        );
    }
}

#[test]
fn equal_content_does_not_mint_same_owner_and_fold_bits_change_identity() {
    let (budget, _, _) = authority(64 << 20, None);
    let a = OriginalModel::compile(fixture(false), &budget, None).unwrap();
    let b = OriginalModel::compile(fixture(false), &budget, None).unwrap();
    assert_eq!(a.identity(), b.identity());
    assert!(!a.same_model(&b));
    let mut changed = fixture(false);
    if let TwinOpSpec::Conv { bias_err, .. } = &mut changed.ops[0] {
        bias_err[0] *= 2.0;
    }
    let c = OriginalModel::compile(changed, &budget, None).unwrap();
    assert_ne!(a.identity().source_digest(), c.identity().source_digest());
    assert_ne!(
        a.identity().compiled_digest(),
        c.identity().compiled_digest()
    );
    assert_eq!(
        a.identity().contract(),
        OriginalParameterContract::Binary64WithFoldUncertainty
    );
}

#[test]
fn source_layout_and_signed_zero_are_part_of_actual_model_identity() {
    let (budget, _, _) = authority(64 << 20, None);
    let spec = wide(2, 1, 2);
    let a = OriginalModel::compile(spec.clone(), &budget, None).unwrap();
    let mut changed = spec;
    if let TwinOpSpec::Gemm { bias, .. } = &mut changed.ops[5] {
        bias[0] = -0.0;
    }
    let b = OriginalModel::compile(changed, &budget, None).unwrap();
    assert_ne!(a.identity().source_digest(), b.identity().source_digest());
    assert_ne!(
        a.identity().compiled_digest(),
        b.identity().compiled_digest()
    );
    let c = OriginalModel::compile(fixture(true), &budget, None).unwrap();
    let d = OriginalModel::compile(fixture(false), &budget, None).unwrap();
    assert_ne!(c.identity().source_digest(), d.identity().source_digest());
}

#[test]
fn source_vector_capacity_is_priced_and_large_parameter_buffers_move_once() {
    let (budget, _, _) = authority(64 << 20, None);
    let mut spec = fixture(false);
    let base = spec_capacity_bytes(&spec, &budget, None).unwrap();
    let (ptr, extra) = match &mut spec.ops[0] {
        TwinOpSpec::Conv { weight, .. } => {
            let previous = weight.capacity();
            weight.reserve_exact(8000);
            (weight.as_ptr(), (weight.capacity() - previous) * 8)
        }
        _ => unreachable!(),
    };
    assert_eq!(
        spec_capacity_bytes(&spec, &budget, None).unwrap(),
        base + extra
    );
    let model = OriginalModel::compile(spec, &budget, None).unwrap();
    match &model.net().ops[0] {
        TwinOp::Conv(c) => assert_eq!(c.wmat.as_ptr(), ptr),
        _ => unreachable!(),
    }
    assert!(model.actual_bytes() >= extra);
    assert!(model.resident_bytes() >= model.actual_bytes());
}

#[test]
fn compilation_cap_refusal_is_atomic_and_keeps_other_owner_claim() {
    let (budget, memory, _) = authority(1 << 20, None);
    let keep = OriginalModel::compile(wide(1, 1, 2), &budget, None).unwrap();
    let before = budget.receipt().unwrap().live_persistent_bytes;
    let pending_before = memory.receipt().unwrap().live_pending_bytes;
    assert!(OriginalModel::compile(wide(5000, 20, 101), &budget, None).is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, pending_before);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, before);
    assert!(budget.receipt().unwrap().refused > 0);
    drop(keep);
    no_live_claims(&budget, &memory);
}

#[test]
fn one_byte_below_predicted_persistent_bound_refuses_before_publication() {
    let (inspect, _, _) = authority(64 << 20, None);
    let spec = fixture(false);
    let (sizes, _) = validate_geometry(&spec, &inspect, None).unwrap();
    let predicted = compile_prediction(&spec, &sizes, &inspect, None)
        .unwrap()
        .persistent_bytes;
    let memory = MemoryAuthority::new(64 << 20);
    let cancellation = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancellation,
        WorkLimits {
            deadline: None,
            scratch_bytes: 32 << 20,
            persistent_bytes: predicted - 1,
            max_work: 1_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    assert!(OriginalModel::compile(spec, &budget, None).is_err());
    assert_eq!(budget.receipt().unwrap().refused, 1);
    no_live_claims(&budget, &memory);
}

#[test]
fn expired_requested_or_inherited_deadline_cannot_be_renewed() {
    let past = Instant::now().checked_sub(Duration::from_secs(1)).unwrap();
    let future = Instant::now() + Duration::from_secs(30);
    let (budget, memory, _) = authority(64 << 20, None);
    assert!(matches!(
        OriginalModel::compile(fixture(false), &budget, Some(past)),
        Err(NyError::DeadlineExceeded(_))
    ));
    no_live_claims(&budget, &memory);
    let (budget, memory, _) = authority(64 << 20, Some(past));
    assert!(matches!(
        OriginalModel::compile(fixture(false), &budget, Some(future)),
        Err(NyError::DeadlineExceeded(_))
    ));
    no_live_claims(&budget, &memory);
}

#[test]
fn model_and_table_builders_poll_the_same_cancellation_authority() {
    let (budget, memory, cancellation) = authority(64 << 20, None);
    let model = OriginalModel::compile(fixture(false), &budget, None).unwrap();
    let claim = model.resident_bytes();
    cancellation.cancel();
    assert!(model.deadline(None).is_err());
    assert!(OriginalModel::compile(fixture(true), &budget, None).is_err());
    assert!(conv(
        0,
        vec![1.0],
        vec![0.0],
        vec![0.0],
        0.0,
        (1, 1, 1, 1),
        (1, 1),
        (0, 0, 0, 0),
        (1, 1, 1),
        (1, 1, 1),
        false,
        &budget,
        None
    )
    .is_err());
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, claim);
    drop(model);
    no_live_claims(&budget, &memory);
}

#[test]
fn malformed_original_geometry_parameters_and_final_fold_fail_atomically() {
    let (budget, memory, _) = authority(64 << 20, None);
    for defect in 0..5 {
        let mut spec = fixture(false);
        match defect {
            0 => spec.ops[1] = TwinOpSpec::Relu { input: 2 },
            1 => {
                if let TwinOpSpec::Conv { kernel, .. } = &mut spec.ops[0] {
                    kernel.2 = usize::MAX;
                }
            }
            2 => {
                if let TwinOpSpec::Conv { weight, .. } = &mut spec.ops[0] {
                    weight[0] = f64::NAN;
                }
            }
            3 => {
                if let TwinOpSpec::Gemm { bias_err, .. } = &mut spec.ops[7] {
                    *bias_err = vec![0.001; 2];
                }
            }
            4 => {
                if let TwinOpSpec::Conv { bias_err, .. } = &mut spec.ops[0] {
                    bias_err[0] = -0.001;
                }
            }
            _ => unreachable!(),
        }
        assert!(OriginalModel::compile(spec, &budget, None).is_err());
        no_live_claims(&budget, &memory);
    }
}

#[test]
fn independently_created_work_ledger_does_not_impersonate_model_budget() {
    let (budget, _, _) = authority(64 << 20, None);
    let (other, _, _) = authority(64 << 20, None);
    let model = OriginalModel::compile(fixture(false), &budget, None).unwrap();
    assert!(!model.budget().shares_ledger(&other));
    assert!(!model.budget().shares_memory_authority(&other));
    let unjoined =
        WorkBudget::new(None, 1 << 20, 1 << 20, 1_000_000_000, 0, Duration::ZERO).unwrap();
    assert!(OriginalModel::compile(fixture(false), &unjoined, None).is_err());
    assert_eq!(unjoined.receipt().unwrap().attempted, 0);
}
