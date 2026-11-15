// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Original-source and numerical controls for the prepared dense adapter.

use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{LeaseClass, WorkBudget, WorkLimits, WorkRequest, WorkReserve};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::support_program::{OriginalModel, OriginalRoot};
use crate::margin_row::RootPrecisionRuntimePolicy;
use num_rational::BigRational;
use num_traits::Zero;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const CAP: usize = 64 << 20;

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

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

fn fixture(fold: bool, stable: bool) -> TwinSpec {
    TwinSpec {
        n_in: 2,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![0.5],
                bias: vec![if stable { 2.0 } else { 0.125 }],
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
                scale: vec![-1.75],
                shift: vec![0.25],
                shift_err: vec![if fold { 0.002 } else { 0.0 }],
                scale_rel_err: if fold { 0.003 } else { 0.0 },
                shape: (1, 1, 2),
            },
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![1.0, -0.5, 0.25, 0.75],
                bias: vec![0.125, -0.25],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![1.0, -1.0, -0.5, 0.75],
                bias: vec![0.0, 0.125],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (2, 2),
            },
        ],
    }
}

fn prepare_prefix(spec: TwinSpec, budget: &WorkBudget) -> PreparedPrefix {
    let model = OriginalModel::compile(spec, budget, None).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-0.75, -0.5],
        &[0.75, 1.0],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap();
    let input = root.input_box().unwrap();
    let conv = root.derive(0, &input, None).unwrap();
    let (prefix, _) = root
        .empty_prefix()
        .unwrap()
        .commit_relu(&conv, 1, None)
        .unwrap();
    prefix
}

fn wide(n: usize) -> TwinSpec {
    TwinSpec {
        n_in: n,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![0.5],
                bias: vec![0.125],
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
                weight: vec![0.25; n],
                bias: vec![0.0],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (1, n),
            },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.5; 101],
                bias: vec![0.0; 101],
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (101, 1),
            },
        ],
    }
}

fn rat(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}

fn original(x: f64, fold: bool, signs: [f64; 4]) -> BigRational {
    let error = if fold { 0.003 } else { 0.0 };
    let w = rat(0.5) * (rat(1.0) + rat(signs[0]) * rat(error));
    let b = rat(0.125) + rat(signs[1]) * rat(if fold { 0.001 } else { 0.0 });
    let h = (w * rat(x) + b).max(BigRational::zero());
    let scale = rat(-1.75) * (rat(1.0) + rat(signs[2]) * rat(error));
    scale * &h + rat(0.25) + rat(signs[3]) * rat(if fold { 0.002 } else { 0.0 }) + h
}

fn pointwise_control(fold: bool) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority();
    let prefix = prepare_prefix(fixture(fold, false), &budget);
    let at = deadline();
    let rows = run_prepared_prefix_input_rows(&prefix, &[1, 0], Some(at)).unwrap();
    rows.validate_prepared_prefix(&prefix, 4, &[1, 0]).unwrap();
    for x in [-0.75, -0.25, 0.25, 0.75] {
        for y in [-0.5, 0.0, 0.5, 1.0] {
            let input = [x, y];
            for bits in 0..if fold { 16 } else { 1 } {
                let signs = std::array::from_fn(|i| if bits & (1 << i) == 0 { -1.0 } else { 1.0 });
                for (column, target) in [1usize, 0].into_iter().enumerate() {
                    let value = original(input[target], fold, signs);
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
                        let eta = rat(row.pointwise_error(at).unwrap());
                        assert!(
                            if upper {
                                line + eta >= value
                            } else {
                                line - eta <= value
                            },
                            "fold={fold},input={input:?},target={target},upper={upper}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn prepared_dense_rows_bound_signed_residual_original_in_exact_rational_arithmetic() {
    pointwise_control(false);
}

#[test]
fn prepared_dense_rows_pay_independent_conv_affine_fold_uncertainty_over_declared_domain() {
    pointwise_control(true);
}

#[test]
fn prepared_and_legacy_original_stable_prefix_rows_are_bit_identical_in_both_directions() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority();
    let spec = fixture(true, true);
    let legacy_net = TwinNet::compile(&spec).unwrap();
    let legacy_root = RootGates::build(
        &legacy_net,
        &[-0.75, -0.5],
        &[0.75, 1.0],
        RoundMode::Outward,
        None,
    )
    .unwrap();
    let eng = BackwardEngine::new(&legacy_net, &legacy_root)
        .with_err_lane(ErrLane::PerCol)
        .with_gpu_conv_route(GpuConvRoute::Off);
    let prepared = prepare_prefix(spec, &budget);
    let gate = prepared.gate(0).unwrap();
    assert_eq!(gate.alpha, legacy_root.layers[0].alpha);
    assert_eq!(gate.s, legacy_root.layers[0].s);
    assert_eq!(gate.c, legacy_root.layers[0].c);
    let seed = Seed {
        s: ndarray::array![[0.0, 1.0], [1.0, 0.0]],
        e: None,
    };
    let at = deadline();
    let rows = run_prepared_prefix_input_rows(&prepared, &[1, 0], Some(at)).unwrap();
    for upper in [false, true] {
        let old = eng
            .run_prefix_with_deadline(
                &seed,
                4,
                if upper {
                    LaneDir::Upper
                } else {
                    LaneDir::Lower
                },
                at,
            )
            .unwrap();
        for column in 0..2 {
            let row = if upper {
                rows.upper(column)
            } else {
                rows.lower(column)
            }
            .unwrap();
            assert_eq!(row.bias().to_bits(), old.b[column].to_bits());
            assert_eq!(row.bias_error().to_bits(), old.eb[column].to_bits());
            for (i, a, e) in row.entries() {
                assert_eq!(a.to_bits(), old.a[[i, column]].to_bits());
                assert_eq!(e.to_bits(), old.e.as_ref().unwrap()[[i, column]].to_bits());
            }
        }
    }
}

#[test]
fn prepared_dense_scope_rejects_foreign_prefix_target_order_and_duplicate_or_out_of_range_ids() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority();
    let a = prepare_prefix(fixture(false, false), &budget);
    let b = prepare_prefix(fixture(false, false), &budget);
    let rows = run_prepared_prefix_input_rows(&a, &[1, 0], Some(deadline())).unwrap();
    assert!(rows.validate_prepared_prefix(&b, 4, &[1, 0]).is_err());
    assert!(rows.validate_prepared_prefix(&a, 4, &[0, 1]).is_err());
    assert!(rows.validate_prepared_prefix(&a, 1, &[1, 0]).is_err());
    let before = budget.receipt().unwrap().live_persistent_bytes;
    for targets in [&[0usize, 0][..], &[0, 2], &[]] {
        assert!(run_prepared_prefix_input_rows(&a, targets, Some(deadline())).is_err());
        assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
        assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    }
}

#[test]
fn prepared_dense_expiry_and_shared_cancellation_preserve_prior_owned_evidence() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, token) = authority();
    let prefix = prepare_prefix(fixture(false, false), &budget);
    let before = memory.receipt().unwrap().live_materialized_bytes;
    assert!(run_prepared_prefix_input_rows(
        &prefix,
        &[0],
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    assert!(run_prepared_prefix_input_rows(&prefix, &[0], None).is_err());
    token.cancel();
    assert!(run_prepared_prefix_input_rows(&prefix, &[0], Some(deadline())).is_err());
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, before);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn prepared_dense_one_byte_scratch_cap_refusal_is_atomic_before_rows_allocation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority();
    let prefix = prepare_prefix(fixture(false, false), &budget);
    let at = deadline();
    let request =
        prepared_dense_request(prefix.root().model().net(), 4, 2, 2, at, &prefix).unwrap();
    let before = budget.receipt().unwrap().live_persistent_bytes;
    let block = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                scratch_bytes: CAP / 2 - request.scratch_bytes + 1,
                ..WorkRequest::default()
            },
        )
        .unwrap();
    let scratch = budget.receipt().unwrap().live_scratch_bytes;
    assert!(run_prepared_prefix_input_rows(&prefix, &[0, 1], Some(at)).is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, scratch);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, before);
    drop(block);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn prepared_dense_rows_keep_original_model_input_and_prefix_claims_until_last_owner_drops() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority();
    let prefix = prepare_prefix(fixture(false, false), &budget);
    let before = budget.receipt().unwrap().live_persistent_bytes;
    let rows = run_prepared_prefix_input_rows(&prefix, &[1, 0], Some(deadline())).unwrap();
    let total = budget.receipt().unwrap().live_persistent_bytes;
    assert!(total > before);
    assert!(
        rows.resident_bytes().unwrap()
            >= 4 * 2 * 2 * 8 + 4 * 2 * 8 + super::super::input_rows::INPUT_ROWS_OWNER_BYTES
    );
    drop(prefix);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, total);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
    drop(rows);
    assert_eq!(memory.receipt().unwrap().live_materialized_bytes, 0);
    assert_eq!(memory.receipt().unwrap().live_pending_bytes, 0);
}

#[test]
fn prepared_dense_actual_original_input_and_class_counts_exceed_native_head_helper_caps() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority();
    let n = 5000;
    let model = OriginalModel::compile(wide(n), &budget, None).unwrap();
    let root = OriginalRoot::new(
        model,
        &vec![-0.25; n],
        &vec![0.5; n],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap();
    let prefix = root.empty_prefix().unwrap();
    let rows = run_prepared_prefix_input_rows(&prefix, &[n - 1, 0], Some(deadline())).unwrap();
    assert_eq!(rows.n_in(), n);
    assert_eq!(
        rows.prepared_prefix().unwrap().root().model().net().n_out,
        101
    );
    for (column, target) in [n - 1, 0].into_iter().enumerate() {
        let nonzero: Vec<_> = rows
            .lower(column)
            .unwrap()
            .entries()
            .filter(|(_, a, _)| *a != 0.0)
            .collect();
        assert_eq!(nonzero.len(), 1);
        assert_eq!((nonzero[0].0, nonzero[0].1), (target, 0.5));
    }
}

#[test]
fn prepared_dense_refuses_frozen_envelope_policy_and_oversized_tile_without_bypassing_request() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority();
    let model = OriginalModel::compile(fixture(false, false), &budget, None).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-0.75, -0.5],
        &[0.75, 1.0],
        RootPrecisionRuntimePolicy {
            err_lane: ErrLane::PerChannel,
            ..RootPrecisionRuntimePolicy::default()
        },
        None,
    )
    .unwrap();
    let prefix = root.empty_prefix().unwrap();
    assert!(run_prepared_prefix_input_rows(&prefix, &[0], Some(deadline())).is_err());
    let regular = prepare_prefix(fixture(false, false), &budget);
    let before = budget.receipt().unwrap().live_persistent_bytes;
    assert!(run_prepared_prefix_input_rows(&regular, &vec![0; 65], Some(deadline())).is_err());
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, before);
}

#[test]
fn cooperative_nominal_and_error_contractions_preserve_legacy_bits_and_cancel_inside_worker_arithmetic(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, token) = authority();
    let model = OriginalModel::compile(wide(64), &budget, None).unwrap();
    let TwinOp::Conv(conv) = &model.net().ops[0] else {
        unreachable!()
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let src = Array2::from_shape_fn((64, 3), |(i, j)| {
        if i % 3 == 0 {
            f64::from_bits(1 + j as u64)
        } else {
            (i as f64 - 31.0) * (j as f64 + 0.25)
        }
    });
    let error = Array2::from_shape_fn(
        (64, 3),
        |(i, _)| if i % 3 == 0 { f64::from_bits(1) } else { 0.001 },
    );
    let at = deadline();
    pool.install(|| {
        let mut old = Array2::zeros((64, 3));
        let mut new = old.clone();
        conv_apply_backward_with_deadline(conv, &src, &mut old, false, at).unwrap();
        super::super::net::conv_apply_backward_with_checkpoint(
            conv,
            &src,
            &mut new,
            false,
            at,
            &|| {
                ny_core::require_f64_interval_proof_environment()?;
                budget
                    .checkpoint_for(LeaseClass::Fallback)
                    .map_err(prepared_refusal)
            },
        )
        .unwrap();
        assert_eq!(
            old.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            new.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        error_conv_with_deadline(conv, &error, &src, &mut old, at).unwrap();
        super::super::net::error_conv_with_checkpoint(conv, &error, &src, &mut new, at, &|| {
            budget
                .checkpoint_for(LeaseClass::Fallback)
                .map_err(prepared_refusal)
        })
        .unwrap();
        assert_eq!(
            old.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            new.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let calls = AtomicUsize::new(0);
        let mut partial = Array2::from_elem((64, 3), 7.0);
        let result = super::super::net::conv_apply_backward_with_checkpoint(
            conv,
            &src,
            &mut partial,
            false,
            at,
            &|| {
                if calls.fetch_add(1, Ordering::SeqCst) == 30 {
                    token.cancel();
                }
                budget
                    .checkpoint_for(LeaseClass::Fallback)
                    .map_err(prepared_refusal)
            },
        );
        assert!(result.is_err());
        assert!(token.is_cancelled());
        assert!(partial.iter().any(|&x| x != 7.0));
        assert!(partial.iter().any(|&x| x == 7.0));
    });
    assert_eq!(
        budget.receipt().unwrap().live_persistent_bytes,
        model.resident_bytes()
    );
}
