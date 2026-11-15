// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Actual complete original root sources, row delivery and shared lifetimes.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::input_rows::InputRowsBacking;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::support_program::OriginalModel;
use crate::margin_row::RootPrecisionRuntimePolicy;
use num_rational::BigRational;
use num_traits::Zero;
use std::time::Duration;

const CAP: usize = 256 << 20;
fn authority(cap: usize) -> (WorkBudget, MemoryAuthority, CancellationToken) {
    let memory = MemoryAuthority::new(cap.checked_mul(2).unwrap());
    let cancel = CancellationToken::default();
    let budget = WorkBudget::new_in(
        &memory,
        &cancel,
        WorkLimits {
            deadline: None,
            scratch_bytes: cap,
            persistent_bytes: cap,
            max_work: 10_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory, cancel)
}
fn at() -> Instant {
    Instant::now() + Duration::from_secs(45)
}
fn live(memory: &MemoryAuthority) -> usize {
    let r = memory.receipt().unwrap();
    r.live_pending_bytes + r.live_materialized_bytes
}
fn tail(ops: &mut Vec<TwinOpSpec>, input: usize, n: usize) {
    let flattened = ops.len() + 1;
    ops.push(TwinOpSpec::Flatten { input });
    let g = ops.len() + 1;
    ops.push(TwinOpSpec::Gemm {
        input: flattened,
        weight: vec![0.25; n],
        bias: vec![0.0],
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (1, n),
    });
    let r = ops.len() + 1;
    ops.push(TwinOpSpec::Relu { input: g });
    ops.push(TwinOpSpec::Gemm {
        input: r,
        weight: vec![0.5; 2],
        bias: vec![0.0; 2],
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (2, 1),
    });
}
fn conv(n: usize, input: usize, bias: f64, fold: bool) -> TwinOpSpec {
    TwinOpSpec::Conv {
        input,
        weight: vec![1.0],
        bias: vec![bias],
        bias_err: vec![if fold { 0.001 } else { 0.0 }],
        weight_rel_err: if fold { 0.003 } else { 0.0 },
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, n),
        oshape: (1, 1, n),
    }
}
fn residual(n: usize, scale: f64, shift: f64, fold: bool) -> TwinSpec {
    let mut ops = vec![
        conv(n, 0, 0.0, fold),
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::ChannelAffine {
            input: 2,
            scale: vec![scale],
            shift: vec![shift],
            shift_err: vec![if fold { 0.002 } else { 0.0 }],
            scale_rel_err: if fold { 0.003 } else { 0.0 },
            shape: (1, 1, n),
        },
        TwinOpSpec::Add { lhs: 3, rhs: 2 },
        TwinOpSpec::Relu { input: 4 },
    ];
    tail(&mut ops, 5, n);
    TwinSpec { n_in: n, ops }
}
fn root(spec: TwinSpec, budget: &WorkBudget) -> OriginalRoot {
    let n = spec.n_in;
    let model = OriginalModel::compile(spec, budget, None).unwrap();
    OriginalRoot::new(
        model,
        &vec![-1.0; n],
        &vec![1.0; n],
        RootPrecisionRuntimePolicy::default(),
        None,
    )
    .unwrap()
}
fn keep() -> PreparedConstructionOptions {
    PreparedConstructionOptions {
        keep_input_rows: true,
        ..PreparedConstructionOptions::default()
    }
}
fn rational(x: f64) -> BigRational {
    BigRational::from_float(x).unwrap()
}
fn bits(xs: &[f64]) -> Vec<u64> {
    xs.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn construction_residual_dependency_cancellation_changes_crossing_reference_to_active_original_gate(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let root = root(residual(5, -1.0, 0.25, false), &budget);
    let input = root.input_box().unwrap();
    let cv = root.derive(0, &input, None).unwrap();
    let h = root.derive(1, &cv, None).unwrap();
    let affine = root.derive(2, &h, None).unwrap();
    let loose = root.derive(3, &affine, Some(&h)).unwrap();
    assert!(loose.lo().iter().all(|v| *v < 0.0));
    assert!(loose.hi().iter().all(|v| *v > 0.0));
    drop((input, cv, h, affine, loose));
    let source = build_prepared_root(root, keep(), Some(at())).unwrap();
    let gate = source.prefix().gate(1).unwrap();
    assert_eq!((gate.unst).len(), 0);
    assert!(gate.l.iter().all(|v| *v > 0.0));
    assert!(gate.u.iter().all(|v| *v < 0.2500001));
    assert!(source.layer_products(1).unwrap().tiles().is_empty());
    assert!(source.stats().refined_rows >= 10);
    assert!(source.coverage_complete(None).unwrap());
}

#[test]
fn construction_varied_dense_tiles_preserve_endpoint_bits_and_original_unstable_mapping() {
    let _env = crate::tests::lock_env_shared();
    let mut endpoints = None;
    for width in [1, 2, 3, 64] {
        let (budget, _, _) = authority(CAP);
        let options = PreparedConstructionOptions {
            prefer_patch: false,
            dense_tile: width,
            ..keep()
        };
        let source = build_prepared_root(
            root(residual(7, -0.5, -0.1, false), &budget),
            options,
            Some(at()),
        )
        .unwrap();
        let current = (
            bits(source.trunk_output().lo()),
            bits(source.trunk_output().hi()),
        );
        if let Some(expected) = &endpoints {
            assert_eq!(&current, expected);
        } else {
            endpoints = Some(current);
        }
        assert_eq!(
            source
                .layer_products(1)
                .unwrap()
                .coverage()
                .iter()
                .map(|p| p.target)
                .collect::<Vec<_>>(),
            (0..7).collect::<Vec<_>>()
        );
        assert!(source.stats().patch_tiles == 0 && source.stats().dense_tiles > 0);
        assert!(source.coverage_complete(None).unwrap());
    }
}

#[test]
fn construction_patch_and_dense_sources_contain_exact_signed_original_residual_values() {
    let _env = crate::tests::lock_env_shared();
    for patch in [false, true] {
        let (budget, _, _) = authority(CAP);
        let options = PreparedConstructionOptions {
            prefer_patch: patch,
            dense_tile: 2,
            patch_tile: 3,
            patch_ratio: 1.0,
            ..keep()
        };
        let source = build_prepared_root(
            root(residual(9, -0.5, -0.1, false), &budget),
            options,
            Some(at()),
        )
        .unwrap();
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let h = rational(x).max(BigRational::zero());
            let exact = (rational(0.5) * h + rational(-0.1)).max(BigRational::zero());
            for i in 0..9 {
                assert!(rational(source.trunk_output().lo()[i]) <= exact);
                assert!(exact <= rational(source.trunk_output().hi()[i]));
            }
        }
        if patch {
            assert!(source.stats().patch_tiles > 0);
        } else {
            assert!(source.stats().dense_tiles > 0);
        }
        assert!(source.coverage_complete(None).unwrap());
    }
}

#[test]
fn construction_folded_original_model_independent_parameter_errors_remain_rationally_contained() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let source = build_prepared_root(
        root(residual(2, -1.0, 0.25, true), &budget),
        keep(),
        Some(at()),
    )
    .unwrap();
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        for b in 0_usize..16 {
            let sign = |k: usize| {
                if b & (1_usize << k) == 0_usize {
                    -1.0
                } else {
                    1.0
                }
            };
            let h = ((rational(1.0) + rational(sign(0)) * rational(0.003)) * rational(x)
                + rational(sign(1)) * rational(0.001))
            .max(BigRational::zero());
            let exact = (-(rational(1.0) + rational(sign(2)) * rational(0.003)) * &h
                + rational(0.25)
                + rational(sign(3)) * rational(0.002)
                + h)
                .max(BigRational::zero());
            for i in 0..2 {
                assert!(
                    rational(source.trunk_output().lo()[i]) <= exact
                        && exact <= rational(source.trunk_output().hi()[i])
                );
            }
        }
    }
}

#[test]
fn construction_binary32_reference_preserves_exact_folded_residual_and_paid_coverage() {
    let _env = crate::tests::lock_env_shared();
    for patch in [false, true] {
        let (budget, memory, _) = authority(CAP);
        let source = build_prepared_root(
            root(residual(2, -1.0, 0.25, true), &budget),
            PreparedConstructionOptions {
                reference_arithmetic: PreparedReferenceArithmetic::Binary32Certified,
                prefer_patch: patch,
                ..keep()
            },
            Some(at()),
        )
        .unwrap();
        let r = source.reference_receipt();
        assert_eq!(r.arithmetic, PreparedReferenceArithmetic::Binary32Certified);
        assert_eq!((r.expected_conv_ops, r.completed_conv_ops), (1, 1));
        assert_eq!((r.expected_affine_ops, r.completed_affine_ops), (1, 1));
        assert_eq!((r.binary32_conv_dot_ops, r.binary32_affine_dot_ops), (1, 1));
        assert!(r.multiply_add_pairs > 0 && r.converted_cells > 0);
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            for bits in 0usize..16 {
                let sign = |k| {
                    if bits & (1usize << k) == 0usize {
                        -1.0
                    } else {
                        1.0
                    }
                };
                let h = ((rational(1.0) + rational(sign(0)) * rational(0.003)) * rational(x)
                    + rational(sign(1)) * rational(0.001))
                .max(BigRational::zero());
                let exact = (-(rational(1.0) + rational(sign(2)) * rational(0.003)) * &h
                    + rational(0.25)
                    + rational(sign(3)) * rational(0.002)
                    + h)
                    .max(BigRational::zero());
                for i in 0..2 {
                    assert!(rational(source.trunk_output().lo()[i]) <= exact);
                    assert!(exact <= rational(source.trunk_output().hi()[i]));
                }
            }
        }
        assert!(source.coverage_complete(None).unwrap());
        source.validate_reference(None).unwrap();
        assert!(!source.prefix().root().policy().root_f32);
        assert!(!source.prefix().root().policy().tableau_f32);
        drop(source);
        assert_eq!(live(&memory), 0);
    }
}

#[test]
fn construction_binary32_nominal_overflow_refuses_without_f64_retry_or_leaked_claims() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP);
    let mut spec = residual(1, 1.0, 0.0, false);
    let TwinOpSpec::Conv { weight, .. } = &mut spec.ops[0] else {
        unreachable!()
    };
    weight[0] = f64::from(f32::MAX) * 2.0;
    let original = root(spec, &budget);
    let before = live(&memory);
    assert!(build_prepared_root(
        original.clone(),
        PreparedConstructionOptions {
            reference_arithmetic: PreparedReferenceArithmetic::Binary32Certified,
            ..Default::default()
        },
        Some(at())
    )
    .is_err());
    assert_eq!(live(&memory), before);
    let f64 = build_prepared_root(original.clone(), Default::default(), Some(at())).unwrap();
    assert_eq!(
        f64.reference_receipt().arithmetic,
        PreparedReferenceArithmetic::F64
    );
    drop((f64, original));
    assert_eq!(live(&memory), 0);
}

#[test]
fn construction_complete_coverage_has_actual_ordered_rows_and_shared_tiles_keep_last_owner_claims()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP);
    let source = build_prepared_root(
        root(residual(5, -0.5, -0.1, false), &budget),
        PreparedConstructionOptions {
            prefer_patch: false,
            dense_tile: 2,
            ..keep()
        },
        Some(at()),
    )
    .unwrap();
    for layer in source.layers() {
        let gate = source.prefix().gate(layer.layer()).unwrap();
        assert_eq!(layer.op(), gate.op);
        for (&target, mapped) in gate.unst.iter().zip(layer.coverage()) {
            assert_eq!(target, mapped.target);
            let (rows, column) = layer.row(target).unwrap();
            assert_eq!(rows.target(column), Some(target));
            rows.validate_prepared_prefix(layer.prefix_before(), layer.op(), rows.targets())
                .unwrap();
        }
    }
    let tile = source.layer_products(1).unwrap().tiles()[0].clone();
    drop(source);
    assert!(live(&memory) > 0);
    let rows = tile.rows();
    assert_eq!(rows.targets(), &[0, 1]);
    assert!(rows
        .lower(0)
        .unwrap()
        .pointwise_error(at())
        .unwrap()
        .is_finite());
    drop(tile);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn construction_no_persistent_row_request_never_mints_coverage_authority() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP);
    let source = build_prepared_root(
        root(residual(5, -0.5, -0.1, false), &budget),
        PreparedConstructionOptions::default(),
        Some(at()),
    )
    .unwrap();
    assert!(!source.coverage_complete(None).unwrap());
    assert!(source
        .layers()
        .iter()
        .all(|p| p.tiles().is_empty() && p.coverage().is_empty()));
    assert_ne!((source.prefix().gate(1).unwrap().unst).len(), 0);
    drop(source);
    assert_eq!(live(&memory), 0);
}

#[test]
fn construction_solved_tile_compaction_preserves_sparse_original_ids_and_actual_columns() {
    let _env = crate::tests::lock_env_shared();
    let n = 6;
    let shape = (n, 1, 1);
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: (0..n * n)
                .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
                .collect(),
            bias: vec![0.0; n],
            bias_err: vec![0.0; n],
            weight_rel_err: 0.0,
            kernel: (n, n, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: shape,
            oshape: shape,
        },
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::ChannelAffine {
            input: 2,
            scale: vec![-0.5; n],
            shift: (0..n)
                .map(|i| if i % 2 == 0 { 0.25 } else { -0.1 })
                .collect(),
            shift_err: vec![0.0; n],
            scale_rel_err: 0.0,
            shape,
        },
        TwinOpSpec::Add { lhs: 3, rhs: 2 },
        TwinOpSpec::Relu { input: 4 },
    ];
    tail(&mut ops, 5, n);
    let (budget, _, _) = authority(CAP);
    let source = build_prepared_root(
        root(TwinSpec { n_in: n, ops }, &budget),
        PreparedConstructionOptions {
            prefer_patch: false,
            dense_tile: 1,
            ..keep()
        },
        Some(at()),
    )
    .unwrap();
    let product = source.layer_products(1).unwrap();
    assert_eq!(source.prefix().gate(1).unwrap().unst, [1, 3, 5]);
    assert_eq!(product.tiles().len(), 3);
    for (index, target) in [1, 3, 5].into_iter().enumerate() {
        assert_eq!(
            product.coverage()[index],
            PreparedRowLocation {
                target,
                tile: index,
                column: 0
            }
        );
        let (rows, column) = product.row(target).unwrap();
        assert_eq!(rows.targets(), &[target]);
        assert_eq!(rows.target(column), Some(target));
    }
    for target in [0, 2, 4] {
        assert!(product.row(target).is_none());
    }
    assert!(source.coverage_complete(None).unwrap());
}

#[test]
fn construction_original_last_use_releases_reference_buffers_in_deep_trunk() {
    let _env = crate::tests::lock_env_shared();
    let n = 8;
    let mut ops = vec![conv(n, 0, 2.0, false), TwinOpSpec::Relu { input: 1 }];
    for _ in 0..32 {
        let input = ops.len();
        ops.push(TwinOpSpec::ChannelAffine {
            input,
            scale: vec![1.0],
            shift: vec![0.25],
            shift_err: vec![0.0],
            scale_rel_err: 0.0,
            shape: (1, 1, n),
        });
        let input = ops.len();
        ops.push(TwinOpSpec::Relu { input });
    }
    let input = ops.len();
    tail(&mut ops, input, n);
    let trunk_ops = ops.len() - 3;
    let (budget, memory, _) = authority(CAP);
    let source =
        build_prepared_root(root(TwinSpec { n_in: n, ops }, &budget), keep(), Some(at())).unwrap();
    assert_eq!(source.stats().derived_ops, trunk_ops);
    assert_eq!(source.stats().max_live_reference_boxes, 2);
    assert!(source.stats().released_reference_boxes >= trunk_ops);
    assert_eq!(source.stats().refined_rows, 0);
    assert!(source.coverage_complete(None).unwrap());
    drop(source);
    assert_eq!(live(&memory), 0);
}

#[test]
fn construction_head_input_identity_survives_later_unused_original_branch() {
    let _env = crate::tests::lock_env_shared();
    let n = 3;
    let mut ops = vec![
        conv(n, 0, 0.0, false),
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::Flatten { input: 2 },
        TwinOpSpec::ChannelAffine {
            input: 2,
            scale: vec![1.0],
            shift: vec![10.0],
            shift_err: vec![0.0],
            scale_rel_err: 0.0,
            shape: (1, 1, n),
        },
    ];
    ops.push(TwinOpSpec::Gemm {
        input: 3,
        weight: vec![0.25; n],
        bias: vec![0.0],
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (1, n),
    });
    ops.push(TwinOpSpec::Relu { input: 5 });
    ops.push(TwinOpSpec::Gemm {
        input: 6,
        weight: vec![0.5; 2],
        bias: vec![0.0; 2],
        bias_err: vec![],
        weight_rel_err: 0.0,
        shape: (2, 1),
    });
    let (budget, _, _) = authority(CAP);
    let source =
        build_prepared_root(root(TwinSpec { n_in: n, ops }, &budget), keep(), Some(at())).unwrap();
    assert_eq!(source.trunk_output().tensor(), 3);
    assert!(source.trunk_output().hi().iter().all(|v| *v < 1.000001));
    assert!(source.coverage_complete(None).unwrap());
}

#[test]
fn construction_real_f32_row_storage_has_paid_cells_scope_and_complete_final_coverage() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let source = build_prepared_root(
        root(residual(5, -0.5, -0.1, true), &budget),
        PreparedConstructionOptions {
            row_f32: true,
            prefer_patch: false,
            dense_tile: 2,
            ..keep()
        },
        Some(at()),
    )
    .unwrap();
    assert!(
        source.stats().f32_tiles > 0
            && source.stats().f32_coefficient_cells > 0
            && source.stats().f32_bias_cells > 0
    );
    assert!(source
        .layers()
        .iter()
        .flat_map(|p| p.tiles())
        .all(|tile| tile.rows().backing() == InputRowsBacking::F32));
    assert!(source.coverage_complete(None).unwrap());
    for i in 0..5 {
        assert!(source.trunk_output().lo()[i] <= 0.0);
        assert!(source.trunk_output().hi()[i] >= 0.4);
    }
}

#[test]
fn construction_product_and_precision_requests_refuse_without_partial_delivery() {
    let _env = crate::tests::lock_env_shared();
    for options in [
        PreparedConstructionOptions {
            row_f32: true,
            ..PreparedConstructionOptions::default()
        },
        PreparedConstructionOptions {
            dense_tile: 65,
            ..keep()
        },
        PreparedConstructionOptions {
            patch_tile: 2049,
            ..keep()
        },
        PreparedConstructionOptions {
            patch_ratio: f64::NAN,
            ..keep()
        },
    ] {
        let (budget, memory, _) = authority(CAP);
        let root = root(residual(2, -0.5, -0.1, false), &budget);
        let admitted = budget.receipt().unwrap().admitted;
        assert!(build_prepared_root(root, options, Some(at())).is_err());
        assert_eq!(budget.receipt().unwrap().admitted, admitted);
        assert_eq!(live(&memory), 0);
    }
    for root_f32 in [false, true] {
        let (budget, _, _) = authority(CAP);
        let model = OriginalModel::compile(residual(2, -0.5, -0.1, false), &budget, None).unwrap();
        let policy = RootPrecisionRuntimePolicy {
            root_f32,
            tableau_f32: !root_f32,
            ..RootPrecisionRuntimePolicy::default()
        };
        assert!(OriginalRoot::new(model, &[-1.0; 2], &[1.0; 2], policy, None).is_err());
    }
}

#[test]
fn construction_expiry_and_actual_mid_dag_cancellation_publish_nothing_and_release_all_new_claims()
{
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(CAP);
    assert!(build_prepared_root(
        root(residual(5, -0.5, -0.1, false), &budget),
        keep(),
        Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap())
    )
    .is_err());
    assert_eq!(live(&memory), 0);
    let (budget, memory, cancel) = authority(CAP);
    let root = root(residual(5, -0.5, -0.1, false), &budget);
    let mut observed = 0;
    assert!(build_inner(
        root,
        keep(),
        Some(at()),
        Some(&mut |op| {
            if op == 1 {
                observed += 1;
                cancel.cancel();
            }
        })
    )
    .is_err());
    assert_eq!(observed, 1);
    assert_eq!(live(&memory), 0);
    assert_eq!(budget.receipt().unwrap().live_scratch_bytes, 0);
}

#[test]
fn construction_atomic_total_ceiling_rejects_live_overpayment_before_registry_allocation() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory, _) = authority(ROOT_SOURCE_BYTES_CAP * 2);
    let root = root(residual(2, -0.5, -0.1, false), &budget);
    let owner = live(&memory);
    let blocker = budget
        .lease_for(
            LeaseClass::Fallback,
            WorkRequest {
                persistent_bytes: ROOT_SOURCE_BYTES_CAP + 1 - owner,
                ..WorkRequest::default()
            },
        )
        .unwrap()
        .commit_persistent()
        .unwrap();
    let before = budget.receipt().unwrap();
    let peak = memory.receipt().unwrap().peak_leased_bytes;
    assert!(build_prepared_root(root, keep(), Some(at())).is_err());
    assert_eq!(budget.receipt().unwrap().admitted, before.admitted);
    assert_eq!(memory.receipt().unwrap().peak_leased_bytes, peak);
    assert_eq!(live(&memory), blocker.bytes());
    drop(blocker);
    assert_eq!(live(&memory), 0);
}

#[test]
fn construction_layer_products_reject_foreign_completed_root_and_model_identity() {
    let _env = crate::tests::lock_env_shared();
    let (budget, _, _) = authority(CAP);
    let a = build_prepared_root(
        root(residual(2, -0.5, -0.1, false), &budget),
        keep(),
        Some(at()),
    )
    .unwrap();
    let b = build_prepared_root(
        root(residual(2, -0.5, -0.2, false), &budget),
        keep(),
        Some(at()),
    )
    .unwrap();
    assert_ne!(a.identity(), b.identity());
    let frame = b.prefix().frame(Some(at())).unwrap();
    assert!(a
        .layer_products(0)
        .unwrap()
        .validate(b.prefix(), frame.gate(0).unwrap(), at())
        .is_err());
    assert!(a.coverage_complete(None).unwrap());
    assert!(b.coverage_complete(None).unwrap());
}
