// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Actual original DAG controls for a second paid lower-row candidate.
use super::*;
use crate::certify::residency::{CancellationToken, MemoryAuthority};
use crate::certify::work::{WorkLimits, WorkReserve};
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::support_program::OriginalModel;
use crate::margin_row::RootPrecisionRuntimePolicy;
use num_rational::BigRational as Q;
use num_traits::Zero;
use std::time::Duration;

const N: usize = 4;
const CAP: usize = 128 << 20;

fn q(value: f64) -> Q {
    Q::from_float(value).unwrap()
}
fn at() -> Instant {
    Instant::now() + Duration::from_secs(45)
}
fn authority() -> (WorkBudget, MemoryAuthority) {
    let memory = MemoryAuthority::new(CAP * 2);
    let budget = WorkBudget::new_in(
        &memory,
        &CancellationToken::default(),
        WorkLimits {
            deadline: None,
            scratch_bytes: CAP,
            persistent_bytes: CAP,
            max_work: 1_000_000_000_000,
            max_proposals: 0,
            certificate_reserve: Duration::ZERO,
            reserve: WorkReserve::default(),
        },
    )
    .unwrap();
    (budget, memory)
}
fn original(budget: &WorkBudget, fold: bool) -> OriginalRoot {
    let shape = (N, 1, 1);
    let spec = TwinSpec {
        n_in: N,
        ops: vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: (0..N * N)
                    .map(|i| if i / N == i % N { 1.0 } else { 0.0 })
                    .collect(),
                bias: vec![0.0; N],
                bias_err: vec![if fold { 0.001 } else { 0.0 }; N],
                weight_rel_err: if fold { 0.003 } else { 0.0 },
                kernel: (N, N, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: shape,
                oshape: shape,
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![-0.5; N],
                shift: vec![0.25, -0.1, 0.25, -0.1],
                shift_err: vec![if fold { 0.002 } else { 0.0 }; N],
                scale_rel_err: if fold { 0.003 } else { 0.0 },
                shape,
            },
            TwinOpSpec::Add { lhs: 3, rhs: 2 },
            TwinOpSpec::Relu { input: 4 },
            TwinOpSpec::Flatten { input: 5 },
            TwinOpSpec::Gemm {
                input: 6,
                weight: vec![0.25; N],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, N),
            },
            TwinOpSpec::Relu { input: 7 },
            TwinOpSpec::Gemm {
                input: 8,
                weight: vec![0.5; 2],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    };
    let model = OriginalModel::compile(spec, budget, Some(at())).unwrap();
    OriginalRoot::new(
        model,
        &[-1.0; N],
        &[1.0; N],
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap()
}
fn options(patch: bool, tile: usize) -> PreparedConstructionOptions {
    PreparedConstructionOptions {
        keep_input_rows: true,
        prefer_patch: patch,
        dense_tile: tile,
        patch_tile: tile,
        patch_ratio: 1.0,
        ..PreparedConstructionOptions::default()
    }
}

#[test]
fn construction_alternative_lower_candidates_preserve_sparse_coverage_across_dense_and_patch_tiles()
{
    let _env = crate::tests::lock_env_shared();
    for (patch, tile) in [(false, 1), (false, 3), (true, 2), (true, 4)] {
        let (budget, _) = authority();
        let source =
            build_prepared_root(original(&budget, false), options(patch, tile), Some(at()))
                .unwrap();
        let gate = source.prefix().gate(1).unwrap();
        assert_eq!(gate.unst, [1, 3]);
        assert!(gate.l[0] > 0.2 && gate.l[2] > 0.2);
        let products = source.layer_products(1).unwrap();
        assert_eq!(
            products
                .coverage()
                .iter()
                .map(|p| p.target)
                .collect::<Vec<_>>(),
            [1, 3]
        );
        assert!(source.coverage_complete(None).unwrap());
        if patch {
            assert!(source.stats().patch_tiles > 0);
        }
        for target in [1, 3] {
            let (rows, column) = products.row(target).unwrap();
            assert_eq!(rows.target(column), Some(target));
            rows.validate_prepared_prefix(products.prefix_before(), 4, rows.targets())
                .unwrap();
            for k in -8..=8 {
                let mut input = [0.0; N];
                input[target] = f64::from(k) / 8.0;
                let exact = q(0.5) * q(input[target]).max(Q::zero()) + q(-0.1);
                for upper in [false, true] {
                    let row = if upper {
                        rows.upper(column)
                    } else {
                        rows.lower(column)
                    }
                    .unwrap();
                    let nominal = row
                        .entries()
                        .fold(q(row.bias()), |s, (i, a, _)| s + q(a) * q(input[i]));
                    let eta = q(row.pointwise_error(at()).unwrap());
                    assert!(if upper {
                        nominal + eta >= exact
                    } else {
                        nominal - eta <= exact
                    });
                }
            }
        }
    }
}

#[test]
fn construction_alternative_lower_candidates_keep_rational_fold_family_and_last_row_owner() {
    let _env = crate::tests::lock_env_shared();
    let (budget, memory) = authority();
    let source =
        build_prepared_root(original(&budget, true), options(true, 3), Some(at())).unwrap();
    let gate = source.prefix().gate(1).unwrap();
    assert_eq!(gate.unst, [1, 3]);
    assert!(gate.l[0] > 0.2 && gate.l[2] > 0.2);
    let products = source.layer_products(1).unwrap();
    let location = products.coverage()[0];
    let retained = products.tiles()[location.tile].clone();
    drop(source);
    let rows = retained.rows();
    assert_eq!(rows.target(location.column), Some(1));
    for bits in 0..16 {
        let signs: [f64; 4] =
            std::array::from_fn(|i| if bits & (1 << i) == 0 { -1.0 } else { 1.0 });
        for k in -8..=8 {
            let mut input = [0.0; N];
            input[1] = f64::from(k) / 8.0;
            let pre = (q(1.0) + q(signs[0]) * q(0.003)) * q(input[1]) + q(signs[1]) * q(0.001);
            let h = pre.max(Q::zero());
            let value = q(-0.5) * (q(1.0) + q(signs[2]) * q(0.003)) * &h
                + q(-0.1)
                + q(signs[3]) * q(0.002)
                + h;
            for upper in [false, true] {
                let row = if upper {
                    rows.upper(location.column)
                } else {
                    rows.lower(location.column)
                }
                .unwrap();
                let nominal = row
                    .entries()
                    .fold(q(row.bias()), |s, (i, a, _)| s + q(a) * q(input[i]));
                let eta = q(row.pointwise_error(at()).unwrap());
                assert!(if upper {
                    nominal + eta >= value
                } else {
                    nominal - eta <= value
                });
            }
        }
    }
    assert!(budget.receipt().unwrap().live_persistent_bytes > 0);
    drop(retained);
    assert_eq!(budget.receipt().unwrap().live_persistent_bytes, 0);
    let receipt = memory.receipt().unwrap();
    assert_eq!(
        receipt.live_pending_bytes + receipt.live_materialized_bytes,
        0
    );
}

#[test]
fn construction_alternative_lower_candidates_choose_complete_columns_and_keep_the_better_legacy_row(
) {
    let _env = crate::tests::lock_env_shared();
    let (budget, _) = authority();
    let shape = (2, 1, 1);
    // The first column needs alpha=0 to prove .5*ReLU(x)+.25 >= .25.
    // The second needs the legacy alpha=1 cancellation in ReLU(y)-y+.25.
    // A wholesale replacement by either policy misses one positive phase.
    let spec = TwinSpec {
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
                ishape: shape,
                oshape: shape,
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![0.5, 1.0],
                shift: vec![0.25; 2],
                shift_err: vec![0.0; 2],
                scale_rel_err: 0.0,
                shape,
            },
            TwinOpSpec::ChannelAffine {
                input: 0,
                scale: vec![0.0, -1.0],
                shift: vec![0.0; 2],
                shift_err: vec![0.0; 2],
                scale_rel_err: 0.0,
                shape,
            },
            TwinOpSpec::Add { lhs: 3, rhs: 4 },
            TwinOpSpec::Relu { input: 5 },
            TwinOpSpec::Flatten { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![0.25; 2],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 2),
            },
            TwinOpSpec::Relu { input: 8 },
            TwinOpSpec::Gemm {
                input: 9,
                weight: vec![0.5; 2],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    };
    let model = OriginalModel::compile(spec, &budget, Some(at())).unwrap();
    let root = OriginalRoot::new(
        model,
        &[-1.0; 2],
        &[1.0; 2],
        RootPrecisionRuntimePolicy::default(),
        Some(at()),
    )
    .unwrap();
    let source = build_prepared_root(root, options(false, 2), Some(at())).unwrap();
    let gate = source.prefix().gate(1).unwrap();
    assert_eq!((gate.unst).len(), 0);
    assert!(gate.l.iter().all(|&l| l > 0.2));
    assert!(source.coverage_complete(None).unwrap());
    assert_eq!((source.layer_products(1).unwrap().coverage()).len(), 0);
    for x in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        for y in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let exact = [
                q(0.5) * q(x).max(Q::zero()) + q(0.25),
                q(y).max(Q::zero()) - q(y) + q(0.25),
            ];
            for (i, value) in exact.into_iter().enumerate() {
                assert!(q(gate.l[i]) <= value && value <= q(gate.u[i]));
            }
        }
    }
}
