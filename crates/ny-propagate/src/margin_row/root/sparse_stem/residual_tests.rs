// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::margin_row::spec::{TwinOpSpec, TwinSpec};
use crate::margin_row::{
    BackwardIntermRuntimePolicy, ErrLane, RootBoxSource, RootPrecisionRuntimePolicy,
};
use num_rational::BigRational as Q;
use num_traits::{Signed, Zero};

fn q(x: f64) -> Q {
    Q::from_float(x).expect("finite exact oracle value")
}

fn policy(store: bool) -> RootPrecisionRuntimePolicy {
    RootPrecisionRuntimePolicy {
        root_f32: true,
        tableau_f32: store,
        err_lane: ErrLane::PerCol,
        ibp_seed: false,
        box_source: RootBoxSource::Tableau,
    }
}

fn build(
    net: &TwinNet,
    store: bool,
    lo: f64,
    hi: f64,
    splits: &[(usize, usize, i8)],
    deadline: Option<Instant>,
) -> Result<(RootGates, Option<RetainedRows>)> {
    let retain = RetainCfg {
        per_layer: 3,
        budget_bytes: 1 << 20,
    };
    let seed: Vec<Vec<f64>> = net
        .trunk_relus
        .iter()
        .map(|&k| vec![0.375; net.tsize[k + 1]])
        .collect();
    RootGates::build_retaining_with_policy(
        net,
        &vec![lo; net.n_in],
        &vec![hi; net.n_in],
        RoundMode::Outward,
        deadline,
        Some(&retain),
        splits,
        policy(store),
        BackwardIntermRuntimePolicy::default(),
        Some(&seed),
    )
}

fn same(a: &(RootGates, Option<RetainedRows>), b: &(RootGates, Option<RetainedRows>)) {
    assert_eq!(a.0.root_precision_execution, b.0.root_precision_execution);
    assert_eq!(a.0.layers.len(), b.0.layers.len());
    for (x, y) in a.0.layers.iter().zip(&b.0.layers) {
        assert_eq!((x.op, x.n, &x.unst), (y.op, y.n, &y.unst));
        for (u, v) in [
            (&x.l, &y.l),
            (&x.u, &y.u),
            (&x.alpha, &y.alpha),
            (&x.s, &y.s),
            (&x.c, &y.c),
            (&x.ms, &y.ms),
        ] {
            assert_eq!(u.len(), v.len());
            assert!(
                u.iter().zip(v).all(|(p, q)| p.to_bits() == q.to_bits()),
                "gate products differ at op {}",
                x.op
            );
        }
    }
    assert_eq!(
        a.1.as_ref().unwrap().layers.len(),
        b.1.as_ref().unwrap().layers.len()
    );
    for (x, y) in
        a.1.as_ref()
            .unwrap()
            .layers
            .iter()
            .zip(&b.1.as_ref().unwrap().layers)
    {
        assert_eq!(
            (x.naug, x.eligible, &x.idx, &x.unst_pos),
            (y.naug, y.eligible, &y.idx, &y.unst_pos)
        );
        for (u, v) in [(&x.a_l, &y.a_l), (&x.a_u, &y.a_u)] {
            assert_eq!(u.len(), v.len());
            assert!(u.iter().zip(v).all(|(p, q)| p.to_bits() == q.to_bits()));
        }
    }
}

fn conv(
    input: usize,
    ci: usize,
    width: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
    folded: bool,
) -> TwinOpSpec {
    let pad = width / 2;
    TwinOpSpec::Conv {
        input,
        weight: weights,
        bias,
        bias_err: vec![if folded { 1e-4 } else { 0.0 }; 4],
        weight_rel_err: if folded { 1e-5 } else { 0.0 },
        kernel: (4, ci, width, width),
        stride: (1, 1),
        pads: (pad, pad, pad, pad),
        ishape: (ci, 8, 8),
        oshape: (4, 8, 8),
    }
}

fn finish(mut ops: Vec<TwinOpSpec>) -> TwinSpec {
    let input = ops.len();
    ops.extend([
        TwinOpSpec::Flatten { input },
        TwinOpSpec::Gemm {
            input: input + 1,
            weight: vec![0.1; 256],
            bias: vec![0.0],
            bias_err: vec![],
            weight_rel_err: 0.0,
            shape: (1, 256),
        },
        TwinOpSpec::Relu { input: input + 2 },
        TwinOpSpec::Gemm {
            input: input + 3,
            weight: vec![1.0, -1.0],
            bias: vec![0.0; 2],
            bias_err: vec![],
            weight_rel_err: 0.0,
            shape: (2, 1),
        },
    ]);
    TwinSpec { n_in: 64, ops }
}

fn stem() -> Vec<TwinOpSpec> {
    vec![
        conv(
            0,
            1,
            1,
            vec![-1.0000001, -0.75, -0.5, -1.25],
            vec![-2.6, -1.8, -1.25, -3.2],
            true,
        ),
        TwinOpSpec::Relu { input: 1 },
    ]
}

fn delayed_shortcut() -> TwinSpec {
    let mut ops = stem();
    ops.extend([
        // The shortcut is produced first and stays live across both main Convs.
        conv(2, 4, 1, vec![0.125; 16], vec![-0.1; 4], false),
        conv(2, 4, 3, vec![0.0625; 144], vec![-0.1; 4], false),
        TwinOpSpec::Relu { input: 4 },
        conv(5, 4, 3, vec![0.0625; 144], vec![-0.1; 4], true),
        TwinOpSpec::Add { lhs: 3, rhs: 6 },
        TwinOpSpec::Relu { input: 7 },
    ]);
    finish(ops)
}

fn aliased_add() -> TwinSpec {
    let mut ops = stem();
    ops.extend([
        conv(2, 4, 3, vec![0.0625; 144], vec![-0.1; 4], false),
        TwinOpSpec::Relu { input: 3 },
        TwinOpSpec::Add { lhs: 4, rhs: 4 },
        conv(4, 4, 3, vec![0.0625; 144], vec![-0.1; 4], true),
        TwinOpSpec::Add { lhs: 5, rhs: 6 },
        TwinOpSpec::Relu { input: 7 },
    ]);
    finish(ops)
}

// Independent rational evaluation of the entire input/fold box. In these
// fixtures all first weights are negative and all later weights nonnegative;
// all inputs have the same interval sign. One common input corner and common
// fold endpoint attain each bound, including both residual branches. Thus the
// interval endpoints below are full attainable hulls, not sampled values.
// Geometry comes only from the specification, never the production gather.
fn exact_hulls(spec: &TwinSpec, lo: f64, hi: f64) -> Vec<Vec<(Q, Q)>> {
    let mut tensors = vec![vec![(q(lo), q(hi)); spec.n_in]];
    for (k, op) in spec.ops.iter().enumerate() {
        let out = match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                bias_err,
                weight_rel_err,
                kernel: (co, ci, kh, kw),
                stride: (sh, sw),
                pads: (pt, pl, _, _),
                ishape: (_, ih, iw),
                oshape: (_, oh, ow),
            } => {
                let mut out = Vec::new();
                for oc in 0..*co {
                    for oy in 0..*oh {
                        for ox in 0..*ow {
                            let be = q(bias_err[oc]);
                            let mut l = q(bias[oc]) - &be;
                            let mut u = q(bias[oc]) + &be;
                            for ic in 0..*ci {
                                for ky in 0..*kh {
                                    for kx in 0..*kw {
                                        let y = oy * sh + ky;
                                        let x = ox * sw + kx;
                                        if y < *pt || x < *pl || y - pt >= *ih || x - pl >= *iw {
                                            continue;
                                        }
                                        let w = q(weight[((oc * ci + ic) * kh + ky) * kw + kx]);
                                        if k == 0 {
                                            assert!(w < Q::zero());
                                        } else {
                                            assert!(w >= Q::zero());
                                        }
                                        let dw = w.abs() * q(*weight_rel_err);
                                        let wl = &w - &dw;
                                        let wu = &w + &dw;
                                        let (xl, xu) =
                                            &tensors[*input][(ic * ih + y - pt) * iw + x - pl];
                                        let products = [&wl * xl, &wl * xu, &wu * xl, &wu * xu];
                                        l += products.iter().min().unwrap();
                                        u += products.iter().max().unwrap();
                                    }
                                }
                            }
                            out.push((l, u));
                        }
                    }
                }
                out
            }
            TwinOpSpec::Relu { input } => tensors[*input]
                .iter()
                .map(|(l, u)| (l.clone().max(Q::zero()), u.clone().max(Q::zero())))
                .collect(),
            TwinOpSpec::Add { lhs, rhs } => tensors[*lhs]
                .iter()
                .zip(&tensors[*rhs])
                .map(|((al, au), (bl, bu))| (al + bl, au + bu))
                .collect(),
            TwinOpSpec::Flatten { input } => tensors[*input].clone(),
            TwinOpSpec::Gemm { .. } => break,
            _ => panic!("fixture has an unexpected operator"),
        };
        tensors.push(out);
    }
    tensors
}

fn check_hulls(spec: &TwinSpec, root: &RootGates) {
    let hulls = exact_hulls(spec, -3.0, -2.0);
    for layer in &root.layers {
        let TwinOpSpec::Relu { input } = spec.ops[layer.op] else {
            panic!("ReLU layer")
        };
        let hull = &hulls[input];
        assert_eq!(layer.n, hull.len());
        for (j, (l, u)) in hull.iter().enumerate() {
            assert!(
                q(layer.l[j]) <= *l,
                "lower full-box hull escaped at op {} row {j}",
                layer.op
            );
            assert!(
                q(layer.u[j]) >= *u,
                "upper full-box hull escaped at op {} row {j}",
                layer.op
            );
        }
    }
}

#[test]
fn packed_residual_roots_preserve_delayed_shortcut_alias_fanout_and_exact_hulls() {
    for (spec, expected_convs, expected_adds) in [
        (delayed_shortcut(), vec![0, 2, 3, 5], vec![6]),
        (aliased_add(), vec![0, 2, 5], vec![4, 6]),
    ] {
        let net = TwinNet::compile(&spec).unwrap();
        for mask in ["0", "1"] {
            ny_test_utils::env::with_serialized_env_vars(
                &[
                    ("NY_MARGIN_ROW_SUPPORT_MASK", mask),
                    (
                        ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
                        "0",
                    ),
                ],
                || {
                    let (packed, convs, adds) =
                        trace_detail(false, || build(&net, true, -3.0, -2.0, &[], None).unwrap());
                    let (dense, dconvs, dadds) =
                        trace_detail(true, || build(&net, true, -3.0, -2.0, &[], None).unwrap());
                    assert_eq!(
                        convs, expected_convs,
                        "each expected Conv must publish packed rows"
                    );
                    assert_eq!(
                        adds, expected_adds,
                        "each residual Add must publish packed rows"
                    );
                    assert_eq!(dconvs, [0]);
                    assert!(dadds.is_empty());
                    same(&packed, &dense);
                    assert!(packed.0.layers[0].alpha.iter().any(|&a| a == 0.375));
                    assert!(packed
                        .1
                        .as_ref()
                        .unwrap()
                        .layers
                        .iter()
                        .any(|r| !r.idx.is_empty()));
                    check_hulls(&spec, &packed.0);
                    assert!(packed.0.layers[0].unst.contains(&0));
                    assert!(packed.0.layers[0].unst.contains(&70));
                    let splits = [(0, 0, 1), (0, 70, -1)];
                    let (a, ac, aa) = trace_detail(false, || {
                        build(&net, true, -3.0, -2.0, &splits, None).unwrap()
                    });
                    let (b, _, _) = trace_detail(true, || {
                        build(&net, true, -3.0, -2.0, &splits, None).unwrap()
                    });
                    assert_eq!(ac, expected_convs);
                    assert_eq!(aa, expected_adds);
                    same(&a, &b);
                    let first = &a.0.layers[0];
                    for (j, slope) in [(0, 1.0), (70, 0.0)] {
                        assert_eq!(first.alpha[j], slope);
                        assert_eq!(first.s[j], slope);
                        assert_eq!(first.c[j], 0.0);
                        assert!(!first.unst.contains(&j));
                    }
                },
            );
        }
    }
}

#[test]
fn exceptional_deep_conv_materializes_into_corrected_dense_path() {
    let identity: Vec<f64> = (0..16)
        .map(|i| if i / 4 == i % 4 { 1.0 } else { 0.0 })
        .collect();
    let tiny: Vec<f64> = identity.iter().map(|w| w * 2.0_f64.powi(-150)).collect();
    let spec = finish(vec![
        conv(0, 1, 1, vec![2.0_f64.powi(126); 4], vec![0.0; 4], false),
        TwinOpSpec::Relu { input: 1 },
        conv(2, 4, 1, identity, vec![0.0; 4], false),
        TwinOpSpec::Relu { input: 3 },
        conv(4, 4, 1, tiny, vec![0.0; 4], false),
        TwinOpSpec::Relu { input: 5 },
    ]);
    let net = TwinNet::compile(&spec).unwrap();
    for mask in ["0", "1"] {
        for store in [false, true] {
            ny_test_utils::env::with_serialized_env_vars(
                &[
                    ("NY_MARGIN_ROW_SUPPORT_MASK", mask),
                    (
                        ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
                        "0",
                    ),
                ],
                || {
                    let (packed, convs, adds) =
                        trace_detail(false, || build(&net, store, 1.0, 2.0, &[], None).unwrap());
                    let (dense, _, _) =
                        trace_detail(true, || build(&net, store, 1.0, 2.0, &[], None).unwrap());
                    assert_eq!(
                        convs.as_slice(),
                        if store { &[0, 2][..] } else { &[][..] },
                        "tiny weight at op 4 must decline; f64 storage has no packed admission"
                    );
                    assert!(adds.is_empty());
                    same(&packed, &dense);
                    // Exercise the production root -> checked contraction bridge after
                    // an exceptional packed Conv refusal, including error/bias folds.
                    let (timed, timed_convs, timed_adds) = trace_detail(false, || {
                        build(
                            &net,
                            store,
                            1.0,
                            2.0,
                            &[],
                            Some(Instant::now() + std::time::Duration::from_secs(60)),
                        )
                        .unwrap()
                    });
                    same(&packed, &timed);
                    assert_eq!((&convs, &adds), (&timed_convs, &timed_adds));
                    let last = packed.0.layers.last().unwrap();
                    assert_eq!(last.op, 5);
                    for j in 0..last.n {
                        assert!(q(last.l[j]) <= q(2.0_f64.powi(-24)));
                        assert!(
                            q(last.u[j]) >= q(2.0_f64.powi(-23)),
                            "lost tiny original weight on fallback"
                        );
                    }
                },
            );
        }
    }
}

#[test]
fn packed_accelerator_preserves_residual_hulls_retention_and_splits() {
    use crate::margin_row::packed_root_engine::{test_support::Scalar, with_test_engine};
    use std::sync::{atomic::Ordering, Arc};
    for spec in [delayed_shortcut(), aliased_add()] {
        let net = TwinNet::compile(&spec).unwrap();
        for mask in ["0", "1"] {
            ny_test_utils::env::with_serialized_env_vars(
                &[
                    ("NY_MARGIN_ROW_SUPPORT_MASK", mask),
                    (
                        ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
                        "0",
                    ),
                ],
                || {
                    let engine = Arc::new(Scalar::default());
                    for splits in [&[][..], &[(0, 0, 1), (0, 70, -1)][..]] {
                        let cpu = trace_detail(false, || {
                            build(&net, true, -3.0, -2.0, splits, None).unwrap()
                        });
                        let before_calls = engine.calls.load(Ordering::SeqCst);
                        let accelerated = with_test_engine(engine.clone(), || {
                            trace_detail(false, || {
                                build(&net, true, -3.0, -2.0, splits, None).unwrap()
                            })
                        });
                        assert_eq!(
                            engine.calls.load(Ordering::SeqCst) - before_calls,
                            accelerated.1.len() - 1,
                            "every packed Conv after the specialized first Conv must dispatch"
                        );
                        same(&cpu.0, &accelerated.0);
                        assert_eq!((&cpu.1, &cpu.2), (&accelerated.1, &accelerated.2));
                        if splits.is_empty() {
                            check_hulls(&spec, &accelerated.0 .0);
                        }
                    }
                    assert!(engine.calls.load(Ordering::SeqCst) > 0);
                },
            );
        }
    }
}

#[cfg(feature = "gpu-tests-cuda")]
#[test]
fn packed_cuda_preserves_residual_roots_and_exact_hulls_on_required_hardware() {
    use crate::margin_row::packed_root_engine::with_test_engine;
    use std::sync::Arc;
    let engine = Arc::new(
        ny_cuda::packed_root::PackedRootEngine::new(0)
            .expect("required CUDA root qualification")
            .expect("required native CUDA root architecture"),
    );
    let mut completed = 0;
    for spec in [delayed_shortcut(), aliased_add()] {
        let net = TwinNet::compile(&spec).unwrap();
        for mask in ["0", "1"] {
            ny_test_utils::env::with_serialized_env_vars(
                &[
                    ("NY_MARGIN_ROW_SUPPORT_MASK", mask),
                    (
                        ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
                        "0",
                    ),
                ],
                || {
                    for splits in [&[][..], &[(0, 0, 1), (0, 70, -1)][..]] {
                        let cpu = trace_detail(false, || {
                            build(&net, true, -3.0, -2.0, splits, None).unwrap()
                        });
                        let before = ny_cuda::packed_root::stats();
                        let gpu = with_test_engine(engine.clone(), || {
                            trace_detail(false, || {
                                build(&net, true, -3.0, -2.0, splits, None).unwrap()
                            })
                        });
                        let after = ny_cuda::packed_root::stats();
                        assert_eq!(after.completed - before.completed, (gpu.1.len() - 1) as u64);
                        assert!(after.dispatches > before.dispatches);
                        completed += after.completed - before.completed;
                        same(&cpu.0, &gpu.0);
                        assert_eq!((&cpu.1, &cpu.2), (&gpu.1, &gpu.2));
                        if splits.is_empty() {
                            check_hulls(&spec, &gpu.0 .0);
                        }
                    }
                },
            );
        }
    }
    assert_eq!(completed, 20);
    eprintln!("packed-root actual CUDA: eight residual roots, twenty production contractions, CPU parity and unsplit exact hulls passed");
}
