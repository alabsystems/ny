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
fn q32(x: f32) -> Q {
    q(ny_core::f32_to_f64_exact(x))
}

fn conv(input: usize, ci: usize, weights: Vec<f64>, bias: Vec<f64>, folded: bool) -> TwinOpSpec {
    TwinOpSpec::Conv {
        input,
        weight: weights,
        bias,
        bias_err: vec![if folded { 1e-4 } else { 0.0 }; 2],
        weight_rel_err: if folded { 1e-5 } else { 0.0 },
        kernel: (2, ci, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (ci, 3, 3),
        oshape: (2, 3, 3),
    }
}

fn fixture() -> TwinSpec {
    TwinSpec {
        n_in: 9,
        ops: vec![
            conv(0, 1, vec![-1.0000001, -0.75], vec![-2.6, -1.8], true),
            TwinOpSpec::Relu { input: 1 },
            conv(2, 2, vec![1.25, 0.5, 0.25, 1.5], vec![-0.2, 0.1], false),
            conv(2, 2, vec![0.75, 0.125, 0.5, 0.25], vec![0.05, -0.2], false),
            TwinOpSpec::Add { lhs: 3, rhs: 4 },
            TwinOpSpec::Relu { input: 5 },
            TwinOpSpec::Flatten { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![0.1; 18],
                bias: vec![0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 18),
            },
            TwinOpSpec::Relu { input: 8 },
            TwinOpSpec::Gemm {
                input: 9,
                weight: vec![1.0, -1.0],
                bias: vec![0.0; 2],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    }
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
        &[-3.0; 9],
        &[-2.0; 9],
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
            assert!(
                u.iter().zip(v).all(|(p, q)| p.to_bits() == q.to_bits()),
                "gate products differ at op {}",
                x.op
            );
        }
    }
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

// Independent exact hull: every first weight is negative; both consumer
// matrices are positive. All outputs are monotone in one original pixel and
// all independent fold-ball choices. These corners attain the entire hull;
// no sampled vertices or production gather/concretize routines are used.
fn exact_hulls(spec: &TwinSpec) -> Vec<Vec<(Q, Q)>> {
    let TwinOpSpec::Conv {
        weight,
        bias,
        bias_err,
        weight_rel_err,
        ..
    } = &spec.ops[0]
    else {
        unreachable!()
    };
    let first: Vec<(Q, Q)> = (0..2)
        .map(|c| {
            let delta = q(weight[c]).abs() * q(*weight_rel_err);
            (
                (q(weight[c]) + &delta) * q(-2.0) + q(bias[c]) - q(bias_err[c]),
                (q(weight[c]) - &delta) * q(-3.0) + q(bias[c]) + q(bias_err[c]),
            )
        })
        .collect();
    let mut second = vec![(Q::zero(), Q::zero()); 2];
    for op in [&spec.ops[2], &spec.ops[3]] {
        let TwinOpSpec::Conv { weight, bias, .. } = op else {
            unreachable!()
        };
        for oc in 0..2 {
            second[oc].0 += q(bias[oc]);
            second[oc].1 += q(bias[oc]);
            for ic in 0..2 {
                assert!(weight[oc * 2 + ic] > 0.0);
                second[oc].0 += q(weight[oc * 2 + ic]) * first[ic].0.clone().max(Q::zero());
                second[oc].1 += q(weight[oc * 2 + ic]) * first[ic].1.clone().max(Q::zero());
            }
        }
    }
    vec![first, second]
}

#[test]
fn packed_stem_fanout_retention_and_exact_fold_ball_hulls() {
    let spec = fixture();
    let hulls = exact_hulls(&spec);
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
                let (compact, events) = trace(false, || build(&net, true, &[], None).unwrap());
                let (dense, control_events) = trace(true, || build(&net, true, &[], None).unwrap());
                assert_eq!(
                    events,
                    [0, 2, 3],
                    "both Conv consumers must use the carrier"
                );
                assert_eq!(
                    control_events,
                    [0],
                    "same tight identity certificate, dense representation"
                );
                same(&compact, &dense);
                assert!(compact.0.layers[0].alpha.iter().any(|&a| a == 0.375));
                for (layer, hull) in compact.0.layers.iter().zip(&hulls) {
                    for j in 0..layer.n {
                        assert!(
                            q(layer.l[j]) <= hull[j / 9].0,
                            "exact full-box lower hull escaped"
                        );
                        assert!(
                            q(layer.u[j]) >= hull[j / 9].1,
                            "exact full-box upper hull escaped"
                        );
                    }
                }
                let (a, ae) = trace(false, || {
                    build(&net, true, &[(0, 0, 1), (0, 10, -1)], None).unwrap()
                });
                let (b, _) = trace(true, || {
                    build(&net, true, &[(0, 0, 1), (0, 10, -1)], None).unwrap()
                });
                assert_eq!(ae, [0, 2, 3]);
                same(&a, &b);
            },
        );
    }
}

#[test]
fn packed_stem_refuses_unsupported_routes_and_expired_walk() {
    let spec = fixture();
    ny_test_utils::env::with_serialized_env_vars(
        &[(
            ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
            "0",
        )],
        || {
            let net = TwinNet::compile(&spec).unwrap();
            let (_, events) = trace(false, || build(&net, false, &[], None).unwrap());
            assert!(events.is_empty());
            let (expired, events) = trace(false, || {
                build(
                    &net,
                    true,
                    &[],
                    Some(Instant::now() - std::time::Duration::from_secs(1)),
                )
            });
            assert!(matches!(expired, Err(NyError::DeadlineExceeded(_))));
            assert!(events.is_empty());
            for rhs in [1, 2] {
                let mut shared = spec.clone();
                shared.ops[4] = TwinOpSpec::Add { lhs: 3, rhs };
                let net = TwinNet::compile(&shared).unwrap();
                let (_, events) = trace(false, || build(&net, true, &[], None).unwrap());
                assert!(
                    events.is_empty(),
                    "preactivation sharing and non-Conv postactivation sharing decline"
                );
            }
            for tiny in [2.0_f64.powi(-150), 3.0 * 2.0_f64.powi(-151)] {
                let mut exceptional = spec.clone();
                let TwinOpSpec::Conv { weight, .. } = &mut exceptional.ops[2] else {
                    unreachable!()
                };
                weight[0] = tiny;
                let net = TwinNet::compile(&exceptional).unwrap();
                let (_, events) = trace(false, || build(&net, true, &[], None).unwrap());
                assert!(
                    events.is_empty(),
                    "exceptional consumer conversion needs a separate certificate"
                );
            }
        },
    );
    ny_test_utils::env::with_serialized_env_vars(
        &[(
            ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
            "1",
        )],
        || {
            let net = TwinNet::compile(&spec).unwrap();
            let (root, events) = trace(false, || build(&net, true, &[], None).unwrap());
            assert!(events.is_empty());
            assert!(root.0.layers.iter().all(|g| g.clip_rows.is_some()));
        },
    );
}

#[test]
fn packed_signed_rows_preserve_original_frame_and_exact_gate_walls() {
    let n = 127;
    let mut rows = PackedRows {
        n_in: n,
        p: 1,
        coordinates: Array2::from_shape_vec((1, 5), vec![0, 17, 63, 126, usize::MAX]).unwrap(),
        m: Array2::from_shape_vec((1, 6), vec![-0.0, 0.0, f32::from_bits(1), -0.7, 0.0, 0.2])
            .unwrap(),
        d: Array2::from_shape_vec((1, 6), vec![0.0, -0.125, -f32::from_bits(1), 0.3, 0.0, 0.6])
            .unwrap(),
    };
    let (mi, di) = rows.materialize();
    let mut h = vec![1.0; n + 1];
    h[63] = 2.0_f64.powi(120);
    let (a, s, c) = (0.375, 0.625, 0.4);
    let dense = apply_gates_owned(
        mi.clone(),
        di.clone(),
        &[a],
        &[s],
        &[c],
        n,
        RoundMode::Outward,
        None,
        support_blk(n + 1),
        &h,
    );
    let gs = rows.apply_gates(&[a], &[s], &[c], &h, None).unwrap();
    let (mo, do_) = rows.materialize();
    for (x, y) in [(&mo, &dense.0), (&do_, &dense.1)] {
        assert!(x.iter().zip(y).all(|(u, v)| u.to_bits() == v.to_bits()));
    }
    assert_eq!(gs[0].to_bits(), dense.2[0].to_bits());
    let (mut lo_excess, mut up_excess) = (-q(gs[0]), -q(gs[0]));
    for i in 0..=n {
        let lower = (q32(mi[[0, i]]) - q32(di[[0, i]])) * q(a);
        let mut upper = (q32(mi[[0, i]]) + q32(di[[0, i]])) * q(s);
        if i == n {
            upper += q(c);
        }
        let dl = q32(mo[[0, i]]) - q32(do_[[0, i]]) - lower;
        let du = upper - q32(mo[[0, i]]) - q32(do_[[0, i]]);
        lo_excess += if i == n { dl } else { dl.abs() * q(h[i]) };
        up_excess += if i == n { du } else { du.abs() * q(h[i]) };
    }
    assert!(lo_excess <= Q::zero() && up_excess <= Q::zero());
    assert!(rows.apply_gates(&[f64::NAN], &[s], &[c], &h, None).is_err());
    h[1] = f64::INFINITY; // even an omitted coordinate belongs to the frame
    assert!(rows.apply_gates(&[a], &[s], &[c], &h, None).is_err());
}

#[test]
fn packed_padded_stem_encloses_full_monotone_hull() {
    let mut spec = fixture();
    spec.n_in = 80;
    spec.ops[0] = TwinOpSpec::Conv {
        input: 0,
        weight: (0..18).map(|i| -1.0000001 - (i as f64) * 0.001).collect(),
        bias: vec![-2.6; 2],
        bias_err: vec![1e-4; 2],
        weight_rel_err: 1e-5,
        kernel: (2, 1, 3, 3),
        stride: (1, 1),
        pads: (2, 2, 2, 2),
        ishape: (1, 4, 20),
        oshape: (2, 6, 22),
    };
    for op in &mut spec.ops[2..4] {
        let TwinOpSpec::Conv { ishape, oshape, .. } = op else {
            unreachable!()
        };
        *ishape = (2, 6, 22);
        *oshape = (2, 6, 22);
    }
    let TwinOpSpec::Gemm { weight, shape, .. } = &mut spec.ops[7] else {
        unreachable!()
    };
    *weight = vec![0.1; 264];
    *shape = (1, 264);
    let net = TwinNet::compile(&spec).unwrap();
    let TwinOpSpec::Conv {
        weight,
        bias,
        bias_err,
        weight_rel_err,
        ..
    } = &spec.ops[0]
    else {
        unreachable!()
    };
    // Spec coordinates only: intentionally do not consult compiled gathers.
    let mut first = vec![(Q::zero(), Q::zero()); 264];
    for oc in 0..2 {
        for oy in 0..6 {
            for ox in 0..22 {
                let j = oc * 132 + oy * 22 + ox;
                first[j] = (q(bias[oc]) - q(bias_err[oc]), q(bias[oc]) + q(bias_err[oc]));
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = (oy + ky) as isize - 2;
                        let ix = (ox + kx) as isize - 2;
                        if !(0..4).contains(&iy) || !(0..20).contains(&ix) {
                            continue;
                        }
                        let w = q(weight[oc * 9 + ky * 3 + kx]);
                        let delta = w.abs() * q(*weight_rel_err);
                        first[j].0 += (&w + &delta) * q(-2.0);
                        first[j].1 += (&w - &delta) * q(-3.0);
                    }
                }
            }
        }
    }
    let mut second = vec![(Q::zero(), Q::zero()); 264];
    for op in &spec.ops[2..4] {
        let TwinOpSpec::Conv { weight, bias, .. } = op else {
            unreachable!()
        };
        for oc in 0..2 {
            for sp in 0..132 {
                let j = oc * 132 + sp;
                second[j].0 += q(bias[oc]);
                second[j].1 += q(bias[oc]);
                for ic in 0..2 {
                    second[j].0 +=
                        q(weight[oc * 2 + ic]) * first[ic * 132 + sp].0.clone().max(Q::zero());
                    second[j].1 +=
                        q(weight[oc * 2 + ic]) * first[ic * 132 + sp].1.clone().max(Q::zero());
                }
            }
        }
    }
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
                let run = || {
                    let retain = RetainCfg {
                        per_layer: 7,
                        budget_bytes: 1 << 20,
                    };
                    RootGates::build_retaining_with_policy(
                        &net,
                        &[-3.0; 80],
                        &[-2.0; 80],
                        RoundMode::Outward,
                        None,
                        Some(&retain),
                        &[],
                        policy(true),
                        BackwardIntermRuntimePolicy::default(),
                        Some(&vec![vec![0.375; 264]; 2]),
                    )
                    .unwrap()
                };
                let (packed, events) = trace(false, run);
                let (dense, _) = trace(true, run);
                assert_eq!(events, [0, 2, 3]);
                same(&packed, &dense);
                assert!(packed.0.layers[0].alpha.iter().any(|&a| a == 0.375));
                for (layer, hull) in packed.0.layers.iter().zip([&first, &second]) {
                    for j in 0..layer.n {
                        assert!(q(layer.l[j]) <= hull[j].0);
                        assert!(q(layer.u[j]) >= hull[j].1);
                    }
                }
            },
        );
    }
}
