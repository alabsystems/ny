// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::margin_row::gpu_conv::GpuConvRoute;
use crate::margin_row::root::{with_identity_conv_trace_for_test, RetainCfg, RetainedRows};
use crate::margin_row::{
    BackwardIntermRuntimePolicy, ErrLane, RootBoxSource, RootPrecisionRuntimePolicy,
};

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn equal_products(
    a: &RootGates,
    b: &RootGates,
    ar: &Option<RetainedRows>,
    br: &Option<RetainedRows>,
) {
    assert_eq!(a.mode, b.mode);
    for (x, y) in [
        (&a.lo, &b.lo),
        (&a.hi, &b.hi),
        (&a.mid, &b.mid),
        (&a.rad, &b.rad),
        (&a.xabs, &b.xabs),
    ] {
        assert_eq!(bits(x), bits(y));
    }
    assert_eq!(a.root_precision_policy, b.root_precision_policy);
    assert_eq!(a.root_precision_execution, b.root_precision_execution);
    assert_eq!(a.backward_interm_execution, b.backward_interm_execution);
    assert!(a.spec_alpha_policy.is_none() && b.spec_alpha_policy.is_none());
    assert!(a.head_alpha.is_none() && b.head_alpha.is_none());
    assert_eq!(a.layers.len(), b.layers.len());
    assert!(a.layers.iter().any(|g| !g.unst.is_empty()));
    for (x, y) in a.layers.iter().zip(&b.layers) {
        assert_eq!((x.op, x.n, &x.unst), (y.op, y.n, &y.unst));
        for (u, v) in [
            (&x.l, &y.l),
            (&x.u, &y.u),
            (&x.alpha, &y.alpha),
            (&x.s, &y.s),
            (&x.c, &y.c),
            (&x.ms, &y.ms),
        ] {
            assert_eq!(bits(u), bits(v));
        }
        let xc = x.clip_rows.as_ref().expect("requested clip rows");
        let yc = y.clip_rows.as_ref().expect("requested clip rows");
        assert_eq!(xc.m.dim(), yc.m.dim());
        assert_eq!(xc.d.dim(), yc.d.dim());
        for (u, v) in [
            (xc.m.as_slice().unwrap(), yc.m.as_slice().unwrap()),
            (xc.d.as_slice().unwrap(), yc.d.as_slice().unwrap()),
            (&xc.bm, &yc.bm),
            (&xc.bd, &yc.bd),
            (&xc.sl_lo, &yc.sl_lo),
            (&xc.sl_up, &yc.sl_up),
        ] {
            assert_eq!(bits(u), bits(v));
        }
    }
    let ar = ar.as_ref().expect("retention requested");
    let br = br.as_ref().expect("retention requested");
    assert_eq!(ar.layers.len(), br.layers.len());
    assert!(ar.layers.iter().any(|r| !r.idx.is_empty()));
    for (x, y) in ar.layers.iter().zip(&br.layers) {
        assert_eq!(
            (x.eligible, x.naug, &x.idx, &x.unst_pos),
            (y.eligible, y.naug, &y.idx, &y.unst_pos)
        );
        for (u, v) in [(&x.a_l, &y.a_l), (&x.a_u, &y.a_u)] {
            assert!(u
                .iter()
                .map(|f| f.to_bits())
                .eq(v.iter().map(|f| f.to_bits())));
        }
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

fn retained_indices_match(root: &RootGates, retained: &Option<RetainedRows>) {
    let retained = retained.as_ref().expect("retention requested");
    assert_eq!(retained.layers.len(), root.layers.len());
    assert!(retained.layers.iter().any(|r| !r.idx.is_empty()));
    for (g, r) in root.layers.iter().zip(&retained.layers) {
        assert!(g
            .l
            .iter()
            .zip(&g.u)
            .all(|(l, u)| l.is_finite() && u.is_finite() && l <= u));
        assert_eq!(r.naug, root.lo.len() + 1);
        assert_eq!(r.eligible, g.unst.len());
        assert!(r.eligible >= r.idx.len());
        assert_eq!(r.idx.len(), r.unst_pos.len());
        assert_eq!(r.a_l.len(), r.idx.len() * r.naug);
        assert_eq!(r.a_u.len(), r.a_l.len());
        assert!(r.idx.windows(2).all(|w| w[0] < w[1]));
        for (&idx, &pos) in r.idx.iter().zip(&r.unst_pos) {
            assert_eq!(g.unst[pos], idx);
        }
        let clip = g
            .clip_rows
            .as_ref()
            .expect("requested clip rows must survive");
        assert_eq!(clip.m.dim(), (g.unst.len(), root.lo.len()));
        assert_eq!(clip.d.dim(), clip.m.dim());
        assert!(clip.m.iter().chain(&clip.d).all(|v| v.is_finite()));
        for v in [&clip.bm, &clip.bd, &clip.sl_lo, &clip.sl_up] {
            assert_eq!(v.len(), g.unst.len());
            assert!(v.iter().all(|x| x.is_finite()));
        }
        assert!(clip.sl_lo.iter().chain(&clip.sl_up).all(|x| *x >= 0.0));
    }
    // Retained f32 ranker rows may reorder when gates tighten; validate their
    // indices against their own root rather than treating them as enclosures.
}

#[test]
fn owned_gates_preserve_shared_source_retention_and_backward_bits() {
    use crate::margin_row::root::with_borrowed_gates_for_test;

    let mut rng = StdRng::seed_from_u64(51);
    let original = tiny_spec(&mut rng, 0.6);
    let mut shared = original.clone();
    // The first preactivation now feeds both ReLU and the later residual Add.
    // Reusing ReLU's arrays must leave the other consumer's input untouched.
    shared.ops[3] = TwinOpSpec::Add { lhs: 3, rhs: 1 };
    for spec in [original, shared] {
        let net = TwinNet::compile(&spec).unwrap();
        let lo = vec![-0.3; spec.n_in];
        let hi = vec![0.5; spec.n_in];
        let retain = RetainCfg {
            per_layer: 7,
            budget_bytes: 1 << 20,
        };
        let seed: Vec<Vec<f64>> = net
            .trunk_relus
            .iter()
            .map(|&k| vec![0.375; net.tsize[k + 1]])
            .collect();
        for mask in ["0", "1"] {
            ny_test_utils::env::with_serialized_env_vars(
                &[
                    ("NY_MARGIN_ROW_SUPPORT_MASK", mask),
                    (
                        ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
                        "1",
                    ),
                ],
                || {
                    for store in [false, true] {
                        for seeded in [false, true] {
                            let build = || {
                                RootGates::build_retaining_with_policy(
                                    &net,
                                    &lo,
                                    &hi,
                                    RoundMode::Outward,
                                    None,
                                    Some(&retain),
                                    &[],
                                    policy(store),
                                    BackwardIntermRuntimePolicy::default(),
                                    seeded.then_some(seed.as_slice()),
                                )
                                .unwrap()
                            };
                            let (a, ar) = build();
                            let (b, br) = with_borrowed_gates_for_test(build);
                            equal_products(&a, &b, &ar, &br);
                            let ea = BackwardEngine::new(&net, &a)
                                .with_gpu_conv_route(GpuConvRoute::Off);
                            let eb = BackwardEngine::new(&net, &b)
                                .with_gpu_conv_route(GpuConvRoute::Off);
                            let (al, au) = ea.y_rows(None).unwrap();
                            let (bl, bu) = eb.y_rows(None).unwrap();
                            assert_pass_bits("owned gate lower", &al, &bl);
                            assert_pass_bits("owned gate upper", &au, &bu);
                        }
                    }
                },
            );
        }
    }
}

#[test]
fn identity_conv_preserves_coefficients_and_retention_consumers_with_tighter_error() {
    let mut rng = StdRng::seed_from_u64(51);
    for spec in [tiny_spec(&mut rng, 0.6), cgan_like_spec(&mut rng, 0.6)] {
        let net = TwinNet::compile(&spec).unwrap();
        let lo = vec![-0.3; spec.n_in];
        let hi = vec![0.5; spec.n_in];
        let retain = RetainCfg {
            per_layer: 7,
            budget_bytes: 1 << 20,
        };
        for mask in ["0", "1"] {
            ny_test_utils::env::with_serialized_env_vars(
                &[
                    ("NY_MARGIN_ROW_SUPPORT_MASK", mask),
                    (
                        ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
                        "1",
                    ),
                ],
                || {
                    for store in [false, true] {
                        let build = || {
                            RootGates::build_retaining_with_policy(
                                &net,
                                &lo,
                                &hi,
                                RoundMode::Outward,
                                None,
                                Some(&retain),
                                &[],
                                policy(store),
                                BackwardIntermRuntimePolicy::default(),
                                None,
                            )
                            .unwrap()
                        };
                        let ((fast, fr), mut ft) = with_identity_conv_trace_for_test(false, build);
                        let ((dense, dr), dt) = with_identity_conv_trace_for_test(true, build);
                        assert_eq!((ft.len(), dt.len()), (1, 1));
                        assert!(ft[0].implicit, "optimization must engage");
                        assert!(!dt[0].implicit);
                        assert!(ft[0]
                            .g
                            .iter()
                            .zip(&dt[0].g)
                            .all(|(&f, &d)| f64::from_bits(f) <= f64::from_bits(d)));
                        assert!(ft[0]
                            .g
                            .iter()
                            .zip(&dt[0].g)
                            .any(|(&f, &d)| f64::from_bits(f) < f64::from_bits(d)));
                        ft[0].implicit = false;
                        ft[0].g.clone_from(&dt[0].g);
                        // Only the error vector changes: M, D, shape, and
                        // support bits remain exactly the dense result.
                        assert_eq!(ft, dt);
                        assert_eq!(ft[0].support.is_empty(), mask == "0");
                        retained_indices_match(&fast, &fr);
                        retained_indices_match(&dense, &dr);
                        assert!(fast.layers[0]
                            .l
                            .iter()
                            .zip(&dense.layers[0].l)
                            .all(|(f, d)| f >= d));
                        assert!(fast.layers[0]
                            .u
                            .iter()
                            .zip(&dense.layers[0].u)
                            .all(|(f, d)| f <= d));
                        let ef =
                            BackwardEngine::new(&net, &fast).with_gpu_conv_route(GpuConvRoute::Off);
                        let ed = BackwardEngine::new(&net, &dense)
                            .with_gpu_conv_route(GpuConvRoute::Off);
                        // Tighter error can change downstream gates: require
                        // containment, not whole-root bit identity or monotonicity.
                        ef.y_rows(None).unwrap();
                        ed.y_rows(None).unwrap();
                    }
                },
            );
        }
    }
}

#[test]
fn identity_conv_preserves_seeded_split_roots_and_deadline_refusal() {
    let mut rng = StdRng::seed_from_u64(51);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).unwrap();
    let lo = vec![-0.3; spec.n_in];
    let hi = vec![0.5; spec.n_in];
    let retain = RetainCfg {
        per_layer: 7,
        budget_bytes: 1 << 20,
    };
    ny_test_utils::env::with_serialized_env_vars(
        &[(
            ny_levers::decls::sound_channel::MARGIN_ROW_CLIP_ROWS.name,
            "1",
        )],
        || {
            let seed: Vec<Vec<f64>> = net
                .trunk_relus
                .iter()
                .map(|&k| vec![0.375; net.tsize[k + 1]])
                .collect();
            for store in [false, true] {
                let (initial, _) = RootGates::build_retaining_with_policy(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    None,
                    None,
                    &[],
                    policy(store),
                    BackwardIntermRuntimePolicy::default(),
                    None,
                )
                .unwrap();
                assert!(initial.layers[0].unst.len() >= 2);
                let split = initial.layers[0].unst[0];
                for dir in [-1, 1] {
                    let build = || {
                        RootGates::build_retaining_with_policy(
                            &net,
                            &lo,
                            &hi,
                            RoundMode::Outward,
                            None,
                            Some(&retain),
                            &[(0, split, dir)],
                            policy(store),
                            BackwardIntermRuntimePolicy::default(),
                            Some(&seed),
                        )
                        .unwrap()
                    };
                    let ((a, ar), mut at) = with_identity_conv_trace_for_test(false, build);
                    let ((b, br), bt) = with_identity_conv_trace_for_test(true, build);
                    assert!(at[0].implicit);
                    at[0].implicit = false;
                    at[0].g.clone_from(&bt[0].g);
                    assert_eq!(at, bt);
                    retained_indices_match(&a, &ar);
                    retained_indices_match(&b, &br);
                    for root in [&a, &b] {
                        assert!(!root.layers[0].unst.contains(&split));
                        assert_eq!(root.layers[0].alpha[split], if dir > 0 { 1.0 } else { 0.0 });
                        assert!(
                            root.layers[0]
                                .unst
                                .iter()
                                .any(|&i| root.layers[0].alpha[i] == 0.375),
                            "fractional seed must affect a surviving unstable neuron"
                        );
                    }
                }
            }
            let (result, trace) = with_identity_conv_trace_for_test(false, || {
                RootGates::build_retaining_with_policy(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    Some(std::time::Instant::now() - std::time::Duration::from_secs(1)),
                    None,
                    &[],
                    policy(true),
                    BackwardIntermRuntimePolicy::default(),
                    None,
                )
            });
            assert!(matches!(result, Err(ny_core::NyError::DeadlineExceeded(_))));
            assert!(trace.is_empty(), "expired root must publish no Conv");
        },
    );
}

#[test]
fn identity_conv_tighter_error_encloses_exact_fold_ball_on_both_storage_lanes() {
    use num_rational::BigRational as Q;
    let q = |v: f64| Q::from_float(v).expect("finite bound");
    let _env = ny_test_utils::env::lock_env_shared();
    let configs = [
        (0.0, 0.0, None),
        (1e-5, 1e-4, None),
        (1e-5, 1e-4, Some((1e-5, 1e-4))),
    ];
    let mut checked = 0;
    let mut load_bearing = [0usize; 2];
    for (ci, (rel, berr, affine)) in configs.into_iter().enumerate() {
        let spec = fold_ball_spec(rel, berr, affine, 0xF01D_BA11 ^ ci as u64);
        let net = TwinNet::compile(&spec).unwrap();
        let lo = vec![-0.7; spec.n_in];
        let hi = vec![1.3; spec.n_in];
        let ball = fold_ball_hull(&spec, &lo, &hi, true);
        let stored = fold_ball_hull(&spec, &lo, &hi, false);
        for store in [false, true] {
            let ((root, _), trace) = with_identity_conv_trace_for_test(false, || {
                RootGates::build_retaining_with_policy(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    None,
                    None,
                    &[],
                    policy(store),
                    BackwardIntermRuntimePolicy::default(),
                    None,
                )
                .unwrap()
            });
            assert_eq!(trace.len(), 1);
            assert!(trace[0].implicit);
            assert!(!root.root_precision_policy.ibp_seed);
            assert_eq!(root.layers[0].l.len(), ball.len());
            for (j, ((mn, mx), (smn, smx))) in ball.iter().zip(&stored).enumerate() {
                let l = q(root.layers[0].l[j]);
                let u = q(root.layers[0].u[j]);
                assert!(
                    l <= *mn && u >= *mx,
                    "config {ci}, store {store}, row {j}: exact fold-ball hull escapes"
                );
                if (mx - smx) > (&u - mx) || (smn - mn) > (mn - &l) {
                    load_bearing[usize::from(store)] += 1;
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 192);
    assert!(
        load_bearing.iter().all(|&count| count > 0),
        "fold uncertainty must exceed numerical headroom in both storage lanes"
    );
}

#[test]
fn identity_conv_tighter_error_certifies_an_active_gate_missed_by_generic_charge() {
    use num_rational::BigRational as Q;
    let _env = ny_test_utils::env::lock_env_shared();
    let conv = TwinOpSpec::Conv {
        input: 0,
        weight: vec![1.0],
        bias: vec![0.0],
        bias_err: vec![0.0],
        weight_rel_err: 0.0,
        kernel: (1, 1, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (1, 1, 1),
        oshape: (1, 1, 1),
    };
    let spec = single_conv_spec(&mut StdRng::seed_from_u64(716), conv);
    let net = TwinNet::compile(&spec).unwrap();
    let lo = [2.0_f64.powi(-25)];
    let hi = [1.0];
    for store in [false, true] {
        let build = || {
            RootGates::build_retaining_with_policy(
                &net,
                &lo,
                &hi,
                RoundMode::Outward,
                None,
                None,
                &[],
                policy(store),
                BackwardIntermRuntimePolicy::default(),
                None,
            )
            .unwrap()
        };
        let ((fast, _), ft) = with_identity_conv_trace_for_test(false, build);
        let ((dense, _), dt) = with_identity_conv_trace_for_test(true, build);
        assert_eq!((ft.len(), dt.len()), (1, 1));
        assert!(ft[0].implicit && !dt[0].implicit);
        assert!(
            dense.layers[0].unst.contains(&0),
            "generic charge must miss the active gate"
        );
        assert!(!fast.layers[0].unst.contains(&0));
        assert!(fast.layers[0].l[0] > 0.0);
        assert_eq!(fast.layers[0].alpha[0], 1.0);
        assert_eq!(fast.layers[0].s[0], 1.0);
        // The original first preactivation is exactly x over the whole box.
        assert!(Q::from_float(fast.layers[0].l[0]).unwrap() <= Q::from_float(lo[0]).unwrap());
        assert!(Q::from_float(fast.layers[0].u[0]).unwrap() >= Q::from_float(hi[0]).unwrap());
    }
}

#[test]
fn identity_conv_falls_back_for_parity_f64_and_legacy_charges() {
    let _env = ny_test_utils::env::lock_env_shared();
    let mut rng = StdRng::seed_from_u64(51);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).unwrap();
    let lo = vec![-0.3; spec.n_in];
    let hi = vec![0.5; spec.n_in];
    for (mode, p) in [
        (RoundMode::Parity, policy(false)),
        (
            RoundMode::Outward,
            RootPrecisionRuntimePolicy {
                root_f32: false,
                ..policy(false)
            },
        ),
    ] {
        let (_, trace) = with_identity_conv_trace_for_test(false, || {
            RootGates::build_retaining_with_policy(
                &net,
                &lo,
                &hi,
                mode,
                None,
                None,
                &[],
                p,
                BackwardIntermRuntimePolicy::default(),
                None,
            )
            .unwrap()
        });
        assert_eq!(trace.len(), 1);
        assert!(!trace[0].implicit);
    }
    let build = || {
        with_identity_conv_trace_for_test(false, || {
            RootGates::build_retaining_with_policy(
                &net,
                &lo,
                &hi,
                RoundMode::Outward,
                None,
                None,
                &[],
                policy(true),
                BackwardIntermRuntimePolicy::default(),
                None,
            )
            .unwrap()
        })
    };
    let (_, trace) = crate::margin_row::root::with_legacy_rounding_charge_for_test(build);
    assert_eq!(trace.len(), 1);
    assert!(!trace[0].implicit);
    let (_, trace) = crate::margin_row::root::with_legacy_fold_charge_for_test(build);
    assert_eq!(trace.len(), 1);
    assert!(!trace[0].implicit);
}

#[test]
fn identity_conv_preserves_reused_input_until_its_last_consumer() {
    let _env = ny_test_utils::env::lock_env_shared();
    let mut rng = StdRng::seed_from_u64(51);
    let conv = conv_case(
        &mut rng,
        (2, 2, 1, 1),
        (1, 1),
        (0, 0, 0, 0),
        (2, 3, 3),
        (2, 3, 3),
    );
    let mut spec = single_conv_spec(&mut rng, conv);
    spec.ops.insert(1, TwinOpSpec::Add { lhs: 1, rhs: 0 });
    for op in spec.ops.iter_mut().skip(2) {
        match op {
            TwinOpSpec::Relu { input }
            | TwinOpSpec::Flatten { input }
            | TwinOpSpec::Gemm { input, .. } => *input += 1,
            _ => panic!("unexpected single-Conv fixture op"),
        }
    }
    let net = TwinNet::compile(&spec).unwrap();
    let lo = vec![-0.3; spec.n_in];
    let hi = vec![0.5; spec.n_in];
    for store in [false, true] {
        let build = || {
            RootGates::build_retaining_with_policy(
                &net,
                &lo,
                &hi,
                RoundMode::Outward,
                None,
                None,
                &[],
                policy(store),
                BackwardIntermRuntimePolicy::default(),
                None,
            )
            .unwrap()
        };
        let ((a, _), at) = with_identity_conv_trace_for_test(false, build);
        let ((b, _), bt) = with_identity_conv_trace_for_test(true, build);
        assert_eq!(at.len(), 1);
        assert!(!at[0].implicit);
        assert_eq!(at, bt);
        for (x, y) in a.layers.iter().zip(&b.layers) {
            assert_eq!(bits(&x.l), bits(&y.l));
            assert_eq!(bits(&x.u), bits(&y.u));
        }
    }
}
