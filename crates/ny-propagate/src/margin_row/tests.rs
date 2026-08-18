// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Synthetic-net tests for the margin-row lane (#twinwall). The cifar100
//! differential gates (Python-parity root bound, recorded trajectory replay,
//! end-to-end closures) live in ny-cli where the ONNX extractor is.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::bab::{
    absorb_class_stats, classwise_conjunction_complete, classwise_root_cache_isolated_for_test,
    classwise_schedule, BabConfig, BabStats, ClassBabStats, MarginRowBab, MarginRowOutcome,
};
use super::bounds::{
    compose_viay, head_gates, head_variant, margin_seed, per_class_direct, row_dots, variant_state,
    MarginBatch, YBox,
};
use super::engine::{
    domain_gates, BackwardEngine, Exc, Exceptions, LaneDir, RowDomainGateBlock,
    Seed as DomainStackSeed,
};
use super::net::TwinNet;
use super::root::RootGates;
use super::rounding::RoundMode;
use super::spec::{TwinOpSpec, TwinSpec};

/// Tiny twin-wall net: 2x6x6 input, residual block, 2 trunk relus,
/// 5-wide head, 4 classes.
fn tiny_spec(rng: &mut StdRng, scale: f64) -> TwinSpec {
    let mut w =
        |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random_range(-scale..scale)).collect() };
    let conv1 = TwinOpSpec::Conv {
        input: 0,
        weight: w(4 * 2 * 3 * 3),
        bias: w(4),
        bias_err: vec![0.0; 4],
        weight_rel_err: 1e-15,
        kernel: (4, 2, 3, 3),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (2, 6, 6),
        oshape: (4, 3, 3),
    };
    let conv2 = TwinOpSpec::Conv {
        input: 2,
        weight: w(4 * 4 * 3 * 3),
        bias: w(4),
        bias_err: vec![1e-18; 4],
        weight_rel_err: 1e-15,
        kernel: (4, 4, 3, 3),
        stride: (1, 1),
        pads: (1, 1, 1, 1),
        ishape: (4, 3, 3),
        oshape: (4, 3, 3),
    };
    TwinSpec {
        n_in: 72,
        ops: vec![
            conv1,                              // t1
            TwinOpSpec::Relu { input: 1 },      // t2 (trunk relu 0)
            conv2,                              // t3
            TwinOpSpec::Add { lhs: 3, rhs: 2 }, // t4 (residual, unrectified)
            TwinOpSpec::Relu { input: 4 },      // t5 (trunk relu 1)
            TwinOpSpec::Flatten { input: 5 },   // t6
            TwinOpSpec::Gemm {
                input: 6,
                weight: w(5 * 36),
                bias: w(5),
                shape: (5, 36),
            }, // t7 (y)
            TwinOpSpec::Relu { input: 7 },      // t8 (head relu)
            TwinOpSpec::Gemm {
                input: 8,
                weight: w(4 * 5),
                bias: w(4),
                shape: (4, 5),
            }, // t9
        ],
    }
}

#[test]
fn compile_rejects_invalid_conv_error_budgets() {
    let mut rng = StdRng::seed_from_u64(5);
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        let mut spec = tiny_spec(&mut rng, 0.2);
        let TwinOpSpec::Conv { bias_err, .. } = &mut spec.ops[0] else {
            unreachable!("tiny spec starts with a conv")
        };
        bias_err[0] = invalid;
        assert!(
            TwinNet::compile(&spec).is_err(),
            "invalid certified error budget {invalid:?} was accepted"
        );
    }
}

#[test]
fn production_route_authority_granted() {
    // Quarantine lifted 2026-07-18 after both proof obligations were discharged
    // (rounding fixes 5628ed6f/55455cbb + adapter b7342fdf, ~850k zero-violation
    // enclosure checks). The lane now runs the certified algorithm in production.
    assert!(super::margin_row_bab_enabled());
}

#[test]
fn external_head_has_no_lane_impl_authority_parameter() {
    // Compile-time architecture gate: the verdict-bearing implementation has
    // exactly the pre-experiment signature. An external head box can exist
    // only at the compatibility wrapper, where it is dropped before this
    // function is called.
    let lane_without_external_head: fn(
        &TwinSpec,
        &[f64],
        &[f64],
        usize,
        &[usize],
        Option<std::time::Instant>,
        usize,
    ) -> MarginRowOutcome = super::lane_impl;
    let _ = lane_without_external_head;
}

fn sample_box(rng: &mut StdRng, root: &RootGates, n: usize) -> Array2<f64> {
    let n_in = root.mid.len();
    let mut x = Array2::<f64>::zeros((n_in, n));
    for i in 0..n_in {
        for b in 0..n {
            let u: f64 = rng.random_range(-1.0..1.0);
            x[[i, b]] = root.mid[i] + u * root.rad[i];
        }
    }
    x
}

/// Margin values `Y_t - Y_j` at sampled points from the head pre-activation.
fn margins_at(net: &TwinNet, y: &Array2<f64>, t: usize, adv: &[usize]) -> Vec<Vec<f64>> {
    let (w2, b2, (_, n_y)) = net.gemm2();
    let bcols = y.ncols();
    adv.iter()
        .map(|&j| {
            (0..bcols)
                .map(|b| {
                    let mut m = b2[t] - b2[j];
                    for k in 0..n_y {
                        m += (w2[t * n_y + k] - w2[j * n_y + k]) * y[[k, b]].max(0.0);
                    }
                    m
                })
                .collect()
        })
        .collect()
}

#[test]
fn forward_tableau_boxes_enclose_sampled_preacts() {
    let mut rng = StdRng::seed_from_u64(7);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.3; 72];
    let hi = vec![0.5; 72];
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let root = RootGates::build(&net, &lo, &hi, mode, None).expect("root");
        let x = sample_box(&mut rng, &root, 64);
        let sel: std::collections::BTreeMap<usize, Vec<usize>> = root
            .layers
            .iter()
            .map(|lg| (lg.op, (0..lg.n).collect()))
            .collect();
        let (_, pre) = net.forward_points(&x, &sel).expect("forward");
        for lg in &root.layers {
            let p = &pre[&lg.op];
            for j in 0..lg.n {
                for b in 0..x.ncols() {
                    let v = p[[j, b]];
                    assert!(
                        v >= lg.l[j] - 1e-9 && v <= lg.u[j] + 1e-9,
                        "{mode:?} layer op {} neuron {j}: {v} outside [{}, {}]",
                        lg.op,
                        lg.l[j],
                        lg.u[j]
                    );
                }
            }
        }
    }
}

#[test]
fn y_rows_bracket_sampled_y() {
    let mut rng = StdRng::seed_from_u64(11);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let root = RootGates::build(&net, &lo, &hi, mode, None).expect("root");
        let eng = BackwardEngine::new(&net, &root);
        let (al, au) = eng.y_rows(None).expect("y_rows");
        let ybox = YBox::from_rows(&eng, &al, &au);
        let x = sample_box(&mut rng, &root, 64);
        let (y, _) = net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        for j in 0..net.n_y {
            for b in 0..x.ncols() {
                let v = y[[j, b]];
                assert!(
                    v >= ybox.ly[j] - 1e-9 && v <= ybox.uy[j] + 1e-9,
                    "{mode:?} y[{j}] = {v} outside [{}, {}]",
                    ybox.ly[j],
                    ybox.uy[j]
                );
            }
        }
    }
}

#[test]
fn root_margin_bounds_enclose_sampled_margins_and_modes_agree() {
    let mut rng = StdRng::seed_from_u64(13);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    let mut bounds_by_mode = Vec::new();
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let root = RootGates::build(&net, &lo, &hi, mode, None).expect("root");
        let eng = BackwardEngine::new(&net, &root);
        let (al, au) = eng.y_rows(None).expect("y_rows");
        let ybox = YBox::from_rows(&eng, &al, &au);
        let mb = MarginBatch::new(&net, t, &adv).expect("mb");
        let gates = head_gates(&ybox, mode);
        let ms = margin_seed(&mb, &gates, &ybox, mode);
        let pass = eng
            .run(&ms.seed, None, LaneDir::Lower, None, false)
            .expect("pass");
        let direct = per_class_direct(&eng, &pass, &ms, 0..adv.len());
        let ald = row_dots(&root, &al);
        let aud = row_dots(&root, &au);
        let m2v = compose_viay(&eng, &mb, &gates, &al, &au, &ald, &aud, mode);
        let per: Vec<f64> = (0..adv.len())
            .map(|k| direct[k].max(ms.m1[k]).max(m2v[k]))
            .collect();
        // Enclosure vs sampled margins.
        let x = sample_box(&mut rng, &root, 200);
        let (y, _) = net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        let margins = margins_at(&net, &y, t, &adv);
        for (k, mrow) in margins.iter().enumerate() {
            let min_m = mrow.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                per[k] <= min_m + 1e-9,
                "mode {mode:?} class {k}: bound {} > sampled min {min_m}",
                per[k]
            );
        }
        bounds_by_mode.push(per);
    }
    // Parity vs outward. Two separate properties, checked separately:
    //   (a) DIRECTION (soundness-relevant, always strict): the certified
    //       outward bound must never sit ABOVE the nearest-mode bound — it is
    //       the conservative one by construction.
    //   (b) TIGHTNESS (quality, tolerance depends on the root kernel): the two
    //       must stay close. The 1e-6 budget assumes the f64 root conv; the
    //       opt-in f32 root tableau (`NY_MARGIN_ROW_ROOT_F32`, added for the
    //       tinyimagenet tier) carries ~1e-7 RELATIVE error, which on margins
    //       of magnitude ~30 is a ~1e-5 absolute drift — so scale the budget
    //       by the bound magnitude when that kernel is active. Measured with
    //       BLAS+f32 both on: drift 2.6e-4 on a -27.4 margin, i.e. ~9e-6
    //       relative, and in the SAFE direction.
    let f32_root = std::env::var("NY_MARGIN_ROW_ROOT_F32").is_ok();
    for (p, o) in bounds_by_mode[0].iter().zip(&bounds_by_mode[1]) {
        assert!(
            o <= &(p + 1e-9),
            "outward {o} is ABOVE parity {p} — the certified bound must be the \
             conservative one"
        );
        let budget = if f32_root {
            1e-6 + 1e-4 * (1.0 + p.abs())
        } else {
            1e-6
        };
        assert!(
            (p - o).abs() < budget,
            "parity {p} vs outward {o} drifted more than {budget:.3e}"
        );
    }
}

/// T5 analog: ONE batched exception pass == per-candidate rebuilt passes.
#[test]
fn exception_batched_pass_matches_separate_passes() {
    let mut rng = StdRng::seed_from_u64(17);
    let spec = tiny_spec(&mut rng, 0.7);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.5; 72];
    let hi = vec![0.5; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let mb = MarginBatch::new(&net, 0, &[1, 2]).expect("mb");
    let gates = head_gates(&ybox, RoundMode::Parity);
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Parity);
    let nf = mb.nf();
    // Candidates: first two unstable neurons of each trunk layer.
    let mut cands: Vec<(usize, usize)> = Vec::new();
    for (li, lg) in root.layers.iter().enumerate() {
        for pos in 0..lg.unst.len().min(2) {
            cands.push((li, pos));
        }
    }
    assert!(!cands.is_empty(), "need unstable neurons for the test");
    // Batched exception pass.
    let total = 2 * cands.len() * nf;
    let mut seed = Array2::<f64>::zeros((net.n_y, total));
    let mut exc = Exceptions::default();
    for (kc, &(li, pos)) in cands.iter().enumerate() {
        let idx = root.layers[li].unst[pos];
        for (d_i, fix) in [(0usize, (1.0, 1.0, 0.0)), (1usize, (0.0, 0.0, 0.0))] {
            let r0 = (2 * kc + d_i) * nf;
            for j in 0..net.n_y {
                for f in 0..nf {
                    seed[[j, r0 + f]] = ms.seed.s[[j, f]];
                }
            }
            for f in 0..nf {
                exc.by_layer.entry(li).or_default().push(Exc {
                    row: r0 + f,
                    neuron: idx,
                    a2: fix.0,
                    s2: fix.1,
                    c2: fix.2,
                });
            }
        }
    }
    let batched = eng
        .run(
            &super::engine::Seed { s: seed, e: None },
            None,
            LaneDir::Lower,
            Some(&exc),
            false,
        )
        .expect("batched");
    let low_b = eng.concretize_lower(&batched);
    // Separate passes with full gate overrides.
    for (kc, &(li, pos)) in cands.iter().enumerate() {
        for (d_i, dr) in [(0usize, 1i8), (1usize, -1i8)] {
            let dom = domain_gates(&root, &[(li, pos, dr)]);
            let sep = eng
                .run(&ms.seed, Some(&dom), LaneDir::Lower, None, false)
                .expect("separate");
            let low_s = eng.concretize_lower(&sep);
            for f in 0..nf {
                let vb = low_b[(2 * kc + d_i) * nf + f];
                let vs = low_s[f];
                assert!(
                    (vb - vs).abs() < 1e-10,
                    "cand {kc} dir {dr} row {f}: batched {vb} vs separate {vs}"
                );
            }
        }
    }
}

/// head_variant (the exact single-gate ranker) matches a fresh seed rebuild.
#[test]
fn head_variant_matches_rebuilt_gates() {
    let mut rng = StdRng::seed_from_u64(23);
    let spec = tiny_spec(&mut rng, 0.7);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.5; 72];
    let hi = vec![0.5; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let mb = MarginBatch::new(&net, 0, &[1, 3]).expect("mb");
    let gates = head_gates(&ybox, RoundMode::Parity);
    let vs = variant_state(&mb, &gates, &ybox, &al, &au);
    let ald = row_dots(&root, &al);
    let aud = row_dots(&root, &au);
    for i in 0..net.n_y {
        if !(ybox.ly[i] < 0.0 && ybox.uy[i] > 0.0) {
            continue;
        }
        for dir in [1i8, -1] {
            let fast = head_variant(&mb, &vs, &gates, &ybox, &al, &au, &root, i, dir);
            // Rebuild: modified gates -> m1' and m2v', min over classes.
            let mut g2 = head_gates(&ybox, RoundMode::Parity);
            let mut yb2 = ybox.clone();
            if dir > 0 {
                g2.alpha[i] = 1.0;
                g2.s[i] = 1.0;
                g2.c[i] = 0.0;
            } else {
                g2.alpha[i] = 0.0;
                g2.s[i] = 0.0;
                g2.c[i] = 0.0;
                yb2.uy[i] = yb2.uy[i].min(0.0);
            }
            let ms2 = margin_seed(&mb, &g2, &yb2, RoundMode::Parity);
            let m2v = compose_viay(&eng, &mb, &g2, &al, &au, &ald, &aud, RoundMode::Parity);
            let slow = (0..mb.nf())
                .map(|k| ms2.m1[k].max(m2v[k]))
                .fold(f64::INFINITY, f64::min);
            assert!(
                (fast - slow).abs() < 1e-10,
                "variant i={i} dir={dir}: fast {fast} vs rebuilt {slow}"
            );
        }
    }
}

/// Full-lane smoke: a fat-margin spec verifies (Unsat); a spec violated AT THE
/// BOX MIDPOINT must never verify (the moat).
#[test]
fn lane_verdicts_unsat_and_fail_closed() {
    let mut rng = StdRng::seed_from_u64(29);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).expect("compile");
    // Find the argmax class at the midpoint, then verify with a tiny box.
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let mid = Array2::from_shape_fn((72, 1), |(i, _)| f64::midpoint(lo[i], hi[i]));
    let (y, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let scores: Vec<f64> = (0..n_out)
        .map(|o| {
            let mut sm = b2[o];
            for k in 0..n_y {
                sm += w2[o * n_y + k] * y[[k, 0]].max(0.0);
            }
            sm
        })
        .collect();
    let t = (0..n_out)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let worst = (0..n_out)
        .filter(|&o| o != t)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let adv: Vec<usize> = (0..n_out).filter(|&o| o != t).collect();
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    match MarginRowBab::run(&net, &root, t, &adv, BabConfig::default()) {
        MarginRowOutcome::Unsat(stats) => {
            assert_eq!(
                stats.tree_classes.len() + stats.root_closed_classes,
                adv.len()
            );
        }
        MarginRowOutcome::Unknown { reason, .. } => {
            panic!("tiny-box argmax instance should verify, got Unknown: {reason}")
        }
    }
    // MOAT: swap true/adv so the property is violated at the midpoint — the
    // lane must NOT return Unsat (margin t'-t is negative at mid).
    match MarginRowBab::run(&net, &root, worst, &[t], BabConfig::default()) {
        MarginRowOutcome::Unsat(_) => panic!("MOAT VIOLATION: verified a falsified instance"),
        MarginRowOutcome::Unknown { .. } => {}
    }
}

#[test]
fn classwise_switch_and_schedule_are_bounded_and_fail_closed() {
    assert!(!super::margin_row_classwise_from_env(None));
    assert!(!super::margin_row_classwise_from_env(Some("0")));
    assert!(!super::margin_row_classwise_from_env(Some("invalid")));
    assert!(super::margin_row_classwise_from_env(Some("1")));
    assert!(super::margin_row_classwise_from_env(Some(" TRUE ")));

    // Non-contiguous class values catch an unsound `dj[class]` lookup. Class 7
    // closes; ties sort by class id. Strict zero remains scheduled.
    assert_eq!(
        classwise_schedule(1, &[7, 9, 2, 4], &[0.2, -0.1, -0.1, 0.0]).unwrap(),
        vec![(2, -0.1), (9, -0.1), (4, 0.0)]
    );
    assert!(classwise_schedule(1, &[1], &[-0.1]).is_err());
    assert!(classwise_schedule(1, &[2, 2], &[-0.1, -0.2]).is_err());
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(classwise_schedule(1, &[2], &[nonfinite]).is_err());
    }
    assert!(classwise_schedule(1, &[2], &[]).is_err());
}

#[test]
fn classwise_aggregation_requires_every_scheduled_certificate() {
    let mut aggregate = BabStats {
        root_bound: -0.2,
        tree_classes: vec![2, 9],
        root_closed_classes: 1,
        expansions: 0,
        domains_created: 0,
        closed: 0,
        max_depth: 0,
        mono_raw_dips: 0,
        mono_worst: 0.0,
        stop: "test".into(),
        elapsed_secs: 0.0,
        class_runs: Vec::new(),
        epochs_attempted: 0,
        epochs_closed: 0,
        ledger_ok: Some(true),
    };
    let record = |class, verified, expansions, domains_created, max_depth| ClassBabStats {
        class,
        root_bound: if class == 2 { -0.2 } else { -0.1 },
        verified,
        expansions,
        domains_created,
        closed: domains_created / 2,
        max_depth,
        mono_raw_dips: 1,
        mono_worst: -0.01,
        stop: if verified { "queue_empty" } else { "wallclock" }.into(),
        elapsed_secs: 0.1,
        epochs_attempted: 0,
        epochs_closed: 0,
        ledger_ok: verified.then_some(true),
    };
    absorb_class_stats(&mut aggregate, record(2, true, 3, 7, 2));
    assert!(!classwise_conjunction_complete(
        &aggregate.tree_classes,
        &aggregate.class_runs
    ));
    absorb_class_stats(&mut aggregate, record(9, true, 4, 9, 5));
    assert!(classwise_conjunction_complete(
        &aggregate.tree_classes,
        &aggregate.class_runs
    ));
    assert_eq!(
        (
            aggregate.expansions,
            aggregate.domains_created,
            aggregate.max_depth
        ),
        (7, 16, 5)
    );
    let mut failed = aggregate.class_runs.clone();
    failed[1].verified = false;
    assert!(!classwise_conjunction_complete(
        &aggregate.tree_classes,
        &failed
    ));
    failed[1].verified = true;
    failed.swap(0, 1);
    assert!(!classwise_conjunction_complete(
        &aggregate.tree_classes,
        &failed
    ));
}

#[test]
fn classwise_root_pack_is_shared_but_class_caches_are_isolated() {
    let mut rng = StdRng::seed_from_u64(30);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).unwrap();
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).unwrap();
    let (t, _, adv) = tiny_targets(&net, &lo, &hi);
    let eng = BackwardEngine::new(&net, &root);
    let re = super::bab::root_eval(&eng, &net, t, &adv).unwrap();
    assert!(classwise_root_cache_isolated_for_test(&re));
}

#[test]
fn classwise_and_joint_both_certify_a_robust_box() {
    let mut rng = StdRng::seed_from_u64(29);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).unwrap();
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let (t, _, adv) = tiny_targets(&net, &lo, &hi);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).unwrap();
    let cfg = BabConfig {
        lru_cap: 64,
        frontier: 1,
        ..BabConfig::default()
    };
    assert!(matches!(
        MarginRowBab::run(&net, &root, t, &adv, cfg.clone()),
        MarginRowOutcome::Unsat(_)
    ));
    let MarginRowOutcome::Unsat(stats) = MarginRowBab::run_classwise(&net, &root, t, &adv, cfg)
    else {
        panic!("classwise should certify the same robust box")
    };
    assert_eq!(
        stats.root_closed_classes + stats.tree_classes.len(),
        adv.len()
    );
}

#[test]
fn classwise_one_class_identity_deadline_and_global_cap() {
    let mut rng = StdRng::seed_from_u64(32);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).unwrap();
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let (t, runner_up, adv) = tiny_targets(&net, &lo, &hi);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).unwrap();
    let cfg = BabConfig {
        max_expansions: 8,
        lru_cap: 64,
        frontier: 1,
        ..BabConfig::default()
    };
    let joint = MarginRowBab::run(&net, &root, runner_up, &[t], cfg.clone());
    let classwise = MarginRowBab::run_classwise(&net, &root, runner_up, &[t], cfg.clone());
    let MarginRowOutcome::Unknown {
        stats: Some(js), ..
    } = joint
    else {
        panic!()
    };
    let MarginRowOutcome::Unknown {
        stats: Some(cs), ..
    } = classwise
    else {
        panic!()
    };
    assert_eq!(js.root_bound.to_bits(), cs.root_bound.to_bits());
    assert_eq!(js.expansions, cs.expansions);
    assert!(cs.expansions <= cfg.max_expansions);
    assert_eq!(cs.class_runs.len(), 1);
    assert!(!cs.class_runs[0].verified);

    let expired = BabConfig {
        deadline: Some(
            std::time::Instant::now()
                .checked_sub(std::time::Duration::from_millis(1))
                .unwrap(),
        ),
        ..BabConfig::default()
    };
    // Robust/all-root-closed input still may not certify after its deadline.
    assert!(matches!(
        MarginRowBab::run_classwise(&net, &root, t, &adv, expired),
        MarginRowOutcome::Unknown { .. }
    ));
    let parity = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).unwrap();
    assert!(matches!(
        MarginRowBab::run_classwise(&net, &parity, t, &adv, BabConfig::default()),
        MarginRowOutcome::Unknown { .. }
    ));

    let all_adv: Vec<_> = (0..4).filter(|class| *class != runner_up).collect();
    let zero = BabConfig {
        max_expansions: 0,
        ..BabConfig::default()
    };
    let MarginRowOutcome::Unknown {
        stats: Some(zs), ..
    } = MarginRowBab::run_classwise(&net, &root, runner_up, &all_adv, zero)
    else {
        panic!("zero budget must fail closed")
    };
    assert_eq!(zs.expansions, 0);
    assert_eq!(zs.class_runs.len(), 1);
    assert_eq!(zs.class_runs[0].stop, "global_max_expansions_before_class");
}

// ===================== ADVERSARIAL VERIFIER TESTS (not for commit) =========
// Pointwise sign-cover soundness: sample 1000 points in a SPLIT parent,
// compute the exact pre-activation / head value per point, and check that the
// bound of the point's OWN branch is <= the true margin at that point, for
// every adv class. Exercises: domain_gates piece-fixing, head clamps,
// per-domain gate refresh, the direct/m1/m2v verdict paths, and the batched
// EXCEPTION scoring path.

/// Per-class certified bound of a domain (direct ∨ m1 ∨ m2v), outward mode.
#[allow(clippy::type_complexity)]
fn domain_class_bounds(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    trunk: &[(usize, usize, i8)],
    heads: &[(usize, i8)],
) -> Vec<f64> {
    let eng = BackwardEngine::new(net, root);
    let dom = domain_gates(root, trunk);
    let dom_opt = (!trunk.is_empty()).then_some(&dom);
    let (al, au) = eng.y_rows(dom_opt).expect("y_rows");
    let mut ybox = YBox::from_rows(&eng, &al, &au);
    ybox.clamp(heads);
    let mb = MarginBatch::new(net, t, adv).expect("mb");
    let gates = head_gates(&ybox, root.mode);
    let ms = margin_seed(&mb, &gates, &ybox, root.mode);
    let pass = eng
        .run(&ms.seed, dom_opt, LaneDir::Lower, None, false)
        .expect("pass");
    let direct = per_class_direct(&eng, &pass, &ms, 0..adv.len());
    let ald = row_dots(root, &al);
    let aud = row_dots(root, &au);
    let m2v = compose_viay(&eng, &mb, &gates, &al, &au, &ald, &aud, root.mode);
    (0..adv.len())
        .map(|k| direct[k].max(ms.m1[k]).max(m2v[k]))
        .collect()
}

#[test]
fn verifier_pointwise_sign_cover_1000pts() {
    let mut rng = StdRng::seed_from_u64(424_242);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.35; 72];
    let hi = vec![0.45; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    // Pick a trunk split (layer with unstable neurons) and a head split.
    let (li, pos) = root
        .layers
        .iter()
        .enumerate()
        .find_map(|(li, lg)| (!lg.unst.is_empty()).then_some((li, 0usize)))
        .expect("need an unstable trunk neuron");
    let relu_op = root.layers[li].op;
    let idx = root.layers[li].unst[pos];
    // Head neuron: unstable in the root ybox.
    let eng0 = BackwardEngine::new(&net, &root);
    let (al0, au0) = eng0.y_rows(None).expect("y_rows");
    let ybox0 = YBox::from_rows(&eng0, &al0, &au0);
    let hi_idx = (0..net.n_y)
        .find(|&i| ybox0.ly[i] < 0.0 && ybox0.uy[i] > 0.0)
        .expect("need an unstable head neuron");
    // 4 leaf domains: (trunk±, head±). Bounds per class.
    let mut leaf_bounds = std::collections::BTreeMap::new();
    for td in [1i8, -1] {
        for hd in [1i8, -1] {
            let b = domain_class_bounds(&net, &root, t, &adv, &[(li, pos, td)], &[(hi_idx, hd)]);
            leaf_bounds.insert((td, hd), b);
        }
    }
    // Sample 1000 points; route each by its EXACT sign pair; check its own
    // leaf's bound <= its margin for every class. Boundary (=0) belongs to
    // both; check the branch we route to (sign >= 0 -> +).
    let x = sample_box(&mut rng, &root, 1000);
    let sel: std::collections::BTreeMap<usize, Vec<usize>> =
        [(relu_op, vec![idx])].into_iter().collect();
    let (y, pre) = net.forward_points(&x, &sel).expect("forward");
    let margins = margins_at(&net, &y, t, &adv);
    let pre_t = &pre[&relu_op];
    let mut counts = std::collections::BTreeMap::new();
    for b in 0..x.ncols() {
        let td: i8 = if pre_t[[0, b]] >= 0.0 { 1 } else { -1 };
        let hd: i8 = if y[[hi_idx, b]] >= 0.0 { 1 } else { -1 };
        *counts.entry((td, hd)).or_insert(0usize) += 1;
        let bounds = &leaf_bounds[&(td, hd)];
        for (k, mrow) in margins.iter().enumerate() {
            assert!(
                bounds[k] <= mrow[b] + 0.0,
                "SIGN-COVER VIOLATION: point {b} in leaf ({td},{hd}) class {k}: \
                 bound {} > true margin {} (pre={}, y_h={})",
                bounds[k],
                mrow[b],
                pre_t[[0, b]],
                y[[hi_idx, b]]
            );
        }
    }
    // The union covers: every point routed somewhere; require >=2 leaves hit
    // so the test isn't vacuous.
    assert!(counts.len() >= 2, "degenerate sampling: {counts:?}");
    eprintln!("[verifier] sign-cover leaf population: {counts:?}");
}

/// Batched-exception child scoring is pointwise sound: for each trunk
/// candidate direction, the scored child bound must be <= the true margin at
/// every sampled point whose exact pre-activation matches that direction.
#[test]
fn verifier_exception_scored_children_pointwise_sound() {
    let mut rng = StdRng::seed_from_u64(313_131);
    let spec = tiny_spec(&mut rng, 0.65);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    let mb = MarginBatch::new(&net, t, &adv).expect("mb");
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let gates = head_gates(&ybox, RoundMode::Outward);
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Outward);
    let nf = mb.nf();
    // All (layer, pos) candidates over every unstable trunk neuron.
    let mut cands: Vec<(usize, usize)> = Vec::new();
    for (li, lg) in root.layers.iter().enumerate() {
        for pos in 0..lg.unst.len() {
            cands.push((li, pos));
        }
    }
    let total = 2 * cands.len() * nf;
    let mut seed = Array2::<f64>::zeros((net.n_y, total));
    let mut seed_e = Array2::<f64>::zeros((net.n_y, total));
    let mut exc = Exceptions::default();
    for (kc, &(li, pos)) in cands.iter().enumerate() {
        let idx = root.layers[li].unst[pos];
        for (d_i, fix) in [(0usize, (1.0, 1.0, 0.0)), (1usize, (0.0, 0.0, 0.0))] {
            let r0 = (2 * kc + d_i) * nf;
            for j in 0..net.n_y {
                for f in 0..nf {
                    seed[[j, r0 + f]] = ms.seed.s[[j, f]];
                    seed_e[[j, r0 + f]] = ms.seed.e.as_ref().expect("outward")[[j, f]];
                }
            }
            for f in 0..nf {
                exc.by_layer.entry(li).or_default().push(Exc {
                    row: r0 + f,
                    neuron: idx,
                    a2: fix.0,
                    s2: fix.1,
                    c2: fix.2,
                });
            }
        }
    }
    let pass = eng
        .run(
            &super::engine::Seed {
                s: seed,
                e: Some(seed_e),
            },
            None,
            LaneDir::Lower,
            Some(&exc),
            false,
        )
        .expect("exception pass");
    let low = eng.concretize_lower(&pass);
    // Ground truth at 1000 points, with pre-activations at every candidate.
    let mut sel: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (li, lg) in root.layers.iter().enumerate() {
        let _ = li;
        sel.entry(lg.op).or_default();
    }
    for &(li, pos) in &cands {
        let lg = &root.layers[li];
        sel.get_mut(&lg.op).expect("inserted").push(lg.unst[pos]);
    }
    let x = sample_box(&mut rng, &root, 1000);
    let (y, pre) = net.forward_points(&x, &sel).expect("forward");
    let margins = margins_at(&net, &y, t, &adv);
    let mut checked = 0usize;
    for (kc, &(li, pos)) in cands.iter().enumerate() {
        let lg = &root.layers[li];
        let row_in_sel = sel[&lg.op]
            .iter()
            .position(|&n| n == lg.unst[pos])
            .expect("present");
        let pv = &pre[&lg.op];
        for b in 0..x.ncols() {
            let sgn = pv[[row_in_sel, b]];
            let d_i = usize::from(sgn < 0.0); // >=0 -> block 0 (active fix)
            let r0 = (2 * kc + d_i) * nf;
            for f in 0..nf {
                let v = low[r0 + f] + ms.cst[f];
                let v = super::rounding::next_down(super::rounding::next_down(
                    v - super::rounding::next_up(ms.cst_err[f]),
                ));
                let m = margins[f][b];
                assert!(
                    v <= m,
                    "EXCEPTION-CHILD VIOLATION: cand {kc} (li={li},pos={pos}) dir_i={d_i} \
                     class {f} point {b}: scored bound {v} > true margin {m} (pre={sgn})"
                );
                checked += 1;
            }
        }
    }
    eprintln!("[verifier] exception-child pointwise checks: {checked}");
}

/// 1-ulp tamper sensitivity: nudging ONE weight by one ulp must change the
/// parity bound when the bound's own ulp is below the contribution (small-box
/// regime; tampers the final-Gemm margin weight which enters the seed
/// directly). Reports the measured resolution of the differential oracle.
#[test]
fn verifier_one_ulp_weight_tamper_shifts_bound() {
    let mut rng = StdRng::seed_from_u64(717_171);
    let spec = tiny_spec(&mut rng, 0.6);
    // Small box => |bound| small => bound ulp below the tamper contribution.
    let lo = vec![0.005; 72];
    let hi = vec![0.02; 72];
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    let bound_of = |sp: &TwinSpec| -> f64 {
        let net = TwinNet::compile(sp).expect("compile");
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("root");
        let b = domain_class_bounds(&net, &root, t, &adv, &[], &[]);
        b.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let b0 = bound_of(&spec);
    // Tamper each of the final Gemm's weights in turn (they enter the margin
    // rows directly); record the largest bound shift a single-ulp nudge makes.
    let mut max_delta = 0.0f64;
    let n_last = match spec.ops.last().expect("ops") {
        TwinOpSpec::Gemm { weight, .. } => weight.len(),
        _ => 0,
    };
    for wi in 0..n_last {
        let mut spec2 = spec.clone();
        if let Some(TwinOpSpec::Gemm { weight, .. }) = spec2.ops.last_mut() {
            weight[wi] = super::rounding::next_up(weight[wi]);
        }
        let b1 = bound_of(&spec2);
        max_delta = max_delta.max((b1 - b0).abs());
        if max_delta > 0.0 {
            eprintln!(
                "[verifier] 1-ulp gemm2 tamper at w[{wi}]: bound {b0:.17} -> {b1:.17} \
                 (delta {:.3e})",
                (b1 - b0).abs()
            );
            break;
        }
    }
    eprintln!("[verifier] max single-ulp tamper delta observed: {max_delta:.3e}");
    assert!(
        max_delta > 0.0,
        "no single-ulp gemm2 tamper shifted the bound: differential oracle \
         has no resolution at the ulp scale (bound {b0:.17})"
    );
}
// ===================== END ADVERSARIAL VERIFIER TESTS ======================

/// Test-only scalar oracle for [`super::net::conv_apply_backward`]. It keeps
/// the historic per-row arithmetic contract explicit: taps in table order,
/// then output channels in ascending order, then the contiguous row payload.
fn conv_backward_serial_reference(
    c: &super::net::ConvOp,
    src: &Array2<f64>,
    dst: &mut Array2<f64>,
    abs_w: bool,
) {
    let (co, ci, kh, kw) = c.kernel;
    let ip = c.ishape.1 * c.ishape.2;
    let op_ = c.oshape.1 * c.oshape.2;
    let r = src.ncols();
    assert_eq!(dst.dim(), (ci * ip, r));
    for ic in 0..ci {
        for isp in 0..ip {
            let mut acc = dst.row_mut(ic * ip + isp);
            acc.fill(0.0);
            for &(ky, kx, osp) in &c.back_taps[isp] {
                let wbase = ((ic * kh + ky) * kw + kx) * co;
                for oc in 0..co {
                    let w0 = c.wt[wbase + oc];
                    let w = if abs_w { w0.abs() } else { w0 };
                    if w == 0.0 {
                        continue;
                    }
                    let src_row = src.row(oc * op_ + osp);
                    for (a, &s) in acc.iter_mut().zip(src_row) {
                        *a += w * s;
                    }
                }
            }
        }
    }
}

/// Exact-semantic regression for the backward-conv Rayon regraining. The
/// production row grain must equal a serial oracle to the last bit for both
/// the signed coefficient tableau and the non-negative absolute-weight pass
/// that feeds the certified coefficient-error array.
#[test]
fn conv_backward_row_grain_bit_identical_for_coeff_and_error_lanes() {
    use super::net::{conv_apply_backward, TwinOp};

    let mut rng = StdRng::seed_from_u64(0xBACC_6A1A_2026);
    let specs = [tiny_spec(&mut rng, 0.8), tiny_spec(&mut rng, 1.3)];
    for spec in &specs {
        let net = TwinNet::compile(spec).expect("compile");
        for op in &net.ops {
            let TwinOp::Conv(c) = op else { continue };
            let n_in_t = c.ishape.0 * c.ishape.1 * c.ishape.2;
            let n_out_t = c.oshape.0 * c.oshape.1 * c.oshape.2;
            // 320 is the maximum candidate-tableau width for ten live margin
            // rows with the production 8+8 shortlist; the smaller widths
            // cover tail/scheduling boundaries.
            for &r in &[1usize, 7, 32, 320] {
                let signed =
                    Array2::<f64>::from_shape_fn((n_out_t, r), |_| rng.random_range(-2.0..2.0));
                let error =
                    Array2::<f64>::from_shape_fn((n_out_t, r), |_| rng.random_range(0.0..2.0));
                for (lane, src, abs_w) in [("coefficient", &signed, false), ("error", &error, true)]
                {
                    let mut serial = Array2::<f64>::zeros((n_in_t, r));
                    conv_backward_serial_reference(c, src, &mut serial, abs_w);
                    let mut row_grain = Array2::<f64>::zeros((n_in_t, r));
                    conv_apply_backward(c, src, &mut row_grain, abs_w);
                    for (idx, (&a, &b)) in serial.iter().zip(&row_grain).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{lane} lane mismatch at flat index {idx}, r={r}, \
                             abs_w={abs_w}: serial={a} row_grain={b}"
                        );
                    }
                }
            }
        }
    }
}

/// VALUE-IDENTITY PROOF (kernel level): the row-parallel and cache-blocked
/// forward convs are BIT-IDENTICAL to the legacy per-output-channel grain (the
/// oracle). All three fold each output row's taps in ascending order into a
/// freshly zeroed accumulator with the same non-fused `acc += w*s` step,
/// skipping zero weights / padded taps identically; only the loop nesting and
/// thread ownership differ. Checked to the last bit (`to_bits`) over several
/// conv geometries, batch widths (incl. the r=3073 tableau width and widths
/// that straddle the block size), and both the signed (M) and abs (D) lanes.
#[test]
fn conv_forward_grains_bit_identical_to_oc_grain() {
    use super::net::{
        conv_forward_blocked, conv_forward_ocgrain, conv_forward_rowgrain, ConvOp, TwinOp,
    };
    let mut rng = StdRng::seed_from_u64(20_250_717);
    let specs = [tiny_spec(&mut rng, 0.8), tiny_spec(&mut rng, 1.3)];
    for spec in &specs {
        let net = TwinNet::compile(spec).expect("compile");
        for op in &net.ops {
            let TwinOp::Conv(c) = op else { continue };
            let cc: &ConvOp = c;
            let n_in_t = cc.ishape.0 * cc.ishape.1 * cc.ishape.2;
            let n_out_t = cc.oshape.0 * cc.oshape.1 * cc.oshape.2;
            for &r in &[1usize, 4, 37, 500, 3073] {
                let src =
                    Array2::<f64>::from_shape_fn((n_in_t, r), |_| rng.random_range(-2.0..2.0));
                for abs_w in [false, true] {
                    let mut d_oc = Array2::<f64>::zeros((n_out_t, r));
                    conv_forward_ocgrain(cc, &src, &mut d_oc, abs_w);
                    let mut d_row = Array2::<f64>::zeros((n_out_t, r));
                    conv_forward_rowgrain(cc, &src, &mut d_row, abs_w);
                    let mut d_blk = Array2::<f64>::zeros((n_out_t, r));
                    conv_forward_blocked(cc, &src, &mut d_blk, abs_w);
                    for ((a, b), d) in d_oc.iter().zip(d_row.iter()).zip(d_blk.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "row-grain mismatch abs_w={abs_w} r={r}: {a} vs {b}"
                        );
                        assert_eq!(
                            a.to_bits(),
                            d.to_bits(),
                            "blocked mismatch abs_w={abs_w} r={r}: {a} vs {d}"
                        );
                    }
                }
            }
        }
    }
}

/// VALUE-IDENTITY PROOF (build level): `RootGates::build` produces BIT-IDENTICAL
/// gates and boxes under EVERY conv grain — the cache-blocked default, the
/// `NY_MARGIN_ROW_ROOT_PAR=row` grain, and the `=0` legacy oracle — in the
/// verdict-grade Outward mode AND in Parity. Every per-neuron gate field and
/// box endpoint matches the oracle to the last bit, so the default flip cannot
/// move any bound. (All grains are bit-identical, so any concurrent test
/// reading the env is unaffected.)
#[test]
fn root_build_bit_identical_across_conv_grain() {
    let mut rng = StdRng::seed_from_u64(0x00C0_FFEE);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.3; 72];
    let hi = vec![0.5; 72];
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    // Serialized env scope (clippy env wall); pre-test state restored on exit.
    ny_test_utils::env::with_env_edits(|env| {
        for mode in [RoundMode::Outward, RoundMode::Parity] {
            // Oracle: legacy per-channel grain.
            env.set("NY_MARGIN_ROW_ROOT_PAR", "0");
            let a = RootGates::build(&net, &lo, &hi, mode, None).expect("root oc-grain");
            // Every other grain must match the oracle bit-for-bit.
            for grain in ["row", "blocked-default"] {
                if grain == "blocked-default" {
                    env.remove("NY_MARGIN_ROW_ROOT_PAR");
                } else {
                    env.set("NY_MARGIN_ROW_ROOT_PAR", grain);
                }
                let b = RootGates::build(&net, &lo, &hi, mode, None).expect("root grain");
                assert_eq!(
                    a.layers.len(),
                    b.layers.len(),
                    "{mode:?}/{grain} layer count"
                );
                assert_eq!(bits(&a.mid), bits(&b.mid), "{mode:?}/{grain} mid");
                assert_eq!(bits(&a.rad), bits(&b.rad), "{mode:?}/{grain} rad");
                assert_eq!(bits(&a.xabs), bits(&b.xabs), "{mode:?}/{grain} xabs");
                for (la, lb) in a.layers.iter().zip(&b.layers) {
                    assert_eq!(la.op, lb.op, "{mode:?}/{grain} op");
                    assert_eq!(la.n, lb.n, "{mode:?}/{grain} n");
                    assert_eq!(la.unst, lb.unst, "{mode:?}/{grain} unst");
                    assert_eq!(bits(&la.l), bits(&lb.l), "{mode:?}/{grain} l");
                    assert_eq!(bits(&la.u), bits(&lb.u), "{mode:?}/{grain} u");
                    assert_eq!(bits(&la.alpha), bits(&lb.alpha), "{mode:?}/{grain} alpha");
                    assert_eq!(bits(&la.s), bits(&lb.s), "{mode:?}/{grain} s");
                    assert_eq!(bits(&la.c), bits(&lb.c), "{mode:?}/{grain} c");
                    assert_eq!(bits(&la.ms), bits(&lb.ms), "{mode:?}/{grain} ms");
                }
            }
            env.remove("NY_MARGIN_ROW_ROOT_PAR");
        }
    });
}

/// Single-conv probe net for the backward grain tests: Conv (or ConvTranspose)
/// -> trunk Relu -> Flatten -> Gemm(3) -> Relu -> Gemm(2). Only the conv
/// matters — the head exists solely to satisfy `TwinNet::compile`'s twin-wall
/// structure validation.
fn single_conv_spec(rng: &mut StdRng, conv: TwinOpSpec) -> TwinSpec {
    let (n_in, n_out_t) = match &conv {
        TwinOpSpec::Conv { ishape, oshape, .. }
        | TwinOpSpec::ConvTranspose { ishape, oshape, .. } => (
            ishape.0 * ishape.1 * ishape.2,
            oshape.0 * oshape.1 * oshape.2,
        ),
        _ => unreachable!("probe spec takes a conv"),
    };
    let mut w = |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random_range(-0.5..0.5)).collect() };
    TwinSpec {
        n_in,
        ops: vec![
            conv,                             // t1
            TwinOpSpec::Relu { input: 1 },    // t2 (trunk relu 0)
            TwinOpSpec::Flatten { input: 2 }, // t3
            TwinOpSpec::Gemm {
                input: 3,
                weight: w(3 * n_out_t),
                bias: w(3),
                shape: (3, n_out_t),
            }, // t4 (y)
            TwinOpSpec::Relu { input: 4 },    // t5 (head relu)
            TwinOpSpec::Gemm {
                input: 5,
                weight: w(2 * 3),
                bias: w(2),
                shape: (2, 3),
            }, // t6
        ],
    }
}

/// Random conv spec for a given geometry; every 7th weight is EXACTLY zero so
/// the zero-weight skip is exercised identically across the backward grains.
#[allow(clippy::too_many_arguments)]
fn conv_case(
    rng: &mut StdRng,
    kernel: (usize, usize, usize, usize),
    stride: (usize, usize),
    pads: (usize, usize, usize, usize),
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
) -> TwinOpSpec {
    let (co, ci, kh, kw) = kernel;
    let weight: Vec<f64> = (0..co * ci * kh * kw)
        .map(|i| {
            if i % 7 == 0 {
                0.0
            } else {
                rng.random_range(-0.5..0.5)
            }
        })
        .collect();
    TwinOpSpec::Conv {
        input: 0,
        weight,
        bias: vec![0.0; co],
        bias_err: vec![0.0; co],
        weight_rel_err: 1e-15,
        kernel,
        stride,
        pads,
        ishape,
        oshape,
    }
}

/// VALUE-IDENTITY PROOF (kernel level): the cache-blocked backward conv is
/// BIT-IDENTICAL to the legacy per-input-channel grain (the oracle). Both fold
/// each output row's `back_taps[isp]` triples in table order and, within each
/// tap, output channels in ascending `oc` order, into a freshly zeroed
/// accumulator with the same non-fused `acc += w*s` step, skipping zero
/// weights identically; only the loop nesting, the column blocking, and the
/// thread ownership differ. Checked to the last bit (`to_bits`) over: the two
/// tiny_spec convs (stride-2 downsample + residual geometry), a
/// tinyimagenet-class 56x56 body conv, the tinyimagenet-class ci=3
/// input-adjacent stride-2 conv (the pathological 3-rayon-unit legacy grain),
/// a cifar-class 32x32 conv, odd/asymmetric kernel+stride+pads, a 1x1 kernel,
/// a padded kernel larger than the input, a ci=64 case whose r=700 straddles
/// the r_blk=512 accumulator tile (two blocks incl. a 188-column remainder),
/// and a ConvTranspose (same kernels, transpose-built tap tables) — both the
/// signed (coef) and abs (certified-error) weight lanes.
#[test]
fn conv_backward_grains_bit_identical_to_ic_grain() {
    use super::net::{conv_backward_blocked, conv_backward_icgrain, ConvOp, TwinOp};
    let mut rng = StdRng::seed_from_u64(20_260_718);
    let mut cases: Vec<(TwinSpec, Vec<usize>)> = vec![
        // tiny_spec: conv1 (2,6,6)->(4,3,3) 3x3 s2 p1; conv2 (4,3,3)->(4,3,3).
        (tiny_spec(&mut rng, 0.8), vec![1, 4, 37, 500]),
    ];
    // tinyimagenet-class body conv: 56x56, multi-channel, 3x3 s1 p1.
    let c = conv_case(
        &mut rng,
        (8, 8, 3, 3),
        (1, 1),
        (1, 1, 1, 1),
        (8, 56, 56),
        (8, 56, 56),
    );
    cases.push((single_conv_spec(&mut rng, c), vec![1, 40]));
    // tinyimagenet-class input-adjacent downsample: ci=3 (the 3-unit legacy
    // grain), stride 2.
    let c = conv_case(
        &mut rng,
        (16, 3, 3, 3),
        (2, 2),
        (1, 1, 1, 1),
        (3, 56, 56),
        (16, 28, 28),
    );
    cases.push((single_conv_spec(&mut rng, c), vec![1, 40]));
    // cifar-class 32x32 conv.
    let c = conv_case(
        &mut rng,
        (8, 16, 3, 3),
        (1, 1),
        (1, 1, 1, 1),
        (16, 32, 32),
        (8, 32, 32),
    );
    cases.push((single_conv_spec(&mut rng, c), vec![17]));
    // Odd/asymmetric: 4x2 kernel, stride (3,2), pads (2,1,1,0), 9x7 input.
    let c = conv_case(
        &mut rng,
        (3, 2, 4, 2),
        (3, 2),
        (2, 1, 1, 0),
        (2, 9, 7),
        (3, 3, 4),
    );
    cases.push((single_conv_spec(&mut rng, c), vec![1, 37]));
    // 1x1 kernel, no padding, non-square input.
    let c = conv_case(
        &mut rng,
        (4, 5, 1, 1),
        (1, 1),
        (0, 0, 0, 0),
        (5, 7, 5),
        (4, 7, 5),
    );
    cases.push((single_conv_spec(&mut rng, c), vec![37]));
    // Kernel larger than the (padded) input dim: 3x3 on a 2x2 input.
    let c = conv_case(
        &mut rng,
        (2, 1, 3, 3),
        (1, 1),
        (1, 1, 1, 1),
        (1, 2, 2),
        (2, 2, 2),
    );
    cases.push((single_conv_spec(&mut rng, c), vec![1, 37]));
    // ci=64: r_blk = (1<<15)/64 = 512, so r=700 exercises TWO column blocks
    // (512 + a 188 remainder) of the blocked kernel.
    let c = conv_case(
        &mut rng,
        (8, 64, 3, 3),
        (1, 1),
        (1, 1, 1, 1),
        (64, 4, 4),
        (8, 4, 4),
    );
    cases.push((single_conv_spec(&mut rng, c), vec![700]));
    // ConvTranspose: stride 2, out_pad 1 — its own transpose-built tap tables
    // through the SAME backward kernels.
    let (co, ci, kh, kw) = (3usize, 4usize, 3usize, 3usize);
    let weight: Vec<f64> = (0..co * ci * kh * kw)
        .map(|i| {
            if i % 7 == 0 {
                0.0
            } else {
                rng.random_range(-0.5..0.5)
            }
        })
        .collect();
    let ct = TwinOpSpec::ConvTranspose {
        input: 0,
        weight,
        bias: vec![0.0; co],
        bias_err: vec![0.0; co],
        weight_rel_err: 1e-15,
        kernel: (co, ci, kh, kw),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (4, 5, 5),
        oshape: (3, 10, 10),
        out_pad: (1, 1),
    };
    cases.push((single_conv_spec(&mut rng, ct), vec![1, 37]));
    for (spec, rs) in &cases {
        let net = TwinNet::compile(spec).expect("compile");
        for op in &net.ops {
            let TwinOp::Conv(c) = op else { continue };
            let cc: &ConvOp = c;
            let n_in_t = cc.ishape.0 * cc.ishape.1 * cc.ishape.2;
            let n_out_t = cc.oshape.0 * cc.oshape.1 * cc.oshape.2;
            for &r in rs {
                let src =
                    Array2::<f64>::from_shape_fn((n_out_t, r), |_| rng.random_range(-2.0..2.0));
                for abs_w in [false, true] {
                    let mut d_ic = Array2::<f64>::zeros((n_in_t, r));
                    conv_backward_icgrain(cc, &src, &mut d_ic, abs_w);
                    let mut d_blk = Array2::<f64>::zeros((n_in_t, r));
                    conv_backward_blocked(cc, &src, &mut d_blk, abs_w);
                    for (a, b) in d_ic.iter().zip(d_blk.iter()) {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "blocked backward mismatch kernel={:?} abs_w={abs_w} r={r}: {a} vs {b}",
                            cc.kernel
                        );
                    }
                }
            }
        }
    }
}

/// GATE DISPATCH: `conv_apply_backward` routes unset/truthy values to the
/// scored cache-blocked default and explicit false values to the row-grain
/// oracle. Both kernels are bit-identical
/// (grain test above), so every assertion here is race-immune: a concurrent
/// test reading or flipping the env cannot change any compared bit — the same
/// reasoning as `root_build_bit_identical_across_conv_grain`'s env use.
#[test]
fn conv_backward_gate_dispatch_byte_identical() {
    use super::net::{
        conv_apply_backward, conv_backward_blocked, conv_backward_icgrain, ConvOp, TwinOp,
    };
    let mut rng = StdRng::seed_from_u64(0xB0D_6A7E);
    let spec = tiny_spec(&mut rng, 0.7);
    let net = TwinNet::compile(&spec).expect("compile");
    let TwinOp::Conv(c) = &net.ops[0] else {
        unreachable!("tiny spec starts with a conv")
    };
    let cc: &ConvOp = c;
    let n_in_t = cc.ishape.0 * cc.ishape.1 * cc.ishape.2;
    let n_out_t = cc.oshape.0 * cc.oshape.1 * cc.oshape.2;
    let r = 37;
    let src = Array2::<f64>::from_shape_fn((n_out_t, r), |_| rng.random_range(-2.0..2.0));
    let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    // Serialized env scope (clippy env wall); pre-test state restored on exit.
    ny_test_utils::env::with_env_edits(|env| {
        for abs_w in [false, true] {
            let mut d_ref = Array2::<f64>::zeros((n_in_t, r));
            conv_backward_icgrain(cc, &src, &mut d_ref, abs_w);
            let mut d_blk = Array2::<f64>::zeros((n_in_t, r));
            conv_backward_blocked(cc, &src, &mut d_blk, abs_w);
            for gate in [
                None,
                Some("0"),
                Some("false"),
                Some("off"),
                Some("no"),
                Some("1"),
                Some("true"),
            ] {
                match gate {
                    None => env.remove("NY_MARGIN_ROW_CONV_BWD_BLOCKED"),
                    Some(v) => env.set("NY_MARGIN_ROW_CONV_BWD_BLOCKED", v),
                }
                let mut d = Array2::<f64>::zeros((n_in_t, r));
                conv_apply_backward(cc, &src, &mut d, abs_w);
                assert_eq!(
                bits(&d),
                bits(&d_ref),
                "dispatch (gate={gate:?}, abs_w={abs_w}) not byte-identical to the ic-grain oracle"
            );
                assert_eq!(
                    bits(&d),
                    bits(&d_blk),
                    "dispatch (gate={gate:?}, abs_w={abs_w}) differs from the blocked kernel"
                );
            }
        }
    });
}

/// VALUE-IDENTITY PROOF (engine + tree level): a full backward pass (coef AND
/// certified-error lanes — both `conv_apply_backward` call sites, incl. the
/// abs-weight lane) and an entire BaB run produce BIT-IDENTICAL results with
/// the backward gate off vs forced on, so flipping
/// `NY_MARGIN_ROW_CONV_BWD_BLOCKED` cannot move any bound, verdict, or stat.
/// (Race-immune under concurrent env readers: the kernels are bit-identical.)
#[test]
fn engine_and_bab_bit_identical_across_backward_gate() {
    let mut rng = StdRng::seed_from_u64(0x0BAB_B10C);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.25; 72];
    let hi = vec![0.35; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let bits2 = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let bits1 = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    // Serialized env scope (clippy env wall); pre-test state restored on exit.
    ny_test_utils::env::with_env_edits(|env| {
        // Oracle: legacy grain (explicit gate-off).
        env.set("NY_MARGIN_ROW_CONV_BWD_BLOCKED", "0");
        let (al_a, au_a) = eng.y_rows(None).expect("y_rows oracle");
        // Forced-on: blocked backward under the full outward engine pass.
        env.set("NY_MARGIN_ROW_CONV_BWD_BLOCKED", "1");
        let (al_b, au_b) = eng.y_rows(None).expect("y_rows blocked");
        for (lane, pa, pb) in [("lower", &al_a, &al_b), ("upper", &au_a, &au_b)] {
            assert_eq!(bits2(&pa.a), bits2(&pb.a), "{lane} coef plane");
            assert_eq!(
                pa.e.as_ref().map(&bits2),
                pb.e.as_ref().map(&bits2),
                "{lane} certified-error plane"
            );
            assert_eq!(bits1(&pa.b), bits1(&pb.b), "{lane} bias");
            assert_eq!(bits1(&pa.eb), bits1(&pb.eb), "{lane} certified bias error");
        }
        // Full tree, closing orientation: verdict + root_bound + expansions + stop
        // identical across the gate.
        let (t, worst, adv) = tiny_targets(&net, &lo, &hi);
        let run = |env: &mut ny_test_utils::env::EnvEditor,
                   gate: &str,
                   t: usize,
                   adv: &[usize],
                   max_exp: usize| {
            env.set("NY_MARGIN_ROW_CONV_BWD_BLOCKED", gate);
            let cfg = BabConfig {
                max_expansions: max_exp,
                lru_cap: 64,
                ..BabConfig::default()
            };
            verdict_bits(&MarginRowBab::run(&net, &root, t, adv, cfg))
        };
        let a = run(env, "0", t, &adv, 2000);
        let b = run(env, "1", t, &adv, 2000);
        assert_eq!(
            a, b,
            "BaB (closing orientation) not bit-identical across the backward gate"
        );
        // Falsified orientation (never closes -> bounded deep walk): many
        // y-refresh/eval/score passes under the gate, still bit-identical.
        let adv_f: Vec<usize> = (0..4).filter(|&o| o != worst).collect();
        let a = run(env, "0", worst, &adv_f, 300);
        let b = run(env, "1", worst, &adv_f, 300);
        assert_eq!(
            a, b,
            "BaB (falsified orientation) not bit-identical across the backward gate"
        );
    });
}

/// Certified outward bounds stay below parity bounds on the same node when the
/// gate choices agree (tiny net, direct path), and the outward lane closes
/// nothing a sampled point refutes.
#[test]
fn outward_bound_no_higher_than_true_margin_min() {
    let mut rng = StdRng::seed_from_u64(31);
    let spec = tiny_spec(&mut rng, 0.5);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.2; 72];
    let hi = vec![0.3; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    let mb = MarginBatch::new(&net, t, &adv).expect("mb");
    let gates = head_gates(&ybox, RoundMode::Outward);
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Outward);
    let pass = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .expect("pass");
    let direct = per_class_direct(&eng, &pass, &ms, 0..adv.len());
    let x = sample_box(&mut rng, &root, 300);
    let (y, _) = net
        .forward_points(&x, &std::collections::BTreeMap::new())
        .expect("forward");
    let margins = margins_at(&net, &y, t, &adv);
    for (k, mrow) in margins.iter().enumerate() {
        let min_m = mrow.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            direct[k] <= min_m,
            "outward direct bound {} above sampled margin {min_m}",
            direct[k]
        );
    }
}

// ============ VERIFIER: parallel frontier vs serial oracle (added by the
// adversarial reviewer of c3cf76d4) ============
//
// Establishes, on synthetic nets, the three properties the parallelization
// must have: (1) a domain's certified bound is BIT-IDENTICAL whether computed
// serially or from a rayon worker (concurrency cannot move a bound); (2) the
// frontier lane's verdict + root bound match the serial oracle and are
// deterministic across runs; (3) the MOAT holds — a falsified instance is
// never certified UNSAT under the parallel lane.

/// Argmax target / runner-up / adv list for a tiny box, used to build both a
/// closing (target=argmax) and a falsified (target=runner-up) instance.
fn tiny_targets(net: &TwinNet, lo: &[f64], hi: &[f64]) -> (usize, usize, Vec<usize>) {
    let n_in = lo.len();
    let mid = Array2::from_shape_fn((n_in, 1), |(i, _)| f64::midpoint(lo[i], hi[i]));
    let (y, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let scores: Vec<f64> = (0..n_out)
        .map(|o| {
            let mut sm = b2[o];
            for k in 0..n_y {
                sm += w2[o * n_y + k] * y[[k, 0]].max(0.0);
            }
            sm
        })
        .collect();
    let t = (0..n_out)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .unwrap();
    let worst = (0..n_out)
        .filter(|&o| o != t)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .unwrap();
    (t, worst, (0..n_out).filter(|&o| o != t).collect())
}

fn verdict_bits(o: &MarginRowOutcome) -> (bool, u64, usize, String) {
    match o {
        MarginRowOutcome::Unsat(s) => (true, s.root_bound.to_bits(), s.expansions, s.stop.clone()),
        MarginRowOutcome::Unknown { stats, reason } => {
            stats.as_ref().map_or((false, 0, 0, reason.clone()), |s| {
                (false, s.root_bound.to_bits(), s.expansions, s.stop.clone())
            })
        }
    }
}

fn outcome_stats(out: &MarginRowOutcome) -> &BabStats {
    match out {
        MarginRowOutcome::Unsat(stats) => stats,
        MarginRowOutcome::Unknown { stats, .. } => stats.as_ref().expect("tree stats"),
    }
}

/// (1) DOMAIN-BOUND CONCURRENCY INVARIANCE: at the root and 5+ deeper domains,
/// the serial `eval_node` bound and 24 concurrent `eval_with_pack` bounds are
/// bit-identical. This is the load-bearing soundness claim of the frontier
/// lane — its per-domain bound is a pure fn of (pack, entry), so running it in
/// a worker cannot change a single bit.
#[test]
fn parallel_domain_bound_bit_identical_to_serial() {
    let mut rng = StdRng::seed_from_u64(101);
    let spec = tiny_spec(&mut rng, 0.5);
    let net = TwinNet::compile(&spec).expect("compile");
    // A wider box + falsified orientation so the tree keeps branching and we
    // reach deep domains (never closes → guaranteed depth).
    let lo = vec![-0.2; 72];
    let hi = vec![0.2; 72];
    let (t, worst, _adv) = tiny_targets(&net, &lo, &hi);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    // Target the runner-up so it never closes; adv = {argmax} plus fillers.
    let adv: Vec<usize> = (0..4).filter(|&o| o != worst).collect();
    let _ = t;
    let walk = MarginRowBab::diff_walk_serial_vs_worker(&net, &root, worst, &adv, 8);
    assert!(
        walk.len() >= 6,
        "need root + >=5 deep domains, only reached {}",
        walk.len()
    );
    let max_depth = walk.iter().map(|w| w.0).max().unwrap();
    assert!(max_depth >= 5, "did not reach depth 5, max={max_depth}");
    for (depth, bits, all_eq) in &walk {
        assert!(
            *all_eq,
            "SOUNDNESS BREAK: domain at depth {depth} bound {bits:#018x} differs between \
             serial and a concurrent worker",
        );
    }
}

/// CROSS-DOMAIN ENGINE ORACLE: three independently evaluated candidate-row
/// matrices with different piece-fixed gates, different local exceptions,
/// different widths, and certified outward error lanes must be bit-for-bit
/// equal to one explicitly owned contiguous stack. Both backward directions
/// are covered even though the initial canary consumes only the lower lane.
#[test]
fn domain_stack_engine_is_bit_exact_across_gates_exceptions_and_outward_lanes() {
    let mut rng = StdRng::seed_from_u64(0xd05a_1a5e);
    let spec = tiny_spec(&mut rng, 0.45);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.25; 72];
    let hi = vec![0.30; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let unstable: Vec<(usize, usize, usize)> = root
        .layers
        .iter()
        .enumerate()
        .filter_map(|(li, rec)| rec.unst.first().map(|&idx| (li, 0, idx)))
        .collect();
    assert!(
        unstable.len() >= 2,
        "fixture needs two unstable trunk layers"
    );
    let (li0, pos0, idx0) = unstable[0];
    let (li1, pos1, idx1) = unstable[1];
    let domains = [
        domain_gates(&root, &[]),
        domain_gates(&root, &[(li0, pos0, 1)]),
        domain_gates(&root, &[(li0, pos0, -1), (li1, pos1, 1)]),
    ];
    let widths = [3usize, 5, 4];
    let mut exceptions: Vec<Exceptions> = (0..3).map(|_| Exceptions::default()).collect();
    exceptions[0].by_layer.entry(li0).or_default().push(Exc {
        row: 1,
        neuron: idx0,
        a2: 1.0,
        s2: 1.0,
        c2: 0.0,
    });
    // Same neuron as domain 1's override: using domain 0's non-zero root
    // intercept here would move the exception correction and fail the oracle.
    exceptions[1].by_layer.entry(li0).or_default().push(Exc {
        row: 4,
        neuron: idx0,
        a2: 0.0,
        s2: 0.0,
        c2: 0.0,
    });
    exceptions[2].by_layer.entry(li0).or_default().push(Exc {
        row: 0,
        neuron: idx0,
        a2: 1.0,
        s2: 1.0,
        c2: 0.0,
    });
    exceptions[2].by_layer.entry(li1).or_default().push(Exc {
        row: 3,
        neuron: idx1,
        a2: 0.0,
        s2: 0.0,
        c2: 0.0,
    });
    let seeds: Vec<DomainStackSeed> = widths
        .iter()
        .enumerate()
        .map(|(d, &width)| {
            let s = Array2::from_shape_fn((net.n_y, width), |(j, r)| {
                let k = 1 + d * 97 + j * 17 + r * 7;
                ((k as f64) * 0.173).sin() * (1.0 + d as f64 * 0.125)
            });
            let e = Array2::from_shape_fn((net.n_y, width), |(j, r)| {
                (1 + d + j + 3 * r) as f64 * 1e-16
            });
            DomainStackSeed { s, e: Some(e) }
        })
        .collect();
    let total: usize = widths.iter().sum();
    let mut stacked_s = Array2::<f64>::zeros((net.n_y, total));
    let mut stacked_e = Array2::<f64>::zeros((net.n_y, total));
    let mut offset = 0usize;
    for (seed, &width) in seeds.iter().zip(&widths) {
        for j in 0..net.n_y {
            let src = &seed.s.as_slice().expect("layout")[j * width..(j + 1) * width];
            stacked_s.as_slice_mut().expect("layout")
                [j * total + offset..j * total + offset + width]
                .copy_from_slice(src);
            let src = &seed
                .e
                .as_ref()
                .expect("outward")
                .as_slice()
                .expect("layout")[j * width..(j + 1) * width];
            stacked_e.as_slice_mut().expect("layout")
                [j * total + offset..j * total + offset + width]
                .copy_from_slice(src);
        }
        offset += width;
    }
    let mut offset = 0usize;
    let blocks: Vec<RowDomainGateBlock<'_>> = domains
        .iter()
        .zip(&exceptions)
        .zip(&widths)
        .map(|((gates, exceptions), &width)| {
            let start = offset;
            offset += width;
            RowDomainGateBlock {
                columns: start..offset,
                gates,
                exceptions,
            }
        })
        .collect();

    for dir in [LaneDir::Lower, LaneDir::Upper] {
        let independent: Vec<_> = (0..3)
            .map(|d| {
                eng.run(
                    &seeds[d],
                    Some(&domains[d]),
                    dir,
                    Some(&exceptions[d]),
                    false,
                )
                .expect("independent")
            })
            .collect();
        let stacked = eng
            .run_domain_stacked(
                &DomainStackSeed {
                    s: stacked_s.clone(),
                    e: Some(stacked_e.clone()),
                },
                &blocks,
                dir,
            )
            .expect("stacked");
        let mut offset = 0usize;
        for (d, (&width, one)) in widths.iter().zip(&independent).enumerate() {
            for i in 0..one.a.nrows() {
                for r in 0..width {
                    assert_eq!(
                        one.a[[i, r]].to_bits(),
                        stacked.a[[i, offset + r]].to_bits(),
                        "coefficient moved: dir={dir:?} domain={d} input={i} row={r}"
                    );
                    assert_eq!(
                        one.e.as_ref().expect("outward")[[i, r]].to_bits(),
                        stacked.e.as_ref().expect("outward")[[i, offset + r]].to_bits(),
                        "error coefficient moved: dir={dir:?} domain={d} input={i} row={r}"
                    );
                }
            }
            for r in 0..width {
                assert_eq!(one.b[r].to_bits(), stacked.b[offset + r].to_bits());
                assert_eq!(one.eb[r].to_bits(), stacked.eb[offset + r].to_bits());
            }
            let one_c = if matches!(dir, LaneDir::Lower) {
                eng.concretize_lower(one)
            } else {
                eng.concretize_upper(one)
            };
            let stacked_c = if matches!(dir, LaneDir::Lower) {
                eng.concretize_lower(&stacked)
            } else {
                eng.concretize_upper(&stacked)
            };
            for r in 0..width {
                assert_eq!(
                    one_c[r].to_bits(),
                    stacked_c[offset + r].to_bits(),
                    "concretized bound moved: dir={dir:?} domain={d} row={r}"
                );
            }
            offset += width;
        }
    }
}

#[test]
fn domain_stack_layout_and_exception_ownership_fail_closed() {
    let mut rng = StdRng::seed_from_u64(0xfa11_c105e);
    let spec = tiny_spec(&mut rng, 0.4);
    let net = TwinNet::compile(&spec).expect("compile");
    let root = RootGates::build(
        &net,
        &vec![-0.2; 72],
        &vec![0.2; 72],
        RoundMode::Outward,
        None,
    )
    .expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let seed = DomainStackSeed {
        s: Array2::zeros((net.n_y, 2)),
        e: Some(Array2::zeros((net.n_y, 2))),
    };
    let gates = domain_gates(&root, &[]);
    let no_exc = Exceptions::default();
    for columns in [1..2, 0..1, 0..0, 0..3] {
        let blocks = [RowDomainGateBlock {
            columns,
            gates: &gates,
            exceptions: &no_exc,
        }];
        assert!(
            eng.run_domain_stacked(&seed, &blocks, LaneDir::Lower)
                .is_err(),
            "malformed ownership layout was accepted"
        );
    }
    let (li, rec) = root
        .layers
        .iter()
        .enumerate()
        .find(|(_, rec)| !rec.unst.is_empty())
        .expect("unstable layer");
    let mut escaped = Exceptions::default();
    escaped.by_layer.entry(li).or_default().push(Exc {
        row: 1,
        neuron: rec.unst[0],
        a2: 0.0,
        s2: 0.0,
        c2: 0.0,
    });
    let blocks = [
        RowDomainGateBlock {
            columns: 0..1,
            gates: &gates,
            exceptions: &escaped,
        },
        RowDomainGateBlock {
            columns: 1..2,
            gates: &gates,
            exceptions: &no_exc,
        },
    ];
    assert!(
        eng.run_domain_stacked(&seed, &blocks, LaneDir::Lower)
            .is_err(),
        "exception escaped its owning domain span"
    );
}

/// (2)+(3) FRONTIER LANE == SERIAL ORACLE: verdict + root bound match, the
/// parallel run is deterministic, and a falsified instance is never UNSAT.
#[test]
fn parallel_frontier_matches_serial_and_moat_holds() {
    let mut rng = StdRng::seed_from_u64(202);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).expect("compile");
    // Closing instance (tiny box, argmax target).
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let (t, worst, adv) = tiny_targets(&net, &lo, &hi);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let cfg = |frontier: usize| BabConfig {
        lru_cap: 64,
        frontier,
        ..BabConfig::default()
    };
    let serial = MarginRowBab::run(&net, &root, t, &adv, cfg(1));
    let par = MarginRowBab::run(&net, &root, t, &adv, cfg(8));
    let (sv, srb, _se, _ss) = verdict_bits(&serial);
    let (pv, prb, _pe, _ps) = verdict_bits(&par);
    assert_eq!(sv, pv, "frontier verdict differs from serial oracle");
    assert_eq!(
        srb, prb,
        "root_bound not bit-identical (serial vs frontier)"
    );

    // MOAT: falsified orientation must NEVER be UNSAT under the parallel lane.
    // Bounded so it terminates (it can never close a leaf containing the SAT
    // point → Unknown by max_expansions).
    let moat_cfg = BabConfig {
        lru_cap: 64,
        frontier: 8,
        max_expansions: 400,
        ..BabConfig::default()
    };
    let adv_f: Vec<usize> = (0..4).filter(|&o| o != worst).collect();
    let moat = MarginRowBab::run(&net, &root, worst, &adv_f, moat_cfg);
    assert!(
        !matches!(moat, MarginRowOutcome::Unsat(_)),
        "MOAT VIOLATION: parallel lane certified a falsified instance UNSAT"
    );

    // DETERMINISM: two parallel runs on the branching (falsified) instance,
    // same budget, must produce identical stats — no float race, no order
    // nondeterminism in the verdict-bearing quantities.
    let det_cfg = || BabConfig {
        lru_cap: 64,
        frontier: 8,
        max_expansions: 300,
        ..BabConfig::default()
    };
    let a = verdict_bits(&MarginRowBab::run(&net, &root, worst, &adv_f, det_cfg()));
    let b = verdict_bits(&MarginRowBab::run(&net, &root, worst, &adv_f, det_cfg()));
    assert_eq!(a, b, "parallel frontier is non-deterministic across runs");
}

/// SCORE-CANDIDATE STACK DIFFERENTIAL: same frontier width, same pop order,
/// same budget; the only changed seam is independent vs cross-domain scoring.
/// The test-only oracle inside `score_candidates_domain_stacked` additionally
/// compares every child score's bits before any candidate is selected.
#[test]
fn domain_stacked_candidate_scores_match_independent_frontier_bit_exact() {
    let mut rng = StdRng::seed_from_u64(0x5c0e_cafe);
    let spec = tiny_spec(&mut rng, 0.48);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.25; 72];
    let hi = vec![0.25; 72];
    let (_t, worst, _) = tiny_targets(&net, &lo, &hi);
    let adv: Vec<usize> = (0..4).filter(|&o| o != worst).collect();
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let cfg = |domain_stack: bool| BabConfig {
        lru_cap: 64,
        frontier: 8,
        domain_stack,
        max_expansions: 80,
        ..BabConfig::default()
    };
    let independent = MarginRowBab::run(&net, &root, worst, &adv, cfg(false));
    let stacked = MarginRowBab::run(&net, &root, worst, &adv, cfg(true));
    let a = verdict_bits(&independent);
    let b = verdict_bits(&stacked);
    assert!(a.2 >= 3, "fixture did not reach a multi-domain frontier");
    assert_eq!(a, b, "domain-stacked scoring changed frontier outcome");
    let a = outcome_stats(&independent);
    let b = outcome_stats(&stacked);
    assert_eq!(a.domains_created, b.domains_created);
    assert_eq!(a.closed, b.closed);
    assert_eq!(a.max_depth, b.max_depth);
    assert_eq!(a.mono_raw_dips, b.mono_raw_dips);
    assert_eq!(a.mono_worst.to_bits(), b.mono_worst.to_bits());
    assert_eq!(a.ledger_ok, b.ledger_ok);
}

#[test]
fn production_performance_switches_default_on_and_zero_kills() {
    use super::net::{
        blocked_backward_enabled_from_env, root_gemm_backend_from_env, RootGemmBackend,
    };

    assert!(super::net::blas_conv_enabled_from_env(None));
    assert!(super::net::blas_conv_enabled_from_env(Some("1")));
    assert!(!super::net::blas_conv_enabled_from_env(Some("0")));
    assert!(!super::net::blas_conv_enabled_from_env(Some("off")));
    let platform_default = if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        RootGemmBackend::Faer
    } else {
        RootGemmBackend::Ndarray
    };
    assert_eq!(
        root_gemm_backend_from_env(None, None),
        Some(platform_default)
    );
    assert_eq!(
        root_gemm_backend_from_env(Some("1"), Some("ndarray")),
        Some(RootGemmBackend::Ndarray)
    );
    assert_eq!(
        root_gemm_backend_from_env(Some("1"), Some(" FaEr ")),
        Some(RootGemmBackend::Faer)
    );
    assert_eq!(
        root_gemm_backend_from_env(Some("1"), Some("")),
        Some(RootGemmBackend::Ndarray),
        "an explicit empty backend must fail safe to ndarray"
    );
    assert_eq!(
        root_gemm_backend_from_env(Some("1"), Some("unrecognized")),
        Some(RootGemmBackend::Ndarray),
        "unknown backends must fail safe to ndarray"
    );
    assert_eq!(
        root_gemm_backend_from_env(Some("0"), Some("faer")),
        None,
        "NY_ROOT_BLAS=0 must override the experimental backend"
    );

    assert!(blocked_backward_enabled_from_env(None));
    assert!(blocked_backward_enabled_from_env(Some("1")));
    assert!(blocked_backward_enabled_from_env(Some("true")));
    assert!(!blocked_backward_enabled_from_env(Some("0")));
    assert!(!blocked_backward_enabled_from_env(Some("false")));
    assert!(!blocked_backward_enabled_from_env(Some("off")));
    assert!(!blocked_backward_enabled_from_env(Some("no")));

    assert_eq!(super::margin_row_frontier_from_env(None, 12), 12);
    assert_eq!(super::margin_row_frontier_from_env(Some(""), 12), 1);
    assert_eq!(super::margin_row_frontier_from_env(Some("1"), 12), 12);
    assert_eq!(super::margin_row_frontier_from_env(Some("true"), 12), 12);
    assert_eq!(super::margin_row_frontier_from_env(Some("8"), 12), 8);
    assert_eq!(super::margin_row_frontier_from_env(Some("0"), 12), 1);
    assert_eq!(super::margin_row_frontier_from_env(Some("false"), 12), 1);
    assert_eq!(super::margin_row_frontier_from_env(Some("invalid"), 12), 1);

    assert!(!super::margin_row_domain_stack_from_env(None));
    assert!(!super::margin_row_domain_stack_from_env(Some("")));
    assert!(!super::margin_row_domain_stack_from_env(Some("0")));
    assert!(!super::margin_row_domain_stack_from_env(Some("true")));
    assert!(!super::margin_row_domain_stack_from_env(Some("invalid")));
    assert!(super::margin_row_domain_stack_from_env(Some("1")));
    assert!(super::margin_row_domain_stack_from_env(Some(" 1 ")));
}

// ===================== ENCLOSURE ORACLE: deep-conv near-zero-margin =========
// Adversarial soundness oracle for the backward-bias running-accumulator
// rounding (commit 5628ed6f). Targets the audited regime: a DEEP conv trunk
// with feature widths >> n_in, a LARGE running backward bias |b|~O(1e3..1e5)
// carried from the head, and MANY later wide conv layers with TINY biases
// whose per-step increment rounds AWAY against the large accumulator (each
// commits ~u*|b| real rounding that the pre-fix certified `eb` failed to
// carry). The certified per-class lower bound `dj[k] = root_eval.dj[k]` is
// checked against an INDEPENDENT double-double (compensated, ~1e-30 rel)
// forward over the RAW TwinSpec at the box center + corners + interior:
// soundness requires `dj[k] <= true_margin(x) + 1e-12` for every sampled x.
//
// The pre-fix code UNDER-widens `eb` on the tiny-increment conv steps
// (unfixed term ~ n_out*u*|bias*coef|, vs the real rounding ~u*|b|), so its
// `dj` overshoots the true margin -> a FALSE UNSAT this oracle catches. The
// fix carries 2u*|b_running| per step, restoring soundness. See the two
// counts printed by `oracle_deep_conv_bias_running_rounding`.

/// Double-double (two-f64) compensated arithmetic for the reference forward.
#[derive(Clone, Copy)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    fn from(x: f64) -> Self {
        Dd { hi: x, lo: 0.0 }
    }
    fn val(self) -> f64 {
        // hi already carries the rounded sum; hi+lo is the best f64.
        self.hi + self.lo
    }
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    (s, err)
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

#[inline]
fn dd_add(x: Dd, y: Dd) -> Dd {
    let (s, e) = two_sum(x.hi, y.hi);
    let e = e + x.lo + y.lo;
    let (h, l) = two_sum(s, e);
    Dd { hi: h, lo: l }
}

/// `acc += a*b` in double-double (a, b plain f64).
#[inline]
fn dd_fma(acc: Dd, a: f64, b: f64) -> Dd {
    let (p, e) = two_prod(a, b);
    let (s, e2) = two_sum(acc.hi, p);
    let e = e2 + e + acc.lo;
    let (h, l) = two_sum(s, e);
    Dd { hi: h, lo: l }
}

/// Independent compensated forward over the RAW TwinSpec at one point `x`
/// (length n_in). Returns the class scores (final Gemm outputs). Mirrors
/// `net::forward_points` semantics op-for-op but in double-double.
fn dd_forward_scores(spec: &TwinSpec, x: &[f64]) -> Vec<f64> {
    let mut tensors: Vec<Option<Vec<Dd>>> = vec![None; spec.ops.len() + 1];
    tensors[0] = Some(x.iter().map(|&v| Dd::from(v)).collect());
    for (k, op) in spec.ops.iter().enumerate() {
        let out: Vec<Dd> = match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let src = tensors[*input].as_ref().expect("topo");
                let (co, ci, kh, kw) = *kernel;
                let (_ic, ih, iw) = *ishape;
                let (_oc, oh, ow) = *oshape;
                let (sh, sw) = *stride;
                let (pt, pl, _pb, _pr) = *pads;
                let mut out = vec![Dd::from(0.0); co * oh * ow];
                for oc in 0..co {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = Dd::from(bias[oc]);
                            for c in 0..ci {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oy * sh + ky) as isize - pt as isize;
                                        let ix = (ox * sw + kx) as isize - pl as isize;
                                        if iy >= 0
                                            && (iy as usize) < ih
                                            && ix >= 0
                                            && (ix as usize) < iw
                                        {
                                            let w = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            let sidx =
                                                c * ih * iw + (iy as usize) * iw + ix as usize;
                                            acc = dd_fma(acc, w, src[sidx].val());
                                        }
                                    }
                                }
                            }
                            out[(oc * oh + oy) * ow + ox] = acc;
                        }
                    }
                }
                out
            }
            TwinOpSpec::ConvTranspose {
                input,
                weight,
                bias,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let src = tensors[*input].as_ref().expect("topo");
                let (co, ci, kh, kw) = *kernel;
                let (_ic, ih, iw) = *ishape;
                let (_oc, oh, ow) = *oshape;
                let (sh, sw) = *stride;
                let (pt, pl, _pb, _pr) = *pads;
                let mut out = vec![Dd::from(0.0); co * oh * ow];
                for (oc, chunk) in out.chunks_mut(oh * ow).enumerate() {
                    for v in chunk.iter_mut() {
                        *v = Dd::from(bias[oc]);
                    }
                }
                for c in 0..ci {
                    for iy in 0..ih {
                        for ix in 0..iw {
                            let xv = src[c * ih * iw + iy * iw + ix].val();
                            for oc in 0..co {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let ny = iy * sh + ky;
                                        let nx = ix * sw + kx;
                                        if ny < pt || nx < pl {
                                            continue;
                                        }
                                        let (oy, ox) = (ny - pt, nx - pl);
                                        if oy < oh && ox < ow {
                                            let w = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            let oi = (oc * oh + oy) * ow + ox;
                                            out[oi] = dd_fma(out[oi], w, xv);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                out
            }
            TwinOpSpec::ChannelAffine {
                input,
                scale,
                shift,
                shape,
                ..
            } => {
                let hw = shape.1 * shape.2;
                tensors[*input]
                    .as_ref()
                    .expect("topo")
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| dd_fma(Dd::from(shift[i / hw]), scale[i / hw], d.val()))
                    .collect()
            }
            TwinOpSpec::Relu { input } => tensors[*input]
                .as_ref()
                .expect("topo")
                .iter()
                .map(|&d| if d.val() > 0.0 { d } else { Dd::from(0.0) })
                .collect(),
            TwinOpSpec::Add { lhs, rhs } => {
                let a = tensors[*lhs].as_ref().expect("topo");
                let b = tensors[*rhs].as_ref().expect("topo");
                a.iter().zip(b).map(|(&p, &q)| dd_add(p, q)).collect()
            }
            TwinOpSpec::Flatten { input } => tensors[*input].as_ref().expect("topo").clone(),
            TwinOpSpec::Gemm {
                input,
                weight,
                bias,
                shape,
            } => {
                let src = tensors[*input].as_ref().expect("topo");
                let (no, ni) = *shape;
                let mut out = vec![Dd::from(0.0); no];
                for o in 0..no {
                    let mut acc = Dd::from(bias[o]);
                    for i in 0..ni {
                        acc = dd_fma(acc, weight[o * ni + i], src[i].val());
                    }
                    out[o] = acc;
                }
                out
            }
        };
        tensors[k + 1] = Some(out);
    }
    tensors[spec.ops.len()]
        .as_ref()
        .expect("final")
        .iter()
        .map(|d| d.val())
        .collect()
}

/// Compensated margin `Y_t - Y_j` per adv class at point `x` (double-double
/// difference of the two class scores, then flattened to f64).
fn dd_margins(spec: &TwinSpec, x: &[f64], t: usize, adv: &[usize]) -> Vec<f64> {
    let scores = dd_forward_scores(spec, x);
    // Recompute t - j as a compensated difference for the last cancellation.
    adv.iter()
        .map(|&j| {
            let d = dd_add(Dd::from(scores[t]), Dd::from(-scores[j]));
            d.val()
        })
        .collect()
}

/// Build the adversarial deep-conv net (SIGN-DEFINITE, near-head ReLU-bias).
///
/// The layout is chosen so that the WHOLE input->margin coefficient chain is
/// sign-definite (all weights are POSITIVE and every ReLU is stably ACTIVE),
/// which makes the tiny-bias backward increments accumulate SYSTEMATICALLY
/// (no random cancellation) — the only way the ACTUAL rounding error reaches
/// the worst-case bound the fix carries and the pre-fix code omits:
///
/// * `k_tiny` wide convs consume the input (feature width `cw*hs*ws >> n_in`)
///   with TINY negative biases. Their backward increment `bias*coef` is below
///   `ulp(|b|)/2` and ROUNDS AWAY against the large running seed bias, so the
///   computed backward bias misses their (systematically negative) true margin
///   contribution.
/// * `conv_relu` sits just before the trunk ReLU (near the head, so its
///   backward coefficient — and hence its legitimate `eb` term — is SMALL) and
///   carries a moderate positive bias so the trunk ReLU is stably active.
/// * `gemm1` carries a LARGE positive bias -> large seed running bias `b` and
///   a stably-active head ReLU.
/// * `gemm2` row `t` dominates (large positive), the others are ~0, so the
///   margins track `Y_t` and the coefficient signs stay definite.
struct AdvCfg {
    cin: usize,
    hs: usize,
    ws: usize,
    cw: usize,
    k_tiny: usize,
    tiny_bias: f64,
    relu_bias: f64,
    gemm1_bias: f64,
    n_y: usize,
    n_out: usize,
    /// Conv-kernel weight scale (POSITIVE weights in `[0.25,1]*scale`).
    conv_wscale: f64,
    /// gemm1 weight scale (tiny -> negligible input sensitivity; the large
    /// gemm1 BIAS still drives the seed running bias).
    gemm1_wscale: f64,
    /// gemm2 dominant-row (class `t`=0) weight scale.
    gemm2_wscale: f64,
    /// gemm2 bias for class `t` (calibrated to `-L` so the true margins land
    /// within ~1e-4 of 0 -> the pre-fix `dj` overshoot lands PAST 0, a literal
    /// false UNSAT under the strict `b>0` closure).
    t_bias: f64,
    seed: u64,
}

fn adversarial_spec(cfg: &AdvCfg) -> TwinSpec {
    let (cin, hs, ws, cw) = (cfg.cin, cfg.hs, cfg.ws, cfg.cw);
    let n_in = cin * hs * ws;
    let spatial = hs * ws;
    // CONSTANT positive weights + 1x1 kernels: every neuron in a layer has the
    // IDENTICAL backward coefficient (no random spread, no padding edge
    // effects), so the calibrated tiny-bias increment `tiny_bias * lr` is the
    // same for all neurons -> all round away together -> the dropped increments
    // accumulate SYSTEMATICALLY (no cancellation). Gains are ~unit (contractive
    // enough) so the certified bound does not blow up.
    let mut ops = Vec::new();
    // k_tiny 1x1 convs consuming the input, TINY negative biases.
    for i in 0..cfg.k_tiny {
        let (input, in_c, ishape) = if i == 0 {
            (0usize, cin, (cin, hs, ws))
        } else {
            (i, cw, (cw, hs, ws))
        };
        ops.push(TwinOpSpec::Conv {
            input,
            weight: vec![cfg.conv_wscale; cw * in_c],
            bias: vec![cfg.tiny_bias; cw],
            bias_err: vec![0.0; cw],
            weight_rel_err: 0.0,
            kernel: (cw, in_c, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape,
            oshape: (cw, hs, ws),
        });
    }
    // conv_relu: moderate positive bias -> trunk ReLU stably active. Near the
    // head, so its backward coefficient (and legitimate eb) is small.
    let relu_conv_in = cfg.k_tiny; // tensor id of last tiny conv output
    ops.push(TwinOpSpec::Conv {
        input: relu_conv_in,
        weight: vec![cfg.conv_wscale; cw * cw],
        bias: vec![cfg.relu_bias; cw],
        bias_err: vec![0.0; cw],
        weight_rel_err: 0.0,
        kernel: (cw, cw, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (cw, hs, ws),
        oshape: (cw, hs, ws),
    });
    let relu_conv_out = relu_conv_in + 1;
    // relu0: trunk relu, stably active.
    ops.push(TwinOpSpec::Relu {
        input: relu_conv_out,
    });
    let relu0_out = relu_conv_out + 1;
    ops.push(TwinOpSpec::Flatten { input: relu0_out });
    let flat_out = relu0_out + 1;
    let n_h = cw * spatial;
    // gemm1: large positive bias -> large seed running bias + active head relu.
    ops.push(TwinOpSpec::Gemm {
        input: flat_out,
        weight: vec![cfg.gemm1_wscale; cfg.n_y * n_h],
        bias: vec![cfg.gemm1_bias; cfg.n_y],
        shape: (cfg.n_y, n_h),
    });
    let gemm1_out = flat_out + 1;
    ops.push(TwinOpSpec::Relu { input: gemm1_out });
    let head_relu_out = gemm1_out + 1;
    // gemm2: dominant positive constant row for class t=0, exactly 0 for the
    // others, so every margin tracks Y_t (coefficient signs stay definite) and
    // the classes differ only by their bias constant.
    let mut w2 = vec![0.0f64; cfg.n_out * cfg.n_y];
    for k in 0..cfg.n_y {
        w2[k] = cfg.gemm2_wscale; // row 0 = t
    }
    // Class t=0 carries the -L offset (margins near 0); the others small.
    #[allow(clippy::cast_precision_loss)]
    let b2: Vec<f64> = (0..cfg.n_out)
        .map(|o| {
            if o == 0 {
                cfg.t_bias
            } else {
                1.0e-6 * (o as f64)
            }
        })
        .collect();
    ops.push(TwinOpSpec::Gemm {
        input: head_relu_out,
        weight: w2,
        bias: b2,
        shape: (cfg.n_out, cfg.n_y),
    });
    TwinSpec { n_in, ops }
}

/// Result of one oracle evaluation.
struct OracleOut {
    violations: usize,
    min_true: f64,
    max_dj: f64,
    /// max over (class, point) of `dj[k] - true_margin(x)`.
    worst_over: f64,
    /// max |running backward bias| in the margin-seeded direct pass.
    max_running_b: f64,
    /// max certified bias error `eb` in the margin-seeded direct pass.
    max_eb: f64,
    /// max signed rounding error (computed affine const - exact) at center.
    max_err: f64,
    /// certified dj per adv class.
    dj: Vec<f64>,
}

/// Count `dj[k] > true_margin(x) + 1e-12` violations across a sample set.
///
/// `n_corner`/`n_interior` control the sampled point count (the double-double
/// forward is the cost; a tight box needs few points because the margin is
/// near-constant over the box).
fn oracle_violations(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    rng: &mut StdRng,
    n_corner: usize,
    n_interior: usize,
) -> OracleOut {
    let net = TwinNet::compile(spec).expect("compile");
    let root = RootGates::build(&net, lo, hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let re = super::bab::root_eval(&eng, &net, t, adv).expect("root_eval");
    let dj = re.dj;
    let n_in = spec.n_in;

    // Diagnostic: the margin-seeded direct backward pass exposes the running
    // bias `b` whose accumulation rounding is the bug's subject.
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let mb = MarginBatch::new(&net, t, adv).expect("mb");
    let gates = head_gates(&ybox, RoundMode::Outward);
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Outward);
    let pass = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .expect("pass");
    let max_running_b = pass.b.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let max_eb = pass.eb.iter().fold(0.0f64, |m, &v| m.max(v.abs()));

    let center: Vec<f64> = (0..n_in).map(|i| f64::midpoint(lo[i], hi[i])).collect();
    // Diagnostic: actual signed rounding error of the computed affine constant
    // (pass.b + cst) vs the double-double-EXACT margin at the box center
    // (mid=0 so A@mid=0). A POSITIVE err means the computed backward bias
    // OVERSHOOTS the truth -> the direction that makes an under-widened `eb`
    // produce a false UNSAT.
    let dd_center = dd_margins(spec, &center, t, adv);
    let mut max_err = f64::NEG_INFINITY;
    for k in 0..adv.len() {
        let computed_affine = pass.b[k] + ms.cst[k];
        max_err = max_err.max(computed_affine - dd_center[k]);
    }
    let mut pts: Vec<Vec<f64>> = vec![center, lo.to_vec(), hi.to_vec()];
    for _ in 0..n_corner {
        pts.push(
            (0..n_in)
                .map(|i| {
                    if rng.random_range(0.0..1.0) < 0.5 {
                        lo[i]
                    } else {
                        hi[i]
                    }
                })
                .collect(),
        );
    }
    for _ in 0..n_interior {
        pts.push(
            (0..n_in)
                .map(|i| lo[i] + rng.random_range(0.0..1.0) * (hi[i] - lo[i]))
                .collect(),
        );
    }
    let mut out = OracleOut {
        violations: 0,
        min_true: f64::INFINITY,
        max_dj: f64::NEG_INFINITY,
        worst_over: f64::NEG_INFINITY,
        max_running_b,
        max_eb,
        max_err,
        dj: dj.clone(),
    };
    for x in &pts {
        let m = dd_margins(spec, x, t, adv);
        for k in 0..adv.len() {
            out.min_true = out.min_true.min(m[k]);
            out.max_dj = out.max_dj.max(dj[k]);
            out.worst_over = out.worst_over.max(dj[k] - m[k]);
            if dj[k] > m[k] + 1e-12 {
                out.violations += 1;
            }
        }
    }
    out
}

/// Measure, for a given net, the running backward bias magnitude `L` and the
/// per-class TOTAL tiny-bias sensitivity `Sigma_lr = d(margin)/d(uniform tiny
/// conv bias)` (the exact coefficient sum the backward accumulation rounds).
/// The margin is affine in the tiny bias (stable ReLUs), so a finite
/// difference recovers it exactly. `Sigma_lr` is a property of the WEIGHTS
/// only (independent of the tiny-bias value), so the caller can rebuild the
/// spec with a calibrated tiny bias without changing it.
fn measure_l_and_sigma_lr(cfg: &AdvCfg, t: usize, adv: &[usize]) -> (f64, Vec<f64>) {
    // Running bias L from a zero-input, tiny-box pass.
    let spec0 = adversarial_spec(cfg);
    let net = TwinNet::compile(&spec0).expect("compile");
    let n_in = spec0.n_in;
    let lo = vec![-1.0e-9; n_in];
    let hi = vec![1.0e-9; n_in];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let mb = MarginBatch::new(&net, t, adv).expect("mb");
    let gates = head_gates(&ybox, RoundMode::Outward);
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Outward);
    let pass = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .expect("pass");
    let l = pass.b.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    // Sigma_lr per class via a double-double finite difference at the origin.
    let center = vec![0.0f64; n_in];
    let mut cfg_a = adv_cfg_clone(cfg);
    cfg_a.tiny_bias = 0.0;
    let mut cfg_b = adv_cfg_clone(cfg);
    let db = 1.0e-3;
    cfg_b.tiny_bias = db;
    let m_a = dd_margins(&adversarial_spec(&cfg_a), &center, t, adv);
    let m_b = dd_margins(&adversarial_spec(&cfg_b), &center, t, adv);
    let sigma: Vec<f64> = m_a.iter().zip(&m_b).map(|(&a, &b)| (b - a) / db).collect();
    (l, sigma)
}

fn adv_cfg_clone(cfg: &AdvCfg) -> AdvCfg {
    AdvCfg {
        cin: cfg.cin,
        hs: cfg.hs,
        ws: cfg.ws,
        cw: cfg.cw,
        k_tiny: cfg.k_tiny,
        tiny_bias: cfg.tiny_bias,
        relu_bias: cfg.relu_bias,
        gemm1_bias: cfg.gemm1_bias,
        n_y: cfg.n_y,
        n_out: cfg.n_out,
        conv_wscale: cfg.conv_wscale,
        gemm1_wscale: cfg.gemm1_wscale,
        gemm2_wscale: cfg.gemm2_wscale,
        t_bias: cfg.t_bias,
        seed: cfg.seed,
    }
}

/// THE DISCRIMINATING ENCLOSURE ORACLE. On HEAD (with commit 5628ed6f) this
/// records ZERO violations; with the two `2.0*UNIT*b[ri].abs()` fix terms
/// removed the same net produces a FALSE UNSAT (`dj > true_margin`) that this
/// test catches. Deterministic (seeded RNG, no wall-clock).
#[test]
fn oracle_deep_conv_bias_running_rounding() {
    // DEEP wide conv trunk: n_in small (small concretize gamma floor
    // (n_in+16)*u*|b|), feature width `cw*hs*ws >> n_in`, many tiny-bias conv
    // steps so the bias-accumulation rounding (~ u*|b_running| per step, over
    // >> n_in steps) is the dominant slack in the certified bound.
    let mut cfg = AdvCfg {
        cin: 3,
        hs: 4,
        ws: 4,
        cw: 256,
        k_tiny: 12,
        tiny_bias: 0.0, // calibrated below
        relu_bias: 5.0,
        gemm1_bias: 3.0e6,
        n_y: 24,
        n_out: 10,
        // constant 1x1 conv weight = 1/fan_in -> ~unit backward gain, so the
        // per-neuron coefficient is uniform across ALL depths (no explosion,
        // no decay): the calibrated increment rounds away at EVERY tiny neuron.
        conv_wscale: 1.0 / 256.0,
        gemm1_wscale: 1.0e-4,
        gemm2_wscale: 0.5,
        t_bias: 0.0, // calibrated below to -L
        seed: 0xA5EE_D001,
    };
    let t = 0usize;
    let n_out = cfg.n_out;
    let adv_all: Vec<usize> = (1..n_out).collect();
    // Calibrate the tiny bias so each backward increment `tiny_bias * lr_j`
    // sits at ~a quarter of ulp(L)/2: below the round-away threshold (so the
    // increment is DROPPED from the computed backward bias) yet large enough
    // that the dropped increments accumulate SYSTEMATICALLY into a drift the
    // pre-fix `eb` cannot cover. Uses the dominant-sensitivity class.
    let (l, sigma) = measure_l_and_sigma_lr(&cfg, t, &adv_all);
    let width = cfg.cw * cfg.hs * cfg.ws;
    let n_tiny_neurons = (cfg.k_tiny * width) as f64;
    let dom_sigma = sigma.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
    let lr_avg = dom_sigma / n_tiny_neurons; // avg |lr| over tiny neurons
    let ulp_half = 0.5 * (super::rounding::next_up(l) - l); // ulp(L)/2
                                                            // per-step increment target = 0.25 * ulp(L)/2 -> rounds away.
    let tiny_bias = -(0.25 * ulp_half) / lr_avg;
    cfg.tiny_bias = tiny_bias;
    // Offset class t so the true margins land within ~1e-4 of 0: the pre-fix
    // `dj` overshoot then lands PAST 0 (a literal false UNSAT). L=pass.b is
    // independent of this bias, so it does not perturb the calibration.
    cfg.t_bias = -l;
    eprintln!(
        "[enclosure-oracle] CALIBRATION L={l:.3e} dom_sigma_lr={dom_sigma:.3e} \
         lr_avg={lr_avg:.3e} ulp_half={ulp_half:.3e} -> tiny_bias={tiny_bias:.3e} \
         t_bias={:.3e}",
        cfg.t_bias
    );
    let spec = adversarial_spec(&cfg);
    let net = TwinNet::compile(&spec).expect("compile");
    // TINY box centered at ORIGIN: mid = 0 so `A@mid = 0` and (with contractive
    // weights) `rr = sum|A|*rad` is negligible; the ONLY meaningful slack in the
    // certified bound is the bias-accumulation rounding.
    let n_in = spec.n_in;
    let rad = 1.0e-9;
    let lo = vec![-rad; n_in];
    let hi = vec![rad; n_in];
    let adv = adv_all;
    let mut rng = StdRng::seed_from_u64(0x0BAD_C0DE);
    let o = oracle_violations(&spec, &lo, &hi, t, &adv, &mut rng, 24, 12);
    let _ = &net;
    let depth = spec.ops.len();
    let width = cfg.cw * cfg.hs * cfg.ws;
    let tiny_steps = cfg.k_tiny * width; // conv-bias-loop accumulation steps
    let dj_str: Vec<String> = o.dj.iter().map(|v| format!("{v:.4e}")).collect();
    eprintln!(
        "[enclosure-oracle] violations={} min_true_margin={:.6e} max_dj={:.6e} \
         worst_overshoot(dj-true)={:.3e} max_running_bias={:.3e} max_eb={:.3e} \
         max_actual_err={:.3e} \
         trunk_depth_ops={depth} feature_width={width} tiny_bias_steps={tiny_steps} \
         n_in={n_in} gemm1_bias={:.1e} tiny_bias={:.1e}\n[enclosure-oracle] dj=[{}]",
        o.violations,
        o.min_true,
        o.max_dj,
        o.worst_over,
        o.max_running_b,
        o.max_eb,
        o.max_err,
        cfg.gemm1_bias,
        cfg.tiny_bias,
        dj_str.join(", ")
    );
    assert_eq!(
        o.violations, 0,
        "ENCLOSURE ORACLE: {} certified dj[k] exceed the true margin \
         (worst overshoot {:.3e}) -> false UNSAT (backward-bias \
         running-accumulator rounding undercounted)",
        o.violations, o.worst_over
    );
}

// ============ FORWARD-TABLEAU ENCLOSURE ORACLE (root.rs) ====================
// Adversarial soundness oracle for the FORWARD ROOT-TABLEAU coefficient
// rounding and its CROSS-LAYER accumulation (root.rs `RootGates::build` ->
// `apply_gates` -> `concretize_box`). Distinct from the backward-bias oracle
// above: this targets the frozen root gates themselves.
//
// MECHANISM UNDER TEST. The twin tableau carries DeepPoly lower/upper linear
// rows as `A_l = M - D`, `A_u = M + D` over augmented input coords. Per trunk
// relu, `apply_gates` forms `lo_c = (M-D)*alpha`, `up_c = (M+D)*s + c`, then
// `M' = (lo_c+up_c)/2`, `D' = (up_c-lo_c)/2` and WIDENS `D'` by
// `~6u*(|M'|+D')` (root.rs l.482). Widening `D'` UP pushes the lower row
// `M'-D'` DOWN coefficient-wise. Over a SIGNED input box with midpoint ~0 the
// concretized lower `sum_i (a_i*mid_i - |a_i|*rad_i) = -sum_i |a_i|*rad_i`
// is NOT monotone in `a_i`: pushing a POSITIVE lower-row coefficient DOWN
// (toward 0) makes `|a_i|` SMALLER and hence the concretized min LARGER — a
// per-layer OVERSHOOT of the true pre-activation lower bound. The ONLY
// compensator is the next `concretize_box`'s `gamma_n(n_in+16)` slack
// (root.rs l.349/380). The audited worry (workflow-1 "uncertain"): on
// DEEP / residual / cancellation nets the per-layer overshoot may ACCUMULATE
// across layers faster than a single per-concretize `+16u` absorbs.
//
// ORACLE. Build DEEP 1x1-conv ("dense") trunks with SIGNED SYMMETRIC input
// boxes (`mid = 0`, the worst geometry for the widening-down overshoot),
// weights engineered for catastrophic cross-layer cancellation (so `D`
// approaches `|M|` — 100% relative error, where the overshoot `~2*min(|M|,D)`
// is maximal), tuned so the true margin sits within ~1e-10 of 0. Two rigorous,
// INDEPENDENT double-double checks (compensated, ~1e-30 rel), for every
// sampled `x` (corners + center + interior + gradient-guided worst-case):
//
//   (1) BOX ENCLOSURE (direct probe of the tableau): every frozen root box
//       must enclose the EXACT pre-activation — `lg.l[j] <= preact_j(x)` and
//       `lg.u[j] >= preact_j(x)`. A violation `lg.l[j] > min_x preact_j(x)`
//       is the raw forward-tableau overshoot (a false enclosure -> wrong gate
//       -> false UNSAT).
//   (2) MARGIN (task-required end-to-end): `dj[k] = root_eval.dj[k]` must
//       satisfy `dj[k] <= true_margin(x) + 1e-12`.
//
// Sampling gives an UPPER bound on the true min, so ANY exceedance is a REAL
// bug (the true min is at most the sampled value). Deterministic seeded RNG.

/// DD forward returning EVERY tensor (topo order): `t[0]` = input, `t[k+1]` =
/// output of op `k`. Op semantics mirror `net::forward_points` /
/// `dd_forward_scores` exactly, in double-double.
fn dd_forward_all(spec: &TwinSpec, x: &[f64]) -> Vec<Vec<Dd>> {
    let mut t: Vec<Vec<Dd>> = Vec::with_capacity(spec.ops.len() + 1);
    t.push(x.iter().map(|&v| Dd::from(v)).collect());
    for op in &spec.ops {
        let out: Vec<Dd> = match op {
            TwinOpSpec::Conv {
                input,
                weight,
                bias,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let src = &t[*input];
                let (co, ci, kh, kw) = *kernel;
                let (_ic, ih, iw) = *ishape;
                let (_oc, oh, ow) = *oshape;
                let (sh, sw) = *stride;
                let (pt, pl, _pb, _pr) = *pads;
                let mut out = vec![Dd::from(0.0); co * oh * ow];
                for oc in 0..co {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = Dd::from(bias[oc]);
                            for c in 0..ci {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oy * sh + ky) as isize - pt as isize;
                                        let ix = (ox * sw + kx) as isize - pl as isize;
                                        if iy >= 0
                                            && (iy as usize) < ih
                                            && ix >= 0
                                            && (ix as usize) < iw
                                        {
                                            let w = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            let sidx =
                                                c * ih * iw + (iy as usize) * iw + ix as usize;
                                            acc = dd_fma(acc, w, src[sidx].val());
                                        }
                                    }
                                }
                            }
                            out[(oc * oh + oy) * ow + ox] = acc;
                        }
                    }
                }
                out
            }
            TwinOpSpec::ConvTranspose {
                input,
                weight,
                bias,
                kernel,
                stride,
                pads,
                ishape,
                oshape,
                ..
            } => {
                let src = &t[*input];
                let (co, ci, kh, kw) = *kernel;
                let (_ic, ih, iw) = *ishape;
                let (_oc, oh, ow) = *oshape;
                let (sh, sw) = *stride;
                let (pt, pl, _pb, _pr) = *pads;
                let mut out = vec![Dd::from(0.0); co * oh * ow];
                for (oc, chunk) in out.chunks_mut(oh * ow).enumerate() {
                    for v in chunk.iter_mut() {
                        *v = Dd::from(bias[oc]);
                    }
                }
                for c in 0..ci {
                    for iy in 0..ih {
                        for ix in 0..iw {
                            let xv = src[c * ih * iw + iy * iw + ix].val();
                            for oc in 0..co {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let ny = iy * sh + ky;
                                        let nx = ix * sw + kx;
                                        if ny < pt || nx < pl {
                                            continue;
                                        }
                                        let (oy, ox) = (ny - pt, nx - pl);
                                        if oy < oh && ox < ow {
                                            let w = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            let oi = (oc * oh + oy) * ow + ox;
                                            out[oi] = dd_fma(out[oi], w, xv);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                out
            }
            TwinOpSpec::ChannelAffine {
                input,
                scale,
                shift,
                shape,
                ..
            } => {
                let hw = shape.1 * shape.2;
                t[*input]
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| dd_fma(Dd::from(shift[i / hw]), scale[i / hw], d.val()))
                    .collect()
            }
            TwinOpSpec::Relu { input } => t[*input]
                .iter()
                .map(|&d| if d.val() > 0.0 { d } else { Dd::from(0.0) })
                .collect(),
            TwinOpSpec::Add { lhs, rhs } => t[*lhs]
                .iter()
                .zip(&t[*rhs])
                .map(|(&p, &q)| dd_add(p, q))
                .collect(),
            TwinOpSpec::Flatten { input } => t[*input].clone(),
            TwinOpSpec::Gemm {
                input,
                weight,
                bias,
                shape,
            } => {
                let src = &t[*input];
                let (no, ni) = *shape;
                let mut out = vec![Dd::from(0.0); no];
                for o in 0..no {
                    let mut acc = Dd::from(bias[o]);
                    for i in 0..ni {
                        acc = dd_fma(acc, weight[o * ni + i], src[i].val());
                    }
                    out[o] = acc;
                }
                out
            }
        };
        t.push(out);
    }
    t
}

/// Cross-layer cancellation weight schemes for the deep tableau trunk.
#[derive(Clone, Copy, Debug)]
enum WScheme {
    /// Uniform signed `[-s, s]`.
    RandSigned,
    /// Uniform signed then row-mean-subtracted (zero row sum): a constant-ish
    /// activation folds to ~0, so the composed input->neuron coefficient
    /// cancels hard while the abs-path (the `D` lane) keeps growing.
    ZeroRowSum,
    /// Structured `+/- s` (alternating columns): deterministic sign
    /// cancellation, uniform per-neuron magnitude.
    AltSign,
}

/// Deep forward-tableau adversary.
struct TabCfg {
    n_in: usize,
    width: usize,
    depth: usize,
    n_y: usize,
    n_out: usize,
    wscale: f64,
    scheme: WScheme,
    /// Per-conv bias magnitude (signed): small keeps relus UNSTABLE over the
    /// symmetric box; the relaxation (`s`,`c`) then engages `apply_gates`.
    bias_scale: f64,
    seed: u64,
}

/// Build a DEEP `[Conv1x1 -> ReLU]^depth -> Flatten -> Gemm1 -> ReLU -> Gemm2`
/// net over a `(n_in,1,1)` input (1x1 convs = dense layers; `n_in` channels),
/// with cancellation-engineered trunk weights. `weight_rel_err`/`bias_err` are
/// 0 (no BN) so the ONLY slack in the certified box is the tableau rounding
/// under test (nothing masks an overshoot).
fn wgen(rng: &mut StdRng, scheme: WScheme, s: f64, rows: usize, cols: usize) -> Vec<f64> {
    let mut w = vec![0.0f64; rows * cols];
    for r in 0..rows {
        match scheme {
            WScheme::RandSigned => {
                for c in 0..cols {
                    w[r * cols + c] = rng.random_range(-s..s);
                }
            }
            WScheme::ZeroRowSum => {
                let mut mean = 0.0;
                for c in 0..cols {
                    let v = rng.random_range(-s..s);
                    w[r * cols + c] = v;
                    mean += v;
                }
                mean /= cols as f64;
                for c in 0..cols {
                    w[r * cols + c] -= mean;
                }
            }
            WScheme::AltSign => {
                for c in 0..cols {
                    let sign = if (r + c) % 2 == 0 { 1.0 } else { -1.0 };
                    // small jitter breaks exact structural degeneracy
                    w[r * cols + c] = sign * s * (0.75 + 0.25 * rng.random_range(0.0..1.0));
                }
            }
        }
    }
    w
}

fn bgen(rng: &mut StdRng, bias_scale: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|_| rng.random_range(-bias_scale..bias_scale))
        .collect()
}

fn tableau_spec(cfg: &TabCfg) -> TwinSpec {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let s = cfg.wscale;
    let bs = cfg.bias_scale;
    let sc = cfg.scheme;
    let (cin, cw) = (cfg.n_in, cfg.width);
    let mut ops = Vec::new();
    for i in 0..cfg.depth {
        let (input, in_c) = if i == 0 { (0usize, cin) } else { (i, cw) };
        ops.push(TwinOpSpec::Conv {
            input,
            weight: wgen(&mut rng, sc, s, cw, in_c),
            bias: bgen(&mut rng, bs, cw),
            bias_err: vec![0.0; cw],
            weight_rel_err: 0.0,
            kernel: (cw, in_c, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (in_c, 1, 1),
            oshape: (cw, 1, 1),
        });
        ops.push(TwinOpSpec::Relu { input: i + 1 });
    }
    let flat_in = 2 * cfg.depth; // last relu output tensor id
    ops.push(TwinOpSpec::Flatten { input: flat_in });
    let flat_out = flat_in + 1;
    ops.push(TwinOpSpec::Gemm {
        input: flat_out,
        weight: wgen(&mut rng, sc, s, cfg.n_y, cw),
        bias: bgen(&mut rng, bs, cfg.n_y),
        shape: (cfg.n_y, cw),
    });
    let g1_out = flat_out + 1;
    ops.push(TwinOpSpec::Relu { input: g1_out });
    let hr_out = g1_out + 1;
    ops.push(TwinOpSpec::Gemm {
        input: hr_out,
        weight: wgen(&mut rng, sc, s, cfg.n_out, cfg.n_y),
        bias: bgen(&mut rng, bs, cfg.n_out),
        shape: (cfg.n_out, cfg.n_y),
    });
    TwinSpec { n_in: cin, ops }
}

/// Gradient-guided worst-case corner search on a cheap f64 scalar objective
/// (drives the sampled min toward the true argmin of a CPWL function): from a
/// start, finite-diff the gradient, jump each coord to the box end that lowers
/// the local linear model, re-evaluate, iterate to a fixed point. Returns the
/// best (lowest-objective) point seen.
fn grad_min<F: Fn(&[f64]) -> f64>(
    lo: &[f64],
    hi: &[f64],
    start: &[f64],
    f: &F,
    iters: usize,
) -> Vec<f64> {
    let n = lo.len();
    let mut x = start.to_vec();
    let mut best = x.clone();
    let mut best_v = f(&x);
    for _ in 0..iters {
        let base = f(&x);
        let mut nx = x.clone();
        for i in 0..n {
            let h = ((hi[i] - lo[i]) * 1e-4).max(1e-9);
            let mut xp = x.clone();
            xp[i] += h;
            let g = (f(&xp) - base) / h;
            nx[i] = if g > 0.0 { lo[i] } else { hi[i] };
        }
        let v = f(&nx);
        if v < best_v {
            best_v = v;
            best = nx.clone();
        }
        if nx == x {
            break;
        }
        x = nx;
    }
    best
}

/// One config's result.
struct TabOut {
    /// Box-enclosure violations `lg.l[j] > preact_j(x)` or `lg.u[j] < preact`.
    box_viol: usize,
    /// Margin violations `dj[k] > true_margin(x) + 1e-12`.
    dj_viol: usize,
    /// max over (layer, neuron, x) of `lg.l[j] - preact_j(x)` (box overshoot).
    worst_box_over: f64,
    /// max over (class, x) of `dj[k] - true_margin(x)`.
    worst_dj_over: f64,
    /// deepest-layer max |D|/|M| coefficient ratio (cancellation strength).
    max_dm_ratio: f64,
    n_checks: usize,
}

/// Run the forward-tableau oracle for one config over a signed symmetric box.
fn tableau_oracle(cfg: &TabCfg, rad: f64, n_rand_corner: usize, n_interior: usize) -> TabOut {
    let base_spec = tableau_spec(cfg);
    let net = TwinNet::compile(&base_spec).expect("compile");
    let n_in = cfg.n_in;
    let lo = vec![-rad; n_in];
    let hi = vec![rad; n_in];
    let t = 0usize;
    let adv: Vec<usize> = (1..cfg.n_out).collect();

    // Build the candidate point set (shared by both checks). Full corner
    // enumeration when small; else random corners. Plus center + interior +
    // gradient-guided worst-case corners for the margin objective.
    let mut pts: Vec<Vec<f64>> = Vec::new();
    pts.push(vec![0.0; n_in]); // center (mid = 0)
    if n_in <= 10 {
        // Full corner enumeration (<= 1024 corners): exhaustive over the box
        // vertices where a CPWL box-min is most often attained.
        for mask in 0u32..(1u32 << n_in) {
            pts.push(
                (0..n_in)
                    .map(|i| if mask & (1 << i) != 0 { hi[i] } else { lo[i] })
                    .collect(),
            );
        }
    } else {
        let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0xC0DE);
        for _ in 0..n_rand_corner {
            pts.push(
                (0..n_in)
                    .map(|i| if rng.random_bool(0.5) { hi[i] } else { lo[i] })
                    .collect(),
            );
        }
    }
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0xBEEF);
    for _ in 0..n_interior {
        pts.push(
            (0..n_in)
                .map(|i| lo[i] + rng.random_range(0.0..1.0) * (hi[i] - lo[i]))
                .collect(),
        );
    }
    // Gradient-guided worst-case corners for EACH adv-class margin (drives the
    // sampled margin toward its true box-min). Cheap f64 objective via the net.
    let f64_margin = |x: &[f64], k: usize| -> f64 {
        let xa = Array2::from_shape_fn((n_in, 1), |(i, _)| x[i]);
        let (y, _) = net
            .forward_points(&xa, &std::collections::BTreeMap::new())
            .expect("fwd");
        let m = margins_at(&net, &y, t, &adv);
        m[k][0]
    };
    for k in 0..adv.len() {
        for st in 0..4usize {
            let start: Vec<f64> = (0..n_in)
                .map(|i| if (st + i) % 2 == 0 { lo[i] } else { hi[i] })
                .collect();
            let obj = |x: &[f64]| f64_margin(x, k);
            pts.push(grad_min(&lo, &hi, &start, &obj, n_in + 4));
        }
    }

    use rayon::prelude::*;

    // --- Calibrate class-t gemm2 bias so the TRUE min margin ~ +1e-11 ---
    // Every margin shifts by +b2[t]. A cheap f64 min over the sample set fixes
    // where the min lands (calibration precision is not verdict-critical; the
    // actual soundness comparison below is double-double).
    let min_base = {
        let xa = Array2::from_shape_fn((n_in, pts.len()), |(i, b)| pts[b][i]);
        let (y, _) = net
            .forward_points(&xa, &std::collections::BTreeMap::new())
            .expect("fwd");
        margins_at(&net, &y, t, &adv)
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min)
    };
    let shift = -min_base + 1e-11; // additive constant applied to class t
    let mut spec = base_spec.clone();
    if let Some(TwinOpSpec::Gemm { bias, .. }) = spec.ops.last_mut() {
        bias[t] += shift; // true min margin -> ~ +1e-11
    }

    // --- Frozen root tableau + boxes + dj ---
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    // NOTE: root_eval/gates depend only on the TRUNK tableau + gemm structure;
    // the class-t bias shift lands in gemm2 (past the last trunk relu), so it
    // moves dj and the true margin by the SAME additive constant. Use the
    // calibrated spec's net for dj so both sides shift together.
    let net_cal = TwinNet::compile(&spec).expect("compile-cal");
    let root_cal =
        RootGates::build(&net_cal, &lo, &hi, RoundMode::Outward, None).expect("root-cal");
    let eng_cal = BackwardEngine::new(&net_cal, &root_cal);
    let re = super::bab::root_eval(&eng_cal, &net_cal, t, &adv).expect("root_eval");
    let dj = re.dj;

    // Deepest-layer coefficient cancellation ratio (diagnostic on the tableau
    // strength): rebuild is internal, so approximate via the box width vs
    // midpoint magnitude at the last trunk relu.
    let mut max_dm_ratio = 0.0f64;
    if let Some(lg) = root.layers.last() {
        for j in 0..lg.n {
            let mmag = 0.5 * (lg.u[j] + lg.l[j]).abs();
            let dmag = 0.5 * (lg.u[j] - lg.l[j]).abs();
            if mmag > 0.0 {
                max_dm_ratio = max_dm_ratio.max(dmag / mmag);
            }
        }
    }

    // Which tensor id feeds each trunk relu (for exact pre-activations).
    let relu_input: Vec<(usize, usize)> = root
        .layers
        .iter()
        .map(|lg| {
            let inp = match &base_spec.ops[lg.op] {
                TwinOpSpec::Relu { input } => *input,
                _ => unreachable!("trunk layer op is a relu"),
            };
            (lg.op, inp)
        })
        .collect();

    // Parallel over sample points. ONE double-double forward per point yields
    // BOTH the exact pre-activations (box check) and the exact class scores
    // (margin/dj check). The class-t calibration is an additive constant
    // `shift` (applied in dd), so no second forward is needed.
    let fold = |mut a: TabOut, b: TabOut| -> TabOut {
        a.box_viol += b.box_viol;
        a.dj_viol += b.dj_viol;
        a.worst_box_over = a.worst_box_over.max(b.worst_box_over);
        a.worst_dj_over = a.worst_dj_over.max(b.worst_dj_over);
        a.n_checks += b.n_checks;
        a.max_dm_ratio = a.max_dm_ratio.max(b.max_dm_ratio);
        a
    };
    let ident = || TabOut {
        box_viol: 0,
        dj_viol: 0,
        worst_box_over: f64::NEG_INFINITY,
        worst_dj_over: f64::NEG_INFINITY,
        max_dm_ratio: 0.0,
        n_checks: 0,
    };
    let mut out = pts
        .par_iter()
        .map(|x| {
            let mut o = ident();
            let tens = dd_forward_all(&base_spec, x);
            // (1) box enclosure vs EXACT pre-activations (trunk tensors are
            // identical for base and calibrated specs).
            for (li, lg) in root.layers.iter().enumerate() {
                let pre = &tens[relu_input[li].1];
                for j in 0..lg.n {
                    let p = pre[j].val();
                    o.worst_box_over = o.worst_box_over.max(lg.l[j] - p);
                    o.n_checks += 1;
                    if lg.l[j] > p || lg.u[j] < p {
                        o.box_viol += 1;
                    }
                }
            }
            // (2) margin vs dj: calibrated margin = (score_t - score_j) + shift
            // (all in double-double), then compared to the certified dj.
            let scores = tens.last().expect("final");
            for (k, &j) in adv.iter().enumerate() {
                let base_m = dd_add(
                    scores[t],
                    Dd {
                        hi: -scores[j].hi,
                        lo: -scores[j].lo,
                    },
                );
                let m = dd_add(base_m, Dd::from(shift)).val();
                o.worst_dj_over = o.worst_dj_over.max(dj[k] - m);
                if dj[k] > m + 1e-12 {
                    o.dj_viol += 1;
                }
            }
            o
        })
        .reduce(ident, fold);
    out.max_dm_ratio = max_dm_ratio;
    out
}

/// THE FORWARD-TABLEAU DISCRIMINATING ENCLOSURE ORACLE. Sweeps depth, width,
/// n_in, and cancellation scheme over SIGNED SYMMETRIC boxes, checking both the
/// raw root-tableau box enclosure and the end-to-end `dj` margin bound against
/// a double-double reference. ANY violation = the `+16u` per-concretize
/// headroom is INSUFFICIENT to absorb the cross-layer `apply_gates` widening
/// overshoot (a root-tableau false-UNSAT). Deterministic (seeded RNG).
#[test]
fn oracle_forward_tableau_cross_layer_rounding() {
    let schemes = [WScheme::RandSigned, WScheme::ZeroRowSum, WScheme::AltSign];
    // Depth/width/n_in sweep. Small n_in => full 2^n corner enumeration.
    // wscale ~ 2/width keeps the abs-path gain ~1 (bounded magnitudes) while
    // signed cancellation shrinks the composed coefficient => D/|M| climbs with
    // depth toward the ~100%-relative-error regime where the widening-down
    // overshoot is maximal.
    let configs: Vec<(usize, usize, usize, f64)> = vec![
        // (n_in, width, depth, rad)
        (6, 32, 8, 1.0),
        (6, 64, 16, 1.0),
        (8, 48, 12, 10.0),
        (8, 96, 24, 1.0),
        (10, 64, 20, 100.0),
        (10, 128, 30, 1.0),
        (12, 96, 24, 10.0),
        (4, 64, 40, 1.0),   // extra-deep, tiny n_in (full 16-corner enum)
        (16, 64, 20, 10.0), // random-corner regime
    ];
    let mut total_box_viol = 0usize;
    let mut total_dj_viol = 0usize;
    let mut worst_box = f64::NEG_INFINITY;
    let mut worst_dj = f64::NEG_INFINITY;
    let mut worst_box_regime = String::new();
    let mut worst_dj_regime = String::new();
    let mut total_checks = 0usize;
    let mut seed_ctr: u64 = 0x7AB1_EA00;
    for scheme in schemes {
        for &(n_in, width, depth, rad) in &configs {
            seed_ctr = seed_ctr.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let cfg = TabCfg {
                n_in,
                width,
                depth,
                n_y: 16,
                n_out: 6,
                wscale: 2.0 / width as f64,
                scheme,
                bias_scale: rad * 0.5, // keep many trunk relus UNSTABLE
                seed: seed_ctr,
            };
            let o = tableau_oracle(&cfg, rad, 2048, 64);
            total_box_viol += o.box_viol;
            total_dj_viol += o.dj_viol;
            total_checks += o.n_checks;
            let regime = format!(
                "{scheme:?} n_in={n_in} width={width} depth={depth} rad={rad} \
                 max_D/M={:.2e}",
                o.max_dm_ratio
            );
            if o.worst_box_over > worst_box {
                worst_box = o.worst_box_over;
                worst_box_regime = regime.clone();
            }
            if o.worst_dj_over > worst_dj {
                worst_dj = o.worst_dj_over;
                worst_dj_regime = regime.clone();
            }
            eprintln!(
                "[fwd-tableau] {regime} box_viol={} dj_viol={} \
                 worst_box_over={:.3e} worst_dj_over={:.3e} checks={}",
                o.box_viol, o.dj_viol, o.worst_box_over, o.worst_dj_over, o.n_checks
            );
        }
    }
    eprintln!(
        "[fwd-tableau] SUMMARY total_checks={total_checks} box_viol={total_box_viol} \
         dj_viol={total_dj_viol}\n[fwd-tableau]   worst_box_overshoot={worst_box:.3e} @ {worst_box_regime}\
         \n[fwd-tableau]   worst_dj_overshoot={worst_dj:.3e} @ {worst_dj_regime}"
    );
    assert_eq!(
        total_box_viol, 0,
        "FWD-TABLEAU ORACLE: {total_box_viol} root-tableau box endpoints fail to \
         enclose the exact pre-activation (worst overshoot {worst_box:.3e} @ \
         {worst_box_regime}) -> cross-layer apply_gates widening overshoot \
         EXCEEDS the +16u concretize headroom (false enclosure / false UNSAT)"
    );
    assert_eq!(
        total_dj_viol, 0,
        "FWD-TABLEAU ORACLE: {total_dj_viol} certified dj[k] exceed the true margin \
         (worst overshoot {worst_dj:.3e} @ {worst_dj_regime}) -> root-tableau false UNSAT"
    );
}

// ============ RESIDUAL-RISK ENCLOSURE ORACLES (final completeness gate) ======
// The two oracles above target the two SPECIFIC obligations the fixes closed
// (backward-bias running-accumulator rounding; forward-tableau cross-layer
// widening). This block hunts the regimes those oracles do NOT exercise — the
// residual risk surface flagged by the completeness gate — with the SAME rigor:
// the certified per-class lower bound `dj[k] = root_eval.dj[k]` (root_eval,
// bab.rs) is checked against an INDEPENDENT double-double (compensated, ~1e-30
// rel) reference over a sampled set (center + corners + interior + gradient-
// guided worst-case). Sampling gives an UPPER bound on `min_x (Y_t - Y_j)`, so
// ANY `dj[k] > true_margin(x) + 1e-12` is a REAL false-UNSAT (records net+seed).
//
// Regimes:
//   (1) NO-CONV relu chains (relu -> Add/Flatten -> relu, no intervening conv):
//       the exact leak the depth-scaled `8*trunk_relus` concretize headroom
//       targets — deep, signed symmetric boxes, near-zero margins.
//   (2) Conv+BN folded paths with extreme scale/var (non-zero weight_rel_err /
//       bias_err error budgets, extreme-magnitude folded weights).
//   (3) Residual Add-heavy nets (many skip connections) with cross-branch
//       cancellation (root.rs Add-lane `da+db+2u|mo|` widening under depth).
//   (4) The Gemm2 head / final margin composition with LARGE heads (big n_y,
//       gamma_n(2*n_y+8) row envelope in compose_viay).
//   (5) NaN/Inf & subnormal & extreme-magnitude weights: the fail-closed
//       firewall must reject non-finite params / Inf tableaux and NEVER emit a
//       finite-but-unsound bound (a +Inf dj that would close a class is caught
//       by the same `dj > true_margin` check).

/// Aggregate result of a `dj` vs true-margin enclosure sweep.
struct DjEncl {
    /// `dj[k] > true_margin(x) + 1e-12` count over the sample set.
    dj_viol: usize,
    /// max over (class, point) of `dj[k] - true_margin(x)` (overshoot).
    worst_over: f64,
    /// min sampled true margin (upper bound on the box-min).
    min_true: f64,
    /// max finite certified `dj`.
    max_dj: f64,
    /// checks performed (points * classes).
    n_checks: usize,
    /// root build / root_eval returned Err (fail-closed to Unknown).
    failed_closed: bool,
    /// certified dj per adv class (empty when failed_closed at build).
    dj: Vec<f64>,
}

impl DjEncl {
    fn closed(dj: Vec<f64>) -> Self {
        DjEncl {
            dj_viol: 0,
            worst_over: f64::NEG_INFINITY,
            min_true: f64::INFINITY,
            max_dj: f64::NEG_INFINITY,
            n_checks: 0,
            failed_closed: true,
            dj,
        }
    }
}

/// Build the shared candidate point set: center + lo/hi + corners (full 2^n
/// enumeration when `n_in <= 12`, else `n_corner` random corners) + `n_interior`
/// interior points + gradient-guided worst-case corners per adv class (drives
/// the sampled margin toward its true box-min for a CPWL net). Deterministic.
fn encl_points(
    net: &TwinNet,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    n_corner: usize,
    n_interior: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let n_in = lo.len();
    let mut pts: Vec<Vec<f64>> = Vec::new();
    pts.push((0..n_in).map(|i| f64::midpoint(lo[i], hi[i])).collect()); // center
    pts.push(lo.to_vec());
    pts.push(hi.to_vec());
    if n_in <= 12 {
        for mask in 0u32..(1u32 << n_in) {
            pts.push(
                (0..n_in)
                    .map(|i| if mask & (1 << i) != 0 { hi[i] } else { lo[i] })
                    .collect(),
            );
        }
    } else {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0DE);
        for _ in 0..n_corner {
            pts.push(
                (0..n_in)
                    .map(|i| if rng.random_bool(0.5) { hi[i] } else { lo[i] })
                    .collect(),
            );
        }
    }
    let mut rng = StdRng::seed_from_u64(seed ^ 0xBEEF);
    for _ in 0..n_interior {
        pts.push(
            (0..n_in)
                .map(|i| lo[i] + rng.random_range(0.0..1.0) * (hi[i] - lo[i]))
                .collect(),
        );
    }
    // Gradient-guided worst-case corners for EACH class margin (cheap f64 net).
    let f64_margin = |x: &[f64], k: usize| -> f64 {
        let xa = Array2::from_shape_fn((n_in, 1), |(i, _)| x[i]);
        let (y, _) = net
            .forward_points(&xa, &std::collections::BTreeMap::new())
            .expect("fwd");
        margins_at(net, &y, t, adv)[k][0]
    };
    for k in 0..adv.len() {
        for st in 0..4usize {
            let start: Vec<f64> = (0..n_in)
                .map(|i| if (st + i) % 2 == 0 { lo[i] } else { hi[i] })
                .collect();
            let obj = |x: &[f64]| f64_margin(x, k);
            pts.push(grad_min(lo, hi, &start, &obj, n_in + 4));
        }
    }
    pts
}

/// End-to-end enclosure oracle for an arbitrary twin spec. Optionally calibrates
/// the class-`t` Gemm2 bias so the sampled true min margin lands at ~ +1e-11
/// (the strict-`b>0` closure boundary — where any overshoot is a literal false
/// UNSAT), builds `dj` from `root_eval` (fail-closed tolerant), then checks
/// `dj[k] <= true_margin(x) + 1e-12` against the double-double reference over
/// the sample set. The calibration shift is a pure additive constant on class
/// `t` (lands in Gemm2, past the last trunk relu), so it moves `dj` and the true
/// margin by the SAME amount; the dd reference uses the base spec + shift.
fn dj_enclosure_run(
    base_spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    n_corner: usize,
    n_interior: usize,
    calibrate: bool,
    seed: u64,
) -> DjEncl {
    use rayon::prelude::*;
    let net = TwinNet::compile(base_spec).expect("compile");
    let pts = encl_points(&net, lo, hi, t, adv, n_corner, n_interior, seed);
    // Calibration: bring the sampled f64 min margin to ~ +1e-11 (precision here
    // is not verdict-critical; the soundness comparison below is double-double).
    let shift = if calibrate {
        let xa = Array2::from_shape_fn((base_spec.n_in, pts.len()), |(i, b)| pts[b][i]);
        let (y, _) = net
            .forward_points(&xa, &std::collections::BTreeMap::new())
            .expect("fwd");
        let min_base = margins_at(&net, &y, t, adv)
            .iter()
            .flatten()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if min_base.is_finite() {
            -min_base + 1e-11
        } else {
            0.0
        }
    } else {
        0.0
    };
    let mut spec = base_spec.clone();
    if calibrate && shift != 0.0 {
        if let Some(TwinOpSpec::Gemm { bias, .. }) = spec.ops.last_mut() {
            bias[t] += shift;
        }
    }
    // dj on the calibrated net — every stage fail-closed tolerant.
    let net_cal = match TwinNet::compile(&spec) {
        Ok(n) => n,
        Err(_) => return DjEncl::closed(Vec::new()),
    };
    let root = match RootGates::build(&net_cal, lo, hi, RoundMode::Outward, None) {
        Ok(r) => r,
        Err(_) => return DjEncl::closed(Vec::new()),
    };
    let eng = BackwardEngine::new(&net_cal, &root);
    let re = match super::bab::root_eval(&eng, &net_cal, t, adv) {
        Ok(r) => r,
        Err(_) => return DjEncl::closed(Vec::new()),
    };
    let dj = re.dj;
    let max_dj = dj
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    // Parallel enclosure check: one double-double forward per point.
    let (dj_viol, worst_over, min_true) = pts
        .par_iter()
        .map(|x| {
            let m = dd_margins(base_spec, x, t, adv);
            let mut v = 0usize;
            let mut wo = f64::NEG_INFINITY;
            let mut mt = f64::INFINITY;
            for k in 0..adv.len() {
                let mk = m[k] + shift;
                mt = mt.min(mk);
                wo = wo.max(dj[k] - mk);
                // dj[k] > true margin (incl. a +Inf dj that would wrongly close
                // the class) is a false UNSAT. NaN dj compares false -> safe
                // (fail-closed), so only the dangerous direction is flagged.
                if dj[k] > mk + 1e-12 {
                    v += 1;
                }
            }
            (v, wo, mt)
        })
        .reduce(
            || (0usize, f64::NEG_INFINITY, f64::INFINITY),
            |a, b| (a.0 + b.0, a.1.max(b.1), a.2.min(b.2)),
        );
    DjEncl {
        dj_viol,
        worst_over,
        min_true,
        max_dj,
        n_checks: pts.len() * adv.len(),
        failed_closed: false,
        dj,
    }
}

// ---- Regime (1): NO-CONV relu chains (Add/Flatten-separated) ---------------

/// Deep `Relu -> [Add(., input) -> Relu]^(depth-1) -> Flatten -> Gemm1 -> Relu
/// -> Gemm2` net with NO conv anywhere in the trunk. Re-adding the RAW signed
/// input after each relu keeps every trunk relu UNSTABLE (pre-act straddles 0)
/// while growing only LINEARLY in depth (no overflow). Each trunk relu is
/// separated from the next `concretize_box` by an Add ONLY — no intervening
/// conv g-term to re-absorb its `apply_gates` widening — so the `8*trunk_relus`
/// depth-scaled headroom is the SOLE compensator. Head Gemms carry signed
/// random weights (cancellation -> near-zero margins after calibration).
fn no_conv_chain_spec(n_in: usize, depth: usize, n_y: usize, n_out: usize, seed: u64) -> TwinSpec {
    assert!(depth >= 1);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ops: Vec<TwinOpSpec> = Vec::new();
    ops.push(TwinOpSpec::Relu { input: 0 }); // trunk relu 0
    let mut cur = ops.len(); // tensor id of last relu output
    for _ in 1..depth {
        ops.push(TwinOpSpec::Add { lhs: cur, rhs: 0 });
        let add_id = ops.len();
        ops.push(TwinOpSpec::Relu { input: add_id }); // trunk relu i
        cur = ops.len();
    }
    ops.push(TwinOpSpec::Flatten { input: cur });
    let flat_id = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input: flat_id,
        weight: wgen(&mut rng, WScheme::RandSigned, 0.7, n_y, n_in),
        bias: bgen(&mut rng, 0.5, n_y),
        shape: (n_y, n_in),
    });
    let g1_id = ops.len();
    ops.push(TwinOpSpec::Relu { input: g1_id });
    let hr_id = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input: hr_id,
        weight: wgen(&mut rng, WScheme::ZeroRowSum, 0.7, n_out, n_y),
        bias: bgen(&mut rng, 0.5, n_out),
        shape: (n_out, n_y),
    });
    TwinSpec { n_in, ops }
}

#[test]
fn oracle_no_conv_relu_chain_enclosure() {
    // (n_in <= 12 -> full 2^n corner enumeration), sweep trunk depth so the
    // 8*trunk_relus headroom term is stressed at DEEP relu counts with no conv
    // re-absorption between them. Signed symmetric box (mid = 0, worst geometry).
    let configs: &[(usize, usize, f64)] = &[
        // (n_in, trunk_relu_depth, rad). WIDE boxes stress the depth-scaled
        // 8*trunk_relus headroom under O(1) coefficient magnitudes; TINY boxes
        // (rad ~ 1e-7, mid=0) collapse the DeepPoly relaxation slack so `dj`
        // rides close to the true margin and the ACCUMULATED rounding becomes
        // the visible gap (near-boundary / discriminating regime).
        (8, 16, 1.0),
        (8, 40, 1.0),
        (8, 60, 4.0),
        (10, 32, 1.0),
        (6, 80, 1.0),
        (12, 24, 10.0),
        // tight-box discriminating variants (unstable relus, rounding-scale gap)
        (8, 40, 1.0e-7),
        (10, 60, 1.0e-6),
        (6, 100, 1.0e-7),
        (12, 30, 1.0e-8),
    ];
    let mut total_viol = 0usize;
    let mut total_checks = 0usize;
    let mut worst = f64::NEG_INFINITY;
    let mut worst_regime = String::new();
    let mut seed: u64 = 0x1CE0_C0FF;
    for &(n_in, depth, rad) in configs {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let n_out = 6usize;
        let spec = no_conv_chain_spec(n_in, depth, 20, n_out, seed);
        let lo = vec![-rad; n_in];
        let hi = vec![rad; n_in];
        let adv: Vec<usize> = (1..n_out).collect();
        let o = dj_enclosure_run(&spec, &lo, &hi, 0, &adv, 2048, 64, true, seed);
        assert!(!o.failed_closed, "no-conv chain should build (finite net)");
        total_viol += o.dj_viol;
        total_checks += o.n_checks;
        let regime = format!("n_in={n_in} trunk_relus={depth} rad={rad} seed={seed:#x}");
        if o.worst_over > worst {
            worst = o.worst_over;
            worst_regime = regime.clone();
        }
        eprintln!(
            "[no-conv-chain] {regime} dj_viol={} worst_over={:.3e} min_true={:.3e} \
             max_dj={:.3e} checks={}",
            o.dj_viol, o.worst_over, o.min_true, o.max_dj, o.n_checks
        );
    }
    eprintln!(
        "[no-conv-chain] SUMMARY checks={total_checks} viol={total_viol} \
         worst_over={worst:.3e} @ {worst_regime}"
    );
    assert_eq!(
        total_viol, 0,
        "NO-CONV CHAIN ORACLE: {total_viol} certified dj[k] exceed the true margin \
         (worst overshoot {worst:.3e} @ {worst_regime}) -> the Add/Flatten-separated \
         per-relu widening escaped the 8*trunk_relus depth-scaled headroom (false UNSAT)"
    );
}

// ---- Regime (2): Conv+BN folded paths, extreme scale/var -------------------

/// Conv+BN folded trunk: two 1x1 convs each carrying a NON-ZERO certified
/// `weight_rel_err` (BN kernel fold error) and per-channel `bias_err` (BN shift
/// fold error), with extreme-magnitude folded weights/biases (extreme BN
/// scale / small var). The double-double reference forwards the FOLDED f64 spec;
/// the certified engine additionally charges the error budgets (which only WIDEN
/// D -> shrink dj). This confirms the BN error plumbing is monotone-safe under
/// extreme magnitudes and never yields a finite-but-unsound bound.
fn conv_bn_spec(
    cin: usize,
    h: usize,
    w: usize,
    cw: usize,
    wscale: f64,
    rel_err: f64,
    bias_err: f64,
    n_y: usize,
    n_out: usize,
    seed: u64,
) -> TwinSpec {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_in = cin * h * w;
    let flat = cw * h * w;
    let ops: Vec<TwinOpSpec> = vec![
        // conv1 (BN folded): extreme scale, non-zero certified error budgets.
        TwinOpSpec::Conv {
            input: 0,
            weight: wgen(&mut rng, WScheme::RandSigned, wscale, cw, cin),
            bias: bgen(&mut rng, wscale, cw),
            bias_err: (0..cw).map(|_| bias_err).collect(),
            weight_rel_err: rel_err,
            kernel: (cw, cin, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (cin, h, w),
            oshape: (cw, h, w),
        },
        TwinOpSpec::Relu { input: 1 }, // trunk relu 0
        // conv2 (BN folded).
        TwinOpSpec::Conv {
            input: 2,
            weight: wgen(&mut rng, WScheme::ZeroRowSum, wscale, cw, cw),
            bias: bgen(&mut rng, wscale, cw),
            bias_err: (0..cw).map(|_| bias_err).collect(),
            weight_rel_err: rel_err,
            kernel: (cw, cw, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (cw, h, w),
            oshape: (cw, h, w),
        },
        TwinOpSpec::Relu { input: 3 }, // trunk relu 1
        TwinOpSpec::Flatten { input: 4 },
        TwinOpSpec::Gemm {
            input: 5,
            weight: wgen(&mut rng, WScheme::RandSigned, 0.3, n_y, flat),
            bias: bgen(&mut rng, 0.3, n_y),
            shape: (n_y, flat),
        },
        TwinOpSpec::Relu { input: 6 },
        TwinOpSpec::Gemm {
            input: 7,
            weight: wgen(&mut rng, WScheme::ZeroRowSum, 0.3, n_out, n_y),
            bias: bgen(&mut rng, 0.3, n_out),
            shape: (n_out, n_y),
        },
    ];
    TwinSpec { n_in, ops }
}

#[test]
fn oracle_conv_bn_extreme_scale_enclosure() {
    // (cin,h,w) chosen so n_in <= 12 (full corner enumeration). Sweep extreme
    // folded-weight scale and BN error budgets.
    let configs: &[(usize, usize, usize, usize, f64, f64, f64, f64)] = &[
        // (cin, h, w, cw, wscale, rel_err, bias_err, rad)
        (3, 2, 2, 8, 5.0, 1e-13, 1e-6, 1.0),
        (2, 2, 3, 8, 50.0, 1e-12, 1e-3, 1.0),
        (3, 2, 2, 12, 1.0e3, 1e-12, 1e-1, 0.5),
        (2, 3, 2, 10, 200.0, 1e-14, 1e-2, 2.0),
        (2, 2, 2, 16, 1.0e4, 1e-13, 1.0, 0.25),
        // tight-box variants: relaxation slack collapses so the BN error-budget
        // charge is the visible gap (near-boundary discrimination).
        (3, 2, 2, 8, 5.0, 1e-13, 1e-6, 1.0e-7),
        (2, 2, 3, 10, 20.0, 1e-12, 1e-4, 1.0e-8),
    ];
    let mut total_viol = 0usize;
    let mut total_checks = 0usize;
    let mut worst = f64::NEG_INFINITY;
    let mut worst_regime = String::new();
    let mut n_closed = 0usize;
    let mut seed: u64 = 0xB0BA_0001;
    for &(cin, h, w, cw, wscale, rel_err, bias_err, rad) in configs {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let n_out = 6usize;
        let spec = conv_bn_spec(cin, h, w, cw, wscale, rel_err, bias_err, 16, n_out, seed);
        let n_in = cin * h * w;
        let lo = vec![-rad; n_in];
        let hi = vec![rad; n_in];
        let adv: Vec<usize> = (1..n_out).collect();
        let o = dj_enclosure_run(&spec, &lo, &hi, 0, &adv, 2048, 64, true, seed);
        let regime = format!(
            "cin={cin} {h}x{w} cw={cw} wscale={wscale:.0e} rel_err={rel_err:.0e} \
             bias_err={bias_err:.0e} rad={rad} seed={seed:#x}"
        );
        if o.failed_closed {
            n_closed += 1;
            eprintln!("[conv-bn] {regime} FAILED-CLOSED (Unknown, sound)");
            continue;
        }
        total_viol += o.dj_viol;
        total_checks += o.n_checks;
        if o.worst_over > worst {
            worst = o.worst_over;
            worst_regime = regime.clone();
        }
        eprintln!(
            "[conv-bn] {regime} dj_viol={} worst_over={:.3e} min_true={:.3e} \
             max_dj={:.3e} checks={}",
            o.dj_viol, o.worst_over, o.min_true, o.max_dj, o.n_checks
        );
    }
    eprintln!(
        "[conv-bn] SUMMARY checks={total_checks} viol={total_viol} closed={n_closed} \
         worst_over={worst:.3e} @ {worst_regime}"
    );
    assert_eq!(
        total_viol, 0,
        "CONV-BN ORACLE: {total_viol} certified dj[k] exceed the true (folded-net) \
         margin (worst overshoot {worst:.3e} @ {worst_regime}) -> BN error-budget \
         accounting produced a finite-but-unsound bound (false UNSAT)"
    );
}

// ---- Regime (3): Residual Add-heavy nets with cancellation -----------------

/// Add-heavy residual net: a width-`cw` running tensor, then `nblocks` blocks
/// each adding a SIGNED (cancellation-engineered `ZeroRowSum`) 1x1-conv branch
/// into the running tensor before a trunk ReLU. Every Add re-introduces signed
/// values so the following relu straddles 0 (unstable), and the many skip
/// connections stress the root.rs Add lane (`da + db + 2u|mo|` widening) under
/// depth combined with the concretize headroom.
fn residual_add_spec(
    cin: usize,
    h: usize,
    w: usize,
    cw: usize,
    nblocks: usize,
    wscale: f64,
    n_y: usize,
    n_out: usize,
    seed: u64,
) -> TwinSpec {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_in = cin * h * w;
    let flat = cw * h * w;
    let mut ops: Vec<TwinOpSpec> = Vec::new();
    // Stem: map cin -> cw, then a trunk relu.
    ops.push(TwinOpSpec::Conv {
        input: 0,
        weight: wgen(&mut rng, WScheme::RandSigned, wscale, cw, cin),
        bias: bgen(&mut rng, wscale, cw),
        bias_err: vec![0.0; cw],
        weight_rel_err: 0.0,
        kernel: (cw, cin, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (cin, h, w),
        oshape: (cw, h, w),
    });
    ops.push(TwinOpSpec::Relu { input: 1 }); // running = trunk relu 0
    let mut running = ops.len(); // tensor id of running (non-neg) tensor
    for _ in 0..nblocks {
        // Signed conv branch off the running tensor.
        ops.push(TwinOpSpec::Conv {
            input: running,
            weight: wgen(&mut rng, WScheme::ZeroRowSum, wscale, cw, cw),
            bias: bgen(&mut rng, wscale, cw),
            bias_err: vec![0.0; cw],
            weight_rel_err: 0.0,
            kernel: (cw, cw, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (cw, h, w),
            oshape: (cw, h, w),
        });
        let branch = ops.len();
        ops.push(TwinOpSpec::Add {
            lhs: branch,
            rhs: running,
        }); // skip connection
        let add_id = ops.len();
        ops.push(TwinOpSpec::Relu { input: add_id }); // trunk relu
        running = ops.len();
    }
    ops.push(TwinOpSpec::Flatten { input: running });
    let flat_id = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input: flat_id,
        weight: wgen(&mut rng, WScheme::RandSigned, 0.3, n_y, flat),
        bias: bgen(&mut rng, 0.3, n_y),
        shape: (n_y, flat),
    });
    let g1 = ops.len();
    ops.push(TwinOpSpec::Relu { input: g1 });
    let hr = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input: hr,
        weight: wgen(&mut rng, WScheme::ZeroRowSum, 0.3, n_out, n_y),
        bias: bgen(&mut rng, 0.3, n_out),
        shape: (n_out, n_y),
    });
    TwinSpec { n_in, ops }
}

#[test]
fn oracle_residual_add_heavy_enclosure() {
    // n_in <= 12 for full corner enumeration; sweep skip-connection count and
    // width. wscale ~ 1/cw keeps the abs-path gain ~1 while ZeroRowSum branches
    // cancel hard across the many Adds.
    let configs: &[(usize, usize, usize, usize, usize, f64)] = &[
        // (cin, h, w, cw, nblocks, rad). Tight-box (rad ~ 1e-7) variants make the
        // Add-lane rounding accumulation the visible gap (near-boundary).
        (2, 2, 2, 6, 6, 1.0),
        (2, 2, 2, 8, 12, 1.0),
        (3, 2, 2, 6, 10, 2.0),
        (2, 2, 3, 8, 8, 1.0),
        (2, 2, 2, 10, 20, 0.5),
        (2, 2, 2, 8, 16, 1.0e-7),
        (3, 2, 2, 6, 24, 1.0e-7),
        (2, 2, 2, 10, 30, 1.0e-8),
    ];
    let mut total_viol = 0usize;
    let mut total_checks = 0usize;
    let mut worst = f64::NEG_INFINITY;
    let mut worst_regime = String::new();
    let mut seed: u64 = 0x5C1D_0001;
    for &(cin, h, w, cw, nblocks, rad) in configs {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let n_out = 6usize;
        let wscale = 2.0 / cw as f64;
        let spec = residual_add_spec(cin, h, w, cw, nblocks, wscale, 16, n_out, seed);
        let n_in = cin * h * w;
        let lo = vec![-rad; n_in];
        let hi = vec![rad; n_in];
        let adv: Vec<usize> = (1..n_out).collect();
        let o = dj_enclosure_run(&spec, &lo, &hi, 0, &adv, 2048, 64, true, seed);
        assert!(!o.failed_closed, "residual net should build (finite net)");
        total_viol += o.dj_viol;
        total_checks += o.n_checks;
        let regime = format!("cin={cin} {h}x{w} cw={cw} skips={nblocks} rad={rad} seed={seed:#x}");
        if o.worst_over > worst {
            worst = o.worst_over;
            worst_regime = regime.clone();
        }
        eprintln!(
            "[residual-add] {regime} dj_viol={} worst_over={:.3e} min_true={:.3e} \
             max_dj={:.3e} checks={}",
            o.dj_viol, o.worst_over, o.min_true, o.max_dj, o.n_checks
        );
    }
    eprintln!(
        "[residual-add] SUMMARY checks={total_checks} viol={total_viol} \
         worst_over={worst:.3e} @ {worst_regime}"
    );
    assert_eq!(
        total_viol, 0,
        "RESIDUAL-ADD ORACLE: {total_viol} certified dj[k] exceed the true margin \
         (worst overshoot {worst:.3e} @ {worst_regime}) -> Add-lane widening under \
         many skip connections escaped the certified envelope (false UNSAT)"
    );
}

// ---- Regime (4): Gemm2 head / large-head margin composition ----------------

/// Shallow trunk but LARGE head: `Gemm1` widens to a big `n_y`, `Gemm2` (n_out,
/// n_y) uses `ZeroRowSum` rows so each margin `Y_t - Y_j` is a huge cancelling
/// sum over `n_y` terms. This stresses `margin_seed` / `compose_viay` where the
/// row envelope `gamma_n(2*n_y + 8)` and the seed accumulation grow with head
/// width — the head-composition slack the trunk oracles do not reach.
fn large_head_spec(
    cin: usize,
    h: usize,
    w: usize,
    cw: usize,
    n_y: usize,
    n_out: usize,
    seed: u64,
) -> TwinSpec {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_in = cin * h * w;
    let flat = cw * h * w;
    let ops: Vec<TwinOpSpec> = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: wgen(&mut rng, WScheme::RandSigned, 1.0 / cin as f64, cw, cin),
            bias: bgen(&mut rng, 0.3, cw),
            bias_err: vec![0.0; cw],
            weight_rel_err: 0.0,
            kernel: (cw, cin, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (cin, h, w),
            oshape: (cw, h, w),
        },
        TwinOpSpec::Relu { input: 1 }, // trunk relu 0
        TwinOpSpec::Flatten { input: 2 },
        TwinOpSpec::Gemm {
            input: 3,
            weight: wgen(&mut rng, WScheme::RandSigned, 1.0, n_y, flat),
            bias: bgen(&mut rng, 1.0, n_y),
            shape: (n_y, flat),
        },
        TwinOpSpec::Relu { input: 4 },
        // Large head with hard cross-class cancellation over n_y terms.
        TwinOpSpec::Gemm {
            input: 5,
            weight: wgen(&mut rng, WScheme::ZeroRowSum, 1.0, n_out, n_y),
            bias: bgen(&mut rng, 1.0, n_out),
            shape: (n_out, n_y),
        },
    ];
    TwinSpec { n_in, ops }
}

#[test]
fn oracle_large_head_margin_composition_enclosure() {
    // Small n_in (full corner enum) but big head widths, so the head margin
    // composition — not the trunk — is the dominant certified-slack source.
    let configs: &[(usize, usize, usize, usize, usize, usize, f64)] = &[
        // (cin, h, w, cw, n_y, n_out, rad). Tight-box variants pin the head-y box
        // narrow so the gamma_n(2*n_y+8) row-envelope rounding is the visible gap.
        (2, 1, 3, 8, 128, 10, 1.0),
        (2, 2, 2, 8, 256, 16, 1.0),
        (3, 1, 2, 12, 384, 12, 2.0),
        (2, 2, 2, 6, 512, 20, 0.5),
        (2, 1, 2, 10, 200, 8, 4.0),
        (2, 2, 2, 8, 256, 16, 1.0e-7),
        (2, 2, 2, 6, 512, 20, 1.0e-8),
    ];
    let mut total_viol = 0usize;
    let mut total_checks = 0usize;
    let mut worst = f64::NEG_INFINITY;
    let mut worst_regime = String::new();
    let mut seed: u64 = 0x4EAD_0001;
    for &(cin, h, w, cw, n_y, n_out, rad) in configs {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let spec = large_head_spec(cin, h, w, cw, n_y, n_out, seed);
        let n_in = cin * h * w;
        let lo = vec![-rad; n_in];
        let hi = vec![rad; n_in];
        let adv: Vec<usize> = (1..n_out).collect();
        let o = dj_enclosure_run(&spec, &lo, &hi, 0, &adv, 2048, 64, true, seed);
        assert!(!o.failed_closed, "large-head net should build (finite net)");
        total_viol += o.dj_viol;
        total_checks += o.n_checks;
        let regime =
            format!("cin={cin} {h}x{w} cw={cw} n_y={n_y} n_out={n_out} rad={rad} seed={seed:#x}");
        if o.worst_over > worst {
            worst = o.worst_over;
            worst_regime = regime.clone();
        }
        eprintln!(
            "[large-head] {regime} dj_viol={} worst_over={:.3e} min_true={:.3e} \
             max_dj={:.3e} checks={}",
            o.dj_viol, o.worst_over, o.min_true, o.max_dj, o.n_checks
        );
    }
    eprintln!(
        "[large-head] SUMMARY checks={total_checks} viol={total_viol} \
         worst_over={worst:.3e} @ {worst_regime}"
    );
    assert_eq!(
        total_viol, 0,
        "LARGE-HEAD ORACLE: {total_viol} certified dj[k] exceed the true margin \
         (worst overshoot {worst:.3e} @ {worst_regime}) -> Gemm2 head margin \
         composition (gamma_n(2*n_y+8) row envelope) under-widened (false UNSAT)"
    );
}

// ---- f32 fast-path differential: f32-ON dj is LOOSER-or-equal to f64-OFF -----

/// `(root, dj)` for a spec, built with an EXPLICIT f32 override (bypassing the
/// env gate). `None` on fail-closed. Outward verdict mode.
fn root_dj_prec(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    t: usize,
    adv: &[usize],
    use_f32: bool,
) -> Option<(RootGates, Vec<f64>)> {
    let net = TwinNet::compile(spec).ok()?;
    let root = RootGates::build_prec(&net, lo, hi, RoundMode::Outward, None, use_f32).ok()?;
    let eng = BackwardEngine::new(&net, &root);
    let re = super::bab::root_eval(&eng, &net, t, adv).ok()?;
    Some((root, re.dj))
}

/// THE f32 FAST-PATH DIFFERENTIAL GATE. Across a wide net zoo (conv+BN,
/// residual-add, no-conv chains, large heads) and radii — including the
/// tight-box / hard-cancellation regimes where the f32 rounding is most visible
/// — the `NY_MARGIN_ROW_ROOT_F32` fast path is a PURE LOOSENING of the frozen
/// root gates: every f32 concretized box must CONTAIN the corresponding f64 box
/// (`l32 <= l64` and `u32 >= u64`). This is the moat-relevant invariant the
/// additive slack guarantees; a single containment breach would mean the slack
/// UNDER-covered the f32 rounding (a candidate false enclosure -> false UNSAT).
///
/// NOTE on `dj`: because the DeepPoly lower-line choice `alpha = [u >= -l]` is a
/// DISCRETE area heuristic, a (contained) wider box can flip `alpha` and nudge
/// the composed `dj` in EITHER direction — `dj_f32 > dj_f64` occurs but is still
/// SOUND (both bound the same true margin; the alpha=0/1 lines are both valid
/// lower relaxations). So `dj` monotonicity is REPORTED, not asserted; the
/// end-to-end soundness arbiter is the enclosure-oracle suite run under
/// `NY_MARGIN_ROW_ROOT_F32=1` (`dj <= true margin`). Box containment is the
/// invariant that is genuinely monotone and is what we hard-gate here.
#[test]
fn f32_fastpath_boxes_contain_f64_and_report_dj() {
    let mut box_checks = 0usize;
    let mut box_viol = 0usize;
    let mut worst_box_breach = f64::NEG_INFINITY;
    let mut worst_box_regime = String::new();
    let mut dj_checks = 0usize;
    let mut dj_tighter = 0usize; // dj_f32 > dj_f64 (sound, informational)
    let mut worst_dj_over = f64::NEG_INFINITY;
    let n_out = 6usize;
    let adv: Vec<usize> = (1..n_out).collect();
    let mut seed: u64 = 0xF32D_0001;
    let radii = [1.0f64, 10.0, 1.0e-3, 1.0e-7];
    for trial in 0..24usize {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        for &rad in &radii {
            let (spec, n_in): (TwinSpec, usize) = match trial % 4 {
                0 => {
                    let (cin, h, w, cw) = (3usize, 2usize, 2usize, 10usize);
                    (
                        conv_bn_spec(cin, h, w, cw, 20.0, 1e-13, 1e-4, 16, n_out, seed),
                        cin * h * w,
                    )
                }
                1 => {
                    let (cin, h, w, cw, nb) = (2usize, 2usize, 2usize, 8usize, 16usize);
                    (
                        residual_add_spec(cin, h, w, cw, nb, 2.0 / cw as f64, 16, n_out, seed),
                        cin * h * w,
                    )
                }
                2 => {
                    let (n_in, depth) = (8usize, 40usize);
                    (no_conv_chain_spec(n_in, depth, 20, n_out, seed), n_in)
                }
                _ => {
                    let (cin, h, w, cw, n_y) = (2usize, 2usize, 2usize, 8usize, 256usize);
                    (
                        large_head_spec(cin, h, w, cw, n_y, n_out, seed),
                        cin * h * w,
                    )
                }
            };
            let lo = vec![-rad; n_in];
            let hi = vec![rad; n_in];
            let r64 = root_dj_prec(&spec, &lo, &hi, 0, &adv, false);
            let r32 = root_dj_prec(&spec, &lo, &hi, 0, &adv, true);
            let (Some((root64, d64)), Some((root32, d32))) = (r64, r32) else {
                continue;
            };
            // (1) HARD: every f32 box must contain the f64 box, per layer/neuron.
            assert_eq!(root32.layers.len(), root64.layers.len());
            for (lg32, lg64) in root32.layers.iter().zip(&root64.layers) {
                for j in 0..lg64.n {
                    if !(lg32.l[j].is_finite()
                        && lg32.u[j].is_finite()
                        && lg64.l[j].is_finite()
                        && lg64.u[j].is_finite())
                    {
                        continue;
                    }
                    box_checks += 1;
                    // Relative slack for the two directed-round steps in concretize.
                    let eps = 8.0 * f64::EPSILON * (lg64.l[j].abs().max(lg64.u[j].abs()) + 1.0);
                    let lbreach = lg32.l[j] - lg64.l[j] - eps; // > 0 => f32 lower ABOVE f64 (bad)
                    let ubreach = lg64.u[j] - lg32.u[j] - eps; // > 0 => f32 upper BELOW f64 (bad)
                    let breach = lbreach.max(ubreach);
                    if breach > 0.0 {
                        box_viol += 1;
                        if breach > worst_box_breach {
                            worst_box_breach = breach;
                            worst_box_regime =
                                format!("trial={trial} kind={} rad={rad} j={j}", trial % 4);
                        }
                    }
                }
            }
            // (2) REPORT: dj monotonicity (informational; alpha-flip may break it).
            for k in 0..d64.len().min(d32.len()) {
                if d64[k].is_finite() && d32[k].is_finite() {
                    dj_checks += 1;
                    let over = d32[k] - d64[k];
                    if over > 1e-9 + 1e-9 * d64[k].abs() {
                        dj_tighter += 1;
                        worst_dj_over = worst_dj_over.max(over);
                    }
                }
            }
        }
    }
    eprintln!(
        "[f32-diff] box_checks={box_checks} box_viol={box_viol} worst_breach={worst_box_breach:.3e} @ {worst_box_regime}\n\
         [f32-diff] dj_checks={dj_checks} dj_tighter_than_f64={dj_tighter} (SOUND, alpha-heuristic) worst_dj_over={worst_dj_over:.3e}"
    );
    assert_eq!(
        box_viol, 0,
        "F32 FAST-PATH DIFFERENTIAL: {box_viol}/{box_checks} f32 boxes FAIL to contain the \
         f64 box (worst breach {worst_box_breach:.3e} @ {worst_box_regime}) -> the additive \
         slack UNDER-covered the f32 conv rounding (candidate false enclosure / false UNSAT)"
    );
}

// ---- Regime (5): NaN/Inf/subnormal/extreme -> fail-closed firewall ----------

#[test]
fn firewall_rejects_non_finite_parameters() {
    // compile() MUST reject non-finite conv/gemm weights and biases (never
    // silently propagate to a finite bound). bias_err is covered by
    // compile_rejects_invalid_conv_error_budgets; here: weights + biases.
    let mut rng = StdRng::seed_from_u64(0xF12E_0001);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        // conv weight[0]
        {
            let mut spec = tiny_spec(&mut rng, 0.3);
            if let TwinOpSpec::Conv { weight, .. } = &mut spec.ops[0] {
                weight[0] = bad;
            }
            assert!(
                TwinNet::compile(&spec).is_err(),
                "conv weight {bad:?} accepted"
            );
        }
        // conv bias[0]
        {
            let mut spec = tiny_spec(&mut rng, 0.3);
            if let TwinOpSpec::Conv { bias, .. } = &mut spec.ops[0] {
                bias[0] = bad;
            }
            assert!(
                TwinNet::compile(&spec).is_err(),
                "conv bias {bad:?} accepted"
            );
        }
        // gemm weight (last op)
        {
            let mut spec = tiny_spec(&mut rng, 0.3);
            if let Some(TwinOpSpec::Gemm { weight, .. }) = spec.ops.last_mut() {
                weight[0] = bad;
            }
            assert!(
                TwinNet::compile(&spec).is_err(),
                "gemm weight {bad:?} accepted"
            );
        }
        // gemm bias (last op)
        {
            let mut spec = tiny_spec(&mut rng, 0.3);
            if let Some(TwinOpSpec::Gemm { bias, .. }) = spec.ops.last_mut() {
                bias[0] = bad;
            }
            assert!(
                TwinNet::compile(&spec).is_err(),
                "gemm bias {bad:?} accepted"
            );
        }
    }
    // Non-finite INPUT box must also fail closed at RootGates::build.
    let spec = tiny_spec(&mut rng, 0.3);
    let net = TwinNet::compile(&spec).expect("compile");
    let mut lo = vec![-0.3; net.n_in];
    let hi = vec![0.3; net.n_in];
    lo[0] = f64::NAN;
    assert!(
        RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).is_err(),
        "non-finite input box accepted"
    );
    // Inverted box (lo > hi) must fail closed too.
    let mut lo2 = vec![-0.3; net.n_in];
    lo2[0] = 0.4;
    assert!(
        RootGates::build(&net, &lo2, &hi, RoundMode::Outward, None).is_err(),
        "inverted input box accepted"
    );
}

#[test]
fn firewall_subnormal_and_extreme_weights_never_unsound() {
    // Finite but pathological magnitudes (subnormal ~1e-320; extreme ~1e150 that
    // overflows the accumulation to +/-Inf). For EACH: either the lane
    // fail-closes (Err -> Unknown) OR every certified dj is finite AND sound
    // (dj <= true_margin + 1e-12). A +Inf dj that would close a class is caught
    // by the same enclosure check (Inf > finite margin). No calibration (the
    // magnitudes preclude a meaningful f64 min); the check holds unconditionally.
    let n_out = 6usize;
    let adv: Vec<usize> = (1..n_out).collect();
    let cases: &[(&str, f64, f64)] = &[
        // (label, conv/head weight scale, rad)
        ("subnormal", 5e-320, 1.0),
        ("subnormal-tiny-box", 1e-315, 1e-10),
        ("extreme-1e150", 1e150, 1.0),
        ("extreme-1e200", 1e200, 0.5),
        ("mixed-extreme", 1e120, 10.0),
    ];
    let mut n_closed = 0usize;
    let mut n_sound = 0usize;
    let mut total_viol = 0usize;
    let mut worst = f64::NEG_INFINITY;
    let mut worst_label = String::new();
    let mut seed: u64 = 0xEE7A_0001;
    for &(label, scale, rad) in cases {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        // A small conv trunk with the pathological magnitude threaded through
        // the conv AND the head, so both the tableau and the backward pass see
        // it. n_in = 8 (full corner enum for the sound branch).
        let mut rng = StdRng::seed_from_u64(seed);
        let (cin, h, w, cw) = (2usize, 2usize, 2usize, 6usize);
        let n_in = cin * h * w;
        let flat = cw * h * w;
        let spec = TwinSpec {
            n_in,
            ops: vec![
                TwinOpSpec::Conv {
                    input: 0,
                    weight: wgen(&mut rng, WScheme::RandSigned, scale, cw, cin),
                    bias: bgen(&mut rng, scale, cw),
                    bias_err: vec![0.0; cw],
                    weight_rel_err: 0.0,
                    kernel: (cw, cin, 1, 1),
                    stride: (1, 1),
                    pads: (0, 0, 0, 0),
                    ishape: (cin, h, w),
                    oshape: (cw, h, w),
                },
                TwinOpSpec::Relu { input: 1 },
                TwinOpSpec::Flatten { input: 2 },
                TwinOpSpec::Gemm {
                    input: 3,
                    weight: wgen(&mut rng, WScheme::RandSigned, scale, 12, flat),
                    bias: bgen(&mut rng, scale, 12),
                    shape: (12, flat),
                },
                TwinOpSpec::Relu { input: 4 },
                TwinOpSpec::Gemm {
                    input: 5,
                    weight: wgen(&mut rng, WScheme::RandSigned, scale, n_out, 12),
                    bias: bgen(&mut rng, scale, n_out),
                    shape: (n_out, 12),
                },
            ],
        };
        // Weights are FINITE (subnormal/extreme but not NaN/Inf) so compile
        // succeeds; the firewall must engage downstream if the tableau/pass
        // overflows.
        assert!(
            TwinNet::compile(&spec).is_ok(),
            "{label}: finite params should compile"
        );
        let lo = vec![-rad; n_in];
        let hi = vec![rad; n_in];
        let o = dj_enclosure_run(&spec, &lo, &hi, 0, &adv, 512, 32, false, seed);
        if o.failed_closed {
            n_closed += 1;
            eprintln!("[firewall] {label} scale={scale:.0e} rad={rad} FAILED-CLOSED (sound)");
            continue;
        }
        // Not closed -> every dj must be finite and sound.
        let all_finite = o.dj.iter().all(|v| v.is_finite());
        assert!(
            all_finite,
            "[firewall] {label}: lane returned a NON-FINITE dj without failing closed \
             (dj={:?}) — a +Inf dj can close a class and emit a false UNSAT",
            o.dj
        );
        n_sound += 1;
        total_viol += o.dj_viol;
        if o.worst_over > worst {
            worst = o.worst_over;
            worst_label = format!("{label} scale={scale:.0e} rad={rad} seed={seed:#x}");
        }
        eprintln!(
            "[firewall] {label} scale={scale:.0e} rad={rad} SOUND-FINITE dj_viol={} \
             worst_over={:.3e} min_true={:.3e} max_dj={:.3e} checks={}",
            o.dj_viol, o.worst_over, o.min_true, o.max_dj, o.n_checks
        );
    }
    eprintln!(
        "[firewall] SUMMARY closed={n_closed} sound_finite={n_sound} viol={total_viol} \
         worst_over={worst:.3e} @ {worst_label}"
    );
    assert_eq!(
        total_viol, 0,
        "FIREWALL ORACLE: {total_viol} certified dj[k] exceed the true margin under \
         subnormal/extreme weights (worst overshoot {worst:.3e} @ {worst_label}) -> a \
         finite-but-unsound bound escaped the fail-closed firewall (false UNSAT)"
    );
}

/// Exact-arithmetic backend parity gate for the experimental faer root GEMM.
/// The integer × dyadic inputs keep every product and sum exactly representable
/// in f64, so summation order cannot obscure layout, im2col, or scatter bugs.
#[test]
fn faer_root_gemm_matches_ndarray_on_exact_inputs() {
    use super::net::{conv_apply_forward_blas, conv_apply_forward_faer, ConvOp};

    let co = 3usize;
    let k = 4usize;
    let p = 2usize;
    let r = 513usize;
    let conv = ConvOp {
        input: 0,
        kernel: (co, k, 1, 1),
        ishape: (5, 1, 1),
        oshape: (co, 1, p),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        transposed: false,
        wmat: vec![
            1.0, -2.0, 3.0, 0.0, -4.0, 5.0, -6.0, 7.0, 2.0, 1.0, -3.0, 4.0,
        ],
        wt: Vec::new(),
        bias: vec![0.0; co],
        bias_err: vec![0.0; co],
        weight_rel_err: 0.0,
        // gather[t*p + sp], including one padding tap.
        gather: vec![0, 1, 1, 2, 2, usize::MAX, 3, 4],
        back_taps: Vec::new(),
        k_fwd: k,
        k_bwd: 0,
    };
    let src = Array2::from_shape_fn((5, r), |(row, col)| {
        let numerator = ((row * 11 + col % 17) as i32 - 8) as f64;
        numerator / 8.0
    });
    for &abs_w in &[false, true] {
        let mut ndarray = Array2::<f64>::zeros((co * p, r));
        let mut faer = Array2::<f64>::zeros((co * p, r));
        conv_apply_forward_blas(&conv, &src, &mut ndarray, abs_w, co, p, k, r);
        conv_apply_forward_faer(&conv, &src, &mut faer, abs_w, co, p, k, r);
        assert_eq!(
            faer, ndarray,
            "faer im2col/scatter parity failed (abs_w={abs_w})"
        );
    }
}

/// GEMM im2col conv soundness gate (#twinwall-blas, ndarray default; faer
/// opt-in for wide root tableaus; `NY_ROOT_BLAS=0` selects the scalar
/// fallback). The
/// DGEMM forward-conv reorders the `k`-term contraction; this asserts that both
/// the scalar path AND the BLAS path stay within the Higham `γ_{k}·S` envelope
/// of a Neumaier-compensated (≈exact) reference — the precise statement that
/// lets the twin-wall D-lane envelope (`γ_n(k_fwd+2..8)·S`, built in `root.rs`)
/// DOMINATE the reordered rounding, so the concretized box is an outward
/// enclosure of the exact real value under either path. Mirrors the
/// order-independence certificate `sound_f64_gemm`/`faer` already rely on for
/// verdict-grade CROWN backward. Padding taps and zero weights add exact `0`,
/// so the only difference between the paths is summation order.
#[test]
fn blas_conv_within_gamma_envelope_of_exact() {
    use super::net::{
        conv_apply_forward_blas, conv_apply_forward_faer, conv_forward_blocked, TwinNet, TwinOp,
    };
    use super::rounding::{gamma_n, UNIT};
    let mut rng = StdRng::seed_from_u64(0xB1A5_5EED);
    for trial in 0..24 {
        let scale = 0.4 + (trial % 4) as f64 * 0.4;
        let spec = tiny_spec(&mut rng, scale);
        let net = TwinNet::compile(&spec).expect("compile");
        // Exercise every conv op in the tiny twin-wall net.
        for op in &net.ops {
            let TwinOp::Conv(c) = op else { continue };
            let n_in_t = net.tsize[c.input];
            let (co, _, _, _) = c.kernel;
            let p = c.oshape.1 * c.oshape.2;
            let k = c.k_fwd;
            let r = 600usize; // wide enough to hit the BLAS tiling
            let mut src = Array2::<f64>::zeros((n_in_t, r));
            for v in src.iter_mut() {
                *v = rng.random_range(-2.0..2.0);
            }
            let src_s = src.as_slice().unwrap();
            for &abs_w in &[false, true] {
                let mut out_scalar = Array2::<f64>::zeros((co * p, r));
                conv_forward_blocked(c, &src, &mut out_scalar, abs_w);
                let mut out_blas = Array2::<f64>::zeros((co * p, r));
                conv_apply_forward_blas(c, &src, &mut out_blas, abs_w, co, p, k, r);
                let mut out_faer = Array2::<f64>::zeros((co * p, r));
                conv_apply_forward_faer(c, &src, &mut out_faer, abs_w, co, p, k, r);
                let g = gamma_n(k + 2);
                let osl = out_scalar.as_slice().unwrap();
                let obl = out_blas.as_slice().unwrap();
                let ofl = out_faer.as_slice().unwrap();
                for oc in 0..co {
                    for sp in 0..p {
                        for col in 0..r {
                            // Neumaier-compensated reference sum + abs-sum S.
                            let (mut sum, mut comp, mut s_abs) = (0.0f64, 0.0f64, 0.0f64);
                            for t in 0..k {
                                let gi = c.gather[t * p + sp];
                                if gi == usize::MAX {
                                    continue;
                                }
                                let w0 = c.wmat[oc * k + t];
                                let w = if abs_w { w0.abs() } else { w0 };
                                let x = src_s[gi * r + col];
                                let term = w * x;
                                s_abs += w.abs() * x.abs();
                                let ns = sum + term;
                                comp += if sum.abs() >= term.abs() {
                                    (sum - ns) + term
                                } else {
                                    (term - ns) + sum
                                };
                                sum = ns;
                            }
                            let refv = sum + comp;
                            let idx = (oc * p + sp) * r + col;
                            let tol = g * s_abs + 32.0 * UNIT * (s_abs + refv.abs());
                            assert!(
                                (osl[idx] - refv).abs() <= tol,
                                "scalar outside gamma envelope: v={} ref={refv} tol={tol} k={k}",
                                osl[idx]
                            );
                            assert!(
                                (obl[idx] - refv).abs() <= tol,
                                "ndarray outside gamma envelope: v={} ref={refv} tol={tol} k={k}",
                                obl[idx]
                            );
                            assert!(
                                (ofl[idx] - refv).abs() <= tol,
                                "faer outside gamma envelope: v={} ref={refv} tol={tol} k={k}",
                                ofl[idx]
                            );
                            assert!(
                                (ofl[idx] - obl[idx]).abs() <= 2.0 * tol,
                                "backend parity exceeds their shared envelopes: ndarray={} \
                                 faer={} ref={refv} tol={tol} k={k}",
                                obl[idx],
                                ofl[idx]
                            );
                        }
                    }
                }
            }
        }
    }
}

/// #twinwall-blas EXACT-RATIONAL outward-envelope gate (adversarial verifier of
/// ad256b02). Drives the opt-in DGEMM path `conv_apply_forward_blas` on a WIDE
/// 1x1 conv (k up to 9408 taps, ALL active — no `usize::MAX` padding skip, so
/// the full k-term contraction is exercised) with adversarial operands
/// (subnormals, 1e±100 dynamic range, ±0.0, and the forced near-1-ulp ramp vs
/// an all-ones column that maximizes sub-ulp accumulation drift at production
/// width), and asserts the DGEMM output is within the Higham dot-product
/// envelope `γ_k·Σ|w·x| (+ gradual-underflow slack)` of the EXACT rational
/// contraction `Σ_t w[oc,t]·src[t,col]` (BigRational, bit-exact from each f64 —
/// a TRUE oracle, unlike the compensated-float reference of
/// `blas_conv_within_gamma_envelope_of_exact`, which also never reaches
/// production k). This is the precise per-output statement that lets the
/// twin-wall D-lane envelope built in `root.rs` (`γ_n(k_fwd+2)·(|M|+D)` center
/// absorption + `γ_n(k_fwd+8)` D self-cert, both ≥ γ_k) DOMINATE the reordered
/// DGEMM rounding, so the concretized center stays an OUTWARD enclosure of the
/// real value under `NY_ROOT_BLAS`. Mirrors the AW gold oracle
/// `rank1_magnitude_bound_dominates_exact_rational`.
#[test]
fn blas_conv_exact_rational_outward_envelope() {
    use super::net::{conv_apply_forward_blas, conv_apply_forward_faer, ConvOp};
    use super::rounding::UNIT;
    use num_rational::BigRational;
    use num_traits::Signed;
    let exact = |v: f64| BigRational::from_float(v).expect("finite operand");
    let mut rng = StdRng::seed_from_u64(0x0B1A_50AC_C0DE_2026);
    // Adversarial value generator (mirrors the AW gold-oracle regimes).
    let gen_val = |mode: u32, rng: &mut StdRng| -> f64 {
        let sign = if rng.random_range(0.0..1.0) < 0.5 {
            -1.0
        } else {
            1.0
        };
        match mode {
            // subnormals (ulp-uniform down to 2^-1074)
            0 => sign * f64::from_bits(rng.random_range(0.0..4.5e15) as u64 + 1),
            // mixed magnitudes 1e-100..1e100 (products stay < f64::MAX)
            1 => sign * 10f64.powf(rng.random_range(-100.0..100.0)),
            // signed/exact zero (adds exact 0 to the contraction)
            2 => {
                if rng.random_range(0.0..1.0) < 0.5 {
                    -0.0
                } else {
                    0.0
                }
            }
            // near-1 ulp-adversarial ramp (worst nearest-rounding sums)
            _ => sign * (1.0 + rng.random_range(0.0..1.0) * 2f64.powi(-30)),
        }
    };
    let eta = 2f64.powi(-1074); // smallest positive subnormal (underflow unit)
    let mut checked = 0usize;
    // (k, r, forced): forced widths use the near-1 ramp vs all-ones column
    // (small rationals -> cheap even at k=9408); random widths sweep the
    // dynamic-range/subnormal regimes at smaller k (big rationals, bounded).
    let cases: [(usize, usize, bool); 6] = [
        (2, 48, false),
        (3, 48, false),
        (96, 48, false),
        (512, 20, false), // dynamic-range/subnormal at r>=512-class width (big rationals)
        (2048, 12, true), // production-width forced near-1 ramp
        (9408, 6, true),  // naug-class forced ramp (widest tinyimagenet conv-class)
    ];
    for (ci, &(k, r, forced)) in cases.iter().enumerate() {
        let co = 3usize;
        let p = 1usize; // single spatial position -> gather[t] = t, all taps active
        let mode_w = if forced { 3 } else { (ci as u32 + 1) % 4 };
        let wmat: Vec<f64> = (0..co * k).map(|_| gen_val(mode_w, &mut rng)).collect();
        let gather: Vec<usize> = (0..k).collect(); // identity, no usize::MAX padding
        let conv = ConvOp {
            input: 0,
            kernel: (co, k, 1, 1),
            ishape: (k, 1, 1),
            oshape: (co, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            transposed: false,
            wmat: wmat.clone(),
            wt: Vec::new(),
            bias: vec![0.0; co],
            bias_err: vec![0.0; co],
            weight_rel_err: 0.0,
            gather,
            back_taps: Vec::new(),
            k_fwd: k,
            k_bwd: 0,
        };
        let mut src = Array2::<f64>::zeros((k, r));
        let mode_x = (ci as u32 + 2) % 4;
        for col in 0..r {
            for t in 0..k {
                src[[t, col]] = if forced {
                    1.0
                } else {
                    gen_val(mode_x, &mut rng)
                };
            }
        }
        // γ_k = k·u/(1-k·u) (Higham dot-product relative bound, any summation order).
        let gk = (k as f64 * UNIT) / (1.0 - k as f64 * UNIT);
        let gk_r = exact(gk);
        for &abs_w in &[false, true] {
            let mut out_ndarray = Array2::<f64>::zeros((co * p, r));
            conv_apply_forward_blas(&conv, &src, &mut out_ndarray, abs_w, co, p, k, r);
            let mut out_faer = Array2::<f64>::zeros((co * p, r));
            conv_apply_forward_faer(&conv, &src, &mut out_faer, abs_w, co, p, k, r);
            for oc in 0..co {
                for col in 0..r {
                    let mut sum = exact(0.0);
                    let mut sabs = exact(0.0);
                    for t in 0..k {
                        let w0 = wmat[oc * k + t];
                        let w = if abs_w { w0.abs() } else { w0 };
                        let term = exact(w) * exact(src[[t, col]]);
                        sabs += term.abs();
                        sum += term;
                    }
                    // Envelope root.rs relies on: γ_k·Σ|term| + gradual-underflow
                    // slack (8k·η covers the ~2k blocked f64 ops of the DGEMM).
                    let bound = gk_r.clone() * sabs.clone() + exact(8.0 * k as f64) * exact(eta);
                    for (backend, got_f64) in [
                        ("ndarray", out_ndarray[[oc, col]]),
                        ("faer", out_faer[[oc, col]]),
                    ] {
                        let err = (exact(got_f64) - sum.clone()).abs();
                        assert!(
                            err <= bound,
                            "backend={backend} k={k} abs_w={abs_w} oc={oc} col={col}: \
                             GEMM abs err EXCEEDS the γ_k·Σ|term| outward envelope \
                             (out={got_f64})"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(
        checked >= 1000,
        "coverage regressed: {checked} entries checked"
    );
}

/// SINGLE-GATE INVARIANT: `MarginRowBab::run` must actually run the
/// algorithm. It previously carried a second, undocumented
/// `cfg(not(test))` quarantine that made it a no-op in every non-test
/// build, so flipping the documented gate silently did nothing. A no-op
/// would show up here as `Unknown` on an instance the lane provably closes.
#[test]
fn run_entry_is_not_a_second_quarantine() {
    let mut rng = StdRng::seed_from_u64(29);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let mid = Array2::from_shape_fn((72, 1), |(i, _)| f64::midpoint(lo[i], hi[i]));
    let (y, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let scores: Vec<f64> = (0..n_out)
        .map(|o| {
            let mut sm = b2[o];
            for k in 0..n_y {
                sm += w2[o * n_y + k] * y[[k, 0]].max(0.0);
            }
            sm
        })
        .collect();
    let t = (0..n_out)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let adv: Vec<usize> = (0..n_out).filter(|&o| o != t).collect();
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    match MarginRowBab::run(&net, &root, t, &adv, BabConfig::default()) {
        MarginRowOutcome::Unsat(_) => {}
        MarginRowOutcome::Unknown { reason, .. } => panic!(
            "MarginRowBab::run did not decide a provably-closable instance \
             ({reason}) — a second quarantine has been reintroduced"
        ),
    }
}

// ===================== EPOCH-BAB PHASE A TESTS (#epoch-bab) =================
// Tier-0 substrate: retained tableau rows + trunk variant ranker + the
// tier0 lane configuration. Design: docs/EPOCH_BAB_DESIGN.md.

/// Retained (f32) sandwich rows really sandwich sampled pre-activations:
/// `A_l . x^ <= z <= A_u . x^` up to the f32 cast error of the rows.
#[test]
fn epoch_retained_rows_sandwich_sampled_preacts() {
    let mut rng = StdRng::seed_from_u64(51);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.3; 72];
    let hi = vec![0.5; 72];
    let cfg = super::root::RetainCfg {
        per_layer: 1_000_000,
        budget_bytes: 1 << 30,
    };
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let (root, ret) =
            RootGates::build_retaining(&net, &lo, &hi, mode, None, Some(&cfg), &[]).expect("root");
        let ret = ret.expect("retention requested");
        assert_eq!(ret.layers.len(), root.layers.len());
        let n_ret: usize = ret.layers.iter().map(|l| l.idx.len()).sum();
        assert!(n_ret > 0, "no rows retained on a net with unstable neurons");
        let x = sample_box(&mut rng, &root, 64);
        let sel: std::collections::BTreeMap<usize, Vec<usize>> = root
            .layers
            .iter()
            .zip(&ret.layers)
            .map(|(lg, lr)| (lg.op, lr.idx.clone()))
            .collect();
        let (_, pre) = net.forward_points(&x, &sel).expect("forward");
        for (lg, lr) in root.layers.iter().zip(&ret.layers) {
            let p = &pre[&lg.op];
            let naug = lr.naug;
            for (ri, &pos) in lr.unst_pos.iter().enumerate() {
                assert_eq!(lg.unst[pos], lr.idx[ri], "unst_pos <-> idx mismatch");
                let al = &lr.a_l[ri * naug..(ri + 1) * naug];
                let au = &lr.a_u[ri * naug..(ri + 1) * naug];
                for b in 0..x.ncols() {
                    let (mut vl, mut vu, mut mag) = (
                        f64::from(al[naug - 1]),
                        f64::from(au[naug - 1]),
                        f64::from(al[naug - 1]).abs() + f64::from(au[naug - 1]).abs(),
                    );
                    for k in 0..naug - 1 {
                        vl += f64::from(al[k]) * x[[k, b]];
                        vu += f64::from(au[k]) * x[[k, b]];
                        mag += (f64::from(al[k]).abs() + f64::from(au[k]).abs()) * x[[k, b]].abs();
                    }
                    // f32 cast slack: rel 2^-24 on each coefficient.
                    let tol = 1e-9 + mag * 1.2e-7;
                    let z = p[[ri, b]];
                    assert!(
                        vl - tol <= z && z <= vu + tol,
                        "{mode:?} op {} neuron {}: z={z} outside f32 sandwich \
                         [{vl}, {vu}] (tol {tol:.3e})",
                        lg.op,
                        lr.idx[ri]
                    );
                }
            }
        }
    }
}

/// The Tier-0 trunk variant ranker is (a) pointwise sound on matching-sign
/// samples (up to f32-row slack) and (b) correlated with the exact
/// exception-pass child scores it approximates.
#[test]
fn epoch_trunk_variant_pointwise_sound_and_correlates() {
    let mut rng = StdRng::seed_from_u64(53);
    let spec = tiny_spec(&mut rng, 0.65);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let cfg = super::root::RetainCfg {
        per_layer: 1_000_000,
        budget_bytes: 1 << 30,
    };
    let (root, ret) =
        RootGates::build_retaining(&net, &lo, &hi, RoundMode::Parity, None, Some(&cfg), &[])
            .expect("root");
    let ret = ret.expect("retention requested");
    let eng = BackwardEngine::new(&net, &root);
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    let mb = MarginBatch::new(&net, t, &adv).expect("mb");
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let gates = head_gates(&ybox, RoundMode::Parity);
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Parity);
    let pass = eng
        .run_collect(
            &ms.seed,
            None,
            LaneDir::Lower,
            None,
            super::engine::Collect {
                unst_abs: true,
                rows: Some(&ret),
            },
        )
        .expect("direct pass");
    let nf = mb.nf();
    // ---- Tier-0 scores for every retained candidate ----
    let mut cands: Vec<(usize, usize, usize)> = Vec::new(); // (li, ri, pos)
    for (li, lr) in ret.layers.iter().enumerate() {
        for (ri, &pos) in lr.unst_pos.iter().enumerate() {
            cands.push((li, ri, pos));
        }
    }
    assert!(!cands.is_empty(), "need retained candidates");
    let coll_rows = pass.coll_rows.as_ref().expect("rows captured");
    let t0_scores: Vec<(f64, f64)> = cands
        .iter()
        .map(|&(li, ri, _)| {
            let vmat = &coll_rows[&li];
            let r = vmat.ncols();
            let vs = vmat.as_slice().expect("layout");
            let vrow = &vs[ri * r..(ri + 1) * r];
            let ba =
                super::bounds::trunk_variant(&root, &ret.layers[li], li, ri, vrow, &pass, &ms, 1);
            let bi =
                super::bounds::trunk_variant(&root, &ret.layers[li], li, ri, vrow, &pass, &ms, -1);
            (ba, bi)
        })
        .collect();
    // ---- (a) pointwise soundness on matching-sign samples ----
    let mut sel: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for (li, lr) in ret.layers.iter().enumerate() {
        let op = root.layers[li].op;
        sel.entry(op).or_default().extend(lr.idx.iter().copied());
    }
    let x = sample_box(&mut rng, &root, 500);
    let (y, pre) = net.forward_points(&x, &sel).expect("forward");
    let margins = margins_at(&net, &y, t, &adv);
    let mut checked = 0usize;
    for (ci, &(li, ri, _)) in cands.iter().enumerate() {
        let op = root.layers[li].op;
        let row_in_sel = sel[&op]
            .iter()
            .position(|&n| n == ret.layers[li].idx[ri])
            .expect("present");
        let pv = &pre[&op];
        let (ba, bi) = t0_scores[ci];
        for b in 0..x.ncols() {
            let sgn = pv[[row_in_sel, b]];
            let est = if sgn >= 0.0 { ba } else { bi };
            let min_m = (0..nf).map(|f| margins[f][b]).fold(f64::INFINITY, f64::min);
            // Slack: f32 rows + nearest-mode arithmetic (ranker grade).
            assert!(
                est <= min_m + 1e-4,
                "TIER0 POINTWISE VIOLATION: cand {ci} (li={li}) sign {sgn:+.3e} \
                 point {b}: est {est} > true min margin {min_m}",
            );
            checked += 1;
        }
    }
    eprintln!("[epoch] tier0 pointwise checks: {checked}");
    // ---- (b) correlation vs the exact exception-pass child scores ----
    let total = 2 * cands.len() * nf;
    let mut seed = Array2::<f64>::zeros((net.n_y, total));
    let mut exc = Exceptions::default();
    for (kc, &(li, _ri, pos)) in cands.iter().enumerate() {
        let idx = root.layers[li].unst[pos];
        for (d_i, fix) in [(0usize, (1.0, 1.0, 0.0)), (1usize, (0.0, 0.0, 0.0))] {
            let r0 = (2 * kc + d_i) * nf;
            for j in 0..net.n_y {
                for f in 0..nf {
                    seed[[j, r0 + f]] = ms.seed.s[[j, f]];
                }
            }
            for f in 0..nf {
                exc.by_layer.entry(li).or_default().push(Exc {
                    row: r0 + f,
                    neuron: idx,
                    a2: fix.0,
                    s2: fix.1,
                    c2: fix.2,
                });
            }
        }
    }
    let epass = eng
        .run(
            &super::engine::Seed { s: seed, e: None },
            None,
            LaneDir::Lower,
            Some(&exc),
            false,
        )
        .expect("exception pass");
    let low = eng.concretize_lower(&epass);
    let exact: Vec<(f64, f64)> = (0..cands.len())
        .map(|kc| {
            let mut pair = [f64::INFINITY, f64::INFINITY];
            for (d_i, p) in pair.iter_mut().enumerate() {
                let r0 = (2 * kc + d_i) * nf;
                let mut worst = f64::INFINITY;
                for f in 0..nf {
                    let v = (low[r0 + f] + ms.cst[f]).max(ms.m1[f]);
                    worst = worst.min(v);
                }
                *p = worst;
            }
            (pair[0], pair[1])
        })
        .collect();
    // Rank both by the pick key (min child, sum). The tier0 argmax must land
    // in the exact top half, and Spearman over min-scores must be positive.
    let key = |p: &(f64, f64)| (p.0.min(p.1), p.0 + p.1);
    let rank_of = |scores: &[(f64, f64)]| -> Vec<usize> {
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            let (ka, kb) = (key(&scores[a]), key(&scores[b]));
            kb.0.total_cmp(&ka.0).then_with(|| kb.1.total_cmp(&ka.1))
        });
        let mut rank = vec![0usize; scores.len()];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r;
        }
        rank
    };
    let rank_t0 = rank_of(&t0_scores);
    let rank_ex = rank_of(&exact);
    let n = cands.len();
    let t0_best = rank_t0.iter().position(|&r| r == 0).expect("has best");
    eprintln!(
        "[epoch] tier0-best candidate exact rank: {}/{n} (exact scores {:?} t0 {:?})",
        rank_ex[t0_best], exact[t0_best], t0_scores[t0_best]
    );
    // Measured 2026-07-18 (seed 53): rank 1/72, spearman 0.911. Thresholds
    // pinned well inside that with headroom for platform FP variation.
    assert!(
        rank_ex[t0_best] <= n / 4,
        "tier0 argmax ranks {}/{n} by the exact scorer — ranking degraded",
        rank_ex[t0_best]
    );
    let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
    let (ma, mb_) = (mean(&rank_t0), mean(&rank_ex));
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for i in 0..n {
        let a = rank_t0[i] as f64 - ma;
        let b = rank_ex[i] as f64 - mb_;
        num += a * b;
        da += a * a;
        db += b * b;
    }
    let spearman = num / (da.sqrt() * db.sqrt()).max(1e-12);
    eprintln!("[epoch] tier0-vs-exact spearman over {n} candidates: {spearman:.3}");
    assert!(
        spearman > 0.6,
        "tier0 ranking decorrelated from exact child scores: spearman {spearman:.3}"
    );
}

/// Full-lane smoke with Tier-0 enabled: the argmax instance still verifies,
/// the swapped (falsified) instance still must NOT verify (moat), and the
/// tier0 configuration actually engages (retained rows present).
#[test]
fn epoch_tier0_lane_closes_and_moat_holds() {
    let mut rng = StdRng::seed_from_u64(57);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let mid = Array2::from_shape_fn((72, 1), |(i, _)| f64::midpoint(lo[i], hi[i]));
    let (y, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let scores: Vec<f64> = (0..n_out)
        .map(|o| {
            let mut sm = b2[o];
            for k in 0..n_y {
                sm += w2[o * n_y + k] * y[[k, 0]].max(0.0);
            }
            sm
        })
        .collect();
    let t = (0..n_out)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let worst = (0..n_out)
        .filter(|&o| o != t)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let adv: Vec<usize> = (0..n_out).filter(|&o| o != t).collect();
    let cfg = super::root::RetainCfg::default();
    let (root, ret) =
        RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, Some(&cfg), &[])
            .expect("root");
    let retained = std::sync::Arc::new(ret.expect("retention requested"));
    let mk_cfg = || BabConfig {
        tier0_exact: 4,
        tier0_universe: 64,
        retained: Some(retained.clone()),
        ..BabConfig::default()
    };
    match MarginRowBab::run(&net, &root, t, &adv, mk_cfg()) {
        MarginRowOutcome::Unsat(stats) => {
            assert_eq!(
                stats.tree_classes.len() + stats.root_closed_classes,
                adv.len()
            );
            assert_eq!(
                stats.ledger_ok,
                Some(true),
                "Kraft ledger must verify on a tier0-closed tree"
            );
        }
        MarginRowOutcome::Unknown { reason, .. } => {
            panic!("tier0 argmax instance should verify, got Unknown: {reason}")
        }
    }
    match MarginRowBab::run(&net, &root, worst, &[t], mk_cfg()) {
        MarginRowOutcome::Unsat(_) => {
            panic!("MOAT VIOLATION: tier0 lane verified a falsified instance")
        }
        MarginRowOutcome::Unknown { .. } => {}
    }
}
// ===================== END EPOCH-BAB PHASE A TESTS ==========================

// ===================== EPOCH-BAB PHASE B TESTS (#epoch-bab Tier 2) ==========
// Epoch re-linearization: baked-split tableau rebuilds + the nested lane.

/// Epoch gates (splits baked into the forward tableau) are pointwise sound
/// on their halfspaces: for every sampled point whose exact pre-activation
/// sign matches the baked direction, every per-class epoch bound is <= the
/// true margin. Also: the baked neuron leaves the epoch's unstable list.
#[test]
fn epoch_baked_splits_pointwise_sound_on_halfspaces() {
    let mut rng = StdRng::seed_from_u64(61);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.35; 72];
    let hi = vec![0.45; 72];
    let (root, _) = RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, None, &[])
        .expect("root");
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    // Pick the unstable trunk neuron whose sampled pre-activation signs are
    // most balanced, so both halfspaces get real coverage.
    let x = sample_box(&mut rng, &root, 1000);
    let sel_all: std::collections::BTreeMap<usize, Vec<usize>> = root
        .layers
        .iter()
        .map(|lg| (lg.op, lg.unst.clone()))
        .collect();
    let (y, pre_all) = net.forward_points(&x, &sel_all).expect("forward");
    let (mut li, mut idx, mut best_bal) = (usize::MAX, usize::MAX, 0usize);
    for (l, lg) in root.layers.iter().enumerate() {
        let p = &pre_all[&lg.op];
        for (row, _) in lg.unst.iter().enumerate() {
            let pos_cnt = (0..x.ncols()).filter(|&b| p[[row, b]] >= 0.0).count();
            let bal = pos_cnt.min(x.ncols() - pos_cnt);
            if bal > best_bal {
                best_bal = bal;
                li = l;
                idx = lg.unst[row];
            }
        }
    }
    assert!(li != usize::MAX, "need an unstable trunk neuron");
    let relu_op = root.layers[li].op;
    let row_in_sel = sel_all[&relu_op]
        .iter()
        .position(|&n| n == idx)
        .expect("present");
    let margins = margins_at(&net, &y, t, &adv);
    let pre_t = &pre_all[&relu_op];
    for dir in [1i8, -1] {
        let (eroot, _) = RootGates::build_retaining(
            &net,
            &lo,
            &hi,
            RoundMode::Outward,
            None,
            None,
            &[(li, idx, dir)],
        )
        .expect("epoch root");
        assert!(
            !eroot.layers[li].unst.contains(&idx),
            "baked neuron still in the epoch unstable list"
        );
        let bounds = domain_class_bounds(&net, &eroot, t, &adv, &[], &[]);
        let mut on_side = 0usize;
        for b in 0..x.ncols() {
            let sgn = pre_t[[row_in_sel, b]];
            let matches = if dir > 0 { sgn >= 0.0 } else { sgn <= 0.0 };
            if !matches {
                continue;
            }
            on_side += 1;
            for (k, mrow) in margins.iter().enumerate() {
                assert!(
                    bounds[k] <= mrow[b],
                    "EPOCH HALFSPACE VIOLATION: dir {dir} class {k} point {b}: \
                     epoch bound {} > true margin {} (pre={sgn})",
                    bounds[k],
                    mrow[b]
                );
            }
        }
        assert!(on_side > 50, "degenerate halfspace sampling: {on_side}");
        eprintln!("[epoch-b] dir {dir}: {on_side} on-side points checked, sound");
    }
}

/// Epoch rebuild vs frozen-gates domain overrides on the same split sets:
/// measures the bound delta (the Tier-2 thesis is that baking splits into
/// the forward tableau tightens downstream gates). Requires improvement on
/// at least one tested split set and soundness is covered by the halfspace
/// test above.
#[test]
fn epoch_rebuild_vs_frozen_domain_bounds() {
    let mut rng = StdRng::seed_from_u64(67);
    let spec = tiny_spec(&mut rng, 0.65);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let (root, _) = RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, None, &[])
        .expect("root");
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    let mut best_gain = f64::NEG_INFINITY;
    let mut tested = 0usize;
    for (li, lg) in root.layers.iter().enumerate() {
        for pos in 0..lg.unst.len().min(3) {
            let idx = lg.unst[pos];
            for dir in [1i8, -1] {
                let frozen = domain_class_bounds(&net, &root, t, &adv, &[(li, pos, dir)], &[]);
                let (eroot, _) = RootGates::build_retaining(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    None,
                    None,
                    &[(li, idx, dir)],
                )
                .expect("epoch root");
                let epoch = domain_class_bounds(&net, &eroot, t, &adv, &[], &[]);
                let fmin = frozen.iter().copied().fold(f64::INFINITY, f64::min);
                let emin = epoch.iter().copied().fold(f64::INFINITY, f64::min);
                best_gain = best_gain.max(emin - fmin);
                tested += 1;
                eprintln!(
                    "[epoch-b] split (li={li},idx={idx},dir={dir}): frozen {fmin:.6} \
                     epoch {emin:.6} gain {:+.3e}",
                    emin - fmin
                );
            }
        }
    }
    assert!(tested >= 4, "need several split sets, got {tested}");
    eprintln!("[epoch-b] best epoch-vs-frozen gain over {tested} splits: {best_gain:+.3e}");
    assert!(
        best_gain > 0.0,
        "epoch rebuild never beat frozen-gates domain overrides on {tested} splits \
         (best gain {best_gain:+.3e}) — Tier-2 thesis violated on the fixture"
    );
}

/// Full-lane smoke with Tier-2 epochs enabled (k_head=0 forces trunk splits
/// so depth-1 domains trigger epochs): the argmax instance still verifies,
/// the swapped falsified instance still must NOT verify, and epoch stats
/// account attempts.
#[test]
fn epoch_nested_lane_closes_and_moat_holds() {
    let mut rng = StdRng::seed_from_u64(71);
    let spec = tiny_spec(&mut rng, 0.5);
    let net = TwinNet::compile(&spec).expect("compile");
    // A box wide enough that the tree actually expands (unlike the tiny
    // argmax box) but still closable.
    let lo = vec![-0.02; 72];
    let hi = vec![0.06; 72];
    let mid = Array2::from_shape_fn((72, 1), |(i, _)| f64::midpoint(lo[i], hi[i]));
    let (y, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let scores: Vec<f64> = (0..n_out)
        .map(|o| {
            let mut sm = b2[o];
            for k in 0..n_y {
                sm += w2[o * n_y + k] * y[[k, 0]].max(0.0);
            }
            sm
        })
        .collect();
    let t = (0..n_out)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let worst = (0..n_out)
        .filter(|&o| o != t)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let adv: Vec<usize> = (0..n_out).filter(|&o| o != t).collect();
    let retain = super::root::RetainCfg::default();
    let (root, _) =
        RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, Some(&retain), &[])
            .expect("root");
    let mk_cfg = || BabConfig {
        k_head: 0,
        epoch_depth: 1,
        epoch_max_attempts: 8,
        retain_cfg: Some(retain),
        ..BabConfig::default()
    };
    match MarginRowBab::run(&net, &root, t, &adv, mk_cfg()) {
        MarginRowOutcome::Unsat(stats) => {
            eprintln!(
                "[epoch-b] nested lane closed: expansions {} depth {} epochs {}/{} ledger {:?}",
                stats.expansions,
                stats.max_depth,
                stats.epochs_closed,
                stats.epochs_attempted,
                stats.ledger_ok
            );
            assert_eq!(
                stats.ledger_ok,
                Some(true),
                "Kraft ledger must verify on an epoch-closed tree"
            );
        }
        MarginRowOutcome::Unknown { reason, stats } => {
            panic!(
                "epoch lane should verify, got Unknown: {reason} (stats {:?})",
                stats.map(|s| (s.expansions, s.epochs_attempted))
            )
        }
    }
    match MarginRowBab::run(&net, &root, worst, &[t], mk_cfg()) {
        MarginRowOutcome::Unsat(_) => {
            panic!("MOAT VIOLATION: epoch lane verified a falsified instance")
        }
        MarginRowOutcome::Unknown { .. } => {}
    }
}
// ===================== END EPOCH-BAB PHASE B TESTS ==========================

// ===================== EPOCH-BAB PHASE D TESTS (#epoch-bab) =================
// ConvTranspose + ChannelAffine core ops (the cgan/metaroom class).

/// cgan-shaped synthetic net: 1x3x3 input -> ConvTranspose(2ch, k3, s2) ->
/// ChannelAffine -> Relu -> Conv -> Relu -> Flatten -> Gemm -> Relu -> Gemm.
fn cgan_like_spec(rng: &mut StdRng, scale: f64) -> TwinSpec {
    let mut w =
        |n: usize| -> Vec<f64> { (0..n).map(|_| rng.random_range(-scale..scale)).collect() };
    // ConvTranspose: (co=2, ci=1, k=3, s=2, p=1): ih=3 -> oh=(3-1)*2-2+3=5.
    let convt = TwinOpSpec::ConvTranspose {
        input: 0,
        weight: w(2 * 3 * 3),
        bias: w(2),
        bias_err: vec![0.0; 2],
        weight_rel_err: 1e-15,
        kernel: (2, 1, 3, 3),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (1, 3, 3),
        oshape: (2, 5, 5),
        out_pad: (0, 0),
    };
    let aff_scale: Vec<f64> = w(2).iter().map(|v| 0.5 + v.abs()).collect();
    let chaff = TwinOpSpec::ChannelAffine {
        input: 1,
        scale: aff_scale,
        shift: w(2),
        scale_rel_err: 1e-15,
        shift_err: vec![1e-18; 2],
        shape: (2, 5, 5),
    };
    let conv = TwinOpSpec::Conv {
        input: 3,
        weight: w(3 * 2 * 3 * 3),
        bias: w(3),
        bias_err: vec![0.0; 3],
        weight_rel_err: 0.0,
        kernel: (3, 2, 3, 3),
        stride: (2, 2),
        pads: (1, 1, 1, 1),
        ishape: (2, 5, 5),
        oshape: (3, 3, 3),
    };
    TwinSpec {
        n_in: 9,
        ops: vec![
            convt,                            // t1 (2,5,5)
            chaff,                            // t2
            TwinOpSpec::Relu { input: 2 },    // t3 (trunk relu 0)
            conv,                             // t4 (3,3,3)
            TwinOpSpec::Relu { input: 4 },    // t5 (trunk relu 1)
            TwinOpSpec::Flatten { input: 5 }, // t6
            TwinOpSpec::Gemm {
                input: 6,
                weight: w(6 * 27),
                bias: w(6),
                shape: (6, 27),
            }, // t7 (y)
            TwinOpSpec::Relu { input: 7 },    // t8 (head relu)
            TwinOpSpec::Gemm {
                input: 8,
                weight: w(4 * 6),
                bias: w(4),
                shape: (4, 6),
            }, // t9
        ],
    }
}

/// Naive dense ConvTranspose reference: out[oc][oy][ox] += w[oc][ic][ky][kx]
/// * in[ic][iy][ix] where oy = iy*s - pt + ky (the standard definition).
#[allow(clippy::too_many_arguments)]
fn naive_conv_transpose(
    x: &[f64],
    weight: &[f64],
    bias: &[f64],
    kernel: (usize, usize, usize, usize),
    stride: (usize, usize),
    pads: (usize, usize, usize, usize),
    ishape: (usize, usize, usize),
    oshape: (usize, usize, usize),
) -> Vec<f64> {
    let (co, ci, kh, kw) = kernel;
    let (_, ih, iw) = ishape;
    let (_, oh, ow) = oshape;
    let mut out = vec![0.0; co * oh * ow];
    for (oc, o) in out.iter_mut().enumerate() {
        *o = bias[oc / (oh * ow)];
    }
    for ic in 0..ci {
        for iy in 0..ih {
            for ix in 0..iw {
                let v = x[ic * ih * iw + iy * iw + ix];
                for oc in 0..co {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ny = iy * stride.0 + ky;
                            let nx = ix * stride.1 + kx;
                            if ny < pads.0 || nx < pads.1 {
                                continue;
                            }
                            let (oy, ox) = (ny - pads.0, nx - pads.1);
                            if oy < oh && ox < ow {
                                out[oc * oh * ow + oy * ow + ox] +=
                                    weight[((oc * ci + ic) * kh + ky) * kw + kx] * v;
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// ConvTranspose gather tables match the naive dense reference pointwise.
#[test]
fn epoch_convtranspose_matches_naive_reference() {
    let mut rng = StdRng::seed_from_u64(81);
    let spec = cgan_like_spec(&mut rng, 0.7);
    let net = TwinNet::compile(&spec).expect("compile");
    let (weight, bias, kernel, stride, pads, ishape, oshape) = match &spec.ops[0] {
        TwinOpSpec::ConvTranspose {
            weight,
            bias,
            kernel,
            stride,
            pads,
            ishape,
            oshape,
            ..
        } => (weight, bias, *kernel, *stride, *pads, *ishape, *oshape),
        _ => unreachable!("fixture starts with convtranspose"),
    };
    let mut x = Array2::<f64>::zeros((9, 16));
    for v in x.iter_mut() {
        *v = rng.random_range(-1.0..1.0);
    }
    // forward_points runs through gemm1; instead check the FIRST op alone by
    // truncating: build a net of just [ConvT] + head filler is invasive, so
    // compare through pre-activations: pre at trunk relu 0 = ChannelAffine(
    // ConvT(x)); invert the (known) affine per channel to recover ConvT(x).
    let sel: std::collections::BTreeMap<usize, Vec<usize>> =
        [(2usize, (0..50).collect::<Vec<_>>())]
            .into_iter()
            .collect();
    let (_, pre) = net.forward_points(&x, &sel).expect("forward");
    let p = &pre[&2];
    let (aff_scale, aff_shift) = match &spec.ops[1] {
        TwinOpSpec::ChannelAffine { scale, shift, .. } => (scale.clone(), shift.clone()),
        _ => unreachable!(),
    };
    for b in 0..x.ncols() {
        let xb: Vec<f64> = (0..9).map(|i| x[[i, b]]).collect();
        let want = naive_conv_transpose(&xb, weight, bias, kernel, stride, pads, ishape, oshape);
        for j in 0..50 {
            let ch = j / 25;
            let got = (p[[j, b]] - aff_shift[ch]) / aff_scale[ch];
            assert!(
                (got - want[j]).abs() <= 1e-9 * (1.0 + want[j].abs()),
                "convtranspose mismatch at out {j} point {b}: got {got} want {}",
                want[j]
            );
        }
    }
}

/// Tableau boxes + margin bounds through ConvTranspose + ChannelAffine
/// enclose sampled values in both modes (the Phase D core soundness pin).
#[test]
fn epoch_cgan_like_enclosure_and_lane() {
    let mut rng = StdRng::seed_from_u64(83);
    let spec = cgan_like_spec(&mut rng, 0.7);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 9];
    let hi = vec![0.4; 9];
    let t = 0usize;
    let adv = vec![1usize, 2, 3];
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let root = RootGates::build(&net, &lo, &hi, mode, None).expect("root");
        // Boxes enclose pre-activations.
        let x = sample_box(&mut rng, &root, 200);
        let sel: std::collections::BTreeMap<usize, Vec<usize>> = root
            .layers
            .iter()
            .map(|lg| (lg.op, (0..lg.n).collect()))
            .collect();
        let (y, pre) = net.forward_points(&x, &sel).expect("forward");
        for lg in &root.layers {
            let p = &pre[&lg.op];
            for j in 0..lg.n {
                for b in 0..x.ncols() {
                    let v = p[[j, b]];
                    assert!(
                        v >= lg.l[j] - 1e-9 && v <= lg.u[j] + 1e-9,
                        "{mode:?} op {} neuron {j}: {v} outside [{}, {}]",
                        lg.op,
                        lg.l[j],
                        lg.u[j]
                    );
                }
            }
        }
        // Root margin bounds enclose sampled margins.
        let eng = BackwardEngine::new(&net, &root);
        let (al, au) = eng.y_rows(None).expect("y_rows");
        let ybox = YBox::from_rows(&eng, &al, &au);
        let mb = MarginBatch::new(&net, t, &adv).expect("mb");
        let gates = head_gates(&ybox, mode);
        let ms = margin_seed(&mb, &gates, &ybox, mode);
        let pass = eng
            .run(&ms.seed, None, LaneDir::Lower, None, false)
            .expect("pass");
        let direct = per_class_direct(&eng, &pass, &ms, 0..adv.len());
        let margins = margins_at(&net, &y, t, &adv);
        for (k, mrow) in margins.iter().enumerate() {
            let min_m = mrow.iter().copied().fold(f64::INFINITY, f64::min);
            let bound = direct[k].max(ms.m1[k]);
            assert!(
                bound <= min_m + 1e-9,
                "mode {mode:?} class {k}: bound {bound} > sampled min {min_m}"
            );
        }
    }
    // Full lane on a tiny argmax box (the smoke pattern) + moat swap.
    let lo2 = vec![0.01; 9];
    let hi2 = vec![0.03; 9];
    let mid = Array2::from_shape_fn((9, 1), |(i, _)| f64::midpoint(lo2[i], hi2[i]));
    let (ymid, _) = net
        .forward_points(&mid, &std::collections::BTreeMap::new())
        .expect("forward");
    let (w2, b2, (n_out, n_y)) = net.gemm2();
    let scores: Vec<f64> = (0..n_out)
        .map(|o| {
            let mut sm = b2[o];
            for k in 0..n_y {
                sm += w2[o * n_y + k] * ymid[[k, 0]].max(0.0);
            }
            sm
        })
        .collect();
    let tt = (0..n_out)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let worst = (0..n_out)
        .filter(|&o| o != tt)
        .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .expect("classes");
    let advt: Vec<usize> = (0..n_out).filter(|&o| o != tt).collect();
    let root = RootGates::build(&net, &lo2, &hi2, RoundMode::Outward, None).expect("root");
    match MarginRowBab::run(&net, &root, tt, &advt, BabConfig::default()) {
        MarginRowOutcome::Unsat(stats) => {
            assert_eq!(stats.ledger_ok, Some(true), "ledger on cgan-like tree");
        }
        MarginRowOutcome::Unknown { reason, .. } => {
            panic!("cgan-like argmax instance should verify: {reason}")
        }
    }
    match MarginRowBab::run(&net, &root, worst, &[tt], BabConfig::default()) {
        MarginRowOutcome::Unsat(_) => panic!("MOAT VIOLATION on cgan-like fixture"),
        MarginRowOutcome::Unknown { .. } => {}
    }
}
// ===================== END EPOCH-BAB PHASE D TESTS ==========================

// ===========================================================================
// Adversarial soundness enclosure oracles for `RootEval.dj` (banked 2026-07-19
// from wf_e46f0621 d7f-6/7/8). Each oracle is isolated in its own module so its
// from-scratch reference forward cannot name-collide with the others. Property
// under test in all three: the certified per-class lower bound `dj[k]` never
// exceeds the true feasible margin at any concrete x in the box, in
// RoundMode::Outward (the verdict mode) => no false-UNSAT. Pure test additions.
// ===========================================================================

#[cfg(test)]
mod enclosure_oracle_dense {
    use super::super::bab::root_eval;
    use super::*;

    // =====================================================================
    // ENCLOSURE ORACLE (adversarial soundness hunt for `root_eval`.dj).
    //
    // Claim under test (mod.rs / bab.rs): the certified per-class lower bound
    // `RootEval.dj[k]` (= max(m1, m2v, direct)) is a SOUND lower bound on
    //   min_{x in [lo,hi]} ( Y_t(x) - Y_{adv[k]}(x) )
    // in RoundMode::Outward (the verdict mode). If dj[k] ever EXCEEDS a true
    // feasible margin at any concrete x in the box, the lane is UNSOUND (a
    // false-UNSAT risk). We hunt for such an x with broad random-dense nets and
    // heavy sampling.
    //
    // Independence: `ref_forward` below is a from-scratch textbook forward eval
    // over the raw `TwinSpec` op list (naive conv triple-loop, relu, add, gemm).
    // It shares NO code with net.rs (no compiled gather table, no
    // `conv_apply_forward`, no `forward_points`) and NONE of the interval /
    // tableau / backward machinery that produces `dj`. So agreement/violation is
    // a genuine cross-check, not a tautology.
    // =====================================================================

    /// INDEPENDENT exact-f64 forward reference. Evaluates a `TwinSpec` directly
    /// from its op list and returns the length-`n_out` output vector for input `x`
    /// (flat, length `n_in`). No net.rs, no interval code.
    fn ref_forward(spec: &TwinSpec, x: &[f64]) -> Vec<f64> {
        // tensors[0] = input; op k (0-based) produces tensors[k+1].
        let mut tensors: Vec<Vec<f64>> = Vec::with_capacity(spec.ops.len() + 1);
        tensors.push(x.to_vec());
        for op in &spec.ops {
            let out = match op {
                TwinOpSpec::ConvTranspose { .. } | TwinOpSpec::ChannelAffine { .. } => {
                    unreachable!("oracle net generators never emit ConvTranspose/ChannelAffine")
                }
                TwinOpSpec::Conv {
                    input,
                    weight,
                    bias,
                    kernel,
                    stride,
                    pads,
                    ishape,
                    oshape,
                    ..
                } => {
                    let (co, ci, kh, kw) = *kernel;
                    let (_ic, ih, iw) = *ishape;
                    let (_oc, oh, ow) = *oshape;
                    let (sh, sw) = *stride;
                    let (pt, pl, _pb, _pr) = *pads;
                    let src = &tensors[*input];
                    let mut out = vec![0.0f64; co * oh * ow];
                    for oc in 0..co {
                        for oy in 0..oh {
                            for ox in 0..ow {
                                let mut acc = bias[oc];
                                for c in 0..ci {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            // iy = oy*sh + ky - pt (textbook conv);
                                            // skip when the tap falls in padding.
                                            let ty = oy * sh + ky;
                                            let tx = ox * sw + kx;
                                            if ty < pt || tx < pl {
                                                continue;
                                            }
                                            let iy = ty - pt;
                                            let ix = tx - pl;
                                            if iy >= ih || ix >= iw {
                                                continue;
                                            }
                                            let wv = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            let sv = src[(c * ih + iy) * iw + ix];
                                            acc += wv * sv;
                                        }
                                    }
                                }
                                out[(oc * oh + oy) * ow + ox] = acc;
                            }
                        }
                    }
                    out
                }
                TwinOpSpec::Relu { input } => tensors[*input].iter().map(|v| v.max(0.0)).collect(),
                TwinOpSpec::Add { lhs, rhs } => tensors[*lhs]
                    .iter()
                    .zip(&tensors[*rhs])
                    .map(|(a, b)| a + b)
                    .collect(),
                TwinOpSpec::Flatten { input } => tensors[*input].clone(),
                TwinOpSpec::Gemm {
                    input,
                    weight,
                    bias,
                    shape,
                } => {
                    let (no, ni) = *shape;
                    let src = &tensors[*input];
                    (0..no)
                        .map(|o| {
                            let mut acc = bias[o];
                            for i in 0..ni {
                                acc += weight[o * ni + i] * src[i];
                            }
                            acc
                        })
                        .collect()
                }
            };
            tensors.push(out);
        }
        tensors.pop().expect("at least one op")
    }

    /// Build a random VALID twin-wall net + its input shape. Trunk = 1..3 blocks,
    /// each either a plain conv->relu (changes channels/spatial) or a residual
    /// conv->add->relu (shape-preserving), then Flatten->Gemm1->Relu->Gemm2. All
    /// certified error budgets are ZERO (weights exact vs the reference), the
    /// strictest setting: any dj overshoot is then a real bug, not slack.
    fn rand_net(rng: &mut StdRng) -> (TwinSpec, (usize, usize, usize)) {
        let scale: f64 = rng.random_range(0.3..1.1);
        let wv = |n: usize, rng: &mut StdRng| -> Vec<f64> {
            (0..n).map(|_| rng.random_range(-scale..scale)).collect()
        };
        let c0: usize = rng.random_range(1usize..4); // 1..=3 channels
        let hw: usize = rng.random_range(3usize..8); // 3..=7 (square)
        let (mut cc, mut ch, mut cw) = (c0, hw, hw);
        let mut cur_id = 0usize;
        let mut ops: Vec<TwinOpSpec> = Vec::new();
        let nblocks: usize = rng.random_range(1usize..4); // 1..=3 trunk blocks
        for _ in 0..nblocks {
            let residual = rng.random_range(0usize..2) == 0;
            let k = if rng.random_range(0usize..2) == 0 {
                1
            } else {
                3
            };
            let pad = (k - 1) / 2;
            if residual {
                // Shape-preserving conv (stride 1, out channels == in channels),
                // then Add with the block input, then Relu.
                let (co, ci, kh, kw) = (cc, cc, k, k);
                let oh = (ch + 2 * pad - k) + 1;
                let ow = (cw + 2 * pad - k) + 1;
                debug_assert_eq!((oh, ow), (ch, cw));
                let conv = TwinOpSpec::Conv {
                    input: cur_id,
                    weight: wv(co * ci * kh * kw, rng),
                    bias: wv(co, rng),
                    bias_err: vec![0.0; co],
                    weight_rel_err: 0.0,
                    kernel: (co, ci, kh, kw),
                    stride: (1, 1),
                    pads: (pad, pad, pad, pad),
                    ishape: (cc, ch, cw),
                    oshape: (co, oh, ow),
                };
                ops.push(conv);
                let conv_id = ops.len(); // output id = index+1 = len
                ops.push(TwinOpSpec::Add {
                    lhs: conv_id,
                    rhs: cur_id,
                });
                let add_id = ops.len();
                ops.push(TwinOpSpec::Relu { input: add_id });
                cur_id = ops.len();
                // shape unchanged
            } else {
                let co: usize = rng.random_range(2usize..5); // 2..=4 out channels
                let stride = if rng.random_range(0usize..2) == 0 {
                    1
                } else {
                    2
                };
                let oh = (ch + 2 * pad - k) / stride + 1;
                let ow = (cw + 2 * pad - k) / stride + 1;
                let conv = TwinOpSpec::Conv {
                    input: cur_id,
                    weight: wv(co * cc * k * k, rng),
                    bias: wv(co, rng),
                    bias_err: vec![0.0; co],
                    weight_rel_err: 0.0,
                    kernel: (co, cc, k, k),
                    stride: (stride, stride),
                    pads: (pad, pad, pad, pad),
                    ishape: (cc, ch, cw),
                    oshape: (co, oh, ow),
                };
                ops.push(conv);
                let conv_id = ops.len();
                ops.push(TwinOpSpec::Relu { input: conv_id });
                cur_id = ops.len();
                cc = co;
                ch = oh;
                cw = ow;
            }
        }
        // Head: Flatten -> Gemm1 -> Relu -> Gemm2.
        let flat = cc * ch * cw;
        ops.push(TwinOpSpec::Flatten { input: cur_id });
        let flat_id = ops.len();
        let n_y: usize = rng.random_range(3usize..8); // 3..=7
        let n_out: usize = rng.random_range(3usize..6); // 3..=5
        ops.push(TwinOpSpec::Gemm {
            input: flat_id,
            weight: wv(n_y * flat, rng),
            bias: wv(n_y, rng),
            shape: (n_y, flat),
        });
        let g1_id = ops.len();
        ops.push(TwinOpSpec::Relu { input: g1_id });
        let hr_id = ops.len();
        ops.push(TwinOpSpec::Gemm {
            input: hr_id,
            weight: wv(n_out * n_y, rng),
            bias: wv(n_out, rng),
            shape: (n_out, n_y),
        });
        let n_in = c0 * hw * hw;
        (TwinSpec { n_in, ops }, (c0, hw, hw))
    }

    /// THE ADVERSARIAL ENCLOSURE ORACLE. Random-dense: broad random valid nets
    /// across depths/widths, both Outward and Parity, uniform-in-box + all corners
    /// (small inputs) + random corners + center. For every (net, sample, class):
    /// assert `dj[k] <= (Y_t(x) - Y_j(x)) + tol`. A single overshoot in Outward is
    /// a soundness break (false-UNSAT); we record the seed/net/x/dj/true-margin.
    #[test]
    fn verifier_random_dense_enclosure_oracle() {
        const BASE_SEED: u64 = 0x4D41_5247_494E_524Fu64; // "MARGINRO"
        const NETS_PER_MODE: u64 = 45;

        let mut total_checks: u64 = 0;
        let mut nets_tested: u64 = 0;
        let mut violations_outward: u64 = 0;
        let mut violations_parity: u64 = 0;
        let mut worst_over_outward = f64::NEG_INFINITY; // max(dj - margin) in Outward
        let mut worst_over_parity = f64::NEG_INFINITY;
        let mut first_ce: Option<String> = None;

        for (mode_tag, mode) in [RoundMode::Outward, RoundMode::Parity]
            .into_iter()
            .enumerate()
        {
            let outward = mode.outward();
            for idx in 0..NETS_PER_MODE {
                // Seeded, deterministic, varied by loop index + mode. No entropy.
                let seed =
                    BASE_SEED ^ ((mode_tag as u64) << 40) ^ idx.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut rng = StdRng::seed_from_u64(seed);
                let (spec, (c0, h, w)) = rand_net(&mut rng);
                let n_in = c0 * h * w;
                let net = match TwinNet::compile(&spec) {
                    Ok(n) => n,
                    Err(_) => continue, // structurally rejected: not our concern
                };
                let (_, _, (n_out, _)) = net.gemm2();

                // Random input box. Vary the radius regime widely (tiny boxes make
                // dj tight against the margin — the stress regime — while larger
                // boxes exercise the composition over a real polytope).
                let base_rad = *[0.001f64, 0.01, 0.05, 0.2, 0.5]
                    .get(rng.random_range(0usize..5))
                    .unwrap();
                let mut lo = vec![0.0f64; n_in];
                let mut hi = vec![0.0f64; n_in];
                for i in 0..n_in {
                    let center: f64 = rng.random_range(-1.0..1.0);
                    let r: f64 = base_rad * rng.random_range(0.25..1.75);
                    lo[i] = center - r;
                    hi[i] = center + r;
                }
                let mid: Vec<f64> = (0..n_in).map(|i| f64::midpoint(lo[i], hi[i])).collect();

                // Robustness spec. Half the nets: t = argmax at the box center
                // (the TIGHT regime where dj hugs the true margin — most likely to
                // expose an overshoot). Other half: a random true class.
                let t: usize = if idx % 2 == 0 {
                    let s = ref_forward(&spec, &mid);
                    (0..n_out).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap()
                } else {
                    rng.random_range(0usize..n_out)
                };
                let adv: Vec<usize> = (0..n_out).filter(|&j| j != t).collect();

                let root = match RootGates::build(&net, &lo, &hi, mode, None) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let eng = BackwardEngine::new(&net, &root);
                let re = match root_eval(&eng, &net, t, &adv) {
                    Ok(r) => r,
                    Err(_) => continue, // fail-closed error: sound (Unknown), skip
                };
                assert_eq!(re.dj.len(), adv.len(), "dj length must match adv");
                nets_tested += 1;

                // ---- Build the sample set ----
                let mut samples: Vec<Vec<f64>> = Vec::new();
                samples.push(mid.clone()); // center
                                           // Uniform-in-box interior samples.
                for _ in 0..300 {
                    let x: Vec<f64> = (0..n_in)
                        .map(|i| {
                            let u: f64 = rng.random_range(0.0..1.0);
                            lo[i] + u * (hi[i] - lo[i])
                        })
                        .collect();
                    samples.push(x);
                }
                // Random corners (each coord independently lo or hi).
                for _ in 0..96 {
                    let x: Vec<f64> = (0..n_in)
                        .map(|i| {
                            if rng.random_range(0usize..2) == 0 {
                                lo[i]
                            } else {
                                hi[i]
                            }
                        })
                        .collect();
                    samples.push(x);
                }
                // ALL corners for small inputs (exhaustive box vertices).
                if n_in <= 12 {
                    for mask in 0u32..(1u32 << n_in) {
                        let x: Vec<f64> = (0..n_in)
                            .map(|i| if (mask >> i) & 1 == 0 { lo[i] } else { hi[i] })
                            .collect();
                        samples.push(x);
                    }
                }

                // ---- Check enclosure at every sample & class ----
                for x in &samples {
                    let out = ref_forward(&spec, x);
                    for (k, &j) in adv.iter().enumerate() {
                        let margin = out[t] - out[j];
                        let over = re.dj[k] - margin;
                        total_checks += 1;
                        // Tolerance: Outward is the verdict mode and rounds toward
                        // -inf, so the bound must be <= the EXACT real margin. We
                        // allow only for the REFERENCE forward's own f64 rounding
                        // (a few ulps on the accumulated margin) — a hard 1e-9 plus
                        // a tiny relative term. A real sign/composition/missing-
                        // error bug overshoots by orders more than this. Parity is
                        // not verdict-bearing (no directed rounding); give it a
                        // looser relative tolerance to avoid rounding-noise flags.
                        let tol = if outward {
                            1e-9 + 1e-11 * margin.abs()
                        } else {
                            1e-6 * (1.0 + margin.abs())
                        };
                        if outward {
                            worst_over_outward = worst_over_outward.max(over);
                        } else {
                            worst_over_parity = worst_over_parity.max(over);
                        }
                        if over > tol {
                            if outward {
                                violations_outward += 1;
                            } else {
                                violations_parity += 1;
                            }
                            if first_ce.is_none() && outward {
                                first_ce = Some(format!(
                                "UNSOUND {mode:?}: seed={seed:#x} shape=({c0},{h},{w}) n_in={n_in} \
                                 t={t} j={j} (adv idx {k}) dj={dj:.17e} true_margin={margin:.17e} \
                                 overshoot={over:.3e} box_rad~{base_rad} x={x:?}",
                                dj = re.dj[k]
                            ));
                            }
                        }
                    }
                }
            }
        }

        eprintln!(
            "[enclosure-oracle] nets_tested={nets_tested} total_checks={total_checks} \
         violations_outward={violations_outward} violations_parity={violations_parity} \
         worst_overshoot_outward={worst_over_outward:.3e} \
         worst_overshoot_parity={worst_over_parity:.3e}"
        );
        if let Some(ce) = &first_ce {
            eprintln!("[enclosure-oracle] COUNTEREXAMPLE: {ce}");
        }

        // Sanity: the harness must actually have exercised a meaningful volume.
        assert!(
            total_checks >= 50_000,
            "oracle too small: {total_checks} checks"
        );
        assert!(nets_tested >= 60, "too few nets exercised: {nets_tested}");

        // THE SOUNDNESS ASSERTION: no Outward overshoot of a true feasible margin.
        assert_eq!(
            violations_outward,
            0,
            "MARGIN-ROW UNSOUND (Outward false-UNSAT risk): {}",
            first_ce.unwrap_or_default()
        );
    }
}

#[cfg(test)]
mod enclosure_oracle_near_cancel {
    use super::super::bab::root_eval;
    use super::*;

    // =================== ENCLOSURE ORACLE (adversarial, near-cancellation) ======
    //
    // Claim under test: for the ROOT domain, `RootEval.dj[k]` (= max(m1, m2v,
    // direct) from `root_eval`, RoundMode::Outward) is a SOUND lower bound on
    //   min_{x in [lo,hi]}  Y_t(x) - Y_{adv[k]}(x).
    // If dj[k] EVER exceeds a true feasible margin, the lane is UNSOUND (false-UNSAT
    // risk). This oracle tries to break that: nets engineered for SMALL true
    // margins (near-cancellation) where an insufficiently-outward rounding would
    // most likely overstate dj, plus dense corner/boundary/interior sampling.
    //
    // The reference forward eval below is FULLY INDEPENDENT of the interval /
    // tableau machinery: it reads the raw f64 TwinSpec parameters and does a naive
    // conv/relu/add/flatten/gemm point evaluation. All conv error budgets are set
    // to EXACTLY ZERO (see `zero_conv_err`), so the stored parameters ARE the true
    // net — the certified bound gets no legitimate slack to hide behind, making the
    // enclosure check as tight as possible.

    /// Independent exact-f64 forward eval of a [`TwinSpec`] at one point `x`.
    /// Returns the final logit vector. Uses ONLY raw spec parameters (no TwinNet,
    /// no conv kernels, no interval code).
    fn ref_forward(spec: &TwinSpec, x: &[f64]) -> Vec<f64> {
        let mut tensors: Vec<Vec<f64>> = Vec::with_capacity(spec.ops.len() + 1);
        tensors.push(x.to_vec());
        for op in &spec.ops {
            let out = match op {
                TwinOpSpec::ConvTranspose { .. } | TwinOpSpec::ChannelAffine { .. } => {
                    unreachable!("oracle net generators never emit ConvTranspose/ChannelAffine")
                }
                TwinOpSpec::Conv {
                    input,
                    weight,
                    bias,
                    kernel,
                    stride,
                    pads,
                    ishape,
                    oshape,
                    ..
                } => {
                    let (co, ci, kh, kw) = *kernel;
                    let (_ic, ih, iw) = *ishape;
                    let (_oc, oh, ow) = *oshape;
                    let src = &tensors[*input];
                    let mut out = vec![0.0f64; co * oh * ow];
                    for oc in 0..co {
                        for oy in 0..oh {
                            for ox in 0..ow {
                                let mut acc = bias[oc];
                                for c in 0..ci {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let iy_raw = oy * stride.0 + ky;
                                            let ix_raw = ox * stride.1 + kx;
                                            if iy_raw < pads.0 || ix_raw < pads.1 {
                                                continue;
                                            }
                                            let iy = iy_raw - pads.0;
                                            let ix = ix_raw - pads.1;
                                            if iy >= ih || ix >= iw {
                                                continue;
                                            }
                                            let w = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            let v = src[c * ih * iw + iy * iw + ix];
                                            acc += w * v;
                                        }
                                    }
                                }
                                out[oc * oh * ow + oy * ow + ox] = acc;
                            }
                        }
                    }
                    out
                }
                TwinOpSpec::Relu { input } => tensors[*input].iter().map(|v| v.max(0.0)).collect(),
                TwinOpSpec::Add { lhs, rhs } => tensors[*lhs]
                    .iter()
                    .zip(&tensors[*rhs])
                    .map(|(a, b)| a + b)
                    .collect(),
                TwinOpSpec::Flatten { input } => tensors[*input].clone(),
                TwinOpSpec::Gemm {
                    input,
                    weight,
                    bias,
                    shape,
                } => {
                    let (no, ni) = *shape;
                    let src = &tensors[*input];
                    let mut out = vec![0.0f64; no];
                    for o in 0..no {
                        let mut acc = bias[o];
                        for i in 0..ni {
                            acc += weight[o * ni + i] * src[i];
                        }
                        out[o] = acc;
                    }
                    out
                }
            };
            tensors.push(out);
        }
        tensors.pop().expect("at least one op")
    }

    /// Force every conv error budget to zero: the stored params become the exact
    /// true net, so the reference eval and the certified machinery describe the
    /// SAME function — no error slack for the bound to hide in.
    fn zero_conv_err(spec: &mut TwinSpec) {
        for op in &mut spec.ops {
            if let TwinOpSpec::Conv {
                bias_err,
                weight_rel_err,
                ..
            } = op
            {
                for e in bias_err.iter_mut() {
                    *e = 0.0;
                }
                *weight_rel_err = 0.0;
            }
        }
    }

    /// Rewrite the final Gemm so output row `jc` mirrors row `tc` up to a tiny
    /// per-entry perturbation. This drives `Y_tc - Y_jc` to within ~epsilon of zero
    /// across the WHOLE box (near-cancellation) — the regime where any
    /// insufficiently-outward rounding of dj is most likely to poke above the true
    /// margin.
    fn near_cancel_last_gemm(
        spec: &mut TwinSpec,
        tc: usize,
        jc: usize,
        eps: f64,
        rng: &mut StdRng,
    ) {
        if let Some(TwinOpSpec::Gemm {
            weight,
            bias,
            shape,
            ..
        }) = spec.ops.last_mut()
        {
            let (_no, ni) = *shape;
            for i in 0..ni {
                let base = weight[tc * ni + i];
                weight[jc * ni + i] = base + rng.random_range(-eps..eps);
            }
            bias[jc] = bias[tc] + rng.random_range(-eps..eps);
        }
    }

    /// Draw one sample point in `[lo,hi]` under one of several regimes chosen by
    /// `kind`: 0 center, 1 all-lo, 2 all-hi, 3 random corner (each coord lo|hi),
    /// 4 random interior, 5 boundary-mix (each coord lo|hi|random).
    fn draw_point(rng: &mut StdRng, lo: &[f64], hi: &[f64], kind: usize) -> Vec<f64> {
        let n = lo.len();
        (0..n)
            .map(|i| match kind {
                0 => f64::midpoint(lo[i], hi[i]),
                1 => lo[i],
                2 => hi[i],
                3 => {
                    if rng.random_range(0.0..1.0) < 0.5 {
                        lo[i]
                    } else {
                        hi[i]
                    }
                }
                4 => lo[i] + rng.random_range(0.0..1.0) * (hi[i] - lo[i]),
                _ => {
                    let r: f64 = rng.random_range(0.0..1.0);
                    if r < 0.4 {
                        lo[i]
                    } else if r < 0.8 {
                        hi[i]
                    } else {
                        lo[i] + rng.random_range(0.0..1.0) * (hi[i] - lo[i])
                    }
                }
            })
            .collect()
    }

    /// THE oracle. Generate many near-cancellation twin-nets over random boxes,
    /// compute the certified root `dj[k]` for each adversarial class, and assert it
    /// never exceeds the true feasible margin at ANY sampled point.
    #[test]
    fn enclosure_oracle_root_dj_lower_bounds_true_margin() {
        // Every check must satisfy dj[k] <= (Y_t - Y_j) + SLACK. SLACK is the
        // task-specified tolerance; a real overstatement must clear it to count.
        const SLACK: f64 = 1e-9;
        const N_NETS: usize = 240;
        // Per-net sample budget across the regimes (center/lo/hi are singletons).
        const N_CORNER: usize = 160;
        const N_INTERIOR: usize = 220;
        const N_BMIX: usize = 160;

        let mut total_checks: u64 = 0;
        let mut violations: u64 = 0;
        // Closest approach of the bound to a true margin: max over ALL checks of
        // (dj - margin). Negative => bound always strictly below the sampled
        // margin; a value near 0^- proves the enclosure is actually being tested at
        // the boundary (non-vacuous), not merely passing by a wide slack.
        let mut worst_over: f64 = f64::NEG_INFINITY;
        let mut tight_checks: u64 = 0; // checks where dj within 1e-6 below margin
        let mut counterexample: Option<String> = None;

        for net_i in 0..N_NETS {
            // Deterministic, seed varies by loop index (no wall-clock entropy).
            let mut rng = StdRng::seed_from_u64(
                0x00E1_C105_0000_0000 ^ (net_i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            // Vary net scale and box across nets for regime diversity.
            let scale = 0.25 + rng.random_range(0.0..1.0) * 0.9;
            let mut spec = tiny_spec(&mut rng, scale);
            zero_conv_err(&mut spec);

            // Half the nets get an explicit near-cancellation pair (t=0 vs j=1) so
            // the true margin hugs zero across the whole box.
            let near_cancel = net_i % 2 == 0;
            if near_cancel {
                let eps = 10f64.powf(-6.0 - rng.random_range(0.0..1.0) * 3.0); // 1e-6..1e-9
                near_cancel_last_gemm(&mut spec, 0, 1, eps, &mut rng);
            }

            // Random asymmetric box.
            let n_in = spec.n_in;
            let mut lo = vec![0.0f64; n_in];
            let mut hi = vec![0.0f64; n_in];
            for i in 0..n_in {
                let a: f64 = rng.random_range(-0.35..0.35);
                let r: f64 = 0.03 + rng.random_range(0.0..1.0) * 0.45;
                lo[i] = a - r;
                hi[i] = a + r;
            }

            // Choose t and the adversarial set. For near-cancel nets force t=0 so
            // class 1 is the razor-thin competitor; otherwise t = argmax at center.
            let center: Vec<f64> = (0..n_in).map(|i| f64::midpoint(lo[i], hi[i])).collect();
            let out_c = ref_forward(&spec, &center);
            let n_out = out_c.len();
            let t = if near_cancel {
                0usize
            } else {
                (0..n_out)
                    .max_by(|&a, &b| out_c[a].total_cmp(&out_c[b]))
                    .expect("classes")
            };
            let adv: Vec<usize> = (0..n_out).filter(|&o| o != t).collect();

            // Certified root bounds (the thing under test), Outward mode.
            let net = TwinNet::compile(&spec).expect("compile");
            let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            let eng = BackwardEngine::new(&net, &root);
            let re = root_eval(&eng, &net, t, &adv).expect("root_eval");
            assert_eq!(re.dj.len(), adv.len(), "dj length must match adv");

            // Build the sample set: singletons + corners + interior + boundary-mix.
            let mut points: Vec<Vec<f64>> = Vec::new();
            for kind in 0..3 {
                points.push(draw_point(&mut rng, &lo, &hi, kind));
            }
            for _ in 0..N_CORNER {
                points.push(draw_point(&mut rng, &lo, &hi, 3));
            }
            for _ in 0..N_INTERIOR {
                points.push(draw_point(&mut rng, &lo, &hi, 4));
            }
            for _ in 0..N_BMIX {
                points.push(draw_point(&mut rng, &lo, &hi, 5));
            }

            for x in &points {
                // INDEPENDENT true logits at this feasible point.
                let out = ref_forward(&spec, x);
                for (k, &j) in adv.iter().enumerate() {
                    let margin = out[t] - out[j];
                    let dj = re.dj[k];
                    total_checks += 1;
                    if !dj.is_finite() {
                        // -inf/NaN bound cannot overstate a finite margin; skip.
                        continue;
                    }
                    let over = dj - margin; // > SLACK  <=>  unsound overstatement
                    if over > worst_over {
                        worst_over = over;
                    }
                    if over > -1e-6 {
                        tight_checks += 1;
                    }
                    if over > SLACK {
                        violations += 1;
                        if counterexample.is_none() {
                            counterexample = Some(format!(
                                "net_i={net_i} near_cancel={near_cancel} scale={scale:.4} \
                             t={t} j={j} (k={k}) dj={dj:.17e} true_margin={margin:.17e} \
                             over=+{over:.3e} n_in={n_in} box0=[{:.4},{:.4}]",
                                lo[0], hi[0]
                            ));
                        }
                    }
                }
            }
        }

        eprintln!(
            "[enclosure-oracle] nets={N_NETS} total_checks={total_checks} \
         violations={violations} closest_approach(dj-margin)={worst_over:.3e} \
         tight_checks(within_1e-6)={tight_checks}"
        );
        if let Some(ce) = &counterexample {
            eprintln!("[enclosure-oracle] COUNTEREXAMPLE: {ce}");
        }
        assert!(
            total_checks > 50_000,
            "oracle too weak: only {total_checks} checks (want tens of thousands)"
        );
        assert_eq!(
            violations,
            0,
            "UNSOUNDNESS: root dj exceeded a true feasible margin in {violations} check(s); \
         first: {}",
            counterexample.unwrap_or_default()
        );
    }
}

#[cfg(test)]
mod enclosure_oracle_deep {
    use super::super::bab::root_eval;
    use super::*;

    // ===================== DEEP-RESIDUAL ADVERSARIAL ENCLOSURE ORACLE ==========
    //
    // Purpose: try to PROVE the certified per-class root lower bound `dj[k]`
    // (RootEval.dj from `root_eval`, = max(m1, m2v, direct)) UNSOUND. `dj[k]` is
    // CLAIMED to be a sound lower bound on `min_{x in box}(Y_t(x) - Y_{adv[k]}(x))`
    // in RoundMode::Outward. If `dj[k]` ever exceeds a TRUE feasible margin at any
    // concrete point x in the box, the twin-wall lane is a false-UNSAT risk.
    //
    // Strategy (deep-residual-adversarial): synthesize deep residual stacks with
    // adversarially-scaled weights (large magnitudes + alternating-sign patterns
    // that force catastrophic cancellation in the forward tableau and the CROWN
    // backward), plus wide ("Gemm-heavy") heads. For each net:
    //   (1) build valid TwinSpec/TwinNet with a SEEDED deterministic RNG,
    //   (2) pick a random input box [lo,hi] and a robustness spec (t, adv),
    //   (3) compute the certified dj[k] via `root_eval`,
    //   (4) evaluate an INDEPENDENT exact-f64 reference forward (over the raw
    //       TwinSpec, NOT the interval machinery, with compensated Dot2 dot
    //       products so cancellation does not corrupt the reference),
    //   (5) Monte-Carlo sample many x (center + all-lo/all-hi + random corners +
    //       interior) and ASSERT dj[k] <= (Y_t(x) - Y_j(x)) + 1e-9 for every
    //       sample and class.
    // Any dj[k] exceeding a true feasible margin is recorded (seed/net/x/dj/margin)
    // and FAILS the test.

    /// `two_sum`: exact `a + b = s + e` (Knuth/Møller, no branch).
    #[inline]
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        let e = (a - (s - bb)) + (b - bb);
        (s, e)
    }

    /// `two_prod`: exact `a * b = p + e` via a fused multiply-add.
    #[inline]
    fn two_prod(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        let e = a.mul_add(b, -p);
        (p, e)
    }

    /// Compensated (Ogita-Rump-Oishi Dot2/Sum2) accumulator: computes sums and
    /// dot products with error ~ eps^2 relative to the result even under heavy
    /// cancellation, so the reference margin is trustworthy well below the 1e-9
    /// assertion cushion. This is the INDEPENDENT arithmetic — it shares no code
    /// with the certified interval engine.
    #[derive(Clone, Copy)]
    struct Acc {
        s: f64,
        c: f64,
    }

    impl Acc {
        #[inline]
        fn new(init: f64) -> Self {
            Self { s: init, c: 0.0 }
        }
        #[inline]
        fn add_prod(&mut self, a: f64, b: f64) {
            let (p, ep) = two_prod(a, b);
            let (s, es) = two_sum(self.s, p);
            self.s = s;
            self.c += ep + es;
        }
        #[inline]
        fn value(&self) -> f64 {
            self.s + self.c
        }
    }

    /// INDEPENDENT exact-f64 reference forward evaluation over the raw TwinSpec.
    /// Returns the full class-logit vector `Y(x)` (through the final Gemm). Does
    /// NOT touch TwinNet compilation, the root tableau, or the CROWN engine — it is
    /// the ground truth the certified bound must never exceed. Uses compensated
    /// Dot2 for every conv/gemm reduction so catastrophic cancellation in the
    /// weights cannot corrupt the reference.
    fn forward_ref_logits(spec: &TwinSpec, x: &[f64]) -> Vec<f64> {
        let mut tens: Vec<Vec<f64>> = Vec::with_capacity(spec.ops.len() + 1);
        tens.push(x.to_vec());
        for op in &spec.ops {
            let out: Vec<f64> = match op {
                TwinOpSpec::ConvTranspose { .. } | TwinOpSpec::ChannelAffine { .. } => {
                    unreachable!("oracle net generators never emit ConvTranspose/ChannelAffine")
                }
                TwinOpSpec::Conv {
                    input,
                    weight,
                    bias,
                    kernel,
                    stride,
                    pads,
                    ishape,
                    oshape,
                    ..
                } => {
                    let (co, ci, kh, kw) = *kernel;
                    let (_ic, ih, iw) = *ishape;
                    let (_oc, oh, ow) = *oshape;
                    let (sh, sw) = *stride;
                    let (pt, pl, _pb, _pr) = *pads;
                    let src = &tens[*input];
                    let mut o = vec![0.0f64; co * oh * ow];
                    for oc in 0..co {
                        for oy in 0..oh {
                            for ox in 0..ow {
                                let mut acc = Acc::new(bias[oc]);
                                for c in 0..ci {
                                    for ky in 0..kh {
                                        for kx in 0..kw {
                                            let iy = (oy * sh + ky) as isize - pt as isize;
                                            let ix = (ox * sw + kx) as isize - pl as isize;
                                            if iy < 0
                                                || iy as usize >= ih
                                                || ix < 0
                                                || ix as usize >= iw
                                            {
                                                continue;
                                            }
                                            let iidx = c * ih * iw + iy as usize * iw + ix as usize;
                                            let widx = ((oc * ci + c) * kh + ky) * kw + kx;
                                            acc.add_prod(weight[widx], src[iidx]);
                                        }
                                    }
                                }
                                o[oc * oh * ow + oy * ow + ox] = acc.value();
                            }
                        }
                    }
                    o
                }
                TwinOpSpec::Relu { input } => tens[*input].iter().map(|v| v.max(0.0)).collect(),
                TwinOpSpec::Add { lhs, rhs } => tens[*lhs]
                    .iter()
                    .zip(&tens[*rhs])
                    .map(|(a, b)| a + b)
                    .collect(),
                TwinOpSpec::Flatten { input } => tens[*input].clone(),
                TwinOpSpec::Gemm {
                    input,
                    weight,
                    bias,
                    shape,
                } => {
                    let (no, ni) = *shape;
                    let src = &tens[*input];
                    let mut o = vec![0.0f64; no];
                    for oo in 0..no {
                        let mut acc = Acc::new(bias[oo]);
                        let wrow = &weight[oo * ni..(oo + 1) * ni];
                        for (i, &w) in wrow.iter().enumerate() {
                            acc.add_prod(w, src[i]);
                        }
                        o[oo] = acc.value();
                    }
                    o
                }
            };
            tens.push(out);
        }
        tens.pop().expect("network produced no output tensor")
    }

    /// Per-net configuration drawn deterministically from the seeded RNG.
    struct DeepCfg {
        c: usize,     // channels (constant through the residual trunk)
        hw: usize,    // spatial H = W
        depth: usize, // residual blocks
        n_y: usize,   // head width (Gemm-heavy tail when large)
        n_out: usize, // classes
        scale: f64,   // weight magnitude
        cancel: bool, // alternating-sign catastrophic-cancellation kernels
    }

    /// Build a deep residual twin-wall net honoring the compile contract
    /// (... Flatten -> Gemm1 -> Relu -> Gemm2 tail, >=1 trunk relu, exactly 2
    /// gemms, no gemm before the last trunk relu). Weights are adversarially
    /// scaled; `cancel` biases each conv toward sign-alternating taps and
    /// channel-opposed kernels to maximize cancellation in the certified tableau.
    /// Adversarial conv weight generator: `co*ci*3*3`, optionally sign-alternating
    /// by tap and opposed by output channel to force cancellation.
    fn conv_w(rng: &mut StdRng, co: usize, ci: usize, scale: f64, cancel: bool) -> Vec<f64> {
        let (kh, kw) = (3usize, 3usize);
        let mut w = vec![0.0f64; co * ci * kh * kw];
        for oc in 0..co {
            for c2 in 0..ci {
                for t in 0..(kh * kw) {
                    let base: f64 = rng.random_range(-scale..scale);
                    let v = if cancel {
                        let sgn = if (t + oc) % 2 == 0 { 1.0 } else { -1.0 };
                        sgn * (scale - base.abs())
                    } else {
                        base
                    };
                    w[((oc * ci + c2) * kh + t / kw) * kw + (t % kw)] = v;
                }
            }
        }
        w
    }

    /// Uniform `[-s, s]` vector.
    fn vecf(rng: &mut StdRng, n: usize, s: f64) -> Vec<f64> {
        (0..n).map(|_| rng.random_range(-s..s)).collect()
    }

    fn build_deep_resnet(rng: &mut StdRng, cfg: &DeepCfg) -> TwinSpec {
        let (c, hw) = (cfg.c, cfg.hw);
        let spatial = hw * hw;
        let n_in = c * spatial;
        let ishape = (c, hw, hw);
        let mut ops: Vec<TwinOpSpec> = Vec::new();
        // Stem: Conv(C->C) -> Relu.
        ops.push(TwinOpSpec::Conv {
            input: 0,
            weight: conv_w(rng, c, c, cfg.scale, cfg.cancel),
            bias: vecf(rng, c, cfg.scale),
            bias_err: vec![0.0; c],
            weight_rel_err: 0.0,
            kernel: (c, c, 3, 3),
            stride: (1, 1),
            pads: (1, 1, 1, 1),
            ishape,
            oshape: ishape,
        }); // tensor 1
        ops.push(TwinOpSpec::Relu { input: 1 }); // tensor 2, trunk relu 0
        let mut cur = 2usize; // id of the current (C,H,W) trunk feature map
        for _ in 0..cfg.depth {
            let skip = cur;
            // Conv -> Relu -> Conv -> Add(skip) -> Relu.
            let a = ops.len() + 1; // tensor id produced by the next push
            ops.push(TwinOpSpec::Conv {
                input: cur,
                weight: conv_w(rng, c, c, cfg.scale, cfg.cancel),
                bias: vecf(rng, c, cfg.scale),
                bias_err: vec![0.0; c],
                weight_rel_err: 0.0,
                kernel: (c, c, 3, 3),
                stride: (1, 1),
                pads: (1, 1, 1, 1),
                ishape,
                oshape: ishape,
            });
            let b = ops.len() + 1;
            ops.push(TwinOpSpec::Relu { input: a });
            let d = ops.len() + 1;
            ops.push(TwinOpSpec::Conv {
                input: b,
                weight: conv_w(rng, c, c, cfg.scale, cfg.cancel),
                bias: vecf(rng, c, cfg.scale),
                bias_err: vec![0.0; c],
                weight_rel_err: 0.0,
                kernel: (c, c, 3, 3),
                stride: (1, 1),
                pads: (1, 1, 1, 1),
                ishape,
                oshape: ishape,
            });
            let e = ops.len() + 1;
            ops.push(TwinOpSpec::Add { lhs: d, rhs: skip });
            ops.push(TwinOpSpec::Relu { input: e }); // trunk relu
            cur = ops.len(); // tensor id of this relu's output
        }
        // Head: Flatten -> Gemm1 -> Relu -> Gemm2.
        let flat = ops.len() + 1;
        ops.push(TwinOpSpec::Flatten { input: cur });
        let g1 = ops.len() + 1;
        ops.push(TwinOpSpec::Gemm {
            input: flat,
            weight: vecf(rng, cfg.n_y * n_in, cfg.scale),
            bias: vecf(rng, cfg.n_y, cfg.scale),
            shape: (cfg.n_y, n_in),
        });
        let hr = ops.len() + 1;
        ops.push(TwinOpSpec::Relu { input: g1 });
        ops.push(TwinOpSpec::Gemm {
            input: hr,
            weight: vecf(rng, cfg.n_out * cfg.n_y, cfg.scale),
            bias: vecf(rng, cfg.n_out, cfg.scale),
            shape: (cfg.n_out, cfg.n_y),
        });
        TwinSpec { n_in, ops }
    }

    /// Draw one seeded deterministic RNG from a FIXED base array, varied by index.
    fn seeded_rng(idx: usize) -> StdRng {
        let mut seed = [0u8; 32];
        seed[0..8].copy_from_slice(&0xA5A5_1234_DEAD_BEEF_u64.to_le_bytes());
        seed[8..16].copy_from_slice(&(idx as u64).to_le_bytes());
        seed[16..24].copy_from_slice(&0x00C0_FFEE_5EED_1DEA_u64.to_le_bytes());
        seed[24..32].copy_from_slice(&((idx as u64).wrapping_mul(0x9E37_79B9)).to_le_bytes());
        StdRng::from_seed(seed)
    }

    #[test]
    fn verifier_deep_residual_enclosure_oracle() {
        const NETS: usize = 90;
        const SAMPLES: usize = 56;

        let mut total_checks: u64 = 0;
        let mut nets_tested: u64 = 0;
        let mut nets_skipped: u64 = 0;
        let mut violations: u64 = 0;
        // Tightest (smallest) slack = margin - dj observed; near 0 = the bound is
        // being pushed to its limit (best evidence the oracle has resolution).
        let mut min_slack = f64::INFINITY;
        let mut first_violation: Option<String> = None;

        for net_idx in 0..NETS {
            let mut rng = seeded_rng(net_idx);
            // Deterministic config draw.
            let c = *[2usize, 3].get(net_idx % 2).unwrap();
            let hw = 4usize;
            let depth = 2 + (net_idx % 4); // 2..=5 residual blocks (deep)
            let n_y = [8usize, 12, 20, 28][net_idx % 4]; // Gemm-heavy tails
            let n_out = 4 + (net_idx % 3); // 4..=6 classes
            let scale = [0.35f64, 0.7, 1.4, 2.6][net_idx % 4];
            let cancel = net_idx % 3 != 0; // 2/3 of nets use cancellation kernels
            let radius = [1e-3f64, 8e-3, 4e-2, 0.2, 0.45][net_idx % 5];
            let cfg = DeepCfg {
                c,
                hw,
                depth,
                n_y,
                n_out,
                scale,
                cancel,
            };
            let spec = build_deep_resnet(&mut rng, &cfg);
            let net = match TwinNet::compile(&spec) {
                Ok(n) => n,
                Err(_) => {
                    nets_skipped += 1;
                    continue;
                }
            };
            let n_in = spec.n_in;
            // Random input box [lo, hi] = center +/- radius.
            let center: Vec<f64> = (0..n_in).map(|_| rng.random_range(-0.6..0.6)).collect();
            let lo: Vec<f64> = center.iter().map(|c0| c0 - radius).collect();
            let hi: Vec<f64> = center.iter().map(|c0| c0 + radius).collect();

            // Certified root gates + engine (Outward = the ONLY verdict-grade mode).
            let root = match RootGates::build(&net, &lo, &hi, RoundMode::Outward, None) {
                Ok(r) => r,
                Err(_) => {
                    // Fail-closed build: nothing to certify, safe. Skip.
                    nets_skipped += 1;
                    continue;
                }
            };
            let eng = BackwardEngine::new(&net, &root);

            // Robustness spec: t = argmax at the box center (tight positive margins,
            // the regime where dj is closest to the true margin — best hunting
            // ground for a false-UNSAT), plus a second spec with a random t so
            // negative-margin cases are exercised too.
            let center_logits = forward_ref_logits(&spec, &center);
            let t_argmax = (0..n_out)
                .max_by(|&a, &b| center_logits[a].total_cmp(&center_logits[b]))
                .unwrap();
            let t_rand = (t_argmax + 1 + (net_idx % (n_out - 1))) % n_out;

            let instances: [(usize, Vec<usize>); 2] = [
                (t_argmax, (0..n_out).filter(|&o| o != t_argmax).collect()),
                (t_rand, (0..n_out).filter(|&o| o != t_rand).collect()),
            ];

            // Pre-generate the sample points once per net (same set for both specs).
            let mut xs: Vec<Vec<f64>> = Vec::with_capacity(SAMPLES);
            xs.push(center.clone()); // center
            xs.push(lo.clone()); // all-lo corner
            xs.push(hi.clone()); // all-hi corner
            while xs.len() < SAMPLES {
                let is_corner = xs.len().is_multiple_of(3);
                let x: Vec<f64> = (0..n_in)
                    .map(|i| {
                        if is_corner {
                            if rng.random_range(0.0..1.0) < 0.5 {
                                lo[i]
                            } else {
                                hi[i]
                            }
                        } else {
                            let u: f64 = rng.random_range(-1.0..1.0);
                            (center[i] + u * radius).clamp(lo[i], hi[i])
                        }
                    })
                    .collect();
                xs.push(x);
            }

            let mut net_had_finite_spec = false;
            for (t, adv) in &instances {
                let re = match root_eval(&eng, &net, *t, adv) {
                    Ok(re) => re,
                    Err(_) => continue, // fail-closed root pass: safe, nothing certified
                };
                if re.dj.iter().any(|v| !v.is_finite()) {
                    continue;
                }
                net_had_finite_spec = true;
                // Ground-truth margins per sample per class, then enclosure check.
                for x in &xs {
                    let logits = forward_ref_logits(&spec, x);
                    for (k, &j) in adv.iter().enumerate() {
                        let margin = logits[*t] - logits[j];
                        let dj = re.dj[k];
                        let slack = margin - dj;
                        if slack < min_slack {
                            min_slack = slack;
                        }
                        total_checks += 1;
                        // Soundness contract: dj is a lower bound on the box-wide min
                        // margin, so dj <= margin(x) for EVERY feasible x. Tolerance
                        // 1e-9 absorbs the reference's residual (Dot2) error only.
                        if dj > margin + 1e-9 {
                            violations += 1;
                            if first_violation.is_none() {
                                first_violation = Some(format!(
                                    "net_idx={net_idx} cfg(c={c},hw={hw},depth={depth},n_y={n_y},\
                                 n_out={n_out},scale={scale},cancel={cancel},radius={radius:e}) \
                                 t={t} j={j} (adv pos {k}): dj={dj:.17e} > margin={margin:.17e} \
                                 (excess={:.3e}); Y_t={:.17e} Y_j={:.17e}",
                                    dj - margin,
                                    logits[*t],
                                    logits[j],
                                ));
                            }
                        }
                    }
                }
            }
            if net_had_finite_spec {
                nets_tested += 1;
            } else {
                nets_skipped += 1;
            }
        }

        eprintln!(
            "[deep-resnet oracle] nets_tested={nets_tested} nets_skipped={nets_skipped} \
         total_checks={total_checks} violations={violations} min_slack(margin-dj)={min_slack:.3e}"
        );
        assert!(
            total_checks >= 10_000,
            "insufficient coverage: only {total_checks} enclosure checks ran \
         (nets_tested={nets_tested}, nets_skipped={nets_skipped})"
        );
        assert_eq!(
            violations,
            0,
            "UNSOUNDNESS: certified root bound dj exceeded a TRUE feasible margin. \
         First witness: {}",
            first_violation.unwrap_or_default()
        );
    }
    // =================== END DEEP-RESIDUAL ADVERSARIAL ENCLOSURE ORACLE =========
}
