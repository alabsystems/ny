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
    classwise_runtime_policy_for_test, classwise_schedule, BabConfig, BabStats, ClassBabStats,
    FrontierExecutionEvidence, MarginRowBab, MarginRowOutcome, PolicyExecutionEvidence,
};
use super::bounds::{
    compose_viay, head_gates, head_variant, margin_seed, per_class_direct, row_dots, variant_state,
    MarginBatch, YBox,
};
use super::engine::{
    domain_gates, domain_gates_acc, BackwardEngine, DomainGates, Exc, Exceptions, LaneDir,
    RowDomainGateBlock, Seed as DomainStackSeed,
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
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (5, 36),
            }, // t7 (y)
            TwinOpSpec::Relu { input: 7 },      // t8 (head relu)
            TwinOpSpec::Gemm {
                input: 8,
                weight: w(4 * 5),
                bias: w(4),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
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
fn root_precision_receipt_binds_real_dispatch_and_logical_store_sites() {
    let mut rng = StdRng::seed_from_u64(0xF32_E71D);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let build = |mode, root_f32, tableau_f32| {
        RootGates::build_prec_store(&net, &lo, &hi, mode, None, root_f32, tableau_f32)
            .expect("root")
    };

    let assert_common = |root: &RootGates, root_f32, tableau_f32| {
        assert_eq!(
            root.root_precision_policy,
            super::RootPrecisionRuntimePolicy {
                root_f32,
                tableau_f32,
                err_lane: super::ErrLane::PerCol,
                ibp_seed: false,
                box_source: super::RootBoxSource::Tableau,
            }
        );
        let evidence = root.root_precision_execution;
        assert_eq!(evidence.configured_root_f32, root_f32);
        assert_eq!(evidence.configured_tableau_f32, tableau_f32);
        assert!(!evidence.policy_mismatch);
        evidence
    };

    // The completed tiny root has two conv compute sites and four logical
    // tableau stores: two Conv, one Add, and the nonterminal ReLU. Flatten
    // aliases storage and the terminal ReLU only concretizes the final box.
    let off = assert_common(&build(RoundMode::Outward, false, false), false, false);
    assert_eq!(
        (
            off.eligible_conv_ops,
            off.root_f32_conv_ops,
            off.root_f64_conv_ops,
        ),
        (2, 0, 2)
    );
    assert_eq!(
        (
            off.eligible_store_ops,
            off.tableau_f32_store_ops,
            off.tableau_f64_store_ops,
        ),
        (4, 0, 4)
    );
    assert!(!off.tableau_refused);

    let compute = assert_common(&build(RoundMode::Outward, true, false), true, false);
    assert_eq!(
        (compute.root_f32_conv_ops, compute.root_f64_conv_ops),
        (2, 0)
    );
    assert_eq!(
        (compute.tableau_f32_store_ops, compute.tableau_f64_store_ops,),
        (0, 4)
    );
    assert!(!compute.tableau_refused);

    let both = assert_common(&build(RoundMode::Outward, true, true), true, true);
    assert_eq!((both.root_f32_conv_ops, both.root_f64_conv_ops), (2, 0));
    assert_eq!(
        (both.tableau_f32_store_ops, both.tableau_f64_store_ops),
        (4, 0)
    );
    assert!(!both.tableau_refused);

    let refused = assert_common(&build(RoundMode::Outward, false, true), false, true);
    assert_eq!(
        (refused.root_f32_conv_ops, refused.root_f64_conv_ops),
        (0, 2)
    );
    assert_eq!(
        (refused.tableau_f32_store_ops, refused.tableau_f64_store_ops,),
        (0, 4)
    );
    assert!(refused.tableau_refused);

    let parity = assert_common(&build(RoundMode::Parity, true, true), true, true);
    assert!(!parity.outward_eligible);
    assert_eq!(parity.eligible_conv_ops, 0);
    assert_eq!(parity.eligible_store_ops, 0);
    assert!(parity.tableau_refused);
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
    let mut outward_f32 = false;
    for mode in [RoundMode::Parity, RoundMode::Outward] {
        let root = RootGates::build(&net, &lo, &hi, mode, None).expect("root");
        if mode.outward() {
            outward_f32 = root.root_precision_execution.root_f32_conv_ops > 0;
        }
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
    for (p, o) in bounds_by_mode[0].iter().zip(&bounds_by_mode[1]) {
        assert!(
            o <= &(p + 1e-9),
            "outward {o} is ABOVE parity {p} — the certified bound must be the \
             conservative one"
        );
        let budget = if outward_f32 {
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
        MarginRowOutcome::Unsat(stats, _) => {
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
        MarginRowOutcome::Unsat(..) => panic!("MOAT VIOLATION: verified a falsified instance"),
        MarginRowOutcome::Unknown { .. } => {}
    }
}

/// #soundness-spine M2c MUTATION PIN: on a REAL lane tree (the first seeded
/// fixture that closes through expansions), the cover gate authorizes the
/// intact tree — so G2 does not false-trip on it either — and REFUSES the same
/// deterministic run with one recorded leaf dropped. A fixture that closes at
/// the root would make this vacuous, so the search fails loudly if no seed
/// needs a tree.
#[test]
fn cover_gate_refuses_a_real_tree_missing_one_leaf() {
    let cfg = || BabConfig {
        frontier: 1,
        frontier_contextual: 1,
        lru_cap: 64,
        max_expansions: 400,
        ..BabConfig::default()
    };
    for seed in 0..64u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let spec = tiny_spec(&mut rng, 0.35);
        let Ok(net) = TwinNet::compile(&spec) else {
            continue;
        };
        for width in [0.01, 0.02, 0.04] {
            let lo = vec![0.01; 72];
            let hi: Vec<f64> = lo.iter().map(|l| l + width).collect();
            let (t, _worst, adv) = tiny_targets(&net, &lo, &hi);
            let Ok(root) = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None) else {
                continue;
            };
            let MarginRowOutcome::Unsat(stats, token) =
                MarginRowBab::run(&net, &root, t, &adv, cfg())
            else {
                continue;
            };
            if stats.expansions == 0 {
                continue;
            }
            assert!(
                token.publishable(),
                "outer frame must speak for the instance"
            );
            assert_eq!(stats.ledger_ok, Some(true));
            super::bab::inject_cover_fault_drop_next_leaf();
            match MarginRowBab::run(&net, &root, t, &adv, cfg()) {
                MarginRowOutcome::Unsat(..) => {
                    panic!("MUTATION SURVIVED (seed {seed}): a tree missing a leaf was authorized")
                }
                MarginRowOutcome::Unknown { reason, stats } => {
                    assert!(reason.contains("cover gate refused"), "{reason}");
                    assert_eq!(stats.and_then(|s| s.ledger_ok), Some(false));
                }
            }
            return;
        }
    }
    panic!("no seeded fixture closed through a tree: the mutation pin is VOID");
}

#[test]
fn classwise_switch_and_schedule_are_bounded_and_fail_closed() {
    assert!(!super::margin_row_classwise_from_env(None));
    assert!(!super::margin_row_classwise_from_env(Some("0")));
    assert!(!super::margin_row_classwise_from_env(Some("invalid")));
    assert!(super::margin_row_classwise_from_env(Some("1")));
    assert!(super::margin_row_classwise_from_env(Some(" TRUE ")));
    assert!(super::margin_row_classwise_from_env(Some(" YeS ")));
    assert!(super::margin_row_classwise_from_env(Some("\tON\n")));

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
fn classwise_declaration_preserves_the_frozen_legacy_text_predicate() {
    use ny_levers::{decls::margin_row::MARGIN_ROW_CLASSWISE, LeverSet, RawLeverInputs};
    for (raw, expected) in [
        (None, false),
        (Some(""), false),
        (Some("0"), false),
        (Some("invalid"), false),
        (Some("1x"), false),
        (Some("1"), true),
        (Some(" TRUE "), true),
        (Some("YeS"), true),
        (Some("\tON\n"), true),
    ] {
        let frozen = LeverSet::freeze_with(ny_levers::all(), |name| {
            (name == MARGIN_ROW_CLASSWISE.name)
                .then(|| raw.map(str::to_owned))
                .flatten()
        });
        let value = frozen.resolved(&MARGIN_ROW_CLASSWISE).unwrap();
        assert_eq!(
            super::MarginRowRuntimePolicy::from_levers(&frozen)
                .unwrap()
                .classwise,
            expected
        );
        assert_eq!(
            super::margin_row_classwise_from_env(value.value.as_str()),
            expected,
            "{raw:?}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        let raw = RawLeverInputs::capture_with(ny_levers::all(), |name| {
            (name == MARGIN_ROW_CLASSWISE.name)
                .then(|| std::ffi::OsString::from_vec(vec![b'1', 255]))
        });
        let frozen = LeverSet::resolve(ny_levers::all(), &raw);
        let value = frozen.resolved(&MARGIN_ROW_CLASSWISE).unwrap();
        assert_eq!(value.env_utf8, Some(false));
        assert!(!super::margin_row_classwise_from_env(value.value.as_str()));
        assert!(
            !super::MarginRowRuntimePolicy::from_levers(&frozen)
                .unwrap()
                .classwise
        );
    }
}

#[test]
fn classwise_lane_dispatch_uses_the_receipt_snapshot_under_opposite_ambient_environment() {
    let mut rng = StdRng::seed_from_u64(32);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).unwrap();
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let (target, runner_up, _) = tiny_targets(&net, &lo, &hi);
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_CLASSWISE;
    for (frozen_value, ambient_value, requested) in [("0", " YeS ", false), (" YeS ", "0", true)] {
        let levers = ny_levers::LeverSet::freeze_with(ny_levers::all(), |name| {
            (name == decl.name).then(|| frozen_value.to_owned())
        });
        let policy = super::MarginRowRuntimePolicy::from_levers(&levers).unwrap();
        assert_eq!(policy.classwise, requested);
        ny_test_utils::env::with_env_edits(|env| {
            env.set(decl.name, ambient_value);
            let outcome = super::run_margin_row_lane_with_policy(
                &spec,
                &lo,
                &hi,
                runner_up,
                &[target],
                Some(std::time::Instant::now() + std::time::Duration::from_secs(10)),
                0,
                policy,
            );
            let MarginRowOutcome::Unknown {
                stats: Some(stats), ..
            } = outcome
            else {
                panic!("unsafe zero-expansion fixture must remain open")
            };
            assert_eq!(
                stats.frontier_execution.route.stable_token() == "classwise",
                requested
            );
        });
    }
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
        frontier_execution: FrontierExecutionEvidence::classwise(),
        policy_execution: PolicyExecutionEvidence::default(),
        epoch_execution: Default::default(),
        width_execution: Default::default(),
        root_precision_execution: Default::default(),
        backward_interm_execution: Default::default(),
        root_alpha_execution: Default::default(),
        close_integrity: Default::default(),
    };
    let record = |class, verified, expansions, domains_created, max_depth| {
        let mut frontier_execution = FrontierExecutionEvidence::classwise();
        frontier_execution.observe(max_depth + 1);
        ClassBabStats {
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
            frontier_execution,
            policy_execution: PolicyExecutionEvidence::default(),
            epoch_execution: Default::default(),
            width_execution: Default::default(),
        }
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
    assert_eq!(aggregate.frontier_execution.steps, 2);
    assert_eq!(aggregate.frontier_execution.counterfactual_differences, 0);
    assert_eq!(aggregate.frontier_execution.max_open, 6);
    assert_eq!(aggregate.frontier_execution.max_batch, 1);
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
fn frontier_execution_evidence_records_only_counterfactual_tree_decisions() {
    let mut serial = FrontierExecutionEvidence::joint(1, 8);
    serial.observe(1);
    assert_eq!(serial.counterfactual_differences, 0);
    serial.observe(8);
    assert_eq!(
        (
            serial.configured,
            serial.contextual,
            serial.steps,
            serial.counterfactual_differences,
            serial.max_open,
            serial.max_batch,
        ),
        (1, 8, 2, 1, 8, 1),
    );

    let mut wider_than_context = FrontierExecutionEvidence::joint(16, 8);
    wider_than_context.observe(8);
    assert_eq!(wider_than_context.counterfactual_differences, 0);
    wider_than_context.observe(9);
    assert_eq!(wider_than_context.counterfactual_differences, 1);
    assert_eq!(wider_than_context.max_batch, 9);

    let mut classwise = FrontierExecutionEvidence::classwise();
    classwise.observe(32);
    assert_eq!(classwise.steps, 1);
    assert_eq!(classwise.counterfactual_differences, 0);
    assert_eq!(classwise.max_batch, 1);
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
fn classwise_policy_floors_the_cache_and_suppresses_parallel_frontiers() {
    assert_eq!(classwise_runtime_policy_for_test(0, 16, 12), (1, 1, 1));
    assert_eq!(classwise_runtime_policy_for_test(64, 8, 24), (64, 1, 1));
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
        MarginRowOutcome::Unsat(..)
    ));
    let MarginRowOutcome::Unsat(stats, _) = MarginRowBab::run_classwise(&net, &root, t, &adv, cfg)
    else {
        panic!("classwise should certify the same robust box")
    };
    assert_eq!(
        stats.root_closed_classes + stats.tree_classes.len(),
        adv.len()
    );
    assert_eq!(stats.frontier_execution.route.stable_token(), "classwise");
    assert_eq!(stats.frontier_execution.configured, 1);
    assert_eq!(stats.frontier_execution.contextual, 1);
    assert_eq!(stats.frontier_execution.counterfactual_differences, 0);
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
    assert_eq!(cs.frontier_execution.route.stable_token(), "classwise");
    assert_eq!(cs.frontier_execution.counterfactual_differences, 0);

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
    assert!(matches!(
        MarginRowBab::run(&net, &parity, t, &adv, BabConfig::default()),
        MarginRowOutcome::Unknown { stats: None, .. }
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
    assert_eq!(zs.frontier_execution.steps, 0);
    assert_eq!(zs.frontier_execution.counterfactual_differences, 0);
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
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (3, n_out_t),
            }, // t4 (y)
            TwinOpSpec::Relu { input: 4 },    // t5 (head relu)
            TwinOpSpec::Gemm {
                input: 5,
                weight: w(2 * 3),
                bias: w(2),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
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
        MarginRowOutcome::Unsat(s, _) => {
            (true, s.root_bound.to_bits(), s.expansions, s.stop.clone())
        }
        MarginRowOutcome::Unknown { stats, reason } => {
            stats.as_ref().map_or((false, 0, 0, reason.clone()), |s| {
                (false, s.root_bound.to_bits(), s.expansions, s.stop.clone())
            })
        }
    }
}

fn outcome_stats(out: &MarginRowOutcome) -> &BabStats {
    match out {
        MarginRowOutcome::Unsat(stats, _) => stats,
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
    let default_policy = outcome_stats(&par).policy_execution;
    assert!(!default_policy.parallel_prebuild);
    assert_eq!(default_policy.parallel_prebuild_batches, 0);
    assert_eq!(default_policy.parallel_prebuild_domains, 0);

    // MOAT: falsified orientation must NEVER be UNSAT under the parallel lane.
    // Bounded so it terminates (it can never close a leaf containing the SAT
    // point → Unknown by max_expansions).
    let moat_cfg = BabConfig {
        lru_cap: 64,
        frontier: 8,
        frontier_contextual: 1,
        parallel_prebuild: true,
        max_expansions: 400,
        ..BabConfig::default()
    };
    let adv_f: Vec<usize> = (0..4).filter(|&o| o != worst).collect();
    let moat = MarginRowBab::run(&net, &root, worst, &adv_f, moat_cfg);
    assert!(
        !matches!(moat, MarginRowOutcome::Unsat(..)),
        "MOAT VIOLATION: parallel lane certified a falsified instance UNSAT"
    );
    let moat_evidence = outcome_stats(&moat).frontier_execution;
    assert!(moat_evidence.steps > 0, "frontier never reached tree work");
    assert!(
        moat_evidence.counterfactual_differences > 0,
        "branching fixture never distinguished width 8 from contextual serial: \
         {moat_evidence:?}"
    );
    assert!(moat_evidence.max_open >= 2);
    assert!(moat_evidence.max_batch >= 2 && moat_evidence.max_batch <= 8);
    let moat_policy = outcome_stats(&moat).policy_execution;
    assert!(moat_policy.parallel_prebuild);
    assert!(
        moat_policy.parallel_prebuild_batches > 0,
        "branching fixture never executed a parallel prebuild: {moat_policy:?}"
    );
    assert!(
        moat_policy.parallel_prebuild_domains >= 2 * moat_policy.parallel_prebuild_batches,
        "the fixture must meet the optimizer's multi-domain engagement boundary: {moat_policy:?}"
    );

    // DETERMINISM: two parallel runs on the branching (falsified) instance,
    // same budget, must produce identical stats — no float race, no order
    // nondeterminism in the verdict-bearing quantities.
    let det_cfg = || BabConfig {
        lru_cap: 64,
        frontier: 8,
        parallel_prebuild: true,
        max_expansions: 300,
        ..BabConfig::default()
    };
    let a_out = MarginRowBab::run(&net, &root, worst, &adv_f, det_cfg());
    let b_out = MarginRowBab::run(&net, &root, worst, &adv_f, det_cfg());
    let a = verdict_bits(&a_out);
    let b = verdict_bits(&b_out);
    assert_eq!(a, b, "parallel frontier is non-deterministic across runs");
    let a_policy = outcome_stats(&a_out).policy_execution;
    let b_policy = outcome_stats(&b_out).policy_execution;
    assert_eq!(
        a_policy, b_policy,
        "parallel prebuild execution evidence is non-deterministic across runs"
    );
    assert!(
        a_policy.parallel_prebuild_batches > 0,
        "determinism fixture never executed a parallel prebuild: {a_policy:?}"
    );
    assert!(
        a_policy.parallel_prebuild_domains >= 2 * a_policy.parallel_prebuild_batches,
        "determinism fixture did not meet the multi-domain engagement boundary: {a_policy:?}"
    );
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

// ===================== #mr-stack (throughput plan B5) =====================
// The domain-stacked frontier now runs a batch's MAIN eval passes as ONE
// stacked pass (`bab/stack_eval.rs`). These tests pin that each domain's block
// is bit-identical to its own pass — at the engine, at a finished domain
// evaluation, and over a whole tree.

/// The class with the LOWEST score at the box midpoint: with it as the target,
/// every margin `f_target - f_j` is negative at a point of the root box, so no
/// class can close at the root (the certified bound never exceeds a realised
/// value) and the root eval width is exactly the adversarial count.
fn tiny_argmin_target(net: &TwinNet, lo: &[f64], hi: &[f64]) -> usize {
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
    (0..n_out)
        .min_by(|&a, &b| scores[a].total_cmp(&scores[b]))
        .unwrap()
}

/// Bit-compare two engine passes (coefficients, error lane, bias, bias error,
/// the `unst_abs` mass and the `unst_rows` capture).
fn assert_pass_bits(tag: &str, a: &super::engine::PassOut, b: &super::engine::PassOut) {
    assert_eq!(a.a.dim(), b.a.dim(), "{tag}: coefficient shape");
    for ((i, j), x) in a.a.indexed_iter() {
        assert_eq!(
            x.to_bits(),
            b.a[[i, j]].to_bits(),
            "{tag}: coefficient moved at ({i}, {j})"
        );
    }
    match (&a.e, &b.e) {
        (Some(ea), Some(eb)) => {
            assert_eq!(ea.dim(), eb.dim(), "{tag}: error-lane shape");
            for ((i, j), x) in ea.indexed_iter() {
                assert_eq!(
                    x.to_bits(),
                    eb[[i, j]].to_bits(),
                    "{tag}: error coefficient moved at ({i}, {j})"
                );
            }
        }
        (None, None) => {}
        _ => panic!("{tag}: error-lane presence differs"),
    }
    let bits = |v: &[f64]| -> Vec<u64> { v.iter().map(|x| x.to_bits()).collect() };
    assert_eq!(bits(&a.b), bits(&b.b), "{tag}: bias moved");
    assert_eq!(bits(&a.eb), bits(&b.eb), "{tag}: bias error moved");
    let coll_bits = |c: &Option<std::collections::BTreeMap<usize, Vec<f64>>>| {
        c.as_ref().map(|m| {
            m.iter()
                .map(|(&li, v)| (li, bits(v)))
                .collect::<Vec<(usize, Vec<u64>)>>()
        })
    };
    assert_eq!(
        coll_bits(&a.coll),
        coll_bits(&b.coll),
        "{tag}: unst_abs branching mass moved"
    );
    let rows_bits = |c: &Option<std::collections::BTreeMap<usize, Array2<f64>>>| {
        c.as_ref().map(|m| {
            m.iter()
                .map(|(&li, mat)| (li, mat.dim(), mat.iter().map(|x| x.to_bits()).collect()))
                .collect::<Vec<(usize, (usize, usize), Vec<u64>)>>()
        })
    };
    assert_eq!(
        rows_bits(&a.unst_rows),
        rows_bits(&b.unst_rows),
        "{tag}: unst_rows capture moved"
    );
    assert_eq!(
        rows_bits(&a.coll_rows),
        rows_bits(&b.coll_rows),
        "{tag}: Tier-0 capture moved"
    );
}

/// #mr-stack ENGINE ORACLE: `run_domain_stacked_split` hands every block back
/// the exact pass its domain runs alone with the same collection spec —
/// coefficients, certified error lane, bias, bias error, the PER-BLOCK
/// `unst_abs` branching mass (the one cross-column reduction a pass has) and
/// the `unst_rows` gradient capture — across blocks of width 1 and 3, split
/// and unsplit gates, and a block carrying a shared β term; and without
/// `unst_abs` the blocks carry no `coll`, exactly like their own passes.
#[test]
fn domain_stacked_split_matches_independent_collect_passes() {
    let mut rng = StdRng::seed_from_u64(0x00b5_5711);
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
    let (li1, pos1, _) = unstable[1];
    let mut domains = [
        domain_gates(&root, &[]),
        domain_gates(&root, &[(li0, pos0, 1)]),
        domain_gates(&root, &[(li0, pos0, -1), (li1, pos1, 1)]),
        domain_gates(&root, &[(li1, pos1, -1)]),
    ];
    domains[2].beta.insert(
        li0,
        vec![super::engine::BetaSplit {
            neuron: idx0,
            sign: -1,
            beta: 0.4,
        }],
    );
    let widths = [1usize, 3, 3, 1];
    let seed_for = |d: usize, width: usize| DomainStackSeed {
        s: Array2::from_shape_fn((net.n_y, width), |(j, r)| {
            let k = 1 + d * 97 + j * 17 + r * 7;
            ((k as f64) * 0.173).sin() * (1.0 + d as f64 * 0.125)
        }),
        e: Some(Array2::from_shape_fn((net.n_y, width), |(j, r)| {
            (1 + d + j + 3 * r) as f64 * 1e-16
        })),
    };
    let seeds: Vec<DomainStackSeed> = widths
        .iter()
        .enumerate()
        .map(|(d, &w)| seed_for(d, w))
        .collect();
    let total: usize = widths.iter().sum();
    let mut stacked_s = Array2::<f64>::zeros((net.n_y, total));
    let mut stacked_e = Array2::<f64>::zeros((net.n_y, total));
    let mut offset = 0usize;
    for (seed, &w) in seeds.iter().zip(&widths) {
        stacked_s
            .slice_mut(ndarray::s![.., offset..offset + w])
            .assign(&seed.s);
        stacked_e
            .slice_mut(ndarray::s![.., offset..offset + w])
            .assign(seed.e.as_ref().expect("outward seed"));
        offset += w;
    }
    let stacked_seed = DomainStackSeed {
        s: stacked_s,
        e: Some(stacked_e),
    };
    let no_exc = Exceptions::default();
    let mut offset = 0usize;
    let blocks: Vec<RowDomainGateBlock<'_>> = domains
        .iter()
        .zip(&widths)
        .map(|(gates, &w)| {
            let start = offset;
            offset += w;
            RowDomainGateBlock {
                columns: start..offset,
                gates,
                exceptions: &no_exc,
            }
        })
        .collect();
    for unst_rows in [false, true] {
        let collect = super::engine::Collect {
            unst_abs: true,
            rows: None,
            unst_rows,
        };
        let split = eng
            .run_domain_stacked_split(&stacked_seed, &blocks, LaneDir::Lower, collect)
            .expect("stacked split");
        assert_eq!(split.len(), domains.len(), "one pass per block");
        for (d, block_pass) in split.iter().enumerate() {
            let own = eng
                .run_collect(&seeds[d], Some(&domains[d]), LaneDir::Lower, None, collect)
                .expect("own pass");
            let tag = format!("unst_rows={unst_rows} domain={d} width={}", widths[d]);
            assert!(
                block_pass.coll.as_ref().is_some_and(|m| !m.is_empty()),
                "{tag}: the per-block unst_abs mass must be populated (VOID otherwise)"
            );
            assert_eq!(
                block_pass.unst_rows.is_some(),
                unst_rows,
                "{tag}: capture presence"
            );
            assert_pass_bits(&tag, block_pass, &own);
            let lo_stk = eng.concretize_lower(block_pass);
            let lo_own = eng.concretize_lower(&own);
            assert_eq!(
                lo_stk.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                lo_own.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{tag}: concretized bound moved"
            );
        }
    }
    let plain = eng
        .run_domain_stacked_split(
            &stacked_seed,
            &blocks,
            LaneDir::Lower,
            super::engine::Collect::default(),
        )
        .expect("plain split");
    for (d, block_pass) in plain.iter().enumerate() {
        let own = eng
            .run_collect(
                &seeds[d],
                Some(&domains[d]),
                LaneDir::Lower,
                None,
                super::engine::Collect::default(),
            )
            .expect("own plain pass");
        assert!(block_pass.coll.is_none(), "no unst_abs requested, no coll");
        assert_pass_bits(&format!("plain domain={d}"), block_pass, &own);
    }
}

/// #mr-stack (B5) DOMAIN-EVAL DIFFERENTIAL — the load-bearing claim: a
/// domain evaluated through its block of ONE stacked main pass is
/// bit-identical to the same domain evaluated through its own pass, both at
/// the pass (coefficients, error lane, bias, bias error, captures) and at the
/// FINISHED evaluation (bound, per-column state, retirement, β / alpha state,
/// node gates). Covers K in {2, 5, 8} stacked domains at mixed depths, nf in
/// {1, 3}, β-carrying and β-free blocks in one stack, the domain-alpha arm
/// on and off, capturing and non-capturing blocks in ONE stack (the fixture
/// disarms the ascents on every third slot, so the stacked pass's per-block
/// `unst_rows` DROP runs and is compared), and one more domain whose gates
/// carry `beta_pc`, which must fall back to its own pass and still agree.
#[test]
fn stacked_main_eval_bit_identical_to_per_domain() {
    let mut rng = StdRng::seed_from_u64(0x5c0e_cafe);
    let spec = tiny_spec(&mut rng, 0.48);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.25; 72];
    let hi = vec![0.25; 72];
    let (argmax, _, _) = tiny_targets(&net, &lo, &hi);
    let target = tiny_argmin_target(&net, &lo, &hi);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let adv_one = vec![argmax];
    let adv_three: Vec<usize> = (0..4).filter(|&o| o != target).collect();
    assert_ne!(argmax, target, "fixture needs distinct argmax / argmin");
    for (adv, nf) in [(&adv_one, 1usize), (&adv_three, 3usize)] {
        for (case, k) in [2usize, 5, 8].into_iter().enumerate() {
            let alpha = case % 2 == 1;
            let cfg = BabConfig {
                lru_cap: 64,
                frontier: 8,
                domain_stack: true,
                col_retire: true,
                domain_alpha_policy: super::DomainAlphaRuntimePolicy {
                    enabled: alpha,
                    ..super::DomainAlphaRuntimePolicy::default()
                },
                ..BabConfig::default()
            };
            let diff = MarginRowBab::stacked_main_eval_differential(
                &net, &root, target, adv, k, cfg, true, true,
            );
            let tag = format!("nf={nf} K={k} alpha={alpha}");
            assert!(
                diff.mismatches.is_empty(),
                "{tag}: the stacked main eval moved bits:\n{}",
                diff.mismatches.join("\n")
            );
            assert_eq!(
                diff.compared,
                k + 1,
                "{tag}: fixture did not seed K+1 domains (depths {:?})",
                diff.depths
            );
            assert_eq!(diff.widths[0], nf, "{tag}: root eval width");
            let mut depths = diff.depths.clone();
            depths.sort_unstable();
            depths.dedup();
            assert!(
                depths.len() >= 2,
                "{tag}: depths not mixed {:?}",
                diff.depths
            );
            assert!(diff.beta_blocks >= 1, "{tag}: no β-carrying block");
            if alpha {
                // The mixed stack is the only thing that runs the per-block
                // `unst_rows` DROP; without it that branch is untested (VOID).
                assert!(
                    diff.capture_blocks > 0 && diff.capture_blocks < k,
                    "{tag}: capturing and non-capturing blocks must mix in one stack: \
                     {} of {k} stacked domains capture",
                    diff.capture_blocks
                );
            }
            assert_eq!(diff.beta_pc_index, Some(k), "{tag}: beta_pc domain");
            let r = diff.report;
            assert_eq!(r.passes, 1, "{tag}: exactly one stacked pass: {r:?}");
            assert_eq!(r.blocks, k as u64, "{tag}: {r:?}");
            assert_eq!(r.k_max, k as u64, "{tag}: {r:?}");
            assert_eq!(
                r.cols,
                diff.widths[..k].iter().sum::<usize>() as u64,
                "{tag}: stacked width is the sum of the stacked domains' widths: {r:?}"
            );
            assert_eq!(
                r.fallback_beta_pc, 1,
                "{tag}: beta_pc must fall back: {r:?}"
            );
            assert_eq!(
                (r.fallback_err, r.fallback_seam, r.solo),
                (0, 0, 0),
                "{tag}: {r:?}"
            );
        }
    }
}

/// #mr-stack (B5) WHOLE-TREE DIFFERENTIAL: same fixture, frontier width, pop
/// order and budget; three schedules — fully independent (`domain_stack`
/// off), score-stacked only (the pre-B5 schedule), and score + main-eval
/// stacked (B5). The tree trajectory hash — every merged decision, every
/// pushed child's bound, splits, y-box, β and retirement state, as exact bits
/// — must agree across all three, with the stacked main eval PROVABLY engaged
/// (a null from a schedule that never stacked would be VOID, not a pass).
/// Run with the domain-alpha arm off and on.
#[test]
fn expand_batch_trajectory_hash_unchanged() {
    // First fixture (seed, half-width) whose B5 run actually stacks a main
    // pass; fail closed if none does.
    let fixtures = [
        (0x5c0e_cafe_u64, 0.25f64),
        (0x00b5_7a1e, 0.30),
        (0x00b5_0bad, 0.40),
    ];
    let mut engaged_fixtures = 0usize;
    for (seed, half) in fixtures {
        let mut rng = StdRng::seed_from_u64(seed);
        let spec = tiny_spec(&mut rng, 0.48);
        let net = TwinNet::compile(&spec).expect("compile");
        let lo = vec![-half; 72];
        let hi = vec![half; 72];
        let target = tiny_argmin_target(&net, &lo, &hi);
        let adv: Vec<usize> = (0..4).filter(|&o| o != target).collect();
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        for alpha in [false, true] {
            let cfg = |domain_stack: bool, main_eval: bool| BabConfig {
                lru_cap: 64,
                frontier: 8,
                domain_stack,
                domain_stack_main_eval: main_eval,
                col_retire: true,
                max_expansions: 60,
                domain_alpha_policy: super::DomainAlphaRuntimePolicy {
                    enabled: alpha,
                    ..super::DomainAlphaRuntimePolicy::default()
                },
                ..BabConfig::default()
            };
            let mut arms = Vec::new();
            for (name, stack, main) in [
                ("independent", false, true),
                ("score_stacked", true, false),
                ("b5_stacked", true, true),
            ] {
                let _ = super::bab::take_last_trajectory();
                let out = MarginRowBab::run(&net, &root, target, &adv, cfg(stack, main));
                let (hash, steps, report) = super::bab::take_last_trajectory()
                    .unwrap_or_else(|| panic!("{name}: the tree published no trajectory"));
                arms.push((name, out, hash, steps, report));
            }
            let tag = format!("seed={seed:#x} half={half} alpha={alpha}");
            let (_, ref_out, ref_hash, ref_steps, ref_report) = &arms[0];
            assert!(*ref_steps > 0, "{tag}: empty trajectory");
            assert_eq!(
                (ref_report.batches, ref_report.passes, ref_report.blocks),
                (0, 0, 0),
                "{tag}: the independent schedule never enters the staged eval"
            );
            let ref_stats = outcome_stats(ref_out);
            for (name, out, hash, steps, report) in &arms[1..] {
                assert_eq!(
                    verdict_bits(out),
                    verdict_bits(ref_out),
                    "{tag} {name}: verdict / root bound / expansions / stop moved"
                );
                assert_eq!(
                    (*hash, *steps),
                    (*ref_hash, *ref_steps),
                    "{tag} {name}: TRAJECTORY HASH moved (a merged decision or a child's \
                     bound/state bits differ)"
                );
                let stats = outcome_stats(out);
                assert_eq!(
                    stats.domains_created, ref_stats.domains_created,
                    "{tag} {name}"
                );
                assert_eq!(stats.closed, ref_stats.closed, "{tag} {name}");
                assert_eq!(stats.max_depth, ref_stats.max_depth, "{tag} {name}");
                assert_eq!(stats.mono_raw_dips, ref_stats.mono_raw_dips, "{tag} {name}");
                assert_eq!(
                    stats.mono_worst.to_bits(),
                    ref_stats.mono_worst.to_bits(),
                    "{tag} {name}"
                );
                assert_eq!(
                    stats.policy_execution, ref_stats.policy_execution,
                    "{tag} {name}: execution evidence moved"
                );
                if *name == "score_stacked" {
                    // The kill switch: every staged batch evaluated per domain.
                    assert_eq!(report.passes, 0, "{tag}: kill switch leaked: {report:?}");
                    assert_eq!(report.blocks, 0, "{tag}: kill switch leaked: {report:?}");
                    assert!(
                        report.batches == 0 || report.disabled > 0,
                        "{tag}: {report:?}"
                    );
                }
            }
            let b5 = &arms[2].4;
            assert_eq!(b5.disabled, 0, "{tag}: {b5:?}");
            assert_eq!((b5.fallback_err, b5.fallback_seam), (0, 0), "{tag}: {b5:?}");
            if b5.passes > 0 {
                assert!(b5.blocks >= 2 * b5.passes, "{tag}: {b5:?}");
                engaged_fixtures += 1;
            }
        }
    }
    assert!(
        engaged_fixtures > 0,
        "VOID: no fixture ever stacked a main pass — the differential proved nothing"
    );
}

// ---------------------------------------------------------------------------
// #mr-close-telemetry: the proof-integrity telemetry is PURE (a tree run with
// and without it is bit-identical) and COMPLETE (it accounts for every leaf
// the tree closed, every class the root closed, and never records a
// certificate that is not strictly positive under outward rounding).
// ---------------------------------------------------------------------------

/// Every close that is a Kraft LEAF of the tree (root closes and retirements
/// are proof ingredients but not tree leaves).
const CLOSE_LEAF_KINDS: [super::close_integrity::CertKind; 8] = [
    super::close_integrity::CertKind::Eval,
    super::close_integrity::CertKind::Beta,
    super::close_integrity::CertKind::DomainAlpha,
    super::close_integrity::CertKind::Raw,
    super::close_integrity::CertKind::Lift,
    super::close_integrity::CertKind::Queued,
    super::close_integrity::CertKind::Infeasible,
    super::close_integrity::CertKind::HeadInfeasible,
];

fn close_leaf_count(ev: &super::close_integrity::CloseIntegrityEvidence) -> u64 {
    CLOSE_LEAF_KINDS.iter().map(|&k| ev.kind(k).n).sum()
}

/// `(seed, weight scale, box lo, box hi, argmax target)`: the three trees the
/// B5 differential uses (argmin target: a tree that can never verify), the
/// Tier-2 epoch fixture's box (`epoch_nested_lane_closes_and_moat_holds`: "wide
/// enough that the tree actually expands but still closable"), and
/// argmax-target boxes of growing width, so some roots close outright, some
/// trees close leaves, and some verify.
const CLOSE_TELEMETRY_FIXTURES: [(u64, f64, f64, f64, bool); 10] = [
    (0x5c0e_cafe, 0.48, -0.25, 0.25, false),
    (0x00b5_7a1e, 0.48, -0.30, 0.30, false),
    (0x00b5_0bad, 0.48, -0.40, 0.40, false),
    (71, 0.5, -0.02, 0.06, true),
    (29, 0.35, -0.04, 0.04, true),
    (29, 0.35, -0.08, 0.08, true),
    (30, 0.35, -0.06, 0.06, true),
    (30, 0.35, -0.12, 0.12, true),
    (32, 0.35, -0.05, 0.05, true),
    (32, 0.35, -0.10, 0.10, true),
];

fn close_telemetry_fixture(
    seed: u64,
    scale: f64,
    box_lo: f64,
    box_hi: f64,
    argmax: bool,
) -> (TwinNet, RootGates, usize, Vec<usize>, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let spec = tiny_spec(&mut rng, scale);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![box_lo; 72];
    let hi = vec![box_hi; 72];
    let (best, runner_up, _) = tiny_targets(&net, &lo, &hi);
    let target = if argmax {
        best
    } else {
        tiny_argmin_target(&net, &lo, &hi)
    };
    let adv: Vec<usize> = (0..4).filter(|&o| o != target).collect();
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    // `runner_up` is the MOAT swap partner for an argmax target: the margin
    // `f_runner_up - f_best` is negative at the box midpoint.
    (net, root, target, adv, runner_up)
}

/// `close_telemetry_trajectory_hash_unchanged`: the SAME tree run with the
/// proof-integrity telemetry on and off (the test-only
/// `BabConfig::close_telemetry_off`, which skips every recording site in the
/// driver, its workers, its class trees and its nested epoch trees) must merge
/// the same tree bit-for-bit: equal trajectory hash and merged-decision count
/// on the parallel routes (per-domain and domain-stacked), equal verdict /
/// root bound / expansions / stop, and equal statistics and execution evidence
/// on every route (serial and classwise included). The OFF arm must record
/// nothing (the switch really bites) and the ON arms must record tree leaves —
/// a differential in which the telemetry never fired would be VOID.
#[test]
fn close_telemetry_trajectory_hash_unchanged() {
    let mut compared_hashes = 0usize;
    let mut recorded_leaves = 0u64;
    for (seed, scale, box_lo, box_hi, argmax) in CLOSE_TELEMETRY_FIXTURES {
        let (net, root, target, adv, _) =
            close_telemetry_fixture(seed, scale, box_lo, box_hi, argmax);
        for alpha in [false, true] {
            for (route, frontier, stack, classwise) in [
                ("parallel", 8usize, false, false),
                ("stacked", 8, true, false),
                ("serial", 1, false, false),
                ("classwise", 1, false, true),
            ] {
                let cfg = |off: bool| BabConfig {
                    lru_cap: 64,
                    frontier,
                    domain_stack: stack,
                    col_retire: true,
                    max_expansions: 60,
                    domain_alpha_policy: super::DomainAlphaRuntimePolicy {
                        enabled: alpha,
                        ..super::DomainAlphaRuntimePolicy::default()
                    },
                    close_telemetry_off: off,
                    ..BabConfig::default()
                };
                let run = |off: bool| {
                    let _ = super::bab::take_last_trajectory();
                    let out = if classwise {
                        MarginRowBab::run_classwise(&net, &root, target, &adv, cfg(off))
                    } else {
                        MarginRowBab::run(&net, &root, target, &adv, cfg(off))
                    };
                    (out, super::bab::take_last_trajectory())
                };
                let (on, on_traj) = run(false);
                let (off, off_traj) = run(true);
                let tag = format!(
                    "seed={seed:#x} box=[{box_lo},{box_hi}] argmax={argmax} alpha={alpha} {route}"
                );
                assert_eq!(
                    verdict_bits(&on),
                    verdict_bits(&off),
                    "{tag}: verdict / root bound / expansions / stop moved"
                );
                assert_eq!(
                    on_traj, off_traj,
                    "{tag}: TRAJECTORY HASH moved (a merged decision or a child's bound/state \
                     bits differ between telemetry on and off)"
                );
                if frontier > 1 {
                    if let Some((_, steps, _)) = on_traj {
                        if steps > 0 {
                            compared_hashes += 1;
                        }
                    }
                }
                let stats_of = |o: &MarginRowOutcome| -> Option<BabStats> {
                    match o {
                        MarginRowOutcome::Unsat(s, _) => Some(s.clone()),
                        MarginRowOutcome::Unknown { stats, .. } => stats.clone(),
                    }
                };
                let (Some(s_on), Some(s_off)) = (stats_of(&on), stats_of(&off)) else {
                    assert_eq!(
                        stats_of(&on).is_some(),
                        stats_of(&off).is_some(),
                        "{tag}: one arm owns stats and the other does not"
                    );
                    continue;
                };
                assert_eq!(s_on.domains_created, s_off.domains_created, "{tag}");
                assert_eq!(s_on.closed, s_off.closed, "{tag}");
                assert_eq!(s_on.max_depth, s_off.max_depth, "{tag}");
                assert_eq!(s_on.mono_raw_dips, s_off.mono_raw_dips, "{tag}");
                assert_eq!(
                    s_on.mono_worst.to_bits(),
                    s_off.mono_worst.to_bits(),
                    "{tag}"
                );
                assert_eq!(s_on.root_closed_classes, s_off.root_closed_classes, "{tag}");
                assert_eq!(s_on.tree_classes, s_off.tree_classes, "{tag}");
                assert_eq!(s_on.ledger_ok, s_off.ledger_ok, "{tag}");
                assert_eq!(s_on.epochs_attempted, s_off.epochs_attempted, "{tag}");
                assert_eq!(s_on.epochs_closed, s_off.epochs_closed, "{tag}");
                assert_eq!(
                    s_on.policy_execution, s_off.policy_execution,
                    "{tag}: execution evidence moved"
                );
                assert_eq!(s_on.epoch_execution, s_off.epoch_execution, "{tag}");
                assert_eq!(s_on.width_execution, s_off.width_execution, "{tag}");
                let off_ev = &s_off.close_integrity;
                assert_eq!(
                    (
                        off_ev.closes(),
                        off_ev.epoch_closes,
                        off_ev.frozen_dual_closes
                    ),
                    (0, 0, 0),
                    "{tag}: the OFF arm recorded telemetry — the switch does not bite, so this \
                     differential compares nothing"
                );
                recorded_leaves += close_leaf_count(&s_on.close_integrity);
            }
        }
    }
    assert!(
        compared_hashes > 0,
        "VOID: no parallel tree published a non-empty trajectory"
    );
    assert!(
        recorded_leaves > 0,
        "VOID: the ON arms never recorded a tree leaf — the differential proved nothing"
    );
}

/// `close_telemetry_accounts_every_leaf_and_is_strictly_positive`: on every
/// route (parallel, domain-stacked, serial, classwise), for every run that
/// owns stats:
///
/// * the tree-leaf certificates number EXACTLY the tree's closed domains (no
///   close site forgotten, none counted twice) whenever no epoch ran;
/// * the root certificates number exactly the classes the root closed, and
///   their minimum is bit-identical to the smallest closed `dj` of an
///   independent `root_eval` of the same root;
/// * no recorded certificate is non-positive, NaN, or below the kind's own
///   closure rule (strict `> 0` under outward rounding) — a value that
///   disagreed with its close decision would show here;
/// * a verified run has a thinnest certificate, and it is strictly positive;
///
/// and the MOAT swap of every argmax fixture (a property violated at the box
/// midpoint) still never verifies with the telemetry on. At least one run
/// must record tree leaves and at least one must verify through the tree
/// (fail closed: VOID otherwise).
#[test]
fn close_telemetry_accounts_every_leaf_and_is_strictly_positive() {
    use super::close_integrity::{CertKind, ALL_KINDS};

    let mut leaves_seen = 0u64;
    let mut tree_verdicts = 0usize;
    for (seed, scale, box_lo, box_hi, argmax) in CLOSE_TELEMETRY_FIXTURES {
        let (net, root, target, adv, runner_up) =
            close_telemetry_fixture(seed, scale, box_lo, box_hi, argmax);
        let eng = BackwardEngine::new(&net, &root);
        let re = super::bab::root_eval(&eng, &net, target, &adv).expect("root_eval");
        let closed_dj: Vec<f64> = re
            .dj
            .iter()
            .copied()
            .filter(|&d| super::bab::closed(RoundMode::Outward, d))
            .collect();
        let expect_min_dj = closed_dj.iter().copied().fold(f64::INFINITY, f64::min);
        for (route, frontier, stack, classwise) in [
            ("parallel", 8usize, false, false),
            ("stacked", 8, true, false),
            ("serial", 1, false, false),
            ("classwise", 1, false, true),
        ] {
            let cfg = BabConfig {
                lru_cap: 64,
                frontier,
                domain_stack: stack,
                col_retire: true,
                max_expansions: 400,
                ..BabConfig::default()
            };
            let out = if classwise {
                MarginRowBab::run_classwise(&net, &root, target, &adv, cfg)
            } else {
                MarginRowBab::run(&net, &root, target, &adv, cfg)
            };
            let tag = format!("seed={seed:#x} box=[{box_lo},{box_hi}] argmax={argmax} {route}");
            let (verified, stats) = match &out {
                MarginRowOutcome::Unsat(s, _) => (true, s),
                MarginRowOutcome::Unknown { stats: Some(s), .. } => (false, s),
                MarginRowOutcome::Unknown { stats: None, .. } => continue,
            };
            let ev = &stats.close_integrity;
            assert_eq!(
                ev.nonpositive(),
                0,
                "{tag}: a non-positive certificate: {ev:?}"
            );
            for kind in ALL_KINDS {
                if let Some(min) = ev.kind(kind).min_value() {
                    assert!(
                        super::bab::closed(RoundMode::Outward, min),
                        "{tag}: kind {} recorded a certificate {min:e} its close rule rejects",
                        kind.name()
                    );
                }
            }
            assert_eq!(
                ev.frozen_dual_closes, 0,
                "{tag}: frozen-dual tier is dark here"
            );
            assert_eq!(
                ev.kind(CertKind::Root).n,
                closed_dj.len() as u64,
                "{tag}: root certificates vs classes the root closed"
            );
            assert_eq!(
                ev.kind(CertKind::Root).n,
                stats.root_closed_classes as u64,
                "{tag}: root certificates vs BabStats::root_closed_classes"
            );
            assert_eq!(
                ev.min_root_dj().map(f64::to_bits),
                (!closed_dj.is_empty()).then_some(expect_min_dj.to_bits()),
                "{tag}: smallest root dj differs from an independent root_eval"
            );
            if stats.epochs_attempted == 0 {
                assert_eq!(
                    close_leaf_count(ev),
                    stats.closed as u64,
                    "{tag}: tree-leaf certificates vs closed domains: {ev:?}"
                );
                assert_eq!(ev.kind(CertKind::EpochRoot).n, 0, "{tag}");
                assert_eq!(ev.epoch_closes, 0, "{tag}");
            }
            if verified {
                let min = ev.min_close().map(|(_, v)| v);
                assert!(
                    min.is_some_and(|v| v > 0.0),
                    "{tag}: a verified run must carry a strictly positive thinnest certificate"
                );
                if stats.expansions > 0 {
                    tree_verdicts += 1;
                    assert!(
                        close_leaf_count(ev) > 0,
                        "{tag}: a tree verdict with no leaves"
                    );
                }
            }
            leaves_seen += close_leaf_count(ev);
        }
        if argmax {
            // MOAT: violated at the box midpoint; the telemetry-carrying lane
            // must still never verify it.
            let cfg = BabConfig {
                lru_cap: 64,
                frontier: 8,
                col_retire: true,
                max_expansions: 200,
                ..BabConfig::default()
            };
            assert!(
                !matches!(
                    MarginRowBab::run(&net, &root, runner_up, &[target], cfg),
                    MarginRowOutcome::Unsat(..)
                ),
                "MOAT VIOLATION: seed={seed:#x} box=[{box_lo},{box_hi}] verified a falsified \
                 instance"
            );
        }
    }
    assert!(leaves_seen > 0, "VOID: no run recorded a tree leaf");
    assert!(
        tree_verdicts > 0,
        "VOID: no fixture verified through the tree, so the verified-run checks never ran"
    );
}

/// `close_telemetry_is_pure_and_bounded_on_the_epoch_path`: the Tier-2 epoch
/// route — nested `run_inner` subtrees whose CLOSED certificates are absorbed
/// (`epoch_root` for their roots) and whose FAILED ones are not — on the
/// fixture of `epoch_nested_lane_closes_and_moat_holds`. With the telemetry on
/// and off the verdict, statistics and all execution evidence are identical;
/// the recorded certificates are strictly positive; and the absorb is bounded
/// by the tree's own counters: `leaves + epoch_closes <= closed` and
/// `epoch_closes <= epochs_closed`, with EQUALITY of the first whenever no
/// epoch attempt failed (a failed subtree's closes are counted in `closed`
/// but are not proof and are never absorbed). VOID unless epochs ran.
#[test]
fn close_telemetry_is_pure_and_bounded_on_the_epoch_path() {
    use super::close_integrity::CertKind;

    let mut rng = StdRng::seed_from_u64(71);
    let spec = tiny_spec(&mut rng, 0.5);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.02; 72];
    let hi = vec![0.06; 72];
    let (t, _, adv) = tiny_targets(&net, &lo, &hi);
    let retain = super::root::RetainCfg::default();
    let (root, _) =
        RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, Some(&retain), &[])
            .expect("root");
    let mk_cfg = |off: bool| BabConfig {
        k_head: 0,
        epoch_policy: super::EpochRuntimePolicy {
            depth: 1,
            max_attempts: 8,
            ..super::EpochRuntimePolicy::default()
        },
        epoch_trigger_depth: 1,
        epoch_attempt_cap: 8,
        retain_cfg: Some(retain),
        close_telemetry_off: off,
        ..BabConfig::default()
    };
    let on = MarginRowBab::run(&net, &root, t, &adv, mk_cfg(false));
    let off = MarginRowBab::run(&net, &root, t, &adv, mk_cfg(true));
    assert_eq!(
        verdict_bits(&on),
        verdict_bits(&off),
        "epoch route: verdict / root bound / expansions / stop moved"
    );
    let (s_on, s_off) = (outcome_stats(&on), outcome_stats(&off));
    assert_eq!(s_on.closed, s_off.closed);
    assert_eq!(s_on.domains_created, s_off.domains_created);
    assert_eq!(s_on.max_depth, s_off.max_depth);
    assert_eq!(s_on.epochs_attempted, s_off.epochs_attempted);
    assert_eq!(s_on.epochs_closed, s_off.epochs_closed);
    assert_eq!(s_on.ledger_ok, s_off.ledger_ok);
    assert_eq!(s_on.mono_worst.to_bits(), s_off.mono_worst.to_bits());
    assert_eq!(s_on.policy_execution, s_off.policy_execution);
    assert_eq!(s_on.epoch_execution, s_off.epoch_execution);
    assert_eq!(s_on.width_execution, s_off.width_execution);
    assert!(
        s_on.epochs_attempted > 0,
        "VOID: no Tier-2 epoch was attempted, so the nested absorb never ran"
    );
    let off_ev = &s_off.close_integrity;
    assert_eq!((off_ev.closes(), off_ev.epoch_closes), (0, 0));
    let ev = &s_on.close_integrity;
    assert_eq!(ev.nonpositive(), 0, "a non-positive certificate: {ev:?}");
    let leaves = close_leaf_count(ev);
    let closed = s_on.closed as u64;
    assert!(
        leaves + ev.epoch_closes <= closed,
        "absorbed more closes than the tree closed: leaves {leaves} + epochs {} > {closed}",
        ev.epoch_closes
    );
    assert!(ev.epoch_closes <= s_on.epochs_closed as u64);
    if s_on.epochs_attempted == s_on.epochs_closed {
        assert_eq!(
            leaves + ev.epoch_closes,
            closed,
            "no epoch failed, so every close must be accounted: {ev:?}"
        );
    }
    if ev.epoch_closes == 0 {
        assert_eq!(ev.kind(CertKind::EpochRoot).n, 0);
    }
    eprintln!(
        "[test mr-close] epoch path: closed={closed} leaves={leaves} epoch_closes={} \
         epoch_root={} epochs {}/{}",
        ev.epoch_closes,
        ev.kind(CertKind::EpochRoot).n,
        s_on.epochs_closed,
        s_on.epochs_attempted
    );
}

/// `clip_crossing_variant_agrees_and_prices_every_crossing`: the
/// `_crossing` variant returns EXACTLY the tightening and crossed count of
/// `tighten_intermediate_acc` (the verify decision is untouched), and its
/// slack is strictly positive exactly when a neuron crossed.
#[test]
fn clip_crossing_variant_agrees_and_prices_every_crossing() {
    ny_test_utils::env::with_env_edits(|env| {
        let Some((_net, root)) = clip_chain_fixture(env, "20") else {
            panic!("fixture retained no clip rows — the assertion would be vacuous");
        };
        let l0 = &root.layers[0];
        let n = l0.unst.len();
        let mut chains: Vec<Vec<(usize, usize, i8)>> = Vec::new();
        for pos in 0..n.min(6) {
            for dir in [-1i8, 1] {
                chains.push(vec![(0, pos, dir)]);
                chains.push(vec![(0, pos, dir), (0, (pos + 1) % n, -dir)]);
            }
        }
        let mut crossed_seen = 0usize;
        let mut tightened_seen = 0usize;
        for chain in &chains {
            let hs = super::clip::halfspaces_for_splits(&root, chain);
            for (li, layer) in root.layers.iter().enumerate() {
                let (a, ca) = super::clip::tighten_intermediate_acc(
                    layer,
                    li,
                    &hs,
                    root.mid.as_slice(),
                    root.rad.as_slice(),
                    20,
                    None,
                );
                let (b, cb, slack) = super::clip::tighten_intermediate_acc_crossing(
                    layer,
                    li,
                    &hs,
                    root.mid.as_slice(),
                    root.rad.as_slice(),
                    20,
                    None,
                );
                assert_eq!(ca, cb, "crossed count moved");
                assert_eq!(a.len(), b.len(), "tightening moved");
                for (x, y) in a.iter().zip(&b) {
                    assert_eq!((x.pos, x.idx), (y.pos, y.idx));
                    assert_eq!(x.l.to_bits(), y.l.to_bits());
                    assert_eq!(x.u.to_bits(), y.u.to_bits());
                }
                tightened_seen += b.len();
                if cb > 0 {
                    crossed_seen += 1;
                    assert!(
                        slack > 0.0,
                        "a crossing must price a positive slack: {slack:e}"
                    );
                } else {
                    assert!(
                        slack.is_infinite() && slack.is_sign_negative(),
                        "no crossing, no slack: {slack:e}"
                    );
                }
            }
        }
        eprintln!(
            "[test mr-close] clip crossing chains={} tightened={tightened_seen} \
             layers_with_crossings={crossed_seen}",
            chains.len()
        );
        assert!(
            tightened_seen + crossed_seen > 0,
            "VOID: no chain tightened or crossed anything — the agreement compared nothing"
        );
    });
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
                ..
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
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
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
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
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
                ..
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
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (cfg.n_y, cw),
    });
    let g1_out = flat_out + 1;
    ops.push(TwinOpSpec::Relu { input: g1_out });
    let hr_out = g1_out + 1;
    ops.push(TwinOpSpec::Gemm {
        input: hr_out,
        weight: wgen(&mut rng, sc, s, cfg.n_out, cfg.n_y),
        bias: bgen(&mut rng, bs, cfg.n_out),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
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
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (n_y, n_in),
    });
    let g1_id = ops.len();
    ops.push(TwinOpSpec::Relu { input: g1_id });
    let hr_id = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input: hr_id,
        weight: wgen(&mut rng, WScheme::ZeroRowSum, 0.7, n_out, n_y),
        bias: bgen(&mut rng, 0.5, n_out),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
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
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (n_y, flat),
        },
        TwinOpSpec::Relu { input: 6 },
        TwinOpSpec::Gemm {
            input: 7,
            weight: wgen(&mut rng, WScheme::ZeroRowSum, 0.3, n_out, n_y),
            bias: bgen(&mut rng, 0.3, n_out),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
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

// ---- #fold-ball: the root box encloses the EXACT fold-ball hull ------------

/// One-trunk-relu net whose trunk-0 pre-activation is `[affine](conv1x1(x))`:
/// Conv(1x1, `rel`, `berr`) [-> ChannelAffine(`srel`, `serr`), scales of both
/// signs] -> Relu -> Flatten -> Gemm -> Relu -> Gemm.
fn fold_ball_spec(rel: f64, berr: f64, affine: Option<(f64, f64)>, seed: u64) -> TwinSpec {
    let (cin, h, w, cw, n_y, n_out) = (3usize, 2usize, 2usize, 8usize, 16usize, 6usize);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ops = vec![TwinOpSpec::Conv {
        input: 0,
        weight: wgen(&mut rng, WScheme::RandSigned, 5.0, cw, cin),
        bias: bgen(&mut rng, 5.0, cw),
        bias_err: vec![berr; cw],
        weight_rel_err: rel,
        kernel: (cw, cin, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (cin, h, w),
        oshape: (cw, h, w),
    }];
    if let Some((srel, serr)) = affine {
        let scale: Vec<f64> = (0..cw)
            .map(|c| {
                let v: f64 = rng.random_range(0.5..3.0);
                if c % 2 == 0 {
                    v
                } else {
                    -v
                }
            })
            .collect();
        ops.push(TwinOpSpec::ChannelAffine {
            input: 1,
            scale,
            shift: bgen(&mut rng, 2.0, cw),
            scale_rel_err: srel,
            shift_err: vec![serr; cw],
            shape: (cw, h, w),
        });
    }
    // Tensor id of the trunk-0 pre-activation (op `k` writes tensor `k + 1`).
    let pre = ops.len();
    let flat = cw * h * w;
    ops.push(TwinOpSpec::Relu { input: pre });
    ops.push(TwinOpSpec::Flatten { input: pre + 1 });
    ops.push(TwinOpSpec::Gemm {
        input: pre + 2,
        weight: wgen(&mut rng, WScheme::RandSigned, 0.3, n_y, flat),
        bias: bgen(&mut rng, 0.3, n_y),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (n_y, flat),
    });
    ops.push(TwinOpSpec::Relu { input: pre + 3 });
    ops.push(TwinOpSpec::Gemm {
        input: pre + 4,
        weight: wgen(&mut rng, WScheme::ZeroRowSum, 0.3, n_out, n_y),
        bias: bgen(&mut rng, 0.3, n_out),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (n_out, n_y),
    });
    TwinSpec {
        n_in: cin * h * w,
        ops,
    }
}

/// EXACT hull `[min, max]` of every trunk-0 pre-activation of
/// [`fold_ball_spec`] over the input box `[lo, hi]` and — when `ball` — over
/// every member of the certified fold ball (`|W - W'| <= rel |W'|`,
/// `|b - b'| <= berr`, `|s - s'| <= srel |s'|`, `|t - t'| <= serr`); with
/// `ball = false` the stored network alone. The pre-activation is a sum of
/// per-input-coordinate terms `w~ x` (plus the bias), each BILINEAR on its own
/// rectangle, so the extremes are the per-term vertex extremes, summed; the
/// affine `s~ v + t~` is bilinear on `[s-, s+] x [v-, v+]`, again a vertex.
/// No sampling: this is the whole box and the whole ball.
fn fold_ball_hull(
    spec: &TwinSpec,
    lo: &[f64],
    hi: &[f64],
    ball: bool,
) -> Vec<(num_rational::BigRational, num_rational::BigRational)> {
    use num_rational::BigRational as Q;
    use num_traits::{Signed, Zero};
    let q = |v: f64| Q::from_float(v).expect("finite");
    let TwinOpSpec::Conv {
        weight,
        bias,
        bias_err,
        weight_rel_err,
        kernel: (cw, cin, _, _),
        ishape: (_, h, w),
        ..
    } = &spec.ops[0]
    else {
        panic!("fold_ball_spec starts with a conv");
    };
    let p = h * w;
    let rel = if ball { q(*weight_rel_err) } else { Q::zero() };
    let vertex_range = |a: &Q, b: &Q, c: &Q, d: &Q| -> (Q, Q) {
        // extremes of s * v over s in [a, b], v in [c, d]
        let cands = [a * c, a * d, b * c, b * d];
        let mn = cands.iter().min().expect("four").clone();
        let mx = cands.iter().max().expect("four").clone();
        (mn, mx)
    };
    let mut conv = Vec::with_capacity(cw * p);
    for oc in 0..*cw {
        let be = if ball { q(bias_err[oc]) } else { Q::zero() };
        for sp in 0..p {
            let mut mn = q(bias[oc]) - &be;
            let mut mx = q(bias[oc]) + &be;
            for c in 0..*cin {
                let wv = q(weight[oc * cin + c]);
                let dw = &rel * wv.abs();
                let i = c * p + sp;
                let (tmn, tmx) = vertex_range(&(&wv - &dw), &(&wv + &dw), &q(lo[i]), &q(hi[i]));
                mn += tmn;
                mx += tmx;
            }
            conv.push((mn, mx));
        }
    }
    match spec.ops.get(1) {
        Some(TwinOpSpec::ChannelAffine {
            scale,
            shift,
            scale_rel_err,
            shift_err,
            ..
        }) => conv
            .iter()
            .enumerate()
            .map(|(j, (vmn, vmx))| {
                let ch = j / p;
                let s = q(scale[ch]);
                let ds = if ball {
                    q(*scale_rel_err) * s.abs()
                } else {
                    Q::zero()
                };
                let e = if ball { q(shift_err[ch]) } else { Q::zero() };
                let (amn, amx) = vertex_range(&(&s - &ds), &(&s + &ds), vmn, vmx);
                (amn + q(shift[ch]) - &e, amx + q(shift[ch]) + e)
            })
            .collect(),
        _ => conv,
    }
}

/// #fold-ball regression (the `#frozen-dual-state` exact-rational containment
/// gate failed on it, 2026-09-27): the root tableau's trunk-0 box must enclose
/// the pre-activation of EVERY member of the certified fold ball over the
/// WHOLE input box, computed exactly by [`fold_ball_hull`].
///
/// The historical charge put the relative weight/scale error into the
/// COEFFICIENT columns of the sandwich D lane, which moves the lower wall UP
/// at negative inputs (`root.rs` #fold-ball block). On these fixtures (12-dim
/// box straddling 0, one trunk relu, concretize headroom
/// `gamma_36 ~ 4.0e-15`) the pre-fix root missed the hull on 156 of 160
/// neurons by up to 4.3e-11 — including config 0, whose `bias_err = 0` makes
/// the STORED network itself escape — and the fixed root on none
/// (Fraction replicas in the fix report, two independent ones agreeing).
///
/// Non-vacuity: the ball must be LOAD-BEARING (on some neuron its excess over
/// the stored hull exceeds the box's headroom over the ball, so a box that
/// ignored the fold charge would fail); the HISTORICAL coefficient-column
/// charge, forced by the test-only switch
/// (`root::with_legacy_fold_charge_for_test`), must miss the hull on EVERY
/// config (the code-mutation pin); and the checker must reject a box edge one
/// ulp inside the exact hull.
#[test]
fn root_box_encloses_the_exact_fold_ball_hull() {
    use num_rational::BigRational as Q;
    use num_traits::ToPrimitive;
    let q = |v: f64| Q::from_float(v).expect("finite");
    // THE containment predicate (used for the real check AND the mutation).
    let contains = |l: &Q, u: &Q, mn: &Q, mx: &Q| l <= mn && u >= mx;
    let configs: &[(f64, f64, Option<(f64, f64)>)] = &[
        (1e-13, 0.0, None),
        (1e-13, 1e-6, None),
        (1e-12, 1e-3, None),
        (0.0, 0.0, Some((1e-12, 0.0))),
        (1e-13, 1e-6, Some((1e-13, 1e-6))),
    ];
    let mut checked = 0usize;
    let mut load_bearing = 0usize;
    let mut straddles = 0usize;
    let mut failures = Vec::new();
    let mut legacy_blind: Vec<usize> = Vec::new();
    for (ci, &(rel, berr, affine)) in configs.iter().enumerate() {
        let seed = 0xF01D_BA11_u64 ^ ci as u64;
        let spec = fold_ball_spec(rel, berr, affine, seed);
        let net = TwinNet::compile(&spec).expect("compile");
        let mut rng = StdRng::seed_from_u64(seed ^ 0xB0C5);
        let mut lo = Vec::with_capacity(spec.n_in);
        let mut hi = Vec::with_capacity(spec.n_in);
        for _ in 0..spec.n_in {
            let c: f64 = rng.random_range(-0.2..0.2);
            lo.push(c - 1.0);
            hi.push(c + 1.0);
        }
        straddles += lo
            .iter()
            .zip(&hi)
            .filter(|(l, h)| **l < 0.0 && **h > 0.0)
            .count();
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let (l, u) = (&root.layers[0].l, &root.layers[0].u);
        let ball = fold_ball_hull(&spec, &lo, &hi, true);
        let stored = fold_ball_hull(&spec, &lo, &hi, false);
        assert_eq!(ball.len(), l.len(), "config {ci}: hull / box width");
        // MUTATION PIN: the HISTORICAL coefficient-column charge (the
        // pre-#fold-ball walk, forced by the test-only switch) must miss the
        // same exact hull on this config, or the config cannot tell the
        // repaired charge from the defect it repairs. Replicas: at least 28 of
        // 32 neurons escape on every config.
        let legacy = super::root::with_legacy_fold_charge_for_test(|| {
            RootGates::build(&net, &lo, &hi, RoundMode::Outward, None)
        })
        .expect("legacy root");
        assert_eq!(
            legacy.layers[0].l.len(),
            ball.len(),
            "config {ci}: legacy box width"
        );
        let mut legacy_escapes = 0usize;
        for (j, (bmn, bmx)) in ball.iter().enumerate() {
            let (ll, lu) = (q(legacy.layers[0].l[j]), q(legacy.layers[0].u[j]));
            if !contains(&ll, &lu, bmn, bmx) {
                legacy_escapes += 1;
            }
        }
        if legacy_escapes == 0 {
            legacy_blind.push(ci);
        }
        for (j, ((bmn, bmx), (smn, smx))) in ball.iter().zip(&stored).enumerate() {
            checked += 1;
            let (lq, uq) = (q(l[j]), q(u[j]));
            if !contains(&lq, &uq, bmn, bmx) {
                failures.push(format!(
                    "config {ci} (rel={rel:e} berr={berr:e} affine={affine:?}) neuron {j}: \
                     box [{:e}, {:e}] vs exact ball hull [{:e}, {:e}]",
                    l[j],
                    u[j],
                    bmn.to_f64().unwrap_or(f64::NAN),
                    bmx.to_f64().unwrap_or(f64::NAN),
                ));
            }
            if (bmx - smx) > (&uq - bmx) || (smn - bmn) > (bmn - &lq) {
                load_bearing += 1;
            }
            // 1-ulp mutations of THIS box: its upper edge moved to the largest
            // double strictly below the exact hull max, or its lower edge to
            // the smallest double strictly above the hull min, must be
            // rejected by the same predicate.
            let mut up_edge = bmx.to_f64().expect("finite hull");
            while q(up_edge) >= *bmx {
                up_edge = super::rounding::next_down(up_edge);
            }
            let mut lo_edge = bmn.to_f64().expect("finite hull");
            while q(lo_edge) <= *bmn {
                lo_edge = super::rounding::next_up(lo_edge);
            }
            assert!(
                !contains(&lq, &q(up_edge), bmn, bmx) && !contains(&q(lo_edge), &uq, bmn, bmx),
                "config {ci} neuron {j}: the containment predicate accepted a box edge one \
                 ulp inside the exact hull (mutation not caught)"
            );
        }
    }
    assert!(
        checked > 0 && straddles > 0,
        "VOID: no neuron checked or no input coordinate straddles 0"
    );
    assert!(
        load_bearing > 0,
        "VOID: the fold ball never exceeded the box's headroom, so a box that ignored \
         the fold charge would pass too"
    );
    assert!(
        legacy_blind.is_empty(),
        "VOID: the historical coefficient-column fold charge passes on configs \
         {legacy_blind:?}, so they cannot detect the defect #fold-ball repairs"
    );
    assert!(
        failures.is_empty(),
        "ROOT BOX MISSES THE FOLD-BALL HULL (a false-unsat generator): {failures:#?}"
    );
}

// ---- #root-rounding: the root box encloses the EXACT stored-network hull ---

/// `Flatten -> Gemm -> Relu -> Gemm` head on tensor `last` (flat size
/// `flat`). The root walk stops at the last trunk relu, so the head never
/// touches a trunk box; its weights come from their own RNG.
fn rounding_leak_head(ops: &mut Vec<TwinOpSpec>, last: usize, flat: usize, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed ^ 0x4EAD_5EED);
    let (n_y, n_out) = (4usize, 3usize);
    ops.push(TwinOpSpec::Flatten { input: last });
    ops.push(TwinOpSpec::Gemm {
        input: last + 1,
        weight: wgen(&mut rng, WScheme::RandSigned, 0.3, n_y, flat),
        bias: bgen(&mut rng, 0.3, n_y),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (n_y, flat),
    });
    ops.push(TwinOpSpec::Relu { input: last + 2 });
    ops.push(TwinOpSpec::Gemm {
        input: last + 3,
        weight: wgen(&mut rng, WScheme::RandSigned, 0.3, n_out, n_y),
        bias: bgen(&mut rng, 0.3, n_out),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (n_out, n_y),
    });
}

/// #root-rounding regression, the reviewer's hand-checkable counterexample
/// (`review/minimal_leak.py` of the #fold-ball fix report), no fold error at
/// all: `x in [-1, 0.01]`, `z1 = w x + 0.02` with `w = (-1, -1 + 1e-6)` (both
/// trunk-0 relus stable-active on the box), `z2 = relu(z1_0) - relu(z1_1) =
/// -1e-6 x`, whose exact range is the affine hull `[-1e-8, 1e-6]` (+ rounding
/// of the stored weights). The trunk-1 box must contain it.
///
/// The historical coefficient-column rounding charge (forced by the TEST-ONLY
/// switch `root::with_legacy_rounding_charge_for_test`) widened the NEGATIVE
/// upper coefficient of `z2`, which LOWERS the upper wall at the maximiser
/// `x = -1`, and the concretize envelope scales with this cancelling,
/// bias-free row's own ~1e-6 magnitudes: its box ended at
/// 9.999999971380213e-07, 2.891e-15 below the exact max. That walk must still
/// miss (the code-mutation pin), and the repaired one must contain the hull
/// (replica headroom +3.4e-15 on both ends, `fix_root_rounding/validate.py`).
#[test]
fn root_box_encloses_the_exact_minimal_rounding_leak() {
    use num_rational::BigRational as Q;
    use num_traits::{ToPrimitive, Zero};
    let q = |v: f64| Q::from_float(v).expect("finite");
    let w1 = [-1.0f64, -1.0 + 1e-6];
    let b1 = [0.02f64, 0.02];
    let w2 = [1.0f64, -1.0];
    let mut ops = vec![
        TwinOpSpec::Conv {
            input: 0,
            weight: w1.to_vec(),
            bias: b1.to_vec(),
            bias_err: vec![0.0; 2],
            weight_rel_err: 0.0,
            kernel: (2, 1, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (1, 1, 1),
            oshape: (2, 1, 1),
        },
        TwinOpSpec::Relu { input: 1 },
        TwinOpSpec::Conv {
            input: 2,
            weight: w2.to_vec(),
            bias: vec![0.0],
            bias_err: vec![0.0],
            weight_rel_err: 0.0,
            kernel: (1, 2, 1, 1),
            stride: (1, 1),
            pads: (0, 0, 0, 0),
            ishape: (2, 1, 1),
            oshape: (1, 1, 1),
        },
        TwinOpSpec::Relu { input: 3 },
    ];
    rounding_leak_head(&mut ops, 4, 1, 0x0001_EA4F);
    let spec = TwinSpec { n_in: 1, ops };
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = ([-1.0f64], [0.01f64]);
    // Premise, exactly: both trunk-0 pre-activations are positive on the whole
    // box (affine in x, so the vertices decide), hence z2 is affine there.
    for c in 0..2 {
        for x in [lo[0], hi[0]] {
            assert!(
                q(w1[c]) * q(x) + q(b1[c]) > Q::zero(),
                "premise: trunk-0 relu {c} is not active at x = {x}"
            );
        }
    }
    let coef = q(w2[0]) * q(w1[0]) + q(w2[1]) * q(w1[1]);
    let cst = q(w2[0]) * q(b1[0]) + q(w2[1]) * q(b1[1]);
    let (va, vb) = (&coef * q(lo[0]) + &cst, &coef * q(hi[0]) + &cst);
    let (mn, mx) = if va <= vb { (va, vb) } else { (vb, va) };
    let build = || RootGates::build_prec(&net, &lo, &hi, RoundMode::Outward, None, false);
    let fixed = build().expect("root");
    let legacy = super::root::with_legacy_rounding_charge_for_test(build).expect("legacy root");
    for (tag, root) in [("fixed", &fixed), ("legacy", &legacy)] {
        assert_eq!(root.layers.len(), 2, "{tag}: two trunk relus");
        assert!(
            root.layers[0].l.iter().all(|&v| v > 0.0),
            "{tag}: the root no longer sees both trunk-0 relus as stable-active: {:?}",
            root.layers[0].l
        );
    }
    let head = |root: &RootGates| -> (f64, f64) {
        let (l, u) = (root.layers[1].l[0], root.layers[1].u[0]);
        (
            (&mn - q(l)).to_f64().expect("finite"),
            (q(u) - &mx).to_f64().expect("finite"),
        )
    };
    let (fl, fu) = head(&fixed);
    let (ll, lu) = head(&legacy);
    eprintln!(
        "[root-rounding] minimal leak: exact [{:e}, {:e}]; fixed box [{:e}, {:e}] headroom \
         lo {fl:+.3e} up {fu:+.3e}; legacy box [{:e}, {:e}] headroom lo {ll:+.3e} up {lu:+.3e}",
        mn.to_f64().unwrap_or(f64::NAN),
        mx.to_f64().unwrap_or(f64::NAN),
        fixed.layers[1].l[0],
        fixed.layers[1].u[0],
        legacy.layers[1].l[0],
        legacy.layers[1].u[0],
    );
    // MUTATION PIN: the historical charge misses the exact max (replica:
    // -2.891e-15), or this fixture cannot see the defect it pins.
    assert!(
        q(legacy.layers[1].u[0]) < mx,
        "VOID: the historical coefficient-column rounding charge contains the exact \
         hull here, so this fixture no longer detects the #root-rounding defect"
    );
    assert!(
        q(fixed.layers[1].l[0]) <= mn && q(fixed.layers[1].u[0]) >= mx,
        "ROOT TRUNK-1 BOX MISSES THE EXACT STORED-NETWORK HULL (false-unsat generator): \
         box [{:e}, {:e}] vs exact [{:e}, {:e}]",
        fixed.layers[1].l[0],
        fixed.layers[1].u[0],
        mn.to_f64().unwrap_or(f64::NAN),
        mx.to_f64().unwrap_or(f64::NAN),
    );
}

/// Parameters of [`cancel_stable_spec`] the exact hull needs.
struct CancelStable {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    scale: Option<Vec<f64>>,
    residual: bool,
}

/// #root-rounding sweep fixture (`cin = 3`, 2x2 spatial, `cw = 8`): Conv 1x1
/// with NEGATIVE, near-constant columns (`w1[c][i] = a_i + U(-eps, eps)`,
/// `a_i in -[0.5, 1.5)`) and small positive biases, so every trunk-0 relu is
/// stable-active on `[-1, 0.01]^12`; Relu; a ZERO-ROW-SUM Conv 1x1 with zero
/// bias (the composed row cancels to `~eps`); then per `family`: 0 = Relu,
/// 1 = ChannelAffine (scales of both signs, zero shift) -> Relu, 2 = Add with
/// the trunk-0 output (skip) -> Relu. The draw order is load-bearing: the
/// Fraction replica (`fix_root_rounding/validate.py::sweep_net`) mirrors it.
fn cancel_stable_spec(seed: u64, eps: f64, bscale: f64, family: u8) -> (TwinSpec, CancelStable) {
    let (cin, h, w, cw) = (3usize, 2usize, 2usize, 8usize);
    let mut rng = StdRng::seed_from_u64(seed);
    let a: Vec<f64> = (0..cin).map(|_| -rng.random_range(0.5..1.5)).collect();
    let mut w1 = Vec::with_capacity(cw * cin);
    for _ in 0..cw {
        for &ai in &a {
            w1.push(ai + rng.random_range(-eps..eps));
        }
    }
    let b1: Vec<f64> = (0..cw)
        .map(|_| rng.random_range(bscale..1.2 * bscale))
        .collect();
    let mut w2 = Vec::with_capacity(cw * cw);
    for _ in 0..cw {
        let row: Vec<f64> = (0..cw).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut mean = 0.0f64;
        for &v in &row {
            mean += v;
        }
        mean /= cw as f64;
        w2.extend(row.iter().map(|v| v - mean));
    }
    let conv = |input: usize, weight: Vec<f64>, bias: Vec<f64>, ci: usize| TwinOpSpec::Conv {
        input,
        weight,
        bias,
        bias_err: vec![0.0; cw],
        weight_rel_err: 0.0,
        kernel: (cw, ci, 1, 1),
        stride: (1, 1),
        pads: (0, 0, 0, 0),
        ishape: (ci, h, w),
        oshape: (cw, h, w),
    };
    let mut ops = vec![
        conv(0, w1.clone(), b1.clone(), cin),
        TwinOpSpec::Relu { input: 1 },
        conv(2, w2.clone(), vec![0.0; cw], cw),
    ];
    let mut scale = None;
    match family {
        0 => ops.push(TwinOpSpec::Relu { input: 3 }),
        1 => {
            let sc: Vec<f64> = (0..cw)
                .map(|c| {
                    if c % 2 == 0 {
                        rng.random_range(0.5..2.0)
                    } else {
                        -rng.random_range(0.5..2.0)
                    }
                })
                .collect();
            ops.push(TwinOpSpec::ChannelAffine {
                input: 3,
                scale: sc.clone(),
                shift: vec![0.0; cw],
                scale_rel_err: 0.0,
                shift_err: vec![0.0; cw],
                shape: (cw, h, w),
            });
            ops.push(TwinOpSpec::Relu { input: 4 });
            scale = Some(sc);
        }
        _ => {
            ops.push(TwinOpSpec::Add { lhs: 3, rhs: 2 });
            ops.push(TwinOpSpec::Relu { input: 4 });
        }
    }
    let last = ops.len();
    rounding_leak_head(&mut ops, last, cw * h * w, seed);
    (
        TwinSpec {
            n_in: cin * h * w,
            ops,
        },
        CancelStable {
            w1,
            b1,
            w2,
            scale,
            residual: family == 2,
        },
    )
}

/// #root-rounding randomized sweep: 96 nets (3 families x eps in {1e-1, 1e-2,
/// 1e-3, 1e-4} x b1 scale in {0.05, 0.5} x 4 seeds) of [`cancel_stable_spec`]
/// — CANCELLING weights, ALWAYS-ACTIVE trunk-0 relus, the ASYMMETRIC box
/// `[-1, 0.01]^12`. With every trunk-0 relu active (proved exactly per net),
/// the trunk-1 pre-activation is affine in `x`, so its exact hull over the
/// whole box is a per-coordinate vertex sum in BigRational — no sampling. The
/// root's trunk-1 box must contain it on every neuron of every net.
///
/// Mutation pin: the historical coefficient-column rounding charge
/// (`root::with_legacy_rounding_charge_for_test`) must miss the hull on some
/// net of EVERY family (Fraction replica: 32/32 plain, 32/32 affine, 20/32
/// residual, worst -5.9e-14), so each family exercises the repaired site
/// (conv + gate; + channel affine; + residual Add). Replica of the repair:
/// 0/96 escape, min headroom +1.2e-14.
#[test]
fn root_box_encloses_the_exact_affine_hull_on_cancelling_stable_nets() {
    use num_rational::BigRational as Q;
    use num_traits::{ToPrimitive, Zero};
    let q = |v: f64| Q::from_float(v).expect("finite");
    let (cin, p, cw) = (3usize, 4usize, 8usize);
    let (lo_v, hi_v) = (-1.0f64, 0.01f64);
    let (qlo, qhi) = (q(lo_v), q(hi_v));
    let lo = vec![lo_v; cin * p];
    let hi = vec![hi_v; cin * p];
    let mut nets = [0usize; 3];
    let mut legacy_escapes = [0usize; 3];
    let mut failures: Vec<String> = Vec::new();
    let mut min_head = f64::INFINITY;
    let mut checked = 0usize;
    for family in 0u8..3 {
        for eps in [1e-1f64, 1e-2, 1e-3, 1e-4] {
            for bscale in [0.05f64, 0.5] {
                for seed in 0u64..4 {
                    let (spec, cs) = cancel_stable_spec(seed, eps, bscale, family);
                    let net = TwinNet::compile(&spec).expect("compile");
                    // Premise, exactly: every trunk-0 pre-activation is
                    // positive on the whole box (affine => vertex minimum).
                    for c in 0..cw {
                        let mut mn = q(cs.b1[c]);
                        for i in 0..cin {
                            let wv = q(cs.w1[c * cin + i]);
                            let (ta, tb) = (&wv * &qlo, &wv * &qhi);
                            mn += if ta <= tb { ta } else { tb };
                        }
                        assert!(
                            mn > Q::zero(),
                            "premise: family {family} eps {eps:e} b1~{bscale} seed {seed}: \
                             trunk-0 channel {c} is not active on the whole box"
                        );
                    }
                    // Exact hull of every trunk-1 pre-activation.
                    let mut hull: Vec<(Q, Q)> = Vec::with_capacity(cw * p);
                    for n in 0..cw {
                        let mut arow: Vec<Q> = (0..cw).map(|c| q(cs.w2[n * cw + c])).collect();
                        if cs.residual {
                            arow[n] += q(1.0);
                        }
                        let mut cst = Q::zero();
                        let mut coef: Vec<Q> = vec![Q::zero(); cin];
                        for (c, ac) in arow.iter().enumerate() {
                            cst += ac * q(cs.b1[c]);
                            for (i, co) in coef.iter_mut().enumerate() {
                                *co += ac * q(cs.w1[c * cin + i]);
                            }
                        }
                        if let Some(sc) = &cs.scale {
                            let s = q(sc[n]);
                            for co in coef.iter_mut() {
                                *co = &*co * &s;
                            }
                            cst = &cst * &s;
                        }
                        let (mut mn, mut mx) = (cst.clone(), cst);
                        for co in &coef {
                            let (ta, tb) = (co * &qlo, co * &qhi);
                            if ta <= tb {
                                mn += ta;
                                mx += tb;
                            } else {
                                mn += tb;
                                mx += ta;
                            }
                        }
                        for _ in 0..p {
                            hull.push((mn.clone(), mx.clone()));
                        }
                    }
                    let build =
                        || RootGates::build_prec(&net, &lo, &hi, RoundMode::Outward, None, false);
                    let fixed = build().expect("root");
                    let legacy =
                        super::root::with_legacy_rounding_charge_for_test(build).expect("legacy");
                    let fi = usize::from(family);
                    nets[fi] += 1;
                    let mut legacy_missed = false;
                    for (tag, root) in [("fixed", &fixed), ("legacy", &legacy)] {
                        assert_eq!(root.layers.len(), 2, "{tag}: two trunk relus");
                        assert!(
                            root.layers[0].l.iter().all(|&v| v > 0.0),
                            "{tag}: family {family} eps {eps:e} b1~{bscale} seed {seed}: the \
                             root no longer sees every trunk-0 relu as stable-active"
                        );
                        assert_eq!(root.layers[1].l.len(), hull.len(), "{tag}: width");
                    }
                    for (j, (mn, mx)) in hull.iter().enumerate() {
                        let (ll, lu) = (q(legacy.layers[1].l[j]), q(legacy.layers[1].u[j]));
                        if !(ll <= *mn && lu >= *mx) {
                            legacy_missed = true;
                        }
                        let (fl, fu) = (fixed.layers[1].l[j], fixed.layers[1].u[j]);
                        let (flq, fuq) = (q(fl), q(fu));
                        checked += 1;
                        let hd = (mn - &flq).min(&fuq - mx);
                        min_head = min_head.min(hd.to_f64().unwrap_or(f64::NEG_INFINITY));
                        if !(flq <= *mn && fuq >= *mx) {
                            failures.push(format!(
                                "family {family} eps {eps:e} b1~{bscale} seed {seed} neuron {j}: \
                                 box [{fl:e}, {fu:e}] vs exact [{:e}, {:e}]",
                                mn.to_f64().unwrap_or(f64::NAN),
                                mx.to_f64().unwrap_or(f64::NAN),
                            ));
                        }
                    }
                    if legacy_missed {
                        legacy_escapes[fi] += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "[root-rounding] cancelling stable-active sweep: nets {nets:?}, legacy nets escaping \
         {legacy_escapes:?}, fixed neurons checked {checked}, failures {}, min headroom \
         {min_head:+.3e}",
        failures.len()
    );
    assert_eq!(nets, [32, 32, 32], "sweep did not run every net");
    assert!(
        legacy_escapes.iter().all(|&e| e > 0),
        "VOID: the historical coefficient-column rounding charge contains the exact hull on \
         every net of some family ({legacy_escapes:?} escaping of {nets:?}), so that family \
         no longer detects the #root-rounding defect"
    );
    assert!(
        failures.is_empty(),
        "ROOT TRUNK-1 BOX MISSES THE EXACT STORED-NETWORK HULL (false-unsat generator): \
         {failures:#?}"
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
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
        shape: (n_y, flat),
    });
    let g1 = ops.len();
    ops.push(TwinOpSpec::Relu { input: g1 });
    let hr = ops.len();
    ops.push(TwinOpSpec::Gemm {
        input: hr,
        weight: wgen(&mut rng, WScheme::ZeroRowSum, 0.3, n_out, n_y),
        bias: bgen(&mut rng, 0.3, n_out),
        bias_err: Vec::new(),
        weight_rel_err: 0.0,
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
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (n_y, flat),
        },
        TwinOpSpec::Relu { input: 4 },
        // Large head with hard cross-class cancellation over n_y terms.
        TwinOpSpec::Gemm {
            input: 5,
            weight: wgen(&mut rng, WScheme::ZeroRowSum, 1.0, n_out, n_y),
            bias: bgen(&mut rng, 1.0, n_out),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
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
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
                    shape: (12, flat),
                },
                TwinOpSpec::Relu { input: 4 },
                TwinOpSpec::Gemm {
                    input: 5,
                    weight: wgen(&mut rng, WScheme::RandSigned, scale, n_out, 12),
                    bias: bgen(&mut rng, scale, n_out),
                    bias_err: Vec::new(),
                    weight_rel_err: 0.0,
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
        MarginRowOutcome::Unsat(..) => {}
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
                unst_rows: false,
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
        epoch_policy: super::EpochRuntimePolicy {
            tier0_exact: 4,
            ..super::EpochRuntimePolicy::default()
        },
        retained: Some(retained.clone()),
        ..BabConfig::default()
    };
    match MarginRowBab::run(&net, &root, t, &adv, mk_cfg()) {
        MarginRowOutcome::Unsat(stats, _) => {
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
        MarginRowOutcome::Unsat(..) => {
            panic!("MOAT VIOLATION: tier0 lane verified a falsified instance")
        }
        MarginRowOutcome::Unknown { .. } => {}
    }
}
// ===================== END EPOCH-BAB PHASE A TESTS ==========================

// ===================== EPOCH-BAB PHASE B TESTS (#epoch-bab Tier 2) ==========
// Epoch re-linearization: baked-split tableau rebuilds + the nested lane.

/// The default-off parent-frame epoch is a shrink-only certified refresh on
/// exactly the split halfspace it claims, and batching does not move a bit.
///
/// This is the load-bearing oracle for the cheap idx_1298 path: it exercises a
/// residual trunk, exact split baking, downstream prefix re-bounding, gate
/// regeneration, unstable-index rebuilding, and final margin scoring. Sampled
/// preactivations and margins on the matching halfspace must remain enclosed.
#[test]
fn epoch_parent_frame_refresh_is_sound_shrink_only_and_chunk_invariant() {
    let mut rng = StdRng::seed_from_u64(60_1298);
    let spec = tiny_spec(&mut rng, 0.65);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    assert_eq!(
        BabConfig::default().epoch_policy.parent_topk,
        0,
        "incremental epochs must remain dark by default"
    );
    assert!(root.layers.len() >= 2 && !root.layers[0].unst.is_empty());

    let x = sample_box(&mut rng, &root, 1200);
    let sel_all: std::collections::BTreeMap<usize, Vec<usize>> = root
        .layers
        .iter()
        .map(|layer| (layer.op, (0..layer.n).collect()))
        .collect();
    let (y, pre) = net.forward_points(&x, &sel_all).expect("forward");
    let first_pre = &pre[&root.layers[0].op];

    // Prefer a balanced first-layer split, ensuring the sampled soundness
    // oracle has substantial coverage on both sides.
    let (idx, _) = root.layers[0]
        .unst
        .iter()
        .copied()
        .map(|idx| {
            let positive = (0..x.ncols())
                .filter(|&column| first_pre[[idx, column]] >= 0.0)
                .count();
            (idx, positive.min(x.ncols() - positive))
        })
        .max_by_key(|(_, balance)| *balance)
        .expect("unstable split");

    let mut any_tightened = false;
    for dir in [1i8, -1] {
        let split = [(0usize, idx, dir)];
        let (child1, stats1) =
            super::backward_interm::refresh_parent_frame(&net, &root, &split, usize::MAX, 1, None)
                .expect("columnwise refresh");
        let (child17, stats17) =
            super::backward_interm::refresh_parent_frame(&net, &root, &split, usize::MAX, 17, None)
                .expect("chunked refresh");
        assert_eq!(stats1.baked, 1);
        assert!(stats1.columns > 0 && stats1.layers > 0);
        assert_eq!(stats1.columns, stats17.columns);
        assert_eq!(stats1.tightened, stats17.tightened);
        any_tightened |= stats1.tightened > 0;

        for (li, ((parent, a), b)) in root
            .layers
            .iter()
            .zip(&child1.layers)
            .zip(&child17.layers)
            .enumerate()
        {
            for j in 0..parent.n {
                assert!(
                    a.l[j] >= parent.l[j] && a.u[j] <= parent.u[j],
                    "refresh widened layer {li} neuron {j}"
                );
                for (name, av, bv) in [
                    ("l", a.l[j], b.l[j]),
                    ("u", a.u[j], b.u[j]),
                    ("alpha", a.alpha[j], b.alpha[j]),
                    ("s", a.s[j], b.s[j]),
                    ("c", a.c[j], b.c[j]),
                    ("ms", a.ms[j], b.ms[j]),
                ] {
                    assert_eq!(
                        av.to_bits(),
                        bv.to_bits(),
                        "chunking moved {name} at layer {li} neuron {j}"
                    );
                }
            }
            assert_eq!(a.unst, b.unst, "chunking moved unstable frame at {li}");
        }
        let fixed = &child1.layers[0];
        assert!(!fixed.unst.contains(&idx));
        if dir > 0 {
            assert!(fixed.l[idx] >= 0.0);
            assert_eq!(
                (fixed.alpha[idx], fixed.s[idx], fixed.c[idx], fixed.ms[idx]),
                (1.0, 1.0, 0.0, 1.0)
            );
        } else {
            assert!(fixed.u[idx] <= 0.0);
            assert_eq!(
                (fixed.alpha[idx], fixed.s[idx], fixed.c[idx], fixed.ms[idx]),
                (0.0, 0.0, 0.0, 0.0)
            );
        }

        let t = 0usize;
        let adv = [1usize, 2, 3];
        let margins = margins_at(&net, &y, t, &adv);
        let bounds = domain_class_bounds(&net, &child1, t, &adv, &[], &[]);
        let mut on_side = 0usize;
        for column in 0..x.ncols() {
            let zsplit = first_pre[[idx, column]];
            if (dir > 0 && zsplit < 0.0) || (dir < 0 && zsplit > 0.0) {
                continue;
            }
            on_side += 1;
            for (li, layer) in child1.layers.iter().enumerate() {
                let exact = &pre[&layer.op];
                for j in 0..layer.n {
                    let value = exact[[j, column]];
                    assert!(
                        layer.l[j] <= value + 1e-9 && value <= layer.u[j] + 1e-9,
                        "parent-refresh box violation dir={dir} layer={li} neuron={j}: \
                         {value} outside [{},{}]",
                        layer.l[j],
                        layer.u[j]
                    );
                }
            }
            for (class, margin) in margins.iter().enumerate() {
                assert!(
                    bounds[class] <= margin[column] + 1e-9,
                    "parent-refresh margin violation dir={dir} class={class}: \
                     bound {} > exact {}",
                    bounds[class],
                    margin[column]
                );
            }
        }
        assert!(on_side > 100, "degenerate sampled split side: {on_side}");
    }
    assert!(
        any_tightened,
        "parent-frame prefix refresh did not tighten either child on the fixture"
    );
}

/// Refusal cases cannot accidentally turn the parity oracle, an empty request,
/// or an invalid/duplicated split into a refreshed verdict frame.
#[test]
fn epoch_parent_frame_refresh_refuses_non_authoritative_inputs() {
    let mut rng = StdRng::seed_from_u64(61_1298);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let outward = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let parity = RootGates::build(&net, &lo, &hi, RoundMode::Parity, None).expect("parity");
    let idx = outward.layers[0].unst[0];
    let split = [(0usize, idx, 1i8)];
    assert!(
        super::backward_interm::refresh_parent_frame(&net, &parity, &split, 8, 8, None).is_err()
    );
    assert!(
        super::backward_interm::refresh_parent_frame(&net, &outward, &split, 0, 8, None).is_err()
    );
    assert!(super::backward_interm::refresh_parent_frame(
        &net,
        &outward,
        &[(0, idx, 1), (0, idx, 1)],
        8,
        8,
        None,
    )
    .is_err());
    assert!(super::backward_interm::refresh_parent_frame(
        &net,
        &outward,
        &[(0, idx, 0)],
        8,
        8,
        None,
    )
    .is_err());
}

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
        epoch_policy: super::EpochRuntimePolicy {
            depth: 1,
            max_attempts: 8,
            ..super::EpochRuntimePolicy::default()
        },
        epoch_trigger_depth: 1,
        epoch_attempt_cap: 8,
        retain_cfg: Some(retain),
        ..BabConfig::default()
    };
    match MarginRowBab::run(&net, &root, t, &adv, mk_cfg()) {
        MarginRowOutcome::Unsat(stats, _) => {
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
        MarginRowOutcome::Unsat(..) => {
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
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
                shape: (6, 27),
            }, // t7 (y)
            TwinOpSpec::Relu { input: 7 },    // t8 (head relu)
            TwinOpSpec::Gemm {
                input: 8,
                weight: w(4 * 6),
                bias: w(4),
                bias_err: Vec::new(),
                weight_rel_err: 0.0,
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
        MarginRowOutcome::Unsat(stats, _) => {
            assert_eq!(stats.ledger_ok, Some(true), "ledger on cgan-like tree");
        }
        MarginRowOutcome::Unknown { reason, .. } => {
            panic!("cgan-like argmax instance should verify: {reason}")
        }
    }
    match MarginRowBab::run(&net, &root, worst, &[tt], BabConfig::default()) {
        MarginRowOutcome::Unsat(..) => panic!("MOAT VIOLATION on cgan-like fixture"),
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
                    ..
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
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (n_y, flat),
        });
        let g1_id = ops.len();
        ops.push(TwinOpSpec::Relu { input: g1_id });
        let hr_id = ops.len();
        ops.push(TwinOpSpec::Gemm {
            input: hr_id,
            weight: wv(n_out * n_y, rng),
            bias: wv(n_out, rng),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
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
                samples.push(mid); // center
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
                    ..
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
                    ..
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
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
            shape: (cfg.n_y, n_in),
        });
        let hr = ops.len() + 1;
        ops.push(TwinOpSpec::Relu { input: g1 });
        ops.push(TwinOpSpec::Gemm {
            input: hr,
            weight: vecf(rng, cfg.n_out * cfg.n_y, cfg.scale),
            bias: vecf(rng, cfg.n_out, cfg.scale),
            bias_err: Vec::new(),
            weight_rel_err: 0.0,
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

// ===========================================================================
// #alpha-opt: fractional-alpha soundness + optimizer invariants (no GPU).
// ===========================================================================

/// Fractional lower lines are valid DeepPoly relaxations for ANY `alpha` in
/// `[0, 1]` — checked on a grid, together with the repaired upper chord.
#[test]
fn fractional_alpha_gates_sound_on_grid() {
    use super::root::{gates_from_box, repair_upper_lines};
    for &l in &[-2.0, -1.0, -0.3, -1e-3] {
        for &u in &[1e-3, 0.4, 1.0, 2.5] {
            let (_a, mut s, mut c) = gates_from_box(&[l], &[u]);
            repair_upper_lines(&[l], &[u], &mut s, &mut c);
            for alpha in [0.0, 0.125, 0.3, 0.5, 0.75, 0.99, 1.0] {
                for gi in 0..=100 {
                    let y = l + (u - l) * (gi as f64) / 100.0;
                    let relu = y.max(0.0);
                    assert!(
                        alpha * y <= relu + 1e-15,
                        "lower line above relu: alpha={alpha} y={y} on [{l}, {u}]"
                    );
                    assert!(
                        s[0] * y + c[0] >= relu - 1e-12,
                        "repaired chord below relu: y={y} on [{l}, {u}]"
                    );
                }
                // The lower line is valid on ALL of R (box-independent), which
                // is why a frozen tableau stays sound under a changed backward
                // alpha.
                for y in [-10.0, -1.0, 0.0, 1.0, 10.0] {
                    assert!(alpha * y <= y.max(0.0) + 1e-15);
                }
            }
        }
    }
}

/// The certified Outward pass stays SOUND under handcrafted fractional trunk
/// alpha (with `ms = max(alpha, s)` re-derived): every per-class certified
/// lower bound must sit under the true margin at sampled box points.
#[test]
fn fractional_alpha_backward_pass_sound_vs_samples() {
    let mut rng = StdRng::seed_from_u64(97);
    let spec = tiny_spec(&mut rng, 0.65);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let (t, adv) = (0usize, vec![1usize, 2, 3]);
    let mb = MarginBatch::new(&net, t, &adv).expect("mb");
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let gates = head_gates(&ybox, RoundMode::Outward);
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Outward);
    // Fractional alpha on every unstable neuron; `ms` re-derived per the root
    // formula. Stable neurons keep their exact lines.
    let mut dg = DomainGates::default();
    for (li, layer) in root.layers.iter().enumerate() {
        if layer.unst.is_empty() {
            continue;
        }
        let mut alpha = layer.alpha.clone();
        for (k, &j) in layer.unst.iter().enumerate() {
            alpha[j] = match k % 4 {
                0 => 0.17,
                1 => 0.5,
                2 => 0.83,
                _ => 0.99,
            };
        }
        let msv: Vec<f64> = alpha.iter().zip(&layer.s).map(|(a, s)| a.max(*s)).collect();
        dg.layers.insert(
            li,
            super::engine::GateVecs {
                alpha,
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms: msv,
            },
        );
    }
    assert!(
        dg.layers
            .values()
            .any(|gv| gv.alpha.iter().any(|a| *a != 0.0 && *a != 1.0)),
        "fixture produced no fractional alpha (no unstable neurons?)"
    );
    let pass = eng
        .run(&ms.seed, Some(&dg), LaneDir::Lower, None, false)
        .expect("fractional-alpha pass");
    let dj = per_class_direct(&eng, &pass, &ms, 0..mb.nf());
    let x = sample_box(&mut rng, &root, 256);
    let (y, _) = net
        .forward_points(&x, &std::collections::BTreeMap::new())
        .expect("forward");
    let margins = margins_at(&net, &y, t, &adv);
    for (r0, row) in margins.iter().enumerate() {
        for &m in row {
            assert!(
                dj[r0] <= m + 1e-9,
                "UNSOUND under fractional alpha: class {} certified {} > sampled margin {}",
                adv[r0],
                dj[r0],
                m
            );
        }
    }
}

/// The optimizer's structural contract: monotone accept (never worse than the
/// heuristic under its certified objective), alpha in `[0, 1]` moved ONLY at
/// unstable neurons, `s`/`c`/`l`/`u`/`unst` untouched, and the `ms =
/// max(alpha, s)` invariant re-established everywhere.
#[test]
fn alpha_opt_monotone_accept_and_ms_invariant() {
    let mut rng = StdRng::seed_from_u64(53);
    let spec = tiny_spec(&mut rng, 0.65);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.5; 72];
    let hi = vec![0.5; 72];
    let mut root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    type Snap = (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<usize>,
    );
    let snap: Vec<Snap> = root
        .layers
        .iter()
        .map(|l| {
            (
                l.alpha.clone(),
                l.s.clone(),
                l.c.clone(),
                l.ms.clone(),
                l.l.clone(),
                l.u.clone(),
                l.unst.clone(),
            )
        })
        .collect();
    let report = super::alpha_opt::optimize_root_alpha(
        &net,
        &mut root,
        0,
        &[1, 2, 3],
        None,
        super::alpha_opt::AlphaOptCfg {
            iters: 6,
            secs: 60.0,
            ..super::alpha_opt::AlphaOptCfg::default()
        },
    );
    for (li, layer) in root.layers.iter().enumerate() {
        let (a0, s0, c0, ms0, l0, u0, unst0) = &snap[li];
        assert_eq!(&layer.s, s0, "s moved at layer {li}");
        assert_eq!(&layer.c, c0, "c moved at layer {li}");
        assert_eq!(&layer.l, l0, "l moved at layer {li}");
        assert_eq!(&layer.u, u0, "u moved at layer {li}");
        assert_eq!(&layer.unst, unst0, "unst moved at layer {li}");
        let unst: std::collections::BTreeSet<usize> = layer.unst.iter().copied().collect();
        for j in 0..layer.n {
            assert!(
                (0.0..=1.0).contains(&layer.alpha[j]),
                "alpha out of [0,1] at {li}/{j}"
            );
            assert_eq!(
                layer.ms[j],
                layer.alpha[j].max(layer.s[j]),
                "ms invariant broken at {li}/{j}"
            );
            if !unst.contains(&j) {
                assert_eq!(
                    layer.alpha[j], a0[j],
                    "non-unstable alpha moved at {li}/{j}"
                );
                assert_eq!(layer.ms[j], ms0[j], "non-unstable ms moved at {li}/{j}");
            }
        }
    }
    match report {
        Some(r) => {
            assert!(
                r.best >= r.baseline,
                "monotone accept violated: best {} < baseline {}",
                r.best,
                r.baseline
            );
            if !r.accepted {
                for (li, layer) in root.layers.iter().enumerate() {
                    assert_eq!(
                        &layer.alpha, &snap[li].0,
                        "non-accepted run must leave alpha bit-identical (layer {li})"
                    );
                }
            } else {
                assert!(r.moved_neurons > 0, "accepted with zero moved neurons");
                assert!(r.max_dalpha > 0.0);
            }
        }
        None => {
            for (li, layer) in root.layers.iter().enumerate() {
                assert_eq!(
                    &layer.alpha, &snap[li].0,
                    "refused run must leave alpha bit-identical (layer {li})"
                );
            }
        }
    }
}

/// Split piece-fixes must stay EXACT binary lines even when the frozen root
/// carries fractional (optimized) alpha; all other neurons inherit the
/// fractional gates verbatim.
#[test]
fn domain_gates_split_overrides_stay_exact_under_fractional_alpha() {
    let mut rng = StdRng::seed_from_u64(23);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.35; 72];
    let hi = vec![0.35; 72];
    let mut root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    for layer in &mut root.layers {
        let unst = layer.unst.clone();
        for j in unst {
            layer.alpha[j] = 0.37;
            layer.ms[j] = layer.alpha[j].max(layer.s[j]);
        }
    }
    let (li, pos) = root
        .layers
        .iter()
        .enumerate()
        .find_map(|(li, l)| (!l.unst.is_empty()).then_some((li, 0usize)))
        .expect("fixture has an unstable neuron");
    let idx = root.layers[li].unst[pos];
    for dir in [1i8, -1] {
        let dg = domain_gates(&root, &[(li, pos, dir)]);
        let gv = dg.layers.get(&li).expect("override layer present");
        if dir > 0 {
            assert_eq!(
                (gv.alpha[idx], gv.s[idx], gv.c[idx], gv.ms[idx]),
                (1.0, 1.0, 0.0, 1.0),
                "positive piece-fix not exact"
            );
        } else {
            assert_eq!(
                (gv.alpha[idx], gv.s[idx], gv.c[idx], gv.ms[idx]),
                (0.0, 0.0, 0.0, 0.0),
                "negative piece-fix not exact"
            );
        }
        for j in 0..root.layers[li].n {
            if j != idx {
                assert_eq!(gv.alpha[j], root.layers[li].alpha[j]);
                assert_eq!(gv.ms[j], root.layers[li].ms[j]);
            }
        }
    }
}

// ===================== #rebake: joint forward re-bake =======================

/// #rebake obligation 1, re-derived for #root-rounding: `apply_gates`' Outward
/// result must enclose the gate's exact transform FUNCTIONALLY — for every
/// `x~ = (x, 1)` with `|x_i| <= xabs_i`, in real arithmetic over the stored
/// inputs,
///
///   `(M' - D') x~ <= (m - d)*alpha x~`  and  `(M' + D') x~ >= ((m + d)*s (+ c on bias)) x~`,
///
/// i.e. `sum_i |(M'-D')_i - L_i| xabs_i + ((M'-D')_b - L_b) <= 0` and the
/// mirrored upper condition, checked EXACTLY in BigRational. This is the
/// obligation the tableau's sandwich semantics actually needs; the historical
/// per-coefficient one (`M' - D' <= L` coefficientwise) is not sufficient,
/// because a coefficientwise-lower lower wall is HIGHER at a negative input —
/// the #root-rounding defect. The grid covers binary AND fractional alpha
/// (`alpha >> s`, where the old chord coupling is gone), SIGNED `d` (the
/// tableau's D lane goes negative after any gate with `U < L`), bias-column
/// cancellation, and a zero bias row (`c = 0`, `m_b = d_b = 0`), the shape of
/// the counterexample.
///
/// Also pinned: (a) the TEST-ONLY legacy switch reproduces the historical
/// binary line `next_up((dd + 6u(|mm| + dd))(1 + 1e-15))` and fractional line
/// `next_up(dd + u(|mm| + |dd| + 8W))` bit for bit, and (b) those historical
/// lines VIOLATE the functional obligation somewhere in the grid — the
/// discriminating half: this test fails on the pre-repair gate.
#[test]
fn fractional_alpha_apply_gates_widening_covers_the_extra_rounding() {
    use super::rounding::{next_up, UNIT};
    use num_rational::BigRational as Q;
    use num_traits::{Signed, Zero};
    let q = |v: f64| Q::from_float(v).expect("finite");
    let ms_grid = [0.0, 1e-3, -1e-3, 0.7, -0.7, 3.0, -3.0, 1e8, -1e8, 1e-300];
    let d_grid = [0.0, 1e-12, 1e-3, 0.5, 2.0, -0.5, -1e-3];
    let a_grid = [0.0, 1.0, 1e-6, 0.1, 0.37, 0.75, 0.999];
    let s_grid = [0.0, 1e-9, 1e-3, 0.5, 0.999, 1.0];
    let c_grid = [0.0, 1e-9, 0.4, 5.0];
    let n_in = 1usize;
    let naug = n_in + 1;
    let blk = naug.div_ceil(64);
    let mut legacy_violations = 0usize;
    let mut checked = 0usize;
    // Exact functional excess of a computed (mo, do) row over the exact walls
    // (L, U): `<= 0` on both sides means the sandwich encloses them for every
    // `|x| <= xabs`.
    let excess = |mo: &Array2<f64>, do_: &Array2<f64>, l: &[Q; 2], u: &[Q; 2], xe0: &Q| {
        let a_l0 = q(mo[[0, 0]]) - q(do_[[0, 0]]);
        let a_u0 = q(mo[[0, 0]]) + q(do_[[0, 0]]);
        let a_lb = q(mo[[0, 1]]) - q(do_[[0, 1]]);
        let a_ub = q(mo[[0, 1]]) + q(do_[[0, 1]]);
        let low = (&a_l0 - &l[0]).abs() * xe0 + (a_lb - &l[1]);
        let upp = (&u[0] - &a_u0).abs() * xe0 + (&u[1] - a_ub);
        (low, upp)
    };
    for xe0f in [1.0f64, 37.5] {
        let xabs_ext = vec![xe0f, 1.0];
        let xe0 = q(xe0f);
        for &m in &ms_grid {
            for &d in &d_grid {
                // Bias column: the same (m, d), or an exact zero row.
                for (mb, db) in [(m, d), (0.0, 0.0)] {
                    let mm_in = Array2::from_shape_vec((1, naug), vec![m, mb]).expect("shape");
                    let dd_in = Array2::from_shape_vec((1, naug), vec![d, db]).expect("shape");
                    for &sv in &s_grid {
                        for &cv in &c_grid {
                            for &a in &a_grid {
                                let l = [q(a) * (q(m) - q(d)), q(a) * (q(mb) - q(db))];
                                let u = [q(sv) * (q(m) + q(d)), q(sv) * (q(mb) + q(db)) + q(cv)];
                                let gate = || {
                                    super::root::apply_gates(
                                        &mm_in,
                                        &dd_in,
                                        &[a],
                                        &[sv],
                                        &[cv],
                                        n_in,
                                        RoundMode::Outward,
                                        None,
                                        blk,
                                        &xabs_ext,
                                    )
                                };
                                let (mo, do_, _) = gate();
                                let (low, upp) = excess(&mo, &do_, &l, &u, &xe0);
                                assert!(
                                    low <= Q::zero() && upp <= Q::zero(),
                                    "gate sandwich misses the exact transform: m={m} d={d} \
mb={mb} db={db} alpha={a} s={sv} c={cv} xabs={xe0f}: lower excess {:e}, upper excess {:e}",
                                    num_traits::ToPrimitive::to_f64(&low).unwrap_or(f64::NAN),
                                    num_traits::ToPrimitive::to_f64(&upp).unwrap_or(f64::NAN),
                                );
                                checked += 1;
                                // (a) the legacy switch IS the historical code.
                                let (lmo, ldo, _) =
                                    super::root::with_legacy_rounding_charge_for_test(gate);
                                for (i, (mi, di)) in [(m, d), (mb, db)].into_iter().enumerate() {
                                    let lo_c = (mi - di) * a;
                                    let up0 = (mi + di) * sv;
                                    let up_c = if i == n_in { up0 + cv } else { up0 };
                                    #[allow(clippy::manual_midpoint)]
                                    let mmv = (lo_c + up_c) * 0.5;
                                    let ddv = (up_c - lo_c) * 0.5;
                                    let drow_old = if a != 0.0 && a != 1.0 {
                                        let w = lo_c.abs()
                                            + up_c.abs()
                                            + if i == n_in { up0.abs() } else { 0.0 };
                                        next_up(ddv + UNIT * (mmv.abs() + ddv.abs() + 8.0 * w))
                                    } else {
                                        next_up(
                                            (ddv + 6.0 * UNIT * (mmv.abs() + ddv)) * (1.0 + 1e-15),
                                        )
                                    };
                                    assert_eq!(
                                        lmo[[0, i]].to_bits(),
                                        mmv.to_bits(),
                                        "legacy M' moved: m={m} d={d} a={a} s={sv} c={cv} col={i}"
                                    );
                                    assert_eq!(
                                        ldo[[0, i]].to_bits(),
                                        drow_old.to_bits(),
                                        "legacy D' moved: m={m} d={d} a={a} s={sv} c={cv} col={i}"
                                    );
                                }
                                // (b) does the historical line meet the
                                // functional obligation here?
                                let (llow, lupp) = excess(&lmo, &ldo, &l, &u, &xe0);
                                if llow > Q::zero() || lupp > Q::zero() {
                                    legacy_violations += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 40_000, "grid unexpectedly small: {checked}");
    // The obligation was REAL: the historical coefficient-column widening
    // fails it somewhere (e.g. any nonzero coefficient over a zero bias row:
    // its widening moves the wall INWARD at `x = -xabs`). If this ever reads
    // 0 the grid stopped exercising the regime — fix the grid, not the assert.
    eprintln!(
        "[root-rounding] gate functional obligation: {checked} exact checks, historical-line \
         violations {legacy_violations}"
    );
    assert!(
        legacy_violations > 0,
        "the historical coefficient-column widening met the functional obligation on every \
         grid point — the grid no longer exercises the #root-rounding defect"
    );
}

/// #rebake disarmed/binary bit-identity at the BUILD level: rebaking with the
/// build's own (binary, heuristic) alpha as the seed reproduces the plain
/// build bit-for-bit — the seed plumbing and the `apply_gates` branch are
/// provably inert until the optimizer actually moves an alpha.
#[test]
fn rebake_binary_seed_build_is_bit_identical() {
    let mut rng = StdRng::seed_from_u64(7);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.3; 72];
    let hi = vec![0.5; 72];
    let base = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let seed: Vec<Vec<f64>> = base.layers.iter().map(|l| l.alpha.clone()).collect();
    let (rebaked, ret) = RootGates::build_rebaked(
        &net,
        &lo,
        &hi,
        RoundMode::Outward,
        None,
        None,
        &seed,
        None,
        base.root_precision_policy,
    )
    .expect("rebaked");
    assert!(ret.is_none());
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&base.mid), bits(&rebaked.mid));
    assert_eq!(bits(&base.rad), bits(&rebaked.rad));
    assert_eq!(base.layers.len(), rebaked.layers.len());
    for (a, b) in base.layers.iter().zip(&rebaked.layers) {
        assert_eq!(a.op, b.op);
        assert_eq!(bits(&a.l), bits(&b.l), "l moved at op {}", a.op);
        assert_eq!(bits(&a.u), bits(&b.u), "u moved at op {}", a.op);
        assert_eq!(bits(&a.alpha), bits(&b.alpha), "alpha moved at op {}", a.op);
        assert_eq!(bits(&a.s), bits(&b.s), "s moved at op {}", a.op);
        assert_eq!(bits(&a.c), bits(&b.c), "c moved at op {}", a.op);
        assert_eq!(bits(&a.ms), bits(&b.ms), "ms moved at op {}", a.op);
        assert_eq!(a.unst, b.unst, "unst moved at op {}", a.op);
    }
}

/// #rebake obligation 1 end-to-end: a tableau REBUILT with fractional alpha
/// baked through `apply_gates` must still enclose exact pre-activations at
/// sampled box points, and the certified margins from that root must sit
/// under sampled true margins. The seed deliberately includes alpha values
/// far above the layer's `s` (the regime the re-derived widening exists for).
#[test]
fn rebaked_root_boxes_enclose_sampled_preacts_under_fractional_alpha() {
    let mut rng = StdRng::seed_from_u64(7);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.3; 72];
    let hi = vec![0.5; 72];
    let base = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let seed: Vec<Vec<f64>> = base
        .layers
        .iter()
        .map(|l| {
            let mut a = l.alpha.clone();
            for (k, &j) in l.unst.iter().enumerate() {
                a[j] = [0.05, 0.37, 0.63, 0.99][k % 4];
            }
            a
        })
        .collect();
    let (root, _) = RootGates::build_rebaked(
        &net,
        &lo,
        &hi,
        RoundMode::Outward,
        None,
        None,
        &seed,
        None,
        base.root_precision_policy,
    )
    .expect("rebaked");
    // Engagement: the bake must actually carry fractional alpha.
    assert!(
        root.layers
            .iter()
            .any(|l| l.alpha.iter().any(|a| *a != 0.0 && *a != 1.0)),
        "fixture produced no fractional alpha (no unstable neurons?)"
    );
    for lg in &root.layers {
        for j in 0..lg.n {
            assert_eq!(
                lg.ms[j],
                lg.alpha[j].max(lg.s[j]),
                "ms invariant broken at op {} neuron {j}",
                lg.op
            );
            assert!(
                (0.0..=1.0).contains(&lg.alpha[j]),
                "alpha out of range at op {} neuron {j}",
                lg.op
            );
        }
    }
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
                    "rebaked layer op {} neuron {j}: {v} outside [{}, {}]",
                    lg.op,
                    lg.l[j],
                    lg.u[j]
                );
            }
        }
    }
    // End-to-end: certified per-class root bounds from the rebaked build stay
    // under sampled true margins.
    let (t, adv) = (0usize, vec![1usize, 2, 3]);
    let bound_ok = super::rebake::certified_root_bound(&net, &root, t, &adv).expect("root_eval");
    let eng = BackwardEngine::new(&net, &root);
    let re = super::bab::root_eval(&eng, &net, t, &adv).expect("root_eval");
    let (y, _) = net
        .forward_points(&x, &std::collections::BTreeMap::new())
        .expect("forward");
    let margins = margins_at(&net, &y, t, &adv);
    for (k, row) in margins.iter().enumerate() {
        for &mv in row {
            assert!(
                re.dj[k] <= mv + 1e-9,
                "UNSOUND rebaked root: class {} certified {} > sampled margin {}",
                adv[k],
                re.dj[k],
                mv
            );
        }
    }
    assert!((bound_ok - re.dj.iter().copied().fold(f64::INFINITY, f64::min)).abs() <= 1e-12);
}

/// #rebake driver contract: round-level MONOTONE ACCEPT on the certified
/// `root_eval` bound (never worse than the entry build; bit-identical root
/// when nothing is accepted), and the final root's certified bounds stay
/// sound against sampled margins. Also the disarmed entry: with the frozen
/// policy gate off, `maybe_rebake_root` must not touch the root at all.
#[test]
fn rebake_round_monotone_acceptance_and_disarmed_identity() {
    let mut rng = StdRng::seed_from_u64(53);
    let spec = tiny_spec(&mut rng, 0.65);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.5; 72];
    let hi = vec![0.5; 72];
    let mut root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let (t, adv) = (0usize, vec![1usize, 2, 3]);
    let bits_of = |r: &RootGates| -> Vec<Vec<u64>> {
        r.layers
            .iter()
            .flat_map(|l| {
                [&l.l, &l.u, &l.alpha, &l.s, &l.c, &l.ms]
                    .into_iter()
                    .map(|v| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>())
            })
            .collect()
    };
    // Disarmed entry: frozen policy gate off => no-op.
    let snap0 = bits_of(&root);
    let mut retained = None;
    assert!(super::rebake::maybe_rebake_root(
        &net,
        &mut root,
        &mut retained,
        None,
        t,
        &adv,
        None,
        0.01,
        super::RootAlphaRuntimePolicy::default(),
        super::BackwardIntermRuntimePolicy::default(),
    )
    .is_none());
    assert_eq!(
        snap0,
        bits_of(&root),
        "disarmed maybe_rebake_root moved the root"
    );
    assert!(retained.is_none());
    // Armed loop, called below the gate.
    let entry_bound = super::rebake::certified_root_bound(&net, &root, t, &adv).expect("baseline");
    let snap1 = bits_of(&root);
    let rebake_policy = super::RootAlphaRuntimePolicy {
        rebake_enabled: true,
        rebake_rounds: 3,
        rebake_secs: 60.0,
        ..super::RootAlphaRuntimePolicy::default()
    };
    let report = super::rebake::rebake_root(
        &net,
        &mut root,
        &mut retained,
        None,
        t,
        &adv,
        None,
        0.01,
        rebake_policy,
        None,
    )
    .expect("tiny net with unstable neurons must not refuse");
    assert!(
        (report.baseline - entry_bound).abs() <= 1e-12,
        "baseline {} != independent root_eval {}",
        report.baseline,
        entry_bound
    );
    assert!(
        report.best >= report.baseline,
        "round-monotone accept violated: best {} < baseline {}",
        report.best,
        report.baseline
    );
    assert!(report.rounds_accepted <= report.rounds_run);
    let final_bound = super::rebake::certified_root_bound(&net, &root, t, &adv).expect("final");
    assert!(
        (final_bound - report.best).abs() <= 1e-12,
        "published root ({final_bound}) disagrees with the accepted round ({})",
        report.best
    );
    if report.rounds_accepted == 0 {
        assert_eq!(
            snap1,
            bits_of(&root),
            "no round accepted, yet the root moved"
        );
    } else {
        assert!(
            report.best > report.baseline,
            "accepted rounds without a strict improvement"
        );
    }
    // Soundness of whatever build the loop left behind.
    let eng = BackwardEngine::new(&net, &root);
    let re = super::bab::root_eval(&eng, &net, t, &adv).expect("root_eval");
    let x = sample_box(&mut rng, &root, 128);
    let (y, _) = net
        .forward_points(&x, &std::collections::BTreeMap::new())
        .expect("forward");
    let margins = margins_at(&net, &y, t, &adv);
    for (k, row) in margins.iter().enumerate() {
        for &mv in row {
            assert!(
                re.dj[k] <= mv + 1e-9,
                "UNSOUND rebaked root: class {} certified {} > sampled margin {}",
                adv[k],
                re.dj[k],
                mv
            );
        }
    }
}

/// The #alpha-opt capture returns the SAME pre-transform coefficients as the
/// established Tier-0 retained capture when retention keeps every unstable
/// neuron (discriminating gate for the new capture path).
#[test]
fn unst_rows_capture_matches_retained_capture() {
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
        RootGates::build_retaining(&net, &lo, &hi, RoundMode::Outward, None, Some(&cfg), &[])
            .expect("root");
    let ret = ret.expect("retention requested");
    let eng = BackwardEngine::new(&net, &root);
    let (t, adv) = (0usize, vec![1usize, 2, 3]);
    let mb = MarginBatch::new(&net, t, &adv).expect("mb");
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let gates = head_gates(&ybox, RoundMode::Outward);
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Outward);
    let pass = eng
        .run_collect(
            &ms.seed,
            None,
            LaneDir::Lower,
            None,
            super::engine::Collect {
                unst_abs: false,
                rows: Some(&ret),
                unst_rows: true,
            },
        )
        .expect("pass");
    let ur = pass.unst_rows.as_ref().expect("unst_rows captured");
    let cr = pass.coll_rows.as_ref().expect("retained rows captured");
    let mut checked = 0usize;
    for (li, layer) in root.layers.iter().enumerate() {
        if layer.unst.is_empty() {
            continue;
        }
        // Full retention keeps every unstable neuron in ascending order, so
        // the two captures index the same rows at the same positions.
        assert_eq!(ret.layers[li].idx, layer.unst);
        let u = ur.get(&li).expect("unst layer captured");
        let c = cr.get(&li).expect("retained layer captured");
        assert_eq!(u.nrows(), layer.unst.len());
        assert_eq!(u, c, "pre-transform capture mismatch at layer {li}");
        checked += 1;
    }
    assert!(checked > 0, "fixture has no unstable layers");
}

// ===========================================================================
// #margin-row-beta (BUILD 1) + #backward-interm (BUILD 2).
//
// Soundness contracts under test:
//  * beta: ANY beta >= 0 yields a valid bound on the split region (weak
//    duality) — checked against exact forward samples restricted to the
//    region, in Outward mode (so the E-lane invariant at the application
//    site is exercised too);
//  * beta: an absent/zero/sign-less term is BIT-IDENTICAL to no term at all
//    (the -0.0 perturbation hazard);
//  * beta: the stacked arm matches the independent passes bit-for-bit;
//  * backward-interm: shrink-only vs the forward-only build, chunk-count
//    invariant, zero-budget bit-identity, and sampled enclosure of the
//    published boxes;
//  * clip calibration: slack must be recovered against the FORWARD bounds —
//    the tightened ones under-recover (clamp to 0), the false-`unsat` trap
//    root.rs's `bi_cal` exists to prevent.
// ===========================================================================

/// No-conv chain fixture shared by the beta tests: depth-3 trunk, 8-wide
/// head, certified Outward root.
fn beta_fixture(seed: u64) -> (TwinNet, RootGates) {
    let spec = no_conv_chain_spec(6, 3, 8, 3, seed);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.8; 6];
    let hi = vec![0.8; 6];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    (net, root)
}

#[test]
fn beta_zero_terms_never_perturb_a_pass() {
    let (net, root) = beta_fixture(0xB37A_0001);
    let eng = BackwardEngine::new(&net, &root);
    let (li, pos) = (1usize, 0usize);
    assert!(!root.layers[li].unst.is_empty(), "fixture: layer 1 stable");
    let split = (li, pos, -1i8);
    let dom0 = domain_gates(&root, &[split]);
    let seed = eng.identity_seed();
    let base = eng
        .run(&seed, Some(&dom0), LaneDir::Lower, None, false)
        .expect("base pass");
    // `set_terms` filters beta == 0 (must not install a term at all).
    let mut dom1 = domain_gates(&root, &[split]);
    super::beta::set_terms(&mut dom1, &root, &[split], &[0.0]);
    assert!(dom1.beta.is_empty(), "beta = 0 installed a term");
    // Even manually injected degenerate terms are skipped by the engine:
    // beta = 0 (would flip -0.0 bits), sign = 0 (delta would be +/-0.0).
    let neuron = root.layers[li].unst[pos];
    dom1.beta
        .entry(li)
        .or_default()
        .push(super::engine::BetaSplit {
            neuron,
            sign: -1,
            beta: 0.0,
        });
    dom1.beta
        .entry(li)
        .or_default()
        .push(super::engine::BetaSplit {
            neuron,
            sign: 0,
            beta: 1.0,
        });
    let same = eng
        .run(&seed, Some(&dom1), LaneDir::Lower, None, false)
        .expect("degenerate-term pass");
    for (x, y) in base.a.iter().zip(same.a.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "coefficients moved");
    }
    for (x, y) in base.b.iter().zip(same.b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "bias moved");
    }
    for (x, y) in base.eb.iter().zip(same.eb.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "bias error moved");
    }
    let (Some(ea), Some(eb2)) = (base.e.as_ref(), same.e.as_ref()) else {
        panic!("outward passes must carry error lanes");
    };
    for (x, y) in ea.iter().zip(eb2.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "error lane moved");
    }
}

#[test]
fn beta_bounds_stay_sound_on_the_split_region_for_any_beta() {
    let (net, root) = beta_fixture(0xB37A_0002);
    let eng = BackwardEngine::new(&net, &root);
    let mut rng = StdRng::seed_from_u64(77);
    let x = sample_box(&mut rng, &root, 600);
    for &(li, pos) in &[(1usize, 0usize), (2usize, 0usize)] {
        let rec = &root.layers[li];
        assert!(!rec.unst.is_empty(), "fixture: layer {li} has no unstable");
        let neuron = rec.unst[pos];
        let relu_op = rec.op;
        let mut pre_sel = std::collections::BTreeMap::new();
        pre_sel.insert(relu_op, vec![neuron]);
        let (y, pre) = net.forward_points(&x, &pre_sel).expect("forward");
        let z = &pre[&relu_op];
        for &sign in &[1i8, -1i8] {
            let split = (li, pos, sign);
            // beta = 25 deliberately stresses the E-lane at the application
            // site: an under-scaled error carry would surface here as a
            // lower bound exceeding a sampled y (the false-UNSAT direction).
            for &beta in &[0.0f64, 0.05, 0.5, 2.0, 25.0] {
                let mut dom = domain_gates(&root, &[split]);
                super::beta::set_terms(&mut dom, &root, &[split], &[beta]);
                let seed = eng.identity_seed();
                let plo = eng
                    .run(&seed, Some(&dom), LaneDir::Lower, None, false)
                    .expect("lower pass");
                let lows = eng.concretize_lower(&plo);
                let pup = eng
                    .run(&seed, Some(&dom), LaneDir::Upper, None, false)
                    .expect("upper pass");
                let ups = eng.concretize_upper(&pup);
                let mut kept = 0usize;
                for bcol in 0..x.ncols() {
                    let zb = z[[0, bcol]];
                    if (sign > 0 && zb < 0.0) || (sign < 0 && zb > 0.0) {
                        continue; // outside this split's region
                    }
                    kept += 1;
                    for i in 0..y.nrows() {
                        let yv = y[[i, bcol]];
                        assert!(
                            lows[i] <= yv + 1e-7,
                            "li={li} sign={sign} beta={beta} row {i}: certified lower \
{} EXCEEDS sampled y {yv} (false-UNSAT direction)",
                            lows[i]
                        );
                        assert!(
                            ups[i] >= yv - 1e-7,
                            "li={li} sign={sign} beta={beta} row {i}: certified upper \
{} BELOW sampled y {yv}",
                            ups[i]
                        );
                    }
                }
                assert!(kept > 20, "vacuous region filter: kept={kept}");
            }
        }
    }
}

#[test]
fn beta_domain_stacked_matches_independent_passes() {
    let (net, root) = beta_fixture(0xB37A_0003);
    let eng = BackwardEngine::new(&net, &root);
    let n_y = net.n_y;
    let split_a = (1usize, 0usize, -1i8);
    let split_b = (2usize, 0usize, 1i8);
    let mut dom_a = domain_gates(&root, &[split_a]);
    super::beta::set_terms(&mut dom_a, &root, &[split_a], &[0.7]);
    let mut dom_b = domain_gates(&root, &[split_b]);
    super::beta::set_terms(&mut dom_b, &root, &[split_b], &[1.3]);
    let seed1 = eng.identity_seed();
    let pa = eng
        .run(&seed1, Some(&dom_a), LaneDir::Lower, None, false)
        .expect("a");
    let pb = eng
        .run(&seed1, Some(&dom_b), LaneDir::Lower, None, false)
        .expect("b");
    let mut s = Array2::<f64>::zeros((n_y, 2 * n_y));
    for j in 0..n_y {
        s[[j, j]] = 1.0;
        s[[j, n_y + j]] = 1.0;
    }
    let ex_a = Exceptions::default();
    let ex_b = Exceptions::default();
    let blocks = vec![
        RowDomainGateBlock {
            columns: 0..n_y,
            gates: &dom_a,
            exceptions: &ex_a,
        },
        RowDomainGateBlock {
            columns: n_y..2 * n_y,
            gates: &dom_b,
            exceptions: &ex_b,
        },
    ];
    let stacked = eng
        .run_domain_stacked(&DomainStackSeed { s, e: None }, &blocks, LaneDir::Lower)
        .expect("stacked");
    let n_in = root.mid.len();
    for c in 0..n_y {
        for j in 0..n_in {
            assert_eq!(
                stacked.a[[j, c]].to_bits(),
                pa.a[[j, c]].to_bits(),
                "stacked block A coefficient moved at ({j}, {c})"
            );
            assert_eq!(
                stacked.a[[j, n_y + c]].to_bits(),
                pb.a[[j, c]].to_bits(),
                "stacked block B coefficient moved at ({j}, {c})"
            );
        }
        assert_eq!(stacked.b[c].to_bits(), pa.b[c].to_bits());
        assert_eq!(stacked.b[n_y + c].to_bits(), pb.b[c].to_bits());
        assert_eq!(stacked.eb[c].to_bits(), pa.eb[c].to_bits());
        assert_eq!(stacked.eb[n_y + c].to_bits(), pb.eb[c].to_bits());
    }
}

#[test]
fn beta_step_projection_and_term_filters() {
    let (_net, root) = beta_fixture(0xB37A_0004);
    let trunk = [(1usize, 0usize, -1i8), (2usize, 0usize, 1i8)];
    let terms = super::beta::terms_for_test(&root, &trunk, &[0.0, 2.0]);
    assert_eq!(terms.len(), 1, "only the positive beta installs a term");
    let t = &terms[&2][0];
    assert_eq!(t.neuron, root.layers[2].unst[0]);
    assert_eq!(t.sign, 1);
    assert_eq!(t.beta, 2.0);
    // apply_step: g > 0 raises by eta*scale, g < 0 lowers with projection at
    // zero, an all-zero gradient proposes nothing.
    let dir = [(1.0, 0.5), (-1.0, 10.0)];
    let stepped = super::beta::apply_step(&[0.0, 0.2], &dir, 0.5).expect("moves");
    assert!((stepped[0] - 0.25).abs() < 1e-12);
    assert_eq!(stepped[1], 0.0, "projection at zero failed");
    assert!(super::beta::apply_step(&[0.3, 0.3], &[(0.0, 1.0), (0.0, 1.0)], 0.5).is_none());
    assert!(
        super::beta::apply_step(&[0.3], &dir, 0.5).is_none(),
        "shape drift"
    );
}

// #margin-row-beta-percol: the per-column engine arm is proven by REDUCTION to
// the shared arm rather than by re-sampling: (a) per-column terms on EVERY
// column reproduce the shared pass bit-for-bit, and (b) terms on ONE column
// leave every other column bit-identical to the no-beta pass while making the
// owned column bit-identical to the shared pass. Both endpoints are covered by
// `beta_bounds_stay_sound_on_the_split_region_for_any_beta` above, so the
// per-column path inherits its soundness sampling transitively.
#[test]
fn beta_percol_all_columns_matches_shared_and_isolates_columns() {
    let (net, root) = beta_fixture(0xB37A_0005);
    let eng = BackwardEngine::new(&net, &root);
    let n_y = net.n_y;
    let split = (1usize, 0usize, -1i8);
    let seed = eng.identity_seed();
    let beta = [0.7f64];
    // Shared pass (the established arm).
    let mut dom_shared = domain_gates(&root, &[split]);
    super::beta::set_terms(&mut dom_shared, &root, &[split], &beta);
    let shared = eng
        .run(&seed, Some(&dom_shared), LaneDir::Lower, None, false)
        .expect("shared pass");
    // No-beta pass (the isolation baseline).
    let dom_base = domain_gates(&root, &[split]);
    let base = eng
        .run(&seed, Some(&dom_base), LaneDir::Lower, None, false)
        .expect("base pass");
    // (a) Per-column terms on EVERY column == shared, bit-for-bit.
    let mut dom_pc_all = domain_gates(&root, &[split]);
    let cols_all: Vec<(usize, &[f64])> = (0..n_y).map(|c| (c, &beta[..])).collect();
    super::beta::set_terms_pc(&mut dom_pc_all, &root, &[split], &cols_all);
    assert!(
        dom_pc_all.beta.is_empty() && !dom_pc_all.beta_pc.is_empty(),
        "pc terms must ride beta_pc only"
    );
    let pc_all = eng
        .run(&seed, Some(&dom_pc_all), LaneDir::Lower, None, false)
        .expect("pc-all pass");
    for (x, y) in shared.a.iter().zip(pc_all.a.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "pc-all coefficients moved");
    }
    let (Some(es), Some(ep)) = (shared.e.as_ref(), pc_all.e.as_ref()) else {
        panic!("outward passes must carry error lanes");
    };
    for (x, y) in es.iter().zip(ep.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "pc-all error lane moved");
    }
    for (x, y) in shared.b.iter().zip(pc_all.b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "pc-all bias moved");
    }
    // (b) Terms on ONE column: that column == shared, every other == base.
    let own = 2usize.min(n_y - 1);
    let mut dom_pc_one = domain_gates(&root, &[split]);
    super::beta::set_terms_pc(&mut dom_pc_one, &root, &[split], &[(own, &beta[..])]);
    let pc_one = eng
        .run(&seed, Some(&dom_pc_one), LaneDir::Lower, None, false)
        .expect("pc-one pass");
    let n_in = root.mid.len();
    let eo = pc_one.e.as_ref().expect("outward error lane");
    let eb2 = base.e.as_ref().expect("outward error lane");
    for j in 0..n_in {
        for c in 0..n_y {
            let (want_a, want_e, what) = if c == own {
                (shared.a[[j, c]], es[[j, c]], "owned column != shared")
            } else {
                (base.a[[j, c]], eb2[[j, c]], "foreign column != base")
            };
            assert_eq!(pc_one.a[[j, c]].to_bits(), want_a.to_bits(), "{what}");
            assert_eq!(eo[[j, c]].to_bits(), want_e.to_bits(), "{what} (e)");
        }
    }
}

#[test]
fn beta_percol_seed_shift_isolates_columns() {
    let n_y = 4usize;
    let nf = 3usize;
    let mut s = Array2::<f64>::zeros((n_y, nf));
    for i in 0..n_y {
        for c in 0..nf {
            s[[i, c]] = (i as f64) - 0.5 * (c as f64);
        }
    }
    let e = Array2::<f64>::from_elem((n_y, nf), 1e-13);
    let base = DomainStackSeed {
        s: s.clone(),
        e: Some(e.clone()),
    };
    let heads = [(1usize, 1i8), (3usize, -1i8)];
    let hb = [0.4f64, 1.1];
    let shared = super::beta::seed_with_head_terms(&base, &heads, &hb, true).expect("shared shift");
    let own = 1usize;
    let pc = super::beta::seed_with_head_terms_pc(&base, &heads, &[(own, &hb[..])], true)
        .expect("pc shift");
    for i in 0..n_y {
        for c in 0..nf {
            let (want_s, want_e, what) = if c == own {
                (
                    shared.s[[i, c]],
                    shared.e.as_ref().expect("e")[[i, c]],
                    "owned column != shared shift",
                )
            } else {
                (base.s[[i, c]], e[[i, c]], "foreign column shifted")
            };
            assert_eq!(pc.s[[i, c]].to_bits(), want_s.to_bits(), "{what}");
            assert_eq!(
                pc.e.as_ref().expect("e")[[i, c]].to_bits(),
                want_e.to_bits(),
                "{what} (e)"
            );
        }
    }
    // Refusals: outward without an error lane; out-of-range column.
    let bare = DomainStackSeed {
        s: s.clone(),
        e: None,
    };
    assert!(
        super::beta::seed_with_head_terms_pc(&bare, &heads, &[(0, &hb[..])], true).is_none(),
        "outward + e=None must refuse"
    );
    assert!(
        super::beta::seed_with_head_terms_pc(&base, &heads, &[(nf, &hb[..])], true).is_none(),
        "out-of-range column must refuse"
    );
}

#[test]
fn beta_percol_term_filters_and_grouping() {
    let (_net, root) = beta_fixture(0xB37A_0006);
    let trunk = [(1usize, 0usize, -1i8), (2usize, 0usize, 1i8)];
    let mut dom = DomainGates::default();
    // Column 0 prices only the layer-2 split (layer-1 beta is 0); column 3
    // prices only the layer-1 split.
    let c0 = [0.0f64, 2.0];
    let c3 = [1.0f64, 0.0];
    super::beta::set_terms_pc(&mut dom, &root, &trunk, &[(0, &c0[..]), (3, &c3[..])]);
    assert_eq!(dom.beta_pc.len(), 2, "two layers touched");
    let l1 = &dom.beta_pc[&1];
    assert_eq!(l1.len(), 1);
    assert_eq!(l1[0].col, 3);
    assert_eq!(l1[0].terms.len(), 1);
    assert_eq!(l1[0].terms[0].beta, 1.0);
    assert_eq!(l1[0].terms[0].sign, -1);
    let l2 = &dom.beta_pc[&2];
    assert_eq!(l2.len(), 1);
    assert_eq!(l2[0].col, 0);
    assert_eq!(l2[0].terms[0].beta, 2.0);
    assert_eq!(l2[0].terms[0].sign, 1);
    // All-zero vectors install nothing.
    let z = [0.0f64, 0.0];
    super::beta::set_terms_pc(&mut dom, &root, &trunk, &[(0, &z[..])]);
    assert!(dom.beta_pc.is_empty(), "zero betas installed pc terms");
}

#[test]
fn backward_interm_shrink_only_sound_and_chunk_invariant() {
    let mut rng = StdRng::seed_from_u64(0xB1_0001);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let build = |bi: Option<super::backward_interm::BiCfg>| {
        RootGates::build_retaining_bi(&net, &lo, &hi, RoundMode::Outward, None, bi)
            .expect("build")
            .0
    };
    let off = build(None);
    // `chunk_secs: 0.0` pins the FIXED grain, which is what the chunk-invariance
    // assertion below is about; the sizer's own invariance is covered by
    // `backward_interm_pass_sizer_is_bit_identical_to_the_fixed_grain`.
    let cfg = |chunk: usize| super::backward_interm::BiCfg {
        yield_window: 0,
        yield_ratio: 0.0,
        secs: 60.0,
        chunk,
        topk: 512,
        chunk_secs: 0.0,
        interm_alpha: 0,
    };
    let on1 = build(Some(cfg(1)));
    let on8 = build(Some(cfg(8)));
    let on256 = build(Some(cfg(256)));
    let top1 = build(Some(super::backward_interm::BiCfg {
        yield_window: 0,
        yield_ratio: 0.0,
        secs: 60.0,
        chunk: 8,
        topk: 1,
        chunk_secs: 0.0,
        interm_alpha: 0,
    }));
    let zero = build(Some(super::backward_interm::BiCfg {
        yield_window: 0,
        yield_ratio: 0.0,
        secs: 0.0,
        chunk: 4,
        topk: 512,
        chunk_secs: 0.0,
        interm_alpha: 0,
    }));
    let off_exec = off.backward_interm_execution;
    assert!(!off_exec.enabled);
    assert!(off_exec.eligible_layers > 0);
    assert!(off_exec.suppressed_setup_layers > 0);
    assert_eq!(off_exec.layers_started, 0);
    assert_eq!(off_exec.chunks_completed, 0);
    assert_eq!(off_exec.columns_completed, 0);
    assert_eq!(off_exec.outcome(), "disabled");
    let on_exec = on1.backward_interm_execution;
    assert!(on_exec.enabled);
    assert_eq!(on_exec.configured_chunk, 1);
    assert!(on_exec.layers_started > 0);
    assert!(on_exec.chunks_completed > 0);
    assert!(on_exec.columns_completed > 0);
    assert!(on_exec.chunk_counterfactual_chunks > 0);
    assert!(!on_exec.policy_mismatch);
    assert_eq!(on_exec.outcome(), "completed");
    let fixed_width_exec = on8.backward_interm_execution;
    assert!(fixed_width_exec.fixed_width_chunks > 0);
    assert!(fixed_width_exec.chunk_ceiling_binding_chunks > 0);
    let default_chunk_exec = on256.backward_interm_execution;
    assert_eq!(default_chunk_exec.configured_chunk, 256);
    assert_eq!(default_chunk_exec.chunk_counterfactual_chunks, 0);
    assert_eq!(default_chunk_exec.chunk_ceiling_binding_chunks, 0);
    assert!(default_chunk_exec.fixed_width_chunks > 0);
    let top1_exec = top1.backward_interm_execution;
    assert!(top1_exec.topk_binding_layers > 0);
    assert_eq!(top1_exec.configured_topk, 1);
    let zero_exec = zero.backward_interm_execution;
    assert!(zero_exec.enabled);
    assert!(zero_exec.eligible_layers > 0);
    assert_eq!(zero_exec.layers_started, 0);
    assert_eq!(zero_exec.columns_completed, 0);
    assert!(zero_exec.budget_exhausted);
    assert!(!zero_exec.deadline_capped);
    assert_eq!(zero_exec.outcome(), "exhausted");
    assert_eq!(off.layers.len(), on1.layers.len());
    // Shrink-only against the forward-only build.
    let mut moved = 0usize;
    for (fwd, bi) in off.layers.iter().zip(&on1.layers) {
        for j in 0..fwd.n {
            assert!(bi.l[j] >= fwd.l[j], "lower LOOSENED at neuron {j}");
            assert!(bi.u[j] <= fwd.u[j], "upper LOOSENED at neuron {j}");
            if bi.l[j] > fwd.l[j] || bi.u[j] < fwd.u[j] {
                moved += 1;
            }
        }
    }
    eprintln!("[test backward-interm] tightened neurons: {moved}");
    // Chunk count must not change any value (columns are independent passes).
    for (a, b) in on1.layers.iter().zip(&on8.layers) {
        for j in 0..a.n {
            assert_eq!(a.l[j].to_bits(), b.l[j].to_bits(), "chunking moved l");
            assert_eq!(a.u[j].to_bits(), b.u[j].to_bits(), "chunking moved u");
        }
    }
    // Zero budget = disarmed, bit-identical to the forward-only build.
    for (a, b) in off.layers.iter().zip(&zero.layers) {
        for j in 0..a.n {
            assert_eq!(a.l[j].to_bits(), b.l[j].to_bits(), "zero-budget moved l");
            assert_eq!(a.u[j].to_bits(), b.u[j].to_bits(), "zero-budget moved u");
        }
    }
    // Sampled enclosure of the PUBLISHED (tightened) boxes.
    let mut rng2 = StdRng::seed_from_u64(9);
    let x = sample_box(&mut rng2, &on1, 400);
    let mut pre_sel = std::collections::BTreeMap::new();
    for l in &on1.layers {
        pre_sel.insert(l.op, (0..l.n).collect::<Vec<_>>());
    }
    let (_y, pre) = net.forward_points(&x, &pre_sel).expect("forward");
    for l in &on1.layers {
        let z = &pre[&l.op];
        for j in 0..l.n {
            for b in 0..x.ncols() {
                let zv = z[[j, b]];
                assert!(
                    l.l[j] <= zv + 1e-9 && l.u[j] >= zv - 1e-9,
                    "published box [{}, {}] excludes sampled z {zv} (neuron {j})",
                    l.l[j],
                    l.u[j]
                );
            }
        }
    }
}

/// #interm-alpha: the per-START-NODE prefix alpha candidates.
///
/// THREE OBLIGATIONS, and the third is what makes the test evidence-carrying
/// rather than a tautology:
///
/// 1. **Shrink-only against the SHIPPED phase.** The dosed build's published
///    box must be a SUBSET of the dose-0 build's, neuron by neuron. This is
///    the moat property: the axis can only ever hand `gates_from_box` a
///    tighter interval than the phase already handed it, so no banked proof
///    can be lost to a looser relaxation. A regression here is a real defect,
///    not a tuning question.
/// 2. **The published box still encloses the true range.** Sampled forward
///    points must lie inside the DOSED box. A shrink-only intersection of
///    invalid enclosures would pass (1) and fail this — which is exactly the
///    false-`unsat` shape the whole lane's soundness argument is about.
/// 3. **The candidates actually MOVE something.** If the dose changed no
///    neuron on this net, (1) and (2) would both pass vacuously and the test
///    would be a no-op guarding nothing. Asserting movement means "the lever
///    was armed but inert" fails here, in CI, instead of being discovered as
///    a VOID measurement on the box.
#[test]
fn interm_alpha_per_start_node_shrinks_only_and_still_encloses() {
    let mut rng = StdRng::seed_from_u64(0xB1_0001);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let build = |dose: usize| {
        RootGates::build_retaining_bi(
            &net,
            &lo,
            &hi,
            RoundMode::Outward,
            None,
            Some(super::backward_interm::BiCfg {
                yield_window: 0,
                yield_ratio: 0.0,
                secs: 600.0,
                chunk: 8,
                topk: 512,
                chunk_secs: 0.0,
                interm_alpha: dose,
            }),
        )
        .expect("build")
        .0
    };
    let base = build(0);
    let dosed = build(super::backward_interm::INTERM_ALPHA_POLICIES);
    assert_eq!(base.layers.len(), dosed.layers.len());

    // (1) + (3)
    let mut moved = 0usize;
    let mut newly_stable = 0usize;
    for (b, d) in base.layers.iter().zip(&dosed.layers) {
        assert_eq!(b.n, d.n);
        for j in 0..b.n {
            assert!(
                d.l[j] >= b.l[j],
                "interm-alpha LOOSENED the lower bound at neuron {j}: {} -> {}",
                b.l[j],
                d.l[j]
            );
            assert!(
                d.u[j] <= b.u[j],
                "interm-alpha LOOSENED the upper bound at neuron {j}: {} -> {}",
                b.u[j],
                d.u[j]
            );
            if d.l[j] > b.l[j] || d.u[j] < b.u[j] {
                moved += 1;
                let was_unstable = b.l[j] < 0.0 && b.u[j] > 0.0;
                if was_unstable && (d.l[j] >= 0.0 || d.u[j] <= 0.0) {
                    newly_stable += 1;
                }
            }
        }
    }
    eprintln!(
        "[test interm-alpha] neurons tightened by the per-start-node candidates: \
{moved} (newly stable: {newly_stable})"
    );
    assert!(
        moved > 0,
        "the per-start-node alpha candidates changed NOTHING — the lever is inert, \
so obligations (1) and (2) above are vacuous"
    );

    // (2) The dosed box must still contain the true pre-activation range.
    let mut rng2 = StdRng::seed_from_u64(9);
    let x = sample_box(&mut rng2, &dosed, 400);
    let mut pre_sel = std::collections::BTreeMap::new();
    for l in &dosed.layers {
        pre_sel.insert(l.op, (0..l.n).collect::<Vec<_>>());
    }
    let (_y, pre) = net.forward_points(&x, &pre_sel).expect("forward");
    for l in &dosed.layers {
        let z = &pre[&l.op];
        for j in 0..l.n {
            for b in 0..x.ncols() {
                let zv = z[[j, b]];
                assert!(
                    l.l[j] <= zv + 1e-9 && l.u[j] >= zv - 1e-9,
                    "interm-alpha published box [{}, {}] EXCLUDES sampled z {zv} \
(neuron {j}) — this is the false-unsat shape",
                    l.l[j],
                    l.u[j]
                );
            }
        }
    }
}

/// #interm-alpha: dose 0 must be byte-identical to the pre-#interm-alpha
/// phase. The default is 0, so this is the assertion that arming nothing
/// changes nothing — the property every shipped preset depends on.
#[test]
fn interm_alpha_dose_zero_is_bit_identical_to_the_shipped_phase() {
    let mut rng = StdRng::seed_from_u64(0xB1_0001);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let build = |dose: usize| {
        RootGates::build_retaining_bi(
            &net,
            &lo,
            &hi,
            RoundMode::Outward,
            None,
            Some(super::backward_interm::BiCfg {
                yield_window: 0,
                yield_ratio: 0.0,
                secs: 600.0,
                chunk: 8,
                topk: 512,
                chunk_secs: 0.0,
                interm_alpha: dose,
            }),
        )
        .expect("build")
        .0
    };
    let a = build(0);
    let b = build(0);
    for (x, y) in a.layers.iter().zip(&b.layers) {
        for j in 0..x.n {
            assert_eq!(x.l[j].to_bits(), y.l[j].to_bits(), "dose-0 l not stable");
            assert_eq!(x.u[j].to_bits(), y.u[j].to_bits(), "dose-0 u not stable");
            assert_eq!(
                x.alpha[j].to_bits(),
                y.alpha[j].to_bits(),
                "dose-0 alpha not stable"
            );
        }
    }
}

/// #bi-adaptive-chunk: sizing a pass from a MEASURED rate instead of a fixed
/// grain changes only the partition of the selected columns, and a partition is
/// not a mathematical object here — each column is its own independent prefix
/// pass committing its own shrink-only intersection. So with a budget large
/// enough that neither arm truncates, the two published roots must be
/// BIT-IDENTICAL. This is the regression gate for the whole mechanism: if the
/// sizer ever changed a bound it would be doing something it has no licence to
/// do, and the seconds it saves would be bought with a different proof.
#[test]
fn backward_interm_pass_sizer_is_bit_identical_to_the_fixed_grain() {
    let mut rng = StdRng::seed_from_u64(0xB1_0001);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.4; 72];
    let hi = vec![0.4; 72];
    let build = |chunk_secs: f64| {
        RootGates::build_retaining_bi(
            &net,
            &lo,
            &hi,
            RoundMode::Outward,
            None,
            Some(super::backward_interm::BiCfg {
                yield_window: 0,
                yield_ratio: 0.0,
                secs: 600.0,
                chunk: 8,
                topk: 512,
                chunk_secs,
                interm_alpha: 0,
            }),
        )
        .expect("build")
        .0
    };
    let fixed = build(0.0);
    // A target far larger than any pass on this toy net costs, so the sizer is
    // ceiling-bound and must reproduce the grain exactly; and a target far
    // smaller, so it is rate-bound and runs single-column passes.
    let ceiling_sized = build(600.0);
    let rate_sized = build(1e-9);
    assert_eq!(fixed.backward_interm_execution.configured_chunk_secs, 0.0);
    assert_eq!(
        ceiling_sized
            .backward_interm_execution
            .configured_chunk_secs,
        600.0
    );
    assert_eq!(
        rate_sized.backward_interm_execution.configured_chunk_secs,
        1e-9
    );
    assert!(rate_sized.backward_interm_execution.adaptive_width_changes > 0);
    for sized in [ceiling_sized, rate_sized] {
        assert_eq!(fixed.layers.len(), sized.layers.len());
        for (a, b) in fixed.layers.iter().zip(&sized.layers) {
            for j in 0..a.n {
                assert_eq!(a.l[j].to_bits(), b.l[j].to_bits(), "pass sizing moved l");
                assert_eq!(a.u[j].to_bits(), b.u[j].to_bits(), "pass sizing moved u");
                assert_eq!(a.s[j].to_bits(), b.s[j].to_bits(), "pass sizing moved s");
                assert_eq!(a.c[j].to_bits(), b.c[j].to_bits(), "pass sizing moved c");
            }
        }
    }
}

#[test]
fn prefix_pass_is_a_sound_enclosure_and_rejects_bad_entry() {
    let mut rng = StdRng::seed_from_u64(0xB1_0002);
    let spec = tiny_spec(&mut rng, 0.5);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.3; 72];
    let hi = vec![0.3; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    // Rejections: head relu (not in the trunk), non-relu op, seed shape drift.
    let n0 = root.layers[0].n;
    let mut s0 = Array2::<f64>::zeros((n0, n0));
    for j in 0..n0 {
        s0[[j, j]] = 1.0;
    }
    let seed0 = DomainStackSeed { s: s0, e: None };
    assert!(eng
        .run_prefix(&seed0, net.i_gemm1 + 1, LaneDir::Lower)
        .is_err());
    assert!(
        eng.run_prefix(&seed0, 0, LaneDir::Lower).is_err(),
        "conv op accepted"
    );
    let bad = DomainStackSeed {
        s: Array2::<f64>::zeros((n0 + 1, n0)),
        e: None,
    };
    assert!(eng
        .run_prefix(&bad, root.layers[0].op, LaneDir::Lower)
        .is_err());
    // Soundness: prefix bounds enclose sampled pre-activations at every layer.
    let mut rng2 = StdRng::seed_from_u64(11);
    let x = sample_box(&mut rng2, &root, 300);
    let mut pre_sel = std::collections::BTreeMap::new();
    for l in &root.layers {
        pre_sel.insert(l.op, (0..l.n).collect::<Vec<_>>());
    }
    let (_y, pre) = net.forward_points(&x, &pre_sel).expect("forward");
    for l in &root.layers {
        let n = l.n;
        let mut s = Array2::<f64>::zeros((n, n));
        for j in 0..n {
            s[[j, j]] = 1.0;
        }
        let seed = DomainStackSeed { s, e: None };
        let plo = eng.run_prefix(&seed, l.op, LaneDir::Lower).expect("lower");
        let lows = eng.concretize_lower(&plo);
        let pup = eng.run_prefix(&seed, l.op, LaneDir::Upper).expect("upper");
        let ups = eng.concretize_upper(&pup);
        let z = &pre[&l.op];
        for j in 0..n {
            for b in 0..x.ncols() {
                let zv = z[[j, b]];
                assert!(
                    lows[j] <= zv + 1e-9 && ups[j] >= zv - 1e-9,
                    "prefix enclosure [{}, {}] excludes sampled z {zv} (op {}, neuron {j})",
                    lows[j],
                    ups[j],
                    l.op
                );
            }
        }
    }
}

#[test]
fn clip_slack_calibration_must_use_forward_bounds() {
    // The #backward-interm ordering trap in isolation: slack recovered
    // against the line's own (forward) bound is exact; recovered against a
    // TIGHTER published bound it under-recovers and clamps to zero — which
    // would let a Clip-and-Verify halfspace cut into the true subdomain.
    let mi = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 0.5]).expect("shape");
    let di = Array2::from_shape_vec((1, 3), vec![0.1, 0.2, 0.05]).expect("shape");
    let mid = [0.1f64, -0.2];
    let rad = [0.5f64, 0.4];
    let (bm, bd) = (0.5f64, 0.05f64);
    let (mut lo_min, mut up_max) = (bm - bd, bm + bd);
    for j in 0..2 {
        let lo_c: f64 = mi[[0, j]] - di[[0, j]];
        let up_c: f64 = mi[[0, j]] + di[[0, j]];
        lo_min += lo_c.mul_add(mid[j], -(lo_c.abs() * rad[j]));
        up_max += up_c.mul_add(mid[j], up_c.abs() * rad[j]);
    }
    let l_fwd = [lo_min - 0.01];
    let u_fwd = [up_max + 0.02];
    let rows = super::root::retain_clip_rows(&mi, &di, &[0], &mid, &rad, &l_fwd, &u_fwd);
    assert!(
        (rows.sl_lo[0] - 0.01).abs() < 1e-12,
        "forward slack not recovered"
    );
    assert!(
        (rows.sl_up[0] - 0.02).abs() < 1e-12,
        "forward slack not recovered"
    );
    let l_tight = [lo_min + 0.3];
    let u_tight = [up_max - 0.3];
    let wrong = super::root::retain_clip_rows(&mi, &di, &[0], &mid, &rad, &l_tight, &u_tight);
    assert_eq!(
        wrong.sl_lo[0], 0.0,
        "tightened calibration must clamp (the trap)"
    );
    assert_eq!(
        wrong.sl_up[0], 0.0,
        "tightened calibration must clamp (the trap)"
    );
}

// ======================= #tableau-f32 STORAGE-LANE TESTS =====================
//
// THE PINNED CONTAINMENT RULE (docs/TABLEAU_F32_2026-08-20.md, must-fix MF2).
// Baseline: the f64 path at HEAD (`build_prec` with `use_f32 = false` for the
// end-to-end boxes; the literal HEAD site expressions for the per-site rule).
// Per-coefficient obligation at every f64->f32 store site, on the SAME inputs:
//
//     m32 - d32 <= m64 - d64   AND   m32 + d32 >= m64 + d64
//
// (equivalently `d32 - d64 >= |m32 - m64|`): the f32-stored sandwich interval
// CONTAINS the f64-path interval. Checked by the per-site ORACLE below over a
// grid including subnormals — not by a per-run assertion. The end-to-end zoo
// test then checks the composed boxes against the f64 baseline.

/// f32-representable CENTER-role grid values (signed, incl. f32 subnormals and
/// the 2^-25 teeth case), widened exactly to f64 so both lanes see identical
/// inputs.
fn tf32_grid_m() -> Vec<f64> {
    let mags: [f32; 10] = [
        0.0,
        f32::from_bits(1), // min subnormal (1.4e-45)
        1.0e-40,           // subnormal
        f32::MIN_POSITIVE, // 1.17549435e-38
        2.0f32.powi(-25),  // the uncharged-formula killer
        1.0e-20,
        0.5,
        1.0 + f32::EPSILON, // 1 + 2^-23
        3.303_03,
        1.0e8,
    ];
    let mut v: Vec<f64> = Vec::new();
    for m in mags {
        v.push(f64::from(m));
        if m != 0.0 {
            v.push(-f64::from(m));
        }
    }
    v
}

/// f32-representable RADIUS-role grid values (nonneg).
fn tf32_grid_d() -> Vec<f64> {
    [
        0.0f32,
        f32::from_bits(1),
        1.0e-40,
        f32::MIN_POSITIVE,
        1.0e-16,
        2.5e-7,
        0.25,
        7.0e5,
    ]
    .into_iter()
    .map(f64::from)
    .collect()
}

/// Per-site containment check: the f32-stored (m, d) pair must contain the
/// f64-path pair per coefficient. EXACT — no tolerance: the charges must
/// cover every store perturbation outright.
#[allow(clippy::too_many_arguments)]
fn tf32_assert_contains(
    site: &str,
    m64: &Array2<f64>,
    d64: &Array2<f64>,
    m32: &Array2<f32>,
    d32: &Array2<f32>,
    checks: &mut usize,
) {
    for ((idx, &mv64), &dv64) in m64.indexed_iter().zip(d64.iter()) {
        let mv32 = f64::from(m32[idx]);
        let dv32 = f64::from(d32[idx]);
        assert!(
            mv32 - dv32 <= mv64 - dv64 && mv32 + dv32 >= mv64 + dv64,
            "{site} @ {idx:?}: f32-stored sandwich [{:e}, {:e}] does NOT contain \
             f64 sandwich [{:e}, {:e}] (m64={mv64:e} d64={dv64:e} m32={mv32:e} \
             d32={dv32:e}) — a store charge under-covers (false-tightening risk)",
            mv32 - dv32,
            mv32 + dv32,
            mv64 - dv64,
            mv64 + dv64,
        );
        *checks += 1;
    }
}

/// MF2 per-site oracle, two obligations per store site on identical inputs
/// (f32-representable grid incl. subnormals), NO tolerance:
/// (1) per-coefficient sandwich containment against the f64 path at HEAD
///     (`d32 - d64 >= |m32 - m64|`, via the same-site D fold), and
/// (2) the site's fused g_err storage-dust charge dominates the exact
///     deviation functional: `gstore[j] >= sum_i (|m32-m64| + |d32-d64|) *
///     xabs_ext_i` — the sign-independent account `concretize_box` consumes,
///     which is what makes the storage lane endpoint-sound on MIXED-SIGN
///     input boxes (tinyimagenet's are mixed-sign).
/// Plus: the UNcharged Add formula demonstrably VIOLATES containment (the
/// test has teeth), and the din up-store is elementwise monotone.
#[test]
fn tableau_f32_store_per_site_containment_oracle() {
    use super::root::{add_op, chaffine_op, conv_bias_fold, par_zip2_radius_masked};
    let gm = tf32_grid_m();
    let gd = tf32_grid_d();
    let mut checks = 0usize;

    // Exact deviation functional of one row: sum_i (|dM| + |dD|) * x_i.
    let dev_functional = |m64: &Array2<f64>,
                          d64: &Array2<f64>,
                          m32: &Array2<f32>,
                          d32: &Array2<f32>,
                          row: usize,
                          xabs_ext: &[f64]|
     -> f64 {
        let mut acc = 0.0f64;
        for i in 0..m64.ncols() {
            acc += ((f64::from(m32[[row, i]]) - m64[[row, i]]).abs()
                + (f64::from(d32[[row, i]]) - d64[[row, i]]).abs())
                * xabs_ext[i];
        }
        acc
    };
    let assert_gstore = |site: &str,
                         m64: &Array2<f64>,
                         d64: &Array2<f64>,
                         m32: &Array2<f32>,
                         d32: &Array2<f32>,
                         gstore: &[f64],
                         xabs_ext: &[f64],
                         checks: &mut usize| {
        for row in 0..m64.nrows() {
            let need = dev_functional(m64, d64, m32, d32, row, xabs_ext);
            assert!(
                gstore[row] >= need,
                "{site} row {row}: g_err storage charge {:.6e} < exact deviation \
                 functional {:.6e} — endpoint soundness on mixed-sign boxes broken",
                gstore[row],
                need
            );
            *checks += 1;
        }
    };

    // ---- Site: residual Add (fused center+radius store + g_err charge) -----
    {
        let mut ma = Vec::new();
        let mut mb = Vec::new();
        let mut da = Vec::new();
        let mut db = Vec::new();
        for &a in &gm {
            for &b in &gm {
                for &x in &gd {
                    for &y in &gd {
                        ma.push(a);
                        mb.push(b);
                        da.push(x);
                        db.push(y);
                    }
                }
            }
        }
        let n = ma.len();
        // Mixed-magnitude xabs weights, cycled.
        let xabs_ext: Vec<f64> = (0..n).map(|i| [0.0, 1e-6, 1.0, 37.5][i % 4]).collect();
        let to2 = |v: &[f64]| Array2::from_shape_vec((1, n), v.to_vec()).expect("shape");
        let to2f = |v: &[f64]| {
            #[allow(clippy::cast_possible_truncation)]
            Array2::from_shape_vec((1, n), v.iter().map(|&x| x as f32).collect::<Vec<f32>>())
                .expect("shape")
        };
        let (mo64, do64, _) = add_op::<f64>(
            to2(&ma),
            &to2(&mb),
            to2(&da),
            &to2(&db),
            &[],
            1,
            RoundMode::Outward,
            &xabs_ext,
        );
        let (mo32, do32, gs32) = add_op::<f32>(
            to2f(&ma),
            &to2f(&mb),
            to2f(&da),
            &to2f(&db),
            &[],
            1,
            RoundMode::Outward,
            &xabs_ext,
        );
        tf32_assert_contains("add_op", &mo64, &do64, &mo32, &do32, &mut checks);
        assert_gstore(
            "add_op",
            &mo64,
            &do64,
            &mo32,
            &do32,
            &gs32,
            &xabs_ext,
            &mut checks,
        );

        // TEETH: the UNcharged formula (HEAD's f64 widening with a bare f32
        // nearest store, no `f32_store_slack`) must VIOLATE containment on
        // ma = 1, mb = 2^-25, da = db = 0: the sum 1 + 2^-25 stores to 1.0f32
        // (delta 2^-25 ~ 3e-8) while the uncharged D is only ~2e-16.
        let (sum, dsum) = (1.0f64 + f64::from(2.0f32.powi(-25)), 0.0f64);
        let m64v = sum;
        let d64v = super::rounding::next_up(
            (dsum + 2.0 * super::rounding::UNIT * sum.abs()) * (1.0 + 1e-15),
        );
        #[allow(clippy::cast_possible_truncation)]
        let m32v = f64::from(sum as f32);
        let d32_uncharged = f64::from(ny_core::f64_to_f32_up(d64v));
        assert!(
            m32v + d32_uncharged < m64v + d64v,
            "teeth case no longer violates containment WITHOUT the charge — the \
             oracle lost its discriminating power"
        );
    }

    // ---- Site: ChannelAffine (folded BN), coefficients + bias column -------
    {
        let scales = [-3.0f64, -1.0, -1e-3, 0.0, 1e-3, 1.0, 2.5, 1e10];
        let shifts = [-1.0f64, 0.0, 1e-8, 2.0];
        let mut scale = Vec::new();
        let mut shift = Vec::new();
        let mut shift_err = Vec::new();
        for (i, &sc) in scales.iter().enumerate() {
            for &sh in &shifts {
                scale.push(sc);
                shift.push(sh);
                shift_err.push(if i % 2 == 0 { 0.0 } else { 1e-9 });
            }
        }
        let rows = scale.len();
        let mut cols_m = Vec::new();
        let mut cols_d = Vec::new();
        for &m in &gm {
            for &d in &gd {
                cols_m.push(m);
                cols_d.push(d);
            }
        }
        let n_in = cols_m.len();
        let naug = n_in + 1;
        let xabs_ext: Vec<f64> = (0..n_in)
            .map(|i| [0.0, 1e-6, 1.0, 37.5][i % 4])
            .chain(std::iter::once(1.0))
            .collect();
        let mut m64 = Array2::<f64>::zeros((rows, naug));
        let mut d64 = Array2::<f64>::zeros((rows, naug));
        for r in 0..rows {
            for i in 0..n_in {
                m64[[r, i]] = cols_m[i];
                d64[[r, i]] = cols_d[i];
            }
            m64[[r, n_in]] = gm[r % gm.len()];
            d64[[r, n_in]] = gd[r % gd.len()];
        }
        #[allow(clippy::cast_possible_truncation)]
        let mut m32 = m64.mapv(|v| v as f32);
        #[allow(clippy::cast_possible_truncation)]
        let mut d32 = d64.mapv(|v| v as f32);
        let blk = naug.div_ceil(64);
        let _ = chaffine_op::<f64>(
            &mut m64,
            &mut d64,
            &scale,
            &shift,
            1e-15,
            &shift_err,
            n_in,
            RoundMode::Outward,
            None,
            blk,
            &xabs_ext,
        );
        let gs32 = chaffine_op::<f32>(
            &mut m32,
            &mut d32,
            &scale,
            &shift,
            1e-15,
            &shift_err,
            n_in,
            RoundMode::Outward,
            None,
            blk,
            &xabs_ext,
        );
        tf32_assert_contains("chaffine_op", &m64, &d64, &m32, &d32, &mut checks);
        assert_gstore(
            "chaffine_op",
            &m64,
            &d64,
            &m32,
            &d32,
            &gs32,
            &xabs_ext,
            &mut checks,
        );
    }

    // ---- Site: conv bias fold (bias column only, g_err weight 1.0) ---------
    {
        let biases = [-2.0f64, -1e-9, 0.0, 3.5e-8, 1.0, 4.0e7];
        let mut mo = Vec::new();
        let mut do_ = Vec::new();
        let mut bias = Vec::new();
        let mut bias_err = Vec::new();
        for &b in &biases {
            for &m in &gm {
                for &d in &gd {
                    mo.push(m);
                    do_.push(d);
                    bias.push(b);
                    bias_err.push(if bias.len() % 3 == 0 { 1e-12 } else { 0.0 });
                }
            }
        }
        let n_out = mo.len();
        // naug = 2: col 0 is untouched payload, col 1 is the bias column.
        let mk = |bias_col: &[f64]| {
            let mut a = Array2::<f64>::zeros((n_out, 2));
            for (j, &v) in bias_col.iter().enumerate() {
                a[[j, 1]] = v;
            }
            a
        };
        let mut mo64 = mk(&mo);
        let mut do64 = mk(&do_);
        #[allow(clippy::cast_possible_truncation)]
        let mut mo32 = mo64.mapv(|v| v as f32);
        #[allow(clippy::cast_possible_truncation)]
        let mut do32 = do64.mapv(|v| v as f32);
        let mut g64 = vec![0.0f64; n_out];
        let mut g32 = vec![0.0f64; n_out];
        conv_bias_fold::<f64>(
            &mut mo64,
            &mut do64,
            &bias,
            &bias_err,
            1,
            1,
            RoundMode::Outward,
            &mut g64,
        );
        conv_bias_fold::<f32>(
            &mut mo32,
            &mut do32,
            &bias,
            &bias_err,
            1,
            1,
            RoundMode::Outward,
            &mut g32,
        );
        assert!(g64.iter().all(|&v| v == 0.0), "f64 lane must not charge");
        tf32_assert_contains("conv_bias_fold", &mo64, &do64, &mo32, &do32, &mut checks);
        // Bias column dev functional with weight 1.0 (payload col 0 exact).
        for j in 0..n_out {
            let need = (f64::from(mo32[[j, 1]]) - mo64[[j, 1]]).abs()
                + (f64::from(do32[[j, 1]]) - do64[[j, 1]]).abs();
            assert!(
                g32[j] >= need,
                "conv_bias_fold row {j}: g_err charge {:.3e} < deviation {need:.3e}",
                g32[j]
            );
            checks += 1;
        }
    }

    // ---- Site: apply_gates (binary AND fractional alpha, bias column) ------
    {
        let gates: &[(f64, f64, f64)] = &[
            (0.0, 0.0, 0.0),        // inactive
            (1.0, 1.0, 0.0),        // active
            (1.0, 0.5, 0.7),        // binary unstable
            (0.0, 0.5, 0.7),        // binary unstable, alpha 0
            (0.3, 0.5, 0.7),        // fractional (#rebake regime)
            (0.999_999, 1e-6, 0.2), // fractional near-degenerate
        ];
        let alpha: Vec<f64> = gates.iter().map(|g| g.0).collect();
        let s: Vec<f64> = gates.iter().map(|g| g.1).collect();
        let c: Vec<f64> = gates.iter().map(|g| g.2).collect();
        let rows = gates.len();
        let mut cols_m = Vec::new();
        let mut cols_d = Vec::new();
        for &m in &gm {
            for &d in &gd {
                cols_m.push(m);
                cols_d.push(d);
            }
        }
        let n_in = cols_m.len();
        let naug = n_in + 1;
        let xabs_ext: Vec<f64> = (0..n_in)
            .map(|i| [0.0, 1e-6, 1.0, 37.5][i % 4])
            .chain(std::iter::once(1.0))
            .collect();
        let mut m64 = Array2::<f64>::zeros((rows, naug));
        let mut d64 = Array2::<f64>::zeros((rows, naug));
        for r in 0..rows {
            for i in 0..n_in {
                m64[[r, i]] = cols_m[i];
                d64[[r, i]] = cols_d[i];
            }
            m64[[r, n_in]] = gm[(r + 3) % gm.len()];
            d64[[r, n_in]] = gd[(r + 2) % gd.len()];
        }
        #[allow(clippy::cast_possible_truncation)]
        let m32 = m64.mapv(|v| v as f32);
        #[allow(clippy::cast_possible_truncation)]
        let d32 = d64.mapv(|v| v as f32);
        let blk = naug.div_ceil(64);
        let (o64m, o64d, _) = super::root::apply_gates::<f64>(
            &m64,
            &d64,
            &alpha,
            &s,
            &c,
            n_in,
            RoundMode::Outward,
            None,
            blk,
            &xabs_ext,
        );
        let (o32m, o32d, gs32) = super::root::apply_gates::<f32>(
            &m32,
            &d32,
            &alpha,
            &s,
            &c,
            n_in,
            RoundMode::Outward,
            None,
            blk,
            &xabs_ext,
        );
        tf32_assert_contains("apply_gates", &o64m, &o64d, &o32m, &o32d, &mut checks);
        assert_gstore(
            "apply_gates",
            &o64m,
            &o64d,
            &o32m,
            &o32d,
            &gs32,
            &xabs_ext,
            &mut checks,
        );
    }

    // ---- Site: din prep (radius-role up-store monotonicity, design 7.4) ----
    {
        let g = super::rounding::next_up(super::rounding::gamma_n(66) + 1e-15);
        let mut am = Vec::new();
        let mut ad = Vec::new();
        for &m in &gm {
            for &d in &gd {
                am.push(m);
                ad.push(d);
            }
        }
        let n = am.len();
        let a64 = Array2::from_shape_vec((1, n), am.clone()).expect("shape");
        let b64 = Array2::from_shape_vec((1, n), ad.clone()).expect("shape");
        #[allow(clippy::cast_possible_truncation)]
        let a32 = a64.mapv(|v| v as f32);
        #[allow(clippy::cast_possible_truncation)]
        let b32 = b64.mapv(|v| v as f32);
        let mut din64 = Array2::<f64>::zeros((1, n));
        let mut din32 = Array2::<f32>::zeros((1, n));
        par_zip2_radius_masked::<f64>(&mut din64, &a64, &b64, &[], 1, |m, d| d + g * (m.abs() + d));
        par_zip2_radius_masked::<f32>(&mut din32, &a32, &b32, &[], 1, |m, d| d + g * (m.abs() + d));
        for i in 0..n {
            assert!(
                f64::from(din32[[0, i]]) >= din64[[0, i]],
                "din up-store NOT monotone at {i}: {} < {}",
                f64::from(din32[[0, i]]),
                din64[[0, i]]
            );
            checks += 1;
        }
    }

    eprintln!("[tableau-f32] per-site containment oracle: {checks} exact checks, 0 breaches");
    assert!(checks > 20_000, "grid unexpectedly small ({checks})");
}

/// End-to-end #tableau-f32 gate over the net zoo:
/// (1) HARD: every f32-STORED box contains the f64-path box (the MF2 baseline:
///     `build_prec(use_f32 = false)` at HEAD), per layer/neuron;
/// (2) HARD off-arm bit-identity: with the storage flag OFF the generic walk's
///     published gates are BITWISE the f32-compute lane's (and with
///     `use_f32 = false` the refused store request is BITWISE the f64 lane's —
///     the refusal path is loud but inert).
#[test]
fn tableau_f32_store_boxes_contain_f64_and_off_arm_is_bitwise() {
    let n_out = 6usize;
    let mut seed: u64 = 0x7AB1_EAF3;
    let radii = [1.0f64, 1.0e-3];
    let mut box_checks = 0usize;
    for trial in 0..8usize {
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
            let net = match TwinNet::compile(&spec) {
                Ok(n) => n,
                Err(_) => continue,
            };
            let b = |use_f32: bool, store: bool| {
                RootGates::build_prec_store(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    None,
                    use_f32,
                    store,
                )
                .ok()
            };
            let (Some(r64), Some(r32), Some(rst)) =
                (b(false, false), b(true, false), b(true, true))
            else {
                continue;
            };
            // (2a) store OFF == the f32-compute lane, bitwise.
            let r32_prec = RootGates::build_prec(&net, &lo, &hi, RoundMode::Outward, None, true)
                .expect("f32 lane");
            for (a, c) in r32.layers.iter().zip(&r32_prec.layers) {
                for (x, y) in
                    a.l.iter()
                        .chain(&a.u)
                        .chain(&a.alpha)
                        .chain(&a.s)
                        .chain(&a.c)
                        .chain(&a.ms)
                        .zip(
                            c.l.iter()
                                .chain(&c.u)
                                .chain(&c.alpha)
                                .chain(&c.s)
                                .chain(&c.c)
                                .chain(&c.ms),
                        )
                {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "store-OFF arm drifted from the f32-compute lane (trial {trial})"
                    );
                }
            }
            // (2b) store requested WITHOUT the compute lane == the f64 lane,
            // bitwise (refused loudly, then inert).
            let refused = b(false, true).expect("refused build");
            for (a, c) in refused.layers.iter().zip(&r64.layers) {
                for (x, y) in a.l.iter().chain(&a.u).zip(c.l.iter().chain(&c.u)) {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "REFUSED store arm drifted from the f64 lane (trial {trial})"
                    );
                }
            }
            // (1) containment of the composed boxes vs the f64 path at HEAD.
            assert_eq!(rst.layers.len(), r64.layers.len());
            for (lgs, lg64) in rst.layers.iter().zip(&r64.layers) {
                for j in 0..lg64.n {
                    if !(lgs.l[j].is_finite()
                        && lgs.u[j].is_finite()
                        && lg64.l[j].is_finite()
                        && lg64.u[j].is_finite())
                    {
                        continue;
                    }
                    // Same directed-rounding allowance as the shipped f32 gate.
                    let eps = 8.0 * f64::EPSILON * (lg64.l[j].abs().max(lg64.u[j].abs()) + 1.0);
                    assert!(
                        lgs.l[j] <= lg64.l[j] + eps && lgs.u[j] >= lg64.u[j] - eps,
                        "trial {trial} rad {rad} j={j}: f32-STORED box [{}, {}] fails to \
                         contain f64 box [{}, {}]",
                        lgs.l[j],
                        lgs.u[j],
                        lg64.l[j],
                        lg64.u[j]
                    );
                    box_checks += 1;
                }
            }
        }
    }
    eprintln!("[tableau-f32] zoo containment: {box_checks} box checks, 0 breaches");
    assert!(box_checks > 500, "zoo unexpectedly small ({box_checks})");
}

/// Class (a) refactor gate: the native f32-store conv publishes BITWISE the
/// value the shipped widened path publishes (same shared core, dense and
/// mask-blocked), protecting the shipped `root_f32` lane through the
/// cast/core split.
#[test]
fn conv_f32_native_store_is_bitwise_the_widened_path() {
    use super::net::{conv_apply_forward_prec_masked, conv_forward_f32_native, TwinOp};
    let mut rng = StdRng::seed_from_u64(0xC0DE_F32D);
    let spec = conv_bn_spec(3, 3, 3, 12, 4.0, 1e-13, 1e-4, 16, 6, 0xFEED);
    let net = TwinNet::compile(&spec).expect("compile");
    let naug = 40usize;
    let mut tested = 0usize;
    for (k, op) in net.ops.iter().enumerate() {
        let TwinOp::Conv(c) = op else { continue };
        let n_src = net.tsize[c.input];
        let n_out = net.tsize[k + 1];
        let src64 = Array2::from_shape_fn((n_src, naug), |_| rng.random_range(-2.0..2.0f64));
        #[allow(clippy::cast_possible_truncation)]
        let src32 = src64.mapv(|v| v as f32);
        for abs_w in [false, true] {
            // Dense.
            let mut dst64 = Array2::<f64>::zeros((n_out, naug));
            conv_apply_forward_prec_masked(c, &src64, &mut dst64, abs_w, true, None);
            let mut dst32 = Array2::<f32>::zeros((n_out, naug));
            conv_forward_f32_native(c, &src32, &mut dst32, abs_w, None);
            for (a, b) in dst64.iter().zip(dst32.iter()) {
                assert_eq!(
                    a.to_bits(),
                    f64::from(*b).to_bits(),
                    "native f32 store drifted from the widened path (op {k}, dense)"
                );
            }
            // Mask-blocked (all-live masks: exercises the mask blocking with
            // the support invariant trivially satisfied).
            let src_mask = vec![u64::MAX; n_src];
            let out_mask = vec![u64::MAX; n_out];
            let mask = Some((src_mask.as_slice(), out_mask.as_slice(), 64usize));
            let mut dst64m = Array2::<f64>::zeros((n_out, naug));
            conv_apply_forward_prec_masked(c, &src64, &mut dst64m, abs_w, true, mask);
            let mut dst32m = Array2::<f32>::zeros((n_out, naug));
            conv_forward_f32_native(c, &src32, &mut dst32m, abs_w, mask);
            for (a, b) in dst64m.iter().zip(dst32m.iter()) {
                assert_eq!(
                    a.to_bits(),
                    f64::from(*b).to_bits(),
                    "native f32 store drifted from the widened path (op {k}, masked)"
                );
            }
            tested += 1;
        }
    }
    assert!(tested >= 4, "too few conv ops exercised ({tested})");
}

/// #root-rounding follow-up (f32 compute lane): `conv_f32_gerr` must bound the
/// f32 error of BOTH conv lanes against exact arithmetic, and the tableau's D
/// lane is SIGNED (every gate with `U < L` makes it negative). The historical
/// functional `B[t] = sum_i (|m| + d) xabs_ext_i` charged a source row whose
/// coefficients have `d = -|m|` NOTHING beyond the FTZ floor, while both f32
/// products of that row are off by `~u32 |W| |m|`. This pin drives the real
/// f32 conv on exactly such rows (centers not f32-representable, `d = -|m|`
/// exactly) and requires the charge to dominate a rigorous UPPER bound on the
/// realized deviation functional `sum_i (|dM| + |dD|) xabs_ext_i`, and the
/// historical charge to fall below a rigorous LOWER bound on it (teeth).
#[test]
fn conv_f32_gerr_charges_a_signed_d_lane_by_magnitude() {
    use super::net::{conv_apply_forward_prec_masked, TwinOp};
    use super::root::conv_f32_gerr;
    use super::rounding::{gamma_n, next_up, SUBNORMAL_F32};
    let mut rng = StdRng::seed_from_u64(0x5167_D1A6);
    let spec = conv_bn_spec(3, 3, 3, 12, 4.0, 1e-13, 1e-4, 16, 6, 0xFEED);
    let net = TwinNet::compile(&spec).expect("compile");
    let c = net
        .ops
        .iter()
        .find_map(|op| match op {
            TwinOp::Conv(c) => Some(c),
            _ => None,
        })
        .expect("conv op");
    let n_src = net.tsize[c.input];
    let n_out = c.oshape.0 * c.oshape.1 * c.oshape.2;
    let naug = 9usize;
    let n_in = naug - 1;
    let mi = Array2::from_shape_fn((n_src, naug), |_| {
        rng.random_range(-2.0..2.0f64) * (1.0 + 1e-9)
    });
    // `|m| + d == 0` EXACTLY on every coefficient, bias column included.
    let di = mi.mapv(|v| -v.abs());
    let xabs: Vec<f64> = (0..n_in).map(|i| [0.5, 1.0, 2.0, 37.5][i % 4]).collect();
    let xabs_ext: Vec<f64> = xabs.iter().copied().chain(std::iter::once(1.0)).collect();
    let s_xabs = next_up(xabs.iter().sum::<f64>() + 1.0);
    let zeros = vec![0.0f64; n_src];
    let charge = conv_f32_gerr::<f64>(
        c,
        &mi,
        &di,
        &zeros,
        &xabs,
        s_xabs,
        n_in,
        None,
        naug.div_ceil(64),
        false,
    );
    assert_eq!(charge.len(), n_out, "one charge per output row");
    let run = |src: &Array2<f64>, abs_w: bool, f32_lane: bool| {
        let mut dst = Array2::<f64>::zeros((n_out, naug));
        conv_apply_forward_prec_masked(c, src, &mut dst, abs_w, f32_lane, None);
        dst
    };
    let (m32, d32) = (run(&mi, false, true), run(&di, true, true));
    let (m64, d64) = (run(&mi, false, false), run(&di, true, false));
    // `|W| |m|` (== `|W| |d|` here) in f64: each f64 reference is within
    // `gamma_k |W| |.|` of exact, and `2 gamma_{k+2}` also covers the envelope's
    // own rounding.
    let env = run(&mi.mapv(f64::abs), true, false);
    let g64 = 2.0 * gamma_n(c.k_fwd + 2);
    // The historical charge on these rows: `B[t] = 0` exactly, so what reaches
    // the output is the FTZ floor plus subnormal dust; `1e-290` dominates the
    // dust with hundreds of orders to spare.
    let ftz = next_up(2.0 * (c.k_fwd as f64 + 2.0) * SUBNORMAL_F32 * s_xabs);
    let retired_ub = next_up(ftz + 1e-290);
    let mut teeth = 0usize;
    for o in 0..n_out {
        let (mut lb, mut ub) = (0.0f64, 0.0f64);
        for i in 0..naug {
            let slack = g64 * env[[o, i]];
            let dm = (m32[[o, i]] - m64[[o, i]]).abs();
            let dd = (d32[[o, i]] - d64[[o, i]]).abs();
            lb += ((dm - slack).max(0.0) + (dd - slack).max(0.0)) * xabs_ext[i];
            ub += (dm + dd + 2.0 * slack) * xabs_ext[i];
        }
        // The f64 sum of non-negative terms under-reports by far less than 1e-12.
        let ub = ub * (1.0 + 1e-12);
        assert!(
            charge[o] >= ub,
            "row {o}: conv_f32_gerr charge {:.6e} < realized f32 deviation bound {ub:.6e} on a \
             signed D lane (false-unsat risk on the f32 root lane)",
            charge[o]
        );
        if lb > retired_ub {
            teeth += 1;
        }
    }
    eprintln!(
        "[root-rounding] conv_f32_gerr signed-D pin: {teeth}/{n_out} rows beat the retired charge"
    );
    assert!(
        teeth > 0,
        "VOID: no row's realized f32 deviation exceeds the historical signed-D charge, so this \
         fixture no longer tells |d| from d"
    );
}

/// #root-rounding follow-up, END TO END on the shipped f32 root lanes
/// (`root_f32`, and `tableau_f32` on top of it): the trunk-1 box must contain
/// the EXACT stored-network hull when the conv consumes a SIGNED D lane.
///
/// Fixture: `x in [-3, -2]`; 256 identical trunk-0 neurons `z1 = w1 x + b1`
/// (`w1 = -1.0000001`, `b1 = -2.6`), each unstable with `u < -l`, so
/// `alpha = 0` and the upper line `U = s (m + d) + c` is NEGATIVE on every
/// column (bias included, since 0 is outside the box). The gate output is
/// then `M' = D' = U/2 < 0` on every coefficient: `|M'| + D' == 0` exactly
/// (plus the bias-column rounding charge), which the historical signed
/// functional `B[t] = sum_i (|m| + d) xabs_ext_i` charged NOTHING. The
/// trunk-1 conv sums the 256 identical rows in f32, whose coherent rounding
/// (plus `w2`'s f32 conversion) moves the walls by ~1e-7 relative with no
/// incoming `g_err` large enough to cover it. The exact hull is
/// `[0, 256 w2 (w1 lo + b1)]` (`w1 < 0`, so the relu is non-increasing in `x`
/// and `z1(hi) < 0`).
///
/// Replica (`vbs/diag10/fix_root_rounding/review/`, exact rationals over an
/// emulated `conv_f32_core`): the pre-repair f32 compute lane (the legacy
/// switch, which also restores the signed functional) misses the exact max on
/// all five `w2` below, by -3.1e-7 .. -8.6e-7 relative (-2.2e-5 .. -2.2e-4
/// absolute); the repaired `root_f32` and `tableau_f32` lanes contain it with
/// >= +7.6e-5 relative headroom, the f64 lane with +1.6e-13.
#[test]
fn f32_root_lanes_enclose_the_exact_hull_over_a_signed_d_lane() {
    use num_rational::BigRational as Q;
    use num_traits::{ToPrimitive, Zero};
    let q = |v: f64| Q::from_float(v).expect("finite");
    let (n1, w1, b1) = (256usize, -1.000_000_1f64, -2.6f64);
    let (lo, hi) = ([-3.0f64], [-2.0f64]);
    let z1 = |x: f64| q(w1) * q(x) + q(b1);
    assert!(
        z1(lo[0]) > Q::zero() && z1(hi[0]) < Q::zero(),
        "premise: the trunk-0 pre-activation must straddle 0 on the box"
    );
    let w2s = [
        0.7f64,
        0.378_410_126_452_143_45,
        0.553_420_884_290_434_2,
        2.621_855_627_747_282_6,
        2.733_854_135_551_005_5,
    ];
    let mut failures: Vec<String> = Vec::new();
    let mut legacy_misses = 0usize;
    let mut checked = 0usize;
    for &w2 in &w2s {
        let mut ops = vec![
            TwinOpSpec::Conv {
                input: 0,
                weight: vec![w1; n1],
                bias: vec![b1; n1],
                bias_err: vec![0.0; n1],
                weight_rel_err: 0.0,
                kernel: (n1, 1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (1, 1, 1),
                oshape: (n1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::Conv {
                input: 2,
                weight: vec![w2; n1],
                bias: vec![0.0],
                bias_err: vec![0.0],
                weight_rel_err: 0.0,
                kernel: (1, n1, 1, 1),
                stride: (1, 1),
                pads: (0, 0, 0, 0),
                ishape: (n1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 3 },
        ];
        rounding_leak_head(&mut ops, 4, 1, 0x5167_E2E0);
        let net = TwinNet::compile(&TwinSpec { n_in: 1, ops }).expect("compile");
        let mx = q(n1 as f64) * q(w2) * z1(lo[0]);
        let mn = Q::zero();
        for (lane, use_f32, store) in [
            ("f64", false, false),
            ("root_f32", true, false),
            ("tableau_f32", true, true),
        ] {
            let build = || {
                RootGates::build_prec_store(
                    &net,
                    &lo,
                    &hi,
                    RoundMode::Outward,
                    None,
                    use_f32,
                    store,
                )
            };
            let fixed = build().expect("root");
            assert_eq!(fixed.layers.len(), 2, "{lane} w2={w2}: two trunk relus");
            let l0 = &fixed.layers[0];
            assert!(
                (0..l0.n).all(|j| l0.l[j] < 0.0 && l0.u[j] > 0.0 && l0.alpha[j] == 0.0),
                "{lane} w2={w2}: premise lost — every trunk-0 relu must be unstable with \
                 alpha = 0 (u < -l), or the gate rows are no longer |M| + D == 0"
            );
            let (l, u) = (fixed.layers[1].l[0], fixed.layers[1].u[0]);
            checked += 1;
            if !(q(l) <= mn && q(u) >= mx) {
                failures.push(format!(
                    "{lane} w2={w2}: box [{l:e}, {u:e}] vs exact [0, {:e}] (upper headroom {:+e})",
                    mx.to_f64().unwrap_or(f64::NAN),
                    (q(u) - &mx).to_f64().unwrap_or(f64::NAN),
                ));
            }
            if lane == "root_f32" {
                // MUTATION PIN: the pre-repair f32 compute lane (the legacy
                // switch restores the signed `B` too) must miss somewhere.
                let legacy =
                    super::root::with_legacy_rounding_charge_for_test(build).expect("legacy root");
                let (ll, lu) = (legacy.layers[1].l[0], legacy.layers[1].u[0]);
                if !(q(ll) <= mn && q(lu) >= mx) {
                    legacy_misses += 1;
                }
                eprintln!(
                    "[root-rounding] f32 signed-D e2e w2={w2}: exact max {:e}; root_f32 up {u:e} \
                     (headroom {:+.3e}); legacy up {lu:e} (headroom {:+.3e})",
                    mx.to_f64().unwrap_or(f64::NAN),
                    (q(u) - &mx).to_f64().unwrap_or(f64::NAN),
                    (q(lu) - &mx).to_f64().unwrap_or(f64::NAN),
                );
            }
        }
    }
    assert_eq!(checked, 3 * w2s.len(), "sweep did not run every lane");
    assert!(
        legacy_misses > 0,
        "VOID: the pre-repair f32 compute lane contains the exact hull for every w2, so this \
         fixture no longer detects the signed-D g_err under-count"
    );
    assert!(
        failures.is_empty(),
        "F32 ROOT LANE BOX MISSES THE EXACT STORED-NETWORK HULL (false-unsat generator): \
         {failures:#?}"
    );
}

/// #tableau-f32 interface gates for the ranker/clip consumers:
/// (1) `retain_clip_rows` over f32-stored rows == over their EXACT f64
///     widenings, bitwise (retention happens once; rows STAY f64);
/// (2) Tier-0 retained rows built from the f32-stored tableau still sandwich
///     sampled pre-activations (the epoch property, store arm).
#[test]
fn tableau_f32_store_retention_interfaces_hold() {
    // (1) clip rows: f32 input rows == widened f64 input rows, bitwise.
    let m32 = Array2::from_shape_vec((2, 3), vec![1.5f32, -2.25, 0.5, 3.0e-7, -1.0e-40, 2.0])
        .expect("shape");
    let d32 = Array2::from_shape_vec((2, 3), vec![0.25f32, 0.125, 0.0625, 1.0e-8, 4.0e-41, 0.5])
        .expect("shape");
    let m64 = m32.mapv(f64::from);
    let d64 = d32.mapv(f64::from);
    let mid = [0.1f64, -0.2];
    let rad = [0.5f64, 0.4];
    let (l_cal, u_cal) = ([-5.0f64, -9.0], [5.0f64, 9.0]);
    let a = super::root::retain_clip_rows(&m32, &d32, &[0, 1], &mid, &rad, &l_cal, &u_cal);
    let b = super::root::retain_clip_rows(&m64, &d64, &[0, 1], &mid, &rad, &l_cal, &u_cal);
    assert_eq!(a.m, b.m);
    assert_eq!(a.d, b.d);
    assert_eq!(a.bm, b.bm);
    assert_eq!(a.bd, b.bd);
    assert_eq!(a.sl_lo, b.sl_lo);
    assert_eq!(a.sl_up, b.sl_up);

    // (2) retained rows sandwich sampled pre-activations on the STORE arm.
    let mut rng = StdRng::seed_from_u64(51);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.3; 72];
    let hi = vec![0.5; 72];
    let cfg = super::root::RetainCfg {
        per_layer: 1_000_000,
        budget_bytes: 1 << 30,
    };
    let (root, ret) = RootGates::build_retaining_prec_store(
        &net,
        &lo,
        &hi,
        RoundMode::Outward,
        None,
        Some(&cfg),
        true,
        true,
    )
    .expect("store-arm root");
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
                // f32 cast slack + the storage lane's own f32 grid: 2x the
                // f64-lane allowance.
                let tol = 1e-9 + mag * 2.4e-7;
                let z = p[[ri, b]];
                assert!(
                    vl - tol <= z && z <= vu + tol,
                    "store arm op {} neuron {}: z={z} outside f32 sandwich [{vl}, {vu}] \
                     (tol {tol:.3e})",
                    lg.op,
                    lr.idx[ri]
                );
            }
        }
    }
}

/// Synthetic-net floor-time smoke (#tableau-f32): build the same walk with
/// storage OFF vs ON (both on the f32 compute lane) and report the RELATIVE
/// build-time cut. Nothing wall-clock is asserted — the printed ratio is the
/// observable; correctness (containment vs the f64 path) IS asserted.
#[test]
fn tableau_f32_store_floor_smoke() {
    let spec = conv_bn_spec(3, 12, 12, 24, 2.0, 1e-13, 1e-4, 16, 6, 0x50FA_F32D);
    let net = TwinNet::compile(&spec).expect("compile");
    let n_in = 3 * 12 * 12;
    let lo = vec![-0.05; n_in];
    let hi = vec![0.05; n_in];
    let time_build = |store: bool| {
        // Min of 3 builds: the walk is deterministic; min removes scheduler noise.
        let mut best = f64::INFINITY;
        let mut root = None;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let r =
                RootGates::build_prec_store(&net, &lo, &hi, RoundMode::Outward, None, true, store)
                    .expect("build");
            best = best.min(t0.elapsed().as_secs_f64());
            root = Some(r);
        }
        (best, root.expect("built"))
    };
    let (t_off, r_off) = time_build(false);
    let (t_on, r_on) = time_build(true);
    let r64 = RootGates::build_prec(&net, &lo, &hi, RoundMode::Outward, None, false).expect("f64");
    for (lgs, lg64) in r_on.layers.iter().zip(&r64.layers) {
        for j in 0..lg64.n {
            let eps = 8.0 * f64::EPSILON * (lg64.l[j].abs().max(lg64.u[j].abs()) + 1.0);
            assert!(
                lgs.l[j] <= lg64.l[j] + eps && lgs.u[j] >= lg64.u[j] - eps,
                "smoke: store box fails containment at {j}"
            );
        }
    }
    assert_eq!(r_on.layers.len(), r_off.layers.len());
    eprintln!(
        "[tableau-f32-smoke] off={t_off:.3}s on={t_on:.3}s ratio={:.2} (relative only; \
         the real observable is the tiny A/B's [backward-interm] done secs= line)",
        t_on / t_off.max(1e-12)
    );
}

/// Release-grade floor projection (#tableau-f32): a wider synthetic conv net
/// whose tableau lanes leave cache, so the storage-width cut is visible.
/// This is compiled only for release test builds; run it explicitly with:
/// `targo --unverified test --release -p ny-propagate tableau_f32_store_floor_smoke_big -- --nocapture`
#[cfg(not(debug_assertions))]
#[test]
fn tableau_f32_store_floor_smoke_big() {
    let spec = conv_bn_spec(3, 24, 24, 32, 2.0, 1e-13, 1e-4, 16, 6, 0x5160_F32D);
    let net = TwinNet::compile(&spec).expect("compile");
    let n_in = 3 * 24 * 24;
    let lo = vec![-0.05; n_in];
    let hi = vec![0.05; n_in];
    let time_build = |store: bool| {
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            let _ =
                RootGates::build_prec_store(&net, &lo, &hi, RoundMode::Outward, None, true, store)
                    .expect("build");
            best = best.min(t0.elapsed().as_secs_f64());
        }
        best
    };
    let t_off = time_build(false);
    let t_on = time_build(true);
    eprintln!(
        "[tableau-f32-smoke-big] n_in={n_in} off={t_off:.3}s on={t_on:.3}s ratio={:.2}",
        t_on / t_off.max(1e-12)
    );
}

/// #tableau-f32 charge-magnitude REGRESSION PIN (2026-08-20 ship-gate FAIL,
/// re-entry (a)): the live sites charge the EXACT per-site deviation
/// `(|dM| + |dD|) * xabs_ext` — not the retired worst-case formula
/// (`rounding::f32_store_gerr_term`, kept only as this pin's yardstick) —
/// and the conv `store_dust` arm budgets the din/do_ up-store dust against
/// the D-LANE-ONLY functional `Bd`, not the full `|M|+D` envelope `B`.
/// This pin measures both tightenings and FAILS if a charge regresses toward
/// the retired magnitudes; every measured factor is printed for the ledger.
/// (The grids deliberately mix `d >> m` rows, which UNDERSTATE the production
/// win — tableau coefficients run `|M| >> D` through the early layers.)
#[test]
fn tableau_f32_store_dust_charge_regression_pin() {
    use super::net::TwinOp;
    use super::root::{add_op, apply_gates, chaffine_op, conv_f32_gerr};
    use super::rounding::{f32_store_gerr_term, gamma_n_f32, next_up, UNIT_F32};

    // Retired worst-case row charge, reconstructed from the stored outputs
    // (bitwise the retired site code for add_op/apply_gates; a LOWER bound
    // for chaffine_op, whose retired code charged the bias column twice —
    // an understated factor is a conservative pin).
    let old_row = |m32: &Array2<f32>, d32: &Array2<f32>, row: usize, x: &[f64]| -> f64 {
        let mut acc = 0.0f64;
        for i in 0..m32.ncols() {
            acc += f32_store_gerr_term(f64::from(m32[[row, i]]), f64::from(d32[[row, i]])) * x[i];
        }
        next_up(acc)
    };

    // f32-representable grids (identical inputs on both lanes).
    let gm: Vec<f64> = [
        0.0f32, 0.5, -0.5, 3.303_03, -3.303_03, 1.0e-6, -1.0e-6, 1.0e8, -1.0e8,
    ]
    .into_iter()
    .map(f64::from)
    .collect();
    let gd: Vec<f64> = [0.0f32, 2.5e-7, 0.25, 7.0e5]
        .into_iter()
        .map(f64::from)
        .collect();

    // ---- Site: residual Add ------------------------------------------------
    {
        let mut ma = Vec::new();
        let mut mb = Vec::new();
        let mut da = Vec::new();
        let mut db = Vec::new();
        for &a in &gm {
            for &b in &gm {
                for &x in &gd {
                    for &y in &gd {
                        ma.push(a);
                        mb.push(b);
                        da.push(x);
                        db.push(y);
                    }
                }
            }
        }
        let n = ma.len();
        let xabs_ext: Vec<f64> = (0..n).map(|i| [0.5, 1e-6, 1.0, 37.5][i % 4]).collect();
        let to2f = |v: &[f64]| {
            #[allow(clippy::cast_possible_truncation)]
            Array2::from_shape_vec((1, n), v.iter().map(|&x| x as f32).collect::<Vec<f32>>())
                .expect("shape")
        };
        let (mo32, do32, gs32) = add_op::<f32>(
            to2f(&ma),
            &to2f(&mb),
            to2f(&da),
            &to2f(&db),
            &[],
            1,
            RoundMode::Outward,
            &xabs_ext,
        );
        let new_total: f64 = gs32.iter().sum();
        let old_total: f64 = (0..mo32.nrows())
            .map(|r| old_row(&mo32, &do32, r, &xabs_ext))
            .sum();
        let factor = old_total / new_total;
        eprintln!("[tableau-f32 pin] add_op charge factor old/new = {factor:.3}");
        assert!(
            factor >= 1.3,
            "add_op exact dust charge regressed toward the retired worst-case \
             formula (old/new {factor:.3} < 1.3)"
        );
    }

    // ---- Site: ChannelAffine -----------------------------------------------
    {
        let scales = [-3.0f64, -1.0, -1e-3, 0.0, 1e-3, 1.0, 2.5, 1e10];
        let shifts = [-1.0f64, 0.0, 1e-8, 2.0];
        let mut scale = Vec::new();
        let mut shift = Vec::new();
        let mut shift_err = Vec::new();
        for (i, &sc) in scales.iter().enumerate() {
            for &sh in &shifts {
                scale.push(sc);
                shift.push(sh);
                shift_err.push(if i % 2 == 0 { 0.0 } else { 1e-9 });
            }
        }
        let rows = scale.len();
        let mut cols_m = Vec::new();
        let mut cols_d = Vec::new();
        for &m in &gm {
            for &d in &gd {
                cols_m.push(m);
                cols_d.push(d);
            }
        }
        let n_in = cols_m.len();
        let naug = n_in + 1;
        let xabs_ext: Vec<f64> = (0..n_in)
            .map(|i| [0.5, 1e-6, 1.0, 37.5][i % 4])
            .chain(std::iter::once(1.0))
            .collect();
        let mut m64 = Array2::<f64>::zeros((rows, naug));
        let mut d64 = Array2::<f64>::zeros((rows, naug));
        for r in 0..rows {
            for i in 0..n_in {
                m64[[r, i]] = cols_m[i];
                d64[[r, i]] = cols_d[i];
            }
            m64[[r, n_in]] = gm[r % gm.len()];
            d64[[r, n_in]] = gd[r % gd.len()];
        }
        #[allow(clippy::cast_possible_truncation)]
        let mut m32 = m64.mapv(|v| v as f32);
        #[allow(clippy::cast_possible_truncation)]
        let mut d32 = d64.mapv(|v| v as f32);
        let blk = naug.div_ceil(64);
        let gs32 = chaffine_op::<f32>(
            &mut m32,
            &mut d32,
            &scale,
            &shift,
            1e-15,
            &shift_err,
            n_in,
            RoundMode::Outward,
            None,
            blk,
            &xabs_ext,
        );
        let new_total: f64 = gs32.iter().sum();
        let old_total: f64 = (0..rows).map(|r| old_row(&m32, &d32, r, &xabs_ext)).sum();
        let factor = old_total / new_total;
        eprintln!("[tableau-f32 pin] chaffine_op charge factor old/new = {factor:.3}");
        assert!(
            factor >= 1.3,
            "chaffine_op exact dust charge regressed toward the retired \
             worst-case formula (old/new {factor:.3} < 1.3)"
        );
    }

    // ---- Site: apply_gates (binary + fractional alpha) ---------------------
    {
        let gates: &[(f64, f64, f64)] = &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (1.0, 0.5, 0.7),
            (0.0, 0.5, 0.7),
            (0.3, 0.5, 0.7),
            (0.999_999, 1e-6, 0.2),
        ];
        let alpha: Vec<f64> = gates.iter().map(|g| g.0).collect();
        let s: Vec<f64> = gates.iter().map(|g| g.1).collect();
        let c: Vec<f64> = gates.iter().map(|g| g.2).collect();
        let rows = gates.len();
        let mut cols_m = Vec::new();
        let mut cols_d = Vec::new();
        for &m in &gm {
            for &d in &gd {
                cols_m.push(m);
                cols_d.push(d);
            }
        }
        let n_in = cols_m.len();
        let naug = n_in + 1;
        let xabs_ext: Vec<f64> = (0..n_in)
            .map(|i| [0.5, 1e-6, 1.0, 37.5][i % 4])
            .chain(std::iter::once(1.0))
            .collect();
        let mut m64 = Array2::<f64>::zeros((rows, naug));
        let mut d64 = Array2::<f64>::zeros((rows, naug));
        for r in 0..rows {
            for i in 0..n_in {
                m64[[r, i]] = cols_m[i];
                d64[[r, i]] = cols_d[i];
            }
            m64[[r, n_in]] = gm[(r + 3) % gm.len()];
            d64[[r, n_in]] = gd[(r + 2) % gd.len()];
        }
        #[allow(clippy::cast_possible_truncation)]
        let m32 = m64.mapv(|v| v as f32);
        #[allow(clippy::cast_possible_truncation)]
        let d32 = d64.mapv(|v| v as f32);
        let blk = naug.div_ceil(64);
        let (o32m, o32d, gs32) = apply_gates::<f32>(
            &m32,
            &d32,
            &alpha,
            &s,
            &c,
            n_in,
            RoundMode::Outward,
            None,
            blk,
            &xabs_ext,
        );
        let new_total: f64 = gs32.iter().sum();
        let old_total: f64 = (0..rows).map(|r| old_row(&o32m, &o32d, r, &xabs_ext)).sum();
        let factor = old_total / new_total;
        eprintln!("[tableau-f32 pin] apply_gates charge factor old/new = {factor:.3}");
        assert!(
            factor >= 1.3,
            "apply_gates exact dust charge regressed toward the retired \
             worst-case formula (old/new {factor:.3} < 1.3)"
        );
    }

    // ---- Conv store-dust arm (production regime: D orders below |M|) -------
    {
        let mut rng = StdRng::seed_from_u64(0xD057);
        let spec = conv_bn_spec(3, 3, 3, 12, 4.0, 1e-13, 1e-4, 16, 6, 0xFEED);
        let net = TwinNet::compile(&spec).expect("compile");
        let c = net
            .ops
            .iter()
            .find_map(|op| match op {
                TwinOp::Conv(c) => Some(c),
                _ => None,
            })
            .expect("conv op");
        let n_src = net.tsize[c.input];
        let naug = 40usize;
        let n_in = naug - 1;
        let mi64 = Array2::from_shape_fn((n_src, naug), |_| rng.random_range(-2.0..2.0f64));
        #[allow(clippy::cast_possible_truncation)]
        let mi32 = mi64.mapv(|v| v as f32);
        // Early-layer regime: the D lane runs ~gamma-scale below the centers.
        #[allow(clippy::cast_possible_truncation)]
        let din32 = mi32.mapv(|v| (f64::from(v).abs() * 1e-13) as f32);
        let xabs: Vec<f64> = (0..n_in).map(|i| [0.5, 1.0, 2.0, 37.5][i % 4]).collect();
        let s_xabs = next_up(xabs.iter().sum::<f64>() + 1.0);
        let zeros = vec![0.0f64; n_src];
        let blk = naug.div_ceil(64);
        let with_dust = conv_f32_gerr::<f32>(
            c, &mi32, &din32, &zeros, &xabs, s_xabs, n_in, None, blk, true,
        );
        let compute_only = conv_f32_gerr::<f32>(
            c, &mi32, &din32, &zeros, &xabs, s_xabs, n_in, None, blk, false,
        );
        let new_dust: f64 = with_dust
            .iter()
            .zip(&compute_only)
            .map(|(a, b)| a - b)
            .sum();
        assert!(new_dust > 0.0, "store-dust arm charged nothing");
        // The retired arm charged (gf32 + 8*UNIT_F32) * B — an increment of
        // (8*UNIT_F32 / gf32) of the compute charge itself.
        let gf32 = next_up(gamma_n_f32(c.k_fwd + 8));
        let old_dust: f64 = compute_only
            .iter()
            .map(|v| v * (8.0 * UNIT_F32 / gf32))
            .sum();
        let factor = old_dust / new_dust;
        eprintln!("[tableau-f32 pin] conv store-dust charge factor old/new = {factor:.3e}");
        assert!(
            factor >= 1.0e6,
            "conv store-dust charge no longer tracks the D-lane-only functional \
             (old/new {factor:.3e} < 1e6 in the |M| >> D regime)"
        );
    }
}

/// End-to-end SOUNDNESS backstop under the STORE ARM (#tableau-f32): the
/// composed `dj` from an f32-STORED build must lower-bound the true margin
/// (double-double reference) at gradient-guided corners + interior samples —
/// the same obligation the shipped enclosure-oracle suite pins for the f64 and
/// f32-compute lanes. Any overshoot is a literal false-UNSAT generator.
#[test]
fn tableau_f32_store_dj_encloses_true_margin() {
    let n_out = 6usize;
    let adv: Vec<usize> = (1..n_out).collect();
    let mut seed: u64 = 0xE9C1_05ED;
    let mut checks = 0usize;
    let mut viol = 0usize;
    let mut worst = f64::NEG_INFINITY;
    for trial in 0..6usize {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let (spec, n_in): (TwinSpec, usize) = match trial % 3 {
            0 => {
                let (cin, h, w, cw) = (3usize, 2usize, 2usize, 10usize);
                (
                    conv_bn_spec(cin, h, w, cw, 20.0, 1e-13, 1e-4, 16, n_out, seed),
                    cin * h * w,
                )
            }
            1 => {
                let (cin, h, w, cw, nb) = (2usize, 2usize, 2usize, 8usize, 8usize);
                (
                    residual_add_spec(cin, h, w, cw, nb, 2.0 / cw as f64, 16, n_out, seed),
                    cin * h * w,
                )
            }
            _ => {
                let (n_in, depth) = (8usize, 20usize);
                (no_conv_chain_spec(n_in, depth, 20, n_out, seed), n_in)
            }
        };
        // MIXED-SIGN box on purpose: the regime where the g_err storage-dust
        // account (not the D fold) carries the endpoint soundness.
        let lo = vec![-1.0; n_in];
        let hi = vec![1.0; n_in];
        let Ok(net) = TwinNet::compile(&spec) else {
            continue;
        };
        let Ok(root) =
            RootGates::build_prec_store(&net, &lo, &hi, RoundMode::Outward, None, true, true)
        else {
            continue;
        };
        let eng = BackwardEngine::new(&net, &root);
        let Ok(re) = super::bab::root_eval(&eng, &net, 0, &adv) else {
            continue;
        };
        let pts = encl_points(&net, &lo, &hi, 0, &adv, 256, 16, seed);
        for x in &pts {
            let m = dd_margins(&spec, x, 0, &adv);
            for k in 0..adv.len() {
                if re.dj[k].is_finite() {
                    checks += 1;
                    let over = re.dj[k] - m[k];
                    worst = worst.max(over);
                    if over > 1e-12 {
                        viol += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "[tableau-f32] store-arm dj enclosure: checks={checks} viol={viol} worst_over={worst:.3e}"
    );
    assert!(
        checks > 3_000,
        "enclosure sample unexpectedly small ({checks})"
    );
    assert_eq!(
        viol, 0,
        "STORE-ARM ENCLOSURE: {viol}/{checks} dj exceed the true margin \
         (worst {worst:.3e}) — the storage-dust account under-charges (false UNSAT)"
    );
}

/// #gate-gerr-max (A1) TEETH PIN: the forward gate charge contracts by
/// `max(alpha, s)`, and that constant is both SOUND and ATTAINED.
///
/// Two obligations per case, measured against `apply_gates` itself (the real
/// transform) on a layer whose slopes DIFFER, so `max` and the historical `sum`
/// are distinguishable:
///
/// (1) SOUND — the exact deviation functional the gate actually produces,
///     `sum_i (|dM'| + |dD'|) * xabs_i`, is dominated by
///     [`gate_gerr_out`]'s charge. This is the obligation `concretize_box`
///     consumes; under-charging it is a false-UNSAT generator.
/// (2) STRICTLY TIGHTER — that charge is strictly below the historical
///     `(alpha + s)` one, by exactly `1 + min/max`. Restoring the old constant
///     inside `gate_gerr_out` fails this assertion.
///
/// Plus TIGHTNESS: with `dD = 0` the bound is an EQUALITY, so `max` is the best
/// uniform constant available — anything smaller is unsound, which is why this
/// pin also guards against over-shaving in the other direction.
///
/// `RoundMode::Parity` on purpose: it stores `mm`/`dd` unwidened, so what is
/// measured is the gate TRANSFORM, not the outward enclosure widening (a
/// separate account — see `apply_gates`' header).
#[test]
fn gate_gerr_charge_contracts_by_max_alpha_s_not_the_sum() {
    use super::root::{apply_gates, gate_gerr_out};
    use super::rounding::next_up;

    let n_in = 4usize;
    let naug = n_in + 1;
    let blk = naug.div_ceil(64);
    // Bias column carries weight 1.0 in the g_err convention.
    let xabs_ext: Vec<f64> = vec![1.0, 0.25, 4.0, 0.5, 1.0];
    // Dyadic values: every reference/perturbed pair below is EXACT in f64, so
    // the realized (dM, dD) are exactly the ones written here.
    let m_ref = [0.75f64, -1.5, 0.125, 2.0, 0.5];
    let d_ref = [0.25f64, 0.5, 0.0625, 0.25, 0.125];
    let dm_full = [
        1.0f64 / 1024.0,
        -1.0 / 2048.0,
        1.0 / 4096.0,
        -1.0 / 512.0,
        0.0,
    ];
    let dd_full = [
        1.0f64 / 2048.0,
        1.0 / 1024.0,
        -1.0 / 8192.0,
        1.0 / 4096.0,
        0.0,
    ];

    // (alpha, s, c): a heuristic-reachable binary row (alpha = 1 forces
    // s >= 1/2 through `gates_from_box`), a #rebake fractional row, and the
    // task's constructed (1, 0.4) worst case.
    let cases: &[(f64, f64, f64)] = &[(1.0, 0.6, 0.3), (0.7, 0.3, 0.11), (1.0, 0.4, 0.0)];
    let mut compared = 0usize;

    for &zero_dd in &[false, true] {
        let dm = dm_full;
        let dd = if zero_dd { [0.0f64; 5] } else { dd_full };
        // The functional the INPUT error represents, in the sum convention.
        let gerr_in: f64 = (0..naug)
            .map(|i| (dm[i].abs() + dd[i].abs()) * xabs_ext[i])
            .sum();
        for &(alpha, s, c) in cases {
            let mk = |pert: bool| {
                let mv: Vec<f64> = (0..naug)
                    .map(|i| if pert { m_ref[i] + dm[i] } else { m_ref[i] })
                    .collect();
                let dv: Vec<f64> = (0..naug)
                    .map(|i| if pert { d_ref[i] + dd[i] } else { d_ref[i] })
                    .collect();
                (
                    Array2::from_shape_vec((1, naug), mv).expect("shape"),
                    Array2::from_shape_vec((1, naug), dv).expect("shape"),
                )
            };
            let (mr, dr) = mk(false);
            let (mp, dp) = mk(true);
            // The realized perturbation must be EXACTLY (dm, dd) — otherwise the
            // comparison below measures the wrong thing.
            for i in 0..naug {
                assert_eq!(mp[[0, i]] - mr[[0, i]], dm[i], "dM not exact at {i}");
                assert_eq!(dp[[0, i]] - dr[[0, i]], dd[i], "dD not exact at {i}");
            }
            let g = |m: &Array2<f64>, d: &Array2<f64>| {
                apply_gates::<f64>(
                    m,
                    d,
                    &[alpha],
                    &[s],
                    &[c],
                    n_in,
                    RoundMode::Parity,
                    None,
                    blk,
                    &xabs_ext,
                )
            };
            let (mo_r, do_r, _) = g(&mr, &dr);
            let (mo_p, do_p, _) = g(&mp, &dp);
            let measured: f64 = (0..naug)
                .map(|i| {
                    ((mo_p[[0, i]] - mo_r[[0, i]]).abs() + (do_p[[0, i]] - do_r[[0, i]]).abs())
                        * xabs_ext[i]
                })
                .sum();

            let new_charge = gate_gerr_out(alpha, s, gerr_in);
            let old_charge = next_up((alpha + s) * gerr_in);
            let ms = alpha.max(s);
            // The gate evaluates in f64, so allow its own ~few-ulp rounding of
            // ROW MAGNITUDES (that leak is the `concretize_box` depth envelope's
            // job, not this account's).
            let tol = 1.0 + 1.0e-9;

            eprintln!(
                "[gate-gerr-max] alpha={alpha} s={s} c={c} dD={} | measured={measured:.12e} \
                 new(max)={new_charge:.12e} old(sum)={old_charge:.12e} \
                 old/new={:.6}",
                if zero_dd { "0" } else { "nz" },
                old_charge / new_charge
            );

            // (1) SOUND.
            assert!(
                measured <= new_charge * tol,
                "gate charge UNDER-covers: measured {measured:.6e} > max(alpha,s) charge \
                 {new_charge:.6e} (alpha={alpha} s={s}) — false-UNSAT generator"
            );
            // (2) STRICTLY TIGHTER than the historical (alpha + s) constant.
            assert!(
                new_charge < old_charge,
                "gate charge is NOT tighter than the historical (alpha+s) one \
                 (new {new_charge:.6e} vs old {old_charge:.6e}, alpha={alpha} s={s})"
            );
            let ratio = old_charge / new_charge;
            let want = (alpha + s) / ms;
            assert!(
                (ratio - want).abs() <= 1.0e-12 * want,
                "over-charge factor {ratio} != (alpha+s)/max(alpha,s) = {want}"
            );
            // TIGHTNESS: with dD = 0 the derivation is an EQUALITY.
            if zero_dd {
                assert!(
                    measured >= new_charge / tol,
                    "max(alpha,s) is not ATTAINED (measured {measured:.6e} << charge \
                     {new_charge:.6e}) — the pin has lost its grip on the constant"
                );
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 6, "teeth pin did not run every case");
}

/// #gate-gerr-max (A1) NO-CHANGE PIN: neurons with `alpha = 0` (and every other
/// row where a slope is zero) see the charge BIT-IDENTICALLY — the over-charge
/// factor `(alpha + s)/max(alpha, s)` is exactly 1.0 there.
///
/// This is the honesty guard on any aggregate claim: ny's rule
/// (`gates_from_box`, `alpha = [u >= -l]`) sets `alpha = 0` on every unstable
/// neuron with `u < -l`, and those rows are unchanged by A1, so a measured
/// aggregate win can only come from the `alpha = 1` population. Stable-inactive
/// rows and negative baked splits (`(0, 0)`) are likewise untouched.
#[test]
fn gate_gerr_charge_is_bit_identical_where_either_slope_is_zero() {
    use super::root::{gate_gerr_out, gates_from_box};
    use super::rounding::next_up;

    // ny's rule, quoted from the source of truth rather than assumed.
    let l = [-2.0f64, -1.0, -0.25, 0.5, -3.0];
    let u = [1.0f64, 2.0, 1.0, 2.0, 0.0];
    let (alpha, s, _c) = gates_from_box(&l, &u);
    assert_eq!(alpha[0], 0.0, "u < -l must give alpha = 0 (ny's rule)");
    assert_eq!(alpha[4], 0.0, "stable-inactive must give alpha = 0");
    assert_eq!(alpha[1], 1.0, "u >= -l must give alpha = 1");

    let gs: &[f64] = &[
        0.0,
        f64::MIN_POSITIVE,
        1.0e-300,
        3.7e-17,
        1.0,
        6.25e3,
        1.0e300,
    ];
    let dusts: &[f64] = &[0.0, 1.0e-30, 2.5e-9];
    let mut pinned = 0usize;
    for j in 0..alpha.len() {
        if alpha[j] != 0.0 && s[j] != 0.0 {
            continue;
        }
        let (a, sv) = (alpha[j], s[j]);
        assert_eq!(
            (a + sv).to_bits(),
            a.max(sv).to_bits(),
            "a zero slope must make sum and max the SAME f64 (alpha={a} s={sv})"
        );
        for &g in gs {
            let new = gate_gerr_out(a, sv, g);
            let old = next_up((a + sv) * g);
            assert_eq!(
                new.to_bits(),
                old.to_bits(),
                "alpha={a} s={sv} g={g:e}: A1 moved a zero-slope row \
                 ({new:e} vs {old:e}) — it must be a NO-OP there"
            );
            // The #tableau-f32 storage composition must be a no-op too.
            for &dust in dusts {
                assert_eq!(
                    next_up(new + dust).to_bits(),
                    next_up(old + dust).to_bits(),
                    "alpha={a} s={sv} g={g:e} dust={dust:e}: storage-lane \
                     composition moved on a zero-slope row"
                );
            }
            pinned += 1;
        }
    }
    assert!(pinned >= 14, "zero-slope pin covered too little ({pinned})");
    // Positive control: the SAME comparison on an alpha = 1 row does move.
    assert!(
        gate_gerr_out(1.0, 0.6, 1.0) < next_up((1.0 + 0.6) * 1.0),
        "positive control failed — the pin above would pass vacuously"
    );
}

/// #gate-gerr-max-nonuniform: the uniformity hypothesis is NOT load-bearing.
///
/// `gate_gerr_out`'s header ends "alpha_j and s_j are uniform over the
/// coefficient index i, so the factor pulls straight out of the aggregate", and
/// two call sites turned that into "a per-start-node alpha cannot be baked
/// forward". This simulates the gate EXACTLY on per-column errors with slopes
/// that vary over the column index, and checks the generalized charge
/// `A_j = max_i max(a_i, s_i)` still dominates.
///
/// Deterministic LCG, not a random source: a property test that cannot be
/// replayed is not evidence.
#[test]
fn nonuniform_gate_charge_dominates_the_exact_per_column_carry() {
    use super::root::gate_gerr_out_nonuniform;

    let mut state: u64 = 0x2026_0913_5EED;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut trials = 0usize;
    let mut tightest_slack = f64::INFINITY;
    for _ in 0..20_000 {
        let n = 1 + (next() * 6.0) as usize;
        let mut alphas = Vec::with_capacity(n);
        let mut slopes = Vec::with_capacity(n);
        let mut dm = Vec::with_capacity(n);
        let mut dd = Vec::with_capacity(n);
        let mut xabs = Vec::with_capacity(n);
        for _ in 0..n {
            alphas.push(next());
            slopes.push(next());
            dm.push(next() * 6.0 - 3.0);
            dd.push(next() * 6.0 - 3.0);
            xabs.push(next() * 4.0);
        }

        // g_err is the SUM-convention functional: sum_i (|dM_i| + |dD_i|) * xabs_i.
        let g_in: f64 = (0..n).map(|i| (dm[i].abs() + dd[i].abs()) * xabs[i]).sum();
        if g_in <= 0.0 {
            continue;
        }
        // The EXACT carry, column by column, each with its OWN slopes.
        let g_out: f64 = (0..n)
            .map(|i| {
                let dl = alphas[i] * (dm[i] - dd[i]);
                let du = slopes[i] * (dm[i] + dd[i]);
                (f64::midpoint(dl, du).abs() + ((du - dl) / 2.0).abs()) * xabs[i]
            })
            .sum();

        let charged = gate_gerr_out_nonuniform(&alphas, &slopes, g_in);
        assert!(
            charged >= g_out - 1e-9,
            "non-uniform charge {charged} under-charged the exact carry {g_out}"
        );
        // The contraction property the `max` form exists for must survive.
        assert!(
            charged <= g_in * (1.0 + 1e-12),
            "charge {charged} exceeds g_in {g_in}: the functional GREW across a relu"
        );
        tightest_slack = tightest_slack.min(charged - g_out);
        trials += 1;
    }
    assert!(trials > 15_000, "too few usable trials ({trials})");
    // The bound must be ATTAINED somewhere, or it is not the best constant and
    // this test would pass for a wildly loose charge too.
    assert!(
        tightest_slack < 1e-6,
        "the charge is never tight (min slack {tightest_slack}); the test has no power"
    );
}

/// NEGATIVE CONTROL for the test above: charging ONE column's slopes as if they
/// were uniform is genuinely unsound, so the property being checked is not
/// vacuous. If this ever stops finding a violation, the test above is asserting
/// nothing.
#[test]
fn charging_a_single_columns_slopes_as_if_uniform_is_unsound() {
    // Column 0 is nearly flat, column 1 is steep. Charging column 0's slopes
    // for the whole aggregate under-charges column 1's carry.
    let alphas = [0.0_f64, 1.0];
    let slopes = [0.0_f64, 1.0];
    let dm = [1.0_f64, 1.0];
    let dd = [0.0_f64, 0.0];
    let xabs = [1.0_f64, 1.0];

    let g_in: f64 = (0..2).map(|i| (dm[i].abs() + dd[i].abs()) * xabs[i]).sum();
    let g_out: f64 = (0..2)
        .map(|i| {
            let dl = alphas[i] * (dm[i] - dd[i]);
            let du = slopes[i] * (dm[i] + dd[i]);
            (f64::midpoint(dl, du).abs() + ((du - dl) / 2.0).abs()) * xabs[i]
        })
        .sum();

    let unsound = super::root::gate_gerr_out(alphas[0], slopes[0], g_in);
    assert!(
        unsound < g_out,
        "the negative control did not reproduce the under-charge it exists to show"
    );

    let sound = super::root::gate_gerr_out_nonuniform(&alphas, &slopes, g_in);
    assert!(
        sound >= g_out,
        "the generalized charge must cover what the single-column charge misses"
    );
}

/// Uniform slopes must collapse to exactly the shipped scalar charge, so
/// adopting the general form can never move a bound that is uniform today.
#[test]
fn nonuniform_charge_collapses_to_the_scalar_charge_when_slopes_are_uniform() {
    use super::root::{gate_gerr_out, gate_gerr_out_nonuniform};
    for &(a, s) in &[
        (0.0_f64, 0.0_f64),
        (1.0, 1.0),
        (1.0, 0.6),
        (0.25, 0.75),
        (0.5, 0.0),
    ] {
        for &g in &[0.0_f64, 1.0, 1e-9, 1e6] {
            let uniform = gate_gerr_out(a, s, g);
            let general = gate_gerr_out_nonuniform(&[a, a, a], &[s, s, s], g);
            assert_eq!(
                uniform, general,
                "uniform ({a}, {s}) must be bit-identical, got {uniform} vs {general}"
            );
        }
    }
    // Empty must be the IDENTITY charge, never 0 — a zero factor would admit
    // everything.
    assert_eq!(gate_gerr_out_nonuniform(&[], &[], 3.5), 3.5);
}

/// #gate-gerr-max (A1) AGGREGATE REPORT — the honest measurement.
///
/// Prints, over the zoo nets the f32 gates already use, (a) the gate CENSUS
/// (which populations can save at all) and the per-layer aggregate charge
/// multiplier `sum_j (alpha+s) / sum_j max(alpha,s)`, and (b) the END-TO-END
/// observable: the mean/max published box-width EXCESS of the two f32 lanes
/// over the f64 lane, which is what the g_err charge buys or costs. Re-running
/// this test on a build with the old constant is the before/after measurement;
/// the census half is lane-invariant (it is read off the f64 build) and is
/// there to keep any headline number attributable.
///
/// It is also a pin: the f32 lanes must still CONTAIN the f64 boxes (a
/// nonnegative excess under the shipped directed-rounding allowance), and both
/// the saving and the non-saving populations must be non-empty, or the report
/// means nothing.
#[test]
fn gate_gerr_aggregate_charge_excess_report() {
    let n_out = 6usize;
    let mut seed: u64 = 0x7AB1_EAF3;
    let (mut n_active, mut n_a1_unstable, mut n_a0, mut n_inactive) =
        (0usize, 0usize, 0usize, 0usize);
    let (mut sum_old, mut sum_new) = (0.0f64, 0.0f64);
    let (mut exc_cmp_sum, mut exc_cmp_max, mut exc_cmp_n) = (0.0f64, 0.0f64, 0usize);
    let (mut exc_st_sum, mut exc_st_max, mut exc_st_n) = (0.0f64, 0.0f64, 0usize);
    // (trunk relu layers, neurons, f32-compute excess sum, f32-store excess sum)
    let kinds = ["conv_bn", "residual_add", "no_conv_chain", "large_head"];
    let mut by_kind = [(0usize, 0usize, 0.0f64, 0.0f64); 4];
    for trial in 0..8usize {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
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
        let lo = vec![-1.0; n_in];
        let hi = vec![1.0; n_in];
        let Ok(net) = TwinNet::compile(&spec) else {
            continue;
        };
        let b = |use_f32: bool, store: bool| {
            RootGates::build_prec_store(&net, &lo, &hi, RoundMode::Outward, None, use_f32, store)
                .ok()
        };
        let (Some(r64), Some(rcmp), Some(rst)) = (b(false, false), b(true, false), b(true, true))
        else {
            continue;
        };
        for lg in &r64.layers {
            for j in 0..lg.n {
                let (a, sv) = (lg.alpha[j], lg.s[j]);
                sum_old += a + sv;
                sum_new += a.max(sv);
                if a == 1.0 && sv == 1.0 {
                    n_active += 1;
                } else if a == 1.0 {
                    n_a1_unstable += 1;
                } else if sv == 0.0 {
                    n_inactive += 1;
                } else {
                    n_a0 += 1;
                }
            }
        }
        let tally = |r: &RootGates, sum: &mut f64, mx: &mut f64, cnt: &mut usize| -> f64 {
            let mut local = 0.0f64;
            for (lgf, lg64) in r.layers.iter().zip(&r64.layers) {
                for j in 0..lg64.n {
                    if !(lgf.l[j].is_finite()
                        && lgf.u[j].is_finite()
                        && lg64.l[j].is_finite()
                        && lg64.u[j].is_finite())
                    {
                        continue;
                    }
                    let eps = 8.0 * f64::EPSILON * (lg64.l[j].abs().max(lg64.u[j].abs()) + 1.0);
                    let excess = (lgf.u[j] - lgf.l[j]) - (lg64.u[j] - lg64.l[j]);
                    assert!(
                        excess >= -2.0 * eps,
                        "trial {trial} j={j}: an f32 lane box is NARROWER than the f64 box \
                         (excess {excess:.3e}) — containment broken"
                    );
                    local += excess.max(0.0);
                    *sum += excess.max(0.0);
                    *mx = mx.max(excess);
                    *cnt += 1;
                }
            }
            local
        };
        let n_before = exc_cmp_n;
        let l_cmp = tally(&rcmp, &mut exc_cmp_sum, &mut exc_cmp_max, &mut exc_cmp_n);
        let l_st = tally(&rst, &mut exc_st_sum, &mut exc_st_max, &mut exc_st_n);
        // Per-net-kind attribution: the charge COMPOUNDS with trunk depth, so a
        // pooled headline number is only interpretable next to the depth that
        // produced it.
        let k = &mut by_kind[trial % 4];
        k.0 = r64.layers.len();
        k.1 += exc_cmp_n - n_before;
        k.2 += l_cmp;
        k.3 += l_st;
    }
    let n_tot = n_active + n_a1_unstable + n_a0 + n_inactive;
    assert!(
        n_tot > 0 && exc_cmp_n > 0 && exc_st_n > 0,
        "zoo produced nothing"
    );
    eprintln!(
        "[gate-gerr-max] CENSUS neurons={n_tot} stable_active={n_active} \
         alpha1_unstable={n_a1_unstable} alpha0_unstable={n_a0} inactive={n_inactive} \
         layer_charge_ratio(old_sum/new_sum)={:.4}",
        sum_old / sum_new
    );
    eprintln!(
        "[gate-gerr-max] EXCESS f32compute n={exc_cmp_n} mean={:.6e} max={exc_cmp_max:.6e} | \
         f32store n={exc_st_n} mean={:.6e} max={exc_st_max:.6e}",
        exc_cmp_sum / exc_cmp_n as f64,
        exc_st_sum / exc_st_n as f64
    );
    for (name, k) in kinds.iter().zip(&by_kind) {
        if k.1 == 0 {
            continue;
        }
        eprintln!(
            "[gate-gerr-max] EXCESS[{name}] relu_layers={} n={} mean_f32compute={:.6e} \
             mean_f32store={:.6e}",
            k.0,
            k.1,
            k.2 / k.1 as f64,
            k.3 / k.1 as f64
        );
    }
    assert!(
        n_active + n_a1_unstable > 0,
        "no alpha = 1 neurons — nothing in this zoo can save, report is vacuous"
    );
    assert!(
        n_a0 + n_inactive > 0,
        "no zero-slope neurons — the no-change population is missing from the report"
    );
}

// ---------------------------------------------------------------------------
// #margin-row-domain-alpha
// ---------------------------------------------------------------------------

/// THE TEETH TEST. A DELIBERATELY TERRIBLE per-domain alpha must still produce
/// an ENCLOSING bound.
///
/// This is the module's whole soundness claim in executable form: `alpha*y <=
/// relu(y)` holds pointwise on all of R for ANY `alpha in [0, 1]`, so a bad
/// optimizer costs tightness and CANNOT manufacture a proof — provided the
/// write site clamps to `[0, 1]` and re-derives the error-contraction constant
/// `ms = max(alpha, s)`. The perverse vectors below include the two extremes,
/// the midpoint, the exact INVERSION of the heuristic value (the worst thing a
/// broken gradient could converge to), and out-of-range garbage that only the
/// clamp saves. Both lanes are checked: alpha enters the Upper lane too, and an
/// under-scaled `ms` would surface here as a certified lower bound EXCEEDING a
/// sampled value — the false-UNSAT direction.
#[test]
fn domain_alpha_bounds_stay_sound_for_any_alpha() {
    let (net, root) = beta_fixture(0xDA1F_0001);
    let eng = BackwardEngine::new(&net, &root);
    let mut rng = StdRng::seed_from_u64(4242);
    let x = sample_box(&mut rng, &root, 600);
    let mut checked = 0usize;
    for &(li, pos) in &[(1usize, 0usize), (2usize, 0usize)] {
        let rec = &root.layers[li];
        assert!(!rec.unst.is_empty(), "fixture: layer {li} has no unstable");
        let neuron = rec.unst[pos];
        let relu_op = rec.op;
        let mut pre_sel = std::collections::BTreeMap::new();
        pre_sel.insert(relu_op, vec![neuron]);
        let (y, pre) = net.forward_points(&x, &pre_sel).expect("forward");
        let z = &pre[&relu_op];
        for &sign in &[1i8, -1i8] {
            let split = (li, pos, sign);
            let base_dom = domain_gates(&root, &[split]);
            let slots = super::domain_alpha::movable(&root, &base_dom, &[split]);
            assert!(
                !slots.is_empty(),
                "fixture: no movable neuron at li={li} sign={sign}"
            );
            let heuristic = super::domain_alpha::read_alphas(&root, &base_dom, &slots);
            let perverse: Vec<Vec<f64>> = vec![
                vec![0.0; slots.len()],
                vec![1.0; slots.len()],
                vec![0.5; slots.len()],
                heuristic.iter().map(|a| 1.0 - a).collect(),
                // Out of range in BOTH directions: only the clamp saves these.
                vec![-3.0; slots.len()],
                vec![7.0; slots.len()],
            ];
            for (k, vals) in perverse.iter().enumerate() {
                let mut dom = domain_gates(&root, &[split]);
                super::domain_alpha::write_alphas(&root, &mut dom, &slots, vals);
                // The write site's two obligations, checked directly.
                for (lj, gv) in &dom.layers {
                    for j in 0..gv.alpha.len() {
                        assert!(
                            (0.0..=1.0).contains(&gv.alpha[j]),
                            "alpha out of [0,1] at {lj}/{j} (case {k})"
                        );
                        assert_eq!(
                            gv.ms[j],
                            gv.alpha[j].max(gv.s[j]),
                            "ms invariant broken at {lj}/{j} (case {k})"
                        );
                    }
                }
                let seed = eng.identity_seed();
                let plo = eng
                    .run(&seed, Some(&dom), LaneDir::Lower, None, false)
                    .expect("lower pass");
                let lows = eng.concretize_lower(&plo);
                let pup = eng
                    .run(&seed, Some(&dom), LaneDir::Upper, None, false)
                    .expect("upper pass");
                let ups = eng.concretize_upper(&pup);
                let mut kept = 0usize;
                for bcol in 0..x.ncols() {
                    let zb = z[[0, bcol]];
                    if (sign > 0 && zb < 0.0) || (sign < 0 && zb > 0.0) {
                        continue; // outside this split's region
                    }
                    kept += 1;
                    for i in 0..y.nrows() {
                        let yv = y[[i, bcol]];
                        assert!(
                            lows[i] <= yv + 1e-7,
                            "li={li} sign={sign} case={k} row {i}: certified lower {} \
EXCEEDS sampled y {yv} (false-UNSAT direction)",
                            lows[i]
                        );
                        assert!(
                            ups[i] >= yv - 1e-7,
                            "li={li} sign={sign} case={k} row {i}: certified upper {} \
BELOW sampled y {yv}",
                            ups[i]
                        );
                    }
                }
                assert!(kept > 20, "vacuous region filter: kept={kept}");
                checked += 1;
            }
        }
    }
    assert!(checked >= 24, "vacuous sweep: checked={checked}");
}

/// Admission: only genuine relaxation TRIANGLES (`0 < s < 1`) are movable, and
/// a split-pinned position is excluded on top of that. Every exact line —
/// stable-active, stable-inactive, split-baked — must be untouchable, because
/// loosening one costs the child exactly the tightness the split bought.
#[test]
fn domain_alpha_movable_admits_only_relaxed_unpinned_neurons() {
    let (_net, root) = beta_fixture(0xDA1F_0002);
    let split = (1usize, 0usize, 1i8);
    let pinned_idx = root.layers[1].unst[0];
    let dom = domain_gates(&root, &[split]);
    let slots = super::domain_alpha::movable(&root, &dom, &[split]);
    assert!(!slots.is_empty(), "fixture: nothing movable");
    assert!(
        !slots.iter().any(|s| s.li == 1 && s.idx == pinned_idx),
        "the split-pinned neuron is movable"
    );
    for s in &slots {
        let sv = dom
            .layers
            .get(&s.li)
            .map_or(root.layers[s.li].s[s.idx], |g| g.s[s.idx]);
        assert!(
            sv > 0.0 && sv < 1.0,
            "admitted an exact line at {}/{} (s={sv})",
            s.li,
            s.idx
        );
        assert!(
            root.layers[s.li].unst.contains(&s.idx),
            "admitted a neuron outside `unst` at {}/{} — no gradient row exists",
            s.li,
            s.idx
        );
    }
    // The unpinned domain admits strictly more: the pin is the only difference.
    let bare = super::domain_alpha::movable(&root, &DomainGates::default(), &[]);
    assert_eq!(
        bare.len(),
        slots.len() + 1,
        "the pin should remove exactly one slot"
    );
}

/// THE SCHEDULE IS THE COST KNOB, so it is pinned as a pure rule rather than
/// only observed through a 100 s run. The shipped defaults (`min_depth 0`,
/// `stride 1`, `gap` unlimited) must admit EVERYTHING — otherwise arming the
/// mechanism would silently measure a different arm than the one the 2x2
/// measured.
#[test]
fn domain_alpha_schedule_defaults_admit_every_domain() {
    let admits = super::domain_alpha::schedule_admits_with;
    let unlimited = f64::INFINITY;
    for depth in 0..24usize {
        for b in [-9.0, -0.2, -1e-9, 0.0, 0.5] {
            assert!(
                admits(depth, b, 0, 1, unlimited),
                "default schedule declined depth={depth} b={b}"
            );
        }
    }
    // A non-finite bound is only consulted when a WINDOW is set; with no
    // window the pre-schedule behaviour stands exactly.
    assert!(admits(3, f64::NAN, 0, 1, unlimited));
    assert!(!admits(3, f64::NAN, 0, 1, 0.05));

    // Depth floor: strictly shallower is declined, the floor itself pays.
    assert!(!admits(2, -0.1, 3, 1, unlimited));
    assert!(admits(3, -0.1, 3, 1, unlimited));
    assert!(admits(30, -0.1, 3, 1, unlimited));

    // Stride: `depth % k == 0` pays, and depth 0 (the root domain) always does.
    assert!(admits(0, -0.1, 0, 3, unlimited));
    assert!(!admits(1, -0.1, 0, 3, unlimited));
    assert!(!admits(2, -0.1, 0, 3, unlimited));
    assert!(admits(3, -0.1, 0, 3, unlimited));
    // A zero stride cannot divide by zero: it reads as 1.
    assert!(admits(7, -0.1, 0, 0, unlimited));

    // Closure window: `b >= -gap` pays. The boundary is INCLUSIVE, and an
    // already-closed domain is trivially inside it (the caller's `closed`
    // guard is what actually keeps it out).
    assert!(admits(5, -0.05, 0, 1, 0.05));
    assert!(!admits(5, -0.050_001, 0, 1, 0.05));
    assert!(admits(5, 0.3, 0, 1, 0.05));

    // The three compose as a conjunction: failing any one declines.
    assert!(admits(4, -0.01, 2, 2, 0.05));
    assert!(!admits(4, -0.5, 2, 2, 0.05), "window");
    assert!(!admits(5, -0.01, 2, 2, 0.05), "stride");
    assert!(!admits(1, -0.01, 2, 1, 0.05), "floor");
}

/// The proposal is PROJECTED into `[0, 1]` and NORMALIZED so the largest-`|g|`
/// slot moves by exactly `eta`. A degenerate direction refuses rather than
/// proposing a no-op the caller would pay a certified pass to score.
#[test]
fn domain_alpha_propose_is_projected_and_normalized() {
    let slots = [
        super::domain_alpha::Slot { li: 0, idx: 0 },
        super::domain_alpha::Slot { li: 0, idx: 1 },
        super::domain_alpha::Slot { li: 0, idx: 2 },
    ];
    let cur = [0.5, 0.5, 0.02];
    // g: max |g| is 2.0 at slot 1, so slot 1 moves by exactly eta.
    let g = vec![vec![1.0, -2.0, -1.0]];
    let out = super::domain_alpha::propose(&slots, &cur, &g, 0.25).expect("proposal");
    assert!((out[0] - (0.5 + 0.25 * 0.5)).abs() < 1e-12, "{out:?}");
    assert!((out[1] - (0.5 - 0.25)).abs() < 1e-12, "{out:?}");
    // Projection: 0.02 - 0.125 would be negative; it clamps at the floor.
    assert_eq!(out[2], 0.0, "{out:?}");
    // A saturated iterate that cannot move refuses rather than burning a pass.
    assert!(
        super::domain_alpha::propose(&slots, &[1.0, 1.0, 1.0], &[vec![1.0, 2.0, 1.0]], 0.25)
            .is_none(),
        "an all-saturated ascent must refuse"
    );
    // Degenerate directions refuse.
    assert!(super::domain_alpha::propose(&slots, &cur, &[vec![0.0; 3]], 0.25).is_none());
    assert!(super::domain_alpha::propose(&slots, &cur, &g, 0.0).is_none());
    assert!(super::domain_alpha::propose(&slots, &cur, &g, f64::NAN).is_none());
    assert!(
        super::domain_alpha::propose(&slots, &cur[..2], &g, 0.25).is_none(),
        "length mismatch must refuse"
    );
}

/// Inheritance round-trips, and an entry naming a neuron the CHILD has since
/// split is DROPPED rather than written over its exact line.
#[test]
fn domain_alpha_carry_warm_starts_children_but_never_over_an_exact_line() {
    let (_net, root) = beta_fixture(0xDA1F_0003);
    let parent_split = (1usize, 0usize, 1i8);
    let mut parent = domain_gates(&root, &[parent_split]);
    let slots = super::domain_alpha::movable(&root, &parent, &[parent_split]);
    assert!(slots.len() > 2, "fixture: too few movable slots");
    // A "parent accepted" vector: every slot moved off the root's heuristic.
    let accepted: Vec<f64> = super::domain_alpha::read_alphas(&root, &parent, &slots)
        .iter()
        .map(|a| if *a > 0.5 { 0.375 } else { 0.625 })
        .collect();
    super::domain_alpha::write_alphas(&root, &mut parent, &slots, &accepted);
    let carried = super::domain_alpha::carry(&root, &slots, &accepted);
    assert_eq!(
        carried.len(),
        slots.len(),
        "every moved slot must be carried"
    );
    assert!(
        carried
            .windows(2)
            .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)),
        "carry must be sorted by (layer, neuron) for the binary search"
    );

    // The child splits one MORE neuron: the one the parent moved at slots[1].
    let child_pos = root.layers[slots[1].li]
        .unst
        .iter()
        .position(|&j| j == slots[1].idx)
        .expect("slot is an unstable neuron");
    let child_split = (slots[1].li, child_pos, -1i8);
    let child_trunk = [parent_split, child_split];
    let mut child = domain_gates(&root, &child_trunk);
    let child_slots = super::domain_alpha::movable(&root, &child, &child_trunk);
    let mut child_cur = super::domain_alpha::read_alphas(&root, &child, &child_slots);
    let applied = super::domain_alpha::apply_inherited(
        &root,
        &mut child,
        &child_slots,
        &mut child_cur,
        &carried,
    );
    assert_eq!(
        applied,
        carried.len() - 1,
        "exactly the freshly-split neuron's entry must be dropped"
    );
    // The fresh split's exact line survived the warm start untouched.
    let gv = child.layers.get(&child_split.0).expect("split layer");
    assert_eq!(
        gv.alpha[slots[1].idx], 0.0,
        "inactive split alpha overwritten"
    );
    assert_eq!(gv.s[slots[1].idx], 0.0, "inactive split chord overwritten");
    assert_eq!(gv.ms[slots[1].idx], 0.0, "inactive split ms overwritten");
    // Every other carried value arrived, with `ms` re-established.
    for (k, sl) in child_slots.iter().enumerate() {
        let g = child.layers.get(&sl.li).expect("materialized layer");
        assert_eq!(g.alpha[sl.idx], child_cur[k], "warm start not applied");
        assert_eq!(
            g.ms[sl.idx],
            g.alpha[sl.idx].max(g.s[sl.idx]),
            "ms invariant broken by the warm start at {}/{}",
            sl.li,
            sl.idx
        );
    }
}

/// The carry is CAPPED, and the cap keeps the LARGEST deltas — the truncation
/// argument in `domain_alpha::CARRY_CAP`. A dropped entry is a tightness event,
/// never a soundness one, so this pins the selection rule rather than the size.
#[test]
fn domain_alpha_carry_is_capped_at_the_largest_deltas() {
    let (_net, root) = beta_fixture(0xDA1F_0004);
    let dom = DomainGates::default();
    let slots = super::domain_alpha::movable(&root, &dom, &[]);
    assert!(slots.len() > 4, "fixture: too few movable slots");
    let base = super::domain_alpha::read_alphas(&root, &dom, &slots);
    // Unchanged alphas are not carried at all.
    assert_eq!((super::domain_alpha::carry(&root, &slots, &base)).len(), 0);
    // Give slot 0 the largest delta and slot 1 the second largest, everything
    // else a hair; then check a cap of that size would keep the right ones.
    let mut cur = base.clone();
    for (k, a) in cur.iter_mut().enumerate() {
        let delta = match k {
            0 => 0.9,
            1 => 0.4,
            _ => 1e-3,
        };
        *a = (*a + delta).clamp(0.0, 1.0);
        if *a == base[k] {
            *a = (base[k] - delta).clamp(0.0, 1.0);
        }
    }
    let carried = super::domain_alpha::carry(&root, &slots, &cur);
    assert!(
        carried.len() <= super::domain_alpha::CARRY_CAP,
        "carry exceeded the cap"
    );
    let by_delta = |li: usize, idx: usize| -> f64 {
        (carried
            .iter()
            .find(|&&(l, i, _)| l as usize == li && i as usize == idx)
            .map(|&(_, _, a)| a)
            .expect("slot carried")
            - root.layers[li].alpha[idx])
            .abs()
    };
    assert!(
        by_delta(slots[0].li, slots[0].idx) > by_delta(slots[2].li, slots[2].idx),
        "the ranking the cap selects on is inverted"
    );
}

/// DEFAULT OFF, and the whole block is gated on it. `enabled()` latches once
/// per process, so this also pins that no other test in this binary arms it.
#[test]
fn domain_alpha_is_dark_by_default() {
    assert!(
        !super::domain_alpha::enabled(),
        "per-domain alpha must ship dark: an armed default would change every \
         published margin-row bound without an A/B"
    );
}

/// THE STALE-ALPHA PROBE DOES NOT TRANSFER TO THIS LANE, and this pins why.
///
/// `9e695ba92` measured, in the MULTI-OBJECTIVE joint lane
/// (`beta_crown/state/alpha/graph_runtime.rs`, `NY_MO_ALPHA_HEURISTIC`), that
/// simply RE-DERIVING the current-box heuristic `alpha = [u > -l]` beat the
/// stored ancestor-optimized alpha by +0.34 at depth 16 — "staleness costs more
/// than optimization gains". The obvious port is a zero-pass fifth arm here.
///
/// IT WOULD MEASURE EXACTLY ZERO, by construction, and the reason is a property
/// of THIS lane rather than an opinion:
///
/// * `engine::domain_gates` = `domain_gates_split_only` + `clip_tighten_gates`,
///   and NOTHING ELSE writes a per-domain trunk alpha. The first clones the
///   root's vectors and bakes exact lines at the SPLIT positions only; the
///   second is the only per-domain `(l, u)` mover — and it ALREADY calls
///   `root::gates_from_box` + `repair_upper_lines` on the tightened box
///   (`engine.rs`), which IS the probe's intervention. The Tier-2 epoch
///   refresh (`bab.rs`) does the same on its refreshed box.
/// * So for every neuron this domain has not split, `(l, u)` is BIT-IDENTICAL
///   to the root's, and the root's own alpha came from `gates_from_box` on
///   exactly those `(l, u)`. "Fresh" and "stored" are the same number.
///
/// A stale alpha therefore exists in this lane only where something moved the
/// root alpha OFF the heuristic — `#alpha-opt` or `#rebake`, both default OFF —
/// and in that case resetting to the binary heuristic DISCARDS the optimization
/// rather than refreshing it. What is genuinely missing here is the other half:
/// nothing ever OPTIMIZES alpha against the domain. That is
/// `#margin-row-domain-alpha`, and it is why this lane needs the optimizer arm
/// and not the free one.
#[test]
fn fresh_current_box_heuristic_is_a_noop_in_this_lane() {
    let mut rng = StdRng::seed_from_u64(0xDA1F_0005);
    let spec = tiny_spec(&mut rng, 0.65);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.5; 72];
    let hi = vec![0.5; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let splits = [(0usize, 0usize, 1i8), (1usize, 0usize, -1i8)];
    let dom = domain_gates(&root, &splits);
    let pinned: Vec<(usize, usize)> = splits
        .iter()
        .map(|&(li, pos, _)| (li, root.layers[li].unst[pos]))
        .collect();
    let mut compared = 0usize;
    let mut compared_unstable = 0usize;
    for (li, layer) in root.layers.iter().enumerate() {
        // The domain carries the ROOT's intermediate box verbatim: the only
        // per-domain mover is clip, which is disarmed here (the shipped state)
        // and which re-derives these very gates when it is not.
        let (fresh_alpha, mut fresh_s, mut fresh_c) =
            super::root::gates_from_box(&layer.l, &layer.u);
        if root.mode.outward() {
            super::root::repair_upper_lines(&layer.l, &layer.u, &mut fresh_s, &mut fresh_c);
        }
        let gv = dom.layers.get(&li);
        let unst: std::collections::BTreeSet<usize> = layer.unst.iter().copied().collect();
        for j in 0..layer.n {
            if pinned.contains(&(li, j)) {
                continue; // exact split line, deliberately not the heuristic
            }
            let cur = gv.map_or(layer.alpha[j], |g| g.alpha[j]);
            assert_eq!(
                cur.to_bits(),
                fresh_alpha[j].to_bits(),
                "the current-box heuristic DIFFERS from the stored gate at {li}/{j} \
({cur} vs {}) — the stale-alpha port would no longer be a no-op here and this \
test's reasoning must be re-derived",
                fresh_alpha[j]
            );
            compared += 1;
            if unst.contains(&j) {
                compared_unstable += 1;
            }
        }
    }
    // UNSTABLE neurons are the only ones with a heuristic CHOICE to get wrong
    // (`alpha = [u >= -l]`); a sweep that only saw stable neurons would prove
    // nothing.
    assert!(
        compared > 20 && compared_unstable > 4,
        "vacuous comparison: compared={compared} unstable={compared_unstable}"
    );
}

// ===================== #clip-stateful (accumulate + inherit) =================
//
// D1 of the alpha-beta-CROWN 2024 -> 2025 cifar100 diff: abc intersects each
// round's tightening against the PARENT's stored per-domain interval and writes
// it back, so children inherit an object that compounds down the tree. ny's port
// recomputed from the root frame and discarded the result, which is why its
// dose-response measured FLAT. These tests pin the three things that can go
// wrong with making it stateful: the dark arm must be byte-identical, the
// accumulated intervals must still ENCLOSE the true activations over the child
// region (the moat), and the accumulation must actually grow with depth
// (otherwise the mechanism is unfalsifiable and its null vacuous — R9).

/// Build a root with Clip-and-Verify row retention armed. Returns `None` when
/// the fixture retains nothing, so the callers stay honest about vacuity.
fn clip_chain_fixture(
    env: &mut ny_test_utils::env::EnvEditor,
    topk: &str,
) -> Option<(TwinNet, RootGates)> {
    env.set("NY_MARGIN_ROW_CLIP_ROWS", "1");
    env.set("NY_MARGIN_ROW_CLIP_INTERM", "1");
    env.set("NY_MARGIN_ROW_CLIP_TOPK", topk);
    let mut rng = StdRng::seed_from_u64(0xC11B_5EED);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.35; 72];
    let hi = vec![0.35; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let ok = root
        .layers
        .first()
        .is_some_and(|l| l.clip_rows.is_some() && l.unst.len() >= 4);
    ok.then_some((net, root))
}

#[test]
fn stateful_clip_is_byte_identical_when_dark() {
    // DEFAULT OFF, and "off" must mean the gate vectors come out bit-for-bit as
    // they did before the accumulator existed.
    ny_test_utils::env::with_env_edits(|env| {
        let Some((_net, root)) = clip_chain_fixture(env, "20") else {
            panic!("fixture retained no clip rows — the assertion would be vacuous");
        };
        env.remove("NY_MARGIN_ROW_CLIP_STATEFUL");
        let l0 = &root.layers[0];
        let splits = vec![(0usize, l0.unst.len() / 2, -1i8), (0usize, 0usize, 1i8)];
        let base = domain_gates(&root, &splits);
        let (got, acc) = domain_gates_acc(&root, &splits, None);
        assert!(acc.is_none(), "dark arm must not produce an accumulator");
        assert_eq!(base.layers.len(), got.layers.len());
        for ((ka, a), (kb, b)) in base.layers.iter().zip(&got.layers) {
            assert_eq!(ka, kb);
            for j in 0..a.alpha.len() {
                assert_eq!(a.alpha[j].to_bits(), b.alpha[j].to_bits(), "alpha at {j}");
                assert_eq!(a.s[j].to_bits(), b.s[j].to_bits(), "s at {j}");
                assert_eq!(a.c[j].to_bits(), b.c[j].to_bits(), "c at {j}");
                assert_eq!(a.ms[j].to_bits(), b.ms[j].to_bits(), "ms at {j}");
            }
        }
    });
}

#[test]
fn accumulated_clip_intervals_enclose_the_true_child_region() {
    // THE MOAT TEST. Every stored interval must enclose its neuron's true range
    // over the region the domain's splits carve out — the induction that makes
    // parent -> child inheritance sound. A violation is a false-`unsat`
    // generator, so it is checked against SAMPLED forward activations filtered
    // to the split region, not against another bound.
    ny_test_utils::env::with_env_edits(|env| {
        let Some((net, root)) = clip_chain_fixture(env, "3") else {
            panic!("fixture retained no clip rows — the assertion would be vacuous");
        };
        env.set("NY_MARGIN_ROW_CLIP_STATEFUL", "1");
        let nu = root.layers[0].unst.len();
        let chain: Vec<(usize, usize, i8)> = vec![(0, nu / 2, -1), (0, 0, 1), (0, nu - 1, -1)];
        let mut rng = StdRng::seed_from_u64(0x5AFE_5EED);
        let x = sample_box(&mut rng, &root, 3000);
        let mut pre_sel = std::collections::BTreeMap::new();
        for l in &root.layers {
            pre_sel.insert(l.op, (0..l.n).collect::<Vec<_>>());
        }
        let (_y, pre) = net.forward_points(&x, &pre_sel).expect("forward");

        let mut acc: Option<std::sync::Arc<super::clip::ClipAcc>> = None;
        let mut carried = Vec::new();
        let mut checked = 0usize;
        for depth in 1..=chain.len() {
            let splits = &chain[..depth];
            let (_dg, out) = domain_gates_acc(&root, splits, acc.as_deref());
            let out = out.expect("armed arm must produce an accumulator");
            carried.push(out.len());
            assert_eq!(
                out.refused(),
                0,
                "depth {depth}: a refused merge means an endpoint crossed — investigate \
                 before trusting any measurement from this arm"
            );
            // Sampled points that lie in THIS domain's region.
            let inside: Vec<usize> = (0..x.ncols())
                .filter(|&b| {
                    splits.iter().all(|&(li, pos, sign)| {
                        let idx = root.layers[li].unst[pos];
                        let z = pre[&root.layers[li].op][[idx, b]];
                        if sign > 0 {
                            z >= 0.0
                        } else {
                            z <= 0.0
                        }
                    })
                })
                .collect();
            for (li, layer) in root.layers.iter().enumerate() {
                for (idx, lo, hi) in out.layer_entries(li) {
                    assert!(
                        lo >= layer.l[idx] - 1e-12 && hi <= layer.u[idx] + 1e-12,
                        "depth {depth}: accumulated [{lo}, {hi}] is not a SHRINK of the root \
                         [{}, {}] at layer {li} neuron {idx}",
                        layer.l[idx],
                        layer.u[idx]
                    );
                    for &b in &inside {
                        let z = pre[&layer.op][[idx, b]];
                        assert!(
                            lo <= z + 1e-9 && hi >= z - 1e-9,
                            "depth {depth}: accumulated [{lo}, {hi}] EXCLUDES a true \
                             activation {z} at layer {li} neuron {idx} — false-unsat generator"
                        );
                        checked += 1;
                    }
                }
            }
            acc = Some(out);
        }
        eprintln!("[test clip-stateful] carried per depth: {carried:?} checks: {checked}");
        assert!(
            carried.last().is_some_and(|&n| n > 0),
            "nothing was ever accumulated — the moat check is vacuous: {carried:?}"
        );
        for w in carried.windows(2) {
            assert!(w[1] >= w[0], "accumulation must be monotone: {carried:?}");
        }
    });
}

// ======================= #clip-head MOAT + FRAME TESTS =======================
//
// Every one of these exists because a halfspace built from a certified line is
// a NECESSARY condition of its split ONLY IF the slack the lane already paid is
// subtracted from it. Drop the slack and the constraint cuts into the TRUE
// subdomain, the dual then reports a bound that is not a bound, and the lane
// manufactures `unsat` — the class fixed five times in the trunk port. So the
// tests do not merely check that the port is sound; two of them DEMONSTRATE the
// defect on a fixture where it is visible, which is the only way to know the
// check would notice.

/// The head analogue of `clip_chain_fixture`: a net, its root gates, one
/// trunk-split domain, that domain's y-pack rows, their UNTIGHTENED
/// concretization, and the recovered head slack. `None` when the fixture would
/// make an assertion vacuous, so every caller panics rather than passing
/// silently.
#[allow(clippy::type_complexity)]
fn head_clip_fixture(
    seed: u64,
    trunk: &[(usize, usize, i8)],
) -> Option<(
    TwinNet,
    RootGates,
    super::engine::PassOut,
    super::engine::PassOut,
    Vec<f64>,
    Vec<f64>,
    super::clip::HeadSlack,
)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let spec = tiny_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![-0.35; 72];
    let hi = vec![0.35; 72];
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let eng = BackwardEngine::new(&net, &root);
    let dom = domain_gates(&root, trunk);
    let (al, au) = eng
        .y_rows(if trunk.is_empty() { None } else { Some(&dom) })
        .expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let (ly_fwd, uy_fwd) = (ybox.ly.clone(), ybox.uy);
    let slack = super::clip::head_slack(
        &al, &au, &ly_fwd, &uy_fwd, &root.mid, &root.rad, &root.xabs, 0.5,
    )?;
    // Non-vacuity: at least one head neuron must actually be unstable, or a
    // head fix has nothing to constrain and every downstream assertion is empty.
    let unstable = (0..ly_fwd.len()).any(|i| ly_fwd[i] < 0.0 && uy_fwd[i] > 0.0);
    unstable.then_some((net, root, al, au, ly_fwd, uy_fwd, slack))
}

#[test]
fn head_slack_calibration_must_use_fresh_concretization() {
    // THE ORDERING TRAP, for the head lines. `pack.ly0/uy0` are mutated in place
    // by `clip_tighten_pack` whenever `margin_row.clip` is armed, and
    // `entry.ly/uy` are ancestor-intersected. Calibrating against either
    // under-recovers the slack, and an under-recovered slack makes the halfspace
    // claim the line bounds `y_i` more tightly than certified. The port must
    // REFUSE, not clamp — a clamped zero is indistinguishable from an exact line.
    let (_net, root, al, au, ly_fwd, uy_fwd, slack) =
        head_clip_fixture(0x11EA_D0C1, &[]).expect("fixture must produce a usable head pack");
    let n_y = ly_fwd.len();
    assert!(n_y > 0, "fixture has no head columns — assertions vacuous");

    // Leg 1: the recovered slack IS the gap between the line's own box minimum
    // and the bound the lane published from it, up to the rounding pad.
    let mut nonzero = 0usize;
    for i in 0..n_y {
        let mut bmin = al.b[i];
        for j in 0..root.mid.len() {
            let c = al.a[[j, i]];
            bmin += c.mul_add(root.mid[j], -(c.abs() * root.rad[j]));
        }
        let gap = bmin - ly_fwd[i];
        assert!(
            slack.sl_lo[i] >= gap - 1e-12,
            "column {i}: recovered slack {} UNDER-recovers the published gap {gap}",
            slack.sl_lo[i]
        );
        assert!(
            slack.sl_lo[i] >= 0.0 && slack.sl_up[i] >= 0.0,
            "column {i}: negative slack survived the frame gate"
        );
        if gap > 0.0 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero > 0,
        "no column carried a certified slack — this fixture cannot distinguish the \
         slacked halfspace from the unslacked one, so the moat legs would be vacuous"
    );

    // Leg 2: calibrate against a TIGHTENED bound (what reading a
    // `clip_tighten_pack`-mutated `ly0`, or an ancestor-intersected `entry.ly`,
    // would hand us) and the port must refuse the whole pack.
    let ly_tight: Vec<f64> = ly_fwd.iter().map(|v| v + 0.3).collect();
    assert!(
        super::clip::head_slack(
            &al, &au, &ly_tight, &uy_fwd, &root.mid, &root.rad, &root.xabs, 0.5
        )
        .is_none(),
        "a tightened lower calibration must REFUSE (the trap), never clamp to zero"
    );
    let uy_tight: Vec<f64> = uy_fwd.iter().map(|v| v - 0.3).collect();
    assert!(
        super::clip::head_slack(
            &al, &au, &ly_fwd, &uy_tight, &root.mid, &root.rad, &root.xabs, 0.5
        )
        .is_none(),
        "a tightened upper calibration must REFUSE (the trap)"
    );
}

#[test]
fn head_frame_gate_refuses_a_transposed_or_shifted_pack() {
    // A frame error is wrong by FACTORS, not by one slack term. Both shapes the
    // trunk port was burned by are checked here: a transposed read (caught on
    // shape, because `PassOut.a` is `(n_in, R)` while `ClipRows.m` is
    // `(unst, n_in)` — reading one as the other is "wrong in both axes at
    // once"), and a bias that does not belong to the published bound, which is
    // what a dropped augmented-bias column looks like. The +/- shift is used
    // rather than a literal `b = 0` so that both refusal legs — negative slack
    // and slack larger than the interval — fire deterministically instead of
    // depending on the fixture's random bias signs.
    let (_net, root, al, au, ly_fwd, uy_fwd, _slack) =
        head_clip_fixture(0x11EA_D0C2, &[]).expect("fixture");
    let n_y = ly_fwd.len();

    let mut t = super::engine::PassOut {
        a: al.a.t().to_owned(),
        e: None,
        b: al.b.clone(),
        eb: al.eb.clone(),
        coll: None,
        coll_rows: None,
        unst_rows: None,
    };
    assert!(
        super::clip::head_slack(&t, &au, &ly_fwd, &uy_fwd, &root.mid, &root.rad, &root.xabs, 0.5)
            .is_none(),
        "a transposed coefficient matrix must be refused on shape"
    );

    let width: Vec<f64> = (0..n_y)
        .map(|i| (uy_fwd[i] - ly_fwd[i]).abs().max(1.0))
        .collect();
    t = super::engine::PassOut {
        a: al.a.clone(),
        e: al.e.clone(),
        b: (0..n_y).map(|i| al.b[i] + 10.0 * width[i]).collect(),
        eb: al.eb.clone(),
        coll: None,
        coll_rows: None,
        unst_rows: None,
    };
    assert!(
        super::clip::head_slack(&t, &au, &ly_fwd, &uy_fwd, &root.mid, &root.rad, &root.xabs, 0.5)
            .is_none(),
        "a bias that overstates the line must be refused (slack >> interval width)"
    );

    t = super::engine::PassOut {
        a: al.a.clone(),
        e: al.e.clone(),
        b: (0..n_y).map(|i| al.b[i] - 10.0 * width[i]).collect(),
        eb: al.eb,
        coll: None,
        coll_rows: None,
        unst_rows: None,
    };
    assert!(
        super::clip::head_slack(&t, &au, &ly_fwd, &uy_fwd, &root.mid, &root.rad, &root.xabs, 0.5)
            .is_none(),
        "a bias that understates the line yields a NEGATIVE slack and must be refused, \
         never clamped to zero"
    );
}

/// A fixture in which a head split's halfspace actually BITES, and in which
/// there is something for it to bite ON.
///
/// THREE things have to hold at once, and on a random net they never do
/// together. Each was measured failing before this builder existed:
///
///  * THE HALFSPACE MUST REMOVE PART OF THE BOX. For an INACTIVE fix it is
///    `lower_line(x) - sl_lo <= 0`, whose range centre over the root box is
///    `lower_line(mid) - sl_lo`; unless that is POSITIVE the whole box already
///    satisfies it, the dual correctly returns `beta = 0`, and the step engages
///    and tightens nothing. On the plain `tiny_spec` fixture the best of 60
///    (neuron, direction) candidates across 12 seeds scored `-4.85` — satisfied
///    with five box-widths to spare.
///  * THE CHILD MUST BE NON-EMPTY under SAMPLING, not merely under the bounds.
///    `lower_line(mid)` is measured against the line's L1 half-width `hw`, but a
///    sampled `y_i` concentrates far inside that: for 72 inputs its standard
///    deviation is about `hw / 15`. Offsetting by `0.35 * hw` — an unremarkable
///    5 sigma — put ZERO of 40 000 samples in the child.
///  * THERE MUST BE OTHER UNSTABLE HEAD COLUMNS. `head_columns` only offers
///    neurons with `ly < 0 < uy`, and the split neuron is clamped stable by its
///    own fix, so a fixture where one neuron straddles zero gives `cols = 0`.
///
/// So this builder shrinks the box until the trunk relaxation is nearly exact,
/// then shifts EVERY head neuron's Gemm bias: the split neuron to `k` sampling
/// sigmas above zero (biting, and reachable), every other neuron to zero (so it
/// straddles and the dual has somewhere to put the constraint). The shifts are
/// exact bookkeeping, not tuning: the head Gemm is linear, so adding `delta` to
/// `bias[j]` adds exactly `delta` to `al.b[j]`, `au.b[j]`, `ly_fwd[j]`,
/// `uy_fwd[j]` and to the true `y_j`, and changes nothing about the trunk, the
/// gates or the certified error.
#[allow(clippy::type_complexity)]
fn head_clip_biting_fixture(
    seed: u64,
    radius: f64,
    sigmas: f64,
) -> Option<(
    TwinNet,
    RootGates,
    super::engine::PassOut,
    super::engine::PassOut,
    Vec<f64>,
    Vec<f64>,
    super::clip::HeadSlack,
    (usize, i8),
)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut spec = tiny_spec(&mut rng, 0.6);
    let lo = vec![-radius; 72];
    let hi = vec![radius; 72];
    let net = TwinNet::compile(&spec).ok()?;
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).ok()?;
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).ok()?;
    let n_in = root.mid.len();
    let n_y = al.a.ncols();
    // Per column: the line's value at the box centre, its L1 half-width, and the
    // standard deviation a uniform sample of the box actually produces.
    let mut lower_mid = vec![0.0f64; n_y];
    let mut rel = vec![f64::INFINITY; n_y];
    let mut sd = vec![0.0f64; n_y];
    for i in 0..n_y {
        let (mut lm, mut um, mut hw, mut sq) = (al.b[i], au.b[i], 0.0f64, 0.0f64);
        for j in 0..n_in {
            lm += al.a[[j, i]] * root.mid[j];
            um += au.a[[j, i]] * root.mid[j];
            hw += al.a[[j, i]].abs() * root.rad[j];
            sq += (al.a[[j, i]] * root.rad[j]).powi(2);
        }
        if !hw.is_finite() || hw <= 0.0 || !lm.is_finite() || !um.is_finite() {
            return None;
        }
        lower_mid[i] = lm;
        rel[i] = (um - lm) / hw;
        sd[i] = (sq / 3.0).sqrt();
    }
    // The split neuron is the one whose lines are TIGHTEST relative to their own
    // spread: a loose lower line can sit far below zero while the true `y_i`
    // never crosses it, which is the empty-child failure above.
    let (split, _) = rel
        .iter()
        .enumerate()
        .filter(|(_, r)| r.is_finite())
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    if rel[split] >= 0.5 || sd[split] <= 0.0 {
        return None;
    }
    let TwinOpSpec::Gemm { bias, .. } = &mut spec.ops[6] else {
        return None;
    };
    for (i, b) in bias.iter_mut().enumerate().take(n_y) {
        let want = if i == split { sigmas * sd[i] } else { 0.0 };
        *b += want - lower_mid[i];
    }
    let net = TwinNet::compile(&spec).ok()?;
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).ok()?;
    let eng = BackwardEngine::new(&net, &root);
    let (al, au) = eng.y_rows(None).ok()?;
    let ybox = YBox::from_rows(&eng, &al, &au);
    let (ly_fwd, uy_fwd) = (ybox.ly.clone(), ybox.uy);
    // Non-vacuity: the split must be splittable, and at least two OTHER columns
    // must be offerable to the dual or `cols` comes out 0.
    if !(ly_fwd[split] < 0.0 && uy_fwd[split] > 0.0) {
        return None;
    }
    if (0..n_y)
        .filter(|&i| i != split && ly_fwd[i] < 0.0 && uy_fwd[i] > 0.0)
        .count()
        < 2
    {
        return None;
    }
    let slack = super::clip::head_slack(
        &al, &au, &ly_fwd, &uy_fwd, &root.mid, &root.rad, &root.xabs, 0.5,
    )?;
    Some((net, root, al, au, ly_fwd, uy_fwd, slack, (split, -1)))
}

#[test]
fn head_clip_encloses_the_true_subdomain() {
    // THE MOAT TEST. Three legs, all against SAMPLED forward activations rather
    // than against another bound:
    //   * NECESSARY CONDITION — every sample in the true subdomain satisfies
    //     every emitted halfspace. A violation means the constraint CUT the
    //     region, which is the false-`unsat` generator this family exists for.
    //   * ENCLOSURE — after the step, every head neuron's interval still
    //     contains every in-region sample's true pre-activation.
    //   * SHRINK-ONLY, against the forward box and against the incoming box.
    // Plus the CHILD leg, the only coverage of the INDIRECT surface: a clipped
    // box is inherited one level down, re-clamped, and clipped again, which also
    // exercises the MULTI-halfspace path the parent leg does not.
    //
    // R9 is enforced by the step's OWN counter, not assumed: see
    // `head_clip_biting_fixture` for why a fixture that engages but tightens
    // nothing is the easy accident here, and what it takes to avoid one.
    let mut diag: Vec<String> = Vec::new();
    let mut ran = false;
    'search: for seed in [0x11EA_D0C3u64, 0x11EA_D0D1, 0x11EA_D0D2, 0x11EA_D0D3] {
        for radius in [0.02f64, 0.05, 0.1] {
            for target in [1.0f64, 0.6, 1.5] {
                let Some((net, root, al, au, ly_fwd, uy_fwd, slack, head_fix)) =
                    head_clip_biting_fixture(seed, radius, target)
                else {
                    diag.push(format!("seed {seed:#x} r={radius} t={target}: no fixture"));
                    continue;
                };
                let (mid, rad) = (root.mid.as_slice(), root.rad.as_slice());
                let heads = vec![head_fix];
                let (fi, fdir) = head_fix;
                let hs = super::clip::halfspaces_for_heads(
                    &al,
                    &au,
                    &slack,
                    &heads,
                    super::clip::HEAD_H_CAP,
                )
                .expect("halfspaces must build on a frame-checked pack");
                let (rlo, rhi) = super::clip::halfspace_range(&hs[0], mid, rad);
                let score = (rlo + rhi) / (rhi - rlo);

                let mut pre_sel = std::collections::BTreeMap::new();
                for l in &root.layers {
                    pre_sel.insert(l.op, (0..l.n).collect::<Vec<_>>());
                }
                let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED);
                let x = sample_box(&mut rng, &root, 40_000);
                let (y, _pre) = net.forward_points(&x, &pre_sel).expect("forward");
                let inside: Vec<usize> = (0..x.ncols())
                    .filter(|&b| {
                        if fdir > 0 {
                            y[[fi, b]] >= 0.0
                        } else {
                            y[[fi, b]] <= 0.0
                        }
                    })
                    .collect();

                let mut ybox = YBox {
                    ly: ly_fwd.clone(),
                    uy: uy_fwd.clone(),
                };
                ybox.clamp(&heads);
                let before = ybox.clone();
                let out = super::clip::tighten_head_box(
                    &al,
                    &au,
                    &slack,
                    &ly_fwd,
                    &uy_fwd,
                    &heads,
                    mid,
                    rad,
                    &mut ybox.ly,
                    &mut ybox.uy,
                    super::clip::HEAD_H_CAP,
                    super::clip::HEAD_C_CAP,
                );
                if out.moved == 0 || inside.len() < 20 {
                    diag.push(format!(
                        "seed {seed:#x} r={radius} t={target} fix={head_fix:?} score={score:.3}: \
                         hs={} cols={} moved={} sum_delta={:.3e} inside={}",
                        out.hs,
                        out.cols,
                        out.moved,
                        out.sum_delta,
                        inside.len()
                    ));
                    continue;
                }
                ran = true;
                eprintln!(
                    "[test clip-head] seed={seed:#x} r={radius} t={target} fix={head_fix:?} \
                     score={score:.3} hs={} cols={} moved={} sum_delta={:.6e} closes={} \
                     refused={} inside={}",
                    out.hs,
                    out.cols,
                    out.moved,
                    out.sum_delta,
                    out.closes,
                    out.refused_cross,
                    inside.len()
                );
                assert_eq!(
                    out.refused_cross, 0,
                    "a refused crossing needs investigating before any measurement is believed"
                );

                // LEG 1: every halfspace is a NECESSARY condition of the domain.
                for h in &hs {
                    for &b in &inside {
                        let mut v = h.d;
                        for j in 0..h.a.len() {
                            v += h.a[j] * x[[j, b]];
                        }
                        assert!(
                            v <= 1e-9,
                            "seed {seed:#x}: a head halfspace EXCLUDES a true-subdomain point \
                             (a.x + d = {v}) — false-unsat generator"
                        );
                    }
                }

                // LEGS 2 + 3: shrink-only, and enclosure of every in-region sample.
                for j in 0..ybox.ly.len() {
                    assert!(
                        ybox.ly[j] >= ly_fwd[j] - 1e-12 && ybox.uy[j] <= uy_fwd[j] + 1e-12,
                        "column {j}: [{}, {}] is not a shrink of the forward [{}, {}]",
                        ybox.ly[j],
                        ybox.uy[j],
                        ly_fwd[j],
                        uy_fwd[j]
                    );
                    assert!(
                        ybox.ly[j] >= before.ly[j] - 1e-12 && ybox.uy[j] <= before.uy[j] + 1e-12,
                        "column {j}: the step WIDENED the incoming box"
                    );
                    for &b in &inside {
                        let v = y[[j, b]];
                        assert!(
                            ybox.ly[j] <= v + 1e-9 && ybox.uy[j] >= v - 1e-9,
                            "column {j}: clipped [{}, {}] EXCLUDES a true activation {v} — \
                             false-unsat generator",
                            ybox.ly[j],
                            ybox.uy[j]
                        );
                    }
                }

                // CHILD LEG: inherit the clipped box, clamp the child's own fix
                // on top, clip again against the same pack.
                let child = (0..ly_fwd.len())
                    .find(|&i| ybox.ly[i] < 0.0 && ybox.uy[i] > 0.0 && i != fi)
                    .map(|child_i| {
                        let pos = inside.iter().filter(|&&b| y[[child_i, b]] >= 0.0).count();
                        (child_i, if pos * 2 >= inside.len() { 1i8 } else { -1i8 })
                    });
                if let Some((child_i, child_dir)) = child {
                    let mut cheads = heads.clone();
                    cheads.push((child_i, child_dir));
                    let mut cbox = ybox.clone();
                    cbox.clamp(&cheads);
                    let cinside: Vec<usize> = inside
                        .iter()
                        .copied()
                        .filter(|&b| {
                            if child_dir > 0 {
                                y[[child_i, b]] >= 0.0
                            } else {
                                y[[child_i, b]] <= 0.0
                            }
                        })
                        .collect();
                    let cout = super::clip::tighten_head_box(
                        &al,
                        &au,
                        &slack,
                        &ly_fwd,
                        &uy_fwd,
                        &cheads,
                        mid,
                        rad,
                        &mut cbox.ly,
                        &mut cbox.uy,
                        super::clip::HEAD_H_CAP,
                        super::clip::HEAD_C_CAP,
                    );
                    assert_eq!(cout.refused_cross, 0, "child step refused a crossing");
                    assert_eq!(
                        cout.hs, 2,
                        "the child must fold BOTH head fixes into one dual"
                    );
                    for j in 0..cbox.ly.len() {
                        for &b in &cinside {
                            let v = y[[j, b]];
                            assert!(
                                cbox.ly[j] <= v + 1e-9 && cbox.uy[j] >= v - 1e-9,
                                "CHILD column {j}: inherited+clipped [{}, {}] EXCLUDES a true \
                                 activation {v}",
                                cbox.ly[j],
                                cbox.uy[j]
                            );
                        }
                    }
                    eprintln!(
                        "[test clip-head] child: fix={child_i}/{child_dir} hs={} moved={} \
                         cinside={}",
                        cout.hs,
                        cout.moved,
                        cinside.len()
                    );
                }
                break 'search;
            }
        }
    }
    assert!(
        ran,
        "no fixture produced a TIGHTENING head clip, so every enclosure assertion \
         would have passed trivially (R9). Search log: {diag:#?}"
    );
}

#[test]
fn head_clip_is_byte_identical_when_dark() {
    // DEFAULT OFF, and off must mean "the y-box, the head gates and the node's
    // interval-exact path come out bit-for-bit as they did before this port".
    // The gate is read PER CALL (never latched), so this also pins that the dark
    // resolution costs nothing: `pack_from_rows` computes no slack at all.
    ny_test_utils::env::with_env_edits(|env| {
        env.remove("NY_MARGIN_ROW_CLIP_HEAD");
        assert!(
            !crate::preset_gates::margin_row_clip_head_enabled(),
            "the head clip must default DARK"
        );
        let (_net, root, al, au, ly_fwd, uy_fwd, slack) =
            head_clip_fixture(0x11EA_D0C4, &[]).expect("fixture");
        // With no head fixes the step is a structural no-op: this is the shape
        // the dark lane (and every root-domain eval) takes.
        let mut ly = ly_fwd.clone();
        let mut uy = uy_fwd.clone();
        let out = super::clip::tighten_head_box(
            &al,
            &au,
            &slack,
            &ly_fwd,
            &uy_fwd,
            &[],
            &root.mid,
            &root.rad,
            &mut ly,
            &mut uy,
            super::clip::HEAD_H_CAP,
            super::clip::HEAD_C_CAP,
        );
        assert_eq!(out.hs, 0);
        assert_eq!(out.moved, 0);
        let ybox = YBox {
            ly: ly.clone(),
            uy: uy.clone(),
        };
        let base = YBox {
            ly: ly_fwd.clone(),
            uy: uy_fwd.clone(),
        };
        for j in 0..ly.len() {
            assert_eq!(ly[j].to_bits(), ly_fwd[j].to_bits(), "ly at {j}");
            assert_eq!(uy[j].to_bits(), uy_fwd[j].to_bits(), "uy at {j}");
        }
        for mode in [RoundMode::Parity, RoundMode::Outward] {
            let a = head_gates(&ybox, mode);
            let b = head_gates(&base, mode);
            for j in 0..a.alpha.len() {
                assert_eq!(a.alpha[j].to_bits(), b.alpha[j].to_bits(), "alpha at {j}");
                assert_eq!(a.s[j].to_bits(), b.s[j].to_bits(), "s at {j}");
                assert_eq!(a.c[j].to_bits(), b.c[j].to_bits(), "c at {j}");
            }
        }
    });
}

#[test]
fn head_clip_never_writes_the_shared_pack_and_is_order_independent() {
    // The y-pack LRU is keyed by TRUNK splits alone, so every sibling with a
    // different HEAD state holds the SAME `Arc<YPack>`. Writing a head-derived
    // delta back would leak one child's constraints into all of them. The step
    // takes the pack by shared reference and the box by `&mut`, so this is
    // structural — the test pins it anyway, together with determinism across
    // evaluation order (the parallel frontier's bit-identity contract).
    let (_net, root, al, au, ly_fwd, uy_fwd, slack) =
        head_clip_fixture(0x11EA_D0C5, &[]).expect("fixture");
    let unst: Vec<usize> = (0..ly_fwd.len())
        .filter(|&i| ly_fwd[i] < 0.0 && uy_fwd[i] > 0.0)
        .collect();
    assert!(
        unst.len() >= 2,
        "need two unstable head neurons for the sibling test"
    );
    let a_heads = vec![(unst[0], 1i8)];
    let b_heads = vec![(unst[0], -1i8), (unst[1], 1i8)];
    let al_bits: Vec<u64> = al.a.iter().map(|v| v.to_bits()).collect();
    let fwd_bits: Vec<u64> = ly_fwd.iter().chain(&uy_fwd).map(|v| v.to_bits()).collect();

    let run = |heads: &[(usize, i8)]| {
        let mut ly = ly_fwd.clone();
        let mut uy = uy_fwd.clone();
        let mut bx = YBox {
            ly: ly.clone(),
            uy: uy.clone(),
        };
        bx.clamp(heads);
        ly.clone_from(&bx.ly);
        uy.clone_from(&bx.uy);
        let out = super::clip::tighten_head_box(
            &al,
            &au,
            &slack,
            &ly_fwd,
            &uy_fwd,
            heads,
            &root.mid,
            &root.rad,
            &mut ly,
            &mut uy,
            super::clip::HEAD_H_CAP,
            super::clip::HEAD_C_CAP,
        );
        (ly, uy, out.moved, out.hs)
    };
    let first_a = run(&a_heads);
    let first_b = run(&b_heads);
    // Reversed order, and again from several rayon workers: pure function, so
    // every result must be bit-identical.
    let second_b = run(&b_heads);
    let second_a = run(&a_heads);
    assert_eq!(
        first_a.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second_a.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        first_b.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second_b.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert!(first_a.3 >= 1 && first_b.3 >= 1, "no halfspaces — vacuous");
    let par: Vec<Vec<u64>> = {
        use rayon::prelude::*;
        (0..8)
            .into_par_iter()
            .map(|_| {
                run(&b_heads)
                    .0
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<u64>>()
            })
            .collect()
    };
    for p in &par {
        assert_eq!(
            p,
            &first_b.0.iter().map(|v| v.to_bits()).collect::<Vec<u64>>(),
            "a rayon worker produced a different clipped box"
        );
    }
    // The pack and its forward calibration are untouched.
    assert_eq!(
        al_bits,
        al.a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the shared pack's coefficients were mutated"
    );
    assert_eq!(
        fwd_bits,
        ly_fwd
            .iter()
            .chain(&uy_fwd)
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        "the forward calibration base was mutated"
    );
}

#[test]
fn accumulation_carries_more_neurons_than_one_visit_can() {
    // R9: separates "ran and compounded" from "ran and re-derived the same
    // numbers". With `topk = 1` a single visit touches at most one neuron per
    // layer, so a carried set larger than that can only come from inheritance.
    ny_test_utils::env::with_env_edits(|env| {
        let Some((_net, root)) = clip_chain_fixture(env, "1") else {
            panic!("fixture retained no clip rows — the assertion would be vacuous");
        };
        env.set("NY_MARGIN_ROW_CLIP_STATEFUL", "1");
        let chain: Vec<(usize, usize, i8)> = vec![(0, 0, -1), (0, 1, 1), (0, 2, -1), (0, 3, 1)];
        let mut acc: Option<std::sync::Arc<super::clip::ClipAcc>> = None;
        let mut sizes = Vec::new();
        for depth in 1..=chain.len() {
            let (_dg, out) = domain_gates_acc(&root, &chain[..depth], acc.as_deref());
            let out = out.expect("armed");
            sizes.push(out.len());
            acc = Some(out);
        }
        // A STATELESS visit of the DEEPEST domain, for contrast.
        env.remove("NY_MARGIN_ROW_CLIP_STATEFUL");
        let stateless = {
            let (_dg, out) = domain_gates_acc(&root, &chain, None);
            assert!(out.is_none(), "dark arm must not accumulate");
            let mut n = 0usize;
            let hs = super::clip::halfspaces_for_splits(&root, &chain);
            for layer in &root.layers {
                let (t, _) = super::clip::tighten_intermediate(
                    layer,
                    &hs,
                    root.mid.as_slice(),
                    root.rad.as_slice(),
                    1,
                );
                n += t.len();
            }
            n
        };
        eprintln!(
            "[test clip-stateful] topk=1 carried per depth: {sizes:?} vs one stateless visit: \
             {stateless}"
        );
        for w in sizes.windows(2) {
            assert!(w[1] >= w[0], "accumulation shrank: {sizes:?}");
        }
        assert!(
            sizes.last().is_some_and(|&n| n >= stateless),
            "accumulated {sizes:?} must dominate a single stateless visit ({stateless})"
        );
    });
}

// ============================================================================
// #spec-alpha — the per-START-NODE (spec) alpha axis at the root direct pass.
//
// Reading the gates: T2 carries the column-separability argument, T3 carries
// the enclosure, T4 carries the `ms = max(alpha, s)` obligation, and T7 is the
// only one that asserts the MECHANISM rather than the plumbing. Assert those
// four BY NAME; the rest are structure.
// ============================================================================

use super::spec_alpha::{
    assign_columns, build_exceptions, gradient_moves, ladder_moves, refine, Cand, CandOrigin,
    SpecAlphaPolicy, SPEC_ALPHA_ETA, SPEC_ALPHA_LADDER,
};

/// A fresh root + engine inputs for the #spec-alpha tests.
struct SpecFix {
    net: TwinNet,
    lo: Vec<f64>,
    hi: Vec<f64>,
    t: usize,
    adv: Vec<usize>,
}

fn spec_fixture(seed: u64, scale: f64, half: f64) -> SpecFix {
    let mut rng = StdRng::seed_from_u64(seed);
    let spec = tiny_spec(&mut rng, scale);
    let net = TwinNet::compile(&spec).expect("compile");
    SpecFix {
        net,
        lo: vec![-half; 72],
        hi: vec![half; 72],
        t: 0,
        adv: vec![1usize, 2, 3],
    }
}

/// Every unstable neuron of every trunk layer, as `(layer, neuron)`.
fn all_unstable(root: &RootGates) -> Vec<(usize, usize)> {
    root.layers
        .iter()
        .enumerate()
        .flat_map(|(li, l)| l.unst.iter().map(move |&j| (li, j)))
        .collect()
}

/// Full gate override for a candidate's full-width alpha, with `ms` re-derived
/// exactly the way the root build does. This is the RANK-1 REFERENCE T2
/// compares the wide exception pass against.
fn rank1_dom(root: &RootGates, alpha: &[Vec<f64>]) -> DomainGates {
    let mut dg = DomainGates::default();
    for (li, layer) in root.layers.iter().enumerate() {
        if layer.unst.is_empty() {
            continue;
        }
        let ms: Vec<f64> = alpha[li]
            .iter()
            .zip(&layer.s)
            .map(|(a, s)| a.max(*s))
            .collect();
        dg.layers.insert(
            li,
            super::engine::GateVecs {
                alpha: alpha[li].clone(),
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms,
            },
        );
    }
    dg
}

/// S1: the DARK path is byte-identical, and an armed-but-refusing policy is
/// byte-identical too.
///
/// Bites during the `RootGates` field addition: a literal initialised to
/// anything but `None` (the epoch child is the one that matters) fails here,
/// and so does any accidental mutation of a gate by the refine path. The
/// `entries: 0` arm is the negative control that proves the refusal is a
/// refusal and not a silent run.
#[test]
fn spec_alpha_dark_root_eval_is_bitwise() {
    for seed in 0..8u64 {
        let f = spec_fixture(100 + seed, 0.6, 0.4);
        let mut root =
            RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        assert!(
            root.spec_alpha_policy.is_none(),
            "seed {seed}: a freshly built root must carry NO #spec-alpha policy — \
             the driver installs it after #rebake, never the builder"
        );
        let dark = {
            let eng = BackwardEngine::new(&f.net, &root);
            super::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("dark root_eval")
        };
        assert!(
            dark.spec_alpha.is_none(),
            "seed {seed}: a dark root_eval must report no #spec-alpha evidence"
        );
        let dark_bits: Vec<u64> = dark.dj.iter().map(|v| v.to_bits()).collect();

        // Armed but refusing: `entries = 0` must refuse loudly and change nothing.
        root.spec_alpha_policy = Some(SpecAlphaPolicy {
            width: 3,
            cands: 3,
            rounds: 1,
            entries: 0,
            deadline: None,
        });
        let refused = {
            let eng = BackwardEngine::new(&f.net, &root);
            super::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("refusing root_eval")
        };
        let refused_bits: Vec<u64> = refused.dj.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            dark_bits, refused_bits,
            "seed {seed}: entries_cap=0 must be BYTE-IDENTICAL to dark"
        );
        // And the gates themselves must be untouched by either evaluation.
        for (li, layer) in root.layers.iter().enumerate() {
            for &j in &layer.unst {
                assert_eq!(
                    layer.ms[j],
                    layer.alpha[j].max(layer.s[j]),
                    "seed {seed} layer {li} neuron {j}: ms drifted from max(alpha, s)"
                );
            }
        }
    }
}

/// T2: ONE wide pass whose columns carry per-column alpha through `Exceptions`
/// reproduces, PER COLUMN, the separate rank-1 pass under that column's alpha —
/// bitwise on the coefficient and coefficient-error lanes and on the bias, and
/// `>=` (never `<`) on the bias-error lane.
///
/// This is the test that carries the whole soundness argument: it establishes
/// column separability by EXECUTION rather than by reading the engine.
///
/// The `eb >= ` asymmetry is REAL and must not be asserted as equality: the
/// exception path charges one extra `slack16` per entry on the bias-error lane
/// (`engine.rs`, the `eb[x.row] = slack16(...)` line), which inflates the error
/// and therefore LOWERS the certified bound. A correct implementation fails an
/// equality assertion here.
#[test]
fn spec_alpha_replicated_pass_is_column_exact() {
    let mut checked_cols = 0usize;
    let mut saw_eb_slack = false;
    for seed in 0..4u64 {
        let f = spec_fixture(200 + seed, 0.7, 0.45);
        let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        let unst = all_unstable(&root);
        assert!(!unst.is_empty(), "seed {seed}: need unstable neurons");
        let eng = BackwardEngine::new(&f.net, &root);
        let (al, au) = eng.y_rows(None).expect("y_rows");
        let ybox = YBox::from_rows(&eng, &al, &au);
        let gates = head_gates(&ybox, RoundMode::Outward);

        // Random per-column alpha, one candidate per column, all on ONE class
        // so the columns are bit-identical replicas of each other apart from
        // their exceptions.
        let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ seed);
        let r = 5usize;
        // Build the move sets first, drop the empty ones, and ONLY THEN assign
        // columns -- filtering after assignment leaves a non-dense column set,
        // which `build_exceptions` correctly refuses.
        let mut move_sets: Vec<Vec<(usize, usize, f64)>> = Vec::new();
        for _ in 0..r {
            let mut moves = Vec::new();
            for &(li, j) in &unst {
                if rng.random_range(0.0..1.0) < 0.6 {
                    moves.push((li, j, rng.random_range(0.0..1.0)));
                }
            }
            if !moves.is_empty() {
                move_sets.push(moves);
            }
        }
        let cands: Vec<Cand> = move_sets
            .into_iter()
            .enumerate()
            .map(|(col, moves)| Cand {
                class_pos: 0,
                col,
                origin: CandOrigin::Ladder(col),
                moves,
            })
            .collect();
        assert!(
            cands.len() >= 2,
            "seed {seed}: need at least two candidates"
        );
        let r = cands.len();

        let klass = f.adv[0];
        let replicated: Vec<usize> = vec![klass; r];
        let mbr = MarginBatch::new(&f.net, f.t, &replicated).expect("mb");
        let msr = margin_seed(&mbr, &gates, &ybox, RoundMode::Outward);
        let (exc, entries, _max_row) = build_exceptions(&root, &cands, r).expect("exception build");
        assert!(entries > 0, "seed {seed}: no entries built");
        let wide = eng
            .run(&msr.seed, None, LaneDir::Lower, Some(&exc), false)
            .expect("wide pass");

        let mb1 = MarginBatch::new(&f.net, f.t, &[klass]).expect("mb1");
        let ms1 = margin_seed(&mb1, &gates, &ybox, RoundMode::Outward);
        for c in &cands {
            let dom = rank1_dom(&root, &c.full_alpha(&root));
            let sep = eng
                .run(&ms1.seed, Some(&dom), LaneDir::Lower, None, false)
                .expect("rank1 pass");
            for i in 0..wide.a.nrows() {
                assert_eq!(
                    wide.a[[i, c.col]].to_bits(),
                    sep.a[[i, 0]].to_bits(),
                    "seed {seed} col {} input {i}: coefficient lane differs",
                    c.col
                );
            }
            if let (Some(ew), Some(es)) = (wide.e.as_ref(), sep.e.as_ref()) {
                for i in 0..ew.nrows() {
                    assert_eq!(
                        ew[[i, c.col]].to_bits(),
                        es[[i, 0]].to_bits(),
                        "seed {seed} col {} input {i}: coefficient-ERROR lane differs",
                        c.col
                    );
                }
            }
            assert_eq!(
                wide.b[c.col].to_bits(),
                sep.b[0].to_bits(),
                "seed {seed} col {}: bias differs — s2/c2 must be the certified \
                 chord VERBATIM so the exception's bias correction is exactly 0.0",
                c.col
            );
            assert!(
                wide.eb[c.col] >= sep.eb[0],
                "seed {seed} col {}: exception bias-error {} sits BELOW the rank-1 \
                 reference {} — the extra slack16 per entry must only ever inflate it",
                c.col,
                wide.eb[c.col],
                sep.eb[0]
            );
            if wide.eb[c.col] > sep.eb[0] {
                saw_eb_slack = true;
            }
            checked_cols += 1;
        }
    }
    assert!(checked_cols >= 8, "only {checked_cols} columns checked");
    assert!(
        saw_eb_slack,
        "the per-entry slack16 on the bias-error lane never showed — the test is \
         not actually exercising the asymmetry it exists to pin"
    );
}

/// T3: the published `dj` still lower-bounds the true margin when the axis is
/// fed DELIBERATELY HOSTILE candidates — values pinned at both clamp vertices,
/// values strictly worse than the heuristic, and a dense assignment over every
/// unstable neuron.
///
/// Would fail if `ms` were not re-derived per entry from the REPLACEMENT
/// slopes: that is the false-UNSAT generator this design exists to sidestep.
#[test]
fn spec_alpha_root_dj_lower_bounds_true_margin_under_hostile_candidates() {
    const SLACK: f64 = 1e-9;
    let mut checks = 0u64;
    let mut worst_over = f64::NEG_INFINITY;
    for seed in 0..24u64 {
        let f = spec_fixture(
            300 + seed,
            0.35 + (seed % 5) as f64 * 0.18,
            0.3 + (seed % 3) as f64 * 0.12,
        );
        let mut root =
            RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        if all_unstable(&root).is_empty() {
            continue;
        }
        root.spec_alpha_policy = Some(SpecAlphaPolicy {
            width: f.adv.len(),
            cands: SPEC_ALPHA_LADDER.len(),
            rounds: 3,
            entries: 1 << 20,
            deadline: None,
        });
        let re = {
            let eng = BackwardEngine::new(&f.net, &root);
            super::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("armed root_eval")
        };
        // Hostile: every clamp vertex and a "worse than heuristic" flip, dense
        // over all unstable neurons, folded the same way `refine` folds.
        let unst = all_unstable(&root);
        let hostile: Vec<Cand> = [0.0f64, 1.0, 0.5]
            .iter()
            .enumerate()
            .map(|(col, &v)| Cand {
                class_pos: 0,
                col,
                origin: CandOrigin::Ladder(col),
                moves: unst.iter().map(|&(li, j)| (li, j, v)).collect(),
            })
            .collect();
        let eng = BackwardEngine::new(&f.net, &root);
        let (al, au) = eng.y_rows(None).expect("y_rows");
        let ybox = YBox::from_rows(&eng, &al, &au);
        let gates = head_gates(&ybox, RoundMode::Outward);
        let mut dj = re.dj;
        for (ki, &k) in f.adv.iter().enumerate() {
            let replicated: Vec<usize> = vec![k; hostile.len()];
            let mbr = MarginBatch::new(&f.net, f.t, &replicated).expect("mb");
            let msr = margin_seed(&mbr, &gates, &ybox, RoundMode::Outward);
            let (exc, _e, _m) =
                build_exceptions(&root, &hostile, hostile.len()).expect("hostile build");
            let pass = eng
                .run(&msr.seed, None, LaneDir::Lower, Some(&exc), false)
                .expect("hostile pass");
            let spec = per_class_direct(&eng, &pass, &msr, 0..hostile.len());
            for v in spec {
                if v.is_finite() {
                    dj[ki] = dj[ki].max(v);
                }
            }
        }
        // Independent enclosure check against sampled forward margins.
        let mut rng = StdRng::seed_from_u64(0xBADF00D ^ seed);
        let mut x = sample_box(&mut rng, &root, 180);
        // Corners and the centre, where a relaxation is tightest and an
        // overstatement is most likely to be visible.
        for i in 0..root.mid.len() {
            x[[i, 0]] = root.mid[i] - root.rad[i];
            x[[i, 1]] = root.mid[i] + root.rad[i];
            x[[i, 2]] = root.mid[i];
        }
        let (y, _) = f
            .net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        let margins = margins_at(&f.net, &y, f.t, &f.adv);
        for (ki, mrow) in margins.iter().enumerate() {
            for &m in mrow {
                checks += 1;
                worst_over = worst_over.max(dj[ki] - m);
                assert!(
                    dj[ki] <= m + SLACK,
                    "seed {seed} class {ki}: #spec-alpha dj {} OVERSTATES the true \
                     margin {m} — a hostile candidate produced an UNSOUND bound",
                    dj[ki]
                );
            }
        }
    }
    assert!(checks > 5_000, "only {checks} enclosure checks ran");
    assert!(
        worst_over > -1e3,
        "closest approach {worst_over} suggests the oracle never came near the bound"
    );
}

/// T4: every emitted exception carries `a2` in `[0,1]`, the UNTOUCHED certified
/// chord (`s2`/`c2` bitwise equal to the layer's), and therefore an implied
/// `m2 = |a2|.max(|s2|)` equal to `max(alpha, s)` — the one alpha-coupled
/// certified quantity, asserted directly at the only site that writes it.
#[test]
fn spec_alpha_exceptions_carry_max_alpha_s_and_untouched_chord() {
    let mut seen = 0usize;
    for seed in 0..4u64 {
        let f = spec_fixture(400 + seed, 0.65, 0.4);
        let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        // Columns assigned AFTER dropping the empty ladder entries, so the set
        // is dense; `build_exceptions` refuses a col >= cands.len().
        let cands: Vec<Cand> = (0..SPEC_ALPHA_LADDER.len())
            .map(|e| ladder_moves(&root, e))
            .filter(|m| !m.is_empty())
            .enumerate()
            .map(|(col, moves)| Cand {
                class_pos: 0,
                col,
                origin: CandOrigin::Ladder(col),
                moves,
            })
            .collect();
        if cands.is_empty() {
            continue;
        }
        let cols = cands.len();
        let (exc, _entries, _max_row) = build_exceptions(&root, &cands, cols).expect("build");
        for (&li, list) in &exc.by_layer {
            let layer = &root.layers[li];
            for x in list {
                seen += 1;
                assert!(
                    (0.0..=1.0).contains(&x.a2) && x.a2.is_finite(),
                    "layer {li} neuron {}: a2 {} escaped [0, 1]",
                    x.neuron,
                    x.a2
                );
                assert_eq!(
                    x.s2.to_bits(),
                    layer.s[x.neuron].to_bits(),
                    "layer {li} neuron {}: the certified UPPER chord slope was modified",
                    x.neuron
                );
                assert_eq!(
                    x.c2.to_bits(),
                    layer.c[x.neuron].to_bits(),
                    "layer {li} neuron {}: the certified upper INTERCEPT was modified",
                    x.neuron
                );
                assert_eq!(
                    x.a2.abs().max(x.s2.abs()).to_bits(),
                    x.a2.max(layer.s[x.neuron]).to_bits(),
                    "layer {li} neuron {}: the engine's per-entry Lipschitz constant \
                     m2 = |a2|.max(|s2|) does not equal max(alpha, s)",
                    x.neuron
                );
                assert!(
                    layer.unst.contains(&x.neuron),
                    "layer {li} neuron {}: a STABLE neuron was given a candidate",
                    x.neuron
                );
            }
        }
    }
    assert!(seen > 50, "only {seen} exceptions inspected");
}

/// T5a: an out-of-range column must REFUSE. The uniform exception arm indexes
/// `cur.l[[x.neuron, x.row]]` with no validation, so this is a PANIC inside a
/// verdict pass if the builder lets it through — not a wrong answer, a crash.
#[test]
fn spec_alpha_builder_refuses_out_of_range_row() {
    let f = spec_fixture(501, 0.6, 0.4);
    let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    let unst = all_unstable(&root);
    assert_ne!((unst).len(), 0);
    let cands = vec![Cand {
        class_pos: 0,
        col: 7,
        origin: CandOrigin::Ladder(0),
        moves: vec![(unst[0].0, unst[0].1, 0.5)],
    }];
    assert!(
        build_exceptions(&root, &cands, 3).is_err(),
        "a column beyond the pass width must refuse, not index out of bounds"
    );
}

/// T5b: a STABLE (or split-baked) neuron must REFUSE — it keeps its exact fixed
/// line and a fractional lower slope is not the relaxation its certificate
/// describes.
#[test]
fn spec_alpha_builder_refuses_stable_neuron() {
    // A STABLE neuron is a fixture requirement, not a given: at a wide enough box
    // every trunk neuron is unstable and this test would have nothing to assert.
    // Search a grid rather than assume one seed -- a NARROW box is what makes
    // neurons stable -- and fail loudly if the whole grid has none, because a
    // fixture that silently cannot bite is worse than no test.
    let mut found = None;
    'grid: for seed in 0..6u64 {
        for half in [0.02f64, 0.05, 0.12, 0.3] {
            let f = spec_fixture(502 + seed, 0.6, half);
            let root =
                RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
            // Need BOTH: a stable neuron to aim at, and at least one unstable
            // one, so the refusal is about stability rather than about the
            // layer being degenerate.
            if root.layers.iter().all(|l| l.unst.is_empty()) {
                continue;
            }
            for (li, layer) in root.layers.iter().enumerate() {
                if let Some(j) = (0..layer.n).find(|j| !layer.unst.contains(j)) {
                    found = Some((root, li, j, seed, half));
                    break 'grid;
                }
            }
        }
    }
    let (root, li, j, seed, half) =
        found.expect("no net in the grid had a stable neuron — widen the fixture search");
    assert!(
        !root.layers[li].unst.contains(&j),
        "seed {seed} half {half}: picked neuron is not stable"
    );
    let cands = vec![Cand {
        class_pos: 0,
        col: 0,
        origin: CandOrigin::Ladder(0),
        moves: vec![(li, j, 0.5)],
    }];
    assert!(
        build_exceptions(&root, &cands, 1).is_err(),
        "seed {seed} half {half}: a STABLE neuron must refuse a candidate alpha — it \
         keeps its exact fixed line and a fractional lower slope is not the \
         relaxation its certificate describes"
    );
}

/// T5c: a duplicate `(row, neuron)` at one layer must REFUSE — it would
/// double-apply the gate from a stale pre-transform snapshot and charge the
/// bias-error lane twice.
#[test]
fn spec_alpha_builder_refuses_duplicate_row_neuron() {
    let f = spec_fixture(503, 0.6, 0.4);
    let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    let unst = all_unstable(&root);
    assert_ne!((unst).len(), 0);
    let (li, j) = unst[0];
    let cands = vec![Cand {
        class_pos: 0,
        col: 0,
        origin: CandOrigin::Ladder(0),
        moves: vec![(li, j, 0.25), (li, j, 0.75)],
    }];
    assert!(
        build_exceptions(&root, &cands, 1).is_err(),
        "a duplicate (row, neuron) must refuse"
    );
}

/// T5d: an alpha outside `[0, 1]` must REFUSE. `alpha*z <= relu(z)` is what
/// makes every entry a valid DeepPoly lower line, and it holds ONLY on `[0,1]`.
#[test]
fn spec_alpha_builder_refuses_out_of_range_alpha() {
    let f = spec_fixture(504, 0.6, 0.4);
    let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    let unst = all_unstable(&root);
    assert_ne!((unst).len(), 0);
    for bad in [-0.001f64, 1.001, f64::NAN, f64::INFINITY] {
        let cands = vec![Cand {
            class_pos: 0,
            col: 0,
            origin: CandOrigin::Ladder(0),
            moves: vec![(unst[0].0, unst[0].1, bad)],
        }];
        assert!(
            build_exceptions(&root, &cands, 1).is_err(),
            "alpha {bad} outside [0, 1] must refuse"
        );
    }
}

/// T6: the fold is an ELEMENT-WISE MAX, so the armed root is never below the
/// dark one — for every class, for every candidate, however bad. Bites if
/// anyone later "simplifies" the fold into an overwrite.
#[test]
fn spec_alpha_dj_never_below_dark() {
    for seed in 0..10u64 {
        let f = spec_fixture(600 + seed, 0.5 + (seed % 4) as f64 * 0.15, 0.38);
        let mut root =
            RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        let dark = {
            let eng = BackwardEngine::new(&f.net, &root);
            super::bab::root_eval(&eng, &f.net, f.t, &f.adv)
                .expect("dark")
                .dj
        };
        for rounds in [1usize, 2, 3] {
            root.spec_alpha_policy = Some(SpecAlphaPolicy {
                width: 8,
                cands: SPEC_ALPHA_LADDER.len(),
                rounds,
                entries: 1 << 20,
                deadline: None,
            });
            let armed = {
                let eng = BackwardEngine::new(&f.net, &root);
                super::bab::root_eval(&eng, &f.net, f.t, &f.adv)
                    .expect("armed")
                    .dj
            };
            for (k, (&d, &a)) in dark.iter().zip(&armed).enumerate() {
                assert!(
                    a >= d,
                    "seed {seed} rounds {rounds} class {k}: armed dj {a} is BELOW dark {d} — \
                     the element-wise max fold is what makes this mechanism unable to \
                     cost a banked proof"
                );
            }
        }
    }
}

/// T7: the axis EARNS its name — two adversarial classes prefer DIFFERENT
/// candidates on the same net, so a single shared alpha cannot be optimal for
/// both and the per-start-node index is doing real work.
///
/// Realizability is established rather than assumed: the test scans a fixed,
/// deterministic seed list and requires that at least one net exhibits two
/// classes whose argmax candidate index differs AND that both strictly improve.
/// If no net does, the correct conclusion is that a rank-1 "best global
/// candidate" would do just as well and the axis is unproven — so this test
/// fails, which is exactly the bite it exists for.
#[test]
fn spec_alpha_beats_incumbent_on_a_conflicting_two_class_fixture() {
    let mut conflicts = 0usize;
    let mut both_improved = 0usize;
    let mut nets_examined = 0usize;
    let mut witness = String::new();
    for seed in 0..40u64 {
        let f = spec_fixture(
            700 + seed,
            0.45 + (seed % 7) as f64 * 0.12,
            0.32 + (seed % 4) as f64 * 0.08,
        );
        let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        if all_unstable(&root).is_empty() {
            continue;
        }
        nets_examined += 1;
        let eng = BackwardEngine::new(&f.net, &root);
        let (al, au) = eng.y_rows(None).expect("y_rows");
        let ybox = YBox::from_rows(&eng, &al, &au);
        let gates = head_gates(&ybox, RoundMode::Outward);
        let e_max = SPEC_ALPHA_LADDER.len();
        let ladders: Vec<Vec<(usize, usize, f64)>> =
            (0..e_max).map(|e| ladder_moves(&root, e)).collect();
        if ladders.iter().all(|m| m.is_empty()) {
            continue;
        }
        // Per class: the incumbent direct bound, and the bound under each
        // candidate. Same pass width for every class, so the comparison is
        // apples to apples.
        let mut best_e: Vec<Option<usize>> = Vec::new();
        let mut gained: Vec<bool> = Vec::new();
        for &k in &f.adv {
            let mb1 = MarginBatch::new(&f.net, f.t, &[k]).expect("mb1");
            let ms1 = margin_seed(&mb1, &gates, &ybox, RoundMode::Outward);
            let base = eng
                .run(&ms1.seed, None, LaneDir::Lower, None, false)
                .expect("base");
            let incumbent = per_class_direct(&eng, &base, &ms1, 0..1)[0];

            let cands: Vec<Cand> = ladders
                .iter()
                .enumerate()
                .filter(|(_, m)| !m.is_empty())
                .enumerate()
                .map(|(col, (e, m))| Cand {
                    class_pos: e,
                    col,
                    origin: CandOrigin::Ladder(e),
                    moves: m.clone(),
                })
                .collect();
            let cols = cands.len();
            let replicated: Vec<usize> = vec![k; cols];
            let mbr = MarginBatch::new(&f.net, f.t, &replicated).expect("mb");
            let msr = margin_seed(&mbr, &gates, &ybox, RoundMode::Outward);
            let (exc, _e, _m) = build_exceptions(&root, &cands, cols).expect("build");
            let pass = eng
                .run(&msr.seed, None, LaneDir::Lower, Some(&exc), false)
                .expect("pass");
            let spec = per_class_direct(&eng, &pass, &msr, 0..cols);
            let mut best = (None::<usize>, incumbent);
            for c in &cands {
                let v = spec[c.col];
                if v.is_finite() && v > best.1 {
                    best = (Some(c.class_pos), v);
                }
            }
            gained.push(best.0.is_some());
            best_e.push(best.0);
        }
        for a in 0..best_e.len() {
            for b in (a + 1)..best_e.len() {
                if gained[a] && gained[b] {
                    both_improved += 1;
                    if best_e[a] != best_e[b] {
                        conflicts += 1;
                        if witness.is_empty() {
                            witness = format!(
                                "seed {seed}: class {} prefers candidate {:?} while class {} \
                                 prefers {:?}",
                                f.adv[a], best_e[a], f.adv[b], best_e[b]
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(nets_examined >= 20, "only {nets_examined} usable nets");
    assert!(
        both_improved > 0,
        "no class on any of {nets_examined} nets improved over its incumbent — \
         the candidate ladder has no purchase at all and the mechanism is unproven"
    );
    assert!(
        conflicts > 0,
        "across {nets_examined} nets and {both_improved} improving class pairs, NO two \
         classes ever preferred different candidates. That means a single shared \
         candidate is as good as the per-start-node axis on this fixture family, and \
         the axis is UNPROVEN — do not fund a measurement on the strength of the \
         plumbing tests alone."
    );
    assert_ne!(witness, "");
    eprintln!("[spec-alpha-test] conflict witness: {witness} ({conflicts} conflicting pairs)");
}

/// T8: IDENTITY exceptions (`a2 == alpha[j]`) leave the coefficient and bias
/// lanes bitwise unchanged and can only INFLATE the bias error, i.e. lower the
/// bound. Pins that dense exception use is strictly one-directional, which is
/// why `ladder_moves` skips identity entries rather than emitting them.
#[test]
fn spec_alpha_identity_exceptions_only_loosen() {
    let f = spec_fixture(800, 0.65, 0.4);
    let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    let unst = all_unstable(&root);
    assert_ne!((unst).len(), 0);
    let eng = BackwardEngine::new(&f.net, &root);
    let (al, au) = eng.y_rows(None).expect("y_rows");
    let ybox = YBox::from_rows(&eng, &al, &au);
    let gates = head_gates(&ybox, RoundMode::Outward);
    let mb = MarginBatch::new(&f.net, f.t, &[f.adv[0]]).expect("mb");
    let ms = margin_seed(&mb, &gates, &ybox, RoundMode::Outward);

    let plain = eng
        .run(&ms.seed, None, LaneDir::Lower, None, false)
        .expect("plain");
    let cands = vec![Cand {
        class_pos: 0,
        col: 0,
        origin: CandOrigin::Ladder(0),
        moves: unst
            .iter()
            .map(|&(li, j)| (li, j, root.layers[li].alpha[j]))
            .collect(),
    }];
    let (exc, entries, _) = build_exceptions(&root, &cands, 1).expect("build");
    assert!(entries > 0);
    let ident = eng
        .run(&ms.seed, None, LaneDir::Lower, Some(&exc), false)
        .expect("identity");
    for i in 0..plain.a.nrows() {
        assert_eq!(
            plain.a[[i, 0]].to_bits(),
            ident.a[[i, 0]].to_bits(),
            "input {i}: an identity exception moved the coefficient lane"
        );
    }
    assert_eq!(
        plain.b[0].to_bits(),
        ident.b[0].to_bits(),
        "an identity exception moved the bias"
    );
    assert!(
        ident.eb[0] >= plain.eb[0],
        "identity bias-error {} fell below plain {}",
        ident.eb[0],
        plain.eb[0]
    );
    let dj_plain = per_class_direct(&eng, &plain, &ms, 0..1)[0];
    let dj_ident = per_class_direct(&eng, &ident, &ms, 0..1)[0];
    assert!(
        dj_ident <= dj_plain,
        "identity exceptions RAISED the bound ({dj_ident} > {dj_plain}) — \
         the slack16 charge must only ever loosen"
    );
}

/// T9: a neuron with ZERO supergradient in this column must not move.
///
/// `f64::signum` is the trap: `0.0f64.signum() == 1.0`, so a signum-based step
/// would move every neuron with no support in the column — precisely the
/// neurons that must stay put.
#[test]
fn spec_alpha_zero_gradient_neuron_does_not_move() {
    let f = spec_fixture(900, 0.6, 0.4);
    let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    let unst = all_unstable(&root);
    assert!(unst.len() >= 3, "need several unstable neurons");
    let base: Vec<Vec<f64>> = root.layers.iter().map(|l| l.alpha.clone()).collect();

    // All-zero gradient: nothing may move, at any step size.
    let zero: Vec<Vec<f64>> = root.layers.iter().map(|l| vec![0.0; l.n]).collect();
    for &eta in SPEC_ALPHA_ETA.iter() {
        assert!(
            gradient_moves(&root, &base, &zero, eta).is_empty(),
            "an all-zero supergradient produced moves at eta {eta}"
        );
    }
    // Mixed: only the supported neurons move.
    let mut mixed = zero;
    let (li0, j0) = unst[0];
    let (li1, j1) = unst[1];
    mixed[li0][j0] = 1.0;
    mixed[li1][j1] = -1.0;
    let moves = gradient_moves(&root, &base, &mixed, 0.5);
    for &(li, j, _) in &moves {
        assert!(
            (li, j) == (li0, j0) || (li, j) == (li1, j1),
            "layer {li} neuron {j} moved with a ZERO gradient"
        );
    }
    assert!(!moves.is_empty(), "supported neurons failed to move at all");
}

/// T10: an EPOCH CHILD root must carry no #spec-alpha policy. The axis fires
/// ONCE PER INSTANCE at the root direct pass; an epoch re-derives gates through
/// `gates_from_box` and a second firing would re-pay the pass against a frame
/// the axis was never measured on.
#[test]
fn spec_alpha_policy_is_none_for_epoch_child_roots() {
    let f = spec_fixture(1000, 0.6, 0.4);
    let mut root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    root.spec_alpha_policy = Some(SpecAlphaPolicy {
        width: 3,
        cands: 3,
        rounds: 1,
        entries: 1 << 16,
        deadline: None,
    });
    let unst = all_unstable(&root);
    assert_ne!((unst).len(), 0);
    let (li, j) = unst[0];
    let pos = root.layers[li]
        .unst
        .iter()
        .position(|&x| x == j)
        .expect("position");
    let refreshed = super::backward_interm::refresh_parent_frame(
        &f.net,
        &root,
        &[(li, pos, 1i8)],
        usize::MAX,
        8,
        None,
    );
    let (child, _stats) = refreshed.expect("parent-frame refresh");
    assert!(
        child.spec_alpha_policy.is_none(),
        "the epoch child root INHERITED the #spec-alpha policy — it must not. The \
         axis fires ONCE PER INSTANCE at the root direct pass; an epoch re-derives \
         gates through gates_from_box and a second firing would re-pay the pass \
         against a frame the axis was never measured on."
    );
}

/// T11: nothing in the root BUILD or in the two root-ascent owners installs the
/// policy. The driver installs it strictly AFTER `#rebake` returns, which is
/// what keeps the axis out of the FORWARD lane: `gate_gerr_out`'s contraction
/// requires alpha uniform over the coefficient index, so a per-start-node alpha
/// cannot be baked forward. This test makes that ordering mechanical rather
/// than a comment.
#[test]
fn spec_alpha_policy_is_set_after_rebake_not_before() {
    let f = spec_fixture(1100, 0.6, 0.4);
    let mut root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    assert!(
        root.spec_alpha_policy.is_none(),
        "RootGates::build installed a #spec-alpha policy"
    );
    let policy = super::RootAlphaRuntimePolicy {
        alpha_enabled: true,
        spec_alpha_enabled: true,
        ..Default::default()
    };
    let _ =
        super::alpha_opt::maybe_optimize_root_alpha(&f.net, &mut root, f.t, &f.adv, None, policy);
    assert!(
        root.spec_alpha_policy.is_none(),
        "#alpha-opt installed a #spec-alpha policy — the ascent's published-frame \
         guard calls root_eval, and it MUST observe None"
    );
    // And the policy builder is the only thing that can turn it on.
    assert!(SpecAlphaPolicy::from_policy(super::RootAlphaRuntimePolicy::default(), None).is_none());
    assert!(SpecAlphaPolicy::from_policy(policy, None).is_some());
}

/// The candidate ladder and the column assigner: deterministic, cap-respecting,
/// and it never emits a PARTIAL candidate (a half-applied ladder value is a
/// different and undocumented relaxation).
#[test]
fn spec_alpha_entry_cap_drops_whole_candidates_only() {
    let f = spec_fixture(1200, 0.65, 0.4);
    let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    let raw: Vec<(usize, CandOrigin, Vec<(usize, usize, f64)>)> = (0..SPEC_ALPHA_LADDER.len())
        .map(|e| (0usize, CandOrigin::Ladder(e), ladder_moves(&root, e)))
        .filter(|(_, _, m)| !m.is_empty())
        .collect();
    assert!(raw.len() >= 2, "need at least two non-empty ladder entries");
    let full: usize = raw.iter().map(|(_, _, m)| m.len()).sum();
    let widths: Vec<usize> = raw.iter().map(|(_, _, m)| m.len()).collect();

    let (all, entries, truncated) = assign_columns(raw.clone(), full);
    assert_eq!(entries, full);
    assert_eq!(truncated, 0);
    assert_eq!(all.len(), widths.len());
    for (i, c) in all.iter().enumerate() {
        assert_eq!(c.col, i, "columns must be dense and ascending");
        assert_eq!(c.moves.len(), widths[i], "a candidate was truncated");
    }

    let (some, entries, truncated) = assign_columns(raw.clone(), widths[0]);
    assert!(truncated > 0, "a tight cap must drop candidates");
    assert!(entries <= widths[0]);
    for c in &some {
        assert!(
            widths.contains(&c.moves.len()),
            "a PARTIAL candidate survived the cap"
        );
    }

    let (none, entries, _truncated) = assign_columns(raw, 0);
    assert!(
        none.is_empty() && entries == 0,
        "a zero cap must admit nothing"
    );
}

/// F1: a PERMUTATION of the column set must refuse.
///
/// `run_round` builds the replicated seed as
/// `replicated[i] = fail_classes[cands[i].class_pos]` but folds `spec[c.col]`
/// into `dj[fail[c.class_pos]]`. Those agree only if `cands[i].col == i`. The
/// range check (`col < cols`) and the duplicate check both PASS on a
/// permutation, and a permutation folds one class's certified bound into
/// another class's `dj` — which can close a class that was never proved. That
/// is the only false-`unsat` this module could produce, so it is refused at the
/// one site that builds the exception set.
#[test]
fn spec_alpha_builder_refuses_permuted_columns() {
    let f = spec_fixture(1301, 0.6, 0.4);
    let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    let unst = all_unstable(&root);
    assert!(unst.len() >= 2, "need two unstable neurons");
    let mk = |class_pos: usize, col: usize, which: usize| Cand {
        class_pos,
        col,
        origin: CandOrigin::Ladder(0),
        moves: vec![(unst[which].0, unst[which].1, 0.5)],
    };
    // Dense and ascending: accepted.
    let ok = vec![mk(0, 0, 0), mk(1, 1, 1)];
    assert!(
        build_exceptions(&root, &ok, 2).is_ok(),
        "a dense ascending column set must be accepted"
    );
    // The SAME two candidates with their columns swapped: every other check
    // still passes (both columns are in range, no duplicate (layer, col,
    // neuron)), so this is precisely the case the invariant exists for.
    let permuted = vec![mk(0, 1, 0), mk(1, 0, 1)];
    assert!(
        build_exceptions(&root, &permuted, 2).is_err(),
        "a PERMUTED column set must refuse: it folds one class's certified bound \
         into another class's dj and can close a class that was never proved"
    );
}

/// THE BEHAVIOURAL TEST: drive `refine` itself and assert the per-start-node
/// axis actually delivers per-class relaxations.
///
/// Every other test in this block exercises a PIECE — the builder, the
/// exception encoding, the enclosure, the fold. None of them calls `refine`, so
/// a regression in its selection (`order.sort_by`), in its round-1 candidate
/// construction, or in its round-2 base choice (`best_by_class`) would pass the
/// entire suite. That regression has already happened once: round 1's ladder is
/// class-INDEPENDENT, so a build whose only candidates are ladder values has
/// per-class *selection* and no per-class *optimisation* at all, and its null
/// would read as "the per-start-node alpha axis is a real negative".
///
/// Three properties, each of which is the thing a fix could silently lose:
///
/// 1. `classes_improved > 0` — the axis moves something at all.
/// 2. `distinct_winners >= 2` — two classes end on DIFFERENT alphas. If every
///    improving class ends on the same alpha, a rank-1 global candidate is as
///    good and there is no per-start-node axis to measure.
/// 3. `incumbent_grad_improvements > 0` — at least one class's winner came from
///    `clamp(alpha_incumbent + eta*sign(g_k), 0, 1)`, i.e. from a step taken at
///    the PUBLISHED point using that class's own supergradient. This is the
///    property that dies if the incumbent collect pass is removed, or if
///    `best_by_class` stops being seeded with the incumbent.
///
/// Aggregated across a seed grid because any single net may already sit at each
/// class's optimum; the assertions name which property failed.
#[test]
fn spec_alpha_refine_ascends_per_class_from_the_incumbent() {
    let mut nets_examined = 0usize;
    let mut refines_run = 0usize;
    let mut seeds_improved = 0usize;
    let mut seeds_distinct = 0usize;
    let mut seeds_incumbent_grad = 0usize;
    let mut seeds_grads_ok = 0usize;
    let mut best_distinct = 0usize;
    let mut witness = String::new();
    for seed in 0..32u64 {
        let f = spec_fixture(
            1400 + seed,
            0.4 + (seed % 6) as f64 * 0.14,
            0.3 + (seed % 4) as f64 * 0.09,
        );
        let root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        if all_unstable(&root).is_empty() {
            continue;
        }
        nets_examined += 1;
        // The DARK published root: exactly what `refine` is handed in
        // production, after `m1`/`m2v` and the direct pass have folded.
        let re = {
            let eng = BackwardEngine::new(&f.net, &root);
            super::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("dark root_eval")
        };
        let fail: Vec<usize> = (0..f.adv.len())
            .filter(|&k| !super::bab::closed(RoundMode::Outward, re.dj[k]))
            .collect();
        if fail.len() < 2 {
            continue;
        }
        let fail_classes: Vec<usize> = fail.iter().map(|&k| f.adv[k]).collect();
        let eng = BackwardEngine::new(&f.net, &root);
        let (al, au) = eng.y_rows(None).expect("y_rows");
        let ybox = YBox::from_rows(&eng, &al, &au);
        let gates = head_gates(&ybox, RoundMode::Outward);
        let before = re.dj.clone();
        let mut dj = re.dj;
        let pol = SpecAlphaPolicy {
            width: f.adv.len(),
            cands: 3,
            rounds: 2,
            entries: 1 << 20,
            deadline: None,
        };
        let report = refine(
            &eng,
            &f.net,
            f.t,
            pol,
            &gates,
            &ybox,
            &fail,
            &fail_classes,
            &mut dj,
        );
        refines_run += 1;
        // The fold is element-wise max, so this can never fail — assert it
        // anyway, because it is the property the whole soundness case rests on
        // and it costs nothing to re-check at the only site that drives the
        // real entry point.
        for k in 0..dj.len() {
            assert!(
                dj[k] >= before[k],
                "seed {seed} class {k}: refine LOWERED a published bound \
                 ({} -> {})",
                before[k],
                dj[k]
            );
        }
        assert!(
            report.classes_improved <= fail.len(),
            "seed {seed}: classes_improved {} exceeds the failing set {}",
            report.classes_improved,
            fail.len()
        );
        assert!(
            report.distinct_winners <= report.classes_improved,
            "seed {seed}: distinct_winners {} cannot exceed classes_improved {}",
            report.distinct_winners,
            report.classes_improved
        );
        assert!(
            report.incumbent_grad_improvements <= report.classes_improved,
            "seed {seed}: incumbent_grad_improvements {} cannot exceed \
             classes_improved {}",
            report.incumbent_grad_improvements,
            report.classes_improved
        );
        if report.classes_improved > 0 {
            seeds_improved += 1;
        }
        if report.distinct_winners >= 2 {
            seeds_distinct += 1;
            if witness.is_empty() {
                witness = format!(
                    "seed {seed}: {} classes improved onto {} DISTINCT alphas \
                     ({} of them from the incumbent gradient), delta {:+.9}",
                    report.classes_improved,
                    report.distinct_winners,
                    report.incumbent_grad_improvements,
                    report.root_after - report.root_before
                );
            }
        }
        if report.incumbent_grad_improvements > 0 {
            seeds_incumbent_grad += 1;
        }
        if report.incumbent_grads_ok > 0 {
            seeds_grads_ok += 1;
        }
        best_distinct = best_distinct.max(report.distinct_winners);
    }
    assert!(
        nets_examined >= 16 && refines_run >= 8,
        "only {nets_examined} usable nets and {refines_run} refine calls — the \
         fixture grid stopped exercising `refine` and this test is a no-op"
    );
    assert!(
        seeds_improved > 0,
        "across {refines_run} refine calls NO class ever improved. Either the \
         mechanism is inert or the fixture family is already at its optimum; \
         both make a measurement uninterpretable."
    );
    assert!(
        seeds_distinct > 0,
        "across {refines_run} refine calls the improving classes ALWAYS ended on \
         the same alpha (best distinct_winners = {best_distinct}). A single \
         shared candidate is then as good as the per-start-node axis, which is \
         the `cols=8 delivers cols=1` shape — do not fund a measurement."
    );
    // Split the two ways the axis can be absent, because they need different
    // fixes: the gradient never RESOLVED (the incumbent collect pass is gone,
    // or `supergradient` refused this net family) versus it resolved and its
    // candidates never WON.
    assert!(
        seeds_grads_ok > 0,
        "across {refines_run} refine calls the INCUMBENT supergradient never \
         resolved for a single selected class. The per-class starting point \
         does not exist, so every candidate was a global-uniform ladder value \
         and the axis was never exercised — VOID, not negative."
    );
    assert!(
        seeds_incumbent_grad > 0,
        "across {refines_run} refine calls NOT ONE class's winner came from the \
         incumbent-gradient family. Round 1 has lost its per-class starting \
         point and the mechanism has degraded to the global-uniform ladder — \
         which is the already-measured `#interm-alpha` family, not this axis."
    );
    eprintln!("[spec-alpha-test] refine witness: {witness}");
}

// ============================================================================
// #head-alpha: ascended HEAD relu slopes (`RootGates::head_alpha`).
// ============================================================================

use super::bounds::head_gates_with;

/// The head slopes the published root bound reads are SOUND FOR ANY VALUE:
/// every clamp vertex, the midpoint, random interior slopes, and hostile
/// out-of-range / non-finite entries (clamped or ignored by `head_gates_with`).
/// Independent oracle: the net's own forward margins at sampled points,
/// corners and the centre.
#[test]
fn head_alpha_root_dj_lower_bounds_true_margin_for_any_head_slopes() {
    const SLACK: f64 = 1e-9;
    let mut checks = 0u64;
    let mut head_unstable_seen = 0usize;
    for seed in 0..16u64 {
        let f = spec_fixture(
            900 + seed,
            0.35 + (seed % 5) as f64 * 0.18,
            0.3 + (seed % 3) as f64 * 0.12,
        );
        let mut root =
            RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        let n_y = f.net.n_y;
        let (ly, uy) = {
            let eng = BackwardEngine::new(&f.net, &root);
            let (al, au) = eng.y_rows(None).expect("y_rows");
            let yb = YBox::from_rows(&eng, &al, &au);
            (yb.ly, yb.uy)
        };
        head_unstable_seen += (0..n_y).filter(|&j| ly[j] < 0.0 && uy[j] > 0.0).count();
        let mut rng = StdRng::seed_from_u64(0x4EAD ^ seed);
        let mut slopes: Vec<Vec<f64>> = vec![vec![0.0; n_y], vec![1.0; n_y], vec![0.5; n_y]];
        slopes.push((0..n_y).map(|_| rng.random_range(0.0..1.0)).collect());
        slopes.push(
            (0..n_y)
                .map(|j| match j % 4 {
                    0 => -0.3,
                    1 => 1.7,
                    2 => f64::NAN,
                    _ => f64::INFINITY,
                })
                .collect(),
        );
        let mut x = sample_box(&mut rng, &root, 160);
        for i in 0..root.mid.len() {
            x[[i, 0]] = root.mid[i] - root.rad[i];
            x[[i, 1]] = root.mid[i] + root.rad[i];
            x[[i, 2]] = root.mid[i];
        }
        let (y, _) = f
            .net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        let margins = margins_at(&f.net, &y, f.t, &f.adv);
        for ha in slopes {
            root.head_alpha = Some(ha);
            let re = {
                let eng = BackwardEngine::new(&f.net, &root);
                super::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("root_eval")
            };
            for (ki, mrow) in margins.iter().enumerate() {
                for &m in mrow {
                    checks += 1;
                    assert!(
                        re.dj[ki] <= m + SLACK,
                        "seed {seed} class {ki}: head-alpha dj {} OVERSTATES the true margin {m}",
                        re.dj[ki]
                    );
                }
            }
        }
    }
    assert!(checks > 5_000, "only {checks} enclosure checks ran");
    assert!(
        head_unstable_seen > 0,
        "no fixture had an unstable head neuron: the override was never exercised"
    );
}

/// `None` and `Some(the heuristic's own slopes)` publish bit-identical bounds,
/// and `head_gates_with(.., None)` IS `head_gates`: the disarmed path is the
/// shipped path.
#[test]
fn head_alpha_none_and_the_heuristic_slopes_are_bitwise_the_shipped_root() {
    for seed in 0..6u64 {
        let f = spec_fixture(1200 + seed, 0.5 + (seed % 3) as f64 * 0.2, 0.4);
        let mut root =
            RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        let (dj0, heur) = {
            let eng = BackwardEngine::new(&f.net, &root);
            let (al, au) = eng.y_rows(None).expect("y_rows");
            let yb = YBox::from_rows(&eng, &al, &au);
            let g0 = head_gates(&yb, RoundMode::Outward);
            let g1 = head_gates_with(&yb, RoundMode::Outward, None);
            assert_eq!(g0.alpha, g1.alpha);
            assert_eq!(g0.s, g1.s);
            assert_eq!(g0.c, g1.c);
            let re = super::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("root_eval");
            (re.dj, g0.alpha)
        };
        root.head_alpha = Some(heur);
        let dj1 = {
            let eng = BackwardEngine::new(&f.net, &root);
            super::bab::root_eval(&eng, &f.net, f.t, &f.adv)
                .expect("root_eval")
                .dj
        };
        let b0: Vec<u64> = dj0.iter().map(|v| v.to_bits()).collect();
        let b1: Vec<u64> = dj1.iter().map(|v| v.to_bits()).collect();
        assert_eq!(
            b0, b1,
            "seed {seed}: heuristic slopes moved the published root"
        );
    }
}

/// Stable head neurons keep their exact lines whatever the override says; the
/// upper chord is never touched.
#[test]
fn head_alpha_override_touches_only_unstable_slopes() {
    let yb = YBox {
        ly: vec![0.5, -2.0, -1.0, -0.5, -3.0],
        uy: vec![1.5, -0.25, 3.0, 0.25, 1.0],
    };
    let ha = [0.3, 0.3, 0.3, 0.3, 7.0];
    let base = head_gates(&yb, RoundMode::Outward);
    let g = head_gates_with(&yb, RoundMode::Outward, Some(&ha));
    assert_eq!(g.alpha, vec![1.0, 0.0, 0.3, 0.3, 1.0]);
    assert_eq!(g.s, base.s);
    assert_eq!(g.c, base.c);
    // A length mismatch is refused wholesale (the heuristic stands).
    let g2 = head_gates_with(&yb, RoundMode::Outward, Some(&ha[..3]));
    assert_eq!(g2.alpha, base.alpha);
}

/// The ascent with head coordinates: monotone under the published guard,
/// head slopes committed only inside `[0, 1]`, and the committed root still
/// encloses the true margins. Engagement is asserted across the fixture sweep
/// so a head ascent that never moves cannot pass silently.
#[test]
fn head_alpha_ascent_is_guarded_and_sound_and_engages() {
    const SLACK: f64 = 1e-9;
    let mut engaged = 0usize;
    for seed in 0..12u64 {
        let f = spec_fixture(
            1500 + seed,
            0.4 + (seed % 4) as f64 * 0.2,
            0.35 + (seed % 3) as f64 * 0.1,
        );
        let mut root =
            RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        let report = super::alpha_opt::optimize_root_alpha(
            &f.net,
            &mut root,
            f.t,
            &f.adv,
            None,
            super::alpha_opt::AlphaOptCfg {
                iters: 8,
                secs: 60.0,
                guard: true,
                ..super::alpha_opt::AlphaOptCfg::default()
            },
        );
        if let Some(ha) = root.head_alpha.as_ref() {
            let r = report
                .as_ref()
                .expect("head alpha committed without a report");
            assert!(
                r.accepted,
                "seed {seed}: head alpha committed by a rejected ascent"
            );
            let (before, after) = (
                r.published_before.expect("guarded"),
                r.published_after.expect("guarded"),
            );
            assert!(after > before, "seed {seed}: published {after} !> {before}");
            assert!(ha.iter().all(|a| (0.0..=1.0).contains(a)));
            engaged += 1;
        }
        let re = {
            let eng = BackwardEngine::new(&f.net, &root);
            super::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("root_eval")
        };
        let mut rng = StdRng::seed_from_u64(0xA5CE ^ seed);
        let x = sample_box(&mut rng, &root, 200);
        let (y, _) = f
            .net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        for (ki, mrow) in margins_at(&f.net, &y, f.t, &f.adv).iter().enumerate() {
            for &m in mrow {
                assert!(
                    re.dj[ki] <= m + SLACK,
                    "seed {seed} class {ki}: {} > {m}",
                    re.dj[ki]
                );
            }
        }
    }
    assert!(
        engaged > 0,
        "the head ascent committed on no fixture: VOID, not a pass"
    );
}

/// #router-r3 `rule.ia_yield` (ROUTER_PROGRAM §5.2): the ascent reads a
/// CONTRACT. Its share is taken of the remaining window MINUS the interm
/// credit, so the credit can only SHRINK the ascent (never extend it), and a
/// zero credit is today's budget exactly.
#[test]
fn router_r3_ascent_contract_only_shrinks_the_ascent() {
    let f = spec_fixture(3400, 0.6, 0.45);
    let cfg = |credit_secs: u64| super::alpha_opt::AlphaOptCfg {
        iters: 2,
        secs: 600.0,
        guard: false,
        contract_credit: std::time::Duration::from_secs(credit_secs),
        ..super::alpha_opt::AlphaOptCfg::default()
    };
    let run = |credit_secs: u64| {
        let mut root =
            RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(50);
        super::alpha_opt::optimize_root_alpha(
            &f.net,
            &mut root,
            f.t,
            &f.adv,
            Some(deadline),
            cfg(credit_secs),
        )
        .expect("the ascent runs on this fixture")
    };
    let today = run(0);
    let contracted = run(20);
    // Today: 0.4 x ~50 s.
    assert!(
        (today.effective_secs - 20.0).abs() < 0.5,
        "today's share is 0.4 x remaining: {}",
        today.effective_secs
    );
    assert_eq!(today.contract_credit_secs, 0.0);
    // Contract: 0.4 x (~50 - 20) s, strictly less than today's.
    assert!(
        (contracted.effective_secs - 12.0).abs() < 0.5,
        "the contract excludes the credit: {}",
        contracted.effective_secs
    );
    assert!(contracted.effective_secs < today.effective_secs);
    assert_eq!(contracted.contract_credit_secs, 20.0);
    // A credit past the window leaves no budget: the ascent refuses, never
    // runs on a negative or wrapped window.
    let mut root = RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(50);
    assert!(super::alpha_opt::optimize_root_alpha(
        &f.net,
        &mut root,
        f.t,
        &f.adv,
        Some(deadline),
        cfg(500),
    )
    .is_none());
}

/// #router-r3 `rule.governor` (ROUTER_PROGRAM §5.2): the governed ascent
/// stops no later than the ungoverned one on the SAME deterministic
/// trajectory, keeps its BEST iterate, and certifies it exactly ONCE (one
/// published-frame guard evaluation of the committed alphas); and the
/// unarmed path (`governor: None` with an unarmed stop authority) is
/// bit-identical run to run. Engagement is asserted across the sweep so a
/// governor that never fires cannot pass silently.
#[test]
fn router_r3_governed_ascent_keeps_the_best_iterate_and_certifies_it_once() {
    let alpha_bits = |root: &RootGates| -> Vec<u64> {
        root.layers
            .iter()
            .flat_map(|l| l.alpha.iter().map(|a| a.to_bits()))
            .chain(root.head_alpha.iter().flatten().map(|a| a.to_bits()))
            .collect()
    };
    let mut engaged = 0usize;
    for seed in 0..10u64 {
        let f = spec_fixture(
            3300 + seed,
            0.4 + (seed % 4) as f64 * 0.2,
            0.35 + (seed % 3) as f64 * 0.1,
        );
        let cfg = |governor: Option<f64>| super::alpha_opt::AlphaOptCfg {
            iters: 30,
            secs: 120.0,
            guard: true,
            governor,
            ..super::alpha_opt::AlphaOptCfg::default()
        };
        let build =
            || RootGates::build(&f.net, &f.lo, &f.hi, RoundMode::Outward, None).expect("root");
        // Unarmed twice: today's ascent, bit for bit.
        let mut plain_a = build();
        let mut plain_b = build();
        let ra = super::alpha_opt::optimize_root_alpha(
            &f.net,
            &mut plain_a,
            f.t,
            &f.adv,
            None,
            cfg(None),
        );
        let rb = super::alpha_opt::optimize_root_alpha(
            &f.net,
            &mut plain_b,
            f.t,
            &f.adv,
            None,
            cfg(None),
        );
        assert_eq!(alpha_bits(&plain_a), alpha_bits(&plain_b), "seed {seed}");
        let (Some(ra), Some(rb)) = (ra, rb) else {
            continue;
        };
        assert_eq!(
            ra.iters, rb.iters,
            "seed {seed}: the unarmed ascent is deterministic"
        );
        assert_eq!(
            ra.governor_yield_iter, None,
            "seed {seed}: unarmed never yields"
        );
        // Governed.
        let mut governed = build();
        let rg = super::alpha_opt::optimize_root_alpha(
            &f.net,
            &mut governed,
            f.t,
            &f.adv,
            None,
            cfg(Some(0.15)),
        )
        .expect("the governed ascent runs wherever the plain one did");
        assert!(
            rg.iters <= ra.iters,
            "seed {seed}: a stop rule only stops EARLIER ({} > {})",
            rg.iters,
            ra.iters
        );
        assert!(rg.best >= rg.baseline, "seed {seed}: keep-best is monotone");
        assert!(
            rg.best <= ra.best,
            "seed {seed}: the governed run is a prefix of the same trajectory"
        );
        assert!(
            rg.guard_candidate_evals <= 1,
            "seed {seed}: the chosen iterate is certified ONCE ({} evaluations)",
            rg.guard_candidate_evals
        );
        if rg.accepted {
            let (before, after) = (
                rg.published_before.expect("guarded"),
                rg.published_after.expect("guarded"),
            );
            assert!(after > before, "seed {seed}: published {after} !> {before}");
        } else {
            assert_eq!(
                alpha_bits(&governed),
                alpha_bits(&build()),
                "seed {seed}: a rejected governed ascent leaves the heuristic alphas"
            );
        }
        if let Some(at) = rg.governor_yield_iter {
            assert_eq!(at, rg.iters, "seed {seed}: the yield ends the ascent");
            assert!(rg.governor_freed_secs >= 0.0);
            assert!(rg.governor_freed_secs <= rg.effective_secs + 1e-9);
            engaged += 1;
        }
    }
    assert!(
        engaged > 0,
        "the governor fired on no fixture: VOID, not a pass"
    );
}

/// #mr-errlane (margin-row throughput plan B3): envelope error-lane
/// dominance, exact-rational falsifier and off-is-bit-identical tests. A child
/// of this module so it reuses the fixtures and oracles above.
mod eel;

/// #mr-gpu-eval: the certified conv-backward offload — dark default, exact-
/// rational containment on random and real conv shapes, the routed-pass
/// falsifier, the beta Lagrangian oracle and fail-closed bit identity. A child
/// of this module so it reuses the fixtures above.
mod gpu_conv_tests;

/// #frozen-dual-state (V1 stage A): exact-rational and double-double
/// containment of the frozen alpha-zonotope query state, the certified node
/// bound (exact spot checks, the gamma mutation pin, sampled validity along
/// every split path), refusals and the stage-A probe's verdict inertness. A
/// child of this module so it reuses the fixtures above.
mod zono_state_tests;

/// #root-certify-once: the search-untrusted / certify-once root alpha — dark
/// default, the published-root-is-the-certified-one contract, sampled and
/// exact-rational containment, the falsified-property control, the certified
/// joint-refresh path, the ingest gate and the untrusted entries' refusals. A
/// child of this module so it reuses the fixtures above.
mod root_certify_once_tests;

/// #mr-ibp-seed: exact dyadic-rational containment of the IBP-seeded root boxes
/// on CNN7-shaped chains with BN folds, asymmetric boxes and active relus (with
/// the dropped-fold-charge mutation pin), the exact trunk-0 fold-ball hull,
/// shrink-only / crossing refusal, the `bi_cal` ordering mutation, dark byte
/// identity, the residual decline, never-verifies-a-false-bound and the typed
/// key's reachability. A child of this module so it reuses the fixtures above.
mod ibp_seed_tests;

/// #wave-beta-bab (ENGINE_PROGRAM M6): the wave tree on injected test
/// backends — default bit identity, containment of every certified leaf,
/// garbage / panicking / refusing backends, the claim-as-close, clip-side,
/// dropped-leaf and Err-as-close mutation pins, and thread-count
/// independence. A child of this module so it reuses the fixtures above.
mod wave_tests;

/// ENGINE_PROGRAM §1.7 B1 (proof bundle v2): hint capture is WRITE-ONLY.
/// The same instance, serial and parallel frontier, with capture off and on,
/// reaches the same verdict with the same statistics and the same token
/// margin bits; with capture on, an `unsat` streams nodes whose literals all
/// name Relu sites of the net.
#[test]
fn margin_row_verdicts_are_bit_identical_with_hint_capture_on_and_off() {
    use std::sync::{Arc, Mutex};

    use crate::certify::{
        force_hint_capture_for_test, with_thread_sink, AlphaRecord, LeafRecord, NodeRecord,
        ProofSink, SitesRecord, VerdictRecord,
    };

    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<NodeRecord>>>);
    impl ProofSink for Shared {
        fn leaf(&mut self, _leaf: &LeafRecord) {}
        fn node(&mut self, node: &NodeRecord) {
            if let Ok(mut v) = self.0.lock() {
                v.push(node.clone());
            }
        }
        fn alpha(&mut self, _alpha: &AlphaRecord) {}
        fn sites(&mut self, _sites: &SitesRecord) {}
        fn fd_state(&mut self, _state: &crate::certify::FdStateRecord) {}
        fn finish(&mut self, _verdict: &VerdictRecord) {}
    }

    let mut rng = StdRng::seed_from_u64(202);
    let spec = tiny_spec(&mut rng, 0.35);
    let net = TwinNet::compile(&spec).expect("compile");
    let lo = vec![0.01; 72];
    let hi = vec![0.02; 72];
    let (t, _worst, adv) = tiny_targets(&net, &lo, &hi);
    let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let relu_ops: Vec<u32> = net
        .ops
        .iter()
        .enumerate()
        .filter(|(_, op)| matches!(op, super::net::TwinOp::Relu { .. }))
        .map(|(k, _)| k as u32)
        .collect();
    let mut streamed_any = false;
    for frontier in [1usize, 8] {
        let cfg = || BabConfig {
            lru_cap: 64,
            frontier,
            ..BabConfig::default()
        };
        let run = |capture: bool| {
            let sink = Shared::default();
            force_hint_capture_for_test(Some(capture));
            let (out, _) = with_thread_sink(Box::new(sink.clone()), || {
                MarginRowBab::run(&net, &root, t, &adv, cfg())
            });
            force_hint_capture_for_test(None);
            let nodes = sink.0.lock().expect("lock").clone();
            (out, nodes)
        };
        let (off, off_nodes) = run(false);
        let (on, on_nodes) = run(true);
        assert!(off_nodes.is_empty(), "capture off streams no node");
        assert_eq!(verdict_bits(&off), verdict_bits(&on), "frontier {frontier}");
        let (so, sn) = (outcome_stats(&off), outcome_stats(&on));
        assert_eq!(
            (
                so.expansions,
                so.domains_created,
                so.closed,
                so.max_depth,
                so.mono_raw_dips
            ),
            (
                sn.expansions,
                sn.domains_created,
                sn.closed,
                sn.max_depth,
                sn.mono_raw_dips
            ),
            "frontier {frontier}: capture changed the search"
        );
        if let (MarginRowOutcome::Unsat(_, a), MarginRowOutcome::Unsat(_, b)) = (&off, &on) {
            assert_eq!(a.min_margin().to_bits(), b.min_margin().to_bits());
            assert!(
                !on_nodes.is_empty(),
                "an unsat under capture streams its nodes"
            );
            for n in &on_nodes {
                assert!(n.lits.iter().all(|l| relu_ops.contains(&l.site.op)));
                assert_eq!(n.beta.len(), n.lits.len());
            }
            streamed_any = true;
        }
    }
    assert!(
        streamed_any,
        "the fixture never proved unsat: the test is VOID"
    );
}

/// B2.3 (ENGINE_PROGRAM §1.7 B1): the Competition profile (hint capture OFF,
/// `NullSink`) and the Proof profile (capture ON, a recording sink) reach the
/// same verdict bits and the same search on EVERY margin-row route — the
/// classwise class trees, nested epoch subtrees, the frozen-dual tier (armed
/// by the harness field) and the joint tree — and Competition streams
/// nothing. Proof streams the routes' nodes: class trees' leaves from the
/// instance root (one class each), and a closed epoch subtree's nested nodes
/// under its parent leaf's prefix.
#[test]
fn margin_row_routes_are_bit_identical_in_both_profiles() {
    use std::sync::{Arc, Mutex};

    use crate::certify::{
        force_hint_capture_for_test, with_thread_sink, AlphaRecord, FdStateRecord, LeafRecord,
        NodeClose, NodeRecord, NullSink, ProofSink, RecordingSink, SitesRecord, VerdictRecord,
    };

    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<RecordingSink>>);
    impl ProofSink for Shared {
        fn leaf(&mut self, leaf: &LeafRecord) {
            self.0.lock().expect("lock").leaf(leaf);
        }
        fn node(&mut self, node: &NodeRecord) {
            self.0.lock().expect("lock").node(node);
        }
        fn alpha(&mut self, alpha: &AlphaRecord) {
            self.0.lock().expect("lock").alpha(alpha);
        }
        fn sites(&mut self, sites: &SitesRecord) {
            self.0.lock().expect("lock").sites(sites);
        }
        fn fd_state(&mut self, state: &FdStateRecord) {
            self.0.lock().expect("lock").fd_state(state);
        }
        fn finish(&mut self, verdict: &VerdictRecord) {
            self.0.lock().expect("lock").finish(verdict);
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Route {
        Joint,
        Classwise,
        Epoch,
        Fd,
    }
    let all = [Route::Joint, Route::Classwise, Route::Epoch, Route::Fd];
    let mut proved: Vec<Route> = Vec::new();
    'sweep: for seed in 0..48u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let spec = tiny_spec(&mut rng, 0.35);
        let Ok(net) = TwinNet::compile(&spec) else {
            continue;
        };
        for width in [0.01, 0.02, 0.04, 0.08, 0.16] {
            if all.iter().all(|r| proved.contains(r)) {
                break 'sweep;
            }
            let lo = vec![0.01; 72];
            let hi: Vec<f64> = lo.iter().map(|l| l + width).collect();
            let (t, _worst, adv) = tiny_targets(&net, &lo, &hi);
            let Ok(root) = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None) else {
                continue;
            };
            for route in all {
                let cfg = || BabConfig {
                    lru_cap: 64,
                    frontier: 1,
                    max_expansions: 400,
                    frozen_dual_tier: Some(route == Route::Fd),
                    epoch_policy: if route == Route::Epoch {
                        super::EpochRuntimePolicy {
                            depth: 1,
                            max_attempts: 16,
                            ..super::EpochRuntimePolicy::default()
                        }
                    } else {
                        super::EpochRuntimePolicy::default()
                    },
                    ..BabConfig::default()
                };
                let go = |capture: bool, sink: Box<dyn ProofSink>| {
                    force_hint_capture_for_test(Some(capture));
                    let (out, _) = with_thread_sink(sink, || {
                        if route == Route::Classwise {
                            MarginRowBab::run_classwise(&net, &root, t, &adv, cfg())
                        } else {
                            MarginRowBab::run(&net, &root, t, &adv, cfg())
                        }
                    });
                    force_hint_capture_for_test(None);
                    out
                };
                let competition = go(false, Box::new(NullSink));
                let sink = Shared::default();
                let proof = go(true, Box::new(sink.clone()));
                let rec = sink.0.lock().expect("lock").clone();
                assert_eq!(
                    verdict_bits(&competition),
                    verdict_bits(&proof),
                    "{route:?} seed {seed} width {width}: the profile moved the verdict"
                );
                let key = |o: &MarginRowOutcome| {
                    let s = match o {
                        MarginRowOutcome::Unsat(s, _) => Some(s),
                        MarginRowOutcome::Unknown { stats, .. } => stats.as_ref(),
                    };
                    s.map(|s| {
                        (
                            s.expansions,
                            s.domains_created,
                            s.closed,
                            s.max_depth,
                            s.epochs_closed,
                            s.root_closed_classes,
                        )
                    })
                };
                assert_eq!(
                    key(&competition),
                    key(&proof),
                    "{route:?}: the profile moved the search"
                );
                let MarginRowOutcome::Unsat(stats, _) = &proof else {
                    continue;
                };
                assert!(
                    !rec.nodes.is_empty(),
                    "{route:?}: a Proof unsat streams nodes"
                );
                match route {
                    Route::Classwise if stats.expansions > 0 => {
                        assert!(rec
                            .nodes
                            .iter()
                            .filter(|n| n.src == "tree" && n.close == NodeClose::Bound)
                            .all(|n| n.dis.as_ref().is_some_and(|d| d.len() == 1)));
                        proved.push(route);
                    }
                    Route::Epoch if stats.epochs_closed > 0 => {
                        let subtrees: Vec<&NodeRecord> = rec
                            .nodes
                            .iter()
                            .filter(|n| n.close == NodeClose::Subtree)
                            .collect();
                        assert!(!subtrees.is_empty(), "a closed epoch streams its leaf");
                        for s in &subtrees {
                            assert!(
                                rec.nodes.iter().any(|n| n.close != NodeClose::Subtree
                                    && n.lits.len() >= s.lits.len()
                                    && n.lits[..s.lits.len()] == s.lits[..]),
                                "a closed subtree streamed nothing under its prefix"
                            );
                        }
                        proved.push(route);
                    }
                    Route::Fd if !rec.fd_states.is_empty() => {
                        for st in &rec.fd_states {
                            assert!(rec.nodes.iter().any(|n| n.src == "fd_prune"
                                && n.dis.as_deref() == Some(&[st.class][..])));
                        }
                        proved.push(route);
                    }
                    Route::Joint if stats.expansions > 0 => proved.push(route),
                    _ => {}
                }
            }
        }
    }
    for route in all {
        assert!(
            proved.contains(&route),
            "no fixture exercised the {route:?} route under Proof: VOID ({proved:?})"
        );
    }
}

// Parent refresh clocks also cover a valid split with no downstream work.
fn parent_refresh_deadline_fixture() -> (TwinNet, RootGates) {
    let spec = TwinSpec {
        n_in: 1,
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
                ishape: (1, 1, 1),
                oshape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 1 },
            TwinOpSpec::ChannelAffine {
                input: 2,
                scale: vec![1.0],
                shift: vec![-0.25],
                shift_err: vec![0.0],
                scale_rel_err: 0.0,
                shape: (1, 1, 1),
            },
            TwinOpSpec::Relu { input: 3 },
            TwinOpSpec::Flatten { input: 4 },
            TwinOpSpec::Gemm {
                input: 5,
                weight: vec![0.25],
                bias: vec![0.5],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (1, 1),
            },
            TwinOpSpec::Relu { input: 6 },
            TwinOpSpec::Gemm {
                input: 7,
                weight: vec![0.5, -0.25],
                bias: vec![0.0, 0.0],
                bias_err: vec![],
                weight_rel_err: 0.0,
                shape: (2, 1),
            },
        ],
    };
    let net = TwinNet::compile(&spec).expect("original two-layer model");
    let root = RootGates::build(&net, &[-1.0], &[1.0], RoundMode::Outward, None)
        .expect("original outward root");
    assert_eq!(root.layers.len(), 2);
    assert_eq!(root.layers[0].unst, [0]);
    assert_eq!(root.layers[1].unst, [0]);
    (net, root)
}

#[test]
fn epoch_parent_frame_refresh_expired_request_refuses_even_without_downstream_layers() {
    let _env = crate::tests::lock_env_shared();
    let (net, root) = parent_refresh_deadline_fixture();
    let last = root.layers.len() - 1;
    // Splitting the final trunk layer makes the downstream loop empty. This
    // remains a genuine valid split rather than a malformed-input refusal.
    assert_eq!(last + 1, root.layers.len());
    for layer in [0, last] {
        for direction in [-1i8, 1i8] {
            let expired = std::time::Instant::now()
                .checked_sub(std::time::Duration::from_millis(1))
                .expect("test clock supports an expired deadline");
            let error = super::backward_interm::refresh_parent_frame(
                &net,
                &root,
                &[(layer, 0, direction)],
                1,
                1,
                Some(expired),
            )
            .err()
            .expect("an expired original request cannot return a refreshed frame");
            assert!(matches!(error, ny_core::NyError::DeadlineExceeded(_)));
        }
    }
}

#[test]
fn epoch_parent_frame_refresh_live_empty_path_preserves_original_gate_bits() {
    let _env = crate::tests::lock_env_shared();
    let (net, root) = parent_refresh_deadline_fixture();
    let last = root.layers.len() - 1;
    for direction in [-1i8, 1i8] {
        let split = [(last, 0, direction)];
        let (plain, plain_stats) =
            super::backward_interm::refresh_parent_frame(&net, &root, &split, 1, 1, None)
                .expect("original no-deadline empty path");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        let (timed, timed_stats) =
            super::backward_interm::refresh_parent_frame(&net, &root, &split, 1, 1, Some(deadline))
                .expect("live original deadline empty path");
        assert_eq!(plain_stats.baked, 1);
        assert_eq!(timed_stats.baked, 1);
        assert_eq!(plain_stats.layers, 0);
        assert_eq!(timed_stats.layers, 0);
        assert_eq!(plain_stats.columns, 0);
        assert_eq!(timed_stats.columns, 0);
        assert_eq!(plain_stats.tightened, 0);
        assert_eq!(timed_stats.tightened, 0);
        assert_eq!(plain.layers.len(), timed.layers.len());
        for (left, right) in plain.layers.iter().zip(&timed.layers) {
            assert_eq!(left.op, right.op);
            assert_eq!(left.n, right.n);
            assert_eq!(left.unst, right.unst);
            for (name, a, b) in [
                ("l", &left.l, &right.l),
                ("u", &left.u, &right.u),
                ("alpha", &left.alpha, &right.alpha),
                ("s", &left.s, &right.s),
                ("c", &left.c, &right.c),
                ("ms", &left.ms, &right.ms),
            ] {
                assert!(
                    a.iter()
                        .map(|v| v.to_bits())
                        .eq(b.iter().map(|v| v.to_bits())),
                    "a live deadline changed {name}"
                );
            }
        }
    }
}

mod identity_conv_tests;
