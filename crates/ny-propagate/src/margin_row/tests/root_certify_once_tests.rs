// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #root-certify-once soundness, contract and dark-default tests.
//!
//! * `root_certify_once_is_dark_by_default_and_bit_identical` — the default
//!   policy is disarmed, the lane entry returns `None` and the root is
//!   bit-identical.
//! * `root_certify_once_publishes_only_the_certified_root` — THE CONTRACT: the
//!   bound the phase reports as published is the bit-exact certified
//!   `root_eval` of the root it left (recomputed on a memo-free copy), it is one
//!   of the CERTIFIED numbers (incumbent / alpha candidate / refreshed
//!   candidate) and never the untrusted one, it is never below the certified
//!   incumbent, a kept incumbent is bit-identical, and an accepted alpha only
//!   moved unstable slopes inside `[0, 1]` with `ms = max(alpha, s)`.
//!   Engagement (a candidate strictly beating the incumbent) is asserted across
//!   the sweep, so a phase that never accepts cannot pass silently.
//! * `root_certify_once_never_verifies_a_false_bound` — on the runner-up target
//!   (falsified at the midpoint) the published bound stays negative, and on
//!   every target it stays at or below every sampled margin.
//! * `root_certify_once_published_bound_is_below_the_exact_rational_margin` —
//!   the per-class certified `root_eval` of the root the phase left lower-bounds
//!   the EXACT (`BigRational`) margin at the midpoint, the two box corners and
//!   random corners / interior points.
//! * `root_certify_once_joint_refresh_path_is_certified_shrink_only_and_sound`
//!   — with untrusted joint rounds on, the certified refresh engages; a kept
//!   refreshed root is shrink-only against the incumbent, contains every sampled
//!   pre-activation, keeps `ms = max(alpha, s)` and lower-bounds every sampled
//!   margin.
//! * `untrusted_entries_refuse_certified_roots` — the untrusted ascent and the
//!   untrusted refresh refuse an Outward root, the certified refresh refuses the
//!   Parity scratch, and the scratch is Parity with no clip rows.
//! * `ingest_gate_clamps_and_refuses_non_finite_proposals` — the install gate.
//! * `root_certify_once_refuses_loudly_and_leaves_the_root_bit_identical` — a
//!   Parity incumbent, no adv class, a zero wall request; the terminal line's
//!   fixed schema.

use super::*;

use num_rational::BigRational;

use crate::margin_row::alpha_opt;
use crate::margin_row::backward_interm::window_tests::chain_spec;
use crate::margin_row::backward_interm::{
    refresh_root_intermediates, refresh_untrusted_intermediates,
};
use crate::margin_row::rebake::certified_root_bound;
use crate::margin_row::root::LayerGates;
use crate::margin_row::root_certify_once::{
    certify_once_root, install_candidate_alpha, maybe_certify_once_root, untrusted_scratch, Kept,
    RootCertifyOnceReport,
};
use crate::margin_row::{
    BackwardIntermRuntimePolicy, RootAlphaRuntimePolicy, RootCertifyOnceRuntimePolicy, RootSearch,
};

// ---------------------------------------------------------------------------
//  Fixtures
// ---------------------------------------------------------------------------

/// A tiny twin-wall net with its spec (the exact forward needs the spec).
struct Fx {
    spec: TwinSpec,
    net: TwinNet,
    lo: Vec<f64>,
    hi: Vec<f64>,
}

fn tiny_fixture(seed: u64, scale: f64, half: f64) -> Fx {
    let mut rng = StdRng::seed_from_u64(seed);
    let spec = tiny_spec(&mut rng, scale);
    let net = TwinNet::compile(&spec).expect("compile");
    Fx {
        spec,
        net,
        lo: vec![-half; 72],
        hi: vec![half; 72],
    }
}

/// The sweep the head-alpha engagement test uses: varied scale and box width.
fn sweep() -> Vec<Fx> {
    (0..8u64)
        .map(|seed| {
            tiny_fixture(
                0x0C0E_0000 + seed,
                0.4 + (seed % 4) as f64 * 0.2,
                0.35 + (seed % 3) as f64 * 0.1,
            )
        })
        .collect()
}

/// Armed spec-phase-only policy with a generous wall (no deadline in tests).
fn armed(iters: usize, refresh: usize) -> RootCertifyOnceRuntimePolicy {
    RootCertifyOnceRuntimePolicy {
        enabled: true,
        iters,
        refresh,
        secs: 600.0,
        search: RootSearch::Ascent,
    }
}

fn bi_policy(enabled: bool) -> BackwardIntermRuntimePolicy {
    BackwardIntermRuntimePolicy {
        enabled,
        secs: 1_000.0,
        chunk: 16,
        chunk_secs: 0.0,
        topk: 4_096,
        interm_alpha: 0,
        window: false,
        // #bi-gpu (merged into int/wave7 alongside this file) added the field;
        // the fixture predates it. Dark: the CPU engine, as before.
        gpu: false,
        // #bi-patch: dark, the historical dense pairs.
        patch: crate::margin_row::BiPatchRuntimePolicy::default(),
    }
}

/// Run the armed phase on a FRESH certified root for property `(t, adv)`.
fn run_phase(
    fx: &Fx,
    t: usize,
    adv: &[usize],
    policy: RootCertifyOnceRuntimePolicy,
    bi: BackwardIntermRuntimePolicy,
) -> (RootGates, RootGates, RootCertifyOnceReport) {
    let before = RootGates::build(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None).expect("root");
    let mut root =
        RootGates::build(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None).expect("root");
    let report = certify_once_root(
        &fx.net,
        &mut root,
        false,
        t,
        adv,
        None,
        None,
        policy,
        RootAlphaRuntimePolicy::default(),
        bi,
    );
    (before, root, report)
}

/// A memo-free copy of a certified root: evaluating it is a genuine
/// recomputation, never a stored value.
fn fresh_copy(root: &RootGates) -> RootGates {
    assert!(
        root.spec_alpha_policy.is_none(),
        "fixture roots carry no #spec-alpha"
    );
    RootGates {
        mode: root.mode,
        mid: root.mid.clone(),
        rad: root.rad.clone(),
        xabs: root.xabs.clone(),
        layers: root
            .layers
            .iter()
            .map(|layer| LayerGates {
                op: layer.op,
                n: layer.n,
                l: layer.l.clone(),
                u: layer.u.clone(),
                alpha: layer.alpha.clone(),
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms: layer.ms.clone(),
                unst: layer.unst.clone(),
                clip_rows: layer.clip_rows.clone(),
            })
            .collect(),
        lo: root.lo.clone(),
        hi: root.hi.clone(),
        backward_interm_execution: root.backward_interm_execution,
        root_precision_policy: root.root_precision_policy,
        root_precision_execution: root.root_precision_execution,
        spec_alpha_policy: None,
        head_alpha: root.head_alpha.clone(),
        root_eval_memo: crate::margin_row::bab::RootEvalMemo::default(),
        head_group_source: None,
    }
}

type GateBits = Vec<(
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<u64>,
    Vec<usize>,
)>;

fn gate_bits(root: &RootGates) -> (GateBits, Option<Vec<u64>>, bool) {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    (
        root.layers
            .iter()
            .map(|l| {
                (
                    bits(&l.l),
                    bits(&l.u),
                    bits(&l.alpha),
                    bits(&l.s),
                    bits(&l.c),
                    bits(&l.ms),
                    l.unst.clone(),
                )
            })
            .collect(),
        root.head_alpha.as_deref().map(bits),
        root.mode.outward(),
    )
}

/// Sampled points: the midpoint, both box corners, random corners and random
/// interior points (column-major, `(n_in, n)`).
fn points(root: &RootGates, lo: &[f64], hi: &[f64], n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_in = root.mid.len();
    let mut x = Array2::<f64>::zeros((n_in, n.max(3)));
    for i in 0..n_in {
        x[[i, 0]] = f64::midpoint(lo[i], hi[i]);
        x[[i, 1]] = lo[i];
        x[[i, 2]] = hi[i];
        for b in 3..x.ncols() {
            let u: f64 = if b % 2 == 0 {
                if rng.random_range(0.0..1.0) < 0.5 {
                    0.0
                } else {
                    1.0
                }
            } else {
                rng.random_range(0.0..1.0)
            };
            x[[i, b]] = lo[i] + u * (hi[i] - lo[i]);
        }
    }
    x
}

/// Minimum over sampled points and adversarial classes of `Y_t - Y_j` (f64).
fn sampled_min_margin(net: &TwinNet, x: &Array2<f64>, t: usize, adv: &[usize]) -> f64 {
    let (y, _) = net
        .forward_points(x, &std::collections::BTreeMap::new())
        .expect("forward");
    margins_at(net, &y, t, adv)
        .iter()
        .flat_map(|row| row.iter().copied())
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
//  Exact dyadic-rational forward (the #mr-errlane falsifier's oracle)
// ---------------------------------------------------------------------------

fn q(v: f64) -> BigRational {
    BigRational::from_float(v).expect("finite operand")
}

/// Class scores with NO rounding at all: every f64 is a dyadic rational and
/// ReLU is an exact comparison.
fn exact_scores(spec: &TwinSpec, x: &[f64]) -> Vec<BigRational> {
    let zero = q(0.0);
    let mut t: Vec<Vec<BigRational>> = Vec::with_capacity(spec.ops.len() + 1);
    t.push(x.iter().map(|&v| q(v)).collect());
    for op in &spec.ops {
        let out: Vec<BigRational> = match op {
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
                let (_, ih, iw) = *ishape;
                let (_, oh, ow) = *oshape;
                let mut out = Vec::with_capacity(co * oh * ow);
                for oc in 0..co {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let mut acc = q(bias[oc]);
                            for c in 0..ci {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iy = (oy * stride.0 + ky) as isize - pads.0 as isize;
                                        let ix = (ox * stride.1 + kx) as isize - pads.1 as isize;
                                        if iy >= 0
                                            && (iy as usize) < ih
                                            && ix >= 0
                                            && (ix as usize) < iw
                                        {
                                            let w = weight[((oc * ci + c) * kh + ky) * kw + kx];
                                            if w != 0.0 {
                                                let s = &src[c * ih * iw
                                                    + (iy as usize) * iw
                                                    + ix as usize];
                                                acc += q(w) * s;
                                            }
                                        }
                                    }
                                }
                            }
                            out.push(acc);
                        }
                    }
                }
                out
            }
            TwinOpSpec::ConvTranspose { .. } => {
                panic!("the certify-once fixtures carry no transposed conv")
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
                    .map(|(i, v)| q(shift[i / hw]) + q(scale[i / hw]) * v)
                    .collect()
            }
            TwinOpSpec::Relu { input } => t[*input]
                .iter()
                .map(|v| if *v > zero { v.clone() } else { zero.clone() })
                .collect(),
            TwinOpSpec::Add { lhs, rhs } => {
                t[*lhs].iter().zip(&t[*rhs]).map(|(a, b)| a + b).collect()
            }
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
                (0..no)
                    .map(|o| {
                        let mut acc = q(bias[o]);
                        for (i, s) in src.iter().enumerate().take(ni) {
                            let w = weight[o * ni + i];
                            if w != 0.0 {
                                acc += q(w) * s;
                            }
                        }
                        acc
                    })
                    .collect()
            }
        };
        t.push(out);
    }
    t.pop().expect("final tensor")
}

// ---------------------------------------------------------------------------
//  Dark by default
// ---------------------------------------------------------------------------

#[test]
fn root_certify_once_is_dark_by_default_and_bit_identical() {
    let policy = RootCertifyOnceRuntimePolicy::default();
    assert!(!policy.enabled, "#root-certify-once must ship disarmed");
    let fx = tiny_fixture(0x0C0E_0101, 0.6, 0.4);
    let (t, _, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
    let mut root =
        RootGates::build(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None).expect("root");
    let before = gate_bits(&root);
    for disarmed in [
        policy,
        // Every other field set to something armed-looking: only `enabled`
        // may open the phase.
        RootCertifyOnceRuntimePolicy {
            enabled: false,
            iters: 64,
            refresh: 8,
            secs: 1_000.0,
            search: RootSearch::JointF32,
        },
    ] {
        assert!(maybe_certify_once_root(
            &fx.net,
            &mut root,
            false,
            t,
            &adv,
            None,
            Some(-1.0),
            disarmed,
            RootAlphaRuntimePolicy::default(),
            bi_policy(true),
        )
        .is_none());
        assert!(
            gate_bits(&root) == before,
            "the dark entry must not touch the root"
        );
    }
}

// ---------------------------------------------------------------------------
//  The contract: only the certified root is ever published
// ---------------------------------------------------------------------------

#[test]
fn root_certify_once_publishes_only_the_certified_root() {
    let mut accepted = 0usize;
    let mut runs = 0usize;
    for (k, fx) in sweep().iter().enumerate() {
        let (t, worst, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
        let adv_worst: Vec<usize> = (0..fx.net.n_out).filter(|&o| o != worst).collect();
        for (target, classes) in [(t, adv), (worst, adv_worst)] {
            let (before, root, report) = run_phase(
                fx,
                target,
                &classes,
                armed(8, 0),
                BackwardIntermRuntimePolicy::default(),
            );
            runs += 1;
            assert!(
                root.mode.outward(),
                "fixture {k}: the published root must stay certified"
            );
            assert_ne!(
                report.kept,
                Kept::None,
                "fixture {k}: an armed run on a valid root must score the incumbent ({})",
                report.stop
            );
            assert!(report.incumbent.is_finite());
            // Never worse than today's root.
            assert!(
                report.published >= report.incumbent,
                "fixture {k}: published {} below the certified incumbent {}",
                report.published,
                report.incumbent
            );
            // The published number is the certified evaluation of exactly the
            // root the phase left — recomputed with no memo in the way.
            let recomputed =
                certified_root_bound(&fx.net, &fresh_copy(&root), target, &classes).expect("eval");
            assert_eq!(
                recomputed.to_bits(),
                report.published.to_bits(),
                "fixture {k}: published {} is not the certified root_eval {recomputed} of the \
                 root the phase left",
                report.published
            );
            // ... and it is one of the CERTIFIED numbers the phase computed.
            let certified_set = [
                report.incumbent,
                report.certified_alpha,
                report.certified_refresh,
            ];
            assert!(
                certified_set
                    .iter()
                    .any(|v| v.to_bits() == report.published.to_bits()),
                "fixture {k}: published {} is none of the certified numbers {certified_set:?}",
                report.published
            );
            match report.kept {
                Kept::Incumbent => {
                    assert!(
                        gate_bits(&root) == gate_bits(&before),
                        "fixture {k}: a kept incumbent must be bit-identical to the entry root"
                    );
                    assert_eq!(report.published.to_bits(), report.incumbent.to_bits());
                }
                Kept::Alpha => {
                    accepted += 1;
                    assert!(report.certified_alpha > report.incumbent);
                    assert_eq!(report.published.to_bits(), report.certified_alpha.to_bits());
                    assert!(report.moved > 0, "fixture {k}: accepted with nothing moved");
                    for (li, (old, new)) in before.layers.iter().zip(&root.layers).enumerate() {
                        assert_eq!(old.l, new.l, "fixture {k}: l moved at layer {li}");
                        assert_eq!(old.u, new.u, "fixture {k}: u moved at layer {li}");
                        assert_eq!(old.s, new.s, "fixture {k}: s moved at layer {li}");
                        assert_eq!(old.c, new.c, "fixture {k}: c moved at layer {li}");
                        assert_eq!(old.unst, new.unst, "fixture {k}: unst moved at {li}");
                        let unstable: std::collections::BTreeSet<usize> =
                            old.unst.iter().copied().collect();
                        for j in 0..new.n {
                            assert!((0.0..=1.0).contains(&new.alpha[j]));
                            assert_eq!(
                                new.ms[j],
                                new.alpha[j].max(new.s[j]),
                                "fixture {k}: ms != max(alpha, s) at {li}/{j}"
                            );
                            if !unstable.contains(&j) {
                                assert_eq!(
                                    new.alpha[j].to_bits(),
                                    old.alpha[j].to_bits(),
                                    "fixture {k}: a stable slope moved at {li}/{j}"
                                );
                            }
                        }
                    }
                    if let Some(head) = root.head_alpha.as_ref() {
                        assert!(head.iter().all(|a| (0.0..=1.0).contains(a)));
                    }
                }
                Kept::Refresh => panic!("fixture {k}: no joint round ran, yet a refresh was kept"),
                Kept::None => unreachable!("asserted above"),
            }
            // The untrusted value never reaches the published bound: when the
            // two differ, the published one is the certified recomputation.
            if report.untrusted_best.is_finite()
                && report.untrusted_best.to_bits() != report.published.to_bits()
            {
                assert_eq!(recomputed.to_bits(), report.published.to_bits());
            }
            let line = report.line();
            assert!(line.starts_with("[root-certify-once] armed=1 iters="));
            for field in [
                " untrusted_best=",
                " certified=",
                " kept=",
                " published=",
                " stop=",
            ] {
                assert!(line.contains(field), "terminal line lacks {field}: {line}");
            }
        }
    }
    eprintln!("[root-certify-once-test] contract: runs={runs} accepted={accepted}");
    assert!(
        accepted > 0,
        "no certified candidate beat its incumbent on any of {runs} runs: the accept path is VOID"
    );
}

// ---------------------------------------------------------------------------
//  Soundness: sampled containment and the falsified property
// ---------------------------------------------------------------------------

#[test]
fn root_certify_once_never_verifies_a_false_bound() {
    let mut falsified = 0usize;
    // Candidates that actually REPLACED the incumbent. Without this count the
    // containment checks below could pass on incumbents alone — today's root,
    // which says nothing about the phase. `iters = 8` is exactly the contract
    // sweep's configuration (so this guard is implied by that test's own
    // engagement guard); `12` adds the longer ascent.
    let mut accepted = 0usize;
    for (k, fx) in sweep().iter().enumerate() {
        let (t, worst, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
        let adv_worst: Vec<usize> = (0..fx.net.n_out).filter(|&o| o != worst).collect();
        for (iters, (target, classes)) in [8usize, 12].into_iter().flat_map(|iters| {
            [(worst, adv_worst.clone()), (t, adv.clone())]
                .into_iter()
                .map(move |pair| (iters, pair))
        }) {
            let (_, root, report) = run_phase(
                fx,
                target,
                &classes,
                armed(iters, 0),
                BackwardIntermRuntimePolicy::default(),
            );
            if matches!(report.kept, Kept::Alpha | Kept::Refresh) {
                accepted += 1;
            }
            let x = points(&root, &fx.lo, &fx.hi, 256, 0x0C0E_1000 ^ k as u64);
            let truth = sampled_min_margin(&fx.net, &x, target, &classes);
            assert!(
                report.published <= truth + 1e-9,
                "fixture {k} t={target}: certified {} ABOVE the sampled margin {truth}",
                report.published
            );
            if target == worst {
                assert!(
                    truth < 0.0,
                    "fixture {k}: the runner-up property must be falsified at the midpoint"
                );
                assert!(
                    report.published < 0.0,
                    "fixture {k}: a FALSE property must never certify (published {})",
                    report.published
                );
                falsified += 1;
            }
        }
    }
    eprintln!("[root-certify-once-test] false-bound: falsified={falsified} accepted={accepted}");
    assert!(falsified > 0, "no falsified property was exercised: VOID");
    assert!(
        accepted > 0,
        "no candidate replaced its incumbent: the containment checks only saw today's root \
         (VOID for the phase)"
    );
}

#[test]
fn root_certify_once_published_bound_is_below_the_exact_rational_margin() {
    let mut checks = 0usize;
    let mut accepted = 0usize;
    let mut min_slack = f64::INFINITY;
    // The WHOLE contract sweep at the contract's configuration, so this test's
    // engagement guard is implied by that test's (a `take(6)` prefix could miss
    // every accepting fixture and go VOID-red on a sound phase).
    for (k, fx) in sweep().iter().enumerate() {
        let (t, worst, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
        let adv_worst: Vec<usize> = (0..fx.net.n_out).filter(|&o| o != worst).collect();
        for (target, classes) in [(t, adv), (worst, adv_worst)] {
            let (_, root, report) = run_phase(
                fx,
                target,
                &classes,
                armed(8, 0),
                BackwardIntermRuntimePolicy::default(),
            );
            // Only a candidate that REPLACED the incumbent counts; a refusal
            // (`Kept::None`) is not an acceptance.
            if matches!(report.kept, Kept::Alpha | Kept::Refresh) {
                accepted += 1;
            }
            // Per-class certified bounds of the root the phase LEFT.
            let eng = BackwardEngine::new(&fx.net, &root);
            let re = crate::margin_row::bab::root_eval(&eng, &fx.net, target, &classes)
                .expect("root_eval");
            let published = re.dj.iter().copied().fold(f64::INFINITY, f64::min);
            assert_eq!(
                published.to_bits(),
                report.published.to_bits(),
                "fixture {k}: the per-class evaluation disagrees with the published bound"
            );
            let x = points(&root, &fx.lo, &fx.hi, 10, 0x0C0E_2000 ^ k as u64);
            for b in 0..x.ncols() {
                let xb: Vec<f64> = (0..x.nrows()).map(|i| x[[i, b]]).collect();
                let scores = exact_scores(&fx.spec, &xb);
                for (ki, &j) in classes.iter().enumerate() {
                    let exact = &scores[target] - &scores[j];
                    assert!(
                        q(re.dj[ki]) <= exact,
                        "FALSIFIED: fixture {k} t={target} class {j} point {b}: certified {:e} \
                         exceeds the EXACT margin",
                        re.dj[ki]
                    );
                    let slack = num_traits::ToPrimitive::to_f64(&(&exact - q(re.dj[ki])))
                        .unwrap_or(f64::NAN);
                    min_slack = min_slack.min(slack);
                    checks += 1;
                }
            }
        }
    }
    eprintln!(
        "[root-certify-once-test] exact-rational: checks={checks} accepted={accepted} \
         min(exact - certified)={min_slack:.3e}"
    );
    assert!(checks > 100, "only {checks} exact checks ran");
    assert!(
        accepted > 0,
        "no candidate was accepted: the exact check only saw incumbents (VOID for the phase)"
    );
}

// ---------------------------------------------------------------------------
//  The joint (refresh) path
// ---------------------------------------------------------------------------

#[test]
fn root_certify_once_joint_refresh_path_is_certified_shrink_only_and_sound() {
    let mut refresh_certified = 0usize;
    let mut refresh_kept = 0usize;
    let mut rounds_run = 0usize;
    for (k, seed) in [0x0C0E_3001u64, 0x0C0E_3002, 0x0C0E_3003]
        .iter()
        .enumerate()
    {
        let mut rng = StdRng::seed_from_u64(*seed);
        let spec = chain_spec(&mut rng, 0.6);
        let net = TwinNet::compile(&spec).expect("compile");
        let fx = Fx {
            spec,
            net,
            lo: vec![-0.4; 128],
            hi: vec![0.35; 128],
        };
        let (t, worst, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
        let adv_worst: Vec<usize> = (0..fx.net.n_out).filter(|&o| o != worst).collect();
        for (target, classes) in [(t, adv), (worst, adv_worst)] {
            let (before, root, report) =
                run_phase(&fx, target, &classes, armed(6, 3), bi_policy(true));
            rounds_run += report.refresh_rounds;
            if report.certified_refresh.is_finite() {
                refresh_certified += 1;
            }
            if report.kept == Kept::Refresh {
                refresh_kept += 1;
                assert_eq!(
                    report.published.to_bits(),
                    report.certified_refresh.to_bits(),
                    "fixture {k}: a kept refresh must publish its own certified evaluation"
                );
                assert!(
                    report.certified_refresh > report.incumbent,
                    "fixture {k}: a kept refresh must strictly beat the incumbent"
                );
            }
            assert!(report.published >= report.incumbent);
            let recomputed =
                certified_root_bound(&fx.net, &fresh_copy(&root), target, &classes).expect("eval");
            assert_eq!(
                recomputed.to_bits(),
                report.published.to_bits(),
                "fixture {k}: published is not the certified evaluation of the kept root"
            );
            let x = points(&root, &fx.lo, &fx.hi, 192, *seed ^ 0xBEEF);
            let sel: std::collections::BTreeMap<usize, Vec<usize>> = root
                .layers
                .iter()
                .map(|layer| (layer.op, (0..layer.n).collect()))
                .collect();
            let (_, pre) = fx.net.forward_points(&x, &sel).expect("forward");
            for (li, (old, new)) in before.layers.iter().zip(&root.layers).enumerate() {
                let p = &pre[&new.op];
                for j in 0..new.n {
                    assert!(
                        new.l[j] >= old.l[j] && new.u[j] <= old.u[j],
                        "fixture {k} ({li},{j}): the kept root WIDENED a box"
                    );
                    for b in 0..x.ncols() {
                        let v = p[[j, b]];
                        assert!(
                            v >= new.l[j] - 1e-9 && v <= new.u[j] + 1e-9,
                            "fixture {k} ({li},{j}): sampled {v} outside the kept box"
                        );
                    }
                    assert!((0.0..=1.0).contains(&new.alpha[j]));
                    assert_eq!(new.ms[j], new.alpha[j].max(new.s[j]), "ms != max(alpha, s)");
                }
            }
            let truth = sampled_min_margin(&fx.net, &x, target, &classes);
            assert!(
                report.published <= truth + 1e-9,
                "fixture {k}: certified {} ABOVE the sampled margin {truth}",
                report.published
            );
            if target == worst && truth < 0.0 {
                assert!(
                    report.published < 0.0,
                    "a false property must never certify"
                );
            }
        }
    }
    assert!(
        rounds_run > 0,
        "no untrusted joint round ran on any fixture: VOID"
    );
    eprintln!(
        "[root-certify-once-test] joint: rounds={rounds_run} refresh_certified={refresh_certified} \
         refresh_kept={refresh_kept}"
    );
    assert!(
        refresh_certified > 0,
        "the certified refresh never engaged on any fixture: the refresh path is VOID"
    );
    // The shrink-only / containment checks above are about the REFRESHED boxes
    // only when a refreshed root was actually kept; otherwise they re-check the
    // build's own boxes and prove nothing about this phase.
    assert!(
        refresh_kept > 0,
        "no refreshed root was ever kept: the box containment checks only saw unrefreshed \
         frames (VOID for the refresh path)"
    );
}

/// Tier-0 retained rows forbid the certified refresh, so the untrusted joint
/// rounds are refused up front (they could only steer alpha toward boxes that
/// are never certified) and the spec phase alone runs; the published root is
/// still a certified evaluation never below the incumbent, with its boxes
/// untouched.
#[test]
fn retained_rows_refuse_the_joint_phase() {
    let mut rng = StdRng::seed_from_u64(0x0C0E_3001);
    let spec = chain_spec(&mut rng, 0.6);
    let net = TwinNet::compile(&spec).expect("compile");
    let (lo, hi) = (vec![-0.4; 128], vec![0.35; 128]);
    let (_, worst, _) = tiny_targets(&net, &lo, &hi);
    let classes: Vec<usize> = (0..net.n_out).filter(|&o| o != worst).collect();
    let before = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let mut root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
    let report = certify_once_root(
        &net,
        &mut root,
        true,
        worst,
        &classes,
        None,
        None,
        armed(6, 3),
        RootAlphaRuntimePolicy::default(),
        bi_policy(true),
    );
    assert_ne!(report.kept, Kept::None, "stop={}", report.stop);
    assert_eq!(
        report.refresh_rounds, 0,
        "a joint round ran under retained rows"
    );
    assert_eq!(report.refresh_tightened, 0);
    assert!(report.certified_refresh.is_nan());
    assert_ne!(report.kept, Kept::Refresh);
    assert!(report.published >= report.incumbent);
    for (li, (old, new)) in before.layers.iter().zip(&root.layers).enumerate() {
        assert_eq!(old.l, new.l, "l moved at layer {li}");
        assert_eq!(old.u, new.u, "u moved at layer {li}");
        assert_eq!(old.unst, new.unst, "unst moved at layer {li}");
    }
    let recomputed = certified_root_bound(&net, &fresh_copy(&root), worst, &classes).expect("eval");
    assert_eq!(recomputed.to_bits(), report.published.to_bits());
}

// ---------------------------------------------------------------------------
//  Refusals of the untrusted entries
// ---------------------------------------------------------------------------

#[test]
fn untrusted_entries_refuse_certified_roots() {
    let fx = tiny_fixture(0x0C0E_4001, 0.6, 0.4);
    let (t, _, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
    let mut root =
        RootGates::build(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None).expect("root");
    let before = gate_bits(&root);
    assert!(
        alpha_opt::untrusted_ascent(&fx.net, &mut root, t, &adv, 60.0, 8, 0.1, 8).is_none(),
        "the untrusted ascent must refuse a certified root"
    );
    assert!(
        gate_bits(&root) == before,
        "a refused untrusted ascent must not touch the certified root"
    );
    assert!(
        refresh_untrusted_intermediates(&fx.net, &root, 4_096, 16, false, None).is_err(),
        "the untrusted refresh must refuse a certified root"
    );
    let scratch = untrusted_scratch(&root);
    assert!(!scratch.mode.outward(), "the scratch must be Parity");
    assert!(scratch.layers.iter().all(|l| l.clip_rows.is_none()));
    assert!(scratch.spec_alpha_policy.is_none());
    assert!(
        refresh_root_intermediates(&fx.net, &scratch, 4_096, 16, false, None, None).is_err(),
        "the certified refresh must refuse the Parity scratch"
    );
    // The scratch is a copy: working on it leaves the certified root alone.
    let mut scratch = scratch;
    let _ = alpha_opt::untrusted_ascent(&fx.net, &mut scratch, t, &adv, 60.0, 4, 0.1, 8);
    assert_eq!(gate_bits(&root), before);
}

// ---------------------------------------------------------------------------
//  The ingest gate
// ---------------------------------------------------------------------------

#[test]
fn ingest_gate_clamps_and_refuses_non_finite_proposals() {
    let fx = tiny_fixture(0x0C0E_5001, 0.6, 0.45);
    let mut root =
        RootGates::build(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None).expect("root");
    let before = gate_bits(&root);
    let (li, unst) = root
        .layers
        .iter()
        .enumerate()
        .map(|(li, l)| (li, l.unst.clone()))
        .find(|(_, u)| u.len() >= 5)
        .expect("a layer with five unstable neurons");
    let stable = (0..root.layers[li].n).find(|j| !unst.contains(j));
    let mut scratch = untrusted_scratch(&root);
    let proposals = [f64::NAN, f64::INFINITY, 1.5, -0.25, 0.3];
    for (&j, &a) in unst.iter().zip(&proposals) {
        scratch.layers[li].alpha[j] = a;
    }
    // A stable neuron's proposal must be ignored whatever it says.
    if let Some(j) = stable {
        scratch.layers[li].alpha[j] = 0.5;
    }
    // A head vector with one non-finite entry is refused wholesale.
    let mut bad_head = vec![0.25; fx.net.n_y];
    bad_head[0] = f64::NAN;
    scratch.head_alpha = Some(bad_head);
    let old_alpha = root.layers[li].alpha.clone();
    let moved = install_candidate_alpha(&mut root, &scratch, fx.net.n_y);
    let layer = &root.layers[li];
    assert_eq!(
        layer.alpha[unst[0]].to_bits(),
        old_alpha[unst[0]].to_bits(),
        "NaN ingested"
    );
    assert_eq!(
        layer.alpha[unst[1]].to_bits(),
        old_alpha[unst[1]].to_bits(),
        "inf ingested"
    );
    assert_eq!(layer.alpha[unst[2]], 1.0, "1.5 must clamp to 1");
    assert_eq!(layer.alpha[unst[3]], 0.0, "-0.25 must clamp to 0");
    assert_eq!(layer.alpha[unst[4]], 0.3);
    if let Some(j) = stable {
        assert_eq!(
            layer.alpha[j].to_bits(),
            old_alpha[j].to_bits(),
            "a stable slope moved"
        );
    }
    for j in 0..layer.n {
        assert_eq!(
            layer.ms[j],
            layer.alpha[j].max(layer.s[j]),
            "ms != max(alpha, s)"
        );
    }
    assert!(
        root.head_alpha.is_none() || before.1.is_some(),
        "a NaN head was installed"
    );
    let changed = [2usize, 3, 4]
        .iter()
        .filter(|&&p| old_alpha[unst[p]].to_bits() != layer.alpha[unst[p]].to_bits())
        .count();
    assert_eq!(
        moved, changed,
        "moved must count exactly the slopes that changed"
    );
    // l/u/s/c/unst untouched everywhere.
    for (lj, (b, l)) in before.0.iter().zip(&root.layers).enumerate() {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
        assert_eq!(b.0, bits(&l.l), "l moved at {lj}");
        assert_eq!(b.1, bits(&l.u), "u moved at {lj}");
        assert_eq!(b.3, bits(&l.s), "s moved at {lj}");
        assert_eq!(b.4, bits(&l.c), "c moved at {lj}");
        assert_eq!(b.6, l.unst, "unst moved at {lj}");
    }
    // A finite head of the right width is installed, clamped; a wrong width is
    // refused.
    let mut scratch2 = untrusted_scratch(&root);
    scratch2.head_alpha = Some(vec![0.5; fx.net.n_y + 1]);
    let head_before = root.head_alpha.clone();
    install_candidate_alpha(&mut root, &scratch2, fx.net.n_y);
    assert_eq!(
        root.head_alpha, head_before,
        "a wrong-width head was installed"
    );
    let mut head = vec![0.5; fx.net.n_y];
    head[0] = 2.0;
    scratch2.head_alpha = Some(head);
    install_candidate_alpha(&mut root, &scratch2, fx.net.n_y);
    let installed = root.head_alpha.as_ref().expect("a valid head is installed");
    assert_eq!(installed[0], 1.0, "head slopes clamp to [0, 1]");
    assert!(installed.iter().all(|a| (0.0..=1.0).contains(a)));
}

// ---------------------------------------------------------------------------
//  Refusals of the phase itself
// ---------------------------------------------------------------------------

#[test]
fn root_certify_once_refuses_loudly_and_leaves_the_root_bit_identical() {
    let fx = tiny_fixture(0x0C0E_6001, 0.6, 0.4);
    let (t, _, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
    // (1) A Parity incumbent is not a certified root.
    let mut parity =
        RootGates::build(&fx.net, &fx.lo, &fx.hi, RoundMode::Parity, None).expect("root");
    let parity_before = gate_bits(&parity);
    let report = certify_once_root(
        &fx.net,
        &mut parity,
        false,
        t,
        &adv,
        None,
        None,
        armed(8, 0),
        RootAlphaRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
    );
    assert_eq!(report.kept, Kept::None);
    assert_eq!(report.stop, "refused");
    assert!(report.published.is_nan(), "a refusal publishes nothing");
    assert_eq!(gate_bits(&parity), parity_before);
    let line = report.line();
    assert!(line.starts_with("[root-certify-once] armed=1 iters=0 untrusted_best=NaN"));
    assert!(line.ends_with(" stop=refused"), "{line}");
    // (2) No adversarial class.
    let mut root =
        RootGates::build(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None).expect("root");
    let before = gate_bits(&root);
    let report = certify_once_root(
        &fx.net,
        &mut root,
        false,
        t,
        &[],
        None,
        None,
        armed(8, 0),
        RootAlphaRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
    );
    assert_eq!(report.kept, Kept::None);
    assert_eq!(gate_bits(&root), before);
    // (3) A zero wall request.
    let report = certify_once_root(
        &fx.net,
        &mut root,
        false,
        t,
        &adv,
        None,
        None,
        RootCertifyOnceRuntimePolicy {
            secs: 0.0,
            ..armed(8, 0)
        },
        RootAlphaRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
    );
    assert_eq!(report.kept, Kept::None);
    assert_eq!(report.stop, "refused");
    assert_eq!(gate_bits(&root), before);
    // (4) A deadline already behind us leaves no budget either.
    let report = certify_once_root(
        &fx.net,
        &mut root,
        false,
        t,
        &adv,
        Some(std::time::Instant::now()),
        None,
        armed(8, 0),
        RootAlphaRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
    );
    assert_eq!(report.kept, Kept::None);
    assert_eq!(gate_bits(&root), before);
    // (5) A certified incumbent that already closes every class has nothing to
    // buy: the phase stops before any search and leaves the root untouched.
    let report = certify_once_root(
        &fx.net,
        &mut root,
        false,
        t,
        &adv,
        None,
        Some(1.0),
        armed(8, 0),
        RootAlphaRuntimePolicy::default(),
        BackwardIntermRuntimePolicy::default(),
    );
    assert_eq!(report.kept, Kept::Incumbent);
    assert_eq!(report.stop, "closed");
    assert_eq!(report.iters, 0);
    assert_eq!(report.untrusted_passes, 0);
    assert_eq!(report.published.to_bits(), 1.0f64.to_bits());
    assert_eq!(gate_bits(&root), before);
    assert!(report.line().ends_with(" stop=closed"));
}

/// #bbe-joint-root (ENGINE_PROGRAM M5a): the `margin_row.root_search` selector —
/// `ascent` bit-identity with the backend never consulted, the fault-injected
/// ingest (garbage hints never loosen the root and never certify past the
/// truth), lowering fidelity, refusal paths and the `[bbe]` line. A child of
/// this module so it reuses the fixtures and oracles above.
mod root_search_tests;
