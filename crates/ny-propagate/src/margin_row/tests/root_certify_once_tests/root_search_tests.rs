// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #bbe-joint-root (ENGINE_PROGRAM M5a) — the `margin_row.root_search`
//! selector of `#root-certify-once`.
//!
//! The search is UNTRUSTED, so these tests do not need the real backend (which
//! lives in `ny-bbe`, a crate this one may not depend on): they drive the
//! phase with FAULT-INJECTING backends and check what the certify half makes
//! of whatever comes back.
//!
//! * `root_search_ascent_is_bit_identical_and_never_consults_the_backend` —
//!   the default selector is `ascent` (policy, declaration, token set), and
//!   with a backend handed in the phase is bit-identical to the no-backend
//!   phase and never calls it.
//! * `joint_f32_garbage_hints_never_loosen_and_never_certify_past_the_truth`
//!   — THE INGEST CONTRACT under fault injection (NaN, +-inf, +-1e30,
//!   out-of-range, uniform, wide, truncated, wrong ops, wrong head width,
//!   empty, refusal, panic, all-ones, all-zeros, and a LYING search bound):
//!   the published root is never below the certified incumbent, is bit-exactly
//!   the certified evaluation of the root left, stays below every sampled
//!   margin and the EXACT-rational margin, negative on the falsified property,
//!   and the kept root's boxes/chords never move while every slope stays in
//!   `[0, 1]` with `ms = max(alpha, s)`.
//! * `joint_f32_without_a_backend_refuses_and_keeps_the_incumbent`.
//! * `lowered_search_net_evaluates_like_the_twin_net` — the lowering is the
//!   lane IR op for op (sizes, head flags, reference boxes, f64 forward).
//! * `hint_into_scratch_writes_only_matching_layers_and_the_gate_validates`.

use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};

use ny_core::search::{
    BackendCaps, Refusal, RootAlphaHint, RootSearchOutcome, RootSearchRequest, RootSearchStats,
    SearchBackend, SearchNet, SearchOp, SearchPrecision, Untrusted,
};

use crate::margin_row::net::TwinOp;
use crate::margin_row::root_certify_once::certify_once_root_with;
use crate::margin_row::root_search::{hint_into_scratch, lower_twin_net, reference_boxes};

/// What a fault-injecting backend hands back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Garbage {
    Nan,
    PosInf,
    NegInf,
    Huge,
    OutOfRange,
    Uniform01,
    Wide,
    Truncated,
    WrongOps,
    WrongHead,
    Empty,
    Refuse,
    Panic,
    Ones,
    Zeros,
}

const ALL_GARBAGE: [Garbage; 15] = [
    Garbage::Nan,
    Garbage::PosInf,
    Garbage::NegInf,
    Garbage::Huge,
    Garbage::OutOfRange,
    Garbage::Uniform01,
    Garbage::Wide,
    Garbage::Truncated,
    Garbage::WrongOps,
    Garbage::WrongHead,
    Garbage::Empty,
    Garbage::Refuse,
    Garbage::Panic,
    Garbage::Ones,
    Garbage::Zeros,
];

/// The search bound every fault backend CLAIMS. It must never be published.
const LYING_ROOT_LB: f64 = 1.0e9;

struct FaultBackend {
    mode: Garbage,
    seed: u64,
    calls: AtomicUsize,
}

impl FaultBackend {
    fn new(mode: Garbage, seed: u64) -> Self {
        Self {
            mode,
            seed,
            calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    fn value(&self, rng: &mut StdRng) -> f32 {
        match self.mode {
            Garbage::Nan => f32::NAN,
            Garbage::PosInf => f32::INFINITY,
            Garbage::NegInf => f32::NEG_INFINITY,
            Garbage::Huge => {
                if rng.random_range(0.0..1.0) < 0.5 {
                    1.0e30
                } else {
                    -1.0e30
                }
            }
            Garbage::OutOfRange => {
                if rng.random_range(0.0..1.0) < 0.5 {
                    -3.0
                } else {
                    7.0
                }
            }
            Garbage::Wide => rng.random_range(-2.0f32..3.0),
            Garbage::Ones => 1.0,
            Garbage::Zeros => 0.0,
            _ => rng.random_range(0.0f32..1.0),
        }
    }
}

impl SearchBackend for FaultBackend {
    fn name(&self) -> &'static str {
        "fault"
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps::Cpu {
            threads: 1,
            lanes: 1,
            precision: SearchPrecision::F32,
        }
    }

    fn root_joint_alpha(
        &self,
        net: &SearchNet,
        request: &RootSearchRequest,
    ) -> Result<RootSearchOutcome, Refusal> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        match self.mode {
            Garbage::Refuse => return Err(Refusal::Nothing("fault injection")),
            Garbage::Panic => panic!("fault-injected search backend panic"),
            _ => {}
        }
        let sizes = net.tensor_sizes().map_err(Refusal::Invalid)?;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut trunk = Vec::new();
        let mut head = Vec::new();
        for (k, op) in net.ops.iter().enumerate() {
            let SearchOp::Relu {
                input,
                head: is_head,
            } = op
            else {
                continue;
            };
            let n = sizes[*input];
            let width = if self.mode == Garbage::Truncated {
                n.saturating_sub(1)
            } else {
                n
            };
            let slopes: Vec<f32> = (0..width).map(|_| self.value(&mut rng)).collect();
            if *is_head {
                head = if self.mode == Garbage::WrongHead {
                    vec![0.5; n + 1]
                } else {
                    slopes
                };
            } else {
                let op = if self.mode == Garbage::WrongOps {
                    k + 1
                } else {
                    k
                };
                trunk.push((op, slopes));
            }
        }
        if self.mode == Garbage::Empty {
            trunk.clear();
            head.clear();
        }
        Ok(RootSearchOutcome {
            alpha: Untrusted::new(RootAlphaHint { trunk, head }),
            stats: RootSearchStats {
                iters: 3,
                passes: 7,
                search_secs: 0.0,
                crown_lb: Untrusted::new(f64::NAN),
                root_lb: Untrusted::new(LYING_ROOT_LB),
                specs: request.adv.len(),
                pruned: 0,
                rows: 0,
                share_rows: false,
                threads: 1,
                tightened: 0,
                stabilized: 0,
                stop: "iters",
            },
        })
    }
}

fn joint(iters: usize) -> RootCertifyOnceRuntimePolicy {
    RootCertifyOnceRuntimePolicy {
        search: RootSearch::JointF32,
        ..armed(iters, 0)
    }
}

fn outward_root(fx: &Fx) -> RootGates {
    RootGates::build(&fx.net, &fx.lo, &fx.hi, RoundMode::Outward, None).expect("root")
}

// ---------------------------------------------------------------------------
//  Dark default: `ascent` is today's phase, bit for bit
// ---------------------------------------------------------------------------

#[test]
fn root_search_ascent_is_bit_identical_and_never_consults_the_backend() {
    assert_eq!(RootSearch::default(), RootSearch::Ascent);
    assert_eq!(
        RootCertifyOnceRuntimePolicy::default().search,
        RootSearch::Ascent,
        "the search selector must ship as the bit-identical ascent"
    );
    let decl = &ny_levers::decls::margin_row::MARGIN_ROW_ROOT_SEARCH;
    assert!(
        matches!(decl.default, ny_levers::DefaultSpec::Text(token)
            if RootSearch::from_token(token) == Some(RootSearch::Ascent)),
        "the declaration default must be the ascent token"
    );
    assert!(
        matches!(decl.kind, ny_levers::LeverKind::Enum(members)
            if members == &RootSearch::TOKENS[..]),
        "the declaration's members must be exactly the selector's tokens"
    );
    for token in RootSearch::TOKENS {
        assert_eq!(
            RootSearch::from_token(token).map(RootSearch::token),
            Some(token)
        );
    }
    assert_eq!(RootSearch::from_token("joint"), None);

    let backend = FaultBackend::new(Garbage::Uniform01, 0xA5CE);
    let mut runs = 0usize;
    for (k, fx) in sweep().iter().enumerate().take(4) {
        let (t, worst, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
        let adv_worst: Vec<usize> = (0..fx.net.n_out).filter(|&o| o != worst).collect();
        for (target, classes) in [(t, adv), (worst, adv_worst)] {
            let mut plain = outward_root(fx);
            let mut offered = outward_root(fx);
            let a = certify_once_root(
                &fx.net,
                &mut plain,
                false,
                target,
                &classes,
                None,
                None,
                armed(8, 0),
                RootAlphaRuntimePolicy::default(),
                BackwardIntermRuntimePolicy::default(),
            );
            let b = certify_once_root_with(
                &fx.net,
                &mut offered,
                false,
                target,
                &classes,
                None,
                None,
                armed(8, 0),
                RootAlphaRuntimePolicy::default(),
                BackwardIntermRuntimePolicy::default(),
                Some(&backend),
            );
            runs += 1;
            assert!(
                gate_bits(&plain) == gate_bits(&offered),
                "fixture {k}: an offered backend changed the ascent's root"
            );
            assert_eq!(a.published.to_bits(), b.published.to_bits());
            assert_eq!(a.incumbent.to_bits(), b.incumbent.to_bits());
            assert_eq!(a.untrusted_best.to_bits(), b.untrusted_best.to_bits());
            assert_eq!((a.kept, a.moved, a.iters), (b.kept, b.moved, b.iters));
            assert_eq!(a.stop, b.stop);
            assert!(
                a.bbe.is_none() && b.bbe.is_none(),
                "ascent carries no [bbe] telemetry"
            );
        }
    }
    assert!(runs >= 8);
    assert_eq!(
        backend.calls(),
        0,
        "the ascent search must never consult the injected backend"
    );
}

// ---------------------------------------------------------------------------
//  THE INGEST CONTRACT under fault injection
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::too_many_lines)]
fn joint_f32_garbage_hints_never_loosen_and_never_certify_past_the_truth() {
    let (mut runs, mut moved_runs, mut accepted, mut refused, mut falsified) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut exact_checks = 0usize;
    for (k, fx) in sweep().iter().enumerate() {
        let (t, worst, adv) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
        let adv_worst: Vec<usize> = (0..fx.net.n_out).filter(|&o| o != worst).collect();
        for (target, classes) in [(t, adv), (worst, adv_worst)] {
            for (mi, &mode) in ALL_GARBAGE.iter().enumerate() {
                let backend = FaultBackend::new(mode, 0xBBE0 ^ ((k as u64) << 8) ^ mi as u64);
                let before = outward_root(fx);
                let mut root = outward_root(fx);
                let report = certify_once_root_with(
                    &fx.net,
                    &mut root,
                    false,
                    target,
                    &classes,
                    None,
                    None,
                    joint(8),
                    RootAlphaRuntimePolicy::default(),
                    BackwardIntermRuntimePolicy::default(),
                    Some(&backend),
                );
                runs += 1;
                let ctx = format!("fixture {k} target {target} mode {mode:?}");
                let bbe = report
                    .bbe
                    .as_ref()
                    .unwrap_or_else(|| panic!("{ctx}: joint_f32 must carry its [bbe] telemetry"));
                let line = bbe.line(&report);
                assert!(
                    line.starts_with("[bbe] phase=joint_root iters="),
                    "{ctx}: {line}"
                );
                for field in [
                    " search_s=",
                    " certify_s=",
                    " root_lb=",
                    " kept=",
                    " published=",
                    " refused=",
                ] {
                    assert!(
                        line.contains(field),
                        "{ctx}: [bbe] line lacks {field}: {line}"
                    );
                }
                if report.stop != "closed" {
                    assert_eq!(backend.calls(), 1, "{ctx}: the backend runs exactly once");
                }
                // (1) Never loosen: the published root is at least the
                // certified incumbent.
                assert!(
                    report.published >= report.incumbent,
                    "{ctx}: published {} below the incumbent {}",
                    report.published,
                    report.incumbent
                );
                // (2) The published number is the certified evaluation of the
                // root the phase left — never the search's claim.
                let recomputed =
                    certified_root_bound(&fx.net, &fresh_copy(&root), target, &classes)
                        .expect("eval");
                assert_eq!(
                    recomputed.to_bits(),
                    report.published.to_bits(),
                    "{ctx}: published {} is not the certified root_eval {recomputed}",
                    report.published
                );
                assert!(
                    report.published < LYING_ROOT_LB,
                    "{ctx}: the lying bound leaked"
                );
                // (3) The root stays a valid relaxation: boxes, chords and the
                // unstable set untouched; every slope in [0, 1]; ms re-derived;
                // stable slopes unmoved.
                for (li, (old, new)) in before.layers.iter().zip(&root.layers).enumerate() {
                    assert_eq!(old.l, new.l, "{ctx}: l moved at layer {li}");
                    assert_eq!(old.u, new.u, "{ctx}: u moved at layer {li}");
                    assert_eq!(old.s, new.s, "{ctx}: s moved at layer {li}");
                    assert_eq!(old.c, new.c, "{ctx}: c moved at layer {li}");
                    assert_eq!(old.unst, new.unst, "{ctx}: unst moved at layer {li}");
                    let unstable: std::collections::BTreeSet<usize> =
                        old.unst.iter().copied().collect();
                    for j in 0..new.n {
                        assert!(
                            new.alpha[j].is_finite() && (0.0..=1.0).contains(&new.alpha[j]),
                            "{ctx}: slope {} at {li}/{j}",
                            new.alpha[j]
                        );
                        assert_eq!(
                            new.ms[j],
                            new.alpha[j].max(new.s[j]),
                            "{ctx}: ms != max(alpha, s) at {li}/{j}"
                        );
                        if !unstable.contains(&j) {
                            assert_eq!(
                                new.alpha[j].to_bits(),
                                old.alpha[j].to_bits(),
                                "{ctx}: a stable slope moved at {li}/{j}"
                            );
                        }
                    }
                }
                if let Some(head) = root.head_alpha.as_ref() {
                    assert!(head
                        .iter()
                        .all(|a| a.is_finite() && (0.0..=1.0).contains(a)));
                }
                // (4) Never better than the truth: sampled margins ...
                let x = points(&root, &fx.lo, &fx.hi, 48, 0xBBE1 ^ ((k as u64) << 4));
                let truth = sampled_min_margin(&fx.net, &x, target, &classes);
                assert!(
                    report.published <= truth + 1e-9,
                    "{ctx}: certified {} ABOVE the sampled margin {truth}",
                    report.published
                );
                if target == worst {
                    assert!(truth < 0.0, "{ctx}: the runner-up property is falsified");
                    assert!(
                        report.published < 0.0,
                        "{ctx}: a FALSE property must never certify ({})",
                        report.published
                    );
                    falsified += 1;
                }
                // ... and the EXACT-rational margin, per class, at the
                // midpoint and both corners.
                let eng = BackwardEngine::new(&fx.net, &root);
                let re = crate::margin_row::bab::root_eval(&eng, &fx.net, target, &classes)
                    .expect("root_eval");
                for b in 0..3 {
                    let xb: Vec<f64> = (0..x.nrows()).map(|i| x[[i, b]]).collect();
                    let scores = exact_scores(&fx.spec, &xb);
                    for (ki, &j) in classes.iter().enumerate() {
                        let exact = &scores[target] - &scores[j];
                        assert!(
                            q(re.dj[ki]) <= exact,
                            "FALSIFIED: {ctx} class {j} point {b}: certified {:e} exceeds the \
                             EXACT margin",
                            re.dj[ki]
                        );
                        exact_checks += 1;
                    }
                }
                // (5) A refusal, a panic, an empty or all-non-finite hint moves
                // nothing: the incumbent stands bit-for-bit.
                if matches!(
                    mode,
                    Garbage::Refuse
                        | Garbage::Panic
                        | Garbage::Empty
                        | Garbage::Nan
                        | Garbage::PosInf
                        | Garbage::NegInf
                ) && report.stop != "closed"
                {
                    assert_eq!(report.moved, 0, "{ctx}: nothing admissible was offered");
                    assert_eq!(report.kept, Kept::Incumbent, "{ctx}");
                    assert!(gate_bits(&root) == gate_bits(&before), "{ctx}: root moved");
                }
                if matches!(mode, Garbage::Refuse | Garbage::Panic) && report.stop != "closed" {
                    assert_eq!(report.stop, "search_refused", "{ctx}");
                    assert!(bbe.refused.is_some(), "{ctx}: the refusal must be reported");
                }
                moved_runs += usize::from(report.moved > 0);
                accepted += usize::from(matches!(report.kept, Kept::Alpha | Kept::Refresh));
                refused += usize::from(bbe.refused.is_some());
            }
        }
    }
    eprintln!(
        "[bbe-joint-root-test] fault injection: runs={runs} moved={moved_runs} \
         accepted={accepted} refused={refused} falsified={falsified} exact={exact_checks}"
    );
    assert!(
        moved_runs > 0,
        "no garbage hint was ever installed: the ingest path is VOID"
    );
    assert!(falsified > 0, "no falsified property was exercised: VOID");
    assert!(refused > 0, "no refusal was exercised: VOID");
    assert!(exact_checks > 100, "only {exact_checks} exact checks ran");
}

#[test]
fn joint_f32_without_a_backend_refuses_and_keeps_the_incumbent() {
    let fx = tiny_fixture(0x0BBE_0001, 0.6, 0.4);
    let (_, worst, _) = tiny_targets(&fx.net, &fx.lo, &fx.hi);
    let classes: Vec<usize> = (0..fx.net.n_out).filter(|&o| o != worst).collect();
    let before = outward_root(&fx);
    for with_wrapper in [false, true] {
        let mut root = outward_root(&fx);
        let report = if with_wrapper {
            certify_once_root(
                &fx.net,
                &mut root,
                false,
                worst,
                &classes,
                None,
                None,
                joint(8),
                RootAlphaRuntimePolicy::default(),
                BackwardIntermRuntimePolicy::default(),
            )
        } else {
            certify_once_root_with(
                &fx.net,
                &mut root,
                false,
                worst,
                &classes,
                None,
                None,
                joint(8),
                RootAlphaRuntimePolicy::default(),
                BackwardIntermRuntimePolicy::default(),
                None,
            )
        };
        let bbe = report.bbe.as_ref().expect("joint_f32 telemetry");
        assert_eq!(bbe.backend, "none");
        assert_eq!(bbe.refused.as_deref(), Some("no search backend registered"));
        assert_eq!(report.stop, "search_refused");
        assert_eq!(report.kept, Kept::Incumbent);
        assert_eq!(report.moved, 0);
        assert_eq!(gate_bits(&root), gate_bits(&before));
        assert!(bbe
            .line(&report)
            .ends_with(" refused=no search backend registered"));
    }
}

// ---------------------------------------------------------------------------
//  The lowering and the scratch writer
// ---------------------------------------------------------------------------

#[test]
fn lowered_search_net_evaluates_like_the_twin_net() {
    for (k, fx) in sweep().iter().enumerate() {
        let lowered = lower_twin_net(&fx.net).expect("lower");
        assert_eq!(lowered.n_in, fx.net.n_in);
        assert_eq!(lowered.ops.len(), fx.net.ops.len());
        assert_eq!(lowered.tensor_sizes().expect("sizes"), fx.net.tsize);
        for (kk, (low, twin)) in lowered.ops.iter().zip(&fx.net.ops).enumerate() {
            match (low, twin) {
                (SearchOp::Relu { input: a, head }, TwinOp::Relu { input: b, layer }) => {
                    assert_eq!(a, b);
                    assert_eq!(*head, *layer == usize::MAX, "fixture {k} op {kk}");
                }
                (SearchOp::Conv(c), TwinOp::Conv(t)) => {
                    assert_eq!((c.input, c.cout, c.k), (t.input, t.kernel.0, t.k_fwd));
                    assert_eq!(c.weight, t.wmat);
                }
                (SearchOp::Gemm { weight, .. }, TwinOp::Gemm { weight: w, .. }) => {
                    assert_eq!(weight, w);
                }
                (SearchOp::Add { .. }, TwinOp::Add { .. })
                | (SearchOp::Flatten { .. }, TwinOp::Flatten { .. })
                | (SearchOp::Affine { .. }, TwinOp::ChannelAffine { .. }) => {}
                _ => panic!("fixture {k} op {kk}: lowering changed the op kind"),
            }
        }
        let root = outward_root(fx);
        let refs = reference_boxes(&root);
        assert_eq!(refs.len(), root.layers.len());
        for (rb, layer) in refs.iter().zip(&root.layers) {
            assert_eq!(rb.op, layer.op);
            assert_eq!((&rb.l, &rb.u), (&layer.l, &layer.u));
            assert!(matches!(
                lowered.ops[rb.op],
                SearchOp::Relu { head: false, .. }
            ));
        }
        let x = points(&root, &fx.lo, &fx.hi, 12, 0xBBE2 ^ k as u64);
        let (y, _) = fx
            .net
            .forward_points(&x, &std::collections::BTreeMap::new())
            .expect("forward");
        for b in 0..x.ncols() {
            let xb: Vec<f64> = (0..x.nrows()).map(|i| x[[i, b]]).collect();
            let vals = lowered.forward_f64(&xb).expect("lowered forward");
            let y_low = &vals[fx.net.i_gemm1 + 1];
            for i in 0..fx.net.n_y {
                assert!(
                    (y_low[i] - y[[i, b]]).abs() <= 1e-12 * (1.0 + y[[i, b]].abs()),
                    "fixture {k} point {b} y[{i}]: lowered {} vs twin {}",
                    y_low[i],
                    y[[i, b]]
                );
            }
        }
    }
}

#[test]
fn hint_into_scratch_writes_only_matching_layers_and_the_gate_validates() {
    let fx = tiny_fixture(0x0BBE_0002, 0.6, 0.4);
    let root = outward_root(&fx);
    let n_y = fx.net.n_y;
    let (l0, l1) = (&root.layers[0], &root.layers[1]);
    let hint = RootAlphaHint {
        trunk: vec![
            (l0.op, vec![0.25; l0.n]),
            // Not a trunk ReLU op: ignored.
            (l0.op + 1, vec![0.5; l0.n]),
            // The right op at the wrong width: ignored.
            (l1.op, vec![0.5; l1.n + 1]),
        ],
        head: vec![0.75; n_y],
    };
    let mut scratch = untrusted_scratch(&root);
    let (layers, head) = hint_into_scratch(&mut scratch, &hint, n_y);
    assert_eq!((layers, head), (1, true));
    assert!(scratch.layers[0].alpha.iter().all(|&a| a == 0.25));
    assert_eq!(
        scratch.layers[1].alpha, l1.alpha,
        "a mismatched vector must not land"
    );
    assert_eq!(scratch.head_alpha.as_deref(), Some(&vec![0.75; n_y][..]));
    // A head of the wrong width is not written at all.
    let mut other = untrusted_scratch(&root);
    let wrong_head = RootAlphaHint {
        trunk: Vec::new(),
        head: vec![0.75; n_y + 1],
    };
    assert_eq!(hint_into_scratch(&mut other, &wrong_head, n_y), (0, false));
    assert_eq!(other.head_alpha, root.head_alpha);
    // The ingest gate then moves ONLY layer-0 unstable slopes (and the head).
    let mut certified = outward_root(&fx);
    let moved = install_candidate_alpha(&mut certified, &scratch, n_y);
    assert!(moved <= l0.unst.len() + n_y);
    let unstable: std::collections::BTreeSet<usize> = l0.unst.iter().copied().collect();
    for j in 0..l0.n {
        let a = certified.layers[0].alpha[j];
        if unstable.contains(&j) {
            assert_eq!(a, 0.25);
        } else {
            assert_eq!(a.to_bits(), l0.alpha[j].to_bits());
        }
        assert_eq!(certified.layers[0].ms[j], a.max(certified.layers[0].s[j]));
    }
    assert_eq!(certified.layers[1].alpha, l1.alpha);
}
