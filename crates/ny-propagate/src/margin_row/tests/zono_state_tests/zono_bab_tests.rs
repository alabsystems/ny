// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#frozen-dual-bab` (V1 stage B) soundness tests — the certified search over
//! stage A's state. A child of `zono_state_tests`, so it reuses that module's
//! synthetic fixtures and its EXACT-RATIONAL node evaluator.
//!
//! Soundness first:
//!
//! * `frozen_dual_bab_prunes_reverify_in_exact_rationals` — every node the
//!   search itself pruned (sides + the exact multipliers that certified it) is
//!   re-evaluated in exact rationals from the problem's stored floats: the
//!   exact node value is `>= certified > 0`;
//! * `frozen_dual_bab_never_closes_a_witnessed_sat_class` — sat-row fixtures:
//!   the true class swapped for the runner-up (the midpoint is a
//!   counterexample), plus every class with a sampled negative margin, over
//!   all five stage A fixtures and a box sweep: never `Closed`, neither by the
//!   search nor by the tier;
//! * `frozen_dual_bab_planted_violation_is_never_closed` — random nets whose
//!   final bias is moved so the midpoint violates by `delta` in
//!   {1e-3, 1e-6, 1e-9} (checked in double-double): never `Closed`;
//! * `frozen_dual_bab_prunes_only_on_the_certified_value` — the MUTATION PIN:
//!   a state whose every node has a POSITIVE raw value and a NEGATIVE
//!   certified value; a search that pruned on the raw value would close it;
//! * `frozen_dual_bab_closed_classes_hold_at_sampled_points` — the positive
//!   control: the sweep closes classes by splitting (VOID if none), and every
//!   closed class has a positive double-double margin at every sampled point;
//! * the level ledger, fail-closed stops (node cap, memory, deadline),
//!   determinism, the tier's hand-back and its dark knob defaults.

use super::*;

use crate::margin_row::bab::root_eval;
use crate::margin_row::spec::TwinOpSpec;
use crate::margin_row::zono_bab::{
    ledger_closes, run_tier, search, Evaluator, LevelTrace, OpenReason, SearchOptions,
    SearchReport, SearchVerdict, TierPolicy, FROZEN_DUAL_CROSSCHECK_EVERY,
    FROZEN_DUAL_DEFAULT_EVALUATOR, FROZEN_DUAL_DEFAULT_NODE_CAP, FROZEN_DUAL_MAX_NODE_CAP,
    FROZEN_DUAL_MAX_SHARE,
};

/// A box sweep of the tiny twin net: several seeds and half-widths, so some
/// classes are open at the root and closable by splitting.
pub(super) fn sweep() -> Vec<Fx> {
    let mut v = Vec::new();
    for seed in [3u64, 5, 11, 17] {
        for half in [0.3, 0.15, 0.08, 0.04] {
            let mut rng = StdRng::seed_from_u64(seed);
            v.push(fx(
                &format!("tiny{seed}_h{half}"),
                tiny_spec(&mut rng, 0.4),
                half,
                seed,
                false,
            ));
        }
    }
    v
}

pub(super) fn problem_of(st: &FrozenDualState) -> FrozenDualProblem {
    let order = static_order(st);
    FrozenDualProblem::new(st, &order).expect("problem")
}

pub(super) fn opts(node_cap: u64, record: usize) -> SearchOptions {
    SearchOptions {
        node_cap,
        record_prunes: record,
        ..SearchOptions::default()
    }
}

/// The double-double margin `y_t - y_j` at `x`.
fn dd_margin(net: &TwinNet, t: usize, j: usize, x: &[f64]) -> f64 {
    let vals = dd_forward(net, x).expect("dd forward");
    let logits = &vals[net.ops.len()];
    (logits[t] - logits[j]).to_f64()
}

fn midpoint(lo: &[f64], hi: &[f64]) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(l, h)| f64::midpoint(*l, *h))
        .collect()
}

/// `(argmax, runner-up)` of the double-double logits at `x`.
fn top_two(net: &TwinNet, x: &[f64]) -> (usize, usize) {
    let vals = dd_forward(net, x).expect("dd forward");
    let logits: Vec<f64> = vals[net.ops.len()].iter().map(|v| v.to_f64()).collect();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    (idx[0], idx[1])
}

// ---------------------------------------------------------------------------
// The level ledger
// ---------------------------------------------------------------------------

#[test]
fn frozen_dual_bab_ledger_accepts_only_complete_trees() {
    let lv = |depth: usize, frontier: u64, pruned: u64| LevelTrace {
        depth,
        frontier,
        pruned,
    };
    // Complete: 2 -> (2-1)*2 = 2 -> (2-0)*2 = 4, all pruned.
    assert!(ledger_closes(&[lv(1, 2, 1), lv(2, 2, 0), lv(3, 4, 4)]));
    assert!(ledger_closes(&[lv(1, 2, 2)]));
    // Empty, wrong root, dropped node, extra node, depth gap, open tail.
    assert!(!ledger_closes(&[]));
    assert!(!ledger_closes(&[lv(1, 1, 1)]));
    assert!(!ledger_closes(&[lv(2, 2, 2)]));
    assert!(!ledger_closes(&[lv(1, 2, 1), lv(2, 1, 1)]));
    assert!(!ledger_closes(&[lv(1, 2, 1), lv(2, 4, 4)]));
    assert!(!ledger_closes(&[lv(1, 2, 0), lv(3, 4, 4)]));
    assert!(!ledger_closes(&[lv(1, 2, 0), lv(2, 4, 3)]));
    assert!(!ledger_closes(&[lv(1, 2, 3)]));
    // Overflow-safe.
    assert!(!ledger_closes(&[lv(1, 2, 0), lv(2, u64::MAX, 0)]));
}

// ---------------------------------------------------------------------------
// Soundness: the search's own prunes, in exact rationals
// ---------------------------------------------------------------------------

#[test]
fn frozen_dual_bab_prunes_reverify_in_exact_rationals() {
    prunes_reverify_in_exact_rationals(FROZEN_DUAL_DEFAULT_EVALUATOR);
}

/// The reference (per-node) evaluator's twin: the scalar path stays pinned
/// while the default runs the block form.
#[test]
fn frozen_dual_bab_prunes_reverify_in_exact_rationals_node_path() {
    prunes_reverify_in_exact_rationals(Evaluator::Node);
}

fn prunes_reverify_in_exact_rationals(evaluator: Evaluator) {
    let mut checked = 0usize;
    let mut by_candidate = [0usize; 6];
    let mut min_margin = f64::INFINITY;
    let mut fxs = fixtures();
    fxs.extend(sweep());
    for f in &fxs {
        let frame = frame_of(f);
        for &j in &f.adv {
            let st = state_of(f, &frame, j);
            let prob = problem_of(&st);
            let rep = search(
                &prob,
                &SearchOptions {
                    evaluator,
                    ..opts(3_000, 48)
                },
            );
            assert_eq!(
                rep.evaluator,
                evaluator.token(),
                "VOID: another evaluator ran"
            );
            for node in &rep.pruned {
                assert!(
                    node.certified > 0.0,
                    "{} class {j}: a recorded prune is not certified positive",
                    f.name
                );
                let exact = exact_node_value(&prob, &node.sides, &node.l0, &node.l1);
                assert!(
                    q(node.certified) <= exact,
                    "{} class {j}: the search pruned {:?} at certified {:e}, above its EXACT \
                     node value {:e} (a false-unsat generator)",
                    f.name,
                    node.sides,
                    node.certified,
                    exact.to_f64().unwrap_or(f64::NAN)
                );
                assert!(
                    exact.is_positive(),
                    "{}: exact node value not positive",
                    f.name
                );
                let m = (&exact - q(node.certified)).to_f64().unwrap_or(f64::NAN);
                min_margin = min_margin.min(m);
                let slot = match node.candidate {
                    crate::margin_row::zono_bab::Candidate::Warm => 0,
                    crate::margin_row::zono_bab::Candidate::Clamped => 1,
                    crate::margin_row::zono_bab::Candidate::Floor => 2,
                    crate::margin_row::zono_bab::Candidate::Far => 3,
                    crate::margin_row::zono_bab::Candidate::Breakpoint => 4,
                    crate::margin_row::zono_bab::Candidate::Relation => 5,
                };
                by_candidate[slot] += 1;
                checked += 1;
            }
        }
    }
    eprintln!(
        "[frozen-dual-bab-exact] eval={} prunes={checked} warm/clamped/floor/far/breakpoint/relation={by_candidate:?} \
         min(exact - certified)={min_margin:.3e}",
        evaluator.token()
    );
    assert!(
        checked > 0,
        "the search pruned no node on any fixture: the exact re-verification is VOID"
    );
}

// ---------------------------------------------------------------------------
// Soundness: negative controls
// ---------------------------------------------------------------------------

#[test]
fn frozen_dual_bab_never_closes_a_witnessed_sat_class() {
    never_closes_a_witnessed_sat_class(FROZEN_DUAL_DEFAULT_EVALUATOR);
}

/// The reference (per-node) evaluator's twin.
#[test]
fn frozen_dual_bab_never_closes_a_witnessed_sat_class_node_path() {
    never_closes_a_witnessed_sat_class(Evaluator::Node);
}

fn never_closes_a_witnessed_sat_class(evaluator: Evaluator) {
    let ev_opts = |cap: u64| SearchOptions {
        evaluator,
        ..opts(cap, 0)
    };
    let mut witnessed = 0usize;
    let mut failures = Vec::new();
    let mut fxs = fixtures();
    fxs.extend(sweep());
    for f in &fxs {
        let x0 = midpoint(&f.lo, &f.hi);
        // (a) The swapped target: the runner-up as the "true" class makes the
        // midpoint a counterexample for the argmax class.
        let (top, second) = top_two(&f.net, &x0);
        let m = dd_margin(&f.net, second, top, &x0);
        if m < 0.0 {
            witnessed += 1;
            let frame = FrozenDualFrame::from_root(&f.net, &f.root, &f.ybox, second)
                .expect("swapped frame");
            let st = FrozenDualState::build(&f.net, &frame, top, &FrozenDualOptions::default())
                .expect("swapped state");
            let rep = search(&problem_of(&st), &ev_opts(4_000));
            assert_eq!(
                rep.evaluator,
                evaluator.token(),
                "VOID: another evaluator ran"
            );
            if rep.closed() {
                failures.push(format!(
                    "{}: swapped target {second} vs {top} (margin {m:e} at the midpoint) CLOSED \
                     after {} nodes",
                    f.name, rep.nodes
                ));
            }
            let adv: Vec<usize> = (0..f.net.n_out).filter(|&c| c != second).collect();
            let dj = vec![f64::NAN; adv.len()];
            let tier = run_tier(
                &f.net,
                &f.root,
                second,
                &adv,
                &dj,
                &[top],
                &f.ybox,
                None,
                &TierPolicy::clamped(4_000, 5.0),
            );
            if tier.closed.contains(&top) {
                failures.push(format!(
                    "{}: the TIER closed the swapped class {top}",
                    f.name
                ));
            }
        }
        // (b) Every class of the real target with a sampled counterexample.
        let frame = frame_of(f);
        for &j in &f.adv {
            let mut rng = StdRng::seed_from_u64(0x5A7 ^ (j as u64) << 4);
            let witness = (0..48).find_map(|s| {
                let x = sample(&mut rng, &f.lo, &f.hi, s);
                let mj = dd_margin(&f.net, f.t, j, &x);
                (mj < 0.0).then_some(mj)
            });
            let Some(mj) = witness else { continue };
            witnessed += 1;
            let st = state_of(f, &frame, j);
            let rep = search(&problem_of(&st), &ev_opts(4_000));
            if rep.closed() {
                failures.push(format!(
                    "{} class {j}: a sampled point has margin {mj:e} but the search CLOSED",
                    f.name
                ));
            }
        }
    }
    assert!(
        witnessed >= 4,
        "only {witnessed} witnessed sat classes: the negative control is VOID"
    );
    assert!(
        failures.is_empty(),
        "FROZEN-DUAL BAB CLOSED A SAT CLASS (soundness emergency): {failures:#?}"
    );
}

#[test]
fn frozen_dual_bab_planted_violation_is_never_closed() {
    planted_violation_is_never_closed(FROZEN_DUAL_DEFAULT_EVALUATOR);
}

/// The reference (per-node) evaluator's twin.
#[test]
fn frozen_dual_bab_planted_violation_is_never_closed_node_path() {
    planted_violation_is_never_closed(Evaluator::Node);
}

fn planted_violation_is_never_closed(evaluator: Evaluator) {
    let mut planted = 0usize;
    let mut failures = Vec::new();
    for seed in 0..9u64 {
        let half = [0.2, 0.08, 0.03][(seed % 3) as usize];
        let mut rng = StdRng::seed_from_u64(0x9A17 + seed);
        let spec0 = tiny_spec(&mut rng, 0.4);
        let f0 = fx("plant0", spec0.clone(), half, 0x9A17 + seed, false);
        let x0 = midpoint(&f0.lo, &f0.hi);
        let (t, j) = top_two(&f0.net, &x0);
        let m0 = dd_margin(&f0.net, t, j, &x0);
        for delta in [1e-3, 1e-6, 1e-9] {
            // Move y_j up by m0 + delta: the midpoint then violates by delta.
            let mut spec = spec0.clone();
            let Some(TwinOpSpec::Gemm { bias, .. }) = spec.ops.last_mut() else {
                panic!("the tiny spec ends in its final Gemm");
            };
            bias[j] += m0 + delta;
            let f = fx("plant", spec, half, 0x9A17 + seed, false);
            let m = dd_margin(&f.net, t, j, &x0);
            if m >= 0.0 {
                continue;
            }
            planted += 1;
            let frame =
                FrozenDualFrame::from_root(&f.net, &f.root, &f.ybox, t).expect("planted frame");
            let st = FrozenDualState::build(&f.net, &frame, j, &FrozenDualOptions::default())
                .expect("planted state");
            let rep = search(
                &problem_of(&st),
                &SearchOptions {
                    evaluator,
                    ..opts(4_000, 0)
                },
            );
            assert_eq!(
                rep.evaluator,
                evaluator.token(),
                "VOID: another evaluator ran"
            );
            if rep.closed() {
                failures.push(format!(
                    "seed {seed} delta {delta:e}: planted margin {m:e} at the midpoint, yet the \
                     search CLOSED after {} nodes",
                    rep.nodes
                ));
            }
            let adv: Vec<usize> = (0..f.net.n_out).filter(|&c| c != t).collect();
            let dj = vec![f64::NAN; adv.len()];
            let tier = run_tier(
                &f.net,
                &f.root,
                t,
                &adv,
                &dj,
                &[j],
                &f.ybox,
                None,
                &TierPolicy::clamped(4_000, 5.0),
            );
            if tier.closed.contains(&j) {
                failures.push(format!(
                    "seed {seed} delta {delta:e}: the TIER closed planted class {j}"
                ));
            }
        }
    }
    assert!(planted >= 9, "only {planted} planted violations: VOID");
    assert!(
        failures.is_empty(),
        "FROZEN-DUAL BAB CLOSED A PLANTED VIOLATION (soundness emergency): {failures:#?}"
    );
}

/// A state every node of which has a POSITIVE raw value but a NEGATIVE
/// certified value: `c_obj` exceeds `sum|d| + r_obj` by `5e-6` (about 21 ulps
/// of `sum|d| = 1.2e9`; the problem's own rounding of `c_obj - r_obj` keeps
/// ~3.8e-6 of it), while the node certificate charges
/// `gamma_{n+3D+32} sum|terms| ~ 2.5e-5` on `sum|terms| ~ 3.6e9`. Every row is
/// zero, so no split or multiplier can improve the bound.
pub(super) fn raw_positive_state(s_rows: usize) -> FrozenDualState {
    let n_in = 24usize;
    let n = n_in + s_rows;
    let mut d = vec![0.0f64; n];
    for (i, v) in d.iter_mut().enumerate().take(n_in) {
        *v = if i % 2 == 0 { 5.0e7 } else { -5.0e7 };
    }
    let abs_d: f64 = d.iter().map(|v| v.abs()).sum();
    FrozenDualState {
        t: 0,
        class: 1,
        n_gens: n,
        n_in_gens: n_in,
        c_obj: abs_d + 5.0e-6,
        d,
        r_obj: 0.0,
        a: vec![0.0; s_rows * n],
        c_in: vec![0.0; s_rows],
        r: vec![0.0; s_rows],
        lam: vec![0.5; s_rows],
        mu: vec![1.0; s_rows],
        lo: vec![-1.0; s_rows],
        hi: vec![1.0; s_rows],
        enc: (0..s_rows).map(|k| n_in + k).collect(),
        layer: vec![0; s_rows],
        site: vec![ReluSite::Trunk(0); s_rows],
        neuron: (0..s_rows).collect(),
        stats: FrozenDualBuildStats::default(),
    }
}

#[test]
fn frozen_dual_bab_prunes_only_on_the_certified_value() {
    prunes_only_on_the_certified_value(FROZEN_DUAL_DEFAULT_EVALUATOR);
}

/// The reference (per-node) evaluator's twin.
#[test]
fn frozen_dual_bab_prunes_only_on_the_certified_value_node_path() {
    prunes_only_on_the_certified_value(Evaluator::Node);
}

fn prunes_only_on_the_certified_value(evaluator: Evaluator) {
    // MUTATION PIN: raw > 0 everywhere, certified < 0 everywhere. A search
    // that pruned on the raw value (or dropped the gamma charge) would close
    // this state at the root.
    let st = raw_positive_state(6);
    let order: Vec<usize> = (0..st.num_unstable()).collect();
    let prob = FrozenDualProblem::new(&st, &order).expect("problem");
    let mut scratch = NodeScratch::default();
    let root = prob.eval_node(&[], &[], &[], &mut scratch, None);
    assert!(
        root.value > 0.0,
        "the pin needs a positive RAW root ({:e}): VOID",
        root.value
    );
    assert!(
        root.certified <= 0.0,
        "the pin needs a non-positive CERTIFIED root ({:e}): VOID",
        root.certified
    );
    let rep = search(
        &prob,
        &SearchOptions {
            evaluator,
            ..opts(100_000, 0)
        },
    );
    assert_eq!(
        rep.evaluator,
        evaluator.token(),
        "VOID: another evaluator ran"
    );
    assert!(
        !rep.closed(),
        "the search closed a state whose every node is certified <= 0: it pruned on a raw value"
    );
    assert_eq!(
        rep.certified_prunes, 0,
        "a node pruned without a certificate"
    );
    assert_eq!(
        rep.verdict,
        SearchVerdict::Open(OpenReason::SplitsExhausted),
        "the whole tree must be walked and handed back"
    );
    assert!(
        rep.worst_open > 0.0,
        "the raw values stayed positive to the last level ({:e}), so a raw-prune search \
         would have closed it: the pin is live",
        rep.worst_open
    );
    assert_eq!(
        rep.nodes,
        (1u64 << 7) - 2,
        "every node of the 6-row tree evaluated"
    );
}

// ---------------------------------------------------------------------------
// Positive control
// ---------------------------------------------------------------------------

#[test]
fn frozen_dual_bab_closed_classes_hold_at_sampled_points() {
    let mut closed_by_split = 0usize;
    let mut closed_points = 0usize;
    let mut failures = Vec::new();
    for f in &sweep() {
        let frame = frame_of(f);
        for &j in &f.adv {
            let st = state_of(f, &frame, j);
            let rep = search(&problem_of(&st), &opts(6_000, 0));
            if !rep.closed() {
                continue;
            }
            if rep.nodes > 0 {
                closed_by_split += 1;
            }
            let mut rng = StdRng::seed_from_u64(0xC105 ^ (j as u64) << 3);
            for s in 0..64 {
                let x = sample(&mut rng, &f.lo, &f.hi, s);
                let m = dd_margin(&f.net, f.t, j, &x);
                closed_points += 1;
                if m <= 0.0 {
                    failures.push(format!(
                        "{} class {j}: CLOSED ({} nodes) but the point {s} has margin {m:e}",
                        f.name, rep.nodes
                    ));
                }
            }
        }
    }
    eprintln!(
        "[frozen-dual-bab-positive] closed_by_split={closed_by_split} points={closed_points}"
    );
    assert!(
        failures.is_empty(),
        "FROZEN-DUAL BAB CLOSED A CLASS WITH A COUNTEREXAMPLE: {failures:#?}"
    );
    assert!(
        closed_by_split > 0,
        "no class of the sweep closed by splitting: the positive control is VOID"
    );
}

// ---------------------------------------------------------------------------
// Fail-closed stops, determinism, the tier
// ---------------------------------------------------------------------------

/// The first (fixture, class) of the sweep whose state root is open.
pub(super) fn open_problem() -> (String, FrozenDualProblem) {
    for f in &sweep() {
        let frame = frame_of(f);
        for &j in &f.adv {
            let st = state_of(f, &frame, j);
            if st.root_bound() < -1e-6 && st.num_unstable() > 0 {
                return (format!("{} class {j}", f.name), problem_of(&st));
            }
        }
    }
    panic!("no open objective in the sweep: VOID");
}

#[test]
fn frozen_dual_bab_stops_open_on_cap_memory_and_deadline() {
    let (name, prob) = open_problem();
    let capped = search(&prob, &opts(1, 0));
    assert_eq!(
        capped.verdict,
        SearchVerdict::Open(OpenReason::NodeCap),
        "{name}: node cap 1 must stop before the first level"
    );
    assert_eq!(capped.nodes, 0);
    let starved = search(
        &prob,
        &SearchOptions {
            max_frontier_bytes: 1,
            ..SearchOptions::default()
        },
    );
    assert_eq!(
        starved.verdict,
        SearchVerdict::Open(OpenReason::Memory),
        "{name}: an unadmitted level must stop the search open"
    );
    let late = search(
        &prob,
        &SearchOptions {
            deadline: Some(std::time::Instant::now()),
            ..SearchOptions::default()
        },
    );
    assert_eq!(
        late.verdict,
        SearchVerdict::Open(OpenReason::Deadline),
        "{name}: a passed deadline must stop the search open"
    );
}

#[test]
fn frozen_dual_bab_search_is_deterministic() {
    let (name, prob) = open_problem();
    let digest = |r: &SearchReport| {
        (
            r.verdict,
            r.nodes,
            r.depth,
            r.certified_prunes,
            r.cert_min_slack.to_bits(),
            r.far_used,
            r.levels.clone(),
        )
    };
    let a = search(&prob, &opts(3_000, 0));
    let b = search(&prob, &opts(3_000, 0));
    assert_eq!(
        digest(&a),
        digest(&b),
        "{name}: the parallel search is not deterministic"
    );
    // Chunking only schedules: one node per work item gives the same tree.
    let c = search(
        &prob,
        &SearchOptions {
            chunk: 1,
            ..opts(3_000, 0)
        },
    );
    assert_eq!(
        digest(&a),
        digest(&c),
        "{name}: the chunk size changed the tree"
    );
}

#[test]
fn frozen_dual_bab_tier_reports_every_open_class_and_hands_back_on_cap() {
    let mut seen = 0usize;
    let mut open_objectives = 0usize;
    for f in fixtures() {
        let eng = BackwardEngine::new(&f.net, &f.root);
        let re = root_eval(&eng, &f.net, f.t, &f.adv).expect("root eval");
        // Node cap 1: no class can close by splitting, and the state root is
        // never above the lane's own root, so nothing closes at all.
        let tier = run_tier(
            &f.net,
            &f.root,
            f.t,
            &f.adv,
            &re.dj,
            &re.tree_classes,
            &re.root_ybox(),
            None,
            &TierPolicy::clamped(1, 5.0),
        );
        assert_eq!(tier.objectives, re.tree_classes.len(), "{}", f.name);
        assert_eq!(tier.classes.len(), re.tree_classes.len(), "{}", f.name);
        assert!(
            tier.closed.is_empty(),
            "{}: node cap 1 closed {:?}",
            f.name,
            tier.closed
        );
        // The lane hook itself, at the governed (default) policy and a short
        // lane deadline: whatever it closes is a class the root left open.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let hooked = crate::margin_row::zono_bab::lane_tier(
            &f.net,
            &f.root,
            f.t,
            &f.adv,
            &re,
            Some(deadline),
            FROZEN_DUAL_CROSSCHECK_EVERY,
            false,
        );
        assert_eq!(hooked.objectives, re.tree_classes.len());
        assert!(hooked.closed.iter().all(|c| re.tree_classes.contains(c)));
        open_objectives += re.tree_classes.len();
        seen += 1;
    }
    assert!(seen > 0, "VOID");
    // Every assertion above is vacuous on a fixture whose root closes every
    // class: at least one objective must reach the tier.
    assert!(
        open_objectives > 0,
        "no fixture left an objective open at the root: the hand-back check is VOID"
    );
}

/// G4 RATE (ENGINE_PROGRAM M2d, §1.7 B4): the tier's block-path cross-check
/// rate is the POLICY's — the declaration default `256` (the Competition
/// profile's), floored at `1` so no caller can switch the guard off — and the
/// policy's rate reaches EVERY class's search. At rate `1` (the Proof
/// profile's) every certified prune of every class is re-checked against the
/// scalar evaluator with zero mismatches, and the tree is the default rate's
/// exactly (closed classes, nodes, certified prunes): the rate moves audit
/// work, never a verdict. The lane sets the rate from the entry point's audit
/// rates (`bab.rs` `fd_crosscheck_every`); this module never names the profile.
#[test]
fn frozen_dual_bab_g4_rate_rides_the_tier_policy_into_every_class() {
    let d = TierPolicy::clamped(10, 0.0);
    assert_eq!(d.crosscheck_every, FROZEN_DUAL_CROSSCHECK_EVERY);
    assert_eq!(
        TierPolicy::from_gates().crosscheck_every,
        FROZEN_DUAL_CROSSCHECK_EVERY
    );
    assert_eq!(d.with_crosscheck_every(0).crosscheck_every, 1, "never off");
    assert_eq!(d.with_crosscheck_every(1).crosscheck_every, 1);
    assert_eq!(
        d.with_crosscheck_every(7),
        TierPolicy {
            crosscheck_every: 7,
            ..d
        },
        "only the rate moves"
    );
    assert!(
        matches!(FROZEN_DUAL_DEFAULT_EVALUATOR, Evaluator::Block { .. }),
        "the cross-check guards the block path, which must be the default"
    );
    let mut rechecked = 0u64;
    for f in fixtures() {
        let eng = BackwardEngine::new(&f.net, &f.root);
        let re = root_eval(&eng, &f.net, f.t, &f.adv).expect("root eval");
        let run = |policy: TierPolicy| {
            run_tier(
                &f.net,
                &f.root,
                f.t,
                &f.adv,
                &re.dj,
                &re.tree_classes,
                &re.root_ybox(),
                None,
                &policy,
            )
        };
        let base = TierPolicy::clamped(4_000, 60.0);
        let sampled = run(base);
        let every = run(base.with_crosscheck_every(1));
        assert_eq!(
            sampled.closed, every.closed,
            "{}: the rate moved a verdict",
            f.name
        );
        assert_eq!(
            sampled.nodes, every.nodes,
            "{}: the rate moved the tree",
            f.name
        );
        assert_eq!(
            every.crosscheck_failures, 0,
            "{}: a block/scalar mismatch",
            f.name
        );
        for (a, b) in sampled.classes.iter().zip(&every.classes) {
            let (Ok(a), Ok(b)) = (&a.result, &b.result) else {
                assert_eq!(a.result.is_ok(), b.result.is_ok(), "{}: refusals", f.name);
                continue;
            };
            assert_eq!(
                a.search.certified_prunes, b.search.certified_prunes,
                "{}: the rate moved a prune",
                f.name
            );
            assert_eq!(
                b.search.crosschecks, b.search.certified_prunes,
                "{}: rate 1 must re-check EVERY certified prune",
                f.name
            );
            assert!(
                a.search.crosschecks <= b.search.crosschecks,
                "{}: the sampled rate re-checked more than every prune",
                f.name
            );
            rechecked += b.search.crosschecks;
        }
    }
    assert!(
        rechecked > 0,
        "VOID: no fixture produced a certified prune to re-check"
    );
}

#[test]
fn frozen_dual_bab_knobs_are_dark_and_bounded_by_default() {
    use ny_levers::decls::margin_row::{
        MARGIN_ROW_FROZEN_DUAL_BAB_NODE_CAP as CAP, MARGIN_ROW_FROZEN_DUAL_BAB_SECS as SECS,
    };
    assert_eq!(
        CAP.default,
        ny_levers::DefaultSpec::U64(FROZEN_DUAL_DEFAULT_NODE_CAP)
    );
    assert_eq!(SECS.default, ny_levers::DefaultSpec::F64(0.0));
    assert_eq!(CAP.moat, ny_levers::MoatRisk::High);
    assert_eq!(SECS.moat, ny_levers::MoatRisk::High);
    // Governed reads only (no raw environment access in this test): when
    // no lever is overridden, the knobs resolve to their declarations (the
    // policy now also carries `margin_row.frozen_dual_bab_state` and M4's
    // `margin_row.frozen_dual_bab_search`).
    if ny_levers::read(&CAP).source == ny_levers::Source::Default
        && ny_levers::read(&SECS).source == ny_levers::Source::Default
        && ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_STATE).source
            == ny_levers::Source::Default
        && ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH).source
            == ny_levers::Source::Default
    {
        assert_eq!(
            TierPolicy::from_gates(),
            TierPolicy::clamped(FROZEN_DUAL_DEFAULT_NODE_CAP, 0.0),
            "with no environment and no preset cell the knobs resolve to their declarations"
        );
    }
    // Reader clamps and the budget rule.
    assert_eq!(TierPolicy::clamped(0, f64::NAN).node_cap, 1);
    assert_eq!(TierPolicy::clamped(0, f64::NAN).secs, 0.0);
    assert_eq!(
        TierPolicy::clamped(u64::MAX, -3.0).node_cap,
        FROZEN_DUAL_MAX_NODE_CAP
    );
    let auto = TierPolicy::clamped(10, 0.0);
    assert_eq!(auto.budget_secs(Some(80.0)), 4.0);
    assert_eq!(auto.budget_secs(Some(20.0)), 1.0);
    assert_eq!(auto.budget_secs(Some(0.5)), 0.0);
    assert_eq!(auto.budget_secs(None), 4.0);
    let ask = TierPolicy::clamped(10, 90.0);
    assert_eq!(ask.budget_secs(Some(80.0)), FROZEN_DUAL_MAX_SHARE * 80.0);
    assert_eq!(ask.budget_secs(Some(200.0)), 90.0);
    assert_eq!(auto.budget_secs(Some(-5.0)), 0.0);
}
