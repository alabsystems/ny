// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! B2.3 (ENGINE_PROGRAM §1.7 B1): the frozen-dual tier's WRITE-ONLY prune
//! trace. A child of `zono_state_tests`, so it reuses the stage-B sweep.
//!
//! * `fd_prune_trace_is_complete_and_streams_absolute_literals` — a class the
//!   tier closes by splitting, under hint capture, carries a complete trace
//!   whose stream is one state record and one node per certified prune: the
//!   split order's absolute sites with the prune's sides, the class, the dual
//!   point, and a certified value the lane's own evaluator reproduces bit for
//!   bit at that dual point (the trace binds to the state it was certified
//!   in);
//! * `fd_tier_report_is_bit_identical_with_capture_on_and_off` — recording
//!   the prunes moves nothing the cover adapter reads: every class's search
//!   report (verdict, levels, counts, certified values) is the same bits with
//!   capture on and off, and capture off builds no trace at all;
//! * `margin_row_fd_tier_streams_prunes_through_the_frame` — the lane itself
//!   (the tier armed by the harness field, not the process gate): an outer
//!   frame closed by the tier streams the tier's prunes instead of the
//!   `close: fd` claim, and its verdict is bit-identical with capture off;
//! * `fd_prune_trace_past_the_record_cap_is_incomplete` — a class closed by
//!   splitting whose certified prunes outran the record cap (forced small by
//!   the test hook) carries an INCOMPLETE trace, at a cap of one below its
//!   prune count as at a cap of one, and a complete one at exactly its prune
//!   count; the capped search is the uncapped search, bit for bit;
//! * `margin_row_fd_trace_past_the_record_cap_streams_the_claim_not_its_prunes`
//!   — through the lane, such a class streams its `close: fd` CLAIM (which
//!   the checker can only report unconfirmed) and never a `hint.fd_state` or
//!   an `fd_prune` node: a partial prune set must never reach the bundle as
//!   if it were the class's certificate (the checker would read the missing
//!   cells as a claimed-cover gap, INVALID on a correct unsat).
//!
//! `#fd-compact-checkable` (#breakpoint-dual RANKED compact states):
//!
//! * `fd_prune_trace_of_a_ranked_compact_state_is_never_complete` (the eng9
//!   pin, kept) — WITHOUT its class relaxation a compact trace is never
//!   complete; `fd_prune_trace_of_a_ranked_compact_state_is_complete_with_its_class_relaxation`
//!   — with its verified class relaxation it is, and its table is the full
//!   state's relaxation symbol for symbol;
//! * `fd_compact_class_relaxation_mutations_are_refused` — a dropped omitted
//!   entry, a perturbed omitted or retained slope and the wrong class are
//!   each refused by name;
//!   `fd_compact_class_relaxation_entry_the_margin_does_not_read_is_refused`
//!   (review) — an omitted entry whose margin coefficient is exactly zero,
//!   invisible to the margin replay but read by the retained rows above it,
//!   is refused by the re-choice (`table_choice`);
//! * `fd_tier_compact_traces_are_complete_and_the_search_is_bit_identical`,
//!   `fd_compact_trace_past_the_table_cap_is_incomplete` and
//!   `margin_row_fd_tier_streams_compact_prunes_through_the_frame` — the
//!   same through the tier and the lane (compact states forced by the
//!   threshold hook), Competition bit-identical.

use super::zono_bab_tests::sweep;
use super::*;

use std::sync::{Arc, Mutex};

use crate::certify::{
    force_hint_capture_for_test, with_thread_sink, AlphaRecord, FdStateRecord, LeafRecord,
    NodeClose, NodeRecord, ProofSink, RecordingSink, SitesRecord, VerdictRecord,
};
use crate::margin_row::bab::{root_eval, BabConfig, MarginRowBab, MarginRowOutcome};
use crate::margin_row::zono_bab::{
    fd_proof_table_bytes, fd_proof_trace_for_test, force_fd_proof_record_cap_for_test,
    force_fd_proof_table_cap_for_test, force_fd_streamed_threshold_for_test, run_tier, search,
    ClassRun, FdTable, SearchOptions, SearchReport, TierPolicy, TierReport, FROZEN_DUAL_MAX_DEPTH,
};
use crate::margin_row::zono_state::CompactRelax;
use std::collections::{BTreeMap, BTreeSet};

/// A recording sink the test keeps a handle to (the thread sink is boxed).
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

fn search_digest(s: &SearchReport) -> String {
    format!(
        "{:?} {} {} {} {:x} {:x} {} {} {:x} {:?} {:?} {} {}",
        s.verdict,
        s.nodes,
        s.depth,
        s.certified_prunes,
        s.cert_min_slack.to_bits(),
        s.root_certified.to_bits(),
        s.far_used,
        s.peak_frontier,
        s.worst_open.to_bits(),
        s.levels,
        s.slack_hist,
        s.crosschecks,
        s.crosscheck_failures,
    )
}

fn tier(f: &Fx, capture: bool) -> TierReport {
    let eng = BackwardEngine::new(&f.net, &f.root);
    let re = root_eval(&eng, &f.net, f.t, &f.adv).expect("root eval");
    force_hint_capture_for_test(Some(capture));
    let report = run_tier(
        &f.net,
        &f.root,
        f.t,
        &f.adv,
        &re.dj,
        &re.tree_classes,
        &re.root_ybox(),
        None,
        &TierPolicy::clamped(6_000, 30.0),
    );
    force_hint_capture_for_test(None);
    report
}

fn runs(report: &TierReport) -> Vec<(usize, &ClassRun)> {
    report
        .classes
        .iter()
        .filter_map(|c| c.result.as_ref().ok().map(|r| (c.class, r)))
        .collect()
}

#[test]
fn fd_prune_trace_is_complete_and_streams_absolute_literals() {
    let mut streamed = 0usize;
    for f in &sweep() {
        let report = tier(f, true);
        let frame = frame_of(f);
        for (class, run) in runs(&report) {
            let proof = run.proof.as_ref().expect("capture on builds a trace");
            assert_eq!(proof.class(), class);
            if !run.search.closed() {
                continue;
            }
            assert!(
                proof.complete(),
                "{} class {class}: a closed class's trace must be complete",
                f.name
            );
            assert!(
                run.search.pruned.is_empty(),
                "the prunes are MOVED out of the report the cover reads"
            );
            if run.search.levels.is_empty() {
                continue;
            }
            // Stream it and re-derive every node with the lane's OWN
            // evaluator on a freshly built state (same build, same order).
            let mut sink = RecordingSink::default();
            (**proof).clone().stream_into(&[], &mut sink);
            assert_eq!(sink.fd_states.len(), 1);
            let rec = &sink.fd_states[0];
            assert_eq!(rec.class, class);
            assert_eq!(rec.state_sha256.len(), 64);
            assert_eq!(rec.prunes, run.search.certified_prunes);
            assert_eq!(sink.nodes.len() as u64, run.search.certified_prunes);
            let st = state_of(f, &frame, class);
            let order = static_order(&st);
            let prob =
                FrozenDualProblem::new(&st, &order[..run.depth_cap]).expect("the same problem");
            // The order's absolute sites: the state row's layer op + neuron.
            for (i, site) in rec.order.iter().enumerate() {
                let k = order[i];
                assert_eq!(site.op as usize, frame.layers[st.layer[k]].op);
                assert_eq!(site.index as usize, st.neuron[k]);
            }
            let unstable: usize = rec.lambda.iter().map(|l| l.idx.len()).sum();
            assert_eq!(unstable, st.num_unstable());
            let mut scratch = NodeScratch::default();
            for n in &sink.nodes {
                assert_eq!(n.close, NodeClose::Bound);
                assert_eq!(n.src, "fd_prune");
                assert_eq!(n.dis.as_deref(), Some(&[class][..]));
                let hint = n.fd.as_ref().expect("an fd prune carries its dual point");
                assert_eq!(hint.state, rec.id);
                assert_eq!(hint.l0.len(), n.lits.len());
                assert_eq!(hint.l1.len(), n.lits.len());
                let sides: Vec<Side> = n
                    .lits
                    .iter()
                    .enumerate()
                    .map(|(i, l)| {
                        assert_eq!(l.site, rec.order[i], "literal {i} is order[{i}]");
                        if l.positive {
                            Side::On
                        } else {
                            Side::Off
                        }
                    })
                    .collect();
                let b = prob.eval_node(&sides, &hint.l0, &hint.l1, &mut scratch, None);
                assert!(
                    b.prunes(),
                    "{} class {class}: a streamed prune re-evaluates open",
                    f.name
                );
                assert_eq!(
                    b.certified.to_bits(),
                    n.margin.to_bits(),
                    "the streamed dual point reproduces the certified value bit for bit"
                );
            }
            streamed += 1;
        }
    }
    assert!(
        streamed > 0,
        "no class of the sweep closed by splitting under capture: VOID"
    );
}

#[test]
fn fd_tier_report_is_bit_identical_with_capture_on_and_off() {
    let mut compared = 0usize;
    for f in &sweep() {
        let off = tier(f, false);
        let on = tier(f, true);
        assert_eq!(off.closed, on.closed, "{}", f.name);
        assert_eq!(off.nodes, on.nodes, "{}", f.name);
        let (a, b) = (runs(&off), runs(&on));
        assert_eq!(a.len(), b.len(), "{}", f.name);
        for ((ca, ra), (cb, rb)) in a.iter().zip(&b) {
            assert_eq!(ca, cb);
            assert!(ra.proof.is_none(), "capture off must build no trace");
            // review(b2-fd): Competition RECORDS nothing (`record_prunes: 0`,
            // the historical value) — `search_digest` leaves `pruned` out, so
            // without this pin an always-on recorder (memory and wall inside
            // the tier's budget under Competition) would pass.
            assert!(
                ra.search.pruned.is_empty(),
                "{} class {ca}: capture off recorded {} prune(s)",
                f.name,
                ra.search.pruned.len()
            );
            assert!(rb.proof.is_some(), "capture on builds every class's trace");
            assert_eq!(
                search_digest(&ra.search),
                search_digest(&rb.search),
                "{} class {ca}: recording the prunes moved the search",
                f.name
            );
            assert_eq!(ra.root_g.to_bits(), rb.root_g.to_bits());
            compared += 1;
        }
    }
    assert!(compared > 0, "no class reached the tier: VOID");
}

#[test]
fn margin_row_fd_tier_streams_prunes_through_the_frame() {
    let mut exercised = 0usize;
    for f in &sweep() {
        let cfg = || BabConfig {
            frontier: 1,
            frontier_contextual: 1,
            lru_cap: 64,
            max_expansions: 400,
            frozen_dual_tier: Some(true),
            ..BabConfig::default()
        };
        let run = |capture: bool| {
            let sink = Shared::default();
            force_hint_capture_for_test(Some(capture));
            let (out, _) = with_thread_sink(Box::new(sink.clone()), || {
                MarginRowBab::run(&f.net, &f.root, f.t, &f.adv, cfg())
            });
            force_hint_capture_for_test(None);
            let rec = sink.0.lock().expect("lock").clone();
            (out, rec)
        };
        let (off, off_rec) = run(false);
        assert!(off_rec.nodes.is_empty() && off_rec.fd_states.is_empty());
        let (on, rec) = run(true);
        assert_eq!(
            verdict_bits(&off),
            verdict_bits(&on),
            "{}: capture moved the verdict",
            f.name
        );
        let MarginRowOutcome::Unsat(stats, _) = &on else {
            continue;
        };
        let fd_classes: Vec<usize> = rec.fd_states.iter().map(|s| s.class).collect();
        if fd_classes.is_empty() {
            continue;
        }
        exercised += 1;
        for class in &fd_classes {
            assert!(
                !rec.nodes
                    .iter()
                    .any(|n| n.close == NodeClose::Fd && n.dis.as_deref() == Some(&[*class][..])),
                "{}: class {class} streamed its prunes AND its claim",
                f.name
            );
            assert!(rec
                .nodes
                .iter()
                .any(|n| n.src == "fd_prune" && n.dis.as_deref() == Some(&[*class][..])));
        }
        assert!(stats.root_closed_classes >= fd_classes.len());
    }
    assert!(
        exercised > 0,
        "no lane run closed a class through the frozen-dual tier: VOID"
    );
}

/// #breakpoint-dual x B2.3 (eng9 reconciliation), `#fd-compact-checkable`:
/// a RANKED (streamed) state keeps only the retained split rows, while every
/// omitted unstable neuron still owns its relaxation symbol at the class's
/// slope. Its trace is complete ONLY with its verified class relaxation:
///
/// * labelled `Full` (its rows' slopes alone) or with a refused table, a
///   closed compact search's trace is never complete (the old pin);
/// * with the class relaxation the trace is complete, its table covers every
///   relaxation symbol of the frame exactly once, at the slopes and boxes the
///   FULL state of the same class streams (the same margin pass chose them),
///   and every streamed prune re-evaluates on the compact problem bit for bit;
/// * the kept build is the historical build bit for bit (Competition calls
///   `build_ranked`, which drops the relaxation as it always did).
///
/// VOID unless a compact search closes by splitting somewhere.
#[test]
fn fd_prune_trace_of_a_ranked_compact_state_is_complete_with_its_class_relaxation() {
    let opts = SearchOptions {
        record_prunes: 1 << 20,
        ..SearchOptions::default()
    };
    let (mut closed, mut deep) = (0usize, 0usize);
    for f in &sweep() {
        let frame = frame_of(f);
        if frame.n_unstable < 2 {
            continue;
        }
        for &class in &f.adv {
            let keep = frame.n_unstable - 1;
            let (ranked, mut relax) = FrozenDualState::build_ranked_with_relax(
                &f.net,
                &frame,
                class,
                &FrozenDualOptions::default(),
                keep,
            )
            .expect("ranked state");
            let plain = FrozenDualState::build_ranked(
                &f.net,
                &frame,
                class,
                &FrozenDualOptions::default(),
                keep,
            )
            .expect("ranked state");
            let (mut a, mut b) = (ranked.clone(), plain);
            a.state.stats.build_ms = 0.0;
            b.state.stats.build_ms = 0.0;
            assert_eq!(a, b, "{}: keeping the relaxation moved the build", f.name);
            assert_eq!(relax.len(), frame.n_unstable);
            assert_eq!(
                relax.verify(&f.net, &frame, &ranked.state, None),
                Ok(()),
                "{} class {class}: the class relaxation re-derives the state's margin row",
                f.name
            );
            let k = ranked.state.num_unstable();
            assert!(
                k < frame.n_unstable,
                "{}: a ranked state drops a row",
                f.name
            );
            let order: Vec<usize> = (0..k).collect();
            let prob = FrozenDualProblem::new(&ranked.state, &order).expect("problem");
            let rep = search(&prob, &opts);
            if !rep.closed() {
                continue;
            }
            closed += 1;
            for table in [FdTable::Full, FdTable::Refused("test")] {
                let mut r = rep.clone();
                let tr =
                    fd_proof_trace_for_test(&ranked.state, &frame, &order, &prob, &mut r, table);
                assert!(
                    !tr.complete(),
                    "{} class {class}: a compact trace without its class relaxation is complete",
                    f.name
                );
            }
            let mut r = rep.clone();
            let trace = fd_proof_trace_for_test(
                &ranked.state,
                &frame,
                &order,
                &prob,
                &mut r,
                FdTable::Compact(relax),
            );
            assert!(
                trace.complete(),
                "{} class {class}: a closed compact trace with its class relaxation is complete",
                f.name
            );
            assert_eq!(trace.relaxed(), frame.n_unstable);
            let mut sink = RecordingSink::default();
            trace.stream_into(&[], &mut sink);
            assert_eq!(sink.fd_states.len(), 1);
            let rec = &sink.fd_states[0];
            assert_eq!(rec.unstable, k);
            assert_eq!(rec.relaxed, frame.n_unstable);
            // The table: every relaxation symbol of the frame once, at the
            // full state's slopes and boxes.
            let full = state_of(f, &frame, class);
            let want: BTreeMap<(u32, u32), (u64, u64, u64)> = (0..full.num_unstable())
                .map(|i| {
                    (
                        (frame.layers[full.layer[i]].op as u32, full.neuron[i] as u32),
                        (
                            full.lam[i].to_bits(),
                            full.lo[i].to_bits(),
                            full.hi[i].to_bits(),
                        ),
                    )
                })
                .collect();
            let mut got: BTreeMap<(u32, u32), (u64, u64, u64)> = BTreeMap::new();
            for l in &rec.lambda {
                assert_eq!(l.idx.len(), l.val.len());
                for i in 0..l.idx.len() {
                    let prior = got.insert(
                        (l.op, l.idx[i]),
                        (l.val[i].to_bits(), l.lo[i].to_bits(), l.hi[i].to_bits()),
                    );
                    assert!(prior.is_none(), "{}: a table entry twice", f.name);
                }
            }
            assert_eq!(
                got, want,
                "{} class {class}: the compact table is not the full state's relaxation",
                f.name
            );
            if rep.levels.is_empty() {
                assert_eq!(sink.nodes.len(), 1, "a root close streams one node");
                continue;
            }
            assert_eq!(sink.nodes.len() as u64, rep.certified_prunes);
            let mut scratch = NodeScratch::default();
            for n in &sink.nodes {
                let hint = n.fd.as_ref().expect("an fd prune carries its dual point");
                let sides: Vec<Side> = n
                    .lits
                    .iter()
                    .map(|l| if l.positive { Side::On } else { Side::Off })
                    .collect();
                let b = prob.eval_node(&sides, &hint.l0, &hint.l1, &mut scratch, None);
                assert!(
                    b.prunes(),
                    "{} class {class}: a streamed compact prune re-evaluates open",
                    f.name
                );
                assert_eq!(b.certified.to_bits(), n.margin.to_bits());
            }
            deep += 1;
        }
    }
    assert!(closed > 0, "VOID: no ranked compact search closed");
    assert!(
        deep > 0,
        "VOID: no ranked compact search closed by splitting"
    );
}

/// #breakpoint-dual x B2.3 (the eng9 reconciliation pin, kept under
/// `#fd-compact-checkable`): a RANKED compact state's trace WITHOUT its class
/// relaxation — its rows' slopes alone (labelled `Full`), or a refused table
/// — is never complete, so the lane keeps the class's opaque claim. Positive
/// control: the FULL state of the same classes yields complete traces. Both
/// sides must actually close somewhere, or the test is VOID.
#[test]
fn fd_prune_trace_of_a_ranked_compact_state_is_never_complete() {
    let opts = SearchOptions {
        record_prunes: 1 << 20,
        ..SearchOptions::default()
    };
    let (mut ranked_closed, mut full_closed) = (0usize, 0usize);
    for f in &sweep() {
        let frame = frame_of(f);
        if frame.n_unstable < 2 {
            continue;
        }
        for &class in &f.adv {
            let keep = frame.n_unstable - 1;
            let ranked = FrozenDualState::build_ranked(
                &f.net,
                &frame,
                class,
                &FrozenDualOptions::default(),
                keep,
            )
            .expect("ranked state");
            let k = ranked.state.num_unstable();
            assert!(
                k < frame.n_unstable,
                "{}: a ranked state drops a row",
                f.name
            );
            let order: Vec<usize> = (0..k).collect();
            let prob = FrozenDualProblem::new(&ranked.state, &order).expect("problem");
            let rep = search(&prob, &opts);
            if rep.closed() {
                ranked_closed += 1;
                for table in [FdTable::Full, FdTable::Refused("table_cap")] {
                    let mut r = rep.clone();
                    let trace = fd_proof_trace_for_test(
                        &ranked.state,
                        &frame,
                        &order,
                        &prob,
                        &mut r,
                        table,
                    );
                    assert!(
                        !trace.complete(),
                        "{} class {class}: a ranked compact state's trace without its class \
                         relaxation must never be complete",
                        f.name
                    );
                }
            }
            let full = state_of(f, &frame, class);
            let full_order = static_order(&full);
            let depth = full_order.len().min(FROZEN_DUAL_MAX_DEPTH);
            let fprob = FrozenDualProblem::new(&full, &full_order[..depth]).expect("problem");
            let mut frep = search(&fprob, &opts);
            if frep.closed() {
                full_closed += 1;
                let trace = fd_proof_trace_for_test(
                    &full,
                    &frame,
                    &full_order[..depth],
                    &fprob,
                    &mut frep,
                    FdTable::Full,
                );
                assert!(
                    trace.complete(),
                    "{} class {class}: the full state's closed trace is complete",
                    f.name
                );
                assert_eq!(trace.relaxed(), full.num_unstable());
            }
        }
    }
    assert!(ranked_closed > 0, "VOID: no ranked compact search closed");
    assert!(full_closed > 0, "VOID: no full-state search closed");
}

/// The (layer, position) of every unstable symbol a ranked state OMITS.
fn omitted_positions(frame: &FrozenDualFrame, st: &FrozenDualState) -> Vec<(usize, usize)> {
    let retained: BTreeSet<(usize, usize)> = (0..st.num_unstable())
        .map(|k| (st.layer[k], frame.layers[st.layer[k]].pos[st.neuron[k]]))
        .collect();
    frame
        .layers
        .iter()
        .enumerate()
        .flat_map(|(li, l)| (0..l.unst.len()).map(move |p| (li, p)))
        .filter(|x| !retained.contains(x))
        .collect()
}

/// `#fd-compact-checkable`: the class relaxation is RE-DERIVED, never
/// trusted. Each mutation is refused by name and a refused table leaves the
/// trace incomplete:
///
/// * one OMITTED row's entry dropped -> `table_shape`;
/// * one OMITTED row's slope perturbed, on a symbol the margin reads
///   (nonzero coefficient) -> `table_mismatch` (the margin row replayed from
///   the table alone is no longer the state's);
/// * one RETAINED row's slope perturbed -> `table_row`;
/// * the WRONG class: relabelled -> `table_class`; another class's real
///   table -> `table_class`, and under this class's label `table_row` or
///   `table_mismatch` (whenever its slopes differ).
///
/// VOID unless every mutation kind was exercised.
#[test]
fn fd_compact_class_relaxation_mutations_are_refused() {
    let mut exercised = [0usize; 5];
    for f in &sweep() {
        let frame = frame_of(f);
        if frame.n_unstable < 3 || f.adv.len() < 2 {
            continue;
        }
        let keep = frame.n_unstable - 2;
        let opts = FrozenDualOptions::default();
        for &class in &f.adv {
            let (ranked, pristine) =
                FrozenDualState::build_ranked_with_relax(&f.net, &frame, class, &opts, keep)
                    .expect("ranked state");
            let st = &ranked.state;
            let verify = |mut t: CompactRelax| t.verify(&f.net, &frame, st, None);
            assert_eq!(verify(pristine.clone()), Ok(()));
            let omitted = omitted_positions(&frame, st);
            assert_eq!(omitted.len(), frame.n_unstable - st.num_unstable());
            assert!(omitted.len() >= 2);
            // (1) one omitted row's entry dropped.
            let (li, p) = omitted[0];
            let mut t = pristine.clone();
            t.drop_entry_for_test(li, p);
            assert_eq!(verify(t), Err("table_shape"), "{}", f.name);
            exercised[0] += 1;
            // (2) one omitted slope the margin reads, perturbed.
            if let Some(&(li, p)) = omitted
                .iter()
                .find(|&&(li, p)| st.d[frame.layers[li].col0 + p] != 0.0)
            {
                let (lam, _) = pristine.slope(li, p).expect("in the table");
                let moved = if lam > 0.5 { lam - 0.25 } else { lam + 0.25 };
                let mut t = pristine.clone();
                t.perturb_slope_for_test(li, p, moved);
                assert_eq!(verify(t), Err("table_mismatch"), "{}", f.name);
                exercised[1] += 1;
            }
            // (3) one retained row's slope perturbed.
            let li = st.layer[0];
            let p = frame.layers[li].pos[st.neuron[0]];
            let mut t = pristine.clone();
            t.perturb_slope_for_test(li, p, if st.lam[0] > 0.5 { 0.0 } else { 1.0 });
            assert_eq!(verify(t), Err("table_row"), "{}", f.name);
            exercised[2] += 1;
            // (4) the wrong class.
            for &other in f.adv.iter().filter(|&&c| c != class) {
                let mut t = pristine.clone();
                t.relabel_for_test(other);
                assert_eq!(verify(t), Err("table_class"), "{}", f.name);
                exercised[3] += 1;
                let (_, theirs) =
                    FrozenDualState::build_ranked_with_relax(&f.net, &frame, other, &opts, keep)
                        .expect("ranked state");
                assert_eq!(theirs.class_for_test(), other);
                assert_eq!(verify(theirs.clone()), Err("table_class"), "{}", f.name);
                let differs = frame.layers.iter().enumerate().any(|(li, l)| {
                    (0..l.unst.len()).any(|p| {
                        theirs.slope(li, p).map(|(a, _)| a.to_bits())
                            != pristine.slope(li, p).map(|(a, _)| a.to_bits())
                    })
                });
                if differs {
                    let mut t = theirs;
                    t.relabel_for_test(class);
                    let got = verify(t);
                    assert!(
                        matches!(got, Err("table_row" | "table_mismatch" | "table_choice")),
                        "{} class {class}: class {other}'s slopes under class {class}'s label: {got:?}",
                        f.name
                    );
                    exercised[4] += 1;
                }
            }
        }
    }
    assert!(
        exercised.iter().all(|&n| n > 0),
        "VOID: a mutation kind was never exercised {exercised:?}"
    );
}

/// Review of `#fd-compact-checkable`: the margin-row replay
/// (`table_mismatch`) is BLIND to a table entry whose neuron has margin
/// coefficient exactly zero — `lambda * 0` and `0 * mu` leave
/// `(c_obj, d, r_obj)` bit-identical whatever the entry says — while the
/// retained rows of the layers above that neuron read its slope. Every such
/// OMITTED entry, perturbed, must be refused by the re-choice
/// (`table_choice`): the margin pass re-run from the chord start writes the
/// build's own slope there.
///
/// How a zero coefficient arises on a real net: the margin pass gives an
/// unstable neuron with a POSITIVE incoming coefficient its lower slope
/// `alpha`, which the area heuristic sets to exactly 0 whenever `-l > u`; a
/// neuron every downstream path of which runs through such slopes (or dead
/// neurons) gets coefficient 0, while the ROW of each such downstream neuron
/// (seeded at its pre-activation, before the slope applies) still reads it.
///
/// Fixtures: the sweep (natural zero coefficients, counted, not required)
/// and LAZY-HEAD nets: the sweep's nets with the final Gemm made
/// `y_0 = 1 + sum h`, `y_c = 0` (class 0 is the true class and every head
/// coefficient is +1) and the head Gemm's bias shifted down, so that on some
/// boxes every head neuron is dead or unstable with `-l > u` (slope 0): then
/// EVERY trunk symbol is blind, while every retained head row reads every
/// trunk-relu-1 slope through its own head-Gemm weight. "Read" is exact: a
/// retained head row `m` and an omitted blind trunk-relu-1 entry `j` with
/// `W1[m, j] != 0`. VOID unless such a blind entry was refused.
#[test]
fn fd_compact_class_relaxation_entry_the_margin_does_not_read_is_refused() {
    let mut fixtures = sweep();
    for seed in [3u64, 5, 11, 17] {
        for half in [0.3, 0.15, 0.08] {
            for shift in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8] {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut spec = tiny_spec(&mut rng, 0.4);
                let TwinOpSpec::Gemm { bias, shape, .. } = &mut spec.ops[6] else {
                    panic!("the tiny spec's op 6 is the head Gemm");
                };
                assert_eq!(*shape, (5, 36));
                bias.iter_mut().for_each(|b| *b -= shift);
                let Some(TwinOpSpec::Gemm {
                    weight,
                    bias,
                    shape,
                    ..
                }) = spec.ops.last_mut()
                else {
                    panic!("the tiny spec ends in its final Gemm");
                };
                assert_eq!(*shape, (4, 5));
                for (k, w) in weight.iter_mut().enumerate() {
                    *w = if k < 5 { 1.0 } else { 0.0 };
                }
                for (k, b) in bias.iter_mut().enumerate() {
                    *b = if k == 0 { 1.0 } else { 0.0 };
                }
                fixtures.push(fx(
                    &format!("lazy_head{seed}_h{half}_s{shift}"),
                    spec,
                    half,
                    seed,
                    false,
                ));
            }
        }
    }
    let opts = FrozenDualOptions::default();
    let (mut natural, mut read_by_rows, mut refused) = (0usize, 0usize, 0usize);
    for f in &fixtures {
        let frame = frame_of(f);
        if frame.n_unstable < 3 {
            continue;
        }
        let w1 = match &f.spec.ops[6] {
            TwinOpSpec::Gemm { weight, .. }
                if frame.layers.len() == 3 && frame.layers[1].l.len() == 36 =>
            {
                Some(weight.clone())
            }
            _ => None,
        };
        let n = frame.n_unstable;
        let mut keeps = vec![n - 1, n - 2, (3 * n) / 4, n / 2, n / 4, 1];
        keeps.retain(|&k| k >= 1 && k < n);
        keeps.sort_unstable();
        keeps.dedup();
        for &class in &f.adv {
            for &keep in &keeps {
                let (ranked, pristine) =
                    FrozenDualState::build_ranked_with_relax(&f.net, &frame, class, &opts, keep)
                        .expect("ranked state");
                let st = &ranked.state;
                assert_eq!(
                    pristine.clone().verify(&f.net, &frame, st, None),
                    Ok(()),
                    "{} class {class} keep {keep}: the pristine table verifies",
                    f.name
                );
                let blind: Vec<(usize, usize)> = omitted_positions(&frame, st)
                    .into_iter()
                    .filter(|&(li, p)| st.d[frame.layers[li].col0 + p] == 0.0)
                    .collect();
                for &(li, p) in &blind {
                    let (lam, _) = pristine.slope(li, p).expect("in the table");
                    let moved = if lam > 0.5 { lam - 0.25 } else { lam + 0.25 };
                    let mut t = pristine.clone();
                    t.perturb_slope_for_test(li, p, moved);
                    assert_eq!(
                        t.verify(&f.net, &frame, st, None),
                        Err("table_choice"),
                        "{} class {class} keep {keep}: a blind entry ({li}, {p}) perturbed \
                         {lam} -> {moved}",
                        f.name
                    );
                    refused += 1;
                    if !f.name.starts_with("lazy_head") {
                        natural += 1;
                    }
                    if let (Some(w1), 1) = (&w1, li) {
                        let j = frame.layers[1].unst[p];
                        let read = (0..st.num_unstable())
                            .any(|k| st.layer[k] == 2 && w1[st.neuron[k] * 36 + j] != 0.0);
                        if read {
                            read_by_rows += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "[fd-compact-review] blind entries refused={refused} natural={natural} \
         read_by_retained_rows={read_by_rows}"
    );
    assert!(
        read_by_rows > 0,
        "VOID: no blind entry that a retained row reads was exercised (refused {refused})"
    );
}

/// The frame the tier (and the lane) builds: over the root evaluation's own
/// head box, exactly as `run_tier` is called by [`tier`].
fn tier_frame(f: &Fx) -> FrozenDualFrame {
    let eng = BackwardEngine::new(&f.net, &f.root);
    let re = root_eval(&eng, &f.net, f.t, &f.adv).expect("root eval");
    FrozenDualFrame::from_root(&f.net, &f.root, &re.root_ybox(), f.t).expect("frame")
}

/// [`tier`] with per-class states built RANKED above `threshold` unstable
/// rows and the compact table cap forced to `table_cap` (`None`: the real
/// cap).
fn tier_compact(f: &Fx, capture: bool, threshold: usize, table_cap: Option<u64>) -> TierReport {
    force_fd_streamed_threshold_for_test(Some(threshold));
    force_fd_proof_table_cap_for_test(table_cap);
    let report = tier(f, capture);
    force_fd_streamed_threshold_for_test(None);
    force_fd_proof_table_cap_for_test(None);
    report
}

/// `#fd-compact-checkable` through the TIER (`run_class`, the threshold
/// lowered by the test hook so the sweep's frames build compact states):
/// recording the class relaxation moves nothing the cover reads — every
/// class's search report is the same bits with capture on and off, capture
/// off builds no trace and records nothing — and under capture every
/// compact class the tier closes carries a COMPLETE trace whose table covers
/// every relaxation symbol of the frame. VOID unless a compact class closes.
#[test]
fn fd_tier_compact_traces_are_complete_and_the_search_is_bit_identical() {
    let mut exercised = 0usize;
    for f in &sweep() {
        let frame = tier_frame(f);
        if frame.n_unstable < 2 {
            continue;
        }
        let th = frame.n_unstable - 1;
        let off = tier_compact(f, false, th, None);
        let on = tier_compact(f, true, th, None);
        assert_eq!(off.closed, on.closed, "{}", f.name);
        assert_eq!(off.nodes, on.nodes, "{}", f.name);
        let (a, b) = (runs(&off), runs(&on));
        assert_eq!(a.len(), b.len(), "{}", f.name);
        for ((ca, ra), (cb, rb)) in a.iter().zip(&b) {
            assert_eq!(ca, cb);
            assert!(
                ra.streamed_rows && rb.streamed_rows,
                "{}: not compact",
                f.name
            );
            assert!(ra.proof.is_none(), "capture off must build no trace");
            assert!(ra.search.pruned.is_empty(), "capture off recorded prunes");
            assert_eq!(
                search_digest(&ra.search),
                search_digest(&rb.search),
                "{} class {ca}: recording the class relaxation moved the search",
                f.name
            );
            assert_eq!(ra.root_g.to_bits(), rb.root_g.to_bits());
            assert_eq!(ra.depth_cap, rb.depth_cap);
            let proof = rb
                .proof
                .as_ref()
                .expect("capture on builds every class's trace");
            if rb.search.closed() {
                assert!(
                    proof.complete(),
                    "{} class {cb}: a closed compact class's trace is incomplete",
                    f.name
                );
                assert_eq!(proof.relaxed(), frame.n_unstable, "{}", f.name);
                exercised += 1;
            }
        }
    }
    assert!(exercised > 0, "VOID: no compact class closed in the tier");
}

/// `#fd-compact-checkable`: a compact class whose class relaxation does not
/// fit the table cap (forced by the test hook) is never recorded — its trace
/// is incomplete, at a cap of one byte below the table's size, while the
/// search is the uncapped search bit for bit — and at exactly the table's
/// size it is complete. VOID unless a compact class closes.
#[test]
fn fd_compact_trace_past_the_table_cap_is_incomplete() {
    let mut exercised = 0usize;
    for f in &sweep() {
        let frame = tier_frame(f);
        if frame.n_unstable < 2 {
            continue;
        }
        let th = frame.n_unstable - 1;
        let need = fd_proof_table_bytes(&frame);
        let at = tier_compact(f, true, th, Some(need));
        let below = tier_compact(f, true, th, Some(need - 1));
        for ((ca, ra), (cb, rb)) in runs(&at).iter().zip(&runs(&below)) {
            assert_eq!(ca, cb);
            assert_eq!(
                search_digest(&ra.search),
                search_digest(&rb.search),
                "{}: the table cap moved the search",
                f.name
            );
            if !ra.search.closed() {
                continue;
            }
            let (pa, pb) = (
                ra.proof.as_ref().expect("trace"),
                rb.proof.as_ref().expect("trace"),
            );
            assert!(
                pa.complete(),
                "{} class {ca}: complete at the table's size",
                f.name
            );
            assert!(
                !pb.complete(),
                "{} class {cb}: a table past its cap must leave the trace incomplete",
                f.name
            );
            exercised += 1;
        }
    }
    assert!(exercised > 0, "VOID: no compact class closed");
}

/// `#fd-compact-checkable` through the LANE (the tier armed by the harness
/// field, compact states forced by the threshold hook): an outer frame whose
/// class the tier closes on a COMPACT state streams that class's prunes —
/// its `hint.fd_state` covers more relaxation symbols than the state has
/// rows — instead of the `close: fd` claim, and the verdict is bit-identical
/// with capture off. VOID unless such a class streams.
#[test]
fn margin_row_fd_tier_streams_compact_prunes_through_the_frame() {
    let mut exercised = 0usize;
    for f in &sweep() {
        let frame = tier_frame(f);
        if frame.n_unstable < 2 {
            continue;
        }
        let cfg = || BabConfig {
            frontier: 1,
            frontier_contextual: 1,
            lru_cap: 64,
            max_expansions: 400,
            frozen_dual_tier: Some(true),
            ..BabConfig::default()
        };
        let run = |capture: bool| {
            let sink = Shared::default();
            force_hint_capture_for_test(Some(capture));
            force_fd_streamed_threshold_for_test(Some(frame.n_unstable - 1));
            let (out, _) = with_thread_sink(Box::new(sink.clone()), || {
                MarginRowBab::run(&f.net, &f.root, f.t, &f.adv, cfg())
            });
            force_fd_streamed_threshold_for_test(None);
            force_hint_capture_for_test(None);
            let rec = sink.0.lock().expect("lock").clone();
            (out, rec)
        };
        let (off, off_rec) = run(false);
        assert!(off_rec.nodes.is_empty() && off_rec.fd_states.is_empty());
        let (on, rec) = run(true);
        assert_eq!(
            verdict_bits(&off),
            verdict_bits(&on),
            "{}: capture moved the verdict",
            f.name
        );
        if !matches!(on, MarginRowOutcome::Unsat(..)) {
            continue;
        }
        for st in rec.fd_states.iter().filter(|s| s.relaxed > s.unstable) {
            let class = st.class;
            let entries: usize = st.lambda.iter().map(|l| l.idx.len()).sum();
            assert_eq!(
                entries, st.relaxed,
                "{}: the table is the relaxed count",
                f.name
            );
            assert!(
                !rec.nodes
                    .iter()
                    .any(|n| n.close == NodeClose::Fd && n.dis.as_deref() == Some(&[class][..])),
                "{}: class {class} streamed its compact prunes AND its claim",
                f.name
            );
            assert!(rec
                .nodes
                .iter()
                .any(|n| n.src == "fd_prune" && n.dis.as_deref() == Some(&[class][..])));
            exercised += 1;
        }
    }
    assert!(
        exercised > 0,
        "VOID: no lane run streamed a compact frozen-dual class"
    );
}

/// [`tier`] under capture with the per-class record cap forced to `cap`.
fn tier_capped(f: &Fx, cap: usize) -> TierReport {
    force_fd_proof_record_cap_for_test(Some(cap));
    let report = tier(f, true);
    force_fd_proof_record_cap_for_test(None);
    report
}

/// The trace of `class` in `report` (capture on builds one per searched
/// class).
fn trace_of(report: &TierReport, class: usize) -> &crate::certify::FdProofTrace {
    runs(report)
        .into_iter()
        .find(|(c, _)| *c == class)
        .and_then(|(_, r)| r.proof.as_deref())
        .expect("capture on builds every searched class's trace")
}

#[test]
fn fd_prune_trace_past_the_record_cap_is_incomplete() {
    let mut incomplete = 0usize;
    let mut boundaries = 0usize;
    for f in &sweep() {
        let full = tier(f, true);
        // Classes closed by SPLITTING (a root-closed class records nothing
        // and needs no record): (class, certified prunes).
        let split: Vec<(usize, u64)> = runs(&full)
            .into_iter()
            .filter(|(_, r)| r.search.closed() && !r.search.levels.is_empty())
            .map(|(c, r)| (c, r.search.certified_prunes))
            .collect();
        if split.is_empty() {
            continue;
        }
        for &(class, prunes) in &split {
            assert!(
                prunes >= 2,
                "{} class {class}: a split tree has >= 2 leaves",
                f.name
            );
            assert!(
                trace_of(&full, class).complete(),
                "{} class {class}: uncapped, a closed class's trace is complete",
                f.name
            );
        }
        // A cap of one: every split-closed class's trace is incomplete, a
        // root-closed class's stays complete, and the search is unmoved.
        let one = tier_capped(f, 1);
        let (a, b) = (runs(&full), runs(&one));
        assert_eq!(a.len(), b.len(), "{}", f.name);
        for ((ca, ra), (cb, rb)) in a.iter().zip(&b) {
            assert_eq!(ca, cb);
            assert_eq!(
                search_digest(&ra.search),
                search_digest(&rb.search),
                "{} class {ca}: a capped recorder moved the search",
                f.name
            );
            if !rb.search.closed() {
                continue;
            }
            let proof = rb.proof.as_ref().expect("capture on builds a trace");
            if rb.search.levels.is_empty() {
                assert!(proof.complete(), "{} class {cb}: root-closed", f.name);
            } else {
                assert!(
                    !proof.complete(),
                    "{} class {cb}: {} certified prunes, 1 recorded, and the trace claims \
                     to be complete",
                    f.name,
                    rb.search.certified_prunes
                );
                incomplete += 1;
            }
        }
        // The boundary, on the class with the most prunes: one short of its
        // prune count is incomplete, exactly its prune count is complete.
        let (class, prunes) = *split.iter().max_by_key(|s| s.1).expect("non-empty");
        let cap = usize::try_from(prunes).expect("prune count fits usize");
        assert!(
            !trace_of(&tier_capped(f, cap - 1), class).complete(),
            "{} class {class}: {} of {prunes} prunes recorded, claimed complete",
            f.name,
            cap - 1
        );
        assert!(
            trace_of(&tier_capped(f, cap), class).complete(),
            "{} class {class}: every one of {prunes} prunes recorded, yet incomplete",
            f.name
        );
        boundaries += 1;
    }
    assert!(
        incomplete > 0 && boundaries > 0,
        "no class of the sweep closed by splitting under capture: VOID"
    );
}

#[test]
fn margin_row_fd_trace_past_the_record_cap_streams_the_claim_not_its_prunes() {
    let mut exercised = 0usize;
    for f in &sweep() {
        let cfg = || BabConfig {
            frontier: 1,
            frontier_contextual: 1,
            lru_cap: 64,
            max_expansions: 400,
            frozen_dual_tier: Some(true),
            ..BabConfig::default()
        };
        let run = |cap: Option<usize>| {
            let sink = Shared::default();
            force_hint_capture_for_test(Some(true));
            force_fd_proof_record_cap_for_test(cap);
            let (out, _) = with_thread_sink(Box::new(sink.clone()), || {
                MarginRowBab::run(&f.net, &f.root, f.t, &f.adv, cfg())
            });
            force_fd_proof_record_cap_for_test(None);
            force_hint_capture_for_test(None);
            let rec = sink.0.lock().expect("lock").clone();
            (out, rec)
        };
        let (full, full_rec) = run(None);
        if !matches!(full, MarginRowOutcome::Unsat(..)) {
            continue;
        }
        // Classes the uncapped stream proved prune by prune (>= 2 prunes:
        // closed by splitting; a root-closed class streams one node).
        let split: Vec<usize> = full_rec
            .fd_states
            .iter()
            .filter(|s| s.prunes >= 2)
            .map(|s| s.class)
            .collect();
        if split.is_empty() {
            continue;
        }
        let (capped, rec) = run(Some(1));
        assert_eq!(
            verdict_bits(&full),
            verdict_bits(&capped),
            "{}: the record cap moved the verdict",
            f.name
        );
        for class in &split {
            let dis = Some(&[*class][..]);
            assert!(
                !rec.fd_states.iter().any(|s| s.class == *class),
                "{}: class {class} streamed a hint.fd_state past the record cap",
                f.name
            );
            let prunes = rec
                .nodes
                .iter()
                .filter(|n| n.src == "fd_prune" && n.dis.as_deref() == dis)
                .count();
            assert_eq!(
                prunes, 0,
                "{}: class {class} streamed {prunes} fd_prune node(s) of an incomplete trace \
                 as its certificate",
                f.name
            );
            assert!(
                rec.nodes
                    .iter()
                    .any(|n| n.close == NodeClose::Fd && n.dis.as_deref() == dis),
                "{}: class {class} lost its close: fd claim",
                f.name
            );
        }
        exercised += 1;
    }
    assert!(
        exercised > 0,
        "no lane run closed a class by splitting through the frozen-dual tier: VOID"
    );
}
