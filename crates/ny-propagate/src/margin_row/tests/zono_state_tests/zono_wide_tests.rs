// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! `#wide-fd-search` (ENGINE_PROGRAM M4) tests: the frozen-dual tier's level
//! search on an injected UNTRUSTED backend, certified in f64. A child of
//! `zono_state_tests` (fixtures, `sample`, the exact-rational node value) and a
//! sibling of `zono_bab_tests` (sweep, problem and option helpers).
//!
//! The backends here are test doubles: the parent's REFERENCE backend (the f64
//! search step itself, `ReferenceBackend`) and ROGUE backends that pass the
//! lane-start probe honestly and then return garbage, refuse, panic, stop on
//! the deadline or return a malformed level.
//!
//! Bit identity of the default:
//! * `fd_wide_default_is_the_f64_path_bit_identical`;
//! * `fd_wide_tier_without_a_registered_backend_runs_f64`.
//!
//! Soundness (fault injection — garbage never creates a close):
//! * `fd_wide_garbage_backends_never_close_a_witnessed_sat_class`;
//! * `fd_wide_prunes_reverify_in_exact_rationals` (every prune of the wide
//!   path, honest or garbage backend, is `exact >= certified > 0`, and the f64
//!   path certifies the same bits at the same point);
//! * `fd_wide_closed_classes_hold_at_sampled_points` (containment of what the
//!   wide path closes; VOID if nothing closes by splitting).
//!
//! Mutation pins:
//! * `fd_wide_charge_is_load_bearing` (charge off: the raw-positive state
//!   closes; on: every claim is refuted and the whole tree is walked);
//! * `fd_wide_claims_are_not_prunes` (the claim-as-prune mutant closes a SAT
//!   class; the real path does not; the cross-check catches the mutant);
//! * `fd_wide_crosscheck_catches_injected_fault`;
//! * `fd_wide_prunes_only_through_the_certified_predicate` (source ratchet).
//!
//! The protocol: `fd_wide_reference_backend_reproduces_the_f64_tree`,
//! `fd_wide_search_is_thread_count_independent`,
//! `fd_wide_refusals_panics_and_malformed_levels_fall_back_to_f64`,
//! `fd_wide_probe_rejects_a_lying_backend`,
//! `fd_wide_deadline_refusal_stops_open`,
//! `fd_wide_tier_auto_falls_through_a_refusing_backend`,
//! `fd_wide_search_copy_is_the_problem_rounded`,
//! `fd_wide_search_rules_match_the_f64_path`.
//!
//! Composition with the M3 shared trunk (`frozen_dual_bab_state: shared`): the
//! trunk decides each class's STATE, the backend only scores the levels of
//! the problem substituted from it (every backend built per class from that
//! class's own problem, `WideTier::with_backend_factory`):
//! * `fd_wide_policy_state_and_search_are_independent_selectors` (both
//!   defaults are the M1 tier; neither builder resets the other);
//! * `fd_wide_shared_tier_reference_backend_reproduces_the_shared_f64_tier`
//!   (the state side bit for bit, each class's backend handed its own
//!   problem once, the shared f64 tier's trees);
//! * `fd_wide_shared_tier_garbage_backends_never_close_a_witnessed_sat_class`;
//! * `fd_wide_shared_tier_closed_classes_hold_at_sampled_points` (positive
//!   control; VOID if nothing closes by splitting on the wide path);
//! * `fd_wide_shared_tier_bindings_refuse_before_any_backend` (mutation pins:
//!   a swapped tail, a moved frame and corrupted trunk rows are refused by the
//!   M3 assembly on the f64 AND the wide search, and no backend is handed the
//!   refused class);
//! * `fd_wide_shared_tier_is_thread_count_independent`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ny_core::search::{
    BackendCaps, DeviceLease, DualMultipliers, DualSearchOutcome, DualSearchRequest, FdHandle,
    FdSearchProblem, LevelChunk, LevelOut, LevelStats, Refusal, SearchBackend, SearchPrecision,
    Untrusted, FD_LOG_BUCKETS, FD_POINTS, FD_SEARCH_TOL, FD_WARM,
};

use super::zono_bab_tests::{open_problem, opts, problem_of, raw_positive_state, sweep};
use super::*;

use crate::margin_row::spec::TwinOpSpec;
use crate::margin_row::zono_bab::{
    run_tier, run_tier_with_wide, search, search_copy, search_wide_with_test_hooks,
    search_with_backend, wide_search_rules, wide_tau_rel, Candidate, Evaluator, FrozenDualSearch,
    FrozenDualStateMode, LevelTrace, OpenReason, ReferenceBackend, SearchOptions, SearchReport,
    SearchVerdict, TierMutant, TierPolicy, TierReport, WideTier,
};

// ---------------------------------------------------------------------------
//  Test doubles
// ---------------------------------------------------------------------------

/// What a rogue backend does AFTER answering the probe honestly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum After {
    Honest,
    /// Every point claims (`+1e30`), every multiplier is poison (NaN, inf,
    /// negative, huge or random).
    ClaimAll,
    /// Every value, sum and multiplier NaN.
    Nan,
    /// Random values, sums and multipliers (some NaN, some negative).
    Random,
    /// Every point claims `+1`; every multiplier 0.
    ZeroMultipliers,
    /// Every value, sum and multiplier `f32::MAX`.
    Huge,
    Panic,
    Deadline,
    Malformed,
    Refuse,
}

const GARBAGE: [After; 5] = [
    After::ClaimAll,
    After::Nan,
    After::Random,
    After::ZeroMultipliers,
    After::Huge,
];

/// A backend that passes the lane-start probe (its first `fd_level`)
/// honestly — or, with `lie_in_probe`, dishonestly — then does `after`.
struct Rogue {
    honest: ReferenceBackend,
    after: After,
    lie_in_probe: bool,
    refuse_upload: bool,
    calls: AtomicUsize,
}

impl Rogue {
    fn new(prob: &FrozenDualProblem, after: After) -> Self {
        Self {
            honest: ReferenceBackend::new(prob),
            after,
            lie_in_probe: false,
            refuse_upload: false,
            calls: AtomicUsize::new(0),
        }
    }
}

/// splitmix64.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A per-node seed from the node's own content (so the garbage does not depend
/// on how the lane chunks a level or on the thread count).
fn node_seed(chunk: &LevelChunk<'_>, k: usize, salt: u64) -> u64 {
    let d = chunk.depth;
    let mut h = 0xCBF2_9CE4_8422_2325u64 ^ salt;
    for i in k * d..(k + 1) * d {
        h = mix(h ^ u64::from(chunk.sides[i]));
        h = mix(h ^ u64::from(chunk.l0[i].to_bits()));
        h = mix(h ^ (u64::from(chunk.l1[i].to_bits()) << 1));
    }
    h
}

fn unit(state: &mut u64) -> f32 {
    *state = mix(*state);
    #[allow(clippy::cast_precision_loss)]
    let u = (*state >> 40) as f32 / (1u64 << 24) as f32;
    u
}

fn poison(state: &mut u64) -> f32 {
    let u = unit(state);
    match (u * 8.0) as u32 {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => -1.0,
        3 => 1e30,
        4 => -0.0,
        _ => 4.0 * unit(state),
    }
}

fn fill_garbage(after: After, chunk: &LevelChunk<'_>, out: &mut LevelOut) {
    let (count, d) = (chunk.count, chunk.depth);
    out.prepare(count, d);
    let w = out.writer();
    for k in 0..count {
        let mut st = node_seed(chunk, k, after as u64);
        for p in 0..FD_POINTS {
            let (v, s) = match after {
                After::ClaimAll => (1e30, 1.0),
                After::Nan => (f32::NAN, f32::NAN),
                After::Random => (4.0 * unit(&mut st) - 2.0, 3.0 * unit(&mut st)),
                After::ZeroMultipliers => (1.0, 1.0),
                _ => (f32::MAX, f32::MAX),
            };
            w.value[k * FD_POINTS + p] = v;
            w.scale[k * FD_POINTS + p] = s;
        }
        for i in 0..2 * d {
            let at = k * 2 * d + i;
            let (c, f, n) = match after {
                After::ClaimAll => (poison(&mut st), poison(&mut st), poison(&mut st)),
                After::Nan => (f32::NAN, f32::NAN, f32::NAN),
                After::Random => {
                    let mut r = || {
                        let u = unit(&mut st);
                        if u < 0.125 {
                            f32::NAN
                        } else if u < 0.25 {
                            -u
                        } else {
                            8.0 * u
                        }
                    };
                    (r(), r(), r())
                }
                After::ZeroMultipliers => (0.0, 0.0, 0.0),
                _ => (f32::MAX, f32::MAX, f32::MAX),
            };
            w.clamped[at] = c;
            w.far[at] = f;
            w.next[at] = n;
        }
    }
}

impl SearchBackend for Rogue {
    fn name(&self) -> &'static str {
        "rogue"
    }

    fn caps(&self) -> BackendCaps {
        BackendCaps::Cpu {
            threads: 1,
            lanes: 1,
            precision: SearchPrecision::F32,
        }
    }

    fn upload_fd(
        &self,
        problem: &FdSearchProblem,
        lease: &DeviceLease,
    ) -> Result<FdHandle, Refusal> {
        if self.refuse_upload {
            return Err(Refusal::Unsupported("rogue upload"));
        }
        self.honest.upload_fd(problem, lease)
    }

    fn fd_level(
        &self,
        handle: &FdHandle,
        chunk: &LevelChunk<'_>,
        out: &mut LevelOut,
    ) -> Result<Untrusted<LevelStats>, Refusal> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if call == 0 {
            let stats = self.honest.fd_level(handle, chunk, out)?;
            if self.lie_in_probe {
                let w = out.writer();
                for k in 0..w.count {
                    let s = w.scale[k * FD_POINTS + FD_WARM];
                    w.value[k * FD_POINTS + FD_WARM] += s.max(1.0);
                }
            }
            return Ok(stats);
        }
        match self.after {
            After::Honest => self.honest.fd_level(handle, chunk, out),
            After::Panic => panic!("rogue backend panics after its probe"),
            After::Deadline => Err(Refusal::Deadline),
            After::Refuse => Err(Refusal::Invalid("rogue refuses after its probe".into())),
            After::Malformed => {
                out.prepare(chunk.count + 1, chunk.depth);
                Ok(Untrusted::new(LevelStats::default()))
            }
            garbage => {
                fill_garbage(garbage, chunk, out);
                Ok(Untrusted::new(LevelStats::default()))
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------

fn midpoint(lo: &[f64], hi: &[f64]) -> Vec<f64> {
    lo.iter()
        .zip(hi)
        .map(|(l, h)| f64::midpoint(*l, *h))
        .collect()
}

fn dd_margin(net: &TwinNet, t: usize, j: usize, x: &[f64]) -> f64 {
    let vals = dd_forward(net, x).expect("dd forward");
    let logits = &vals[net.ops.len()];
    (logits[t] - logits[j]).to_f64()
}

fn top_two(net: &TwinNet, x: &[f64]) -> (usize, usize) {
    let vals = dd_forward(net, x).expect("dd forward");
    let logits: Vec<f64> = vals[net.ops.len()].iter().map(|v| v.to_f64()).collect();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    (idx[0], idx[1])
}

/// Witnessed-sat objectives: swapped targets (the midpoint is a
/// counterexample) over the fixtures and part of the sweep, plus planted
/// violations (final bias moved so the midpoint violates by `delta`).
fn sat_problems() -> Vec<(String, FrozenDualProblem)> {
    let mut out = Vec::new();
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().step_by(3));
    for f in &fxs {
        let x0 = midpoint(&f.lo, &f.hi);
        let (top, second) = top_two(&f.net, &x0);
        let m = dd_margin(&f.net, second, top, &x0);
        if m < 0.0 {
            let frame = FrozenDualFrame::from_root(&f.net, &f.root, &f.ybox, second)
                .expect("swapped frame");
            let st = FrozenDualState::build(&f.net, &frame, top, &FrozenDualOptions::default())
                .expect("swapped state");
            out.push((
                format!("{} swapped {second} vs {top} ({m:e})", f.name),
                problem_of(&st),
            ));
        }
    }
    for seed in 0..6u64 {
        let half = [0.2, 0.08, 0.03][(seed % 3) as usize];
        let mut rng = StdRng::seed_from_u64(0x9A17 + seed);
        let spec0 = tiny_spec(&mut rng, 0.4);
        let f0 = fx("plant0", spec0.clone(), half, 0x9A17 + seed, false);
        let x0 = midpoint(&f0.lo, &f0.hi);
        let (t, j) = top_two(&f0.net, &x0);
        let m0 = dd_margin(&f0.net, t, j, &x0);
        for delta in [1e-3, 1e-9] {
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
            let frame =
                FrozenDualFrame::from_root(&f.net, &f.root, &f.ybox, t).expect("planted frame");
            let st = FrozenDualState::build(&f.net, &frame, j, &FrozenDualOptions::default())
                .expect("planted state");
            out.push((
                format!("planted seed {seed} delta {delta:e} ({m:e})"),
                problem_of(&st),
            ));
        }
    }
    out
}

/// Two classes of every stage-A fixture and of a slice of the sweep.
fn tree_problems() -> Vec<(String, FrozenDualProblem)> {
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().step_by(5));
    let mut out = Vec::new();
    for f in &fxs {
        let frame = frame_of(f);
        for &j in f.adv.iter().take(2) {
            out.push((
                format!("{} class {j}", f.name),
                problem_of(&state_of(f, &frame, j)),
            ));
        }
    }
    out
}

/// The tree (everything the wide path must reproduce from an honest backend).
#[derive(Debug, PartialEq)]
struct Tree {
    verdict: SearchVerdict,
    root_certified: u64,
    nodes: u64,
    depth: usize,
    certified_prunes: u64,
    peak_frontier: u64,
    levels: Vec<LevelTrace>,
}

fn tree(r: &SearchReport) -> Tree {
    Tree {
        verdict: r.verdict,
        root_certified: r.root_certified.to_bits(),
        nodes: r.nodes,
        depth: r.depth,
        certified_prunes: r.certified_prunes,
        peak_frontier: r.peak_frontier,
        levels: r.levels.clone(),
    }
}

/// The whole deterministic report (wall clock excluded).
#[allow(clippy::type_complexity)]
fn digest(
    r: &SearchReport,
) -> (
    Tree,
    u64,
    u64,
    u64,
    u64,
    Vec<(Vec<Side>, Vec<u64>, Vec<u64>, u64)>,
) {
    (
        tree(r),
        r.cert_min_slack.to_bits(),
        r.claimed,
        r.refuted,
        r.crosschecks,
        r.pruned
            .iter()
            .map(|p| {
                (
                    p.sides.clone(),
                    p.l0.iter().map(|v| v.to_bits()).collect(),
                    p.l1.iter().map(|v| v.to_bits()).collect(),
                    p.certified.to_bits(),
                )
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
//  Bit identity of the default
// ---------------------------------------------------------------------------

#[test]
fn fd_wide_default_is_the_f64_path_bit_identical() {
    assert_eq!(FrozenDualSearch::default(), FrozenDualSearch::F64);
    assert_eq!(
        ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH.default,
        ny_levers::DefaultSpec::Text(FrozenDualSearch::F64.token())
    );
    assert_eq!(
        TierPolicy::clamped(7, 1.0).search,
        FrozenDualSearch::F64,
        "the clamped policy must default to the f64 search"
    );
    if ny_levers::read(&ny_levers::decls::margin_row::MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH).source
        == ny_levers::Source::Default
    {
        assert_eq!(TierPolicy::from_gates().search, FrozenDualSearch::F64);
    }
    // The default search reports itself as f64 and never claims.
    let mut compared = 0usize;
    for (name, prob) in tree_problems() {
        let r = search(&prob, &opts(2_000, 0));
        assert_eq!(
            (r.search, r.probe, r.claimed, r.refuted, r.fallback_chunks),
            ("f64", "-", 0, 0, 0),
            "{name}: the default search touched the wide path"
        );
        compared += 1;
    }
    assert!(compared > 0, "VOID");
    // The default tier runs exactly `search` on every class (no backend).
    for f in fixtures() {
        let eng = BackwardEngine::new(&f.net, &f.root);
        let re = crate::margin_row::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("root eval");
        let tier = run_tier(
            &f.net,
            &f.root,
            f.t,
            &f.adv,
            &re.dj,
            &re.tree_classes,
            &re.root_ybox(),
            None,
            &TierPolicy::clamped(1_500, 5.0),
        );
        for class in &tier.classes {
            if let Ok(run) = &class.result {
                assert_eq!(run.search.search, "f64", "{}", f.name);
                assert_eq!(run.search.claimed, 0, "{}", f.name);
            }
        }
        assert_eq!((tier.claimed, tier.refuted), (0, 0), "{}", f.name);
    }
}

#[test]
fn fd_wide_tier_without_a_registered_backend_runs_f64() {
    // ny-propagate's tests register no level-search backend: an armed
    // `cpu_f32` tier must run every class on the f64 search, class for class
    // identical to the default tier.
    let mut open_classes = 0usize;
    for f in fixtures() {
        let eng = BackwardEngine::new(&f.net, &f.root);
        let re = crate::margin_row::bab::root_eval(&eng, &f.net, f.t, &f.adv).expect("root eval");
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
        let base = run(TierPolicy::clamped(1_500, 30.0));
        let armed = run(TierPolicy::clamped(1_500, 30.0).with_search(FrozenDualSearch::CpuF32));
        assert_eq!(base.closed, armed.closed, "{}", f.name);
        assert_eq!(base.nodes, armed.nodes, "{}", f.name);
        for (a, b) in base.classes.iter().zip(&armed.classes) {
            if let (Ok(x), Ok(y)) = (&a.result, &b.result) {
                assert_eq!(
                    tree(&x.search),
                    tree(&y.search),
                    "{} class {}",
                    f.name,
                    a.class
                );
                assert_eq!(y.search.search, "f64");
                if y.search.nodes > 0 {
                    assert_eq!(y.search.probe, "refused", "{}", f.name);
                    open_classes += 1;
                }
            }
        }
    }
    assert!(open_classes > 0, "VOID: no class reached a level");
}

// ---------------------------------------------------------------------------
//  Soundness: fault injection
// ---------------------------------------------------------------------------

#[test]
fn fd_wide_garbage_backends_never_close_a_witnessed_sat_class() {
    let problems = sat_problems();
    assert!(
        problems.len() >= 6,
        "only {} witnessed sat problems: VOID",
        problems.len()
    );
    let mut failures = Vec::new();
    let (mut engaged, mut claimed) = (0usize, 0u64);
    for (name, prob) in &problems {
        for after in std::iter::once(After::Honest).chain(GARBAGE) {
            let backend = Rogue::new(prob, after);
            let r = search_with_backend(prob, &opts(1_500, 0), &backend);
            if r.search == "rogue" {
                engaged += 1;
                claimed += r.claimed;
            }
            if r.closed() {
                failures.push(format!(
                    "{name}: backend {after:?} CLOSED a witnessed sat class after {} nodes",
                    r.nodes
                ));
            }
        }
        // The tier's own plumbing (WideTier) with a garbage backend.
        let mut tier = WideTier::with_backends(
            FrozenDualSearch::CpuF32,
            vec![Arc::new(Rogue::new(prob, After::ClaimAll)) as Arc<dyn SearchBackend>],
        );
        if tier.search_class(prob, &opts(1_500, 0)).closed() {
            failures.push(format!(
                "{name}: the tier path CLOSED a witnessed sat class"
            ));
        }
    }
    eprintln!(
        "[fd-wide-sat] problems={} engaged={engaged} claimed={claimed}",
        problems.len()
    );
    assert!(
        failures.is_empty(),
        "THE WIDE PATH CLOSED A SAT CLASS (soundness emergency): {failures:#?}"
    );
    assert!(
        engaged > problems.len() && claimed > 0,
        "VOID: the rogue backends never searched (engaged={engaged}, claimed={claimed})"
    );
}

#[test]
fn fd_wide_prunes_reverify_in_exact_rationals() {
    // The population and options of the f64 path's own exact re-verification
    // (`zono_bab_tests::prunes_reverify_in_exact_rationals`): every stage-A
    // fixture and the WHOLE sweep, EVERY adversarial class, `opts(3_000, 48)`.
    // Measured on int/eng3 84651b5a4: the f64 search records 337 prunes on it
    // (`[frozen-dual-bab-exact] eval=block prunes=337`), and the reference
    // backend walks the f64 search's tree on 36 of 36 `tree_problems()` runs
    // (`[fd-wide-ref] compared=36 identical=36`), so the honest backend hands
    // this test about as many prunes. The earlier population — the first
    // eight `tree_problems()` at `opts(2_000, 48)` — certified only 9 (5
    // honest, 4 garbage), under the guard below by construction. Those eight
    // problems are inside this population at a larger cap (a level-granular
    // cap and a deterministic, node-seeded garbage backend), so the 4 garbage
    // prunes are still re-verified here.
    //
    // Every prune is re-verified, but only a prune of a run the rogue backend
    // scored end to end (`search=rogue`, no chunk fell back to the f64
    // search) counts toward the two guards below: a backend that silently
    // fell back would otherwise satisfy them with the f64 search's prunes.
    let (mut checked, mut honest, mut garbage_prunes, mut problems, mut unengaged) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut scratch = NodeScratch::default();
    let mut fxs = fixtures();
    fxs.extend(sweep());
    for f in &fxs {
        let frame = frame_of(f);
        for &j in &f.adv {
            let name = format!("{} class {j}", f.name);
            let prob = problem_of(&state_of(f, &frame, j));
            problems += 1;
            for after in [After::Honest, After::ClaimAll, After::ZeroMultipliers] {
                let r = search_with_backend(&prob, &opts(3_000, 48), &Rogue::new(&prob, after));
                let engaged = r.search == "rogue" && r.fallback_chunks == 0;
                for p in &r.pruned {
                    assert!(p.certified > 0.0, "{name}: a record with certified <= 0");
                    let exact = exact_node_value(&prob, &p.sides, &p.l0, &p.l1);
                    assert!(
                        exact >= q(p.certified),
                        "{name} {after:?}: a wide-path prune certified {:e} above its exact \
                         value {:e} (FALSE UNSAT GENERATOR)",
                        p.certified,
                        exact.to_f64().unwrap_or(f64::NAN)
                    );
                    // The f64 path certifies the same bits at the same point.
                    let f64_path = prob.eval_node(&p.sides, &p.l0, &p.l1, &mut scratch, None);
                    assert_eq!(
                        f64_path.certified.to_bits(),
                        p.certified.to_bits(),
                        "{name} {after:?}: the certified value is not the f64 evaluator's"
                    );
                    checked += 1;
                    if !engaged {
                        unengaged += 1;
                    } else if after == After::Honest {
                        honest += 1;
                    } else {
                        garbage_prunes += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "[fd-wide-exact] problems={problems} checked={checked} honest={honest} \
         garbage_prunes={garbage_prunes} unengaged={unengaged}"
    );
    assert!(checked > 20, "VOID: only {checked} prunes re-verified");
    assert!(
        honest >= 100,
        "VOID: only {honest} honest-backend prunes re-verified on runs the backend scored (the \
         f64 search records 337 on this population; {unengaged} prunes came from runs that fell \
         back to it)"
    );
    assert!(
        garbage_prunes > 0,
        "VOID: no garbage backend's prune was re-verified on a run it scored ({unengaged} prunes \
         came from runs that fell back to the f64 search)"
    );
}

#[test]
fn fd_wide_closed_classes_hold_at_sampled_points() {
    let mut closed_by_split = 0usize;
    let mut failures = Vec::new();
    for f in &sweep() {
        let frame = frame_of(f);
        for &j in &f.adv {
            let prob = problem_of(&state_of(f, &frame, j));
            for after in [After::Honest, After::Random] {
                let r = search_with_backend(&prob, &opts(6_000, 0), &Rogue::new(&prob, after));
                if !r.closed() {
                    continue;
                }
                closed_by_split += usize::from(r.nodes > 0 && r.search == "rogue");
                let mut rng = StdRng::seed_from_u64(0xC105 ^ (j as u64) << 3);
                for s in 0..32 {
                    let x = sample(&mut rng, &f.lo, &f.hi, s);
                    let m = dd_margin(&f.net, f.t, j, &x);
                    if m <= 0.0 {
                        failures.push(format!(
                            "{} class {j} {after:?}: CLOSED but point {s} has margin {m:e}",
                            f.name
                        ));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "THE WIDE PATH CLOSED A CLASS WITH A COUNTEREXAMPLE: {failures:#?}"
    );
    assert!(
        closed_by_split > 0,
        "VOID: the wide path closed no class by splitting"
    );
}

// ---------------------------------------------------------------------------
//  Mutation pins
// ---------------------------------------------------------------------------

#[test]
fn fd_wide_charge_is_load_bearing() {
    // Raw > 0 at every node, certified <= 0 at every node: the honest backend
    // claims every node; the certificate must refute every claim.
    let st = raw_positive_state(6);
    let order: Vec<usize> = (0..st.num_unstable()).collect();
    let prob = FrozenDualProblem::new(&st, &order).expect("problem");
    let real = search_wide_with_test_hooks(
        &prob,
        &opts(100_000, 0),
        &ReferenceBackend::new(&prob),
        false,
        false,
        false,
    );
    assert_eq!(
        real.search, "reference_f64",
        "VOID: the backend did not search"
    );
    assert!(
        !real.closed(),
        "the wide path closed a raw-positive, certified-negative state"
    );
    assert_eq!(real.certified_prunes, 0);
    assert_eq!(
        real.nodes,
        (1u64 << 7) - 2,
        "a refuted node must be SPLIT: the whole tree is walked"
    );
    assert!(
        real.claimed > 0 && real.refuted == real.claimed,
        "every claim must be refuted (claimed {}, refuted {})",
        real.claimed,
        real.refuted
    );
    let mutant = search_wide_with_test_hooks(
        &prob,
        &SearchOptions {
            crosscheck_every: 0,
            ..opts(100_000, 0)
        },
        &ReferenceBackend::new(&prob),
        true,
        false,
        false,
    );
    assert!(
        mutant.closed(),
        "with the charge OFF the same wide path must close this state, else the pin is VOID \
         ({:?} after {} nodes)",
        mutant.verdict,
        mutant.nodes
    );
}

#[test]
fn fd_wide_claims_are_not_prunes() {
    let problems = sat_problems();
    let mut mutant_closed = 0usize;
    let mut caught = 0usize;
    for (name, prob) in &problems {
        let no_xchk = SearchOptions {
            crosscheck_every: 0,
            ..opts(1_500, 0)
        };
        let real = search_wide_with_test_hooks(
            prob,
            &no_xchk,
            &Rogue::new(prob, After::ClaimAll),
            false,
            false,
            false,
        );
        assert!(
            !real.closed(),
            "{name}: the real wide path closed a sat class"
        );
        let mutant = search_wide_with_test_hooks(
            prob,
            &no_xchk,
            &Rogue::new(prob, After::ClaimAll),
            false,
            false,
            true,
        );
        mutant_closed += usize::from(mutant.closed());
        // With the lane's cross-check on, the mutant cannot close either.
        let checked = search_wide_with_test_hooks(
            prob,
            &opts(1_500, 0),
            &Rogue::new(prob, After::ClaimAll),
            false,
            false,
            true,
        );
        assert!(
            !checked.closed(),
            "{name}: the cross-check let the mutant close"
        );
        caught += usize::from(checked.crosscheck_failures > 0);
    }
    assert!(
        mutant_closed > 0,
        "VOID: the claim-as-prune mutant closed no sat class, so the pin shows nothing"
    );
    assert!(caught > 0, "VOID: the cross-check never saw the mutant");
}

/// The first class of the sweep the wide path (reference backend) closes by
/// splitting.
fn wide_closing_problem() -> (String, FrozenDualProblem) {
    for f in &sweep() {
        let frame = frame_of(f);
        for &j in &f.adv {
            let prob = problem_of(&state_of(f, &frame, j));
            let r = search_with_backend(&prob, &opts(6_000, 0), &ReferenceBackend::new(&prob));
            if r.closed() && r.nodes > 0 && r.search == "reference_f64" {
                return (format!("{} class {j}", f.name), prob);
            }
        }
    }
    panic!("VOID: the wide path closes no class of the sweep by splitting");
}

#[test]
fn fd_wide_crosscheck_catches_injected_fault() {
    let (name, prob) = wide_closing_problem();
    let every = SearchOptions {
        crosscheck_every: 1,
        ..opts(6_000, 0)
    };
    let clean = search_wide_with_test_hooks(
        &prob,
        &every,
        &ReferenceBackend::new(&prob),
        false,
        false,
        false,
    );
    assert!(clean.closed(), "{name}: VOID, the clean run must close");
    assert_eq!(clean.crosscheck_failures, 0);
    assert!(clean.crosschecks > 0);
    let faulty = search_wide_with_test_hooks(
        &prob,
        &every,
        &ReferenceBackend::new(&prob),
        false,
        true,
        false,
    );
    assert!(
        !faulty.closed(),
        "{name}: an injected 1-ulp fault in the certified values closed the class"
    );
    assert_eq!(faulty.verdict, SearchVerdict::Open(OpenReason::Refused));
    assert!(faulty.crosscheck_failures >= 1);
}

#[test]
fn fd_wide_prunes_only_through_the_certified_predicate() {
    let src = include_str!("../../zono_bab/wide.rs");
    let needle = concat!("certified", " > ", "0");
    let weak = concat!("certified", " >= ");
    let fused = [concat!("mul", "_add"), concat!(".f", "ma(")];
    assert!(src.contains("fn certify_item("), "VOID: wrong file");
    assert!(
        !src.contains(needle) && !src.contains(weak),
        "wide.rs compares a certified value itself: every prune must go through \
         `NodeBound::prunes` / `NodeOut::prunes`"
    );
    for f in fused {
        assert!(!src.contains(f), "wide.rs (certify side) contains `{f}`");
    }
    // The only production writes of a winning point come from a certified
    // `prunes()` (the claim mutant's write is test-only).
    let wins: Vec<usize> = src.match_indices("won = Some(").map(|(i, _)| i).collect();
    assert_eq!(
        wins.len(),
        2,
        "expected the certified write and the test mutant's"
    );
    let certified = src.find("if nb.prunes() {").expect("the certified win");
    assert!(wins.iter().any(|&w| w > certified && w - certified < 80));
    let mutant = src
        .find("if ctx.hooks.claim_prunes() {")
        .expect("the mutant");
    assert!(wins.iter().any(|&w| w > mutant && w - mutant < 400));
    // Sanity of the claim scale.
    assert!(wide_tau_rel(30) > 0.0 && wide_tau_rel(30) < 1e-4);
}

// ---------------------------------------------------------------------------
//  The protocol
// ---------------------------------------------------------------------------

#[test]
fn fd_wide_reference_backend_reproduces_the_f64_tree() {
    // The reference backend proposes exactly the f64 search's points and warm
    // starts, so the wide path must walk the f64 search's tree. The one
    // admitted difference: the wide protocol stores the FAR multipliers in f32,
    // so a node whose only certifying point is far, within ~1e-7 S of zero,
    // can flip — only on a search whose own smallest certified slack is that
    // small.
    let (mut identical, mut compared, mut with_prunes) = (0usize, 0usize, 0usize);
    for (name, prob) in tree_problems() {
        for base in [opts(3_000, 0), opts(40, 0)] {
            let f64_path = search(&prob, &base);
            let wide = search_with_backend(&prob, &base, &ReferenceBackend::new(&prob));
            compared += 1;
            if f64_path.levels.is_empty() {
                assert_eq!(tree(&f64_path), tree(&wide), "{name}: root-decided");
                identical += 1;
                continue;
            }
            assert_eq!(
                (wide.search, wide.probe),
                ("reference_f64", "ok"),
                "{name}: VOID, the wide path did not run"
            );
            assert!(wide.claimed >= wide.certified_prunes);
            if tree(&f64_path) == tree(&wide) {
                identical += 1;
            } else {
                assert!(
                    f64_path.cert_min_slack < 1e-6,
                    "{name}: the wide path's tree differs from the f64 search's although no \
                     prune is near the knife edge (cert_min_slack {:e}):\n f64  {:?}\n wide {:?}",
                    f64_path.cert_min_slack,
                    tree(&f64_path),
                    tree(&wide)
                );
            }
            with_prunes += usize::from(f64_path.certified_prunes > 0);
        }
    }
    eprintln!("[fd-wide-ref] compared={compared} identical={identical} with_prunes={with_prunes}");
    assert!(with_prunes > 0, "VOID: no search pruned a node");
    assert!(
        identical * 4 >= compared * 3,
        "only {identical} of {compared} trees identical"
    );
}

#[test]
fn fd_wide_search_is_thread_count_independent() {
    let (name, prob) = open_problem();
    for after in [After::Honest, After::Random] {
        let run = |threads: usize, chunk: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            pool.install(|| {
                search_with_backend(
                    &prob,
                    &SearchOptions {
                        chunk,
                        ..opts(3_000, 16)
                    },
                    &Rogue::new(&prob, after),
                )
            })
        };
        let a = run(1, 128);
        assert!(a.nodes > 0 && a.search == "rogue", "VOID: {name}");
        for (threads, chunk) in [(3, 128), (8, 128), (8, 7), (3, 1)] {
            assert_eq!(
                digest(&a),
                digest(&run(threads, chunk)),
                "{name} {after:?}: {threads} threads / chunk {chunk} changed the wide search"
            );
        }
    }
}

#[test]
fn fd_wide_refusals_panics_and_malformed_levels_fall_back_to_f64() {
    let (name, prob) = open_problem();
    let reference = search(&prob, &opts(3_000, 0));
    assert!(reference.nodes > 0, "VOID: {name}");
    // An upload refusal: the whole class runs f64.
    let mut refusing = Rogue::new(&prob, After::Honest);
    refusing.refuse_upload = true;
    let r = search_with_backend(&prob, &opts(3_000, 0), &refusing);
    assert_eq!((r.search, r.probe), ("f64", "refused"));
    assert_eq!(tree(&reference), tree(&r), "{name}: refused upload");
    // Mid-search refusals: every level after the probe falls back, and the
    // backend is retired after the first.
    for after in [After::Panic, After::Refuse, After::Malformed] {
        let r = search_with_backend(&prob, &opts(3_000, 0), &Rogue::new(&prob, after));
        assert_eq!(
            r.search, "rogue",
            "{after:?}: the probe passed, so the backend ran"
        );
        assert!(r.fallback_chunks > 0, "{after:?}: no chunk fell back");
        assert_eq!(r.claimed, 0, "{after:?}: nothing was scored by the backend");
        assert_eq!(
            tree(&reference),
            tree(&r),
            "{name} {after:?}: the fallback is not the f64 search"
        );
    }
}

#[test]
fn fd_wide_probe_rejects_a_lying_backend() {
    let (name, prob) = open_problem();
    let mut liar = Rogue::new(&prob, After::ClaimAll);
    liar.lie_in_probe = true;
    let r = search_with_backend(&prob, &opts(3_000, 0), &liar);
    assert_eq!(
        (r.search, r.probe),
        ("f64", "fail"),
        "{name}: a backend whose warm values are off by S must fail the probe"
    );
    assert_eq!(tree(&search(&prob, &opts(3_000, 0))), tree(&r));
    assert_eq!(
        liar.calls.load(Ordering::SeqCst),
        1,
        "only the probe may reach a backend that failed it"
    );
}

#[test]
fn fd_wide_deadline_refusal_stops_open() {
    let (name, prob) = open_problem();
    let r = search_with_backend(&prob, &opts(3_000, 0), &Rogue::new(&prob, After::Deadline));
    assert_eq!(
        r.verdict,
        SearchVerdict::Open(OpenReason::Deadline),
        "{name}: a deadline refusal must stop the search open"
    );
    assert_eq!(r.certified_prunes, 0);
}

#[test]
fn fd_wide_tier_auto_falls_through_a_refusing_backend() {
    let (name, prob) = open_problem();
    let mut cuda_like = Rogue::new(&prob, After::Honest);
    cuda_like.refuse_upload = true;
    let mut tier = WideTier::with_backends(
        FrozenDualSearch::Auto,
        vec![
            Arc::new(cuda_like) as Arc<dyn SearchBackend>,
            Arc::new(ReferenceBackend::new(&prob)),
        ],
    );
    let r = tier.search_class(&prob, &opts(3_000, 0));
    assert_eq!((r.search, r.probe), ("reference_f64", "ok"), "{name}");
    assert_eq!(tier.probe_status(), "ok");
    tier.emit();
    // Every backend refusing: f64, and the tier reports the refusal.
    let mut a = Rogue::new(&prob, After::Honest);
    a.refuse_upload = true;
    let mut tier = WideTier::with_backends(
        FrozenDualSearch::Auto,
        vec![Arc::new(a) as Arc<dyn SearchBackend>],
    );
    let r = tier.search_class(&prob, &opts(3_000, 0));
    assert_eq!((r.search, r.probe), ("f64", "refused"));
    assert_eq!(tier.probe_status(), "refused");
}

#[test]
fn fd_wide_search_copy_is_the_problem_rounded() {
    let (name, prob) = open_problem();
    let copy = search_copy(&prob).expect("copy");
    assert!(copy.validate().is_ok());
    assert_eq!((copy.n, copy.depth), (prob.n(), prob.max_depth()), "{name}");
    #[allow(clippy::cast_possible_truncation)]
    let f = |v: f64| v as f32;
    for i in 0..prob.max_depth() {
        let row = prob.split_row(i);
        assert_eq!(copy.enc[i] as usize, row.enc);
        for (j, &a) in row.a.iter().enumerate() {
            assert_eq!(copy.a[i * copy.n + j].to_bits(), f(a).to_bits());
        }
        for (got, want) in [
            (copy.rho[i], row.rho),
            (copy.c0s[i], row.c0s),
            (copy.b0[i], row.b0),
            (copy.b1[i], row.b1),
        ] {
            assert_eq!(got, [f(want[0]), f(want[1])]);
        }
        assert_eq!(copy.row_abs[i], f(row.row_abs));
    }
    for (got, &want) in copy.d.iter().zip(prob.d()) {
        assert_eq!(got.to_bits(), f(want).to_bits());
    }
    assert_eq!(copy.c0, f(prob.c0()));
    // An f32 overflow refuses the copy (the class then runs f64).
    let huge = prob.with_c0_for_test(1e300);
    assert!(search_copy(&huge).is_err());
}

#[test]
fn fd_wide_search_rules_match_the_f64_path() {
    let (tol, buckets) = wide_search_rules();
    assert_eq!(tol.to_bits(), FD_SEARCH_TOL.to_bits());
    assert_eq!(buckets, FD_LOG_BUCKETS);
}

// ---------------------------------------------------------------------------
//  M3 x M4: the wide search over the shared trunk
// ---------------------------------------------------------------------------

/// A `shared`-trunk tier policy (60 s, so the node cap governs) with the M4
/// search `search`.
fn shared_policy(node_cap: u64, search: FrozenDualSearch) -> TierPolicy {
    TierPolicy::clamped(node_cap, 60.0)
        .with_state(FrozenDualStateMode::Shared)
        .with_search(search)
}

/// An armed tier's M4 state whose backend for each class is built from THAT
/// class's problem (the reference search, or a rogue doing `after`). Records,
/// in call order, the certified root of every problem a backend was handed.
fn per_class_backends(after: Option<After>, handed: &Arc<Mutex<Vec<u64>>>) -> WideTier {
    let handed = Arc::clone(handed);
    WideTier::with_backend_factory(FrozenDualSearch::CpuF32, move |prob| {
        let root = prob.eval_node(&[], &[], &[], &mut NodeScratch::default(), None);
        handed
            .lock()
            .expect("handed")
            .push(root.certified.to_bits());
        let backend: Arc<dyn SearchBackend> = match after {
            None => Arc::new(ReferenceBackend::new(prob)),
            Some(after) => Arc::new(Rogue::new(prob, after)),
        };
        vec![backend]
    })
}

/// The tier on `f` with true class `t` and objectives `adv`, all open, with
/// `wide` injected (`None`: the f64 search) and `mutant`.
fn shared_tier(
    f: &Fx,
    t: usize,
    adv: &[usize],
    policy: &TierPolicy,
    wide: Option<WideTier>,
    mutant: TierMutant,
) -> TierReport {
    let dj = vec![f64::NAN; adv.len()];
    run_tier_with_wide(
        &f.net, &f.root, t, adv, &dj, adv, &f.ybox, None, policy, wide, mutant,
    )
}

/// Every class outcome of a tier (the search's whole deterministic report and
/// the class's state), wall clock excluded.
fn class_digests(rep: &TierReport) -> Vec<String> {
    rep.classes
        .iter()
        .map(|c| match &c.result {
            Ok(run) => format!(
                "class={} {:?} root_g={:016x} loss={:?} state={} search={} probe={} \
                 fallback={}",
                c.class,
                digest(&run.search),
                run.root_g.to_bits(),
                run.root_loss.map(f64::to_bits),
                run.state.token(),
                run.search.search,
                run.search.probe,
                run.search.fallback_chunks
            ),
            Err(why) => format!("class={} refused={}", c.class, why.token()),
        })
        .collect()
}

#[test]
fn fd_wide_policy_state_and_search_are_independent_selectors() {
    use ny_levers::decls::margin_row::{
        MARGIN_ROW_FROZEN_DUAL_BAB_SEARCH as SEARCH, MARGIN_ROW_FROZEN_DUAL_BAB_STATE as STATE,
    };
    let p = TierPolicy::clamped(10, 3.0);
    assert_eq!(
        (p.state, p.search),
        (FrozenDualStateMode::PerClass, FrozenDualSearch::F64),
        "both defaults together are the M1 tier"
    );
    let searches = [
        FrozenDualSearch::F64,
        FrozenDualSearch::CpuF32,
        FrozenDualSearch::Cuda,
        FrozenDualSearch::Auto,
    ];
    for state in [FrozenDualStateMode::PerClass, FrozenDualStateMode::Shared] {
        for search in searches {
            let a = p.with_state(state).with_search(search);
            assert_eq!(a, p.with_search(search).with_state(state));
            assert_eq!(
                (a.state, a.search, a.node_cap, a.secs.to_bits()),
                (state, search, p.node_cap, p.secs.to_bits()),
                "{} x {}",
                state.token(),
                search.token()
            );
        }
    }
    // Governed reads (no raw environment access).
    if ny_levers::read(&STATE).source == ny_levers::Source::Default
        && ny_levers::read(&SEARCH).source == ny_levers::Source::Default
    {
        let g = TierPolicy::from_gates();
        assert_eq!(
            (g.state, g.search),
            (FrozenDualStateMode::PerClass, FrozenDualSearch::F64)
        );
    }
}

#[test]
fn fd_wide_shared_tier_reference_backend_reproduces_the_shared_f64_tier() {
    // The trunk decides each class's STATE; the backend only scores the levels
    // of the problem substituted from it. Over the same trunk the state side
    // must be the shared f64 tier's bit for bit (slopes, roots, losses, depth
    // caps), each built class's backend must be handed exactly that class's
    // problem, once, in tier order, and an honest backend must walk the shared
    // f64 tier's trees (the far-point knife edge of
    // `fd_wide_reference_backend_reproduces_the_f64_tree` aside), with the
    // sampled scalar cross-check (G4) running on the wide path's prunes.
    let (mut compared, mut identical, mut searched) = (0usize, 0usize, 0usize);
    let mut xchk = 0u64;
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().step_by(3));
    for f in &fxs {
        if f.adv.len() < 2 {
            continue;
        }
        let base = shared_tier(
            f,
            f.t,
            &f.adv,
            &shared_policy(2_000, FrozenDualSearch::F64),
            None,
            TierMutant::default(),
        );
        let handed = Arc::new(Mutex::new(Vec::new()));
        let wide = shared_tier(
            f,
            f.t,
            &f.adv,
            &shared_policy(2_000, FrozenDualSearch::CpuF32),
            Some(per_class_backends(None, &handed)),
            TierMutant::default(),
        );
        assert_eq!(
            (base.state, wide.state),
            (FrozenDualStateMode::Shared, FrozenDualStateMode::Shared)
        );
        assert_eq!(
            base.trunk_lambda, wide.trunk_lambda,
            "{}: the slope choice",
            f.name
        );
        assert_eq!(base.classes_built, wide.classes_built, "{}", f.name);
        assert_eq!(
            base.shared_root_loss_max.map(f64::to_bits),
            wide.shared_root_loss_max.map(f64::to_bits),
            "{}",
            f.name
        );
        assert_eq!(base.classes.len(), wide.classes.len(), "{}", f.name);
        let mut own = Vec::new();
        for (a, b) in base.classes.iter().zip(&wide.classes) {
            assert_eq!(a.class, b.class, "{}: the tier order", f.name);
            let (x, y) = match (&a.result, &b.result) {
                (Ok(x), Ok(y)) => (x, y),
                (Err(p), Err(q)) => {
                    assert_eq!(p.token(), q.token(), "{} class {}", f.name, a.class);
                    continue;
                }
                _ => panic!(
                    "{} class {}: one search path built the class, the other refused it",
                    f.name, a.class
                ),
            };
            compared += 1;
            assert_eq!(
                (x.state, y.state),
                (FrozenDualStateMode::Shared, FrozenDualStateMode::Shared)
            );
            assert_eq!(
                (
                    x.root_g.to_bits(),
                    x.root_loss.map(f64::to_bits),
                    x.depth_cap,
                    x.state_gens,
                    x.unstable,
                    x.search.root_certified.to_bits()
                ),
                (
                    y.root_g.to_bits(),
                    y.root_loss.map(f64::to_bits),
                    y.depth_cap,
                    y.state_gens,
                    y.unstable,
                    y.search.root_certified.to_bits()
                ),
                "{} class {}: the class's shared state depends on the search",
                f.name,
                a.class
            );
            own.push(y.search.root_certified.to_bits());
            if x.search.levels.is_empty() {
                assert_eq!(
                    tree(&x.search),
                    tree(&y.search),
                    "{} class {}: root-decided",
                    f.name,
                    a.class
                );
                identical += 1;
                continue;
            }
            assert_eq!(
                (y.search.search, y.search.probe),
                ("reference_f64", "ok"),
                "{} class {}: VOID, the wide path did not run",
                f.name,
                a.class
            );
            assert!(y.search.claimed >= y.search.certified_prunes);
            assert_eq!(
                y.search.crosscheck_failures, 0,
                "{} class {}: CROSSCHECK-MISMATCH on the wide path over the shared trunk",
                f.name, a.class
            );
            xchk += y.search.crosschecks;
            searched += 1;
            if tree(&x.search) == tree(&y.search) {
                identical += 1;
            } else {
                assert!(
                    x.search.cert_min_slack < 1e-6,
                    "{} class {}: the wide tree differs from the shared f64 tier's although no \
                     prune is near the knife edge (cert_min_slack {:e}):\n f64  {:?}\n wide {:?}",
                    f.name,
                    a.class,
                    x.search.cert_min_slack,
                    tree(&x.search),
                    tree(&y.search)
                );
            }
        }
        assert_eq!(
            *handed.lock().expect("handed"),
            own,
            "{}: each built class's backend must be handed that class's own problem, once, \
             in tier order",
            f.name
        );
        assert_eq!(
            (wide.claimed, wide.refuted),
            wide.classes
                .iter()
                .filter_map(|c| c.result.as_ref().ok())
                .fold((0, 0), |(c, r), run| (
                    c + run.search.claimed,
                    r + run.search.refuted
                )),
            "{}: the tier's claim ledger is its classes'",
            f.name
        );
    }
    eprintln!(
        "[fd-wide-shared-ref] compared={compared} identical={identical} searched={searched} \
         xchk={xchk}"
    );
    assert!(
        searched > 0,
        "VOID: no class of a shared tier reached the wide path"
    );
    assert!(
        xchk > 0,
        "VOID: the cross-check never ran on a wide prune over the shared trunk"
    );
    assert!(
        identical * 4 >= compared * 3,
        "only {identical} of {compared} class trees identical"
    );
}

#[test]
fn fd_wide_shared_tier_garbage_backends_never_close_a_witnessed_sat_class() {
    // Fault injection at the TIER, over the shared trunk: every class's levels
    // are scored by a rogue backend (honest in its probe, then garbage). A
    // wrong proposal can only fail to certify, so no witnessed sat class may
    // close: swapped targets and sampled counterexamples over the fixtures and
    // the sweep, and planted violations.
    let afters: Vec<After> = std::iter::once(After::Honest).chain(GARBAGE).collect();
    let mut witnessed = 0usize;
    let (mut engaged, mut claimed) = (0usize, 0u64);
    let mut failures = Vec::new();
    let mut check = |f: &Fx, t: usize, adv: &[usize], sat: &[usize], what: &str| {
        for &after in &afters {
            let handed = Arc::new(Mutex::new(Vec::new()));
            let tier = shared_tier(
                f,
                t,
                adv,
                &shared_policy(1_500, FrozenDualSearch::CpuF32),
                Some(per_class_backends(Some(after), &handed)),
                TierMutant::default(),
            );
            engaged += tier
                .classes
                .iter()
                .filter(|c| {
                    c.result
                        .as_ref()
                        .is_ok_and(|run| run.search.search == "rogue")
                })
                .count();
            claimed += tier.claimed;
            if tier.crosscheck_failures > 0 {
                failures.push(format!(
                    "{} {what}: rogue {after:?} over the shared trunk: {} CROSSCHECK-MISMATCH",
                    f.name, tier.crosscheck_failures
                ));
            }
            for j in sat {
                if tier.closed.contains(j) {
                    failures.push(format!(
                        "{} {what}: rogue {after:?} over the shared trunk CLOSED the witnessed \
                         sat class {j}",
                        f.name
                    ));
                }
            }
        }
    };
    let mut fxs = fixtures();
    fxs.extend(sweep().into_iter().step_by(2));
    for f in &fxs {
        let x0 = midpoint(&f.lo, &f.hi);
        let (top, second) = top_two(&f.net, &x0);
        if dd_margin(&f.net, second, top, &x0) < 0.0 {
            witnessed += 1;
            let adv: Vec<usize> = (0..f.net.n_out).filter(|&c| c != second).collect();
            check(f, second, &adv, &[top], "swapped");
        }
        let sat: Vec<usize> = f
            .adv
            .iter()
            .copied()
            .filter(|&j| {
                let mut rng = StdRng::seed_from_u64(0x5A7 ^ (j as u64) << 4);
                (0..48).any(|s| {
                    let x = sample(&mut rng, &f.lo, &f.hi, s);
                    dd_margin(&f.net, f.t, j, &x) < 0.0
                })
            })
            .collect();
        if !sat.is_empty() {
            witnessed += sat.len();
            check(f, f.t, &f.adv, &sat, "sampled");
        }
    }
    for seed in 0..6u64 {
        let half = [0.2, 0.08, 0.03][(seed % 3) as usize];
        let mut rng = StdRng::seed_from_u64(0x9A17 + seed);
        let spec0 = tiny_spec(&mut rng, 0.4);
        let f0 = fx("plant0", spec0.clone(), half, 0x9A17 + seed, false);
        let x0 = midpoint(&f0.lo, &f0.hi);
        let (t, j) = top_two(&f0.net, &x0);
        let m0 = dd_margin(&f0.net, t, j, &x0);
        for delta in [1e-3, 1e-9] {
            let mut spec = spec0.clone();
            let Some(TwinOpSpec::Gemm { bias, .. }) = spec.ops.last_mut() else {
                panic!("the tiny spec ends in its final Gemm");
            };
            bias[j] += m0 + delta;
            let f = fx("plant", spec, half, 0x9A17 + seed, false);
            if dd_margin(&f.net, t, j, &x0) >= 0.0 {
                continue;
            }
            witnessed += 1;
            let adv: Vec<usize> = (0..f.net.n_out).filter(|&c| c != t).collect();
            check(&f, t, &adv, &[j], "planted");
        }
    }
    eprintln!("[fd-wide-shared-sat] witnessed={witnessed} engaged={engaged} claimed={claimed}");
    assert!(
        witnessed >= 4,
        "only {witnessed} witnessed sat classes: the negative control is VOID"
    );
    assert!(
        failures.is_empty(),
        "THE WIDE PATH OVER THE SHARED TRUNK CLOSED A SAT CLASS (soundness emergency): \
         {failures:#?}"
    );
    assert!(
        engaged > 0 && claimed > 0,
        "VOID: the rogue backends never searched a shared-trunk class (engaged={engaged}, \
         claimed={claimed})"
    );
}

#[test]
fn fd_wide_shared_tier_closed_classes_hold_at_sampled_points() {
    let mut closed_by_split = 0usize;
    let mut failures = Vec::new();
    for f in &sweep() {
        // Every class open together (one shared trunk), then each class alone
        // (a one-class trunk is its per-class state), on the honest and on a
        // garbage backend.
        let mut runs: Vec<Vec<usize>> = vec![f.adv.clone()];
        runs.extend(f.adv.iter().map(|&j| vec![j]));
        for open in &runs {
            for after in [None, Some(After::Random)] {
                let handed = Arc::new(Mutex::new(Vec::new()));
                let tier = shared_tier(
                    f,
                    f.t,
                    open,
                    &shared_policy(6_000, FrozenDualSearch::CpuF32),
                    Some(per_class_backends(after, &handed)),
                    TierMutant::default(),
                );
                for c in &tier.classes {
                    let Ok(run) = &c.result else {
                        continue;
                    };
                    if !tier.closed.contains(&c.class) {
                        continue;
                    }
                    if run.search.nodes > 0 && run.search.search != "f64" {
                        closed_by_split += 1;
                    }
                    let j = c.class;
                    let mut rng = StdRng::seed_from_u64(0xC105 ^ (j as u64) << 3);
                    for s in 0..32 {
                        let x = sample(&mut rng, &f.lo, &f.hi, s);
                        let m = dd_margin(&f.net, f.t, j, &x);
                        if m <= 0.0 {
                            failures.push(format!(
                                "{} class {j} ({} open, {after:?}): CLOSED over the shared trunk \
                                 but point {s} has margin {m:e}",
                                f.name,
                                open.len()
                            ));
                        }
                    }
                }
            }
        }
    }
    eprintln!("[fd-wide-shared-positive] closed_by_split={closed_by_split}");
    assert!(
        failures.is_empty(),
        "THE WIDE PATH OVER THE SHARED TRUNK CLOSED A CLASS WITH A COUNTEREXAMPLE: \
         {failures:#?}"
    );
    assert!(
        closed_by_split > 0,
        "VOID: no shared-trunk class closed by splitting on the wide path"
    );
}

#[test]
fn fd_wide_shared_tier_bindings_refuse_before_any_backend() {
    // M3's assembly bindings hold on the M4 path. A class whose tail is another
    // class's (CLASS), whose frame has another generator layout (LAYOUT), or
    // whose trunk rows changed after the build (DIGEST) is REFUSED by
    // `FrozenDualTrunk::state` before its problem exists — on the f64 search
    // and on the wide search alike — so no backend is ever handed it, and it
    // is never closed.
    let mutants = [
        (
            "class",
            TierMutant {
                swap_tails: Some((0, 1)),
                ..TierMutant::default()
            },
            "offered as class",
        ),
        (
            "layout",
            TierMutant {
                move_frame: true,
                ..TierMutant::default()
            },
            "layout",
        ),
        (
            "digest",
            TierMutant {
                corrupt_trunk: true,
                ..TierMutant::default()
            },
            "rows changed",
        ),
    ];
    let mut pinned = [0usize; 3];
    for f in &fixtures() {
        if f.adv.len() < 2 {
            continue;
        }
        // VOID guard: unmutated, the wide tier builds and searches the classes
        // the mutants must refuse (and the trunk has rows to corrupt).
        let handed = Arc::new(Mutex::new(Vec::new()));
        let honest = shared_tier(
            f,
            f.t,
            &f.adv,
            &shared_policy(1_500, FrozenDualSearch::CpuF32),
            Some(per_class_backends(None, &handed)),
            TierMutant::default(),
        );
        let all_built = honest
            .classes
            .iter()
            .all(|c| c.result.as_ref().is_ok_and(|run| run.unstable > 0));
        if !all_built {
            continue;
        }
        assert_eq!(
            handed.lock().expect("handed").len(),
            honest.classes_built,
            "{}: VOID, the unmutated tier did not hand every class to a backend",
            f.name
        );
        for (k, (what, mutant, needle)) in mutants.iter().enumerate() {
            for search in [FrozenDualSearch::F64, FrozenDualSearch::CpuF32] {
                let handed = Arc::new(Mutex::new(Vec::new()));
                let wide =
                    (search != FrozenDualSearch::F64).then(|| per_class_backends(None, &handed));
                let tier =
                    shared_tier(f, f.t, &f.adv, &shared_policy(1_500, search), wide, *mutant);
                let hit: Vec<usize> = if *what == "class" {
                    vec![0, 1]
                } else {
                    (0..tier.classes.len()).collect()
                };
                for &pos in &hit {
                    let c = &tier.classes[pos];
                    match &c.result {
                        Err(ZonoRefusal::Internal(msg)) => assert!(
                            msg.contains(needle),
                            "{} {what} mutant, search {}: class {} refused for another \
                             reason: {msg}",
                            f.name,
                            search.token(),
                            c.class
                        ),
                        other => panic!(
                            "{} {what} mutant, search {}: class {} at tier position {pos} was \
                             not refused by the binding: {:?}",
                            f.name,
                            search.token(),
                            c.class,
                            other.as_ref().map(|run| run.search.verdict)
                        ),
                    }
                    assert!(
                        !tier.closed.contains(&c.class),
                        "{} {what}: a refused class closed",
                        f.name
                    );
                }
                let built = tier.classes.iter().filter(|c| c.result.is_ok()).count();
                assert_eq!(built, tier.classes_built, "{} {what}", f.name);
                assert_eq!(
                    built + hit.len(),
                    tier.classes.len(),
                    "{} {what}: the mutant refused a class it does not touch",
                    f.name
                );
                let expected = if search == FrozenDualSearch::F64 {
                    0
                } else {
                    built
                };
                assert_eq!(
                    handed.lock().expect("handed").len(),
                    expected,
                    "{} {what} mutant, search {}: a backend was handed a class the binding \
                     refused (or a built class was never handed)",
                    f.name,
                    search.token()
                );
                pinned[k] += hit.len();
            }
        }
    }
    eprintln!("[fd-wide-shared-bindings] refused class/layout/digest = {pinned:?}");
    assert!(
        pinned.iter().all(|&p| p > 0),
        "VOID: a binding mutant refused nothing ({pinned:?})"
    );
}

#[test]
fn fd_wide_shared_tier_is_thread_count_independent() {
    let f = &fixture0();
    assert!(f.adv.len() >= 2, "the pin needs several open classes: VOID");
    for after in [None, Some(After::Random)] {
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            pool.install(|| {
                let handed = Arc::new(Mutex::new(Vec::new()));
                shared_tier(
                    f,
                    f.t,
                    &f.adv,
                    &shared_policy(3_000, FrozenDualSearch::CpuF32),
                    Some(per_class_backends(after, &handed)),
                    TierMutant::default(),
                )
            })
        };
        let a = run(1);
        assert!(
            a.classes.iter().any(|c| c
                .result
                .as_ref()
                .is_ok_and(|run| run.search.search != "f64" && run.search.nodes > 0)),
            "VOID ({after:?}): no class searched a level on the wide path"
        );
        for threads in [3, 8] {
            let b = run(threads);
            assert_eq!(
                a.trunk_lambda, b.trunk_lambda,
                "{threads} threads: the slope choice"
            );
            assert_eq!(
                class_digests(&a),
                class_digests(&b),
                "{threads} threads ({after:?}): the shared wide tier depends on the thread count"
            );
            assert_eq!(a.closed, b.closed);
            assert_eq!((a.claimed, a.refuted), (b.claimed, b.refuted));
        }
    }
}

// ---------------------------------------------------------------------------
//  Composition: wide level search plus certified breakpoint refinement
// ---------------------------------------------------------------------------

/// Exact margin `c_obj - 1 + (x+y)/4 + relu(x) + relu(y) + relu(-x-y)`.
/// Its minimum on [-1,1]^2 is `c_obj - 1`, attained at x=y=0.
fn breakpoint_composition_problem(c_obj: f64) -> FrozenDualProblem {
    let state = FrozenDualState {
        t: 0,
        class: 1,
        n_gens: 5,
        n_in_gens: 2,
        c_obj,
        d: vec![0.25, 0.25, 0.25, 0.25, 0.5],
        r_obj: 0.0,
        a: vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0,
        ],
        c_in: vec![0.0; 3],
        r: vec![0.0; 3],
        lam: vec![0.5; 3],
        mu: vec![0.25, 0.25, 0.5],
        lo: vec![-1.0, -1.0, -2.0],
        hi: vec![1.0, 1.0, 2.0],
        enc: vec![2, 3, 4],
        layer: vec![0; 3],
        site: vec![ReluSite::Trunk(0); 3],
        neuron: vec![0, 1, 2],
        stats: FrozenDualBuildStats::default(),
    };
    FrozenDualProblem::new(&state, &[0, 1, 2]).expect("composition fixture")
}

/// The level surface passes the sixteen-node known-answer probe honestly,
/// then deliberately makes no positive claims and proposes zero next multipliers.
/// Refinement finds the analytic optimum for the first two split rows, so
/// the test witnesses both inheritance of an accepted negative depth-one
/// hint and new certified depth-two prunes on a successful wide call.
struct ComposingBackend {
    honest: ReferenceBackend,
    refinement_calls: AtomicUsize,
    inherited_nodes: AtomicUsize,
}

impl ComposingBackend {
    fn new(prob: &FrozenDualProblem) -> Self {
        Self {
            honest: ReferenceBackend::new(prob),
            refinement_calls: AtomicUsize::new(0),
            inherited_nodes: AtomicUsize::new(0),
        }
    }
}

impl SearchBackend for ComposingBackend {
    fn name(&self) -> &'static str {
        "composing_reference"
    }

    fn caps(&self) -> BackendCaps {
        self.honest.caps()
    }

    fn upload_fd(
        &self,
        problem: &FdSearchProblem,
        lease: &DeviceLease,
    ) -> Result<FdHandle, Refusal> {
        self.honest.upload_fd(problem, lease)
    }

    fn fd_level(
        &self,
        handle: &FdHandle,
        chunk: &LevelChunk<'_>,
        out: &mut LevelOut,
    ) -> Result<Untrusted<LevelStats>, Refusal> {
        let stats = self.honest.fd_level(handle, chunk, out)?;
        if chunk.depth == 2 {
            for k in 0..chunk.count {
                let expected = if chunk.sides[k * 2] == 0 { 0.25 } else { 0.75 };
                if chunk.l0[k * 2] == expected && chunk.l1[k * 2] == 0.0 {
                    self.inherited_nodes.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        let w = out.writer();
        w.next.fill(0.0);
        // This fixture has at most eight nodes per real level. Its sixteen
        // probe nodes retain the honest values; suppressing later claims must
        // neither prevent certified refinement nor create pruning authority.
        if chunk.count != 16 {
            w.value.fill(-100.0);
        }
        Ok(stats)
    }

    fn refine_dual(&self, request: &DualSearchRequest<'_>) -> Result<DualSearchOutcome, Refusal> {
        self.refinement_calls.fetch_add(1, Ordering::Relaxed);
        assert!(
            request.l0.iter().chain(request.l1).all(|&x| x == 0.0),
            "refinement must start from the sanitised view.next point"
        );
        let mut l0 = vec![0.0; request.rows.len()];
        for (value, row) in l0.iter_mut().zip(request.rows).take(2) {
            *value = 0.5 - 0.25 * row.sign;
        }
        Ok(DualSearchOutcome {
            multipliers: Untrusted::new(DualMultipliers {
                l0,
                l1: vec![0.0; request.rows.len()],
            }),
            steps: 1,
            gain: Untrusted::new(f64::INFINITY),
            stop: "claimed-unsat",
            support: Default::default(),
        })
    }
}

#[test]
fn fd_wide_breakpoint_composition_inherits_hints_and_replays_new_prunes() {
    let positive = breakpoint_composition_problem(1.05);
    let unsafe_problem = breakpoint_composition_problem(0.95);
    assert!((q(0.95) - q(1.0)).is_negative(), "SAT control is void");
    for evaluator in [
        Evaluator::Node,
        Evaluator::Block { far_first: false },
        Evaluator::Block { far_first: true },
    ] {
        let base_opts = SearchOptions {
            evaluator,
            crosscheck_every: 1,
            ..opts(256, 32)
        };
        let enabled_opts = SearchOptions {
            refine_steps: 4,
            ..base_opts
        };
        let backend = ComposingBackend::new(&positive);
        let base =
            search_wide_with_test_hooks(&positive, &base_opts, &backend, false, false, false);
        let unchanged =
            search_wide_with_test_hooks(&positive, &base_opts, &backend, false, false, false);
        assert_eq!(digest(&base), digest(&unchanged));
        assert_eq!(backend.refinement_calls.load(Ordering::Relaxed), 0);
        assert!(!base.closed());
        assert_eq!((base.nodes, base.certified_prunes), (14, 0));
        let report =
            search_wide_with_test_hooks(&positive, &enabled_opts, &backend, false, false, false);
        assert!(report.closed());
        assert_eq!((report.nodes, report.certified_prunes), (6, 4));
        assert_eq!(
            (report.search, report.probe, report.fallback_chunks),
            ("composing_reference", "ok", 0)
        );
        assert_eq!(
            (
                report.refine_calls,
                report.refine_accepted,
                report.refine_prunes
            ),
            (6, 6, 4)
        );
        assert_eq!(backend.refinement_calls.load(Ordering::Relaxed), 6);
        assert!(
            backend.inherited_nodes.load(Ordering::Relaxed) >= 4,
            "accepted negative hints did not reach the next wide level"
        );
        assert_eq!((report.crosschecks, report.crosscheck_failures), (4, 0));
        assert_eq!(report.pruned.len(), 4, "certificate replay is void");
        assert!(crate::margin_row::zono_bab::ledger_closes(&report.levels));
        for node in &report.pruned {
            assert_eq!(node.candidate, Candidate::Breakpoint);
            assert_eq!(node.sides.len(), 2);
            for (i, side) in node.sides.iter().enumerate() {
                assert_eq!(node.l0[i], 0.5 - 0.25 * side.sign());
            }
            assert!(node.l1.iter().all(|&x| x == 0.0));
            let replay = positive.eval_node(
                &node.sides,
                &node.l0,
                &node.l1,
                &mut NodeScratch::default(),
                None,
            );
            assert_eq!(replay.certified.to_bits(), node.certified.to_bits());
            assert!(
                exact_node_value(&positive, &node.sides, &node.l0, &node.l1) >= q(node.certified)
            );
        }
        let unsafe_backend = ComposingBackend::new(&unsafe_problem);
        let unsafe_report = search_wide_with_test_hooks(
            &unsafe_problem,
            &enabled_opts,
            &unsafe_backend,
            false,
            false,
            false,
        );
        assert!(!unsafe_report.closed(), "SAT margin was falsely closed");
        assert!(
            unsafe_report.refine_calls > 0,
            "SAT refinement path is void"
        );
        assert!(!crate::margin_row::zono_bab::ledger_closes(
            &unsafe_report.levels
        ));
        for node in &unsafe_report.pruned {
            assert!(
                exact_node_value(&unsafe_problem, &node.sides, &node.l0, &node.l1)
                    >= q(node.certified)
            );
        }
        let fault_backend = ComposingBackend::new(&positive);
        let faulty = search_wide_with_test_hooks(
            &positive,
            &enabled_opts,
            &fault_backend,
            false,
            true,
            false,
        );
        assert_eq!(faulty.verdict, SearchVerdict::Open(OpenReason::Refused));
        assert!(
            faulty.crosscheck_failures > 0,
            "refinement fault bypassed cross-check"
        );
    }
}
