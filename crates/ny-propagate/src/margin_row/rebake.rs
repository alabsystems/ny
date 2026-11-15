// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #rebake: the joint forward re-bake for the margin-row root (#twinwall).
//!
//! WHY. The lane's root pipeline is forward tableau (`root.rs`
//! `build_retaining_inner`) -> per-layer backward intermediates
//! (#backward-interm, shrink-only, compounding) -> `gates_from_box` ->
//! alpha ascent (#alpha-opt, backward-frame only, frozen tableau). That is
//! HALF of alpha-CROWN's actual root mechanism: alpha-CROWN's optimized alpha
//! also tightens every INTERMEDIATE box, because the forward relaxation is
//! rebuilt with it. Here the optimized alpha never re-enters the forward bake
//! — `apply_gates` keeps consuming the area heuristic — so the boxes, and
//! therefore the triangles, and therefore the next alpha, are all frozen at
//! the heuristic's quality. Measured state this module answers: one-shot
//! backward-interm is SATURATED (bit-identical roots at 2x budget / 4x topk)
//! and alpha-after-interm alone buys ~+0.003; the remaining root-quality
//! capability is the JOINT loop.
//!
//! WHAT. Per round (default 2, `NY_MARGIN_ROW_REBAKE_ROUNDS`):
//!   1. alpha ascent on the CURRENT build (`alpha_opt::optimize_root_alpha`);
//!   2. REBUILD the forward tableau with the optimized alpha baked into
//!      `apply_gates` (`RootGates::build_rebaked`), re-running the
//!      backward-interm phase (halved budget) and the clip/Tier-0 retention
//!      on the new prefix gates — each round compounds: tighter boxes ->
//!      narrower triangles -> better alpha -> tighter boxes;
//!   3. score the candidate by the lane's OWN published number
//!      (`bab::root_eval`, min over adv classes) and MONOTONE-ACCEPT at the
//!      round level: the rebuilt `RootGates` replaces the previous one only
//!      if that certified bound strictly improves; otherwise the previous
//!      build stands and the loop stops (the step is deterministic, so a
//!      rejected round would only repeat).
//!
//! SOUNDNESS. Every candidate is INTERNALLY CERTIFIED BY CONSTRUCTION: it is
//! produced by the unchanged build path — same concretize envelopes, same
//! `repair_upper_lines` chord certificate, same shrink-only interm
//! intersection, same `bi_cal` forward-only clip calibration (re-captured
//! inside the rebuild, so the #backward-interm ordering trap cannot re-open
//! here), same `ms = max(alpha, s)` derivation — for ANY seed in `[0, 1]`
//! (the seed is clamped). The ONE new obligation of the whole program is
//! `apply_gates`' outward widening under a FRACTIONAL lower product, which
//! carries its own re-derived magnitude-scaled branch (see the derivation in
//! `root.rs::apply_gates`; the historical `6u(|mm|+dd)` budget leaned on the
//! `gates_from_box` coupling `alpha=1 => s>=1/2` and does NOT survive a free
//! alpha). Acceptance is merely a CHOICE among valid certified builds, made
//! on the certified number itself; the driver recomputes the published bound
//! from the accepted root through the unchanged `root_eval`.
//!
//! BUDGET. `NY_MARGIN_ROW_REBAKE_SECS` (default 30) capped to 40% of the
//! remaining deadline (the `alpha_opt`/#backward-interm starvation guard).
//! One round costs one alpha ascent (its own sub-slice) + one forward build
//! (~the measured root build) + one certified `root_eval`; the loop refuses
//! a round it cannot afford, loudly, and every abort path leaves the
//! previously accepted (or original) root standing.
//!
//! GATE. The scored [`super::RootAlphaRuntimePolicy`] is frozen once from the
//! typed preset/LeverSet; its exact legacy `NY_MARGIN_ROW_REBAKE=1` overlay also
//! arms it, and the declaration default is OFF. The lane is byte-identical when
//! disarmed (the entry returns before touching anything).

use std::time::Instant;

use super::alpha_opt;
use super::bab::root_eval_store;
use super::backward_interm;
use super::bi_gpu::{BiGpu, BiGpuScope};
use super::engine::BackwardEngine;
use super::net::TwinNet;
use super::root::{RetainCfg, RetainedRows, RootGates};

/// Alpha-ascent iterations per round (the #alpha-opt default; the round's
/// wall sub-slice is the binding cap in practice).
const ROUND_ALPHA_ITERS: usize = 8;

/// Trunk `(alpha, ms)` per layer, plus the #head-alpha slopes.
type AlphaState = (Vec<(Vec<f64>, Vec<f64>)>, Option<Vec<f64>>);

fn snapshot_alpha_state(root: &RootGates) -> AlphaState {
    (
        root.layers
            .iter()
            .map(|layer| (layer.alpha.clone(), layer.ms.clone()))
            .collect(),
        root.head_alpha.clone(),
    )
}

fn restore_alpha_state(root: &mut RootGates, state: AlphaState) {
    let (layers, head) = state;
    assert_eq!(
        root.layers.len(),
        layers.len(),
        "root layer count changed during alpha-only ascent"
    );
    for (layer, (alpha, ms)) in root.layers.iter_mut().zip(layers) {
        layer.alpha = alpha;
        layer.ms = ms;
    }
    root.head_alpha = head;
}

fn alpha_differs_from_area_heuristic(
    lower: &[f64],
    upper: &[f64],
    alpha: &[f64],
    unstable: &[usize],
) -> bool {
    unstable.iter().any(|&index| {
        let baked = if upper[index] >= -lower[index] {
            1.0
        } else {
            0.0
        };
        alpha[index] != baked
    })
}

/// What one re-bake run did (telemetry + test surface).
pub struct RebakeReport {
    pub(crate) policy: super::RootAlphaRuntimePolicy,
    /// Effective wall slice after the 40%-of-remaining cap.
    pub effective_secs: f64,
    /// Whether the instance deadline reduced the configured wall request.
    pub deadline_capped: bool,
    /// Whether a round was refused by the effective wall remainder.
    pub budget_exhausted: bool,
    /// Whether the loop reached the configured round ceiling.
    pub round_cap_exhausted: bool,
    /// Whether the entry root carried standalone-alpha movement not yet baked
    /// into its forward tableau.
    pub entry_unbaked_alpha: bool,
    /// Affordable rounds that entered the alpha-ascent phase.
    pub rounds_started: usize,
    /// Nested alpha runs that returned a complete report.
    pub alpha_completed: usize,
    /// Candidate alpha updates across all rebake rounds.
    pub alpha_iters_attempted: usize,
    /// Candidate alpha updates whose certified pass completed.
    pub alpha_candidate_steps: usize,
    /// Candidate updates using the legacy normalized rule.
    pub alpha_legacy_steps: usize,
    /// Candidate updates using Adam.
    pub alpha_adam_steps: usize,
    /// Certified alpha passes across all rebake rounds.
    pub alpha_passes: usize,
    /// Largest failing-column population seen by a rebake alpha run.
    pub alpha_max_fail_classes: usize,
    /// Largest effective steering width seen by a rebake alpha run.
    pub alpha_max_steer_cols: usize,
    /// Alpha neurons committed as candidate rebake seeds.
    pub alpha_moved_neurons: usize,
    /// Candidate forward tableaux built successfully.
    pub rebuilds_completed: usize,
    /// Fixed-point exits before a redundant rebuild.
    pub fixed_point_stops: usize,
    /// Affordability exits before a round began.
    pub affordability_refusals: usize,
    /// Build or published-evaluation aborts.
    pub aborts: usize,
    /// Certified published root bound (min over adv classes of `root_eval`'s
    /// `dj`) of the build this run STARTED from.
    pub baseline: f64,
    /// Same number for the build this run left behind (>= `baseline` by
    /// round-level monotone accept; == when no round was accepted).
    pub best: f64,
    /// Rounds that ran to a scored candidate.
    pub rounds_run: usize,
    /// Rounds whose candidate was accepted.
    pub rounds_accepted: usize,
}

/// Frozen-policy entry: no-op unless re-bake is armed.
///
/// Called from `lane_impl` after `alpha_opt::maybe_optimize_root_alpha`,
/// while the root (and its Tier-0 retention) is still owned mutably.
/// `root_build_secs` is the measured cost of the initial build — the honest
/// estimate for a round's rebuild until the first rebuild re-measures it.
#[allow(clippy::too_many_arguments)]
pub fn maybe_rebake_root(
    net: &TwinNet,
    root: &mut RootGates,
    retained: &mut Option<RetainedRows>,
    retain_cfg: Option<&RetainCfg>,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    root_build_secs: f64,
    policy: super::RootAlphaRuntimePolicy,
    backward_interm_policy: super::BackwardIntermRuntimePolicy,
) -> Option<RebakeReport> {
    if !policy.rebake_enabled {
        return None;
    }
    let bi_base = backward_interm_policy.config(root.mode);
    rebake_root(
        net,
        root,
        retained,
        retain_cfg,
        t,
        adv,
        deadline,
        root_build_secs,
        policy,
        bi_base,
    )
}

/// The loop proper (test-visible; the policy gate lives in
/// [`maybe_rebake_root`]).
#[allow(clippy::too_many_arguments)]
/// `#rebake-round1-pricing` — price round 1 against the HALVED interm dose it
/// will actually run, instead of against the full initial build.
///
/// SHIPPED DEFAULT: `false`, so the affordability gate is byte-identical to the
/// measured 4/4 refusal until a sweep clears the armed value. Arming this makes
/// the gate ACCEPT rounds it currently declines, which spends budget — that is a
/// MOAT question (a round that overruns can cost a proof), not a free win.
///
/// SOUNDNESS: scheduling only. `#rebake` chooses among internally certified
/// builds by a monotone accept; admitting one more round cannot make a bound
/// tighter than the truth, only spend wall.
const REBAKE_PRICE_HALVED_INTERM: bool = false;

pub(crate) fn rebake_root(
    net: &TwinNet,
    root: &mut RootGates,
    retained: &mut Option<RetainedRows>,
    retain_cfg: Option<&RetainCfg>,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    root_build_secs: f64,
    policy: super::RootAlphaRuntimePolicy,
    bi_base: Option<backward_interm::BiCfg>,
) -> Option<RebakeReport> {
    let t0 = Instant::now();
    let max_rounds = policy.rebake_rounds;
    let secs_budget = policy.rebake_secs;
    if !root.mode.outward() {
        // Parity is the bit-parity oracle against core.py; it must not move.
        eprintln!("[rebake] refuse: parity mode");
        return None;
    }
    if adv.is_empty() || root.layers.iter().all(|l| l.unst.is_empty()) {
        eprintln!("[rebake] refuse: nothing to optimize (no adv classes or no unstable neurons)");
        return None;
    }
    // Never starve the tree search: cap at 40% of the remaining deadline.
    let budget = match deadline {
        Some(dl) => secs_budget.min(0.4 * dl.saturating_duration_since(t0).as_secs_f64()),
        None => secs_budget,
    };
    let deadline_capped = budget < secs_budget;
    // Engagement telemetry (R9): an armed run must be visible in one line
    // even when every round below is refused.
    eprintln!(
        "[rebake] armed: rounds={max_rounds} budget={budget:.1}s build_est={root_build_secs:.1}s"
    );
    // The round objective is root_eval's OWN number — the frame that converts
    // rows — so the baseline pass also prices one candidate evaluation.
    let te = Instant::now();
    let baseline = match certified_root_bound(net, root, t, adv) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[rebake] refuse: baseline root_eval: {e}");
            return None;
        }
    };
    let mut eval_secs = te.elapsed().as_secs_f64().max(1e-3);
    let mut build_secs = root_build_secs.max(1e-3);
    // Base interm dose for the rebuild rounds: the armed config's budget
    // HALVED (the shipped one-shot dose is measured saturated; a re-run over
    // already-tightened gates converges faster), re-capped per round below.
    let mut best = baseline;
    let mut rounds_run = 0usize;
    let mut rounds_accepted = 0usize;
    let mut rounds_started = 0usize;
    let mut alpha_completed = 0usize;
    let mut alpha_iters_attempted = 0usize;
    let mut alpha_candidate_steps = 0usize;
    let mut alpha_legacy_steps = 0usize;
    let mut alpha_adam_steps = 0usize;
    let mut alpha_passes = 0usize;
    let mut alpha_max_fail_classes = 0usize;
    let mut alpha_max_steer_cols = 0usize;
    let mut alpha_moved_neurons = 0usize;
    let mut rebuilds_completed = 0usize;
    let mut fixed_point_stops = 0usize;
    let mut affordability_refusals = 0usize;
    let mut aborts = 0usize;
    // Did the CURRENT root enter with alpha differing from its own bake?
    // `lane_impl` builds the entry tableau with the exact area heuristic and
    // only then lets #alpha-opt mutate its stored alpha. Compare against that
    // heuristic rather than asking whether alpha is fractional: projection can
    // accept a real endpoint flip (0→1 or 1→0), which is still unbaked even
    // though every stored value is binary. Round 1 may rebuild such an entry
    // even if its own ascent is already at a fixed point.
    let entry_unbaked_alpha = root.layers.iter().any(|layer| {
        alpha_differs_from_area_heuristic(&layer.l, &layer.u, &layer.alpha, &layer.unst)
    });
    // #rebake-round1-pricing. `build_secs` is the FULL initial root build, but
    // round 1 rebuilds with the interm dose HALVED (see the `bi_round` capping
    // below, `cfg.secs * 0.5`). Pricing round 1 at the full build therefore
    // OVER-CHARGES it by the half of the interm work the round will not do.
    //
    // Measured consequence (banked, 4/4): the gate refused with
    // `need 23.7-26.2s vs rem 17.8-19.0s`, short 4.7-8.4s, on every row — and
    // that refusal was attributed to the ascent skimming freed time rather than
    // to this bookkeeping. Backward-interm is ~72% of the build, so the true
    // round-1 cost is ~0.64x the initial build, not 1.0x.
    //
    // Only round 1 is affected: `build_secs` is re-measured at the bottom of the
    // loop from a round that ALREADY ran the halved dose.
    //
    // Ships OFF: the shipped refusal behaviour is byte-identical until armed.
    let bi_work = root.backward_interm_execution.measured_work_secs.max(0.0);
    // ⚠ The round's interm budget is `min(cfg.secs * 0.5, rem_now)`, and the phase
    // costs `min(natural_work, that_budget)`. So HALVING THE BUDGET ONLY SAVES
    // ANYTHING WHEN THE BUDGET BINDS. Measured 2026-09-17:
    //   shipped : bi_work 25.6s under a 40s config -> round pays min(25.6, 20.0)
    //             = 20.0s, a real saving of 5.6s (NOT 12.8s)
    //   bi_topk : bi_work 13.0s under a 40s config -> round pays min(13.0, 20.0)
    //             = 13.0s, a real saving of ZERO
    // Crediting `0.5 * bi_work` unconditionally — which is what I first wrote —
    // OVER-CREDITS in both cases and credits 6.5s of nothing under bi_topk. That
    // is the dangerous direction: it makes the gate ACCEPT rounds it cannot
    // afford, overrunning mid-round and risking a proof.
    let bi_cfg_secs = root.backward_interm_execution.configured_secs.max(0.0);
    let bi_round_cost = bi_work.min(0.5 * bi_cfg_secs);
    let bi_saving = (bi_work - bi_round_cost).max(0.0);
    for round in 1..=max_rounds {
        let rem = budget - t0.elapsed().as_secs_f64();
        let round1_credit = if REBAKE_PRICE_HALVED_INTERM && round == 1 {
            // Credit only the saving the halved budget ACTUALLY produces.
            bi_saving
        } else {
            0.0
        };
        let need = build_secs - round1_credit + eval_secs + 1.0;
        if rem < need {
            affordability_refusals += 1;
            // The lever's own observable: `credit` is 0.0 when disarmed, so a
            // refusal that WOULD have been afforded under corrected pricing is
            // one grep away without re-running anything.
            eprintln!(
                "[rebake] refuse round={round}: remaining {rem:.1}s < {need:.1}s \
(build {build_secs:.1}s - round1_credit {round1_credit:.1}s + eval {eval_secs:.1}s + alpha 1s) \
armed={} bi_work={bi_work:.1}s — budget honesty, previous build stands",
                u8::from(REBAKE_PRICE_HALVED_INTERM)
            );
            break;
        }
        rounds_started += 1;
        // (1) Alpha ascent on the current build. Its sub-slice is half of
        // what remains after reserving the rebuild + eval (alpha_opt also
        // re-caps against the instance deadline internally).
        let alpha_secs = ((rem - build_secs - eval_secs) * 0.5).max(1.0);
        // The round's own published-frame accept (step 3) already scores this
        // build through `root_eval`, so the ascent's guard would duplicate two
        // evaluations per round for nothing — disable it HERE and only here.
        // Snapshot the only two fields the ascent may mutate. The candidate
        // alpha is passed to `build_rebaked` as a seed; the accepted root is
        // replaced atomically only after its independent published-frame score
        // improves. Every error/refusal path therefore really leaves the
        // previous `(root, retained)` pair standing.
        let previous_alpha = snapshot_alpha_state(root);
        let rep = alpha_opt::optimize_root_alpha(
            net,
            root,
            t,
            adv,
            deadline,
            alpha_opt::AlphaOptCfg::for_rebake(policy, ROUND_ALPHA_ITERS, alpha_secs),
        );
        if let Some(report) = &rep {
            alpha_completed += 1;
            alpha_iters_attempted += report.iters;
            alpha_candidate_steps += report.candidate_steps;
            alpha_legacy_steps += report.legacy_steps;
            alpha_adam_steps += report.adam_steps;
            alpha_passes += report.passes;
            alpha_max_fail_classes = alpha_max_fail_classes.max(report.fail_classes);
            alpha_max_steer_cols = alpha_max_steer_cols.max(report.max_steer_cols);
            alpha_moved_neurons += report.moved_neurons;
        }
        let ascent_accepted = rep.as_ref().is_some_and(|r| r.accepted);
        if !(ascent_accepted || (round == 1 && entry_unbaked_alpha)) {
            restore_alpha_state(root, previous_alpha);
            fixed_point_stops += 1;
            // Fixed point: after an accepted rebuild the stored alpha IS the
            // baked alpha, so an ascent that cannot move it means a rebuild
            // would reproduce the current tableau bit-identically.
            eprintln!(
                "[rebake] stop round={round}: alpha ascent at a fixed point, nothing to bake"
            );
            break;
        }
        // (2) Rebuild the forward tableau with the optimized alpha baked in.
        let seed: Vec<Vec<f64>> = root.layers.iter().map(|l| l.alpha.clone()).collect();
        // #head-alpha: the builder never installs head slopes; carry the
        // ascent's onto the rebuilt candidate (valid on any box).
        let head_seed = root.head_alpha.clone();
        restore_alpha_state(root, previous_alpha);
        let bi_round = bi_base.map(|mut cfg| {
            let rem_now = (budget - t0.elapsed().as_secs_f64() - eval_secs).max(0.0);
            cfg.secs = (cfg.secs * 0.5).min(rem_now);
            cfg
        });
        // Rebuild under the exact precision configuration the initial root
        // consumed. Reopening process-global gates here would let one scored
        // run publish a mixed-policy tree under a single frozen receipt.
        let precision_policy = root.root_precision_policy;
        let tb = Instant::now();
        let built = RootGates::build_rebaked(
            net,
            &root.lo,
            &root.hi,
            root.mode,
            deadline,
            retain_cfg,
            &seed,
            bi_round,
            precision_policy,
        );
        let (mut cand, cand_ret) = match built {
            Ok(v) => v,
            Err(e) => {
                aborts += 1;
                eprintln!("[rebake] abort round={round}: rebuild: {e} — previous build stands");
                break;
            }
        };
        rebuilds_completed += 1;
        cand.head_alpha = head_seed;
        build_secs = tb.elapsed().as_secs_f64().max(1e-3);
        // (3) Score by the published frame; monotone accept at the round
        // level. Acceptance chooses among internally certified builds — the
        // comparison itself carries no soundness weight.
        let tv = Instant::now();
        let cand_bound = match certified_root_bound(net, &cand, t, adv) {
            Ok(b) => b,
            Err(e) => {
                aborts += 1;
                eprintln!("[rebake] abort round={round}: candidate root_eval: {e} — previous build stands");
                break;
            }
        };
        eval_secs = eval_secs.max(tv.elapsed().as_secs_f64());
        rounds_run = round;
        let accepted = cand_bound > best;
        eprintln!(
            "[rebake] round={round} root_bound_before={best:.6} after={cand_bound:.6} \
accepted={accepted} build_secs={build_secs:.2} total_secs={:.2}",
            t0.elapsed().as_secs_f64()
        );
        if !accepted {
            break; // deterministic step: a repeat would only repeat
        }
        *root = cand;
        *retained = cand_ret;
        best = cand_bound;
        rounds_accepted += 1;
    }
    eprintln!(
        "[rebake] done rounds={rounds_run} accepted={rounds_accepted} \
root_bound_before={baseline:.6} after={best:.6} secs={:.2}",
        t0.elapsed().as_secs_f64()
    );
    Some(RebakeReport {
        policy,
        effective_secs: budget,
        deadline_capped,
        budget_exhausted: affordability_refusals > 0,
        round_cap_exhausted: rounds_started == max_rounds,
        entry_unbaked_alpha,
        rounds_started,
        alpha_completed,
        alpha_iters_attempted,
        alpha_candidate_steps,
        alpha_legacy_steps,
        alpha_adam_steps,
        alpha_passes,
        alpha_max_fail_classes,
        alpha_max_steer_cols,
        alpha_moved_neurons,
        rebuilds_completed,
        fixed_point_stops,
        affordability_refusals,
        aborts,
        baseline,
        best,
        rounds_run,
        rounds_accepted,
    })
}

/// The round objective: `root_eval`'s own certified number for this build,
/// collapsed to the min over adv classes (the worst class is what a
/// conversion needs to cross zero). Errors bubble to the caller's abort path.
///
/// #root-eval-reuse (throughput plan B1(c)): the evaluation is STORED on
/// `root` (`bab::root_eval_store`), so the tree's own root evaluation of the
/// same gates generation (`bab::run_inner`) and the alpha ascent's frozen
/// y-rows (`alpha_opt::search_alpha`) reuse it instead of recomputing it. The
/// returned number is unchanged.
pub(crate) fn certified_root_bound(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
) -> ny_core::Result<f64> {
    let eng = BackwardEngine::new(net, root);
    let re = root_eval_store(&eng, net, t, adv)?;
    Ok(re.dj.iter().copied().fold(f64::INFINITY, f64::min))
}

// ---------------------------------------------------------------------------
//  #joint-refresh — the backward-only joint loop (cct plan P5 step 2)
// ---------------------------------------------------------------------------

/// #joint-refresh: alpha-ascent iterations per round. Small on purpose: the
/// joint loop's value is ALTERNATION (refresh the frame, re-steer the alpha,
/// refresh again), and the census on the cct ship pin showed the frozen-frame
/// ascent SATURATING by iteration ~12 (idx448: -0.2957 at iter 12, -0.2943 at
/// iter 20) or inert outright (idx1031: 0/20 accepted, 2 of 3 classes pinned
/// by the frozen head frame) — a longer ascent on a frozen frame buys nothing.
pub(crate) const JOINT_ROUND_ALPHA_ITERS: usize = 4;

/// #joint-refresh: the certified `root_eval` cost assumed before the first one
/// is measured (a memo-served baseline costs ~0 s and would under-price the
/// round). Measured 4.5 s per evaluation on cct idx448 (`[mr-census]
/// site.root_eval=2:18:9.091`); refusing a round is the fail-safe direction.
const JOINT_DEFAULT_EVAL_SECS: f64 = 4.0;

/// #joint-refresh: the ascent slice assumed before one is measured (4 passes
/// at the census's 0.385-0.47 s per R=3..5 pass, plus the head-frame rebuild).
const JOINT_DEFAULT_ASCENT_SECS: f64 = 2.0;

/// What one joint-refresh run did (telemetry + test surface).
#[derive(Debug, Clone)]
pub struct JointRefreshReport {
    /// Effective wall slice after the 40%-of-remaining cap.
    pub effective_secs: f64,
    /// Rounds that started a refresh.
    pub rounds_started: usize,
    /// Rounds whose candidate replaced the root.
    pub rounds_accepted: usize,
    /// Columns re-bounded / tightened / stabilized across all rounds.
    pub columns: usize,
    pub tightened: usize,
    pub stabilized: usize,
    /// Cropped (#bi-window) and dense prefix pass pairs across all rounds.
    pub windowed_pairs: usize,
    pub dense_pairs: usize,
    /// Certified published root bound of the root this run STARTED from.
    pub baseline: f64,
    /// Same number for the root it left behind (>= `baseline` by round-level
    /// monotone accept).
    pub best: f64,
    /// Why the loop stopped.
    pub stop: &'static str,
}

fn joint_done(t0: Instant, budget: f64, report: JointRefreshReport) -> Option<JointRefreshReport> {
    // The lever's OWN terminal receipt, printed on EVERY armed return path
    // (refusals included), so "ran and changed nothing" is distinguishable
    // from "never ran" in one grep.
    eprintln!(
        "[joint-refresh] done rounds={} accepted={} columns={} tightened={} stabilized={} \
windowed_pairs={} dense_pairs={} root_before={:.6} after={:.6} secs={:.2} budget={budget:.1} \
stop={}",
        report.rounds_started,
        report.rounds_accepted,
        report.columns,
        report.tightened,
        report.stabilized,
        report.windowed_pairs,
        report.dense_pairs,
        report.baseline,
        report.best,
        t0.elapsed().as_secs_f64(),
        report.stop
    );
    Some(report)
}

/// Frozen-policy entry: no-op (and silent) unless `margin_row.joint_refresh`
/// requests at least one round.
///
/// Called from `lane_impl` after `#alpha-opt` and `#rebake`, while the root is
/// still owned mutably and before `#spec-alpha` or any consumer reads it.
/// `known_baseline` is the certified `root_eval` bound the previous owner
/// already computed for EXACTLY the current gates (`None` = compute it here).
#[allow(clippy::too_many_arguments)]
pub fn maybe_joint_refresh_root(
    net: &TwinNet,
    root: &mut RootGates,
    retained_present: bool,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    known_baseline: Option<f64>,
    policy: super::JointRefreshRuntimePolicy,
    root_alpha: super::RootAlphaRuntimePolicy,
    backward_interm: super::BackwardIntermRuntimePolicy,
) -> Option<JointRefreshReport> {
    if policy.rounds == 0 {
        return None;
    }
    joint_refresh_root(
        net,
        root,
        retained_present,
        t,
        adv,
        deadline,
        known_baseline,
        policy,
        root_alpha,
        backward_interm,
    )
}

/// #joint-refresh: the loop proper (test-visible; the policy gate lives in
/// [`maybe_joint_refresh_root`]).
///
/// Per round:
///   1. REFRESH — `backward_interm::refresh_root_intermediates` re-derives the
///      intermediate boxes of every trunk layer from the CURRENT gates (the
///      committed alpha included), shrink-only, and re-certifies the touched
///      gates; the result is a NEW candidate root.
///   2. SCORE the refreshed candidate with the unchanged certified `root_eval`
///      (`certified_root_bound`, which also STORES the evaluation, so the
///      ascent below borrows its y-rows instead of recomputing them).
///   3. ASCEND — a short `#alpha-opt` ascent on the candidate. Its frozen
///      evaluation frame (y-box, head gates, margin seeds, m1/m2v) is built
///      FROM THE REFRESHED GATES: this is the per-round head y-box refresh. If
///      it commits, the candidate is scored again and the ascent is KEPT only
///      on a strict improvement (otherwise the refreshed alpha is restored).
///   4. ACCEPT the round only if the candidate's certified bound strictly
///      improves on the current root's; otherwise stop (the refresh is a
///      deterministic function of the gates, so a rejected round would repeat).
///
/// SOUNDNESS. Every candidate is internally certified by construction (the
/// refresh's own contract: certified enclosures intersected shrink-only, gates
/// re-certified, `ms` re-derived; the ascent's: any `alpha in [0, 1]`, `ms`
/// re-derived on commit), and acceptance only CHOOSES among such roots on the
/// certified number itself. The published bound is recomputed from the
/// accepted root by the unchanged `root_eval` (a memo hit on the stored
/// evaluation of exactly these gates). Every refusal, abort and expiry leaves
/// the previously accepted root standing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_refresh_root(
    net: &TwinNet,
    root: &mut RootGates,
    retained_present: bool,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    known_baseline: Option<f64>,
    policy: super::JointRefreshRuntimePolicy,
    root_alpha: super::RootAlphaRuntimePolicy,
    backward_interm: super::BackwardIntermRuntimePolicy,
) -> Option<JointRefreshReport> {
    let t0 = Instant::now();
    // Never starve the tree search: cap at 40% of the remaining deadline (the
    // `#alpha-opt` / `#backward-interm` / `#rebake` starvation guard).
    let budget = match deadline {
        Some(dl) => policy
            .secs
            .min(0.4 * dl.saturating_duration_since(t0).as_secs_f64()),
        None => policy.secs,
    };
    // Engagement telemetry (R9): an armed run is visible in one line even when
    // every round below is refused.
    eprintln!(
        "[joint-refresh] armed: rounds={} budget={budget:.1}s topk={} chunk={} window={} \
iters_per_round={JOINT_ROUND_ALPHA_ITERS}",
        policy.rounds, backward_interm.topk, backward_interm.chunk, backward_interm.window
    );
    // #bi-gpu: the refresh's prefix pairs on the certified device route, one
    // state (and one `[bi-gpu] done ... scope=joint_refresh` receipt, printed
    // when it drops on ANY return below) across all rounds. Dark: `None`, and
    // every pair is the historical CPU pair.
    // R2: refused outside tests (`[bi-gpu] refused`), leaving `None`.
    let mut joint_gpu = if backward_interm.enabled && backward_interm.gpu {
        BiGpu::armed(BiGpuScope::JointRefresh)
    } else {
        None
    };
    let mut report = JointRefreshReport {
        effective_secs: budget,
        rounds_started: 0,
        rounds_accepted: 0,
        columns: 0,
        tightened: 0,
        stabilized: 0,
        windowed_pairs: 0,
        dense_pairs: 0,
        baseline: f64::NAN,
        best: f64::NAN,
        stop: "refused",
    };
    let refusal = if !root.mode.outward() {
        Some("parity mode")
    } else if !backward_interm.enabled {
        Some("backward_interm is not armed (the refresh re-runs that phase's frozen policy)")
    } else if retained_present {
        Some("Tier-0/epoch retained rows are present (they index the unrefreshed frame)")
    } else if root.spec_alpha_policy.is_some() {
        Some("#spec-alpha is already installed")
    } else if adv.is_empty() || root.layers.iter().all(|layer| layer.unst.is_empty()) {
        Some("nothing to refresh (no adv classes or no unstable neurons)")
    } else {
        None
    };
    if let Some(why) = refusal {
        eprintln!("[joint-refresh] refuse: {why}");
        return joint_done(t0, budget, report);
    }
    // The round objective is `root_eval`'s OWN number for the current gates.
    let mut eval_secs = JOINT_DEFAULT_EVAL_SECS;
    let baseline = match known_baseline.filter(|value| value.is_finite()) {
        Some(value) => value,
        None => {
            let te = Instant::now();
            match certified_root_bound(net, root, t, adv) {
                Ok(value) if value.is_finite() => {
                    eval_secs = eval_secs.max(te.elapsed().as_secs_f64());
                    value
                }
                Ok(_) => {
                    eprintln!("[joint-refresh] refuse: non-finite baseline root_eval");
                    return joint_done(t0, budget, report);
                }
                Err(error) => {
                    eprintln!("[joint-refresh] refuse: baseline root_eval: {error}");
                    return joint_done(t0, budget, report);
                }
            }
        }
    };
    report.baseline = baseline;
    report.best = baseline;
    let mut ascent_secs = JOINT_DEFAULT_ASCENT_SECS;
    report.stop = "round_cap";
    for round in 1..=policy.rounds {
        let rem = budget - t0.elapsed().as_secs_f64();
        // Reserve two certified evaluations and one ascent slice; the refresh
        // gets the rest and is cut at its slice (a partial refresh is sound).
        let refresh_slice = rem - (2.0 * eval_secs + ascent_secs);
        if refresh_slice < 0.5 {
            eprintln!(
                "[joint-refresh] refuse round={round}: remaining {rem:.1}s < 2 x eval \
{eval_secs:.1}s + ascent {ascent_secs:.1}s + 0.5s refresh — budget honesty, previous root \
stands"
            );
            report.stop = "budget";
            break;
        }
        report.rounds_started += 1;
        // Panic-free slice end. `joint_refresh_secs` is admitted up to
        // `f64::MAX`, and with no instance deadline the 40% cap never applies,
        // so `Duration::from_secs_f64` / `Instant + Duration` could PANIC on a
        // huge request inside a verdict lane (a panic is a lost row). An
        // unrepresentable slice end means "no slice cut": the instance deadline,
        // if any, still stops the refresh, and a cut is only ever a looser
        // (still sound) frame.
        let slice_end = std::time::Duration::try_from_secs_f64(refresh_slice)
            .ok()
            .and_then(|slice| Instant::now().checked_add(slice));
        let stop_at = match (slice_end, deadline) {
            (Some(end), Some(dl)) => Some(end.min(dl)),
            (Some(end), None) => Some(end),
            (None, dl) => dl,
        };
        // (1) Refresh.
        // #bi-patch: the refresh's prefix pairs run as per-column
        // receptive-field walks when the frozen phase policy arms patch mode
        // (`cfg()` is `None` when dark: exactly the historical refresh).
        let (mut cand, stats) = match backward_interm::refresh_root_intermediates_with_patch(
            net,
            root,
            backward_interm.topk,
            backward_interm.chunk,
            backward_interm.window,
            backward_interm.patch.cfg(),
            stop_at,
            joint_gpu.as_mut(),
        ) {
            Ok(value) => value,
            Err(error) => {
                eprintln!(
                    "[joint-refresh] abort round={round}: refresh: {error} — previous root stands"
                );
                report.stop = "refresh_error";
                break;
            }
        };
        report.columns += stats.columns;
        report.tightened += stats.tightened;
        report.stabilized += stats.stabilized;
        report.windowed_pairs += stats.windowed_pairs;
        report.dense_pairs += stats.dense_pairs;
        if stats.tightened == 0 {
            // The gates are unchanged, so an ascent here would re-solve the
            // problem the previous owner already solved.
            eprintln!(
                "[joint-refresh] stop round={round}: refresh tightened nothing \
(columns={} cut={}) — fixed point",
                stats.columns, stats.deadline_cut
            );
            report.stop = "fixed_point";
            break;
        }
        // (2) Score the refreshed frame (and store its y-rows for the ascent).
        let te = Instant::now();
        let refreshed = match certified_root_bound(net, &cand, t, adv) {
            Ok(value) if value.is_finite() => value,
            Ok(_) => {
                eprintln!(
                    "[joint-refresh] abort round={round}: non-finite candidate root_eval — \
previous root stands"
                );
                report.stop = "eval_error";
                break;
            }
            Err(error) => {
                eprintln!(
                    "[joint-refresh] abort round={round}: candidate root_eval: {error} — \
previous root stands"
                );
                report.stop = "eval_error";
                break;
            }
        };
        eval_secs = eval_secs.max(te.elapsed().as_secs_f64());
        // (3) A short ascent on the refreshed frame; its own published-frame
        // guard is off because this loop scores the candidate itself.
        let mut after = refreshed;
        let mut ascent_iters = 0usize;
        let mut ascent_kept = false;
        let alpha_secs = budget - t0.elapsed().as_secs_f64() - eval_secs;
        if alpha_secs >= 0.5 {
            let snapshot = snapshot_alpha_state(&cand);
            let ta = Instant::now();
            let rep = alpha_opt::optimize_root_alpha(
                net,
                &mut cand,
                t,
                adv,
                deadline,
                alpha_opt::AlphaOptCfg::for_rebake(root_alpha, JOINT_ROUND_ALPHA_ITERS, alpha_secs),
            );
            ascent_secs = ascent_secs.max(ta.elapsed().as_secs_f64());
            if let Some(rep) = rep {
                ascent_iters = rep.iters;
                if rep.accepted {
                    let tv = Instant::now();
                    match certified_root_bound(net, &cand, t, adv) {
                        Ok(value) if value.is_finite() && value > refreshed => {
                            after = value;
                            ascent_kept = true;
                        }
                        _ => restore_alpha_state(&mut cand, snapshot),
                    }
                    eval_secs = eval_secs.max(tv.elapsed().as_secs_f64());
                }
            }
        }
        // (4) Round-level monotone accept on the certified number.
        let accepted = after > report.best;
        eprintln!(
            "[joint-refresh] round={round} layers={} columns={} tightened={} stabilized={} \
nonfinite={} windowed_pairs={} dense_pairs={} patch_pairs={} patch_fallbacks={} \
window_fallbacks={} row_ratio={:.4} \
refresh_secs={:.2} cut={} ascent_iters={ascent_iters} ascent_kept={ascent_kept} \
root_before={:.6} refreshed={refreshed:.6} after={after:.6} accepted={accepted} \
total_secs={:.2}",
            stats.layers,
            stats.columns,
            stats.tightened,
            stats.stabilized,
            stats.nonfinite,
            stats.windowed_pairs,
            stats.dense_pairs,
            stats.patch_pairs,
            stats.patch_fallbacks,
            stats.window_fallbacks,
            if stats.rows_dense > 0 {
                stats.rows_window as f64 / stats.rows_dense as f64
            } else {
                1.0
            },
            stats.secs,
            stats.deadline_cut,
            report.best,
            t0.elapsed().as_secs_f64()
        );
        if !accepted {
            report.stop = "rejected";
            break;
        }
        *root = cand;
        report.best = after;
        report.rounds_accepted += 1;
    }
    joint_done(t0, budget, report)
}

#[cfg(test)]
mod entry_alpha_tests {
    use super::alpha_differs_from_area_heuristic;

    #[test]
    fn binary_endpoint_flips_are_still_unbaked() {
        let lower = [-0.25, -0.75];
        let upper = [0.75, 0.25];
        let unstable = [0usize, 1usize];

        assert!(!alpha_differs_from_area_heuristic(
            &lower,
            &upper,
            &[1.0, 0.0],
            &unstable,
        ));
        assert!(alpha_differs_from_area_heuristic(
            &lower,
            &upper,
            &[0.0, 0.0],
            &unstable,
        ));
        assert!(alpha_differs_from_area_heuristic(
            &lower,
            &upper,
            &[1.0, 1.0],
            &unstable,
        ));
    }
}

#[cfg(test)]
mod round1_pricing_tests {
    use super::REBAKE_PRICE_HALVED_INTERM;

    /// The CORRECTED model. The round's interm budget is `min(cfg*0.5, rem)` and
    /// the phase costs `min(natural_work, budget)`, so halving the budget saves
    /// something ONLY when the budget binds.
    fn saving(bi_work: f64, bi_cfg: f64) -> f64 {
        (bi_work - bi_work.min(0.5 * bi_cfg)).max(0.0)
    }
    fn need(armed: bool, round: usize, build: f64, bi_work: f64, bi_cfg: f64, eval: f64) -> f64 {
        let c = if armed && round == 1 {
            saving(bi_work, bi_cfg)
        } else {
            0.0
        };
        build - c + eval + 1.0
    }

    #[test]
    fn round1_pricing_ships_off() {
        const {
            assert!(
                !REBAKE_PRICE_HALVED_INTERM,
                "arming spends budget: a MOAT question"
            )
        };
    }

    /// ⚠ THE BUG I SHIPPED FIRST AND CAUGHT BY MEASUREMENT. Crediting
    /// `0.5 * bi_work` unconditionally credits a saving that does not exist when
    /// the phase is already under budget — and over-crediting makes the gate
    /// ACCEPT rounds it cannot afford, overrunning mid-round. Measured 2026-09-17
    /// under `bi_topk:128`: bi_work 13.0s under a 40s config, so the round pays
    /// the SAME 13.0s and the true saving is ZERO, while the naive formula
    /// credited 6.5s.
    #[test]
    fn credit_is_zero_when_the_interm_budget_does_not_bind() {
        assert_eq!(
            saving(13.0, 40.0),
            0.0,
            "13.0s of work under a 20.0s half-budget saves nothing"
        );
        assert!(
            (0.5_f64 * 13.0 - 6.5).abs() < 1e-9,
            "the naive formula would have credited 6.5s"
        );
        assert_eq!(
            need(true, 1, 22.6, 13.0, 40.0, 2.5),
            need(false, 1, 22.6, 13.0, 40.0, 2.5)
        );
    }

    /// When the budget DOES bind, the saving is the part above the half-budget.
    #[test]
    fn credit_is_the_part_of_the_work_above_the_half_budget() {
        assert!(
            (saving(25.6, 40.0) - 5.6).abs() < 1e-9,
            "25.6 - min(25.6, 20.0) = 5.6"
        );
        assert!(
            saving(25.6, 40.0) < 0.5 * 25.6,
            "and it is LESS than the naive half-of-work"
        );
    }

    /// Both MEASURED configurations still refuse under correct pricing. This is
    /// the honest outcome: the pricing bug was real, the fix is real, and it does
    /// not make a round affordable at either build cost.
    #[test]
    fn neither_measured_configuration_affords_a_round() {
        // (label, build, bi_work, bi_cfg, eval, remaining) — all measured 2026-09-17.
        for (label, build, bi, cfg, eval, rem) in [
            ("shipped", 35.9, 25.6, 40.0, 2.3, 12.4),
            ("bi_topk", 22.6, 13.0, 40.0, 2.5, 17.9),
        ] {
            let n = need(true, 1, build, bi, cfg, eval);
            assert!(
                n > rem,
                "{label}: must still refuse (need {n:.1} vs rem {rem:.1})"
            );
        }
    }

    /// Only round 1 is credited; later rounds re-measure a build that already ran
    /// the halved dose, so crediting again would double-count.
    #[test]
    fn only_round_one_is_credited() {
        for r in 2..=4 {
            assert_eq!(
                need(true, r, 35.9, 25.6, 40.0, 2.3),
                need(false, r, 35.9, 25.6, 40.0, 2.3)
            );
        }
    }
}

#[cfg(test)]
mod joint_refresh_tests {
    //! #joint-refresh soundness pins: the refresh is shrink-only, contains
    //! every sampled pre-activation and re-certifies every touched gate; the
    //! loop never certifies a bound above a sampled margin (in particular never
    //! a positive one on a falsifiable property), only ever raises the
    //! published bound, and is dark by default.

    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::super::backward_interm::window_tests::chain_spec;
    use super::super::backward_interm::{filter_clip_rows, refresh_root_intermediates};
    use super::super::net::TwinNet;
    use super::super::root::{ClipRows, RootGates};
    use super::super::rounding::RoundMode;
    use super::super::{
        BackwardIntermRuntimePolicy, JointRefreshRuntimePolicy, RootAlphaRuntimePolicy,
    };
    use super::{certified_root_bound, joint_refresh_root, maybe_joint_refresh_root};

    const N_IN: usize = 128;

    fn fixture(seed: u64, scale: f64) -> (TwinNet, Vec<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let net = TwinNet::compile(&chain_spec(&mut rng, scale)).expect("compile");
        (net, vec![-0.4; N_IN], vec![0.35; N_IN])
    }

    fn samples(root: &RootGates, n: usize, seed: u64) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = Array2::<f64>::zeros((root.mid.len(), n));
        for i in 0..root.mid.len() {
            for b in 0..n {
                // Column 0 is the box midpoint itself, so a property falsified
                // at the midpoint is falsified among the samples.
                let u: f64 = if b == 0 {
                    0.0
                } else if b % 3 == 0 {
                    if rng.random_range(0.0..1.0) < 0.5 {
                        -1.0
                    } else {
                        1.0
                    }
                } else {
                    rng.random_range(-1.0..1.0)
                };
                x[[i, b]] = root.mid[i] + u * root.rad[i];
            }
        }
        x
    }

    /// Minimum over sampled points and adversarial classes of `Y_t - Y_j`.
    fn sampled_min_margin(net: &TwinNet, x: &Array2<f64>, t: usize, adv: &[usize]) -> f64 {
        let (y, _) = net
            .forward_points(x, &std::collections::BTreeMap::new())
            .expect("forward");
        let (w2, b2, (_, n_y)) = net.gemm2();
        let mut min = f64::INFINITY;
        for &j in adv {
            for b in 0..x.ncols() {
                let mut m = b2[t] - b2[j];
                for k in 0..n_y {
                    m += (w2[t * n_y + k] - w2[j * n_y + k]) * y[[k, b]].max(0.0);
                }
                min = min.min(m);
            }
        }
        min
    }

    /// (argmax, runner-up) classes at the box midpoint.
    fn midpoint_classes(net: &TwinNet, root: &RootGates) -> (usize, usize) {
        let mid = Array2::from_shape_fn((root.mid.len(), 1), |(i, _)| root.mid[i]);
        let (y, _) = net
            .forward_points(&mid, &std::collections::BTreeMap::new())
            .expect("forward");
        let (w2, b2, (n_out, n_y)) = net.gemm2();
        let scores: Vec<f64> = (0..n_out)
            .map(|o| {
                let mut s = b2[o];
                for k in 0..n_y {
                    s += w2[o * n_y + k] * y[[k, 0]].max(0.0);
                }
                s
            })
            .collect();
        let top = (0..n_out)
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        let second = (0..n_out)
            .filter(|&o| o != top)
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
            .unwrap();
        (top, second)
    }

    fn bi_policy(window: bool) -> BackwardIntermRuntimePolicy {
        BackwardIntermRuntimePolicy {
            enabled: true,
            secs: 1_000.0,
            chunk: 16,
            chunk_secs: 0.0,
            topk: 4_096,
            interm_alpha: 0,
            window,
            gpu: false,
            patch: super::super::BiPatchRuntimePolicy::default(),
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

    fn gate_bits(root: &RootGates) -> GateBits {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
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
            .collect()
    }

    /// SHRINK-ONLY, CONTAINED, RE-CERTIFIED — dense and cropped refresh both.
    #[test]
    fn joint_refresh_is_shrink_only_contained_and_recertified() {
        let (net, lo, hi) = fixture(0x7F_0001, 0.6);
        // A root WITHOUT backward-interm: the forward boxes are loose, so the
        // refresh has real work (a no-op fixture would make this test VOID).
        let root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let x = samples(&root, 192, 0x7F_0002);
        let sel: std::collections::BTreeMap<usize, Vec<usize>> = root
            .layers
            .iter()
            .map(|layer| (layer.op, (0..layer.n).collect()))
            .collect();
        let (_, pre) = net.forward_points(&x, &sel).expect("forward");
        for window in [false, true] {
            let (cand, stats) =
                refresh_root_intermediates(&net, &root, 4_096, 16, window, None, None)
                    .expect("refresh");
            assert!(
                stats.tightened > 0,
                "window={window}: the refresh tightened nothing — VOID fixture"
            );
            assert!(!stats.deadline_cut);
            if window {
                assert!(stats.windowed_pairs > 0, "the crop never engaged — VOID");
            } else {
                assert_eq!(stats.windowed_pairs, 0, "dark refresh must never crop");
            }
            for (li, (old, new)) in root.layers.iter().zip(&cand.layers).enumerate() {
                assert_eq!(old.n, new.n);
                let p = &pre[&old.op];
                for j in 0..old.n {
                    assert!(
                        new.l[j] >= old.l[j] && new.u[j] <= old.u[j],
                        "window={window} ({li},{j}): refresh WIDENED [{}, {}] -> [{}, {}]",
                        old.l[j],
                        old.u[j],
                        new.l[j],
                        new.u[j]
                    );
                    for b in 0..x.ncols() {
                        let v = p[[j, b]];
                        assert!(
                            v >= new.l[j] - 1e-9 && v <= new.u[j] + 1e-9,
                            "window={window} ({li},{j}): sampled {v} outside the refreshed box"
                        );
                    }
                    let (l, u, a, s, c) = (new.l[j], new.u[j], new.alpha[j], new.s[j], new.c[j]);
                    assert_eq!(
                        new.ms[j],
                        a.max(s),
                        "ms must be re-derived as max(alpha, s)"
                    );
                    assert!((0.0..=1.0).contains(&a), "alpha out of [0, 1]");
                    if l >= 0.0 {
                        assert_eq!(
                            (a, s, c),
                            (1.0, 1.0, 0.0),
                            "active-stable needs the exact line"
                        );
                    } else if u <= 0.0 {
                        assert_eq!(
                            (a, s, c),
                            (0.0, 0.0, 0.0),
                            "inactive-stable needs the exact line"
                        );
                    } else {
                        // The chord dominates relu at both kinks (real
                        // arithmetic is certified by `repair_upper_lines`; the
                        // f64 check here gets a rounding allowance).
                        let tol = 1e-12 * (1.0 + l.abs() + u.abs());
                        assert!(c >= 0.0 && s * l + c >= -tol && s * u + c >= u - tol);
                    }
                }
                // `unst` is exactly the old list's still-unstable neurons.
                let expect: Vec<usize> = old
                    .unst
                    .iter()
                    .copied()
                    .filter(|&j| new.l[j] < 0.0 && new.u[j] > 0.0)
                    .collect();
                assert_eq!(
                    new.unst, expect,
                    "window={window} layer {li}: unst mismatch"
                );
            }
        }
    }

    /// NEVER VERIFIES A FALSE BOUND. On a FALSIFIABLE property (target = the
    /// midpoint's runner-up, so the margin is negative at the midpoint) and on
    /// the argmax property, the joint loop's published bound stays at or below
    /// every sampled margin — and so is never positive on the false one — while
    /// only ever raising the bound it started from.
    #[test]
    fn joint_refresh_never_verifies_a_false_bound() {
        // Rounds ACCEPTED across every fixture: the published-bound checks
        // below are trivially true of an untouched root, so a run in which no
        // round is ever accepted exercises nothing of the accept path (VOID).
        let mut accepted_rounds = 0usize;
        for (seed, window) in [(0x7F_0101u64, false), (0x7F_0102, true), (0x7F_0103, true)] {
            let (net, lo, hi) = fixture(seed, 0.6);
            let probe = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
            let (top, second) = midpoint_classes(&net, &probe);
            let x = samples(&probe, 256, seed ^ 0xABCD);
            for t in [second, top] {
                let adv: Vec<usize> = (0..net.n_out).filter(|&o| o != t).collect();
                let mut root =
                    RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
                let report = joint_refresh_root(
                    &net,
                    &mut root,
                    false,
                    t,
                    &adv,
                    None,
                    None,
                    JointRefreshRuntimePolicy {
                        rounds: 3,
                        secs: 1_000.0,
                    },
                    RootAlphaRuntimePolicy::default(),
                    bi_policy(window),
                )
                .expect("armed run always reports");
                assert_ne!(report.stop, "refused", "VOID fixture: the loop refused");
                assert!(report.rounds_started > 0, "VOID fixture: no round started");
                assert!(
                    report.best >= report.baseline,
                    "round-level accept must be monotone"
                );
                let published = certified_root_bound(&net, &root, t, &adv).expect("root_eval");
                let truth = sampled_min_margin(&net, &x, t, &adv);
                assert!(
                    published <= truth + 1e-9,
                    "seed {seed:#x} t={t}: certified {published} ABOVE sampled margin {truth}"
                );
                if t == second {
                    assert!(
                        truth < 0.0,
                        "the runner-up property must be falsified at the midpoint"
                    );
                    assert!(published < 0.0, "a false property must never certify");
                }
                let tol = 1e-12 * (1.0 + published.abs());
                assert!(
                    (published - report.best).abs() <= tol,
                    "the published root must be the one the loop scored: {published} vs {}",
                    report.best
                );
                accepted_rounds += report.rounds_accepted;
            }
        }
        assert!(
            accepted_rounds > 0,
            "no joint-refresh round was accepted on any fixture: the accept path is VOID"
        );
    }

    /// DARK BY DEFAULT: zero rounds returns before touching anything and the
    /// root is bit-identical; an armed run without backward-interm refuses and
    /// leaves the root bit-identical too.
    #[test]
    fn joint_refresh_is_dark_by_default_and_refuses_without_backward_interm() {
        let (net, lo, hi) = fixture(0x7F_0201, 0.5);
        let mut root = RootGates::build(&net, &lo, &hi, RoundMode::Outward, None).expect("root");
        let before = gate_bits(&root);
        assert_eq!(JointRefreshRuntimePolicy::default().rounds, 0);
        assert!(maybe_joint_refresh_root(
            &net,
            &mut root,
            false,
            0,
            &[1, 2],
            None,
            None,
            JointRefreshRuntimePolicy::default(),
            RootAlphaRuntimePolicy::default(),
            bi_policy(true),
        )
        .is_none());
        assert!(
            gate_bits(&root) == before,
            "the dark entry must not touch the root"
        );

        let refused = maybe_joint_refresh_root(
            &net,
            &mut root,
            false,
            0,
            &[1, 2],
            None,
            None,
            JointRefreshRuntimePolicy {
                rounds: 2,
                secs: 1_000.0,
            },
            RootAlphaRuntimePolicy::default(),
            BackwardIntermRuntimePolicy::default(),
        )
        .expect("an armed run always reports");
        assert_eq!(refused.stop, "refused");
        assert_eq!(refused.rounds_started, 0);
        assert!(
            gate_bits(&root) == before,
            "a refusal must not touch the root"
        );

        let retained = maybe_joint_refresh_root(
            &net,
            &mut root,
            true,
            0,
            &[1, 2],
            None,
            None,
            JointRefreshRuntimePolicy {
                rounds: 2,
                secs: 1_000.0,
            },
            RootAlphaRuntimePolicy::default(),
            bi_policy(false),
        )
        .expect("an armed run always reports");
        assert_eq!(
            retained.stop, "refused",
            "retained Tier-0 rows index the old frame"
        );
        assert_eq!(gate_bits(&root), before);
    }

    /// Clip rows are indexed by POSITION in `unst`: every surviving neuron
    /// keeps exactly its own row, and a non-subsequence carries none.
    #[test]
    fn filter_clip_rows_keeps_each_row_on_its_neuron() {
        let old_unst = [2usize, 5, 7, 9];
        let n_in = 3;
        let rows = ClipRows {
            m: Array2::from_shape_fn((4, n_in), |(r, c)| (old_unst[r] * 10 + c) as f64),
            d: Array2::from_shape_fn((4, n_in), |(r, c)| (old_unst[r] * 100 + c) as f64),
            sl_lo: old_unst.iter().map(|&j| j as f64 + 0.25).collect(),
            sl_up: old_unst.iter().map(|&j| j as f64 + 0.5).collect(),
            bm: old_unst.iter().map(|&j| j as f64 + 0.75).collect(),
            bd: old_unst.iter().map(|&j| j as f64 + 0.125).collect(),
        };
        let kept = filter_clip_rows(&rows, &old_unst, &[5, 9]).expect("subsequence");
        assert_eq!(kept.m.nrows(), 2);
        for (r, &j) in [5usize, 9].iter().enumerate() {
            assert_eq!(kept.m[[r, 1]], (j * 10 + 1) as f64);
            assert_eq!(kept.d[[r, 2]], (j * 100 + 2) as f64);
            assert_eq!(kept.sl_lo[r], j as f64 + 0.25);
            assert_eq!(kept.sl_up[r], j as f64 + 0.5);
            assert_eq!(kept.bm[r], j as f64 + 0.75);
            assert_eq!(kept.bd[r], j as f64 + 0.125);
        }
        assert!(filter_clip_rows(&rows, &old_unst, &[5, 3]).is_none());
        assert!(filter_clip_rows(&rows, &old_unst, &[9, 5]).is_none());
        let same = filter_clip_rows(&rows, &old_unst, &old_unst).expect("identity");
        assert_eq!(same.sl_lo, rows.sl_lo);
        let empty = filter_clip_rows(&rows, &old_unst, &[]).expect("all stabilized");
        assert_eq!(empty.m.nrows(), 0);
    }
}
