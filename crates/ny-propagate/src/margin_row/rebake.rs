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
use super::bab::root_eval;
use super::backward_interm;
use super::engine::BackwardEngine;
use super::net::TwinNet;
use super::root::{RetainCfg, RetainedRows, RootGates};

/// Alpha-ascent iterations per round (the #alpha-opt default; the round's
/// wall sub-slice is the binding cap in practice).
const ROUND_ALPHA_ITERS: usize = 8;

type AlphaState = Vec<(Vec<f64>, Vec<f64>)>;

fn snapshot_alpha_state(root: &RootGates) -> AlphaState {
    root.layers
        .iter()
        .map(|layer| (layer.alpha.clone(), layer.ms.clone()))
        .collect()
}

fn restore_alpha_state(root: &mut RootGates, state: AlphaState) {
    assert_eq!(
        root.layers.len(),
        state.len(),
        "root layer count changed during alpha-only ascent"
    );
    for (layer, (alpha, ms)) in root.layers.iter_mut().zip(state) {
        layer.alpha = alpha;
        layer.ms = ms;
    }
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
    for round in 1..=max_rounds {
        let rem = budget - t0.elapsed().as_secs_f64();
        let need = build_secs + eval_secs + 1.0;
        if rem < need {
            affordability_refusals += 1;
            eprintln!(
                "[rebake] refuse round={round}: remaining {rem:.1}s < {need:.1}s \
(build {build_secs:.1}s + eval {eval_secs:.1}s + alpha 1s) — budget honesty, previous build stands"
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
        let (cand, cand_ret) = match built {
            Ok(v) => v,
            Err(e) => {
                aborts += 1;
                eprintln!("[rebake] abort round={round}: rebuild: {e} — previous build stands");
                break;
            }
        };
        rebuilds_completed += 1;
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
pub(crate) fn certified_root_bound(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
) -> ny_core::Result<f64> {
    let eng = BackwardEngine::new(net, root);
    let re = root_eval(&eng, net, t, adv)?;
    Ok(re.dj.iter().copied().fold(f64::INFINITY, f64::min))
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
