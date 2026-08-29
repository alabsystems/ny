// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #alpha-opt: projected supergradient optimization of the root trunk alpha
//! (#twinwall).
//!
//! WHY. The lane's lower-line slopes are born binary — `gates_from_box`
//! (`root.rs`) pins `alpha = [u >= -l]` per unstable neuron and nothing ever
//! moves them. alpha-beta-CROWN OPTIMIZES the same free parameter, and on the
//! cifar100 deep band (lane roots -0.61..-1.76) it proves at the root rows this
//! lane starts too deep to close (idx_7704: lane root -0.35, abc INIT=1 with
//! zero BaB). Every row the lane has ever proven started shallower than -0.57,
//! so the root bound is the binding constraint — this module moves it.
//!
//! WHAT. A root-phase loop (before the tree starts, never per-domain):
//! run one certified margin-seeded Lower pass under a candidate alpha, extract
//! a per-neuron supergradient from the SAME pass, step alpha toward a better
//! bound (projected to `[0, 1]`, restricted to unstable neurons), and keep the
//! best CERTIFIED objective ever seen (monotone accept — the committed alpha is
//! never worse than the binary heuristic under the frozen evaluation frame).
//!
//! THE GRADIENT (exact for the frozen-sign linearization; heuristic overall).
//! At trunk relu `li` the Lower-lane transform is `lv = vp*alpha_j + vn*s_j`
//! with `vp = max(v, 0)`, `vn = min(v, 0)` (`engine.rs`), so alpha enters ONLY
//! through the positive part: `d lv / d alpha_j = vp >= 0`. Holding fixed (a)
//! the sign selections the pass made at every other relu and (b) the concretize
//! argmin signs, the final bound is LINEAR in `lv`, and its coefficient is the
//! value at the argmin point `x*` of neuron `j`'s pre-activation under the
//! pass's own selected lines:
//!
//!   d bound_f / d alpha_{li,j} = vp_{li,j,f} * U_{li,j,f},
//!
//! where `x*_i = mid_i - sign(A_{i,f}) * rad_i` (the derivative of the
//! concretize `A@mid - |A|@rad` term is `mid_i - sign(A_i)*rad_i`, i.e. it
//! FLIPS with the coefficient sign — that is exactly what seeding the walk at
//! `x*` accounts for), and `U` is one cheap single-column forward walk of `x*`
//! through the linearized trunk (each earlier relu replaced by the affine line
//! the pass selected for that column: lower `alpha*z` on `v > 0`, upper chord
//! `s*z + c` on `v < 0`). Across sign flips the objective is only piecewise —
//! we do NOT chase the exact Clarke supergradient (alpha-beta-CROWN does not
//! either) — and the certified-slack terms' own alpha-dependence (the E-lane
//! grows with `ms = max(alpha, s)` when alpha exceeds `s`, plus the gamma
//! envelopes) is IGNORED in the direction: the direction is the gradient of
//! the frozen-sign LINEARIZATION, not of the slack-included certified bound.
//! Both approximations are direction-only: a wrong direction wastes one pass
//! and nothing more, because every ACCEPTED iterate is scored by the
//! unchanged certified Outward pass, slack included.
//!
//! SOUNDNESS.
//! * `alpha*y <= relu(y)` holds pointwise on all of R for ANY `alpha in
//!   [0, 1]` — validity of the lower line is box-independent and does not
//!   require binariness (the GPU seam's own docs state the same generality).
//! * The upper chord `(s, c)` depends only on `(l, u)`, which this module
//!   never moves — it is NOT touched (the build's `repair_upper_lines`
//!   certificate stands verbatim).
//! * THE ONE LOAD-BEARING COUPLING: the backward error lane contracts by
//!   `ms = max(alpha, s)` — the Lipschitz constant of `v -> vp*a + vn*s`.
//!   A stale binary-era `ms` under a raised alpha would UNDER-scale the carried
//!   error (a false-UNSAT generator), so `ms` is re-derived from the candidate
//!   alpha on every evaluation and on commit, by the same `max(alpha, s)`
//!   formula the root uses.
//! * The forward tableau is NOT rebuilt: the stored `l`/`u`/`clip_rows` were
//!   produced by a valid relaxation and remain valid enclosures of the true
//!   pre-activation ranges regardless of which (valid) backward alpha is used
//!   later. In particular `apply_gates`' outward widening — whose HISTORICAL
//!   constant's derivation assumes a binary lower product — never sees a
//!   fractional alpha through this module, because backward-only optimization
//!   never executes it. (#rebake DOES bake fractional alpha forward; for that
//!   path `apply_gates` carries its own magnitude-scaled widening branch with
//!   the re-derivation in its header doc.)
//! * Search iterates evaluate through `DomainGates` overrides (the engine's
//!   established injection point); the root struct is mutated ONCE, on accept.
//!   With the gate off, on refusal before a certified candidate, or on
//!   no-improvement, the root is byte-identical to the build. A later pass
//!   error stops new candidates but may still publish the best earlier,
//!   strictly improving certified candidate.
//!
//! GATE. The scored [`super::RootAlphaRuntimePolicy`] is frozen once from the
//! typed preset/LeverSet; its exact legacy `NY_MARGIN_ROW_ALPHA_OPT=1` overlay
//! also arms it, and the declaration default is OFF. The frozen iteration
//! control (default 20) caps iterations; the frozen wall request (default 20,
//! further capped to 40% of the remaining deadline) caps wall clock. Engagement telemetry
//! (`[alpha-opt] iter=...`) prints unconditionally when armed, so an inert
//! arming is detectable in one log (a null from a lever that never fired is
//! vacuous).
//!
//! BANKED CAVEAT (why this is pass-frugal): on the INTERNAL verifier a working
//! alpha gradient measured 0 conversions because cost-per-iteration was
//! binding. Here one iteration costs ONE backward pass (~600 ms on cifar100)
//! plus a single-column forward walk (milliseconds); the y-pack and head frame
//! are built once and frozen.

use ndarray::Array2;
use std::collections::BTreeMap;
use std::time::Instant;

use super::bounds::{
    compose_viay, head_gates, margin_seed, per_class_direct, row_dots, MarginBatch, YBox,
};
use super::engine::{BackwardEngine, Collect, DomainGates, GateVecs, LaneDir, PassOut};
use super::net::{conv_apply_forward, TwinNet, TwinOp};
use super::root::RootGates;

/// Fully resolved root-alpha optimizer controls. `Default` deliberately means
/// "resolve the compatibility layered process" for direct tests and research
/// callers. Scored execution always uses [`AlphaOptCfg::from_policy`] so the
/// optimizer cannot reopen a process-global authority after the flight receipt
/// was frozen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AlphaOptCfg {
    pub(crate) iters: usize,
    pub(crate) secs: f64,
    /// Adam step. Zero selects the historical normalized-supergradient path.
    pub(crate) lr: f64,
    /// Number of worst frozen-frame columns averaged for steering.
    pub(crate) cols: usize,
    /// Revert unless the independently recomputed published root improves.
    pub(crate) guard: bool,
}

impl Default for AlphaOptCfg {
    fn default() -> Self {
        Self::from_policy(super::RootAlphaRuntimePolicy::from_layered_process())
    }
}

impl AlphaOptCfg {
    pub(super) const fn from_policy(policy: super::RootAlphaRuntimePolicy) -> Self {
        Self {
            iters: policy.alpha_iters,
            secs: policy.alpha_secs,
            lr: policy.alpha_lr,
            cols: policy.alpha_cols,
            guard: policy.alpha_guard,
        }
    }

    /// Controls shared by the joint re-bake ascent. Re-bake owns its round
    /// iteration/wall policy and performs the published-frame guard after the
    /// rebuild, so only the two steering controls are resolved here. Keeping
    /// this constructor explicit prevents discarded reads of the standalone
    /// optimizer's iters/secs/guard axes from polluting attribution receipts.
    pub(super) const fn for_rebake(
        policy: super::RootAlphaRuntimePolicy,
        iters: usize,
        secs: f64,
    ) -> Self {
        Self {
            iters,
            secs,
            lr: policy.alpha_lr,
            cols: policy.alpha_cols,
            guard: false,
        }
    }
}

/// Cheap, side-effect-free admission fact shared by the standalone and rebake
/// execution receipts.  It deliberately excludes budget/data-path failures:
/// those require a completed report before T1 can award armed engagement.
pub(crate) fn execution_eligible(root: &RootGates, adv: &[usize]) -> bool {
    root.mode.outward() && !adv.is_empty() && root.layers.iter().any(|layer| !layer.unst.is_empty())
}

/// What one optimization run did (telemetry + test surface).
pub struct AlphaOptReport {
    pub(crate) cfg: AlphaOptCfg,
    /// Effective wall slice after the 40%-of-remaining cap.
    pub effective_secs: f64,
    /// Whether the instance deadline, rather than the configured request,
    /// reduced the effective wall slice.
    pub deadline_capped: bool,
    /// Whether the next certified pass was refused by the effective wall cap.
    pub budget_exhausted: bool,
    /// Whether all configured iterations completed their candidate pass.
    pub iters_cap_exhausted: bool,
    /// Largest number of frozen-frame columns actually used by a gradient.
    pub max_steer_cols: usize,
    /// Published-frame baseline evaluations owned by the guard.
    pub guard_baseline_evals: usize,
    /// Published-frame candidate evaluations owned by the guard.
    pub guard_candidate_evals: usize,
    /// Guard-triggered rollbacks.
    pub guard_reversions: usize,
    /// Frozen-frame failing objective columns available to steering.
    pub fail_classes: usize,
    /// Candidate updates applied before a certified evaluation.
    pub candidate_steps: usize,
    /// Candidate updates that used the legacy normalized step.
    pub legacy_steps: usize,
    /// Candidate updates that used Adam.
    pub adam_steps: usize,
    /// Candidate evaluations that strictly improved the frozen objective.
    pub frozen_improvements: usize,
    /// Candidate commits for which the guard had a publish/revert decision.
    pub guard_opportunities: usize,
    /// Published baseline already computed by the guard, when present.
    pub published_before: Option<f64>,
    /// Published candidate already computed by the guard, when finite.
    pub published_after: Option<f64>,
    /// Certified frozen-frame objective of the heuristic (binary) alpha.
    pub baseline: f64,
    /// Best certified frozen-frame objective found (>= baseline by monotone
    /// accept; == baseline when nothing improved).
    pub best: f64,
    /// Gradient iterations attempted (excludes the baseline pass).
    pub iters: usize,
    /// Certified backward passes spent (baseline + iterations).
    pub passes: usize,
    /// Was a non-heuristic alpha committed to the root?
    pub accepted: bool,
    /// Neurons whose committed alpha differs from the heuristic.
    pub moved_neurons: usize,
    /// Largest committed |delta alpha|.
    pub max_dalpha: f64,
}

/// Closed root-phase execution evidence for the jointly governed standalone
/// alpha and rebake controls.  Configuration is copied from the immutable
/// receipt-owned policy; observations are populated only from reports returned
/// by the actual dispatches.  Tree code can carry this value but never reads it
/// to make a bound or scheduling decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootAlphaExecutionEvidence {
    pub alpha_enabled: bool,
    pub alpha_iters: usize,
    pub alpha_secs: f64,
    pub alpha_lr: f64,
    pub alpha_cols: usize,
    pub alpha_guard: bool,
    pub rebake_enabled: bool,
    pub rebake_rounds: usize,
    pub rebake_secs: f64,
    pub eligible: bool,
    pub standalone_suppressed: usize,
    pub standalone_attempted: usize,
    pub standalone_completed: usize,
    pub standalone_effective_secs: f64,
    pub standalone_deadline_capped: bool,
    pub standalone_budget_exhausted: bool,
    pub standalone_iters_cap_exhausted: bool,
    pub standalone_fail_classes: usize,
    pub standalone_iters_attempted: usize,
    pub standalone_candidate_steps: usize,
    pub standalone_legacy_steps: usize,
    pub standalone_adam_steps: usize,
    pub standalone_passes: usize,
    pub standalone_max_steer_cols: usize,
    pub standalone_frozen_improvements: usize,
    pub standalone_accepted: bool,
    pub standalone_moved_neurons: usize,
    pub standalone_guard_opportunities: usize,
    pub standalone_guard_baseline_evals: usize,
    pub standalone_guard_candidate_evals: usize,
    pub standalone_guard_reversions: usize,
    pub standalone_frozen_before_bits: Option<u64>,
    pub standalone_frozen_best_bits: Option<u64>,
    pub standalone_published_before_bits: Option<u64>,
    pub standalone_published_after_bits: Option<u64>,
    pub rebake_suppressed: usize,
    pub rebake_attempted: usize,
    pub rebake_completed: usize,
    pub rebake_effective_secs: f64,
    pub rebake_deadline_capped: bool,
    pub rebake_budget_exhausted: bool,
    pub rebake_round_cap_exhausted: bool,
    pub rebake_entry_unbaked_alpha: bool,
    pub rebake_baseline_evals: usize,
    pub rebake_rounds_started: usize,
    pub rebake_alpha_completed: usize,
    pub rebake_alpha_iters_attempted: usize,
    pub rebake_alpha_candidate_steps: usize,
    pub rebake_alpha_legacy_steps: usize,
    pub rebake_alpha_adam_steps: usize,
    pub rebake_alpha_passes: usize,
    pub rebake_alpha_max_fail_classes: usize,
    pub rebake_alpha_max_steer_cols: usize,
    pub rebake_alpha_moved_neurons: usize,
    pub rebake_rebuilds_completed: usize,
    pub rebake_rounds_scored: usize,
    pub rebake_rounds_accepted: usize,
    pub rebake_fixed_point_stops: usize,
    pub rebake_affordability_refusals: usize,
    pub rebake_aborts: usize,
    pub rebake_published_before_bits: Option<u64>,
    pub rebake_published_after_bits: Option<u64>,
    pub policy_mismatch: bool,
}

impl RootAlphaExecutionEvidence {
    pub(crate) fn completed(
        policy: super::RootAlphaRuntimePolicy,
        eligible: bool,
        standalone: Option<&AlphaOptReport>,
        rebake: Option<&super::rebake::RebakeReport>,
    ) -> Self {
        let standalone_matches = standalone.is_none_or(|report| {
            policy.alpha_enabled && eligible && report.cfg == AlphaOptCfg::from_policy(policy)
        });
        let rebake_matches = rebake
            .is_none_or(|report| policy.rebake_enabled && eligible && report.policy == policy);
        Self {
            alpha_enabled: policy.alpha_enabled,
            alpha_iters: policy.alpha_iters,
            alpha_secs: policy.alpha_secs,
            alpha_lr: policy.alpha_lr,
            alpha_cols: policy.alpha_cols,
            alpha_guard: policy.alpha_guard,
            rebake_enabled: policy.rebake_enabled,
            rebake_rounds: policy.rebake_rounds,
            rebake_secs: policy.rebake_secs,
            eligible,
            standalone_suppressed: usize::from(eligible && !policy.alpha_enabled),
            standalone_attempted: usize::from(policy.alpha_enabled),
            standalone_completed: usize::from(standalone.is_some()),
            standalone_effective_secs: standalone.map_or(0.0, |report| report.effective_secs),
            standalone_deadline_capped: standalone.is_some_and(|report| report.deadline_capped),
            standalone_budget_exhausted: standalone.is_some_and(|report| report.budget_exhausted),
            standalone_iters_cap_exhausted: standalone
                .is_some_and(|report| report.iters_cap_exhausted),
            standalone_fail_classes: standalone.map_or(0, |report| report.fail_classes),
            standalone_iters_attempted: standalone.map_or(0, |report| report.iters),
            standalone_candidate_steps: standalone.map_or(0, |report| report.candidate_steps),
            standalone_legacy_steps: standalone.map_or(0, |report| report.legacy_steps),
            standalone_adam_steps: standalone.map_or(0, |report| report.adam_steps),
            standalone_passes: standalone.map_or(0, |report| report.passes),
            standalone_max_steer_cols: standalone.map_or(0, |report| report.max_steer_cols),
            standalone_frozen_improvements: standalone
                .map_or(0, |report| report.frozen_improvements),
            standalone_accepted: standalone.is_some_and(|report| report.accepted),
            standalone_moved_neurons: standalone.map_or(0, |report| report.moved_neurons),
            standalone_guard_opportunities: standalone
                .map_or(0, |report| report.guard_opportunities),
            standalone_guard_baseline_evals: standalone
                .map_or(0, |report| report.guard_baseline_evals),
            standalone_guard_candidate_evals: standalone
                .map_or(0, |report| report.guard_candidate_evals),
            standalone_guard_reversions: standalone.map_or(0, |report| report.guard_reversions),
            standalone_frozen_before_bits: standalone.map(|report| report.baseline.to_bits()),
            standalone_frozen_best_bits: standalone.map(|report| report.best.to_bits()),
            standalone_published_before_bits: standalone
                .and_then(|report| report.published_before.map(f64::to_bits)),
            standalone_published_after_bits: standalone
                .and_then(|report| report.published_after.map(f64::to_bits)),
            rebake_suppressed: usize::from(eligible && !policy.rebake_enabled),
            rebake_attempted: usize::from(policy.rebake_enabled),
            rebake_completed: usize::from(rebake.is_some()),
            rebake_effective_secs: rebake.map_or(0.0, |report| report.effective_secs),
            rebake_deadline_capped: rebake.is_some_and(|report| report.deadline_capped),
            rebake_budget_exhausted: rebake.is_some_and(|report| report.budget_exhausted),
            rebake_round_cap_exhausted: rebake.is_some_and(|report| report.round_cap_exhausted),
            rebake_entry_unbaked_alpha: rebake.is_some_and(|report| report.entry_unbaked_alpha),
            rebake_baseline_evals: usize::from(rebake.is_some()),
            rebake_rounds_started: rebake.map_or(0, |report| report.rounds_started),
            rebake_alpha_completed: rebake.map_or(0, |report| report.alpha_completed),
            rebake_alpha_iters_attempted: rebake.map_or(0, |report| report.alpha_iters_attempted),
            rebake_alpha_candidate_steps: rebake.map_or(0, |report| report.alpha_candidate_steps),
            rebake_alpha_legacy_steps: rebake.map_or(0, |report| report.alpha_legacy_steps),
            rebake_alpha_adam_steps: rebake.map_or(0, |report| report.alpha_adam_steps),
            rebake_alpha_passes: rebake.map_or(0, |report| report.alpha_passes),
            rebake_alpha_max_fail_classes: rebake.map_or(0, |report| report.alpha_max_fail_classes),
            rebake_alpha_max_steer_cols: rebake.map_or(0, |report| report.alpha_max_steer_cols),
            rebake_alpha_moved_neurons: rebake.map_or(0, |report| report.alpha_moved_neurons),
            rebake_rebuilds_completed: rebake.map_or(0, |report| report.rebuilds_completed),
            rebake_rounds_scored: rebake.map_or(0, |report| report.rounds_run),
            rebake_rounds_accepted: rebake.map_or(0, |report| report.rounds_accepted),
            rebake_fixed_point_stops: rebake.map_or(0, |report| report.fixed_point_stops),
            rebake_affordability_refusals: rebake.map_or(0, |report| report.affordability_refusals),
            rebake_aborts: rebake.map_or(0, |report| report.aborts),
            rebake_published_before_bits: rebake.map(|report| report.baseline.to_bits()),
            rebake_published_after_bits: rebake.map(|report| report.best.to_bits()),
            policy_mismatch: !(standalone_matches && rebake_matches),
        }
    }
}

impl Default for RootAlphaExecutionEvidence {
    fn default() -> Self {
        Self::completed(super::RootAlphaRuntimePolicy::default(), false, None, None)
    }
}

/// Frozen-policy entry: no-op unless root alpha is armed.
///
/// Called from `lane_impl` between the root build and the BaB driver, where
/// the root is still owned mutably. Bound-quality only: an internal error
/// stops further candidates; only the best earlier, strictly improving
/// certified candidate can still be committed.
pub fn maybe_optimize_root_alpha(
    net: &TwinNet,
    root: &mut RootGates,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    policy: super::RootAlphaRuntimePolicy,
) -> Option<AlphaOptReport> {
    if !policy.alpha_enabled {
        return None;
    }
    optimize_root_alpha(
        net,
        root,
        t,
        adv,
        deadline,
        AlphaOptCfg::from_policy(policy),
    )
}

/// The optimizer proper (test-visible; the policy gate lives in
/// [`maybe_optimize_root_alpha`]).
pub(crate) fn optimize_root_alpha(
    net: &TwinNet,
    root: &mut RootGates,
    t: usize,
    adv: &[usize],
    deadline: Option<Instant>,
    cfg: AlphaOptCfg,
) -> Option<AlphaOptReport> {
    let t0 = Instant::now();
    if !root.mode.outward() {
        // Parity is the bit-parity oracle against core.py; it must not move.
        eprintln!("[alpha-opt] refuse: parity mode");
        return None;
    }
    if adv.is_empty() || root.layers.iter().all(|l| l.unst.is_empty()) {
        eprintln!(
            "[alpha-opt] refuse: nothing to optimize (no adv classes or no unstable neurons)"
        );
        return None;
    }
    // Never starve the tree search: cap at 40% of the remaining deadline.
    let budget = match deadline {
        Some(dl) => cfg
            .secs
            .min(0.4 * dl.saturating_duration_since(t0).as_secs_f64()),
        None => cfg.secs,
    };
    let deadline_capped = budget < cfg.secs;
    if budget < 0.5 {
        eprintln!("[alpha-opt] refuse: no budget (deadline too close)");
        return None;
    }
    let published_before = if cfg.guard {
        match super::rebake::certified_root_bound(net, root, t, adv) {
            Ok(bound) if bound.is_finite() => Some(bound),
            Ok(_) => {
                eprintln!("[alpha-opt] refuse: non-finite published baseline");
                return None;
            }
            Err(error) => {
                eprintln!("[alpha-opt] refuse: published baseline root_eval: {error}");
                return None;
            }
        }
    } else {
        None
    };
    let sr = search_alpha(net, &*root, t, adv, t0, budget, cfg)?;
    // COMMIT: strictly-better certified objective only (monotone accept). The
    // heuristic alpha is otherwise retained bit-identically.
    let mut accepted = sr.best > sr.baseline;
    let mut moved_neurons = 0usize;
    let mut max_dalpha = 0.0f64;
    // A rollback copy is needed only when the published-frame guard may reject
    // an otherwise accepted frozen-frame iterate. Avoid cloning every layer on
    // the common no-improvement path (and when the guard is explicitly off).
    let original = (accepted && published_before.is_some()).then(|| {
        root.layers
            .iter()
            .map(|layer| (layer.alpha.clone(), layer.ms.clone()))
            .collect::<Vec<_>>()
    });
    let guard_opportunities = usize::from(accepted);
    let guard_candidate_evals = usize::from(accepted && published_before.is_some());
    let mut guard_reversions = 0usize;
    let mut published_after = None;
    if accepted {
        for (li, layer) in root.layers.iter_mut().enumerate() {
            let cand = &sr.best_alpha[li];
            // Touch ONLY unstable neurons: stable and split-baked neurons keep
            // their exact fixed lines bit-identically.
            let unst = layer.unst.clone();
            for j in unst {
                let a_new = cand[j].clamp(0.0, 1.0);
                let a_old = layer.alpha[j];
                if a_new != a_old {
                    moved_neurons += 1;
                    max_dalpha = max_dalpha.max((a_new - a_old).abs());
                    layer.alpha[j] = a_new;
                }
                // Re-derive the alpha-coupled certified quantities via the same
                // path the root build uses:
                //  * `s`/`c` are functions of `(l, u)` ONLY (`gates_from_box` +
                //    `repair_upper_lines`), and `(l, u)` are frozen here, so
                //    their build-time certificate stands verbatim — nothing to
                //    recompute.
                //  * `ms = max(alpha, s)` (root.rs) IS alpha-coupled: it is the
                //    error-contraction Lipschitz constant of the backward gate
                //    transform, and a stale value under a raised alpha would
                //    under-scale the carried error (false-UNSAT risk). Recompute.
                layer.ms[j] = layer.alpha[j].max(layer.s[j]);
            }
        }
    }
    if accepted {
        if let Some(before) = published_before {
            let after = super::rebake::certified_root_bound(net, root, t, adv);
            let keep = after
                .as_ref()
                .is_ok_and(|value| value.is_finite() && *value > before);
            published_after = after
                .as_ref()
                .ok()
                .copied()
                .filter(|value| value.is_finite());
            match after {
                Ok(value) => eprintln!(
                    "[alpha-opt] published_guard before={before:.6} after={value:.6} accepted={keep}"
                ),
                Err(error) => eprintln!(
                    "[alpha-opt] published_guard before={before:.6} after=error accepted=false: {error}"
                ),
            }
            if !keep {
                let original = original.expect("accepted guarded iterate retained rollback state");
                for (layer, (alpha, ms)) in root.layers.iter_mut().zip(original) {
                    layer.alpha = alpha;
                    layer.ms = ms;
                }
                accepted = false;
                moved_neurons = 0;
                max_dalpha = 0.0;
                guard_reversions = 1;
            }
        }
    }
    eprintln!(
        "[alpha-opt] done iters={} passes={} root_bound_before={:.6} after={:.6} accepted={} \
moved_neurons={} max_dalpha={:.4} lr={:.4} cols={} guard={} secs={:.2}",
        sr.iters,
        sr.passes,
        sr.baseline,
        sr.best,
        accepted,
        moved_neurons,
        max_dalpha,
        cfg.lr,
        cfg.cols,
        cfg.guard,
        t0.elapsed().as_secs_f64()
    );
    Some(AlphaOptReport {
        cfg,
        effective_secs: budget,
        deadline_capped,
        budget_exhausted: sr.budget_exhausted,
        iters_cap_exhausted: sr.candidate_steps == cfg.iters,
        max_steer_cols: sr.max_steer_cols,
        guard_baseline_evals: usize::from(published_before.is_some()),
        guard_candidate_evals,
        guard_reversions,
        fail_classes: sr.fail_classes,
        candidate_steps: sr.candidate_steps,
        legacy_steps: if cfg.lr == 0.0 { sr.candidate_steps } else { 0 },
        adam_steps: if cfg.lr == 0.0 { 0 } else { sr.candidate_steps },
        frozen_improvements: sr.frozen_improvements,
        guard_opportunities,
        published_before,
        published_after,
        baseline: sr.baseline,
        best: sr.best,
        iters: sr.iters,
        passes: sr.passes,
        accepted,
        moved_neurons,
        max_dalpha,
    })
}

struct SearchRes {
    best_alpha: Vec<Vec<f64>>,
    baseline: f64,
    best: f64,
    iters: usize,
    passes: usize,
    candidate_steps: usize,
    budget_exhausted: bool,
    fail_classes: usize,
    max_steer_cols: usize,
    frozen_improvements: usize,
}

/// The search: immutable root, candidates injected as `DomainGates`.
fn search_alpha(
    net: &TwinNet,
    root: &RootGates,
    t: usize,
    adv: &[usize],
    t0: Instant,
    budget: f64,
    cfg: AlphaOptCfg,
) -> Option<SearchRes> {
    let eng = BackwardEngine::new(net, root);
    let mode = root.mode;
    // ---- Frozen evaluation frame (built ONCE under the heuristic gates) ----
    // y-pack -> y-box -> head gates -> margin seeds -> m1 / m2v. All stay
    // fixed while trunk alpha moves; every quantity is a valid certified bound
    // for ANY valid trunk relaxation, so freezing costs tightness of the
    // OBJECTIVE only, never soundness. The published root bound is recomputed
    // from scratch by `root_eval` under the committed gates.
    let mb_all = match MarginBatch::new(net, t, adv) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[alpha-opt] abort: margin batch: {e}");
            return None;
        }
    };
    let (al, au) = match eng.y_rows(None) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[alpha-opt] abort: y-pack: {e}");
            return None;
        }
    };
    let ybox = YBox::from_rows(&eng, &al, &au);
    let gates = head_gates(&ybox, mode);
    let ms_all = margin_seed(&mb_all, &gates, &ybox, mode);
    let al_dots = row_dots(root, &al);
    let au_dots = row_dots(root, &au);
    let m2v_all = compose_viay(&eng, &mb_all, &gates, &al, &au, &al_dots, &au_dots, mode);
    // Mirror root_eval's admission: only classes the frozen head paths do NOT
    // already close get a direct-pass column (same width as the root's own
    // direct pass; outward closure is strict `> 0`).
    let fail: Vec<usize> = (0..adv.len())
        .filter(|&k| {
            let d = ms_all.m1[k].max(m2v_all[k]);
            !(d.is_finite() && d > 0.0)
        })
        .collect();
    if fail.is_empty() {
        eprintln!("[alpha-opt] refuse: all classes close under the frozen head frame");
        return None;
    }
    let fail_classes: Vec<usize> = fail.iter().map(|&k| adv[k]).collect();
    let mbf = match MarginBatch::new(net, t, &fail_classes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[alpha-opt] abort: fail-set margin batch: {e}");
            return None;
        }
    };
    let ms_f = margin_seed(&mbf, &gates, &ybox, mode);
    let m2v_f: Vec<f64> = fail.iter().map(|&k| m2v_all[k]).collect();
    let nf = mbf.nf();

    // One evaluation = one certified Lower pass under the candidate alpha
    // (with the pre-transform coefficient capture the gradient needs) plus the
    // frozen-frame objective: min over classes of max(direct, m1, m2v).
    let eval = |alpha: &[Vec<f64>]| -> Result<(f64, Vec<usize>, PassOut), String> {
        let dom = gates_override(root, alpha);
        let pass = eng
            .run_collect(
                &ms_f.seed,
                Some(&dom),
                LaneDir::Lower,
                None,
                Collect {
                    unst_abs: false,
                    rows: None,
                    unst_rows: true,
                },
            )
            .map_err(|e| e.to_string())?;
        let direct = per_class_direct(&eng, &pass, &ms_f, 0..nf);
        let mut scored = Vec::with_capacity(nf);
        for f in 0..nf {
            let d = direct[f].max(ms_f.m1[f]).max(m2v_f[f]);
            if !d.is_finite() {
                return Err("non-finite objective".into());
            }
            scored.push((d, f));
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let worst = scored[0].0;
        let steer_cols = scored
            .into_iter()
            .take(cfg.cols.min(nf).max(1))
            .map(|(_, col)| col)
            .collect();
        Ok((worst, steer_cols, pass))
    };

    let init: Vec<Vec<f64>> = root.layers.iter().map(|l| l.alpha.clone()).collect();
    let te = Instant::now();
    let (baseline, mut steer_cols, baseline_pass) = match eval(&init) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[alpha-opt] abort: baseline pass: {e}");
            return None;
        }
    };
    let mut pass_secs = te.elapsed().as_secs_f64().max(1e-3);
    eprintln!(
        "[alpha-opt] iter=0 root_bound_before={baseline:.6} after={baseline:.6} accepted=true \
(baseline; frozen-frame objective over {nf} classes)"
    );
    let mut best = baseline;
    let mut best_alpha = init;
    let mut grad = mean_supergradient(net, root, &best_alpha, &baseline_pass, &steer_cols);
    let mut eta = 0.25f64;
    let mut adam_m: Vec<Vec<f64>> = root.layers.iter().map(|l| vec![0.0; l.n]).collect();
    let mut adam_v = adam_m.clone();
    let mut passes = 1usize;
    let mut iters = 0usize;
    let mut candidate_steps = 0usize;
    let mut budget_exhausted = false;
    let mut max_steer_cols = steer_cols.len();
    let mut frozen_improvements = 0usize;
    for it in 1..=cfg.iters {
        // Leave room for the pass we are about to spend.
        if t0.elapsed().as_secs_f64() + 1.5 * pass_secs > budget {
            budget_exhausted = true;
            break;
        }
        let Some(g) = grad.as_ref() else { break };
        let gmax = g
            .iter()
            .flat_map(|v| v.iter())
            .fold(0.0f64, |m, &x| m.max(x.abs()));
        if !(gmax.is_finite() && gmax > 0.0) {
            break;
        }
        // Projected step FROM THE BEST iterate. `lr == 0` is the exact legacy
        // globally-normalized update; otherwise Adam makes the step scale-free
        // per coordinate, with alpha-beta-CROWN's 0.98 learning-rate decay.
        let mut cand = best_alpha.clone();
        let mut moved = false;
        for (li, layer) in root.layers.iter().enumerate() {
            for &j in &layer.unst {
                let gn = g[li][j] / gmax;
                let step = if cfg.lr == 0.0 {
                    eta * gn
                } else {
                    adam_m[li][j] = 0.9 * adam_m[li][j] + 0.1 * gn;
                    adam_v[li][j] = 0.999 * adam_v[li][j] + 0.001 * gn * gn;
                    let mhat = adam_m[li][j] / (1.0 - 0.9f64.powi(it as i32));
                    let vhat = adam_v[li][j] / (1.0 - 0.999f64.powi(it as i32));
                    let lr = cfg.lr * 0.98f64.powi((it - 1) as i32);
                    lr * mhat / (vhat.sqrt() + 1e-8)
                };
                let a2 = (cand[li][j] + step).clamp(0.0, 1.0);
                if a2 != cand[li][j] {
                    cand[li][j] = a2;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
        iters = it;
        let te = Instant::now();
        let (f, fcols, pass) = match eval(&cand) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[alpha-opt] abort: pass failed at iter {it}: {e}");
                break;
            }
        };
        pass_secs = pass_secs.max(te.elapsed().as_secs_f64());
        passes += 1;
        candidate_steps += 1;
        max_steer_cols = max_steer_cols.max(fcols.len());
        let improved = f > best;
        eprintln!(
            "[alpha-opt] iter={it} root_bound_before={best:.6} after={f:.6} accepted={improved} \
step={} cols={}",
            if cfg.lr == 0.0 {
                format!("legacy_eta:{eta:.3}")
            } else {
                format!("adam_lr:{:.4}", cfg.lr * 0.98f64.powi((it - 1) as i32))
            },
            steer_cols.len()
        );
        if improved {
            frozen_improvements += 1;
            best = f;
            best_alpha = cand;
            steer_cols = fcols;
            grad = mean_supergradient(net, root, &best_alpha, &pass, &steer_cols);
        } else if cfg.lr == 0.0 {
            eta *= 0.5;
            if eta < 0.01 {
                break;
            }
        }
    }
    Some(SearchRes {
        best_alpha,
        baseline,
        best,
        iters,
        passes,
        candidate_steps,
        budget_exhausted,
        fail_classes: nf,
        max_steer_cols,
        frozen_improvements,
    })
}

/// Average the frozen-sign supergradients of the K worst objective columns.
/// K=1 deliberately calls the historical primitive once and is therefore the
/// exact steering control arm.
fn mean_supergradient(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    pass: &PassOut,
    cols: &[usize],
) -> Option<Vec<Vec<f64>>> {
    if cols.is_empty() {
        return None;
    }
    let mut mean: Vec<Vec<f64>> = root.layers.iter().map(|l| vec![0.0; l.n]).collect();
    for &col in cols {
        let grad = supergradient(net, root, alpha, pass, col)?;
        for (dst_layer, src_layer) in mean.iter_mut().zip(grad) {
            for (dst, src) in dst_layer.iter_mut().zip(src_layer) {
                *dst += src;
            }
        }
    }
    let scale = 1.0 / cols.len() as f64;
    for layer in &mut mean {
        for value in layer {
            *value *= scale;
        }
    }
    Some(mean)
}

/// Full-layer gate override for a candidate alpha. `s`/`c` are the root's
/// certified chords (untouched); `ms` is re-derived from the CANDIDATE alpha —
/// the one alpha-coupled certified quantity (see the module doc).
fn gates_override(root: &RootGates, alpha: &[Vec<f64>]) -> DomainGates {
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
            GateVecs {
                alpha: alpha[li].clone(),
                s: layer.s.clone(),
                c: layer.c.clone(),
                ms,
            },
        );
    }
    dg
}

/// Supergradient of the objective column `col` w.r.t. every unstable trunk
/// alpha: `g[li][j] = vp_{li,j,col} * U_{li,j,col}` (see the module doc).
///
/// Heuristic-only: the direction never feeds a verdict — every accepted
/// iterate is re-scored by the unchanged certified pass.
///
/// `pub(crate)`: #margin-row-domain-alpha reuses this EXACT gradient per BaB
/// domain, passing the domain's effective alpha vectors — the same
/// direction-only contract `beta::step_scales` already relies on. Note the
/// gradient is NOT sign-definite: `vp >= 0` but `U` is the pre-activation of an
/// UNSTABLE neuron at `x*`, so it straddles zero. (The banked "ny's alpha
/// gradient is sign-definite <= 0" finding is about the INTERNAL verifier's
/// alpha-CROWN, a different lane; it does not apply here.)
pub(crate) fn supergradient(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    pass: &PassOut,
    col: usize,
) -> Option<Vec<Vec<f64>>> {
    let vsigns = pass.unst_rows.as_ref()?;
    // Concretize argmin of the Lower bound for this column:
    // x*_i = mid_i - sign(A_i)*rad_i (mid on a zero coefficient).
    let n_in = root.mid.len();
    let r = pass.a.ncols();
    if col >= r {
        return None;
    }
    let asl = pass.a.as_slice()?;
    let mut xstar = vec![0.0; n_in];
    for i in 0..n_in {
        let a = asl[i * r + col];
        xstar[i] = if a > 0.0 {
            root.mid[i] - root.rad[i]
        } else if a < 0.0 {
            root.mid[i] + root.rad[i]
        } else {
            root.mid[i]
        };
    }
    let (u, _y) = linearized_walk(net, root, alpha, vsigns, col, &xstar, false)?;
    let mut g: Vec<Vec<f64>> = root.layers.iter().map(|l| vec![0.0; l.n]).collect();
    for (li, layer) in root.layers.iter().enumerate() {
        let (Some(uvals), Some(vmat)) = (u.get(&li), vsigns.get(&li)) else {
            continue;
        };
        let vs = vmat.as_slice()?;
        let rr = vmat.ncols();
        if uvals.len() != layer.unst.len() || vmat.nrows() != layer.unst.len() || col >= rr {
            return None;
        }
        for (pos, &j) in layer.unst.iter().enumerate() {
            let vp = vs[pos * rr + col].max(0.0);
            g[li][j] = vp * uvals[pos];
        }
    }
    Some(g)
}

/// Forward walk of `x*` through the FROZEN-SIGN linearized trunk: every op is
/// applied exactly (f64, with bias); every relu is replaced per neuron by the
/// affine line the pass selected for column `col` (Lower lane: `alpha*z` on a
/// positive incoming coefficient, `s*z + c` on a negative one; exact fixed
/// lines — stable and split-baked neurons — are sign-independent).
///
/// Returns `(U, y)`: per trunk layer, the walk values at that layer's UNSTABLE
/// neurons captured BEFORE the layer's own transform — the sensitivities `U` —
/// and, when `head` is set, the head pre-activation vector `ŷ(x*)` (the first
/// head Gemm's output; empty when `head` is false).
///
/// `head = false` reproduces the original walk BIT-IDENTICALLY (break at the
/// last trunk relu, no head ops touched). `head = true` (#margin-row-beta C3)
/// additionally applies the LAST trunk relu's selected lines — the same code
/// path as interior relus — and continues through the remaining trunk-tail ops
/// to `net.i_gemm1`, applying the head Gemm there. A Gemm anywhere OTHER than
/// exactly `i_gemm1` still refuses (`None`); relus between the last trunk relu
/// and `i_gemm1` cannot exist (net.rs compile: every relu before gemm1 is a
/// trunk relu).
///
/// `pub(crate)`: #margin-row-beta reuses this walk for its per-split
/// supergradient (`beta::step_scales`) — same frozen-sign linearization, same
/// direction-only contract (nothing verdict-bearing reads the result).
pub(crate) fn linearized_walk(
    net: &TwinNet,
    root: &RootGates,
    alpha: &[Vec<f64>],
    vsigns: &BTreeMap<usize, Array2<f64>>,
    col: usize,
    xstar: &[f64],
    head: bool,
) -> Option<(BTreeMap<usize, Vec<f64>>, Vec<f64>)> {
    let last_relu = *net.trunk_relus.last()?;
    let stop = if head { net.i_gemm1 } else { last_relu };
    let mut vals: Vec<Option<Vec<f64>>> = vec![None; net.ops.len() + 1];
    vals[0] = Some(xstar.to_vec());
    let mut u_out: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (k, op) in net.ops.iter().enumerate().take(stop + 1) {
        let out: Vec<f64> = match op {
            TwinOp::Conv(c) => {
                let src = vals[c.input].as_ref()?;
                let n_out = net.tsize[k + 1];
                let src_m = Array2::from_shape_vec((src.len(), 1), src.clone()).ok()?;
                let mut dst = Array2::<f64>::zeros((n_out, 1));
                conv_apply_forward(c, &src_m, &mut dst, false);
                let p = c.oshape.1 * c.oshape.2;
                let mut v = dst.as_slice()?.to_vec();
                for (j, vj) in v.iter_mut().enumerate() {
                    *vj += c.bias[j / p];
                }
                v
            }
            TwinOp::Add { lhs, rhs } => {
                let a = vals[*lhs].as_ref()?;
                let b = vals[*rhs].as_ref()?;
                a.iter().zip(b).map(|(x, y)| x + y).collect()
            }
            TwinOp::Flatten { input } => vals[*input].as_ref()?.clone(),
            TwinOp::ChannelAffine {
                input,
                scale,
                shift,
                ..
            } => {
                let src = vals[*input].as_ref()?;
                src.iter()
                    .enumerate()
                    .map(|(j, &v)| scale[j] * v + shift[j])
                    .collect()
            }
            TwinOp::Relu { input, layer } => {
                let li = *layer;
                let src = vals[*input].as_ref()?;
                let rec = root.layers.get(li)?;
                if src.len() != rec.n {
                    return None;
                }
                // Sensitivities: pre-transform values at the unstable neurons.
                if !rec.unst.is_empty() {
                    u_out.insert(li, rec.unst.iter().map(|&j| src[j]).collect());
                }
                if k == last_relu && !head {
                    break;
                }
                // Default per-neuron map: the (candidate) lower line `a*z`.
                // For every exact-line neuron — stable active (1,1,0), stable
                // inactive (0,0,0), split-baked — this IS the selected map
                // (candidates only move unstable alphas). Unstable neurons are
                // overwritten below with the pass's per-column selection.
                let mut out: Vec<f64> = src
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| alpha[li][j] * v)
                    .collect();
                if let Some(vmat) = vsigns.get(&li) {
                    let vs = vmat.as_slice()?;
                    let rr = vmat.ncols();
                    if vmat.nrows() != rec.unst.len() || col >= rr {
                        return None;
                    }
                    for (pos, &j) in rec.unst.iter().enumerate() {
                        // Negative incoming coefficient -> the pass selected
                        // the upper chord for this (neuron, column). A zero
                        // coefficient contributes nothing to the bound either
                        // way; keep the lower line (matches `vp = 0`).
                        if vs[pos * rr + col] < 0.0 {
                            out[j] = rec.s[j] * src[j] + rec.c[j];
                        }
                    }
                }
                out
            }
            TwinOp::Gemm {
                input,
                weight,
                bias,
                shape,
            } => {
                // #margin-row-beta C3: the FIRST head Gemm maps the trunk
                // output to the pre-head-relu vector y — exactly the layer the
                // head supergradient needs. Allowed only at exactly `i_gemm1`
                // in head mode; any other Gemm refuses as before.
                if !(head && k == net.i_gemm1) {
                    return None; // structurally pre-gemm otherwise
                }
                let src = vals[*input].as_ref()?;
                let (no, ni) = *shape;
                if src.len() != ni {
                    return None;
                }
                (0..no)
                    .map(|j| {
                        let row = &weight[j * ni..(j + 1) * ni];
                        row.iter().zip(src).map(|(w, v)| w * v).sum::<f64>() + bias[j]
                    })
                    .collect()
            }
        };
        vals[k + 1] = Some(out);
    }
    let y = if head {
        vals[net.i_gemm1 + 1].take()?
    } else {
        Vec::new()
    };
    Some((u_out, y))
}

#[cfg(test)]
mod execution_evidence_tests {
    use super::{AlphaOptCfg, AlphaOptReport, RootAlphaExecutionEvidence};
    use crate::margin_row::{rebake::RebakeReport, RootAlphaRuntimePolicy};

    #[test]
    fn root_alpha_evidence_is_route_split_and_policy_bound() {
        let policy = RootAlphaRuntimePolicy {
            alpha_enabled: true,
            alpha_iters: 4,
            alpha_secs: 12.5,
            alpha_lr: 0.05,
            alpha_cols: 3,
            alpha_guard: true,
            rebake_enabled: true,
            rebake_rounds: 2,
            rebake_secs: 30.0,
        };
        let standalone = AlphaOptReport {
            cfg: AlphaOptCfg::from_policy(policy),
            effective_secs: 10.0,
            deadline_capped: true,
            budget_exhausted: false,
            iters_cap_exhausted: true,
            max_steer_cols: 3,
            guard_baseline_evals: 1,
            guard_candidate_evals: 1,
            guard_reversions: 0,
            fail_classes: 7,
            candidate_steps: 4,
            legacy_steps: 0,
            adam_steps: 4,
            frozen_improvements: 1,
            guard_opportunities: 1,
            published_before: Some(-1.0),
            published_after: Some(-0.5),
            baseline: -1.25,
            best: -0.75,
            iters: 4,
            passes: 5,
            accepted: true,
            moved_neurons: 6,
            max_dalpha: 0.25,
        };
        let rebake = RebakeReport {
            policy,
            effective_secs: 20.0,
            deadline_capped: true,
            budget_exhausted: true,
            round_cap_exhausted: false,
            entry_unbaked_alpha: true,
            rounds_started: 1,
            alpha_completed: 1,
            alpha_iters_attempted: 4,
            alpha_candidate_steps: 4,
            alpha_legacy_steps: 0,
            alpha_adam_steps: 4,
            alpha_passes: 5,
            alpha_max_fail_classes: 7,
            alpha_max_steer_cols: 3,
            alpha_moved_neurons: 6,
            rebuilds_completed: 1,
            fixed_point_stops: 0,
            affordability_refusals: 1,
            aborts: 0,
            baseline: -1.0,
            best: -0.25,
            rounds_run: 1,
            rounds_accepted: 1,
        };

        let evidence =
            RootAlphaExecutionEvidence::completed(policy, true, Some(&standalone), Some(&rebake));
        assert!(!evidence.policy_mismatch);
        assert_eq!(evidence.standalone_candidate_steps, 4);
        assert_eq!(
            evidence.standalone_frozen_before_bits,
            Some((-1.25f64).to_bits())
        );
        assert_eq!(
            evidence.standalone_published_after_bits,
            Some((-0.5f64).to_bits())
        );
        assert_eq!(evidence.rebake_alpha_candidate_steps, 4);
        assert_eq!(evidence.rebake_rounds_scored, 1);
        assert_eq!(
            evidence.rebake_published_after_bits,
            Some((-0.25f64).to_bits())
        );

        let suppressed = RootAlphaExecutionEvidence::completed(
            RootAlphaRuntimePolicy::default(),
            true,
            None,
            None,
        );
        assert_eq!(suppressed.standalone_suppressed, 1);
        assert_eq!(suppressed.rebake_suppressed, 1);

        let disabled_report = RootAlphaExecutionEvidence::completed(
            RootAlphaRuntimePolicy {
                alpha_enabled: false,
                ..policy
            },
            true,
            Some(&standalone),
            None,
        );
        assert!(disabled_report.policy_mismatch);
    }
}
