// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Adaptive attack-phase extension (#attack-extend): grant ONE bounded retry of
//! the disjunctive attack when it ended WITHOUT a counterexample but with a
//! PROMISING closest-to-violation margin.
//!
//! Rationale (metaroom_2023 measurement, 2026-07): presets that protect
//! near-wall UNSAT proofs cap the attack slice (metaroom
//! `disjunctive_pgd_fraction: 0.10` ⇒ ~20s of a 200s internal budget). On
//! GT=sat instances the graph disjunctive PGD is regularly cut mid-ascent by
//! that cap with a best margin a hair below violation — measured spec_idx_129:
//! best margin -0.083 at the 20s deadline (>= 0 is a counterexample) — and the
//! remaining ~90% of the budget then burns in BaB, which can never prove a sat
//! instance: a certain timeout. The same instance flips to `sat` when the
//! attack keeps running.
//!
//! Two gates, both required, both measured:
//!
//! 1. **Budget-bound** (`hit_deadline`): the attack must have been CUT by its
//!    phase deadline, not have exhausted its configured restarts. A finished
//!    attack that found nothing gets nothing — near-wall UNSAT instances
//!    (spec_idx_28 class, unsat with ~16s to spare at 210s) exhaust their
//!    restart budget under the batched exact-VJP lane and hand off with their
//!    full BaB slice intact.
//! 2. **Promising margin**: the best observed margin must sit within
//!    `NY_ATTACK_EXTEND_MARGIN` of violation. NOTE (measured): margin alone
//!    CANNOT separate sat-reachable from near-wall unsat — spec_idx_28
//!    (GT=unsat) plateaus at -0.033 across 14 restarts (the unsat sup) while
//!    spec_idx_129 (GT=sat) sat at -0.083 mid-ascent when cut. The margin gate
//!    exists to refuse extensions on HOPELESS cuts (spec_idx_148 measured
//!    -5.01 at its 20s cut on the sequential lane), not to prove promise; the
//!    budget-bound gate is what keeps plateaued unsat instances out.
//!
//! One refusal on top of the two gates (#attack-stall): an attack ended by the
//! adaptive STALL cutoff (`super::attack_stall`) is never extended. That cutoff
//! fires on the one signal the margin gate above cannot read — the ascent
//! stopped IMPROVING — so extending it would hand back the budget it just
//! reclaimed. Its absence is not evidence of anything about the property; both
//! paths lead to the same unchanged bound/BaB phases.
//!
//! The extension is granted at most ONCE, is a bounded fraction of the
//! REMAINING budget, and CONTINUES the restart seed sequence where the first
//! run's cap cut it (the first run's trajectories are already explored;
//! replaying them would waste the grant, and the longer continuous run is the
//! measured reference that lands the counterexample).
//!
//! ATTACK-ONLY: the retry generates candidates; every candidate still passes
//! the independent re-evaluation and the trusted-ORT vnncomp gate before any
//! `sat` is emitted, and bounds/verdicts are untouched — a wrong decision here
//! can only misplace wall-clock budget, never produce a wrong answer.
//!
//! Kill switch: `NY_ATTACK_EXTEND=0` (batteries-included default ON).
//! Tunables: `NY_ATTACK_EXTEND_MARGIN` (promising distance-to-violation,
//! default 0.10), `NY_ATTACK_EXTEND_FRAC` (fraction of remaining budget,
//! default 0.15, clamped to [0, 0.5]).

use std::time::Duration;

use super::disjunctive_pgd::DisjunctiveAttackFeedback;

/// Below this much remaining budget the handoff to BaB is always immediate:
/// small-budget categories (lsnc_relu class) are BaB-starved already and the
/// small-budget attack cap (`PhaseBudgetLedger::attack_phase_deadline`)
/// intentionally keeps the attack slice minimal there.
const MIN_REMAINING_FOR_EXTENSION: Duration = Duration::from_mins(1);

/// Default promising-margin distance: extend when the best observed margin is
/// within this distance of violation (margin >= -0.10). Measured (metaroom
/// 2026-07): GT=sat spec_idx_129 was cut at the preset cap at -0.083 and flips
/// to `sat` when the attack continues. Margins are raw logit differences, so
/// this is model-scale dependent — the default is deliberately tight (extend
/// rarely; a missed extension is status quo, a spurious one costs BaB budget),
/// and the budget-bound gate above carries the real discriminative weight.
const DEFAULT_PROMISING_MARGIN: f32 = 0.10;

fn attack_extend_enabled() -> bool {
    std::env::var("NY_ATTACK_EXTEND").ok().as_deref() != Some("0")
}

fn promising_margin() -> f32 {
    std::env::var("NY_ATTACK_EXTEND_MARGIN")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(DEFAULT_PROMISING_MARGIN)
}

/// Extension slice as a fraction of the remaining budget. The preset-scoped
/// policy value (`PhaseBudgetConfig::attack_extension_fraction`, default 0.15)
/// is the base; `NY_ATTACK_EXTEND_FRAC` still overrides it at runtime.
/// Presets set `0.0` to disable the extension for categories where the margin
/// gate cannot discriminate (cgan_2023 band properties: every UNSAT instance
/// reports a "promising" margin, so the extension only starves bound phases).
fn extension_fraction(policy_fraction: f32) -> f32 {
    std::env::var("NY_ATTACK_EXTEND_FRAC")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(policy_fraction)
        .clamp(0.0, 0.5)
}

/// Decide the single attack-extension grant.
///
/// Returns the wall-clock slice for the retry, or `None` for the status-quo
/// immediate handoff to BaB. `remaining` is the ledger's remaining budget
/// (`None` = unbounded: the attack already got its full configured work, so
/// there is nothing to extend against). `policy_fraction` is the preset-scoped
/// `PhaseBudgetConfig::attack_extension_fraction`.
pub(super) fn attack_extension_slice(
    remaining: Option<Duration>,
    feedback: &DisjunctiveAttackFeedback,
    policy_fraction: f32,
) -> Option<Duration> {
    decide(
        remaining,
        feedback.best_margin,
        feedback.hit_deadline,
        feedback.stalled_out,
        attack_extend_enabled(),
        promising_margin(),
        extension_fraction(policy_fraction),
    )
}

#[allow(clippy::too_many_arguments)]
fn decide(
    remaining: Option<Duration>,
    best_margin: Option<f32>,
    hit_deadline: bool,
    stalled_out: bool,
    enabled: bool,
    promising_margin: f32,
    fraction: f32,
) -> Option<Duration> {
    if !enabled || fraction <= 0.0 {
        return None;
    }
    // #attack-stall: an attack cut for a PLATEAUED margin gets nothing. The
    // cutoff's whole claim is "this ascent stopped improving"; granting it more
    // time would undo the reclaim it just made, and the plateau is exactly the
    // near-wall UNSAT shape the budget-bound gate below already refuses.
    if stalled_out {
        return None;
    }
    // Work-bound attacks (configured restarts exhausted) get nothing: more
    // time was not what they lacked, and near-wall UNSAT plateaus land here.
    if !hit_deadline {
        return None;
    }
    let remaining = remaining?;
    if remaining < MIN_REMAINING_FOR_EXTENSION {
        return None;
    }
    // No margin telemetry (legacy sampling/SPSA lanes) ⇒ no signal ⇒ status quo.
    let best_margin = best_margin?;
    if !best_margin.is_finite() || best_margin < -promising_margin {
        return None;
    }
    Some(remaining.mul_f32(fraction))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: u64) -> Option<Duration> {
        Some(Duration::from_secs(s))
    }

    #[test]
    fn promising_budget_bound_cut_grants_fraction_of_remaining() {
        // metaroom spec_idx_129 shape: cut at the phase deadline mid-ascent
        // with margin -0.083.
        let slice =
            decide(secs(180), Some(-0.08), true, false, true, 0.10, 0.15).expect("promising");
        assert_eq!(slice, Duration::from_mins(3).mul_f32(0.15));
    }

    #[test]
    fn margin_at_violation_boundary_grants() {
        // Internal candidate found but rejected by confirmation (margin >= 0):
        // the continuation retry is exactly the right response.
        assert!(decide(secs(180), Some(0.02), true, false, true, 0.10, 0.15).is_some());
    }

    #[test]
    fn work_bound_attack_hands_off_even_with_promising_margin() {
        // metaroom spec_idx_28 shape (GT=unsat near-wall): the batched lane
        // exhausts the configured restarts with the unsat sup plateau at
        // -0.033 — more attack time is NOT what it lacked; BaB keeps its slice.
        assert!(decide(secs(180), Some(-0.03), false, false, true, 0.10, 0.15).is_none());
    }

    #[test]
    fn stall_cut_attack_is_never_extended() {
        // #attack-stall: the cutoff fires precisely BECAUSE the ascent stopped
        // improving. Extending it would hand the reclaimed budget straight back
        // to the lane that just proved it had nothing to do with it — even
        // though the margin looks "promising" by the level gate, which cannot
        // tell a plateaued unsat sup from a mid-ascent sat (spec_idx_28 at
        // -0.033 vs spec_idx_129 at -0.083).
        assert!(decide(secs(180), Some(-0.03), true, true, true, 0.10, 0.15).is_none());
        assert!(decide(secs(180), Some(0.02), true, true, true, 0.10, 0.15).is_none());
    }

    #[test]
    fn hopeless_margin_hands_off_immediately() {
        // metaroom spec_idx_148 sequential-lane shape: cut at -5.01.
        assert!(decide(secs(180), Some(-5.01), true, false, true, 0.10, 0.15).is_none());
    }

    #[test]
    fn margin_just_below_threshold_hands_off() {
        assert!(decide(secs(180), Some(-0.101), true, false, true, 0.10, 0.15).is_none());
    }

    #[test]
    fn no_margin_telemetry_hands_off() {
        assert!(decide(secs(180), None, true, false, true, 0.10, 0.15).is_none());
    }

    #[test]
    fn non_finite_margin_hands_off() {
        assert!(decide(
            secs(180),
            Some(f32::NEG_INFINITY),
            true,
            false,
            true,
            0.10,
            0.15
        )
        .is_none());
        assert!(decide(secs(180), Some(f32::NAN), true, false, true, 0.10, 0.15).is_none());
    }

    #[test]
    fn kill_switch_disables() {
        assert!(decide(secs(180), Some(-0.01), true, false, false, 0.10, 0.15).is_none());
    }

    #[test]
    fn small_remaining_budget_hands_off() {
        // Below the floor the remaining time is BaB-critical (lsnc_relu class).
        assert!(decide(secs(45), Some(-0.01), true, false, true, 0.10, 0.15).is_none());
    }

    #[test]
    fn unbounded_budget_hands_off() {
        assert!(decide(None, Some(-0.01), true, false, true, 0.10, 0.15).is_none());
    }

    #[test]
    fn zero_fraction_hands_off() {
        assert!(decide(secs(180), Some(-0.01), true, false, true, 0.10, 0.0).is_none());
    }
}
