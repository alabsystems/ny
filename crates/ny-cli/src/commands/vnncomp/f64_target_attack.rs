// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #sbench-f64-attack: the post-BaB float64 target-margin repair, and the
//! typed caps of the trusted-ORT active-set repair it follows.
//!
//! # Why
//!
//! SoundnessBench plants counterexamples in regions thinner than f32 output
//! resolution. MEASURED (pin c73c289c2, `scratchpad/refine2/sbs_*.raw`): the
//! field's accepted witnesses on the seven open `model_residual` rows have ONNX
//! Runtime margins of 1e-7..1e-5, while one f32 ULP at the constrained outputs'
//! magnitude (|Y| up to ~8) is ~1e-6. ny's post-BaB repair works on the ORT
//! (f32) surface, so it ranks points by rounding noise: on `property_054` it
//! reached an ORT-confirmed point at +6.8e-8 that the true-f64 gate correctly
//! refused (-1.3e-6), and ny's f64 polish — finite differences over the sound
//! f64 cell walk, 129 walks per step on this 25M-MAC net — never completed a
//! step inside its slice.
//!
//! # What it does
//!
//! [`f64_target_margin_repair`] is the active-set Newton lift of
//! `ort_active_set_repair_falsify`, moved onto an EXACT surface:
//!
//! * margins and their gradient come from [`PointVjpF64Plan`], the exact-f64
//!   twin of the batched point-VJP plan (side branches re-captured through the
//!   f64 cell), evaluated at the witness's own input view — so the residual is
//!   the real-arithmetic margin to ~1e-13 and the Jacobian rows are exact, one
//!   VJP per active conjunct;
//! * every conjunct is lifted to a target margin `delta = ulps * ulp_f32(max
//!   |Y_j|)` (`attack.f64_target_margin_ulps`), so a point that clears it in
//!   f64 survives the f32 forward's rounding;
//! * each coordinate of a step is rounded to f32 TOWARD the side that raises
//!   the lagging margins, so rounding onto the witness lattice helps instead of
//!   costing a random ~1e-6;
//! * the trusted forward is consulted only at points whose exact margin is
//!   already non-negative; if f64 clears `delta` there but the trusted forward
//!   does not, `delta` doubles (at most [`TARGET_ESCALATIONS`] times).
//!
//! # Soundness
//!
//! Attack-only. A candidate leaves [`f64_target_margin_repair`] only when the
//! trusted forward's output satisfies the unchanged zero-tolerance
//! `property_violated_f64` at the witness input view AND its exact-f64 margin
//! is non-negative; the caller then applies the same local true-f64 precheck
//! as every other post-BaB lineage (`emit_found!`) and the unchanged
//! `gate_sat_with_trusted_oracle`. There is no verdict-bearing return, so no
//! `unsat` can come from here.
//!
//! `property_033` is a POSITIVE row, not a negative control: the field's `sat`
//! there was graded incorrect because abc's witness missed one atom by one f32
//! ULP, but two witnesses found off-box replay as violated on the organisers'
//! x86-64 ONNX Runtime 1.26.0 (+4.9e-6 and +3.5e-6,
//! `scratchpad/vbs/frontier/rest.md` §2). The pre-registered banking rule
//! (FRONTIER_PLAN §1.3) is that a 033 `sat` is bankable iff its own scored-run
//! witness replays as violated on the certified x86 rig at >= 10 ULPs; a
//! witness that fails the rig is a stop-and-report, not a bank.
//!
//! # Step rule (#sbench-f64-maximin)
//!
//! `attack.f64_lift_step: lp_maximin` replaces the min-norm lift below with the
//! LP-maximin step and `attack.f64_basin_hops` basin hops of
//! `f64_lp_maximin.rs` (same scope, same target, same trusted-forward contract
//! and the same exit). Absent / `min_norm` runs [`f64_target_margin_repair`]
//! exactly as before.
//!
//! # Dark by default
//!
//! All four keys are absent from every shipped preset. The resolved policy is
//! then `PostBabActiveSetPolicy::HISTORICAL`, which reproduces the active-set
//! constants bit-for-bit and runs no f64 stage; the pins live in this module's
//! tests.

use std::time::{Duration, Instant};

use ny_core::{f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up};
use ny_onnx::vnnlib::{OutputConstraint, VnnLibSpec};
use ny_propagate::{GraphNetwork, PointVjpF64Plan, PointVjpF64Tape};

use super::{
    bounded_min_norm_row_lift, clamp_inside_box, clamp_to_box, constraint_margin,
    property_violated_f64, refine_emit_view, OrtActiveSetCaps,
};
use crate::preset::F64LiftStep;

#[path = "f64_lp_maximin.rs"]
mod lp_maximin;

/// Largest searched input dimension (the bounded solve is `O(dim^2 k^2)` in
/// the worst case; the ORT repair uses the same cap).
const MAX_DIM: usize = 512;
/// Largest conjunction handled (the ORT repair's cap).
const MAX_CONJUNCTS: usize = 64;
/// Seeds whose exact-f64 margin is below `-NEAR_BAND` are not near misses.
const NEAR_BAND: f64 = 2e-2;
/// Conjuncts within this band of the worst one join the active set.
const ACTIVE_BAND: f64 = 1e-4;
/// How many times `delta` may double when f64 clears it but the trusted
/// forward still disagrees.
pub(super) const TARGET_ESCALATIONS: u32 = 4;
/// Backtracking steps per Newton iteration.
const LINE_SEARCH_STEPS: usize = 14;
/// Newton iterations per lineage before `OrtActiveSetCaps::iteration_scale`.
pub(super) const BASE_ITERS: usize = 64;
/// Smallest phase worth building the twin for. The side-branch plan build
/// walks the full graph a handful of times and is not itself deadline-bounded,
/// so a phase shorter than this is declined rather than risk the build running
/// past the descent deadline.
pub(super) const MIN_PHASE_BUDGET: Duration = Duration::from_secs(2);

/// Frozen post-BaB active-set policy, read once from the preset snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PostBabActiveSetPolicy {
    /// `attack.active_set_repair_secs`; `None` = the historical constants.
    repair_total_cap: Option<Duration>,
    /// `attack.f64_target_margin_ulps`; `None` = no f64 stage.
    f64_target_ulps: Option<u32>,
    /// `attack.f64_lift_step` (#sbench-f64-maximin); `MinNorm` whenever the
    /// f64 stage is unarmed.
    lift_step: F64LiftStep,
    /// `attack.f64_basin_hops`; `0` unless `lift_step` is `LpMaximin`.
    basin_hops: usize,
}

impl PostBabActiveSetPolicy {
    /// No key set: the historical 3 s / 6 s / 28 / 32 active-set constants and
    /// no f64 stage — the tail every shipped preset runs.
    pub(super) const HISTORICAL: Self = Self {
        repair_total_cap: None,
        f64_target_ulps: None,
        lift_step: F64LiftStep::MinNorm,
        basin_hops: 0,
    };

    /// Resolve from the frozen, already-validated preset snapshot. Values the
    /// validator would refuse are treated as absent here (defence in depth for
    /// hand-built presets; a shipped yaml cannot reach this with one).
    pub(super) fn from_preset(preset: Option<&crate::preset::PresetConfig>) -> Self {
        let Some(attack) = preset.map(|preset| &preset.attack) else {
            return Self::HISTORICAL;
        };
        let f64_target_ulps = attack
            .f64_target_margin_ulps
            .filter(|ulps| (1..=crate::preset::F64_TARGET_MARGIN_ULPS_MAX).contains(ulps));
        // The step keys configure the f64 stage; without it they configure
        // nothing (the validator refuses that combination outright).
        let lift_step = if f64_target_ulps.is_some() {
            attack.f64_lift_step.unwrap_or_default()
        } else {
            F64LiftStep::MinNorm
        };
        let basin_hops = if lift_step == F64LiftStep::LpMaximin {
            attack
                .f64_basin_hops
                .filter(|hops| *hops <= crate::preset::F64_BASIN_HOPS_MAX)
                .unwrap_or(0)
        } else {
            0
        };
        Self {
            repair_total_cap: attack
                .active_set_repair_secs
                .filter(|secs| {
                    secs.is_finite()
                        && (crate::preset::ACTIVE_SET_REPAIR_SECS_MIN
                            ..=crate::preset::ACTIVE_SET_REPAIR_SECS_MAX)
                            .contains(secs)
                })
                .and_then(|secs| Duration::try_from_secs_f64(secs).ok()),
            f64_target_ulps,
            lift_step,
            basin_hops,
        }
    }

    /// The active-set caps both phases use.
    pub(super) fn caps(self) -> OrtActiveSetCaps {
        match self.repair_total_cap {
            None => OrtActiveSetCaps::HISTORICAL,
            Some(total) => OrtActiveSetCaps::from_total(total),
        }
    }

    /// The caps of the trusted-ORT active-set phase. Identical to
    /// [`Self::caps`] except under `attack.f64_lift_step: lp_maximin`, where
    /// the ORT phase keeps the historical 3 s / 6 s / x1 caps and the
    /// configured `attack.active_set_repair_secs` total funds the f64 lift
    /// instead (#sbench-f64-maximin): on SoundnessBench the ORT phase stalls
    /// at its first-seed cap (`scratchpad/refine2/sbs_9.raw`: -1.1e-5 ->
    /// -4.1e-6 in 3 s, then "trust region exhausted"), and under shared caps it
    /// takes the whole post-BaB window before the f64 stage can start.
    pub(super) fn ort_phase_caps(self) -> OrtActiveSetCaps {
        match self.lift_step {
            F64LiftStep::MinNorm => self.caps(),
            F64LiftStep::LpMaximin => OrtActiveSetCaps::HISTORICAL,
        }
    }

    /// `Some(ulps)` iff the f64 target-margin stage is armed.
    pub(super) fn f64_target_ulps(self) -> Option<u32> {
        self.f64_target_ulps
    }

    /// The f64 stage's step rule (`MinNorm` unless the stage is armed with
    /// `attack.f64_lift_step: lp_maximin`).
    pub(super) fn lift_step(self) -> F64LiftStep {
        self.lift_step
    }

    /// Basin hops of the LP-maximin lift (`0` under `MinNorm`).
    pub(super) fn basin_hops(self) -> usize {
        self.basin_hops
    }

    /// Where the f64 phase's wall budget comes from (flight record).
    pub(super) fn f64_budget_source(self) -> &'static str {
        if self.repair_total_cap.is_some() {
            "attack.active_set_repair_secs"
        } else {
            "historical_caps"
        }
    }

    /// True iff either key is set.
    pub(super) fn is_configured(self) -> bool {
        self != Self::HISTORICAL
    }

    /// The scored-run-entry policy line. Printed whenever either key is set,
    /// so a run that never reaches the post-BaB tail still shows the arming.
    pub(super) fn announce(self) {
        if !self.is_configured() {
            return;
        }
        let caps = self.caps();
        eprintln!(
            "[f64-attack] policy armed={} target_ulps={} active_set_repair_secs={} \
             per_seed_cap_s={:.3} total_cap_s={:.3} iteration_scale={}",
            u8::from(self.f64_target_ulps.is_some()),
            self.f64_target_ulps
                .map_or_else(|| "none".to_string(), |ulps| ulps.to_string()),
            self.repair_total_cap.map_or_else(
                || "default".to_string(),
                |cap| format!("{:.3}", cap.as_secs_f64())
            ),
            caps.per_seed.as_secs_f64(),
            caps.total.as_secs_f64(),
            caps.iteration_scale,
        );
        if self.lift_step == F64LiftStep::LpMaximin {
            eprintln!(
                "[f64-lift] policy step={} basin_hops={} target_ulps={} f64_phase_total_s={:.3} \
                 f64_per_seed_cap_s={:.3} ort_phase_caps=historical budget_source={}",
                self.lift_step.stable_token(),
                self.basin_hops,
                self.f64_target_ulps.unwrap_or(0),
                caps.total.as_secs_f64(),
                caps.per_seed.as_secs_f64(),
                self.f64_budget_source(),
            );
        }
    }
}

/// The per-tail `[f64-attack]` engagement line. Printed exactly once, when the
/// value drops, on every post-BaB tail entered with the stage armed — `Drop`
/// covers every early return — so "never reached", "declined (reason)",
/// "ran and found nothing" and "handed a candidate to the gate" are four
/// different lines, never one silent absence.
pub(super) struct F64AttackTelemetry {
    armed: bool,
    target_ulps: u32,
    reached: bool,
    reason: &'static str,
    plan_kind: &'static str,
    branches: usize,
    exact_branch_maps: usize,
    seeds: usize,
    seeds_run: usize,
    seeds_declined: usize,
    newton_iters: usize,
    trusted_evals: usize,
    candidates: usize,
    handoffs: usize,
    initial_margin_f64: f64,
    best_margin_f64: f64,
    best_margin_ort: f64,
    target_delta: f64,
    started: Instant,
    /// #sbench-f64-maximin: print the `[f64-lift]` line too (armed AND
    /// `lp_maximin`).
    lift_armed: bool,
    basin_hops_cap: usize,
    lift: F64LiftStats,
    lift_phase_budget_s: f64,
    lift_budget_source: &'static str,
}

impl F64AttackTelemetry {
    pub(super) fn new(policy: PostBabActiveSetPolicy) -> Self {
        Self {
            lift_armed: policy.f64_target_ulps.is_some()
                && policy.lift_step == F64LiftStep::LpMaximin,
            basin_hops_cap: policy.basin_hops,
            lift: F64LiftStats::default(),
            lift_phase_budget_s: f64::NAN,
            lift_budget_source: policy.f64_budget_source(),
            armed: policy.f64_target_ulps.is_some(),
            target_ulps: policy.f64_target_ulps.unwrap_or(0),
            reached: false,
            reason: "not_reached",
            plan_kind: "none",
            branches: 0,
            exact_branch_maps: 0,
            seeds: 0,
            seeds_run: 0,
            seeds_declined: 0,
            newton_iters: 0,
            trusted_evals: 0,
            candidates: 0,
            handoffs: 0,
            initial_margin_f64: f64::NEG_INFINITY,
            best_margin_f64: f64::NEG_INFINITY,
            best_margin_ort: f64::NEG_INFINITY,
            target_delta: f64::NAN,
            started: Instant::now(),
        }
    }

    /// The stage point was reached with `seeds` lineages.
    pub(super) fn reach(&mut self, seeds: usize) {
        self.reached = true;
        self.seeds = seeds;
        self.reason = "reached";
    }

    /// The stage declined before running any lineage.
    pub(super) fn decline(&mut self, reason: &'static str) {
        self.reason = reason;
    }

    /// The twin was built.
    pub(super) fn note_plan(&mut self, plan: &PointVjpF64Plan) {
        self.plan_kind = plan.kind();
        self.branches = plan.branch_count();
        self.exact_branch_maps = plan.exact_branch_maps();
        self.reason = "ran";
    }

    /// A lineage the repair refused (arity, clause shape, far seed).
    pub(super) fn note_seed_declined(&mut self) {
        self.seeds_declined += 1;
    }

    /// A lineage ran.
    pub(super) fn absorb(&mut self, outcome: &F64RepairOutcome) {
        self.seeds_run += 1;
        self.newton_iters += outcome.newton_iters;
        self.trusted_evals += outcome.trusted_evals;
        self.candidates += outcome.candidates;
        if outcome.best_margin_f64 > self.best_margin_f64 {
            self.best_margin_f64 = outcome.best_margin_f64;
            self.initial_margin_f64 = outcome.initial_margin_f64;
            self.target_delta = outcome.target_delta;
        }
        if outcome.best_margin_ort > self.best_margin_ort {
            self.best_margin_ort = outcome.best_margin_ort;
        }
    }

    /// A candidate was handed to the caller's local precheck and gate.
    pub(super) fn note_handoff(&mut self) {
        self.handoffs += 1;
    }

    /// The f64 phase's wall budget (#sbench-f64-maximin telemetry).
    pub(super) fn note_lift_budget(&mut self, phase_budget: Duration) {
        self.lift_phase_budget_s = phase_budget.as_secs_f64();
    }

    /// One LP-maximin lineage's counters.
    pub(super) fn absorb_lift(&mut self, stats: &F64LiftStats) {
        self.lift.slp_iters += stats.slp_iters;
        self.lift.accepted += stats.accepted;
        self.lift.lp_solves += stats.lp_solves;
        self.lift.lp_failures += stats.lp_failures;
        self.lift.lp_pivots += stats.lp_pivots;
        self.lift.lp_cut_short += stats.lp_cut_short;
        self.lift.hops += stats.hops;
        self.lift.hops_improved += stats.hops_improved;
        self.lift.rho_final = stats.rho_final;
        self.lift.stop = stats.stop;
    }

    /// The fixed-schema `[f64-lift]` line (printed on every armed
    /// `lp_maximin` tail, from `Drop`). The brief's prefix comes first.
    fn lift_line(&self) -> String {
        format!(
            "[f64-lift] step=lp_maximin iters={} hops={} best_f64={:.3e} best_ort={:.3e} \
             rho_final={:.3e} armed={} reached={} seeds_run={} accepted={} lp_solves={} \
             lp_failures={} lp_pivots={} lp_cut_short={} hops_cap={} hops_improved={} \
             candidates={} handoffs={} target_delta={:.3e} phase_budget_s={:.3} \
             budget_source={} stop={} reason={} elapsed_s={:.3}",
            self.lift.slp_iters,
            self.lift.hops,
            self.best_margin_f64,
            self.best_margin_ort,
            self.lift.rho_final,
            u8::from(self.armed),
            u8::from(self.reached),
            self.seeds_run,
            self.lift.accepted,
            self.lift.lp_solves,
            self.lift.lp_failures,
            self.lift.lp_pivots,
            self.lift.lp_cut_short,
            self.basin_hops_cap,
            self.lift.hops_improved,
            self.candidates,
            self.handoffs,
            self.target_delta,
            self.lift_phase_budget_s,
            self.lift_budget_source,
            if self.lift.stop.is_empty() {
                "none"
            } else {
                self.lift.stop
            },
            self.reason,
            self.started.elapsed().as_secs_f64(),
        )
    }

    fn line(&self) -> String {
        let engaged = self.seeds_run > 0;
        format!(
            "[f64-attack] armed={} engaged={} reached={} target_ulps={} target_delta={:.3e} \
             plan={} branches={} exact_branch_maps={} seeds={} seeds_run={} seeds_declined={} \
             iters={} trusted_evals={} candidates={} handoffs={} initial_margin_f64={:.3e} \
             best_margin_f64={:.3e} best_margin_ort={:.3e} reason={} elapsed_s={:.3}",
            u8::from(self.armed),
            u8::from(engaged),
            u8::from(self.reached),
            self.target_ulps,
            self.target_delta,
            self.plan_kind,
            self.branches,
            self.exact_branch_maps,
            self.seeds,
            self.seeds_run,
            self.seeds_declined,
            self.newton_iters,
            self.trusted_evals,
            self.candidates,
            self.handoffs,
            self.initial_margin_f64,
            self.best_margin_f64,
            self.best_margin_ort,
            self.reason,
            self.started.elapsed().as_secs_f64(),
        )
    }
}

impl Drop for F64AttackTelemetry {
    fn drop(&mut self) {
        if self.armed {
            eprintln!("{}", self.line());
        }
        if self.lift_armed {
            eprintln!("{}", self.lift_line());
        }
    }
}

/// One LP-maximin lineage's counters (#sbench-f64-maximin).
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct F64LiftStats {
    /// SLP iterations (every LP-maximin step tried, accepted or not).
    pub(super) slp_iters: usize,
    /// Steps accepted on a strict exact-f64 improvement.
    pub(super) accepted: usize,
    pub(super) lp_solves: usize,
    /// LP solves that returned no step (malformed or degenerate input).
    pub(super) lp_failures: usize,
    /// Simplex basis changes plus bound flips, summed.
    pub(super) lp_pivots: usize,
    /// LP solves cut short by the pivot cap or the deadline (their step was
    /// still feasible and was still judged exactly).
    pub(super) lp_cut_short: usize,
    /// Basin hops started.
    pub(super) hops: usize,
    /// Hops whose run beat the incumbent.
    pub(super) hops_improved: usize,
    /// The last run's final trust radius (fraction of the width).
    pub(super) rho_final: f64,
    pub(super) stop: &'static str,
}

/// The per-lineage generator seed of the hop kicks: a pure function of the
/// seed point, so a run is reproducible.
pub(super) fn lift_rng_seed(seed: &[f32]) -> u64 {
    seed.iter().fold(0xCBF2_9CE4_8422_2325u64, |hash, value| {
        (hash ^ u64::from(value.to_bits())).wrapping_mul(0x0100_0000_01B3)
    })
}

/// One lineage of the armed f64 stage, by `attack.f64_lift_step`:
/// [`f64_target_margin_repair`] (min-norm, the default — the call and its
/// outcome are exactly C8's) or the LP-maximin lift with its basin hops. The
/// second element is `Some` only for the LP-maximin lift.
#[allow(clippy::too_many_arguments)]
pub(super) fn f64_lift_lineage<F>(
    policy: PostBabActiveSetPolicy,
    plan: &PointVjpF64Plan,
    trusted: F,
    spec: &VnnLibSpec,
    box_lo: &[f32],
    box_hi: &[f32],
    emit_pin: &[Option<f64>],
    seed: &[f32],
    target_ulps: u32,
    deadline: Instant,
    max_iters: usize,
) -> Option<(F64RepairOutcome, Option<F64LiftStats>)>
where
    F: FnMut(&[f32]) -> Option<Vec<f64>>,
{
    match policy.lift_step {
        F64LiftStep::MinNorm => f64_target_margin_repair(
            plan,
            trusted,
            spec,
            box_lo,
            box_hi,
            emit_pin,
            seed,
            target_ulps,
            deadline,
            max_iters,
        )
        .map(|outcome| (outcome, None)),
        F64LiftStep::LpMaximin => lp_maximin::f64_lp_maximin_lift(
            plan,
            trusted,
            spec,
            box_lo,
            box_hi,
            emit_pin,
            seed,
            target_ulps,
            policy.basin_hops,
            lift_rng_seed(seed),
            deadline,
            max_iters,
        )
        .map(|(outcome, stats)| (outcome, Some(stats))),
    }
}

/// One lineage's outcome.
#[derive(Debug, Clone)]
pub(super) struct F64RepairOutcome {
    /// A point the trusted forward confirms at zero tolerance AND whose exact
    /// f64 margin is non-negative — the strongest by `min(trusted, f64)`.
    pub(super) violation: Option<Vec<f32>>,
    /// The best exact-f64 point, when strictly better than the seed.
    pub(super) guidance: Option<(Vec<f32>, f64)>,
    pub(super) initial_margin_f64: f64,
    pub(super) best_margin_f64: f64,
    /// Trusted-forward min-conjunct margin at the retained candidate, or the
    /// best one seen at a checked point; `-inf` when never consulted.
    pub(super) best_margin_ort: f64,
    /// The initial `delta` (before any escalation).
    pub(super) target_delta: f64,
    pub(super) newton_iters: usize,
    pub(super) trusted_evals: usize,
    pub(super) candidates: usize,
    pub(super) stop: &'static str,
}

/// One f32 ULP at magnitude `|v|`: the spacing of the f32 binade holding `|v|`
/// rounded to f32 (floored at the smallest normal binade; NaN/overflow read as
/// `f32::MAX`'s binade).
pub(super) fn f32_ulp_at(v: f64) -> f64 {
    let magnitude = v.abs();
    let as_f32 = if magnitude < f64::from(f32::MAX) {
        (magnitude as f32).max(f32::MIN_POSITIVE)
    } else {
        f32::MAX
    };
    let biased_exponent = ((as_f32.to_bits() >> 23) & 0xff) as i32;
    2f64.powi(biased_exponent - 150)
}

/// The conjunction the repair lifts: the single clause (or the flat
/// conjunctive property), as in the ORT repair. `None` for multi-clause
/// disjunctions and per-clause input boxes.
pub(super) fn repair_conjuncts(spec: &VnnLibSpec) -> Option<&[OutputConstraint]> {
    if !spec.per_clause_input_bounds.is_empty() {
        return None;
    }
    let conjuncts: &[OutputConstraint] = if spec.output_constraint_clauses.len() == 1 {
        &spec.output_constraint_clauses[0]
    } else if spec.output_constraint_clauses.is_empty() {
        &spec.output_constraints
    } else {
        return None;
    };
    (!conjuncts.is_empty() && conjuncts.len() <= MAX_CONJUNCTS).then_some(conjuncts)
}

/// `d margin / d Y` for one conjunct, matching `constraint_margin`'s
/// convention (margin >= 0 on the violating side). `None` for an index outside
/// the output or an unknown future constraint shape.
pub(super) fn margin_output_row(c: &OutputConstraint, output_dim: usize) -> Option<Vec<f64>> {
    let (plus, minus): (Option<usize>, Option<usize>) = match c {
        OutputConstraint::LessEq(i, j) | OutputConstraint::LessThan(i, j) => (Some(*j), Some(*i)),
        OutputConstraint::GreaterEq(i, j) | OutputConstraint::GreaterThan(i, j) => {
            (Some(*i), Some(*j))
        }
        OutputConstraint::LessEqConst(i, _) | OutputConstraint::LessThanConst(i, _) => {
            (None, Some(*i))
        }
        OutputConstraint::GreaterEqConst(i, _) | OutputConstraint::GreaterThanConst(i, _) => {
            (Some(*i), None)
        }
        // `OutputConstraint` is #[non_exhaustive]: no gradient for an unknown
        // shape, so the lineage declines (guidance only; never a verdict).
        _ => return None,
    };
    let mut row = vec![0.0f64; output_dim];
    if let Some(p) = plus {
        *row.get_mut(p)? += 1.0;
    }
    if let Some(m) = minus {
        *row.get_mut(m)? -= 1.0;
    }
    Some(row)
}

/// `ulps` f32 ULPs at the largest magnitude among the outputs the conjuncts
/// reference.
pub(super) fn target_delta_for(ulps: u32, outputs: &[f64], rows: &[Vec<f64>]) -> f64 {
    let mut scale = 0.0f64;
    for row in rows {
        for (j, &weight) in row.iter().enumerate() {
            if weight != 0.0 {
                if let Some(&y) = outputs.get(j) {
                    scale = scale.max(y.abs());
                }
            }
        }
    }
    f64::from(ulps) * f32_ulp_at(scale)
}

/// Round `v` to f32 toward the side that raises the lagging margins
/// (`direction > 0` up, `< 0` down, `0` nearest), then clamp into the f32
/// search box.
fn round_toward(v: f64, direction: f64, lo: f32, hi: f32) -> f32 {
    let rounded = if direction > 0.0 {
        f64_to_f32_up(v)
    } else if direction < 0.0 {
        f64_to_f32_down(v)
    } else {
        v as f32
    };
    clamp_to_box(rounded, lo, hi)
}

/// Squared shortfall of every conjunct below `target`.
fn shortfall(margins: &[f64], target: f64) -> f64 {
    margins
        .iter()
        .map(|&m| (target - m).max(0.0))
        .map(|d| d * d)
        .sum()
}

fn min_of(margins: &[f64]) -> f64 {
    margins.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Trusted outputs at an f32 point, widened exactly.
pub(super) fn trusted_outputs(
    forward: &mut ny_onnx::diff::OrtForward,
    x: &[f32],
) -> Option<Vec<f64>> {
    let out = forward.run(x).ok()?;
    Some(out.iter().map(|&v| f32_to_f64_exact(v)).collect())
}

/// Build the twin over the f32 search box, in the graph's own input
/// convention (`attack_point_gradient_input_shape`, as the ORT-refine grad
/// lane does). `None` when the graph is outside the chain / side-branch
/// fragment or the flat dimensions disagree.
pub(super) fn build_plan(
    graph: &GraphNetwork,
    declared_shape: &[usize],
    box_lo: &[f32],
    box_hi: &[f32],
    deadline: Instant,
) -> Option<PointVjpF64Plan> {
    let input_shape = graph.attack_point_gradient_input_shape(declared_shape);
    let lower =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&input_shape), box_lo.to_vec()).ok()?;
    let upper =
        ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&input_shape), box_hi.to_vec()).ok()?;
    let attack_box = ny_tensor::BoundedTensor::new(lower, upper).ok()?;
    let plan = PointVjpF64Plan::build(graph, &attack_box, Some(deadline))?;
    (plan.input_dim() == box_lo.len()).then_some(plan)
}

/// Exact-f64 per-conjunct margins at the f32 point `x`'s witness input view.
fn exact_margins(
    plan: &PointVjpF64Plan,
    conjuncts: &[OutputConstraint],
    emit_pin: &[Option<f64>],
    x: &[f32],
) -> Option<(Vec<f64>, Vec<f64>, PointVjpF64Tape)> {
    let view = refine_emit_view(x, emit_pin);
    let (outputs, tape) = plan.forward(&view)?;
    let margins: Vec<f64> = conjuncts
        .iter()
        .map(|c| constraint_margin(c, &outputs))
        .collect();
    margins
        .iter()
        .all(|m| m.is_finite())
        .then_some((margins, outputs, tape))
}

/// The trusted verdict at `x`: `(violated at zero tolerance, min-conjunct
/// trusted margin)`. Acceptance is the unchanged `property_violated_f64` on
/// the witness input view.
fn trusted_check<F>(
    trusted: &mut F,
    spec: &VnnLibSpec,
    conjuncts: &[OutputConstraint],
    emit_pin: &[Option<f64>],
    x: &[f32],
) -> Option<(bool, f64)>
where
    F: FnMut(&[f32]) -> Option<Vec<f64>>,
{
    let outputs = trusted(x)?;
    let margin = min_of(
        &conjuncts
            .iter()
            .map(|c| constraint_margin(c, &outputs))
            .collect::<Vec<f64>>(),
    );
    let violated = property_violated_f64(spec, &refine_emit_view(x, emit_pin), &outputs);
    Some((violated, margin))
}

/// Exact-f64 target-margin active-set repair of one seed lineage (module
/// docs). `trusted` is the trusted forward (ONNX Runtime on the scored path),
/// returning the flat outputs at an f32 point; it is consulted only at points
/// whose exact-f64 margin is non-negative.
///
/// Returns `None` when the lineage is out of scope (arity, clause shape, a
/// seed farther than `NEAR_BAND`, a twin/output mismatch), else the outcome.
/// `outcome.violation` is `Some` ONLY for a point the trusted forward
/// confirms at zero tolerance and whose exact-f64 margin is `>= 0`.
#[allow(clippy::too_many_arguments)]
pub(super) fn f64_target_margin_repair<F>(
    plan: &PointVjpF64Plan,
    mut trusted: F,
    spec: &VnnLibSpec,
    box_lo: &[f32],
    box_hi: &[f32],
    emit_pin: &[Option<f64>],
    seed: &[f32],
    target_ulps: u32,
    deadline: Instant,
    max_iters: usize,
) -> Option<F64RepairOutcome>
where
    F: FnMut(&[f32]) -> Option<Vec<f64>>,
{
    let dim = box_lo.len();
    if dim == 0
        || dim > MAX_DIM
        || seed.len() != dim
        || box_hi.len() != dim
        || emit_pin.len() != dim
        || plan.input_dim() != dim
        || target_ulps == 0
    {
        return None;
    }
    let conjuncts = repair_conjuncts(spec)?;
    let output_dim = plan.output_dim();
    let conjunct_rows: Vec<Vec<f64>> = conjuncts
        .iter()
        .map(|c| margin_output_row(c, output_dim))
        .collect::<Option<Vec<Vec<f64>>>>()?;
    let width: Vec<f64> = box_lo
        .iter()
        .zip(box_hi)
        .map(|(&lo, &hi)| (f32_to_f64_exact(hi) - f32_to_f64_exact(lo)).max(0.0))
        .collect();
    if width.iter().all(|&w| w <= 0.0) {
        return None;
    }

    let mut x: Vec<f32> = seed
        .iter()
        .enumerate()
        .map(|(d, &v)| clamp_to_box(v, box_lo[d], box_hi[d]))
        .collect();
    let (mut margins, seed_outputs, mut tape) = exact_margins(plan, conjuncts, emit_pin, &x)?;
    let initial = min_of(&margins);
    if initial < -NEAR_BAND {
        return None;
    }
    let base_target = target_delta_for(target_ulps, &seed_outputs, &conjunct_rows);
    if !(base_target.is_finite() && base_target > 0.0) {
        return None;
    }

    let mut outcome = F64RepairOutcome {
        violation: None,
        guidance: None,
        initial_margin_f64: initial,
        best_margin_f64: initial,
        best_margin_ort: f64::NEG_INFINITY,
        target_delta: base_target,
        newton_iters: 0,
        trusted_evals: 0,
        candidates: 0,
        stop: "iteration cap",
    };
    let mut target = base_target;
    let mut escalations = 0u32;
    let mut best_point = x.clone();
    // Retained candidate, ranked by min(trusted margin, exact-f64 margin).
    let mut best_candidate: Option<(Vec<f32>, f64)> = None;
    let mut last_checked: Option<Vec<f32>> = None;
    let mut cap = 0.25f64;

    // Consult the trusted forward at `point` (exact-f64 min margin `f64_min`
    // >= 0) once; retain it when it violates. Returns `(violated,
    // trusted_min)`, or `None` if the trusted forward failed. A caller whose
    // point a LATER check compares against records it in `last_checked`
    // first (the final consultation has no later reader).
    macro_rules! consult {
        ($point:expr, $f64_min:expr) => {{
            let point: &[f32] = $point;
            let f64_min: f64 = $f64_min;
            outcome.trusted_evals += 1;
            match trusted_check(&mut trusted, spec, conjuncts, emit_pin, point) {
                None => None,
                Some((violated, trusted_min)) => {
                    if violated {
                        outcome.candidates += 1;
                        let joint = trusted_min.min(f64_min);
                        if best_candidate
                            .as_ref()
                            .is_none_or(|(_, best_joint)| joint > *best_joint)
                        {
                            best_candidate = Some((point.to_vec(), joint));
                            outcome.best_margin_ort = trusted_min;
                        }
                    } else if best_candidate.is_none() && trusted_min > outcome.best_margin_ort {
                        outcome.best_margin_ort = trusted_min;
                    }
                    Some((violated, trusted_min))
                }
            }
        }};
    }

    for iteration in 0..max_iters {
        if Instant::now() >= deadline {
            outcome.stop = "deadline";
            break;
        }
        outcome.newton_iters = iteration + 1;
        let min_margin = min_of(&margins);
        if min_margin >= 0.0 && last_checked.as_deref() != Some(&x[..]) {
            last_checked = Some(x.clone());
            let Some((violated, _)) = consult!(&x[..], min_margin) else {
                outcome.stop = "trusted forward failed";
                break;
            };
            if min_margin >= target {
                if violated {
                    outcome.stop = "target reached";
                    break;
                }
                // Exact f64 clears delta, the trusted forward does not: the
                // f32 rounding here exceeds delta. Raise it and keep lifting.
                if escalations >= TARGET_ESCALATIONS {
                    outcome.stop = "target escalations exhausted";
                    break;
                }
                escalations += 1;
                target *= 2.0;
            }
        }

        let cutoff = (min_margin + ACTIVE_BAND).max(2.0 * target);
        let active: Vec<usize> = margins
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| (m < cutoff).then_some(i))
            .collect();
        if active.is_empty() {
            outcome.stop = "no active constraints";
            break;
        }
        let active_rows: Vec<Vec<f64>> = active.iter().map(|&c| conjunct_rows[c].clone()).collect();
        let Some(rows) = plan.vjp_rows(&tape, &active_rows) else {
            outcome.stop = "exact vjp failed";
            break;
        };
        let rhs: Vec<f64> = active.iter().map(|&c| target - margins[c]).collect();
        let lower: Vec<f64> = (0..dim)
            .map(|d| (f32_to_f64_exact(box_lo[d]) - f32_to_f64_exact(x[d])).max(-cap * width[d]))
            .collect();
        let upper: Vec<f64> = (0..dim)
            .map(|d| (f32_to_f64_exact(box_hi[d]) - f32_to_f64_exact(x[d])).min(cap * width[d]))
            .collect();
        let Some(delta) = bounded_min_norm_row_lift(&rows, &rhs, &lower, &upper, Some(deadline))
        else {
            outcome.stop = if Instant::now() >= deadline {
                "deadline during bounded solve"
            } else {
                "bounded solve failed"
            };
            break;
        };
        if !delta.iter().any(|&v| v != 0.0) {
            outcome.stop = "zero bounded step";
            break;
        }
        // Favourable rounding: the shortfall-weighted gradient of the lagging
        // conjuncts says which f32 neighbour raises them.
        let direction: Vec<f64> = (0..dim)
            .map(|d| {
                active
                    .iter()
                    .zip(&rows)
                    .map(|(&c, row)| (target - margins[c]).max(0.0) * row[d])
                    .sum()
            })
            .collect();

        let old_shortfall = shortfall(&margins, target);
        let mut step = 1.0f64;
        let mut accepted = false;
        for _ in 0..LINE_SEARCH_STEPS {
            if Instant::now() >= deadline {
                break;
            }
            let candidate: Vec<f32> = (0..dim)
                .map(|d| {
                    round_toward(
                        f32_to_f64_exact(x[d]) + step * delta[d],
                        direction[d],
                        box_lo[d],
                        box_hi[d],
                    )
                })
                .collect();
            if candidate == x {
                step *= 0.25;
                continue;
            }
            let Some((cand_margins, _, cand_tape)) =
                exact_margins(plan, conjuncts, emit_pin, &candidate)
            else {
                step *= 0.25;
                continue;
            };
            let cand_min = min_of(&cand_margins);
            if cand_min > outcome.best_margin_f64 {
                outcome.best_margin_f64 = cand_min;
                best_point = candidate.clone();
            }
            if shortfall(&cand_margins, target) < old_shortfall || cand_min > min_margin {
                x = candidate;
                margins = cand_margins;
                tape = cand_tape;
                cap = (cap * 1.3).min(0.5);
                accepted = true;
                break;
            }
            step *= 0.25;
        }
        if !accepted {
            cap *= 0.25;
            if cap < 1e-9 {
                outcome.stop = "trust region exhausted";
                break;
            }
        }
    }

    // A final accepted iterate (or the best exact point) that cleared zero but
    // was never shown to the trusted forward gets its one consultation now.
    let final_min = min_of(&margins);
    if final_min >= 0.0 && last_checked.as_deref() != Some(&x[..]) {
        let point = x.clone();
        last_checked = Some(point.clone());
        let _ = consult!(&point[..], final_min);
    }
    if outcome.best_margin_f64 >= 0.0 && last_checked.as_deref() != Some(&best_point[..]) {
        let best = outcome.best_margin_f64;
        let _ = consult!(&best_point[..], best);
    }

    outcome.violation = best_candidate
        .as_ref()
        .and_then(|(point, _)| clamp_inside_box(point, box_lo, box_hi));
    if outcome.best_margin_f64 > initial {
        outcome.guidance = Some((best_point, outcome.best_margin_f64));
    }
    Some(outcome)
}

#[cfg(test)]
#[path = "f64_target_attack_tests.rs"]
mod tests;
