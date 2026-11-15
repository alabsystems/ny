// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #disj-odi (FRONTIER_PLAN §7.F1): output-diversified initialization (ODI)
//! and dangerous-class focus lanes for the batched disjunctive exact-VJP
//! driver (`graph_pgd_vjp_batched_disj.rs`).
//!
//! # Why
//!
//! Every cifar100/tinyimagenet row falsifies through the disjunctive driver,
//! and every one of them reported `odi_steps_configured=8 odi_vjp_steps=0`:
//! the conjunctive sibling runs ODI by default, the disjunctive driver never
//! read the field. The calibrated torch recipe (`ct/odi.py`: 8 ODI steps at
//! eta = 1 half-range, then bias-corrected-Adam sign steps on the hinge
//! `min(Y_t - Y_label, 0.01)` with a cosine step from 0.3h) finds the two
//! tinyimagenet GAP counterexamples ny times out on (idx_1831 class 43,
//! idx_5529 class 22) within 50 steps; plain sign PGD does not. The dangerous
//! class ranks 2nd (43) and 1st (22) at the clean point, so the focus wave fans
//! out over the two closest clauses instead of trusting rank 1 alone.
//!
//! # What
//!
//! * `attack.pgd_disj_odi` — before the step loop of every restart wave, the
//!   wave's DYNAMIC lanes take `attack.pgd_odi_steps` box-clamped sign steps
//!   of `attack.pgd_odi_eta` half-ranges up a fixed random output direction
//!   (one batched exact VJP per step), screening every ODI point for a free
//!   candidate. Targeted lanes keep their clean init.
//! * `attack.pgd_disj_focus_lanes` — after the restart waves finish with no
//!   candidate and no deadline, that many lanes are pinned round-robin to the
//!   [`FOCUS_FANOUT`] closest-to-violation clauses seen so far (clean point
//!   plus every restart-wave point: rule `wave0`; `clean` when no wave point
//!   was observed), start uniform plus the same ODI phase, and ascend their
//!   clause's margin row under the `attack.pgd_hinge_tau` hinge with a
//!   momentum-sign step on the recipe's cosine schedule, until
//!   [`FOCUS_SCHEDULE_STEPS`] steps or the attack deadline. The wave is
//!   RSS-admitted and halves its width on a wide-VJP refusal.
//!
//! Both keys dark (unset/false/0): the driver never enters this module's
//! phases, reads none of the ODI/hinge fields, prints nothing, and is
//! byte-identical (`disj_odi_dark_is_byte_identical`).
//!
//! # Soundness
//!
//! Attack-only. Every point these phases reach passes the driver's SAME
//! screen: a template-forward hit is re-screened on the exact/ORT-routed
//! forward against `noise_scaled_margin` (`max(1e-4, 1e-5·max|y|)`) before it
//! becomes a candidate; the caller then runs `re_evaluate_and_confirm`, and
//! vnncomp's `gate_sat_with_trusted_oracle` re-confirms before any `sat` is
//! published. Nothing here touches a bound, and nothing can produce `unsat`.
//! The focus wave never sets the restart-sequence deadline flag and never
//! adds to the restart count, so the attack-extension decision and its seed
//! continuation are unchanged.
//!
//! # Observable
//!
//! Armed, one fixed-schema line per driver invocation on every return path
//! (`[pgd-vjp-disj] disj-odi schema=v1 odi= ...`, [`DisjOdiReceipt::line`]),
//! plus `[pgd-vjp-disj] focus class=<c1,c2> lanes=<n> rule=<wave0|clean> ...`
//! when the focus wave starts. The shared `exact_vjp_execution` evidence
//! counts the ODI passes (`odi_vjp_steps`, `odi_move_count`). An armed run
//! with `odi_vjp_steps=0` is VOID.

use std::time::Instant;

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::{GemmEngine, GpuCrownBackward};
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::{ExactVjpPgdConfig, GraphNetwork, PointVjpWavePlan};
use ny_tensor::BoundedTensor;

use super::attack_stall::AttackStallMonitor;
use super::graph_pgd::{
    best_clause_bottleneck_margin, constraint_target, GraphDisjunctiveAttackOutcome, GraphPgdTarget,
};
use super::graph_pgd_exact_vjp_evidence::ExactVjpExecution;
use super::graph_pgd_init::{evaluate_graph, sample_center_point, sample_uniform_point, SimpleRng};

/// Clauses the focus wave fans out over. Gate 0 (clean-point ranking, ORT):
/// the dangerous class is rank 2 on tinyimagenet idx_1831 (43 behind 76) and
/// rank 1 on idx_5529 (22), so a single-class focus would miss one of the two
/// calibration rows under the clean rule.
pub(super) const FOCUS_FANOUT: usize = 2;
/// Length S of the focus cosine schedule — the calibrated recipe's 200-step
/// schedule, whose counterexamples appeared within its first 50 steps (step
/// sizes 0.26-0.30 half-ranges). The wave also stops at the attack deadline.
pub(super) const FOCUS_SCHEDULE_STEPS: usize = 200;
/// Hard cap on focus lanes, whatever the key asks for (the receipt reports
/// both the request and the admitted width).
pub(super) const FOCUS_MAX_LANES: usize = 256;
/// Admission ceiling when the memory headroom is unobservable.
const FOCUS_UNOBSERVED_HEADROOM_CAP: u64 = 4 << 30;
/// Seed base of the focus lanes' RNG streams — disjoint from the restart
/// waves' `42 + r`, so a focus lane never replays a restart lane.
const FOCUS_SEED_BASE: u64 = 0x0D15_F0C5_0000_0000;
/// First-moment decay of the focus lanes' Adam direction (the recipe's
/// beta1). `sign(m_hat / (sqrt(v_hat) + eps)) == sign(m)`: bias correction
/// and the second moment never change the sign, so only `m` is kept.
const FOCUS_BETA1: f32 = 0.9;

/// The two #disj-odi keys plus the existing fields they compose with, frozen
/// once per driver invocation from the typed per-run policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DisjOdiPolicy {
    /// `attack.pgd_disj_odi`.
    pub(super) odi: bool,
    /// `attack.pgd_odi_steps` (shared with the conjunctive driver).
    pub(super) odi_steps: usize,
    /// `attack.pgd_odi_eta` (shared with the conjunctive driver).
    pub(super) odi_eta: f32,
    /// `attack.pgd_disj_focus_lanes`.
    pub(super) focus_lanes: usize,
    /// `attack.pgd_hinge_tau` (shared with the conjunctive driver).
    pub(super) hinge_tau: f32,
}

impl DisjOdiPolicy {
    pub(super) fn from_config(policy: ExactVjpPgdConfig) -> Self {
        Self {
            odi: policy.disj_odi,
            odi_steps: policy.odi_steps,
            odi_eta: policy.odi_eta,
            focus_lanes: policy.disj_focus_lanes,
            hinge_tau: policy.hinge_tau,
        }
    }

    /// Either key armed: the receipt prints and the phases may run.
    pub(super) fn armed(self) -> bool {
        self.odi || self.focus_lanes > 0
    }

    /// ODI steps each restart wave's dynamic lanes take (zero when dark).
    pub(super) fn wave_odi_steps(self) -> usize {
        if self.odi {
            self.odi_steps
        } else {
            0
        }
    }

    /// Whether the driver must track per-clause margins for the focus wave.
    pub(super) fn tracks_focus(self) -> bool {
        self.focus_lanes > 0
    }
}

/// Per-invocation receipt. Counters mirror the shared exact-VJP evidence for
/// the ODI passes this module issues (the evidence goes to the flight record,
/// this line goes to stderr).
#[derive(Debug, Clone)]
pub(super) struct DisjOdiReceipt {
    policy: DisjOdiPolicy,
    /// ODI phases that completed at least one VJP step.
    pub(super) odi_waves: usize,
    /// Lanes that entered an ODI phase.
    pub(super) odi_lanes: usize,
    /// Successful ODI VJP passes.
    pub(super) odi_vjp_steps: usize,
    /// Coordinates whose f32 bits changed under an ODI step.
    pub(super) odi_move_count: usize,
    /// Template-screen hits the exact re-screen refused (ODI + focus lanes).
    pub(super) near_misses: usize,
    /// `none` | `wave0` | `clean`.
    pub(super) focus_rule: &'static str,
    /// Labels of the pinned clauses, best first.
    pub(super) focus_classes: Vec<String>,
    /// Focus lanes that ran (after RSS admission and any width halving).
    pub(super) focus_admitted: usize,
    /// Width halvings after a wide-VJP refusal.
    pub(super) focus_retries: usize,
    /// Successful focus ascent VJP passes.
    pub(super) focus_steps: usize,
    /// Best margin of a focus lane's own pinned clause.
    pub(super) focus_best_margin: f32,
    /// Why the focus stage ended (`off` when the key is dark).
    pub(super) focus_reason: &'static str,
    /// Driver outcome token, filled by the driver's wrapper.
    pub(super) outcome: &'static str,
}

impl DisjOdiReceipt {
    pub(super) fn new(policy: DisjOdiPolicy) -> Self {
        Self {
            policy,
            odi_waves: 0,
            odi_lanes: 0,
            odi_vjp_steps: 0,
            odi_move_count: 0,
            near_misses: 0,
            focus_rule: "none",
            focus_classes: Vec::new(),
            focus_admitted: 0,
            focus_retries: 0,
            focus_steps: 0,
            focus_best_margin: f32::NEG_INFINITY,
            focus_reason: if policy.tracks_focus() {
                "not_reached"
            } else {
                "off"
            },
            outcome: "internal_error",
        }
    }

    pub(super) fn policy(&self) -> DisjOdiPolicy {
        self.policy
    }

    /// The fixed-schema observable. Every field is always present.
    pub(super) fn line(&self) -> String {
        let classes = if self.focus_classes.is_empty() {
            "none".to_string()
        } else {
            self.focus_classes.join(",")
        };
        let best = if self.focus_best_margin.is_finite() {
            format!("{:.6}", self.focus_best_margin)
        } else {
            "none".to_string()
        };
        format!(
            "[pgd-vjp-disj] disj-odi schema=v1 odi={} odi_steps={} odi_eta={} odi_waves={} odi_lanes={} odi_vjp_steps={} odi_move_count={} near_misses={} focus_lanes={} focus_admitted={} focus_rule={} focus_class={classes} focus_steps={} focus_retries={} focus_tau={} focus_best_margin={best} focus_reason={} outcome={}",
            u8::from(self.policy.odi),
            self.policy.odi_steps,
            self.policy.odi_eta,
            self.odi_waves,
            self.odi_lanes,
            self.odi_vjp_steps,
            self.odi_move_count,
            self.near_misses,
            self.policy.focus_lanes,
            self.focus_admitted,
            self.focus_rule,
            self.focus_steps,
            self.focus_retries,
            self.policy.hinge_tau,
            self.focus_reason,
            self.outcome,
        )
    }
}

// ---------------------------------------------------------------------------
// Pure step rules (unit-tested without a network)
// ---------------------------------------------------------------------------

/// `torch.sign` semantics (the calibrated recipe's): zero and NaN do not move.
fn sign0(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Box clamp that cannot panic: NaN maps to the lower bound, like
/// `ny_propagate::project_to_bounds`.
fn clamp_box(value: f32, lower: f32, upper: f32) -> f32 {
    if value.is_nan() {
        lower
    } else {
        value.max(lower).min(upper)
    }
}

/// One ODI step on one lane: `x <- clip(x + eta * 0.5 * (hi - lo) * sign(g))`.
/// Returns the number of coordinates whose stored f32 bits changed.
pub(super) fn odi_sign_step(
    point: &mut ArrayD<f32>,
    gradient: &[f32],
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
    eta: f32,
) -> usize {
    let mut moves = 0usize;
    for ((value, &g), (&lo, &hi)) in point
        .iter_mut()
        .zip(gradient.iter())
        .zip(lower.iter().zip(upper.iter()))
    {
        let before = (*value).to_bits();
        let step = eta * 0.5 * (hi - lo) * sign0(g);
        *value = clamp_box(*value + step, lo, hi);
        if (*value).to_bits() != before {
            moves = moves.saturating_add(1);
        }
    }
    moves
}

/// The recipe's cosine step, in half-range units:
/// `a_k / h = 0.15 * (1 + cos(pi * k / S)) + 0.003`.
pub(super) fn focus_step_scale(step: usize, schedule: usize) -> f32 {
    let phase = std::f32::consts::PI * step as f32 / schedule.max(1) as f32;
    0.15 * (1.0 + phase.cos()) + 0.003
}

/// One focus step on one lane: `m <- beta1*m + (1-beta1)*g`, then
/// `x <- clip(x + scale * 0.5 * (hi - lo) * sign(m))`. A hinge-held lane
/// (zero gradient) keeps coasting on its momentum, exactly as the recipe's
/// Adam does. Returns the number of coordinates that moved.
pub(super) fn focus_momentum_step(
    point: &mut ArrayD<f32>,
    momentum: &mut [f32],
    gradient: &[f32],
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
    scale: f32,
) -> usize {
    let mut moves = 0usize;
    for (((value, m), &g), (&lo, &hi)) in point
        .iter_mut()
        .zip(momentum.iter_mut())
        .zip(gradient.iter())
        .zip(lower.iter().zip(upper.iter()))
    {
        let g = if g.is_finite() { g } else { 0.0 };
        *m = FOCUS_BETA1 * *m + (1.0 - FOCUS_BETA1) * g;
        let direction = if (*m).is_finite() { sign0(*m) } else { 0.0 };
        let before = (*value).to_bits();
        *value = clamp_box(*value + scale * 0.5 * (hi - lo) * direction, lo, hi);
        if (*value).to_bits() != before {
            moves = moves.saturating_add(1);
        }
    }
    moves
}

/// The bottleneck (smallest-margin) modeled constraint of ONE clause, with
/// the same comparison rule as `best_disjunctive_target` (a NaN margin is
/// never selected).
pub(super) fn clause_bottleneck(
    output: &ArrayD<f32>,
    clause: &[OutputConstraint],
) -> Option<(f32, GraphPgdTarget)> {
    let mut margin_min = f32::INFINITY;
    let mut target_min: Option<GraphPgdTarget> = None;
    for constraint in clause {
        if let Some(target) = constraint_target(constraint) {
            let margin = target.margin(output);
            if margin < margin_min {
                margin_min = margin;
                target_min = Some(target);
            }
        }
    }
    target_min.map(|target| (margin_min, target))
}

/// Fold one evaluated output into the per-clause running max of bottleneck
/// margins (the focus wave's `wave0` ranking).
pub(super) fn observe_clause_margins(
    best: &mut [f32],
    output: &ArrayD<f32>,
    clauses: &[Vec<OutputConstraint>],
) {
    for (slot, clause) in best.iter_mut().zip(clauses.iter()) {
        if let Some((margin, _)) = clause_bottleneck(output, clause) {
            if margin > *slot {
                *slot = margin;
            }
        }
    }
}

/// The `fanout` clauses closest to violation (largest best margin first; ties
/// keep clause order). Clauses never observed with a finite margin are
/// excluded.
pub(super) fn rank_focus_clauses(best: &[f32], fanout: usize) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..best.len()).filter(|&i| best[i].is_finite()).collect();
    ranked.sort_by(|&a, &b| {
        best[b]
            .partial_cmp(&best[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(fanout);
    ranked
}

/// Observable label of a focus clause: the target class of its first
/// relational constraint (`Y_t >= Y_label` in either vnnlib spelling), else
/// `c<clause index>`.
pub(super) fn focus_clause_label(clauses: &[Vec<OutputConstraint>], index: usize) -> String {
    clauses
        .get(index)
        .and_then(|clause| {
            clause
                .iter()
                .find_map(|constraint| match constraint_target(constraint) {
                    Some(GraphPgdTarget::Relational(class, _)) => Some(class.to_string()),
                    _ => None,
                })
        })
        .unwrap_or_else(|| format!("c{index}"))
}

/// A focus lane's spec row under the tau hinge: its clause's bottleneck row
/// while the clause margin is below `tau`, a zero row (no gradient) once it
/// reaches `tau` or when the clause has no modeled constraint. The bool is
/// whether the row is live.
pub(super) fn focus_spec_row(
    output: &ArrayD<f32>,
    clause: &[OutputConstraint],
    tau: f32,
    out_dim: usize,
) -> (Vec<f32>, bool) {
    match clause_bottleneck(output, clause) {
        Some((margin, target)) if margin < tau => (
            target.to_spec_row(out_dim).row(0).iter().copied().collect(),
            true,
        ),
        _ => (vec![0.0; out_dim], false),
    }
}

/// Resident bytes one focus lane is charged: its per-lane ReLU masks (host
/// copy, wide-pass staging and fold working set, x4), six input-sized
/// vectors (point, momentum, gradient, flattened point, two transients) and
/// four output-sized ones.
pub(super) fn focus_lane_bytes(mask_floats: usize, input_dim: usize, output_dim: usize) -> u64 {
    let word = size_of::<f32>() as u64;
    let masks = (mask_floats as u64).saturating_mul(word).saturating_mul(4);
    let point = (input_dim as u64).saturating_mul(word).saturating_mul(6);
    let output = (output_dim as u64).saturating_mul(word).saturating_mul(4);
    masks.saturating_add(point).saturating_add(output).max(1)
}

/// Lanes admitted against half the observed memory headroom (or
/// [`FOCUS_UNOBSERVED_HEADROOM_CAP`] when unobservable), capped at
/// [`FOCUS_MAX_LANES`].
pub(super) fn admit_focus_lanes(
    requested: usize,
    per_lane_bytes: u64,
    headroom: Option<u64>,
) -> usize {
    let budget = headroom.map_or(FOCUS_UNOBSERVED_HEADROOM_CAP, |bytes| bytes / 2);
    let fit = usize::try_from(budget / per_lane_bytes.max(1)).unwrap_or(usize::MAX);
    requested.min(FOCUS_MAX_LANES).min(fit)
}

// ---------------------------------------------------------------------------
// Lane phases (driven from `graph_pgd_vjp_batched_disj.rs`)
// ---------------------------------------------------------------------------

/// Everything the phases share with the driver, borrowed for one invocation.
pub(super) struct DisjLaneEnv<'a> {
    pub(super) graph: &'a GraphNetwork,
    pub(super) input: &'a BoundedTensor,
    pub(super) clauses: &'a [Vec<OutputConstraint>],
    pub(super) plan: &'a PointVjpWavePlan,
    pub(super) gpu: &'a dyn GpuCrownBackward,
    pub(super) gemm_engine: Option<&'a dyn GemmEngine>,
    pub(super) deadline: Option<Instant>,
}

impl DisjLaneEnv<'_> {
    fn deadline_hit(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    fn satisfied(&self, output: &ArrayD<f32>) -> bool {
        self.clauses
            .iter()
            .any(|clause| super::check_unsafe_counterexample(output, clause))
    }
}

/// How one lane phase ended.
pub(super) enum LanePhaseEnd {
    /// The phase finished (or declined non-fatally); the lanes continue.
    Continue,
    /// A lane cleared the exact re-screen; `outcome.candidate` holds it.
    Candidate { lane: usize, step: usize },
    /// The attack deadline passed.
    Deadline,
    /// The stall monitor cut the attack.
    Stalled,
}

/// How the focus stage ended.
pub(super) enum FocusEnd {
    /// Ran to its schedule or declined; the driver finishes as before.
    Done,
    /// A lane cleared the exact re-screen; `outcome.candidate` holds it.
    Candidate { lane: usize, step: usize, odi: bool },
    /// The attack deadline passed inside the focus stage.
    Deadline,
    /// The stall monitor cut the attack.
    Stalled,
}

enum LaneScreen {
    Continue,
    Candidate,
    Stalled,
}

/// The driver's per-lane screen, verbatim: best-margin and stall bookkeeping,
/// then a template hit is re-screened on the exact/ORT-routed forward and
/// becomes a candidate only if it clears `noise_scaled_margin`.
#[allow(clippy::too_many_arguments)]
fn screen_lane(
    env: &DisjLaneEnv<'_>,
    output: &ArrayD<f32>,
    point: &ArrayD<f32>,
    outcome: &mut GraphDisjunctiveAttackOutcome,
    stall_monitor: &mut AttackStallMonitor,
    now: Option<Instant>,
    receipt: &mut DisjOdiReceipt,
) -> Result<LaneScreen> {
    if let Some(margin) = best_clause_bottleneck_margin(output, env.clauses) {
        outcome.best_margin = outcome.best_margin.max(margin);
        if let Some(now) = now {
            if stall_monitor.observe(
                margin,
                super::disjunctive_pgd::noise_scaled_margin(output),
                now,
            ) {
                return Ok(LaneScreen::Stalled);
            }
        }
    }
    if env.satisfied(output) {
        let exact = evaluate_graph(env.graph, point, env.gemm_engine)?;
        let floor = super::disjunctive_pgd::noise_scaled_margin(&exact);
        let clears_floor = env.satisfied(&exact)
            && best_clause_bottleneck_margin(&exact, env.clauses).is_some_and(|m| m >= floor);
        if clears_floor {
            outcome.candidate = Some(point.clone());
            return Ok(LaneScreen::Candidate);
        }
        receipt.near_misses = receipt.near_misses.saturating_add(1);
    }
    Ok(LaneScreen::Continue)
}

/// Run one ODI phase over `lanes` (indices into `points`): `steps` batched
/// exact-VJP passes up the fixed per-lane output `directions`
/// (`lanes.len() x output_dim`, row-major), each followed by the box-clamped
/// sign step. A forward/VJP failure ends the phase non-fatally (the lanes keep
/// their current points), like the conjunctive driver's ODI.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_odi_phase(
    env: &DisjLaneEnv<'_>,
    points: &mut [ArrayD<f32>],
    lanes: &[usize],
    directions: &[f32],
    steps: usize,
    eta: f32,
    mut clause_best: Option<&mut [f32]>,
    receipt: &mut DisjOdiReceipt,
    evidence: &ExactVjpExecution,
    outcome: &mut GraphDisjunctiveAttackOutcome,
    stall_monitor: &mut AttackStallMonitor,
) -> Result<LanePhaseEnd> {
    if steps == 0 || lanes.is_empty() {
        return Ok(LanePhaseEnd::Continue);
    }
    let out_dim = env.plan.output_dim();
    let lower = env.input.lower();
    let upper = env.input.upper();
    receipt.odi_lanes = receipt.odi_lanes.saturating_add(lanes.len());
    let mut completed = 0usize;
    for step in 0..steps {
        if env.deadline_hit() {
            note_odi_phase(receipt, completed);
            return Ok(LanePhaseEnd::Deadline);
        }
        let flat: Vec<Vec<f32>> = lanes
            .iter()
            .map(|&lane| points[lane].iter().copied().collect())
            .collect();
        let Ok((masks, outputs)) = env.plan.forward_masks(&flat) else {
            break;
        };
        if outputs.len() != lanes.len() {
            break;
        }
        let now = stall_monitor.is_armed().then(Instant::now);
        for (row, &lane) in lanes.iter().enumerate() {
            let output = ArrayD::from_shape_vec(IxDyn(&[out_dim]), outputs[row].clone())?;
            if let Some(best) = clause_best.as_deref_mut() {
                observe_clause_margins(best, &output, env.clauses);
            }
            match screen_lane(
                env,
                &output,
                &points[lane],
                outcome,
                stall_monitor,
                now,
                receipt,
            )? {
                LaneScreen::Continue => {}
                LaneScreen::Candidate => {
                    note_odi_phase(receipt, completed);
                    return Ok(LanePhaseEnd::Candidate { lane, step });
                }
                LaneScreen::Stalled => {
                    note_odi_phase(receipt, completed);
                    return Ok(LanePhaseEnd::Stalled);
                }
            }
        }
        let Ok(grads) = env.plan.gpu_vjp(env.gpu, &masks, directions) else {
            break;
        };
        if grads.len() != lanes.len() {
            break;
        }
        let mut moves = 0usize;
        for (row, &lane) in lanes.iter().enumerate() {
            moves = moves.saturating_add(odi_sign_step(
                &mut points[lane],
                &grads[row],
                lower,
                upper,
                eta,
            ));
        }
        evidence.odi_moved(moves);
        evidence.successful_vjp(true);
        receipt.odi_vjp_steps = receipt.odi_vjp_steps.saturating_add(1);
        receipt.odi_move_count = receipt.odi_move_count.saturating_add(moves);
        completed = completed.saturating_add(1);
    }
    if completed == steps {
        evidence.odi_completed_phase();
    }
    note_odi_phase(receipt, completed);
    Ok(LanePhaseEnd::Continue)
}

fn note_odi_phase(receipt: &mut DisjOdiReceipt, completed: usize) {
    if completed > 0 {
        receipt.odi_waves = receipt.odi_waves.saturating_add(1);
    }
}

/// Fixed random output direction per lane, drawn from the lane's own stream.
pub(super) fn odi_directions(rngs: &mut [SimpleRng], lanes: &[usize], out_dim: usize) -> Vec<f32> {
    let mut directions = Vec::with_capacity(lanes.len().saturating_mul(out_dim));
    for &lane in lanes {
        for _ in 0..out_dim {
            directions.push(rngs[lane].next_f32() * 2.0 - 1.0);
        }
    }
    directions
}

/// The dangerous-class focus stage. Called once, after the restart waves
/// finished without a candidate and without hitting the deadline.
///
/// `clause_best` is the per-clause running max of bottleneck margins over the
/// clean point and (when `observed_waves`) every restart-wave point.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_focus_wave(
    env: &DisjLaneEnv<'_>,
    clause_best: &[f32],
    observed_waves: bool,
    seed_offset: u64,
    receipt: &mut DisjOdiReceipt,
    evidence: &ExactVjpExecution,
    outcome: &mut GraphDisjunctiveAttackOutcome,
    stall_monitor: &mut AttackStallMonitor,
) -> Result<FocusEnd> {
    let policy = receipt.policy();
    if env.deadline_hit() {
        receipt.focus_reason = "deadline_before_start";
        return Ok(FocusEnd::Deadline);
    }
    let order = rank_focus_clauses(clause_best, FOCUS_FANOUT);
    if order.is_empty() {
        receipt.focus_reason = "no_target";
        return Ok(FocusEnd::Done);
    }
    receipt.focus_rule = if observed_waves { "wave0" } else { "clean" };
    receipt.focus_classes = order
        .iter()
        .map(|&clause| focus_clause_label(env.clauses, clause))
        .collect();

    // RSS admission: one single-lane forward prices a lane's mask footprint
    // before any lane state is allocated.
    let out_dim = env.plan.output_dim();
    let input_dim = env.plan.input_dim();
    let probe = vec![sample_center_point(env.input)
        .iter()
        .copied()
        .collect::<Vec<f32>>()];
    let Ok((probe_masks, _)) = env.plan.forward_masks(&probe) else {
        receipt.focus_reason = "forward_failure";
        return Ok(FocusEnd::Done);
    };
    let mask_floats: usize = probe_masks
        .first()
        .map_or(0, |lane| lane.iter().map(Vec::len).sum());
    let lane_bytes = focus_lane_bytes(mask_floats, input_dim, out_dim);
    let admitted = admit_focus_lanes(
        policy.focus_lanes,
        lane_bytes,
        ny_propagate::network::crown_memory::process_memory_headroom_bytes(),
    );
    receipt.focus_admitted = admitted;
    if admitted == 0 {
        receipt.focus_reason = "rss_refused";
        return Ok(FocusEnd::Done);
    }
    eprintln!(
        "[pgd-vjp-disj] focus class={} lanes={admitted} rule={} requested={} steps={} odi_steps={} tau={} lane_bytes={lane_bytes}",
        receipt.focus_classes.join(","),
        receipt.focus_rule,
        policy.focus_lanes,
        FOCUS_SCHEDULE_STEPS,
        policy.odi_steps,
        policy.hinge_tau,
    );

    // Lane init: uniform point, pinned clause round-robin over the ranked
    // set, and a fixed ODI direction from the lane's own stream.
    let mut points: Vec<ArrayD<f32>> = Vec::with_capacity(admitted);
    let mut lane_clause: Vec<usize> = Vec::with_capacity(admitted);
    let mut rngs: Vec<SimpleRng> = Vec::with_capacity(admitted);
    for lane in 0..admitted {
        let mut rng = SimpleRng::new(
            FOCUS_SEED_BASE
                .wrapping_add(seed_offset)
                .wrapping_add(lane as u64),
        );
        points.push(sample_uniform_point(env.input, &mut rng));
        lane_clause.push(order[lane % order.len()]);
        rngs.push(rng);
    }
    let all_lanes: Vec<usize> = (0..admitted).collect();
    let directions = odi_directions(&mut rngs, &all_lanes, out_dim);
    match run_odi_phase(
        env,
        &mut points,
        &all_lanes,
        &directions,
        policy.odi_steps,
        policy.odi_eta,
        None,
        receipt,
        evidence,
        outcome,
        stall_monitor,
    )? {
        LanePhaseEnd::Continue => {}
        LanePhaseEnd::Candidate { lane, step } => {
            receipt.focus_reason = "candidate";
            return Ok(FocusEnd::Candidate {
                lane,
                step,
                odi: true,
            });
        }
        LanePhaseEnd::Deadline => {
            receipt.focus_reason = "deadline";
            return Ok(FocusEnd::Deadline);
        }
        LanePhaseEnd::Stalled => {
            receipt.focus_reason = "stalled";
            return Ok(FocusEnd::Stalled);
        }
    }

    let lower = env.input.lower();
    let upper = env.input.upper();
    let mut momentum: Vec<Vec<f32>> = vec![vec![0.0; input_dim]; admitted];
    let mut step = 0usize;
    while step < FOCUS_SCHEDULE_STEPS {
        if env.deadline_hit() {
            receipt.focus_reason = "deadline";
            return Ok(FocusEnd::Deadline);
        }
        let width = points.len();
        let flat: Vec<Vec<f32>> = points
            .iter()
            .map(|point| point.iter().copied().collect())
            .collect();
        let (masks, outputs) = match env.plan.forward_masks(&flat) {
            Ok((masks, outputs)) if outputs.len() == width => (masks, outputs),
            _ => {
                receipt.focus_reason = "forward_failure";
                return Ok(FocusEnd::Done);
            }
        };
        let now = stall_monitor.is_armed().then(Instant::now);
        let mut spec_rows: Vec<f32> = Vec::with_capacity(width.saturating_mul(out_dim));
        for lane in 0..width {
            let output = ArrayD::from_shape_vec(IxDyn(&[out_dim]), outputs[lane].clone())?;
            let clause = &env.clauses[lane_clause[lane]];
            if let Some((margin, _)) = clause_bottleneck(&output, clause) {
                if margin > receipt.focus_best_margin {
                    receipt.focus_best_margin = margin;
                }
            }
            match screen_lane(
                env,
                &output,
                &points[lane],
                outcome,
                stall_monitor,
                now,
                receipt,
            )? {
                LaneScreen::Continue => {}
                LaneScreen::Candidate => {
                    receipt.focus_reason = "candidate";
                    return Ok(FocusEnd::Candidate {
                        lane,
                        step,
                        odi: false,
                    });
                }
                LaneScreen::Stalled => {
                    receipt.focus_reason = "stalled";
                    return Ok(FocusEnd::Stalled);
                }
            }
            let (row, _live) = focus_spec_row(&output, clause, policy.hinge_tau, out_dim);
            spec_rows.extend_from_slice(&row);
        }
        let grads = match env.plan.gpu_vjp(env.gpu, &masks, &spec_rows) {
            Ok(grads) if grads.len() == width => grads,
            _ => {
                // A wide-pass refusal (adapter capacity) halves the focus
                // width and retries this step; one lane that still fails ends
                // the stage without a verdict implication.
                if width <= 1 {
                    receipt.focus_reason = "vjp_failure";
                    return Ok(FocusEnd::Done);
                }
                let keep = width / 2;
                points.truncate(keep);
                momentum.truncate(keep);
                lane_clause.truncate(keep);
                receipt.focus_admitted = keep;
                receipt.focus_retries = receipt.focus_retries.saturating_add(1);
                continue;
            }
        };
        evidence.successful_vjp(false);
        receipt.focus_steps = receipt.focus_steps.saturating_add(1);
        let scale = focus_step_scale(step, FOCUS_SCHEDULE_STEPS);
        for lane in 0..width {
            focus_momentum_step(
                &mut points[lane],
                &mut momentum[lane],
                &grads[lane],
                lower,
                upper,
                scale,
            );
        }
        step += 1;
    }

    // Final check at the last iterates (one more batched forward).
    let flat: Vec<Vec<f32>> = points
        .iter()
        .map(|point| point.iter().copied().collect())
        .collect();
    if let Ok((_masks, outputs)) = env.plan.forward_masks(&flat) {
        if outputs.len() == points.len() {
            let now = stall_monitor.is_armed().then(Instant::now);
            for lane in 0..points.len() {
                let output = ArrayD::from_shape_vec(IxDyn(&[out_dim]), outputs[lane].clone())?;
                match screen_lane(
                    env,
                    &output,
                    &points[lane],
                    outcome,
                    stall_monitor,
                    now,
                    receipt,
                )? {
                    LaneScreen::Continue => {}
                    LaneScreen::Candidate => {
                        receipt.focus_reason = "candidate";
                        return Ok(FocusEnd::Candidate {
                            lane,
                            step,
                            odi: false,
                        });
                    }
                    LaneScreen::Stalled => {
                        receipt.focus_reason = "stalled";
                        return Ok(FocusEnd::Stalled);
                    }
                }
            }
        }
    }
    receipt.focus_reason = "exhausted";
    Ok(FocusEnd::Done)
}

#[cfg(test)]
#[path = "graph_pgd_vjp_disj_odi_tests.rs"]
mod tests;
