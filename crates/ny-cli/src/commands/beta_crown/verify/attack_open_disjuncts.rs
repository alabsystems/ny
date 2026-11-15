// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #attack-open-disjuncts (VIBECHECK_PLAN §6.V3): the root-pruned,
//! per-disjunct, many-restart attack stage.
//!
//! # What it does
//!
//! On a multi-clause disjunction, after the joint disjunctive attack phase and
//! before the BaB hand-off (`disjunctive::verify_multi_clause_disjunction`):
//!
//! 1. take the margin-row lane's certified ROOT per-class bounds
//!    (`ny_propagate::margin_row::root_open`), waiting at most
//!    `attack.open_disjunct_root_wait_secs` for them — and never waiting on a
//!    lane that is not running;
//! 2. drop every disjunct the root closed (certified bound strictly > 0),
//!    order the rest by root lower bound, LOWEST first (the hardest to verify
//!    is the likeliest to be violated), ties by clean-point margin;
//! 3. attack them with the exact batched VJP: restart `r` is assigned to open
//!    disjunct `r % n_attacked` and ascends that disjunct's OWN margin row
//!    (a fixed target, never re-picked), starting from a uniform point on even
//!    cycles `r / n_attacked` and from an OSI point on odd cycles (the first
//!    `pgd_osi_steps` steps ascend a random output direction, Vibecheck's
//!    `_osi_diversify`);
//! 4. give each attacked disjunct at least `attack.open_disjunct_min_secs` of
//!    the ATTACK wall (`attack.open_disjunct_max_secs`, counted from the end
//!    of the root wait — it does not include the wait). The WHOLE stage (root
//!    wait + attack) is clamped to `attack.open_disjunct_max_frac` of the time
//!    remaining at stage start and to a fixed tail reserve before the instance
//!    deadline, and the wait itself always leaves one `min_secs` floor of that
//!    limit for the attack. So the stage wall never exceeds
//!    `min(max_frac * remaining, remaining - 2 s)`, and without a deadline it
//!    is about `root_wait_secs + max_secs` (plus the plan build and one clean
//!    forward before the attack clock starts).
//!    When the open set does not fit, only the lowest-bound disjuncts that fit
//!    are attacked. The stage ends at its wall, or when
//!    `attack.open_disjunct_restarts` restarts per attacked disjunct have all
//!    run.
//!
//! This is Vibecheck's `restrict_disj` + `per_restart_disj` (cct route C) and
//! its Phase 2.6 per-open-class attack (tinyimagenet). Without a published
//! root every disjunct is open and the clean-point margin sets the order.
//!
//! # Soundness
//!
//! Attack-only. The stage changes WHERE candidates are proposed, never how one
//! is accepted. A candidate is handed up only after (a) the ORT-routed exact
//! re-screen clears `noise_scaled_margin` (`max(1e-4, 1e-5·max|y|)`, the floor
//! at `disjunctive_pgd.rs`, untouched and reused), then (b) the unchanged
//! `re_evaluate_and_confirm` funnel (attack box, per-clause box, the same
//! floor, a fresh forward) returns it — and the resulting `Violated` still goes
//! through vnncomp's trusted-ORT `gate_sat_with_trusted_oracle`. The root set
//! only decides what to attack: a certified positive root bound excludes every
//! counterexample of that disjunct, so skipping it cannot lose a real `sat`,
//! and a wrong or missing root can only aim the attack badly. Nothing here can
//! produce `unsat`.
//!
//! # Observable
//!
//! Whenever the key is armed, one fixed-schema `[attack-open]` line per
//! consultation, printed by the receipt's `Drop` on EVERY exit (see
//! [`OpenDisjunctReceipt::line`]), mirrored as a flight event
//! `attack_open_disjuncts`; `phase=entry` from vnncomp when the root
//! publication is armed, `phase=stage` from this stage — also when the route
//! gates an armed stage off (`pgd_disabled` / `demand_zeroed`). Dark = no
//! line.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use ndarray::{ArrayD, IxDyn};
use ny_core::GemmEngine;
use ny_onnx::vnnlib::OutputConstraint;
use ny_propagate::margin_row::root_open::{self, RootBoundState};
use ny_propagate::{
    BetaCrownConfig, BetaCrownResult, GraphNetwork, OpenDisjunctAttackConfig, PgdAlphaMode,
    PgdStepState, PointVjpWavePlan,
};
use ny_tensor::BoundedTensor;

use super::graph_pgd::{
    best_clause_bottleneck_margin, constraint_target, evaluate_graph,
    graph_supports_exact_gradients, sample_center_point, sample_uniform_point, GraphPgdTarget,
    SimpleRng,
};
use super::BetaCrownModel;

/// The stage never runs closer than this to the verifier's overall deadline,
/// so the BaB hand-off and verdict publication keep their tail.
const STAGE_TAIL_RESERVE: Duration = Duration::from_secs(2);
/// Poll interval while waiting for the lane's certified root.
const ROOT_POLL: Duration = Duration::from_millis(50);
/// OSI step per dimension, in units of that dimension's box width
/// (Vibecheck `_osi_diversify`: `step = 0.1 * (hi - lo)` on `sign(grad)`).
const OSI_STEP_FRAC: f32 = 0.1;
/// Seed base of the stage's per-restart RNG streams — disjoint from the joint
/// phase's `42 + r`, so the stage never replays a joint-phase start.
const STAGE_SEED_BASE: u64 = 0x0A77_AC0F_0D15_0000;

// ---------------------------------------------------------------------------
// Pure schedule (unit-tested without a network)
// ---------------------------------------------------------------------------

/// The lane's published root, decoupled from the ny-propagate type so the
/// schedule is testable on hand-written values.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RootBoundsView {
    /// True class of the lane's robustness property.
    pub(super) t: usize,
    /// Adversarial classes, aligned with `dj`.
    pub(super) adv: Vec<usize>,
    /// Certified lower bound on `Y_t - Y_{adv[k]}` over the full input box.
    pub(super) dj: Vec<f64>,
}

/// One disjunct the stage may attack.
#[derive(Clone, Copy)]
pub(super) struct OpenDisjunct {
    /// Index into the property's clause list.
    pub(super) clause: usize,
    /// Certified root lower bound on this disjunct's SAFETY margin; `-inf`
    /// when no root was published or the clause is not a lane class.
    pub(super) root_lb: f64,
    /// Clause margin at the clean point (`>= 0` would be a violation).
    pub(super) clean_margin: f32,
    /// The margin whose ascent drives toward satisfying the clause.
    pub(super) target: GraphPgdTarget,
}

/// The open set in attack order plus the census the receipt reports.
pub(super) struct OpenSet {
    pub(super) open: Vec<OpenDisjunct>,
    /// Clauses the certified root closed (skipped).
    pub(super) closed: usize,
    /// Targetable clauses a published root could not be mapped onto
    /// (kept OPEN, ordered first: an unknown bound is never a closed one).
    pub(super) unmapped: usize,
    /// Clauses that are not a single modeled constraint (left to the joint
    /// phase and BaB).
    pub(super) untargetable: usize,
    pub(super) total: usize,
}

/// The single fixed ascent target of a one-constraint clause.
fn single_target(clause: &[OutputConstraint]) -> Option<GraphPgdTarget> {
    match clause {
        [only] => constraint_target(only),
        _ => None,
    }
}

/// The certified root bound of the disjunct `Y_j >= Y_t` (in any of its
/// spellings, all normalized by `constraint_target` to `Relational(j, t)`).
fn root_lb_for(root: &RootBoundsView, target: GraphPgdTarget) -> Option<f64> {
    let GraphPgdTarget::Relational(j, t) = target else {
        return None;
    };
    if t != root.t {
        return None;
    }
    root.adv
        .iter()
        .position(|&class| class == j)
        .and_then(|k| root.dj.get(k).copied())
}

/// Build the open set: closed disjuncts dropped (the margin-row lane's own
/// strict closure predicate: finite and `> 0`), the rest sorted by root lower
/// bound ascending, then clean margin descending, then clause index.
pub(super) fn open_disjuncts(
    clauses: &[Vec<OutputConstraint>],
    root: Option<&RootBoundsView>,
    clean_margins: &[f32],
) -> OpenSet {
    let root = root.filter(|r| r.adv.len() == r.dj.len());
    let mut open = Vec::with_capacity(clauses.len());
    let (mut closed, mut unmapped, mut untargetable) = (0usize, 0usize, 0usize);
    for (clause, constraints) in clauses.iter().enumerate() {
        let Some(target) = single_target(constraints) else {
            untargetable += 1;
            continue;
        };
        let clean_margin = clean_margins
            .get(clause)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        let root_lb = match root.map(|r| root_lb_for(r, target)) {
            Some(Some(lb)) if lb.is_finite() && lb > 0.0 => {
                closed += 1;
                continue;
            }
            Some(Some(lb)) => lb,
            Some(None) => {
                unmapped += 1;
                f64::NEG_INFINITY
            }
            None => f64::NEG_INFINITY,
        };
        open.push(OpenDisjunct {
            clause,
            root_lb,
            clean_margin,
            target,
        });
    }
    open.sort_by(|a, b| {
        a.root_lb
            .total_cmp(&b.root_lb)
            .then(b.clean_margin.total_cmp(&a.clean_margin))
            .then(a.clause.cmp(&b.clause))
    });
    OpenSet {
        open,
        closed,
        unmapped,
        untargetable,
        total: clauses.len(),
    }
}

/// How much of the open set the stage attacks, for how long.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct StagePlan {
    /// The first `attacked` disjuncts of the ordered open set.
    pub(super) attacked: usize,
    /// Stage wall, seconds; `stage_secs / attacked >= min_secs`.
    pub(super) stage_secs: f64,
    /// `attacked * restarts`.
    pub(super) total_restarts: usize,
}

/// Size the stage. `remaining_secs` is the time left until the stage's hard
/// limit (instance deadline minus the tail reserve).
pub(super) fn plan_stage(
    n_open: usize,
    policy: &OpenDisjunctAttackConfig,
    remaining_secs: f64,
) -> Result<StagePlan, &'static str> {
    if n_open == 0 {
        return Err("all_closed");
    }
    let min_secs = f64::from(policy.min_secs);
    let available = f64::from(policy.max_secs).min(remaining_secs.max(0.0));
    if !(min_secs > 0.0 && available >= min_secs) {
        return Err("no_budget");
    }
    let fit = (available / min_secs).floor();
    let attacked = if fit >= n_open as f64 {
        n_open
    } else {
        (fit as usize).max(1)
    };
    Ok(StagePlan {
        attacked,
        stage_secs: available,
        total_restarts: attacked.saturating_mul(policy.restarts.max(1)),
    })
}

/// Restart `r`'s disjunct slot in the ordered, attacked prefix.
pub(super) fn restart_slot(r: usize, attacked: usize) -> usize {
    r % attacked.max(1)
}

/// Start kind of a restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InitKind {
    /// Uniform point in the box.
    Uniform,
    /// Uniform point, then `osi_steps` sign steps along a random output
    /// direction before the disjunct's own ascent.
    Osi,
}

/// Even cycles start uniform, odd cycles start OSI (never OSI when the
/// configured OSI step count is zero).
pub(super) fn restart_init(r: usize, attacked: usize, osi_steps: usize) -> InitKind {
    if osi_steps > 0 && (r / attacked.max(1)) % 2 == 1 {
        InitKind::Osi
    } else {
        InitKind::Uniform
    }
}

/// The stage's hard limit, covering the root wait AND the attack: the
/// earlier of (a) the verifier's overall deadline minus the tail reserve and
/// (b) `now + max_frac * (deadline - now)`. Never in the past (`now` means no
/// budget). No deadline, no limit.
pub(super) fn stage_hard_limit(
    now: Instant,
    overall_deadline: Option<Instant>,
    max_frac: f32,
) -> Option<Instant> {
    let deadline = overall_deadline?;
    let reserve_limit = deadline
        .checked_sub(STAGE_TAIL_RESERVE)
        .map_or(now, |limit| limit.max(now));
    let share = deadline
        .saturating_duration_since(now)
        .mul_f64(f64::from(max_frac).clamp(0.0, 1.0));
    let frac_limit = now.checked_add(share).unwrap_or(deadline);
    Some(reserve_limit.min(frac_limit))
}

/// When the root wait gives up: the earlier of `stage_start +
/// root_wait_secs` and the stage's hard limit LESS one disjunct's attack
/// floor (`min_secs`). Without that floor a lane that stays `Pending` could
/// hold the wait to the hard limit itself, and the stage would spend its whole
/// share of the clock waiting and then decline `no_budget` — seconds taken
/// from BaB that bought no attack at all. Never before `stage_start`.
pub(super) fn root_wait_until(
    stage_start: Instant,
    hard_limit: Option<Instant>,
    policy: &OpenDisjunctAttackConfig,
) -> Instant {
    // `validate` capped both keys, so these conversions cannot fail; a failure
    // would mean "do not wait" / "reserve no floor".
    let wait_cap = Duration::try_from_secs_f32(policy.root_wait_secs)
        .ok()
        .and_then(|wait| stage_start.checked_add(wait))
        .unwrap_or(stage_start);
    let attack_floor = Duration::try_from_secs_f32(policy.min_secs).unwrap_or(Duration::ZERO);
    hard_limit.map_or(wait_cap, |limit| {
        let latest = limit
            .checked_sub(attack_floor)
            .map_or(stage_start, |at| at.max(stage_start));
        wait_cap.min(latest)
    })
}

/// Outcome of waiting for the lane's root.
pub(super) struct RootWait {
    pub(super) view: Option<RootBoundsView>,
    pub(super) token: &'static str,
    pub(super) first_published_at: Option<Instant>,
    pub(super) publications: usize,
}

/// Wait for the published root until `wait_until`. Never waits on `Idle` (no
/// armed lane is running) or `Abandoned` (it returned without a root); a
/// `Pending` lane is polled until the root lands or `wait_until` passes.
pub(super) fn wait_for_root(
    wait_until: Instant,
    mut state: impl FnMut() -> RootBoundState,
    mut sleep: impl FnMut(Duration),
) -> RootWait {
    let none = |token: &'static str| RootWait {
        view: None,
        token,
        first_published_at: None,
        publications: 0,
    };
    loop {
        match state() {
            RootBoundState::Published(published) => {
                return RootWait {
                    view: Some(RootBoundsView {
                        t: published.t,
                        adv: published.adv.clone(),
                        dj: published.dj.clone(),
                    }),
                    token: "lane",
                    first_published_at: Some(published.first_published_at),
                    publications: published.publications,
                };
            }
            RootBoundState::Idle => return none("none_idle"),
            RootBoundState::Abandoned => return none("none_abandoned"),
            RootBoundState::Pending => {
                let now = Instant::now();
                if now >= wait_until {
                    return none("none_pending_timeout");
                }
                sleep(ROOT_POLL.min(wait_until - now));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Receipt
// ---------------------------------------------------------------------------

/// The `[attack-open]` receipt. Every field is always printed, in this order,
/// by `Drop` — so every exit of an armed consultation (declines, errors
/// propagated with `?`, deadline, exhaustion, candidate) leaves exactly one
/// line, and a missing line means the stage was never consulted.
pub(super) struct OpenDisjunctReceipt {
    phase: &'static str,
    policy: OpenDisjunctAttackConfig,
    reason: &'static str,
    root: &'static str,
    root_wait_ms: u64,
    root_age_ms: i64,
    root_publications: usize,
    open: usize,
    total: usize,
    closed: usize,
    unmapped: usize,
    untargetable: usize,
    attacked: usize,
    restarts_started: usize,
    restarts_planned: usize,
    waves: usize,
    osi_lanes: usize,
    near_misses: usize,
    rejected: usize,
    hit_clause: Option<usize>,
    hit_class: Option<usize>,
    best_margin: f32,
    started: Instant,
}

impl OpenDisjunctReceipt {
    fn new(phase: &'static str, policy: OpenDisjunctAttackConfig) -> Self {
        Self {
            phase,
            policy,
            // Replaced on every normal exit; survives only an early `?`.
            reason: "error",
            root: "n/a",
            root_wait_ms: 0,
            root_age_ms: -1,
            root_publications: 0,
            open: 0,
            total: 0,
            closed: 0,
            unmapped: 0,
            untargetable: 0,
            attacked: 0,
            restarts_started: 0,
            restarts_planned: 0,
            waves: 0,
            osi_lanes: 0,
            near_misses: 0,
            rejected: 0,
            hit_clause: None,
            hit_class: None,
            best_margin: f32::NEG_INFINITY,
            started: Instant::now(),
        }
    }

    fn observe_margin(&mut self, margin: f32) {
        if margin.is_finite() {
            self.best_margin = self.best_margin.max(margin);
        }
    }

    /// The fixed schema. Tokens: `reason` is one of `root_publication_armed`
    /// (entry), `pgd_disabled`, `demand_zeroed` (gated off by the route, see
    /// [`note_stage_declined`]), `not_armed`, `per_clause_boxes`,
    /// `no_exact_gradient`, `lowering_failed`, `gpu_unavailable`,
    /// `batch_disabled`, `unsupported_plan`, `untargetable` (no clause is a
    /// single modeled constraint), `all_closed` (every targetable disjunct was
    /// root-closed), `no_budget`, `forward_failure`, `vjp_failure`, `deadline`,
    /// `exhausted`, `candidate`, `error`; `root` is `lane`, `none_idle`,
    /// `none_abandoned`, `none_pending_timeout` or `n/a` (not reached);
    /// `hit`/`hit_class` are a clause index / output class or `none`; `margin`
    /// is the best clause bottleneck margin the stage saw (`nan` when none).
    fn line(&self) -> String {
        let opt = |v: Option<usize>| v.map_or_else(|| "none".to_string(), |v| v.to_string());
        let margin = if self.best_margin.is_finite() {
            format!("{:+.6e}", self.best_margin)
        } else {
            "nan".to_string()
        };
        format!(
            "[attack-open] armed=1 phase={} reason={} root={} root_wait_ms={} root_age_ms={} \
             root_publications={} open={}/{} closed={} unmapped={} untargetable={} attacked={} \
             restarts={}/{} per_disjunct={} min_secs={:.3} max_secs={:.3} root_wait_secs={:.3} \
             max_frac={:.3} waves={} osi_lanes={} near_misses={} rejected={} hit={} hit_class={} \
             margin={} secs={:.3}",
            self.phase,
            self.reason,
            self.root,
            self.root_wait_ms,
            self.root_age_ms,
            self.root_publications,
            self.open,
            self.total,
            self.closed,
            self.unmapped,
            self.untargetable,
            self.attacked,
            self.restarts_started,
            self.restarts_planned,
            self.policy.restarts,
            self.policy.min_secs,
            self.policy.max_secs,
            self.policy.root_wait_secs,
            self.policy.max_frac,
            self.waves,
            self.osi_lanes,
            self.near_misses,
            self.rejected,
            opt(self.hit_clause),
            opt(self.hit_class),
            margin,
            self.started.elapsed().as_secs_f64(),
        )
    }
}

impl Drop for OpenDisjunctReceipt {
    fn drop(&mut self) {
        let line = self.line();
        eprintln!("{line}");
        // The same line as a flight event: `ny benchmarks run` nulls stdout
        // and keeps no stderr, so without this a sweep could never tell an
        // engaged stage from a dark binary.
        crate::flight::note(
            "attack_open_disjuncts",
            crate::flight::FlightStatus::Ran,
            Some(line),
        );
    }
}

/// vnncomp entry hook: arm the margin-row lane's root publication and print
/// the `phase=entry` receipt. Called before any lane can start, only with a
/// policy the preset resolved; a dark policy (`restrict: false`) does nothing.
pub(in crate::commands::beta_crown) fn arm_open_disjunct_attack(policy: OpenDisjunctAttackConfig) {
    if !policy.restrict {
        return;
    }
    root_open::arm_root_bound_publication();
    let mut receipt = OpenDisjunctReceipt::new("entry", policy);
    receipt.reason = "root_publication_armed";
}

// ---------------------------------------------------------------------------
// The stage
// ---------------------------------------------------------------------------

/// Everything the stage reads; borrowed from `verify_multi_clause_disjunction`.
pub(super) struct StageInputs<'a> {
    pub(super) model_net: &'a BetaCrownModel,
    pub(super) input: &'a BoundedTensor,
    pub(super) clauses: &'a [Vec<OutputConstraint>],
    pub(super) per_clause_input_bounds: &'a [BTreeMap<usize, (f64, f64)>],
    pub(super) config: &'a BetaCrownConfig,
    pub(super) pgd_steps: usize,
    pub(super) attack_engine: Option<&'a dyn GemmEngine>,
    pub(super) overall_deadline: Option<Instant>,
}

fn property_violated(output: &ArrayD<f32>, clauses: &[Vec<OutputConstraint>]) -> bool {
    clauses
        .iter()
        .any(|clause| super::check_unsafe_counterexample(output, clause))
}

/// The ONLY way this stage produces a result: the ORT-routed exact re-screen
/// against the untouched noise floor, then the unchanged
/// `re_evaluate_and_confirm` funnel. `None` means "keep attacking".
fn screen_and_confirm(
    inputs: &StageInputs<'_>,
    graph: &GraphNetwork,
    point: &ArrayD<f32>,
    receipt: &mut OpenDisjunctReceipt,
) -> Result<Option<BetaCrownResult>> {
    let exact = evaluate_graph(graph, point, inputs.attack_engine)?;
    let floor = super::disjunctive_pgd::noise_scaled_margin(&exact);
    let clears_floor = property_violated(&exact, inputs.clauses)
        && best_clause_bottleneck_margin(&exact, inputs.clauses).is_some_and(|m| m >= floor);
    if !clears_floor {
        receipt.near_misses += 1;
        return Ok(None);
    }
    match super::disjunctive_pgd::re_evaluate_and_confirm(
        inputs.model_net,
        point,
        inputs.clauses,
        inputs.per_clause_input_bounds,
        inputs.input,
        inputs.attack_engine,
    )? {
        Some((witness, output)) => {
            receipt.hit_clause = inputs
                .clauses
                .iter()
                .position(|clause| super::check_unsafe_counterexample(&output, clause));
            receipt.hit_class = receipt
                .hit_clause
                .and_then(|clause| single_target(&inputs.clauses[clause]))
                .and_then(|target| match target {
                    GraphPgdTarget::Relational(j, _) => Some(j),
                    GraphPgdTarget::Constant(..) | GraphPgdTarget::NegConstant(..) => None,
                });
            if let Some(margin) = best_clause_bottleneck_margin(&output, inputs.clauses) {
                receipt.observe_margin(margin);
            }
            receipt.reason = "candidate";
            Ok(Some(super::disjunctive_pgd::make_violated_result(
                witness, output,
            )))
        }
        None => {
            receipt.rejected += 1;
            Ok(None)
        }
    }
}

fn sign_of(gradient: &ArrayD<f32>) -> ArrayD<f32> {
    gradient.mapv(|g| {
        if g > 0.0 {
            1.0
        } else if g < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

/// The receipt of an ARMED stage that the multi-clause route gates off before
/// consulting it; `None` for a dark policy (no line).
fn declined_receipt(
    policy: OpenDisjunctAttackConfig,
    reason: &'static str,
) -> Option<OpenDisjunctReceipt> {
    policy.restrict.then(|| {
        let mut receipt = OpenDisjunctReceipt::new("stage", policy);
        receipt.reason = reason;
        receipt
    })
}

/// Print the `phase=stage` receipt for an ARMED stage that the multi-clause
/// route skips before consulting it (`pgd_disabled`: the attack phases are
/// off for this run; `demand_zeroed`: the net was statically demand-zeroed out
/// of the exact-gradient fragment). Without this line such a row reads exactly
/// like one an earlier lane decided. A dark policy prints nothing.
pub(super) fn note_stage_declined(policy: OpenDisjunctAttackConfig, reason: &'static str) {
    drop(declined_receipt(policy, reason));
}

/// Run the stage. `Ok(Some(_))` is a confirmed `Violated` result (still
/// subject to vnncomp's trusted-ORT gate); `Ok(None)` hands off to BaB with
/// no verdict implied. The caller checks `open_disjunct_attack.restrict`, so
/// the dark path never enters here.
pub(super) fn run_open_disjunct_attack_stage(
    inputs: &StageInputs<'_>,
) -> Result<Option<BetaCrownResult>> {
    // #phase-cost: see `ny_core::phase_cost` (observability only).
    let _phase_cost = ny_core::phase_cost::enter(ny_core::phase_cost::CostPhase::Attack);
    let mut receipt = OpenDisjunctReceipt::new("stage", inputs.config.open_disjunct_attack);
    run_stage(inputs, &mut receipt)
}

/// The stage body. Every exit records its reason in `receipt`, which prints
/// when the caller drops it — including after an early `?` (reason `error`).
fn run_stage(
    inputs: &StageInputs<'_>,
    receipt: &mut OpenDisjunctReceipt,
) -> Result<Option<BetaCrownResult>> {
    let policy = inputs.config.open_disjunct_attack;
    if !policy.restrict || policy.validate().is_err() {
        receipt.reason = "not_armed";
        return Ok(None);
    }
    let stage_start = Instant::now();
    let hard_limit = stage_hard_limit(stage_start, inputs.overall_deadline, policy.max_frac);
    let clauses = inputs.clauses;
    let input = inputs.input;
    // Not even one disjunct's floor fits: decline before paying for the
    // lowering and the exact-VJP plan build.
    if hard_limit.is_some_and(|limit| {
        limit.saturating_duration_since(stage_start).as_secs_f64() < f64::from(policy.min_secs)
    }) {
        receipt.reason = "no_budget";
        return Ok(None);
    }

    if inputs
        .per_clause_input_bounds
        .iter()
        .any(|clause_box| !clause_box.is_empty())
    {
        receipt.reason = "per_clause_boxes";
        return Ok(None);
    }
    let lowered;
    let graph: &GraphNetwork = match inputs.model_net {
        BetaCrownModel::Graph(graph) => graph,
        BetaCrownModel::Sequential(network) => match GraphNetwork::from_sequential(network) {
            Ok(graph) => {
                lowered = graph;
                &lowered
            }
            Err(_) => {
                receipt.reason = "lowering_failed";
                return Ok(None);
            }
        },
    };
    if !graph_supports_exact_gradients(graph) {
        receipt.reason = "no_exact_gradient";
        return Ok(None);
    }
    let Some(gpu) = inputs
        .attack_engine
        .and_then(|engine| engine.as_gpu_crown_backward())
    else {
        receipt.reason = "gpu_unavailable";
        return Ok(None);
    };
    if !inputs.config.pgd_exact_vjp.batch_enabled {
        receipt.reason = "batch_disabled";
        return Ok(None);
    }
    let Some(plan) = PointVjpWavePlan::build(graph, input) else {
        receipt.reason = "unsupported_plan";
        return Ok(None);
    };
    let out_dim = plan.output_dim();

    // (1) The lane's certified root, bounded wait that always leaves one
    // disjunct's floor of the hard limit for the attack itself.
    let wait_until = root_wait_until(stage_start, hard_limit, &policy);
    let wait_start = Instant::now();
    let root = wait_for_root(wait_until, root_open::root_bound_state, std::thread::sleep);
    receipt.root = root.token;
    // The wait alone (the lowering and plan build before it are in `secs`).
    receipt.root_wait_ms = u64::try_from(wait_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    receipt.root_publications = root.publications;
    receipt.root_age_ms = root.first_published_at.map_or(-1, |at| {
        i64::try_from(at.elapsed().as_millis()).unwrap_or(i64::MAX)
    });

    // (2) Clean point: ordering tie-break, and a free candidate check.
    let clean = sample_center_point(input);
    let clean_out = evaluate_graph(graph, &clean, inputs.attack_engine)?;
    if let Some(margin) = best_clause_bottleneck_margin(&clean_out, clauses) {
        receipt.observe_margin(margin);
    }
    if property_violated(&clean_out, clauses) {
        if let Some(result) = screen_and_confirm(inputs, graph, &clean, receipt)? {
            return Ok(Some(result));
        }
    }
    let clean_margins: Vec<f32> = clauses
        .iter()
        .map(|clause| single_target(clause).map_or(f32::NEG_INFINITY, |t| t.margin(&clean_out)))
        .collect();
    let set = open_disjuncts(clauses, root.view.as_ref(), &clean_margins);
    receipt.open = set.open.len();
    receipt.total = set.total;
    receipt.closed = set.closed;
    receipt.unmapped = set.unmapped;
    receipt.untargetable = set.untargetable;
    if set.open.is_empty() && set.closed == 0 {
        // No clause is a single modeled constraint: nothing to aim at, and
        // NOT a root verdict — `all_closed` would misread as one.
        receipt.reason = "untargetable";
        return Ok(None);
    }

    // (3) Size the stage against its hard limit.
    let now = Instant::now();
    let remaining_secs = hard_limit.map_or(f64::INFINITY, |limit| {
        limit.saturating_duration_since(now).as_secs_f64()
    });
    let stage = match plan_stage(set.open.len(), &policy, remaining_secs) {
        Ok(stage) => stage,
        Err(reason) => {
            receipt.reason = reason;
            return Ok(None);
        }
    };
    receipt.attacked = stage.attacked;
    receipt.restarts_planned = stage.total_restarts;
    // `stage_secs <= max_secs <= MAX_SECS_CAP`, so this cannot fail; if it
    // ever did, the stage would stop before its first wave.
    let stage_deadline = Duration::try_from_secs_f64(stage.stage_secs)
        .ok()
        .and_then(|wall| now.checked_add(wall))
        .unwrap_or(now);
    let attacked = &set.open[..stage.attacked];

    // (4) Round-robin waves on the exact batched VJP.
    let steps = inputs.pgd_steps.max(10);
    let pgd = super::disjunctive_pgd::beta_crown_pgd_config(
        inputs.config,
        stage.total_restarts,
        steps,
        Some(stage_deadline),
    );
    let osi_steps = pgd.osi_steps.min(steps / 2);
    let width = pgd.exact_vjp.disjunctive_batch_width().max(1);
    let past_deadline = || Instant::now() >= stage_deadline;

    let mut restart_base = 0usize;
    while restart_base < stage.total_restarts {
        if past_deadline() {
            receipt.reason = "deadline";
            return Ok(None);
        }
        let k = width.min(stage.total_restarts - restart_base);
        let mut rngs = Vec::with_capacity(k);
        let mut xs = Vec::with_capacity(k);
        let mut targets = Vec::with_capacity(k);
        let mut osi_rows: Vec<Option<Vec<f32>>> = Vec::with_capacity(k);
        for i in 0..k {
            let r = restart_base + i;
            let mut rng = SimpleRng::new(STAGE_SEED_BASE.wrapping_add(r as u64));
            xs.push(sample_uniform_point(input, &mut rng));
            targets.push(attacked[restart_slot(r, attacked.len())].target);
            osi_rows.push(
                (restart_init(r, attacked.len(), osi_steps) == InitKind::Osi).then(|| {
                    (0..out_dim)
                        .map(|_| rng.next_f32() * 2.0 - 1.0)
                        .collect::<Vec<f32>>()
                }),
            );
            rngs.push(rng);
        }
        receipt.restarts_started += k;
        receipt.waves += 1;
        receipt.osi_lanes += osi_rows.iter().filter(|row| row.is_some()).count();
        let mut states: Vec<PgdStepState> = (0..k)
            .map(|_| {
                PgdStepState::from_config(
                    pgd.optimizer,
                    pgd.alpha_mode,
                    pgd.step_size,
                    pgd.adam,
                    input,
                    input.shape(),
                )
            })
            .collect();
        let mut osi_states: Vec<PgdStepState> = (0..k)
            .map(|_| {
                PgdStepState::new_signed_gradient(
                    PgdAlphaMode::InputRangeScaled(OSI_STEP_FRAC),
                    pgd.step_size,
                    input,
                )
            })
            .collect();

        for step in 0..steps {
            if past_deadline() {
                receipt.reason = "deadline";
                return Ok(None);
            }
            let points: Vec<Vec<f32>> = xs.iter().map(|x| x.iter().copied().collect()).collect();
            let Some((masks, outputs)) = plan
                .forward_masks(&points)
                .ok()
                .filter(|(_, outputs)| outputs.len() == k)
            else {
                receipt.reason = "forward_failure";
                return Ok(None);
            };
            let mut spec_rows: Vec<f32> = Vec::with_capacity(k * out_dim);
            for kk in 0..k {
                let out = ArrayD::from_shape_vec(IxDyn(&[out_dim]), outputs[kk].clone())?;
                if let Some(margin) = best_clause_bottleneck_margin(&out, clauses) {
                    receipt.observe_margin(margin);
                }
                if property_violated(&out, clauses) {
                    if let Some(result) = screen_and_confirm(inputs, graph, &xs[kk], receipt)? {
                        return Ok(Some(result));
                    }
                }
                match (&osi_rows[kk], step < osi_steps) {
                    (Some(direction), true) => spec_rows.extend_from_slice(direction),
                    _ => spec_rows.extend(targets[kk].to_spec_row(out_dim).row(0).iter().copied()),
                }
            }
            let grads = match plan.gpu_vjp(gpu, &masks, &spec_rows) {
                Ok(grads) if grads.len() == k => grads,
                _ => {
                    receipt.reason = "vjp_failure";
                    return Ok(None);
                }
            };
            for kk in 0..k {
                let grad = ArrayD::from_shape_vec(IxDyn(xs[kk].shape()), grads[kk].clone())?;
                if osi_rows[kk].is_some() && step < osi_steps {
                    xs[kk] = osi_states[kk].step(&sign_of(&grad), &xs[kk], input, true);
                    continue;
                }
                let previous = pgd.restart_when_stuck.then(|| xs[kk].clone());
                xs[kk] = states[kk].step(&grad, &xs[kk], input, true);
                if let Some(previous) = previous {
                    if previous.iter().zip(xs[kk].iter()).all(|(&a, &b)| a == b) {
                        xs[kk] = sample_uniform_point(input, &mut rngs[kk]);
                        states[kk].reset();
                    }
                }
            }
        }

        // Post-wave check at the final iterates.
        let points: Vec<Vec<f32>> = xs.iter().map(|x| x.iter().copied().collect()).collect();
        if let Some((_masks, outputs)) = plan
            .forward_masks(&points)
            .ok()
            .filter(|(_, outputs)| outputs.len() == k)
        {
            for kk in 0..k {
                let out = ArrayD::from_shape_vec(IxDyn(&[out_dim]), outputs[kk].clone())?;
                if let Some(margin) = best_clause_bottleneck_margin(&out, clauses) {
                    receipt.observe_margin(margin);
                }
                if property_violated(&out, clauses) {
                    if let Some(result) = screen_and_confirm(inputs, graph, &xs[kk], receipt)? {
                        return Ok(Some(result));
                    }
                }
            }
        }
        restart_base += k;
    }
    receipt.reason = "exhausted";
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ny_propagate::layers::LinearLayer;
    use ny_propagate::{BabVerificationStatus, Layer, Network};
    use std::sync::Arc;

    fn ge(j: usize, t: usize) -> Vec<OutputConstraint> {
        vec![OutputConstraint::GreaterEq(j, t)]
    }

    fn le(t: usize, j: usize) -> Vec<OutputConstraint> {
        vec![OutputConstraint::LessEq(t, j)]
    }

    fn clauses_of(order: &OpenSet) -> Vec<usize> {
        order.open.iter().map(|d| d.clause).collect()
    }

    fn policy(min_secs: f32, max_secs: f32, restarts: usize) -> OpenDisjunctAttackConfig {
        OpenDisjunctAttackConfig {
            restrict: true,
            min_secs,
            max_secs,
            restarts,
            root_wait_secs: 0.0,
            max_frac: 1.0,
        }
    }

    #[test]
    fn open_disjunct_attack_skips_root_closed_disjuncts() {
        // True class 0; clauses for classes 1..=5, both vnnlib spellings.
        let clauses = vec![ge(1, 0), le(0, 2), ge(3, 0), le(0, 4), ge(5, 0)];
        let root = RootBoundsView {
            t: 0,
            adv: vec![1, 2, 3, 4],
            // class 1 closed, 2 open (-0.30), 3 closed, 4 open at EXACTLY 0.0
            // (the lane's closure is strict: margin 0 admits a sat point).
            dj: vec![0.25, -0.30, 1e-9, 0.0],
        };
        let set = open_disjuncts(&clauses, Some(&root), &[0.0; 5]);
        assert_eq!(set.closed, 2, "classes 1 and 3 are certified closed");
        assert_eq!(set.total, 5);
        assert_eq!(set.unmapped, 1, "class 5 is not a lane class");
        assert_eq!(set.untargetable, 0);
        // Unmapped (-inf) first, then lowest root bound first.
        assert_eq!(clauses_of(&set), vec![4, 1, 3]);
        assert!(set.open.iter().all(|d| d.clause != 0 && d.clause != 2));

        // A root for another true class maps nothing: everything stays open.
        let foreign = RootBoundsView {
            t: 7,
            ..root.clone()
        };
        let set = open_disjuncts(&clauses, Some(&foreign), &[0.0; 5]);
        assert_eq!((set.closed, set.unmapped, set.open.len()), (0, 5, 5));

        // A misaligned root is ignored rather than trusted.
        let misaligned = RootBoundsView {
            dj: vec![1.0],
            ..root.clone()
        };
        let set = open_disjuncts(&clauses, Some(&misaligned), &[0.0; 5]);
        assert_eq!((set.closed, set.unmapped, set.open.len()), (0, 0, 5));

        // Multi-constraint clauses are left to the joint phase.
        let conj = vec![
            vec![
                OutputConstraint::GreaterEq(1, 0),
                OutputConstraint::GreaterEq(2, 0),
            ],
            ge(3, 0),
        ];
        let set = open_disjuncts(&conj, Some(&root), &[0.0; 2]);
        assert_eq!((set.untargetable, set.closed, set.open.len()), (1, 1, 0));
    }

    #[test]
    fn open_disjunct_attack_orders_by_clean_margin_without_a_root() {
        let clauses = vec![ge(1, 0), ge(2, 0), ge(3, 0)];
        let set = open_disjuncts(&clauses, None, &[-3.0, -0.5, -1.0]);
        assert_eq!((set.closed, set.unmapped), (0, 0));
        // Closest to violation first.
        assert_eq!(clauses_of(&set), vec![1, 2, 0]);
        assert!(set.open.iter().all(|d| d.root_lb == f64::NEG_INFINITY));
    }

    #[test]
    fn open_disjunct_attack_assigns_round_robin() {
        let n = 3;
        let mut per_slot = [0usize; 3];
        for r in 0..(4 * n) {
            let slot = restart_slot(r, n);
            assert_eq!(slot, r % n);
            per_slot[slot] += 1;
        }
        assert_eq!(
            per_slot,
            [4, 4, 4],
            "every attacked disjunct gets equal restarts"
        );
        // Slot 0 is the lowest root bound: restart 0 attacks the hardest one.
        let clauses = vec![ge(1, 0), ge(2, 0), ge(3, 0)];
        let root = RootBoundsView {
            t: 0,
            adv: vec![1, 2, 3],
            dj: vec![-0.1, -0.9, -0.4],
        };
        let set = open_disjuncts(&clauses, Some(&root), &[0.0; 3]);
        assert_eq!(set.open[restart_slot(0, n)].clause, 1);
        assert_eq!(set.open[restart_slot(1, n)].clause, 2);
        assert_eq!(set.open[restart_slot(2, n)].clause, 0);
        assert_eq!(set.open[restart_slot(3, n)].clause, 1);
        // Even cycles uniform, odd cycles OSI; never OSI with zero OSI steps.
        let kinds: Vec<InitKind> = (0..(4 * n)).map(|r| restart_init(r, n, 20)).collect();
        for (r, kind) in kinds.iter().enumerate() {
            let expect = if (r / n) % 2 == 1 {
                InitKind::Osi
            } else {
                InitKind::Uniform
            };
            assert_eq!(*kind, expect, "restart {r}");
        }
        assert!((0..(4 * n)).all(|r| restart_init(r, n, 0) == InitKind::Uniform));
        // A degenerate attacked count cannot divide by zero.
        assert_eq!(restart_slot(5, 0), 0);
    }

    #[test]
    fn open_disjunct_attack_respects_min_secs_and_deadline() {
        // Everything fits: all open disjuncts, the whole max_secs wall.
        let plan = plan_stage(4, &policy(0.5, 10.0, 100), 60.0).unwrap();
        assert_eq!(plan.attacked, 4);
        assert_eq!(plan.stage_secs, 10.0);
        assert_eq!(plan.total_restarts, 400);
        // The deadline binds before max_secs; the open set is truncated so
        // that NO attacked disjunct gets less than min_secs.
        let plan = plan_stage(40, &policy(0.5, 20.0, 100), 3.0).unwrap();
        assert_eq!(plan.stage_secs, 3.0);
        assert_eq!(plan.attacked, 6);
        assert!(plan.stage_secs / plan.attacked as f64 >= 0.5);
        // max_secs binds.
        let plan = plan_stage(40, &policy(0.5, 2.0, 7), 50.0).unwrap();
        assert_eq!(
            (plan.attacked, plan.stage_secs, plan.total_restarts),
            (4, 2.0, 28)
        );
        // Not even one disjunct's floor fits: decline, spend nothing.
        assert_eq!(
            plan_stage(3, &policy(0.5, 10.0, 100), 0.25),
            Err("no_budget")
        );
        assert_eq!(
            plan_stage(3, &policy(0.5, 0.25, 100), 99.0),
            Err("no_budget")
        );
        assert_eq!(
            plan_stage(3, &policy(0.5, 10.0, 100), -1.0),
            Err("no_budget")
        );
        assert_eq!(
            plan_stage(0, &policy(0.5, 10.0, 100), 99.0),
            Err("all_closed")
        );

        // The hard limit keeps the tail reserve, caps the whole stage at
        // `max_frac` of what remains, and never lies in the past.
        let now = Instant::now();
        let deadline = now + Duration::from_secs(30);
        assert_eq!(
            stage_hard_limit(now, Some(deadline), 1.0),
            deadline.checked_sub(STAGE_TAIL_RESERVE)
        );
        assert_eq!(
            stage_hard_limit(now, Some(deadline), 0.25),
            Some(now + Duration::from_millis(7500)),
            "a 30 s row keeps 75% of its remaining clock for the BaB hand-off"
        );
        assert_eq!(
            stage_hard_limit(now, Some(now + Duration::from_millis(500)), 1.0),
            Some(now)
        );
        assert_eq!(stage_hard_limit(now, None, 0.25), None);

        // The root wait always leaves one disjunct's floor of the hard limit
        // for the attack: a lane that stays Pending cannot turn the stage into
        // a pure wait that then declines `no_budget`.
        let mut wait_policy = policy(0.5, 10.0, 100);
        wait_policy.root_wait_secs = 90.0;
        let limit = now + Duration::from_secs(8);
        assert_eq!(
            root_wait_until(now, Some(limit), &wait_policy),
            limit.checked_sub(Duration::from_millis(500)).unwrap(),
            "the wait never runs past hard_limit - min_secs"
        );
        wait_policy.root_wait_secs = 2.0;
        assert_eq!(
            root_wait_until(now, Some(limit), &wait_policy),
            now + Duration::from_secs(2)
        );
        assert_eq!(
            root_wait_until(now, None, &wait_policy),
            now + Duration::from_secs(2)
        );
        wait_policy.root_wait_secs = 0.0;
        assert_eq!(root_wait_until(now, Some(limit), &wait_policy), now);
        // A hard limit inside the floor never puts the wait in the past.
        wait_policy.root_wait_secs = 90.0;
        assert_eq!(
            root_wait_until(now, Some(now + Duration::from_millis(100)), &wait_policy),
            now
        );

        // The root wait: never on an idle or abandoned lane, bounded on a
        // pending one, and a published root is taken as it lands.
        let mut slept = 0usize;
        let w = wait_for_root(
            now + Duration::from_secs(3600),
            || RootBoundState::Idle,
            |_| slept += 1,
        );
        assert_eq!((w.token, slept), ("none_idle", 0));
        let w = wait_for_root(
            now + Duration::from_secs(3600),
            || RootBoundState::Abandoned,
            |_| slept += 1,
        );
        assert_eq!((w.token, slept), ("none_abandoned", 0));
        let w = wait_for_root(Instant::now(), || RootBoundState::Pending, |_| slept += 1);
        assert_eq!(w.token, "none_pending_timeout");
        assert!(w.view.is_none());
        let published = Arc::new(root_open::PublishedRootBounds {
            t: 0,
            adv: vec![1, 2],
            dj: vec![0.5, -0.5],
            publications: 2,
            first_published_at: now,
        });
        let mut calls = 0usize;
        let w = wait_for_root(
            now + Duration::from_secs(3600),
            || {
                calls += 1;
                if calls < 3 {
                    RootBoundState::Pending
                } else {
                    RootBoundState::Published(Arc::clone(&published))
                }
            },
            |pause| assert!(pause <= ROOT_POLL),
        );
        assert_eq!(w.token, "lane");
        assert_eq!(w.publications, 2);
        assert_eq!(
            w.view,
            Some(RootBoundsView {
                t: 0,
                adv: vec![1, 2],
                dj: vec![0.5, -0.5],
            })
        );
    }

    /// y0 = x, y1 = 0 on x in [-1, 1]. Disjunct 0, `Y_0 >= Y_1`, holds iff
    /// x >= 0 with margin x; disjunct 1, `Y_1 >= 1`, never holds.
    fn identity_margin_model() -> (BetaCrownModel, BoundedTensor, Vec<Vec<OutputConstraint>>) {
        let linear =
            LinearLayer::new(arr2(&[[1.0_f32], [0.0]]), Some(arr1(&[0.0_f32, 0.0]))).unwrap();
        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear));
        let input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
        let clauses = vec![ge(0, 1), vec![OutputConstraint::GreaterEqConst(1, 1.0)]];
        (
            BetaCrownModel::Sequential(Box::new(network)),
            input,
            clauses,
        )
    }

    #[test]
    fn open_disjunct_attack_never_emits_an_unvalidated_sat() {
        let (model, input, clauses) = identity_margin_model();
        let BetaCrownModel::Sequential(network) = &model else {
            unreachable!()
        };
        let graph = GraphNetwork::from_sequential(network).unwrap();
        let config = BetaCrownConfig {
            open_disjunct_attack: policy(0.5, 10.0, 4),
            ..BetaCrownConfig::default()
        };
        let inputs = StageInputs {
            model_net: &model,
            input: &input,
            clauses: &clauses,
            per_clause_input_bounds: &[],
            config: &config,
            pgd_steps: 10,
            attack_engine: None,
            overall_deadline: None,
        };
        let point = |x: f32| arr1(&[x]).into_dyn();
        let mut receipt = OpenDisjunctReceipt::new("test", config.open_disjunct_attack);

        // Violated in exact arithmetic but BELOW the 1e-4 noise floor: the
        // candidate is never handed up (the floor is reused, not relaxed).
        assert!(
            screen_and_confirm(&inputs, &graph, &point(5e-5), &mut receipt)
                .unwrap()
                .is_none()
        );
        // Not violated at all.
        assert!(
            screen_and_confirm(&inputs, &graph, &point(-0.5), &mut receipt)
                .unwrap()
                .is_none()
        );
        // Outside the attack box: the confirmation funnel refuses it even
        // though the property reads violated there.
        assert!(
            screen_and_confirm(&inputs, &graph, &point(2.0), &mut receipt)
                .unwrap()
                .is_none()
        );
        assert_eq!((receipt.near_misses, receipt.rejected), (2, 1));
        assert!(receipt.hit_clause.is_none());

        // A clear violation inside the box passes the funnel, and the result
        // carries exactly the confirmed point and its fresh forward.
        let result = screen_and_confirm(&inputs, &graph, &point(0.5), &mut receipt)
            .unwrap()
            .expect("a well-separated in-box violation confirms");
        match result.result {
            BabVerificationStatus::Violated {
                counterexample,
                output,
            } => {
                // The witness is the confirmed point, bit for bit.
                assert_eq!(counterexample, vec![0.5]);
                assert_eq!(output.len(), 2);
                assert!((output[0] - 0.5).abs() < 1e-6 && output[1].abs() < 1e-6);
            }
            other => panic!("expected Violated, got {other:?}"),
        }
        assert_eq!((receipt.hit_clause, receipt.hit_class), (Some(0), Some(0)));
        assert_eq!(receipt.reason, "candidate");
    }

    #[test]
    fn open_disjunct_stage_declines_without_the_batched_vjp_lane() {
        // No GPU engine: the stage declines with its reason and hands off,
        // it never falls through to an unguarded attack.
        let (model, input, clauses) = identity_margin_model();
        let config = BetaCrownConfig {
            open_disjunct_attack: policy(0.5, 10.0, 4),
            ..BetaCrownConfig::default()
        };
        let inputs = StageInputs {
            model_net: &model,
            input: &input,
            clauses: &clauses,
            per_clause_input_bounds: &[],
            config: &config,
            pgd_steps: 10,
            attack_engine: None,
            overall_deadline: Some(Instant::now() + Duration::from_secs(30)),
        };
        // `None` alone is what an exhausted attack returns too, so the test
        // reads the receipt: the decline must name THIS gate, before any
        // wave was started.
        let mut receipt = OpenDisjunctReceipt::new("stage", config.open_disjunct_attack);
        assert!(run_stage(&inputs, &mut receipt).unwrap().is_none());
        assert_eq!(receipt.reason, "gpu_unavailable");
        assert_eq!((receipt.restarts_started, receipt.waves), (0, 0));
        std::mem::forget(receipt); // keep the test's stderr quiet
        assert!(run_open_disjunct_attack_stage(&inputs).unwrap().is_none());
        // Dark config: nothing runs.
        let dark = BetaCrownConfig::default();
        assert!(!dark.open_disjunct_attack.restrict, "the stage ships dark");
        let inputs = StageInputs {
            config: &dark,
            ..inputs
        };
        let mut receipt = OpenDisjunctReceipt::new("stage", dark.open_disjunct_attack);
        assert!(run_stage(&inputs, &mut receipt).unwrap().is_none());
        assert_eq!(receipt.reason, "not_armed");
        std::mem::forget(receipt);
    }

    /// An ARMED stage the multi-clause route gates off still leaves a
    /// `phase=stage` line naming why; a dark policy leaves none.
    #[test]
    fn open_disjunct_stage_gated_off_leaves_its_reason() {
        for reason in ["pgd_disabled", "demand_zeroed"] {
            let receipt = declined_receipt(policy(0.5, 10.0, 4), reason)
                .expect("an armed policy always gets a receipt");
            let line = receipt.line();
            assert!(line.starts_with("[attack-open] armed=1 phase=stage "));
            assert!(line.contains(&format!(" reason={reason} ")), "{line}");
            assert!(line.contains(" restarts=0/0 "), "{line}");
            std::mem::forget(receipt);
        }
        assert!(declined_receipt(OpenDisjunctAttackConfig::default(), "pgd_disabled").is_none());
    }

    #[test]
    fn open_disjunct_receipt_schema_is_fixed() {
        let mut receipt = OpenDisjunctReceipt::new("stage", policy(0.5, 10.0, 100));
        let empty = receipt.line();
        receipt.reason = "candidate";
        receipt.root = "lane";
        receipt.open = 3;
        receipt.total = 9;
        receipt.hit_clause = Some(4);
        receipt.hit_class = Some(5);
        receipt.observe_margin(2.14e-4);
        let full = receipt.line();
        let keys = |line: &str| -> Vec<String> {
            line.split_whitespace()
                .skip(1)
                .map(|kv| kv.split('=').next().unwrap_or("").to_string())
                .collect()
        };
        assert_eq!(keys(&empty), keys(&full), "one schema for every exit");
        assert!(empty.starts_with("[attack-open] armed=1 phase=stage reason=error "));
        assert!(empty.contains(" hit=none hit_class=none margin=nan "));
        assert!(full.contains(" open=3/9 "));
        assert!(full.contains(" hit=4 hit_class=5 margin=+2.140000e-4 "));
        std::mem::forget(receipt); // keep the test's stderr quiet
    }

    /// The stage has no environment route: it is reachable only through the
    /// typed preset keys.
    #[test]
    fn open_disjunct_stage_reads_no_environment() {
        let source = include_str!("attack_open_disjuncts.rs");
        let production = source
            .split("#[cfg(test)]\nmod tests")
            .next()
            .expect("module has a production half");
        assert!(
            production.len() < source.len(),
            "test marker not found: VOID"
        );
        assert!(!production.contains("env::var"));
        assert!(!production.contains("std::env"));
        // Exactly one construction of a `Violated` result, inside the
        // confirmation funnel's `Some` arm.
        assert_eq!(production.matches("make_violated_result(").count(), 1);
        let confirm_at = production
            .find("re_evaluate_and_confirm(")
            .expect("the funnel is called");
        let emit_at = production.find("make_violated_result(").unwrap();
        assert!(confirm_at < emit_at);
    }
}
