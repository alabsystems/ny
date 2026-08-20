// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #lane-value-scheduler — the marginal-value ledger ACROSS the per-instance
//! attack lanes.
//!
//! ## The measured defect
//!
//! The per-instance budget is carved into fixed slices handed to lanes that run
//! blind. A lane producing nothing keeps its slice; a lane that is producing
//! keeps starving. Measured this session, one process per row:
//!
//! | row | LP sign-space | STE-PGD | upfront | BaB | unclaimed |
//! |-----|---------------|---------|---------|-----|-----------|
//! | `model_48_idx_1703_eps_1` | 217.52 s, 34 flips, margin -82, NO candidate | 117.51 s | 71.00 s | 46.78 s, 0 domains | 27.19 s |
//! | `model_64_idx_1703_eps_1` | 53.56 s (stall rule fired) | 240.10 s | 90.10 s | 69.06 s, 0 domains | 24.03 s |
//! | `model_30_idx_1703_eps_1` | 131.04 s -> **candidate** | — | — | — | — |
//!
//! 999.52 s of 1002.68 s of wall across the three non-productive rows went to
//! lanes that produced nothing, and a further 57.32 s was claimed by no lane at
//! all.
//!
//! ## Why this is not "give the productive lane more seconds"
//!
//! Row A's LP lane held 217.52 s, so the STE lane's cap computed to 117.51 s.
//! Row B's LP lane yielded at 53.56 s, so STE's cap computed to its 240.10 s
//! ceiling. A **163.96 s yield moved the downstream cap by 122.59 s** — and the
//! cap is a SCHEDULE, not a quantity: `bnn_ste_pgd`'s stage boundary is
//! `started + max_wall_time * (1 - climb_fraction)`, which sat at 88.1 s on row
//! A and 180.1 s on row B. A lane must therefore be handed a CAP it can plan
//! against, never a dribble of leftover seconds.
//!
//! That reallocation happened at all only because the TRAILING lane sizes
//! itself by subtraction from the LIVE remaining budget. Every other lane in
//! `vnncomp.rs` sizes itself by a private fraction of what it was HANDED, so a
//! yield upstream of them evaporates. **That asymmetry is what this module
//! fixes**: one pool, and every lane's cap re-derived from live remaining at
//! its own admission, through that lane's own unchanged plan function.
//!
//! ## What is reused, not rebuilt
//!
//! * [`ny_core::instance_budget`] — the published deadline. [`WallClockBudget`]
//!   is a thin reader over it; `remaining()` is `Some(ZERO)` when spent and
//!   `None` when unpublished, and the two are never conflated.
//! * [`ny_core::phase_window::admit`] (I1) — an unaffordable lane DECLINES
//!   rather than half-running, and a declined lane is charged nothing.
//! * [`ny_core::phase_yield::PhaseYield`] (I2) — a lane cannot express "abort
//!   the instance". `Declined` is not an error.
//! * [`ny_core::phase_scheduler::run_schedule`] — the loop and the ledger.
//!   Lanes report their own `actual_cost`, so an overrun is charged truthfully
//!   instead of the granted window being believed.
//!
//! ## Soundness
//!
//! Scheduling only. Every lane here returns a CLAIMED counterexample INPUT and
//! never a verdict; the claim becomes a `sat` only by passing the unchanged
//! `gate_sat_with_trusted_oracle`. Nothing in this module can produce an
//! `unsat`, because no scheduled lane has a verdict-bearing return type.
//!
//! ## Darkness
//!
//! Everything below is behind [`lane_value_scheduler_armed`]
//! (`NY_LANE_VALUE_SCHEDULER`, declared in `ny-levers`). Disarmed, the caller
//! never builds a ledger and each lane derives exactly the window it derives
//! today, from the same helper, in the same order.

// In a build without `mip` neither BNN lane exists, so the ledger has no
// consumer: the arming predicate is still read (the call site is
// feature-independent), but the scheduling surface below is unreachable. Say so
// rather than let a `mip`-less build accumulate warnings.
#![cfg_attr(not(feature = "mip"), allow(dead_code))]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use ny_core::phase_scheduler::{
    run_schedule, PhaseRun, SchedulablePhase, SchedulerPolicy, SchedulerReport,
};
use ny_core::phase_window::{admit, WindowPolicy};
use ny_core::phase_yield::{DeclineReason, PhaseYield};

/// Whether the cross-lane marginal-value scheduler is admitted.
///
/// Exact `"1"` arms it, exact `"0"` disarms it, every other byte sequence is a
/// recorded rejection falling back to the declaration's `false`. Fails CLOSED.
pub(crate) fn lane_value_scheduler_armed() -> bool {
    ny_levers::read(&ny_levers::decls::dark_probes::LANE_VALUE_SCHEDULER)
        .value
        .as_bool()
}

/// The arming rule as a pure predicate over one raw environment string.
///
/// Same declaration, same parser, same chokepoint — only the lookup is
/// injected — so a test of this is a test of the production rule.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn lane_value_scheduler_armed_from(raw: Option<&str>) -> bool {
    let owned = raw.map(str::to_owned);
    ny_levers::read_with(
        &ny_levers::decls::dark_probes::LANE_VALUE_SCHEDULER,
        move |_| owned,
    )
    .value
    .as_bool()
}

// ---------------------------------------------------------------------------
// The budget source
// ---------------------------------------------------------------------------

/// Where a lane's cost plan reads "how much is left".
///
/// Production reads the process-global published instance deadline; the wall
/// clock does the charging, so [`Self::charge`] is a no-op there. Tests drive a
/// simulated cell, so every assertion in this module is on the LEDGER and not
/// on a clock.
pub(crate) trait LaneBudgetSource {
    /// Live remaining INSTANCE budget. `None` means "no deadline published" —
    /// which is information absent, never budget spent.
    fn remaining(&self) -> Option<Duration>;
    /// Account `spent` seconds. Production is a no-op: real time already moved.
    fn charge(&self, spent: Duration);
}

/// Reads an explicit deadline, falling back to [`ny_core::instance_budget`].
///
/// The attack lanes size themselves against the post-BaB-preset deadline the
/// orchestrator already holds, so the scheduler must read the SAME clock they
/// do or the armed arm would be planning against a different budget than the
/// disarmed one. The published instance deadline is the fallback, and its
/// absence stays absent — `None` here is "no information", never "no budget".
pub(crate) struct DeadlineBudget {
    pub deadline: Option<std::time::Instant>,
}

impl LaneBudgetSource for DeadlineBudget {
    fn remaining(&self) -> Option<Duration> {
        match self.deadline {
            Some(deadline) => Some(deadline.saturating_duration_since(std::time::Instant::now())),
            None => ny_core::instance_budget::remaining(),
        }
    }
    fn charge(&self, _spent: Duration) {}
}

/// A deterministic budget for tests: no clock, no sleeping, no flakiness.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct SimulatedBudget(RefCell<Duration>);

#[cfg_attr(not(test), allow(dead_code))]
impl SimulatedBudget {
    pub(crate) fn new(remaining: Duration) -> Self {
        Self(RefCell::new(remaining))
    }
}

impl LaneBudgetSource for SimulatedBudget {
    fn remaining(&self) -> Option<Duration> {
        Some(*self.0.borrow())
    }
    fn charge(&self, spent: Duration) {
        let mut r = self.0.borrow_mut();
        *r = r.saturating_sub(spent);
    }
}

// ---------------------------------------------------------------------------
// The value contract
// ---------------------------------------------------------------------------

/// What a lane reports about its own progress, in ITS OWN units.
///
/// Deliberately three-valued. A lane that cannot honestly price itself must be
/// able to SAY so rather than dress a work counter up as progress: measured,
/// BaB reported `domains_explored = 0` on all three timeout rows and
/// `output_bound_width = null` on every one, so on those routes it has a
/// confident zero and no value curve at all.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LaneValue {
    /// The lane produced a claimed counterexample: one unit of the only value
    /// the attack slice banks. Still NOT a verdict.
    Candidate,
    /// The lane measured its own progress. `gain` is in the lane's own unit
    /// (pattern-margin movement) and `work_units` is its own denominator (LP
    /// solves, probes) — never seconds, which read three different things on
    /// three rows of one family (0.42 / 0.59 / 1.67 s per LP, measured).
    Measured { gain: f64, work_units: u64 },
    /// The lane ran and can state a confident zero.
    Zero { work_units: u64 },
    /// The lane has no honest value signal on this route.
    NoSignal,
}

impl LaneValue {
    /// The scalar the scheduler's I3 retirement rule sees.
    ///
    /// [`Self::NoSignal`] scores zero because it produced nothing, not because
    /// it is known worthless — and since each attack lane is consulted at most
    /// once per instance, retirement never re-decides anything for it. The
    /// distinction is kept in the ledger, where it is the honest record.
    fn scalar(&self) -> f64 {
        match self {
            Self::Candidate => 1.0,
            Self::Measured { gain, .. } => gain.max(0.0),
            Self::Zero { .. } | Self::NoSignal => 0.0,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn work_units(&self) -> u64 {
        match self {
            Self::Measured { work_units, .. } | Self::Zero { work_units } => *work_units,
            Self::Candidate | Self::NoSignal => 0,
        }
    }
}

/// One consultation of one lane.
pub(crate) struct LaneBlock {
    /// The lane's own report. `Declined` is not an error (I2).
    pub value: PhaseYield<LaneValue>,
    /// Wall time the LANE says it consumed. May exceed the granted window; an
    /// overrun is charged, not disbelieved.
    pub actual_cost: Duration,
    /// A CLAIMED counterexample witness, pending the caller's trusted-oracle
    /// gate. Never a verdict.
    pub witness: Option<String>,
}

/// A lane the scheduler can hand a cap to.
pub(crate) struct LaneSpec {
    pub name: &'static str,
    /// The lane's OWN cost rule, as a function of LIVE remaining budget.
    ///
    /// This is the same function the unscheduled lane uses. Handing the
    /// scheduler the rule instead of a number is the whole point: a yield
    /// upstream re-enters here as a bigger CAP the lane can plan against.
    /// `None` means "not worth starting on this much budget".
    pub plan: Box<dyn Fn(Option<Duration>) -> Option<Duration>>,
    /// Measured prior on this lane's value, used only to break ties among
    /// simultaneously ready lanes. Must be > 0 or the lane is never scheduled.
    pub prior_value: f64,
    /// Run for at most `window`.
    pub run: Box<dyn FnMut(Duration) -> LaneBlock>,
}

// ---------------------------------------------------------------------------
// The ledger
// ---------------------------------------------------------------------------

/// One lane's row in the budget ledger.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LaneLedgerRow {
    pub name: &'static str,
    /// The window the scheduler admitted, re-derived from LIVE remaining.
    pub granted: Duration,
    /// What the lane REPORTED it consumed.
    pub spent: Duration,
    /// `granted - spent`: the seconds the lane gave back to the pool.
    pub yielded: Duration,
    /// The window this lane would have been granted had every lane before it
    /// spent its whole grant. `granted - counterfactual_grant` is the yield
    /// this lane actually ABSORBED, which is what makes "reallocated, not
    /// lost" an assertion about the ledger rather than about a log line.
    pub counterfactual_grant: Duration,
    pub value: Option<LaneValue>,
    pub declined: Option<String>,
}

impl LaneLedgerRow {
    /// Seconds this lane absorbed from upstream yields.
    pub fn absorbed(&self) -> Duration {
        self.granted.saturating_sub(self.counterfactual_grant)
    }
}

/// The whole schedule's budget ledger.
#[derive(Debug, Clone)]
pub(crate) struct LaneLedger {
    /// Seconds the scheduler was allowed to hand out in total. Sized so the
    /// publication margin can never be crossed.
    pub pool: Duration,
    /// Live remaining instance budget when the schedule opened.
    // Staged dark (8429e0466): recorded for the wiring commit; no consumer
    // or test reads it yet, on any feature set.
    #[allow(dead_code)]
    pub opening_remaining: Option<Duration>,
    pub rows: Vec<LaneLedgerRow>,
    /// Total charged, including any overrun.
    pub spent: Duration,
    // Staged dark (8429e0466): see opening_remaining above.
    #[allow(dead_code)]
    pub stop: ny_core::phase_scheduler::StopReason,
}

impl LaneLedger {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn row(&self, name: &str) -> Option<&LaneLedgerRow> {
        self.rows.iter().find(|r| r.name == name)
    }
    /// Seconds handed to no lane at all.
    pub fn unspent(&self) -> Duration {
        self.pool.saturating_sub(self.spent)
    }
    /// Σ of every lane's returned remainder.
    // Staged dark (8429e0466): see opening_remaining above.
    #[allow(dead_code)]
    pub fn total_yielded(&self) -> Duration {
        self.rows.iter().map(|r| r.yielded).sum()
    }
    /// Σ of every lane's absorbed upstream yield.
    pub fn total_reallocated(&self) -> Duration {
        self.rows.iter().map(LaneLedgerRow::absorbed).sum()
    }
    pub fn describe(&self) -> String {
        let rows: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                format!(
                    "{}: granted {:.2}s (counterfactual {:.2}s, absorbed {:.2}s), spent {:.2}s, \
                     yielded {:.2}s, value {:?}{}",
                    r.name,
                    r.granted.as_secs_f64(),
                    r.counterfactual_grant.as_secs_f64(),
                    r.absorbed().as_secs_f64(),
                    r.spent.as_secs_f64(),
                    r.yielded.as_secs_f64(),
                    r.value,
                    r.declined
                        .as_ref()
                        .map_or(String::new(), |d| format!(", declined: {d}")),
                )
            })
            .collect();
        format!(
            "lane ledger: pool {:.2}s, spent {:.2}s, unspent {:.2}s, reallocated {:.2}s [{}]",
            self.pool.as_secs_f64(),
            self.spent.as_secs_f64(),
            self.unspent().as_secs_f64(),
            self.total_reallocated().as_secs_f64(),
            rows.join("; "),
        )
    }
}

// ---------------------------------------------------------------------------
// The schedule
// ---------------------------------------------------------------------------

/// The admission policy for the attack lanes.
///
/// * `max_frac = 1.0` and NO safety multiplier, because each lane's `plan`
///   ALREADY reserves what must be left behind — the LP lane halves what
///   remains after the 45 s publication margin, the trailing lane subtracts
///   that margin AND the 100 s downstream reserve. Padding a plan that already
///   reserved would hand out seconds those reserves exist to protect, and
///   scaling it a second time would compound halves, which is the defect the
///   trailing lane's subtraction exists to avoid.
/// * `floor` is the lane floor: under this, starting is pointless.
///
/// The aggregate is bounded separately, by the POOL: see
/// [`lane_schedule_pool`].
fn lane_window_policy(floor: Duration) -> WindowPolicy {
    WindowPolicy {
        max_frac: 1.0,
        margin_num: 1,
        margin_den: 1,
        floor,
    }
}

/// Seconds the scheduled lanes may collectively spend.
///
/// `remaining - publication margin`. The margin is the hard wall and it is
/// already close: measured max wall 455.97 s against a 480 s budget, and one
/// `error` row makes a family unbankable. `run_schedule` stops admitting once
/// this is exhausted, so even a lane that overruns its own window cannot pull
/// the aggregate across the margin — the overrun is charged and the next lane
/// is refused.
///
/// The DOWNSTREAM RESERVE is not subtracted here because it is not a
/// scheduler-level reserve: it belongs to the trailing lane's own plan, which
/// subtracts it from live remaining every time it is consulted. Taking it twice
/// would shrink the slice the measurement says is already the difference
/// between four rows and seven.
pub(crate) fn lane_schedule_pool(
    remaining: Option<Duration>,
    margin: Duration,
) -> Option<Duration> {
    remaining?.checked_sub(margin)
}

/// Shared mutable state the lanes and the ledger both touch.
struct SchedState {
    budget: Box<dyn LaneBudgetSource>,
    /// Set once a lane produces a claimed counterexample: every later lane
    /// stops being ready, so the schedule ends instead of spending the pool on
    /// a question already answered.
    witness: RefCell<Option<String>>,
    /// Per-lane "this lane's turn is over" flags, in pipeline order.
    done: RefCell<Vec<bool>>,
    rows: RefCell<Vec<LaneLedgerRow>>,
    /// `pool - Σ charged`, mirroring `run_schedule`'s own arithmetic so the
    /// pre-admission check and the scheduler's `admit` can never disagree.
    pool_left: RefCell<Duration>,
    /// Live remaining MINUS every grant handed out so far, whether or not the
    /// lane used it. This is the counterfactual budget in which nothing yields.
    counterfactual_left: RefCell<Option<Duration>>,
    policy: WindowPolicy,
}

struct LanePhase {
    index: usize,
    spec: LaneSpec,
    state: Rc<SchedState>,
}

impl LanePhase {
    /// The window this lane's own rule asks for against the live budget, or
    /// `None` when the lane says it is not worth starting.
    fn planned(&self) -> Option<Duration> {
        (self.spec.plan)(self.state.budget.remaining())
    }
}

impl SchedulablePhase for LanePhase {
    fn name(&self) -> &'static str {
        self.spec.name
    }

    /// The lane's cost is re-derived HERE, from live remaining, every time it
    /// is considered. That is the fix: the number is never precomputed against
    /// a stale total, so a yield upstream arrives as a larger cap.
    fn predicted_cost(&self) -> Duration {
        let Some(cost) = self.planned() else {
            // The lane's own rule says do not start. Retire its turn now, so
            // the lane behind it becomes ready in this same pass rather than
            // the schedule stalling on a lane that will never run.
            self.retire_turn("lane plan declined: budget below the lane floor");
            return Duration::MAX;
        };
        if !admit(cost, *self.state.pool_left.borrow(), self.state.policy).is_admitted() {
            self.retire_turn("unaffordable inside the remaining pool");
            return Duration::MAX;
        }
        cost
    }

    fn predicted_value(&self) -> f64 {
        self.spec.prior_value
    }

    /// PIPELINE ORDER, and it is measured, not assumed. On the three
    /// `model_30` eps=1 rows the LP lane owns, the STE lane was measured
    /// `exhausted` over its whole budget, so running it first would spend the
    /// slice that recovers those rows on a search that cannot recover them.
    /// The survey also measured that ordering ALONE fixes nothing: with the LP
    /// lane disabled entirely, STE got 217.5 s and returned the SAME rows. So
    /// the scheduler keeps the measured order and reallocates inside it.
    fn is_ready(&self) -> bool {
        if self.state.witness.borrow().is_some() {
            return false;
        }
        let done = self.state.done.borrow();
        !done[self.index] && done[..self.index].iter().all(|d| *d)
    }

    fn run(&mut self, window: Duration) -> PhaseRun {
        // Record the counterfactual BEFORE charging, so it answers "what would
        // this lane have been granted if nothing upstream had yielded?".
        let counterfactual = {
            let cf = *self.state.counterfactual_left.borrow();
            (self.spec.plan)(cf).unwrap_or(Duration::ZERO)
        };

        let block = (self.spec.run)(window);
        let actual = block.actual_cost;

        // Charge the ACTUAL cost everywhere: the live budget the next lane
        // plans against, the pool the next admission is checked against, and
        // the ledger. Believing the grant over the observation is how a budget
        // silently goes negative.
        self.state.budget.charge(actual);
        {
            let mut left = self.state.pool_left.borrow_mut();
            *left = left.saturating_sub(actual);
        }
        {
            // The counterfactual charges the GRANT, not the spend: that is
            // exactly what "nobody yields" means.
            let mut cf = self.state.counterfactual_left.borrow_mut();
            *cf = cf.map(|c| c.saturating_sub(window));
        }

        let (value, declined) = match &block.value {
            PhaseYield::Complete(v) | PhaseYield::Partial(v) => (Some(v.clone()), None),
            PhaseYield::Declined(reason) => (None, Some(format!("{reason:?}"))),
        };
        self.state.rows.borrow_mut().push(LaneLedgerRow {
            name: self.spec.name,
            granted: window,
            spent: actual,
            yielded: window.saturating_sub(actual),
            counterfactual_grant: counterfactual,
            value,
            declined,
        });
        if let Some(witness) = block.witness {
            *self.state.witness.borrow_mut() = Some(witness);
        }
        self.state.done.borrow_mut()[self.index] = true;

        PhaseRun {
            yielded: block.value.map(|v| v.scalar()),
            actual_cost: actual,
        }
    }
}

impl LanePhase {
    /// End this lane's turn without running it, and record WHY. A lane that
    /// never ran is charged nothing — the ledger row carries `spent = 0`.
    fn retire_turn(&self, why: &'static str) {
        let mut done = self.state.done.borrow_mut();
        if done[self.index] {
            return;
        }
        done[self.index] = true;
        drop(done);
        self.state.rows.borrow_mut().push(LaneLedgerRow {
            name: self.spec.name,
            granted: Duration::ZERO,
            spent: Duration::ZERO,
            yielded: Duration::ZERO,
            counterfactual_grant: Duration::ZERO,
            value: None,
            declined: Some(why.to_string()),
        });
    }
}

/// Run the attack lanes under one marginal-value ledger.
///
/// Returns the first CLAIMED counterexample (never a verdict — the caller must
/// still route it through the unchanged trusted-oracle gate) alongside the
/// ledger.
pub(crate) fn run_lane_schedule(
    lanes: Vec<LaneSpec>,
    budget: Box<dyn LaneBudgetSource>,
    pool: Duration,
    floor: Duration,
) -> (Option<String>, LaneLedger) {
    let opening_remaining = budget.remaining();
    let policy = lane_window_policy(floor);
    let state = Rc::new(SchedState {
        budget,
        witness: RefCell::new(None),
        done: RefCell::new(vec![false; lanes.len()]),
        rows: RefCell::new(Vec::with_capacity(lanes.len())),
        pool_left: RefCell::new(pool),
        counterfactual_left: RefCell::new(opening_remaining),
        policy,
    });

    let mut phases: Vec<Box<dyn SchedulablePhase>> = lanes
        .into_iter()
        .enumerate()
        .map(|(index, spec)| {
            Box::new(LanePhase {
                index,
                spec,
                state: Rc::clone(&state),
            }) as Box<dyn SchedulablePhase>
        })
        .collect();

    let report: SchedulerReport = run_schedule(
        &mut phases,
        pool,
        SchedulerPolicy {
            window: policy,
            // Each lane is consulted at most once, so I3's consecutive-block
            // retirement never re-decides anything here; it is left at the
            // ny-core default rather than tuned to a number this shape cannot
            // exercise.
            ..SchedulerPolicy::default()
        },
    );
    drop(phases);

    let witness = state.witness.borrow().clone();
    let rows = state.rows.borrow().clone();
    (
        witness,
        LaneLedger {
            pool,
            opening_remaining,
            rows,
            spent: report.spent,
            stop: report.stop,
        },
    )
}

/// A lane block that produced a claimed counterexample.
pub(crate) fn candidate_block(witness: String, actual_cost: Duration) -> LaneBlock {
    LaneBlock {
        value: PhaseYield::Complete(LaneValue::Candidate),
        actual_cost,
        witness: Some(witness),
    }
}

/// A lane block that ran, produced nothing publishable, and can price itself.
pub(crate) fn measured_block(gain: f64, work_units: u64, actual_cost: Duration) -> LaneBlock {
    LaneBlock {
        value: PhaseYield::Partial(if gain > 0.0 {
            LaneValue::Measured { gain, work_units }
        } else {
            LaneValue::Zero { work_units }
        }),
        actual_cost,
        witness: None,
    }
}

/// A lane block that could not start. NOT an error (I2), and charged nothing
/// beyond whatever it actually burned deciding.
pub(crate) fn declined_block(reason: DeclineReason, actual_cost: Duration) -> LaneBlock {
    LaneBlock {
        value: PhaseYield::Declined(reason),
        actual_cost,
        witness: None,
    }
}

/// A lane block from a lane with no honest value signal on this route.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn unmeasurable_block(actual_cost: Duration) -> LaneBlock {
    LaneBlock {
        value: PhaseYield::Partial(LaneValue::NoSignal),
        actual_cost,
        witness: None,
    }
}

// The tests plan against the two production lanes' own budget rules, which
// live behind the `mip` feature with the lanes themselves.
#[cfg(all(test, feature = "mip"))]
#[path = "lane_schedule_tests.rs"]
mod lane_schedule_tests;
