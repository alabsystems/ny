// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r6 — O-TRIAL: the pure trial state machines (ROUTER_PROGRAM §5.1,
//! the §5.2 rows tagged R6, §5.3, §5.4, §10 R6).
//!
//! PURE: no clock, no environment, no global. A lane owns one machine per
//! trialled knob, feeds it ITS OWN counters and the time IT measured at its
//! own declared safe points (unit boundaries), and reads back the value to
//! use for the NEXT unit. The same stream of observations gives the same
//! decisions bit for bit, which the synthetic-stream tests pin.
//!
//! # The four machines (each owned by its own acting rule)
//!
//! | machine | rule | knob it moves | protocol |
//! |---|---|---|---|
//! | [`AimdTrial`] | `rule.trial_width` | a fast lane's batch / wave width | AIMD on certified closes per second inside the lane's own static memory budget: x2 on a >= 5% rise with lease use < 70%, /2 on a >= 10% fall; a raise that falls twice goes back to the last good width and FREEZES; lease use > 85% reverts a pending raise at once (the §5.3 guard) and otherwise starts a halving TRIAL through the same protocol gate (review fix) |
//! | [`LrBackoff`] | `rule.trial_lr_backoff` | an ascent's step size (a multiplier on the lane's own lr) | BACKOFF ONLY: 3 rejected iterates in a row => lr x 0.5, floor lr0 / 8; revert (and tabu) when the next 3 iterates' gain per second falls |
//! | [`SplitRestart`] | `rule.trial_split_restart` | a lane's split heuristic | RESTART FORM: FAR on the running lane AND a zero-lift split fraction > 0.9 over the two trailing windows => ONE restart with the other heuristic; the caller discards its tree and carries no fact into the new one |
//! | [`Halving`] | `rule.trial_attack_halving` | an attack's initial step size | SUCCESSIVE HALVING across a fixed step-size pool by improvement rate: each round gives every surviving arm the same number of restarts (alternating restart parity), then keeps the better half |
//!
//! # The trial protocol (§5.3), shared by every throughput trial
//!
//! A trial starts only when the lane is FAST and has room to measure it
//! ([`start_refusal`]): a window of at least [`TRIAL_WINDOW_UNITS`] work
//! units, at least [`TRIAL_WINDOWS_LEFT_MIN`] windows left in the lane at
//! the current window rate, never in the last quarter of the lane, and at
//! most [`TRIAL_MAX_PER_KNOB`] trials per knob per row. A trial is judged
//! against the last full window before it (`baseline`) over 1..=2 windows:
//! kept on `metric >= baseline x (1 + keep_margin)`, else reverted, and the
//! direction it moved in becomes TABU for the rest of the row.
//!
//! # Soundness
//!
//! Every value a machine can produce lies inside the range its rule declares
//! for the site ([`WidthRange`], the floor of [`LrRule`], the fixed pool of
//! [`HalvingRule`], the declared split heuristics). Each knob moves SEARCH or
//! EFFORT only: which domains a wave bounds together, how far an ascent
//! steps between iterates it keeps or discards (every iterate is a valid
//! relaxation; the kept one is checked once by the phase's own path), where
//! a box is bisected (any bisection covers its box), where an attack starts
//! its steps (every candidate passes the unchanged acceptance gate). No
//! machine touches a precision, a tolerance or an error term, and a restart
//! carries nothing from the discarded tree. A wrong trial can cost a proof
//! or a counterexample, never mint one.
//!
//! # Never panics
//!
//! f64 or saturating integer arithmetic only; a non-finite or degenerate
//! observation is a refusal (`Unknown`), never a decision.

// NaN-refusing comparisons are deliberate here: `!(x < y)` is TRUE when either
// side is NaN, and this module reads every NaN as a refusal (never a decision).
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use serde::{Deserialize, Serialize};

use super::project::{self, FarClass, Window};

/// Record schema of every [`TrialEvent`].
pub const TRIAL_SCHEMA: &str = "ny-router-trial/v1";

/// The smallest window a throughput trial may judge over, in the lane's own
/// work units (§5.1: ">= 16 units per window"; the same window every
/// frontier projection uses).
pub const TRIAL_WINDOW_UNITS: u64 = project::FRONTIER_WINDOW_UNITS;

/// Windows that must still fit in the lane, at the current window's rate,
/// for a trial to start (§5.1: ">= 8 windows left in the lane").
pub const TRIAL_WINDOWS_LEFT_MIN: f64 = 8.0;

/// No trial starts once this fraction of the lane has elapsed (§5.1: "never
/// in the last 25% of the lane").
pub const TRIAL_LATE_FRAC: f64 = 0.75;

/// Trials per knob per row (§5.1: "<= 2 trials per knob per row").
pub const TRIAL_MAX_PER_KNOB: u32 = 2;

/// Windows a trial is judged over (§5.3: "eval_windows 1..=2").
pub const TRIAL_EVAL_WINDOWS: u32 = 2;

/// The direction a trial moves its knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialDir {
    /// Raise the value.
    Up,
    /// Lower the value.
    Down,
    /// Switch to the other declared alternative (restart form).
    Switch,
    /// One successive-halving elimination round.
    Halve,
}

impl TrialDir {
    /// Stable token for logs.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
            Self::Switch => "switch",
            Self::Halve => "halve",
        }
    }
}

/// What a trial event records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialDecision {
    /// A trial moved its knob and is being measured.
    Started,
    /// A trial's new value beat its baseline and stays.
    Kept,
    /// A trial's new value lost and the old value is back.
    Reverted,
    /// A trigger fired in a direction the row already made tabu.
    Tabu,
    /// The machine will move nothing more on this row.
    Frozen,
    /// A memory guard moved the value (not a trial; never counted as one).
    /// Review fix (router-r6): no machine emits it any more — lease pressure
    /// with nothing pending STARTS a protocol-gated halving trial, and a
    /// pending raise under pressure is `Reverted` (`why=lease_guard`). Kept
    /// so recorded ledgers still deserialize.
    Guard,
    /// A trigger fired but the protocol refused to start a trial (`why`).
    Blocked,
}

impl TrialDecision {
    /// Stable token for logs (`kept|reverted|tabu|...`).
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::Kept => "kept",
            Self::Reverted => "reverted",
            Self::Tabu => "tabu",
            Self::Frozen => "frozen",
            Self::Guard => "guard",
            Self::Blocked => "blocked",
        }
    }
}

/// One trial decision, as the lane prints it and the authority ledgers it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialEvent {
    /// The acting rule (`rule.trial_width`, ...).
    pub rule: String,
    /// The capability the knob belongs to (registry id).
    pub site: String,
    /// The knob (its typed key where one exists).
    pub knob: String,
    /// Direction.
    pub dir: TrialDir,
    /// Decision.
    pub decision: TrialDecision,
    /// The value before the event.
    pub from: f64,
    /// The value after the event.
    pub to: f64,
    /// The metric of the window that decided (0 when none).
    pub metric: f64,
    /// The baseline it was compared against (0 when none).
    pub baseline: f64,
    /// The lane's work-unit count at the decision.
    pub unit: u64,
    /// Windows the machine had closed at the decision.
    pub window: u64,
    /// Why (stable token, plus detail where the decision needs it).
    pub why: String,
}

fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

impl TrialEvent {
    /// The event with every float made JSON-safe (a non-finite value would
    /// serialize as null and break the flight record's replay).
    #[must_use]
    pub fn sanitized(mut self) -> Self {
        self.from = finite_or_zero(self.from);
        self.to = finite_or_zero(self.to);
        self.metric = finite_or_zero(self.metric);
        self.baseline = finite_or_zero(self.baseline);
        self
    }

    /// `[router] trial knob=<key> dir=<up|down|switch|halve>
    /// decision=<started|kept|reverted|tabu|frozen|guard|blocked> rule=<id>
    /// site=<cap> from= to= metric= baseline= unit= window= why=`.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "[router] trial knob={} dir={} decision={} rule={} site={} from={} to={} \
             metric={:.6e} baseline={:.6e} unit={} window={} why={}",
            self.knob,
            self.dir.token(),
            self.decision.token(),
            self.rule,
            self.site,
            self.from,
            self.to,
            self.metric,
            self.baseline,
            self.unit,
            self.window,
            self.why.replace(char::is_whitespace, "_"),
        )
    }
}

/// Why a throughput trial may not start now (`None`: it may). The §5.1 power
/// requirement and the §5.3 preconditions, in work units and in the lane's
/// own time: a window of at least [`TRIAL_WINDOW_UNITS`] units, at least
/// `windows_left_min` windows of `window_secs` left before the lane's own
/// deadline, the lane not past `late_frac` of its span, and the knob's trial
/// budget not spent. A missing deadline is a refusal (`Unknown` never
/// decides).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn start_refusal(
    window_units: u64,
    window_secs: f64,
    elapsed_s: f64,
    remaining_s: Option<f64>,
    trials_used: u32,
    max_trials: u32,
    windows_left_min: f64,
    late_frac: f64,
) -> Option<&'static str> {
    if trials_used >= max_trials {
        return Some("trials_exhausted");
    }
    if window_units < TRIAL_WINDOW_UNITS {
        return Some("window_units");
    }
    let Some(remaining) = remaining_s.filter(|r| r.is_finite() && *r >= 0.0) else {
        return Some("deadline_unknown");
    };
    if !(window_secs.is_finite() && window_secs > 0.0) {
        return Some("window_secs");
    }
    if !(remaining / window_secs >= windows_left_min) {
        return Some("windows_left");
    }
    let total = elapsed_s + remaining;
    if !(elapsed_s.is_finite() && elapsed_s >= 0.0 && total > 0.0) {
        return Some("lane_clock");
    }
    if !(elapsed_s / total < late_frac) {
        return Some("late");
    }
    None
}

// ---------------------------------------------------------------------------
// rule.trial_width: AIMD on certified closes per second
// ---------------------------------------------------------------------------

/// `rule.trial_width`'s constants (ROUTER_PROGRAM §5.2, batch / wave width).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WidthRule {
    /// Work units per judged window (`>= TRIAL_WINDOW_UNITS`).
    pub window_units: u64,
    /// A window whose metric rose by at least this fraction over the one
    /// before it may start a raise (`x2`).
    pub raise_rise: f64,
    /// A window whose metric fell by at least this fraction may start a
    /// lowering (`/2`).
    pub fall_drop: f64,
    /// A raise needs lease use strictly below this.
    pub lease_raise_max: f64,
    /// Lease use above this reverts a pending raise at once (the §5.3
    /// guard), or else STARTS a halving trial (the §5.2 trigger; review fix:
    /// through the protocol gate, never outside it).
    pub lease_guard: f64,
    /// A trial is kept on `metric >= baseline x (1 + keep_margin)`.
    pub keep_margin: f64,
    /// Trials per knob per row.
    pub max_trials: u32,
    /// Windows a trial is judged over.
    pub eval_windows: u32,
    /// Windows that must still fit for a trial to start.
    pub windows_left_min: f64,
    /// No trial starts past this fraction of the lane.
    pub late_frac: f64,
    /// A site's trial range is `[base / factor, base x factor]` inside the
    /// site's own clamp and declared span.
    pub range_factor: u64,
}

impl WidthRule {
    /// The shipped constants (ROUTER_PROGRAM §5.2: x2 on >= 5% rise with lease
    /// < 70%, /2 on >= 10% fall or lease > 85%; §5.1/§5.3 protocol).
    pub const SHIPPED: Self = Self {
        window_units: TRIAL_WINDOW_UNITS,
        raise_rise: 0.05,
        fall_drop: 0.10,
        lease_raise_max: 0.70,
        lease_guard: 0.85,
        keep_margin: 0.05,
        max_trials: TRIAL_MAX_PER_KNOB,
        eval_windows: TRIAL_EVAL_WINDOWS,
        windows_left_min: TRIAL_WINDOWS_LEFT_MIN,
        late_frac: TRIAL_LATE_FRAC,
        range_factor: 4,
    };

    /// Whether the constants are inside the protocol's domain (a looser
    /// value would weaken a §5.1 power requirement).
    #[must_use]
    pub fn is_valid(self) -> bool {
        let unit = |x: f64| x.is_finite() && x > 0.0 && x < 1.0;
        self.window_units >= TRIAL_WINDOW_UNITS
            && unit(self.raise_rise)
            && unit(self.fall_drop)
            && unit(self.lease_raise_max)
            && self.lease_guard.is_finite()
            && self.lease_guard > self.lease_raise_max
            && self.lease_guard <= 1.0
            && self.keep_margin.is_finite()
            && self.keep_margin >= 0.0
            && (1..=TRIAL_MAX_PER_KNOB).contains(&self.max_trials)
            && (1..=TRIAL_EVAL_WINDOWS).contains(&self.eval_windows)
            && self.windows_left_min.is_finite()
            && self.windows_left_min >= TRIAL_WINDOWS_LEFT_MIN
            && self.late_frac.is_finite()
            && self.late_frac > 0.0
            && self.late_frac <= TRIAL_LATE_FRAC
            && (2..=16).contains(&self.range_factor)
    }
}

/// The closed range a width trial may move inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WidthRange {
    /// Smallest admissible width (`>= 1`).
    pub min: u64,
    /// Largest admissible width.
    pub max: u64,
}

impl WidthRange {
    /// The range around today's value `base`: `[base / factor, base x
    /// factor]` intersected with the site's own `[hard_min, hard_max]` (its
    /// reader clamp and declared span). `None` when the intersection is empty
    /// or does not contain `base` (the site then runs today's value).
    #[must_use]
    pub fn around(base: u64, factor: u64, hard_min: u64, hard_max: u64) -> Option<Self> {
        let factor = factor.max(1);
        let min = (base / factor).max(hard_min).max(1);
        let max = base.saturating_mul(factor).min(hard_max);
        (min <= base && base <= max).then_some(Self { min, max })
    }

    /// Whether `value` is inside the range.
    #[must_use]
    pub const fn contains(self, value: u64) -> bool {
        self.min <= value && value <= self.max
    }
}

/// What a width site reports at a unit boundary (cumulative counters).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WidthObs {
    /// Work units completed so far (waves, batches).
    pub units: u64,
    /// Certified closes so far.
    pub closes: u64,
    /// Seconds since the lane started (the lane's own clock).
    pub t_s: f64,
    /// Seconds left before the lane's own deadline (`None`: unknown).
    pub remaining_s: Option<f64>,
    /// The lane's working set over its own static memory budget (`None`:
    /// unknown — a raise is then refused).
    pub lease_use: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Mark {
    units: u64,
    closes: u64,
    t: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Pending {
    dir: TrialDir,
    from: u64,
    baseline: f64,
    evals: u32,
    falls: u32,
}

/// `rule.trial_width`'s AIMD machine for one site's width.
#[derive(Debug, Clone, PartialEq)]
pub struct AimdTrial {
    rule: WidthRule,
    rule_id: String,
    site: String,
    knob: String,
    range: WidthRange,
    value: u64,
    last_good: u64,
    pending: Option<Pending>,
    tabu_up: bool,
    tabu_down: bool,
    frozen: bool,
    trials_used: u32,
    prev_metric: Option<f64>,
    mark: Option<Mark>,
    windows: u64,
    reported: Vec<&'static str>,
}

impl AimdTrial {
    /// A machine starting at today's value `base`. `None` when the rule's
    /// constants are invalid or `base` is outside `range`.
    #[must_use]
    pub fn new(
        rule: WidthRule,
        rule_id: &str,
        site: &str,
        knob: &str,
        base: u64,
        range: WidthRange,
    ) -> Option<Self> {
        if !rule.is_valid() || !range.contains(base) || range.min == 0 {
            return None;
        }
        Some(Self {
            rule,
            rule_id: rule_id.to_owned(),
            site: site.to_owned(),
            knob: knob.to_owned(),
            range,
            value: base,
            last_good: base,
            pending: None,
            tabu_up: false,
            tabu_down: false,
            frozen: false,
            trials_used: 0,
            prev_metric: None,
            mark: None,
            windows: 0,
            reported: Vec::new(),
        })
    }

    /// The same machine with `prior` trials already spent on this knob in
    /// this row (by an earlier run of the same lane): the per-row budget of
    /// [`WidthRule::max_trials`] is the row's, not the call's.
    #[must_use]
    pub fn with_prior_trials(mut self, prior: u32) -> Self {
        self.trials_used = self.trials_used.max(prior);
        self
    }

    /// The width the lane uses for its next unit (always inside the range).
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.value
    }

    /// The range the machine moves inside.
    #[must_use]
    pub const fn range(&self) -> WidthRange {
        self.range
    }

    /// Whether the machine will move nothing more.
    #[must_use]
    pub const fn frozen(&self) -> bool {
        self.frozen
    }

    /// Trials started so far.
    #[must_use]
    pub const fn trials_used(&self) -> u32 {
        self.trials_used
    }

    /// Whether a direction is tabu.
    #[must_use]
    pub const fn tabu(&self, dir: TrialDir) -> bool {
        match dir {
            TrialDir::Up => self.tabu_up,
            TrialDir::Down => self.tabu_down,
            TrialDir::Switch | TrialDir::Halve => false,
        }
    }

    /// Windows closed so far.
    #[must_use]
    pub const fn windows(&self) -> u64 {
        self.windows
    }

    fn event(
        &self,
        dir: TrialDir,
        decision: TrialDecision,
        from: u64,
        metric: f64,
        baseline: f64,
        unit: u64,
        why: &str,
    ) -> TrialEvent {
        TrialEvent {
            rule: self.rule_id.clone(),
            site: self.site.clone(),
            knob: self.knob.clone(),
            dir,
            decision,
            from: from as f64,
            to: self.value as f64,
            metric,
            baseline,
            unit,
            window: self.windows,
            why: why.to_owned(),
        }
        .sanitized()
    }

    /// Report `why` once per machine (a refusal must not flood the log).
    fn once(&mut self, why: &'static str) -> bool {
        if self.reported.contains(&why) {
            false
        } else {
            self.reported.push(why);
            true
        }
    }

    /// Feed the site's cumulative counters at a unit boundary. A window
    /// closes every `window_units` units; each closed window may decide.
    /// Returns the events of this call (usually none).
    pub fn observe(&mut self, obs: WidthObs) -> Vec<TrialEvent> {
        let here = Mark {
            units: obs.units,
            closes: obs.closes,
            t: obs.t_s,
        };
        let Some(mark) = self.mark else {
            self.mark = Some(here);
            return Vec::new();
        };
        let units = obs.units.saturating_sub(mark.units);
        if units < self.rule.window_units {
            return Vec::new();
        }
        let closes = obs.closes.saturating_sub(mark.closes);
        let secs = obs.t_s - mark.t;
        self.mark = Some(here);
        self.windows = self.windows.saturating_add(1);
        let metric = if secs.is_finite() && secs > 0.0 {
            closes as f64 / secs
        } else {
            f64::NAN
        };
        let prev = self.prev_metric;
        self.prev_metric = metric.is_finite().then_some(metric);
        self.decide(metric, prev, units, secs, &obs)
    }

    fn decide(
        &mut self,
        m: f64,
        prev: Option<f64>,
        units: u64,
        secs: f64,
        obs: &WidthObs,
    ) -> Vec<TrialEvent> {
        let mut out = Vec::new();
        if self.frozen {
            return out;
        }
        let rule = self.rule;
        let lease = obs.lease_use.filter(|u| u.is_finite() && *u >= 0.0);
        let unit = obs.units;
        if let Some(mut p) = self.pending.take() {
            // A raise under memory pressure goes back at once (§5.3 guard).
            if p.dir == TrialDir::Up && lease.is_some_and(|u| u > rule.lease_guard) {
                let from = self.value;
                self.value = p.from;
                self.last_good = p.from;
                self.tabu_up = true;
                out.push(self.event(
                    p.dir,
                    TrialDecision::Reverted,
                    from,
                    m,
                    p.baseline,
                    unit,
                    "lease_guard",
                ));
                return out;
            }
            p.evals = p.evals.saturating_add(1);
            let kept = m.is_finite() && m > 0.0 && m >= p.baseline * (1.0 + rule.keep_margin);
            if kept {
                self.last_good = self.value;
                out.push(self.event(
                    p.dir,
                    TrialDecision::Kept,
                    p.from,
                    m,
                    p.baseline,
                    unit,
                    "beat_baseline",
                ));
                return out;
            }
            if !(m.is_finite() && m >= p.baseline) {
                p.falls = p.falls.saturating_add(1);
            }
            if p.dir == TrialDir::Up && p.falls >= 2 {
                // Two falls after a raise: back to the last good width, and
                // the machine freezes for the row.
                let from = self.value;
                self.value = self.last_good;
                self.tabu_up = true;
                self.frozen = true;
                out.push(self.event(
                    p.dir,
                    TrialDecision::Reverted,
                    from,
                    m,
                    p.baseline,
                    unit,
                    "two_falls_after_raise",
                ));
                out.push(self.event(
                    p.dir,
                    TrialDecision::Frozen,
                    self.value,
                    m,
                    p.baseline,
                    unit,
                    "two_falls_after_raise",
                ));
                return out;
            }
            if p.evals >= rule.eval_windows {
                let from = self.value;
                self.value = p.from;
                match p.dir {
                    TrialDir::Up => self.tabu_up = true,
                    TrialDir::Down => self.tabu_down = true,
                    TrialDir::Switch | TrialDir::Halve => {}
                }
                out.push(self.event(
                    p.dir,
                    TrialDecision::Reverted,
                    from,
                    m,
                    p.baseline,
                    unit,
                    "not_kept",
                ));
                return out;
            }
            self.pending = Some(p);
            return out;
        }
        // Review fix (router-r6): lease use > 85% with nothing pending is the
        // §5.2 "/2 on >= 10% fall OR lease > 85%" TRIGGER — it STARTS a
        // halving trial through the same §5.1 / §5.3 gate as a fall (>= 8
        // windows left, never in the last 25% of the lane, <= 2 trials per
        // knob, tabu), judged and kept or reverted like any trial. It used to
        // halve OUTSIDE the protocol (uncounted, in the last quarter, and on a
        // static lease — the reordered input-split loop's — at today's own
        // clamped value, i.e. with no pressure today's code does not already
        // bound). Only a PENDING raise is still reverted at once under
        // pressure (the §5.3 guard, above): that restores the pre-trial value.
        if !m.is_finite() {
            return out;
        }
        let pressure = lease.is_some_and(|u| u > rule.lease_guard);
        let rise =
            !pressure && prev.is_some_and(|p| m > 0.0 && m > p && m >= p * (1.0 + rule.raise_rise));
        let fall = prev.is_some_and(|p| p > 0.0 && m <= p * (1.0 - rule.fall_drop));
        let dir = if pressure || fall {
            TrialDir::Down
        } else if rise {
            TrialDir::Up
        } else {
            return out;
        };
        let prev = prev.unwrap_or(0.0);
        if self.tabu(dir) {
            if self.once(if dir == TrialDir::Up {
                "tabu_up"
            } else {
                "tabu_down"
            }) {
                out.push(self.event(dir, TrialDecision::Tabu, self.value, m, prev, unit, "tabu"));
            }
            return out;
        }
        let to = match dir {
            TrialDir::Up => {
                let Some(u) = lease else {
                    if self.once("lease_unknown") {
                        out.push(self.event(
                            dir,
                            TrialDecision::Blocked,
                            self.value,
                            m,
                            prev,
                            unit,
                            "lease_unknown",
                        ));
                    }
                    return out;
                };
                if !(u < rule.lease_raise_max) {
                    return out;
                }
                let to = self.value.saturating_mul(2);
                if to > self.range.max {
                    return out;
                }
                to
            }
            _ => {
                let to = self.value / 2;
                if to < self.range.min {
                    return out;
                }
                to
            }
        };
        if let Some(why) = start_refusal(
            units,
            secs,
            obs.t_s,
            obs.remaining_s,
            self.trials_used,
            rule.max_trials,
            rule.windows_left_min,
            rule.late_frac,
        ) {
            if self.once(why) {
                out.push(self.event(dir, TrialDecision::Blocked, self.value, m, prev, unit, why));
            }
            return out;
        }
        let from = self.value;
        self.last_good = from;
        self.value = to;
        self.trials_used = self.trials_used.saturating_add(1);
        self.pending = Some(Pending {
            dir,
            from,
            baseline: m,
            evals: 0,
            falls: 0,
        });
        out.push(self.event(
            dir,
            TrialDecision::Started,
            from,
            m,
            m,
            unit,
            if dir == TrialDir::Up {
                "rise"
            } else if pressure {
                "lease"
            } else {
                "fall"
            },
        ));
        out
    }
}

// ---------------------------------------------------------------------------
// rule.trial_lr_backoff: step-size backoff (never a raise)
// ---------------------------------------------------------------------------

/// `rule.trial_lr_backoff`'s constants (ROUTER_PROGRAM §5.2, learning rates).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LrRule {
    /// Consecutive rejected iterates that trigger a backoff.
    pub reject_run: u32,
    /// The backoff multiplier (`0 < factor < 1`).
    pub factor: f64,
    /// The floor is `lr0 / floor_div`.
    pub floor_div: f64,
    /// Iterates the backoff is judged over.
    pub eval_iterates: u32,
    /// Backoffs per row.
    pub max_trials: u32,
    /// No backoff starts past this fraction of the phase.
    pub late_frac: f64,
}

impl LrRule {
    /// The shipped constants: 3 rejected iterates => x 0.5, floor lr0 / 8,
    /// judged over the next 3 iterates.
    pub const SHIPPED: Self = Self {
        reject_run: 3,
        factor: 0.5,
        floor_div: 8.0,
        eval_iterates: 3,
        max_trials: TRIAL_MAX_PER_KNOB,
        late_frac: TRIAL_LATE_FRAC,
    };

    /// Whether the constants are inside the rule's domain (backoff only: the
    /// factor and the floor can only ever LOWER the step).
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.reject_run >= 1
            && self.factor.is_finite()
            && self.factor > 0.0
            && self.factor < 1.0
            && self.floor_div.is_finite()
            && self.floor_div >= 1.0
            && self.eval_iterates >= 1
            && (1..=TRIAL_MAX_PER_KNOB).contains(&self.max_trials)
            && self.late_frac.is_finite()
            && self.late_frac > 0.0
            && self.late_frac <= TRIAL_LATE_FRAC
    }

    /// The smallest multiplier the machine can reach.
    #[must_use]
    pub fn floor(self) -> f64 {
        1.0 / self.floor_div
    }
}

/// One ascent iterate, as the lane measured it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Iterate {
    /// The iteration index.
    pub index: u64,
    /// Whether the phase kept the iterate (its own acceptance rule).
    pub accepted: bool,
    /// The candidate's objective minus the best so far (negative when
    /// rejected).
    pub gain: f64,
    /// Seconds the iterate took.
    pub secs: f64,
    /// Fraction of the phase spent (`max` of iterations and time).
    pub done_frac: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LrPending {
    from: f64,
    baseline: f64,
    n: u32,
    gain: f64,
    secs: f64,
}

/// `rule.trial_lr_backoff`'s machine: a multiplier on the lane's own lr.
#[derive(Debug, Clone, PartialEq)]
pub struct LrBackoff {
    rule: LrRule,
    rule_id: String,
    site: String,
    knob: String,
    scale: f64,
    rejects: u32,
    recent: Vec<(f64, f64)>,
    pending: Option<LrPending>,
    tabu: bool,
    trials_used: u32,
    reported: Vec<&'static str>,
}

/// Gain per second of a run of iterates (zero time reads as a tiny span, so
/// a run of instantaneous iterates is still ordered by its gain).
fn gain_rate(gain: f64, secs: f64) -> f64 {
    let gain = finite_or_zero(gain);
    let secs = if secs.is_finite() && secs > 0.0 {
        secs
    } else {
        1e-9
    };
    gain / secs
}

impl LrBackoff {
    /// A machine at multiplier 1 (today's lr). `None` on invalid constants.
    #[must_use]
    pub fn new(rule: LrRule, rule_id: &str, site: &str, knob: &str) -> Option<Self> {
        rule.is_valid().then(|| Self {
            rule,
            rule_id: rule_id.to_owned(),
            site: site.to_owned(),
            knob: knob.to_owned(),
            scale: 1.0,
            rejects: 0,
            recent: Vec::new(),
            pending: None,
            tabu: false,
            trials_used: 0,
            reported: Vec::new(),
        })
    }

    /// The same machine with `prior` backoffs already spent on this knob in
    /// this row (the per-row budget of [`LrRule::max_trials`]).
    #[must_use]
    pub fn with_prior_trials(mut self, prior: u32) -> Self {
        self.trials_used = self.trials_used.max(prior);
        self
    }

    /// The multiplier on the lane's own lr for the next iterate (`1.0` =
    /// today's step; never above 1, never below the rule's floor).
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Whether the backoff direction became tabu.
    #[must_use]
    pub const fn tabu(&self) -> bool {
        self.tabu
    }

    /// Backoffs started so far.
    #[must_use]
    pub const fn trials_used(&self) -> u32 {
        self.trials_used
    }

    fn event(
        &self,
        decision: TrialDecision,
        from: f64,
        metric: f64,
        baseline: f64,
        unit: u64,
        why: &str,
    ) -> TrialEvent {
        TrialEvent {
            rule: self.rule_id.clone(),
            site: self.site.clone(),
            knob: self.knob.clone(),
            dir: TrialDir::Down,
            decision,
            from,
            to: self.scale,
            metric,
            baseline,
            unit,
            window: u64::from(self.trials_used),
            why: why.to_owned(),
        }
        .sanitized()
    }

    fn once(&mut self, why: &'static str) -> bool {
        if self.reported.contains(&why) {
            false
        } else {
            self.reported.push(why);
            true
        }
    }

    /// Feed one iterate. Returns this call's events.
    pub fn iterate(&mut self, it: Iterate) -> Vec<TrialEvent> {
        let mut out = Vec::new();
        let gain = finite_or_zero(it.gain);
        let secs = if it.secs.is_finite() && it.secs > 0.0 {
            it.secs
        } else {
            0.0
        };
        let run = self.rule.reject_run.max(1) as usize;
        self.recent.push((gain, secs));
        if self.recent.len() > run {
            let drop = self.recent.len() - run;
            self.recent.drain(..drop);
        }
        if let Some(mut p) = self.pending.take() {
            p.n = p.n.saturating_add(1);
            p.gain += gain;
            p.secs += secs;
            if p.n < self.rule.eval_iterates {
                self.pending = Some(p);
                return out;
            }
            let rate = gain_rate(p.gain, p.secs);
            self.rejects = 0;
            if rate < p.baseline {
                let from = self.scale;
                self.scale = p.from;
                self.tabu = true;
                out.push(self.event(
                    TrialDecision::Reverted,
                    from,
                    rate,
                    p.baseline,
                    it.index,
                    "gain_rate_fell",
                ));
            } else {
                out.push(self.event(
                    TrialDecision::Kept,
                    p.from,
                    rate,
                    p.baseline,
                    it.index,
                    "gain_rate_held",
                ));
            }
            return out;
        }
        if it.accepted {
            self.rejects = 0;
            return out;
        }
        self.rejects = self.rejects.saturating_add(1);
        if self.rejects < self.rule.reject_run {
            return out;
        }
        self.rejects = 0;
        let (gain_sum, secs_sum) = self
            .recent
            .iter()
            .fold((0.0f64, 0.0f64), |(g, s), (gi, si)| (g + gi, s + si));
        let baseline = gain_rate(gain_sum, secs_sum);
        if self.tabu {
            if self.once("tabu") {
                out.push(self.event(
                    TrialDecision::Tabu,
                    self.scale,
                    0.0,
                    baseline,
                    it.index,
                    "tabu",
                ));
            }
            return out;
        }
        if self.trials_used >= self.rule.max_trials {
            if self.once("trials_exhausted") {
                out.push(self.event(
                    TrialDecision::Blocked,
                    self.scale,
                    0.0,
                    baseline,
                    it.index,
                    "trials_exhausted",
                ));
            }
            return out;
        }
        if !(it.done_frac.is_finite() && it.done_frac < self.rule.late_frac) {
            if self.once("late") {
                out.push(self.event(
                    TrialDecision::Blocked,
                    self.scale,
                    0.0,
                    baseline,
                    it.index,
                    "late",
                ));
            }
            return out;
        }
        let to = (self.scale * self.rule.factor).max(self.rule.floor());
        if !(to < self.scale) {
            if self.once("floor") {
                out.push(self.event(
                    TrialDecision::Blocked,
                    self.scale,
                    0.0,
                    baseline,
                    it.index,
                    "floor",
                ));
            }
            return out;
        }
        let from = self.scale;
        self.scale = to;
        self.trials_used = self.trials_used.saturating_add(1);
        self.pending = Some(LrPending {
            from,
            baseline,
            n: 0,
            gain: 0.0,
            secs: 0.0,
        });
        out.push(self.event(
            TrialDecision::Started,
            from,
            baseline,
            baseline,
            it.index,
            "rejected_run",
        ));
        out
    }
}

// ---------------------------------------------------------------------------
// rule.trial_split_restart: the restart-form heuristic switch
// ---------------------------------------------------------------------------

/// `rule.trial_split_restart`'s constants (ROUTER_PROGRAM §5.2, split
/// heuristic).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RestartRule {
    /// The zero-lift split fraction over the two trailing windows must be
    /// strictly above this.
    pub zero_lift_min: f64,
    /// Volume FAR's flatness threshold (§4.3).
    pub volume_eps: f64,
    /// Work units per window.
    pub window_units: u64,
    /// Windows that must still fit for the restart.
    pub windows_left_min: f64,
    /// No restart past this fraction of the lane.
    pub late_frac: f64,
}

impl RestartRule {
    /// The shipped constants: FAR (Volume, two trailing windows) AND
    /// zero-lift fraction > 0.9 (collins: 98% of splits zero-lift).
    pub const SHIPPED: Self = Self {
        zero_lift_min: 0.9,
        volume_eps: project::VOLUME_FLAT_EPS,
        window_units: TRIAL_WINDOW_UNITS,
        windows_left_min: TRIAL_WINDOWS_LEFT_MIN,
        late_frac: TRIAL_LATE_FRAC,
    };

    /// Whether the constants are inside the rule's domain.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.zero_lift_min.is_finite()
            && self.zero_lift_min >= 0.9
            && self.zero_lift_min < 1.0
            && self.volume_eps.is_finite()
            && self.volume_eps >= 0.0
            && self.volume_eps <= project::VOLUME_FLAT_EPS
            && self.window_units >= TRIAL_WINDOW_UNITS
            && self.windows_left_min.is_finite()
            && self.windows_left_min >= TRIAL_WINDOWS_LEFT_MIN
            && self.late_frac.is_finite()
            && self.late_frac > 0.0
            && self.late_frac <= TRIAL_LATE_FRAC
    }
}

/// A child's bound LIFT over its parent counts as zero below this relative
/// tolerance: an outward-rounded bound that moved by a few ulps did not lift.
pub const ZERO_LIFT_REL_EPS: f64 = 1e-9;

/// Whether a child bounded at `child` (its margin, larger is closer to
/// closing) did NOT lift its parent's `parent` margin. A non-finite margin
/// is never counted (`None`).
#[must_use]
pub fn zero_lift(parent: f64, child: f64) -> Option<bool> {
    (parent.is_finite() && child.is_finite())
        .then(|| child <= parent + ZERO_LIFT_REL_EPS * parent.abs().max(1.0))
}

/// What a Volume lane reports at a unit boundary (cumulative counters of the
/// CURRENT tree, except time).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitObs {
    /// Work units completed so far.
    pub units: u64,
    /// Certified volume fraction of the current tree.
    pub closed_volume: f64,
    /// Open queue.
    pub open: u64,
    /// Children bounded so far whose parent bound is known.
    pub children: u64,
    /// Of those, children whose bound did not rise above their parent's.
    pub zero_lift: u64,
    /// Seconds since the lane started.
    pub t_s: f64,
    /// Seconds left before the lane's own deadline.
    pub remaining_s: Option<f64>,
}

/// `rule.trial_split_restart`'s machine: at most ONE restart per row.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitRestart {
    rule: RestartRule,
    rule_id: String,
    site: String,
    knob: String,
    from: String,
    to: String,
    mark: Option<SplitObs>,
    trailing: Vec<(Window, u64, u64)>,
    windows: u64,
    used: bool,
    reported: Vec<&'static str>,
}

impl SplitRestart {
    /// A machine that may switch heuristic `from` to `to` once. `None` on
    /// invalid constants or a switch to the same heuristic.
    #[must_use]
    pub fn new(
        rule: RestartRule,
        rule_id: &str,
        site: &str,
        knob: &str,
        from: &str,
        to: &str,
    ) -> Option<Self> {
        (rule.is_valid() && from != to).then(|| Self {
            rule,
            rule_id: rule_id.to_owned(),
            site: site.to_owned(),
            knob: knob.to_owned(),
            from: from.to_owned(),
            to: to.to_owned(),
            mark: None,
            trailing: Vec::new(),
            windows: 0,
            used: false,
            reported: Vec::new(),
        })
    }

    /// Whether the one restart was taken.
    #[must_use]
    pub const fn used(&self) -> bool {
        self.used
    }

    /// The heuristic the restart switches to.
    #[must_use]
    pub fn target(&self) -> &str {
        &self.to
    }

    fn once(&mut self, why: &'static str) -> bool {
        if self.reported.contains(&why) {
            false
        } else {
            self.reported.push(why);
            true
        }
    }

    fn event(&self, decision: TrialDecision, metric: f64, unit: u64, why: String) -> TrialEvent {
        TrialEvent {
            rule: self.rule_id.clone(),
            site: self.site.clone(),
            knob: self.knob.clone(),
            dir: TrialDir::Switch,
            decision,
            from: 0.0,
            to: f64::from(u8::from(self.used)),
            metric,
            baseline: self.rule.zero_lift_min,
            unit,
            window: self.windows,
            why,
        }
        .sanitized()
    }

    /// Feed the lane's counters at a unit boundary. `Some(Started)` means:
    /// RESTART NOW with [`Self::target`] — discard the tree and carry no fact
    /// of it into the new one. Any other event is a refusal record.
    pub fn observe(&mut self, obs: SplitObs) -> Option<TrialEvent> {
        if self.used {
            return None;
        }
        let Some(mark) = self.mark else {
            self.mark = Some(obs);
            return None;
        };
        if obs.units.saturating_sub(mark.units) < self.rule.window_units {
            return None;
        }
        let window = Window {
            work: obs.units.saturating_sub(mark.units),
            closes: 0,
            pruned: 0,
            open_before: mark.open,
            open_after: obs.open,
            level: 0,
            value_before: mark.closed_volume,
            value_after: obs.closed_volume,
            t0: mark.t_s,
            t1: obs.t_s,
        };
        let children = obs.children.saturating_sub(mark.children);
        let zero = obs.zero_lift.saturating_sub(mark.zero_lift);
        self.mark = Some(obs);
        self.windows = self.windows.saturating_add(1);
        self.trailing.push((window, children, zero));
        if self.trailing.len() > 2 {
            self.trailing.remove(0);
        }
        let [(w1, c1, z1), (w2, c2, z2)] = self.trailing.as_slice() else {
            return None;
        };
        if project::far_volume(w1, w2, self.rule.volume_eps) != FarClass::Far {
            return None;
        }
        let children = c1.saturating_add(*c2);
        if children == 0 {
            return None;
        }
        let frac = z1.saturating_add(*z2) as f64 / children as f64;
        if !(frac > self.rule.zero_lift_min) {
            return None;
        }
        if let Some(why) = start_refusal(
            w2.work,
            w2.t1 - w2.t0,
            obs.t_s,
            obs.remaining_s,
            0,
            1,
            self.rule.windows_left_min,
            self.rule.late_frac,
        ) {
            return self
                .once(why)
                .then(|| self.event(TrialDecision::Blocked, frac, obs.units, why.to_owned()));
        }
        self.used = true;
        Some(self.event(
            TrialDecision::Started,
            frac,
            obs.units,
            format!("far+zero_lift from={} to={}", self.from, self.to),
        ))
    }
}

// ---------------------------------------------------------------------------
// rule.trial_attack_halving: successive halving across a step-size pool
// ---------------------------------------------------------------------------

/// `rule.trial_attack_halving`'s constants (ROUTER_PROGRAM §5.2, attack
/// restarts / length).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HalvingRule {
    /// The step-size pool, arm 0 first: arm 0 MUST be today's step, so the
    /// first restart is today's restart.
    pub pool: [f64; 4],
    /// No elimination AND no untested step past this fraction of the lane:
    /// the machine freezes and its leader takes every restart (review fix;
    /// today's step while no elimination has been decided).
    pub late_frac: f64,
}

impl HalvingRule {
    /// The upfront APGD lane's initial box-relative step `0.25` (today's)
    /// and its halvings / doubling.
    pub const SHIPPED: Self = Self {
        pool: [0.25, 0.5, 0.125, 0.0625],
        late_frac: TRIAL_LATE_FRAC,
    };

    /// Whether the constants are inside the rule's domain: positive finite
    /// steps no larger than the whole box, distinct, and the lane's own step
    /// first.
    #[must_use]
    pub fn is_valid(self) -> bool {
        let steps_ok = self
            .pool
            .iter()
            .all(|s| s.is_finite() && *s > 0.0 && *s <= 1.0);
        let distinct = self
            .pool
            .iter()
            .enumerate()
            .all(|(i, a)| self.pool.iter().skip(i + 1).all(|b| a != b));
        steps_ok
            && distinct
            && self.late_frac.is_finite()
            && self.late_frac > 0.0
            && self.late_frac <= TRIAL_LATE_FRAC
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Arm {
    step: f64,
    restarts: u64,
    gain: f64,
    secs: f64,
}

impl Arm {
    fn rate(&self) -> f64 {
        gain_rate(self.gain, self.secs)
    }
}

/// `rule.trial_attack_halving`'s machine.
#[derive(Debug, Clone, PartialEq)]
pub struct Halving {
    rule: HalvingRule,
    rule_id: String,
    site: String,
    knob: String,
    arms: Vec<Arm>,
    alive: Vec<usize>,
    round: u32,
    queue: Vec<usize>,
    cursor: usize,
    outstanding: usize,
    frozen: bool,
    rotate: usize,
    restarts: u64,
}

impl Halving {
    /// A machine over the rule's pool. `None` on invalid constants.
    #[must_use]
    pub fn new(rule: HalvingRule, rule_id: &str, site: &str, knob: &str) -> Option<Self> {
        if !rule.is_valid() {
            return None;
        }
        let arms: Vec<Arm> = rule
            .pool
            .iter()
            .map(|&step| Arm {
                step,
                restarts: 0,
                gain: 0.0,
                secs: 0.0,
            })
            .collect();
        let alive: Vec<usize> = (0..arms.len()).collect();
        let mut out = Self {
            rule,
            rule_id: rule_id.to_owned(),
            site: site.to_owned(),
            knob: knob.to_owned(),
            arms,
            alive,
            round: 0,
            queue: Vec::new(),
            cursor: 0,
            outstanding: 0,
            frozen: false,
            rotate: 0,
            restarts: 0,
        };
        out.schedule_round();
        Some(out)
    }

    /// Round `r` gives every surviving arm `2^(r + 1)` restarts in a
    /// balanced order (forward, then reversed, ...), so on a lane that
    /// alternates its loss by restart parity each arm sees both parities.
    fn schedule_round(&mut self) {
        let per_arm = 1usize << (self.round.min(16) + 1);
        self.queue.clear();
        for j in 0..per_arm {
            if j % 2 == 0 {
                self.queue.extend(self.alive.iter().copied());
            } else {
                self.queue.extend(self.alive.iter().rev().copied());
            }
        }
        self.cursor = 0;
        self.outstanding = self.queue.len();
    }

    /// The surviving arms' step sizes.
    #[must_use]
    pub fn alive_steps(&self) -> Vec<f64> {
        self.alive.iter().map(|&a| self.arms[a].step).collect()
    }

    /// Whether eliminations stopped (late in the lane).
    #[must_use]
    pub const fn frozen(&self) -> bool {
        self.frozen
    }

    /// The arm a FROZEN machine exploits (review fix, router-r6): no new trial
    /// starts in the last quarter of the lane, so before any elimination it is
    /// arm 0 (today's step), and after one the leading survivor by
    /// improvement per second (ties keep the lower arm, today's first).
    fn leader(&self) -> usize {
        if self.round == 0 {
            return 0;
        }
        self.alive
            .iter()
            .copied()
            .min_by(|a, b| {
                self.arms[*b]
                    .rate()
                    .total_cmp(&self.arms[*a].rate())
                    .then_with(|| a.cmp(b))
            })
            .unwrap_or(0)
    }

    /// The arm and step size for the next restart. The first call returns
    /// arm 0, the lane's own step. Once frozen (late in the lane) every
    /// restart takes [`Self::leader`]: no untested step is tried there.
    pub fn next_arm(&mut self) -> (usize, f64) {
        let arm = if self.alive.len() == 1 {
            self.alive[0]
        } else if self.frozen {
            self.leader()
        } else if self.cursor >= self.queue.len() {
            let arm = self.alive[self.rotate % self.alive.len()];
            self.rotate = self.rotate.wrapping_add(1);
            arm
        } else {
            let arm = self.queue[self.cursor];
            self.cursor += 1;
            arm
        };
        (arm, self.arms[arm].step)
    }

    /// One restart of `arm` finished: its best-margin `improvement` over its
    /// own start, in `secs`, with `done_frac` of the lane spent. A round ends
    /// when every restart it scheduled reported; the better half by
    /// improvement per second survives (ties keep the lower arm, i.e. the
    /// lane's own step first).
    pub fn done_restart(
        &mut self,
        arm: usize,
        improvement: f64,
        secs: f64,
        done_frac: f64,
    ) -> Vec<TrialEvent> {
        let mut out = Vec::new();
        let Some(slot) = self.arms.get_mut(arm) else {
            return out;
        };
        slot.restarts = slot.restarts.saturating_add(1);
        slot.gain += finite_or_zero(improvement);
        if secs.is_finite() && secs > 0.0 {
            slot.secs += secs;
        }
        self.restarts = self.restarts.saturating_add(1);
        if self.alive.len() <= 1 || self.frozen {
            return out;
        }
        let before = self.alive.len();
        // Review fix (router-r6): the late check runs at EVERY restart, not
        // only at a round's end — a round still handing out untested steps
        // when the lane passes 75% would otherwise keep trying them (and,
        // frozen, rotate over every survivor) in the last quarter. Frozen,
        // the machine exploits its leader (`next_arm`), eliminating nothing.
        if !(done_frac.is_finite() && done_frac < self.rule.late_frac) {
            self.frozen = true;
            let leader = self.arms[self.leader()].step;
            out.push(
                TrialEvent {
                    rule: self.rule_id.clone(),
                    site: self.site.clone(),
                    knob: self.knob.clone(),
                    dir: TrialDir::Halve,
                    decision: TrialDecision::Frozen,
                    from: before as f64,
                    to: before as f64,
                    metric: 0.0,
                    baseline: 0.0,
                    unit: self.restarts,
                    window: u64::from(self.round),
                    why: format!(
                        "late alive={} leader={leader}",
                        steps_token(&self.alive_steps())
                    ),
                }
                .sanitized(),
            );
            return out;
        }
        self.outstanding = self.outstanding.saturating_sub(1);
        if self.outstanding > 0 || self.cursor < self.queue.len() {
            return out;
        }
        let mut ranked = self.alive.clone();
        ranked.sort_by(|a, b| {
            self.arms[*b]
                .rate()
                .total_cmp(&self.arms[*a].rate())
                .then_with(|| a.cmp(b))
        });
        let keep = before.div_ceil(2);
        let dropped: Vec<usize> = ranked[keep..].to_vec();
        let mut kept: Vec<usize> = ranked[..keep].to_vec();
        kept.sort_unstable();
        let best_kept = self.arms[ranked[0]].rate();
        let best_dropped = dropped.first().map_or(0.0, |&a| self.arms[a].rate());
        self.alive = kept;
        self.round = self.round.saturating_add(1);
        let kept_steps = self.alive_steps();
        let dropped_steps: Vec<f64> = dropped.iter().map(|&a| self.arms[a].step).collect();
        out.push(
            TrialEvent {
                rule: self.rule_id.clone(),
                site: self.site.clone(),
                knob: self.knob.clone(),
                dir: TrialDir::Halve,
                decision: TrialDecision::Kept,
                from: before as f64,
                to: self.alive.len() as f64,
                metric: best_kept,
                baseline: best_dropped,
                unit: self.restarts,
                window: u64::from(self.round),
                why: format!(
                    "kept={} dropped={}",
                    steps_token(&kept_steps),
                    steps_token(&dropped_steps)
                ),
            }
            .sanitized(),
        );
        if self.alive.len() > 1 {
            self.schedule_round();
        }
        out
    }
}

fn steps_token(steps: &[f64]) -> String {
    steps
        .iter()
        .map(|s| format!("{s}"))
        .collect::<Vec<_>>()
        .join(",")
}
