// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #router-r3 — O-STOP: THE IN-LANE STOP AUTHORITY (ROUTER_PROGRAM §5.1,
//! §5.2, §10 R3; layer L5 of §2.1).
//!
//! # What this is
//!
//! R2 gave every governed LANE one monotone deadline cell
//! ([`crate::deadline_authority`]). R3 acts one level down, INSIDE a lane: a
//! phase that has stopped paying ends early and hands the rest of its time to
//! the phase after it. The four acting optimizer rules each own one
//! [`StopSite`]:
//!
//! | site | rule | what may stop early | who receives the time |
//! |---|---|---|---|
//! | [`StopSite::AlphaAscent`] | `rule.governor` | the margin-row root alpha ascent, when an iteration's gain falls below `ratio` of its own best (after `min_samples`) | the tree (the next phase reads the same lane deadline) |
//! | [`StopSite::IntermAlpha`] | `rule.ia_yield` | the `#interm-alpha` candidate loop inside `backward_interm` (the baseline pass continues) | `backward_interm`'s own baseline pass, then the tree: the ascent reads a CONTRACT that excludes it ([`StopAuthority::credit_ascent`]) |
//! | [`StopSite::AttackUpfront`] | `rule.attack_defer` (CR-ATK) | the upfront DLR-APGD attack, at a restart boundary, when its best margin has not risen by more than the noise floor over half its slice | a LATE SAT RESERVE held out of the verifier and spent by the post-verifier attack — DEFERRED, never deleted |
//! | [`StopSite::RootPhase`] | `rule.root_admission` | a root tightening phase whose padded cost does not fit the window it would run in | whatever follows it in the root build |
//!
//! This module holds the ARMING (what the router committed) and the LEDGER
//! (what each rule did, and whether the time it freed reached its recipient).
//! The decisions themselves are the phases' own rules (`MarginalYield`, the
//! slope predicate `router::ostop::attack_flat`, `phase_window::admit`); the
//! authority only answers "is this rule armed, and with which constants".
//!
//! # One direction only
//!
//! Nothing here can extend a phase or a lane. An armed rule can only end a
//! phase EARLIER than its own cap; every cap and deadline stays exactly the
//! one its owner derives (the R2 cells are never written from here). The
//! ledger makes the hand-on checkable: every freed second is recorded with
//! the window of the phase that received it (`realized` = the recipient's
//! window at its start covers the freed time), and a deferred attack second
//! is recorded with the late attack budget that actually spent it.
//!
//! # Dark by default, fail open
//!
//! Unarmed — every run without `optimizer.mode: on` — every read returns
//! `None` / `false` / zero and every phase keeps exactly today's rule, so an
//! unarmed process is byte-identical to a binary without this module. A
//! router fault disarms it (`router::guard::fail_open`).
//!
//! # Soundness
//!
//! Scheduling only. Every iterate an ascent or a tightening pass produces is
//! a valid relaxation (alpha in `[0, 1]`, shrink-only boxes), and the chosen
//! one is certified exactly once by the phase's own unchanged path; stopping
//! earlier can only leave a LOOSER (still valid) bound. An attack only ever
//! proposes candidates; every one still passes the unchanged trusted-oracle
//! gate, and no `unsat` is ever concluded from an attack that stopped. A
//! skipped root phase leaves wider, still sound intermediate boxes. So a
//! wrong stop can cost a proof (or a sat), never mint a wrong verdict.
//!
//! # Never panics
//!
//! Every lock tolerates poison, every arithmetic step saturates, and every
//! ledger is bounded ([`MAX_EVENTS`]). It is read from lane threads.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Events one row may record per ledger (counted past it, not stored).
pub const MAX_EVENTS: usize = 64;

/// The smallest window fraction a slope rule may judge over: a plateau must
/// be observed over at least HALF the slice the phase was granted
/// (ROUTER_PROGRAM §4.2 CR-ATK: `W >= 0.5` of the slice), because sat rows
/// that are still climbing look flat over a shorter window.
pub const MIN_DEFER_WINDOW_FRAC: f64 = 0.5;

/// A phase that may stop early.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopSite {
    /// The margin-row root alpha ascent (`rule.governor`).
    AlphaAscent,
    /// The `#interm-alpha` candidate loop in `backward_interm` (`rule.ia_yield`).
    IntermAlpha,
    /// The upfront DLR-APGD attack lane (`rule.attack_defer`, CR-ATK).
    AttackUpfront,
    /// A root tightening phase of the graph engine (`rule.root_admission`).
    RootPhase,
}

impl StopSite {
    /// Every site, in rule order.
    pub const ALL: [Self; 4] = [
        Self::AlphaAscent,
        Self::IntermAlpha,
        Self::AttackUpfront,
        Self::RootPhase,
    ];

    /// The capability id of the phase (registry ids).
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Self::AlphaAscent => "margin-row.alpha",
            Self::IntermAlpha => "margin-row.backward_interm",
            Self::AttackUpfront => "attack.upfront_apgd",
            Self::RootPhase => "relu_split.graph.root",
        }
    }

    /// The rule that owns the site (registry ids).
    #[must_use]
    pub const fn rule(self) -> &'static str {
        match self {
            Self::AlphaAscent => "rule.governor",
            Self::IntermAlpha => "rule.ia_yield",
            Self::AttackUpfront => "rule.attack_defer",
            Self::RootPhase => "rule.root_admission",
        }
    }
}

/// RELATIVE SELF-DECAY arming (`rule.governor`, `rule.ia_yield`): yield when
/// a unit's gain falls below `ratio` of the phase's OWN best, once at least
/// `min_samples` units were observed. Unit-free and scale-free.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RatioRule {
    /// `0 < ratio < 1`.
    pub ratio: f64,
    /// `>= 1`.
    pub min_samples: u32,
}

impl RatioRule {
    /// Whether the constants are inside the rule's domain.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.ratio.is_finite() && self.ratio > 0.0 && self.ratio < 1.0 && self.min_samples >= 1
    }
}

/// Slope-plateau DEFER arming (`rule.attack_defer`, CR-ATK).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DeferRule {
    /// Window as a fraction of the lane's slice, `[MIN_DEFER_WINDOW_FRAC, 1]`.
    pub window_frac: f64,
    /// The smallest unspent time worth deferring, milliseconds: the late
    /// consumer's own minimum budget (below it the late attack would not
    /// start, so a defer would be a cut).
    pub min_defer_ms: u64,
}

impl DeferRule {
    /// Whether the constants are inside the rule's domain.
    #[must_use]
    pub fn is_valid(self) -> bool {
        self.window_frac.is_finite()
            && self.window_frac >= MIN_DEFER_WINDOW_FRAC
            && self.window_frac <= 1.0
            && self.min_defer_ms > 0
    }
}

/// What the router committed for one row (`router::ostop::commit`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StopDirectives {
    /// `rule.governor`.
    pub governor: Option<RatioRule>,
    /// `rule.ia_yield` (implies the ascent contract).
    pub ia_yield: Option<RatioRule>,
    /// `rule.attack_defer`.
    pub attack_defer: Option<DeferRule>,
    /// `rule.root_admission`.
    pub root_admission: bool,
}

impl StopDirectives {
    /// Nothing armed.
    pub const NONE: Self = Self {
        governor: None,
        ia_yield: None,
        attack_defer: None,
        root_admission: false,
    };

    /// Whether any rule is armed.
    #[must_use]
    pub fn any(&self) -> bool {
        self.governor.is_some()
            || self.ia_yield.is_some()
            || self.attack_defer.is_some()
            || self.root_admission
    }
}

/// Why directives were refused (the authority then stays unarmed: today).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopArmRefusal {
    /// A ratio rule's constants are outside `0 < ratio < 1`, `min_samples >= 1`.
    RatioOutOfRange,
    /// The defer window is below half the slice, above it, or the minimum
    /// defer is zero.
    DeferOutOfRange,
}

/// One phase that stopped early.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YieldEvent {
    /// The site.
    pub site: StopSite,
    /// The caller's own tag (`[alpha-opt]`, `[alpha-opt:untrusted]`, ...).
    pub tag: String,
    /// The unit at which it stopped (iteration, layer).
    pub unit: u64,
    /// The phase's budget, milliseconds.
    pub budget_ms: u64,
    /// What it used, milliseconds.
    pub used_ms: u64,
    /// `budget - used`, milliseconds: the time it handed downstream.
    pub freed_ms: u64,
    /// The unit's rate when it fired.
    pub rate: f64,
    /// The phase's own best rate.
    pub best: f64,
    /// `ratio * best`.
    pub floor: f64,
}

/// One freed slice reaching the phase that follows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StopHandOn {
    /// The site that stopped.
    pub from: StopSite,
    /// The rule.
    pub rule: String,
    /// The receiving phase.
    pub to: String,
    /// Time the stop freed, milliseconds.
    pub freed_ms: u64,
    /// The recipient's window at its start (`None` = unbounded).
    pub window_ms: Option<u64>,
    /// `min(freed, window)`: what the recipient could actually spend.
    pub credited_ms: u64,
    /// Whether the recipient's window covers the freed time.
    pub realized: bool,
}

/// One upfront-attack DEFER (CR-ATK).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeferEvent {
    /// The restart boundary at which it deferred.
    pub restart: u64,
    /// The lane's slice, milliseconds.
    pub slice_ms: u64,
    /// Elapsed in the slice, milliseconds.
    pub elapsed_ms: u64,
    /// Unspent slice handed to the late reserve, milliseconds: what the lane
    /// would still have spent today — `min(deadline - now, its remaining
    /// restart cap at the shortest restart observed)` for a restart-capped
    /// lane (#router-r3 review), never more than its slice has left.
    pub unspent_ms: u64,
    /// Best margin over the whole lane (a running max).
    pub best_margin: f64,
    /// Its rise over the trailing window.
    pub rise: f64,
    /// The noise floor the rise had to beat.
    pub eps: f64,
    /// The trailing window, milliseconds.
    pub window_ms: u64,
}

/// What happened to a deferred slice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeferOutcome {
    /// Unspent time the defer carried, milliseconds.
    pub deferred_ms: u64,
    /// What the late reserve held out of the verifier, milliseconds.
    pub held_ms: u64,
    /// The late attack's budget when it started (`None`: it never started).
    pub late_budget_ms: Option<u64>,
    /// `true` when the late attack had at least the deferred time.
    pub realized: bool,
    /// `true` when the row was decided before the late attack could run (a
    /// decided row needs no further attack: not a loss).
    pub moot: bool,
    /// Why (stable token).
    pub reason: String,
}

/// One root-phase admission consult, with the predicate's inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootAdmissionEvent {
    /// The phase.
    pub phase: String,
    /// Its own predicted cost, milliseconds.
    pub predicted_ms: u64,
    /// The padded prediction (`phase_window`'s safety margin), milliseconds.
    pub padded_ms: u64,
    /// The remaining window it would run in, milliseconds.
    pub remaining_ms: u64,
    /// Where that window came from (`verifier_cell` | `instance`).
    pub source: String,
    /// The affordable share of it, milliseconds.
    pub affordable_ms: u64,
    /// Whether the phase runs.
    pub admitted: bool,
}

/// The authority, as recorded in the flight `router` record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StopSnapshot {
    /// Whether it was armed when the snapshot was taken.
    pub armed: bool,
    /// The directives it held.
    pub directives: Option<StopDirectives>,
    /// Every early stop, in order.
    pub yields: Vec<YieldEvent>,
    /// Every freed slice that reached a recipient, in order.
    pub hand_ons: Vec<StopHandOn>,
    /// The interm credit the ascent's contract excluded, milliseconds.
    pub ascent_credit_ms: u64,
    /// Every defer.
    pub defers: Vec<DeferEvent>,
    /// What happened to each deferred slice.
    pub defer_outcomes: Vec<DeferOutcome>,
    /// Every root-phase admission consult.
    pub root_admissions: Vec<RootAdmissionEvent>,
    /// Events past [`MAX_EVENTS`] (counted, not recorded).
    pub dropped: u64,
}

#[derive(Debug, Default)]
struct Ledger {
    directives: Option<StopDirectives>,
    yields: Vec<YieldEvent>,
    /// Index of the first yield not yet handed to a recipient.
    pending_from: usize,
    hand_ons: Vec<StopHandOn>,
    defers: Vec<DeferEvent>,
    /// A defer awaiting its late reserve.
    pending_defer: Option<DeferEvent>,
    /// The held reserve awaiting its late attack: `(deferred_ms, held_ms)`.
    held: Option<(u64, u64)>,
    defer_outcomes: Vec<DeferOutcome>,
    root_admissions: Vec<RootAdmissionEvent>,
    dropped: u64,
}

impl Ledger {
    fn push<T>(list: &mut Vec<T>, dropped: &mut u64, item: T) {
        if list.len() < MAX_EVENTS {
            list.push(item);
        } else {
            *dropped = dropped.saturating_add(1);
        }
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// The stop authority. One per process ([`global`]); instantiable for tests.
#[derive(Debug)]
pub struct StopAuthority {
    armed: AtomicBool,
    /// Interm-phase time the ascent's contract excludes, nanoseconds.
    ascent_credit_ns: AtomicU64,
    /// Total credit ever excluded this row (for the record), nanoseconds.
    ascent_credit_total_ns: AtomicU64,
    ledger: Mutex<Ledger>,
}

impl Default for StopAuthority {
    fn default() -> Self {
        Self::new()
    }
}

impl StopAuthority {
    /// An unarmed authority.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            ascent_credit_ns: AtomicU64::new(0),
            ascent_credit_total_ns: AtomicU64::new(0),
            ledger: Mutex::new(Ledger {
                directives: None,
                yields: Vec::new(),
                pending_from: 0,
                hand_ons: Vec::new(),
                defers: Vec::new(),
                pending_defer: None,
                held: None,
                defer_outcomes: Vec::new(),
                root_admissions: Vec::new(),
                dropped: 0,
            }),
        }
    }

    fn ledger(&self) -> std::sync::MutexGuard<'_, Ledger> {
        self.ledger.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Arm the row's directives. Clears every ledger first. Refuses (and stays
    /// unarmed) when a rule's constants are outside its domain; an empty set
    /// of directives arms nothing.
    pub fn arm(&self, directives: StopDirectives) -> Result<(), StopArmRefusal> {
        if directives.governor.is_some_and(|rule| !rule.is_valid())
            || directives.ia_yield.is_some_and(|rule| !rule.is_valid())
        {
            return Err(StopArmRefusal::RatioOutOfRange);
        }
        if directives.attack_defer.is_some_and(|rule| !rule.is_valid()) {
            return Err(StopArmRefusal::DeferOutOfRange);
        }
        self.disarm();
        self.ascent_credit_ns.store(0, Ordering::Release);
        self.ascent_credit_total_ns.store(0, Ordering::Release);
        {
            let mut ledger = self.ledger();
            *ledger = Ledger {
                directives: Some(directives),
                ..Ledger::default()
            };
        }
        if directives.any() {
            self.armed.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// Disarm: every read returns today's value from here on. The ledger is
    /// kept for the snapshot.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
        self.ascent_credit_ns.store(0, Ordering::Release);
    }

    /// Whether any rule is armed.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    /// The armed directives.
    #[must_use]
    pub fn directives(&self) -> Option<StopDirectives> {
        if !self.is_armed() {
            return None;
        }
        self.ledger().directives
    }

    /// `rule.governor`'s arming.
    #[must_use]
    pub fn governor(&self) -> Option<RatioRule> {
        self.directives().and_then(|d| d.governor)
    }

    /// `rule.ia_yield`'s arming.
    #[must_use]
    pub fn ia_yield(&self) -> Option<RatioRule> {
        self.directives().and_then(|d| d.ia_yield)
    }

    /// `rule.attack_defer`'s arming.
    #[must_use]
    pub fn attack_defer(&self) -> Option<DeferRule> {
        self.directives().and_then(|d| d.attack_defer)
    }

    /// Whether `rule.root_admission` is armed.
    #[must_use]
    pub fn root_admission(&self) -> bool {
        self.directives().is_some_and(|d| d.root_admission)
    }

    /// Record an early stop. A no-op unless the site's rule is armed.
    pub fn record_yield(&self, event: YieldEvent) {
        let Some(directives) = self.directives() else {
            return;
        };
        let armed = match event.site {
            StopSite::AlphaAscent => directives.governor.is_some(),
            StopSite::IntermAlpha => directives.ia_yield.is_some(),
            StopSite::AttackUpfront => directives.attack_defer.is_some(),
            StopSite::RootPhase => directives.root_admission,
        };
        if !armed {
            return;
        }
        // Records are serialized into the flight record and replayed: a
        // non-finite rate would serialize as null and fail the replay.
        let finite = |x: f64| if x.is_finite() { x } else { 0.0 };
        let event = YieldEvent {
            rate: finite(event.rate),
            best: finite(event.best),
            floor: finite(event.floor),
            ..event
        };
        let mut ledger = self.ledger();
        let ledger = &mut *ledger;
        Ledger::push(&mut ledger.yields, &mut ledger.dropped, event);
    }

    /// The phase `to` starts with `window` left (`None`: unbounded). Every
    /// ascent / interm yield recorded since the last call is handed to it:
    /// credited `min(freed, window)`, realized when the window covers the
    /// freed time. Returns the new hand-ons (empty when unarmed or none).
    pub fn downstream_start(&self, to: &str, window: Option<Duration>) -> Vec<StopHandOn> {
        if !self.is_armed() {
            return Vec::new();
        }
        let window_ms = window.map(millis);
        let mut ledger = self.ledger();
        let ledger = &mut *ledger;
        let start = ledger.pending_from.min(ledger.yields.len());
        let mut out = Vec::new();
        for event in ledger.yields.get(start..).unwrap_or(&[]) {
            if !matches!(event.site, StopSite::AlphaAscent | StopSite::IntermAlpha) {
                continue;
            }
            let credited = window_ms.map_or(event.freed_ms, |w| event.freed_ms.min(w));
            out.push(StopHandOn {
                from: event.site,
                rule: event.site.rule().to_owned(),
                to: to.to_owned(),
                freed_ms: event.freed_ms,
                window_ms,
                credited_ms: credited,
                realized: credited == event.freed_ms,
            });
        }
        ledger.pending_from = ledger.yields.len();
        for hand_on in &out {
            Ledger::push(&mut ledger.hand_ons, &mut ledger.dropped, hand_on.clone());
        }
        out
    }

    /// `rule.ia_yield`: the interm phase ended `credit` before its own budget
    /// after its candidate loop yielded. The ascent's contract excludes it, so
    /// the freed time cannot inflate the ascent's share of the remaining
    /// window (it reaches the tree). A no-op unless `rule.ia_yield` is armed.
    pub fn credit_ascent(&self, credit: Duration) {
        if self.ia_yield().is_none() {
            return;
        }
        let ns = u64::try_from(credit.as_nanos()).unwrap_or(u64::MAX);
        for cell in [&self.ascent_credit_ns, &self.ascent_credit_total_ns] {
            // Saturating, never wrapping: a wrapped credit would GROW the
            // ascent's contract instead of shrinking it.
            let mut held = cell.load(Ordering::Acquire);
            loop {
                match cell.compare_exchange_weak(
                    held,
                    held.saturating_add(ns),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(actual) => held = actual,
                }
            }
        }
    }

    /// The ascent's contract credit, consumed (zero unless `rule.ia_yield`
    /// is armed and its phase freed time). The caller SUBTRACTS it from the
    /// remaining window before applying its own share: the ascent can only
    /// get less than today's rule would give it, never more.
    pub fn take_ascent_credit(&self) -> Duration {
        if !self.is_armed() {
            return Duration::ZERO;
        }
        Duration::from_nanos(self.ascent_credit_ns.swap(0, Ordering::AcqRel))
    }

    /// CR-ATK: the upfront attack deferred. A no-op unless armed.
    pub fn record_defer(&self, event: DeferEvent) {
        if self.attack_defer().is_none() {
            return;
        }
        let finite = |x: f64| if x.is_finite() { x } else { 0.0 };
        let event = DeferEvent {
            best_margin: finite(event.best_margin),
            rise: finite(event.rise),
            eps: finite(event.eps),
            ..event
        };
        let mut ledger = self.ledger();
        let ledger = &mut *ledger;
        ledger.pending_defer = Some(event.clone());
        Ledger::push(&mut ledger.defers, &mut ledger.dropped, event);
    }

    /// The defer awaiting its late reserve, consumed.
    pub fn take_pending_defer(&self) -> Option<DeferEvent> {
        self.attack_defer()?;
        self.ledger().pending_defer.take()
    }

    /// The late reserve held `held` out of the verifier for a defer carrying
    /// `deferred`.
    pub fn note_defer_held(&self, deferred: Duration, held: Duration) {
        if !self.is_armed() {
            return;
        }
        self.ledger().held = Some((millis(deferred), millis(held)));
    }

    /// The held reserve awaiting its late attack: `(deferred, held)`.
    #[must_use]
    pub fn held_reserve(&self) -> Option<(Duration, Duration)> {
        if !self.is_armed() {
            return None;
        }
        self.ledger()
            .held
            .map(|(d, h)| (Duration::from_millis(d), Duration::from_millis(h)))
    }

    /// Close a deferred slice with its outcome; clears the held reserve.
    pub fn record_defer_outcome(&self, outcome: DeferOutcome) {
        if !self.is_armed() {
            return;
        }
        let mut ledger = self.ledger();
        let ledger = &mut *ledger;
        ledger.held = None;
        Ledger::push(&mut ledger.defer_outcomes, &mut ledger.dropped, outcome);
    }

    /// Record one root-phase admission consult. A no-op unless armed.
    pub fn record_root_admission(&self, event: RootAdmissionEvent) {
        if !self.root_admission() {
            return;
        }
        let mut ledger = self.ledger();
        let ledger = &mut *ledger;
        Ledger::push(&mut ledger.root_admissions, &mut ledger.dropped, event);
    }

    /// The authority as recorded (whether or not armed now).
    #[must_use]
    pub fn snapshot(&self) -> StopSnapshot {
        let ledger = self.ledger();
        StopSnapshot {
            armed: self.is_armed(),
            directives: ledger.directives,
            yields: ledger.yields.clone(),
            hand_ons: ledger.hand_ons.clone(),
            ascent_credit_ms: self.ascent_credit_total_ns.load(Ordering::Acquire) / 1_000_000,
            defers: ledger.defers.clone(),
            defer_outcomes: ledger.defer_outcomes.clone(),
            root_admissions: ledger.root_admissions.clone(),
            dropped: ledger.dropped,
        }
    }
}

static GLOBAL: StopAuthority = StopAuthority::new();

/// The process's one stop authority. `ny vnncomp` runs one row per process.
#[must_use]
pub fn global() -> &'static StopAuthority {
    &GLOBAL
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOV: RatioRule = RatioRule {
        ratio: 0.15,
        min_samples: 5,
    };
    const IA: RatioRule = RatioRule {
        ratio: 0.15,
        min_samples: 2,
    };
    const DEFER: DeferRule = DeferRule {
        window_frac: 0.5,
        min_defer_ms: 5_000,
    };
    const ALL: StopDirectives = StopDirectives {
        governor: Some(GOV),
        ia_yield: Some(IA),
        attack_defer: Some(DEFER),
        root_admission: true,
    };

    fn ascent_yield(freed_ms: u64) -> YieldEvent {
        YieldEvent {
            site: StopSite::AlphaAscent,
            tag: "[alpha-opt]".to_owned(),
            unit: 12,
            budget_ms: 20_000,
            used_ms: 20_000 - freed_ms,
            freed_ms,
            rate: 0.0009,
            best: 0.0079,
            floor: 0.0012,
        }
    }

    #[test]
    fn unarmed_stop_authority_answers_nothing() {
        let a = StopAuthority::new();
        assert!(!a.is_armed());
        assert_eq!(a.directives(), None);
        assert_eq!(a.governor(), None);
        assert_eq!(a.ia_yield(), None);
        assert_eq!(a.attack_defer(), None);
        assert!(!a.root_admission());
        a.record_yield(ascent_yield(5_000));
        a.credit_ascent(Duration::from_secs(3));
        assert_eq!(a.take_ascent_credit(), Duration::ZERO);
        assert_eq!(
            a.downstream_start("margin-row.tree", Some(Duration::from_secs(60)))
                .len(),
            0
        );
        assert_eq!(a.take_pending_defer(), None);
        assert_eq!(a.held_reserve(), None);
        let snap = a.snapshot();
        assert!(snap.yields.is_empty() && snap.hand_ons.is_empty());
        assert_eq!(snap.ascent_credit_ms, 0);
        // Arming NOTHING arms nothing.
        a.arm(StopDirectives::NONE)
            .expect("empty directives are valid");
        assert!(!a.is_armed());
    }

    #[test]
    fn stop_arm_refuses_constants_outside_their_domain() {
        let a = StopAuthority::new();
        for bad in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
            let d = StopDirectives {
                governor: Some(RatioRule {
                    ratio: bad,
                    min_samples: 5,
                }),
                ..StopDirectives::NONE
            };
            assert_eq!(a.arm(d), Err(StopArmRefusal::RatioOutOfRange), "{bad}");
        }
        let zero_samples = StopDirectives {
            ia_yield: Some(RatioRule {
                ratio: 0.15,
                min_samples: 0,
            }),
            ..StopDirectives::NONE
        };
        assert_eq!(a.arm(zero_samples), Err(StopArmRefusal::RatioOutOfRange));
        for frac in [0.49, 1.01, f64::NAN] {
            let d = StopDirectives {
                attack_defer: Some(DeferRule {
                    window_frac: frac,
                    min_defer_ms: 5_000,
                }),
                ..StopDirectives::NONE
            };
            assert_eq!(a.arm(d), Err(StopArmRefusal::DeferOutOfRange), "{frac}");
        }
        assert!(!a.is_armed(), "a refused arming arms nothing");
        a.arm(ALL).expect("positive control");
        assert!(a.is_armed());
        assert_eq!(a.governor(), Some(GOV));
        a.disarm();
        assert_eq!(a.governor(), None);
    }

    /// Freed time is handed to the phase that starts next, exactly once, and
    /// `realized` says whether that phase's window covers it.
    #[test]
    fn freed_time_reaches_the_next_phase_once_and_is_realized() {
        let a = StopAuthority::new();
        a.arm(ALL).expect("arm");
        a.record_yield(ascent_yield(7_000));
        let hand = a.downstream_start("margin-row.tree", Some(Duration::from_secs(60)));
        assert_eq!(hand.len(), 1);
        assert_eq!(hand[0].freed_ms, 7_000);
        assert_eq!(hand[0].credited_ms, 7_000);
        assert!(hand[0].realized);
        assert!(
            a.downstream_start("margin-row.tree", Some(Duration::from_secs(60)))
                .is_empty(),
            "a freed slice is handed on exactly once"
        );
        // A window that cannot hold the freed time credits only what fits.
        a.record_yield(ascent_yield(9_000));
        let short = a.downstream_start("margin-row.tree", Some(Duration::from_secs(4)));
        assert_eq!(short[0].credited_ms, 4_000);
        assert!(!short[0].realized);
        // Conservation over the ledger: credited never exceeds freed.
        let snap = a.snapshot();
        for h in &snap.hand_ons {
            assert!(h.credited_ms <= h.freed_ms);
        }
    }

    /// The ascent contract only ever SHRINKS the ascent: the credit is
    /// consumed once, and only `rule.ia_yield` can grant it.
    #[test]
    fn ascent_credit_is_consumed_once_and_only_under_ia_yield() {
        let a = StopAuthority::new();
        a.arm(StopDirectives {
            ia_yield: None,
            ..ALL
        })
        .expect("arm");
        a.credit_ascent(Duration::from_secs(3));
        assert_eq!(
            a.take_ascent_credit(),
            Duration::ZERO,
            "without rule.ia_yield nothing is credited"
        );
        a.arm(ALL).expect("arm");
        a.credit_ascent(Duration::from_secs(3));
        a.credit_ascent(Duration::from_millis(500));
        assert_eq!(a.take_ascent_credit(), Duration::from_millis(3_500));
        assert_eq!(a.take_ascent_credit(), Duration::ZERO, "consumed once");
        assert_eq!(a.snapshot().ascent_credit_ms, 3_500);
    }

    #[test]
    fn a_defer_is_held_once_and_closed_with_its_outcome() {
        let a = StopAuthority::new();
        a.arm(ALL).expect("arm");
        a.record_defer(DeferEvent {
            restart: 9,
            slice_ms: 30_000,
            elapsed_ms: 16_000,
            unspent_ms: 14_000,
            best_margin: -0.033,
            rise: 0.0,
            eps: 1e-4,
            window_ms: 15_000,
        });
        let pending = a.take_pending_defer().expect("pending");
        assert_eq!(pending.unspent_ms, 14_000);
        assert_eq!(a.take_pending_defer(), None, "held once");
        a.note_defer_held(Duration::from_secs(14), Duration::from_secs(14));
        assert_eq!(
            a.held_reserve(),
            Some((Duration::from_secs(14), Duration::from_secs(14)))
        );
        a.record_defer_outcome(DeferOutcome {
            deferred_ms: 14_000,
            held_ms: 14_000,
            late_budget_ms: Some(15_200),
            realized: true,
            moot: false,
            reason: "late_attack_started".to_owned(),
        });
        assert_eq!(a.held_reserve(), None);
        let snap = a.snapshot();
        assert_eq!(snap.defers.len(), 1);
        assert_eq!(snap.defer_outcomes.len(), 1);
    }

    #[test]
    fn ledgers_are_bounded_and_count_what_they_drop() {
        let a = StopAuthority::new();
        a.arm(ALL).expect("arm");
        for _ in 0..(MAX_EVENTS + 10) {
            a.record_yield(ascent_yield(1));
        }
        let snap = a.snapshot();
        assert_eq!(snap.yields.len(), MAX_EVENTS);
        assert_eq!(snap.dropped, 10);
    }

    /// Every recorded value survives the flight record's JSON round trip (a
    /// non-finite rate would serialize as null and break the replay).
    #[test]
    fn stop_snapshot_round_trips_through_json() {
        let a = StopAuthority::new();
        a.arm(ALL).expect("arm");
        a.record_yield(YieldEvent {
            rate: f64::NAN,
            best: f64::INFINITY,
            ..ascent_yield(3_000)
        });
        a.record_defer(DeferEvent {
            restart: 1,
            slice_ms: 1,
            elapsed_ms: 1,
            unspent_ms: 1,
            best_margin: f64::NEG_INFINITY,
            rise: f64::NAN,
            eps: 1e-4,
            window_ms: 1,
        });
        let snap = a.snapshot();
        let text = serde_json::to_string(&snap).expect("serialize");
        let back: StopSnapshot = serde_json::from_str(&text).expect("deserialize");
        assert_eq!(back, snap);
    }

    #[test]
    fn stop_extreme_values_never_panic() {
        let a = StopAuthority::new();
        a.arm(ALL).expect("arm");
        a.record_yield(YieldEvent {
            freed_ms: u64::MAX,
            budget_ms: u64::MAX,
            used_ms: 0,
            rate: f64::NAN,
            best: f64::INFINITY,
            floor: f64::NEG_INFINITY,
            ..ascent_yield(0)
        });
        let _ = a.downstream_start("x", Some(Duration::MAX));
        let _ = a.downstream_start("x", None);
        a.credit_ascent(Duration::MAX);
        a.credit_ascent(Duration::MAX);
        let _ = a.take_ascent_credit();
        a.note_defer_held(Duration::MAX, Duration::MAX);
        let _ = a.held_reserve();
        let _ = a.snapshot();
    }
}
