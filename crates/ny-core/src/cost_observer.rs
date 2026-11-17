// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #cost-observer — measured per-call costs for invariant I1.
//!
//! [`crate::phase_window`] states the rule this module exists to make
//! satisfiable: *"`predicted` is the phase's own cost estimate **on this host**
//! — from a measured rate, not a constant."* It takes `predicted: Duration` as
//! an argument. Analytic cost producers include:
//!
//! * `#forward-linear-cost-gate` multiplies a MAC count by a probed `GMAC/s`;
//! * `WalkCostModel` does the same for `graph_alpha` walks.
//!
//! This module handles repeated seams without a MAC model. It measures the
//! seam's **own** per-call cost as the run proceeds and derives the budget from
//! those observations plus the live remaining wall.
//!
//! ## Why a constant cannot work here — the measurement that motivated this
//!
//! `bab.branching.input_split.child_bound_max_secs` (`#cgan-dark-tail` FIX-1),
//! A/B'd 2026-08-26 at the official 1200 s on the two 1200 s cgan rows with
//! `cap = 20.0`:
//!
//! | row | result at cap = 20 s | reading |
//! |---|---|---|
//! | `nonlinear_activations` | explored `2 -> 957`, depth `1 -> 100`, best_gap `-0.6548 -> -0.0016`, ZERO collection truncations | **20 s is right** — it removes pure waste |
//! | `upsample` | explored `1 -> 19`, best_gap UNCHANGED, 37 truncations cutting a 27-layer intermediate collection to 5-6 (22-23 IBP-filled), every call pinned at the cap | **20 s is far too short** |
//!
//! One number, two sibling rows of the same benchmark, right for one and an
//! order of magnitude wrong for the other. The distinguishing fact is not
//! knowable in advance and is not written anywhere in the instance: it is the
//! seam's per-call cost *on this row on this host*, which differs by ~3 orders
//! of magnitude between the two (`0.13 s` median vs `>= 300 s`).
//!
//! ## The rule
//!
//! A call may run for a multiple of what a call on **this** row typically
//! costs, never more than a fraction of what is left, and the allowance grows
//! while the evidence says the allowance is too small:
//!
//! ```text
//! slice = min( headroom x median(completed costs) , share x remaining ) x growth^consecutive_truncated
//! ```
//!
//! Every parameter in it is **dimensionless**. That is the property that makes
//! this different from a constant: nothing in the policy has to know how long a
//! bound takes, what the budget is, or how fast the host is. Those three facts
//! all arrive by measurement.
//!
//! ## The two regimes it has to reproduce, and does
//!
//! * *typical call cheap, rare pathological tail* (`nonlinear_activations`:
//!   median `0.13 s`, tail `70-285 s`) — the measured term governs and lands on
//!   `headroom x 0.13 s`, i.e. the same order as the 20 s that was measured
//!   good. Waste is cut, normal calls are untouched.
//! * *every call genuinely expensive* (`upsample`) — the first probe truncates,
//!   nothing has ever completed, and [`Slice::Ceiling`] hands back the caller's
//!   own ceiling: **byte-for-byte the pre-calibration behaviour**. The observer
//!   spends one `probe_share` slice discovering that and then gets out of the
//!   way. It is a state predicate, not a latch — the moment any call completes,
//!   calibration resumes.
//!
//! ## Soundness
//!
//! This module produces a **wall-clock allowance and nothing else**. It reads
//! durations and emits durations; it never sees a bound, a domain, an alpha or
//! a verdict, and no consumer may use its output for anything but composing an
//! EARLIER deadline (`min(ceiling, now + slice)`). A shorter budget can only
//! make a bound looser or fail the call outright — never tighter — so no
//! allowance this module can emit, including a wrong one, can turn a `timeout`
//! into a wrong verdict. [`Slice`] deliberately has no variant that can EXTEND
//! a caller's ceiling, which is what makes that argument structural rather than
//! a convention a caller could break.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

/// Defensive upper clamp, in seconds, on any derived slice.
///
/// `Duration::from_secs_f64` PANICS on a negative, non-finite or
/// `u64`-overflowing argument. A policy is untrusted input (it arrives from a
/// preset), so every path into a `Duration` is clamped rather than trusted.
const SLICE_MAX_SECS: f64 = 86_400.0;

/// What one call to the observed seam did with its allowance.
// `Eq` alongside `PartialEq`: every field is a `Duration`, which is `Eq`, so
// equality here is already total — the sibling `Slice` below derives both for
// the same reason. Required by `clippy::derive_partial_eq_without_eq`, which
// the workspace denies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observation {
    /// The call finished on its own, in this long. Only these feed the typical
    /// cost: a truncated call reports the ALLOWANCE, not its cost, so including
    /// it would pin the estimate at whatever the allowance already was and the
    /// allowance could never grow. That self-confirming loop is the single
    /// easiest way to get this wrong, which is why the type makes the two
    /// outcomes impossible to conflate.
    Completed(Duration),
    /// The call was cut off by its allowance, having run this long.
    Truncated(Duration),
}

/// The allowance for one call.
///
/// There is deliberately no variant carrying a deadline or an extension: the
/// only thing a consumer can do with a [`Slice::Bounded`] is compose
/// `min(ceiling, now + d)`, and [`Slice::Ceiling`] means "use the ceiling you
/// already have, unchanged". Neither can move a ceiling later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slice {
    /// Run this call against `min(ceiling, now + d)`.
    Bounded(Duration),
    /// Hand the call the caller's own ceiling, unchanged — the behaviour that
    /// shipped before calibration. Emitted when there is no evidence that any
    /// finite allowance completes, and when the remaining wall is unusable.
    Ceiling,
}

impl Slice {
    /// The bounded allowance, if this slice is one.
    #[must_use]
    pub const fn bounded(self) -> Option<Duration> {
        match self {
            Self::Bounded(d) => Some(d),
            Self::Ceiling => None,
        }
    }
}

/// Dimensionless policy for deriving one call's allowance.
///
/// Not one field carries a unit. Every quantity with a unit — the typical cost,
/// the remaining wall — is measured at runtime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlicePolicy {
    /// How many times the typical cost a call may run before it is treated as
    /// pathological and cut off.
    ///
    /// This is the parameter that decides whether normal work survives, so it
    /// is derived from the one row where a cap was measured GOOD rather than
    /// chosen: `nonlinear_activations` has a `0.13 s` median and was healthy at
    /// a `20 s` cap = **154x the median**, with the pathological tail starting
    /// at `70 s` = 538x. Anything in `[154, 538]` reproduces that row; the
    /// default sits at the safe end of the interval.
    pub headroom: f64,
    /// Largest share of the REMAINING wall one call may be allowed, once at
    /// least one call has completed. The same starvation guard `alpha_opt` and
    /// `#phase-window` apply, expressed against the live remaining budget so it
    /// tightens automatically as the instance runs out.
    pub max_share_of_remaining: f64,
    /// Share used before ANY call has completed, i.e. for the probe that finds
    /// out which regime this row is in. Small, because its whole job is to fail
    /// cheaply on a row where every call is expensive: on that row it is pure
    /// overhead before [`Slice::Ceiling`] takes over.
    pub probe_share: f64,
    /// Completed observations required before the measured term is allowed to
    /// govern. Below this only the share term applies, so a single fast call
    /// cannot pin the allowance.
    pub min_samples: usize,
    /// Completed observations retained. A sliding window, so a row whose cost
    /// changes with depth recalibrates instead of averaging over the run.
    pub window: usize,
    /// Multiplicative growth applied per CONSECUTIVE truncated call. A
    /// truncation is direct evidence the allowance is too small, and this is
    /// how the observer corrects a measured term that is too tight without
    /// waiting for the window to turn over.
    pub growth: f64,
    /// Cap on the growth exponent, so the term stays finite.
    pub growth_steps_max: u32,
}

impl Default for SlicePolicy {
    fn default() -> Self {
        Self {
            headroom: 160.0,
            max_share_of_remaining: 0.25,
            probe_share: 0.05,
            min_samples: 8,
            window: 128,
            growth: 2.0,
            growth_steps_max: 8,
        }
    }
}

impl SlicePolicy {
    /// Clamp every field into a range that cannot produce a `NaN`, a negative
    /// duration or an overflow downstream. Applied once at construction so the
    /// derivation itself has no error paths.
    #[must_use]
    fn sanitized(mut self) -> Self {
        let finite = |v: f64, fallback: f64, lo: f64, hi: f64| {
            if v.is_finite() {
                v.clamp(lo, hi)
            } else {
                fallback
            }
        };
        let d = Self::default();
        self.headroom = finite(self.headroom, d.headroom, 1.0, 1.0e9);
        self.max_share_of_remaining = finite(
            self.max_share_of_remaining,
            d.max_share_of_remaining,
            0.0,
            1.0,
        );
        self.probe_share = finite(self.probe_share, d.probe_share, 0.0, 1.0);
        self.growth = finite(self.growth, d.growth, 1.0, 1.0e3);
        self.window = self.window.clamp(1, 65_536);
        self.min_samples = self.min_samples.clamp(1, self.window);
        self.growth_steps_max = self.growth_steps_max.min(64);
        self
    }
}

/// A snapshot of what the observer has measured and derived. Everything a null
/// needs to be interpretable: whether the seam ran at all, whether the
/// allowance ever bound, and what the measurement underneath it was.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ObserverReport {
    /// Allowances handed out.
    pub derived: u64,
    /// Of those, how many were [`Slice::Ceiling`] — i.e. the observer declined
    /// to bound the call. `derived == ceiling` means the lever is armed and
    /// INERT.
    pub ceiling: u64,
    /// Calls that finished on their own.
    pub completed: u64,
    /// Calls cut off by their allowance.
    pub truncated: u64,
    /// The measured typical cost, in seconds: the median of the completed
    /// window. `0.0` when nothing has completed.
    pub median_secs: f64,
    /// Largest completed cost in the window, in seconds.
    pub max_completed_secs: f64,
    /// The most recent bounded allowance, in seconds. `0.0` if none was issued.
    pub last_slice_secs: f64,
    /// Smallest and largest bounded allowances issued, in seconds.
    pub min_slice_secs: f64,
    pub max_slice_secs: f64,
}

impl fmt::Display for ObserverReport {
    /// One stable `key=value` line. Both the MEASUREMENT (`median_secs`,
    /// `completed`) and the DERIVED value (`*_slice_secs`) are present, because
    /// a report showing only the derived value cannot distinguish "measured a
    /// cheap seam and bounded it correctly" from "measured nothing".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "derived={} ceiling={} completed={} truncated={} median_secs={:.4} \
             max_completed_secs={:.4} last_slice_secs={:.3} min_slice_secs={:.3} \
             max_slice_secs={:.3}",
            self.derived,
            self.ceiling,
            self.completed,
            self.truncated,
            self.median_secs,
            self.max_completed_secs,
            self.last_slice_secs,
            self.min_slice_secs,
            self.max_slice_secs,
        )
    }
}

#[derive(Debug, Default)]
struct ObserverState {
    /// Completed costs in seconds, most recent last. Bounded by `policy.window`.
    completed_window: VecDeque<f64>,
    derived: u64,
    ceiling: u64,
    completed: u64,
    truncated: u64,
    consecutive_truncated: u32,
    last_slice_secs: f64,
    min_slice_secs: Option<f64>,
    max_slice_secs: f64,
    max_completed_secs: f64,
}

/// Online per-call cost observer for one repeated seam.
///
/// Shared by reference across the closures of a single run. Locking is a
/// `Mutex` around a few words; the seams this governs cost milliseconds to
/// minutes per call, so contention is not measurable and determinism matters
/// more than the last nanosecond.
#[derive(Debug)]
pub struct CostObserver {
    policy: SlicePolicy,
    state: Mutex<ObserverState>,
}

impl CostObserver {
    /// Build an observer. The policy is sanitized once, here, so no later path
    /// can panic on a hostile preset value.
    #[must_use]
    pub fn new(policy: SlicePolicy) -> Self {
        Self {
            policy: policy.sanitized(),
            state: Mutex::new(ObserverState::default()),
        }
    }

    /// The policy actually in force, after sanitization.
    #[must_use]
    pub const fn policy(&self) -> SlicePolicy {
        self.policy
    }

    /// Poisoning is not a reason to stop verifying: a poisoned lock here means
    /// some other thread panicked while updating a counter, and the worst
    /// consequence of using the state anyway is a slightly stale allowance.
    fn state(&self) -> std::sync::MutexGuard<'_, ObserverState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Derive the allowance for the call that is about to start.
    ///
    /// `remaining` is the live wall left to the enclosing phase, measured by
    /// the caller at the moment of the call.
    #[must_use]
    pub fn slice(&self, remaining: Duration) -> Slice {
        let mut state = self.state();
        state.derived += 1;

        let rem = remaining.as_secs_f64();
        // Nothing to allocate: let the caller's own ceiling decide.
        if !rem.is_finite() || rem <= 0.0 {
            state.ceiling += 1;
            return Slice::Ceiling;
        }
        // No call has EVER completed and at least one was cut off: there is no
        // evidence that any finite allowance is enough, so the only defensible
        // allowance is the caller's ceiling — which is exactly the behaviour
        // that shipped before calibration, so this cannot be a regression.
        //
        // A state predicate, NOT a latch: the first completed call clears it
        // and calibration resumes.
        if state.completed == 0 && state.truncated > 0 {
            state.ceiling += 1;
            return Slice::Ceiling;
        }

        let growth_factor = self.policy.growth.powi(
            i32::try_from(
                state
                    .consecutive_truncated
                    .min(self.policy.growth_steps_max),
            )
            .unwrap_or(i32::MAX),
        );
        let growth_factor = if growth_factor.is_finite() {
            growth_factor.max(1.0)
        } else {
            1.0
        };

        let share_frac = if state.completed == 0 {
            self.policy.probe_share
        } else {
            self.policy.max_share_of_remaining
        };
        let share_secs = (share_frac * growth_factor).clamp(0.0, 1.0) * rem;

        let measured_secs = (state.completed_window.len() >= self.policy.min_samples)
            .then(|| median(&state.completed_window) * self.policy.headroom * growth_factor);

        let secs = match measured_secs {
            Some(measured) if measured.is_finite() => measured.min(share_secs),
            _ => share_secs,
        };

        if !secs.is_finite() || secs <= 0.0 {
            state.ceiling += 1;
            return Slice::Ceiling;
        }
        let secs = secs.min(SLICE_MAX_SECS);
        state.last_slice_secs = secs;
        state.min_slice_secs = Some(state.min_slice_secs.map_or(secs, |m: f64| m.min(secs)));
        state.max_slice_secs = state.max_slice_secs.max(secs);
        Slice::Bounded(Duration::from_secs_f64(secs))
    }

    /// Record what the call did with its allowance.
    pub fn observe(&self, observation: Observation) {
        let mut state = self.state();
        match observation {
            Observation::Completed(elapsed) => {
                let secs = elapsed.as_secs_f64();
                state.completed += 1;
                state.consecutive_truncated = 0;
                if secs.is_finite() && secs >= 0.0 {
                    state.max_completed_secs = state.max_completed_secs.max(secs);
                    if state.completed_window.len() >= self.policy.window {
                        state.completed_window.pop_front();
                    }
                    state.completed_window.push_back(secs);
                }
            }
            Observation::Truncated(_) => {
                state.truncated += 1;
                state.consecutive_truncated = state.consecutive_truncated.saturating_add(1);
            }
        }
    }

    /// The measured typical cost of one call **on this host, on this row**, for
    /// consumers that want [`crate::phase_window::admit`]'s decline-rather-than-
    /// truncate semantics instead of a slice. `None` until the window is warm.
    #[must_use]
    pub fn predict(&self) -> Option<Duration> {
        let state = self.state();
        if state.completed_window.len() < self.policy.min_samples {
            return None;
        }
        let secs = median(&state.completed_window);
        (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs.min(SLICE_MAX_SECS)))
    }

    /// Snapshot for telemetry.
    #[must_use]
    pub fn report(&self) -> ObserverReport {
        let state = self.state();
        ObserverReport {
            derived: state.derived,
            ceiling: state.ceiling,
            completed: state.completed,
            truncated: state.truncated,
            median_secs: if state.completed_window.is_empty() {
                0.0
            } else {
                median(&state.completed_window)
            },
            max_completed_secs: state.max_completed_secs,
            last_slice_secs: state.last_slice_secs,
            min_slice_secs: state.min_slice_secs.unwrap_or(0.0),
            max_slice_secs: state.max_slice_secs,
        }
    }
}

/// Exact median of the window.
///
/// A sort of at most `policy.window` `f64`s, paid once per derivation against a
/// seam whose calls cost milliseconds at minimum. Deliberately not an EWMA: the
/// distribution this has to survive is heavy-tailed by construction (that tail
/// is what the allowance exists to cut), and a mean-like statistic is dragged
/// by exactly the samples it must ignore. Deliberately not a high percentile
/// either: `headroom` is the parameter that covers spread, and keeping the
/// central statistic robust keeps the two concerns separable.
fn median(window: &VecDeque<f64>) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = window.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        f64::midpoint(sorted[mid - 1], sorted[mid])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(v: f64) -> Duration {
        Duration::from_secs_f64(v)
    }

    /// Feed `n` completed calls of `cost` seconds each, asking for a slice
    /// before each one exactly as a real consumer does.
    fn warm(observer: &CostObserver, n: usize, cost: f64, remaining: f64) {
        for _ in 0..n {
            let _ = observer.slice(secs(remaining));
            observer.observe(Observation::Completed(secs(cost)));
        }
    }

    #[test]
    fn cheap_seam_with_a_pathological_tail_lands_near_the_measured_good_cap() {
        // `nonlinear_activations`: 0.13 s median over ~1640 calls, tail 70-285 s,
        // measured GOOD at a hand-set 20 s cap. The derived allowance must land
        // in the interval that reproduces that row: above legitimate calls,
        // below the 70 s where the tail starts.
        let observer = CostObserver::new(SlicePolicy::default());
        warm(&observer, 32, 0.13, 1200.0);
        let slice = observer
            .slice(secs(1200.0))
            .bounded()
            .expect("a warm, cheap seam must be bounded")
            .as_secs_f64();
        assert!(
            (1.0..70.0).contains(&slice),
            "derived {slice}s must cut the 70-285s tail without touching 0.13s calls"
        );
        // Sharper, and the reason to believe the mechanism rather than the
        // policy: 0.13 s x headroom 160 = 20.8 s, against the 20.0 s that a
        // human chose and that measured GOOD on this row. The number is not
        // copied from that A/B — it falls out of the median the observer
        // measured — which is exactly why the sibling row gets a different one.
        assert!(
            (20.0..22.0).contains(&slice),
            "derived {slice}s should land on the measured-good 20 s by measurement"
        );
        let report = observer.report();
        assert!(
            (report.median_secs - 0.13).abs() < 1e-9,
            "the MEASUREMENT must be reported alongside the derived value, got {report}"
        );
    }

    #[test]
    fn a_row_where_nothing_ever_completes_falls_back_to_the_callers_ceiling() {
        // `upsample`: every call genuinely needs more than the allowance. After
        // ONE truncated probe the observer must stop bounding entirely, because
        // the ceiling is the pre-calibration behaviour and therefore cannot be
        // a regression.
        let observer = CostObserver::new(SlicePolicy::default());
        let probe = observer.slice(secs(1200.0));
        let probe_secs = probe
            .bounded()
            .expect("first probe is bounded")
            .as_secs_f64();
        assert!(
            probe_secs <= 0.05 * 1200.0 + 1e-9,
            "the probe must be CHEAP ({probe_secs}s), its job is to fail fast"
        );
        observer.observe(Observation::Truncated(secs(probe_secs)));
        for _ in 0..16 {
            assert_eq!(
                observer.slice(secs(1100.0)),
                Slice::Ceiling,
                "no completed sample + a truncation means no evidence any finite slice works"
            );
        }
        let report = observer.report();
        assert_eq!(report.completed, 0);
        assert_eq!(report.truncated, 1);
        assert_eq!(report.ceiling, 16, "armed but declining to bound: {report}");
    }

    #[test]
    fn the_ceiling_fallback_is_a_predicate_not_a_latch() {
        // One early pathological call must not disable calibration for the run.
        let observer = CostObserver::new(SlicePolicy::default());
        let _ = observer.slice(secs(1200.0));
        observer.observe(Observation::Truncated(secs(60.0)));
        assert_eq!(observer.slice(secs(1000.0)), Slice::Ceiling);
        observer.observe(Observation::Completed(secs(0.2)));
        assert!(
            observer.slice(secs(1000.0)).bounded().is_some(),
            "a completed call must clear the fallback and resume calibration"
        );
    }

    #[test]
    fn one_row_and_its_sibling_get_allowances_orders_of_magnitude_apart() {
        // THE point of the module, stated as a test: the same policy, two rows
        // of the same benchmark, and the derived allowances differ by orders of
        // magnitude because the MEASURED per-call costs do.
        let cheap = CostObserver::new(SlicePolicy::default());
        warm(&cheap, 16, 0.13, 1200.0);
        let cheap_slice = cheap.slice(secs(1200.0)).bounded().unwrap().as_secs_f64();

        let dear = CostObserver::new(SlicePolicy::default());
        warm(&dear, 16, 40.0, 1200.0);
        let dear_slice = dear.slice(secs(1200.0)).bounded().unwrap().as_secs_f64();

        // ~20.8 s versus ~300 s from ONE policy, because the two rows' measured
        // costs differ. The expensive row is share-limited (its measured term is
        // far larger still); the cheap row is measurement-limited.
        assert!(
            dear_slice > cheap_slice * 10.0,
            "expensive row must get a far larger allowance: {dear_slice}s vs {cheap_slice}s"
        );
    }

    #[test]
    fn consecutive_truncations_grow_the_allowance_and_a_completion_resets_it() {
        let observer = CostObserver::new(SlicePolicy::default());
        warm(&observer, 16, 0.5, 100_000.0);
        let base = observer.slice(secs(100_000.0)).bounded().unwrap();
        observer.observe(Observation::Truncated(base));
        let grown_once = observer.slice(secs(100_000.0)).bounded().unwrap();
        observer.observe(Observation::Truncated(grown_once));
        let grown_twice = observer.slice(secs(100_000.0)).bounded().unwrap();
        assert!(
            grown_twice > grown_once && grown_once > base,
            "must escalate"
        );

        observer.observe(Observation::Completed(secs(0.5)));
        let reset = observer.slice(secs(100_000.0)).bounded().unwrap();
        assert_eq!(
            reset, base,
            "a completion is evidence the allowance is fine"
        );
    }

    #[test]
    fn a_truncated_call_never_feeds_the_typical_cost() {
        // The self-confirming loop this module has to avoid: if truncated calls
        // fed the median, the median would sit at the allowance and the
        // allowance could never grow past itself.
        let observer = CostObserver::new(SlicePolicy::default());
        for _ in 0..32 {
            observer.observe(Observation::Truncated(secs(20.0)));
        }
        assert_eq!(observer.report().median_secs, 0.0);
        assert_eq!(observer.predict(), None, "truncations are not measurements");
    }

    #[test]
    fn the_allowance_tightens_as_the_instance_runs_out() {
        // The share term is against the LIVE remaining wall, so the same
        // observer state yields a smaller allowance later in the run. A
        // constant cannot express this at all.
        let observer = CostObserver::new(SlicePolicy::default());
        warm(&observer, 16, 30.0, 1200.0);
        let early = observer.slice(secs(1200.0)).bounded().unwrap();
        let late = observer.slice(secs(40.0)).bounded().unwrap();
        assert!(late < early, "{late:?} must be under {early:?}");
        assert!(
            late <= secs(0.25 * 40.0 + 1e-6),
            "never more than max_share_of_remaining of what is left"
        );
    }

    #[test]
    fn no_slice_can_ever_exceed_the_remaining_wall() {
        // THE MOAT ARGUMENT, as an executable property. The consumer composes
        // `min(ceiling, now + slice)`, so a slice at or under `remaining` can
        // never move a ceiling later. `Slice::Ceiling` returns the caller's own
        // ceiling and is therefore trivially safe.
        for growth_steps in 0..12u32 {
            for remaining in [1e-3, 0.5, 20.0, 1200.0, 86_400.0] {
                let observer = CostObserver::new(SlicePolicy::default());
                warm(&observer, 16, remaining * 10.0, remaining);
                for _ in 0..growth_steps {
                    let s = observer.slice(secs(remaining));
                    if let Some(d) = s.bounded() {
                        observer.observe(Observation::Truncated(d));
                    }
                }
                if let Some(d) = observer.slice(secs(remaining)).bounded() {
                    assert!(
                        d.as_secs_f64() <= remaining + 1e-9,
                        "slice {d:?} exceeded remaining {remaining}s at {growth_steps} growth steps"
                    );
                }
            }
        }
    }

    #[test]
    fn a_hostile_policy_is_sanitized_rather_than_panicking() {
        // A preset is untrusted input and `Duration::from_secs_f64` panics on
        // NaN, negatives and overflow.
        let hostile = SlicePolicy {
            headroom: f64::NAN,
            max_share_of_remaining: -3.0,
            probe_share: f64::INFINITY,
            min_samples: 0,
            window: 0,
            growth: f64::NEG_INFINITY,
            growth_steps_max: u32::MAX,
        };
        let observer = CostObserver::new(hostile);
        let p = observer.policy();
        assert!(p.headroom.is_finite() && p.headroom >= 1.0);
        assert!((0.0..=1.0).contains(&p.max_share_of_remaining));
        assert!(p.window >= 1 && p.min_samples >= 1 && p.min_samples <= p.window);
        for _ in 0..32 {
            let s = observer.slice(secs(100.0));
            if let Some(d) = s.bounded() {
                assert!(d.as_secs_f64() <= 100.0 + 1e-9);
                observer.observe(Observation::Truncated(d));
            }
            observer.observe(Observation::Completed(secs(0.01)));
        }
    }

    #[test]
    fn zero_and_nonfinite_remaining_declines_to_the_ceiling() {
        let observer = CostObserver::new(SlicePolicy::default());
        warm(&observer, 16, 0.1, 100.0);
        assert_eq!(observer.slice(Duration::ZERO), Slice::Ceiling);
    }

    #[test]
    fn predict_feeds_phase_window_admit_once_warm() {
        // The bridge to invariant I1's existing consumer: a measured prediction
        // is exactly what `phase_window::admit` has always wanted and never had
        // a producer for on a seam with no MAC model.
        let observer = CostObserver::new(SlicePolicy::default());
        assert_eq!(observer.predict(), None, "cold observer predicts nothing");
        warm(&observer, 16, 2.0, 1200.0);
        let predicted = observer.predict().expect("warm observer predicts");
        assert!((predicted.as_secs_f64() - 2.0).abs() < 1e-9);
        let admission = crate::phase_window::admit(
            predicted,
            secs(1200.0),
            crate::phase_window::WindowPolicy::default(),
        );
        assert!(admission.is_admitted());
    }

    #[test]
    fn the_report_distinguishes_inert_from_engaged() {
        // A null is only interpretable if "never bounded anything" and "bounded
        // everything and nothing was slow" look different.
        let inert = CostObserver::new(SlicePolicy::default());
        let _ = inert.slice(secs(1200.0));
        inert.observe(Observation::Truncated(secs(60.0)));
        let _ = inert.slice(secs(1200.0));
        let r = inert.report();
        assert!(r.ceiling > 0 && r.completed == 0, "inert: {r}");

        let engaged = CostObserver::new(SlicePolicy::default());
        warm(&engaged, 16, 0.1, 1200.0);
        let r = engaged.report();
        assert!(
            r.ceiling == 0 && r.completed == 16 && r.max_slice_secs > 0.0,
            "engaged: {r}"
        );
        assert!(
            format!("{r}").contains("median_secs="),
            "line carries the measurement"
        );
    }
}
