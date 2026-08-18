// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Adaptive attack STALL cutoff (#attack-stall, design S4): end the disjunctive
//! global PGD phase early when its closest-to-violation margin has stopped
//! improving, so the rest of the attack slice goes to the phases that can still
//! decide the instance.
//!
//! # The measurement this exists for
//!
//! `docs/CONV_CROWN_WALL_DESIGN_2026-07-27.md` S4, on oval21 — all three rows
//! GT-`unsat`, so no attack can EVER succeed:
//!
//! | row | falsification | BaB | verdict |
//! |---|---:|---:|---|
//! | base img395 | 21.8 s (38%) | 33.5 s | timeout |
//! | wide img7176 | 34.4 s (60%) | 15.5 s | timeout |
//! | deep img6995 | 32.7 s (57%) | 18.1 s | timeout |
//!
//! Spending 38–60 % of a budget on an impossible attack is indefensible even
//! when the reclaimed seconds convert nothing (see "why default-off" below).
//!
//! # Why the signal is a RATE, not a LEVEL
//!
//! The margin LEVEL provably cannot separate "unsat, plateaued at its
//! supremum" from "sat, still climbing". Measured pair (metaroom_2023, recorded
//! in `super::attack_extension`):
//!
//! - `spec_idx_28`, GT=**unsat**: plateaus at **-0.033** across 14 restarts —
//!   that is the unsat sup, and the attack is done improving.
//! - `spec_idx_129`, GT=**sat**: was at **-0.083** mid-ascent when its preset
//!   cap cut it, and flips to `sat` when the attack keeps running.
//!
//! The unsat instance is CLOSER to violation than the sat one, so any threshold
//! on the level itself gets one of the two populations wrong. The *improvement
//! rate* orders them correctly on that same measured pair: flat across 14
//! restarts versus climbing when cut. That is the signal this module watches.
//!
//! Concretely: the lanes track `best_margin` as a running MAX over every
//! evaluated point, so it can only ever RISE. If a whole window of
//! `window_fraction × attack_slice` passes without that maximum rising by more
//! than the confirmation path's own noise floor
//! (`super::disjunctive_pgd::noise_scaled_margin` — the measured ny-vs-ORT
//! forward disagreement, so "improvement" means improvement distinguishable
//! from f32 accumulation noise), the ascent has plateaued and the attack stops.
//! Nothing here invents a magnitude: the window is a fraction of the slice the
//! phase was actually granted, and the epsilon is the noise floor already
//! measured at the call site.
//!
//! # Why it is DEFAULT-OFF
//!
//! Both directions have a measured cost, and only one of them has a measured
//! gain — which is zero:
//!
//! - Reclaiming the budget converts nothing on the population it was designed
//!   for. The decisive S4 A/B (3 GT-unsat oval21 rows, `disjunctive_pgd_fraction`
//!   0.50 vs a capped 0.10 / 5 s) came back NEGATIVE: `unknown 57 s` in both
//!   arms on all three rows. `relusplitter.yaml`'s `#reclaim-pgd` note repeats
//!   it — "It converted ZERO rows; these instances are held by Conv2d CROWN
//!   throughput, not falsification budget."
//! - Cutting too eagerly LOSES rows. On tinyimagenet the disjunctive-PGD lane
//!   is what FINDS the sat rows, and they land at **12.35 s / 17.67 s /
//!   20.39 s** of a 100 s budget (b61b5f10, which measured 8/15 → 3/15 when a
//!   sibling benchmark's allocation was ported in). The rule that commit
//!   recorded is the one this module obeys: *A/B a budget change against the
//!   sat rows, not only the unsat rows being chased.*
//!
//! So the mechanism ships inert. `PhaseBudgetConfig::disjunctive_pgd_stall_window_fraction`
//! is `None` in the sealed defaults and set by no shipped preset; arming it for
//! a category requires an A/B for THAT category covering its sat rows.
//! `NY_ATTACK_STALL_WINDOW=<fraction>` arms it for a sweep without editing a
//! preset (that is how such an A/B is run); `NY_ATTACK_STALL_CUT=0` is the
//! kill switch.
//!
//! Sizing, if an A/B does license it: the window must be longer than the time
//! that category's sat rows need to CROSS, not merely to start climbing — the
//! running max is judged over the window, and a row that improves early and
//! then jumps to violation later still looks flat in between. On tinyimagenet
//! (30 s attack slice, sat rows landing by ~20 s of a 100 s budget) that puts
//! the plausible floor at `0.5`, not at the design sketch's "15 % of budget".
//!
//! Scope: the global DISJUNCTIVE PGD phase only — the phase the S4 measurement
//! is about. The post-BaB fallback attack is deliberately excluded (nothing
//! downstream could use the reclaimed time), as is the attack extension (a cut
//! attack is never extended). A cut ends that whole phase, including its
//! sampling/SPSA fallback lanes: they run to the same deadline, so leaving them
//! alive would reclaim nothing.
//!
//! # Soundness: completeness-only, by construction
//!
//! An attack can only ever produce a counterexample CANDIDATE. Every candidate
//! still passes `re_evaluate_and_confirm` and the trusted-ORT vnncomp gate
//! before any `sat` is emitted, and **no `unsat` is ever concluded from an
//! attack that failed or was cut** — a finished attack, a deadline-cut attack
//! and a stall-cut attack all return the same "no counterexample" to the same
//! caller, which then runs the same bound/BaB path over the same certified
//! bounds. Cutting an attack short can therefore lose a `sat` (a completeness
//! cost) and can NEVER narrow a bound, mint an `unsat`, or change any verdict
//! that is reached.
//!
//! The one place a cut is visible downstream is `super::attack_extension`,
//! which is told to DECLINE on a stall cut (extending an attack that was
//! stopped precisely because it was not improving would undo the cut). That is
//! also attack-budget-only.

use std::time::{Duration, Instant};

use ny_propagate::PhaseBudgetConfig;

/// Kill switch: `NY_ATTACK_STALL_CUT=0` disables the cutoff even where a preset
/// or `NY_ATTACK_STALL_WINDOW` arms it.
fn stall_cutoff_enabled() -> bool {
    std::env::var("NY_ATTACK_STALL_CUT").ok().as_deref() != Some("0")
}

/// Runtime arming/override: `NY_ATTACK_STALL_WINDOW=<fraction>`. Outranks the
/// preset value so an A/B sweep needs no `--configs-dir` tree.
fn env_window_fraction() -> Option<f32> {
    std::env::var("NY_ATTACK_STALL_WINDOW")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
}

/// The stall-cutoff policy for one attack phase. `Default` (and
/// [`Self::disabled`]) is INERT — every lane that is handed it behaves exactly
/// as it did before this module existed.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct AttackStallPolicy {
    /// Probe window as a fraction of the attack phase's own slice. `None`
    /// disables the cutoff.
    window_fraction: Option<f32>,
}

impl AttackStallPolicy {
    /// The inert policy — no cutoff, whatever the preset or environment says.
    pub(super) const fn disabled() -> Self {
        Self {
            window_fraction: None,
        }
    }

    /// Build from an explicit window fraction. Anything outside `(0, 1]`, or
    /// non-finite, is treated as "unset": a malformed knob must never arm a
    /// budget cut, and a fraction of 1.0 is already inert (the window can only
    /// close at the phase deadline, where the phase ends anyway).
    pub(super) fn from_fraction(window_fraction: Option<f32>) -> Self {
        Self {
            window_fraction: window_fraction.filter(|f| f.is_finite() && *f > 0.0 && *f <= 1.0),
        }
    }

    /// Resolve the phase-scoped policy: `NY_ATTACK_STALL_WINDOW` outranks the
    /// preset knob, and `NY_ATTACK_STALL_CUT=0` disables both.
    pub(super) fn from_phase_policy(policy: &PhaseBudgetConfig) -> Self {
        if !stall_cutoff_enabled() {
            return Self::disabled();
        }
        Self::from_fraction(env_window_fraction().or(policy.disjunctive_pgd_stall_window_fraction))
    }

    /// Whether a cutoff is armed at all.
    ///
    /// Reported in the phase flight note: an A/B whose arming silently did not
    /// take is worse than no A/B, so the run says out loud which arm it is in.
    pub(super) fn is_armed(self) -> bool {
        self.window_fraction.is_some()
    }

    /// Open a monitor for a phase running `[phase_start, phase_deadline)`.
    ///
    /// An UNBOUNDED phase (`phase_deadline == None`) yields an inert monitor:
    /// there is no slice to reclaim and no measurement at the call site to size
    /// a window from, so the cutoff declines rather than inventing one.
    pub(super) fn monitor(
        self,
        phase_start: Instant,
        phase_deadline: Option<Instant>,
    ) -> AttackStallMonitor {
        let window = self.window_fraction.and_then(|fraction| {
            let slice = phase_deadline?.checked_duration_since(phase_start)?;
            let window = slice.mul_f32(fraction);
            (window > Duration::ZERO).then_some(window)
        });
        AttackStallMonitor {
            window,
            window_start: phase_start,
            window_start_best: None,
            best: None,
            tripped: false,
        }
    }
}

/// Per-phase stall watchdog. Fed the margin of every point an attack lane
/// evaluates; answers exactly one question — "has the best margin stopped
/// improving for a whole window?".
///
/// ATTACK-BUDGET ONLY: the monitor holds no bound, no certificate and no
/// verdict, and its output is consumed only as "stop generating candidates".
pub(super) struct AttackStallMonitor {
    /// `None` = inert (policy unarmed, or the phase has no deadline).
    window: Option<Duration>,
    /// Start of the window currently being judged.
    window_start: Instant,
    /// Best margin as of `window_start` (`None` until the first finite margin).
    window_start_best: Option<f32>,
    /// Best margin over every point observed — a running MAX, so non-decreasing.
    best: Option<f32>,
    tripped: bool,
}

impl AttackStallMonitor {
    /// Can this monitor cut at all? One `Option` discriminant load.
    ///
    /// Attack lanes call this FIRST so the default (unarmed) path pays neither
    /// the noise-floor fold nor a clock read per step — the cutoff must be free
    /// where it is off, which is everywhere until an A/B arms it.
    pub(super) fn is_armed(&self) -> bool {
        self.window.is_some()
    }

    /// Record one evaluated point's clause margin and report whether the attack
    /// should stop.
    ///
    /// - `margin`: closest-to-violation clause margin at this point (`>= 0`
    ///   would be a counterexample).
    /// - `noise_floor`: `super::disjunctive_pgd::noise_scaled_margin` at the
    ///   same output — the improvement must beat the SAME threshold the
    ///   confirmation gate uses to call a margin real rather than f32 noise.
    /// - `now`: the observation time (injected so the policy is unit-testable
    ///   without a clock).
    ///
    /// Returns `true` at most once per phase: the moment a full window has
    /// elapsed with no better-than-noise improvement.
    pub(super) fn observe(&mut self, margin: f32, noise_floor: f32, now: Instant) -> bool {
        let Some(window) = self.window else {
            return false;
        };
        if self.tripped {
            return true;
        }
        // Sentinel/non-finite margins carry no signal (`f32::NEG_INFINITY` is
        // the lanes' "no modeled constraint" value). Never cut on them.
        if !margin.is_finite() {
            return false;
        }
        let best = self.best.map_or(margin, |b| b.max(margin));
        self.best = Some(best);

        let Some(window_start_best) = self.window_start_best else {
            // First finite margin opens the first window; a cut can therefore
            // never happen before a full window of ATTACK has been observed.
            self.window_start = now;
            self.window_start_best = Some(best);
            return false;
        };

        // A satisfying point is in hand (template screen fired, exact re-screen
        // still pending/near-miss). The lane's own candidate path owns this
        // case; a budget watchdog must not interrupt an attack that is already
        // at the violation surface.
        if best >= 0.0 {
            self.window_start = now;
            self.window_start_best = Some(best);
            return false;
        }

        if now
            .checked_duration_since(self.window_start)
            .is_none_or(|elapsed| elapsed < window)
        {
            return false;
        }

        // A non-finite noise floor degrades to "any strict rise counts", which
        // makes a cut HARDER, never easier.
        let epsilon = if noise_floor.is_finite() {
            noise_floor.max(0.0)
        } else {
            0.0
        };
        if best - window_start_best > epsilon {
            // Real progress: this is the `spec_idx_129` (GT=sat, mid-ascent)
            // shape — roll the window and let the attack keep climbing.
            self.window_start = now;
            self.window_start_best = Some(best);
            return false;
        }

        // Plateau: the `spec_idx_28` (GT=unsat, at its sup) shape.
        self.tripped = true;
        true
    }

    /// Whether this monitor ever cut the attack. Production code acts on
    /// [`Self::observe`]'s return value at the point of the cut, so this is a
    /// test-only accessor.
    #[cfg(test)]
    pub(super) fn tripped(&self) -> bool {
        self.tripped
    }

    /// Best margin observed, for the flight note on a cut.
    pub(super) fn best_margin(&self) -> Option<f32> {
        self.best
    }
}

/// Operator/diagnostic note for a stall cut, shared by the attack lanes.
///
/// The wording is deliberate: a cut attack reports that IT stopped improving,
/// never anything about the property. An attack's silence is not evidence of
/// `unsat` — the bound/BaB phases that follow are the only things that can
/// produce one, and they run over the same certified bounds either way.
pub(super) fn report_stall_cut(
    json: bool,
    diag: bool,
    location: std::fmt::Arguments<'_>,
    best_margin: Option<f32>,
) {
    let best = best_margin.unwrap_or(f32::NEG_INFINITY);
    if diag {
        eprintln!(
            "[pgd-stall] cut at {location}: best disjunctive margin {best:.5} flat for a full probe window"
        );
    }
    if !json {
        println!(
            "  Attack stall cutoff (#attack-stall): best margin {best:.5} has not improved for a full probe window at {location}; handing the remaining ATTACK budget to the bound/BaB phases (this says nothing about the property)."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 30 s attack slice with a 0.5 window — the shape a tinyimagenet-class
    /// preset would arm (`disjunctive_pgd_fraction: 0.40`,
    /// `disjunctive_pgd_max_secs: 30`).
    fn monitor_30s_slice(fraction: f32) -> (Instant, AttackStallMonitor) {
        let start = Instant::now();
        let monitor = AttackStallPolicy::from_fraction(Some(fraction))
            .monitor(start, Some(start + Duration::from_secs(30)));
        (start, monitor)
    }

    fn at(start: Instant, secs: u64) -> Instant {
        start + Duration::from_secs(secs)
    }

    /// What the lanes pass: `noise_scaled_margin` = `max(1e-4, 1e-5 * |y|max)`,
    /// the measured ny-vs-ORT forward disagreement. Its floor is 1e-4, which is
    /// what a small-logit model produces.
    const NOISE: f32 = 1e-4;

    #[test]
    fn unarmed_policy_is_inert() {
        let start = Instant::now();
        let mut monitor =
            AttackStallPolicy::disabled().monitor(start, Some(start + Duration::from_secs(30)));
        for secs in 0..600 {
            assert!(
                !monitor.observe(-0.033, NOISE, at(start, secs)),
                "the default policy must never cut an attack"
            );
        }
        assert!(!monitor.tripped());
    }

    #[test]
    fn monitor_reports_unarmed_so_lanes_can_skip_the_work() {
        // The lanes gate the noise-floor fold and the clock read on this, so an
        // unarmed monitor must report unarmed for EVERY reason it can be inert.
        let start = Instant::now();
        let deadline = Some(start + Duration::from_secs(30));
        assert!(!AttackStallPolicy::disabled()
            .monitor(start, deadline)
            .is_armed());
        assert!(!AttackStallPolicy::from_fraction(Some(0.5))
            .monitor(start, None)
            .is_armed());
        assert!(AttackStallPolicy::from_fraction(Some(0.5))
            .monitor(start, deadline)
            .is_armed());
    }

    #[test]
    fn unbounded_phase_is_inert() {
        // No deadline ⇒ no slice ⇒ no measured window to size the probe from.
        let start = Instant::now();
        let mut monitor = AttackStallPolicy::from_fraction(Some(0.5)).monitor(start, None);
        assert!(!monitor.observe(-0.033, NOISE, at(start, 600)));
    }

    #[test]
    fn out_of_range_fractions_do_not_arm() {
        for bad in [f32::NAN, f32::INFINITY, 0.0, -0.25, 1.5] {
            assert!(
                !AttackStallPolicy::from_fraction(Some(bad)).is_armed(),
                "a malformed window fraction ({bad}) must not arm a budget cut"
            );
        }
        assert!(!AttackStallPolicy::from_fraction(None).is_armed());
        assert!(AttackStallPolicy::from_fraction(Some(0.5)).is_armed());
        // 1.0 parses as armed but can only close at the phase deadline.
        assert!(AttackStallPolicy::from_fraction(Some(1.0)).is_armed());
    }

    #[test]
    fn plateaued_unsat_shape_cuts_after_one_window() {
        // metaroom spec_idx_28 (GT=unsat): the attack sits at the unsat sup.
        let (start, mut monitor) = monitor_30s_slice(0.5);
        for secs in 0..15 {
            assert!(
                !monitor.observe(-0.033, NOISE, at(start, secs)),
                "no cut before a FULL window has elapsed"
            );
        }
        assert!(
            monitor.observe(-0.033, NOISE, at(start, 15)),
            "a full 15 s window with a flat best margin is the stall signal"
        );
        assert!(monitor.tripped());
    }

    #[test]
    fn climbing_sat_shape_never_cuts() {
        // metaroom spec_idx_129 (GT=sat) shape: still ascending when the
        // preset cap cut it at -0.083. A climbing ascent must survive.
        let (start, mut monitor) = monitor_30s_slice(0.5);
        for secs in 0..30 {
            let margin = -5.0 + 0.1 * secs as f32;
            assert!(
                !monitor.observe(margin, NOISE, at(start, secs)),
                "a still-improving attack must keep its budget (secs={secs})"
            );
        }
        assert!(!monitor.tripped());
    }

    #[test]
    fn tinyimagenet_sat_landing_horizon_survives() {
        // b61b5f10: the tinyimagenet sat rows land at 12.35 s / 17.67 s /
        // 20.39 s. Replay the slowest as a steady ascent that crosses at 21 s —
        // it must still be alive at every step up to the crossing.
        let (start, mut monitor) = monitor_30s_slice(0.5);
        for secs in 0..21 {
            let margin = -2.1 + 0.1 * secs as f32;
            assert!(
                !monitor.observe(margin, NOISE, at(start, secs)),
                "the sat rows this lane finds must not be cut (secs={secs})"
            );
        }
    }

    #[test]
    fn improvement_below_the_noise_floor_is_not_improvement() {
        // A creeping f32 max is exactly what a plateaued PGD produces; if it
        // counted as progress the cutoff would be inert on the population it
        // exists for.
        let (start, mut monitor) = monitor_30s_slice(0.5);
        assert!(!monitor.observe(-0.033, NOISE, at(start, 0)));
        for secs in 1..15 {
            let creep = -0.033 + 1e-6 * secs as f32;
            assert!(!monitor.observe(creep, NOISE, at(start, secs)));
        }
        assert!(
            monitor.observe(-0.033 + 1e-5, NOISE, at(start, 15)),
            "sub-noise creep must not roll the window"
        );
    }

    #[test]
    fn above_noise_improvement_rolls_the_window() {
        let (start, mut monitor) = monitor_30s_slice(0.5);
        assert!(!monitor.observe(-0.033, NOISE, at(start, 0)));
        // A clear improvement at the window boundary rolls it instead of cutting.
        assert!(!monitor.observe(-0.030, NOISE, at(start, 15)));
        // ... and the next window is judged from there.
        assert!(!monitor.observe(-0.030, NOISE, at(start, 29)));
        assert!(monitor.observe(-0.030, NOISE, at(start, 30)));
    }

    #[test]
    fn running_max_ignores_worse_points() {
        // Restarts routinely re-init far from the best point; the running max
        // must not read that as regress-then-improve churn.
        let (start, mut monitor) = monitor_30s_slice(0.5);
        assert!(!monitor.observe(-0.033, NOISE, at(start, 0)));
        assert!(!monitor.observe(-5.0, NOISE, at(start, 5)));
        assert!(!monitor.observe(-4.0, NOISE, at(start, 10)));
        assert!(
            monitor.observe(-0.034, NOISE, at(start, 15)),
            "the best margin never rose above its window-start value"
        );
    }

    #[test]
    fn margin_at_or_above_violation_never_cuts() {
        // Template screen fired but the exact re-screen has not confirmed: the
        // lane's candidate path owns this, not a budget watchdog.
        let (start, mut monitor) = monitor_30s_slice(0.5);
        for secs in 0..60 {
            assert!(!monitor.observe(0.001, NOISE, at(start, secs)));
        }
        assert!(!monitor.tripped());
    }

    #[test]
    fn non_finite_margins_never_cut() {
        let (start, mut monitor) = monitor_30s_slice(0.5);
        for secs in 0..60 {
            assert!(!monitor.observe(f32::NEG_INFINITY, NOISE, at(start, secs)));
            assert!(!monitor.observe(f32::NAN, NOISE, at(start, secs)));
        }
        assert!(!monitor.tripped());
        assert_eq!(monitor.best_margin(), None);
    }

    #[test]
    fn non_finite_noise_floor_only_makes_cutting_harder() {
        let (start, mut monitor) = monitor_30s_slice(0.5);
        assert!(!monitor.observe(-0.033, f32::NAN, at(start, 0)));
        // Any strict rise still counts as progress.
        assert!(!monitor.observe(-0.033 + 1e-7, f32::NAN, at(start, 15)));
        // A flat max still cuts.
        assert!(monitor.observe(-0.033 + 1e-7, f32::NAN, at(start, 30)));
    }

    #[test]
    fn cut_is_sticky_once_tripped() {
        let (start, mut monitor) = monitor_30s_slice(0.5);
        assert!(!monitor.observe(-0.033, NOISE, at(start, 0)));
        assert!(monitor.observe(-0.033, NOISE, at(start, 15)));
        assert!(
            monitor.observe(-0.001, NOISE, at(start, 16)),
            "a tripped monitor stays tripped for the rest of the phase"
        );
    }

    #[test]
    fn window_scales_with_the_measured_slice() {
        // The window is a fraction of the slice the phase was GRANTED — there
        // is no absolute number in the policy. A 5 s slice (relusplitter's
        // `disjunctive_pgd_max_secs: 5`) gets a 2.5 s window.
        let start = Instant::now();
        let mut monitor = AttackStallPolicy::from_fraction(Some(0.5))
            .monitor(start, Some(start + Duration::from_secs(5)));
        assert!(!monitor.observe(-0.033, NOISE, start));
        assert!(!monitor.observe(-0.033, NOISE, start + Duration::from_millis(2_400)));
        assert!(monitor.observe(-0.033, NOISE, start + Duration::from_millis(2_500)));
    }

    #[test]
    fn already_expired_phase_is_inert() {
        // Deadline before the phase start (an exhausted budget): no window.
        let deadline = Instant::now();
        let start = deadline + Duration::from_secs(1);
        let mut monitor =
            AttackStallPolicy::from_fraction(Some(0.5)).monitor(start, Some(deadline));
        assert!(!monitor.observe(-0.033, NOISE, at(start, 600)));
    }
}
