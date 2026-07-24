// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Runtime budget ledger for β-CROWN verification phases.
//!
//! All timeout consumers derive from one ledger instead of recomputing
//! fractions directly from raw `timeout`. Later phases consume the
//! **remaining** time, not the original timeout.
//!
//! Part of #2206. Design: designs/2026-03-16-issue-2206-adaptive-phase-budgeting.md

use ny_propagate::PhaseBudgetConfig;
use std::time::{Duration, Instant};

/// Budgets at or below this total get the small-budget attack cap.
const SMALL_BUDGET_TOTAL: Duration = Duration::from_secs(30);

/// Attack-phase ceiling (fraction of total) on small budgets.
const SMALL_BUDGET_ATTACK_FRACTION: f32 = 0.15;

/// Whether the small-budget attack time cap is disabled (`NY_NO_PGD_TIME_CAP=1`).
fn pgd_time_cap_disabled() -> bool {
    std::env::var("NY_NO_PGD_TIME_CAP").is_ok_and(|v| v == "1")
}

/// Runtime budget ledger for β-CROWN verification.
///
/// Tracks wall-clock elapsed time and derives per-phase deadlines from
/// a single [`PhaseBudgetConfig`]. Replaces the scattered local fraction
/// arithmetic in `attack_budget.rs`, `sequential.rs`, `graph.rs`,
/// `disjunctive.rs`, and `mod.rs`.
pub(in crate::commands::beta_crown) struct PhaseBudgetLedger {
    start: Instant,
    total: Option<Duration>,
    policy: PhaseBudgetConfig,
    /// FIX 2 (default-on Graph-MIP + feature `mip`): when the Graph-MIP whole-net
    /// escalation is armed and the phase policy requests a nonzero reservation,
    /// the BaB deadline RESERVES the MIP slice inside the
    /// scored budget (so the escalation is not reaped by the vnncomp watchdog)
    /// and `mip_timeout` never overdraws the actual remaining wall (measured:
    /// the un-capped floor granted 27 s with 0 s remaining at a 120 s budget
    /// → watchdog kill mid-escalation). `NY_GRAPH_MIP=0` or a non-MIP build
    /// keeps every formula byte-identical. A zero-reservation category policy
    /// also leaves the full BaB slice available.
    mip_reservation_armed: bool,
}

impl PhaseBudgetLedger {
    /// Create a new ledger.
    ///
    /// - `timeout_secs`: 0 means unbounded (no deadline).
    /// - `policy`: phase budget fractions from `BetaCrownConfig.phase_budget`.
    pub(in crate::commands::beta_crown) fn new(
        timeout_secs: u64,
        policy: PhaseBudgetConfig,
    ) -> Self {
        let total = if timeout_secs > 0 {
            Some(Duration::from_secs(timeout_secs))
        } else {
            None
        };
        #[cfg(feature = "mip")]
        let mip_reservation_armed =
            super::super::graph_mip::graph_mip_enabled() && policy.requests_mip_reservation();
        #[cfg(not(feature = "mip"))]
        let mip_reservation_armed = false;
        Self {
            start: Instant::now(),
            total,
            policy,
            mip_reservation_armed,
        }
    }

    /// Test-only override for the arming flag (env-free, race-free tests).
    #[cfg(test)]
    fn with_mip_reservation(mut self, armed: bool) -> Self {
        self.mip_reservation_armed = armed;
        self
    }

    /// The MIP slice the armed reservation carves out of the scored budget:
    /// `total * mip_min_fraction` clamped to `[mip_min_secs, mip_max_secs]` —
    /// the SAME formula as [`Self::mip_timeout`]'s floor, so the reservation
    /// and the grant agree.
    fn mip_reserved_slice(&self) -> Option<Duration> {
        let total = self.total?;
        let pb = &self.policy;
        if !pb.requests_mip_reservation() {
            return Some(Duration::ZERO);
        }
        let fraction = if pb.mip_min_fraction.is_finite() {
            pb.mip_min_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let secs = total
            .mul_f32(fraction)
            .as_secs()
            .clamp(pb.mip_min_secs, pb.mip_max_secs);
        Some(Duration::from_secs(secs))
    }

    /// Overall deadline (`start + total`), or `None` if unbounded.
    pub(in crate::commands::beta_crown) fn overall_deadline(&self) -> Option<Instant> {
        self.total.map(|d| self.start + d)
    }

    /// Deadline for a phase identified by a fractional budget.
    ///
    /// Returns `start + total * fraction`, or `None` if unbounded.
    pub(in crate::commands::beta_crown) fn phase_deadline(&self, fraction: f32) -> Option<Instant> {
        self.total
            .map(|d| self.start + d.mul_f32(fraction.clamp(0.0, 1.0)))
    }

    /// Phase deadline computed from **now** (not the ledger start):
    /// `now + total * fraction`, capped at the overall deadline.
    ///
    /// Use for phases that must still get their slice even when earlier phases
    /// ran long. The CROWN precheck previously computed `start + total * 0.20`,
    /// which was already in the past after a 50% attack phase — the root CROWN
    /// pass never ran and root bounds silently fell back to IBP
    /// (lsnc_relu: roots [-30, 30] vs thresholds ~0.4). #four-walls
    pub(in crate::commands::beta_crown) fn phase_deadline_from_now(
        &self,
        fraction: f32,
    ) -> Option<Instant> {
        let total = self.total?;
        let slice = total.mul_f32(fraction.clamp(0.0, 1.0));
        let overall = self.start + total;
        Some((Instant::now() + slice).min(overall))
    }

    /// Attack-phase deadline with a hard cap on tiny budgets.
    ///
    /// Fraction-based attack slices (upfront 0.20, disjunctive 0.50) are sized
    /// for 100s+ budgets. On tiny budgets (total <= 30s) they starve
    /// CROWN + BaB: lsnc_relu (20s ny budget) burned 10.0s in the disjunctive
    /// attack before any bound ran, while the preset intent was ~3s. Cap the
    /// attack slice at 15% of total there — by TIME, not by restart count,
    /// since per-eval cost varies by orders of magnitude across models.
    /// Presets that set an even smaller fraction keep their smaller slice.
    /// Disable the cap with `NY_NO_PGD_TIME_CAP=1`. #four-walls
    pub(in crate::commands::beta_crown) fn attack_phase_deadline(
        &self,
        fraction: f32,
    ) -> Option<Instant> {
        let total = self.total?;
        let capped_fraction = if total <= SMALL_BUDGET_TOTAL && !pgd_time_cap_disabled() {
            fraction.min(SMALL_BUDGET_ATTACK_FRACTION)
        } else {
            fraction
        };
        self.phase_deadline(capped_fraction)
    }

    /// Disjunctive global-PGD deadline: the fraction slice
    /// ([`Self::attack_phase_deadline`] with `disjunctive_pgd_fraction`, so the
    /// tiny-budget cap still applies), additionally clamped to the optional
    /// ABSOLUTE ceiling `policy.disjunctive_pgd_max_secs`.
    ///
    /// The fraction is anchored at the ledger start (`start + total*frac`), so on
    /// a hard UNSAT instance a large budget hands the falsifier a huge slice it
    /// wastes (it never finds a counterexample). `disjunctive_pgd_max_secs`
    /// bounds that in absolute terms; every second not spent here is reclaimed by
    /// the fast-path BaB, which re-bases on [`Self::remaining`]. `None` leaves the
    /// pure-fraction behavior unchanged.
    pub(in crate::commands::beta_crown) fn disjunctive_pgd_deadline(&self) -> Option<Instant> {
        let frac_deadline = self.attack_phase_deadline(self.policy.disjunctive_pgd_fraction)?;
        match self.policy.disjunctive_pgd_max_secs {
            Some(max_secs) => {
                let abs_cap = self.start + Duration::from_secs(max_secs);
                Some(frac_deadline.min(abs_cap))
            }
            None => Some(frac_deadline),
        }
    }

    /// Wall-clock time remaining, or `None` if unbounded.
    pub(in crate::commands::beta_crown) fn remaining(&self) -> Option<Duration> {
        self.total.map(|d| d.saturating_sub(self.start.elapsed()))
    }

    /// Remaining wall-clock seconds, clamped to `[0, u64::MAX]`.
    ///
    /// Returns `u64::MAX` if the timeout is unbounded.
    pub(in crate::commands::beta_crown) fn remaining_secs_clamped(&self) -> u64 {
        match self.remaining() {
            Some(d) => d.as_secs(),
            None => u64::MAX,
        }
    }

    /// Upfront PGD deadline: `start + total * upfront_pgd_fraction`,
    /// time-capped on tiny budgets (see [`Self::attack_phase_deadline`]).
    pub(in crate::commands::beta_crown) fn upfront_pgd_deadline(&self) -> Option<Instant> {
        self.attack_phase_deadline(self.policy.upfront_pgd_fraction)
    }

    /// BaB deadline: `start + total * (1 - post_bab_pgd_fraction)`.
    ///
    /// Used by CLI paths to thread the wall-clock deadline into BaB engine
    /// entry points, so the engine derives its phase budgets from remaining
    /// time rather than the original configured timeout (#4321).
    pub(in crate::commands::beta_crown) fn bab_deadline(&self) -> Option<Instant> {
        let total = self.total?;
        let frac = self.policy.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let mut bab = total.mul_f32(1.0 - frac);
        // FIX 2: with the Graph-MIP escalation armed, reserve its slice INSIDE
        // the scored budget so the escalation runs before the watchdog, not
        // after it. BaB keeps at least half the total (a pathological policy
        // cannot starve it). The runtime ledger uses the same default-on gate
        // as dispatch; the exact-zero kill switch and category policies that
        // request a zero MIP slice leave this unarmed.
        if self.mip_reservation_armed {
            if let Some(reserved) = self.mip_reserved_slice() {
                bab = bab.saturating_sub(reserved).max(total.mul_f32(0.5));
            }
        }
        Some(self.start + bab)
    }

    /// Access the underlying policy for callers that need specific fractions.
    pub(in crate::commands::beta_crown) fn policy(&self) -> &PhaseBudgetConfig {
        &self.policy
    }

    /// Per-clause timeout for constraint iteration.
    ///
    /// - **Conjunctive** (SAFE if ANY violated): returns full remaining time
    ///   since we early-exit on first success.
    /// - **Disjunctive** (SAFE if ALL violated): splits remaining time evenly
    ///   across `remaining_clauses`, with a floor of `min_clause_secs`.
    ///
    /// Returns `None` if the timeout is unbounded.
    pub(in crate::commands::beta_crown) fn per_clause_timeout(
        &self,
        remaining_clauses: usize,
        disjunctive: bool,
    ) -> Option<Duration> {
        let remaining = self.remaining()?;
        if !disjunctive || remaining_clauses == 0 {
            return Some(remaining);
        }
        let n = remaining_clauses.max(1) as u32;
        let per_clause = remaining / n;
        let min = Duration::from_secs(self.policy.min_clause_secs);
        Some(per_clause.max(min))
    }

    /// MIP fallback timeout: actual remaining wall-clock seconds.
    ///
    /// `mip_min_fraction` / `mip_min_secs` size the reservation made by
    /// [`Self::bab_deadline`]; they must never inflate the solver grant after
    /// that deadline has already been consumed.  In particular, an unreserved
    /// auto-MIP fallback used to turn `0s` remaining into a 25--30s timeout and
    /// was then killed by the outer VNN-COMP watchdog before it could report
    /// the already-sound BaB timeout.
    ///
    /// Returns `None` if the timeout is unbounded.
    // Only the mip-feature escalation path consumes this at runtime; the unit
    // test below still exercises it, so the allow only applies to non-mip
    // non-test builds.
    #[cfg_attr(all(not(feature = "mip"), not(test)), allow(dead_code))]
    pub(in crate::commands::beta_crown) fn mip_timeout(&self) -> Option<u64> {
        let total = self.total?;
        let remaining_secs = self.remaining()?.as_secs();
        // #postbab-small-budget: on small internal budgets (scored 20-30s
        // instances, internal <= 25s) the auto-MIP fallback taking the FULL
        // remaining wall ran the verify to its internal deadline, so the
        // scored-budget leftover shrank below the post-BaB falsification
        // lane's minimum and the lane never started — losing razor-thin SAT
        // rows (safenlp class) that the same binary wins whenever the lane
        // runs. Reserve 3s of the final MIP grant there so the verify returns
        // early enough for the attack window. Measured (2026-07-20,
        // hyperrectangle_1997): the reserved seconds came from an ABANDONED
        // final phase-split slice (15/16 subproblems, no verdict), so the
        // reservation costs nothing on that class; a genuinely-late MIP proof
        // loses at most 3s of its final slice. Larger budgets are unchanged.
        let small_budget_attack_reserve = if total <= Duration::from_secs(25) {
            3
        } else {
            0
        };
        // Hard invariant: a phase-local timeout may not exceed the enclosing
        // wall-clock budget.  Reservation (armed or not) only determines how
        // much time BaB leaves; it does not authorize borrowing beyond the
        // overall deadline.
        Some(remaining_secs.saturating_sub(small_budget_attack_reserve))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_unbounded_timeout_returns_none_deadlines() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.overall_deadline().is_none());
        assert!(ledger.upfront_pgd_deadline().is_none());
        assert!(ledger.remaining().is_none());
        assert_eq!(ledger.remaining_secs_clamped(), u64::MAX);
    }

    #[test]
    fn ledger_bounded_timeout_produces_deadlines() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        assert!(ledger.overall_deadline().is_some());
        assert!(ledger.upfront_pgd_deadline().is_some());
        assert!(ledger.remaining().is_some());
        assert!(ledger.remaining_secs_clamped() <= 100);
    }

    #[test]
    fn upfront_pgd_deadline_matches_current_attack_budget_behavior() {
        // Current behavior: attack_budget.rs line 12-17 → timeout / 5 = 0.20
        let ledger = PhaseBudgetLedger::new(150, PhaseBudgetConfig::default());
        let deadline = ledger.upfront_pgd_deadline().expect("bounded");
        let budget = deadline.duration_since(ledger.start);
        // 150 * 0.20 = 30 seconds
        assert_eq!(budget, Duration::from_secs(30));
    }

    #[test]
    fn phase_deadline_fraction_zero_returns_start() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let deadline = ledger.phase_deadline(0.0).expect("bounded");
        // fraction=0 → deadline = start
        let budget = deadline.duration_since(ledger.start);
        assert_eq!(budget, Duration::ZERO);
    }

    #[test]
    fn phase_deadline_fraction_one_returns_overall() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let phase = ledger.phase_deadline(1.0).expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert_eq!(phase, overall);
    }

    #[test]
    fn per_clause_timeout_conjunctive_returns_remaining() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let per_clause = ledger.per_clause_timeout(5, false).expect("bounded");
        // Conjunctive: full remaining time, not divided by 5.
        assert!(per_clause.as_secs() > 10);
    }

    #[test]
    fn per_clause_timeout_disjunctive_splits_evenly() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let per_clause = ledger.per_clause_timeout(10, true).expect("bounded");
        // Disjunctive: ~100s / 10 = ~10s, floor is min_clause_secs (1).
        assert!(per_clause.as_secs() >= 1);
        assert!(per_clause.as_secs() <= 15);
    }

    #[test]
    fn per_clause_timeout_unbounded_returns_none() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.per_clause_timeout(5, true).is_none());
    }

    #[test]
    fn mip_timeout_returns_actual_remaining_budget() {
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let mip = ledger.mip_timeout().expect("bounded");
        assert!(mip <= 100);
        assert!(mip >= 99);
    }

    #[test]
    fn mip_timeout_small_budget_reserves_attack_window() {
        // #postbab-small-budget: internal totals <= 25s reserve 3s of the MIP
        // grant so the post-BaB falsification lane keeps a scored-budget
        // window (safenlp-class 20s instances → internal 15s → grant <= 12s).
        let ledger = PhaseBudgetLedger::new(15, PhaseBudgetConfig::default());
        let mip = ledger.mip_timeout().expect("bounded");
        assert!(mip <= 12, "small-budget grant must reserve 3s (got {mip}s)");
        // Larger budgets are unchanged (no reserve).
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let mip = ledger.mip_timeout().expect("bounded");
        assert!(
            mip >= 99,
            "large-budget grant must be untouched (got {mip}s)"
        );
    }

    #[test]
    fn mip_timeout_unbounded_returns_none() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.mip_timeout().is_none());
    }

    #[test]
    fn bab_deadline_with_zero_reservation_matches_overall() {
        // With post_bab_pgd_fraction = 0.0 → no PGD reservation, so the BaB
        // deadline equals the overall deadline. (The *default* fraction is 0.10
        // — see PhaseBudgetConfig::default and its config tests — so this case
        // must set the fraction explicitly rather than rely on the default.)
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.0,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(100, policy).with_mip_reservation(false);
        let bab = ledger.bab_deadline().expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert_eq!(bab, overall);
    }

    #[test]
    fn bab_deadline_with_reservation_is_earlier_than_overall() {
        // Reserve 10% for PGD → bab_deadline = start + 100 * 0.90 = 90s.
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(100, policy).with_mip_reservation(false);
        let bab = ledger.bab_deadline().expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert!(bab < overall);
        // Verify the gap is ~10s (the PGD reservation).
        let gap = overall.duration_since(bab);
        assert!(gap.as_secs() >= 9 && gap.as_secs() <= 11);
    }

    #[test]
    fn bab_deadline_unbounded_returns_none() {
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(0, policy);
        assert!(ledger.bab_deadline().is_none());
    }

    #[test]
    fn attack_phase_deadline_caps_small_budgets_at_fifteen_percent() {
        // lsnc_relu class: 20s ny budget, disjunctive fraction 0.50 would give
        // 10s — the cap holds the attack slice to 0.15 * 20 = 3s.
        let ledger = PhaseBudgetLedger::new(20, PhaseBudgetConfig::default());
        let deadline = ledger.attack_phase_deadline(0.50).expect("bounded");
        let slice = deadline.duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(20).mul_f32(0.15));
    }

    #[test]
    fn attack_phase_deadline_keeps_smaller_preset_fractions_on_small_budgets() {
        // A preset that already asks for less than the cap keeps its slice.
        let ledger = PhaseBudgetLedger::new(20, PhaseBudgetConfig::default());
        let deadline = ledger.attack_phase_deadline(0.10).expect("bounded");
        let slice = deadline.duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(20).mul_f32(0.10));
    }

    #[test]
    fn attack_phase_deadline_uncapped_on_large_budgets() {
        // 180s budget: fraction-based slice unchanged (traffic_signs class).
        let ledger = PhaseBudgetLedger::new(180, PhaseBudgetConfig::default());
        let deadline = ledger.attack_phase_deadline(0.50).expect("bounded");
        let slice = deadline.duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(90));
    }

    #[test]
    fn attack_phase_deadline_unbounded_returns_none() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.attack_phase_deadline(0.50).is_none());
    }

    #[test]
    fn disjunctive_pgd_deadline_none_cap_matches_fraction() {
        // With no absolute cap, the disjunctive deadline equals the plain
        // fraction slice (0.20 * 200s = 40s).
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.20,
            disjunctive_pgd_max_secs: None,
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(200, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(40));
    }

    #[test]
    fn disjunctive_pgd_deadline_absolute_cap_bounds_large_budget() {
        // 0.20 * 300s = 60s by fraction, but the 30s absolute cap clamps it —
        // the reclaim/robustness fix for hold-heavy conv benchmarks.
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.20,
            disjunctive_pgd_max_secs: Some(30),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(300, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(30));
    }

    #[test]
    fn disjunctive_pgd_deadline_cap_inactive_when_fraction_smaller() {
        // At a small budget the fraction slice is already under the cap, so the
        // cap is a no-op (0.20 * 100s = 20s < 30s cap).
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_fraction: 0.20,
            disjunctive_pgd_max_secs: Some(30),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(100, policy);
        let slice = ledger
            .disjunctive_pgd_deadline()
            .expect("bounded")
            .duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(20));
    }

    #[test]
    fn disjunctive_pgd_deadline_unbounded_returns_none() {
        let policy = PhaseBudgetConfig {
            disjunctive_pgd_max_secs: Some(30),
            ..Default::default()
        };
        let ledger = PhaseBudgetLedger::new(0, policy);
        assert!(ledger.disjunctive_pgd_deadline().is_none());
    }

    #[test]
    fn upfront_pgd_deadline_capped_on_small_budgets() {
        // 25s budget with default upfront fraction 0.20 → 5s uncapped;
        // small-budget cap holds it to 0.15 * 25 = 3.75s.
        let ledger = PhaseBudgetLedger::new(25, PhaseBudgetConfig::default());
        let deadline = ledger.upfront_pgd_deadline().expect("bounded");
        let slice = deadline.duration_since(ledger.start);
        assert_eq!(slice, Duration::from_secs(25).mul_f32(0.15));
    }

    #[test]
    fn phase_deadline_from_now_grants_fresh_slice() {
        // Even immediately after ledger creation, the from-now deadline is at
        // least the fraction slice into the future (up to the overall cap).
        let ledger = PhaseBudgetLedger::new(100, PhaseBudgetConfig::default());
        let before = Instant::now();
        let deadline = ledger.phase_deadline_from_now(0.20).expect("bounded");
        // Slice ≈ 20s from now; allow generous slack below for scheduling.
        let slice = deadline.duration_since(before);
        assert!(slice >= Duration::from_secs(19));
        assert!(slice <= Duration::from_secs(21));
    }

    #[test]
    fn phase_deadline_from_now_capped_at_overall() {
        // fraction 1.0 from now would exceed the overall deadline; the result
        // must be clamped to it.
        let ledger = PhaseBudgetLedger::new(10, PhaseBudgetConfig::default());
        let deadline = ledger.phase_deadline_from_now(1.0).expect("bounded");
        let overall = ledger.overall_deadline().expect("bounded");
        assert!(deadline <= overall);
    }

    #[test]
    fn phase_deadline_from_now_unbounded_returns_none() {
        let ledger = PhaseBudgetLedger::new(0, PhaseBudgetConfig::default());
        assert!(ledger.phase_deadline_from_now(0.20).is_none());
    }

    /// FIX 2 — armed reservation math: the BaB deadline carves the MIP slice
    /// (`total*mip_min_fraction` clamped `[mip_min_secs, mip_max_secs]`) out of
    /// the scored budget, floored at half the total.
    #[test]
    fn armed_bab_deadline_reserves_the_mip_slice() {
        let policy = PhaseBudgetConfig::default();
        let unarmed = PhaseBudgetLedger::new(120, policy.clone()).with_mip_reservation(false);
        let armed = PhaseBudgetLedger::new(120, policy.clone()).with_mip_reservation(true);
        let reserved = armed.mip_reserved_slice().expect("bounded slice");
        // Default: 120*0.25 = 30 clamped [mip_min_secs, mip_max_secs].
        let expected = Duration::from_secs((120f32 * policy.mip_min_fraction) as u64)
            .min(Duration::from_secs(policy.mip_max_secs))
            .max(Duration::from_secs(policy.mip_min_secs));
        assert_eq!(
            reserved, expected,
            "reservation = mip_timeout's floor formula"
        );

        let unarmed_dl = unarmed.bab_deadline().expect("bounded");
        let armed_dl = armed.bab_deadline().expect("bounded");
        // The armed BaB deadline is EARLIER by ~the reserved slice (start times
        // differ by nanoseconds between the two ledgers; allow 1 s slack).
        let delta = unarmed_dl.saturating_duration_since(armed_dl);
        assert!(
            delta + Duration::from_secs(1) >= reserved && delta <= reserved,
            "armed BaB deadline must reserve the MIP slice (delta={delta:?}, reserved={reserved:?})"
        );
    }

    #[test]
    fn zero_mip_policy_disarms_and_preserves_the_exact_bab_deadline() {
        let policy = PhaseBudgetConfig {
            post_bab_pgd_fraction: 0.10,
            mip_min_fraction: 0.0,
            mip_min_secs: 0,
            ..Default::default()
        };
        assert!(!policy.requests_mip_reservation());

        let unarmed = PhaseBudgetLedger::new(64, policy.clone()).with_mip_reservation(false);
        let mut forced_armed = PhaseBudgetLedger::new(64, policy).with_mip_reservation(true);
        // Compare the policy arithmetic exactly, without constructor-time skew.
        forced_armed.start = unarmed.start;

        assert_eq!(
            forced_armed.mip_reserved_slice(),
            Some(Duration::ZERO),
            "a zero policy must carve out no hidden minimum"
        );
        assert_eq!(forced_armed.bab_deadline(), unarmed.bab_deadline());
        assert_eq!(
            forced_armed.bab_deadline(),
            Some(unarmed.start + Duration::from_secs(64).mul_f32(0.90))
        );

        let runtime = PhaseBudgetLedger::new(
            64,
            PhaseBudgetConfig {
                mip_min_fraction: 0.0,
                mip_min_secs: 0,
                ..Default::default()
            },
        );
        assert!(
            !runtime.mip_reservation_armed,
            "the runtime constructor must share the zero-policy admission predicate"
        );
    }

    #[test]
    fn unordered_mip_reservation_clamp_declines_without_panicking() {
        let policy = PhaseBudgetConfig {
            mip_min_fraction: 0.25,
            mip_min_secs: 30,
            mip_max_secs: 5,
            ..Default::default()
        };
        assert!(!policy.requests_mip_reservation());

        let ledger = PhaseBudgetLedger::new(64, policy).with_mip_reservation(true);
        assert_eq!(ledger.mip_reserved_slice(), Some(Duration::ZERO));
    }

    #[cfg(feature = "mip")]
    #[test]
    fn default_on_graph_mip_arms_the_runtime_ledger_when_unset() {
        assert!(
            std::env::var_os("NY_GRAPH_MIP").is_none(),
            "run this regression with NY_GRAPH_MIP unset"
        );
        let ledger = PhaseBudgetLedger::new(120, PhaseBudgetConfig::default());
        assert!(
            ledger.mip_reservation_armed,
            "the runtime ledger must reserve a slice under the default-on gate"
        );
    }

    /// FIX 2 — the armed BaB deadline never drops below half the total budget
    /// (a pathological reservation cannot starve BaB).
    #[test]
    fn armed_bab_deadline_keeps_half_the_budget() {
        let policy = PhaseBudgetConfig {
            mip_min_fraction: 1.0, // pathological: reserve everything
            mip_min_secs: 0,
            mip_max_secs: u64::MAX,
            ..Default::default()
        };
        let armed = PhaseBudgetLedger::new(100, policy).with_mip_reservation(true);
        let dl = armed.bab_deadline().expect("bounded");
        let bab = dl.saturating_duration_since(armed.start);
        assert!(
            bab >= Duration::from_secs(50),
            "BaB must keep at least half the total (got {bab:?})"
        );
    }

    /// Every `mip_timeout` is capped at the actual remaining wall.  Reservation
    /// changes when BaB stops, not whether the fallback may overdraw the outer
    /// deadline.
    #[test]
    fn mip_timeout_capped_at_remaining_with_or_without_reservation() {
        let policy = PhaseBudgetConfig::default();
        let mut armed = PhaseBudgetLedger::new(120, policy.clone()).with_mip_reservation(true);
        // Simulate 118 s elapsed: remaining ≈ 2 s.
        armed.start = Instant::now()
            .checked_sub(Duration::from_secs(118))
            .unwrap();
        let granted = armed.mip_timeout().expect("bounded");
        assert!(
            granted <= 2,
            "armed grant must not exceed remaining (got {granted}s)"
        );

        // An explicitly unarmed ledger is subject to the same hard wall.
        let mut unarmed = PhaseBudgetLedger::new(120, policy).with_mip_reservation(false);
        unarmed.start = Instant::now()
            .checked_sub(Duration::from_secs(118))
            .unwrap();
        let granted = unarmed.mip_timeout().expect("bounded");
        assert!(
            granted <= 2,
            "unarmed grant must not exceed remaining (got {granted}s)"
        );
    }

    /// An explicitly unarmed BaB retains its legacy deadline, while the
    /// subsequent solver grant remains bounded by the actual wall clock.
    #[test]
    fn unarmed_bab_deadline_is_legacy_but_mip_grant_is_wall_clamped() {
        let policy = PhaseBudgetConfig::default();
        let ledger = PhaseBudgetLedger::new(100, policy.clone()).with_mip_reservation(false);
        let frac = policy.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let expected = ledger.start + Duration::from_secs(100).mul_f32(1.0 - frac);
        assert_eq!(ledger.bab_deadline(), Some(expected), "legacy BaB deadline");
        let granted = ledger.mip_timeout().expect("bounded");
        assert!(granted <= 100, "MIP grant cannot exceed total wall");
        assert!(granted >= 99, "fresh ledger should retain nearly all wall");
    }
}
