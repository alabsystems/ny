// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Phase-level time budget configuration for β-CROWN verification.
//!
//! Centralizes the timeout policy that was previously scattered across
//! `attack_budget.rs`, `sequential.rs`, `graph.rs`, `disjunctive.rs`,
//! and `mod.rs` as hardcoded magic constants.
//!
//! Default values are tuned for VNN-COMP competitive performance
//! (20% warmup cap, 10% post-BaB PGD reservation).
//!
//! Part of #2206. Design: designs/2026-03-16-issue-2206-adaptive-phase-budgeting.md

use serde::{Deserialize, Serialize};

/// Phase-level time budget fractions for β-CROWN verification.
///
/// Each fraction is relative to the total verification timeout.
/// All callers derive their phase budgets from this single config
/// instead of computing local fractions from raw `timeout`.
///
/// Default values are tuned for VNN-COMP competitive performance:
/// 20% warmup cap, 10% post-BaB PGD reservation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhaseBudgetConfig {
    /// Fraction of total timeout allocated to initial alpha/CROWN/CROWN-IBP bounds.
    ///
    /// Default: `0.20` (warmup gets 20% of BaB timeout).
    /// On a 100s timeout with 10% PGD reservation: warmup gets 18s.
    pub initial_bounds_fraction: f32,

    /// Fraction of total timeout allocated to upfront PGD attack.
    ///
    /// Current behavior: `0.20` (upfront PGD gets 20% of timeout).
    /// Source: `verify/attack_budget.rs:12-17` → `timeout / 5`.
    pub upfront_pgd_fraction: f32,

    /// Fraction of total timeout allocated to reduced verification
    /// (sequential path only).
    ///
    /// Current behavior: `0.40` (sequential reduced verification gets 40%).
    /// Source: `verify/sequential.rs:89-113` → `timeout * 2 / 5`.
    pub reduced_verification_fraction: f32,

    /// Fraction of total timeout allocated to disjunctive global PGD.
    ///
    /// Current behavior: `0.50`.
    /// Source: `verify/disjunctive.rs:96-126`.
    pub disjunctive_pgd_fraction: f32,

    /// Fraction of total timeout allocated to disjunctive CROWN/alpha precheck.
    ///
    /// Current behavior: `0.20`.
    /// Source: `verify/disjunctive.rs:197-289`.
    pub disjunctive_precheck_fraction: f32,

    /// Minimum fraction of total timeout guaranteed for MIP fallback.
    ///
    /// Current behavior: `0.25` → `timeout / 4`.
    /// Source: `beta_crown/mod.rs:1193-1242`.
    pub mip_min_fraction: f32,

    /// Minimum MIP timeout in seconds (floor clamp).
    ///
    /// Current behavior: `5`.
    /// Source: `beta_crown/mod.rs:1193-1242`.
    pub mip_min_secs: u64,

    /// Maximum MIP timeout in seconds (ceiling clamp).
    ///
    /// Current behavior: `30`.
    /// Source: `beta_crown/mod.rs:1193-1242`.
    pub mip_max_secs: u64,

    /// Fraction of MIP timeout allocated to CROWN-IBP preprocessing.
    ///
    /// Current behavior: `0.05`.
    /// Source: `mip_highs_intermediate_bounds.rs:6-33`.
    pub mip_crown_ibp_fraction: f64,

    /// Minimum MIP CROWN-IBP preprocessing timeout in seconds.
    ///
    /// Current behavior: `0.25`.
    /// Source: `mip_highs_intermediate_bounds.rs:6-33`.
    pub mip_crown_ibp_min_secs: f64,

    /// Maximum MIP CROWN-IBP preprocessing timeout in seconds.
    ///
    /// Current behavior: `2.0`.
    /// Source: `mip_highs_intermediate_bounds.rs:6-33`.
    pub mip_crown_ibp_max_secs: f64,

    /// Minimum per-clause timeout in seconds.
    ///
    /// Current behavior: `1` (implicit floor in per-constraint iteration).
    pub min_clause_secs: u64,

    /// Fraction of total timeout reserved for post-BaB PGD attack.
    ///
    /// When BaB returns Timeout or Unknown, the engine tries PGD as a
    /// last-resort counterexample search. Without a reservation, BaB
    /// consumes the entire timeout and PGD's deadline is already expired.
    ///
    /// The BaB loop stops at `timeout * (1 - post_bab_pgd_fraction)` so
    /// PGD gets the remaining `timeout * post_bab_pgd_fraction`.
    ///
    /// Default: `0.10` (reserve 10% for PGD fallback).
    /// On a 100s timeout: BaB gets 90s, post-BaB PGD gets 10s.
    ///
    /// Part of #2206 acceptance criterion 3.
    pub post_bab_pgd_fraction: f32,

    /// Fraction of the REMAINING budget granted to the single adaptive
    /// attack-phase extension (#attack-extend) when the disjunctive attack was
    /// deadline-cut with a promising margin.
    ///
    /// Default: `0.15` (batteries-included ON). Set `0.0` in a preset to
    /// disable the extension for categories where the margin gate cannot
    /// discriminate — e.g. cgan_2023 band properties, where the reachable set
    /// hugs the disjunctive thresholds so every UNSAT instance reports a
    /// "promising" margin and the extension only starves the bound phases.
    /// `NY_ATTACK_EXTEND_FRAC` still overrides this at runtime;
    /// `NY_ATTACK_EXTEND=0` remains the global kill switch.
    pub attack_extension_fraction: f32,
    /// Optional ABSOLUTE ceiling (seconds) on the disjunctive global PGD phase,
    /// applied on top of [`Self::disjunctive_pgd_fraction`].
    ///
    /// The disjunctive PGD deadline is `start + total*frac`, anchored at the
    /// ledger start. On a HARD UNSAT (hold) instance the falsifier never finds a
    /// counterexample and runs its whole slice, so at large budgets `frac*total`
    /// balloons — e.g. cifar100 at a 300s budget hands PGD 0.40*300 = 120s of
    /// dead falsification before BaB, and a concurrent forward-linear warmer
    /// under `thread::scope` can stall the phase across that window. Since the
    /// disjunctive fast-path BaB re-bases on `ledger.remaining()`, any second
    /// not spent in PGD is automatically reclaimed for BaB.
    ///
    /// `Some(secs)` caps the phase at `min(frac*total, secs)` — a robustness
    /// floor for hold-heavy conv benchmarks where PGD beyond a few seconds is
    /// wasted (sat instances are falsified in a few seconds). `None` (default)
    /// preserves the pure-fraction behavior for every other benchmark — no
    /// regression. Only the disjunctive global PGD reads this; upfront PGD is
    /// untouched.
    #[serde(default)]
    pub disjunctive_pgd_max_secs: Option<u64>,
}

impl PhaseBudgetConfig {
    /// Whether this policy requests a nonzero whole-network MIP reservation.
    ///
    /// This is the shared admission predicate for the pre-BaB time slice, the
    /// root-bounds reuse stash, and whole-net Graph-MIP in Auto mode. A zero
    /// reservation must not trigger a potentially large bounds-map clone or a
    /// late bounds recompute merely because Graph-MIP is enabled; explicit MIP
    /// remains an override. The independent per-leaf MIP oracle receives its
    /// bounds directly and is deliberately outside this policy.
    pub fn requests_mip_reservation(&self) -> bool {
        self.mip_max_secs > 0
            && self.mip_min_secs <= self.mip_max_secs
            && ((self.mip_min_fraction.is_finite() && self.mip_min_fraction > 0.0)
                || self.mip_min_secs > 0)
    }
}

impl Default for PhaseBudgetConfig {
    fn default() -> Self {
        Self {
            initial_bounds_fraction: 0.20,
            upfront_pgd_fraction: 0.20,
            reduced_verification_fraction: 0.40,
            disjunctive_pgd_fraction: 0.50,
            disjunctive_precheck_fraction: 0.20,
            mip_min_fraction: 0.25,
            mip_min_secs: 5,
            mip_max_secs: 30,
            mip_crown_ibp_fraction: 0.05,
            mip_crown_ibp_min_secs: 0.25,
            mip_crown_ibp_max_secs: 2.0,
            min_clause_secs: 1,
            post_bab_pgd_fraction: 0.10,
            attack_extension_fraction: 0.15,
            disjunctive_pgd_max_secs: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_current_behavior() {
        let cfg = PhaseBudgetConfig::default();
        assert!(
            cfg.requests_mip_reservation(),
            "the default/MSCN policy must retain its whole-net MIP reservation"
        );
        // attack_budget.rs: timeout / 5 = 0.20
        assert!((cfg.upfront_pgd_fraction - 0.20).abs() < 1e-6);
        // sequential.rs: timeout * 2 / 5 = 0.40
        assert!((cfg.reduced_verification_fraction - 0.40).abs() < 1e-6);
        // disjunctive.rs: 50% for PGD
        assert!((cfg.disjunctive_pgd_fraction - 0.50).abs() < 1e-6);
        // disjunctive.rs: 20% for precheck
        assert!((cfg.disjunctive_precheck_fraction - 0.20).abs() < 1e-6);
        // mod.rs: timeout/4 minimum for MIP
        assert!((cfg.mip_min_fraction - 0.25).abs() < 1e-6);
        // mod.rs: 5..30s MIP clamp
        assert_eq!(cfg.mip_min_secs, 5);
        assert_eq!(cfg.mip_max_secs, 30);
        // mip_highs: 5%, 0.25..2.0s
        assert!((cfg.mip_crown_ibp_fraction - 0.05).abs() < 1e-9);
        assert!((cfg.mip_crown_ibp_min_secs - 0.25).abs() < 1e-9);
        assert!((cfg.mip_crown_ibp_max_secs - 2.0).abs() < 1e-9);
        // Initial bounds: 0.20 (warmup capped to 20% of BaB budget)
        assert!((cfg.initial_bounds_fraction - 0.20).abs() < 1e-6);
        // Post-BaB PGD: 0.10 (reserve 10% for PGD fallback)
        assert!((cfg.post_bab_pgd_fraction - 0.10).abs() < 1e-6);
        // Attack extension: 0.15 of remaining (matches the historical
        // DEFAULT_EXTENSION_FRACTION in attack_extension.rs)
        assert!((cfg.attack_extension_fraction - 0.15).abs() < 1e-6);
    }

    #[test]
    fn mip_reservation_request_requires_a_finite_nonzero_policy() {
        let mut cfg = PhaseBudgetConfig {
            mip_min_fraction: 0.0,
            mip_min_secs: 0,
            ..Default::default()
        };
        assert!(!cfg.requests_mip_reservation());

        cfg.mip_min_secs = 5;
        assert!(
            cfg.requests_mip_reservation(),
            "an explicit seconds floor independently requests a reservation"
        );
        cfg.mip_min_fraction = f32::NAN;
        assert!(
            cfg.requests_mip_reservation(),
            "a valid seconds floor remains usable when the fractional arm is non-finite"
        );

        cfg.mip_min_secs = 0;
        cfg.mip_min_fraction = 0.25;
        assert!(cfg.requests_mip_reservation());

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.25] {
            cfg.mip_min_fraction = invalid;
            assert!(
                !cfg.requests_mip_reservation(),
                "invalid/non-positive fraction {invalid:?} must not arm a reservation"
            );
        }

        cfg.mip_min_fraction = 0.25;
        cfg.mip_max_secs = 0;
        assert!(
            !cfg.requests_mip_reservation(),
            "a zero ceiling makes every reservation zero"
        );

        cfg.mip_min_secs = 6;
        cfg.mip_max_secs = 5;
        assert!(
            !cfg.requests_mip_reservation(),
            "unordered clamps must decline instead of panicking in deadline arithmetic"
        );
    }

    #[test]
    fn serde_roundtrip_preserves_defaults() {
        let cfg = PhaseBudgetConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let deserialized: PhaseBudgetConfig = serde_json::from_str(&json).expect("deserialize");
        assert!((deserialized.upfront_pgd_fraction - cfg.upfront_pgd_fraction).abs() < 1e-6);
        assert!((deserialized.mip_crown_ibp_fraction - cfg.mip_crown_ibp_fraction).abs() < 1e-9);
    }

    #[test]
    fn partial_override_preserves_other_defaults() {
        let json = r#"{"upfront_pgd_fraction": 0.10}"#;
        let cfg: PhaseBudgetConfig = serde_json::from_str(json).expect("partial deserialize");
        assert!((cfg.upfront_pgd_fraction - 0.10).abs() < 1e-6);
        // Other fields should remain at default
        assert!((cfg.reduced_verification_fraction - 0.40).abs() < 1e-6);
        assert!((cfg.disjunctive_pgd_fraction - 0.50).abs() < 1e-6);
        assert!((cfg.post_bab_pgd_fraction - 0.10).abs() < 1e-6);
    }

    #[test]
    fn post_bab_pgd_reservation_override() {
        let json = r#"{"post_bab_pgd_fraction": 0.05}"#;
        let cfg: PhaseBudgetConfig = serde_json::from_str(json).expect("partial deserialize");
        assert!((cfg.post_bab_pgd_fraction - 0.05).abs() < 1e-6);
        // Other fields should remain at default
        assert!((cfg.upfront_pgd_fraction - 0.20).abs() < 1e-6);
        assert!((cfg.initial_bounds_fraction - 0.20).abs() < 1e-6);
    }

    /// Verifies the Duration arithmetic used in all 6 graph BaB entry points
    /// for phase_budget warmup cap (#4095). The pattern:
    ///
    /// ```ignore
    /// let pgd_frac = config.post_bab_pgd_fraction.clamp(0.0, 0.5);
    /// let bab_timeout = timeout.mul_f32(1.0 - pgd_frac);
    /// let frac = config.initial_bounds_fraction.clamp(0.0, 1.0);
    /// let initial_deadline = start + bab_timeout.mul_f32(frac);
    /// ```
    #[test]
    fn graph_bab_warmup_cap_duration_arithmetic() {
        use std::time::Duration;

        let timeout = Duration::from_secs(100);

        // Competitive settings: 15% warmup, 10% PGD reservation
        let cfg: PhaseBudgetConfig = serde_json::from_str(
            r#"{"initial_bounds_fraction": 0.15, "post_bab_pgd_fraction": 0.10}"#,
        )
        .expect("deserialize");

        let pgd_frac = cfg.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let bab_timeout = timeout.mul_f32(1.0 - pgd_frac);
        let frac = cfg.initial_bounds_fraction.clamp(0.0, 1.0);
        let initial_budget = bab_timeout.mul_f32(frac);

        // BaB gets 90% of total (100s * 0.9 = 90s)
        assert!(
            (bab_timeout.as_secs_f64() - 90.0).abs() < 0.5,
            "bab_timeout should be ~90s, got {:.1}s",
            bab_timeout.as_secs_f64()
        );

        // Warmup gets 15% of BaB budget (90s * 0.15 = 13.5s)
        assert!(
            (initial_budget.as_secs_f64() - 13.5).abs() < 0.5,
            "initial_budget should be ~13.5s, got {:.1}s",
            initial_budget.as_secs_f64()
        );
    }

    /// Default config allocates competitive budget splits:
    /// BaB gets 90% (post_bab_pgd=0.10), warmup gets 20% of BaB budget.
    /// On a 100s timeout: warmup=18s, BaB=72s, post-BaB PGD=10s.
    #[test]
    fn graph_bab_default_config_competitive_budgets() {
        use std::time::Duration;

        let timeout = Duration::from_secs(100);
        let cfg = PhaseBudgetConfig::default();

        let pgd_frac = cfg.post_bab_pgd_fraction.clamp(0.0, 0.5);
        let bab_timeout = timeout.mul_f32(1.0 - pgd_frac);
        let frac = cfg.initial_bounds_fraction.clamp(0.0, 1.0);
        let initial_budget = bab_timeout.mul_f32(frac);

        // BaB gets 90% of total (100s * 0.9 = 90s)
        assert!(
            (bab_timeout.as_secs_f64() - 90.0).abs() < 0.5,
            "bab_timeout should be ~90s, got {:.1}s",
            bab_timeout.as_secs_f64()
        );

        // Warmup gets 20% of BaB budget (90s * 0.2 = 18s)
        assert!(
            (initial_budget.as_secs_f64() - 18.0).abs() < 0.5,
            "initial_budget should be ~18s, got {:.1}s",
            initial_budget.as_secs_f64()
        );
    }
}
