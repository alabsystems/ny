// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_propagate::BetaCrownConfig;

pub(super) fn graph_upfront_pgd_budget(config: &BetaCrownConfig) -> (usize, usize) {
    (config.pgd_restarts, config.pgd_steps.max(50))
}

/// Original PGD deadline computation (kept for test compatibility).
/// Production code now uses `PhaseBudgetLedger::upfront_pgd_deadline()`.
#[cfg(test)]
pub(super) fn upfront_pgd_deadline(
    start_time: std::time::Instant,
    timeout: u64,
) -> Option<std::time::Instant> {
    if timeout > 0 {
        Some(start_time + std::time::Duration::from_secs(timeout) / 5)
    } else {
        None
    }
}

pub(super) fn upfront_conjunctive_sampling_budget(
    pgd_restarts: usize,
    pgd_steps: usize,
) -> (usize, usize) {
    (pgd_restarts, pgd_steps.max(20))
}

/// Budget for disjunctive sampling attack (random + SPSA).
///
/// The deadline (`overall_timeout / 2` in disjunctive.rs) is the real budget
/// control — it cuts the attack short if evaluations are expensive. Both
/// restart count and step count pass through from the preset config.
/// The step floor of 10 ensures minimal gradient signal even for low presets.
///
/// Previously this capped restarts at 200, but that silently overrode configs
/// like lsnc_relu (10000 restarts) by 50x. The PhaseBudgetLedger deadline
/// already prevents runaway PGD, making the restart cap redundant.
pub(super) fn disjunctive_sampling_budget(pgd_restarts: usize, pgd_steps: usize) -> (usize, usize) {
    (pgd_restarts, pgd_steps.max(10))
}

/// Budget for direct disjunctive classification attack.
///
/// Ensures a minimum of 50 restarts for classification quality. No upper cap —
/// the phase deadline controls total PGD time.
pub(super) fn direct_disjunctive_classification_budget(
    pgd_restarts: usize,
    pgd_steps: usize,
) -> (usize, usize) {
    (pgd_restarts.max(50), pgd_steps.max(20))
}

#[cfg(test)]
mod tests {
    use super::{
        direct_disjunctive_classification_budget, disjunctive_sampling_budget,
        graph_upfront_pgd_budget, upfront_conjunctive_sampling_budget, upfront_pgd_deadline,
    };
    use ny_propagate::BetaCrownConfig;
    use std::time::{Duration, Instant};

    #[test]
    fn graph_upfront_budget_respects_low_restart_presets() {
        let config = BetaCrownConfig {
            pgd_restarts: 50,
            pgd_steps: 10,
            ..Default::default()
        };

        assert_eq!(graph_upfront_pgd_budget(&config), (50, 50));
    }

    #[test]
    fn upfront_pgd_deadline_reserves_eighty_percent_for_bab() {
        let start = Instant::now();
        let deadline = upfront_pgd_deadline(start, 150).expect("finite timeout should cap PGD");

        assert_eq!(deadline.duration_since(start), Duration::from_secs(30));
    }

    #[test]
    fn upfront_pgd_deadline_none_for_unbounded_timeout() {
        assert!(upfront_pgd_deadline(Instant::now(), 0).is_none());
    }

    #[test]
    fn conjunctive_sampling_budget_respects_low_restart_presets() {
        assert_eq!(upfront_conjunctive_sampling_budget(50, 5), (50, 20));
    }

    #[test]
    fn disjunctive_sampling_budget_respects_low_restart_presets() {
        // Floor of 10 steps for very low presets
        assert_eq!(disjunctive_sampling_budget(50, 5), (50, 10));
    }

    #[test]
    fn disjunctive_sampling_budget_passes_through_high_restarts() {
        // Restarts pass through uncapped — deadline controls budget
        assert_eq!(disjunctive_sampling_budget(500, 200), (500, 200));
    }

    #[test]
    fn disjunctive_sampling_budget_passes_through_very_high_restarts() {
        // lsnc_relu uses 10000 restarts — must not be capped
        assert_eq!(disjunctive_sampling_budget(10000, 50), (10000, 50));
    }

    #[test]
    fn disjunctive_sampling_budget_passes_through_moderate_steps() {
        assert_eq!(disjunctive_sampling_budget(50, 25), (50, 25));
    }

    #[test]
    fn direct_disjunctive_classification_budget_raises_low_restart_presets() {
        assert_eq!(direct_disjunctive_classification_budget(10, 5), (50, 20));
    }

    #[test]
    fn direct_disjunctive_classification_budget_passes_through_high_restarts() {
        // No upper cap — deadline controls budget
        assert_eq!(direct_disjunctive_classification_budget(500, 25), (500, 25));
    }

    #[test]
    fn direct_disjunctive_classification_budget_passes_through_10k_restarts() {
        assert_eq!(
            direct_disjunctive_classification_budget(10000, 100),
            (10000, 100)
        );
    }
}
