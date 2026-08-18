// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Domain bound verification checks for DomainList BaB.
//!
//! Consolidates the repeated verified/violated/unresolved check pattern that
//! appeared 7 times in the original monolithic `gpu_bab.rs`.

use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;

#[cfg(test)]
use crate::beta_crown::result::BabVerificationStatus;
#[cfg(test)]
use std::time::Instant;

/// Outcome of checking a domain's bounds against the verification threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum DomainCheckResult {
    /// Domain's bounds prove the property holds.
    Verified,
    /// Domain's bounds prove the property is violated.
    Violation,
    /// Domain's bounds are inconclusive; further splitting needed.
    Undecided,
}

/// Check whether a domain with the given bounds is verified, violated, or undecided.
///
/// # Arguments
/// * `lower` - Lower bound on the domain's objective
/// * `upper` - Upper bound on the domain's objective
/// * `threshold` - Verification threshold
/// * `verify_upper_bound` - If true, verifies `upper < threshold`; otherwise `lower > threshold`
pub(crate) fn check_domain_bounds(
    lower: f32,
    upper: f32,
    threshold: f32,
    verify_upper_bound: bool,
) -> DomainCheckResult {
    if BetaCrownConfig::domain_is_verified_for_mode(verify_upper_bound, lower, upper, threshold) {
        DomainCheckResult::Verified
    } else if BetaCrownConfig::domain_is_violation_for_mode(
        verify_upper_bound,
        lower,
        upper,
        threshold,
    ) {
        DomainCheckResult::Violation
    } else {
        DomainCheckResult::Undecided
    }
}

/// DomainList BaB loop state — now a type alias for the shared `GraphBabLifecycle`.
///
/// All builder methods (`build_result`, `build_result_with_bounds`,
/// `check_termination`, `build_final_result`) are provided by
/// `GraphBabLifecycle`. DomainList BaB callers use the same lifecycle type
/// as ReLU split and multi-objective verifiers.
///
/// Part of #1860 (graph BaB service convergence, Packet A).
pub(crate) type BabLoopState = GraphBabLifecycle;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_domain_bounds_verify_upper() {
        // verify_upper_bound=true: want upper < threshold
        assert_eq!(
            check_domain_bounds(0.0, 0.5, 1.0, true),
            DomainCheckResult::Verified,
        );
        assert_eq!(
            check_domain_bounds(1.5, 2.0, 1.0, true),
            DomainCheckResult::Violation,
        );
        assert_eq!(
            check_domain_bounds(0.5, 1.5, 1.0, true),
            DomainCheckResult::Undecided,
        );
        // Edge: lower == threshold is violation
        assert_eq!(
            check_domain_bounds(1.0, 2.0, 1.0, true),
            DomainCheckResult::Violation,
        );
    }

    #[test]
    fn test_check_domain_bounds_verify_lower() {
        // verify_upper_bound=false: want lower > threshold
        assert_eq!(
            check_domain_bounds(2.0, 3.0, 1.0, false),
            DomainCheckResult::Verified,
        );
        assert_eq!(
            check_domain_bounds(0.0, 0.5, 1.0, false),
            DomainCheckResult::Violation,
        );
        assert_eq!(
            check_domain_bounds(0.5, 1.5, 1.0, false),
            DomainCheckResult::Undecided,
        );
        // Edge: lower == threshold is undecided (not strictly >)
        assert_eq!(
            check_domain_bounds(1.0, 2.0, 1.0, false),
            DomainCheckResult::Undecided,
        );
    }

    /// NaN bounds always produce Undecided (#2922). This documents why callers
    /// must guard with is_finite() before calling check_domain_bounds — without
    /// the guard, NaN domains loop forever (Undecided → re-queue → re-check).
    #[test]
    fn test_check_domain_bounds_nan_always_undecided_2922() {
        // NaN lower
        assert_eq!(
            check_domain_bounds(f32::NAN, 1.0, 0.5, true),
            DomainCheckResult::Undecided,
        );
        assert_eq!(
            check_domain_bounds(f32::NAN, 1.0, 0.5, false),
            DomainCheckResult::Undecided,
        );
        // NaN upper
        assert_eq!(
            check_domain_bounds(0.0, f32::NAN, 0.5, true),
            DomainCheckResult::Undecided,
        );
        assert_eq!(
            check_domain_bounds(0.0, f32::NAN, 0.5, false),
            DomainCheckResult::Undecided,
        );
        // Both NaN
        assert_eq!(
            check_domain_bounds(f32::NAN, f32::NAN, 0.5, true),
            DomainCheckResult::Undecided,
        );
        assert_eq!(
            check_domain_bounds(f32::NAN, f32::NAN, 0.5, false),
            DomainCheckResult::Undecided,
        );
    }

    /// Inf bounds always produce Undecided (#2993). Inf indicates propagation
    /// failure (e.g., reciprocal zero-crossing), not a genuine result.
    #[test]
    fn test_check_domain_bounds_inf_always_undecided_2993() {
        // +Inf lower: the exact bug — lower >= threshold would have returned Violation
        assert_eq!(
            check_domain_bounds(f32::INFINITY, f32::INFINITY, 0.5, true),
            DomainCheckResult::Undecided,
        );
        assert_eq!(
            check_domain_bounds(f32::INFINITY, f32::INFINITY, 0.5, false),
            DomainCheckResult::Undecided,
        );
        // -Inf upper: NEG_INFINITY < threshold would have returned Verified
        assert_eq!(
            check_domain_bounds(f32::NEG_INFINITY, f32::NEG_INFINITY, 0.5, true),
            DomainCheckResult::Undecided,
        );
        assert_eq!(
            check_domain_bounds(f32::NEG_INFINITY, f32::NEG_INFINITY, 0.5, false),
            DomainCheckResult::Undecided,
        );
        // Mixed: finite lower, Inf upper
        assert_eq!(
            check_domain_bounds(0.0, f32::INFINITY, 0.5, true),
            DomainCheckResult::Undecided,
        );
        assert_eq!(
            check_domain_bounds(0.0, f32::INFINITY, 0.5, false),
            DomainCheckResult::Undecided,
        );
        // Mixed: Inf lower, finite upper
        assert_eq!(
            check_domain_bounds(f32::NEG_INFINITY, 1.0, 0.5, true),
            DomainCheckResult::Undecided,
        );
        assert_eq!(
            check_domain_bounds(f32::NEG_INFINITY, 1.0, 0.5, false),
            DomainCheckResult::Undecided,
        );
    }

    #[test]
    fn test_build_final_result_reports_propagation_failure_reason() {
        let mut state = BabLoopState::new(Instant::now());
        state.unresolved_due_to_propagation_failure = true;

        let result = state.build_final_result();
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("Child propagation failed for some domains"),
                    "unknown reason should include propagation failure, got: {reason}",
                );
            }
            other => unreachable!("expected Unknown result, got {other:?}"),
        }
    }

    #[test]
    fn test_build_final_result_combines_all_unresolved_reasons() {
        let mut state = BabLoopState::new(Instant::now());
        state.max_depth_reached = 7;
        state.unresolved_due_to_depth = true;
        state.unresolved_due_to_no_unstable_neurons = true;
        state.unresolved_due_to_genbab_no_split = true;
        state.unresolved_due_to_propagation_failure = true;

        let result = state.build_final_result();
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("Max depth 7 reached"),
                    "reason missing depth clause: {reason}",
                );
                assert!(
                    reason.contains("No unstable ReLU/Sign neurons left in some domains"),
                    "reason missing no-unstable-neurons clause: {reason}",
                );
                assert!(
                    reason.contains("GenBaB found no splittable nodes in some domains"),
                    "reason missing genbab clause: {reason}",
                );
                assert!(
                    reason.contains("Child propagation failed for some domains"),
                    "reason missing propagation-failure clause: {reason}",
                );
            }
            other => unreachable!("expected Unknown result, got {other:?}"),
        }
    }

    /// #1925: GenBaB no-split and no-unstable-neurons produce distinct reason
    /// strings in the terminal Unknown status.
    #[test]
    fn test_build_final_result_distinguishes_genbab_from_no_unstable_neurons() {
        // GenBaB-only path
        let mut state = BabLoopState::new(Instant::now());
        state.unresolved_due_to_genbab_no_split = true;
        let result = state.build_final_result();
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("GenBaB found no splittable nodes"),
                    "genbab reason expected, got: {reason}",
                );
                assert!(
                    !reason.contains("No unstable ReLU/Sign neurons"),
                    "genbab-only must NOT mention ReLU/Sign neurons, got: {reason}",
                );
            }
            other => unreachable!("expected Unknown for genbab, got {other:?}"),
        }

        // No-unstable-neurons-only path
        let mut state2 = BabLoopState::new(Instant::now());
        state2.unresolved_due_to_no_unstable_neurons = true;
        let result2 = state2.build_final_result();
        match result2.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("No unstable ReLU/Sign neurons"),
                    "no-unstable reason expected, got: {reason}",
                );
                assert!(
                    !reason.contains("GenBaB"),
                    "no-unstable-only must NOT mention GenBaB, got: {reason}",
                );
            }
            other => unreachable!("expected Unknown for no-unstable, got {other:?}"),
        }
    }
}
