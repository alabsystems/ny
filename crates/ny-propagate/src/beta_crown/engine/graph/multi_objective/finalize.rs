// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Final result assembly for multi-objective graph BaB verification.
//!
//! Thin adapter over `GraphBabLifecycle::build_final_result()` that adds
//! the multi-objective-specific "could not verify all objectives" fallback.
//!
//! Part of #1860 (graph BaB service convergence, Packet A).

use std::time::Duration;

use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};

/// Assemble the terminal result after queue exhaustion/termination checks.
///
/// If any domains were unresolved (depth, no-branch, violated-drop, or
/// propagation failure), returns Unknown with cause-specific reasons.
/// If the queue is empty and domains were verified, returns Verified.
/// Otherwise returns Unknown "could not verify all objectives".
///
/// #1861/#1866: Check for unresolved domains before claiming Verified.
pub(super) fn finalize_multi_objective_result(
    lifecycle: &GraphBabLifecycle,
    queue_is_empty: bool,
) -> BetaCrownResult {
    if lifecycle.has_unresolved() {
        return lifecycle.build_result(BabVerificationStatus::Unknown {
            reason: lifecycle.unresolved_reason(),
        });
    }

    if lifecycle.domains_verified > 0 && queue_is_empty {
        lifecycle.build_result(BabVerificationStatus::Verified)
    } else {
        lifecycle.build_result(BabVerificationStatus::Unknown {
            reason: "Could not verify all objectives in explored domains".to_string(),
        })
    }
}

/// Resolve the outer-loop boundary without losing a proof completed by the
/// preceding batch.
///
/// A clean queue exhaustion is terminal evidence: every region popped from the
/// frontier has been folded into a verified close. It therefore has precedence
/// over a deadline/domain-limit check performed on the *next* loop iteration.
/// Checking the budget first can convert a proof completed by the final batch
/// into `Timeout`.
///
/// This ordering is sound because [`finalize_multi_objective_result`] checks all
/// unresolved flags before it can return `Verified`. If the empty frontier has
/// an unresolved flag, timeout/domain-limit still takes precedence so
/// deadline-truncated child work remains `Timeout`, not a misleading ordinary
/// queue drain. A non-empty frontier always observes the normal policy.
pub(super) fn resolve_multi_objective_loop_boundary(
    lifecycle: &GraphBabLifecycle,
    queue_is_empty: bool,
    timeout: Duration,
    max_domains: usize,
) -> Option<BetaCrownResult> {
    if queue_is_empty && !lifecycle.has_unresolved() {
        return Some(finalize_multi_objective_result(lifecycle, true));
    }
    lifecycle
        .check_termination(timeout, max_domains)
        .or_else(|| queue_is_empty.then(|| finalize_multi_objective_result(lifecycle, true)))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn expired_lifecycle() -> GraphBabLifecycle {
        GraphBabLifecycle::new(
            Instant::now()
                .checked_sub(Duration::from_secs(2))
                .expect("test instant subtraction"),
        )
    }

    /// A proof completed by the last batch must not be overwritten by the
    /// next-iteration deadline check.
    #[test]
    fn drained_verified_frontier_wins_over_expired_deadline() {
        let mut lifecycle = expired_lifecycle();
        lifecycle.domains_verified = 96;

        let result = resolve_multi_objective_loop_boundary(
            &lifecycle,
            true,
            Duration::from_secs(1),
            usize::MAX,
        )
        .expect("drained frontier is terminal");

        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    /// Queue exhaustion is not itself a proof when any region was lost to an
    /// unresolved path. An expired budget remains a timeout.
    #[test]
    fn drained_unresolved_frontier_at_deadline_stays_timeout() {
        let mut lifecycle = expired_lifecycle();
        lifecycle.domains_verified = 95;
        lifecycle.unresolved_due_to_propagation_failure = true;

        let result = resolve_multi_objective_loop_boundary(
            &lifecycle,
            true,
            Duration::from_secs(1),
            usize::MAX,
        )
        .expect("drained frontier is terminal");

        assert_eq!(result.result, BabVerificationStatus::Timeout);
    }

    #[test]
    fn drained_unresolved_frontier_before_deadline_is_unknown() {
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        lifecycle.domains_verified = 95;
        lifecycle.unresolved_due_to_propagation_failure = true;

        let result = resolve_multi_objective_loop_boundary(
            &lifecycle,
            true,
            Duration::from_mins(1),
            usize::MAX,
        )
        .expect("drained frontier is terminal");

        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(reason.contains("Child propagation failed"));
            }
            other => panic!("unresolved drained frontier must be Unknown, got {other:?}"),
        }
    }

    #[test]
    fn nonempty_frontier_still_times_out() {
        let lifecycle = expired_lifecycle();
        let result = resolve_multi_objective_loop_boundary(
            &lifecycle,
            false,
            Duration::from_secs(1),
            usize::MAX,
        )
        .expect("expired nonempty frontier must terminate");

        assert_eq!(result.result, BabVerificationStatus::Timeout);
    }

    #[test]
    fn drained_verified_frontier_wins_over_domain_limit() {
        let mut lifecycle = GraphBabLifecycle::new(Instant::now());
        lifecycle.domains_explored = 100;
        lifecycle.domains_verified = 100;

        let result =
            resolve_multi_objective_loop_boundary(&lifecycle, true, Duration::from_mins(1), 100)
                .expect("drained frontier is terminal");

        assert_eq!(result.result, BabVerificationStatus::Verified);
    }
}
