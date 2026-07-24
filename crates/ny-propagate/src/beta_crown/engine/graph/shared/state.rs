// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph-BaB lifecycle state and result assembly.
//!
//! `GraphBabLifecycle` collects the running counters and unresolved flags
//! shared across all three graph-BaB verifiers (ReLU split, multi-objective,
//! GPU BaB). Builder methods replace the 20+ inline `BetaCrownResult` struct
//! literals scattered across those verifiers.
//!
//! Design: `designs/2026-03-14-issue-1860-graph-bab-service-convergence.md`
//! Issue: #1860 (Packet A)

use std::time::{Duration, Instant};

use ny_tensor::BoundedTensor;

use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};

/// Shared lifecycle state for graph-BaB verification loops.
///
/// Accumulates counters, unresolved flags, and the start timestamp needed
/// to construct `BetaCrownResult` at any termination point. All three
/// graph-BaB modes (single-objective ReLU split, multi-objective, GPU BaB)
/// share this type instead of maintaining separate state structs.
///
/// # Unresolved reason flags
///
/// Multiple distinct conditions can leave domains unresolved. Each gets a
/// separate flag so the terminal `Unknown` reason string preserves the actual
/// cause instead of collapsing into a generic message.
#[derive(Clone)]
pub(crate) struct GraphBabLifecycle {
    pub(crate) start_time: Instant,
    pub(crate) domains_explored: usize,
    pub(crate) domains_verified: usize,
    pub(crate) max_depth_reached: usize,
    pub(crate) cuts_generated: usize,
    pub(crate) unresolved_due_to_depth: bool,
    pub(crate) unresolved_due_to_unsplittable: bool,
    pub(crate) unresolved_due_to_no_branch: bool,
    pub(crate) unresolved_due_to_no_unstable_neurons: bool,
    pub(crate) unresolved_due_to_genbab_no_split: bool,
    pub(crate) unresolved_due_to_propagation_failure: bool,
    pub(crate) unresolved_due_to_violated_drop: bool,
    pub(crate) unresolved_due_to_eviction: bool,
}

impl GraphBabLifecycle {
    pub(crate) fn new(start_time: Instant) -> Self {
        Self {
            start_time,
            domains_explored: 0,
            domains_verified: 0,
            max_depth_reached: 0,
            cuts_generated: 0,
            unresolved_due_to_depth: false,
            unresolved_due_to_unsplittable: false,
            unresolved_due_to_no_branch: false,
            unresolved_due_to_no_unstable_neurons: false,
            unresolved_due_to_genbab_no_split: false,
            unresolved_due_to_propagation_failure: false,
            unresolved_due_to_violated_drop: false,
            unresolved_due_to_eviction: false,
        }
    }

    /// Build a `BetaCrownResult` with the given status and current counters.
    pub(crate) fn build_result(&self, status: BabVerificationStatus) -> BetaCrownResult {
        BetaCrownResult {
            result: status,
            domains_explored: self.domains_explored,
            time_elapsed: self.start_time.elapsed(),
            max_depth_reached: self.max_depth_reached,
            output_bounds: None,
            cuts_generated: self.cuts_generated,
            domains_verified: self.domains_verified,
        }
    }

    /// Build a `BetaCrownResult` with output bounds (used for root early exits).
    pub(crate) fn build_result_with_bounds(
        &self,
        status: BabVerificationStatus,
        output_bounds: BoundedTensor,
    ) -> BetaCrownResult {
        BetaCrownResult {
            result: status,
            domains_explored: self.domains_explored,
            time_elapsed: self.start_time.elapsed(),
            max_depth_reached: self.max_depth_reached,
            output_bounds: Some(output_bounds),
            cuts_generated: self.cuts_generated,
            domains_verified: self.domains_verified,
        }
    }

    /// Build a `Timeout` result.
    pub(crate) fn timeout_result(&self) -> BetaCrownResult {
        self.build_result(BabVerificationStatus::Timeout)
    }

    /// Build an `Unknown` result for hitting the domain exploration limit.
    pub(crate) fn domain_limit_result(&self, max_domains: usize) -> BetaCrownResult {
        self.build_result(BabVerificationStatus::Unknown {
            reason: format!("Domain limit {} reached", max_domains),
        })
    }

    /// Check timeout and domain-limit termination conditions.
    /// Returns `Some(result)` if the loop should terminate.
    pub(crate) fn check_termination(
        &self,
        timeout: Duration,
        max_domains: usize,
    ) -> Option<BetaCrownResult> {
        if self.start_time.elapsed() > timeout {
            return Some(self.timeout_result());
        }
        if self.domains_explored >= max_domains {
            return Some(self.domain_limit_result(max_domains));
        }
        None
    }

    /// Whether any unresolved flags are set.
    pub(crate) fn has_unresolved(&self) -> bool {
        self.unresolved_due_to_depth
            || self.unresolved_due_to_unsplittable
            || self.unresolved_due_to_no_branch
            || self.unresolved_due_to_no_unstable_neurons
            || self.unresolved_due_to_genbab_no_split
            || self.unresolved_due_to_propagation_failure
            || self.unresolved_due_to_violated_drop
            || self.unresolved_due_to_eviction
    }

    /// Build a human-readable reason string from the active unresolved flags.
    pub(crate) fn unresolved_reason(&self) -> String {
        let mut parts = Vec::new();
        if self.unresolved_due_to_depth {
            parts.push(format!("Max depth {} reached", self.max_depth_reached));
        }
        if self.unresolved_due_to_unsplittable {
            parts.push(
                "Unsplittable domain (invalid split dimension, non-finite, or zero-width bounds)"
                    .to_string(),
            );
        }
        if self.unresolved_due_to_no_branch {
            parts.push("No unstable ReLU/Sign neurons left in some domains".to_string());
        }
        if self.unresolved_due_to_no_unstable_neurons {
            parts.push("No unstable ReLU/Sign neurons left in some domains (GPU BaB)".to_string());
        }
        if self.unresolved_due_to_genbab_no_split {
            parts.push("GenBaB found no splittable nodes in some domains".to_string());
        }
        if self.unresolved_due_to_propagation_failure {
            parts.push("Child propagation failed for some domains".to_string());
        }
        if self.unresolved_due_to_violated_drop {
            parts.push("Some domains conclusively violated the property".to_string());
        }
        if self.unresolved_due_to_eviction {
            parts.push(
                "Queue cap (max_queue_size) evicted unverified domains before they were explored"
                    .to_string(),
            );
        }
        parts.join("; ")
    }

    /// Build the final result after the BaB loop exits normally.
    ///
    /// If any domains were unresolved, returns Unknown with cause-specific
    /// reasons. Otherwise returns Verified (queue exhaustion = all verified).
    pub(crate) fn build_final_result(&self) -> BetaCrownResult {
        if self.has_unresolved() {
            self.build_result(BabVerificationStatus::Unknown {
                reason: self.unresolved_reason(),
            })
        } else {
            self.build_result(BabVerificationStatus::Verified)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lifecycle_new_defaults() {
        let lc = GraphBabLifecycle::new(Instant::now());
        assert_eq!(lc.domains_explored, 0);
        assert_eq!(lc.domains_verified, 0);
        assert_eq!(lc.max_depth_reached, 0);
        assert_eq!(lc.cuts_generated, 0);
        assert!(!lc.has_unresolved());
    }

    #[test]
    fn test_lifecycle_build_result_fields() {
        let lc = GraphBabLifecycle::new(Instant::now());
        let result = lc.build_result(BabVerificationStatus::Verified);
        assert_eq!(result.result, BabVerificationStatus::Verified);
        assert_eq!(result.domains_explored, 0);
        assert!(result.output_bounds.is_none());
        assert_eq!(result.cuts_generated, 0);
    }

    #[test]
    fn test_lifecycle_timeout_result() {
        let lc = GraphBabLifecycle::new(Instant::now());
        let result = lc.timeout_result();
        assert_eq!(result.result, BabVerificationStatus::Timeout);
    }

    #[test]
    fn test_lifecycle_domain_limit_result() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        lc.domains_explored = 50;
        let result = lc.domain_limit_result(100);
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(reason.contains("Domain limit 100 reached"));
            }
            other => unreachable!("expected Unknown, got {other:?}"),
        }
        assert_eq!(result.domains_explored, 50);
    }

    #[test]
    fn test_lifecycle_has_unresolved_flags() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        assert!(!lc.has_unresolved());

        lc.unresolved_due_to_depth = true;
        assert!(lc.has_unresolved());

        lc.unresolved_due_to_depth = false;
        lc.unresolved_due_to_violated_drop = true;
        assert!(lc.has_unresolved());
    }

    #[test]
    fn test_lifecycle_unsplittable_reason() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        lc.unresolved_due_to_unsplittable = true;

        let reason = lc.unresolved_reason();
        assert!(reason.contains("Unsplittable domain"));
        assert!(reason.contains("invalid split dimension"));
    }

    #[test]
    fn test_lifecycle_unresolved_reason_combines_all() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        lc.max_depth_reached = 7;
        lc.unresolved_due_to_depth = true;
        lc.unresolved_due_to_propagation_failure = true;
        lc.unresolved_due_to_violated_drop = true;

        let reason = lc.unresolved_reason();
        assert!(reason.contains("Max depth 7 reached"));
        assert!(reason.contains("Child propagation failed"));
        assert!(reason.contains("conclusively violated"));
    }

    #[test]
    fn test_lifecycle_build_final_result_verified() {
        let lc = GraphBabLifecycle::new(Instant::now());
        let result = lc.build_final_result();
        assert_eq!(result.result, BabVerificationStatus::Verified);
    }

    /// Queue-cap eviction discards unverified domains, so a drained queue
    /// must produce Unknown, never Verified.
    #[test]
    fn test_lifecycle_eviction_flag_forces_unknown() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        assert!(!lc.has_unresolved());

        lc.unresolved_due_to_eviction = true;
        assert!(lc.has_unresolved());

        let result = lc.build_final_result();
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("evicted unverified domains"),
                    "reason must attribute the Unknown to queue-cap eviction, got: {reason}"
                );
            }
            other => unreachable!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn test_lifecycle_build_final_result_unknown() {
        let mut lc = GraphBabLifecycle::new(Instant::now());
        lc.unresolved_due_to_no_branch = true;
        let result = lc.build_final_result();
        match result.result {
            BabVerificationStatus::Unknown { reason } => {
                assert!(reason.contains("No unstable ReLU/Sign neurons"));
            }
            other => unreachable!("expected Unknown, got {other:?}"),
        }
    }

    /// The sequential-BaB `no_branch` and GPU-BaB `no_unstable_neurons` flags
    /// must produce distinct reason strings so diagnostics identify the source.
    #[test]
    fn test_lifecycle_no_branch_vs_no_unstable_neurons_reasons_distinct() {
        let mut lc_seq = GraphBabLifecycle::new(Instant::now());
        lc_seq.unresolved_due_to_no_branch = true;
        let reason_seq = lc_seq.unresolved_reason();

        let mut lc_gpu = GraphBabLifecycle::new(Instant::now());
        lc_gpu.unresolved_due_to_no_unstable_neurons = true;
        let reason_gpu = lc_gpu.unresolved_reason();

        assert_eq!(
            reason_seq,
            "No unstable ReLU/Sign neurons left in some domains"
        );
        assert_eq!(
            reason_gpu,
            "No unstable ReLU/Sign neurons left in some domains (GPU BaB)"
        );
    }
}
