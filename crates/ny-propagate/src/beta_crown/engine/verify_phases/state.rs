// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use crate::beta_crown::conflict_clauses::ClauseStore;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};

pub(in crate::beta_crown::engine) struct BabLoopState {
    pub domains_explored: usize,
    pub max_depth: usize,
    pub domains_verified: usize,
    pub unresolved_due_to_depth: bool,
    pub unresolved_due_to_no_branch: bool,
    pub unresolved_due_to_unsplittable: bool,
    pub unresolved_due_to_propagation_failure: bool,
    pub cut_generation_enabled: bool,
    /// Per-run conflict clause store (NY_BAB_CLAUSE_LEARN=1; default disabled
    /// => byte-identical baseline). Lives here so both the record sites and
    /// the prune site reach it through the existing `&mut BabLoopState`.
    pub clause_store: ClauseStore,
    /// Domains closed as verified via clause subsumption WITHOUT a bound
    /// computation (subset of `domains_verified`).
    pub domains_clause_pruned: usize,
}

impl BabLoopState {
    pub(in crate::beta_crown::engine) fn new(cut_generation_enabled: bool) -> Self {
        Self {
            domains_explored: 0,
            max_depth: 0,
            domains_verified: 0,
            unresolved_due_to_depth: false,
            unresolved_due_to_no_branch: false,
            unresolved_due_to_unsplittable: false,
            unresolved_due_to_propagation_failure: false,
            cut_generation_enabled,
            clause_store: ClauseStore::disabled(),
            domains_clause_pruned: 0,
        }
    }

    pub(in crate::beta_crown::engine) fn has_unresolved(&self) -> bool {
        self.unresolved_due_to_propagation_failure
            || self.unresolved_due_to_depth
            || self.unresolved_due_to_no_branch
            || self.unresolved_due_to_unsplittable
    }

    pub(in crate::beta_crown::engine) fn unresolved_reason(&self, max_depth: usize) -> String {
        let mut reason_parts = Vec::new();
        if self.unresolved_due_to_propagation_failure {
            reason_parts.push("Child propagation failed for some domains".to_string());
        }
        if self.unresolved_due_to_depth {
            reason_parts.push(format!("Max depth {} reached", max_depth));
        }
        if self.unresolved_due_to_no_branch {
            reason_parts.push("No unstable ReLU/Sign neurons left in some domains".to_string());
        }
        if self.unresolved_due_to_unsplittable {
            reason_parts.push("No splittable input dimension left in some domains".to_string());
        }
        reason_parts.join("; ")
    }

    pub(in crate::beta_crown::engine) fn unknown_result(
        &self,
        start_time: Instant,
        cuts_generated: usize,
        reason: String,
    ) -> BetaCrownResult {
        BetaCrownResult {
            result: BabVerificationStatus::Unknown { reason },
            domains_explored: self.domains_explored,
            time_elapsed: start_time.elapsed(),
            max_depth_reached: self.max_depth,
            output_bounds: None,
            cuts_generated,
            domains_verified: self.domains_verified,
        }
    }

    pub(in crate::beta_crown::engine) fn verified_result(
        &self,
        start_time: Instant,
        cuts_generated: usize,
    ) -> BetaCrownResult {
        BetaCrownResult {
            result: BabVerificationStatus::Verified,
            domains_explored: self.domains_explored,
            time_elapsed: start_time.elapsed(),
            max_depth_reached: self.max_depth,
            output_bounds: None,
            cuts_generated,
            domains_verified: self.domains_verified,
        }
    }
}
