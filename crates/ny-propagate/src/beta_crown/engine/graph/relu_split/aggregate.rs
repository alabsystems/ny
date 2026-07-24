// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Result aggregation helpers for ReLU-split branch-and-bound.

use std::collections::BinaryHeap;

use ny_core::Result;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};

use super::super::super::domain_results::GraphDomainResult;
use super::super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Aggregate BaB results: track verified/violations, enqueue children.
    ///
    /// Returns `Some(result)` if a violation was found and we should exit.
    // Justification: Aggregation needs results, threshold, queue + priority fn,
    // lifecycle and cut pool - the full BaB iteration context.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn aggregate_bab_results(
        &self,
        results: Vec<GraphDomainResult>,
        threshold: f32,
        queue_priority: &impl Fn(f32, f32) -> Result<f32>,
        queue: &mut BinaryHeap<GraphBabDomain>,
        lifecycle: &mut GraphBabLifecycle,
        cut_pool: &mut GraphCutPool,
    ) -> Result<Option<BetaCrownResult>> {
        for result in results {
            match result {
                GraphDomainResult::AlreadyVerified => {
                    lifecycle.domains_verified += 1;
                }
                GraphDomainResult::Violation => {
                    lifecycle.cuts_generated = cut_pool.total_generated;
                    return Ok(Some(
                        lifecycle.build_result(BabVerificationStatus::PotentialViolation),
                    ));
                }
                GraphDomainResult::Children(children) => {
                    for (mut child, verified) in children {
                        if verified {
                            lifecycle.domains_verified += 1;
                            self.try_generate_verified_cut(
                                child.depth,
                                &child.history,
                                cut_pool,
                                "verified child",
                            )?;
                        } else {
                            child.priority = queue_priority(child.lower_bound, child.upper_bound)?;
                            queue.push(child);
                        }
                    }
                }
                GraphDomainResult::NoUnstable {
                    lower,
                    upper,
                    verified,
                } => {
                    if verified {
                        lifecycle.domains_verified += 1;
                    } else {
                        if self.config.domain_is_violation(lower, upper, threshold) {
                            lifecycle.cuts_generated = cut_pool.total_generated;
                            return Ok(Some(
                                lifecycle.build_result(BabVerificationStatus::PotentialViolation),
                            ));
                        }
                        lifecycle.unresolved_due_to_no_branch = true;
                    }
                }
                GraphDomainResult::PropagationFailure => {
                    // #1861: child propagation failed, input sub-region unexplored
                    lifecycle.unresolved_due_to_propagation_failure = true;
                }
            }
        }
        Ok(None)
    }
}
