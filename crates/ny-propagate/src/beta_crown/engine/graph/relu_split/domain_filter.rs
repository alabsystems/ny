// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Domain pre-filtering and cut-generation helpers for ReLU-split BaB.

use ny_core::Result;
use tracing::debug;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;

use super::super::super::BetaCrownVerifier;

/// Result of pre-filtering a batch of domains.
pub(super) enum PreFilterOutcome {
    /// Continue processing these domains.
    Process(Vec<GraphBabDomain>),
    /// A violation was found; return immediately.
    Violation,
}

impl BetaCrownVerifier {
    /// Try to generate a cut from a verified domain or child.
    pub(super) fn try_generate_verified_cut(
        &self,
        domain_depth: usize,
        history: &GraphSplitHistory,
        cut_pool: &mut GraphCutPool,
        context: &str,
    ) -> Result<()> {
        if self.config.enable_cuts
            && domain_depth >= self.config.min_cut_depth
            && cut_pool.add_from_verified_domain(history)?
        {
            debug!(
                "Generated cut from {context} (depth={domain_depth}, total cuts={})",
                cut_pool.len()
            );
            let merged_len = cut_pool.merge_cuts();
            debug!("Merged {context} graph cuts (pool_len={merged_len})");
        }
        Ok(())
    }

    /// Pre-filter a batch of domains: separate verified, detect violations, apply depth limits.
    pub(super) fn pre_filter_batch(
        &self,
        batch: Vec<GraphBabDomain>,
        threshold: f32,
        lifecycle: &mut GraphBabLifecycle,
        cut_pool: &mut GraphCutPool,
    ) -> Result<PreFilterOutcome> {
        let mut domains_to_process: Vec<GraphBabDomain> = Vec::with_capacity(batch.len());

        for domain in batch {
            lifecycle.domains_explored += 1;
            lifecycle.max_depth_reached = lifecycle.max_depth_reached.max(domain.depth);

            // Guard: drop NaN/Inf-bounded domains (same as GPU prefilter.rs:46-56, #2953).
            // Without this, NaN domains fall through domain_is_verified/domain_is_violation
            // to Undecided, enter splitting, and produce NaN children indefinitely.
            if !domain.lower_bound.is_finite() || !domain.upper_bound.is_finite() {
                tracing::warn!(
                    depth = domain.depth,
                    lower = domain.lower_bound,
                    upper = domain.upper_bound,
                    "relu_split pre_filter_batch: domain dropped — non-finite bounds"
                );
                lifecycle.unresolved_due_to_propagation_failure = true;
                continue;
            }

            if self
                .config
                .domain_is_verified(domain.lower_bound, domain.upper_bound, threshold)
            {
                lifecycle.domains_verified += 1;
                self.try_generate_verified_cut(
                    domain.depth,
                    &domain.history,
                    cut_pool,
                    "verified domain",
                )?;
                continue;
            }

            if self
                .config
                .domain_is_violation(domain.lower_bound, domain.upper_bound, threshold)
            {
                return Ok(PreFilterOutcome::Violation);
            }

            if self.config.enable_cuts
                && self.config.enable_near_miss_cuts
                && domain.depth >= self.config.min_cut_depth
            {
                let bound_for_check = self
                    .config
                    .relevant_bound(domain.lower_bound, domain.upper_bound);
                if cut_pool.add_from_near_miss_domain(
                    &domain.history,
                    bound_for_check,
                    threshold,
                    self.config.near_miss_margin,
                )? {
                    debug!(
                        "Generated near-miss cut (depth={}, lb={:.4}, threshold={:.4}, total cuts={})",
                        domain.depth,
                        bound_for_check,
                        threshold,
                        cut_pool.len()
                    );
                    let merged_len = cut_pool.merge_cuts();
                    debug!("Merged near-miss graph cuts (pool_len={merged_len})");
                }
            }

            if domain.depth >= self.config.max_depth {
                lifecycle.unresolved_due_to_depth = true;
                continue;
            }

            domains_to_process.push(domain);
        }

        Ok(PreFilterOutcome::Process(domains_to_process))
    }
}
