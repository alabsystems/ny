// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Root-status and termination helpers for ReLU-split branch-and-bound.

use ndarray::Array1;
use ny_tensor::BoundedTensor;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::GraphNetwork;

use super::super::super::domain_results::GraphDomainResult;
use super::super::super::BetaCrownVerifier;

impl BetaCrownVerifier {
    /// Check if the root domain can be resolved without BaB.
    pub(super) fn check_root_early_exit(
        &self,
        root_lower: f32,
        root_upper: f32,
        threshold: f32,
        lifecycle: &mut GraphBabLifecycle,
    ) -> ny_core::Result<Option<BetaCrownResult>> {
        let root_status = if self
            .config
            .domain_is_verified(root_lower, root_upper, threshold)
        {
            Some((BabVerificationStatus::Verified, 1))
        } else if self
            .config
            .domain_is_violation(root_lower, root_upper, threshold)
        {
            Some((BabVerificationStatus::PotentialViolation, 0))
        } else {
            None
        };
        if let Some((status, root_verified_count)) = root_status {
            lifecycle.domains_explored = 1;
            lifecycle.domains_verified = root_verified_count;
            return Ok(Some(lifecycle.build_result_with_bounds(
                status,
                BoundedTensor::new(
                    Array1::from_vec(vec![root_lower]).into_dyn(),
                    Array1::from_vec(vec![root_upper]).into_dyn(),
                )?,
            )));
        }
        Ok(None)
    }

    /// Check timeout and domain limit termination conditions.
    ///
    /// `bab_timeout` is the budget the BaB loop derived at entry: wall-clock
    /// deadline aware (#4321) and minus the post-BaB PGD reservation (#4095), so
    /// the loop stops early enough for post-BaB PGD to get its reserved time.
    /// Recomputing it here from `config.timeout` (as this used to) re-granted BaB
    /// a fresh full budget measured from loop entry, ignoring wall-clock time
    /// already consumed by pre-BaB phases — the loop could then overrun an
    /// expired deadline by most of `config.timeout`.
    pub(super) fn check_termination(
        &self,
        lifecycle: &mut GraphBabLifecycle,
        cut_pool: &GraphCutPool,
        bab_timeout: std::time::Duration,
    ) -> Option<BetaCrownResult> {
        lifecycle.cuts_generated = cut_pool.total_generated;
        lifecycle.check_termination(bab_timeout, self.config.max_domains)
    }

    pub(super) fn select_graph_branch_or_propagation_failure_in_relu_split(
        &self,
        graph: &GraphNetwork,
        domain: &GraphBabDomain,
        unstable: &[(String, usize)],
    ) -> Result<(String, usize, f32), GraphDomainResult> {
        match self.select_graph_branch(graph, domain, unstable) {
            Ok(selection) => Ok(selection),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    depth = domain.depth,
                    "select_graph_branch failed in ReLU split loop; marking domain as PropagationFailure (#2038, #1915)"
                );
                Err(GraphDomainResult::PropagationFailure)
            }
        }
    }
}
