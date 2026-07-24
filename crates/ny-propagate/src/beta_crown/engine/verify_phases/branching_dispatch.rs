// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Branching dispatch for the BaB domain-processing loop.
//!
//! Owns the parallel-vs-sequential decision and bound-gain aggregation
//! that sits between domain prefiltering and child settlement.

use std::time::Instant;

use ny_core::GemmEngine;
use ny_tensor::BoundedTensor;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::domain::{BabDomain, DomainProcessingConfig};
use crate::faer_parallelism::RayonTaskGuard;
use crate::Network;

use super::super::BetaCrownVerifier;

pub(in crate::beta_crown::engine) struct BranchBatchOutcome {
    pub child_results: Vec<BabDomain>,
    pub bound_gain_sum: f32,
    pub bound_gain_count: usize,
    pub had_propagation_failure: bool,
    pub had_no_branch: bool,
    pub had_unsplittable: bool,
}

impl BetaCrownVerifier {
    // This contingency extraction mirrors the loop's existing dispatch block and
    // keeps the orchestrator within the issue's line budget without new behavior.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine) fn process_branching_batch(
        &self,
        network: &Network,
        input: &BoundedTensor,
        domains_to_process: &[BabDomain],
        threshold: f32,
        batch_size: usize,
        cut_pool: &mut CutPool,
        engine: Option<&dyn GemmEngine>,
        deadline: Option<Instant>,
    ) -> BranchBatchOutcome {
        let has_cuts = !cut_pool.is_empty() && self.config.enable_cuts;
        let domain_config = DomainProcessingConfig::for_deadline(
            threshold,
            self.config.parallel_children,
            deadline,
        );

        if batch_size > 1 && !has_cuts {
            let (
                child_results,
                bound_gain_sum,
                bound_gain_count,
                had_propagation_failure,
                had_no_branch,
                had_unsplittable,
            ) = domains_to_process
                .par_iter()
                .map(|domain| {
                    let _rayon_task_guard = RayonTaskGuard::new();
                    let mut empty_pool = CutPool::new(0);
                    let result = self.process_domain_parallel(
                        network,
                        input,
                        domain,
                        &domain_config,
                        &mut empty_pool,
                        engine,
                    );
                    let (mut gain_sum, mut gain_count) = (0.0f32, 0usize);
                    for child in &result.children {
                        let gain = self.bound_gain(domain, child);
                        if gain > 0.0 {
                            gain_sum += gain;
                            gain_count += 1;
                        }
                    }
                    (
                        result.children,
                        gain_sum,
                        gain_count,
                        result.had_propagation_failure,
                        result.had_no_branch,
                        result.had_unsplittable,
                    )
                })
                .reduce(
                    || (Vec::new(), 0.0, 0usize, false, false, false),
                    |mut acc, item| {
                        acc.0.extend(item.0);
                        acc.1 += item.1;
                        acc.2 += item.2;
                        acc.3 |= item.3;
                        acc.4 |= item.4;
                        acc.5 |= item.5;
                        acc
                    },
                );
            return BranchBatchOutcome {
                child_results,
                bound_gain_sum,
                bound_gain_count,
                had_propagation_failure,
                had_no_branch,
                had_unsplittable,
            };
        }

        let mut outcome = BranchBatchOutcome {
            child_results: Vec::new(),
            bound_gain_sum: 0.0,
            bound_gain_count: 0,
            had_propagation_failure: false,
            had_no_branch: false,
            had_unsplittable: false,
        };
        for domain in domains_to_process {
            let result = self.process_domain_sequential(
                network, input, domain, threshold, cut_pool, engine, deadline,
            );
            outcome.had_propagation_failure |= result.had_propagation_failure;
            outcome.had_no_branch |= result.had_no_branch;
            outcome.had_unsplittable |= result.had_unsplittable;
            for child in result.children {
                let gain = self.bound_gain(domain, &child);
                if gain > 0.0 {
                    outcome.bound_gain_sum += gain;
                    outcome.bound_gain_count += 1;
                }
                outcome.child_results.push(child);
            }
        }
        outcome
    }
}
