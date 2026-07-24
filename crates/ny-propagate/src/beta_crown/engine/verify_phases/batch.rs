// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Instant;

use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, trace, warn};

use crate::beta_crown::bab_cuts::CutPool;
use crate::beta_crown::branching::BranchingHeuristic;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::domain::BabDomain;
use crate::beta_crown::result::{BabVerificationStatus, BetaCrownResult};
use crate::Network;

use super::super::cut_gate::{apply_event, CutGateBatchStats, CutGateState};
use super::super::tensor_ext::BoundedTensorExt;
use super::super::BetaCrownVerifier;
use super::state::BabLoopState;

pub(in crate::beta_crown::engine) struct PrefilterOutcome {
    pub domains_to_process: Vec<BabDomain>,
    pub verified_in_batch: usize,
    pub batch_domain_count: usize,
    pub violation: Option<BetaCrownResult>,
}

pub(in crate::beta_crown::engine) fn pop_domain_batch(
    queue: &mut BinaryHeap<BabDomain>,
    batch_size: usize,
) -> Vec<BabDomain> {
    let mut batch = Vec::with_capacity(batch_size);
    while batch.len() < batch_size {
        if let Some(domain) = queue.pop() {
            batch.push(domain);
        } else {
            break;
        }
    }
    batch
}

#[allow(clippy::too_many_arguments)]
pub(in crate::beta_crown::engine) fn record_cut_gate_batch(
    config: &BetaCrownConfig,
    cut_gate: &mut CutGateState,
    state: &mut BabLoopState,
    cut_pool: &CutPool,
    total_domains: usize,
    verified_domains: usize,
    bound_gain_avg: Option<f32>,
    cuts_active_for_batch: bool,
) {
    let batch_stats = CutGateBatchStats {
        total_domains,
        verified_domains,
        bound_gain_avg,
        cut_pruned_domains: if cuts_active_for_batch {
            verified_domains
        } else {
            0
        },
        cut_total_domains: if cuts_active_for_batch {
            total_domains
        } else {
            0
        },
    };
    if let Some(event) = cut_gate.record_batch(config, batch_stats, cut_pool.total_generated) {
        apply_event(
            &event,
            &mut state.cut_generation_enabled,
            state.domains_verified,
            cut_pool.total_generated,
        );
    }
}

impl BetaCrownVerifier {
    pub(in crate::beta_crown::engine) fn build_root_domain(
        &self,
        initial_layer_bounds: Vec<BoundedTensor>,
        initial_bounds: &BoundedTensor,
        input: &BoundedTensor,
    ) -> Result<BabDomain> {
        let initial_lower = initial_bounds.lower_scalar();
        let initial_upper = initial_bounds.upper_scalar();
        if matches!(
            self.config.branching_heuristic,
            BranchingHeuristic::InputSplit
        ) {
            BabDomain::root_with_input(initial_layer_bounds, initial_lower, initial_upper, input)
        } else {
            BabDomain::root(initial_layer_bounds, initial_lower, initial_upper)
        }
    }

    // These loop helpers consume independent verification resources and state;
    // bundling them into a context struct would hide the phase contract.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine) fn prefilter_domain_batch(
        &self,
        batch: Vec<BabDomain>,
        threshold: f32,
        state: &mut BabLoopState,
        cut_pool: &mut CutPool,
        network: &Network,
        input: &BoundedTensor,
        base_layer_bounds: &[Arc<BoundedTensor>],
        engine: Option<&dyn GemmEngine>,
        start_time: Instant,
    ) -> Result<PrefilterOutcome> {
        let mut verified_in_batch = 0usize;
        let mut domains_to_process = Vec::new();

        for domain in batch {
            state.domains_explored += 1;
            state.max_depth = state.max_depth.max(domain.depth());

            trace!(
                "Processing domain {}: depth={}, lb={:.4}, ub={:.4}",
                state.domains_explored,
                domain.depth(),
                domain.lower_bound,
                domain.upper_bound
            );

            if !domain.lower_bound.is_finite() || !domain.upper_bound.is_finite() {
                warn!(
                    "Sequential BaB prefilter: domain dropped — non-finite bounds \
                     (depth={}, lb={}, ub={})",
                    domain.depth(),
                    domain.lower_bound,
                    domain.upper_bound
                );
                state.unresolved_due_to_propagation_failure = true;
                continue;
            }

            // Conflict-clause prune (NY_BAB_CLAUSE_LEARN=1, default off): if a
            // recorded clause is a subset of this domain's literal set, its
            // region is a subregion of an already-certified one (same run,
            // same root box by construction) — close it as verified WITHOUT
            // the split/bound work. Fails closed for input-split domains.
            // Deliberately not fed to cut generation: its history is a
            // superset of a stored clause whose domain the cut machinery
            // already saw.
            if state.clause_store.should_prune(&domain) {
                trace!(
                    "Domain clause-pruned: depth={}, subsumed by recorded conflict clause",
                    domain.depth()
                );
                state.domains_verified += 1;
                state.domains_clause_pruned += 1;
                verified_in_batch += 1;
                continue;
            }

            if self
                .config
                .domain_is_verified(domain.lower_bound, domain.upper_bound, threshold)
            {
                trace!(
                    "Domain verified: lb={}, ub={}, threshold={}",
                    domain.lower_bound,
                    domain.upper_bound,
                    threshold
                );
                state.domains_verified += 1;
                verified_in_batch += 1;
                state.clause_store.record_verified_domain(&domain);

                if state.cut_generation_enabled
                    && self.config.enable_cuts
                    && domain.depth() >= self.config.min_cut_depth
                {
                    let mut cut_added = self.try_add_strengthened_cut(
                        cut_pool,
                        network,
                        input,
                        threshold,
                        base_layer_bounds,
                        &domain,
                        engine,
                    )?;
                    if !cut_added && cut_pool.add_from_verified_domain(&domain.history)? {
                        trace!(
                            "Generated cut from verified domain (depth={}, total cuts={})",
                            domain.depth(),
                            cut_pool.len()
                        );
                        cut_added = true;
                    }
                    if cut_added {
                        trace!(
                            "Merged verified-domain cuts (pool_len={})",
                            cut_pool.merge_cuts()
                        );
                    }
                }
                continue;
            }

            if self
                .config
                .domain_is_violation(domain.lower_bound, domain.upper_bound, threshold)
            {
                debug!(
                    "Sequential BaB prefilter: violation found (depth={}, lb={}, ub={}, threshold={})",
                    domain.depth(),
                    domain.lower_bound,
                    domain.upper_bound,
                    threshold
                );
                return Ok(PrefilterOutcome {
                    domains_to_process,
                    verified_in_batch,
                    batch_domain_count: 0,
                    violation: Some(BetaCrownResult {
                        result: BabVerificationStatus::PotentialViolation,
                        domains_explored: state.domains_explored,
                        time_elapsed: start_time.elapsed(),
                        max_depth_reached: state.max_depth,
                        output_bounds: None,
                        cuts_generated: cut_pool.total_generated,
                        domains_verified: state.domains_verified,
                    }),
                });
            }

            if domain.depth() >= self.config.max_depth {
                debug!("Domain at max depth {}, skipping", self.config.max_depth);
                state.unresolved_due_to_depth = true;
                continue;
            }

            domains_to_process.push(domain);
        }

        Ok(PrefilterOutcome {
            batch_domain_count: domains_to_process.len() + verified_in_batch,
            domains_to_process,
            verified_in_batch,
            violation: None,
        })
    }

    // Child settlement needs the queue, cut pool, and bound context separately;
    // grouping those mutable resources would make queue ownership less explicit.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::beta_crown::engine) fn process_batch_children(
        &self,
        child_results: Vec<BabDomain>,
        threshold: f32,
        queue: &mut BinaryHeap<BabDomain>,
        state: &mut BabLoopState,
        cut_pool: &mut CutPool,
        network: &Network,
        input: &BoundedTensor,
        base_layer_bounds: &[Arc<BoundedTensor>],
        engine: Option<&dyn GemmEngine>,
    ) -> Result<usize> {
        let mut verified_children_in_batch = 0usize;

        for mut child in child_results {
            if !child.lower_bound.is_finite() || !child.upper_bound.is_finite() {
                warn!(
                    "Sequential BaB: child dropped — non-finite bounds \
                     (depth={}, lb={}, ub={})",
                    child.depth(),
                    child.lower_bound,
                    child.upper_bound
                );
                state.unresolved_due_to_propagation_failure = true;
                continue;
            }

            if !self
                .config
                .domain_is_verified(child.lower_bound, child.upper_bound, threshold)
            {
                child.priority = self
                    .config
                    .violation_priority(child.lower_bound, child.upper_bound)?;
                queue.push(child);
                continue;
            }

            trace!("Child verified immediately");
            state.domains_verified += 1;
            verified_children_in_batch += 1;
            state.clause_store.record_verified_domain(&child);

            if state.cut_generation_enabled
                && self.config.enable_cuts
                && child.depth() >= self.config.min_cut_depth
            {
                let mut cut_added = self.try_add_strengthened_cut(
                    cut_pool,
                    network,
                    input,
                    threshold,
                    base_layer_bounds,
                    &child,
                    engine,
                )?;
                if !cut_added && cut_pool.add_from_verified_domain(&child.history)? {
                    trace!(
                        "Generated cut from verified child (depth={}, total cuts={})",
                        child.depth(),
                        cut_pool.len()
                    );
                    cut_added = true;
                }
                if cut_added {
                    trace!(
                        "Merged verified-child cuts (pool_len={})",
                        cut_pool.merge_cuts()
                    );
                }
            }
        }

        Ok(verified_children_in_batch)
    }
}
