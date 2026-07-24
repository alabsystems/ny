// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Sequential multi-objective domain processing helpers.
//!
//! Extracted from `verify.rs` to keep the top-level verification loop focused
//! on queue management while preserving the same child-domain semantics.

use std::collections::BinaryHeap;
use std::time::Instant;

use ny_core::{GemmEngine, Result};
use tracing::debug;

use super::queue::{consult_leaf_oracle, LeafOracleCtx};
use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::{GraphNeuronConstraint, GraphSplitHistory};
use crate::beta_crown::domain::{
    GraphCrownContext, MultiObjectiveGraphBabDomain, MultiObjectiveTargets,
};
use crate::GraphNetwork;

use super::super::super::BetaCrownVerifier;
use super::per_disjunct_eval::{PerDisjunctChildOutcome, PerDisjunctContext};
use super::shared::{
    merge_pruned_cached_las, merge_pruned_objective_bounds, prune_cached_las_for_targets,
    prune_verified_multi_objective_targets,
};

/// Mutable and immutable inputs for sequential multi-objective domain processing.
pub(super) struct SequentialMultiObjectiveContext<'a> {
    pub(super) graph: &'a GraphNetwork,
    pub(super) domains_to_process: Vec<MultiObjectiveGraphBabDomain>,
    pub(super) relu_nodes: &'a [String],
    pub(super) objectives: &'a [Vec<f32>],
    pub(super) thresholds: &'a [f32],
    pub(super) engine: Option<&'a dyn GemmEngine>,
    pub(super) cut_pool: &'a mut GraphCutPool,
    pub(super) queue: &'a mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    pub(super) domains_verified: &'a mut usize,
    pub(super) unresolved_due_to_no_branch: &'a mut bool,
    pub(super) unresolved_due_to_violated_drop: &'a mut bool,
    pub(super) unresolved_due_to_propagation_failure: &'a mut bool,
    /// Whether this is a conjunctive (AND) property. When true, a domain is
    /// verified if ANY objective is verified, and dropped only if ALL violated.
    pub(super) conjunctive: bool,
    /// Deadline for timeout enforcement (#3388). When set, sequential processing
    /// checks this before each domain to bail early when the verification timeout
    /// budget is exhausted. Without this, a single batch of CROWN passes across
    /// many objectives can exceed the timeout without the outer BaB loop noticing.
    pub(super) deadline: Option<Instant>,
}

/// Check if a domain is verified according to the aggregation mode.
/// Disjunctive: ALL objectives verified. Conjunctive: ANY objective verified.
fn is_domain_verified(domain: &MultiObjectiveGraphBabDomain, conjunctive: bool) -> bool {
    if conjunctive {
        domain.any_verified()
    } else {
        domain.all_verified()
    }
}

/// Check if a domain should be dropped according to the aggregation mode.
/// Disjunctive: ANY objective violated. Conjunctive: ALL objectives violated.
fn is_domain_dropped(
    domain: &MultiObjectiveGraphBabDomain,
    thresholds: &[f32],
    conjunctive: bool,
) -> bool {
    if conjunctive {
        domain.all_violated(thresholds, false)
    } else {
        domain.any_violated(thresholds, false)
    }
}

/// Outcome of processing a single BaB child domain.
///
/// Returned by [`BetaCrownVerifier::process_multi_objective_child`] so the
/// caller can update counters, push to the queue, and decide control flow.
enum ChildOutcome {
    /// Child was verified (contributes to `domains_verified` count).
    /// Carries the split history for cut generation when cuts are enabled.
    /// Boxed to avoid inflating the enum size (GraphSplitHistory is large).
    Verified(Option<Box<GraphSplitHistory>>),
    /// Child was dropped due to conclusive violation.
    Dropped,
    /// Child is unresolved and ready to enqueue (caller must push to queue).
    Enqueued,
    /// Propagation failure — sub-region unexplored, continue with other children.
    PropagationFailure,
    /// NaN corruption in child bounds — skip remaining children.
    /// Matches the original `continue` semantics that skip the inactive child
    /// when the active child hits an internal NaN failure.
    NaNCorruption,
}

/// Shared immutable state for child domain processing.
///
/// Groups the parameters that are identical for both active and inactive
/// children, reducing the argument count of `process_multi_objective_child`.
struct ChildProcessingContext<'a> {
    graph: &'a GraphNetwork,
    objectives: &'a [Vec<f32>],
    thresholds: &'a [f32],
    engine: Option<&'a dyn GemmEngine>,
    cut_pool: &'a GraphCutPool,
    conjunctive: bool,
}

impl BetaCrownVerifier {
    /// Process a batch of domains with the sequential (non-batched) multi-objective path.
    pub(super) fn process_multi_objective_domains_sequential(
        &self,
        context: SequentialMultiObjectiveContext<'_>,
    ) -> Result<()> {
        let SequentialMultiObjectiveContext {
            graph,
            domains_to_process,
            relu_nodes,
            objectives,
            thresholds,
            engine,
            cut_pool,
            queue,
            domains_verified,
            unresolved_due_to_no_branch,
            unresolved_due_to_violated_drop,
            unresolved_due_to_propagation_failure,
            conjunctive,
            deadline,
        } = context;

        'domain: for mut domain in domains_to_process {
            // Deadline check: bail early if verification timeout exceeded (#3388).
            // The outer BaB loop checks timeout only between batches. For large
            // networks with many objectives, a single batch of sequential CROWN
            // passes can exceed the timeout budget. This per-domain check ensures
            // we return to the BaB loop promptly for the timeout to fire.
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(());
            }

            // Find unstable neurons to branch on
            let unstable = self.find_unstable_graph_neurons_multi(graph, &domain, relu_nodes);
            if unstable.is_empty() {
                // No unstable neurons left - recompute bounds one more time
                let cut_pool_ref = if self.config.enable_cuts && !cut_pool.is_empty() {
                    Some(&*cut_pool)
                } else {
                    None
                };
                let context = GraphCrownContext::new(
                    &domain.history,
                    cut_pool_ref,
                    Some(&domain.node_bounds),
                    engine,
                )
                .with_alpha(&domain.alpha_state);
                match self.propagate_crown_with_graph_constraints(
                    graph,
                    domain.input_bounds.as_ref(),
                    &context,
                    None,
                    None,
                ) {
                    Ok((output, _node_cache)) => {
                        match Self::objective_bounds_multi(&output, objectives) {
                            Ok(new_bounds) => {
                                if domain.update_bounds(new_bounds, thresholds, false).is_err() {
                                    // NaN in objective bounds → treat as propagation failure (#2982)
                                    *unresolved_due_to_propagation_failure = true;
                                    continue;
                                }
                                if is_domain_verified(&domain, conjunctive) {
                                    *domains_verified += 1;
                                    if self.config.enable_cuts
                                        && cut_pool.add_from_verified_domain(&domain.history)?
                                    {
                                        let merged_len = cut_pool.merge_cuts();
                                        debug!(
                                            "Merged verified-domain graph cuts (pool_len={})",
                                            merged_len
                                        );
                                    }
                                } else if is_domain_dropped(&domain, thresholds, conjunctive) {
                                    // #1866: Fully-constrained domain with conclusive violation.
                                    *unresolved_due_to_violated_drop = true;
                                } else {
                                    // #1866: No unstable neurons and not verified — unresolved.
                                    *unresolved_due_to_no_branch = true;
                                }
                            }
                            Err(e) => {
                                // #1871: objective bound extraction failed on a fully-constrained
                                // domain, so this sub-region remains unresolved.
                                tracing::warn!(
                                    "Objective bound extraction failed (NoUnstable): {e}"
                                );
                                *unresolved_due_to_propagation_failure = true;
                            }
                        }
                    }
                    Err(ref e) if e.is_infeasible_domain() => {
                        // #2926: Infeasible domain = empty = trivially verified.
                        tracing::debug!("NoUnstable domain infeasible (empty), pruning: {e}");
                        *domains_verified += 1;
                    }
                    Err(e) => {
                        // #1871: final constrained propagation failed for a fully-constrained
                        // domain, so this sub-region remains unresolved.
                        tracing::warn!("NoUnstable CROWN propagation failed: {e}");
                        *unresolved_due_to_propagation_failure = true;
                    }
                }
                continue;
            }

            // Select neuron to split
            let (node_name, neuron_idx, score) = match self
                .select_graph_branch_multi(graph, &domain, &unstable, objectives, engine)
            {
                Ok(v) => v,
                Err(e) => {
                    // #1915/#1871: branch selection failed — domain unexplored, must not claim Verified.
                    tracing::warn!("select_graph_branch_multi failed: {e} (#1915)");
                    *unresolved_due_to_propagation_failure = true;
                    continue;
                }
            };

            // Collect histories of verified children for cut generation
            let mut verified_histories: Vec<GraphSplitHistory> = Vec::new();

            // Process active (x >= 0) and inactive (x < 0) children.
            // Both branches share identical logic — only the constraint differs.
            // Pattern: domain_process.rs uses the same `for is_active in [true, false]` loop.
            for is_active in [true, false] {
                let constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx,
                    is_active,
                    score,
                };
                match domain.with_constraint(graph, constraint, false, thresholds) {
                    Ok(Some(mut child)) => {
                        let child_ctx = ChildProcessingContext {
                            graph,
                            objectives,
                            thresholds,
                            engine,
                            cut_pool: &*cut_pool,
                            conjunctive,
                        };
                        let outcome = self.process_multi_objective_child(
                            &child_ctx, &domain, &mut child, is_active,
                        )?;
                        match outcome {
                            ChildOutcome::Verified(history) => {
                                *domains_verified += 1;
                                if let Some(h) = history {
                                    verified_histories.push(*h);
                                }
                            }
                            ChildOutcome::Dropped => {
                                *unresolved_due_to_violated_drop = true;
                            }
                            ChildOutcome::PropagationFailure => {
                                *unresolved_due_to_propagation_failure = true;
                            }
                            ChildOutcome::NaNCorruption => {
                                *unresolved_due_to_propagation_failure = true;
                                continue 'domain;
                            }
                            ChildOutcome::Enqueued => {
                                // Graph-MIP LEAF escalation (increment 6): same
                                // hook as the batched lane's requeue — an
                                // attached oracle may decide the subdomain
                                // exactly before it re-enters the heap;
                                // `Undecided` (and no-oracle) pushes unchanged.
                                let leaf_ctx =
                                    self.graph_mip_leaf_oracle().map(|oracle| LeafOracleCtx {
                                        oracle,
                                        graph,
                                        objectives,
                                        thresholds,
                                        deadline,
                                    });
                                let verdict = consult_leaf_oracle(leaf_ctx.as_ref(), &child);
                                match &verdict {
                                    crate::beta_crown::graph_mip_leaf::GraphMipLeafVerdict::VerifiedAllRows => {
                                        *domains_verified += 1;
                                        verified_histories.push(child.history.clone());
                                    }
                                    // ADVISORY (see queue.rs `apply_leaf_verdict`):
                                    // log-and-requeue — a leaf verdict must never
                                    // drop a child / end the run early.
                                    crate::beta_crown::graph_mip_leaf::GraphMipLeafVerdict::Violated { .. } => {
                                        tracing::warn!(
                                            "Graph-MIP leaf oracle: confirmed SAT witness \
                                             (advisory — child requeued)"
                                        );
                                        queue.push(child);
                                    }
                                    crate::beta_crown::graph_mip_leaf::GraphMipLeafVerdict::Undecided => {
                                        queue.push(child);
                                    }
                                }
                            }
                        }
                    }
                    Ok(None) => {}
                    Err(ref e) if e.is_infeasible_domain() => {
                        let label = if is_active { "active" } else { "inactive" };
                        tracing::debug!("with_constraint ({label}) infeasible: {e}");
                        *domains_verified += 1;
                    }
                    Err(e) => {
                        let label = if is_active { "active" } else { "inactive" };
                        tracing::warn!("with_constraint ({label}) failed: {e}");
                        *unresolved_due_to_propagation_failure = true;
                    }
                }
            }

            // Add cuts from verified children
            for history in verified_histories {
                if cut_pool.add_from_verified_domain(&history)? {
                    let merged_len = cut_pool.merge_cuts();
                    debug!("Merged verified-child graph cuts (pool_len={})", merged_len);
                }
            }
        }

        Ok(())
    }

    /// Process a single child domain (active or inactive branch).
    ///
    /// Called twice per parent domain: once for `is_active: true` (x >= 0) and
    /// once for `is_active: false` (x < 0). The logic is identical for both
    /// branches — only the constraint differs.
    ///
    /// The caller is responsible for:
    /// - creating the child via `with_constraint`
    /// - pushing `Enqueued` children into the queue
    /// - tracking verification counts and cut histories
    fn process_multi_objective_child(
        &self,
        ctx: &ChildProcessingContext<'_>,
        parent: &MultiObjectiveGraphBabDomain,
        child: &mut MultiObjectiveGraphBabDomain,
        is_active: bool,
    ) -> Result<ChildOutcome> {
        // Per-disjunct alpha (#4355): when the child carries per-disjunct alpha
        // states, dispatch to the per-disjunct evaluation module which runs
        // separate 1-row CROWN passes per unverified objective.
        if child.per_disjunct_alphas().is_some() {
            let pd_ctx = PerDisjunctContext {
                graph: ctx.graph,
                objectives: ctx.objectives,
                thresholds: ctx.thresholds,
                engine: ctx.engine,
                cut_pool: ctx.cut_pool,
                conjunctive: ctx.conjunctive,
            };
            return self
                .process_multi_objective_child_per_disjunct(&pd_ctx, parent, child, is_active)
                .map(|outcome| match outcome {
                    PerDisjunctChildOutcome::Verified(h) => ChildOutcome::Verified(h),
                    PerDisjunctChildOutcome::Dropped => ChildOutcome::Dropped,
                    PerDisjunctChildOutcome::Enqueued => ChildOutcome::Enqueued,
                    PerDisjunctChildOutcome::PropagationFailure => ChildOutcome::PropagationFailure,
                    PerDisjunctChildOutcome::NaNCorruption => ChildOutcome::NaNCorruption,
                });
        }

        let cut_pool_ref = if self.config.enable_cuts && !ctx.cut_pool.is_empty() {
            Some(ctx.cut_pool)
        } else {
            None
        };
        // #cone-delta: the child inherited `node_bounds` verbatim from
        // `parent`, so its delta describes exactly the base map passed here
        // (dark, NY_CONE_REFRESH-gated).
        let crown_ctx = GraphCrownContext::new(
            &child.history,
            cut_pool_ref,
            Some(&parent.node_bounds),
            ctx.engine,
        )
        .with_alpha(&child.alpha_state)
        .with_delta_seeds(&child.delta_pre_nodes);
        let pruned_targets =
            prune_verified_multi_objective_targets(ctx.objectives, ctx.thresholds, &child.verified);
        let targets = MultiObjectiveTargets::new(
            &pruned_targets.objectives,
            &pruned_targets.thresholds,
            &pruned_targets.verified_mask,
        );
        // Borrow the child's cached lA slice directly (disjoint from the
        // `&mut child.beta_state` taken below) instead of deep-cloning it via
        // `.to_vec()`. `prune_cached_las_for_targets` and `merge_pruned_cached_las`
        // both take `&[Option<CachedLinearBounds>]` by shared reference and never
        // need ownership, so this reads bit-identical data with no clone.
        let pruned_cached_las = prune_cached_las_for_targets(&child.cached_las, &pruned_targets);
        // Only run β optimization when enabled and for shallow domains
        // When beta_iterations=0, skip optimization entirely for all domains
        let should_optimize =
            self.config.beta_iterations > 0 && child.depth <= self.config.beta_max_depth;
        let result = if should_optimize {
            self.optimize_graph_beta_analytical_multi_objective_with_cache(
                ctx.graph,
                child.input_bounds.as_ref(),
                &crown_ctx,
                &mut child.beta_state,
                &targets,
                ctx.conjunctive,
                &pruned_cached_las,
                true,
            )
        } else {
            // Skip optimization, just propagate with inherited β
            self.propagate_multi_objective_with_beta_and_cache(
                ctx.graph,
                child.input_bounds.as_ref(),
                &crown_ctx,
                &child.beta_state,
                &targets,
                &pruned_cached_las,
                true,
            )
        };
        let label = if is_active { "Active" } else { "Inactive" };
        match result {
            Ok((child_bounds, node_cache, child_cached_las)) => {
                let new_bounds = merge_pruned_objective_bounds(
                    &child.objective_bounds,
                    &pruned_targets,
                    child_bounds,
                );
                // #cone-delta increment 2: already Arc-shared — move.
                child.node_bounds = node_cache;
                // #cone-delta: post-bounding replacement — delta restarts empty.
                child.delta_pre_nodes.clear();
                if child
                    .update_bounds(new_bounds, ctx.thresholds, false)
                    .is_err()
                {
                    // NaN in objective bounds → treat as NaN corruption (#2982)
                    return Ok(ChildOutcome::NaNCorruption);
                }
                if is_domain_dropped(child, ctx.thresholds, ctx.conjunctive) {
                    Ok(ChildOutcome::Dropped)
                } else if is_domain_verified(child, ctx.conjunctive) {
                    let history = if self.config.enable_cuts {
                        Some(Box::new(child.history.clone()))
                    } else {
                        None
                    };
                    Ok(ChildOutcome::Verified(history))
                } else {
                    let merged_cached_las = merge_pruned_cached_las(
                        &child.cached_las,
                        &pruned_targets,
                        child_cached_las,
                    );
                    if child.set_cached_las(merged_cached_las).is_err() {
                        return Ok(ChildOutcome::NaNCorruption);
                    }
                    Ok(ChildOutcome::Enqueued)
                }
            }
            Err(ref e) if e.is_infeasible_domain() => {
                // #2926: Infeasible domain = empty = trivially verified.
                tracing::debug!("{label} child infeasible (empty): {e}");
                Ok(ChildOutcome::Verified(None))
            }
            Err(e) => {
                // #1861: propagation failed — sub-region unexplored.
                tracing::warn!("{label} child propagation failed: {e}");
                Ok(ChildOutcome::PropagationFailure)
            }
        }
    }
}
