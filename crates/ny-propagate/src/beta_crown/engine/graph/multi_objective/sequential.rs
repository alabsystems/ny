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
use crate::beta_crown::conflict_clauses_graph::GraphClauseStore;
use crate::beta_crown::domain::{
    GraphCrownContext, MultiObjectiveGraphBabDomain, MultiObjectiveTargets,
};
use crate::beta_crown::graph_mip_leaf::GraphMipLeafVerdict;
use crate::beta_crown::result::BabVerificationStatus;
use crate::GraphNetwork;

use super::super::super::BetaCrownVerifier;
use super::super::objectives::{
    mo_cuda_beta_spsa_frontier_tracking_enabled, MoCudaBetaSpsaFrontier,
};
use super::per_disjunct_eval::{PerDisjunctChildOutcome, PerDisjunctContext};
use super::shared::{
    merge_pruned_cached_las, merge_pruned_objective_bounds, prune_cached_las_for_targets,
    prune_verified_multi_objective_targets,
};
// #bab-monotone-inherit: shared monotone parent-bound merge + its dark gate,
// re-exported at `multi_objective` module scope.
use super::{bab_monotone_inherit_enabled, inherit_parent_lower_only};

/// Mutable and immutable inputs for sequential multi-objective domain processing.
pub(super) struct SequentialMultiObjectiveContext<'a> {
    pub(super) graph: &'a GraphNetwork,
    pub(super) domains_to_process: Vec<MultiObjectiveGraphBabDomain>,
    pub(super) relu_nodes: &'a [String],
    pub(super) objectives: &'a [Vec<f32>],
    pub(super) thresholds: &'a [f32],
    pub(super) engine: Option<&'a dyn GemmEngine>,
    pub(super) cut_pool: &'a mut GraphCutPool,
    pub(super) clause_store: &'a mut GraphClauseStore,
    pub(super) queue: &'a mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    pub(super) domains_verified: &'a mut usize,
    pub(super) unresolved_due_to_no_branch: &'a mut bool,
    pub(super) unresolved_due_to_violated_drop: &'a mut bool,
    pub(super) unresolved_due_to_propagation_failure: &'a mut bool,
    /// RETURN channel for the Graph-MIP leaf oracle's sat candidate
    /// (#mip-leaf-witness). Set only by [`apply_sequential_leaf_verdict`], and
    /// only for a graph-forward-confirmed in-box witness that violates EVERY
    /// objective row. Purely additive: nothing in this lane reads it, no queue
    /// push or lifecycle counter depends on it, and the child that produced it
    /// is still enqueued. The caller (`verify.rs`) returns it as the run's
    /// verdict, where the trusted ONNX-Runtime gate re-confirms it.
    pub(super) leaf_violation: &'a mut Option<BabVerificationStatus>,
    /// Whether this is a conjunctive (AND) property. When true, a domain is
    /// verified if ANY objective is verified, and dropped only if ALL violated.
    pub(super) conjunctive: bool,
    /// Deadline for timeout enforcement (#3388). When set, sequential processing
    /// checks this before each domain to bail early when the verification timeout
    /// budget is exhausted. Without this, a single batch of CROWN passes across
    /// many objectives can exceed the timeout without the outer BaB loop noticing.
    pub(super) deadline: Option<Instant>,
}

/// Whether the sequential batch covered every domain handed to it.
///
/// A deadline exit must be explicit: returning plain `Ok(())` discards the
/// unprocessed suffix, so an empty queue can no longer distinguish complete
/// coverage from interrupted work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SequentialMultiObjectiveBatchStatus {
    Completed,
    DeadlineExpired,
}

/// Apply the sequential lane's sole terminal disposition for a leaf verdict.
///
/// Keeping this transition in one helper makes the exact-close side effects
/// explicit: a verified child is counted, recorded for clause reuse, and fed
/// to cut generation, while advisory/undecided children return to the queue.
///
/// `Violated` mirrors the batched lane (`queue.rs::apply_leaf_verdict`) exactly:
/// the child is REQUEUED unconditionally — this lane never drains the queue —
/// and the oracle's graph-forward-confirmed witness is published into
/// `leaf_violation` only when it violates EVERY objective row
/// (`queue::witness_violates_every_objective_row`, which carries the full
/// soundness argument). This is the lane that serves CONJUNCTIVE properties
/// (`verify.rs` routes `conjunctive` here — the batched lane requires
/// `!conjunctive`), and there the all-rows test IS the property's violation
/// predicate, so the check is exact rather than merely sufficient. The verdict
/// itself is only a candidate: the CLI's unchanged
/// `vnncomp.rs::gate_sat_with_trusted_oracle` re-confirms it with a real
/// ONNX-Runtime forward and downgrades anything it cannot reproduce to a sound
/// `unknown`.
fn apply_sequential_leaf_verdict(
    verdict: GraphMipLeafVerdict,
    child: MultiObjectiveGraphBabDomain,
    queue: &mut BinaryHeap<MultiObjectiveGraphBabDomain>,
    clause_store: &mut GraphClauseStore,
    domains_verified: &mut usize,
    verified_histories: &mut Vec<GraphSplitHistory>,
    objectives: &[Vec<f32>],
    thresholds: &[f32],
    leaf_violation: &mut Option<BabVerificationStatus>,
) {
    match verdict {
        GraphMipLeafVerdict::VerifiedAllRows => {
            *domains_verified += 1;
            // Exact all-row closure has the same region-wide authority as a
            // bound close. Record it exactly once here; `verified_histories`
            // is only the independent cut-generation feed.
            clause_store.record_verified_close(&child.history);
            verified_histories.push(child.history.clone());
        }
        GraphMipLeafVerdict::Violated { witness, output } => {
            let covers_property = super::queue::leaf_sat_return_enabled()
                && super::queue::witness_violates_every_objective_row(
                    objectives, thresholds, &output,
                );
            // NEVER DRAIN: requeue first, on both paths.
            queue.push(child);
            if covers_property && leaf_violation.is_none() {
                tracing::warn!(
                    witness_len = witness.len(),
                    output_len = output.len(),
                    rows = objectives.len(),
                    "Graph-MIP leaf oracle: CONFIRMED in-box counterexample violating EVERY \
                     objective row — published as the run's sat candidate (child stays queued; \
                     the trusted ONNX-Runtime gate remains the verdict authority)"
                );
                *leaf_violation = Some(BabVerificationStatus::Violated {
                    counterexample: witness,
                    output,
                });
            } else {
                tracing::warn!(
                    "Graph-MIP leaf oracle: confirmed SAT witness (advisory — it does not cover \
                     every objective row; child requeued)"
                );
            }
        }
        GraphMipLeafVerdict::Undecided => {
            queue.push(child);
        }
    }
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
    // #violdrop: only the ROOT may be abandoned on a `upper < threshold`
    // reading. A BaB child's objective interval is produced with its split
    // enforced ONLY through the β (Lagrangian) dual, which certifies the LOWER
    // bound direction — the upper end carries no certificate for the child's
    // sub-region. See `shared::violation_drop_is_certified`.
    //
    // This lane is the FALLBACK arm of the same BaB loop as the batched lane
    // (`verify.rs` picks per batch), so leaving it ungated re-created the exact
    // failure: measured on vit_2023 ibp_3_3_8_3005, the root split ran on the
    // batched lane and its two children were then dropped HERE, still emptying
    // the queue at `explored=3 max_depth=1` after 1.72 s of a 90.25 s grant.
    let violated = if conjunctive {
        domain.all_violated(thresholds, false)
    } else {
        domain.any_violated(thresholds, false)
    };
    if violated {
        super::shared::violdrop_site_probe("is_domain_dropped/sequential", domain.depth);
    }
    violated && super::shared::violation_drop_is_certified(domain.depth)
}

/// Return one queued domain's validated aggregation-critical proof margin.
///
/// Queue entries normally carry finite, shape-consistent bounds by
/// construction. This still validates every lookup and arithmetic operation:
/// malformed/non-finite advisory metadata refuses SPSA rather than becoming an
/// admission authority.
fn queued_domain_proof_margin(
    domain: &MultiObjectiveGraphBabDomain,
    thresholds: &[f32],
) -> MoCudaBetaSpsaFrontier {
    let row = match domain.critical_objective_index(thresholds) {
        Ok(Some(row)) => row,
        Ok(None) => return MoCudaBetaSpsaFrontier::Empty,
        Err(_) => return MoCudaBetaSpsaFrontier::Invalid,
    };
    let Some(&(lower, upper)) = domain.objective_bounds().get(row) else {
        return MoCudaBetaSpsaFrontier::Invalid;
    };
    let Some(&threshold) = thresholds.get(row) else {
        return MoCudaBetaSpsaFrontier::Invalid;
    };
    let margin = if domain.verify_upper() {
        threshold - upper
    } else {
        lower - threshold
    };
    if margin.is_finite() {
        MoCudaBetaSpsaFrontier::Finite(margin)
    } else {
        MoCudaBetaSpsaFrontier::Invalid
    }
}

fn combine_proof_frontier(
    margins: impl IntoIterator<Item = MoCudaBetaSpsaFrontier>,
) -> MoCudaBetaSpsaFrontier {
    let mut combined = MoCudaBetaSpsaFrontier::Empty;
    for candidate in margins {
        let candidate = match candidate {
            MoCudaBetaSpsaFrontier::Empty => continue,
            MoCudaBetaSpsaFrontier::Finite(value) if value.is_finite() => value,
            MoCudaBetaSpsaFrontier::Finite(_) | MoCudaBetaSpsaFrontier::Invalid => {
                return MoCudaBetaSpsaFrontier::Invalid;
            }
        };
        combined = match combined {
            MoCudaBetaSpsaFrontier::Empty => MoCudaBetaSpsaFrontier::Finite(candidate),
            MoCudaBetaSpsaFrontier::Finite(current) => {
                MoCudaBetaSpsaFrontier::Finite(current.min(candidate))
            }
            MoCudaBetaSpsaFrontier::Invalid => return MoCudaBetaSpsaFrontier::Invalid,
        };
    }
    combined
}

/// Deterministic completed-open-domain frontier used by the sequential CUDA
/// β-SPSA admission gate.
///
/// The outer loop removes a whole batch from the heap before dispatching the
/// sequential fallback, so the open frontier is the union of the remaining
/// heap and the not-yet-processed suffix of that batch. Excluding that suffix
/// would make SPSA admission depend on `batch_size`/pop partitioning. The
/// current parent is deliberately absent because its two children replace it.
/// Recomputing immediately before each child also lets an enqueued active
/// sibling gate the subsequent inactive sibling.
fn completed_frontier_minimum_proof_margin(
    queue: &BinaryHeap<MultiObjectiveGraphBabDomain>,
    unprocessed_batch_suffix: &[MoCudaBetaSpsaFrontier],
    thresholds: &[f32],
) -> MoCudaBetaSpsaFrontier {
    combine_proof_frontier(
        queue
            .iter()
            .map(|domain| queued_domain_proof_margin(domain, thresholds))
            .chain(unprocessed_batch_suffix.iter().copied()),
    )
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
    ) -> Result<SequentialMultiObjectiveBatchStatus> {
        let SequentialMultiObjectiveContext {
            graph,
            domains_to_process,
            relu_nodes,
            objectives,
            thresholds,
            engine,
            cut_pool,
            clause_store,
            queue,
            domains_verified,
            unresolved_due_to_no_branch,
            unresolved_due_to_violated_drop,
            unresolved_due_to_propagation_failure,
            leaf_violation,
            conjunctive,
            deadline,
        } = context;

        // `pop_domain_batch` removed every entry in `domains_to_process` from
        // `queue`. Preserve their already-certified proof margins so each child
        // sees the complete open frontier: remaining heap plus later parents in
        // this batch, excluding only the parent currently being replaced.
        let track_cuda_beta_spsa_frontier = self.config.beta_iterations > 0
            && mo_cuda_beta_spsa_frontier_tracking_enabled(
                self.effective_graph_bab_deadline().is_some(),
            );
        let batch_proof_margins = track_cuda_beta_spsa_frontier.then(|| {
            domains_to_process
                .iter()
                .map(|domain| queued_domain_proof_margin(domain, thresholds))
                .collect::<Vec<_>>()
        });
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Ok(SequentialMultiObjectiveBatchStatus::DeadlineExpired);
        }

        'domain: for (domain_index, mut domain) in domains_to_process.into_iter().enumerate() {
            // Deadline check: bail early if verification timeout exceeded (#3388).
            // The outer BaB loop checks timeout only between batches. For large
            // networks with many objectives, a single batch of sequential CROWN
            // passes can exceed the timeout budget. This per-domain check ensures
            // we return to the BaB loop promptly for the timeout to fire.
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Ok(SequentialMultiObjectiveBatchStatus::DeadlineExpired);
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
                let context = GraphCrownContext::new_with_node_bounds_map(
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
            let (node_name, neuron_idx, score) = match self.select_graph_branch_multi(
                graph, &domain, &unstable, objectives, thresholds, engine,
            ) {
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
                        // Recompute from all completed queued domains at the
                        // child boundary. The active child, when unresolved,
                        // is pushed below before the inactive child reaches
                        // this point, so sibling admission is dynamic without
                        // shared state or scheduler-order dependence.
                        let cuda_beta_spsa_frontier = batch_proof_margins.as_ref().map_or(
                            MoCudaBetaSpsaFrontier::Empty,
                            |batch_proof_margins| {
                                completed_frontier_minimum_proof_margin(
                                    queue,
                                    &batch_proof_margins[domain_index + 1..],
                                    thresholds,
                                )
                            },
                        );
                        let outcome = self.process_multi_objective_child(
                            &child_ctx,
                            &domain,
                            &mut child,
                            is_active,
                            cuda_beta_spsa_frontier,
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
                                apply_sequential_leaf_verdict(
                                    verdict,
                                    child,
                                    queue,
                                    clause_store,
                                    domains_verified,
                                    &mut verified_histories,
                                    objectives,
                                    thresholds,
                                    leaf_violation,
                                );
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

        Ok(SequentialMultiObjectiveBatchStatus::Completed)
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
        cuda_beta_spsa_frontier: MoCudaBetaSpsaFrontier,
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
        let crown_ctx = GraphCrownContext::new_with_node_bounds_map(
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
        // `.to_vec()`. `prune_cached_las_for_targets` borrows the shared payloads
        // and `merge_pruned_cached_las` Arc-clones unchanged rows, so this reads
        // bit-identical data without cloning coefficient ndarrays.
        let pruned_cached_las = prune_cached_las_for_targets(&child.cached_las, &pruned_targets);
        // Only run β optimization when enabled and for shallow domains
        // When beta_iterations=0, skip optimization entirely for all domains
        let should_optimize =
            self.config.beta_iterations > 0 && child.depth <= self.config.beta_max_depth;
        let result = if should_optimize {
            self.optimize_graph_beta_analytical_multi_objective_with_cache_at_frontier(
                ctx.graph,
                child.input_bounds.as_ref(),
                &crown_ctx,
                &mut child.beta_state,
                &targets,
                ctx.conjunctive,
                &pruned_cached_las,
                true,
                cuda_beta_spsa_frontier,
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
                // #bab-monotone-inherit (dark, NY_BAB_MONOTONE_INHERIT=1):
                // monotone parent-bound inheritance on the SEQUENTIAL lane —
                // the same merge the batched lane has always applied.
                //
                // SOUND: `child` was produced by `domain.with_constraint(..)`
                // a few lines up in `process_multi_objective_domains_sequential`,
                // which appends ONE ReLU/Sign split to the parent's history and
                // may only NARROW `input_bounds`. The child's region is therefore
                // a (strict) SUBSET of the parent's, so any valid parent lower
                // bound is a valid child lower bound: `max(parent_l, child_l)` is
                // sound, and symmetrically `min(parent_u, child_u)`.
                //
                // `child.objective_bounds` is still the verbatim clone of the
                // parent's vector here — `update_bounds` (below) is this lane's
                // first and only write to it, and `merge_pruned_objective_bounds`
                // above returned a fresh Vec without touching the child.
                // Gate absent/malformed => `new_bounds` passes through unchanged,
                // byte-identical to today.
                let new_bounds = if bab_monotone_inherit_enabled() {
                    inherit_parent_lower_only(&child.objective_bounds, new_bounds)
                } else {
                    new_bounds
                };
                // #cone-delta increment 2: already Arc-shared — move.
                child.node_bounds =
                    crate::beta_crown::domain::NodeBoundsMap::from_shared_hash_map(node_cache);
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
                    if child.set_shared_cached_las(merged_cached_las).is_err() {
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

#[cfg(test)]
mod leaf_disposition_tests {
    use std::collections::{BinaryHeap, HashMap};

    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    use super::*;
    use crate::beta_crown::conflict_clauses_graph::{
        reset_test_record_attempts, reset_test_store_mutations, test_record_attempts,
        test_store_mutations,
    };

    fn child_with_pure_relu_history() -> MultiObjectiveGraphBabDomain {
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("test input bounds");
        let mut child = MultiObjectiveGraphBabDomain::root(
            HashMap::new(),
            vec![(-1.0, 1.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("test child");
        child.history.add_constraint(GraphNeuronConstraint {
            node_name: "relu1".to_string(),
            neuron_idx: 0,
            is_active: true,
            score: 1.0,
        });
        child.history.add_constraint(GraphNeuronConstraint {
            node_name: "relu2".to_string(),
            neuron_idx: 1,
            is_active: false,
            score: 1.0,
        });
        child
    }

    #[test]
    fn exact_leaf_verified_all_rows_records_history_once_without_requeueing() {
        reset_test_record_attempts();
        reset_test_store_mutations();
        let child = child_with_pure_relu_history();
        let history = child.history.clone();
        let mut queue = BinaryHeap::new();
        let mut clause_store = GraphClauseStore::with_capacity(true, 16);
        let mut domains_verified = 0;
        let mut verified_histories = Vec::new();

        let mut leaf_violation = None;

        apply_sequential_leaf_verdict(
            GraphMipLeafVerdict::VerifiedAllRows,
            child,
            &mut queue,
            &mut clause_store,
            &mut domains_verified,
            &mut verified_histories,
            &[vec![1.0_f32]],
            &[0.0_f32],
            &mut leaf_violation,
        );

        assert_eq!(leaf_violation, None, "a verified close is not a sat");
        assert_eq!(domains_verified, 1);
        assert!(queue.is_empty(), "a verified child must not be requeued");
        assert_eq!(clause_store.len(), 1);
        assert_eq!(
            test_record_attempts(),
            1,
            "the sequential terminal disposition must invoke recording once"
        );
        assert_eq!(
            test_store_mutations(),
            1,
            "the fresh pure history must produce one store mutation"
        );
        assert!(
            clause_store.should_prune(&history),
            "the exact leaf's pure ReLU region must be reusable"
        );
        assert_eq!(verified_histories.len(), 1);
        assert_eq!(verified_histories[0].constraints, history.constraints);
    }

    /// The CONJUNCTIVE lane's sat return (#mip-leaf-witness). `verify.rs` routes
    /// conjunctive properties here (the batched lane requires `!conjunctive`),
    /// and there "violates every objective row" IS the property's violation
    /// predicate — so an oracle-confirmed witness passing it is an exact
    /// counterexample and must reach the verifier through the typed carrier.
    #[test]
    fn conjunctive_leaf_witness_covering_every_row_returns_sat_and_requeues_the_child() {
        let child = child_with_pure_relu_history();
        let mut queue = BinaryHeap::new();
        let mut clause_store = GraphClauseStore::disabled();
        let mut domains_verified = 0;
        let mut verified_histories = Vec::new();
        let mut leaf_violation = None;

        apply_sequential_leaf_verdict(
            GraphMipLeafVerdict::Violated {
                witness: vec![0.75_f32],
                output: vec![-2.0_f32, 3.0],
            },
            child,
            &mut queue,
            &mut clause_store,
            &mut domains_verified,
            &mut verified_histories,
            // Both conjuncts hold at y = (-2, 3): `1*y0 <= 0` and
            // `-1*y1 <= -1` (i.e. y1 >= 1).
            &[vec![1.0_f32, 0.0], vec![0.0, -1.0]],
            &[0.0_f32, -1.0],
            &mut leaf_violation,
        );

        assert_eq!(
            leaf_violation,
            Some(BabVerificationStatus::Violated {
                counterexample: vec![0.75_f32],
                output: vec![-2.0_f32, 3.0],
            })
        );
        assert_eq!(
            queue.len(),
            1,
            "#violdrop/prop1498: the sat path must requeue the child, never drain"
        );
        assert_eq!(domains_verified, 0);
        assert!(verified_histories.is_empty());
    }

    /// A conjunct that does NOT hold at the witness means the conjunction does
    /// not hold: no counterexample, advisory only, child requeued.
    #[test]
    fn conjunctive_leaf_witness_missing_a_conjunct_stays_advisory() {
        let child = child_with_pure_relu_history();
        let mut queue = BinaryHeap::new();
        let mut clause_store = GraphClauseStore::disabled();
        let mut domains_verified = 0;
        let mut verified_histories = Vec::new();
        let mut leaf_violation = None;

        apply_sequential_leaf_verdict(
            GraphMipLeafVerdict::Violated {
                witness: vec![0.75_f32],
                output: vec![-2.0_f32, 3.0],
            },
            child,
            &mut queue,
            &mut clause_store,
            &mut domains_verified,
            &mut verified_histories,
            // Second conjunct `-1*y1 <= -10` (y1 >= 10) fails at y1 = 3.
            &[vec![1.0_f32, 0.0], vec![0.0, -1.0]],
            &[0.0_f32, -10.0],
            &mut leaf_violation,
        );

        assert_eq!(leaf_violation, None);
        assert_eq!(queue.len(), 1);
        assert_eq!(domains_verified, 0);
    }
}

#[cfg(test)]
mod frontier_tests {
    use std::collections::{BinaryHeap, HashMap};

    use ndarray::arr1;
    use ny_tensor::BoundedTensor;

    use super::{
        combine_proof_frontier, completed_frontier_minimum_proof_margin,
        queued_domain_proof_margin, MoCudaBetaSpsaFrontier, MultiObjectiveGraphBabDomain,
    };

    fn queued_child(lower: f32) -> MultiObjectiveGraphBabDomain {
        let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("valid test input");
        MultiObjectiveGraphBabDomain::root(
            HashMap::new(),
            vec![(lower, 1.0)],
            &input,
            &[0.0],
            false,
        )
        .expect("valid queued child")
    }

    #[test]
    fn minimum_frontier_is_order_independent_and_invalid_metadata_fails_closed() {
        let left = combine_proof_frontier([
            MoCudaBetaSpsaFrontier::Finite(-2.0),
            MoCudaBetaSpsaFrontier::Empty,
            MoCudaBetaSpsaFrontier::Finite(-4.0),
        ]);
        let right = combine_proof_frontier([
            MoCudaBetaSpsaFrontier::Finite(-4.0),
            MoCudaBetaSpsaFrontier::Empty,
            MoCudaBetaSpsaFrontier::Finite(-2.0),
        ]);
        assert_eq!(left, MoCudaBetaSpsaFrontier::Finite(-4.0));
        assert_eq!(right, left);
        for invalid in [
            MoCudaBetaSpsaFrontier::Invalid,
            MoCudaBetaSpsaFrontier::Finite(f32::NAN),
            MoCudaBetaSpsaFrontier::Finite(f32::INFINITY),
            MoCudaBetaSpsaFrontier::Finite(f32::NEG_INFINITY),
        ] {
            assert_eq!(
                combine_proof_frontier([
                    MoCudaBetaSpsaFrontier::Finite(-4.0),
                    invalid,
                    MoCudaBetaSpsaFrontier::Finite(-6.0),
                ]),
                MoCudaBetaSpsaFrontier::Invalid
            );
        }
    }

    #[test]
    fn frontier_is_invariant_to_batch_width_and_heap_suffix_partition() {
        let thresholds = [0.0];
        let partitions: &[(&[f32], &[f32])] = &[
            (&[-6.0, -4.0, -2.0], &[]),
            (&[-4.0, -2.0], &[-6.0]),
            (&[-2.0], &[-4.0, -6.0]),
            (&[], &[-2.0, -6.0, -4.0]),
        ];
        for &(queued, suffix) in partitions {
            let queue = queued
                .iter()
                .map(|&lower| queued_child(lower))
                .collect::<BinaryHeap<_>>();
            let suffix = suffix
                .iter()
                .map(|&lower| queued_domain_proof_margin(&queued_child(lower), &thresholds))
                .collect::<Vec<_>>();
            assert_eq!(
                completed_frontier_minimum_proof_margin(&queue, &suffix, &thresholds),
                MoCudaBetaSpsaFrontier::Finite(-6.0),
                "moving the same completed competitors between the heap and popped-batch \
                 suffix must not change admission"
            );
        }
    }

    #[test]
    fn malformed_heap_or_suffix_frontier_fails_closed() {
        let thresholds = [0.0];
        let queue = BinaryHeap::from([queued_child(-3.0)]);
        assert_eq!(
            completed_frontier_minimum_proof_margin(
                &queue,
                &[MoCudaBetaSpsaFrontier::Invalid],
                &thresholds,
            ),
            MoCudaBetaSpsaFrontier::Invalid
        );
        assert_eq!(
            completed_frontier_minimum_proof_margin(&queue, &[], &[]),
            MoCudaBetaSpsaFrontier::Invalid,
            "a malformed queued-domain threshold view must not look like an empty frontier"
        );
    }

    #[test]
    fn frontier_recomputation_observes_a_newly_completed_sibling() {
        let thresholds = [0.0];
        let mut completed_queued_children = BinaryHeap::new();
        assert_eq!(
            completed_frontier_minimum_proof_margin(&completed_queued_children, &[], &thresholds,),
            MoCudaBetaSpsaFrontier::Empty,
            "the first child has no completed queued frontier"
        );

        completed_queued_children.push(queued_child(-3.0));
        assert_eq!(
            completed_frontier_minimum_proof_margin(&completed_queued_children, &[], &thresholds,),
            MoCudaBetaSpsaFrontier::Finite(-3.0),
            "an enqueued active sibling must gate the inactive sibling"
        );

        completed_queued_children.push(queued_child(-5.0));
        assert_eq!(
            completed_frontier_minimum_proof_margin(&completed_queued_children, &[], &thresholds,),
            MoCudaBetaSpsaFrontier::Finite(-5.0),
            "the gate must use the minimum over all completed queued children"
        );
    }
}
