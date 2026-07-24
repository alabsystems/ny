// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! GPU batched execution path for BaB.
//!
//! When a GPU engine is available, domains are branched (ReLU fast path, BaBSR,
//! or GenBaB), batched together, and evaluated via a single GPU backward pass.
//! This is the hot path for GPU-accelerated verification.
//!
//! # Phases
//!
//! 1. **Branching**: For each picked domain, select a neuron to split on and
//!    create child domains. Multiple strategies: ReLU fast path (batched
//!    intercept scoring), BaBSR (per-domain CROWN coefficient weighting),
//!    and GenBaB (general non-linear node splitting).
//!
//! 2. **Batched evaluation**: Collect all child domains into `BatchedDomains`,
//!    run a single batched backward pass, then process results per-child.
//!
//! 3. **Beta refinement** (#1484): For shallow children, run a per-child
//!    optimization pass to tighten bounds beyond the inherited beta values.
//!
//! Extracted from `verify_graph_gpu_domain_list` lines 253-729.

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::arr1;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{debug, warn};

use crate::batched_domain::{
    BatchedDomainOptions, BatchedDomains, CachedLinearBounds, DomainList, PickedDomains,
};
use crate::beta_crown::branching::{BranchingHeuristic, GraphNeuronConstraint};
use crate::beta_crown::engine::domain_results::GraphDomainResult;
use crate::beta_crown::engine::graph::domain_conversion::{
    branch_relu_from_picked, graph_domain_from_picked, processed_from_backward_results,
};
use crate::beta_crown::engine::graph::propagation::{
    BatchedBackwardContext, BatchedBackwardResult,
};
use crate::beta_crown::engine::graph::DomainCrownResult;
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::BetaCrownVerifier;
use crate::beta_crown::GraphBabDomain;
use crate::GraphNetwork;

use super::check::{check_domain_bounds, BabLoopState, DomainCheckResult};
use super::init::BabSetupContext;

/// Outcome of GPU batched execution for one BaB iteration.
pub(crate) enum GpuBatchOutcome {
    /// Processing completed; surviving children added to domain_list.
    /// Contains captured linear bounds for child domains (may be empty).
    Continue(Vec<Arc<CachedLinearBounds>>),
    /// A violation was found; return immediately.
    Violation,
}

/// Context for one GPU batched iteration, bundling the caller-scope values
/// that the GPU path needs. This avoids a 15+ parameter function signature.
pub(crate) struct GpuBatchContext<'a> {
    pub graph: &'a GraphNetwork,
    pub eng: &'a dyn GemmEngine,
    pub picked: &'a PickedDomains,
    pub processable_picked_indices: &'a [usize],
    pub unstable_batched: &'a [Vec<(String, usize)>],
    pub branches_batched: &'a [Option<(String, usize, f32)>],
    pub is_genbab: bool,
    pub setup: &'a BabSetupContext,
    pub objective: &'a [f32],
    pub threshold: f32,
    pub layer_names: &'a [String],
}

impl BetaCrownVerifier {
    /// Process a batch of picked domains using GPU-accelerated evaluation.
    ///
    /// For each processable domain, selects a branching strategy (ReLU fast path,
    /// BaBSR, or GenBaB), creates child domains, batches them, runs a GPU backward
    /// pass, and processes results. Surviving children are added to `domain_list`.
    ///
    /// # Returns
    /// `GpuBatchOutcome::Violation` if any child proves a violation,
    /// `GpuBatchOutcome::Continue(captured_la)` otherwise.
    pub(crate) fn process_gpu_batched(
        &self,
        ctx: &GpuBatchContext<'_>,
        state: &mut BabLoopState,
        domain_list: &mut DomainList,
    ) -> Result<GpuBatchOutcome> {
        let mut batched_children: Vec<GraphBabDomain> =
            Vec::with_capacity(2 * ctx.processable_picked_indices.len());
        // captured_la_for_children is reassigned via `.collect()` in Phase 3
        // post-refinement compaction, so pre-allocation here has no effect.
        let mut captured_la_for_children: Vec<Arc<CachedLinearBounds>> = Vec::new();

        // Phase 1: Branching — create child domains
        for &picked_idx in ctx.processable_picked_indices {
            let unstable = if ctx.is_genbab {
                Vec::new()
            } else {
                ctx.unstable_batched
                    .get(picked_idx)
                    .cloned()
                    .ok_or_else(|| {
                        NyError::InternalError(format!(
                            "GPU BaB: unstable_batched missing at picked_idx={} \
                             (len={}) — domain would be treated as fully stable",
                            picked_idx,
                            ctx.unstable_batched.len(),
                        ))
                    })?
            };

            // ReLU fast path (Direction 2 of #1668): branch directly from PickedDomains
            // without materializing an intermediate GraphBabDomain.
            if !ctx.is_genbab {
                if unstable.is_empty() {
                    // No unstable neurons — all ReLUs are already fixed, so this
                    // is a fully-decided affine leaf. A single CROWN pass with
                    // inherited β leaves the split constraints loosely enforced
                    // and can return Undecided on a leaf that is in fact
                    // verifiable (#1896). `process_graph_leaf_no_unstable`
                    // optimizes β/α on the leaf and reports the tightest *sound*
                    // bound, so verifiable leaves verify.
                    let domain = graph_domain_from_picked(
                        picked_idx,
                        ctx.picked,
                        ctx.layer_names,
                        self.config.verify_upper_bound,
                        None,
                    )?;
                    match self.process_graph_leaf_no_unstable(
                        ctx.graph,
                        &domain,
                        ctx.objective,
                        ctx.threshold,
                        Some(ctx.eng),
                    ) {
                        GraphDomainResult::AlreadyVerified => {
                            // #2926: Infeasible domain = empty = trivially verified.
                            state.domains_verified += 1;
                        }
                        GraphDomainResult::NoUnstable {
                            lower,
                            upper,
                            verified,
                        } => {
                            if verified {
                                state.domains_verified += 1;
                            } else {
                                match check_domain_bounds(
                                    lower,
                                    upper,
                                    ctx.threshold,
                                    self.config.verify_upper_bound,
                                ) {
                                    DomainCheckResult::Verified => {
                                        // Defensive: helper said unverified but the
                                        // bound clears the threshold — verify.
                                        state.domains_verified += 1;
                                    }
                                    DomainCheckResult::Violation => {
                                        return Ok(GpuBatchOutcome::Violation);
                                    }
                                    DomainCheckResult::Undecided => {
                                        state.unresolved_due_to_no_unstable_neurons = true;
                                    }
                                }
                            }
                        }
                        GraphDomainResult::PropagationFailure => {
                            warn!(
                                picked_idx,
                                domain_depth = domain.depth,
                                "GPU BaB NoUnstable domain propagation failed; aborting to avoid silent unknown classification"
                            );
                            return Err(NyError::InternalError(format!(
                                "GPU BaB: NoUnstable leaf propagation failed at picked_idx={picked_idx}"
                            )));
                        }
                        other => {
                            return Err(NyError::InternalError(format!(
                                "GPU BaB: unexpected leaf result {other:?} at picked_idx={picked_idx}"
                            )));
                        }
                    }
                    continue;
                }

                // Has unstable neurons — try batched branch decision for fast path.
                let requires_materialized_selection = matches!(
                    self.config.branching_heuristic,
                    BranchingHeuristic::BoundImpact
                        | BranchingHeuristic::Kfsb
                        | BranchingHeuristic::KfsbInterceptOnly
                );
                if !requires_materialized_selection {
                    if let Some(Some((ref node_name, neuron_idx, score))) =
                        ctx.branches_batched.get(picked_idx)
                    {
                        let (active, inactive, had_propagation_failure) = branch_relu_from_picked(
                            picked_idx,
                            ctx.picked,
                            ctx.graph,
                            node_name,
                            *neuron_idx,
                            *score,
                            ctx.layer_names,
                            self.config.verify_upper_bound,
                        )?;
                        if had_propagation_failure {
                            state.unresolved_due_to_propagation_failure = true;
                        }
                        if let Some(child) = active {
                            batched_children.push(child);
                        }
                        if let Some(child) = inactive {
                            batched_children.push(child);
                        }
                        continue;
                    }
                }

                // BaBSR or no batched branch decision — use per-domain path
                let domain = graph_domain_from_picked(
                    picked_idx,
                    ctx.picked,
                    ctx.layer_names,
                    self.config.verify_upper_bound,
                    Some(ctx.graph),
                )?;
                let selection = if matches!(
                    self.config.branching_heuristic,
                    BranchingHeuristic::Kfsb | BranchingHeuristic::KfsbInterceptOnly
                ) {
                    self.select_graph_branch_kfsb_in_gpu_batched(
                        ctx.graph,
                        &domain,
                        &unstable,
                        ctx.objective,
                        Some(ctx.eng),
                    )
                } else {
                    self.select_graph_branch(ctx.graph, &domain, &unstable)
                };
                let Some((node_name, neuron_idx, score)) = self
                    .selection_or_mark_propagation_failure_in_gpu_batched(
                        selection,
                        state,
                        picked_idx,
                        domain.depth,
                    )
                else {
                    continue;
                };
                let active_constraint = GraphNeuronConstraint {
                    node_name: node_name.clone(),
                    neuron_idx,
                    is_active: true,
                    score,
                };
                if let Some(child) = domain.with_constraint(
                    ctx.graph,
                    active_constraint,
                    self.config.verify_upper_bound,
                )? {
                    batched_children.push(child);
                }
                let inactive_constraint = GraphNeuronConstraint {
                    node_name,
                    neuron_idx,
                    is_active: false,
                    score,
                };
                if let Some(child) = domain.with_constraint(
                    ctx.graph,
                    inactive_constraint,
                    self.config.verify_upper_bound,
                )? {
                    batched_children.push(child);
                }
                continue;
            }

            // GenBaB path: requires full materialization for find_splittable_graph_nodes
            let domain = graph_domain_from_picked(
                picked_idx,
                ctx.picked,
                ctx.layer_names,
                self.config.verify_upper_bound,
                Some(ctx.graph),
            )?;
            if let Some(genbab) = &ctx.setup.genbab_instance {
                let splittable = self.find_splittable_graph_nodes(
                    ctx.graph,
                    &domain,
                    &ctx.setup.nonlinear_nodes,
                    genbab,
                );
                if splittable.is_empty() {
                    state.unresolved_due_to_genbab_no_split = true;
                    continue;
                }
                if let Some(decision) =
                    self.select_genbab_branch(ctx.graph, &domain, &splittable, genbab)?
                {
                    for split in decision.to_splits()? {
                        if let Some(child) = domain.with_general_split(
                            ctx.graph,
                            split,
                            self.config.verify_upper_bound,
                        )? {
                            batched_children.push(child);
                        }
                    }
                } else {
                    state.unresolved_due_to_genbab_no_split = true;
                }
            }
        }

        // Phase 2: Batched evaluation
        if batched_children.is_empty() {
            return Ok(GpuBatchOutcome::Continue(captured_la_for_children));
        }

        let child_refs: Vec<&GraphBabDomain> = batched_children.iter().collect();
        let batched_options = BatchedDomainOptions {
            enable_interm_transfer: self.config.enable_interm_transfer,
        };
        let batched = BatchedDomains::from_graph_domains_with_options(
            &child_refs,
            ctx.layer_names,
            batched_options,
        )?;
        let bctx = BatchedBackwardContext::from_domains(&child_refs, &batched)?;

        // Use lA-capturing variant for backward pass.
        #[allow(clippy::type_complexity)]
        let (child_results, captured_la): (
            Vec<Option<DomainCrownResult>>,
            Option<Vec<HashMap<String, crate::LinearBounds>>>,
        ) = match self.propagate_crown_batched_with_context_capture_la(
            ctx.graph,
            &bctx,
            ctx.objective,
            ctx.eng,
        ) {
            Ok(BatchedBackwardResult {
                results,
                intermediate_la,
                stage_timing: _, // Timing consumed by domain_batch metrics layer
            }) => (results.into_iter().map(Some).collect(), intermediate_la),
            Err(err) => {
                warn!(
                    "GPU BaB batched backward failed ({}); falling back to sequential with beta optimization",
                    err
                );
                // Part of #1484: Use evaluate_graph_child_bounds for fallback
                let fallback_results: Vec<_> = batched_children
                    .iter_mut()
                    .enumerate()
                    .map(|(fallback_child_idx, child)| {
                        let parent_bounds = child.node_bounds.clone();
                        match self.evaluate_graph_child_bounds(
                            ctx.graph,
                            child,
                            &parent_bounds,
                            ctx.objective,
                            None,
                            Some(ctx.eng),
                        ) {
                            Ok(true) => {
                                // Guard: NaN/Inf in BaB child scalar bounds →
                                // drop child (unsound to propagate).
                                if !child.lower_bound.is_finite()
                                    || !child.upper_bound.is_finite()
                                {
                                    warn!(
                                        child_idx = fallback_child_idx,
                                        child_depth = child.depth,
                                        lower = child.lower_bound,
                                        upper = child.upper_bound,
                                        "GPU BaB fallback child dropped: non-finite bounds"
                                    );
                                    return None;
                                }
                                let output = BoundedTensor::new(
                                    arr1(&[child.lower_bound]).into_dyn(),
                                    arr1(&[child.upper_bound]).into_dyn(),
                                );
                                match output {
                                    Ok(bt) => {
                                        // #cone-delta increment 2: Arc-clone the
                                        // domain's map (no tensor copies).
                                        Some((bt, child.node_bounds.clone()))
                                    }
                                    Err(err) => {
                                        warn!(
                                            child_idx = fallback_child_idx,
                                            child_depth = child.depth,
                                            error = %err,
                                            "GPU BaB fallback child dropped: failed to build output bounded tensor"
                                        );
                                        None
                                    }
                                }
                            }
                            Ok(false) => {
                                warn!(
                                    child_idx = fallback_child_idx,
                                    child_depth = child.depth,
                                    "GPU BaB fallback child dropped: evaluate_graph_child_bounds returned no bounds"
                                );
                                None
                            }
                            Err(ref err) if err.is_infeasible_domain() => {
                                // #2926: Infeasible domain = empty = verified.
                                debug!(
                                    child_idx = fallback_child_idx,
                                    child_depth = child.depth,
                                    "GPU BaB fallback: infeasible domain (empty), pruning"
                                );
                                None
                            }
                            Err(err) => {
                                warn!(
                                    child_idx = fallback_child_idx,
                                    child_depth = child.depth,
                                    error = %err,
                                    "GPU BaB fallback child dropped: evaluate_graph_child_bounds failed"
                                );
                                None
                            }
                        }
                    })
                    .collect();
                (fallback_results, None)
            }
        };

        // Phase 3: Process backward results and build ProcessedDomains
        let n_children = batched_children.len();
        let mut lower_bounds_vec = Vec::with_capacity(n_children);
        let mut upper_bounds_vec = Vec::with_capacity(n_children);
        let mut node_caches_vec: Vec<HashMap<String, Arc<BoundedTensor>>> =
            Vec::with_capacity(n_children);
        let mut keep_mask_vec = Vec::with_capacity(n_children);

        let should_optimize_any = self.config.beta_iterations > 0;
        let mut needs_refinement: Vec<usize> = Vec::new();
        // Per-child lA cache indexed by child_idx (not kept-domain order).
        // Compacted to kept-order after refinement to prevent misalignment
        // when refinement flips keep_mask entries. Fix: #1916.
        let mut captured_la_per_child: Vec<Option<Arc<CachedLinearBounds>>> =
            vec![None; n_children];

        for (child_idx, (child, bounds_result)) in
            batched_children.iter().zip(child_results).enumerate()
        {
            let Some((output, node_cache)) = bounds_result else {
                lower_bounds_vec.push(f32::NEG_INFINITY);
                upper_bounds_vec.push(f32::INFINITY);
                node_caches_vec.push(HashMap::new());
                keep_mask_vec.push(false);
                state.unresolved_due_to_propagation_failure = true;
                continue;
            };
            let lower = output.lower_scalar();
            let upper = output.upper_scalar();

            // Guard: NaN/Inf from GPU backward pass → drop child (#2652).
            // Matches fallback path guard (line 338). Without this, NaN domains
            // fall through to Undecided and loop indefinitely.
            if !lower.is_finite() || !upper.is_finite() {
                warn!(
                    child_idx,
                    child_depth = child.depth,
                    lower,
                    upper,
                    "GPU BaB primary path child dropped: non-finite bounds"
                );
                lower_bounds_vec.push(f32::NEG_INFINITY);
                upper_bounds_vec.push(f32::INFINITY);
                node_caches_vec.push(HashMap::new());
                keep_mask_vec.push(false);
                state.unresolved_due_to_propagation_failure = true;
                continue;
            }

            lower_bounds_vec.push(lower);
            upper_bounds_vec.push(upper);

            state.max_depth_reached = state.max_depth_reached.max(child.depth);
            if child.depth >= self.config.max_depth {
                state.unresolved_due_to_depth = true;
                node_caches_vec.push(node_cache);
                keep_mask_vec.push(false);
                continue;
            }

            match check_domain_bounds(lower, upper, ctx.threshold, self.config.verify_upper_bound) {
                DomainCheckResult::Verified => {
                    state.domains_verified += 1;
                    node_caches_vec.push(node_cache);
                    keep_mask_vec.push(false);
                    continue;
                }
                DomainCheckResult::Violation => {
                    return Ok(GpuBatchOutcome::Violation);
                }
                DomainCheckResult::Undecided => {}
            }

            // Capture lA for this surviving child if available.
            // Stored per-child (indexed by child_idx), not in kept-order,
            // so post-refinement keep_mask flips don't misalign entries.
            // Compacted to kept-order after refinement. Fix: #1916.
            if let Some(ref la_vec) = captured_la {
                if let Some(la_map) = la_vec.get(child_idx) {
                    let cached = CachedLinearBounds::from_linear_bounds_map(la_map.clone());
                    captured_la_per_child[child_idx] = Some(Arc::new(cached));
                }
            }

            node_caches_vec.push(node_cache);
            keep_mask_vec.push(true);

            // Part of #1484: Mark shallow unverified children for beta refinement
            if should_optimize_any && child.depth <= self.config.beta_max_depth {
                needs_refinement.push(child_idx);
            }
        }

        // Structural invariant: parallel result arrays must match n_children.
        // (#2671, #2920 WP-B: upgraded from debug_assert to runtime check)
        if lower_bounds_vec.len() != n_children
            || upper_bounds_vec.len() != n_children
            || node_caches_vec.len() != n_children
            || keep_mask_vec.len() != n_children
        {
            return Err(NyError::InternalError(format!(
                "process_gpu_batched: parallel array length mismatch — \
                 n_children={}, lower={}, upper={}, caches={}, mask={}",
                n_children,
                lower_bounds_vec.len(),
                upper_bounds_vec.len(),
                node_caches_vec.len(),
                keep_mask_vec.len()
            )));
        }

        // Part of #1484: Post-batched beta optimization refinement
        for &child_idx in &needs_refinement {
            let child = &mut batched_children[child_idx];
            let parent_bounds = child.node_bounds.clone();
            match self.evaluate_graph_child_bounds(
                ctx.graph,
                child,
                &parent_bounds,
                ctx.objective,
                None,
                Some(ctx.eng),
            ) {
                Ok(true) => {
                    // Guard: NaN/Inf from beta refinement → keep batched bounds (#2652).
                    if !child.lower_bound.is_finite() || !child.upper_bound.is_finite() {
                        warn!(
                            child_idx,
                            child_depth = child.depth,
                            lower = child.lower_bound,
                            upper = child.upper_bound,
                            "GPU BaB beta refinement child has non-finite bounds, \
                             keeping batched bounds"
                        );
                        // Don't update bounds — keep the batched values from the
                        // primary path (which already passed the is_finite guard).
                        continue;
                    }

                    lower_bounds_vec[child_idx] = child.lower_bound;
                    upper_bounds_vec[child_idx] = child.upper_bound;
                    // #cone-delta increment 2: Arc-clone the domain's map
                    // (no tensor copies).
                    node_caches_vec[child_idx] = child.node_bounds.clone();

                    match check_domain_bounds(
                        child.lower_bound,
                        child.upper_bound,
                        ctx.threshold,
                        self.config.verify_upper_bound,
                    ) {
                        DomainCheckResult::Verified => {
                            state.domains_verified += 1;
                            keep_mask_vec[child_idx] = false;
                        }
                        DomainCheckResult::Violation => {
                            return Ok(GpuBatchOutcome::Violation);
                        }
                        DomainCheckResult::Undecided => {}
                    }
                }
                Ok(false) => {
                    warn!(
                        child_idx,
                        child_depth = child.depth,
                        "[#1484/#1852] beta refinement returned no update, keeping batched bounds"
                    );
                }
                Err(ref err) if err.is_infeasible_domain() => {
                    // #2926: Constraint interaction revealed infeasible domain
                    // during refinement. Drop this child — the domain is empty.
                    debug!(
                        child_idx,
                        child_depth = child.depth,
                        "Beta refinement: infeasible domain (empty), dropping"
                    );
                    keep_mask_vec[child_idx] = false;
                    state.domains_verified += 1;
                }
                Err(err) => {
                    warn!(
                        child_idx,
                        child_depth = child.depth,
                        error = %err,
                        "[#1484/#1852] beta refinement failed, keeping batched bounds"
                    );
                }
            }
        }

        // Compact per-child lA into final kept-domain order using the
        // post-refinement keep_mask. Before this fix (#1916), lA entries
        // were accumulated in Phase 3 (pre-refinement kept order), so
        // children verified during refinement left stale entries that
        // shifted lA assignments for subsequent kept children.
        captured_la_for_children = captured_la_per_child
            .into_iter()
            .zip(keep_mask_vec.iter())
            .filter(|(_, &kept)| kept)
            .filter_map(|(la, _)| la)
            .collect();

        // Build ProcessedDomains from backward results
        let has_kept = keep_mask_vec.iter().any(|&k| k);
        if has_kept {
            let cached_la_opt = if captured_la_for_children.is_empty() {
                None
            } else {
                Some(std::mem::take(&mut captured_la_for_children))
            };
            let processed = processed_from_backward_results(
                node_caches_vec,
                &batched_children,
                &lower_bounds_vec,
                &upper_bounds_vec,
                &keep_mask_vec,
                ctx.layer_names,
                cached_la_opt,
            )?;
            domain_list.add(processed)?;
        }

        Ok(GpuBatchOutcome::Continue(captured_la_for_children))
    }

    fn selection_or_mark_propagation_failure_in_gpu_batched(
        &self,
        selection: Result<(String, usize, f32)>,
        state: &mut BabLoopState,
        picked_idx: usize,
        domain_depth: usize,
    ) -> Option<(String, usize, f32)> {
        match selection {
            Ok(selection) => Some(selection),
            Err(err) => {
                warn!(
                    picked_idx,
                    domain_depth,
                    error = %err,
                    "select_graph_branch failed in GPU batched loop; marking picked domain as PropagationFailure (#2097, #2038)"
                );
                state.unresolved_due_to_propagation_failure = true;
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// Regression test for #2097: GPU batched branch-selection failure must set
    /// propagation-failure state and continue, not abort process_gpu_batched.
    #[ntest::timeout(5000)]
    #[test]
    fn test_select_graph_branch_failure_marks_propagation_failure_2097() {
        let verifier = BetaCrownVerifier::new(crate::beta_crown::BetaCrownConfig::default());

        let mut graph = GraphNetwork::new();
        graph.add_node(crate::GraphNode::from_input(
            "relu",
            crate::Layer::ReLU(crate::ReLULayer),
        ));
        graph.add_node(crate::GraphNode::new(
            "linear1",
            crate::Layer::Linear(
                crate::LinearLayer::new(ndarray::arr2(&[[1.0, 1.0]]), None).unwrap(),
            ),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear1");

        let input = BoundedTensor::new(
            arr1(&[-1.0f32, -1.0]).into_dyn(),
            arr1(&[1.0f32, 1.0]).into_dyn(),
        )
        .unwrap();
        let domain = GraphBabDomain::root(HashMap::new(), -1.0, 1.0, &input, false).unwrap();

        let mut state = BabLoopState::new(Instant::now());
        let empty_unstable: Vec<(String, usize)> = vec![];
        let selection = verifier.selection_or_mark_propagation_failure_in_gpu_batched(
            verifier.select_graph_branch(&graph, &domain, &empty_unstable),
            &mut state,
            0,
            domain.depth,
        );

        assert!(
            selection.is_none(),
            "select_graph_branch failure should suppress branch selection, got {selection:?}"
        );
        assert!(
            state.unresolved_due_to_propagation_failure,
            "branch-selection failure must mark unresolved_due_to_propagation_failure"
        );
    }

    /// Regression test for #2784: when `branch_relu_from_picked` cannot build a
    /// child due to invalid bounds (for example NaN contamination), GPU BaB must
    /// mark propagation failure so final classification is Unknown, not Verified.
    #[ntest::timeout(5000)]
    #[test]
    fn test_branch_relu_child_construction_failure_marks_unknown_2784() -> Result<()> {
        use ndarray::{ArrayD as AD, IxDyn as Ix};

        let mut graph = GraphNetwork::new();
        graph.add_node(crate::GraphNode::from_input(
            "relu",
            crate::Layer::ReLU(crate::ReLULayer),
        ));
        graph.set_output("relu");

        let picked = PickedDomains {
            batch_size: 1,
            layer_lowers: HashMap::new(),
            layer_uppers: HashMap::new(),
            // NaN simulates contaminated parent input bounds from an upstream pass.
            input_lowers: AD::from_shape_vec(Ix(&[1, 1]), vec![f32::NAN]).unwrap(),
            input_uppers: AD::from_shape_vec(Ix(&[1, 1]), vec![1.0]).unwrap(),
            global_lbs: vec![-1.0],
            global_ubs: vec![1.0],
            metadata: vec![crate::batched_domain::DomainMetadata {
                lower_bound: -1.0,
                upper_bound: 1.0,
                depth: 0,
                constraints: vec![],
                cached_la: None,
                needs_bounding: false,
                node_bounds_override: None,
                alpha_state: None,
            }],
        };

        let mut state = BabLoopState::new(Instant::now());
        let layer_names: Vec<String> = vec![];
        let (_active, _inactive, had_propagation_failure) =
            branch_relu_from_picked(0, &picked, &graph, "relu", 0, 0.5, &layer_names, false)?;
        if had_propagation_failure {
            state.unresolved_due_to_propagation_failure = true;
        }

        assert!(
            state.unresolved_due_to_propagation_failure,
            "child-construction failure must set unresolved_due_to_propagation_failure"
        );
        let result = state.build_final_result();
        match result.result {
            crate::beta_crown::result::BabVerificationStatus::Unknown { reason } => {
                assert!(
                    reason.contains("Child propagation failed"),
                    "reason must include propagation failure, got: {reason}",
                );
            }
            other => unreachable!("expected Unknown after propagation failure, got {other:?}"),
        }

        Ok(())
    }

    /// Build a CachedLinearBounds with a single node whose coefficient equals
    /// `coeff`, so different children's lA caches are distinguishable.
    fn make_tagged_cached_la(coeff: f32) -> CachedLinearBounds {
        use ndarray::{arr1, arr2};
        let mut lower_a = HashMap::new();
        let mut upper_a = HashMap::new();
        let mut lower_b = HashMap::new();
        let mut upper_b = HashMap::new();
        lower_a.insert("node".to_string(), arr2(&[[coeff]]));
        upper_a.insert("node".to_string(), arr2(&[[coeff]]));
        lower_b.insert("node".to_string(), arr1(&[coeff]));
        upper_b.insert("node".to_string(), arr1(&[coeff]));
        CachedLinearBounds {
            lower_a,
            upper_a,
            lower_b,
            upper_b,
        }
    }

    /// Regression test for #1916: lA compaction must skip children that were
    /// verified during refinement, preserving correct lA identity for later
    /// kept children.
    #[ntest::timeout(5000)]
    #[test]
    fn test_la_compaction_skips_refinement_verified_1916() {
        let captured: Vec<Option<CachedLinearBounds>> = vec![
            Some(make_tagged_cached_la(1.0)), // child A
            Some(make_tagged_cached_la(2.0)), // child B (verified in refinement)
            Some(make_tagged_cached_la(3.0)), // child C
        ];
        let keep_mask = [true, false, true];

        let compacted: Vec<Arc<CachedLinearBounds>> = captured
            .into_iter()
            .zip(keep_mask.iter())
            .filter(|(_, &kept)| kept)
            .filter_map(|(la, _)| la.map(Arc::new))
            .collect();

        assert_eq!(compacted.len(), 2);
        assert_eq!(compacted[0].lower_a["node"][[0, 0]], 1.0); // child A
        assert_eq!(compacted[1].lower_a["node"][[0, 0]], 3.0); // child C, not B
    }

    /// Regression test for #1916 end-to-end: compacted lA flows correctly
    /// through `processed_from_backward_results` into DomainMetadata.
    #[ntest::timeout(5000)]
    #[test]
    fn test_la_compacted_metadata_alignment_1916() -> Result<()> {
        use ndarray::arr1;

        let compacted = vec![
            Arc::new(make_tagged_cached_la(1.0)),
            Arc::new(make_tagged_cached_la(3.0)),
        ];
        let keep_mask = [true, false, true];

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();
        let children: Vec<GraphBabDomain> = (0..3)
            .map(|_| GraphBabDomain::root(HashMap::new(), -1.0, 1.0, &input_bounds, false).unwrap())
            .collect();

        let processed = processed_from_backward_results(
            vec![HashMap::new(); 3],
            &children,
            &[-0.5, -0.1, -0.3],
            &[0.5, 0.1, 0.3],
            &keep_mask,
            &[],
            Some(compacted),
        )?;

        assert_eq!(processed.metadata.len(), 2);
        let la_a = processed.metadata[0].cached_la.as_ref().unwrap();
        let la_c = processed.metadata[1].cached_la.as_ref().unwrap();
        assert_eq!(la_a.lower_a["node"][[0, 0]], 1.0, "child A lA coeff");
        assert_eq!(la_c.lower_a["node"][[0, 0]], 3.0, "child C lA coeff");

        Ok(())
    }
    /// Regression test for #1916 audit: when beta refinement keeps every child,
    /// compaction must be the identity and preserve lA order end-to-end.
    #[ntest::timeout(5000)]
    #[test]
    fn test_la_compaction_preserves_identity_without_refinement_flips_1916() -> Result<()> {
        use ndarray::arr1;

        let compacted = vec![
            Arc::new(make_tagged_cached_la(1.0)),
            Arc::new(make_tagged_cached_la(2.0)),
            Arc::new(make_tagged_cached_la(3.0)),
        ];
        let keep_mask = vec![true, true, true];

        let input_bounds =
            BoundedTensor::new(arr1(&[-1.0f32]).into_dyn(), arr1(&[1.0f32]).into_dyn()).unwrap();
        let children: Vec<GraphBabDomain> = (0..3)
            .map(|_| GraphBabDomain::root(HashMap::new(), -1.0, 1.0, &input_bounds, false).unwrap())
            .collect();

        let processed = processed_from_backward_results(
            vec![HashMap::new(); 3],
            &children,
            &[-0.5, -0.1, -0.3],
            &[0.5, 0.1, 0.3],
            &keep_mask,
            &[],
            Some(compacted),
        )?;

        assert_eq!(processed.metadata.len(), 3);
        for (idx, expected) in [1.0_f32, 2.0, 3.0].into_iter().enumerate() {
            let la = processed.metadata[idx].cached_la.as_ref().unwrap();
            assert_eq!(la.lower_a["node"][[0, 0]], expected, "child {idx} lA coeff");
        }

        Ok(())
    }
}
