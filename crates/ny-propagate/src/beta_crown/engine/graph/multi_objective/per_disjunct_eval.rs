// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-disjunct alpha evaluation in BaB child domain processing (#4355).
//!
//! When `optimize_disjuncts_separately` is enabled and the domain carries
//! per-disjunct alpha states, evaluates each unverified disjunct with its
//! own alpha state. Two execution modes:
//!
//! - **Batched** (default, ≥2 active disjuncts): packs active disjuncts as
//!   pseudo-domains into a single batched CROWN backward pass. Forward bounds
//!   computed once and shared. Uses `propagate_crown_batched_backward_core_per_domain_obj`.
//!
//! - **Serial reference** (1 active disjunct or batched fallback): runs one
//!   independent CROWN pass per disjunct. Kept as a correctness reference.
//!
//! Reference: alpha-beta-CROWN `optimize_disjuncts_separately`.
//! Source: `beta_CROWN_solver.py:1098`

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::beta_crown::domain::{
    GraphCrownContext, MultiObjectiveGraphBabDomain, MultiObjectiveTargets,
};
use crate::GraphNetwork;

use super::super::super::BetaCrownVerifier;
use super::shared::{
    merge_pruned_cached_las, merge_pruned_objective_bounds, prune_verified_multi_objective_targets,
    PrunedMultiObjectiveTargets,
};
// #bab-monotone-inherit: shared monotone parent-bound merge + its dark gate,
// re-exported at `multi_objective` module scope.
use super::{bab_monotone_inherit_enabled, inherit_parent_lower_only};

/// Outcome of processing a single BaB child domain via per-disjunct evaluation.
///
/// Mirrors `ChildOutcome` in `sequential.rs`. Separate type to avoid circular
/// dependency while keeping the sequential module under 500 lines.
pub(super) enum PerDisjunctChildOutcome {
    /// Child was verified (contributes to `domains_verified` count).
    Verified(Option<Box<crate::beta_crown::branching::GraphSplitHistory>>),
    /// Child was dropped due to conclusive violation.
    Dropped,
    /// Child is unresolved and ready to enqueue.
    Enqueued,
    /// Propagation failure — sub-region unexplored.
    PropagationFailure,
    /// NaN corruption in child bounds.
    NaNCorruption,
}

/// Shared immutable context for per-disjunct child processing.
pub(super) struct PerDisjunctContext<'a> {
    pub(super) graph: &'a GraphNetwork,
    pub(super) objectives: &'a [Vec<f32>],
    pub(super) thresholds: &'a [f32],
    pub(super) engine: Option<&'a dyn ny_core::GemmEngine>,
    pub(super) cut_pool: &'a crate::beta_crown::bab_cuts::GraphCutPool,
    pub(super) conjunctive: bool,
}

impl BetaCrownVerifier {
    /// Process a child domain using per-disjunct alpha states (#4355).
    ///
    /// Dispatches to batched evaluation when ≥2 active disjuncts; falls back to
    /// serial reference for 0-1 active disjuncts or on batched error.
    pub(super) fn process_multi_objective_child_per_disjunct(
        &self,
        ctx: &PerDisjunctContext<'_>,
        parent: &MultiObjectiveGraphBabDomain,
        child: &mut MultiObjectiveGraphBabDomain,
        is_active: bool,
    ) -> Result<PerDisjunctChildOutcome> {
        let pruned_targets =
            prune_verified_multi_objective_targets(ctx.objectives, ctx.thresholds, &child.verified);

        if pruned_targets.active_indices.len() >= 2 {
            match self.process_per_disjunct_batched(ctx, parent, child, is_active, &pruned_targets)
            {
                Ok(outcome) => return Ok(outcome),
                Err(e) => {
                    tracing::warn!("Batched per-disjunct failed, falling back to serial: {e}");
                }
            }
        }

        self.process_per_disjunct_serial(ctx, parent, child, is_active, &pruned_targets)
    }

    /// Batched per-disjunct evaluation (#4355 Packet B).
    ///
    /// Packs N active disjuncts as N pseudo-domains with the same constrained
    /// input but per-disjunct alpha and objective. Forward bounds computed once.
    fn process_per_disjunct_batched(
        &self,
        ctx: &PerDisjunctContext<'_>,
        parent: &MultiObjectiveGraphBabDomain,
        child: &mut MultiObjectiveGraphBabDomain,
        _is_active: bool,
        pruned_targets: &PrunedMultiObjectiveTargets,
    ) -> Result<PerDisjunctChildOutcome> {
        let per_disjunct_alphas = child.per_disjunct_alphas().ok_or_else(|| {
            ny_core::NyError::InternalError(
                "per_disjunct_alphas must be Some when this method is called".into(),
            )
        })?;
        let inherited_cached_las = child.cached_las().to_vec();
        let n_active = pruned_targets.active_indices.len();

        // Compute constrained forward bounds ONCE for the shared child input.
        // #cone-delta: the child inherited `node_bounds` verbatim from
        // `parent`, so its delta describes exactly the base map passed here
        // (dark, NY_CONE_REFRESH-gated).
        let (mut bounds_cache, constrained_input) = self
            .compute_constrained_forward_bounds_from_view(
                ctx.graph,
                child.input_bounds.as_ref(),
                &child.history,
                Some((&parent.node_bounds).into()),
                Some(&child.delta_pre_nodes),
            )?;
        let per_domain_objectives: Vec<Vec<f32>> = pruned_targets.objectives.clone();
        let output_dim = per_domain_objectives.first().map_or(0, Vec::len);
        if output_dim == 0
            || per_domain_objectives
                .iter()
                .any(|objective| objective.len() != output_dim)
        {
            return Err(ny_core::NyError::InvalidSpec(
                "per-disjunct objectives must be a nonempty rectangular matrix".into(),
            ));
        }
        let stacked_spec = ndarray::Array2::from_shape_vec(
            (n_active, output_dim),
            per_domain_objectives.iter().flatten().copied().collect(),
        )
        .map_err(|error| {
            ny_core::NyError::InvalidSpec(format!(
                "failed to stack per-disjunct objectives: {error}"
            ))
        })?;
        let representative_alpha = pruned_targets
            .active_indices
            .first()
            .and_then(|&full_idx| per_disjunct_alphas.get(full_idx))
            .ok_or_else(|| {
                ny_core::NyError::InternalError("active per-disjunct alpha state is missing".into())
            })?;
        let clip_context = GraphCrownContext::new_with_node_bounds_map(
            &child.history,
            Some(ctx.cut_pool),
            Some(&parent.node_bounds),
            ctx.engine,
        )
        .with_alpha(representative_alpha)
        .with_delta_seeds(&child.delta_pre_nodes);
        let exec_order = ctx.graph.exec_order()?;
        self.maybe_apply_complete_clip_root_bank(
            ctx.graph,
            &clip_context,
            Some(&child.beta_state),
            None,
            Some(&stacked_spec),
            &constrained_input,
            exec_order,
            &mut bounds_cache,
        );

        // Build parallel arrays for N pseudo-domains (all share the same forward).
        // #cone-delta increment 2: per-pseudo-domain map clones are Arc-clones
        // (entry aliasing), not tensor copies.
        let bounds_caches: Vec<HashMap<String, Arc<BoundedTensor>>> =
            std::iter::repeat_with(|| bounds_cache.clone())
                .take(n_active)
                .collect();
        let constrained_inputs: Vec<BoundedTensor> =
            std::iter::repeat_with(|| constrained_input.clone())
                .take(n_active)
                .collect();
        let beta_states: Vec<Option<&_>> =
            std::iter::repeat_n(Some(&child.beta_state), n_active).collect();
        let alpha_states: Vec<Option<&_>> = pruned_targets
            .active_indices
            .iter()
            .map(|&full_idx| Some(&per_disjunct_alphas[full_idx]))
            .collect();
        let plan = ctx.graph.dispatch_plan()?;
        let engine: &dyn ny_core::GemmEngine = ctx.engine.unwrap_or(&ny_core::NaiveCpuGemmEngine);

        let result = self.propagate_crown_batched_backward_core_per_domain_obj(
            ctx.graph,
            n_active,
            plan,
            &bounds_caches,
            &constrained_inputs,
            &beta_states,
            &alpha_states,
            &per_domain_objectives,
            engine,
            self.config.enable_la_warm_start,
            None,
        )?;

        // Map batched results back to active bounds and node caches.
        let mut active_bounds: Vec<(f32, f32)> = Vec::with_capacity(n_active);
        let mut merged_node_bounds: Option<HashMap<String, Arc<BoundedTensor>>> = None;

        for (crown_output, node_cache) in result.results {
            // Each domain has a 1-element objective, so crown_output has shape [1].
            let lower = crown_output
                .lower()
                .iter()
                .next()
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            let upper = crown_output
                .upper()
                .iter()
                .next()
                .copied()
                .unwrap_or(f32::INFINITY);
            active_bounds.push((lower, upper));

            // Intersect node bounds across disjunct passes for tighter bounds.
            match &mut merged_node_bounds {
                None => {
                    merged_node_bounds = Some(node_cache);
                }
                Some(existing) => {
                    for (name, new_bt) in node_cache {
                        if let Some(prev_bt) = existing.get(&name) {
                            // Fresh Arc for the intersected tensor: entries are
                            // replaced, never mutated in place (#cone-delta
                            // increment 2 aliasing rule).
                            if let Some((tighter, _)) =
                                prev_bt.intersection_per_element(new_bt.as_ref())
                            {
                                existing.insert(name, Arc::new(tighter));
                            }
                        } else {
                            existing.insert(name, new_bt);
                        }
                    }
                }
            }
        }

        // Extract captured lA from intermediate results.
        let active_cached_las: Vec<Option<crate::batched_domain::CachedLinearBounds>> =
            if let Some(intermediate_la) = result.intermediate_la {
                intermediate_la
                    .into_iter()
                    .map(|la_map| {
                        if la_map.is_empty() {
                            None
                        } else {
                            Some(
                                crate::batched_domain::CachedLinearBounds::from_linear_bounds_map(
                                    la_map,
                                ),
                            )
                        }
                    })
                    .collect()
            } else {
                std::iter::repeat_n(None, n_active).collect()
            };

        apply_per_disjunct_results(
            child,
            pruned_targets,
            &inherited_cached_las,
            active_bounds,
            active_cached_las,
            merged_node_bounds,
            ctx.thresholds,
            ctx.conjunctive,
            self.config.enable_cuts,
        )
    }

    /// Serial reference per-disjunct evaluation (original implementation).
    ///
    /// Runs one independent CROWN pass per active disjunct. Kept as reference
    /// and fallback for ≤1 active disjuncts or batched errors.
    fn process_per_disjunct_serial(
        &self,
        ctx: &PerDisjunctContext<'_>,
        parent: &MultiObjectiveGraphBabDomain,
        child: &mut MultiObjectiveGraphBabDomain,
        is_active: bool,
        pruned_targets: &PrunedMultiObjectiveTargets,
    ) -> Result<PerDisjunctChildOutcome> {
        let per_disjunct_alphas = child.per_disjunct_alphas().ok_or_else(|| {
            ny_core::NyError::InternalError(
                "per_disjunct_alphas must be Some when this method is called".into(),
            )
        })?;
        let cut_pool_ref = if self.config.enable_cuts && !ctx.cut_pool.is_empty() {
            Some(ctx.cut_pool)
        } else {
            None
        };
        let inherited_cached_las = child.cached_las().to_vec();

        let mut active_bounds: Vec<(f32, f32)> =
            Vec::with_capacity(pruned_targets.active_indices.len());
        let mut active_cached_las: Vec<Option<crate::batched_domain::CachedLinearBounds>> =
            Vec::with_capacity(pruned_targets.active_indices.len());
        let mut merged_node_bounds: Option<HashMap<String, Arc<BoundedTensor>>> = None;

        for (active_pos, &full_idx) in pruned_targets.active_indices.iter().enumerate() {
            let alpha = &per_disjunct_alphas[full_idx];
            // #cone-delta: same base/delta pairing as the batched arm above.
            let crown_ctx = GraphCrownContext::new_with_node_bounds_map(
                &child.history,
                cut_pool_ref,
                Some(&parent.node_bounds),
                ctx.engine,
            )
            .with_alpha(alpha)
            .with_delta_seeds(&child.delta_pre_nodes);

            let single_obj = [pruned_targets.objectives[active_pos].clone()];
            let single_thresh = [pruned_targets.thresholds[active_pos]];
            let single_verified = [false];
            let targets = MultiObjectiveTargets::new(&single_obj, &single_thresh, &single_verified);

            let seed_cache = [inherited_cached_las
                .get(full_idx)
                .and_then(Option::as_deref)];

            let result = self.propagate_multi_objective_with_beta_and_cache(
                ctx.graph,
                child.input_bounds.as_ref(),
                &crown_ctx,
                &child.beta_state,
                &targets,
                &seed_cache,
                true,
            );

            match result {
                Ok((obj_bounds, node_cache, cached_la)) => {
                    if let Some(&bound) = obj_bounds.first() {
                        active_bounds.push(bound);
                    } else {
                        let label = if is_active { "Active" } else { "Inactive" };
                        tracing::warn!("{label} per-disjunct {full_idx} returned empty bounds");
                        return Ok(PerDisjunctChildOutcome::PropagationFailure);
                    }
                    active_cached_las.push(cached_la.into_iter().next().flatten());

                    match &mut merged_node_bounds {
                        None => {
                            merged_node_bounds = Some(node_cache);
                        }
                        Some(existing) => {
                            for (name, new_bt) in node_cache {
                                if let Some(prev_bt) = existing.get(&name) {
                                    // Fresh Arc: entries are replaced, never
                                    // mutated in place (#cone-delta increment 2
                                    // aliasing rule).
                                    if let Some((tighter, _)) =
                                        prev_bt.intersection_per_element(new_bt.as_ref())
                                    {
                                        existing.insert(name, Arc::new(tighter));
                                    }
                                } else {
                                    existing.insert(name, new_bt);
                                }
                            }
                        }
                    }
                }
                Err(ref e) if e.is_infeasible_domain() => {
                    tracing::debug!("Per-disjunct {full_idx} infeasible (empty): {e}");
                    return Ok(PerDisjunctChildOutcome::Verified(None));
                }
                Err(e) => {
                    let label = if is_active { "Active" } else { "Inactive" };
                    tracing::warn!("{label} per-disjunct {full_idx} propagation failed: {e}");
                    return Ok(PerDisjunctChildOutcome::PropagationFailure);
                }
            }
        }

        apply_per_disjunct_results(
            child,
            pruned_targets,
            &inherited_cached_las,
            active_bounds,
            active_cached_las,
            merged_node_bounds,
            ctx.thresholds,
            ctx.conjunctive,
            self.config.enable_cuts,
        )
    }
}

/// Apply per-disjunct evaluation results to the child domain.
///
/// Shared finalization for both batched and serial paths.
#[allow(clippy::too_many_arguments)]
fn apply_per_disjunct_results(
    child: &mut MultiObjectiveGraphBabDomain,
    pruned_targets: &PrunedMultiObjectiveTargets,
    inherited_cached_las: &[Option<Arc<crate::batched_domain::CachedLinearBounds>>],
    active_bounds: Vec<(f32, f32)>,
    active_cached_las: Vec<Option<crate::batched_domain::CachedLinearBounds>>,
    merged_node_bounds: Option<HashMap<String, Arc<BoundedTensor>>>,
    thresholds: &[f32],
    conjunctive: bool,
    enable_cuts: bool,
) -> Result<PerDisjunctChildOutcome> {
    let new_bounds =
        merge_pruned_objective_bounds(&child.objective_bounds, pruned_targets, active_bounds);
    // #bab-monotone-inherit (dark, NY_BAB_MONOTONE_INHERIT=1): monotone
    // parent-bound inheritance on the PER-DISJUNCT lane — the same merge the
    // batched lane has always applied. This is the single install point shared
    // by BOTH per-disjunct callers (`process_per_disjunct_batched` and
    // `process_per_disjunct_serial`), so gating it here covers the whole lane.
    //
    // SOUND: `child` reaches this module only through
    // `sequential::process_multi_objective_child`, whose child was produced by
    // `domain.with_constraint(..)` — ONE extra ReLU/Sign split appended to the
    // parent's history, with `input_bounds` only ever NARROWED. The child's
    // region is therefore a SUBSET of the parent's, so any valid parent lower
    // bound is a valid child lower bound: `max(parent_l, child_l)` is sound, and
    // symmetrically `min(parent_u, child_u)`.
    //
    // `child.objective_bounds` still holds the verbatim clone of the parent's
    // vector at this point: `with_constraint` copies it, nothing between there
    // and here writes it, and the batched→serial fallback in
    // `process_multi_objective_child_per_disjunct` only retries when the batched
    // arm errored BEFORE reaching this function. Gate absent/malformed =>
    // `new_bounds` passes through unchanged, byte-identical to today.
    let new_bounds = if bab_monotone_inherit_enabled() {
        inherit_parent_lower_only(&child.objective_bounds, new_bounds)
    } else {
        new_bounds
    };
    if let Some(node_cache) = merged_node_bounds {
        // #cone-delta increment 2: already Arc-shared — install by move.
        child.node_bounds =
            crate::beta_crown::domain::NodeBoundsMap::from_shared_hash_map(node_cache);
        // #cone-delta: `node_bounds` was replaced post-bounding — the delta
        // restarts empty. (No replacement ⇒ the delta keeps describing the
        // inherited map, so it is NOT cleared in that case.)
        child.delta_pre_nodes.clear();
    }
    if child.update_bounds(new_bounds, thresholds, false).is_err() {
        return Ok(PerDisjunctChildOutcome::NaNCorruption);
    }

    if is_domain_dropped(child, thresholds, conjunctive) {
        Ok(PerDisjunctChildOutcome::Dropped)
    } else if is_domain_verified(child, conjunctive) {
        let history = if enable_cuts {
            Some(Box::new(child.history.clone()))
        } else {
            None
        };
        Ok(PerDisjunctChildOutcome::Verified(history))
    } else {
        let merged_cached_las =
            merge_pruned_cached_las(inherited_cached_las, pruned_targets, active_cached_las);
        if child.set_shared_cached_las(merged_cached_las).is_err() {
            return Ok(PerDisjunctChildOutcome::NaNCorruption);
        }
        Ok(PerDisjunctChildOutcome::Enqueued)
    }
}

/// Check if a domain is verified according to the aggregation mode.
fn is_domain_verified(domain: &MultiObjectiveGraphBabDomain, conjunctive: bool) -> bool {
    if conjunctive {
        domain.any_verified()
    } else {
        domain.all_verified()
    }
}

/// Check if a domain should be dropped according to the aggregation mode.
fn is_domain_dropped(
    domain: &MultiObjectiveGraphBabDomain,
    thresholds: &[f32],
    conjunctive: bool,
) -> bool {
    // #violdrop: same gate as the batched and sequential lanes — a BaB child's
    // β-derived UPPER bound is not a certificate for its sub-region, so only the
    // ROOT may be abandoned on a `upper < threshold` reading. See
    // `shared::violation_drop_is_certified`.
    let violated = if conjunctive {
        domain.all_violated(thresholds, false)
    } else {
        domain.any_violated(thresholds, false)
    };
    if violated {
        super::shared::violdrop_site_probe("is_domain_dropped/per_disjunct", domain.depth);
    }
    violated && super::shared::violation_drop_is_certified(domain.depth)
}
