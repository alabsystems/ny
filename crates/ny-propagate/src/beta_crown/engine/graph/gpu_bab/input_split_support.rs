// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

use std::sync::Arc;

use ndarray::Axis;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::{trace, warn};

use crate::batched_domain::ProcessedDomains;
use crate::beta_crown::config::InputClipType;
use crate::beta_crown::engine::graph::input_split::parent_clip::clip_child_with_parent_linear;
use crate::beta_crown::engine::graph::input_split::shared::{
    compute_crown_or_ibp_bounds_with_node_bounds, graph_spec_ibp_fallback,
};
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::LinearBounds;
use crate::GraphNetwork;

use super::check::{check_domain_bounds, BabLoopState, DomainCheckResult};
use super::init::{cache_input_split_linear_bounds, InputSplitBootstrap};
pub(super) use super::parent_contexts::build_parent_contexts;

pub(super) enum ChildDomainAction {
    Skip,
    Queue(Box<ProcessedDomains>),
    Violation,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn screen_child_domain(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    objective: &[f32],
    bootstrap: &InputSplitBootstrap,
    threshold: f32,
    engine: Option<&dyn GemmEngine>,
    picked_idx: usize,
    split_dim: usize,
    parent_lower_bound: f32,
    parent_upper_bound: f32,
    parent_linear_bounds: Option<&LinearBounds>,
    state: &mut BabLoopState,
    mut child_processed: ProcessedDomains,
) -> Result<ChildDomainAction> {
    let mut child_node_bounds_override = None;
    let child_input_lower = child_processed
        .input_lowers
        .index_axis(Axis(0), 0)
        .to_owned()
        .into_dyn();
    let child_input_upper = child_processed
        .input_uppers
        .index_axis(Axis(0), 0)
        .to_owned()
        .into_dyn();
    let mut child_input = match BoundedTensor::new(child_input_lower, child_input_upper) {
        Ok(bt) => bt,
        Err(err) => {
            warn!(
                picked_idx,
                split_dim,
                error = %err,
                "DomainList BaB input split: invalid child input bounds"
            );
            state.unresolved_due_to_propagation_failure = true;
            return Ok(ChildDomainAction::Skip);
        }
    };
    let child_depth = child_processed
        .metadata
        .first()
        .map(|m| m.depth())
        .unwrap_or(0);
    state.max_depth_reached = state.max_depth_reached.max(child_depth);

    if verifier.config.enable_relaxed_clip {
        let child_shape = child_input.lower().shape().to_vec();
        if parent_linear_bounds.is_some() {
            // #3870 Gap B: prefer parent linear bounds for child clipping.
            // The child box is a subset of the parent, so the parent's CROWN
            // linear coefficients are valid over-approximations for clipping.
            let result = clip_child_with_parent_linear(
                verifier,
                graph,
                &child_input,
                &child_shape,
                objective,
                threshold,
                parent_linear_bounds,
                engine,
            )?;
            if result.verified {
                state.domains_verified += 1;
                return Ok(ChildDomainAction::Skip);
            }
            child_input = result.bounds;
            child_node_bounds_override = result.complete_clip_node_bounds;
        } else {
            let clip_outcome = match verifier.config.input_clip_type {
                InputClipType::Relaxed => verifier.apply_relaxed_clipping_graph(
                    graph,
                    &child_input,
                    &child_shape,
                    objective,
                    threshold,
                    engine,
                )?,
                InputClipType::Complete => {
                    let pre_clip_linear = match compute_crown_or_ibp_bounds_with_node_bounds(
                        graph,
                        &child_input,
                        &bootstrap.spec_matrix,
                        engine,
                        bootstrap.fixed_node_bounds.as_ref(),
                        None,
                        bootstrap.root_alpha_state.as_ref(),
                        bootstrap.mul_binary_alphas.as_ref(),
                        bootstrap.deadline,
                        verifier.config.crown_backward_layers,
                        verifier.config.input_split_ibp_enhancement,
                    ) {
                        Ok((_bounds, linear)) => linear,
                        Err(err) => {
                            trace!(
                                picked_idx,
                                split_dim,
                                error = %err,
                                "DomainList BaB input split: precomputed linear bounds unavailable, falling back to direct complete clip"
                            );
                            None
                        }
                    };
                    let clip_outcome = match pre_clip_linear.as_ref() {
                        Some(linear_bounds) => verifier.complete_clip_with_precomputed_specs(
                            &child_input,
                            &child_shape,
                            linear_bounds,
                            &[threshold],
                        )?,
                        None => verifier.apply_complete_clipping_graph(
                            graph,
                            &child_input,
                            &child_shape,
                            objective,
                            threshold,
                            engine,
                        )?,
                    };
                    if !clip_outcome.verified {
                        if let Some(linear_bounds) = pre_clip_linear.as_ref() {
                            match crate::beta_crown::engine::graph::clip_complete::build_graph_complete_clip_node_bounds(
                                graph,
                                &clip_outcome.bounds,
                                linear_bounds,
                                &[threshold],
                                verifier.config.verify_upper_bound,
                                verifier.config.clip_neuron_selection_ratio,
                                engine,
                            ) {
                                Ok(node_bounds) => child_node_bounds_override = node_bounds,
                                Err(err) => trace!(
                                    picked_idx,
                                    split_dim,
                                    error = %err,
                                    "DomainList BaB input split: skipping hidden-layer tightening after fallback complete clip"
                                ),
                            }
                        }
                    }
                    clip_outcome
                }
            };
            if clip_outcome.verified {
                state.domains_verified += 1;
                return Ok(ChildDomainAction::Skip);
            }
            child_input = clip_outcome.bounds;
        }
        child_processed.input_lowers = child_input.lower().clone().insert_axis(Axis(0));
        child_processed.input_uppers = child_input.upper().clone().insert_axis(Axis(0));
    }

    if verifier.config.input_split_ibp_enhancement {
        let (ibp_bounds, _) =
            graph_spec_ibp_fallback(graph, &child_input, &bootstrap.spec_matrix, engine, None)?;
        if verifier.config.domain_is_verified(
            ibp_bounds.lower_scalar(),
            ibp_bounds.upper_scalar(),
            threshold,
        ) {
            state.domains_verified += 1;
            return Ok(ChildDomainAction::Skip);
        }
    }

    let Some(meta) = child_processed.metadata.first_mut() else {
        return Err(NyError::InvalidSpec(
            "DomainList BaB input split: child metadata missing".to_string(),
        ));
    };

    if verifier.config.reorder_bab {
        // #3870: When complete clipping's Lagrangian path returns no active
        // constraints (e.g., the spec constraint is fully covered at the input
        // level for a simple graph), the deferred CROWN pass still benefits from
        // carrying IBP bounds for the clipped child input. These bounds are at
        // least as tight as what the deferred pass would recompute, and they
        // match the alpha-beta-CROWN contract where clipped domains carry their
        // node bounds through the branching queue.
        // Source: batch_branch_and_bound.py:input_split_and_repeat → clip_domains.
        if child_node_bounds_override.is_none()
            && matches!(verifier.config.input_clip_type, InputClipType::Complete)
        {
            child_node_bounds_override =
                Some(graph.collect_node_bounds_with_engine(&child_input, engine)?);
        }
        child_processed.global_lbs = vec![parent_lower_bound];
        child_processed.global_ubs = vec![parent_upper_bound];
        meta.update_bounds(parent_lower_bound, parent_upper_bound)?;
        meta.cached_la = None;
        meta.set_node_bounds_override(child_node_bounds_override.map(Arc::new))?;
        meta.set_needs_bounding(true);
        return Ok(ChildDomainAction::Queue(Box::new(child_processed)));
    }

    let (child_bounds, child_linear_bounds) = match compute_crown_or_ibp_bounds_with_node_bounds(
        graph,
        &child_input,
        &bootstrap.spec_matrix,
        engine,
        bootstrap.fixed_node_bounds.as_ref(),
        child_node_bounds_override.as_ref(),
        bootstrap.root_alpha_state.as_ref(),
        bootstrap.mul_binary_alphas.as_ref(),
        bootstrap.deadline,
        verifier.config.crown_backward_layers,
        verifier.config.input_split_ibp_enhancement,
    ) {
        Ok(result) => result,
        Err(err) => {
            warn!(
                picked_idx,
                split_dim,
                error = %err,
                "DomainList BaB input split: CROWN/IBP child bound computation failed"
            );
            state.unresolved_due_to_propagation_failure = true;
            return Ok(ChildDomainAction::Skip);
        }
    };
    let child_lower_bound = child_bounds.lower_scalar();
    let child_upper_bound = child_bounds.upper_scalar();
    if !child_lower_bound.is_finite() || !child_upper_bound.is_finite() {
        warn!(
            picked_idx,
            split_dim,
            lower = child_lower_bound,
            upper = child_upper_bound,
            "DomainList BaB input split: non-finite child bounds after CROWN/IBP"
        );
        state.unresolved_due_to_propagation_failure = true;
        return Ok(ChildDomainAction::Skip);
    }

    child_processed.input_lowers = child_input.lower().clone().insert_axis(Axis(0));
    child_processed.input_uppers = child_input.upper().clone().insert_axis(Axis(0));
    child_processed.global_lbs = vec![child_lower_bound];
    child_processed.global_ubs = vec![child_upper_bound];
    meta.update_bounds(child_lower_bound, child_upper_bound)?;
    meta.cached_la = child_linear_bounds
        .as_ref()
        .map(|lb| Arc::new(cache_input_split_linear_bounds(lb)));
    meta.set_node_bounds_override(None)?;
    meta.set_needs_bounding(false);

    match check_domain_bounds(
        child_lower_bound,
        child_upper_bound,
        threshold,
        verifier.config.verify_upper_bound,
    ) {
        DomainCheckResult::Verified => {
            state.domains_verified += 1;
            Ok(ChildDomainAction::Skip)
        }
        DomainCheckResult::Violation => Ok(ChildDomainAction::Violation),
        DomainCheckResult::Undecided => {
            if child_depth >= verifier.config.max_depth {
                state.unresolved_due_to_depth = true;
                Ok(ChildDomainAction::Skip)
            } else {
                Ok(ChildDomainAction::Queue(Box::new(child_processed)))
            }
        }
    }
}
