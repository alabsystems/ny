// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use ndarray::Array2;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::trace;

use crate::beta_crown::config::InputClipType;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::GraphNetwork;

use super::super::shared::{
    extract_obj_bounds, multi_obj_domain_priority, multi_obj_domain_verified,
    try_graph_spec_ibp_prescreen_bounds, MultiObjBounds, MultiObjInputDomain,
};

/// Per-sub-domain warm-start bounding closure for the multi-objective loop:
/// given the child input box, an optional child-local node-bounds override, and
/// the parent's refined α slopes, returns the per-objective bounds + linear
/// bounds + the child's refined α state (to seed its own children). Only
/// invoked when `input_split_alpha_iteration > 0`. See
/// `shared::compute_warm_start_crown_bounds_with_refined_alpha`.
pub(super) type WarmMultiObjBoundsResult = (Vec<(f32, f32)>, Option<LinearBounds>, GraphAlphaState);

/// Boxed form of the multi-objective warm-start bounding closure (see
/// `WarmMultiObjBoundsResult`). A trait object lets callers pass `None`
/// without naming the closure type.
pub(super) type WarmMultiObjComputeBoundsFn<'a> = dyn Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
        &GraphAlphaState,
    ) -> Result<WarmMultiObjBoundsResult>
    + 'a;

/// Screen a multi-objective child domain through clipping and bounding. Part of #3882.
#[allow(clippy::too_many_arguments)]
pub(super) fn screen_multi_obj_child(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    mut child_input: BoundedTensor,
    spec_matrix: &Array2<f32>,
    thresholds: &[f32],
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &impl Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<MultiObjBounds>,
    warm_compute_bounds: Option<&WarmMultiObjComputeBoundsFn<'_>>,
    parent_domain: &MultiObjInputDomain,
    queue: &mut BinaryHeap<MultiObjInputDomain>,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_clip: &mut usize,
) -> Result<()> {
    let mut complete_clip_node_bounds: Option<HashMap<String, BoundedTensor>> = None;

    if verifier.config.reorder_bab {
        if verifier.config.enable_relaxed_clip {
            if let Some(linear_bounds) = parent_domain.linear_bounds.as_ref() {
                let shape = child_input.lower().shape().to_vec();
                match verifier.config.input_clip_type {
                    InputClipType::Relaxed => {
                        for (i, &threshold) in thresholds.iter().enumerate() {
                            let clip_outcome = verifier.clip_with_precomputed_linear(
                                &child_input,
                                &shape,
                                linear_bounds,
                                i,
                                threshold,
                            )?;
                            if clip_outcome.verified {
                                *domains_verified_by_clip += 1;
                                lifecycle.domains_verified += 1;
                                return Ok(());
                            }
                            child_input = clip_outcome.bounds;
                        }
                    }
                    InputClipType::Complete => {
                        let clip_outcome = verifier.complete_clip_with_precomputed_specs(
                            &child_input,
                            &shape,
                            linear_bounds,
                            thresholds,
                        )?;
                        if clip_outcome.verified {
                            *domains_verified_by_clip += 1;
                            lifecycle.domains_verified += 1;
                            return Ok(());
                        }
                        child_input = clip_outcome.bounds;
                        complete_clip_node_bounds =
                            match super::super::super::clip_complete::build_graph_complete_clip_node_bounds(
                                graph,
                                &child_input,
                                linear_bounds,
                                thresholds,
                                verifier.config.verify_upper_bound,
                                verifier.config.clip_neuron_selection_ratio,
                                engine,
                            ) {
                                Ok(node_bounds) => node_bounds,
                                Err(err) => {
                                    trace!(
                                        "graph complete clip: skipping hidden-layer tightening due to {}",
                                        err
                                    );
                                    None
                                }
                            };
                    }
                }
            }
        }

        if verifier.config.input_split_ibp_enhancement {
            if let Some(ibp_bounds) = try_graph_spec_ibp_prescreen_bounds(
                graph,
                &child_input,
                spec_matrix,
                engine,
                None,
                "multi-objective reorder child",
            )? {
                let ibp_obj_bounds = extract_obj_bounds(&ibp_bounds, thresholds.len())?;
                if multi_obj_domain_verified(&ibp_obj_bounds, thresholds) {
                    lifecycle.domains_verified += 1;
                    return Ok(());
                }
            }
        }

        queue.push(MultiObjInputDomain {
            input_bounds: Arc::new(child_input),
            obj_bounds: parent_domain.obj_bounds.clone(),
            linear_bounds: None,
            depth: parent_domain.depth + 1,
            priority: parent_domain.priority,
            needs_bounding: true,
            node_bounds_override: complete_clip_node_bounds.map(Arc::new),
            // Carry the parent's refined α slopes forward unchanged: reorder mode
            // defers bounding, so per-domain refinement is not applied here.
            inherited_alpha_state: parent_domain.inherited_alpha_state.clone(),
        });
        return Ok(());
    }

    // Per-sub-domain α refinement (alpha-beta-CROWN input_split/bounding.py:90-179):
    // when enabled AND the parent carries refined α slopes, warm-start from them
    // and re-optimize for the child's tighter box, then save the refined α onto the
    // child so its own children warm-start from it. Otherwise (the default), fall
    // through to the historical single frozen-alpha pass — byte-identical behavior.
    let (mut obj_bounds, mut linear, mut child_alpha) = match (
        warm_compute_bounds,
        parent_domain.inherited_alpha_state.as_deref(),
    ) {
        (Some(warm), Some(parent_alpha)) => {
            let (obj_bounds, linear, refined_alpha) = warm(&child_input, None, parent_alpha)?;
            (obj_bounds, linear, Some(Arc::new(refined_alpha)))
        }
        _ => {
            let (obj_bounds, linear) = compute_bounds(&child_input, None)?;
            // Frozen path: pass the parent's α state (if any) through unchanged.
            (
                obj_bounds,
                linear,
                parent_domain.inherited_alpha_state.clone(),
            )
        }
    };
    if multi_obj_domain_verified(&obj_bounds, thresholds) {
        lifecycle.domains_verified += 1;
        return Ok(());
    }

    // Relaxed mode keeps the single-pass clip-and-bound optimization
    // (#1579 Phase 4). Complete mode recomputes once after clipping so the
    // spec pass can consume the tightened child-local node cache.
    if verifier.config.enable_relaxed_clip {
        if let Some(ref linear_bounds) = linear {
            let shape = child_input.lower().shape().to_vec();
            match verifier.config.input_clip_type {
                InputClipType::Relaxed => {
                    for (i, &threshold) in thresholds.iter().enumerate() {
                        let clip_outcome = verifier.clip_with_precomputed_linear(
                            &child_input,
                            &shape,
                            linear_bounds,
                            i,
                            threshold,
                        )?;
                        if clip_outcome.verified {
                            *domains_verified_by_clip += 1;
                            lifecycle.domains_verified += 1;
                            return Ok(());
                        }
                        child_input = clip_outcome.bounds;
                    }
                }
                InputClipType::Complete => {
                    let clip_outcome = verifier.complete_clip_with_precomputed_specs(
                        &child_input,
                        &shape,
                        linear_bounds,
                        thresholds,
                    )?;
                    if clip_outcome.verified {
                        *domains_verified_by_clip += 1;
                        lifecycle.domains_verified += 1;
                        return Ok(());
                    }
                    child_input = clip_outcome.bounds;

                    let complete_clip_node_bounds =
                        match super::super::super::clip_complete::build_graph_complete_clip_node_bounds(
                            graph,
                            &child_input,
                            linear_bounds,
                            thresholds,
                            verifier.config.verify_upper_bound,
                            verifier.config.clip_neuron_selection_ratio,
                            engine,
                        ) {
                            Ok(node_bounds) => node_bounds,
                            Err(err) => {
                                trace!(
                                    "graph complete clip: skipping hidden-layer tightening due to {}",
                                    err
                                );
                                None
                            }
                        };
                    // Complete-clip recompute: reuse the warm path when enabled,
                    // seeding from this child's already-refined α (sound: any
                    // α in [0,1] warm-start yields a valid CROWN bound; failure
                    // propagates just like the frozen recompute).
                    (obj_bounds, linear) = match (warm_compute_bounds, child_alpha.as_deref()) {
                        (Some(warm), Some(seed_alpha)) => {
                            let (obj_bounds, linear, refined_alpha) =
                                warm(&child_input, complete_clip_node_bounds.as_ref(), seed_alpha)?;
                            child_alpha = Some(Arc::new(refined_alpha));
                            (obj_bounds, linear)
                        }
                        _ => compute_bounds(&child_input, complete_clip_node_bounds.as_ref())?,
                    };
                    if multi_obj_domain_verified(&obj_bounds, thresholds) {
                        *domains_verified_by_clip += 1;
                        lifecycle.domains_verified += 1;
                        return Ok(());
                    }
                }
            }
        }
    }

    // Monotonicity guard: per-spec lower bound cannot regress below parent.
    // Reference: alpha-beta-CROWN input_split/bounding.py:154
    let obj_bounds: Vec<(f32, f32)> = obj_bounds
        .into_iter()
        .zip(parent_domain.obj_bounds.iter())
        .map(|((new_l, new_u), &(old_l, _))| (new_l.max(old_l), new_u))
        .collect();
    let priority = multi_obj_domain_priority(&obj_bounds, thresholds);
    queue.push(MultiObjInputDomain {
        input_bounds: Arc::new(child_input),
        obj_bounds,
        linear_bounds: linear,
        depth: parent_domain.depth + 1,
        priority,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: child_alpha,
    });
    Ok(())
}

#[cfg(test)]
mod tests;
