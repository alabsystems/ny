// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use ndarray::Array2;
use ny_core::{GemmEngine, Result};
use ny_tensor::BoundedTensor;
use tracing::trace;

use crate::beta_crown::config::InputClipType;
use crate::beta_crown::engine::graph::shared::state::GraphBabLifecycle;
use crate::beta_crown::engine::tensor_ext::BoundedTensorExt;
use crate::beta_crown::engine::BetaCrownVerifier;
use crate::bounds::{GraphAlphaState, LinearBounds};
use crate::GraphNetwork;

use super::super::parent_clip::clip_child_with_parent_linear;
use super::super::shared::{try_graph_spec_ibp_prescreen_bounds, GraphInputDomain};

/// Per-sub-domain warm-start bounding closure: given the child input box, an
/// optional child-local node-bounds override, and the parent's refined α slopes,
/// returns the child's objective bounds + linear bounds + the child's refined α
/// state (to seed its own children). Only invoked when
/// `input_split_alpha_iteration > 0`. See
/// `shared::compute_warm_start_crown_bounds_with_refined_alpha`.
pub(super) type WarmComputeBoundsResult = (f32, f32, Option<LinearBounds>, GraphAlphaState);

/// Boxed form of the warm-start bounding closure (see `WarmComputeBoundsResult`).
/// A trait object lets callers pass `None` without naming the closure type.
pub(super) type WarmComputeBoundsFn<'a> = dyn Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
        &GraphAlphaState,
    ) -> Result<WarmComputeBoundsResult>
    + 'a;

/// Screen a single child domain through clipping, IBP pre-screen, and
/// optional CROWN bounding. Part of #3882.
#[allow(clippy::too_many_arguments)]
pub(super) fn screen_single_child(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    mut child_input: BoundedTensor,
    shape: &[usize],
    objective: &[f32],
    threshold: f32,
    spec_matrix: &Array2<f32>,
    engine: Option<&dyn GemmEngine>,
    compute_bounds: &impl Fn(
        &BoundedTensor,
        Option<&HashMap<String, BoundedTensor>>,
    ) -> Result<(f32, f32, Option<LinearBounds>)>,
    warm_compute_bounds: Option<&WarmComputeBoundsFn<'_>>,
    parent_domain: &GraphInputDomain,
    lifecycle: &mut GraphBabLifecycle,
    domains_verified_by_ibp: &mut usize,
    domains_screened_by_crown: &mut usize,
) -> Result<Option<GraphInputDomain>> {
    let mut complete_clip_node_bounds: Option<HashMap<String, BoundedTensor>> = None;

    if verifier.config.enable_relaxed_clip {
        // #3870 Gap B: prefer parent linear bounds for clipping. The child box
        // is a subset of the parent, so the parent's CROWN linear coefficients
        // are valid over-approximations for clipping. This avoids a redundant
        // child CROWN backward pass just for clipping in reorder mode.
        // Reference: alpha-beta-CROWN input_split_and_repeat() → clip_domains().
        if parent_domain.linear_bounds.is_some() {
            let result = clip_child_with_parent_linear(
                verifier,
                graph,
                &child_input,
                shape,
                objective,
                threshold,
                parent_domain.linear_bounds.as_ref(),
                engine,
            )?;
            if result.verified {
                lifecycle.domains_verified += 1;
                return Ok(None);
            }
            child_input = result.bounds;
            complete_clip_node_bounds = result.complete_clip_node_bounds;
        } else {
            // Fallback: no parent linear — fresh CROWN pass for clipping.
            let clip_outcome = match verifier.config.input_clip_type {
                InputClipType::Relaxed => verifier.apply_relaxed_clipping_graph(
                    graph,
                    &child_input,
                    shape,
                    objective,
                    threshold,
                    engine,
                )?,
                InputClipType::Complete => {
                    let pre_clip_linear = match compute_bounds(&child_input, None) {
                        Ok((_lower, _upper, linear)) => linear,
                        Err(err) => {
                            trace!(
                                "graph complete clip: precomputed linear bounds unavailable, \
                                 falling back to direct clip path: {}",
                                err
                            );
                            None
                        }
                    };
                    let clip_outcome = match pre_clip_linear.as_ref() {
                        Some(linear_bounds) => verifier.complete_clip_with_precomputed_specs(
                            &child_input,
                            shape,
                            linear_bounds,
                            &[threshold],
                        )?,
                        None => verifier.apply_complete_clipping_graph(
                            graph,
                            &child_input,
                            shape,
                            objective,
                            threshold,
                            engine,
                        )?,
                    };
                    if !clip_outcome.verified {
                        if let Some(linear_bounds) = pre_clip_linear.as_ref() {
                            match super::super::super::clip_complete::build_graph_complete_clip_node_bounds(
                                graph,
                                &clip_outcome.bounds,
                                linear_bounds,
                                &[threshold],
                                verifier.config.verify_upper_bound,
                                verifier.config.clip_neuron_selection_ratio,
                                engine,
                            ) {
                                Ok(node_bounds) => complete_clip_node_bounds = node_bounds,
                                Err(err) => trace!(
                                    "graph complete clip: skipping hidden-layer tightening due to {}",
                                    err
                                ),
                            }
                        }
                    }
                    clip_outcome
                }
            };
            if clip_outcome.verified {
                lifecycle.domains_verified += 1;
                return Ok(None);
            }
            child_input = clip_outcome.bounds;
        }
    }

    if verifier.config.input_split_ibp_enhancement {
        if let Some(ibp_bounds) = try_graph_spec_ibp_prescreen_bounds(
            graph,
            &child_input,
            spec_matrix,
            engine,
            None,
            "single-objective reorder child",
        )? {
            if verifier.config.domain_is_verified(
                ibp_bounds.lower_scalar(),
                ibp_bounds.upper_scalar(),
                threshold,
            ) {
                *domains_verified_by_ibp += 1;
                lifecycle.domains_verified += 1;
                return Ok(None);
            }
        }
    }

    if verifier.config.reorder_bab {
        return Ok(Some(GraphInputDomain {
            input_bounds: Arc::new(child_input),
            lower_bound: parent_domain.lower_bound,
            upper_bound: parent_domain.upper_bound,
            depth: parent_domain.depth + 1,
            priority: parent_domain.priority,
            linear_bounds: None,
            needs_bounding: true,
            node_bounds_override: complete_clip_node_bounds.map(Arc::new),
            // Carry the parent's refined α slopes forward unchanged: reorder mode
            // defers bounding, so per-domain refinement is not applied here.
            inherited_alpha_state: parent_domain.inherited_alpha_state.clone(),
        }));
    }

    *domains_screened_by_crown += 1;
    // Per-sub-domain α refinement (alpha-beta-CROWN input_split/bounding.py:90-179):
    // when enabled AND the parent carries refined α slopes, warm-start from them
    // and re-optimize for the child's tighter box, then save the refined α onto the
    // child so its own children warm-start from it. Otherwise (the default), fall
    // through to the historical single frozen-alpha pass — byte-identical behavior.
    let (obj_l, obj_u, linear, child_alpha) = match (
        warm_compute_bounds,
        parent_domain.inherited_alpha_state.as_deref(),
    ) {
        (Some(warm), Some(parent_alpha)) => {
            let (obj_l, obj_u, linear, refined_alpha) = warm(
                &child_input,
                complete_clip_node_bounds.as_ref(),
                parent_alpha,
            )?;
            (obj_l, obj_u, linear, Some(Arc::new(refined_alpha)))
        }
        _ => {
            let (obj_l, obj_u, linear) =
                compute_bounds(&child_input, complete_clip_node_bounds.as_ref())?;
            // Frozen path: pass the parent's α state (if any) through unchanged.
            (
                obj_l,
                obj_u,
                linear,
                parent_domain.inherited_alpha_state.clone(),
            )
        }
    };
    // Monotonicity guard: child lower bound cannot regress below parent.
    // Reference: alpha-beta-CROWN input_split/bounding.py:154
    let obj_l = obj_l.max(parent_domain.lower_bound);
    let child_priority = verifier.config.domain_priority(obj_l, obj_u)?;
    Ok(Some(GraphInputDomain {
        input_bounds: Arc::new(child_input),
        lower_bound: obj_l,
        upper_bound: obj_u,
        depth: parent_domain.depth + 1,
        priority: child_priority,
        linear_bounds: linear,
        needs_bounding: false,
        node_bounds_override: None,
        inherited_alpha_state: child_alpha,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphNetwork, GraphNode, Layer, LinearBounds, LinearLayer};
    use ndarray::{arr1, arr2};

    fn complete_reorder_test_graph() -> GraphNetwork {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input(
            "hidden",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("hidden linear")),
        ));
        graph.add_node(GraphNode::new(
            "out",
            Layer::Linear(LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("out linear")),
            vec!["hidden".to_string()],
        ));
        graph.set_output("out");
        graph
    }

    #[test]
    fn test_screen_single_child_reorder_complete_defers_with_node_bounds_override() {
        let verifier = BetaCrownVerifier::new(crate::beta_crown::config::BetaCrownConfig {
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Complete,
            reorder_bab: true,
            input_split_ibp_enhancement: false,
            ..Default::default()
        });
        let graph = complete_reorder_test_graph();
        let child_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite child bounds");
        let parent_domain = GraphInputDomain {
            input_bounds: Arc::new(child_input.clone()),
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            priority: 1.0,
            linear_bounds: None,
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: None,
        };
        let linear_bounds = LinearBounds {
            lower_a: arr2(&[[1.0_f32]]),
            lower_b: arr1(&[0.0_f32]),
            upper_a: arr2(&[[1.0_f32]]),
            upper_b: arr1(&[0.0_f32]),
            lower_a_err: None,
            upper_a_err: None,
        };
        let compute_bounds = |_input: &BoundedTensor,
                              _node_bounds: Option<&HashMap<String, BoundedTensor>>|
         -> Result<(f32, f32, Option<LinearBounds>)> {
            Ok((-1.0, 1.0, Some(linear_bounds.clone())))
        };
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_ibp = 0usize;
        let mut domains_screened_by_crown = 0usize;

        let child = screen_single_child(
            &verifier,
            &graph,
            child_input,
            &[1],
            &[1.0_f32],
            0.2,
            &arr2(&[[1.0_f32]]),
            None,
            &compute_bounds,
            None,
            &parent_domain,
            &mut lifecycle,
            &mut domains_verified_by_ibp,
            &mut domains_screened_by_crown,
        )
        .expect("complete reorder child should not error")
        .expect("child should remain unresolved");

        assert!(
            child.needs_bounding,
            "reorder_bab must still defer child bounding in complete mode"
        );
        assert!(
            child.node_bounds_override.is_some(),
            "complete clipping should carry its child-local node bounds into the deferred pass"
        );
        assert_eq!(
            domains_screened_by_crown, 0,
            "deferred complete child should not eagerly consume a CROWN screening pass"
        );
    }

    /// #3870 Gap B regression: when parent linear bounds are present, the
    /// reorder clip path MUST reuse them via `clip_child_with_parent_linear`
    /// instead of running a fresh child CROWN pass. If `compute_bounds` is
    /// called, this test panics — proving the clip step consumed parent data.
    ///
    /// Reference: alpha-beta-CROWN `input_split_and_repeat()` duplicates parent
    /// `lA`/`lbias` into split children; `clip_domains()` consumes them.
    /// Source: `batch_branch_and_bound.py:151-169`, `clip.py:174-232`.
    #[test]
    fn test_screen_single_child_reorder_reuses_parent_linear_for_clip_3870() {
        let verifier = BetaCrownVerifier::new(crate::beta_crown::config::BetaCrownConfig {
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Relaxed,
            reorder_bab: true,
            input_split_ibp_enhancement: false,
            ..Default::default()
        });
        let graph = complete_reorder_test_graph();
        let child_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite child bounds");

        // Parent domain WITH linear bounds (the reuse source).
        let parent_linear = LinearBounds {
            lower_a: arr2(&[[1.0_f32]]),
            lower_b: arr1(&[0.0_f32]),
            upper_a: arr2(&[[1.0_f32]]),
            upper_b: arr1(&[0.0_f32]),
            lower_a_err: None,
            upper_a_err: None,
        };
        let parent_domain = GraphInputDomain {
            input_bounds: Arc::new(child_input.clone()),
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            priority: 1.0,
            linear_bounds: Some(parent_linear),
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: None,
        };

        // compute_bounds panics if called — proves parent linear was reused.
        let compute_bounds = |_input: &BoundedTensor,
                              _node_bounds: Option<&HashMap<String, BoundedTensor>>|
         -> Result<(f32, f32, Option<LinearBounds>)> {
            panic!("compute_bounds must NOT be called when parent linear bounds are available")
        };
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_ibp = 0usize;
        let mut domains_screened_by_crown = 0usize;

        let child = screen_single_child(
            &verifier,
            &graph,
            child_input,
            &[1],
            &[1.0_f32],
            0.2,
            &arr2(&[[1.0_f32]]),
            None,
            &compute_bounds,
            None,
            &parent_domain,
            &mut lifecycle,
            &mut domains_verified_by_ibp,
            &mut domains_screened_by_crown,
        )
        .expect("parent-linear clip should not error")
        .expect("child should remain unresolved in reorder mode");

        assert!(
            child.needs_bounding,
            "reorder_bab child must be deferred with needs_bounding=true"
        );
        assert!(
            child.linear_bounds.is_none(),
            "reorder child must NOT carry parent linear as its own bound cache"
        );
        assert_eq!(
            domains_screened_by_crown, 0,
            "parent-linear clip must not count as a CROWN screening pass"
        );
    }

    #[test]
    fn test_screen_single_child_reorder_complete_reuses_parent_linear_and_keeps_override_3870() {
        let verifier = BetaCrownVerifier::new(crate::beta_crown::config::BetaCrownConfig {
            enable_relaxed_clip: true,
            input_clip_type: InputClipType::Complete,
            reorder_bab: true,
            input_split_ibp_enhancement: false,
            ..Default::default()
        });
        let graph = complete_reorder_test_graph();
        let child_input =
            BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
                .expect("finite child bounds");

        let parent_linear = LinearBounds {
            lower_a: arr2(&[[1.0_f32]]),
            lower_b: arr1(&[0.0_f32]),
            upper_a: arr2(&[[1.0_f32]]),
            upper_b: arr1(&[0.0_f32]),
            lower_a_err: None,
            upper_a_err: None,
        };
        let parent_domain = GraphInputDomain {
            input_bounds: Arc::new(child_input.clone()),
            lower_bound: -1.0,
            upper_bound: 1.0,
            depth: 0,
            priority: 1.0,
            linear_bounds: Some(parent_linear),
            needs_bounding: false,
            node_bounds_override: None,
            inherited_alpha_state: None,
        };

        let compute_bounds = |_input: &BoundedTensor,
                              _node_bounds: Option<&HashMap<String, BoundedTensor>>|
         -> Result<(f32, f32, Option<LinearBounds>)> {
            panic!(
                "compute_bounds must NOT be called when complete clipping can reuse parent linear bounds"
            )
        };
        let mut lifecycle = GraphBabLifecycle::new(std::time::Instant::now());
        let mut domains_verified_by_ibp = 0usize;
        let mut domains_screened_by_crown = 0usize;

        let child = screen_single_child(
            &verifier,
            &graph,
            child_input,
            &[1],
            &[1.0_f32],
            0.2,
            &arr2(&[[1.0_f32]]),
            None,
            &compute_bounds,
            None,
            &parent_domain,
            &mut lifecycle,
            &mut domains_verified_by_ibp,
            &mut domains_screened_by_crown,
        )
        .expect("parent-linear complete clip should not error")
        .expect("child should remain unresolved in reorder mode");

        assert!(child.needs_bounding);
        assert!(
            child.node_bounds_override.is_some(),
            "complete clipping should keep the clipped child node bounds for the deferred pass"
        );
        assert!(
            child.linear_bounds.is_none(),
            "reorder child must still defer its real child-domain linear bounds"
        );
        assert_eq!(
            domains_screened_by_crown, 0,
            "parent-linear complete clip must not count as a CROWN screening pass"
        );
    }
}
