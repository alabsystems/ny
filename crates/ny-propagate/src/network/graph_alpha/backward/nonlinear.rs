// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::CrownBounds;
use crate::bounds::{GraphAlphaCrownIntermediate, GraphAlphaState, LinearBounds};
use crate::invprop::InvpropState;
use crate::layers::{Layer, ReLULayer};
use crate::network::core::{crown_backward_step_patches, CrownStepResult, GraphNetwork, GraphNode};
use crate::network::CrownMergeAccumulator;

use ndarray::Array1;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::collections::HashMap;
use std::time::Instant;
use tracing::debug;

// `ReturnBounds` keeps `BoundedTensor` inline: the value is constructed and
// consumed within one backward step, so a 232-byte move beats a heap
// allocation per nonlinear node on this hot path.
#[allow(clippy::large_enum_variant)]
pub(super) enum NonlinearNodeResult {
    /// Boxed: `CrownBounds` dwarfs the other variants (clippy `large_enum_variant`).
    NotHandled(Box<CrownBounds>),
    Continue,
    ReturnBounds(BoundedTensor),
}

pub(super) struct DagAlphaNodeContext<'a> {
    pub(super) input: &'a BoundedTensor,
    pub(super) relu_name_to_idx: &'a HashMap<String, usize>,
    pub(super) alpha_state: &'a GraphAlphaState,
    pub(super) invprop_state: Option<&'a InvpropState>,
    pub(super) gradients: &'a mut [Array1<f32>],
    pub(super) gradients_upper: &'a mut [Array1<f32>],
    pub(super) track_gradients: bool,
    pub(super) node_crown_bounds: &'a mut CrownMergeAccumulator,
    pub(super) intermediate: Option<&'a mut GraphAlphaCrownIntermediate>,
    pub(super) output_dim: usize,
    pub(super) input_dim: usize,
    pub(super) input_accumulated: &'a mut bool,
    pub(super) engine: Option<&'a dyn GemmEngine>,
    pub(super) deadline: Option<Instant>,
}

pub(super) fn retry_monotone_shape_mismatch_with_fixed_slope<FAlpha, FFixed>(
    node_name: &str,
    layer_type: &str,
    node_lb: &LinearBounds,
    pre_activation: &BoundedTensor,
    alpha_propagate: FAlpha,
    fixed_propagate: FFixed,
) -> Result<LinearBounds>
where
    FAlpha: FnOnce(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
    FFixed: FnOnce(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
{
    match alpha_propagate(node_lb, pre_activation) {
        Ok(bounds) => Ok(bounds),
        Err(NyError::ShapeMismatch { .. }) => {
            debug!(
                "DAG α-CROWN: {layer_type} {node_name} alpha propagation ShapeMismatch, retrying local fixed-slope CROWN"
            );
            fixed_propagate(node_lb, pre_activation)
        }
        Err(err) => Err(err),
    }
}

pub(super) fn handle_nonlinear_node(
    network: &GraphNetwork,
    node_name: &str,
    node: &GraphNode,
    first_input: &str,
    node_cb: CrownBounds,
    pre_activation: &BoundedTensor,
    mut context: DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    match &node.layer {
        Layer::ReLU(relu) => handle_relu_node(
            network,
            node_name,
            node,
            first_input,
            node_cb,
            pre_activation,
            relu,
            &mut context,
        ),
        Layer::Sigmoid(sigmoid) => {
            let Some(alpha) = context.alpha_state.monotone_s_shaped_alpha(node_name) else {
                return Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb)));
            };
            handle_monotone_node(
                network,
                node_name,
                "Sigmoid",
                first_input,
                node_cb,
                pre_activation,
                |node_lb, pre_activation| {
                    sigmoid.propagate_linear_with_alpha(node_lb, pre_activation, alpha)
                },
                |node_lb, pre_activation| {
                    sigmoid.propagate_linear_with_bounds(node_lb, pre_activation)
                },
                &mut context,
            )
        }
        Layer::Tanh(tanh) => {
            let Some(alpha) = context.alpha_state.monotone_s_shaped_alpha(node_name) else {
                return Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb)));
            };
            handle_monotone_node(
                network,
                node_name,
                "Tanh",
                first_input,
                node_cb,
                pre_activation,
                |node_lb, pre_activation| {
                    tanh.propagate_linear_with_alpha(node_lb, pre_activation, alpha)
                },
                |node_lb, pre_activation| {
                    tanh.propagate_linear_with_bounds(node_lb, pre_activation)
                },
                &mut context,
            )
        }
        Layer::Sqrt(sqrt) => {
            let Some(alpha) = context.alpha_state.sqrt_alpha(node_name) else {
                return Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb)));
            };
            handle_monotone_node(
                network,
                node_name,
                "Sqrt",
                first_input,
                node_cb,
                pre_activation,
                |node_lb, pre_act| {
                    sqrt.propagate_linear_with_alpha(
                        node_lb,
                        pre_act,
                        &alpha.lower_path_mid,
                        Some(&alpha.upper_path_mid),
                    )
                },
                |node_lb, pre_act| sqrt.propagate_linear_with_bounds(node_lb, pre_act),
                &mut context,
            )
        }
        Layer::Reciprocal(reciprocal) => {
            let Some(alpha) = context.alpha_state.reciprocal_alpha(node_name) else {
                return Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb)));
            };
            handle_monotone_node(
                network,
                node_name,
                "Reciprocal",
                first_input,
                node_cb,
                pre_activation,
                |node_lb, pre_act| {
                    reciprocal.propagate_linear_with_alpha(
                        node_lb,
                        pre_act,
                        &alpha.lower_path_mid,
                        Some(&alpha.upper_path_mid),
                    )
                },
                |node_lb, pre_act| reciprocal.propagate_linear_with_bounds(node_lb, pre_act),
                &mut context,
            )
        }
        _ => Ok(NonlinearNodeResult::NotHandled(Box::new(node_cb))),
    }
}

// DagAlphaNodeContext already bundles 14 fields; the remaining args are
// per-call inputs that differ across the match arms in handle_nonlinear_node.
#[allow(clippy::too_many_arguments)]
fn handle_relu_node(
    network: &GraphNetwork,
    node_name: &str,
    node: &GraphNode,
    first_input: &str,
    mut node_cb: CrownBounds,
    pre_activation: &BoundedTensor,
    relu: &ReLULayer,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    if matches!(&node_cb, CrownBounds::Patches(_)) && node.inputs.len() == 1 {
        if let Some(&relu_idx) = context.relu_name_to_idx.get(node_name) {
            if let Some(alpha) = context.alpha_state.alpha(node_name) {
                if let CrownBounds::Patches(ref pb) = node_cb {
                    // NOTE(#3782): Only lower-path gradient is captured here.
                    // Patches-mode ReLU is single-alpha (one optimizable lower
                    // slope + fixed upper chord). A dual-alpha patches path
                    // would need `propagate_patches_with_alpha` to return a
                    // separate `grad_upper`, which requires a relaxation redesign.
                    // #4404: expand channel-only alpha to full spatial before use.
                    let alpha_expanded = context.alpha_state.expand_alpha(node_name, alpha);
                    let propagated = if context.track_gradients {
                        relu.propagate_patches_with_alpha(pb, pre_activation, &alpha_expanded)
                            .map(|(bounds, gradient)| (bounds, Some(gradient)))
                    } else {
                        relu.propagate_patches_with_alpha_bound_only(
                            pb,
                            pre_activation,
                            &alpha_expanded,
                        )
                        .map(|bounds| (bounds, None))
                    };
                    match propagated {
                        Ok((new_cb, grad)) => {
                            // #4404: reduce gradient back to per-channel if channel-only.
                            if let Some(grad) = grad {
                                context.gradients[relu_idx] =
                                    context.alpha_state.reduce_gradient(node_name, &grad);
                            }
                            record_patches_relu_intermediate(
                                node_name,
                                &node_cb,
                                pre_activation,
                                context,
                            )?;
                            return accumulate_crown_result(network, first_input, new_cb, context);
                        }
                        Err(e) => {
                            debug!(
                                "DAG α-CROWN: Patches alpha-ReLU failed at {}: {}, \
                                 falling back to Dense alpha",
                                node_name, e
                            );
                        }
                    }
                }
            }
        }

        // Patches-native ReLU backward (heuristic slope, no alpha or alpha failed).
        //
        // #1937: record Dense intermediates BEFORE the backward step mutates
        // `node_cb` (the stored A matrix must be the coefficients at the ReLU
        // output, matching the dense/alpha branches). Without this, a ReLU that
        // takes the heuristic branch during the AnalyticChain intermediates
        // pass leaves no A matrix / pre-ReLU bounds, so
        // `compute_graph_chain_rule_gradients` emits a zero-length gradient and
        // `GraphAlphaState::update` skips the node every iteration ("gradient
        // length 0 != alpha length N"). Recording is gradient-only — bounds
        // are untouched and any alpha in [0,1] stays a valid slope — so a
        // recording failure (e.g. densify memory cap) just leaves the gradient
        // zero, exactly the pre-fix behavior.
        if context.intermediate.is_some() {
            if let Err(e) =
                record_patches_relu_intermediate(node_name, &node_cb, pre_activation, context)
            {
                debug!(
                    "DAG α-CROWN: heuristic patches-ReLU intermediate recording failed at {}: {} \
                     (#1937); gradient for this node stays zero",
                    node_name, e
                );
            }
        }
        match crown_backward_step_patches(
            &node.layer,
            &mut node_cb,
            pre_activation,
            context.engine,
            0,
            "DAG-α-CROWN",
            context.deadline,
        ) {
            Ok(CrownStepResult::Continue) => {
                return accumulate_crown_result(network, first_input, node_cb, context);
            }
            Ok(CrownStepResult::IbpFallback(fallback)) => {
                if fallback.reason == crate::types::CrownIbpFallbackReason::MemoryBudgetExceeded {
                    debug!(
                        "DAG α-CROWN: ReLU Patches dispatch hit memory budget at {}: {}; falling back to CROWN",
                        node_name, fallback.details
                    );
                    return crown_fallback_result(network, context);
                }
                debug!(
                    "DAG α-CROWN: ReLU Patches dispatch failed at {}, converting to Dense",
                    node_name
                );
            }
            Err(_) => {
                debug!(
                    "DAG α-CROWN: ReLU Patches dispatch failed at {}, converting to Dense",
                    node_name
                );
            }
        }
    }

    if let Some(result) =
        ensure_dense_or_crown_fallback(network, node_name, "ReLU", &mut node_cb, context)?
    {
        return Ok(result);
    }
    let node_lb = node_cb.into_dense()?;
    let node_lb =
        GraphNetwork::apply_invprop_constraints(node_name, node_lb, context.invprop_state);
    record_dense_relu_intermediate(node_name, &node_lb, pre_activation, context)?;

    if let Some(&relu_idx) = context.relu_name_to_idx.get(node_name) {
        if let Some(alpha) = context.alpha_state.alpha(node_name) {
            let alpha_upper = context.alpha_state.alpha_upper(node_name);
            // #4404: expand channel-only alpha to full spatial before backward.
            let alpha_expanded = context.alpha_state.expand_alpha(node_name, alpha);
            let alpha_upper_expanded =
                alpha_upper.map(|au| context.alpha_state.expand_alpha(node_name, au));
            // #3813: Catch ShapeMismatch from ReLU alpha propagation
            // (RSPLITTER models change intermediate dimensions). Fall back
            // to plain CROWN, which is always sound.
            let propagated = if context.track_gradients {
                relu.propagate_linear_with_alpha(
                    &node_lb,
                    pre_activation,
                    &alpha_expanded,
                    alpha_upper_expanded.as_ref(),
                )
                .map(|(bounds, lower, upper)| (bounds, Some((lower, upper))))
            } else {
                relu.propagate_linear_with_alpha_bound_only(
                    &node_lb,
                    pre_activation,
                    &alpha_expanded,
                    alpha_upper_expanded.as_ref(),
                )
                .map(|bounds| (bounds, None))
            };
            match propagated {
                Ok((new_lb, grads)) => {
                    // #4404: reduce per-neuron gradient to per-channel if channel-only.
                    if let Some((grad, grad_upper)) = grads {
                        context.gradients[relu_idx] =
                            context.alpha_state.reduce_gradient(node_name, &grad);
                        context.gradients_upper[relu_idx] =
                            context.alpha_state.reduce_gradient(node_name, &grad_upper);
                    }
                    return accumulate_dense_result(network, first_input, new_lb, context);
                }
                Err(NyError::ShapeMismatch { .. }) => {
                    debug!(
                        "DAG α-CROWN: ReLU {} alpha propagation ShapeMismatch, CROWN fallback",
                        node_name,
                    );
                    return crown_fallback_result(network, context);
                }
                Err(e) => return Err(e),
            }
        }
    }

    // Fallback: propagate without alpha
    // #3813: Catch ShapeMismatch instead of wrapping in InvalidSpec.
    match relu.propagate_linear_with_bounds(&node_lb, pre_activation) {
        Ok(new_lb) => accumulate_dense_result(network, first_input, new_lb, context),
        Err(NyError::ShapeMismatch { .. }) => {
            debug!(
                "DAG α-CROWN: ReLU {} fallback propagation ShapeMismatch, CROWN fallback",
                node_name,
            );
            crown_fallback_result(network, context)
        }
        Err(e) => Err(NyError::InvalidSpec(format!(
            "DAG α-CROWN failed at node '{}' (ReLU): {}",
            node_name, e
        ))),
    }
}

// Same rationale as handle_relu_node: per-call args differ across match arms
// and DagAlphaNodeContext already bundles the shared state.
#[allow(clippy::too_many_arguments)]
fn handle_monotone_node<FAlpha, FFixed>(
    network: &GraphNetwork,
    node_name: &str,
    layer_type: &str,
    first_input: &str,
    mut node_cb: CrownBounds,
    pre_activation: &BoundedTensor,
    alpha_propagate: FAlpha,
    fixed_propagate: FFixed,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult>
where
    FAlpha: FnOnce(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
    FFixed: FnOnce(&LinearBounds, &BoundedTensor) -> Result<LinearBounds>,
{
    if let Some(result) =
        ensure_dense_or_crown_fallback(network, node_name, layer_type, &mut node_cb, context)?
    {
        return Ok(result);
    }
    let node_lb = node_cb.into_dense()?;
    let node_lb =
        GraphNetwork::apply_invprop_constraints(node_name, node_lb, context.invprop_state);

    // #4118: Catch ShapeMismatch/UnsupportedConfiguration from both
    // alpha AND fixed-slope retry, falling back to plain CROWN instead
    // of propagating the error up to graph-wide IBP fallback.
    match retry_monotone_shape_mismatch_with_fixed_slope(
        node_name,
        layer_type,
        &node_lb,
        pre_activation,
        alpha_propagate,
        fixed_propagate,
    ) {
        Ok(new_lb) => accumulate_dense_result(network, first_input, new_lb, context),
        Err(NyError::ShapeMismatch { .. }) | Err(NyError::UnsupportedConfiguration(_)) => {
            debug!(
                "DAG α-CROWN: {layer_type} {} fixed-slope retry also failed, CROWN fallback",
                node_name,
            );
            crown_fallback_result(network, context)
        }
        Err(e) => Err(e),
    }
}

fn ensure_dense_or_crown_fallback(
    network: &GraphNetwork,
    node_name: &str,
    layer_type: &str,
    node_cb: &mut CrownBounds,
    context: &DagAlphaNodeContext<'_>,
) -> Result<Option<NonlinearNodeResult>> {
    if matches!(node_cb, CrownBounds::Patches(_)) {
        match node_cb.ensure_dense() {
            Ok(_) => {}
            Err(e) => {
                debug!(
                    "DAG α-CROWN: ensure_dense failed at {layer_type} {}: {}, CROWN fallback",
                    node_name, e
                );
                return Ok(Some(crown_fallback_result(network, context)?));
            }
        }
    }
    Ok(None)
}

fn record_patches_relu_intermediate(
    node_name: &str,
    node_cb: &CrownBounds,
    pre_activation: &BoundedTensor,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<()> {
    // Store Dense intermediate for chain-rule gradient
    // computation (#3293). The Patches alpha-ReLU path
    // bypasses Dense intermediate storage. Convert
    // Patches->Dense only for A-matrix storage so
    // compute_graph_chain_rule_gradients gets non-zero
    // gradients for Patches-mode ReLUs.
    // Reference: design doc 2026-03-04-alpha-gradient-
    // patches-alternative.md Approach B.
    let Some(inter) = context.intermediate.as_deref_mut() else {
        return Ok(());
    };

    let dense_lb = node_cb.clone().into_dense()?;
    inter
        .a_at_relu
        .insert(node_name.to_string(), dense_lb.lower_a().clone());
    let (lower, upper) = pre_activation.flatten_to_ix1(&format!("pre-ReLU '{}'", node_name))?;
    inter
        .pre_relu_bounds
        .insert(node_name.to_string(), (lower, upper));
    Ok(())
}

fn record_dense_relu_intermediate(
    node_name: &str,
    node_lb: &LinearBounds,
    pre_activation: &BoundedTensor,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<()> {
    // When capturing intermediates, store A matrix and pre-ReLU bounds
    // BEFORE the ReLU is applied (for chain-rule gradients).
    let Some(inter) = context.intermediate.as_deref_mut() else {
        return Ok(());
    };

    inter
        .a_at_relu
        .insert(node_name.to_string(), node_lb.lower_a().clone());
    let (lower, upper) = pre_activation.flatten_to_ix1(&format!("pre-ReLU '{}'", node_name))?;
    inter
        .pre_relu_bounds
        .insert(node_name.to_string(), (lower, upper));
    Ok(())
}

fn accumulate_crown_result(
    network: &GraphNetwork,
    first_input: &str,
    node_cb: CrownBounds,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    network.accumulate_crown_bounds_to_input(
        first_input,
        node_cb,
        context.node_crown_bounds,
        context.output_dim,
        context.input_dim,
        context.input_accumulated,
    )?;
    Ok(NonlinearNodeResult::Continue)
}

fn accumulate_dense_result(
    network: &GraphNetwork,
    first_input: &str,
    node_lb: LinearBounds,
    context: &mut DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    network.accumulate_dense_bounds_to_input(
        first_input,
        node_lb,
        context.node_crown_bounds,
        context.output_dim,
        context.input_dim,
        context.input_accumulated,
    )?;
    Ok(NonlinearNodeResult::Continue)
}

fn crown_fallback_result(
    network: &GraphNetwork,
    context: &DagAlphaNodeContext<'_>,
) -> Result<NonlinearNodeResult> {
    Ok(NonlinearNodeResult::ReturnBounds(
        network
            .propagate_crown_with_engine_and_deadline(
                context.input,
                context.engine,
                context.deadline,
            )?
            .bounds,
    ))
}
