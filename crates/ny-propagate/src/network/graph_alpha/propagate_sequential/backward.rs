// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential α-CROWN backward pass dispatch.
//!
//! Contains the per-layer backward propagation logic for sequential graph α-CROWN:
//! - [`GraphNetwork::sequential_backward_pass`] — main backward loop over execution order
//! - [`GraphNetwork::propagate_alpha_crown_single_pass_sequential_graph`] — single-pass wrapper
//!   used by numerical gradient computation
//! - [`propagate_sequential_generic_layer_backward`] — generic layer dispatch with conv engine
//!   threading and error normalization

use crate::bounds::{AlphaState, LinearBounds};
use crate::invprop::{InvpropConfig, OutputConstraints};
use crate::layers::{BoundPropagation, Layer, ReLULayer};
use crate::network::core::{GraphNetwork, GraphNode};
use crate::network::graph_alpha::invprop_backward::augment_bounds_with_constraints;
use crate::NETWORK_INPUT;

use ndarray::Array1;
use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use std::collections::HashMap;
use tracing::trace;

use super::{
    BackwardPassResult, SequentialBackwardPassContext, SequentialBackwardPassRequest,
    SequentialSinglePassRequest,
};

impl GraphNetwork {
    /// Shared backward pass for sequential GraphNetwork alpha-CROWN.
    ///
    /// Returns `(linear_bounds, gradients_if_collected)`.
    pub(super) fn sequential_backward_pass(
        &self,
        request: SequentialBackwardPassRequest<'_>,
    ) -> Result<BackwardPassResult> {
        let SequentialBackwardPassRequest {
            context,
            invprop_config,
            output_constraints,
            collect_gradients,
            mut bounds_without_oc,
        } = request;
        let mut linear_bounds = LinearBounds::identity(context.output_dim);
        let mut gradients = collect_gradients.then(|| {
            context
                .alpha_state
                .alphas
                .iter()
                .map(|alpha| Array1::zeros(alpha.len()))
                .collect::<Vec<_>>()
        });
        let mut gradients_upper = collect_gradients.then(|| {
            context
                .alpha_state
                .alphas_upper
                .iter()
                .map(|alpha| Array1::zeros(alpha.len()))
                .collect::<Vec<_>>()
        });

        {
            let mut gradient_buffers = SequentialGradientBuffers {
                collect_gradients,
                gradients: &mut gradients,
                gradients_upper: &mut gradients_upper,
            };

            for node_name in context.exec_order.iter().rev() {
                let node = self.nodes.get(node_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!("Node not found: {}", node_name))
                })?;
                let pre_activation =
                    resolve_sequential_backward_input(context.input, context.node_bounds, node)?;
                propagate_sequential_backward_node(
                    context,
                    node_name,
                    &node.layer,
                    pre_activation,
                    &mut linear_bounds,
                    &mut gradient_buffers,
                )?;
                maybe_apply_invprop_to_node(
                    context.alpha_state,
                    invprop_config,
                    output_constraints,
                    node_name,
                    &node.layer,
                    &mut linear_bounds,
                    &mut bounds_without_oc,
                );
            }
        }

        maybe_apply_invprop_to_input(
            context.alpha_state,
            invprop_config,
            output_constraints,
            &mut linear_bounds,
        );

        Ok((linear_bounds, gradients, gradients_upper))
    }

    /// Single backward pass for sequential GraphNetwork α-CROWN, used for numerical gradients.
    pub(in crate::network::graph_alpha) fn propagate_alpha_crown_single_pass_sequential_graph(
        &self,
        request: SequentialSinglePassRequest<'_>,
    ) -> Result<BoundedTensor> {
        let SequentialSinglePassRequest {
            input,
            node_bounds,
            exec_order,
            output_dim,
            relu_name_to_idx,
            alpha_state,
            engine,
            deadline,
        } = request;
        let backward_pass = SequentialBackwardPassRequest {
            context: SequentialBackwardPassContext {
                input,
                node_bounds,
                exec_order,
                output_dim,
                relu_name_to_idx,
                alpha_state,
                engine,
            },
            invprop_config: None,
            output_constraints: None,
            collect_gradients: false,
            bounds_without_oc: None,
        };
        let (linear_bounds, _, _) = match self.sequential_backward_pass(backward_pass) {
            Ok(result) => result,
            // #3166: Catch both UnsupportedOp and UnsupportedConfiguration.
            // #3795: DeadlineExceeded also falls back.
            Err(
                NyError::UnsupportedOp(_)
                | NyError::UnsupportedConfiguration(_)
                | NyError::DeadlineExceeded(_),
            ) => {
                return self
                    .propagate_crown_with_engine_and_deadline(input, engine, deadline)
                    .map(|r| r.bounds);
            }
            Err(e) => return Err(e),
        };
        Ok(linear_bounds.concretize_sound(input))
    }
}

/// Bundles optional sequential alpha-gradient capture state for helper dispatch.
struct SequentialGradientBuffers<'a> {
    collect_gradients: bool,
    gradients: &'a mut Option<Vec<Array1<f32>>>,
    gradients_upper: &'a mut Option<Vec<Array1<f32>>>,
}

fn resolve_sequential_backward_input<'a>(
    input: &'a BoundedTensor,
    node_bounds: &'a HashMap<String, BoundedTensor>,
    node: &GraphNode,
) -> Result<&'a BoundedTensor> {
    let first_input = node.require_unary_input()?;
    if first_input == NETWORK_INPUT {
        return Ok(input);
    }
    node_bounds.get(first_input).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "Pre-activation bounds for {} not found",
            first_input
        ))
    })
}

fn propagate_sequential_backward_node(
    context: SequentialBackwardPassContext<'_>,
    node_name: &str,
    layer: &Layer,
    pre_activation: &BoundedTensor,
    linear_bounds: &mut LinearBounds,
    gradient_buffers: &mut SequentialGradientBuffers<'_>,
) -> Result<()> {
    match layer {
        Layer::Linear(linear) => {
            let next = linear.propagate_linear_with_engine(linear_bounds, context.engine)?;
            if let Cow::Owned(next) = next {
                *linear_bounds = next;
            }
        }
        Layer::ReLU(relu) => {
            if let Some(&relu_idx) = context.relu_name_to_idx.get(node_name) {
                if let Some(new_bounds) = propagate_sequential_relu_with_alpha(
                    relu,
                    node_name,
                    relu_idx,
                    context.alpha_state,
                    linear_bounds,
                    pre_activation,
                    gradient_buffers,
                )? {
                    *linear_bounds = new_bounds;
                    return Ok(());
                }
            }
            *linear_bounds = relu.propagate_linear_with_bounds(linear_bounds, pre_activation)?;
        }
        Layer::Transpose(transpose) => {
            // Clone transpose and set input_shape for proper column permutation.
            let input_shape = pre_activation.shape().to_vec();
            let mut transpose_with_shape = transpose.clone();
            transpose_with_shape.set_input_shape(input_shape);
            let next = transpose_with_shape.propagate_linear(linear_bounds)?;
            if let Cow::Owned(next) = next {
                *linear_bounds = next;
            }
        }
        // Tile: needs input_shape before dispatch (same as Transpose).
        Layer::Tile(tile) => {
            let input_shape = pre_activation.shape().to_vec();
            let mut tile_with_shape = tile.clone();
            tile_with_shape.set_input_shape(input_shape);
            let next = tile_with_shape.propagate_linear(linear_bounds)?;
            if let Cow::Owned(next) = next {
                *linear_bounds = next;
            }
        }
        // === All other layers: exhaustive trait dispatch (#3424) ===
        // Every variant listed — no catch-all. Compiler catches new variants.
        #[rustfmt::skip]
        Layer::Conv1d(_) | Layer::Conv2d(_) | Layer::ConvTranspose1d(_) | Layer::ConvTranspose2d(_)
        | Layer::Slice(_) | Layer::Gather(_) | Layer::ScatterAdd(_) | Layer::IndexAdd(_)
        | Layer::ScatterNd(_) | Layer::Pad(_) | Layer::Resize(_)
        | Layer::Add(_) | Layer::Sub(_) | Layer::Concat(_) | Layer::MatMul(_) | Layer::BilinearCrown(_)
        | Layer::SkipMerge(_) | Layer::OpaqueSkip(_) | Layer::MulBinary(_) | Layer::Where(_)
        | Layer::Div(_) | Layer::Atan2(_)
        | Layer::MinBinary(_) | Layer::MaxBinary(_) | Layer::ExpandLikeLastAxis(_)
        | Layer::GELU(_) | Layer::SiLU(_) | Layer::Tanh(_) | Layer::Sigmoid(_) | Layer::Exp(_)
        | Layer::Log(_) | Layer::Sqrt(_) | Layer::Reciprocal(_) | Layer::Softplus(_) | Layer::HardSwish(_)
        | Layer::Mish(_) | Layer::Selu(_) | Layer::Softsign(_) | Layer::Arctan(_) | Layer::Tan(_)
        | Layer::Sin(_) | Layer::Cos(_) | Layer::Elu(_) | Layer::Celu(_) | Layer::LeakyReLU(_)
        | Layer::HardSigmoid(_) | Layer::Clip(_) | Layer::ThresholdedRelu(_) | Layer::Abs(_)
        | Layer::PowConstant(_) | Layer::Floor(_) | Layer::Ceil(_) | Layer::Round(_) | Layer::Trunc(_) | Layer::Sign(_)
        | Layer::PRelu(_) | Layer::Shrink(_) | Layer::Snake(_) | Layer::Compare(_) | Layer::CompareTensor(_)
        | Layer::Softmax(_) | Layer::CausalSoftmax(_)
        | Layer::LogSoftmax(_) | Layer::LogSumExp(_) | Layer::LayerNorm(_) | Layer::RmsNorm(_)
        | Layer::InstanceNorm1d(_) | Layer::GroupNorm(_) | Layer::AdaIN1d(_) | Layer::BatchNorm(_)
        | Layer::AddConstant(_) | Layer::MulConstant(_) | Layer::DivConstant(_) | Layer::SubConstant(_)
        | Layer::ReduceMean(_) | Layer::ReduceSum(_) | Layer::CumSum(_) | Layer::ReduceMax(_) | Layer::ReduceMin(_)
        | Layer::Topk(_) | Layer::ArgMax(_) | Layer::ArgMin(_) | Layer::ArgSort(_)
        | Layer::Flatten(_) | Layer::Reshape(_) | Layer::Squeeze(_) | Layer::Unsqueeze(_)
        | Layer::QdqPerturbation(_)
        | Layer::AveragePool(_) | Layer::MaxPool2d(_) | Layer::RoPE(_) | Layer::NonZero(_) | Layer::SelfAttention(_) => {
            *linear_bounds = propagate_sequential_generic_layer_backward(
                layer,
                linear_bounds,
                pre_activation,
                node_name,
                context.engine,
            )?;
        }
    }

    Ok(())
}

fn propagate_sequential_relu_with_alpha(
    relu: &ReLULayer,
    node_name: &str,
    relu_idx: usize,
    alpha_state: &AlphaState,
    linear_bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    gradient_buffers: &mut SequentialGradientBuffers<'_>,
) -> Result<Option<LinearBounds>> {
    let Some(alpha) = alpha_state.alpha(relu_idx) else {
        if gradient_buffers.collect_gradients {
            return Err(NyError::InvalidSpec(format!(
                "Missing alpha for ReLU node {}",
                node_name
            )));
        }
        return Ok(None);
    };
    let alpha_upper = alpha_state.alpha_upper(relu_idx);
    let (new_bounds, grad, grad_upper) =
        relu.propagate_linear_with_alpha(linear_bounds, pre_activation, alpha, alpha_upper)?;

    if let Some(grads) = gradient_buffers.gradients.as_mut() {
        let slot = grads.get_mut(relu_idx).ok_or_else(|| {
            NyError::InvalidSpec(format!("Gradient slot missing for ReLU index {}", relu_idx))
        })?;
        *slot = grad;
    }
    if let Some(grads_upper) = gradient_buffers.gradients_upper.as_mut() {
        let slot = grads_upper.get_mut(relu_idx).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Upper gradient slot missing for ReLU index {}",
                relu_idx
            ))
        })?;
        *slot = grad_upper;
    }

    Ok(Some(new_bounds))
}

fn maybe_apply_invprop_to_node(
    alpha_state: &AlphaState,
    invprop_config: Option<&InvpropConfig>,
    output_constraints: Option<&OutputConstraints>,
    node_name: &str,
    layer: &Layer,
    linear_bounds: &mut LinearBounds,
    bounds_without_oc: &mut Option<&mut Option<LinearBounds>>,
) {
    let Some(state) = alpha_state.invprop_state.as_ref() else {
        return;
    };
    let apply_constraints = match invprop_config {
        Some(config) => {
            let layer_type = format!("Bound{}", layer.layer_type());
            config.should_apply_to(node_name, &layer_type)
        }
        None => state.layer_gammas(node_name).is_some(),
    };
    if !apply_constraints {
        return;
    }
    let Some(gammas) = state.layer_gammas(node_name) else {
        return;
    };
    if !gammas.active {
        return;
    }
    let constraints = match invprop_config {
        Some(_) => output_constraints,
        None => Some(&state.constraints),
    };
    let Some(constraints) = constraints else {
        return;
    };

    if let Some(config) = invprop_config {
        if config.best_of_oc_and_no_oc {
            if let Some(slot) = bounds_without_oc.as_mut() {
                **slot = Some(linear_bounds.clone());
            }
        }
    }
    let gammas_lower = gammas.lower_gammas().to_owned();
    let gammas_upper = gammas.upper_gammas().to_owned();
    *linear_bounds =
        augment_bounds_with_constraints(linear_bounds, constraints, &gammas_lower, &gammas_upper);
    trace!(
        "INVPROP: Applied constraint augmentation at layer {}",
        node_name
    );
}

fn maybe_apply_invprop_to_input(
    alpha_state: &AlphaState,
    invprop_config: Option<&InvpropConfig>,
    output_constraints: Option<&OutputConstraints>,
    linear_bounds: &mut LinearBounds,
) {
    let Some(state) = alpha_state.invprop_state.as_ref() else {
        return;
    };
    let apply_input_constraints = match invprop_config {
        Some(config) => config.should_apply_to_input(),
        None => state.layer_gammas(NETWORK_INPUT).is_some(),
    };
    if !apply_input_constraints {
        return;
    }
    let Some(gammas) = state.layer_gammas(NETWORK_INPUT) else {
        return;
    };
    if !gammas.active {
        return;
    }
    let constraints = match invprop_config {
        Some(_) => output_constraints,
        None => Some(&state.constraints),
    };
    let Some(constraints) = constraints else {
        return;
    };

    let gammas_lower = gammas.lower_gammas().to_owned();
    let gammas_upper = gammas.upper_gammas().to_owned();
    *linear_bounds =
        augment_bounds_with_constraints(linear_bounds, constraints, &gammas_lower, &gammas_upper);
}

/// Propagate a generic layer backward in sequential alpha-CROWN context.
///
/// UnsupportedOp/UnsupportedConfiguration returns as-is (caller's problem).
/// Other errors are wrapped in `InvalidSpec` with node context.
fn propagate_sequential_generic_layer_backward(
    layer: &Layer,
    linear_bounds: &LinearBounds,
    pre_activation: &BoundedTensor,
    node_name: &str,
    engine: Option<&dyn GemmEngine>,
) -> Result<LinearBounds> {
    // Conv layers: use engine-aware path for GPU acceleration (#3598).
    // Local macros deduplicate the clone-set-propagate pattern shared by
    // Conv1d/ConvTranspose1d (set_input_length) and Conv2d/ConvTranspose2d
    // (set_input_shape). Both pairs are structurally identical. (#3812)
    macro_rules! conv1d_backward {
        ($conv:expr) => {{
            let in_len = *pre_activation
                .shape()
                .last()
                .ok_or_else(|| NyError::ShapeMismatch {
                    expected: vec![1],
                    got: pre_activation.shape().to_vec(),
                })?;
            let mut conv = $conv.clone();
            conv.set_input_length(in_len);
            conv.propagate_linear_with_engine(linear_bounds, engine)
                .map(|cow| cow.into_owned())
        }};
    }
    macro_rules! conv2d_backward {
        ($conv:expr) => {{
            let input_shape = pre_activation.shape();
            let (in_h, in_w) = if input_shape.len() >= 3 {
                (
                    input_shape[input_shape.len() - 2],
                    input_shape[input_shape.len() - 1],
                )
            } else {
                return Err(NyError::ShapeMismatch {
                    expected: vec![0, 0, 0],
                    got: input_shape.to_vec(),
                });
            };
            let mut conv = $conv.clone();
            conv.set_input_shape(in_h, in_w);
            conv.propagate_linear_with_engine(linear_bounds, engine)
                .map(|cow| cow.into_owned())
        }};
    }
    let backward_result = match layer {
        Layer::Conv1d(c) => conv1d_backward!(c),
        Layer::ConvTranspose1d(c) => conv1d_backward!(c),
        Layer::Conv2d(c) => conv2d_backward!(c),
        Layer::ConvTranspose2d(c) => conv2d_backward!(c),
        _ => layer.propagate_crown_backward(linear_bounds, Some(pre_activation)),
    };
    match backward_result {
        Ok(lb) => Ok(lb),
        // #3166: Catch both UnsupportedOp and UnsupportedConfiguration.
        // #3795: DeadlineExceeded also falls back.
        // #3813: ShapeMismatch from Dense Conv2d backward when graph restructuring
        // (e.g., RSPLITTER) changes intermediate dimensions — fallback to CROWN.
        // #2888: NumericalInstability from non-finite pre-activation bounds.
        Err(
            NyError::UnsupportedOp(_)
            | NyError::UnsupportedConfiguration(_)
            | NyError::DeadlineExceeded(_)
            | NyError::ShapeMismatch { .. }
            | NyError::NumericalInstability(_),
        ) => Err(NyError::UnsupportedOp(layer.layer_type().to_string())),
        Err(e) => Err(NyError::InvalidSpec(format!(
            "Sequential α-CROWN failed at node '{}' ({}): {}",
            node_name,
            layer.layer_type(),
            e
        ))),
    }
}
