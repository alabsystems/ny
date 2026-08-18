// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::helpers::resolve_tensor_node_name;
use crate::{LayerSpec, WeightStore};
use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_propagate::Layer;
use std::collections::{HashMap, HashSet};
use tracing::warn;

/// Determine input node names for building a GraphNetwork.
///
/// For each non-weight, non-constant input tensor:
/// - If it's produced by a previous node, use that node's name
/// - If it's a declared network input, use "_input"
/// - Otherwise, return an error (dangling tensor reference)
///
/// Special handling for Concat: In ViT models, Concat is used to combine
/// the CLS token with patch embeddings. The CLS token typically comes from
/// ConstantOfShape (constant value, but not a weight), so we should NOT
/// filter it out as a "constant tensor" for Concat inputs.
pub(super) fn find_graph_input_nodes(
    weights: &WeightStore,
    spec: &LayerSpec,
    layer: &Layer,
    tensor_to_node: &HashMap<String, String>,
    tensor_producer: &HashMap<String, String>,
    constant_tensors: &HashSet<String>,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
) -> Result<Vec<String>> {
    let mut input_nodes = Vec::new();

    // For Concat with embedded constants, the converter already embedded constant
    // data via convert_concat_with_evaluated. The ConcatLayer handles mixed
    // constant + activation inputs internally, so we only need graph connections
    // for the activation (non-constant) inputs.
    let is_concat = matches!(layer, Layer::Concat(_));
    let is_embedded_parameter_unary = matches!(
        layer,
        Layer::LayerNorm(_)
            | Layer::RmsNorm(_)
            | Layer::InstanceNorm1d(_)
            | Layer::GroupNorm(_)
            | Layer::BatchNorm(_)
    );

    // Filter to activation inputs only.
    // - Weights are always baked into the layer (filtered for all ops)
    // - Evaluated constants are embedded in the layer (filtered for most ops)
    // - Constant tensors: for Concat, keep them (data); for other ops, filter them
    //   (but the converter should have already created unary variants for binary ops
    //   with constant tensor inputs via constant_value())
    let activation_inputs: Vec<&String> = if is_embedded_parameter_unary {
        spec.inputs.iter().take(1).collect()
    } else {
        spec.inputs
            .iter()
            .filter(|name| {
                // Always filter out weights (they're baked into the layer)
                if weights.get(name).is_some() {
                    return false;
                }
                // Filter out evaluated constants (they're already embedded in the layer).
                if evaluated_constants.contains_key(*name) {
                    return false;
                }
                // For Concat, keep constant tensor inputs (they're data, not shape info).
                // ViT uses Concat to combine CLS token (from ConstantOfShape) with patches.
                // For other ops, filter out constant tensors only if their value was
                // successfully evaluated (available in weights or evaluated_constants).
                // If the value isn't available, the converter couldn't absorb it and the
                // tensor must remain as an activation input (#411).
                if is_concat {
                    true
                } else if constant_tensors.contains(*name) {
                    // Only filter out if value is available (converter absorbed it)
                    let value_available =
                        weights.get(name).is_some() || evaluated_constants.contains_key(*name);
                    !value_available
                } else {
                    true
                }
            })
            .collect()
    };

    // For binary ops, check activation input count.
    // Exception: Concat with embedded constants may have fewer graph connections
    // than the original input count, since some inputs are embedded as BoundedTensors.
    if layer.is_binary() && activation_inputs.len() < 2 && !is_concat {
        return Err(NyError::ModelLoad(format!(
            "Layer {} ({:?}) expects 2 activation inputs for binary op, got {} (inputs: {:?})",
            spec.name,
            spec.layer_type,
            activation_inputs.len(),
            spec.inputs
        )));
    }

    if layer.is_ternary() && activation_inputs.len() < 3 {
        return Err(NyError::ModelLoad(format!(
            "Layer {} ({:?}) expects 3 activation inputs for ternary op, got {} (inputs: {:?})",
            spec.name,
            spec.layer_type,
            activation_inputs.len(),
            spec.inputs
        )));
    }
    if layer.is_ternary() && activation_inputs.len() > 3 {
        return Err(NyError::ModelLoad(format!(
            "Layer {} ({:?}) expects exactly 3 activation inputs for ternary op, got {} (inputs: {:?})",
            spec.name,
            spec.layer_type,
            activation_inputs.len(),
            spec.inputs
        )));
    }

    if !is_concat && !layer.is_binary() && activation_inputs.is_empty() {
        return Err(NyError::ModelLoad(format!(
            "Layer {} ({:?}) has no activation inputs after filtering (inputs: {:?})",
            spec.name, spec.layer_type, spec.inputs
        )));
    }

    // Resolve each activation input tensor to its producing graph node.
    // Network input tensors are pre-registered in tensor_to_node (builder.rs:55-58),
    // so resolve_tensor_node_name returning None means a genuine dangling reference.
    let resolve = |tensor_name: &str| -> Result<String> {
        resolve_tensor_node_name(tensor_name, tensor_to_node, tensor_producer).ok_or_else(|| {
            warn!(
                "Layer '{}' ({:?}): tensor '{}' not produced by any graph node",
                spec.name, spec.layer_type, tensor_name
            );
            NyError::ModelLoad(format!(
                "Layer '{}' ({:?}) references unresolvable tensor '{}' \
                 — no producer found in graph (possible dangling ONNX reference)",
                spec.name, spec.layer_type, tensor_name
            ))
        })
    };

    if is_concat {
        // Concat can have N inputs (not just 2). Include ALL activation inputs.
        for input_tensor in &activation_inputs {
            input_nodes.push(resolve(input_tensor)?);
        }
    } else if layer.is_binary() {
        // Binary ops (MatMul with two bounded inputs, Add with two activations)
        // need two input nodes
        for input_tensor in activation_inputs.iter().take(2) {
            input_nodes.push(resolve(input_tensor)?);
        }
        // If we only found one activation input, it's a unary operation
        // (e.g., MatMul where one input is a weight, converted to Linear)
        // This case is handled correctly by the single input
    } else if layer.is_ternary() {
        for input_tensor in activation_inputs.iter().take(3) {
            input_nodes.push(resolve(input_tensor)?);
        }
    } else {
        // Unary ops need one input
        if let Some(input_tensor) = activation_inputs.first() {
            input_nodes.push(resolve(input_tensor)?);
        }
    }

    Ok(input_nodes)
}
