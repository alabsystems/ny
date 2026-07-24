// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::compound_nodes::rewrite_compound_nodes;
use super::helpers::{
    declared_output_shape, evaluate_constant_split_outputs, find_activation_inputs,
    handle_split_layer, internal_shape_from_onnx_shape, map_outputs_to_activation_inputs_or_input,
    map_outputs_to_node, map_skipped_outputs, model_is_unbatched, resolve_tensor_node_name,
    resolve_tensor_node_name_via_first_producer, SplitBuildContext, SplitGraphBuildOutcome,
};
use super::inputs::find_graph_input_nodes;
use super::normalization_fusion::try_instance_norm_fusion;
use super::outputs::select_output_node;
use super::INPUT_NODE_NAME;
use crate::graph_options::GraphNetworkOptions;
use crate::{is_multi_output_split, ConvertContext, LayerSpec, TensorSpec, WeightStore};
use ndarray::ArrayD;
use ny_core::{LayerType, NyError, Result};
use ny_propagate::layers::{ExpandLikeLastAxisLayer, SqueezeLayer, WhereLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// Inputs required by the graph network builder, decoupled from any
/// specific model type.
pub struct GraphBuildInputs<'a> {
    /// Network layer specifications in topological order.
    pub layers: &'a [LayerSpec],
    /// Network input tensor specs (names and shapes).
    pub inputs: &'a [TensorSpec],
    /// Network output tensor specs.
    pub outputs: &'a [TensorSpec],
    /// Model weights.
    pub weights: &'a WeightStore,
    /// Maps each tensor name to its producer tensor (first input of producing op).
    /// Used for tracing through intermediate ops like Cast, Transpose, Reshape.
    pub tensor_producer: &'a HashMap<String, String>,
    /// Set of tensor names that are outputs of constant-producing ops.
    pub constant_tensors: &'a HashSet<String>,
    /// Known tensor shapes keyed by tensor name.
    pub tensor_shapes: &'a HashMap<String, Vec<i64>>,
}

fn layer_type_embeds_parameter_inputs(layer_type: &LayerType) -> bool {
    matches!(
        layer_type,
        LayerType::LayerNorm
            | LayerType::RMSNorm
            | LayerType::InstanceNorm
            | LayerType::GroupNorm
            | LayerType::BatchNorm
    )
}

fn can_lower_expand_with_live_shape_reference(
    spec: &LayerSpec,
    context: &ConvertContext<'_>,
    tensor_to_node: &HashMap<String, String>,
    tensor_producer: &HashMap<String, String>,
) -> bool {
    if spec.layer_type != LayerType::Expand || spec.inputs.len() < 2 {
        return false;
    }
    let data_name = &spec.inputs[0];
    let shape_name = &spec.inputs[1];
    !context.is_constant(data_name)
        && resolve_tensor_node_name_via_first_producer(shape_name, tensor_to_node, tensor_producer)
            .is_some()
}

/// Build a DAG-based [`GraphNetwork`] from model specification data.
///
/// Unlike [`build_propagate_network`](crate::build_propagate_network) which
/// creates a sequential network, this builds a proper directed acyclic graph
/// (DAG) that can handle binary operations like attention MatMul (Q@K^T) where
/// both inputs are bounded tensors.
pub fn build_graph_network(
    data: &GraphBuildInputs<'_>,
    options: GraphNetworkOptions,
) -> Result<GraphNetwork> {
    let mut graph = GraphNetwork::new();
    let rewritten = rewrite_compound_nodes(
        data.layers,
        data.weights,
        data.tensor_shapes,
        options.compound_node_policy,
    );
    let (layers, weights, tensor_shapes) =
        rewritten
            .as_ref()
            .map_or((data.layers, data.weights, data.tensor_shapes), |result| {
                (
                    result.layers.as_slice(),
                    &result.weights,
                    &result.tensor_shapes,
                )
            });
    // Globally-unbatched models (all graph inputs rank <= 1) convert ONNX
    // axes / reshape targets verbatim (#cctsdb B5).
    let model_unbatched = model_is_unbatched(data.inputs);
    let context = ConvertContext::new(weights, tensor_shapes, data.constant_tensors)
        .with_model_unbatched(model_unbatched);

    // Track which tensor names are produced by which node names
    let mut tensor_to_node: HashMap<String, String> = HashMap::new();
    for input in data.inputs {
        tensor_to_node.insert(input.name.clone(), INPUT_NODE_NAME.to_string());
    }
    let output_to_spec: HashMap<String, usize> = layers
        .iter()
        .enumerate()
        .flat_map(|(idx, spec)| {
            spec.outputs
                .iter()
                .cloned()
                .map(move |output| (output, idx))
        })
        .collect();

    // Track constant tensors: combine pre-computed set with dynamically discovered ones
    // (e.g., outputs of all-constant Add layers)
    let mut constant_tensors_local = context.constant_tensors.clone();

    // Pre-evaluate all exact constant chains up front so downstream mixed ops can
    // embed their constant side without requiring synthetic graph inputs.
    let mut evaluated_constants: HashMap<String, ArrayD<f32>> = HashMap::new();
    for spec in layers {
        let is_constant_input = |inp: &String| {
            let is_weight = context.weights.get(inp).is_some();
            let is_const = context.constant_tensors.contains(inp);
            let is_eval = evaluated_constants.contains_key(inp);
            is_weight || is_const || is_eval
        };
        let all_inputs_constant = if layer_type_embeds_parameter_inputs(&spec.layer_type) {
            spec.inputs.first().is_some_and(is_constant_input)
        } else {
            spec.inputs.iter().all(is_constant_input)
        };

        // Shape op: requires static tensor shape (from tensor_shapes), not constant value inputs.
        // If input has known shape, fold it even if the input itself isn't a constant.
        let is_shape_with_static_input = spec.layer_type == LayerType::Shape
            && !spec.inputs.is_empty()
            && tensor_shapes.contains_key(&spec.inputs[0]);

        if (all_inputs_constant && !spec.inputs.is_empty())
            || spec.inputs.is_empty()
            || is_shape_with_static_input
        {
            if is_multi_output_split(spec) {
                if let Some(split_outputs) = evaluate_constant_split_outputs(
                    spec,
                    context.weights,
                    &evaluated_constants,
                    data.inputs,
                    tensor_shapes,
                )? {
                    for (output, value) in split_outputs {
                        debug!(
                            "Pre-evaluated constant {} -> shape {:?}",
                            output,
                            value.shape()
                        );
                        evaluated_constants.insert(output, value);
                    }
                }
                continue;
            }
            if spec.outputs.len() != 1 {
                debug!(
                    "Skipping constant evaluation for {} with {} outputs",
                    spec.name,
                    spec.outputs.len()
                );
                continue;
            }
            let result = {
                let eval_context = ConvertContext::with_evaluated_constants(
                    weights,
                    tensor_shapes,
                    data.constant_tensors,
                    &evaluated_constants,
                )
                .with_model_unbatched(model_unbatched);
                eval_context.evaluate_constant_layer(spec, &evaluated_constants)
            };
            if let Some(result) = result {
                let output = &spec.outputs[0];
                debug!(
                    "Pre-evaluated constant {} -> shape {:?}",
                    output,
                    result.shape()
                );
                evaluated_constants.insert(output.clone(), result);
            }
        }
    }

    // Recreate context with evaluated constants so converters can use them
    let context = ConvertContext::with_evaluated_constants(
        weights,
        tensor_shapes,
        data.constant_tensors,
        &evaluated_constants,
    )
    .with_model_unbatched(model_unbatched);

    // Track which layers were successfully converted
    let mut skipped_count = 0;
    let mut constant_skipped = 0;
    let mut last_added_node: Option<String> = None;

    for (spec_idx, spec) in layers.iter().enumerate() {
        // Check if all inputs are constants (weights or constant tensors from skipped ops).
        // If so, skip this layer - it's a constant computation that doesn't depend on activations.
        let is_constant_input = |inp: &String| {
            context.weights.get(inp).is_some()
                || context.weights.get_integers(inp).is_some()
                || constant_tensors_local.contains(inp)
                || evaluated_constants.contains_key(inp)
        };
        let all_inputs_constant = if layer_type_embeds_parameter_inputs(&spec.layer_type) {
            spec.inputs.first().is_some_and(is_constant_input)
        } else {
            spec.inputs.iter().all(is_constant_input)
        };

        if all_inputs_constant && !spec.inputs.is_empty() {
            // Only skip and mark outputs as constant if output values are actually
            // available (in evaluated_constants or weights). If values aren't available,
            // downstream layers can't use them and the converter will fail to create
            // unary variants for binary ops that reference them (#411).
            let outputs_have_values = spec.outputs.iter().all(|output| {
                evaluated_constants.contains_key(output) || context.weights.get(output).is_some()
            });
            if outputs_have_values {
                debug!(
                    "Skipping layer {} (all inputs constant, outputs evaluated): {:?}",
                    spec.name, spec.inputs
                );
                for output in &spec.outputs {
                    constant_tensors_local.insert(output.clone());
                }
                constant_skipped += 1;
                skipped_count += 1;
                continue;
            }
            // All inputs constant but output values not computed by const_fold.
            // Include this layer in the graph so bounds can propagate through it.
            debug!(
                "Layer {} has all constant inputs but output values not evaluated, including in graph",
                spec.name
            );
        }

        // Shape ops are shape-only: their output depends on the input tensor's
        // STATIC shape, never its runtime values. Load-time const-folding
        // (ny-onnx const_fold) resolves them from graph shape inference with the
        // batch dim pinned to 1, storing the value in the WeightStore even though
        // the Shape node's input is an activation. Skip such nodes here —
        // downstream consumers already read the folded constant — instead of
        // falling through to convert_layer_spec, which unconditionally rejects
        // Shape and (in permissive mode) inserts a dangling OpaqueSkipLayer with
        // [-inf, +inf] bounds (vit_2023 #8).
        let outputs_already_folded = !spec.outputs.is_empty()
            && spec.outputs.iter().all(|output| {
                evaluated_constants.contains_key(output)
                    || context.weights.get(output).is_some()
                    || context.weights.get_integers(output).is_some()
            });
        if spec.layer_type == LayerType::Shape && outputs_already_folded {
            debug!(
                "Skipping Shape layer {} (output already const-folded at load time)",
                spec.name
            );
            for output in &spec.outputs {
                constant_tensors_local.insert(output.clone());
            }
            constant_skipped += 1;
            skipped_count += 1;
            continue;
        }

        if spec.inputs.is_empty() {
            debug!(
                "Skipping layer {} with no inputs; treating outputs as constants",
                spec.name
            );
            for output in &spec.outputs {
                constant_tensors_local.insert(output.clone());
            }
            constant_skipped += 1;
            skipped_count += 1;
            continue;
        }

        // Handle Split op specially: creates multiple Slice layers (one per output)
        if is_multi_output_split(spec) {
            let mut split_ctx = SplitBuildContext {
                weights: context.weights,
                evaluated_constants: &evaluated_constants,
                constant_tensors: &constant_tensors_local,
                inputs: data.inputs,
                tensor_shapes: context.tensor_shapes,
                graph: &mut graph,
                tensor_to_node: &mut tensor_to_node,
                last_added_node: &mut last_added_node,
            };
            let outcome = handle_split_layer(spec, &mut split_ctx)?;
            if matches!(outcome, SplitGraphBuildOutcome::Skipped) {
                skipped_count += 1;
            }
            continue;
        }

        // Handle Where op with constant true/false values specially
        if spec.layer_type == LayerType::Where && spec.inputs.len() >= 3 {
            let condition_input = &spec.inputs[0];
            let true_input = &spec.inputs[1];
            let false_input = &spec.inputs[2];

            // Check if true_value and false_value are constants
            let true_const = data
                .weights
                .get(true_input)
                .cloned()
                .or_else(|| evaluated_constants.get(true_input).cloned());
            let false_const = data
                .weights
                .get(false_input)
                .cloned()
                .or_else(|| evaluated_constants.get(false_input).cloned());

            // If both are constants, create WhereLayer with embedded constants
            if let (Some(tc), Some(fc)) = (true_const, false_const) {
                let where_layer = Layer::Where(WhereLayer::with_constants(Some(tc), Some(fc)));

                // Only need the condition as activation input. If the condition
                // were constant, the all-constant check (line 147) would skip this layer.
                let cond_node = tensor_to_node
                    .get(condition_input)
                    .cloned()
                    .ok_or_else(|| {
                        warn!(
                            "Where '{}': condition '{}' not in tensor_to_node",
                            spec.name, condition_input
                        );
                        NyError::ModelLoad(format!(
                            "Where '{}' references unresolvable condition tensor '{}' \
                         — no producer found in graph",
                            spec.name, condition_input
                        ))
                    })?;

                let node = GraphNode::new(spec.name.clone(), where_layer, vec![cond_node]);
                graph.try_add_node(node)?;
                last_added_node = Some(spec.name.clone());

                map_outputs_to_node(&mut tensor_to_node, &spec.outputs, &spec.name);

                debug!("Created Where node '{}' with embedded constants", spec.name);
                continue;
            }
            // If not both constants, fall through to normal handling
        }

        if let Some(fusion) = try_instance_norm_fusion(spec_idx, layers, &output_to_spec, &context)?
        {
            let input_node = resolve_tensor_node_name(
                &fusion.activation_input_tensor,
                &tensor_to_node,
                data.tensor_producer,
            )
            .ok_or_else(|| {
                warn!(
                    "Fused InstanceNorm '{}' references unresolvable tensor '{}'",
                    spec.name, fusion.activation_input_tensor
                );
                NyError::ModelLoad(format!(
                    "Fused InstanceNorm '{}' references unresolvable tensor '{}' \
                     — no producer found in graph",
                    spec.name, fusion.activation_input_tensor
                ))
            })?;
            let node = GraphNode::new(spec.name.clone(), fusion.layer, vec![input_node]);
            graph.try_add_node(node)?;
            last_added_node = Some(spec.name.clone());
            map_outputs_to_node(&mut tensor_to_node, &spec.outputs, &spec.name);
            continue;
        }

        // Try to convert the layer - skip unsupported ops with warnings
        // For Concat, use evaluated constants map to find pre-computed constant inputs
        let layer = match if spec.layer_type == LayerType::Concat {
            context.convert_concat_with_evaluated(spec, &evaluated_constants)
        } else {
            context.convert_layer(spec)
        } {
            Ok(l) => l,
            Err(NyError::UnsupportedOp(msg))
                if msg.contains("dynamic shape") && options.allow_dynamic_reshape =>
            {
                skipped_count += 1;
                let activation_inputs = find_activation_inputs(
                    &spec.inputs,
                    context.weights,
                    &constant_tensors_local,
                    &evaluated_constants,
                );
                if activation_inputs.len() <= 1 {
                    // Single activation input: Reshape is value-preserving (identity for
                    // values, only shape changes), so identity pass-through is sound.
                    debug!(
                        "Skipping Reshape {} with dynamic shape (identity pass-through)",
                        spec.name
                    );
                    map_outputs_to_activation_inputs_or_input(
                        &mut tensor_to_node,
                        &spec.outputs,
                        &activation_inputs,
                        &spec.name,
                    )?;
                } else {
                    // Multiple activation inputs (e.g., dynamic shape tensor is also a
                    // graph input): we can't guarantee shape at verification time, so
                    // use OpaqueSkipLayer for conservative bounds.
                    warn!(
                        "Skipping Reshape {} with dynamic shape and {} activation inputs; using conservative bounds",
                        spec.name, activation_inputs.len()
                    );
                    let declared_shape =
                        declared_output_shape(&spec.outputs, context.tensor_shapes, data.inputs);
                    map_skipped_outputs(
                        &mut graph,
                        &mut tensor_to_node,
                        &spec.outputs,
                        &activation_inputs,
                        &spec.name,
                        &mut last_added_node,
                        declared_shape,
                    )?;
                }
                continue;
            }
            Err(NyError::UnsupportedOp(msg)) if msg.contains("dynamic shape") => {
                return Err(NyError::UnsupportedOp(format!(
                    "{}; use GraphNetworkOptions::permissive() to allow skipping",
                    msg
                )));
            }
            Err(NyError::UnsupportedOp(msg))
                if msg.contains("constant-side Expand")
                    && can_lower_expand_with_live_shape_reference(
                        spec,
                        &context,
                        &tensor_to_node,
                        data.tensor_producer,
                    ) =>
            {
                debug!(
                    "Lowering Expand '{}' with pre-evaluated Shape(reference) to ExpandLikeLastAxis: {}",
                    spec.name, msg
                );
                Layer::ExpandLikeLastAxis(ExpandLikeLastAxisLayer::new())
            }
            Err(NyError::UnsupportedOp(msg)) => {
                // Skip unsupported layers — insert OpaqueSkipLayer for conservative [-inf, +inf] bounds.
                // Using identity pass-through would be unsound for non-identity ops.
                warn!(
                    "Skipping unsupported layer {} (type {:?}) in graph: {}; using conservative unbounded bounds",
                    spec.name, spec.layer_type, msg
                );
                skipped_count += 1;

                let activation_inputs = find_activation_inputs(
                    &spec.inputs,
                    context.weights,
                    &constant_tensors_local,
                    &evaluated_constants,
                );
                // Shape-carrying OpaqueSkip (#cctsdb A1): emit the skipped
                // op's DECLARED (ORT-inferred) output shape so downstream
                // shape-sensitive ops (Concat, ScatterND, ...) see consistent
                // shapes and propagate [-inf, +inf] instead of hard-erroring.
                let declared_shape =
                    declared_output_shape(&spec.outputs, context.tensor_shapes, data.inputs);
                map_skipped_outputs(
                    &mut graph,
                    &mut tensor_to_node,
                    &spec.outputs,
                    &activation_inputs,
                    &spec.name,
                    &mut last_added_node,
                    declared_shape,
                )?;
                continue;
            }
            Err(e) => return Err(e),
        };

        // Find input node names for this layer
        let input_nodes = find_graph_input_nodes(
            context.weights,
            spec,
            &layer,
            &tensor_to_node,
            data.tensor_producer,
            &constant_tensors_local,
            &evaluated_constants,
        )?;

        // Create and add the graph node
        let node = GraphNode::new(spec.name.clone(), layer.clone(), input_nodes.clone());
        graph.try_add_node(node)?;
        last_added_node = Some(spec.name.clone());

        // Record this node's output tensor -> node mapping
        for output_name in &spec.outputs {
            tensor_to_node.insert(output_name.clone(), spec.name.clone());
        }

        // ONNX MatMul spec: when a 1D input is promoted to 2D, the extra dimension
        // must be removed after the matmul. If B was 1D (K,) → (K,1), output has
        // trailing dim=1 that must be squeezed. If A was 1D (K,) → (1,K), output has
        // a dimension after batch that must be squeezed.
        // The converter already promoted the weight to 2D; here we add the Squeeze.
        if spec.layer_type == LayerType::MatMul && spec.inputs.len() >= 2 {
            let input_b = &spec.inputs[1];
            let input_a = &spec.inputs[0];
            let b_1d = context.weights.get(input_b).is_some_and(|w| w.ndim() == 1);
            let a_1d = context.weights.get(input_a).is_some_and(|w| w.ndim() == 1);
            if b_1d || a_1d {
                // B was 1D: squeeze trailing dimension (-1).
                // A was 1D: squeeze the dimension just after batch (-2 for 3D, but
                // SqueezeLayer with axis=-1 also works because the prepended dim
                // becomes the last after W@B transpose). Use -1 for both cases.
                let squeeze_axis = -1_i32;
                let squeeze_name = format!("{}_onnx_1d_squeeze", spec.name);
                let squeeze_layer = Layer::Squeeze(SqueezeLayer::new(squeeze_axis));
                let squeeze_node =
                    GraphNode::new(squeeze_name.clone(), squeeze_layer, vec![spec.name.clone()]);
                graph.try_add_node(squeeze_node)?;
                last_added_node = Some(squeeze_name.clone());

                // Remap outputs to the Squeeze node
                for output_name in &spec.outputs {
                    tensor_to_node.insert(output_name.clone(), squeeze_name.clone());
                }
                debug!(
                    "MatMul {} had 1D weight input; inserted Squeeze({}) node '{}'",
                    spec.name, squeeze_axis, squeeze_name
                );
            }
        }
    }

    if constant_skipped > 0 {
        debug!(
            "Skipped {} layers with all-constant inputs",
            constant_skipped
        );
    }

    // Record declared (load-time shape-inferred) output shapes per node.
    // Metadata for the taint-gated IBP degrade path (#cctsdb A2): when bound
    // computation fails at a node downstream of an OpaqueSkip, the propagator
    // substitutes [-inf, +inf] bounds in this declared shape. Never used for
    // finite bound values.
    {
        let model_unbatched = model_is_unbatched(data.inputs);
        for spec in layers {
            for output in &spec.outputs {
                let Some(node_name) = tensor_to_node.get(output) else {
                    continue;
                };
                if node_name == INPUT_NODE_NAME
                    || !graph.contains_node(node_name)
                    || graph.declared_shape(node_name).is_some()
                {
                    continue;
                }
                let Some(onnx_shape) = tensor_shapes.get(output) else {
                    continue;
                };
                if let Some(shape) = internal_shape_from_onnx_shape(onnx_shape, model_unbatched) {
                    graph.set_declared_shape(node_name.clone(), shape);
                }
            }
        }
    }

    let output_node = select_output_node(
        data.outputs,
        data.tensor_producer,
        &options,
        &tensor_to_node,
        last_added_node,
        &graph,
    )?;

    if let Some(node_name) = output_node {
        graph.set_output(node_name);
    }

    if skipped_count > 0 {
        info!(
            "Built GraphNetwork with {} nodes ({} layers skipped)",
            graph.num_nodes(),
            skipped_count
        );
    } else {
        info!("Built GraphNetwork with {} nodes", graph.num_nodes());
    }

    Ok(graph)
}

#[cfg(test)]
#[path = "builder_tests.rs"]
mod tests;
