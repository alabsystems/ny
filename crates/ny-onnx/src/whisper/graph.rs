// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{AttributeValue, LayerSpec};
use ndarray::ArrayD;
use ny_core::{LayerType, NyError, Result};
use ny_propagate::{
    layers::{MatMulLayer, ReshapeLayer, TransposeLayer},
    GraphNetwork, GraphNode, Layer,
};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use super::loader::block_index::parse_block_index_from_layer;
use super::loader::scope::{detect_block_scope, WhisperBlockScope};
use super::model::WhisperModel;
use super::subgraph::attention::AttentionRewritePlan;

/// Attention-specific overrides for the block graph builder.
/// When present, the builder inserts QKV reshape/transpose nodes and
/// overrides the attention core layer/input wiring.
struct AttentionOverrides {
    q_src: String,
    k_src: String,
    v_src: String,
    q_reshape: String,
    q_transpose: String,
    k_reshape: String,
    k_transpose: String,
    v_reshape: String,
    v_transpose: String,
    ctx_transpose: String,
    ctx_reshape: String,
    attn_scores: String,
    attn_softmax: String,
    attn_ctx: String,
    attn_out: String,
    head_dim: usize,
    hidden_dim: usize,
    qkv_target_shape: Vec<i64>,
    qkv_perm: Vec<usize>,
    /// Original exporter shape-plumbing nodes (reshape/transpose/scale between
    /// the projections and the attention core) whose role the synthetic wiring
    /// takes over. They are skipped and traced through so downstream consumers
    /// stay connected. Collected structurally in `attention_rewrite_plan`.
    replaced_nodes: std::collections::HashSet<String>,
}

impl WhisperModel {
    fn trace_passthrough_outputs(
        spec: &LayerSpec,
        input: &str,
        tensor_to_node: &mut HashMap<String, String>,
        external_tensors: &mut std::collections::HashSet<String>,
    ) {
        for output in &spec.outputs {
            if output.is_empty() {
                continue;
            }
            if let Some(src_node) = tensor_to_node.get(input) {
                tensor_to_node.insert(output.clone(), src_node.clone());
            } else if external_tensors.contains(input) {
                external_tensors.insert(output.clone());
            } else {
                external_tensors.insert(input.to_string());
                external_tensors.insert(output.clone());
            }
        }
    }

    fn find_block_entry_tensor(
        &self,
        index: usize,
        block_start: usize,
        block_end: usize,
        block_scope: WhisperBlockScope,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Result<String> {
        let mut constant_tensors_local = self.model.constant_tensors.clone();
        for layer_idx in block_start..block_end {
            let spec = &self.model.network.layers[layer_idx];
            let scoped_index = parse_block_index_from_layer(spec, block_scope);
            let parsed_index =
                scoped_index.or_else(|| parse_block_index_from_layer(spec, WhisperBlockScope::All));
            if let Some(other_idx) = parsed_index {
                if other_idx != index {
                    continue;
                }
            }

            // Intentional: find_activation_input returns Err for non-activation nodes during
            // the scan — this is expected, not an error worth logging.
            if let Ok(entry) = self.find_activation_input(
                &spec.inputs,
                &constant_tensors_local,
                evaluated_constants,
            ) {
                return Ok(entry);
            }

            for output in &spec.outputs {
                if !output.is_empty() {
                    constant_tensors_local.insert(output.clone());
                }
            }
        }

        Err(NyError::InvalidSpec(format!(
            "No activation input found for block {}",
            index
        )))
    }

    /// Get a single encoder layer as a GraphNetwork for proper DAG verification.
    ///
    /// Unlike `encoder_layer()` which returns a sequential Network, this method
    /// returns a `GraphNetwork` that properly represents the residual connections
    /// in each transformer block.
    ///
    /// # Arguments
    /// * `index` - Block index (0 to encoder_layers-1)
    ///
    /// # Returns
    /// A `GraphNetwork` with nodes for each operation and edges representing
    /// the data flow including residual connections.
    pub fn encoder_layer_graph(&self, index: usize) -> Result<GraphNetwork> {
        self.build_block_graph(index, None)
    }

    /// Get a single encoder layer as a GraphNetwork with explicit attention shape transforms.
    ///
    /// This method augments the extracted block graph by inserting the expected Whisper
    /// attention reshapes/transposes between the Q/K/V projections and the attention core.
    ///
    /// This enables end-to-end IBP over the full block for inputs shaped `[batch, seq, hidden]`.
    pub fn encoder_layer_graph_full(&self, index: usize) -> Result<GraphNetwork> {
        if index >= self.encoder_layers {
            return Err(NyError::InvalidSpec(format!(
                "Encoder layer {} out of range (max {})",
                index, self.encoder_layers
            )));
        }

        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads;
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(NyError::InvalidSpec(format!(
                "hidden_dim {} not divisible by num_heads {}",
                hidden_dim, num_heads
            )));
        }
        let head_dim = hidden_dim / num_heads;

        // Discover the attention nodes structurally (op-type + topology +
        // weight tokens) so exporter naming conventions — legacy
        // `{prefix}/query/MatMul` or torch dynamo `node_view`/`node_MatMul_N`
        // — do not break the extraction.
        let nodes = match self.discover_attention_nodes(index) {
            Ok(nodes) => nodes,
            Err(err) => {
                warn!(
                    "Block {} attention discovery failed ({}); falling back to encoder_layer_graph()",
                    index, err
                );
                return self.encoder_layer_graph(index);
            }
        };

        // Collect the exporter's shape plumbing between the projections and
        // the attention core. It is replaced by synthetic seq-agnostic
        // reshapes/transposes below; the collection fails closed if it finds
        // anything the synthetic wiring would not reproduce, in which case we
        // fall back to the plain graph (original semantics preserved).
        let replaced_nodes = match self.attention_rewrite_plan(index, &nodes) {
            Ok(AttentionRewritePlan { replaced_nodes }) => replaced_nodes,
            Err(err) => {
                warn!(
                    "Block {} attention plumbing not replaceable ({}); falling back to encoder_layer_graph()",
                    index, err
                );
                return self.encoder_layer_graph(index);
            }
        };

        let q_src = nodes.q_src().name.clone();
        let k_src = nodes.k_src().name.clone();
        let v_src = nodes.v_src().name.clone();
        let attn_scores = nodes.attn_scores.name.clone();
        let attn_softmax = nodes.attn_softmax.name.clone();
        let attn_ctx = nodes.attn_ctx.name.clone();
        let attn_out = nodes.out_matmul.name.clone();

        let q_reshape = format!("{q_src}::__reshape_bshd");
        let q_transpose = format!("{q_src}::__transpose_bhsd");
        let k_reshape = format!("{k_src}::__reshape_bshd");
        let k_transpose = format!("{k_src}::__transpose_bhsd");
        let v_reshape = format!("{v_src}::__reshape_bshd");
        let v_transpose = format!("{v_src}::__transpose_bhsd");

        let ctx_transpose = format!("{attn_ctx}::__transpose_bshd");
        let ctx_reshape = format!("{attn_ctx}::__reshape_bsd");

        // Target shapes use ONNX Reshape semantics:
        // - 0 copies the corresponding input dim
        // - fixed dims specify heads and head_dim
        let qkv_target_shape = vec![0, 0, num_heads as i64, head_dim as i64];
        let qkv_perm = vec![0, 2, 1, 3]; // [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]

        let overrides = AttentionOverrides {
            q_src,
            k_src,
            v_src,
            q_reshape,
            q_transpose,
            k_reshape,
            k_transpose,
            v_reshape,
            v_transpose,
            ctx_transpose,
            ctx_reshape,
            attn_scores,
            attn_softmax,
            attn_ctx,
            attn_out,
            head_dim,
            hidden_dim,
            qkv_target_shape,
            qkv_perm,
            replaced_nodes,
        };

        self.build_block_graph(index, Some(overrides))
    }

    /// Core block graph builder shared by `encoder_layer_graph` and `encoder_layer_graph_full`.
    ///
    /// When `attention` is `None`, builds a plain block graph. When `Some`, inserts
    /// QKV reshape/transpose nodes and overrides attention core layer/input wiring.
    fn build_block_graph(
        &self,
        index: usize,
        attention: Option<AttentionOverrides>,
    ) -> Result<GraphNetwork> {
        if index >= self.encoder_layers {
            return Err(NyError::InvalidSpec(format!(
                "Encoder layer {} out of range (max {})",
                index, self.encoder_layers
            )));
        }

        let has_attention = attention.is_some();
        let (block_start, block_end) = self.block_onnx_bounds(index)?;
        let block_scope = detect_block_scope(&self.model.network);

        // Build mapping from tensor name -> producing node name
        let mut tensor_to_node: HashMap<String, String> = HashMap::new();

        // Track block-local constant tensors so activation filtering stays accurate for this block.
        let mut constant_tensors_local = self.model.constant_tensors.clone();

        // Pre-evaluate constant chains (e.g., ConstantOfShape + Add → CLS token).
        // Without this, constant tensors from ConstantOfShape/Shape ops have no values
        // and cannot be embedded into ConcatLayer, causing unresolvable graph inputs (#696).
        // Must run before find_block_entry_tensor so evaluated_constants are filtered (#697).
        let evaluated_constants =
            self.evaluate_block_constants(block_start, block_end, &constant_tensors_local);

        // Identify the block's entry point (first layer's activation input)
        let entry_tensor = self.find_block_entry_tensor(
            index,
            block_start,
            block_end,
            block_scope,
            &evaluated_constants,
        )?;

        // Also track all external tensors (inputs from outside the block)
        let mut external_tensors: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        external_tensors.insert(entry_tensor);

        let mut graph = GraphNetwork::new();
        let mut last_node_name: Option<String> = None;

        let label = if has_attention { "graph_full" } else { "graph" };

        // Process layers in block
        for layer_idx in block_start..block_end {
            let spec = &self.model.network.layers[layer_idx];
            let scoped_index = parse_block_index_from_layer(spec, block_scope);
            let parsed_index =
                scoped_index.or_else(|| parse_block_index_from_layer(spec, WhisperBlockScope::All));
            if let Some(other_idx) = parsed_index {
                if other_idx != index {
                    continue;
                }
            }

            // Structural attention-plumbing replacement: these nodes' role is
            // taken over by the synthetic reshape/transpose/scale wiring the
            // overrides insert (verified in attention_rewrite_plan). Skip
            // them and map their outputs to their data inputs so downstream
            // consumers stay connected. Exporters that bake the export-time
            // sequence length into reshape targets (torch dynamo bakes
            // seq=1500 for Whisper-tiny) would otherwise break shorter-seq
            // propagation with an empty inferred dimension.
            if let Some(ref attn) = attention {
                if attn.replaced_nodes.contains(&spec.name) {
                    let data_input = spec
                        .inputs
                        .iter()
                        .find(|name| {
                            !name.is_empty()
                                && !self.model.weights.contains_key(name)
                                && !constant_tensors_local.contains(*name)
                                && !evaluated_constants.contains_key(*name)
                        })
                        .or_else(|| spec.inputs.iter().find(|name| !name.is_empty()));
                    if let Some(input) = data_input {
                        Self::trace_passthrough_outputs(
                            spec,
                            input,
                            &mut tensor_to_node,
                            &mut external_tensors,
                        );
                    }
                    debug!(
                        "Skipping replaced attention plumbing node {} in {} extraction",
                        spec.name, label
                    );
                    continue;
                }
            }

            // Skip layers whose outputs were already const-folded at load time.
            // A Shape-of-activation node is the canonical case: its input is an
            // activation (so the activation-input filter below would keep it),
            // but the loader already folded its output value from the tensor's
            // static shape into the weight store. Converting the spec instead
            // errors ("Shape op ... input shape was not static"), so treat the
            // spec as constant and let consumers read the folded value.
            let real_outputs: Vec<&String> = spec
                .outputs
                .iter()
                .filter(|name| !name.is_empty())
                .collect();
            let outputs_prefolded = !real_outputs.is_empty()
                && real_outputs.iter().all(|output| {
                    self.model.weights.contains_key(output)
                        || constant_tensors_local.contains(*output)
                        || evaluated_constants.contains_key(*output)
                });
            if outputs_prefolded {
                debug!(
                    "Skipping layer {} (outputs const-folded at load) in {} extraction",
                    spec.name, label
                );
                for output in real_outputs {
                    constant_tensors_local.insert(output.clone());
                }
                continue;
            }

            // Check if this is a constant-only operation (all inputs are weights/constants).
            // Also filter evaluated_constants (#697): pre-evaluated constant chains
            // (e.g., ConstantOfShape from adjacent block indices) may not be in
            // constant_tensors_local but are still constant.
            let activation_inputs: Vec<&String> = spec
                .inputs
                .iter()
                .filter(|name| {
                    !name.is_empty()
                        && !self.model.weights.contains_key(name)
                        && !constant_tensors_local.contains(*name)
                        && !evaluated_constants.contains_key(*name)
                })
                .collect();

            if spec.layer_type == LayerType::Concat {
                let all_constants = spec.inputs.iter().all(|name| {
                    name.is_empty()
                        || self.model.weights.contains_key(name)
                        || constant_tensors_local.contains(name)
                        || evaluated_constants.contains_key(name)
                });
                if all_constants {
                    debug!(
                        "Skipping Concat {} with all-constant inputs in {} extraction",
                        spec.name, label
                    );
                    for output in &spec.outputs {
                        if !output.is_empty() {
                            constant_tensors_local.insert(output.clone());
                        }
                    }
                    continue;
                }
            }

            if activation_inputs.is_empty() {
                // All inputs are constants - skip this layer
                // Don't add to tensor_to_node - outputs will be treated as external
                debug!(
                    "Skipping constant-only layer {} in {} extraction",
                    spec.name, label
                );
                for output in &spec.outputs {
                    if !output.is_empty() {
                        constant_tensors_local.insert(output.clone());
                    }
                }
                continue;
            }

            // Skip Concat layers with only 1 activation input when there are no
            // constant data inputs. Shape-computing Concats (e.g., Reshape target shapes)
            // fit this pattern, but data Concats with constant data (CLS token from
            // weights, constant_tensors, or evaluated_constants) should remain in the graph.
            if spec.layer_type == LayerType::Concat
                && activation_inputs.len() < 2
                && !spec
                    .inputs
                    .iter()
                    .any(|name| self.model.weights.contains_key(name))
                && !spec
                    .inputs
                    .iter()
                    .any(|name| constant_tensors_local.contains(name))
                && !spec
                    .inputs
                    .iter()
                    .any(|name| evaluated_constants.contains_key(name))
            {
                debug!(
                    "Skipping Concat {} with {} activation input(s) - likely shape-computing",
                    spec.name,
                    activation_inputs.len()
                );
                for output in &spec.outputs {
                    if !output.is_empty() {
                        constant_tensors_local.insert(output.clone());
                    }
                }
                continue;
            }

            // Convert LayerSpec to Layer.
            // For Concat, use convert_concat_with_evaluated to embed pre-evaluated
            // constant values into ConcatLayer (#696).
            let mut layer = match if spec.layer_type == LayerType::Concat {
                self.model
                    .convert_context()
                    .convert_concat_with_evaluated(spec, &evaluated_constants)
            } else {
                self.model.convert_layer(spec)
            } {
                Ok(l) => l,
                Err(NyError::UnsupportedOp(msg)) if msg.contains("dynamic shape") => {
                    // Dynamic Reshape - skip and trace through
                    debug!(
                        "Skipping Reshape {} with dynamic shape in {} extraction",
                        spec.name, label
                    );
                    // Map output to input for tracing (first input is data, second is shape)
                    if let Some(input) = spec.inputs.iter().find(|name| !name.is_empty()) {
                        Self::trace_passthrough_outputs(
                            spec,
                            input,
                            &mut tensor_to_node,
                            &mut external_tensors,
                        );
                    } else {
                        for output in &spec.outputs {
                            if !output.is_empty() {
                                external_tensors.insert(output.clone());
                            }
                        }
                    }
                    continue;
                }
                Err(NyError::UnsupportedOp(msg))
                    if msg.contains("all constant inputs")
                        && spec.layer_type == LayerType::Concat =>
                {
                    // All-constant Concat treated as shape-computing by converter — skip.
                    debug!(
                        "Skipping all-constant Concat {} via converter: {}",
                        spec.name, msg
                    );
                    for output in &spec.outputs {
                        if !output.is_empty() {
                            constant_tensors_local.insert(output.clone());
                        }
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };

            if let Layer::Reshape(reshape) = &layer {
                let allowzero = matches!(
                    spec.attributes.get("allowzero"),
                    Some(AttributeValue::Int(1))
                );
                if allowzero && reshape.target_shape.contains(&0) {
                    debug!(
                        "Skipping Reshape {} with allowzero=1 and zero dims in {} extraction",
                        spec.name, label
                    );
                    if let Some(input) = spec.inputs.iter().find(|name| !name.is_empty()) {
                        Self::trace_passthrough_outputs(
                            spec,
                            input,
                            &mut tensor_to_node,
                            &mut external_tensors,
                        );
                    } else {
                        for output in &spec.outputs {
                            if !output.is_empty() {
                                external_tensors.insert(output.clone());
                            }
                        }
                    }
                    continue;
                }
            }

            // Determine input node names.
            // Pass evaluated_constants so pre-evaluated constant inputs are filtered
            // from graph edges (they're already embedded in the layer) (#696).
            let mut input_nodes = self.find_input_nodes(
                spec,
                &layer,
                &tensor_to_node,
                &mut external_tensors,
                &constant_tensors_local,
                &evaluated_constants,
            )?;

            // When attention overrides are present, override the attention core nodes
            // to consume the explicit reshape/transpose nodes instead of the original wiring.
            if let Some(ref attn) = attention {
                if spec.name == attn.attn_scores {
                    let scale = 1.0 / (attn.head_dim as f32).sqrt();
                    layer = Layer::MatMul(MatMulLayer::try_new(true, Some(scale))?);
                    input_nodes = vec![attn.q_transpose.clone(), attn.k_transpose.clone()];
                } else if spec.name == attn.attn_softmax {
                    input_nodes = vec![attn.attn_scores.clone()];
                } else if spec.name == attn.attn_ctx {
                    layer = Layer::MatMul(MatMulLayer::try_new(false, None)?);
                    input_nodes = vec![attn.attn_softmax.clone(), attn.v_transpose.clone()];
                } else if spec.name == attn.attn_out {
                    input_nodes = vec![attn.ctx_reshape.clone()];
                }
            }

            // Create the graph node (#2686: try_add_node returns Result instead of panicking)
            let node = GraphNode::new(spec.name.clone(), layer, input_nodes);
            graph.try_add_node(node)?;

            // Record all outputs so downstream inputs resolve to this node.
            for output_name in &spec.outputs {
                if !output_name.is_empty() {
                    tensor_to_node.insert(output_name.clone(), spec.name.clone());
                }
            }
            last_node_name = Some(spec.name.clone());

            // When attention overrides are present, insert reshape/transpose nodes
            // for Q/K/V after their source nodes exist in the graph.
            if let Some(ref attn) = attention {
                if spec.name == attn.q_src {
                    graph.try_add_node(GraphNode::new(
                        attn.q_reshape.clone(),
                        Layer::Reshape(ReshapeLayer::new(attn.qkv_target_shape.clone())),
                        vec![attn.q_src.clone()],
                    ))?;
                    graph.try_add_node(GraphNode::new(
                        attn.q_transpose.clone(),
                        Layer::Transpose(TransposeLayer::new(attn.qkv_perm.clone())),
                        vec![attn.q_reshape.clone()],
                    ))?;
                } else if spec.name == attn.k_src {
                    graph.try_add_node(GraphNode::new(
                        attn.k_reshape.clone(),
                        Layer::Reshape(ReshapeLayer::new(attn.qkv_target_shape.clone())),
                        vec![attn.k_src.clone()],
                    ))?;
                    graph.try_add_node(GraphNode::new(
                        attn.k_transpose.clone(),
                        Layer::Transpose(TransposeLayer::new(attn.qkv_perm.clone())),
                        vec![attn.k_reshape.clone()],
                    ))?;
                } else if spec.name == attn.v_src {
                    graph.try_add_node(GraphNode::new(
                        attn.v_reshape.clone(),
                        Layer::Reshape(ReshapeLayer::new(attn.qkv_target_shape.clone())),
                        vec![attn.v_src.clone()],
                    ))?;
                    graph.try_add_node(GraphNode::new(
                        attn.v_transpose.clone(),
                        Layer::Transpose(TransposeLayer::new(attn.qkv_perm.clone())),
                        vec![attn.v_reshape.clone()],
                    ))?;
                }

                // Insert transpose+reshape after attention context to restore [batch, seq, hidden].
                if spec.name == attn.attn_ctx {
                    graph.try_add_node(GraphNode::new(
                        attn.ctx_transpose.clone(),
                        Layer::Transpose(TransposeLayer::new(vec![0, 2, 1, 3])),
                        vec![attn.attn_ctx.clone()],
                    ))?;
                    graph.try_add_node(GraphNode::new(
                        attn.ctx_reshape.clone(),
                        Layer::Reshape(ReshapeLayer::new(vec![0, 0, attn.hidden_dim as i64])),
                        vec![attn.ctx_transpose.clone()],
                    ))?;
                }
            }
        }

        // Set the output node (last layer in block)
        if let Some(last_name) = last_node_name {
            graph.set_output(&last_name);
        } else {
            return Err(NyError::InvalidSpec(format!(
                "Block {} produced no graph nodes",
                index
            )));
        }

        if has_attention {
            info!(
                "Built full GraphNetwork for block {} with {} nodes (includes attention shape transforms)",
                index,
                graph.num_nodes()
            );
        } else {
            info!(
                "Built GraphNetwork for block {} with {} nodes",
                index,
                graph.num_nodes()
            );
        }

        Ok(graph)
    }

    pub(crate) fn find_activation_input(
        &self,
        inputs: &[String],
        constant_tensors: &std::collections::HashSet<String>,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Result<String> {
        for input in inputs {
            if input.is_empty() {
                continue;
            }
            if !self.model.weights.contains_key(input)
                && !constant_tensors.contains(input)
                && !evaluated_constants.contains_key(input)
            {
                return Ok(input.clone());
            }
        }
        Err(NyError::InvalidSpec(
            "No activation input found (all inputs are constants)".to_string(),
        ))
    }

    /// Determine the input node names for a layer in the graph.
    ///
    /// `evaluated_constants` contains pre-evaluated constant tensor values that have
    /// been embedded into the layer (e.g., `ConcatLayer::constant_inputs`). These
    /// are filtered from graph edges since the layer handles them internally (#696).
    pub(crate) fn find_input_nodes(
        &self,
        spec: &LayerSpec,
        layer: &Layer,
        tensor_to_node: &HashMap<String, String>,
        external_tensors: &mut std::collections::HashSet<String>,
        constant_tensors: &std::collections::HashSet<String>,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Result<Vec<String>> {
        let mut input_nodes = Vec::new();
        let mut activation_input_count = 0;
        let mut has_external_input = false;
        let allows_constant_inputs = matches!(layer, Layer::Concat(_));

        let is_activation_input =
            |name: &str| !self.model.weights.contains_key(name) && !constant_tensors.contains(name);

        for input in &spec.inputs {
            if input.is_empty() {
                continue;
            }
            // Skip evaluated constants — their values are embedded in the layer (#696).
            if evaluated_constants.contains_key(input) {
                continue;
            }
            // Skip weights that were embedded in the layer during conversion.
            // For Concat, weights are stored in ConcatLayer::constant_inputs.
            if self.model.weights.contains_key(input) && allows_constant_inputs {
                continue;
            }
            if is_activation_input(input) {
                activation_input_count += 1;
                if let Some(node_name) = tensor_to_node.get(input) {
                    input_nodes.push(node_name.clone());
                } else {
                    external_tensors.insert(input.clone());
                    has_external_input = true;
                }
            } else if self.model.weights.contains_key(input) {
                // weights are handled in layer conversion
                if layer.is_binary() {
                    input_nodes.push(input.clone());
                }
            } else if constant_tensors.contains(input) {
                if allows_constant_inputs {
                    if let Some(node_name) = tensor_to_node.get(input) {
                        input_nodes.push(node_name.clone());
                    } else {
                        // Constant tensor with no evaluated value and no producing node.
                        // Previously pushed raw tensor name which fails at propagation.
                        // Return error with actionable message (#696).
                        return Err(NyError::InvalidSpec(format!(
                            "Layer '{}': constant tensor '{}' has no evaluated value and \
                             no producing graph node — cannot resolve as Concat input",
                            spec.name, input
                        )));
                    }
                } else if layer.is_binary() {
                    return Err(NyError::InvalidSpec(
                        "constant tensor inputs without a weight constant".to_string(),
                    ));
                }
            }
        }

        if activation_input_count == 0 {
            return Err(NyError::InvalidSpec(format!(
                "No activation inputs found for layer {}",
                spec.name
            )));
        }

        if has_external_input && !input_nodes.iter().any(|name| name == "_input") {
            input_nodes.push("_input".to_string());
        }

        if input_nodes.is_empty() {
            warn!(
                "Layer {} had activation inputs but resolved to no input nodes; falling back to _input",
                spec.name
            );
            input_nodes.push("_input".to_string());
        }

        Ok(input_nodes)
    }

    /// Pre-evaluate constant chains within a block range.
    ///
    /// Iterates through layers in the block and evaluates constant-producing ops
    /// (Add, Mul, Sub, ConstantOfShape) when all their inputs are known constants.
    /// Returns a map of tensor name → evaluated value.
    ///
    /// This mirrors `ny-build::builder.rs` constant pre-evaluation for the
    /// whisper pipeline, which previously lacked it (#696).
    pub(crate) fn evaluate_block_constants(
        &self,
        block_start: usize,
        block_end: usize,
        constant_tensors: &std::collections::HashSet<String>,
    ) -> HashMap<String, ArrayD<f32>> {
        let mut evaluated: HashMap<String, ArrayD<f32>> = HashMap::new();
        let context = self.model.convert_context();
        for layer_idx in block_start..block_end {
            let spec = &self.model.network.layers[layer_idx];
            if spec.inputs.is_empty() && spec.outputs.is_empty() {
                continue;
            }
            let all_inputs_constant = spec.inputs.iter().all(|inp| {
                inp.is_empty()
                    || self.model.weights.contains_key(inp)
                    || constant_tensors.contains(inp)
                    || evaluated.contains_key(inp)
            });
            if (all_inputs_constant && !spec.inputs.is_empty()) || spec.inputs.is_empty() {
                if let Some(result) = context.evaluate_constant_layer(spec, &evaluated) {
                    if let Some(output) = spec.outputs.first() {
                        if !output.is_empty() {
                            debug!(
                                "Pre-evaluated constant {} -> shape {:?}",
                                output,
                                result.shape()
                            );
                            evaluated.insert(output.clone(), result);
                        }
                    }
                }
            }
        }
        evaluated
    }
}
