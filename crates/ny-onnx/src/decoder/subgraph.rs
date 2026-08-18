// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decoder subgraph extraction for structural analysis.
//!
//! Reconstructs causal self-attention graphs as heuristic `GraphNetwork`
//! artifacts. MLP extraction lives in the sibling module. Cross-attention
//! extraction fails closed because `GraphNetwork` lacks a multi-input contract.
//! These artifacts are not proven equivalent to the loaded ONNX graph and must
//! not authorize a verdict.

use crate::LayerSpec;
use ndarray::ArrayD;
use ny_core::{LayerType, NyError, Result};
use ny_propagate::{
    layers::{CausalSoftmaxLayer, MatMulLayer, ReshapeLayer, TransposeLayer},
    GraphNetwork, GraphNode, Layer,
};
use std::collections::HashMap;
use tracing::{debug, info};

use super::DecoderModel;

impl DecoderModel {
    /// Check if a layer with the given name exists.
    pub(super) fn has_layer(&self, name: &str) -> bool {
        self.model.network.layers.iter().any(|l| l.name == name)
    }

    /// Pre-evaluate constant chains across all model layers.
    ///
    /// Iterates through all layers and evaluates constant-producing ops
    /// (ConstantOfShape, Add with all-constant inputs, etc.) so their values can
    /// be embedded into ConcatLayer via `convert_concat_with_evaluated` (#3317).
    ///
    /// This mirrors `WhisperModel::evaluate_block_constants` from the whisper
    /// pipeline (#696), adapted for decoder subgraph extraction.
    pub(super) fn evaluate_model_constants(&self) -> HashMap<String, ArrayD<f32>> {
        let mut evaluated: HashMap<String, ArrayD<f32>> = HashMap::new();
        let context = self.model.convert_context();
        for spec in &self.model.network.layers {
            if spec.inputs.is_empty() && spec.outputs.is_empty() {
                continue;
            }
            let all_inputs_constant = spec.inputs.iter().all(|inp| {
                inp.is_empty()
                    || self.model.weights.contains_key(inp)
                    || self.model.constant_tensors.contains(inp)
                    || evaluated.contains_key(inp)
            });
            if (all_inputs_constant && !spec.inputs.is_empty()) || spec.inputs.is_empty() {
                if let Some(result) = context.evaluate_constant_layer(spec, &evaluated) {
                    if let Some(output) = spec.outputs.first() {
                        if !output.is_empty() {
                            debug!(
                                "Decoder: pre-evaluated constant {} -> shape {:?}",
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

    /// Reconstruct a causal self-attention graph for structural analysis.
    ///
    /// This extracts: norm1 → Q/K/V projections → causal attention → output projection
    /// The output is the attention delta to be added to the residual.
    ///
    /// The result uses a heuristic head-count hint and is not proof-equivalent
    /// to the loaded ONNX graph. Bounds from it must not authorize a verdict.
    ///
    /// # Arguments
    /// * `block_index` - Index of the decoder block (0 for single-block models)
    ///
    /// # Returns
    /// GraphNetwork representing the attention subgraph.
    pub fn causal_attention_subgraph(&self, block_index: usize) -> Result<GraphNetwork> {
        self.block_info(block_index)?;

        // Determine naming pattern based on structure
        let prefix = if self.num_blocks == 1 && !self.has_layer("/blocks.0/self_attn/q_proj/MatMul")
        {
            // Single block without block index prefix
            String::new()
        } else {
            format!("/blocks.{}", block_index)
        };

        let norm1_out = if prefix.is_empty() {
            "/norm1/Add_1".to_string()
        } else {
            format!("{}/norm1/Add_1", prefix)
        };

        let self_attn_prefix = if prefix.is_empty() {
            "/self_attn".to_string()
        } else {
            format!("{}/self_attn", prefix)
        };

        // Layer names
        let q_matmul = format!("{}/q_proj/MatMul", self_attn_prefix);
        let q_add = format!("{}/q_proj/Add", self_attn_prefix);
        let k_matmul = format!("{}/k_proj/MatMul", self_attn_prefix);
        let k_add = format!("{}/k_proj/Add", self_attn_prefix);
        let v_matmul = format!("{}/v_proj/MatMul", self_attn_prefix);
        let v_add = format!("{}/v_proj/Add", self_attn_prefix);
        let attn_scores = format!("{}/MatMul", self_attn_prefix);
        let attn_softmax = format!("{}/Softmax", self_attn_prefix);
        let attn_ctx = format!("{}/MatMul_1", self_attn_prefix);
        let out_matmul = format!("{}/out_proj/MatMul", self_attn_prefix);
        let out_add = format!("{}/out_proj/Add", self_attn_prefix);

        // Find which layer names exist for Q/K/V (some might not have bias)
        let q_src = if self.has_layer(&q_add) {
            &q_add
        } else {
            &q_matmul
        };
        let k_src = if self.has_layer(&k_add) {
            &k_add
        } else {
            &k_matmul
        };
        let v_src = if self.has_layer(&v_add) {
            &v_add
        } else {
            &v_matmul
        };

        let hidden_dim = self.hidden_dim;
        let head_dim = self.structure.head_dim;
        let qkv_target_shape = vec![0, 0, self.num_heads as i64, head_dim as i64];
        let qkv_perm = vec![0, 2, 1, 3];

        let mut graph = GraphNetwork::new();
        let mut tensor_to_node: HashMap<String, String> = HashMap::new();

        // Build layer name set: norm1 chain + Q/K/V projections.
        // out_matmul/out_add handled separately (must use ctx_reshape as input).
        let mut all_attn_layers: std::collections::HashSet<String> = [
            &norm1_out, &q_matmul, &q_add, &k_matmul, &k_add, &v_matmul, &v_add,
        ]
        .iter()
        .filter(|s| self.has_layer(s))
        .map(|s| (*s).clone())
        .collect();
        let norm1_prefix = if prefix.is_empty() {
            "/norm1/".to_string()
        } else {
            format!("{}/norm1/", prefix)
        };
        all_attn_layers.extend(
            self.model
                .network
                .layers
                .iter()
                .filter(|l| l.name.starts_with(&norm1_prefix))
                .map(|l| l.name.clone()),
        );

        // Pre-evaluate constant chains (#3317).
        let evaluated_constants = self.evaluate_model_constants();

        for spec in &self.model.network.layers {
            if !all_attn_layers.contains(&spec.name) {
                continue;
            }
            let layer = self.convert_layer_with_constants(spec, &evaluated_constants)?;
            let input_nodes = self.find_input_nodes_decoder(
                spec,
                &layer,
                &tensor_to_node,
                &self.model.constant_tensors,
                &evaluated_constants,
            );
            graph.try_add_node(GraphNode::new(spec.name.clone(), layer, input_nodes))?;
            if let Some(output_name) = spec.outputs.first() {
                tensor_to_node.insert(output_name.clone(), spec.name.clone());
            }
            // Insert shape transform nodes after Q/K/V projections (#2686)
            if spec.name == *q_src || spec.name == *k_src || spec.name == *v_src {
                Self::add_qkv_shape_transform(
                    &mut graph,
                    &spec.name,
                    &qkv_target_shape,
                    &qkv_perm,
                )?;
            }
        }

        // Attention core: Q@K^T (scaled) → CausalSoftmax → Attn@V
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_transpose = format!("{}::__transpose_bhsd", q_src);
        let k_transpose = format!("{}::__transpose_bhsd", k_src);
        let v_transpose = format!("{}::__transpose_bhsd", v_src);

        graph.try_add_node(GraphNode::new(
            attn_scores.clone(),
            Layer::MatMul(MatMulLayer::new(true, Some(scale))),
            vec![q_transpose, k_transpose],
        ))?;
        graph.try_add_node(GraphNode::new(
            attn_softmax.clone(),
            Layer::CausalSoftmax(CausalSoftmaxLayer::new(-1)),
            vec![attn_scores],
        ))?;
        graph.try_add_node(GraphNode::new(
            attn_ctx.clone(),
            Layer::MatMul(MatMulLayer::new(false, None)),
            vec![attn_softmax, v_transpose],
        ))?;

        // Transpose and reshape back to (B, S, hidden_dim)
        let ctx_transpose_name = format!("{}::__transpose_bshd", attn_ctx);
        let ctx_reshape_name = format!("{}::__reshape_bsd", attn_ctx);
        graph.try_add_node(GraphNode::new(
            ctx_transpose_name.clone(),
            Layer::Transpose(TransposeLayer::new(vec![0, 2, 1, 3])),
            vec![attn_ctx],
        ))?;
        graph.try_add_node(GraphNode::new(
            ctx_reshape_name.clone(),
            Layer::Reshape(ReshapeLayer::new(vec![0, 0, hidden_dim as i64])),
            vec![ctx_transpose_name],
        ))?;

        self.add_output_projection(
            &mut graph,
            &mut tensor_to_node,
            &out_matmul,
            &out_add,
            &ctx_reshape_name,
        )?;

        info!(
            "Built causal attention subgraph for block {} with {} nodes",
            block_index,
            graph.num_nodes()
        );

        Ok(graph)
    }

    /// Find input node names for a layer in decoder structure.
    ///
    /// For each non-weight, non-constant, non-evaluated input:
    /// - If in tensor_to_node: use the producing node name
    /// - Otherwise: use "_input" (external input)
    ///
    /// Evaluated constants are filtered because their values are embedded
    /// directly into ConcatLayer via `convert_concat_with_evaluated` (#3317).
    pub(super) fn find_input_nodes_decoder(
        &self,
        spec: &LayerSpec,
        layer: &Layer,
        tensor_to_node: &HashMap<String, String>,
        constant_tensors: &std::collections::HashSet<String>,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Vec<String> {
        let mut input_nodes = Vec::new();

        let is_concat = matches!(layer, Layer::Concat(_));

        // Filter out weight inputs - they're handled by the Layer itself.
        // For non-Concat ops, also filter out constant tensors.
        // Skip evaluated constants — their values are embedded in the layer (#3317).
        let activation_inputs: Vec<&String> = spec
            .inputs
            .iter()
            .filter(|name| {
                if self.model.weights.contains_key(name) {
                    return false;
                }
                // Skip evaluated constants — values embedded in ConcatLayer (#3317).
                if evaluated_constants.contains_key(*name) {
                    return false;
                }
                if is_concat {
                    return true;
                }
                !constant_tensors.contains(*name)
            })
            .collect();

        if is_concat {
            for input_tensor in activation_inputs.iter() {
                if let Some(node_name) = tensor_to_node.get(*input_tensor) {
                    input_nodes.push(node_name.clone());
                } else {
                    input_nodes.push("_input".to_string());
                }
            }
        } else if layer.is_binary() {
            // Binary ops need two inputs
            for input_tensor in activation_inputs.iter().take(2) {
                if let Some(node_name) = tensor_to_node.get(*input_tensor) {
                    input_nodes.push(node_name.clone());
                } else {
                    // External input
                    input_nodes.push("_input".to_string());
                }
            }
        } else {
            // Unary ops need one input
            if let Some(input_tensor) = activation_inputs.first() {
                if let Some(node_name) = tensor_to_node.get(*input_tensor) {
                    input_nodes.push(node_name.clone());
                } else {
                    // External input
                    input_nodes.push("_input".to_string());
                }
            }
        }

        input_nodes
    }

    /// Convert a layer, using `convert_concat_with_evaluated` for Concat (#3317).
    pub(super) fn convert_layer_with_constants(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Result<Layer> {
        if spec.layer_type == LayerType::Concat {
            self.model
                .convert_context()
                .convert_concat_with_evaluated(spec, evaluated_constants)
        } else {
            self.model.convert_layer(spec)
        }
    }

    /// Add reshape (B,S,H,D) + transpose (B,H,S,D) nodes after a Q/K/V projection.
    fn add_qkv_shape_transform(
        graph: &mut GraphNetwork,
        proj_name: &str,
        target_shape: &[i64],
        perm: &[usize],
    ) -> Result<()> {
        let reshape = format!("{}::__reshape_bshd", proj_name);
        let transpose = format!("{}::__transpose_bhsd", proj_name);
        graph.try_add_node(GraphNode::new(
            reshape.clone(),
            Layer::Reshape(ReshapeLayer::new(target_shape.to_vec())),
            vec![proj_name.to_string()],
        ))?;
        graph.try_add_node(GraphNode::new(
            transpose,
            Layer::Transpose(TransposeLayer::new(perm.to_vec())),
            vec![reshape],
        ))?;
        Ok(())
    }

    /// Add output projection (MatMul + optional bias Add) to a graph.
    fn add_output_projection(
        &self,
        graph: &mut GraphNetwork,
        tensor_to_node: &mut HashMap<String, String>,
        out_matmul: &str,
        out_add: &str,
        input_node: &str,
    ) -> Result<()> {
        if self.has_layer(out_matmul) {
            let spec = self
                .model
                .network
                .layers
                .iter()
                .find(|l| l.name == out_matmul)
                .ok_or_else(|| NyError::InvalidSpec("out_proj/MatMul not found".into()))?;
            let layer = self.model.convert_layer(spec)?;
            graph.try_add_node(GraphNode::new(
                out_matmul.to_string(),
                layer,
                vec![input_node.to_string()],
            ))?;
            tensor_to_node.insert(spec.outputs[0].clone(), out_matmul.to_string());
        }
        if self.has_layer(out_add) {
            let spec = self
                .model
                .network
                .layers
                .iter()
                .find(|l| l.name == out_add)
                .ok_or_else(|| NyError::InvalidSpec("out_proj/Add not found".into()))?;
            let layer = self.model.convert_layer(spec)?;
            graph.try_add_node(GraphNode::new(
                out_add.to_string(),
                layer,
                vec![out_matmul.to_string()],
            ))?;
            graph.set_output(out_add);
        } else {
            graph.set_output(out_matmul);
        }
        Ok(())
    }

    /// Cross-attention reconstruction is unavailable.
    ///
    /// `GraphNetwork` has a single external-input propagation contract, while
    /// cross-attention requires independent decoder and encoder inputs. Return
    /// no graph until a sound multi-input representation exists.
    pub fn cross_attention_subgraph(&self, block_index: usize) -> Result<GraphNetwork> {
        let block_info = self.block_info(block_index)?;

        if !block_info.has_cross_attention {
            return Err(NyError::InvalidSpec(format!(
                "decoder block {block_index} does not have cross-attention"
            )));
        }
        Err(NyError::UnsupportedConfiguration(
            "cross-attention subgraph extraction requires two independent external inputs, but \
             GraphNetwork currently has a single-input propagation contract; no graph or bounds \
             were produced"
                .to_string(),
        ))
    }
}
