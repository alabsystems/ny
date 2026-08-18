// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{AttributeValue, LayerSpec};
use ny_core::{LayerType, NyError, Result};
use ny_propagate::{
    layers::{MatMulLayer, ReshapeLayer, TransposeLayer},
    GraphNetwork, GraphNode, Layer,
};
use tracing::{debug, info};

use super::super::helpers::layer_any_name_matches;
use super::super::loader::block_index::parse_block_index;
use super::super::model::WhisperModel;

#[cfg(test)]
pub(crate) struct AttentionSubgraphArtifacts {
    pub graph: GraphNetwork,
    pub scores_node: String,
    pub softmax_node: String,
    pub context_node: String,
    pub output_node: String,
}

/// Controls where the attention subgraph starts.
///
/// - `BlockInput`: full attention graph including LayerNorm (today's default)
/// - `LayerNormOutput`: suffix graph starting after the attention LayerNorm,
///   with the unresolved `ln_output` tensor mapping to `_input`
///
/// Part of #318: shared-source prefix cut for zonotope attention.
enum AttentionGraphRoot {
    BlockInput,
    #[cfg(test)]
    LayerNormOutput,
}

/// Attention nodes for one encoder block, discovered structurally
/// (op-type + topology + weight-name tokens; no exporter-specific node names).
///
/// Shared by the attention subgraph builder and `encoder_layer_graph_full` so
/// both survive exporter renames (e.g. torch dynamo's `node_view` /
/// `node_MatMul_N` convention vs the legacy `{prefix}/query/MatMul` names).
pub(crate) struct DiscoveredAttentionNodes<'a> {
    pub(crate) attn_ln: &'a LayerSpec,
    pub(crate) q_matmul: &'a LayerSpec,
    pub(crate) q_add: Option<&'a LayerSpec>,
    pub(crate) k_matmul: &'a LayerSpec,
    pub(crate) k_add: Option<&'a LayerSpec>,
    pub(crate) v_matmul: &'a LayerSpec,
    pub(crate) v_add: Option<&'a LayerSpec>,
    pub(crate) attn_scores: &'a LayerSpec,
    pub(crate) attn_softmax: &'a LayerSpec,
    pub(crate) attn_ctx: &'a LayerSpec,
    pub(crate) out_matmul: &'a LayerSpec,
    pub(crate) out_add: Option<&'a LayerSpec>,
}

/// Exporter plumbing that has been proven equivalent to NY's synthetic,
/// sequence-agnostic Whisper attention wiring.
///
/// The set is owned so both the full-block and attention-subgraph builders can
/// consume the same validation result without retaining model borrows.
#[derive(Debug)]
pub(crate) struct AttentionRewritePlan {
    pub(crate) replaced_nodes: std::collections::HashSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionPlumbingRole {
    Query,
    Key,
    Value,
    ScoresToSoftmax,
    SoftmaxToContext,
    ContextToOutput,
}

struct AttentionPlumbingTrace<'a> {
    stop: String,
    /// Traversed nodes in output-to-source (backward) order.
    nodes: Vec<&'a LayerSpec>,
    scale: f64,
}

impl<'a> DiscoveredAttentionNodes<'a> {
    /// Q projection source node (bias Add when present, else the MatMul).
    pub(crate) fn q_src(&self) -> &'a LayerSpec {
        self.q_add.unwrap_or(self.q_matmul)
    }

    /// K projection source node (bias Add when present, else the MatMul).
    pub(crate) fn k_src(&self) -> &'a LayerSpec {
        self.k_add.unwrap_or(self.k_matmul)
    }

    /// V projection source node (bias Add when present, else the MatMul).
    pub(crate) fn v_src(&self) -> &'a LayerSpec {
        self.v_add.unwrap_or(self.v_matmul)
    }
}

fn multiply_scale(acc: &mut f64, factor: f64, label: &str) -> Result<()> {
    if !factor.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "attention plumbing {label} scale is non-finite: {factor}"
        )));
    }
    *acc *= factor;
    if !acc.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "attention plumbing {label} scale product is non-finite"
        )));
    }
    Ok(())
}

impl WhisperModel {
    /// Extract the attention subgraph (without the residual Add).
    ///
    /// This extracts: attn_ln → Q/K/V projections → attention core → output projection → bias Add
    /// Output is the attention delta to be added to the residual.
    ///
    /// For compositional verification, this lets us bound the attention contribution
    /// separately from the residual path.
    pub fn attention_subgraph(&self, index: usize) -> Result<GraphNetwork> {
        let (graph, _, _, _, _) =
            self.build_attention_subgraph_with_root(index, AttentionGraphRoot::BlockInput)?;
        Ok(graph)
    }

    #[cfg(test)]
    pub(crate) fn attention_subgraph_artifacts(
        &self,
        index: usize,
    ) -> Result<AttentionSubgraphArtifacts> {
        let (graph, scores_node, softmax_node, context_node, output_node) =
            self.build_attention_subgraph_with_root(index, AttentionGraphRoot::BlockInput)?;
        Ok(AttentionSubgraphArtifacts {
            graph,
            scores_node,
            softmax_node,
            context_node,
            output_node,
        })
    }

    #[cfg(test)]
    pub(crate) fn attention_suffix_subgraph_artifacts_from_layernorm_output(
        &self,
        index: usize,
    ) -> Result<AttentionSubgraphArtifacts> {
        let (graph, scores_node, softmax_node, context_node, output_node) =
            self.build_attention_subgraph_with_root(index, AttentionGraphRoot::LayerNormOutput)?;
        Ok(AttentionSubgraphArtifacts {
            graph,
            scores_node,
            softmax_node,
            context_node,
            output_node,
        })
    }

    fn attention_activation_inputs<'a>(&self, spec: &'a LayerSpec) -> Vec<&'a str> {
        spec.inputs
            .iter()
            .filter(|name| {
                !name.is_empty()
                    && !self.model.weights.contains_key(name)
                    && !self.model.constant_tensors.contains(*name)
            })
            .map(String::as_str)
            .collect()
    }

    /// Prove that the exporter's attention plumbing is exactly the layout and
    /// scaling that NY's sequence-agnostic synthetic wiring implements.
    ///
    /// This is the single gate for both `encoder_layer_graph_full` and the
    /// attention-subgraph builders. It deliberately rejects unfamiliar but
    /// potentially valid exports: falling back/erroring is safe; silently
    /// replacing an unproven reshape, permutation, division, or fan-out is not.
    pub(crate) fn attention_rewrite_plan(
        &self,
        index: usize,
        nodes: &DiscoveredAttentionNodes<'_>,
    ) -> Result<AttentionRewritePlan> {
        let mut global_layer_names = std::collections::HashSet::new();
        for spec in &self.model.network.layers {
            if spec.name.is_empty() {
                return Err(NyError::InvalidSpec(
                    "attention rewrite requires every layer to have a nonempty name; name-keyed \
                     replacement identity would otherwise be ambiguous"
                        .to_string(),
                ));
            }
            if !global_layer_names.insert(spec.name.as_str()) {
                return Err(NyError::InvalidSpec(format!(
                    "attention rewrite requires globally unique nonempty layer names; duplicate \
                     name '{}' would make replacement identity ambiguous",
                    spec.name
                )));
            }
        }
        if self.num_heads == 0
            || self.hidden_dim == 0
            || !self.hidden_dim.is_multiple_of(self.num_heads)
        {
            return Err(NyError::InvalidSpec(format!(
                "attention rewrite requires non-zero hidden_dim divisible by non-zero num_heads; \
                 got hidden_dim={} num_heads={}",
                self.hidden_dim, self.num_heads
            )));
        }
        let head_dim = self.hidden_dim / self.num_heads;
        let block_layers = self.block_layers_for_index(index)?;
        let mut output_to_spec: std::collections::HashMap<&str, &LayerSpec> =
            std::collections::HashMap::new();
        let mut output_consumers: std::collections::HashMap<&str, Vec<&str>> =
            std::collections::HashMap::new();
        let mut spec_by_name: std::collections::HashMap<&str, &LayerSpec> =
            std::collections::HashMap::new();
        for spec in &block_layers {
            spec_by_name.insert(spec.name.as_str(), spec);
            for output in &spec.outputs {
                if !output.is_empty() {
                    if output_to_spec.insert(output.as_str(), spec).is_some() {
                        return Err(NyError::InvalidSpec(format!(
                            "attention block {index} has multiple producers for tensor '{output}'"
                        )));
                    }
                }
            }
        }
        // Fan-out is a whole-model property. A block-local rewrite must not
        // erase a plumbing value that also feeds an unscoped/out-of-block
        // consumer omitted by block detection.
        for spec in &self.model.network.layers {
            for input in &spec.inputs {
                if !input.is_empty() {
                    output_consumers
                        .entry(input.as_str())
                        .or_default()
                        .push(spec.name.as_str());
                }
            }
        }

        // The synthetic core assumes ordinary ONNX A@B MatMuls. Any fused
        // transpose/scale attribute would be applied differently after rewrite.
        self.validate_unfused_attention_matmul(nodes.attn_scores, "scores")?;
        self.validate_unfused_attention_matmul(nodes.attn_ctx, "context")?;

        let score_stops: std::collections::HashSet<&str> =
            [nodes.q_src().name.as_str(), nodes.k_src().name.as_str()]
                .into_iter()
                .collect();
        let score_inputs = self.attention_activation_inputs(nodes.attn_scores);
        if score_inputs.len() != 2 {
            return Err(NyError::InvalidSpec(format!(
                "attention scores '{}' must have exactly two activation operands, got {}",
                nodes.attn_scores.name,
                score_inputs.len()
            )));
        }
        let q_trace =
            self.trace_attention_plumbing(score_inputs[0], &score_stops, &output_to_spec)?;
        let k_trace =
            self.trace_attention_plumbing(score_inputs[1], &score_stops, &output_to_spec)?;
        if q_trace.stop != nodes.q_src().name || k_trace.stop != nodes.k_src().name {
            return Err(NyError::InvalidSpec(format!(
                "attention scores '{}' operand order is not Q then K (resolved '{}' then '{}')",
                nodes.attn_scores.name, q_trace.stop, k_trace.stop
            )));
        }
        self.validate_attention_trace_layout(&q_trace, AttentionPlumbingRole::Query, head_dim)?;
        self.validate_attention_trace_layout(&k_trace, AttentionPlumbingRole::Key, head_dim)?;

        let softmax_inputs = self.attention_activation_inputs(nodes.attn_softmax);
        if softmax_inputs.len() != 1 {
            return Err(NyError::InvalidSpec(format!(
                "attention softmax '{}' must have exactly one activation operand, got {}",
                nodes.attn_softmax.name,
                softmax_inputs.len()
            )));
        }
        let softmax_stops: std::collections::HashSet<&str> =
            [nodes.attn_scores.name.as_str()].into_iter().collect();
        let scores_to_softmax =
            self.trace_attention_plumbing(softmax_inputs[0], &softmax_stops, &output_to_spec)?;
        if scores_to_softmax.stop != nodes.attn_scores.name {
            return Err(NyError::InvalidSpec(format!(
                "attention softmax '{}' does not resolve to scores '{}'",
                nodes.attn_softmax.name, nodes.attn_scores.name
            )));
        }
        self.validate_attention_trace_layout(
            &scores_to_softmax,
            AttentionPlumbingRole::ScoresToSoftmax,
            head_dim,
        )?;

        let context_inputs = self.attention_activation_inputs(nodes.attn_ctx);
        if context_inputs.len() != 2 {
            return Err(NyError::InvalidSpec(format!(
                "attention context '{}' must have exactly two activation operands, got {}",
                nodes.attn_ctx.name,
                context_inputs.len()
            )));
        }
        let context_stops: std::collections::HashSet<&str> = [
            nodes.attn_softmax.name.as_str(),
            nodes.v_src().name.as_str(),
        ]
        .into_iter()
        .collect();
        let softmax_to_context =
            self.trace_attention_plumbing(context_inputs[0], &context_stops, &output_to_spec)?;
        let v_trace =
            self.trace_attention_plumbing(context_inputs[1], &context_stops, &output_to_spec)?;
        if softmax_to_context.stop != nodes.attn_softmax.name || v_trace.stop != nodes.v_src().name
        {
            return Err(NyError::InvalidSpec(format!(
                "attention context '{}' operand order is not Softmax then V (resolved '{}' then '{}')",
                nodes.attn_ctx.name, softmax_to_context.stop, v_trace.stop
            )));
        }
        self.validate_attention_trace_layout(
            &softmax_to_context,
            AttentionPlumbingRole::SoftmaxToContext,
            head_dim,
        )?;
        self.validate_attention_trace_layout(&v_trace, AttentionPlumbingRole::Value, head_dim)?;

        let out_inputs = self.attention_activation_inputs(nodes.out_matmul);
        if out_inputs.len() != 1 {
            return Err(NyError::InvalidSpec(format!(
                "attention output projection '{}' must have exactly one activation operand, got {}",
                nodes.out_matmul.name,
                out_inputs.len()
            )));
        }
        let out_stops: std::collections::HashSet<&str> =
            [nodes.attn_ctx.name.as_str()].into_iter().collect();
        let context_to_output =
            self.trace_attention_plumbing(out_inputs[0], &out_stops, &output_to_spec)?;
        if context_to_output.stop != nodes.attn_ctx.name {
            return Err(NyError::InvalidSpec(format!(
                "attention output projection '{}' does not resolve to context '{}'",
                nodes.out_matmul.name, nodes.attn_ctx.name
            )));
        }
        self.validate_attention_trace_layout(
            &context_to_output,
            AttentionPlumbingRole::ContextToOutput,
            head_dim,
        )?;

        let traces = [
            &q_trace,
            &k_trace,
            &scores_to_softmax,
            &softmax_to_context,
            &v_trace,
            &context_to_output,
        ];
        let mut scores_scale = 1.0f64;
        for (label, trace) in [
            ("query", &q_trace),
            ("key", &k_trace),
            ("scores-to-softmax", &scores_to_softmax),
        ] {
            multiply_scale(&mut scores_scale, trace.scale, label)?;
        }
        // Match the exact arithmetic used by the synthetic MatMul constructor,
        // not an idealized f64 1/sqrt(d). Split legacy Q/K f32 factors can
        // accumulate one extra rounding, so permit at most two neighboring f32
        // ULPs — enough for that proven exporter form, far tighter than the old
        // 1e-4 relative tolerance.
        let synthetic_scale = 1.0f32 / (head_dim as f32).sqrt();
        if !synthetic_scale.is_finite() || synthetic_scale <= 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "attention expected score scale is invalid for head_dim {head_dim}"
            )));
        }
        let expected_scale = f64::from(synthetic_scale);
        let bits = synthetic_scale.to_bits();
        let next_ulp = f64::from(f32::from_bits(bits + 1)) - expected_scale;
        let prev_ulp = expected_scale - f64::from(f32::from_bits(bits - 1));
        let tolerance = 2.0 * next_ulp.max(prev_ulp);
        let error = (scores_scale - expected_scale).abs();
        if !error.is_finite() || error > tolerance {
            return Err(NyError::InvalidSpec(format!(
                "attention scores plumbing applies scale {scores_scale:.9e}, but synthetic core \
                 applies f32 1/sqrt(head_dim)={expected_scale:.9e} (abs err {error:.3e}, \
                 two-ULP tolerance {tolerance:.3e})"
            )));
        }

        let mut unit_scale = 1.0f64;
        for (label, trace) in [
            ("softmax-to-context", &softmax_to_context),
            ("value", &v_trace),
            ("context-to-output", &context_to_output),
        ] {
            multiply_scale(&mut unit_scale, trace.scale, label)?;
        }
        // The synthetic value/context/output path applies no scalar. Constants
        // are stored as f32, so bound accumulated exporter rounding to two f32
        // ULPs at 1.0 instead of the old, materially looser 1e-6 threshold.
        let unit_tolerance = 2.0 * f64::from(f32::EPSILON);
        let unit_error = (unit_scale - 1.0).abs();
        if !unit_error.is_finite() || unit_error > unit_tolerance {
            return Err(NyError::InvalidSpec(format!(
                "attention value/output plumbing applies scale {unit_scale:.9e} that synthetic \
                 wiring would drop (abs err {unit_error:.3e}, two-ULP unit tolerance \
                 {unit_tolerance:.3e})"
            )));
        }

        let mut replaced_nodes = std::collections::HashSet::new();
        for trace in traces {
            for spec in &trace.nodes {
                replaced_nodes.insert(spec.name.clone());
            }
        }
        let mut model_spec_by_name: std::collections::HashMap<&str, &LayerSpec> =
            std::collections::HashMap::new();
        for spec in &self.model.network.layers {
            model_spec_by_name.insert(spec.name.as_str(), spec);
        }
        // A consumer that only reads the *shape* of a plumbing tensor is not a
        // live data consumer: a Shape op reads dimensions, never values, and the
        // loader has already folded such a Shape-of-activation to a constant.
        // That folded constant is immutable, so deleting the runtime producer
        // cannot change it. (In the torch-dynamo export these Shape reads feed
        // only the target operands of the very reshapes being replaced.)
        let is_folded_shape_consumer = |consumer: &str| -> bool {
            model_spec_by_name.get(consumer).is_some_and(|spec| {
                spec.layer_type == LayerType::Shape
                    && spec.outputs.iter().any(|out| !out.is_empty())
                    && spec
                        .outputs
                        .iter()
                        .filter(|out| !out.is_empty())
                        .all(|out| {
                            self.model.weights.contains_key(out)
                                || self.model.constant_tensors.contains(out)
                        })
            })
        };
        for name in &replaced_nodes {
            let spec = *spec_by_name.get(name.as_str()).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "attention replacement node '{name}' is not in block {index}"
                ))
            })?;
            let outputs: Vec<&str> = spec
                .outputs
                .iter()
                .filter(|output| !output.is_empty())
                .map(String::as_str)
                .collect();
            if outputs.len() != 1 {
                return Err(NyError::InvalidSpec(format!(
                    "attention replacement node '{}' must have exactly one output, got {}",
                    spec.name,
                    outputs.len()
                )));
            }
            if self
                .model
                .network
                .outputs
                .iter()
                .any(|output| output.name == outputs[0])
            {
                return Err(NyError::InvalidSpec(format!(
                    "attention replacement node '{}' output '{}' is a network output; synthetic \
                     deletion would change the externally visible model contract",
                    spec.name, outputs[0]
                )));
            }
            let consumers = output_consumers
                .get(outputs[0])
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let data_consumers: Vec<&str> = consumers
                .iter()
                .copied()
                .filter(|consumer| !is_folded_shape_consumer(consumer))
                .collect();
            if data_consumers.len() != 1 {
                return Err(NyError::InvalidSpec(format!(
                    "attention replacement node '{}' output '{}' has {} live data consumers {:?}; \
                     synthetic deletion requires exactly one (folded Shape reads excluded)",
                    spec.name,
                    outputs[0],
                    data_consumers.len(),
                    data_consumers
                )));
            }
        }

        Ok(AttentionRewritePlan { replaced_nodes })
    }

    fn validate_unfused_attention_matmul(&self, spec: &LayerSpec, role: &str) -> Result<()> {
        if spec.inputs.len() != 2 {
            return Err(NyError::InvalidSpec(format!(
                "attention {role} MatMul '{}' must have exactly two inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }
        for (name, value) in &spec.attributes {
            let proven_default = name == "transpose_b"
                && (matches!(value, AttributeValue::Int(0))
                    || matches!(value, AttributeValue::Float(v) if *v == 0.0));
            if !proven_default {
                return Err(NyError::InvalidSpec(format!(
                    "attention {role} MatMul '{}' has unproven attribute {name}={value:?}; \
                     synthetic replacement accepts only an explicit default transpose_b=0",
                    spec.name
                )));
            }
        }
        Ok(())
    }

    fn trace_attention_plumbing<'a>(
        &self,
        start: &str,
        stops: &std::collections::HashSet<&str>,
        output_to_spec: &std::collections::HashMap<&str, &'a LayerSpec>,
    ) -> Result<AttentionPlumbingTrace<'a>> {
        let mut current = start.to_string();
        let mut visited = std::collections::HashSet::new();
        let mut nodes = Vec::new();
        let mut scale = 1.0f64;
        loop {
            let spec = *output_to_spec.get(current.as_str()).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "attention plumbing from '{start}' reached tensor '{current}' with no \
                     producer inside the block"
                ))
            })?;
            if stops.contains(spec.name.as_str()) {
                return Ok(AttentionPlumbingTrace {
                    stop: spec.name.clone(),
                    nodes,
                    scale,
                });
            }
            if !visited.insert(spec.name.clone()) {
                return Err(NyError::InvalidSpec(format!(
                    "attention plumbing from '{start}' revisited node '{}'",
                    spec.name
                )));
            }

            let next = match &spec.layer_type {
                LayerType::Reshape => {
                    let data = spec
                        .inputs
                        .first()
                        .filter(|name| !name.is_empty())
                        .ok_or_else(|| {
                            NyError::InvalidSpec(format!(
                                "attention Reshape '{}' has no data operand at input 0",
                                spec.name
                            ))
                        })?;
                    data.clone()
                }
                LayerType::Transpose => {
                    let nonempty = spec.inputs.iter().filter(|name| !name.is_empty()).count();
                    if nonempty != 1 {
                        return Err(NyError::InvalidSpec(format!(
                            "attention Transpose '{}' must have one data operand, got {nonempty}",
                            spec.name
                        )));
                    }
                    spec.inputs[0].clone()
                }
                LayerType::Mul | LayerType::Div => {
                    if !spec.attributes.is_empty() {
                        return Err(NyError::InvalidSpec(format!(
                            "attention {:?} '{}' has unproven attributes {:?}",
                            spec.layer_type, spec.name, spec.attributes
                        )));
                    }
                    let nonempty: Vec<(usize, &str)> = spec
                        .inputs
                        .iter()
                        .enumerate()
                        .filter(|(_, name)| !name.is_empty())
                        .map(|(index, name)| (index, name.as_str()))
                        .collect();
                    if nonempty.len() != 2 {
                        return Err(NyError::InvalidSpec(format!(
                            "attention {:?} '{}' must have exactly two operands, got {}",
                            spec.layer_type,
                            spec.name,
                            nonempty.len()
                        )));
                    }
                    let constants: Vec<(usize, &str, f64)> = nonempty
                        .iter()
                        .filter_map(|(index, name)| {
                            self.model.weights.get(name).and_then(|tensor| {
                                (tensor.len() == 1).then(|| {
                                    (*index, *name, f64::from(*tensor.iter().next().unwrap()))
                                })
                            })
                        })
                        .collect();
                    if constants.len() != 1 {
                        return Err(NyError::InvalidSpec(format!(
                            "attention {:?} '{}' must have exactly one known scalar constant, got {}",
                            spec.layer_type,
                            spec.name,
                            constants.len()
                        )));
                    }
                    let (constant_index, _, value) = constants[0];
                    let (data_index, data) = nonempty
                        .iter()
                        .copied()
                        .find(|(index, _)| *index != constant_index)
                        .expect("two operands and one constant imply one data operand");
                    if !value.is_finite() {
                        return Err(NyError::InvalidSpec(format!(
                            "attention {:?} '{}' scalar constant must be finite, got {value}",
                            spec.layer_type, spec.name
                        )));
                    }
                    if spec.layer_type == LayerType::Div {
                        if data_index != 0 || constant_index != 1 {
                            return Err(NyError::InvalidSpec(format!(
                                "attention Div '{}' must be activation / constant; reverse or \
                                 reordered division is not replaceable",
                                spec.name
                            )));
                        }
                        if value == 0.0 {
                            return Err(NyError::InvalidSpec(format!(
                                "attention Div '{}' divides by zero",
                                spec.name
                            )));
                        }
                        scale /= value;
                    } else {
                        scale *= value;
                    }
                    if !scale.is_finite() {
                        return Err(NyError::InvalidSpec(format!(
                            "attention plumbing scale became non-finite at '{}'",
                            spec.name
                        )));
                    }
                    data.to_string()
                }
                other => {
                    return Err(NyError::InvalidSpec(format!(
                        "attention plumbing node '{}' has unreplaceable op {other:?}",
                        spec.name
                    )));
                }
            };
            nodes.push(spec);
            current = next;
        }
    }

    fn validate_attention_trace_layout(
        &self,
        trace: &AttentionPlumbingTrace<'_>,
        role: AttentionPlumbingRole,
        head_dim: usize,
    ) -> Result<()> {
        let layout: Vec<&LayerSpec> = trace
            .nodes
            .iter()
            .rev()
            .copied()
            .filter(|spec| matches!(spec.layer_type, LayerType::Reshape | LayerType::Transpose))
            .collect();
        match role {
            AttentionPlumbingRole::ScoresToSoftmax | AttentionPlumbingRole::SoftmaxToContext => {
                if !layout.is_empty() {
                    return Err(NyError::InvalidSpec(format!(
                        "attention {role:?} path contains unproven layout nodes {:?}",
                        layout.iter().map(|spec| &spec.name).collect::<Vec<_>>()
                    )));
                }
            }
            AttentionPlumbingRole::Query
            | AttentionPlumbingRole::Key
            | AttentionPlumbingRole::Value => {
                // The head-split Reshape must come first; the synthetic wiring
                // rebuilds it seq-agnostically. Everything after it must be a
                // pure permutation of the {B,S,heads,head_dim} atomic axes —
                // expressed either as plain Transposes (torchscript) or as a
                // transpose realised through a flatten -> transpose -> unflatten
                // of batch*heads (torch dynamo's scaled_dot_product_attention
                // decomposition). We track the atomic axes through both forms
                // and require the net permutation to equal the synthetic layout,
                // which proves the intervening merge/unmerge Reshapes only
                // *reorder* whole head axes and never reshuffle within one.
                if layout
                    .first()
                    .is_none_or(|spec| spec.layer_type != LayerType::Reshape)
                {
                    return Err(NyError::InvalidSpec(format!(
                        "attention {role:?} path must start with one head-split Reshape"
                    )));
                }
                self.validate_qkv_reshape(layout[0], head_dim)?;
                let source = self.concrete_attention_reshape_source_shape(layout[0], "Q/K/V", 3)?;
                let heads = i64::try_from(self.num_heads).map_err(|_| {
                    NyError::InvalidSpec("Whisper num_heads does not fit i64".to_string())
                })?;
                let head_dim_i = i64::try_from(head_dim).map_err(|_| {
                    NyError::InvalidSpec("Whisper head_dim does not fit i64".to_string())
                })?;
                // Atomic axes of the head-split output [B, S, heads, head_dim],
                // each carried as (axis-id, concrete size).
                let mut groups: Vec<Vec<(usize, i64)>> = vec![
                    vec![(0usize, source[0])],
                    vec![(1, source[1])],
                    vec![(2, heads)],
                    vec![(3, head_dim_i)],
                ];
                for spec in &layout[1..] {
                    groups = match &spec.layer_type {
                        LayerType::Transpose => {
                            let identity: Vec<usize> = (0..groups.len()).collect();
                            let perm = self.apply_attention_transpose(spec, &identity)?;
                            perm.into_iter().map(|axis| groups[axis].clone()).collect()
                        }
                        LayerType::Reshape => self.regroup_attention_atoms(spec, groups)?,
                        other => {
                            return Err(NyError::InvalidSpec(format!(
                                "attention {role:?} layout has an unexpected op {other:?} at '{}'",
                                spec.name
                            )));
                        }
                    };
                }
                let mut axes = Vec::with_capacity(groups.len());
                for group in &groups {
                    match group.as_slice() {
                        [(atom, _)] => axes.push(*atom),
                        _ => {
                            return Err(NyError::InvalidSpec(format!(
                                "attention {role:?} plumbing does not fully unflatten the head \
                                 axes; composite axis {group:?} remains, so it is not a pure \
                                 permutation of the synthetic layout"
                            )));
                        }
                    }
                }
                let expected = match role {
                    AttentionPlumbingRole::Key => vec![0, 2, 3, 1], // B,H,D,S
                    _ => vec![0, 2, 1, 3],                          // B,H,S,D
                };
                if axes != expected {
                    return Err(NyError::InvalidSpec(format!(
                        "attention {role:?} transpose composition is {axes:?}, expected {expected:?}"
                    )));
                }
            }
            AttentionPlumbingRole::ContextToOutput => {
                if layout.len() < 2
                    || layout
                        .last()
                        .is_none_or(|spec| spec.layer_type != LayerType::Reshape)
                    || layout[..layout.len() - 1]
                        .iter()
                        .any(|spec| spec.layer_type != LayerType::Transpose)
                {
                    return Err(NyError::InvalidSpec(
                        "attention context-to-output path must contain Transpose node(s) followed \
                         by exactly one Reshape"
                            .to_string(),
                    ));
                }
                let mut axes = vec![0usize, 1, 2, 3]; // B,H,S,D
                for spec in &layout[..layout.len() - 1] {
                    axes = self.apply_attention_transpose(spec, &axes)?;
                }
                let expected = vec![0, 2, 1, 3]; // B,S,H,D before flattening H,D
                if axes != expected {
                    return Err(NyError::InvalidSpec(format!(
                        "attention context transpose composition is {axes:?}, expected {expected:?}"
                    )));
                }
                self.validate_context_reshape(layout[layout.len() - 1])?;
            }
        }
        Ok(())
    }

    /// Re-partition the tracked atomic axes to match a Reshape's concrete output
    /// shape, proving the Reshape only *merges or splits* whole atomic axes on
    /// their boundaries (a pure regrouping) rather than reshuffling elements
    /// within an axis.
    ///
    /// Torch dynamo lowers `scaled_dot_product_attention` by flattening
    /// batch*heads into a single leading axis for a 3-D batched matmul and then
    /// unflattening it — e.g. `[B,H,S,D] -> [B*H,S,D]` then `[B*H,D,S] ->
    /// [B,H,D,S]`. Both are clean merges/splits: because the atomic axes form a
    /// contiguous, order-preserving list and the output ranks/sizes are
    /// concrete, the grouping is forced, so the composition stays a genuine
    /// permutation of the synthetic head layout. Any output axis that would cut
    /// through the middle of an atomic axis (a real element reshuffle) fails
    /// closed here.
    fn regroup_attention_atoms(
        &self,
        spec: &LayerSpec,
        groups: Vec<Vec<(usize, i64)>>,
    ) -> Result<Vec<Vec<(usize, i64)>>> {
        let output = spec
            .outputs
            .first()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "attention Reshape '{}' has no output tensor to size its regrouping",
                    spec.name
                ))
            })?;
        let target = self.model.tensor_shapes.get(output).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "attention Reshape '{}' output '{}' has no concrete shape metadata; its axis \
                 regrouping cannot be proven equivalent to the synthetic layout",
                spec.name, output
            ))
        })?;
        if target.is_empty() || target.iter().any(|&dim| dim <= 0) {
            return Err(NyError::InvalidSpec(format!(
                "attention Reshape '{}' output '{}' shape {target:?} is not an unambiguous \
                 concrete shape",
                spec.name, output
            )));
        }
        let flat: Vec<(usize, i64)> = groups.into_iter().flatten().collect();
        let flat_product = flat.iter().try_fold(1i128, |product, &(_, size)| {
            product.checked_mul(i128::from(size))
        });
        let target_product = target
            .iter()
            .try_fold(1i128, |product, &dim| product.checked_mul(i128::from(dim)));
        match (flat_product, target_product) {
            (Some(a), Some(b)) if a == b => {}
            (Some(_), Some(_)) => {
                return Err(NyError::InvalidSpec(format!(
                    "attention Reshape '{}' output {target:?} changes the element count of the \
                     traced head axes; not a pure regrouping",
                    spec.name
                )));
            }
            _ => {
                return Err(NyError::InvalidSpec(format!(
                    "attention Reshape '{}' element product overflows while proving its regrouping",
                    spec.name
                )));
            }
        }
        let mut new_groups: Vec<Vec<(usize, i64)>> = Vec::with_capacity(target.len());
        let mut idx = 0usize;
        for &dim in target {
            let mut acc = 1i64;
            let mut group: Vec<(usize, i64)> = Vec::new();
            while acc != dim {
                let atom = *flat.get(idx).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "attention Reshape '{}' output {target:?} cannot be aligned to the traced \
                         head axes on whole-axis boundaries",
                        spec.name
                    ))
                })?;
                let next = acc.checked_mul(atom.1).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "attention Reshape '{}' axis product overflows while proving its regrouping",
                        spec.name
                    ))
                })?;
                if next > dim {
                    return Err(NyError::InvalidSpec(format!(
                        "attention Reshape '{}' output axis {dim} splits an atomic head axis; not \
                         a pure permutation of the synthetic layout",
                        spec.name
                    )));
                }
                acc = next;
                group.push(atom);
                idx += 1;
            }
            if group.is_empty() {
                // `dim == 1` with no atoms consumed: bind exactly one size-1
                // atomic axis (e.g. batch=1) to this output axis.
                match flat.get(idx) {
                    Some(&atom) if atom.1 == 1 => {
                        group.push(atom);
                        idx += 1;
                    }
                    _ => {
                        return Err(NyError::InvalidSpec(format!(
                            "attention Reshape '{}' has a size-1 output axis that does not align \
                             to a size-1 atomic head axis",
                            spec.name
                        )));
                    }
                }
            }
            new_groups.push(group);
        }
        if idx != flat.len() {
            return Err(NyError::InvalidSpec(format!(
                "attention Reshape '{}' leaves {} traced head axes unassigned; not a pure \
                 regrouping",
                spec.name,
                flat.len() - idx
            )));
        }
        Ok(new_groups)
    }

    fn reshape_target(&self, spec: &LayerSpec) -> Result<Vec<i64>> {
        if let Some(unknown) = spec
            .attributes
            .keys()
            .find(|name| name.as_str() != "allowzero" && name.as_str() != "shape")
        {
            return Err(NyError::InvalidSpec(format!(
                "attention Reshape '{}' has unproven attribute '{unknown}'",
                spec.name
            )));
        }
        // `allowzero` only changes the meaning of an *explicit 0* in the target
        // (default: 0 copies the source axis; allowzero=1: 0 is a literal empty
        // axis). Torch dynamo emits allowzero=1 on head-split reshapes whose
        // targets carry no 0 at all (e.g. [1,1500,-1,64]); there the two modes
        // are identical, so accepting allowzero=1 is sound. We reject only the
        // genuinely ambiguous case — allowzero=1 *with* an explicit 0 — because
        // resolve_attention_reshape_target reads 0 as "copy source axis".
        let allowzero_one = match spec.attributes.get("allowzero") {
            None | Some(AttributeValue::Int(0)) => false,
            Some(AttributeValue::Int(1)) => true,
            other => {
                return Err(NyError::InvalidSpec(format!(
                    "attention Reshape '{}' has unsupported allowzero attribute {other:?}",
                    spec.name
                )));
            }
        };
        let shape: Vec<i64> = if let Some(shape_attr) = spec.attributes.get("shape") {
            let AttributeValue::Ints(shape) = shape_attr else {
                return Err(NyError::InvalidSpec(format!(
                    "attention Reshape '{}' has a non-integer shape attribute",
                    spec.name
                )));
            };
            if spec.inputs.len() != 1 || spec.inputs[0].is_empty() {
                return Err(NyError::InvalidSpec(format!(
                    "attention Reshape '{}' with a shape attribute must have exactly one data \
                     operand; refusing an ambiguous/conflicting input shape",
                    spec.name
                )));
            }
            shape.clone()
        } else {
            if spec.inputs.len() != 2 || spec.inputs[0].is_empty() || spec.inputs[1].is_empty() {
                return Err(NyError::InvalidSpec(format!(
                    "attention Reshape '{}' without a shape attribute must have exactly data,target \
                     operands",
                    spec.name
                )));
            }
            let shape_name = spec.inputs.get(1).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "attention Reshape '{}' has no constant target-shape operand",
                    spec.name
                ))
            })?;
            if let Some(shape) = self.model.weights.get_integers(shape_name) {
                shape.iter().copied().collect()
            } else {
                let shape = self.model.weights.get(shape_name).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "attention Reshape '{}' target '{}' is not a known constant",
                        spec.name, shape_name
                    ))
                })?;
                shape
                    .iter()
                    .map(|&value| {
                        if !value.is_finite()
                            || value.fract() != 0.0
                            || value < i64::MIN as f32
                            // i64::MAX rounds up to 2^63 in f32, so equality is
                            // already outside the representable i64 range.
                            || value >= i64::MAX as f32
                        {
                            Err(NyError::InvalidSpec(format!(
                                "attention Reshape '{}' has non-integral/unrepresentable target value {value}",
                                spec.name
                            )))
                        } else {
                            Ok(value as i64)
                        }
                    })
                    .collect::<Result<Vec<i64>>>()?
            }
        };
        if allowzero_one && shape.contains(&0) {
            return Err(NyError::InvalidSpec(format!(
                "attention Reshape '{}' has an unsupported allowzero=1 with an explicit 0 target \
                 dimension; a literal empty axis is not a proven head-split reshape",
                spec.name
            )));
        }
        Ok(shape)
    }

    fn validate_qkv_reshape(&self, spec: &LayerSpec, head_dim: usize) -> Result<()> {
        let target = self.reshape_target(spec)?;
        let heads = i64::try_from(self.num_heads)
            .map_err(|_| NyError::InvalidSpec("Whisper num_heads does not fit i64".to_string()))?;
        let head_dim = i64::try_from(head_dim)
            .map_err(|_| NyError::InvalidSpec("Whisper head_dim does not fit i64".to_string()))?;
        let dynamic_or_positive = |value: i64| value == -1 || value == 0 || value > 0;
        // The heads/head_dim axes may be given literally (torchscript
        // [1,-1,heads,head_dim]) or left inferred (torch dynamo bakes the
        // sequence and infers heads: [1,1500,-1,head_dim]). Either is fine here
        // because the resolve-against-concrete-source equality below proves the
        // reshape yields exactly synthetic [B,S,heads,head_dim] regardless of
        // which single axis carries the -1. The "at most one -1" cap keeps that
        // inference unambiguous.
        let heads_or_infer = |value: i64| value == -1 || value == heads;
        let head_dim_or_infer = |value: i64| value == -1 || value == head_dim;
        if target.len() != 4
            || target.iter().filter(|&&value| value == -1).count() > 1
            || !matches!(target[0], -1..=1)
            || !dynamic_or_positive(target[1])
            || !heads_or_infer(target[2])
            || !head_dim_or_infer(target[3])
        {
            return Err(NyError::InvalidSpec(format!(
                "attention Q/K/V Reshape '{}' target {target:?} is not proven B,S,heads,head_dim \
                 with heads={} head_dim={}",
                spec.name, self.num_heads, head_dim
            )));
        }
        let source = self.concrete_attention_reshape_source_shape(spec, "Q/K/V", 3)?;
        let hidden = i64::try_from(self.hidden_dim)
            .map_err(|_| NyError::InvalidSpec("Whisper hidden_dim does not fit i64".to_string()))?;
        if source[2] != hidden {
            return Err(NyError::InvalidSpec(format!(
                "attention Q/K/V Reshape '{}' source shape {source:?} does not have hidden_dim={} \
                 in its final axis",
                spec.name, self.hidden_dim
            )));
        }
        let resolved = self.resolve_attention_reshape_target(spec, &source, &target)?;
        let expected = vec![source[0], source[1], heads, head_dim];
        if resolved != expected {
            return Err(NyError::InvalidSpec(format!(
                "attention Q/K/V Reshape '{}' resolves original target {target:?} against source \
                 {source:?} to {resolved:?}, not synthetic B,S,heads,head_dim {expected:?}",
                spec.name
            )));
        }
        Ok(())
    }

    fn validate_context_reshape(&self, spec: &LayerSpec) -> Result<()> {
        let target = self.reshape_target(spec)?;
        let hidden = i64::try_from(self.hidden_dim)
            .map_err(|_| NyError::InvalidSpec("Whisper hidden_dim does not fit i64".to_string()))?;
        let dynamic_or_positive = |value: i64| value == -1 || value == 0 || value > 0;
        // Dynamo bakes the sequence and infers the merged hidden axis
        // ([1,1500,-1]); torchscript writes it literally. Accept either — the
        // resolve-against-concrete-source equality below proves the merge yields
        // exactly synthetic [B,S,hidden_dim] whichever single axis is inferred.
        let hidden_or_infer = |value: i64| value == -1 || value == hidden;
        if target.len() != 3
            || target.iter().filter(|&&value| value == -1).count() > 1
            || !matches!(target[0], -1..=1)
            || !dynamic_or_positive(target[1])
            || !hidden_or_infer(target[2])
        {
            return Err(NyError::InvalidSpec(format!(
                "attention context Reshape '{}' target {target:?} is not proven B,S,hidden_dim \
                 with hidden_dim={}",
                spec.name, self.hidden_dim
            )));
        }
        let source = self.concrete_attention_reshape_source_shape(spec, "context", 4)?;
        let heads = i64::try_from(self.num_heads)
            .map_err(|_| NyError::InvalidSpec("Whisper num_heads does not fit i64".to_string()))?;
        let head_dim = i64::try_from(self.hidden_dim / self.num_heads)
            .map_err(|_| NyError::InvalidSpec("Whisper head_dim does not fit i64".to_string()))?;
        if source[2] != heads || source[3] != head_dim {
            return Err(NyError::InvalidSpec(format!(
                "attention context Reshape '{}' source shape {source:?} is not concrete \
                 B,S,heads,head_dim with heads={} head_dim={}",
                spec.name, self.num_heads, head_dim
            )));
        }
        let resolved = self.resolve_attention_reshape_target(spec, &source, &target)?;
        let expected = vec![source[0], source[1], hidden];
        if resolved != expected {
            return Err(NyError::InvalidSpec(format!(
                "attention context Reshape '{}' resolves original target {target:?} against \
                 source {source:?} to {resolved:?}, not synthetic B,S,hidden_dim {expected:?}",
                spec.name
            )));
        }
        Ok(())
    }

    fn concrete_attention_reshape_source_shape(
        &self,
        spec: &LayerSpec,
        role: &str,
        rank: usize,
    ) -> Result<Vec<i64>> {
        let source_name = spec
            .inputs
            .first()
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "attention {role} Reshape '{}' has no source tensor at input 0",
                    spec.name
                ))
            })?;
        let source = self.model.tensor_shapes.get(source_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "attention {role} Reshape '{}' source tensor '{}' has no shape metadata; exact \
                 equivalence to synthetic wiring is unproven",
                spec.name, source_name
            ))
        })?;
        if source.len() != rank || source.iter().any(|&dim| dim <= 0) {
            return Err(NyError::InvalidSpec(format!(
                "attention {role} Reshape '{}' source tensor '{}' shape {source:?} is not an \
                 unambiguous concrete rank-{rank} shape",
                spec.name, source_name
            )));
        }
        Ok(source.clone())
    }

    fn resolve_attention_reshape_target(
        &self,
        spec: &LayerSpec,
        source: &[i64],
        target: &[i64],
    ) -> Result<Vec<i64>> {
        let checked_product = |dims: &[i64], label: &str| -> Result<i128> {
            dims.iter().try_fold(1i128, |product, &dim| {
                product.checked_mul(i128::from(dim)).ok_or_else(|| {
                    NyError::InvalidSpec(format!(
                        "attention Reshape '{}' {label} element product overflows",
                        spec.name
                    ))
                })
            })
        };
        let source_elements = checked_product(source, "source")?;
        let mut resolved = Vec::with_capacity(target.len());
        let mut inferred_axis = None;
        for (axis, &dim) in target.iter().enumerate() {
            match dim {
                -1 => {
                    if inferred_axis.replace(axis).is_some() {
                        return Err(NyError::InvalidSpec(format!(
                            "attention Reshape '{}' target {target:?} has more than one inferred \
                             dimension",
                            spec.name
                        )));
                    }
                    resolved.push(-1);
                }
                0 => {
                    let copied = source.get(axis).copied().ok_or_else(|| {
                        NyError::InvalidSpec(format!(
                            "attention Reshape '{}' target {target:?} copies axis {axis}, but \
                             source shape {source:?} has rank {}",
                            spec.name,
                            source.len()
                        ))
                    })?;
                    resolved.push(copied);
                }
                positive if positive > 0 => resolved.push(positive),
                _ => {
                    return Err(NyError::InvalidSpec(format!(
                        "attention Reshape '{}' target {target:?} contains invalid dimension {dim}",
                        spec.name
                    )));
                }
            }
        }

        let known_dims: Vec<i64> = resolved.iter().copied().filter(|&dim| dim != -1).collect();
        let known_elements = checked_product(&known_dims, "known output")?;
        if let Some(axis) = inferred_axis {
            if known_elements == 0 || source_elements % known_elements != 0 {
                return Err(NyError::InvalidSpec(format!(
                    "attention Reshape '{}' target {target:?} cannot infer an integral dimension \
                     from source shape {source:?}",
                    spec.name
                )));
            }
            let inferred = source_elements / known_elements;
            resolved[axis] = i64::try_from(inferred).map_err(|_| {
                NyError::InvalidSpec(format!(
                    "attention Reshape '{}' inferred dimension {inferred} does not fit i64",
                    spec.name
                ))
            })?;
        } else if known_elements != source_elements {
            return Err(NyError::InvalidSpec(format!(
                "attention Reshape '{}' target {target:?} has {known_elements} elements, but \
                 source shape {source:?} has {source_elements}",
                spec.name
            )));
        }
        Ok(resolved)
    }

    fn apply_attention_transpose(
        &self,
        spec: &LayerSpec,
        input_axes: &[usize],
    ) -> Result<Vec<usize>> {
        if let Some(unknown) = spec.attributes.keys().find(|name| name.as_str() != "perm") {
            return Err(NyError::InvalidSpec(format!(
                "attention Transpose '{}' has unproven attribute '{unknown}'",
                spec.name
            )));
        }
        if spec.inputs.iter().filter(|name| !name.is_empty()).count() != 1
            || spec.inputs.first().is_none_or(String::is_empty)
        {
            return Err(NyError::InvalidSpec(format!(
                "attention Transpose '{}' does not have exactly one data operand at input 0",
                spec.name
            )));
        }
        let perm: Vec<usize> = match spec.attributes.get("perm") {
            Some(AttributeValue::Ints(values)) => values
                .iter()
                .map(|&value| {
                    usize::try_from(value).map_err(|_| {
                        NyError::InvalidSpec(format!(
                            "attention Transpose '{}' has negative perm index {value}",
                            spec.name
                        ))
                    })
                })
                .collect::<Result<_>>()?,
            None => (0..input_axes.len()).rev().collect(),
            other => {
                return Err(NyError::InvalidSpec(format!(
                    "attention Transpose '{}' has invalid perm attribute {other:?}",
                    spec.name
                )));
            }
        };
        let mut sorted = perm.clone();
        sorted.sort_unstable();
        if sorted != (0..input_axes.len()).collect::<Vec<_>>() {
            return Err(NyError::InvalidSpec(format!(
                "attention Transpose '{}' perm {perm:?} is not a rank-{} permutation",
                spec.name,
                input_axes.len()
            )));
        }
        Ok(perm.into_iter().map(|axis| input_axes[axis]).collect())
    }

    /// Structurally discover the attention nodes of encoder block `index`.
    ///
    /// Projections are located via their bias Adds / weight-name tokens, and
    /// the attention core (scores → softmax → context → output projection)
    /// purely by op-type + topology (tracing activation sources through
    /// reshape/transpose/scale plumbing), so exporter naming conventions do
    /// not affect the result.
    pub(crate) fn discover_attention_nodes(
        &self,
        index: usize,
    ) -> Result<DiscoveredAttentionNodes<'_>> {
        if index >= self.encoder_layers {
            return Err(NyError::InvalidSpec(format!(
                "Encoder layer {} out of range (max {})",
                index, self.encoder_layers
            )));
        }

        let block_layers = self.block_layers_for_index(index)?;
        let weights = &self.model.weights;
        let constants = &self.model.constant_tensors;
        let attn_ln_tokens = ["self_attn_layer_norm", "attn_ln"];

        let find_weight_name = |tokens: &[&str], suffix: &str| -> Option<String> {
            weights
                .keys()
                .find(|name| {
                    parse_block_index(name) == Some(index)
                        && name.contains(suffix)
                        && tokens.iter().any(|token| name.contains(token))
                })
                .map(|s| s.to_string())
        };

        let find_layer_with_input =
            |layer_type: LayerType, weight_name: &str| -> Option<&LayerSpec> {
                block_layers.iter().copied().find(|spec| {
                    spec.layer_type == layer_type
                        && spec.inputs.iter().any(|input| input == weight_name)
                })
            };

        let q_bias = find_weight_name(
            &["self_attn.q_proj", "q_proj", "attn/query", "attn.query"],
            "bias",
        );
        let k_bias = find_weight_name(
            &["self_attn.k_proj", "k_proj", "attn/key", "attn.key"],
            "bias",
        );
        let v_bias = find_weight_name(
            &["self_attn.v_proj", "v_proj", "attn/value", "attn.value"],
            "bias",
        );
        let out_bias = find_weight_name(
            &["self_attn.out_proj", "out_proj", "attn/out", "attn.out"],
            "bias",
        );
        let mut output_to_spec: std::collections::HashMap<String, &LayerSpec> =
            std::collections::HashMap::new();
        for spec in self.model.network.layers.iter() {
            for output in &spec.outputs {
                if !output.is_empty() {
                    output_to_spec.insert(output.clone(), spec);
                }
            }
        }

        let trace_activation_source = |tensor: &str| -> String {
            let mut current = tensor.to_string();
            let mut visited = std::collections::HashSet::new();
            while let Some(spec) = output_to_spec.get(&current) {
                if !visited.insert(spec.name.clone()) {
                    break;
                }
                let passthrough = match spec.layer_type {
                    LayerType::Reshape | LayerType::Transpose => true,
                    LayerType::Mul | LayerType::Div => {
                        let activation_inputs = spec.inputs.iter().filter(|name| {
                            !weights.contains_key(name) && !constants.contains(*name)
                        });
                        activation_inputs.count() == 1
                    }
                    _ => false,
                };
                if !passthrough {
                    break;
                }
                let next = spec
                    .inputs
                    .iter()
                    .find(|name| !weights.contains_key(name) && !constants.contains(*name));
                if let Some(next) = next {
                    current = next.clone();
                } else {
                    break;
                }
            }
            current
        };

        let trace_weight_source = |tensor: &str| -> Option<String> {
            if weights.contains_key(tensor) {
                return Some(tensor.to_string());
            }
            let mut current = tensor.to_string();
            let mut visited = std::collections::HashSet::new();
            loop {
                let spec = output_to_spec.get(&current)?;
                if !visited.insert(spec.name.clone()) {
                    return None;
                }
                let next = match spec.layer_type {
                    LayerType::Transpose => spec.inputs.first(),
                    LayerType::Reshape => spec
                        .inputs
                        .iter()
                        .find(|name| weights.contains_key(name))
                        .or_else(|| spec.inputs.iter().find(|name| !constants.contains(*name))),
                    LayerType::Mul | LayerType::Div => {
                        spec.inputs.iter().find(|name| !constants.contains(*name))
                    }
                    _ => None,
                };
                let next = next?;
                if weights.contains_key(next) {
                    return Some(next.clone());
                }
                current = next.clone();
            }
        };

        let find_add_by_bias = |bias: &str| -> Option<&LayerSpec> {
            block_layers.iter().copied().find(|spec| {
                spec.layer_type == LayerType::Add && spec.inputs.iter().any(|input| input == bias)
            })
        };

        let find_matmul_from_add = |add_spec: &LayerSpec| -> Option<&LayerSpec> {
            let activation_input = add_spec
                .inputs
                .iter()
                .find(|name| !weights.contains_key(name) && !constants.contains(*name))?;
            let origin = trace_activation_source(activation_input);
            let producer = output_to_spec.get(&origin)?;
            if matches!(producer.layer_type, LayerType::MatMul | LayerType::Linear) {
                Some(*producer)
            } else {
                None
            }
        };

        let weight_matches_tokens = |name: &str, tokens: &[&str]| -> bool {
            parse_block_index(name) == Some(index)
                && name.contains("weight")
                && tokens.iter().any(|token| name.contains(token))
        };

        let find_matmul_by_weight_tokens = |tokens: &[&str]| -> Option<&LayerSpec> {
            block_layers.iter().copied().find(|spec| {
                if !matches!(spec.layer_type, LayerType::MatMul | LayerType::Linear) {
                    return false;
                }
                if let Some(weights_ref) = &spec.weights {
                    if weight_matches_tokens(&weights_ref.name, tokens) {
                        return true;
                    }
                }
                spec.inputs.iter().any(|input| {
                    trace_weight_source(input)
                        .is_some_and(|name| weight_matches_tokens(&name, tokens))
                })
            })
        };
        let find_matmul_by_tokens = |tokens: &[&str]| -> Option<&LayerSpec> {
            block_layers.iter().copied().find(|spec| {
                matches!(spec.layer_type, LayerType::MatMul | LayerType::Linear)
                    && layer_any_name_matches(spec, |name| {
                        tokens.iter().any(|token| name.contains(token))
                    })
            })
        };

        let q_add_spec = q_bias.as_ref().and_then(|bias| find_add_by_bias(bias));
        let v_add_spec = v_bias.as_ref().and_then(|bias| find_add_by_bias(bias));
        let out_add_spec = out_bias.as_ref().and_then(|bias| find_add_by_bias(bias));
        let k_add_spec = k_bias.as_ref().and_then(|bias| find_add_by_bias(bias));

        let q_matmul_spec = q_add_spec
            .and_then(&find_matmul_from_add)
            .or_else(|| {
                find_matmul_by_weight_tokens(&[
                    "self_attn.q_proj",
                    "q_proj",
                    "attn/query",
                    "attn.query",
                ])
            })
            .or_else(|| {
                find_matmul_by_tokens(&["self_attn.q_proj", "q_proj", "attn/query", "attn.query"])
            })
            .ok_or_else(|| {
                NyError::InvalidSpec(format!("Q projection MatMul not found for block {}", index))
            })?;
        let v_matmul_spec = v_add_spec
            .and_then(&find_matmul_from_add)
            .or_else(|| {
                find_matmul_by_weight_tokens(&[
                    "self_attn.v_proj",
                    "v_proj",
                    "attn/value",
                    "attn.value",
                ])
            })
            .or_else(|| {
                find_matmul_by_tokens(&["self_attn.v_proj", "v_proj", "attn/value", "attn.value"])
            })
            .ok_or_else(|| {
                NyError::InvalidSpec(format!("V projection MatMul not found for block {}", index))
            })?;
        let out_matmul_spec = out_add_spec
            .and_then(&find_matmul_from_add)
            .or_else(|| {
                find_matmul_by_weight_tokens(&[
                    "self_attn.out_proj",
                    "out_proj",
                    "attn/out",
                    "attn.out",
                ])
            })
            .or_else(|| {
                find_matmul_by_tokens(&["self_attn.out_proj", "out_proj", "attn/out", "attn.out"])
            })
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Attention output MatMul not found for block {}",
                    index
                ))
            })?;

        let find_layernorm_from_projection = |spec: &LayerSpec| -> Option<&LayerSpec> {
            let activation_input = spec
                .inputs
                .iter()
                .find(|name| !weights.contains_key(name) && !constants.contains(*name))?;
            let origin = trace_activation_source(activation_input);
            let producer = output_to_spec.get(&origin)?;
            if producer.layer_type == LayerType::LayerNorm {
                Some(*producer)
            } else {
                None
            }
        };

        let attn_ln_weight = find_weight_name(&attn_ln_tokens, "weight");
        let attn_ln_spec = attn_ln_weight
            .as_ref()
            .and_then(|weight| find_layer_with_input(LayerType::LayerNorm, weight))
            .or_else(|| Self::find_layernorm_by_tokens(&block_layers, &attn_ln_tokens))
            .or_else(|| find_layernorm_from_projection(q_matmul_spec))
            .or_else(|| find_layernorm_from_projection(v_matmul_spec))
            .ok_or_else(|| {
                let detail = if attn_ln_weight.is_some() {
                    "Attention LayerNorm node not found"
                } else {
                    "Attention LayerNorm weight/node not found"
                };
                NyError::InvalidSpec(format!("{} for block {}", detail, index))
            })?;

        let ln_output = attn_ln_spec.outputs.first().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Attention LayerNorm output missing for block {}",
                index
            ))
        })?;

        let mut k_matmul_spec = k_add_spec.and_then(find_matmul_from_add).or_else(|| {
            find_matmul_by_weight_tokens(&["self_attn.k_proj", "k_proj", "attn/key", "attn.key"])
        });
        if k_matmul_spec.is_none() {
            k_matmul_spec =
                find_matmul_by_tokens(&["self_attn.k_proj", "k_proj", "attn/key", "attn.key"]);
        }
        if k_matmul_spec.is_none() {
            k_matmul_spec = block_layers.iter().copied().find(|spec| {
                matches!(spec.layer_type, LayerType::MatMul | LayerType::Linear)
                    && spec.inputs.iter().any(|input| input == ln_output)
                    && spec.name != q_matmul_spec.name
                    && spec.name != v_matmul_spec.name
            });
        }
        let k_matmul_spec = k_matmul_spec.ok_or_else(|| {
            NyError::InvalidSpec(format!("K projection MatMul not found for block {}", index))
        })?;

        let q_src_spec = q_add_spec.unwrap_or(q_matmul_spec);
        let k_src_spec = k_add_spec.unwrap_or(k_matmul_spec);
        let v_src_spec = v_add_spec.unwrap_or(v_matmul_spec);

        let q_src_output = q_src_spec.outputs.first().ok_or_else(|| {
            NyError::InvalidSpec(format!("Q projection output missing for block {}", index))
        })?;
        let k_src_output = k_src_spec.outputs.first().ok_or_else(|| {
            NyError::InvalidSpec(format!("K projection output missing for block {}", index))
        })?;
        let v_src_output = v_src_spec.outputs.first().ok_or_else(|| {
            NyError::InvalidSpec(format!("V projection output missing for block {}", index))
        })?;

        let mut attn_scores_spec = None;
        for spec in block_layers
            .iter()
            .copied()
            .filter(|s| s.layer_type == LayerType::MatMul)
        {
            let activation_inputs: Vec<&String> = spec
                .inputs
                .iter()
                .filter(|name| !weights.contains_key(name) && !constants.contains(*name))
                .collect();
            if activation_inputs.len() == 2 {
                let a = trace_activation_source(activation_inputs[0]);
                let b = trace_activation_source(activation_inputs[1]);
                if (a == *q_src_output && b == *k_src_output)
                    || (a == *k_src_output && b == *q_src_output)
                {
                    attn_scores_spec = Some(spec);
                    break;
                }
            }
        }
        let attn_scores_spec = attn_scores_spec.ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Attention scores MatMul not found for block {}",
                index
            ))
        })?;
        let attn_scores_output = attn_scores_spec.outputs.first().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Attention scores output missing for block {}",
                index
            ))
        })?;

        let mut attn_softmax_spec = None;
        for spec in block_layers
            .iter()
            .copied()
            .filter(|s| s.layer_type == LayerType::Softmax)
        {
            let activation_input = spec
                .inputs
                .iter()
                .find(|name| !weights.contains_key(name) && !constants.contains(*name));
            if let Some(input) = activation_input {
                let origin = trace_activation_source(input);
                if origin == *attn_scores_output {
                    attn_softmax_spec = Some(spec);
                    break;
                }
            }
        }
        let attn_softmax_spec = attn_softmax_spec.ok_or_else(|| {
            NyError::InvalidSpec(format!("Attention softmax not found for block {}", index))
        })?;
        let attn_softmax_output = attn_softmax_spec.outputs.first().ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Attention softmax output missing for block {}",
                index
            ))
        })?;

        let mut attn_ctx_spec = None;
        for spec in block_layers
            .iter()
            .copied()
            .filter(|s| s.layer_type == LayerType::MatMul)
        {
            if spec.name == attn_scores_spec.name {
                continue;
            }
            let activation_inputs: Vec<&String> = spec
                .inputs
                .iter()
                .filter(|name| !weights.contains_key(name) && !constants.contains(*name))
                .collect();
            if activation_inputs.len() == 2 {
                let a = trace_activation_source(activation_inputs[0]);
                let b = trace_activation_source(activation_inputs[1]);
                if (a == *attn_softmax_output && b == *v_src_output)
                    || (a == *v_src_output && b == *attn_softmax_output)
                {
                    attn_ctx_spec = Some(spec);
                    break;
                }
            }
        }
        let attn_ctx_spec = attn_ctx_spec.ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Attention context MatMul not found for block {}",
                index
            ))
        })?;

        Ok(DiscoveredAttentionNodes {
            attn_ln: attn_ln_spec,
            q_matmul: q_matmul_spec,
            q_add: q_add_spec,
            k_matmul: k_matmul_spec,
            k_add: k_add_spec,
            v_matmul: v_matmul_spec,
            v_add: v_add_spec,
            attn_scores: attn_scores_spec,
            attn_softmax: attn_softmax_spec,
            attn_ctx: attn_ctx_spec,
            out_matmul: out_matmul_spec,
            out_add: out_add_spec,
        })
    }

    fn build_attention_subgraph_with_root(
        &self,
        index: usize,
        root: AttentionGraphRoot,
    ) -> Result<(GraphNetwork, String, String, String, String)> {
        let hidden_dim = self.hidden_dim;
        let num_heads = self.num_heads;
        if num_heads == 0 {
            return Err(NyError::InvalidSpec(
                "Attention subgraph requires num_heads > 0".to_string(),
            ));
        }
        if !hidden_dim.is_multiple_of(num_heads) {
            return Err(NyError::InvalidSpec(format!(
                "hidden_dim {} not divisible by num_heads {}",
                hidden_dim, num_heads
            )));
        }
        let head_dim = hidden_dim / num_heads;

        let block_layers = self.block_layers_for_index(index)?;
        let nodes = self.discover_attention_nodes(index)?;
        // Unlike the full-block route, an attention-only subgraph has no plain
        // graph with the same output contract to fall back to. Fail closed if
        // structural equivalence of every replaced plumbing node is unproven.
        let AttentionRewritePlan {
            replaced_nodes: replaced_attention_nodes,
        } = self.attention_rewrite_plan(index, &nodes)?;
        let attn_ln_spec = nodes.attn_ln;
        let q_matmul_spec = nodes.q_matmul;
        let k_matmul_spec = nodes.k_matmul;
        let v_matmul_spec = nodes.v_matmul;
        let out_matmul_spec = nodes.out_matmul;
        let attn_scores_spec = nodes.attn_scores;
        let attn_softmax_spec = nodes.attn_softmax;
        let attn_ctx_spec = nodes.attn_ctx;
        let q_add_spec = nodes.q_add;
        let k_add_spec = nodes.k_add;
        let v_add_spec = nodes.v_add;
        let out_add_spec = nodes.out_add;
        let q_src_spec = nodes.q_src();
        let k_src_spec = nodes.k_src();
        let v_src_spec = nodes.v_src();

        // When root is LayerNormOutput, omit the LayerNorm node so its output
        // tensor (`ln_output`) stays unresolved and maps to `_input` via the
        // existing external-activation collapse. This gives Q, K, and V one
        // shared zonotope source. Part of #318.
        let include_ln = match root {
            AttentionGraphRoot::BlockInput => true,
            #[cfg(test)]
            AttentionGraphRoot::LayerNormOutput => false,
        };
        let attn_layer_names: std::collections::HashSet<String> = [
            &q_matmul_spec.name,
            &k_matmul_spec.name,
            &v_matmul_spec.name,
            &attn_scores_spec.name,
            &attn_softmax_spec.name,
            &attn_ctx_spec.name,
            &out_matmul_spec.name,
        ]
        .iter()
        .map(|s| (*s).clone())
        .chain(if include_ln {
            Some(attn_ln_spec.name.clone())
        } else {
            None
        })
        .chain(q_add_spec.map(|spec| spec.name.clone()))
        .chain(k_add_spec.map(|spec| spec.name.clone()))
        .chain(v_add_spec.map(|spec| spec.name.clone()))
        .chain(out_add_spec.map(|spec| spec.name.clone()))
        .collect();

        let q_src = &q_src_spec.name;
        let k_src = &k_src_spec.name;
        let v_src = &v_src_spec.name;
        let attn_scores = &attn_scores_spec.name;
        let attn_softmax = &attn_softmax_spec.name;
        let attn_ctx = &attn_ctx_spec.name;
        let out_matmul = &out_matmul_spec.name;
        let out_add = out_add_spec.map(|spec| spec.name.clone());

        let mut block_layer_names: std::collections::HashSet<&str> =
            std::collections::HashSet::new();
        for spec in &block_layers {
            block_layer_names.insert(spec.name.as_str());
        }
        let mut extra_layer_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for spec in [
            attn_ln_spec,
            q_matmul_spec,
            k_matmul_spec,
            v_matmul_spec,
            attn_scores_spec,
            attn_softmax_spec,
            attn_ctx_spec,
            out_matmul_spec,
        ] {
            if !block_layer_names.contains(spec.name.as_str()) {
                extra_layer_names.insert(spec.name.clone());
            }
        }
        for spec in [q_add_spec, k_add_spec, v_add_spec, out_add_spec]
            .into_iter()
            .flatten()
        {
            if !block_layer_names.contains(spec.name.as_str()) {
                extra_layer_names.insert(spec.name.clone());
            }
        }

        let mut graph_layers: Vec<&LayerSpec> = Vec::new();
        for spec in &self.model.network.layers {
            if block_layer_names.contains(spec.name.as_str())
                || extra_layer_names.contains(&spec.name)
            {
                graph_layers.push(spec);
            }
        }

        let q_reshape = format!("{}::__reshape_bshd", q_src);
        let q_transpose = format!("{}::__transpose_bhsd", q_src);
        let k_reshape = format!("{}::__reshape_bshd", k_src);
        let k_transpose = format!("{}::__transpose_bhsd", k_src);
        let v_reshape = format!("{}::__reshape_bshd", v_src);
        let v_transpose = format!("{}::__transpose_bhsd", v_src);
        let ctx_transpose = format!("{}::__transpose_bshd", attn_ctx);
        let ctx_reshape = format!("{}::__reshape_bsd", attn_ctx);

        let qkv_target_shape = vec![0, 0, num_heads as i64, head_dim as i64];
        let qkv_perm = vec![0, 2, 1, 3];

        let mut graph = GraphNetwork::new();
        let mut tensor_to_node: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut external_tensors: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut constant_tensors_local = self.model.constant_tensors.clone();

        // Pre-evaluate constant chains for the attention subgraph block (#696).
        let (block_start, block_end) = self.block_onnx_bounds(index)?;
        let evaluated_constants =
            self.evaluate_block_constants(block_start, block_end, &constant_tensors_local);

        for &spec in &graph_layers {
            let include_layer = attn_layer_names.contains(&spec.name)
                || replaced_attention_nodes.contains(&spec.name);
            if !include_layer {
                continue;
            }

            // Filter evaluated_constants (#697, #1204): pre-evaluated constant chains
            // may not be in constant_tensors_local but are still constant.
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
                        "Skipping Concat {} with all-constant inputs in attention subgraph",
                        spec.name
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
                debug!(
                    "Skipping constant-only layer {} in attention subgraph",
                    spec.name
                );
                for output in &spec.outputs {
                    if !output.is_empty() {
                        constant_tensors_local.insert(output.clone());
                    }
                }
                continue;
            }

            // Skip Concat with <2 activation inputs when there are no constant data
            // inputs. Shape-computing Concats fit this pattern, but data Concats with
            // constant data should remain in the graph (#1204).
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
                    "Skipping Concat {} with {} activation input(s) in attention subgraph",
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

            // Skip original attention reshape/transpose/mul nodes - they're replaced by synthetic nodes.
            // Trace their outputs to their activation inputs so downstream nodes remain connected.
            if replaced_attention_nodes.contains(&spec.name) {
                if let Some(output) = spec.outputs.first() {
                    let activation_input = activation_inputs.first().ok_or_else(|| {
                        NyError::InvalidSpec(
                            "No activation input found in layer inputs".to_string(),
                        )
                    })?;
                    if let Some(src_node) = tensor_to_node.get(*activation_input) {
                        tensor_to_node.insert(output.clone(), src_node.clone());
                    } else if external_tensors.contains(*activation_input) {
                        external_tensors.insert(output.clone());
                    } else {
                        external_tensors.insert((*activation_input).clone());
                        external_tensors.insert(output.clone());
                    }
                }
                continue;
            }

            // Only include attention layers
            if !attn_layer_names.contains(&spec.name) {
                continue;
            }

            // For Concat, use convert_concat_with_evaluated to embed constants (#696).
            let mut layer = if spec.layer_type == LayerType::Concat {
                self.model
                    .convert_context()
                    .convert_concat_with_evaluated(spec, &evaluated_constants)?
            } else {
                self.model.convert_layer(spec)?
            };

            let mut input_nodes = self.find_input_nodes(
                spec,
                &layer,
                &tensor_to_node,
                &mut external_tensors,
                &constant_tensors_local,
                &evaluated_constants,
            )?;

            // Override attention core to use explicit reshape/transpose nodes
            if spec.name == *attn_scores {
                let scale = 1.0 / (head_dim as f32).sqrt();
                layer = Layer::MatMul(MatMulLayer::try_new(true, Some(scale))?);
                input_nodes = vec![q_transpose.clone(), k_transpose.clone()];
            } else if spec.name == *attn_softmax {
                input_nodes = vec![attn_scores.clone()];
            } else if spec.name == *attn_ctx {
                layer = Layer::MatMul(MatMulLayer::try_new(false, None)?);
                input_nodes = vec![attn_softmax.clone(), v_transpose.clone()];
            } else if spec.name == *out_matmul {
                input_nodes = vec![ctx_reshape.clone()];
            }

            graph.try_add_node(GraphNode::new(spec.name.clone(), layer, input_nodes))?;

            if let Some(output_name) = spec.outputs.first() {
                tensor_to_node.insert(output_name.clone(), spec.name.clone());
            }

            // Insert shape transform nodes (#2686: try_add_node returns Result)
            if spec.name == *q_src {
                graph.try_add_node(GraphNode::new(
                    q_reshape.clone(),
                    Layer::Reshape(ReshapeLayer::new(qkv_target_shape.clone())),
                    vec![q_src.clone()],
                ))?;
                graph.try_add_node(GraphNode::new(
                    q_transpose.clone(),
                    Layer::Transpose(TransposeLayer::new(qkv_perm.clone())),
                    vec![q_reshape.clone()],
                ))?;
            } else if spec.name == *k_src {
                graph.try_add_node(GraphNode::new(
                    k_reshape.clone(),
                    Layer::Reshape(ReshapeLayer::new(qkv_target_shape.clone())),
                    vec![k_src.clone()],
                ))?;
                graph.try_add_node(GraphNode::new(
                    k_transpose.clone(),
                    Layer::Transpose(TransposeLayer::new(qkv_perm.clone())),
                    vec![k_reshape.clone()],
                ))?;
            } else if spec.name == *v_src {
                graph.try_add_node(GraphNode::new(
                    v_reshape.clone(),
                    Layer::Reshape(ReshapeLayer::new(qkv_target_shape.clone())),
                    vec![v_src.clone()],
                ))?;
                graph.try_add_node(GraphNode::new(
                    v_transpose.clone(),
                    Layer::Transpose(TransposeLayer::new(qkv_perm.clone())),
                    vec![v_reshape.clone()],
                ))?;
            } else if spec.name == *attn_ctx {
                graph.try_add_node(GraphNode::new(
                    ctx_transpose.clone(),
                    Layer::Transpose(TransposeLayer::new(vec![0, 2, 1, 3])),
                    vec![attn_ctx.clone()],
                ))?;
                graph.try_add_node(GraphNode::new(
                    ctx_reshape.clone(),
                    Layer::Reshape(ReshapeLayer::new(vec![0, 0, hidden_dim as i64])),
                    vec![ctx_transpose.clone()],
                ))?;
            }
        }

        let output_node = if let Some(out_add_name) = out_add.as_ref() {
            if graph.contains_node(out_add_name.as_str()) {
                out_add_name.clone()
            } else if graph.contains_node(out_matmul.as_str()) {
                out_matmul.clone()
            } else if graph.contains_node(attn_ctx.as_str()) {
                attn_ctx.clone()
            } else if let Some(last) = graph.node_names().last() {
                last.clone()
            } else {
                return Err(NyError::InvalidSpec(format!(
                    "Attention subgraph for block {} produced no nodes",
                    index
                )));
            }
        } else if graph.contains_node(out_matmul.as_str()) {
            out_matmul.clone()
        } else if graph.contains_node(attn_ctx.as_str()) {
            attn_ctx.clone()
        } else if let Some(last) = graph.node_names().last() {
            last.clone()
        } else {
            return Err(NyError::InvalidSpec(format!(
                "Attention subgraph for block {} produced no nodes",
                index
            )));
        };

        graph.set_output(output_node.clone());

        info!(
            "Built attention subgraph for block {} with {} nodes",
            index,
            graph.num_nodes()
        );

        Ok((
            graph,
            attn_scores.clone(),
            attn_softmax.clone(),
            attn_ctx.clone(),
            output_node,
        ))
    }
}
