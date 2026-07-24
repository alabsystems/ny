// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! MLP subgraph extraction for decoder compositional verification.
//!
//! Supports multiple MLP variants:
//! - GELU: fc1 → GELU → fc2 (GPT-2, Whisper)
//! - SwiGLU: gate_proj → SiLU → up_proj → Mul → down_proj (Qwen3, LLaMA, Mistral)
//! - Structural fallback: all layers under mlp_prefix
//!
//! Reference: Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.

use ny_core::{NyError, Result};
use ny_propagate::{GraphNetwork, GraphNode};
use tracing::info;

use super::DecoderModel;

impl DecoderModel {
    /// Extract the MLP subgraph for compositional verification.
    ///
    /// Dispatches to variant-specific extraction based on detected MLP topology:
    /// - GELU MLP: norm2 → fc1 → GELU → fc2 (GPT-2, Whisper)
    /// - SwiGLU MLP: norm2 → gate_proj → SiLU → up_proj → Mul → down_proj (Qwen3, LLaMA, Mistral)
    /// - Structural fallback: all layers under mlp_prefix
    pub fn mlp_subgraph(&self, block_index: usize) -> Result<GraphNetwork> {
        // Determine block prefix. For single-block models without block-indexed naming,
        // use empty prefix. Check all supported MLP variants (GELU, SwiGLU HuggingFace,
        // SwiGLU alternative) to avoid falling through to structural fallback when
        // the model uses block-prefixed naming.
        let has_block_prefix = self.has_layer("/blocks.0/mlp/fc1/MatMul")
            || self.has_layer("/blocks.0/mlp/gate_proj/MatMul")
            || self.has_layer("/blocks.0/mlp/w1/MatMul");
        let prefix = if self.num_blocks == 1 && !has_block_prefix {
            String::new()
        } else {
            format!("/blocks.{}", block_index)
        };

        let mlp_prefix = if prefix.is_empty() {
            "/mlp".to_string()
        } else {
            format!("{}/mlp", prefix)
        };

        // Try GELU MLP first (existing logic — backward compatible)
        let fc1_matmul = format!("{}/fc1/MatMul", mlp_prefix);
        if self.has_layer(&fc1_matmul) {
            return self.mlp_subgraph_gelu(block_index, &prefix, &mlp_prefix);
        }

        // Try SwiGLU MLP (Qwen3, LLaMA, Mistral naming conventions)
        let gate_matmul = format!("{}/gate_proj/MatMul", mlp_prefix);
        let gate_matmul_alt = format!("{}/w1/MatMul", mlp_prefix);
        if self.has_layer(&gate_matmul) || self.has_layer(&gate_matmul_alt) {
            return self.mlp_subgraph_swiglu(block_index, &prefix, &mlp_prefix);
        }

        // Structural fallback: include all layers under mlp_prefix
        self.mlp_subgraph_structural(block_index, &prefix, &mlp_prefix)
    }

    /// Extract GELU MLP subgraph: norm2 → fc1 → GELU → fc2.
    ///
    /// This is the original extraction logic for GPT-2/Whisper style models.
    fn mlp_subgraph_gelu(
        &self,
        block_index: usize,
        prefix: &str,
        mlp_prefix: &str,
    ) -> Result<GraphNetwork> {
        let fc1_matmul = format!("{}/fc1/MatMul", mlp_prefix);
        let fc1_add = format!("{}/fc1/Add", mlp_prefix);
        let fc2_matmul = format!("{}/fc2/MatMul", mlp_prefix);
        let fc2_add = format!("{}/fc2/Add", mlp_prefix);

        let mut mlp_layer_names: std::collections::HashSet<String> =
            [&fc1_matmul, &fc1_add, &fc2_matmul, &fc2_add]
                .iter()
                .filter(|s| self.has_layer(s))
                .map(|s| (*s).clone())
                .collect();

        // Include norm2 chain
        let norm2_prefix = self.norm2_prefix(prefix);
        mlp_layer_names.extend(self.collect_norm2_layers(&norm2_prefix));

        // Include GELU layers — scoped to this block's mlp_prefix to avoid
        // matching GELU layers from other blocks in multi-block models.
        let gelu_prefix = format!("{}/gelu/", mlp_prefix);
        let gelu_layers: Vec<String> = self
            .model
            .network
            .layers
            .iter()
            .filter(|l| l.name.starts_with(&gelu_prefix))
            .map(|l| l.name.clone())
            .collect();
        mlp_layer_names.extend(gelu_layers);

        let mut graph = self.build_mlp_graph(&mlp_layer_names)?;

        // Set the output to fc2 bias add (or matmul if no bias)
        let output = if self.has_layer(&fc2_add) {
            fc2_add
        } else {
            fc2_matmul
        };
        graph.set_output(&output);

        info!(
            "Built GELU MLP subgraph for block {} with {} nodes",
            block_index,
            graph.num_nodes()
        );

        Ok(graph)
    }

    /// Extract SwiGLU MLP subgraph: norm2 → gate_proj → SiLU → up_proj → Mul → down_proj.
    ///
    /// SwiGLU(x, W_gate, W_up) = SiLU(x · W_gate) ⊙ (x · W_up)
    /// Output = W_down · SwiGLU(x, W_gate, W_up)
    ///
    /// Supports two naming conventions:
    /// - HuggingFace: gate_proj, up_proj, down_proj (Qwen3, LLaMA, Mistral)
    /// - Alternative: w1 (gate), w3 (up), w2 (down)
    fn mlp_subgraph_swiglu(
        &self,
        block_index: usize,
        prefix: &str,
        mlp_prefix: &str,
    ) -> Result<GraphNetwork> {
        let gate_names = [
            format!("{}/gate_proj/MatMul", mlp_prefix),
            format!("{}/w1/MatMul", mlp_prefix),
        ];
        let up_names = [
            format!("{}/up_proj/MatMul", mlp_prefix),
            format!("{}/w3/MatMul", mlp_prefix),
        ];
        let down_names = [
            format!("{}/down_proj/MatMul", mlp_prefix),
            format!("{}/w2/MatMul", mlp_prefix),
        ];

        let mut mlp_layer_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Add projection layers (MatMul and optional Add for bias)
        for name_variants in [&gate_names[..], &up_names[..], &down_names[..]] {
            for name in name_variants {
                if self.has_layer(name) {
                    mlp_layer_names.insert(name.clone());
                }
                let bias_name = name.replace("/MatMul", "/Add");
                if self.has_layer(&bias_name) {
                    mlp_layer_names.insert(bias_name);
                }
            }
        }

        // Include SiLU activation layers under mlp_prefix.
        // Match all known casings: /Silu (HuggingFace), /silu (GGUF-style),
        // /SiLU (ONNX op type casing), /Swish (alias).
        let silu_layers: Vec<String> = self
            .model
            .network
            .layers
            .iter()
            .filter(|l| {
                l.name.starts_with(mlp_prefix)
                    && (l.name.contains("/Silu")
                        || l.name.contains("/silu")
                        || l.name.contains("/SiLU")
                        || l.name.contains("/Swish"))
            })
            .map(|l| l.name.clone())
            .collect();
        mlp_layer_names.extend(silu_layers);

        // Include Mul layer (SwiGLU gating multiply) — exclude MatMul (linear projections)
        let mul_layers: Vec<String> = self
            .model
            .network
            .layers
            .iter()
            .filter(|l| {
                l.name.starts_with(mlp_prefix)
                    && l.name.contains("/Mul")
                    && !l.name.contains("/MatMul")
            })
            .map(|l| l.name.clone())
            .collect();
        mlp_layer_names.extend(mul_layers);

        // Include norm2 chain
        let norm2_prefix = self.norm2_prefix(prefix);
        mlp_layer_names.extend(self.collect_norm2_layers(&norm2_prefix));

        let mut graph = self.build_mlp_graph(&mlp_layer_names)?;

        // Set output to down_proj (last linear in SwiGLU)
        let output = down_names
            .iter()
            .flat_map(|n| [n.replace("/MatMul", "/Add"), n.clone()])
            .find(|n| self.has_layer(n))
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "SwiGLU MLP: no down_proj found for block {}",
                    block_index
                ))
            })?;
        graph.set_output(&output);

        info!(
            "Built SwiGLU MLP subgraph for block {} with {} nodes",
            block_index,
            graph.num_nodes()
        );

        Ok(graph)
    }

    /// Structural fallback: include ALL layers under mlp_prefix plus norm2.
    ///
    /// For models with non-standard naming, this best-effort approach includes
    /// every layer under the MLP prefix and relies on the propagation engine to
    /// handle whatever topology it finds.
    fn mlp_subgraph_structural(
        &self,
        block_index: usize,
        prefix: &str,
        mlp_prefix: &str,
    ) -> Result<GraphNetwork> {
        let mut mlp_layer_names: std::collections::HashSet<String> = self
            .model
            .network
            .layers
            .iter()
            .filter(|l| l.name.starts_with(mlp_prefix))
            .map(|l| l.name.clone())
            .collect();

        // Include norm2 chain
        let norm2_prefix = self.norm2_prefix(prefix);
        mlp_layer_names.extend(self.collect_norm2_layers(&norm2_prefix));

        if mlp_layer_names.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "No MLP layers found for block {} under prefix '{}'",
                block_index, mlp_prefix
            )));
        }

        let mut graph = self.build_mlp_graph(&mlp_layer_names)?;

        // Set output to the last layer in topological order under mlp_prefix
        let last_mlp_layer = self
            .model
            .network
            .layers
            .iter()
            .rev()
            .find(|l| l.name.starts_with(mlp_prefix))
            .map(|l| l.name.clone())
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Structural fallback: no output layer for block {}",
                    block_index
                ))
            })?;
        graph.set_output(&last_mlp_layer);

        info!(
            "Built structural MLP subgraph for block {} with {} nodes (fallback)",
            block_index,
            graph.num_nodes()
        );

        Ok(graph)
    }

    /// Compute the norm2 prefix for a given block prefix.
    pub(super) fn norm2_prefix(&self, prefix: &str) -> String {
        if prefix.is_empty() {
            "/norm2/".to_string()
        } else {
            format!("{}/norm2/", prefix)
        }
    }

    /// Collect all layer names matching the norm2 prefix.
    pub(super) fn collect_norm2_layers(&self, norm2_prefix: &str) -> Vec<String> {
        self.model
            .network
            .layers
            .iter()
            .filter(|l| l.name.starts_with(norm2_prefix))
            .map(|l| l.name.clone())
            .collect()
    }

    /// Build a GraphNetwork from a set of layer names.
    ///
    /// Shared graph construction logic used by all MLP subgraph variants.
    /// Pre-evaluates constant chains and uses `convert_concat_with_evaluated`
    /// for Concat layers (#3317).
    pub(super) fn build_mlp_graph(
        &self,
        mlp_layer_names: &std::collections::HashSet<String>,
    ) -> Result<GraphNetwork> {
        let mut graph = GraphNetwork::new();
        let mut tensor_to_node: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        // Pre-evaluate constant chains (#3317).
        let evaluated_constants = self.evaluate_model_constants();

        for spec in &self.model.network.layers {
            if !mlp_layer_names.contains(&spec.name) {
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
        }

        Ok(graph)
    }
}
