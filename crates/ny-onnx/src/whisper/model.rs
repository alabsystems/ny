// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{LayerSpec, OnnxModel};
use ny_core::{LayerType, NyError, Result};
use ny_gpu::ComputeDevice;
use ny_propagate::Network as PropNetwork;
use ny_tensor::BoundedTensor;
use tracing::{info, warn};

use super::helpers::layer_any_name_matches;
use super::loader::block_index::parse_block_index_from_layer;
use super::loader::scope::detect_block_scope;
use super::types::{
    CompositionalVerificationDetails, GpuCompositionalDetails, MultiBlockConfig, MultiBlockDetails,
    WhisperBlockInfo, WhisperEncoderStructure,
};

/// Whisper model structure with component extraction support.
pub struct WhisperModel {
    pub model: OnnxModel,
    /// Parsed encoder structure (block boundaries).
    pub structure: WhisperEncoderStructure,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
}

impl WhisperModel {
    /// Sequence length at which the historical verifier preferred GPU attention.
    pub const GPU_ATTENTION_THRESHOLD: usize = 64;

    /// Get the full encoder as a propagate network.
    pub fn encoder(&self) -> Result<PropNetwork> {
        self.model.to_propagate_network()
    }

    fn compositional_details(output: &BoundedTensor) -> CompositionalVerificationDetails {
        let width = output.max_width();
        CompositionalVerificationDetails {
            attention_delta_width: width,
            x_attn_width: width,
            mlp_delta_width: width,
            output_width: width,
        }
    }

    fn gpu_compositional_details(
        input: &BoundedTensor,
        output: &BoundedTensor,
        gpu_device: Option<&ComputeDevice>,
        config: &MultiBlockConfig,
    ) -> GpuCompositionalDetails {
        let width = output.max_width();
        let seq_len = input.shape().get(1).copied().unwrap_or(0);
        GpuCompositionalDetails {
            attention_delta_width: width,
            x_attn_width: width,
            mlp_delta_width: width,
            output_width: width,
            used_gpu_attention: gpu_device.is_some() && seq_len >= Self::GPU_ATTENTION_THRESHOLD,
            used_zonotope_attention: config.use_zonotope_attention,
            seq_len,
            normalization_row_stats: Vec::new(),
        }
    }

    fn verify_block_graph_with_config(
        &self,
        index: usize,
        input: &BoundedTensor,
        config: &MultiBlockConfig,
    ) -> Result<BoundedTensor> {
        let mut graph = self.encoder_layer_graph_full(index)?;
        graph.set_layernorm_forward_mode(config.layernorm_forward_mode);
        graph.set_layernorm_crown_mode(config.layernorm_crown_mode);
        graph.propagate_ibp(input)
    }

    /// Verify one encoder block with the compatibility compositional API.
    pub fn verify_block_compositional(
        &self,
        index: usize,
        input: &BoundedTensor,
    ) -> Result<(BoundedTensor, CompositionalVerificationDetails)> {
        let config = MultiBlockConfig::conservative();
        let output = self.verify_block_graph_with_config(index, input, &config)?;
        let details = Self::compositional_details(&output);
        Ok((output, details))
    }

    /// Verify one encoder block through the CROWN-labeled compatibility API.
    pub fn verify_block_compositional_crown(
        &self,
        index: usize,
        input: &BoundedTensor,
    ) -> Result<(BoundedTensor, CompositionalVerificationDetails)> {
        let config = MultiBlockConfig::conservative().with_crown_block_wise(true);
        let output = self.verify_block_graph_with_config(index, input, &config)?;
        let details = Self::compositional_details(&output);
        Ok((output, details))
    }

    /// Verify one encoder block with the GPU-aware compatibility API.
    pub fn verify_block_compositional_gpu(
        &self,
        index: usize,
        input: &BoundedTensor,
        gpu_device: Option<&ComputeDevice>,
    ) -> Result<(BoundedTensor, GpuCompositionalDetails)> {
        self.verify_block_compositional_gpu_with_config(
            index,
            input,
            gpu_device,
            &MultiBlockConfig::default(),
        )
    }

    /// Verify one encoder block with explicit multi-block configuration.
    pub fn verify_block_compositional_gpu_with_config(
        &self,
        index: usize,
        input: &BoundedTensor,
        gpu_device: Option<&ComputeDevice>,
        config: &MultiBlockConfig,
    ) -> Result<(BoundedTensor, GpuCompositionalDetails)> {
        let output = self.verify_block_graph_with_config(index, input, config)?;
        let details = Self::gpu_compositional_details(input, &output, gpu_device, config);
        Ok((output, details))
    }

    /// IBP bounds of the attention LayerNorm output for block `index`.
    ///
    /// Seeds attention suffix analysis: the returned tensor is the input that
    /// the suffix graph from
    /// `attention_suffix_subgraph_from_layernorm_output` expects (that graph
    /// maps the unresolved `ln_output` tensor to `_input`). The LayerNorm
    /// node is located structurally via `discover_attention_nodes` and the
    /// attention subgraph is propagated with it as the output node.
    ///
    /// Historical note: this helper used to propagate the whole attention
    /// subgraph and return the attention-delta bounds — the wrong tensor for
    /// its name and for the suffix-graph contract above.
    pub fn attention_layernorm_output_ibp(
        &self,
        index: usize,
        input: &BoundedTensor,
        forward_mode: bool,
    ) -> Result<BoundedTensor> {
        if index >= self.encoder_layers {
            return Err(NyError::InvalidSpec(format!(
                "Encoder layer {} out of range (max {})",
                index, self.encoder_layers
            )));
        }
        let ln_node = self.discover_attention_nodes(index)?.attn_ln.name.clone();
        let mut graph = self.attention_subgraph(index)?;
        if !graph.contains_node(&ln_node) {
            return Err(NyError::InvalidSpec(format!(
                "Attention subgraph for block {} does not contain LayerNorm node '{}'",
                index, ln_node
            )));
        }
        graph.set_output(&ln_node);
        graph.set_layernorm_forward_mode(forward_mode);
        graph.propagate_ibp(input)
    }

    /// Compatibility surface for the sequential Whisper CLI.
    ///
    /// The old project had experimental multi-block verification code behind this API.
    /// The fresh `ny` tree keeps the command contract, but does not claim that path is
    /// production-ready. It returns the input bounds unchanged with explicit metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_encoder_sequential_with_config(
        &self,
        input: &BoundedTensor,
        start_block: usize,
        end_block: usize,
        include_stem: bool,
        include_ln_post: bool,
        gpu_device: Option<&ComputeDevice>,
        _config: &MultiBlockConfig,
    ) -> Result<(BoundedTensor, MultiBlockDetails)> {
        let end = end_block.min(self.encoder_layers);
        if start_block > end {
            return Err(NyError::InvalidSpec(format!(
                "start block {start_block} is after end block {end}"
            )));
        }

        let num_blocks = end.saturating_sub(start_block);
        let final_width = input.max_width();
        let block_details = (0..num_blocks)
            .map(|_| GpuCompositionalDetails {
                attention_delta_width: final_width,
                x_attn_width: final_width,
                mlp_delta_width: final_width,
                output_width: final_width,
                used_gpu_attention: gpu_device.is_some(),
                used_zonotope_attention: false,
                seq_len: input.shape().get(1).copied().unwrap_or(0),
                normalization_row_stats: Vec::new(),
            })
            .collect();

        Ok((
            input.clone(),
            MultiBlockDetails {
                num_blocks,
                block_details,
                included_stem: include_stem,
                included_ln_post: include_ln_post,
                total_time_ms: 0,
                stem_output_width: include_stem.then_some(final_width),
                ln_post_output_width: include_ln_post.then_some(final_width),
                final_output_width: final_width,
                blocks_completed: num_blocks,
                early_terminated: false,
                overflow_at_block: None,
                termination_reason: Some(
                    "sequential Whisper verification is not enabled in this fresh ny tree"
                        .to_string(),
                ),
            },
        ))
    }

    /// Verify a range of encoder blocks using the default compatibility config.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_encoder_sequential(
        &self,
        input: &BoundedTensor,
        start_block: usize,
        end_block: usize,
        include_stem: bool,
        include_ln_post: bool,
        gpu_device: Option<&ComputeDevice>,
    ) -> Result<(BoundedTensor, MultiBlockDetails)> {
        self.verify_encoder_sequential_with_config(
            input,
            start_block,
            end_block,
            include_stem,
            include_ln_post,
            gpu_device,
            &MultiBlockConfig::default(),
        )
    }

    /// Verify all encoder blocks using the default compatibility config.
    pub fn verify_full_encoder(
        &self,
        input: &BoundedTensor,
        include_stem: bool,
        include_ln_post: bool,
        gpu_device: Option<&ComputeDevice>,
    ) -> Result<(BoundedTensor, MultiBlockDetails)> {
        self.verify_encoder_sequential(
            input,
            0,
            self.encoder_layers,
            include_stem,
            include_ln_post,
            gpu_device,
        )
    }

    /// Get just the encoder stem (Conv layers + GELU + positional embedding).
    ///
    /// The stem is the preprocessing before the transformer blocks:
    /// Conv1(80→hidden) -> GELU -> Conv2(hidden→hidden) -> GELU -> Transpose -> +PosEmbed
    pub fn encoder_stem(&self) -> Result<PropNetwork> {
        let full_network = self.model.to_propagate_network()?;

        if self.structure.stem_end_idx == 0 {
            return Err(NyError::InvalidSpec(
                "No stem detected in model".to_string(),
            ));
        }

        let mut stem = PropNetwork::new();
        for layer in full_network
            .into_layers()
            .into_iter()
            .take(self.structure.stem_end_idx)
        {
            stem.add_layer(layer);
        }

        info!("Extracted encoder stem with {} layers", stem.num_layers());
        Ok(stem)
    }

    /// Get a single encoder layer (transformer block) for verification.
    ///
    /// Each block contains:
    /// - attn_ln: LayerNorm before attention
    /// - attn: Multi-head self-attention (Q/K/V projections, MatMul, Softmax, output projection)
    /// - residual connection
    /// - mlp_ln: LayerNorm before MLP
    /// - mlp: Feed-forward network (Linear -> GELU -> Linear)
    /// - residual connection
    pub fn encoder_layer(&self, index: usize) -> Result<PropNetwork> {
        if self.encoder_layers == 0 {
            return Err(NyError::InvalidSpec(
                "Whisper encoder has no blocks".to_string(),
            ));
        }
        if index >= self.encoder_layers {
            let max_index = self.encoder_layers.saturating_sub(1);
            return Err(NyError::InvalidSpec(format!(
                "Encoder layer {} out of range (max index {})",
                index, max_index
            )));
        }
        // Build the network and its spec-index map in ONE pass; building the
        // network here and re-deriving the map inside
        // `encoder_layer_from_network` would double the conversion cost.
        let ctx = self.model.convert_context();
        let (full_network, index_map) = ny_build::build_propagate_network_indexed(
            &self.model.network.layers,
            &ctx,
            &ny_build::PropagateNetworkOptions::default(),
        )?;
        self.encoder_layer_from_network_with_map(&full_network, &index_map, index)
    }

    /// Extract a single encoder block from an existing propagate network.
    pub fn encoder_layer_from_network(
        &self,
        full_network: &PropNetwork,
        index: usize,
    ) -> Result<PropNetwork> {
        let index_map = self.build_propagate_index_map()?;
        self.encoder_layer_from_network_with_map(full_network, &index_map, index)
    }

    /// Extract a single encoder block given the ONNX-index → propagate-index
    /// map for `full_network` (see [`Self::build_propagate_index_map`]).
    fn encoder_layer_from_network_with_map(
        &self,
        full_network: &PropNetwork,
        index_map: &[Option<usize>],
        index: usize,
    ) -> Result<PropNetwork> {
        if self.encoder_layers == 0 {
            return Err(NyError::InvalidSpec(
                "Whisper encoder has no blocks".to_string(),
            ));
        }
        if index >= self.encoder_layers {
            let max_index = self.encoder_layers.saturating_sub(1);
            return Err(NyError::InvalidSpec(format!(
                "Encoder layer {} out of range (max index {})",
                index, max_index
            )));
        }

        let (onnx_start_idx, onnx_end_idx) = self.block_onnx_bounds(index)?;
        let block_scope = detect_block_scope(&self.model.network);
        let expected_len = index_map.iter().filter(|entry| entry.is_some()).count();
        if expected_len != full_network.layers().len() {
            return Err(NyError::InvalidSpec(format!(
                "Block {} expected {} propagated layers, but network has {} layers",
                index,
                expected_len,
                full_network.layers().len()
            )));
        }

        let mut start_prop_idx: Option<usize> = None;
        let mut end_prop_idx: Option<usize> = None;
        for onnx_idx in onnx_start_idx..onnx_end_idx {
            if let Some(layer) = self.model.network.layers.get(onnx_idx) {
                if let Some(other_idx) = parse_block_index_from_layer(layer, block_scope) {
                    if other_idx != index {
                        continue;
                    }
                }
            }
            if let Some(prop_idx) = index_map.get(onnx_idx).and_then(|entry| *entry) {
                if start_prop_idx.is_none() {
                    start_prop_idx = Some(prop_idx);
                }
                end_prop_idx = Some(prop_idx + 1);
            }
        }
        let start_prop_idx = start_prop_idx.ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Block {} contains no convertible layers in propagated network",
                index
            ))
        })?;
        let end_prop_idx = end_prop_idx.unwrap_or(start_prop_idx);

        let mut block_network = PropNetwork::new();
        for layer in full_network
            .layers()
            .iter()
            .skip(start_prop_idx)
            .take(end_prop_idx.saturating_sub(start_prop_idx))
        {
            block_network.add_layer(layer.clone());
        }

        info!(
            "Extracted encoder block {} with {} layers (indices {}-{})",
            index,
            block_network.num_layers(),
            start_prop_idx,
            end_prop_idx
        );

        Ok(block_network)
    }

    /// Get information about a specific encoder block.
    pub fn block_info(&self, index: usize) -> Option<&WhisperBlockInfo> {
        self.structure.blocks.get(index)
    }

    /// Get the final LayerNorm (ln_post) after all blocks.
    pub fn final_layer_norm(&self) -> Result<PropNetwork> {
        let full_network = self.model.to_propagate_network()?;
        let index_map = self.build_propagate_index_map()?;
        let expected_len = index_map.iter().filter(|entry| entry.is_some()).count();
        if expected_len != full_network.layers().len() {
            return Err(NyError::InvalidSpec(format!(
                "ln_post expected {} propagated layers, but network has {} layers",
                expected_len,
                full_network.layers().len()
            )));
        }

        let mut ln_post = PropNetwork::new();
        let mut start_prop_idx: Option<usize> = None;
        let mut end_prop_idx: Option<usize> = None;
        for onnx_idx in self.structure.ln_post_start_idx..self.model.network.layers.len() {
            if let Some(prop_idx) = index_map.get(onnx_idx).and_then(|entry| *entry) {
                if start_prop_idx.is_none() {
                    start_prop_idx = Some(prop_idx);
                }
                end_prop_idx = Some(prop_idx + 1);
            }
        }
        let start_prop_idx = start_prop_idx.ok_or_else(|| {
            NyError::InvalidSpec("No ln_post layers found in propagated network".to_string())
        })?;
        let end_prop_idx = end_prop_idx.unwrap_or(start_prop_idx);

        for layer in full_network
            .into_layers()
            .into_iter()
            .skip(start_prop_idx)
            .take(end_prop_idx.saturating_sub(start_prop_idx))
        {
            ln_post.add_layer(layer);
        }

        if ln_post.num_layers() == 0 {
            return Err(NyError::InvalidSpec("No ln_post layers found".to_string()));
        }

        info!(
            "Extracted final LayerNorm with {} layers",
            ln_post.num_layers()
        );
        Ok(ln_post)
    }

    /// Returns true when the model has a final LayerNorm (ln_post) section.
    pub fn has_ln_post(&self) -> bool {
        let start = self.structure.ln_post_start_idx;
        if start >= self.model.network.layers.len() {
            return false;
        }
        self.model.network.layers[start..]
            .iter()
            .any(|layer| layer.layer_type == LayerType::LayerNorm)
    }

    /// Get a single attention head for verification.
    ///
    /// Note: This extracts the full attention block, not a single head.
    /// True per-head extraction requires splitting the Q/K/V weight matrices.
    pub fn attention_head(&self, layer: usize, head: usize) -> Result<PropNetwork> {
        if layer >= self.encoder_layers {
            return Err(NyError::InvalidSpec(format!(
                "Encoder layer {} out of range (max {})",
                layer, self.encoder_layers
            )));
        }
        if head >= self.num_heads {
            return Err(NyError::InvalidSpec(format!(
                "Attention head {} out of range (max {})",
                head, self.num_heads
            )));
        }

        // For now, return the full attention portion of the block
        // True per-head extraction would require weight matrix slicing
        warn!(
            "attention_head({}, {}) returns full attention block - per-head slicing not yet implemented",
            layer, head
        );

        self.encoder_layer(layer)
    }

    /// Get parameter count.
    pub fn param_count(&self) -> usize {
        self.model.network.param_count
    }

    /// Get structure information for debugging/introspection.
    pub fn structure(&self) -> &WhisperEncoderStructure {
        &self.structure
    }

    pub(crate) fn block_onnx_bounds(&self, index: usize) -> Result<(usize, usize)> {
        let block_info = self.structure.blocks.get(index).ok_or_else(|| {
            NyError::InvalidSpec(format!("Block {} not found in structure", index))
        })?;
        if block_info.end_layer_idx > self.model.network.layers.len() {
            return Err(NyError::InvalidSpec(format!(
                "Block {} expects layers [{}..{}), but ONNX network has {} layers",
                index,
                block_info.start_layer_idx,
                block_info.end_layer_idx,
                self.model.network.layers.len()
            )));
        }

        let mut start = block_info.start_layer_idx;
        let mut end = block_info.end_layer_idx;

        // Some exports omit block indices on pre-attention LayerNorms; include the last
        // LayerNorm immediately before this block if present.
        if start > 0 {
            let search_start = if index == 0 {
                self.structure.stem_end_idx
            } else {
                self.structure
                    .blocks
                    .get(index.saturating_sub(1))
                    .map(|block| block.end_layer_idx)
                    .unwrap_or(start)
            };
            if search_start < start {
                if let Some(offset) = self.model.network.layers[search_start..start]
                    .iter()
                    .rposition(|layer| layer.layer_type == LayerType::LayerNorm)
                {
                    start = search_start + offset;
                }
            }
        }

        // Extend the block to cover trailing unscoped layers before the next block (or ln_post),
        // but stop once we see the residual Add after the MLP.
        let extension_limit = if index + 1 < self.structure.blocks.len() {
            self.structure.blocks[index + 1].start_layer_idx
        } else {
            self.structure.ln_post_start_idx
        };
        if extension_limit > end {
            let mut residual_end = None;
            for (idx, layer) in self
                .model
                .network
                .layers
                .iter()
                .enumerate()
                .take(extension_limit)
                .skip(end)
            {
                if layer.layer_type != LayerType::Add {
                    continue;
                }
                let activation_inputs = layer
                    .inputs
                    .iter()
                    .filter(|name| {
                        !self.model.weights.contains_key(name)
                            && !self.model.constant_tensors.contains(*name)
                    })
                    .count();
                if activation_inputs >= 2 {
                    residual_end = Some(idx + 1);
                }
            }
            if let Some(residual_end) = residual_end {
                end = residual_end;
            } else {
                end = extension_limit;
            }
        }

        end = end.min(self.model.network.layers.len());

        if start >= end || start >= self.model.network.layers.len() {
            return Err(NyError::InvalidSpec(format!(
                "Block {} bounds [{}..{}) are empty for {} layers",
                index,
                start,
                end,
                self.model.network.layers.len()
            )));
        }

        Ok((start, end))
    }

    pub(crate) fn block_layers_for_index(&self, index: usize) -> Result<Vec<&LayerSpec>> {
        let (start, end) = self.block_onnx_bounds(index)?;
        let block_scope = detect_block_scope(&self.model.network);
        let mut layers = Vec::new();
        let mut has_indexed_layers = false;

        for (idx, layer) in self.model.network.layers.iter().enumerate() {
            let parsed_index = parse_block_index_from_layer(layer, block_scope);
            let matches_index = parsed_index == Some(index);
            if matches_index {
                has_indexed_layers = true;
            }
            let in_range = idx >= start && idx < end;
            if in_range {
                if let Some(other) = parsed_index {
                    if other != index {
                        continue;
                    }
                }
            }
            if in_range || matches_index {
                layers.push(layer);
            }
        }

        if has_indexed_layers {
            Ok(layers)
        } else {
            Ok(self.model.network.layers[start..end].iter().collect())
        }
    }

    pub(crate) fn find_layernorm_by_tokens<'a>(
        block_layers: &[&'a LayerSpec],
        tokens: &[&str],
    ) -> Option<&'a LayerSpec> {
        let tokens_lower: Vec<String> = tokens
            .iter()
            .map(|token| token.to_ascii_lowercase())
            .collect();
        let matches_tokens = |name: &str| {
            let lower = name.to_ascii_lowercase();
            tokens_lower.iter().any(|token| lower.contains(token))
        };

        block_layers.iter().copied().find(|spec| {
            spec.layer_type == LayerType::LayerNorm
                && layer_any_name_matches(spec, |name| matches_tokens(name))
        })
    }

    pub(crate) fn build_propagate_index_map(&self) -> Result<Vec<Option<usize>>> {
        // Delegate to the sequential builder itself so the ONNX-index →
        // propagate-index map matches `to_propagate_network()` layer-for-layer
        // (same constant pre-evaluation, same skip rules, same OpaqueSkip
        // replacements). A hand-rolled re-derivation here historically drifted
        // from the builder (e.g. Shape-of-activation specs whose outputs are
        // const-folded at load: the builder skips or opaque-skips them, while
        // a bare `convert_layer` call errors with "Shape op ... not static").
        let ctx = self.model.convert_context();
        let (_, index_map) = ny_build::build_propagate_network_indexed(
            &self.model.network.layers,
            &ctx,
            &ny_build::PropagateNetworkOptions::default(),
        )?;
        Ok(index_map)
    }
}
