// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::config::ModelConfig;
use super::super::helpers::{generate_decomposed_attention, layernorm_attributes};
use crate::{DataType, LayerSpec, Network, TensorSpec, WeightStore};
use ny_core::{LayerType, Result};
use std::collections::HashMap;

/// Build generic transformer encoder from weights.
pub(in crate::native) fn build_transformer_encoder(
    weights: &WeightStore,
    config: &ModelConfig,
) -> Result<Network> {
    let mut layers = Vec::new();
    let hidden_dim = config.hidden_dim;
    let num_layers = config.num_layers.unwrap_or(6);

    let mut prev_output = "input".to_string();

    for layer_idx in 0..num_layers {
        // Self-attention block (decomposed into constituent layers)
        let layer_prefix = format!("layer{}", layer_idx);
        let num_heads = config.num_heads.unwrap_or(8);
        let (attn_layers, attn_out) = generate_decomposed_attention(
            weights,
            &layer_prefix,
            &prev_output,
            hidden_dim,
            num_heads,
            false, // encoder uses bidirectional attention
        );
        layers.extend(attn_layers);

        // Residual + LayerNorm
        let ln1_out = format!("layer{}_ln1_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_ln1", layer_idx),
            layer_type: LayerType::LayerNorm,
            inputs: vec![prev_output.clone(), attn_out],
            outputs: vec![ln1_out.clone()],
            weights: None,
            attributes: layernorm_attributes(config),
        });

        // FFN block (linear -> activation -> linear)
        let ffn_out = format!("layer{}_ffn_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_ffn", layer_idx),
            layer_type: LayerType::Linear,
            inputs: vec![ln1_out.clone()],
            outputs: vec![ffn_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        // GELU activation
        let gelu_out = format!("layer{}_gelu_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_gelu", layer_idx),
            layer_type: LayerType::GELU,
            inputs: vec![ffn_out],
            outputs: vec![gelu_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        // Second linear
        let ffn2_out = format!("layer{}_ffn2_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_ffn2", layer_idx),
            layer_type: LayerType::Linear,
            inputs: vec![gelu_out],
            outputs: vec![ffn2_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        // Residual + LayerNorm
        let ln2_out = format!("layer{}_ln2_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_ln2", layer_idx),
            layer_type: LayerType::LayerNorm,
            inputs: vec![ln1_out, ffn2_out],
            outputs: vec![ln2_out.clone()],
            weights: None,
            attributes: layernorm_attributes(config),
        });

        prev_output = ln2_out;
    }

    let param_count: usize = weights.iter().map(|(_, w)| w.len()).sum();

    Ok(Network {
        name: "transformer_encoder".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![-1, -1, hidden_dim as i64],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: prev_output,
            shape: vec![-1, -1, hidden_dim as i64],
            dtype: DataType::Float32,
        }],
        layers,
        param_count,
    })
}
