// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::config::ModelConfig;
use super::gguf_attention::generate_gguf_attention;
use crate::{DataType, LayerSpec, Network, TensorSpec, WeightStore};
use ny_core::{LayerType, Result};
use std::collections::HashMap;
use tracing::info;

/// Build transformer decoder network from GGUF weights (LLM architecture).
///
/// This handles the llama.cpp GGUF naming convention:
/// - `token_embd.weight` - Token embedding
/// - `blk.N.attn_q.weight` - Q projection
/// - `blk.N.attn_k.weight` - K projection
/// - `blk.N.attn_v.weight` - V projection
/// - `blk.N.attn_output.weight` - Output projection
/// - `blk.N.attn_norm.weight` - Pre-attention RMSNorm
/// - `blk.N.ffn_up.weight` - FFN up projection
/// - `blk.N.ffn_gate.weight` - FFN gate (for SwiGLU)
/// - `blk.N.ffn_down.weight` - FFN down projection
/// - `blk.N.ffn_norm.weight` - Pre-FFN RMSNorm
/// - `output_norm.weight` - Final RMSNorm
/// - `output.weight` - LM head
pub(in crate::native) fn build_transformer_decoder(
    weights: &WeightStore,
    config: &ModelConfig,
) -> Result<Network> {
    let mut layers = Vec::new();
    let hidden_dim = config.hidden_dim;
    let num_layers = config.num_layers.unwrap_or(32);
    let num_heads = config.num_heads.unwrap_or(32);
    let vocab_size = config.output_dim.unwrap_or(32000);

    info!(
        "Building transformer decoder: hidden_dim={}, layers={}, heads={}, vocab={}",
        hidden_dim, num_layers, num_heads, vocab_size
    );

    // For verification, we skip the embedding layer and start with embedded tokens.
    // The embedding layer maps discrete tokens to continuous embeddings, which is
    // not meaningful for perturbation-based verification.
    let mut prev_output = "input".to_string();

    // Transformer blocks
    for layer_idx in 0..num_layers {
        let prefix = format!("blk.{}", layer_idx);

        // Pre-attention RMSNorm
        let attn_norm_name = format!("{}.attn_norm.weight", prefix);
        let norm1_out = format!("layer{}_norm1_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_attn_norm", layer_idx),
            layer_type: LayerType::RMSNorm,
            inputs: vec![prev_output.clone(), attn_norm_name.clone()],
            outputs: vec![norm1_out.clone()],
            weights: Some(crate::WeightRef {
                name: attn_norm_name,
                shape: vec![hidden_dim],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::from([(
                "normalized_shape".to_string(),
                crate::AttributeValue::Ints(vec![hidden_dim as i64]),
            )]),
        });

        // Self-attention with causal mask (decomposed)
        let (attn_layers, attn_out) = generate_gguf_attention(
            weights, &prefix, &norm1_out, hidden_dim, num_heads, layer_idx,
        );
        layers.extend(attn_layers);

        // Residual connection after attention
        let add1_out = format!("layer{}_add1_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_add1", layer_idx),
            layer_type: LayerType::Add,
            inputs: vec![prev_output.clone(), attn_out],
            outputs: vec![add1_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        // Pre-FFN RMSNorm
        let ffn_norm_name = format!("{}.ffn_norm.weight", prefix);
        let norm2_out = format!("layer{}_norm2_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_ffn_norm", layer_idx),
            layer_type: LayerType::RMSNorm,
            inputs: vec![add1_out.clone(), ffn_norm_name.clone()],
            outputs: vec![norm2_out.clone()],
            weights: Some(crate::WeightRef {
                name: ffn_norm_name,
                shape: vec![hidden_dim],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::from([(
                "normalized_shape".to_string(),
                crate::AttributeValue::Ints(vec![hidden_dim as i64]),
            )]),
        });

        // FFN (SwiGLU): gate * silu(up) then down
        let ffn_up_name = format!("{}.ffn_up.weight", prefix);
        let ffn_gate_name = format!("{}.ffn_gate.weight", prefix);
        let ffn_down_name = format!("{}.ffn_down.weight", prefix);

        // Get FFN intermediate dim from weight shape
        let ffn_dim = weights
            .get(&ffn_up_name)
            .map(|w| w.shape()[1])
            .unwrap_or(hidden_dim * 4);

        // Up projection
        // GGUF: ffn_up.weight [hidden_dim, ffn_dim], LinearLayer expects [ffn_dim, hidden_dim]
        let up_out = format!("layer{}_ffn_up_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_ffn_up", layer_idx),
            layer_type: LayerType::Linear,
            inputs: vec![norm2_out.clone(), ffn_up_name.clone()],
            outputs: vec![up_out.clone()],
            weights: Some(crate::WeightRef {
                name: ffn_up_name,
                // LinearLayer expects (out_features, in_features)
                shape: vec![ffn_dim, hidden_dim],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::new(),
        });

        // Gate projection
        // GGUF: ffn_gate.weight [hidden_dim, ffn_dim], LinearLayer expects [ffn_dim, hidden_dim]
        let gate_out = format!("layer{}_ffn_gate_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_ffn_gate", layer_idx),
            layer_type: LayerType::Linear,
            inputs: vec![norm2_out, ffn_gate_name.clone()],
            outputs: vec![gate_out.clone()],
            weights: Some(crate::WeightRef {
                name: ffn_gate_name,
                // LinearLayer expects (out_features, in_features)
                shape: vec![ffn_dim, hidden_dim],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::new(),
        });

        // SiLU activation on gate
        let silu_out = format!("layer{}_silu_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_silu", layer_idx),
            layer_type: LayerType::SiLU,
            inputs: vec![gate_out],
            outputs: vec![silu_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        // Element-wise multiply (SwiGLU)
        let swiglu_out = format!("layer{}_swiglu_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_swiglu", layer_idx),
            layer_type: LayerType::Mul,
            inputs: vec![up_out, silu_out],
            outputs: vec![swiglu_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        // Down projection
        // GGUF: ffn_down.weight [ffn_dim, hidden_dim], LinearLayer expects [hidden_dim, ffn_dim]
        let down_out = format!("layer{}_ffn_down_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_ffn_down", layer_idx),
            layer_type: LayerType::Linear,
            inputs: vec![swiglu_out, ffn_down_name.clone()],
            outputs: vec![down_out.clone()],
            weights: Some(crate::WeightRef {
                name: ffn_down_name,
                // LinearLayer expects (out_features, in_features)
                shape: vec![hidden_dim, ffn_dim],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::new(),
        });

        // Residual connection after FFN
        let add2_out = format!("layer{}_add2_out", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_add2", layer_idx),
            layer_type: LayerType::Add,
            inputs: vec![add1_out, down_out],
            outputs: vec![add2_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        prev_output = add2_out;
    }

    // Final RMSNorm
    let output_norm_name = "output_norm.weight".to_string();
    layers.push(LayerSpec {
        name: "output_norm".to_string(),
        layer_type: LayerType::RMSNorm,
        inputs: vec![prev_output, output_norm_name.clone()],
        outputs: vec!["norm_out".to_string()],
        weights: Some(crate::WeightRef {
            name: output_norm_name,
            shape: vec![hidden_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::from([(
            "normalized_shape".to_string(),
            crate::AttributeValue::Ints(vec![hidden_dim as i64]),
        )]),
    });

    let param_count: usize = weights.iter().map(|(_, w)| w.len()).sum();

    Ok(Network {
        name: "transformer_decoder".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            // For verification, input is embedded tokens [seq, hidden_dim], not token indices
            // The embedding layer is conceptually external to the verifiable network
            shape: vec![-1, hidden_dim as i64],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "norm_out".to_string(),
            shape: vec![-1, hidden_dim as i64], // [seq, hidden]
            dtype: DataType::Float32,
        }],
        layers,
        param_count,
    })
}
