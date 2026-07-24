// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::config::ModelConfig;
use super::super::helpers::{
    find_weight, find_weight_name, generate_decomposed_attention, layernorm_attributes,
};
use crate::{DataType, LayerSpec, Network, TensorSpec, WeightStore};
use ny_core::{LayerType, Result};
use std::collections::HashMap;

/// Build Whisper encoder network from weights.
pub(in crate::native) fn build_whisper_encoder(
    weights: &WeightStore,
    config: &ModelConfig,
) -> Result<Network> {
    let mut layers = Vec::new();
    let mut param_count = 0;

    // Input: [batch, n_mels, time] -> typically [1, 80, 3000] or [1, 128, 3000]
    let n_mels = config.input_dim.unwrap_or(80);
    let hidden_dim = config.hidden_dim;

    // Conv1: [batch, n_mels, time] -> [batch, hidden, time/2]
    if let Some(conv1_w) = find_weight(weights, &["conv1.weight", "encoder.conv1.weight"]) {
        let out_ch = conv1_w.shape()[0];
        let in_ch = conv1_w.shape()[1];
        let kernel = conv1_w.shape()[2];
        let weight_name = find_weight_name(weights, &["conv1.weight", "encoder.conv1.weight"])
            .unwrap_or("conv1.weight".to_string());
        let bias_name = find_weight_name(weights, &["conv1.bias", "encoder.conv1.bias"]);
        let mut conv1_inputs = vec!["input".to_string(), weight_name.clone()];
        if let Some(bn) = &bias_name {
            conv1_inputs.push(bn.clone());
        }
        layers.push(LayerSpec {
            name: "conv1".to_string(),
            layer_type: LayerType::Conv1d,
            inputs: conv1_inputs,
            outputs: vec!["conv1_out".to_string()],
            weights: Some(crate::WeightRef {
                name: weight_name,
                shape: vec![out_ch, in_ch, kernel],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::from([
                (
                    "kernel_size".to_string(),
                    crate::AttributeValue::Int(kernel as i64),
                ),
                ("strides".to_string(), crate::AttributeValue::Ints(vec![1])),
                ("pads".to_string(), crate::AttributeValue::Ints(vec![1, 1])),
            ]),
        });
        param_count += conv1_w.len();
        if let Some(bias) = bias_name.and_then(|n| find_weight(weights, &[&n])) {
            param_count += bias.len();
        }

        // GELU after conv1
        layers.push(LayerSpec {
            name: "conv1_gelu".to_string(),
            layer_type: LayerType::GELU,
            inputs: vec!["conv1_out".to_string()],
            outputs: vec!["conv1_gelu_out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        });
    }

    // Conv2: [batch, hidden, time/2] -> [batch, hidden, time/2]
    if let Some(conv2_w) = find_weight(weights, &["conv2.weight", "encoder.conv2.weight"]) {
        let out_ch = conv2_w.shape()[0];
        let in_ch = conv2_w.shape()[1];
        let kernel = conv2_w.shape()[2];
        let weight_name = find_weight_name(weights, &["conv2.weight", "encoder.conv2.weight"])
            .unwrap_or("conv2.weight".to_string());
        let bias_name = find_weight_name(weights, &["conv2.bias", "encoder.conv2.bias"]);
        let mut conv2_inputs = vec!["conv1_gelu_out".to_string(), weight_name.clone()];
        if let Some(bn) = &bias_name {
            conv2_inputs.push(bn.clone());
        }
        layers.push(LayerSpec {
            name: "conv2".to_string(),
            layer_type: LayerType::Conv1d,
            inputs: conv2_inputs,
            outputs: vec!["conv2_out".to_string()],
            weights: Some(crate::WeightRef {
                name: weight_name,
                shape: vec![out_ch, in_ch, kernel],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::from([
                (
                    "kernel_size".to_string(),
                    crate::AttributeValue::Int(kernel as i64),
                ),
                ("strides".to_string(), crate::AttributeValue::Ints(vec![2])),
                ("pads".to_string(), crate::AttributeValue::Ints(vec![1, 1])),
            ]),
        });
        param_count += conv2_w.len();
        if let Some(bias) = bias_name.and_then(|n| find_weight(weights, &[&n])) {
            param_count += bias.len();
        }

        // GELU after conv2
        layers.push(LayerSpec {
            name: "conv2_gelu".to_string(),
            layer_type: LayerType::GELU,
            inputs: vec!["conv2_out".to_string()],
            outputs: vec!["conv2_gelu_out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        });
    }

    // Transpose from [channels, length] to [length, channels] for transformer blocks
    // This converts the conv output to sequence format expected by attention
    layers.push(LayerSpec {
        name: "conv_transpose".to_string(),
        layer_type: LayerType::Transpose,
        inputs: vec!["conv2_gelu_out".to_string()],
        outputs: vec!["encoder_input".to_string()],
        weights: None,
        attributes: HashMap::from([
            ("perm".to_string(), crate::AttributeValue::Ints(vec![1, 0])), // Swap dims
        ]),
    });

    // Transformer blocks
    let num_blocks = config.num_layers.unwrap_or(6);
    let mut prev_output = "encoder_input".to_string(); // Use transposed output

    for block_idx in 0..num_blocks {
        let block_prefix = format!("block{}", block_idx);

        // Self-attention (decomposed into constituent layers)
        let num_heads = config.num_heads.unwrap_or(6);
        let (attn_layers, attn_out) = generate_decomposed_attention(
            weights,
            &block_prefix,
            &prev_output,
            hidden_dim,
            num_heads,
            false, // encoder uses bidirectional attention, not causal
        );
        layers.extend(attn_layers);

        // Add residual
        let add1_out = format!("block{}_add1_out", block_idx);
        layers.push(LayerSpec {
            name: format!("block{}_add1", block_idx),
            layer_type: LayerType::Add,
            inputs: vec![prev_output.clone(), attn_out],
            outputs: vec![add1_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        // LayerNorm 1 (self_attn_layer_norm in Whisper)
        let ln1_ny_name = find_weight_name(
            weights,
            &[
                &format!(
                    "model.encoder.layers.{}.self_attn_layer_norm.weight",
                    block_idx
                ),
                &format!("encoder.layers.{}.self_attn_layer_norm.weight", block_idx),
                &format!("layers.{}.self_attn_layer_norm.weight", block_idx),
            ],
        );
        let ln1_beta_name = find_weight_name(
            weights,
            &[
                &format!(
                    "model.encoder.layers.{}.self_attn_layer_norm.bias",
                    block_idx
                ),
                &format!("encoder.layers.{}.self_attn_layer_norm.bias", block_idx),
                &format!("layers.{}.self_attn_layer_norm.bias", block_idx),
            ],
        );
        let ln1_out = format!("block{}_ln1_out", block_idx);
        let mut ln1_inputs = vec![add1_out.clone()];
        if let Some(ref g) = ln1_ny_name {
            ln1_inputs.push(g.clone());
        }
        if let Some(ref b) = ln1_beta_name {
            ln1_inputs.push(b.clone());
        }
        layers.push(LayerSpec {
            name: format!("block{}_ln1", block_idx),
            layer_type: LayerType::LayerNorm,
            inputs: ln1_inputs,
            outputs: vec![ln1_out.clone()],
            weights: ln1_ny_name.as_ref().map(|g| crate::WeightRef {
                name: g.clone(),
                shape: vec![hidden_dim],
                original_dtype: DataType::Float32,
            }),
            attributes: {
                let mut attributes = layernorm_attributes(config);
                attributes.insert(
                    "normalized_shape".to_string(),
                    crate::AttributeValue::Ints(vec![hidden_dim as i64]),
                );
                attributes
            },
        });

        // MLP (fc1 -> gelu -> fc2)
        // Look up MLP weights using Whisper naming convention
        let fc1_weight_name = find_weight_name(
            weights,
            &[
                &format!("model.encoder.layers.{}.fc1.weight", block_idx),
                &format!("encoder.layers.{}.fc1.weight", block_idx),
                &format!("layers.{}.fc1.weight", block_idx),
            ],
        );
        let fc1_bias_name = find_weight_name(
            weights,
            &[
                &format!("model.encoder.layers.{}.fc1.bias", block_idx),
                &format!("encoder.layers.{}.fc1.bias", block_idx),
                &format!("layers.{}.fc1.bias", block_idx),
            ],
        );
        let fc1_out = format!("block{}_fc1_out", block_idx);
        let mut fc1_inputs = vec![ln1_out.clone()];
        if let Some(ref w) = fc1_weight_name {
            fc1_inputs.push(w.clone());
        }
        if let Some(ref b) = fc1_bias_name {
            fc1_inputs.push(b.clone());
        }
        layers.push(LayerSpec {
            name: format!("block{}_fc1", block_idx),
            layer_type: LayerType::Linear,
            inputs: fc1_inputs,
            outputs: vec![fc1_out.clone()],
            weights: fc1_weight_name.as_ref().map(|w| crate::WeightRef {
                name: w.clone(),
                shape: vec![hidden_dim * 4, hidden_dim], // MLP expansion
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::new(),
        });

        let gelu_out = format!("block{}_gelu_out", block_idx);
        layers.push(LayerSpec {
            name: format!("block{}_gelu", block_idx),
            layer_type: LayerType::GELU,
            inputs: vec![fc1_out],
            outputs: vec![gelu_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        let fc2_weight_name = find_weight_name(
            weights,
            &[
                &format!("model.encoder.layers.{}.fc2.weight", block_idx),
                &format!("encoder.layers.{}.fc2.weight", block_idx),
                &format!("layers.{}.fc2.weight", block_idx),
            ],
        );
        let fc2_bias_name = find_weight_name(
            weights,
            &[
                &format!("model.encoder.layers.{}.fc2.bias", block_idx),
                &format!("encoder.layers.{}.fc2.bias", block_idx),
                &format!("layers.{}.fc2.bias", block_idx),
            ],
        );
        let fc2_out = format!("block{}_fc2_out", block_idx);
        let mut fc2_inputs = vec![gelu_out];
        if let Some(ref w) = fc2_weight_name {
            fc2_inputs.push(w.clone());
        }
        if let Some(ref b) = fc2_bias_name {
            fc2_inputs.push(b.clone());
        }
        layers.push(LayerSpec {
            name: format!("block{}_fc2", block_idx),
            layer_type: LayerType::Linear,
            inputs: fc2_inputs,
            outputs: vec![fc2_out.clone()],
            weights: fc2_weight_name.as_ref().map(|w| crate::WeightRef {
                name: w.clone(),
                shape: vec![hidden_dim, hidden_dim * 4], // MLP projection back
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::new(),
        });

        // Add residual
        let add2_out = format!("block{}_add2_out", block_idx);
        layers.push(LayerSpec {
            name: format!("block{}_add2", block_idx),
            layer_type: LayerType::Add,
            inputs: vec![ln1_out, fc2_out],
            outputs: vec![add2_out.clone()],
            weights: None,
            attributes: HashMap::new(),
        });

        // LayerNorm 2 (final_layer_norm in Whisper)
        let ln2_ny_name = find_weight_name(
            weights,
            &[
                &format!("model.encoder.layers.{}.final_layer_norm.weight", block_idx),
                &format!("encoder.layers.{}.final_layer_norm.weight", block_idx),
                &format!("layers.{}.final_layer_norm.weight", block_idx),
            ],
        );
        let ln2_beta_name = find_weight_name(
            weights,
            &[
                &format!("model.encoder.layers.{}.final_layer_norm.bias", block_idx),
                &format!("encoder.layers.{}.final_layer_norm.bias", block_idx),
                &format!("layers.{}.final_layer_norm.bias", block_idx),
            ],
        );
        let ln2_out = format!("block{}_ln2_out", block_idx);
        let mut ln2_inputs = vec![add2_out];
        if let Some(ref g) = ln2_ny_name {
            ln2_inputs.push(g.clone());
        }
        if let Some(ref b) = ln2_beta_name {
            ln2_inputs.push(b.clone());
        }
        layers.push(LayerSpec {
            name: format!("block{}_ln2", block_idx),
            layer_type: LayerType::LayerNorm,
            inputs: ln2_inputs,
            outputs: vec![ln2_out.clone()],
            weights: ln2_ny_name.as_ref().map(|g| crate::WeightRef {
                name: g.clone(),
                shape: vec![hidden_dim],
                original_dtype: DataType::Float32,
            }),
            attributes: {
                let mut attributes = layernorm_attributes(config);
                attributes.insert(
                    "normalized_shape".to_string(),
                    crate::AttributeValue::Ints(vec![hidden_dim as i64]),
                );
                attributes
            },
        });

        prev_output = ln2_out;
    }

    // Final layer norm
    layers.push(LayerSpec {
        name: "ln_post".to_string(),
        layer_type: LayerType::LayerNorm,
        inputs: vec![prev_output],
        outputs: vec!["output".to_string()],
        weights: None,
        attributes: {
            let mut attributes = layernorm_attributes(config);
            attributes.insert(
                "normalized_shape".to_string(),
                crate::AttributeValue::Ints(vec![hidden_dim as i64]),
            );
            attributes
        },
    });

    // Count all parameters
    for (_, w) in weights.iter() {
        param_count += w.len();
    }

    Ok(Network {
        name: "whisper_encoder".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![-1, n_mels as i64, -1], // [batch, n_mels, time]
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![-1, -1, hidden_dim as i64], // [batch, time, hidden]
            dtype: DataType::Float32,
        }],
        layers,
        param_count,
    })
}
