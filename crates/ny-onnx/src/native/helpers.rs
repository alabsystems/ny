// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::config::ModelConfig;
use crate::{AttributeValue, DataType, LayerSpec, WeightStore};
use ndarray::ArrayD;
use ny_core::LayerType;
use ny_propagate::layers::LayerNormMode;
use std::collections::HashMap;

/// Find a weight by trying multiple possible names.
pub(super) fn find_weight<'a>(weights: &'a WeightStore, names: &[&str]) -> Option<&'a ArrayD<f32>> {
    for name in names {
        if let Some(w) = weights.get(name) {
            return Some(w);
        }
        // Try with common prefixes
        for prefix in ["", "model.", "encoder.", "model.encoder."] {
            let full_name = format!("{}{}", prefix, name);
            if let Some(w) = weights.get(&full_name) {
                return Some(w);
            }
        }
    }
    None
}

/// Find the actual weight name from a list of possible names.
pub(super) fn find_weight_name(weights: &WeightStore, names: &[&str]) -> Option<String> {
    for name in names {
        if weights.get(name).is_some() {
            return Some((*name).to_string());
        }
        for prefix in ["", "model.", "encoder.", "model.encoder."] {
            let full_name = format!("{}{}", prefix, name);
            if weights.get(&full_name).is_some() {
                return Some(full_name);
            }
        }
    }
    None
}

/// Extract layer number from name like "layer_3" or "fc3".
pub(crate) fn extract_layer_number(name: &str) -> Option<usize> {
    // Try various patterns
    for pattern in ["layer_", "layer", "fc", "linear", "."] {
        if let Some(idx) = name.find(pattern) {
            let rest = &name[idx + pattern.len()..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(num) = num_str.parse() {
                return Some(num);
            }
        }
    }
    None
}

pub(super) fn layernorm_attributes(config: &ModelConfig) -> HashMap<String, AttributeValue> {
    let mut attributes = HashMap::new();
    if matches!(config.layernorm_mode, LayerNormMode::MeanOnly) {
        attributes.insert(
            "layernorm_mode".to_string(),
            AttributeValue::String("mean_only".to_string()),
        );
    }
    attributes
}

/// Generate decomposed self-attention layers.
///
/// This creates the constituent layers for multi-head self-attention:
/// Q, K, V projections → Q @ K^T → Scale → Softmax → Attention @ V → Output projection
///
/// Returns (layers, output_name) where output_name is the tensor name of the attention output.
pub(super) fn generate_decomposed_attention(
    weights: &WeightStore,
    prefix: &str,
    input: &str,
    hidden_dim: usize,
    num_heads: usize,
    is_causal: bool,
) -> (Vec<LayerSpec>, String) {
    let mut layers = Vec::new();
    let head_dim = hidden_dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Extract block index from prefix (e.g., "block0" -> "0")
    let block_idx = prefix
        .trim_start_matches("block")
        .trim_start_matches("layer");

    // Q projection
    let q_weight_name = find_weight_name(
        weights,
        &[
            &format!("model.encoder.layers.{}.self_attn.q_proj.weight", block_idx),
            &format!("model.encoder.layer.{}.self_attn.q_proj.weight", block_idx),
            &format!("model.encoder.layers.{}.attn.q_proj.weight", block_idx),
            &format!("model.encoder.layer.{}.attn.q_proj.weight", block_idx),
            &format!("encoder.layers.{}.self_attn.q_proj.weight", block_idx),
            &format!("encoder.layer.{}.self_attn.q_proj.weight", block_idx),
            &format!("encoder.layers.{}.attn.q_proj.weight", block_idx),
            &format!("encoder.layer.{}.attn.q_proj.weight", block_idx),
            &format!("layers.{}.self_attn.q_proj.weight", block_idx),
            &format!("layers_{}.self_attn.q_proj.weight", block_idx),
            &format!("layer.{}.self_attn.q_proj.weight", block_idx),
            &format!("layer_{}.self_attn.q_proj.weight", block_idx),
            &format!("layers.{}.attn.q_proj.weight", block_idx),
            &format!("layers_{}.attn.q_proj.weight", block_idx),
            &format!("layer.{}.attn.q_proj.weight", block_idx),
            &format!("layer_{}.attn.q_proj.weight", block_idx),
            &format!("{}.q_proj.weight", prefix),
            &format!("{}.self_attn.q_proj.weight", prefix),
            &format!("{}.attn.q_proj.weight", prefix),
        ],
    );
    let q_bias_name = find_weight_name(
        weights,
        &[
            &format!("model.encoder.layers.{}.self_attn.q_proj.bias", block_idx),
            &format!("model.encoder.layer.{}.self_attn.q_proj.bias", block_idx),
            &format!("model.encoder.layers.{}.attn.q_proj.bias", block_idx),
            &format!("model.encoder.layer.{}.attn.q_proj.bias", block_idx),
            &format!("encoder.layers.{}.self_attn.q_proj.bias", block_idx),
            &format!("encoder.layer.{}.self_attn.q_proj.bias", block_idx),
            &format!("encoder.layers.{}.attn.q_proj.bias", block_idx),
            &format!("encoder.layer.{}.attn.q_proj.bias", block_idx),
            &format!("layers.{}.self_attn.q_proj.bias", block_idx),
            &format!("layers_{}.self_attn.q_proj.bias", block_idx),
            &format!("layer.{}.self_attn.q_proj.bias", block_idx),
            &format!("layer_{}.self_attn.q_proj.bias", block_idx),
            &format!("layers.{}.attn.q_proj.bias", block_idx),
            &format!("layers_{}.attn.q_proj.bias", block_idx),
            &format!("layer.{}.attn.q_proj.bias", block_idx),
            &format!("layer_{}.attn.q_proj.bias", block_idx),
            &format!("{}.q_proj.bias", prefix),
            &format!("{}.self_attn.q_proj.bias", prefix),
            &format!("{}.attn.q_proj.bias", prefix),
        ],
    );
    let q_out = format!("{}_q", prefix);
    let mut q_inputs = vec![input.to_string()];
    if let Some(ref w) = q_weight_name {
        q_inputs.push(w.clone());
    }
    if let Some(ref b) = q_bias_name {
        q_inputs.push(b.clone());
    }
    layers.push(LayerSpec {
        name: format!("{}_q_proj", prefix),
        layer_type: LayerType::Linear,
        inputs: q_inputs,
        outputs: vec![q_out.clone()],
        weights: q_weight_name.as_ref().map(|w| crate::WeightRef {
            name: w.clone(),
            shape: vec![hidden_dim, hidden_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::new(),
    });

    // K projection
    let k_weight_name = find_weight_name(
        weights,
        &[
            &format!("model.encoder.layers.{}.self_attn.k_proj.weight", block_idx),
            &format!("model.encoder.layer.{}.self_attn.k_proj.weight", block_idx),
            &format!("model.encoder.layers.{}.attn.k_proj.weight", block_idx),
            &format!("model.encoder.layer.{}.attn.k_proj.weight", block_idx),
            &format!("encoder.layers.{}.self_attn.k_proj.weight", block_idx),
            &format!("encoder.layer.{}.self_attn.k_proj.weight", block_idx),
            &format!("encoder.layers.{}.attn.k_proj.weight", block_idx),
            &format!("encoder.layer.{}.attn.k_proj.weight", block_idx),
            &format!("layers.{}.self_attn.k_proj.weight", block_idx),
            &format!("layers_{}.self_attn.k_proj.weight", block_idx),
            &format!("layer.{}.self_attn.k_proj.weight", block_idx),
            &format!("layer_{}.self_attn.k_proj.weight", block_idx),
            &format!("layers.{}.attn.k_proj.weight", block_idx),
            &format!("layers_{}.attn.k_proj.weight", block_idx),
            &format!("layer.{}.attn.k_proj.weight", block_idx),
            &format!("layer_{}.attn.k_proj.weight", block_idx),
            &format!("{}.k_proj.weight", prefix),
            &format!("{}.self_attn.k_proj.weight", prefix),
            &format!("{}.attn.k_proj.weight", prefix),
        ],
    );
    let k_bias_name = find_weight_name(
        weights,
        &[
            &format!("model.encoder.layers.{}.self_attn.k_proj.bias", block_idx),
            &format!("model.encoder.layer.{}.self_attn.k_proj.bias", block_idx),
            &format!("model.encoder.layers.{}.attn.k_proj.bias", block_idx),
            &format!("model.encoder.layer.{}.attn.k_proj.bias", block_idx),
            &format!("encoder.layers.{}.self_attn.k_proj.bias", block_idx),
            &format!("encoder.layer.{}.self_attn.k_proj.bias", block_idx),
            &format!("encoder.layers.{}.attn.k_proj.bias", block_idx),
            &format!("encoder.layer.{}.attn.k_proj.bias", block_idx),
            &format!("layers.{}.self_attn.k_proj.bias", block_idx),
            &format!("layers_{}.self_attn.k_proj.bias", block_idx),
            &format!("layer.{}.self_attn.k_proj.bias", block_idx),
            &format!("layer_{}.self_attn.k_proj.bias", block_idx),
            &format!("layers.{}.attn.k_proj.bias", block_idx),
            &format!("layers_{}.attn.k_proj.bias", block_idx),
            &format!("layer.{}.attn.k_proj.bias", block_idx),
            &format!("layer_{}.attn.k_proj.bias", block_idx),
            &format!("{}.k_proj.bias", prefix),
            &format!("{}.self_attn.k_proj.bias", prefix),
            &format!("{}.attn.k_proj.bias", prefix),
        ],
    );
    let k_out = format!("{}_k", prefix);
    let mut k_inputs = vec![input.to_string()];
    if let Some(ref w) = k_weight_name {
        k_inputs.push(w.clone());
    }
    if let Some(ref b) = k_bias_name {
        k_inputs.push(b.clone());
    }
    layers.push(LayerSpec {
        name: format!("{}_k_proj", prefix),
        layer_type: LayerType::Linear,
        inputs: k_inputs,
        outputs: vec![k_out.clone()],
        weights: k_weight_name.as_ref().map(|w| crate::WeightRef {
            name: w.clone(),
            shape: vec![hidden_dim, hidden_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::new(),
    });

    // V projection
    let v_weight_name = find_weight_name(
        weights,
        &[
            &format!("model.encoder.layers.{}.self_attn.v_proj.weight", block_idx),
            &format!("model.encoder.layer.{}.self_attn.v_proj.weight", block_idx),
            &format!("model.encoder.layers.{}.attn.v_proj.weight", block_idx),
            &format!("model.encoder.layer.{}.attn.v_proj.weight", block_idx),
            &format!("encoder.layers.{}.self_attn.v_proj.weight", block_idx),
            &format!("encoder.layer.{}.self_attn.v_proj.weight", block_idx),
            &format!("encoder.layers.{}.attn.v_proj.weight", block_idx),
            &format!("encoder.layer.{}.attn.v_proj.weight", block_idx),
            &format!("layers.{}.self_attn.v_proj.weight", block_idx),
            &format!("layers_{}.self_attn.v_proj.weight", block_idx),
            &format!("layer.{}.self_attn.v_proj.weight", block_idx),
            &format!("layer_{}.self_attn.v_proj.weight", block_idx),
            &format!("layers.{}.attn.v_proj.weight", block_idx),
            &format!("layers_{}.attn.v_proj.weight", block_idx),
            &format!("layer.{}.attn.v_proj.weight", block_idx),
            &format!("layer_{}.attn.v_proj.weight", block_idx),
            &format!("{}.v_proj.weight", prefix),
            &format!("{}.self_attn.v_proj.weight", prefix),
            &format!("{}.attn.v_proj.weight", prefix),
        ],
    );
    let v_bias_name = find_weight_name(
        weights,
        &[
            &format!("model.encoder.layers.{}.self_attn.v_proj.bias", block_idx),
            &format!("model.encoder.layer.{}.self_attn.v_proj.bias", block_idx),
            &format!("model.encoder.layers.{}.attn.v_proj.bias", block_idx),
            &format!("model.encoder.layer.{}.attn.v_proj.bias", block_idx),
            &format!("encoder.layers.{}.self_attn.v_proj.bias", block_idx),
            &format!("encoder.layer.{}.self_attn.v_proj.bias", block_idx),
            &format!("encoder.layers.{}.attn.v_proj.bias", block_idx),
            &format!("encoder.layer.{}.attn.v_proj.bias", block_idx),
            &format!("layers.{}.self_attn.v_proj.bias", block_idx),
            &format!("layers_{}.self_attn.v_proj.bias", block_idx),
            &format!("layer.{}.self_attn.v_proj.bias", block_idx),
            &format!("layer_{}.self_attn.v_proj.bias", block_idx),
            &format!("layers.{}.attn.v_proj.bias", block_idx),
            &format!("layers_{}.attn.v_proj.bias", block_idx),
            &format!("layer.{}.attn.v_proj.bias", block_idx),
            &format!("layer_{}.attn.v_proj.bias", block_idx),
            &format!("{}.v_proj.bias", prefix),
            &format!("{}.self_attn.v_proj.bias", prefix),
            &format!("{}.attn.v_proj.bias", prefix),
        ],
    );
    let v_out = format!("{}_v", prefix);
    let mut v_inputs = vec![input.to_string()];
    if let Some(ref w) = v_weight_name {
        v_inputs.push(w.clone());
    }
    if let Some(ref b) = v_bias_name {
        v_inputs.push(b.clone());
    }
    layers.push(LayerSpec {
        name: format!("{}_v_proj", prefix),
        layer_type: LayerType::Linear,
        inputs: v_inputs,
        outputs: vec![v_out.clone()],
        weights: v_weight_name.as_ref().map(|w| crate::WeightRef {
            name: w.clone(),
            shape: vec![hidden_dim, hidden_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::new(),
    });

    // MatMul Q @ K^T (use transpose_b attribute so propagate can exploit structure)
    let qk_out = format!("{}_qk", prefix);
    layers.push(LayerSpec {
        name: format!("{}_qk_matmul", prefix),
        layer_type: LayerType::MatMul,
        inputs: vec![q_out, k_out],
        outputs: vec![qk_out.clone()],
        weights: None,
        attributes: HashMap::from([
            ("transpose_b".to_string(), AttributeValue::Int(1)),
            ("scale".to_string(), AttributeValue::Float(scale)),
        ]),
    });

    // Softmax (or causal softmax)
    let softmax_out = format!("{}_softmax", prefix);
    layers.push(LayerSpec {
        name: format!("{}_softmax", prefix),
        layer_type: if is_causal {
            LayerType::CausalSoftmax
        } else {
            LayerType::Softmax
        },
        inputs: vec![qk_out],
        outputs: vec![softmax_out.clone()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(-1))]),
    });

    // MatMul attention_probs @ V
    let attn_v_out = format!("{}_attn_v", prefix);
    layers.push(LayerSpec {
        name: format!("{}_attn_v_matmul", prefix),
        layer_type: LayerType::MatMul,
        inputs: vec![softmax_out, v_out],
        outputs: vec![attn_v_out.clone()],
        weights: None,
        attributes: HashMap::new(),
    });

    // Output projection
    let out_weight_name = find_weight_name(
        weights,
        &[
            &format!(
                "model.encoder.layers.{}.self_attn.out_proj.weight",
                block_idx
            ),
            &format!(
                "model.encoder.layer.{}.self_attn.out_proj.weight",
                block_idx
            ),
            &format!("model.encoder.layers.{}.attn.out_proj.weight", block_idx),
            &format!("model.encoder.layer.{}.attn.out_proj.weight", block_idx),
            &format!("encoder.layers.{}.self_attn.out_proj.weight", block_idx),
            &format!("encoder.layer.{}.self_attn.out_proj.weight", block_idx),
            &format!("encoder.layers.{}.attn.out_proj.weight", block_idx),
            &format!("encoder.layer.{}.attn.out_proj.weight", block_idx),
            &format!("layers.{}.self_attn.out_proj.weight", block_idx),
            &format!("layers_{}.self_attn.out_proj.weight", block_idx),
            &format!("layer.{}.self_attn.out_proj.weight", block_idx),
            &format!("layer_{}.self_attn.out_proj.weight", block_idx),
            &format!("layers.{}.attn.out_proj.weight", block_idx),
            &format!("layers_{}.attn.out_proj.weight", block_idx),
            &format!("layer.{}.attn.out_proj.weight", block_idx),
            &format!("layer_{}.attn.out_proj.weight", block_idx),
            &format!("{}.out_proj.weight", prefix),
            &format!("{}.self_attn.out_proj.weight", prefix),
            &format!("{}.attn.out_proj.weight", prefix),
        ],
    );
    let out_bias_name = find_weight_name(
        weights,
        &[
            &format!("model.encoder.layers.{}.self_attn.out_proj.bias", block_idx),
            &format!("model.encoder.layer.{}.self_attn.out_proj.bias", block_idx),
            &format!("model.encoder.layers.{}.attn.out_proj.bias", block_idx),
            &format!("model.encoder.layer.{}.attn.out_proj.bias", block_idx),
            &format!("encoder.layers.{}.self_attn.out_proj.bias", block_idx),
            &format!("encoder.layer.{}.self_attn.out_proj.bias", block_idx),
            &format!("encoder.layers.{}.attn.out_proj.bias", block_idx),
            &format!("encoder.layer.{}.attn.out_proj.bias", block_idx),
            &format!("layers.{}.self_attn.out_proj.bias", block_idx),
            &format!("layers_{}.self_attn.out_proj.bias", block_idx),
            &format!("layer.{}.self_attn.out_proj.bias", block_idx),
            &format!("layer_{}.self_attn.out_proj.bias", block_idx),
            &format!("layers.{}.attn.out_proj.bias", block_idx),
            &format!("layers_{}.attn.out_proj.bias", block_idx),
            &format!("layer.{}.attn.out_proj.bias", block_idx),
            &format!("layer_{}.attn.out_proj.bias", block_idx),
            &format!("{}.out_proj.bias", prefix),
            &format!("{}.self_attn.out_proj.bias", prefix),
            &format!("{}.attn.out_proj.bias", prefix),
        ],
    );
    let attn_out = format!("{}_out", prefix);
    let mut out_inputs = vec![attn_v_out];
    if let Some(ref w) = out_weight_name {
        out_inputs.push(w.clone());
    }
    if let Some(ref b) = out_bias_name {
        out_inputs.push(b.clone());
    }
    layers.push(LayerSpec {
        name: format!("{}_out_proj", prefix),
        layer_type: LayerType::Linear,
        inputs: out_inputs,
        outputs: vec![attn_out.clone()],
        weights: out_weight_name.as_ref().map(|w| crate::WeightRef {
            name: w.clone(),
            shape: vec![hidden_dim, hidden_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::new(),
    });

    (layers, attn_out)
}
