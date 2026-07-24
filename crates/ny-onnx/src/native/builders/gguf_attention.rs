// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{DataType, LayerSpec, WeightStore};
use ny_core::LayerType;
use std::collections::HashMap;
use tracing::info;

/// Generate self-attention layers for GGUF LLM (causal attention).
///
/// Supports both standard Multi-Head Attention (MHA) and Grouped Query Attention (GQA).
/// In GQA, K and V have fewer heads than Q, so we expand them using Tile operations.
pub(super) fn generate_gguf_attention(
    weights: &WeightStore,
    prefix: &str,
    input: &str,
    hidden_dim: usize,
    num_heads: usize,
    layer_idx: usize,
) -> (Vec<LayerSpec>, String) {
    let mut layers = Vec::new();

    // Q projection - read actual output dimension from weight
    // GGUF stores weights as [in_dim, out_dim]
    let q_name = format!("{}.attn_q.weight", prefix);
    let q_dim = weights
        .get(&q_name)
        .map(|w| w.shape()[1])
        .unwrap_or(hidden_dim);

    // Calculate head_dim from Q dimension and num_heads
    // In GQA models, Q dim may differ from hidden_dim
    let head_dim = q_dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_out = format!("layer{}_q", layer_idx);
    layers.push(LayerSpec {
        name: format!("layer{}_q_proj", layer_idx),
        layer_type: LayerType::Linear,
        inputs: vec![input.to_string(), q_name.clone()],
        outputs: vec![q_out.clone()],
        weights: Some(crate::WeightRef {
            name: q_name,
            // LinearLayer expects (out_features, in_features)
            shape: vec![q_dim, hidden_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::new(),
    });

    // K projection
    let k_name = format!("{}.attn_k.weight", prefix);
    let k_dim = weights
        .get(&k_name)
        .map(|w| w.shape()[1])
        .unwrap_or(hidden_dim);
    let k_out_proj = format!("layer{}_k_proj_out", layer_idx);
    layers.push(LayerSpec {
        name: format!("layer{}_k_proj", layer_idx),
        layer_type: LayerType::Linear,
        inputs: vec![input.to_string(), k_name.clone()],
        outputs: vec![k_out_proj.clone()],
        weights: Some(crate::WeightRef {
            name: k_name,
            // LinearLayer expects (out_features, in_features)
            shape: vec![k_dim, hidden_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::new(),
    });

    // V projection
    let v_name = format!("{}.attn_v.weight", prefix);
    let v_dim = weights
        .get(&v_name)
        .map(|w| w.shape()[1])
        .unwrap_or(hidden_dim);
    let v_out_proj = format!("layer{}_v_proj_out", layer_idx);
    layers.push(LayerSpec {
        name: format!("layer{}_v_proj", layer_idx),
        layer_type: LayerType::Linear,
        inputs: vec![input.to_string(), v_name.clone()],
        outputs: vec![v_out_proj.clone()],
        weights: Some(crate::WeightRef {
            name: v_name,
            // LinearLayer expects (out_features, in_features)
            shape: vec![v_dim, hidden_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::new(),
    });

    // GQA support: expand K and V if they have fewer heads than Q
    // head_dim is derived from Q dimension (q_dim / num_heads)
    let (k_out, v_out) = if k_dim != q_dim && k_dim > 0 && head_dim > 0 {
        // GQA mode: k_dim < q_dim
        // num_kv_heads = k_dim / head_dim
        // groups = num_heads / num_kv_heads
        let num_kv_heads = k_dim / head_dim;
        let groups = num_heads.checked_div(num_kv_heads).unwrap_or(1);

        info!(
            "GQA detected: {} Q heads (q_dim={}), {} KV heads (k_dim={}), {} groups, head_dim={}",
            num_heads, q_dim, num_kv_heads, k_dim, groups, head_dim
        );

        // Expand K: [seq, k_dim] -> [seq, q_dim]
        // Step 1: Reshape to [seq, num_kv_heads, 1, head_dim]
        let k_reshaped = format!("layer{}_k_reshaped", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_k_reshape1", layer_idx),
            layer_type: LayerType::Reshape,
            inputs: vec![k_out_proj],
            outputs: vec![k_reshaped.clone()],
            weights: None,
            attributes: HashMap::from([(
                "shape".to_string(),
                crate::AttributeValue::Ints(vec![-1, num_kv_heads as i64, 1, head_dim as i64]),
            )]),
        });

        // Step 2: Tile along axis 2 (the "1" dimension) to repeat groups times
        let k_tiled = format!("layer{}_k_tiled", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_k_tile", layer_idx),
            layer_type: LayerType::Tile,
            inputs: vec![k_reshaped],
            outputs: vec![k_tiled.clone()],
            weights: None,
            attributes: HashMap::from([
                ("axis".to_string(), crate::AttributeValue::Int(2)),
                (
                    "reps".to_string(),
                    crate::AttributeValue::Int(groups as i64),
                ),
            ]),
        });

        // Step 3: Reshape back to [seq, q_dim] (expanded K matches Q dimension)
        let k_expanded = format!("layer{}_k", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_k_reshape2", layer_idx),
            layer_type: LayerType::Reshape,
            inputs: vec![k_tiled],
            outputs: vec![k_expanded.clone()],
            weights: None,
            attributes: HashMap::from([(
                "shape".to_string(),
                crate::AttributeValue::Ints(vec![-1, q_dim as i64]),
            )]),
        });

        // Expand V: [seq, v_dim] -> [seq, q_dim]
        // (same process as K)
        let v_reshaped = format!("layer{}_v_reshaped", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_v_reshape1", layer_idx),
            layer_type: LayerType::Reshape,
            inputs: vec![v_out_proj],
            outputs: vec![v_reshaped.clone()],
            weights: None,
            attributes: HashMap::from([(
                "shape".to_string(),
                crate::AttributeValue::Ints(vec![-1, num_kv_heads as i64, 1, head_dim as i64]),
            )]),
        });

        let v_tiled = format!("layer{}_v_tiled", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_v_tile", layer_idx),
            layer_type: LayerType::Tile,
            inputs: vec![v_reshaped],
            outputs: vec![v_tiled.clone()],
            weights: None,
            attributes: HashMap::from([
                ("axis".to_string(), crate::AttributeValue::Int(2)),
                (
                    "reps".to_string(),
                    crate::AttributeValue::Int(groups as i64),
                ),
            ]),
        });

        let v_expanded = format!("layer{}_v", layer_idx);
        layers.push(LayerSpec {
            name: format!("layer{}_v_reshape2", layer_idx),
            layer_type: LayerType::Reshape,
            inputs: vec![v_tiled],
            outputs: vec![v_expanded.clone()],
            weights: None,
            attributes: HashMap::from([(
                "shape".to_string(),
                crate::AttributeValue::Ints(vec![-1, q_dim as i64]),
            )]),
        });

        (k_expanded, v_expanded)
    } else {
        // Standard MHA: K and V already have same dimensions as Q
        (
            k_out_proj.replace("_proj_out", ""),
            v_out_proj.replace("_proj_out", ""),
        )
    };

    // Rename outputs only if we didn't go through GQA path (k_dim == q_dim means standard MHA)
    // In GQA mode (k_dim != q_dim), the reshape layers depend on k_out_proj/v_out_proj, so we must NOT rename
    let (k_out, v_out) = if k_dim == q_dim {
        // In standard MHA, rename the projection outputs
        let k_renamed = format!("layer{}_k", layer_idx);
        let v_renamed = format!("layer{}_v", layer_idx);

        // Update the last K projection output name
        if let Some(layer) = layers
            .iter_mut()
            .rev()
            .find(|l| l.name.ends_with("_k_proj"))
        {
            layer.outputs = vec![k_renamed.clone()];
        }
        // Update the last V projection output name
        if let Some(layer) = layers
            .iter_mut()
            .rev()
            .find(|l| l.name.ends_with("_v_proj"))
        {
            layer.outputs = vec![v_renamed.clone()];
        }

        (k_renamed, v_renamed)
    } else {
        (k_out, v_out)
    };

    // MatMul Q @ K^T (use transpose_b attribute so propagate can exploit structure)
    let qk_out = format!("layer{}_qk", layer_idx);
    layers.push(LayerSpec {
        name: format!("layer{}_qk_matmul", layer_idx),
        layer_type: LayerType::MatMul,
        inputs: vec![q_out, k_out],
        outputs: vec![qk_out.clone()],
        weights: None,
        attributes: HashMap::from([
            ("transpose_b".to_string(), crate::AttributeValue::Int(1)),
            ("scale".to_string(), crate::AttributeValue::Float(scale)),
        ]),
    });

    // Causal softmax (decoder uses causal attention)
    let softmax_out = format!("layer{}_softmax", layer_idx);
    layers.push(LayerSpec {
        name: format!("layer{}_softmax", layer_idx),
        layer_type: LayerType::CausalSoftmax,
        inputs: vec![qk_out],
        outputs: vec![softmax_out.clone()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), crate::AttributeValue::Int(-1))]),
    });

    // MatMul attention @ V
    let attn_v_out = format!("layer{}_attn_v", layer_idx);
    layers.push(LayerSpec {
        name: format!("layer{}_attn_v_matmul", layer_idx),
        layer_type: LayerType::MatMul,
        inputs: vec![softmax_out, v_out],
        outputs: vec![attn_v_out.clone()],
        weights: None,
        attributes: HashMap::new(),
    });

    // Output projection
    // GGUF: attn_output.weight shape is [attn_dim, hidden_dim] = [q_dim, hidden_dim]
    // LinearLayer expects (out_features, in_features) = (hidden_dim, attn_dim)
    let out_name = format!("{}.attn_output.weight", prefix);
    // shape()[0] = input dim (attention dim = q_dim), shape()[1] = output dim (hidden_dim)
    let attn_dim = weights
        .get(&out_name)
        .map(|w| w.shape()[0])
        .unwrap_or(q_dim);
    let attn_out = format!("layer{}_attn_out", layer_idx);
    layers.push(LayerSpec {
        name: format!("layer{}_out_proj", layer_idx),
        layer_type: LayerType::Linear,
        inputs: vec![attn_v_out, out_name.clone()],
        outputs: vec![attn_out.clone()],
        weights: Some(crate::WeightRef {
            name: out_name,
            // LinearLayer expects (out_features, in_features)
            shape: vec![hidden_dim, attn_dim],
            original_dtype: DataType::Float32,
        }),
        attributes: HashMap::new(),
    });

    (layers, attn_out)
}
