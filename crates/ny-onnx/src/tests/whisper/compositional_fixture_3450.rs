// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{
    AttributeValue, LayerSpec, Network, OnnxModel, WeightStore, WhisperBlockInfo,
    WhisperEncoderStructure, WhisperModel,
};
use ndarray::Array2;
use ny_core::LayerType;
use std::collections::{HashMap, HashSet};

pub(super) fn minimal_whisper_gpu_compositional_fixture_3450() -> WhisperModel {
    let hidden_dim = 4usize;
    let mlp_dim = 8usize;

    let diag = |scale: f32| {
        Array2::from_shape_fn((hidden_dim, hidden_dim), |(row, col)| {
            if row == col {
                scale
            } else {
                0.0
            }
        })
        .into_dyn()
    };
    let expand = Array2::from_shape_fn((mlp_dim, hidden_dim), |(row, col)| {
        if row % hidden_dim == col {
            0.25
        } else {
            0.0
        }
    })
    .into_dyn();
    let contract = Array2::from_shape_fn((hidden_dim, mlp_dim), |(row, col)| {
        if col % hidden_dim == row {
            0.25
        } else {
            0.0
        }
    })
    .into_dyn();

    let mut weights = WeightStore::new();
    let mut param_count = 0usize;
    for (name, tensor) in [
        (
            "encoder.layers.0.self_attn_layer_norm.weight",
            ndarray::Array1::ones(hidden_dim).into_dyn(),
        ),
        (
            "encoder.layers.0.self_attn_layer_norm.bias",
            ndarray::Array1::zeros(hidden_dim).into_dyn(),
        ),
        ("encoder.layers.0.self_attn.q_proj.weight", diag(1.0)),
        (
            "encoder.layers.0.self_attn.q_proj.bias",
            ndarray::Array1::zeros(hidden_dim).into_dyn(),
        ),
        ("encoder.layers.0.self_attn.k_proj.weight", diag(1.0)),
        (
            "encoder.layers.0.self_attn.k_proj.bias",
            ndarray::Array1::zeros(hidden_dim).into_dyn(),
        ),
        ("encoder.layers.0.self_attn.v_proj.weight", diag(1.0)),
        (
            "encoder.layers.0.self_attn.v_proj.bias",
            ndarray::Array1::zeros(hidden_dim).into_dyn(),
        ),
        ("encoder.layers.0.self_attn.out_proj.weight", diag(1.0)),
        (
            "encoder.layers.0.self_attn.out_proj.bias",
            ndarray::Array1::zeros(hidden_dim).into_dyn(),
        ),
        (
            "encoder.layers.0.final_layer_norm.weight",
            ndarray::Array1::ones(hidden_dim).into_dyn(),
        ),
        (
            "encoder.layers.0.final_layer_norm.bias",
            ndarray::Array1::zeros(hidden_dim).into_dyn(),
        ),
        ("encoder.layers.0.fc1.weight", expand),
        (
            "encoder.layers.0.fc1.bias",
            ndarray::Array1::zeros(mlp_dim).into_dyn(),
        ),
        ("encoder.layers.0.fc2.weight", contract),
        (
            "encoder.layers.0.fc2.bias",
            ndarray::Array1::zeros(hidden_dim).into_dyn(),
        ),
    ] {
        param_count += tensor.len();
        weights.insert(name.to_string(), tensor);
    }

    let layer_norm_attrs = HashMap::from([
        (
            "normalized_shape".to_string(),
            AttributeValue::Ints(vec![hidden_dim as i64]),
        ),
        ("epsilon".to_string(), AttributeValue::Float(1e-5)),
    ]);
    let linear_attrs = HashMap::from([("transB".to_string(), AttributeValue::Int(1))]);

    let layer_norm = |name: &str, input: &str, ny: &str, beta: &str, output: &str| -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::LayerNorm,
            inputs: vec![input.to_string(), ny.to_string(), beta.to_string()],
            outputs: vec![output.to_string()],
            weights: None,
            attributes: layer_norm_attrs.clone(),
        }
    };
    let linear = |name: &str, input: &str, weight: &str, bias: &str, output: &str| -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type: LayerType::Linear,
            inputs: vec![input.to_string(), weight.to_string(), bias.to_string()],
            outputs: vec![output.to_string()],
            weights: None,
            attributes: linear_attrs.clone(),
        }
    };

    let layers = vec![
        layer_norm(
            "encoder.layers.0.self_attn_layer_norm",
            "input",
            "encoder.layers.0.self_attn_layer_norm.weight",
            "encoder.layers.0.self_attn_layer_norm.bias",
            "encoder.layers.0.self_attn_layer_norm.out",
        ),
        linear(
            "encoder.layers.0.self_attn.q_proj",
            "encoder.layers.0.self_attn_layer_norm.out",
            "encoder.layers.0.self_attn.q_proj.weight",
            "encoder.layers.0.self_attn.q_proj.bias",
            "encoder.layers.0.self_attn.q_proj.out",
        ),
        linear(
            "encoder.layers.0.self_attn.k_proj",
            "encoder.layers.0.self_attn_layer_norm.out",
            "encoder.layers.0.self_attn.k_proj.weight",
            "encoder.layers.0.self_attn.k_proj.bias",
            "encoder.layers.0.self_attn.k_proj.out",
        ),
        linear(
            "encoder.layers.0.self_attn.v_proj",
            "encoder.layers.0.self_attn_layer_norm.out",
            "encoder.layers.0.self_attn.v_proj.weight",
            "encoder.layers.0.self_attn.v_proj.bias",
            "encoder.layers.0.self_attn.v_proj.out",
        ),
        linear(
            "encoder.layers.0.self_attn.out_proj",
            "encoder.layers.0.self_attn.context",
            "encoder.layers.0.self_attn.out_proj.weight",
            "encoder.layers.0.self_attn.out_proj.bias",
            "encoder.layers.0.self_attn.out_proj.out",
        ),
        layer_norm(
            "encoder.layers.0.final_layer_norm",
            "encoder.layers.0.residual1",
            "encoder.layers.0.final_layer_norm.weight",
            "encoder.layers.0.final_layer_norm.bias",
            "encoder.layers.0.final_layer_norm.out",
        ),
        linear(
            "encoder.layers.0.fc1",
            "encoder.layers.0.final_layer_norm.out",
            "encoder.layers.0.fc1.weight",
            "encoder.layers.0.fc1.bias",
            "encoder.layers.0.fc1.out",
        ),
        LayerSpec {
            name: "encoder.layers.0.gelu".to_string(),
            layer_type: LayerType::GELU,
            inputs: vec!["encoder.layers.0.fc1.out".to_string()],
            outputs: vec!["encoder.layers.0.gelu.out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        },
        linear(
            "encoder.layers.0.fc2",
            "encoder.layers.0.gelu.out",
            "encoder.layers.0.fc2.weight",
            "encoder.layers.0.fc2.bias",
            "encoder.layers.0.fc2.out",
        ),
    ];
    WhisperModel {
        model: OnnxModel {
            network: Network {
                name: "whisper-gpu-soundness-3450".to_string(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                layers,
                param_count,
            },
            weights,
            tensor_producer: HashMap::new(),
            constant_tensors: HashSet::new(),
            tensor_shapes: HashMap::new(),
            original_float32_initializers: HashMap::new(),
            original_network_topology: None,
            opset_imports: HashMap::new(),
        },
        structure: WhisperEncoderStructure {
            stem_end_idx: 0,
            blocks: vec![WhisperBlockInfo {
                index: 0,
                start_layer_idx: 0,
                end_layer_idx: 9,
                num_layers: 9,
            }],
            ln_post_start_idx: 9,
        },
        encoder_layers: 1,
        decoder_layers: 0,
        hidden_dim,
        num_heads: 2,
    }
}
