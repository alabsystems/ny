// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::config::ModelConfig;
use super::super::helpers::{extract_layer_number, find_weight_name};
use crate::{DataType, LayerSpec, Network, TensorSpec, WeightStore};
use ndarray::ArrayD;
use ny_core::{LayerType, Result};
use std::collections::HashMap;

/// Build MLP network from weights.
pub(in crate::native) fn build_mlp_network(
    weights: &WeightStore,
    config: &ModelConfig,
) -> Result<Network> {
    let mut layers = Vec::new();
    let mut layer_weights: Vec<(&str, &ArrayD<f32>)> = weights
        .iter()
        .filter(|(n, w)| n.contains("weight") && w.ndim() == 2)
        .collect();

    // Sort by layer number
    layer_weights.sort_by(|(a, _), (b, _)| {
        extract_layer_number(a)
            .unwrap_or(0)
            .cmp(&extract_layer_number(b).unwrap_or(0))
    });

    let hidden_dim = config.hidden_dim;
    let input_dim = config.input_dim.unwrap_or(hidden_dim);
    let output_dim = config.output_dim.unwrap_or_else(|| {
        layer_weights
            .last()
            .map(|(_, w)| w.shape()[0])
            .unwrap_or(hidden_dim)
    });

    let mut prev_output = "input".to_string();

    for (idx, (name, weight)) in layer_weights.iter().enumerate() {
        let in_features = weight.shape()[1];
        let out_features = weight.shape()[0];
        let is_last = idx == layer_weights.len() - 1;

        let output_name = if is_last {
            "output".to_string()
        } else {
            format!("layer{}_out", idx)
        };

        // Find bias
        let bias_name_pattern = name.replace("weight", "bias");
        let bias_name = find_weight_name(weights, &[&bias_name_pattern]);

        let mut inputs = vec![prev_output.clone()];
        if let Some(ref bn) = bias_name {
            inputs.push(bn.clone());
        }

        layers.push(LayerSpec {
            name: format!("layer{}", idx),
            layer_type: LayerType::Linear,
            inputs,
            outputs: vec![output_name.clone()],
            weights: Some(crate::WeightRef {
                name: (*name).to_string(),
                shape: vec![out_features, in_features],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::new(),
        });

        // Add ReLU after every layer except the last
        if !is_last {
            let relu_out = format!("layer{}_relu_out", idx);
            layers.push(LayerSpec {
                name: format!("layer{}_relu", idx),
                layer_type: LayerType::ReLU,
                inputs: vec![output_name.clone()],
                outputs: vec![relu_out.clone()],
                weights: None,
                attributes: HashMap::new(),
            });
            prev_output = relu_out;
        } else {
            prev_output = output_name;
        }
    }

    let param_count: usize = weights.iter().map(|(_, w)| w.len()).sum();
    let last_out_dim = layer_weights
        .last()
        .map(|(_, w)| w.shape()[0])
        .unwrap_or(output_dim);

    Ok(Network {
        name: "mlp".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![-1, input_dim as i64],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![-1, last_out_dim as i64],
            dtype: DataType::Float32,
        }],
        layers,
        param_count,
    })
}
