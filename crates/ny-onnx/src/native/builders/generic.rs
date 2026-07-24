// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::config::ModelConfig;
use crate::{DataType, LayerSpec, Network, TensorSpec, WeightStore};
use ndarray::ArrayD;
use ny_core::{LayerType, NyError, Result};
use std::collections::HashMap;

/// Build generic network from weights (fallback).
pub(in crate::native) fn build_generic_network(
    weights: &WeightStore,
    _config: &ModelConfig,
) -> Result<Network> {
    // For unknown architectures, we create a simple sequential network
    // based on weight shapes

    let mut layers = Vec::new();
    let mut linear_weights: Vec<(&str, &ArrayD<f32>)> = weights
        .iter()
        .filter(|(n, w)| n.contains("weight") && w.ndim() == 2)
        .collect();

    if linear_weights.is_empty() {
        return Err(NyError::ModelLoad(
            "No linear weights found in model".to_string(),
        ));
    }

    // Sort by name
    linear_weights.sort_by_key(|&(name, _)| name);

    let input_dim = linear_weights[0].1.shape()[1];
    let output_dim = linear_weights
        .last()
        .map(|(_, w)| w.shape()[0])
        .unwrap_or(input_dim);

    let mut prev_output = "input".to_string();

    for (idx, (name, weight)) in linear_weights.iter().enumerate() {
        let in_features = weight.shape()[1];
        let out_features = weight.shape()[0];

        let output_name = if idx == linear_weights.len() - 1 {
            "output".to_string()
        } else {
            format!("layer{}_out", idx)
        };

        layers.push(LayerSpec {
            name: format!("layer{}", idx),
            layer_type: LayerType::Linear,
            inputs: vec![prev_output.clone()],
            outputs: vec![output_name.clone()],
            weights: Some(crate::WeightRef {
                name: (*name).to_string(),
                shape: vec![out_features, in_features],
                original_dtype: DataType::Float32,
            }),
            attributes: HashMap::new(),
        });

        prev_output = output_name;
    }

    let param_count: usize = weights.iter().map(|(_, w)| w.len()).sum();

    Ok(Network {
        name: "generic".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![-1, input_dim as i64],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![-1, output_dim as i64],
            dtype: DataType::Float32,
        }],
        layers,
        param_count,
    })
}
