// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::config::ModelConfig;
use crate::{DataType, LayerSpec, Network, TensorSpec, WeightStore};
use ny_core::{LayerType, Result};
use std::collections::HashMap;

/// Build Kokoro TTS network from weights.
pub(in crate::native) fn build_kokoro_network(
    weights: &WeightStore,
    config: &ModelConfig,
) -> Result<Network> {
    // Kokoro has multiple components: bert, bert_encoder, predictor, decoder, text_encoder
    // For verification, we focus on the forward path

    let mut layers = Vec::new();
    let hidden_dim = config.hidden_dim;

    // This is a simplified structure - actual implementation would need
    // to parse the full model architecture

    // BERT encoder (text processing)
    layers.push(LayerSpec {
        name: "bert_proj".to_string(),
        layer_type: LayerType::Linear,
        inputs: vec!["text_input".to_string()],
        outputs: vec!["bert_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    });

    // Predictor (LSTM-based duration prediction)
    // Note: LSTM not directly supported, would need to unroll

    // Decoder (Conv-based mel generation)
    layers.push(LayerSpec {
        name: "decoder_conv".to_string(),
        layer_type: LayerType::Conv1d,
        inputs: vec!["bert_out".to_string()],
        outputs: vec!["output".to_string()],
        weights: None,
        attributes: HashMap::new(),
    });

    let param_count: usize = weights.iter().map(|(_, w)| w.len()).sum();

    Ok(Network {
        name: "kokoro".to_string(),
        inputs: vec![TensorSpec {
            name: "text_input".to_string(),
            shape: vec![-1, -1, hidden_dim as i64],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "output".to_string(),
            shape: vec![-1, -1, 80], // mel output
            dtype: DataType::Float32,
        }],
        layers,
        param_count,
    })
}
