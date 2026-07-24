// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::structure::parse_whisper_structure;
use ny_core::{LayerType, NyError, Result};
use std::path::Path;
use tracing::warn;

use super::super::WhisperModel;

/// Load a Whisper model specifically.
pub fn load_whisper<P: AsRef<Path>>(path: P) -> Result<WhisperModel> {
    let model = crate::load_onnx(path)?;

    // Parse block structure from layer names
    let mut structure = parse_whisper_structure(&model.network)?;
    if structure.blocks.is_empty() {
        return Err(NyError::InvalidSpec(
            "Whisper model has no encoder blocks; unable to parse structure".to_string(),
        ));
    }
    if structure.ln_post_start_idx >= model.network.layers.len() {
        warn!("Whisper model missing ln_post layer; continuing without final LayerNorm");
    }
    if let Some(last_block) = structure.blocks.last() {
        if structure.ln_post_start_idx < last_block.end_layer_idx {
            warn!(
                "Whisper ln_post appears before the final encoder block; treating ln_post as missing"
            );
            structure.ln_post_start_idx = model.network.layers.len();
        }
    }
    let encoder_layers = structure.blocks.len();

    // Detect model size from hidden dimension (first LayerNorm ny size)
    let hidden_dim = model
        .network
        .layers
        .iter()
        .find(|l| l.layer_type == LayerType::LayerNorm)
        .and_then(|l| l.inputs.get(1))
        .and_then(|ny_name| model.weights.get(ny_name))
        .map(|ny| ny.len())
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "Whisper LayerNorm ny not found; unable to infer hidden_dim".to_string(),
            )
        })?;

    // Calculate num_heads from hidden_dim (Whisper uses head_dim=64)
    if hidden_dim % 64 != 0 {
        return Err(NyError::InvalidSpec(format!(
            "Whisper hidden_dim {} is not divisible by 64",
            hidden_dim
        )));
    }
    let num_heads = hidden_dim / 64;
    if num_heads == 0 {
        return Err(NyError::InvalidSpec(format!(
            "Whisper hidden_dim {} yields zero heads",
            hidden_dim
        )));
    }

    Ok(WhisperModel {
        model,
        structure,
        encoder_layers,
        decoder_layers: encoder_layers, // Whisper has symmetric encoder/decoder
        hidden_dim,
        num_heads,
    })
}
