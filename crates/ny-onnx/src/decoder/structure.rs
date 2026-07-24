// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decoder structure parsing from ONNX layer naming patterns.

use crate::OnnxModel;
use ny_core::{LayerType, Result};
use tracing::info;

use super::{DecoderBlockInfo, DecoderModel, DecoderStructure};

/// Load a decoder transformer model.
///
/// Supports both decoder-only models (like LLaMA) and encoder-decoder decoder blocks
/// (like Whisper decoder). The model structure is parsed from layer naming patterns.
///
/// # Naming Patterns Supported
/// - Single block: `/norm1/...`, `/self_attn/...`, `/mlp/...`
/// - Multi-block: `/blocks.{i}/norm1/...` or `/decoder.blocks.{i}/...`
///
/// # Arguments
/// * `path` - Path to ONNX model file
///
/// # Returns
/// DecoderModel with parsed structure for compositional verification.
pub fn load_decoder<P: AsRef<std::path::Path>>(path: P) -> Result<DecoderModel> {
    let model = crate::load_onnx(path)?;
    let structure = parse_decoder_structure(&model)?;

    let num_blocks = structure.blocks.len().max(1);
    let hidden_dim = structure.hidden_dim;
    let num_heads = structure.num_heads;

    Ok(DecoderModel {
        model,
        structure,
        num_blocks,
        hidden_dim,
        num_heads,
    })
}

/// Parse decoder structure from layer naming patterns.
pub(super) fn parse_decoder_structure(model: &OnnxModel) -> Result<DecoderStructure> {
    let mut blocks = Vec::new();
    let mut hidden_dim = 0;
    let mut num_heads = 4; // Default, will be inferred if possible

    // Detect if this is a single-block or multi-block model
    let has_block_indices = model
        .network
        .layers
        .iter()
        .any(|l| l.name.contains("blocks.") || l.name.contains("decoder.blocks."));

    if has_block_indices {
        // Multi-block model: Parse block indices
        let mut seen_blocks = std::collections::HashSet::new();
        for layer in &model.network.layers {
            if let Some(idx) = parse_decoder_block_index(&layer.name) {
                if seen_blocks.insert(idx) {
                    // Check for cross-attention
                    let has_cross = model.network.layers.iter().any(|l| {
                        l.name.contains(&format!("blocks.{}/cross_attn", idx))
                            || l.name
                                .contains(&format!("decoder.blocks.{}/cross_attn", idx))
                    });
                    blocks.push(DecoderBlockInfo {
                        index: idx,
                        has_cross_attention: has_cross,
                    });
                }
            }
        }
        blocks.sort_by_key(|b| b.index);
    } else {
        // Single-block model (e.g., decoder_block.onnx)
        let has_self_attn = model
            .network
            .layers
            .iter()
            .any(|l| l.name.contains("/self_attn/"));
        let has_cross_attn = model
            .network
            .layers
            .iter()
            .any(|l| l.name.contains("/cross_attn/"));

        if has_self_attn {
            blocks.push(DecoderBlockInfo {
                index: 0,
                has_cross_attention: has_cross_attn,
            });
        }
    }

    // Infer hidden dimension from layer norm or linear layer weights
    for layer in &model.network.layers {
        if layer.layer_type == LayerType::LayerNorm {
            // LayerNorm ny is the hidden dimension
            if let Some(ny_name) = layer.inputs.get(1) {
                if let Some(ny) = model.weights.get(ny_name) {
                    hidden_dim = ny.len();
                    break;
                }
            }
        }
    }

    // Try to infer num_heads from q_proj weight shape
    // q_proj: [hidden_dim, hidden_dim] but we know head_dim is typically 64 or hidden_dim/num_heads
    if hidden_dim > 0 {
        // Common head dimensions: 64 (GPT-2/LLaMA), 80 (GPT-3), 96 (GPT-J)
        // Try to infer from hidden_dim
        if hidden_dim % 64 == 0 {
            num_heads = hidden_dim / 64;
        } else if hidden_dim % 80 == 0 {
            num_heads = hidden_dim / 80;
        } else {
            // Fallback: assume 4 heads for small test models
            num_heads = 4;
        }
    }

    let head_dim = hidden_dim.checked_div(num_heads).unwrap_or(hidden_dim);

    info!(
        "Parsed decoder structure: {} blocks, hidden_dim={}, num_heads={}, head_dim={}",
        blocks.len(),
        hidden_dim,
        num_heads,
        head_dim
    );

    Ok(DecoderStructure {
        blocks,
        num_heads,
        hidden_dim,
        head_dim,
    })
}

/// Parse block index from a layer name.
pub(super) fn parse_decoder_block_index(name: &str) -> Option<usize> {
    // Look for patterns like "blocks.0", "decoder.blocks.1", etc.
    if let Some(pos) = name.find("blocks.") {
        let rest = &name[pos + 7..];
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        return num_str.parse().ok();
    }
    None
}
