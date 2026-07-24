// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::config::{Architecture, ModelConfig};
use crate::WeightStore;
use ny_core::Result;
use tracing::{debug, info, warn};

/// Detect architecture from weight names.
pub(super) fn detect_architecture(weights: &WeightStore) -> Result<ModelConfig> {
    let names: Vec<&str> = weights.keys().collect();
    debug!("Detecting architecture from {} weights", names.len());

    // Check for Kokoro patterns first (more specific than Whisper)
    if has_kokoro_patterns(&names) {
        return Ok(detect_kokoro_config(weights));
    }

    // Check for Whisper patterns
    if has_whisper_encoder_patterns(&names) {
        return Ok(detect_whisper_config(weights));
    }

    // Check for CosyVoice patterns
    if has_cosyvoice_patterns(&names) {
        return Ok(detect_cosyvoice_config(weights));
    }

    // Check for GGUF LLM patterns (llama.cpp naming: blk.N.attn_q, etc.)
    // This must come before generic transformer patterns
    if has_gguf_llm_patterns(&names) {
        info!("Detected GGUF LLM architecture (decoder transformer)");
        return Ok(detect_gguf_llm_config(weights));
    }

    // Check for generic transformer patterns
    if has_transformer_patterns(&names) {
        return Ok(detect_transformer_config(weights));
    }

    // Check for MLP patterns
    if has_mlp_patterns(&names) {
        return Ok(detect_mlp_config(weights));
    }

    // Fallback to unknown
    warn!("Could not detect architecture, using generic handling");
    Ok(ModelConfig::new(Architecture::Unknown))
}

fn has_whisper_encoder_patterns(names: &[&str]) -> bool {
    names.iter().any(|n| {
        n.contains("encoder.conv1")
            || n.contains("encoder.blocks")
            || n.contains("model.encoder.conv1")
            || (n.contains("conv1.weight") && names.iter().any(|m| m.contains("blocks")))
    })
}

fn has_kokoro_patterns(names: &[&str]) -> bool {
    names
        .iter()
        .any(|n| n.contains("bert_encoder") || n.contains("predictor.lstm"))
        && names.iter().any(|n| n.contains("decoder"))
}

fn has_cosyvoice_patterns(names: &[&str]) -> bool {
    names
        .iter()
        .any(|n| n.contains("flow") || n.contains("hift"))
        && names
            .iter()
            .any(|n| n.contains("mel") || n.contains("speech"))
}

fn has_transformer_patterns(names: &[&str]) -> bool {
    names
        .iter()
        .any(|n| n.contains("attention") || n.contains("self_attn") || n.contains("mha"))
        && names
            .iter()
            .any(|n| n.contains("ffn") || n.contains("mlp") || n.contains("fc"))
}

fn has_mlp_patterns(names: &[&str]) -> bool {
    // Look for sequential layer patterns like layer1, layer2, fc1, fc2
    let has_fc = names
        .iter()
        .any(|n| n.contains("fc") || n.contains("linear"));
    let has_numbered = names
        .iter()
        .any(|n| n.contains(".0.") || n.contains(".1.") || n.contains("layer"));
    has_fc || has_numbered
}

/// Detect GGUF LLM patterns (llama.cpp naming convention).
///
/// GGUF LLMs use patterns like:
/// - `blk.N.attn_q.weight` - Q projection
/// - `blk.N.attn_k.weight` - K projection
/// - `blk.N.attn_v.weight` - V projection
/// - `blk.N.attn_output.weight` - Output projection
/// - `blk.N.ffn_up.weight` - FFN up projection
/// - `blk.N.ffn_down.weight` - FFN down projection
/// - `token_embd.weight` - Token embedding
/// - `output.weight` - LM head
fn has_gguf_llm_patterns(names: &[&str]) -> bool {
    // Check for GGUF LLM specific patterns
    let has_blk_attn = names
        .iter()
        .any(|n| n.starts_with("blk.") && n.contains(".attn_q."));
    let has_ffn = names
        .iter()
        .any(|n| n.starts_with("blk.") && (n.contains(".ffn_up.") || n.contains(".ffn_down.")));
    let has_token_embd = names.contains(&"token_embd.weight");

    has_blk_attn && has_ffn && has_token_embd
}

fn detect_whisper_config(weights: &WeightStore) -> ModelConfig {
    // Try to detect size from conv1 weight shape
    let mut config = ModelConfig::whisper_base();

    // Find conv1 weight to determine hidden dim
    for (name, weight) in weights.iter() {
        if name.contains("conv1.weight") && weight.ndim() == 3 {
            let out_channels = weight.shape()[0];
            config.hidden_dim = out_channels;

            // Determine model size from hidden dim
            config = match out_channels {
                384 => ModelConfig::whisper_tiny(),
                512 => ModelConfig::whisper_base(),
                768 => ModelConfig::whisper_small(),
                1024 => ModelConfig::whisper_medium(),
                1280 | 1536 => ModelConfig::whisper_large(),
                _ => {
                    let mut c = ModelConfig::new(Architecture::WhisperEncoder);
                    c.hidden_dim = out_channels;
                    c
                }
            };
            break;
        }
    }

    // Count number of encoder blocks (try both "blocks" and "layers" patterns)
    let mut max_block = 0;
    for name in weights.keys() {
        // Try "encoder.layers.X" pattern (Whisper HuggingFace format)
        if let Some(idx) = extract_block_number(name, "encoder.layers") {
            max_block = max_block.max(idx + 1);
        }
        // Try "blocks.X" pattern (other formats)
        else if let Some(idx) = extract_block_number(name, "blocks") {
            max_block = max_block.max(idx + 1);
        }
    }
    if max_block > 0 {
        config.num_layers = Some(max_block);
    }

    config
}

fn detect_kokoro_config(weights: &WeightStore) -> ModelConfig {
    let mut config = ModelConfig::kokoro();

    // Try to detect hidden dimension from bert_encoder
    for (name, weight) in weights.iter() {
        if name.contains("bert_encoder") && name.contains("weight") && weight.ndim() == 2 {
            config.hidden_dim = weight.shape()[0];
            break;
        }
    }

    config
}

fn detect_cosyvoice_config(weights: &WeightStore) -> ModelConfig {
    let mut config = ModelConfig::new(Architecture::CosyVoice);

    // Try to detect dimensions from flow model
    for (name, weight) in weights.iter() {
        if name.contains("flow") && name.contains("weight") && weight.ndim() == 2 {
            config.hidden_dim = weight.shape()[0];
            break;
        }
    }

    config
}

fn detect_transformer_config(weights: &WeightStore) -> ModelConfig {
    let mut config = ModelConfig::new(Architecture::TransformerEncoder);

    // Try to detect hidden dimension from attention weights
    for (name, weight) in weights.iter() {
        if (name.contains("attention") || name.contains("self_attn"))
            && name.contains("weight")
            && weight.ndim() == 2
        {
            // Usually q_proj or similar has shape [hidden, hidden]
            config.hidden_dim = weight.shape()[0];
            break;
        }
    }

    // Count transformer layers
    let mut max_layer = 0;
    for name in weights.keys() {
        if let Some(layer_num) = extract_block_number(name, "layer") {
            max_layer = max_layer.max(layer_num + 1);
        }
        if let Some(layer_num) = extract_block_number(name, "blocks") {
            max_layer = max_layer.max(layer_num + 1);
        }
    }
    if max_layer > 0 {
        config.num_layers = Some(max_layer);
    }

    config
}

fn detect_mlp_config(weights: &WeightStore) -> ModelConfig {
    let mut config = ModelConfig::new(Architecture::MLP);

    // Try to detect dimensions from first layer
    for (name, weight) in weights.iter() {
        if (name.contains("0") || name.contains("fc1") || name.contains("linear1"))
            && name.contains("weight")
            && weight.ndim() == 2
        {
            config.input_dim = Some(weight.shape()[1]);
            config.hidden_dim = weight.shape()[0];
            break;
        }
    }

    config
}

/// Detect GGUF LLM config from weights (llama.cpp naming convention).
///
/// Extracts hidden dimension, number of layers, and head count from weight shapes.
fn detect_gguf_llm_config(weights: &WeightStore) -> ModelConfig {
    let mut config = ModelConfig::new(Architecture::TransformerDecoder);

    // Find hidden dimension from token_embd.weight [vocab_size, hidden_dim]
    // or blk.0.attn_q.weight [hidden_dim, hidden_dim] (for standard attention)
    // Note: GGUF stores shapes as [out_dim, in_dim] for linear weights
    if let Some(embd) = weights.get("token_embd.weight") {
        // token_embd.weight shape is [hidden_dim, vocab_size] in GGUF
        config.hidden_dim = embd.shape()[0];
        config.input_dim = Some(embd.shape()[0]); // Input is embedded tokens
        config.output_dim = Some(embd.shape()[1]); // Output vocab size
    }

    // Count number of layers (max blk.N + 1)
    let mut max_layer = 0;
    for name in weights.keys() {
        if name.starts_with("blk.") {
            if let Some(layer_num) = extract_gguf_layer_number(name) {
                max_layer = max_layer.max(layer_num + 1);
            }
        }
    }
    if max_layer > 0 {
        config.num_layers = Some(max_layer);
    }

    // Try to infer head count from attn_q weight shape
    // GGUF stores weights as [in_dim, out_dim], so:
    // - shape[0] = hidden_dim (input to Q projection)
    // - shape[1] = q_dim = num_heads * head_dim (Q output dimension)
    // For GQA: q_dim may be larger than hidden_dim
    if let Some(q_weight) = weights.get("blk.0.attn_q.weight") {
        let q_out_dim = q_weight.shape()[1]; // Query output dimension (GGUF: [in, out])

        // Common head dimensions: 64, 128, 256
        // Try to find a reasonable head count based on Q dimension
        for head_dim in [128, 64, 256, 96, 80] {
            if q_out_dim % head_dim == 0 {
                config.num_heads = Some(q_out_dim / head_dim);
                debug!(
                    "Detected {} Q heads from q_dim={} / head_dim={}",
                    q_out_dim / head_dim,
                    q_out_dim,
                    head_dim
                );
                break;
            }
        }
    }

    debug!(
        "Detected GGUF LLM config: hidden_dim={}, num_layers={:?}, num_heads={:?}",
        config.hidden_dim, config.num_layers, config.num_heads
    );

    config
}

/// Extract layer number from GGUF weight name (e.g., "blk.5.attn_q" -> 5).
pub(crate) fn extract_gguf_layer_number(name: &str) -> Option<usize> {
    name.strip_prefix("blk.").and_then(|rest| {
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        num_str.parse().ok()
    })
}

/// Extract block number from weight name (e.g., "blocks.5.attn" -> 5).
pub(crate) fn extract_block_number(name: &str, prefix: &str) -> Option<usize> {
    let pattern = format!("{}.", prefix);
    if let Some(idx) = name.find(&pattern) {
        let rest = &name[idx + pattern.len()..];
        let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        num_str.parse().ok()
    } else {
        None
    }
}
