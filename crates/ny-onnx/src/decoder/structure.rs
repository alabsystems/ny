// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decoder structure parsing from ONNX layer naming patterns.

use crate::OnnxModel;
use ny_core::{LayerType, NyError, Result};
use tracing::info;

use super::{DecoderBlockInfo, DecoderModel, DecoderStructure};

/// Load a decoder transformer model.
///
/// Recognizes the narrow naming patterns used by the structural-analysis
/// helpers. Causal self-attention and MLP structural artifacts are available;
/// cross-attention extraction and all decoder verification fail closed.
///
/// # Naming Patterns Supported
/// - Single block: `/norm1/...`, `/self_attn/...`, `/mlp/...`
/// - Multi-block: `/blocks.{i}/norm1/...`
///
/// # Arguments
/// * `path` - Path to ONNX model file
///
/// # Returns
/// DecoderModel with parsed structure for inspection and supported structural artifacts.
pub fn load_decoder<P: AsRef<std::path::Path>>(path: P) -> Result<DecoderModel> {
    let model = crate::load_onnx(path)?;
    let structure = parse_decoder_structure(&model)?;

    let num_blocks = structure.blocks.len();
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
    if model
        .network
        .layers
        .iter()
        .any(|layer| layer.name.contains("decoder.blocks."))
    {
        return Err(NyError::UnsupportedConfiguration(
            "decoder structure prefix 'decoder.blocks.{i}' is not supported by the current \
             subgraph extractors"
                .to_string(),
        ));
    }

    let mut blocks = Vec::new();
    let mut num_heads = 4; // Default, will be inferred if possible

    // Detect if this is a single-block or multi-block model
    let has_block_indices = model
        .network
        .layers
        .iter()
        .any(|l| l.name.contains("blocks."));

    if has_block_indices {
        // Multi-block model: Parse block indices
        let mut seen_blocks = std::collections::HashSet::new();
        for layer in &model.network.layers {
            // A generic transformer/ViT may also use `blocks.N` names. Only a
            // recognized decoder self-attention path establishes a decoder
            // block.
            if layer.name.contains("/self_attn/") {
                let Some(idx) = parse_decoder_block_index(&layer.name) else {
                    continue;
                };
                if seen_blocks.insert(idx) {
                    // Check for cross-attention
                    let has_cross = model
                        .network
                        .layers
                        .iter()
                        .any(|l| l.name.contains(&format!("blocks.{}/cross_attn", idx)));
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

    if blocks.is_empty() {
        return Err(NyError::InvalidSpec(
            "model does not contain a recognized decoder self-attention block".to_string(),
        ));
    }
    for (expected, block) in blocks.iter().enumerate() {
        if block.index != expected {
            return Err(NyError::InvalidSpec(format!(
                "decoder block indices must be contiguous from zero; expected {expected}, got {}",
                block.index
            )));
        }
    }

    // Infer a single hidden dimension from normalization scales belonging to
    // recognized decoder blocks. Do not borrow an unrelated/final model norm,
    // accept a matrix-shaped scale, or silently pick among conflicting blocks.
    let mut inferred_hidden_dim = None;
    for layer in &model.network.layers {
        if !matches!(layer.layer_type, LayerType::LayerNorm | LayerType::RMSNorm) {
            continue;
        }
        let block_level_norm = is_decoder_block_level_norm(&layer.name);
        let belongs_to_block = if has_block_indices {
            parse_decoder_block_index(&layer.name)
                .is_some_and(|index| blocks.iter().any(|block| block.index == index))
                && block_level_norm
        } else {
            block_level_norm
        };
        if !belongs_to_block {
            continue;
        }

        let scale_name = layer.inputs.get(1).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "decoder normalization layer '{}' has no scale input",
                layer.name
            ))
        })?;
        let scale = model.weights.get(scale_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "decoder normalization layer '{}' scale '{}' is not a constant weight",
                layer.name, scale_name
            ))
        })?;
        if scale.ndim() != 1 || scale.is_empty() {
            return Err(NyError::InvalidSpec(format!(
                "decoder normalization layer '{}' scale must be a nonempty vector, got shape {:?}",
                layer.name,
                scale.shape()
            )));
        }
        let candidate = scale.len();
        match inferred_hidden_dim {
            Some(previous) if previous != candidate => {
                return Err(NyError::InvalidSpec(format!(
                    "decoder normalization dimensions disagree: expected {previous}, layer '{}' \
                     has {candidate}",
                    layer.name
                )));
            }
            Some(_) => {}
            None => inferred_hidden_dim = Some(candidate),
        }
    }
    let hidden_dim = inferred_hidden_dim.unwrap_or(0);

    if hidden_dim == 0 {
        return Err(NyError::InvalidSpec(
            "could not determine decoder hidden dimension from LayerNorm/RMSNorm weights"
                .to_string(),
        ));
    }

    // Retained structural hint only. Decoder verification fails closed because
    // head count and attention semantics are not yet proven from graph topology.
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

    if num_heads == 0 || !hidden_dim.is_multiple_of(num_heads) {
        return Err(NyError::InvalidSpec(format!(
            "decoder hidden dimension {hidden_dim} is not divisible by inferred head count \
             {num_heads}"
        )));
    }
    let head_dim = hidden_dim / num_heads;

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

fn is_decoder_block_level_norm(name: &str) -> bool {
    [
        "/norm1/",
        "/norm2/",
        "/norm_cross/",
        "/input_layernorm/",
        "/post_attention_layernorm/",
        "/pre_attention_layernorm/",
        "/pre_feedforward_layernorm/",
        "/post_feedforward_layernorm/",
        "/ln_1/",
        "/ln_2/",
    ]
    .iter()
    .any(|marker| name.contains(marker))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Network, WeightStore};
    use crate::LayerSpec;
    use ndarray::{ArrayD, IxDyn};
    use std::collections::HashMap;

    fn layer(name: &str, layer_type: LayerType, inputs: &[&str]) -> LayerSpec {
        LayerSpec {
            name: name.to_string(),
            layer_type,
            inputs: inputs.iter().map(|input| (*input).to_string()).collect(),
            outputs: vec![format!("{name}:out")],
            weights: None,
            attributes: HashMap::new(),
        }
    }

    fn model(layers: Vec<LayerSpec>, scales: &[(&str, &[usize])]) -> OnnxModel {
        let mut weights = WeightStore::new();
        for (name, shape) in scales {
            weights.insert((*name).to_string(), ArrayD::zeros(IxDyn(shape)));
        }
        OnnxModel::empty_with_network(
            Network {
                name: "decoder-structure-test".to_string(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                layers,
                param_count: 0,
            },
            weights,
        )
    }

    #[test]
    fn rmsnorm_vector_establishes_hidden_dimension() {
        let model = model(
            vec![
                layer("/blocks.0/self_attn/q_proj", LayerType::MatMul, &[]),
                layer(
                    "/blocks.0/norm1/RMSNorm",
                    LayerType::RMSNorm,
                    &["x", "scale"],
                ),
            ],
            &[("scale", &[8])],
        );

        let structure = parse_decoder_structure(&model).expect("valid RMSNorm decoder");
        assert_eq!(structure.hidden_dim, 8);
        assert_eq!(structure.blocks.len(), 1);
    }

    #[test]
    fn malformed_normalization_scale_fails_closed() {
        let model = model(
            vec![
                layer("/blocks.0/self_attn/q_proj", LayerType::MatMul, &[]),
                layer(
                    "/blocks.0/norm1/LayerNorm",
                    LayerType::LayerNorm,
                    &["x", "scale"],
                ),
            ],
            &[("scale", &[2, 4])],
        );

        let error =
            parse_decoder_structure(&model).expect_err("matrix-shaped scale must be rejected");
        assert!(error.to_string().contains("nonempty vector"));
    }

    #[test]
    fn conflicting_block_dimensions_fail_closed() {
        let model = model(
            vec![
                layer("/blocks.0/self_attn/q_proj", LayerType::MatMul, &[]),
                layer(
                    "/blocks.0/norm1/LayerNorm",
                    LayerType::LayerNorm,
                    &["x0", "s0"],
                ),
                layer("/blocks.1/self_attn/q_proj", LayerType::MatMul, &[]),
                layer(
                    "/blocks.1/norm1/LayerNorm",
                    LayerType::LayerNorm,
                    &["x1", "s1"],
                ),
            ],
            &[("s0", &[8]), ("s1", &[12])],
        );

        let error =
            parse_decoder_structure(&model).expect_err("conflicting block dimensions must fail");
        assert!(error.to_string().contains("dimensions disagree"));
    }

    #[test]
    fn per_head_q_norm_does_not_conflict_with_block_hidden_dimension() {
        let model = model(
            vec![
                layer("/blocks.0/self_attn/q_proj", LayerType::MatMul, &[]),
                layer(
                    "/blocks.0/input_layernorm/RMSNorm",
                    LayerType::RMSNorm,
                    &["x", "hidden_scale"],
                ),
                layer(
                    "/blocks.0/self_attn/q_norm/RMSNorm",
                    LayerType::RMSNorm,
                    &["q", "head_scale"],
                ),
            ],
            &[("hidden_scale", &[8]), ("head_scale", &[2])],
        );

        let structure = parse_decoder_structure(&model).expect("q_norm is not a block-level norm");
        assert_eq!(structure.hidden_dim, 8);
    }

    #[test]
    fn generic_indexed_blocks_are_not_decoder_blocks() {
        let model = model(
            vec![
                layer("/blocks.0/attention/q_proj", LayerType::MatMul, &[]),
                layer(
                    "/blocks.0/norm1/LayerNorm",
                    LayerType::LayerNorm,
                    &["x", "scale"],
                ),
            ],
            &[("scale", &[8])],
        );

        let error = parse_decoder_structure(&model)
            .expect_err("generic transformer blocks must not be classified as decoder blocks");
        assert!(error
            .to_string()
            .contains("recognized decoder self-attention block"));
    }
}
