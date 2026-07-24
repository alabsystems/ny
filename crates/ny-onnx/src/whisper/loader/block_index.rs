// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::helpers::name_is_unscoped;
use super::scope::WhisperBlockScope;
use crate::LayerSpec;

/// Parse block index from a layer name like "/blocks.2/attn/..." or "layers.2.self_attn".
pub(crate) fn parse_block_index(name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    // Look for patterns like "blocks.0", "layers.0", "blocks_0", or "layers_0".
    // ONNX exports may use underscore-separated parameter names (e.g., "p_layers_0_*").
    for token in [
        "blocks.", "layers.", "block.", "layer.", "blocks_", "layers_", "block_", "layer_",
        "blocks/", "layers/", "block/", "layer/",
    ]
    .iter()
    {
        if let Some(pos) = lower.find(token) {
            let rest = &lower[pos + token.len()..];
            // Find the block number (digits before the next separator)
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(index) = num_str.parse() {
                return Some(index);
            }
        }
    }
    None
}

pub(crate) fn parse_block_index_from_layer(
    layer: &LayerSpec,
    scope: WhisperBlockScope,
) -> Option<usize> {
    if let Some(index) = parse_block_index_with_scope(&layer.name, scope) {
        return Some(index);
    }

    if let Some(weights) = &layer.weights {
        // Some ONNX exports omit block names on nodes but keep them on initializers.
        if let Some(index) = parse_block_index_with_scope(&weights.name, scope) {
            return Some(index);
        }
    }

    let mut indices: Vec<usize> = Vec::new();
    for input in &layer.inputs {
        if let Some(index) = parse_block_index_with_scope(input, scope) {
            indices.push(index);
        }
    }
    for output in &layer.outputs {
        if let Some(index) = parse_block_index_with_scope(output, scope) {
            indices.push(index);
        }
    }

    if indices.is_empty() {
        return None;
    }
    let first = indices[0];
    if indices.iter().all(|&idx| idx == first) {
        Some(first)
    } else {
        None
    }
}

pub(crate) fn parse_block_index_with_scope(name: &str, scope: WhisperBlockScope) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    match scope {
        WhisperBlockScope::All => parse_block_index(name),
        WhisperBlockScope::Encoder => {
            if let Some(index) = parse_block_index_for_prefix(
                &lower,
                &[
                    "/encoder/blocks.",
                    "/encoder/layers.",
                    "encoder/blocks.",
                    "encoder/layers.",
                    "encoder.blocks.",
                    "encoder.layers.",
                ],
            ) {
                return Some(index);
            }
            if name_is_unscoped(name) {
                return parse_block_index(name);
            }
            None
        }
        WhisperBlockScope::Decoder => {
            if let Some(index) = parse_block_index_for_prefix(
                &lower,
                &[
                    "/decoder/blocks.",
                    "/decoder/layers.",
                    "decoder/blocks.",
                    "decoder/layers.",
                    "decoder.blocks.",
                    "decoder.layers.",
                ],
            ) {
                return Some(index);
            }
            if name_is_unscoped(name) {
                return parse_block_index(name);
            }
            None
        }
    }
}

fn parse_block_index_for_prefix(name: &str, prefixes: &[&str]) -> Option<usize> {
    for prefix in prefixes {
        if let Some(pos) = name.find(prefix) {
            let rest = &name[pos + prefix.len()..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(index) = num_str.parse() {
                return Some(index);
            }
        }
    }
    None
}

pub(crate) fn parse_block_index_from_layer_name_or_weight(
    layer: &LayerSpec,
    scope: WhisperBlockScope,
) -> Option<usize> {
    if let Some(index) = parse_block_index_with_scope(&layer.name, scope) {
        return Some(index);
    }
    if let Some(weights) = &layer.weights {
        if let Some(index) = parse_block_index_with_scope(&weights.name, scope) {
            return Some(index);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_block_index_case_insensitive() {
        assert_eq!(parse_block_index("Encoder/Blocks.3/attn"), Some(3));
        assert_eq!(
            parse_block_index_with_scope("DECODER/LAYERS.4/self_attn", WhisperBlockScope::Decoder),
            Some(4)
        );
    }
}
