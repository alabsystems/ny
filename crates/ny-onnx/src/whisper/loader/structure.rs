// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::helpers::{layer_has_explicit_ln_post_marker, layer_has_ln_post_marker};
use super::super::{WhisperBlockInfo, WhisperEncoderStructure};
use super::block_index::{
    parse_block_index_from_layer, parse_block_index_from_layer_name_or_weight,
};
use super::scope::detect_block_scope;
use crate::Network;
use ny_core::{LayerType, Result};
use std::collections::BTreeMap;
use tracing::{debug, info, warn};

/// Parse the Whisper encoder structure by examining layer names.
pub(crate) fn parse_whisper_structure(network: &Network) -> Result<WhisperEncoderStructure> {
    let mut ln_post_start_idx = network.layers.len();
    let mut ln_post_candidates = Vec::new();
    let block_scope = detect_block_scope(network);
    let mut block_bounds: BTreeMap<usize, (usize, usize)> = BTreeMap::new();

    // Map each block index to its observed layer span.
    for (idx, layer) in network.layers.iter().enumerate() {
        let block_idx = parse_block_index_from_layer(layer, block_scope);
        // Avoid classifying ln_post as a block layer when inputs/outputs carry block names.
        let name_weight_block_idx = parse_block_index_from_layer_name_or_weight(layer, block_scope);
        let has_explicit_ln_post = layer_has_explicit_ln_post_marker(layer, block_scope);
        let has_ln_post_fallback =
            layer_has_ln_post_marker(layer, block_scope) && !has_explicit_ln_post;
        if has_explicit_ln_post || (has_ln_post_fallback && name_weight_block_idx.is_none()) {
            // Final LayerNorm after blocks (explicit ln_post markers, even if block-scoped).
            ln_post_candidates.push(idx);
            continue;
        }

        if let Some(block_idx) = block_idx {
            let entry = block_bounds.entry(block_idx).or_insert((idx, idx + 1));
            entry.0 = entry.0.min(idx);
            entry.1 = entry.1.max(idx + 1);
        }
    }

    let mut blocks: Vec<WhisperBlockInfo> = block_bounds
        .iter()
        .map(|(index, (start, end))| WhisperBlockInfo {
            index: *index,
            start_layer_idx: *start,
            end_layer_idx: *end,
            num_layers: end.saturating_sub(*start),
        })
        .collect();

    if blocks.is_empty() {
        warn!("No Whisper blocks detected from layer names");
    }

    blocks.sort_by_key(|block| block.index);
    let mut stem_end_idx = blocks
        .first()
        .map(|block| block.start_layer_idx)
        .unwrap_or(0);

    // If no blocks found, this might be a non-standard export
    if blocks.is_empty() {
        stem_end_idx = 0;
        ln_post_start_idx = network.layers.len();
    } else if let Some(last_block) = blocks.last() {
        if let Some(candidate) = ln_post_candidates
            .iter()
            .copied()
            .filter(|idx| *idx >= last_block.end_layer_idx)
            .min()
        {
            ln_post_start_idx = candidate;
        } else if let Some(layernorm_idx) = find_layernorm_after(network, last_block.end_layer_idx)
        {
            // Some exports omit ln_post in names; fall back to the first LayerNorm after blocks.
            ln_post_start_idx = layernorm_idx;
        } else if last_block.end_layer_idx < network.layers.len()
            && network.layers[last_block.end_layer_idx].layer_type == LayerType::LayerNorm
        {
            ln_post_start_idx = last_block.end_layer_idx;
        } else if !ln_post_candidates.is_empty() {
            debug!(
                "ln_post candidates before final block: candidates={:?}, last_block_end={}",
                ln_post_candidates, last_block.end_layer_idx
            );
        }
    }
    if blocks.is_empty() && !ln_post_candidates.is_empty() {
        ln_post_start_idx = ln_post_candidates
            .iter()
            .copied()
            .min()
            .unwrap_or(ln_post_start_idx);
    }
    if !blocks.is_empty() && ln_post_start_idx == network.layers.len() {
        let last_layer_idx = network.layers.len().saturating_sub(1);
        if let Some(last_layer) = network.layers.last() {
            if last_layer.layer_type == LayerType::LayerNorm
                && parse_block_index_from_layer(last_layer, block_scope).is_none()
            {
                if let Some(last_block) = blocks.last_mut() {
                    if last_block.end_layer_idx == network.layers.len() {
                        last_block.end_layer_idx = last_layer_idx;
                        last_block.num_layers = last_layer_idx - last_block.start_layer_idx;
                        ln_post_start_idx = last_layer_idx;
                    }
                }
            }
        }
    }
    if !blocks.is_empty() && ln_post_start_idx == network.layers.len() {
        if let Some(last_layer) = network.layers.last() {
            if last_layer.layer_type == LayerType::LayerNorm
                && parse_block_index_from_layer(last_layer, block_scope).is_none()
            {
                ln_post_start_idx = network.layers.len().saturating_sub(1);
            }
        }
    }

    info!(
        "Parsed Whisper structure: {} stem layers, {} blocks, ln_post at {}",
        stem_end_idx,
        blocks.len(),
        ln_post_start_idx
    );

    Ok(WhisperEncoderStructure {
        stem_end_idx,
        blocks,
        ln_post_start_idx,
    })
}

fn find_layernorm_after(network: &Network, start: usize) -> Option<usize> {
    network
        .layers
        .iter()
        .enumerate()
        .skip(start)
        .find(|(_, layer)| layer.layer_type == LayerType::LayerNorm)
        .map(|(idx, _)| idx)
}
