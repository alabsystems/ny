// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decoder transformer model loading and structural analysis.
//!
//! Decoder-only and encoder-decoder block patterns can be loaded for inspection.
//! Causal self-attention and MLP helpers return structural artifacts;
//! cross-attention extraction fails closed until `GraphNetwork` has a sound
//! multi-input contract. All verification compatibility methods fail closed:
//! decoder attention semantics are not yet proven equivalent to the loaded ONNX
//! graph, so they return no bounds or verification details.

use crate::OnnxModel;
use ny_core::{NyError, Result};

mod gpu;
mod structure;
mod subgraph;
mod subgraph_mlp;
mod verify;

pub use structure::load_decoder;

/// Information about a single decoder block.
#[derive(Debug, Clone)]
pub struct DecoderBlockInfo {
    /// Index of the block (0-indexed).
    pub index: usize,
    /// Whether this block has cross-attention (encoder-decoder models).
    pub has_cross_attention: bool,
}

/// Structure describing a decoder transformer layout.
#[derive(Debug, Clone)]
pub struct DecoderStructure {
    /// Information about each decoder block.
    pub blocks: Vec<DecoderBlockInfo>,
    /// Heuristic attention-head hint for structural analysis only.
    ///
    /// This value is not admitted as verified graph semantics.
    pub num_heads: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Heuristic head dimension (`hidden_dim / num_heads`) for analysis only.
    pub head_dim: usize,
}

/// Decoder model with structural-analysis and fail-closed verification APIs.
pub struct DecoderModel {
    /// The underlying ONNX model.
    pub model: OnnxModel,
    /// Parsed decoder structure.
    pub structure: DecoderStructure,
    /// Number of decoder blocks.
    pub num_blocks: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Heuristic attention-head hint for structural analysis only.
    ///
    /// This value is not admitted as verified graph semantics.
    pub num_heads: usize,
}

impl DecoderModel {
    /// Resolve a parsed block by its declared index.
    pub(super) fn block_info(&self, block_index: usize) -> Result<&DecoderBlockInfo> {
        self.structure
            .blocks
            .iter()
            .find(|block| block.index == block_index)
            .ok_or_else(|| NyError::InvalidSpec(format!("decoder block {block_index} not found")))
    }
}

/// Reserved details type for future compositional decoder verification.
#[derive(Debug, Clone)]
pub struct DecoderVerificationDetails {
    /// Max width of attention delta bounds.
    pub attention_delta_width: f32,
    /// Max width after first residual (x + attn_delta).
    pub x_attn_width: f32,
    /// Max width of MLP delta bounds.
    pub mlp_delta_width: f32,
    /// Max width of final output.
    pub output_width: f32,
}
