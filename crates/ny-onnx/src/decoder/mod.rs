// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decoder transformer model loading and compositional verification.
//!
//! Supports both decoder-only models (like LLaMA) and encoder-decoder decoder blocks
//! (like Whisper decoder). Compositional verification decomposes each block into
//! independently verifiable subgraphs (attention + MLP) connected via residual streams.

use crate::OnnxModel;

mod gpu;
mod structure;
mod subgraph;
mod subgraph_mlp;
mod verify;
mod weights;

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
    /// Number of attention heads.
    pub num_heads: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Head dimension (hidden_dim / num_heads).
    pub head_dim: usize,
}

/// Decoder model with compositional verification support.
pub struct DecoderModel {
    /// The underlying ONNX model.
    pub model: OnnxModel,
    /// Parsed decoder structure.
    pub structure: DecoderStructure,
    /// Number of decoder blocks.
    pub num_blocks: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
}

/// Details from compositional decoder block verification.
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
