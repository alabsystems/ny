// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Network graph construction from weights and architecture config.
//!
//! Each architecture has its own submodule:
//! - `whisper` - Whisper speech encoder
//! - `transformer_enc` - Generic transformer encoder
//! - `transformer_dec` - LLaMA-style decoder (GGUF)
//! - `kokoro` - Kokoro TTS
//! - `mlp` - Simple feedforward MLP
//! - `generic` - Fallback for unknown architectures

mod generic;
mod gguf_attention;
mod kokoro;
mod mlp;
mod transformer_dec;
mod transformer_enc;
mod whisper;

use super::config::{Architecture, ModelConfig};
use crate::{Network, WeightStore};
use ny_core::{NyError, Result};

/// Build network graph from weights and config.
pub(super) fn build_network(weights: &WeightStore, config: &ModelConfig) -> Result<Network> {
    match config.architecture {
        Architecture::WhisperEncoder => whisper::build_whisper_encoder(weights, config),
        Architecture::Kokoro => kokoro::build_kokoro_network(weights, config),
        Architecture::TransformerEncoder => {
            transformer_enc::build_transformer_encoder(weights, config)
        }
        Architecture::TransformerDecoder => {
            transformer_dec::build_transformer_decoder(weights, config)
        }
        Architecture::MLP => mlp::build_mlp_network(weights, config),
        Architecture::Unknown => generic::build_generic_network(weights, config),
        _ => Err(NyError::ModelLoad(format!(
            "Architecture {:?} not yet implemented",
            config.architecture
        ))),
    }
}
