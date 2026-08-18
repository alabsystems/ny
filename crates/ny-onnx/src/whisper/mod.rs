// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Whisper-specific ONNX loading, structure parsing, and verification helpers.
//!
//! This module was split to keep per-component logic isolated while preserving
//! the public API surface re-exported in `ny_onnx::whisper`.

mod export;
mod graph;
mod helpers;
mod loader;
mod model;
mod subgraph;
#[cfg(test)]
mod tests;
mod types;

/// Generates helper scripts for exporting Whisper models to ONNX.
pub use export::generate_whisper_export_script;
/// Loads a Whisper ONNX model with block-level structure extraction.
pub use loader::load_whisper;
/// Whisper model container with parsed architecture metadata.
pub use model::WhisperModel;
/// Whisper block-compatibility and encoder-structure metadata types.
pub use types::{
    CompositionalVerificationDetails, GpuCompositionalDetails, MultiBlockConfig, MultiBlockDetails,
    WhisperBlockInfo, WhisperEncoderStructure,
};

pub(crate) use loader::scope::WhisperBlockScope;

#[cfg(test)]
pub(crate) use crate::{AttributeValue, LayerSpec, OnnxModel};
#[cfg(test)]
pub(crate) use loader::block_index::parse_block_index_with_scope;
#[cfg(test)]
pub(crate) use loader::structure::parse_whisper_structure;
#[cfg(test)]
pub(crate) use ny_core::LayerType;
#[cfg(test)]
pub(crate) use ny_propagate::Layer;
