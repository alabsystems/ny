// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GGUF format support for loading llama.cpp model weights.
//!
//! GGUF (GPT-Generated Unified Format) is the format used by llama.cpp
//! for efficient storage and loading of LLM weights. This module provides
//! support for loading weight metadata and data from GGUF files.
//!
//! # Supported Data Types
//!
//! **Unquantized:**
//! - F32: Full precision (directly loaded)
//! - F16: Half precision (converted to f32)
//!
//! **Simple Quants (32 elements per block):**
//! - Q8_0: 8-bit quantized (dequantized to f32)
//! - Q4_0: 4-bit quantized (dequantized to f32)
//! - Q4_1: 4-bit quantized with min (dequantized to f32)
//! - Q5_0: 5-bit quantized (dequantized to f32)
//! - Q5_1: 5-bit quantized with min (dequantized to f32)
//! - Q8_1: 8-bit quantized with sum (dequantized to f32)
//!
//! **K-Quants (256 elements per super-block):**
//! - Q2_K: 2-bit quantized with per-group scales
//! - Q3_K: 3-bit quantized with per-group scales
//! - Q4_K: 4-bit quantized with per-group scales
//! - Q5_K: 5-bit quantized with per-group scales
//! - Q6_K: 6-bit quantized with per-group scales
//!
//! # Usage
//!
//! ```rust,no_run
//! use ny_onnx::gguf::{load_gguf, gguf_info};
//!
//! // Get info about a GGUF file
//! let info = gguf_info("model.gguf").unwrap();
//! println!("Model has {} tensors", info.tensor_count);
//!
//! // Load weights (including dequantized quantized tensors)
//! let weights = load_gguf("model.gguf").unwrap();
//! ```

mod dequant;
mod file_data;
mod info;
mod load;
mod metadata;
mod parser;

/// GGUF metadata inspection helper and parsed metadata summary type.
pub use info::{gguf_info, GGUFInfo};
/// GGUF weight loader with built-in dequantization for supported tensor formats.
pub use load::load_gguf;

#[cfg(test)]
mod tests;
