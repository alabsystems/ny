// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::load::is_quantized_type;
use super::metadata::format_metadata_value;
use super::mmap::map_read_only_gguf;
use gguf::{GGUFFile, GGUFMetadataValue};
use ny_core::{NyError, Result};
use std::fs::File;
use std::path::Path;

/// Metadata about a GGUF file.
#[derive(Debug, Clone)]
pub struct GGUFInfo {
    /// GGUF format version.
    pub version: u32,
    /// Number of tensors in the file.
    pub tensor_count: usize,
    /// Total parameter count (sum of all tensor elements).
    pub param_count: usize,
    /// Model architecture (from metadata, if present).
    pub architecture: Option<String>,
    /// Model name (from metadata, if present).
    pub model_name: Option<String>,
    /// Tensor information (name, shape, dtype, quantized).
    pub tensors: Vec<(String, Vec<u64>, String, bool)>,
    /// Key metadata entries.
    pub metadata: Vec<(String, String)>,
}

/// Checked product of u64 dimensions, returning error on overflow.
fn checked_dim_product(dims: &[u64]) -> Result<u64> {
    dims.iter()
        .try_fold(1u64, |a, &d| a.checked_mul(d))
        .ok_or_else(|| NyError::ModelLoad("Tensor dimensions overflow u64".into()))
}

/// Get information about a GGUF file without fully loading tensor data.
///
/// The GGUF file must remain immutable for the duration of the call. Like all
/// file-backed memory maps, concurrent writes from this or another process
/// while inspection is in progress are outside the loader's safety contract.
pub fn gguf_info<P: AsRef<Path>>(path: P) -> Result<GGUFInfo> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    // Use memory-mapped I/O for efficient large file handling.
    let file = File::open(path)
        .map_err(|e| NyError::ModelLoad(format!("Failed to open GGUF file: {}", e)))?;

    let mmap = map_read_only_gguf(&file, path)?;

    let data: &[u8] = &mmap;

    let gguf_file = GGUFFile::read(data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse GGUF: {}", e)))?
        .ok_or_else(|| NyError::ModelLoad("Incomplete GGUF file".to_string()))?;

    // Extract architecture and model name from metadata
    let mut architecture = None;
    let mut model_name = None;
    let mut metadata = Vec::new();

    for meta in &gguf_file.header.metadata {
        let value_str = format_metadata_value(&meta.value);

        // Look for key metadata
        if meta.key == "general.architecture" {
            if let GGUFMetadataValue::String(s) = &meta.value {
                architecture = Some(s.clone());
            }
        }
        if meta.key == "general.name" {
            if let GGUFMetadataValue::String(s) = &meta.value {
                model_name = Some(s.clone());
            }
        }

        // Store interesting metadata
        if meta.key.starts_with("general.")
            || meta.key.contains(".context_length")
            || meta.key.contains(".embedding_length")
            || meta.key.contains(".block_count")
            || meta.key.contains(".attention.head_count")
        {
            metadata.push((meta.key.clone(), value_str));
        }
    }

    // Process tensor info
    let mut tensors = Vec::new();
    let mut param_count = 0;

    for tensor in &gguf_file.tensors {
        let elements = checked_dim_product(&tensor.dimensions)?;
        param_count += elements as usize;

        let is_quantized = is_quantized_type(&tensor.tensor_type);
        tensors.push((
            tensor.name.clone(),
            tensor.dimensions.clone(),
            format!("{:?}", tensor.tensor_type),
            is_quantized,
        ));
    }

    Ok(GGUFInfo {
        version: gguf_file.header.version,
        tensor_count: gguf_file.tensors.len(),
        param_count,
        architecture,
        model_name,
        tensors,
        metadata,
    })
}
