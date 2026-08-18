// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::file_data::{capture_file_stamp, ensure_file_unchanged};
use super::load::is_quantized_type;
use super::parser::{read_streamed_gguf_descriptor, StreamedGgufDescriptor};
use ny_core::{NyError, Result};
use std::{fs::File, mem::size_of, path::Path};

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

/// Checked conversion and product of GGUF's u64 dimensions.
fn checked_dim_product(dims: &[u64]) -> Result<usize> {
    dims.iter().try_fold(1usize, |product, &dimension| {
        let dimension = usize::try_from(dimension).map_err(|_| {
            NyError::ModelLoad(format!(
                "GGUF tensor dimension {dimension} does not fit usize"
            ))
        })?;
        product
            .checked_mul(dimension)
            .ok_or_else(|| NyError::ModelLoad("Tensor dimensions overflow usize".into()))
    })
}

fn info_from_descriptor(descriptor: StreamedGgufDescriptor) -> Result<GGUFInfo> {
    let tensor_count = descriptor.tensors.len();
    let required_bytes = tensor_count
        .checked_mul(size_of::<(String, Vec<u64>, String, bool)>())
        .ok_or_else(|| NyError::ModelLoad("GGUF info tensor allocation overflows usize".into()))?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(tensor_count)
        .map_err(|_| NyError::CpuMemoryExceeded {
            required_bytes,
            budget_bytes: usize::MAX,
            site: "ny-onnx::gguf::gguf_info/tensors",
        })?;

    let mut param_count = 0usize;
    for tensor in descriptor.tensors {
        let elements = checked_dim_product(&tensor.dimensions)?;
        param_count = param_count.checked_add(elements).ok_or_else(|| {
            NyError::ModelLoad("Total GGUF parameter count overflows usize".into())
        })?;
        tensors.push((
            tensor.name,
            tensor.dimensions,
            format!("{:?}", tensor.tensor_type),
            is_quantized_type(&tensor.tensor_type),
        ));
    }

    Ok(GGUFInfo {
        version: descriptor.version,
        tensor_count,
        param_count,
        architecture: descriptor.architecture,
        model_name: descriptor.model_name,
        tensors,
        metadata: descriptor.metadata,
    })
}

/// Get information about a GGUF file without reading its tensor payloads.
///
/// Only the metadata/tensor-descriptor prefix is streamed, subject to a fixed
/// safety bound. The file must remain immutable for the duration of the call.
/// Normal identity, size, or modification-time changes are detected and
/// rejected as a potentially mixed read.
pub fn gguf_info<P: AsRef<Path>>(path: P) -> Result<GGUFInfo> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let mut file = File::open(path)
        .map_err(|e| NyError::ModelLoad(format!("Failed to open GGUF file: {}", e)))?;
    let stamp = capture_file_stamp(&file, path)?;
    let descriptor = read_streamed_gguf_descriptor(&mut file, path, stamp.len())?;
    ensure_file_unchanged(&file, path, &stamp, "reading metadata")?;

    info_from_descriptor(descriptor)
}
