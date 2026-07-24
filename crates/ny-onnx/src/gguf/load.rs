// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::dequant::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    dequantize_q8_1, get_block_elements, get_block_size,
};
use super::mmap::map_read_only_gguf;
use super::parser::compute_data_section_offset;
use crate::safetensors::half_to_f32;
use crate::WeightStore;
use gguf::{GGMLType, GGUFFile, GGUFTensorInfo};
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::fs::File;
use std::path::Path;
use tracing::{debug, info, warn};

/// Load weights from a GGUF file.
///
/// Loads F32/F16 tensors directly and dequantizes supported quantized tensors into f32.
///
/// # Arguments
///
/// * `path` - Path to the .gguf file
///
/// # Returns
///
/// A `WeightStore` with all loadable tensors, with supported quantized tensors
/// dequantized to `f32`.
///
/// The GGUF file must remain immutable for the duration of the call. Like all
/// file-backed memory maps, concurrent writes from this or another process
/// while loading is in progress are outside the loader's safety contract.
pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    info!("Loading GGUF from: {}", path.display());

    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    // Use memory-mapped I/O for efficient large file handling.
    // This allows the OS to page in tensor data on-demand rather than
    // loading the entire file (which can be 30GB+) into memory upfront.
    let file = File::open(path)
        .map_err(|e| NyError::ModelLoad(format!("Failed to open GGUF file: {}", e)))?;

    let mmap = map_read_only_gguf(&file, path)?;

    let data: &[u8] = &mmap;

    let gguf_file = GGUFFile::read(data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse GGUF: {}", e)))?
        .ok_or_else(|| NyError::ModelLoad("Incomplete GGUF file".to_string()))?;

    let data_section_offset = compute_data_section_offset(data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to compute GGUF data section: {}", e)))?;

    let mut weights = WeightStore::new();
    let mut loaded_count = 0;
    let mut dequant_count = 0;
    let mut skipped_unsupported_count = 0;

    for tensor in &gguf_file.tensors {
        let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();
        let elements: usize = checked_shape_product(&shape).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "Tensor '{}' shape {:?} overflows usize on product",
                tensor.name, shape
            ))
        })?;

        match load_tensor_data(data, data_section_offset, tensor, elements) {
            Ok(arr) => {
                if is_quantized_type(&tensor.tensor_type) {
                    debug!(
                        "Dequantized tensor: {} ({:?}) shape {:?}",
                        tensor.name, tensor.tensor_type, shape
                    );
                    dequant_count += 1;
                } else {
                    debug!("Loaded tensor: {} shape {:?}", tensor.name, shape);
                }
                weights.insert(tensor.name.clone(), arr);
                loaded_count += 1;
            }
            Err(e) => {
                if is_quantized_type(&tensor.tensor_type) && !is_dequantizable(&tensor.tensor_type)
                {
                    warn!(
                        "Skipping unsupported quantized tensor '{}' ({:?}): {}",
                        tensor.name, tensor.tensor_type, e
                    );
                    skipped_unsupported_count += 1;
                    continue;
                }

                return Err(NyError::ModelLoad(format!(
                    "Failed to load GGUF tensor '{}' ({:?}): {}",
                    tensor.name, tensor.tensor_type, e
                )));
            }
        }
    }

    if dequant_count > 0 {
        info!(
            "Loaded {} tensors from GGUF ({} dequantized, {} unsupported skipped)",
            loaded_count, dequant_count, skipped_unsupported_count
        );
    } else {
        info!(
            "Loaded {} tensors from GGUF ({} unsupported skipped)",
            loaded_count, skipped_unsupported_count
        );
    }

    weights.validate_no_nan()?;
    Ok(weights)
}

/// Load tensor data from the GGUF file.
pub(super) fn load_tensor_data(
    file_data: &[u8],
    data_section_offset: usize,
    tensor: &GGUFTensorInfo,
    elements: usize,
) -> std::result::Result<ArrayD<f32>, String> {
    let offset = data_section_offset
        .checked_add(tensor.offset as usize)
        .ok_or("Tensor data offset overflow")?;
    // Fail closed if the start offset is already past EOF. The per-type byte_size
    // checks below cover offset+size, but a zero-byte tensor (some dimension == 0)
    // skips that check, so `&file_data[offset..]` could otherwise panic when a
    // crafted `general.alignment`/tensor offset pushes `offset` beyond the file.
    if offset > file_data.len() {
        return Err(format!(
            "Tensor data offset {} beyond file end {}",
            offset,
            file_data.len()
        ));
    }
    let shape: Vec<usize> = tensor.dimensions.iter().map(|&d| d as usize).collect();

    match tensor.tensor_type {
        GGMLType::F32 => {
            let byte_size = elements.checked_mul(4).ok_or("F32 byte size overflow")?;
            if file_data.len().saturating_sub(offset) < byte_size {
                return Err(format!(
                    "Tensor data out of bounds (offset={}, size={}, file_len={})",
                    offset,
                    byte_size,
                    file_data.len()
                ));
            }

            let data = &file_data[offset..][..byte_size];
            let floats: Vec<f32> = data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect();

            ArrayD::from_shape_vec(IxDyn(&shape), floats)
                .map_err(|e| format!("Shape mismatch: {}", e))
        }
        GGMLType::F16 => {
            let byte_size = elements.checked_mul(2).ok_or("F16 byte size overflow")?;
            if file_data.len().saturating_sub(offset) < byte_size {
                return Err(format!(
                    "Tensor data out of bounds (offset={}, size={}, file_len={})",
                    offset,
                    byte_size,
                    file_data.len()
                ));
            }

            let data = &file_data[offset..][..byte_size];
            let floats: Vec<f32> = data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| {
                    let bits = u16::from_le_bytes(*chunk);
                    half_to_f32(bits)
                })
                .collect();

            ArrayD::from_shape_vec(IxDyn(&shape), floats)
                .map_err(|e| format!("Shape mismatch: {}", e))
        }
        // Quantized types - dequantize to f32
        GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_1
        | GGMLType::Q5_0
        | GGMLType::Q5_1
        | GGMLType::Q8_1
        // K-quants (256 elements per super-block)
        | GGMLType::Q2K
        | GGMLType::Q3K
        | GGMLType::Q4K
        | GGMLType::Q5K
        | GGMLType::Q6K => {
            let block_size = get_block_size(&tensor.tensor_type)
                .ok_or_else(|| format!("Unknown block size for {:?}", tensor.tensor_type))?;
            let block_elements = get_block_elements(&tensor.tensor_type)
                .ok_or_else(|| format!("Unknown block elements for {:?}", tensor.tensor_type))?;

            if !elements.is_multiple_of(block_elements) {
                return Err(format!(
                    "Element count {} not divisible by block size {} for {:?}",
                    elements, block_elements, tensor.tensor_type
                ));
            }

            let num_blocks = elements / block_elements;
            let byte_size = num_blocks.checked_mul(block_size).ok_or("Quantized byte size overflow")?;

            if file_data.len().saturating_sub(offset) < byte_size {
                return Err(format!(
                    "Tensor data out of bounds (offset={}, size={}, file_len={})",
                    offset,
                    byte_size,
                    file_data.len()
                ));
            }

            let data = &file_data[offset..][..byte_size];
            let floats = match tensor.tensor_type {
                GGMLType::Q8_0 => dequantize_q8_0(data, elements)?,
                GGMLType::Q4_0 => dequantize_q4_0(data, elements)?,
                GGMLType::Q4_1 => dequantize_q4_1(data, elements)?,
                GGMLType::Q5_0 => dequantize_q5_0(data, elements)?,
                GGMLType::Q5_1 => dequantize_q5_1(data, elements)?,
                GGMLType::Q8_1 => dequantize_q8_1(data, elements)?,
                // K-quants
                GGMLType::Q2K => dequantize_q2_k(data, elements)?,
                GGMLType::Q3K => dequantize_q3_k(data, elements)?,
                GGMLType::Q4K => dequantize_q4_k(data, elements)?,
                GGMLType::Q5K => dequantize_q5_k(data, elements)?,
                GGMLType::Q6K => dequantize_q6_k(data, elements)?,
                _ => {
                    return Err(format!(
                        "Quantized type {:?} matched outer arm but not inner dequantize dispatch",
                        tensor.tensor_type
                    ))
                }
            };

            ArrayD::from_shape_vec(IxDyn(&shape), floats)
                .map_err(|e| format!("Shape mismatch: {}", e))
        }
        // Other quantization types not yet supported
        _ => Err(format!(
            "Quantized type {:?} not yet supported",
            tensor.tensor_type
        )),
    }
}

/// Check if a GGML type is quantized.
pub(super) fn is_quantized_type(dtype: &GGMLType) -> bool {
    !matches!(dtype, GGMLType::F32 | GGMLType::F16)
}

/// Check if a GGML type can be dequantized by this library.
pub(super) fn is_dequantizable(dtype: &GGMLType) -> bool {
    matches!(
        dtype,
        GGMLType::F32
            | GGMLType::F16
            | GGMLType::Q8_0
            | GGMLType::Q4_0
            | GGMLType::Q4_1
            | GGMLType::Q5_0
            | GGMLType::Q5_1
            | GGMLType::Q8_1
            // K-quants
            | GGMLType::Q2K
            | GGMLType::Q3K
            | GGMLType::Q4K
            | GGMLType::Q5K
            | GGMLType::Q6K
    )
}
