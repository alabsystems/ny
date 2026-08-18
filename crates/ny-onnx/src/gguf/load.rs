// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::dequant::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
    dequantize_q8_1, get_block_elements, get_block_size,
};
use super::file_data::{capture_file_stamp, ensure_file_unchanged};
use super::parser::read_streamed_gguf_descriptor;
use crate::safetensors::half_to_f32;
use crate::WeightStore;
use gguf::{GGMLType, GGUFTensorInfo};
use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    mem::size_of,
    path::Path,
};
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
/// Metadata is streamed first, then each tensor payload is read into a
/// short-lived buffer and decoded. The loader therefore does not retain a
/// second full-file copy alongside the decoded weights; peak overhead is the
/// largest encoded tensor plus its decoded output. The file must remain
/// immutable for the duration of the call. Normal identity, size, or
/// modification-time changes are detected and rejected.
pub fn load_gguf<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    info!("Loading GGUF from: {}", path.display());

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

    let mut weights = WeightStore::new();
    let mut loaded_count = 0;
    let mut dequant_count = 0;
    let mut skipped_unsupported_count = 0;

    for tensor in &descriptor.tensors {
        if is_quantized_type(&tensor.tensor_type) && !is_dequantizable(&tensor.tensor_type) {
            warn!(
                "Skipping unsupported quantized tensor '{}' ({:?})",
                tensor.name, tensor.tensor_type
            );
            skipped_unsupported_count += 1;
            continue;
        }

        let (shape, elements) = checked_tensor_shape(tensor).map_err(|e| {
            NyError::ModelLoad(format!(
                "Invalid GGUF tensor '{}' shape {:?}: {e}",
                tensor.name, tensor.dimensions
            ))
        })?;

        match load_tensor_from_file(
            &mut file,
            path,
            stamp.len(),
            descriptor.data_section_offset,
            tensor,
            elements,
            &shape,
        ) {
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
                return Err(NyError::ModelLoad(format!(
                    "Failed to load GGUF tensor '{}' ({:?}): {}",
                    tensor.name, tensor.tensor_type, e
                )));
            }
        }
    }

    ensure_file_unchanged(&file, path, &stamp, "loading tensor data")?;

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

fn checked_tensor_shape(
    tensor: &GGUFTensorInfo,
) -> std::result::Result<(Vec<usize>, usize), String> {
    let shape_bytes = tensor
        .dimensions
        .len()
        .checked_mul(size_of::<usize>())
        .ok_or("Tensor shape allocation overflow")?;
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(tensor.dimensions.len())
        .map_err(|_| {
            format!(
                "Unable to allocate {shape_bytes} bytes for tensor shape {:?}",
                tensor.dimensions
            )
        })?;

    let mut elements = 1usize;
    for &dimension in &tensor.dimensions {
        let dimension = usize::try_from(dimension)
            .map_err(|_| format!("Tensor dimension {dimension} does not fit usize"))?;
        elements = elements
            .checked_mul(dimension)
            .ok_or("Tensor shape product overflows usize")?;
        shape.push(dimension);
    }
    Ok((shape, elements))
}

fn tensor_payload_size(
    tensor_type: &GGMLType,
    elements: usize,
) -> std::result::Result<usize, String> {
    match tensor_type {
        GGMLType::F32 => elements
            .checked_mul(4)
            .ok_or("F32 byte size overflow".into()),
        GGMLType::F16 => elements
            .checked_mul(2)
            .ok_or("F16 byte size overflow".into()),
        dtype if is_dequantizable(dtype) => {
            let block_size =
                get_block_size(dtype).ok_or_else(|| format!("Unknown block size for {dtype:?}"))?;
            let block_elements = get_block_elements(dtype)
                .ok_or_else(|| format!("Unknown block elements for {dtype:?}"))?;
            if !elements.is_multiple_of(block_elements) {
                return Err(format!(
                    "Element count {elements} not divisible by block size {block_elements} for \
                     {dtype:?}"
                ));
            }
            (elements / block_elements)
                .checked_mul(block_size)
                .ok_or_else(|| "Quantized byte size overflow".into())
        }
        _ => Err(format!("Quantized type {tensor_type:?} not yet supported")),
    }
}

fn decode_tensor_payload(
    data: &[u8],
    tensor: &GGUFTensorInfo,
    elements: usize,
    shape: &[usize],
) -> std::result::Result<ArrayD<f32>, String> {
    let expected_size = tensor_payload_size(&tensor.tensor_type, elements)?;
    if data.len() != expected_size {
        return Err(format!(
            "Tensor payload size mismatch (got={}, expected={expected_size})",
            data.len()
        ));
    }

    let floats = match tensor.tensor_type {
        GGMLType::F32 => data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        GGMLType::F16 => data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| half_to_f32(u16::from_le_bytes(*chunk)))
            .collect(),
        GGMLType::Q8_0 => dequantize_q8_0(data, elements)?,
        GGMLType::Q4_0 => dequantize_q4_0(data, elements)?,
        GGMLType::Q4_1 => dequantize_q4_1(data, elements)?,
        GGMLType::Q5_0 => dequantize_q5_0(data, elements)?,
        GGMLType::Q5_1 => dequantize_q5_1(data, elements)?,
        GGMLType::Q8_1 => dequantize_q8_1(data, elements)?,
        GGMLType::Q2K => dequantize_q2_k(data, elements)?,
        GGMLType::Q3K => dequantize_q3_k(data, elements)?,
        GGMLType::Q4K => dequantize_q4_k(data, elements)?,
        GGMLType::Q5K => dequantize_q5_k(data, elements)?,
        GGMLType::Q6K => dequantize_q6_k(data, elements)?,
        _ => {
            return Err(format!(
                "Quantized type {:?} not yet supported",
                tensor.tensor_type
            ))
        }
    };

    ArrayD::from_shape_vec(IxDyn(shape), floats).map_err(|e| format!("Shape mismatch: {e}"))
}

fn load_tensor_from_file(
    file: &mut File,
    path: &Path,
    file_len: u64,
    data_section_offset: u64,
    tensor: &GGUFTensorInfo,
    elements: usize,
    shape: &[usize],
) -> std::result::Result<ArrayD<f32>, String> {
    let payload_size = tensor_payload_size(&tensor.tensor_type, elements)?;
    let payload_size_u64 = u64::try_from(payload_size)
        .map_err(|_| format!("Tensor payload size {payload_size} does not fit u64"))?;
    let offset = data_section_offset
        .checked_add(tensor.offset)
        .ok_or("Tensor data offset overflows u64")?;
    let end = offset
        .checked_add(payload_size_u64)
        .ok_or("Tensor data end offset overflows u64")?;
    if offset > file_len || end > file_len {
        return Err(format!(
            "Tensor data out of bounds (offset={offset}, size={payload_size}, file_len={file_len})"
        ));
    }

    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("Failed to seek tensor payload in '{}': {e}", path.display()))?;
    let mut payload = Vec::new();
    payload.try_reserve_exact(payload_size).map_err(|_| {
        format!(
            "Unable to allocate {payload_size} bytes for encoded tensor '{}'",
            tensor.name
        )
    })?;
    payload.resize(payload_size, 0);
    file.read_exact(&mut payload).map_err(|e| {
        format!(
            "Failed to read complete tensor payload from '{}': {e}",
            path.display()
        )
    })?;
    decode_tensor_payload(&payload, tensor, elements, shape)
}

/// Load tensor data from an in-memory GGUF image.
///
/// Kept for focused decoder tests; production loading streams one encoded
/// tensor at a time through `load_tensor_from_file`.
#[cfg(test)]
pub(super) fn load_tensor_data(
    file_data: &[u8],
    data_section_offset: usize,
    tensor: &GGUFTensorInfo,
    elements: usize,
) -> std::result::Result<ArrayD<f32>, String> {
    let tensor_offset = usize::try_from(tensor.offset)
        .map_err(|_| format!("Tensor offset {} does not fit usize", tensor.offset))?;
    let offset = data_section_offset
        .checked_add(tensor_offset)
        .ok_or("Tensor data offset overflow")?;
    let payload_size = tensor_payload_size(&tensor.tensor_type, elements)?;
    let end = offset
        .checked_add(payload_size)
        .ok_or("Tensor data end offset overflow")?;
    if offset > file_data.len() || end > file_data.len() {
        return Err(format!(
            "Tensor data out of bounds (offset={offset}, size={payload_size}, file_len={})",
            file_data.len()
        ));
    }
    let (shape, shape_elements) = checked_tensor_shape(tensor)?;
    if shape_elements != elements {
        return Err(format!(
            "Tensor element count mismatch (shape={shape_elements}, supplied={elements})"
        ));
    }
    decode_tensor_payload(&file_data[offset..end], tensor, elements, shape.as_slice())
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
