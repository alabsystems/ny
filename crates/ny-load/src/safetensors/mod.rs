// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! SafeTensors format support for loading model weights.
//!
//! SafeTensors is a simple, safe, and fast format for storing tensors,
//! commonly used by Hugging Face for model weights.
//!
//! # Usage
//!
//! ```rust,no_run
//! use ny_load::safetensors::load_safetensors;
//!
//! let weights = load_safetensors("model.safetensors")
//!     .expect("failed to load weights");
//! for (name, tensor) in weights.iter() {
//!     tracing::info!("{}: {:?}", name, tensor.shape());
//! }
//! ```

mod convert;

#[cfg(test)]
mod tests;

use ny_build::WeightStore;
use ny_core::{checked_shape_product, NyError, Result};
use safetensors::SafeTensors;
use std::path::Path;
use tracing::{debug, info};

pub use convert::{bf16_to_f32, half_to_f32};

/// Load weights from a SafeTensors file.
///
/// Returns a `WeightStore` containing all tensors from the file.
/// Only f32, f16, bf16, f64, i32, and i64 tensors are supported
/// (all converted to f32).
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    info!("Loading SafeTensors from: {}", path.display());

    let data = read_safetensors_bytes(path)?;
    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse SafeTensors: {}", e)))?;

    let mut weights = WeightStore::new();

    for (name, tensor_view) in tensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let arr = convert::tensor_view_to_f32_array(&tensor_view, &shape, &name)?;
        debug!("Loaded tensor: {} shape {:?}", name, arr.shape());
        weights.insert(name.clone(), arr);
    }

    weights.validate_no_nan()?;
    info!("Loaded {} tensors from SafeTensors", weights.len());

    Ok(weights)
}

/// Metadata about a SafeTensors file.
#[derive(Debug, Clone)]
pub struct SafeTensorsInfo {
    /// Number of tensors in the file.
    pub tensor_count: usize,
    /// Total parameter count (sum of all tensor elements).
    pub param_count: usize,
    /// Tensor names and their shapes.
    pub tensors: Vec<(String, Vec<usize>, String)>, // (name, shape, dtype)
}

/// Get information about a SafeTensors file without fully loading it.
pub fn safetensors_info<P: AsRef<Path>>(path: P) -> Result<SafeTensorsInfo> {
    let path = path.as_ref();
    let data = read_safetensors_bytes(path)?;
    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse SafeTensors: {}", e)))?;

    let mut tensor_info = Vec::new();
    let mut param_count = 0;

    for (name, tensor_view) in tensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let dtype = format!("{:?}", tensor_view.dtype());
        let elements: usize = checked_shape_product(&shape).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "Tensor '{}' shape {:?} overflows usize",
                name, shape
            ))
        })?;
        param_count += elements;
        tensor_info.push((name.clone(), shape, dtype));
    }

    Ok(SafeTensorsInfo {
        tensor_count: tensor_info.len(),
        param_count,
        tensors: tensor_info,
    })
}

/// Read raw bytes from a SafeTensors file path with existence check.
fn read_safetensors_bytes(path: &Path) -> Result<Vec<u8>> {
    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }
    std::fs::read(path).map_err(|e| NyError::ModelLoad(format!("Failed to read file: {}", e)))
}
