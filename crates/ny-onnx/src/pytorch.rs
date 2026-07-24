// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! PyTorch format support for loading model weights.
//!
//! PyTorch saves models as pickle files (optionally compressed in a zip archive).
//! This module uses candle-core's pickle parser to extract tensor data.
//!
//! # Usage
//!
//! ```rust,no_run
//! use ny_onnx::pytorch::load_pytorch;
//!
//! let weights = load_pytorch("model.pt").unwrap();
//! for (name, tensor) in weights.iter() {
//!     println!("{}: {:?}", name, tensor.shape());
//! }
//! ```
//!
//! # Supported Formats
//!
//! - `.pt` - PyTorch model files (state_dict or full model)
//! - `.pth` - PyTorch checkpoint files
//! - `.bin` - PyTorch binary format (used by some Hugging Face models)
//!
//! # Nested State Dicts
//!
//! Some models (like Kokoro TTS) save weights as nested dicts (dict of state_dicts).
//! This module automatically flattens such structures.

use crate::WeightStore;
use candle_core::pickle::PthTensors;
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use tracing::{debug, info, warn};
use zip::ZipArchive;

/// Load weights from a PyTorch .pt/.pth file.
///
/// Returns a `WeightStore` containing all tensors from the file.
/// All tensor dtypes are converted to f32.
///
/// Handles both flat state_dicts and nested dicts (dict of state_dicts).
///
/// # Arguments
///
/// * `path` - Path to the .pt/.pth file
///
/// # Returns
///
/// A `WeightStore` with all loaded tensors.
pub fn load_pytorch<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    info!("Loading PyTorch weights from: {}", path.display());

    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    // First try standard candle loader
    let tensors = PthTensors::new(path, None)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse PyTorch file: {}", e)))?;

    let tensor_names: Vec<String> = tensors.tensor_infos().keys().cloned().collect();

    // If we got tensors, use the standard path
    if !tensor_names.is_empty() {
        return load_flat_pytorch(tensors, &tensor_names);
    }

    // Otherwise, try to load as nested dict
    info!("No flat tensors found, attempting nested dict loading...");
    load_nested_pytorch(path)
}

/// Load PyTorch file with flat structure.
fn load_flat_pytorch(tensors: PthTensors, tensor_names: &[String]) -> Result<WeightStore> {
    let mut weights = WeightStore::new();
    let mut loaded_count = 0;
    let mut skipped_count = 0;

    for name in tensor_names {
        match tensors.get(name) {
            Ok(Some(tensor)) => match candle_to_ndarray(&tensor) {
                Ok(array) => {
                    debug!("Loaded tensor: {} shape {:?}", name, array.shape());
                    weights.insert(name.clone(), array);
                    loaded_count += 1;
                }
                Err(e) => {
                    warn!("Failed to convert tensor '{}': {}", name, e);
                    skipped_count += 1;
                }
            },
            Ok(None) => {
                warn!("Tensor '{}' not found in file", name);
                skipped_count += 1;
            }
            Err(e) => {
                warn!("Failed to load tensor '{}': {}", name, e);
                skipped_count += 1;
            }
        }
    }

    weights.validate_no_nan()?;
    info!(
        "Loaded {} tensors from PyTorch file ({} skipped)",
        loaded_count, skipped_count
    );

    Ok(weights)
}

/// Read pickle data from a PyTorch zip archive.
fn read_pickle_from_archive<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
) -> Result<Vec<u8>> {
    // Try model/data.pkl first (standard PyTorch format)
    let pkl_name = if archive.by_name("model/data.pkl").is_ok() {
        "model/data.pkl"
    } else if archive.by_name("data.pkl").is_ok() {
        "data.pkl"
    } else {
        return Err(NyError::ModelLoad(
            "No pickle file found in archive".to_string(),
        ));
    };

    let mut pkl_file = archive
        .by_name(pkl_name)
        .map_err(|e| NyError::ModelLoad(format!("Failed to open pickle file: {}", e)))?;

    let mut data = Vec::new();
    pkl_file
        .read_to_end(&mut data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to read pickle file: {}", e)))?;

    Ok(data)
}

/// Load PyTorch file with nested dict structure.
///
/// This handles models that save as dict[str, state_dict] rather than flat state_dict.
fn load_nested_pytorch<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    let file =
        File::open(path).map_err(|e| NyError::ModelLoad(format!("Failed to open file: {}", e)))?;

    let mut archive = ZipArchive::new(BufReader::new(file))
        .map_err(|e| NyError::ModelLoad(format!("Failed to read zip archive: {}", e)))?;

    // Read the pickle file to understand the structure
    let pickle_data = read_pickle_from_archive(&mut archive)?;

    // Parse pickle to get nested structure
    let nested_keys = parse_nested_pickle_keys(&pickle_data)?;
    info!("Found nested keys: {:?}", nested_keys);

    // Try to load each nested state dict
    let mut weights = WeightStore::new();
    let mut total_loaded = 0;

    for prefix in nested_keys {
        info!("Loading nested dict: {}", prefix);

        // Try loading with the prefix
        match PthTensors::new(path, Some(&prefix)) {
            Ok(tensors) => {
                let tensor_names: Vec<String> = tensors.tensor_infos().keys().cloned().collect();
                for name in &tensor_names {
                    match tensors.get(name) {
                        Ok(Some(tensor)) => match candle_to_ndarray(&tensor) {
                            Ok(array) => {
                                let full_name = format!("{}.{}", prefix, name);
                                debug!("Loaded tensor: {} shape {:?}", full_name, array.shape());
                                weights.insert(full_name, array);
                                total_loaded += 1;
                            }
                            Err(e) => {
                                warn!("PyTorch weight {}.{}: conversion failed: {e}", prefix, name);
                            }
                        },
                        Ok(None) => {} // weight not present in this prefix
                        Err(e) => {
                            warn!(
                                "PyTorch weight {}.{}: deserialization failed: {e}",
                                prefix, name
                            );
                        }
                    }
                }
            }
            Err(e) => {
                debug!("Failed to load nested dict '{}': {}", prefix, e);
            }
        }
    }

    if total_loaded == 0 {
        // Last resort: try to parse all data files directly
        info!("Attempting direct data file parsing...");
        return load_pytorch_data_files(path);
    }

    weights.validate_no_nan()?;
    info!("Loaded {} tensors from nested PyTorch file", total_loaded);
    Ok(weights)
}

/// Parse pickle data to extract nested dict keys.
fn parse_nested_pickle_keys(data: &[u8]) -> Result<Vec<String>> {
    // Simple heuristic: look for string patterns that appear to be dict keys
    // PyTorch pickle uses BINUNICODE (X) or SHORT_BINUNICODE (U) for strings

    let mut keys = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Look for SHORT_BINUNICODE (U = 0x55 followed by length byte)
        if data[i] == 0x55 && i + 1 < data.len() {
            let len = data[i + 1] as usize;
            if i + 2 + len <= data.len() {
                if let Ok(s) = std::str::from_utf8(&data[i + 2..i + 2 + len]) {
                    // Filter to likely model component names
                    if is_likely_model_key(s) {
                        keys.push(s.to_string());
                    }
                }
            }
        }
        // Look for BINUNICODE (X = 0x58 followed by 4-byte length)
        else if data[i] == 0x58 && i + 5 < data.len() {
            let len =
                u32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]) as usize;
            if i + 5 + len <= data.len() {
                if let Ok(s) = std::str::from_utf8(&data[i + 5..i + 5 + len]) {
                    if is_likely_model_key(s) {
                        keys.push(s.to_string());
                    }
                }
            }
        }
        i += 1;
    }

    // Deduplicate
    keys.sort();
    keys.dedup();

    Ok(keys)
}

/// Check if a string looks like a top-level model component name.
fn is_likely_model_key(s: &str) -> bool {
    // Top-level model component names (exact matches only)
    let top_level_names = [
        "bert",
        "encoder",
        "decoder",
        "predictor",
        "text_encoder",
        "bert_encoder",
        "flow",
        "hift",
        "transformer",
        "model",
        "generator",
        "discriminator",
        "embeddings",
        "attention",
    ];

    // Only accept exact matches for top-level components
    // This prevents picking up intermediate paths like "module.F0.0.conv1"
    top_level_names.contains(&s)
}

/// Load PyTorch data files directly as a fallback.
fn load_pytorch_data_files<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    let file =
        File::open(path).map_err(|e| NyError::ModelLoad(format!("Failed to open file: {}", e)))?;

    let mut archive = ZipArchive::new(BufReader::new(file))
        .map_err(|e| NyError::ModelLoad(format!("Failed to read zip archive: {}", e)))?;

    // Get list of data files
    let mut data_files = Vec::new();
    for i in 0..archive.len() {
        match archive.by_index(i) {
            Ok(file) => {
                let name = file.name().to_string();
                if name.contains("data/") && !name.ends_with('/') {
                    data_files.push(name);
                }
            }
            Err(e) => {
                warn!("PyTorch archive: skipping entry {i}: {e}");
            }
        }
    }

    info!("Found {} data files in archive", data_files.len());

    // For now, we can't load without the tensor metadata
    // Return an error indicating the format isn't supported
    Err(NyError::ModelLoad(format!(
        "Model has nested dict structure that cannot be loaded directly. \
         Found {} data files but no flat tensor structure. \
         Consider converting to SafeTensors format for better compatibility.",
        data_files.len()
    )))
}

/// Convert a candle tensor to an ndarray.
fn candle_to_ndarray(tensor: &candle_core::Tensor) -> Result<ArrayD<f32>> {
    use candle_core::DType;

    // Get shape
    let shape: Vec<usize> = tensor.dims().to_vec();

    // Convert to f32 and flatten
    let tensor_f32 = match tensor.dtype() {
        DType::F32 => tensor.clone(),
        DType::F64 => tensor
            .to_dtype(DType::F32)
            .map_err(|e| NyError::ModelLoad(format!("Failed to convert f64 to f32: {}", e)))?,
        DType::F16 => tensor
            .to_dtype(DType::F32)
            .map_err(|e| NyError::ModelLoad(format!("Failed to convert f16 to f32: {}", e)))?,
        DType::BF16 => tensor
            .to_dtype(DType::F32)
            .map_err(|e| NyError::ModelLoad(format!("Failed to convert bf16 to f32: {}", e)))?,
        DType::I64 => tensor
            .to_dtype(DType::F32)
            .map_err(|e| NyError::ModelLoad(format!("Failed to convert i64 to f32: {}", e)))?,
        DType::U8 => tensor
            .to_dtype(DType::F32)
            .map_err(|e| NyError::ModelLoad(format!("Failed to convert u8 to f32: {}", e)))?,
        DType::U32 => tensor
            .to_dtype(DType::F32)
            .map_err(|e| NyError::ModelLoad(format!("Failed to convert u32 to f32: {}", e)))?,
        DType::I16 => tensor
            .to_dtype(DType::F32)
            .map_err(|e| NyError::ModelLoad(format!("Failed to convert i16 to f32: {}", e)))?,
        DType::I32 => tensor
            .to_dtype(DType::F32)
            .map_err(|e| NyError::ModelLoad(format!("Failed to convert i32 to f32: {}", e)))?,
        DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
            tensor.to_dtype(DType::F32).map_err(|e| {
                NyError::ModelLoad(format!(
                    "Failed to convert {:?} to f32: {}",
                    tensor.dtype(),
                    e
                ))
            })?
        }
        // candle_core::DType is #[non_exhaustive]; any future dtype takes the
        // same widen-to-f32 path as every non-F32 case above.
        _ => tensor.to_dtype(DType::F32).map_err(|e| {
            NyError::ModelLoad(format!(
                "Failed to convert {:?} to f32: {}",
                tensor.dtype(),
                e
            ))
        })?,
    };

    // Flatten to CPU and get data
    let flattened = tensor_f32
        .flatten_all()
        .map_err(|e| NyError::ModelLoad(format!("Failed to flatten tensor: {}", e)))?;

    let data: Vec<f32> = flattened
        .to_vec1()
        .map_err(|e| NyError::ModelLoad(format!("Failed to extract tensor data: {}", e)))?;

    // Create ndarray
    ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| NyError::ModelLoad(format!("Failed to create ndarray: {}", e)))
}

/// Metadata about a PyTorch file.
#[derive(Debug, Clone)]
pub struct PyTorchInfo {
    /// Number of tensors in the file.
    pub tensor_count: usize,
    /// Total parameter count (sum of all tensor elements).
    pub param_count: usize,
    /// Tensor names and their shapes.
    pub tensors: Vec<(String, Vec<usize>, String)>, // (name, shape, dtype)
}

/// Get information about a PyTorch file without fully loading tensors.
pub fn pytorch_info<P: AsRef<Path>>(path: P) -> Result<PyTorchInfo> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let tensors = PthTensors::new(path, None)
        .map_err(|e| NyError::ModelLoad(format!("Failed to parse PyTorch file: {}", e)))?;

    let mut tensor_info = Vec::new();
    let mut param_count = 0;

    for (name, info) in tensors.tensor_infos() {
        let shape: Vec<usize> = info.layout.shape().dims().to_vec();
        let dtype = format!("{:?}", info.dtype);
        let elements: usize = checked_shape_product(&shape).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "Tensor '{}' shape {:?} overflows usize on product",
                name, shape
            ))
        })?;
        param_count += elements;
        tensor_info.push((name.clone(), shape, dtype));
    }

    Ok(PyTorchInfo {
        tensor_count: tensor_info.len(),
        param_count,
        tensors: tensor_info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== load_pytorch / pytorch_info file not found tests =====

    #[ntest::timeout(10000)]
    #[test]
    fn test_file_not_found() {
        let result = load_pytorch("/nonexistent/path/model.pt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_info_file_not_found() {
        let result = pytorch_info("/nonexistent/path/model.pt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    // ===== is_likely_model_key tests =====

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_bert() {
        assert!(is_likely_model_key("bert"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_encoder() {
        assert!(is_likely_model_key("encoder"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_decoder() {
        assert!(is_likely_model_key("decoder"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_predictor() {
        assert!(is_likely_model_key("predictor"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_text_encoder() {
        assert!(is_likely_model_key("text_encoder"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_bert_encoder() {
        assert!(is_likely_model_key("bert_encoder"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_flow() {
        assert!(is_likely_model_key("flow"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_hift() {
        assert!(is_likely_model_key("hift"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_transformer() {
        assert!(is_likely_model_key("transformer"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_model() {
        assert!(is_likely_model_key("model"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_generator() {
        assert!(is_likely_model_key("generator"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_discriminator() {
        assert!(is_likely_model_key("discriminator"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_embeddings() {
        assert!(is_likely_model_key("embeddings"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_attention() {
        assert!(is_likely_model_key("attention"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_not_a_model_key_weight() {
        // "weight" is not a top-level model component
        assert!(!is_likely_model_key("weight"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_not_a_model_key_bias() {
        assert!(!is_likely_model_key("bias"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_not_a_model_key_layer_name() {
        // Nested paths should not match
        assert!(!is_likely_model_key("module.F0.0.conv1"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_not_a_model_key_random() {
        assert!(!is_likely_model_key("random_string"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_not_a_model_key_empty() {
        assert!(!is_likely_model_key(""));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_not_a_model_key_partial_match() {
        // Should not match partial strings like "encoder_layer"
        assert!(!is_likely_model_key("encoder_layer"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_is_likely_model_key_not_a_model_key_substring() {
        // "model_state" contains "model" but is not an exact match
        assert!(!is_likely_model_key("model_state"));
    }

    // ===== parse_nested_pickle_keys tests =====

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_empty() {
        let data: &[u8] = &[];
        let keys = parse_nested_pickle_keys(data).unwrap();
        assert!(keys.is_empty());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_no_model_keys() {
        // Some random bytes that don't contain model keys
        let data: &[u8] = &[0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
        let keys = parse_nested_pickle_keys(data).unwrap();
        assert!(keys.is_empty());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_short_binunicode_encoder() {
        // SHORT_BINUNICODE (0x55) followed by length 7, then "encoder"
        let mut data = vec![0x55, 7];
        data.extend_from_slice(b"encoder");
        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(keys.contains(&"encoder".to_string()));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_short_binunicode_decoder() {
        let mut data = vec![0x55, 7];
        data.extend_from_slice(b"decoder");
        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(keys.contains(&"decoder".to_string()));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_binunicode_transformer() {
        // BINUNICODE (0x58) followed by 4-byte length (little-endian), then "transformer"
        let mut data = vec![0x58, 11, 0, 0, 0]; // length = 11
        data.extend_from_slice(b"transformer");
        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(keys.contains(&"transformer".to_string()));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_binunicode_model() {
        let mut data = vec![0x58, 5, 0, 0, 0]; // length = 5
        data.extend_from_slice(b"model");
        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(keys.contains(&"model".to_string()));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_multiple_keys() {
        // Two SHORT_BINUNICODE strings
        let mut data = vec![];
        // "encoder"
        data.push(0x55);
        data.push(7);
        data.extend_from_slice(b"encoder");
        // Some padding
        data.extend_from_slice(&[0x00, 0x01, 0x02]);
        // "decoder"
        data.push(0x55);
        data.push(7);
        data.extend_from_slice(b"decoder");

        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(keys.contains(&"encoder".to_string()));
        assert!(keys.contains(&"decoder".to_string()));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_deduplication() {
        // Same key twice should be deduplicated
        let mut data = vec![];
        // "encoder" first time
        data.push(0x55);
        data.push(7);
        data.extend_from_slice(b"encoder");
        // "encoder" second time
        data.push(0x55);
        data.push(7);
        data.extend_from_slice(b"encoder");

        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert_eq!(keys.iter().filter(|&k| k == "encoder").count(), 1);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_ignores_non_model_strings() {
        // SHORT_BINUNICODE with a non-model key like "weight"
        let mut data = vec![0x55, 6];
        data.extend_from_slice(b"weight");
        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(!keys.contains(&"weight".to_string()));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_truncated_short_binunicode() {
        // SHORT_BINUNICODE says length 10 but only has 5 bytes
        let mut data = vec![0x55, 10];
        data.extend_from_slice(b"short");
        // Should not crash, just skip this entry
        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(keys.is_empty());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_truncated_binunicode() {
        // BINUNICODE says length 20 but only has 5 bytes
        let mut data = vec![0x58, 20, 0, 0, 0];
        data.extend_from_slice(b"short");
        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(keys.is_empty());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_parse_nested_pickle_keys_mixed_valid_invalid() {
        let mut data = vec![];
        // Valid "encoder"
        data.push(0x55);
        data.push(7);
        data.extend_from_slice(b"encoder");
        // Invalid (truncated) "trans..."
        data.push(0x55);
        data.push(20);
        data.extend_from_slice(b"trans");

        let keys = parse_nested_pickle_keys(&data).unwrap();
        assert!(keys.contains(&"encoder".to_string()));
    }

    // ===== PyTorchInfo tests =====

    #[ntest::timeout(10000)]
    #[test]
    fn test_pytorch_info_struct_fields() {
        let info = PyTorchInfo {
            tensor_count: 10,
            param_count: 1000,
            tensors: vec![
                (
                    "layer1.weight".to_string(),
                    vec![256, 128],
                    "F32".to_string(),
                ),
                ("layer1.bias".to_string(), vec![256], "F32".to_string()),
            ],
        };
        assert_eq!(info.tensor_count, 10);
        assert_eq!(info.param_count, 1000);
        assert_eq!(info.tensors.len(), 2);
        assert_eq!(info.tensors[0].0, "layer1.weight");
        assert_eq!(info.tensors[0].1, vec![256, 128]);
        assert_eq!(info.tensors[0].2, "F32");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_pytorch_info_debug() {
        let info = PyTorchInfo {
            tensor_count: 5,
            param_count: 500,
            tensors: vec![],
        };
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("PyTorchInfo"));
        assert!(debug_str.contains("tensor_count: 5"));
        assert!(debug_str.contains("param_count: 500"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_pytorch_info_clone() {
        let info = PyTorchInfo {
            tensor_count: 3,
            param_count: 300,
            tensors: vec![("test".to_string(), vec![10], "F16".to_string())],
        };
        let cloned = info.clone();
        assert_eq!(cloned.tensor_count, info.tensor_count);
        assert_eq!(cloned.param_count, info.param_count);
        assert_eq!(cloned.tensors.len(), info.tensors.len());
    }
}
