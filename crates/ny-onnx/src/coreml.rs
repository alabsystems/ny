// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CoreML format support for loading model weights.
//!
//! CoreML is Apple's machine learning framework. Models are stored as:
//! - `.mlmodel` - Single file format (older)
//! - `.mlpackage` - Directory format (newer)
//!
//! # Usage
//!
//! ```rust,no_run
//! use ny_onnx::coreml::load_coreml;
//!
//! let weights = load_coreml("model.mlmodel").unwrap();
//! for (name, tensor) in weights.iter() {
//!     println!("{}: {:?}", name, tensor.shape());
//! }
//! ```

use crate::WeightStore;
use coreml_proto::proto::Model;
use ndarray::{ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use prost::Message as _; // prost 0.14 Message trait for decode
use std::path::Path;
use tracing::{debug, info, warn};

/// Load weights from a CoreML .mlmodel or .mlpackage file.
///
/// Returns a `WeightStore` containing all weight tensors from the model.
///
/// # Arguments
///
/// * `path` - Path to the .mlmodel or .mlpackage file
///
/// # Returns
///
/// A `WeightStore` with all loaded weight tensors.
pub fn load_coreml<P: AsRef<Path>>(path: P) -> Result<WeightStore> {
    let path = path.as_ref();
    info!("Loading CoreML weights from: {}", path.display());

    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    // Determine if it's a .mlpackage (directory) or .mlmodel (file)
    let model_data = if path.is_dir() {
        // .mlpackage format - find the model.mlmodel inside
        load_mlpackage_data(path)?
    } else {
        // .mlmodel format - read directly
        std::fs::read(path)
            .map_err(|e| NyError::ModelLoad(format!("Failed to read file: {}", e)))?
    };

    // Parse the protobuf
    let model = Model::decode(&model_data[..])
        .map_err(|e| NyError::ModelLoad(format!("Failed to decode CoreML model: {}", e)))?;

    // Extract weights from the model
    extract_weights(&model)
}

/// Load model data from a .mlpackage directory.
fn load_mlpackage_data(package_path: &Path) -> Result<Vec<u8>> {
    // Common locations for the model file inside .mlpackage
    let possible_paths = [
        package_path.join("Data/com.apple.CoreML/model.mlmodel"),
        package_path.join("Data/com.apple.CoreML/weights/weight.bin"),
        package_path.join("model.mlmodel"),
    ];

    for model_path in &possible_paths {
        if model_path.exists() {
            return std::fs::read(model_path)
                .map_err(|e| NyError::ModelLoad(format!("Failed to read model file: {}", e)));
        }
    }

    // Try to find any .mlmodel file recursively
    if let Some(mlmodel) = find_mlmodel_in_dir(package_path) {
        return std::fs::read(&mlmodel)
            .map_err(|e| NyError::ModelLoad(format!("Failed to read model file: {}", e)));
    }

    Err(NyError::ModelLoad(format!(
        "Could not find model file in .mlpackage: {}",
        package_path.display()
    )))
}

/// Recursively find a .mlmodel file in a directory.
fn find_mlmodel_in_dir(dir: &Path) -> Option<std::path::PathBuf> {
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();

        if path.is_dir() {
            if let Some(found) = find_mlmodel_in_dir(&path) {
                return Some(found);
            }
        } else if path.extension().map(|e| e == "mlmodel").unwrap_or(false) {
            return Some(path);
        }
    }
    None
}

/// Extract weights from a parsed CoreML model.
fn extract_weights(model: &Model) -> Result<WeightStore> {
    use coreml_proto::proto::model::Type;

    let mut weights = WeightStore::new();
    let mut loaded_count = 0;

    // Extract based on model type
    if let Some(ref model_type) = model.r#type {
        match model_type {
            Type::NeuralNetwork(nn) => {
                extract_neural_network_weights(&mut weights, nn, "")?;
                loaded_count = weights.len();
            }
            Type::NeuralNetworkClassifier(nn) => {
                // NeuralNetworkClassifier has layers directly
                extract_classifier_weights(&mut weights, nn, "")?;
                loaded_count = weights.len();
            }
            Type::NeuralNetworkRegressor(nn) => {
                // NeuralNetworkRegressor has layers directly
                extract_regressor_weights(&mut weights, nn, "")?;
                loaded_count = weights.len();
            }
            Type::MlProgram(_) => {
                // ML Program format (newer CoreML) - more complex to parse
                warn!(
                    "ML Program format not fully supported - weight extraction may be incomplete"
                );
            }
            _ => {
                warn!(
                    "Model type {:?} not supported for weight extraction",
                    model_type
                );
            }
        }
    }

    weights.validate_no_nan()?;
    info!("Loaded {} weight tensors from CoreML model", loaded_count);
    Ok(weights)
}

/// Extract weights from a NeuralNetwork proto message.
fn extract_neural_network_weights(
    weights: &mut WeightStore,
    nn: &coreml_proto::proto::NeuralNetwork,
    prefix: &str,
) -> Result<()> {
    for layer in &nn.layers {
        let layer_name = if prefix.is_empty() {
            layer.name.clone()
        } else {
            format!("{}.{}", prefix, layer.name)
        };

        // Extract weights from each layer type
        if let Some(ref layer_type) = layer.layer {
            extract_layer_weights(weights, &layer_name, layer_type)?;
        }
    }
    Ok(())
}

/// Extract weights from a NeuralNetworkClassifier proto message.
fn extract_classifier_weights(
    weights: &mut WeightStore,
    nn: &coreml_proto::proto::NeuralNetworkClassifier,
    prefix: &str,
) -> Result<()> {
    for layer in &nn.layers {
        let layer_name = if prefix.is_empty() {
            layer.name.clone()
        } else {
            format!("{}.{}", prefix, layer.name)
        };

        if let Some(ref layer_type) = layer.layer {
            extract_layer_weights(weights, &layer_name, layer_type)?;
        }
    }
    Ok(())
}

/// Extract weights from a NeuralNetworkRegressor proto message.
fn extract_regressor_weights(
    weights: &mut WeightStore,
    nn: &coreml_proto::proto::NeuralNetworkRegressor,
    prefix: &str,
) -> Result<()> {
    for layer in &nn.layers {
        let layer_name = if prefix.is_empty() {
            layer.name.clone()
        } else {
            format!("{}.{}", prefix, layer.name)
        };

        if let Some(ref layer_type) = layer.layer {
            extract_layer_weights(weights, &layer_name, layer_type)?;
        }
    }
    Ok(())
}

/// Extract weights from a specific layer type.
fn extract_layer_weights(
    weights: &mut WeightStore,
    layer_name: &str,
    layer_type: &coreml_proto::proto::neural_network_layer::Layer,
) -> Result<()> {
    use coreml_proto::proto::neural_network_layer::Layer;

    match layer_type {
        Layer::InnerProduct(layer) => {
            // Linear layer weights
            if let Some(ref w) = layer.weights {
                let tensor = weight_params_to_array(
                    w,
                    &[
                        layer.output_channels as usize,
                        layer.input_channels as usize,
                    ],
                )?;
                weights.insert(format!("{}.weight", layer_name), tensor);
                debug!("Loaded {}.weight", layer_name);
            }
            if let Some(ref b) = layer.bias {
                let tensor = weight_params_to_array(b, &[layer.output_channels as usize])?;
                weights.insert(format!("{}.bias", layer_name), tensor);
                debug!("Loaded {}.bias", layer_name);
            }
        }
        Layer::Convolution(layer) => {
            // Conv layer weights
            if let Some(ref w) = layer.weights {
                let shape = vec![
                    layer.output_channels as usize,
                    layer.kernel_channels as usize,
                    *layer.kernel_size.first().unwrap_or(&1) as usize,
                    *layer.kernel_size.get(1).unwrap_or(&1) as usize,
                ];
                let tensor = weight_params_to_array(w, &shape)?;
                weights.insert(format!("{}.weight", layer_name), tensor);
                debug!("Loaded {}.weight", layer_name);
            }
            if let Some(ref b) = layer.bias {
                let tensor = weight_params_to_array(b, &[layer.output_channels as usize])?;
                weights.insert(format!("{}.bias", layer_name), tensor);
                debug!("Loaded {}.bias", layer_name);
            }
        }
        Layer::Batchnorm(layer) => {
            // BatchNorm parameters
            if let Some(ref gamma) = layer.gamma {
                let tensor = weight_params_to_array(gamma, &[layer.channels as usize])?;
                weights.insert(format!("{}.weight", layer_name), tensor);
            }
            if let Some(ref beta) = layer.beta {
                let tensor = weight_params_to_array(beta, &[layer.channels as usize])?;
                weights.insert(format!("{}.bias", layer_name), tensor);
            }
            if let Some(ref mean) = layer.mean {
                let tensor = weight_params_to_array(mean, &[layer.channels as usize])?;
                weights.insert(format!("{}.running_mean", layer_name), tensor);
            }
            if let Some(ref variance) = layer.variance {
                let tensor = weight_params_to_array(variance, &[layer.channels as usize])?;
                weights.insert(format!("{}.running_var", layer_name), tensor);
            }
        }
        Layer::Embedding(layer) => {
            // Embedding weights: input_dim = vocab size, output_channels = embedding size
            if let Some(ref w) = layer.weights {
                let tensor = weight_params_to_array(
                    w,
                    &[layer.output_channels as usize, layer.input_dim as usize],
                )?;
                weights.insert(format!("{}.weight", layer_name), tensor);
                debug!("Loaded {}.weight", layer_name);
            }
        }
        _ => {
            // Other layer types - skip weight extraction
            debug!("Skipping layer type for weight extraction: {}", layer_name);
        }
    }

    Ok(())
}

/// Convert CoreML WeightParams to an ndarray.
fn weight_params_to_array(
    params: &coreml_proto::proto::WeightParams,
    shape: &[usize],
) -> Result<ArrayD<f32>> {
    let total_elements: usize = checked_shape_product(shape).ok_or_else(|| {
        NyError::ModelLoad(format!(
            "Weight shape {:?} overflows usize on product",
            shape
        ))
    })?;

    // Try different value formats in order of preference
    let floats: Vec<f32> = if !params.float_value.is_empty() {
        // Direct f32 values
        params.float_value.clone()
    } else if !params.float16_value.is_empty() {
        // f16 values stored as bytes
        params
            .float16_value
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| {
                let bits = u16::from_le_bytes(*chunk);
                half_to_f32(bits)
            })
            .collect()
    } else if !params.raw_value.is_empty() {
        // Quantized values - interpret as u8 for now
        warn!("Loading quantized weights - values may need dequantization");
        params.raw_value.iter().map(|&v| v as f32).collect()
    } else if !params.int8_raw_value.is_empty() {
        // Int8 quantized values
        warn!("Loading int8 quantized weights - values may need dequantization");
        params
            .int8_raw_value
            .iter()
            .map(|&v| v as i8 as f32)
            .collect()
    } else {
        return Err(NyError::ModelLoad(
            "Weight params has no value data".to_string(),
        ));
    };

    // Verify element count
    if floats.len() != total_elements {
        return Err(NyError::ModelLoad(format!(
            "Weight params element count {} doesn't match shape {:?} (expected {})",
            floats.len(),
            shape,
            total_elements
        )));
    }

    ArrayD::from_shape_vec(IxDyn(shape), floats)
        .map_err(|e| NyError::ModelLoad(format!("Failed to create weight array: {}", e)))
}

/// Convert IEEE 754 half-precision (f16) to f32.
#[inline]
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = -14i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FF;
            let new_exp = ((127 + e) as u32) & 0xFF;
            f32::from_bits((sign << 31) | (new_exp << 23) | (f << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if frac == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::from_bits((sign << 31) | 0x7FC00000 | (frac << 13))
        }
    } else {
        // Normal
        let new_exp = (exp + 127 - 15) & 0xFF;
        f32::from_bits((sign << 31) | (new_exp << 23) | (frac << 13))
    }
}

/// Metadata about a CoreML file.
#[derive(Debug, Clone)]
pub struct CoreMLInfo {
    /// Specification version of the model.
    pub spec_version: i32,
    /// Model description (if available).
    pub description: Option<String>,
    /// Number of weight tensors extracted.
    pub tensor_count: usize,
    /// Total parameter count.
    pub param_count: usize,
    /// Tensor names and their shapes.
    pub tensors: Vec<(String, Vec<usize>, String)>, // (name, shape, dtype)
}

/// Get information about a CoreML file.
pub fn coreml_info<P: AsRef<Path>>(path: P) -> Result<CoreMLInfo> {
    let weights = load_coreml(path.as_ref())?;

    // Re-read the model to get metadata (we could optimize this)
    let model_data = if path.as_ref().is_dir() {
        load_mlpackage_data(path.as_ref())?
    } else {
        std::fs::read(path.as_ref())
            .map_err(|e| NyError::ModelLoad(format!("Failed to read file: {}", e)))?
    };

    let model = Model::decode(&model_data[..])
        .map_err(|e| NyError::ModelLoad(format!("Failed to decode CoreML model: {}", e)))?;

    let mut tensor_info = Vec::new();
    let mut param_count = 0;

    for (name, tensor) in weights.iter() {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let elements: usize = checked_shape_product(&shape).ok_or_else(|| {
            NyError::ModelLoad(format!(
                "Tensor '{}' shape {:?} overflows usize",
                name, shape
            ))
        })?;
        param_count += elements;
        tensor_info.push((name.to_string(), shape, "F32".to_string()));
    }

    let description = model.description.as_ref().and_then(|d| {
        if d.metadata.is_some() || !d.input.is_empty() || !d.output.is_empty() {
            Some(format!(
                "Inputs: {}, Outputs: {}",
                d.input.len(),
                d.output.len()
            ))
        } else {
            None
        }
    });

    Ok(CoreMLInfo {
        spec_version: model.specification_version,
        description,
        tensor_count: tensor_info.len(),
        param_count,
        tensors: tensor_info,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== load_coreml / coreml_info file not found tests =====

    #[ntest::timeout(10000)]
    #[test]
    fn test_file_not_found() {
        let result = load_coreml("/nonexistent/path/model.mlmodel");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_coreml_info_file_not_found() {
        let result = coreml_info("/nonexistent/path/model.mlmodel");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }

    // ===== half_to_f32 tests =====

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_positive_zero() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        // Check the sign bit is positive
        assert!(half_to_f32(0x0000).is_sign_positive());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_negative_zero() {
        let neg_zero = half_to_f32(0x8000);
        assert_eq!(neg_zero, -0.0);
        assert!(neg_zero.is_sign_negative());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_one() {
        assert!((half_to_f32(0x3C00) - 1.0).abs() < 1e-6);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_negative_one() {
        assert!((half_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_two() {
        // 2.0 in f16 is 0x4000
        assert!((half_to_f32(0x4000) - 2.0).abs() < 1e-6);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_half() {
        // 0.5 in f16 is 0x3800
        assert!((half_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_negative_two() {
        // -2.0 in f16 is 0xC000
        assert!((half_to_f32(0xC000) - (-2.0)).abs() < 1e-6);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_positive_infinity() {
        // +Infinity in f16 is 0x7C00
        let inf = half_to_f32(0x7C00);
        assert!(inf.is_infinite());
        assert!(inf.is_sign_positive());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_negative_infinity() {
        // -Infinity in f16 is 0xFC00
        let neg_inf = half_to_f32(0xFC00);
        assert!(neg_inf.is_infinite());
        assert!(neg_inf.is_sign_negative());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_nan() {
        // NaN in f16: exponent all 1s, fraction non-zero
        // 0x7C01 is a quiet NaN
        let nan = half_to_f32(0x7C01);
        assert!(nan.is_nan());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_negative_nan() {
        // -NaN in f16: sign=1, exponent all 1s, fraction non-zero
        let nan = half_to_f32(0xFC01);
        assert!(nan.is_nan());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_max_normal() {
        // Max normal f16 is 65504 = 0x7BFF
        let max_normal = half_to_f32(0x7BFF);
        assert!((max_normal - 65504.0).abs() < 1.0);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_negative_max_normal() {
        // -Max normal f16 is -65504 = 0xFBFF
        let neg_max = half_to_f32(0xFBFF);
        assert!((neg_max - (-65504.0)).abs() < 1.0);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_min_positive_normal() {
        // Min positive normal f16 is ~6.10e-5 = 0x0400
        let min_normal = half_to_f32(0x0400);
        assert!(min_normal > 0.0);
        assert!((min_normal - 6.103_515_6e-5).abs() < 1e-8);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_subnormal() {
        // Smallest positive subnormal f16 is 0x0001
        let subnormal = half_to_f32(0x0001);
        assert!(subnormal > 0.0);
        assert!(subnormal < 1e-4); // Very small
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_larger_subnormal() {
        // A larger subnormal: 0x03FF (all fraction bits set, exp=0)
        let subnormal = half_to_f32(0x03FF);
        assert!(subnormal > 0.0);
        // This should be close to the min normal
        assert!(subnormal < 6.2e-5);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_negative_subnormal() {
        // Negative subnormal: 0x8001
        let neg_subnormal = half_to_f32(0x8001);
        assert!(neg_subnormal < 0.0);
        assert!(neg_subnormal.abs() < 1e-4);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_small_positive() {
        // 0.1 in f16 is approximately 0x2E66
        let val = half_to_f32(0x2E66);
        assert!((val - 0.1).abs() < 0.001);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_half_to_f32_pi_approx() {
        // PI in f16 is approximately 0x4248 (3.140625)
        let val = half_to_f32(0x4248);
        assert!((val - std::f64::consts::PI as f32).abs() < 0.01);
    }

    // ===== CoreMLInfo tests =====

    #[ntest::timeout(10000)]
    #[test]
    fn test_coreml_info_struct_fields() {
        let info = CoreMLInfo {
            spec_version: 6,
            description: Some("Test model".to_string()),
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
        assert_eq!(info.spec_version, 6);
        assert_eq!(info.description, Some("Test model".to_string()));
        assert_eq!(info.tensor_count, 10);
        assert_eq!(info.param_count, 1000);
        assert_eq!(info.tensors.len(), 2);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_coreml_info_no_description() {
        let info = CoreMLInfo {
            spec_version: 5,
            description: None,
            tensor_count: 5,
            param_count: 500,
            tensors: vec![],
        };
        assert!(info.description.is_none());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_coreml_info_debug() {
        let info = CoreMLInfo {
            spec_version: 6,
            description: Some("Test".to_string()),
            tensor_count: 3,
            param_count: 300,
            tensors: vec![],
        };
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("CoreMLInfo"));
        assert!(debug_str.contains("spec_version: 6"));
        assert!(debug_str.contains("tensor_count: 3"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_coreml_info_clone() {
        let info = CoreMLInfo {
            spec_version: 7,
            description: Some("Clone test".to_string()),
            tensor_count: 8,
            param_count: 800,
            tensors: vec![("test".to_string(), vec![10, 20], "F16".to_string())],
        };
        let cloned = info.clone();
        assert_eq!(cloned.spec_version, info.spec_version);
        assert_eq!(cloned.description, info.description);
        assert_eq!(cloned.tensor_count, info.tensor_count);
        assert_eq!(cloned.param_count, info.param_count);
        assert_eq!(cloned.tensors.len(), info.tensors.len());
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_coreml_info_empty_tensors() {
        let info = CoreMLInfo {
            spec_version: 1,
            description: None,
            tensor_count: 0,
            param_count: 0,
            tensors: vec![],
        };
        assert_eq!(info.tensor_count, 0);
        assert_eq!(info.param_count, 0);
        assert!(info.tensors.is_empty());
    }
}
