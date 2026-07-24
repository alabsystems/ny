// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ONNX file parsing and layer spec extraction.

mod attributes;
mod batch_resolve;
mod config;
mod const_fold;
mod convert;
mod fusion;
mod lstm_unroll;
pub(crate) mod numeric_cast;
mod parse;
mod shape_infer;
mod tensor;

use crate::model::{OriginalFloat32Initializer, OriginalOnnxNetwork};
use crate::{LayerSpec, Network, OnnxModel, TensorSpec, WeightStore};
use ny_core::{NyError, Result};
use std::path::Path;
use tracing::info;

/// Custom-operator extension traits and configuration for ONNX loading.
pub use config::{
    CustomOpHandler, CustomOpRegistry, OnnxLoadConfig, OnnxOptimizationFlag, ShapeInferBackend,
    ShapeInferencePolicy,
};
/// Subprocess shape-inference protocol: server entry + hidden subcommand name.
pub use shape_infer::subprocess::{serve_shape_infer_request, SHAPE_INFER_SUBCOMMAND};

/// Result of parsing an ONNX file, including loader-private raw-initializer
/// provenance captured before graph rewrites.
type ParsedOnnx = (
    Vec<LayerSpec>,
    WeightStore,
    Vec<TensorSpec>,
    Vec<TensorSpec>,
    std::collections::HashMap<String, String>, // tensor_producer map
    std::collections::HashSet<String>,         // constant_tensors set
    std::collections::HashMap<String, Vec<i64>>, // tensor_shapes map
    std::collections::HashMap<String, i64>,    // opset_imports map
    std::collections::HashMap<String, OriginalFloat32Initializer>,
);

/// Load an ONNX model from a file.
///
/// This function:
/// 1. Parses the ONNX protobuf
/// 2. Extracts graph structure (nodes, inputs, outputs)
/// 3. Extracts weights from initializers
/// 4. Creates a Network specification
/// 5. Uses ONNX Runtime shape inference to enrich tensor shapes when available
pub fn load_onnx<P: AsRef<Path>>(path: P) -> Result<OnnxModel> {
    let config = OnnxLoadConfig::default();
    load_onnx_with_config(path, &config)
}

/// Load an ONNX model from a file with explicit configuration.
pub fn load_onnx_with_config<P: AsRef<Path>>(
    path: P,
    config: &OnnxLoadConfig,
) -> Result<OnnxModel> {
    let path = path.as_ref();
    info!("Loading ONNX model from: {}", path.display());

    if !path.exists() {
        return Err(NyError::ModelLoad(format!(
            "File not found: {}",
            path.display()
        )));
    }

    let registry = config.merged_registry();
    let capture_provenance = config.raw_float32_initializer_provenance_enabled();
    reject_custom_handlers_for_provenance(&registry, capture_provenance)?;

    let (
        layers,
        weights,
        inputs,
        outputs,
        tensor_producer,
        constant_tensors,
        tensor_shapes,
        opset_imports,
        original_float32_initializers,
    ) = parse::parse_onnx_file(
        path,
        &registry,
        config.shape_inference_policy(),
        config.shape_infer_backend(),
        config.has_optimization_flag(OnnxOptimizationFlag::MergeLinear),
        capture_provenance,
    )?;

    let param_count = weights.iter().map(|(_, w)| w.len()).sum();

    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let network = Network {
        name,
        inputs,
        outputs,
        layers,
        param_count,
    };
    let original_network_topology =
        capture_provenance.then(|| OriginalOnnxNetwork::from_network(&network));

    info!(
        "Loaded model: {} layers, {} parameters, {} constant tensors",
        network.layers.len(),
        param_count,
        constant_tensors.len()
    );

    Ok(OnnxModel {
        network,
        weights,
        tensor_producer,
        constant_tensors,
        tensor_shapes,
        original_float32_initializers,
        original_network_topology,
        opset_imports,
    })
}

/// Load an ONNX model from in-memory bytes.
pub fn load_onnx_bytes(name: &str, data: &[u8]) -> Result<OnnxModel> {
    let config = OnnxLoadConfig::default();
    load_onnx_bytes_with_config(name, data, &config)
}

/// Load an ONNX model from in-memory bytes with explicit configuration.
pub fn load_onnx_bytes_with_config(
    name: &str,
    data: &[u8],
    config: &OnnxLoadConfig,
) -> Result<OnnxModel> {
    info!("Loading ONNX model from memory: {}", name);

    let registry = config.merged_registry();
    let capture_provenance = config.raw_float32_initializer_provenance_enabled();
    reject_custom_handlers_for_provenance(&registry, capture_provenance)?;

    let (
        layers,
        weights,
        inputs,
        outputs,
        tensor_producer,
        constant_tensors,
        tensor_shapes,
        opset_imports,
        original_float32_initializers,
    ) = parse::parse_onnx_bytes(
        data,
        &registry,
        config.shape_inference_policy(),
        config.shape_infer_backend(),
        config.has_optimization_flag(OnnxOptimizationFlag::MergeLinear),
        capture_provenance,
    )?;

    let param_count = weights.iter().map(|(_, w)| w.len()).sum();

    let network = Network {
        name: name.to_string(),
        inputs,
        outputs,
        layers,
        param_count,
    };
    let original_network_topology =
        capture_provenance.then(|| OriginalOnnxNetwork::from_network(&network));

    info!(
        "Loaded model: {} layers, {} parameters, {} constant tensors",
        network.layers.len(),
        param_count,
        constant_tensors.len()
    );

    Ok(OnnxModel {
        network,
        weights,
        tensor_producer,
        constant_tensors,
        tensor_shapes,
        original_float32_initializers,
        original_network_topology,
        opset_imports,
    })
}

fn reject_custom_handlers_for_provenance(
    registry: &CustomOpRegistry,
    capture_provenance: bool,
) -> Result<()> {
    if capture_provenance && !registry.handlers().is_empty() {
        return Err(NyError::ModelLoad(
            "raw FLOAT provenance requires the built-in ONNX conversion path without custom operator handlers"
                .to_string(),
        ));
    }
    Ok(())
}
