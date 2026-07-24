// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_code)]

//! ONNX model loading and conversion to ny IR.
//!
//! This crate handles loading ONNX models and converting them to the internal
//! representation used by ny for neural network verification.
//!
//! ## Supported Operations
//!
//! - **Linear/Dense:** Gemm, MatMul, Linear
//! - **Convolution:** Conv (1D/2D), ConvTranspose, MaxPool, AveragePool, GlobalAveragePool
//! - **Attention:** MultiHeadAttention
//! - **Normalization:** BatchNorm, LayerNorm
//! - **Activations:** ReLU, LeakyReLU, PReLU, ELU, CELU, SELU, Sigmoid, Tanh, Softmax,
//!   LogSoftmax, GELU, SiLU (Swish), HardSigmoid, HardSwish, Mish, Softplus, Softsign,
//!   ThresholdedRelu
//!   - ONNX `Swish` is treated as a `SiLU` alias during conversion.
//! - **Math:** Add, Sub, Mul, Div, Neg, Min, Max, Abs, Exp, Log, Sqrt, Pow, Sin, Cos, Tan,
//!   Atan, Ceil, Floor, Round, Sign, Reciprocal, Shrink
//! - **Shape:** Reshape, Flatten, Squeeze, Unsqueeze, Transpose, Concat, Split (→Slice),
//!   Gather, ReduceMean, ReduceSum
//! - **Other:** Clip, Where, NonZero, CausalSoftmax (for transformers)
//!
//! ## Custom ONNX operators
//!
//! Use `OnnxLoadConfig` with a `CustomOpRegistry` to override or extend ONNX
//! operator conversion without global mutable state. Custom handlers are
//! tried in registry order before built-in op mappings, and each handler must
//! lower the custom node into a native [`LayerSpec`] using an existing
//! [`LayerType`]. ny does not keep a runtime custom layer node after
//! loading.
//!
//! ```rust,no_run
//! use ny_onnx::{CustomOpHandler, CustomOpRegistry, OnnxLoadConfig, load_onnx_with_config};
//! use ny_onnx::onnx_proto::NodeProto;
//! use ny_onnx::LayerSpec;
//! use ny_core::LayerType;
//! use std::sync::Arc;
//!
//! struct MyCustomOp;
//!
//! impl CustomOpHandler for MyCustomOp {
//!     fn try_convert(&self, node: &NodeProto) -> Option<LayerSpec> {
//!         if node.op_type == "MyCustomOp" {
//!             return Some(LayerSpec {
//!                 name: "my_custom".to_string(),
//!                 layer_type: LayerType::ReLU,
//!                 inputs: node.input.clone(),
//!                 outputs: node.output.clone(),
//!                 weights: None,
//!                 attributes: std::collections::HashMap::new(),
//!             });
//!         }
//!         None
//!     }
//! }
//!
//! let mut registry = CustomOpRegistry::default();
//! registry.register(Arc::new(MyCustomOp));
//! let config = OnnxLoadConfig::new(registry);
//! let _model = load_onnx_with_config("model.onnx", &config)?;
//! # Ok::<(), ny_core::NyError>(())
//! ```

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

/// Bound export: serialize per-node IBP bounds to safetensors for downstream training (#3520).
pub mod bound_export;
/// Static compute-cost analysis: per-layer FLOP and activation-memory estimates.
pub mod cost;
/// Model comparison and layer-by-layer diff for identifying where two models' outputs diverge.
pub mod diff;
/// Joint ONNX + VNN-LIB optimization passes (e.g., peeling off final layers).
pub mod optimization;
/// Bound width profiling: analyzes how bound widths propagate through a network.
pub mod profile;
/// Quantization safety analysis: checks whether outputs can safely be quantized to float16/int8.
pub mod quantize;
/// SafeTensors format loader for reading Hugging Face model weights.
pub mod safetensors;
/// Sensitivity analysis: measures how each layer amplifies input uncertainty.
pub mod sensitivity;
/// Weak-region mining for verification-guided training (#3520).
pub mod training_signal;

/// PyTorch format loader: reads `.pt` pickle/zip model files via candle-core's pickle parser.
#[cfg(feature = "pytorch")]
pub mod pytorch;

/// CoreML format loader: reads Apple `.mlmodel` and `.mlpackage` model formats.
#[cfg(feature = "coreml")]
pub mod coreml;

/// GGUF format loader: reads llama.cpp model weights including quantized data types.
#[cfg(feature = "gguf")]
pub mod gguf;

/// Native model loading without ONNX export: auto-detects architecture from weight files.
pub mod native;
/// NNet format loader: reads the text format for ReLU networks used in VNN-COMP (e.g., ACAS-Xu).
pub mod nnet;
/// VNN-LIB property specification parser: parses the SMT-LIB v2 based verification format.
pub mod vnnlib;

/// Shared error type for analysis modules (quantize, sensitivity, profile).
pub mod analysis_error;
mod decoder;
/// Bridge conversions from analysis error types to `NyError`.
mod error_bridge;
mod fallback_logging;
mod io;
mod loader;
mod model;
mod whisper;

/// Prost-derived protobuf struct definitions (`ModelProto`, etc.) for the ONNX format.
pub mod onnx_proto;

#[cfg(test)]
pub(crate) mod test_fixtures;
#[cfg(test)]
mod tests;

/// Static compute-cost analysis for fixed-shape models.
pub use cost::{
    estimate_model_cost, estimate_model_timing, CostResult, FamilyTimingCalibration, LayerCost,
    LayerTimingEstimate, TimingEstimate, TimingProfile,
};
/// Decoder model loading: auto-detect and parse transformer decoder blocks.
pub use decoder::{load_decoder, DecoderBlockInfo, DecoderModel, DecoderStructure};
/// ONNX model loaders with optional custom operator registry.
pub use loader::{
    load_onnx, load_onnx_bytes, load_onnx_bytes_with_config, load_onnx_with_config,
    serve_shape_infer_request, CustomOpHandler, CustomOpRegistry, OnnxLoadConfig,
    OnnxOptimizationFlag, ShapeInferBackend, ShapeInferencePolicy, SHAPE_INFER_SUBCOMMAND,
};
/// ONNX model representation: layer specs, weight storage, and conversion options.
pub use model::{
    is_multi_output_split, resolve_dynamic_dim, resolve_dynamic_shape, AttributeValue,
    CompoundNodePolicy, DataType, GraphNetworkOptions, LayerSpec, MissingOutputPolicy, Network,
    OnnxModel, PropagateNetworkOptions, TensorSpec, WeightRef, WeightStore,
};
/// Network optimization passes (e.g., peeling off final softmax layers).
pub use optimization::{
    peel_off_last_softmax_layer, peel_off_terminal_sigmoid_auto, PeelOffReport,
};
/// Whisper model loading: encoder structure detection and compositional verification support.
pub use whisper::{
    generate_whisper_export_script, load_whisper, CompositionalVerificationDetails,
    GpuCompositionalDetails, MultiBlockConfig, MultiBlockDetails, WhisperBlockInfo,
    WhisperEncoderStructure, WhisperModel,
};
