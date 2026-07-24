// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Model comparison and layer-by-layer diff functionality.
//!
//! This module provides tools to compare two ONNX models layer-by-layer,
//! identifying where outputs first diverge. Useful for debugging model ports
//! (e.g., PyTorch → CoreML/Metal conversions).
//!
//! ## Intermediate Layer Extraction
//!
//! To compare layer-by-layer, we modify the ONNX graph to expose all intermediate
//! tensors as outputs. This is done by:
//! 1. Parsing the ONNX protobuf
//! 2. Adding all node outputs to graph outputs
//! 3. Serializing back to bytes
//! 4. Loading into ONNX Runtime via commit_from_memory
//!
//! ## Root Cause Diagnosis
//!
//! When `--diagnose` is enabled, the diff analyzes divergence patterns to identify
//! common numerical issues:
//! - Softmax overflow (large logits near exp boundary)
//! - Accumulation order differences (non-associative float ops)
//! - Mixed precision errors (fp16/fp32 boundaries)
//! - Weight mismatches (actual different values, not numerical drift)

mod compare;
mod diagnosis;
mod engine;
mod inference;
mod io;
mod matching;
#[cfg(test)]
mod tests;
mod types;

/// End-to-end model diff entry points for file-based and in-memory ONNX models.
pub use engine::{diff_models, diff_models_bytes};
/// ONNX Runtime inference helpers, including intermediate-output exposure and a
/// session-reuse forward for repeated evaluations.
pub use inference::{
    expose_intermediate_outputs, read_input_shape_maybe_gzip, run_inference, run_inference_bytes,
    run_inference_with_intermediates, run_inference_with_intermediates_bytes, OrtForward,
};
/// Input/output loading helpers for model metadata and `.npy` tensors.
pub use io::{load_model_info, load_model_info_bytes, load_npy};
/// Layer-name matching heuristics used to pair corresponding layers across models.
pub use matching::match_layer_names;
/// Diff configuration, diagnosis, and per-layer comparison result types.
pub use types::{
    DiffConfig, DiffDiagnosis, DiffError, DiffResult, DiffStatus, DivergencePattern,
    LayerComparison, ModelInfo,
};

#[cfg(test)]
use compare::compare_arrays;
#[cfg(test)]
use diagnosis::{
    check_accumulation_pattern, check_gelu_pattern, check_layernorm_pattern,
    check_quantization_pattern, suggest_root_cause,
};
#[cfg(test)]
use matching::normalize_layer_name;
#[cfg(test)]
use ndarray::{ArrayD, IxDyn};
#[cfg(test)]
use ny_core::LayerType;
#[cfg(test)]
use std::collections::HashMap;
