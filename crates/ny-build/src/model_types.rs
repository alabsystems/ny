// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core model specification types shared between ny-onnx (parsing) and
//! ny-build (construction).
//!
//! These types describe a loaded neural network's structure before conversion
//! to propagation types. They are intentionally decoupled from the full
//! `OnnxModel` to allow construction code to live in a separate crate (#1752).

use ny_core::LayerType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a loaded neural network in ny's internal format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    /// Name of the network.
    pub name: String,
    /// Input specifications.
    pub inputs: Vec<TensorSpec>,
    /// Output specifications.
    pub outputs: Vec<TensorSpec>,
    /// Layers in topological order.
    pub layers: Vec<LayerSpec>,
    /// Total parameter count.
    pub param_count: usize,
}

/// Specification of a tensor (input/output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<i64>,
    pub dtype: DataType,
}

/// Supported data types.
///
/// `Default` is [`DataType::Float32`], NY's idealized verification precision.
/// This lets newer fields (e.g. [`WeightRef::original_dtype`]) use
/// `#[serde(default)]` so older serialized models that predate the field
/// continue to deserialize unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DataType {
    #[default]
    Float32,
    Float16,
    Int64,
    Int32,
}

/// A layer specification in the network (before conversion to propagate types).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSpec {
    pub name: String,
    pub layer_type: LayerType,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub weights: Option<WeightRef>,
    pub attributes: HashMap<String, AttributeValue>,
}

/// Returns true for ONNX Split nodes after loader normalization.
///
/// The ONNX loader represents Split as `LayerType::Slice` with multiple outputs;
/// ordinary Slice has a single output. Multi-output Split needs graph lowering so
/// each named output remains connected to the original input tensor.
pub fn is_multi_output_split(spec: &LayerSpec) -> bool {
    spec.layer_type == LayerType::Slice && spec.outputs.len() > 1
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    Float(f32),
    Int(i64),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
}

/// Resolve an unresolved tensor dimension to a concrete size.
///
/// `TensorSpec.shape` stores dimensions after ny-onnx loader normalization:
/// symbolic/missing ONNX dimensions become `-1`, and some upstream paths may
/// preserve `0`-valued dimensions. Both are treated as unresolved here and
/// mapped to the provided `default` value.
pub fn resolve_dynamic_dim(dim: i64, default: usize) -> usize {
    if dim <= 0 {
        default
    } else {
        dim as usize
    }
}

/// Resolve all dynamic dimensions in an ONNX shape to concrete sizes.
///
/// Convenience wrapper over [`resolve_dynamic_dim`] for full shape vectors.
pub fn resolve_dynamic_shape(shape: &[i64], default: usize) -> Vec<usize> {
    shape
        .iter()
        .map(|&d| resolve_dynamic_dim(d, default))
        .collect()
}

/// Reference to weights in the weight store.
///
/// `original_dtype` records the precision the weights were authored in before
/// NY loaded them as f32. It is ADDITIVE and OPT-IN: it defaults to
/// [`DataType::Float32`] (via `#[serde(default)]`), so existing callers that
/// build `WeightRef` without naming the field, and older serialized forms that
/// lack it entirely, are unaffected and stay on today's f32 path. Mixed-precision
/// verification (P8) reads this tag to decide whether a stored f32 weight must be
/// widened to its deployed precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightRef {
    pub name: String,
    pub shape: Vec<usize>,
    /// Precision the weight was authored in; defaults to [`DataType::Float32`].
    #[serde(default)]
    pub original_dtype: DataType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_type_default_is_float32() {
        assert_eq!(DataType::default(), DataType::Float32);
    }

    #[test]
    fn weight_ref_old_serialized_form_deserializes_to_f32() {
        // BACK-COMPAT: a serialized WeightRef from before `original_dtype`
        // existed has no such field. `#[serde(default)]` must fill it with the
        // f32 idealization so old models keep loading unchanged.
        let old_json = r#"{"name":"w","shape":[4,8]}"#;
        let parsed: WeightRef = serde_json::from_str(old_json).expect("old form must deserialize");
        assert_eq!(parsed.name, "w");
        assert_eq!(parsed.shape, vec![4, 8]);
        assert_eq!(
            parsed.original_dtype,
            DataType::Float32,
            "missing original_dtype must default to Float32"
        );
    }

    #[test]
    fn weight_ref_new_form_round_trips() {
        let original = WeightRef {
            name: "w".to_string(),
            shape: vec![2, 3],
            original_dtype: DataType::Float16,
        };
        let json = serde_json::to_string(&original).expect("serialize");
        let back: WeightRef = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.name, original.name);
        assert_eq!(back.shape, original.shape);
        assert_eq!(back.original_dtype, DataType::Float16);
    }

    #[test]
    fn weight_ref_preserves_non_f32_dtype_through_serde() {
        for dt in [
            DataType::Float32,
            DataType::Float16,
            DataType::Int32,
            DataType::Int64,
        ] {
            let wr = WeightRef {
                name: "t".to_string(),
                shape: vec![1],
                original_dtype: dt,
            };
            let json = serde_json::to_string(&wr).expect("serialize");
            let back: WeightRef = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back.original_dtype, dt);
        }
    }
}
