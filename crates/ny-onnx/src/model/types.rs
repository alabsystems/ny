// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Re-exports model specification types from `ny-build` and defines
//! `OnnxModel` which adds parsing-specific fields.

use std::collections::{HashMap, HashSet};

use ndarray::ArrayD;
use ny_build::WeightRevision;
use sha2::{Digest, Sha256};

pub use ny_build::{
    is_multi_output_split, resolve_dynamic_dim, resolve_dynamic_shape, AttributeValue, DataType,
    LayerSpec, Network, TensorSpec, WeightRef, WeightStore,
};

/// Loaded ONNX model with weights and graph structure.
#[derive(Debug)]
pub struct OnnxModel {
    /// Network specification (graph structure).
    pub network: Network,
    /// Weight storage.
    pub weights: WeightStore,
    /// Maps each tensor name to its producer tensor (first input of producing op).
    /// Used for tracing through intermediate ops like Cast, Transpose, Reshape.
    pub(crate) tensor_producer: HashMap<String, String>,
    /// Set of tensor names that are outputs of constant-producing ops (ConstantOfShape, Shape, etc.).
    /// These tensors contain values that don't depend on activation inputs, even if we can't
    /// evaluate them statically (e.g., because shape depends on dynamic batch size).
    /// Used to correctly handle nodes that consume these tensors.
    pub(crate) constant_tensors: HashSet<String>,
    /// Known tensor shapes keyed by tensor name (input/output/value_info/weights).
    pub(crate) tensor_shapes: HashMap<String, Vec<i64>>,
    /// Immutable evidence captured only while parsing raw ONNX FLOAT
    /// initializers, before constant folding or graph fusion can rewrite them.
    pub(crate) original_float32_initializers: HashMap<String, OriginalFloat32Initializer>,
    /// Exact finalized network representation captured only by the qualified
    /// provenance loader path. This seals public `network` mutations.
    pub(crate) original_network_topology: Option<OriginalOnnxNetwork>,
    /// Model opset imports keyed by domain.
    /// Use an empty map for non-ONNX sources (e.g., native/safetensors loads),
    /// otherwise preserve what the ONNX graph declares.
    pub(crate) opset_imports: HashMap<String, i64>,
}

impl OnnxModel {
    /// Read-only access to tensor producer map (tensor name → producer tensor name).
    pub fn tensor_producer(&self) -> &HashMap<String, String> {
        &self.tensor_producer
    }

    /// Read-only access to the set of constant tensor names.
    pub fn constant_tensors(&self) -> &HashSet<String> {
        &self.constant_tensors
    }

    /// Read-only access to known tensor shapes.
    pub fn tensor_shapes(&self) -> &HashMap<String, Vec<i64>> {
        &self.tensor_shapes
    }

    /// Read-only access to opset imports.
    pub fn opset_imports(&self) -> &HashMap<String, i64> {
        &self.opset_imports
    }

    /// Whether the current weight named `name` still exactly matches a raw
    /// ONNX FLOAT initializer captured by this model's loader.
    ///
    /// `None` means there was no such initializer (for example, a synthetic
    /// store, Constant node, or constant-fold result). `Some(false)` means the
    /// original existed but its store, revision, shape, or exact float bits no
    /// longer match. Loader provenance itself is private and cannot be created
    /// or reset by callers.
    #[must_use]
    pub fn original_float32_initializer_matches_current(&self, name: &str) -> Option<bool> {
        let original = self.original_float32_initializers.get(name)?;
        Some(original.matches_current(name, &self.weights))
    }

    /// Whether the current public network still exactly matches the finalized
    /// representation produced by an opt-in qualified ONNX load.
    ///
    /// `None` means topology provenance was not requested. Callers cannot
    /// create or reset the private snapshot.
    #[must_use]
    pub fn original_network_topology_matches_current(&self) -> Option<bool> {
        self.original_network_topology
            .as_ref()
            .map(|original| original.matches_current(&self.network))
    }

    /// Extract a format-neutral [`GraphModel`](ny_build::GraphModel) from this ONNX model.
    ///
    /// `GraphModel` is the format-neutral contract in `ny-build` that
    /// both ONNX-loaded models and future traced producers converge on.
    /// This conversion drops ONNX-specific metadata
    /// (`opset_imports`) that is only needed during parsing.
    ///
    /// The resulting `GraphModel` produces an identical `GraphNetwork` when
    /// built with the same options as `self.to_graph_network_with_options()`.
    pub fn to_graph_model(self) -> ny_build::GraphModel {
        ny_build::GraphModel {
            network: self.network,
            weights: self.weights,
            tensor_producer: self.tensor_producer,
            constant_tensors: self.constant_tensors,
            tensor_shapes: self.tensor_shapes,
            // ADDITIVE: ONNX-loaded models default to the f32 idealization.
            mixed_precision: None,
        }
    }

    /// Construct a minimal model for non-ONNX sources (native/safetensors).
    pub fn empty_with_network(network: Network, weights: WeightStore) -> Self {
        Self {
            network,
            weights,
            tensor_producer: HashMap::new(),
            constant_tensors: HashSet::new(),
            tensor_shapes: HashMap::new(),
            original_float32_initializers: HashMap::new(),
            original_network_topology: None,
            opset_imports: HashMap::new(),
        }
    }

    /// Set known tensor shapes. Builder-style: returns `self` for chaining.
    pub fn with_tensor_shapes(mut self, shapes: HashMap<String, Vec<i64>>) -> Self {
        self.tensor_shapes = shapes;
        self
    }

    /// Freeze named activation inputs as concrete weight tensors.
    ///
    /// For multi-input models (e.g. talker attention with cos/sin/mask, or
    /// kokoro vocoder with style/har), this converts auxiliary inputs into
    /// constant weights so the model has a single bounded activation input
    /// for propagation.
    ///
    /// Each `(name, tensor)` pair:
    /// 1. Inserts the tensor into `self.weights`
    /// 2. Marks the name as a constant tensor
    /// 3. Removes it from `self.network.inputs`
    ///
    /// Panics if any named input does not exist in `self.network.inputs`.
    pub fn freeze_inputs(&mut self, inputs: impl IntoIterator<Item = (String, ArrayD<f32>)>) {
        let mut frozen_names = Vec::new();
        for (name, tensor) in inputs {
            assert!(
                self.network.inputs.iter().any(|s| s.name == name),
                "freeze_inputs: '{}' not found in model inputs {:?}",
                name,
                self.network
                    .inputs
                    .iter()
                    .map(|s| &s.name)
                    .collect::<Vec<_>>()
            );
            self.weights.insert(name.clone(), tensor);
            self.constant_tensors.insert(name.clone());
            frozen_names.push(name);
        }
        self.network
            .inputs
            .retain(|spec| !frozen_names.contains(&spec.name));
    }
}

/// Loader-private fingerprint for a raw ONNX FLOAT initializer.
#[derive(Debug)]
pub(crate) struct OriginalFloat32Initializer {
    shape: Vec<usize>,
    bit_digest: [u8; 32],
    weight_revision: WeightRevision,
}

impl OriginalFloat32Initializer {
    pub(crate) fn from_tensor(tensor: &ArrayD<f32>, weight_revision: WeightRevision) -> Self {
        Self {
            shape: tensor.shape().to_vec(),
            bit_digest: tensor_bit_digest(tensor),
            weight_revision,
        }
    }

    fn matches_current(&self, name: &str, weights: &WeightStore) -> bool {
        let Some(current) = weights.get(name) else {
            return false;
        };
        weights.matches_revision(name, &self.weight_revision)
            && current.shape() == self.shape
            && tensor_bit_digest(current) == self.bit_digest
    }
}

fn tensor_bit_digest(tensor: &ArrayD<f32>) -> [u8; 32] {
    let mut digest = Sha256::new();
    for &value in tensor {
        digest.update(value.to_bits().to_le_bytes());
    }
    digest.finalize().into()
}

/// Loader-private exact snapshot of the finalized network representation.
#[derive(Debug)]
pub(crate) struct OriginalOnnxNetwork {
    network: Network,
}

impl OriginalOnnxNetwork {
    pub(crate) fn from_network(network: &Network) -> Self {
        Self {
            network: network.clone(),
        }
    }

    fn matches_current(&self, current: &Network) -> bool {
        network_matches_exactly(&self.network, current)
    }
}

fn network_matches_exactly(original: &Network, current: &Network) -> bool {
    original.name == current.name
        && original.param_count == current.param_count
        && tensor_specs_match_exactly(&original.inputs, &current.inputs)
        && tensor_specs_match_exactly(&original.outputs, &current.outputs)
        && original.layers.len() == current.layers.len()
        && original
            .layers
            .iter()
            .zip(&current.layers)
            .all(|(original, current)| layer_specs_match_exactly(original, current))
}

fn tensor_specs_match_exactly(original: &[TensorSpec], current: &[TensorSpec]) -> bool {
    original.len() == current.len()
        && original.iter().zip(current).all(|(original, current)| {
            original.name == current.name
                && original.shape == current.shape
                && original.dtype == current.dtype
        })
}

fn layer_specs_match_exactly(original: &LayerSpec, current: &LayerSpec) -> bool {
    original.name == current.name
        && original.layer_type == current.layer_type
        && original.inputs == current.inputs
        && original.outputs == current.outputs
        && attributes_match_exactly(&original.attributes, &current.attributes)
        && weight_refs_match_exactly(original.weights.as_ref(), current.weights.as_ref())
}

fn attributes_match_exactly(
    original: &HashMap<String, AttributeValue>,
    current: &HashMap<String, AttributeValue>,
) -> bool {
    // Drive lookups from the current/public map. The proof bridge budgets all
    // of its key bytes and value elements before reaching this comparison. If
    // a caller replaces a huge original key/value with a short one, hashing
    // and payload comparison therefore remain bounded and fail on key/value
    // length without scanning immutable attacker-sized bytes.
    original.len() == current.len()
        && current.iter().all(|(name, value)| {
            original
                .get(name)
                .is_some_and(|original| attribute_values_match_bits(original, value))
        })
}

fn attribute_values_match_bits(original: &AttributeValue, current: &AttributeValue) -> bool {
    match (original, current) {
        (AttributeValue::Float(original), AttributeValue::Float(current)) => {
            original.to_bits() == current.to_bits()
        }
        (AttributeValue::Int(original), AttributeValue::Int(current)) => original == current,
        (AttributeValue::String(original), AttributeValue::String(current)) => original == current,
        (AttributeValue::Floats(original), AttributeValue::Floats(current)) => {
            original.len() == current.len()
                && original
                    .iter()
                    .zip(current)
                    .all(|(original, current)| original.to_bits() == current.to_bits())
        }
        (AttributeValue::Ints(original), AttributeValue::Ints(current)) => original == current,
        _ => false,
    }
}

fn weight_refs_match_exactly(original: Option<&WeightRef>, current: Option<&WeightRef>) -> bool {
    match (original, current) {
        (None, None) => true,
        (Some(original), Some(current)) => {
            original.name == current.name
                && original.shape == current.shape
                && original.original_dtype == current.original_dtype
        }
        _ => false,
    }
}

#[cfg(test)]
mod provenance_tests {
    use super::*;

    #[test]
    fn float_attributes_match_exact_bits() {
        assert!(!attribute_values_match_bits(
            &AttributeValue::Float(-0.0),
            &AttributeValue::Float(0.0),
        ));

        let nan = f32::from_bits(0x7fc0_1234);
        assert!(attribute_values_match_bits(
            &AttributeValue::Float(nan),
            &AttributeValue::Float(f32::from_bits(nan.to_bits())),
        ));
        assert!(!attribute_values_match_bits(
            &AttributeValue::Floats(vec![nan, -0.0]),
            &AttributeValue::Floats(vec![nan, 0.0]),
        ));
    }
}
