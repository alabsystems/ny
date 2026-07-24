// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core ONNX model structures and conversion helpers.

pub(crate) mod convert;
// Keep graph in a module directory (graph/mod.rs). Avoid reintroducing graph.rs.
#[path = "graph/mod.rs"]
mod graph;
mod options;
mod propagate;
mod types;
mod weights;

/// Options controlling graph conversion and propagation behavior.
pub use options::{
    CompoundNodePolicy, GraphNetworkOptions, MissingOutputPolicy, PropagateNetworkOptions,
};
/// Core ONNX graph, layer, tensor, and attribute model types.
pub use types::{
    is_multi_output_split, resolve_dynamic_dim, resolve_dynamic_shape, AttributeValue, DataType,
    LayerSpec, Network, OnnxModel, TensorSpec, WeightRef,
};
pub(crate) use types::{OriginalFloat32Initializer, OriginalOnnxNetwork};
/// Model weight storage container keyed by tensor name.
pub use weights::WeightStore;
