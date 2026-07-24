// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

#![forbid(unsafe_code)]

//! ONNX model types and layer conversion for ny.
//!
//! This crate owns the model specification types ([`Network`], [`LayerSpec`],
//! [`WeightStore`]) and the conversion dispatch ([`ConvertContext`]) that turns
//! layer specs into bound-propagation [`Layer`](ny_propagate::Layer) values.
//!
//! Part of the `ny-onnx` split (#1752): parsing stays in `ny-onnx`,
//! model types and construction logic live here.

// Link macOS Accelerate BLAS for ndarray::dot() acceleration (#4259).
#[cfg(target_os = "macos")]
extern crate blas_src;

pub mod convert;
/// SOUND dequantization to f32 intervals for quantized weights (P8).
pub mod dequant_interval;
pub mod graph;
mod graph_model;
mod graph_model_builder;
mod graph_options;
mod layernorm_mode;
/// Mixed-precision policy for verifying models at their deployed precision (P8).
mod mixed_precision;
mod model_types;
mod propagate;
mod weight_store;

// Re-export model types at crate root for ergonomic imports.
pub use convert::{model_is_unbatched, ConvertContext};
pub use dequant_interval::{dequant_block_affine_interval, dequant_interval};
pub use graph::{build_graph_network, GraphBuildInputs};
pub use graph_model::GraphModel;
pub use graph_model_builder::GraphModelBuilder;
pub use graph_options::{CompoundNodePolicy, GraphNetworkOptions, MissingOutputPolicy};
pub(crate) use layernorm_mode::layernorm_mode_from_attrs;
pub use mixed_precision::MixedPrecisionPolicy;
pub use model_types::{
    is_multi_output_split, resolve_dynamic_dim, resolve_dynamic_shape, AttributeValue, DataType,
    LayerSpec, Network, TensorSpec, WeightRef,
};
pub use propagate::{
    build_propagate_network, build_propagate_network_indexed, PropagateNetworkOptions,
};
pub use weight_store::{WeightRevision, WeightStore};
