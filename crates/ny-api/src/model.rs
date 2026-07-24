// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Owned graph-build contract for external model producers.
//!
//! This surface defines the graph format ny expects from non-ONNX
//! producers. Tracers or translators can populate [`GraphModel`] directly or
//! via [`GraphModelBuilder`], then call [`GraphModel::build_graph_network`]
//! without depending on parser-specific `ny-onnx` internals. External
//! traced producers should target this surface; direct `ny_propagate` graph
//! construction is not the ny-owned cross-repo contract.
//!
//! External producers are expected to provide:
//! - a [`NetworkSpec`] with [`LayerSpec`] entries in topological order
//! - frozen auxiliary inputs via `GraphModelBuilder::frozen_input(...)` when a
//!   multi-input traced model needs exactly one bounded activation input
//! - `tensor_producer` links for structural tensors that trace back to an
//!   activation-producing tensor
//! - `constant_tensors` for tensors that do not depend on runtime inputs
//! - `tensor_shapes` for conversion-time shape reasoning through structural ops;
//!   frozen auxiliary inputs keep their original declared shape here even though
//!   their stored weight value is already unbatched
//!
//! The curated public network-spec name in this facade is [`NetworkSpec`].

pub use ny_build::Network as NetworkSpec;
pub use ny_build::{
    AttributeValue, CompoundNodePolicy, DataType, GraphModel, GraphModelBuilder,
    GraphNetworkOptions, LayerSpec, MissingOutputPolicy, TensorSpec, WeightRef, WeightStore,
};
pub use ny_core::LayerType;
