// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! DAG-based graph network builder.
//!
//! Converts model specification data into a [`GraphNetwork`](ny_propagate::GraphNetwork)
//! for bound propagation on models with branching/merging patterns (attention,
//! skip connections, etc.).

mod builder;
mod compound_nodes;
mod helpers;
mod inputs;
mod multi_input;
mod norm_decompose;
mod normalization_fusion;
mod outputs;

pub use builder::{build_graph_network, build_graph_network_with_linear_storage, GraphBuildInputs};
pub use multi_input::{concat_input_layout, ConcatInputLayout, ConcatInputSlot};

pub(super) const INPUT_NODE_NAME: &str = "_input";
