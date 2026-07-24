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
mod norm_decompose;
mod normalization_fusion;
mod outputs;

pub use builder::{build_graph_network, GraphBuildInputs};

pub(super) const INPUT_NODE_NAME: &str = "_input";
