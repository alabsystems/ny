// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Thin delegate to `ny_build::graph` — graph construction logic lives
//! in `ny-build` as of #1752.

use ny_build::GraphBuildInputs;
use ny_core::Result;
use ny_propagate::GraphNetwork;

use super::{GraphNetworkOptions, OnnxModel};

impl OnnxModel {
    /// Convert to a GraphNetwork for DAG-based bound propagation.
    ///
    /// Unlike `to_propagate_network()` which creates a sequential network,
    /// this builds a proper directed acyclic graph (DAG) that can handle
    /// binary operations like attention MatMul (Q@K^T) where both inputs
    /// are bounded tensors.
    ///
    /// Use this for models with attention or other branching/merging patterns.
    ///
    /// # Example
    /// ```rust,no_run
    /// use ny_onnx::load_onnx;
    /// let model = load_onnx("attention_model.onnx").unwrap();
    /// let graph = model.to_graph_network().unwrap();
    /// // let output_bounds = graph.propagate_ibp(&input_bounds).unwrap();
    /// ```
    pub fn to_graph_network(&self) -> Result<GraphNetwork> {
        self.to_graph_network_with_options(GraphNetworkOptions::default())
    }

    /// Convert to a GraphNetwork with explicit conversion options.
    ///
    /// By default this returns an error if a Reshape has a dynamic (non-constant) shape.
    /// Set `allow_dynamic_reshape` to true to explicitly skip such Reshape ops.
    pub fn to_graph_network_with_options(
        &self,
        options: GraphNetworkOptions,
    ) -> Result<GraphNetwork> {
        let data = GraphBuildInputs {
            layers: &self.network.layers,
            inputs: &self.network.inputs,
            outputs: &self.network.outputs,
            weights: &self.weights,
            tensor_producer: &self.tensor_producer,
            constant_tensors: &self.constant_tensors,
            tensor_shapes: &self.tensor_shapes,
        };
        ny_build::build_graph_network(&data, options)
    }
}

#[cfg(test)]
mod tests;
