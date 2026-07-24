// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Builder for constructing attention-like graph patterns.
//!
//! Part of #145 - extracted from network/core.rs.

use crate::layers::{
    AddLayer, BilinearCrownLayer, GELULayer, Layer, LinearLayer, ReLULayer, SoftmaxLayer,
};
use ndarray::{Array1, Array2};
use ny_core::Result;

use super::core::{GraphNetwork, GraphNode};

/// Builder for constructing attention-like graph patterns.
///
/// Provides convenient methods for building common attention patterns
/// without manually creating each node.
#[derive(Debug)]
pub(crate) struct AttentionGraphBuilder {
    graph: GraphNetwork,
    node_counter: usize,
}

impl AttentionGraphBuilder {
    /// Create a new attention graph builder.
    pub(crate) fn new() -> Self {
        Self {
            graph: GraphNetwork::new(),
            node_counter: 0,
        }
    }

    /// Generate a unique node name that doesn't collide with existing nodes.
    ///
    /// Starts from the current counter and increments until finding an unused name.
    /// This prevents collisions when user-provided names overlap with auto-generated ones.
    fn next_name(&mut self, prefix: &str) -> String {
        loop {
            let name = format!("{}_{}", prefix, self.node_counter);
            self.node_counter += 1;
            if !self.graph.contains_node(&name) {
                return name;
            }
            // Name exists, try next counter value
        }
    }

    /// Add an input node that reads from the graph's input.
    ///
    /// This creates a node that takes its input from the special "_input" source,
    /// which represents the graph's external input.
    ///
    /// # Arguments
    /// * `name` - Name for the input node
    /// * `layer` - Layer to apply to the graph input
    ///
    /// # Returns
    /// The name of the created node (same as input name), or an error if the name collides.
    ///
    /// # Errors
    /// Returns an error if a node with the given name already exists in the graph.
    pub(crate) fn add_input(&mut self, name: impl Into<String>, layer: Layer) -> Result<String> {
        let name = name.into();
        self.graph
            .try_add_node(GraphNode::from_input(&name, layer))?;
        Ok(name)
    }

    /// Add a linear projection from network input.
    ///
    /// # Errors
    /// Returns an error if the layer creation fails or if the name collides with an existing node.
    pub(crate) fn add_projection(
        &mut self,
        name: impl Into<String>,
        weight: Array2<f32>,
        bias: Option<Array1<f32>>,
    ) -> Result<String> {
        let name = name.into();
        let layer = Layer::Linear(LinearLayer::new(weight, bias)?);
        self.graph
            .try_add_node(GraphNode::from_input(&name, layer))?;
        Ok(name)
    }

    /// Add a bounded matrix multiplication (Q @ K^T or probs @ V pattern).
    ///
    /// Uses BilinearCrownLayer for all bilinear MatMuls. Broadcast McCormick (#286)
    /// handles any seq length without O(seq^4) memory.
    ///
    /// Returns the auto-generated node name, or an error on name collision. (#2686)
    pub(crate) fn add_matmul(
        &mut self,
        input_a: &str,
        input_b: &str,
        transpose_b: bool,
        scale: Option<f32>,
    ) -> Result<String> {
        let name = self.next_name("matmul");
        let layer = Layer::BilinearCrown(BilinearCrownLayer::new(transpose_b, scale));
        self.graph
            .try_add_node(GraphNode::binary(&name, layer, input_a, input_b))?;
        Ok(name)
    }

    /// Add a softmax operation.
    ///
    /// Returns the auto-generated node name, or an error on name collision. (#2686)
    pub(crate) fn add_softmax(&mut self, input: &str, axis: i32) -> Result<String> {
        let name = self.next_name("softmax");
        let layer = Layer::Softmax(SoftmaxLayer::new(axis));
        self.graph
            .try_add_node(GraphNode::new(&name, layer, vec![input.to_string()]))?;
        Ok(name)
    }

    /// Add element-wise addition (residual connection).
    ///
    /// Returns the auto-generated node name, or an error on name collision. (#2686)
    pub(crate) fn add_residual(&mut self, input_a: &str, input_b: &str) -> Result<String> {
        let name = self.next_name("add");
        let layer = Layer::Add(AddLayer);
        self.graph
            .try_add_node(GraphNode::binary(&name, layer, input_a, input_b))?;
        Ok(name)
    }

    /// Add a ReLU activation.
    ///
    /// Returns the auto-generated node name, or an error on name collision. (#2686)
    pub(crate) fn add_relu(&mut self, input: &str) -> Result<String> {
        let name = self.next_name("relu");
        let layer = Layer::ReLU(ReLULayer);
        self.graph
            .try_add_node(GraphNode::new(&name, layer, vec![input.to_string()]))?;
        Ok(name)
    }

    /// Add a GELU activation.
    ///
    /// Returns the auto-generated node name, or an error on name collision. (#2686)
    pub(crate) fn add_gelu(&mut self, input: &str) -> Result<String> {
        let name = self.next_name("gelu");
        let layer = Layer::GELU(GELULayer::default());
        self.graph
            .try_add_node(GraphNode::new(&name, layer, vec![input.to_string()]))?;
        Ok(name)
    }

    /// Set the output node and return the built graph.
    pub(crate) fn build(mut self, output: &str) -> GraphNetwork {
        self.graph.set_output(output);
        self.graph
    }

    /// Get reference to the graph being built.
    pub(crate) fn graph(&self) -> &GraphNetwork {
        &self.graph
    }
}

impl Default for AttentionGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ReLULayer;

    // Part of #170 - AttentionGraphBuilder unit tests moved from core.rs

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_new() {
        let builder = AttentionGraphBuilder::new();
        assert_eq!(builder.graph().num_nodes(), 0);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_default() {
        let builder = AttentionGraphBuilder::default();
        assert_eq!(builder.graph().num_nodes(), 0);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_add_input() {
        // Part of #171 - Test new add_input API
        let mut builder = AttentionGraphBuilder::new();
        let name = builder.add_input("input", Layer::ReLU(ReLULayer)).unwrap();
        assert_eq!(name, "input");
        assert!(builder.graph().node("input").is_some());
        assert_eq!(builder.graph().num_nodes(), 1);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_add_relu() {
        let mut builder = AttentionGraphBuilder::new();
        builder.add_input("input", Layer::ReLU(ReLULayer)).unwrap();

        let output_name = builder.add_relu("input").unwrap();
        assert!(output_name.contains("relu"));
        let node = builder
            .graph()
            .node(&output_name)
            .expect("relu node should exist in graph");
        assert!(
            matches!(node.layer(), Layer::ReLU(_)),
            "node should be ReLU layer"
        );
        assert_eq!(node.inputs(), &["input"]);
        assert_eq!(builder.graph().num_nodes(), 2);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_add_gelu() {
        let mut builder = AttentionGraphBuilder::new();
        builder.add_input("input", Layer::ReLU(ReLULayer)).unwrap();

        let output_name = builder.add_gelu("input").unwrap();
        assert!(output_name.contains("gelu"));
        let node = builder
            .graph()
            .node(&output_name)
            .expect("gelu node should exist in graph");
        assert!(
            matches!(node.layer(), Layer::GELU(_)),
            "node should be GELU layer"
        );
        assert_eq!(node.inputs(), &["input"]);
        assert_eq!(builder.graph().num_nodes(), 2);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_add_softmax() {
        let mut builder = AttentionGraphBuilder::new();
        builder.add_input("input", Layer::ReLU(ReLULayer)).unwrap();

        let output_name = builder.add_softmax("input", -1).unwrap();
        assert!(output_name.contains("softmax"));
        let node = builder
            .graph()
            .node(&output_name)
            .expect("softmax node should exist in graph");
        assert!(
            matches!(node.layer(), Layer::Softmax(_)),
            "node should be Softmax layer"
        );
        assert_eq!(node.inputs(), &["input"]);
        assert_eq!(builder.graph().num_nodes(), 2);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_add_residual() {
        let mut builder = AttentionGraphBuilder::new();
        builder.add_input("a", Layer::ReLU(ReLULayer)).unwrap();
        builder.add_input("b", Layer::ReLU(ReLULayer)).unwrap();

        let output_name = builder.add_residual("a", "b").unwrap();
        assert!(output_name.contains("add"));
        let node = builder
            .graph()
            .node(&output_name)
            .expect("residual node should exist in graph");
        assert!(
            matches!(node.layer(), Layer::Add(_)),
            "node should be Add layer"
        );
        assert_eq!(node.inputs(), &["a", "b"]);
        assert_eq!(builder.graph().num_nodes(), 3);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_build() {
        let mut builder = AttentionGraphBuilder::new();
        builder.add_input("input", Layer::ReLU(ReLULayer)).unwrap();
        let relu_out = builder.add_relu("input").unwrap();

        let graph = builder.build(&relu_out);
        assert_eq!(graph.output_name(), relu_out);
        assert_eq!(graph.num_nodes(), 2);
        assert!(
            graph.node("input").is_some(),
            "input node should exist in built graph"
        );
        assert!(
            graph.node(&relu_out).is_some(),
            "relu node should exist in built graph"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_attention_graph_builder_node_counter() {
        // Verify auto-generated names increment
        let mut builder = AttentionGraphBuilder::new();
        builder.add_input("input", Layer::ReLU(ReLULayer)).unwrap();

        let relu1 = builder.add_relu("input").unwrap();
        let relu2 = builder.add_relu(&relu1).unwrap();
        let relu3 = builder.add_relu(&relu2).unwrap();

        // Names should follow relu_0, relu_1, relu_2 pattern
        assert_eq!(relu1, "relu_0");
        assert_eq!(relu2, "relu_1");
        assert_eq!(relu3, "relu_2");
        assert_eq!(builder.graph().num_nodes(), 4); // input + 3 relus
    }

    // Part of #172 - Test collision detection

    #[ntest::timeout(10000)]
    #[test]
    fn test_add_input_collision_error() {
        let mut builder = AttentionGraphBuilder::new();
        builder.add_input("input", Layer::ReLU(ReLULayer)).unwrap();

        // Adding same name should error
        let result = builder.add_input("input", Layer::ReLU(ReLULayer));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_next_name_skips_collisions() {
        // Verify auto-names skip over user-defined names
        let mut builder = AttentionGraphBuilder::new();

        // Pre-create "relu_0" with user-provided name
        builder.add_input("relu_0", Layer::ReLU(ReLULayer)).unwrap();

        // Auto-generated relu should skip to relu_1
        let auto_name = builder.add_relu("relu_0").unwrap();
        assert_eq!(auto_name, "relu_1");
    }
}
