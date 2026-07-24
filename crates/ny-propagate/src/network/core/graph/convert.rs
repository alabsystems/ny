// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conversion and chain-extraction utilities for GraphNetwork.
//!
//! Contains:
//! - [`GraphNetwork::from_sequential`]: Convert sequential `Network` → `GraphNetwork`
//! - [`GraphNetwork::is_sequential_graph`]: Detect linear chain topology
//! - [`GraphNetwork::try_unary_chain`]: Extract a unary-layer chain as `GraphUnaryChain`

use std::collections::HashMap;

use ny_core::Result;

use super::{GraphNetwork, GraphNode, NETWORK_INPUT};
use crate::network::Network;

/// A sequential chain extracted from a `GraphNetwork` that contains only unary layers.
///
/// Produced by [`GraphNetwork::try_unary_chain`]. Consumers can apply further
/// policy filters (e.g., SDP-CROWN accepts only Linear/ReLU) on
/// `network.layers` before using the lowered chain.
#[derive(Debug, Clone)]
pub(crate) struct GraphUnaryChain {
    /// The lowered sequential network (layers in execution order).
    pub(crate) network: Network,
    /// Node names in the same order as `network.layers`, for provenance mapping.
    pub(crate) node_names: Vec<String>,
}

impl GraphNetwork {
    /// Convert a sequential Network to a GraphNetwork.
    ///
    /// Creates a linear chain of nodes: input -> layer0 -> layer1 -> ... -> output
    /// For empty networks, returns an empty graph with output node [`NETWORK_INPUT`].
    ///
    /// Returns `Err` if duplicate node names are detected (#2686).
    pub fn from_sequential(network: &Network) -> Result<Self> {
        let mut graph = GraphNetwork::new();

        if network.layers.is_empty() {
            // Empty network: output is the input itself
            graph.set_output(NETWORK_INPUT);
            return Ok(graph);
        }

        for (i, layer) in network.layers.iter().enumerate() {
            let name = format!("layer_{}", i);
            let input_name = if i == 0 {
                NETWORK_INPUT.to_string()
            } else {
                format!("layer_{}", i - 1)
            };

            let node = GraphNode::new(name.clone(), layer.clone(), vec![input_name]);
            graph.try_add_node(node)?;

            if i == network.layers.len() - 1 {
                graph.set_output(name);
            }
        }

        Ok(graph)
    }

    /// Check if the graph is essentially sequential (a linear chain).
    ///
    /// A sequential graph has no binary ops and no nodes with multiple consumers.
    /// Moved from `graph_alpha/propagate_helpers.rs` to share with non-alpha
    /// consumers (e.g., resident PGD in #4081).
    pub(crate) fn is_sequential_graph(&self, exec_order: &[String]) -> bool {
        if exec_order.is_empty() {
            return true;
        }

        // Count how many times each node is used as input (consumer count)
        let mut consumer_count: HashMap<String, usize> = HashMap::new();
        consumer_count.insert(NETWORK_INPUT.to_string(), 0);

        for name in exec_order {
            if let Some(node) = self.nodes.get(name) {
                // Check for binary ops
                if node.layer.is_binary() {
                    return false;
                }
                // Count inputs
                for input_name in &node.inputs {
                    *consumer_count.entry(input_name.clone()).or_insert(0) += 1;
                }
            }
        }

        // Check that no node has more than one consumer (except _input which can have one)
        for (name, count) in &consumer_count {
            if name == NETWORK_INPUT {
                if *count > 1 {
                    return false; // Input used by multiple nodes
                }
            } else if *count > 1 {
                return false; // Intermediate node used by multiple nodes (branching)
            }
        }

        true
    }

    /// Extract a unary-layer chain from the graph, if the topology is sequential.
    ///
    /// Returns `Ok(Some(GraphUnaryChain))` when every node in `exec_order` is
    /// unary (single input) and forms a strict chain from `NETWORK_INPUT`.
    /// Returns `Ok(None)` for non-chain graphs (branches, binary ops, gaps).
    /// Returns `Err` only for invariant violations (missing node, malformed unary arity).
    ///
    /// This is the shared lowering seam for SDP-CROWN, CROWN-IBP sequential
    /// fast path, and future resident PGD (#4081 Slice 2).
    pub(crate) fn try_unary_chain(&self, exec_order: &[String]) -> Result<Option<GraphUnaryChain>> {
        if !self.is_sequential_graph(exec_order) {
            return Ok(None);
        }

        let mut network = Network::new();
        let mut node_names = Vec::with_capacity(exec_order.len());
        let mut expected_input = NETWORK_INPUT;

        for node_name in exec_order {
            let node = self.nodes.get(node_name).ok_or_else(|| {
                ny_core::NyError::InvalidSpec(format!(
                    "try_unary_chain: missing node '{node_name}' in graph"
                ))
            })?;
            let input_name = node.require_unary_input()?;
            if input_name != expected_input {
                return Ok(None);
            }
            network.add_layer(node.layer.clone());
            node_names.push(node_name.clone());
            expected_input = node_name;
        }

        Ok(Some(GraphUnaryChain {
            network,
            node_names,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::binary_ops::AddLayer;
    use crate::layers::transform::FlattenLayer;
    use crate::layers::{Layer, ReLULayer};

    /// #4097: A valid unary chain with non-SDP layers (ReLU + Flatten) returns
    /// `Some(GraphUnaryChain)` with correct layer count and node name order.
    #[test]
    fn graph_unary_chain_extracts_generic_layers_and_names_4097() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "flatten",
            Layer::Flatten(FlattenLayer::new(1)),
            vec!["relu".to_string()],
        ));
        graph.set_output("flatten");

        let exec_order = graph.exec_order().unwrap();
        let chain = graph
            .try_unary_chain(exec_order)
            .expect("should not error on valid graph")
            .expect("valid unary chain should return Some");

        assert_eq!(chain.network.layers.len(), 2);
        assert_eq!(chain.node_names, vec!["relu", "flatten"]);
    }

    /// #4097: A branching graph (binary Add with two inputs) returns `None`.
    #[test]
    fn graph_unary_chain_rejects_branch_4097() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::from_input("relu2", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::new(
            "add",
            Layer::Add(AddLayer),
            vec!["relu1".to_string(), "relu2".to_string()],
        ));
        graph.set_output("add");

        let exec_order = graph.exec_order().unwrap();
        let result = graph.try_unary_chain(exec_order).expect("should not error");

        assert!(result.is_none(), "branching graph should return None");
    }

    /// #4097: malformed multi-input unary nodes must surface `InvalidSpec`
    /// instead of being silently lowered as a one-input chain.
    #[test]
    fn graph_unary_chain_rejects_multi_input_unary_node_4097() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::new(
            "relu_malformed",
            Layer::ReLU(ReLULayer),
            vec![NETWORK_INPUT.to_string(), "ghost".to_string()],
        ));
        graph.set_output("relu_malformed");

        let error = graph
            .try_unary_chain(&["relu_malformed".to_string()])
            .expect_err("malformed unary node should return InvalidSpec");
        assert!(
            matches!(
                &error,
                ny_core::NyError::InvalidSpec(msg) if msg.contains("exactly 1 input")
            ),
            "unexpected error: {error:?}"
        );
    }
}
