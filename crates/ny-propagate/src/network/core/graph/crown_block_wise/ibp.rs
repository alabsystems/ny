// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP (Interval Bound Propagation) forward pass within a transformer block.
//!
//! Collects per-node bounds by running IBP forward through block nodes. Used
//! to provide intermediate bounds for the subsequent CROWN backward pass.
//!
//! Part of #3221, #3447.

use std::collections::{HashMap, HashSet};

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::layers::Layer;
use crate::network::core::graph::node::NETWORK_INPUT;
use crate::network::core::graph::GraphNetwork;

impl GraphNetwork {
    /// Run IBP forward through a block's nodes, collecting bounds at each node.
    ///
    /// Handles unary layers (most layers), binary Add (residual connections),
    /// and binary MatMul/MulBinary. For nodes whose inputs are outside the block,
    /// uses `block_input` as the input bounds.
    pub(crate) fn collect_block_ibp_bounds(
        &self,
        block_nodes: &[String],
        block_input: &BoundedTensor,
    ) -> Result<HashMap<String, BoundedTensor>> {
        let block_set: HashSet<&str> = block_nodes.iter().map(|s| s.as_str()).collect();
        let mut bounds_cache: HashMap<String, BoundedTensor> =
            HashMap::with_capacity(block_nodes.len());

        for node_name in block_nodes {
            let node = self
                .nodes
                .get(node_name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {}", node_name)))?;

            let output_bounds = match &node.layer {
                // Binary Add: get bounds for both inputs and add.
                Layer::Add(add) => {
                    let input_a = self.resolve_block_node_input(
                        &node.inputs,
                        0,
                        &block_set,
                        &bounds_cache,
                        block_input,
                    )?;
                    let input_b = self.resolve_block_node_input(
                        &node.inputs,
                        1,
                        &block_set,
                        &bounds_cache,
                        block_input,
                    )?;
                    add.propagate_ibp_binary(&input_a, &input_b)?
                }
                // Binary MatMul: get bounds for both inputs.
                Layer::MatMul(matmul) => {
                    let input_a = self.resolve_block_node_input(
                        &node.inputs,
                        0,
                        &block_set,
                        &bounds_cache,
                        block_input,
                    )?;
                    let input_b = self.resolve_block_node_input(
                        &node.inputs,
                        1,
                        &block_set,
                        &bounds_cache,
                        block_input,
                    )?;
                    matmul.propagate_ibp_binary(&input_a, &input_b)?
                }
                // Binary MulBinary (element-wise multiply).
                Layer::MulBinary(mul) => {
                    let input_a = self.resolve_block_node_input(
                        &node.inputs,
                        0,
                        &block_set,
                        &bounds_cache,
                        block_input,
                    )?;
                    let input_b = self.resolve_block_node_input(
                        &node.inputs,
                        1,
                        &block_set,
                        &bounds_cache,
                        block_input,
                    )?;
                    mul.propagate_ibp_binary(&input_a, &input_b)?
                }
                // Unary layers: standard IBP through BoundPropagation trait.
                _ => {
                    use crate::layers::common::BoundPropagation;
                    let input_bounds = self.resolve_block_node_input(
                        &node.inputs,
                        0,
                        &block_set,
                        &bounds_cache,
                        block_input,
                    )?;
                    node.layer.propagate_ibp(&input_bounds)?
                }
            };

            bounds_cache.insert(node_name.clone(), output_bounds);
        }

        Ok(bounds_cache)
    }

    /// Resolve the bounds for a node's Nth input within a block context.
    ///
    /// If the input node is within the block, returns its cached bounds.
    /// If outside the block (or is NETWORK_INPUT), returns block_input.
    fn resolve_block_node_input(
        &self,
        inputs: &[String],
        index: usize,
        block_set: &HashSet<&str>,
        bounds_cache: &HashMap<String, BoundedTensor>,
        block_input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let input_name = inputs.get(index).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Node has {} inputs, expected at least {}",
                inputs.len(),
                index + 1
            ))
        })?;

        if input_name == NETWORK_INPUT {
            Ok(block_input.clone())
        } else if block_set.contains(input_name.as_str()) {
            bounds_cache.get(input_name).cloned().ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Block IBP: bounds not yet computed for in-block node '{}'",
                    input_name
                ))
            })
        } else {
            // Outside block → use block input bounds.
            Ok(block_input.clone())
        }
    }
}
