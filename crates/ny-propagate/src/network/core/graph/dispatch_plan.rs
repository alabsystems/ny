// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pre-compiled dispatch plan for the CROWN backward loop.
//!
//! Built once per [`GraphNetwork`], reused across all BaB iterations.
//! Replaces per-call HashMap lookups, toposort recomputation, and graph
//! property queries with O(1) Vec-indexed access.
//!
//! Reference pattern: DAG alpha-CROWN backward (`backward/mod.rs:186-263`)
//! already uses inline `name_to_idx` / `nodes_by_idx` / `bounds_by_idx` Vecs.
//! This struct extracts that pattern into a reusable artifact.
//!
//! Design: `designs/2026-03-21-issue-4258-crown-dispatch-plan-compiled.md` Phase 2.

use std::collections::HashMap;

use ny_core::{NyError, Result};

use super::{GraphNetwork, NETWORK_INPUT};
use crate::layers::Layer;

/// Pre-compiled dispatch metadata for the CROWN backward loop.
///
/// All node-level data is stored in `Vec`s indexed by a compact sequential
/// index assigned during construction. The `NETWORK_INPUT` sentinel gets the
/// last index (`node_count`).
///
/// # Lifecycle
///
/// Built once via [`GraphNetwork::dispatch_plan()`], then borrowed by every
/// backward pass. The plan is invalidated when the graph structure mutates
/// (same lifecycle as `cached_exec_order`).
#[derive(Debug, Clone)]
pub(crate) struct CrownDispatchPlan {
    /// Forward topological execution order as node indices.
    pub exec_order: Vec<usize>,
    /// Reverse topological order for backward pass (same as `exec_order` reversed).
    pub reverse_order: Vec<usize>,

    /// Map from node name to sequential index.
    pub name_to_idx: HashMap<String, usize>,
    /// Map from sequential index to node name.
    pub idx_to_name: Vec<String>,

    /// Per-node dispatch route (indexed by node index).
    pub routes: Vec<DispatchRoute>,

    /// Whether any node in the graph uses 2D spatial convolution semantics.
    pub has_conv2d: bool,
    /// Sequential index of the output node.
    pub output_node_idx: usize,

    /// Index assigned to `NETWORK_INPUT` (always `node_count`).
    pub network_input_idx: usize,
}

/// Pre-categorized dispatch route for a node's backward step.
///
/// Input indices refer to the same sequential index space as the plan.
/// `NETWORK_INPUT` is mapped to `plan.network_input_idx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DispatchRoute {
    /// Single-input layer (most common: 64 of 75 layer variants).
    Unary { input_idx: usize },
    /// Two-input layer (Add, Sub, Mul, MatMul, etc.).
    Binary { left_idx: usize, right_idx: usize },
    /// Three-input layer (Where, variable AdaIN, etc.).
    Ternary {
        a_idx: usize,
        b_idx: usize,
        c_idx: usize,
    },
}

impl CrownDispatchPlan {
    /// Build a dispatch plan from a graph's cached execution order and node metadata.
    ///
    /// Requires `exec_order` to already be computed (call `graph.exec_order()` first).
    pub(crate) fn build(graph: &GraphNetwork) -> Result<Self> {
        let exec_order = graph.exec_order()?;
        let node_count = exec_order.len();
        let network_input_idx = node_count; // sentinel at end

        // Build bidirectional name↔index maps.
        let mut name_to_idx = HashMap::with_capacity(node_count + 1);
        let mut idx_to_name = Vec::with_capacity(node_count + 1);

        for (i, name) in exec_order.iter().enumerate() {
            name_to_idx.insert(name.clone(), i);
            idx_to_name.push(name.clone());
        }
        name_to_idx.insert(NETWORK_INPUT.to_string(), network_input_idx);
        idx_to_name.push(NETWORK_INPUT.to_string());

        // Build forward and reverse index orders.
        let forward_order: Vec<usize> = (0..node_count).collect();
        let reverse_order: Vec<usize> = (0..node_count).rev().collect();

        // Resolve output node index. Empty output_node means "last exec-order node",
        // matching the existing graph CROWN entrypoints.
        let output_node_name = if graph.output_node.is_empty() {
            exec_order
                .last()
                .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
                .as_str()
        } else {
            graph.output_node.as_str()
        };
        let output_node_idx = *name_to_idx.get(output_node_name).ok_or_else(|| {
            NyError::InvalidSpec(format!(
                "Output node '{}' not found in exec_order",
                output_node_name
            ))
        })?;

        // Build per-node dispatch routes and detect 2D spatial conv layers.
        let mut routes = Vec::with_capacity(node_count);
        let mut has_conv2d = false;

        for name in exec_order.iter() {
            let node = graph.nodes.get(name).ok_or_else(|| {
                NyError::InvalidSpec(format!("Node '{}' in exec_order but not in nodes", name))
            })?;

            if matches!(node.layer, Layer::Conv2d(_) | Layer::ConvTranspose2d(_)) {
                has_conv2d = true;
            }

            let route = Self::classify_node(node, &name_to_idx)?;
            routes.push(route);
        }

        Ok(Self {
            exec_order: forward_order,
            reverse_order,
            name_to_idx,
            idx_to_name,
            routes,
            has_conv2d,
            output_node_idx,
            network_input_idx,
        })
    }

    /// Classify a node into its dispatch route based on layer arity and input edges.
    fn classify_node(
        node: &super::GraphNode,
        name_to_idx: &HashMap<String, usize>,
    ) -> Result<DispatchRoute> {
        let resolve = |input_name: &str| -> Result<usize> {
            name_to_idx.get(input_name).copied().ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Node '{}' references unknown input '{}'",
                    node.name, input_name
                ))
            })
        };

        let layer = &node.layer;

        if layer.is_ternary() && node.inputs.len() == 3 {
            let a_idx = resolve(&node.inputs[0])?;
            let b_idx = resolve(&node.inputs[1])?;
            let c_idx = resolve(&node.inputs[2])?;
            Ok(DispatchRoute::Ternary {
                a_idx,
                b_idx,
                c_idx,
            })
        } else if layer.is_binary() && node.inputs.len() >= 2 {
            let left_idx = resolve(&node.inputs[0])?;
            let right_idx = resolve(&node.inputs[1])?;
            Ok(DispatchRoute::Binary {
                left_idx,
                right_idx,
            })
        } else if !node.inputs.is_empty() {
            let input_idx = resolve(&node.inputs[0])?;
            Ok(DispatchRoute::Unary { input_idx })
        } else {
            Err(NyError::InvalidSpec(format!(
                "Node '{}' has no inputs and is not NETWORK_INPUT",
                node.name
            )))
        }
    }

    /// Total number of graph nodes (excluding NETWORK_INPUT).
    #[inline]
    pub(crate) fn node_count(&self) -> usize {
        self.exec_order.len()
    }

    /// Look up a node's index by name, or `None` if not found.
    #[inline]
    pub(crate) fn index_of(&self, name: &str) -> Option<usize> {
        self.name_to_idx.get(name).copied()
    }

    /// Get the node name for a given index.
    #[inline]
    pub(crate) fn name_of(&self, idx: usize) -> &str {
        &self.idx_to_name[idx]
    }

    /// Check if an index refers to the NETWORK_INPUT sentinel.
    #[inline]
    pub(crate) fn is_network_input(&self, idx: usize) -> bool {
        idx == self.network_input_idx
    }

    /// Get the first input index for a node, preserving the existing
    /// "first input drives shared pre-activation logic" contract.
    #[inline]
    pub(crate) fn first_input_idx(&self, idx: usize) -> usize {
        match self.routes[idx] {
            DispatchRoute::Unary { input_idx } => input_idx,
            DispatchRoute::Binary { left_idx, .. } => left_idx,
            DispatchRoute::Ternary { a_idx, .. } => a_idx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{AddLayer, ConvTranspose2dLayer, Layer, LinearLayer, ReLULayer};
    use crate::network::core::graph::GraphNode;
    use ndarray::{ArrayD, IxDyn};

    fn make_simple_graph() -> GraphNetwork {
        let mut g = GraphNetwork::new();
        g.try_add_node(GraphNode::from_input(
            "linear1",
            Layer::Linear(
                LinearLayer::new(
                    ndarray::Array2::zeros((4, 3)),
                    Some(ndarray::Array1::zeros(4)),
                )
                .unwrap(),
            ),
        ))
        .unwrap();
        g.try_add_node(GraphNode::new(
            "relu1",
            Layer::ReLU(ReLULayer),
            vec!["linear1".to_string()],
        ))
        .unwrap();
        g.set_output("relu1");
        g
    }

    #[test]
    fn test_dispatch_plan_basic_indices() {
        let g = make_simple_graph();
        let plan = CrownDispatchPlan::build(&g).unwrap();

        assert_eq!(plan.node_count(), 2);
        assert_eq!(plan.network_input_idx, 2);

        // linear1 takes NETWORK_INPUT
        let linear_idx = plan.index_of("linear1").unwrap();
        assert!(matches!(
            plan.routes[linear_idx],
            DispatchRoute::Unary { input_idx } if input_idx == plan.network_input_idx
        ));

        // relu1 takes linear1
        let relu_idx = plan.index_of("relu1").unwrap();
        assert!(matches!(
            plan.routes[relu_idx],
            DispatchRoute::Unary { input_idx } if input_idx == linear_idx
        ));

        assert_eq!(plan.output_node_idx, relu_idx);
        assert!(!plan.has_conv2d);
    }

    #[test]
    fn test_dispatch_plan_binary_node() {
        let mut g = GraphNetwork::new();
        g.try_add_node(GraphNode::from_input(
            "a",
            Layer::Linear(
                LinearLayer::new(
                    ndarray::Array2::zeros((4, 3)),
                    Some(ndarray::Array1::zeros(4)),
                )
                .unwrap(),
            ),
        ))
        .unwrap();
        g.try_add_node(GraphNode::from_input(
            "b",
            Layer::Linear(
                LinearLayer::new(
                    ndarray::Array2::zeros((4, 3)),
                    Some(ndarray::Array1::zeros(4)),
                )
                .unwrap(),
            ),
        ))
        .unwrap();
        g.try_add_node(GraphNode::binary("sum", Layer::Add(AddLayer), "a", "b"))
            .unwrap();
        g.set_output("sum");

        let plan = CrownDispatchPlan::build(&g).unwrap();
        let sum_idx = plan.index_of("sum").unwrap();
        let a_idx = plan.index_of("a").unwrap();
        let b_idx = plan.index_of("b").unwrap();

        assert!(matches!(
            plan.routes[sum_idx],
            DispatchRoute::Binary { left_idx, right_idx }
                if left_idx == a_idx && right_idx == b_idx
        ));
    }

    #[test]
    fn test_dispatch_plan_reverse_order() {
        let g = make_simple_graph();
        let plan = CrownDispatchPlan::build(&g).unwrap();

        // reverse_order should be the reversed forward order
        let expected_reverse: Vec<usize> = plan.exec_order.iter().rev().copied().collect();
        assert_eq!(plan.reverse_order, expected_reverse);
    }

    #[test]
    fn test_dispatch_plan_network_input_sentinel() {
        let g = make_simple_graph();
        let plan = CrownDispatchPlan::build(&g).unwrap();

        assert!(plan.is_network_input(plan.network_input_idx));
        assert!(!plan.is_network_input(0));
        assert_eq!(plan.name_of(plan.network_input_idx), NETWORK_INPUT);
    }

    #[test]
    fn test_dispatch_plan_marks_convtranspose2d_as_spatial_4297() {
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32])
            .expect("kernel shape should be valid");
        let conv_transpose =
            ConvTranspose2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 1, 1)
                .expect("convtranspose2d should construct");

        let mut graph = GraphNetwork::new();
        graph
            .try_add_node(GraphNode::from_input(
                "deconv",
                Layer::ConvTranspose2d(conv_transpose),
            ))
            .expect("graph node should be valid");
        graph.set_output("deconv");

        let plan = CrownDispatchPlan::build(&graph).expect("dispatch plan should build");
        assert!(
            plan.has_conv2d,
            "#4297 regression: ConvTranspose2d graphs must stay on the spatial-conv batched path"
        );
    }

    #[test]
    fn test_dispatch_plan_retargets_output_without_clearing_exec_order_cache() {
        let mut g = make_simple_graph();

        g.exec_order().unwrap();
        let original_exec_order_cache =
            std::ptr::from_ref::<Vec<String>>(g.cached_exec_order.get().unwrap());
        let original_output_idx = g.dispatch_plan().unwrap().output_node_idx;

        assert!(g.cached_exec_order.get().is_some());
        assert!(g.cached_dispatch_plan.get().is_some());

        g.set_output("linear1");

        assert!(g.cached_exec_order.get().is_some());
        assert!(g.cached_dispatch_plan.get().is_none());

        let rebuilt_plan = g.dispatch_plan().unwrap();
        g.exec_order().unwrap();
        let rebuilt_exec_order_cache =
            std::ptr::from_ref::<Vec<String>>(g.cached_exec_order.get().unwrap());
        let linear_idx = rebuilt_plan.index_of("linear1").unwrap();

        assert_eq!(rebuilt_exec_order_cache, original_exec_order_cache);
        assert_eq!(rebuilt_plan.output_node_idx, linear_idx);
        assert_ne!(rebuilt_plan.output_node_idx, original_output_idx);
    }
}
