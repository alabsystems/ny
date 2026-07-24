// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared graph-BaB setup helpers.
//!
//! Centralizes the root setup steps shared across the graph-BaB engines:
//! preparing immutable root setup inputs, discovering ReLU nodes in a
//! deterministic order, and building the root alpha state.
//!
//! Design: `designs/2026-03-14-issue-1860-graph-bab-service-convergence.md`
//! Issue: #1860 (Packet B)

use std::collections::HashMap;
use std::sync::Arc;

use ny_core::Result;
use ny_tensor::BoundedTensor;
use tracing::info;

use crate::beta_crown::bab_cuts::GraphCutPool;
use crate::beta_crown::branching::GraphSplitHistory;
use crate::beta_crown::config::BetaCrownConfig;
use crate::beta_crown::state::GraphDomainAlphaState;
use crate::bounds::GraphAlphaState;
use crate::{GraphNetwork, Layer};

/// Shared root setup state used across graph-BaB engines.
#[must_use]
pub(crate) struct GraphBabSetup {
    pub(crate) relu_nodes: Vec<String>,
    pub(crate) initial_node_bounds_arc: HashMap<String, Arc<BoundedTensor>>,
}

/// Convert initial node bounds to `Arc` for cheap sharing across root setup.
pub(crate) fn build_initial_node_bounds_arc(
    initial_node_bounds: &HashMap<String, BoundedTensor>,
) -> HashMap<String, Arc<BoundedTensor>> {
    initial_node_bounds
        .iter()
        .map(|(name, bounds)| (name.clone(), Arc::new(bounds.clone())))
        .collect()
}

/// Collect zero-threshold binary activation (ReLU and Sign) node names in
/// deterministic branching order.
///
/// Part of #3769: Sign neurons use the same x=0 threshold as ReLU, so they
/// are first-class branching candidates for graph BaB.
pub(crate) fn build_sorted_relu_nodes(graph: &GraphNetwork) -> Vec<String> {
    let mut relu_nodes: Vec<String> = graph
        .nodes
        .iter()
        .filter_map(|(name, node)| {
            if matches!(node.layer, Layer::ReLU(_) | Layer::Sign(_)) {
                Some(name.clone())
            } else {
                None
            }
        })
        .collect();
    relu_nodes.sort();
    relu_nodes
}

/// Build the immutable setup inputs shared by graph-BaB engines.
pub(crate) fn build_graph_bab_setup(
    graph: &GraphNetwork,
    initial_node_bounds: &HashMap<String, BoundedTensor>,
) -> GraphBabSetup {
    GraphBabSetup {
        relu_nodes: build_sorted_relu_nodes(graph),
        initial_node_bounds_arc: build_initial_node_bounds_arc(initial_node_bounds),
    }
}

/// Build root-domain alpha state by transferring optimized root alpha values.
pub(crate) fn build_root_alpha_state_from_root_alpha(
    root_alpha: &GraphAlphaState,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    history: &GraphSplitHistory,
    initial_node_bounds_arc: &HashMap<String, Arc<BoundedTensor>>,
) -> GraphDomainAlphaState {
    GraphDomainAlphaState::from_root_alpha_state(
        root_alpha,
        graph,
        initial_node_bounds_arc,
        history,
        input,
    )
}

/// Build root-domain alpha state from graph bounds (no prior root alpha optimization).
pub(crate) fn build_root_alpha_state_from_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    history: &GraphSplitHistory,
    initial_node_bounds_arc: &HashMap<String, Arc<BoundedTensor>>,
) -> GraphDomainAlphaState {
    GraphDomainAlphaState::from_graph_bounds(graph, initial_node_bounds_arc, history, input)
}

/// Build root-domain alpha state from either optimized root alpha or bounds.
///
/// `use_root_alpha_warm_start` should be `false` when child domains will not
/// re-optimize alpha (for example `beta_iterations == 0`), so BaB descendants
/// fall back to heuristic slopes instead of inheriting frozen root-domain alpha.
pub(crate) fn build_root_alpha_state(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    history: &GraphSplitHistory,
    initial_node_bounds_arc: &HashMap<String, Arc<BoundedTensor>>,
    root_alpha_state: Option<&GraphAlphaState>,
    use_root_alpha_warm_start: bool,
) -> GraphDomainAlphaState {
    match (use_root_alpha_warm_start, root_alpha_state) {
        (true, Some(root_alpha)) => build_root_alpha_state_from_root_alpha(
            root_alpha,
            graph,
            input,
            history,
            initial_node_bounds_arc,
        ),
        _ => build_root_alpha_state_from_bounds(graph, input, history, initial_node_bounds_arc),
    }
}

/// Initialize graph cut pool from config, optionally generating proactive cuts.
pub(crate) fn build_graph_cut_pool(
    graph: &GraphNetwork,
    initial_node_bounds_arc: &HashMap<String, Arc<BoundedTensor>>,
    relu_nodes: &[String],
    config: &BetaCrownConfig,
) -> Result<GraphCutPool> {
    let mut cut_pool = if config.enable_cuts {
        GraphCutPool::from_config(config)
    } else {
        GraphCutPool::new(0)
    };

    if config.enable_proactive_cuts && config.enable_cuts {
        let proactive_count = cut_pool.generate_proactive_cuts(
            graph,
            initial_node_bounds_arc,
            config.max_proactive_cuts,
        )?;
        if proactive_count > 0 {
            info!(
                "Generated {} proactive cuts for {} ReLU nodes",
                proactive_count,
                relu_nodes.len(),
            );
        }
    }

    Ok(cut_pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    use crate::bounds::GraphAlphaState;
    use crate::layers::LinearLayer;
    use crate::network::GraphNode;
    use crate::ReLULayer;

    #[test]
    fn test_build_graph_bab_setup_collects_sorted_relu_nodes_and_arc_bounds() {
        let bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, 0.25]).into_dyn(),
            arr1(&[1.5_f32, 2.0]).into_dyn(),
        )
        .expect("bounds should be valid");
        let mut initial_node_bounds = HashMap::new();
        initial_node_bounds.insert("relu_b".to_string(), bounds);

        let linear = LinearLayer::new(arr2(&[[1.0_f32]]), None).expect("valid linear layer");
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu_z", Layer::ReLU(ReLULayer)));
        graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
        graph.add_node(GraphNode::from_input("relu_a", Layer::ReLU(ReLULayer)));

        let setup = build_graph_bab_setup(&graph, &initial_node_bounds);
        let shared = setup
            .initial_node_bounds_arc
            .get("relu_b")
            .expect("cloned bounds should preserve the original key");

        assert_eq!(shared.lower(), &arr1(&[-1.0_f32, 0.25]).into_dyn());
        assert_eq!(shared.upper(), &arr1(&[1.5_f32, 2.0]).into_dyn());
        assert_eq!(
            setup.relu_nodes,
            vec!["relu_a".to_string(), "relu_z".to_string()]
        );
    }

    #[test]
    fn test_build_root_alpha_state_uses_bounds_when_root_alpha_missing() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
        graph.set_output("relu0");

        let input_bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.5]).into_dyn(),
            arr1(&[0.75_f32, 1.25]).into_dyn(),
        )
        .expect("input bounds should be valid");
        let history = GraphSplitHistory::new();
        let setup = build_graph_bab_setup(&graph, &HashMap::new());

        let alpha_state = build_root_alpha_state(
            &graph,
            &input_bounds,
            &history,
            &setup.initial_node_bounds_arc,
            None,
            true,
        );

        assert_eq!(alpha_state.len(), 2, "two unstable input neurons expected");
    }

    #[test]
    fn test_build_root_alpha_state_skips_root_alpha_when_warm_start_disabled() {
        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("relu0", Layer::ReLU(ReLULayer)));
        graph.set_output("relu0");

        let input_bounds = BoundedTensor::new(
            arr1(&[-1.0_f32, -0.25]).into_dyn(),
            arr1(&[0.75_f32, 0.5]).into_dyn(),
        )
        .expect("input bounds should be valid");
        let history = GraphSplitHistory::new();
        let setup = build_graph_bab_setup(&graph, &HashMap::new());

        let mut root_alpha = GraphAlphaState::new();
        root_alpha
            .alphas
            .insert("relu0".to_string(), arr1(&[0.33_f32, 0.67]));

        let warm_started = build_root_alpha_state(
            &graph,
            &input_bounds,
            &history,
            &setup.initial_node_bounds_arc,
            Some(&root_alpha),
            true,
        );
        assert!(
            (warm_started.alpha("relu0", 0) - 0.33).abs() < 1e-6,
            "enabled warm start should transfer optimized root alpha"
        );
        assert!(
            (warm_started.alpha("relu0", 1) - 0.67).abs() < 1e-6,
            "enabled warm start should transfer all optimized root alpha entries"
        );

        let heuristic = build_root_alpha_state(
            &graph,
            &input_bounds,
            &history,
            &setup.initial_node_bounds_arc,
            Some(&root_alpha),
            false,
        );
        assert!(
            heuristic.alpha("relu0", 0).abs() < 1e-6,
            "disabled warm start should keep heuristic alpha for neuron 0"
        );
        assert!(
            (heuristic.alpha("relu0", 1) - 1.0).abs() < 1e-6,
            "disabled warm start should keep heuristic alpha for neuron 1"
        );
    }
}
