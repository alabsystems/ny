// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Demand-driven intermediate-bound selection for CROWN-IBP (#3775).
//!
//! Computes which graph nodes need CROWN-IBP tightened bounds based on
//! downstream consumer demand. Nodes that no nonlinear consumer requires
//! keep their forward IBP bounds without attempting CROWN backward.
//!
//! Reference: alpha-beta-CROWN `check_prior_bounds` recursively selects
//! nodes needing intermediate bounds. Source: `auto_LiRPA/bound_general.py:923-968`

use ny_tensor::BoundedTensor;
use std::collections::{HashMap, HashSet};

use crate::network::core::graph::NETWORK_INPUT;
use crate::network::core::GraphNetwork;

/// Identify which nodes need CROWN-IBP tightened bounds.
///
/// A node needs tightened bounds if a downstream layer lists that input index
/// in `required_input_bound_indices()` and the producer's IBP bounds are not
/// already concrete (lower == upper). The graph output is always included so
/// CROWN-IBP preserves the existing contract that exact output nodes still run
/// backward CROWN unless a real fallback fires. Network input is excluded —
/// this is about intermediate node selection, not re-tightening the input
/// domain.
pub(super) fn nodes_requiring_crown_tightening(
    graph: &GraphNetwork,
    exec_order: &[String],
    ibp_bounds: &HashMap<String, BoundedTensor>,
) -> HashSet<String> {
    let mut needs_bounds = HashSet::new();

    let output_name = if graph.output_name().is_empty() {
        exec_order.last().map(String::as_str)
    } else {
        Some(graph.output_name())
    };
    if let Some(output_name) = output_name.filter(|name| *name != NETWORK_INPUT) {
        needs_bounds.insert(output_name.to_string());
    }

    for node_name in exec_order {
        let Some(node) = graph.nodes.get(node_name) else {
            continue;
        };
        let required_indices = node.layer.required_input_bound_indices();
        for &idx in required_indices {
            if let Some(input_name) = node.inputs.get(idx) {
                // Skip the network input — not an intermediate to tighten.
                if input_name == NETWORK_INPUT {
                    continue;
                }
                // Skip producers whose bounds are already concrete.
                if let Some(bounds) = ibp_bounds.get(input_name) {
                    if bounds.lower() == bounds.upper() {
                        continue;
                    }
                }
                needs_bounds.insert(input_name.clone());
            }
        }
    }
    needs_bounds
}
