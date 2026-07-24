// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::boundary::EcapaCompositionBoundary;
use super::*;
use std::collections::{HashMap, HashSet};

/// Extract a single-input subgraph from `full_graph` containing all nodes
/// between `root_input` and `output_node`.
///
/// Semantics (from the design doc):
/// 1. Compute reverse-reachable closure from `output_node` back to `root_input`.
/// 2. Iterate nodes in `full_graph.topological_sort()` order.
/// 3. Clone only nodes in the closure.
/// 4. Rewrite input edges equal to `root_input` to `NETWORK_INPUT`.
/// 5. Preserve other internal edges.
/// 6. Error if a retained node depends on a dynamic input outside the closure.
/// 7. Set output to `output_node`.
pub(super) fn extract_single_input_subgraph(
    full_graph: &GraphNetwork,
    root_input: &str,
    output_node: &str,
) -> Result<GraphNetwork, String> {
    let topo_order = full_graph
        .topological_sort()
        .map_err(|e| format!("topological sort failed: {e}"))?;

    let closure = backward_closure_until(full_graph, output_node, root_input);

    let mut sub = GraphNetwork::new();
    for name in &topo_order {
        if !closure.contains(name.as_str()) {
            continue;
        }
        let node = full_graph.node(name).unwrap();
        let remapped_inputs: Vec<String> = node
            .inputs()
            .iter()
            .map(|inp| {
                if inp == root_input || inp == ny_propagate::NETWORK_INPUT {
                    ny_propagate::NETWORK_INPUT.to_string()
                } else {
                    // Both in-closure and out-of-closure inputs pass through
                    // unchanged; the loop below rejects out-of-closure ones.
                    inp.clone()
                }
            })
            .collect();

        for inp in &remapped_inputs {
            if inp != ny_propagate::NETWORK_INPUT && !closure.contains(inp.as_str()) {
                return Err(format!(
                    "node '{}' depends on '{}' which is outside the subgraph closure \
                     and is not the designated root input '{}'",
                    name, inp, root_input
                ));
            }
        }

        sub.add_node(GraphNode::new(
            node.name().to_string(),
            node.layer().clone(),
            remapped_inputs,
        ));
    }
    sub.set_output(output_node);
    Ok(sub)
}

/// Concatenate the three MFA block-output bounds from the node-bound map
/// using the discovered axis. Returns the concatenated BoundedTensor.
pub(super) fn concat_mfa_block_bounds(
    boundary: &EcapaCompositionBoundary,
    node_bounds: &HashMap<String, BoundedTensor>,
) -> BoundedTensor {
    let block_bounds: Vec<BoundedTensor> = boundary
        .block_outputs
        .iter()
        .enumerate()
        .map(|(i, name)| {
            node_bounds
                .get(name)
                .unwrap_or_else(|| {
                    panic!("block output x{} '{name}' missing from node bounds", i + 2)
                })
                .clone()
        })
        .collect();
    // Resolve the raw ONNX concat axis (possibly negative) against the actual
    // rank of the extracted block-output bounds. These are batch-squeezed
    // (2D [C, T]) relative to the original graph, so the ECAPA MFA axis of -2
    // resolves to 0 (channel concat) here.
    let rank = block_bounds[0].shape().len();
    let axis = ny_core::resolve_axis(boundary.concat_axis, rank, "ECAPA MFA concat")
        .expect("MFA concat axis should resolve within the block-output tensor rank");
    BoundedTensor::concat(&block_bounds, axis)
        .expect("BoundedTensor::concat at MFA boundary should succeed")
}

/// Compute the backward-reachable closure from `output_node`, stopping
/// traversal at `root_input` (not including `root_input` in the closure).
fn backward_closure_until<'a>(
    graph: &'a GraphNetwork,
    output_node: &str,
    root_input: &str,
) -> HashSet<&'a str> {
    let mut visited = HashSet::new();
    let mut stack = vec![output_node.to_string()];
    while let Some(name) = stack.pop() {
        if name == ny_propagate::NETWORK_INPUT || name == root_input {
            continue;
        }
        if let Some(node) = graph.node(&name) {
            let node_name: &str = node.name();
            if !visited.insert(node_name) {
                continue;
            }
            for inp in node.inputs() {
                if inp != ny_propagate::NETWORK_INPUT
                    && inp != root_input
                    && !visited.contains(inp.as_str())
                {
                    stack.push(inp.clone());
                }
            }
        }
    }
    visited
}
