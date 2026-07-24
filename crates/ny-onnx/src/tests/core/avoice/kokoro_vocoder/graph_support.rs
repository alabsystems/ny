// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Kokoro vocoder graph/prefix subgraph helpers.

use super::*;

pub(super) fn instance_norm_node_count(graph: &GraphNetwork) -> usize {
    graph
        .node_names()
        .iter()
        .filter(|name| {
            graph
                .node(name)
                .map(|node| node.layer().layer_type() == "InstanceNorm1d")
                .unwrap_or(false)
        })
        .count()
}

/// Build a prefix subgraph containing only nodes up to and including
/// `cut_node_name` in topological order.
///
/// Uses the Whisper subgraph extraction pattern
/// (ny-onnx/src/whisper/subgraph/attention.rs:49): build a fresh
/// `GraphNetwork` with selective `add_node()` calls to reduce IBP cost.
pub(super) fn vocoder_prefix_subgraph(
    full_graph: &GraphNetwork,
    cut_node_name: &str,
) -> GraphNetwork {
    let topo = full_graph
        .topological_sort()
        .expect("topo sort should succeed");

    let cut_idx = topo
        .iter()
        .position(|name| name == cut_node_name)
        .unwrap_or_else(|| {
            panic!(
                "cut node '{}' not found in topological order",
                cut_node_name
            )
        });

    let mut prefix = GraphNetwork::new();
    for name in &topo[..=cut_idx] {
        let node = full_graph
            .node(name)
            .unwrap_or_else(|| panic!("node '{}' missing from graph", name))
            .clone();
        prefix.add_node(node);
    }
    prefix.set_output(cut_node_name);
    prefix
}

/// Find the first ConvTranspose node name in topological order.
pub(super) fn first_conv_transpose_node(graph: &GraphNetwork) -> String {
    let topo = graph.topological_sort().expect("topo sort should succeed");
    topo.iter()
        .find(|name| {
            graph
                .node(name)
                .map(|n| {
                    n.layer().layer_type() == "ConvTranspose1d"
                        || n.layer().layer_type() == "ConvTranspose2d"
                })
                .unwrap_or(false)
        })
        .cloned()
        .expect("vocoder should have at least one ConvTranspose node")
}

/// Find the first InstanceNorm1d node name in topological order.
pub(super) fn first_instance_norm_node(graph: &GraphNetwork) -> String {
    let topo = graph.topological_sort().expect("topo sort should succeed");
    topo.iter()
        .find(|name| {
            graph
                .node(name)
                .map(|n| n.layer().layer_type() == "InstanceNorm1d")
                .unwrap_or(false)
        })
        .cloned()
        .expect("vocoder should have at least one InstanceNorm1d node")
}
