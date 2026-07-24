// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Pre-loop setup helpers for spec-guided CROWN backward propagation.
//!
//! Isolates the "prepare state for the backward loop" responsibility from the
//! loop itself: intermediate bounds collection, output node resolution, and
//! spec-column contract validation. Split from `core.rs` as part of #3960.

use crate::network::core::{GraphNetwork, GraphNode, GraphTargetShapeContract};

use ny_core::{GemmEngine, NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::Instant;
use tracing::info;

/// Collect intermediate node bounds when no pre-computed bounds are provided.
///
/// Selects between per-node CROWN-IBP (O(N²) backward per graph model) and
/// simple IBP forward collection based on graph structure heuristics.
pub(crate) fn collect_intermediate_bounds(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    deadline: Option<Instant>,
    engine: Option<&dyn GemmEngine>,
) -> Result<std::collections::HashMap<String, BoundedTensor>> {
    // Conv-DAG forward-linear intermediates (#vnncomp-image-forward-linear):
    // same policy and disable flag as the alpha reference collection. Without
    // this, every spec-CROWN setup (PGD prechecks, root/spec passes) re-ran the
    // O(L²) per-node CROWN-IBP repair pass — measured 43s of a 57s cifar100
    // warmup, starving alpha of iterations — on top of intermediates that the
    // certified forward pass (cached: ~22s once, then free) already bounds
    // finitely. Fail-closed to the existing selection on any refusal.
    let conv_dag = graph.has_conv_layers()
        && graph
            .exec_order()
            .map(|order| !graph.is_sequential_graph(order))
            .unwrap_or(false);
    if conv_dag && GraphNetwork::forward_linear_reference_enabled() {
        match graph.collect_forward_linear_bounds_dag_cached(input, engine, deadline) {
            Ok(bounds) => {
                info!("GraphNetwork spec-CROWN: forward-linear intermediates (conv DAG, cached)");
                return Ok((*bounds).clone());
            }
            Err(
                error @ (NyError::UnsupportedOp(_)
                | NyError::UnsupportedConfiguration(_)
                | NyError::DeadlineExceeded(_)
                | NyError::ShapeMismatch { .. }
                | NyError::CpuMemoryExceeded { .. }),
            ) => {
                info!(
                    "GraphNetwork spec-CROWN: forward-linear intermediates unavailable \
                     ({error}); falling back (fail-closed)"
                );
            }
            Err(error) => return Err(error),
        }
    }

    let use_crown_ibp = graph.should_use_crown_ibp_intermediates();
    let use_per_node_crown_ibp = graph.should_collect_per_node_crown_ibp_intermediates();
    if use_per_node_crown_ibp {
        // Pass deadline to CROWN-IBP collection so the O(N²) per-node backward
        // passes respect the verification timeout. Without this, large CNN DAGs
        // (e.g., metaroom 6cnn_ry_49_8) run CROWN-IBP unbounded. Fixed in #3397.
        Ok(graph
            .collect_crown_ibp_bounds_dag_with_status_and_deadline(input, deadline, engine)?
            .bounds)
    } else {
        if use_crown_ibp {
            info!(
                "GraphNetwork spec-CROWN: {} nodes exceeds per-node CROWN-IBP threshold {}, using IBP intermediates for final backward pass",
                graph.nodes.len(),
                crate::network::core::graph::CROWN_IBP_PER_NODE_THRESHOLD
            );
        }
        // Thread the deadline (#4321): the IBP intermediate sweep over a deep conv
        // DAG can overrun the verifier timeout. collect_node_bounds_core checks the
        // deadline between nodes.
        graph.collect_node_bounds_with_engine_and_deadline(input, engine, deadline)
    }
}

/// Resolve the output node name and validate the spec-column contract.
///
/// Returns the output node name after verifying that the spec matrix columns
/// match the output node's shape.
pub(super) fn resolve_output_contract<'a>(
    graph: &'a GraphNetwork,
    exec_order: &'a [String],
    node_bounds: &std::collections::HashMap<String, BoundedTensor>,
    spec_output_dim: usize,
) -> Result<&'a str> {
    let output_node_name = if graph.output_node.is_empty() {
        exec_order
            .last()
            .ok_or_else(|| NyError::InvalidSpec("No nodes in graph".to_string()))?
            .as_str()
    } else {
        &graph.output_node
    };

    let output_bounds = node_bounds.get(output_node_name).ok_or_else(|| {
        NyError::InvalidSpec(format!("Output node {} not found", output_node_name))
    })?;
    let output_contract = GraphTargetShapeContract::from_bounds(output_node_name, output_bounds);
    output_contract.validate_spec_cols(spec_output_dim, "Spec-guided CROWN spec columns")?;

    Ok(output_node_name)
}

/// Build exec-order-indexed node references for the hot backward loop.
pub(super) fn collect_nodes_by_idx<'a>(
    graph: &'a GraphNetwork,
    exec_order: &[String],
) -> Result<Vec<&'a GraphNode>> {
    exec_order
        .iter()
        .map(|name| {
            graph
                .nodes
                .get(name)
                .ok_or_else(|| NyError::InvalidSpec(format!("Node not found: {name}")))
        })
        .collect()
}
