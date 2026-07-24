// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::helpers::resolve_tensor_node_name;
use super::INPUT_NODE_NAME;
use crate::graph_options::{GraphNetworkOptions, MissingOutputPolicy};
use crate::TensorSpec;
use ny_core::{NyError, Result};
use ny_propagate::GraphNetwork;
use std::collections::{HashMap, HashSet};
use tracing::warn;

pub(super) fn select_output_node(
    output_specs: &[TensorSpec],
    tensor_producer: &HashMap<String, String>,
    options: &GraphNetworkOptions,
    tensor_to_node: &HashMap<String, String>,
    last_added_node: Option<String>,
    graph: &GraphNetwork,
) -> Result<Option<String>> {
    // Prefer ONNX graph outputs when selecting the GraphNetwork output node.
    // Fall back to the last successfully added node if no outputs map to graph nodes.
    let mut missing_outputs = Vec::new();
    let mut missing_nodes = Vec::new();
    let mut input_outputs = Vec::new();

    let outputs = output_specs;
    if outputs.is_empty() {
        if options.missing_output_policy == MissingOutputPolicy::Error {
            return Err(NyError::ModelLoad("ONNX graph has no outputs".to_string()));
        }
        warn!("ONNX graph has no outputs; falling back to last added node");
    }

    let selected_outputs: Vec<String> = if let Some(index) = options.output_index {
        if index >= outputs.len() {
            return Err(NyError::ModelLoad(format!(
                "GraphNetwork output_index {} out of range ({} outputs)",
                index,
                outputs.len()
            )));
        }
        vec![outputs[index].name.clone()]
    } else {
        if outputs.len() > 1 {
            warn!(
                "ONNX graph has {} outputs; GraphNetwork supports one output, using first output",
                outputs.len()
            );
        }
        outputs.iter().map(|output| output.name.clone()).collect()
    };

    // Own node names to avoid borrowing graph-local strings during selection.
    let mut output_candidates: Vec<String> = Vec::with_capacity(selected_outputs.len());
    let mut output_seen: HashSet<String> = HashSet::with_capacity(selected_outputs.len());

    for output_name in selected_outputs {
        match resolve_tensor_node_name(&output_name, tensor_to_node, tensor_producer) {
            Some(node_name) if node_name == INPUT_NODE_NAME => {
                input_outputs.push(output_name);
            }
            Some(node_name) if graph.contains_node(&node_name) => {
                if output_seen.insert(node_name.clone()) {
                    output_candidates.push(node_name);
                }
            }
            Some(node_name) => {
                missing_nodes.push(format!("{}->{}", output_name, node_name));
            }
            None => {
                missing_outputs.push(output_name);
            }
        }
    }

    if !input_outputs.is_empty() {
        warn!(
            "ONNX outputs mapped to input tensor(s) and are unusable: {:?}",
            input_outputs
        );
    }

    if options.missing_output_policy == MissingOutputPolicy::Error
        && (!missing_outputs.is_empty() || !missing_nodes.is_empty() || !input_outputs.is_empty())
    {
        let mut parts = Vec::new();
        if !missing_outputs.is_empty() {
            parts.push(format!("missing outputs: [{}]", missing_outputs.join(", ")));
        }
        if !missing_nodes.is_empty() {
            parts.push(format!(
                "outputs not in graph: [{}]",
                missing_nodes.join(", ")
            ));
        }
        if !input_outputs.is_empty() {
            parts.push(format!(
                "outputs mapped to inputs: [{}]",
                input_outputs.join(", ")
            ));
        }
        return Err(NyError::ModelLoad(format!(
            "ONNX output resolution failed: {}",
            parts.join("; ")
        )));
    }

    let output_node = if !output_candidates.is_empty() {
        Some(output_candidates[0].clone())
    } else {
        if !missing_outputs.is_empty() || !missing_nodes.is_empty() {
            warn!(
                "Falling back to last graph node due to unresolved outputs: missing={:?}, unmapped={:?}",
                missing_outputs, missing_nodes
            );
        }
        // Note: The final ONNX layer may be unsupported and skipped (e.g., dynamic Reshape).
        // In that case, using the original last layer name would produce an output node that
        // doesn't exist in the graph.
        last_added_node
    };

    if output_node.is_none()
        && options.missing_output_policy == MissingOutputPolicy::WarnAndFallback
    {
        warn!("ONNX output resolution produced no graph output node");
    } else if output_node.is_none() {
        return Err(NyError::ModelLoad(
            "ONNX output resolution produced no graph output node".to_string(),
        ));
    }

    Ok(output_node)
}
