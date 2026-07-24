// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{AttributeValue, LayerSpec};
use ny_core::LayerType;
use std::collections::HashMap;

/// Try to fuse Trilu + Add + Softmax into CausalSoftmax.
///
/// Pattern (PyTorch causal attention export):
///   mask = Trilu(ones, upper=True)  # Create upper triangular mask
///   mask_cast = Cast(mask)           # Optional: cast to float
///   masked_scores = Add(scores, mask)  # Add mask (with -inf for masked positions)
///   probs = Softmax(masked_scores)
pub(in crate::loader) fn try_fuse_causal_softmax(
    nodes: &[onnx_proto::NodeProto],
    softmax_idx: usize,
    producer_by_output: &HashMap<&str, usize>,
    _consumers_by_input: &HashMap<&str, Vec<usize>>,
) -> Option<(usize, LayerSpec, Vec<usize>)> {
    let softmax = &nodes[softmax_idx];
    if softmax.op_type != "Softmax" {
        return None;
    }

    let softmax_input = softmax.input.first()?.as_str();

    // Look for Add node feeding into Softmax
    let add_idx = *producer_by_output.get(softmax_input)?;
    let add = &nodes[add_idx];
    if add.op_type != "Add" {
        return None;
    }

    // Check if one of the Add inputs comes from Trilu (possibly through Cast)
    let mut trilu_idx: Option<usize> = None;
    let mut cast_idx: Option<usize> = None;
    let mut attention_scores_input: Option<&str> = None;

    // Helper to trace through to Trilu, going through Cast and Mul nodes
    fn trace_to_trilu(
        nodes: &[onnx_proto::NodeProto],
        producer_by_output: &HashMap<&str, usize>,
        start_input: &str,
        max_depth: usize,
    ) -> Option<usize> {
        if max_depth == 0 {
            return None;
        }
        let idx = *producer_by_output.get(start_input)?;
        let node = &nodes[idx];
        match node.op_type.as_str() {
            "Trilu" => Some(idx),
            "Cast" | "Mul" => {
                // Check inputs for Trilu
                for inp in &node.input {
                    if let Some(trilu_idx) =
                        trace_to_trilu(nodes, producer_by_output, inp, max_depth - 1)
                    {
                        return Some(trilu_idx);
                    }
                }
                None
            }
            _ => None,
        }
    }

    for add_input in &add.input {
        // Try to trace this input to a Trilu node through Cast/Mul chain
        if let Some(t_idx) = trace_to_trilu(nodes, producer_by_output, add_input, 5) {
            trilu_idx = Some(t_idx);
            // Check if there's a Cast in the chain
            if let Some(&idx) = producer_by_output.get(add_input.as_str()) {
                let node = &nodes[idx];
                if node.op_type == "Cast" {
                    cast_idx = Some(idx);
                }
            }
        } else if let Some(&idx) = producer_by_output.get(add_input.as_str()) {
            // This might be the attention scores input
            let node = &nodes[idx];
            if node.op_type != "Trilu" && node.op_type != "Cast" && node.op_type != "Mul" {
                attention_scores_input = Some(add_input.as_str());
            }
        } else {
            // Input not from a node - this is likely the attention scores
            attention_scores_input = Some(add_input.as_str());
        }
    }

    // Must have found Trilu for this to be a causal mask pattern
    let trilu_idx = trilu_idx?;
    let trilu = &nodes[trilu_idx];

    // Verify Trilu is upper triangular (causal mask)
    // upper=1 means upper triangular, which creates causal mask when set to -inf
    let upper = trilu
        .attribute
        .iter()
        .find(|a| a.name == "upper")
        .map(|a| a.i)
        .unwrap_or(1); // Default is upper=1

    if upper != 1 {
        return None;
    }

    // Get the actual input to the attention scores (before masking)
    let attention_input = attention_scores_input.unwrap_or_else(|| {
        add.input
            .iter()
            .find(|inp| {
                producer_by_output
                    .get(inp.as_str())
                    .map(|&idx| {
                        nodes[idx].op_type != "Trilu"
                            && nodes[idx].op_type != "Cast"
                            && nodes[idx].op_type != "Mul"
                    })
                    .unwrap_or(true)
            })
            .map(|s| s.as_str())
            .unwrap_or("")
    });

    let out = softmax.output.first()?.clone();

    // Get softmax axis attribute
    let axis = softmax
        .attribute
        .iter()
        .find(|a| a.name == "axis")
        .map(|a| a.i)
        .unwrap_or(-1);

    let mut attributes = HashMap::new();
    attributes.insert("axis".to_string(), AttributeValue::Int(axis));

    let spec = LayerSpec {
        name: if softmax.name.is_empty() {
            out.clone()
        } else {
            softmax.name.clone()
        },
        layer_type: LayerType::CausalSoftmax,
        inputs: vec![attention_input.to_string()],
        outputs: vec![out],
        weights: None,
        attributes,
    };

    // Collect indices of fused nodes
    let mut consumed = vec![add_idx, softmax_idx, trilu_idx];
    if let Some(idx) = cast_idx {
        consumed.push(idx);
    }

    let start_idx = *consumed.iter().min()?;

    Some((start_idx, spec, consumed))
}
