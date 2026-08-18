// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::WeightStore;
use std::collections::{HashMap, HashSet};

use super::super::tensor::scalar_for_input;

/// Return true only when replacing `fused_nodes` preserves every observable
/// tensor other than the replacement's own outputs.
///
/// Consumer maps do not include authored graph outputs, so both checks are
/// required.  This is the common fail-closed boundary for proto-level semantic
/// fusions that delete their matched nodes.
pub(super) fn fused_subgraph_is_closed(
    nodes: &[onnx_proto::NodeProto],
    fused_nodes: &[usize],
    replacement_outputs: &[String],
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    graph_output_names: &HashSet<String>,
) -> bool {
    let fused: HashSet<usize> = fused_nodes.iter().copied().collect();
    fused_nodes.iter().copied().all(|node_idx| {
        nodes.get(node_idx).is_some_and(|node| {
            node.output.iter().all(|output| {
                replacement_outputs.iter().any(|kept| kept == output)
                    || (!graph_output_names.contains(output)
                        && consumers_by_input
                            .get(output.as_str())
                            .into_iter()
                            .flatten()
                            .all(|consumer_idx| fused.contains(consumer_idx)))
            })
        })
    })
}

/// Match a scalar whose value will be discarded by a semantic graph rewrite.
///
/// Every fusion below replaces an authored arithmetic subgraph with a
/// canonical operator.  A nearby constant still denotes a different function,
/// so approximate matching is not admissible at this boundary.
pub(super) fn matches_exact_scalar(a: f32, b: f32) -> bool {
    a == b
}

pub(super) fn add_has_erf_and_one(
    node: &onnx_proto::NodeProto,
    erf_out: &str,
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
) -> bool {
    add_has_input_and_const(node, erf_out, 1.0, nodes, producer_by_output, weights)
}

pub(super) fn add_has_input_and_const(
    node: &onnx_proto::NodeProto,
    input: &str,
    target: f32,
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
) -> bool {
    if node.op_type != "Add" || node.input.len() != 2 || !node.attribute.is_empty() {
        return false;
    }
    for candidate in &node.input {
        if candidate != input {
            continue;
        }
        let other = if node.input[0] == *candidate {
            node.input.get(1)
        } else {
            node.input.first()
        };
        let other = match other {
            Some(value) => value.as_str(),
            None => return false,
        };
        if let Some(scalar) = scalar_for_input(nodes, producer_by_output, weights, other) {
            return matches_exact_scalar(scalar, target);
        }
        return false;
    }
    false
}

pub(super) fn mul_has_inputs(node: &onnx_proto::NodeProto, a: &str, b: &str) -> bool {
    node.op_type == "Mul"
        && node.input.len() == 2
        && node.attribute.is_empty()
        && ((node.input[0] == a && node.input[1] == b)
            || (node.input[0] == b && node.input[1] == a))
}

pub(super) fn mul_has_input_and_const(
    node: &onnx_proto::NodeProto,
    input: &str,
    target: f32,
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
) -> bool {
    if node.op_type != "Mul" || node.input.len() != 2 || !node.attribute.is_empty() {
        return false;
    }
    if !node.input.iter().any(|s| s == input) {
        return false;
    }
    for other in &node.input {
        if other == input {
            continue;
        }
        if let Some(value) = scalar_for_input(nodes, producer_by_output, weights, other) {
            return matches_exact_scalar(value, target);
        }
    }
    false
}

pub(super) fn mul_const_other_input(
    node: &onnx_proto::NodeProto,
    target: f32,
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
) -> Option<String> {
    if node.op_type != "Mul" || node.input.len() != 2 || !node.attribute.is_empty() {
        return None;
    }
    for candidate in &node.input {
        if let Some(value) = scalar_for_input(nodes, producer_by_output, weights, candidate) {
            if matches_exact_scalar(value, target) {
                let other = if node.input[0] == *candidate {
                    node.input.get(1)
                } else {
                    node.input.first()
                };
                return other.cloned();
            }
        }
    }
    None
}

pub(super) fn match_x2(
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
    input: &str,
) -> Option<(String, Vec<usize>)> {
    let idx = *producer_by_output.get(input)?;
    let node = &nodes[idx];
    if node.input.len() != 2 || node.output.len() != 1 || !node.attribute.is_empty() {
        return None;
    }
    match node.op_type.as_str() {
        "Mul" => {
            let a = node.input.first()?.as_str();
            let b = node.input.get(1)?.as_str();
            if a == b {
                return Some((a.to_string(), vec![idx]));
            }
        }
        "Pow" => {
            let base = node.input.first()?.as_str();
            let exp = node.input.get(1)?.as_str();
            if let Some(value) = scalar_for_input(nodes, producer_by_output, weights, exp) {
                if matches_exact_scalar(value, 2.0) {
                    return Some((base.to_string(), vec![idx]));
                }
            }
        }
        _ => {}
    }
    None
}

pub(super) fn match_x3(
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
    input: &str,
) -> Option<(String, Vec<usize>)> {
    let idx = *producer_by_output.get(input)?;
    let node = &nodes[idx];
    if node.input.len() != 2 || node.output.len() != 1 || !node.attribute.is_empty() {
        return None;
    }
    match node.op_type.as_str() {
        "Pow" => {
            let base = node.input.first()?.as_str();
            let exp = node.input.get(1)?.as_str();
            if let Some(value) = scalar_for_input(nodes, producer_by_output, weights, exp) {
                if matches_exact_scalar(value, 3.0) {
                    return Some((base.to_string(), vec![idx]));
                }
            }
        }
        "Mul" => {
            let a = node.input.first()?.as_str();
            let b = node.input.get(1)?.as_str();
            if let Some((x, mut used)) = match_x2(nodes, producer_by_output, weights, a) {
                if b == x {
                    used.push(idx);
                    return Some((x, used));
                }
            }
            if let Some((x, mut used)) = match_x2(nodes, producer_by_output, weights, b) {
                if a == x {
                    used.push(idx);
                    return Some((x, used));
                }
            }
        }
        _ => {}
    }
    None
}

pub(super) fn match_x3_scaled(
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
    input: &str,
) -> Option<(String, Vec<usize>)> {
    let idx = *producer_by_output.get(input)?;
    let node = &nodes[idx];
    let other = mul_const_other_input(node, 0.044_715, nodes, producer_by_output, weights)?;
    let (x, mut used) = match_x3(nodes, producer_by_output, weights, other.as_str())?;
    used.push(idx);
    Some((x, used))
}

pub(super) fn match_gelu_tanh_add(
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
    add_idx: usize,
) -> Option<(String, Vec<usize>)> {
    let add = &nodes[add_idx];
    if add.op_type != "Add" || add.input.len() != 2 || !add.attribute.is_empty() {
        return None;
    }
    let a = add.input.first()?.as_str();
    let b = add.input.get(1)?.as_str();
    if let Some((x, mut used)) = match_x3_scaled(nodes, producer_by_output, weights, a) {
        if b == x {
            used.push(add_idx);
            return Some((x, used));
        }
    }
    if let Some((x, mut used)) = match_x3_scaled(nodes, producer_by_output, weights, b) {
        if a == x {
            used.push(add_idx);
            return Some((x, used));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::matches_exact_scalar;

    #[test]
    fn semantic_fusion_scalar_match_rejects_adjacent_values() {
        for target in [0.044_715_f32, 0.5, 1.0, 2.0, 3.0] {
            assert!(matches_exact_scalar(target, target));
            assert!(!matches_exact_scalar(
                f32::from_bits(target.to_bits() - 1),
                target
            ));
            assert!(!matches_exact_scalar(
                f32::from_bits(target.to_bits() + 1),
                target
            ));
        }
    }
}
