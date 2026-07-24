// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::WeightStore;
use std::collections::HashMap;

use super::super::tensor::scalar_for_input;

pub(super) fn is_close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 1.0e-3
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
    if node.op_type != "Add" || node.input.len() < 2 {
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
            return is_close(scalar, target);
        }
        return false;
    }
    false
}

pub(super) fn mul_has_inputs(node: &onnx_proto::NodeProto, a: &str, b: &str) -> bool {
    node.op_type == "Mul" && node.input.iter().any(|s| s == a) && node.input.iter().any(|s| s == b)
}

pub(super) fn mul_has_input_and_const(
    node: &onnx_proto::NodeProto,
    input: &str,
    target: f32,
    nodes: &[onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
) -> bool {
    if node.op_type != "Mul" {
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
            return is_close(value, target);
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
    if node.op_type != "Mul" || node.input.len() < 2 {
        return None;
    }
    for candidate in &node.input {
        if let Some(value) = scalar_for_input(nodes, producer_by_output, weights, candidate) {
            if is_close(value, target) {
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
                if is_close(value, 2.0) {
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
    match node.op_type.as_str() {
        "Pow" => {
            let base = node.input.first()?.as_str();
            let exp = node.input.get(1)?.as_str();
            if let Some(value) = scalar_for_input(nodes, producer_by_output, weights, exp) {
                if is_close(value, 3.0) {
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
    if add.op_type != "Add" || add.input.len() < 2 {
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
