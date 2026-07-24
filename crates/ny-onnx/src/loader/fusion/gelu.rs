// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::LayerType;
use std::collections::HashMap;

use super::super::tensor::scalar_for_input;
use super::helpers::{add_has_erf_and_one, mul_has_input_and_const, mul_has_inputs};

pub(crate) fn try_fuse_gelu(
    nodes: &[onnx_proto::NodeProto],
    erf_idx: usize,
    producer_by_output: &HashMap<&str, usize>,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &WeightStore,
) -> Option<(usize, LayerSpec, Vec<usize>)> {
    // Pattern:
    //   x -> (Div(x, sqrt2) | Mul(x, 1/sqrt2)) -> Erf -> Add(erf, 1)
    //     -> Mul(x, add) -> Mul(prev, 0.5)
    //   or Add(erf, 1) -> Mul(add, 0.5) -> Mul(prev, x)
    let erf = &nodes[erf_idx];
    let erf_out = erf.output.first()?.as_str();
    let pre_out = erf.input.first()?.as_str();
    let pre_idx = *producer_by_output.get(pre_out)?;
    let pre = &nodes[pre_idx];

    let x = match pre.op_type.as_str() {
        "Div" => {
            let lhs = pre.input.first()?.as_str();
            let rhs = pre.input.get(1)?.as_str();
            if scalar_for_input(nodes, producer_by_output, weights, rhs).is_some() {
                lhs
            } else {
                return None;
            }
        }
        "Mul" => {
            let lhs = pre.input.first()?.as_str();
            let rhs = pre.input.get(1)?.as_str();
            if scalar_for_input(nodes, producer_by_output, weights, lhs).is_some() {
                rhs
            } else if scalar_for_input(nodes, producer_by_output, weights, rhs).is_some() {
                lhs
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let add_idx = consumers_by_input
        .get(erf_out)?
        .iter()
        .copied()
        .find(|&i| add_has_erf_and_one(&nodes[i], erf_out, nodes, producer_by_output, weights))?;
    let add = &nodes[add_idx];
    let add_out = add.output.first()?.as_str();

    let add_consumers = consumers_by_input.get(add_out)?;
    for &mul1_idx in add_consumers {
        let mul1 = &nodes[mul1_idx];
        if mul1.op_type != "Mul" {
            continue;
        }
        let mul1_out = match mul1.output.first() {
            Some(value) if !value.is_empty() => value.as_str(),
            _ => continue,
        };

        if mul_has_inputs(mul1, add_out, x) {
            if let Some(mul2_idx) = consumers_by_input.get(mul1_out).and_then(|consumers| {
                consumers.iter().copied().find(|&i| {
                    mul_has_input_and_const(
                        &nodes[i],
                        mul1_out,
                        0.5,
                        nodes,
                        producer_by_output,
                        weights,
                    )
                })
            }) {
                let mul2 = &nodes[mul2_idx];
                let out = mul2.output.first()?.clone();
                let start_idx = pre_idx
                    .min(erf_idx)
                    .min(add_idx)
                    .min(mul1_idx)
                    .min(mul2_idx);
                let mut attributes = HashMap::new();
                attributes.insert(
                    "approximate".to_string(),
                    AttributeValue::String("none".to_string()),
                );
                let spec = LayerSpec {
                    name: if mul2.name.is_empty() {
                        out.clone()
                    } else {
                        mul2.name.clone()
                    },
                    layer_type: LayerType::GELU,
                    inputs: vec![x.to_string()],
                    outputs: vec![out],
                    weights: None,
                    attributes,
                };
                return Some((
                    start_idx,
                    spec,
                    vec![pre_idx, erf_idx, add_idx, mul1_idx, mul2_idx],
                ));
            }
        }

        if mul_has_input_and_const(mul1, add_out, 0.5, nodes, producer_by_output, weights) {
            if let Some(mul2_idx) = consumers_by_input.get(mul1_out).and_then(|consumers| {
                consumers
                    .iter()
                    .copied()
                    .find(|&i| mul_has_inputs(&nodes[i], mul1_out, x))
            }) {
                let mul2 = &nodes[mul2_idx];
                let out = mul2.output.first()?.clone();
                let start_idx = pre_idx
                    .min(erf_idx)
                    .min(add_idx)
                    .min(mul1_idx)
                    .min(mul2_idx);
                let mut attributes = HashMap::new();
                attributes.insert(
                    "approximate".to_string(),
                    AttributeValue::String("none".to_string()),
                );
                let spec = LayerSpec {
                    name: if mul2.name.is_empty() {
                        out.clone()
                    } else {
                        mul2.name.clone()
                    },
                    layer_type: LayerType::GELU,
                    inputs: vec![x.to_string()],
                    outputs: vec![out],
                    weights: None,
                    attributes,
                };
                return Some((
                    start_idx,
                    spec,
                    vec![pre_idx, erf_idx, add_idx, mul1_idx, mul2_idx],
                ));
            }
        }
    }

    None
}
