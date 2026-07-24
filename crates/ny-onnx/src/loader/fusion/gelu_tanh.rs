// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::LayerType;
use std::collections::HashMap;

use super::helpers::{
    add_has_input_and_const, match_gelu_tanh_add, mul_const_other_input, mul_has_input_and_const,
    mul_has_inputs,
};

pub(crate) fn try_fuse_gelu_tanh(
    nodes: &[onnx_proto::NodeProto],
    tanh_idx: usize,
    producer_by_output: &HashMap<&str, usize>,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &WeightStore,
) -> Option<(usize, LayerSpec, Vec<usize>)> {
    // Pattern (tanh approximation):
    //   x -> (x^3) -> Mul(0.044715) -> Add(x, ...) -> Mul(sqrt(2/pi))
    //     -> Tanh -> Add(tanh, 1) -> Mul(x, add) -> Mul(prev, 0.5)
    //   or Add(tanh, 1) -> Mul(add, 0.5) -> Mul(prev, x)
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;

    let tanh = &nodes[tanh_idx];
    let tanh_out = tanh.output.first()?.as_str();
    let tanh_in = tanh.input.first()?.as_str();
    let scale_idx = *producer_by_output.get(tanh_in)?;
    let scale = &nodes[scale_idx];
    let add_out = mul_const_other_input(scale, SQRT_2_OVER_PI, nodes, producer_by_output, weights)?;
    let add_idx = *producer_by_output.get(add_out.as_str())?;
    let (x, mut used) = match_gelu_tanh_add(nodes, producer_by_output, weights, add_idx)?;

    let add1_idx = consumers_by_input
        .get(tanh_out)?
        .iter()
        .copied()
        .find(|&i| {
            add_has_input_and_const(&nodes[i], tanh_out, 1.0, nodes, producer_by_output, weights)
        })?;
    let add1 = &nodes[add1_idx];
    let add1_out = add1.output.first()?.as_str();

    used.push(scale_idx);
    used.push(tanh_idx);
    used.push(add1_idx);

    let add_consumers = consumers_by_input.get(add1_out)?;
    for &mul1_idx in add_consumers {
        let mul1 = &nodes[mul1_idx];
        if mul1.op_type != "Mul" {
            continue;
        }
        let mul1_out = match mul1.output.first() {
            Some(value) if !value.is_empty() => value.as_str(),
            _ => continue,
        };

        if mul_has_inputs(mul1, add1_out, x.as_str()) {
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
                let start_idx = used
                    .iter()
                    .copied()
                    .chain([mul1_idx, mul2_idx])
                    .min()
                    .unwrap_or(tanh_idx);
                let mut attributes = HashMap::new();
                attributes.insert(
                    "approximate".to_string(),
                    AttributeValue::String("tanh".to_string()),
                );
                let spec = LayerSpec {
                    name: if mul2.name.is_empty() {
                        out.clone()
                    } else {
                        mul2.name.clone()
                    },
                    layer_type: LayerType::GELU,
                    inputs: vec![x],
                    outputs: vec![out],
                    weights: None,
                    attributes,
                };
                used.push(mul1_idx);
                used.push(mul2_idx);
                return Some((start_idx, spec, used));
            }
        }

        if mul_has_input_and_const(mul1, add1_out, 0.5, nodes, producer_by_output, weights) {
            if let Some(mul2_idx) = consumers_by_input.get(mul1_out).and_then(|consumers| {
                consumers
                    .iter()
                    .copied()
                    .find(|&i| mul_has_inputs(&nodes[i], mul1_out, x.as_str()))
            }) {
                let mul2 = &nodes[mul2_idx];
                let out = mul2.output.first()?.clone();
                let start_idx = used
                    .iter()
                    .copied()
                    .chain([mul1_idx, mul2_idx])
                    .min()
                    .unwrap_or(tanh_idx);
                let mut attributes = HashMap::new();
                attributes.insert(
                    "approximate".to_string(),
                    AttributeValue::String("tanh".to_string()),
                );
                let spec = LayerSpec {
                    name: if mul2.name.is_empty() {
                        out.clone()
                    } else {
                        mul2.name.clone()
                    },
                    layer_type: LayerType::GELU,
                    inputs: vec![x],
                    outputs: vec![out],
                    weights: None,
                    attributes,
                };
                used.push(mul1_idx);
                used.push(mul2_idx);
                return Some((start_idx, spec, used));
            }
        }
    }

    None
}
