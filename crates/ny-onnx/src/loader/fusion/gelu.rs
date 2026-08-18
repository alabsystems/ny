// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::LayerType;
use std::collections::{HashMap, HashSet};

use super::super::tensor::scalar_for_input;
use super::helpers::{
    add_has_erf_and_one, fused_subgraph_is_closed, matches_exact_scalar, mul_has_input_and_const,
    mul_has_inputs,
};

pub(crate) fn try_fuse_gelu(
    nodes: &[onnx_proto::NodeProto],
    erf_idx: usize,
    producer_by_output: &HashMap<&str, usize>,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &WeightStore,
    graph_output_names: &HashSet<String>,
) -> Option<(usize, LayerSpec, Vec<usize>)> {
    // Pattern:
    //   x -> (Div(x, sqrt2) | Mul(x, 1/sqrt2)) -> Erf -> Add(erf, 1)
    //     -> Mul(x, add) -> Mul(prev, 0.5)
    //   or Add(erf, 1) -> Mul(add, 0.5) -> Mul(prev, x)
    let erf = &nodes[erf_idx];
    if erf.op_type != "Erf"
        || erf.input.len() != 1
        || erf.output.len() != 1
        || erf.input[0].is_empty()
        || erf.output[0].is_empty()
        || !erf.attribute.is_empty()
    {
        return None;
    }
    let erf_out = erf.output.first()?.as_str();
    let pre_out = erf.input.first()?.as_str();
    let pre_idx = *producer_by_output.get(pre_out)?;
    let pre = &nodes[pre_idx];
    if pre.input.len() != 2
        || pre.output.len() != 1
        || pre.output[0] != pre_out
        || !pre.attribute.is_empty()
    {
        return None;
    }

    let x = match pre.op_type.as_str() {
        "Div" => {
            let lhs = pre.input.first()?.as_str();
            let rhs = pre.input.get(1)?.as_str();
            if scalar_for_input(nodes, producer_by_output, weights, rhs)
                .is_some_and(|value| matches_exact_scalar(value, std::f32::consts::SQRT_2))
            {
                lhs
            } else {
                return None;
            }
        }
        "Mul" => {
            let lhs = pre.input.first()?.as_str();
            let rhs = pre.input.get(1)?.as_str();
            if scalar_for_input(nodes, producer_by_output, weights, lhs)
                .is_some_and(|value| matches_exact_scalar(value, std::f32::consts::FRAC_1_SQRT_2))
            {
                rhs
            } else if scalar_for_input(nodes, producer_by_output, weights, rhs)
                .is_some_and(|value| matches_exact_scalar(value, std::f32::consts::FRAC_1_SQRT_2))
            {
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
                let fused = vec![pre_idx, erf_idx, add_idx, mul1_idx, mul2_idx];
                if !fused_subgraph_is_closed(
                    nodes,
                    &fused,
                    &spec.outputs,
                    consumers_by_input,
                    graph_output_names,
                ) {
                    return None;
                }
                return Some((start_idx, spec, fused));
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
                let fused = vec![pre_idx, erf_idx, add_idx, mul1_idx, mul2_idx];
                if !fused_subgraph_is_closed(
                    nodes,
                    &fused,
                    &spec.outputs,
                    consumers_by_input,
                    graph_output_names,
                ) {
                    return None;
                }
                return Some((start_idx, spec, fused));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr0;

    fn node(op_type: &str, inputs: &[&str], output: &str) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            op_type: op_type.to_string(),
            input: inputs.iter().map(|input| (*input).to_string()).collect(),
            output: vec![output.to_string()],
            ..Default::default()
        }
    }

    fn maps(nodes: &[onnx_proto::NodeProto]) -> (HashMap<&str, usize>, HashMap<&str, Vec<usize>>) {
        let mut producers = HashMap::new();
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, node) in nodes.iter().enumerate() {
            for output in &node.output {
                producers.insert(output.as_str(), idx);
            }
            for input in &node.input {
                consumers.entry(input.as_str()).or_default().push(idx);
            }
        }
        (producers, consumers)
    }

    fn weights(scale: f32) -> WeightStore {
        let mut weights = WeightStore::new();
        for (name, value) in [("scale", scale), ("one", 1.0), ("half", 0.5)] {
            weights.insert(name.to_string(), arr0(value).into_dyn());
        }
        weights
    }

    fn div_gelu_nodes() -> Vec<onnx_proto::NodeProto> {
        vec![
            node("Div", &["x", "scale"], "pre"),
            node("Erf", &["pre"], "erf"),
            node("Add", &["erf", "one"], "add"),
            node("Mul", &["x", "add"], "mul"),
            node("Mul", &["mul", "half"], "out"),
        ]
    }

    fn can_fuse(
        nodes: &[onnx_proto::NodeProto],
        weights: &WeightStore,
        graph_outputs: &[&str],
    ) -> bool {
        let (producers, consumers) = maps(nodes);
        let graph_outputs = graph_outputs
            .iter()
            .map(|output| (*output).to_string())
            .collect();
        try_fuse_gelu(nodes, 1, &producers, &consumers, weights, &graph_outputs).is_some()
    }

    #[test]
    fn erf_gelu_requires_exact_canonical_prescale() {
        let nodes = div_gelu_nodes();
        assert!(can_fuse(&nodes, &weights(std::f32::consts::SQRT_2), &[]));
        for scale in [
            1.0,
            f32::from_bits(std::f32::consts::SQRT_2.to_bits() - 1),
            f32::from_bits(std::f32::consts::SQRT_2.to_bits() + 1),
        ] {
            assert!(!can_fuse(&nodes, &weights(scale), &[]));
        }

        let mut mul_nodes = div_gelu_nodes();
        mul_nodes[0] = node("Mul", &["x", "scale"], "pre");
        assert!(can_fuse(
            &mul_nodes,
            &weights(std::f32::consts::FRAC_1_SQRT_2),
            &[]
        ));
    }

    #[test]
    fn erf_gelu_preserves_observable_intermediates() {
        let canonical_weights = weights(std::f32::consts::SQRT_2);
        for intermediate in ["pre", "erf", "add", "mul"] {
            let mut nodes = div_gelu_nodes();
            nodes.push(node("Identity", &[intermediate], "aux"));
            assert!(!can_fuse(&nodes, &canonical_weights, &[]));

            let nodes = div_gelu_nodes();
            assert!(!can_fuse(&nodes, &canonical_weights, &[intermediate]));
        }
    }
}
