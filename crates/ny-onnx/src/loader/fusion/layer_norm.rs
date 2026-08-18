// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto::{self, attribute_type};
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::LayerType;
use ny_propagate::layers::NORMALIZATION_MIN_EPS;
use std::collections::{HashMap, HashSet};

use super::super::const_fold::common::read_tensor_i64s;
use super::super::tensor::scalar_for_input;
use super::helpers::{fused_subgraph_is_closed, matches_exact_scalar};

pub(crate) fn try_fuse_layer_norm(
    nodes: &[onnx_proto::NodeProto],
    reduce_mean_idx: usize,
    producer_by_output: &HashMap<&str, usize>,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &WeightStore,
    graph_output_names: &HashSet<String>,
) -> Option<(usize, LayerSpec, Vec<usize>)> {
    // Pattern (PyTorch LayerNorm export):
    //   mean = ReduceMean(x, axes=[-1])
    //   centered = Sub(x, mean)
    //   var = ReduceMean(Pow(centered, 2), axes=[-1])
    //   std = Sqrt(Add(var, eps))
    //   y = Add(Mul(Div(centered, std), ny), beta)
    let mean1 = &nodes[reduce_mean_idx];
    if mean1.op_type != "ReduceMean" {
        return None;
    }
    let (axes1, keeps_dims1) = reduce_mean_semantics(mean1, weights)?;
    if axes1.as_slice() != [-1] || !keeps_dims1 {
        return None;
    }

    let x = mean1.input.first()?.as_str();
    let mean1_out = mean1.output.first()?.as_str();

    let sub_idx = consumers_by_input
        .get(mean1_out)?
        .iter()
        .copied()
        .find(|&i| nodes[i].op_type == "Sub" && has_exact_inputs(&nodes[i], &[x, mean1_out]))?;
    let sub = &nodes[sub_idx];
    let sub_out = sub.output.first()?.as_str();

    let (squared_idx, squared_out) = if let Some(pow_idx) =
        consumers_by_input.get(sub_out)?.iter().copied().find(|&i| {
            let pow = &nodes[i];
            if pow.op_type != "Pow" || pow.input.len() != 2 || pow.input[0] != sub_out {
                return false;
            }
            scalar_for_input(nodes, producer_by_output, weights, &pow.input[1])
                .is_some_and(|value| matches_exact_scalar(value, 2.0))
        }) {
        let pow = &nodes[pow_idx];
        (pow_idx, pow.output.first()?.as_str())
    } else {
        let mul_idx = consumers_by_input
            .get(sub_out)?
            .iter()
            .copied()
            .find(|&i| {
                if nodes[i].op_type != "Mul" {
                    return false;
                }
                let mul = &nodes[i];
                has_exact_inputs(mul, &[sub_out, sub_out])
            })?;
        let mul = &nodes[mul_idx];
        (mul_idx, mul.output.first()?.as_str())
    };

    let mean2_idx = consumers_by_input
        .get(squared_out)?
        .iter()
        .copied()
        .find(|&i| {
            let mean = &nodes[i];
            mean.op_type == "ReduceMean"
                && mean.input.first().is_some_and(|input| input == squared_out)
                && reduce_mean_semantics(mean, weights)
                    .is_some_and(|(axes, keeps_dims)| keeps_dims && axes.as_slice() == [-1])
        })?;
    let mean2 = &nodes[mean2_idx];
    let mean2_out = mean2.output.first()?.as_str();

    let add_eps_idx = consumers_by_input
        .get(mean2_out)?
        .iter()
        .copied()
        .find(|&i| {
            let node = &nodes[i];
            if node.op_type != "Add" {
                return false;
            }
            other_binary_input(node, mean2_out).is_some_and(|name| {
                scalar_for_input(nodes, producer_by_output, weights, name).is_some()
            })
        })?;
    let add_eps = &nodes[add_eps_idx];
    let add_eps_out = add_eps.output.first()?.as_str();

    let sqrt_idx = consumers_by_input
        .get(add_eps_out)?
        .iter()
        .copied()
        .find(|&i| nodes[i].op_type == "Sqrt" && has_exact_inputs(&nodes[i], &[add_eps_out]))?;
    let sqrt = &nodes[sqrt_idx];
    let sqrt_out = sqrt.output.first()?.as_str();

    let mut fused_nodes = vec![
        reduce_mean_idx,
        sub_idx,
        squared_idx,
        mean2_idx,
        add_eps_idx,
        sqrt_idx,
    ];
    let (_norm_idx, norm_out) = if let Some(div_idx) = consumers_by_input
        .get(sqrt_out)?
        .iter()
        .copied()
        .find(|&i| nodes[i].op_type == "Div" && has_exact_inputs(&nodes[i], &[sub_out, sqrt_out]))
    {
        let div = &nodes[div_idx];
        fused_nodes.push(div_idx);
        (div_idx, div.output.first()?.as_str())
    } else {
        let inv_idx = consumers_by_input
            .get(sqrt_out)?
            .iter()
            .copied()
            .find(|&i| {
                nodes[i].op_type == "Reciprocal" && has_exact_inputs(&nodes[i], &[sqrt_out])
            })?;
        let inv = &nodes[inv_idx];
        let inv_out = inv.output.first()?.as_str();
        let mul_norm_idx = consumers_by_input
            .get(inv_out)?
            .iter()
            .copied()
            .find(|&i| {
                nodes[i].op_type == "Mul" && other_binary_input(&nodes[i], inv_out) == Some(sub_out)
            })?;
        let mul_norm = &nodes[mul_norm_idx];
        fused_nodes.push(inv_idx);
        fused_nodes.push(mul_norm_idx);
        (mul_norm_idx, mul_norm.output.first()?.as_str())
    };

    let mul_idx = consumers_by_input
        .get(norm_out)?
        .iter()
        .copied()
        .find(|&i| {
            let node = &nodes[i];
            node.op_type == "Mul"
                && other_binary_input(node, norm_out).is_some_and(|name| {
                    resolve_weight_input(nodes, producer_by_output, weights, name).is_some()
                })
        })?;
    let mul = &nodes[mul_idx];
    let mul_out = mul.output.first()?.as_str();
    fused_nodes.push(mul_idx);

    let ny_input = other_binary_input(mul, norm_out)?;
    let ny_name = resolve_weight_input(nodes, producer_by_output, weights, ny_input)?;

    let add_beta_idx = consumers_by_input
        .get(mul_out)?
        .iter()
        .copied()
        .find(|&i| {
            let node = &nodes[i];
            node.op_type == "Add"
                && other_binary_input(node, mul_out).is_some_and(|name| {
                    resolve_weight_input(nodes, producer_by_output, weights, name).is_some()
                })
        })?;
    let add_beta = &nodes[add_beta_idx];
    let out = add_beta.output.first()?.clone();
    fused_nodes.push(add_beta_idx);

    let beta_input = other_binary_input(add_beta, mul_out)?;
    let beta_name = resolve_weight_input(nodes, producer_by_output, weights, beta_input)?;

    let eps_input = other_binary_input(add_eps, mean2_out)?;
    let eps = scalar_for_input(nodes, producer_by_output, weights, eps_input)?;
    if !eps.is_finite() || eps < NORMALIZATION_MIN_EPS {
        // Fused normalization intentionally refuses smaller epsilons instead
        // of changing them.  Keep the authored primitive graph intact so it
        // can retain its exact arithmetic semantics.
        return None;
    }
    // Constant producers are already materialized in WeightStore and are
    // skipped by ordinary conversion.  Do not consume them: a constant may be
    // shared by an unrelated branch.

    if !fused_subgraph_is_closed(
        nodes,
        &fused_nodes,
        std::slice::from_ref(&out),
        consumers_by_input,
        graph_output_names,
    ) {
        return None;
    }

    let start_idx = *fused_nodes.iter().min()?;

    let mut attributes = HashMap::new();
    attributes.insert("epsilon".to_string(), AttributeValue::Float(eps));

    let spec = LayerSpec {
        name: if add_beta.name.is_empty() {
            out.clone()
        } else {
            add_beta.name.clone()
        },
        layer_type: LayerType::LayerNorm,
        inputs: vec![x.to_string(), ny_name.to_string(), beta_name.to_string()],
        outputs: vec![out],
        weights: None,
        attributes,
    };

    Some((start_idx, spec, fused_nodes))
}

fn has_exact_inputs(node: &onnx_proto::NodeProto, expected: &[&str]) -> bool {
    node.input.len() == expected.len()
        && node
            .input
            .iter()
            .zip(expected)
            .all(|(actual, expected)| actual == expected)
}

fn other_binary_input<'a>(node: &'a onnx_proto::NodeProto, known: &str) -> Option<&'a str> {
    if node.input.len() != 2 {
        return None;
    }
    match (
        node.input[0].as_str() == known,
        node.input[1].as_str() == known,
    ) {
        (true, false) => Some(node.input[1].as_str()),
        (false, true) => Some(node.input[0].as_str()),
        _ => None,
    }
}

fn resolve_weight_input<'a>(
    nodes: &'a [onnx_proto::NodeProto],
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
    name: &'a str,
) -> Option<&'a str> {
    if weights.contains_key(name) {
        return Some(name);
    }

    let producer_idx = *producer_by_output.get(name)?;
    let producer = &nodes[producer_idx];
    if producer.op_type != "Cast" && producer.op_type != "Identity" {
        return None;
    }
    let source = producer.input.first()?.as_str();
    if weights.contains_key(source) {
        return Some(source);
    }
    None
}

fn reduce_mean_semantics(
    mean: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<(Vec<i64>, bool)> {
    if mean.op_type != "ReduceMean"
        || !matches!(mean.input.len(), 1 | 2)
        || mean.input[0].is_empty()
        || mean.output.len() != 1
        || mean.output[0].is_empty()
    {
        return None;
    }

    let mut axes = None;
    let mut keepdims = None;
    let mut noop_with_empty_axes = None;
    for attribute in &mean.attribute {
        match attribute.name.as_str() {
            "axes" if axes.is_none() && attribute.r#type == attribute_type::INTS => {
                axes = Some(attribute.ints.clone());
            }
            "keepdims"
                if keepdims.is_none()
                    && attribute.r#type == attribute_type::INT
                    && matches!(attribute.i_value(), 0 | 1) =>
            {
                keepdims = Some(attribute.i_value() == 1);
            }
            "noop_with_empty_axes"
                if noop_with_empty_axes.is_none()
                    && attribute.r#type == attribute_type::INT
                    && matches!(attribute.i_value(), 0 | 1) =>
            {
                noop_with_empty_axes = Some(attribute.i_value() == 1);
            }
            _ => return None,
        }
    }

    let axes = match axes {
        Some(axes) => {
            // Attribute-form axes and input-form axes belong to different
            // ReduceMean schemas.  Never guess which one wins in a malformed
            // graph that supplies both encodings.
            if mean.input.len() != 1 {
                return None;
            }
            axes
        }
        None => {
            let axes_name = mean.input.get(1)?.as_str();
            if axes_name.is_empty() {
                return None;
            }
            // The axes input is tensor(int64).  WeightStore also exposes a
            // compatibility f32 view, but an integral FLOAT tensor is not an
            // authored discrete operand and must not authorize this rewrite.
            weights.get_integers(axes_name)?;
            read_tensor_i64s(weights, axes_name)?
        }
    };

    if noop_with_empty_axes.unwrap_or(false) && axes.is_empty() {
        return None;
    }
    Some((axes, keepdims.unwrap_or(true)))
}

#[cfg(test)]
mod discrete_axis_tests {
    use super::*;
    use ndarray::arr1;

    fn mean_with_axis_input() -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: vec!["x".to_string(), "axes".to_string()],
            output: vec!["mean".to_string()],
            op_type: "ReduceMean".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn reduce_mean_axes_reject_adjacent_fractional_float() {
        for axis in [
            f32::from_bits(1.0_f32.to_bits() - 1),
            f32::from_bits(1.0_f32.to_bits() + 1),
        ] {
            let mut weights = WeightStore::new();
            weights.insert("axes".to_string(), arr1(&[axis]).into_dyn());
            assert_eq!(
                reduce_mean_semantics(&mean_with_axis_input(), &weights),
                None
            );
        }
    }

    #[test]
    fn reduce_mean_axes_prefers_exact_integer_payload() {
        let mut weights = WeightStore::new();
        weights.insert("axes".to_string(), arr1(&[1.0_f32]).into_dyn());
        weights.insert_integers("axes".to_string(), arr1(&[2_i64]).into_dyn());
        assert_eq!(
            reduce_mean_semantics(&mean_with_axis_input(), &weights),
            Some((vec![2], true))
        );
    }

    #[test]
    fn reduce_mean_axes_require_discrete_provenance() {
        let mut weights = WeightStore::new();
        weights.insert("axes".to_string(), arr1(&[-1.0_f32]).into_dyn());
        assert_eq!(
            reduce_mean_semantics(&mean_with_axis_input(), &weights),
            None
        );
    }

    fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            i: Some(value),
            r#type: attribute_type::INT,
            ..Default::default()
        }
    }

    fn axes_attr(values: &[i64]) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: "axes".to_string(),
            ints: values.to_vec(),
            r#type: attribute_type::INTS,
            ..Default::default()
        }
    }

    #[test]
    fn reduce_mean_semantics_reject_malformed_schema_lookalikes() {
        let weights = WeightStore::new();
        let mut canonical = onnx_proto::NodeProto {
            input: vec!["x".to_string()],
            output: vec!["mean".to_string()],
            op_type: "ReduceMean".to_string(),
            attribute: vec![axes_attr(&[-1])],
            ..Default::default()
        };
        assert_eq!(
            reduce_mean_semantics(&canonical, &weights),
            Some((vec![-1], true))
        );

        canonical.attribute.push(axes_attr(&[-1]));
        assert_eq!(reduce_mean_semantics(&canonical, &weights), None);
        canonical.attribute.pop();

        canonical.attribute.push(int_attr("mystery", 1));
        assert_eq!(reduce_mean_semantics(&canonical, &weights), None);
        canonical.attribute.pop();

        canonical.input.push("axes".to_string());
        assert_eq!(reduce_mean_semantics(&canonical, &weights), None);
        canonical.input.pop();

        canonical.attribute.push(int_attr("keepdims", 2));
        assert_eq!(reduce_mean_semantics(&canonical, &weights), None);
    }
}
