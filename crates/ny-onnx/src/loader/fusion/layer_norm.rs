// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::LayerType;
use std::collections::HashMap;

use super::super::attributes::node_attr_ints;
use super::super::tensor::{extract_constant_value, scalar_for_input};
use super::helpers::is_close;
use tracing::warn;

pub(crate) fn try_fuse_layer_norm(
    nodes: &[onnx_proto::NodeProto],
    reduce_mean_idx: usize,
    producer_by_output: &HashMap<&str, usize>,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &WeightStore,
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
    let axes1 = reduce_mean_axes(nodes, mean1, producer_by_output, weights)?;
    if axes1.len() != 1 {
        return None;
    }

    let x = mean1.input.first()?.as_str();
    let mean1_out = mean1.output.first()?.as_str();

    let sub_idx = consumers_by_input
        .get(mean1_out)?
        .iter()
        .copied()
        .find(|&i| nodes[i].op_type == "Sub" && nodes[i].input.iter().any(|s| s == x))?;
    let sub = &nodes[sub_idx];
    let sub_out = sub.output.first()?.as_str();

    let (squared_idx, squared_out, pow_const_idx) = if let Some(pow_idx) = consumers_by_input
        .get(sub_out)?
        .iter()
        .copied()
        .find(|&i| nodes[i].op_type == "Pow" && nodes[i].input.iter().any(|s| s == sub_out))
    {
        let pow = &nodes[pow_idx];
        let exp_name = pow
            .input
            .iter()
            .find(|s| s.as_str() != sub_out)
            .map(|s| s.as_str())?;
        let exp_value = scalar_for_input(nodes, producer_by_output, weights, exp_name)?;
        if !is_close(exp_value, 2.0) {
            return None;
        }
        let const_idx = producer_by_output
            .get(exp_name)
            .copied()
            .filter(|&i| nodes[i].op_type == "Constant");
        (pow_idx, pow.output.first()?.as_str(), const_idx)
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
                mul.input.iter().filter(|s| s.as_str() == sub_out).count() == 2
            })?;
        let mul = &nodes[mul_idx];
        (mul_idx, mul.output.first()?.as_str(), None)
    };

    let mean2_idx = consumers_by_input
        .get(squared_out)?
        .iter()
        .copied()
        .find(|&i| nodes[i].op_type == "ReduceMean")?;
    let mean2 = &nodes[mean2_idx];
    let axes2 = reduce_mean_axes(nodes, mean2, producer_by_output, weights)?;
    if axes2.len() != 1 {
        return None;
    }
    let mean2_out = mean2.output.first()?.as_str();

    let add_eps_idx = consumers_by_input
        .get(mean2_out)?
        .iter()
        .copied()
        .find(|&i| {
            let node = &nodes[i];
            if node.op_type != "Add" || node.input.len() < 2 {
                return false;
            }
            let other = node
                .input
                .iter()
                .find(|s| s.as_str() != mean2_out)
                .map(|s| s.as_str());
            match other {
                Some(name) => scalar_for_input(nodes, producer_by_output, weights, name).is_some(),
                None => false,
            }
        })?;
    let add_eps = &nodes[add_eps_idx];
    let add_eps_out = add_eps.output.first()?.as_str();

    let sqrt_idx = consumers_by_input
        .get(add_eps_out)?
        .iter()
        .copied()
        .find(|&i| nodes[i].op_type == "Sqrt")?;
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
    if let Some(pow_const_idx) = pow_const_idx {
        fused_nodes.push(pow_const_idx);
    }

    let (_norm_idx, norm_out) = if let Some(div_idx) = consumers_by_input
        .get(sqrt_out)?
        .iter()
        .copied()
        .find(|&i| {
            nodes[i].op_type == "Div"
                && nodes[i].input.iter().any(|s| s == sub_out)
                && nodes[i].input.iter().any(|s| s == sqrt_out)
        }) {
        let div = &nodes[div_idx];
        fused_nodes.push(div_idx);
        (div_idx, div.output.first()?.as_str())
    } else {
        let inv_idx = consumers_by_input
            .get(sqrt_out)?
            .iter()
            .copied()
            .find(|&i| nodes[i].op_type == "Reciprocal")?;
        let inv = &nodes[inv_idx];
        let inv_out = inv.output.first()?.as_str();
        let mul_norm_idx = consumers_by_input
            .get(inv_out)?
            .iter()
            .copied()
            .find(|&i| {
                nodes[i].op_type == "Mul"
                    && nodes[i].input.iter().any(|s| s == sub_out)
                    && nodes[i].input.iter().any(|s| s == inv_out)
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
        .find(|&i| nodes[i].op_type == "Mul")?;
    let mul = &nodes[mul_idx];
    let mul_out = mul.output.first()?.as_str();
    fused_nodes.push(mul_idx);

    let ny_input = mul
        .input
        .iter()
        .find(|s| s.as_str() != norm_out)
        .map(|s| s.as_str())?;
    let ny_name = resolve_weight_input(nodes, producer_by_output, weights, ny_input)?;

    let add_beta_idx = consumers_by_input
        .get(mul_out)?
        .iter()
        .copied()
        .find(|&i| nodes[i].op_type == "Add")?;
    let add_beta = &nodes[add_beta_idx];
    let out = add_beta.output.first()?.clone();
    fused_nodes.push(add_beta_idx);

    let beta_input = add_beta
        .input
        .iter()
        .find(|s| s.as_str() != mul_out)
        .map(|s| s.as_str())?;
    let beta_name = resolve_weight_input(nodes, producer_by_output, weights, beta_input)?;

    let eps_input = add_eps
        .input
        .iter()
        .find(|s| s.as_str() != mean2_out)
        .map(|s| s.as_str())?;
    let eps = scalar_for_input(nodes, producer_by_output, weights, eps_input)?;
    if let Some(idx) = producer_by_output
        .get(eps_input)
        .copied()
        .filter(|&i| nodes[i].op_type == "Constant")
    {
        fused_nodes.push(idx);
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

fn reduce_mean_axes(
    nodes: &[onnx_proto::NodeProto],
    mean: &onnx_proto::NodeProto,
    producer_by_output: &HashMap<&str, usize>,
    weights: &WeightStore,
) -> Option<Vec<i64>> {
    if let Some(axes) = node_attr_ints(mean, "axes") {
        return Some(axes);
    }

    let axes_name = mean.input.get(1)?.as_str();
    if axes_name.is_empty() {
        return None;
    }

    if let Some(axes) = weights.get(axes_name) {
        return Some(axes.iter().map(|v| *v as i64).collect());
    }

    let producer_idx = *producer_by_output.get(axes_name)?;
    let producer = &nodes[producer_idx];
    let axes = extract_constant_value(producer)
        .map_err(|e| {
            warn!("layer_norm axes extraction failed: {e}");
            e
        })
        .ok()
        .flatten()?;
    Some(axes.iter().map(|v| *v as i64).collect())
}
