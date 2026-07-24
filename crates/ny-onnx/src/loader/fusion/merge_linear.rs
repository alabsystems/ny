// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::model::{AttributeValue, DataType, LayerSpec, WeightRef};
use crate::onnx_proto;
use crate::WeightStore;
use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2, IxDyn};
use ny_core::LayerType;
use std::collections::{HashMap, HashSet};

struct AffineStep {
    input_name: String,
    output_name: String,
    weight: Array2<f32>,
    consumed: Vec<usize>,
    removable_tensors: Vec<String>,
    bias: Array1<f32>,
    has_bias: bool,
}

pub(in crate::loader) fn try_fuse_merge_linear(
    nodes: &[onnx_proto::NodeProto],
    start_idx: usize,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &mut WeightStore,
) -> Option<(LayerSpec, HashSet<usize>)> {
    let first = extract_affine_step(nodes, start_idx, consumers_by_input, weights)?;
    let input_name = first.input_name.clone();
    let mut output_name = first.output_name.clone();
    let mut fused_weight = first.weight;
    let mut fused_bias = first.bias;
    let mut consumed = first.consumed;
    let mut removable_tensors: HashSet<String> = first.removable_tensors.into_iter().collect();
    let mut reduced = consumed.len() > 1;
    let mut has_bias = first.has_bias;

    while let Some(next_idx) = single_consumer(consumers_by_input, output_name.as_str()) {
        let Some(next) = extract_affine_step(nodes, next_idx, consumers_by_input, weights) else {
            break;
        };
        if next.input_name != output_name {
            break;
        }

        fused_bias = next.weight.dot(&fused_bias) + &next.bias;
        fused_weight = next.weight.dot(&fused_weight);
        output_name = next.output_name.clone();
        removable_tensors.extend(next.removable_tensors);
        consumed.extend(next.consumed);
        reduced = true;
        has_bias |= next.has_bias;
    }

    if !reduced {
        return None;
    }

    let base = node_name_or_output(nodes.get(start_idx)?);
    let fused_weight_name = format!("{base}__merge_linear_weight");
    let fused_weight_shape = fused_weight.shape().to_vec();
    weights.insert(fused_weight_name.clone(), fused_weight.into_dyn());

    let mut spec_inputs = vec![input_name, fused_weight_name.clone()];
    let mut weights_ref = Some(WeightRef {
        name: fused_weight_name,
        shape: fused_weight_shape,
        original_dtype: DataType::Float32,
    });

    if has_bias {
        let fused_bias_name = format!("{base}__merge_linear_bias");
        spec_inputs.push(fused_bias_name.clone());
        weights.insert(fused_bias_name, fused_bias.into_dyn());
    }

    for name in removable_tensors {
        let _ = weights.remove(name.as_str());
    }

    let mut attributes = HashMap::new();
    attributes.insert("transB".to_string(), AttributeValue::Int(1));

    let spec = LayerSpec {
        name: format!("{base}__merge_linear"),
        layer_type: LayerType::Linear,
        inputs: spec_inputs,
        outputs: vec![output_name],
        weights: weights_ref.take(),
        attributes,
    };

    Some((spec, consumed.into_iter().collect()))
}

fn extract_affine_step(
    nodes: &[onnx_proto::NodeProto],
    node_idx: usize,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &WeightStore,
) -> Option<AffineStep> {
    let node = nodes.get(node_idx)?;
    if node.op_type != "MatMul" || node.input.len() != 2 || node.output.len() != 1 {
        return None;
    }

    let input_name = node.input.first()?.clone();
    let weight_name = node.input.get(1)?.clone();
    if input_name.is_empty() || weight_name.is_empty() {
        return None;
    }
    if weights.get(input_name.as_str()).is_some() {
        return None;
    }
    if !tensor_consumed_only_by(consumers_by_input, weight_name.as_str(), node_idx) {
        return None;
    }

    let weight_tensor = weights.get(weight_name.as_str())?;
    let weight =
        matmul_weight_to_linear(weight_tensor, matmul_transpose_b(node), matmul_scale(node))?;
    let output_dim = *weight.shape().first()?;
    let mut output_name = node.output.first()?.clone();
    let mut consumed = vec![node_idx];
    let mut removable_tensors = vec![weight_name];
    let mut bias = Array1::zeros(output_dim);
    let mut has_bias = false;

    if let Some(add_idx) = single_consumer(consumers_by_input, output_name.as_str()) {
        if let Some((bias_name, bias_vec, add_output)) = parse_bias_add(
            nodes,
            add_idx,
            output_name.as_str(),
            output_dim,
            consumers_by_input,
            weights,
        ) {
            bias = bias_vec;
            output_name = add_output;
            consumed.push(add_idx);
            removable_tensors.push(bias_name);
            has_bias = true;
        }
    }

    Some(AffineStep {
        input_name,
        output_name,
        weight,
        consumed,
        removable_tensors,
        bias,
        has_bias,
    })
}

fn parse_bias_add(
    nodes: &[onnx_proto::NodeProto],
    add_idx: usize,
    affine_output: &str,
    expected_len: usize,
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    weights: &WeightStore,
) -> Option<(String, Array1<f32>, String)> {
    let node = nodes.get(add_idx)?;
    if node.op_type != "Add" || node.input.len() != 2 || node.output.len() != 1 {
        return None;
    }

    let bias_name = if node.input.first()?.as_str() == affine_output {
        node.input.get(1)?.clone()
    } else if node.input.get(1)?.as_str() == affine_output {
        node.input.first()?.clone()
    } else {
        return None;
    };
    if bias_name.is_empty()
        || !tensor_consumed_only_by(consumers_by_input, bias_name.as_str(), add_idx)
    {
        return None;
    }

    let bias = parse_bias_tensor(weights.get(bias_name.as_str())?, expected_len)?;
    Some((bias_name, bias, node.output.first()?.clone()))
}

fn parse_bias_tensor(tensor: &ArrayD<f32>, expected_len: usize) -> Option<Array1<f32>> {
    let bias = tensor.clone().into_dimensionality::<Ix1>().ok()?;
    if bias.len() != expected_len {
        return None;
    }
    Some(bias)
}

fn matmul_weight_to_linear(
    weight: &ArrayD<f32>,
    transpose_b: bool,
    scale: Option<f32>,
) -> Option<Array2<f32>> {
    let weight = if weight.ndim() == 1 {
        let k = weight.len();
        weight.clone().into_shape_with_order(IxDyn(&[k, 1])).ok()?
    } else {
        weight.clone()
    };
    let weight_2d = weight.into_dimensionality::<Ix2>().ok()?;
    let mut linear_weight = if transpose_b {
        weight_2d
    } else {
        weight_2d.t().to_owned()
    };
    if let Some(scale) = scale {
        if !scale.is_finite() {
            return None;
        }
        linear_weight.mapv_inplace(|value| value * scale);
    }
    Some(linear_weight)
}

fn matmul_transpose_b(node: &onnx_proto::NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attr| attr.name == "transpose_b")
        .is_some_and(|attr| attr.i != 0)
}

fn matmul_scale(node: &onnx_proto::NodeProto) -> Option<f32> {
    node.attribute
        .iter()
        .find(|attr| attr.name == "scale")
        .map(|attr| attr.f)
}

fn single_consumer(
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    tensor_name: &str,
) -> Option<usize> {
    let consumers = consumers_by_input.get(tensor_name)?;
    (consumers.len() == 1).then_some(consumers[0])
}

fn tensor_consumed_only_by(
    consumers_by_input: &HashMap<&str, Vec<usize>>,
    tensor_name: &str,
    expected_consumer: usize,
) -> bool {
    consumers_by_input
        .get(tensor_name)
        .is_some_and(|consumers| consumers.as_slice() == [expected_consumer])
}

fn node_name_or_output(node: &onnx_proto::NodeProto) -> String {
    if !node.name.is_empty() {
        node.name.clone()
    } else {
        node.output
            .first()
            .cloned()
            .unwrap_or_else(|| "merge_linear".to_string())
    }
}
