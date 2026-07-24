// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::attributes::parse_node_attributes;
use super::const_fold::common::read_tensor_i64s;
use super::fusion::{
    fold_batch_norm_into_conv_linear_with_context, try_discriminate_instance_norm,
    try_fuse_causal_softmax, try_fuse_gelu, try_fuse_gelu_tanh, try_fuse_layer_norm,
    try_fuse_logsumexp, try_fuse_merge_linear,
};
use super::CustomOpRegistry;
use crate::onnx_proto;
use crate::{AttributeValue, LayerSpec, WeightStore};
use ny_core::{NyError, Result};
use tracing::debug;
mod op_map;
use op_map::op_type_to_layer_type;

/// Whether a Cast node's target dtype (`to` attribute) is an integer type.
///
/// ONNX TensorProto.DataType: UINT8=2, INT8=3, UINT16=4, INT16=5, INT32=6,
/// INT64=7, UINT32=12, UINT64=13. Unsigned targets are included: for
/// in-range non-negative values the cast is exactly trunc, and out-of-range /
/// negative inputs are undefined behavior per the ONNX spec (no enclosure is
/// "the" correct one), so trunc is never less sound than the identity drop it
/// replaces. BOOL(9) is NOT integer here — cast-to-bool is `x != 0`, not
/// truncation.
fn cast_target_is_integer(node: &onnx_proto::NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attr| attr.name == "to")
        .is_some_and(|attr| matches!(attr.i, 2 | 3 | 4 | 5 | 6 | 7 | 12 | 13))
}

/// Whether a Cast node's target dtype (`to` attribute) is a reduced-precision
/// float: FLOAT16=10, BFLOAT16=16. Casting an f32 activation to these rounds
/// with up to 2^-11 (f16) / 2^-8 (bf16) relative error, so the identity drop
/// used for full-precision float targets is NOT exact for them.
fn cast_target_is_reduced_float(node: &onnx_proto::NodeProto) -> bool {
    node.attribute
        .iter()
        .find(|attr| attr.name == "to")
        .is_some_and(|attr| matches!(attr.i, 10 | 16))
}

fn compare_op_attribute(op_type: &str) -> Option<&'static str> {
    match op_type {
        "Greater" => Some("Gt"),
        "GreaterOrEqual" => Some("Ge"),
        "Less" => Some("Lt"),
        "LessOrEqual" => Some("Le"),
        "Equal" => Some("Eq"),
        _ => None,
    }
}

pub(super) fn convert_graph_to_layers(
    nodes: &mut [onnx_proto::NodeProto],
    weights: &mut WeightStore,
    registry: &CustomOpRegistry,
    opset_imports: &std::collections::HashMap<String, i64>,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
    graph_output_names: &std::collections::HashSet<String>,
    merge_linear_enabled: bool,
) -> Result<Vec<LayerSpec>> {
    use std::collections::{HashMap, HashSet};

    let mut consumed: HashSet<usize> = fold_batch_norm_into_conv_linear_with_context(
        nodes,
        weights,
        tensor_shapes,
        graph_output_names,
    );

    let mut producer_by_output: HashMap<&str, usize> = HashMap::new();
    let mut consumers_by_input: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        for out in &node.output {
            producer_by_output.insert(out.as_str(), idx);
        }
        for inp in &node.input {
            consumers_by_input
                .entry(inp.as_str())
                .or_default()
                .push(idx);
        }
    }
    let mut fused_starts: HashMap<usize, LayerSpec> = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        if consumed.contains(&idx) {
            continue;
        }
        if node.op_type == "QuantizeLinear" {
            if let Some((spec, taken)) = try_fuse_qdq_relaxation(nodes, idx, &consumers_by_input) {
                fused_starts.insert(idx, spec);
                consumed.extend(taken);
                continue;
            }
        }
        if merge_linear_enabled && node.op_type == "MatMul" {
            if let Some((spec, taken)) =
                try_fuse_merge_linear(nodes, idx, &consumers_by_input, weights)
            {
                fused_starts.insert(idx, spec);
                consumed.extend(taken);
                continue;
            }
        }
        if node.op_type == "Erf" {
            if let Some((start_idx, spec, taken)) = try_fuse_gelu(
                nodes,
                idx,
                &producer_by_output,
                &consumers_by_input,
                weights,
            ) {
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        } else if node.op_type == "Tanh" {
            if let Some((start_idx, spec, taken)) = try_fuse_gelu_tanh(
                nodes,
                idx,
                &producer_by_output,
                &consumers_by_input,
                weights,
            ) {
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        } else if node.op_type == "ReduceMean" {
            if let Some((start_idx, mut spec, taken)) = try_fuse_layer_norm(
                nodes,
                idx,
                &producer_by_output,
                &consumers_by_input,
                weights,
            ) {
                // Discriminate InstanceNorm from LayerNorm (Part of #3591).
                try_discriminate_instance_norm(&mut spec, tensor_shapes, weights);
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        } else if node.op_type == "Softmax" {
            // Check if this Softmax is preceded by Trilu -> Add (causal mask pattern)
            if let Some((start_idx, spec, taken)) =
                try_fuse_causal_softmax(nodes, idx, &producer_by_output, &consumers_by_input)
            {
                debug!(
                    "Fused causal softmax pattern starting at node {}",
                    start_idx
                );
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        } else if node.op_type == "Log" {
            if let Some((start_idx, spec, taken)) =
                try_fuse_logsumexp(nodes, idx, &producer_by_output, &consumers_by_input)
            {
                debug!("Fused logsumexp pattern starting at node {}", start_idx);
                fused_starts.insert(start_idx, spec);
                consumed.extend(taken);
            }
        }
    }
    let mut layers = Vec::new();
    for (idx, node) in nodes.iter().enumerate() {
        if let Some(spec) = fused_starts.get(&idx) {
            layers.push(spec.clone());
            continue;
        }
        if consumed.contains(&idx) {
            continue;
        }
        if let Some(mut layer) = convert_node_to_layer(node, registry, opset_imports)? {
            normalize_conv_rank_layer(&mut layer, weights);
            normalize_tile_layer(&mut layer, weights, tensor_shapes)?;
            layers.push(layer);
        }
    }
    Ok(layers)
}

fn normalize_conv_rank_layer(layer: &mut LayerSpec, weights: &WeightStore) {
    let is_conv = layer.layer_type == ny_core::LayerType::Conv2d;
    let is_conv_transpose = layer.layer_type == ny_core::LayerType::ConvTranspose2d;
    if !is_conv && !is_conv_transpose {
        return;
    }
    let Some(kernel_name) = layer.inputs.get(1) else {
        return;
    };
    let Some(kernel) = weights.get(kernel_name) else {
        return;
    };
    if is_conv && kernel.ndim() == 3 {
        layer.layer_type = ny_core::LayerType::Conv1d;
    } else if is_conv_transpose && kernel.ndim() == 3 {
        layer.layer_type = ny_core::LayerType::ConvTranspose1d;
    }
}

fn normalize_tile_layer(
    layer: &mut LayerSpec,
    weights: &WeightStore,
    tensor_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Result<()> {
    if layer.layer_type != ny_core::LayerType::Tile
        || (layer.attributes.contains_key("axis") && layer.attributes.contains_key("reps"))
    {
        return Ok(());
    }

    let repeats_name = layer.inputs.get(1).ok_or_else(|| {
        NyError::ModelLoad(format!("Tile '{}' requires ONNX repeats input", layer.name))
    })?;
    let repeats = read_tensor_i64s(weights, repeats_name).ok_or_else(|| {
        NyError::UnsupportedOp(format!(
            "Tile '{}' requires constant repeats input '{}'",
            layer.name, repeats_name
        ))
    })?;
    let repeated_axes: Vec<(usize, i64)> = repeats
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, reps)| *reps != 1)
        .collect();
    let (onnx_axis, reps) = match repeated_axes.as_slice() {
        [] => (0_usize, 1_i64),
        [(axis, reps)] if *reps > 0 => (*axis, *reps),
        [(axis, reps)] => {
            return Err(NyError::ModelLoad(format!(
                "Tile '{}': repeats on axis {} must be positive (got {})",
                layer.name, axis, reps
            )))
        }
        _ => {
            return Err(NyError::UnsupportedOp(format!(
                "Tile '{}' only supports a single non-unit repeat axis, got {:?}",
                layer.name, repeats
            )))
        }
    };

    let data_name = layer
        .inputs
        .first()
        .ok_or_else(|| NyError::ModelLoad(format!("Tile '{}' requires data input", layer.name)))?;
    let data_had_batch_axis = tensor_shapes
        .get(data_name)
        .map(|shape| shape.len() > 1)
        .unwrap_or(true);
    let internal_axis = if onnx_axis == 0 {
        if data_had_batch_axis && reps != 1 {
            return Err(NyError::UnsupportedOp(format!(
                "Tile '{}' repeats ONNX batch axis 0, which is stripped in unbatched propagation",
                layer.name
            )));
        }
        0_i64
    } else {
        i64::try_from(onnx_axis - 1).map_err(|_| {
            NyError::ModelLoad(format!(
                "Tile '{}': axis {} out of i64 range",
                layer.name, onnx_axis
            ))
        })?
    };

    layer
        .attributes
        .insert("axis".to_string(), AttributeValue::Int(internal_axis));
    layer
        .attributes
        .insert("reps".to_string(), AttributeValue::Int(reps));
    Ok(())
}

fn try_fuse_qdq_relaxation(
    nodes: &[onnx_proto::NodeProto],
    quant_idx: usize,
    consumers_by_input: &std::collections::HashMap<&str, Vec<usize>>,
) -> Option<(LayerSpec, Vec<usize>)> {
    let quant = &nodes[quant_idx];
    if quant.op_type != "QuantizeLinear" || quant.input.len() < 2 || quant.output.len() != 1 {
        return None;
    }
    let quant_output = quant.output[0].as_str();
    let consumers = consumers_by_input.get(quant_output)?;
    if consumers.len() != 1 {
        return None;
    }
    let dequant_idx = consumers[0];
    let dequant = &nodes[dequant_idx];
    if dequant.op_type != "DequantizeLinear" || dequant.input.len() < 2 {
        return None;
    }
    if dequant.input.first().map(String::as_str) != Some(quant_output) {
        return None;
    }
    if quant.input.get(1) != dequant.input.get(1) {
        return None;
    }
    let quant_zero_point = quant.input.get(2).filter(|name| !name.is_empty());
    let dequant_zero_point = dequant.input.get(2).filter(|name| !name.is_empty());
    if quant_zero_point != dequant_zero_point {
        return None;
    }

    let name = if dequant.name.is_empty() {
        dequant.output.first().cloned().unwrap_or_default()
    } else {
        dequant.name.clone()
    };
    let mut inputs = vec![quant.input[0].clone(), quant.input[1].clone()];
    if let Some(zero_point) = quant_zero_point {
        inputs.push(zero_point.clone());
    }
    let mut attributes = std::collections::HashMap::new();
    attributes.insert("qdq_relaxation".to_string(), AttributeValue::Int(1));

    Some((
        LayerSpec {
            name,
            layer_type: ny_core::LayerType::QuantizeLinear,
            inputs,
            outputs: dequant.output.clone(),
            weights: None,
            attributes,
        },
        vec![dequant_idx],
    ))
}

fn is_standard_domain(domain: &str) -> bool {
    matches!(domain, "" | "ai.onnx" | "ai.onnx.ml")
}

fn lookup_opset_version(
    opset_imports: &std::collections::HashMap<String, i64>,
    domain: &str,
) -> Option<i64> {
    if let Some(version) = opset_imports.get(domain) {
        return Some(*version);
    }
    if domain.is_empty() {
        if let Some(version) = opset_imports.get("ai.onnx") {
            return Some(*version);
        }
    } else if domain == "ai.onnx" {
        if let Some(version) = opset_imports.get("") {
            return Some(*version);
        }
    }
    None
}

fn convert_node_to_layer(
    node: &onnx_proto::NodeProto,
    registry: &CustomOpRegistry,
    opset_imports: &std::collections::HashMap<String, i64>,
) -> Result<Option<LayerSpec>> {
    let op_type = &node.op_type;
    let domain = node.domain.as_str();
    let opset_version = lookup_opset_version(opset_imports, domain);
    let name = if node.name.is_empty() {
        node.output.first().cloned().unwrap_or_default()
    } else {
        node.name.clone()
    };

    for handler in registry.handlers() {
        if let Some(layer) = handler.try_convert_with_context(node, opset_version) {
            return Ok(Some(layer));
        }
        if handler.supports_with_context(op_type, domain, opset_version) {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Custom op handler {} claimed support for domain=\"{}\", op_type=\"{}\" but returned None",
                handler.name(),
                domain,
                op_type
            )));
        }
    }

    if !domain.is_empty() && !is_standard_domain(domain) {
        let version = opset_version
            .map(|version| version.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        if opset_version.is_none() {
            return Err(NyError::UnsupportedConfiguration(format!(
                "Custom op missing opset import: domain=\"{}\", op_type=\"{}\", opset_version={}. \
Hint: add an opset import for the domain or register a CustomOpHandler via OnnxLoadConfig::new",
                domain, op_type, version
            )));
        }
        return Err(NyError::UnsupportedConfiguration(format!(
            "Custom op missing registration: domain=\"{}\", op_type=\"{}\", opset_version={}. \
Hint: register a CustomOpHandler via OnnxLoadConfig::new",
            domain, op_type, version
        )));
    }

    // Cast soundness (#cctsdb B1): a float->int Cast TRUNCATES toward zero, so
    // dropping it as identity is unsound for fractional values on the
    // activation path (trunc(0.5) = 0, not in [0.5, 62]). Lower integer-target
    // Casts to a Trunc layer; if the input turns out to be constant, load-time
    // const-fold / builder constant pre-evaluation still fold the node away
    // (trunc on integer-valued constants is a no-op). Full-precision float
    // targets (f32/f64) keep the identity drop (all bound math is f32).
    // Bool/string/other targets also keep the legacy identity drop: bool cast
    // is (x != 0), a separate mask-propagation feature, and changing it here
    // would silently alter existing behavior.
    if op_type == "Cast" && cast_target_is_integer(node) {
        debug!("Cast op '{}' has integer target; lowering to Trunc", name);
        return Ok(Some(LayerSpec {
            name,
            layer_type: ny_core::LayerType::Trunc,
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            weights: None,
            attributes: parse_node_attributes(node),
        }));
    }

    // Reduced-precision float targets (f16/bf16) round with up to 2^-11
    // relative error the identity drop would silently ignore — fail closed
    // instead: emit a Cast layer, which ny-build's `convert_layer` refuses
    // with `UnsupportedOp`, so the permissive graph build degrades it to a
    // sound OpaqueSkip [-inf, +inf] and strict builds surface the error.
    // Constant-path Casts are still folded away before conversion.
    if op_type == "Cast" && cast_target_is_reduced_float(node) {
        debug!(
            "Cast op '{}' has reduced-precision float target (f16/bf16); \
             refusing identity drop (fail closed)",
            name
        );
        return Ok(Some(LayerSpec {
            name,
            layer_type: ny_core::LayerType::Cast,
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            weights: None,
            attributes: parse_node_attributes(node),
        }));
    }

    let (layer_type, supported) = op_type_to_layer_type(op_type, &name)?;

    if !supported {
        return Ok(None);
    }

    let mut attributes = parse_node_attributes(node);
    if let Some(compare_op) = compare_op_attribute(op_type) {
        attributes.insert(
            "compare_op".to_string(),
            AttributeValue::String(compare_op.to_string()),
        );
    }

    Ok(Some(LayerSpec {
        name,
        layer_type,
        inputs: node.input.clone(),
        outputs: node.output.clone(),
        weights: None,
        attributes,
    }))
}

#[cfg(test)]
mod tests;
