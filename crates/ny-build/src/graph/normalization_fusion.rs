// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::compound_nodes::is_compound_generated;
use std::collections::HashMap;

use ndarray::ArrayD;
use ny_core::{LayerType, Result};
use ny_propagate::layers::InstanceNorm1dLayer;
use ny_propagate::Layer;
use tracing::debug;

use crate::{AttributeValue, ConvertContext, LayerSpec};

#[derive(Debug)]
pub(super) struct FusedUnaryLayer {
    pub(super) activation_input_tensor: String,
    pub(super) layer: Layer,
}

pub(super) fn try_instance_norm_fusion(
    spec_idx: usize,
    layers: &[LayerSpec],
    output_to_spec: &HashMap<String, usize>,
    context: &ConvertContext<'_>,
) -> Result<Option<FusedUnaryLayer>> {
    let spec = &layers[spec_idx];
    if is_compound_generated(spec) {
        return Ok(None);
    }

    // Entry via Div (existing path): Div(centered, Sqrt(Add(variance, eps)))
    if spec.layer_type == LayerType::Div && spec.inputs.len() == 2 {
        return try_instance_norm_via_div(spec, layers, output_to_spec, context);
    }

    // Entry via Mul (new path): Mul(centered, Reciprocal(Sqrt(Add(variance, eps))))
    if spec.layer_type == LayerType::Mul && spec.inputs.len() == 2 {
        return try_instance_norm_via_reciprocal_mul(spec, layers, output_to_spec, context);
    }

    Ok(None)
}

/// Match decomposed InstanceNorm ending with `Div(centered, std)`.
fn try_instance_norm_via_div(
    spec: &LayerSpec,
    layers: &[LayerSpec],
    output_to_spec: &HashMap<String, usize>,
    context: &ConvertContext<'_>,
) -> Result<Option<FusedUnaryLayer>> {
    let Some(numerator_spec) = producer_spec(&spec.inputs[0], layers, output_to_spec) else {
        return Ok(None);
    };
    let Some(sqrt_spec) = producer_spec(&spec.inputs[1], layers, output_to_spec) else {
        return Ok(None);
    };
    if sqrt_spec.layer_type != LayerType::Sqrt || sqrt_spec.inputs.len() != 1 {
        return Ok(None);
    }

    let Some(source_tensor) =
        extract_centered_input_tensor(numerator_spec, layers, output_to_spec, context)
    else {
        return Ok(None);
    };

    build_fused_instance_norm(
        spec,
        &source_tensor,
        sqrt_spec,
        layers,
        output_to_spec,
        context,
    )
}

/// Match decomposed InstanceNorm ending with `Mul(centered, Reciprocal(std))`.
///
/// This variant is emitted by ONNX exports that use `Reciprocal → Mul` instead
/// of `Div`. The kokoro vocoder's AdaIN produces this pattern after the `style`
/// auxiliary input is frozen to a constant.
fn try_instance_norm_via_reciprocal_mul(
    spec: &LayerSpec,
    layers: &[LayerSpec],
    output_to_spec: &HashMap<String, usize>,
    context: &ConvertContext<'_>,
) -> Result<Option<FusedUnaryLayer>> {
    // One of the Mul inputs should be Reciprocal(Sqrt(...)), the other should
    // be the centered activation Sub(x, ReduceMean(x)).
    let spec_a = producer_spec(&spec.inputs[0], layers, output_to_spec);
    let spec_b = producer_spec(&spec.inputs[1], layers, output_to_spec);

    // Try both orderings: (centered, Reciprocal) and (Reciprocal, centered)
    let (centered_spec, reciprocal_spec) = match (spec_a, spec_b) {
        (Some(a), Some(b)) if b.layer_type == LayerType::Reciprocal && b.inputs.len() == 1 => {
            (a, b)
        }
        (Some(a), Some(b)) if a.layer_type == LayerType::Reciprocal && a.inputs.len() == 1 => {
            (b, a)
        }
        _ => return Ok(None),
    };

    // Reciprocal input must be Sqrt(Add(variance, eps))
    let Some(sqrt_spec) = producer_spec(&reciprocal_spec.inputs[0], layers, output_to_spec) else {
        return Ok(None);
    };
    if sqrt_spec.layer_type != LayerType::Sqrt || sqrt_spec.inputs.len() != 1 {
        return Ok(None);
    }

    let Some(source_tensor) =
        extract_centered_input_tensor(centered_spec, layers, output_to_spec, context)
    else {
        return Ok(None);
    };

    build_fused_instance_norm(
        spec,
        &source_tensor,
        sqrt_spec,
        layers,
        output_to_spec,
        context,
    )
}

/// Shared core: validate the variance chain under Sqrt and build the fused layer.
fn build_fused_instance_norm(
    spec: &LayerSpec,
    source_tensor: &str,
    sqrt_spec: &LayerSpec,
    layers: &[LayerSpec],
    output_to_spec: &HashMap<String, usize>,
    context: &ConvertContext<'_>,
) -> Result<Option<FusedUnaryLayer>> {
    let Some((variance_input, eps)) = extract_nonconstant_scalar_add(
        sqrt_spec.inputs[0].as_str(),
        layers,
        output_to_spec,
        context,
    ) else {
        return Ok(None);
    };
    if !eps.is_finite() || eps <= 0.0 {
        return Ok(None);
    }

    let Some(variance_reduce_spec) = producer_spec(&variance_input, layers, output_to_spec) else {
        return Ok(None);
    };
    if !matches_reduce_mean_last_axis(variance_reduce_spec, context)
        || variance_reduce_spec.inputs.len() != 1
    {
        return Ok(None);
    }

    let Some(variance_source) = extract_squared_centered_input_tensor(
        variance_reduce_spec.inputs[0].as_str(),
        layers,
        output_to_spec,
        context,
    ) else {
        return Ok(None);
    };
    if variance_source != source_tensor {
        return Ok(None);
    }

    let Some(num_channels) = infer_channel_count(source_tensor, context) else {
        return Ok(None);
    };

    debug!(
        "Fusing decomposed InstanceNorm pattern at '{}' -> source='{}', channels={}, eps={}",
        spec.name, source_tensor, num_channels, eps
    );

    Ok(Some(FusedUnaryLayer {
        activation_input_tensor: source_tensor.to_string(),
        layer: Layer::InstanceNorm1d(InstanceNorm1dLayer::new_default(num_channels, eps)?),
    }))
}

fn producer_spec<'a>(
    tensor_name: &str,
    layers: &'a [LayerSpec],
    output_to_spec: &HashMap<String, usize>,
) -> Option<&'a LayerSpec> {
    output_to_spec
        .get(tensor_name)
        .and_then(|idx| layers.get(*idx))
}

fn extract_centered_input_tensor(
    sub_spec: &LayerSpec,
    layers: &[LayerSpec],
    output_to_spec: &HashMap<String, usize>,
    context: &ConvertContext<'_>,
) -> Option<String> {
    if sub_spec.layer_type != LayerType::Sub || sub_spec.inputs.len() != 2 {
        return None;
    }

    let source_tensor = sub_spec.inputs[0].clone();
    let mean_spec = producer_spec(sub_spec.inputs[1].as_str(), layers, output_to_spec)?;
    if !matches_reduce_mean_last_axis(mean_spec, context) {
        return None;
    }
    if mean_spec.inputs.len() != 1 || mean_spec.inputs[0] != source_tensor {
        return None;
    }

    Some(source_tensor)
}

fn extract_squared_centered_input_tensor(
    squared_tensor: &str,
    layers: &[LayerSpec],
    output_to_spec: &HashMap<String, usize>,
    context: &ConvertContext<'_>,
) -> Option<String> {
    let square_spec = producer_spec(squared_tensor, layers, output_to_spec)?;
    match square_spec.layer_type {
        LayerType::Mul
            if square_spec.inputs.len() == 2 && square_spec.inputs[0] == square_spec.inputs[1] =>
        {
            let centered_spec =
                producer_spec(square_spec.inputs[0].as_str(), layers, output_to_spec)?;
            extract_centered_input_tensor(centered_spec, layers, output_to_spec, context)
        }
        LayerType::Pow if square_spec.inputs.len() == 2 => {
            let exponent = context.constant_value(&square_spec.inputs[1])?;
            if (scalar_tensor_value(&exponent)? - 2.0).abs() > f32::EPSILON {
                return None;
            }
            let centered_spec =
                producer_spec(square_spec.inputs[0].as_str(), layers, output_to_spec)?;
            extract_centered_input_tensor(centered_spec, layers, output_to_spec, context)
        }
        _ => None,
    }
}

fn extract_nonconstant_scalar_add(
    add_tensor: &str,
    layers: &[LayerSpec],
    output_to_spec: &HashMap<String, usize>,
    context: &ConvertContext<'_>,
) -> Option<(String, f32)> {
    let add_spec = producer_spec(add_tensor, layers, output_to_spec)?;
    if add_spec.layer_type != LayerType::Add || add_spec.inputs.len() != 2 {
        return None;
    }

    let first_constant = context.constant_value(&add_spec.inputs[0]);
    let second_constant = context.constant_value(&add_spec.inputs[1]);
    match (first_constant, second_constant) {
        (Some(constant), None) => {
            Some((add_spec.inputs[1].clone(), scalar_tensor_value(&constant)?))
        }
        (None, Some(constant)) => {
            Some((add_spec.inputs[0].clone(), scalar_tensor_value(&constant)?))
        }
        _ => None,
    }
}

fn scalar_tensor_value(tensor: &ArrayD<f32>) -> Option<f32> {
    (tensor.len() == 1)
        .then(|| tensor.iter().next().copied())
        .flatten()
        .filter(|value| value.is_finite())
}

fn matches_reduce_mean_last_axis(spec: &LayerSpec, context: &ConvertContext<'_>) -> bool {
    if spec.layer_type != LayerType::ReduceMean {
        return false;
    }

    let keepdims = match spec.attributes.get("keepdims") {
        Some(AttributeValue::Int(value)) => *value != 0,
        _ => true,
    };
    if !keepdims {
        return false;
    }

    let Some(input_name) = spec.inputs.first() else {
        return false;
    };
    let Some(input_shape) = context.tensor_shapes.get(input_name) else {
        return false;
    };
    let Some(last_axis) = input_shape.len().checked_sub(1).map(|axis| axis as i64) else {
        return false;
    };

    let Some(axes) = read_reduce_axes(spec, context) else {
        return false;
    };
    axes.len() == 1 && (axes[0] == last_axis || axes[0] == -1)
}

fn read_reduce_axes(spec: &LayerSpec, context: &ConvertContext<'_>) -> Option<Vec<i64>> {
    if let Some(AttributeValue::Ints(axes)) = spec.attributes.get("axes") {
        return Some(axes.clone());
    }

    let axes_name = spec.inputs.get(1)?;
    let axes_tensor = context.constant_value(axes_name)?;
    axes_tensor
        .iter()
        .map(|value| {
            if !value.is_finite() {
                return None;
            }
            // #2360: Reject non-integer axis values. Without this check,
            // values like 2.7 silently round to 3, which may fuse the
            // wrong axis in normalization patterns.
            if value.trunc() != *value {
                return None;
            }
            Some(*value as i64)
        })
        .collect()
}

fn infer_channel_count(source_tensor: &str, context: &ConvertContext<'_>) -> Option<usize> {
    let shape = context.tensor_shapes.get(source_tensor)?;
    match shape.as_slice() {
        [_, channels, ..] if *channels > 0 => Some(*channels as usize),
        [channels, ..] if *channels > 0 => Some(*channels as usize),
        _ => None,
    }
}

#[cfg(test)]
#[path = "normalization_fusion_tests.rs"]
mod tests;
