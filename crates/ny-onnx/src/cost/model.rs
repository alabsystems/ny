// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Model-level static cost estimation.

use std::mem::size_of;

use std::collections::HashMap;

use ny_core::LayerType;

use super::layer_metadata::{
    activation_input_bytes, activation_input_names, activation_input_shapes, is_runtime_tensor,
    parameter_input_bytes, parameter_input_tensors, timing_family,
};
use super::lookup::ShapeLookup;
use super::{CostError, CostResult, LayerCost};
use crate::{AttributeValue, OnnxModel};

const F32_BYTES: u64 = size_of::<f32>() as u64;

pub fn estimate_model_cost(model: &OnnxModel) -> Result<CostResult, CostError> {
    validate_non_empty_model(model)?;

    let mut lookup = ShapeLookup::new(model);
    let (mut live_tensors, mut live_bytes) = initialize_live_inputs(model, &lookup)?;
    let mut uses_left = build_remaining_uses(model);
    let parameter_bytes = count_parameter_bytes(model);
    let (layers, total_flops, peak_activation_bytes) = analyze_layers(
        model,
        &mut lookup,
        &mut live_tensors,
        &mut uses_left,
        &mut live_bytes,
    )?;

    build_cost_result(layers, total_flops, parameter_bytes, peak_activation_bytes)
}

fn validate_non_empty_model(model: &OnnxModel) -> Result<(), CostError> {
    if model.network.layers.is_empty() {
        return Err(CostError::no_layers("static cost estimate"));
    }
    Ok(())
}

fn initialize_live_inputs(
    model: &OnnxModel,
    lookup: &ShapeLookup<'_>,
) -> Result<(HashMap<String, u64>, u64), CostError> {
    let mut live_tensors = HashMap::<String, u64>::new();
    let mut live_bytes = 0_u64;

    for input in &model.network.inputs {
        let shape = lookup.shape_from_spec(input)?;
        let bytes = bytes_for_shape(&shape)?;
        live_bytes = checked_add(live_bytes, bytes, "input activation bytes overflow")?;
        live_tensors.insert(input.name.clone(), bytes);
    }

    Ok((live_tensors, live_bytes))
}

fn analyze_layers(
    model: &OnnxModel,
    lookup: &mut ShapeLookup<'_>,
    live_tensors: &mut HashMap<String, u64>,
    uses_left: &mut HashMap<String, usize>,
    live_bytes: &mut u64,
) -> Result<(Vec<LayerCost>, u64, u64), CostError> {
    let mut layers = Vec::with_capacity(model.network.layers.len());
    let mut cumulative_flops = 0_u64;
    let mut peak_activation_bytes = *live_bytes;

    for layer in &model.network.layers {
        // The ONNX importer retains producer topology after constant folding,
        // so a folded MatMul/Add/etc. may still have a LayerSpec even though
        // every output is already in WeightStore.  Such a node performs no
        // runtime work and has no activation input; trying to cost it as an
        // ordinary MatMul both over-counts and fails on the missing activation.
        if !layer
            .outputs
            .iter()
            .any(|name| is_runtime_tensor(model, name))
        {
            continue;
        }
        let output_shapes = collect_runtime_output_shapes(model, lookup, layer)?;
        // Register computed output shapes so subsequent layers can find shapes
        // of intermediate tensors produced by decomposed ops (ReduceL2, LSTM).
        for (name, shape) in layer
            .outputs
            .iter()
            .filter(|name| is_runtime_tensor(model, name))
            .zip(output_shapes.iter())
        {
            lookup.register_shape(name.clone(), shape.clone());
        }
        let output_elements = sum_output_elements(&output_shapes)?;
        let output_bytes = sum_output_bytes(&output_shapes)?;
        let activation_input_bytes = activation_input_bytes(model, lookup, layer)?;
        let parameter_input_bytes = parameter_input_bytes(model, layer)?;
        let total_tensor_traffic_bytes = checked_add(
            checked_add(
                activation_input_bytes,
                parameter_input_bytes,
                "layer tensor traffic byte count overflow",
            )?,
            output_bytes,
            "layer tensor traffic byte count overflow",
        )?;

        *live_bytes = checked_add(*live_bytes, output_bytes, "live activation bytes overflow")?;
        peak_activation_bytes = peak_activation_bytes.max(*live_bytes);

        let layer_flops = estimate_layer_flops(model, lookup, layer, output_elements)?;
        let layer_timing_family = timing_family(&layer.layer_type, &layer.name)?.to_string();
        cumulative_flops = checked_add(cumulative_flops, layer_flops, "cumulative FLOPs overflow")?;

        insert_runtime_outputs(model, live_tensors, layer, &output_shapes)?;
        release_consumed_inputs(model, live_tensors, uses_left, layer, live_bytes);

        layers.push(LayerCost {
            name: layer.name.clone(),
            layer_type: layer.layer_type.to_string(),
            output_shapes,
            output_elements,
            flops: layer_flops,
            activation_input_bytes,
            parameter_input_bytes,
            output_bytes,
            total_tensor_traffic_bytes,
            timing_family: layer_timing_family,
            peak_live_activation_bytes: peak_activation_bytes.max(*live_bytes),
            cumulative_flops,
        });
    }

    Ok((layers, cumulative_flops, peak_activation_bytes))
}

fn build_cost_result(
    layers: Vec<LayerCost>,
    total_flops: u64,
    parameter_bytes: u64,
    peak_activation_bytes: u64,
) -> Result<CostResult, CostError> {
    Ok(CostResult {
        layers,
        total_flops,
        parameter_bytes,
        peak_activation_bytes,
        peak_total_bytes: checked_add(
            parameter_bytes,
            peak_activation_bytes,
            "peak total byte count overflow",
        )?,
        assumptions: vec![
            "Requires fully-known positive tensor shapes; dynamic dimensions are rejected."
                .to_string(),
            "Counts activation memory as f32 tensors and excludes backend-specific workspace buffers."
                .to_string(),
            "FLOPs are arithmetic estimates for static graph analysis, not cycle-accurate timing."
                .to_string(),
        ],
    })
}

fn collect_runtime_output_shapes(
    model: &OnnxModel,
    lookup: &ShapeLookup<'_>,
    layer: &crate::LayerSpec,
) -> Result<Vec<Vec<usize>>, CostError> {
    layer
        .outputs
        .iter()
        .filter(|name| is_runtime_tensor(model, name))
        .map(|name| {
            lookup.tensor_shape(name).or_else(|_| {
                // Fallback only for audited decomposed intermediates that
                // ny itself generates or rewrites deterministically.
                lookup.infer_output_shape(layer)
            })
        })
        .collect()
}

fn sum_output_elements(output_shapes: &[Vec<usize>]) -> Result<u64, CostError> {
    output_shapes.iter().try_fold(0_u64, |acc, shape| {
        checked_add(
            acc,
            elements_for_shape(shape)?,
            "output element count overflow",
        )
    })
}

fn sum_output_bytes(output_shapes: &[Vec<usize>]) -> Result<u64, CostError> {
    output_shapes.iter().try_fold(0_u64, |acc, shape| {
        checked_add(acc, bytes_for_shape(shape)?, "output byte count overflow")
    })
}

fn insert_runtime_outputs(
    model: &OnnxModel,
    live_tensors: &mut HashMap<String, u64>,
    layer: &crate::LayerSpec,
    output_shapes: &[Vec<usize>],
) -> Result<(), CostError> {
    for (name, shape) in layer
        .outputs
        .iter()
        .filter(|name| is_runtime_tensor(model, name))
        .zip(output_shapes.iter())
    {
        live_tensors.insert(name.clone(), bytes_for_shape(shape)?);
    }
    Ok(())
}

fn release_consumed_inputs(
    model: &OnnxModel,
    live_tensors: &mut HashMap<String, u64>,
    uses_left: &mut HashMap<String, usize>,
    layer: &crate::LayerSpec,
    live_bytes: &mut u64,
) {
    for input_name in activation_input_names(model, layer) {
        decrement_use_count(uses_left, input_name);
        if uses_left.get(input_name).copied().unwrap_or(0) == 0
            && !is_graph_output(model, input_name)
        {
            if let Some(bytes) = live_tensors.remove(input_name) {
                *live_bytes = live_bytes.saturating_sub(bytes);
            }
        }
    }
}

fn estimate_layer_flops(
    model: &OnnxModel,
    lookup: &ShapeLookup<'_>,
    layer: &crate::LayerSpec,
    output_elements: u64,
) -> Result<u64, CostError> {
    let activation_inputs = activation_input_shapes(model, lookup, layer)?;
    let parameter_inputs = parameter_input_tensors(model, layer);

    let first_input_elements = activation_inputs
        .first()
        .map(|shape| elements_for_shape(shape))
        .transpose()?
        .unwrap_or(0);
    let bias_terms = parameter_inputs.len().saturating_sub(1) as u64;

    let flops = match &layer.layer_type {
        LayerType::Linear => {
            let reduction = last_dim(activation_inputs.first(), "Linear input")?;
            output_elements
                .checked_mul(2 * reduction + bias_terms)
                .ok_or_else(|| {
                    CostError::propagation_msg("static cost estimate", "Linear FLOPs overflow")
                })?
        }
        LayerType::Conv1d
        | LayerType::Conv2d
        | LayerType::ConvTranspose1d
        | LayerType::ConvTranspose2d => {
            let weight = parameter_inputs.first().ok_or_else(|| {
                CostError::propagation_msg(
                    "static cost estimate",
                    format!(
                        "{} layer '{}' is missing a weight tensor",
                        layer.layer_type, layer.name
                    ),
                )
            })?;
            let kernel_terms = weight.shape().iter().skip(1).try_fold(1_u64, |acc, dim| {
                checked_mul(acc, *dim as u64, "kernel term overflow")
            })?;
            output_elements
                .checked_mul(2 * kernel_terms + bias_terms)
                .ok_or_else(|| {
                    CostError::propagation_msg("static cost estimate", "Convolution FLOPs overflow")
                })?
        }
        LayerType::MatMul => {
            let reduction = last_dim(activation_inputs.first(), "MatMul input")?;
            output_elements.checked_mul(2 * reduction).ok_or_else(|| {
                CostError::propagation_msg("static cost estimate", "MatMul FLOPs overflow")
            })?
        }
        LayerType::AveragePool | LayerType::MaxPool => {
            let kernel_terms = match int_list_attr(layer, "kernel_shape")? {
                Some(values) if !values.is_empty() => {
                    values.into_iter().try_fold(1_u64, |acc, dim| {
                        checked_mul(acc, dim as u64, "pool kernel FLOPs overflow")
                    })?
                }
                _ => 1,
            };
            output_elements.checked_mul(kernel_terms).ok_or_else(|| {
                CostError::propagation_msg("static cost estimate", "pool FLOPs overflow")
            })?
        }
        LayerType::ReduceMean | LayerType::ReduceSum => first_input_elements,
        LayerType::Softmax | LayerType::CausalSoftmax | LayerType::LogSoftmax => {
            output_elements.checked_mul(5).ok_or_else(|| {
                CostError::propagation_msg("static cost estimate", "softmax FLOPs overflow")
            })?
        }
        LayerType::LayerNorm
        | LayerType::RMSNorm
        | LayerType::InstanceNorm
        | LayerType::GroupNorm
        | LayerType::BatchNorm
        | LayerType::AdaIN => output_elements.checked_mul(6).ok_or_else(|| {
            CostError::propagation_msg("static cost estimate", "normalization FLOPs overflow")
        })?,
        LayerType::SiLU => output_elements.checked_mul(4).ok_or_else(|| {
            CostError::propagation_msg("static cost estimate", "SiLU FLOPs overflow")
        })?,
        LayerType::Sigmoid | LayerType::Tanh | LayerType::Softplus => {
            output_elements.checked_mul(4).ok_or_else(|| {
                CostError::propagation_msg("static cost estimate", "activation FLOPs overflow")
            })?
        }
        LayerType::Exp
        | LayerType::Log
        | LayerType::Sin
        | LayerType::Cos
        | LayerType::Tan
        | LayerType::Arctan
        | LayerType::Erf
        | LayerType::Mish
        | LayerType::GELU => output_elements.checked_mul(6).ok_or_else(|| {
            CostError::propagation_msg("static cost estimate", "transcendental FLOPs overflow")
        })?,
        LayerType::ReLU
        | LayerType::LeakyRelu
        | LayerType::Clip
        | LayerType::Elu
        | LayerType::Selu
        | LayerType::PRelu
        | LayerType::HardSigmoid
        | LayerType::HardSwish
        | LayerType::Celu
        | LayerType::ThresholdedRelu
        | LayerType::Shrink
        | LayerType::Softsign
        | LayerType::Snake
        | LayerType::Floor
        | LayerType::Ceil
        | LayerType::Round
        | LayerType::Trunc
        | LayerType::Sign
        | LayerType::Reciprocal
        | LayerType::Neg
        | LayerType::Triu
        | LayerType::Tril
        | LayerType::Abs
        | LayerType::Sqrt
        | LayerType::Div
        | LayerType::Sub
        | LayerType::Pow
        | LayerType::Add
        | LayerType::Mul
        | LayerType::Min
        | LayerType::Max
        | LayerType::Resize
        | LayerType::RoPE => output_elements,
        LayerType::Concat
        | LayerType::Shape
        | LayerType::Reshape
        | LayerType::Flatten
        | LayerType::Transpose
        | LayerType::Squeeze
        | LayerType::Unsqueeze
        | LayerType::Embedding
        | LayerType::Slice
        | LayerType::Gather
        | LayerType::Pad
        | LayerType::Tile
        | LayerType::Expand
        | LayerType::Where => 0,
        other => {
            return Err(CostError::propagation_msg(
                "static cost estimate",
                format!(
                    "unsupported layer type {} in static cost analysis for layer '{}'",
                    other, layer.name
                ),
            ))
        }
    };

    Ok(flops)
}
fn build_remaining_uses(model: &OnnxModel) -> HashMap<String, usize> {
    let mut counts = HashMap::new();

    for layer in &model.network.layers {
        for input in activation_input_names(model, layer) {
            *counts.entry(input.clone()).or_insert(0) += 1;
        }
    }
    for output in &model.network.outputs {
        *counts.entry(output.name.clone()).or_insert(0) += 1;
    }

    counts
}

fn decrement_use_count(counts: &mut HashMap<String, usize>, name: &str) {
    if let Some(value) = counts.get_mut(name) {
        *value = value.saturating_sub(1);
    }
}

fn is_graph_output(model: &OnnxModel, tensor_name: &str) -> bool {
    model
        .network
        .outputs
        .iter()
        .any(|output| output.name == tensor_name)
}

fn count_parameter_bytes(model: &OnnxModel) -> u64 {
    model
        .weights
        .iter()
        .map(|(_, weight)| weight.len() as u64 * F32_BYTES)
        .sum()
}

fn int_list_attr(layer: &crate::LayerSpec, name: &str) -> Result<Option<Vec<i64>>, CostError> {
    match layer.attributes.get(name) {
        None => Ok(None),
        Some(AttributeValue::Int(value)) => Ok(Some(vec![*value])),
        Some(AttributeValue::Ints(values)) => Ok(Some(values.clone())),
        Some(other) => Err(CostError::propagation_msg(
            "static cost estimate",
            format!(
                "layer '{}' has unsupported {} attribute type {:?}",
                layer.name, name, other
            ),
        )),
    }
}

fn last_dim(shape: Option<&Vec<usize>>, label: &str) -> Result<u64, CostError> {
    let shape = shape.ok_or_else(|| {
        CostError::invalid_input_shape(
            "static cost estimate",
            format!("{label} is missing for cost estimation"),
        )
    })?;
    shape
        .last()
        .copied()
        .map(|value| value as u64)
        .ok_or_else(|| {
            CostError::invalid_input_shape(
                "static cost estimate",
                format!("{label} has no dimensions"),
            )
        })
}

fn elements_for_shape(shape: &[usize]) -> Result<u64, CostError> {
    shape.iter().try_fold(1_u64, |acc, dim| {
        checked_mul(acc, *dim as u64, "tensor element count overflow")
    })
}

fn bytes_for_shape(shape: &[usize]) -> Result<u64, CostError> {
    checked_mul(
        elements_for_shape(shape)?,
        F32_BYTES,
        "tensor byte count overflow",
    )
}

fn checked_mul(lhs: u64, rhs: u64, msg: &str) -> Result<u64, CostError> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| CostError::propagation_msg("static cost estimate", msg))
}

fn checked_add(lhs: u64, rhs: u64, msg: &str) -> Result<u64, CostError> {
    lhs.checked_add(rhs)
        .ok_or_else(|| CostError::propagation_msg("static cost estimate", msg))
}
