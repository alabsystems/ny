// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Per-layer traffic accounting and timing-family classification helpers.

use std::mem::size_of;

use ndarray::ArrayD;
use ny_core::LayerType;

use super::lookup::ShapeLookup;
use super::CostError;
use crate::OnnxModel;

const F32_BYTES: u64 = size_of::<f32>() as u64;

pub(super) fn activation_input_shapes(
    model: &OnnxModel,
    lookup: &ShapeLookup<'_>,
    layer: &crate::LayerSpec,
) -> Result<Vec<Vec<usize>>, CostError> {
    activation_input_names(model, layer)
        .into_iter()
        .map(|name| lookup.tensor_shape(name))
        .collect()
}

pub(super) fn activation_input_names<'a>(
    model: &OnnxModel,
    layer: &'a crate::LayerSpec,
) -> Vec<&'a String> {
    let input_count = if layer_type_embeds_parameter_inputs(&layer.layer_type) {
        layer.inputs.len().min(1)
    } else {
        layer.inputs.len()
    };
    layer.inputs[..input_count]
        .iter()
        .filter(|name| is_runtime_tensor(model, name))
        .collect()
}

pub(super) fn activation_input_bytes(
    model: &OnnxModel,
    lookup: &ShapeLookup<'_>,
    layer: &crate::LayerSpec,
) -> Result<u64, CostError> {
    activation_input_shapes(model, lookup, layer)?
        .iter()
        .try_fold(0_u64, |acc, shape| {
            checked_add(
                acc,
                bytes_for_shape(shape)?,
                "activation input byte count overflow",
            )
        })
}

pub(super) fn parameter_input_tensors<'a>(
    model: &'a OnnxModel,
    layer: &'a crate::LayerSpec,
) -> Vec<&'a ArrayD<f32>> {
    layer
        .inputs
        .iter()
        .filter_map(|name| model.weights.get(name))
        .collect()
}

pub(super) fn parameter_input_bytes(
    model: &OnnxModel,
    layer: &crate::LayerSpec,
) -> Result<u64, CostError> {
    parameter_input_tensors(model, layer)
        .into_iter()
        .try_fold(0_u64, |acc, weight| {
            checked_add(
                acc,
                checked_mul(
                    weight.len() as u64,
                    F32_BYTES,
                    "parameter input byte count overflow",
                )?,
                "parameter input byte count overflow",
            )
        })
}

pub(super) fn timing_family(
    layer_type: &LayerType,
    layer_name: &str,
) -> Result<&'static str, CostError> {
    let family = match layer_type {
        LayerType::Linear | LayerType::MatMul => "dense_mac",
        LayerType::Conv1d
        | LayerType::Conv2d
        | LayerType::ConvTranspose1d
        | LayerType::ConvTranspose2d => "convolution",
        LayerType::AveragePool
        | LayerType::MaxPool
        | LayerType::ReduceMean
        | LayerType::ReduceSum => "reduction",
        LayerType::Softmax | LayerType::CausalSoftmax | LayerType::LogSoftmax => "softmax",
        LayerType::LayerNorm
        | LayerType::RMSNorm
        | LayerType::InstanceNorm
        | LayerType::GroupNorm
        | LayerType::BatchNorm
        | LayerType::AdaIN => "normalization",
        LayerType::SiLU
        | LayerType::Sigmoid
        | LayerType::Tanh
        | LayerType::Softplus
        | LayerType::Exp
        | LayerType::Log
        | LayerType::Sin
        | LayerType::Cos
        | LayerType::Tan
        | LayerType::Arctan
        | LayerType::Erf
        | LayerType::Mish
        | LayerType::GELU
        | LayerType::ReLU
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
        | LayerType::RoPE => "elementwise",
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
        | LayerType::Where => "shape_only",
        other => {
            return Err(CostError::propagation_msg(
                "static cost estimate",
                format!(
                    "unsupported timing family classification for layer type {} in layer '{}'",
                    other, layer_name
                ),
            ))
        }
    };

    Ok(family)
}

pub(super) fn is_runtime_tensor(model: &OnnxModel, tensor_name: &str) -> bool {
    // Empty names are ONNX optional-input sentinels (unconnected inputs).
    !tensor_name.is_empty()
        && !model.weights.contains_key(tensor_name)
        && !model.constant_tensors().contains(tensor_name)
}

fn layer_type_embeds_parameter_inputs(layer_type: &LayerType) -> bool {
    matches!(
        layer_type,
        LayerType::LayerNorm
            | LayerType::RMSNorm
            | LayerType::InstanceNorm
            | LayerType::GroupNorm
            | LayerType::BatchNorm
    )
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
