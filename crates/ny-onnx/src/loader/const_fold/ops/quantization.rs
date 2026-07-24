// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::loader::numeric_cast::i64_to_f32_checked;
use crate::onnx_proto;
use crate::onnx_proto::attribute_type;
use crate::WeightStore;
use ndarray::{ArrayD, IxDyn};
use tracing::debug;

use super::super::FoldedTensor;

const DEFAULT_QUANT_RANGE: (i64, i64) = (0, u8::MAX as i64);

pub(super) fn try_fold(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    match node.op_type.as_str() {
        "DequantizeLinear" if node.input.len() >= 2 => try_fold_dequantize_linear(node, weights),
        "QuantizeLinear" if node.input.len() >= 2 => try_fold_quantize_linear(node, weights),
        _ => None,
    }
}

fn try_fold_dequantize_linear(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    let x_name = &node.input[0];
    let x = integer_or_float_f64(weights, x_name)?;
    let scale = weights.get(&node.input[1])?;
    if !scale.iter().all(|value| value.is_finite()) {
        return None;
    }
    let zero_point = node
        .input
        .get(2)
        .filter(|name| !name.is_empty())
        .and_then(|name| integer_or_float_f64(weights, name));

    let scale = quant_param_for_input(scale, x.shape(), node)?;
    let zero_point = match zero_point {
        Some(zero_point) => quant_param_for_input_f64(&zero_point, x.shape(), node)?,
        None => ArrayD::zeros(IxDyn(x.shape())),
    };

    let values: Vec<f32> = x
        .iter()
        .zip(zero_point.iter())
        .zip(scale.iter())
        .map(|((&x, &zero_point), &scale)| ((x - zero_point) * scale as f64) as f32)
        .collect();
    let result = ArrayD::from_shape_vec(IxDyn(x.shape()), values).ok()?;
    if !result.iter().all(|value| value.is_finite()) {
        return None;
    }
    debug!("Constant folded DequantizeLinear {}", node.name);
    Some(FoldedTensor::from_float(result))
}

fn try_fold_quantize_linear(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    let x = weights.get(&node.input[0])?;
    let scale = weights.get(&node.input[1])?;
    if !x.iter().all(|value| value.is_finite()) || !scale.iter().all(|value| *value > 0.0) {
        return None;
    }

    // Opset 21: the output type comes from `output_dtype` when y_zero_point is
    // absent; uint8 is only the default when NEITHER is supplied.
    let output_dtype_range = match node
        .attribute
        .iter()
        .find(|attr| attr.name == "output_dtype")
    {
        None => None,
        Some(attr) => {
            if attr.r#type != attribute_type::INT {
                return None;
            }
            // Unmodelled output dtypes (e.g. the float8 family, whose result
            // also depends on the `saturate` attribute) leave the node
            // unfolded rather than clamping to a guessed range.
            Some(quant_range_for_output_dtype(attr.i)?)
        }
    };

    let (zero_point, range) = match node.input.get(2).filter(|name| !name.is_empty()) {
        Some(name) => {
            let zero_point = integer_or_exact_float(weights, name)?;
            let range = weights.get_integer_range(name)?;
            // A y_zero_point whose type contradicts output_dtype is malformed;
            // leave the node unfolded.
            if output_dtype_range.is_some_and(|dtype_range| dtype_range != range) {
                return None;
            }
            (zero_point, range)
        }
        None => (
            ArrayD::zeros(IxDyn(&[])),
            output_dtype_range.unwrap_or(DEFAULT_QUANT_RANGE),
        ),
    };

    let scale = quant_param_for_input(scale, x.shape(), node)?;
    let zero_point = quant_param_for_input(&zero_point, x.shape(), node)?;
    let mut ints = Vec::with_capacity(x.len());
    for ((&value, &scale), &zero_point) in x.iter().zip(scale.iter()).zip(zero_point.iter()) {
        let rounded = round_ties_to_even(value / scale);
        if !rounded.is_finite() || !zero_point.is_finite() {
            return None;
        }
        let quantized = rounded + zero_point;
        if quantized < i64::MIN as f32 || quantized >= i64::MAX as f32 {
            return None;
        }
        let clamped = (quantized as i64).clamp(range.0, range.1);
        ints.push(clamped);
    }

    let integer_data = ArrayD::from_shape_vec(IxDyn(x.shape()), ints).ok()?;
    let float_values: Vec<f32> = integer_data
        .iter()
        .map(|&value| i64_to_f32_checked(value, "QuantizeLinear constant fold").ok())
        .collect::<Option<Vec<_>>>()?;
    let float_data = ArrayD::from_shape_vec(IxDyn(integer_data.shape()), float_values).ok()?;
    debug!("Constant folded QuantizeLinear {}", node.name);
    Some(FoldedTensor {
        float_data,
        integer_data: Some(integer_data),
        integer_range: Some(range),
    })
}

/// Clamp range implied by a QuantizeLinear `output_dtype` attribute value
/// (an ONNX TensorProto.DataType). Returns None for dtypes whose quantization
/// semantics this fold does not model exactly.
fn quant_range_for_output_dtype(dtype: i64) -> Option<(i64, i64)> {
    match dtype {
        2 => Some((0, u8::MAX as i64)),                // UINT8
        3 => Some((i8::MIN as i64, i8::MAX as i64)),   // INT8
        4 => Some((0, u16::MAX as i64)),               // UINT16
        5 => Some((i16::MIN as i64, i16::MAX as i64)), // INT16
        21 => Some((0, 15)),                           // UINT4
        22 => Some((-8, 7)),                           // INT4
        _ => None,
    }
}

fn integer_or_exact_float(weights: &WeightStore, name: &str) -> Option<ArrayD<f32>> {
    if let Some(integers) = weights.get_integers(name) {
        let values = integers
            .iter()
            .map(|&value| i64_to_f32_checked(value, "quantization constant fold").ok())
            .collect::<Option<Vec<_>>>()?;
        return ArrayD::from_shape_vec(IxDyn(integers.shape()), values).ok();
    }
    let floats = weights.get(name)?;
    floats
        .iter()
        .all(|value| value.is_finite() && value.fract() == 0.0)
        .then(|| floats.clone())
}

fn integer_or_float_f64(weights: &WeightStore, name: &str) -> Option<ArrayD<f64>> {
    if let Some(integers) = weights.get_integers(name) {
        let values = integers.iter().map(|&value| value as f64).collect();
        return ArrayD::from_shape_vec(IxDyn(integers.shape()), values).ok();
    }
    let floats = weights.get(name)?;
    floats
        .iter()
        .all(|value| value.is_finite())
        .then(|| floats.mapv(|value| value as f64))
}

fn quant_param_for_input(
    param: &ArrayD<f32>,
    input_shape: &[usize],
    node: &onnx_proto::NodeProto,
) -> Option<ArrayD<f32>> {
    if param.len() == 1 {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned());
    }
    if param.ndim() == input_shape.len() {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned());
    }
    if param.ndim() == 1 && !input_shape.is_empty() {
        let axis = quant_axis(node, input_shape.len())?;
        let mut shape = vec![1usize; input_shape.len()];
        shape[axis] = param.len();
        let reshaped = param.clone().into_shape_with_order(IxDyn(&shape)).ok()?;
        return reshaped
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned());
    }
    None
}

fn quant_param_for_input_f64(
    param: &ArrayD<f64>,
    input_shape: &[usize],
    node: &onnx_proto::NodeProto,
) -> Option<ArrayD<f64>> {
    if param.len() == 1 {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned());
    }
    if param.ndim() == input_shape.len() {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned());
    }
    if param.ndim() == 1 && !input_shape.is_empty() {
        let axis = quant_axis(node, input_shape.len())?;
        let mut shape = vec![1usize; input_shape.len()];
        shape[axis] = param.len();
        let reshaped = param.clone().into_shape_with_order(IxDyn(&shape)).ok()?;
        return reshaped
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned());
    }
    None
}

fn quant_axis(node: &onnx_proto::NodeProto, rank: usize) -> Option<usize> {
    let axis = node
        .attribute
        .iter()
        .find(|attr| attr.name == "axis")
        .map(|attr| attr.i)
        .unwrap_or(1);
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    usize::try_from(axis).ok().filter(|&axis| axis < rank)
}

fn round_ties_to_even(value: f32) -> f32 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}
