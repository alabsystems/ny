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
        "DequantizeLinear" if quantization_io_is_canonical(node) => {
            try_fold_dequantize_linear(node, weights)
        }
        "QuantizeLinear" if quantization_io_is_canonical(node) => {
            try_fold_quantize_linear(node, weights)
        }
        _ => None,
    }
}

fn quantization_io_is_canonical(node: &onnx_proto::NodeProto) -> bool {
    matches!(node.input.len(), 2 | 3)
        && node.input[0..2].iter().all(|name| !name.is_empty())
        && node.output.len() == 1
        && !node.output[0].is_empty()
}

/// Semantic subset implemented by the typed FLOAT32 evaluator below. Opset
/// legality is checked at the raw-model preflight; this local gate protects
/// direct unit/API callers and prevents unknown attributes from disappearing
/// when a constant node is folded away.
fn attributes_supported(node: &onnx_proto::NodeProto) -> bool {
    let mut seen = std::collections::HashSet::new();
    node.attribute.iter().all(|attribute| {
        if !seen.insert(attribute.name.as_str()) || attribute.r#type != attribute_type::INT {
            return false;
        }
        match (node.op_type.as_str(), attribute.name.as_str()) {
            ("QuantizeLinear" | "DequantizeLinear", "axis") => true,
            ("QuantizeLinear" | "DequantizeLinear", "block_size") => attribute.i_value() == 0,
            ("QuantizeLinear", "saturate") => matches!(attribute.i_value(), 0 | 1),
            ("QuantizeLinear", "output_dtype") => {
                attribute.i_value() == 0
                    || quant_range_for_output_dtype(attribute.i_value()).is_some()
            }
            ("QuantizeLinear", "precision") => matches!(attribute.i_value(), 0 | 1),
            ("DequantizeLinear", "output_dtype") => matches!(attribute.i_value(), 0 | 1),
            _ => false,
        }
    })
}

fn try_fold_dequantize_linear(
    node: &onnx_proto::NodeProto,
    weights: &WeightStore,
) -> Option<FoldedTensor> {
    if !attributes_supported(node) {
        return None;
    }
    let x_name = &node.input[0];
    let (x, x_range) = authenticated_quantized_integers(weights, x_name)?;
    let scale = weights.get(&node.input[1])?;
    if !scale.iter().all(|value| value.is_finite() && *value > 0.0) {
        return None;
    }
    let zero_point = match node.input.get(2).filter(|name| !name.is_empty()) {
        Some(name) => {
            // ONNX requires x and x_zero_point to have the same quantized type.
            // INT32 dequantization has no zero point.
            if x_range == (i32::MIN as i64, i32::MAX as i64) {
                return None;
            }
            let (values, range) = authenticated_quantized_integers(weights, name)?;
            if range != x_range {
                return None;
            }
            Some(values)
        }
        None => None,
    };

    let scale = quant_param_for_input(scale, x.shape(), node)?;
    let zero_point = match zero_point {
        Some(zero_point) => quant_param_for_input_i64(&zero_point, x.shape(), node)?,
        None => ArrayD::<i64>::zeros(IxDyn(x.shape())),
    };

    let values: Vec<f32> = x
        .iter()
        .zip(zero_point.iter())
        .zip(scale.iter())
        .map(|((&x, &zero_point), &scale)| dequantized_value_f32(x, zero_point, scale))
        .collect::<Option<Vec<_>>>()?;
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
    if !attributes_supported(node) {
        return None;
    }
    let x = quantize_input_at_float_precision(weights, &node.input[0])?;
    let scale = weights.get(&node.input[1])?;
    if !x.iter().all(|value| value.is_finite())
        || !scale.iter().all(|value| value.is_finite() && *value > 0.0)
    {
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
            if attr.i_value() == 0 {
                None
            } else {
                Some(quant_range_for_output_dtype(attr.i_value())?)
            }
        }
    };

    let (zero_point, range) = match node.input.get(2).filter(|name| !name.is_empty()) {
        Some(name) => {
            let (zero_point, range) = authenticated_quantized_integers(weights, name)?;
            if range == (i32::MIN as i64, i32::MAX as i64) {
                // INT32 is a legal QuantizeLinear input precision, not a legal
                // quantized output/zero-point type.
                return None;
            }
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
    let zero_point = quant_param_for_input_i64(&zero_point, x.shape(), node)?;
    let mut ints = Vec::with_capacity(x.len());
    for ((&value, &scale), &zero_point) in x.iter().zip(scale.iter()).zip(zero_point.iter()) {
        ints.push(certified_quantized_integer(
            value, scale, zero_point, range,
        )?);
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
        _ => None,
    }
}

fn authenticated_quantized_integers(
    weights: &WeightStore,
    name: &str,
) -> Option<(ArrayD<i64>, (i64, i64))> {
    let range = weights.get_integer_range(name)?;
    if !matches!(
        range,
        (0, 255) | (-128, 127) | (0, 65_535) | (-32_768, 32_767) | (-2_147_483_648, 2_147_483_647)
    ) {
        return None;
    }
    let values = weights.get_integers(name)?;
    values
        .iter()
        .all(|&value| value >= range.0 && value <= range.1)
        .then(|| (values.clone(), range))
}

fn quantize_input_at_float_precision(weights: &WeightStore, name: &str) -> Option<ArrayD<f32>> {
    match weights.get_integer_range(name) {
        Some(range) if range == (i32::MIN as i64, i32::MAX as i64) => {
            let integers = weights.get_integers(name)?;
            Some(integers.mapv(|value| value as f32))
        }
        Some(_) => None,
        None if weights.get_integers(name).is_none() => weights.get(name).cloned(),
        None => None,
    }
}

fn dequantized_value_f32(x: i64, zero_point: i64, scale: f32) -> Option<f32> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    // FLOAT DequantizeLinear executes the integer conversion, subtraction,
    // and multiplication at binary32 precision. Reproduce that typed program;
    // exact-real multiplication is a different semantics near rounding ties.
    let result = ((x as f32) - (zero_point as f32)) * scale;
    result.is_finite().then_some(result)
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
    // Same-rank parameters are blocked quantization, which requires replicated
    // block indexing rather than ndarray broadcasting. The local attribute gate
    // admits only block_size=0, so leave that distinct semantics unfolded.
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

fn quant_param_for_input_i64(
    param: &ArrayD<i64>,
    input_shape: &[usize],
    node: &onnx_proto::NodeProto,
) -> Option<ArrayD<i64>> {
    if param.len() == 1 {
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
        .map(|attr| attr.i_value())
        .unwrap_or(1);
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    usize::try_from(axis).ok().filter(|&axis| axis < rank)
}

fn certified_quantized_integer(
    value: f32,
    scale: f32,
    zero_point: i64,
    range: (i64, i64),
) -> Option<i64> {
    if !value.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    if zero_point < range.0 || zero_point > range.1 {
        return None;
    }

    let lower_ratio = range.0.checked_sub(zero_point)?;
    let upper_ratio = range.1.checked_sub(zero_point)?;
    // The scale dtype determines the division precision. NY admits only FLOAT
    // scales, so the binary32 quotient is rounded before ties-to-even integer
    // rounding. An exact-real midpoint comparison can disagree at this seam.
    let quotient = value / scale;
    if quotient.is_nan() {
        return None;
    }
    let rounded = quotient.round_ties_even();
    if rounded <= lower_ratio as f32 {
        return Some(range.0);
    }
    if rounded >= upper_ratio as f32 {
        return Some(range.1);
    }
    if !rounded.is_finite() || rounded < i64::MIN as f32 || rounded > i64::MAX as f32 {
        return None;
    }
    Some(
        (rounded as i64)
            .checked_add(zero_point)?
            .clamp(range.0, range.1),
    )
}
