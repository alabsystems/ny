// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_propagate::layers::LinearLayer;
use ny_propagate::Layer;
use tracing::debug;

use super::{i64_to_f32_checked, AttributeValue, ConvertContext, LayerSpec};

const DEFAULT_QUANT_RANGE: (i64, i64) = (0, u8::MAX as i64);

fn validate_quantization_attributes(spec: &LayerSpec) -> Result<()> {
    if !matches!(spec.inputs.len(), 2 | 3)
        || spec.inputs[0..2].iter().any(String::is_empty)
        || spec.outputs.len() != 1
        || spec.outputs[0].is_empty()
    {
        return Err(NyError::ModelLoad(format!(
            "{} {} requires 2 or 3 inputs and exactly one non-empty output",
            spec.layer_type, spec.name
        )));
    }

    let quantize = spec.layer_type == ny_core::LayerType::QuantizeLinear;
    for (name, value) in &spec.attributes {
        let supported = match (quantize, name.as_str(), value) {
            (_, "axis", AttributeValue::Int(_)) => true,
            (_, "block_size", AttributeValue::Int(0)) => true,
            (true, "saturate", AttributeValue::Int(value)) => matches!(value, 0 | 1),
            (true, "output_dtype", AttributeValue::Int(0)) => true,
            (true, "output_dtype", AttributeValue::Int(dtype)) => {
                quant_range_for_output_dtype(*dtype).is_some()
            }
            (true, "precision", AttributeValue::Int(value)) => matches!(value, 0 | 1),
            (false, "output_dtype", AttributeValue::Int(value)) => matches!(value, 0 | 1),
            _ => false,
        };
        if !supported {
            return Err(NyError::UnsupportedOp(format!(
                "{} {} has unsupported quantization attribute {name}={value:?}",
                spec.layer_type, spec.name
            )));
        }
    }
    Ok(())
}

impl ConvertContext<'_> {
    pub(crate) fn convert_dequantize_linear(&self, spec: &LayerSpec) -> Result<Layer> {
        validate_quantization_attributes(spec)?;
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "DequantizeLinear {} requires at least 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }

        let x_name = &spec.inputs[0];
        if self.constant_value(x_name).is_some() {
            return Err(NyError::UnsupportedOp(format!(
                "DequantizeLinear {} has constant input but was not constant-folded",
                spec.name
            )));
        }

        let input_shape = self.tensor_shape_usize_required(x_name, &spec.name)?;
        let scale = self.required_constant(&spec.inputs[1], "scale", &spec.name)?;
        let zero_point = spec
            .inputs
            .get(2)
            .filter(|name| !name.is_empty())
            .map(|name| self.required_integral_constant(name, "zero_point", &spec.name))
            .transpose()?;
        if zero_point
            .as_ref()
            .is_some_and(|values| values.iter().any(|value| *value != 0.0))
        {
            return Err(NyError::UnsupportedOp(format!(
                "DequantizeLinear {} with activation input and nonzero zero_point is unsupported: affine reassociation does not preserve typed FLOAT32 subtraction/multiplication rounding",
                spec.name
            )));
        }
        let (scale, bias) =
            dequant_scale_bias_for_input(&scale, zero_point.as_ref(), &input_shape, spec)?;

        let (last_axis_scale, last_axis_bias) =
            last_axis_affine(scale, bias, &input_shape, &spec.name)?;
        let features = last_axis_scale.len();
        let mut weight = Array2::<f32>::zeros((features, features));
        for (idx, value) in last_axis_scale.iter().copied().enumerate() {
            weight[[idx, idx]] = value;
        }
        debug!(
            "DequantizeLinear {} lowered to exact last-axis affine Linear({features}x{features})",
            spec.name
        );
        Ok(Layer::Linear(LinearLayer::new(
            weight,
            Some(last_axis_bias),
        )?))
    }

    pub(crate) fn convert_quantize_linear(&self, spec: &LayerSpec) -> Result<Layer> {
        validate_quantization_attributes(spec)?;
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "QuantizeLinear {} requires at least 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }
        if self.constant_value(&spec.inputs[0]).is_some() {
            return Err(NyError::UnsupportedOp(format!(
                "QuantizeLinear {} has constant input but was not constant-folded",
                spec.name
            )));
        }
        Err(NyError::UnsupportedOp(format!(
            "QuantizeLinear {} with activation input is unsupported: rounding is non-linear",
            spec.name
        )))
    }

    pub(crate) fn evaluate_dequantize_linear_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        validate_quantization_attributes(spec).ok()?;
        let (x, x_range) = self.authenticated_quantized_integer_value(&spec.inputs[0])?;
        let scale = self.constant_from_anywhere(&spec.inputs[1], evaluated_constants)?;
        if !scale.iter().all(|value| value.is_finite() && *value > 0.0) {
            return None;
        }
        let zero_point = match spec.inputs.get(2).filter(|name| !name.is_empty()) {
            Some(name) => {
                if x_range == (i32::MIN as i64, i32::MAX as i64) {
                    return None;
                }
                let (values, range) = self.authenticated_quantized_integer_value(name)?;
                (range == x_range).then_some(values)?
            }
            // The omitted zero point is a scalar quantization parameter.  Let
            // the common parameter broadcaster expand it; passing a same-rank
            // all-zero tensor here would be mistaken for unsupported blocked
            // quantization whenever x has more than one element.
            None => ArrayD::<i64>::from_elem(IxDyn(&[]), 0),
        };
        let scale = quant_param_for_input(&scale, x.shape(), spec).ok()?;
        let zero_point = quant_param_for_input_i64(&zero_point, x.shape(), spec).ok()?;
        let values: Vec<f32> = x
            .iter()
            .zip(zero_point.iter())
            .zip(scale.iter())
            .map(|((&x, &zero_point), &scale)| dequantized_value_f32(x, zero_point, scale))
            .collect::<Option<Vec<_>>>()?;
        let result = ArrayD::from_shape_vec(IxDyn(x.shape()), values).ok()?;
        result
            .iter()
            .all(|value| value.is_finite())
            .then_some(result)
    }

    pub(crate) fn evaluate_quantize_linear_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        validate_quantization_attributes(spec).ok()?;
        let x = self.quantize_input_at_float_precision(&spec.inputs[0], evaluated_constants)?;
        let scale = self.constant_from_anywhere(&spec.inputs[1], evaluated_constants)?;
        if !x.iter().all(|value| value.is_finite())
            || !scale.iter().all(|value| value.is_finite() && *value > 0.0)
        {
            return None;
        }
        let output_dtype_range = quant_output_dtype_range(spec).ok()?;
        let (zero_point, range) = match spec.inputs.get(2).filter(|name| !name.is_empty()) {
            Some(name) => {
                let (zp, range) = self.authenticated_quantized_integer_value(name)?;
                if range == (i32::MIN as i64, i32::MAX as i64) {
                    return None;
                }
                // A y_zero_point whose type contradicts output_dtype is
                // malformed; leave the node unevaluated.
                if output_dtype_range.is_some_and(|dtype_range| dtype_range != range) {
                    return None;
                }
                (zp, range)
            }
            None => (
                ArrayD::zeros(IxDyn(&[])),
                output_dtype_range.unwrap_or(DEFAULT_QUANT_RANGE),
            ),
        };
        let scale = quant_param_for_input(&scale, x.shape(), spec).ok()?;
        let zero_point = quant_param_for_input_i64(&zero_point, x.shape(), spec).ok()?;

        let values: Vec<f32> = x
            .iter()
            .zip(scale.iter())
            .zip(zero_point.iter())
            .map(|((&value, &scale), &zero_point)| {
                let quantized = certified_quantized_integer(value, scale, zero_point, range)?;
                i64_to_f32_checked(quantized, "QuantizeLinear build const-eval").ok()
            })
            .collect::<Option<Vec<_>>>()?;
        ArrayD::from_shape_vec(IxDyn(x.shape()), values).ok()
    }

    fn tensor_shape_usize_required(&self, name: &str, spec_name: &str) -> Result<Vec<usize>> {
        let shape = self.tensor_shapes.get(name).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "DequantizeLinear {spec_name} activation input '{name}' has unknown shape"
            ))
        })?;
        shape
            .iter()
            .map(|&dim| {
                if dim <= 0 {
                    return Err(NyError::UnsupportedOp(format!(
                        "DequantizeLinear {spec_name} input '{name}' has dynamic/non-positive shape {shape:?}"
                    )));
                }
                usize::try_from(dim).map_err(|_| {
                    NyError::UnsupportedOp(format!(
                        "DequantizeLinear {spec_name} input '{name}' has dynamic/non-positive shape {shape:?}"
                    ))
                })
            })
            .collect()
    }

    fn required_constant(&self, name: &str, label: &str, spec_name: &str) -> Result<ArrayD<f32>> {
        self.constant_value(name).ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "DequantizeLinear {spec_name} requires constant {label} input '{name}'"
            ))
        })
    }

    fn required_integral_constant(
        &self,
        name: &str,
        label: &str,
        spec_name: &str,
    ) -> Result<ArrayD<f32>> {
        self.integral_constant_value(name, self.evaluated_constants)
            .ok_or_else(|| {
                NyError::UnsupportedOp(format!(
                    "DequantizeLinear {spec_name} requires integral constant {label} input '{name}'"
                ))
            })
    }

    fn constant_from_anywhere(
        &self,
        name: &str,
        evaluated_constants: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        self.weights
            .get(name)
            .cloned()
            .or_else(|| evaluated_constants.get(name).cloned())
    }

    fn integral_constant_value(
        &self,
        name: &str,
        _evaluated_constants: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        let (integers, _) = self.authenticated_quantized_integer_value(name)?;
        let values = integers
            .iter()
            .map(|&value| i64_to_f32_checked(value, "quantization build const-eval").ok())
            .collect::<Option<Vec<_>>>()?;
        ArrayD::from_shape_vec(IxDyn(integers.shape()), values).ok()
    }

    fn authenticated_quantized_integer_value(
        &self,
        name: &str,
    ) -> Option<(ArrayD<i64>, (i64, i64))> {
        let range = self.weights.get_integer_range(name)?;
        if !matches!(
            range,
            (0, 255)
                | (-128, 127)
                | (0, 65_535)
                | (-32_768, 32_767)
                | (-2_147_483_648, 2_147_483_647)
        ) {
            return None;
        }
        let integers = self.weights.get_integers(name)?;
        integers
            .iter()
            .all(|&value| value >= range.0 && value <= range.1)
            .then(|| (integers.clone(), range))
    }

    fn quantize_input_at_float_precision(
        &self,
        name: &str,
        evaluated_constants: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        match self.weights.get_integer_range(name) {
            Some(range) if range == (i32::MIN as i64, i32::MAX as i64) => {
                let integers = self.weights.get_integers(name)?;
                Some(integers.mapv(|value| value as f32))
            }
            Some(_) => None,
            None if self.weights.get_integers(name).is_none() => {
                self.constant_from_anywhere(name, evaluated_constants)
            }
            None => None,
        }
    }
}

/// Clamp range implied by a QuantizeLinear `output_dtype` attribute value
/// (an ONNX TensorProto.DataType). Returns None for dtypes whose quantization
/// semantics this conversion does not model exactly.
fn quant_range_for_output_dtype(dtype: i64) -> Option<(i64, i64)> {
    match dtype {
        2 => Some((0, u8::MAX as i64)),                // UINT8
        3 => Some((i8::MIN as i64, i8::MAX as i64)),   // INT8
        4 => Some((0, u16::MAX as i64)),               // UINT16
        5 => Some((i16::MIN as i64, i16::MAX as i64)), // INT16
        _ => None,
    }
}

/// Clamp range from a QuantizeLinear `output_dtype` attribute, when present.
///
/// Opset 21: the output type comes from `output_dtype` when y_zero_point is
/// absent; uint8 is only the default when NEITHER is supplied. Unmodelled
/// output dtypes (e.g. the float8 family, whose result also depends on the
/// `saturate` attribute) are rejected rather than clamped to a guessed range.
fn quant_output_dtype_range(spec: &LayerSpec) -> Result<Option<(i64, i64)>> {
    match spec.attributes.get("output_dtype") {
        None => Ok(None),
        Some(AttributeValue::Int(0)) => Ok(None),
        Some(AttributeValue::Int(dtype)) => match quant_range_for_output_dtype(*dtype) {
            Some(range) => Ok(Some(range)),
            None => Err(NyError::UnsupportedOp(format!(
                "{} {} output_dtype {} has unmodelled quantization semantics",
                spec.layer_type, spec.name, dtype
            ))),
        },
        Some(other) => Err(NyError::ModelLoad(format!(
            "{} {} has invalid output_dtype attribute {:?}",
            spec.layer_type, spec.name, other
        ))),
    }
}

fn dequant_scale_bias_for_input(
    scale: &ArrayD<f32>,
    zero_point: Option<&ArrayD<f32>>,
    input_shape: &[usize],
    spec: &LayerSpec,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    if !scale.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Err(NyError::InvalidSpec(format!(
            "DequantizeLinear {} has non-finite scale",
            spec.name
        )));
    }
    let scale = quant_param_for_input(scale, input_shape, spec)?;
    let zero_point = match zero_point {
        Some(zero_point) => quant_param_for_input(zero_point, input_shape, spec)?,
        None => ArrayD::zeros(IxDyn(input_shape)),
    };
    let bias_values = scale
        .iter()
        .zip(zero_point.iter())
        .map(|(&scale, &zero_point)| exact_f32_product(scale, zero_point).map(|value| -value))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "DequantizeLinear {} affine bias is not exactly representable as binary32",
                spec.name
            ))
        })?;
    let bias = ArrayD::from_shape_vec(scale.raw_dim(), bias_values).map_err(|error| {
        NyError::InvalidSpec(format!(
            "DequantizeLinear {} could not materialize affine bias: {error}",
            spec.name
        ))
    })?;
    Ok((scale, bias))
}

fn quant_param_for_input(
    param: &ArrayD<f32>,
    input_shape: &[usize],
    spec: &LayerSpec,
) -> Result<ArrayD<f32>> {
    if param.len() == 1 {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned())
            .ok_or_else(|| quant_broadcast_error(param, input_shape, spec));
    }
    // Same-rank parameters are blocked quantization, whose replicated block
    // indexing is distinct from ndarray broadcasting. NY's certified subset
    // intentionally supports only scalar and 1-D per-axis parameters.
    if param.ndim() == 1 && !input_shape.is_empty() {
        let axis = quant_axis(spec, input_shape.len())?;
        let mut shape = vec![1usize; input_shape.len()];
        shape[axis] = param.len();
        let reshaped = param
            .clone()
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|_| quant_broadcast_error(param, input_shape, spec))?;
        return reshaped
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned())
            .ok_or_else(|| quant_broadcast_error(param, input_shape, spec));
    }
    Err(quant_broadcast_error(param, input_shape, spec))
}

fn quant_param_for_input_i64(
    param: &ArrayD<i64>,
    input_shape: &[usize],
    spec: &LayerSpec,
) -> Result<ArrayD<i64>> {
    if param.len() == 1 {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned())
            .ok_or_else(|| quant_broadcast_error_i64(param, input_shape, spec));
    }
    if param.ndim() == 1 && !input_shape.is_empty() {
        let axis = quant_axis(spec, input_shape.len())?;
        let mut shape = vec![1usize; input_shape.len()];
        shape[axis] = param.len();
        let reshaped = param
            .clone()
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|_| quant_broadcast_error_i64(param, input_shape, spec))?;
        return reshaped
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned())
            .ok_or_else(|| quant_broadcast_error_i64(param, input_shape, spec));
    }
    Err(quant_broadcast_error_i64(param, input_shape, spec))
}

fn quant_broadcast_error(param: &ArrayD<f32>, input_shape: &[usize], spec: &LayerSpec) -> NyError {
    NyError::UnsupportedOp(format!(
        "{} {} quantization parameter shape {:?} cannot broadcast to input shape {:?}",
        spec.layer_type,
        spec.name,
        param.shape(),
        input_shape
    ))
}

fn quant_broadcast_error_i64(
    param: &ArrayD<i64>,
    input_shape: &[usize],
    spec: &LayerSpec,
) -> NyError {
    NyError::UnsupportedOp(format!(
        "{} {} quantization parameter shape {:?} cannot broadcast to input shape {:?}",
        spec.layer_type,
        spec.name,
        param.shape(),
        input_shape
    ))
}

fn quant_axis(spec: &LayerSpec, rank: usize) -> Result<usize> {
    let axis = match spec.attributes.get("axis") {
        Some(AttributeValue::Int(axis)) => *axis,
        Some(other) => {
            return Err(NyError::ModelLoad(format!(
                "{} {} has invalid axis attribute {:?}",
                spec.layer_type, spec.name, other
            )));
        }
        None => 1,
    };
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    usize::try_from(axis)
        .ok()
        .filter(|&axis| axis < rank)
        .ok_or_else(|| {
            NyError::UnsupportedOp(format!(
                "{} {} axis {} is out of bounds for rank {}",
                spec.layer_type, spec.name, axis, rank
            ))
        })
}

fn last_axis_affine(
    scale: ArrayD<f32>,
    bias: ArrayD<f32>,
    input_shape: &[usize],
    spec_name: &str,
) -> Result<(Array1<f32>, Array1<f32>)> {
    let Some(&last_dim) = input_shape.last() else {
        return Err(NyError::UnsupportedOp(format!(
            "DequantizeLinear {spec_name} rank-0 activation input is unsupported"
        )));
    };
    let scale_rows = scale
        .view()
        .into_shape_with_order((scale.len() / last_dim, last_dim))
        .map_err(|_| {
            NyError::UnsupportedOp(format!(
                "DequantizeLinear {spec_name} affine map cannot be viewed over last axis"
            ))
        })?;
    let bias_rows = bias
        .view()
        .into_shape_with_order((bias.len() / last_dim, last_dim))
        .map_err(|_| {
            NyError::UnsupportedOp(format!(
                "DequantizeLinear {spec_name} affine bias cannot be viewed over last axis"
            ))
        })?;
    let first_scale = scale_rows.index_axis(Axis(0), 0).to_owned();
    let first_bias = bias_rows.index_axis(Axis(0), 0).to_owned();
    if scale_rows
        .axis_iter(Axis(0))
        .all(|row| row.iter().zip(first_scale.iter()).all(|(a, b)| a == b))
        && bias_rows
            .axis_iter(Axis(0))
            .all(|row| row.iter().zip(first_bias.iter()).all(|(a, b)| a == b))
    {
        return Ok((first_scale, first_bias));
    }
    Err(NyError::UnsupportedOp(format!(
        "DequantizeLinear {spec_name} activation affine is not constant along the last axis"
    )))
}

fn exact_f32_product(lhs: f32, rhs: f32) -> Option<f32> {
    if !lhs.is_finite() || !rhs.is_finite() {
        return None;
    }
    let exact = (lhs as f64) * (rhs as f64);
    let rounded = lhs * rhs;
    (rounded.is_finite() && rounded as f64 == exact).then_some(rounded)
}

fn dequantized_value_f32(x: i64, zero_point: i64, scale: f32) -> Option<f32> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let result = ((x as f32) - (zero_point as f32)) * scale;
    result.is_finite().then_some(result)
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
    // FLOAT scale means ONNX rounds the division to binary32 before applying
    // ties-to-even. Exact-real midpoint comparisons disagree at this seam.
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

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_core::LayerType;
    use std::collections::{HashMap, HashSet};

    use super::ConvertContext;
    use crate::{AttributeValue, LayerSpec, WeightStore};

    fn scalar_f32(value: f32) -> ArrayD<f32> {
        ArrayD::from_shape_vec(IxDyn(&[]), vec![value]).unwrap()
    }

    fn quantize_spec(
        inputs: Vec<&str>,
        extra_attrs: Vec<(&str, AttributeValue)>,
        qdq_relaxation: bool,
    ) -> LayerSpec {
        let mut attributes = HashMap::new();
        if qdq_relaxation {
            attributes.insert("qdq_relaxation".to_string(), AttributeValue::Int(1));
        }
        for (name, value) in extra_attrs {
            attributes.insert(name.to_string(), value);
        }
        LayerSpec {
            name: "quantize".to_string(),
            layer_type: LayerType::QuantizeLinear,
            inputs: inputs.into_iter().map(str::to_string).collect(),
            outputs: vec!["out".to_string()],
            weights: None,
            attributes,
        }
    }

    fn scale_only_weights() -> WeightStore {
        let mut weights = WeightStore::new();
        weights.insert("scale".to_string(), scalar_f32(1.0));
        weights
    }

    fn int8_zero_point_weights() -> WeightStore {
        let mut weights = scale_only_weights();
        weights.insert_integers(
            "zp".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![0i64]).unwrap(),
        );
        weights.insert_integer_range("zp".to_string(), i8::MIN as i64, i8::MAX as i64);
        weights
    }

    #[ntest::timeout(10000)]
    #[test]
    fn direct_qdq_relaxation_attribute_is_rejected_until_float32_envelope_is_certified() {
        let weights = scale_only_weights();
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(
            vec!["x", "scale"],
            vec![("output_dtype", AttributeValue::Int(3))],
            true,
        );
        let error = ctx
            .convert_quantize_linear(&spec)
            .expect_err("synthetic qdq_relaxation must not bypass typed Q/DQ semantics");
        assert!(error.to_string().contains("qdq_relaxation"), "{error}");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn quantize_linear_const_eval_output_dtype_int8_clamps_to_signed_range() {
        let mut weights = scale_only_weights();
        weights.insert(
            "x".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-200.0, 100.0]).unwrap(),
        );
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(
            vec!["x", "scale"],
            vec![("output_dtype", AttributeValue::Int(3))],
            false,
        );
        let evaluated = HashMap::new();
        let output = ctx
            .evaluate_quantize_linear_constant(&spec, &evaluated)
            .expect("int8 constant quantization must evaluate");
        assert_eq!(
            output.iter().copied().collect::<Vec<_>>(),
            vec![-128.0, 100.0],
            "int8 output_dtype must clamp to [-128, 127], not the uint8 default"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn quantize_linear_const_eval_defaults_to_uint8_range() {
        let mut weights = scale_only_weights();
        weights.insert(
            "x".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![-200.0, 100.0]).unwrap(),
        );
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(vec!["x", "scale"], vec![], false);
        let evaluated = HashMap::new();
        let output = ctx
            .evaluate_quantize_linear_constant(&spec, &evaluated)
            .expect("default uint8 constant quantization must evaluate");
        assert_eq!(output.iter().copied().collect::<Vec<_>>(), vec![0.0, 100.0]);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn dequantize_linear_const_eval_broadcasts_omitted_zero_point() {
        for shape in [vec![2], vec![2, 2]] {
            let len = shape.iter().product();
            let mut weights = scale_only_weights();
            weights.insert_integers(
                "x".to_string(),
                ArrayD::from_shape_vec(IxDyn(&shape), (1..=i64::try_from(len).unwrap()).collect())
                    .unwrap(),
            );
            weights.insert_integer_range("x".to_string(), i8::MIN as i64, i8::MAX as i64);
            weights.insert("scale".to_string(), scalar_f32(0.25));
            let (shapes, constants) = (HashMap::new(), HashSet::new());
            let ctx = ConvertContext::new(&weights, &shapes, &constants);
            let spec = LayerSpec {
                name: "dequantize".to_string(),
                layer_type: LayerType::DequantizeLinear,
                inputs: vec!["x".to_string(), "scale".to_string()],
                outputs: vec!["out".to_string()],
                weights: None,
                attributes: HashMap::new(),
            };

            let output = ctx
                .evaluate_dequantize_linear_constant(&spec, &HashMap::new())
                .expect("omitted zero point must broadcast as scalar zero");
            assert_eq!(output.shape(), shape.as_slice());
            assert_eq!(
                output.iter().copied().collect::<Vec<_>>(),
                (1..=len)
                    .map(|value| value as f32 * 0.25)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn quantize_linear_const_eval_unmodelled_output_dtype_not_evaluated() {
        for dtype in [17, 21, 22] {
            let mut weights = scale_only_weights();
            weights.insert("x".to_string(), scalar_f32(1.0));
            let (shapes, constants) = (HashMap::new(), HashSet::new());
            let ctx = ConvertContext::new(&weights, &shapes, &constants);
            let spec = quantize_spec(
                vec!["x", "scale"],
                vec![("output_dtype", AttributeValue::Int(dtype))],
                false,
            );
            let evaluated = HashMap::new();
            assert!(
                ctx.evaluate_quantize_linear_constant(&spec, &evaluated)
                    .is_none(),
                "unmodelled output_dtype {dtype} must not be constant-evaluated"
            );
        }
    }

    #[ntest::timeout(10000)]
    #[test]
    fn quantize_linear_const_eval_output_dtype_zero_point_mismatch_not_evaluated() {
        let mut weights = int8_zero_point_weights();
        weights.insert("x".to_string(), scalar_f32(1.0));
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(
            vec!["x", "scale", "zp"],
            vec![("output_dtype", AttributeValue::Int(2))],
            false,
        );
        let evaluated = HashMap::new();
        assert!(
            ctx.evaluate_quantize_linear_constant(&spec, &evaluated)
                .is_none(),
            "output_dtype/zero_point range mismatch must not be constant-evaluated"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn quantize_linear_const_eval_rounding_seam_uses_float32_division() {
        let mut weights = scale_only_weights();
        weights.insert("scale".to_string(), scalar_f32(0.3));
        weights.insert("x".to_string(), scalar_f32(0.750_000_06));
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(vec!["x", "scale", ""], vec![], false);
        let output = ctx
            .evaluate_quantize_linear_constant(&spec, &HashMap::new())
            .expect("FLOAT32 quantization should evaluate");
        assert_eq!(output.iter().copied().collect::<Vec<_>>(), vec![2.0]);
    }

    #[ntest::timeout(10000)]
    #[test]
    fn quantize_linear_const_eval_bare_float_zero_point_is_rejected() {
        let mut weights = scale_only_weights();
        weights.insert("x".to_string(), scalar_f32(1.0));
        weights.insert("zp".to_string(), scalar_f32(0.0));
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(vec!["x", "scale", "zp"], vec![], false);
        assert!(
            ctx.evaluate_quantize_linear_constant(&spec, &HashMap::new())
                .is_none(),
            "an integral-looking FLOAT is not integer dtype evidence"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn quantize_linear_const_eval_int32_zero_point_is_rejected() {
        let mut weights = scale_only_weights();
        weights.insert("x".to_string(), scalar_f32(1.0));
        weights.insert_integers(
            "zp".to_string(),
            ArrayD::from_shape_vec(IxDyn(&[]), vec![0]).unwrap(),
        );
        weights.insert_integer_range("zp".to_string(), i32::MIN as i64, i32::MAX as i64);
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(vec!["x", "scale", "zp"], vec![], false);
        assert!(ctx
            .evaluate_quantize_linear_constant(&spec, &HashMap::new())
            .is_none());
    }
}
