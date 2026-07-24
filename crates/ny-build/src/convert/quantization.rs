// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_propagate::layers::{LinearLayer, QdqPerturbationLayer};
use ny_propagate::Layer;
use tracing::debug;

use super::{i64_to_f32_checked, AttributeValue, ConvertContext, LayerSpec};

const DEFAULT_QUANT_RANGE: (i64, i64) = (0, u8::MAX as i64);

impl ConvertContext<'_> {
    pub(crate) fn convert_dequantize_linear(&self, spec: &LayerSpec) -> Result<Layer> {
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
        if spec.inputs.len() < 2 {
            return Err(NyError::ModelLoad(format!(
                "QuantizeLinear {} requires at least 2 inputs, got {}",
                spec.name,
                spec.inputs.len()
            )));
        }
        if spec.attributes.contains_key("qdq_relaxation") {
            return self.convert_qdq_relaxation(spec);
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

    fn convert_qdq_relaxation(&self, spec: &LayerSpec) -> Result<Layer> {
        let scale = self.required_scalar_constant(&spec.inputs[1], "scale", &spec.name)?;
        let output_dtype_range = quant_output_dtype_range(spec)?;
        let (zero_point, range) = match spec.inputs.get(2).filter(|name| !name.is_empty()) {
            Some(name) => {
                let zero_point =
                    self.required_scalar_integral_constant(name, "zero_point", &spec.name)?;
                let range = self.weights.get_integer_range(name).ok_or_else(|| {
                    NyError::UnsupportedOp(format!(
                        "QDQ relaxation {} requires integer range for zero_point '{}'",
                        spec.name, name
                    ))
                })?;
                // A y_zero_point whose type contradicts output_dtype is malformed.
                if let Some(dtype_range) = output_dtype_range {
                    if dtype_range != range {
                        return Err(NyError::ModelLoad(format!(
                            "QDQ relaxation {} output_dtype range {:?} contradicts zero_point '{}' range {:?}",
                            spec.name, dtype_range, name, range
                        )));
                    }
                }
                (zero_point, range)
            }
            None => (0.0, output_dtype_range.unwrap_or(DEFAULT_QUANT_RANGE)),
        };
        let qmin = i64_to_f32_checked(range.0, "QDQ relaxation qmin")?;
        let qmax = i64_to_f32_checked(range.1, "QDQ relaxation qmax")?;
        Ok(Layer::QdqPerturbation(QdqPerturbationLayer::new(
            scale, zero_point, qmin, qmax,
        )?))
    }

    pub(crate) fn evaluate_dequantize_linear_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        let x = self.integral_constant_value_f64(&spec.inputs[0], evaluated_constants)?;
        let scale = self.constant_from_anywhere(&spec.inputs[1], evaluated_constants)?;
        let zero_point = spec
            .inputs
            .get(2)
            .filter(|name| !name.is_empty())
            .and_then(|name| self.integral_constant_value_f64(name, evaluated_constants));
        let scale = quant_param_for_input(&scale, x.shape(), spec).ok()?;
        let zero_point = match zero_point {
            Some(zero_point) => quant_param_for_input_f64(&zero_point, x.shape(), spec).ok()?,
            None => ArrayD::zeros(IxDyn(x.shape())),
        };
        let values: Vec<f32> = x
            .iter()
            .zip(zero_point.iter())
            .zip(scale.iter())
            .map(|((&x, &zero_point), &scale)| ((x - zero_point) * scale as f64) as f32)
            .collect();
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
        let x = self.constant_from_anywhere(&spec.inputs[0], evaluated_constants)?;
        let scale = self.constant_from_anywhere(&spec.inputs[1], evaluated_constants)?;
        if !x.iter().all(|value| value.is_finite()) || !scale.iter().all(|value| *value > 0.0) {
            return None;
        }
        let output_dtype_range = quant_output_dtype_range(spec).ok()?;
        let (zero_point, range) = match spec.inputs.get(2).filter(|name| !name.is_empty()) {
            Some(name) => {
                let zp = self.integral_constant_value(name, evaluated_constants)?;
                let range = self.weights.get_integer_range(name)?;
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
        let zero_point = quant_param_for_input(&zero_point, x.shape(), spec).ok()?;

        let values: Vec<f32> = x
            .iter()
            .zip(scale.iter())
            .zip(zero_point.iter())
            .map(|((&value, &scale), &zero_point)| {
                let rounded = round_ties_to_even(value / scale);
                if !rounded.is_finite() || !zero_point.is_finite() {
                    return None;
                }
                let quantized = rounded + zero_point;
                if quantized < i64::MIN as f32 || quantized >= i64::MAX as f32 {
                    return None;
                }
                let clamped = (quantized as i64).clamp(range.0, range.1);
                i64_to_f32_checked(clamped, "QuantizeLinear build const-eval").ok()
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

    fn required_scalar_constant(&self, name: &str, label: &str, spec_name: &str) -> Result<f32> {
        let value = self.required_constant(name, label, spec_name)?;
        if value.len() != 1 {
            return Err(NyError::UnsupportedOp(format!(
                "QDQ relaxation {spec_name} supports only scalar {label}, got shape {:?}",
                value.shape()
            )));
        }
        Ok(value.iter().copied().next().unwrap())
    }

    fn required_scalar_integral_constant(
        &self,
        name: &str,
        label: &str,
        spec_name: &str,
    ) -> Result<f32> {
        let value = self.required_integral_constant(name, label, spec_name)?;
        if value.len() != 1 {
            return Err(NyError::UnsupportedOp(format!(
                "QDQ relaxation {spec_name} supports only scalar {label}, got shape {:?}",
                value.shape()
            )));
        }
        Ok(value.iter().copied().next().unwrap())
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
        evaluated_constants: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if let Some(integers) = self.weights.get_integers(name) {
            let values = integers
                .iter()
                .map(|&value| i64_to_f32_checked(value, "quantization build const-eval").ok())
                .collect::<Option<Vec<_>>>()?;
            return ArrayD::from_shape_vec(IxDyn(integers.shape()), values).ok();
        }
        let floats = self.constant_from_anywhere(name, evaluated_constants)?;
        floats
            .iter()
            .all(|value| value.is_finite() && value.fract() == 0.0)
            .then_some(floats)
    }

    fn integral_constant_value_f64(
        &self,
        name: &str,
        evaluated_constants: &std::collections::HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f64>> {
        if let Some(integers) = self.weights.get_integers(name) {
            let values = integers.iter().map(|&value| value as f64).collect();
            return ArrayD::from_shape_vec(IxDyn(integers.shape()), values).ok();
        }
        let floats = self.constant_from_anywhere(name, evaluated_constants)?;
        floats
            .iter()
            .all(|value| value.is_finite() && value.fract() == 0.0)
            .then(|| floats.mapv(|value| value as f64))
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
        21 => Some((0, 15)),                           // UINT4
        22 => Some((-8, 7)),                           // INT4
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
    if !scale.iter().all(|value| value.is_finite()) {
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
    let bias = -(&scale * &zero_point);
    if !bias.iter().all(|value| value.is_finite()) {
        return Err(NyError::InvalidSpec(format!(
            "DequantizeLinear {} produced non-finite affine bias",
            spec.name
        )));
    }
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
    if param.ndim() == input_shape.len() {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned())
            .ok_or_else(|| quant_broadcast_error(param, input_shape, spec));
    }
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

fn quant_param_for_input_f64(
    param: &ArrayD<f64>,
    input_shape: &[usize],
    spec: &LayerSpec,
) -> Result<ArrayD<f64>> {
    if param.len() == 1 {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned())
            .ok_or_else(|| quant_broadcast_error_f64(param, input_shape, spec));
    }
    if param.ndim() == input_shape.len() {
        return param
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned())
            .ok_or_else(|| quant_broadcast_error_f64(param, input_shape, spec));
    }
    if param.ndim() == 1 && !input_shape.is_empty() {
        let axis = quant_axis(spec, input_shape.len())?;
        let mut shape = vec![1usize; input_shape.len()];
        shape[axis] = param.len();
        let reshaped = param
            .clone()
            .into_shape_with_order(IxDyn(&shape))
            .map_err(|_| quant_broadcast_error_f64(param, input_shape, spec))?;
        return reshaped
            .broadcast(IxDyn(input_shape))
            .map(|view| view.into_owned())
            .ok_or_else(|| quant_broadcast_error_f64(param, input_shape, spec));
    }
    Err(quant_broadcast_error_f64(param, input_shape, spec))
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

fn quant_broadcast_error_f64(
    param: &ArrayD<f64>,
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

#[cfg(test)]
mod tests {
    use ndarray::{ArrayD, IxDyn};
    use ny_core::{LayerType, NyError};
    use ny_propagate::{BoundPropagation, Layer};
    use ny_tensor::BoundedTensor;
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

    fn propagate_point(layer: &Layer, value: f32) -> ny_core::Result<BoundedTensor> {
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![value]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![value]).unwrap(),
        )
        .expect("point input bounds are valid");
        layer.propagate_ibp(&input)
    }

    #[ntest::timeout(10000)]
    #[test]
    fn qdq_relaxation_output_dtype_int8_uses_signed_range() {
        // output_dtype=INT8 with no y_zero_point: the quantized intermediate
        // lives in [-128, 127], so a negative activation is NOT saturated and
        // must propagate as a small perturbation around itself.
        let weights = scale_only_weights();
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(
            vec!["x", "scale"],
            vec![("output_dtype", AttributeValue::Int(3))],
            true,
        );
        let layer = ctx
            .convert_quantize_linear(&spec)
            .expect("int8 QDQ relaxation must convert");
        let output =
            propagate_point(&layer, -10.0).expect("-10 is inside the int8 non-saturating range");
        assert!(
            output.lower()[[0]] >= -11.0 && output.upper()[[0]] <= -9.0,
            "int8 QDQ output must stay near -10, got [{}, {}]",
            output.lower()[[0]],
            output.upper()[[0]]
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn qdq_relaxation_defaults_to_uint8_without_output_dtype_or_zero_point() {
        // Spec default when NEITHER output_dtype nor y_zero_point is supplied
        // is uint8: a negative activation saturates and must fail closed.
        let weights = scale_only_weights();
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(vec!["x", "scale"], vec![], true);
        let layer = ctx
            .convert_quantize_linear(&spec)
            .expect("default uint8 QDQ relaxation must convert");
        propagate_point(&layer, -10.0)
            .expect_err("-10 saturates the uint8 range and must be rejected");
        propagate_point(&layer, 10.0).expect("10 is inside the uint8 non-saturating range");
    }

    #[ntest::timeout(10000)]
    #[test]
    fn qdq_relaxation_unmodelled_output_dtype_rejected() {
        // FLOAT8E4M3FN (17) quantization also depends on the `saturate`
        // attribute; it must be rejected, never clamped to a guessed range.
        let weights = scale_only_weights();
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(
            vec!["x", "scale"],
            vec![("output_dtype", AttributeValue::Int(17))],
            true,
        );
        let err = ctx
            .convert_quantize_linear(&spec)
            .expect_err("unmodelled output_dtype must be rejected");
        assert!(
            matches!(err, NyError::UnsupportedOp(ref msg) if msg.contains("output_dtype")),
            "expected UnsupportedOp mentioning output_dtype, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn qdq_relaxation_output_dtype_zero_point_mismatch_rejected() {
        // output_dtype=UINT8 contradicting an int8 y_zero_point is malformed.
        let weights = int8_zero_point_weights();
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(
            vec!["x", "scale", "zp"],
            vec![("output_dtype", AttributeValue::Int(2))],
            true,
        );
        let err = ctx
            .convert_quantize_linear(&spec)
            .expect_err("output_dtype/zero_point range mismatch must be rejected");
        assert!(
            matches!(err, NyError::ModelLoad(ref msg) if msg.contains("output_dtype")),
            "expected ModelLoad mentioning output_dtype, got: {err:?}"
        );
    }

    #[ntest::timeout(10000)]
    #[test]
    fn qdq_relaxation_output_dtype_matching_zero_point_accepted() {
        let weights = int8_zero_point_weights();
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(
            vec!["x", "scale", "zp"],
            vec![("output_dtype", AttributeValue::Int(3))],
            true,
        );
        let layer = ctx
            .convert_quantize_linear(&spec)
            .expect("output_dtype agreeing with zero_point must convert");
        propagate_point(&layer, -10.0).expect("-10 is inside the int8 non-saturating range");
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
    fn quantize_linear_const_eval_unmodelled_output_dtype_not_evaluated() {
        let mut weights = scale_only_weights();
        weights.insert("x".to_string(), scalar_f32(1.0));
        let (shapes, constants) = (HashMap::new(), HashSet::new());
        let ctx = ConvertContext::new(&weights, &shapes, &constants);
        let spec = quantize_spec(
            vec!["x", "scale"],
            vec![("output_dtype", AttributeValue::Int(17))],
            false,
        );
        let evaluated = HashMap::new();
        assert!(
            ctx.evaluate_quantize_linear_constant(&spec, &evaluated)
                .is_none(),
            "unmodelled output_dtype must not be constant-evaluated"
        );
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
}
