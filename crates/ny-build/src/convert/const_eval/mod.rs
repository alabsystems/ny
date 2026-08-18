// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

mod arithmetic;
mod shape;

#[cfg(test)]
mod tests;

use ndarray::{ArrayD, IxDyn};
use ny_core::LayerType;
use ny_propagate::Layer;
use std::collections::HashMap;
use tracing::debug;

use self::arithmetic::{evaluate_convolution_constant_exact, evaluate_linear_constant_exact};
use super::{ConvertContext, LayerSpec};

fn propagate_constant_through_layer(
    layer: &Layer,
    input: ArrayD<f32>,
    _layer_name: &str,
) -> Option<ArrayD<f32>> {
    // The layer-level policy runs point IBP and admits only certified exact
    // copy/select/shape operations.  A singleton produced by ordinary f32
    // arithmetic is not enough to materialize an exact-real constant.
    layer.propagate_concrete(input).ok()
}

fn lookup_constant_value(
    weights: &crate::WeightStore,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
    name: &str,
) -> Option<ArrayD<f32>> {
    weights
        .get(name)
        .cloned()
        .or_else(|| evaluated_constants.get(name).cloned())
}

fn lookup_integral_constant_values(
    weights: &crate::WeightStore,
    evaluated_constants: &HashMap<String, ArrayD<f32>>,
    spec: &LayerSpec,
    name: &str,
    field: &str,
    allow_positive_infinity: bool,
) -> Option<(Vec<i64>, Vec<usize>)> {
    if let Some(integers) = weights.get_integers(name) {
        if weights
            .get(name)
            .is_some_and(|floats| floats.shape() != integers.shape())
        {
            debug!(
                "{} {} exact integer {} tensor '{}' has a mismatched float-view shape",
                spec.layer_type, spec.name, field, name
            );
            return None;
        }
        return Some((
            integers.iter().copied().collect(),
            integers.shape().to_vec(),
        ));
    }
    let values = lookup_constant_value(weights, evaluated_constants, name)?;
    let parsed = values
        .iter()
        .copied()
        .map(|value| parse_integral_constant_value(spec, field, value, allow_positive_infinity))
        .collect::<Option<Vec<_>>>()?;
    Some((parsed, values.shape().to_vec()))
}

fn resolve_constant_axis(spec: &LayerSpec, op_name: &str, ndim: usize, axis: i64) -> Option<usize> {
    let resolved = if axis < 0 { ndim as i64 + axis } else { axis };
    if resolved < 0 || resolved as usize >= ndim {
        debug!(
            "{} {} axis {} is out of bounds for rank {}",
            op_name, spec.name, axis, ndim
        );
        return None;
    }
    Some(resolved as usize)
}

fn adjust_constant_slice_axis(spec: &LayerSpec, axis: i64) -> Option<i64> {
    if axis == 0 {
        debug!(
            "Slice {} constant evaluation rejects ONNX axis=0 in unbatched mode",
            spec.name
        );
        return None;
    }
    Some(if axis > 0 { axis - 1 } else { axis })
}

fn parse_integral_constant_value(
    spec: &LayerSpec,
    field: &str,
    value: f32,
    allow_positive_infinity: bool,
) -> Option<i64> {
    if value.is_nan() {
        debug!("{} {} has NaN {}", spec.layer_type, spec.name, field);
        return None;
    }
    if value.is_infinite() {
        if allow_positive_infinity && value.is_sign_positive() {
            return Some(i64::MAX);
        }
        debug!(
            "{} {} has unsupported infinite {}={}",
            spec.layer_type, spec.name, field, value
        );
        return None;
    }
    let integral = value.trunc();
    if integral != value {
        debug!(
            "{} {} has non-integral {}={}",
            spec.layer_type, spec.name, field, value
        );
        return None;
    }
    if integral < i64::MIN as f32 || integral >= i64::MAX as f32 {
        debug!(
            "{} {} has out-of-range {}={}",
            spec.layer_type, spec.name, field, value
        );
        return None;
    }
    Some(integral as i64)
}

fn normalize_slice_bound(bound: i64, axis_len: i64) -> i64 {
    let resolved = if bound < 0 { axis_len + bound } else { bound };
    resolved.clamp(0, axis_len)
}

impl ConvertContext<'_> {
    pub fn evaluate_constant_layer(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        match spec.layer_type {
            LayerType::Add => self.evaluate_add_constant(spec, evaluated_constants),
            LayerType::Mul => self.evaluate_mul_constant(spec, evaluated_constants),
            LayerType::Div => self.evaluate_div_constant(spec, evaluated_constants),
            LayerType::Sub => self.evaluate_sub_constant(spec, evaluated_constants),
            LayerType::Pow => self.evaluate_pow_constant(spec, evaluated_constants),
            LayerType::DequantizeLinear => {
                self.evaluate_dequantize_linear_constant(spec, evaluated_constants)
            }
            LayerType::QuantizeLinear => {
                self.evaluate_quantize_linear_constant(spec, evaluated_constants)
            }
            LayerType::Concat => self.evaluate_concat_constant(spec, evaluated_constants),
            LayerType::Linear => {
                let input =
                    lookup_constant_value(self.weights, evaluated_constants, spec.inputs.first()?)?;
                let layer = self.convert_layer(spec).ok()?;
                evaluate_linear_constant_exact(&layer, input)
            }
            LayerType::Conv1d | LayerType::Conv2d => {
                let input =
                    lookup_constant_value(self.weights, evaluated_constants, spec.inputs.first()?)?;
                let layer = self.convert_layer(spec).ok()?;
                evaluate_convolution_constant_exact(&layer, input)
            }
            // InstanceNorm contains non-trivial floating-point arithmetic. A
            // point IBP image may still collapse around a rounded result, so
            // it is not an exact-real materialization certificate.
            LayerType::InstanceNorm => None,
            LayerType::ConvTranspose1d | LayerType::ConvTranspose2d => {
                // Skip constant evaluation for transposed convolutions.
                // conv1d_transpose_forward uses a scalar loop that is
                // O(in_c * in_len * out_c * kernel_size). In HiFi-GAN
                // upsampler chains (e.g., Kokoro vocoder), the temporal
                // dimension grows multiplicatively through 3 stages
                // (har_t=61 -> 610 -> 3660 -> 18300 at strides 10, 6, 5),
                // making const-fold take >180s in debug mode. Unevaluated
                // layers stay in the graph and propagate correctly at
                // IBP/CROWN runtime.
                debug!(
                    "Skipping constant fold for ConvTranspose {} (scalar loop too expensive for const-fold)",
                    spec.name
                );
                None
            }
            LayerType::Reshape => {
                let input =
                    lookup_constant_value(self.weights, evaluated_constants, spec.inputs.first()?)?;
                let layer = self.convert_layer(spec).ok()?;
                let Layer::Reshape(reshape) = layer else {
                    return None;
                };
                let output_shape = reshape.compute_output_shape(input.shape()).ok()?;
                input.into_shape_with_order(IxDyn(&output_shape)).ok()
            }
            LayerType::ReduceMax | LayerType::ReduceMin => {
                let input =
                    lookup_constant_value(self.weights, evaluated_constants, spec.inputs.first()?)?;
                let layer = self.convert_layer(spec).ok()?;
                propagate_constant_through_layer(&layer, input, &spec.name)
            }
            // Sum/mean need an exact accumulation certificate (or an interval
            // frozen-constant carrier); generic point propagation is not one.
            LayerType::ReduceSum | LayerType::ReduceMean => None,
            LayerType::Slice => self.evaluate_slice_constant(spec, evaluated_constants),
            LayerType::Squeeze => self.evaluate_squeeze_constant(spec, evaluated_constants),
            LayerType::Unsqueeze => self.evaluate_unsqueeze_constant(spec, evaluated_constants),
            LayerType::Transpose => self.evaluate_transpose_constant(spec, evaluated_constants),
            LayerType::Gather => self.evaluate_gather_constant(spec, evaluated_constants),
            LayerType::Shape => self.evaluate_shape_constant(spec, evaluated_constants),
            LayerType::Cast => self.evaluate_cast_constant(spec, evaluated_constants),
            _ => self.evaluate_fallback_constant(spec, evaluated_constants),
        }
    }
}
