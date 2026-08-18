// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_propagate::network::broadcast_shapes;
use ny_propagate::Layer;
use std::collections::HashMap;
use tracing::debug;

use super::super::{ConvertContext, LayerSpec};
use super::lookup_constant_value;

fn exact_f64_sum(lhs: f32, rhs: f32) -> Option<f32> {
    if !lhs.is_finite() || !rhs.is_finite() {
        return None;
    }
    let a = lhs as f64;
    let b = rhs as f64;
    // Error-free TwoSum: s + error is the exact real a+b.  Admission requires
    // the error to vanish and the exact f64 sum to be representable as f32.
    let s = a + b;
    let b_virtual = s - a;
    let error = (a - (s - b_virtual)) + (b - b_virtual);
    let rounded = lhs + rhs;
    (error == 0.0 && rounded.is_finite() && rounded as f64 == s).then_some(rounded)
}

fn exact_f64_product(lhs: f32, rhs: f32) -> Option<f32> {
    if !lhs.is_finite() || !rhs.is_finite() {
        return None;
    }
    // A product of two binary32 significands has at most 48 bits, so binary64
    // represents it exactly throughout binary32's finite exponent range.
    let exact = (lhs as f64) * (rhs as f64);
    let rounded = lhs * rhs;
    (rounded.is_finite() && rounded as f64 == exact).then_some(rounded)
}

fn exact_f64_quotient(lhs: f32, rhs: f32) -> Option<f32> {
    if !lhs.is_finite() || !rhs.is_finite() || rhs == 0.0 {
        return None;
    }
    let rounded = lhs / rhs;
    if !rounded.is_finite() {
        return None;
    }
    // q is the exact quotient iff q*rhs == lhs. The binary32 product is exact
    // in binary64, so this is a proof rather than a tolerance check.
    ((rounded as f64) * (rhs as f64) == lhs as f64).then_some(rounded)
}

fn exact_broadcast_binary(
    lhs: &ArrayD<f32>,
    rhs: &ArrayD<f32>,
    operation: fn(f32, f32) -> Option<f32>,
) -> Option<ArrayD<f32>> {
    let shape = broadcast_shapes(lhs.shape(), rhs.shape())?;
    let lhs = lhs.broadcast(IxDyn(&shape))?;
    let rhs = rhs.broadcast(IxDyn(&shape))?;
    let values = lhs
        .iter()
        .zip(rhs.iter())
        .map(|(&lhs, &rhs)| operation(lhs, rhs))
        .collect::<Option<Vec<_>>>()?;
    ArrayD::from_shape_vec(IxDyn(&shape), values).ok()
}

/// Evaluate a frozen affine layer only when every multiply and accumulation is
/// exactly representable as binary32.  This is deliberately a certificate,
/// not an ordinary f32 GEMM: publishing a rounded dot product as a point
/// constant would change the exact-real network consumed by the verifier.
pub(super) fn evaluate_linear_constant_exact(
    layer: &Layer,
    input: ArrayD<f32>,
) -> Option<ArrayD<f32>> {
    let Layer::Linear(linear) = layer else {
        return None;
    };
    let input_shape = input.shape();
    let in_features = *input_shape.last()?;
    if in_features != linear.in_features() {
        return None;
    }
    let row_count = input_shape[..input_shape.len() - 1]
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))?;
    let input_values = input.as_standard_layout();
    let input_values = input_values.as_slice()?;
    let mut output = Vec::with_capacity(row_count.checked_mul(linear.out_features())?);

    for row in 0..row_count {
        let row_start = row.checked_mul(in_features)?;
        for output_feature in 0..linear.out_features() {
            let mut sum = 0.0_f32;
            for input_feature in 0..in_features {
                let product = exact_f64_product(
                    input_values[row_start + input_feature],
                    linear.weight()[[output_feature, input_feature]],
                )?;
                sum = exact_f64_sum(sum, product)?;
            }
            if let Some(bias) = linear.bias() {
                sum = exact_f64_sum(sum, bias[output_feature])?;
            }
            output.push(sum);
        }
    }

    let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
    output_shape.push(linear.out_features());
    ArrayD::from_shape_vec(IxDyn(&output_shape), output).ok()
}

/// Evaluate a frozen convolution only when every scalar multiply and every
/// accumulation is exactly representable as binary32.  Ordinary convolution
/// kernels round after each operation; their point output is therefore not a
/// certificate for the exact-real network represented by an ONNX model.
pub(super) fn evaluate_convolution_constant_exact(
    layer: &Layer,
    input: ArrayD<f32>,
) -> Option<ArrayD<f32>> {
    match layer {
        Layer::Conv1d(conv) => evaluate_conv1d_constant_exact(conv, &input),
        Layer::Conv2d(conv) => evaluate_conv2d_constant_exact(conv, &input),
        _ => None,
    }
}

fn exact_convolution_sum(
    products: impl Iterator<Item = Option<f32>>,
    bias: Option<f32>,
) -> Option<f32> {
    let mut sum = 0.0_f32;
    for product in products {
        sum = exact_f64_sum(sum, product?)?;
    }
    match bias {
        Some(bias) => exact_f64_sum(sum, bias),
        None => Some(sum),
    }
}

fn evaluate_conv1d_constant_exact(
    conv: &ny_propagate::layers::Conv1dLayer,
    input: &ArrayD<f32>,
) -> Option<ArrayD<f32>> {
    let (batch, in_channels, input_len, batched) = match input.shape() {
        [in_channels, input_len] => (1, *in_channels, *input_len, false),
        [batch, in_channels, input_len] => (*batch, *in_channels, *input_len, true),
        _ => return None,
    };
    let kernel_shape = conv.kernel.shape();
    let [out_channels, in_channels_per_group, kernel_len] = kernel_shape else {
        return None;
    };
    if conv.groups == 0
        || in_channels_per_group.checked_mul(conv.groups)? != in_channels
        || !out_channels.is_multiple_of(conv.groups)
    {
        return None;
    }
    let output_len = conv.output_length(input_len).ok()?;
    let out_channels_per_group = out_channels / conv.groups;
    let mut output = Vec::with_capacity(batch.checked_mul(*out_channels)?.checked_mul(output_len)?);

    for batch_index in 0..batch {
        for output_channel in 0..*out_channels {
            let group = output_channel / out_channels_per_group;
            let input_channel_start = group.checked_mul(*in_channels_per_group)?;
            for output_index in 0..output_len {
                let products = (0..*in_channels_per_group).flat_map(|local_input_channel| {
                    (0..*kernel_len).map(move |kernel_index| {
                        let padded_index = output_index
                            .checked_mul(conv.stride)?
                            .checked_add(kernel_index.checked_mul(conv.dilation)?)?;
                        if padded_index < conv.padding {
                            return Some(0.0);
                        }
                        let input_index = padded_index - conv.padding;
                        if input_index >= input_len {
                            return Some(0.0);
                        }
                        let input_channel = input_channel_start + local_input_channel;
                        let input_value = if batched {
                            *input.get(IxDyn(&[batch_index, input_channel, input_index]))?
                        } else {
                            *input.get(IxDyn(&[input_channel, input_index]))?
                        };
                        let kernel_value = *conv.kernel.get(IxDyn(&[
                            output_channel,
                            local_input_channel,
                            kernel_index,
                        ]))?;
                        exact_f64_product(input_value, kernel_value)
                    })
                });
                output.push(exact_convolution_sum(
                    products,
                    conv.bias.as_ref().map(|bias| bias[output_channel]),
                )?);
            }
        }
    }

    let output_shape = if batched {
        vec![batch, *out_channels, output_len]
    } else {
        vec![*out_channels, output_len]
    };
    ArrayD::from_shape_vec(IxDyn(&output_shape), output).ok()
}

fn evaluate_conv2d_constant_exact(
    conv: &ny_propagate::layers::Conv2dLayer,
    input: &ArrayD<f32>,
) -> Option<ArrayD<f32>> {
    let (batch, in_channels, input_h, input_w, batched) = match input.shape() {
        [in_channels, input_h, input_w] => (1, *in_channels, *input_h, *input_w, false),
        [batch, in_channels, input_h, input_w] => (*batch, *in_channels, *input_h, *input_w, true),
        _ => return None,
    };
    let kernel_shape = conv.kernel.shape();
    let [out_channels, in_channels_per_group, kernel_h, kernel_w] = kernel_shape else {
        return None;
    };
    if conv.groups == 0
        || in_channels_per_group.checked_mul(conv.groups)? != in_channels
        || !out_channels.is_multiple_of(conv.groups)
    {
        return None;
    }
    let (output_h, output_w) = conv.output_size(input_h, input_w).ok()?;
    let out_channels_per_group = out_channels / conv.groups;
    let output_elements = batch
        .checked_mul(*out_channels)?
        .checked_mul(output_h)?
        .checked_mul(output_w)?;
    let mut output = Vec::with_capacity(output_elements);

    for batch_index in 0..batch {
        for output_channel in 0..*out_channels {
            let group = output_channel / out_channels_per_group;
            let input_channel_start = group.checked_mul(*in_channels_per_group)?;
            for output_row in 0..output_h {
                for output_col in 0..output_w {
                    let products = (0..*in_channels_per_group).flat_map(|local_input_channel| {
                        (0..*kernel_h).flat_map(move |kernel_row| {
                            (0..*kernel_w).map(move |kernel_col| {
                                let padded_row = output_row
                                    .checked_mul(conv.stride.0)?
                                    .checked_add(kernel_row.checked_mul(conv.dilation.0)?)?;
                                let padded_col = output_col
                                    .checked_mul(conv.stride.1)?
                                    .checked_add(kernel_col.checked_mul(conv.dilation.1)?)?;
                                if padded_row < conv.padding.0 || padded_col < conv.padding.1 {
                                    return Some(0.0);
                                }
                                let input_row = padded_row - conv.padding.0;
                                let input_col = padded_col - conv.padding.1;
                                if input_row >= input_h || input_col >= input_w {
                                    return Some(0.0);
                                }
                                let input_channel = input_channel_start + local_input_channel;
                                let input_value = if batched {
                                    *input.get(IxDyn(&[
                                        batch_index,
                                        input_channel,
                                        input_row,
                                        input_col,
                                    ]))?
                                } else {
                                    *input.get(IxDyn(&[input_channel, input_row, input_col]))?
                                };
                                let kernel_value = *conv.kernel.get(IxDyn(&[
                                    output_channel,
                                    local_input_channel,
                                    kernel_row,
                                    kernel_col,
                                ]))?;
                                exact_f64_product(input_value, kernel_value)
                            })
                        })
                    });
                    output.push(exact_convolution_sum(
                        products,
                        conv.bias.as_ref().map(|bias| bias[output_channel]),
                    )?);
                }
            }
        }
    }

    let output_shape = if batched {
        vec![batch, *out_channels, output_h, output_w]
    } else {
        vec![*out_channels, output_h, output_w]
    };
    ArrayD::from_shape_vec(IxDyn(&output_shape), output).ok()
}

impl ConvertContext<'_> {
    pub(super) fn evaluate_add_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.len() < 2 {
            return None;
        }
        let input_a = &spec.inputs[0];
        let input_b = &spec.inputs[1];
        let a_value = lookup_constant_value(self.weights, evaluated_constants, input_a);
        let b_value = lookup_constant_value(self.weights, evaluated_constants, input_b);
        match (a_value, b_value) {
            (Some(a), Some(b)) => {
                debug!("Evaluating {} as Add with both constants", spec.name);
                exact_broadcast_binary(&a, &b, exact_f64_sum)
            }
            _ => None,
        }
    }

    pub(super) fn evaluate_mul_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.len() < 2 {
            return None;
        }
        let a = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])?;
        let b = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[1])?;
        debug!("Evaluating {} as Mul with both constants", spec.name);
        exact_broadcast_binary(&a, &b, exact_f64_product)
    }

    pub(super) fn evaluate_div_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.len() < 2 {
            return None;
        }
        let lhs = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])?;
        let rhs = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[1])?;
        exact_broadcast_binary(&lhs, &rhs, exact_f64_quotient)
    }

    pub(super) fn evaluate_sub_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.len() < 2 {
            return None;
        }
        let a = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])?;
        let b = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[1])?;
        debug!("Evaluating {} as Sub with both constants", spec.name);
        exact_broadcast_binary(&a, &b, |lhs, rhs| exact_f64_sum(lhs, -rhs))
    }

    pub(super) fn evaluate_pow_constant(
        &self,
        spec: &LayerSpec,
        evaluated_constants: &HashMap<String, ArrayD<f32>>,
    ) -> Option<ArrayD<f32>> {
        if spec.inputs.len() < 2 {
            return None;
        }
        let base = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[0])?;
        let exponent = lookup_constant_value(self.weights, evaluated_constants, &spec.inputs[1])?;
        let exponent = if exponent.len() == 1 {
            exponent.iter().next().copied().unwrap_or(1.0)
        } else {
            let first = exponent.iter().next().copied().unwrap_or(1.0);
            exponent
                .iter()
                .all(|&value| value == first)
                .then_some(first)?
        };
        if exponent == 1.0 {
            return base.iter().all(|value| value.is_finite()).then_some(base);
        }
        if exponent == 0.0 && base.iter().all(|value| value.is_finite()) {
            return Some(ArrayD::from_elem(base.raw_dim(), 1.0));
        }
        None
    }
}
