// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, Axis, Ix1, Ix2, IxDyn};
use ny_propagate::network::broadcast_shapes;
use ny_propagate::Layer;
use std::collections::HashMap;
use tracing::debug;

use super::super::{ConvertContext, LayerSpec};
use super::lookup_constant_value;

pub(super) fn evaluate_linear_constant(layer: &Layer, input: ArrayD<f32>) -> Option<ArrayD<f32>> {
    let Layer::Linear(linear) = layer else {
        return None;
    };

    if input.ndim() == 1 {
        let input = input.into_dimensionality::<Ix1>().ok()?;
        let mut output = linear.weight.dot(&input);
        if let Some(bias) = &linear.bias {
            output += bias;
        }
        Some(output.into_dyn())
    } else {
        let shape = input.shape().to_vec();
        let in_features = *shape.last()?;
        if in_features != linear.in_features() {
            return None;
        }
        let batch_size = shape[..shape.len() - 1]
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))?;
        let input = input
            .into_shape_with_order((batch_size, in_features))
            .ok()?;
        let mut output = input.dot(&linear.weight.t());
        if let Some(bias) = &linear.bias {
            let bias = bias.broadcast((batch_size, linear.out_features()))?;
            output += &bias;
        }
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(linear.out_features());
        output.into_shape_with_order(IxDyn(&out_shape)).ok()
    }
}

pub(super) fn evaluate_instance_norm_constant(
    layer: &Layer,
    input: ArrayD<f32>,
) -> Option<ArrayD<f32>> {
    let Layer::InstanceNorm1d(instance_norm) = layer else {
        return None;
    };

    if input.ndim() == 3 && input.shape().first().copied() == Some(1) {
        let input = input
            .index_axis(Axis(0), 0)
            .to_owned()
            .into_dimensionality::<Ix2>()
            .ok()?;
        return instance_norm
            .eval_2d(&input)
            .ok()
            .map(|output| output.insert_axis(Axis(0)).into_dyn());
    }

    let input = input.into_dimensionality::<Ix2>().ok()?;
    instance_norm
        .eval_2d(&input)
        .ok()
        .map(|output| output.into_dyn())
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
        let a_is_constant_tensor = self.constant_tensors.contains(input_a);
        let b_is_constant_tensor = self.constant_tensors.contains(input_b);

        match (a_is_constant_tensor, b_is_constant_tensor, a_value, b_value) {
            (_, _, Some(a), Some(b)) => {
                debug!("Evaluating {} as Add with both constants", spec.name);
                Some(&a + &b)
            }
            (true, _, _, Some(weight)) => {
                let expanded = if weight.ndim() == 1 {
                    let h = weight.len();
                    weight.into_shape_with_order(IxDyn(&[1, h])).ok()
                } else {
                    Some(weight)
                };
                debug!(
                    "Add {} Case 1: A is constant tensor, returning B with shape {:?}",
                    spec.name,
                    expanded.as_ref().map(|tensor| tensor.shape().to_vec())
                );
                expanded
            }
            (_, true, Some(weight), _) => {
                let expanded = if weight.ndim() == 1 {
                    let h = weight.len();
                    weight.into_shape_with_order(IxDyn(&[1, h])).ok()
                } else {
                    Some(weight)
                };
                debug!(
                    "Add {} Case 2: B is constant tensor, returning A with shape {:?}",
                    spec.name,
                    expanded.as_ref().map(|tensor| tensor.shape().to_vec())
                );
                expanded
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
        if a.len() == 1 {
            let scalar = a.iter().next().copied().unwrap_or(1.0);
            Some(b.mapv(|value| value * scalar))
        } else if b.len() == 1 {
            let scalar = b.iter().next().copied().unwrap_or(1.0);
            Some(a.mapv(|value| value * scalar))
        } else {
            let output_shape = broadcast_shapes(a.shape(), b.shape())?;
            let a = a.broadcast(IxDyn(&output_shape))?;
            let b = b.broadcast(IxDyn(&output_shape))?;
            Some((&a * &b).into_owned())
        }
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
        let result = if lhs.len() == 1 {
            let scalar = lhs.iter().next().copied().unwrap_or(1.0);
            rhs.mapv(|value| scalar / value)
        } else if rhs.len() == 1 {
            let scalar = rhs.iter().next().copied().unwrap_or(1.0);
            lhs.mapv(|value| value / scalar)
        } else {
            let output_shape = broadcast_shapes(lhs.shape(), rhs.shape())?;
            let lhs = lhs.broadcast(IxDyn(&output_shape))?;
            let rhs = rhs.broadcast(IxDyn(&output_shape))?;
            (&lhs / &rhs).into_owned()
        };
        result
            .iter()
            .all(|value| value.is_finite())
            .then_some(result)
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
        Some(&a - &b)
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
        let result = base.mapv(|value| value.powf(exponent));
        result
            .iter()
            .all(|value| value.is_finite())
            .then_some(result)
    }
}
