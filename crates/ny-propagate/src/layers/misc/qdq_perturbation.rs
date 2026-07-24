// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound relaxation for ONNX activation QuantizeLinear -> DequantizeLinear pairs.

use ndarray::{Array1, Axis};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

#[derive(Debug, Clone)]
pub struct QdqPerturbationLayer {
    epsilon: f32,
    lower_saturation_edge: f32,
    upper_saturation_edge: f32,
}

impl QdqPerturbationLayer {
    pub fn new(scale: f32, zero_point: f32, qmin: f32, qmax: f32) -> Result<Self> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(NyError::InvalidSpec(format!(
                "QDQ perturbation requires positive finite scale, got {scale}"
            )));
        }
        if !(zero_point.is_finite() && qmin.is_finite() && qmax.is_finite() && qmin <= qmax) {
            return Err(NyError::InvalidSpec(format!(
                "QDQ perturbation has invalid quantization range [{qmin}, {qmax}] and zero_point {zero_point}"
            )));
        }

        let epsilon = next_up_f32(scale * 0.5);
        if !(epsilon.is_finite() && epsilon > 0.0) {
            return Err(NyError::InvalidSpec(format!(
                "QDQ perturbation produced invalid epsilon from scale {scale}"
            )));
        }

        Ok(Self {
            epsilon,
            lower_saturation_edge: scale * (qmin - zero_point - 0.5),
            upper_saturation_edge: scale * (qmax - zero_point + 0.5),
        })
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    fn saturation_inactive_for(&self, input: &BoundedTensor) -> bool {
        input
            .lower()
            .iter()
            .all(|value| value.is_finite() && *value >= self.lower_saturation_edge)
            && input
                .upper()
                .iter()
                .all(|value| value.is_finite() && *value <= self.upper_saturation_edge)
    }

    fn require_no_saturation(&self, input: &BoundedTensor, context: &str) -> Result<()> {
        if self.saturation_inactive_for(input) {
            Ok(())
        } else {
            Err(NyError::UnsupportedOp(format!(
                "QDQ perturbation {context} requires pre-activation bounds inside non-saturating range [{}, {}]",
                self.lower_saturation_edge, self.upper_saturation_edge
            )))
        }
    }

    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        self.require_no_saturation(pre_activation, "batched CROWN")?;

        let lower_adjust = bounds.lower_a().mapv(|value| value.abs() * self.epsilon);
        let upper_adjust = bounds.upper_a().mapv(|value| value.abs() * self.epsilon);
        let lower_delta = lower_adjust.sum_axis(Axis(lower_adjust.ndim() - 1));
        let upper_delta = upper_adjust.sum_axis(Axis(upper_adjust.ndim() - 1));

        BatchedLinearBounds::new(
            bounds.lower_a().clone(),
            bounds.lower_b() - &lower_delta,
            bounds.upper_a().clone(),
            bounds.upper_b() + &upper_delta,
            bounds.input_shape().to_vec(),
            bounds.output_shape().to_vec(),
        )
    }
}

impl BoundPropagation for QdqPerturbationLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.require_no_saturation(input, "IBP")?;
        BoundedTensor::new(
            input.lower().mapv(|value| value - self.epsilon),
            input.upper().mapv(|value| value + self.epsilon),
        )
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        let lower_delta: Array1<f32> = bounds.lower_a().map_axis(Axis(1), |row| {
            row.iter().map(|value| value.abs() * self.epsilon).sum()
        });
        let upper_delta: Array1<f32> = bounds.upper_a().map_axis(Axis(1), |row| {
            row.iter().map(|value| value.abs() * self.epsilon).sum()
        });

        Ok(Cow::Owned(LinearBounds::new(
            bounds.lower_a().clone(),
            bounds.lower_b() - &lower_delta,
            bounds.upper_a().clone(),
            bounds.upper_b() + &upper_delta,
        )?))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        self.require_no_saturation(pre_activation, "CROWN")?;
        self.propagate_linear(bounds).map(Cow::into_owned)
    }
}

fn next_up_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == -0.0 {
        return f32::MIN_POSITIVE.min(f32::from_bits(1));
    }
    let bits = value.to_bits();
    if value >= 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::QdqPerturbationLayer;
    use crate::layers::common::BoundPropagation;
    use crate::LinearBounds;
    use ndarray::{arr1, arr2};
    use ny_tensor::BoundedTensor;

    #[test]
    fn ibp_widens_by_outward_half_scale_inside_non_saturating_range() {
        let layer = QdqPerturbationLayer::new(0.25, 10.0, 0.0, 255.0).unwrap();
        let input = BoundedTensor::new(arr1(&[-1.0, 2.0]).into_dyn(), arr1(&[0.5, 3.0]).into_dyn())
            .unwrap();

        let output = layer.propagate_ibp(&input).unwrap();

        assert!(output.lower()[[0]] <= -1.125);
        assert!(output.upper()[[1]] >= 3.125);
    }

    #[test]
    fn crown_bias_widening_uses_absolute_coefficients() {
        let layer = QdqPerturbationLayer::new(0.5, 0.0, -128.0, 127.0).unwrap();
        let bounds = LinearBounds::new(
            arr2(&[[2.0, -3.0]]),
            arr1(&[1.0]),
            arr2(&[[-4.0, 5.0]]),
            arr1(&[2.0]),
        )
        .unwrap();
        let input =
            BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn())
                .unwrap();

        let output = layer.propagate_linear_with_bounds(&bounds, &input).unwrap();

        assert!(output.lower_b()[0] <= 1.0 - 1.25);
        assert!(output.upper_b()[0] >= 2.0 + 2.25);
    }

    #[test]
    fn crown_rejects_when_saturation_may_be_active() {
        let layer = QdqPerturbationLayer::new(1.0, 0.0, 0.0, 255.0).unwrap();
        let bounds = LinearBounds::identity(1);
        let input =
            BoundedTensor::new(arr1(&[-10.0]).into_dyn(), arr1(&[-9.0]).into_dyn()).unwrap();

        assert!(layer.propagate_linear_with_bounds(&bounds, &input).is_err());
    }
}
