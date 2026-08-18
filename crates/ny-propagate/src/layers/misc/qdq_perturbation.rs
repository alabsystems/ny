// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Disabled candidate relaxation for activation QuantizeLinear -> DequantizeLinear pairs.
//!
//! A half-scale exact-real envelope is not sufficient for ONNX's typed
//! binary32 divide, ties-to-even rounding, and binary32 reconstruction.  The
//! type remains in the internal layer inventory for compatibility, but its
//! only constructor fails closed until that full program has a certificate.

use ndarray::{Array1, Axis};
use ny_core::{NyError, Result};
use ny_tensor::{add_up_f32, sub_down_f32, BoundedTensor};
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

        Err(NyError::UnsupportedOp(format!(
            "QDQ perturbation for scale {scale}, zero_point {zero_point}, and range [{qmin}, {qmax}] is disabled: the full FLOAT32 division, rounding, reconstruction, and outward-bound envelope is not certified"
        )))
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
        // DIRECTED: `value - epsilon` under round-to-nearest can land ABOVE the
        // true difference, so the widened lower bound could sit inside the
        // interval it is meant to enlarge — the perturbation would be silently
        // smaller than requested. Exact whenever the shift is representable.
        BoundedTensor::new(
            input
                .lower()
                .mapv(|value| sub_down_f32(value, self.epsilon)),
            input.upper().mapv(|value| add_up_f32(value, self.epsilon)),
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

#[cfg(test)]
mod tests {
    use super::QdqPerturbationLayer;

    #[test]
    fn constructor_fails_closed_at_float32_rounding_counterexample() {
        let scale = 0.1_f32;
        let x = f32::from_bits(0x4154_0001);
        let quantized = (x / scale).round_ties_even();
        let dequantized = quantized * scale;
        let old_half_scale = f32::from_bits((scale * 0.5).to_bits() + 1);
        assert_eq!(quantized, 132.0);
        assert!(dequantized < x - old_half_scale);

        let error = QdqPerturbationLayer::new(scale, 0.0, 0.0, 255.0)
            .expect_err("uncertified FLOAT32 QDQ envelope must remain unavailable");
        assert!(error.to_string().contains("not certified"), "{error}");
    }
}
