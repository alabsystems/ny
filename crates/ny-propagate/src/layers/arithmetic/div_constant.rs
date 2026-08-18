// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Divide by constant layer: y = x / c (element-wise).

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{div_down_f32, div_up_f32, BoundedTensor};
use std::borrow::Cow;

use super::common::DIV_CONSTANT_EPS;
use super::mul_constant::MulConstantLayer;
use super::validate::validate_finite_array;
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

/// Divide by constant layer: y = x / c (element-wise).
///
/// Used in LayerNorm for division by standard deviation.
#[derive(Debug, Clone)]
pub struct DivConstantLayer {
    /// The constant tensor divisor.
    pub(crate) constant: ArrayD<f32>,
    /// Original input shape from conversion, used for broadcast-aware CROWN backward.
    input_shape: Option<Vec<usize>>,
}

impl DivConstantLayer {
    fn validate_divisor(constant: &ArrayD<f32>) -> Result<()> {
        validate_finite_array(constant, "DivConstantLayer", "constant")?;
        if let Some((index, value)) = constant
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| value.abs() < DIV_CONSTANT_EPS)
        {
            return Err(NyError::InvalidSpec(format!(
                "DivConstantLayer constant contains near-zero divisor at flat index {index}: {value}"
            )));
        }
        Ok(())
    }

    /// Validate and create a new divide by constant layer.
    pub fn try_new(constant: ArrayD<f32>) -> Result<Self> {
        Self::validate_divisor(&constant)?;
        Ok(Self {
            constant,
            input_shape: None,
        })
    }

    /// Create a new divide by constant layer.
    pub fn new(constant: ArrayD<f32>) -> Self {
        Self::try_new(constant)
            .expect("invariant: DivConstantLayer::new requires finite non-zero constant")
    }

    /// Validate and create a divide-by-constant layer with the original input shape.
    pub fn try_with_input_shape(constant: ArrayD<f32>, input_shape: Vec<usize>) -> Result<Self> {
        let mut layer = Self::try_new(constant)?;
        layer.input_shape = Some(input_shape);
        Ok(layer)
    }

    /// Create a divide-by-constant layer with the original input shape.
    pub fn with_input_shape(constant: ArrayD<f32>, input_shape: Vec<usize>) -> Self {
        Self::try_with_input_shape(constant, input_shape).expect(
            "invariant: DivConstantLayer::with_input_shape requires finite non-zero constant",
        )
    }

    /// Validate and create a scalar divisor layer.
    pub fn try_scalar(value: f32) -> Result<Self> {
        Self::try_new(ArrayD::from_elem(IxDyn(&[]), value))
    }

    /// Create a scalar divisor layer.
    pub fn scalar(value: f32) -> Self {
        Self::try_scalar(value)
            .expect("invariant: DivConstantLayer::scalar requires finite non-zero divisor")
    }

    /// Return the divisor tensor.
    pub fn constant(&self) -> &ArrayD<f32> {
        &self.constant
    }

    /// Return the original input shape if conversion recorded it.
    pub fn input_shape(&self) -> Option<&[usize]> {
        self.input_shape.as_deref()
    }

    fn inverse_constant(&self) -> Result<ArrayD<f32>> {
        if self.constant.iter().any(|v| v.abs() < DIV_CONSTANT_EPS) {
            return Err(NyError::NumericalInstability(
                "Division by near-zero constant".to_string(),
            ));
        }
        // Compute 1/c in f64 to reduce rounding error in the CROWN coefficient
        // path. The remaining ULP from f64->f32 cast is covered by
        // concretize_sound()'s directed rounding at concretization time.
        // Part of #1483.
        Ok(self.constant.mapv(|v| (1.0_f64 / v as f64) as f32))
    }
}

impl BoundPropagation for DivConstantLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // For y = x / c: division by c is equivalent to multiplication by 1/c
        // Bounds depend on sign of c:
        // If c > 0: y ∈ [l/c, u/c]
        // If c < 0: y ∈ [u/c, l/c]
        //
        // ONNX Div broadcasts both inputs to a common shape. When the constant
        // has higher rank than the input, we broadcast both to the output shape.

        let input_shape = input.shape();
        let const_shape = self.constant.shape();

        // Compute ONNX bidirectional broadcast output shape.
        let output_shape =
            crate::shape::broadcast_shapes(input_shape, const_shape).ok_or_else(|| {
                NyError::ShapeMismatch {
                    expected: input_shape.to_vec(),
                    got: const_shape.to_vec(),
                }
            })?;

        // Broadcast constant to output shape.
        let c = self
            .constant
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: const_shape.to_vec(),
            })?;

        // Broadcast input bounds to output shape.
        let lower_in = input
            .lower()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: input_shape.to_vec(),
            })?;
        let upper_in = input
            .upper()
            .broadcast(IxDyn(&output_shape))
            .ok_or_else(|| NyError::ShapeMismatch {
                expected: output_shape.clone(),
                got: input_shape.to_vec(),
            })?;

        // Compute bounds element-wise, handling sign
        let mut out_lower = ArrayD::zeros(IxDyn(&output_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(&output_shape));

        for (idx, &c_val) in c.indexed_iter() {
            let l = lower_in[idx.clone()];
            let u = upper_in[idx.clone()];

            // Avoid division by zero - if divisor is near zero, bounds explode
            if c_val.abs() < DIV_CONSTANT_EPS {
                return Err(NyError::NumericalInstability(
                    "Division by near-zero constant".to_string(),
                ));
            }

            // Directed division (#1483). `div_down_f32`/`div_up_f32` certify
            // the f64 quotient with its own exact fma remainder and then narrow
            // in the correct direction, so they step only when the division
            // actually rounded — unlike the previous
            // `next_down_f32((l64 / c64) as f32)`, which gave away a full ULP
            // even for exact quotients and returns an infinity as a finite
            // bound on overflow.
            if c_val > 0.0 {
                out_lower[idx.clone()] = div_down_f32(l, c_val);
                out_upper[idx] = div_up_f32(u, c_val);
            } else {
                out_lower[idx.clone()] = div_down_f32(u, c_val);
                out_upper[idx] = div_up_f32(l, c_val);
            }
        }

        // OpaqueSkip taint (#opaque-skip-six-sites): an upstream OpaqueSkip
        // legitimately emits ±Inf endpoints. The divisor is validated finite
        // with |c| >= DIV_CONSTANT_EPS both at construction and per element
        // above, so `±inf / c` passes through `div_down_f32`/`div_up_f32` as
        // clean ±Inf and the NaN-producing quotient patterns (inf/inf, 0/0)
        // are unreachable. A NaN here therefore implies a NaN INPUT — a real
        // bug — which `new_allow_infinite` still rejects as a hard error.
        BoundedTensor::new_allow_infinite(out_lower, out_upper)
    }

    #[inline]
    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // For y = x / c, this is the same as y = x * (1/c)
        self.as_mul_layer()?.propagate_linear(bounds)
    }

    /// CROWN backward with the actual input bounds available.
    ///
    /// Delegates to the equivalent MulConstant (y = x * (1/c)), which can
    /// recover a per-channel broadcast layout from the pre-activation shape
    /// when conversion could not record `input_shape` (see
    /// `MulConstantLayer::scale_for_linear_bounds_with_runtime_shape`).
    fn propagate_crown_backward(
        &self,
        bounds: &LinearBounds,
        pre_activation: Option<&BoundedTensor>,
    ) -> Result<LinearBounds> {
        self.as_mul_layer()?
            .propagate_crown_backward(bounds, pre_activation)
    }
}

impl DivConstantLayer {
    /// The equivalent multiply-by-1/c layer used for CROWN backward.
    fn as_mul_layer(&self) -> Result<MulConstantLayer> {
        let inv = self.inverse_constant()?;
        Ok(match self.input_shape() {
            Some(input_shape) => MulConstantLayer::with_input_shape(inv, input_shape.to_vec()),
            None => MulConstantLayer::new(inv),
        })
    }
}

impl DivConstantLayer {
    /// Batched CROWN backward propagation through DivConstant.
    ///
    /// For y = x / c, this is equivalent to y = x * (1/c), so we delegate to MulConstant.
    #[inline]
    pub fn propagate_linear_batched(
        &self,
        bounds: &BatchedLinearBounds,
    ) -> Result<BatchedLinearBounds> {
        // Division by c is multiplication by 1/c
        self.as_mul_layer()?.propagate_linear_batched(bounds)
    }
}

/// OpaqueSkip taint probes (#opaque-skip-six-sites): the IBP output
/// constructor must let the legitimate ±Inf an upstream OpaqueSkip emits flow
/// through as widened bounds, while NaN inputs remain a hard error.
#[cfg(test)]
mod opaque_skip_taint_tests {
    use super::*;

    fn opaque_input() -> BoundedTensor {
        BoundedTensor::new_allow_infinite(
            ArrayD::from_elem(IxDyn(&[2]), f32::NEG_INFINITY),
            ArrayD::from_elem(IxDyn(&[2]), f32::INFINITY),
        )
        .unwrap()
    }

    /// [-inf, +inf] / c must propagate as [-inf, +inf] for both divisor signs,
    /// not abort with NumericalInstability. The divisor is finite and bounded
    /// away from zero, so inf/inf and 0/0 NaN patterns are unreachable.
    #[test]
    fn test_ibp_opaque_skip_inf_input_flows() {
        for divisor in [2.0_f32, -2.0] {
            let layer = DivConstantLayer::scalar(divisor);
            let out = layer
                .propagate_ibp(&opaque_input())
                .expect("±inf input must propagate through x / c");
            assert_eq!(out.lower()[[0]], f32::NEG_INFINITY, "divisor {divisor}");
            assert_eq!(out.upper()[[1]], f32::INFINITY, "divisor {divisor}");
        }
    }

    /// NaN input (a real bug, not OpaqueSkip taint) must still hard-error:
    /// `new_allow_infinite` rejects NaN.
    #[test]
    fn test_ibp_nan_input_still_errors() {
        let layer = DivConstantLayer::scalar(2.0);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_elem(IxDyn(&[1]), f32::NAN),
            ArrayD::from_elem(IxDyn(&[1]), 1.0_f32),
        )
        .unwrap();
        assert!(
            layer.propagate_ibp(&input).is_err(),
            "NaN input must remain a hard error"
        );
    }
}
