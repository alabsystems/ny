// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP and `BoundPropagation` trait wiring for BatchNorm.

use ndarray::{ArrayD, IxDyn};
use ny_core::{
    dd::{next_down_f64, next_up_f64, two_sum},
    f32_to_f64_exact, f64_to_f32_down, f64_to_f32_up, require_f64_interval_proof_environment,
    NyError, Result,
};
use ny_tensor::{BoundedTensor, RepairStrategy};
use std::borrow::Cow;

use super::math::{detect_input_layout, nonnegative_add_up, nonnegative_mul_up};
use super::types::BatchNormLayer;
use crate::bounds::safe_mul_for_bounds_f64;
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;

/// Add two binary64 values toward negative infinity.
///
/// Every finite BatchNorm product below is exact in binary64 because both
/// operands are binary32 (at most 48 significant product bits). `two_sum`
/// therefore needs to direct only the subsequent addition/subtraction.
#[inline]
fn add_down_f64(left: f64, right: f64) -> f64 {
    if !left.is_finite() || !right.is_finite() {
        let sum = left + right;
        return if sum.is_nan() { f64::NEG_INFINITY } else { sum };
    }
    let (sum, residual) = two_sum(left, right);
    if residual < 0.0 {
        next_down_f64(sum)
    } else {
        sum
    }
}

/// Upper-bound mirror of [`add_down_f64`].
#[inline]
fn add_up_f64(left: f64, right: f64) -> f64 {
    if !left.is_finite() || !right.is_finite() {
        let sum = left + right;
        return if sum.is_nan() { f64::INFINITY } else { sum };
    }
    let (sum, residual) = two_sum(left, right);
    if residual > 0.0 {
        next_up_f64(sum)
    } else {
        sum
    }
}

/// Certified effect of the stored scale/bias radii over one input interval.
#[inline]
fn affine_model_error(x_magnitude: f32, scale_error: f32, bias_error: f32) -> f64 {
    nonnegative_add_up(
        nonnegative_mul_up(f32_to_f64_exact(x_magnitude), f32_to_f64_exact(scale_error)),
        f32_to_f64_exact(bias_error),
    )
}

/// Maximum endpoint magnitude by representation, independent of DAZ.
#[inline]
fn max_magnitude(left: f32, right: f32) -> f32 {
    let left = left.to_bits() & 0x7fff_ffff;
    let right = right.to_bits() & 0x7fff_ffff;
    f32::from_bits(left.max(right))
}

/// Evaluate the stored affine in binary64 and publish one outward endpoint.
///
/// This avoids using `fl32(fl32(x*s)+b)` as the center. A single ULP at the
/// post-cancellation result cannot cover multiplication error incurred at the
/// much larger `x*s` magnitude. Representation-exact binary32 decoding also
/// prevents DAZ from erasing a subnormal input or coefficient.
#[inline]
fn affine_endpoint(operand: f32, scale: f32, bias: f32, model_error: f64, lower: bool) -> f32 {
    let product = safe_mul_for_bounds_f64(f32_to_f64_exact(operand), f32_to_f64_exact(scale));
    let bias = f32_to_f64_exact(bias);
    if lower {
        let center = add_down_f64(product, bias);
        f64_to_f32_down(add_down_f64(center, -model_error))
    } else {
        let center = add_up_f64(product, bias);
        f64_to_f32_up(add_up_f64(center, model_error))
    }
}

impl BoundPropagation for BatchNormLayer {
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        self.validate_affine_parameters()?;
        // `TwoSum` residual direction and adjacent-f64 publication are proof
        // arithmetic only under the probed RN/gradual binary64 environment.
        // `from_scale_bias` does not pass through `BatchNormLayer::new`, so the
        // gate belongs at this consumer as well as at raw-model construction.
        require_f64_interval_proof_environment()?;
        let input_shape = input.shape();
        let layout =
            detect_input_layout(input_shape, self.num_channels, None, self.channel_axis_hint)?;

        // Create output arrays
        let mut out_lower = ArrayD::zeros(IxDyn(input_shape));
        let mut out_upper = ArrayD::zeros(IxDyn(input_shape));

        // Apply per-channel affine transform
        // y = x * scale + bias
        // If scale > 0: y_l = x_l * scale + bias, y_u = x_u * scale + bias
        // If scale < 0: y_l = x_u * scale + bias, y_u = x_l * scale + bias

        for ((idx, &l), &u) in input.lower().indexed_iter().zip(input.upper().iter()) {
            // Channel index is at the determined position
            let channel_idx = idx[layout.channel_idx];
            let s = self.scale[[channel_idx]];
            let b = self.bias[[channel_idx]];

            // The stored affine differs from exact-real BatchNorm by at most
            // `max(|l|,|u|) * scale_err + bias_err`. Form that radius with
            // upward binary64 operations, then evaluate the stored affine with
            // directed binary64 endpoints. This covers both precomputation and
            // evaluation without relying on a result-scaled f32 ULP.
            let se = self.scale_err[[channel_idx]];
            let be = self.bias_err[[channel_idx]];
            let xmag = max_magnitude(l, u);
            let model_error = affine_model_error(xmag, se, be);

            // Read the sign bit rather than comparing in floating point: DAZ
            // may otherwise reinterpret a negative-subnormal scale as -0.
            let (lower_operand, upper_operand) = if s.to_bits() >> 31 == 0 {
                (l, u)
            } else {
                (u, l)
            };
            out_lower[idx.clone()] = affine_endpoint(lower_operand, s, b, model_error, true);
            out_upper[idx] = affine_endpoint(upper_operand, s, b, model_error, false);
        }

        // Repair non-finite outputs for consistency with linear IBP.
        BoundedTensor::new_repaired(out_lower, out_upper, RepairStrategy::Conservative)
    }

    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        // BatchNorm propagate_linear lacks input shape needed for channel-axis heuristic.
        // Use propagate_linear_with_bounds instead - sequential/graph/streaming CROWN all
        // call the shape-aware method directly.
        Err(NyError::UnsupportedOp(
            "BatchNorm propagate_linear needs shape; use propagate_linear_with_bounds (CROWN paths already do)".to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        BatchNormLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

#[cfg(test)]
#[path = "ibp_certificate_tests.rs"]
mod certificate_tests;
