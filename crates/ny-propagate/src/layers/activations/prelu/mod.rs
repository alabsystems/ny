// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::Array1;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

use super::LinearRelaxation;

/// PRelu (Parametric ReLU) layer: y = x if x >= 0, else slope * x
///
/// Unlike LeakyReLU which has a fixed slope, PRelu has learned per-channel
/// slopes. Common in detection models like RetinaNet.
#[derive(Debug, Clone)]
pub struct PReluLayer {
    /// Per-channel slopes for negative inputs (can be a single value broadcast to all channels)
    pub(crate) slope: Array1<f32>,
}

impl PReluLayer {
    /// Create a new PRelu layer with the given slopes.
    ///
    /// Returns an error if `slope` is empty, since `get_slope()` indexes
    /// with modulo by `slope.len()` which would panic on zero.
    /// Part of #2865.
    pub fn new(slope: Array1<f32>) -> Result<Self> {
        if slope.is_empty() {
            return Err(NyError::InvalidSpec(
                "PReluLayer slope must be non-empty".to_string(),
            ));
        }
        Ok(Self { slope })
    }

    /// Create a PRelu layer with a single slope value (broadcast to all elements).
    pub fn from_scalar(slope: f32) -> Self {
        Self {
            slope: Array1::from_elem(1, slope),
        }
    }

    /// The slope for a given index (modulo-based, only correct for 1D inputs).
    ///
    /// **WARNING:** This uses `idx % len` which gives wrong channel mapping
    /// for inputs with spatial dimensions (e.g. `[C, H, W]`). Production code
    /// must use `slope_for_flat()` with a precomputed stride. Retained for
    /// backward-compatible unit tests on 1D inputs only.
    #[cfg(test)]
    #[inline]
    fn slope(&self, idx: usize) -> f32 {
        if self.slope.len() == 1 {
            self.slope[0]
        } else {
            self.slope[idx % self.slope.len()]
        }
    }

    /// Compute per-channel spatial stride via shared helper. Part of #4169.
    fn per_channel_stride(&self, total_elements: usize) -> Result<usize> {
        crate::layers::common::per_channel::per_channel_spatial_stride(
            total_elements,
            self.slope.len(),
            "PReLU",
        )
    }

    /// Slope for a flat index given the per-channel stride.
    ///
    /// Maps `flat_idx / stride` to the channel index, which is correct for
    /// inputs with spatial dimensions (row-major layout: elements within a
    /// channel are contiguous). Part of #4169.
    #[inline]
    fn slope_for_flat(&self, flat_idx: usize, stride: usize) -> f32 {
        if self.slope.len() == 1 {
            self.slope[0]
        } else {
            self.slope[crate::layers::common::per_channel::channel_index_for_flat(flat_idx, stride)]
        }
    }
}

/// PReLU crossing-region chord: f64 computation with directed rounding.
///
/// Computes the chord connecting (l, alpha*l) to (u, u) in f64, converts
/// slope to f32, and recomputes the intercept from both endpoints with
/// directed rounding to ensure the bound is sound after f32 conversion.
///
/// When alpha <= 1 the function is convex at origin (chord is upper bound);
/// when alpha > 1 it is concave (chord is lower bound). Part of #3313.
pub(super) fn prelu_crossing_relaxation(l: f32, u: f32, alpha: f32) -> LinearRelaxation {
    let l_f64 = l as f64;
    let u_f64 = u as f64;
    let alpha_f64 = alpha as f64;
    let chord_slope_f64 = (u_f64 - alpha_f64 * l_f64) / (u_f64 - l_f64);
    let chord_slope_f32 = chord_slope_f64 as f32;
    let intercept_at_l = alpha_f64 * l_f64 - (chord_slope_f32 as f64) * l_f64;
    let intercept_at_u = u_f64 - (chord_slope_f32 as f64) * u_f64;
    if alpha <= 1.0 {
        // Convex at origin: chord is upper, tangent is lower.
        let chord_intercept = next_up_f32(intercept_at_l.max(intercept_at_u) as f32);
        let lower_s = if u > (-alpha * l).abs() { 1.0 } else { alpha };
        LinearRelaxation::new(lower_s, 0.0, chord_slope_f32, chord_intercept)
    } else {
        // Concave at origin: chord is lower, tangent is upper.
        let chord_intercept = next_down_f32(intercept_at_l.min(intercept_at_u) as f32);
        let upper_s = if u > (-alpha * l).abs() { alpha } else { 1.0 };
        LinearRelaxation::new(chord_slope_f32, chord_intercept, upper_s, 0.0)
    }
}

/// PReLU linear relaxation for a single neuron on interval [l, u] with slope `alpha`.
///
/// Returns `(lower_slope, lower_intercept, upper_slope, upper_intercept)`.
///
/// PReLU: `y = x` if `x >= 0`, else `alpha * x` (per-channel slope).
///
/// Cases:
/// - NaN bounds: drive CROWN to ±inf via ±inf intercepts.
/// - `l >= 0`: identity region (slope=1, intercept=0).
/// - `u <= 0`: scaled region (slope=alpha, intercept=0).
/// - Infinite bounds: case-by-case depending on alpha sign and magnitude.
/// - Crossing (`l < 0 < u`, both finite): chord + tangent relaxation,
///   convex when `alpha <= 1`, concave when `alpha > 1`.
///
/// Reference: LeakyReLU relaxation generalized to per-neuron alpha.
/// See `leaky_relu.rs:leaky_relu_linear_relaxation` for the fixed-alpha variant.
#[inline]
pub(super) fn prelu_linear_relaxation(l: f32, u: f32, alpha: f32) -> LinearRelaxation {
    if l.is_nan() || u.is_nan() || !alpha.is_finite() {
        // NaN bounds: ±inf intercepts so CROWN drives bounds to ±inf.
        LinearRelaxation::nan_fallback()
    } else if (u - l).abs() < 1e-8 {
        // Denominator guard for near-point intervals. alpha-beta-CROWN's tensor
        // ReLU relaxation similarly floors (u-l) by +1e-8 before division
        // (auto_LiRPA/operators/relu.py::_relu_upper_bound).
        let y_l = if l >= 0.0 { l } else { alpha * l };
        let y_u = if u >= 0.0 { u } else { alpha * u };
        let mut y_min = nan_propagating_min(y_l, y_u);
        let mut y_max = nan_propagating_max(y_l, y_u);
        // For alpha < 0 and l < 0 < u, PReLU has a cusp minimum at x=0.
        if l < 0.0 && u > 0.0 {
            y_min = nan_propagating_min(y_min, 0.0);
            y_max = nan_propagating_max(y_max, 0.0);
        }
        LinearRelaxation::new(0.0, y_min, 0.0, y_max)
    } else if l >= 0.0 {
        // Always positive: identity
        LinearRelaxation::identity()
    } else if u <= 0.0 {
        // Always negative: scaled by alpha
        LinearRelaxation::new(alpha, 0.0, alpha, 0.0)
    } else if l.is_infinite() && u.is_infinite() {
        // Both infinite: alpha <= 1: y = alpha*x is global lower.
        // alpha > 1: no finite affine lower bound exists.
        if alpha <= 1.0 {
            LinearRelaxation::new(alpha, 0.0, 0.0, f32::INFINITY)
        } else {
            LinearRelaxation::nan_fallback()
        }
    } else if l.is_infinite() {
        // l = -inf, u > 0.
        // alpha < 0: upper must have negative slope to stay above alpha*x as x->-inf.
        //   Use upper y=alpha*x + (1-alpha)*u and lower y=x.
        // alpha in [0,1]: lower y=alpha*x, upper y=u.
        // alpha > 1: no finite affine lower bound (use -inf intercept).
        if alpha < 0.0 {
            LinearRelaxation::new(1.0, 0.0, alpha, (1.0 - alpha) * u)
        } else if alpha <= 1.0 {
            LinearRelaxation::new(alpha, 0.0, 0.0, u)
        } else {
            LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, u)
        }
    } else if u.is_infinite() {
        // l < 0, u = +inf.
        // alpha <= 1: y = alpha*x. alpha > 1: y = x + (alpha-1)*l.
        let upper_s = nan_propagating_max(alpha, 1.0);
        if alpha <= 1.0 {
            LinearRelaxation::new(alpha, 0.0, upper_s, l * (alpha - upper_s))
        } else {
            LinearRelaxation::new(1.0, (alpha - 1.0) * l, upper_s, l * (alpha - upper_s))
        }
    } else {
        // Crossing region (l < 0 < u, both finite).
        prelu_crossing_relaxation(l, u, alpha)
    }
}

impl BoundPropagation for PReluLayer {
    /// IBP for PRelu: y = x if x >= 0, else slope * x
    ///
    /// Similar to LeakyReLU but with per-channel slopes.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let flat_lower = input
            .lower()
            .view()
            .into_shape_with_order(input.lower().len())
            .map_err(|e| {
                NyError::InvalidSpec(format!("PRelu: failed to flatten lower bounds: {e}"))
            })?;
        let flat_upper = input
            .upper()
            .view()
            .into_shape_with_order(input.upper().len())
            .map_err(|e| {
                NyError::InvalidSpec(format!("PRelu: failed to flatten upper bounds: {e}"))
            })?;

        let mut lower = Array1::zeros(flat_lower.len());
        let mut upper = Array1::zeros(flat_upper.len());

        // Per-channel slope: compute stride so flat index maps to correct channel.
        // For [C, T] input with C-channel slope, stride = T, channel = i / T.
        // Part of #4168.
        let stride = self.per_channel_stride(flat_lower.len())?;
        for i in 0..flat_lower.len() {
            let slope = self.slope_for_flat(i, stride);
            let l = flat_lower[i];
            let u = flat_upper[i];

            if !l.is_finite() || !u.is_finite() || !slope.is_finite() {
                lower[i] = f32::NEG_INFINITY;
                upper[i] = f32::INFINITY;
                continue;
            }
            if slope >= 0.0 {
                lower[i] = if l >= 0.0 { l } else { slope * l };
                upper[i] = if u >= 0.0 { u } else { slope * u };
            } else if l >= 0.0 {
                lower[i] = l;
                upper[i] = u;
            } else if u <= 0.0 {
                lower[i] = slope * u;
                upper[i] = slope * l;
            } else {
                // Crossing case: slope < 0, l < 0, u > 0.
                // PReLU has a cusp minimum at x=0 where PReLU(0) = 0.
                // For x < 0: PReLU(x) = slope * x > 0 (slope < 0, x < 0).
                // For x >= 0: PReLU(x) = x >= 0.
                // True minimum is 0 at x=0; true maximum is max(slope*l, u) at endpoints.
                // Previous code incorrectly included slope*u (slope applied to positive x).
                // Part of #1914.
                lower[i] = 0.0;
                upper[i] = nan_propagating_max(slope * l, u);
            }
        }
        let lower = lower
            .into_shape_with_order(input.shape())
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "PRelu: failed to reshape lower bounds to {:?}: {e}",
                    input.shape()
                ))
            })?
            .to_owned();
        let upper = upper
            .into_shape_with_order(input.shape())
            .map_err(|e| {
                NyError::InvalidSpec(format!(
                    "PRelu: failed to reshape upper bounds to {:?}: {e}",
                    input.shape()
                ))
            })?
            .to_owned();
        BoundedTensor::new_allow_infinite(lower, upper)
    }

    impl_elementwise_activation!(
        @trait_methods
        PReluLayer,
        NyError::InvalidSpec(
            "PRelu CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

impl PReluLayer {
    /// CROWN backward propagation with pre-activation bounds (stride-based).
    ///
    /// Computes per-channel stride from `pre_activation.len()` before building
    /// the relaxation closure, so each flat neuron index maps to the correct
    /// channel slope even for spatial inputs (`[C, H, W]`). Part of #4168.
    #[inline]
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &crate::LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::LinearBounds> {
        crate::layers::common::non_finite_domain_guard("PReLU", pre_activation)?;
        let stride = self.per_channel_stride(pre_activation.len())?;
        let relax_fn = |l: f32, u: f32, i: usize| {
            prelu_linear_relaxation(l, u, self.slope_for_flat(i, stride))
        };
        crate::layers::common::crown_elementwise_backward_indexed(bounds, pre_activation, relax_fn)
    }

    /// Batched CROWN backward propagation with pre-activation bounds (stride-based).
    ///
    /// Same stride-based channel mapping as `propagate_linear_with_bounds`.
    /// Part of #4168.
    #[inline]
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &crate::BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::BatchedLinearBounds> {
        crate::layers::common::non_finite_domain_guard("PReLU", pre_activation)?;
        let stride = self.per_channel_stride(pre_activation.len())?;
        let relax_fn = |l: f32, u: f32, i: usize| {
            prelu_linear_relaxation(l, u, self.slope_for_flat(i, stride))
        };
        crate::layers::common::crown_elementwise_backward_batched_indexed(
            bounds,
            pre_activation,
            relax_fn,
        )
    }
}

#[cfg(test)]
mod tests;
