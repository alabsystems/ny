// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound softmax CROWN backward propagation via LSE affine bounds.
//!
//! Computes provably valid linear relaxations of softmax using the
//! log-sum-exp (LSE) decomposition: softmax_i(x) = exp(x_i) / sum_exp(x).
//! The numerator exp(x_i) is bounded by its chord (upper) and tangent (lower),
//! and the denominator sum_exp is bounded similarly, yielding affine bounds
//! on each softmax output.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::bounds::nan_propagating_max;
use crate::LinearBounds;

use super::super::bounds::constant_bounds_from_output;
use super::super::layer::SoftmaxLayer;
use super::super::utils;

impl SoftmaxLayer {
    pub(super) fn propagate_linear_with_bounds_1d_sound(
        &self,
        bounds: &LinearBounds,
        pre_lower: &Array1<f32>,
        pre_upper: &Array1<f32>,
    ) -> Result<LinearBounds> {
        let num_neurons = pre_lower.len();
        if pre_upper.len() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![pre_upper.len()],
            });
        }
        if bounds.num_inputs() != num_neurons {
            return Err(NyError::ShapeMismatch {
                expected: vec![num_neurons],
                got: vec![bounds.num_inputs()],
            });
        }

        let has_non_finite =
            pre_lower.iter().any(|&v| !v.is_finite()) || pre_upper.iter().any(|&v| !v.is_finite());
        if has_non_finite {
            let lower = ArrayD::from_elem(IxDyn(&[num_neurons]), 0.0);
            let upper = ArrayD::from_elem(IxDyn(&[num_neurons]), 1.0);
            // [0, 1] bounds are finite by construction — use validated constructor.
            let output_bounds = BoundedTensor::new(lower, upper)?;
            return constant_bounds_from_output(bounds, &output_bounds);
        }

        let Some((lower_a, lower_b, upper_a, upper_b)) =
            self.softmax_lse_affine_bounds(pre_lower, pre_upper)
        else {
            let lower = ArrayD::from_elem(IxDyn(&[num_neurons]), 0.0);
            let upper = ArrayD::from_elem(IxDyn(&[num_neurons]), 1.0);
            // [0, 1] bounds are finite by construction — use validated constructor.
            let output_bounds = BoundedTensor::new(lower, upper)?;
            return constant_bounds_from_output(bounds, &output_bounds);
        };

        self.apply_affine_bounds(bounds, &lower_a, &lower_b, &upper_a, &upper_b)
    }

    // Justification: The 4-tuple return (lower_A, lower_b, upper_A, upper_b) represents
    // the standard affine bound coefficient structure for softmax-via-LSE relaxation.
    // Extracting a struct would add overhead for a private method with one call site.
    #[allow(clippy::type_complexity)]
    pub(super) fn softmax_lse_affine_bounds(
        &self,
        pre_lower: &Array1<f32>,
        pre_upper: &Array1<f32>,
    ) -> Option<(Array2<f32>, Array1<f32>, Array2<f32>, Array1<f32>)> {
        let num_neurons = pre_lower.len();
        // Bit-identical linearization center: f32::midpoint rounds differently at overflow edges.
        #[allow(clippy::manual_midpoint)]
        let x_center: Array1<f32> = pre_lower
            .iter()
            .zip(pre_upper.iter())
            .map(|(&l, &u)| (l + u) / 2.0)
            .collect();

        // Directed rounding: lse_upper must not understate, lse_lower must not
        // overstate. logsumexp_1d computes in f64 but casts to f32 with nearest
        // rounding — wrap with next_up/next_down to match logsumexp_directed
        // pattern from logsoftmax/mod.rs:72-100. See #3275 Gap 1.
        let lse_upper = next_up_f32(utils::logsumexp_1d(pre_upper));
        let lse_lower = next_down_f32(utils::logsumexp_1d(pre_lower));
        let lse_center = utils::logsumexp_1d(&x_center);
        if !lse_upper.is_finite() || !lse_lower.is_finite() || !lse_center.is_finite() {
            return None;
        }

        let softmax_center = utils::softmax_1d(&x_center);
        if softmax_center.iter().any(|&v| !v.is_finite()) {
            return None;
        }

        // NaN-propagating fold: NaN in pre_upper must propagate — see #2577.
        let shift = pre_upper
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, nan_propagating_max);
        if !shift.is_finite() {
            return None;
        }

        let mut se_slope = Array1::<f32>::zeros(num_neurons);
        let mut se_intercept = Array1::<f32>::zeros(num_neurons);
        for i in 0..num_neurons {
            let l = pre_lower[i];
            let u = pre_upper[i];
            if !l.is_finite() || !u.is_finite() {
                return None;
            }
            // f64 intermediates for sum-exp chord to reduce rounding error.
            // Nearest rounding on cast: se_slope/se_intercept feed se_center
            // which has its own rationale for nearest rounding (lines 140-142).
            // See #3275 Gap 2.
            if (u - l).abs() < 1e-8 {
                let exp_l = ((l - shift) as f64).exp();
                se_slope[i] = 0.0;
                se_intercept[i] = exp_l as f32;
            } else {
                let exp_l = ((l - shift) as f64).exp();
                let exp_u = ((u - shift) as f64).exp();
                let slope_f64 = (exp_u - exp_l) / (u as f64 - l as f64);
                let intercept_f64 = exp_l - slope_f64 * l as f64;
                se_slope[i] = slope_f64 as f32;
                se_intercept[i] = intercept_f64 as f32;
            }
        }

        // #2423: f64 accumulation for se_center to prevent precision loss when
        // num_neurons is large. This value feeds the lower bound denominators
        // (lower_val = exp_center / se_center, ratio = se_slope[k] / se_center).
        let se_center_f64: f64 = se_slope
            .iter()
            .zip(x_center.iter())
            .map(|(&s, &x)| s as f64 * x as f64)
            .sum::<f64>()
            + se_intercept.iter().map(|&v| v as f64).sum::<f64>();
        // Nearest rounding: se_center feeds both lower_val and lower_a slopes.
        // Directed rounding would change slopes with unpredictable second-order
        // effects on the affine bound; the final intercept has its own next_down_f32.
        let se_center = se_center_f64 as f32;
        if !se_center.is_finite() || se_center <= 0.0 {
            return None;
        }

        let mut lower_a = Array2::<f32>::zeros((num_neurons, num_neurons));
        let mut lower_b = Array1::<f32>::zeros(num_neurons);
        let mut upper_a = Array2::<f32>::zeros((num_neurons, num_neurons));
        let mut upper_b = Array1::<f32>::zeros(num_neurons);

        for i in 0..num_neurons {
            let exp_center = (x_center[i] - shift).exp();
            if !exp_center.is_finite() {
                return None;
            }
            let lower_val = exp_center / se_center;
            if !lower_val.is_finite() {
                return None;
            }

            // #2423: f64 accumulation for lower_dot to prevent precision loss
            // in the intercept computation. num_neurons mixed-sign terms cancel.
            let mut lower_dot = 0.0_f64;
            for k in 0..num_neurons {
                let ratio = se_slope[k] / se_center;
                let grad = if k == i {
                    lower_val * (1.0 - ratio)
                } else {
                    -lower_val * ratio
                };
                lower_a[[i, k]] = grad;
                lower_dot += grad as f64 * x_center[k] as f64;
            }
            // Directed rounding: lower intercept must not overstate the bound.
            lower_b[i] = next_down_f32((lower_val as f64 - lower_dot) as f32);

            let r_min = pre_lower[i] - lse_upper;
            let r_max = pre_upper[i] - lse_lower;
            let (r_lo, r_hi) = if r_min <= r_max {
                (r_min, r_max)
            } else {
                (r_max, r_min)
            };
            if !r_lo.is_finite() || !r_hi.is_finite() {
                return None;
            }

            // f64 intermediates for upper-bound exp chord. Slope uses nearest
            // rounding (directed rounding on A-coefficients is not unconditionally
            // sound, #2208). Intercept uses next_up_f32 — upper bound bias must
            // not understate. See #3275 Gap 2.
            let (slope, intercept) = if (r_hi - r_lo).abs() < 1e-6 {
                let exp_r_f64 = (r_lo as f64).exp();
                let intercept_f64 = exp_r_f64 - exp_r_f64 * r_lo as f64;
                (exp_r_f64 as f32, next_up_f32(intercept_f64 as f32))
            } else {
                let exp_r_lo = (r_lo as f64).exp();
                let exp_r_hi = (r_hi as f64).exp();
                let slope_f64 = (exp_r_hi - exp_r_lo) / (r_hi as f64 - r_lo as f64);
                let intercept_f64 = exp_r_lo - slope_f64 * r_lo as f64;
                (slope_f64 as f32, next_up_f32(intercept_f64 as f32))
            };

            let r_center = x_center[i] - lse_center;
            let upper_val = slope * r_center + intercept;
            if !upper_val.is_finite() {
                return None;
            }

            // #2423: f64 accumulation for upper_dot, same rationale as lower_dot.
            let mut upper_dot = 0.0_f64;
            for k in 0..num_neurons {
                let grad = slope * ((if k == i { 1.0 } else { 0.0 }) - softmax_center[k]);
                upper_a[[i, k]] = grad;
                upper_dot += grad as f64 * x_center[k] as f64;
            }
            // Directed rounding: upper intercept must not understate the bound.
            upper_b[i] = next_up_f32((upper_val as f64 - upper_dot) as f32);
        }

        Some((lower_a, lower_b, upper_a, upper_b))
    }

    pub(super) fn apply_affine_bounds(
        &self,
        bounds: &LinearBounds,
        lower_a: &Array2<f32>,
        lower_b: &Array1<f32>,
        upper_a: &Array2<f32>,
        upper_b: &Array1<f32>,
    ) -> Result<LinearBounds> {
        utils::apply_affine_bounds_f64(bounds, lower_a, lower_b, upper_a, upper_b)
    }
}
