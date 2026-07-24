// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Heuristic (sampling-based) softmax CROWN backward propagation.
//!
//! Uses local linearization at the center point with sampling-based error
//! estimates. Not sound — the error margin is estimated via Monte Carlo sampling
//! with a 10% safety factor.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::LinearBounds;

use super::super::bounds::constant_bounds_from_output;
use super::super::layer::SoftmaxLayer;

impl SoftmaxLayer {
    pub(super) fn propagate_linear_with_bounds_1d_heuristic(
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

        let num_outputs = bounds.num_outputs();

        // Check for infinite or NaN bounds - if any dimension has infinite bounds,
        // softmax linearization sampling fails. Fall back to constant bounds from [0,1].
        let has_infinite = pre_lower.iter().any(|&v| v.is_infinite() || v.is_nan())
            || pre_upper.iter().any(|&v| v.is_infinite() || v.is_nan());
        if has_infinite {
            let lower = ArrayD::from_elem(IxDyn(&[num_neurons]), 0.0);
            let upper = ArrayD::from_elem(IxDyn(&[num_neurons]), 1.0);
            // [0, 1] bounds are finite by construction — use validated constructor.
            let output_bounds = BoundedTensor::new(lower, upper)?;
            return constant_bounds_from_output(bounds, &output_bounds);
        }

        // Bit-identical linearization center: f32::midpoint rounds differently at overflow edges.
        #[allow(clippy::manual_midpoint)]
        let x_center: Array1<f32> = pre_lower
            .iter()
            .zip(pre_upper.iter())
            .map(|(&l, &u)| (l + u) / 2.0)
            .collect();

        let y_center = self.eval(&x_center);
        let jacobian = self.jacobian(&x_center);

        // Linear approximation: y ≈ J @ x + (y_c - J @ x_c)
        let jx_center = jacobian.dot(&x_center);
        let b_approx: Array1<f32> = &y_center - &jx_center;

        // Sample to find max error from linear approximation
        let num_samples = 50;
        let mut max_error_above: Array1<f32> = Array1::zeros(num_neurons); // actual - approx
        let mut max_error_below: Array1<f32> = Array1::zeros(num_neurons); // approx - actual

        // Allocate x_sample once outside loop, reuse buffer for each sample
        let mut x_sample = x_center.clone();
        for sample_idx in 0..num_samples {
            // Reset to center values (reuses allocation instead of cloning)
            x_sample.assign(&x_center);
            for i in 0..num_neurons {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t;
            }

            // Also sample corners (first few samples)
            if sample_idx < num_neurons * 2 {
                let dim = sample_idx / 2;
                if dim < num_neurons {
                    x_sample.assign(&x_center);
                    x_sample[dim] = if sample_idx % 2 == 0 {
                        pre_lower[dim]
                    } else {
                        pre_upper[dim]
                    };
                }
            }

            let y_actual = self.eval(&x_sample);
            let y_approx: Array1<f32> = jacobian.dot(&x_sample) + &b_approx;

            for i in 0..num_neurons {
                let error = y_actual[i] - y_approx[i];
                if error > max_error_above[i] {
                    max_error_above[i] = error;
                }
                if -error > max_error_below[i] {
                    max_error_below[i] = -error;
                }
            }
        }

        // Add safety margin (10% extra for unsampled regions)
        let safety_factor = 1.1;
        for i in 0..num_neurons {
            max_error_above[i] *= safety_factor;
            max_error_below[i] *= safety_factor;

            let min_margin = 1e-6_f32;
            if max_error_above[i] < min_margin {
                max_error_above[i] = min_margin;
            }
            if max_error_below[i] < min_margin {
                max_error_below[i] = min_margin;
            }
        }

        // Weight and bias accumulation use f64 to prevent catastrophic cancellation
        // (#1745, #2169). Softmax Jacobian has mixed signs (positive on diagonal,
        // negative off-diagonal, rows sum to 0), so O(num_neurons) terms with
        // cancellation make f32 accumulation unsound.
        let mut new_lower_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
        let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
        let mut new_upper_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
        let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

        for out_idx in 0..num_outputs {
            for i in 0..num_neurons {
                let la = bounds.lower_a()[[out_idx, i]];
                let ua = bounds.upper_a()[[out_idx, i]];

                // Guard: skip zero coefficients to avoid 0*inf NaN (#1739).
                if la > 0.0 {
                    let la_f64 = la as f64;
                    for k in 0..num_neurons {
                        new_lower_a_f64[[out_idx, k]] += la_f64 * jacobian[[i, k]] as f64;
                    }
                    new_lower_b_f64[out_idx] += la_f64 * (b_approx[i] - max_error_below[i]) as f64;
                } else if la < 0.0 {
                    let la_f64 = la as f64;
                    for k in 0..num_neurons {
                        new_lower_a_f64[[out_idx, k]] += la_f64 * jacobian[[i, k]] as f64;
                    }
                    new_lower_b_f64[out_idx] += la_f64 * (b_approx[i] + max_error_above[i]) as f64;
                }

                if ua > 0.0 {
                    let ua_f64 = ua as f64;
                    for k in 0..num_neurons {
                        new_upper_a_f64[[out_idx, k]] += ua_f64 * jacobian[[i, k]] as f64;
                    }
                    new_upper_b_f64[out_idx] += ua_f64 * (b_approx[i] + max_error_above[i]) as f64;
                } else if ua < 0.0 {
                    let ua_f64 = ua as f64;
                    for k in 0..num_neurons {
                        new_upper_a_f64[[out_idx, k]] += ua_f64 * jacobian[[i, k]] as f64;
                    }
                    new_upper_b_f64[out_idx] += ua_f64 * (b_approx[i] - max_error_below[i]) as f64;
                }
            }
        }

        // A-matrix: standard f64→f32 rounding (round-to-nearest), matching
        // alpha-beta-CROWN. Directed rounding on A is not unconditionally sound (#2208).
        LinearBounds::new_or_conservative(
            new_lower_a_f64.mapv(|x| x as f32),
            new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
            new_upper_a_f64.mapv(|x| x as f32),
            new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
        )
    }
}
