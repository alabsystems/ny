// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result, VerificationSoundnessMode};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use tracing::debug;

use super::super::bounds::batched_constant_bounds_from_output;
use super::CausalSoftmaxLayer;
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

impl CausalSoftmaxLayer {
    /// Batched CROWN backward propagation through CausalSoftmax.
    ///
    /// The batched representation treats the query-row index as the last batch
    /// dimension, so each batch position propagates one causal-softmax row.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
        soundness: VerificationSoundnessMode,
    ) -> Result<BatchedLinearBounds> {
        let pre_shape = pre_activation.shape();
        let ndim = pre_shape.len();
        if ndim < 2 {
            return Err(NyError::InvalidSpec(format!(
                "Causal softmax batched CROWN requires at least 2D input, got {ndim}D",
            )));
        }

        let a_shape = bounds.lower_a().shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }

        let seq_k = pre_shape[ndim - 1];
        let a_in_dim = a_shape[a_shape.len() - 1];
        let is_flat_with_groups = a_shape.len() == 2
            && ndim >= 2
            && seq_k > 0
            && a_in_dim != seq_k
            && a_in_dim.is_multiple_of(seq_k);
        if is_flat_with_groups {
            return Err(NyError::UnsupportedOp(
                "CausalSoftmax batched CROWN does not support flat block-diagonal grouped bounds"
                    .to_string(),
            ));
        }

        let effective_soundness = if self.sound {
            if soundness == VerificationSoundnessMode::Heuristic {
                debug!(
                    "CausalSoftmax batched: heuristic requested, but layer is in sound mode; using IBP constant bounds"
                );
            }
            VerificationSoundnessMode::Sound
        } else {
            soundness
        };

        if effective_soundness == VerificationSoundnessMode::Sound {
            let output_bounds = self.propagate_ibp(pre_activation)?;
            return batched_constant_bounds_from_output(bounds, &output_bounds);
        }

        if pre_activation.lower().iter().any(|&v| !v.is_finite())
            || pre_activation.upper().iter().any(|&v| !v.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "CausalSoftmax batched heuristic CROWN: non-finite pre-activation bounds"
                    .to_string(),
            ));
        }

        let seq_q = pre_shape[ndim - 2];
        let out_dim = a_shape[a_shape.len() - 2];
        let in_dim = a_shape[a_shape.len() - 1];
        if in_dim != seq_k {
            return Err(NyError::ShapeMismatch {
                expected: vec![seq_k],
                got: vec![in_dim],
            });
        }

        let batch_dims = &a_shape[..a_shape.len() - 2];
        let expected_pre_batch_dims = &pre_shape[..ndim - 1];
        if batch_dims != expected_pre_batch_dims {
            return Err(NyError::ShapeMismatch {
                expected: expected_pre_batch_dims.to_vec(),
                got: batch_dims.to_vec(),
            });
        }

        let total_batch = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "CausalSoftmax batched CROWN: batch dims product overflows usize: {batch_dims:?}",
                ))
            })?
            .max(1);

        let pre_lower_flat = pre_activation
            .lower()
            .view()
            .into_shape_with_order((total_batch, seq_k))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape pre_lower for CausalSoftmax".to_string())
            })?;
        let pre_upper_flat = pre_activation
            .upper()
            .view()
            .into_shape_with_order((total_batch, seq_k))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape pre_upper for CausalSoftmax".to_string())
            })?;

        let lower_a_3d = bounds
            .lower_a()
            .view()
            .into_shape_with_order((total_batch, out_dim, seq_k))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape lower_a for CausalSoftmax".to_string())
            })?;
        let upper_a_3d = bounds
            .upper_a()
            .view()
            .into_shape_with_order((total_batch, out_dim, seq_k))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape upper_a for CausalSoftmax".to_string())
            })?;
        let lower_b_2d = bounds
            .lower_b()
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape lower_b for CausalSoftmax".to_string())
            })?;
        let upper_b_2d = bounds
            .upper_b()
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape upper_b for CausalSoftmax".to_string())
            })?;

        let mut new_lower_a = Array3::<f32>::zeros((total_batch, out_dim, seq_k));
        let mut new_upper_a = Array3::<f32>::zeros((total_batch, out_dim, seq_k));
        let mut new_lower_b = Array2::<f32>::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::<f32>::zeros((total_batch, out_dim));

        for batch_idx in 0..total_batch {
            let row_idx = if seq_q == 0 { 0 } else { batch_idx % seq_q };
            let batch_bounds = LinearBounds::new_or_conservative(
                lower_a_3d.slice(ndarray::s![batch_idx, .., ..]).to_owned(),
                lower_b_2d.row(batch_idx).to_owned(),
                upper_a_3d.slice(ndarray::s![batch_idx, .., ..]).to_owned(),
                upper_b_2d.row(batch_idx).to_owned(),
            )?;
            let result = self.propagate_linear_row_with_bounds_heuristic(
                &batch_bounds,
                &pre_lower_flat.row(batch_idx).to_owned(),
                &pre_upper_flat.row(batch_idx).to_owned(),
                row_idx,
            )?;

            for out_idx in 0..out_dim {
                for in_idx in 0..seq_k {
                    new_lower_a[[batch_idx, out_idx, in_idx]] = result.lower_a()[[out_idx, in_idx]];
                    new_upper_a[[batch_idx, out_idx, in_idx]] = result.upper_a()[[out_idx, in_idx]];
                }
                new_lower_b[[batch_idx, out_idx]] = result.lower_b()[out_idx];
                new_upper_b[[batch_idx, out_idx]] = result.upper_b()[out_idx];
            }
        }

        let (new_lower_a_vec, _) = new_lower_a.into_raw_vec_and_offset();
        let (new_upper_a_vec, _) = new_upper_a.into_raw_vec_and_offset();
        let (new_lower_b_vec, _) = new_lower_b.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b.into_raw_vec_and_offset();

        let out_a_shape: Vec<usize> = batch_dims.iter().copied().chain([out_dim, seq_k]).collect();
        let out_b_shape: Vec<usize> = batch_dims.iter().copied().chain([out_dim]).collect();

        BatchedLinearBounds::new_or_conservative(
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_lower_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_lower_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_upper_a_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_a".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_upper_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
            pre_shape.to_vec(),
            bounds.output_shape().to_vec(),
        )
    }

    pub(super) fn propagate_linear_row_with_bounds_heuristic(
        &self,
        bounds: &LinearBounds,
        pre_lower: &Array1<f32>,
        pre_upper: &Array1<f32>,
        row_idx: usize,
    ) -> Result<LinearBounds> {
        let seq_k = pre_lower.len();
        if pre_upper.len() != seq_k {
            return Err(NyError::ShapeMismatch {
                expected: vec![seq_k],
                got: vec![pre_upper.len()],
            });
        }
        if bounds.num_inputs() != seq_k {
            return Err(NyError::ShapeMismatch {
                expected: vec![seq_k],
                got: vec![bounds.num_inputs()],
            });
        }

        let num_outputs = bounds.num_outputs();
        // Bit-identical linearization center: f32::midpoint rounds differently at overflow edges.
        #[allow(clippy::manual_midpoint)]
        let x_center: Array1<f32> = pre_lower
            .iter()
            .zip(pre_upper.iter())
            .map(|(&l, &u)| (l + u) / 2.0)
            .collect();
        let y_center = self.eval_row(&x_center, row_idx);
        let jacobian = self.jacobian_row(&x_center, row_idx);
        let jx_center = jacobian.dot(&x_center);
        let b_approx: Array1<f32> = &y_center - &jx_center;

        let mut max_error_above = Array1::<f32>::zeros(seq_k);
        let mut max_error_below = Array1::<f32>::zeros(seq_k);
        let num_samples = 50;
        let mut x_sample = x_center.clone();

        for sample_idx in 0..num_samples {
            x_sample.assign(&x_center);
            for i in 0..seq_k {
                let t = ((sample_idx as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                x_sample[i] = pre_lower[i] + (pre_upper[i] - pre_lower[i]) * t;
            }

            if sample_idx < seq_k * 2 {
                let dim = sample_idx / 2;
                if dim < seq_k {
                    x_sample.assign(&x_center);
                    x_sample[dim] = if sample_idx % 2 == 0 {
                        pre_lower[dim]
                    } else {
                        pre_upper[dim]
                    };
                }
            }

            let y_actual = self.eval_row(&x_sample, row_idx);
            let y_approx: Array1<f32> = jacobian.dot(&x_sample) + &b_approx;
            for i in 0..seq_k {
                let error = y_actual[i] - y_approx[i];
                if error > max_error_above[i] {
                    max_error_above[i] = error;
                }
                if -error > max_error_below[i] {
                    max_error_below[i] = -error;
                }
            }
        }

        let safety_factor = 1.1_f32;
        for i in 0..seq_k {
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

        let mut new_lower_a_f64 = Array2::<f64>::zeros((num_outputs, seq_k));
        let mut new_lower_b_f64 = bounds.lower_b().mapv(f64::from);
        let mut new_upper_a_f64 = Array2::<f64>::zeros((num_outputs, seq_k));
        let mut new_upper_b_f64 = bounds.upper_b().mapv(f64::from);

        for out_idx in 0..num_outputs {
            for i in 0..seq_k {
                let lower_coeff = bounds.lower_a()[[out_idx, i]];
                let upper_coeff = bounds.upper_a()[[out_idx, i]];

                if lower_coeff > 0.0 {
                    let coeff = f64::from(lower_coeff);
                    for k in 0..seq_k {
                        new_lower_a_f64[[out_idx, k]] += coeff * f64::from(jacobian[[i, k]]);
                    }
                    new_lower_b_f64[out_idx] += coeff * f64::from(b_approx[i] - max_error_below[i]);
                } else if lower_coeff < 0.0 {
                    let coeff = f64::from(lower_coeff);
                    for k in 0..seq_k {
                        new_lower_a_f64[[out_idx, k]] += coeff * f64::from(jacobian[[i, k]]);
                    }
                    new_lower_b_f64[out_idx] += coeff * f64::from(b_approx[i] + max_error_above[i]);
                }

                if upper_coeff > 0.0 {
                    let coeff = f64::from(upper_coeff);
                    for k in 0..seq_k {
                        new_upper_a_f64[[out_idx, k]] += coeff * f64::from(jacobian[[i, k]]);
                    }
                    new_upper_b_f64[out_idx] += coeff * f64::from(b_approx[i] + max_error_above[i]);
                } else if upper_coeff < 0.0 {
                    let coeff = f64::from(upper_coeff);
                    for k in 0..seq_k {
                        new_upper_a_f64[[out_idx, k]] += coeff * f64::from(jacobian[[i, k]]);
                    }
                    new_upper_b_f64[out_idx] += coeff * f64::from(b_approx[i] - max_error_below[i]);
                }
            }
        }

        LinearBounds::new_or_conservative(
            new_lower_a_f64.mapv(|value| value as f32),
            new_lower_b_f64.mapv(|value| next_down_f32(value as f32)),
            new_upper_a_f64.mapv(|value| value as f32),
            new_upper_b_f64.mapv(|value| next_up_f32(value as f32)),
        )
    }
}
