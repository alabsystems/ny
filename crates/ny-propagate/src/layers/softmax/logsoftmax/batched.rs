// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, Array3, ArrayD, IxDyn};
use ny_core::{checked_shape_product, NyError, Result, VerificationSoundnessMode};
use ny_tensor::BoundedTensor;
use tracing::debug;

use super::LogSoftmaxLayer;
use crate::{BatchedLinearBounds, LinearBounds};

impl LogSoftmaxLayer {
    /// Batched CROWN backward propagation through LogSoftmax.
    ///
    /// Batched CROWN treats all leading dimensions as independent batch positions,
    /// so only last-axis LogSoftmax is representable in this form.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
        soundness: VerificationSoundnessMode,
    ) -> Result<BatchedLinearBounds> {
        let pre_shape = pre_activation.shape();
        if pre_shape.is_empty() {
            return Err(NyError::InvalidSpec(
                "LogSoftmax batched CROWN requires at least 1D input".to_string(),
            ));
        }

        let a_shape = bounds.lower_a().shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }
        let pre_softmax_size = *pre_shape.last().ok_or_else(|| {
            NyError::InvalidSpec("LogSoftmax batched CROWN requires non-empty input".to_string())
        })?;
        let a_in_dim = a_shape[a_shape.len() - 1];
        let is_flat_with_groups = a_shape.len() == 2
            && pre_shape.len() >= 2
            && pre_softmax_size > 0
            && a_in_dim != pre_softmax_size
            && a_in_dim.is_multiple_of(pre_softmax_size);
        if is_flat_with_groups {
            return Err(NyError::UnsupportedOp(
                "LogSoftmax batched CROWN does not support flat block-diagonal grouped bounds"
                    .to_string(),
            ));
        }

        let axis = self.resolve_axis(pre_shape.len())?;
        if axis + 1 != pre_shape.len() {
            return Err(NyError::UnsupportedOp(format!(
                "LogSoftmax batched CROWN requires last-axis LogSoftmax, got axis {axis} for shape {pre_shape:?}",
            )));
        }

        let out_dim = a_shape[a_shape.len() - 2];
        let in_dim = a_shape[a_shape.len() - 1];
        let batch_dims = &a_shape[..a_shape.len() - 2];
        let expected_pre_batch_dims = &pre_shape[..pre_shape.len() - 1];
        if batch_dims != expected_pre_batch_dims {
            return Err(NyError::ShapeMismatch {
                expected: expected_pre_batch_dims.to_vec(),
                got: batch_dims.to_vec(),
            });
        }
        let pre_in_dim = *pre_shape.last().ok_or_else(|| NyError::ShapeMismatch {
            expected: vec![in_dim],
            got: vec![],
        })?;
        if pre_in_dim != in_dim {
            return Err(NyError::ShapeMismatch {
                expected: vec![in_dim],
                got: vec![pre_in_dim],
            });
        }

        let total_batch = checked_shape_product(batch_dims)
            .ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "LogSoftmax batched CROWN: batch dims product overflows usize: {batch_dims:?}",
                ))
            })?
            .max(1);

        debug!(
            "LogSoftmax batched CROWN backward propagation: batch_dims={batch_dims:?}, out_dim={out_dim}, in_dim={in_dim}"
        );

        let pre_lower_flat = pre_activation
            .lower()
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape pre_lower for LogSoftmax".to_string())
            })?;
        let pre_upper_flat = pre_activation
            .upper()
            .view()
            .into_shape_with_order((total_batch, in_dim))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape pre_upper for LogSoftmax".to_string())
            })?;

        let lower_a_3d = bounds
            .lower_a()
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape lower_a for LogSoftmax".to_string())
            })?;
        let upper_a_3d = bounds
            .upper_a()
            .view()
            .into_shape_with_order((total_batch, out_dim, in_dim))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape upper_a for LogSoftmax".to_string())
            })?;
        let lower_b_2d = bounds
            .lower_b()
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape lower_b for LogSoftmax".to_string())
            })?;
        let upper_b_2d = bounds
            .upper_b()
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| {
                NyError::InvalidSpec("Cannot reshape upper_b for LogSoftmax".to_string())
            })?;

        let mut new_lower_a = Array3::<f32>::zeros((total_batch, out_dim, in_dim));
        let mut new_upper_a = Array3::<f32>::zeros((total_batch, out_dim, in_dim));
        let mut new_lower_b = Array2::<f32>::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::<f32>::zeros((total_batch, out_dim));

        for batch_idx in 0..total_batch {
            let batch_bounds = LinearBounds::new_or_conservative(
                lower_a_3d.slice(ndarray::s![batch_idx, .., ..]).to_owned(),
                lower_b_2d.row(batch_idx).to_owned(),
                upper_a_3d.slice(ndarray::s![batch_idx, .., ..]).to_owned(),
                upper_b_2d.row(batch_idx).to_owned(),
            )?;
            let batch_pre = BoundedTensor::new(
                pre_lower_flat.row(batch_idx).to_owned().into_dyn(),
                pre_upper_flat.row(batch_idx).to_owned().into_dyn(),
            )?;
            let result = self.propagate_linear_with_bounds(&batch_bounds, &batch_pre, soundness)?;

            for out_idx in 0..out_dim {
                for in_idx in 0..in_dim {
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

        let out_a_shape: Vec<usize> = batch_dims
            .iter()
            .copied()
            .chain([out_dim, in_dim])
            .collect();
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
}
