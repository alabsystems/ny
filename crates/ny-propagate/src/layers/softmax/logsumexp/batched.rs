// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::LogSumExpLayer;
use crate::layers::common::BoundPropagation;
use crate::BatchedLinearBounds;

impl LogSumExpLayer {
    /// Batched CROWN backward propagation through LogSumExp.
    ///
    /// The scalar LogSumExp CROWN path is a zero-slope constant relaxation based
    /// on IBP output bounds. Batched CROWN can represent that relaxation for the
    /// common last-axis `keepdims=true` case used by normalization-style graphs.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        let input_shape = pre_activation.shape();
        let ndim = input_shape.len();
        let a_shape = bounds.lower_a().shape();
        if a_shape.len() < 2 {
            return Err(NyError::InvalidSpec(
                "BatchedLinearBounds must have at least 2 dimensions".to_string(),
            ));
        }
        let input_last_dim = *input_shape
            .last()
            .ok_or_else(|| NyError::InvalidSpec("LogSumExp requires non-empty input".into()))?;
        let a_in_dim = a_shape[a_shape.len() - 1];
        let is_flat_with_groups = a_shape.len() == 2
            && input_shape.len() >= 2
            && input_last_dim > 0
            && a_in_dim != input_last_dim
            && a_in_dim.is_multiple_of(input_last_dim);
        if is_flat_with_groups {
            return Err(NyError::UnsupportedOp(
                "LogSumExp batched CROWN does not support flat block-diagonal grouped bounds"
                    .to_string(),
            ));
        }

        let axes = self.resolve_axes(ndim)?;
        if axes.len() != 1 || axes[0] + 1 != ndim {
            return Err(NyError::UnsupportedOp(format!(
                "LogSumExp batched CROWN requires single last-axis reduction, got axes={axes:?} for shape {input_shape:?}",
            )));
        }
        if !self.keepdims {
            return Err(NyError::UnsupportedOp(
                "LogSumExp batched CROWN requires keepdims=true".to_string(),
            ));
        }
        if pre_activation.lower().iter().any(|&v| !v.is_finite())
            || pre_activation.upper().iter().any(|&v| !v.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "LogSumExp batched CROWN: non-finite pre-activation bounds".to_string(),
            ));
        }

        let output_bounds = self.propagate_ibp(pre_activation)?;
        if bounds.output_shape() != output_bounds.shape() {
            return Err(NyError::ShapeMismatch {
                expected: bounds.output_shape().to_vec(),
                got: output_bounds.shape().to_vec(),
            });
        }

        let mut target_a_shape = bounds.lower_a().shape().to_vec();
        let last_dim = target_a_shape
            .last_mut()
            .ok_or_else(|| NyError::InvalidSpec("BatchedLinearBounds missing A shape".into()))?;
        *last_dim = input_last_dim;

        BatchedLinearBounds::new_or_conservative(
            ArrayD::zeros(IxDyn(&target_a_shape)),
            output_bounds.lower().mapv(next_down_f32),
            ArrayD::zeros(IxDyn(&target_a_shape)),
            output_bounds.upper().mapv(next_up_f32),
            input_shape.to_vec(),
            bounds.output_shape().to_vec(),
        )
    }
}
