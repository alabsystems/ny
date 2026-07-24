// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched graph-PGD reshape helpers that preserve a prepended leading axis.

use super::reshape::{FlattenLayer, ReshapeLayer};
use ndarray::IxDyn;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

fn reshape_standard_layout(
    input: &BoundedTensor,
    output_shape: &[usize],
    context: &str,
) -> Result<BoundedTensor> {
    let lower = input
        .lower()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(IxDyn(output_shape))
        .map_err(|err| NyError::InvalidSpec(format!("{context} failed: {err}")))?;
    let upper = input
        .upper()
        .as_standard_layout()
        .into_owned()
        .into_shape_with_order(IxDyn(output_shape))
        .map_err(|err| NyError::InvalidSpec(format!("{context} failed: {err}")))?;
    BoundedTensor::new(lower, upper)
}

impl ReshapeLayer {
    /// Execute reshape IBP while preserving a prepended restart axis.
    pub fn propagate_ibp_preserve_leading_axis(
        &self,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        let (restart_extent, sample_shape) = input.shape().split_first().ok_or_else(|| {
            NyError::InvalidSpec(
                "Reshape preserve-leading-axis requires at least 1D input".to_string(),
            )
        })?;
        let sample_output_shape = self.compute_output_shape(sample_shape)?;
        let mut output_shape = Vec::with_capacity(sample_output_shape.len() + 1);
        output_shape.push(*restart_extent);
        output_shape.extend(sample_output_shape);
        reshape_standard_layout(input, &output_shape, "Reshape preserve-leading-axis")
    }
}

impl FlattenLayer {
    /// Execute flatten IBP while preserving a prepended restart axis.
    pub fn propagate_ibp_preserve_leading_axis(
        &self,
        input: &BoundedTensor,
    ) -> Result<BoundedTensor> {
        if input.shape().is_empty() {
            return Err(NyError::InvalidSpec(
                "Flatten preserve-leading-axis requires at least 1D input".to_string(),
            ));
        }
        let axis = if self.axis < 0 {
            self.axis
        } else {
            self.axis.checked_add(1).ok_or_else(|| {
                NyError::InvalidSpec(format!(
                    "Flatten preserve-leading-axis axis overflow: {} + 1",
                    self.axis
                ))
            })?
        };
        let output_shape = FlattenLayer::new(axis).compute_output_shape(input.shape())?;
        reshape_standard_layout(input, &output_shape, "Flatten preserve-leading-axis")
    }
}
