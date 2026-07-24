// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batch filtering utilities for domain list operations.
//!
//! Provides `filter_batch` for selecting elements along the batch axis
//! of ndarray tensors based on a boolean mask.

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{checked_dim_product, NyError, Result};

/// Filter batch dimension of an array based on a mask.
pub(crate) fn filter_batch(array: &ArrayD<f32>, mask: &[bool]) -> Result<ArrayD<f32>> {
    let batch_size = array.shape().first().copied().unwrap_or(0);
    if mask.len() != batch_size {
        return Err(NyError::InvalidSpec(format!(
            "filter_batch mask length mismatch (batch={}, mask={})",
            batch_size,
            mask.len()
        )));
    }

    if batch_size == 0 {
        let mut shape = array.shape().to_vec();
        if let Some(batch_dim) = shape.first_mut() {
            *batch_dim = 0;
        } else {
            shape.push(0);
        }
        return Ok(ArrayD::zeros(IxDyn(&shape)));
    }

    let keep_count = mask.iter().filter(|&&x| x).count();
    if keep_count == 0 {
        let mut shape = array.shape().to_vec();
        shape[0] = 0;
        return Ok(ArrayD::zeros(IxDyn(&shape)));
    }

    // Build filtered array
    let element_shape = &array.shape()[1..];
    let elements_per_entry: usize = checked_dim_product(element_shape, "filter_batch")?;

    let mut result_data = Vec::with_capacity(keep_count * elements_per_entry);
    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            result_data.extend(array.index_axis(Axis(0), i).iter().copied());
        }
    }

    let mut result_shape = vec![keep_count];
    result_shape.extend_from_slice(element_shape);
    ArrayD::from_shape_vec(IxDyn(&result_shape), result_data)
        .map_err(|e| NyError::InvalidSpec(format!("failed to build filtered batch tensor: {e}")))
}
