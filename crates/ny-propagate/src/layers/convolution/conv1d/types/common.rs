// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared batch-shape context and helpers for Conv1d / ConvTranspose1d
//! batched CROWN backward propagation.
//!
//! Both layer types share identical scaffolding around their type-specific
//! kernel calls: shape validation, batch-dim computation, non-finite row
//! zeroing, and bias finalization. This module centralizes that scaffold
//! so each type file only contains the kernel-specific math.

use ndarray::{s, Array1, Array2, ArrayD, ArrayView3, IxDyn};
use ny_core::{checked_shape_product, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};
use tracing::debug;

use crate::{contiguous_flat_slice_mut, BatchedLinearBounds};

/// Pre-computed batch geometry for a Conv1d backward pass.
pub(super) struct BackwardBatchContext {
    pub total_batch: usize,
    pub total_rows: usize,
    pub out_dim: usize,
    pub mid_dim: usize,
    pub out_a_shape: Vec<usize>,
    pub out_b_shape: Vec<usize>,
}

/// Validate the incoming `BatchedLinearBounds` shape and derive all
/// batch-dimension bookkeeping needed by the backward pass.
pub(super) fn build_backward_batch_context(
    bounds: &BatchedLinearBounds,
    conv_out_size: usize,
    conv_in_size: usize,
    op_name: &str,
) -> Result<BackwardBatchContext> {
    let a_shape = bounds.lower_a.shape();
    if a_shape.len() < 2 {
        return Err(NyError::InvalidSpec(
            "BatchedLinearBounds must have at least 2 dimensions".to_string(),
        ));
    }

    let out_dim = a_shape[a_shape.len() - 2];
    let mid_dim = a_shape[a_shape.len() - 1];

    if mid_dim != conv_out_size {
        return Err(NyError::ShapeMismatch {
            expected: vec![out_dim, conv_out_size],
            got: vec![out_dim, mid_dim],
        });
    }

    let batch_dims = &a_shape[..a_shape.len() - 2];
    let total_batch = checked_shape_product(batch_dims).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "{op_name} CROWN: batch dims product overflows: {batch_dims:?}",
        ))
    })?;
    let total_batch = total_batch.max(1);

    let mut out_a_shape: Vec<usize> = batch_dims.to_vec();
    out_a_shape.push(out_dim);
    out_a_shape.push(conv_in_size);

    let mut out_b_shape: Vec<usize> = batch_dims.to_vec();
    out_b_shape.push(out_dim);

    let total_rows = total_batch * out_dim;

    Ok(BackwardBatchContext {
        total_batch,
        total_rows,
        out_dim,
        mid_dim,
        out_a_shape,
        out_b_shape,
    })
}

/// Zero A-matrix rows that contain non-finite coefficients (#3256, #2812).
///
/// Rows flagged as non-finite get all coefficients set to 0.0 so the
/// corresponding bias will be overridden to ±inf in `finalize_bias_bounds`.
pub(super) fn zero_nonfinite_rows(
    new_lower_a: &mut Array2<f32>,
    new_upper_a: &mut Array2<f32>,
    lower_nonfinite_rows: &[bool],
    upper_nonfinite_rows: &[bool],
    conv_in_size: usize,
    total_rows: usize,
    op_name: &str,
) {
    let lower_affected = lower_nonfinite_rows.iter().filter(|&&r| r).count();
    let upper_affected = upper_nonfinite_rows.iter().filter(|&&r| r).count();
    if lower_affected > 0 || upper_affected > 0 {
        debug!(
            "{op_name} batched CROWN backward: non-finite A-matrix in {lower_affected}/{total_rows} lower, \
             {upper_affected}/{total_rows} upper rows — ±inf bias fallback",
        );
        for i in 0..total_rows {
            if lower_nonfinite_rows[i] {
                for j in 0..conv_in_size {
                    new_lower_a[[i, j]] = 0.0;
                }
            }
            if upper_nonfinite_rows[i] {
                for j in 0..conv_in_size {
                    new_upper_a[[i, j]] = 0.0;
                }
            }
        }
    }
}

/// Compute the output bias bounds, accounting for convolution bias and
/// non-finite A-matrix rows.
///
/// When `bias` is `Some`, accumulates `sum_c sum_l A[c*out_len+l] * bias[c]`
/// in f64, applies NaN guards, and overrides non-finite rows to ±inf.
/// When `bias` is `None`, clones the input bias and only applies non-finite
/// overrides.
#[allow(clippy::too_many_arguments)]
pub(super) fn finalize_bias_bounds(
    bounds: &BatchedLinearBounds,
    ctx: &BackwardBatchContext,
    out_c: usize,
    out_len: usize,
    lower_a_3d: ArrayView3<'_, f32>,
    upper_a_3d: ArrayView3<'_, f32>,
    bias: Option<&Array1<f32>>,
    lower_nonfinite_rows: &[bool],
    upper_nonfinite_rows: &[bool],
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    let total_batch = ctx.total_batch;
    let out_dim = ctx.out_dim;
    let total_rows = ctx.total_rows;

    if let Some(bias) = bias {
        let lower_b_3d = bounds
            .lower_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape lower_b".to_string()))?;
        let upper_b_3d = bounds
            .upper_b
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|_| NyError::InvalidSpec("Cannot reshape upper_b".to_string()))?;

        let mut new_lower_b = Array2::<f64>::zeros((total_batch, out_dim));
        let mut new_upper_b = Array2::<f64>::zeros((total_batch, out_dim));

        for b in 0..total_batch {
            for d in 0..out_dim {
                let mut lower_sum = 0.0_f64;
                let mut upper_sum = 0.0_f64;

                for c in 0..out_c {
                    let spatial_start = c * out_len;
                    let spatial_end = spatial_start + out_len;

                    let lower_spatial_sum: f64 = lower_a_3d
                        .slice(s![b, d, spatial_start..spatial_end])
                        .iter()
                        .map(|&v| v as f64)
                        .sum();
                    let upper_spatial_sum: f64 = upper_a_3d
                        .slice(s![b, d, spatial_start..spatial_end])
                        .iter()
                        .map(|&v| v as f64)
                        .sum();

                    lower_sum += lower_spatial_sum * bias[c] as f64;
                    upper_sum += upper_spatial_sum * bias[c] as f64;
                }

                // NaN guard: inf + (-inf) → conservative bounds.
                // Same pattern as BatchedLinearBounds::compose().
                let lb_sum = lower_b_3d[[b, d]] as f64 + lower_sum;
                let ub_sum = upper_b_3d[[b, d]] as f64 + upper_sum;
                new_lower_b[[b, d]] = if lb_sum.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    lb_sum
                };
                new_upper_b[[b, d]] = if ub_sum.is_nan() {
                    f64::INFINITY
                } else {
                    ub_sum
                };
            }
        }

        // #3256: Override bias for non-finite A-matrix rows.
        for b in 0..total_batch {
            for d in 0..out_dim {
                let row_idx = b * out_dim + d;
                if lower_nonfinite_rows[row_idx] {
                    new_lower_b[[b, d]] = f64::NEG_INFINITY;
                }
                if upper_nonfinite_rows[row_idx] {
                    new_upper_b[[b, d]] = f64::INFINITY;
                }
            }
        }

        let new_lower_b_f32 = new_lower_b.mapv(|v| next_down_f32(v as f32));
        let new_upper_b_f32 = new_upper_b.mapv(|v| next_up_f32(v as f32));
        let (new_lower_b_vec, _) = new_lower_b_f32.into_raw_vec_and_offset();
        let (new_upper_b_vec, _) = new_upper_b_f32.into_raw_vec_and_offset();
        Ok((
            ArrayD::from_shape_vec(IxDyn(&ctx.out_b_shape), new_lower_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_lower_b".to_string()))?,
            ArrayD::from_shape_vec(IxDyn(&ctx.out_b_shape), new_upper_b_vec)
                .map_err(|_| NyError::InvalidSpec("Cannot reshape new_upper_b".to_string()))?,
        ))
    } else {
        // #3256: Even without conv bias, override for non-finite A-matrix rows.
        let mut lb = bounds.lower_b.clone();
        let mut ub = bounds.upper_b.clone();
        let lower_affected = lower_nonfinite_rows.iter().any(|&r| r);
        let upper_affected = upper_nonfinite_rows.iter().any(|&r| r);
        if lower_affected || upper_affected {
            let lb_flat = contiguous_flat_slice_mut(&mut lb)?;
            let ub_flat = contiguous_flat_slice_mut(&mut ub)?;
            for i in 0..total_rows {
                if lower_nonfinite_rows[i] {
                    lb_flat[i] = f32::NEG_INFINITY;
                }
                if upper_nonfinite_rows[i] {
                    ub_flat[i] = f32::INFINITY;
                }
            }
        }
        Ok((lb, ub))
    }
}

/// Reconstruct the flattened input shape after Conv1d backward, preserving
/// batch dims.
pub(super) fn flattened_input_shape(
    bounds: &BatchedLinearBounds,
    conv_in_size: usize,
) -> Vec<usize> {
    if bounds.input_shape.is_empty() {
        vec![conv_in_size]
    } else {
        let mut shape = bounds.input_shape[..bounds.input_shape.len() - 1].to_vec();
        shape.push(conv_in_size);
        shape
    }
}
