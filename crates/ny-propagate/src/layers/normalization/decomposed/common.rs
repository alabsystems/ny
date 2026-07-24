// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for decomposed normalization CROWN backward.
//!
//! Contains post-processing steps (non-finite fallback, bias conversion,
//! reshape) shared by decomposed normalization paths.
//!
//! Part of #2077, #318, #3447.

use ndarray::{Array2, ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use ndarray::{Ix1, Ix2, Ix3};

use crate::bounds::BatchedLinearBounds;
use crate::layers::common::compose::log_nonfinite_fallback;
use crate::LinearBounds;

pub(crate) struct DecomposedNormFinalizeMetadata<'a> {
    pub lower_nonfinite_rows: &'a [bool],
    pub upper_nonfinite_rows: &'a [bool],
    pub total_rows: usize,
    pub out_dim: usize,
    pub n: usize,
    pub batch_dims: &'a [usize],
    pub input_shape: &'a [usize],
    pub output_shape: &'a [usize],
    pub label: &'a str,
}

/// Finalize decomposed normalization bounds: apply non-finite fallback, convert
/// f64 biases to f32 with directed rounding, and reshape to BatchedLinearBounds.
///
/// Both decomposed LayerNorm and RmsNorm produce intermediate Array2 results
/// that need this identical post-processing before returning.
///
/// Part of #2077, #318, #3447.
pub(crate) fn finalize_decomposed_norm_bounds(
    mut new_a_l: Array2<f32>,
    mut new_a_u: Array2<f32>,
    mut new_b_l: Array2<f64>,
    mut new_b_u: Array2<f64>,
    metadata: DecomposedNormFinalizeMetadata<'_>,
) -> Result<BatchedLinearBounds> {
    let lower_affected = metadata
        .lower_nonfinite_rows
        .iter()
        .filter(|&&row| row)
        .count();
    let upper_affected = metadata
        .upper_nonfinite_rows
        .iter()
        .filter(|&&row| row)
        .count();
    log_nonfinite_fallback(
        metadata.label,
        lower_affected,
        upper_affected,
        metadata.total_rows,
    );

    for row_idx in 0..metadata.total_rows {
        let b = row_idx / metadata.out_dim;
        let j = row_idx % metadata.out_dim;
        if metadata.lower_nonfinite_rows[row_idx] {
            for i in 0..metadata.n {
                new_a_l[[row_idx, i]] = 0.0;
            }
            new_b_l[[b, j]] = f64::NEG_INFINITY;
        }
        if metadata.upper_nonfinite_rows[row_idx] {
            for i in 0..metadata.n {
                new_a_u[[row_idx, i]] = 0.0;
            }
            new_b_u[[b, j]] = f64::INFINITY;
        }
    }

    let (new_b_l_raw, b_l_off) = new_b_l.into_raw_vec_and_offset();
    debug_assert_eq!(
        b_l_off,
        Some(0),
        "freshly allocated b_l should have zero offset"
    );
    let new_b_l_f32: Vec<f32> = new_b_l_raw
        .into_iter()
        .map(|x| next_down_f32(x as f32))
        .collect();
    let (new_b_u_raw, b_u_off) = new_b_u.into_raw_vec_and_offset();
    debug_assert_eq!(
        b_u_off,
        Some(0),
        "freshly allocated b_u should have zero offset"
    );
    let new_b_u_f32: Vec<f32> = new_b_u_raw
        .into_iter()
        .map(|x| next_up_f32(x as f32))
        .collect();

    let (new_a_l_vec, a_l_off) = new_a_l.into_raw_vec_and_offset();
    debug_assert_eq!(
        a_l_off,
        Some(0),
        "freshly allocated a_l should have zero offset"
    );
    let (new_a_u_vec, a_u_off) = new_a_u.into_raw_vec_and_offset();
    debug_assert_eq!(
        a_u_off,
        Some(0),
        "freshly allocated a_u should have zero offset"
    );

    let out_a_shape: Vec<usize> = metadata
        .batch_dims
        .iter()
        .copied()
        .chain([metadata.out_dim, metadata.n])
        .collect();
    let out_b_shape: Vec<usize> = metadata
        .batch_dims
        .iter()
        .copied()
        .chain([metadata.out_dim])
        .collect();

    let lower_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_a_l_vec)
        .map_err(|e| NyError::InternalError(format!("reshape output lower_a: {}", e)))?;
    let upper_a = ArrayD::from_shape_vec(IxDyn(&out_a_shape), new_a_u_vec)
        .map_err(|e| NyError::InternalError(format!("reshape output upper_a: {}", e)))?;
    let lower_b = ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_b_l_f32)
        .map_err(|e| NyError::InternalError(format!("reshape output lower_b: {}", e)))?;
    let upper_b = ArrayD::from_shape_vec(IxDyn(&out_b_shape), new_b_u_f32)
        .map_err(|e| NyError::InternalError(format!("reshape output upper_b: {}", e)))?;

    BatchedLinearBounds::new(
        lower_a,
        lower_b,
        upper_a,
        upper_b,
        metadata.input_shape.to_vec(),
        metadata.output_shape.to_vec(),
    )
}

/// Validate decomposed normalization bounds against fused LayerNorm IBP fallback.
///
/// The decomposed path should never be looser than the fused LayerNorm IBP.
/// When a row escapes that envelope, collapse only that row to the fused
/// IBP interval instead of discarding the entire decomposed bound.
///
/// Returns the number of rows that fell back to fused IBP.
/// Part of #2077, #318.
pub(crate) fn validate_norm_against_fused_ibp(
    result: &mut BatchedLinearBounds,
    a_output: &BatchedLinearBounds,
    fused_ibp_output: &BoundedTensor,
    x_ibp: &BoundedTensor,
    total_rows: usize,
    n: usize,
) -> Result<usize> {
    let fallback_interval = a_output.concretize_sound(fused_ibp_output)?;
    let candidate_interval = result.concretize_sound(x_ibp)?;
    let fallback_lower = fallback_interval
        .lower()
        .view()
        .into_shape_with_order(total_rows)
        .map_err(|e| {
            NyError::InternalError(format!("reshape fused ibp lower for validation: {e}"))
        })?;
    let fallback_upper = fallback_interval
        .upper()
        .view()
        .into_shape_with_order(total_rows)
        .map_err(|e| {
            NyError::InternalError(format!("reshape fused ibp upper for validation: {e}"))
        })?;
    let candidate_lower = candidate_interval
        .lower()
        .view()
        .into_shape_with_order(total_rows)
        .map_err(|e| {
            NyError::InternalError(format!("reshape candidate lower for validation: {e}"))
        })?;
    let candidate_upper = candidate_interval
        .upper()
        .view()
        .into_shape_with_order(total_rows)
        .map_err(|e| {
            NyError::InternalError(format!("reshape candidate upper for validation: {e}"))
        })?;

    let mut fallback_rows = 0usize;
    {
        let mut lower_a_rows = result
            .lower_a
            .view_mut()
            .into_shape_with_order((total_rows, n))
            .map_err(|e| {
                NyError::InternalError(format!("reshape lower_a rows for validation: {e}"))
            })?;
        let mut upper_a_rows = result
            .upper_a
            .view_mut()
            .into_shape_with_order((total_rows, n))
            .map_err(|e| {
                NyError::InternalError(format!("reshape upper_a rows for validation: {e}"))
            })?;
        let mut lower_b_rows = result
            .lower_b
            .view_mut()
            .into_shape_with_order(total_rows)
            .map_err(|e| {
                NyError::InternalError(format!("reshape lower_b rows for validation: {e}"))
            })?;
        let mut upper_b_rows = result
            .upper_b
            .view_mut()
            .into_shape_with_order(total_rows)
            .map_err(|e| {
                NyError::InternalError(format!("reshape upper_b rows for validation: {e}"))
            })?;

        for row_idx in 0..total_rows {
            if candidate_lower[row_idx] < fallback_lower[row_idx]
                || candidate_upper[row_idx] > fallback_upper[row_idx]
            {
                lower_a_rows.row_mut(row_idx).fill(0.0);
                upper_a_rows.row_mut(row_idx).fill(0.0);
                lower_b_rows[row_idx] = fallback_lower[row_idx];
                upper_b_rows[row_idx] = fallback_upper[row_idx];
                fallback_rows += 1;
            }
        }
    }

    Ok(fallback_rows)
}

/// Convert scalar `LinearBounds` to `BatchedLinearBounds` for use with
/// decomposed normalization backward (which operates on batched bounds).
///
/// Shared bridge helper used by LayerNorm and RmsNorm scalar CROWN paths
/// to route IbpValidated mode through the decomposed primitive-chain backward.
///
/// Part of #2077, #3821.
pub(crate) fn scalar_bounds_to_batched(bounds: &LinearBounds) -> Result<BatchedLinearBounds> {
    BatchedLinearBounds::new(
        bounds.lower_a().clone().into_dyn(),
        bounds.lower_b().clone().into_dyn(),
        bounds.upper_a().clone().into_dyn(),
        bounds.upper_b().clone().into_dyn(),
        vec![bounds.num_inputs()],
        vec![bounds.num_outputs()],
    )
}

/// Convert scalar `LinearBounds` into a batched representation by splitting the
/// flattened input axis into `[batch_size, norm_size]`.
///
/// Each batch position keeps the full output-row set while only the input
/// columns are partitioned per normalization slice. The incoming scalar bias is
/// intentionally dropped here and restored exactly once by
/// [`batched_bounds_to_scalar_multi_dim`].
///
/// Part of #4148.
pub(crate) fn scalar_bounds_to_batched_multi_dim(
    bounds: &LinearBounds,
    batch_size: usize,
    norm_size: usize,
) -> Result<BatchedLinearBounds> {
    let out_dim = bounds.num_outputs();
    let expected_inputs = batch_size.checked_mul(norm_size).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "multi-dim scalar->batched reshape overflow: batch_size={batch_size}, \
             norm_size={norm_size}"
        ))
    })?;
    if bounds.num_inputs() != expected_inputs {
        return Err(NyError::shape_mismatch(
            vec![expected_inputs],
            vec![bounds.num_inputs()],
        ));
    }

    // permuted_axes produces a view with non-standard strides. to_owned() on a
    // contiguous permuted view preserves those strides, which causes downstream
    // into_shape_with_order() to fail with IncompatibleLayout. Force standard
    // (C-contiguous) layout so decomposed_norm_crown_backward can reshape freely.
    let lower_a = bounds
        .lower_a()
        .view()
        .into_shape_with_order((out_dim, batch_size, norm_size))
        .map_err(|e| {
            NyError::InternalError(format!(
                "reshape scalar lower_a to ({out_dim}, {batch_size}, {norm_size}): {e}"
            ))
        })?
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .into_owned()
        .into_dyn();
    let upper_a = bounds
        .upper_a()
        .view()
        .into_shape_with_order((out_dim, batch_size, norm_size))
        .map_err(|e| {
            NyError::InternalError(format!(
                "reshape scalar upper_a to ({out_dim}, {batch_size}, {norm_size}): {e}"
            ))
        })?
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .into_owned()
        .into_dyn();
    let zero_bias = Array2::zeros((batch_size, out_dim)).into_dyn();

    BatchedLinearBounds::new(
        lower_a,
        zero_bias.clone(),
        upper_a,
        zero_bias,
        vec![batch_size, norm_size],
        vec![batch_size, out_dim],
    )
}

/// Convert `BatchedLinearBounds` back to scalar `LinearBounds` after decomposed
/// normalization backward.
///
/// Shared bridge helper used by LayerNorm and RmsNorm scalar CROWN paths.
///
/// Part of #2077, #3821.
pub(crate) fn batched_bounds_to_scalar(bounds: &BatchedLinearBounds) -> Result<LinearBounds> {
    let lower_a = bounds
        .lower_a()
        .clone()
        .into_dimensionality::<Ix2>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 2D lower_a from decomposed norm, got {:?}",
                bounds.lower_a().shape()
            ))
        })?;
    let lower_b = bounds
        .lower_b()
        .clone()
        .into_dimensionality::<Ix1>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 1D lower_b from decomposed norm, got {:?}",
                bounds.lower_b().shape()
            ))
        })?;
    let upper_a = bounds
        .upper_a()
        .clone()
        .into_dimensionality::<Ix2>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 2D upper_a from decomposed norm, got {:?}",
                bounds.upper_a().shape()
            ))
        })?;
    let upper_b = bounds
        .upper_b()
        .clone()
        .into_dimensionality::<Ix1>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 1D upper_b from decomposed norm, got {:?}",
                bounds.upper_b().shape()
            ))
        })?;

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}

/// Merge `[batch_size, out_dim, norm_size]` batched bounds back into scalar
/// `LinearBounds`, restoring the original incoming scalar bias exactly once.
///
/// Part of #4148.
pub(crate) fn batched_bounds_to_scalar_multi_dim(
    bounds: &BatchedLinearBounds,
    original_bias_lower: &ndarray::Array1<f32>,
    original_bias_upper: &ndarray::Array1<f32>,
) -> Result<LinearBounds> {
    let lower_a = bounds
        .lower_a()
        .clone()
        .into_dimensionality::<Ix3>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 3D lower_a from multi-dim norm reshape, got {:?}",
                bounds.lower_a().shape()
            ))
        })?;
    let upper_a = bounds
        .upper_a()
        .clone()
        .into_dimensionality::<Ix3>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 3D upper_a from multi-dim norm reshape, got {:?}",
                bounds.upper_a().shape()
            ))
        })?;
    let lower_b = bounds
        .lower_b()
        .clone()
        .into_dimensionality::<Ix2>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 2D lower_b from multi-dim norm reshape, got {:?}",
                bounds.lower_b().shape()
            ))
        })?;
    let upper_b = bounds
        .upper_b()
        .clone()
        .into_dimensionality::<Ix2>()
        .map_err(|_| {
            NyError::InternalError(format!(
                "expected 2D upper_b from multi-dim norm reshape, got {:?}",
                bounds.upper_b().shape()
            ))
        })?;

    let batch_size = lower_a.shape()[0];
    let out_dim = lower_a.shape()[1];
    let norm_size = lower_a.shape()[2];
    let flat_input = batch_size.checked_mul(norm_size).ok_or_else(|| {
        NyError::InvalidSpec(format!(
            "multi-dim batched->scalar reshape overflow: batch_size={batch_size}, \
             norm_size={norm_size}"
        ))
    })?;

    // Force standard layout after permutation (same reasoning as
    // scalar_bounds_to_batched_multi_dim above).
    let lower_a = lower_a
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .into_shape_with_order((out_dim, flat_input))
        .map_err(|e| {
            NyError::InternalError(format!(
                "reshape multi-dim lower_a back to ({out_dim}, {flat_input}): {e}",
            ))
        })?
        .into_owned();
    let upper_a = upper_a
        .permuted_axes([1, 0, 2])
        .as_standard_layout()
        .into_shape_with_order((out_dim, flat_input))
        .map_err(|e| {
            NyError::InternalError(format!(
                "reshape multi-dim upper_a back to ({out_dim}, {flat_input}): {e}",
            ))
        })?
        .into_owned();

    let lower_b = lower_b.sum_axis(Axis(0)) + original_bias_lower;
    let upper_b = upper_b.sum_axis(Axis(0)) + original_bias_upper;

    LinearBounds::new_or_conservative(lower_a, lower_b, upper_a, upper_b)
}
