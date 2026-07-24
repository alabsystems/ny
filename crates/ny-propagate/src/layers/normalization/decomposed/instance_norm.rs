// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decomposed InstanceNorm1d CROWN backward propagation.
//!
//! Flat InstanceNorm is the `num_groups == num_channels` special case of the
//! grouped centered-normalization adapter, while the channel-batched helper
//! below remains the graph-specific `[...outer, C, T, T]` path.

use ndarray::{s, Array1, Array2};
use ny_core::{checked_dim_product, checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::BatchedLinearBounds;
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::InstanceNorm1dLayer;

use super::{
    decomposed_grouped_centered_crown_backward, decomposed_norm_crown_backward,
    finalize_decomposed_norm_bounds, validate_norm_against_fused_ibp, DecomposedNormBackwardResult,
    DecomposedNormFinalizeMetadata, RowValidationCounts,
};

/// Decomposed InstanceNorm1d CROWN backward.
///
/// Delegates to the grouped centered-normalization adapter with one channel per
/// group (`num_groups == num_channels`), preserving the single shared flat
/// centered-normalization implementation from #1744.
pub(crate) fn decomposed_instance_norm_crown_backward(
    a_output: &BatchedLinearBounds,
    ny: &Array1<f32>,
    beta: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
    forward_mode: bool,
    num_channels: usize,
) -> Result<DecomposedNormBackwardResult> {
    decomposed_grouped_centered_crown_backward(
        a_output,
        ny,
        beta,
        eps,
        x_ibp,
        forward_mode,
        num_channels,
        num_channels,
    )
}

/// Decomposed InstanceNorm1d CROWN backward for channel-batched layouts.
///
/// Block-wise graph CROWN carries InstanceNorm bounds as `[...outer, C, T, T]`,
/// where channels already live in the batch dimensions and each coefficient
/// row covers only one channel's time axis. This adapter applies the shared
/// LayerNorm decomposed helper independently to each flattened batch position
/// and reassembles the original batched layout. Part of #3830.
pub(crate) fn decomposed_instance_norm_crown_backward_channel_batched(
    a_output: &BatchedLinearBounds,
    ny: &Array1<f32>,
    beta: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
    forward_mode: bool,
    num_channels: usize,
) -> Result<DecomposedNormBackwardResult> {
    let a_shape = a_output.lower_a().shape();
    let ndim = a_shape.len();
    if ndim < 2 {
        return Err(NyError::InvalidSpec(
            "decomposed_instance_norm_crown_backward_channel_batched: A must have at least 2 dimensions".into(),
        ));
    }
    if num_channels == 0 {
        return Err(NyError::InvalidSpec(
            "decomposed_instance_norm_crown_backward_channel_batched: num_channels must be > 0"
                .into(),
        ));
    }
    if ny.len() != num_channels || beta.len() != num_channels {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_channels],
            got: vec![ny.len()],
        });
    }

    let time_len = a_shape[ndim - 1];
    let out_dim = a_shape[ndim - 2];
    let batch_dims = &a_shape[..ndim - 2];
    let total_batch = checked_shape_product(batch_dims)
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "decomposed_instance_norm(channel-batched): batch dimensions overflow".into(),
            )
        })?
        .max(1);
    if total_batch % num_channels != 0 {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_channels],
            got: batch_dims.to_vec(),
        });
    }

    let x_expanded = x_ibp.reshape(a_output.input_shape())?;
    let lower_a = a_output
        .lower_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, time_len))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_instance_norm(channel-batched): reshape lower_a: {e}"
            ))
        })?;
    let upper_a = a_output
        .upper_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, time_len))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_instance_norm(channel-batched): reshape upper_a: {e}"
            ))
        })?;
    let lower_b = a_output
        .lower_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_instance_norm(channel-batched): reshape lower_b: {e}"
            ))
        })?;
    let upper_b = a_output
        .upper_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_instance_norm(channel-batched): reshape upper_b: {e}"
            ))
        })?;
    let x_lower = x_expanded
        .lower()
        .view()
        .into_shape_with_order((total_batch, time_len))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_instance_norm(channel-batched): reshape x_lower: {e}"
            ))
        })?;
    let x_upper = x_expanded
        .upper()
        .view()
        .into_shape_with_order((total_batch, time_len))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_instance_norm(channel-batched): reshape x_upper: {e}"
            ))
        })?;

    let total_rows = checked_dim_product(
        &[total_batch, out_dim],
        "decomposed_instance_norm_crown_backward_channel_batched total rows",
    )?;
    let mut new_a_l = ndarray::Array3::<f32>::zeros((total_batch, out_dim, time_len));
    let mut new_a_u = ndarray::Array3::<f32>::zeros((total_batch, out_dim, time_len));
    let mut new_b_l = Array2::<f64>::zeros((total_batch, out_dim));
    let mut new_b_u = Array2::<f64>::zeros((total_batch, out_dim));

    let zero_bias = Array2::<f32>::zeros((1, out_dim)).into_dyn();
    for batch_idx in 0..total_batch {
        let channel = batch_idx % num_channels;
        let row_bounds = BatchedLinearBounds::new(
            lower_a
                .slice(s![batch_idx..batch_idx + 1, .., ..])
                .to_owned()
                .into_dyn(),
            zero_bias.clone(),
            upper_a
                .slice(s![batch_idx..batch_idx + 1, .., ..])
                .to_owned()
                .into_dyn(),
            zero_bias.clone(),
            vec![1, time_len],
            vec![1, out_dim],
        )?;
        let row_input = BoundedTensor::new(
            x_lower
                .slice(s![batch_idx..batch_idx + 1, ..])
                .to_owned()
                .into_dyn(),
            x_upper
                .slice(s![batch_idx..batch_idx + 1, ..])
                .to_owned()
                .into_dyn(),
        )?;
        let row_gamma = Array1::from_elem(time_len, ny[channel]);
        let row_beta = Array1::from_elem(time_len, beta[channel]);
        let row_result = decomposed_norm_crown_backward(
            &row_bounds,
            &row_gamma,
            &row_beta,
            eps,
            &row_input,
            forward_mode,
        )?;

        let row_lower_a = row_result
            .bounds
            .lower_a()
            .view()
            .into_shape_with_order((1, out_dim, time_len))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "decomposed_instance_norm(channel-batched): reshape row lower_a: {e}"
                ))
            })?;
        let row_upper_a = row_result
            .bounds
            .upper_a()
            .view()
            .into_shape_with_order((1, out_dim, time_len))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "decomposed_instance_norm(channel-batched): reshape row upper_a: {e}"
                ))
            })?;
        let row_lower_b = row_result
            .bounds
            .lower_b()
            .view()
            .into_shape_with_order((1, out_dim))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "decomposed_instance_norm(channel-batched): reshape row lower_b: {e}"
                ))
            })?;
        let row_upper_b = row_result
            .bounds
            .upper_b()
            .view()
            .into_shape_with_order((1, out_dim))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "decomposed_instance_norm(channel-batched): reshape row upper_b: {e}"
                ))
            })?;

        for j in 0..out_dim {
            new_b_l[[batch_idx, j]] = lower_b[[batch_idx, j]] as f64 + row_lower_b[[0, j]] as f64;
            new_b_u[[batch_idx, j]] = upper_b[[batch_idx, j]] as f64 + row_upper_b[[0, j]] as f64;
            for t in 0..time_len {
                new_a_l[[batch_idx, j, t]] = row_lower_a[[0, j, t]];
                new_a_u[[batch_idx, j, t]] = row_upper_a[[0, j, t]];
            }
        }
    }

    let (new_a_l_vec, _) = new_a_l.into_raw_vec_and_offset();
    let (new_a_u_vec, _) = new_a_u.into_raw_vec_and_offset();
    let new_a_l = Array2::from_shape_vec((total_rows, time_len), new_a_l_vec).map_err(|e| {
        NyError::InternalError(format!(
            "decomposed_instance_norm(channel-batched): reshape assembled lower_a: {e}"
        ))
    })?;
    let new_a_u = Array2::from_shape_vec((total_rows, time_len), new_a_u_vec).map_err(|e| {
        NyError::InternalError(format!(
            "decomposed_instance_norm(channel-batched): reshape assembled upper_a: {e}"
        ))
    })?;

    let lower_nonfinite_rows = vec![false; total_rows];
    let upper_nonfinite_rows = vec![false; total_rows];
    let mut result = finalize_decomposed_norm_bounds(
        new_a_l,
        new_a_u,
        new_b_l,
        new_b_u,
        DecomposedNormFinalizeMetadata {
            lower_nonfinite_rows: &lower_nonfinite_rows,
            upper_nonfinite_rows: &upper_nonfinite_rows,
            total_rows,
            out_dim,
            n: time_len,
            batch_dims,
            input_shape: a_output.input_shape(),
            output_shape: a_output.output_shape(),
            label: "Decomposed InstanceNorm (channel-batched)",
        },
    )?;

    let fused_ibp = InstanceNorm1dLayer::new(ny.to_owned(), beta.to_owned(), eps)?
        .with_forward_mode(forward_mode)
        .propagate_ibp(&x_expanded)?;
    let fallback_rows = validate_norm_against_fused_ibp(
        &mut result,
        a_output,
        &fused_ibp,
        &x_expanded,
        total_rows,
        time_len,
    )?;
    if fallback_rows > 0 {
        debug!(
            "Decomposed InstanceNorm (channel-batched): collapsed {fallback_rows}/{total_rows} rows to fused InstanceNorm IBP"
        );
    }

    Ok(DecomposedNormBackwardResult {
        bounds: result,
        validation: RowValidationCounts {
            fallback_rows,
            total_rows,
        },
    })
}
