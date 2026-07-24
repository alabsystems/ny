// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Decomposed grouped centered-normalization CROWN backward propagation.
//!
//! GroupNorm and flat InstanceNorm both share the alpha-beta-CROWN centered
//! primitive chain `mean -> d -> d^2 -> var -> sqrt -> reciprocal -> affine`,
//! differing only in how flattened `[C*T]` inputs are partitioned into
//! independent blocks. See `auto_LiRPA/operators/normalization.py:309-319`.

use ndarray::{s, Array1, Array2, Array4};
use ny_core::{checked_dim_product, checked_shape_product, NyError, Result};
use ny_tensor::BoundedTensor;
use tracing::debug;

use crate::bounds::BatchedLinearBounds;
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::GroupNormLayer;

use super::{
    decomposed_norm_crown_backward, finalize_decomposed_norm_bounds,
    validate_norm_against_fused_ibp, DecomposedNormBackwardResult, DecomposedNormFinalizeMetadata,
    RowValidationCounts,
};

/// Run decomposed centered-normalization CROWN backward on flattened
/// `[C*T]` layouts partitioned into contiguous group blocks.
#[expect(
    clippy::too_many_arguments,
    reason = "the #3914 design keeps the grouped adapter signature parallel to the shared decomposed norm inputs plus explicit channel/group partitioning"
)]
pub(crate) fn decomposed_grouped_centered_crown_backward(
    a_output: &BatchedLinearBounds,
    ny: &Array1<f32>,
    beta: &Array1<f32>,
    eps: f32,
    x_ibp: &BoundedTensor,
    forward_mode: bool,
    num_channels: usize,
    num_groups: usize,
) -> Result<DecomposedNormBackwardResult> {
    let a_shape = a_output.lower_a().shape();
    let ndim = a_shape.len();
    if ndim < 2 {
        return Err(NyError::InvalidSpec(
            "decomposed_grouped_centered_crown_backward: A must have at least 2 dimensions".into(),
        ));
    }
    if num_channels == 0 {
        return Err(NyError::InvalidSpec(
            "decomposed_grouped_centered_crown_backward: num_channels must be > 0".into(),
        ));
    }
    if num_groups == 0 {
        return Err(NyError::InvalidSpec(
            "decomposed_grouped_centered_crown_backward: num_groups must be > 0".into(),
        ));
    }
    if !num_channels.is_multiple_of(num_groups) {
        return Err(NyError::InvalidSpec(format!(
            "decomposed_grouped_centered_crown_backward: num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        )));
    }
    if ny.len() != num_channels || beta.len() != num_channels {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_channels],
            got: vec![ny.len()],
        });
    }

    let total_dim = a_shape[ndim - 1];
    let out_dim = a_shape[ndim - 2];
    let batch_dims = &a_shape[..ndim - 2];
    let total_batch = checked_shape_product(batch_dims)
        .ok_or_else(|| {
            NyError::InvalidSpec(
                "decomposed_grouped_centered_crown_backward: batch dimensions overflow".into(),
            )
        })?
        .max(1);
    if total_dim == 0 || !total_dim.is_multiple_of(num_channels) {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_channels],
            got: vec![total_dim],
        });
    }

    let time_len = total_dim / num_channels;
    let channels_per_group = num_channels / num_groups;
    let group_size = channels_per_group * time_len;

    let expanded_input_shape: Vec<usize> = batch_dims
        .iter()
        .copied()
        .chain([num_channels, time_len])
        .collect();
    let x_expanded = x_ibp.reshape(&expanded_input_shape)?;
    let x_flat = x_ibp.reshape(a_output.input_shape())?;

    let lower_a = a_output
        .lower_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, num_channels, time_len))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_grouped_centered_crown_backward: reshape lower_a: {e}"
            ))
        })?;
    let upper_a = a_output
        .upper_a()
        .view()
        .into_shape_with_order((total_batch, out_dim, num_channels, time_len))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_grouped_centered_crown_backward: reshape upper_a: {e}"
            ))
        })?;
    let lower_b = a_output
        .lower_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_grouped_centered_crown_backward: reshape lower_b: {e}"
            ))
        })?;
    let upper_b = a_output
        .upper_b()
        .view()
        .into_shape_with_order((total_batch, out_dim))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_grouped_centered_crown_backward: reshape upper_b: {e}"
            ))
        })?;
    let x_lower = x_expanded
        .lower()
        .view()
        .into_shape_with_order((total_batch, num_channels, time_len))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_grouped_centered_crown_backward: reshape x_lower: {e}"
            ))
        })?;
    let x_upper = x_expanded
        .upper()
        .view()
        .into_shape_with_order((total_batch, num_channels, time_len))
        .map_err(|e| {
            NyError::InvalidSpec(format!(
                "decomposed_grouped_centered_crown_backward: reshape x_upper: {e}"
            ))
        })?;

    let total_rows = checked_dim_product(
        &[total_batch, out_dim],
        "decomposed_grouped_centered_crown_backward total rows",
    )?;
    let mut new_a_l = Array4::<f32>::zeros((total_batch, out_dim, num_channels, time_len));
    let mut new_a_u = Array4::<f32>::zeros((total_batch, out_dim, num_channels, time_len));
    let mut new_b_l = Array2::<f64>::zeros((total_batch, out_dim));
    let mut new_b_u = Array2::<f64>::zeros((total_batch, out_dim));
    for b in 0..total_batch {
        for j in 0..out_dim {
            new_b_l[[b, j]] = lower_b[[b, j]] as f64;
            new_b_u[[b, j]] = upper_b[[b, j]] as f64;
        }
    }

    let zero_bias = Array2::<f32>::zeros((total_batch, out_dim)).into_dyn();
    let flat_input_shape = vec![total_batch, group_size];
    let flat_output_shape = vec![total_batch, out_dim];

    for group in 0..num_groups {
        let channel_start = group * channels_per_group;
        let channel_end = channel_start + channels_per_group;
        let group_bounds = BatchedLinearBounds::new(
            lower_a
                .slice(s![.., .., channel_start..channel_end, ..])
                .to_owned()
                .into_shape_with_order((total_batch, out_dim, group_size))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "decomposed_grouped_centered_crown_backward: reshape group lower_a: {e}"
                    ))
                })?
                .into_dyn(),
            zero_bias.clone(),
            upper_a
                .slice(s![.., .., channel_start..channel_end, ..])
                .to_owned()
                .into_shape_with_order((total_batch, out_dim, group_size))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "decomposed_grouped_centered_crown_backward: reshape group upper_a: {e}"
                    ))
                })?
                .into_dyn(),
            zero_bias.clone(),
            flat_input_shape.clone(),
            flat_output_shape.clone(),
        )?;
        let group_input = BoundedTensor::new(
            x_lower
                .slice(s![.., channel_start..channel_end, ..])
                .to_owned()
                .into_shape_with_order((total_batch, group_size))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "decomposed_grouped_centered_crown_backward: reshape group x_lower: {e}"
                    ))
                })?
                .into_dyn(),
            x_upper
                .slice(s![.., channel_start..channel_end, ..])
                .to_owned()
                .into_shape_with_order((total_batch, group_size))
                .map_err(|e| {
                    NyError::InternalError(format!(
                        "decomposed_grouped_centered_crown_backward: reshape group x_upper: {e}"
                    ))
                })?
                .into_dyn(),
        )?;
        let group_gamma = Array1::from_iter(
            (channel_start..channel_end)
                .flat_map(|channel| std::iter::repeat_n(ny[channel], time_len)),
        );
        let group_beta = Array1::from_iter(
            (channel_start..channel_end)
                .flat_map(|channel| std::iter::repeat_n(beta[channel], time_len)),
        );
        let group_result = decomposed_norm_crown_backward(
            &group_bounds,
            &group_gamma,
            &group_beta,
            eps,
            &group_input,
            forward_mode,
        )?;
        let group_lower_a = group_result
            .bounds
            .lower_a()
            .view()
            .into_shape_with_order((total_batch, out_dim, channels_per_group, time_len))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "decomposed_grouped_centered_crown_backward: reshape result lower_a: {e}"
                ))
            })?;
        let group_upper_a = group_result
            .bounds
            .upper_a()
            .view()
            .into_shape_with_order((total_batch, out_dim, channels_per_group, time_len))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "decomposed_grouped_centered_crown_backward: reshape result upper_a: {e}"
                ))
            })?;
        let group_lower_b = group_result
            .bounds
            .lower_b()
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "decomposed_grouped_centered_crown_backward: reshape result lower_b: {e}"
                ))
            })?;
        let group_upper_b = group_result
            .bounds
            .upper_b()
            .view()
            .into_shape_with_order((total_batch, out_dim))
            .map_err(|e| {
                NyError::InternalError(format!(
                    "decomposed_grouped_centered_crown_backward: reshape result upper_b: {e}"
                ))
            })?;

        new_a_l
            .slice_mut(s![.., .., channel_start..channel_end, ..])
            .assign(&group_lower_a);
        new_a_u
            .slice_mut(s![.., .., channel_start..channel_end, ..])
            .assign(&group_upper_a);
        for b in 0..total_batch {
            for j in 0..out_dim {
                new_b_l[[b, j]] += group_lower_b[[b, j]] as f64;
                new_b_u[[b, j]] += group_upper_b[[b, j]] as f64;
            }
        }
    }

    let (new_a_l_vec, _) = new_a_l.into_raw_vec_and_offset();
    let (new_a_u_vec, _) = new_a_u.into_raw_vec_and_offset();
    let new_a_l = Array2::from_shape_vec((total_rows, total_dim), new_a_l_vec).map_err(|e| {
        NyError::InternalError(format!(
            "decomposed_grouped_centered_crown_backward: reshape assembled lower_a: {e}"
        ))
    })?;
    let new_a_u = Array2::from_shape_vec((total_rows, total_dim), new_a_u_vec).map_err(|e| {
        NyError::InternalError(format!(
            "decomposed_grouped_centered_crown_backward: reshape assembled upper_a: {e}"
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
            n: total_dim,
            batch_dims,
            input_shape: a_output.input_shape(),
            output_shape: a_output.output_shape(),
            label: "Decomposed grouped-centered norm",
        },
    )?;

    let fused_ibp = GroupNormLayer::new(ny.to_owned(), beta.to_owned(), num_groups, eps)?
        .with_forward_mode(forward_mode)
        .propagate_ibp(&x_expanded)?
        .reshape(a_output.input_shape())?;
    let fallback_rows = validate_norm_against_fused_ibp(
        &mut result,
        a_output,
        &fused_ibp,
        &x_flat,
        total_rows,
        total_dim,
    )?;
    if fallback_rows > 0 {
        debug!(
            "Decomposed grouped-centered norm: collapsed {fallback_rows}/{total_rows} rows to fused GroupNorm IBP"
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
