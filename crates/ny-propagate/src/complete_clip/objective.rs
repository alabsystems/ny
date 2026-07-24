// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Objective computation helpers for Complete Clipping.

use ndarray::{Array2, Array3, ArrayD, ArrayView2};
use ny_core::{NyError, Result};

use super::{check_clip_deadline_with, shape_err, CLIP_DEADLINE_POLL_STRIDE};

/// Broadcast objective to `(batch, h_dim, x_dim)` while keeping the deadline
/// visible inside the copy.  Avoiding ndarray's opaque `clone`/broadcast copy is
/// load-bearing here: an admitted objective can contain hundreds of millions
/// of cells.
pub(super) fn broadcast_objective_with_deadline_check<F>(
    objective: &ArrayD<f32>,
    batch: usize,
    h_dim: usize,
    x_dim: usize,
    past_deadline: &mut F,
) -> Result<Array3<f32>>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "objective allocation")?;
    let obj_shape = objective.shape();
    let mut result = Array3::<f32>::zeros((batch, h_dim, x_dim));
    check_clip_deadline_with(past_deadline, "objective allocation")?;
    let mut copied = 0usize;

    if obj_shape.len() == 2 {
        if obj_shape[0] != h_dim || obj_shape[1] != x_dim {
            return Err(NyError::InvalidSpec(format!(
                "objective shape {:?} doesn't match h_dim={}, x_dim={}",
                obj_shape, h_dim, x_dim
            )));
        }
        let obj_2d = objective
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(shape_err)?;
        for b in 0..batch {
            for h in 0..h_dim {
                for x in 0..x_dim {
                    if copied.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                        check_clip_deadline_with(past_deadline, "objective broadcast")?;
                    }
                    result[[b, h, x]] = obj_2d[[h, x]];
                    copied = copied.saturating_add(1);
                }
            }
        }
    } else if obj_shape.len() == 3 {
        if obj_shape[0] != batch || obj_shape[1] != h_dim || obj_shape[2] != x_dim {
            return Err(NyError::InvalidSpec(format!(
                "objective shape {:?} doesn't match batch={}, h_dim={}, x_dim={}",
                obj_shape, batch, h_dim, x_dim
            )));
        }
        let obj_3d = objective
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(shape_err)?;
        for b in 0..batch {
            for h in 0..h_dim {
                for x in 0..x_dim {
                    if copied.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                        check_clip_deadline_with(past_deadline, "objective copy")?;
                    }
                    result[[b, h, x]] = obj_3d[[b, h, x]];
                    copied = copied.saturating_add(1);
                }
            }
        }
    } else {
        return Err(NyError::InvalidSpec(format!(
            "objective must be 2D or 3D, got {:?}",
            obj_shape
        )));
    }
    check_clip_deadline_with(past_deadline, "objective copy completion")?;
    Ok(result)
}

/// Compute base term: `c^T x0` for each batch and objective row.
pub(super) fn compute_base_term_with_deadline_check<F>(
    obj_mat: &Array3<f32>,
    x0: ArrayView2<'_, f32>,
    past_deadline: &mut F,
) -> Result<Array2<f32>>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "base-term allocation")?;
    let shape = obj_mat.shape();
    let (batch, h_dim, x_dim) = (shape[0], shape[1], shape[2]);
    let mut base = Array2::<f32>::zeros((batch, h_dim));
    let mut cells = 0usize;
    for b in 0..batch {
        for h in 0..h_dim {
            let mut sum = 0.0f32;
            for x in 0..x_dim {
                if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "base-term fold")?;
                }
                sum += obj_mat[[b, h, x]] * x0[[b, x]];
                cells = cells.saturating_add(1);
            }
            base[[b, h]] = sum;
        }
    }
    check_clip_deadline_with(past_deadline, "base-term completion")?;
    Ok(base)
}

/// Return `obj_mat + beta_scale * beta * constr_a` with deadline polling in
/// the dense coefficient loop. `beta` is a view so coordinate ascent does not
/// need to clone an entire witness slice before each update.
pub(super) fn update_objective_with_deadline_check<F>(
    obj_mat: &Array3<f32>,
    beta: ArrayView2<'_, f32>,
    constr_a: ArrayView2<'_, f32>,
    beta_scale: f32,
    past_deadline: &mut F,
) -> Result<Array3<f32>>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "objective-update allocation")?;
    let shape = obj_mat.shape();
    let (batch, h_dim, x_dim) = (shape[0], shape[1], shape[2]);
    let mut result = Array3::<f32>::zeros((batch, h_dim, x_dim));
    let mut cells = 0usize;
    for b in 0..batch {
        for h in 0..h_dim {
            let scaled_beta = beta_scale * beta[[b, h]];
            for x in 0..x_dim {
                if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "objective-update fold")?;
                }
                result[[b, h, x]] = obj_mat[[b, h, x]] + scaled_beta * constr_a[[b, x]];
                cells = cells.saturating_add(1);
            }
        }
    }
    check_clip_deadline_with(past_deadline, "objective-update completion")?;
    Ok(result)
}

/// Compute `|c|^T eps` with polling in the dense fold.
pub(super) fn compute_eps_term_with_deadline_check<F>(
    obj_mat: &Array3<f32>,
    eps: ArrayView2<'_, f32>,
    past_deadline: &mut F,
) -> Result<Array2<f32>>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "epsilon-term allocation")?;
    let shape = obj_mat.shape();
    let (batch, h_dim, x_dim) = (shape[0], shape[1], shape[2]);
    let mut result = Array2::<f32>::zeros((batch, h_dim));
    let mut cells = 0usize;
    for b in 0..batch {
        for h in 0..h_dim {
            let mut sum = 0.0f32;
            for x in 0..x_dim {
                if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "epsilon-term fold")?;
                }
                sum += obj_mat[[b, h, x]].abs() * eps[[b, x]];
                cells = cells.saturating_add(1);
            }
            result[[b, h]] = sum;
        }
    }
    check_clip_deadline_with(past_deadline, "epsilon-term completion")?;
    Ok(result)
}
