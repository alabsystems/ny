// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Constraint rearrangement and slack computation for Complete Clipping.

use ndarray::{Array2, Array3, ArrayD, ArrayView2};
use ny_core::{NyError, Result};

use super::{check_clip_deadline_with, shape_err, CLIP_DEADLINE_POLL_STRIDE};

/// Rearrange constraints by normalized distance to centroid.
///
/// Heuristic that can improve convergence by processing distant constraints first.
///
/// # References
///
/// - `designs/2026-01-28-clip-and-verify-algorithms.md:318`
/// - `alpha-beta-CROWN/auto_LiRPA/concretize_func.py:_dist_rearrange`
#[cfg(test)]
pub(crate) fn rearrange_constraints(
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
    x0: &ArrayD<f32>,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    let mut never = || false;
    let x0_2d = x0
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(shape_err)?;
    let (a, b) = rearrange_constraints_with_deadline_check(a_matrix, b_vector, x0_2d, &mut never)?;
    Ok((a.into_dyn(), b.into_dyn()))
}

pub(super) fn rearrange_constraints_with_deadline_check<F>(
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
    x0_2d: ArrayView2<'_, f32>,
    past_deadline: &mut F,
) -> Result<(Array3<f32>, Array2<f32>)>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "constraint rearrangement views")?;
    let a_shape = a_matrix.shape();
    let batch = a_shape[0];
    let n_constraints = a_shape[1];
    let x_dim = a_shape[2];

    let a_3d = a_matrix
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(shape_err)?;
    let b_2d = b_vector
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(shape_err)?;
    check_clip_deadline_with(past_deadline, "constraint rearrangement allocation")?;
    let mut a_result = Array3::<f32>::zeros((batch, n_constraints, x_dim));
    let mut b_result = Array2::<f32>::zeros((batch, n_constraints));
    check_clip_deadline_with(past_deadline, "constraint rearrangement allocation")?;

    let mut cells = 0usize;
    for b in 0..batch {
        // Compute normalized distances for each constraint
        check_clip_deadline_with(past_deadline, "constraint distance allocation")?;
        let mut distances: Vec<(f32, usize)> = Vec::new();
        distances.try_reserve_exact(n_constraints).map_err(|e| {
            NyError::InvalidSpec(format!(
                "complete clip constraint distance allocation failed: {e}"
            ))
        })?;

        for k in 0..n_constraints {
            if k.is_multiple_of(64) {
                check_clip_deadline_with(past_deadline, "constraint distance row")?;
            }
            // Distance: (A·x0 + b) / ||A||_2
            let mut ax0: f32 = 0.0;
            let mut l2_norm_sq: f32 = 0.0;

            for x in 0..x_dim {
                if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "constraint distance fold")?;
                }
                ax0 += a_3d[[b, k, x]] * x0_2d[[b, x]];
                l2_norm_sq += a_3d[[b, k, x]] * a_3d[[b, k, x]];
                cells = cells.saturating_add(1);
            }

            let l2_norm = l2_norm_sq.sqrt().max(1e-10);
            let distance = (ax0 + b_2d[[b, k]]) / l2_norm;
            distances.push((distance, k));
        }

        // Sort descending (furthest constraints first, NaN last — #2995)
        // A total index tie-break lets the allocation-free checked heapsort
        // preserve the old stable input order for equal distances.
        super::deadline_heapsort_by(
            &mut distances,
            past_deadline,
            "constraint distance sort",
            |a, b| crate::cmp_utils::nan_last_descending_cmp(&a.0, &b.0).then(a.1.cmp(&b.1)),
        )?;

        // Reorder constraints
        for (new_idx, (_, old_idx)) in distances.iter().enumerate() {
            for x in 0..x_dim {
                if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "constraint reorder copy")?;
                }
                a_result[[b, new_idx, x]] = a_3d[[b, *old_idx, x]];
                cells = cells.saturating_add(1);
            }
            b_result[[b, new_idx]] = b_2d[[b, *old_idx]];
        }
    }

    check_clip_deadline_with(past_deadline, "constraint rearrangement completion")?;
    Ok((a_result, b_result))
}

pub(super) fn compute_constraint_slack_with_deadline_check<F>(
    a_3d: ndarray::ArrayView3<'_, f32>,
    x0_2d: ArrayView2<'_, f32>,
    b_2d: ArrayView2<'_, f32>,
    past_deadline: &mut F,
) -> Result<Array2<f32>>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "constraint slack views")?;
    let a_shape = a_3d.shape();
    let batch = a_shape[0];
    let n_constraints = a_shape[1];
    let x_dim = a_shape[2];

    check_clip_deadline_with(past_deadline, "constraint slack allocation")?;
    let mut d = Array2::<f32>::zeros((batch, n_constraints));

    let mut cells = 0usize;
    for b in 0..batch {
        for k in 0..n_constraints {
            if k.is_multiple_of(64) {
                check_clip_deadline_with(past_deadline, "constraint slack row")?;
            }
            let mut ax0: f32 = 0.0;
            for x in 0..x_dim {
                if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "constraint slack fold")?;
                }
                ax0 += a_3d[[b, k, x]] * x0_2d[[b, x]];
                cells = cells.saturating_add(1);
            }
            d[[b, k]] = ax0 + b_2d[[b, k]];
        }
    }

    check_clip_deadline_with(past_deadline, "constraint slack completion")?;
    Ok(d)
}
