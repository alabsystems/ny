// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shape validation and constraint feasibility checks for Complete Clipping.

use ndarray::ArrayD;
use ny_core::{NyError, Result};

use super::{check_clip_deadline_with, shape_err, CLIP_DEADLINE_POLL_STRIDE};

/// Check if any single constraint is infeasible over the input box.
///
/// For each constraint A_k x + b_k <= 0, compute the minimum value over
/// x in [x_l, x_u]. If the minimum is positive, the constraint is infeasible.
/// This is a conservative check (it does not detect infeasible combinations).
#[cfg(test)]
pub(crate) fn ensure_constraints_feasible(
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
) -> Result<()> {
    let mut never = || false;
    ensure_constraints_feasible_with_deadline_check(a_matrix, b_vector, x_l, x_u, &mut never)
}

pub(super) fn ensure_constraints_feasible_with_deadline_check<F>(
    a_matrix: &ArrayD<f32>,
    b_vector: &ArrayD<f32>,
    x_l: &ArrayD<f32>,
    x_u: &ArrayD<f32>,
    past_deadline: &mut F,
) -> Result<()>
where
    F: FnMut() -> bool,
{
    check_clip_deadline_with(past_deadline, "constraint feasibility views")?;
    let a_3d = a_matrix
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(shape_err)?;
    let b_2d = b_vector
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(shape_err)?;
    let x_l_2d = x_l
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(shape_err)?;
    let x_u_2d = x_u
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .map_err(shape_err)?;

    let batch = a_3d.shape()[0];
    let n_constraints = a_3d.shape()[1];
    let x_dim = a_3d.shape()[2];

    let mut cells = 0usize;
    for b in 0..batch {
        // The clipping baseline uses xhat=(x_U+x_L)/2 and eps=(x_U-x_L)/2, so
        // the raw box must be ordered before feasibility/concretization.
        // Reference: alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) complete_verifier/input_split/clip.py.
        for x in 0..x_dim {
            if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                check_clip_deadline_with(past_deadline, "input-box feasibility")?;
            }
            if x_l_2d[[b, x]] > x_u_2d[[b, x]] {
                return Err(NyError::InvalidSpec(format!(
                    "complete_clip: x_l > x_u at batch={} dim={}",
                    b, x
                )));
            }
            cells = cells.saturating_add(1);
        }
        for k in 0..n_constraints {
            // Poll by row as well as by coefficient. This is load-bearing for
            // X=1, N>>1, where an x-only stride never fires again.
            if k.is_multiple_of(64) {
                check_clip_deadline_with(past_deadline, "constraint feasibility row")?;
            }
            // f64 corner evaluation with a certified summation round-off
            // bound (Higham gamma_n with headroom), matching
            // `filter::sort_out_constraints`: the InfeasibleDomain verdict
            // must hold for the EXACT minimum, so the round-off has to scale
            // with the accumulated magnitude, not sit at a fixed constant.
            let mut min_val = b_2d[[b, k]] as f64;
            let mut abs_sum = min_val.abs();
            for x in 0..x_dim {
                if cells.is_multiple_of(CLIP_DEADLINE_POLL_STRIDE) {
                    check_clip_deadline_with(past_deadline, "constraint feasibility fold")?;
                }
                let a_val = a_3d[[b, k, x]] as f64;
                let x_min = if a_val >= 0.0 {
                    x_l_2d[[b, x]] as f64
                } else {
                    x_u_2d[[b, x]] as f64
                };
                let term = a_val * x_min;
                min_val += term;
                abs_sum += term.abs();
                cells = cells.saturating_add(1);
            }
            let err = (x_dim as f64 + 1.0) * f64::EPSILON * abs_sum;
            // `min_val - err` is a certified lower bound on the exact minimum;
            // NaN/∞-poisoned rows compare false (feasible, fail closed).
            if min_val - err > 0.0 {
                return Err(NyError::InfeasibleDomain(format!(
                    "infeasible constraint: batch={} constraint={} min_slack={}",
                    b, k, min_val
                )));
            }
        }
    }

    check_clip_deadline_with(past_deadline, "constraint feasibility completion")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn test_boundary_feasible_large_magnitude_is_feasible() {
        // Corner terms of magnitude 2^24 that cancel exactly to a box-minimum
        // of 0 (feasible). f32 round-to-nearest accumulation drifts to +0.75
        // on these terms, so the InfeasibleDomain verdict must come from a
        // certified lower bound on the exact minimum.
        let a = arr3(&[[[16777216.0f32, 1.5, -16777216.0, -1.25]]]).into_dyn();
        let b = arr2(&[[-0.25f32]]).into_dyn();
        let x_l = arr2(&[[1.0f32, 1.0, 1.0, 1.0]]).into_dyn();
        let x_u = arr2(&[[1.0f32, 1.0, 1.0, 1.0]]).into_dyn();

        assert!(ensure_constraints_feasible(&a, &b, &x_l, &x_u).is_ok());
    }

    #[test]
    fn test_truly_infeasible_constraint_is_rejected() {
        // min(x1 + x2 + 1) over [0,1]^2 is 1 > 0: no point satisfies the
        // constraint, so the domain is infeasible.
        let a = arr3(&[[[1.0f32, 1.0]]]).into_dyn();
        let b = arr2(&[[1.0f32]]).into_dyn();
        let x_l = arr2(&[[0.0f32, 0.0]]).into_dyn();
        let x_u = arr2(&[[1.0f32, 1.0]]).into_dyn();

        let err = ensure_constraints_feasible(&a, &b, &x_l, &x_u)
            .expect_err("positive box-minimum must be infeasible");
        assert!(matches!(err, NyError::InfeasibleDomain(_)));
    }
}
