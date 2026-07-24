// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Dual variable solver for Complete Clipping coordinate ascent.

#[cfg(test)]
use ndarray::Array1;
use ndarray::{Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use ny_core::{NyError, Result};
#[cfg(test)]
use std::time::Instant;

/// Maximum uninterrupted linear/comparison work between deadline polls.  The
/// sort comparator uses the same stride, so an expiry cannot hide inside one
/// large `sort_by` call until the next constraint/pass boundary.
const DUAL_DEADLINE_POLL_STRIDE: usize = 1024;

fn check_dual_deadline<F>(past_deadline: &mut F, phase: &'static str) -> Result<()>
where
    F: FnMut() -> bool,
{
    if past_deadline() {
        return Err(NyError::DeadlineExceeded(format!(
            "complete clipping exceeded deadline during {phase}"
        )));
    }
    Ok(())
}

/// Solve for optimal Lagrange multiplier β for a single constraint.
///
/// Uses the turning point method: the optimal β is at a turning point where
/// the gradient of the dual objective changes sign from positive to negative.
///
/// # Arguments
///
/// * `constr_a` - Constraint coefficients, shape: `(batch, x_dim)`
/// * `objective_a` - Current objective matrix, shape: `(batch, h_dim, x_dim)`
/// * `constr_d` - Pre-computed slack d = A·x0 + b, shape: `(batch,)`
/// * `epsilon` - Half-width (x_U - x_L)/2, shape: `(batch, x_dim)`
///
/// # Returns
///
/// Optimal β values, shape: `(batch, h_dim)`
///
/// # References
///
/// - `designs/2026-01-28-clip-and-verify-algorithms.md:428`
/// - `alpha-beta-CROWN/auto_LiRPA/concretize_func.py:_solve_dual_var`
#[cfg(test)]
pub(crate) fn solve_dual_variable(
    constr_a: &Array2<f32>,
    objective_a: &Array3<f32>,
    constr_d: &Array1<f32>,
    epsilon: &Array2<f32>,
    deadline: Option<Instant>,
) -> Result<Array2<f32>> {
    let mut past_deadline = || deadline.is_some_and(|d| Instant::now() >= d);
    solve_dual_variable_with_deadline_check(
        constr_a,
        objective_a,
        constr_d,
        epsilon,
        &mut past_deadline,
    )
}

/// Callback-explicit body used by deterministic deadline tests.  Production
/// supplies an `Instant` callback above; keeping the poll source injectable lets
/// tests expire *inside* preprocessing and sorting without timing sleeps.
#[cfg(test)]
pub(super) fn solve_dual_variable_with_deadline_check<F>(
    constr_a: &Array2<f32>,
    objective_a: &Array3<f32>,
    constr_d: &Array1<f32>,
    epsilon: &Array2<f32>,
    past_deadline: &mut F,
) -> Result<Array2<f32>>
where
    F: FnMut() -> bool,
{
    solve_dual_variable_views_with_deadline_check(
        constr_a.view(),
        objective_a.view(),
        constr_d.view(),
        epsilon.view(),
        past_deadline,
    )
}

pub(super) fn solve_dual_variable_views_with_deadline_check<F>(
    constr_a: ArrayView2<'_, f32>,
    objective_a: ArrayView3<'_, f32>,
    constr_d: ArrayView1<'_, f32>,
    epsilon: ArrayView2<'_, f32>,
    past_deadline: &mut F,
) -> Result<Array2<f32>>
where
    F: FnMut() -> bool,
{
    check_dual_deadline(past_deadline, "dual allocation")?;
    let obj_shape = objective_a.shape();
    let batch = obj_shape[0];
    let h_dim = obj_shape[1];
    let x_dim = obj_shape[2];

    // Turning points: q = -objective / constraint_coeff
    // Shape: (batch, h_dim, x_dim)
    let mut q = Array3::<f32>::zeros((batch, h_dim, x_dim));
    for b in 0..batch {
        check_dual_deadline(past_deadline, "dual turning-point preprocessing")?;
        for h in 0..h_dim {
            check_dual_deadline(past_deadline, "dual objective preprocessing")?;
            for x in 0..x_dim {
                if x.is_multiple_of(DUAL_DEADLINE_POLL_STRIDE) {
                    check_dual_deadline(past_deadline, "dual coefficient preprocessing")?;
                }
                let a_val = constr_a[[b, x]];
                if a_val.abs() > 1e-10 {
                    q[[b, h, x]] = -objective_a[[b, h, x]] / a_val;
                } else {
                    // For near-zero constraint coefficients, use large value
                    q[[b, h, x]] = if objective_a[[b, h, x]] >= 0.0 {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    };
                }
            }
        }
    }

    // Result: optimal β for each (batch, h_dim)
    let mut optimal_beta = Array2::<f32>::zeros((batch, h_dim));

    // Process each batch and h_dim independently
    for b in 0..batch {
        for h in 0..h_dim {
            check_dual_deadline(past_deadline, "dual objective")?;
            // Extract turning points and sort
            let mut q_vec = Vec::new();
            q_vec.try_reserve_exact(x_dim).map_err(|e| {
                NyError::InvalidSpec(format!(
                    "complete clip turning-point allocation failed: {e}"
                ))
            })?;
            for x in 0..x_dim {
                if x.is_multiple_of(DUAL_DEADLINE_POLL_STRIDE) {
                    check_dual_deadline(past_deadline, "dual turning-point collection")?;
                }
                let qv = q[[b, h, x]];
                if qv.is_finite() {
                    q_vec.push((qv, x));
                }
            }
            super::deadline_heapsort_by(
                &mut q_vec,
                past_deadline,
                "dual turning-point sort",
                |a, b| crate::cmp_utils::nan_propagating_cmp(&a.0, &b.0).then(a.1.cmp(&b.1)),
            )?;

            // Compute gradient at β = 0
            // grad = Σ |a_i| * eps_i - d
            let d_val = constr_d[b];
            let mut grad_at_zero: f32 = -d_val;
            for x in 0..x_dim {
                if x.is_multiple_of(DUAL_DEADLINE_POLL_STRIDE) {
                    check_dual_deadline(past_deadline, "dual gradient fold")?;
                }
                grad_at_zero += constr_a[[b, x]].abs() * epsilon[[b, x]];
            }

            if grad_at_zero <= 0.0 {
                // Gradient is non-positive at β = 0, optimal is β = 0
                optimal_beta[[b, h]] = 0.0;
                continue;
            }

            // Track cumulative gradient change
            let mut current_grad = grad_at_zero;
            let mut found_optimal = false;

            for (turn, (q_val, x_idx)) in q_vec.iter().enumerate() {
                if turn.is_multiple_of(DUAL_DEADLINE_POLL_STRIDE) {
                    check_dual_deadline(past_deadline, "dual turning-point scan")?;
                }
                if *q_val <= 0.0 {
                    continue; // β must be non-negative
                }

                // At turning point q_val, gradient changes by -2 * |a_i| * eps_i
                let grad_change = -2.0 * constr_a[[b, *x_idx]].abs() * epsilon[[b, *x_idx]];

                if current_grad + grad_change <= 0.0 {
                    // Sign change happens at or before this turning point
                    optimal_beta[[b, h]] = q_val.max(0.0);
                    found_optimal = true;
                    break;
                }

                current_grad += grad_change;
            }

            if !found_optimal {
                // Gradient is always positive, optimal is at infinity
                // In practice, use a large value or the last turning point
                if let Some((last_q, _)) = q_vec.last() {
                    optimal_beta[[b, h]] = last_q.max(0.0);
                }
            }
        }
    }

    check_dual_deadline(past_deadline, "dual completion")?;
    Ok(optimal_beta)
}
