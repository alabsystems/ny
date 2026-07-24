// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Distribution-Aware CROWN (DA-CROWN) analytical tools.
//!
//! Provides path-specific variance tightening and distributional objective
//! functions for guiding alpha optimization toward probabilistic tightness.
//!
//! Key insight: standard `propagate_distribution` uses `max(var_L, var_U)` for
//! both probabilistic bounds. This is over-conservative. Since CROWN gives:
//!   Y_L = A_L @ X + b_L <= f(X) <= A_U @ X + b_U = Y_U
//! the lower probabilistic bound needs only `Var[Y_L]` (not max with Y_U),
//! and the upper needs only `Var[Y_U]`.
//!
//! The distributional objective and gradient enable future integration with
//! the alpha-CROWN optimization loop (`alpha_crown_loop.rs`) to optimize
//! relaxation slopes for distributional tightness instead of worst-case.
//!
//! Part of #3921 / #4249.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::distributional::{AnalyticDistribution, DistributionalBound};
use super::monte_carlo::z_score;
use crate::bounds::LinearBounds;

/// Propagate distribution with path-specific variances for tighter bounds.
///
/// Unlike `propagate_distribution` which uses `max(var_L, var_U)`,
/// this uses `var_L` for the lower probabilistic bound and `var_U` for the
/// upper. This is always at least as tight and often strictly tighter.
///
/// Soundness: f(X) >= Y_L pointwise, so q_alpha(f) >= q_alpha(Y_L).
/// Var[Y_L] uses only A_L coefficients. Symmetric argument for upper.
pub fn propagate_distribution_tight(
    linear_bounds: &LinearBounds,
    input_dist: &AnalyticDistribution,
    input_bounds: &BoundedTensor,
    confidence: f64,
) -> Result<DistributionalBound> {
    if !(0.0..1.0).contains(&confidence) {
        return Err(NyError::InvalidSpec(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }

    let (mu, diag_var) = resolve_dist(input_dist, input_bounds)?;

    let num_inputs = linear_bounds.num_inputs();
    if mu.len() != num_inputs {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_inputs],
            got: vec![mu.len()],
        });
    }

    let num_outputs = linear_bounds.num_outputs();
    let a_l = linear_bounds.lower_a();
    let a_u = linear_bounds.upper_a();
    let b_l = linear_bounds.lower_b();
    let b_u = linear_bounds.upper_b();

    let mu_1d: Array1<f32> = Array1::from_vec(mu.iter().map(|&v| v as f32).collect());
    let mean_lower_1d: Array1<f32> = a_l.dot(&mu_1d) + b_l;
    let mean_upper_1d: Array1<f32> = a_u.dot(&mu_1d) + b_u;

    let dv: Vec<f32> = diag_var.iter().map(|&v| v as f32).collect();
    let z = z_score(confidence) as f32;

    let mut variance_vec = Vec::with_capacity(num_outputs);
    let mut prob_lower_vec = Vec::with_capacity(num_outputs);
    let mut prob_upper_vec = Vec::with_capacity(num_outputs);

    for i in 0..num_outputs {
        let mut var_l = 0.0_f64;
        let mut var_u = 0.0_f64;
        for j in 0..num_inputs {
            let v_j = dv[j] as f64;
            var_l += (a_l[[i, j]] as f64).powi(2) * v_j;
            var_u += (a_u[[i, j]] as f64).powi(2) * v_j;
        }
        // Store max for backward compatibility
        variance_vec.push(var_l.max(var_u) as f32);

        // Path-specific: lower bound uses var_L, upper uses var_U
        let std_l = (var_l.max(0.0).sqrt()) as f32;
        let std_u = (var_u.max(0.0).sqrt()) as f32;
        prob_lower_vec.push(mean_lower_1d[i] - z * std_l);
        prob_upper_vec.push(mean_upper_1d[i] + z * std_u);
    }

    let shape = vec![num_outputs];
    let ix = IxDyn(&shape);
    let mk = |v: Vec<f32>| {
        ArrayD::from_shape_vec(ix.clone(), v)
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))
    };

    Ok(DistributionalBound {
        mean_lower: mk(mean_lower_1d.to_vec())?,
        mean_upper: mk(mean_upper_1d.to_vec())?,
        variance_upper: mk(variance_vec)?,
        prob_lower: mk(prob_lower_vec)?,
        prob_upper: mk(prob_upper_vec)?,
        confidence,
    })
}

/// Compute the distributional objective: sum of probabilistic lower bounds.
///
/// This is the objective DA-CROWN maximizes:
///   sum_i (A_L[i,:] @ mu + b_L[i] - z * sqrt(A_L[i,:]^2 @ var))
///
/// Higher is better (tighter lower bound).
pub fn distributional_objective(
    linear_bounds: &LinearBounds,
    mu: &[f64],
    diag_var: &[f64],
    confidence: f64,
) -> f64 {
    let z = z_score(confidence);
    let a_l = linear_bounds.lower_a();
    let b_l = linear_bounds.lower_b();
    let num_outputs = linear_bounds.num_outputs();
    let num_inputs = linear_bounds.num_inputs();

    let mut objective = 0.0;
    for i in 0..num_outputs {
        let mut mean = b_l[i] as f64;
        let mut var = 0.0_f64;
        for j in 0..num_inputs {
            let a = a_l[[i, j]] as f64;
            mean += a * mu[j];
            var += a * a * diag_var[j];
        }
        objective += mean - z * var.max(0.0).sqrt();
    }
    objective
}

/// Gradient of distributional objective w.r.t. A_L coefficients.
///
/// d(prob_lower_i)/d(A_L[i,j]) = mu_j - z * A_L[i,j] * var_j / sigma_i
///
/// This gradient tells how modifying each element of A_L affects the
/// probabilistic lower bound. Useful for guiding alpha-CROWN optimization.
pub fn distributional_gradient(
    linear_bounds: &LinearBounds,
    mu: &[f64],
    diag_var: &[f64],
    confidence: f64,
) -> Array2<f64> {
    let z = z_score(confidence);
    let a_l = linear_bounds.lower_a();
    let num_outputs = linear_bounds.num_outputs();
    let num_inputs = linear_bounds.num_inputs();

    let mut grad = Array2::zeros((num_outputs, num_inputs));

    for i in 0..num_outputs {
        let mut var_sum = 0.0_f64;
        for j in 0..num_inputs {
            let a = a_l[[i, j]] as f64;
            var_sum += a * a * diag_var[j];
        }
        let sigma = var_sum.max(1e-30).sqrt();

        for j in 0..num_inputs {
            let a = a_l[[i, j]] as f64;
            grad[[i, j]] = mu[j] - z * a * diag_var[j] / sigma;
        }
    }

    grad
}

/// Resolve distribution to f64 mean and variance vectors.
fn resolve_dist(
    dist: &AnalyticDistribution,
    input_bounds: &BoundedTensor,
) -> Result<(Vec<f64>, Vec<f64>)> {
    match dist {
        AnalyticDistribution::DiagonalGaussian { mean, variance } => Ok((
            mean.iter().map(|&v| v as f64).collect(),
            variance.iter().map(|&v| v as f64).collect(),
        )),
        AnalyticDistribution::UniformFromBounds => {
            let mu: Vec<f64> = input_bounds.center().iter().map(|&v| v as f64).collect();
            let var: Vec<f64> = input_bounds
                .width()
                .iter()
                .map(|&w| {
                    let w = w as f64;
                    w * w / 12.0
                })
                .collect();
            Ok((mu, var))
        }
    }
}

#[cfg(test)]
#[path = "distributional_alpha_tests.rs"]
mod tests;
