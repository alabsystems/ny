// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Cumulant-Generating Function (CGF) propagation through CROWN linear bounds.
//!
//! The CGF captures all moments simultaneously and yields Chernoff tail bounds
//! for linear forms of independent inputs.
//!
//! For Y = A @ X + b with independent inputs X_j:
//!   psi_Y(theta) = sum_j psi_{X_j}(a_j * theta) + theta * b
//!
//! Input CGFs:
//! - Uniform on [l, u]: psi(theta) = log((exp(theta*u) - exp(theta*l)) / (theta*(u-l)))
//! - Gaussian N(mu, sigma^2): psi(theta) = theta*mu + theta^2*sigma^2/2
//!
//! Chernoff bound: P(Y >= y) <= min_{theta>0} exp(psi_Y(theta) - theta*y)
//!
//! These bounds are useful analytical comparisons, but the interval widths are
//! not globally ordered against other approximations once composed with CROWN
//! relaxation gaps and numerical search tolerances.
//!
//! Part of #3921 / #4249. Technique from AI Provider 5.4 review.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::distributional::AnalyticDistribution;
use super::{validate_finite_f32, validate_finite_f64};
use crate::bounds::LinearBounds;

/// CGF-based tail bounds from CROWN linear relaxation propagation.
#[derive(Debug, Clone)]
pub struct CgfBound {
    /// Chernoff upper bound: P(Y_U_i >= t) for each output i.
    /// Evaluated at the stored threshold.
    pub exceedance_upper: ArrayD<f64>,
    /// Chernoff lower bound: P(Y_L_i <= t) for each output i.
    pub shortfall_upper: ArrayD<f64>,
    /// Probabilistic lower bound: largest t s.t. P(Y_L < t) <= alpha/2.
    pub prob_lower: ArrayD<f32>,
    /// Probabilistic upper bound: smallest t s.t. P(Y_U > t) <= alpha/2.
    pub prob_upper: ArrayD<f32>,
    /// Confidence level.
    pub confidence: f64,
}

/// Per-element input CGF specification.
#[derive(Debug, Clone)]
enum ElementCgf {
    Uniform { lower: f64, upper: f64 },
    Gaussian { mean: f64, variance: f64 },
}

impl ElementCgf {
    /// Evaluate the CGF: psi(theta) = log E[exp(theta * X)].
    fn psi(&self, theta: f64) -> f64 {
        match self {
            ElementCgf::Gaussian { mean, variance } => {
                theta * mean + 0.5 * theta * theta * variance
            }
            ElementCgf::Uniform { lower, upper } => {
                // Point-mass guard: when lower == upper, CGF is theta * v.
                // Without this, the log-sum-exp produces NaN from 0.0.ln().
                // Part of #4286.
                if (upper - lower).abs() < 1e-15 {
                    return theta * lower;
                }
                if theta.abs() < 1e-15 {
                    return 0.0;
                }
                let tl = theta * lower;
                let tu = theta * upper;
                // psi(theta) = log((exp(tu) - exp(tl)) / (theta * (upper - lower)))
                // Use log-sum-exp trick for numerical stability
                let max_val = tl.max(tu);
                max_val + ((tu - max_val).exp() - (tl - max_val).exp()).ln()
                    - (theta * (upper - lower)).abs().ln()
            }
        }
    }

    /// First derivative: psi'(theta) = E[X * exp(theta*X)] / E[exp(theta*X)].
    fn psi_prime(&self, theta: f64) -> f64 {
        match self {
            ElementCgf::Gaussian { mean, variance } => mean + theta * variance,
            ElementCgf::Uniform { lower, upper } => {
                // Point-mass guard: derivative of theta*v is v.
                // Without this, eu - el = 0 causes division by zero.
                // Part of #4286.
                if (upper - lower).abs() < 1e-15 {
                    return *lower;
                }
                if theta.abs() < 1e-15 {
                    return (lower + upper) / 2.0;
                }
                let tl = theta * lower;
                let tu = theta * upper;
                // psi'(theta) = (u*exp(tu) - l*exp(tl))/(exp(tu)-exp(tl)) - 1/theta
                let max_val = tl.max(tu);
                let el = (tl - max_val).exp();
                let eu = (tu - max_val).exp();
                let numer = upper * eu - lower * el;
                let denom = eu - el;
                numer / denom - 1.0 / theta
            }
        }
    }
}

/// Evaluate the output CGF for row i: psi_Y_i(theta) = sum_j psi_j(a_ij * theta) + theta * b_i.
fn output_cgf(
    coeffs: &ndarray::ArrayView1<'_, f32>,
    bias: f32,
    input_cgfs: &[ElementCgf],
    theta: f64,
) -> f64 {
    let mut psi = theta * bias as f64;
    for (j, cgf) in input_cgfs.iter().enumerate() {
        psi += cgf.psi(coeffs[j] as f64 * theta);
    }
    psi
}

/// First derivative of output CGF.
fn output_cgf_prime(
    coeffs: &ndarray::ArrayView1<'_, f32>,
    bias: f32,
    input_cgfs: &[ElementCgf],
    theta: f64,
) -> f64 {
    let mut dpsi = bias as f64;
    for (j, cgf) in input_cgfs.iter().enumerate() {
        let a = coeffs[j] as f64;
        dpsi += a * cgf.psi_prime(a * theta);
    }
    dpsi
}

/// Find optimal theta for Chernoff bound: min_{theta>0} exp(psi(theta) - theta*y).
/// Uses Newton's method on psi'(theta) = y (saddlepoint equation).
fn optimal_theta_upper(
    coeffs: &ndarray::ArrayView1<'_, f32>,
    bias: f32,
    input_cgfs: &[ElementCgf],
    target: f64,
    max_iters: usize,
) -> Option<f64> {
    // Newton's method: find theta s.t. psi'(theta) = target
    let mut theta = 0.1_f64;

    for _ in 0..max_iters {
        let val = output_cgf_prime(coeffs, bias, input_cgfs, theta);
        let diff = val - target;

        if diff.abs() < 1e-10 {
            return Some(theta);
        }

        // Approximate second derivative numerically
        let h = 1e-6;
        let val_h = output_cgf_prime(coeffs, bias, input_cgfs, theta + h);
        let d2 = (val_h - val) / h;

        if d2.abs() < 1e-20 {
            break;
        }

        let step = diff / d2;
        theta -= step;

        // Keep theta positive
        if theta <= 0.0 {
            theta = 0.01;
        }
    }

    // Fallback: grid search
    let mut best_theta = 0.01;
    let mut best_val = f64::MAX;
    for k in 1..=100 {
        let t = k as f64 * 0.05;
        let val = output_cgf(coeffs, bias, input_cgfs, t) - t * target;
        if val < best_val {
            best_val = val;
            best_theta = t;
        }
    }
    Some(best_theta)
}

/// Compute Chernoff bound: P(Y >= target) <= exp(psi(theta*) - theta* * target).
fn chernoff_upper(
    coeffs: &ndarray::ArrayView1<'_, f32>,
    bias: f32,
    input_cgfs: &[ElementCgf],
    target: f64,
) -> f64 {
    let theta = match optimal_theta_upper(coeffs, bias, input_cgfs, target, 50) {
        Some(t) => t,
        None => return 1.0,
    };

    let exponent = output_cgf(coeffs, bias, input_cgfs, theta) - theta * target;
    exponent.exp().min(1.0)
}

/// Propagate input distribution through CROWN linear bounds using CGF.
///
/// Returns Chernoff-based probabilistic bounds for the supplied CROWN linear
/// relaxation.
pub fn propagate_cgf(
    linear_bounds: &LinearBounds,
    input_dist: &AnalyticDistribution,
    input_bounds: &BoundedTensor,
    confidence: f64,
) -> Result<CgfBound> {
    if !(0.0..1.0).contains(&confidence) {
        return Err(NyError::InvalidSpec(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }

    let input_cgfs = resolve_cgf(input_dist, input_bounds)?;
    let alpha = 1.0 - confidence;

    let num_outputs = linear_bounds.num_outputs();
    let num_inputs = linear_bounds.num_inputs();
    if input_cgfs.len() != num_inputs {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_inputs],
            got: vec![input_cgfs.len()],
        });
    }

    let a_l = linear_bounds.lower_a();
    let a_u = linear_bounds.upper_a();
    let b_l = linear_bounds.lower_b();
    let b_u = linear_bounds.upper_b();

    let mut prob_lower_vec = Vec::with_capacity(num_outputs);
    let mut prob_upper_vec = Vec::with_capacity(num_outputs);
    let mut exc_upper_vec = Vec::with_capacity(num_outputs);
    let mut shortfall_vec = Vec::with_capacity(num_outputs);

    for i in 0..num_outputs {
        let row_u = a_u.row(i);
        let row_l = a_l.row(i);

        // Find prob_upper: smallest t s.t. P(Y_U >= t) <= alpha/2
        // Binary search on t
        let mean_u: f64 = row_u
            .iter()
            .zip(input_cgfs.iter())
            .map(|(&a, cgf)| {
                let a = a as f64;
                match cgf {
                    ElementCgf::Gaussian { mean, .. } => a * mean,
                    ElementCgf::Uniform { lower, upper } => a * (lower + upper) / 2.0,
                }
            })
            .sum::<f64>()
            + b_u[i] as f64;

        let var_u: f64 = row_u
            .iter()
            .zip(input_cgfs.iter())
            .map(|(&a, cgf)| {
                let a = a as f64;
                match cgf {
                    ElementCgf::Gaussian { variance, .. } => a * a * variance,
                    ElementCgf::Uniform { lower, upper } => {
                        let w = upper - lower;
                        a * a * w * w / 12.0
                    }
                }
            })
            .sum();
        let std_u = var_u.max(0.0).sqrt();

        // Search range: mean +/- 6*sigma
        let lo = mean_u - 0.5 * std_u;
        let hi = mean_u + 6.0 * std_u;
        let upper_t = binary_search_quantile(
            |t| chernoff_upper(&row_u, b_u[i], &input_cgfs, t),
            alpha / 2.0,
            lo,
            hi,
            50,
        );
        prob_upper_vec.push(upper_t as f32);
        exc_upper_vec.push(chernoff_upper(&row_u, b_u[i], &input_cgfs, upper_t));

        // Find prob_lower: largest t s.t. P(Y_L <= t) <= alpha/2
        // P(Y_L <= t) = P(-Y_L >= -t), so use Chernoff on -Y_L
        let mean_l: f64 = row_l
            .iter()
            .zip(input_cgfs.iter())
            .map(|(&a, cgf)| {
                let a = a as f64;
                match cgf {
                    ElementCgf::Gaussian { mean, .. } => a * mean,
                    ElementCgf::Uniform { lower, upper } => a * (lower + upper) / 2.0,
                }
            })
            .sum::<f64>()
            + b_l[i] as f64;

        let var_l: f64 = row_l
            .iter()
            .zip(input_cgfs.iter())
            .map(|(&a, cgf)| {
                let a = a as f64;
                match cgf {
                    ElementCgf::Gaussian { variance, .. } => a * a * variance,
                    ElementCgf::Uniform { lower, upper } => {
                        let w = upper - lower;
                        a * a * w * w / 12.0
                    }
                }
            })
            .sum();
        let std_l = var_l.max(0.0).sqrt();

        // For lower tail: P(Y_L <= t) = P(-Y_L >= -t)
        // Negate coefficients and bias for the Chernoff computation
        let neg_row: Array1<f32> = row_l.mapv(|a| -a);
        let neg_bias = -b_l[i];
        let neg_row_view = neg_row.view();
        let lo_neg = -(mean_l + 0.5 * std_l);
        let hi_neg = -(mean_l - 6.0 * std_l);
        let neg_lower_t = binary_search_quantile(
            |t| chernoff_upper(&neg_row_view, neg_bias, &input_cgfs, t),
            alpha / 2.0,
            lo_neg,
            hi_neg,
            50,
        );
        let lower_t = -neg_lower_t;
        prob_lower_vec.push(lower_t as f32);
        shortfall_vec.push(chernoff_upper(
            &neg_row_view,
            neg_bias,
            &input_cgfs,
            neg_lower_t,
        ));
    }

    let shape = vec![num_outputs];
    let ix = IxDyn(&shape);

    let exceedance_upper = ArrayD::from_shape_vec(ix.clone(), exc_upper_vec)
        .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))?;
    let shortfall_upper = ArrayD::from_shape_vec(ix.clone(), shortfall_vec)
        .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))?;
    let prob_lower = ArrayD::from_shape_vec(ix.clone(), prob_lower_vec)
        .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))?;
    let prob_upper = ArrayD::from_shape_vec(ix, prob_upper_vec)
        .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))?;

    // Validate finiteness: NaN/Inf from divergent Newton's method must not propagate.
    validate_finite_f64(&exceedance_upper, "exceedance_upper")?;
    validate_finite_f64(&shortfall_upper, "shortfall_upper")?;
    validate_finite_f32(&prob_lower, "prob_lower")?;
    validate_finite_f32(&prob_upper, "prob_upper")?;

    Ok(CgfBound {
        exceedance_upper,
        shortfall_upper,
        prob_lower,
        prob_upper,
        confidence,
    })
}

/// Binary search for the quantile: find t where tail_prob(t) ~ target.
fn binary_search_quantile(
    tail_prob: impl Fn(f64) -> f64,
    target: f64,
    mut lo: f64,
    mut hi: f64,
    max_iters: usize,
) -> f64 {
    for _ in 0..max_iters {
        let mid = f64::midpoint(lo, hi);
        let p = tail_prob(mid);
        if p > target {
            lo = mid; // tail is too heavy, increase threshold
        } else {
            hi = mid; // tail is small enough, decrease threshold
        }
        if (hi - lo).abs() < 1e-8 {
            break;
        }
    }
    f64::midpoint(lo, hi)
}

/// Resolve input distribution to per-element CGF specifications.
fn resolve_cgf(
    dist: &AnalyticDistribution,
    input_bounds: &BoundedTensor,
) -> Result<Vec<ElementCgf>> {
    match dist {
        AnalyticDistribution::DiagonalGaussian { mean, variance } => Ok(mean
            .iter()
            .zip(variance.iter())
            .map(|(&m, &v)| ElementCgf::Gaussian {
                mean: m as f64,
                variance: v as f64,
            })
            .collect()),
        AnalyticDistribution::UniformFromBounds => {
            let lower = input_bounds.lower();
            let upper = input_bounds.upper();
            Ok(lower
                .iter()
                .zip(upper.iter())
                .map(|(&l, &u)| ElementCgf::Uniform {
                    lower: l as f64,
                    upper: u as f64,
                })
                .collect())
        }
    }
}

#[cfg(test)]
#[path = "cgf_tests.rs"]
mod tests;
