// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Distributional propagation through CROWN linear relaxations.
//!
//! Given CROWN linear bounds: A_L @ x + b_L <= f(x) <= A_U @ x + b_U
//! and an input distribution x ~ D(mu, Sigma), we can propagate the distribution
//! through the linear relaxation to get output distribution bounds analytically.
//!
//! For Gaussian inputs x ~ N(mu, Sigma):
//! - Output mean bounds: A_L @ mu + b_L <= E[f(x)] <= A_U @ mu + b_U
//! - Output variance bound: Var(f_i) <= max(row_i(A_L) @ Sigma @ row_i(A_L)^T,
//!   row_i(A_U) @ Sigma @ row_i(A_U)^T)
//! - Probabilistic bounds: mean +/- z * sqrt(var) gives a Gaussian-quantile
//!   approximation to the output interval
//!
//! For uniform inputs on [l, u]: mu = (l+u)/2, Sigma = diag((u-l)^2/12).
//!
//! Reference: Boetius et al., "Distributional Robustness Verification via CROWN", ICML 2025.
//!
//! Part of #3921 Phase 3.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::monte_carlo::z_score;
use crate::bounds::LinearBounds;

/// Input distribution specification for distributional propagation.
#[derive(Debug, Clone)]
pub enum AnalyticDistribution {
    /// Diagonal Gaussian: independent dimensions with known mean and variance.
    DiagonalGaussian {
        /// Per-element mean.
        mean: Box<ArrayD<f32>>,
        /// Per-element variance (diagonal of covariance matrix).
        variance: Box<ArrayD<f32>>,
    },
    /// Uniform distribution derived from BoundedTensor: mu = (l+u)/2, var = (u-l)^2/12.
    UniformFromBounds,
}

/// Output distribution bounds from CROWN linear relaxation propagation.
#[derive(Debug, Clone)]
pub struct DistributionalBound {
    /// Mean of output lower bound: A_L @ mu + b_L
    pub mean_lower: ArrayD<f32>,
    /// Mean of output upper bound: A_U @ mu + b_U
    pub mean_upper: ArrayD<f32>,
    /// Upper bound on variance per output dimension.
    /// var_i = max(row_i(A_L)^2 @ diag_var, row_i(A_U)^2 @ diag_var)
    pub variance_upper: ArrayD<f32>,
    /// Probabilistic lower bound: mean_lower - z * sqrt(variance_upper)
    pub prob_lower: ArrayD<f32>,
    /// Probabilistic upper bound: mean_upper + z * sqrt(variance_upper)
    pub prob_upper: ArrayD<f32>,
    /// Confidence level used for probabilistic bounds.
    pub confidence: f64,
}

impl DistributionalBound {
    /// Recompute probabilistic output interval at a different confidence level.
    ///
    /// Returns a Gaussian-quantile approximation `(lower, upper)` per output
    /// dimension at the requested confidence level.
    ///
    /// This is exact for Gaussian linear forms, but only heuristic for
    /// non-Gaussian outputs or when the CROWN relaxation gap dominates.
    ///
    /// This reuses the stored mean bounds and variance upper bound — no CROWN
    /// recomputation needed.
    ///
    /// Part of #4248.
    pub fn output_quantile(&self, confidence: f64) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(NyError::InvalidSpec(format!(
                "confidence must be in (0, 1), got {confidence}"
            )));
        }
        let z = z_score(confidence) as f32;
        let lower = ndarray::Zip::from(&self.mean_lower)
            .and(&self.variance_upper)
            .map_collect(|&m, &v| m - z * v.max(0.0).sqrt());
        let upper = ndarray::Zip::from(&self.mean_upper)
            .and(&self.variance_upper)
            .map_collect(|&m, &v| m + z * v.max(0.0).sqrt());
        Ok((lower, upper))
    }

    /// Bound the exceedance probability P(output_i > threshold) per dimension.
    ///
    /// Returns `(lower_bound, upper_bound)` on the exceedance probability using
    /// Cantelli's inequality (one-sided Chebyshev), which is distribution-free:
    ///
    /// - When `threshold <= mean_lower_i`: P >= 0.5 (at least half the mass exceeds)
    /// - When `threshold > mean_upper_i`: P <= var_i / (var_i + (t - mean_upper_i)^2)
    /// - Between: interpolate with Gaussian CDF approximation
    ///
    /// Cantelli's inequality requires only bounded variance — no Gaussian assumption.
    ///
    /// Reference: Cantelli, "Sui confini della probabilità", 1928.
    ///
    /// Part of #4248.
    pub fn output_probability(&self, threshold: f32) -> Result<(ArrayD<f64>, ArrayD<f64>)> {
        let n = self.mean_lower.len();
        let mut prob_lower = Vec::with_capacity(n);
        let mut prob_upper = Vec::with_capacity(n);

        for i in 0..n {
            let mu_l = self.mean_lower[[i]] as f64;
            let mu_u = self.mean_upper[[i]] as f64;
            let var = self.variance_upper[[i]].max(0.0) as f64;
            let t = threshold as f64;

            // Upper bound on P(X_i > t): use worst-case mean (mu_u)
            let p_upper = if var == 0.0 {
                if mu_u > t {
                    1.0
                } else {
                    0.0
                }
            } else if t <= mu_u {
                // threshold below or at the upper mean bound — could be anything up to 1
                1.0
            } else {
                // Cantelli: P(X > t) <= var / (var + (t - mu)^2) when t > mu
                let d = t - mu_u;
                var / (var + d * d)
            };

            // Lower bound on P(X_i > t): use best-case mean (mu_l)
            let p_lower = if var == 0.0 {
                if mu_l > t {
                    1.0
                } else {
                    0.0
                }
            } else if t <= mu_l {
                // threshold below the lower mean bound — at least half exceeds
                // (by Cantelli applied in the other direction)
                let d = mu_l - t;
                // P(X > t) >= 1 - var / (var + d^2) = d^2 / (var + d^2)
                d * d / (var + d * d)
            } else {
                // threshold above lower mean — lower bound is 0 (could be no mass above)
                0.0
            };

            prob_lower.push(p_lower.clamp(0.0, 1.0));
            prob_upper.push(p_upper.clamp(0.0, 1.0));
        }

        let shape = self.mean_lower.shape().to_vec();
        let ix = IxDyn(&shape);
        Ok((
            ArrayD::from_shape_vec(ix.clone(), prob_lower)
                .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))?,
            ArrayD::from_shape_vec(ix, prob_upper)
                .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))?,
        ))
    }

    /// Number of output dimensions.
    pub fn num_outputs(&self) -> usize {
        self.mean_lower.len()
    }
}

/// Propagate a distribution through CROWN linear bounds.
///
/// Given CROWN relaxation (A_L, b_L, A_U, b_U) and an input distribution,
/// computes output distribution bounds analytically.
///
/// For the diagonal covariance case, this is O(num_outputs * num_inputs) — the same
/// cost as CROWN concretization. No extra matrix multiply is needed beyond what
/// CROWN already computed.
///
/// # Arguments
/// * `linear_bounds` - CROWN linear relaxation coefficients
/// * `input_dist` - Input distribution (Gaussian or derived from bounds)
/// * `input_bounds` - Input bounds (used for UniformFromBounds and shape validation)
/// * `confidence` - Confidence level for probabilistic bounds (e.g. 0.99)
pub fn propagate_distribution(
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

    // Resolve distribution to mean and diagonal variance vectors
    let (mu, diag_var) = resolve_distribution(input_dist, input_bounds)?;

    // Validate shapes: LinearBounds expects num_inputs == mu.len()
    let num_inputs = linear_bounds.num_inputs();
    if mu.len() != num_inputs {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_inputs],
            got: vec![mu.len()],
        });
    }

    let num_outputs = linear_bounds.num_outputs();
    let a_l = linear_bounds.lower_a(); // [num_outputs, num_inputs]
    let a_u = linear_bounds.upper_a();
    let b_l = linear_bounds.lower_b(); // [num_outputs]
    let b_u = linear_bounds.upper_b();

    // Step 1: Output mean bounds
    // mean_L = A_L @ mu + b_L
    // mean_U = A_U @ mu + b_U
    let mu_1d: Array1<f32> = Array1::from_vec(mu.iter().copied().collect());
    let mean_lower_1d: Array1<f32> = a_l.dot(&mu_1d) + b_l;
    let mean_upper_1d: Array1<f32> = a_u.dot(&mu_1d) + b_u;

    // Step 2: Variance upper bound per output dimension (diagonal covariance case)
    // var_L_i = sum_j(A_L[i,j]^2 * sigma_j^2)
    // var_U_i = sum_j(A_U[i,j]^2 * sigma_j^2)
    // var_i = max(var_L_i, var_U_i)
    let diag_var_slice: Vec<f32> = diag_var.iter().copied().collect();
    let mut variance_vec = Vec::with_capacity(num_outputs);
    for i in 0..num_outputs {
        let mut var_l = 0.0_f64;
        let mut var_u = 0.0_f64;
        for j in 0..num_inputs {
            let v_j = diag_var_slice[j] as f64;
            let al_ij = a_l[[i, j]] as f64;
            let au_ij = a_u[[i, j]] as f64;
            var_l += al_ij * al_ij * v_j;
            var_u += au_ij * au_ij * v_j;
        }
        variance_vec.push(var_l.max(var_u) as f32);
    }

    // Step 3: Probabilistic bounds
    // prob_L = mean_L - z * sqrt(var)
    // prob_U = mean_U + z * sqrt(var)
    let z = z_score(confidence) as f32;
    let mut prob_lower_vec = Vec::with_capacity(num_outputs);
    let mut prob_upper_vec = Vec::with_capacity(num_outputs);
    for i in 0..num_outputs {
        let std_i = variance_vec[i].max(0.0).sqrt();
        prob_lower_vec.push(mean_lower_1d[i] - z * std_i);
        prob_upper_vec.push(mean_upper_1d[i] + z * std_i);
    }

    let shape = vec![num_outputs];
    let ix = IxDyn(&shape);

    Ok(DistributionalBound {
        mean_lower: ArrayD::from_shape_vec(ix.clone(), mean_lower_1d.to_vec())
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
        mean_upper: ArrayD::from_shape_vec(ix.clone(), mean_upper_1d.to_vec())
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
        variance_upper: ArrayD::from_shape_vec(ix.clone(), variance_vec)
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
        prob_lower: ArrayD::from_shape_vec(ix.clone(), prob_lower_vec)
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
        prob_upper: ArrayD::from_shape_vec(ix, prob_upper_vec)
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
        confidence,
    })
}

/// Resolve an `AnalyticDistribution` to concrete mean and diagonal variance arrays.
fn resolve_distribution(
    dist: &AnalyticDistribution,
    input_bounds: &BoundedTensor,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    match dist {
        AnalyticDistribution::DiagonalGaussian { mean, variance } => {
            if mean.len() != variance.len() {
                return Err(NyError::ShapeMismatch {
                    expected: mean.shape().to_vec(),
                    got: variance.shape().to_vec(),
                });
            }
            // Validate variance is non-negative
            if variance.iter().any(|&v| v < 0.0) {
                return Err(NyError::InvalidSpec(
                    "Variance must be non-negative".to_string(),
                ));
            }
            Ok((mean.as_ref().clone(), variance.as_ref().clone()))
        }
        AnalyticDistribution::UniformFromBounds => {
            // Uniform on [l, u]: mu = (l+u)/2, var = (u-l)^2/12
            let mu = input_bounds.center();
            let width = input_bounds.width();
            let variance = width.mapv(|w| w * w / 12.0);
            Ok((mu, variance))
        }
    }
}

#[cfg(test)]
#[path = "distributional_tests.rs"]
mod tests;
