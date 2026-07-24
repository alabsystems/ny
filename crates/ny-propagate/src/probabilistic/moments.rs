// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Higher-order moment propagation through CROWN linear bounds.
//!
//! Extends distributional propagation with skewness and kurtosis for tighter
//! quantile estimates via the Cornish-Fisher expansion. For uniform inputs,
//! excess kurtosis = -1.2 (lighter tails than Gaussian), giving ~18% tighter
//! intervals at 99% confidence.
//!
//! The Cornish-Fisher and Gaussian intervals produced here are approximation
//! layers, not formal coverage guarantees for arbitrary output distributions.
//!
//! For Y = A @ X + b with independent inputs X_j:
//! - Third central moment: mu3(Y_i) = sum_j a_ij^3 * mu3(X_j)
//! - Fourth central moment: mu4(Y_i) = sum_j a_ij^4 * mu4(X_j)
//!   + 3 * [Var(Y_i)^2 - sum_j a_ij^4 * Var(X_j)^2]
//! - Excess kurtosis: gamma2 = sum_j a_ij^4 * Var_j^2 * kappa_j / Var(Y_i)^2
//!
//! Part of #3921 / #4249.

use ndarray::{Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::distributional::AnalyticDistribution;
use super::monte_carlo::z_score;
use crate::bounds::LinearBounds;

/// Higher-order moment bounds from CROWN linear relaxation propagation.
#[derive(Debug, Clone)]
pub struct MomentBound {
    /// Mean of output lower bound.
    pub mean_lower: ArrayD<f32>,
    /// Mean of output upper bound.
    pub mean_upper: ArrayD<f32>,
    /// Variance upper bound per output dimension.
    pub variance_upper: ArrayD<f32>,
    /// Standardized skewness per output dimension (ny_1 = mu3/sigma^3).
    /// Zero for symmetric input distributions (uniform, Gaussian).
    pub skewness: ArrayD<f64>,
    /// Excess kurtosis per output dimension (ny_2 = mu4/sigma^4 - 3).
    /// Negative for platykurtic distributions (lighter tails than Gaussian).
    pub excess_kurtosis: ArrayD<f64>,
    /// Confidence level.
    pub confidence: f64,
    /// Probabilistic bounds using the Cornish-Fisher approximation.
    pub prob_lower: ArrayD<f32>,
    pub prob_upper: ArrayD<f32>,
    /// Gaussian-quantile baseline used as a conservative comparison point.
    pub prob_lower_gaussian: ArrayD<f32>,
    pub prob_upper_gaussian: ArrayD<f32>,
}

/// Cornish-Fisher expansion: adjust the Gaussian z-score using skewness
/// and kurtosis for tighter quantile estimates.
///
/// w = z + (z^2-1)*gamma1/6 + (z^3-3z)*gamma2/24 - (2z^3-5z)*gamma1^2/36
///
/// For symmetric distributions (gamma1=0): w = z + (z^3-3z)*gamma2/24.
/// For uniform inputs (gamma2=-1.2): w < z (tighter).
///
/// Reference: Cornish & Fisher, "Moments and cumulants in the specification
/// of distributions", 1938.
pub fn cornish_fisher_w(z: f64, gamma1: f64, gamma2: f64) -> f64 {
    let z2 = z * z;
    let z3 = z2 * z;
    z + (z2 - 1.0) * gamma1 / 6.0 + (z3 - 3.0 * z) * gamma2 / 24.0
        - (2.0 * z3 - 5.0 * z) * gamma1 * gamma1 / 36.0
}

/// Propagate first four moments through CROWN linear bounds.
///
/// Returns a `MomentBound` with Cornish-Fisher and Gaussian-quantile
/// approximations for output intervals.
pub fn propagate_moments(
    linear_bounds: &LinearBounds,
    input_dist: &AnalyticDistribution,
    input_bounds: &BoundedTensor,
    confidence: f64,
) -> Result<MomentBound> {
    if !(0.0..1.0).contains(&confidence) {
        return Err(NyError::InvalidSpec(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }

    let im = resolve_four_moments(input_dist, input_bounds)?;

    let num_inputs = linear_bounds.num_inputs();
    if im.mean.len() != num_inputs {
        return Err(NyError::ShapeMismatch {
            expected: vec![num_inputs],
            got: vec![im.mean.len()],
        });
    }

    let num_outputs = linear_bounds.num_outputs();
    let a_l = linear_bounds.lower_a();
    let a_u = linear_bounds.upper_a();
    let b_l = linear_bounds.lower_b();
    let b_u = linear_bounds.upper_b();

    let mu_1d: Array1<f32> = Array1::from_vec(im.mean.iter().copied().collect());
    let mean_lower_1d: Array1<f32> = a_l.dot(&mu_1d) + b_l;
    let mean_upper_1d: Array1<f32> = a_u.dot(&mu_1d) + b_u;

    let dv: Vec<f32> = im.variance.iter().copied().collect();
    let m3: Vec<f64> = im.mu3.iter().copied().collect();
    let m4: Vec<f64> = im.mu4.iter().copied().collect();

    let mut variance_vec = Vec::with_capacity(num_outputs);
    let mut skewness_vec = Vec::with_capacity(num_outputs);
    let mut kurtosis_vec = Vec::with_capacity(num_outputs);

    for i in 0..num_outputs {
        // Variance: max of lower and upper path
        let mut var_l = 0.0_f64;
        let mut var_u = 0.0_f64;
        for j in 0..num_inputs {
            let v_j = dv[j] as f64;
            var_l += (a_l[[i, j]] as f64).powi(2) * v_j;
            var_u += (a_u[[i, j]] as f64).powi(2) * v_j;
        }
        let var = var_l.max(var_u);
        variance_vec.push(var as f32);

        // Third central moment: sum(a^3 * mu3_j)
        // Use worst-case (largest magnitude) across A_L and A_U
        let mut m3_l = 0.0_f64;
        let mut m3_u = 0.0_f64;
        for j in 0..num_inputs {
            m3_l += (a_l[[i, j]] as f64).powi(3) * m3[j];
            m3_u += (a_u[[i, j]] as f64).powi(3) * m3[j];
        }
        let third_moment = if m3_l.abs() > m3_u.abs() { m3_l } else { m3_u };

        // Fourth central moment for excess kurtosis
        // gamma2 = sum(a^4 * var_j^2 * kappa_j) / var^2
        // where kappa_j = mu4_j / var_j^2 - 3
        let mut gamma2_l = 0.0_f64;
        let mut gamma2_u = 0.0_f64;
        if var > 0.0 {
            for j in 0..num_inputs {
                let v_j = dv[j] as f64;
                if v_j > 0.0 {
                    let kappa_j = m4[j] / (v_j * v_j) - 3.0;
                    gamma2_l += (a_l[[i, j]] as f64).powi(4) * v_j * v_j * kappa_j;
                    gamma2_u += (a_u[[i, j]] as f64).powi(4) * v_j * v_j * kappa_j;
                }
            }
            gamma2_l /= var * var;
            gamma2_u /= var * var;
        }

        // Skewness
        let sigma = var.sqrt();
        let gamma1 = if sigma > 0.0 {
            third_moment / (sigma * sigma * sigma)
        } else {
            0.0
        };
        skewness_vec.push(gamma1);

        // Use the kurtosis closest to 0 as the more conservative
        // Cornish-Fisher approximation to avoid over-tightening.
        let gamma2 = if gamma2_l.abs() < gamma2_u.abs() {
            gamma2_l
        } else {
            gamma2_u
        };
        kurtosis_vec.push(gamma2);
    }

    // Compute probabilistic bounds
    let z = z_score(confidence);
    let z_f32 = z as f32;

    let mut pl_cf = Vec::with_capacity(num_outputs);
    let mut pu_cf = Vec::with_capacity(num_outputs);
    let mut pl_g = Vec::with_capacity(num_outputs);
    let mut pu_g = Vec::with_capacity(num_outputs);

    for i in 0..num_outputs {
        let std_i = (variance_vec[i] as f64).max(0.0).sqrt() as f32;

        // Gaussian-quantile baseline
        pl_g.push(mean_lower_1d[i] - z_f32 * std_i);
        pu_g.push(mean_upper_1d[i] + z_f32 * std_i);

        // Cornish-Fisher bounds, clamped to never be worse than Gaussian.
        // CF is an approximation that can diverge for large kurtosis.
        let w = cornish_fisher_w(z, skewness_vec[i], kurtosis_vec[i]);
        let w_f32 = w as f32;
        let cf_lo = mean_lower_1d[i] - w_f32 * std_i;
        let cf_hi = mean_upper_1d[i] + w_f32 * std_i;
        pl_cf.push(cf_lo.max(pl_g[i]));
        pu_cf.push(cf_hi.min(pu_g[i]));
    }

    let shape = vec![num_outputs];
    let ix = IxDyn(&shape);
    let mk = |v: Vec<_>| {
        ArrayD::from_shape_vec(ix.clone(), v)
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))
    };

    Ok(MomentBound {
        mean_lower: mk(mean_lower_1d.to_vec())?,
        mean_upper: mk(mean_upper_1d.to_vec())?,
        variance_upper: mk(variance_vec)?,
        skewness: ArrayD::from_shape_vec(ix.clone(), skewness_vec)
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
        excess_kurtosis: ArrayD::from_shape_vec(ix.clone(), kurtosis_vec)
            .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
        confidence,
        prob_lower: mk(pl_cf)?,
        prob_upper: mk(pu_cf)?,
        prob_lower_gaussian: mk(pl_g)?,
        prob_upper_gaussian: mk(pu_g)?,
    })
}

/// Per-element input moments for moment propagation.
struct InputMoments {
    mean: ArrayD<f32>,
    variance: ArrayD<f32>,
    mu3: ArrayD<f64>,
    mu4: ArrayD<f64>,
}

/// Resolve input distribution to four central moments.
fn resolve_four_moments(
    dist: &AnalyticDistribution,
    input_bounds: &BoundedTensor,
) -> Result<InputMoments> {
    match dist {
        AnalyticDistribution::DiagonalGaussian { mean, variance } => {
            let n = mean.len();
            // Gaussian: mu3 = 0, mu4 = 3 * var^2
            let mu3 = ArrayD::zeros(IxDyn(&[n]));
            let mu4_vec: Vec<f64> = variance
                .iter()
                .map(|&v| 3.0 * (v as f64) * (v as f64))
                .collect();
            let mu4 = ArrayD::from_shape_vec(IxDyn(&[n]), mu4_vec)
                .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))?;
            Ok(InputMoments {
                mean: mean.as_ref().clone(),
                variance: variance.as_ref().clone(),
                mu3,
                mu4,
            })
        }
        AnalyticDistribution::UniformFromBounds => {
            let mu = input_bounds.center();
            let width = input_bounds.width();
            let variance = width.mapv(|w| w * w / 12.0);
            let n = mu.len();
            // Uniform on [a,b]: mu3 = 0, mu4 = (b-a)^4 / 80
            let mu3 = ArrayD::zeros(IxDyn(&[n]));
            let mu4_vec: Vec<f64> = width
                .iter()
                .map(|&w| {
                    let w = w as f64;
                    w * w * w * w / 80.0
                })
                .collect();
            let mu4 = ArrayD::from_shape_vec(IxDyn(&[n]), mu4_vec)
                .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))?;
            Ok(InputMoments {
                mean: mu,
                variance,
                mu3,
                mu4,
            })
        }
    }
}

#[cfg(test)]
#[path = "moments_tests.rs"]
mod tests;
