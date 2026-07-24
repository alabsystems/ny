// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Concentration inequality certificates for probabilistic neural network verification.
//!
//! Provides non-asymptotic probabilistic certificates using:
//! - **Hoeffding's inequality**: For bounded outputs, gives confidence bounds on the
//!   true mean from empirical samples. CROWN bounds provide the bounded range [a_i, b_i].
//! - **McDiarmid's inequality**: For Lipschitz networks, gives confidence bounds on
//!   output deviation from its expectation. Lipschitz constants come from spectral norms.
//!
//! These certificates do not require distributional assumptions beyond boundedness
//! (Hoeffding) or bounded differences (McDiarmid).
//!
//! References:
//! - Hoeffding, "Probability Inequalities for Sums of Bounded Random Variables", JASA 1963.
//! - McDiarmid, "On the method of bounded differences", Surveys in Combinatorics 1989.
//!
//! Part of #3921 Phase 2.

use ndarray::ArrayD;
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use crate::layers::Layer;
use crate::Network;

/// Hoeffding bound for a single output dimension.
///
/// Given n i.i.d. samples with output in [a_i, b_i], the empirical mean satisfies:
/// P(|X_bar_i - E[X_bar_i]| >= epsilon) <= 2 * exp(-2*n*epsilon^2 / (b_i - a_i)^2)
#[derive(Debug, Clone)]
pub struct HoeffdingBound {
    /// Output dimension index.
    pub dimension: usize,
    /// Empirical mean from Monte Carlo samples.
    pub empirical_mean: f32,
    /// Bound range (b_i - a_i) from CROWN.
    pub bound_range: f32,
    /// Half-width of the confidence interval around the empirical mean.
    pub epsilon: f64,
    /// Failure probability: P(|X_bar - E[X_bar]| >= epsilon).
    pub failure_probability: f64,
    /// Number of Monte Carlo samples used.
    pub num_samples: usize,
}

/// McDiarmid bound for a single output dimension of a Lipschitz network.
///
/// If changing input coordinate j by at most delta_j changes output i by at most c_{ij}, then:
/// P(|f_i(X) - E[f_i(X)]| >= epsilon) <= 2 * exp(-2*epsilon^2 / sum_j(c_{ij}^2))
#[derive(Debug, Clone)]
pub struct McDiarmidBound {
    /// Output dimension index.
    pub dimension: usize,
    /// Empirical output value from a sample.
    pub empirical_value: f32,
    /// L2 norm of bounded differences: sqrt(sum_j(c_{ij}^2)).
    pub bounded_difference_norm: f64,
    /// Half-width of the confidence interval.
    pub epsilon: f64,
    /// Failure probability.
    pub failure_probability: f64,
    /// Number of samples used.
    pub num_samples: usize,
    /// Whether the underlying Lipschitz estimate is a sound upper bound.
    /// When `false`, the bound was computed with an optimistic estimate for
    /// unhandled layers and should not be trusted for formal verification.
    /// Part of #4145.
    pub is_sound: bool,
}

/// Result of estimating a network-wide Lipschitz constant.
#[derive(Debug, Clone, PartialEq)]
pub struct LipschitzEstimate {
    /// Estimated global Lipschitz constant (product of recognized per-layer bounds).
    pub value: f32,
    /// Whether the estimate is a sound upper bound for the whole network.
    pub is_sound: bool,
    /// Layer types that were treated optimistically as 1-Lipschitz.
    pub unhandled_layers: Vec<String>,
}

/// Combined concentration inequality certificate.
#[derive(Debug, Clone)]
pub struct ConcentrationCertificate {
    /// Per-dimension Hoeffding bounds (always present when CROWN bounds available).
    pub hoeffding_bounds: Vec<HoeffdingBound>,
    /// Per-dimension McDiarmid bounds (present when Lipschitz constant is available).
    pub mcdiarmid_bounds: Option<Vec<McDiarmidBound>>,
    /// Confidence level requested when constructing the component bounds.
    pub overall_confidence: f64,
    /// Whether all components of this certificate are sound.
    /// `false` when any underlying estimate (e.g., Lipschitz constant) is optimistic.
    /// Callers MUST check this before using the certificate for formal verification.
    /// Part of #4145.
    pub is_sound: bool,
}

impl ConcentrationCertificate {
    /// Apply Bonferroni correction for simultaneous multi-dimensional coverage.
    ///
    /// By the union bound, the failure probability for d simultaneous bounds
    /// is at most d * delta_per_dim. To achieve overall failure probability
    /// delta = 1 - confidence, we use per-dim delta = delta / d, i.e.,
    /// per-dim confidence = 1 - (1 - confidence) / d.
    fn effective_confidence(confidence: f64, num_dims: usize, bonferroni: bool) -> f64 {
        if bonferroni && num_dims > 1 {
            1.0 - (1.0 - confidence) / num_dims as f64
        } else {
            confidence
        }
    }

    /// Build a certificate with Hoeffding bounds only.
    ///
    /// When `bonferroni` is true (recommended for multi-dimensional outputs),
    /// per-dimension failure probability is divided by the number of dimensions
    /// so that `overall_confidence` is a valid simultaneous guarantee.
    ///
    /// Hoeffding bounds require only CROWN range information and are always sound,
    /// so `is_sound` is set to `true`.
    pub fn compute(
        empirical_mean: &ArrayD<f32>,
        crown_bounds: &BoundedTensor,
        num_samples: usize,
        confidence: f64,
        bonferroni: bool,
    ) -> Result<Self> {
        let d = empirical_mean.len();
        let eff_conf = Self::effective_confidence(confidence, d, bonferroni);
        Ok(Self {
            hoeffding_bounds: hoeffding_bound(empirical_mean, crown_bounds, num_samples, eff_conf)?,
            mcdiarmid_bounds: None,
            overall_confidence: confidence,
            is_sound: true,
        })
    }

    /// Build a certificate containing both Hoeffding and McDiarmid bounds.
    ///
    /// Returns an error if `lipschitz_estimate.is_sound` is `false`. Callers
    /// that accept optimistic estimates should use
    /// [`Self::compute_with_mcdiarmid_optimistic`].
    ///
    /// Part of #4145: enforced contract replaces warning-only fallback.
    pub fn compute_with_mcdiarmid(
        empirical_mean: &ArrayD<f32>,
        crown_bounds: &BoundedTensor,
        empirical_output: &ArrayD<f32>,
        input_bounds: &BoundedTensor,
        lipschitz_estimate: &LipschitzEstimate,
        num_samples: usize,
        confidence: f64,
        bonferroni: bool,
    ) -> Result<Self> {
        let d = empirical_mean.len();
        let eff_conf = Self::effective_confidence(confidence, d, bonferroni);
        let mut certificate = Self::compute(
            empirical_mean,
            crown_bounds,
            num_samples,
            confidence,
            bonferroni,
        )?;
        certificate.mcdiarmid_bounds = Some(mcdiarmid_bound(
            empirical_output,
            input_bounds,
            lipschitz_estimate,
            num_samples,
            eff_conf,
        )?);
        certificate.is_sound = lipschitz_estimate.is_sound;
        Ok(certificate)
    }

    /// Build a certificate accepting an optimistic (possibly unsound) Lipschitz
    /// estimate.
    ///
    /// The returned certificate's `is_sound` reflects the estimate's soundness.
    /// Use this when the caller explicitly acknowledges that the McDiarmid
    /// component may not be a valid upper bound.
    ///
    /// Part of #4145: explicit opt-in for unsound path.
    pub fn compute_with_mcdiarmid_optimistic(
        empirical_mean: &ArrayD<f32>,
        crown_bounds: &BoundedTensor,
        empirical_output: &ArrayD<f32>,
        input_bounds: &BoundedTensor,
        lipschitz_estimate: &LipschitzEstimate,
        num_samples: usize,
        confidence: f64,
        bonferroni: bool,
    ) -> Result<Self> {
        let d = empirical_mean.len();
        let eff_conf = Self::effective_confidence(confidence, d, bonferroni);
        let mut certificate = Self::compute(
            empirical_mean,
            crown_bounds,
            num_samples,
            confidence,
            bonferroni,
        )?;
        certificate.mcdiarmid_bounds = Some(mcdiarmid_bound_optimistic(
            empirical_output,
            input_bounds,
            lipschitz_estimate,
            num_samples,
            eff_conf,
        )?);
        certificate.is_sound = lipschitz_estimate.is_sound;
        Ok(certificate)
    }
}

/// Compute Hoeffding concentration bounds from empirical means and CROWN ranges.
///
/// For each output dimension i with CROWN range R_i = upper_i - lower_i:
/// epsilon_i = R_i * sqrt(ln(2/delta) / (2*n))
///
/// where delta = 1 - confidence (per-dimension failure probability).
///
/// # Arguments
/// * `empirical_mean` - Per-element empirical mean from Monte Carlo samples
/// * `crown_bounds` - CROWN sound over-approximation providing [lower, upper] ranges
/// * `num_samples` - Number of Monte Carlo samples used
/// * `confidence` - Desired confidence level (e.g. 0.95 for 95%)
pub fn hoeffding_bound(
    empirical_mean: &ArrayD<f32>,
    crown_bounds: &BoundedTensor,
    num_samples: usize,
    confidence: f64,
) -> Result<Vec<HoeffdingBound>> {
    if num_samples == 0 {
        return Err(NyError::InvalidSpec(
            "Hoeffding bound requires at least 1 sample".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(NyError::InvalidSpec(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }

    let width = crown_bounds.width();
    if empirical_mean.len() != width.len() {
        return Err(NyError::ShapeMismatch {
            expected: width.shape().to_vec(),
            got: empirical_mean.shape().to_vec(),
        });
    }

    let n = num_samples as f64;
    let delta = 1.0 - confidence;
    let ln_term = (2.0_f64 / delta).ln();

    let bounds = empirical_mean
        .iter()
        .zip(width.iter())
        .enumerate()
        .map(|(i, (&mean, &range))| {
            let range = range.max(0.0) as f64;
            let epsilon = if range > 0.0 {
                range * (ln_term / (2.0 * n)).sqrt()
            } else {
                0.0
            };
            let failure_prob = if range > 0.0 {
                2.0 * (-2.0 * n * epsilon * epsilon / (range * range)).exp()
            } else {
                0.0
            };
            HoeffdingBound {
                dimension: i,
                empirical_mean: mean,
                bound_range: range as f32,
                epsilon,
                failure_probability: failure_prob,
                num_samples,
            }
        })
        .collect();

    Ok(bounds)
}

/// Compute McDiarmid concentration bounds using per-layer Lipschitz constants.
///
/// Returns an error if `lipschitz_estimate.is_sound` is `false`. Callers that
/// intentionally accept optimistic estimates should use
/// [`mcdiarmid_bound_optimistic`] instead.
///
/// For a Lipschitz-continuous network with global constant L and input perturbation
/// radii delta_j, the bounded difference for output i is c_{ij} = L * delta_j.
///
/// P(|f_i(X) - E[f_i(X)]| >= epsilon) <= 2 * exp(-2*epsilon^2 / sum_j(c_{ij}^2))
///
/// Part of #4145: enforced contract replaces warning-only fallback.
pub fn mcdiarmid_bound(
    empirical_output: &ArrayD<f32>,
    input_bounds: &BoundedTensor,
    lipschitz_estimate: &LipschitzEstimate,
    num_samples: usize,
    confidence: f64,
) -> Result<Vec<McDiarmidBound>> {
    if !lipschitz_estimate.is_sound {
        return Err(NyError::InvalidSpec(format!(
            "McDiarmid bound requires a sound Lipschitz estimate, but estimate has \
             unhandled layers: {:?}. Use mcdiarmid_bound_optimistic() to opt in \
             to optimistic bounds.",
            lipschitz_estimate.unhandled_layers
        )));
    }
    mcdiarmid_bound_inner(
        empirical_output,
        input_bounds,
        lipschitz_estimate,
        num_samples,
        confidence,
    )
}

/// Compute McDiarmid bounds accepting an optimistic (possibly unsound) Lipschitz
/// estimate.
///
/// Emits a `tracing::warn!` when `lipschitz_estimate.is_sound` is `false` and
/// returns bounds with `is_sound: false`. The caller is responsible for
/// communicating the reduced trustworthiness of the resulting certificate.
///
/// Part of #4145: explicit opt-in for unsound path.
pub fn mcdiarmid_bound_optimistic(
    empirical_output: &ArrayD<f32>,
    input_bounds: &BoundedTensor,
    lipschitz_estimate: &LipschitzEstimate,
    num_samples: usize,
    confidence: f64,
) -> Result<Vec<McDiarmidBound>> {
    if !lipschitz_estimate.is_sound {
        tracing::warn!(
            lipschitz = lipschitz_estimate.value,
            unhandled_layers = ?lipschitz_estimate.unhandled_layers,
            "McDiarmid bound uses an optimistic Lipschitz estimate for unhandled layers"
        );
    }
    mcdiarmid_bound_inner(
        empirical_output,
        input_bounds,
        lipschitz_estimate,
        num_samples,
        confidence,
    )
}

/// Shared implementation for McDiarmid bound computation.
fn mcdiarmid_bound_inner(
    empirical_output: &ArrayD<f32>,
    input_bounds: &BoundedTensor,
    lipschitz_estimate: &LipschitzEstimate,
    num_samples: usize,
    confidence: f64,
) -> Result<Vec<McDiarmidBound>> {
    if num_samples == 0 {
        return Err(NyError::InvalidSpec(
            "McDiarmid bound requires at least 1 sample".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(NyError::InvalidSpec(format!(
            "confidence must be in (0, 1), got {confidence}"
        )));
    }
    let lipschitz_constant = lipschitz_estimate.value;
    if lipschitz_constant < 0.0 || !lipschitz_constant.is_finite() {
        return Err(NyError::InvalidSpec(format!(
            "Lipschitz constant must be finite and non-negative, got {lipschitz_constant}"
        )));
    }

    let width = input_bounds.width();
    let l = lipschitz_constant as f64;

    // sum_j(c_ij^2) = L^2 * sum_j(delta_j^2) where delta_j = width_j
    // (McDiarmid uses the max change when input j varies over its full range)
    let sum_delta_sq: f64 = width.iter().map(|&w| (w as f64) * (w as f64)).sum();
    let sum_c_sq = l * l * sum_delta_sq;
    let bd_norm = sum_c_sq.sqrt();

    let n = num_samples as f64;
    let delta = 1.0 - confidence;
    let ln_term = (2.0_f64 / delta).ln();

    let bounds = empirical_output
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let epsilon = if sum_c_sq > 0.0 {
                (sum_c_sq * ln_term / (2.0 * n)).sqrt()
            } else {
                0.0
            };
            let failure_prob = if sum_c_sq > 0.0 {
                2.0 * (-2.0 * n * epsilon * epsilon / sum_c_sq).exp()
            } else {
                0.0
            };
            McDiarmidBound {
                dimension: i,
                empirical_value: val,
                bounded_difference_norm: bd_norm,
                epsilon,
                is_sound: lipschitz_estimate.is_sound,
                failure_probability: failure_prob,
                num_samples,
            }
        })
        .collect();

    Ok(bounds)
}

/// Estimate a global Lipschitz constant from a Network.
///
/// Uses the product of spectral norms of recognized Linear layers as an upper bound.
/// Layers outside the handled allow-list are recorded in the returned metadata so
/// callers can distinguish sound estimates from optimistic ones.
///
/// This is a coarse upper bound — tighter estimates would require per-layer analysis
/// of activation Lipschitz constants, but the product-of-spectral-norms bound is
/// sound and cheap to compute (spectral norms are precomputed at layer construction).
///
/// Reference: Szegedy et al., "Intriguing properties of neural networks", ICLR 2014.
pub fn estimate_lipschitz_from_network(network: &Network) -> Result<LipschitzEstimate> {
    if network.num_layers() == 0 {
        return Err(NyError::InvalidSpec(
            "Cannot estimate Lipschitz constant for empty network".to_string(),
        ));
    }

    let mut lipschitz = 1.0_f32;
    let mut unhandled_layers = Vec::new();
    for layer in network.layers() {
        match layer {
            Layer::Linear(linear) => {
                lipschitz *= linear.spectral_norm();
            }
            // ReLU, LeakyReLU(alpha<=1), Sigmoid, Tanh, etc. are 1-Lipschitz
            Layer::ReLU(_) | Layer::Sigmoid(_) | Layer::Tanh(_) | Layer::Softmax(_) => {}
            // Reshape/Transpose/Flatten don't change Lipschitz constant
            Layer::Reshape(_) | Layer::Transpose(_) | Layer::Flatten(_) => {}
            // For other layers, we conservatively skip (Lipschitz = 1 assumption).
            // This makes the bound less tight but still sound for layers with
            // Lipschitz constant <= 1. For layers with Lipschitz > 1 (e.g. Exp),
            // the bound may not be sound — callers should check network composition.
            _ => {
                let layer_type = layer.layer_type().to_string();
                if !unhandled_layers.contains(&layer_type) {
                    unhandled_layers.push(layer_type);
                }
            }
        }
    }

    if !lipschitz.is_finite() || lipschitz < 0.0 {
        return Err(NyError::InvalidSpec(format!(
            "Lipschitz estimate is not finite/non-negative: {lipschitz}"
        )));
    }

    Ok(LipschitzEstimate {
        value: lipschitz,
        is_sound: unhandled_layers.is_empty(),
        unhandled_layers,
    })
}

#[cfg(test)]
#[path = "concentration_tests.rs"]
mod tests;
