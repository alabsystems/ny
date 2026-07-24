// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Monte Carlo verification with CROWN sound over-approximation.
//!
//! Samples inputs from a specified distribution, evaluates the network on
//! each sample, and computes empirical output statistics. CROWN bounds
//! provide a sound certificate that the empirical bounds must lie within.
//!
//! Part of #3921.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use rand::RngExt;

/// Input distribution specification for Monte Carlo sampling.
#[derive(Debug, Clone)]
pub enum InputDistribution {
    /// Uniform distribution within the perturbation bounds.
    Uniform,
    /// Gaussian with mean at center of bounds, std = (upper - lower) / 6
    /// (99.7% of samples within bounds, clipped to [lower, upper]).
    Gaussian { clip_to_bounds: bool },
}

/// Result of a Monte Carlo verification run.
#[derive(Debug, Clone)]
pub struct ProbabilisticBound {
    /// Empirical lower bound (minimum observed output per dimension).
    pub empirical_lower: ArrayD<f32>,
    /// Empirical upper bound (maximum observed output per dimension).
    pub empirical_upper: ArrayD<f32>,
    /// Mean output per dimension.
    pub empirical_mean: ArrayD<f32>,
    /// Standard deviation per dimension.
    pub empirical_std: ArrayD<f32>,
    /// Number of samples used.
    pub num_samples: usize,
    /// CROWN sound upper bound (if available).
    pub crown_upper: Option<ArrayD<f32>>,
    /// CROWN sound lower bound (if available).
    pub crown_lower: Option<ArrayD<f32>>,
    /// Confidence level (e.g., 0.95 for 95% confidence).
    pub confidence: f64,
}

impl ProbabilisticBound {
    /// Check if the empirical bounds are consistent with CROWN bounds.
    ///
    /// Returns true if empirical_lower >= crown_lower and
    /// empirical_upper <= crown_upper (within tolerance).
    pub fn is_consistent(&self, tolerance: f32) -> bool {
        if let (Some(cl), Some(cu)) = (&self.crown_lower, &self.crown_upper) {
            self.empirical_lower
                .iter()
                .zip(cl.iter())
                .all(|(&emp, &crown)| emp >= crown - tolerance)
                && self
                    .empirical_upper
                    .iter()
                    .zip(cu.iter())
                    .all(|(&emp, &crown)| emp <= crown + tolerance)
        } else {
            true // No CROWN bounds to check against
        }
    }

    /// Estimated bound width at the given confidence level using normal approximation.
    ///
    /// For each dimension: [mean - z * std/sqrt(n), mean + z * std/sqrt(n)]
    pub fn confidence_interval_width(&self) -> ArrayD<f32> {
        let z = z_score(self.confidence);
        let sqrt_n = (self.num_samples as f32).sqrt();
        self.empirical_std.mapv(|s| 2.0 * z as f32 * s / sqrt_n)
    }

    /// Compute Hoeffding concentration certificate from existing Monte Carlo results.
    ///
    /// Requires CROWN bounds to be present (they provide the bounded range [a_i, b_i]).
    /// Returns one `HoeffdingBound` per output dimension.
    ///
    /// Reference: Hoeffding, "Probability Inequalities for Sums of Bounded Random Variables", 1963.
    pub fn hoeffding_certificate(
        &self,
        confidence: f64,
    ) -> Result<Vec<super::concentration::HoeffdingBound>> {
        let (crown_lower, crown_upper) = match (&self.crown_lower, &self.crown_upper) {
            (Some(l), Some(u)) => (l, u),
            _ => {
                return Err(NyError::InvalidSpec(
                    "Hoeffding certificate requires CROWN bounds".to_string(),
                ))
            }
        };

        let crown_bounds = BoundedTensor::new(crown_lower.clone(), crown_upper.clone())?;
        super::concentration::hoeffding_bound(
            &self.empirical_mean,
            &crown_bounds,
            self.num_samples,
            confidence,
        )
    }
}

/// Monte Carlo verifier for probabilistic bound estimation.
pub struct MonteCarloVerifier {
    /// Number of Monte Carlo samples.
    pub num_samples: usize,
    /// Input distribution type.
    pub distribution: InputDistribution,
    /// Confidence level for statistical bounds.
    pub confidence: f64,
    /// Random seed for reproducibility (None = random).
    pub seed: Option<u64>,
}

impl MonteCarloVerifier {
    /// Create a new Monte Carlo verifier with default settings.
    pub fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            distribution: InputDistribution::Uniform,
            confidence: 0.95,
            seed: None,
        }
    }

    /// Set the input distribution.
    pub fn with_distribution(mut self, dist: InputDistribution) -> Self {
        self.distribution = dist;
        self
    }

    /// Set the confidence level.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sample inputs from the distribution within the given bounds.
    ///
    /// Returns a vector of concrete input arrays, each within [lower, upper].
    pub fn sample_inputs(&self, input_bounds: &BoundedTensor) -> Result<Vec<ArrayD<f32>>> {
        let lower = input_bounds.lower();
        let upper = input_bounds.upper();
        let shape = input_bounds.shape().to_vec();

        let mut rng: Box<dyn rand::Rng> = match self.seed {
            Some(seed) => {
                use rand::SeedableRng;
                Box::new(rand::rngs::StdRng::seed_from_u64(seed))
            }
            None => Box::new(rand::rng()),
        };

        let mut samples = Vec::with_capacity(self.num_samples);

        for _ in 0..self.num_samples {
            let data: Vec<f32> = match &self.distribution {
                InputDistribution::Uniform => lower
                    .iter()
                    .zip(upper.iter())
                    .map(|(&l, &u)| if l >= u { l } else { rng.random_range(l..=u) })
                    .collect(),
                InputDistribution::Gaussian { clip_to_bounds } => {
                    lower
                        .iter()
                        .zip(upper.iter())
                        .map(|(&l, &u)| {
                            // `(l+u)/2` kept verbatim: `f32::midpoint` differs once
                            // |bound| > f32::MAX/2, and samples must stay bit-reproducible.
                            #[allow(clippy::manual_midpoint)]
                            let center = (l + u) / 2.0;
                            let std = (u - l) / 6.0; // 99.7% within bounds
                            if std <= 0.0 {
                                return center;
                            }
                            // Box-Muller transform for normal sampling
                            let u1: f32 = rng.random_range(f32::EPSILON..=1.0);
                            let u2: f32 = rng.random_range(0.0..=std::f32::consts::TAU);
                            let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
                            let sample: f32 = center + std * z;
                            if *clip_to_bounds {
                                sample.clamp(l, u)
                            } else {
                                sample
                            }
                        })
                        .collect()
                }
            };

            let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| NyError::InvalidSpec(format!("Monte Carlo sample error: {e}")))?;
            samples.push(arr);
        }

        Ok(samples)
    }

    /// Run Monte Carlo verification given pre-computed network outputs.
    ///
    /// `outputs` should be the network output for each sampled input.
    /// `crown_bounds` optionally provides sound CROWN bounds for validation.
    pub fn compute_bounds(
        &self,
        outputs: &[ArrayD<f32>],
        crown_bounds: Option<&BoundedTensor>,
    ) -> Result<ProbabilisticBound> {
        if outputs.is_empty() {
            return Err(NyError::InvalidSpec(
                "MonteCarloVerifier: no outputs provided".to_string(),
            ));
        }

        let shape = outputs[0].shape().to_vec();
        let n_elems = outputs[0].len();
        let n = outputs.len() as f32;

        // Compute per-element statistics
        let mut min_vals = vec![f32::INFINITY; n_elems];
        let mut max_vals = vec![f32::NEG_INFINITY; n_elems];
        let mut sum_vals = vec![0.0f64; n_elems];
        let mut sum_sq_vals = vec![0.0f64; n_elems];

        for output in outputs {
            if output.shape() != shape.as_slice() {
                return Err(NyError::ShapeMismatch {
                    expected: shape,
                    got: output.shape().to_vec(),
                });
            }
            for (j, &v) in output.iter().enumerate() {
                min_vals[j] = min_vals[j].min(v);
                max_vals[j] = max_vals[j].max(v);
                sum_vals[j] += v as f64;
                sum_sq_vals[j] += (v as f64) * (v as f64);
            }
        }

        let mean_vals: Vec<f32> = sum_vals.iter().map(|&s| (s / n as f64) as f32).collect();
        let std_vals: Vec<f32> = sum_vals
            .iter()
            .zip(sum_sq_vals.iter())
            .map(|(&s, &sq)| {
                let mean = s / n as f64;
                let variance = sq / n as f64 - mean * mean;
                variance.max(0.0).sqrt() as f32
            })
            .collect();

        Ok(ProbabilisticBound {
            empirical_lower: ArrayD::from_shape_vec(IxDyn(&shape), min_vals)
                .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
            empirical_upper: ArrayD::from_shape_vec(IxDyn(&shape), max_vals)
                .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
            empirical_mean: ArrayD::from_shape_vec(IxDyn(&shape), mean_vals)
                .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
            empirical_std: ArrayD::from_shape_vec(IxDyn(&shape), std_vals)
                .map_err(|e| NyError::InvalidSpec(format!("shape error: {e}")))?,
            num_samples: outputs.len(),
            crown_lower: crown_bounds.map(|cb| cb.lower().clone()),
            crown_upper: crown_bounds.map(|cb| cb.upper().clone()),
            confidence: self.confidence,
        })
    }
}

/// Z-score for a given confidence level (normal approximation).
pub(super) fn z_score(confidence: f64) -> f64 {
    // Common z-scores for standard confidence levels
    if (confidence - 0.90).abs() < 1e-6 {
        1.645
    } else if (confidence - 0.95).abs() < 1e-6 {
        1.960
    } else if (confidence - 0.99).abs() < 1e-6 {
        2.576
    } else if (confidence - 0.999).abs() < 1e-6 {
        3.291
    } else {
        // Fallback: use the probit approximation (Abramowitz & Stegun 26.2.23)
        let p = f64::midpoint(1.0, confidence);
        let t = (-2.0 * (1.0 - p).ln()).sqrt();
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;
        t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    }
}

/// Standard normal CDF: Phi(x) = P(Z <= x) for Z ~ N(0,1).
///
/// Uses Abramowitz & Stegun approximation 26.2.17 (max error 7.5e-8).
pub(super) fn gaussian_cdf(x: f64) -> f64 {
    if x >= 0.0 {
        let t = 1.0 / (1.0 + 0.2316419 * x);
        let p = 0.3989422804014327 * (-x * x / 2.0).exp(); // phi(x)
        let poly = ((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t - 0.356563782) * t
            + 0.319381530)
            * t;
        1.0 - p * poly
    } else {
        1.0 - gaussian_cdf(-x)
    }
}

/// Standard normal PDF: phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2).
pub(super) fn gaussian_pdf(x: f64) -> f64 {
    0.3989422804014327 * (-x * x / 2.0).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;

    #[test]
    fn test_monte_carlo_uniform_sampling() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 0.5]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 1.5]).unwrap();
        let bounds = BoundedTensor::new(lower, upper).unwrap();

        let verifier = MonteCarloVerifier::new(100).with_seed(42);
        let samples = verifier.sample_inputs(&bounds).unwrap();

        assert_eq!(samples.len(), 100);

        // All samples should be within bounds
        for sample in &samples {
            for ((&s, &l), &u) in sample
                .iter()
                .zip(bounds.lower().iter())
                .zip(bounds.upper().iter())
            {
                assert!(s >= l, "sample {} < lower {}", s, l);
                assert!(s <= u, "sample {} > upper {}", s, u);
            }
        }
    }

    #[test]
    fn test_monte_carlo_compute_bounds() {
        let verifier = MonteCarloVerifier::new(1000).with_seed(42);

        // Simulate outputs: uniform in [0, 1] for 2 dimensions
        let outputs: Vec<ArrayD<f32>> = (0..1000)
            .map(|i| {
                let v1 = (i as f32) / 999.0;
                let v2 = 1.0 - v1;
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![v1, v2]).unwrap()
            })
            .collect();

        let result = verifier.compute_bounds(&outputs, None).unwrap();

        assert_eq!(result.num_samples, 1000);
        // min should be close to 0, max close to 1
        assert!(result.empirical_lower[[0]] < 0.01);
        assert!(result.empirical_upper[[0]] > 0.99);
        // Mean should be close to 0.5
        assert!((result.empirical_mean[[0]] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_crown_consistency_check() {
        let bound = ProbabilisticBound {
            empirical_lower: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).unwrap(),
            empirical_upper: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 0.9]).unwrap(),
            empirical_mean: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap(),
            empirical_std: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.1]).unwrap(),
            num_samples: 1000,
            crown_lower: Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap()),
            crown_upper: Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap()),
            confidence: 0.95,
        };

        // Empirical bounds [0.1, 0.8] within CROWN bounds [0.0, 1.0]
        assert!(bound.is_consistent(0.0));

        // Inconsistent: empirical exceeds CROWN
        let bad_bound = ProbabilisticBound {
            crown_upper: Some(ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap()),
            ..bound
        };
        assert!(!bad_bound.is_consistent(0.0));
    }

    #[test]
    fn test_hoeffding_certificate_matches_concentration_module() {
        let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, -1.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 3.0]).unwrap();
        let crown_bounds = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

        let bound = ProbabilisticBound {
            empirical_lower: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).unwrap(),
            empirical_upper: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.8, 2.1]).unwrap(),
            empirical_mean: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 1.0]).unwrap(),
            empirical_std: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.3]).unwrap(),
            num_samples: 512,
            crown_lower: Some(lower),
            crown_upper: Some(upper),
            confidence: 0.95,
        };

        let delegated = bound.hoeffding_certificate(0.95).unwrap();
        let direct = super::super::concentration::hoeffding_bound(
            &bound.empirical_mean,
            &crown_bounds,
            bound.num_samples,
            0.95,
        )
        .unwrap();

        assert_eq!(delegated.len(), direct.len());
        for (actual, expected) in delegated.iter().zip(direct.iter()) {
            assert_eq!(actual.dimension, expected.dimension);
            assert_eq!(actual.empirical_mean, expected.empirical_mean);
            assert_eq!(actual.bound_range, expected.bound_range);
            assert!((actual.epsilon - expected.epsilon).abs() < 1e-12);
            assert!((actual.failure_probability - expected.failure_probability).abs() < 1e-12);
            assert_eq!(actual.num_samples, expected.num_samples);
        }
    }
}
