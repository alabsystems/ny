// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Branch-and-bound refinement for distributional bounds.
//!
//! Splits the input domain into subregions, computes CROWN + distributional
//! bounds per subregion, and combines them using the law of total variance.
//! Enhanced with probability weighting, exact conditional moments, and
//! distributional gradient-guided splitting (#4249).
//!
//! Reference: Boetius, Leue, Sutter, "Branch-and-Bound for Probabilistic
//! Verification of Neural Networks", ICML 2025.
//!
//! Part of #3921 / #4249.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

use super::distributional::{propagate_distribution, AnalyticDistribution, DistributionalBound};
use super::monte_carlo::{gaussian_cdf, gaussian_pdf, z_score};
use crate::bounds::LinearBounds;

/// Strategy for choosing which dimension to split.
#[derive(Debug, Clone, Default)]
pub enum SplitStrategy {
    /// Split the widest input dimension (original Boetius et al.).
    #[default]
    WidestDimension,
    /// Split the dimension that maximizes estimated probabilistic bound improvement.
    DistributionalGradient,
}

/// Configuration for branch-and-bound distributional bound refinement.
#[derive(Debug, Clone)]
pub struct BranchAndBoundConfig {
    pub max_iterations: usize,
    pub max_regions: usize,
    pub tolerance: f64,
    pub confidence: f64,
    /// Weight B&B priority by P(X in region). Default: false.
    pub use_probability_weighting: bool,
    /// Use exact truncated Gaussian conditional means. Default: false.
    pub use_exact_conditional_moments: bool,
    pub split_strategy: SplitStrategy,
}

impl Default for BranchAndBoundConfig {
    fn default() -> Self {
        Self {
            max_iterations: 16,
            max_regions: 64,
            tolerance: 0.01,
            confidence: 0.99,
            use_probability_weighting: false,
            use_exact_conditional_moments: false,
            split_strategy: SplitStrategy::default(),
        }
    }
}

/// Result of branch-and-bound refinement.
#[derive(Debug, Clone)]
pub struct BranchAndBoundResult {
    pub bound: DistributionalBound,
    pub num_regions: usize,
    pub iterations: usize,
    pub max_variance_upper: f32,
    pub converged: bool,
}

/// A subregion in the branch-and-bound tree.
#[derive(Debug, Clone)]
struct Region {
    bounds: BoundedTensor,
    volume_weight: f64,
    dist_bound: DistributionalBound,
    max_weighted_variance: f64,
}

impl PartialEq for Region {
    fn eq(&self, other: &Self) -> bool {
        // Must be consistent with Ord::cmp which uses total_cmp (#4288).
        // f64 == returns false for NaN == NaN, but total_cmp returns Equal.
        self.max_weighted_variance
            .total_cmp(&other.max_weighted_variance)
            == Ordering::Equal
    }
}
impl Eq for Region {}
impl PartialOrd for Region {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Region {
    fn cmp(&self, other: &Self) -> Ordering {
        // Use total_cmp for NaN safety: NaN sorts after all finite values,
        // so NaN-variance regions get highest priority in the max-heap.
        // This prevents BinaryHeap corruption when upstream moments produce NaN (#4288).
        self.max_weighted_variance
            .total_cmp(&other.max_weighted_variance)
    }
}

/// Refine distributional bounds via branch-and-bound over input subregions.
pub fn refine_distributional_bounds<F>(
    crown_fn: &F,
    input_bounds: &BoundedTensor,
    input_dist: &AnalyticDistribution,
    config: &BranchAndBoundConfig,
) -> Result<BranchAndBoundResult>
where
    F: Fn(&BoundedTensor) -> Result<LinearBounds>,
{
    if config.max_iterations == 0 || config.max_regions == 0 {
        return Err(NyError::InvalidSpec(
            "max_iterations and max_regions must be > 0".to_string(),
        ));
    }

    let initial_lb = crown_fn(input_bounds)?;
    let initial_bound =
        propagate_distribution(&initial_lb, input_dist, input_bounds, config.confidence)?;
    let max_var = max_element(&initial_bound.variance_upper);

    let initial_weight = if config.use_probability_weighting {
        region_probability_mass(input_bounds, input_dist)
    } else {
        1.0
    };

    let mut crown_cache: Vec<(BoundedTensor, LinearBounds)> =
        vec![(input_bounds.clone(), initial_lb)];
    let mut heap: BinaryHeap<Region> = BinaryHeap::new();
    heap.push(Region {
        bounds: input_bounds.clone(),
        volume_weight: initial_weight,
        dist_bound: initial_bound.clone(),
        max_weighted_variance: initial_weight * max_var as f64,
    });

    let mut best = initial_bound;
    let mut best_max_var = max_var;
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..config.max_iterations {
        if heap.len() >= config.max_regions {
            break;
        }
        let worst = match heap.pop() {
            Some(r) => r,
            None => break,
        };

        let split_dim = pick_split_dim(&worst, config, crown_fn, &mut crown_cache)?;
        let (left, right) = bisect_region(&worst.bounds, split_dim)?;

        for sub_bounds in [&left, &right] {
            let sub_lb = crown_fn(sub_bounds)?;
            let sub_dist =
                propagate_distribution(&sub_lb, input_dist, sub_bounds, config.confidence)?;
            let sub_max_var = max_element(&sub_dist.variance_upper);
            let sub_weight = if config.use_probability_weighting {
                region_probability_mass(sub_bounds, input_dist)
            } else {
                worst.volume_weight / 2.0
            };
            if matches!(config.split_strategy, SplitStrategy::DistributionalGradient) {
                crown_cache.push((sub_bounds.clone(), sub_lb));
            }
            heap.push(Region {
                bounds: sub_bounds.clone(),
                volume_weight: sub_weight,
                dist_bound: sub_dist,
                max_weighted_variance: sub_weight * sub_max_var as f64,
            });
        }

        let regions: Vec<&Region> = heap.iter().collect();
        let combined = if config.use_probability_weighting {
            combine_weighted(&regions, input_dist, config)?
        } else {
            combine_volume(&regions, config.confidence)?
        };
        let combined_max_var = max_element(&combined.variance_upper);
        iterations += 1;

        let improvement = if best_max_var > 0.0 {
            ((best_max_var - combined_max_var) / best_max_var) as f64
        } else {
            0.0
        };
        best = combined;
        best_max_var = combined_max_var;

        if improvement < config.tolerance && improvement >= 0.0 {
            converged = true;
            break;
        }
    }

    Ok(BranchAndBoundResult {
        bound: best,
        num_regions: heap.len(),
        iterations,
        max_variance_upper: best_max_var,
        converged,
    })
}

/// Pick split dimension based on strategy.
fn pick_split_dim<F>(
    worst: &Region,
    config: &BranchAndBoundConfig,
    crown_fn: &F,
    crown_cache: &mut Vec<(BoundedTensor, LinearBounds)>,
) -> Result<usize>
where
    F: Fn(&BoundedTensor) -> Result<LinearBounds>,
{
    match config.split_strategy {
        SplitStrategy::WidestDimension => Ok(choose_split_widest(&worst.bounds)),
        SplitStrategy::DistributionalGradient => {
            let cached = crown_cache.iter().find(|(b, _)| {
                b.lower() == worst.bounds.lower() && b.upper() == worst.bounds.upper()
            });
            match cached {
                Some((_, lb)) => Ok(choose_split_gradient(&worst.bounds, lb, &worst.dist_bound)),
                None => {
                    let lb = crown_fn(&worst.bounds)?;
                    let dim = choose_split_gradient(&worst.bounds, &lb, &worst.dist_bound);
                    crown_cache.push((worst.bounds.clone(), lb));
                    Ok(dim)
                }
            }
        }
    }
}

/// Split the widest input dimension.
fn choose_split_widest(bounds: &BoundedTensor) -> usize {
    let width = bounds.width();
    let mut best_dim = 0;
    let mut best_w = f32::NEG_INFINITY;
    for (j, &w) in width.iter().enumerate() {
        if w > best_w {
            best_w = w;
            best_dim = j;
        }
    }
    best_dim
}

/// Split the dimension with highest variance-sensitivity score.
fn choose_split_gradient(
    bounds: &BoundedTensor,
    crown_bounds: &LinearBounds,
    dist_bound: &DistributionalBound,
) -> usize {
    let width = bounds.width();
    let a_l = crown_bounds.lower_a();
    let mut best_dim = 0;
    let mut best_score = f64::NEG_INFINITY;

    for j in 0..crown_bounds.num_inputs() {
        let w_j = width[[j]] as f64;
        let var_j = w_j * w_j / 12.0;
        let mut score = 0.0_f64;
        for i in 0..crown_bounds.num_outputs() {
            let sigma_i = (dist_bound.variance_upper[[i]] as f64).max(1e-30).sqrt();
            let a_ij = a_l[[i, j]] as f64;
            score += a_ij * a_ij * var_j / sigma_i;
        }
        if score > best_score {
            best_score = score;
            best_dim = j;
        }
    }
    best_dim
}

/// Bisect a BoundedTensor along dimension `dim` into two halves.
fn bisect_region(bounds: &BoundedTensor, dim: usize) -> Result<(BoundedTensor, BoundedTensor)> {
    let lower = bounds.lower();
    let upper = bounds.upper();
    if dim >= lower.len() {
        return Err(NyError::InvalidSpec(format!(
            "split dimension {dim} >= input size {}",
            lower.len()
        )));
    }
    let mut left_upper: Vec<f32> = upper.iter().copied().collect();
    let mut right_lower: Vec<f32> = lower.iter().copied().collect();
    // `(l+u)/2` kept verbatim: `f32::midpoint` differs once |bound| > f32::MAX/2 (sum
    // overflow), and the split point decides the exact child regions B&B integrates.
    #[allow(clippy::manual_midpoint)]
    let mid = (lower[[dim]] + upper[[dim]]) / 2.0;
    left_upper[dim] = mid;
    right_lower[dim] = mid;
    let shape = bounds.shape().to_vec();
    let mk = |v| {
        ArrayD::from_shape_vec(IxDyn(&shape), v)
            .map_err(|e| NyError::InvalidSpec(format!("bisect: {e}")))
    };
    Ok((
        BoundedTensor::new(lower.clone(), mk(left_upper)?)?,
        BoundedTensor::new(mk(right_lower)?, upper.clone())?,
    ))
}

/// Combine region bounds using volume weights (standard B&B).
fn combine_volume(regions: &[&Region], confidence: f64) -> Result<DistributionalBound> {
    let weights: Vec<f64> = regions.iter().map(|r| r.volume_weight).collect();
    combine_core(regions, &weights, confidence)
}

/// Combine region bounds using probability weights + optional exact moments.
fn combine_weighted(
    regions: &[&Region],
    input_dist: &AnalyticDistribution,
    config: &BranchAndBoundConfig,
) -> Result<DistributionalBound> {
    let weights: Vec<f64> = regions
        .iter()
        .map(|r| region_probability_mass(&r.bounds, input_dist))
        .collect();

    // Compute and log exact conditional means when enabled for Gaussian inputs.
    // These are the E[X_j | X_j in region] values from the truncated Gaussian.
    // Currently used for observability; full CROWN re-propagation with exact
    // conditional means would require passing them into the B&B loop at region
    // construction time (not in combine_weighted).
    if config.use_exact_conditional_moments {
        if let AnalyticDistribution::DiagonalGaussian { mean, variance } = input_dist {
            for (k, r) in regions.iter().enumerate() {
                let lb = r.bounds.lower();
                let ub = r.bounds.upper();
                let mut max_shift = 0.0_f64;
                for j in 0..mean.len() {
                    let mu_j = mean[[j]] as f64;
                    let sigma_j = (variance[[j]] as f64).max(1e-30).sqrt();
                    let l_j = lb[[j]] as f64;
                    let u_j = ub[[j]] as f64;
                    let exact = truncated_gaussian_mean(mu_j, sigma_j, l_j, u_j);
                    let shift = (exact - mu_j).abs();
                    if shift > max_shift {
                        max_shift = shift;
                    }
                }
                tracing::debug!(
                    region = k,
                    max_conditional_mean_shift = max_shift,
                    "exact conditional moment shift"
                );
            }
        }
    }

    combine_core(regions, &weights, config.confidence)
}

/// Core combination logic shared by volume and probability weighting.
fn combine_core(
    regions: &[&Region],
    weights: &[f64],
    confidence: f64,
) -> Result<DistributionalBound> {
    if regions.is_empty() {
        return Err(NyError::InvalidSpec("No regions to combine".to_string()));
    }
    let n = regions[0].dist_bound.num_outputs();
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Err(NyError::InvalidSpec("Total weight <= 0".to_string()));
    }

    let mut ml = vec![0.0_f64; n];
    let mut mu = vec![0.0_f64; n];
    for (k, r) in regions.iter().enumerate() {
        let w = weights[k];
        for i in 0..n {
            ml[i] += w * r.dist_bound.mean_lower[[i]] as f64;
            mu[i] += w * r.dist_bound.mean_upper[[i]] as f64;
        }
    }
    let cml: Vec<f32> = ml.iter().map(|&s| (s / total) as f32).collect();
    let cmu: Vec<f32> = mu.iter().map(|&s| (s / total) as f32).collect();

    let mut cv = vec![0.0_f64; n];
    for (k, r) in regions.iter().enumerate() {
        let w = weights[k];
        for i in 0..n {
            let vk = r.dist_bound.variance_upper[[i]] as f64;
            let mk_val = r.dist_bound.mean_upper[[i]] as f64;
            let d = mk_val - cmu[i] as f64;
            cv[i] += w * (vk + d * d);
        }
    }

    let var: Vec<f32> = cv.iter().map(|&v| (v / total) as f32).collect();

    let z = z_score(confidence) as f32;
    let pl: Vec<f32> = cml
        .iter()
        .zip(var.iter())
        .map(|(&m, &v): (&f32, &f32)| m - z * v.max(0.0).sqrt())
        .collect();
    let pu: Vec<f32> = cmu
        .iter()
        .zip(var.iter())
        .map(|(&m, &v): (&f32, &f32)| m + z * v.max(0.0).sqrt())
        .collect();

    let ix = IxDyn(&[n]);
    let mk = |v: Vec<f32>| {
        ArrayD::from_shape_vec(ix.clone(), v)
            .map_err(|e| NyError::InvalidSpec(format!("shape: {e}")))
    };
    Ok(DistributionalBound {
        mean_lower: mk(cml)?,
        mean_upper: mk(cmu)?,
        variance_upper: mk(var)?,
        prob_lower: mk(pl)?,
        prob_upper: mk(pu)?,
        confidence,
    })
}

/// P(X ∈ region) for DiagonalGaussian; 1.0 for uniform.
fn region_probability_mass(bounds: &BoundedTensor, input_dist: &AnalyticDistribution) -> f64 {
    match input_dist {
        AnalyticDistribution::DiagonalGaussian { mean, variance } => {
            let (lower, upper) = (bounds.lower(), bounds.upper());
            (0..lower.len()).fold(1.0_f64, |acc, j| {
                let mu_j = mean[[j]] as f64;
                let sigma_j = (variance[[j]] as f64).max(1e-30).sqrt();
                let cdf_u = gaussian_cdf((upper[[j]] as f64 - mu_j) / sigma_j);
                let cdf_l = gaussian_cdf((lower[[j]] as f64 - mu_j) / sigma_j);
                acc * (cdf_u - cdf_l).max(0.0)
            })
        }
        AnalyticDistribution::UniformFromBounds => 1.0,
    }
}

/// Exact truncated Gaussian conditional mean: E[X_j | l_j <= X_j <= u_j].
pub(crate) fn truncated_gaussian_mean(mu: f64, sigma: f64, lower: f64, upper: f64) -> f64 {
    if sigma <= 1e-30 {
        return mu;
    }
    let a = (lower - mu) / sigma;
    let b = (upper - mu) / sigma;
    let cdf_diff = gaussian_cdf(b) - gaussian_cdf(a);
    if cdf_diff < 1e-15 {
        // Operands are f32-derived f64 (|x| <= 3.4e38 << f64::MAX/2): identical to (l+u)/2.
        return f64::midpoint(lower, upper);
    }
    mu + sigma * (gaussian_pdf(a) - gaussian_pdf(b)) / cdf_diff
}

/// Find the maximum element, propagating NaN (unlike `f32::max` which discards NaN). (#4288)
fn max_element(arr: &ArrayD<f32>) -> f32 {
    arr.iter().copied().fold(f32::NEG_INFINITY, |a, x| {
        if a.is_nan() || x.is_nan() {
            f32::NAN
        } else {
            a.max(x)
        }
    })
}

#[cfg(test)]
#[path = "branch_and_bound_tests.rs"]
mod tests;
