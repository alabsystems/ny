// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for higher-order moment propagation and Cornish-Fisher expansion.
//! Part of #3921 / #4249.

use super::*;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;
use crate::probabilistic::distributional::AnalyticDistribution;

fn identity_lb(dim: usize) -> LinearBounds {
    LinearBounds::identity(dim)
}

fn scaling_lb(dim: usize, scale: f32) -> LinearBounds {
    let a = Array2::from_diag(&Array1::from_elem(dim, scale));
    let b = Array1::zeros(dim);
    LinearBounds::new(a.clone(), b.clone(), a, b).unwrap()
}

#[test]
fn test_uniform_excess_kurtosis_negative() {
    // Uniform inputs: excess kurtosis should be -1.2
    let dim = 2;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let lb = identity_lb(dim);

    let mb = propagate_moments(&lb, &dist, &bounds, 0.95).unwrap();

    for i in 0..dim {
        let kappa = mb.excess_kurtosis[[i]];
        assert!(
            (kappa - (-1.2)).abs() < 0.01,
            "excess kurtosis should be -1.2 for uniform, got {kappa}"
        );
    }
}

#[test]
fn test_gaussian_excess_kurtosis_zero() {
    // Gaussian inputs: excess kurtosis should be 0
    let dim = 2;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![-3.0, -3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![3.0, 3.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap()),
        variance: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 1.0]).unwrap()),
    };
    let lb = identity_lb(dim);

    let mb = propagate_moments(&lb, &dist, &bounds, 0.95).unwrap();

    for i in 0..dim {
        let kappa = mb.excess_kurtosis[[i]];
        assert!(
            kappa.abs() < 0.01,
            "excess kurtosis should be 0 for Gaussian, got {kappa}"
        );
    }
}

#[test]
fn test_cornish_fisher_tighter_than_gaussian_for_uniform() {
    // For uniform inputs (kurtosis < 0), CF bounds should be inside Gaussian bounds
    let dim = 2;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let lb = identity_lb(dim);

    let mb = propagate_moments(&lb, &dist, &bounds, 0.99).unwrap();

    for i in 0..dim {
        let cf_lower = mb.prob_lower[[i]];
        let g_lower = mb.prob_lower_gaussian[[i]];
        let cf_upper = mb.prob_upper[[i]];
        let g_upper = mb.prob_upper_gaussian[[i]];

        // CF lower bound should be >= Gaussian lower bound (tighter = higher)
        assert!(
            cf_lower >= g_lower - 1e-6,
            "CF lower {cf_lower} should be >= Gaussian lower {g_lower}"
        );
        // CF upper bound should be <= Gaussian upper bound (tighter = lower)
        assert!(
            cf_upper <= g_upper + 1e-6,
            "CF upper {cf_upper} should be <= Gaussian upper {g_upper}"
        );
    }
}

#[test]
fn test_cornish_fisher_matches_gaussian_for_gaussian_input() {
    // For Gaussian inputs (kurtosis=0, skewness=0), CF should equal Gaussian
    let dim = 1;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![-5.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![5.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap()),
        variance: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0]).unwrap()),
    };
    let lb = identity_lb(dim);

    let mb = propagate_moments(&lb, &dist, &bounds, 0.95).unwrap();

    let cf_lower = mb.prob_lower[[0]];
    let g_lower = mb.prob_lower_gaussian[[0]];
    assert!(
        (cf_lower - g_lower).abs() < 1e-4,
        "CF and Gaussian should match for Gaussian input: CF={cf_lower}, G={g_lower}"
    );
}

#[test]
fn test_scaling_preserves_kurtosis() {
    // Scaling Y = k*X: excess kurtosis is preserved
    let dim = 1;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;

    let mb_id = propagate_moments(&identity_lb(dim), &dist, &bounds, 0.95).unwrap();
    let mb_sc = propagate_moments(&scaling_lb(dim, 3.0), &dist, &bounds, 0.95).unwrap();

    let k_id = mb_id.excess_kurtosis[[0]];
    let k_sc = mb_sc.excess_kurtosis[[0]];
    assert!(
        (k_id - k_sc).abs() < 0.01,
        "kurtosis should be scale-invariant: identity={k_id}, scaled={k_sc}"
    );
}

#[test]
fn test_cornish_fisher_w_symmetric() {
    // For gamma1=0 (symmetric), CF simplifies to w = z + (z^3-3z)*gamma2/24
    let z = 2.576; // 99% z-score
    let gamma2 = -1.2; // uniform excess kurtosis
    let w = cornish_fisher_w(z, 0.0, gamma2);

    // Expected: 2.576 + (17.09 - 7.73) * (-1.2)/24 = 2.576 - 0.468 = 2.108
    assert!(w < z, "CF w={w} should be < z={z} for negative kurtosis");
    assert!((w - 2.108).abs() < 0.01, "CF w should be ~2.108, got {w}");

    // Tightening ratio at 99%: ~18%
    let tightening = 1.0 - w / z;
    assert!(
        tightening > 0.15,
        "should be >15% tighter at 99%, got {:.1}%",
        tightening * 100.0
    );
}

#[test]
fn test_cornish_fisher_w_heavy_tail_diverges() {
    // For positive excess kurtosis (heavy tails / leptokurtic), CF w > z,
    // meaning the expansion produces WIDER bounds than Gaussian.
    // This is the scenario where the clamp in propagate_moments (lines 198-199)
    // prevents CF from being worse than the Gaussian baseline.
    let z = 2.576; // 99% z-score

    // gamma2 = +6.0 (heavy-tailed, e.g. t-distribution with low df)
    let w = cornish_fisher_w(z, 0.0, 6.0);
    assert!(
        w > z,
        "CF w={w} should be > z={z} for large positive kurtosis"
    );

    // gamma2 = +20.0 (extremely heavy-tailed)
    let w_extreme = cornish_fisher_w(z, 0.0, 20.0);
    assert!(
        w_extreme > w,
        "heavier tails should produce even larger w: {w_extreme} > {w}"
    );

    // With skewness too (asymmetric heavy tails)
    let w_skewed = cornish_fisher_w(z, 2.0, 6.0);
    assert!(
        w_skewed > z,
        "skewed heavy-tailed CF w={w_skewed} should exceed z={z}"
    );
}

#[test]
fn test_cf_bounds_never_worse_than_gaussian() {
    // Verify the clamp invariant: for all distributions and confidence levels,
    // prob_lower >= prob_lower_gaussian and prob_upper <= prob_upper_gaussian.
    // This ensures the CF expansion never produces bounds worse than Gaussian.
    let dim = 3;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![-2.0, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![2.0, 4.0, 5.0]).unwrap(),
    )
    .unwrap();

    let distributions = vec![
        AnalyticDistribution::UniformFromBounds,
        AnalyticDistribution::DiagonalGaussian {
            mean: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 2.0, 3.0]).unwrap()),
            variance: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 0.5, 2.0]).unwrap()),
        },
    ];

    let confidence_levels = [0.5, 0.9, 0.95, 0.99];

    // Test with identity and non-trivial linear bounds
    let a = Array2::from_shape_vec((2, dim), vec![1.0, -0.5, 0.3, 0.0, 2.0, -1.0]).unwrap();
    let b = Array1::from_vec(vec![0.1, -0.2]);
    let mixed_lb = LinearBounds::new(a.clone(), b.clone(), a, b).unwrap();

    for dist in &distributions {
        for &conf in &confidence_levels {
            for lb in &[identity_lb(dim), mixed_lb.clone()] {
                let mb = propagate_moments(lb, dist, &bounds, conf).unwrap();
                let n = mb.prob_lower.len();
                for i in 0..n {
                    assert!(
                        mb.prob_lower[[i]] >= mb.prob_lower_gaussian[[i]] - 1e-6,
                        "CF lower {:.6} < Gaussian lower {:.6} at conf={conf}, dim={i}",
                        mb.prob_lower[[i]],
                        mb.prob_lower_gaussian[[i]]
                    );
                    assert!(
                        mb.prob_upper[[i]] <= mb.prob_upper_gaussian[[i]] + 1e-6,
                        "CF upper {:.6} > Gaussian upper {:.6} at conf={conf}, dim={i}",
                        mb.prob_upper[[i]],
                        mb.prob_upper_gaussian[[i]]
                    );
                }
            }
        }
    }
}

#[test]
fn test_zero_variance_gives_tight_moment_bounds() {
    // When all dims have zero width, variance=0, bounds collapse to mean
    let dim = 2;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let lb = identity_lb(dim);

    let mb = propagate_moments(&lb, &dist, &bounds, 0.95).unwrap();

    for i in 0..dim {
        assert!((mb.variance_upper[[i]]).abs() < 1e-10);
        assert!((mb.prob_lower[[i]] - mb.mean_lower[[i]]).abs() < 1e-6);
        assert!((mb.prob_upper[[i]] - mb.mean_upper[[i]]).abs() < 1e-6);
    }
}
