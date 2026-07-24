// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for distributional propagation through CROWN linear bounds.
//! Part of #3921 Phase 3.

use super::*;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

fn make_identity_linear_bounds(dim: usize) -> LinearBounds {
    LinearBounds::identity(dim)
}

fn make_scaling_linear_bounds(dim: usize, scale: f32) -> LinearBounds {
    let a = Array2::from_diag(&Array1::from_elem(dim, scale));
    let b = Array1::zeros(dim);
    LinearBounds::new(a.clone(), b.clone(), a, b).unwrap()
}

#[test]
fn test_identity_uniform_propagation() {
    // Identity LinearBounds + uniform input → output distribution = input distribution
    let dim = 3;
    let lb = make_identity_linear_bounds(dim);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap();

    let dist = AnalyticDistribution::UniformFromBounds;
    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    // Mean bounds should equal input center (identity bounds)
    let center = input_bounds.center();
    for i in 0..dim {
        assert!(
            (result.mean_lower[[i]] - center[[i]]).abs() < 1e-5,
            "mean_lower[{i}]={} != center={}",
            result.mean_lower[[i]],
            center[[i]]
        );
        assert!(
            (result.mean_upper[[i]] - center[[i]]).abs() < 1e-5,
            "mean_upper[{i}]={} != center={}",
            result.mean_upper[[i]],
            center[[i]]
        );
    }

    // Variance should be width^2 / 12
    let width = input_bounds.width();
    for i in 0..dim {
        let expected_var = width[[i]] * width[[i]] / 12.0;
        assert!(
            (result.variance_upper[[i]] - expected_var).abs() < 1e-5,
            "var[{i}]={} != expected={}",
            result.variance_upper[[i]],
            expected_var
        );
    }
}

#[test]
fn test_scaling_gaussian_propagation() {
    // A = 2*I (scaling) + Gaussian input → output variance = 4 * input variance
    let dim = 2;
    let scale = 2.0;
    let lb = make_scaling_linear_bounds(dim, scale);

    let mean = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 2.0]).unwrap();
    let variance = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.5, 1.0]).unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(mean.clone()),
        variance: Box::new(variance.clone()),
    };
    // Input bounds not used for DiagonalGaussian, but needed for API
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![-10.0, -10.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![10.0, 10.0]).unwrap(),
    )
    .unwrap();

    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    // Output mean = scale * input mean
    for i in 0..dim {
        let expected_mean = scale * mean[[i]];
        assert!(
            (result.mean_lower[[i]] - expected_mean).abs() < 1e-5,
            "mean_lower[{i}]={} != expected={}",
            result.mean_lower[[i]],
            expected_mean
        );
    }

    // Output variance = scale^2 * input variance
    for i in 0..dim {
        let expected_var = scale * scale * variance[[i]];
        assert!(
            (result.variance_upper[[i]] - expected_var).abs() < 1e-5,
            "var[{i}]={} != expected={}",
            result.variance_upper[[i]],
            expected_var
        );
    }
}

#[test]
fn test_prob_bounds_wider_than_mean_bounds() {
    // Probabilistic bounds should be strictly wider than mean bounds (when variance > 0)
    let dim = 2;
    let lb = make_identity_linear_bounds(dim);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![2.0, 4.0]).unwrap(),
    )
    .unwrap();

    let dist = AnalyticDistribution::UniformFromBounds;
    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.99).unwrap();

    for i in 0..dim {
        assert!(
            result.prob_lower[[i]] < result.mean_lower[[i]],
            "prob_lower should be < mean_lower for dim {i}"
        );
        assert!(
            result.prob_upper[[i]] > result.mean_upper[[i]],
            "prob_upper should be > mean_upper for dim {i}"
        );
    }
}

#[test]
fn test_asymmetric_linear_bounds() {
    // Non-identity bounds: A_L != A_U
    let a_l = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 0.5]).unwrap();
    let a_u = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 1.5]).unwrap();
    let b = Array1::zeros(2);
    let lb = LinearBounds::new(a_l, b.clone(), a_u, b).unwrap();

    let mean = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let variance = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(mean),
        variance: Box::new(variance),
    };
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-5.0, -5.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0, 5.0]).unwrap(),
    )
    .unwrap();

    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    // mean_lower = A_L @ [1,1] = [1.0, 0.5]
    assert!((result.mean_lower[[0]] - 1.0).abs() < 1e-5);
    assert!((result.mean_lower[[1]] - 0.5).abs() < 1e-5);

    // mean_upper = A_U @ [1,1] = [2.0, 1.5]
    assert!((result.mean_upper[[0]] - 2.0).abs() < 1e-5);
    assert!((result.mean_upper[[1]] - 1.5).abs() < 1e-5);

    // Variance for dim 0: max(A_L[0,:]^2 @ var, A_U[0,:]^2 @ var)
    // = max(1^2*1 + 0^2*1, 2^2*1 + 0^2*1) = max(1, 4) = 4
    assert!((result.variance_upper[[0]] - 4.0).abs() < 1e-5);
    // Variance for dim 1: max(0.5^2*1, 1.5^2*1) = max(0.25, 2.25) = 2.25
    assert!((result.variance_upper[[1]] - 2.25).abs() < 1e-5);
}

#[test]
fn test_zero_variance_gives_tight_bounds() {
    // Zero variance → prob bounds == mean bounds
    let dim = 2;
    let lb = make_identity_linear_bounds(dim);
    let mean = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 2.0]).unwrap();
    let variance = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(mean),
        variance: Box::new(variance),
    };
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 2.0]).unwrap(),
    )
    .unwrap();

    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.99).unwrap();

    for i in 0..dim {
        assert!(
            (result.prob_lower[[i]] - result.mean_lower[[i]]).abs() < 1e-6,
            "prob should equal mean when variance is 0"
        );
    }
}

#[test]
fn test_invalid_confidence() {
    let lb = make_identity_linear_bounds(1);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;

    assert!(propagate_distribution(&lb, &dist, &input_bounds, 1.0).is_err());
    assert!(propagate_distribution(&lb, &dist, &input_bounds, -0.5).is_err());
}

#[test]
fn test_shape_mismatch() {
    // LinearBounds with 3 inputs, but input distribution has 2 elements
    let lb = make_identity_linear_bounds(3);
    let mean = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
    let variance = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(mean),
        variance: Box::new(variance),
    };
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    assert!(propagate_distribution(&lb, &dist, &input_bounds, 0.95).is_err());
}

// --- output_quantile tests ---

#[test]
fn test_output_quantile_higher_confidence_wider_interval() {
    let dim = 2;
    let lb = make_identity_linear_bounds(dim);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    let (lo_90, hi_90) = result.output_quantile(0.90).unwrap();
    let (lo_99, hi_99) = result.output_quantile(0.99).unwrap();

    // 99% interval should be strictly wider than 90% interval
    for i in 0..dim {
        assert!(
            lo_99[[i]] < lo_90[[i]],
            "99% lower should be below 90% lower"
        );
        assert!(
            hi_99[[i]] > hi_90[[i]],
            "99% upper should be above 90% upper"
        );
    }
}

#[test]
fn test_output_quantile_matches_stored_at_same_confidence() {
    let dim = 2;
    let lb = make_identity_linear_bounds(dim);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![2.0, 2.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    // Querying at the same confidence should match stored prob bounds
    let (lo, hi) = result.output_quantile(0.95).unwrap();
    for i in 0..dim {
        assert!(
            (lo[[i]] - result.prob_lower[[i]]).abs() < 1e-5,
            "quantile lower should match prob_lower at same confidence"
        );
        assert!(
            (hi[[i]] - result.prob_upper[[i]]).abs() < 1e-5,
            "quantile upper should match prob_upper at same confidence"
        );
    }
}

// --- output_probability tests ---

#[test]
fn test_output_probability_threshold_below_all_means() {
    // threshold well below mean_lower → high exceedance probability
    let dim = 1;
    let lb = make_identity_linear_bounds(dim);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![10.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![12.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    let (p_lo, p_hi) = result.output_probability(5.0).unwrap();
    // Mean is ~11, threshold=5 is far below → very high probability
    assert!(p_lo[[0]] > 0.5, "p_lower={} should be > 0.5", p_lo[[0]]);
    assert!((p_hi[[0]] - 1.0).abs() < 1e-10, "p_upper should be ~1.0");
}

#[test]
fn test_output_probability_threshold_above_all_means() {
    // threshold well above mean_upper → low exceedance probability (Cantelli bound)
    let dim = 1;
    let lb = make_identity_linear_bounds(dim);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![2.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    // mean_upper = center = 1.0, var = (2-0)^2/12 = 1/3
    // threshold = 10.0, far above → Cantelli: var/(var + (t-mu)^2) = (1/3)/(1/3 + 81) ≈ 0.004
    let (p_lo, p_hi) = result.output_probability(10.0).unwrap();
    assert!(p_hi[[0]] < 0.01, "p_upper={} should be < 0.01", p_hi[[0]]);
    assert!(
        p_lo[[0]] == 0.0,
        "p_lower should be 0 when threshold > mean"
    );
}

#[test]
fn test_output_probability_zero_variance() {
    // Zero variance (point distribution) → deterministic
    let dim = 1;
    let lb = make_identity_linear_bounds(dim);
    let mean = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![5.0]).unwrap();
    let variance = ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(mean),
        variance: Box::new(variance),
    };
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![5.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![5.0]).unwrap(),
    )
    .unwrap();
    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    // Point mass at 5.0 → P(X > 3) = 1, P(X > 7) = 0
    let (_, p_hi_3) = result.output_probability(3.0).unwrap();
    assert!((p_hi_3[[0]] - 1.0).abs() < 1e-10);

    let (_, p_hi_7) = result.output_probability(7.0).unwrap();
    assert!(p_hi_7[[0]] < 1e-10);
}

#[test]
fn test_output_probability_monotone_in_threshold() {
    // Higher threshold → lower exceedance probability
    let dim = 1;
    let lb = make_identity_linear_bounds(dim);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let result = propagate_distribution(&lb, &dist, &input_bounds, 0.95).unwrap();

    let (_, p5) = result.output_probability(5.0).unwrap();
    let (_, p10) = result.output_probability(10.0).unwrap();
    let (_, p20) = result.output_probability(20.0).unwrap();
    assert!(p5[[0]] >= p10[[0]], "P(>5) >= P(>10)");
    assert!(p10[[0]] >= p20[[0]], "P(>10) >= P(>20)");
}
