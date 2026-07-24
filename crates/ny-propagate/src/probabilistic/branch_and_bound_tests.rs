// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for branch-and-bound distributional bound refinement.
//! Part of #3921 / #4249.

use super::*;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;

fn make_identity_linear_bounds(dim: usize) -> LinearBounds {
    LinearBounds::identity(dim)
}

/// Helper: create a crown_fn that always returns identity LinearBounds.
/// This represents a network that is the identity function f(x) = x.
fn identity_crown_fn() -> impl Fn(&BoundedTensor) -> Result<LinearBounds> {
    |input: &BoundedTensor| {
        let dim = input.len();
        Ok(make_identity_linear_bounds(dim))
    }
}

/// Helper: create a crown_fn with a fixed scaling matrix.
fn scaling_crown_fn(scale: f32) -> impl Fn(&BoundedTensor) -> Result<LinearBounds> {
    move |input: &BoundedTensor| {
        let dim = input.len();
        let a = Array2::from_diag(&Array1::from_elem(dim, scale));
        let b = Array1::zeros(dim);
        LinearBounds::new(a.clone(), b.clone(), a, b)
    }
}

/// Helper: create a crown_fn that returns tighter bounds on smaller regions.
/// Simulates the behavior of CROWN on a nonlinear network: wider input region
/// → looser relaxation (larger gap between lower_a and upper_a).
fn nonlinear_crown_fn() -> impl Fn(&BoundedTensor) -> Result<LinearBounds> {
    |input: &BoundedTensor| {
        let dim = input.len();
        let width = input.width();
        // Simulate CROWN looseness proportional to input width:
        // A_L[i,i] = 1 - width_i/4, A_U[i,i] = 1 + width_i/4
        let a_l_diag: Vec<f32> = width.iter().map(|&w| 1.0 - w / 4.0).collect();
        let a_u_diag: Vec<f32> = width.iter().map(|&w| 1.0 + w / 4.0).collect();
        let a_l = Array2::from_diag(&Array1::from_vec(a_l_diag));
        let a_u = Array2::from_diag(&Array1::from_vec(a_u_diag));
        let b = Array1::zeros(dim);
        LinearBounds::new(a_l, b.clone(), a_u, b)
    }
}

#[test]
fn test_bab_identity_network_no_improvement() {
    // Identity crown_fn → splitting doesn't help (bounds are already exact)
    let dim = 2;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let config = BranchAndBoundConfig {
        max_iterations: 4,
        max_regions: 16,
        tolerance: 0.001,
        confidence: 0.95,
        ..Default::default()
    };

    let result =
        refine_distributional_bounds(&identity_crown_fn(), &input_bounds, &dist, &config).unwrap();

    // Identity bounds should converge quickly (variance doesn't change with splitting)
    assert!(result.iterations <= 4);
    assert!(result.num_regions > 0);
}

#[test]
fn test_bab_nonlinear_refinement_tighter_than_single() {
    // Nonlinear crown_fn: splitting should produce tighter bounds
    let dim = 2;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;

    // Single-region bound
    let single_lb = nonlinear_crown_fn()(&input_bounds).unwrap();
    let single = propagate_distribution(&single_lb, &dist, &input_bounds, 0.95).unwrap();

    // B&B refined bound
    let config = BranchAndBoundConfig {
        max_iterations: 8,
        max_regions: 32,
        tolerance: 0.001,
        confidence: 0.95,
        ..Default::default()
    };
    let refined =
        refine_distributional_bounds(&nonlinear_crown_fn(), &input_bounds, &dist, &config).unwrap();

    // Refined variance should be <= single-region variance
    for i in 0..dim {
        assert!(
            refined.bound.variance_upper[[i]] <= single.variance_upper[[i]] + 1e-6,
            "refined var[{i}]={} should be <= single var={}",
            refined.bound.variance_upper[[i]],
            single.variance_upper[[i]]
        );
    }

    // Refined prob interval should be tighter (or equal)
    for i in 0..dim {
        let single_width = single.prob_upper[[i]] - single.prob_lower[[i]];
        let refined_width = refined.bound.prob_upper[[i]] - refined.bound.prob_lower[[i]];
        assert!(
            refined_width <= single_width + 1e-5,
            "refined interval width {} should be <= single width {}",
            refined_width,
            single_width
        );
    }
}

#[test]
fn test_bab_respects_max_iterations() {
    let dim = 1;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![8.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let config = BranchAndBoundConfig {
        max_iterations: 3,
        max_regions: 100,
        tolerance: 0.0, // never converge early
        confidence: 0.95,
        ..Default::default()
    };

    let result =
        refine_distributional_bounds(&scaling_crown_fn(2.0), &input_bounds, &dist, &config)
            .unwrap();

    assert!(result.iterations <= 3);
}

#[test]
fn test_bab_respects_max_regions() {
    let dim = 1;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![8.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let config = BranchAndBoundConfig {
        max_iterations: 100,
        max_regions: 4, // stop when we reach 4 regions
        tolerance: 0.0,
        confidence: 0.95,
        ..Default::default()
    };

    let result =
        refine_distributional_bounds(&nonlinear_crown_fn(), &input_bounds, &dist, &config).unwrap();

    assert!(result.num_regions <= 4 + 2); // may overshoot by one split (adds 2, removes 1)
}

#[test]
fn test_bisect_region() {
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 2.0, 4.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![6.0, 8.0, 10.0]).unwrap(),
    )
    .unwrap();

    let (left, right) = bisect_region(&bounds, 1).unwrap();

    // Left: [0,2,4] to [6,5,10] (dim 1 upper becomes midpoint)
    assert!((left.upper()[[1]] - 5.0).abs() < 1e-6);
    assert!((left.lower()[[1]] - 2.0).abs() < 1e-6);

    // Right: [0,5,4] to [6,8,10] (dim 1 lower becomes midpoint)
    assert!((right.lower()[[1]] - 5.0).abs() < 1e-6);
    assert!((right.upper()[[1]] - 8.0).abs() < 1e-6);

    // Other dimensions unchanged
    assert!((left.lower()[[0]] - 0.0).abs() < 1e-6);
    assert!((right.upper()[[2]] - 10.0).abs() < 1e-6);
}

#[test]
fn test_choose_split_widest_picks_widest() {
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 10.0, 3.0]).unwrap(),
    )
    .unwrap();

    let dim = choose_split_widest(&bounds);
    assert_eq!(dim, 1, "should pick dimension 1 (width=10)");
}

#[test]
fn test_bab_gaussian_input_distribution() {
    // B&B should work with DiagonalGaussian too, not just UniformFromBounds
    let dim = 2;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![-2.0, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![2.0, 2.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap()),
        variance: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 1.0]).unwrap()),
    };
    let config = BranchAndBoundConfig {
        max_iterations: 4,
        max_regions: 16,
        tolerance: 0.01,
        confidence: 0.95,
        ..Default::default()
    };

    let result =
        refine_distributional_bounds(&nonlinear_crown_fn(), &input_bounds, &dist, &config).unwrap();

    assert!(result.num_regions > 0);
    assert!(result.bound.variance_upper[[0]].is_finite());
}

#[test]
fn test_bab_probability_weighting_gaussian() {
    // Probability-weighted B&B should focus computation on high-density regions
    let dim = 2;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![-3.0, -3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![3.0, 3.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap()),
        variance: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 1.0]).unwrap()),
    };

    // Without probability weighting
    let config_no_pw = BranchAndBoundConfig {
        max_iterations: 4,
        max_regions: 16,
        tolerance: 0.001,
        confidence: 0.95,
        use_probability_weighting: false,
        ..Default::default()
    };
    let result_no_pw =
        refine_distributional_bounds(&nonlinear_crown_fn(), &input_bounds, &dist, &config_no_pw)
            .unwrap();

    // With probability weighting
    let config_pw = BranchAndBoundConfig {
        max_iterations: 4,
        max_regions: 16,
        tolerance: 0.001,
        confidence: 0.95,
        use_probability_weighting: true,
        ..Default::default()
    };
    let result_pw =
        refine_distributional_bounds(&nonlinear_crown_fn(), &input_bounds, &dist, &config_pw)
            .unwrap();

    // Both should produce finite bounds
    assert!(result_no_pw.bound.variance_upper[[0]].is_finite());
    assert!(result_pw.bound.variance_upper[[0]].is_finite());
    assert!(result_pw.num_regions > 0);
}

#[test]
fn test_bab_distributional_gradient_split() {
    // Gradient-guided splitting should produce valid results
    let dim = 2;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let config = BranchAndBoundConfig {
        max_iterations: 4,
        max_regions: 16,
        tolerance: 0.001,
        confidence: 0.95,
        split_strategy: SplitStrategy::DistributionalGradient,
        ..Default::default()
    };

    let result =
        refine_distributional_bounds(&nonlinear_crown_fn(), &input_bounds, &dist, &config).unwrap();

    assert!(result.num_regions > 0);
    assert!(result.bound.variance_upper[[0]].is_finite());
    // Gradient split should produce at least as good results as widest-dim
    // (may be equal on symmetric problems)
}

#[test]
fn test_bab_exact_conditional_moments() {
    // Exact conditional moments with Gaussian input
    let dim = 2;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![-2.0, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![2.0, 2.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap()),
        variance: Box::new(ArrayD::from_shape_vec(IxDyn(&[dim]), vec![1.0, 1.0]).unwrap()),
    };
    let config = BranchAndBoundConfig {
        max_iterations: 4,
        max_regions: 16,
        tolerance: 0.001,
        confidence: 0.95,
        use_probability_weighting: true,
        use_exact_conditional_moments: true,
        ..Default::default()
    };

    let result =
        refine_distributional_bounds(&nonlinear_crown_fn(), &input_bounds, &dist, &config).unwrap();

    assert!(result.num_regions > 0);
    assert!(result.bound.variance_upper[[0]].is_finite());
}

#[test]
fn test_region_probability_mass_gaussian() {
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::DiagonalGaussian {
        mean: Box::new(ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap()),
        variance: Box::new(ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap()),
    };

    let mass = region_probability_mass(&bounds, &dist);
    // P(-1 <= N(0,1) <= 1) ≈ 0.6827
    assert!(
        (mass - 0.6827).abs() < 0.01,
        "P(|Z|<=1) should be ~0.6827, got {mass}"
    );

    // Full range should have mass ~1
    let wide = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-10.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).unwrap(),
    )
    .unwrap();
    let wide_mass = region_probability_mass(&wide, &dist);
    assert!(
        wide_mass > 0.999,
        "P(|Z|<=10) should be ~1, got {wide_mass}"
    );
}

#[test]
fn test_truncated_gaussian_mean_symmetric() {
    // Symmetric truncation around the mean: E[X | -a <= X <= a] = mean
    let tm = truncated_gaussian_mean(0.0, 1.0, -2.0, 2.0);
    assert!(
        tm.abs() < 0.001,
        "symmetric truncation should give mean ~0, got {tm}"
    );

    // One-sided truncation: E[X | 0 <= X] for N(0,1) = phi(0)/(1-Phi(0)) = sqrt(2/pi)
    let tm_pos = truncated_gaussian_mean(0.0, 1.0, 0.0, 10.0);
    let expected = (2.0_f64 / std::f64::consts::PI).sqrt(); // ~0.7979
    assert!(
        (tm_pos - expected).abs() < 0.01,
        "E[X|X>=0] for N(0,1) should be ~{expected:.4}, got {tm_pos}"
    );
}

/// Verify that NaN variance regions don't corrupt the BinaryHeap ordering.
/// Before #4288 fix, `partial_cmp(...).unwrap_or(Equal)` made NaN regions
/// compare equal to everything, violating total order. With `total_cmp`,
/// NaN sorts after all finite values → highest priority in max-heap.
#[test]
fn test_bab_region_nan_variance_does_not_corrupt_heap() {
    use std::collections::BinaryHeap;

    let mk_bounds = |lo: f32, hi: f32| {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![lo]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![hi]).unwrap(),
        )
        .unwrap()
    };
    let mk_dist = |var: f32| DistributionalBound {
        mean_lower: ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        mean_upper: ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        variance_upper: ArrayD::from_shape_vec(IxDyn(&[1]), vec![var]).unwrap(),
        prob_lower: ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
        prob_upper: ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        confidence: 0.95,
    };

    let finite_low = Region {
        bounds: mk_bounds(0.0, 1.0),
        volume_weight: 0.5,
        dist_bound: mk_dist(1.0),
        max_weighted_variance: 0.5,
    };
    let finite_high = Region {
        bounds: mk_bounds(1.0, 2.0),
        volume_weight: 0.5,
        dist_bound: mk_dist(10.0),
        max_weighted_variance: 5.0,
    };
    let nan_region = Region {
        bounds: mk_bounds(2.0, 3.0),
        volume_weight: 0.5,
        dist_bound: mk_dist(f32::NAN),
        max_weighted_variance: f64::NAN,
    };

    // Verify total_cmp gives NaN > finite (NaN sorts highest)
    assert_eq!(
        f64::NAN.total_cmp(&5.0),
        Ordering::Greater,
        "NaN should sort greater than finite under total_cmp"
    );

    let mut heap = BinaryHeap::new();
    heap.push(finite_low);
    heap.push(nan_region);
    heap.push(finite_high);

    // NaN region should be popped first (highest priority = explored first)
    let first = heap.pop().unwrap();
    assert!(
        first.max_weighted_variance.is_nan(),
        "NaN-variance region should have highest priority, got {}",
        first.max_weighted_variance
    );

    // Then the finite_high (5.0)
    let second = heap.pop().unwrap();
    assert!(
        (second.max_weighted_variance - 5.0).abs() < 1e-10,
        "second pop should be 5.0, got {}",
        second.max_weighted_variance
    );

    // Then finite_low (0.5)
    let third = heap.pop().unwrap();
    assert!(
        (third.max_weighted_variance - 0.5).abs() < 1e-10,
        "third pop should be 0.5, got {}",
        third.max_weighted_variance
    );
}

/// Regression test for #4288: Region::PartialEq must be consistent with Ord.
///
/// Before fix, `PartialEq::eq` used f64's `==` (NaN != NaN = true), while
/// `Ord::cmp` used `total_cmp` (NaN == NaN = Equal). This violates the trait
/// contract: `a.cmp(b) == Equal` must imply `a == b`.
#[test]
fn test_region_partial_eq_consistent_with_ord_4288() {
    let mk_bounds = |lo: f32, hi: f32| {
        BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![lo]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![hi]).unwrap(),
        )
        .unwrap()
    };
    let mk_dist = |var: f32| DistributionalBound {
        mean_lower: ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        mean_upper: ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        variance_upper: ArrayD::from_shape_vec(IxDyn(&[1]), vec![var]).unwrap(),
        prob_lower: ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
        prob_upper: ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        confidence: 0.95,
    };

    let nan_a = Region {
        bounds: mk_bounds(0.0, 1.0),
        volume_weight: 0.5,
        dist_bound: mk_dist(f32::NAN),
        max_weighted_variance: f64::NAN,
    };
    let nan_b = Region {
        bounds: mk_bounds(1.0, 2.0),
        volume_weight: 0.5,
        dist_bound: mk_dist(f32::NAN),
        max_weighted_variance: f64::NAN,
    };

    // Ord says Equal → PartialEq must say equal
    assert_eq!(nan_a.cmp(&nan_b), Ordering::Equal);
    assert!(
        nan_a == nan_b,
        "PartialEq must agree with Ord: NaN regions should be equal"
    );

    // Finite regions: both traits must agree
    let finite_a = Region {
        bounds: mk_bounds(0.0, 1.0),
        volume_weight: 0.5,
        dist_bound: mk_dist(1.0),
        max_weighted_variance: 3.0,
    };
    let finite_b = Region {
        bounds: mk_bounds(0.0, 1.0),
        volume_weight: 0.5,
        dist_bound: mk_dist(1.0),
        max_weighted_variance: 3.0,
    };
    assert_eq!(finite_a.cmp(&finite_b), Ordering::Equal);
    assert_eq!(finite_a, finite_b);

    // Mixed NaN/finite: not equal
    assert_ne!(nan_a.cmp(&finite_a), Ordering::Equal);
    assert_ne!(nan_a, finite_a);
}

/// Regression test for #4288: max_element must propagate NaN.
///
/// Before fix, `f32::max` (IEEE 754 minNum semantics) silently discards NaN,
/// so `max_element([1.0, NaN, 2.0])` returned `2.0` instead of `NaN`.
/// This hid corrupt bounds from upstream diagnostics.
#[test]
fn test_max_element_propagates_nan_4288() {
    // All finite: normal max
    let finite = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 3.0, 2.0]).unwrap();
    assert_eq!(max_element(&finite), 3.0);

    // Contains NaN: must return NaN
    let with_nan = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, f32::NAN, 2.0]).unwrap();
    assert!(
        max_element(&with_nan).is_nan(),
        "max_element should propagate NaN, got {}",
        max_element(&with_nan)
    );

    // All NaN: must return NaN
    let all_nan = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, f32::NAN]).unwrap();
    assert!(max_element(&all_nan).is_nan());

    // Empty array: NEG_INFINITY (fold identity)
    let empty = ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap();
    assert_eq!(max_element(&empty), f32::NEG_INFINITY);

    // NaN at start: must still propagate
    let nan_first = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 1.0, 2.0]).unwrap();
    assert!(max_element(&nan_first).is_nan());
}
