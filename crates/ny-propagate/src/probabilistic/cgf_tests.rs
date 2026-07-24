// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for CGF propagation and Chernoff/saddlepoint bounds.
//! Part of #3921 / #4249.

use super::*;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use crate::bounds::LinearBounds;
use crate::probabilistic::distributional::AnalyticDistribution;

#[test]
fn test_cgf_gaussian_identity() {
    // Gaussian input + identity network → Gaussian output
    // Chernoff should give tight bounds matching Gaussian quantiles
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
    let lb = LinearBounds::identity(dim);

    let cgf_bound = propagate_cgf(&lb, &dist, &bounds, 0.95).unwrap();

    // For N(0,1) at 95% confidence, expected interval is roughly [-1.96, 1.96]
    let lower = cgf_bound.prob_lower[[0]];
    let upper = cgf_bound.prob_upper[[0]];

    assert!(
        lower < -1.0,
        "lower {lower} should be < -1.0 for N(0,1) at 95%"
    );
    assert!(
        upper > 1.0,
        "upper {upper} should be > 1.0 for N(0,1) at 95%"
    );
    // Should be somewhat close to Gaussian quantiles (Chernoff is exact for Gaussian)
    assert!(
        (upper - (-lower)).abs() < 0.5,
        "should be roughly symmetric: lower={lower}, upper={upper}"
    );
}

#[test]
fn test_cgf_sound_for_uniform() {
    // CGF/Chernoff bounds for uniform inputs must be sound (contain the support).
    // Note: Chernoff bounds are NOT necessarily tighter than Gaussian for individual
    // variables — their advantage comes from the additive property when summing many
    // independent inputs. For 1D identity mapping, Chernoff overshoots because the
    // exponential tail bound is conservative for light-tailed distributions.
    let dim = 2;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let lb = LinearBounds::identity(dim);

    let cgf_bound = propagate_cgf(&lb, &dist, &bounds, 0.95).unwrap();

    for i in 0..dim {
        // Sound: CGF interval must contain the true support [0, 4]
        assert!(
            (cgf_bound.prob_lower[[i]] as f64) <= 0.0,
            "dim {i}: CGF lower {:.4} should be <= 0.0 (sound containment)",
            cgf_bound.prob_lower[[i]]
        );
        assert!(
            (cgf_bound.prob_upper[[i]] as f64) >= 4.0,
            "dim {i}: CGF upper {:.4} should be >= 4.0 (sound containment)",
            cgf_bound.prob_upper[[i]]
        );
    }
}

#[test]
fn test_cgf_scaling() {
    // Scaling by 2: output interval should roughly double (wider by 2x)
    let dim = 1;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;

    let lb_id = LinearBounds::identity(dim);
    let a2 = Array2::from_diag(&Array1::from_elem(dim, 2.0_f32));
    let b = Array1::zeros(dim);
    let lb_2x = LinearBounds::new(a2.clone(), b.clone(), a2, b).unwrap();

    let cgf_id = propagate_cgf(&lb_id, &dist, &bounds, 0.95).unwrap();
    let cgf_2x = propagate_cgf(&lb_2x, &dist, &bounds, 0.95).unwrap();

    let width_id = cgf_id.prob_upper[[0]] - cgf_id.prob_lower[[0]];
    let width_2x = cgf_2x.prob_upper[[0]] - cgf_2x.prob_lower[[0]];

    // Width should roughly double (not exactly due to Chernoff nonlinearity)
    let ratio = width_2x / width_id;
    assert!(
        (1.5..2.5).contains(&ratio),
        "2x scaling should roughly double the interval: ratio={ratio}"
    );
}

#[test]
fn test_uniform_cgf_values() {
    // Verify uniform CGF at theta=0 gives 0
    let cgf = ElementCgf::Uniform {
        lower: 0.0,
        upper: 4.0,
    };
    assert!(cgf.psi(0.0).abs() < 1e-10, "CGF at theta=0 should be 0");

    // psi'(0) should be the mean = 2.0
    assert!(
        (cgf.psi_prime(0.0) - 2.0).abs() < 1e-6,
        "CGF'(0) should be the mean"
    );
}

#[test]
fn test_gaussian_cgf_exact() {
    let cgf = ElementCgf::Gaussian {
        mean: 1.0,
        variance: 2.0,
    };
    let theta = 0.5;
    // psi(theta) = theta*mu + theta^2*sigma^2/2 = 0.5 + 0.25
    let expected = 0.5 * 1.0 + 0.5 * 0.25 * 2.0;
    assert!(
        (cgf.psi(theta) - expected).abs() < 1e-10,
        "Gaussian CGF should be exact"
    );
    // psi'(theta) = mu + theta*sigma^2 = 1.0 + 0.5*2.0 = 2.0
    assert!(
        (cgf.psi_prime(theta) - 2.0).abs() < 1e-10,
        "Gaussian CGF' should be exact"
    );
}

#[test]
fn test_cgf_confidence_monotone() {
    // Higher confidence → wider interval
    let dim = 1;
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let lb = LinearBounds::identity(dim);

    let cgf_90 = propagate_cgf(&lb, &dist, &bounds, 0.90).unwrap();
    let cgf_99 = propagate_cgf(&lb, &dist, &bounds, 0.99).unwrap();

    let width_90 = cgf_90.prob_upper[[0]] - cgf_90.prob_lower[[0]];
    let width_99 = cgf_99.prob_upper[[0]] - cgf_99.prob_lower[[0]];

    assert!(
        width_99 >= width_90 - 1e-4,
        "99% interval should be >= 90% interval: w99={width_99}, w90={width_90}"
    );
}

/// Regression test for #4286: point-mass bounds (lower == upper) must not
/// produce NaN in psi() or psi_prime().
#[test]
fn test_cgf_psi_point_mass_does_not_nan_4286() {
    let v = 3.0;
    let cgf = ElementCgf::Uniform { lower: v, upper: v };

    // psi(theta) should be theta * v for a point mass
    for &theta in &[0.0, 0.1, 1.0, -0.5, 10.0] {
        let psi = cgf.psi(theta);
        assert!(!psi.is_nan(), "psi({theta}) is NaN for point mass at {v}");
        assert!(
            (psi - theta * v).abs() < 1e-10,
            "psi({theta}) = {psi}, expected {}",
            theta * v
        );
    }

    // psi_prime(theta) should be v for a point mass
    for &theta in &[0.0, 0.1, 1.0, -0.5, 10.0] {
        let dpsi = cgf.psi_prime(theta);
        assert!(
            !dpsi.is_nan(),
            "psi_prime({theta}) is NaN for point mass at {v}"
        );
        assert!(
            (dpsi - v).abs() < 1e-10,
            "psi_prime({theta}) = {dpsi}, expected {v}"
        );
    }
}

/// Regression test for #4286: end-to-end CGF propagation with point-mass
/// inputs should not produce NaN in CgfBound.
#[test]
fn test_cgf_propagate_point_mass_no_nan_4286() {
    let dim = 2;
    // Second element is a point mass (lower == upper == 3.0)
    let bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![0.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[dim]), vec![4.0, 3.0]).unwrap(),
    )
    .unwrap();
    let dist = AnalyticDistribution::UniformFromBounds;
    let lb = LinearBounds::identity(dim);

    let result = propagate_cgf(&lb, &dist, &bounds, 0.95).unwrap();

    for i in 0..dim {
        assert!(
            !result.prob_upper[[i]].is_nan(),
            "prob_upper[{i}] is NaN with point-mass input"
        );
        assert!(
            !result.prob_lower[[i]].is_nan(),
            "prob_lower[{i}] is NaN with point-mass input"
        );
        assert!(
            !result.exceedance_upper[[i]].is_nan(),
            "exceedance_upper[{i}] is NaN with point-mass input"
        );
        assert!(
            !result.shortfall_upper[[i]].is_nan(),
            "shortfall_upper[{i}] is NaN with point-mass input"
        );
    }

    // Point-mass dimension should have tight bounds around the fixed value
    let pm_lower = result.prob_lower[[1]] as f64;
    let pm_upper = result.prob_upper[[1]] as f64;
    assert!(
        (pm_upper - pm_lower).abs() < 0.5,
        "point-mass dim should have tight interval: [{pm_lower}, {pm_upper}]"
    );
    assert!(
        pm_lower <= 3.0 + 0.1 && pm_upper >= 3.0 - 0.1,
        "point-mass interval should contain 3.0: [{pm_lower}, {pm_upper}]"
    );
}
