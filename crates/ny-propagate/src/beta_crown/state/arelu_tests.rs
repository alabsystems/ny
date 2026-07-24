// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::arelu::{compute_arelu_cut_slope_bias, AreluState};
use ndarray::{Array1, Array2};

#[ntest::timeout(5000)]
#[test]
fn test_arelu_state_empty() {
    let state = AreluState::empty();
    assert!(state.is_empty());
    assert_eq!(state.weighted_coeff(0, 0), None);
    assert!(!state.has_cut(0, 0));
}

#[ntest::timeout(5000)]
#[test]
fn test_arelu_state_from_cut_module() {
    let mut arelu_coeffs = std::collections::HashMap::new();
    // Layer 0 with 3 neurons, 2 cuts
    arelu_coeffs.insert(
        0,
        Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, -1.0, // Cut 0: coeff on neuron 0 and 2
                0.0, 0.5, 0.0, // Cut 1: coeff on neuron 1
            ],
        )
        .unwrap(),
    );

    let lambdas = Array1::from_vec(vec![2.0, 1.0]); // Lambda for each cut

    let state = AreluState::from_cut_module(&arelu_coeffs, &lambdas);

    assert!(!state.is_empty());
    // Neuron 0: 2.0 * 1.0 + 1.0 * 0.0 = 2.0
    assert_eq!(state.weighted_coeff(0, 0), Some(2.0));
    // Neuron 1: 2.0 * 0.0 + 1.0 * 0.5 = 0.5
    assert_eq!(state.weighted_coeff(0, 1), Some(0.5));
    // Neuron 2: 2.0 * -1.0 + 1.0 * 0.0 = -2.0
    assert_eq!(state.weighted_coeff(0, 2), Some(-2.0));

    assert!(state.has_cut(0, 0));
    assert!(state.has_cut(0, 1));
    assert!(state.has_cut(0, 2));
    assert!(!state.has_cut(1, 0)); // Layer 1 has no cuts
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_arelu_cut_slope_bias_stable_positive() {
    // Stable positive neuron: l >= 0
    // Function should not be called for stable neurons, but handles gracefully
    let (slope, lbias) = compute_arelu_cut_slope_bias(0.5, 2.0, -1.0, 0.5);
    assert_eq!(slope, 1.0); // Identity for positive-stable
    assert_eq!(lbias, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_arelu_cut_slope_bias_stable_negative() {
    // Stable negative neuron: u <= 0
    // Function should not be called for stable neurons, but handles gracefully
    let (slope, lbias) = compute_arelu_cut_slope_bias(-2.0, -0.5, -1.0, 0.5);
    assert_eq!(slope, 0.0); // Zero slope for negative-stable
    assert_eq!(lbias, 0.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_arelu_cut_slope_bias_positive_a() {
    // a_coeff >= 0 means using lower bound relaxation, arelu_cut doesn't apply
    // Returns standard upper bound values for consistency
    let l = -1.0;
    let u = 2.0;
    let a_coeff = 1.0;
    let beta_mm = 0.5;
    let (slope, lbias) = compute_arelu_cut_slope_bias(l, u, a_coeff, beta_mm);

    // Standard upper slope = u / (u - l) = 2/3
    let standard_slope = u / (u - l);
    let standard_intercept = -l * u / (u - l); // = 2/3
    assert!((slope - standard_slope).abs() < 1e-6);
    // Returns a_coeff * standard_intercept for consistency
    assert!((lbias - a_coeff * standard_intercept).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_arelu_cut_slope_bias_unstable_zero_coeff() {
    // Unstable neuron with zero beta_mm_coeff
    // This is in the default piecewise case: -u*nu < 0 < -l*nu
    // lbias = pi * lower
    let l = -1.0;
    let u = 2.0;
    let a_coeff = -1.0;
    let beta_mm = 0.0;

    let (slope, lbias) = compute_arelu_cut_slope_bias(l, u, a_coeff, beta_mm);

    // Standard upper slope = u / (u - l) = 2/3
    let standard_slope = u / (u - l);
    // pi = (u * nu_hat_pos + beta_mm) / (u - l) = (2*1 + 0) / 3 = 2/3
    // new_slope = pi / nu_hat_pos = (2/3) / 1 = 2/3 (same as standard)
    assert!((slope - standard_slope).abs() < 1e-6);

    // lbias = pi * lower = (2/3) * (-1) = -2/3
    let expected_lbias = (2.0 / 3.0) * l;
    assert!((lbias - expected_lbias).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_arelu_cut_slope_bias_unstable_with_cut() {
    // Unstable neuron with positive beta_mm_coeff
    // This is in the default piecewise case since -u*nu < beta_mm < -l*nu
    // -2*2 = -4 < 1.0 < -(-1)*2 = 2
    let l = -1.0;
    let u = 2.0;
    let a_coeff = -2.0; // nu_hat_pos = 2.0
    let beta_mm = 1.0;

    let (slope, lbias) = compute_arelu_cut_slope_bias(l, u, a_coeff, beta_mm);

    // pi = (u * nu_hat_pos + beta_mm) / (u - l)
    // pi = (2.0 * 2.0 + 1.0) / 3.0 = 5/3
    // pi = min(pi, nu_hat_pos).max(0) = min(5/3, 2) = 5/3
    // new_slope = pi / nu_hat_pos = (5/3) / 2 = 5/6
    let expected_slope = 5.0 / 6.0;
    assert!((slope - expected_slope).abs() < 1e-6);

    // lbias = pi * lower = (5/3) * (-1) = -5/3
    let pi = 5.0 / 3.0;
    let expected_lbias = pi * l;
    assert!((lbias - expected_lbias).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_arelu_cut_slope_bias_pi_clamped_to_nu() {
    // Case where pi would exceed nu_hat_pos without clamping
    // Large positive beta_mm triggers: beta_mm >= -l * nu_hat_pos
    // -l * nu = -(-0.1) * 1 = 0.1 < 10.0
    // So lbias = -beta_mm
    let l = -0.1;
    let u = 2.0;
    let a_coeff = -1.0; // nu_hat_pos = 1.0
    let beta_mm = 10.0; // Large coefficient

    let (slope, lbias) = compute_arelu_cut_slope_bias(l, u, a_coeff, beta_mm);

    // pi would be very large without clamping
    // pi = min(pi, nu_hat_pos) = 1.0
    // new_slope = 1.0 / 1.0 = 1.0
    assert!((slope - 1.0).abs() < 1e-6);

    // lbias = -beta_mm = -10.0 (since beta_mm >= -l * nu)
    assert!((lbias - (-beta_mm)).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_compute_arelu_cut_slope_bias_pi_clamped_to_zero() {
    // Case where pi would be negative without clamping
    // Large negative beta_mm triggers: beta_mm <= -u * nu_hat_pos
    // -u * nu = -0.1 * 1 = -0.1 > -10.0
    // So lbias = 0
    let l = -2.0;
    let u = 0.1;
    let a_coeff = -1.0; // nu_hat_pos = 1.0
    let beta_mm = -10.0; // Large negative coefficient

    let (slope, lbias) = compute_arelu_cut_slope_bias(l, u, a_coeff, beta_mm);

    // pi = max(pi, 0) = 0
    // new_slope = 0 / nu_hat_pos = 0
    assert!(slope.abs() < 1e-6);

    // lbias = 0 (since beta_mm <= -u * nu)
    assert!(lbias.abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_arelu_state_layer_coeffs() {
    let mut arelu_coeffs = std::collections::HashMap::new();
    arelu_coeffs.insert(0, Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap());
    let lambdas = Array1::from_vec(vec![1.0]);

    let state = AreluState::from_cut_module(&arelu_coeffs, &lambdas);

    let layer_coeffs = state.layer_coeffs(0);
    assert!(layer_coeffs.is_some());
    let coeffs = layer_coeffs.unwrap();
    assert_eq!(coeffs.len(), 2);
    assert_eq!(coeffs[0], 1.0);
    assert_eq!(coeffs[1], 2.0);

    // Non-existent layer
    assert!(state.layer_coeffs(1).is_none());
}
