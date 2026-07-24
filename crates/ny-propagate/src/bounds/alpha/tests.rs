// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Inline optimizer regression tests for shared alpha helpers.

use super::shared::update_alphas_adam;
use super::AdamParams;
use ndarray::Array1;

/// Regression: beta1=1.0 caused bias_correction1=0, division by zero (#2315).
#[test]
fn test_adam_beta1_one_no_div_by_zero() {
    let mut alpha = Array1::from_vec(vec![0.5]);
    let gradient = Array1::from_vec(vec![0.1]);
    let mask = Array1::from_vec(vec![true]);
    let mut m = Array1::zeros(1);
    let mut v = Array1::zeros(1);
    let params = AdamParams {
        learning_rate: 0.01,
        beta1: 1.0,
        beta2: 0.999,
        epsilon: 1e-8,
        t: 1,
    };
    update_alphas_adam(&mut alpha, &gradient, &mask, &mut m, &mut v, &params);
    assert!(alpha[0].is_finite(), "alpha must be finite with beta1=1.0");
    assert!((0.0..=1.0).contains(&alpha[0]), "alpha must be in [0, 1]");
}

/// Regression: beta2=1.0 caused bias_correction2=0, division by zero (#2315).
#[test]
fn test_adam_beta2_one_no_div_by_zero() {
    let mut alpha = Array1::from_vec(vec![0.5]);
    let gradient = Array1::from_vec(vec![0.1]);
    let mask = Array1::from_vec(vec![true]);
    let mut m = Array1::zeros(1);
    let mut v = Array1::zeros(1);
    let params = AdamParams {
        learning_rate: 0.01,
        beta1: 0.9,
        beta2: 1.0,
        epsilon: 1e-8,
        t: 1,
    };
    update_alphas_adam(&mut alpha, &gradient, &mask, &mut m, &mut v, &params);
    assert!(alpha[0].is_finite(), "alpha must be finite with beta2=1.0");
    assert!((0.0..=1.0).contains(&alpha[0]), "alpha must be in [0, 1]");
}
