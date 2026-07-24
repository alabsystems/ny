// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_preactivation_positive_region() {
    // When l >= 0, alpha should be 1.0 (identity region)
    // Kills mutant: replace < with <= at line 889, 894
    let layer_bounds =
        vec![
            BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap(),
        ];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();
    let alpha = state.alpha(0).unwrap();
    assert_eq!(alpha[[0]], 1.0);
    assert_eq!(alpha[[1]], 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_preactivation_negative_region() {
    // When u <= 0, alpha should be 0.0 (zero region)
    let layer_bounds = vec![BoundedTensor::new(
        arr1(&[-3.0, -2.0]).into_dyn(),
        arr1(&[-1.0, -0.5]).into_dyn(),
    )
    .unwrap()];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();
    let alpha = state.alpha(0).unwrap();
    assert_eq!(alpha[[0]], 0.0);
    assert_eq!(alpha[[1]], 0.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_preactivation_unstable_heuristic() {
    // When l < 0 < u, alpha = 1 if u > -l, else 0
    // Kills mutant: replace > with >= at line 911, delete - at line 911
    // l = -1, u = 2: -l = 1, u > -l (2 > 1) => alpha = 1
    // l = -3, u = 1: -l = 3, u > -l (1 > 3) is false => alpha = 0
    let layer_bounds =
        vec![
            BoundedTensor::new(arr1(&[-1.0, -3.0]).into_dyn(), arr1(&[2.0, 1.0]).into_dyn())
                .unwrap(),
        ];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();
    let alpha = state.alpha(0).unwrap();
    assert_eq!(alpha[[0]], 1.0); // u=2 > -l=1
    assert_eq!(alpha[[1]], 0.0); // u=1 not > -l=3
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_from_preactivation_boundary_case() {
    // Test boundary: u == -l (should give alpha = 0 since not strictly >)
    // l = -2, u = 2: -l = 2, u > -l (2 > 2) is false => alpha = 0
    let layer_bounds =
        vec![BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap()];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();
    let alpha = state.alpha(0).unwrap();
    assert_eq!(alpha[[0]], 0.0); // u=2 not > -l=2
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_velocity_formula() {
    // vel[i] = momentum * vel[i] - learning_rate * gradient[i]
    // Kills mutants: replace - with +, replace * with /, replace * with +
    // Use l=-0.5, u=2 so alpha starts at 1 (u > -l => 2 > 0.5)
    let layer_bounds =
        vec![BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap()];
    let relu_indices = vec![0];
    let mut state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();

    // Initial alpha for unstable neuron should be 1.0
    let initial_alpha = state.alpha(0).unwrap()[[0]];
    assert_eq!(initial_alpha, 1.0);

    // Update with specific values: momentum=0.9, lr=0.1, gradient=1.0
    let gradient = arr1(&[1.0]);
    state.update(0, &gradient, 0.1, 0.9);

    // vel = 0.9 * 0 - 0.1 * 1.0 = -0.1
    // alpha = 1.0 + (-0.1) = 0.9 (clamped to [0,1])
    let expected = (initial_alpha - 0.1).clamp(0.0, 1.0);
    let actual = state.alpha(0).unwrap()[[0]];
    assert!(
        (actual - expected).abs() < 1e-6,
        "actual={}, expected={}",
        actual,
        expected
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_momentum_accumulates() {
    // Test that momentum accumulates across updates
    // Kills mutant: replace += with -=, replace += with *=
    // Use l=-0.5, u=2 so that alpha starts at 1 (u > -l => 2 > 0.5)
    let layer_bounds =
        vec![BoundedTensor::new(arr1(&[-0.5]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap()];
    let relu_indices = vec![0];
    let mut state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();

    // Initial alpha should be 1.0
    let alpha0 = state.alpha(0).unwrap()[[0]];
    assert_eq!(alpha0, 1.0);

    let gradient = arr1(&[0.5]);
    // First update: vel = 0 - 0.1*0.5 = -0.05, alpha = 1 + (-0.05) = 0.95
    state.update(0, &gradient, 0.1, 0.5);
    let alpha1 = state.alpha(0).unwrap()[[0]];

    // Second update: vel = 0.5*(-0.05) - 0.1*0.5 = -0.025 - 0.05 = -0.075
    // alpha = 0.95 + (-0.075) = 0.875
    state.update(0, &gradient, 0.1, 0.5);
    let alpha2 = state.alpha(0).unwrap()[[0]];

    // Alpha should have decreased further
    assert!(
        alpha2 < alpha1,
        "alpha2={} should be < alpha1={}",
        alpha2,
        alpha1
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_num_unstable_counts_correctly() {
    // Kills mutant: replace num_unstable -> usize with 0 or 1
    // Create bounds with 2 unstable neurons (crossing zero) and 1 stable
    let layer_bounds = vec![BoundedTensor::new(
        arr1(&[-1.0, -1.0, 1.0]).into_dyn(),
        arr1(&[1.0, 1.0, 2.0]).into_dyn(),
    )
    .unwrap()];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();
    assert_eq!(state.num_unstable(), 2);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_num_unstable_zero() {
    // Test with no unstable neurons
    let layer_bounds =
        vec![
            BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap(),
        ];
    let relu_indices = vec![0];
    let state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();
    assert_eq!(state.num_unstable(), 0);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_gradient_not_one_detects_mul_vs_div() {
    // Kills mutant: replace * with / in AlphaState::update
    let layer_bounds =
        vec![BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap()];
    let relu_indices = vec![0];
    let mut state = AlphaState::from_preactivation_bounds(&layer_bounds, &relu_indices).unwrap();

    let gradient = arr1(&[2.0]);
    state.update(0, &gradient, 0.1, 0.0);
    let alpha = state.alpha(0).unwrap()[[0]];
    assert!((alpha - 0.8).abs() < 1e-6, "alpha={}", alpha);
}
