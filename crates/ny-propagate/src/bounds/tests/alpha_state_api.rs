// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Targeted API coverage for `AlphaState` bilinear and INVPROP helpers.

use crate::bounds::alpha::AdamParams;
use crate::bounds::AlphaState;
use crate::{LayerGammas, OutputConstraints};
use ndarray::{array, Array4};

fn empty_alpha_state() -> AlphaState {
    AlphaState::from_preactivation_bounds(&[], &[])
        .expect("empty preactivation bounds should initialize AlphaState")
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_bilinear_alpha_lifecycle() {
    let mut state = empty_alpha_state();

    assert_eq!(state.bilinear_layer_indices(), Vec::<usize>::new());
    assert!(state.bilinear_alpha(7).is_none());
    assert_eq!(state.num_bilinear_params(), 0);

    state.init_bilinear_alpha(7, 2, 3, 4);
    state.init_bilinear_alpha(4, 1, 1, 2);

    assert_eq!(state.bilinear_layer_indices(), vec![4, 7]);
    // The `* 1` factors mirror the init_bilinear_alpha(4, 1, 1, 2) dims verbatim.
    #[allow(clippy::identity_op)]
    let expected_params = (4 * 2 * 3 * 4) + (4 * 1 * 1 * 2);
    assert_eq!(state.num_bilinear_params(), expected_params);

    let bilinear = state
        .bilinear_alpha(7)
        .expect("layer 7 bilinear alphas should exist after init");
    assert_eq!(bilinear.shape(), &[4, 2, 3, 4]);
    assert!(bilinear.iter().all(|&value| value == 1.0));

    state
        .bilinear_alpha_mut(7)
        .expect("mutable bilinear alphas should exist after init")[[3, 1, 2, 0]] = 0.25;
    assert_eq!(
        state
            .bilinear_alpha(7)
            .expect("layer 7 bilinear alphas should still exist")[[3, 1, 2, 0]],
        0.25
    );
    assert_eq!(
        state
            .bilinear_alpha(4)
            .expect("mutating layer 7 must not disturb other bilinear layers")[[0, 0, 0, 1]],
        1.0
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_invprop_accessors_and_double_init() {
    let mut state = empty_alpha_state();

    assert!(!state.has_invprop());
    assert!(state.invprop().is_none());
    assert!(state.invprop_mut().is_none());
    assert_eq!(state.num_ny_params(), 0);
    assert!(state.ny_params().is_empty());
    state
        .update_ny_params(&[1.0, 2.0])
        .expect("updating ny params without invprop should be a no-op");

    let constraints =
        OutputConstraints::ge_threshold(4, 1, 0.25).expect("threshold constraint should be valid");
    state
        .init_invprop_state(constraints.clone(), 2)
        .expect("first invprop init should succeed");
    assert!(state.has_invprop());
    assert_eq!(
        state
            .invprop()
            .expect("invprop should exist after init")
            .constraints
            .output_dim(),
        4
    );

    state
        .invprop_mut()
        .expect("mutable invprop state should exist after init")
        .add_layer_gammas("hidden".to_string(), LayerGammas::new(1, 3, false));
    assert!(state
        .invprop()
        .expect("invprop should still exist after mutation")
        .layer_gammas("hidden")
        .is_some());
    assert_eq!(state.num_ny_params(), 6);
    assert_eq!(state.ny_params(), vec![0.0; 6]);

    let err = state
        .init_invprop_state(constraints, 2)
        .expect_err("double init must return an error");
    assert!(
        err.to_string().contains("already-initialized"),
        "double-init error should mention the existing state: {err}"
    );
}

fn build_invprop_alpha_state() -> AlphaState {
    let mut state = empty_alpha_state();
    let constraints =
        OutputConstraints::ge_threshold(4, 0, 0.5).expect("threshold constraint should be valid");
    state
        .init_invprop_state(constraints, 1)
        .expect("invprop init should succeed");
    let invprop = state
        .invprop_mut()
        .expect("mutable invprop state should exist after init");
    invprop.add_layer_gammas("layer_a".to_string(), LayerGammas::new(2, 3, false));
    invprop.add_layer_gammas("layer_b".to_string(), LayerGammas::new(1, 2, true));
    state
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_clip_gammas_clamps_negative_values() {
    let mut state = build_invprop_alpha_state();
    let invprop = state
        .invprop_mut()
        .expect("mutable invprop state should exist after init");
    invprop
        .layer_gammas_mut("layer_a")
        .expect("layer_a gammas should exist")
        .gammas[[0, 0, 0]] = -1.0;
    invprop
        .layer_gammas_mut("layer_a")
        .expect("layer_a gammas should exist")
        .gammas[[1, 1, 2]] = -0.5;
    invprop
        .layer_gammas_mut("layer_b")
        .expect("layer_b gammas should exist")
        .gammas[[0, 0, 0]] = -2.0;
    invprop
        .layer_gammas_mut("layer_a")
        .expect("layer_a gammas should exist")
        .gammas[[0, 1, 1]] = 1.5;
    invprop
        .layer_gammas_mut("layer_b")
        .expect("layer_b gammas should exist")
        .gammas[[1, 0, 0]] = 0.75;

    assert_eq!(state.num_ny_params(), 14);
    state.clip_gammas();

    let invprop = state
        .invprop()
        .expect("invprop state should remain available after clipping");
    assert_eq!(
        invprop
            .layer_gammas("layer_a")
            .expect("layer_a gammas should exist")
            .gammas[[0, 0, 0]],
        0.0
    );
    assert_eq!(
        invprop
            .layer_gammas("layer_a")
            .expect("layer_a gammas should exist")
            .gammas[[1, 1, 2]],
        0.0
    );
    assert_eq!(
        invprop
            .layer_gammas("layer_b")
            .expect("layer_b gammas should exist")
            .gammas[[0, 0, 0]],
        0.0
    );
    assert_eq!(
        invprop
            .layer_gammas("layer_a")
            .expect("layer_a positive gammas should remain unchanged")
            .gammas[[0, 1, 1]],
        1.5
    );
    assert_eq!(
        invprop
            .layer_gammas("layer_b")
            .expect("layer_b positive gammas should remain unchanged")
            .gammas[[1, 0, 0]],
        0.75
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_ny_params_roundtrip_and_length_error() {
    let mut state = build_invprop_alpha_state();

    let new_params: Vec<f32> = (0..state.num_ny_params())
        .map(|idx| idx as f32 * 0.25)
        .collect();
    state
        .update_ny_params(&new_params)
        .expect("matching ny param length should update successfully");
    assert_eq!(state.ny_params(), new_params);

    let err = state
        .update_ny_params(&[1.0, 2.0])
        .expect_err("wrong-length ny params must error once invprop is active");
    assert!(
        err.to_string().contains("params length 2"),
        "wrong-length error should mention the actual provided length: {err}"
    );
    assert_eq!(
        state.ny_params(),
        new_params,
        "failed ny update must leave the existing parameters unchanged"
    );
}

// ==================== UPPER-PATH ALPHA TESTS ====================

/// Helper: create AlphaState with two ReLU layers containing unstable neurons.
/// Layer 0: 3 neurons [-1, 2], [-3, 1], [-0.5, 2] (all crossing)
/// Layer 1: 2 neurons [-4, 1], [-1, 5] (both crossing)
fn alpha_state_with_unstable() -> AlphaState {
    use super::checked_bounds;
    let bounds = vec![
        checked_bounds(
            array![-1.0_f32, -3.0, -0.5].into_dyn(),
            array![2.0_f32, 1.0, 2.0].into_dyn(),
        ),
        checked_bounds(
            array![-4.0_f32, -1.0].into_dyn(),
            array![1.0_f32, 5.0].into_dyn(),
        ),
    ];
    let relu_indices = vec![0, 1];
    AlphaState::from_preactivation_bounds(&bounds, &relu_indices)
        .expect("creating AlphaState with unstable neurons should succeed")
}

/// `alpha_upper()` returns upper-path alphas initialized identically to lower-path.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_alpha_upper_initialized_same_as_lower() {
    let state = alpha_state_with_unstable();

    // Both paths should be initialized identically from the heuristic.
    for relu_idx in 0..2 {
        let lower = state.alpha(relu_idx).expect("lower alpha should exist");
        let upper = state
            .alpha_upper(relu_idx)
            .expect("upper alpha should exist");
        assert_eq!(
            lower.as_slice().unwrap(),
            upper.as_slice().unwrap(),
            "lower and upper alphas must start identical for relu_idx {relu_idx}"
        );
    }

    // Out-of-range returns None.
    assert!(state.alpha_upper(99).is_none());
}

/// `update_upper()` applies SGD to upper-path alphas independently of lower-path.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_upper_independent_of_lower() {
    let mut state = alpha_state_with_unstable();

    let lower_before = state.alpha(0).expect("lower alpha should exist").clone();
    let upper_before = state
        .alpha_upper(0)
        .expect("upper alpha should exist")
        .clone();

    // Initial alphas for layer 0:
    //   neuron 0: l=-1, u=2, u > -l -> alpha=1.0 (positive gradient will decrease it)
    //   neuron 1: l=-3, u=1, u < -l -> alpha=0.0 (negative gradient will increase it)
    //   neuron 2: l=-0.5, u=2, u > -l -> alpha=1.0 (positive gradient will decrease it)
    // Use gradient signs that move each neuron away from its initial position.
    let gradient = array![0.5_f32, -0.5, 0.5];
    state.update_upper(0, &gradient, 0.1, 0.0);

    let lower_after = state.alpha(0).expect("lower alpha should exist");
    let upper_after = state.alpha_upper(0).expect("upper alpha should exist");

    // Lower path must be unchanged.
    assert_eq!(
        lower_after.as_slice().unwrap(),
        lower_before.as_slice().unwrap(),
        "update_upper must not modify lower-path alphas"
    );

    // Upper path must have changed (at unstable neurons).
    // All 3 neurons are unstable in layer 0.
    for i in 0..3 {
        assert_ne!(
            upper_after[i], upper_before[i],
            "upper alpha[{i}] should change after update_upper (before={}, after={})",
            upper_before[i], upper_after[i]
        );
        assert!(
            (0.0..=1.0).contains(&upper_after[i]),
            "upper alpha[{i}] must be in [0,1], got {}",
            upper_after[i]
        );
    }
}

/// `update_upper()` with out-of-range index is a no-op.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_upper_invalid_index_noop() {
    let mut state = alpha_state_with_unstable();
    let upper_before = state
        .alpha_upper(0)
        .expect("upper alpha should exist")
        .clone();

    let gradient = array![0.5_f32, 0.5, 0.5];
    state.update_upper(99, &gradient, 0.1, 0.0);

    let upper_after = state.alpha_upper(0).expect("upper alpha should exist");
    assert_eq!(upper_after, &upper_before);
}

/// `update_upper()` with mismatched gradient length is a no-op.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_upper_gradient_mismatch_noop() {
    let mut state = alpha_state_with_unstable();
    let upper_before = state
        .alpha_upper(0)
        .expect("upper alpha should exist")
        .clone();

    let wrong_gradient = array![0.5_f32]; // length 1 vs 3
    state.update_upper(0, &wrong_gradient, 0.1, 0.0);

    let upper_after = state.alpha_upper(0).expect("upper alpha should exist");
    assert_eq!(upper_after, &upper_before);
}

/// `update_adam_upper()` applies Adam optimizer to upper-path independently.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_adam_upper_independent_of_lower() {
    let mut state = alpha_state_with_unstable();

    let lower_before = state.alpha(1).expect("lower alpha should exist").clone();
    let upper_before = state
        .alpha_upper(1)
        .expect("upper alpha should exist")
        .clone();

    let gradient = array![-0.3_f32, 0.2];
    let params = AdamParams::new(0.01, 1);
    state.update_adam_upper(1, &gradient, &params);

    let lower_after = state.alpha(1).expect("lower alpha should exist");
    let upper_after = state.alpha_upper(1).expect("upper alpha should exist");

    // Lower path must be unchanged.
    assert_eq!(
        lower_after.as_slice().unwrap(),
        lower_before.as_slice().unwrap(),
        "update_adam_upper must not modify lower-path alphas"
    );

    // Upper path must have changed for unstable neurons.
    // Layer 1 has 2 unstable neurons.
    for i in 0..2 {
        assert_ne!(
            upper_after[i], upper_before[i],
            "upper alpha[{i}] should change after update_adam_upper"
        );
        assert!(
            (0.0..=1.0).contains(&upper_after[i]),
            "upper alpha[{i}] must be in [0,1], got {}",
            upper_after[i]
        );
        assert!(
            upper_after[i].is_finite(),
            "upper alpha[{i}] must be finite"
        );
    }
}

/// `update_adam_upper()` with mismatched gradient length is a no-op.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_adam_upper_gradient_mismatch_noop() {
    let mut state = alpha_state_with_unstable();
    let upper_before = state
        .alpha_upper(1)
        .expect("upper alpha should exist")
        .clone();

    let wrong_gradient = array![0.5_f32]; // length 1 vs 2
    let params = AdamParams::new(0.01, 1);
    state.update_adam_upper(1, &wrong_gradient, &params);

    let upper_after = state.alpha_upper(1).expect("upper alpha should exist");
    assert_eq!(upper_after, &upper_before);
}

/// `update_adam_upper()` with NaN gradient resets to safe state (0.5).
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_adam_upper_nan_gradient_sanitized() {
    let mut state = alpha_state_with_unstable();

    let nan_gradient = array![f32::NAN, f32::NAN];
    let params = AdamParams::new(0.01, 1);
    state.update_adam_upper(1, &nan_gradient, &params);

    let upper_after = state.alpha_upper(1).expect("upper alpha should exist");
    for i in 0..2 {
        assert!(
            upper_after[i].is_finite(),
            "upper alpha[{i}] must be finite after NaN gradient, got {}",
            upper_after[i]
        );
        assert!(
            (0.0..=1.0).contains(&upper_after[i]),
            "upper alpha[{i}] must be in [0,1] after NaN, got {}",
            upper_after[i]
        );
    }
}

// ==================== BILINEAR ADAM UPDATE TESTS ====================

/// `update_bilinear_adam` updates bilinear McCormick alphas via Adam and clamps to [0,1].
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_bilinear_adam_basic() {
    let mut state = empty_alpha_state();
    // Small bilinear layer: [4, 2, 2, 2] = 32 params
    state.init_bilinear_alpha(0, 2, 2, 2);

    // All alphas initialized to 1.0.
    let before = state
        .bilinear_alpha(0)
        .expect("bilinear alphas should exist")
        .clone();
    assert!(before.iter().all(|&v| v == 1.0));

    // Apply a gradient with positive values (will push alphas down from 1.0).
    let gradient = Array4::from_elem((4, 2, 2, 2), 1.0_f32);
    let params = AdamParams::new(0.1, 1);
    state.update_bilinear_adam(0, &gradient, &params);

    let after = state
        .bilinear_alpha(0)
        .expect("bilinear alphas should exist after update");

    // All values must be finite and in [0, 1].
    for &v in after.iter() {
        assert!(v.is_finite(), "bilinear alpha must be finite, got {v}");
        assert!(
            (0.0..=1.0).contains(&v),
            "bilinear alpha must be in [0,1], got {v}"
        );
    }

    // Positive gradient with lr=0.1 should have decreased alphas from 1.0.
    for &v in after.iter() {
        assert!(
            v < 1.0,
            "positive gradient should decrease alpha from 1.0, got {v}"
        );
    }
}

/// `update_bilinear_adam` with shape-mismatched gradient is a no-op.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_bilinear_adam_shape_mismatch_noop() {
    let mut state = empty_alpha_state();
    state.init_bilinear_alpha(0, 2, 3, 4);

    let before = state
        .bilinear_alpha(0)
        .expect("bilinear alphas should exist")
        .clone();

    // Wrong shape: [4, 1, 1, 1] vs expected [4, 2, 3, 4]
    let wrong_gradient = Array4::ones((4, 1, 1, 1));
    let params = AdamParams::new(0.1, 1);
    state.update_bilinear_adam(0, &wrong_gradient, &params);

    let after = state
        .bilinear_alpha(0)
        .expect("bilinear alphas should exist");
    assert_eq!(
        after, &before,
        "shape-mismatched gradient must leave bilinear alphas unchanged"
    );
}

/// `update_bilinear_adam` for nonexistent layer is a no-op.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_bilinear_adam_missing_layer_noop() {
    let mut state = empty_alpha_state();
    // No bilinear layers initialized.
    let gradient = Array4::ones((4, 2, 2, 2));
    let params = AdamParams::new(0.1, 1);
    // Should silently return without panic.
    state.update_bilinear_adam(99, &gradient, &params);
}

/// `update_bilinear_adam` with NaN gradient sanitizes to 0.5.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_bilinear_adam_nan_gradient_sanitized() {
    let mut state = empty_alpha_state();
    state.init_bilinear_alpha(0, 1, 1, 2);

    let nan_gradient = Array4::from_elem((4, 1, 1, 2), f32::NAN);
    let params = AdamParams::new(0.1, 1);
    state.update_bilinear_adam(0, &nan_gradient, &params);

    let after = state
        .bilinear_alpha(0)
        .expect("bilinear alphas should exist after NaN");
    for &v in after.iter() {
        assert!(
            v.is_finite(),
            "bilinear alpha must be finite after NaN gradient, got {v}"
        );
        assert!(
            (0.0..=1.0).contains(&v),
            "bilinear alpha must be in [0,1] after NaN, got {v}"
        );
    }
}

/// `update_bilinear_adam` preserves Adam moment state across iterations.
/// After multiple updates with consistent gradients, alpha should converge.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_update_bilinear_adam_multi_step_convergence() {
    let mut state = empty_alpha_state();
    state.init_bilinear_alpha(0, 1, 1, 1);

    // Push alpha down with repeated positive gradients.
    for t in 1..=20 {
        let gradient = Array4::from_elem((4, 1, 1, 1), 0.5_f32);
        let params = AdamParams::new(0.05, t);
        state.update_bilinear_adam(0, &gradient, &params);
    }

    let after = state
        .bilinear_alpha(0)
        .expect("bilinear alphas should exist after convergence test");
    for &v in after.iter() {
        assert!(v.is_finite(), "alpha must remain finite after 20 steps");
        assert!((0.0..=1.0).contains(&v), "alpha must stay in [0,1]");
        // After 20 steps with positive gradient, alpha should have moved well below 1.0.
        assert!(
            v < 0.5,
            "alpha should converge toward 0 with consistent positive gradient, got {v}"
        );
    }
}
