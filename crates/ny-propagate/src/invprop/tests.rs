// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for INVPROP: config, constraints, gammas, and state.

use super::*;
use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;

#[ntest::timeout(5000)]
#[test]
fn test_output_constraints_ge_threshold() {
    let constraints =
        OutputConstraints::ge_threshold(3, 1, 0.5).expect("invariant: valid ge_threshold args");
    assert_eq!(constraints.num_constraints(), 1);
    assert_eq!(constraints.output_dim(), 3);
    assert!(constraints.is_conjunction);

    // Constraint: -y[1] <= -0.5, i.e., y[1] >= 0.5
    let satisfied = arr1(&[0.0, 0.6, 0.0]);
    let not_satisfied = arr1(&[0.0, 0.4, 0.0]);

    assert!(constraints.is_satisfied(&satisfied));
    assert!(!constraints.is_satisfied(&not_satisfied));
}

#[ntest::timeout(5000)]
#[test]
fn test_output_constraints_le_threshold() {
    let constraints =
        OutputConstraints::le_threshold(3, 0, 1.0).expect("invariant: valid le_threshold args");

    // Constraint: y[0] <= 1.0
    let satisfied = arr1(&[0.9, 0.0, 0.0]);
    let not_satisfied = arr1(&[1.1, 0.0, 0.0]);

    assert!(constraints.is_satisfied(&satisfied));
    assert!(!constraints.is_satisfied(&not_satisfied));
}

#[ntest::timeout(5000)]
#[test]
fn test_output_constraints_argmax() {
    let constraints = OutputConstraints::argmax(4, 2).expect("invariant: valid argmax args");
    assert_eq!(constraints.num_constraints(), 3); // 4 - 1 = 3

    // y[2] should be largest
    let satisfied = arr1(&[0.1, 0.2, 0.5, 0.3]);
    let not_satisfied = arr1(&[0.5, 0.2, 0.3, 0.1]);

    assert!(constraints.is_satisfied(&satisfied));
    assert!(!constraints.is_satisfied(&not_satisfied));
}

#[ntest::timeout(5000)]
#[test]
fn test_output_constraints_new_dimension_mismatch_returns_error() {
    let err =
        OutputConstraints::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]), arr1(&[0.0]), true).unwrap_err();
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(ref msg) if msg.contains("row/rhs mismatch")),
        "expected InvalidSpec with row/rhs mismatch, got {err:?}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_output_constraints_invalid_target_returns_error() {
    let err = OutputConstraints::argmax(3, 3).unwrap_err();
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(ref msg) if msg.contains("out of bounds")),
        "expected InvalidSpec with out of bounds, got {err:?}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_output_constraints_invalid_threshold_targets_return_error() {
    let ge_err = OutputConstraints::ge_threshold(2, 2, 0.0).unwrap_err();
    assert!(
        matches!(ge_err, ny_core::NyError::InvalidSpec(ref msg) if msg.contains("out of bounds")),
        "expected InvalidSpec from ge_threshold, got {ge_err:?}"
    );

    let le_err = OutputConstraints::le_threshold(2, 2, 0.0).unwrap_err();
    assert!(
        matches!(le_err, ny_core::NyError::InvalidSpec(ref msg) if msg.contains("out of bounds")),
        "expected InvalidSpec from le_threshold, got {le_err:?}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_layer_gammas_clip() {
    let mut gammas = LayerGammas::new(2, 4, false);

    // Set some negative values
    gammas.gammas[[0, 0, 0]] = -1.0;
    gammas.gammas[[1, 1, 2]] = -0.5;
    gammas.gammas[[0, 1, 3]] = 0.5;

    gammas.clip();

    // Negative values should be clipped to 0
    assert_eq!(gammas.gammas[[0, 0, 0]], 0.0);
    assert_eq!(gammas.gammas[[1, 1, 2]], 0.0);
    // Positive value unchanged
    assert_eq!(gammas.gammas[[0, 1, 3]], 0.5);
}

#[ntest::timeout(5000)]
#[test]
fn test_layer_gammas_shared() {
    let gammas = LayerGammas::new(2, 4, true);
    assert_eq!(gammas.gammas.shape(), &[2, 2, 1]);
    assert!(gammas.shared);

    let expanded = gammas.expand_to(4);
    assert_eq!(expanded.shape(), &[2, 2, 4]);
}

#[ntest::timeout(5000)]
#[test]
fn test_invprop_config_should_apply_to() {
    let config = InvpropConfig {
        enabled: true,
        apply_output_constraints_to: vec!["BoundLinear".to_string(), "/input.7".to_string()],
        ..Default::default()
    };

    // Matches by type prefix
    assert!(config.should_apply_to("layer1", "BoundLinear"));
    assert!(config.should_apply_to("layer2", "BoundLinearAdd"));

    // Matches by name prefix
    assert!(config.should_apply_to("/input.7/dense", "BoundMatMul"));

    // Doesn't match
    assert!(!config.should_apply_to("layer3", "BoundReLU"));
    // Type matches "BoundLinear", so this should apply even if name doesn't match.
    assert!(config.should_apply_to("/input.5", "BoundLinear"));

    // Disabled config
    let disabled_config = InvpropConfig::default();
    assert!(!disabled_config.should_apply_to("layer1", "BoundLinear"));
}

#[ntest::timeout(5000)]
#[test]
fn test_invprop_config_directly_optimize() {
    let config = InvpropConfig {
        enabled: true,
        directly_optimize: vec!["/input".to_string(), "layer1".to_string()],
        ..Default::default()
    };

    assert!(config.should_apply_to("layer1", "BoundReLU"));
    assert!(config.should_apply_to("/input/child", "BoundLinear"));
    assert!(!config.should_apply_to("layer2", "BoundReLU"));
}

#[ntest::timeout(5000)]
#[test]
fn test_invprop_config_should_apply_to_input() {
    let config = InvpropConfig {
        enabled: true,
        tighten_input_bounds: true,
        apply_output_constraints_to: vec!["BoundInput".to_string()],
        ..Default::default()
    };
    assert!(config.should_apply_to_input());

    let all_layers = InvpropConfig {
        enabled: true,
        tighten_input_bounds: true,
        apply_output_constraints_to: vec!["all".to_string()],
        ..Default::default()
    };
    assert!(all_layers.should_apply_to_input());

    let direct = InvpropConfig {
        enabled: true,
        tighten_input_bounds: true,
        directly_optimize: vec!["/input".to_string()],
        ..Default::default()
    };
    assert!(direct.should_apply_to_input());

    let disabled = InvpropConfig {
        enabled: true,
        tighten_input_bounds: false,
        apply_output_constraints_to: vec!["BoundInput".to_string()],
        ..Default::default()
    };
    assert!(!disabled.should_apply_to_input());

    let missing = InvpropConfig {
        enabled: true,
        tighten_input_bounds: true,
        apply_output_constraints_to: vec!["BoundLinear".to_string()],
        ..Default::default()
    };
    assert!(!missing.should_apply_to_input());
}

#[ntest::timeout(5000)]
#[test]
fn test_invprop_config_all_layers() {
    let config = InvpropConfig::all_layers();
    assert!(config.should_apply_to("any_layer", "AnyType"));
}

#[ntest::timeout(5000)]
#[test]
fn test_invprop_state() {
    let constraints =
        OutputConstraints::ge_threshold(4, 0, 0.5).expect("invariant: valid threshold");
    let mut state = InvpropState::new(constraints, 3);

    // Add layer gammas
    state.add_layer_gammas("layer1".to_string(), LayerGammas::new(1, 10, false));
    state.add_layer_gammas("layer2".to_string(), LayerGammas::new(1, 5, true));

    assert!(state.layer_gammas("layer1").is_some());
    assert!(state.layer_gammas("layer2").is_some());
    assert!(state.layer_gammas("layer3").is_none());

    // Test infeasibility
    assert!(!state.is_infeasible(0));
    state
        .mark_infeasible(1)
        .expect("invariant: batch_idx=1 within bounds");
    assert!(state.is_infeasible(1));
    assert!(!state.is_infeasible(2));

    // Test clip
    if let Some(gammas) = state.layer_gammas_mut("layer1") {
        gammas.gammas[[0, 0, 0]] = -1.0;
    }
    state.clip_all_gammas();
    let layer1 = state
        .layer_gammas("layer1")
        .expect("invariant: layer1 was just added");
    assert_eq!(layer1.gammas[[0, 0, 0]], 0.0);
}

/// Regression test (#2712): mark_infeasible returns Err on out-of-bounds batch_idx
/// instead of panicking.
#[ntest::timeout(5000)]
#[test]
fn test_mark_infeasible_out_of_bounds_returns_error_2712() {
    let constraints = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[0.0]), true)
        .expect("invariant: valid constraint matrix");
    let mut state = InvpropState::new(constraints, 2);
    let err = state
        .mark_infeasible(5)
        .expect_err("batch_idx=5 exceeds batch_size=2");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_apply_infeasible_mask_single_batch() {
    let constraints = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[0.0]), true)
        .expect("invariant: valid constraint matrix");
    let mut state = InvpropState::new(constraints, 1);
    state
        .mark_infeasible(0)
        .expect("invariant: batch_idx=0 within bounds");

    let lower = arr1(&[0.0, 1.0]).into_dyn();
    let upper = arr1(&[2.0, 3.0]).into_dyn();
    let mut bounds =
        BoundedTensor::new(lower, upper).expect("invariant: lower <= upper for all elements");

    state.apply_infeasible_mask(&mut bounds);

    assert!(bounds
        .lower()
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_positive()));
    assert!(bounds
        .upper()
        .iter()
        .all(|v| v.is_infinite() && v.is_sign_negative()));
}

#[ntest::timeout(5000)]
#[test]
fn test_ny_params_roundtrip() {
    let constraints =
        OutputConstraints::ge_threshold(4, 0, 0.5).expect("invariant: valid threshold");
    let mut state = InvpropState::new(constraints, 1);

    // Add layer gammas: layer1 has 2 constraints * 3 neurons = 6 gammas per bound = 12 total
    // layer2 has 2 constraints * 1 neuron (shared) = 2 gammas per bound = 4 total
    state.add_layer_gammas("layer1".to_string(), LayerGammas::new(2, 3, false));
    state.add_layer_gammas("layer2".to_string(), LayerGammas::new(2, 5, true));

    // Get initial params (should all be zero)
    let initial_params = state.all_ny_params();
    // layer1: 2 * 2 * 3 = 12, layer2: 2 * 2 * 1 = 4, total = 16
    assert_eq!(initial_params.len(), 16);
    assert!(initial_params.iter().all(|&v| v == 0.0));

    // Set new values and verify round-trip
    let new_params: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    state
        .update_ny_params(&new_params)
        .expect("params length matches");

    let retrieved_params = state.all_ny_params();
    assert_eq!(retrieved_params, new_params);

    // Verify specific values in layers
    let layer1 = state
        .layer_gammas("layer1")
        .expect("invariant: layer1 was just added");
    assert_eq!(layer1.gammas[[0, 0, 0]], 0.0); // First element
    assert_eq!(layer1.gammas[[0, 0, 1]], 0.1); // Second element
    assert_eq!(layer1.gammas[[1, 1, 2]], 1.1); // Index 11 (after 2*3*2 lower bounds)

    let layer2 = state
        .layer_gammas("layer2")
        .expect("invariant: layer2 was just added");
    assert_eq!(layer2.gammas[[0, 0, 0]], 1.2); // Index 12 (first of layer2)

    // Regression test (#2712): wrong-length params returns Err instead of panicking
    let bad_params: Vec<f32> = vec![0.0; 5]; // wrong length (expected 16)
    let result = state.update_ny_params(&bad_params);
    assert!(result.is_err(), "wrong-length params should return Err");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("params length 5"),
        "error should mention actual length: {err_msg}"
    );
}
