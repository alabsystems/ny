// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_split_history() {
    let mut history = SplitHistory::new();
    assert_eq!(history.depth(), 0);
    assert!(history.is_constrained(1, 0).is_none());

    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    assert_eq!(history.depth(), 1);
    assert_eq!(history.is_constrained(1, 0), Some(true));
    assert!(history.is_constrained(1, 1).is_none());
}

#[ntest::timeout(10000)]
#[test]
fn test_constraint_tightens_bounds() {
    // Test that adding a constraint tightens the bounds
    let mut history = SplitHistory::new();

    // Add constraint that neuron 0 in layer 1 is active
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let beta_state = BetaState::from_history(&history).unwrap();
    assert_eq!(beta_state.len(), 1);

    // Check that the beta entry has correct sign
    let entry = beta_state.entry(1, 0).unwrap();
    assert_eq!(entry.sign, 1.0); // Active constraint has positive sign
    assert_eq!(entry.value(), 0.0); // Initial beta is 0
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_state_gradient_step() {
    // Test beta gradient step with projection to non-negative
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();

    // Set some values
    beta_state.entries[0].value = 0.5;
    beta_state.entries[1].value = 0.1;

    // Set gradients
    beta_state.entries[0].grad = 0.2;
    beta_state.entries[1].grad = -0.3; // Negative gradient should trigger projection

    // Perform gradient step with lr=1.0
    let max_grad = beta_state.gradient_step(1.0);

    // Check projection
    assert_eq!(beta_state.entries[0].value, 0.7); // 0.5 + 1.0 * 0.2
    assert_eq!(beta_state.entries[1].value, 0.0); // max(0, 0.1 + 1.0 * -0.3) = max(0, -0.2) = 0

    // Check max gradient magnitude
    assert!((max_grad - 0.3).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_state_inactive_constraint_sign() {
    // Test that inactive constraints have negative sign
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: false,
        score: 0.0,
    });

    let beta_state = BetaState::from_history(&history).unwrap();
    let entry = beta_state.entry(1, 0).unwrap();
    assert_eq!(entry.sign, -1.0); // Inactive constraint has negative sign
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_signed_contribution() {
    // Test that get_signed_beta returns value * sign
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(1, 0, 2.0);
    beta_state.set_beta(1, 1, 3.0);

    // Active: sign=+1, so signed_beta = 2.0 * 1.0 = 2.0
    assert_eq!(beta_state.signed_beta(1, 0), Some(2.0));

    // Inactive: sign=-1, so signed_beta = 3.0 * -1.0 = -3.0
    assert_eq!(beta_state.signed_beta(1, 1), Some(-3.0));
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_state_adam_update() {
    // Test that Adam optimizer updates β correctly against analytical trajectory.
    // Default config: beta1=0.9, beta2=0.999, beta_lr=0.05, bias_correction=true.
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig::default();

    // Set a gradient
    beta_state.accumulate_grad(1, 0, 1.0);

    // Step 1: verify against analytical Adam formulas.
    // m = (1 - 0.9) * 1.0 = 0.1
    // v = (1 - 0.999) * 1.0^2 = 0.001
    // m_hat = 0.1 / (1 - 0.9^1) = 0.1 / 0.1 = 1.0
    // v_hat = 0.001 / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
    // update = 0.05 * 1.0 / (sqrt(1.0) + 1e-8) ≈ 0.05
    // value = max(0, 0 + 0.05) = 0.05
    beta_state.gradient_step_adam(&config, 1);

    let m_1 = beta_state.entries[0].m;
    let v_1 = beta_state.entries[0].v;
    let val_1 = beta_state.entries[0].value;

    assert!(
        (m_1 - 0.1).abs() < 1e-6,
        "Step 1: m should be 0.1, got {m_1}"
    );
    assert!(
        (v_1 - 0.001).abs() < 1e-7,
        "Step 1: v should be 0.001, got {v_1}"
    );
    assert!(
        (val_1 - 0.05).abs() < 1e-5,
        "Step 1: beta should be ~0.05, got {val_1}"
    );

    // Continue for several steps, verifying monotonic increase with constant gradient
    beta_state.zero_grad();
    beta_state.accumulate_grad(1, 0, 1.0);
    beta_state.gradient_step_adam(&config, 2);
    let val_2 = beta_state.entries[0].value;
    assert!(
        val_2 > val_1,
        "Step 2: beta should increase: {val_2} > {val_1}"
    );

    for t in 3..=5 {
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 1.0);
        beta_state.gradient_step_adam(&config, t);
    }
    let val_5 = beta_state.entries[0].value;
    assert!(
        val_5 > val_2,
        "Step 5: beta should keep increasing: {val_5} > {val_2}"
    );

    // After 5 steps with constant grad=1, first moment should be near
    // m_5 = 0.9^4 * 0.1 + 0.1 * (0.9^3 + 0.9^2 + 0.9 + 1) = 1 - 0.9^5 ≈ 0.40951
    let expected_m5 = 1.0 - 0.9f32.powi(5);
    assert!(
        (beta_state.entries[0].m - expected_m5).abs() < 1e-5,
        "Step 5: m should be ~{expected_m5}, got {}",
        beta_state.entries[0].m
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_gradient_clipping() {
    // Test that gradient clipping bounds effective gradient to ±clip_value.
    // With grad_clip=1.0, a gradient of 100.0 should be clipped to 1.0,
    // producing the same Adam state as if the gradient were 1.0.
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig {
        grad_clip: 1.0, // Clip to ±1
        ..Default::default()
    };

    // Set a large gradient that should be clipped
    beta_state.accumulate_grad(1, 0, 100.0);

    // Perform Adam step
    beta_state.gradient_step_adam(&config, 1);

    // With clipping to 1.0: m = (1 - 0.9) * 1.0 = 0.1
    assert!(
        (beta_state.entries[0].m - 0.1).abs() < 1e-6,
        "m should be 0.1 (clipped grad), got {}",
        beta_state.entries[0].m
    );

    // v = (1 - 0.999) * 1.0^2 = 0.001
    assert!(
        (beta_state.entries[0].v - 0.001).abs() < 1e-7,
        "v should be 0.001 (clipped grad squared), got {}",
        beta_state.entries[0].v
    );

    // Compare with unclipped: build a second state with grad=1.0 (no clipping needed)
    let mut ref_state = BetaState::from_history(&history).unwrap();
    ref_state.accumulate_grad(1, 0, 1.0);
    ref_state.gradient_step_adam(&config, 1);

    // Clipped-100 should produce identical state to unclipped-1.0
    assert!(
        (beta_state.entries[0].value - ref_state.entries[0].value).abs() < 1e-7,
        "Clipped grad=100 should match grad=1: {} vs {}",
        beta_state.entries[0].value,
        ref_state.entries[0].value,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_bias_correction() {
    // Test that bias correction produces different updates than without
    // (The specific relationship depends on the relative values of β₁ and β₂)
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    // Without bias correction
    let mut beta_no_correction = BetaState::from_history(&history).unwrap();
    let config_no_correction = AdaptiveOptConfig {
        bias_correction: false,
        ..Default::default()
    };
    beta_no_correction.accumulate_grad(1, 0, 1.0);
    beta_no_correction.gradient_step_adam(&config_no_correction, 1);
    let value_no_correction = beta_no_correction.entries[0].value;
    let m_no = beta_no_correction.entries[0].m;
    let v_no = beta_no_correction.entries[0].v;

    // With bias correction
    let mut beta_with_correction = BetaState::from_history(&history).unwrap();
    let config_with_correction = AdaptiveOptConfig {
        bias_correction: true,
        ..Default::default()
    };
    beta_with_correction.accumulate_grad(1, 0, 1.0);
    beta_with_correction.gradient_step_adam(&config_with_correction, 1);
    let value_with_correction = beta_with_correction.entries[0].value;
    let m_with = beta_with_correction.entries[0].m;
    let v_with = beta_with_correction.entries[0].v;

    println!(
        "Without bias correction: beta={:.6}, m={:.6}, v={:.9}",
        value_no_correction, m_no, v_no
    );
    println!(
        "With bias correction:    beta={:.6}, m={:.6}, v={:.9}",
        value_with_correction, m_with, v_with
    );

    // Raw m and v should be the same (bias correction only affects effective values)
    assert!(
        (m_no - m_with).abs() < 1e-6,
        "Raw first moment should be the same"
    );
    assert!(
        (v_no - v_with).abs() < 1e-9,
        "Raw second moment should be the same"
    );

    // The updates should be different due to bias correction scaling
    // Note: With Adam's default parameters, bias correction actually produces
    // smaller updates in iteration 1 because v_hat = v/(1-β₂^t) grows faster
    // than m_hat = m/(1-β₁^t), leading to larger denominator
    assert!(
        (value_no_correction - value_with_correction).abs() > 1e-6,
        "Bias correction should produce different update"
    );

    // Both updates should be positive (gradient is positive)
    assert!(value_no_correction > 0.0, "Update should be positive");
    assert!(value_with_correction > 0.0, "Update should be positive");
}

/// Regression test for #2575/#2586: BetaState Adam must not produce NaN/Inf
/// when beta1=1.0 (disables momentum — valid config).
#[ntest::timeout(10000)]
#[test]
fn test_beta_state_adam_beta1_one_no_div_by_zero_2575() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.accumulate_grad(1, 0, 1.0);

    let config = AdaptiveOptConfig {
        beta1: 1.0,
        bias_correction: true,
        ..Default::default()
    };
    let max_grad = beta_state.gradient_step_adam(&config, 1);

    assert!(
        max_grad.is_finite(),
        "max_grad should be finite with beta1=1.0, got {}",
        max_grad
    );
    assert!(
        beta_state.entries[0].value.is_finite(),
        "beta value should be finite with beta1=1.0, got {}",
        beta_state.entries[0].value
    );
}

/// Regression test for #2575/#2586: BetaState Adam must not produce NaN/Inf
/// when beta2=1.0.
#[ntest::timeout(10000)]
#[test]
fn test_beta_state_adam_beta2_one_no_div_by_zero_2575() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.accumulate_grad(1, 0, 1.0);

    let config = AdaptiveOptConfig {
        beta2: 1.0,
        bias_correction: true,
        ..Default::default()
    };
    let max_grad = beta_state.gradient_step_adam(&config, 1);

    assert!(
        max_grad.is_finite(),
        "max_grad should be finite with beta2=1.0, got {}",
        max_grad
    );
    assert!(
        beta_state.entries[0].value.is_finite(),
        "beta value should be finite with beta2=1.0, got {}",
        beta_state.entries[0].value
    );
}

/// Test for #3112: NaN gradients are filtered at accumulate_grad gate.
/// Previously (#2596), NaN entered the optimizer and was cleaned up post-step.
/// Now, accumulate_grad skips NaN entirely — optimizer state is never corrupted.
#[ntest::timeout(10000)]
#[test]
fn test_beta_state_adam_nan_gradient_filtered_at_gate_3112() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig::default();

    // First: normal gradient to establish non-zero state
    beta_state.accumulate_grad(1, 0, 1.0);
    beta_state.gradient_step_adam(&config, 1);
    assert!(
        beta_state.entries[0].value > 0.0,
        "beta should be positive after normal step"
    );

    // Second: inject NaN gradient — filtered at the gate by #3112.
    beta_state.zero_grad();
    beta_state.accumulate_grad(1, 0, f32::NAN);
    assert_eq!(
        beta_state.entries[0].grad, 0.0,
        "NaN gradient should be silently filtered by accumulate_grad"
    );
    beta_state.gradient_step_adam(&config, 2);

    // Optimizer state must remain finite — NaN never entered
    let entry = &beta_state.entries[0];
    assert!(
        entry.value().is_finite(),
        "beta should be finite after NaN-filtered step"
    );
    assert!(entry.m.is_finite(), "first moment should be finite");
    assert!(entry.v.is_finite(), "second moment should be finite");
    assert!(entry.v_max.is_finite(), "v_max should be finite");
    // Beta value preserved (not reset to 0) — the NaN was filtered, not cleaned up
    assert!(
        entry.value() > 0.0,
        "beta should remain positive (NaN filtered, not reset)"
    );
}
