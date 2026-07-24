// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

// NOTE: split from tests.rs for maintainability.

use super::prelude::*;

// ==================== Lookahead Optimizer Tests ====================

#[ntest::timeout(5000)]
#[test]
fn test_lookahead_config_default() {
    let config = LookaheadConfig::default();
    assert!(!config.enabled, "Lookahead should be disabled by default");
    assert_eq!(config.sync_period, 5);
    assert!((config.alpha - 0.5).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_lookahead_config_new() {
    let config = LookaheadConfig::new(10, 0.8);
    assert!(config.enabled);
    assert_eq!(config.sync_period, 10);
    assert!((config.alpha - 0.8).abs() < 1e-6);

    // Test clamping of alpha
    let config_high = LookaheadConfig::new(5, 1.5);
    assert!(
        (config_high.alpha - 1.0).abs() < 1e-6,
        "alpha should be clamped to 1.0"
    );

    let config_low = LookaheadConfig::new(5, -0.5);
    assert!(
        (config_low.alpha - 0.0).abs() < 1e-6,
        "alpha should be clamped to 0.0"
    );

    // Test sync_period minimum
    let config_min = LookaheadConfig::new(0, 0.5);
    assert_eq!(
        config_min.sync_period, 1,
        "sync_period should be at least 1"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_lookahead_config_should_sync() {
    let config = LookaheadConfig::new(5, 0.5);

    // Should sync at multiples of 5
    assert!(!config.should_sync(0), "Should not sync at iteration 0");
    assert!(!config.should_sync(1));
    assert!(!config.should_sync(4));
    assert!(config.should_sync(5), "Should sync at iteration 5");
    assert!(!config.should_sync(6));
    assert!(config.should_sync(10), "Should sync at iteration 10");
    assert!(config.should_sync(15), "Should sync at iteration 15");

    // Disabled config should never sync
    let disabled = LookaheadConfig::default();
    assert!(!disabled.should_sync(5));
    assert!(!disabled.should_sync(100));
}

#[ntest::timeout(5000)]
#[test]
fn test_beta_state_lookahead_init() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 2,
        neuron_idx: 1,
        is_active: false,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    assert!(!beta_state.has_slow_weights());

    // Set some values
    beta_state.set_beta(1, 0, 0.5);
    beta_state.set_beta(2, 1, 1.0);

    // Initialize slow weights
    beta_state.init_slow_weights();
    assert!(beta_state.has_slow_weights());

    let slow = beta_state.slow_weights().unwrap();
    assert_eq!(slow.len(), 2);
    assert!((slow[0] - 0.5).abs() < 1e-6);
    assert!((slow[1] - 1.0).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_beta_state_lookahead_step() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(1, 0, 0.0); // Initial fast = slow = 0.0
    beta_state.init_slow_weights();

    let config = LookaheadConfig::new(5, 0.5);

    // Simulate inner optimizer updating fast weights to 1.0
    beta_state.set_beta(1, 0, 1.0);

    // After lookahead step:
    // slow = 0.0 + 0.5 * (1.0 - 0.0) = 0.5
    // fast = slow = 0.5
    assert!(
        beta_state.lookahead_step(&config).is_ok(),
        "lookahead_step should succeed after init_slow_weights"
    );

    let new_value = beta_state.beta(1, 0).unwrap();
    assert!(
        (new_value - 0.5).abs() < 1e-6,
        "Fast weight should be 0.5 after lookahead step, got {}",
        new_value
    );

    let slow = beta_state.slow_weights().unwrap();
    assert!(
        (slow[0] - 0.5).abs() < 1e-6,
        "Slow weight should be 0.5 after lookahead step, got {}",
        slow[0]
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_beta_state_lookahead_convergence() {
    // Test that lookahead converges when fast weights are constant
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(1, 0, 0.0);
    beta_state.init_slow_weights();

    let config = LookaheadConfig::new(1, 0.5);

    // Repeatedly set fast to 1.0 and apply lookahead
    for _ in 0..20 {
        beta_state.set_beta(1, 0, 1.0);
        assert!(
            beta_state.lookahead_step(&config).is_ok(),
            "lookahead_step should succeed after init_slow_weights"
        );
    }

    // Should converge to 1.0
    let final_value = beta_state.beta(1, 0).unwrap();
    assert!(
        (final_value - 1.0).abs() < 0.01,
        "Should converge to target, got {}",
        final_value
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_beta_state_lookahead_non_negative() {
    // Test that beta values remain non-negative after lookahead
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(1, 0, 1.0);
    beta_state.init_slow_weights();

    let config = LookaheadConfig::new(1, 2.0); // alpha > 1 causes overshoot

    // Set fast to negative (would be projected in normal update)
    beta_state.entries[0].value = -0.5;
    assert!(
        beta_state.lookahead_step(&config).is_ok(),
        "lookahead_step should succeed after init_slow_weights"
    );

    // Value should be clamped to 0
    let value = beta_state.beta(1, 0).unwrap();
    assert!(value >= 0.0, "Beta should be non-negative, got {}", value);
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_state_lookahead_init() {
    let mut alpha_state = DomainAlphaState::empty();
    alpha_state
        .neurons
        .insert((1, 0), AlphaNeuronState::new(0.3));
    alpha_state
        .neurons
        .insert((1, 1), AlphaNeuronState::new(0.7));
    alpha_state
        .neurons
        .insert((2, 0), AlphaNeuronState::new(0.5));

    assert!(!alpha_state.has_slow_weights());

    alpha_state.init_slow_weights();
    assert!(alpha_state.has_slow_weights());

    let slow = alpha_state.slow_weights().unwrap();
    assert_eq!(slow.len(), 3);
    assert!((slow[&(1, 0)] - 0.3).abs() < 1e-6);
    assert!((slow[&(1, 1)] - 0.7).abs() < 1e-6);
    assert!((slow[&(2, 0)] - 0.5).abs() < 1e-6);
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_state_lookahead_step() {
    let mut alpha_state = DomainAlphaState::empty();
    alpha_state
        .neurons
        .insert((1, 0), AlphaNeuronState::new(0.0)); // Initial fast = slow = 0.0
    alpha_state.init_slow_weights();

    let config = LookaheadConfig::new(5, 0.5);

    // Simulate inner optimizer updating fast weights to 1.0
    alpha_state.neurons.get_mut(&(1, 0)).unwrap().set_alpha(1.0);

    // After lookahead step:
    // slow = 0.0 + 0.5 * (1.0 - 0.0) = 0.5
    // fast = slow = 0.5
    assert!(
        alpha_state.lookahead_step(&config).is_ok(),
        "lookahead_step should succeed after init_slow_weights"
    );

    let new_value = alpha_state.alpha(1, 0);
    assert!(
        (new_value - 0.5).abs() < 1e-6,
        "Fast weight should be 0.5 after lookahead step, got {}",
        new_value
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_state_lookahead_clamping() {
    // Test that alpha values remain in [0, 1] after lookahead
    let mut alpha_state = DomainAlphaState::empty();
    alpha_state
        .neurons
        .insert((1, 0), AlphaNeuronState::new(0.9));
    alpha_state.init_slow_weights();

    let config = LookaheadConfig::new(1, 0.5);

    // Set fast to value > 1
    alpha_state.neurons.get_mut(&(1, 0)).unwrap().alpha = 1.5;
    assert!(
        alpha_state.lookahead_step(&config).is_ok(),
        "lookahead_step should succeed after init_slow_weights"
    );

    // Value should be clamped to 1.0
    let value = alpha_state.alpha(1, 0);
    assert!(value <= 1.0, "Alpha should be at most 1.0, got {}", value);

    // Set fast to value < 0
    alpha_state.neurons.get_mut(&(1, 0)).unwrap().alpha = -0.5;
    assert!(
        alpha_state.lookahead_step(&config).is_ok(),
        "lookahead_step should succeed after init_slow_weights"
    );

    // Value should be clamped to 0.0
    let value = alpha_state.alpha(1, 0);
    assert!(value >= 0.0, "Alpha should be at least 0.0, got {}", value);
}

#[ntest::timeout(5000)]
#[test]
fn test_adaptive_opt_config_has_lookahead() {
    let config = AdaptiveOptConfig::default();
    assert!(
        !config.lookahead.enabled,
        "Lookahead should be disabled by default"
    );

    let config_with_lookahead = AdaptiveOptConfig {
        lookahead: LookaheadConfig::new(5, 0.5),
        ..Default::default()
    };
    assert!(config_with_lookahead.lookahead.enabled);
}

#[ntest::timeout(5000)]
#[test]
fn test_lookahead_beta_with_adam() {
    // Test full lookahead integration with Adam optimizer
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let config = AdaptiveOptConfig {
        beta_lr: 0.5,
        lookahead: LookaheadConfig::new(3, 0.5),
        ..Default::default()
    };

    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.init_slow_weights();

    // Run 3 Adam steps then lookahead
    for t in 1..=3 {
        beta_state.zero_grad();
        beta_state.accumulate_grad(1, 0, 1.0);
        beta_state.gradient_step_adam(&config, t);
    }

    let value_before_lookahead = beta_state.beta(1, 0).unwrap();

    // Apply lookahead at sync point
    assert!(config.lookahead.should_sync(3));
    assert!(
        beta_state.lookahead_step(&config.lookahead).is_ok(),
        "lookahead_step should succeed after init_slow_weights"
    );

    let value_after_lookahead = beta_state.beta(1, 0).unwrap();

    // After lookahead, value should be interpolated between slow (0) and fast
    assert!(
        value_after_lookahead < value_before_lookahead,
        "Lookahead should pull back toward slow weights: {} < {}",
        value_after_lookahead,
        value_before_lookahead
    );
    assert!(
        value_after_lookahead > 0.0,
        "Value should be positive after interpolation"
    );

    println!(
        "Adam+Lookahead: before={:.6}, after={:.6}",
        value_before_lookahead, value_after_lookahead
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_lookahead_alpha_with_adam() {
    // Test full lookahead integration with Adam optimizer for alpha
    let mut alpha_state = DomainAlphaState::empty();
    alpha_state
        .neurons
        .insert((1, 0), AlphaNeuronState::new(0.5));
    alpha_state.init_slow_weights();

    let config = AdaptiveOptConfig {
        alpha_lr: 0.5,
        lookahead: LookaheadConfig::new(3, 0.5),
        ..Default::default()
    };

    // Run 3 Adam steps with positive gradient (pushing toward 1.0)
    for t in 1..=3 {
        alpha_state.zero_grad();
        alpha_state.neurons.get_mut(&(1, 0)).unwrap().grad = 0.5; // Positive gradient
        alpha_state.gradient_step_adam(&config, t);
    }

    let value_before_lookahead = alpha_state.alpha(1, 0);

    // Apply lookahead at sync point
    assert!(config.lookahead.should_sync(3));
    assert!(
        alpha_state.lookahead_step(&config.lookahead).is_ok(),
        "lookahead_step should succeed after init_slow_weights"
    );

    let value_after_lookahead = alpha_state.alpha(1, 0);

    // After lookahead, value should be interpolated between slow (0.5) and fast
    assert!(
        (value_after_lookahead - 0.5).abs() < (value_before_lookahead - 0.5).abs(),
        "Lookahead should pull toward initial slow weight: |{} - 0.5| < |{} - 0.5|",
        value_after_lookahead,
        value_before_lookahead
    );

    println!(
        "Alpha Adam+Lookahead: before={:.6}, after={:.6}",
        value_before_lookahead, value_after_lookahead
    );
}
