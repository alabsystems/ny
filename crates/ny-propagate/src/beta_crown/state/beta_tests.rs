// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for sequential network β state.

use super::*;
use crate::beta_crown::branching::NeuronConstraint;

/// Regression test for #2880: lookahead_step must return Err, not panic,
/// when slow weights length diverges from entries length.
#[test]
fn test_lookahead_step_length_mismatch_returns_err_2880() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();
    state.init_slow_weights();

    // Add a second entry after init, creating length mismatch
    state
        .entries
        .push(BetaEntry::new(1, 0, 0.0, 1.0).expect("test: valid BetaEntry params"));

    let config = LookaheadConfig {
        enabled: true,
        alpha: 0.5,
        sync_period: 5,
    };
    let result = state.lookahead_step(&config);
    assert!(
        result.is_err(),
        "should return Err on length mismatch, not panic"
    );
}

/// Regression test for #2939: standard (non-Adam) gradient_step must recover
/// from NaN gradients instead of permanently corrupting beta values.
///
/// Before the fix, `nan_propagating_max(NaN + lr*NaN, 0.0) = NaN` left
/// `entry.value` permanently NaN with no recovery path (unlike the Adam path
/// which resets m/v/v_max on NaN detection).
#[test]
fn test_gradient_step_nan_recovery_2939() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: false,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();

    // Set one neuron with a valid gradient, one with NaN
    state.entries[0].set_value(0.5);
    state.entries[0].grad = 0.1;
    state.entries[1].set_value(0.3);
    state.entries[1].grad = f32::NAN;

    let max_grad = state.gradient_step(0.01);

    // The NaN-infected entry must be reset to 0.0, not left as NaN
    assert!(
        state.entries[1].value().is_finite(),
        "beta value must be finite after NaN gradient, got {}",
        state.entries[1].value()
    );
    assert_eq!(
        state.entries[1].value(),
        0.0,
        "NaN-infected beta must reset to 0.0"
    );
    assert_eq!(
        state.entries[1].grad(),
        0.0,
        "NaN-infected grad must reset to 0.0"
    );

    // The valid entry must still be updated normally
    assert!(
        state.entries[0].value() > 0.5,
        "valid beta should increase with positive gradient, got {}",
        state.entries[0].value()
    );

    // max_grad must be NaN (NaN-aware tracking propagates NaN)
    assert!(
        max_grad.is_nan(),
        "max_grad should be NaN when any gradient is NaN (nan_propagating_max), got {max_grad}"
    );
}

/// Regression test for #2939: after NaN recovery, the next gradient step must
/// work normally (the corruption is not permanent).
#[test]
fn test_gradient_step_nan_recovery_then_normal_2939() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();

    // Step 1: NaN gradient corrupts the entry
    state.entries[0].set_value(0.5);
    state.entries[0].grad = f32::NAN;
    let _ = state.gradient_step(0.01);

    // After recovery, value should be 0.0
    assert_eq!(
        state.entries[0].value(),
        0.0,
        "post-recovery beta must be 0.0"
    );

    // Step 2: valid gradient — should work normally
    state.entries[0].grad = 1.0;
    let max_grad = state.gradient_step(0.1);

    assert!(
        state.entries[0].value() > 0.0,
        "beta should increase with positive gradient after recovery, got {}",
        state.entries[0].value()
    );
    assert!(
        max_grad.is_finite(),
        "max_grad should be finite with valid gradient, got {max_grad}"
    );
}

/// Regression test for #2416: verify convergence check doesn't terminate prematurely
/// after a flat-then-steep gradient sequence. With the old m_hat-based check, the EMA
/// (β₁=0.9) would lag behind — after 10 near-zero gradient steps, a large gradient
/// would still show m_hat ≈ 0, causing premature convergence. With raw gradient, the
/// large gradient is immediately reflected.
#[test]
fn test_gradient_step_adam_flat_then_steep_no_premature_convergence_2416() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig::default();

    // Phase 1: 10 iterations of near-zero gradient (flat region).
    // This drives m_hat toward zero.
    for t in 1..=10 {
        state.entries[0].grad = 1e-8;
        let max_grad = state.gradient_step_adam(&config, t);
        // Raw gradient is near-zero, so max_grad should be near-zero.
        assert!(
            max_grad < 1e-6,
            "flat phase: max_grad should be near-zero, got {max_grad}"
        );
    }

    // Phase 2: Sudden large gradient (steep region).
    // With m_hat (EMA, β₁=0.9), m_hat ≈ 0.9^1 * old_m + 0.1 * 5.0 ≈ 0.5.
    // With bias correction at t=11: m_hat = m / (1 - 0.9^11) ≈ m / 0.686.
    // But the raw gradient is 5.0 — much larger than what m_hat would show.
    state.entries[0].grad = 5.0;
    let max_grad = state.gradient_step_adam(&config, 11);

    // With raw gradient semantics (#2416), max_grad = |5.0| = 5.0.
    // With old m_hat semantics, max_grad would be ≈ 0.73 (far below 5.0).
    assert!(
        (max_grad - 5.0).abs() < 1e-5,
        "steep phase: max_grad should be 5.0 (raw gradient), got {max_grad}"
    );
}

/// Regression test for #2939: convergence tracking with nan_propagating_max
/// must surface NaN gradients instead of silently dropping them via f32::max.
#[test]
fn test_gradient_step_adam_nan_convergence_tracking_2939() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();
    state.entries[0].grad = f32::NAN;

    let config = AdaptiveOptConfig::default();
    let max_grad = state.gradient_step_adam(&config, 1);

    // With nan_propagating_max, NaN grad → NaN grad.abs() → NaN max_grad (#2416, #2939).
    // Before the fix, f32::max(0.0, NaN) = 0.0, silently hiding the NaN.
    assert!(
        max_grad.is_nan(),
        "max_grad must be NaN when gradient is NaN (nan_propagating_max), got {max_grad}"
    );
}

#[test]
fn test_beta_state_rebuilds_lookup_after_entries_push_2936() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();
    state
        .entries
        .push(BetaEntry::new(1, 0, 0.25, -1.0).expect("valid beta entry"));

    let pushed = state.entry(1, 0).expect("pushed entry must be found");
    assert!((pushed.value() - 0.25).abs() < 1e-6);
    assert_eq!(pushed.sign(), -1.0);

    state.set_beta(1, 0, 0.75);
    assert_eq!(state.beta(1, 0), Some(0.75));
    assert_eq!(state.signed_beta(1, 0), Some(-0.75));

    let layer_entries: Vec<_> = state
        .entries_for_layer(1)
        .map(|entry| (entry.layer_idx(), entry.neuron_idx()))
        .collect();
    assert_eq!(layer_entries, vec![(1, 0)]);
}

// ── BetaEntry::new value validation ──────────────────────────────────────────

/// BetaEntry::new must clamp NaN initial values to 0.0 (lines 53-57 of beta.rs).
/// Without this guard, a NaN beta value would poison all downstream arithmetic.
#[test]
fn test_beta_entry_new_nan_value_clamped_to_zero() {
    let entry =
        BetaEntry::new(0, 0, f32::NAN, 1.0).expect("NaN value should be clamped, not rejected");
    assert_eq!(
        entry.value(),
        0.0,
        "NaN initial value must be clamped to 0.0"
    );
    assert!(entry.value().is_finite());
}

/// BetaEntry::new must clamp positive Inf initial values to 0.0.
#[test]
fn test_beta_entry_new_inf_value_clamped_to_zero() {
    let entry = BetaEntry::new(0, 0, f32::INFINITY, 1.0)
        .expect("Inf value should be clamped, not rejected");
    assert_eq!(
        entry.value(),
        0.0,
        "Inf initial value must be clamped to 0.0"
    );
}

/// BetaEntry::new must clamp negative Inf initial values to 0.0.
#[test]
fn test_beta_entry_new_neg_inf_value_clamped_to_zero() {
    let entry = BetaEntry::new(0, 0, f32::NEG_INFINITY, 1.0)
        .expect("-Inf value should be clamped, not rejected");
    assert_eq!(
        entry.value(),
        0.0,
        "-Inf initial value must be clamped to 0.0"
    );
}

/// BetaEntry::new must clamp negative initial values to 0.0 (β ≥ 0 invariant).
#[test]
fn test_beta_entry_new_negative_value_clamped_to_zero() {
    let entry =
        BetaEntry::new(0, 0, -0.5, 1.0).expect("negative value should be clamped, not rejected");
    assert_eq!(
        entry.value(),
        0.0,
        "negative initial value must be clamped to 0.0"
    );
}

/// BetaEntry::new must accept valid non-negative finite values.
#[test]
fn test_beta_entry_new_valid_value_preserved() {
    let entry = BetaEntry::new(0, 0, 1.5, 1.0).expect("valid value should be accepted");
    assert!(
        (entry.value() - 1.5).abs() < 1e-7,
        "valid value must be preserved"
    );
}

/// BetaEntry::new must reject invalid sign values (not ±1.0).
#[test]
fn test_beta_entry_new_invalid_sign_rejected() {
    let err = BetaEntry::new(0, 0, 0.0, 0.5);
    assert!(err.is_err(), "sign=0.5 must be rejected");
    let err = BetaEntry::new(0, 0, 0.0, 0.0);
    assert!(err.is_err(), "sign=0.0 must be rejected");
    let err = BetaEntry::new(0, 0, 0.0, f32::NAN);
    assert!(err.is_err(), "sign=NaN must be rejected");
}

// ── set_value NaN guard ──────────────────────────────────────────────────────

/// set_value must clamp NaN to 0.0 (line ~97 of beta.rs).
/// This guard is defense-in-depth: optimizer post-step guards catch most NaN,
/// but external callers of set_value (e.g., warm-start from file) could pass NaN.
#[test]
fn test_set_value_nan_clamped_to_zero() {
    let mut entry = BetaEntry::new(0, 0, 1.0, 1.0).expect("valid entry");
    assert!((entry.value() - 1.0).abs() < 1e-7);

    entry.set_value(f32::NAN);
    assert_eq!(entry.value(), 0.0, "set_value(NaN) must clamp to 0.0");
    assert!(entry.value().is_finite());
}

/// set_value must clamp positive Inf to 0.0.
#[test]
fn test_set_value_inf_clamped_to_zero() {
    let mut entry = BetaEntry::new(0, 0, 1.0, 1.0).expect("valid entry");
    entry.set_value(f32::INFINITY);
    assert_eq!(entry.value(), 0.0, "set_value(Inf) must clamp to 0.0");
}

/// set_value must clamp negative values to 0.0 (β ≥ 0 invariant).
#[test]
fn test_set_value_negative_clamped_to_zero() {
    let mut entry = BetaEntry::new(0, 0, 1.0, 1.0).expect("valid entry");
    entry.set_value(-0.1);
    assert_eq!(entry.value(), 0.0, "set_value(-0.1) must clamp to 0.0");
}

/// set_value must accept valid non-negative finite values.
#[test]
fn test_set_value_valid_accepted() {
    let mut entry = BetaEntry::new(0, 0, 0.0, 1.0).expect("valid entry");
    entry.set_value(2.5);
    assert!(
        (entry.value() - 2.5).abs() < 1e-7,
        "valid set_value must be accepted"
    );
}

// ── gradient_step_adam post-step NaN guard (defense-in-depth) ─────────────────

/// The post-step NaN guard in gradient_step_adam (line ~399) must reset m/v/v_max
/// when NaN enters the optimizer state. This tests the guard in isolation by
/// setting entry.grad directly (bypassing the accumulate_grad NaN filter).
///
/// Without this guard, a single NaN that bypasses the accumulate_grad filter
/// would permanently poison the Adam optimizer state (m, v, v_max all become NaN).
#[test]
fn test_gradient_step_adam_post_step_nan_guard_resets_optimizer_state() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();

    // Bypass accumulate_grad by setting grad directly on the pub(crate) field.
    // This simulates a hypothetical path where NaN reaches the optimizer
    // without being filtered by accumulate_grad.
    state.entries[0].grad = f32::NAN;

    let config = AdaptiveOptConfig::default();
    let _max_grad = state.gradient_step_adam(&config, 1);

    // Post-step guard must reset all optimizer state to zero
    assert_eq!(
        state.entries[0].value(),
        0.0,
        "post-step NaN guard must reset value to 0.0"
    );
    assert_eq!(
        state.entries[0].m, 0.0,
        "post-step NaN guard must reset m to 0.0"
    );
    assert_eq!(
        state.entries[0].v, 0.0,
        "post-step NaN guard must reset v to 0.0"
    );
    assert_eq!(
        state.entries[0].v_max, 0.0,
        "post-step NaN guard must reset v_max to 0.0"
    );
}

/// After NaN recovery via the post-step guard, the next Adam step must work normally.
#[test]
fn test_gradient_step_adam_post_nan_recovery_normal_step() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig::default();

    // Step 1: inject NaN directly, trigger post-step guard
    state.entries[0].grad = f32::NAN;
    let _ = state.gradient_step_adam(&config, 1);
    assert_eq!(state.entries[0].value(), 0.0);

    // Step 2: valid gradient — optimizer should recover
    state.entries[0].grad = 1.0;
    let max_grad = state.gradient_step_adam(&config, 2);

    assert!(
        state.entries[0].value() > 0.0,
        "beta should increase with positive gradient after NaN recovery, got {}",
        state.entries[0].value()
    );
    assert!(
        state.entries[0].value().is_finite(),
        "recovered beta must be finite"
    );
    assert!(
        max_grad.is_finite() && max_grad > 0.0,
        "max_grad should be finite and positive after recovery, got {max_grad}"
    );
}

/// The default AdaptiveOptConfig has grad_clip=10.0, which clips Inf to 10.0
/// (a finite value). To exercise the post-step NaN guard with Inf, we must
/// disable gradient clipping (grad_clip=0.0) so Inf flows through to the
/// Adam update and produces non-finite m/v.
#[test]
fn test_gradient_step_adam_inf_gradient_no_clip_resets_state() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig {
        grad_clip: 0.0, // Disable clipping so Inf reaches the update
        ..AdaptiveOptConfig::default()
    };

    // Inject Inf gradient directly (bypasses accumulate_grad filter too)
    state.entries[0].grad = f32::INFINITY;
    let _ = state.gradient_step_adam(&config, 1);

    // Without clipping: m = 0.1 * Inf = Inf, v = 0.001 * Inf^2 = Inf
    // Post-step guard fires: !m.is_finite() → reset all to zero
    assert_eq!(
        state.entries[0].value(),
        0.0,
        "Inf gradient (no clip) must reset value to 0.0, got {}",
        state.entries[0].value()
    );
    assert_eq!(
        state.entries[0].m, 0.0,
        "Inf gradient (no clip) must reset m to 0.0, got {}",
        state.entries[0].m
    );
    assert_eq!(
        state.entries[0].v, 0.0,
        "Inf gradient (no clip) must reset v to 0.0, got {}",
        state.entries[0].v
    );
}

// ── Performance proofs: O(1) indexed lookups at BaB scale (#2936) ─────────────

/// Performance proof: BetaState index covers all entries after construction,
/// guaranteeing O(1) `entry()` lookups at BaB-realistic scale (500 constraints).
///
/// Without indexed lookups, each `entry(layer_idx, neuron_idx)` call scans all
/// B entries → O(B) per call × O(R) ReLU neurons = O(R*B) per backward pass.
/// With the index, each call is O(1) via HashMap.
///
/// This test verifies the structural invariant: `indexed_entries == entries.len()`
/// holds after `from_history()`, meaning the O(1) path is always taken.
/// Tracks: #2936 Finding 5.
#[test]
fn test_beta_state_index_fresh_at_scale_2936() {
    let n = 500;
    let mut history = SplitHistory::new();
    for i in 0..n {
        history.add_constraint(NeuronConstraint {
            layer_idx: i / 10,
            neuron_idx: i % 10,
            is_active: i % 2 == 0,
            score: 0.0,
        });
    }
    let state = BetaState::from_history(&history).unwrap();

    // Index must be fresh — no linear scan fallback.
    assert_eq!(
        state.indexed_entries,
        state.entries.len(),
        "index must cover all {} entries after from_history",
        n
    );

    // Every entry must be reachable via O(1) indexed lookup.
    for entry in &state.entries {
        let found = state
            .entry(entry.layer_idx(), entry.neuron_idx())
            .expect("every entry must be found via indexed lookup");
        assert_eq!(found.layer_idx(), entry.layer_idx());
        assert_eq!(found.neuron_idx(), entry.neuron_idx());
    }

    // entries_for_layer must return correct count for each layer.
    let num_layers = n / 10;
    for layer_idx in 0..num_layers {
        let count = state.entries_for_layer(layer_idx).count();
        assert_eq!(
            count, 10,
            "layer {layer_idx} should have 10 entries, got {count}"
        );
    }
}

/// Performance proof: `entries_for_layer` uses the layer_index HashMap at scale,
/// returning only the entries for that layer in O(k) where k is the per-layer
/// count, not O(B) scanning all entries.
///
/// Verifies that the per-layer grouping is correct across 50 layers × 10 neurons.
/// Tracks: #2936 Finding 5.
#[test]
fn test_beta_state_entries_for_layer_indexed_correctness_2936() {
    let n = 500;
    let mut history = SplitHistory::new();
    for i in 0..n {
        history.add_constraint(NeuronConstraint {
            layer_idx: i / 10,
            neuron_idx: i % 10,
            is_active: true,
            score: 0.0,
        });
    }
    let state = BetaState::from_history(&history).unwrap();

    // Verify indexed path is active.
    assert_eq!(state.indexed_entries, state.entries.len());

    // Cross-check: indexed entries_for_layer matches linear-scan filter.
    for layer_idx in 0..50 {
        let indexed: Vec<usize> = state
            .entries_for_layer(layer_idx)
            .map(|e| e.neuron_idx())
            .collect();
        let linear: Vec<usize> = state
            .entries
            .iter()
            .filter(|e| e.layer_idx() == layer_idx)
            .map(|e| e.neuron_idx())
            .collect();
        assert_eq!(
            indexed, linear,
            "layer {layer_idx}: indexed and linear scan must agree"
        );
    }
}

/// With default grad_clip=10.0, Inf gradients are clipped to finite values
/// and do NOT trigger the post-step NaN guard. This verifies the clipping
/// defense works correctly — the optimizer state remains finite and the
/// beta value advances normally.
#[test]
fn test_gradient_step_adam_inf_gradient_clipped_to_finite() {
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 0,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut state = BetaState::from_history(&history).unwrap();
    let config = AdaptiveOptConfig::default(); // grad_clip=10.0

    state.entries[0].grad = f32::INFINITY;
    let _ = state.gradient_step_adam(&config, 1);

    // Inf.clamp(-10, 10) = 10.0, so update proceeds normally
    assert!(
        state.entries[0].value().is_finite(),
        "clipped Inf gradient should produce finite beta, got {}",
        state.entries[0].value()
    );
    assert!(
        state.entries[0].value() > 0.0,
        "clipped Inf gradient should produce positive beta, got {}",
        state.entries[0].value()
    );
    assert!(
        state.entries[0].m.is_finite(),
        "clipped Inf gradient should keep m finite, got {}",
        state.entries[0].m
    );
}
