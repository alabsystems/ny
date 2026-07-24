// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NaN abort counter and bounded-evaluation regression tests (#2968).

use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;

use crate::layers::*;
use crate::pgd_attack::attacker::PgdAttacker;
use crate::pgd_attack::config::PgdConfig;
use crate::Network;

/// Simulate the NaN-skip counter loop from `run_single_restart` with an
/// injected gradient sequence. Returns the number of gradient steps actually
/// executed (not skipped) and the total step iterations consumed (including
/// NaN-skipped steps).
///
/// This isolates the NaN abort logic (#2968) for deterministic testing without
/// needing a network that naturally produces NaN gradients.
fn simulate_nan_abort_loop(
    num_steps: usize,
    gradient_sequence: &[bool], // true = NaN gradient, false = clean gradient
) -> (usize, usize) {
    let mut consecutive_nan_skips: u32 = 0;
    const MAX_CONSECUTIVE_NAN_SKIPS: u32 = 5;
    let mut clean_steps = 0;
    let mut total_iterations = 0;

    for step in 0..num_steps {
        total_iterations += 1;
        let is_nan = gradient_sequence.get(step).copied().unwrap_or(false);

        if is_nan {
            consecutive_nan_skips += 1;
            if consecutive_nan_skips >= MAX_CONSECUTIVE_NAN_SKIPS {
                break;
            }
            continue;
        }
        consecutive_nan_skips = 0;
        clean_steps += 1;
    }

    (clean_steps, total_iterations)
}

// --- NaN abort counter regression tests (#2968) ---

/// Regression test for #2968: 5 consecutive NaN gradients trigger abort.
#[test]
fn test_nan_abort_five_consecutive_nans_triggers_abort() {
    // All NaN gradients: should abort after exactly 5 steps.
    let (clean, total) = simulate_nan_abort_loop(100, &[true; 100]);
    assert_eq!(clean, 0, "no clean steps should execute");
    assert_eq!(total, 5, "should abort after exactly 5 NaN steps");
}

/// Regression test for #2968: fewer than 5 consecutive NaNs does not abort.
#[test]
fn test_nan_abort_four_nans_does_not_trigger() {
    // 4 NaN then all clean: should NOT abort, should complete all steps.
    let mut seq = vec![true; 4];
    seq.extend(vec![false; 96]);
    let (clean, total) = simulate_nan_abort_loop(100, &seq);
    assert_eq!(clean, 96, "96 clean steps after 4 NaN skips");
    assert_eq!(total, 100, "all 100 steps consumed");
}

/// Regression test for #2968: counter resets on clean gradient.
#[test]
fn test_nan_abort_counter_resets_on_clean_step() {
    // Pattern: 4 NaN, 1 clean, 4 NaN, 1 clean, ... (never hits 5 consecutive)
    let pattern: Vec<bool> = (0..100)
        .map(|i| (i % 5) < 4) // [NaN, NaN, NaN, NaN, clean, NaN, NaN, NaN, NaN, clean, ...]
        .collect();
    let (clean, total) = simulate_nan_abort_loop(100, &pattern);
    assert_eq!(clean, 20, "20 clean steps (every 5th)");
    assert_eq!(total, 100, "all 100 steps consumed — no abort");
}

/// Regression test for #2968: counter resets then hits 5 consecutive.
#[test]
fn test_nan_abort_reset_then_trigger() {
    // Pattern: 3 NaN, 1 clean, then 5+ NaN → abort
    let mut seq = vec![true, true, true, false]; // 3 NaN then 1 clean
    seq.extend(vec![true; 20]); // 20 NaN: should abort after 5
    seq.extend(vec![false; 76]); // unreachable clean steps
    let (clean, total) = simulate_nan_abort_loop(100, &seq);
    assert_eq!(clean, 1, "only the one clean step before the NaN run");
    assert_eq!(total, 9, "3 NaN + 1 clean + 5 NaN abort = 9 iterations");
}

/// Regression test for #2968: integration test — attack with normal network
/// terminates with bounded evaluations (no infinite loop).
#[ntest::timeout(10000)]
#[test]
fn test_pgd_attack_terminates_with_bounded_evaluations() {
    // Simple linear network: y = 2*x
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let config = PgdConfig {
        num_restarts: 3,
        num_steps: 20,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    };
    let attacker = PgdAttacker::new(config);
    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();

    // Upper bound for evaluations: each restart does at most num_steps SPSA
    // evaluations (2 per step) + 1 final evaluation = 2*20 + 1 = 41 per restart.
    // With 3 restarts: max 3 * 41 = 123 evaluations.
    let max_expected_evals = 3 * (2 * 20 + 1);
    assert!(
        result.total_evaluations <= max_expected_evals,
        "evaluations {} should be bounded by {} (no infinite loop)",
        result.total_evaluations,
        max_expected_evals,
    );
    assert_eq!(result.restarts_completed, 3, "all restarts should complete");
}

/// Regression test for #2968: run_single_restart with a normal network
/// uses bounded evaluations per restart.
#[ntest::timeout(10000)]
#[test]
fn test_single_restart_bounded_evaluations() {
    // Network: y = x (identity via linear layer)
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), None).unwrap(),
    ));

    let config = PgdConfig {
        num_restarts: 1,
        num_steps: 50,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    };
    let attacker = PgdAttacker::new(config);
    let input_bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .run_single_restart(&network, &input_bounds, 0, 100.0, true, 42)
        .unwrap();

    // Each step: 2 SPSA evals. Plus 1 final eval. Total max: 2*50 + 1 = 101.
    let max_evals = 2 * 50 + 1;
    assert!(
        result.evaluations <= max_evals,
        "single restart evaluations {} should be bounded by {}",
        result.evaluations,
        max_evals,
    );
    assert!(
        result.evaluations > 0,
        "should have at least one evaluation"
    );
}
