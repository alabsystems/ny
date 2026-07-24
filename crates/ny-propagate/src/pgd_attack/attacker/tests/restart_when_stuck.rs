// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for PGD restart-when-stuck (#4278).
//!
//! Reference: alpha-beta-CROWN `general_spec_attack.py:409-427`.

use ndarray::{arr1, arr2};
use ny_core::NaiveCpuGemmEngine;
use ny_tensor::BoundedTensor;

use crate::layers::*;
use crate::pgd_attack::attacker::restart;
use crate::pgd_attack::optimizer::PgdOptimizer;
use crate::pgd_attack::{PgdAttacker, PgdConfig};
use crate::Network;

/// Zero-gradient network: y = 0 regardless of x.
///
/// SPSA therefore returns an all-zero gradient, so every projected update is a
/// no-op and `restart_when_stuck` is the only source of motion.
fn constant_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[0.0_f32, 0.0_f32]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    network
}

fn unit_box_input_bounds() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[0.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .unwrap()
}

fn restart_config(num_restarts: usize, restart_when_stuck: bool) -> PgdConfig {
    PgdConfig {
        num_restarts,
        num_steps: 2,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck,
        // Use SignedGradient: these tests rely on zero-gradient + constant step
        // producing a no-op (stuck detection). AdamClipping's sign(m/denom) with
        // eps produces nonzero motion even from zero gradients.
        optimizer: PgdOptimizer::SignedGradient,
        ..Default::default()
    }
}

/// `projected_step_is_stuck` returns true when arrays are identical.
#[ntest::timeout(5000)]
#[test]
fn test_projected_step_is_stuck_identical_4278() {
    let a = arr1(&[0.0_f32, 1.0, -1.0]).into_dyn();
    let b = arr1(&[0.0_f32, 1.0, -1.0]).into_dyn();
    assert!(restart::projected_step_is_stuck(&a, &b));
}

/// `projected_step_is_stuck` returns false when arrays differ.
#[ntest::timeout(5000)]
#[test]
fn test_projected_step_is_stuck_different_4278() {
    let a = arr1(&[0.0_f32, 1.0]).into_dyn();
    let b = arr1(&[0.0_f32, 1.0 + 1e-6]).into_dyn();
    assert!(!restart::projected_step_is_stuck(&a, &b));
}

/// Sequential PGD should resample when every projected step is a no-op.
#[ntest::timeout(10000)]
#[test]
fn test_sequential_restart_when_stuck_resamples_after_projected_noop_4278() {
    let network = constant_network();
    let input_bounds = unit_box_input_bounds();
    let without_restart = PgdAttacker::new(restart_config(1, false));
    let with_restart = PgdAttacker::new(restart_config(1, true));
    let mut initial_rng = without_restart.seeded_rng(42);
    let expected_initial = without_restart.sample_uniform(&input_bounds, &mut initial_rng);

    let without_restart_result = without_restart
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();
    let with_restart_result = with_restart
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();

    assert_eq!(
        without_restart_result.counterexample.unwrap(),
        expected_initial,
        "without restart_when_stuck, a zero-gradient projected no-op should leave the initial sample unchanged"
    );
    assert_ne!(
        with_restart_result.counterexample.unwrap(),
        expected_initial,
        "restart_when_stuck must resample after a projected no-op; otherwise this path degenerates into the initial sample"
    );
}

/// Batched PGD should resample the first restart when every projected step is a no-op.
#[ntest::timeout(10000)]
#[test]
fn test_batched_restart_when_stuck_resamples_after_projected_noop_4278() {
    let network = constant_network();
    let input_bounds = unit_box_input_bounds();
    let engine = NaiveCpuGemmEngine;
    let without_restart = PgdAttacker::new(restart_config(3, false)).with_engine(&engine);
    let with_restart = PgdAttacker::new(restart_config(3, true)).with_engine(&engine);
    let mut initial_rng = without_restart.seeded_rng(42);
    let expected_initial = without_restart.sample_uniform(&input_bounds, &mut initial_rng);

    let without_restart_result = without_restart
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();
    let with_restart_result = with_restart
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();

    assert_eq!(
        without_restart_result.counterexample.unwrap(),
        expected_initial,
        "the batched path should keep restart 0 at its initial sample when restart_when_stuck is disabled"
    );
    assert_ne!(
        with_restart_result.counterexample.unwrap(),
        expected_initial,
        "the batched path should resample restart 0 after projected no-op updates"
    );
}
