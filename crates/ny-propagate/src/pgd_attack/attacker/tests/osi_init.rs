// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for OSI (Output Specification Initialization) (#1449).
//!
//! Reference: alpha-beta-CROWN `attack_utils.py:328-362` (`OSI_init_C`).

use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;
use rand::RngExt;

use crate::layers::*;
use crate::pgd_attack::optimizer::PgdOptimizer;
use crate::pgd_attack::{PgdAttacker, PgdConfig, PgdInitialization};
use crate::Network;

/// Simple 2-input, 3-output network for OSI diversity testing.
///
/// y = W * x + b with W ∈ R^{3x2}, producing enough output dimensions
/// for the random scalarization direction `w` to have a meaningful effect.
fn multi_output_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0_f32, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            Some(arr1(&[0.0, 0.0, 0.0])),
        )
        .unwrap(),
    ));
    network
}

fn unit_box_2d() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap()
}

/// 1-input affine network where a one-step OSI update can be checked exactly.
///
/// For scalar input x and output y = W * x + b, the reference OSI gradient is
/// exactly `sum_i w_i * W_i`, and ny's SPSA sign update collapses to
/// the same direction because the Rademacher perturbation is scalar (delta ∈ {±1}).
const SCALAR_OSI_WEIGHTS_NY: [f32; 3] = [2.0, -3.0, 0.5];

fn scalar_input_multi_output_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[
                [SCALAR_OSI_WEIGHTS_NY[0]],
                [SCALAR_OSI_WEIGHTS_NY[1]],
                [SCALAR_OSI_WEIGHTS_NY[2]],
            ]),
            Some(arr1(&[0.1_f32, -0.2, 0.3])),
        )
        .unwrap(),
    ));
    network
}

fn scalar_box_1d() -> BoundedTensor {
    BoundedTensor::new(arr1(&[-5.0_f32]).into_dyn(), arr1(&[5.0_f32]).into_dyn()).unwrap()
}

/// OSI initialization produces a different starting point than Uniform
/// for the same seed, because the OSI steps push the point away from
/// the initial uniform sample.
#[ntest::timeout(10000)]
#[test]
fn test_osi_initialization_changes_seed_point_1449() {
    let network = multi_output_network();
    let input_bounds = unit_box_2d();

    let uniform_config = PgdConfig {
        num_restarts: 1,
        num_steps: 5,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        initialization: PgdInitialization::Uniform,
        osi_steps: 10,
        // Use SignedGradient: this test checks OSI initialization diversity.
        // With AdamClipping on a tiny [0,1]^2 box, both paths converge to
        // the same corner, masking the initialization difference.
        optimizer: PgdOptimizer::SignedGradient,
        ..Default::default()
    };
    let osi_config = PgdConfig {
        initialization: PgdInitialization::Osi,
        ..uniform_config
    };

    let uniform_attacker = PgdAttacker::new(uniform_config);
    let osi_attacker = PgdAttacker::new(osi_config);

    // Run both attacks with the same seed — the OSI path should produce
    // a different counterexample because the initial point is pushed
    // by the scalarization gradient ascent before PGD starts.
    let uniform_result = uniform_attacker
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();
    let osi_result = osi_attacker
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();

    assert_ne!(
        uniform_result.counterexample.as_ref().unwrap(),
        osi_result.counterexample.as_ref().unwrap(),
        "OSI should produce a different initial point than Uniform for the same seed"
    );
}

/// Uniform initialization matches pre-#1449 behavior (regression guard).
///
/// Two PgdAttacker instances with identical Uniform config and seed must
/// produce the exact same result.
#[ntest::timeout(10000)]
#[test]
fn test_uniform_initialization_stays_current_behavior_1449() {
    let network = multi_output_network();
    let input_bounds = unit_box_2d();

    let config = PgdConfig {
        num_restarts: 1,
        num_steps: 5,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        initialization: PgdInitialization::Uniform,
        ..Default::default()
    };

    let attacker_a = PgdAttacker::new(config.clone());
    let attacker_b = PgdAttacker::new(config);

    let result_a = attacker_a
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();
    let result_b = attacker_b
        .attack(&network, &input_bounds, 0, 100.0, true)
        .unwrap();

    assert_eq!(
        result_a.counterexample.unwrap(),
        result_b.counterexample.unwrap(),
        "Uniform initialization with the same seed must produce identical results"
    );
}

/// OSI with zero steps should degenerate to Uniform (only the probe eval runs,
/// no gradient steps, so the initial uniform sample is returned unmodified).
#[ntest::timeout(10000)]
#[test]
fn test_osi_zero_steps_degenerates_to_uniform_1449() {
    let network = multi_output_network();
    let input_bounds = unit_box_2d();

    // OSI with 0 steps: only the probe forward pass and direction draw run.
    // No gradient step executes, so the returned point must equal the initial
    // uniform sample exactly.
    let config = PgdConfig {
        num_restarts: 1,
        num_steps: 3,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        initialization: PgdInitialization::Osi,
        osi_steps: 0,
        ..Default::default()
    };

    let attacker = PgdAttacker::new(config);
    let mut expected_rng = attacker.seeded_rng(42);
    let expected_uniform = attacker.sample_uniform(&input_bounds, &mut expected_rng);

    let mut actual_rng = attacker.seeded_rng(42);
    let actual = attacker
        .initialize_restart(&network, &input_bounds, &mut actual_rng)
        .expect("OSI with 0 steps should return the initial uniform sample");

    assert_eq!(
        actual, expected_uniform,
        "OSI with 0 steps should leave the sampled restart unchanged"
    );
}

/// One-step OSI on a scalar-input affine network should match the reference
/// `OSI_init_C` update direction exactly.
#[ntest::timeout(10000)]
#[test]
fn test_osi_single_step_matches_reference_direction_for_scalar_input_1449() {
    let network = scalar_input_multi_output_network();
    let input_bounds = scalar_box_1d();
    let config = PgdConfig {
        num_restarts: 1,
        num_steps: 0,
        step_size: 0.25,
        spsa_delta: 0.01,
        seed: 7,
        parallel: false,
        initialization: PgdInitialization::Osi,
        osi_steps: 1,
        ..Default::default()
    };
    let attacker = PgdAttacker::new(config.clone());

    let mut expected_rng = attacker.seeded_rng(config.seed);
    let initial = attacker.sample_uniform(&input_bounds, &mut expected_rng);
    let w: Vec<f32> = (0..SCALAR_OSI_WEIGHTS_NY.len())
        .map(|_| expected_rng.random_range(-1.0_f32..=1.0_f32))
        .collect();
    let reference_gradient: f32 = SCALAR_OSI_WEIGHTS_NY
        .iter()
        .zip(w.iter())
        .map(|(weight, direction)| weight * direction)
        .sum();
    assert_ne!(
        reference_gradient, 0.0,
        "test seed must produce a non-zero scalarized gradient"
    );
    let reference_sign = if reference_gradient > 0.0 { 1.0 } else { -1.0 };
    // OSI runs a signed-gradient step whose magnitude is the *resolved* alpha,
    // not the raw `step_size` field: with the default `PgdAlphaMode::Auto` the
    // signed-gradient scale is `auto_alpha(input_bounds)` (here clamped to 1.0 on
    // the [-5, 5] box), so use the same resolved alpha the production path uses.
    let resolved_alpha = config.base_alpha(&input_bounds);
    let expected = attacker.project(
        &arr1(&[initial[[0]] + resolved_alpha * reference_sign]).into_dyn(),
        &input_bounds,
    );

    let mut actual_rng = attacker.seeded_rng(config.seed);
    let actual = attacker
        .initialize_restart(&network, &input_bounds, &mut actual_rng)
        .expect("one-step OSI initialization should succeed");

    assert_eq!(
        actual, expected,
        "one-step scalar OSI should match the reference exact-gradient direction"
    );
}
