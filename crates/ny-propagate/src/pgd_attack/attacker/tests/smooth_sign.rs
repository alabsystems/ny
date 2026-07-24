// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Smooth Sign relaxation and batched SPSA regression tests (#3769).

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::NaiveCpuGemmEngine;
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::layers::*;
use crate::pgd_attack::attacker::eval::SMOOTH_SIGN_BETA;
use crate::pgd_attack::attacker::PgdAttacker;
use crate::pgd_attack::config::PgdConfig;
use crate::Network;

use super::common::sign_threshold_network;

fn flatten_sign_linear_restart_axis_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Sign(SignLayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0, 0.0, 0.0]]), None).unwrap(),
    ));
    network
}

fn reshape_sign_linear_restart_axis_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![4])));
    network.add_layer(Layer::Sign(SignLayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 0.0, 0.0, 0.0]]), None).unwrap(),
    ));
    network
}

fn expected_smooth_sign_first_feature_gradient(x0: f32, delta: f32) -> f32 {
    (((SMOOTH_SIGN_BETA * (x0 + delta)).tanh()) - ((SMOOTH_SIGN_BETA * (x0 - delta)).tanh()))
        / (2.0 * delta)
}

fn assert_uniform_gradient_magnitude(gradient: &ArrayD<f32>, expected: f32, context: &str) {
    for (idx, value) in gradient.iter().enumerate() {
        assert!(
            (value.abs() - expected).abs() < 1e-4,
            "{context} gradient[{idx}] should have |value|≈{expected}, got {value}"
        );
    }
}

/// Smooth Sign relaxation: Sign networks use tanh(β*x) for SPSA probes,
/// giving nonzero gradient in 2 evals instead of multi-scale delta growth.
#[ntest::timeout(10000)]
#[test]
fn test_spsa_with_bounds_reaches_max_delta_with_default_config_3769() {
    let network = sign_threshold_network(0.25);
    let attacker = PgdAttacker::new(PgdConfig::fast());
    let bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let x = arr1(&[0.1_f32]).into_dyn();

    let mut baseline_rng = StdRng::seed_from_u64(42);
    let (baseline_gradient, baseline_evals) = attacker
        .estimate_gradient_spsa(&network, &x, 0, &mut baseline_rng)
        .unwrap();
    assert!(
        baseline_gradient[[0]].abs() < 1e-6,
        "single-scale SPSA should stay flat below the Sign threshold"
    );
    assert_eq!(baseline_evals, 2);

    // Smooth Sign relaxation (#3769): dispatches to tanh(β*x) path,
    // yielding nonzero gradient in exactly 2 evaluations.
    let mut bounded_rng = StdRng::seed_from_u64(42);
    let (gradient, evals) = attacker
        .estimate_gradient_spsa_with_bounds(&network, &x, &bounds, 0, &mut bounded_rng)
        .unwrap();
    assert!(
        gradient[[0]] > 0.0,
        "smooth Sign SPSA should find a positive ascent direction toward the threshold"
    );
    assert_eq!(evals, 2, "smooth Sign path uses exactly 2 evaluations");
}

/// Batched smooth Sign: Sign networks use tanh(β*x) for batch SPSA probes.
#[ntest::timeout(10000)]
#[test]
fn test_batched_spsa_with_bounds_escapes_sign_plateau_3769() {
    let network = sign_threshold_network(0.25);
    let engine = NaiveCpuGemmEngine;
    let attacker = PgdAttacker::new(PgdConfig::fast()).with_engine(&engine);
    let bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let x_batch = arr2(&[[0.1_f32]]).into_dyn();
    let mut rngs = vec![StdRng::seed_from_u64(42)];

    let (gradients, evals) = attacker
        .estimate_gradient_spsa_batch_with_bounds(&network, &x_batch, &bounds, 0, &mut rngs)
        .unwrap();

    assert!(
        gradients[[0, 0]] > 0.0,
        "batched smooth Sign SPSA should find positive gradient toward threshold"
    );
    assert_eq!(
        evals, 2,
        "batched smooth Sign path for 1 sample uses exactly 2 evaluations"
    );
}

/// Verify evaluate_smooth_sign produces tanh(β*x) instead of sign(x).
#[ntest::timeout(10000)]
#[test]
fn test_evaluate_smooth_sign_produces_continuous_output_3769() {
    let network = sign_threshold_network(0.5);
    let attacker = PgdAttacker::new(PgdConfig::fast());

    // Point near the Sign threshold: sign(0.4 - 0.5) = sign(-0.1) = -1
    let input = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.4]).unwrap();
    let exact = attacker.evaluate(&network, &input).unwrap();
    assert_eq!(exact[[0]], -1.0, "exact Sign(-0.1) = -1");

    // Smooth Sign: tanh(10 * (-0.1)) = tanh(-1) ≈ -0.76
    let smooth = attacker
        .evaluate_smooth_sign(&network, &input, SMOOTH_SIGN_BETA)
        .unwrap();
    assert!(
        smooth[[0]] > -1.0 && smooth[[0]] < 0.0,
        "smooth Sign should be between -1 and 0, got {}",
        smooth[[0]]
    );
    // tanh(-1) ≈ -0.7616
    assert!(
        (smooth[[0]] - (-0.7616)).abs() < 0.01,
        "smooth Sign(-0.1) with beta=10 should be ≈ -0.76, got {}",
        smooth[[0]]
    );
}

/// Verify smooth Sign gives nonzero SPSA gradient at Sign plateau (#3769).
///
/// With discrete Sign, SPSA at delta=0.001 gives zero gradient because both
/// probes land on the same side of the threshold. Smooth Sign gives nonzero
/// gradient because tanh(β*x) is continuous.
#[ntest::timeout(10000)]
#[test]
fn test_smooth_sign_spsa_gives_nonzero_gradient_at_plateau_3769() {
    let network = sign_threshold_network(0.5);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    // Test point at x=0.4, near threshold at 0.5
    let x = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.4]).unwrap();

    let attacker = PgdAttacker::new(PgdConfig {
        spsa_delta: 0.001, // Small delta — discrete Sign gives zero diff
        ..PgdConfig::fast()
    });

    let mut rng = StdRng::seed_from_u64(42);

    // Baseline: discrete Sign SPSA gives zero at small delta
    let (baseline_grad, _) = attacker
        .estimate_gradient_spsa(&network, &x, 0, &mut rng)
        .unwrap();
    assert_eq!(
        baseline_grad[[0]],
        0.0,
        "baseline SPSA with small delta should give zero gradient through discrete Sign"
    );

    // Smooth Sign path: should give nonzero gradient
    let mut rng2 = StdRng::seed_from_u64(42);
    let (smooth_grad, evals) = attacker
        .estimate_gradient_spsa_smooth_sign(&network, &x, &input_bounds, 0, &mut rng2)
        .unwrap();

    assert!(
        smooth_grad[[0]].abs() > 0.0,
        "smooth Sign SPSA should give nonzero gradient, got {}",
        smooth_grad[[0]]
    );
    assert!(
        smooth_grad[[0]] > 0.0,
        "gradient should point toward threshold (positive direction), got {}",
        smooth_grad[[0]]
    );
    assert_eq!(evals, 2, "smooth Sign uses exactly 2 evaluations");
}

/// Verify estimate_gradient_spsa_with_bounds dispatches to smooth Sign
/// path for networks with Sign layers (#3769).
#[ntest::timeout(10000)]
#[test]
fn test_spsa_with_bounds_dispatches_smooth_sign_for_sign_networks_3769() {
    let network = sign_threshold_network(0.25);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    let x = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.1]).unwrap();

    let attacker = PgdAttacker::new(PgdConfig {
        spsa_delta: 0.001,
        ..PgdConfig::fast()
    });

    let mut rng = StdRng::seed_from_u64(42);
    let (grad, evals) = attacker
        .estimate_gradient_spsa_with_bounds(&network, &x, &input_bounds, 0, &mut rng)
        .unwrap();

    // Should use smooth Sign (2 evals) not multi-scale (>= 4 evals)
    assert_eq!(
        evals, 2,
        "Sign network should use smooth path (2 evals), got {evals}"
    );
    assert!(
        grad[[0]] > 0.0,
        "gradient should point toward threshold, got {}",
        grad[[0]]
    );
}

/// Verify batched smooth Sign SPSA gives nonzero gradients (#3769).
#[ntest::timeout(10000)]
#[test]
fn test_batched_smooth_sign_spsa_gives_nonzero_gradient_3769() {
    let network = sign_threshold_network(0.5);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    // Batch of 2 inputs near the threshold
    let inputs = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.4, 0.6]).unwrap();
    let mut rngs: Vec<StdRng> = (0..2).map(|i| StdRng::seed_from_u64(42 + i)).collect();

    let attacker = PgdAttacker::new(PgdConfig {
        spsa_delta: 0.001,
        ..PgdConfig::fast()
    });

    let (gradients, evals) = attacker
        .estimate_gradient_spsa_batch_with_bounds(&network, &inputs, &input_bounds, 0, &mut rngs)
        .unwrap();

    // Both should have nonzero gradients via smooth Sign
    assert!(
        gradients[[0, 0]].abs() > 0.0,
        "batch[0] gradient should be nonzero, got {}",
        gradients[[0, 0]]
    );
    assert!(
        gradients[[1, 0]].abs() > 0.0,
        "batch[1] gradient should be nonzero, got {}",
        gradients[[1, 0]]
    );
    // x=0.4 is below threshold: gradient should point up (positive)
    assert!(
        gradients[[0, 0]] > 0.0,
        "batch[0] at x=0.4 should have positive gradient"
    );
    assert_eq!(evals, 4, "batch of 2 should use 4 evaluations");
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_smooth_sign_spsa_preserves_restart_axis_through_flatten_4345() {
    let network = flatten_sign_linear_restart_axis_network();
    let engine = NaiveCpuGemmEngine;
    let delta = 0.01;
    let attacker = PgdAttacker::new(PgdConfig {
        spsa_delta: delta,
        ..PgdConfig::fast()
    })
    .with_engine(&engine);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), vec![-1.0_f32; 4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 1]), vec![1.0_f32; 4]).unwrap(),
    )
    .unwrap();
    let inputs = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 4, 1]),
        vec![0.1_f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
    )
    .unwrap();
    let mut rngs = vec![StdRng::seed_from_u64(7), StdRng::seed_from_u64(19)];

    let (gradient, evals) = attacker
        .estimate_gradient_spsa_batch_with_bounds(&network, &inputs, &input_bounds, 0, &mut rngs)
        .expect("batched smooth Sign SPSA should preserve the restart axis through Flatten(0)");

    assert_eq!(gradient.shape(), &[2, 1, 4, 1]);
    assert_eq!(evals, 4);
    assert!(
        gradient.iter().all(|value| value.is_finite()),
        "Flatten(0) batched smooth Sign SPSA gradients should stay finite: {gradient:?}"
    );
    let expected = expected_smooth_sign_first_feature_gradient(0.1, delta);
    assert!(
        (gradient[[0, 0, 0, 0]] - expected).abs() < 1e-4,
        "first sample first-feature smooth Sign SPSA gradient should stay near {expected}, got {}",
        gradient[[0, 0, 0, 0]]
    );
    assert!(
        (gradient[[1, 0, 0, 0]] - expected).abs() < 1e-4,
        "second sample first-feature smooth Sign SPSA gradient should stay near {expected}, got {}",
        gradient[[1, 0, 0, 0]]
    );
    assert_uniform_gradient_magnitude(
        &gradient,
        expected,
        "Flatten(0) batched smooth Sign preserve-leading-axis",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_smooth_sign_spsa_preserves_restart_axis_through_reshape_3769() {
    let network = reshape_sign_linear_restart_axis_network();
    let engine = NaiveCpuGemmEngine;
    let delta = 0.01;
    let attacker = PgdAttacker::new(PgdConfig {
        spsa_delta: delta,
        ..PgdConfig::fast()
    })
    .with_engine(&engine);
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0_f32; 4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0_f32; 4]).unwrap(),
    )
    .unwrap();
    let inputs = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![0.1_f32, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
    )
    .unwrap();
    let mut rngs = vec![StdRng::seed_from_u64(5), StdRng::seed_from_u64(17)];

    let (gradient, evals) = attacker
        .estimate_gradient_spsa_batch_with_bounds(&network, &inputs, &input_bounds, 0, &mut rngs)
        .expect("batched smooth Sign SPSA should preserve the restart axis through Reshape");

    assert_eq!(gradient.shape(), &[2, 2, 2]);
    assert_eq!(evals, 4);
    assert!(
        gradient.iter().all(|value| value.is_finite()),
        "Reshape batched smooth Sign SPSA gradients should stay finite: {gradient:?}"
    );
    let expected = expected_smooth_sign_first_feature_gradient(0.1, delta);
    assert!(
        (gradient[[0, 0, 0]] - expected).abs() < 1e-4,
        "first sample first-feature reshape smooth Sign SPSA gradient should stay near {expected}, got {}",
        gradient[[0, 0, 0]]
    );
    assert!(
        (gradient[[1, 0, 0]] - expected).abs() < 1e-4,
        "second sample first-feature reshape smooth Sign SPSA gradient should stay near {expected}, got {}",
        gradient[[1, 0, 0]]
    );
    assert_uniform_gradient_magnitude(
        &gradient,
        expected,
        "Reshape batched smooth Sign preserve-leading-axis",
    );
}
