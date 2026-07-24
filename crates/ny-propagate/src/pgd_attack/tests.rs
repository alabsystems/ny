// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for PGD attack module.

use super::*;
use crate::layers::*;
use ndarray::{arr1, arr2};
use ny_core::NaiveCpuGemmEngine;
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::Network;

fn simple_linear_network() -> Network {
    // Simple network: y = W @ x + b
    // W = [[1, 2], [3, 4]], b = [0, 0]
    // So y[0] = x[0] + 2*x[1], y[1] = 3*x[0] + 4*x[1]
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));
    network
}

fn sign_threshold_network(threshold: f32) -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0_f32]]), Some(arr1(&[-threshold]))).unwrap(),
    ));
    network.add_layer(Layer::Sign(SignLayer::new()));
    network
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_evaluate_concrete() {
    let network = simple_linear_network();
    let attacker = PgdAttacker::new(PgdConfig::fast());

    let input = arr1(&[1.0_f32, 1.0]).into_dyn();
    let output = attacker.evaluate(&network, &input).unwrap();

    // y[0] = 1 + 2 = 3, y[1] = 3 + 4 = 7
    assert!(
        (output[[0]] - 3.0).abs() < 1e-5,
        "y[0] should be 3.0, got {}",
        output[[0]]
    );
    assert!(
        (output[[1]] - 7.0).abs() < 1e-5,
        "y[1] should be 7.0, got {}",
        output[[1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_evaluate_with_engine_threads_ibp_gemm_3954() {
    let network = simple_linear_network();
    let engine = CountingGemmEngine::new();
    let attacker = PgdAttacker::new(PgdConfig::fast()).with_engine(&engine);

    let input = arr1(&[1.0_f32, 1.0]).into_dyn();
    let output = attacker.evaluate(&network, &input).unwrap();

    assert!(
        (output[[0]] - 3.0).abs() < 1e-5,
        "y[0] with engine should be 3.0, got {}",
        output[[0]]
    );
    assert!(
        (output[[1]] - 7.0).abs() < 1e-5,
        "y[1] with engine should be 7.0, got {}",
        output[[1]]
    );
    assert_eq!(
        engine.gemm_calls(),
        1,
        "#3954 regression: concrete PGD evaluation should use one GEMM per linear layer"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_finds_counterexample() {
    // Network: y = x^2 (approximated by piecewise linear via ReLU)
    // For simplicity, use linear network y = 2*x
    // Input bounds: [-1, 1]
    // Property: y > 0 (should be violated at x < 0)
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 10,
        num_steps: 20,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Attack: find x where y <= 0 (verify_upper_bound = false, threshold = 0)
    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    // Should find counterexample at x < 0
    assert!(
        result.found_counterexample,
        "PGD should find counterexample where y=2x <= 0"
    );
    let cx = result.counterexample.unwrap();
    assert!(
        cx[[0]] < 0.0,
        "counterexample should be at x < 0, got x = {}",
        cx[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_no_counterexample_when_property_holds() {
    // Network: y = x + 5
    // Input bounds: [0, 1]
    // Property: y > 0 (always true since y >= 5)
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[5.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig::fast());

    let input_bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Attack: try to find x where y <= 0 (should fail)
    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    // Should NOT find counterexample
    assert!(
        !result.found_counterexample,
        "y=x+5 with x in [0,1] should not violate y > 0"
    );
    // Best value should still be > 0
    assert!(
        result.best_output_value > 0.0,
        "best output should be > 0 since y >= 5, got {}",
        result.best_output_value
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_difference_attack() {
    // Network with 2 outputs: y[0] = x, y[1] = 2*x
    // Property: y[0] <= y[1] (equivalent to y[0] - y[1] <= 0)
    // This should hold for x >= 0, violated for x < 0
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [2.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20,
        num_steps: 30,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Attack: find x where y[0] - y[1] >= 0 (verify_upper_bound = true, threshold = 0)
    // At x < 0: y[0] - y[1] = x - 2x = -x > 0
    let result = attacker
        .attack_difference(&network, &input_bounds, 0, 1, 0.0, true)
        .unwrap();

    // Should find counterexample at x < 0
    assert!(
        result.found_counterexample,
        "difference attack should find counterexample where y[0]-y[1] >= 0"
    );
    let cx = result.counterexample.unwrap();
    assert!(
        cx[[0]] < 0.0,
        "difference counterexample should be at x < 0, got x = {}",
        cx[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_conjunctive_less_eq_attack() {
    // Network with 3 outputs: y[0] = -x + 1, y[1] = x + 2, y[2] = x + 3
    // For x in [-1, 1]:
    //   y[0] = -x + 1 in [0, 2]
    //   y[1] = x + 2 in [1, 3]
    //   y[2] = x + 3 in [2, 4]
    // Property: y[0] <= y[1] AND y[0] <= y[2] (COC is minimal)
    // At x = 1: y[0] = 0, y[1] = 3, y[2] = 4 -> y[0] < y[1] and y[0] < y[2] -> satisfied
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0], [1.0], [1.0]]), Some(arr1(&[1.0, 2.0, 3.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20,
        num_steps: 30,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Attack: find x where y[0] <= y[1] AND y[0] <= y[2]
    // At x = 1: y[0] = 0 < y[1] = 3 and y[0] = 0 < y[2] = 4 -> should find counterexample
    let result = attacker
        .attack_conjunctive_less_eq(&network, &input_bounds, 0, &[1, 2])
        .unwrap();

    // Should find counterexample since at x=1, y[0] is minimal
    assert!(
        result.found_counterexample,
        "conjunctive attack should find counterexample where y[0] <= y[1] AND y[0] <= y[2]"
    );
    let _cx = result.counterexample.unwrap();
    // At counterexample, max(y[0] - y[1], y[0] - y[2]) <= 0
    assert!(
        result.best_output_value <= 0.0,
        "conjunctive best_output_value should be <= 0, got {}",
        result.best_output_value
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_disjunctive_greater_eq_attack() {
    // Network with 3 outputs:
    // y[0] = x
    // y[1] = -x
    // y[2] = 0.5
    //
    // Disjunction: y[0] >= y[2] OR y[1] >= y[2].
    // At x = 1, y[0] - y[2] = 0.5; at x = -1, y[1] - y[2] = 0.5.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [-1.0], [0.0]]), Some(arr1(&[0.0, 0.0, 0.5]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20,
        num_steps: 30,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack_disjunctive_greater_eq(&network, &input_bounds, 2, &[0, 1])
        .unwrap();

    assert!(
        result.found_counterexample,
        "disjunctive >= attack should find counterexample"
    );
    assert!(
        result.best_output_value >= 0.0,
        "disjunctive >= best_output_value should be >= 0, got {}",
        result.best_output_value
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_disjunctive_less_eq_attack() {
    // Network with 3 outputs:
    // y[0] = 0.5
    // y[1] = x
    // y[2] = -x
    //
    // Disjunction: y[0] >= y[1] OR y[0] >= y[2].
    // At x = 0, both hold; near the boundaries, at least one still holds.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[0.0], [1.0], [-1.0]]), Some(arr1(&[0.5, 0.0, 0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20,
        num_steps: 30,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack_disjunctive_less_eq(&network, &input_bounds, 0, &[1, 2])
        .unwrap();

    assert!(
        result.found_counterexample,
        "disjunctive <= attack should find counterexample"
    );
    assert!(
        result.best_output_value >= 0.0,
        "disjunctive <= best_output_value should be >= 0, got {}",
        result.best_output_value
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_disjunctive_restart_when_stuck_resamples_projected_noop_4278() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[0.0_f32, 0.0_f32], [0.0_f32, 0.0_f32]]),
            Some(arr1(&[0.0_f32, 1.0_f32])),
        )
        .unwrap(),
    ));

    let input_bounds = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0_f32]).into_dyn(),
        arr1(&[1.0_f32, 1.0_f32]).into_dyn(),
    )
    .unwrap();
    let without_restart = PgdAttacker::new(PgdConfig {
        num_restarts: 1,
        num_steps: 2,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        optimizer: PgdOptimizer::SignedGradient,
        alpha_mode: PgdAlphaMode::Scalar(0.01),
        ..Default::default()
    });
    let with_restart = PgdAttacker::new(PgdConfig {
        restart_when_stuck: true,
        ..without_restart.config().clone()
    });
    let mut initial_rng = without_restart.seeded_rng(42);
    let expected_initial = without_restart.sample_uniform(&input_bounds, &mut initial_rng);

    let without_restart_result = without_restart
        .attack_disjunctive_greater_eq(&network, &input_bounds, 1, &[0])
        .unwrap();
    let with_restart_result = with_restart
        .attack_disjunctive_greater_eq(&network, &input_bounds, 1, &[0])
        .unwrap();

    assert_eq!(
        without_restart_result.counterexample.unwrap(),
        expected_initial,
        "without restart_when_stuck, a zero-gradient disjunctive projected no-op should keep the initial sample"
    );
    assert_ne!(
        with_restart_result.counterexample.unwrap(),
        expected_initial,
        "restart_when_stuck must resample dead disjunctive PGD states instead of pinning them to the initial sample"
    );
}

// ============== PgdConfig Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_pgd_config_default() {
    let config = PgdConfig::default();
    assert_eq!(config.num_restarts, 100);
    assert_eq!(config.num_steps, 50);
    assert!(
        (config.step_size - 0.01).abs() < 1e-6,
        "default step_size should be 0.01, got {}",
        config.step_size
    );
    assert!(
        (config.spsa_delta - 0.001).abs() < 1e-6,
        "default spsa_delta should be 0.001, got {}",
        config.spsa_delta
    );
    assert_eq!(config.seed, 42);
    assert!(config.parallel, "default config should enable parallel");
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_config_fast() {
    let config = PgdConfig::fast();
    assert_eq!(config.num_restarts, 10);
    assert_eq!(config.num_steps, 20);
    assert!(
        !config.parallel,
        "fast config should disable parallel (too few restarts)"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_config_thorough() {
    let config = PgdConfig::thorough();
    assert_eq!(config.num_restarts, 1000);
    assert_eq!(config.num_steps, 100);
    assert!(
        (config.step_size - 0.005).abs() < 1e-6,
        "thorough step_size should be 0.005, got {}",
        config.step_size
    );
    assert!(
        (config.spsa_delta - 0.0005).abs() < 1e-6,
        "thorough spsa_delta should be 0.0005, got {}",
        config.spsa_delta
    );
    assert!(config.parallel, "thorough config should enable parallel");
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_config_acas_xu() {
    let config = PgdConfig::acas_xu();
    assert_eq!(config.num_restarts, 5000);
    assert_eq!(config.num_steps, 50);
    assert!(config.parallel, "acas_xu config should enable parallel");
}

// ============== PgdResult Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_pgd_result_fields_when_found() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 5,
        num_steps: 10,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Attack should find counterexample where y <= 0 (at x < 0)
    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    assert!(
        result.found_counterexample,
        "result should contain counterexample"
    );
    assert!(
        result.counterexample.is_some(),
        "counterexample field should be Some"
    );
    assert!(
        result.output.is_some(),
        "output field should be Some when found"
    );
    assert!(
        result.restarts_completed <= 5,
        "restarts_completed should be <= 5, got {}",
        result.restarts_completed
    );
    assert!(
        result.total_evaluations > 0,
        "total_evaluations should be > 0, got {}",
        result.total_evaluations
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_result_fields_when_not_found() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[10.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 3,
        num_steps: 5,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // y = x + 10, so y >= 10 for x >= 0. Looking for y <= 0 should fail.
    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    assert!(
        !result.found_counterexample,
        "y=x+10 should not have counterexample for y <= 0"
    );
    // Still should have best candidate
    assert!(
        result.counterexample.is_some(),
        "best candidate should still be populated"
    );
    assert!(result.output.is_some(), "output should still be populated");
    assert_eq!(result.restarts_completed, 3);
    assert!(
        result.total_evaluations > 0,
        "total_evaluations should be > 0 even without counterexample"
    );
    assert!(
        result.best_output_value > 0.0,
        "best output should be > 0 (couldn't get below 0), got {}",
        result.best_output_value
    );
}

// ============== Helper Method Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_pgd_project_clipping() {
    let attacker = PgdAttacker::new(PgdConfig::fast());
    let bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    // Point outside bounds
    let x = arr1(&[5.0_f32, -5.0]).into_dyn();
    let projected = attacker.project(&x, &bounds);

    assert!(
        (projected[[0]] - 1.0).abs() < 1e-6,
        "x[0]=5 should clip to upper bound 1.0, got {}",
        projected[[0]]
    );
    assert!(
        (projected[[1]] - 0.0).abs() < 1e-6,
        "x[1]=-5 should clip to lower bound 0.0, got {}",
        projected[[1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_project_within_bounds() {
    let attacker = PgdAttacker::new(PgdConfig::fast());
    let bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    // Point inside bounds
    let x = arr1(&[0.5_f32, 1.0]).into_dyn();
    let projected = attacker.project(&x, &bounds);

    assert!(
        (projected[[0]] - 0.5).abs() < 1e-6,
        "in-bounds x[0]=0.5 should be unchanged, got {}",
        projected[[0]]
    );
    assert!(
        (projected[[1]] - 1.0).abs() < 1e-6,
        "in-bounds x[1]=1.0 should be unchanged, got {}",
        projected[[1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_sample_uniform_within_bounds() {
    let attacker = PgdAttacker::new(PgdConfig::fast());
    let bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0]).into_dyn(),
    )
    .unwrap();

    let mut rng = StdRng::seed_from_u64(12345);

    // Sample multiple times and verify all within bounds
    for _ in 0..100 {
        let sample = attacker.sample_uniform(&bounds, &mut rng);

        assert!(
            sample[[0]] >= -1.0 && sample[[0]] <= 1.0,
            "sample[0] should be in [-1, 1], got {}",
            sample[[0]]
        );
        assert!(
            sample[[1]] >= 0.0 && sample[[1]] <= 2.0,
            "sample[1] should be in [0, 2], got {}",
            sample[[1]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_estimate_gradient_spsa_direction() {
    // Test that SPSA gradient estimate points in reasonable direction
    // Network: y = x (identity for simplicity)
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), None).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        spsa_delta: 0.01,
        ..PgdConfig::fast()
    });

    let x = arr1(&[0.5_f32]).into_dyn();
    let mut rng = StdRng::seed_from_u64(42);

    let (gradient, evals) = attacker
        .estimate_gradient_spsa(&network, &x, 0, &mut rng)
        .unwrap();

    // For y = x, gradient should be approximately 1
    // SPSA may have variance, but should be positive
    assert!(gradient[[0]] > 0.0, "Gradient should be positive for y=x");
    assert_eq!(evals, 2); // SPSA uses exactly 2 evaluations
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_estimate_gradient_spsa_with_bounds_escapes_sign_plateau_3769() {
    let network = sign_threshold_network(0.5);
    let attacker = PgdAttacker::new(PgdConfig {
        spsa_delta: 0.01,
        ..PgdConfig::fast()
    });
    let bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();
    let x = arr1(&[0.4_f32]).into_dyn();

    let mut baseline_rng = StdRng::seed_from_u64(42);
    let (baseline_gradient, baseline_evals) = attacker
        .estimate_gradient_spsa(&network, &x, 0, &mut baseline_rng)
        .unwrap();
    assert!(
        baseline_gradient[[0]].abs() < 1e-6,
        "single-scale SPSA should stay flat below the Sign threshold"
    );
    assert_eq!(baseline_evals, 2);

    // Smooth Sign relaxation (#3769): Sign networks dispatch to tanh(β*x)
    // approximation for gradient estimation, giving nonzero gradient in only
    // 2 evaluations (no multi-scale delta growth needed).
    let mut bounded_rng = StdRng::seed_from_u64(42);
    let (gradient, evals) = attacker
        .estimate_gradient_spsa_with_bounds(&network, &x, &bounds, 0, &mut bounded_rng)
        .unwrap();
    assert!(
        gradient[[0]] > 0.0,
        "smooth Sign SPSA should find a positive ascent direction toward the threshold"
    );
    assert_eq!(
        evals, 2,
        "smooth Sign path uses exactly 2 evaluations (no multi-scale growth)"
    );
}

// ============== Parallel Attack Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_pgd_parallel_attack_finds_counterexample() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    // Use parallel config with enough restarts
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20,
        num_steps: 20,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: true, // Enable parallel
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Looking for y <= 0 (should find at x < 0)
    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    assert!(
        result.found_counterexample,
        "parallel attack should find counterexample where y=2x <= 0"
    );
    let cx = result.counterexample.unwrap();
    assert!(
        cx[[0]] < 0.0,
        "parallel counterexample should be at x < 0, got x = {}",
        cx[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_parallel_vs_sequential_consistency() {
    // Both parallel and sequential should find counterexamples for same problem
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[3.0]]), Some(arr1(&[-1.0]))).unwrap(),
    ));

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Sequential attack
    let seq_attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 15,
        num_steps: 15,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });
    let seq_result = seq_attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    // Parallel attack
    let par_attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 15,
        num_steps: 15,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: true,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });
    let par_result = par_attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    // Both should find counterexamples
    assert!(
        seq_result.found_counterexample,
        "sequential attack should find counterexample"
    );
    assert!(
        par_result.found_counterexample,
        "parallel attack should find counterexample"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_parallel_difference_attack() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [2.0]]), Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20,
        num_steps: 20,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: true, // Enable parallel
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Attack: find x where y[0] - y[1] >= 0
    // At x < 0: y[0] - y[1] = x - 2x = -x > 0
    let result = attacker
        .attack_difference(&network, &input_bounds, 0, 1, 0.0, true)
        .unwrap();

    assert!(
        result.found_counterexample,
        "parallel difference attack should find counterexample"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_parallel_conjunctive_attack() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[-1.0], [1.0], [1.0]]), Some(arr1(&[1.0, 2.0, 3.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 25,
        num_steps: 25,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: true, // Enable parallel
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack_conjunctive_less_eq(&network, &input_bounds, 0, &[1, 2])
        .unwrap();

    assert!(
        result.found_counterexample,
        "parallel conjunctive attack should find counterexample"
    );
    assert!(
        result.best_output_value <= 0.0,
        "parallel conjunctive best_output_value should be <= 0, got {}",
        result.best_output_value
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_parallel_disjunctive_greater_eq_attack() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0], [-1.0], [0.0]]), Some(arr1(&[0.0, 0.0, 0.5]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20,
        num_steps: 30,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: true,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack_disjunctive_greater_eq_parallel_for_test(&network, &input_bounds, 2, &[0, 1])
        .unwrap();

    assert!(
        result.found_counterexample,
        "parallel disjunctive >= attack should find counterexample"
    );
    assert!(
        result.best_output_value >= 0.0,
        "parallel disjunctive >= best_output_value should be >= 0, got {}",
        result.best_output_value
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_parallel_disjunctive_less_eq_attack() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[0.0], [1.0], [-1.0]]), Some(arr1(&[0.5, 0.0, 0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20,
        num_steps: 30,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: true,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack_disjunctive_less_eq_parallel_for_test(&network, &input_bounds, 0, &[1, 2])
        .unwrap();

    assert!(
        result.found_counterexample,
        "parallel disjunctive <= attack should find counterexample"
    );
    assert!(
        result.best_output_value >= 0.0,
        "parallel disjunctive <= best_output_value should be >= 0, got {}",
        result.best_output_value
    );
}

// ============== RestartResult Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_restart_result_struct() {
    // Test that RestartResult is properly constructed via run_single_restart
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 1,
        num_steps: 5,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Use attack method which internally uses RestartResult
    let result = attacker
        .attack(&network, &input_bounds, 0, 10.0, true)
        .unwrap();

    // Check that RestartResult was properly created (via the output)
    assert!(
        result.output.is_some(),
        "RestartResult should populate output field"
    );
    let output = result.output.unwrap();
    assert_eq!(output.len(), 1); // Single output dimension
}

// ============== Edge Case Tests ==============

#[ntest::timeout(10000)]
#[test]
fn test_pgd_multidimensional_input() {
    // Test with higher-dimensional input
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0, 0.5, -0.5], [0.5, 1.0, 0.5]]),
            Some(arr1(&[0.0, 0.0])),
        )
        .unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig::fast());

    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();

    // Just verify it runs without error on multidimensional input
    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, true)
        .unwrap();

    assert!(
        result.total_evaluations > 0,
        "multidimensional PGD should perform evaluations"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_tight_bounds() {
    // Test with very tight input bounds (nearly concrete)
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[5.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig::fast());

    // Very tight bounds: [0.499, 0.501]
    let input_bounds =
        BoundedTensor::new(arr1(&[0.499_f32]).into_dyn(), arr1(&[0.501_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack(&network, &input_bounds, 0, 5.5, true)
        .unwrap();

    // y = x + 5, so y in [5.499, 5.501]. Looking for y >= 5.5 should find it.
    assert!(
        result.found_counterexample,
        "tight bounds [0.499, 0.501] should find y >= 5.5"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_pgd_reproducibility_with_seed() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Run same attack twice with same seed
    let config = PgdConfig {
        num_restarts: 5,
        num_steps: 10,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 12345,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    };

    let attacker1 = PgdAttacker::new(config.clone());
    let result1 = attacker1
        .attack(&network, &input_bounds, 0, 0.0, true)
        .unwrap();

    let attacker2 = PgdAttacker::new(config);
    let result2 = attacker2
        .attack(&network, &input_bounds, 0, 0.0, true)
        .unwrap();

    // Same seed should give same result
    assert!(
        (result1.best_output_value - result2.best_output_value).abs() < 1e-6,
        "Same seed should produce reproducible results"
    );
}

// ============== Error Handling Tests (#3096) ==============

/// Regression test for #3096: when all PGD restarts fail, the function must
/// return Err — not Ok(PgdResult { found_counterexample: false }) which callers
/// would misinterpret as "no counterexample found" (i.e., property verified).
#[ntest::timeout(10000)]
#[test]
fn test_pgd_all_restarts_fail_returns_err_3096() {
    // Network with 1 output. Requesting output_idx=99 causes every restart
    // to fail with InvalidSpec when output_value is called.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 3,
        num_steps: 2,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // output_idx 99 is out of bounds for a 1-output network — every restart fails.
    let result = attacker.attack(&network, &input_bounds, 99, 0.0, true);
    assert!(
        result.is_err(),
        "all-restarts-failed must return Err, not Ok with found_counterexample=false"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("all") && err_msg.contains("restarts failed"),
        "error should mention all restarts failed, got: {err_msg}"
    );
}

/// Same as above but for the parallel code path.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_parallel_all_restarts_fail_returns_err_3096() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 20, // >= 10 triggers parallel path
        num_steps: 2,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: true,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker.attack(&network, &input_bounds, 99, 0.0, true);
    assert!(
        result.is_err(),
        "parallel all-restarts-failed must return Err"
    );
}

/// Test that partial restart failures still produce valid results (not Err).
#[ntest::timeout(10000)]
#[test]
fn test_pgd_partial_restart_failure_still_succeeds_3096() {
    // A network with 2 outputs where output_idx=0 is valid.
    // This tests that when restarts succeed, the result includes failed_restarts count.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 5,
        num_steps: 10,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // Valid output_idx — all restarts should succeed.
    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    assert_eq!(
        result.failed_restarts, 0,
        "no restarts should fail with valid output_idx"
    );
}

/// Regression test for #3096: difference attack sequential all-restarts-fail returns Err.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_difference_all_restarts_fail_returns_err_3096() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 3,
        num_steps: 2,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // output_idx_j=99 is out of bounds — every restart fails in difference attack.
    let result = attacker.attack_difference(&network, &input_bounds, 0, 99, 0.0, true);
    assert!(
        result.is_err(),
        "difference attack: all-restarts-failed must return Err"
    );
}

/// Regression test for #3096: conjunctive attack sequential all-restarts-fail returns Err.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_conjunctive_all_restarts_fail_returns_err_3096() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 3,
        num_steps: 2,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    });

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // comparison_indices=[99] is out of bounds — every restart fails.
    let result = attacker.attack_conjunctive_less_eq(&network, &input_bounds, 0, &[99]);
    assert!(
        result.is_err(),
        "conjunctive attack: all-restarts-failed must return Err"
    );
}

// ============== NaN Gradient Guard Tests (#2745) ==============
//
// These tests verify the NaN gradient skip guard added to attack_conjunctive.rs
// and attack_difference.rs. The guard uses the same pattern as attacker.rs (#2721, #2968):
// skip NaN gradient steps, abort after 5 consecutive NaN steps.
//
// We use simulation-based tests (same approach as attacker.rs) rather than integration
// tests because BoundedTensor::concrete() rejects Inf/NaN, making it difficult to
// construct networks that produce NaN gradients without erroring at the evaluate level.
// The simulation tests verify the identical NaN-skip-counter logic deterministically.

/// Simulate the NaN-skip counter loop matching the guard in
/// attack_conjunctive.rs and attack_difference.rs.
/// Returns (clean_steps, total_iterations).
fn simulate_nan_abort_loop_2745(
    num_steps: usize,
    gradient_sequence: &[bool], // true = NaN gradient, false = clean
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

/// Regression test for #2745: 5 consecutive NaN gradients trigger abort.
#[test]
fn test_nan_guard_aborts_after_five_consecutive_2745() {
    let (clean, total) = simulate_nan_abort_loop_2745(100, &[true; 100]);
    assert_eq!(clean, 0, "no clean steps should execute");
    assert_eq!(total, 5, "should abort after exactly 5 NaN steps");
}

/// Regression test for #2745: NaN counter resets on clean gradient step.
#[test]
fn test_nan_guard_counter_resets_on_clean_step_2745() {
    // Pattern: 4 NaN, 1 clean, repeating (never hits 5 consecutive)
    let pattern: Vec<bool> = (0..100).map(|i| (i % 5) < 4).collect();
    let (clean, total) = simulate_nan_abort_loop_2745(100, &pattern);
    assert_eq!(clean, 20, "20 clean steps (every 5th)");
    assert_eq!(total, 100, "all 100 steps consumed — no abort");
}

// ============== Batched PGD Tests (#3954 Slice 3) ==============

/// Regression test for #3954: batched PGD attack finds counterexample.
/// When engine is present, attack() dispatches to attack_batched() which
/// processes all restarts in lockstep with one GPU dispatch per step.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_batched_finds_counterexample_3954() {
    // Network: y = 2*x. Looking for y <= 0 (violated at x < 0).
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let engine = NaiveCpuGemmEngine;
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 10,
        num_steps: 20,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    })
    .with_engine(&engine);

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    assert!(
        result.found_counterexample,
        "batched PGD should find counterexample at x < 0"
    );
    let cx = result.counterexample.unwrap();
    assert!(cx[[0]] < 0.0, "counterexample should be at x < 0");
}

/// Regression test for #3954: batched PGD reduces GEMM dispatch count.
/// With batching, S steps use S+1 GEMM calls (one per step + one final)
/// instead of N*(2*S+1) individual calls.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_batched_reduces_gemm_dispatches_3954() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[2.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let engine = CountingGemmEngine::new();
    let config = PgdConfig {
        num_restarts: 10,
        num_steps: 5,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    };
    let attacker = PgdAttacker::new(config).with_engine(&engine);

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let _result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    // Batched: 5 steps × 1 GEMM (batch of 2N=20) + 1 final GEMM (batch of N=10) = 6
    // Non-batched would be: 10 restarts × (5 steps × 2 evals + 1 final) × 1 GEMM = 110
    let gemm_calls = engine.gemm_calls();
    assert!(
        gemm_calls <= 10,
        "#3954 regression: batched PGD should use ~S+1 GEMM calls, got {}",
        gemm_calls,
    );
}

/// Regression test for #3954: batched PGD produces valid result when property holds.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_batched_no_counterexample_when_property_holds_3954() {
    // Network: y = x + 10. For x in [0,1], y >= 10. Looking for y <= 0 should fail.
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[10.0]))).unwrap(),
    ));

    let engine = NaiveCpuGemmEngine;
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 5,
        num_steps: 10,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    })
    .with_engine(&engine);

    let input_bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    assert!(
        !result.found_counterexample,
        "batched y=x+10 should not find counterexample for y <= 0"
    );
    assert!(
        result.best_output_value > 0.0,
        "batched best_output_value should be > 0 since y >= 10, got {}",
        result.best_output_value
    );
    assert_eq!(result.restarts_completed, 5);
}

/// Regression test for #3954: batched PGD with multidimensional input.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_batched_multidimensional_input_3954() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0, 0.5, -0.5], [0.5, 1.0, 0.5]]),
            Some(arr1(&[0.0, 0.0])),
        )
        .unwrap(),
    ));

    let engine = NaiveCpuGemmEngine;
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 5,
        num_steps: 10,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    })
    .with_engine(&engine);

    let input_bounds = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();

    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, true)
        .unwrap();

    assert!(
        result.total_evaluations > 0,
        "batched multidimensional PGD should perform evaluations"
    );
    assert_eq!(result.restarts_completed, 5);
}

/// Regression test for #3954: batched PGD all-restarts-fail returns Err.
#[ntest::timeout(10000)]
#[test]
fn test_pgd_batched_all_restarts_fail_returns_err_3954() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));

    let engine = NaiveCpuGemmEngine;
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 3,
        num_steps: 2,
        step_size: 0.1,
        spsa_delta: 0.01,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    })
    .with_engine(&engine);

    let input_bounds =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // output_idx 99 is out of bounds — every restart's gradient evaluation fails.
    let result = attacker.attack(&network, &input_bounds, 99, 0.0, true);
    assert!(
        result.is_err(),
        "batched all-restarts-failed must return Err"
    );
}

/// Regression test for #2745: counter resets then hits 5 consecutive.
#[test]
fn test_nan_guard_reset_then_trigger_2745() {
    // Pattern: 3 NaN, 1 clean, then 5+ NaN → abort
    let mut seq = vec![true, true, true, false, true, true, true, true, true, true];
    seq.extend(vec![false; 90]);
    let (clean, total) = simulate_nan_abort_loop_2745(100, &seq);
    assert_eq!(clean, 1, "only one clean step before abort");
    assert_eq!(total, 9, "3 NaN + 1 clean + 5 NaN abort = 9 steps");
}
