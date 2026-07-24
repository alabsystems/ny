// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;

const SAMPLE_TOLERANCE_NY: f32 = 1.0e-5;

fn simple_network_output_2769(x0: f32, x1: f32) -> f32 {
    (x0 - x1).abs()
}

fn constraint_holds_for_simple_network_2769(
    constraint: &NeuronConstraint,
    x0: f32,
    x1: f32,
) -> bool {
    let pre_activation = match constraint.neuron_idx {
        0 => x0 - x1,
        1 => x1 - x0,
        idx => panic!("unexpected simple_network ReLU neuron index {idx}"),
    };
    if constraint.is_active {
        pre_activation >= -SAMPLE_TOLERANCE_NY
    } else {
        pre_activation <= SAMPLE_TOLERANCE_NY
    }
}

fn assert_simple_network_bounds_contain_samples_2769(
    bounds: &BoundedTensor,
    history: Option<&SplitHistory>,
) {
    let flat = bounds.flatten();
    let lower = flat.lower()[[0]];
    let upper = flat.upper()[[0]];

    for i in -5..=5 {
        let x0 = i as f32 / 10.0;
        for j in -5..=5 {
            let x1 = j as f32 / 10.0;
            if let Some(history) = history {
                if !history
                    .constraints
                    .iter()
                    .all(|c| constraint_holds_for_simple_network_2769(c, x0, x1))
                {
                    continue;
                }
            }

            let y = simple_network_output_2769(x0, x1);
            assert!(
                y >= lower - SAMPLE_TOLERANCE_NY && y <= upper + SAMPLE_TOLERANCE_NY,
                "sample ({x0}, {x1}) -> {y} must stay within [{lower}, {upper}]",
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_optimization_with_constraints() {
    // Test that beta optimization runs and produces valid bounds
    let network = simple_network();

    // Input that creates unstable neurons
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Create config with beta optimization enabled
    let config = BetaCrownConfig {
        max_domains: 20,
        timeout: Duration::from_secs(10),
        beta_lr: 0.05,
        beta_iterations: 5,
        beta_tolerance: 1e-6,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -5.0).unwrap();

    // Should verify (output >= 0 for this network)
    println!("Beta optimization result: {:?}", result);
    assert_eq!(result.result, BabVerificationStatus::Verified);
}

#[ntest::timeout(10000)]
#[test]
fn test_compute_bounds_with_constraints_empty_layer_bounds_errors_2095() {
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let beta_state = BetaState::from_history(&history).unwrap();
    let layer_bounds: Vec<Arc<BoundedTensor>> = Vec::new();
    let verifier = BetaCrownVerifier::default();

    let err = verifier
        .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
        .expect_err("empty layer_bounds must error");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(ref msg) if msg.contains("layer_bounds length 0 does not match network layers")),
        "expected InvalidSpec for empty layer_bounds, got {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_compute_bounds_with_constraints_contains_samples_2769() {
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let beta_state = BetaState::from_history(&history).unwrap();
    let verifier = BetaCrownVerifier::default();

    let bounds = verifier
        .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
        .expect("constrained sequential bound computation should succeed");

    assert_simple_network_bounds_contain_samples_2769(&bounds, Some(&history));
}

#[ntest::timeout(10000)]
#[test]
fn test_beta_optimization_disabled() {
    // Test that setting beta_iterations=0 disables optimization
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = BetaCrownConfig {
        max_domains: 20,
        timeout: Duration::from_secs(10),
        beta_iterations: 0, // Disable optimization
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -5.0).unwrap();

    // Should still work without optimization
    println!("No beta optimization result: {:?}", result);
    assert_eq!(result.result, BabVerificationStatus::Verified);
}

#[ntest::timeout(10000)]
#[test]
fn test_joint_alpha_beta_optimization() {
    // Test that joint α-β optimization produces valid bounds
    let network = simple_network();

    // Input that creates unstable neurons (crossing zero)
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Config with joint α-β optimization enabled
    let config = BetaCrownConfig {
        max_domains: 20,
        timeout: Duration::from_secs(10),
        use_alpha_crown: true,
        beta_lr: 0.05,
        alpha_lr: 0.5,
        alpha_momentum: true,
        beta_iterations: 10,
        beta_tolerance: 1e-6,
        ..Default::default()
    };

    let verifier = BetaCrownVerifier::new(config);
    let result = verifier.verify(&network, &input, -5.0).unwrap();

    println!("Joint α-β optimization result: {:?}", result);
    assert_eq!(result.result, BabVerificationStatus::Verified);
    assert!(
        result.domains_explored > 0,
        "Should explore at least one domain"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_domain_alpha_state_initialization() {
    // Test that domain alpha state is correctly initialized
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[0.5, -0.3], [-0.2, 0.6]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    // Input with crossing bounds to create unstable neurons
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let history = SplitHistory::new();

    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    // Should have some unstable neurons tracked
    println!("DomainAlphaState: {} unstable neurons", alpha_state.len());
    for (&(layer_idx, neuron_idx), n) in &alpha_state.neurons {
        println!(
            "  Layer {}, Neuron {}: α={}",
            layer_idx, neuron_idx, n.alpha
        );
    }

    // Without constraints, should have unstable neurons from layer 1 (ReLU)
    // Check that alphas are in valid range [0, 1]
    for n in alpha_state.neurons.values() {
        assert!((0.0..=1.0).contains(&n.alpha), "Alpha should be in [0, 1]");
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_domain_alpha_state_with_constraints() {
    // Test that constrained neurons are excluded from alpha optimization
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[0.5, -0.3], [-0.2, 0.6]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    // Add constraint on neuron 0 of ReLU layer
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    // Constrained neuron should NOT be in alpha_state
    assert!(
        !alpha_state.is_unstable(1, 0),
        "Constrained neuron should not be tracked for alpha optimization"
    );

    println!(
        "Alpha state after constraint: {} unstable neurons",
        alpha_state.len()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_joint_optimization_improves_bounds() {
    // Verify that joint α-β optimization is at least as efficient as β-only.
    //
    // Network: |x1-x2| via Linear→ReLU→Linear (simple_network), input [-1,1]^2.
    // True minimum output is 0. We use threshold -0.01 (verifiable, tight).
    //
    // Note: For this simple network, CROWN's lower bound is already exact (0),
    // so both methods verify in 1 domain. The domain-count assertion guards
    // against regressions where joint optimization performs worse than β-only.
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let threshold = -0.01;

    // Beta-only optimization
    let config_beta_only = BetaCrownConfig {
        max_domains: 50,
        timeout: Duration::from_secs(10),
        use_alpha_crown: false,
        beta_iterations: 10,
        ..Default::default()
    };
    let verifier_beta = BetaCrownVerifier::new(config_beta_only);
    let result_beta = verifier_beta.verify(&network, &input, threshold).unwrap();

    // Joint α-β optimization
    let config_joint = BetaCrownConfig {
        max_domains: 50,
        timeout: Duration::from_secs(10),
        use_alpha_crown: true,
        beta_iterations: 10,
        alpha_lr: 0.5,
        alpha_momentum: true,
        ..Default::default()
    };
    let verifier_joint = BetaCrownVerifier::new(config_joint);
    let result_joint = verifier_joint.verify(&network, &input, threshold).unwrap();

    // Both should verify (true minimum 0 > -0.01)
    assert_eq!(
        result_beta.result,
        BabVerificationStatus::Verified,
        "Beta-only failed to verify at threshold {}",
        threshold
    );
    assert_eq!(
        result_joint.result,
        BabVerificationStatus::Verified,
        "Joint α-β failed to verify at threshold {}",
        threshold
    );

    // Joint optimization should explore no more domains than beta-only
    // (tighter bounds from α optimization → fewer splits needed)
    assert!(
        result_joint.domains_explored <= result_beta.domains_explored,
        "Joint α-β explored {} domains vs beta-only {} — joint should be at least as efficient",
        result_joint.domains_explored,
        result_beta.domains_explored,
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_adaptive_opt_config_defaults() {
    // Test that AdaptiveOptConfig has sensible defaults
    let config = AdaptiveOptConfig::default();

    // Standard Adam defaults
    assert_eq!(config.beta1, 0.9);
    assert_eq!(config.beta2, 0.999);
    assert_eq!(config.epsilon, 1e-8);
    assert!(config.bias_correction);
    assert!(!config.radam);

    // Our specific defaults for β-CROWN (matching α,β-CROWN)
    assert_eq!(config.beta_lr, 0.05); // α,β-CROWN default
    assert_eq!(config.alpha_lr, 0.01); // α,β-CROWN default
    assert_eq!(config.grad_clip, 10.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_adam_update() {
    // Test that Adam optimizer updates α correctly and respects [0, 1] bounds
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let history = SplitHistory::new();

    let mut alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    // simple_network() with [-0.5, 0.5] input bounds produces unstable ReLU neurons.
    // Assert rather than early-return to ensure the test exercises the real code path.
    assert!(
        !alpha_state.is_empty(),
        "Expected unstable neurons from simple_network() with [-0.5, 0.5] bounds"
    );

    let config = AdaptiveOptConfig::default();

    // Get first unstable neuron for testing
    let key = *alpha_state.neurons.keys().next().unwrap();
    let initial_alpha = alpha_state.alpha(key.0, key.1);

    // Set a positive gradient (should increase α towards 1)
    alpha_state.accumulate_grad(key.0, key.1, 0.5);

    for t in 1..=10 {
        let max_grad = alpha_state.gradient_step_adam(&config, t);

        let current_alpha = alpha_state.alpha(key.0, key.1);

        // Alpha should always be in [0, 1]
        assert!(
            (0.0..=1.0).contains(&current_alpha),
            "Alpha out of bounds: {}",
            current_alpha
        );

        println!(
            "t={}: alpha={:.4}, max_grad={:.4}",
            t, current_alpha, max_grad
        );

        // Reset and set same gradient
        alpha_state.zero_grad();
        alpha_state.accumulate_grad(key.0, key.1, 0.5);
    }

    let final_alpha = alpha_state.alpha(key.0, key.1);

    // With positive gradient, α should have moved towards 1 (or stayed there if already 1)
    assert!(
        final_alpha >= initial_alpha || (initial_alpha == 1.0 && final_alpha == 1.0),
        "Alpha should increase or stay at 1 with positive gradient"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_adaptive_optimization_convergence() {
    // Test that adaptive optimization converges similarly to fixed-LR
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Fixed learning rate
    let config_fixed = BetaCrownConfig {
        max_domains: 30,
        timeout: Duration::from_secs(10),
        use_alpha_crown: true,
        use_adaptive: false,
        beta_iterations: 15,
        ..Default::default()
    };
    let verifier_fixed = BetaCrownVerifier::new(config_fixed);
    let result_fixed = verifier_fixed.verify(&network, &input, -3.0).unwrap();

    // Adaptive learning rate
    let config_adaptive = BetaCrownConfig {
        max_domains: 30,
        timeout: Duration::from_secs(10),
        use_alpha_crown: true,
        use_adaptive: true,
        beta_iterations: 15,
        adaptive_config: AdaptiveOptConfig::default(),
        ..Default::default()
    };
    let verifier_adaptive = BetaCrownVerifier::new(config_adaptive);
    let result_adaptive = verifier_adaptive.verify(&network, &input, -3.0).unwrap();

    println!(
        "Fixed LR: {:?}, domains={}",
        result_fixed.result, result_fixed.domains_explored
    );
    println!(
        "Adaptive: {:?}, domains={}",
        result_adaptive.result, result_adaptive.domains_explored
    );

    // Both should verify this simple network
    assert_eq!(result_fixed.result, BabVerificationStatus::Verified);
    assert_eq!(result_adaptive.result, BabVerificationStatus::Verified);
}

/// Test that compute_bounds_from_layer produces equivalent results to full computation.
///
/// This tests the intermediate bound transfer optimization (#1513): when a child domain
/// only needs to recompute from a split layer, partial computation should give the same
/// results as full computation.
#[ntest::timeout(10000)]
#[test]
fn test_compute_bounds_from_layer_equivalence() {
    // Create a simple network: Linear -> ReLU -> Linear
    let network = simple_network();

    // Input with crossing bounds (creates unstable neurons)
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Get layer bounds via IBP
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let cut_pool = CutPool::default();

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Compute full bounds (the reference)
    let (full_bounds, full_intermediate) = verifier
        .compute_bounds_capturing_intermediate(
            &network,
            &input,
            &history,
            &layer_bounds,
            &beta_state,
            &alpha_state,
            &cut_pool,
            None,
        )
        .unwrap();

    let full_lb: f32 = full_bounds
        .lower()
        .iter()
        .cloned()
        .reduce(f32::min)
        .unwrap_or(0.0);
    let full_ub: f32 = full_bounds
        .upper()
        .iter()
        .cloned()
        .reduce(f32::max)
        .unwrap_or(0.0);
    println!("Full computation bounds: [{}, {}]", full_lb, full_ub);

    // Test partial computation from each valid layer (0 to num_layers-2)
    // We exclude the last layer since starting from there is trivial (just output bounds)
    for start_layer in 0..network.layers.len() - 1 {
        let (partial_bounds, _) = verifier
            .compute_bounds_from_layer(
                &network,
                &input,
                &history,
                &layer_bounds,
                &beta_state,
                &alpha_state,
                &cut_pool,
                start_layer,
                &full_intermediate,
                None,
            )
            .unwrap();

        let partial_lb: f32 = partial_bounds
            .lower()
            .iter()
            .cloned()
            .reduce(f32::min)
            .unwrap_or(0.0);
        let partial_ub: f32 = partial_bounds
            .upper()
            .iter()
            .cloned()
            .reduce(f32::max)
            .unwrap_or(0.0);
        println!(
            "Partial (from layer {}) bounds: [{}, {}]",
            start_layer, partial_lb, partial_ub
        );

        // Allow small floating-point tolerance
        assert!(
            (full_lb - partial_lb).abs() < 1e-5,
            "Lower bounds from layer {} should match full: full={}, partial={}",
            start_layer,
            full_lb,
            partial_lb
        );
        assert!(
            (full_ub - partial_ub).abs() < 1e-5,
            "Upper bounds from layer {} should match full: full={}, partial={}",
            start_layer,
            full_ub,
            partial_ub
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_compute_bounds_from_layer_contains_samples_2769() {
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let cut_pool = CutPool::default();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let (_, full_intermediate) = verifier
        .compute_bounds_capturing_intermediate(
            &network,
            &input,
            &history,
            &layer_bounds,
            &beta_state,
            &alpha_state,
            &cut_pool,
            None,
        )
        .expect("full sequential bound computation should succeed");

    for start_layer in 0..network.layers.len() - 1 {
        let (bounds, _) = verifier
            .compute_bounds_from_layer(
                &network,
                &input,
                &history,
                &layer_bounds,
                &beta_state,
                &alpha_state,
                &cut_pool,
                start_layer,
                &full_intermediate,
                None,
            )
            .unwrap_or_else(|err| {
                panic!("compute_bounds_from_layer({start_layer}) should succeed: {err}")
            });

        assert_simple_network_bounds_contain_samples_2769(&bounds, None);
    }
}

/// Test compute_bounds_from_layer on a deeper 5-layer network.
///
/// This verifies the intermediate bound transfer works correctly when there are
/// multiple layers and we start computation from intermediate layers.
#[ntest::timeout(10000)]
#[test]
fn test_compute_bounds_from_layer_deeper_network() {
    // Create a 5-layer network: L1 -> ReLU -> L2 -> ReLU -> L3
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();
    let w2 = arr2(&[[0.8, -0.2], [-0.3, 0.9]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();
    let w3 = arr2(&[[0.6, 0.4]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));

    // Input with crossing bounds
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Get layer bounds via IBP
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let cut_pool = CutPool::default();

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Compute full bounds (reference)
    let (full_bounds, full_intermediate) = verifier
        .compute_bounds_capturing_intermediate(
            &network,
            &input,
            &history,
            &layer_bounds,
            &beta_state,
            &alpha_state,
            &cut_pool,
            None,
        )
        .unwrap();

    let full_lb: f32 = full_bounds
        .lower()
        .iter()
        .cloned()
        .reduce(f32::min)
        .unwrap_or(0.0);
    let full_ub: f32 = full_bounds
        .upper()
        .iter()
        .cloned()
        .reduce(f32::max)
        .unwrap_or(0.0);
    println!("Full computation bounds: [{}, {}]", full_lb, full_ub);

    // Test partial computation from each valid start layer (0 to num_layers-2)
    // We exclude the last layer since starting from there is trivial
    for start_layer in 0..network.layers.len() - 1 {
        let (partial_bounds, _) = verifier
            .compute_bounds_from_layer(
                &network,
                &input,
                &history,
                &layer_bounds,
                &beta_state,
                &alpha_state,
                &cut_pool,
                start_layer,
                &full_intermediate,
                None,
            )
            .unwrap();

        let partial_lb: f32 = partial_bounds
            .lower()
            .iter()
            .cloned()
            .reduce(f32::min)
            .unwrap_or(0.0);
        let partial_ub: f32 = partial_bounds
            .upper()
            .iter()
            .cloned()
            .reduce(f32::max)
            .unwrap_or(0.0);
        println!(
            "Partial (from layer {}) bounds: [{}, {}]",
            start_layer, partial_lb, partial_ub
        );

        // Both lower and upper bounds should match within tolerance
        assert!(
            (full_lb - partial_lb).abs() < 1e-5,
            "Lower bounds from layer {} should match full: full={}, partial={}",
            start_layer,
            full_lb,
            partial_lb
        );
        assert!(
            (full_ub - partial_ub).abs() < 1e-5,
            "Upper bounds from layer {} should match full: full={}, partial={}",
            start_layer,
            full_ub,
            partial_ub
        );
    }
}

fn bounds_min_max(bounds: &BoundedTensor) -> (f32, f32) {
    let lb = bounds
        .lower()
        .iter()
        .cloned()
        .reduce(f32::min)
        .unwrap_or(0.0);
    let ub = bounds
        .upper()
        .iter()
        .cloned()
        .reduce(f32::max)
        .unwrap_or(0.0);
    (lb, ub)
}

/// Partial parent intermediate bounds must trigger safe full recomputation.
#[ntest::timeout(10000)]
#[test]
fn test_compute_bounds_from_layer_partial_parent_falls_back_to_full_recompute() {
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let cut_pool = CutPool::default();
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());

    let (full_bounds, full_intermediate) = verifier
        .compute_bounds_capturing_intermediate(
            &network,
            &input,
            &history,
            &layer_bounds,
            &beta_state,
            &alpha_state,
            &cut_pool,
            None,
        )
        .unwrap();

    let partial_parent = IntermediateLinearBounds {
        bounds_at_layer: full_intermediate.bounds_at_layer[..network.layers.len() - 1].to_vec(),
        start_layer: full_intermediate.start_layer,
    };

    let (partial_bounds, partial_intermediate) = verifier
        .compute_bounds_from_layer(
            &network,
            &input,
            &history,
            &layer_bounds,
            &beta_state,
            &alpha_state,
            &cut_pool,
            0,
            &partial_parent,
            None,
        )
        .unwrap();

    let (full_lb, full_ub) = bounds_min_max(&full_bounds);
    let (partial_lb, partial_ub) = bounds_min_max(&partial_bounds);

    assert!(
        (full_lb - partial_lb).abs() < 1e-5,
        "full/partial lower mismatch: full={full_lb}, partial={partial_lb}"
    );
    assert!(
        (full_ub - partial_ub).abs() < 1e-5,
        "full/partial upper mismatch: full={full_ub}, partial={partial_ub}"
    );
    assert_eq!(
        partial_intermediate.start_layer,
        network.layers.len() - 1,
        "partial parent must force full recomputation"
    );
    assert_eq!(
        partial_intermediate.bounds_at_layer.len(),
        network.layers.len(),
        "recompute must rebuild full intermediate bounds"
    );
}

/// Regression test for #2595: DomainAlphaState Adam must reset m/v on NaN gradient.
#[ntest::timeout(10000)]
#[test]
fn test_domain_alpha_state_nan_gradient_resets_adam_state_2595() {
    use crate::beta_crown::state::AlphaNeuronState;

    let mut state = DomainAlphaState::empty();
    state.neurons.insert((1, 0), AlphaNeuronState::new(0.5));
    let config = AdaptiveOptConfig::default();

    // Normal gradient to establish non-zero Adam state
    state.neurons.get_mut(&(1, 0)).unwrap().grad = 0.5;
    state.gradient_step_adam(&config, 1);
    let n = state.neurons.get(&(1, 0)).unwrap();
    assert!(
        n.adam_m != 0.0,
        "adam_m should be non-zero after normal step"
    );

    // Inject NaN gradient
    state.neurons.get_mut(&(1, 0)).unwrap().grad = f32::NAN;
    state.gradient_step_adam(&config, 2);

    let n = state.neurons.get(&(1, 0)).unwrap();
    assert_eq!(n.adam_m, 0.0, "adam_m should reset to 0.0 on NaN gradient");
    assert_eq!(n.adam_v, 0.0, "adam_v should reset to 0.0 on NaN gradient");
    assert_eq!(
        n.adam_v_max, 0.0,
        "adam_v_max should reset to 0.0 on NaN gradient"
    );
    assert!(!n.alpha.is_nan(), "alpha should not be NaN");
}

/// Regression test for #2608: DomainAlphaState SGD+momentum must reset velocity on NaN gradient.
///
/// Without the guard, NaN gradient corrupts the velocity field permanently:
/// - NaN grad → velocity = momentum * v + lr * NaN = NaN
/// - sanitize_alpha(alpha + NaN) = 0.5 (caught!)
/// - Next normal grad → velocity = momentum * NaN + lr * grad = NaN (permanent!)
/// - Alpha stuck at 0.5 forever
#[ntest::timeout(10000)]
#[test]
fn test_domain_alpha_state_nan_gradient_resets_velocity_2608() {
    use crate::beta_crown::state::AlphaNeuronState;

    let mut state = DomainAlphaState::empty();
    state.neurons.insert((1, 0), AlphaNeuronState::new(0.5));

    // Normal gradient to establish non-zero velocity
    state.neurons.get_mut(&(1, 0)).unwrap().grad = 0.5;
    state.gradient_step(0.1, 0.9);
    let n = state.neurons.get(&(1, 0)).unwrap();
    assert!(
        n.velocity != 0.0,
        "velocity should be non-zero after normal step"
    );
    let _alpha_after_normal = n.alpha;

    // Inject NaN gradient — velocity should be reset to 0.0
    state.neurons.get_mut(&(1, 0)).unwrap().grad = f32::NAN;
    state.gradient_step(0.1, 0.9);
    let n = state.neurons.get(&(1, 0)).unwrap();
    assert_eq!(
        n.velocity, 0.0,
        "velocity should reset to 0.0 on NaN gradient"
    );
    assert!(
        !n.alpha.is_nan(),
        "alpha should not be NaN after NaN gradient"
    );

    // Normal gradient after recovery — alpha should NOT be stuck at 0.5
    state.neurons.get_mut(&(1, 0)).unwrap().grad = 0.5;
    state.gradient_step(0.1, 0.9);
    let n = state.neurons.get(&(1, 0)).unwrap();
    assert!(
        n.velocity != 0.0,
        "velocity should be non-zero after recovery step"
    );
    // Alpha should have moved from the NaN-recovery midpoint (0.5).
    // Note: alpha may coincidentally equal alpha_after_normal because sanitize_alpha
    // snaps NaN to 0.5 and the same gradient reproduces the same trajectory.
    assert!(n.alpha != 0.5, "alpha stuck at NaN-recovery midpoint 0.5");
}
