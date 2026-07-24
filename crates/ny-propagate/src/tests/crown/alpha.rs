// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN and α-CROWN algorithm tests.

use super::helpers::total_width;
use super::*;
use crate::network::NetworkAlphaCrownExt;
use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

// ============================================================
// α-CROWN TESTS
// ============================================================

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_soundness() {
    // Test that α-CROWN bounds are sound (contain actual outputs)
    // Network: Linear -> ReLU -> Linear
    let mut network = Network::new();

    // First linear: 2 inputs -> 3 outputs
    let w1 = arr2(&[[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Second linear: 3 inputs -> 2 outputs
    let w2 = arr2(&[[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]);
    let b2 = arr1(&[0.0, 0.1]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap(),
    ));

    // Input bounds: x in [-0.5, 0.5]
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Get α-CROWN bounds
    let alpha_crown_output = network.propagate_alpha_crown(&input).unwrap();

    // Sample concrete points and verify they're within bounds
    let test_points = vec![
        vec![-0.5, -0.5],
        vec![0.5, 0.5],
        vec![0.0, 0.0],
        vec![-0.5, 0.5],
        vec![0.5, -0.5],
        vec![0.25, -0.25],
    ];

    for point in test_points {
        // Forward pass through network
        let x = arr1(&point);

        // Layer 1: Linear
        let z1 = w1.dot(&x) + &b1;
        // Layer 2: ReLU
        let a1 = z1.mapv(|v| v.max(0.0));
        // Layer 3: Linear
        let z2 = w2.dot(&a1) + &b2;

        // Verify output is within bounds
        for i in 0..z2.len() {
            assert!(
                z2[i] >= alpha_crown_output.lower()[[i]] - 1e-5,
                "α-CROWN lower bound violated at output {}: {} < {}",
                i,
                z2[i],
                alpha_crown_output.lower()[[i]]
            );
            assert!(
                z2[i] <= alpha_crown_output.upper()[[i]] + 1e-5,
                "α-CROWN upper bound violated at output {}: {} > {}",
                i,
                z2[i],
                alpha_crown_output.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_falls_back_for_silu() {
    let mut network = Network::new();
    let w1 = arr2(&[[1.0]]);
    let b1 = arr1(&[0.0]);
    let w2 = arr2(&[[2.0]]);
    let b2 = arr1(&[0.5]);

    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::SiLU(SiLULayer::new()));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let alpha_bounds = network.propagate_alpha_crown(&input).unwrap();
    let crown_bounds = network.propagate_crown(&input).unwrap();

    assert!(
        (alpha_bounds.lower()[[0]] - crown_bounds.lower()[[0]]).abs() < 1e-5,
        "alpha-CROWN lower should match CROWN when SiLU forces fallback"
    );
    assert!(
        (alpha_bounds.upper()[[0]] - crown_bounds.upper()[[0]]).abs() < 1e-5,
        "alpha-CROWN upper should match CROWN when SiLU forces fallback"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_rejects_self_attention_in_sequential_network() {
    // network/alpha_crown.rs explicitly rejects SelfAttention in sequential mode.
    let mut network = Network::new();
    network.add_layer(Layer::SelfAttention(SelfAttentionLayer::standard()));

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.0, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 2.0, 3.0]).into_dyn(),
    )
    .unwrap();

    let err = network.propagate_alpha_crown(&input).unwrap_err();
    match err {
        NyError::UnsupportedConfiguration(msg) => {
            assert!(
                msg.contains("SelfAttention requires a graph network"),
                "unexpected UnsupportedConfiguration message: {msg}"
            );
        }
        other => panic!("expected UnsupportedConfiguration for SelfAttention, got {other}"),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_invprop_single_pass_routes_constraints() {
    let mut network = Network::new();
    let w1 = arr2(&[[1.0]]);
    let b1 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let layer_bounds = network.collect_crown_ibp_bounds(&input).unwrap();
    let pre_activation_bounds = vec![layer_bounds[0].clone()];

    let mut alpha_state =
        AlphaState::from_preactivation_bounds(&pre_activation_bounds, &[0]).unwrap();
    let constraints = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[0.5]), true).unwrap();
    alpha_state.init_invprop_state(constraints, 1).unwrap();

    // Corrected (output-node-only) channel: dualize the VIOLATION region `y_0 <= 0.5`
    // at the output SEED. Net is relu(x), x in [0,1], so y_0 in [0,1]; on the violation
    // region {x : relu(x) <= 0.5} the true output range is [0, 0.5].
    if let Some(state) = alpha_state.invprop_mut() {
        // Seed gammas: neuron dim = output_dim = 1.
        let mut gammas = LayerGammas::new(1, 1, false);
        gammas.gammas[[0, 0, 0]] = 1.0; // lower dual
        gammas.gammas[[1, 0, 0]] = 1.0; // upper dual
        state.add_layer_gammas(invprop::INVPROP_OUTPUT_SEED.to_string(), gammas);
    }

    let bounds_with = NetworkAlphaCrownExt::propagate_alpha_crown_single_pass_impl(
        &network,
        &input,
        &layer_bounds,
        &alpha_state,
        None,
    )
    .unwrap();

    let mut alpha_state_no = alpha_state.clone();
    if let Some(state) = alpha_state_no.invprop_mut() {
        if let Some(gammas) = state.layer_gammas_mut(invprop::INVPROP_OUTPUT_SEED) {
            gammas.gammas.fill(0.0);
        }
    }

    let bounds_without = NetworkAlphaCrownExt::propagate_alpha_crown_single_pass_impl(
        &network,
        &input,
        &layer_bounds,
        &alpha_state_no,
        None,
    )
    .unwrap();

    // (1) Routing: the seed dual must actually change the concretized bound.
    assert!(
        (bounds_with.lower()[[0]] - bounds_without.lower()[[0]]).abs() > 1e-6
            || (bounds_with.upper()[[0]] - bounds_without.upper()[[0]]).abs() > 1e-6,
        "expected the seed INVPROP dual to change at least one bound: with=[{},{}] without=[{},{}]",
        bounds_with.lower()[[0]],
        bounds_with.upper()[[0]],
        bounds_without.lower()[[0]],
        bounds_without.upper()[[0]]
    );

    // (2) Soundness on the violation region: the assume-violation bound must still
    // ENCLOSE the true output range [0, 0.5] there (lower <= 0, upper >= 0.5). A
    // wrong/suboptimal gamma may loosen, but never breaches soundness.
    assert!(
        bounds_with.lower()[[0]] <= 0.0 + 1e-6,
        "assume-violation lower must be <= true min 0: got {}",
        bounds_with.lower()[[0]]
    );
    assert!(
        bounds_with.upper()[[0]] >= 0.5 - 1e-6,
        "assume-violation upper must be >= true max 0.5: got {}",
        bounds_with.upper()[[0]]
    );
    assert!(
        bounds_with.lower()[[0]].is_finite() && bounds_with.upper()[[0]].is_finite(),
        "with-bounds must be finite (not degraded)"
    );
}

/// Stage-2 end-to-end oracle: the seed-dual gamma ascent (projected SPSA) drives the
/// assume-violation bound to INFEASIBILITY on an empty violation region => HOLD, while
/// gamma == 0 (optimization OFF) stays feasible.
///
/// Net: y0 = relu(x), x in [-1,1] => y0 in [0,1] (UNSTABLE relu, so the alpha loop
/// actually runs). Violation region {y0 <= -2} is EMPTY. With ascent, alpha drives the
/// relu lower slope to 0 while gamma_l grows, so L(gamma_l) ~ -rhs*gamma_l = 2*gamma_l
/// rises past U ~ 1; the loop's infeasibility check fires (lower > upper) => HOLD.
#[ntest::timeout(30000)]
#[test]
fn test_alpha_crown_invprop_gamma_ascent_reaches_infeasibility() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    // Violation region {y0 <= -2} is empty (y0 = relu(x) >= 0).
    let oc = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[-2.0]), true).unwrap();

    let invprop_on = InvpropConfig {
        enabled: true,
        optimize_gammas: true,
        gamma_lr: 0.5,
        ..Default::default()
    };
    let config_on = AlphaCrownConfig {
        iterations: 20,
        invprop: invprop_on,
        output_constraints: Some(oc.clone()),
        ..Default::default()
    };
    let bounds_on =
        NetworkAlphaCrownExt::propagate_alpha_crown_with_config_impl(&network, &input, &config_on)
            .unwrap();

    let invprop_off = InvpropConfig {
        enabled: true,
        optimize_gammas: false,
        ..Default::default()
    };
    let config_off = AlphaCrownConfig {
        iterations: 20,
        invprop: invprop_off,
        output_constraints: Some(oc),
        ..Default::default()
    };
    let bounds_off =
        NetworkAlphaCrownExt::propagate_alpha_crown_with_config_impl(&network, &input, &config_off)
            .unwrap();

    // Optimization ON: gamma ascent reaches infeasibility => HOLD (lower > upper sentinel).
    assert!(
        bounds_on.lower()[[0]] > bounds_on.upper()[[0]],
        "gamma ascent should reach infeasibility (HOLD): got [{}, {}]",
        bounds_on.lower()[[0]],
        bounds_on.upper()[[0]]
    );
    // Optimization OFF: gamma stays 0 => feasible baseline [5,6], no false HOLD.
    assert!(
        bounds_off.lower()[[0]] <= bounds_off.upper()[[0]] + 1e-4,
        "gamma=0 must stay feasible (no infeasibility): got [{}, {}]",
        bounds_off.lower()[[0]],
        bounds_off.upper()[[0]]
    );
}

/// Regression for #1803/#2977: α-CROWN ReLU NaN guard now rejects non-finite
/// pre-activation with NumericalInstability via domain_guard.
#[test]
fn test_relu_alpha_crown_nan_guard_sets_infinite_upper_bias() {
    let layer = ReLULayer::new();
    let incoming = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![-1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    let pre_activation =
        BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())
            .unwrap();
    let alpha = arr1(&[0.5]);

    let result = layer.propagate_linear_with_alpha(&incoming, &pre_activation, &alpha, None);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "α-CROWN with NaN pre-activation should trigger domain_guard: got {:?}",
        result
    );
}

/// Regression test for #2584 item 4: near-zero-width crossing ReLU intervals
/// in the alpha-CROWN backward path must remain finite and sound.
#[ntest::timeout(10000)]
#[test]
fn test_relu_alpha_crown_near_zero_width_crossing_soundness() {
    let layer = ReLULayer::new();
    let epsilon = 1e-12_f32;
    let incoming = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[-epsilon]).into_dyn(), arr1(&[epsilon]).into_dyn()).unwrap();
    let alpha = arr1(&[0.5_f32]);

    let (backward, _gradient, _gradient_upper) = layer
        .propagate_linear_with_alpha(&incoming, &pre_activation, &alpha, None)
        .expect("alpha-CROWN ReLU backward should succeed on near-zero crossing interval");

    assert!(
        backward.lower_a.iter().all(|v| v.is_finite()),
        "lower_a contains non-finite values"
    );
    assert!(
        backward.upper_a.iter().all(|v| v.is_finite()),
        "upper_a contains non-finite values"
    );
    assert!(
        backward.lower_b.iter().all(|v| v.is_finite()),
        "lower_b contains non-finite values"
    );
    assert!(
        backward.upper_b.iter().all(|v| v.is_finite()),
        "upper_b contains non-finite values"
    );

    let tol = 1e-12_f32;
    for x in [-epsilon, 0.0_f32, epsilon] {
        let point = BoundedTensor::new(arr1(&[x]).into_dyn(), arr1(&[x]).into_dyn()).unwrap();
        let concrete = backward.concretize(&point);
        let y = x.max(0.0);
        let lower = concrete.lower()[[0]];
        let upper = concrete.upper()[[0]];

        assert!(
            lower <= y + tol,
            "alpha-CROWN lower unsound at x={x}: lower={lower} > ReLU(x)={y}"
        );
        assert!(
            upper >= y - tol,
            "alpha-CROWN upper unsound at x={x}: upper={upper} < ReLU(x)={y}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_at_least_as_tight_as_crown() {
    // α-CROWN should produce bounds at least as tight as CROWN
    // (or equal when optimization doesn't help)
    let mut network = Network::new();

    // Create a network with crossing ReLUs
    let w1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
    let b1 = arr1(&[0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[[1.0, 1.0]]);
    let b2 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    // Input bounds that create crossing ReLUs
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let crown_output = network.propagate_crown(&input).unwrap();
    let alpha_crown_output = network.propagate_alpha_crown(&input).unwrap();

    // α-CROWN should have tighter or equal bounds
    // Lower bounds should be >= CROWN's lower bounds
    // Upper bounds should be <= CROWN's upper bounds
    // Due to numerical precision, allow small tolerance
    let tol = 1e-4;

    for i in 0..crown_output.len() {
        // α-CROWN lower should be >= CROWN lower (or very close)
        assert!(
            alpha_crown_output.lower()[[i]] >= crown_output.lower()[[i]] - tol,
            "α-CROWN lower bound {} at {} is worse than CROWN {}",
            alpha_crown_output.lower()[[i]],
            i,
            crown_output.lower()[[i]]
        );
        // α-CROWN upper should be <= CROWN upper (or very close)
        assert!(
            alpha_crown_output.upper()[[i]] <= crown_output.upper()[[i]] + tol,
            "α-CROWN upper bound {} at {} is worse than CROWN {}",
            alpha_crown_output.upper()[[i]],
            i,
            crown_output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_deep_network() {
    // Test α-CROWN on a deeper network where optimization should help more
    let mut network = Network::new();

    // 4-layer MLP: 2 -> 4 -> 4 -> 2
    let w1 = arr2(&[[0.5, 0.3], [-0.4, 0.6], [0.2, -0.3], [-0.1, 0.4]]);
    let b1 = arr1(&[0.1, -0.1, 0.0, 0.05]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[
        [0.3, -0.2, 0.4, 0.1],
        [-0.3, 0.5, -0.1, 0.2],
        [0.2, 0.1, -0.3, 0.4],
        [0.1, -0.4, 0.2, -0.1],
    ]);
    let b2 = arr1(&[0.0, 0.1, -0.05, 0.02]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w3 = arr2(&[[0.4, 0.3, -0.2, 0.1], [-0.3, 0.2, 0.4, -0.1]]);
    let b3 = arr1(&[0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // All methods should give sound bounds
    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();
    let alpha_crown_output = network.propagate_alpha_crown(&input).unwrap();

    // Verify soundness by sampling
    let test_points = vec![
        vec![0.0, 0.0],
        vec![-0.5, -0.5],
        vec![0.5, 0.5],
        vec![0.3, -0.2],
        vec![-0.1, 0.4],
    ];

    for point in test_points {
        // Manual forward pass
        let x: Array1<f32> = arr1(&point);
        let z1 = arr2(&[[0.5_f32, 0.3], [-0.4, 0.6], [0.2, -0.3], [-0.1, 0.4]]).dot(&x)
            + arr1(&[0.1_f32, -0.1, 0.0, 0.05]);
        let a1 = z1.mapv(|v: f32| v.max(0.0));
        let z2 = arr2(&[
            [0.3_f32, -0.2, 0.4, 0.1],
            [-0.3, 0.5, -0.1, 0.2],
            [0.2, 0.1, -0.3, 0.4],
            [0.1, -0.4, 0.2, -0.1],
        ])
        .dot(&a1)
            + arr1(&[0.0_f32, 0.1, -0.05, 0.02]);
        let a2 = z2.mapv(|v: f32| v.max(0.0));
        let z3 = arr2(&[[0.4_f32, 0.3, -0.2, 0.1], [-0.3, 0.2, 0.4, -0.1]]).dot(&a2)
            + arr1(&[0.0_f32, 0.0]);

        for i in 0..z3.len() {
            // All methods should be sound
            assert!(
                z3[i] >= ibp_output.lower()[[i]] - 1e-5,
                "IBP violated at {}: {} < {}",
                i,
                z3[i],
                ibp_output.lower()[[i]]
            );
            assert!(
                z3[i] >= crown_output.lower()[[i]] - 1e-5,
                "CROWN violated at {}: {} < {}",
                i,
                z3[i],
                crown_output.lower()[[i]]
            );
            assert!(
                z3[i] >= alpha_crown_output.lower()[[i]] - 1e-5,
                "α-CROWN violated at {}: {} < {}",
                i,
                z3[i],
                alpha_crown_output.lower()[[i]]
            );
        }
    }

    // IBP should be loosest, CROWN tighter, α-CROWN tightest (or equal)
    let ibp_width = total_width(&ibp_output);
    let crown_width = total_width(&crown_output);
    let alpha_crown_width = total_width(&alpha_crown_output);

    // CROWN should be at least as tight as IBP
    assert!(
        crown_width <= ibp_width + 1e-4,
        "CROWN width {} > IBP width {}",
        crown_width,
        ibp_width
    );

    // α-CROWN should be at least as tight as CROWN
    assert!(
        alpha_crown_width <= crown_width + 1e-4,
        "α-CROWN width {} > CROWN width {}",
        alpha_crown_width,
        crown_width
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_optimization_diagnostic() {
    // Diagnostic test to verify α-CROWN optimization is actually working
    // This test prints detailed info about the optimization process
    let mut network = Network::new();

    // 4-layer MLP with many unstable neurons
    let w1 = arr2(&[[0.5, 0.3], [-0.4, 0.6], [0.2, -0.3], [-0.1, 0.4]]);
    let b1 = arr1(&[0.1, -0.1, 0.0, 0.05]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[
        [0.3, -0.2, 0.4, 0.1],
        [-0.3, 0.5, -0.1, 0.2],
        [0.2, 0.1, -0.3, 0.4],
        [0.1, -0.4, 0.2, -0.1],
    ]);
    let b2 = arr1(&[0.0, 0.1, -0.05, 0.02]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w3 = arr2(&[[0.4, 0.3, -0.2, 0.1], [-0.3, 0.2, 0.4, -0.1]]);
    let b3 = arr1(&[0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w3, Some(b3)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Run IBP to check pre-activation bounds for unstable neurons
    let layer_bounds = network.collect_ibp_bounds(&input).unwrap();
    println!(
        "
=== α-CROWN OPTIMIZATION DIAGNOSTIC ==="
    );
    println!("Layer bounds (pre-activation):");
    for (i, lb) in layer_bounds.iter().enumerate() {
        let lower_flat = lb.lower().as_slice().unwrap_or(&[]);
        let upper_flat = lb.upper().as_slice().unwrap_or(&[]);
        let unstable_count = lower_flat
            .iter()
            .zip(upper_flat.iter())
            .filter(|(l, u)| **l < 0.0 && **u > 0.0)
            .count();
        println!(
            "  Layer {}: {} neurons, {} unstable",
            i,
            lower_flat.len(),
            unstable_count
        );
    }

    // IBP bounds
    let ibp_output = network.propagate_ibp(&input).unwrap();
    let ibp_width = total_width(&ibp_output);
    println!(
        "
IBP:     width = {:.6}",
        ibp_width
    );

    // CROWN bounds
    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_width = total_width(&crown_output);
    println!("CROWN:   width = {:.6}", crown_width);

    // Check gradient values by manual perturbation
    println!(
        "
Gradient check (finite differences):"
    );

    // Test perturbing alpha[0] for first ReLU layer
    // This requires direct access to alpha state, so we'll use a simpler test
    // Perturb input epsilon to see if bounds change
    let eps_perturbations = [0.4, 0.5, 0.6];
    for eps in eps_perturbations {
        let perturbed_input =
            BoundedTensor::new(arr1(&[-eps, -eps]).into_dyn(), arr1(&[eps, eps]).into_dyn())
                .unwrap();
        let crown_p = network.propagate_crown(&perturbed_input).unwrap();
        let alpha_p = network.propagate_alpha_crown(&perturbed_input).unwrap();
        let crown_w = total_width(&crown_p);
        let alpha_w = total_width(&alpha_p);
        println!(
            "  eps={:.1}: CROWN={:.4}, α-CROWN={:.4}, diff={:.4}",
            eps,
            crown_w,
            alpha_w,
            crown_w - alpha_w
        );
    }

    // α-CROWN with various configurations
    println!(
        "
α-CROWN iterations test:"
    );
    for iters in [1, 5, 10, 20, 50, 100] {
        let config = AlphaCrownConfig {
            iterations: iters,
            learning_rate: 0.1,
            tolerance: 1e-10, // Very small to prevent early stopping
            use_momentum: true,
            momentum: 0.9,
            lr_decay: 0.98,
            ..Default::default()
        };
        let alpha_output = network
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap();
        let alpha_width = total_width(&alpha_output);
        let improvement = if crown_width > 0.0 {
            (crown_width - alpha_width) / crown_width * 100.0
        } else {
            0.0
        };
        println!(
            "α-CROWN(iters={:3}): width = {:.6}, improvement vs CROWN: {:+.4}%",
            iters, alpha_width, improvement
        );
    }

    println!(
        "=========================================
"
    );

    // If α-CROWN equals CROWN for all configs, there might be a bug
    // Use explicit config with enough iterations to guarantee improvement
    let config_50 = AlphaCrownConfig {
        iterations: 50,
        learning_rate: 0.1,
        tolerance: 1e-10,
        use_momentum: true,
        momentum: 0.9,
        lr_decay: 0.98,
        ..Default::default()
    };
    let alpha_50 = network
        .propagate_alpha_crown_with_config(&input, &config_50)
        .unwrap();
    let alpha_50_width = total_width(&alpha_50);

    // For a deep network with unstable neurons, α-CROWN should provide some improvement
    // Even 0.1% improvement indicates optimization is working
    let improvement_pct = (crown_width - alpha_50_width) / crown_width * 100.0;
    println!(
        "Final: CROWN width={:.6}, α-CROWN(50 iter) width={:.6}, improvement={:+.4}%",
        crown_width, alpha_50_width, improvement_pct
    );

    // Assert that α-CROWN is at least as tight as CROWN
    assert!(
        alpha_50_width <= crown_width + 1e-4,
        "α-CROWN should not be worse than CROWN: α-CROWN={:.6}, CROWN={:.6}",
        alpha_50_width,
        crown_width
    );

    // Assert that α-CROWN actually improves (this network has 8 unstable neurons)
    assert!(
        improvement_pct > 0.0,
        "α-CROWN should improve over CROWN for this network with unstable neurons"
    );
}

/// Build a 4-ReLU-layer MLP (10→20→20→20→20→5) for gradient method tests.
fn build_spsa_fd_test_network() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    let (input_dim, hidden_dim, output_dim) = (10, 20, 5);
    let w = Array2::<f32>::from_shape_fn((hidden_dim, input_dim), |(i, j)| {
        ((i + j) % 5) as f32 * 0.1 - 0.2
    });
    let b = Array1::from_elem(hidden_dim, 0.0);
    network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b.clone())).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    for _ in 0..3 {
        let w = Array2::<f32>::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
            ((i * 3 + j * 7) % 11) as f32 * 0.05 - 0.25
        });
        network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b.clone())).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
    }
    let w_out = Array2::<f32>::from_shape_fn((output_dim, hidden_dim), |(i, j)| {
        ((i + j * 2) % 7) as f32 * 0.1 - 0.3
    });
    network.add_layer(Layer::Linear(
        LinearLayer::new(w_out, Some(Array1::from_elem(output_dim, 0.1))).unwrap(),
    ));
    let input = BoundedTensor::new(
        Array1::from_elem(input_dim, -0.5_f32).into_dyn(),
        Array1::from_elem(input_dim, 0.5_f32).into_dyn(),
    )
    .unwrap();
    (network, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_spsa_vs_finite_diff() {
    use crate::bounds::GradientMethod;

    // 4 ReLU layers × 20 neurons = 80 total, many crossing
    let (network, input) = build_spsa_fd_test_network();
    let crown_width = total_width(&network.propagate_crown(&input).unwrap());

    let spsa_config = AlphaCrownConfig {
        iterations: 50,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        learning_rate: 0.1,
        lr_decay: 0.99,
        ..Default::default()
    };
    let spsa_width = total_width(
        &network
            .propagate_alpha_crown_with_config(&input, &spsa_config)
            .unwrap(),
    );
    let spsa_improvement = (crown_width - spsa_width) / crown_width * 100.0;

    let fd_config = AlphaCrownConfig {
        iterations: 5,
        gradient_method: GradientMethod::FiniteDifferences,
        learning_rate: 0.5,
        lr_decay: 0.98,
        ..Default::default()
    };
    let fd_width = total_width(
        &network
            .propagate_alpha_crown_with_config(&input, &fd_config)
            .unwrap(),
    );
    let fd_improvement = (crown_width - fd_width) / crown_width * 100.0;

    // Both must be at least as tight as CROWN
    assert!(spsa_width <= crown_width + 1e-4, "SPSA bounds unsound");
    assert!(fd_width <= crown_width + 1e-4, "FD bounds unsound");

    // #2035: Both gradient methods must show measurable improvement on 80-neuron network.
    assert!(
        spsa_improvement > 0.01,
        "SPSA (50 iters) must improve >0.01%. Got {spsa_improvement:.4}%"
    );
    assert!(
        fd_improvement > 0.001,
        "FD (5 iters) must improve >0.001%. Got {fd_improvement:.4}%"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_direct_impact() {
    // #2035: Uses the 4-layer MLP (2→4→4→2) with 8 crossing neurons where
    // the per-neuron alpha heuristic is suboptimal (ignores downstream weights).
    let (network, input) = build_2035_test_network();

    let crown_out = network.propagate_crown(&input).unwrap();
    let alpha_out = network.propagate_alpha_crown(&input).unwrap();

    let crown_width = total_width(&crown_out);
    let alpha_width = total_width(&alpha_out);

    assert!(
        alpha_width <= crown_width + 1e-4,
        "α-CROWN must not be worse than CROWN: {alpha_width:.6} vs {crown_width:.6}"
    );

    // #2035: α-CROWN must measurably improve over CROWN.
    let improvement_pct = (crown_width - alpha_width) / crown_width * 100.0;
    assert!(
        improvement_pct > 0.1,
        "α-CROWN must improve >0.1% over CROWN (#2035). Got {improvement_pct:.4}%"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_config() {
    // Test α-CROWN with custom configuration
    let mut network = Network::new();

    let w1 = arr2(&[[1.0, -1.0]]);
    let b1 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[[1.0]]);
    let b2 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Test with different iteration counts
    let config_1iter = AlphaCrownConfig {
        iterations: 1,
        ..Default::default()
    };
    let config_50iter = AlphaCrownConfig {
        iterations: 50,
        ..Default::default()
    };

    let result_1iter = network
        .propagate_alpha_crown_with_config(&input, &config_1iter)
        .unwrap();
    let result_50iter = network
        .propagate_alpha_crown_with_config(&input, &config_50iter)
        .unwrap();

    // Both should be sound
    // 50 iterations should be at least as tight as 1 iteration
    let width_1 = result_1iter.upper()[[0]] - result_1iter.lower()[[0]];
    let width_50 = result_50iter.upper()[[0]] - result_50iter.lower()[[0]];

    assert!(
        width_50 <= width_1 + 1e-4,
        "More iterations should not make bounds worse: {} > {}",
        width_50,
        width_1
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_no_relu() {
    // Test α-CROWN on network without ReLU (should be same as CROWN)
    let mut network = Network::new();

    let w1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b1 = arr1(&[0.5, -0.5]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));

    let w2 = arr2(&[[1.0, -1.0]]);
    let b2 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let crown_output = network.propagate_crown(&input).unwrap();
    let alpha_crown_output = network.propagate_alpha_crown(&input).unwrap();

    // Should be identical for networks without ReLU
    for i in 0..crown_output.len() {
        assert!(
            (crown_output.lower()[[i]] - alpha_crown_output.lower()[[i]]).abs() < 1e-5,
            "Lower bounds differ at {}: CROWN {} vs α-CROWN {}",
            i,
            crown_output.lower()[[i]],
            alpha_crown_output.lower()[[i]]
        );
        assert!(
            (crown_output.upper()[[i]] - alpha_crown_output.upper()[[i]]).abs() < 1e-5,
            "Upper bounds differ at {}: CROWN {} vs α-CROWN {}",
            i,
            crown_output.upper()[[i]],
            alpha_crown_output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_state_initialization() {
    // Test AlphaState initialization from pre-activation bounds
    let bounds = vec![
        // First ReLU layer: 3 neurons
        // neuron 0: always positive (l=1, u=2)
        // neuron 1: always negative (l=-2, u=-1)
        // neuron 2: crossing (l=-1, u=2)
        BoundedTensor::new(
            arr1(&[1.0, -2.0, -1.0]).into_dyn(),
            arr1(&[2.0, -1.0, 2.0]).into_dyn(),
        )
        .unwrap(),
    ];

    let alpha_state = AlphaState::from_preactivation_bounds(&bounds, &[0]).unwrap();

    assert_eq!(alpha_state.alphas.len(), 1);
    assert_eq!(alpha_state.alphas[0].len(), 3);

    // Check alpha values
    assert!(
        (alpha_state.alphas[0][0] - 1.0).abs() < 1e-5,
        "Positive neuron should have α=1"
    );
    assert!(
        (alpha_state.alphas[0][1] - 0.0).abs() < 1e-5,
        "Negative neuron should have α=0"
    );
    // Crossing with u=2 > -l=1, so adaptive heuristic gives α=1
    assert!(
        (alpha_state.alphas[0][2] - 1.0).abs() < 1e-5,
        "Crossing neuron (u > -l) should have α=1"
    );

    // Check unstable mask
    assert!(
        !alpha_state.unstable_mask[0][0],
        "Positive neuron should not be unstable"
    );
    assert!(
        !alpha_state.unstable_mask[0][1],
        "Negative neuron should not be unstable"
    );
    assert!(
        alpha_state.unstable_mask[0][2],
        "Crossing neuron should be unstable"
    );

    assert_eq!(alpha_state.num_unstable(), 1);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_empty_network() {
    // Test α-CROWN on empty network
    let network = Network::new();
    let input =
        BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn()).unwrap();

    let output = network.propagate_alpha_crown(&input).unwrap();

    // Should return input unchanged
    assert_eq!(output.lower()[[0]], 1.0);
    assert_eq!(output.lower()[[1]], 2.0);
    assert_eq!(output.upper()[[0]], 3.0);
    assert_eq!(output.upper()[[1]], 4.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_adaptive_skip() {
    // Test that adaptive skip works correctly based on ReLU layer count

    // Build a network with many ReLU layers (> threshold of 8)
    let mut deep_network = Network::new();
    for _ in 0..12 {
        // 12 ReLU layers
        let w = Array2::<f32>::from_shape_fn((4, 4), |_| 0.1);
        let b = Array1::zeros(4);
        deep_network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));
        deep_network.add_layer(Layer::ReLU(ReLULayer));
    }
    // Output layer
    let w = Array2::<f32>::from_shape_fn((2, 4), |_| 0.1);
    let b = Array1::zeros(2);
    deep_network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));

    // Create input bounds
    let lower = Array1::from_vec(vec![-0.5, -0.5, -0.5, -0.5]).into_dyn();
    let upper = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Config with adaptive skip enabled (default threshold = 8)
    let config_skip = AlphaCrownConfig {
        iterations: 10,
        adaptive_skip: true,
        adaptive_skip_depth_threshold: 8, // 12 > 8 so should skip
        adaptive_skip_pilot: false,       // Disable pilot for this test
        ..Default::default()
    };

    // Config with adaptive skip disabled
    let config_no_skip = AlphaCrownConfig {
        iterations: 10,
        adaptive_skip: false,
        ..Default::default()
    };

    // Get CROWN bounds for comparison
    let crown_output = deep_network.propagate_crown(&input).unwrap();
    let crown_width = total_width(&crown_output);

    // With adaptive skip enabled (12 ReLU > threshold 8), should skip α-CROWN
    let skip_output = deep_network
        .propagate_alpha_crown_with_config(&input, &config_skip)
        .unwrap();
    let skip_width = total_width(&skip_output);

    // With skip disabled, should run α-CROWN normally
    let no_skip_output = deep_network
        .propagate_alpha_crown_with_config(&input, &config_no_skip)
        .unwrap();
    let no_skip_width = total_width(&no_skip_output);

    println!("Deep network (12 ReLU layers):");
    println!("  CROWN width: {:.6}", crown_width);
    println!("  α-CROWN (skip enabled, threshold=8): {:.6}", skip_width);
    println!("  α-CROWN (skip disabled): {:.6}", no_skip_width);

    // With skip enabled for deep network, should return CROWN bounds (or very close)
    // because it skips the optimization
    assert!(
        (skip_width - crown_width).abs() < 1e-4,
        "With adaptive skip enabled for deep network, should return CROWN bounds. \
         Got skip_width={:.6}, crown_width={:.6}, diff={:.6}",
        skip_width,
        crown_width,
        (skip_width - crown_width).abs()
    );

    // Shallow-network tests moved to test_alpha_crown_adaptive_skip_shallow_improves.
}

/// #2035: Shallow network (< adaptive_skip threshold) must NOT skip α-CROWN
/// and must produce measurable improvement over CROWN.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_adaptive_skip_shallow_improves() {
    // 3 ReLU layers with varied weights (crossing neurons, suboptimal heuristic alpha).
    let mut network = Network::new();
    for layer_idx in 0..3_usize {
        let w = Array2::<f32>::from_shape_fn((4, 4), |(i, j)| {
            ((i + j * 2 + layer_idx * 3) % 5) as f32 * 0.15 - 0.35
        });
        let b = Array1::from_shape_fn(4, |i| (i as f32 - 1.5) * 0.1);
        network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));
        network.add_layer(Layer::ReLU(ReLULayer));
    }
    let w = Array2::<f32>::from_shape_fn((2, 4), |(i, j)| ((i + j * 2) % 6) as f32 * 0.2 - 0.5);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w, Some(Array1::zeros(2))).unwrap(),
    ));

    let input = BoundedTensor::new(
        Array1::from_vec(vec![-0.5, -0.5, -0.5, -0.5]).into_dyn(),
        Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]).into_dyn(),
    )
    .unwrap();

    // adaptive_skip enabled but 3 < threshold 8, so should NOT skip
    let config = AlphaCrownConfig {
        iterations: 10,
        adaptive_skip: true,
        adaptive_skip_depth_threshold: 8,
        adaptive_skip_pilot: false,
        ..Default::default()
    };
    let alpha_width = total_width(
        &network
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap(),
    );
    let crown_width = total_width(&network.propagate_crown(&input).unwrap());

    assert!(
        alpha_width <= crown_width + 1e-4,
        "Shallow α-CROWN worse than CROWN: {alpha_width:.6} vs {crown_width:.6}"
    );
    let improvement_pct = if crown_width > 1e-8 {
        (crown_width - alpha_width) / crown_width * 100.0
    } else {
        0.0
    };
    assert!(
        improvement_pct > 0.01,
        "Shallow α-CROWN must improve >0.01% over CROWN (#2035). Got {improvement_pct:.4}%"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_analytic_chain_gradients() {
    use crate::bounds::GradientMethod;
    use std::time::Instant;

    // Test the new AnalyticChain gradient method against other methods.
    // This test verifies that AnalyticChain produces valid bounds that are
    // at least as tight as CROWN (soundness) and comparable to other gradient methods.

    // Build a small network with multiple ReLU layers to test chain-rule gradients
    let mut network = Network::new();
    let input_dim = 4;
    let hidden_dim = 8;
    let output_dim = 2;

    // Input -> Hidden 1
    let w1 = Array2::<f32>::from_shape_fn((hidden_dim, input_dim), |(i, j)| {
        ((i + j * 2) % 5) as f32 * 0.15 - 0.35
    });
    let b1 = Array1::from_elem(hidden_dim, 0.1);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Hidden 1 -> Hidden 2
    let w2 = Array2::<f32>::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
        ((i * 3 + j) % 7) as f32 * 0.1 - 0.3
    });
    let b2 = Array1::from_elem(hidden_dim, -0.05);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Hidden 2 -> Output
    let w_out = Array2::<f32>::from_shape_fn((output_dim, hidden_dim), |(i, j)| {
        ((i + j * 2) % 6) as f32 * 0.2 - 0.5
    });
    let b_out = Array1::from_elem(output_dim, 0.0);
    network.add_layer(Layer::Linear(LinearLayer::new(w_out, Some(b_out)).unwrap()));

    // Input bounds
    let lower = Array1::from_elem(input_dim, -0.3);
    let upper = Array1::from_elem(input_dim, 0.3);
    let input = BoundedTensor::new(lower.into_dyn(), upper.into_dyn()).unwrap();

    // Get baseline CROWN bounds
    let crown_output = network.propagate_crown(&input).unwrap();
    let crown_width = total_width(&crown_output);

    println!(
        "
=== AnalyticChain Gradient Test ==="
    );
    println!("Network: 2 ReLU layers with {} hidden neurons", hidden_dim);
    println!("CROWN baseline width: {:.6}", crown_width);

    // Common config
    let iterations = 30;
    let learning_rate = 0.15;
    let lr_decay = 0.98;

    // Test SPSA (baseline comparison)
    let spsa_config = AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        learning_rate,
        lr_decay,
        ..Default::default()
    };
    let start = Instant::now();
    let spsa_output = network
        .propagate_alpha_crown_with_config(&input, &spsa_config)
        .unwrap();
    let _spsa_time = start.elapsed();
    let spsa_width = total_width(&spsa_output);
    let _spsa_improvement = (crown_width - spsa_width) / crown_width * 100.0;

    // Test local Analytic gradients
    let analytic_config = AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::Analytic,
        learning_rate,
        lr_decay,
        ..Default::default()
    };
    let start = Instant::now();
    let analytic_output = network
        .propagate_alpha_crown_with_config(&input, &analytic_config)
        .unwrap();
    let _analytic_time = start.elapsed();
    let analytic_width = total_width(&analytic_output);
    let _analytic_improvement = (crown_width - analytic_width) / crown_width * 100.0;

    // Test AnalyticChain (new implementation!)
    let chain_config = AlphaCrownConfig {
        iterations,
        gradient_method: GradientMethod::AnalyticChain,
        learning_rate,
        lr_decay,
        ..Default::default()
    };
    let start = Instant::now();
    let chain_output = network
        .propagate_alpha_crown_with_config(&input, &chain_config)
        .unwrap();
    let _chain_time = start.elapsed();
    let chain_width = total_width(&chain_output);
    let chain_improvement = (crown_width - chain_width) / crown_width * 100.0;

    // Soundness: all methods must produce valid bounds (at least as tight as CROWN)
    assert!(spsa_width <= crown_width + 1e-4, "SPSA bounds unsound");
    assert!(
        analytic_width <= crown_width + 1e-4,
        "Analytic bounds unsound"
    );
    assert!(
        chain_width <= crown_width + 1e-4,
        "AnalyticChain bounds unsound"
    );

    // All three methods should produce finite bounds
    for i in 0..chain_output.len() {
        assert!(
            chain_output.lower()[[i]].is_finite(),
            "AnalyticChain lower NaN at {i}"
        );
        assert!(
            chain_output.upper()[[i]].is_finite(),
            "AnalyticChain upper NaN at {i}"
        );
    }

    // #2035: AnalyticChain must produce measurable improvement over CROWN.
    assert!(
        chain_improvement > 0.1,
        "AnalyticChain must improve >0.1% over CROWN. Got {chain_improvement:.2}%"
    );
}

/// Regression test for #1946: AnalyticChain must not silently swallow errors
/// from `propagate_alpha_crown_with_intermediates_impl`.
///
/// Before the fix, AnalyticChain used `unwrap_or_else(|_| default())` which
/// swallowed all errors and substituted empty intermediates. The fallback path
/// then silently degraded to local gradients without surfacing the failure.
///
/// After the fix, real errors propagate via `?`. Only the legitimate case
/// (empty intermediates from unsupported layers) falls back to local gradients
/// with a warning.
///
/// This test verifies that AnalyticChain on a well-formed network produces
/// bounds that are meaningfully optimized (not degraded by silent error swallowing).
#[ntest::timeout(10000)]
#[test]
fn test_analytic_chain_does_not_silently_swallow_errors_1946() {
    // Build a simple Linear -> ReLU -> Linear network.
    let mut network = Network::new();

    let w1 = arr2(&[[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]);
    let b2 = arr1(&[0.0, 0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // AnalyticChain should produce valid bounds (not error, not zeros).
    let chain_config = AlphaCrownConfig {
        iterations: 10,
        gradient_method: GradientMethod::AnalyticChain,
        learning_rate: 0.1,
        lr_decay: 0.98,
        adaptive_skip: false,
        ..Default::default()
    };
    let chain_result = network.propagate_alpha_crown_with_config(&input, &chain_config);
    assert!(
        chain_result.is_ok(),
        "AnalyticChain on well-formed network should not error: {:?}",
        chain_result.err()
    );
    let chain_bounds = chain_result.unwrap();

    // Verify bounds are finite and non-degenerate.
    for i in 0..chain_bounds.shape()[0] {
        let lo = chain_bounds.lower()[[i]];
        let hi = chain_bounds.upper()[[i]];
        assert!(lo.is_finite(), "lower[{i}] must be finite, got {lo}");
        assert!(hi.is_finite(), "upper[{i}] must be finite, got {hi}");
        assert!(lo <= hi, "lower[{i}]={lo} must be <= upper[{i}]={hi}");
    }

    // Compare with plain CROWN: AnalyticChain should be at least as tight
    // (or very close). If intermediates were silently zeroed, the alpha
    // optimization would degrade unpredictably.
    let crown_bounds = network.propagate_crown(&input).unwrap();
    for i in 0..crown_bounds.shape()[0] {
        let crown_width = crown_bounds.upper()[[i]] - crown_bounds.lower()[[i]];
        let chain_width = chain_bounds.upper()[[i]] - chain_bounds.lower()[[i]];
        // Alpha-CROWN optimizes lower slopes; it should produce bounds at least
        // as tight as plain CROWN. Only additive f32 tolerance is needed.
        assert!(
            chain_width <= crown_width + 1e-5,
            "AnalyticChain output[{i}] width {chain_width:.6} exceeds \
             CROWN width {crown_width:.6} — possible silent intermediate failure (#1946)",
        );
    }
}

/// Regression test for #1939: non-contiguous arrays must produce correct sums.
///
/// Before the fix, α-CROWN convergence tracking used `as_slice().map(sum).unwrap_or(-inf)`
/// which returns -inf (or 0.0) for non-contiguous arrays (as_slice() returns None).
/// The fix uses layout-agnostic `iter().sum()` instead.
#[ntest::timeout(10000)]
#[test]
fn test_lower_sum_non_contiguous_layout_1939() {
    use ndarray::ArrayD;

    // Build a non-contiguous array via transpose view.
    let arr = ArrayD::from_shape_vec(vec![3, 2], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let non_contig_view = arr.t();
    assert!(
        non_contig_view.as_slice().is_none(),
        "test setup: transposed view must be non-contiguous"
    );

    // The old pattern: as_slice().map(sum).unwrap_or(-inf) → -inf for non-contiguous
    let old_result: f32 = non_contig_view
        .as_slice()
        .map(|s| s.iter().sum())
        .unwrap_or(f32::NEG_INFINITY);
    assert_eq!(
        old_result,
        f32::NEG_INFINITY,
        "old pattern returns -inf for non-contiguous"
    );

    // The new pattern: iter().sum() → correct sum regardless of layout
    let new_result: f32 = non_contig_view.iter().sum();
    assert!(
        (new_result - 21.0).abs() < 1e-6,
        "iter().sum() must produce correct result (21.0) for non-contiguous array, got {new_result}"
    );
}

/// Regression test for #2013: duplicate init_invprop_state must return an error,
/// not panic via assert!.
#[ntest::timeout(10000)]
#[test]
fn test_init_invprop_state_duplicate_returns_error() {
    // Minimal AlphaState: one ReLU layer with 1 neuron
    let pre_bounds =
        vec![BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap()];
    let mut alpha_state = AlphaState::from_preactivation_bounds(&pre_bounds, &[0]).unwrap();
    let constraints = OutputConstraints::new(arr2(&[[1.0]]), arr1(&[0.5]), true).unwrap();

    // First init succeeds
    alpha_state
        .init_invprop_state(constraints.clone(), 1)
        .unwrap();

    // Second init must return InternalError, not panic
    let err = alpha_state.init_invprop_state(constraints, 1).unwrap_err();
    match err {
        NyError::InternalError(msg) => {
            assert!(
                msg.contains("already-initialized"),
                "unexpected InternalError message: {msg}"
            );
        }
        other => panic!("expected InternalError, got {other:?}"),
    }
}

/// Regression test for #1935: SPSA numerical gradients must propagate CROWN
/// single-pass errors instead of silently substituting concrete_bounds.
///
/// Before the fix, `.unwrap_or_else(|_| concrete_bounds.clone())` swallowed
/// all CROWN errors, producing zero gradients when both +eps and -eps passes
/// failed (both returned concrete_bounds → diff = 0).
///
/// After the fix, `?` propagates errors so the caller sees the failure.
/// This test verifies that calling `propagate_alpha_crown_single_pass_impl`
/// with a wrong-sized alpha state returns Err (not silently Ok).
#[test]
fn test_single_pass_error_propagates_not_swallowed_1935() {
    // Build a simple Linear -> ReLU -> Linear network.
    let mut network = Network::new();
    let w1 = arr2(&[[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w2 = arr2(&[[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]);
    let b2 = arr1(&[0.0, 0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Compute valid per-layer bounds via IBP.
    let layer_bounds = network.collect_ibp_bounds(&input).unwrap();

    // Create an alpha state with WRONG-SIZED alphas (7 instead of 3).
    // This triggers ShapeMismatch in propagate_linear_with_alpha.
    let mut alpha_state = AlphaState {
        alphas: vec![Array1::from_elem(7, 0.5)], // wrong size: 7 vs 3 neurons
        alphas_upper: vec![Array1::from_elem(7, 0.5)],
        unstable_mask: vec![Array1::from_elem(7, true)],
        velocity: vec![Array1::zeros(7)],
        adam_m: vec![Array1::zeros(7)],
        adam_v: vec![Array1::zeros(7)],
        velocity_upper: vec![Array1::zeros(7)],
        adam_m_upper: vec![Array1::zeros(7)],
        adam_v_upper: vec![Array1::zeros(7)],
        bilinear_alphas: std::collections::HashMap::new(),
        bilinear_adam_m: std::collections::HashMap::new(),
        bilinear_adam_v: std::collections::HashMap::new(),
        invprop_state: None,
    };
    let _ = &mut alpha_state; // suppress unused_mut if needed

    // Call single_pass_impl directly — this must return Err, not Ok.
    let result =
        network.propagate_alpha_crown_single_pass_impl(&input, &layer_bounds, &alpha_state, None);
    assert!(
        result.is_err(),
        "Single-pass with wrong-sized alpha should return Err, not silently Ok. Got: {:?}",
        result,
    );
}

/// Regression test for #1935: SPSA numerical gradients propagate errors
/// through the full alpha-CROWN optimization loop.
///
/// Verifies that SPSA on a well-formed network produces valid bounds
/// (the fix preserves the happy path).
#[ntest::timeout(10000)]
#[test]
fn test_spsa_alpha_crown_happy_path_produces_valid_bounds_1935() {
    let mut network = Network::new();
    let w1 = arr2(&[[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w2 = arr2(&[[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]);
    let b2 = arr1(&[0.0, 0.1]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    let config = AlphaCrownConfig {
        iterations: 5,
        gradient_method: GradientMethod::Spsa,
        learning_rate: 0.1,
        lr_decay: 0.98,
        adaptive_skip: false,
        spsa_samples: 2,
        ..Default::default()
    };
    let result = network.propagate_alpha_crown_with_config(&input, &config);
    assert!(
        result.is_ok(),
        "SPSA on well-formed network must succeed after error-propagation fix. Got: {:?}",
        result.err(),
    );

    let bounds = result.unwrap();
    for i in 0..bounds.shape()[0] {
        let lo = bounds.lower()[[i]];
        let hi = bounds.upper()[[i]];
        assert!(lo.is_finite(), "lower[{i}] must be finite, got {lo}");
        assert!(hi.is_finite(), "upper[{i}] must be finite, got {hi}");
        assert!(lo <= hi, "lower[{i}]={lo} must be <= upper[{i}]={hi}");
    }
}

// ============================================================
// Sequential Network UnsupportedOp Fallback Tests (#2138)
// ============================================================

/// Sequential Network alpha-CROWN with an unsupported layer (NonZero) must
/// fall back to CROWN/IBP and produce valid bounds, not panic or silently fail.
///
/// NonZero returns `UnsupportedOp` from `propagate_crown_backward`, which
/// triggers the catch-all fallback in the alpha-CROWN backward pass.
/// The fallback calls `propagate_crown_with_engine`, which in turn falls
/// back to IBP for NonZero.
///
/// Part of #2138: verifies the sequential Network alpha-CROWN UnsupportedOp
/// fallback path has coverage (previously zero coverage, only GraphNetwork
/// was tested via `catch_all_error_propagation.rs`).
#[ntest::timeout(10000)]
#[test]
fn test_sequential_alpha_crown_unsupported_op_fallback_2138() {
    // Build: Linear(2→3) → ReLU → NonZero
    // NonZero hits the catch-all `_` arm in alpha-CROWN backward and returns
    // UnsupportedOp, triggering CROWN→IBP fallback.
    let mut network = Network::new();

    let w = arr2(&[[1.0_f32, -0.5], [0.5, 1.0], [-1.0, 0.3]]);
    let b = arr1(&[0.1, -0.2, 0.3]);
    network.add_layer(Layer::Linear(LinearLayer::new(w, Some(b)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::NonZero(NonZeroLayer));

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    // Alpha-CROWN should succeed via fallback, not error or panic.
    let result = network.propagate_alpha_crown(&input);
    assert!(
        result.is_ok(),
        "Sequential alpha-CROWN with NonZero should fall back to IBP, not error. Got: {:?}",
        result.err()
    );

    let bounds = result.unwrap();
    // NonZero output shape is [rank, max_nonzero] — verify bounds are valid and finite.
    for i in 0..bounds.lower().len() {
        let lo = bounds.lower().iter().nth(i).unwrap();
        let hi = bounds.upper().iter().nth(i).unwrap();
        assert!(
            lo <= hi,
            "Fallback bounds must be valid (lower <= upper) at flat index {i}: lo={lo}, hi={hi}"
        );
        assert!(
            lo.is_finite() && hi.is_finite(),
            "Fallback bounds must be finite at flat index {i}: lo={lo}, hi={hi}. \
             Trivial [-inf, +inf] bounds indicate the fallback path is not propagating."
        );
    }

    // Alpha-CROWN fallback must produce bounds at least as tight as IBP.
    // In the NonZero case, the fallback IS IBP, so bounds should be identical.
    let ibp_bounds = network.propagate_ibp(&input).unwrap();
    assert_eq!(
        bounds.shape(),
        ibp_bounds.shape(),
        "Alpha-CROWN fallback and IBP must produce same output shape"
    );
    for i in 0..bounds.lower().len() {
        let alpha_lo = bounds.lower().iter().nth(i).unwrap();
        let alpha_hi = bounds.upper().iter().nth(i).unwrap();
        let ibp_lo = ibp_bounds.lower().iter().nth(i).unwrap();
        let ibp_hi = ibp_bounds.upper().iter().nth(i).unwrap();
        assert!(
            *alpha_lo >= ibp_lo - 1e-5,
            "Alpha-CROWN lower should be >= IBP lower at {i}: alpha={alpha_lo} < ibp={ibp_lo}"
        );
        assert!(
            *alpha_hi <= ibp_hi + 1e-5,
            "Alpha-CROWN upper should be <= IBP upper at {i}: alpha={alpha_hi} > ibp={ibp_hi}"
        );
    }
}

/// Regression test: sequential Network alpha-CROWN catch-all dispatches WhereLayer
/// (with embedded constants) through CROWN fallback when propagate_crown_backward
/// returns UnsupportedOp.
///
/// WhereLayer.propagate_linear returns UnsupportedOp, so the catch-all at
/// alpha_crown.rs:751 must fall back to regular CROWN. This test verifies:
/// 1. Alpha-CROWN succeeds (falls back to CROWN instead of erroring)
/// 2. Bounds are sound (contain sampled concrete outputs)
/// 3. Bounds are at least as tight as IBP
///
/// Part of #2140: zero test coverage for sequential alpha-CROWN UnsupportedOp fallback.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_sequential_unsupported_op_fallback_2140() {
    // Network: Linear(2->3) -> ReLU -> WhereLayer(embedded) -> Linear(3->2)
    //
    // WhereLayer with embedded constants:
    //   output[i] = const_true[i]  if input[i] > 0
    //   output[i] = const_false[i] if input[i] <= 0
    // IBP: union of const_true and const_false ranges.
    // CROWN backward: UnsupportedOp (Where is nonlinear).
    let mut network = Network::new();

    // Layer 0: Linear 2->3
    let w1 = arr2(&[[1.0, 2.0], [-1.0, 1.0], [0.5, -0.5]]);
    let b1 = arr1(&[0.1, -0.2, 0.3]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1.clone(), Some(b1.clone())).unwrap(),
    ));

    // Layer 1: ReLU (needed so alpha-CROWN doesn't skip to plain CROWN)
    network.add_layer(Layer::ReLU(ReLULayer));

    // Layer 2: WhereLayer with embedded constants (triggers UnsupportedOp in CROWN backward)
    let const_true = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let const_false = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 0.5]).unwrap();
    let where_layer = WhereLayer {
        const_true: Some(const_true.clone()),
        const_false: Some(const_false.clone()),
    };
    network.add_layer(Layer::Where(where_layer));

    // Layer 3: Linear 3->2
    let w2 = arr2(&[[1.0, -1.0, 0.5], [0.5, 1.0, -0.5]]);
    let b2 = arr1(&[0.0, 0.1]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2.clone(), Some(b2.clone())).unwrap(),
    ));

    // Input bounds: x in [-0.5, 0.5]
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Alpha-CROWN must succeed (fallback to CROWN, not error)
    let result = network.propagate_alpha_crown(&input);
    assert!(
        result.is_ok(),
        "Alpha-CROWN with UnsupportedOp layer must fall back to CROWN, not error. Got: {:?}",
        result.err()
    );
    let alpha_bounds = result.unwrap();
    assert_eq!(alpha_bounds.shape(), &[2], "Output shape must be [2]");

    // Verify bounds are valid
    for i in 0..2 {
        assert!(
            alpha_bounds.lower()[[i]] <= alpha_bounds.upper()[[i]],
            "Invalid bounds at {i}: lower={} > upper={}",
            alpha_bounds.lower()[[i]],
            alpha_bounds.upper()[[i]]
        );
    }

    // Soundness: sample concrete points and verify they're within bounds
    let test_points: Vec<[f32; 2]> = vec![
        [-0.5, -0.5],
        [0.5, 0.5],
        [0.0, 0.0],
        [-0.5, 0.5],
        [0.5, -0.5],
        [0.25, -0.25],
        [-0.1, 0.3],
    ];

    for point in &test_points {
        let x = arr1(point);

        // Layer 0: Linear
        let z1 = w1.dot(&x) + &b1;
        // Layer 1: ReLU
        let a1 = z1.mapv(|v| v.max(0.0));
        // Layer 2: Where (condition=a1, true=const_true, false=const_false)
        let a2: Array1<f32> = Array1::from_iter((0..3).map(|i| {
            if a1[i] > 0.0 {
                const_true[[i]]
            } else {
                const_false[[i]]
            }
        }));
        // Layer 3: Linear
        let z2 = w2.dot(&a2) + &b2;

        for i in 0..z2.len() {
            assert!(
                z2[i] >= alpha_bounds.lower()[[i]] - 1e-5,
                "Alpha-CROWN lower bound violated at output {i} for point {point:?}: \
                 concrete={} < lower={}",
                z2[i],
                alpha_bounds.lower()[[i]]
            );
            assert!(
                z2[i] <= alpha_bounds.upper()[[i]] + 1e-5,
                "Alpha-CROWN upper bound violated at output {i} for point {point:?}: \
                 concrete={} > upper={}",
                z2[i],
                alpha_bounds.upper()[[i]]
            );
        }
    }

    // Verify alpha-CROWN bounds are at least as tight as IBP
    let ibp_bounds = network.propagate_ibp(&input).unwrap();
    for i in 0..2 {
        assert!(
            alpha_bounds.lower()[[i]] >= ibp_bounds.lower()[[i]] - 1e-5,
            "Alpha-CROWN lower should be >= IBP lower at {i}: alpha={} < ibp={}",
            alpha_bounds.lower()[[i]],
            ibp_bounds.lower()[[i]]
        );
        assert!(
            alpha_bounds.upper()[[i]] <= ibp_bounds.upper()[[i]] + 1e-5,
            "Alpha-CROWN upper should be <= IBP upper at {i}: alpha={} > ibp={}",
            alpha_bounds.upper()[[i]],
            ibp_bounds.upper()[[i]]
        );
    }
}

/// Regression test for #2835: alpha-CROWN with extreme weights that produce
/// non-finite gradients must still return finite, sound bounds. The non-finite
/// gradient guard at alpha_crown.rs:380 should skip the corrupted update,
/// preserving the current alpha values.
///
/// Uses weights large enough to cause overflow in CROWN backward → non-finite
/// gradient → guard skips → optimizer state stays valid.
#[ntest::timeout(60000)]
#[test]
fn test_alpha_crown_non_finite_gradient_guard_extreme_weights_2835() {
    let mut network = Network::new();

    // First linear: extreme weights that can cause overflow in backward pass.
    // The 1e15 entries will produce Inf during SPSA perturbation when multiplied
    // by the alpha-CROWN backward coefficients.
    let w1 = arr2(&[[1e15, -1e15], [-1e15, 1e15]]);
    let b1 = arr1(&[0.0, 0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));

    let w2 = arr2(&[[1.0, 1.0]]);
    let b2 = arr1(&[0.0]);
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let config = AlphaCrownConfig {
        iterations: 5,
        learning_rate: 0.1,
        ..Default::default()
    };

    // The key assertion: alpha-CROWN must not panic or return non-finite bounds.
    // The non-finite gradient guard should silently skip corrupted updates.
    // #2876: Require Ok — no Err escape hatch. Use is_finite() to catch +/-Inf.
    let bounds = network
        .propagate_alpha_crown_with_config(&input, &config)
        .expect("alpha-CROWN must succeed with extreme weights (non-finite gradient guard should handle overflow)");

    for i in 0..bounds.lower().len() {
        assert!(
            bounds.lower()[[i]].is_finite(),
            "alpha-CROWN lower bound must be finite at index {i}, got {}",
            bounds.lower()[[i]]
        );
        assert!(
            bounds.upper()[[i]].is_finite(),
            "alpha-CROWN upper bound must be finite at index {i}, got {}",
            bounds.upper()[[i]]
        );
        // Soundness: lower <= upper must always hold.
        assert!(
            bounds.lower()[[i]] <= bounds.upper()[[i]],
            "alpha-CROWN bounds inverted at index {i}: lower={} > upper={}",
            bounds.lower()[[i]],
            bounds.upper()[[i]]
        );
    }
}

/// Helper: builds the 4-layer MLP (2→4→4→2) used across #2035 regression tests.
/// Has 8 neurons across 2 ReLU layers with many crossing neurons for input ∈ [-0.5, 0.5]².
fn build_2035_test_network() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    let w1 = arr2(&[[0.5, 0.3], [-0.4, 0.6], [0.2, -0.3], [-0.1, 0.4]]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1, Some(arr1(&[0.1, -0.1, 0.0, 0.05]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w2 = arr2(&[
        [0.3, -0.2, 0.4, 0.1],
        [-0.3, 0.5, -0.1, 0.2],
        [0.2, 0.1, -0.3, 0.4],
        [0.1, -0.4, 0.2, -0.1],
    ]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2, Some(arr1(&[0.0, 0.1, -0.05, 0.02]))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w3 = arr2(&[[0.4, 0.3, -0.2, 0.1], [-0.3, 0.2, 0.4, -0.1]]);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w3, Some(arr1(&[0.0, 0.0]))).unwrap(),
    ));
    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    (network, input)
}

/// Regression test for #2035: default α-CROWN (AnalyticChain) must strictly improve
/// bounds over CROWN on a network with unstable neurons.
///
/// Root cause: SPSA with 1 sample produces noise-dominated gradients (O(n) variance).
/// AnalyticChain matches reference α,β-CROWN's `loss.backward()` (`optimized_bounds.py:870`).
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_analytic_chain_improves_over_crown_2035() {
    use crate::bounds::GradientMethod;

    let (network, input) = build_2035_test_network();
    let crown_width = total_width(&network.propagate_crown(&input).unwrap());

    let config = AlphaCrownConfig {
        iterations: 50,
        tolerance: 1e-10,
        ..Default::default()
    };
    assert_eq!(config.gradient_method, GradientMethod::AnalyticChain);

    let alpha_width = total_width(
        &network
            .propagate_alpha_crown_with_config(&input, &config)
            .unwrap(),
    );

    assert!(
        alpha_width <= crown_width + 1e-4,
        "α-CROWN must not be worse than CROWN: {alpha_width:.6} vs {crown_width:.6}"
    );
    let improvement_pct = (crown_width - alpha_width) / crown_width * 100.0;
    assert!(
        improvement_pct > 0.1,
        "α-CROWN must improve >0.1% over CROWN (#2035). Got {improvement_pct:.4}%"
    );
}

/// Regression test for #2035: default α-CROWN on a 4→8→8→2 network must produce
/// measurable improvement over CROWN using the new AnalyticChain default.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_default_improves_on_multi_relu_network_2035() {
    let mut network = Network::new();
    let w1 = Array2::<f32>::from_shape_fn((8, 4), |(i, j)| ((i + j * 2) % 5) as f32 * 0.15 - 0.35);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w1, Some(Array1::from_elem(8, 0.1))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w2 = Array2::<f32>::from_shape_fn((8, 8), |(i, j)| ((i * 3 + j) % 7) as f32 * 0.1 - 0.3);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2, Some(Array1::from_elem(8, -0.05))).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    let w_out = Array2::<f32>::from_shape_fn((2, 8), |(i, j)| ((i + j * 2) % 6) as f32 * 0.2 - 0.5);
    network.add_layer(Layer::Linear(
        LinearLayer::new(w_out, Some(Array1::zeros(2))).unwrap(),
    ));

    let input = BoundedTensor::new(
        Array1::from_elem(4, -0.3_f32).into_dyn(),
        Array1::from_elem(4, 0.3_f32).into_dyn(),
    )
    .unwrap();

    let crown_width = total_width(&network.propagate_crown(&input).unwrap());
    let alpha_width = total_width(&network.propagate_alpha_crown(&input).unwrap());

    assert!(
        alpha_width <= crown_width + 1e-4,
        "α-CROWN must not be worse than CROWN: {alpha_width:.6} vs {crown_width:.6}"
    );
    let improvement_pct = (crown_width - alpha_width) / crown_width * 100.0;
    assert!(
        improvement_pct > 0.1,
        "Default α-CROWN must improve >0.1% (#2035). Got {improvement_pct:.4}%"
    );
}
