// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::prelude::*;

#[ntest::timeout(10000)]
#[test]
fn test_analytical_gradients_direction() {
    // Test that analytical gradients point in a direction that improves bounds
    // when used for gradient ascent (increasing lower bound)
    let network = simple_network();

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Create a split history with one constraint
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let mut beta_state = BetaState::from_history(&history).unwrap();

    // Set initial beta value
    beta_state.set_beta(1, 0, 0.1);

    // Compute bounds before gradient step
    let verifier = BetaCrownVerifier::default();
    let bounds_before = verifier
        .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
        .unwrap();
    let lb_before = bounds_before.lower_scalar();

    // Compute analytical gradient
    verifier
        .compute_beta_gradients(
            &network,
            &input,
            &history,
            &layer_bounds,
            &mut beta_state,
            0,
        )
        .unwrap();

    let grad = beta_state.entries[0].grad;
    println!("Analytical gradient: {}", grad);

    // Take a small step in gradient direction
    let lr = 0.01;
    let new_beta = (beta_state.entries[0].value + lr * grad).max(0.0);
    beta_state.set_beta(1, 0, new_beta);

    // Compute bounds after gradient step
    let bounds_after = verifier
        .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
        .unwrap();
    let lb_after = bounds_after.lower_scalar();

    println!("Lower bound before: {}, after: {}", lb_before, lb_after);

    // Unconditional: gradient must always be finite (catches NaN/Inf bugs in backward pass)
    assert!(
        grad.is_finite(),
        "Analytical gradient must be finite, got {}",
        grad
    );

    // For gradient ascent, if grad > 0 and we increase beta, lb should increase (or stay same).
    // If grad is near-zero, the bound may not change — but we still verify the direction
    // invariant holds: a step along the gradient must not significantly worsen the bound.
    let improvement = lb_after - lb_before;
    println!("Improvement: {}", improvement);
    if grad.abs() > 1e-6 {
        assert!(
            improvement >= -1e-3,
            "Gradient step should not significantly decrease bound: \
             grad={}, improvement={}, lb_before={}, lb_after={}",
            grad,
            improvement,
            lb_before,
            lb_after
        );
    } else {
        // Near-zero gradient: bound change should also be near-zero
        assert!(
            improvement.abs() < 1e-3,
            "Near-zero gradient (|grad|={}) should produce near-zero bound change, \
             got improvement={}",
            grad.abs(),
            improvement
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_compute_beta_gradients_empty_layer_bounds_errors_2095() {
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

    let mut beta_state = BetaState::from_history(&history).unwrap();
    let layer_bounds: Vec<Arc<BoundedTensor>> = Vec::new();
    let verifier = BetaCrownVerifier::default();

    let err = verifier
        .compute_beta_gradients(
            &network,
            &input,
            &history,
            &layer_bounds,
            &mut beta_state,
            0,
        )
        .expect_err("empty layer_bounds must error");
    assert!(
        matches!(err, ny_core::NyError::InvalidSpec(ref msg) if msg.contains("layer_bounds length 0 does not match network layers")),
        "expected InvalidSpec for empty layer_bounds, got {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_compute_beta_gradients_skips_non_finite_beta_2826() {
    let network = simple_network();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    for is_active in [true, false] {
        let mut history = SplitHistory::new();
        history.add_constraint(NeuronConstraint {
            layer_idx: 1,
            neuron_idx: 0,
            is_active,
            score: 0.0,
        });

        let layer_bounds: Vec<Arc<BoundedTensor>> = network
            .collect_ibp_bounds(&input)
            .unwrap()
            .into_iter()
            .map(Arc::new)
            .collect();

        let mut zero_beta_state = BetaState::from_history(&history).unwrap();
        let mut inf_beta_state = BetaState::from_history(&history).unwrap();
        inf_beta_state.entries[0].value = f32::INFINITY;

        let verifier = BetaCrownVerifier::default();
        verifier
            .compute_beta_gradients(
                &network,
                &input,
                &history,
                &layer_bounds,
                &mut zero_beta_state,
                0,
            )
            .expect("zero-beta gradients should compute");
        verifier
            .compute_beta_gradients(
                &network,
                &input,
                &history,
                &layer_bounds,
                &mut inf_beta_state,
                0,
            )
            .expect("non-finite beta should be skipped during gradient computation");

        let label = if is_active { "+inf" } else { "-inf" };
        let zero_grad = zero_beta_state.entries[0].grad;
        let inf_grad = inf_beta_state.entries[0].grad;
        assert!(
            inf_grad.is_finite(),
            "gradient should remain finite when signed beta is {label}, got {inf_grad}"
        );
        assert!(
            (inf_grad - zero_grad).abs() <= 1e-6,
            "non-finite beta gradient must match zero-beta baseline ({label}): got {inf_grad}, expected {zero_grad}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_analytical_vs_numerical_gradient_consistency() {
    // Test that analytical gradients are consistent with numerical gradients
    // in terms of sign and rough magnitude
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

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(1, 0, 0.5);

    let verifier = BetaCrownVerifier::default();

    // Compute analytical gradient
    verifier
        .compute_beta_gradients(
            &network,
            &input,
            &history,
            &layer_bounds,
            &mut beta_state,
            0,
        )
        .unwrap();
    let analytical_grad = beta_state.entries[0].grad;

    // Compute numerical gradient for comparison
    let eps = 1e-4;
    let original_beta = 0.5;

    beta_state.set_beta(1, 0, original_beta + eps);
    let bounds_plus = verifier
        .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
        .unwrap();
    let lb_plus = bounds_plus.lower_scalar();

    beta_state.set_beta(1, 0, (original_beta - eps).max(0.0));
    let bounds_minus = verifier
        .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_state)
        .unwrap();
    let lb_minus = bounds_minus.lower_scalar();

    let numerical_grad = (lb_plus - lb_minus) / (2.0 * eps);

    println!("Analytical gradient: {}", analytical_grad);
    println!("Numerical gradient: {}", numerical_grad);

    let abs_err = (analytical_grad - numerical_grad).abs();
    let tol = 1e-2 + 1e-2 * numerical_grad.abs();
    assert!(
        abs_err <= tol,
        "Analytical gradient should match numerical: analytical={}, numerical={}, abs_err={}, tol={}",
        analytical_grad,
        numerical_grad,
        abs_err,
        tol
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_analytical_gradient_multiple_constraints() {
    // Test analytical gradients with multiple constraints
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let w2 = arr2(&[[0.5, -0.3, 0.7, 0.1], [-0.4, 0.6, -0.2, 0.5]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let w3 = arr2(&[[1.0, -0.5]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();

    // Create history with multiple constraints on different layers
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
    history.add_constraint(NeuronConstraint {
        layer_idx: 3,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let mut beta_state = BetaState::from_history(&history).unwrap();

    // Set initial beta values
    for entry in &mut beta_state.entries {
        entry.set_value(0.1);
    }

    let verifier = BetaCrownVerifier::default();
    verifier
        .compute_beta_gradients(
            &network,
            &input,
            &history,
            &layer_bounds,
            &mut beta_state,
            0,
        )
        .unwrap();

    println!("Gradients for multiple constraints:");
    for entry in &beta_state.entries {
        println!(
            "  Layer {}, Neuron {}, sign={}, beta={}, grad={}",
            entry.layer_idx,
            entry.neuron_idx,
            entry.sign,
            entry.value(),
            entry.grad()
        );
    }

    // All gradients must be finite (catches NaN AND Inf)
    for entry in &beta_state.entries {
        assert!(
            entry.grad().is_finite(),
            "Gradient must be finite for layer={}, neuron={}, sign={}, got {}",
            entry.layer_idx,
            entry.neuron_idx,
            entry.sign,
            entry.grad()
        );
    }

    // At least one gradient must be non-zero (proving the computation actually ran)
    let any_nonzero = beta_state.entries.iter().any(|e| e.grad().abs() > 1e-10);
    assert!(
        any_nonzero,
        "At least one gradient should be non-zero for a multi-constraint setup; \
         grads: {:?}",
        beta_state
            .entries
            .iter()
            .map(|e| e.grad())
            .collect::<Vec<_>>()
    );

    // Cross-validate: for each constraint with a sufficiently large gradient,
    // verify sign consistency with numerical gradient (finite-difference).
    let eps = 1e-4_f32;
    let mut cross_validated = 0;
    for (idx, entry) in beta_state.entries.iter().enumerate() {
        let analytical = entry.grad();
        if analytical.abs() <= 1e-5 {
            continue; // Skip near-zero gradients where numerical noise dominates
        }

        // Compute numerical gradient by perturbing this beta
        let original = entry.value();

        let mut beta_plus = BetaState::from_history(&history).unwrap();
        for (i, e) in beta_plus.entries.iter_mut().enumerate() {
            e.set_value(beta_state.entries[i].value());
        }
        beta_plus.entries[idx].set_value(original + eps);
        let bounds_plus = verifier
            .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_plus)
            .unwrap();

        let mut beta_minus = BetaState::from_history(&history).unwrap();
        for (i, e) in beta_minus.entries.iter_mut().enumerate() {
            e.set_value(beta_state.entries[i].value());
        }
        beta_minus.entries[idx].set_value((original - eps).max(0.0));
        let bounds_minus = verifier
            .compute_bounds_with_constraints(&network, &input, &history, &layer_bounds, &beta_minus)
            .unwrap();

        let numerical = (bounds_plus.lower_scalar() - bounds_minus.lower_scalar()) / (2.0 * eps);

        // Check sign consistency
        if numerical.abs() > 1e-5 {
            assert!(
                analytical.signum() == numerical.signum(),
                "Gradient sign mismatch at entry {}: analytical={}, numerical={} \
                 (layer={}, neuron={})",
                idx,
                analytical,
                numerical,
                entry.layer_idx,
                entry.neuron_idx,
            );
            cross_validated += 1;
        }
    }
    // At least one gradient must be cross-validated. Without this assertion,
    // the entire numerical cross-validation section could silently validate nothing
    // if all gradients fell between the nonzero threshold (1e-10) and the
    // cross-validation threshold (1e-5).
    assert!(
        cross_validated > 0,
        "At least one gradient should be large enough for numerical cross-validation; \
         cross_validated={}, grads: {:?}",
        cross_validated,
        beta_state
            .entries
            .iter()
            .map(|e| e.grad())
            .collect::<Vec<_>>()
    );
}

/// Regression test for #2192: compute_joint_gradients with an unsupported layer
/// must succeed using IBP concretization (backward) and zero sensitivity (forward),
/// not silently skip the layer or return identity bounds.
#[ntest::timeout(10000)]
#[test]
fn test_joint_gradients_unsupported_layer_uses_ibp_concretization_2192() {
    // Network: Linear(2→2) → ReLU → AddConstant → Linear(2→1)
    // AddConstant is not handled in the gradient dispatch — it hits the catch-all.
    let w1 = arr2(&[[1.0, -0.5], [-0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let constant = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, -0.3]).unwrap();

    let w2 = arr2(&[[1.0, 0.5]]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::AddConstant(AddConstantLayer::new(constant)));
    network.add_layer(Layer::Linear(linear2));

    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    // Need a constraint on the ReLU layer to have non-empty beta state
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();
    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(1, 0, 0.1);
    let mut alpha_state = DomainAlphaState::empty();
    let mut cut_pool = CutPool::new(0);

    let verifier = BetaCrownVerifier::default();
    let result = verifier.compute_joint_gradients(
        &network,
        &input,
        &history,
        &layer_bounds,
        &mut beta_state,
        &mut alpha_state,
        &mut cut_pool,
        0,
    );

    // Must succeed — not error out, not panic
    assert!(
        result.is_ok(),
        "compute_joint_gradients with unsupported layer should succeed, got: {:?}",
        result.err()
    );

    // Beta gradients should be computed (not NaN)
    for entry in &beta_state.entries {
        assert!(
            !entry.grad.is_nan(),
            "Beta gradient should not be NaN after unsupported layer"
        );
    }
}

/// Build a 2→4→1 network with all alpha=0.5 and run joint gradient computation.
/// Returns (network, input, history, layer_bounds, beta_state, alpha_state, cut_pool, verifier).
#[allow(clippy::type_complexity)]
fn alpha_gradient_test_fixture() -> (
    Network,
    BoundedTensor,
    SplitHistory,
    Vec<Arc<BoundedTensor>>,
    BetaState,
    DomainAlphaState,
    CutPool,
    BetaCrownVerifier,
) {
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
    let w2 = arr2(&[[0.5, -0.3, 0.7, 0.1]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    let history = SplitHistory::new();
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let mut alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let mut beta_state = BetaState::from_history(&history).unwrap();
    let mut cut_pool = CutPool::new(0);
    for n in alpha_state.neurons.values_mut() {
        n.set_alpha(0.5);
    }

    let verifier = BetaCrownVerifier::default();
    verifier
        .compute_joint_gradients(
            &network,
            &input,
            &history,
            &layer_bounds,
            &mut beta_state,
            &mut alpha_state,
            &mut cut_pool,
            0,
        )
        .unwrap();
    (
        network,
        input,
        history,
        layer_bounds,
        beta_state,
        alpha_state,
        cut_pool,
        verifier,
    )
}

/// Regression test for #2252: alpha gradient dead conditional fix.
/// Verifies analytical alpha gradients are finite and at least one is non-zero.
#[ntest::timeout(10000)]
#[test]
fn test_alpha_gradient_includes_preactivation_factor_2252() {
    let (_, _, _, _, _, alpha_state, _, _) = alpha_gradient_test_fixture();
    assert!(
        alpha_state.neurons.values().any(|n| n.grad.abs() > 1e-10),
        "At least one unstable neuron should have non-zero alpha gradient"
    );
    for ((li, ni), n) in &alpha_state.neurons {
        assert!(
            !n.grad.is_nan(),
            "Alpha gradient NaN at layer {li}, neuron {ni}"
        );
    }
}

/// Regression test for #2252: numerical gradient consistency.
/// Verifies analytical alpha gradients match numerical gradients in magnitude,
/// not just sign. Uses the same relative+absolute tolerance as the beta gradient
/// test (test_analytical_vs_numerical_gradient_consistency).
#[ntest::timeout(10000)]
#[test]
fn test_alpha_gradient_numerical_sign_consistency_2252() {
    let (network, input, history, layer_bounds, beta_state, alpha_state, cut_pool, verifier) =
        alpha_gradient_test_fixture();

    let lb_at = |target: (usize, usize), val: f32| -> f32 {
        let mut s =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
        for n in s.neurons.values_mut() {
            n.set_alpha(0.5);
        }
        s.set_alpha(target.0, target.1, val);
        verifier
            .compute_bounds_with_alpha_beta(
                &network,
                &input,
                &history,
                &layer_bounds,
                &beta_state,
                &s,
                &cut_pool,
                None,
            )
            .unwrap()
            .flatten()
            .lower()[[0]]
    };

    let eps = 1e-4_f32;
    let mut checked = 0;
    for &key in alpha_state.neurons.keys().collect::<Vec<_>>() {
        let analytical = alpha_state.neurons[&key].grad;
        let numerical = (lb_at(key, 0.5 + eps) - lb_at(key, 0.5 - eps)) / (2.0 * eps);
        if analytical.abs() > 1e-6 || numerical.abs() > 1e-6 {
            // Magnitude check: same tolerance pattern as beta gradient test (line 182)
            let abs_err = (analytical - numerical).abs();
            let tol = 1e-2 + 1e-2 * numerical.abs();
            assert!(
                abs_err <= tol,
                "Alpha gradient magnitude mismatch at ({}, {}): analytical={}, numerical={}, \
                 abs_err={}, tol={}",
                key.0,
                key.1,
                analytical,
                numerical,
                abs_err,
                tol,
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 1,
        "Expected at least 1 neuron with non-trivial gradient, but all were near-zero"
    );
}

/// Build a 2-ReLU network and inject NaN into pre-ReLU(1) neuron 0's lower bound.
/// w2[0][0] is negative so backward coefficient at ReLU(1) neuron 0 is negative,
/// forcing the upper-slope path where unguarded u/(u-l) would produce NaN.
fn nan_pre_relu_fixture() -> (Network, BoundedTensor, Vec<Arc<BoundedTensor>>) {
    let w1 = arr2(&[[1.0, -0.5], [-0.5, 1.0]]);
    let w2 = arr2(&[[-1.0, 0.3], [0.7, 0.1]]);
    let w3 = arr2(&[[1.0, 0.5]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w3, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    let mut layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    // Inject NaN into pre-ReLU(1) neuron 0. new_unchecked bypasses NaN rejection.
    let real = layer_bounds[0].as_ref();
    let mut lower = real.lower().to_owned();
    lower[[0]] = f32::NAN;
    layer_bounds[0] =
        Arc::new(BoundedTensor::new_unchecked(lower, real.upper().to_owned()).unwrap());
    (network, input, layer_bounds)
}

/// Regression test for #2902: compute_joint_gradients NaN guard gap.
/// NaN pre-ReLU bounds must not produce NaN slopes via u/(u-l) division.
#[ntest::timeout(10000)]
#[test]
fn test_joint_gradients_nan_pre_relu_bounds_no_nan_propagation_2902() {
    let (network, input, layer_bounds) = nan_pre_relu_fixture();

    // Constrain ReLU(3) neuron 0 so beta gradient exercises the
    // forward sensitivity path through the NaN-guarded ReLU(1).
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 3,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(3, 0, 0.1);
    let mut alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let mut cut_pool = CutPool::new(0);

    let verifier = BetaCrownVerifier::default();
    verifier
        .compute_joint_gradients(
            &network,
            &input,
            &history,
            &layer_bounds,
            &mut beta_state,
            &mut alpha_state,
            &mut cut_pool,
            0,
        )
        .expect("compute_joint_gradients should not error on NaN bounds");

    for entry in &beta_state.entries {
        assert!(
            entry.grad.is_finite(),
            "Beta grad NaN: l={}, n={}",
            entry.layer_idx,
            entry.neuron_idx
        );
    }
    for (key, neuron) in &alpha_state.neurons {
        assert!(
            neuron.grad.is_finite(),
            "Alpha grad NaN: l={}, n={}",
            key.0,
            key.1
        );
    }
}

/// Build a 2-ReLU network with near-zero-width pre-ReLU bounds at neuron 0.
/// Without the RELU_RELAX_MIN_WIDTH guard at joint.rs, `u / (u - l)` would
/// produce Inf for this near-zero width, corrupting all downstream gradients.
fn near_zero_width_pre_relu_fixture() -> (Network, BoundedTensor, Vec<Arc<BoundedTensor>>) {
    let w1 = arr2(&[[1.0, -0.5], [-0.5, 1.0]]);
    let w2 = arr2(&[[-1.0, 0.3], [0.7, 0.1]]);
    let w3 = arr2(&[[1.0, 0.5]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w3, None).unwrap()));

    let input =
        BoundedTensor::new(arr1(&[-0.5, -0.5]).into_dyn(), arr1(&[0.5, 0.5]).into_dyn()).unwrap();
    let mut layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    // Inject near-zero-width crossing bounds at pre-ReLU(1) neuron 0.
    // u - l = 2e-20, far below RELU_RELAX_MIN_WIDTH = 1e-8.
    // Without the guard, u / (u - l) overflows to Inf in f32.
    let real = layer_bounds[0].as_ref();
    let mut lower = real.lower().to_owned();
    let mut upper = real.upper().to_owned();
    lower[[0]] = -1e-20_f32;
    upper[[0]] = 1e-20_f32;
    layer_bounds[0] = Arc::new(BoundedTensor::new(lower, upper).unwrap());
    (network, input, layer_bounds)
}

/// Regression test for #2697: near-zero-width pre-ReLU bounds must not produce
/// Inf slopes via unguarded u/(u-l) division in the gradient backward path.
#[ntest::timeout(10000)]
#[test]
fn test_joint_gradients_near_zero_width_no_inf_slopes_2697() {
    let (network, input, layer_bounds) = near_zero_width_pre_relu_fixture();

    // Constrain ReLU(3) neuron 0 so beta gradient exercises the
    // forward sensitivity path through the near-zero-width ReLU(1).
    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 3,
        neuron_idx: 0,
        is_active: true,
        score: 0.0,
    });
    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.set_beta(3, 0, 0.1);
    let mut alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let mut cut_pool = CutPool::new(0);

    let verifier = BetaCrownVerifier::default();
    verifier
        .compute_joint_gradients(
            &network,
            &input,
            &history,
            &layer_bounds,
            &mut beta_state,
            &mut alpha_state,
            &mut cut_pool,
            0,
        )
        .expect("compute_joint_gradients should not error on near-zero-width bounds");

    for entry in &beta_state.entries {
        assert!(
            entry.grad.is_finite(),
            "Beta grad should be finite with RELU_RELAX_MIN_WIDTH guard (#2697): \
             l={}, n={}, grad={}",
            entry.layer_idx,
            entry.neuron_idx,
            entry.grad
        );
    }
    for (key, neuron) in &alpha_state.neurons {
        assert!(
            neuron.grad.is_finite(),
            "Alpha grad should be finite with RELU_RELAX_MIN_WIDTH guard (#2697): \
             l={}, n={}, grad={}",
            key.0,
            key.1,
            neuron.grad
        );
    }
}
