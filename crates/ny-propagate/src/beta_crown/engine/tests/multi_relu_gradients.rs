// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Multi-ReLU alpha gradient tests.
//!
//! Tests for #2306: the forward sensitivity pass must use sign-dependent
//! branch selection matching the backward pass. Single-ReLU networks cannot
//! expose this bug because the forward sensitivity is consumed at the same
//! layer before the ReLU transform is applied.

use super::prelude::*;

/// Build a 2→4→4→1 network with two ReLU layers and weights designed to
/// produce negative backward coefficients at the first ReLU.
///
/// Architecture: Linear(2→4) → ReLU → Linear(4→4) → ReLU → Linear(4→1)
///
/// The second linear layer has negative entries to create negative backward
/// coefficients at the first ReLU, which triggers the sign-dependent branch.
fn multi_relu_fixture() -> (
    Network,
    BoundedTensor,
    SplitHistory,
    Vec<Arc<BoundedTensor>>,
    BetaState,
    DomainAlphaState,
    CutPool,
    BetaCrownVerifier,
) {
    // First linear: 2→4 with mixed signs
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    // Second linear: 4→4 with negative weights to create negative backward coeffs
    let w2 = arr2(&[
        [0.5, -0.6, 0.3, -0.4],
        [-0.7, 0.4, -0.2, 0.5],
        [0.3, -0.5, 0.8, -0.1],
        [-0.4, 0.3, -0.6, 0.7],
    ]);
    let linear2 = LinearLayer::new(w2, None).unwrap();

    // Final linear: 4→1
    let w3 = arr2(&[[0.6, -0.4, 0.3, -0.5]]);
    let linear3 = LinearLayer::new(w3, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1)); // layer 0: 2→4
    network.add_layer(Layer::ReLU(ReLULayer)); // layer 1: ReLU
    network.add_layer(Layer::Linear(linear2)); // layer 2: 4→4
    network.add_layer(Layer::ReLU(ReLULayer)); // layer 3: ReLU
    network.add_layer(Layer::Linear(linear3)); // layer 4: 4→1

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

/// Regression test for #2306: multi-ReLU alpha gradient numerical consistency.
///
/// The forward sensitivity pass must use sign-dependent branch selection
/// matching the backward pass. With a single-ReLU network, the forward
/// sensitivity is consumed before the ReLU transform, so the bug is hidden.
/// This test uses a 2-ReLU network where the first ReLU's forward sensitivity
/// transform must correctly select upper vs lower slopes based on backward
/// coefficient sign.
#[ntest::timeout(10000)]
#[test]
fn test_multi_relu_alpha_gradient_numerical_consistency_2306() {
    let (network, input, history, layer_bounds, beta_state, alpha_state, cut_pool, verifier) =
        multi_relu_fixture();

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
    let mut max_err = 0.0_f32;
    for &key in alpha_state.neurons.keys().collect::<Vec<_>>() {
        let analytical = alpha_state.neurons[&key].grad;
        let numerical = (lb_at(key, 0.5 + eps) - lb_at(key, 0.5 - eps)) / (2.0 * eps);
        if analytical.abs() > 1e-6 || numerical.abs() > 1e-6 {
            let abs_err = (analytical - numerical).abs();
            let tol = 1e-2 + 1e-2 * numerical.abs();
            assert!(
                abs_err <= tol,
                "Multi-ReLU alpha gradient mismatch at layer={}, neuron={}: \
                 analytical={}, numerical={}, abs_err={}, tol={} (sign-dependent branch bug #2306)",
                key.0,
                key.1,
                analytical,
                numerical,
                abs_err,
                tol,
            );
            max_err = max_err.max(abs_err);
            checked += 1;
        }
    }
    assert!(
        checked >= 2,
        "Expected at least 2 neurons with non-trivial gradient across 2 ReLU layers, got {}",
        checked,
    );
}
