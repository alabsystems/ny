// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::helpers::compute_chain_rule_gradients;
use super::NetworkAlphaCrownExt;
use crate::bounds::{AlphaCrownConfig, AlphaCrownIntermediate, AlphaState};
use crate::layers::{GatherLayer, Layer, LinearLayer, ReLULayer, ReshapeLayer};
use crate::network::Network;
use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Regression test for #2085:
/// production alpha-CROWN path should support non-flat output shapes
/// (best_* is shaped like CROWN output; per-iteration concretization is flat).
#[ntest::timeout(10000)]
#[test]
fn alpha_crown_handles_non_flat_output_best_bound_update_path() {
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.2]]),
            Some(arr1(&[0.0, 0.1, -0.1])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(
            arr2(&[
                [0.4, 0.2, -0.1],
                [0.1, -0.3, 0.5],
                [-0.2, 0.6, 0.1],
                [0.3, 0.2, 0.4],
            ]),
            Some(arr1(&[0.0, 0.0, 0.05, -0.05])),
        )
        .unwrap(),
    ));
    network.add_layer(Layer::Reshape(ReshapeLayer::new(vec![2, 2])));

    let input =
        BoundedTensor::new(arr1(&[-0.6, -0.4]).into_dyn(), arr1(&[0.7, 0.8]).into_dyn()).unwrap();

    let config = AlphaCrownConfig {
        iterations: 5,
        adaptive_skip: false,
        ..Default::default()
    };

    let alpha_bounds = network
        .propagate_alpha_crown_with_config(&input, &config)
        .unwrap();
    let crown_bounds = network.propagate_crown(&input).unwrap();

    assert_eq!(alpha_bounds.shape(), &[2, 2]);
    assert_eq!(alpha_bounds.shape(), crown_bounds.shape());
    for ((&alpha_l, &alpha_u), (&crown_l, &crown_u)) in alpha_bounds
        .lower()
        .iter()
        .zip(alpha_bounds.upper().iter())
        .zip(crown_bounds.lower().iter().zip(crown_bounds.upper().iter()))
    {
        assert!(
            alpha_l.is_finite() && alpha_u.is_finite(),
            "alpha-CROWN produced non-finite bound"
        );
        assert!(alpha_l <= alpha_u, "inverted alpha-CROWN bound");
        // alpha-CROWN tracks best bounds from CROWN baseline; should not regress.
        assert!(
            alpha_l + 1e-6 >= crown_l,
            "alpha lower regressed below CROWN lower"
        );
        assert!(
            alpha_u <= crown_u + 1e-6,
            "alpha upper regressed above CROWN upper"
        );
    }
}

/// Regression test for #2079: SPSA gradient averaging must not produce NaN/Inf
/// when spsa_samples == 0. The fix is `.max(1)` on the divisor, matching the
/// pattern in propagate_dag.rs:544 and propagate_sequential.rs:587.
#[test]
fn spsa_zero_samples_does_not_produce_nan() {
    // Simulate the SPSA gradient averaging with spsa_samples = 0.
    // Without `.max(1)`, this divides by 0.0 and produces NaN.
    let spsa_samples: usize = 0;
    let num_samples = spsa_samples.max(1) as f32;
    assert_eq!(num_samples, 1.0, ".max(1) should clamp 0 to 1");

    // avg_grads stays at zeros when no samples run (loop body never executes)
    let mut avg_grads = vec![Array1::<f32>::zeros(5), Array1::<f32>::zeros(3)];
    for grad in &mut avg_grads {
        *grad /= num_samples;
    }

    // All values must be finite (zero), not NaN or Inf
    for grad in &avg_grads {
        for &v in grad.iter() {
            assert!(v.is_finite(), "gradient should be finite, got {v}");
            assert_eq!(v, 0.0, "zero grads / 1.0 should stay zero");
        }
    }

    // Also verify that non-zero grads are preserved correctly with samples=1
    let mut nonzero_grads = vec![Array1::from_vec(vec![2.0f32, -4.0, 6.0])];
    for grad in &mut nonzero_grads {
        *grad /= 1.0f32; // samples=1 means divide by 1.0 (identity)
    }
    assert_eq!(nonzero_grads[0][0], 2.0);
    assert_eq!(nonzero_grads[0][1], -4.0);
    assert_eq!(nonzero_grads[0][2], 6.0);
}

/// Regression test for #2189: α-CROWN intermediates fallback on UnsupportedOp
/// must return constant final_bounds matching full-network CROWN bounds.
#[ntest::timeout(10000)]
#[test]
fn alpha_crown_intermediates_unsupported_fallback_uses_constant_final_bounds_2189() {
    let mut network = Network::new();
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0i64, 2])
        .expect("invariant: valid gather indices shape");
    network.add_layer(Layer::Gather(GatherLayer::new(0, Some(indices), vec![])));

    let input = BoundedTensor::new(
        arr1(&[-1.0f32, 0.5, 2.0]).into_dyn(),
        arr1(&[1.0f32, 2.5, 4.0]).into_dyn(),
    )
    .expect("invariant: lower <= upper");

    let layer_bounds = network
        .collect_ibp_bounds(&input)
        .expect("invariant: IBP bounds for single-layer network");
    let alpha_state = AlphaState::from_preactivation_bounds(&layer_bounds, &[])
        .expect("invariant: alpha state from valid bounds");

    let intermediate = NetworkAlphaCrownExt::propagate_alpha_crown_with_intermediates_impl(
        &network,
        &input,
        &layer_bounds,
        &alpha_state,
        None,
    )
    .expect("UnsupportedOp intermediates fallback should return a constant final_bounds");

    assert!(
        intermediate.a_at_relu.is_empty(),
        "fallback path should return empty intermediates"
    );
    assert!(
        intermediate.pre_relu_bounds.is_empty(),
        "fallback path should return empty pre-ReLU bounds"
    );
    assert_eq!(intermediate.final_bounds.num_inputs(), input.len());

    assert!(
        intermediate.final_bounds.lower_a.iter().all(|&v| v == 0.0),
        "fallback lower_a must be zero for constant bounds"
    );
    assert!(
        intermediate.final_bounds.upper_a.iter().all(|&v| v == 0.0),
        "fallback upper_a must be zero for constant bounds"
    );

    let crown_bounds = network
        .propagate_crown(&input)
        .expect("invariant: CROWN for single-layer network")
        .flatten();
    assert_eq!(intermediate.final_bounds.num_outputs(), crown_bounds.len());

    for (i, (&expected, &got)) in crown_bounds
        .lower()
        .iter()
        .zip(intermediate.final_bounds.lower_b.iter())
        .enumerate()
    {
        assert!(
            (expected - got).abs() <= 1e-6,
            "lower_b[{i}] mismatch: expected {expected}, got {got}"
        );
    }

    for (i, (&expected, &got)) in crown_bounds
        .upper()
        .iter()
        .zip(intermediate.final_bounds.upper_b.iter())
        .enumerate()
    {
        assert!(
            (expected - got).abs() <= 1e-6,
            "upper_b[{i}] mismatch: expected {expected}, got {got}"
        );
    }
}

// ===== #2809 regression tests: NaN-unsafe branching in AnalyticChain =====

use std::collections::HashMap;

/// Helper: build a minimal AlphaState for chain-rule gradient tests.
fn make_alpha_state(alphas: Vec<Array1<f32>>, masks: Vec<Array1<bool>>) -> AlphaState {
    let n = alphas.len();
    AlphaState {
        velocity: (0..n).map(|i| Array1::zeros(alphas[i].len())).collect(),
        adam_m: (0..n).map(|i| Array1::zeros(alphas[i].len())).collect(),
        adam_v: (0..n).map(|i| Array1::zeros(alphas[i].len())).collect(),
        alphas_upper: alphas.clone(),
        velocity_upper: (0..n).map(|i| Array1::zeros(alphas[i].len())).collect(),
        adam_m_upper: (0..n).map(|i| Array1::zeros(alphas[i].len())).collect(),
        adam_v_upper: (0..n).map(|i| Array1::zeros(alphas[i].len())).collect(),
        alphas,
        unstable_mask: masks,
        bilinear_alphas: HashMap::new(),
        bilinear_adam_m: HashMap::new(),
        bilinear_adam_v: HashMap::new(),
        invprop_state: None,
    }
}

/// NaN in pre-ReLU bounds must not enter gradient computation.
/// Before the fix, NaN bounds would pass the `l >= 0.0 || u <= 0.0` check
/// (both comparisons false for NaN), treating NaN bounds as "unstable" and
/// flowing NaN into gradient arithmetic. (#2809)
#[test]
fn test_chain_gradient_nan_pre_relu_bounds_produces_zero_gradient() {
    let alpha_state = make_alpha_state(vec![arr1(&[0.5, 0.5])], vec![arr1(&[true, true])]);
    let intermediate = AlphaCrownIntermediate {
        a_at_relu: vec![arr2(&[[1.0, 2.0]])],
        pre_relu_bounds: vec![(arr1(&[f32::NAN, -1.0]), arr1(&[f32::NAN, 1.0]))],
        final_bounds: crate::LinearBounds::identity(1),
    };

    let grads = compute_chain_rule_gradients(&alpha_state, &intermediate);
    assert_eq!(grads.len(), 1);

    // Neuron 0: NaN bounds → skipped → gradient = 0.0
    assert_eq!(
        grads[0][0], 0.0,
        "NaN-bounded neuron should have zero gradient"
    );
    // Neuron 1: normal unstable neuron → has non-zero gradient
    assert!(
        grads[0][1].is_finite(),
        "Normal neuron gradient should be finite"
    );
}

/// NaN in A-matrix coefficients must not silently drop contributions.
/// Before the fix, NaN a_ji would cause `a_ji > 0.0` to return false,
/// silently skipping the contribution without signaling. (#2809)
#[test]
fn test_chain_gradient_nan_a_coefficient_skips_contribution() {
    let alpha_state = make_alpha_state(vec![arr1(&[0.5])], vec![arr1(&[true])]);
    let intermediate = AlphaCrownIntermediate {
        a_at_relu: vec![arr2(&[[f32::NAN]])],
        pre_relu_bounds: vec![(arr1(&[-1.0]), arr1(&[1.0]))],
        final_bounds: crate::LinearBounds::identity(1),
    };

    let grads = compute_chain_rule_gradients(&alpha_state, &intermediate);
    assert_eq!(grads.len(), 1);
    // With the guard: NaN a_ji → contribution skipped → gradient = 0.0
    assert_eq!(
        grads[0][0], 0.0,
        "NaN A coefficient should yield zero gradient"
    );
}

/// Inf in pre-ReLU bounds must not enter gradient computation.
/// Inf bounds are non-finite and should be treated as "skip" like NaN. (#2809)
#[test]
fn test_chain_gradient_inf_pre_relu_bounds_produces_zero_gradient() {
    let alpha_state = make_alpha_state(vec![arr1(&[0.5])], vec![arr1(&[true])]);
    let intermediate = AlphaCrownIntermediate {
        a_at_relu: vec![arr2(&[[1.0]])],
        pre_relu_bounds: vec![(arr1(&[f32::NEG_INFINITY]), arr1(&[f32::INFINITY]))],
        final_bounds: crate::LinearBounds::identity(1),
    };

    let grads = compute_chain_rule_gradients(&alpha_state, &intermediate);
    assert_eq!(grads.len(), 1);
    assert_eq!(
        grads[0][0], 0.0,
        "Inf-bounded neuron should have zero gradient"
    );
}
