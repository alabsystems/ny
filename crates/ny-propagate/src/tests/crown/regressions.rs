// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for CROWN bound propagation.

use super::*;
use ndarray::{arr1, arr2, Array1, Array2};

// ============================================================
// NEGATIVE AFFINE SCALE SOUNDNESS REGRESSION TESTS (#306, #307)
// ============================================================
//
// These tests verify that CROWN backward propagation correctly handles
// negative affine scales in MulConstant and BatchNorm layers.
//
// Background: CROWN backward composes by substitution, not IBP.
// For y = c*x where c < 0, the correct transformation is:
//   A_new = A * c  (just scale coefficients)
// NOT swap lower/upper bounds like IBP does.
//
// The bug (#306) incorrectly swapped bounds for negative scale, which
// could collapse upper bounds to 0 for networks like ReLU(-x).
//
// Reference: designs/2026-01-29-crown-affine-negative-scale.md

/// Regression test: ReLU with lower=-inf now produces a SOUND infinite-domain
/// relaxation instead of being rejected.
///
/// The ReLU CROWN backward path uses a NaN-only domain guard, so infinite
/// pre-activation bounds run the proven infinite-case branches of
/// `relu_linear_relaxation` (l=-inf, finite u>0 → lower y=0, upper y=u). This
/// recovers a tight sound bound that the previous non_finite_domain_guard discarded.
#[ntest::timeout(10000)]
#[test]
fn test_relu_crown_soundness_with_infinite_lower_bound() {
    let relu = ReLULayer::new();
    let pre_activation = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[2.0_f32]).into_dyn(),
    )
    .unwrap();
    let incoming = LinearBounds::identity(1);

    let result = relu
        .propagate_linear_with_bounds(&incoming, &pre_activation)
        .expect("infinite pre-activation should be accepted (NaN-only guard)");

    let ls = result.lower_a[[0, 0]];
    let li = result.lower_b[0];
    let us = result.upper_a[[0, 0]];
    let ui = result.upper_b[0];
    // Soundness over a finite probe grid of (-inf, 2].
    for k in 0..=400 {
        let x = -1.0e6 + (2.0 - (-1.0e6)) * (k as f64 / 400.0);
        let fx = x.max(0.0);
        let lower = ls as f64 * x + li as f64;
        let upper = us as f64 * x + ui as f64;
        assert!(lower <= fx + 1e-3, "lower {lower} > relu({x})={fx}");
        assert!(upper + 1e-3 >= fx, "upper {upper} < relu({x})={fx}");
    }
}

/// Regression test: canonical CROWN path (via `propagate_crown_backward`) with infinite
/// pre-activation bounds now produces SOUND infinite-domain relaxations.
///
/// The ReLU CROWN backward path uses a NaN-only guard, so all three infinite-bound
/// cases run the proven infinite-case branches of `relu_linear_relaxation` and yield
/// sound bounds instead of being rejected.
///
/// Originally Part of #1787 (W2]587 fix).
#[ntest::timeout(10000)]
#[test]
fn test_relu_crown_canonical_infinite_bounds() {
    use crate::layers::common::BoundPropagation;

    let relu_layer = Layer::ReLU(ReLULayer::new());

    // (l, u, finite probe lo, finite probe hi, label)
    let cases: &[(f32, f32, f64, f64, &str)] = &[
        (f32::NEG_INFINITY, 2.0, -1.0e6, 2.0, "l=-inf,u=2"),
        (-5.0, f32::INFINITY, -5.0, 1.0e6, "l=-5,u=+inf"),
        (f32::NEG_INFINITY, f32::INFINITY, -1.0e6, 1.0e6, "both inf"),
    ];

    for &(l, u, lo, hi, label) in cases {
        let pre_act =
            BoundedTensor::new_unchecked(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let incoming = LinearBounds::identity(1);
        let result = relu_layer
            .propagate_crown_backward(&incoming, Some(&pre_act))
            .unwrap_or_else(|e| panic!("{label}: infinite pre-activation should be accepted: {e}"));

        let ls = result.lower_a[[0, 0]];
        let li = result.lower_b[0];
        let us = result.upper_a[[0, 0]];
        let ui = result.upper_b[0];
        assert!(
            !ls.is_nan() && !li.is_nan() && !us.is_nan() && !ui.is_nan(),
            "{label}: NaN in CROWN result"
        );
        for k in 0..=400 {
            let x = lo + (hi - lo) * (k as f64 / 400.0);
            let fx = x.max(0.0);
            if li.is_finite() {
                let lower = ls as f64 * x + li as f64;
                assert!(lower <= fx + 1e-3, "{label} lower {lower} > relu({x})={fx}");
            }
            if ui.is_finite() {
                let upper = us as f64 * x + ui as f64;
                assert!(upper + 1e-3 >= fx, "{label} upper {upper} < relu({x})={fx}");
            }
        }
    }
}

/// Regression test: LeakyReLU with negative alpha and l=-inf now produces a SOUND
/// infinite-domain relaxation instead of being rejected.
///
/// For alpha < 0 and x < 0, f(x)=alpha*x grows to +inf as x->-inf, so a constant
/// upper bound y<=u would be UNSOUND. The proven l=-inf branch of
/// `leaky_relu_linear_relaxation` for alpha<0 instead uses the sloped upper plane
/// y = alpha*x + (1-alpha)*u (and lower y = x), which is sound over the entire
/// unbounded domain. The NaN-only guard lets this branch run.
#[ntest::timeout(10000)]
#[test]
fn test_leaky_relu_crown_negative_alpha_infinite_lower_bound() {
    let alpha = -0.5_f32;
    let layer = LeakyReLULayer::new(alpha);
    let pre_activation = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[2.0_f32]).into_dyn(),
    )
    .unwrap();
    let incoming = LinearBounds::identity(1);

    let result = layer
        .propagate_linear_with_bounds(&incoming, &pre_activation)
        .expect("infinite pre-activation should be accepted (NaN-only guard)");

    let ls = result.lower_a[[0, 0]];
    let li = result.lower_b[0];
    let us = result.upper_a[[0, 0]];
    let ui = result.upper_b[0];
    // Soundness over a finite probe grid of (-inf, 2], crucially out to large -x
    // where the sloped upper plane (not a constant) is what keeps the bound sound.
    for k in 0..=400 {
        let x = -1.0e6 + (2.0 - (-1.0e6)) * (k as f64 / 400.0);
        let fx = if x >= 0.0 { x } else { alpha as f64 * x };
        let lower = ls as f64 * x + li as f64;
        let upper = us as f64 * x + ui as f64;
        let tol = 1e-3 * fx.abs().max(1.0);
        assert!(lower <= fx + tol, "lower {lower} > leaky({x})={fx}");
        assert!(upper + tol >= fx, "upper {upper} < leaky({x})={fx}");
    }
}

/// Regression test: PReLU unbatched CROWN with negative slope and l=-inf.
///
/// Updated for #2977: CROWN backward domain_guard rejects non-finite pre-activation.
#[ntest::timeout(10000)]
#[test]
fn test_prelu_crown_negative_alpha_infinite_lower_bound() {
    let alpha = -0.5_f32;
    let layer = PReluLayer::from_scalar(alpha);
    let pre_activation = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[2.0_f32]).into_dyn(),
    )
    .unwrap();
    let incoming = LinearBounds::identity(1);

    let result = layer.propagate_linear_with_bounds(&incoming, &pre_activation);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "PReLU with l=-inf should trigger domain_guard: got {:?}",
        result
    );
}

/// Regression test: PReLU batched CROWN with negative slope and l=-inf.
///
/// Updated for #2977: CROWN backward domain_guard rejects non-finite pre-activation.
#[ntest::timeout(10000)]
#[test]
fn test_prelu_batched_crown_negative_alpha_infinite_lower_bound() {
    let alpha = -0.5_f32;
    let layer = PReluLayer::from_scalar(alpha);
    let pre_activation = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[2.0_f32]).into_dyn(),
    )
    .unwrap();
    let incoming = BatchedLinearBounds::identity(&[1]).unwrap();

    let result = layer.propagate_linear_batched_with_bounds(&incoming, &pre_activation);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "Batched PReLU with l=-inf should trigger domain_guard: got {:?}",
        result
    );
}

/// Regression test for #306: MulConstant(-1) + ReLU should have correct upper bound.
///
/// Network: x -> MulConstant(-1) -> ReLU -> output
/// Input: x in [-1, 1]
/// Expected: output = ReLU(-x), so output in [0, 1] (max at x = -1 gives ReLU(1) = 1)
///
/// Bug behavior: If lower/upper are incorrectly swapped for c < 0, the CROWN
/// backward produces upper = 0, which is unsound (true max is 1).
#[ntest::timeout(10000)]
#[test]
fn test_negative_scale_crown_soundness_mulconstant_relu() {
    use crate::layers::MulConstantLayer;

    // Build network: MulConstant(-1) -> ReLU
    let mul_neg = MulConstantLayer::scalar(-1.0);
    let relu = ReLULayer;

    let mut network = Network::new();
    network.add_layer(Layer::MulConstant(mul_neg));
    network.add_layer(Layer::ReLU(relu));

    // Input: x in [-1, 1]
    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // CROWN bounds
    let crown_output = network.propagate_crown(&input).unwrap();

    // Verify bounds are valid
    assert!(
        crown_output.lower()[[0]] <= crown_output.upper()[[0]],
        "CROWN bounds must be valid: lower {} > upper {}",
        crown_output.lower()[[0]],
        crown_output.upper()[[0]]
    );

    // Key soundness check: upper bound should be at least 1
    // (since ReLU(-(-1)) = ReLU(1) = 1)
    assert!(
        crown_output.upper()[[0]] >= 1.0 - 1e-5,
        "CROWN upper bound {} should be >= 1.0 for ReLU(-x) with x in [-1, 1]. \
         Bug #306 would cause upper to collapse to 0.",
        crown_output.upper()[[0]]
    );

    // Lower bound should be 0 (ReLU output is always non-negative)
    assert!(
        crown_output.lower()[[0]] >= -1e-5,
        "CROWN lower bound {} should be >= 0 for ReLU output",
        crown_output.lower()[[0]]
    );

    // Verify soundness against concrete evaluations
    let test_inputs = [-1.0_f32, -0.5, 0.0, 0.5, 1.0];
    for x in test_inputs {
        let concrete_in = BoundedTensor::concrete(arr1(&[x]).into_dyn()).unwrap();
        let concrete_out = network.propagate_ibp(&concrete_in).unwrap();
        let y = concrete_out.lower()[[0]]; // concrete, so lower == upper

        assert!(
            y >= crown_output.lower()[[0]] - 1e-5,
            "Soundness: concrete f({}) = {} < CROWN lower {}",
            x,
            y,
            crown_output.lower()[[0]]
        );
        assert!(
            y <= crown_output.upper()[[0]] + 1e-5,
            "Soundness: concrete f({}) = {} > CROWN upper {}",
            x,
            y,
            crown_output.upper()[[0]]
        );
    }

    println!(
        "MulConstant(-1) + ReLU CROWN bounds: [{}, {}]",
        crown_output.lower()[[0]],
        crown_output.upper()[[0]]
    );
    println!("Regression test for #306 PASSED: upper bound is correctly >= 1");
}

/// Regression test for #306: BatchNorm(ny < 0) + ReLU should have correct upper bound.
///
/// Network: x -> BatchNorm(scale=-1, bias=0) -> ReLU -> Linear(sum) -> output
/// Input: x in [-1, 1]^n (symmetric interval)
/// Expected: BatchNorm acts like MulConstant(-1), so same reasoning applies.
///
/// This test verifies BatchNorm CROWN backward handles negative scale correctly.
#[ntest::timeout(10000)]
#[test]
fn test_negative_scale_crown_soundness_batchnorm_relu() {
    use crate::layers::{BatchNormLayer, ReshapeLayer};
    use crate::network::{GraphNetwork, GraphNode};

    // Build graph: Reshape -> BatchNorm(scale<0) -> ReLU -> Linear
    let mut graph = GraphNetwork::new();

    // Reshape to (C=2,) for BatchNorm
    let reshape1 = ReshapeLayer::new(vec![2]);
    graph.add_node(GraphNode::from_input("reshape1", Layer::Reshape(reshape1)));

    // BatchNorm with negative scale: scale = [-1.0, -0.5], bias = [0.0, 0.0]
    let scale = arr1(&[-1.0_f32, -0.5]).into_dyn();
    let bias = arr1(&[0.0_f32, 0.0]).into_dyn();
    let bn = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    graph.add_node(GraphNode::new(
        "bn",
        Layer::BatchNorm(bn),
        vec!["reshape1".to_string()],
    ));

    // ReLU
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["bn".to_string()],
    ));

    // Final linear: sum outputs
    let w = Array2::ones((1, 2));
    let linear = LinearLayer::new(w, None).unwrap();
    graph.add_node(GraphNode::new(
        "linear",
        Layer::Linear(linear),
        vec!["relu".to_string()],
    ));

    graph.set_output("linear");

    // Input: x in [-1, 1]^2 (symmetric)
    let lower = arr1(&[-1.0_f32, -1.0]).into_dyn();
    let upper = arr1(&[1.0_f32, 1.0]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // CROWN bounds
    let crown_output = graph.propagate_crown(&input).unwrap();

    // Verify bounds are valid
    assert!(
        crown_output.lower()[[0]] <= crown_output.upper()[[0]],
        "CROWN bounds must be valid: lower {} > upper {}",
        crown_output.lower()[[0]],
        crown_output.upper()[[0]]
    );

    // Key soundness check: upper bound should be > 0
    // Channel 0: BatchNorm(-1 * x) -> ReLU -> max = 1 (at x = -1)
    // Channel 1: BatchNorm(-0.5 * x) -> ReLU -> max = 0.5 (at x = -1)
    // Sum -> max = 1.5
    assert!(
        crown_output.upper()[[0]] >= 1.5 - 1e-4,
        "CROWN upper bound {} should be >= 1.5 for BatchNorm(ny<0) + ReLU. \
         Bug #306 would cause upper to collapse toward 0.",
        crown_output.upper()[[0]]
    );

    // Verify soundness against corner inputs
    let test_inputs = vec![
        arr1(&[-1.0_f32, -1.0]).into_dyn(), // Expected max: ReLU(1) + ReLU(0.5) = 1.5
        arr1(&[1.0_f32, 1.0]).into_dyn(),   // Expected: ReLU(-1) + ReLU(-0.5) = 0
        arr1(&[0.0_f32, 0.0]).into_dyn(),   // Expected: ReLU(0) + ReLU(0) = 0
        arr1(&[-1.0_f32, 1.0]).into_dyn(),  // Expected: ReLU(1) + ReLU(-0.5) = 1
        arr1(&[1.0_f32, -1.0]).into_dyn(),  // Expected: ReLU(-1) + ReLU(0.5) = 0.5
    ];

    for test_input in &test_inputs {
        let concrete = BoundedTensor::concrete(test_input.clone()).unwrap();
        let concrete_output = graph.propagate_ibp(&concrete).unwrap();

        assert!(
            concrete_output.lower()[[0]] >= crown_output.lower()[[0]] - 1e-4,
            "Soundness: concrete {} < CROWN lower {} for input {:?}",
            concrete_output.lower()[[0]],
            crown_output.lower()[[0]],
            test_input
        );
        assert!(
            concrete_output.upper()[[0]] <= crown_output.upper()[[0]] + 1e-4,
            "Soundness: concrete {} > CROWN upper {} for input {:?}",
            concrete_output.upper()[[0]],
            crown_output.upper()[[0]],
            test_input
        );
    }

    println!(
        "BatchNorm(ny<0) + ReLU CROWN bounds: [{}, {}]",
        crown_output.lower()[[0]],
        crown_output.upper()[[0]]
    );
    println!("Regression test for #306 PASSED: negative scale handled correctly");
}

/// Regression test for #309: SubConstant reverse mode (c - x) + ReLU should have correct bounds.
///
/// Network: x -> SubConstant(c=1, reverse=true) -> ReLU -> output
/// Input: x in [-1, 1]
/// Expected: output = ReLU(1 - x), so output in [0, 2] (max at x = -1 gives ReLU(2) = 2)
///
/// This test verifies SubConstant reverse CROWN backward handles the coefficient
/// negation and bias adjustment correctly.
#[ntest::timeout(10000)]
#[test]
fn test_subconstant_reverse_crown_soundness_relu() {
    use crate::layers::SubConstantLayer;

    // Build network: SubConstant(c=1, reverse=true) -> ReLU
    // y = 1 - x, then ReLU
    let sub_reverse = SubConstantLayer::new_reverse(arr1(&[1.0_f32]).into_dyn());
    let relu = ReLULayer;

    let mut network = Network::new();
    network.add_layer(Layer::SubConstant(sub_reverse));
    network.add_layer(Layer::ReLU(relu));

    // Input: x in [-1, 1]
    let input =
        BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn()).unwrap();

    // CROWN bounds
    let crown_output = network.propagate_crown(&input).unwrap();

    // Verify bounds are valid
    assert!(
        crown_output.lower()[[0]] <= crown_output.upper()[[0]],
        "CROWN bounds must be valid: lower {} > upper {}",
        crown_output.lower()[[0]],
        crown_output.upper()[[0]]
    );

    // Key soundness check: upper bound should be at least 2
    // (since ReLU(1 - (-1)) = ReLU(2) = 2)
    assert!(
        crown_output.upper()[[0]] >= 2.0 - 1e-4,
        "CROWN upper bound {} should be >= 2.0 for ReLU(1-x) with x in [-1, 1]. \
         Bug #309 would cause incorrect bounds due to coefficient/bias issues.",
        crown_output.upper()[[0]]
    );

    // Lower bound should be 0 (ReLU output is always non-negative)
    // Note: at x=1, ReLU(1-1)=ReLU(0)=0
    assert!(
        crown_output.lower()[[0]] >= -1e-5,
        "CROWN lower bound {} should be >= 0 for ReLU output",
        crown_output.lower()[[0]]
    );

    // Verify soundness against concrete evaluations
    let test_inputs = [-1.0_f32, -0.5, 0.0, 0.5, 1.0];
    for x in test_inputs {
        let concrete_in = BoundedTensor::concrete(arr1(&[x]).into_dyn()).unwrap();
        let concrete_out = network.propagate_ibp(&concrete_in).unwrap();
        let y = concrete_out.lower()[[0]]; // concrete, so lower == upper

        assert!(
            y >= crown_output.lower()[[0]] - 1e-4,
            "Soundness: concrete f({}) = {} < CROWN lower {}",
            x,
            y,
            crown_output.lower()[[0]]
        );
        assert!(
            y <= crown_output.upper()[[0]] + 1e-4,
            "Soundness: concrete f({}) = {} > CROWN upper {}",
            x,
            y,
            crown_output.upper()[[0]]
        );
    }

    println!(
        "SubConstant reverse (1-x) + ReLU CROWN bounds: [{}, {}]",
        crown_output.lower()[[0]],
        crown_output.upper()[[0]]
    );
    println!("Regression test for #309 PASSED: SubConstant reverse handled correctly");
}

/// Regression test for #1932: CROWN coefficient overflow in deep networks.
///
/// IBP clamps intermediate bounds to `f32::MAX / 2` after each layer (linear/mod.rs:289-293),
/// but CROWN backward coefficients (`lower_a`, `upper_a`) accumulate across layers with
/// no magnitude clamping. In a deep network with weight spectral norms > 1, CROWN coefficients
/// grow exponentially: `A_n = A_0 @ W_1 @ W_2 @ ... @ W_n`, potentially overflowing to
/// `f32::INFINITY` or producing `NaN` via `inf - inf`.
///
/// This test builds a 25-layer network with 4x4 weight matrices whose spectral norm ≈ 2,
/// which should amplify coefficients by ~2^25 ≈ 33 million. Since f32 range is ~3.4e38,
/// 25 layers with norm-2 weights stays within f32 range but is close enough that any
/// additional accumulation (bias terms, ReLU relaxation slopes) could push it over.
///
/// We verify:
/// 1. CROWN does not produce NaN or Infinity in the final bounds
/// 2. CROWN bounds are sound (contain all concrete evaluations at corners)
/// 3. IBP bounds are finite (IBP has per-layer clamping, so this should always pass)
///
/// If this test fails with NaN/Inf in CROWN but not IBP, it confirms the asymmetry
/// documented in #1932.
#[ntest::timeout(60000)]
#[test]
fn test_crown_coefficient_overflow_deep_network_1932() {
    let n = 4; // neuron width
    let depth = 25; // number of Linear+ReLU layers

    let mut network = Network::new();

    // Build deep network: (Linear -> ReLU) x depth -> Linear(sum)
    // Weight matrix: scaled identity + small perturbation to avoid exact cancellation.
    // Spectral norm ≈ 2.0 (dominant eigenvalue of 2*I + perturbation).
    for layer_idx in 0..depth {
        let mut w = Array2::<f32>::eye(n) * 2.0_f32;
        // Add small asymmetric perturbation so coefficients don't stay perfectly diagonal
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w[[i, j]] = 0.01 * ((i + j + layer_idx) as f32 % 3.0 - 1.0);
                }
            }
        }
        let bias = Array1::<f32>::zeros(n);
        let linear = LinearLayer::new(w, Some(bias)).unwrap();
        network.add_layer(Layer::Linear(linear));
        network.add_layer(Layer::ReLU(ReLULayer));
    }

    // Final summing layer: 1 output = sum of all neurons
    let sum_w = Array2::<f32>::ones((1, n));
    let sum_layer = LinearLayer::new(sum_w, None).unwrap();
    network.add_layer(Layer::Linear(sum_layer));

    // Input: x in [0.1, 0.2]^n (all positive to keep ReLU in linear regime,
    // which maximizes coefficient growth since ReLU slope = 1 everywhere)
    let lower = Array1::from_elem(n, 0.1_f32).into_dyn();
    let upper = Array1::from_elem(n, 0.2_f32).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // IBP should always work (has per-layer clamping)
    let ibp_result = network.propagate_ibp(&input);
    assert!(
        ibp_result.is_ok(),
        "IBP should not fail on deep network: {:?}",
        ibp_result.err()
    );
    let ibp_bounds = ibp_result.unwrap();
    let ibp_lower = ibp_bounds.lower()[[0]];
    let ibp_upper = ibp_bounds.upper()[[0]];
    assert!(
        ibp_lower.is_finite() && ibp_upper.is_finite(),
        "IBP bounds should be finite: [{}, {}]",
        ibp_lower,
        ibp_upper
    );

    // CROWN: this is the path that lacks overflow clamping (#1932)
    let crown_result = network.propagate_crown(&input);

    // The test documents the current behavior. If CROWN coefficients overflow,
    // propagate_crown may return Ok with NaN/Inf bounds or may return Err.
    match crown_result {
        Ok(crown_bounds) => {
            let crown_lower = crown_bounds.lower()[[0]];
            let crown_upper = crown_bounds.upper()[[0]];

            // Check for NaN (the most dangerous case — silent wrong answers)
            assert!(
                !crown_lower.is_nan() && !crown_upper.is_nan(),
                "CROWN bounds contain NaN (coefficient overflow): lower={}, upper={} \
                 (IBP: [{}, {}]). This confirms #1932: CROWN backward chain needs \
                 overflow clamping like IBP has.",
                crown_lower,
                crown_upper,
                ibp_lower,
                ibp_upper,
            );

            // Check for Infinity
            assert!(
                crown_lower.is_finite() && crown_upper.is_finite(),
                "CROWN bounds overflow to Infinity: lower={}, upper={} \
                 (IBP: [{}, {}]). This confirms #1932: CROWN backward chain needs \
                 overflow clamping.",
                crown_lower,
                crown_upper,
                ibp_lower,
                ibp_upper,
            );

            // If bounds are finite, verify soundness at concrete points
            // (all inputs in [0.1, 0.2], so enumerate corners)
            assert!(
                crown_lower <= crown_upper + 1e-4,
                "CROWN bounds inverted: lower {} > upper {}",
                crown_lower,
                crown_upper,
            );

            // Check soundness at corner points.
            //
            // Tolerance: CROWN coefficients grow as ~W^depth through backward chain
            // multiplication. With weight spectral norm ≈ 2 and depth 25, coefficients
            // reach ~2^25 ≈ 3.3e7. At f32 precision (epsilon ≈ 1.2e-7), absolute
            // rounding error in a single multiplication is ~4. Accumulated over 25
            // layers, the error compounds to ~O(depth * max_coeff * f32_eps).
            // Use relative tolerance of 1e-5 as conservative bound for deep f32 chains.
            let tol = crown_upper.abs().max(crown_lower.abs()).max(1.0) * 1e-5;
            let corners: Vec<f32> = vec![0.1, 0.2];
            for &x0 in &corners {
                for &x1 in &corners {
                    for &x2 in &corners {
                        for &x3 in &corners {
                            let point = arr1(&[x0, x1, x2, x3]).into_dyn();
                            let concrete = BoundedTensor::concrete(point).unwrap();
                            let out = network.propagate_ibp(&concrete).unwrap();
                            let y = out.lower()[[0]];

                            assert!(
                                y >= crown_lower - tol,
                                "CROWN lower unsound: f({},{},{},{})={} < lower {} (tol={})",
                                x0,
                                x1,
                                x2,
                                x3,
                                y,
                                crown_lower,
                                tol,
                            );
                            assert!(
                                y <= crown_upper + tol,
                                "CROWN upper unsound: f({},{},{},{})={} > upper {} (tol={})",
                                x0,
                                x1,
                                x2,
                                x3,
                                y,
                                crown_upper,
                                tol,
                            );
                        }
                    }
                }
            }
        }
        Err(e) => {
            // If CROWN returned an error, that's acceptable — it means the propagation
            // detected the overflow and gave up rather than returning wrong answers.
            // Document this for the issue.
            println!(
                "CROWN returned error for deep network (acceptable fail-fast): {e}. \
                 IBP bounds: [{}, {}]",
                ibp_lower, ibp_upper
            );
        }
    }
}

// ============================================================
// ALPHA-CROWN ELEMENTWISE BEST-BOUND UPDATE REGRESSIONS (#2087)
// ============================================================
//
// These tests verify the flat `iter_mut().zip(iter())` pattern used in
// alpha-CROWN for elementwise best-bound updates. Before #2087,
// `ndarray::Zip` was used, which panics when arrays have different ndim
// but the same element count (e.g., Conv2d graphs: shape [1,1] vs [1]).
//
// The old test (`elementwise_best_bound_update_works_with_non_contiguous_arrays`
// in alpha_crown.rs) passes trivially because `.to_owned()` makes the array
// contiguous and `Zip` is not the production pattern. These replacement tests
// exercise the actual #2087 scenario: different ndim, same element count.
//
// Reference: #2085 (test-quality finding), #2087 (the fix)

/// Regression test for #2087: elementwise best-bound update across different ndim.
///
/// Conv2d graph networks produce 2D [1,1] bounds from CROWN forward but
/// 1D [1] bounds from the backward pass. The flat iterator zip must handle
/// this without panicking (unlike ndarray::Zip which requires shape equality).
#[test]
fn test_alpha_crown_best_bound_update_different_ndim_2087() {
    use ndarray::{ArrayD, IxDyn};

    // Simulate the Conv2d graph case: CROWN returns 2D [1,1], backward returns 1D [1].
    let mut best_lower = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0f32]).unwrap();
    let concrete_lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![5.0f32]).unwrap();

    // Precondition: same element count, different ndim.
    assert_eq!(best_lower.ndim(), 2);
    assert_eq!(concrete_lower.ndim(), 1);
    assert_eq!(best_lower.len(), concrete_lower.len());

    // Production pattern from alpha_crown.rs:372-381
    assert_eq!(
        best_lower.len(),
        concrete_lower.len(),
        "best_lower and concrete_lower must have the same number of elements"
    );
    for (best, &curr) in best_lower.iter_mut().zip(concrete_lower.iter()) {
        if curr > *best {
            *best = curr;
        }
    }

    assert_eq!(
        best_lower.iter().next(),
        Some(&5.0f32),
        "Lower bound should be updated from 0.0 to 5.0"
    );

    // Upper bound pattern: min update across different ndim.
    let mut best_upper = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![f32::INFINITY]).unwrap();
    let concrete_upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0f32]).unwrap();

    assert_eq!(
        best_upper.len(),
        concrete_upper.len(),
        "best_upper and concrete_upper must have the same number of elements"
    );
    for (best, &curr) in best_upper.iter_mut().zip(concrete_upper.iter()) {
        if curr < *best {
            *best = curr;
        }
    }

    assert_eq!(
        best_upper.iter().next(),
        Some(&3.0f32),
        "Upper bound should be updated from inf to 3.0"
    );
}

/// Regression test for #2087: multi-element cross-ndim best-bound update.
///
/// Verifies that flat `iter_mut().zip(iter())` traverses elements in consistent
/// row-major order across arrays with different ndim (e.g., [2,3] vs [6]).
/// This ensures the elementwise max/min pairing is correct.
#[test]
fn test_alpha_crown_best_bound_update_multi_element_cross_ndim_2087() {
    use ndarray::{ArrayD, IxDyn};

    // 2D shape [2,3] = 6 elements vs 1D shape [6] = 6 elements.
    let mut best_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
    let concrete_lower =
        ArrayD::from_shape_vec(IxDyn(&[6]), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    assert_eq!(best_lower.len(), concrete_lower.len());
    for (best, &curr) in best_lower.iter_mut().zip(concrete_lower.iter()) {
        if curr > *best {
            *best = curr;
        }
    }

    // All 6 elements should be updated in row-major order.
    let result: Vec<f32> = best_lower.iter().copied().collect();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Upper bound: start high, min-update from 1D array.
    let mut best_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0f32, 10.0, 10.0, 10.0, 10.0, 10.0])
            .unwrap();
    let concrete_upper =
        ArrayD::from_shape_vec(IxDyn(&[6]), vec![9.0f32, 8.0, 7.0, 6.0, 5.0, 4.0]).unwrap();

    assert_eq!(best_upper.len(), concrete_upper.len());
    for (best, &curr) in best_upper.iter_mut().zip(concrete_upper.iter()) {
        if curr < *best {
            *best = curr;
        }
    }

    let result: Vec<f32> = best_upper.iter().copied().collect();
    assert_eq!(result, vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0]);
}

// ============================================================
// CROWN OUTPUT MUST BE INTERSECTED WITH IBP (#2990)
// ============================================================
//
// CROWN's linear relaxation can be strictly looser than IBP for certain
// weight/input configurations. A negative output weight amplifies the
// ReLU lower relaxation error (alpha*x permits y < 0 when x < 0, but
// ReLU(x) >= 0 always). The fix intersects CROWN output with IBP forward
// bounds, matching the graph batched path (#2904).
//
// Reference: alpha-beta-CROWN optimized_bounds.py:937-947.

/// Regression test for #2990: CROWN must not be looser than IBP.
///
/// Minimal failing input from proptest:
///   Linear(2->2) -> ReLU -> Linear(2->1)
///   w1 = [[0.71, -0.84], [0, 0]], b1 = [0, 0]
///   w2 = [[-0.88, 0]], b2 = 0
///   input = [[0, 0], [-0.78, 0.46]]
///
/// IBP upper = 0.0 (exact), old CROWN upper = 0.34 (too loose).
/// After #2990 fix, CROWN output is intersected with IBP so CROWN <= IBP.
#[ntest::timeout(10000)]
#[test]
fn test_crown_tighter_than_ibp_linear_relu_linear_2990() {
    // Exact weights from issue #2990 proptest minimal input.
    let w1 = arr2(&[[0.71184605_f32, -0.8403961], [0.0, 0.0]]);
    let b1 = arr1(&[0.0_f32, 0.0]);
    let w2 = arr2(&[[-0.8830249_f32, 0.0]]);
    let b2 = arr1(&[0.0_f32]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    // Issue specifies input as rows [[0, 0], [-0.78, 0.46]], meaning:
    //   lower = [min(0, -0.78), min(0, 0.46)] = [-0.78, 0]
    //   upper = [max(0, -0.78), max(0, 0.46)] = [0, 0.46]
    let lower = arr1(&[-0.77716047_f32, 0.0]).into_dyn();
    let upper = arr1(&[0.0_f32, 0.4639826]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_output = network.propagate_crown(&input).unwrap();

    let ibp_upper = ibp_output.upper()[[0]];
    let crown_upper = crown_output.upper()[[0]];
    let ibp_lower = ibp_output.lower()[[0]];
    let crown_lower = crown_output.lower()[[0]];

    // CROWN must be at least as tight as IBP (with rounding tolerance).
    // The IBP intersection guarantees CROWN_lower >= IBP_lower and CROWN_upper <= IBP_upper.
    let tol = 1e-5;
    assert!(
        crown_upper <= ibp_upper + tol,
        "CROWN upper {} must not exceed IBP upper {} (relaxation error not intersected with IBP). \
         This is the #2990 regression: CROWN was {:.4} too loose.",
        crown_upper,
        ibp_upper,
        crown_upper - ibp_upper,
    );
    assert!(
        crown_lower >= ibp_lower - tol,
        "CROWN lower {} must not be below IBP lower {} (relaxation error not intersected with IBP)",
        crown_lower,
        ibp_lower,
    );

    // Verify soundness: concrete evaluations must be within CROWN bounds.
    let test_points = vec![
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[-0.77716047_f32, 0.0]).into_dyn(),
        arr1(&[0.0_f32, 0.4639826]).into_dyn(),
        arr1(&[-0.77716047_f32, 0.4639826]).into_dyn(),
    ];
    for point in &test_points {
        let concrete = BoundedTensor::concrete(point.clone()).unwrap();
        let out = network.propagate_ibp(&concrete).unwrap();
        let y = out.lower()[[0]];
        assert!(
            y >= crown_lower - tol,
            "Soundness: f({:?}) = {} < CROWN lower {}",
            point,
            y,
            crown_lower,
        );
        assert!(
            y <= crown_upper + tol,
            "Soundness: f({:?}) = {} > CROWN upper {}",
            point,
            y,
            crown_upper,
        );
    }
}

/// Same test for propagate_crown_ibp path (#2990).
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_tighter_than_ibp_linear_relu_linear_2990() {
    let w1 = arr2(&[[0.71184605_f32, -0.8403961], [0.0, 0.0]]);
    let b1 = arr1(&[0.0_f32, 0.0]);
    let w2 = arr2(&[[-0.8830249_f32, 0.0]]);
    let b2 = arr1(&[0.0_f32]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let lower = arr1(&[-0.77716047_f32, 0.0]).into_dyn();
    let upper = arr1(&[0.0_f32, 0.4639826]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let ibp_output = network.propagate_ibp(&input).unwrap();
    let crown_ibp_output = network.propagate_crown_ibp(&input).unwrap();

    let ibp_upper = ibp_output.upper()[[0]];
    let crown_ibp_upper = crown_ibp_output.upper()[[0]];
    let ibp_lower = ibp_output.lower()[[0]];
    let crown_ibp_lower = crown_ibp_output.lower()[[0]];

    let tol = 1e-5;
    assert!(
        crown_ibp_upper <= ibp_upper + tol,
        "CROWN-IBP upper {} must not exceed IBP upper {} (#2990)",
        crown_ibp_upper,
        ibp_upper,
    );
    assert!(
        crown_ibp_lower >= ibp_lower - tol,
        "CROWN-IBP lower {} must not be below IBP lower {} (#2990)",
        crown_ibp_lower,
        ibp_lower,
    );
}

// ============================================================
// DEADLINE ENFORCEMENT REGRESSION TESTS (#3328)
// ============================================================

/// An already-elapsed fresh CROWN request cannot start a new IBP sweep merely
/// to manufacture a fallback result. It must preserve hard authority as a
/// typed deadline refusal.
#[test]
fn test_crown_deadline_already_elapsed_refuses_before_fresh_ibp_3328() {
    use std::time::{Duration, Instant};

    // Build a simple 2-layer network: Linear(2→3) + ReLU + Linear(3→1)
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7]]);
    let b1 = arr1(&[0.1, -0.1, 0.0]);
    let w2 = arr2(&[[0.5, -0.3, 0.8]]);
    let b2 = arr1(&[0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let lower = arr1(&[-1.0, -1.0]).into_dyn();
    let upper = arr1(&[1.0, 1.0]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // CROWN with no deadline — normal bounds.
    let crown_bounds = network
        .propagate_crown_with_engine_and_deadline(&input, None, None)
        .unwrap();

    // CROWN with a deadline already in the past has no precollected output to
    // publish and therefore must refuse before doing O(N) fallback work.
    let past_deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let error = network
        .propagate_crown_with_engine_and_deadline(&input, None, past_deadline)
        .expect_err("expired fresh CROWN must preserve hard deadline authority");
    assert!(
        matches!(error, NyError::DeadlineExceeded(_)),
        "expected typed deadline refusal, got {error:?}"
    );

    // IBP bounds for reference.
    let ibp_bounds = network.propagate_ibp(&input).unwrap();
    let tol = 1e-6;

    // CROWN bounds should be at least as tight as IBP (CROWN ⊆ IBP).
    for i in 0..crown_bounds.len() {
        assert!(
            crown_bounds.lower().as_slice().unwrap()[i]
                >= ibp_bounds.lower().as_slice().unwrap()[i] - tol,
            "CROWN lower[{}] should be >= IBP lower",
            i
        );
        assert!(
            crown_bounds.upper().as_slice().unwrap()[i]
                <= ibp_bounds.upper().as_slice().unwrap()[i] + tol,
            "CROWN upper[{}] should be <= IBP upper",
            i
        );
    }
}

/// A fresh CROWN-IBP collection cannot start its full forward sweep after the
/// deadline has already elapsed.
#[test]
fn test_crown_ibp_elapsed_deadline_refuses_before_forward_sweep_3328() {
    use std::time::{Duration, Instant};

    // 2-layer network: Linear(2→3) + ReLU + Linear(3→1)
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0], [0.3, -0.7]]);
    let b1 = arr1(&[0.1, -0.1, 0.0]);
    let w2 = arr2(&[[0.5, -0.3, 0.8]]);
    let b2 = arr1(&[0.0]);

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()));

    let lower = arr1(&[-1.0, -1.0]).into_dyn();
    let upper = arr1(&[1.0, 1.0]).into_dyn();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Collect CROWN-IBP with elapsed deadline.
    let past_deadline = Some(Instant::now().checked_sub(Duration::from_secs(1)).unwrap());
    let error = network
        .collect_crown_ibp_bounds_with_engine_and_deadline(&input, None, past_deadline)
        .expect_err("expired collection must refuse before fresh IBP work");
    assert!(matches!(error, NyError::DeadlineExceeded(_)));
}
