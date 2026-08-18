// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the beta-CROWN backward passes.
//!
//! The `arelu_cut` variant these originally targeted has been deleted (it was
//! dead behind `BetaCrownConfig::cut_proof_authority_enabled()`). The cases
//! that ran it with an EMPTY cut state were exercising arithmetic identical to
//! `relu_backward_with_alpha_beta`, so they were retargeted at that live
//! function rather than dropped; the cases that asserted cut arithmetic went
//! with the code.

use super::prelude::*;
use crate::tests::assert_linear_bounds_close;

/// Test relu_backward_with_alpha_beta on stable positive neurons.
/// Stable neurons should have slope=1, intercept=0.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_stable_positive() {
    // Network with one ReLU layer
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Input bounds that make pre-ReLU outputs strictly positive (stable positive)
    let input =
        BoundedTensor::new(arr1(&[1.0, 3.0]).into_dyn(), arr1(&[2.0, 4.0]).into_dyn()).unwrap();

    // Get layer bounds
    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    // Pre-ReLU bounds should be strictly positive
    let pre_relu = &layer_bounds[0];
    println!(
        "Pre-ReLU bounds: L=[{:.2}, {:.2}], U=[{:.2}, {:.2}]",
        pre_relu.lower()[[0]],
        pre_relu.lower()[[1]],
        pre_relu.upper()[[0]],
        pre_relu.upper()[[1]]
    );
    assert!(
        pre_relu.lower()[[0]] > 0.0 && pre_relu.lower()[[1]] > 0.0,
        "Expected stable positive neurons"
    );

    // Empty state objects (no constraints)
    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    // Create verifier and use its backward method
    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Create output bounds (identity matrix for 2 outputs)
    let output_bounds = LinearBounds::identity(2);

    // Call relu_backward_with_alpha_beta through the verifier
    let result = verifier.relu_backward_with_alpha_beta(
        &output_bounds,
        pre_relu,
        None,
        &beta_state,
        &alpha_state,
        1, // ReLU is layer 1
    );

    let new_bounds = result.expect("Should succeed");

    // For stable positive neurons: new_A = I (identity), new_b = 0
    // Since both neurons are stable positive, output should equal input
    for i in 0..2 {
        for j in 0..2 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (new_bounds.lower_a[[i, j]] - expected).abs() < 1e-6,
                "Stable positive: lower_a[{},{}] should be {}",
                i,
                j,
                expected
            );
            assert!(
                (new_bounds.upper_a[[i, j]] - expected).abs() < 1e-6,
                "Stable positive: upper_a[{},{}] should be {}",
                i,
                j,
                expected
            );
        }
        assert!(
            new_bounds.lower_b[i].abs() < 1e-6,
            "Stable positive: lower intercept should be 0"
        );
        assert!(
            new_bounds.upper_b[i].abs() < 1e-6,
            "Stable positive: upper intercept should be 0"
        );
    }
}

/// Test relu_backward_with_alpha_beta on stable negative neurons.
/// Stable neurons should have slope=0, intercept=0.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_stable_negative() {
    let w1 = arr2(&[[1.0, 0.5], [-0.5, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Input bounds that make pre-ReLU outputs strictly negative (stable negative)
    let input = BoundedTensor::new(
        arr1(&[-4.0, -4.0]).into_dyn(),
        arr1(&[-3.0, -3.0]).into_dyn(),
    )
    .unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let pre_relu = &layer_bounds[0];
    println!(
        "Pre-ReLU bounds: L=[{:.2}, {:.2}], U=[{:.2}, {:.2}]",
        pre_relu.lower()[[0]],
        pre_relu.lower()[[1]],
        pre_relu.upper()[[0]],
        pre_relu.upper()[[1]]
    );
    assert!(
        pre_relu.upper()[[0]] < 0.0 && pre_relu.upper()[[1]] < 0.0,
        "Expected stable negative neurons"
    );

    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let output_bounds = LinearBounds::identity(2);

    let result = verifier.relu_backward_with_alpha_beta(
        &output_bounds,
        pre_relu,
        None,
        &beta_state,
        &alpha_state,
        1,
    );

    let new_bounds = result.expect("Should succeed");

    // For stable negative neurons: slope=0 and intercept=0
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                new_bounds.lower_a[[i, j]].abs() < 1e-6,
                "Stable negative: lower_a[{},{}] should be 0",
                i,
                j
            );
            assert!(
                new_bounds.upper_a[[i, j]].abs() < 1e-6,
                "Stable negative: upper_a[{},{}] should be 0",
                i,
                j
            );
        }
        assert!(
            new_bounds.lower_b[i].abs() < 1e-6,
            "Stable negative: lower intercept should be 0"
        );
        assert!(
            new_bounds.upper_b[i].abs() < 1e-6,
            "Stable negative: upper intercept should be 0"
        );
    }
}

/// Test relu_backward_with_alpha_beta on unstable neurons.
/// Unstable neurons use alpha for lower bound and triangle relaxation for upper.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_unstable() {
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // Identity
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Input bounds that make pre-ReLU outputs cross zero (unstable)
    // l=-1, u=2
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn()).unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let pre_relu = &layer_bounds[0];
    let l = pre_relu.lower()[[0]];
    let u = pre_relu.upper()[[0]];
    println!("Pre-ReLU bounds: l={:.2}, u={:.2}", l, u);
    assert!(l < 0.0 && u > 0.0, "Expected unstable neurons");

    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Use identity output bounds
    let output_bounds = LinearBounds::identity(2);

    let result = verifier.relu_backward_with_alpha_beta(
        &output_bounds,
        pre_relu,
        None,
        &beta_state,
        &alpha_state,
        1,
    );

    let new_bounds = result.expect("Should succeed");

    // For unstable neurons:
    // When A_ij >= 0: lower slope = alpha, lower intercept = 0
    // When A_ij >= 0: upper slope = u/(u-l), upper intercept = -l*u/(u-l)
    let expected_upper_slope = u / (u - l);
    let expected_upper_intercept = -l * u / (u - l);

    for j in 0..2 {
        // Check lower bound coefficients
        let alpha_j = alpha_state.alpha(1, j);
        assert!(
            (new_bounds.lower_a[[j, j]] - alpha_j).abs() < 1e-6,
            "Unstable: lower slope should be alpha={} for neuron {}",
            alpha_j,
            j
        );
        assert!(
            new_bounds.lower_b[j].abs() < 1e-6,
            "Unstable: lower intercept should be 0"
        );

        // Check upper bound coefficients
        assert!(
            (new_bounds.upper_a[[j, j]] - expected_upper_slope).abs() < 1e-6,
            "Unstable: upper slope should be u/(u-l)={} for neuron {}",
            expected_upper_slope,
            j
        );
        assert!(
            (new_bounds.upper_b[j] - expected_upper_intercept).abs() < 1e-6,
            "Unstable: upper intercept should be -lu/(u-l)={}",
            expected_upper_intercept
        );
    }
}

/// Test that a branching constraint overrides the unstable relaxation.
/// Constrained neurons should use fixed slopes regardless of their bounds.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_constrained_neurons() {
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Unstable neurons
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[2.0, 2.0]).into_dyn()).unwrap();

    let layer_bounds: Vec<Arc<BoundedTensor>> = network
        .collect_ibp_bounds(&input)
        .unwrap()
        .into_iter()
        .map(Arc::new)
        .collect();

    let pre_relu = &layer_bounds[0];

    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    let output_bounds = LinearBounds::identity(2);

    // Create constraints: neuron 0 is active (is_active=true)
    let mut constraints = std::collections::HashMap::new();
    constraints.insert(0_usize, true); // Neuron 0 is active

    let result = verifier.relu_backward_with_alpha_beta(
        &output_bounds,
        pre_relu,
        Some(&constraints),
        &beta_state,
        &alpha_state,
        1,
    );

    let new_bounds = result.expect("Should succeed");

    // Constrained active neuron should have slope=1, intercept=0
    assert!(
        (new_bounds.lower_a[[0, 0]] - 1.0).abs() < 1e-6,
        "Active constrained neuron should have slope 1"
    );
    assert!(
        new_bounds.lower_b[0].abs() < 1e-6,
        "Active constrained neuron should have intercept 0"
    );
}

// ============================================================================
// Near-zero-width regression tests for RELU_RELAX_MIN_WIDTH (#1645)
//
// These test that all backward.rs ReLU relaxation paths produce finite results
// when the pre-activation interval width (u - l) is extremely small but the
// neuron is still unstable (l < 0 < u). Without the MIN_WIDTH clamp, division
// by (u - l) ≈ 0 would produce Inf/NaN.
// ============================================================================

/// Helper: create near-zero-width pre-activation bounds for 2 neurons.
/// Neuron 0: near-zero width crossing interval (l ≈ -eps, u ≈ eps).
/// Neuron 1: normal unstable interval for comparison.
fn near_zero_width_pre_bounds() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-1e-12_f32, -1.0]).into_dyn(),
        arr1(&[1e-12_f32, 2.0]).into_dyn(),
    )
    .unwrap()
}

/// Helper: create a 2-neuron network and return layer bounds, alpha state.
fn near_zero_width_setup() -> (
    Network,
    Vec<Arc<BoundedTensor>>,
    DomainAlphaState,
    BetaState,
) {
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));

    // Use the near-zero-width bounds as "layer 0 output" (pre-ReLU)
    let pre_bounds = near_zero_width_pre_bounds();
    let layer_bounds: Vec<Arc<BoundedTensor>> = vec![Arc::new(pre_bounds)];

    let history = SplitHistory::new();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let beta_state = BetaState::empty();

    (network, layer_bounds, alpha_state, beta_state)
}

/// Helper: create NaN-tainted pre-activation bounds for 2 neurons.
/// Uses test-only unchecked constructor to simulate upstream numerical corruption.
fn nan_pre_bounds() -> BoundedTensor {
    BoundedTensor::new_unchecked(
        arr1(&[f32::NAN, -1.0_f32]).into_dyn(),
        arr1(&[1.0_f32, f32::NAN]).into_dyn(),
    )
    .expect("test-only unchecked bounds should allow NaN endpoints")
}

/// Helper: create a 2-neuron setup with NaN pre-activation bounds.
fn nan_bounds_setup() -> (
    Network,
    Vec<Arc<BoundedTensor>>,
    DomainAlphaState,
    BetaState,
) {
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));

    let pre_bounds = nan_pre_bounds();
    let layer_bounds: Vec<Arc<BoundedTensor>> = vec![Arc::new(pre_bounds)];

    let history = SplitHistory::new();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let beta_state = BetaState::empty();

    (network, layer_bounds, alpha_state, beta_state)
}

/// Helper: create constrained setup with non-finite β on neuron 0 at layer 1.
/// `is_active=true` yields +Inf signed β, `is_active=false` yields -Inf signed β.
fn non_finite_beta_setup(is_active: bool) -> (Arc<BoundedTensor>, DomainAlphaState, BetaState) {
    let w1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let linear1 = LinearLayer::new(w1, None).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));

    let pre_bounds = near_zero_width_pre_bounds();
    let pre_bounds = Arc::new(pre_bounds);
    let layer_bounds: Vec<Arc<BoundedTensor>> = vec![pre_bounds.clone()];

    let mut history = SplitHistory::new();
    history.add_constraint(NeuronConstraint {
        layer_idx: 1,
        neuron_idx: 0,
        is_active,
        score: 0.0,
    });

    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);
    let mut beta_state = BetaState::from_history(&history).unwrap();
    beta_state.entries[0].value = f32::INFINITY;

    (pre_bounds, alpha_state, beta_state)
}

/// Verify LinearBounds are finite and sound (lower_a, upper_a, lower_b, upper_b all finite).
fn assert_bounds_finite(bounds: &LinearBounds, label: &str) {
    for (i, v) in bounds.lower_a.iter().enumerate() {
        assert!(v.is_finite(), "{label}: lower_a[{i}] = {v} is not finite");
    }
    for (i, v) in bounds.upper_a.iter().enumerate() {
        assert!(v.is_finite(), "{label}: upper_a[{i}] = {v} is not finite");
    }
    for (i, v) in bounds.lower_b.iter().enumerate() {
        assert!(v.is_finite(), "{label}: lower_b[{i}] = {v} is not finite");
    }
    for (i, v) in bounds.upper_b.iter().enumerate() {
        assert!(v.is_finite(), "{label}: upper_b[{i}] = {v} is not finite");
    }
}

/// Verify LinearBounds contain no NaN values (infinite values are allowed).
fn assert_bounds_no_nan(bounds: &LinearBounds, label: &str) {
    for (i, v) in bounds.lower_a.iter().enumerate() {
        assert!(!v.is_nan(), "{label}: lower_a[{i}] is NaN");
    }
    for (i, v) in bounds.upper_a.iter().enumerate() {
        assert!(!v.is_nan(), "{label}: upper_a[{i}] is NaN");
    }
    for (i, v) in bounds.lower_b.iter().enumerate() {
        assert!(!v.is_nan(), "{label}: lower_b[{i}] is NaN");
    }
    for (i, v) in bounds.upper_b.iter().enumerate() {
        assert!(!v.is_nan(), "{label}: upper_b[{i}] is NaN");
    }
}

/// #2826 regression: relu_backward_with_alpha_beta must skip non-finite β.
#[ntest::timeout(5000)]
#[test]
fn test_non_finite_beta_alpha_beta_2826() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let output_bounds = LinearBounds::identity(2);
    for is_active in [true, false] {
        let label = if is_active { "+inf" } else { "-inf" };
        let (pre_relu, alpha, inf_beta) = non_finite_beta_setup(is_active);
        let mut zero_beta = inf_beta.clone();
        zero_beta.entries[0].value = 0.0;
        let baseline = verifier
            .relu_backward_with_alpha_beta(
                &output_bounds,
                pre_relu.as_ref(),
                None,
                &zero_beta,
                &alpha,
                1,
            )
            .expect("zero-beta should succeed");
        let result = verifier
            .relu_backward_with_alpha_beta(
                &output_bounds,
                pre_relu.as_ref(),
                None,
                &inf_beta,
                &alpha,
                1,
            )
            .expect("non-finite beta should succeed");
        assert_linear_bounds_close(&result, &baseline, 1e-6, &format!("alpha_beta ({label})"));
        assert_bounds_finite(&result, &format!("alpha_beta ({label})"));
    }
}

/// #2826 regression: relu_backward_with_beta must skip non-finite β.
#[ntest::timeout(5000)]
#[test]
fn test_non_finite_beta_legacy_2826() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let output_bounds = LinearBounds::identity(2);
    for is_active in [true, false] {
        let label = if is_active { "+inf" } else { "-inf" };
        let (pre_relu, _alpha, inf_beta) = non_finite_beta_setup(is_active);
        let mut zero_beta = inf_beta.clone();
        zero_beta.entries[0].value = 0.0;
        let baseline = verifier
            .relu_backward_with_beta(&output_bounds, pre_relu.as_ref(), None, &zero_beta, 1)
            .expect("zero-beta should succeed");
        let result = verifier
            .relu_backward_with_beta(&output_bounds, pre_relu.as_ref(), None, &inf_beta, 1)
            .expect("non-finite beta should succeed");
        assert_linear_bounds_close(&result, &baseline, 1e-6, &format!("legacy_beta ({label})"));
        assert_bounds_finite(&result, &format!("legacy_beta ({label})"));
    }
}

/// #2826 regression: relu_backward_with_beta_record_relaxation must skip non-finite β.
#[ntest::timeout(5000)]
#[test]
fn test_non_finite_beta_record_relaxation_2826() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let record_bounds = LinearBounds {
        lower_a: Array2::ones((1, 2)),
        lower_b: Array1::zeros(1),
        upper_a: Array2::ones((1, 2)),
        upper_b: Array1::zeros(1),
        lower_a_err: None,
        upper_a_err: None,
    };
    for is_active in [true, false] {
        let label = if is_active { "+inf" } else { "-inf" };
        let (pre_relu, _alpha, inf_beta) = non_finite_beta_setup(is_active);
        let mut zero_beta = inf_beta.clone();
        zero_beta.entries[0].value = 0.0;
        let (baseline, base_relax) = verifier
            .relu_backward_with_beta_record_relaxation(
                &record_bounds,
                pre_relu.as_ref(),
                None,
                &zero_beta,
                1,
            )
            .expect("zero-beta should succeed");
        let (result, inf_relax) = verifier
            .relu_backward_with_beta_record_relaxation(
                &record_bounds,
                pre_relu.as_ref(),
                None,
                &inf_beta,
                1,
            )
            .expect("non-finite beta should succeed");
        assert_linear_bounds_close(
            &result,
            &baseline,
            1e-6,
            &format!("record_relaxation ({label})"),
        );
        assert_bounds_finite(&result, &format!("record_relaxation ({label})"));
        for (i, (a, e)) in inf_relax
            .slopes
            .iter()
            .zip(base_relax.slopes.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 1e-6,
                "slope mismatch at {i} ({label}): {a} vs {e}"
            );
        }
        for (i, (a, e)) in inf_relax
            .intercepts
            .iter()
            .zip(base_relax.intercepts.iter())
            .enumerate()
        {
            assert!(
                (a - e).abs() <= 1e-6,
                "intercept mismatch at {i} ({label}): {a} vs {e}"
            );
        }
    }
}

/// #1645 regression: relu_backward_with_alpha_beta with near-zero-width interval.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_near_zero_width_1645() {
    let (_network, layer_bounds, alpha_state, beta_state) = near_zero_width_setup();
    let pre_relu = &layer_bounds[0];

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);
    let output_bounds = LinearBounds::identity(2);

    let result = verifier.relu_backward_with_alpha_beta(
        &output_bounds,
        pre_relu,
        None,
        &beta_state,
        &alpha_state,
        1,
    );

    let bounds = result.expect("Should succeed with near-zero-width interval");
    assert_bounds_finite(&bounds, "relu_backward_with_alpha_beta");
}

/// #1645 regression: relu_backward_with_beta with near-zero-width interval.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_beta_near_zero_width_1645() {
    let (_network, layer_bounds, _alpha_state, beta_state) = near_zero_width_setup();
    let pre_relu = &layer_bounds[0];

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);
    let output_bounds = LinearBounds::identity(2);

    let result = verifier.relu_backward_with_beta(&output_bounds, pre_relu, None, &beta_state, 1);

    let bounds = result.expect("Should succeed with near-zero-width interval");
    assert_bounds_finite(&bounds, "relu_backward_with_beta");
}

/// #1645 regression: relu_backward_with_beta_record_relaxation with near-zero-width interval.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_beta_record_near_zero_width_1645() {
    let (_network, layer_bounds, _alpha_state, beta_state) = near_zero_width_setup();
    let pre_relu = &layer_bounds[0];

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);
    // This function requires a single objective output row.
    let output_bounds = LinearBounds {
        lower_a: Array2::ones((1, 2)),
        lower_b: Array1::zeros(1),
        upper_a: Array2::ones((1, 2)),
        upper_b: Array1::zeros(1),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = verifier.relu_backward_with_beta_record_relaxation(
        &output_bounds,
        pre_relu,
        None,
        &beta_state,
        1,
    );

    let (bounds, relaxation) = result.expect("Should succeed with near-zero-width interval");
    assert_bounds_finite(&bounds, "relu_backward_with_beta_record_relaxation");

    // Verify relaxation slopes/intercepts are also finite
    for (i, &s) in relaxation.slopes.iter().enumerate() {
        assert!(s.is_finite(), "relaxation slope[{i}] = {s} is not finite");
    }
    for (i, &c) in relaxation.intercepts.iter().enumerate() {
        assert!(
            c.is_finite(),
            "relaxation intercept[{i}] = {c} is not finite"
        );
    }
}

/// #2805 regression: NaN pre-activation bounds must fail closed without NaN poisoning.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_nan_pre_bounds_2805() {
    let (_network, layer_bounds, alpha_state, beta_state) = nan_bounds_setup();
    let pre_relu = &layer_bounds[0];

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);
    let output_bounds = LinearBounds::identity(2);

    let result = verifier.relu_backward_with_alpha_beta(
        &output_bounds,
        pre_relu,
        None,
        &beta_state,
        &alpha_state,
        1,
    );

    let bounds = result.expect("NaN pre-bounds should fail closed, not error");
    assert_bounds_no_nan(&bounds, "relu_backward_with_alpha_beta NaN guard");

    for (i, v) in bounds.lower_b.iter().enumerate() {
        assert!(
            v.is_infinite() && v.is_sign_negative(),
            "lower_b[{i}] should be -inf for fail-closed NaN handling, got {v}"
        );
    }
    for (i, v) in bounds.upper_b.iter().enumerate() {
        assert!(
            v.is_infinite() && v.is_sign_positive(),
            "upper_b[{i}] should be +inf for fail-closed NaN handling, got {v}"
        );
    }
}

/// #2805 regression: NaN pre-activation bounds must fail closed on the NEGATIVE
/// `la_ij` branch too. The sibling test above uses `LinearBounds::identity`, so
/// it only drives `la_ij >= 0`; this one covers `la_ij < 0`, where the
/// fail-closed `+inf` upper intercept has to flip the lower bias to `-inf`.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_nan_pre_bounds_negative_coeffs_2805() {
    let (_network, layer_bounds, alpha_state, beta_state) = nan_bounds_setup();
    let pre_relu = &layer_bounds[0];

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);

    // Include negative, positive, and zero coefficients to cover all sign branches.
    let output_bounds = LinearBounds {
        lower_a: arr2(&[[-1.0, 0.0], [0.0, -2.0]]),
        lower_b: Array1::zeros(2),
        upper_a: Array2::eye(2),
        upper_b: Array1::zeros(2),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = verifier.relu_backward_with_alpha_beta(
        &output_bounds,
        pre_relu,
        None,
        &beta_state,
        &alpha_state,
        1,
    );

    let bounds = result.expect("NaN pre-bounds should fail closed, not error");
    assert_bounds_no_nan(
        &bounds,
        "relu_backward_with_alpha_beta NaN guard (negative coeffs)",
    );

    for (i, v) in bounds.lower_b.iter().enumerate() {
        assert!(
            v.is_infinite() && v.is_sign_negative(),
            "lower_b[{i}] should be -inf for fail-closed NaN handling, got {v}"
        );
    }
    for (i, v) in bounds.upper_b.iter().enumerate() {
        assert!(
            v.is_infinite() && v.is_sign_positive(),
            "upper_b[{i}] should be +inf for fail-closed NaN handling, got {v}"
        );
    }
}

/// #2805 regression: legacy relu_backward_with_beta must fail closed on NaN pre-bounds.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_beta_nan_pre_bounds_2805() {
    let (_network, layer_bounds, _alpha_state, beta_state) = nan_bounds_setup();
    let pre_relu = &layer_bounds[0];

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);
    let output_bounds = LinearBounds::identity(2);

    let result = verifier.relu_backward_with_beta(&output_bounds, pre_relu, None, &beta_state, 1);

    let bounds = result.expect("NaN pre-bounds should fail closed, not error");
    assert_bounds_no_nan(&bounds, "relu_backward_with_beta NaN guard");

    for (i, v) in bounds.lower_b.iter().enumerate() {
        assert!(
            v.is_infinite() && v.is_sign_negative(),
            "lower_b[{i}] should be -inf for fail-closed NaN handling, got {v}"
        );
    }
    for (i, v) in bounds.upper_b.iter().enumerate() {
        assert!(
            v.is_infinite() && v.is_sign_positive(),
            "upper_b[{i}] should be +inf for fail-closed NaN handling, got {v}"
        );
    }
}

/// #2805 regression: legacy relu_backward_with_beta_record_relaxation must fail closed on NaN.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_beta_record_nan_pre_bounds_2805() {
    let (_network, layer_bounds, _alpha_state, beta_state) = nan_bounds_setup();
    let pre_relu = &layer_bounds[0];

    let config = BetaCrownConfig::default();
    let verifier = BetaCrownVerifier::new(config);
    // This function requires a single objective output row.
    let output_bounds = LinearBounds {
        lower_a: Array2::ones((1, 2)),
        lower_b: Array1::zeros(1),
        upper_a: Array2::ones((1, 2)),
        upper_b: Array1::zeros(1),
        lower_a_err: None,
        upper_a_err: None,
    };

    let result = verifier.relu_backward_with_beta_record_relaxation(
        &output_bounds,
        pre_relu,
        None,
        &beta_state,
        1,
    );

    let (bounds, relaxation) = result.expect("NaN pre-bounds should fail closed, not error");
    assert_bounds_no_nan(
        &bounds,
        "relu_backward_with_beta_record_relaxation NaN guard",
    );

    // Both neurons have NaN in at least one bound, so biases should be unbounded.
    // lower_b[0] accumulates contributions from both neurons via ones-row: both add -inf.
    assert!(
        bounds.lower_b[0].is_infinite() && bounds.lower_b[0].is_sign_negative(),
        "lower_b[0] should be -inf for fail-closed NaN handling, got {}",
        bounds.lower_b[0]
    );
    assert!(
        bounds.upper_b[0].is_infinite() && bounds.upper_b[0].is_sign_positive(),
        "upper_b[0] should be +inf for fail-closed NaN handling, got {}",
        bounds.upper_b[0]
    );

    // Relaxation slopes for NaN-tainted neurons should be the NaN-fallback values.
    // With la_j > 0 (ones-row), NaN neurons record lower_slope=0, lower_intercept=-inf.
    for (i, &s) in relaxation.slopes.iter().enumerate() {
        assert!(!s.is_nan(), "relaxation slope[{i}] should not be NaN");
    }
    for (i, &c) in relaxation.intercepts.iter().enumerate() {
        assert!(!c.is_nan(), "relaxation intercept[{i}] should not be NaN");
    }
}

/// Regression for #1840: wildcard dispatch in alpha-beta backward must delegate
/// to layer CROWN backward implementations (e.g., Sigmoid), not identity fallback.
#[ntest::timeout(5000)]
#[test]
fn test_propagate_layer_backward_alpha_beta_dispatches_sigmoid_1840() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let layer = Layer::Sigmoid(crate::SigmoidLayer);
    let output_bounds = LinearBounds::identity(1);
    let pre_bounds = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let beta_state = BetaState::empty();
    let alpha_state = DomainAlphaState::empty();

    let actual = verifier
        .propagate_layer_backward_with_alpha_beta(
            &layer,
            &output_bounds,
            &pre_bounds,
            None,
            &beta_state,
            &alpha_state,
            0,
            None,
        )
        .expect("alpha-beta backward should support Sigmoid via unified dispatch");

    let expected = crate::BoundPropagation::propagate_crown_backward(
        &layer,
        &output_bounds,
        Some(&pre_bounds),
    )
    .expect("Sigmoid CROWN backward should succeed");

    // Identity would leave lower_a=1.0; Sigmoid relaxation should change coefficients.
    assert!(
        (actual.lower_a[[0, 0]] - 1.0).abs() > 1e-4,
        "Sigmoid backward should not degrade to identity fallback"
    );
    assert_linear_bounds_close(&actual, &expected, 1e-5, "sigmoid_1840");
}

/// Regression for #1840: test-only beta backward helper should also route wildcard
/// layers through unified CROWN dispatch before IBP fallback.
#[ntest::timeout(5000)]
#[test]
fn test_propagate_layer_backward_beta_dispatches_sigmoid_1840() {
    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let layer = Layer::Sigmoid(crate::SigmoidLayer);
    let output_bounds = LinearBounds::identity(1);
    let pre_bounds = BoundedTensor::new(arr1(&[-1.5]).into_dyn(), arr1(&[2.5]).into_dyn()).unwrap();

    let beta_state = BetaState::empty();

    let actual = verifier
        .propagate_layer_backward_with_beta(
            &layer,
            &output_bounds,
            &pre_bounds,
            None,
            &beta_state,
            0,
            None,
        )
        .expect("legacy beta backward should support Sigmoid via unified dispatch");

    let expected = crate::BoundPropagation::propagate_crown_backward(
        &layer,
        &output_bounds,
        Some(&pre_bounds),
    )
    .expect("Sigmoid CROWN backward should succeed");

    assert!(
        (actual.lower_a[[0, 0]] - 1.0).abs() > 1e-4,
        "Sigmoid backward should not degrade to identity fallback"
    );
    assert_linear_bounds_close(&actual, &expected, 1e-5, "sigmoid_beta_1840");
}

/// Assert that 1-neuron LinearBounds contain no NaN and have finite slopes.
fn assert_relu_backward_no_nan(bounds: &LinearBounds, label: &str) {
    assert!(!bounds.lower_a[[0, 0]].is_nan(), "{label}: lower_a NaN");
    assert!(!bounds.upper_a[[0, 0]].is_nan(), "{label}: upper_a NaN");
    assert!(!bounds.lower_b[0].is_nan(), "{label}: lower_b NaN");
    assert!(!bounds.upper_b[0].is_nan(), "{label}: upper_b NaN");
    assert!(
        bounds.lower_a[[0, 0]].is_finite(),
        "{label}: lower_a not finite"
    );
    assert!(
        bounds.upper_a[[0, 0]].is_finite(),
        "{label}: upper_a not finite"
    );
}

/// Regression test for #2805: Inf pre-activation bounds must not produce NaN in
/// the beta-CROWN backward pass. Before this fix, Inf bounds fell through to the
/// unstable branch where (u - l) = Inf and u/Inf = NaN or -l*u/Inf = NaN.
///
/// The standalone relu_linear_relaxation() in relu/mod.rs has explicit Inf guards
/// (lines 37-48). The beta-CROWN backward path must match.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_inf_pre_activation_no_nan_2805() {
    let cases: Vec<(f32, f32, &str)> = vec![
        (f32::NEG_INFINITY, 5.0, "l=-Inf, u=5.0"),
        (-3.0, f32::INFINITY, "l=-3.0, u=+Inf"),
        (f32::NEG_INFINITY, f32::INFINITY, "l=-Inf, u=+Inf"),
    ];

    for (l_val, u_val, label) in &cases {
        let pre_bounds =
            BoundedTensor::new_unchecked(arr1(&[*l_val]).into_dyn(), arr1(&[*u_val]).into_dyn())
                .expect("new_unchecked should accept Inf bounds");

        let w1 = arr2(&[[1.0]]);
        let linear1 = LinearLayer::new(w1, None).unwrap();
        let mut network = Network::new();
        network.add_layer(Layer::Linear(linear1));
        network.add_layer(Layer::ReLU(ReLULayer));

        let layer_bounds = vec![Arc::new(pre_bounds.clone())];
        let history = SplitHistory::new();
        let beta_state = BetaState::empty();
        let alpha_state =
            DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

        let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
        let output_bounds = LinearBounds::identity(1);

        let bounds = verifier
            .relu_backward_with_alpha_beta(
                &output_bounds,
                &pre_bounds,
                None,
                &beta_state,
                &alpha_state,
                1,
            )
            .unwrap_or_else(|e| panic!("relu_backward failed for {label}: {e:?}"));

        assert_relu_backward_no_nan(&bounds, label);
    }
}

/// Shared assertion helper for f64 bias accumulation regression tests.
///
/// Verifies both soundness (lower/upper bounds bracket f64 reference) and
/// directed rounding discrimination (f64 accumulation + next_down/next_up
/// pushes bounds away from 1e7, which wouldn't happen with f32 accumulators
/// because each sub-ULP term is lost). Reference: #2336, #1745, #3343.
fn assert_f64_bias_accumulation_bounds(bounds: &LinearBounds, n: usize) {
    let expected_lower = 1e7_f64 + (n as f64) * (-1e-4_f64) * 0.5_f64;
    let expected_upper = 1e7_f64 + (n as f64) * 1e-4_f64 * 0.5_f64;

    // Soundness: lower bound must not exceed f64 reference.
    assert!(
        bounds.lower_b()[0] <= expected_lower as f32 + 1e-4,
        "Lower bias unsound: got {}, expected <= {} (f64: {expected_lower})",
        bounds.lower_b()[0],
        expected_lower as f32,
    );
    // Discrimination: f64 + next_down_f32 must push below 1e7.
    assert!(
        bounds.lower_b()[0] < 1e7_f32,
        "Lower bias not pushed below 1e7 by directed rounding: got {} \
         (f64: {expected_lower:.10}). f32 accumulation losing sub-ULP terms.",
        bounds.lower_b()[0],
    );
    // Soundness: upper bound must not be below f64 reference.
    assert!(
        bounds.upper_b()[0] >= expected_upper as f32 - 1e-4,
        "Upper bias unsound: got {}, expected >= {} (f64: {expected_upper})",
        bounds.upper_b()[0],
        expected_upper as f32,
    );
    // Discrimination: f64 + next_up_f32 must push above 1e7.
    assert!(
        bounds.upper_b()[0] > 1e7_f32,
        "Upper bias not pushed above 1e7 by directed rounding: got {} \
         (f64: {expected_upper:.10}). f32 accumulation losing sub-ULP terms.",
        bounds.upper_b()[0],
    );
}

/// Regression test: f64 bias accumulation prevents catastrophic cancellation
/// in relu_backward_with_alpha_beta (#2336, #1745).
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_f64_bias_accumulation_2336() {
    let n = 256;
    let pre_bounds = BoundedTensor::new(
        Array1::from_elem(n, -1.0_f32).into_dyn(),
        Array1::from_elem(n, 1.0_f32).into_dyn(),
    )
    .unwrap();

    // Negative la_ij: lower bound accumulates la_ij * upper_intercept = -1e-4 * 0.5
    // = -5e-5 per neuron. At 1e7 base, each term is below f32 ULP (1.0).
    let output_bounds = LinearBounds::new_or_conservative(
        Array2::from_elem((1, n), -1e-4_f32),
        Array1::from_elem(1, 1e7_f32),
        Array2::from_elem((1, n), 1e-4_f32),
        Array1::from_elem(1, 1e7_f32),
    )
    .unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(Array2::eye(n), None).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));

    let layer_bounds: Vec<Arc<BoundedTensor>> = vec![Arc::new(pre_bounds.clone())];
    let history = SplitHistory::new();
    let beta_state = BetaState::empty();
    let alpha_state =
        DomainAlphaState::from_layer_bounds_and_constraints(&network, &layer_bounds, &history);

    let verifier = BetaCrownVerifier::new(BetaCrownConfig::default());
    let bounds = verifier
        .relu_backward_with_alpha_beta(
            &output_bounds,
            &pre_bounds,
            None,
            &beta_state,
            &alpha_state,
            1,
        )
        .unwrap();

    assert_f64_bias_accumulation_bounds(&bounds, n);
}

/// Shared setup helper: creates the backward scenario for f64 bias tests.
/// Returns (pre_bounds, output_bounds, network) for the given neuron count.
fn setup_f64_bias_accumulation_scenario(n: usize) -> (BoundedTensor, LinearBounds, Network) {
    let pre_bounds = BoundedTensor::new(
        Array1::from_elem(n, -1.0_f32).into_dyn(),
        Array1::from_elem(n, 1.0_f32).into_dyn(),
    )
    .unwrap();
    let output_bounds = LinearBounds::new_or_conservative(
        Array2::from_elem((1, n), -1e-4_f32),
        Array1::from_elem(1, 1e7_f32),
        Array2::from_elem((1, n), 1e-4_f32),
        Array1::from_elem(1, 1e7_f32),
    )
    .unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(Array2::eye(n), None).unwrap(),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    (pre_bounds, output_bounds, network)
}

/// Variant with n=512 (double the base test). Part of #3343.
#[ntest::timeout(5000)]
#[test]
fn test_relu_backward_alpha_beta_f64_bias_accumulation_large_3343() {
    let n = 512;
    let (pre_bounds, output_bounds, network) = setup_f64_bias_accumulation_scenario(n);
    let layer_bounds: Vec<Arc<BoundedTensor>> = vec![Arc::new(pre_bounds.clone())];
    let alpha_state = DomainAlphaState::from_layer_bounds_and_constraints(
        &network,
        &layer_bounds,
        &SplitHistory::new(),
    );
    let bounds = BetaCrownVerifier::new(BetaCrownConfig::default())
        .relu_backward_with_alpha_beta(
            &output_bounds,
            &pre_bounds,
            None,
            &BetaState::empty(),
            &alpha_state,
            1,
        )
        .unwrap();
    assert_f64_bias_accumulation_bounds(&bounds, n);
}
