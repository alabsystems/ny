// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest CROWN soundness tests for ELU-family activations: ELU, CELU,
//! SELU, Mish, Softplus.
//!
//! These activations share a common structure: they behave like identity
//! for x > 0 and have exponential-based negative tails. The identity-bounds
//! tests verify the relaxation envelope directly. The negative-coeff and
//! asymmetric tests exercise sign-switching in crown_elementwise_backward.
//!
//! Part of #40, #1793.

use crate::layers::{CeluLayer, EluLayer, MishLayer};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2};
use ny_core::NyError;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{
    assert_crown_backward_sound, assert_crown_negative_coeff_sound, assert_relaxation_envelope,
    mish_eval, softplus_eval, CROWN_TOLERANCE, SELU_ALPHA, SELU_LAMBDA,
};

// =============================================================================
// IDENTITY-BOUNDS TESTS (500 cases)
// =============================================================================
// These tests use identity incoming bounds (A = I, b = 0) to verify the
// relaxation envelope directly for each activation.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    // =========================================================================
    // ELU CROWN relaxation soundness
    // =========================================================================

    /// Verify ELU linear relaxation envelope: for all x in [l, u],
    ///   lower_slope * x + lower_intercept <= ELU(x) <= upper_slope * x + upper_intercept
    ///
    /// ELU(x) = x if x >= 0, alpha*(exp(x)-1) if x < 0.
    /// Range [-10, 10], alpha in [0.5, 2.0].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_elu_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0, alpha in 0.5f32..2.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);
        assert_relaxation_envelope(
            l, u,
            |x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) },
            |l, u| crate::layers::activations::elu::elu_linear_relaxation(l, u, alpha),
            "ELU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // CELU CROWN relaxation soundness
    // =========================================================================

    /// Verify CELU linear relaxation envelope: for all x in [l, u],
    ///   lower_slope * x + lower_intercept <= CELU(x) <= upper_slope * x + upper_intercept
    ///
    /// CELU(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1)).
    /// Range [-10, 10], alpha in [0.5, 2.0].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_celu_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0, alpha in 0.5f32..2.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);
        assert_relaxation_envelope(
            l, u,
            |x| if x >= 0.0 { x } else { alpha * ((x / alpha).exp() - 1.0) },
            |l, u| crate::layers::activations::celu::celu_linear_relaxation(l, u, alpha),
            "CELU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // SELU CROWN backward soundness (via identity bounds)
    // =========================================================================

    /// Verify SELU CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= SELU(x) <= CROWN_upper(x)
    ///
    /// SELU(x) = lambda * (x if x >= 0, alpha*(exp(x)-1) if x < 0).
    /// Range [-10, 10].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_selu_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let selu_layer = crate::layers::SeluLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = selu_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| if x >= 0.0 { SELU_LAMBDA * x } else { SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0) },
            &result,
            "SELU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Mish CROWN backward soundness (via identity bounds)
    // =========================================================================

    /// Verify Mish CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= Mish(x) <= CROWN_upper(x)
    ///
    /// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x))).
    /// Range [-10, 10].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_mish_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let mish_layer = MishLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = mish_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            mish_eval,
            &result,
            "Mish",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Softplus CROWN backward soundness (via identity bounds)
    // =========================================================================

    /// Verify Softplus CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= Softplus(x) <= CROWN_upper(x)
    ///
    /// Softplus(x) = ln(1 + exp(x)). Strictly convex.
    /// Range [-20, 20].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_softplus_crown(l in -20.0f32..20.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let softplus_layer = crate::layers::SoftplusLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = softplus_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            softplus_eval,
            &result,
            "Softplus",
            CROWN_TOLERANCE,
        )?;
    }
}

/// Regression for #1803 self-audit: Mish [l, +inf) must use a globally sound
/// lower envelope. Tangent-at-l is unsound for non-convex Mish.
#[test]
fn regression_mish_infinite_upper_lower_bound_sound() {
    // Updated for #2977: domain_guard rejects non-finite pre-activation.
    let mish_layer = MishLayer::new();
    let identity = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new_unchecked(arr1(&[1.0]).into_dyn(), arr1(&[f32::INFINITY]).into_dyn())
            .unwrap();

    let result = mish_layer.propagate_linear_with_bounds(&identity, &pre_activation);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "Mish with u=+inf should trigger domain_guard: got {:?}",
        result
    );
}

/// Regression for #1803/#2977: ELU with infinite upper now rejected by domain_guard.
#[test]
fn regression_elu_infinite_upper_alpha_gt_one_lower_bound_sound() {
    let alpha = 2.0_f32;
    let elu_layer = EluLayer::new(alpha);
    let identity = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new_unchecked(arr1(&[-0.1]).into_dyn(), arr1(&[f32::INFINITY]).into_dyn())
            .unwrap();

    let result = elu_layer.propagate_linear_with_bounds(&identity, &pre_activation);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "ELU with u=+inf should trigger domain_guard: got {:?}",
        result
    );
}

/// Regression for #1803/#2977: SELU with infinite upper now rejected by domain_guard.
#[test]
fn regression_selu_infinite_upper_lower_bound_sound() {
    let selu_layer = crate::layers::SeluLayer::new();
    let identity = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new_unchecked(arr1(&[-0.2]).into_dyn(), arr1(&[f32::INFINITY]).into_dyn())
            .unwrap();

    let result = selu_layer.propagate_linear_with_bounds(&identity, &pre_activation);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "SELU with u=+inf should trigger domain_guard: got {:?}",
        result
    );
}

// =============================================================================
// NEGATIVE-COEFFICIENT TESTS (300 cases)
// =============================================================================
// These tests use 2-neuron inputs with at least one negative incoming
// coefficient, exercising sign-switching logic in crown_elementwise_backward.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // =========================================================================
    // ELU CROWN backward with negative coefficients
    // =========================================================================

    /// ELU CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// ELU(x) = x if x >= 0, alpha*(exp(x)-1) if x < 0.
    /// Range [-10, 10], alpha in [0.5, 2.0].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_elu_crown_negative_coeffs(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
        alpha in 0.5f32..2.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let elu_layer = EluLayer::new(alpha);
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| elu_layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) },
            "ELU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // CELU CROWN backward with negative coefficients
    // =========================================================================

    /// CELU CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// CELU(x) = max(0,x) + min(0, alpha*(exp(x/alpha)-1)).
    /// Range [-10, 10], alpha in [0.5, 2.0].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_celu_crown_negative_coeffs(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
        alpha in 0.5f32..2.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let celu_layer = CeluLayer::new(alpha);
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| celu_layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { x } else { alpha * ((x / alpha).exp() - 1.0) },
            "CELU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // SELU CROWN backward with negative coefficients
    // =========================================================================

    /// SELU CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// SELU(x) = lambda * (x if x >= 0, alpha*(exp(x)-1) if x < 0).
    /// Range [-10, 10].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_selu_crown_negative_coeffs(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let selu_layer = crate::layers::SeluLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| selu_layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { SELU_LAMBDA * x } else { SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0) },
            "SELU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Mish CROWN backward with negative coefficients
    // =========================================================================

    /// Mish CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// Mish(x) = x * tanh(softplus(x)). Neither convex nor concave globally.
    /// Range [-10, 10].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_mish_crown_negative_coeffs(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let mish_layer = MishLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| mish_layer.propagate_linear_with_bounds(bounds, pre),
            mish_eval,
            "Mish",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Softplus CROWN backward with negative coefficients
    // =========================================================================

    /// Softplus CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// Softplus(x) = ln(1 + exp(x)). Strictly convex.
    /// Range [-20, 20].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_softplus_crown_negative_coeffs(
        l0 in -20.0f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in -20.0f32..20.0,
        d1 in 0.0f32..20.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(20.0);
        let u1 = (l1 + d1).min(20.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let softplus_layer = crate::layers::SoftplusLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| softplus_layer.propagate_linear_with_bounds(bounds, pre),
            softplus_eval,
            "Softplus",
            CROWN_TOLERANCE,
        )?;
    }
}

// =============================================================================
// ASYMMETRIC BOUND TESTS (300 cases)
// =============================================================================
// These tests exercise the case where incoming lower and upper coefficient
// matrices differ (lower_a != upper_a), which stresses sign-switching logic
// more than symmetric bounds. Completes the 3-test coverage pattern for
// ELU, CELU, SELU, Mish, and Softplus.
//
// Part of #40, #1793.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // =========================================================================
    // ELU CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// ELU CROWN backward with asymmetric lower_a/upper_a.
    /// Uses alpha=1.0 (globally convex case).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_elu_crown_asymmetric_bounds(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        let upper_c0 = lower_c0 + delta_c0;
        let upper_c1 = lower_c1 + delta_c1;
        let upper_b = lower_b + delta_b;

        prop_assume!(
            lower_c0.abs() > 0.01
                || lower_c1.abs() > 0.01
                || upper_c0.abs() > 0.01
                || upper_c1.abs() > 0.01
        );
        prop_assume!(delta_c0 > 0.01 || delta_c1 > 0.01 || delta_b > 0.01);
        prop_assume!(
            lower_c0 < -0.01 || lower_c1 < -0.01 || upper_c0 < -0.01 || upper_c1 < -0.01
        );

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        // Use alpha=1.0 for asymmetric test (the globally-convex case)
        let elu_layer = EluLayer::new(1.0);
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| elu_layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { x } else { x.exp() - 1.0 },
            "ELU-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // CELU CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// CELU CROWN backward with asymmetric lower_a/upper_a.
    /// Uses alpha=1.0 (globally convex case).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_celu_crown_asymmetric_bounds(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        let upper_c0 = lower_c0 + delta_c0;
        let upper_c1 = lower_c1 + delta_c1;
        let upper_b = lower_b + delta_b;

        prop_assume!(
            lower_c0.abs() > 0.01
                || lower_c1.abs() > 0.01
                || upper_c0.abs() > 0.01
                || upper_c1.abs() > 0.01
        );
        prop_assume!(delta_c0 > 0.01 || delta_c1 > 0.01 || delta_b > 0.01);
        prop_assume!(
            lower_c0 < -0.01 || lower_c1 < -0.01 || upper_c0 < -0.01 || upper_c1 < -0.01
        );

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        // Use alpha=1.0 for asymmetric test
        let celu_layer = CeluLayer::new(1.0);
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| celu_layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { x } else { x.exp() - 1.0 },
            "CELU-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // SELU CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// SELU CROWN backward soundness with asymmetric incoming lower/upper bounds.
    ///
    /// SELU(x) = lambda * (x if x >= 0, alpha*(exp(x)-1) if x < 0).
    /// Range [-10, 10].
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_selu_crown_asymmetric_bounds(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        let upper_c0 = lower_c0 + delta_c0;
        let upper_c1 = lower_c1 + delta_c1;
        let upper_b = lower_b + delta_b;

        prop_assume!(
            lower_c0.abs() > 0.01
                || lower_c1.abs() > 0.01
                || upper_c0.abs() > 0.01
                || upper_c1.abs() > 0.01
        );
        prop_assume!(delta_c0 > 0.01 || delta_c1 > 0.01 || delta_b > 0.01);
        prop_assume!(
            lower_c0 < -0.01 || lower_c1 < -0.01 || upper_c0 < -0.01 || upper_c1 < -0.01
        );

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        let selu_layer = crate::layers::SeluLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| selu_layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { SELU_LAMBDA * x } else { SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0) },
            "SELU-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Mish CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// Mish CROWN backward soundness with asymmetric incoming lower/upper bounds.
    ///
    /// Mish(x) = x * tanh(softplus(x)). Neither convex nor concave globally.
    /// Range [-10, 10].
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_mish_crown_asymmetric_bounds(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        let upper_c0 = lower_c0 + delta_c0;
        let upper_c1 = lower_c1 + delta_c1;
        let upper_b = lower_b + delta_b;

        prop_assume!(
            lower_c0.abs() > 0.01
                || lower_c1.abs() > 0.01
                || upper_c0.abs() > 0.01
                || upper_c1.abs() > 0.01
        );
        prop_assume!(delta_c0 > 0.01 || delta_c1 > 0.01 || delta_b > 0.01);
        prop_assume!(
            lower_c0 < -0.01 || lower_c1 < -0.01 || upper_c0 < -0.01 || upper_c1 < -0.01
        );

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        let mish_layer = MishLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| mish_layer.propagate_linear_with_bounds(bounds, pre),
            mish_eval,
            "Mish-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Softplus CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// Softplus CROWN backward soundness with asymmetric incoming lower/upper bounds.
    ///
    /// Softplus(x) = ln(1 + exp(x)). Strictly convex.
    /// Range [-20, 20].
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_softplus_crown_asymmetric_bounds(
        l0 in -20.0f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in -20.0f32..20.0,
        d1 in 0.0f32..20.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(20.0);
        let u1 = (l1 + d1).min(20.0);

        let upper_c0 = lower_c0 + delta_c0;
        let upper_c1 = lower_c1 + delta_c1;
        let upper_b = lower_b + delta_b;

        prop_assume!(
            lower_c0.abs() > 0.01
                || lower_c1.abs() > 0.01
                || upper_c0.abs() > 0.01
                || upper_c1.abs() > 0.01
        );
        prop_assume!(delta_c0 > 0.01 || delta_c1 > 0.01 || delta_b > 0.01);
        prop_assume!(
            lower_c0 < -0.01 || lower_c1 < -0.01 || upper_c0 < -0.01 || upper_c1 < -0.01
        );

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        let softplus_layer = crate::layers::SoftplusLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| softplus_layer.propagate_linear_with_bounds(bounds, pre),
            softplus_eval,
            "Softplus-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }
}
