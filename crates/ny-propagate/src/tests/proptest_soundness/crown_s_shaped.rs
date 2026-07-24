// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest CROWN soundness tests for S-shaped activations: SiLU, Sigmoid,
//! Tanh, GELU (tanh and erf approximations).
//!
//! These activations are S-shaped (convex for x < 0, concave for x > 0, or
//! similar inflection structure). The identity-bounds tests verify the
//! relaxation envelope directly. The negative-coeff and asymmetric tests
//! exercise sign-switching in crown_elementwise_backward.
//!
//! Part of #40, #1793.

use crate::layers::{SiLULayer, SigmoidLayer, TanhLayer};
use crate::{GELULayer, GeluApproximation, LinearBounds};
use ndarray::{arr1, Array1, Array2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{
    assert_crown_backward_sound, assert_crown_negative_coeff_sound, assert_relaxation_envelope,
    gelu_erf_eval, gelu_tanh_eval, sigmoid_eval, silu_eval, tanh_eval, CROWN_TOLERANCE,
};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    // =========================================================================
    // SiLU CROWN relaxation soundness (identity bounds)
    // =========================================================================

    /// Verify SiLU CROWN relaxation envelope via standalone relaxation function.
    /// For all x in [l, u]:
    ///   lower_slope * x + lower_intercept <= SiLU(x) <= upper_slope * x + upper_intercept
    ///
    /// SiLU(x) = x * sigmoid(x) is neither convex nor concave globally.
    /// Uses chord + tangent bounds based on convexity regions.
    ///
    /// Range [-10, 10]: SiLU is well-behaved everywhere, bounded by [-0.28, +inf).
    /// Reference: designs/2026-02-08-silu-crown-relaxation.md
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_silu_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);
        assert_relaxation_envelope(
            l, u,
            silu_eval,
            crate::layers::silu_sound_linear_relaxation,
            "SiLU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Sigmoid CROWN backward soundness (identity bounds)
    // =========================================================================

    /// Verify Sigmoid CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= sigmoid(x) <= CROWN_upper(x)
    ///
    /// Sigmoid is S-shaped: convex for x < 0, concave for x > 0.
    /// The relaxation uses tangent bounds in each region and chord/tangent
    /// across the inflection point.
    ///
    /// Range [-10, 10]: sigmoid saturates beyond this range.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sigmoid_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let sigmoid_layer = SigmoidLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = sigmoid_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            sigmoid_eval,
            &result,
            "Sigmoid",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Tanh CROWN backward soundness (identity bounds)
    // =========================================================================

    /// Verify Tanh CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= tanh(x) <= CROWN_upper(x)
    ///
    /// Tanh is S-shaped: convex for x < 0, concave for x > 0, output in (-1, 1).
    /// Range [-10, 10]: tanh saturates beyond this range.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tanh_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let tanh_layer = TanhLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = tanh_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            tanh_eval,
            &result,
            "Tanh",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // GELU (tanh) CROWN backward soundness (identity bounds)
    // =========================================================================

    /// Verify GELU (tanh approximation) CROWN backward produces sound bounds.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= GELU_tanh(x) <= CROWN_upper(x)
    ///
    /// GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))).
    /// Range [-10, 10].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gelu_tanh_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let gelu_layer = GELULayer::new(GeluApproximation::Tanh);
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = gelu_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            gelu_tanh_eval,
            &result,
            "GELU-tanh",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // GELU (erf) CROWN backward soundness (identity bounds)
    // =========================================================================

    /// Verify GELU (exact erf) CROWN backward produces sound bounds.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= GELU_erf(x) <= CROWN_upper(x)
    ///
    /// GELU_erf(x) = 0.5 * x * (1 + erf(x/sqrt(2))).
    /// Range [-10, 10].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gelu_erf_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let gelu_layer = GELULayer::new(GeluApproximation::Erf);
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = gelu_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            gelu_erf_eval,
            &result,
            "GELU-erf",
            CROWN_TOLERANCE,
        )?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // =========================================================================
    // SiLU CROWN backward with negative coefficients
    // =========================================================================

    /// SiLU CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// Exercises the sign-switching branches in crown_elementwise_backward
    /// for SiLU(x) = x * sigmoid(x). SiLU is neither convex nor concave
    /// globally, so the relaxation is nontrivial.
    ///
    /// Range [-10, 10]: SiLU is well-behaved everywhere, bounded by [-0.28, +inf).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_silu_crown_negative_coeffs(
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

        let silu_layer = SiLULayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| silu_layer.propagate_linear_with_bounds(bounds, pre),
            silu_eval,
            "SiLU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Sigmoid CROWN backward with negative coefficients
    // =========================================================================

    /// Sigmoid CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// Sigmoid(x) = 1/(1+exp(-x)) is an S-shaped function with inflection at x=0.
    /// It's convex for x < 0 and concave for x > 0, requiring different relaxation
    /// strategies in each region.
    ///
    /// Range [-10, 10]: sigmoid is effectively 0 below -10 and 1 above 10.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_sigmoid_crown_negative_coeffs(
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

        let sigmoid_layer = SigmoidLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| sigmoid_layer.propagate_linear_with_bounds(bounds, pre),
            sigmoid_eval,
            "Sigmoid",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Tanh CROWN backward with negative coefficients
    // =========================================================================

    /// Tanh CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// Tanh(x) = 2*sigmoid(2x) - 1 has the same convexity structure as sigmoid
    /// (inflection at x=0). Output range is (-1, 1). The relaxation is tighter
    /// than sigmoid's because of the smaller output range.
    ///
    /// Range [-8, 8]: tanh saturates well before +/-8.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_tanh_crown_negative_coeffs(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..8.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..8.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(8.0);
        let u1 = (l1 + d1).min(8.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let tanh_layer = TanhLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| tanh_layer.propagate_linear_with_bounds(bounds, pre),
            tanh_eval,
            "Tanh",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // GELU (tanh) CROWN backward with negative coefficients
    // =========================================================================

    /// GELU (tanh approx) CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) has a
    /// critical point near x = -0.68 where the function reaches its minimum.
    /// The relaxation must handle the non-monotone region correctly.
    ///
    /// Range [-8, 8]: GELU is well-behaved in this range.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_gelu_tanh_crown_negative_coeffs(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..8.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..8.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(8.0);
        let u1 = (l1 + d1).min(8.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let gelu_layer = GELULayer::new(GeluApproximation::Tanh);
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| gelu_layer.propagate_linear_with_bounds(bounds, pre),
            gelu_tanh_eval,
            "GELU-tanh",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // GELU (erf) CROWN backward with negative coefficients
    // =========================================================================

    /// GELU (erf) CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// Tests the exact erf-based GELU approximation, which uses a different
    /// relaxation implementation path than the tanh approximation.
    ///
    /// Range [-8, 8]: GELU is well-behaved in this range.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_gelu_erf_crown_negative_coeffs(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..8.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..8.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(8.0);
        let u1 = (l1 + d1).min(8.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let gelu_layer = GELULayer::new(GeluApproximation::Erf);
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| gelu_layer.propagate_linear_with_bounds(bounds, pre),
            gelu_erf_eval,
            "GELU-erf",
            CROWN_TOLERANCE,
        )?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // =========================================================================
    // SiLU CROWN backward with asymmetric bounds
    // =========================================================================

    /// SiLU CROWN backward with asymmetric lower_a/upper_a.
    ///
    /// Tests the case where incoming lower and upper coefficient matrices differ,
    /// which occurs after passing through nonlinear layers in multi-layer CROWN.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_silu_crown_asymmetric_bounds(
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

        let silu_layer = SiLULayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| silu_layer.propagate_linear_with_bounds(bounds, pre),
            silu_eval,
            "SiLU-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Sigmoid CROWN backward with asymmetric bounds
    // =========================================================================

    /// Sigmoid CROWN backward with asymmetric lower_a/upper_a.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_sigmoid_crown_asymmetric_bounds(
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

        let sigmoid_layer = SigmoidLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| sigmoid_layer.propagate_linear_with_bounds(bounds, pre),
            sigmoid_eval,
            "Sigmoid-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Tanh CROWN backward with asymmetric bounds
    // =========================================================================

    /// Tanh CROWN backward with asymmetric lower_a/upper_a.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_tanh_crown_asymmetric_bounds(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..8.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..8.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(8.0);
        let u1 = (l1 + d1).min(8.0);

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

        let tanh_layer = TanhLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| tanh_layer.propagate_linear_with_bounds(bounds, pre),
            tanh_eval,
            "Tanh-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // GELU (tanh) CROWN backward with asymmetric bounds
    // =========================================================================

    /// GELU (tanh approx) CROWN backward with asymmetric lower_a/upper_a.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_gelu_tanh_crown_asymmetric_bounds(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..8.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..8.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(8.0);
        let u1 = (l1 + d1).min(8.0);

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

        let gelu_layer = GELULayer::new(GeluApproximation::Tanh);
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| gelu_layer.propagate_linear_with_bounds(bounds, pre),
            gelu_tanh_eval,
            "GELU-tanh-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // GELU (erf) CROWN backward with asymmetric bounds
    // =========================================================================

    /// GELU (erf) CROWN backward with asymmetric lower_a/upper_a.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_gelu_erf_crown_asymmetric_bounds(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..8.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..8.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(8.0);
        let u1 = (l1 + d1).min(8.0);

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

        let gelu_layer = GELULayer::new(GeluApproximation::Erf);
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| gelu_layer.propagate_linear_with_bounds(bounds, pre),
            gelu_erf_eval,
            "GELU-erf-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }
}
