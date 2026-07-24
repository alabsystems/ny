// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest CROWN soundness tests for Tan, Arctan, and Reciprocal layers.
//!
//! Reciprocal(x) = 1/x is convex for x > 0, concave for x < 0; intervals must
//! not cross zero. Tan(x) has asymptotes at x = pi/2 + k*pi; intervals are
//! constrained to stay within one period. Arctan(x) is S-shaped: concave for
//! x > 0, convex for x < 0, with range (-pi/2, pi/2).
//!
//! Identity-bounds tests verify the relaxation envelope directly. Negative-coeff
//! and asymmetric tests exercise sign-switching in crown_elementwise_backward.
//!
//! Part of #40, #1793.

use crate::layers::{ArctanLayer, ReciprocalLayer, TanLayer};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{assert_crown_backward_sound, sample_points, CROWN_TOLERANCE};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    // =========================================================================
    // Reciprocal CROWN backward soundness (via identity bounds)
    // =========================================================================

    /// Verify Reciprocal CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= 1/x <= CROWN_upper(x)
    ///
    /// Reciprocal(x) = 1/x is convex for x > 0 and concave for x < 0.
    /// The interval must not cross zero.
    ///
    /// Range: positive domain [0.1, 20] and negative domain [-20, -0.1].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reciprocal_crown_positive(l in 0.1f32..10.0, delta in 0.0f32..10.0) {
        let u = l + delta;
        prop_assume!(l > 0.0 && u > l);

        let layer = ReciprocalLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| 1.0 / x,
            &result,
            "Reciprocal(+)",
            CROWN_TOLERANCE,
        )?;
    }

    /// Reciprocal CROWN backward in the negative domain.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_reciprocal_crown_negative(l in -10.0f32..-0.1, delta in 0.0f32..9.9) {
        let u = (l + delta).min(-0.1);
        prop_assume!(l < 0.0 && u < 0.0 && l <= u);

        let layer = ReciprocalLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| 1.0 / x,
            &result,
            "Reciprocal(-)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Reciprocal CROWN backward with negative incoming coefficients
    // =========================================================================

    /// Reciprocal CROWN backward soundness with mixed-sign incoming coefficients.
    /// Pre-activation bounds are strictly positive (avoiding zero).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_reciprocal_crown_negative_coeffs(
        l0 in 0.1f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in 0.1f32..10.0,
        d1 in 0.0f32..10.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = l0 + d0;
        let u1 = l1 + d1;

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let recip_layer = ReciprocalLayer::new();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| recip_layer.propagate_linear_with_bounds(bounds, pre),
            |x| 1.0 / x,
            "Reciprocal",
            CROWN_TOLERANCE,
        )?;
    }

    /// Reciprocal CROWN backward with asymmetric incoming lower/upper coefficients.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_reciprocal_crown_asymmetric_bounds(
        l0 in 0.1f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in 0.1f32..10.0,
        d1 in 0.0f32..10.0,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = l0 + d0;
        let u1 = l1 + d1;

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

        let recip_layer = ReciprocalLayer::new();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| recip_layer.propagate_linear_with_bounds(bounds, pre),
            |x| 1.0 / x,
            "Reciprocal-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    /// Reciprocal CROWN backward with negative incoming coefficients (direct sampling variant).
    /// Narrower range [0.1, 5.0] vs the piecewise helper variant at [0.1, 10.0].
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_reciprocal_crown_negative_coeffs_direct(
        l0 in 0.1f32..5.0,
        d0 in 0.0f32..5.0,
        l1 in 0.1f32..5.0,
        d1 in 0.0f32..5.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = l0 + d0;
        let u1 = l1 + d1;
        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let pre_activation = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let layer = ReciprocalLayer::new();
        let result = layer
            .propagate_linear_with_bounds(&incoming, &pre_activation)
            .unwrap();

        let samples_0 = sample_points(l0, u0, 20);
        let samples_1 = sample_points(l1, u1, 20);

        for &x0 in &samples_0 {
            for &x1 in &samples_1 {
                let fx0 = 1.0 / x0;
                let fx1 = 1.0 / x1;
                let incoming_lower = incoming.lower_a[[0, 0]] * fx0
                    + incoming.lower_a[[0, 1]] * fx1
                    + incoming.lower_b[0];
                let incoming_upper = incoming.upper_a[[0, 0]] * fx0
                    + incoming.upper_a[[0, 1]] * fx1
                    + incoming.upper_b[0];

                let lb = result.lower_a[[0, 0]] * x0 + result.lower_a[[0, 1]] * x1 + result.lower_b[0];
                let ub = result.upper_a[[0, 0]] * x0 + result.upper_a[[0, 1]] * x1 + result.upper_b[0];

                let scale_tol = CROWN_TOLERANCE * incoming_upper.abs().max(incoming_lower.abs()).max(1.0);

                prop_assert!(
                    lb <= incoming_lower + scale_tol,
                    "Reciprocal lower bound violated at ({x0}, {x1}): lb={lb} > incoming_lower={incoming_lower}"
                );
                prop_assert!(
                    ub + scale_tol >= incoming_upper,
                    "Reciprocal upper bound violated at ({x0}, {x1}): ub={ub} < incoming_upper={incoming_upper}"
                );
            }
        }
    }

    // =========================================================================
    // Tan CROWN backward soundness (via identity bounds)
    // =========================================================================

    /// Verify Tan CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= tan(x) <= CROWN_upper(x)
    ///
    /// Tan(x) is periodic with asymptotes at x = pi/2 + k*pi.
    /// Interval must not contain any asymptote.
    ///
    /// Range constrained to (-pi/2 + epsilon, pi/2 - epsilon) to stay within
    /// one period and avoid asymptotes.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tan_crown(l in -1.4f32..1.4, delta in 0.0f32..2.8) {
        let u = (l + delta).min(1.4);
        prop_assume!(l <= u);

        let layer = TanLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation);
        prop_assert!(
            result.is_ok(),
            "Tan CROWN unexpectedly failed on non-asymptotic interval [{l}, {u}]"
        );
        let result = result.unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x.tan(),
            &result,
            "Tan",
            CROWN_TOLERANCE,
        )?;
    }

    /// Tan CROWN backward with negative incoming coefficients.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_tan_crown_negative_coeffs(
        l0 in -1.2f32..1.2,
        d0 in 0.0f32..0.5,
        l1 in -1.2f32..1.2,
        d1 in 0.0f32..0.5,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(1.4);
        let u1 = (l1 + d1).min(1.4);
        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let pre_activation = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let layer = TanLayer::new();
        let result = layer
            .propagate_linear_with_bounds(&incoming, &pre_activation);
        prop_assert!(
            result.is_ok(),
            "Tan CROWN unexpectedly failed on non-asymptotic intervals [{l0}, {u0}] and [{l1}, {u1}]"
        );
        let result = result.unwrap();

        let samples_0 = sample_points(l0, u0, 20);
        let samples_1 = sample_points(l1, u1, 20);

        for &x0 in &samples_0 {
            for &x1 in &samples_1 {
                let fx0 = x0.tan();
                let fx1 = x1.tan();
                let incoming_lower = incoming.lower_a[[0, 0]] * fx0
                    + incoming.lower_a[[0, 1]] * fx1
                    + incoming.lower_b[0];
                let incoming_upper = incoming.upper_a[[0, 0]] * fx0
                    + incoming.upper_a[[0, 1]] * fx1
                    + incoming.upper_b[0];

                let lb = result.lower_a[[0, 0]] * x0 + result.lower_a[[0, 1]] * x1 + result.lower_b[0];
                let ub = result.upper_a[[0, 0]] * x0 + result.upper_a[[0, 1]] * x1 + result.upper_b[0];

                let scale_tol = CROWN_TOLERANCE * incoming_upper.abs().max(incoming_lower.abs()).max(1.0);

                prop_assert!(
                    lb <= incoming_lower + scale_tol,
                    "Tan lower bound violated at ({x0}, {x1}): lb={lb} > incoming_lower={incoming_lower}"
                );
                prop_assert!(
                    ub + scale_tol >= incoming_upper,
                    "Tan upper bound violated at ({x0}, {x1}): ub={ub} < incoming_upper={incoming_upper}"
                );
            }
        }
    }

    /// Tan CROWN backward with asymmetric lower_a/upper_a bounds.
    /// Interval constrained to avoid asymptotes at +/- pi/2.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_tan_crown_asymmetric_bounds(
        l0 in -1.2f32..1.2,
        d0 in 0.0f32..0.5,
        l1 in -1.2f32..1.2,
        d1 in 0.0f32..0.5,
        lower_c0 in -5.0f32..5.0,
        lower_c1 in -5.0f32..5.0,
        delta_c0 in 0.0f32..5.0,
        delta_c1 in 0.0f32..5.0,
        lower_b in -2.0f32..2.0,
        delta_b in 0.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(1.4);
        let u1 = (l1 + d1).min(1.4);

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

        let layer = TanLayer::new();
        let pre_activation = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let result = layer.propagate_linear_with_bounds(&incoming, &pre_activation);
        // Tan legitimately rejects intervals near asymptotes — use prop_assume to
        // discard these cases rather than silently passing.
        prop_assume!(result.is_ok(), "Tan rejected interval near asymptote");
        let result = result.unwrap();

        let samples_0 = sample_points(l0, u0, 20);
        let samples_1 = sample_points(l1, u1, 20);

        for &x0 in &samples_0 {
            for &x1 in &samples_1 {
                let fx0 = x0.tan();
                let fx1 = x1.tan();

                // Skip if tan produces extreme values near asymptotes
                if !fx0.is_finite() || !fx1.is_finite() || fx0.abs() > 1e6 || fx1.abs() > 1e6 {
                    continue;
                }

                let incoming_lower = incoming.lower_a[[0, 0]] * fx0
                    + incoming.lower_a[[0, 1]] * fx1
                    + incoming.lower_b[0];
                let incoming_upper = incoming.upper_a[[0, 0]] * fx0
                    + incoming.upper_a[[0, 1]] * fx1
                    + incoming.upper_b[0];

                let lb = result.lower_a[[0, 0]] * x0 + result.lower_a[[0, 1]] * x1 + result.lower_b[0];
                let ub = result.upper_a[[0, 0]] * x0 + result.upper_a[[0, 1]] * x1 + result.upper_b[0];

                let scale_tol = CROWN_TOLERANCE * incoming_upper.abs().max(incoming_lower.abs()).max(1.0);

                prop_assert!(
                    lb <= incoming_lower + scale_tol,
                    "Tan-asymmetric lower violated at ({x0}, {x1}): lb={lb} > incoming_lower={incoming_lower}"
                );
                prop_assert!(
                    ub + scale_tol >= incoming_upper,
                    "Tan-asymmetric upper violated at ({x0}, {x1}): ub={ub} < incoming_upper={incoming_upper}"
                );
            }
        }
    }

    // =========================================================================
    // Arctan CROWN backward soundness (identity bounds)
    // =========================================================================

    /// Arctan(x) = atan(x). Monotonically increasing, range (-pi/2, pi/2).
    /// S-shaped: concave for x > 0, convex for x < 0.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_arctan_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = l + delta;
        prop_assume!(l <= u);

        let layer = ArctanLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x.atan(),
            &result,
            "Arctan",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Arctan CROWN backward with negative incoming coefficients
    // =========================================================================

    /// Arctan CROWN backward soundness with mixed-sign incoming coefficients.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_arctan_crown_negative_coeffs(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = l0 + d0;
        let u1 = l1 + d1;
        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = ArctanLayer::new();
        let pre_activation = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&incoming, &pre_activation)
            .unwrap();

        let samples_0 = sample_points(l0, u0, 20);
        let samples_1 = sample_points(l1, u1, 20);

        for &x0 in &samples_0 {
            for &x1 in &samples_1 {
                let fx0 = x0.atan();
                let fx1 = x1.atan();
                let incoming_lower = incoming.lower_a[[0, 0]] * fx0
                    + incoming.lower_a[[0, 1]] * fx1
                    + incoming.lower_b[0];
                let incoming_upper = incoming.upper_a[[0, 0]] * fx0
                    + incoming.upper_a[[0, 1]] * fx1
                    + incoming.upper_b[0];

                let lb = result.lower_a[[0, 0]] * x0 + result.lower_a[[0, 1]] * x1 + result.lower_b[0];
                let ub = result.upper_a[[0, 0]] * x0 + result.upper_a[[0, 1]] * x1 + result.upper_b[0];

                let scale_tol = CROWN_TOLERANCE * incoming_upper.abs().max(incoming_lower.abs()).max(1.0);

                prop_assert!(
                    lb <= incoming_lower + scale_tol,
                    "Arctan lower bound violated at ({x0}, {x1}): lb={lb} > incoming_lower={incoming_lower}"
                );
                prop_assert!(
                    ub + scale_tol >= incoming_upper,
                    "Arctan upper bound violated at ({x0}, {x1}): ub={ub} < incoming_upper={incoming_upper}"
                );
            }
        }
    }

    // =========================================================================
    // Arctan CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// Arctan CROWN backward with asymmetric lower_a/upper_a bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_arctan_crown_asymmetric_bounds(
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
        let u0 = l0 + d0;
        let u1 = l1 + d1;

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

        let layer = ArctanLayer::new();
        let pre_activation = BoundedTensor::new(
            arr1(&[l0, l1]).into_dyn(),
            arr1(&[u0, u1]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&incoming, &pre_activation)
            .unwrap();

        let samples_0 = sample_points(l0, u0, 20);
        let samples_1 = sample_points(l1, u1, 20);

        for &x0 in &samples_0 {
            for &x1 in &samples_1 {
                let fx0 = x0.atan();
                let fx1 = x1.atan();
                let incoming_lower = incoming.lower_a[[0, 0]] * fx0
                    + incoming.lower_a[[0, 1]] * fx1
                    + incoming.lower_b[0];
                let incoming_upper = incoming.upper_a[[0, 0]] * fx0
                    + incoming.upper_a[[0, 1]] * fx1
                    + incoming.upper_b[0];

                let lb = result.lower_a[[0, 0]] * x0 + result.lower_a[[0, 1]] * x1 + result.lower_b[0];
                let ub = result.upper_a[[0, 0]] * x0 + result.upper_a[[0, 1]] * x1 + result.upper_b[0];

                let scale_tol = CROWN_TOLERANCE * incoming_upper.abs().max(incoming_lower.abs()).max(1.0);

                prop_assert!(
                    lb <= incoming_lower + scale_tol,
                    "Arctan-asymmetric lower violated at ({x0}, {x1}): lb={lb} > incoming_lower={incoming_lower}"
                );
                prop_assert!(
                    ub + scale_tol >= incoming_upper,
                    "Arctan-asymmetric upper violated at ({x0}, {x1}): ub={ub} < incoming_upper={incoming_upper}"
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn regression_reciprocal_near_zero_positive_upper_endpoint_sound() {
    // Counterexample family from #2225 prover audit: near-zero positive interval.
    let l = 2.333_718_9e-7_f32;
    let u = 2.339_029_7e-7_f32;

    let layer = ReciprocalLayer::new();
    let identity = LinearBounds::identity(1);
    let pre_activation = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
    let result = layer
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Secant upper bound should stay above 1/x at the interval endpoint.
    let x = f64::from(u);
    let fx = 1.0_f64 / x;
    let upper = f64::from(result.upper_a[[0, 0]]) * x + f64::from(result.upper_b[0]);

    assert!(
        upper >= fx,
        "Reciprocal near-zero (+) upper bound unsound at x={x}: upper={upper} < true={fx}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn regression_reciprocal_near_zero_negative_lower_endpoint_sound() {
    // Counterexample family from #2225 prover audit: near-zero negative interval.
    let l = -1.632_887_7e-6_f32;
    let u = -1.630_305_7e-6_f32;

    let layer = ReciprocalLayer::new();
    let identity = LinearBounds::identity(1);
    let pre_activation = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
    let result = layer
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .unwrap();

    // Secant lower bound should stay below 1/x at the interval endpoint.
    let x = f64::from(u);
    let fx = 1.0_f64 / x;
    let lower = f64::from(result.lower_a[[0, 0]]) * x + f64::from(result.lower_b[0]);

    assert!(
        lower <= fx,
        "Reciprocal near-zero (-) lower bound unsound at x={x}: lower={lower} > true={fx}"
    );
}
