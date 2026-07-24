// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest CROWN soundness tests for trigonometric layers: Sin, Cos.
//!
//! Sin and Cos use trigonometric relaxation via tangent/secant bounds within
//! single-concavity regions. Intervals that span inflection points fall back
//! to constant [-1, 1] bounds. The identity-bounds tests constrain intervals
//! to stay within a single half-period to exercise the tangent/secant logic.
//!
//! Part of #40, #1708, #1793.

use crate::layers::{CosLayer, SinLayer};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{assert_crown_backward_sound, assert_crown_negative_coeff_sound, CROWN_TOLERANCE};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    // =========================================================================
    // Sin CROWN backward soundness (identity bounds)
    // =========================================================================

    /// Verify Sin CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= sin(x) <= CROWN_upper(x)
    ///
    /// Constrained to intervals < π within a single concavity region to exercise
    /// the tangent/secant relaxation rather than the constant-bound fallback.
    /// sin is concave on [0, π] and convex on [π, 2π] (mod 2π).
    ///
    /// Reference: periodic.rs trig_tangent_secant_relaxation
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sin_crown(center in -10.0f32..10.0, width in 0.0f32..2.5) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let sin_layer = SinLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = sin_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x.sin(),
            &result,
            "Sin",
            CROWN_TOLERANCE,
        )?;
    }

    /// Sin CROWN backward soundness with mixed-sign incoming coefficients.
    /// Exercises the sign-switching branches for sin(x) which is neither
    /// globally convex nor concave.
    ///
    /// Range constrained to [-8, 8] with intervals < 2.5 to keep within
    /// reasonable relaxation precision.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_sin_crown_negative_coeffs(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..2.5,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..2.5,
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

        let sin_layer = SinLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| sin_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.sin(),
            "Sin",
            CROWN_TOLERANCE,
        )?;
    }

    /// Sin CROWN backward with asymmetric lower_a/upper_a.
    /// Tests the case where incoming lower and upper coefficient matrices differ.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_sin_crown_asymmetric_bounds(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..2.5,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..2.5,
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

        let sin_layer = SinLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| sin_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.sin(),
            "Sin-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Cos CROWN backward soundness (identity bounds)
    // =========================================================================

    /// Verify Cos CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= cos(x) <= CROWN_upper(x)
    ///
    /// Constrained to intervals < π/2 within a single concavity region.
    /// cos is concave on [-π/2, π/2] and convex on [π/2, 3π/2] (mod 2π).
    ///
    /// Reference: periodic.rs cos_linear_relaxation, trig_tangent_secant_relaxation
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cos_crown(center in -10.0f32..10.0, width in 0.0f32..2.5) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let cos_layer = CosLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = cos_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x.cos(),
            &result,
            "Cos",
            CROWN_TOLERANCE,
        )?;
    }

    /// Cos CROWN backward soundness with mixed-sign incoming coefficients.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_cos_crown_negative_coeffs(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..2.5,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..2.5,
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

        let cos_layer = CosLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| cos_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.cos(),
            "Cos",
            CROWN_TOLERANCE,
        )?;
    }

    /// Cos CROWN backward with asymmetric lower_a/upper_a.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_cos_crown_asymmetric_bounds(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..2.5,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..2.5,
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

        let cos_layer = CosLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| cos_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.cos(),
            "Cos-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }
}
