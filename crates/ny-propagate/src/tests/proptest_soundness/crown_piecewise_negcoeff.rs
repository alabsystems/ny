// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN negative-coefficient soundness tests for piecewise layers.
//!
//! Split from `crown_piecewise.rs` to keep files under 1000 lines.
//! Part of #40, #1793.
//!
//! These tests exercise CROWN backward propagation with non-identity incoming
//! bounds -- specifically, incoming coefficient matrices with at least one
//! negative entry. This stresses the sign-switching logic in backward propagation.
//!
//! Asymmetric-bound tests (lower_a != upper_a) are in
//! [`crown_piecewise_asymmetric`](super::crown_piecewise_asymmetric).
//!
//! ## Layers covered
//!
//! **Negative-coefficient tests:** ReLU, LeakyReLU, PReLU, HardSigmoid, Clip,
//! Shrink (identity, biased, negative-coeffs, biased-negative-coeffs),
//! ThresholdedReLU (identity, l≈alpha boundary, negative-coeffs),
//! Softsign, HardSwish.

use crate::layers::{
    ClipLayer, HardSigmoidLayer, HardSwishLayer, LeakyReLULayer, PReluLayer, ReLULayer,
    ShrinkLayer, SoftsignLayer, ThresholdedReluLayer,
};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2};
use ny_core::NyError;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use crate::layers::common::BoundPropagation;

use super::{hardswish_eval, sample_points, CROWN_TOLERANCE, FP_TOLERANCE};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // =========================================================================
    // ReLU CROWN backward with negative coefficients
    // =========================================================================

    /// ReLU CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// Exercises the sign-switching branches in ReLU's custom backward
    /// (relu.rs:122-153): when la < 0, the lower bound uses the upper
    /// relaxation (lambda * x + intercept) instead of the lower relaxation
    /// (alpha * x).
    ///
    /// Range [-10, 10]: ReLU is well-defined everywhere.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_relu_crown_negative_coeffs(
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

        let relu_layer = ReLULayer::new();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| relu_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.max(0.0),
            "ReLU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // LeakyReLU CROWN backward with negative coefficients
    // =========================================================================

    /// LeakyReLU CROWN backward soundness with mixed-sign incoming coefficients.
    ///
    /// LeakyReLU(x) = x if x >= 0, alpha*x if x < 0. The relaxation is
    /// similar to ReLU but with nonzero slope in the negative region, making
    /// the crossing-region relaxation tighter.
    ///
    /// Range [-10, 10], alpha in (-3, 3) excluding near-zero.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_leaky_relu_crown_negative_coeffs(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
        alpha in -3.0f32..3.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);
        prop_assume!(alpha.abs() >= 0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = LeakyReLULayer::new(alpha);
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { x } else { alpha * x },
            "LeakyReLU",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // PReLU CROWN backward with negative incoming coefficients
    // =========================================================================

    /// PReLU CROWN backward soundness with mixed-sign incoming coefficients.
    /// Exercises the sign-switching NaN guard paths (la > 0, la < 0, ua > 0, ua < 0).
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_prelu_crown_negative_coeffs(
        l0 in -10.0f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in -10.0f32..10.0,
        d1 in 0.0f32..10.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
        slope in -50.0f32..50.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);
        prop_assume!(slope.abs() >= 0.01);
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = PReluLayer::from_scalar(slope);
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { x } else { slope * x },
            &format!("PReLU({slope})"),
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // HardSigmoid CROWN backward with negative incoming coefficients
    // =========================================================================

    /// HardSigmoid CROWN backward soundness with mixed-sign incoming coefficients.
    /// HardSigmoid(x) = max(0, min(1, x/6 + 0.5)).
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_hardsigmoid_crown_negative_coeffs(
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

        let layer = HardSigmoidLayer::default_params();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| (0.2 * x + 0.5).clamp(0.0, 1.0),
            "HardSigmoid",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Clip CROWN backward with negative incoming coefficients
    // =========================================================================

    /// Clip(x, 0, 1) CROWN backward soundness with mixed-sign incoming coefficients.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_clip_crown_negative_coeffs(
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

        let layer = ClipLayer::new(0.0, 1.0);
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.clamp(0.0, 1.0),
            "Clip(0,1)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Shrink CROWN backward soundness
    // =========================================================================

    /// Shrink(x) = x - bias if x > lambd, x + bias if x < -lambd, 0 otherwise.
    /// Default: bias=0.0, lambd=0.5. Piecewise linear function.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_shrink_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let bias = 0.0;
        let lambd = 0.5;
        let layer = ShrinkLayer { bias, lambd };
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        super::assert_crown_backward_sound(
            l, u,
            |x| if x > lambd { x - bias } else if x < -lambd { x + bias } else { 0.0 },
            &result,
            "Shrink(0.0,0.5)",
            CROWN_TOLERANCE,
        )?;
    }

    /// Shrink CROWN backward soundness with nonzero bias.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_shrink_crown_biased(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let bias = 0.5;
        let lambd = 1.0;
        let layer = ShrinkLayer { bias, lambd };
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        super::assert_crown_backward_sound(
            l, u,
            |x| if x > lambd { x - bias } else if x < -lambd { x + bias } else { 0.0 },
            &result,
            "Shrink(0.5,1.0)",
            CROWN_TOLERANCE,
        )?;
    }

    /// Shrink CROWN backward with negative incoming coefficients.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_shrink_crown_negative_coeffs(
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

        let bias = 0.0;
        let lambd = 0.5;
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = ShrinkLayer { bias, lambd };
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x > lambd { x - bias } else if x < -lambd { x + bias } else { 0.0 },
            "Shrink",
            CROWN_TOLERANCE,
        )?;
    }

    /// Shrink CROWN backward with negative coefficients AND nonzero bias.
    /// Covers the combination of sign-switching + discontinuity at breakpoints.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_shrink_crown_biased_negative_coeffs(
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

        let bias = 0.5;
        let lambd = 1.0;
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = ShrinkLayer { bias, lambd };
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x > lambd { x - bias } else if x < -lambd { x + bias } else { 0.0 },
            "Shrink(0.5,1.0)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // ThresholdedRelu CROWN backward soundness
    // =========================================================================

    /// ThresholdedRelu(x, alpha) = x if x > alpha, 0 otherwise. Default alpha=1.0.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_thresholded_relu_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let alpha = 1.0;
        let layer = ThresholdedReluLayer::new(alpha);
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        super::assert_crown_backward_sound(
            l, u,
            |x| if x > alpha { x } else { 0.0 },
            &result,
            "ThresholdedRelu(1.0)",
            CROWN_TOLERANCE,
        )?;
    }

    /// ThresholdedRelu CROWN backward with l ≈ alpha boundary (division-by-zero guard).
    /// When l == alpha, the denominator (alpha - l) == 0, producing Inf slopes.
    /// Part of #1759.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_thresholded_relu_crown_l_eq_alpha(
        // Generate alpha, then l near alpha (within ±epsilon), delta for u > alpha
        alpha in -5.0f32..5.0,
        l_offset in -1e-6f32..1e-6,
        delta in 0.01f32..10.0,
    ) {
        let l = alpha + l_offset;
        let u = (l + delta).max(l + 0.01);
        prop_assume!(l < u);
        prop_assume!(l <= alpha); // crossing case: l <= alpha < u
        prop_assume!(u > alpha);

        let layer = ThresholdedReluLayer::new(alpha);
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        // Verify fully finite affine bounds in the l≈alpha boundary regime.
        assert!(result.lower_a[[0, 0]].is_finite(), "lower_a is non-finite for l={l}, u={u}, alpha={alpha}");
        assert!(result.upper_a[[0, 0]].is_finite(), "upper_a is non-finite for l={l}, u={u}, alpha={alpha}");
        assert!(result.lower_b[0].is_finite(), "lower_b is non-finite for l={l}, u={u}, alpha={alpha}");
        assert!(result.upper_b[0].is_finite(), "upper_b is non-finite for l={l}, u={u}, alpha={alpha}");

        super::assert_crown_backward_sound(
            l, u,
            |x| if x > alpha { x } else { 0.0 },
            &result,
            &format!("ThresholdedRelu({alpha}) l≈alpha"),
            CROWN_TOLERANCE,
        )?;
    }

    /// ThresholdedRelu CROWN backward with negative incoming coefficients.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_thresholded_relu_crown_negative_coeffs(
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

        let alpha = 1.0;
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let layer = ThresholdedReluLayer::new(alpha);
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x > alpha { x } else { 0.0 },
            "ThresholdedRelu",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Softsign CROWN backward with negative incoming coefficients
    // =========================================================================

    /// Softsign CROWN backward soundness with mixed-sign incoming coefficients.
    /// Softsign(x) = x / (1 + |x|). S-shaped.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_softsign_crown_negative_coeffs(
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

        let layer = SoftsignLayer::new();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| x / (1.0 + x.abs()),
            "Softsign",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // HardSwish CROWN backward with negative incoming coefficients
    // =========================================================================

    /// HardSwish CROWN backward soundness with mixed-sign incoming coefficients.
    /// HardSwish(x) = x * clamp(x/6 + 0.5, 0, 1). Non-monotonic (min at x≈-1.5).
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_hardswish_crown_negative_coeffs(
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

        let layer = HardSwishLayer::new();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            hardswish_eval,
            "HardSwish",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // IBP soundness proptests for Shrink and ThresholdedReLU
    // (These are the only piecewise activations missing proptest IBP coverage.
    // Their CROWN tests live in this file, so IBP tests go here too.)
    // =========================================================================

    /// Shrink IBP: soft thresholding with configurable bias and lambda.
    /// Shrink(x) = x - bias if x > lambd, x + bias if x < -lambd, 0 otherwise.
    /// The function is piecewise linear with breakpoints at ±lambd and a dead zone.
    /// IBP must correctly handle intervals spanning the dead zone.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_shrink_ibp(
        l in -10.0f32..10.0,
        delta in 0.0f32..20.0,
        bias in 0.0f32..2.0,
        lambd in 0.01f32..3.0,
    ) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = ShrinkLayer::new(bias, lambd);
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        // Output bounds must be valid
        prop_assert!(out_l <= out_u + 1e-6, "Shrink IBP output: lower {} > upper {}", out_l, out_u);

        // Shrink evaluation: match the scalar function exactly
        let shrink = |x: f32| -> f32 {
            if x > lambd { x - bias }
            else if x < -lambd { x + bias }
            else { 0.0 }
        };

        // Sample points must be within output bounds
        for x in sample_points(l, u, 50) {
            let fx = shrink(x);
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "Shrink IBP lower {} > shrink({}) = {} (bias={}, lambd={})", out_l, x, fx, bias, lambd
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "Shrink IBP upper {} < shrink({}) = {} (bias={}, lambd={})", out_u, x, fx, bias, lambd
            );
        }
    }

    /// ThresholdedReLU IBP: y = x if x > alpha, else 0.
    /// Similar to ReLU with a configurable threshold. The discontinuity at x=alpha
    /// makes the crossing case interesting: the function jumps from 0 to alpha.
    /// IBP must handle this correctly.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_thresholded_relu_ibp(
        l in -10.0f32..10.0,
        delta in 0.0f32..20.0,
        alpha in 0.0f32..5.0,
    ) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let layer = ThresholdedReluLayer::new(alpha);
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = layer.propagate_ibp(&input).unwrap();
        let out_l = output.lower()[[0]];
        let out_u = output.upper()[[0]];

        // Output bounds must be valid
        prop_assert!(out_l <= out_u + 1e-6, "ThresholdedReLU IBP output: lower {} > upper {}", out_l, out_u);

        // ThresholdedReLU evaluation
        let trelu = |x: f32| -> f32 {
            if x > alpha { x } else { 0.0 }
        };

        // Sample points must be within output bounds
        for x in sample_points(l, u, 50) {
            let fx = trelu(x);
            prop_assert!(
                out_l <= fx + FP_TOLERANCE,
                "ThresholdedReLU IBP lower {} > trelu({}) = {} (alpha={})", out_l, x, fx, alpha
            );
            prop_assert!(
                out_u >= fx - FP_TOLERANCE,
                "ThresholdedReLU IBP upper {} < trelu({}) = {} (alpha={})", out_u, x, fx, alpha
            );
        }
    }
}

/// Regression for #1803/#2977: PReLU NaN guard now rejects non-finite pre-activation
/// via domain_guard with NumericalInstability.
#[ntest::timeout(10000)]
#[test]
fn regression_prelu_nan_guard_unbatched_sets_infinite_biases() {
    let layer = PReluLayer::from_scalar(0.25);
    let incoming = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    let pre_activation =
        BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())
            .unwrap();

    let result = layer.propagate_linear_with_bounds(&incoming, &pre_activation);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "PReLU with NaN pre-activation should trigger domain_guard: got {:?}",
        result
    );
}
