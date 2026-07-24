// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN asymmetric-bound soundness tests for piecewise layers.
//!
//! Split from `crown_piecewise_negcoeff.rs` to keep files under 1000 lines.
//! Part of #40, #1793.
//!
//! These tests exercise CROWN backward propagation with asymmetric incoming
//! bounds (lower_a != upper_a), which stresses sign-switching logic more than
//! symmetric bounds. Completes the 3-test coverage pattern (identity,
//! negative-coefficient, asymmetric) for all piecewise layers.
//!
//! ## Layers covered
//!
//! ReLU, LeakyReLU, PReLU, Clip, HardSigmoid, ThresholdedReLU, Shrink,
//! Softsign, HardSwish.

use crate::layers::{
    ClipLayer, HardSigmoidLayer, HardSwishLayer, LeakyReLULayer, PReluLayer, ReLULayer,
    ShrinkLayer, SoftsignLayer, ThresholdedReluLayer,
};
use crate::LinearBounds;
use ndarray::{Array1, Array2};
use proptest::prelude::*;

use super::{hardswish_eval, CROWN_TOLERANCE};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // =========================================================================
    // ReLU CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// ReLU CROWN backward with asymmetric lower_a/upper_a.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_relu_crown_asymmetric_bounds(
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

        let relu_layer = ReLULayer::new();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| relu_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.max(0.0),
            "ReLU-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // LeakyReLU CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// LeakyReLU CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_leaky_relu_crown_asymmetric_bounds(
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

        let alpha = 0.2;
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        let layer = LeakyReLULayer::new(alpha);
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { x } else { alpha * x },
            "LeakyReLU-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // PReLU CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// PReLU CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_prelu_crown_asymmetric_bounds(
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

        let slope = 0.25;
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        let layer = PReluLayer::from_scalar(slope);
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x >= 0.0 { x } else { slope * x },
            "PReLU(0.25)-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Clip CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// Clip(x, 0, 1) CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_clip_crown_asymmetric_bounds(
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

        let layer = ClipLayer::new(0.0, 1.0);
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.clamp(0.0, 1.0),
            "Clip(0,1)-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // HardSigmoid CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// HardSigmoid CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_hardsigmoid_crown_asymmetric_bounds(
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

        let layer = HardSigmoidLayer::default_params();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| (0.2 * x + 0.5).clamp(0.0, 1.0),
            "HardSigmoid-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // ThresholdedReLU CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// ThresholdedReLU CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_thresholded_relu_crown_asymmetric_bounds(
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

        let alpha = 1.0;
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        let layer = ThresholdedReluLayer::new(alpha);
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x > alpha { x } else { 0.0 },
            "ThresholdedRelu-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Shrink CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// Shrink CROWN backward with asymmetric lower_a/upper_a bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_shrink_crown_asymmetric_bounds(
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

        let bias = 0.0;
        let lambd = 0.5;
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        let layer = ShrinkLayer { bias, lambd };
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| if x > lambd { x - bias } else if x < -lambd { x + bias } else { 0.0 },
            "Shrink-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Softsign CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// Softsign CROWN backward with asymmetric lower_a/upper_a bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_softsign_crown_asymmetric_bounds(
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

        let layer = SoftsignLayer::new();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            |x| x / (1.0 + x.abs()),
            "Softsign-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // HardSwish CROWN backward with asymmetric incoming coefficients
    // =========================================================================

    /// HardSwish CROWN backward with asymmetric lower_a/upper_a bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_hardswish_crown_asymmetric_bounds(
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

        let layer = HardSwishLayer::new();
        super::assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| layer.propagate_linear_with_bounds(bounds, pre),
            hardswish_eval,
            "HardSwish-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }
}
