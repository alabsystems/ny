// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest CROWN soundness tests for Snake activation: y = x + (1/a) * sin²(a*x).
//!
//! Snake is monotonically non-decreasing (f'(x) = 1 + sin(2ax) >= 0), so IBP is
//! exact. CROWN relaxation uses chord + analytical deviation bounds via critical
//! point enumeration. The frequency parameter `a` controls oscillation density.
//!
//! Tests exercise multiple `a` values to cover:
//! - Low frequency (a=0.5): few critical points per interval
//! - Default frequency (a=1.0): moderate oscillation
//! - High frequency (a=10.0): many critical points, stresses enumeration cap
//!
//! Reference: Ziyin et al. 2020, "Neural Networks Fail to Learn Periodic Functions."
//! Implementation: layers/activations/snake/mod.rs

use crate::layers::activations::snake::snake_eval_f32;
use crate::layers::SnakeLayer;
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{assert_crown_backward_sound, assert_crown_negative_coeff_sound, CROWN_TOLERANCE};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    // =========================================================================
    // Snake CROWN backward soundness (identity bounds, default a=1.0)
    // =========================================================================

    /// Verify Snake CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= snake(x) <= CROWN_upper(x)
    ///
    /// Uses default frequency a=1.0, interval widths up to 10.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_snake_crown_default(center in -10.0f32..10.0, width in 0.0f32..10.0) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let snake_layer = SnakeLayer::new(1.0).expect("test: valid Snake");
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).expect("test: valid Snake");

        let result = snake_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .expect("test: valid Snake");

        assert_crown_backward_sound(
            l, u,
            |x| snake_eval_f32(x, 1.0),
            &result,
            "Snake(a=1.0)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Snake CROWN backward soundness (high frequency a=10.0)
    // =========================================================================

    /// High frequency Snake tests the critical point enumeration with many
    /// oscillation periods in a single interval. At a=10 with width=5,
    /// there are ~16 critical points per interval.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_snake_crown_high_freq(center in -10.0f32..10.0, width in 0.0f32..5.0) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let snake_layer = SnakeLayer::new(10.0).expect("test: valid Snake");
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).expect("test: valid Snake");

        let result = snake_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .expect("test: valid Snake");

        assert_crown_backward_sound(
            l, u,
            |x| snake_eval_f32(x, 10.0),
            &result,
            "Snake(a=10.0)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Snake CROWN backward soundness (low frequency a=0.1)
    // =========================================================================

    /// Low frequency Snake: wide intervals relative to oscillation period.
    /// Tests the multi-period relaxation path from #3051.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_snake_crown_low_freq(center in -20.0f32..20.0, width in 0.0f32..20.0) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let snake_layer = SnakeLayer::new(0.1).expect("test: valid Snake");
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).expect("test: valid Snake");

        let result = snake_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .expect("test: valid Snake");

        assert_crown_backward_sound(
            l, u,
            |x| snake_eval_f32(x, 0.1),
            &result,
            "Snake(a=0.1)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Snake CROWN with mixed-sign incoming coefficients (a=1.0)
    // =========================================================================

    /// Exercises the sign-switching branches in CROWN backward for Snake
    /// with negative incoming coefficients. Two-neuron test.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_snake_crown_negative_coeffs(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..5.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..5.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(8.0);
        let u1 = (l1 + d1).min(8.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).expect("test: valid Snake"),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).expect("test: valid Snake"),
            Array1::zeros(1),
        ).expect("test: valid Snake");

        let snake_layer = SnakeLayer::new(1.0).expect("test: valid Snake");
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| snake_layer.propagate_linear_with_bounds(bounds, pre),
            |x| snake_eval_f32(x, 1.0),
            "Snake(a=1.0)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Snake CROWN with asymmetric lower_a/upper_a
    // =========================================================================

    /// Tests the case where incoming lower and upper coefficient matrices differ.
    /// Exercises the full generality of the CROWN backward pass.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_snake_crown_asymmetric_bounds(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..5.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..5.0,
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
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).expect("test: valid Snake"),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).expect("test: valid Snake"),
            Array1::from_vec(vec![upper_b]),
        ).expect("test: valid Snake");

        let snake_layer = SnakeLayer::new(1.0).expect("test: valid Snake");
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| snake_layer.propagate_linear_with_bounds(bounds, pre),
            |x| snake_eval_f32(x, 1.0),
            "Snake-asymmetric(a=1.0)",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Snake CROWN with varying frequency parameter
    // =========================================================================

    /// Tests CROWN soundness with randomized frequency parameter.
    /// This is the most comprehensive test: random a, random intervals.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_snake_crown_random_freq(
        center in -10.0f32..10.0,
        width in 0.0f32..8.0,
        a in 0.1f32..20.0,
    ) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let snake_layer = SnakeLayer::new(a).expect("test: valid Snake");
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).expect("test: valid Snake");

        let result = snake_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .expect("test: valid Snake");

        assert_crown_backward_sound(
            l, u,
            |x| snake_eval_f32(x, a),
            &result,
            &format!("Snake(a={a})"),
            CROWN_TOLERANCE,
        )?;
    }
}

#[test]
fn regression_snake_crown_per_channel_alpha_4117() {
    let snake_layer =
        SnakeLayer::per_channel(Array1::from_vec(vec![0.5, 2.0])).expect("test: valid Snake");
    let identity = LinearBounds::identity(2);
    let pre_activation = BoundedTensor::new(
        arr1(&[-1.5, -0.5]).into_dyn(),
        arr1(&[0.75, 1.25]).into_dyn(),
    )
    .expect("test: valid Snake");

    let result = snake_layer
        .propagate_linear_with_bounds(&identity, &pre_activation)
        .expect("test: valid Snake");

    for &x0 in &[-1.5, -1.0, -0.25, 0.0, 0.5, 0.75] {
        let y0 = snake_eval_f32(x0, 0.5);
        let lb0 = result.lower_a[[0, 0]] * x0 + result.lower_b[0];
        let ub0 = result.upper_a[[0, 0]] * x0 + result.upper_b[0];
        assert!(
            lb0 <= y0 + CROWN_TOLERANCE,
            "dim 0 lower {} > eval {} at x={}",
            lb0,
            y0,
            x0
        );
        assert!(
            ub0 >= y0 - CROWN_TOLERANCE,
            "dim 0 upper {} < eval {} at x={}",
            ub0,
            y0,
            x0
        );
    }

    for &x1 in &[-0.5, -0.25, 0.0, 0.5, 1.0, 1.25] {
        let y1 = snake_eval_f32(x1, 2.0);
        let lb1 = result.lower_a[[1, 1]] * x1 + result.lower_b[1];
        let ub1 = result.upper_a[[1, 1]] * x1 + result.upper_b[1];
        assert!(
            lb1 <= y1 + CROWN_TOLERANCE,
            "dim 1 lower {} > eval {} at x={}",
            lb1,
            y1,
            x1
        );
        assert!(
            ub1 >= y1 - CROWN_TOLERANCE,
            "dim 1 upper {} < eval {} at x={}",
            ub1,
            y1,
            x1
        );
    }
}

// =========================================================================
// Snake IBP soundness proptests
// =========================================================================
// Snake is the ONLY activation with CROWN proptests but no IBP proptest.
// Since Snake is monotonically non-decreasing (f'(x) = 1 + sin(2ax) >= 0),
// IBP is exact: [f(l), f(u)]. These proptests verify this property.

use super::{sample_points, FP_TOLERANCE};
use crate::layers::common::BoundPropagation;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// IBP soundness for Snake with default frequency a=1.0.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_snake_ibp_default(center in -10.0f32..10.0, width in 0.0f32..10.0) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let snake_layer = SnakeLayer::new(1.0).expect("test: valid Snake");
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = snake_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let y = snake_eval_f32(x, 1.0);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= y
                    && y <= output.upper()[[0]] + FP_TOLERANCE,
                "Snake(a=1.0) IBP soundness violation: f({})={} not in [{}, {}]",
                x, y, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// IBP soundness for Snake with high frequency a=10.0.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_snake_ibp_high_freq(center in -5.0f32..5.0, width in 0.0f32..5.0) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let snake_layer = SnakeLayer::new(10.0).expect("test: valid Snake");
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = snake_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 30) {
            let y = snake_eval_f32(x, 10.0);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= y
                    && y <= output.upper()[[0]] + FP_TOLERANCE,
                "Snake(a=10.0) IBP soundness violation: f({})={} not in [{}, {}]",
                x, y, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// IBP soundness for Snake with low frequency a=0.1.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_snake_ibp_low_freq(center in -20.0f32..20.0, width in 0.0f32..20.0) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let snake_layer = SnakeLayer::new(0.1).expect("test: valid Snake");
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = snake_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let y = snake_eval_f32(x, 0.1);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= y
                    && y <= output.upper()[[0]] + FP_TOLERANCE,
                "Snake(a=0.1) IBP soundness violation: f({})={} not in [{}, {}]",
                x, y, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// IBP soundness for Snake with random frequency a in [0.1, 20.0].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_snake_ibp_random_freq(
        a in 0.1f32..20.0,
        center in -10.0f32..10.0,
        width in 0.0f32..10.0
    ) {
        let l = center - width / 2.0;
        let u = center + width / 2.0;
        prop_assume!(l <= u);

        let snake_layer = SnakeLayer::new(a).expect("test: valid Snake");
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let output = snake_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let y = snake_eval_f32(x, a);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= y
                    && y <= output.upper()[[0]] + FP_TOLERANCE,
                "Snake(a={}) IBP soundness violation: f({})={} not in [{}, {}]",
                a, x, y, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// IBP tightness for point-input: Snake([x,x]) should yield [f(x), f(x)].
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_snake_ibp_point_input(x in -10.0f32..10.0, a in 0.1f32..20.0) {
        let snake_layer = SnakeLayer::new(a).expect("test: valid Snake");
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let output = snake_layer.propagate_ibp(&input).unwrap();
        let y = snake_eval_f32(x, a);

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap.abs() < FP_TOLERANCE,
            "Snake(a={}) IBP tightness violation: point input x={} gave gap={} (lower={}, upper={}, f(x)={})",
            a, x, gap, output.lower()[[0]], output.upper()[[0]], y
        );
    }
}
