// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::arithmetic::PowConstantLayer;
use crate::layers::common::BoundPropagation;
use crate::layers::{AbsLayer, ExpLayer, LogLayer, ReciprocalLayer, SqrtLayer};
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

// =============================================================================
// ELEMENT-WISE OPERATION SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Exp IBP soundness: for any x in [l, u], exp(x) is in computed bounds.
    /// Note: constrained to avoid overflow.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_exp_ibp((l, u) in valid_interval(20.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let exp_layer = ExpLayer::new();
        let output = exp_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let exp_x = x.exp();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= exp_x && exp_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Exp soundness violation: exp({})={} not in [{}, {}]",
                x, exp_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Log IBP soundness: for any x in [l, u], log(x) is in computed bounds.
    /// Note: constrained to positive inputs.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_log_ibp(l in 0.01f32..100.0, delta in 0.0f32..50.0) {
        let u = l + delta;
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let log_layer = LogLayer::new();
        let output = log_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let log_x = x.ln();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= log_x && log_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Log soundness violation: log({})={} not in [{}, {}]",
                x, log_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Sqrt IBP soundness: for any x in [l, u], sqrt(x) is in computed bounds.
    /// Note: constrained to non-negative inputs.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_sqrt_ibp(l in 0.0f32..100.0, delta in 0.0f32..100.0) {
        let u = l + delta;
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let sqrt_layer = SqrtLayer::new();
        let output = sqrt_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let sqrt_x = x.sqrt();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= sqrt_x && sqrt_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Sqrt soundness violation: sqrt({})={} not in [{}, {}]",
                x, sqrt_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Abs IBP soundness: for any x in [l, u], |x| is in computed bounds.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_abs_ibp((l, u) in valid_interval(100.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let abs_layer = AbsLayer::new();
        let output = abs_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let abs_x = x.abs();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= abs_x && abs_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Abs soundness violation: |{}|={} not in [{}, {}]",
                x, abs_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Reciprocal IBP soundness: for any x in [l, u], 1/x is in computed bounds.
    /// Note: constrained to avoid division by zero.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_reciprocal_ibp_positive(l in 0.1f32..100.0, delta in 0.0f32..50.0) {
        let u = l + delta;
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let recip_layer = ReciprocalLayer::new();
        let output = recip_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let recip_x = 1.0 / x;
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= recip_x && recip_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Reciprocal soundness violation: 1/{}={} not in [{}, {}]",
                x, recip_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Reciprocal IBP soundness for negative inputs.
#[ntest::timeout(10000)]
    #[test]
    fn soundness_reciprocal_ibp_negative(u in -100.0f32..-0.1, delta in 0.0f32..50.0) {
        let l = u - delta;
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let recip_layer = ReciprocalLayer::new();
        let output = recip_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let recip_x = 1.0 / x;
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= recip_x && recip_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Reciprocal negative soundness violation: 1/{}={} not in [{}, {}]",
                x, recip_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    // =========================================================================
    // PowConstant(x^2) IBP soundness
    // =========================================================================

    /// PowConstant(2) IBP soundness: for any x in [l, u], x^2 is in computed bounds.
    /// x^2 is convex with minimum at x=0. When the interval crosses zero, the lower
    /// bound of x^2 should be 0 (or close to it). The upper bound is max(l^2, u^2).
    /// Part of #3126, #2435.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_pow2_ibp((l, u) in valid_interval(100.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let pow_layer = PowConstantLayer::square();
        let output = pow_layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 20) {
            let pow_x = x * x;
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= pow_x
                    && pow_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Pow2 IBP soundness violation: ({})^2={} not in [{}, {}]",
                x, pow_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }
}

// =============================================================================
// TIGHTNESS: Point-input (degenerate interval) exactness tests (#2131)
// =============================================================================
//
// For monotone elementwise functions, a point input [x, x] should produce a
// point output [f(x), f(x)] within FP tolerance. A regression that widens bounds
// to [-inf, +inf] or adds unnecessary slack would fail these tests.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Exp tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_exp_point_input(x in -20.0f32..20.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = ExpLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = x.exp();

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE * expected.abs().max(1.0),
            "Exp point-input tightness: gap={gap} exceeds tolerance for exp({x})={expected}, bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE * expected.abs().max(1.0),
            "Exp point-input: lower bound {} differs from exp({x})={expected}",
            output.lower()[[0]]
        );
    }

    /// Log tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_log_point_input(x in 0.01f32..100.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = LogLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = x.ln();

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE * expected.abs().max(1.0),
            "Log point-input tightness: gap={gap} exceeds tolerance for ln({x})={expected}, bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE * expected.abs().max(1.0),
            "Log point-input: lower bound {} differs from ln({x})={expected}",
            output.lower()[[0]]
        );
    }

    /// Sqrt tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_sqrt_point_input(x in 0.0f32..100.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = SqrtLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = x.sqrt();

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE * expected.abs().max(1.0),
            "Sqrt point-input tightness: gap={gap} exceeds tolerance for sqrt({x})={expected}, bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE * expected.abs().max(1.0),
            "Sqrt point-input: lower bound {} differs from sqrt({x})={expected}",
            output.lower()[[0]]
        );
    }

    /// Abs tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_abs_point_input(x in -100.0f32..100.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = AbsLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = x.abs();

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE * expected.abs().max(1.0),
            "Abs point-input tightness: gap={gap} exceeds tolerance for |{x}|={expected}, bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE,
            "Abs point-input: lower bound {} differs from |{x}|={expected}",
            output.lower()[[0]]
        );
    }

    /// Reciprocal tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_reciprocal_point_input(x in 0.1f32..100.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = ReciprocalLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = 1.0 / x;

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE * expected.abs().max(1.0),
            "Reciprocal point-input tightness: gap={gap} exceeds tolerance for 1/{x}={expected}, bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE * expected.abs().max(1.0),
            "Reciprocal point-input: lower bound {} differs from 1/{x}={expected}",
            output.lower()[[0]]
        );
    }
}
