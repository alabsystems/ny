// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IBP soundness proptests for s-shaped and trigonometric activations.
//!
//! Covers the 7 activations identified as missing IBP proptest coverage in #2435:
//! Sigmoid, Tanh, Softplus, Sin, Cos, Tan, Arctan.
//!
//! Each test verifies that for any concrete input x within [l, u],
//! the function output f(x) falls within the IBP-computed output bounds.

use crate::layers::common::BoundPropagation;
use crate::layers::softmax::{GELULayer, GeluApproximation};
use crate::layers::{
    ArctanLayer, CosLayer, SigmoidLayer, SinLayer, SoftplusLayer, TanLayer, TanhLayer,
};
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{
    gelu_erf_eval, gelu_tanh_eval, sample_points, sigmoid_eval, softplus_eval, tanh_eval,
    valid_interval, FP_TOLERANCE,
};

// =============================================================================
// S-SHAPED ACTIVATION IBP SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Sigmoid IBP soundness: for any x in [l, u], sigmoid(x) is in computed bounds.
    /// Sigmoid is monotonically increasing, so IBP bounds should be tight:
    /// [sigmoid(l), sigmoid(u)].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sigmoid_ibp((l, u) in valid_interval(20.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let layer = SigmoidLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let sig_x = sigmoid_eval(x);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= sig_x
                    && sig_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Sigmoid soundness violation: sigmoid({})={} not in [{}, {}]",
                x, sig_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Tanh IBP soundness: for any x in [l, u], tanh(x) is in computed bounds.
    /// Tanh is monotonically increasing, so IBP bounds should be tight:
    /// [tanh(l), tanh(u)].
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tanh_ibp((l, u) in valid_interval(20.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let layer = TanhLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let tanh_x = tanh_eval(x);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= tanh_x
                    && tanh_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Tanh soundness violation: tanh({})={} not in [{}, {}]",
                x, tanh_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Softplus IBP soundness: for any x in [l, u], softplus(x) is in computed bounds.
    /// Softplus is monotonically increasing: softplus(x) = ln(1 + exp(x)).
    /// Constrained to avoid overflow in exp().
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_softplus_ibp((l, u) in valid_interval(20.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let layer = SoftplusLayer;
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let sp_x = softplus_eval(x);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= sp_x
                    && sp_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Softplus soundness violation: softplus({})={} not in [{}, {}]",
                x, sp_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }
}

// =============================================================================
// GELU ACTIVATION IBP SOUNDNESS TESTS
// =============================================================================
// GELU has two approximation variants: tanh and erf. Both need independent IBP
// coverage because they use different evaluation formulas and may have different
// IBP implementations (e.g., different critical-point computations).
// Part of #3126, #2435.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// GELU (tanh approximation) IBP soundness: for any x in [l, u],
    /// GELU_tanh(x) is in computed bounds.
    ///
    /// GELU_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    /// Non-monotonic with a minimum near x ≈ -0.68. The IBP implementation must
    /// handle this via `gelu_bound_interval` which finds the critical point.
    ///
    /// Range constrained to [-10, 10] to match CROWN proptest range and avoid
    /// extreme outputs (GELU(10) ≈ 10, GELU(-10) ≈ 0).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gelu_tanh_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let layer = GELULayer::new(GeluApproximation::Tanh);
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let gelu_x = gelu_tanh_eval(x);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= gelu_x
                    && gelu_x <= output.upper()[[0]] + FP_TOLERANCE,
                "GELU(tanh) IBP soundness violation: gelu_tanh({})={} not in [{}, {}]",
                x, gelu_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// GELU (erf) IBP soundness: for any x in [l, u],
    /// GELU_erf(x) is in computed bounds.
    ///
    /// GELU_erf(x) = 0.5 * x * (1 + erf(x / sqrt(2))).
    /// Same non-monotonic shape as tanh variant, slightly different critical point.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_gelu_erf_ibp((l, u) in valid_interval(10.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let layer = GELULayer::new(GeluApproximation::Erf);
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let gelu_x = gelu_erf_eval(x);
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= gelu_x
                    && gelu_x <= output.upper()[[0]] + FP_TOLERANCE,
                "GELU(erf) IBP soundness violation: gelu_erf({})={} not in [{}, {}]",
                x, gelu_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }
}

// =============================================================================
// TRIGONOMETRIC ACTIVATION IBP SOUNDNESS TESTS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Sin IBP soundness: for any x in [l, u], sin(x) is in computed bounds.
    /// Sin is periodic; IBP must handle extrema at kπ/2.
    /// Range constrained to [-4π, 4π] for meaningful interval widths.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sin_ibp((l, u) in valid_interval(4.0 * std::f32::consts::PI)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let layer = SinLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let sin_x = x.sin();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= sin_x
                    && sin_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Sin soundness violation: sin({})={} not in [{}, {}]",
                x, sin_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Cos IBP soundness: for any x in [l, u], cos(x) is in computed bounds.
    /// Cos is periodic; IBP must handle extrema at kπ.
    /// Range constrained to [-4π, 4π] for meaningful interval widths.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cos_ibp((l, u) in valid_interval(4.0 * std::f32::consts::PI)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let layer = CosLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let cos_x = x.cos();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= cos_x
                    && cos_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Cos soundness violation: cos({})={} not in [{}, {}]",
                x, cos_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Tan IBP soundness: for any x in [l, u], tan(x) is in computed bounds.
    /// Constrained to (-π/2, π/2) to avoid asymptotes for meaningful testing.
    /// Tests with asymptote-crossing intervals are handled separately.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tan_ibp_no_asymptote(
        l in (-1.5f32..0.0),
        u in (0.0f32..1.5)
    ) {
        // Stay well within (-π/2, π/2) ≈ (-1.5708, 1.5708)
        prop_assume!(l < u);
        prop_assume!(l > -std::f32::consts::FRAC_PI_2 + 0.01);
        prop_assume!(u < std::f32::consts::FRAC_PI_2 - 0.01);

        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let layer = TanLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let tan_x = x.tan();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= tan_x
                    && tan_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Tan soundness violation: tan({})={} not in [{}, {}]",
                x, tan_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }

    /// Tan IBP soundness when interval crosses an asymptote.
    /// The implementation should return [-inf, +inf] for safety.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_tan_ibp_asymptote_crossing(
        l in (-3.0f32..0.0),
        u in (2.0f32..5.0)
    ) {
        // This interval [l, u] crosses π/2 ≈ 1.5708, so it has an asymptote.
        prop_assume!(l < u);

        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let layer = TanLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        // With an asymptote crossing, bounds should be conservative: [-inf, +inf]
        prop_assert!(
            output.lower()[[0]] <= f32::NEG_INFINITY + 1.0,
            "Tan asymptote crossing should yield -inf lower bound, got {}",
            output.lower()[[0]]
        );
        prop_assert!(
            output.upper()[[0]] >= f32::INFINITY - 1.0,
            "Tan asymptote crossing should yield +inf upper bound, got {}",
            output.upper()[[0]]
        );
    }

    /// Arctan IBP soundness: for any x in [l, u], arctan(x) is in computed bounds.
    /// Arctan is monotonically increasing with range (-π/2, π/2).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_arctan_ibp((l, u) in valid_interval(100.0)) {
        let input = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn()
        ).unwrap();

        let layer = ArctanLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();

        for x in sample_points(l, u, 50) {
            let atan_x = x.atan();
            prop_assert!(
                output.lower()[[0]] - FP_TOLERANCE <= atan_x
                    && atan_x <= output.upper()[[0]] + FP_TOLERANCE,
                "Arctan soundness violation: arctan({})={} not in [{}, {}]",
                x, atan_x, output.lower()[[0]], output.upper()[[0]]
            );
        }
    }
}

// =============================================================================
// TIGHTNESS: Point-input (degenerate interval) exactness tests
// =============================================================================
//
// For monotone activations, a point input [x, x] should produce a
// point output [f(x), f(x)] within FP tolerance. A regression that widens bounds
// to [-inf, +inf] or adds unnecessary slack would fail these tests.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Sigmoid tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_sigmoid_point_input(x in -20.0f32..20.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = SigmoidLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = sigmoid_eval(x);

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE,
            "Sigmoid point-input tightness: gap={gap} exceeds tolerance for sigmoid({x})={expected}, \
             bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE,
            "Sigmoid point-input: lower bound {} differs from sigmoid({x})={expected}",
            output.lower()[[0]]
        );
    }

    /// Tanh tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_tanh_point_input(x in -20.0f32..20.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = TanhLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = tanh_eval(x);

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE,
            "Tanh point-input tightness: gap={gap} exceeds tolerance for tanh({x})={expected}, \
             bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE,
            "Tanh point-input: lower bound {} differs from tanh({x})={expected}",
            output.lower()[[0]]
        );
    }

    /// Softplus tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_softplus_point_input(x in -20.0f32..20.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = SoftplusLayer;
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = softplus_eval(x);

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE * expected.abs().max(1.0),
            "Softplus point-input tightness: gap={gap} exceeds tolerance for softplus({x})={expected}, \
             bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE * expected.abs().max(1.0),
            "Softplus point-input: lower bound {} differs from softplus({x})={expected}",
            output.lower()[[0]]
        );
    }

    /// Arctan tightness: point input [x, x] → output gap should be near-zero.
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_arctan_point_input(x in -100.0f32..100.0) {
        let input = BoundedTensor::new(
            arr1(&[x]).into_dyn(),
            arr1(&[x]).into_dyn(),
        ).unwrap();

        let layer = ArctanLayer::new();
        let output = layer.propagate_ibp(&input).unwrap();
        let expected = x.atan();

        let gap = output.upper()[[0]] - output.lower()[[0]];
        prop_assert!(
            gap <= FP_TOLERANCE,
            "Arctan point-input tightness: gap={gap} exceeds tolerance for arctan({x})={expected}, \
             bounds=[{}, {}]",
            output.lower()[[0]], output.upper()[[0]]
        );
        prop_assert!(
            (output.lower()[[0]] - expected).abs() <= FP_TOLERANCE,
            "Arctan point-input: lower bound {} differs from arctan({x})={expected}",
            output.lower()[[0]]
        );
    }
}
