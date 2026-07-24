// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest CROWN soundness tests for element-wise nonlinear layers.
//!
//! These tests verify that the linear relaxation envelopes used in CROWN
//! backward propagation are sound: for every `x` in `[l, u]`, the lower
//! linear bound under-approximates `f(x)` and the upper linear bound
//! over-approximates `f(x)`.
//!
//! Covers: Exp, Log, Abs, PowConstant (x^2), Sqrt, HardSwish.
//!
//! Floor/Ceil/Round/Sign tests are in `crown_piecewise_constant.rs`.
//! Sin/Cos tests are in `crown_sincos.rs`.
//! Tan/Arctan/Reciprocal tests are in `crown_trig_reciprocal.rs`.
//! SiLU/Sigmoid/Tanh/GELU tests are in `crown_s_shaped.rs`.
//! ELU/CELU/SELU/Mish/Softplus tests are in `crown_elu_family.rs`.
//!
//! Part of #1705, #1793, #40.

use crate::layers::arithmetic::{AbsLayer, PowConstantLayer};
use crate::LinearBounds;
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{
    assert_crown_backward_sound, assert_relaxation_envelope, sample_points, CROWN_TOLERANCE,
};

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    // =========================================================================
    // Exp CROWN relaxation soundness
    // =========================================================================

    /// Verify exp linear relaxation envelope: for all x in [l, u],
    ///   lower_slope * x + lower_intercept <= exp(x) <= upper_slope * x + upper_intercept
    ///
    /// Range constrained to [-10, 10] to stay within f32 precision limits for
    /// chord intercept computation. At |x| > 10, exp(x) > 20000 and the chord
    /// intercept `exp(l) - slope * l` suffers catastrophic cancellation in f32.
    /// The reference (alpha-beta-CROWN) uses float64 and doesn't hit this.
    ///
    /// Reference: alpha-beta-CROWN `BoundExp.bound_relax` in
    /// `auto_LiRPA/operators/convex_concave.py:298-313`
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_exp_crown(l in -10.0f32..10.0, delta in 0.0f32..10.0) {
        let u = (l + delta).min(10.0);
        assert_relaxation_envelope(
            l, u,
            |x| x.exp(),
            crate::layers::activations::exp::exp_linear_relaxation,
            "Exp",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Log CROWN relaxation soundness
    // =========================================================================

    /// Verify log linear relaxation envelope: for all x in [l, u] with l > 0,
    ///   lower_slope * x + lower_intercept <= ln(x) <= upper_slope * x + upper_intercept
    ///
    /// Range constrained to (0, 100] to keep inputs strictly positive.
    /// Reference: alpha-beta-CROWN `BoundLog.bound_relax` in
    /// `auto_LiRPA/operators/convex_concave.py:36-45`
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_log_crown(l in 0.01f32..100.0, delta in 0.0f32..50.0) {
        let u = l + delta;
        assert_relaxation_envelope(
            l, u,
            |x| x.ln(),
            crate::layers::activations::log::log_linear_relaxation,
            "Log",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Sqrt CROWN relaxation soundness
    // =========================================================================

    /// Verify sqrt linear relaxation envelope: for all x in [l, u] with l >= 0,
    ///   lower_slope * x + lower_intercept <= sqrt(x) <= upper_slope * x + upper_intercept
    ///
    /// Range constrained to [0, 200] to keep inputs non-negative.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sqrt_crown(l in 0.0f32..100.0, delta in 0.0f32..100.0) {
        let u = l + delta;
        assert_relaxation_envelope(
            l, u,
            |x| x.sqrt(),
            crate::layers::arithmetic::sqrt_linear_relaxation,
            "Sqrt",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Abs CROWN backward soundness
    // =========================================================================

    /// Verify Abs CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= |x| <= CROWN_upper(x)
    ///
    /// Abs has a custom backward (not crown_elementwise_backward) with
    /// piecewise linear relaxation and zero-intercept lower bound optimization.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_abs_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        // Skip degenerate intervals where l > u due to float rounding
        prop_assume!(l <= u);

        let abs_layer = AbsLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = abs_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x.abs(),
            &result,
            "Abs",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // PowConstant (x^2) CROWN backward soundness
    // =========================================================================

    /// Verify PowConstant(2) CROWN backward produces sound bounds via identity coefficients.
    /// For all x in [l, u]:
    ///   CROWN_lower(x) <= x^2 <= CROWN_upper(x)
    ///
    /// x^2 is convex: chord is upper bound, tangent is lower bound.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_pow2_crown(l in -10.0f32..10.0, delta in 0.0f32..20.0) {
        let u = (l + delta).max(l);
        prop_assume!(l <= u);

        let pow_layer = PowConstantLayer::square();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = pow_layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        assert_crown_backward_sound(
            l, u,
            |x| x * x,
            &result,
            "Pow2",
            CROWN_TOLERANCE,
        )?;
    }

    // SiLU identity-bounds test moved to crown_s_shaped.rs (#1793)

    // ELU/CELU/SELU/Mish/Softplus identity tests moved to crown_elu_family.rs (#1793)
}

// =============================================================================
// CROWN BACKWARD NEGATIVE-COEFFICIENT SOUNDNESS
// =============================================================================
// All tests above use identity incoming bounds (A = I), which means only
// the positive-coefficient branch in crown_elementwise_backward is exercised.
// These tests use non-identity bounds with both positive AND negative entries
// to verify the sign-switching logic at common.rs:164-180.
//
// The invariant: for any x in [l, u] (per neuron), the composed bound satisfies:
//   sum_i (new_lower_a[j,i] * x_i) + new_lower_b[j]
//     <= sum_i (bounds.lower_a[j,i] * f(x_i)) + bounds.lower_b[j]
// and
//   sum_i (new_upper_a[j,i] * x_i) + new_upper_b[j]
//     >= sum_i (bounds.upper_a[j,i] * f(x_i)) + bounds.upper_b[j]
// including the sign-switching paths for negative coefficients.
//
// We verify by sampling concrete x vectors and checking:
//   concretized_lower <= incoming_lower_expr(f(x))
//   concretized_upper >= incoming_upper_expr(f(x)).

use crate::layers::activations::exp::ExpLayer;
use ndarray::{Array1, Array2};

/// Verify CROWN backward soundness for Exp with arbitrary (possibly negative) coefficients.
///
/// Uses `crown_elementwise_backward` directly via `ExpLayer::propagate_linear_with_bounds`,
/// which is an inherent method (not on the `BoundPropagation` trait).
fn assert_exp_crown_negative_coeff_sound(
    pre_lower: [f32; 2],
    pre_upper: [f32; 2],
    incoming: &LinearBounds,
    tol: f32,
) -> Result<(), TestCaseError> {
    use ndarray::arr1;

    let [l0, l1] = pre_lower;
    let [u0, u1] = pre_upper;

    let pre_activation =
        BoundedTensor::new(arr1(&[l0, l1]).into_dyn(), arr1(&[u0, u1]).into_dyn()).unwrap();

    let exp_layer = ExpLayer::new();
    let result = exp_layer
        .propagate_linear_with_bounds(incoming, &pre_activation)
        .map_err(|e| TestCaseError::fail(format!("Exp propagate failed: {e}")))?;

    // Sample the 2D input space
    let samples_0 = sample_points(l0, u0, 20);
    let samples_1 = sample_points(l1, u1, 20);

    for &x0 in &samples_0 {
        for &x1 in &samples_1 {
            let fx0 = x0.exp();
            let fx1 = x1.exp();
            let incoming_lower = incoming.lower_a[[0, 0]] * fx0
                + incoming.lower_a[[0, 1]] * fx1
                + incoming.lower_b[0];
            let incoming_upper = incoming.upper_a[[0, 0]] * fx0
                + incoming.upper_a[[0, 1]] * fx1
                + incoming.upper_b[0];

            // Concretized lower: new_lower_a[0,0]*x0 + new_lower_a[0,1]*x1 + new_lower_b[0]
            let lb = result.lower_a[[0, 0]] * x0 + result.lower_a[[0, 1]] * x1 + result.lower_b[0];
            // Concretized upper: new_upper_a[0,0]*x0 + new_upper_a[0,1]*x1 + new_upper_b[0]
            let ub = result.upper_a[[0, 0]] * x0 + result.upper_a[[0, 1]] * x1 + result.upper_b[0];

            let scale_tol = tol * incoming_upper.abs().max(incoming_lower.abs()).max(1.0);

            prop_assert!(
                lb <= incoming_lower + scale_tol,
                "Exp lower bound violated at ({x0}, {x1}): \
                 lb={lb} > incoming_lower={incoming_lower} (tol={scale_tol}); \
                 lower_a={:?}, lower_b={}",
                incoming.lower_a,
                incoming.lower_b[0]
            );
            prop_assert!(
                ub + scale_tol >= incoming_upper,
                "Exp upper bound violated at ({x0}, {x1}): \
                 ub={ub} < incoming_upper={incoming_upper} (tol={scale_tol}); \
                 upper_a={:?}, upper_b={}",
                incoming.upper_a,
                incoming.upper_b[0]
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// Verify exp CROWN backward with mixed-sign incoming coefficients.
    ///
    /// This exercises the sign-switching branches in crown_elementwise_backward
    /// (common.rs:164-180): when la < 0, the lower bound uses the upper relaxation
    /// instead of the lower relaxation. This path has zero coverage from identity-
    /// bounds tests.
    ///
    /// Uses 2 neurons with independent intervals and one output with coefficients
    /// c0 in [-5, 5] and c1 in [-5, 5], ensuring both positive and negative coefficients
    /// are tested. Pre-activation bounds constrained to [-8, 8] to stay within exp's
    /// safe range (threshold 88, but CROWN computes exp(u) which overflows earlier
    /// in intermediate arithmetic).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_exp_crown_negative_coeffs(
        l0 in -8.0f32..8.0,
        d0 in 0.0f32..8.0,
        l1 in -8.0f32..8.0,
        d1 in 0.0f32..8.0,
        c0 in -5.0f32..5.0,
        c1 in -5.0f32..5.0,
    ) {
        let u0 = (l0 + d0).min(8.0);
        let u1 = (l1 + d1).min(8.0);

        // Skip near-zero coefficients where the test is trivially satisfied
        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);
        // Ensure at least one coefficient is negative to exercise the sign-switch
        prop_assume!(c0 < -0.01 || c1 < -0.01);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        assert_exp_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
        )?;
    }

    /// Verify exp CROWN backward with asymmetric incoming lower/upper linear bounds.
    ///
    /// Construction enforces incoming_lower <= incoming_upper for all x:
    /// upper_a = lower_a + delta_a (delta_a >= 0) and upper_b = lower_b + delta_b (delta_b >= 0).
    /// Since exp(x) > 0 elementwise, this guarantees validity while still testing
    /// different lower/upper coefficient matrices.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_exp_crown_negative_coeffs_asymmetric_bounds(
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

        // Skip near-zero coefficients where the test is trivially satisfied.
        prop_assume!(
            lower_c0.abs() > 0.01
                || lower_c1.abs() > 0.01
                || upper_c0.abs() > 0.01
                || upper_c1.abs() > 0.01
        );
        // Ensure bounds are truly asymmetric.
        prop_assume!(delta_c0 > 0.01 || delta_c1 > 0.01 || delta_b > 0.01);
        // Ensure a negative coefficient exists to hit sign-switch logic.
        prop_assume!(
            lower_c0 < -0.01 || lower_c1 < -0.01 || upper_c0 < -0.01 || upper_c1 < -0.01
        );

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![lower_c0, lower_c1]).unwrap(),
            Array1::from_vec(vec![lower_b]),
            Array2::from_shape_vec((1, 2), vec![upper_c0, upper_c1]).unwrap(),
            Array1::from_vec(vec![upper_b]),
        ).unwrap();

        assert_exp_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
        )?;
    }
}

use super::assert_crown_negative_coeff_sound;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    // SiLU/Sigmoid/Tanh/GELU negative-coeff + asymmetric tests moved to crown_s_shaped.rs (#1793)
    // ELU/CELU/SELU/Mish/Softplus negative-coeff + asymmetric tests moved to crown_elu_family.rs (#1793)

    // =========================================================================
    // Abs CROWN backward with negative incoming coefficients
    // =========================================================================

    /// Abs CROWN backward soundness with mixed-sign incoming coefficients.
    /// Exercises the sign-switching NaN guard paths in arithmetic.rs Abs backward.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_abs_crown_negative_coeffs(
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

        let abs_layer = AbsLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| abs_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.abs(),
            "Abs",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Abs CROWN backward with asymmetric incoming bounds
    // =========================================================================

    /// Abs CROWN backward soundness with asymmetric lower/upper coefficients.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_abs_crown_asymmetric_bounds(
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

        let abs_layer = AbsLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| abs_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.abs(),
            "Abs-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Log CROWN backward with negative coefficients
    // =========================================================================

    /// Log CROWN backward soundness with mixed-sign incoming coefficients.
    /// Pre-activation bounds are strictly positive (log domain).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_log_crown_negative_coeffs(
        l0 in 0.01f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in 0.01f32..10.0,
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

        let log_layer = crate::layers::activations::log::LogLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| log_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.ln(),
            "Log",
            CROWN_TOLERANCE,
        )?;
    }

    /// Log CROWN backward with asymmetric incoming lower/upper coefficients.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_log_crown_asymmetric_bounds(
        l0 in 0.01f32..10.0,
        d0 in 0.0f32..10.0,
        l1 in 0.01f32..10.0,
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

        let log_layer = crate::layers::activations::log::LogLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| log_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.ln(),
            "Log-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // =========================================================================
    // Sqrt CROWN backward with negative coefficients
    // =========================================================================

    /// Sqrt CROWN backward soundness with mixed-sign incoming coefficients.
    /// Pre-activation bounds are strictly positive (sqrt domain).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_sqrt_crown_negative_coeffs(
        l0 in 0.01f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in 0.01f32..20.0,
        d1 in 0.0f32..20.0,
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

        let sqrt_layer = crate::layers::arithmetic::SqrtLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| sqrt_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.sqrt(),
            "Sqrt",
            CROWN_TOLERANCE,
        )?;
    }

    /// Sqrt CROWN backward with asymmetric incoming lower/upper coefficients.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_sqrt_crown_asymmetric_bounds(
        l0 in 0.01f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in 0.01f32..20.0,
        d1 in 0.0f32..20.0,
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

        let sqrt_layer = crate::layers::arithmetic::SqrtLayer::new();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| sqrt_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x.sqrt(),
            "Sqrt-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }

    // Reciprocal negative-coeff + asymmetric tests moved to crown_trig_reciprocal.rs (#1793)
}

// =============================================================================
// CROWN BACKWARD NaN GUARD: DOMAIN GUARD REJECTION (#1736, #2836)
// =============================================================================
// Originally (#1736): verified that per-neuron maximally-loose relaxation
// (slope=0, intercept=±inf) did not produce NaN via 0.0 * ±inf in the
// sign-switching loop.
//
// After #2836: non_finite_domain_guard rejects the entire tensor at entry
// when ANY pre-activation bound is non-finite, returning NumericalInstability.
// This is the correct behavior: the backward dispatch caller falls back to IBP.
// The per-neuron NaN handling in the relaxation function is now defense-in-depth.

use crate::layers::HardSwishLayer;
use ndarray::ArrayD;
use ny_core::NyError;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// Verify domain_guard rejects non-finite pre-activation bounds at entry.
    ///
    /// Setup: 3 neurons, one with ±inf pre-activation bounds.
    /// The domain_guard (#2836) rejects the entire tensor, returning
    /// NumericalInstability so the caller can fall back to IBP.
    #[ntest::timeout(10000)]
    #[test]
    fn nan_guard_maximally_loose_multi_neuron(
        l1 in -3.0f32..3.0,
        d1 in 0.0f32..6.0,
        l2 in -3.0f32..3.0,
        d2 in 0.0f32..6.0,
    ) {
        let u1 = l1 + d1;
        let u2 = l2 + d2;

        // Neuron 0: ±inf (triggers domain_guard rejection)
        // Neurons 1, 2: normal finite bounds
        let pre = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![f32::NEG_INFINITY, l1, l2]).unwrap(),
            ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![f32::INFINITY, u1, u2]).unwrap(),
        ).unwrap();

        let layer = HardSwishLayer::new();
        let identity = LinearBounds::identity(3);
        let result = layer.propagate_linear_with_bounds(&identity, &pre);

        // domain_guard should reject non-finite pre-activation bounds
        prop_assert!(
            result.is_err(),
            "Expected NumericalInstability for non-finite pre-activation, got Ok"
        );
        match result.unwrap_err() {
            NyError::NumericalInstability(msg) => {
                prop_assert!(
                    msg.contains("HardSwish") && msg.contains("non-finite"),
                    "Expected HardSwish non-finite message, got: {msg}"
                );
            }
            other => {
                prop_assert!(false, "Expected NumericalInstability, got: {other}");
            }
        }
    }
}

// Reciprocal identity/direct + Tan identity/negative-coeff tests moved to crown_trig_reciprocal.rs (#1793)

// Arctan, Tan asymmetric, Reciprocal identity + direct tests moved to crown_trig_reciprocal.rs (#1793)

// Floor, Ceil, Round, Sign tests moved to crown_piecewise_constant.rs (#1793)
// Sin, Cos tests moved to crown_sincos.rs (#1793)

// =============================================================================
// POWCONSTANT (x^2) CROWN NEGATIVE-COEFF AND ASYMMETRIC
// =============================================================================
// PowConstant(2) only had identity-bounds coverage. These add negative-coeff
// and asymmetric tests to exercise sign-switching in the custom backward.
//
// Part of #40.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// PowConstant(2) CROWN backward with mixed-sign incoming coefficients.
    /// x^2 is convex, so the relaxation uses chord upper / tangent lower.
    /// The sign-switching in the backward pass is critical for correctness
    /// when incoming coefficients are negative (flips upper/lower).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_pow2_crown_negative_coeffs(
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

        let pow_layer = PowConstantLayer::square();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| pow_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x * x,
            "Pow2",
            CROWN_TOLERANCE,
        )?;
    }

    /// PowConstant(2) CROWN backward with asymmetric lower/upper coefficients.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_pow2_crown_asymmetric_bounds(
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

        let pow_layer = PowConstantLayer::square();
        assert_crown_negative_coeff_sound(
            [l0, l1],
            [u0, u1],
            &incoming,
            |bounds, pre| pow_layer.propagate_linear_with_bounds(bounds, pre),
            |x| x * x,
            "Pow2-asymmetric",
            CROWN_TOLERANCE,
        )?;
    }
}

// =============================================================================
// TIGHTNESS: CROWN relaxation gap converges to zero for small intervals (#2131)
// =============================================================================
//
// For smooth activation functions f over a small epsilon-ball [x-eps, x+eps],
// the CROWN relaxation gap (upper_bound - lower_bound) at the center point x
// should be O(eps). A regression that returns trivially wide bounds would fail
// these tests.
//
// Specifically, for convex/concave functions the gap is O(eps^2) (tangent line
// touches the function), so the gap/eps ratio should be small. We use a
// generous bound (gap < 10 * eps) to account for FP precision and non-ideal
// relaxation strategies. A tiny negative gap (down to -FP_TOLERANCE) is
// acceptable due to FP rounding in slope/intercept computation.
//
// Additionally, we verify that the midpoint of bounds is close to f(x0).
// Without this check, a regression returning identity (slope=1, intercept=0)
// or zero (slope=0, intercept=0) bounds would produce gap=0 and pass the gap
// test trivially. The midpoint check catches such regressions (P571).

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Exp CROWN tightness: small interval → small gap at center (#2131).
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_exp_crown_small_interval(x0 in -5.0f32..5.0) {
        let eps = 0.001_f32;
        let l = x0 - eps;
        let u = x0 + eps;

        let r = crate::layers::activations::exp::exp_linear_relaxation(l, u);
        let lower_at_center = r.lower_slope * x0 + r.lower_intercept;
        let upper_at_center = r.upper_slope * x0 + r.upper_intercept;
        let gap = upper_at_center - lower_at_center;

        // For exp, the gap at center of a small interval should be O(eps^2).
        // We use a generous bound: gap < 10 * eps.
        prop_assert!(
            gap < 10.0 * eps,
            "Exp CROWN tightness: gap={} at x0={} exceeds 10*eps={} for [{}, {}]",
            gap, x0, 10.0 * eps, l, u
        );
        // Allow tiny negative gap from FP rounding (not a soundness issue —
        // soundness is checked separately by the envelope tests).
        prop_assert!(
            gap >= -super::FP_TOLERANCE,
            "Exp CROWN tightness: gap={} at x0={} is significantly negative",
            gap, x0
        );
        // Verify bounds are near exp(x0), not trivially identity or zero (P571).
        let expected = x0.exp();
        let midpoint = f32::midpoint(lower_at_center, upper_at_center);
        let tol = 10.0 * eps * expected.abs().max(1.0);
        prop_assert!(
            (midpoint - expected).abs() < tol,
            "Exp CROWN midpoint={} far from exp({})={}, tol={}",
            midpoint, x0, expected, tol
        );
    }

    /// Log CROWN tightness: small interval → small gap at center (#2131).
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_log_crown_small_interval(x0 in 0.1f32..50.0) {
        let eps = 0.001_f32;
        let l = x0 - eps;
        let u = x0 + eps;

        let r = crate::layers::activations::log::log_linear_relaxation(l, u);
        let lower_at_center = r.lower_slope * x0 + r.lower_intercept;
        let upper_at_center = r.upper_slope * x0 + r.upper_intercept;
        let gap = upper_at_center - lower_at_center;

        prop_assert!(
            gap < 10.0 * eps,
            "Log CROWN tightness: gap={} at x0={} exceeds 10*eps={} for [{}, {}]",
            gap, x0, 10.0 * eps, l, u
        );
        prop_assert!(
            gap >= -super::FP_TOLERANCE,
            "Log CROWN tightness: gap={} at x0={} is significantly negative",
            gap, x0
        );
        // Verify bounds are near ln(x0), not trivially identity or zero (P571).
        let expected = x0.ln();
        let midpoint = f32::midpoint(lower_at_center, upper_at_center);
        let tol = 10.0 * eps * expected.abs().max(1.0);
        prop_assert!(
            (midpoint - expected).abs() < tol,
            "Log CROWN midpoint={} far from ln({})={}, tol={}",
            midpoint, x0, expected, tol
        );
    }

    /// Sqrt CROWN tightness: small interval → small gap at center (#2131).
    #[ntest::timeout(10000)]
    #[test]
    fn tightness_sqrt_crown_small_interval(x0 in 0.1f32..50.0) {
        let eps = 0.001_f32;
        let l = x0 - eps;
        let u = x0 + eps;

        let r = crate::layers::arithmetic::sqrt_linear_relaxation(l, u);
        let lower_at_center = r.lower_slope * x0 + r.lower_intercept;
        let upper_at_center = r.upper_slope * x0 + r.upper_intercept;
        let gap = upper_at_center - lower_at_center;

        prop_assert!(
            gap < 10.0 * eps,
            "Sqrt CROWN tightness: gap={} at x0={} exceeds 10*eps={} for [{}, {}]",
            gap, x0, 10.0 * eps, l, u
        );
        prop_assert!(
            gap >= -super::FP_TOLERANCE,
            "Sqrt CROWN tightness: gap={} at x0={} is significantly negative",
            gap, x0
        );
        // Verify bounds are near sqrt(x0), not trivially identity or zero (P571).
        let expected = x0.sqrt();
        let midpoint = f32::midpoint(lower_at_center, upper_at_center);
        let tol = 10.0 * eps * expected.abs().max(1.0);
        prop_assert!(
            (midpoint - expected).abs() < tol,
            "Sqrt CROWN midpoint={} far from sqrt({})={}, tol={}",
            midpoint, x0, expected, tol
        );
    }
}

// Domain guard rejection tests moved to crown_domain_guard.rs (#3070)
