// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest CROWN soundness tests for piecewise constant layers:
//! Floor, Ceil, Round, Sign.
//!
//! These layers have zero-slope relaxations (constant bounds equivalent to IBP).
//! CROWN backward with identity incoming bounds should produce slope=0 and
//! intercepts equal to the IBP bounds.
//!
//! Part of #40, #1793.

use crate::layers::{CeilLayer, FloorLayer, RoundLayer, SignLayer, TruncLayer};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{assert_crown_backward_sound, sample_points, CROWN_TOLERANCE};

/// Trait to abstract over Floor/Ceil/Round/Sign's `propagate_linear_with_bounds`.
trait ConstantCrownLayer {
    fn propagate_linear_with_bounds_generic(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> ny_core::Result<LinearBounds>;
}

impl ConstantCrownLayer for FloorLayer {
    fn propagate_linear_with_bounds_generic(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> ny_core::Result<LinearBounds> {
        self.propagate_linear_with_bounds(bounds, pre_activation)
    }
}

impl ConstantCrownLayer for CeilLayer {
    fn propagate_linear_with_bounds_generic(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> ny_core::Result<LinearBounds> {
        self.propagate_linear_with_bounds(bounds, pre_activation)
    }
}

impl ConstantCrownLayer for RoundLayer {
    fn propagate_linear_with_bounds_generic(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> ny_core::Result<LinearBounds> {
        self.propagate_linear_with_bounds(bounds, pre_activation)
    }
}

impl ConstantCrownLayer for TruncLayer {
    fn propagate_linear_with_bounds_generic(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> ny_core::Result<LinearBounds> {
        self.propagate_linear_with_bounds(bounds, pre_activation)
    }
}

impl ConstantCrownLayer for SignLayer {
    fn propagate_linear_with_bounds_generic(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> ny_core::Result<LinearBounds> {
        self.propagate_linear_with_bounds(bounds, pre_activation)
    }
}

/// Verify CROWN backward soundness for a piecewise constant layer with
/// arbitrary (possibly negative) incoming coefficients on 2 neurons.
///
/// For zero-slope relaxations, the CROWN backward result captures the constant
/// intercept in lower_b/upper_b. The key sign-switching behavior:
/// - positive coeff * [lower_intercept, upper_intercept] keeps orientation
/// - negative coeff * [lower_intercept, upper_intercept] flips orientation
fn assert_constant_crown_negative_coeff_sound<F, L>(
    layer: &L,
    f: F,
    pre_lower: [f32; 2],
    pre_upper: [f32; 2],
    incoming: &LinearBounds,
    tol: f32,
    name: &str,
) -> Result<(), TestCaseError>
where
    F: Fn(f32) -> f32,
    L: ConstantCrownLayer,
{
    let [l0, l1] = pre_lower;
    let [u0, u1] = pre_upper;

    let pre_activation =
        BoundedTensor::new(arr1(&[l0, l1]).into_dyn(), arr1(&[u0, u1]).into_dyn()).unwrap();

    let result = layer
        .propagate_linear_with_bounds_generic(incoming, &pre_activation)
        .map_err(|e| TestCaseError::fail(format!("{name} propagate failed: {e}")))?;

    let samples_0 = sample_points(l0, u0, 20);
    let samples_1 = sample_points(l1, u1, 20);

    for &x0 in &samples_0 {
        for &x1 in &samples_1 {
            let fx0 = f(x0);
            let fx1 = f(x1);
            let incoming_lower = incoming.lower_a[[0, 0]] * fx0
                + incoming.lower_a[[0, 1]] * fx1
                + incoming.lower_b[0];
            let incoming_upper = incoming.upper_a[[0, 0]] * fx0
                + incoming.upper_a[[0, 1]] * fx1
                + incoming.upper_b[0];

            let lb = result.lower_a[[0, 0]] * x0 + result.lower_a[[0, 1]] * x1 + result.lower_b[0];
            let ub = result.upper_a[[0, 0]] * x0 + result.upper_a[[0, 1]] * x1 + result.upper_b[0];

            let scale_tol = tol * incoming_upper.abs().max(incoming_lower.abs()).max(1.0);

            prop_assert!(
                lb <= incoming_lower + scale_tol,
                "{name} lower bound violated at ({x0}, {x1}): \
                 lb={lb} > incoming_lower={incoming_lower} (tol={scale_tol})"
            );
            prop_assert!(
                ub + scale_tol >= incoming_upper,
                "{name} upper bound violated at ({x0}, {x1}): \
                 ub={ub} < incoming_upper={incoming_upper} (tol={scale_tol})"
            );
        }
    }
    Ok(())
}

// =============================================================================
// Identity-bounds and negative-coefficient tests
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Floor(x) = ⌊x⌋. Piecewise constant, monotonically non-decreasing.
    /// CROWN relaxation: slope=0, lower_intercept=floor(l), upper_intercept=floor(u).
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_floor_crown(l in -20.0f32..20.0, delta in 0.0f32..20.0) {
        let u = l + delta;

        let layer = FloorLayer::new();
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
            |x| x.floor(),
            &result,
            "Floor",
            CROWN_TOLERANCE,
        )?;
    }

    /// Floor CROWN backward with mixed-sign incoming coefficients (2 neurons).
    /// Exercises sign-switching in crown_elementwise_backward for zero-slope relaxations.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_floor_crown_negative_coeffs(
        l0 in -20.0f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in -20.0f32..20.0,
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

        assert_constant_crown_negative_coeff_sound(
            &FloorLayer::new(),
            |x| x.floor(),
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Floor",
        )?;
    }

    /// Trunc(x) rounds toward zero. Piecewise constant, monotone non-decreasing.
    /// CROWN relaxation: slope=0, lower_intercept=trunc(l), upper_intercept=trunc(u).
    /// Lowered from ONNX Cast-to-int (#cctsdb B1).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_trunc_crown(l in -20.0f32..20.0, delta in 0.0f32..20.0) {
        let u = l + delta;

        let layer = TruncLayer::new();
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
            |x| x.trunc(),
            &result,
            "Trunc",
            CROWN_TOLERANCE,
        )?;
    }

    /// Trunc CROWN backward with mixed-sign incoming coefficients (2 neurons).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_trunc_crown_negative_coeffs(
        l0 in -20.0f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in -20.0f32..20.0,
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

        assert_constant_crown_negative_coeff_sound(
            &TruncLayer::new(),
            |x| x.trunc(),
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Trunc",
        )?;
    }

    /// Ceil(x) = ⌈x⌉. Piecewise constant, monotonically non-decreasing.
    /// CROWN relaxation: slope=0, lower_intercept=ceil(l), upper_intercept=ceil(u).
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_ceil_crown(l in -20.0f32..20.0, delta in 0.0f32..20.0) {
        let u = l + delta;

        let layer = CeilLayer::new();
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
            |x| x.ceil(),
            &result,
            "Ceil",
            CROWN_TOLERANCE,
        )?;
    }

    /// Ceil CROWN backward with mixed-sign incoming coefficients (2 neurons).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_ceil_crown_negative_coeffs(
        l0 in -20.0f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in -20.0f32..20.0,
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

        assert_constant_crown_negative_coeff_sound(
            &CeilLayer::new(),
            |x| x.ceil(),
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Ceil",
        )?;
    }

    /// Round(x) = round-half-to-even. Piecewise constant, monotonically non-decreasing.
    /// CROWN relaxation: slope=0, lower_intercept=round(l), upper_intercept=round(u).
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_round_crown(l in -20.0f32..20.0, delta in 0.0f32..20.0) {
        let u = l + delta;

        let layer = RoundLayer::new();
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
            |x| x.round_ties_even(),
            &result,
            "Round",
            CROWN_TOLERANCE,
        )?;
    }

    /// Round CROWN backward with mixed-sign incoming coefficients (2 neurons).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_round_crown_negative_coeffs(
        l0 in -20.0f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in -20.0f32..20.0,
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

        assert_constant_crown_negative_coeff_sound(
            &RoundLayer::new(),
            |x| x.round_ties_even(),
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Round",
        )?;
    }

    /// Sign(x) = -1 if x<0, 0 if x==0, 1 if x>0. Piecewise constant.
    /// CROWN relaxation: slope=0, intercepts depend on interval position relative to zero.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sign_crown(l in -20.0f32..20.0, delta in 0.0f32..20.0) {
        let u = l + delta;

        let layer = SignLayer::new();
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
            |x| {
                if x > 0.0 { 1.0 }
                else if x < 0.0 { -1.0 }
                else { 0.0 }
            },
            &result,
            "Sign",
            CROWN_TOLERANCE,
        )?;
    }

    /// Sign CROWN backward with mixed-sign incoming coefficients (2 neurons).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_sign_crown_negative_coeffs(
        l0 in -20.0f32..20.0,
        d0 in 0.0f32..20.0,
        l1 in -20.0f32..20.0,
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

        let sign_fn = |x: f32| -> f32 {
            if x > 0.0 { 1.0 }
            else if x < 0.0 { -1.0 }
            else { 0.0 }
        };

        assert_constant_crown_negative_coeff_sound(
            &SignLayer::new(),
            sign_fn,
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Sign",
        )?;
    }

    /// Sign CROWN with intervals guaranteed to cross zero.
    /// Exercises all branches in the Sign relaxation function:
    /// purely positive, purely negative, crossing zero, touching zero from one side.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_sign_crown_crossing_zero(
        neg_part in 0.01f32..10.0,
        pos_part in 0.01f32..10.0,
    ) {
        let l = -neg_part;
        let u = pos_part;

        let layer = SignLayer::new();
        let identity = LinearBounds::identity(1);
        let pre_activation = BoundedTensor::new(
            arr1(&[l]).into_dyn(),
            arr1(&[u]).into_dyn(),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .unwrap();

        // With zero slope, result should have lower_b = -1, upper_b = 1 for crossing-zero
        let li = result.lower_b[0];
        let ui = result.upper_b[0];
        prop_assert!(
            li <= -1.0 + CROWN_TOLERANCE,
            "Sign crossing-zero lower intercept should be <= -1, got {li}"
        );
        prop_assert!(
            ui >= 1.0 - CROWN_TOLERANCE,
            "Sign crossing-zero upper intercept should be >= 1, got {ui}"
        );

        assert_crown_backward_sound(
            l, u,
            |x| {
                if x > 0.0 { 1.0 }
                else if x < 0.0 { -1.0 }
                else { 0.0 }
            },
            &result,
            "Sign-crossing",
            CROWN_TOLERANCE,
        )?;
    }
}

// =============================================================================
// Asymmetric incoming bound tests
// =============================================================================
// These layers use constant CROWN relaxations (slope=0), so asymmetric
// incoming bounds exercise the sign-switching in crown_elementwise_backward
// when lower_a != upper_a. Lower risk than nonlinear layers but completes
// the 3-test coverage pattern.
//
// Part of #40.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Floor CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_floor_crown_asymmetric_bounds(
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

        assert_constant_crown_negative_coeff_sound(
            &FloorLayer::new(),
            |x| x.floor(),
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Floor-asymmetric",
        )?;
    }

    /// Ceil CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_ceil_crown_asymmetric_bounds(
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

        assert_constant_crown_negative_coeff_sound(
            &CeilLayer::new(),
            |x| x.ceil(),
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Ceil-asymmetric",
        )?;
    }

    /// Round CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_round_crown_asymmetric_bounds(
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

        assert_constant_crown_negative_coeff_sound(
            &RoundLayer::new(),
            |x| x.round_ties_even(),
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Round-asymmetric",
        )?;
    }

    /// Sign CROWN backward with asymmetric lower_a/upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_sign_crown_asymmetric_bounds(
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

        let sign_fn = |x: f32| -> f32 {
            if x > 0.0 { 1.0 }
            else if x < 0.0 { -1.0 }
            else { 0.0 }
        };

        assert_constant_crown_negative_coeff_sound(
            &SignLayer::new(),
            sign_fn,
            [l0, l1],
            [u0, u1],
            &incoming,
            CROWN_TOLERANCE,
            "Sign-asymmetric",
        )?;
    }
}
