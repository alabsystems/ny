// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::{cast_f64_to_f32_down, cast_f64_to_f32_up, BoundedTensor};

use super::elu_family::{elu_family_linear_relaxation, EluFamilyParams};
use super::LinearRelaxation;
use crate::bounds::nan_propagating_max;
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

/// SELU (Scaled ELU) layer: y = lambda * (x if x >= 0, else alpha * (exp(x) - 1))
///
/// SELU is self-normalizing: for properly initialized networks, activations
/// converge to zero mean and unit variance. Uses fixed constants:
/// - alpha ≈ 1.6732632423543772848170429916717
/// - lambda ≈ 1.0507009873554804934193349852946
#[derive(Debug, Clone)]
pub struct SeluLayer;

impl SeluLayer {
    /// SELU alpha constant
    pub const ALPHA: f32 = 1.673_263_2;
    /// SELU lambda (scale) constant
    pub const LAMBDA: f32 = 1.050_701;

    /// Create a new SELU layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SeluLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl BoundPropagation for SeluLayer {
    /// IBP for SELU: y = lambda * (x if x >= 0, else alpha * (exp(x) - 1))
    ///
    /// Similar to ELU but with fixed scaling.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Guard: NaN input bounds fail closed — NaN.exp() = NaN would flow
        // silently into output bounds. (#3203, Finding 2)
        //
        // ±Inf endpoints are NOT rejected (deep-stack IBP overflow saturation,
        // dc9da0fde class): the closures below are limit-correct at infinity —
        // x = -inf takes the exp branch, λα·(exp(-inf) − 1) = -λα (the exact
        // SELU lower asymptote; the `selu_floor` clamp pins the lower endpoint
        // to it), and x = +inf takes the linear branch, λ·(+inf) = +inf =
        // SELU(+inf) (the directed-rounding steps are identity at ±inf).
        if input.lower().iter().any(|x| x.is_nan()) || input.upper().iter().any(|x| x.is_nan()) {
            return Err(NyError::NumericalInstability(
                "SELU IBP: NaN input bounds".to_string(),
            ));
        }

        // Directed rounding: for x < 0, exp is a transcendental that can round
        // either direction. Compute in f64, cast with next_down/next_up. (#1483)
        // For x >= 0, SELU(x) = lambda*x — multiplication may also round,
        // so apply directed rounding there too for consistency.
        let lambda64 = Self::LAMBDA as f64;
        let alpha64 = Self::ALPHA as f64;
        // SELU lower asymptote: -lambda * alpha (for x → -∞).
        // OUTWARD-rounded: the round-to-nearest f32 product sits ~2.9e-8 ABOVE
        // the true asymptote, so clamping to it would raise the lower bound
        // above the real SELU(x) for every x < -17 — a false proof (found by
        // adversarial review 2026-08-26). One ULP of width buys soundness.
        let selu_floor = cast_f64_to_f32_down(-(f64::from(Self::LAMBDA) * f64::from(Self::ALPHA)));
        let selu_lower = |x: f32| -> f32 {
            if x >= 0.0 {
                cast_f64_to_f32_down(lambda64 * f64::from(x))
            } else {
                // Range clamp: SELU(x) >= -lambda*alpha for all x. next_down_f32
                // can push past this for extreme negative inputs. (#3316)
                // NaN-propagating: .max() swallows NaN (IEEE 754-2008). (#3316)
                nan_propagating_max(
                    cast_f64_to_f32_down(lambda64 * alpha64 * (f64::from(x).exp() - 1.0)),
                    selu_floor,
                )
            }
        };
        let selu_upper = |x: f32| -> f32 {
            if x >= 0.0 {
                cast_f64_to_f32_up(lambda64 * f64::from(x))
            } else {
                cast_f64_to_f32_up(lambda64 * alpha64 * (f64::from(x).exp() - 1.0))
            }
        };
        let lower = input.lower().mapv(selu_lower);
        let upper = input.upper().mapv(selu_upper);
        // Conservative-bounds convention (dc9da0fde class): a saturated input
        // legitimately yields an infinite upper endpoint (SELU(+inf) = +inf).
        // An infinite endpoint is a sound, maximally conservative bound —
        // reject only NaN (fail closed, and already impossible given the
        // entry guard), never the saturation itself.
        BoundedTensor::new_allow_infinite(lower, upper)
    }
    impl_elementwise_activation!(
        @trait_methods
        SeluLayer,
        NyError::InvalidSpec(
            "SELU CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

/// Analytical linear relaxation for SELU on interval [l, u].
///
/// Returns bounds such that:
///   lower_slope * x + lower_intercept <= SELU(x) <= upper_slope * x + upper_intercept
/// for all x in [l, u].
///
/// Delegates to the shared ELU-family relaxation (elu_family.rs) parameterized with
/// SELU's fixed constants. Part of #2834.
///
/// Ref: alpha-beta-CROWN auto_LiRPA/operators/nonlinear.py (SELU relaxation)
fn selu_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    let alpha64 = SeluLayer::ALPHA as f64;
    let lambda64 = SeluLayer::LAMBDA as f64;
    let la64 = lambda64 * alpha64;

    let params = EluFamilyParams {
        positive_slope: lambda64,
        eval_negative: |x, p| p.scale * (x.exp() - 1.0),
        deriv_negative: |x, p| p.scale * x.exp(),
        // OUTWARD-rounded, for the same reason `selu_floor` is (see propagate_ibp).
        // `params.saturation` is consumed ONLY as a LOWER INTERCEPT
        // (elu_family.rs:70, :82, :108), so the f32 round-to-nearest product
        // -(LAMBDA * ALPHA) = -1.7580993175506592 asserts SELU(x) >= that value —
        // false for every x < -17.93, since the true infimum is
        // -1.7580993463430303. 5d48d59e9 outward-rounded the IBP floor and left
        // this twin unfixed; it is latent only because SELU's CROWN entry still
        // refuses non-finite pre-activations and all three consuming branches
        // require an infinite endpoint. That is precisely the guard 5d48d59e9
        // relaxed on the IBP side, so the landmine sits in the path of the work.
        // Lowering a lower intercept is unconditionally sound: it only loosens.
        saturation: cast_f64_to_f32_down(-(f64::from(SeluLayer::LAMBDA)
            * f64::from(SeluLayer::ALPHA))),
        scale: la64,
        alpha: alpha64,
        globally_convex: false,
    };
    elu_family_linear_relaxation(l, u, &params)
}

impl SeluLayer {
    impl_elementwise_activation!(
        @inherent_methods
        SeluLayer,
        selu_linear_relaxation,
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::non_finite_domain_guard("SELU", pre_activation)
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{assert_relaxation_sound, assert_relaxation_sound_exact};
    use crate::LinearBounds;
    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use ny_tensor::BoundedTensor;
    use proptest::prelude::ProptestConfig;

    /// Reference SELU evaluation in f64 for test accuracy.
    fn selu_eval(x: f32) -> f32 {
        let alpha = SeluLayer::ALPHA as f64;
        let lambda = SeluLayer::LAMBDA as f64;
        let x64 = x as f64;
        let y = if x64 >= 0.0 {
            lambda * x64
        } else {
            lambda * alpha * (x64.exp() - 1.0)
        };
        y as f32
    }

    /// Check SELU relaxation soundness using the consolidated helper from tests/mod.rs (#2496).
    fn assert_relaxation_soundness(l: f32, u: f32, tol: f32) {
        let relaxation = selu_linear_relaxation(l, u);
        assert_relaxation_sound(l, u, relaxation, selu_eval, tol, "SELU");
    }

    #[test]
    fn test_selu_relaxation_nan_lower_returns_infinite_intercepts() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = selu_linear_relaxation(f32::NAN, 1.0);
        assert_eq!(ls, 0.0);
        assert!(li.is_infinite() && li.is_sign_negative());
        assert_eq!(us, 0.0);
        assert!(ui.is_infinite() && ui.is_sign_positive());
    }

    #[test]
    fn test_selu_relaxation_nan_upper_returns_infinite_intercepts() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = selu_linear_relaxation(-1.0, f32::NAN);
        assert_eq!(ls, 0.0, "NaN upper → slope 0");
        assert!(
            li.is_infinite() && li.is_sign_negative(),
            "NaN upper → -inf intercept, got {li}"
        );
        assert_eq!(us, 0.0, "NaN upper → slope 0");
        assert!(
            ui.is_infinite() && ui.is_sign_positive(),
            "NaN upper → +inf intercept, got {ui}"
        );
    }

    #[test]
    fn test_selu_relaxation_nan_both_returns_infinite_intercepts() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = selu_linear_relaxation(f32::NAN, f32::NAN);
        assert_eq!(ls, 0.0, "NaN both → slope 0");
        assert!(
            li.is_infinite() && li.is_sign_negative(),
            "NaN both → -inf intercept, got {li}"
        );
        assert_eq!(us, 0.0, "NaN both → slope 0");
        assert!(
            ui.is_infinite() && ui.is_sign_positive(),
            "NaN both → +inf intercept, got {ui}"
        );
    }

    // ========== Case 1: Purely positive (l >= 0) ==========

    #[test]
    fn test_selu_relaxation_positive_exact() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = selu_linear_relaxation(1.0, 3.0);
        // SELU(x) = lambda*x for x >= 0, so exact relaxation
        assert!((ls - SeluLayer::LAMBDA).abs() < 1e-5);
        assert!(li.abs() < 1e-5);
        assert!((us - SeluLayer::LAMBDA).abs() < 1e-5);
        assert!(ui.abs() < 1e-5);
    }

    #[test]
    fn test_selu_relaxation_positive_soundness_grid() {
        assert_relaxation_soundness(0.0, 5.0, 1e-5);
        assert_relaxation_soundness(0.1, 10.0, 1e-4);
        assert_relaxation_soundness(0.0, 0.01, 1e-5);
    }

    // ========== Case 2: Purely negative (u <= 0) — convex region ==========

    #[test]
    fn test_selu_relaxation_negative_soundness_grid() {
        assert_relaxation_soundness(-3.0, -0.5, 1e-3);
        assert_relaxation_soundness(-1.0, -0.01, 1e-4);
        assert_relaxation_soundness(-10.0, -5.0, 1e-3);
        assert_relaxation_soundness(-0.1, -0.001, 1e-5);
    }

    #[test]
    fn test_selu_relaxation_negative_deep_soundness_grid() {
        // Deep negative: exp(x) ≈ 0, SELU(x) ≈ -lambda*alpha
        assert_relaxation_soundness(-20.0, -10.0, 1e-3);
    }

    // ========== Case 3: Crossing region (l < 0, u > 0) ==========

    #[test]
    fn test_selu_relaxation_crossing_soundness_grid() {
        assert_relaxation_soundness(-2.0, 3.0, 1e-3);
        assert_relaxation_soundness(-1.0, 1.0, 1e-3);
        assert_relaxation_soundness(-5.0, 5.0, 1e-2);
        assert_relaxation_soundness(-0.5, 0.5, 1e-3);
        assert_relaxation_soundness(-10.0, 1.0, 1e-2);
        assert_relaxation_soundness(-0.01, 10.0, 1e-3);
    }

    #[test]
    fn test_selu_relaxation_crossing_asymmetric_soundness_grid() {
        // Highly asymmetric intervals stress the chord computation
        assert_relaxation_soundness(-0.001, 100.0, 1e-1);
        assert_relaxation_soundness(-20.0, 0.001, 1e-2);
    }

    // ========== Point interval ==========

    #[test]
    fn test_selu_relaxation_point_interval() {
        // Point interval should produce tight bounds
        for x in [-2.0f32, -0.5, 0.0, 0.5, 2.0] {
            let LinearRelaxation {
                lower_slope: ls,
                lower_intercept: li,
                upper_slope: us,
                upper_intercept: ui,
            } = selu_linear_relaxation(x, x);
            let y = selu_eval(x);
            let lb = ls * x + li;
            let ub = us * x + ui;
            assert!(
                lb <= y + 1e-3,
                "Point lower violated at x={}: lb={} > y={}",
                x,
                lb,
                y
            );
            assert!(
                ub >= y - 1e-3,
                "Point upper violated at x={}: ub={} < y={}",
                x,
                ub,
                y
            );
        }
    }

    // ========== Infinite bound edge cases ==========

    /// ZERO tolerance, because this is a CERTIFIED direction.
    ///
    /// The `l = -inf` branches publish `params.saturation` directly as the
    /// lower intercept with slope 0 (elu_family.rs:70 and :82), so it must sit
    /// at or BELOW the true infimum -lambda*alpha. The f32 round-to-nearest
    /// product sits 2.879e-8 ABOVE it, certifying `SELU(x) >= v` for a v the
    /// function actually drops below for every x < -17.93 — a false hold.
    ///
    /// This class is invisible to the shared grid helper: `assert_relaxation_sound`
    /// (tests/mod.rs) asserts `lower <= f(x) + tol`, i.e. it PERMITS the certified
    /// lower bound to exceed the function by up to `tol`, and every SELU call site
    /// passes tol >= 1e-5 — three orders of magnitude above this error. The one
    /// zero-tolerance proptest samples `l in -10.0..10.0`, which cannot reach
    /// x < -17.93. So 38 grid assertions and 1000 proptest cases all stayed green
    /// over an unsound intercept. Hence an exact, direction-aware pin here.
    #[test]
    fn selu_relaxation_saturation_intercept_never_exceeds_the_true_infimum() {
        let true_infimum = -(f64::from(SeluLayer::LAMBDA) * f64::from(SeluLayer::ALPHA));
        for (l, u) in [
            (f32::NEG_INFINITY, f32::INFINITY),
            (f32::NEG_INFINITY, 1.0),
            (f32::NEG_INFINITY, 0.0),
            (f32::NEG_INFINITY, -5.0),
        ] {
            let r = selu_linear_relaxation(l, u);
            assert_eq!(r.lower_slope, 0.0, "saturated lower branch is a constant");
            assert!(
                f64::from(r.lower_intercept) <= true_infimum,
                "SELU lower intercept {} EXCEEDS the true infimum {} on [{l}, {u}] \
                 — that certifies a bound SELU(x) goes below for x < -17.93",
                r.lower_intercept,
                true_infimum
            );
        }
    }

    #[test]
    fn test_selu_relaxation_both_infinite() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = selu_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY);
        // Should be conservative: lower covers global minimum, upper is +inf
        assert_eq!(ls, 0.0);
        let la = SeluLayer::LAMBDA * SeluLayer::ALPHA;
        assert!((li - (-la)).abs() < 1e-4, "li={}, expected={}", li, -la);
        assert_eq!(us, 0.0);
        assert!(ui.is_infinite() && ui.is_sign_positive());
    }

    #[test]
    fn test_selu_relaxation_left_infinite_soundness() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = selu_linear_relaxation(f32::NEG_INFINITY, 2.0);
        // Check that bounds are valid for concrete points in the interval
        for x in [-100.0f32, -10.0, -1.0, 0.0, 1.0, 2.0] {
            let y = selu_eval(x);
            let lb = ls * x + li;
            let ub = us * x + ui;
            assert!(
                lb <= y + 1e-2,
                "Left-inf lower violated at x={}: lb={} > y={}",
                x,
                lb,
                y
            );
            assert!(
                ub >= y - 1e-2,
                "Left-inf upper violated at x={}: ub={} < y={}",
                x,
                ub,
                y
            );
        }
    }

    #[test]
    fn test_selu_relaxation_right_infinite_soundness() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_intercept: ui,
            ..
        } = selu_linear_relaxation(-2.0, f32::INFINITY);
        // Upper must be +inf since SELU is unbounded above
        assert!(ui.is_infinite() && ui.is_sign_positive());
        // Lower bound should hold for concrete points
        for x in [-2.0f32, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let y = selu_eval(x);
            let lb = ls * x + li;
            assert!(
                lb <= y + 1e-2,
                "Right-inf lower violated at x={}: lb={} > y={}",
                x,
                lb,
                y
            );
        }
    }

    // ========== IBP soundness ==========

    #[test]
    fn test_selu_ibp_soundness_grid() {
        let layer = SeluLayer::new();
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-3.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        )
        .unwrap();
        let out = layer.propagate_ibp(&input).unwrap();

        for i in 0..=50 {
            let x = -3.0 + (i as f32) * 0.1;
            let y = selu_eval(x);
            assert!(
                out.lower()[[0]] <= y + 1e-5,
                "IBP lower {} > eval {} at x={}",
                out.lower()[[0]],
                y,
                x
            );
            assert!(
                out.upper()[[0]] >= y - 1e-5,
                "IBP upper {} < eval {} at x={}",
                out.upper()[[0]],
                y,
                x
            );
        }
    }

    // ========== CROWN backward propagation soundness ==========

    #[test]
    fn test_selu_crown_backward_soundness_crossing() {
        let layer = SeluLayer::new();
        let bounds = LinearBounds::new(
            Array2::eye(1),
            Array1::zeros(1),
            Array2::eye(1),
            Array1::zeros(1),
        )
        .unwrap();
        let pre_act = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
        )
        .unwrap();
        let result =
            BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act).unwrap();

        // Verify CROWN bounds contain SELU at grid points
        for i in 0..=50 {
            let x = -2.0 + (i as f32) * 0.1;
            let y = selu_eval(x);
            let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
            let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];
            assert!(lb <= y + 1e-3, "CROWN lower {} > {} at x={}", lb, y, x);
            assert!(ub >= y - 1e-3, "CROWN upper {} < {} at x={}", ub, y, x);
        }
    }

    #[test]
    fn test_selu_crown_backward_soundness_negative() {
        let layer = SeluLayer::new();
        let bounds = LinearBounds::new(
            Array2::eye(1),
            Array1::zeros(1),
            Array2::eye(1),
            Array1::zeros(1),
        )
        .unwrap();
        let pre_act = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-4.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.5]).unwrap(),
        )
        .unwrap();
        let result =
            BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act).unwrap();

        for i in 0..=35 {
            let x = -4.0 + (i as f32) * 0.1;
            let y = selu_eval(x);
            let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
            let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];
            assert!(lb <= y + 1e-3, "CROWN lower {} > {} at x={}", lb, y, x);
            assert!(ub >= y - 1e-3, "CROWN upper {} < {} at x={}", ub, y, x);
        }
    }

    #[test]
    fn test_selu_crown_backward_no_pre_activation_errors() {
        let layer = SeluLayer::new();
        let bounds = LinearBounds::new(
            Array2::eye(1),
            Array1::zeros(1),
            Array2::eye(1),
            Array1::zeros(1),
        )
        .unwrap();
        assert!(layer.propagate_crown_backward(&bounds, None).is_err());
    }

    // ── IBP guard regression tests (#3203) ────────────────────────────

    #[test]
    fn test_ibp_nan_input_lower_rejected() {
        let layer = SeluLayer::new();
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("NaN input lower");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    #[test]
    fn test_ibp_nan_input_upper_rejected() {
        let layer = SeluLayer::new();
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap(),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("NaN input upper");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    /// Saturation flow (dc9da0fde class): ±inf input endpoints are the
    /// deep-stack IBP overflow convention and must map to SELU's exact limits
    /// — SELU(-inf) = -λα (the lower asymptote), SELU(+inf) = +inf — instead
    /// of aborting. Supersedes the pre-migration test_ibp_inf_input_rejected
    /// (NaN rejection is retained separately by the two tests above).
    #[test]
    fn test_ibp_saturated_input_flows_with_exact_limits() {
        let layer = SeluLayer::new();
        // The OUTWARD-rounded asymptote: the round-to-nearest f32 product sits
        // ~2.9e-8 above the true -lambda*alpha, so pinning that value would
        // re-assert a bound that excludes reachable SELU(x) for x < -17.
        let floor =
            cast_f64_to_f32_down(-(f64::from(SeluLayer::LAMBDA) * f64::from(SeluLayer::ALPHA)));

        // Fully saturated box [-inf, +inf].
        let input = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NEG_INFINITY]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::INFINITY]).unwrap(),
        )
        .unwrap();
        let out = layer
            .propagate_ibp(&input)
            .expect("saturated input must propagate, not error");
        assert_eq!(
            out.lower()[[0]],
            floor,
            "SELU(-inf) saturates at the outward-rounded -lambda*alpha asymptote"
        );
        // Soundness of the clamp itself: the published floor must sit at or
        // BELOW the true real-arithmetic asymptote, never above it.
        assert!(
            f64::from(out.lower()[[0]])
                <= -(f64::from(SeluLayer::LAMBDA) * f64::from(SeluLayer::ALPHA)),
            "the SELU floor must not exceed the true asymptote"
        );
        assert!(
            out.upper()[[0]].is_infinite() && out.upper()[[0]].is_sign_positive(),
            "SELU(+inf) is +inf, got {}",
            out.upper()[[0]]
        );

        // Half-saturated box [-inf, 1]: lower pinned at the asymptote, upper
        // still the ordinary finite bound.
        let input = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NEG_INFINITY]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let out = layer
            .propagate_ibp(&input)
            .expect("half-saturated input must propagate, not error");
        assert_eq!(out.lower()[[0]], floor);
        assert!(out.upper()[[0]].is_finite());
        assert!(out.upper()[[0]] >= SeluLayer::LAMBDA - 1e-5);
    }

    // ── NaN propagation regression tests (#2714) ──────────────────────

    #[test]
    fn test_relaxation_nan_lower_returns_nan_fallback_2714() {
        // NaN lower bound must produce nan_fallback intercepts (±inf),
        // not 0.0 from silently absorbed NaN.
        let r = selu_linear_relaxation(f32::NAN, 1.0);
        assert!(
            r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative(),
            "NaN lower should trigger nan_fallback, got lower_intercept={}",
            r.lower_intercept
        );
        assert!(
            r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive(),
            "NaN lower should trigger nan_fallback, got upper_intercept={}",
            r.upper_intercept
        );
    }

    #[test]
    fn test_relaxation_nan_upper_returns_nan_fallback_2714() {
        let r = selu_linear_relaxation(-1.0, f32::NAN);
        assert!(
            r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative(),
            "NaN upper should trigger nan_fallback, got lower_intercept={}",
            r.lower_intercept
        );
    }

    // ── CROWN relaxation soundness proptest (#3285) ─────────────────────

    /// Reference SELU in f64, independent of the crate f32 implementation.
    fn selu_f64_reference(x: f64) -> f64 {
        let alpha = SeluLayer::ALPHA as f64;
        let lambda = SeluLayer::LAMBDA as f64;
        if x >= 0.0 {
            lambda * x
        } else {
            lambda * alpha * (x.exp() - 1.0)
        }
    }

    /// ZERO-tolerance f64 pin on the CONSTANT / SATURATING branches — the ones
    /// `assert_relaxation_sound` cannot see, in BOTH of its blind spots at once.
    ///
    /// Every branch reached here needs an INFINITE endpoint, so the 38 grid call
    /// sites (all of which pass finite `l, u`) cannot enter it at all, and the
    /// 1000-case strict proptest below draws `l in -10.0..10.0` and then
    /// `prop_assume!`s the coefficients finite — which discards exactly this
    /// family. Nothing in the crate was checking these lines against real
    /// arithmetic.
    ///
    /// The deep tail is what makes the LOWER side bite. At x = -20, -50, -100,
    /// -1000 the certified lower bound IS the saturation constant, and on the
    /// pre-64f39e061 round-to-nearest `-(LAMBDA * ALPHA)` = -1.7580993175506592
    /// this test is RED at every one of them by 2.879e-08. (x = -17.927388686066095
    /// is the exact crossover and passes on BOTH constants, so it must never be
    /// the only tail sample — that is precisely how a lucky sample hides this
    /// class.)
    ///
    /// The `u > 0` cases pin the UPPER twin on the same three lines: the endpoint
    /// sample is where `f32(LAMBDA * u)` rounding decides soundness, and before
    /// the directed-rounding fix in `elu_family.rs` u = 3.0 and u = 100.0 are RED
    /// by 1.192e-07 and 3.338e-06. u = 2.0 is one of the EXACT cases, which is
    /// why `test_selu_relaxation_left_infinite_soundness` stayed green over it.
    #[test]
    fn selu_saturating_branches_are_exact_in_f64() {
        let cases: [(f32, f32, &[f64], &str); 5] = [
            (
                f32::NEG_INFINITY,
                f32::INFINITY,
                &[-1000.0, -100.0, -50.0, -20.0, -1.0, 0.0, 1.0, 100.0],
                "SELU [-inf, +inf]",
            ),
            (
                f32::NEG_INFINITY,
                0.0,
                &[-1000.0, -100.0, -50.0, -20.0, -1.0, 0.0],
                "SELU [-inf, 0]",
            ),
            (
                f32::NEG_INFINITY,
                -5.0,
                &[-1000.0, -100.0, -50.0, -20.0, -5.0],
                "SELU [-inf, -5]",
            ),
            (
                f32::NEG_INFINITY,
                3.0,
                &[-1000.0, -100.0, -50.0, -20.0, 0.0, 1.5, 3.0],
                "SELU [-inf, 3]",
            ),
            (
                f32::NEG_INFINITY,
                100.0,
                &[-1000.0, -100.0, -50.0, -20.0, 0.0, 100.0],
                "SELU [-inf, 100]",
            ),
        ];

        for (l, u, samples, label) in cases {
            let r = selu_linear_relaxation(l, u);
            // Non-vacuity. `assert_relaxation_sound_exact` deliberately does NOT
            // assert this — a (0, -inf, 0, +inf) plane satisfies every sample —
            // so pin the shape this case depends on, or a silent degradation to
            // the fallback would read as a pass.
            assert_eq!(
                r.lower_slope, 0.0,
                "{label}: expected a CONSTANT lower bound"
            );
            assert_eq!(
                r.upper_slope, 0.0,
                "{label}: expected a CONSTANT upper bound"
            );
            assert!(
                r.lower_intercept.is_finite(),
                "{label}: lower intercept went vacuous ({})",
                r.lower_intercept
            );
            assert_relaxation_sound_exact(l, u, r, selu_f64_reference, samples, label);
        }
    }

    /// ZERO-tolerance f64 pin on the deep negative tail, reached through the
    /// FINITE path and through `u = +inf`.
    ///
    /// The strict proptest draws `l in -10.0..10.0`, so it never sees an interval
    /// where SELU has closed onto its asymptote and the chord slope has underflowed
    /// to zero. Here `[-1000, -900]` produces a degenerate chord whose intercept is
    /// the asymptote itself, so the published constants are compared against the
    /// asymptote with nothing to spare.
    #[test]
    fn selu_deep_negative_tail_is_exact_in_f64() {
        let deep: [(f32, f32, &[f64]); 2] = [
            (-1000.0, -900.0, &[-1000.0, -950.0, -900.0]),
            (-100.0, -20.0, &[-100.0, -50.0, -20.0]),
        ];
        for (l, u, samples) in deep {
            let r = selu_linear_relaxation(l, u);
            assert!(
                r.lower_slope.is_finite()
                    && r.lower_intercept.is_finite()
                    && r.upper_slope.is_finite()
                    && r.upper_intercept.is_finite(),
                "SELU [{l}, {u}]: relaxation went vacuous ({r:?})"
            );
            assert_relaxation_sound_exact(
                l,
                u,
                r,
                selu_f64_reference,
                samples,
                &format!("SELU [{l}, {u}] deep tail"),
            );
        }

        // l finite, u = +inf: lower is the tangent at l (whose slope has
        // underflowed to a subnormal at l = -100), upper is the +inf plane
        // because SELU is unbounded above.
        let r = selu_linear_relaxation(-100.0, f32::INFINITY);
        assert!(
            r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive(),
            "SELU [-100, +inf]: upper must be the +inf plane, got {}",
            r.upper_intercept
        );
        assert!(
            r.lower_intercept.is_finite(),
            "SELU [-100, +inf]: lower intercept went vacuous ({})",
            r.lower_intercept
        );
        assert_relaxation_sound_exact(
            -100.0,
            f32::INFINITY,
            r,
            selu_f64_reference,
            &[-100.0, -50.0, -20.0, -1.0, 0.0, 1.0, 100.0],
            "SELU [-100, +inf]",
        );
    }

    proptest::proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

        /// #3285: Verify selu_linear_relaxation produces strictly sound bounds.
        /// For random intervals, the lower bound must satisfy
        ///   lower_slope * x + lower_intercept <= SELU(x)  for all x in [l, u]
        /// and the upper bound must satisfy
        ///   upper_slope * x + upper_intercept >= SELU(x)  for all x in [l, u]
        /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
        ///
        /// Ref: SiLU proptest_silu_relaxation_strict_soundness (silu/tests.rs:553).
        #[test]
        fn proptest_selu_relaxation_strict_soundness(
            l in -10.0f32..10.0,
            width in 0.01f32..20.0,
        ) {
            let u = l + width;
            let relax = selu_linear_relaxation(l, u);
            let ls = relax.lower_slope;
            let li = relax.lower_intercept;
            let us = relax.upper_slope;
            let ui = relax.upper_intercept;

            // Skip NaN fallback (infinite bounds).
            proptest::prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

            // Dense grid: 200 points, evaluated in f64 for mathematical precision.
            for k in 0..=200 {
                let t = k as f64 / 200.0;
                let x = l as f64 + t * (u as f64 - l as f64);
                let x = x.clamp(l as f64, u as f64);
                let fx = selu_f64_reference(x);

                let lower_val = ls as f64 * x + li as f64;
                proptest::prop_assert!(
                    lower_val <= fx,
                    "SELU lower bound UNSOUND at x={}: {} > SELU({})={}, \
                     interval=[{}, {}], gap={}", x, lower_val, x, fx, l, u, lower_val - fx
                );

                let upper_val = us as f64 * x + ui as f64;
                proptest::prop_assert!(
                    upper_val >= fx,
                    "SELU upper bound UNSOUND at x={}: {} < SELU({})={}, \
                     interval=[{}, {}], gap={}", x, upper_val, x, fx, l, u, fx - upper_val
                );
            }
        }
    }
}
