// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

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
        // Guard: non-finite input bounds → NaN.exp() = NaN flows silently into
        // output bounds. CROWN path rejects via non_finite_domain_guard (selu.rs:277-278).
        // Pattern: Exp layer guard at exp.rs:44-50. (#3203, Finding 2)
        if input.lower().iter().any(|x| !x.is_finite())
            || input.upper().iter().any(|x| !x.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "SELU IBP: non-finite input bounds".to_string(),
            ));
        }

        // Directed rounding: for x < 0, exp is a transcendental that can round
        // either direction. Compute in f64, cast with next_down/next_up. (#1483)
        // For x >= 0, SELU(x) = lambda*x — multiplication may also round,
        // so apply directed rounding there too for consistency.
        let lambda64 = Self::LAMBDA as f64;
        let alpha64 = Self::ALPHA as f64;
        // SELU lower asymptote: -lambda * alpha (for x → -∞)
        let selu_floor = -(Self::LAMBDA * Self::ALPHA);
        let selu_lower = |x: f32| -> f32 {
            if x >= 0.0 {
                next_down_f32((lambda64 * x as f64) as f32)
            } else {
                // Range clamp: SELU(x) >= -lambda*alpha for all x. next_down_f32
                // can push past this for extreme negative inputs. (#3316)
                // NaN-propagating: .max() swallows NaN (IEEE 754-2008). (#3316)
                nan_propagating_max(
                    next_down_f32((lambda64 * alpha64 * ((x as f64).exp() - 1.0)) as f32),
                    selu_floor,
                )
            }
        };
        let selu_upper = |x: f32| -> f32 {
            if x >= 0.0 {
                next_up_f32((lambda64 * x as f64) as f32)
            } else {
                next_up_f32((lambda64 * alpha64 * ((x as f64).exp() - 1.0)) as f32)
            }
        };
        let lower = input.lower().mapv(selu_lower);
        let upper = input.upper().mapv(selu_upper);
        BoundedTensor::new(lower, upper)
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
    let la = SeluLayer::LAMBDA * SeluLayer::ALPHA;

    let params = EluFamilyParams {
        positive_slope: lambda64,
        eval_negative: |x, p| p.scale * (x.exp() - 1.0),
        deriv_negative: |x, p| p.scale * x.exp(),
        saturation: -la,
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
    use crate::tests::assert_relaxation_sound;
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

    #[test]
    fn test_ibp_inf_input_rejected() {
        let layer = SeluLayer::new();
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NEG_INFINITY]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("Inf input");
        assert!(matches!(err, NyError::NumericalInstability(_)));
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
