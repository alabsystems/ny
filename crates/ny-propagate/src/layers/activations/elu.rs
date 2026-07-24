// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::elu_family::{elu_family_linear_relaxation, EluFamilyParams};
use super::validate::validate_positive_finite;
use super::LinearRelaxation;
use crate::bounds::nan_propagating_max;
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

/// ELU (Exponential Linear Unit) layer: y = x if x >= 0, else alpha * (exp(x) - 1)
///
/// ELU helps push mean activations closer to zero (compared to ReLU), which can
/// speed up learning. The exponential term ensures smooth transitions at zero.
#[derive(Debug, Clone)]
pub struct EluLayer {
    /// Scale for negative values (typically 1.0)
    pub(crate) alpha: f32,
}

impl EluLayer {
    /// Validate and create a new ELU layer with the given alpha.
    pub fn try_new(alpha: f32) -> Result<Self> {
        Ok(Self {
            alpha: validate_positive_finite(alpha, "EluLayer", "alpha")?,
        })
    }

    /// Create a new ELU layer with the given alpha.
    pub fn new(alpha: f32) -> Self {
        Self::try_new(alpha).expect("invariant: EluLayer::new requires validated alpha")
    }

    /// Create an ELU layer with default alpha = 1.0.
    pub fn default_alpha() -> Self {
        Self::new(1.0)
    }
}

impl BoundPropagation for EluLayer {
    /// IBP for ELU: y = x if x >= 0, else alpha * (exp(x) - 1)
    ///
    /// For x in [l, u]:
    /// - If l >= 0: y in [l, u] (positive region, identity)
    /// - If u <= 0: y in [alpha*(exp(l)-1), alpha*(exp(u)-1)] (negative region)
    /// - If l < 0 < u: y in [alpha*(exp(l)-1), u] (crossing region)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let alpha = self.alpha;
        // Guard: ELU requires alpha > 0 for sound bound propagation.
        // - alpha < 0: f(x<0) = alpha*(exp(x)-1) is decreasing (non-monotone
        //   overall), so pointwise IBP produces lower > upper. (#2779)
        // - alpha = 0: f(x<0) = 0 (degenerate constant); CROWN relaxation
        //   assumes alpha > 0 for convexity. Reject as likely config error.
        // - non-finite alpha: prevents NaN/Inf propagation.
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "EluLayer requires finite positive alpha, got {}",
                alpha,
            )));
        }
        // Guard: non-finite input bounds → NaN.exp() = NaN flows silently into
        // output bounds. CROWN path rejects via non_finite_domain_guard.
        // Pattern: SELU guard at selu.rs:47-53.
        if input.lower().iter().any(|x| !x.is_finite())
            || input.upper().iter().any(|x| !x.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "ELU IBP: non-finite input bounds".to_string(),
            ));
        }
        // Directed rounding: for x < 0, exp is a transcendental that can round
        // either direction. Compute in f64, cast with next_down/next_up. (#1483)
        // For x >= 0, ELU(x) = x is exact (no rounding needed).
        let alpha64 = alpha as f64;
        let elu_lower = |x: f32| -> f32 {
            if x >= 0.0 {
                x
            } else {
                // Range clamp: ELU(x) >= -alpha for all x. next_down_f32 can push
                // past -alpha for extreme negative inputs (exp(-1000) → 0). (#3316)
                // NaN-propagating: .max() swallows NaN (IEEE 754-2008). (#3316)
                nan_propagating_max(
                    next_down_f32((alpha64 * ((x as f64).exp() - 1.0)) as f32),
                    -alpha,
                )
            }
        };
        let elu_upper = |x: f32| -> f32 {
            if x >= 0.0 {
                x
            } else {
                next_up_f32((alpha64 * ((x as f64).exp() - 1.0)) as f32)
            }
        };
        let lower = input.lower().mapv(elu_lower);
        let upper = input.upper().mapv(elu_upper);
        BoundedTensor::new(lower, upper)
    }
    impl_elementwise_activation!(
        @trait_methods
        EluLayer,
        NyError::InvalidSpec(
            "ELU CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

/// Analytical linear relaxation for ELU on interval [l, u].
///
/// Returns bounds such that:
///   lower_slope * x + lower_intercept <= ELU(x) <= upper_slope * x + upper_intercept
/// for all x in [l, u].
///
/// Delegates to the shared ELU-family relaxation (elu_family.rs) parameterized with
/// ELU's mathematical expressions. Part of #2834.
///
/// Ref: alpha-beta-CROWN auto_LiRPA/operators/nonlinear.py (ELU relaxation)
pub(crate) fn elu_linear_relaxation(l: f32, u: f32, alpha: f32) -> LinearRelaxation {
    // Guard: alpha < 0 makes ELU non-monotone; alpha = 0 is degenerate;
    // non-finite alpha causes NaN/Inf. Return conservative (-inf, +inf). (#2779)
    if !alpha.is_finite() || alpha <= 0.0 {
        return LinearRelaxation::nan_fallback();
    }

    let alpha64 = alpha as f64;
    let params = EluFamilyParams {
        positive_slope: 1.0,
        eval_negative: |x, p| p.scale * (x.exp() - 1.0),
        deriv_negative: |x, p| p.scale * x.exp(),
        saturation: -alpha,
        scale: alpha64,
        alpha: alpha64,
        globally_convex: false,
    };
    elu_family_linear_relaxation(l, u, &params)
}

impl EluLayer {
    impl_elementwise_activation!(
        @inherent_methods_stateful
        EluLayer,
        |layer: &EluLayer, l, u| elu_linear_relaxation(l, u, layer.alpha),
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::non_finite_domain_guard("ELU", pre_activation)
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearBounds;
    use ndarray::{Array1, Array2, ArrayD, IxDyn};
    use proptest::prelude::ProptestConfig;

    fn elu_eval(x: f32, alpha: f32) -> f32 {
        if x >= 0.0 {
            x
        } else {
            alpha * (x.exp() - 1.0)
        }
    }

    #[test]
    fn test_new_stores_alpha() {
        let layer = EluLayer::new(2.0);
        assert!((layer.alpha - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_try_new_rejects_invalid_alpha_2551() {
        for alpha in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let err = EluLayer::try_new(alpha).expect_err("invalid alpha should be rejected");
            assert!(matches!(err, NyError::InvalidSpec(_)));
        }
    }

    #[test]
    fn test_default_alpha() {
        let layer = EluLayer::default_alpha();
        assert!((layer.alpha - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_positive_region() {
        // l >= 0: identity
        let layer = EluLayer::new(1.0);
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
        )
        .unwrap();
        let out = layer.propagate_ibp(&input).unwrap();
        assert!((out.lower()[[0]] - 1.0).abs() < 1e-5);
        assert!((out.upper()[[2]] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_negative_region() {
        // u <= 0: y = alpha*(exp(x)-1)
        let layer = EluLayer::new(1.0);
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.5]).unwrap(),
        )
        .unwrap();
        let out = layer.propagate_ibp(&input).unwrap();
        let expected_lower = (-2.0f32).exp() - 1.0;
        let expected_upper = (-0.5f32).exp() - 1.0;
        assert!((out.lower()[[0]] - expected_lower).abs() < 1e-5);
        assert!((out.upper()[[0]] - expected_upper).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_crossing_region() {
        // l < 0 < u
        let layer = EluLayer::new(1.0);
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        )
        .unwrap();
        let out = layer.propagate_ibp(&input).unwrap();
        let expected_lower = (-1.0f32).exp() - 1.0; // ~-0.632
        assert!((out.lower()[[0]] - expected_lower).abs() < 1e-5);
        assert!((out.upper()[[0]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_ibp_soundness_grid() {
        let layer = EluLayer::new(1.5);
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-3.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        )
        .unwrap();
        let out = layer.propagate_ibp(&input).unwrap();

        for i in 0..51 {
            let x = -3.0 + (i as f32) * 0.1;
            let y = elu_eval(x, 1.5);
            assert!(
                out.lower()[[0]] <= y + 1e-5,
                "Lower {} > eval {} at x={}",
                out.lower()[[0]],
                y,
                x
            );
            assert!(
                out.upper()[[0]] >= y - 1e-5,
                "Upper {} < eval {} at x={}",
                out.upper()[[0]],
                y,
                x
            );
        }
    }

    #[test]
    fn test_ibp_point_input() {
        let layer = EluLayer::new(1.0);
        let vals = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 0.5]).unwrap();
        let input = BoundedTensor::new(vals.clone(), vals).unwrap();
        let out = layer.propagate_ibp(&input).unwrap();
        assert!((out.lower()[[0]] - out.upper()[[0]]).abs() < 1e-5);
        assert!((out.lower()[[1]] - 0.5).abs() < 1e-5);
    }

    // ========== Linear relaxation tests ==========

    #[test]
    fn test_relaxation_positive_exact() {
        // l >= 0: exact identity
        let r = elu_linear_relaxation(1.0, 3.0, 1.0);
        assert!((r.lower_slope - 1.0).abs() < 1e-5);
        assert!((r.lower_intercept - 0.0).abs() < 1e-5);
        assert!((r.upper_slope - 1.0).abs() < 1e-5);
        assert!((r.upper_intercept - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_relaxation_negative_soundness() {
        // u <= 0: convex region
        let r = elu_linear_relaxation(-2.0, -0.5, 1.0);

        // Verify bounds contain f(x) at grid points
        for i in 0..16 {
            let x = -2.0 + (i as f32) * 0.1;
            let y = elu_eval(x, 1.0);
            let lb = r.lower_slope * x + r.lower_intercept;
            let ub = r.upper_slope * x + r.upper_intercept;
            assert!(lb <= y + 1e-5, "Lower {} > {} at x={}", lb, y, x);
            assert!(ub >= y - 1e-5, "Upper {} < {} at x={}", ub, y, x);
        }
    }

    #[test]
    fn test_relaxation_negative_lower_tighter_than_midpoint() {
        // On a wide purely-negative interval the new parallel-to-chord tangent
        // is a strictly tighter lower bound than the old midpoint tangent in the
        // CROWN-relevant sense: it minimizes the maximum vertical gap (the
        // relaxation band width) between the lower line and the chord (upper
        // line). Both lines are tangents to the convex negative branch and so
        // are sound lower bounds, but the parallel-to-chord tangent shrinks the
        // worst-case band the most.
        let (l, u, alpha) = (-6.0f32, -0.5f32, 1.0f32);
        let r = elu_linear_relaxation(l, u, alpha);

        let l64 = l as f64;
        let u64 = u as f64;

        // Chord (upper line) and the legacy midpoint tangent (lower line).
        let chord_slope =
            (alpha as f64 * (u64.exp() - 1.0) - alpha as f64 * (l64.exp() - 1.0)) / (u64 - l64);
        let chord_int = alpha as f64 * (l64.exp() - 1.0) - chord_slope * l64;
        let m = f64::midpoint(l64, u64);
        let mid_slope = alpha as f64 * m.exp();
        let fm = alpha as f64 * (m.exp() - 1.0);
        let mid_intercept = fm - mid_slope * m;

        let mut new_band = 0.0f64;
        let mut mid_band = 0.0f64;
        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l64 + t * (u64 - l64);
            let fx = alpha as f64 * (x.exp() - 1.0); // x <= 0 throughout

            let new_lb = r.lower_slope as f64 * x + r.lower_intercept as f64;
            let mid_lb = mid_slope * x + mid_intercept;
            let chord = chord_slope * x + chord_int;

            // Sound: the new lower bound never exceeds ELU(x).
            assert!(
                new_lb <= fx + 1e-9,
                "new lower {} > ELU({})={}",
                new_lb,
                x,
                fx
            );
            new_band = new_band.max(chord - new_lb);
            mid_band = mid_band.max(chord - mid_lb);
        }
        assert!(
            new_band < mid_band,
            "parallel-to-chord tangent max band {} should be tighter than midpoint band {}",
            new_band,
            mid_band
        );
    }

    #[test]
    fn test_relaxation_crossing_soundness() {
        // l < 0 < u: crossing
        let r = elu_linear_relaxation(-2.0, 3.0, 1.0);

        for i in 0..51 {
            let x = -2.0 + (i as f32) * 0.1;
            let y = elu_eval(x, 1.0);
            let lb = r.lower_slope * x + r.lower_intercept;
            let ub = r.upper_slope * x + r.upper_intercept;
            assert!(lb <= y + 1e-3, "Lower {} > {} at x={}", lb, y, x);
            assert!(ub >= y - 1e-3, "Upper {} < {} at x={}", ub, y, x);
        }
    }

    #[test]
    fn test_relaxation_nan_lower_returns_sound() {
        let r = elu_linear_relaxation(f32::NAN, 1.0, 1.0);
        assert_eq!(r.lower_slope, 0.0, "NaN lower → slope 0");
        assert!(
            r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative(),
            "NaN lower → -inf intercept, got {}",
            r.lower_intercept
        );
        assert_eq!(r.upper_slope, 0.0, "NaN lower → slope 0");
        assert!(
            r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive(),
            "NaN lower → +inf intercept, got {}",
            r.upper_intercept
        );
    }

    #[test]
    fn test_relaxation_nan_upper_returns_sound() {
        let r = elu_linear_relaxation(-1.0, f32::NAN, 1.0);
        assert_eq!(r.lower_slope, 0.0, "NaN upper → slope 0");
        assert!(
            r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative(),
            "NaN upper → -inf intercept, got {}",
            r.lower_intercept
        );
        assert_eq!(r.upper_slope, 0.0, "NaN upper → slope 0");
        assert!(
            r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive(),
            "NaN upper → +inf intercept, got {}",
            r.upper_intercept
        );
    }

    #[test]
    fn test_relaxation_nan_both_returns_sound() {
        let r = elu_linear_relaxation(f32::NAN, f32::NAN, 1.0);
        assert_eq!(r.lower_slope, 0.0, "NaN both → slope 0");
        assert!(
            r.lower_intercept.is_infinite() && r.lower_intercept.is_sign_negative(),
            "NaN both → -inf intercept, got {}",
            r.lower_intercept
        );
        assert_eq!(r.upper_slope, 0.0, "NaN both → slope 0");
        assert!(
            r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive(),
            "NaN both → +inf intercept, got {}",
            r.upper_intercept
        );
    }

    #[test]
    fn test_relaxation_both_infinite_4204() {
        let alpha = 1.5;
        let r = elu_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY, alpha);
        assert_eq!(r.lower_slope, 0.0);
        assert!((r.lower_intercept + alpha).abs() < 1e-5);
        assert_eq!(r.upper_slope, 0.0);
        assert!(r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive());
    }

    #[test]
    fn test_relaxation_left_infinite_soundness_4204() {
        let alpha = 1.5;
        let r = elu_linear_relaxation(f32::NEG_INFINITY, 2.0, alpha);

        for x in [-100.0f32, -10.0, -1.0, 0.0, 1.0, 2.0] {
            let y = elu_eval(x, alpha);
            let lb = r.lower_slope * x + r.lower_intercept;
            let ub = r.upper_slope * x + r.upper_intercept;
            assert!(lb <= y + 1e-3, "left-inf lower {} > {} at x={}", lb, y, x);
            assert!(ub >= y - 1e-3, "left-inf upper {} < {} at x={}", ub, y, x);
        }
    }

    #[test]
    fn test_relaxation_right_infinite_soundness_4204() {
        let alpha = 1.5;
        let r = elu_linear_relaxation(-2.0, f32::INFINITY, alpha);
        assert!(r.upper_intercept.is_infinite() && r.upper_intercept.is_sign_positive());

        for x in [-2.0f32, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let y = elu_eval(x, alpha);
            let lb = r.lower_slope * x + r.lower_intercept;
            assert!(lb <= y + 1e-3, "right-inf lower {} > {} at x={}", lb, y, x);
        }
    }

    #[test]
    fn test_relaxation_alpha_gt_1() {
        // alpha > 1 changes the crossing dynamics
        let r = elu_linear_relaxation(-1.0, 1.0, 2.0);

        for i in 0..21 {
            let x = -1.0 + (i as f32) * 0.1;
            let y = elu_eval(x, 2.0);
            let lb = r.lower_slope * x + r.lower_intercept;
            let ub = r.upper_slope * x + r.upper_intercept;
            assert!(lb <= y + 1e-3, "Lower {} > {} at x={}", lb, y, x);
            assert!(ub >= y - 1e-3, "Upper {} < {} at x={}", ub, y, x);
        }
    }

    #[test]
    fn test_propagate_linear_returns_error() {
        let layer = EluLayer::new(1.0);
        let bounds = LinearBounds::new(
            Array2::eye(2),
            Array1::zeros(2),
            Array2::eye(2),
            Array1::zeros(2),
        )
        .unwrap();
        assert!(layer.propagate_linear(&bounds).is_err());
    }

    #[test]
    fn test_requires_pre_activation_bounds() {
        let layer = EluLayer::new(1.0);
        assert!(layer.requires_pre_activation_bounds());
    }

    #[test]
    fn test_crown_backward_soundness() {
        let layer = EluLayer::new(1.0);
        let pre_act = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        )
        .unwrap();
        let bounds = LinearBounds::new(
            Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();
        let result =
            BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act).unwrap();

        // Verify CROWN bounds contain ELU at grid points
        for i in 0..41 {
            let x = -2.0 + (i as f32) * 0.1;
            let y = elu_eval(x, 1.0);
            let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
            let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];
            assert!(lb <= y + 1e-3, "CROWN lower {} > {} at x={}", lb, y, x);
            assert!(ub >= y - 1e-3, "CROWN upper {} < {} at x={}", ub, y, x);
        }
    }

    #[test]
    fn test_crown_backward_no_pre_activation_errors() {
        let layer = EluLayer::new(1.0);
        let bounds = LinearBounds::new(
            Array2::eye(1),
            Array1::zeros(1),
            Array2::eye(1),
            Array1::zeros(1),
        )
        .unwrap();
        assert!(layer.propagate_crown_backward(&bounds, None).is_err());
    }

    // ========== Regression tests for #2779: negative/invalid alpha ==========

    /// Regression test for #2779: ELU IBP with negative alpha is unsound because
    /// f(x) = alpha*(exp(x)-1) with alpha < 0 is not monotone. The guard must
    /// reject non-positive alpha at propagation time.
    #[test]
    fn test_ibp_negative_alpha_returns_error_2779() {
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.5]).unwrap(),
        )
        .unwrap();

        // alpha = -1.0: f is not monotone in x < 0 region
        let layer_neg = EluLayer { alpha: -1.0 };
        let err = layer_neg
            .propagate_ibp(&input)
            .expect_err("negative alpha should error");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "negative alpha should be InvalidSpec, got: {err:?}"
        );

        // alpha = 0.0: degenerate, f(x < 0) = 0 constant
        let layer_zero = EluLayer { alpha: 0.0 };
        let err = layer_zero
            .propagate_ibp(&input)
            .expect_err("zero alpha should error");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "zero alpha should be InvalidSpec, got: {err:?}"
        );

        // alpha = NaN
        let layer_nan = EluLayer { alpha: f32::NAN };
        let err = layer_nan
            .propagate_ibp(&input)
            .expect_err("NaN alpha should error");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "NaN alpha should be InvalidSpec, got: {err:?}"
        );

        // alpha = Inf
        let layer_inf = EluLayer {
            alpha: f32::INFINITY,
        };
        let err = layer_inf
            .propagate_ibp(&input)
            .expect_err("Inf alpha should error");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "Inf alpha should be InvalidSpec, got: {err:?}"
        );
    }

    /// Regression test for #2779: elu_linear_relaxation with bad alpha returns
    /// conservative (nan_fallback) bounds instead of unsound results.
    #[test]
    fn test_relaxation_negative_alpha_returns_conservative_2779() {
        // alpha = -1.0
        let r = elu_linear_relaxation(-1.0, 1.0, -1.0);
        assert_eq!(r.lower_slope, 0.0);
        assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
        assert_eq!(r.upper_slope, 0.0);
        assert_eq!(r.upper_intercept, f32::INFINITY);

        // alpha = 0.0
        let r = elu_linear_relaxation(-1.0, 1.0, 0.0);
        assert_eq!(r.lower_slope, 0.0);
        assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
        assert_eq!(r.upper_slope, 0.0);
        assert_eq!(r.upper_intercept, f32::INFINITY);

        // alpha = NaN
        let r = elu_linear_relaxation(-1.0, 1.0, f32::NAN);
        assert_eq!(r.lower_slope, 0.0);
        assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
        assert_eq!(r.upper_slope, 0.0);
        assert_eq!(r.upper_intercept, f32::INFINITY);

        // alpha = -Inf
        let r = elu_linear_relaxation(-1.0, 1.0, f32::NEG_INFINITY);
        assert_eq!(r.lower_slope, 0.0);
        assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
        assert_eq!(r.upper_slope, 0.0);
        assert_eq!(r.upper_intercept, f32::INFINITY);
    }

    /// Regression test for #2779: verify that the specific example from the
    /// issue (alpha=-1, l=-2, u=-0.5) is rejected rather than producing
    /// lower > upper.
    #[test]
    fn test_ibp_negative_alpha_example_from_issue_2779() {
        let layer = EluLayer { alpha: -1.0 };
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.5]).unwrap(),
        )
        .unwrap();

        // Before the fix, this would compute:
        //   lower = f(-2) = -1*(exp(-2)-1) ≈ 0.865
        //   upper = f(-0.5) = -1*(exp(-0.5)-1) ≈ 0.394
        //   lower > upper → invariant violation
        // After the fix, this returns an error.
        assert!(
            layer.propagate_ibp(&input).is_err(),
            "negative alpha must be rejected to prevent lower > upper"
        );
    }

    /// Self-audit finding (#2779): CROWN backward with negative alpha must
    /// produce sound (±inf) bounds rather than unsound finite bounds. The
    /// domain_guard only checks pre-activation bounds, not alpha, so the
    /// guard in elu_linear_relaxation (nan_fallback) is the safety net.
    #[test]
    fn test_crown_backward_negative_alpha_produces_sound_bounds_2779() {
        let layer = EluLayer { alpha: -1.0 };
        let pre_act = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let bounds = LinearBounds::new(
            Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        // CROWN backward should succeed (nan_fallback returns ±inf, not error).
        // The result should have ±inf intercepts, which is sound.
        let result =
            BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act).unwrap();

        // The nan_fallback produces slope=0, intercept=±inf through CROWN backward.
        // After backward propagation, the bias terms should be ±inf.
        assert!(
            result.lower_b[0].is_infinite() && result.lower_b[0].is_sign_negative(),
            "negative alpha CROWN should produce -inf lower bias, got {}",
            result.lower_b[0]
        );
        assert!(
            result.upper_b[0].is_infinite() && result.upper_b[0].is_sign_positive(),
            "negative alpha CROWN should produce +inf upper bias, got {}",
            result.upper_b[0]
        );
    }

    // ── IBP guard regression tests (#3278) ────────────────────────────

    #[test]
    fn test_ibp_nan_input_lower_rejected_3278() {
        let layer = EluLayer::new(1.0);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("NaN input lower");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    #[test]
    fn test_ibp_nan_input_upper_rejected_3278() {
        let layer = EluLayer::new(1.0);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap(),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("NaN input upper");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    #[test]
    fn test_ibp_inf_input_rejected_3278() {
        let layer = EluLayer::new(1.0);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NEG_INFINITY]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::INFINITY]).unwrap(),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("Inf input");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    // ── NaN alpha absorption regression tests (#2714) ─────────────────

    #[test]
    fn test_relaxation_nan_alpha_returns_nan_fallback_2714() {
        // NaN alpha triggers the non-finite alpha guard and returns
        // conservative (-inf, +inf) nan_fallback — not 0.0 from silently
        // absorbed NaN via .max(0.0). (#2714 defense-in-depth)
        let r = elu_linear_relaxation(f32::NEG_INFINITY, 1.0, f32::NAN);
        assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
        assert_eq!(r.upper_intercept, f32::INFINITY);
    }

    #[test]
    fn test_relaxation_nan_alpha_crossing_returns_nan_fallback_2714() {
        // NaN alpha in crossing region also returns nan_fallback.
        let r = elu_linear_relaxation(-1.0, 1.0, f32::NAN);
        assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
        assert_eq!(r.upper_intercept, f32::INFINITY);
    }

    #[test]
    fn test_relaxation_inf_alpha_returns_nan_fallback_2714() {
        // Infinite alpha also triggers the guard (#2779).
        let r = elu_linear_relaxation(-1.0, 1.0, f32::INFINITY);
        assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
        assert_eq!(r.upper_intercept, f32::INFINITY);
    }

    // ── CROWN relaxation soundness proptest (#3285) ─────────────────────

    /// Reference ELU in f64, independent of the crate f32 implementation.
    fn elu_f64_reference(x: f64, alpha: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            alpha * (x.exp() - 1.0)
        }
    }

    proptest::proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

        /// #3285: Verify elu_linear_relaxation produces strictly sound bounds.
        /// For random intervals, the lower bound must satisfy
        ///   lower_slope * x + lower_intercept <= ELU(x)  for all x in [l, u]
        /// and the upper bound must satisfy
        ///   upper_slope * x + upper_intercept >= ELU(x)  for all x in [l, u]
        /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
        ///
        /// Ref: SiLU proptest_silu_relaxation_strict_soundness (silu/tests.rs:553).
        #[test]
        fn proptest_elu_relaxation_strict_soundness(
            l in -10.0f32..10.0,
            width in 0.01f32..20.0,
            alpha in 0.1f32..5.0,
        ) {
            let u = l + width;
            let relax = elu_linear_relaxation(l, u, alpha);
            let ls = relax.lower_slope;
            let li = relax.lower_intercept;
            let us = relax.upper_slope;
            let ui = relax.upper_intercept;

            // Skip NaN fallback (infinite bounds).
            proptest::prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

            let alpha64 = alpha as f64;

            // Dense grid: 200 points, evaluated in f64 for mathematical precision.
            for k in 0..=200 {
                let t = k as f64 / 200.0;
                let x = l as f64 + t * (u as f64 - l as f64);
                let x = x.clamp(l as f64, u as f64);
                let fx = elu_f64_reference(x, alpha64);

                let lower_val = ls as f64 * x + li as f64;
                proptest::prop_assert!(
                    lower_val <= fx,
                    "ELU lower bound UNSOUND at x={}: {} > ELU({})={}, \
                     interval=[{}, {}], alpha={}, gap={}", x, lower_val, x, fx, l, u, alpha, lower_val - fx
                );

                let upper_val = us as f64 * x + ui as f64;
                proptest::prop_assert!(
                    upper_val >= fx,
                    "ELU upper bound UNSOUND at x={}: {} < ELU({})={}, \
                     interval=[{}, {}], alpha={}, gap={}", x, upper_val, x, fx, l, u, alpha, fx - upper_val
                );
            }
        }
    }
}
