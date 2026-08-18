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

/// CELU (Continuous ELU) layer: max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
///
/// CELU is a smooth, continuously differentiable variant of ELU.
/// Unlike ELU, CELU is differentiable everywhere (including at x=0).
/// The parameter alpha controls the saturation value for negative inputs.
#[derive(Debug, Clone)]
pub struct CeluLayer {
    /// Scale parameter for the exponential (default: 1.0)
    pub(crate) alpha: f32,
}

impl CeluLayer {
    /// Validate and create a new CELU layer with the given alpha.
    pub fn try_new(alpha: f32) -> Result<Self> {
        Ok(Self {
            alpha: validate_positive_finite(alpha, "CeluLayer", "alpha")?,
        })
    }

    /// Create a new CELU layer with the given alpha.
    pub fn new(alpha: f32) -> Self {
        Self::try_new(alpha).expect("invariant: CeluLayer::new requires validated alpha")
    }

    /// Create a CELU layer with default alpha = 1.0.
    pub fn default_alpha() -> Self {
        Self::new(1.0)
    }
}

impl Default for CeluLayer {
    fn default() -> Self {
        Self::default_alpha()
    }
}

impl BoundPropagation for CeluLayer {
    /// IBP for CELU: y = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    ///
    /// CELU is monotonically increasing, so bounds map directly.
    /// For x >= 0: y = x
    /// For x < 0: y = alpha * (exp(x/alpha) - 1) (approaches -alpha as x -> -inf)
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let alpha = self.alpha;
        // Guard: alpha near zero causes x/alpha overflow to Inf, then
        // alpha * Inf = Inf or 0 * Inf = NaN. Reject non-positive or
        // non-finite alpha. (#2911)
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(NyError::InvalidSpec(format!(
                "CeluLayer requires finite positive alpha, got {}",
                alpha,
            )));
        }
        // Guard: NaN input bounds → NaN.exp() = NaN flows silently into output
        // bounds. NaN ONLY — ±Inf is a legitimate input here. An upstream node
        // that failed closed to an OpaqueSkip hands its consumers `[-inf, +inf]`
        // (`OpaqueSkipLayer::unbounded_like` builds exactly that); rejecting it
        // as `NumericalInstability` aborted the WHOLE graph-IBP pass, because
        // that variant is not in `is_degradable_error`. Pattern: AddConstant
        // (add_constant.rs:69-79). (#3278)
        if input.lower().iter().any(|x| x.is_nan()) || input.upper().iter().any(|x| x.is_nan()) {
            return Err(NyError::NumericalInstability(
                "CELU IBP: NaN input bounds".to_string(),
            ));
        }
        // Directed rounding: for x < 0, exp is a transcendental that can round
        // either direction. Compute in f64, cast with next_down/next_up. (#1483)
        // For x >= 0, CELU(x) = x is exact (no rounding needed).
        let alpha64 = alpha as f64;
        let celu_lower = |x: f32| -> f32 {
            if x >= 0.0 {
                x
            } else {
                // Range clamp: CELU(x) >= -alpha for all x. next_down_f32 can push
                // past -alpha for extreme negative inputs (exp(x/alpha) → 0). (#3316)
                // NaN-propagating: .max() swallows NaN (IEEE 754-2008). (#3316)
                nan_propagating_max(
                    next_down_f32((alpha64 * ((x as f64 / alpha64).exp() - 1.0)) as f32),
                    -alpha,
                )
            }
        };
        let celu_upper = |x: f32| -> f32 {
            if x >= 0.0 {
                x
            } else {
                next_up_f32((alpha64 * ((x as f64 / alpha64).exp() - 1.0)) as f32)
            }
        };
        // CELU is monotonically increasing, so bounds map directly
        let lower = input.lower().mapv(celu_lower);
        let upper = input.upper().mapv(celu_upper);
        // `new_allow_infinite`, not the strict `new`: with alpha guarded finite
        // and > 0 above, ±Inf evaluates cleanly and no repair is needed.
        //   x = +inf → the `x >= 0` identity branch → +inf (no exp call).
        //   x = -inf → (-inf / alpha) = -inf, exp(-inf) = 0, alpha * (0 - 1)
        //              = -alpha, CELU's exact infimum (the lower clamp then
        //              pins next_down_f32(-alpha) back to -alpha).
        // None of the NaN-producing inf patterns is reachable: no inf - inf
        // (the subtraction is exp(..) - 1 with exp(..) in [0, inf)), no
        // 0 * inf and no inf / inf (alpha is finite and non-zero). So NaN can
        // only come from a NaN input, which the guard above already rejects,
        // and NaN in the output still hard-errors here. (#3278)
        BoundedTensor::new_allow_infinite(lower, upper)
    }
    impl_elementwise_activation!(
        @trait_methods
        CeluLayer,
        NyError::InvalidSpec(
            "CELU CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

/// Analytical linear relaxation for CELU on interval [l, u].
///
/// Returns bounds such that:
///   lower_slope * x + lower_intercept <= CELU(x) <= upper_slope * x + upper_intercept
/// for all x in [l, u].
///
/// Delegates to the shared ELU-family relaxation (elu_family.rs) parameterized with
/// CELU's mathematical expressions. CELU is globally convex (f'(0-) = 1 = f'(0+)),
/// so the crossing case uses chord upper + tangent lower (tighter than chord+deviation).
/// Part of #2834.
///
/// Ref: alpha-beta-CROWN auto_LiRPA/operators/nonlinear.py (CELU relaxation)
pub(crate) fn celu_linear_relaxation(l: f32, u: f32, alpha: f32) -> LinearRelaxation {
    // Guard: non-positive or non-finite alpha causes x/alpha overflow. (#2911)
    if !alpha.is_finite() || alpha <= 0.0 {
        return LinearRelaxation::nan_fallback();
    }

    let alpha64 = alpha as f64;
    let params = EluFamilyParams {
        positive_slope: 1.0,
        eval_negative: |x, p| p.alpha * ((x / p.alpha).exp() - 1.0),
        deriv_negative: |x, p| (x / p.alpha).exp(),
        saturation: -alpha,
        scale: alpha64, // Not used in critical point (globally_convex = true)
        alpha: alpha64,
        globally_convex: true,
    };
    elu_family_linear_relaxation(l, u, &params)
}

impl CeluLayer {
    impl_elementwise_activation!(
        @inherent_methods_stateful
        CeluLayer,
        |layer: &CeluLayer, l, u| celu_linear_relaxation(l, u, layer.alpha),
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::non_finite_domain_guard("CELU", pre_activation)
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearBounds;
    use ndarray::{array, ArrayD, IxDyn};
    use proptest::prelude::*;

    fn celu_eval(x: f32, alpha: f32) -> f32 {
        if x >= 0.0 {
            x
        } else {
            alpha * ((x / alpha).exp() - 1.0)
        }
    }

    /// Independent f64 CELU reference for strict proptest. (#3292)
    fn celu_f64_reference(x: f64, alpha: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            alpha * ((x / alpha).exp() - 1.0)
        }
    }

    // ── Relaxation function tests ──────────────────────────────────────

    #[test]
    fn test_relaxation_positive() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = celu_linear_relaxation(1.0, 5.0, 1.0);
        assert!((ls - 1.0).abs() < 1e-5, "positive: identity slope");
        assert!(li.abs() < 1e-5);
        assert!((us - 1.0).abs() < 1e-5);
        assert!(ui.abs() < 1e-5);
    }

    #[test]
    fn test_relaxation_negative() {
        // Entirely negative: CELU is convex → chord above, tangent below
        let LinearRelaxation {
            lower_slope: ls,
            upper_slope: us,
            ..
        } = celu_linear_relaxation(-5.0, -1.0, 1.0);
        // Chord slope = (f(-1) - f(-5)) / (-1 - (-5))
        let fl = celu_eval(-5.0, 1.0);
        let fu = celu_eval(-1.0, 1.0);
        let expected_chord = (fu - fl) / 4.0;
        assert!(
            (us - expected_chord).abs() < 1e-4,
            "negative: upper = chord"
        );
        // Lower slope = tangent at midpoint
        assert!(ls > 0.0, "lower slope should be positive");
        assert!(
            ls <= us + 1e-4,
            "lower slope <= upper slope for convex function"
        );
    }

    #[test]
    fn test_try_new_rejects_invalid_alpha_2551() {
        for alpha in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let err = CeluLayer::try_new(alpha).expect_err("invalid alpha should be rejected");
            assert!(matches!(err, NyError::InvalidSpec(_)));
        }
    }

    #[test]
    fn test_relaxation_crossing() {
        // Crossing: l < 0 < u
        let LinearRelaxation {
            lower_slope: ls,
            upper_slope: us,
            ..
        } = celu_linear_relaxation(-2.0, 3.0, 1.0);
        // Upper = chord, lower = tangent (convex function)
        let fl = celu_eval(-2.0, 1.0);
        let fu = celu_eval(3.0, 1.0);
        let expected_chord = (fu - fl) / 5.0;
        assert!(
            (us - expected_chord).abs() < 1e-4,
            "crossing: upper = chord"
        );
        assert!(ls > 0.0, "lower slope positive");
    }

    #[test]
    fn test_relaxation_nan() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = celu_linear_relaxation(f32::NAN, 1.0, 1.0);
        assert!(ls.abs() < 1e-6);
        assert!(li.is_infinite() && li < 0.0);
        assert!(us.abs() < 1e-6);
        assert!(ui.is_infinite() && ui > 0.0);
    }

    #[test]
    fn test_relaxation_both_infinite_4204() {
        let alpha = 0.75;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = celu_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY, alpha);
        assert_eq!(ls, 0.0);
        assert!((li + alpha).abs() < 1e-5);
        assert_eq!(us, 0.0);
        assert!(ui.is_infinite() && ui.is_sign_positive());
    }

    #[test]
    fn test_relaxation_left_infinite_soundness_4204() {
        let alpha = 0.75;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = celu_linear_relaxation(f32::NEG_INFINITY, 2.0, alpha);

        for x in [-100.0f32, -10.0, -1.0, 0.0, 1.0, 2.0] {
            let y = celu_eval(x, alpha);
            let lb = ls * x + li;
            let ub = us * x + ui;
            assert!(lb <= y + 1e-3, "left-inf lower {} > {} at x={}", lb, y, x);
            assert!(ub >= y - 1e-3, "left-inf upper {} < {} at x={}", ub, y, x);
        }
    }

    #[test]
    fn test_relaxation_right_infinite_soundness_4204() {
        let alpha = 0.75;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_intercept: ui,
            ..
        } = celu_linear_relaxation(-2.0, f32::INFINITY, alpha);
        assert!(ui.is_infinite() && ui.is_sign_positive());

        for x in [-2.0f32, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let y = celu_eval(x, alpha);
            let lb = ls * x + li;
            assert!(lb <= y + 1e-3, "right-inf lower {} > {} at x={}", lb, y, x);
        }
    }

    #[test]
    fn test_relaxation_point_interval() {
        // l ≈ u: tangent at the point
        let LinearRelaxation {
            lower_slope: ls,
            upper_slope: us,
            ..
        } = celu_linear_relaxation(-1.0, -1.0, 1.0);
        // f'(-1) = exp(-1/1) ≈ 0.3679
        let expected_slope = (-1.0_f32).exp();
        assert!((ls - expected_slope).abs() < 1e-3, "point: tangent slope");
        assert!((us - expected_slope).abs() < 1e-3);
    }

    // ── Relaxation soundness ───────────────────────────────────────────

    #[test]
    fn test_relaxation_soundness_grid() {
        let intervals: &[(f32, f32)] = &[
            (-5.0, -1.0),
            (-2.0, 3.0),
            (0.0, 5.0),
            (-0.01, 0.01),
            (-10.0, 0.5),
        ];
        let alphas = [0.5, 1.0, 2.0];

        for &alpha in &alphas {
            for &(l, u) in intervals {
                let LinearRelaxation {
                    lower_slope: ls,
                    lower_intercept: li,
                    upper_slope: us,
                    upper_intercept: ui,
                } = celu_linear_relaxation(l, u, alpha);
                for k in 0..=50 {
                    let x = l + (u - l) * (k as f32 / 50.0);
                    let y = celu_eval(x, alpha);
                    let lower_bound = ls * x + li;
                    let upper_bound = us * x + ui;
                    assert!(
                        lower_bound <= y + 1e-3,
                        "alpha={} [{},{}] x={}: lb {} > y {}",
                        alpha,
                        l,
                        u,
                        x,
                        lower_bound,
                        y
                    );
                    assert!(
                        upper_bound >= y - 1e-3,
                        "alpha={} [{},{}] x={}: ub {} < y {}",
                        alpha,
                        l,
                        u,
                        x,
                        upper_bound,
                        y
                    );
                }
            }
        }
    }

    // ── IBP tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ibp_positive() {
        let layer = CeluLayer::new(1.0);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[3]), 1.0_f32),
            ArrayD::from_elem(IxDyn(&[3]), 5.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        // CELU(x) = x for x >= 0
        for &v in result.lower().iter() {
            assert!((v - 1.0).abs() < 1e-5);
        }
        for &v in result.upper().iter() {
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ibp_negative() {
        let alpha = 1.0;
        let layer = CeluLayer::new(alpha);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2]), -3.0_f32),
            ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        let expected_lo = celu_eval(-3.0, alpha);
        let expected_hi = celu_eval(-1.0, alpha);
        for &v in result.lower().iter() {
            assert!((v - expected_lo).abs() < 1e-4);
        }
        for &v in result.upper().iter() {
            assert!((v - expected_hi).abs() < 1e-4);
        }
    }

    #[test]
    fn test_ibp_crossing() {
        let alpha = 1.0;
        let layer = CeluLayer::new(alpha);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2]), -2.0_f32),
            ArrayD::from_elem(IxDyn(&[2]), 3.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        let expected_lo = celu_eval(-2.0, alpha);
        for &v in result.lower().iter() {
            assert!((v - expected_lo).abs() < 1e-4);
        }
        for &v in result.upper().iter() {
            assert!((v - 3.0).abs() < 1e-5); // CELU(3) = 3
        }
    }

    // ── CROWN backward tests ───────────────────────────────────────────

    #[test]
    fn test_crown_positive_preact() {
        let layer = CeluLayer::new(1.0);
        let pre =
            BoundedTensor::new(array![1.0_f32].into_dyn(), array![5.0_f32].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        assert!(
            (result.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
            "positive: identity"
        );
        assert!((result.upper_a[[0, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_crown_crossing_soundness() {
        let alpha = 1.0;
        let layer = CeluLayer::new(alpha);
        let l = -3.0_f32;
        let u = 2.0_f32;
        let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = celu_eval(x, alpha);
            let lower_bound = la * x + lb;
            let upper_bound = ua * x + ub;
            assert!(
                lower_bound <= y + 1e-3,
                "lb {} > y {} at x={}",
                lower_bound,
                y,
                x
            );
            assert!(
                upper_bound >= y - 1e-3,
                "ub {} < y {} at x={}",
                upper_bound,
                y,
                x
            );
        }
    }

    #[test]
    fn test_crown_different_alpha() {
        // Test with alpha=0.5 to verify alpha parameter is used
        let alpha = 0.5;
        let layer = CeluLayer::new(alpha);
        let l = -2.0_f32;
        let u = 1.0_f32;
        let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = celu_eval(x, alpha);
            assert!(la * x + lb <= y + 1e-3, "alpha=0.5 lb fails at x={}", x);
            assert!(ua * x + ub >= y - 1e-3, "alpha=0.5 ub fails at x={}", x);
        }
    }

    #[test]
    fn test_propagate_linear_requires_preact() {
        let layer = CeluLayer::new(1.0);
        let bounds = LinearBounds::identity(1);
        let err = layer
            .propagate_linear(&bounds)
            .expect_err("requires pre-activation");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    /// Regression test for #2911: near-zero alpha causes x/alpha overflow.
    #[test]
    fn test_ibp_bad_alpha_returns_error_2911() {
        let input =
            BoundedTensor::new(array![-1.0f32].into_dyn(), array![1.0f32].into_dyn()).unwrap();

        // alpha = 0 causes division by zero
        let layer_zero = CeluLayer { alpha: 0.0 };
        let err = layer_zero
            .propagate_ibp(&input)
            .expect_err("alpha=0 should error");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "alpha=0 should be InvalidSpec, got: {err:?}"
        );

        // alpha = -1 is invalid (CELU requires alpha > 0)
        let layer_neg = CeluLayer { alpha: -1.0 };
        let err = layer_neg
            .propagate_ibp(&input)
            .expect_err("negative alpha should error");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "negative alpha should be InvalidSpec, got: {err:?}"
        );

        // alpha = NaN
        let layer_nan = CeluLayer { alpha: f32::NAN };
        let err = layer_nan
            .propagate_ibp(&input)
            .expect_err("NaN alpha should error");
        assert!(
            matches!(err, NyError::InvalidSpec(_)),
            "NaN alpha should be InvalidSpec, got: {err:?}"
        );

        // alpha = Inf
        let layer_inf = CeluLayer {
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

    /// Regression test for #2911: celu_linear_relaxation with bad alpha.
    #[test]
    fn test_relaxation_bad_alpha_returns_conservative_2911() {
        // alpha = 0: should return conservative bounds
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = celu_linear_relaxation(-1.0, 1.0, 0.0);
        assert_eq!(ls, 0.0);
        assert_eq!(li, f32::NEG_INFINITY);
        assert_eq!(us, 0.0);
        assert_eq!(ui, f32::INFINITY);

        // alpha = -1
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = celu_linear_relaxation(-1.0, 1.0, -1.0);
        assert_eq!(ls, 0.0);
        assert_eq!(li, f32::NEG_INFINITY);
        assert_eq!(us, 0.0);
        assert_eq!(ui, f32::INFINITY);
    }

    // ── IBP guard regression tests (#3278) ────────────────────────────

    #[test]
    fn test_ibp_nan_input_lower_rejected_3278() {
        let layer = CeluLayer::default_alpha();
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
        let layer = CeluLayer::default_alpha();
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap(),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("NaN input upper");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    /// #3278 originally rejected ±Inf here too. That was wrong: ±Inf is what an
    /// upstream OpaqueSkip legitimately emits, and `NumericalInstability` is not
    /// degradable, so one tainted element aborted the whole graph-IBP pass.
    /// CELU must now widen instead — saturating to -alpha on the left.
    #[test]
    fn test_ibp_inf_input_propagates_widened_3278() {
        let alpha = 1.0_f32;
        let layer = CeluLayer::new(alpha);
        let input = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NEG_INFINITY]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::INFINITY]).unwrap(),
        )
        .unwrap();
        let out = layer
            .propagate_ibp(&input)
            .expect("a tainted element must widen, not abort the pass");
        // CELU(x) ∈ (-alpha, +inf): the lower saturates at the exact infimum.
        assert_eq!(out.lower()[[0]], -alpha);
        assert_eq!(out.upper()[[0]], f32::INFINITY);
    }

    // ── OpaqueSkip taint propagation (#3278 follow-up) ─────────────────

    /// Probe: a mixed tensor where one element carries an upstream OpaqueSkip's
    /// `[-inf, +inf]` and the other is a normal finite interval. The tainted
    /// element must widen; the finite element must keep its exact bounds.
    #[test]
    fn test_ibp_opaque_skip_taint_widens_only_tainted_element() {
        let alpha = 1.5_f32;
        let layer = CeluLayer::new(alpha);
        let input = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 1.0]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 2.0]).unwrap(),
        )
        .unwrap();
        let out = layer
            .propagate_ibp(&input)
            .expect("[-inf, +inf] is a sound enclosure, not an error");

        assert_eq!(out.lower()[[0]], -alpha);
        assert_eq!(out.upper()[[0]], f32::INFINITY);
        // Identity branch is exact for x >= 0.
        assert!((out.lower()[[1]] - 1.0).abs() < 1e-6);
        assert!((out.upper()[[1]] - 2.0).abs() < 1e-6);
    }

    /// Soundness: the widened output must still enclose CELU over the whole
    /// (unbounded) input interval, including the left saturation limit.
    #[test]
    fn test_ibp_inf_input_output_encloses_true_range() {
        let alpha = 0.75_f32;
        let layer = CeluLayer::new(alpha);
        let input = BoundedTensor::new_allow_infinite(
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NEG_INFINITY]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::INFINITY]).unwrap(),
        )
        .unwrap();
        let out = layer.propagate_ibp(&input).unwrap();
        for x in [-1e30f32, -100.0, -10.0, -1.0, 0.0, 1.0, 100.0, 1e30] {
            let y = celu_eval(x, alpha);
            assert!(
                out.lower()[[0]] <= y,
                "lower {} > CELU({x})={y}",
                out.lower()[[0]]
            );
            assert!(
                out.upper()[[0]] >= y,
                "upper {} < CELU({x})={y}",
                out.upper()[[0]]
            );
        }
    }

    /// The relaxation must NOT relax the NaN firewall: NaN from finite inputs
    /// (or any NaN endpoint) is still a hard error.
    #[test]
    fn test_ibp_nan_still_rejected_after_inf_relaxation() {
        let layer = CeluLayer::default_alpha();
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, f32::NAN]).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap(),
        )
        .unwrap();
        let err = layer
            .propagate_ibp(&input)
            .expect_err("NaN must not be absorbed alongside a tainted element");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    // ── Strict zero-tolerance CROWN relaxation proptest (#3292) ──────────
    //
    // Pattern from #3285: f64-evaluated reference with zero tolerance catches
    // f32 cancellation bugs invisible to magnitude-scaled tolerance tests.

    proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

        /// Strict soundness proptest for CELU CROWN relaxation (alpha=1.0).
        /// Uses f64 reference (celu_f64_reference) with zero tolerance on 200-point grid.
        /// Ref: alpha-beta-CROWN auto_LiRPA CELU relaxation, #3292.
        #[test]
        fn proptest_celu_relaxation_strict_soundness(
            l in -10.0f32..10.0,
            width in 0.01f32..20.0,
        ) {
            let u = l + width;
            let alpha = 1.0_f32;
            let relax = celu_linear_relaxation(l, u, alpha);
            let ls = relax.lower_slope;
            let li = relax.lower_intercept;
            let us = relax.upper_slope;
            let ui = relax.upper_intercept;

            prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

            for k in 0..=200 {
                let t = k as f64 / 200.0;
                let x = l as f64 + t * (u as f64 - l as f64);
                let x = x.clamp(l as f64, u as f64);
                let fx = celu_f64_reference(x, alpha as f64);

                let lower_val = ls as f64 * x + li as f64;
                prop_assert!(
                    lower_val <= fx,
                    "CELU lower bound UNSOUND at x={x}: {lower_val} > CELU({x})={fx}, \
                     interval=[{l}, {u}], alpha={alpha}, gap={}", lower_val - fx
                );

                let upper_val = us as f64 * x + ui as f64;
                prop_assert!(
                    upper_val >= fx,
                    "CELU upper bound UNSOUND at x={x}: {upper_val} < CELU({x})={fx}, \
                     interval=[{l}, {u}], alpha={alpha}, gap={}", fx - upper_val
                );
            }
        }
    }
}
