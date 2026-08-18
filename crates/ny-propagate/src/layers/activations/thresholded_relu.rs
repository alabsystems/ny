// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::{next_up_f32, BoundedTensor};

use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

use super::validate::validate_finite;
use super::LinearRelaxation;

/// ThresholdedRelu layer: y = x if x > alpha, else 0
///
/// Similar to ReLU but with a configurable threshold alpha (default: 1.0).
/// This is useful for sparse feature selection where only sufficiently
/// strong activations are passed through.
#[derive(Debug, Clone)]
pub struct ThresholdedReluLayer {
    /// Threshold value (default: 1.0)
    pub(crate) alpha: f32,
}

impl ThresholdedReluLayer {
    /// Validate and create a new ThresholdedRelu layer.
    ///
    /// Returns an error if `alpha` is NaN or infinite, since non-finite
    /// thresholds cause NaN comparisons in IBP (x > alpha is always false
    /// for NaN alpha). Part of #2551.
    pub fn try_new(alpha: f32) -> Result<Self> {
        Ok(Self {
            alpha: validate_finite(alpha, "ThresholdedReluLayer", "alpha")?,
        })
    }

    /// Create a new ThresholdedRelu layer with the given alpha.
    pub fn new(alpha: f32) -> Self {
        Self::try_new(alpha).expect("invariant: ThresholdedReluLayer::new requires validated alpha")
    }
}

impl Default for ThresholdedReluLayer {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl BoundPropagation for ThresholdedReluLayer {
    /// IBP for ThresholdedRelu: y = x if x > alpha, else 0
    ///
    /// - If lower > alpha: both bounds pass through unchanged
    /// - If upper <= alpha: output is [0, 0]
    /// - If lower <= alpha < upper: output is [0, upper]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Guard: non-finite input bounds → NaN comparisons always return false
        // (NaN > alpha = false, NaN <= alpha = false), so NaN falls to the
        // crossing case which pushes NaN u into upper bounds unchecked.
        // CROWN path rejects via non_finite_domain_guard (thresholded_relu.rs:92-94).
        // (#3203, Finding 3)
        if input.lower().iter().any(|x| !x.is_finite())
            || input.upper().iter().any(|x| !x.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "ThresholdedRelu IBP: non-finite input bounds".to_string(),
            ));
        }

        let alpha = self.alpha;

        let lower_shape = input.lower().shape().to_vec();
        let mut lower_data = Vec::with_capacity(input.lower().len());
        let mut upper_data = Vec::with_capacity(input.upper().len());

        for (&l, &u) in input.lower().iter().zip(input.upper().iter()) {
            if l > alpha {
                // Entirely above threshold: pass through
                lower_data.push(l);
                upper_data.push(u);
            } else if u <= alpha {
                // Entirely below or at threshold: output is 0
                lower_data.push(0.0);
                upper_data.push(0.0);
            } else {
                // Crosses threshold: lower could be 0, upper passes through
                lower_data.push(0.0);
                upper_data.push(u);
            }
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&lower_shape), lower_data)
            .map_err(|e| NyError::InvalidSpec(format!("ThresholdedRelu lower reshape: {}", e)))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&lower_shape), upper_data)
            .map_err(|e| NyError::InvalidSpec(format!("ThresholdedRelu upper reshape: {}", e)))?;

        BoundedTensor::new(lower, upper)
    }

    impl_elementwise_activation!(
        @trait_methods
        ThresholdedReluLayer,
        NyError::InvalidSpec(
            "ThresholdedRelu CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

impl ThresholdedReluLayer {
    impl_elementwise_activation!(
        @inherent_methods_stateful
        ThresholdedReluLayer,
        |layer: &ThresholdedReluLayer, l, u| thresholded_relu_linear_relaxation(l, u, layer.alpha),
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::non_finite_domain_guard("ThresholdedRelu", pre_activation)
        }
    );
}

/// ThresholdedReLU crossing-region relaxation: both l and u finite, l <= alpha < u.
///
/// Upper bound: identity y=x when l >= 0 (avoids catastrophic f32 cancellation
/// from steep slope); line through (l,0) when l < 0 with f64 directed rounding.
/// Lower bound: y = x - max(alpha, 0). Part of #3313.
pub(super) fn thresholded_relu_crossing(l: f32, u: f32, alpha: f32) -> LinearRelaxation {
    let (us, ui) = if l >= 0.0 {
        (1.0, 0.0)
    } else {
        // l < 0: identity invalid, use line through (l, 0).
        // Compute slope/intercept in f64 with directed rounding. Part of #3313.
        let denom = alpha - l;
        if denom.abs() < f32::EPSILON {
            // l ≈ alpha: slope_threshold = alpha/denom is unstable (Inf when
            // l == alpha exactly). Conservative fallback: slope=1, intercept=-l.
            // Restores guard from #1759 (commit e5554173).
            let li = if alpha > 0.0 { -alpha } else { 0.0 };
            return LinearRelaxation::new(1.0, li, 1.0, (-l).max(0.0));
        }
        let l_f64 = l as f64;
        let u_f64 = u as f64;
        let alpha_f64 = alpha as f64;
        let slope_ep = u_f64 / (u_f64 - l_f64);
        let slope_th = alpha_f64 / (alpha_f64 - l_f64);
        let lam_f64 = slope_ep.max(slope_th);
        let lam = lam_f64 as f32;
        // Recompute intercept from all constraint points:
        // y(l) >= 0, y(alpha) >= 0, y(u) >= u. Take max and round up.
        // When lam < 0 (both l, u negative), the dead-zone interior at alpha
        // is the binding constraint, not the endpoints. Part of #3321.
        let c_from_l = -(lam as f64) * l_f64;
        let c_from_alpha = -(lam as f64) * alpha_f64;
        let c_from_u = u_f64 - (lam as f64) * u_f64;
        let ui = next_up_f32(c_from_l.max(c_from_alpha).max(c_from_u) as f32);
        (lam, ui)
    };
    // Lower envelope: y >= x - max(alpha, 0).
    let li = if alpha > 0.0 { -alpha } else { 0.0 };
    LinearRelaxation::new(1.0, li, us, ui)
}

/// Compute CROWN linear relaxation for ThresholdedRelu on interval [l, u].
///
/// ThresholdedRelu: y = x if x > alpha, else 0
///
/// Cases:
/// - NaN: constant ±inf bounds (drives CROWN output to safe ±inf)
/// - l > alpha: identity (always active)
/// - u <= alpha: zero (always inactive)
/// - u = +inf: crossing with special inf-handling
/// - Crossing (finite): upper depends on sign of l
///
/// Upper bound strategy for finite crossing case:
/// - l >= 0: use identity y = x (valid since trelu(x) <= x for x >= 0;
///   avoids catastrophic f32 cancellation from steep alpha/(alpha-l) slope)
/// - l < 0: line through (l, 0) with slope >= max(u/(u-l), alpha/(alpha-l))
///
/// Returns a `LinearRelaxation` (lower_slope, lower_intercept, upper_slope, upper_intercept).
pub(super) fn thresholded_relu_linear_relaxation(l: f32, u: f32, alpha: f32) -> LinearRelaxation {
    if l.is_nan() || u.is_nan() || l.is_infinite() {
        // NaN or l=-inf: safe ±inf intercepts so CROWN drives bounds to ±inf.
        // l=-inf must be caught here; without this guard it falls through to the
        // crossing case where inf-in-denominator produces slope→0, yielding an unsound
        // upper bound of y=0 (#2334). u=+inf is handled explicitly below with tighter
        // relaxation math (the u.is_infinite() branch computes finite slopes).
        return LinearRelaxation::nan_fallback();
    }
    if (u - l).abs() < 1e-8 {
        // Denominator guard for near-point intervals. alpha-beta-CROWN's tensor
        // ReLU relaxation similarly floors (u-l) by +1e-8 before division
        // (auto_LiRPA/operators/relu.py::_relu_upper_bound).
        let y_l = if l > alpha { l } else { 0.0 };
        let y_u = if u > alpha { u } else { 0.0 };
        let mut y_min = y_l.min(y_u);
        // For alpha < 0 and l <= alpha < u, infimum on (alpha, u] is alpha.
        if alpha < 0.0 && l <= alpha && u > alpha {
            y_min = y_min.min(alpha);
        }
        let y_max = y_l.max(y_u);
        return LinearRelaxation::new(0.0, y_min, 0.0, y_max);
    }

    if l > alpha {
        // Always active: identity
        LinearRelaxation::identity()
    } else if u <= alpha {
        // Always inactive: zero
        LinearRelaxation::zero()
    } else if u.is_infinite() {
        // u = +inf: crossing case. lim u/(u-l) = 1 as u -> inf.
        let (us, ui) = if l >= 0.0 {
            (1.0, 0.0)
        } else {
            let slope_threshold = alpha / (alpha - l);
            let lam = 1.0_f32.max(slope_threshold);
            (lam, -lam * l)
        };
        let li = if alpha > 0.0 { -alpha } else { 0.0 };
        LinearRelaxation::new(1.0, li, us, ui)
    } else {
        // Crossing: l and u both finite, l <= alpha < u.
        thresholded_relu_crossing(l, u, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearBounds;
    use ndarray::{array, ArrayD, IxDyn};
    use proptest::prelude::ProptestConfig;

    fn trelu_eval(x: f32, alpha: f32) -> f32 {
        if x > alpha {
            x
        } else {
            0.0
        }
    }

    // ── Constructor validation tests (#2551) ────────────────────────────

    #[test]
    fn test_try_new_rejects_invalid_alpha_2551() {
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = ThresholdedReluLayer::try_new(alpha)
                .expect_err("non-finite alpha should be rejected");
            assert!(matches!(err, NyError::InvalidSpec(_)));
        }
    }

    #[test]
    fn test_try_new_accepts_valid_alpha_2551() {
        // ThresholdedRelu alpha can be any finite value (including negative and zero)
        for alpha in [0.0, 1.0, -1.0, 0.5, -100.0, 100.0] {
            ThresholdedReluLayer::try_new(alpha)
                .unwrap_or_else(|_| panic!("alpha={alpha} should be accepted"));
        }
    }

    // ── IBP tests ──────────────────────────────────────────────────────

    #[test]
    fn test_ibp_above_threshold() {
        let layer = ThresholdedReluLayer::new(1.0);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[3]), 2.0_f32),
            ArrayD::from_elem(IxDyn(&[3]), 5.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        for &v in result.lower().iter() {
            assert!((v - 2.0).abs() < 1e-5, "above threshold: pass through");
        }
        for &v in result.upper().iter() {
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ibp_below_threshold() {
        let layer = ThresholdedReluLayer::new(1.0);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[3]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[3]), 0.5_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        for &v in result.lower().iter() {
            assert!(v.abs() < 1e-5, "below threshold: zero");
        }
        for &v in result.upper().iter() {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_ibp_crossing_threshold() {
        let layer = ThresholdedReluLayer::new(1.0);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2]), 0.0_f32),
            ArrayD::from_elem(IxDyn(&[2]), 3.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        for &v in result.lower().iter() {
            assert!(v.abs() < 1e-5, "crossing: lower = 0");
        }
        for &v in result.upper().iter() {
            assert!((v - 3.0).abs() < 1e-5, "crossing: upper passes through");
        }
    }

    // ── CROWN backward tests ───────────────────────────────────────────

    #[test]
    fn test_crown_above_threshold() {
        let layer = ThresholdedReluLayer::new(1.0);
        let pre =
            BoundedTensor::new(array![2.0_f32].into_dyn(), array![5.0_f32].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        // Identity: slope=1, intercept=0
        assert!((result.lower_a[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((result.upper_a[[0, 0]] - 1.0).abs() < 1e-5);
        assert!(result.lower_b[0].abs() < 1e-5);
        assert!(result.upper_b[0].abs() < 1e-5);
    }

    #[test]
    fn test_crown_below_threshold() {
        let layer = ThresholdedReluLayer::new(1.0);
        let pre =
            BoundedTensor::new(array![-1.0_f32].into_dyn(), array![0.5_f32].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        // Zero: slope=0, intercept=0
        assert!(result.lower_a[[0, 0]].abs() < 1e-5);
        assert!(result.upper_a[[0, 0]].abs() < 1e-5);
        assert!(result.lower_b[0].abs() < 1e-5);
        assert!(result.upper_b[0].abs() < 1e-5);
    }

    #[test]
    fn test_crown_crossing_soundness() {
        // Crossing: l=0 < alpha=1 < u=3
        // trelu(x) = x if x > 1, else 0
        let layer = ThresholdedReluLayer::new(1.0);
        let l = 0.0_f32;
        let u = 3.0_f32;
        let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        // Sample and verify soundness
        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = trelu_eval(x, 1.0);
            let lower_bound = la * x + lb;
            let upper_bound = ua * x + ub;
            assert!(
                lower_bound <= y + 1e-5,
                "lb {} > trelu({}) = {} at x={}",
                lower_bound,
                x,
                y,
                x
            );
            assert!(
                upper_bound >= y - 1e-5,
                "ub {} < trelu({}) = {} at x={}",
                upper_bound,
                x,
                y,
                x
            );
        }
    }

    #[test]
    fn test_crown_crossing_negative_l() {
        // l=-2 < 0 < alpha=1 < u=3
        let layer = ThresholdedReluLayer::new(1.0);
        let l = -2.0_f32;
        let u = 3.0_f32;
        let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = trelu_eval(x, 1.0);
            let lower_bound = la * x + lb;
            let upper_bound = ua * x + ub;
            assert!(
                lower_bound <= y + 1e-5,
                "lb {} > trelu({}) = {} at x={} (l<0 path)",
                lower_bound,
                x,
                y,
                x
            );
            assert!(
                upper_bound >= y - 1e-5,
                "ub {} < trelu({}) = {} at x={} (l<0 path)",
                upper_bound,
                x,
                y,
                x
            );
        }
    }

    #[test]
    fn test_relaxation_near_point_crossing_guard_is_sound() {
        let alpha = 0.0_f32;
        let l = -1e-20_f32;
        let u = 1e-20_f32;
        let r = thresholded_relu_linear_relaxation(l, u, alpha);
        assert_eq!(
            r.lower_slope, 0.0,
            "near-point guard should return constant lower"
        );
        assert_eq!(
            r.upper_slope, 0.0,
            "near-point guard should return constant upper"
        );
        assert!(
            r.lower_intercept.is_finite() && r.upper_intercept.is_finite(),
            "near-point guard must avoid Inf/NaN coefficients"
        );
        for &x in &[l, 0.0, u] {
            let y = trelu_eval(x, alpha);
            assert!(
                r.lower_intercept <= y + 1e-12,
                "lower {} > y {} at x={}",
                r.lower_intercept,
                y,
                x
            );
            assert!(
                r.upper_intercept >= y - 1e-12,
                "upper {} < y {} at x={}",
                r.upper_intercept,
                y,
                x
            );
        }
    }

    #[test]
    fn test_relaxation_near_point_crossing_negative_alpha_guard_is_sound() {
        let alpha = -1e-9_f32;
        let l = -1.5e-9_f32;
        let u = -0.5e-9_f32;
        let r = thresholded_relu_linear_relaxation(l, u, alpha);
        assert_eq!(
            r.lower_slope, 0.0,
            "near-point guard should return constant lower"
        );
        assert_eq!(
            r.upper_slope, 0.0,
            "near-point guard should return constant upper"
        );
        assert!(
            r.lower_intercept.is_finite() && r.upper_intercept.is_finite(),
            "near-point guard must avoid Inf/NaN coefficients"
        );
        for &x in &[l, alpha, alpha + 1e-12_f32, u] {
            let y = trelu_eval(x, alpha);
            assert!(
                r.lower_intercept <= y + 1e-12,
                "lower {} > y {} at x={}",
                r.lower_intercept,
                y,
                x
            );
            assert!(
                r.upper_intercept >= y - 1e-12,
                "upper {} < y {} at x={}",
                r.upper_intercept,
                y,
                x
            );
        }
    }

    #[test]
    fn test_crown_multi_neuron_mixed() {
        // 3 neurons: above, below, crossing
        let layer = ThresholdedReluLayer::new(1.0);
        let pre = BoundedTensor::new(
            array![2.0_f32, -1.0, 0.0].into_dyn(),
            array![5.0_f32, 0.5, 3.0].into_dyn(),
        )
        .unwrap();
        let bounds = LinearBounds::identity(3);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        // Neuron 0 (above): identity
        assert!((result.lower_a[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((result.upper_a[[0, 0]] - 1.0).abs() < 1e-5);

        // Neuron 1 (below): zero
        assert!(result.lower_a[[1, 1]].abs() < 1e-5);
        assert!(result.upper_a[[1, 1]].abs() < 1e-5);

        // Neuron 2 (crossing): nonzero slopes
        assert!(result.lower_a[[2, 2]] > 0.0, "crossing lower slope > 0");
        assert!(result.upper_a[[2, 2]] > 0.0, "crossing upper slope > 0");
    }

    #[test]
    fn test_crown_zero_threshold() {
        // alpha=0: behaves like ReLU
        let layer = ThresholdedReluLayer::new(0.0);
        let l = -2.0_f32;
        let u = 3.0_f32;
        let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = trelu_eval(x, 0.0);
            assert!(la * x + lb <= y + 1e-5, "alpha=0 lb fail at x={}", x);
            assert!(ua * x + ub >= y - 1e-5, "alpha=0 ub fail at x={}", x);
        }
    }

    #[test]
    fn test_propagate_linear_requires_preact() {
        let layer = ThresholdedReluLayer::new(1.0);
        let bounds = LinearBounds::identity(1);
        let err = layer
            .propagate_linear(&bounds)
            .expect_err("requires pre-activation");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    // ── Regression: infinite lower bound (#2334) ─────────────────────
    //
    // When l = -inf, the relaxation must produce safe fallback bounds.
    // Without the Inf guard, the crossing region produces slope → 0 via
    // inf in the denominator, making the upper bound y = 0 UNSOUND
    // (ThresholdedReLU(1.5) = 1.5 > 0 when alpha = 1.0, u = 2.0).

    #[test]
    fn test_relaxation_inf_lower_returns_safe_fallback_2334() {
        // Test the relaxation function directly — the fix returns safe
        // ±inf intercepts that CROWN will propagate to ±inf bounds.
        let r = thresholded_relu_linear_relaxation(f32::NEG_INFINITY, 2.0, 1.0);

        // With the fix: slopes = 0, intercepts = ±inf (safe fallback).
        assert_eq!(r.lower_slope, 0.0, "lower slope must be 0 for l=-inf");
        assert_eq!(
            r.lower_intercept,
            f32::NEG_INFINITY,
            "lower intercept must be -inf"
        );
        assert_eq!(r.upper_slope, 0.0, "upper slope must be 0 for l=-inf");
        assert_eq!(
            r.upper_intercept,
            f32::INFINITY,
            "upper intercept must be +inf"
        );
    }

    #[test]
    fn test_relaxation_inf_lower_buggy_formula_would_be_unsound_2334() {
        // Demonstrate what the buggy crossing-region formula produces:
        // slope = u / (u - l) = 2 / (2 - (-inf)) = 2/inf = 0
        // intercept = -slope * l = -0 * (-inf) = NaN
        // This would make upper bound = 0 * x + NaN = NaN, which is
        // unsound because ThresholdedReLU(1.5) = 1.5 is a finite value.
        let l = f32::NEG_INFINITY;
        let u = 2.0_f32;
        let alpha = 1.0_f32;

        // Simulate the buggy crossing-region formula (without the guard):
        let buggy_slope = u / (u - l); // 2 / inf = 0
        let buggy_intercept = -buggy_slope * l; // -0 * (-inf) = NaN
        assert_eq!(buggy_slope, 0.0, "buggy slope should be 0");
        assert!(
            buggy_intercept.is_nan(),
            "buggy intercept should be NaN (0 * inf)"
        );

        // The buggy upper bound at x=1.5 would be NaN — unsound.
        let x = 1.5_f32;
        let y_true = trelu_eval(x, alpha);
        assert_eq!(y_true, 1.5, "ThresholdedReLU(1.5, alpha=1) = 1.5");
        let buggy_upper = buggy_slope * x + buggy_intercept;
        assert!(
            buggy_upper.is_nan(),
            "buggy formula produces NaN upper bound"
        );

        // The fix catches l=-inf BEFORE the crossing formula runs.
        let r = thresholded_relu_linear_relaxation(l, u, alpha);
        assert_eq!(r.upper_slope, 0.0);
        assert_eq!(r.upper_intercept, f32::INFINITY);
    }

    // ── Regression: l == alpha division-by-zero (#3088, reopens #1759) ──
    //
    // When l == alpha exactly and l < 0, denom = alpha - l = 0, so
    // alpha / denom = Inf. This Inf slope produces unsound Inf upper
    // bounds that propagate through CROWN backward. The epsilon guard
    // (originally added in e5554173, dropped in refactor a4e2d499)
    // returns a conservative fallback before the division.

    #[test]
    fn test_relaxation_l_eq_alpha_negative_returns_finite_3088() {
        // l = alpha = -5.0, u = 10.0: large u - l (15.0) passes the
        // near-point guard, but alpha - l = 0.0 triggers division by zero.
        let l = -5.0_f32;
        let u = 10.0_f32;
        let alpha = -5.0_f32;
        let r = thresholded_relu_linear_relaxation(l, u, alpha);

        // All coefficients must be finite (no Inf/NaN).
        assert!(
            r.lower_slope.is_finite(),
            "lower slope must be finite, got {}",
            r.lower_slope
        );
        assert!(
            r.lower_intercept.is_finite(),
            "lower intercept must be finite, got {}",
            r.lower_intercept
        );
        assert!(
            r.upper_slope.is_finite(),
            "upper slope must be finite, got {}",
            r.upper_slope
        );
        assert!(
            r.upper_intercept.is_finite(),
            "upper intercept must be finite, got {}",
            r.upper_intercept
        );

        // Verify soundness: bounds must contain true output on [l, u].
        for k in 0..=100 {
            let x = l + (u - l) * (k as f32 / 100.0);
            let y = trelu_eval(x, alpha);
            let lower_bound = r.lower_slope * x + r.lower_intercept;
            let upper_bound = r.upper_slope * x + r.upper_intercept;
            assert!(
                lower_bound <= y + 1e-5,
                "lb {} > trelu({}) = {} at x={} (l==alpha regression)",
                lower_bound,
                x,
                y,
                x
            );
            assert!(
                upper_bound >= y - 1e-5,
                "ub {} < trelu({}) = {} at x={} (l==alpha regression)",
                upper_bound,
                x,
                y,
                x
            );
        }
    }

    #[test]
    fn test_relaxation_l_near_alpha_negative_is_sound_3088() {
        // l just slightly less than alpha: denom is tiny but nonzero.
        // The epsilon guard should still fire for |denom| < f32::EPSILON.
        let alpha = -3.0_f32;
        let l = alpha - f32::EPSILON * 0.5; // within epsilon
        let u = 5.0_f32;
        let r = thresholded_relu_linear_relaxation(l, u, alpha);

        assert!(
            r.lower_slope.is_finite() && r.upper_slope.is_finite(),
            "slopes must be finite for near-alpha"
        );
        assert!(
            r.lower_intercept.is_finite() && r.upper_intercept.is_finite(),
            "intercepts must be finite for near-alpha"
        );

        // Soundness check
        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = trelu_eval(x, alpha);
            let lower_bound = r.lower_slope * x + r.lower_intercept;
            let upper_bound = r.upper_slope * x + r.upper_intercept;
            assert!(
                lower_bound <= y + 1e-5,
                "lb {} > trelu({}) = {} at x={}",
                lower_bound,
                x,
                y,
                x
            );
            assert!(
                upper_bound >= y - 1e-5,
                "ub {} < trelu({}) = {} at x={}",
                upper_bound,
                x,
                y,
                x
            );
        }
    }

    #[test]
    fn test_relaxation_l_eq_alpha_via_crown_is_sound_3088() {
        // End-to-end: verify through the CROWN propagation path (not just
        // the scalar relaxation function) that l == alpha doesn't produce
        // Inf bounds.
        let alpha = -5.0_f32;
        let l = alpha;
        let u = 10.0_f32;
        let layer = ThresholdedReluLayer::new(alpha);
        let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        assert!(
            la.is_finite(),
            "CROWN lower slope must be finite, got {}",
            la
        );
        assert!(
            lb.is_finite(),
            "CROWN lower intercept must be finite, got {}",
            lb
        );
        assert!(
            ua.is_finite(),
            "CROWN upper slope must be finite, got {}",
            ua
        );
        assert!(
            ub.is_finite(),
            "CROWN upper intercept must be finite, got {}",
            ub
        );

        // Soundness check across the interval
        for k in 0..=100 {
            let x = l + (u - l) * (k as f32 / 100.0);
            let y = trelu_eval(x, alpha);
            assert!(
                la * x + lb <= y + 1e-5,
                "CROWN lb {} > trelu({}) = {} at x={}",
                la * x + lb,
                x,
                y,
                x
            );
            assert!(
                ua * x + ub >= y - 1e-5,
                "CROWN ub {} < trelu({}) = {} at x={}",
                ua * x + ub,
                x,
                y,
                x
            );
        }
    }

    // ── IBP guard regression tests (#3203) ────────────────────────────

    #[test]
    fn test_ibp_nan_input_lower_rejected() {
        let layer = ThresholdedReluLayer::new(1.0);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_elem(IxDyn(&[2]), f32::NAN),
            ArrayD::from_elem(IxDyn(&[2]), 3.0_f32),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("NaN input lower");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    #[test]
    fn test_ibp_nan_input_upper_rejected() {
        let layer = ThresholdedReluLayer::new(1.0);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_elem(IxDyn(&[2]), 0.0_f32),
            ArrayD::from_elem(IxDyn(&[2]), f32::NAN),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("NaN input upper");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    #[test]
    fn test_ibp_inf_input_rejected() {
        let layer = ThresholdedReluLayer::new(1.0);
        let input = BoundedTensor::new_unchecked(
            ArrayD::from_elem(IxDyn(&[2]), f32::NEG_INFINITY),
            ArrayD::from_elem(IxDyn(&[2]), 3.0_f32),
        )
        .unwrap();
        let err = layer.propagate_ibp(&input).expect_err("Inf input");
        assert!(matches!(err, NyError::NumericalInstability(_)));
    }

    // ── CROWN relaxation soundness proptest (#3321) ─────────────────────

    /// Reference ThresholdedReLU in f64, independent of the crate f32 implementation.
    fn thresholded_relu_f64_reference(x: f64, alpha: f64) -> f64 {
        if x > alpha {
            x
        } else {
            0.0
        }
    }

    proptest::proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

        /// #3321: Verify thresholded_relu_linear_relaxation produces strictly sound bounds.
        /// For random intervals, the lower bound must satisfy
        ///   lower_slope * x + lower_intercept <= ThresholdedReLU(x)  for all x in [l, u]
        /// and the upper bound must satisfy
        ///   upper_slope * x + upper_intercept >= ThresholdedReLU(x)  for all x in [l, u]
        /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
        ///
        /// Ref: ELU proptest_elu_relaxation_strict_soundness (elu.rs:841).
        #[test]
        fn proptest_thresholded_relu_relaxation_strict_soundness(
            l in -10.0f32..10.0,
            width in 0.01f32..20.0,
            alpha in -5.0f32..5.0,
        ) {
            let u = l + width;
            let relax = thresholded_relu_linear_relaxation(l, u, alpha);
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
                let fx = thresholded_relu_f64_reference(x, alpha64);

                let lower_val = ls as f64 * x + li as f64;
                proptest::prop_assert!(
                    lower_val <= fx,
                    "ThresholdedReLU lower bound UNSOUND at x={}: {} > ThresholdedReLU({})={}, \
                     interval=[{}, {}], alpha={}, gap={}", x, lower_val, x, fx, l, u, alpha, lower_val - fx
                );

                let upper_val = us as f64 * x + ui as f64;
                proptest::prop_assert!(
                    upper_val >= fx,
                    "ThresholdedReLU upper bound UNSOUND at x={}: {} < ThresholdedReLU({})={}, \
                     interval=[{}, {}], alpha={}, gap={}", x, upper_val, x, fx, l, u, alpha, fx - upper_val
                );
            }
        }
    }
}
