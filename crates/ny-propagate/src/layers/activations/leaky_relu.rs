// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::Zip;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::bounds::{nan_propagating_max, nan_propagating_min, safe_mul_for_bounds};
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

use super::validate::validate_finite;
use super::LinearRelaxation;

/// Leaky ReLU layer: y = x if x >= 0, else alpha * x
///
/// Leaky ReLU allows a small gradient for negative inputs, which helps prevent
/// "dying ReLU" problems during training. Typical alpha values are 0.01 or 0.1.
#[derive(Debug, Clone)]
pub struct LeakyReLULayer {
    /// Negative slope (typically 0.01)
    pub(crate) alpha: f32,
}

#[inline]
fn leaky_relu_linear_relaxation(l: f32, u: f32, alpha: f32) -> LinearRelaxation {
    if l.is_nan() || u.is_nan() || !alpha.is_finite() {
        return LinearRelaxation::nan_fallback();
    }
    if (u - l).abs() < 1e-8 {
        // Denominator guard for near-point intervals. alpha-beta-CROWN's tensor
        // ReLU path also protects the (u-l) denominator with a +1e-8 floor
        // before division (auto_LiRPA/operators/relu.py::_relu_upper_bound).
        let y_l = if l >= 0.0 { l } else { alpha * l };
        let y_u = if u >= 0.0 { u } else { alpha * u };
        let mut y_min = nan_propagating_min(y_l, y_u);
        let mut y_max = nan_propagating_max(y_l, y_u);
        // For alpha < 0 and l < 0 < u, LeakyReLU has a cusp minimum at x=0.
        if l < 0.0 && u > 0.0 {
            y_min = nan_propagating_min(y_min, 0.0);
            y_max = nan_propagating_max(y_max, 0.0);
        }
        return LinearRelaxation::new(0.0, y_min, 0.0, y_max);
    }
    if l >= 0.0 {
        // Always positive: identity
        LinearRelaxation::identity()
    } else if u <= 0.0 {
        // Always negative: scaled by alpha
        LinearRelaxation::new(alpha, 0.0, alpha, 0.0)
    } else if l.is_infinite() && u.is_infinite() {
        // Both infinite: f(x) = x for x>=0, alpha*x for x<0.
        // When alpha <= 1: y = alpha*x is a global lower bound (alpha*x <= x for x>=0,
        //   exact for x<0). Upper: +inf.
        // When alpha > 1: NO finite affine lower bound exists on (-inf,+inf).
        //   Proof: need s <= 1 (from x>=0 branch) and s >= alpha (from x<0 as x -> -inf).
        //   alpha > 1 > s is impossible. Use -inf intercept.
        if alpha <= 1.0 {
            LinearRelaxation::new(alpha, 0.0, 0.0, f32::INFINITY)
        } else {
            LinearRelaxation::nan_fallback()
        }
    } else if l.is_infinite() {
        // l = -inf, u finite > 0.
        // alpha < 0:
        //   f(x)=alpha*x on x<0 grows to +inf as x->-inf, so constant upper y<=u is UNSOUND.
        //   Use upper y = alpha*x + (1-alpha)*u:
        //     - At x=u: y=u=f(u)
        //     - On [0,u]: y-x = (1-alpha)(u-x) >= 0
        //     - On (-inf,0): y-alpha*x = (1-alpha)*u >= 0
        //   Lower y = x is sound globally (x<=alpha*x for x<0 when alpha<0, and exact for x>=0).
        //
        // alpha in [0,1]:
        //   Lower y = alpha*x (global lower), upper y = u (tight constant upper).
        //
        // alpha > 1:
        //   No finite affine lower bound exists; use -inf intercept.
        if alpha < 0.0 {
            LinearRelaxation::new(1.0, 0.0, alpha, (1.0 - alpha) * u)
        } else if alpha <= 1.0 {
            LinearRelaxation::new(alpha, 0.0, 0.0, u)
        } else {
            LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, u)
        }
    } else if u.is_infinite() {
        // l finite < 0, u = +inf.
        // Upper: y <= max(alpha,1)*x + l*(alpha - max(alpha,1)).
        //   When alpha <= 1: slope 1, intercept l*(alpha-1). At x=l: alpha*l. At x->+inf: ~x.
        //   When alpha > 1: slope alpha, intercept 0. At x=l: alpha*l. At x->+inf: ~alpha*x >= x.
        // Lower: when alpha <= 1: y = alpha*x (sound: alpha*x <= x for x>=0, exact for x<0).
        //   When alpha > 1: y = x + (alpha-1)*l.
        //   Proof: at x=l: l+(alpha-1)*l = alpha*l = f(l). For x in [l,0]:
        //     f(x)-y = (alpha-1)(x-l) >= 0. For x >= 0: f(x)-y = -(alpha-1)*l > 0.
        let upper_s = nan_propagating_max(alpha, 1.0);
        let upper_i = l * (alpha - upper_s);
        if alpha <= 1.0 {
            LinearRelaxation::new(alpha, 0.0, upper_s, upper_i)
        } else {
            LinearRelaxation::new(1.0, (alpha - 1.0) * l, upper_s, upper_i)
        }
    } else {
        // Crossing: linear relaxation for l < 0 < u, both finite.
        // Compute chord connecting (l, alpha*l) to (u, u) in f64 to avoid
        // precision loss when alpha is large. Apply directed rounding so the
        // chord remains a valid bound after f32 conversion. Part of #3313.
        let l_f64 = l as f64;
        let u_f64 = u as f64;
        let alpha_f64 = alpha as f64;
        let chord_slope_f64 = (u_f64 - alpha_f64 * l_f64) / (u_f64 - l_f64);
        let chord_slope_f32 = chord_slope_f64 as f32;
        // Recompute intercept from both endpoints using the f32 slope, then
        // take the direction that guarantees the bound.
        let intercept_at_l = alpha_f64 * l_f64 - (chord_slope_f32 as f64) * l_f64;
        let intercept_at_u = u_f64 - (chord_slope_f32 as f64) * u_f64;

        if alpha <= 1.0 {
            // alpha <= 1: f is convex at origin (slope increases from alpha to 1).
            // Upper: chord (above convex function). Lower: tangent (y = x or y = alpha*x).
            // Upper intercept must be >= both endpoint intercepts.
            let chord_intercept = next_up_f32(intercept_at_l.max(intercept_at_u) as f32);
            let lower_slope = if u > (-alpha * l).abs() { 1.0 } else { alpha };
            LinearRelaxation::new(lower_slope, 0.0, chord_slope_f32, chord_intercept)
        } else {
            // alpha > 1: f is concave at origin (slope decreases from alpha to 1).
            // Lower: chord (below concave function).
            // Lower intercept must be <= both endpoint intercepts.
            let chord_intercept = next_down_f32(intercept_at_l.min(intercept_at_u) as f32);
            // Upper: tangent (y = x or y = alpha*x — both lie above f).
            let upper_slope = if u > (-alpha * l).abs() { alpha } else { 1.0 };
            LinearRelaxation::new(chord_slope_f32, chord_intercept, upper_slope, 0.0)
        }
    }
}

impl LeakyReLULayer {
    /// Validate and create a new Leaky ReLU layer with the given negative slope.
    ///
    /// Returns an error if `alpha` is NaN or infinite, since non-finite
    /// slopes cause NaN/Inf propagation in IBP (alpha * x = NaN for NaN alpha).
    /// Note: negative alpha is valid (creates a non-monotone piecewise linear
    /// function); the relaxation code handles this case correctly. Part of #2551.
    pub fn try_new(alpha: f32) -> Result<Self> {
        Ok(Self {
            alpha: validate_finite(alpha, "LeakyReLULayer", "alpha")?,
        })
    }

    /// Create a new Leaky ReLU layer with the given negative slope.
    pub fn new(alpha: f32) -> Self {
        Self::try_new(alpha).expect("invariant: LeakyReLULayer::new requires validated alpha")
    }

    /// Create a Leaky ReLU layer with default alpha = 0.01.
    pub fn default_alpha() -> Self {
        Self { alpha: 0.01 }
    }
}

impl BoundPropagation for LeakyReLULayer {
    /// IBP for Leaky ReLU: y = x if x >= 0, else alpha * x
    ///
    /// For x in [l, u]:
    /// - If l >= 0: y in [l, u]
    /// - If u <= 0:
    ///   - alpha >= 0: y in [alpha*l, alpha*u]
    ///   - alpha < 0: y in [alpha*u, alpha*l] (decreasing on negative region)
    /// - If l < 0 < u:
    ///   - alpha >= 0: y in [alpha*l, u]
    ///   - alpha < 0: y in [0, max(alpha*l, u)]
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let alpha = self.alpha;
        let mut lower = input.lower().clone();
        let mut upper = input.upper().clone();

        Zip::from(&mut lower)
            .and(&mut upper)
            .for_each(|l_out, u_out| {
                let l = *l_out;
                let u = *u_out;
                let (new_l, new_u) = if l.is_nan() || u.is_nan() || !alpha.is_finite() {
                    (f32::NEG_INFINITY, f32::INFINITY)
                } else if l >= 0.0 {
                    (l, u)
                } else if u <= 0.0 {
                    let fl = safe_mul_for_bounds(alpha, l);
                    let fu = safe_mul_for_bounds(alpha, u);
                    if alpha >= 0.0 {
                        (fl, fu)
                    } else {
                        (fu, fl)
                    }
                } else if alpha >= 0.0 {
                    (safe_mul_for_bounds(alpha, l), u)
                } else {
                    (0.0, nan_propagating_max(safe_mul_for_bounds(alpha, l), u))
                };
                *l_out = new_l;
                *u_out = new_u;
            });

        // NaN alpha produces conservative [-inf, +inf] fallback bounds, so
        // infinite values are expected. Use new_allow_infinite to accept them.
        BoundedTensor::new_allow_infinite(lower, upper)
    }
    impl_elementwise_activation!(
        @trait_methods
        LeakyReLULayer,
        NyError::InvalidSpec(
            "LeakyReLU CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string()
        )
    );
}

impl LeakyReLULayer {
    impl_elementwise_activation!(
        @inherent_methods_stateful
        LeakyReLULayer,
        |layer: &LeakyReLULayer, l, u| leaky_relu_linear_relaxation(l, u, layer.alpha),
        // NaN-only guard: leaky_relu_linear_relaxation has proven over-approximation
        // branches for l=-inf and/or u=+inf (see the infinite-case arms in
        // leaky_relu_linear_relaxation), so infinite pre-activation bounds yield a tight
        // sound bound instead of an IBP fallback. Genuinely-unbounded sub-cases (alpha>1
        // on a domain reaching -inf) fail closed to a conservative ±Inf plane. NaN still
        // bails (cannot be bounded).
        domain_guard: |pre_activation: &BoundedTensor| {
            crate::layers::common::nan_only_domain_guard("LeakyReLU", pre_activation)
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearBounds;
    use ndarray::{array, ArrayD, IxDyn};
    use proptest::prelude::ProptestConfig;

    // ── Constructor validation tests (#2551) ────────────────────────────

    #[test]
    fn test_try_new_rejects_invalid_alpha_2551() {
        for alpha in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err =
                LeakyReLULayer::try_new(alpha).expect_err("non-finite alpha should be rejected");
            assert!(matches!(err, NyError::InvalidSpec(_)));
        }
    }

    #[test]
    fn test_try_new_accepts_valid_alpha_2551() {
        // LeakyReLU alpha can be any finite value (including negative)
        for alpha in [0.0, 0.01, -0.5, 1.0, -1.0] {
            LeakyReLULayer::try_new(alpha)
                .unwrap_or_else(|_| panic!("alpha={alpha} should be accepted"));
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
        } = leaky_relu_linear_relaxation(1.0, 5.0, 0.01);
        assert!((ls - 1.0).abs() < 1e-6, "positive: identity slope");
        assert!(li.abs() < 1e-6);
        assert!((us - 1.0).abs() < 1e-6);
        assert!(ui.abs() < 1e-6);
    }

    #[test]
    fn test_relaxation_negative() {
        let alpha = 0.1;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = leaky_relu_linear_relaxation(-5.0, -1.0, alpha);
        assert!((ls - alpha).abs() < 1e-6, "negative: slope=alpha");
        assert!(li.abs() < 1e-6);
        assert!((us - alpha).abs() < 1e-6);
        assert!(ui.abs() < 1e-6);
    }

    #[test]
    fn test_relaxation_crossing_default_alpha() {
        // alpha=0.01, crossing [-2, 3]: chord from (-2, -0.02) to (3, 3)
        let alpha = 0.01;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: _li,
            upper_slope: us,
            upper_intercept: _ui,
        } = leaky_relu_linear_relaxation(-2.0, 3.0, alpha);
        let expected_chord = (3.0 - alpha * (-2.0)) / (3.0 - (-2.0));
        assert!((us - expected_chord).abs() < 1e-5, "chord slope");
        // Lower: u=3 > |alpha*l|=0.02 → identity (slope=1)
        assert!((ls - 1.0).abs() < 1e-5, "lower: identity");
    }

    #[test]
    fn test_relaxation_crossing_alpha_lower() {
        // alpha=0.5, crossing [-5, 1]: u=1 < |alpha*l|=2.5 → lower slope=alpha
        let alpha = 0.5;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: _li,
            upper_slope: _us,
            upper_intercept: _ui,
        } = leaky_relu_linear_relaxation(-5.0, 1.0, alpha);
        assert!((ls - alpha).abs() < 1e-5, "lower: alpha (u < |alpha*l|)");
    }

    #[test]
    fn test_relaxation_near_point_crossing_guard_is_sound() {
        let alpha = -0.5;
        let l = -1e-20_f32;
        let u = 1e-20_f32;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = leaky_relu_linear_relaxation(l, u, alpha);
        assert_eq!(ls, 0.0, "near-point guard should return constant lower");
        assert_eq!(us, 0.0, "near-point guard should return constant upper");
        assert!(
            li.is_finite() && ui.is_finite(),
            "near-point guard must avoid Inf/NaN slopes"
        );
        for &x in &[l, 0.0, u] {
            let y = if x >= 0.0 { x } else { alpha * x };
            assert!(li <= y + 1e-12, "lower {} > y {} at x={}", li, y, x);
            assert!(ui >= y - 1e-12, "upper {} < y {} at x={}", ui, y, x);
        }
    }

    #[test]
    fn test_relaxation_nan() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = leaky_relu_linear_relaxation(f32::NAN, 1.0, 0.01);
        assert!(ls.abs() < 1e-6);
        assert!(li.is_infinite() && li < 0.0);
        assert!(us.abs() < 1e-6);
        assert!(ui.is_infinite() && ui > 0.0);
    }

    #[test]
    fn test_relaxation_boundary_l_zero() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = leaky_relu_linear_relaxation(0.0, 1.0, 0.1);
        assert!((ls - 1.0).abs() < 1e-6, "l=0: positive region");
        assert!(li.abs() < 1e-6);
        assert!((us - 1.0).abs() < 1e-6);
        assert!(ui.abs() < 1e-6);
    }

    #[test]
    fn test_relaxation_boundary_u_zero() {
        let alpha = 0.2;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = leaky_relu_linear_relaxation(-3.0, 0.0, alpha);
        assert!((ls - alpha).abs() < 1e-6, "u=0: negative region");
        assert!(li.abs() < 1e-6);
        assert!((us - alpha).abs() < 1e-6);
        assert!(ui.abs() < 1e-6);
    }

    #[test]
    fn test_relaxation_alpha_gt_1() {
        // alpha > 1: concave at origin, lower=chord, upper=tangent
        let alpha = 2.0;
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: _li,
            upper_slope: us,
            upper_intercept: _ui,
        } = leaky_relu_linear_relaxation(-1.0, 3.0, alpha);
        let expected_chord = (3.0 + alpha) / (3.0 + 1.0);
        assert!((ls - expected_chord).abs() < 1e-5, "alpha>1: lower=chord");
        // u=3 > |alpha*l|=2 → upper=alpha
        assert!((us - alpha).abs() < 1e-5, "alpha>1: upper=alpha");
    }

    // ── Relaxation soundness ───────────────────────────────────────────

    #[test]
    fn test_relaxation_soundness_grid() {
        let alphas = [0.01, 0.1, 0.5, 1.5, 2.0, -0.5, -1.0];
        let intervals: &[(f32, f32)] = &[(-3.0, 2.0), (-1.0, 1.0), (-5.0, 0.5), (-0.1, 10.0)];
        let leaky_relu = |alpha: f32, x: f32| -> f32 {
            if x >= 0.0 {
                x
            } else {
                alpha * x
            }
        };

        for &alpha in &alphas {
            for &(l, u) in intervals {
                let LinearRelaxation {
                    lower_slope: ls,
                    lower_intercept: li,
                    upper_slope: us,
                    upper_intercept: ui,
                } = leaky_relu_linear_relaxation(l, u, alpha);
                for k in 0..=50 {
                    let x = l + (u - l) * (k as f32 / 50.0);
                    let y = leaky_relu(alpha, x);
                    let lower_bound = ls * x + li;
                    let upper_bound = us * x + ui;
                    assert!(
                        lower_bound <= y + 1e-5,
                        "alpha={} [{},{}] x={}: lb {} > y {}",
                        alpha,
                        l,
                        u,
                        x,
                        lower_bound,
                        y
                    );
                    assert!(
                        upper_bound >= y - 1e-5,
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
        let layer = LeakyReLULayer::new(0.1);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[3]), 1.0_f32),
            ArrayD::from_elem(IxDyn(&[3]), 5.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        for &v in result.lower().iter() {
            assert!((v - 1.0).abs() < 1e-5);
        }
        for &v in result.upper().iter() {
            assert!((v - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ibp_negative() {
        let alpha = 0.1;
        let layer = LeakyReLULayer::new(alpha);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2]), -5.0_f32),
            ArrayD::from_elem(IxDyn(&[2]), -1.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        for &v in result.lower().iter() {
            assert!((v - alpha * (-5.0)).abs() < 1e-5);
        }
        for &v in result.upper().iter() {
            assert!((v + alpha).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ibp_crossing() {
        let alpha = 0.1;
        let layer = LeakyReLULayer::new(alpha);
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[2]), -3.0_f32),
            ArrayD::from_elem(IxDyn(&[2]), 2.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        for &v in result.lower().iter() {
            assert!((v - alpha * (-3.0)).abs() < 1e-5);
        }
        for &v in result.upper().iter() {
            assert!((v - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ibp_nan_alpha_falls_back_to_infinite_bounds() {
        // Bypass try_new validation to test defense-in-depth in IBP path.
        // In production, NaN alpha is rejected at construction (#2551), but
        // pub(crate) field access can still create invalid layers.
        let layer = LeakyReLULayer { alpha: f32::NAN };
        let input = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[1]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[1]), 2.0_f32),
        )
        .unwrap();
        let result = layer.propagate_ibp(&input).unwrap();
        let lower = result.lower()[[0]];
        let upper = result.upper()[[0]];
        assert!(lower.is_infinite() && lower.is_sign_negative());
        assert!(upper.is_infinite() && upper.is_sign_positive());
    }

    // ── CROWN backward tests ───────────────────────────────────────────

    #[test]
    fn test_crown_positive_preact() {
        let layer = LeakyReLULayer::new(0.1);
        let pre =
            BoundedTensor::new(array![1.0_f32].into_dyn(), array![5.0_f32].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        assert!((result.lower_a[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((result.upper_a[[0, 0]] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_crown_negative_preact() {
        let alpha = 0.1;
        let layer = LeakyReLULayer::new(alpha);
        let pre =
            BoundedTensor::new(array![-5.0_f32].into_dyn(), array![-1.0_f32].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();
        assert!(
            (result.lower_a[[0, 0]] - alpha).abs() < 1e-5,
            "negative: slope=alpha"
        );
        assert!((result.upper_a[[0, 0]] - alpha).abs() < 1e-5);
    }

    #[test]
    fn test_crown_crossing_soundness() {
        let alpha = 0.1;
        let layer = LeakyReLULayer::new(alpha);
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
            let y = if x >= 0.0 { x } else { alpha * x };
            let lower_bound = la * x + lb;
            let upper_bound = ua * x + ub;
            assert!(
                lower_bound <= y + 1e-5,
                "lb {} > y {} at x={}",
                lower_bound,
                y,
                x
            );
            assert!(
                upper_bound >= y - 1e-5,
                "ub {} < y {} at x={}",
                upper_bound,
                y,
                x
            );
        }
    }

    #[test]
    fn test_propagate_linear_requires_preact() {
        let layer = LeakyReLULayer::new(0.1);
        let bounds = LinearBounds::identity(1);
        let err = layer
            .propagate_linear(&bounds)
            .expect_err("requires pre-activation");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    // ── CROWN relaxation soundness proptest (#3321) ─────────────────────

    /// Reference LeakyReLU in f64, independent of the crate f32 implementation.
    fn leaky_relu_f64_reference(x: f64, alpha: f64) -> f64 {
        if x >= 0.0 {
            x
        } else {
            alpha * x
        }
    }

    proptest::proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

        /// #3321: Verify leaky_relu_linear_relaxation produces strictly sound bounds.
        /// For random intervals, the lower bound must satisfy
        ///   lower_slope * x + lower_intercept <= LeakyReLU(x)  for all x in [l, u]
        /// and the upper bound must satisfy
        ///   upper_slope * x + upper_intercept >= LeakyReLU(x)  for all x in [l, u]
        /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
        ///
        /// Ref: ELU proptest_elu_relaxation_strict_soundness (elu.rs:841).
        #[test]
        fn proptest_leaky_relu_relaxation_strict_soundness(
            l in -10.0f32..10.0,
            width in 0.01f32..20.0,
            alpha in -5.0f32..5.0,
        ) {
            let u = l + width;
            let relax = leaky_relu_linear_relaxation(l, u, alpha);
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
                let fx = leaky_relu_f64_reference(x, alpha64);

                let lower_val = ls as f64 * x + li as f64;
                proptest::prop_assert!(
                    lower_val <= fx,
                    "LeakyReLU lower bound UNSOUND at x={}: {} > LeakyReLU({})={}, \
                     interval=[{}, {}], alpha={}, gap={}", x, lower_val, x, fx, l, u, alpha, lower_val - fx
                );

                let upper_val = us as f64 * x + ui as f64;
                proptest::prop_assert!(
                    upper_val >= fx,
                    "LeakyReLU upper bound UNSOUND at x={}: {} < LeakyReLU({})={}, \
                     interval=[{}, {}], alpha={}, gap={}", x, upper_val, x, fx, l, u, alpha, fx - upper_val
                );
            }
        }

        /// Soundness with INFINITE pre-activation bounds (l=-inf and/or u=+inf).
        /// The relaxation must be sound on a finite grid over the bounded part of the
        /// domain, and the plane must also bound the function in the unbounded
        /// direction (verified by sampling out to large magnitudes). Genuinely
        /// unbounded sub-cases (alpha>1 toward -inf) fail closed to a ±Inf plane,
        /// which is trivially sound and skipped by the finite-coefficient assume.
        ///
        /// This exercises the proven infinite-case arms of leaky_relu_linear_relaxation
        /// that the NaN-only domain guard now allows to run on unbounded inputs.
        #[test]
        fn proptest_leaky_relu_relaxation_infinite_domain_soundness(
            // which endpoints are infinite: 0=l-inf, 1=u-inf, 2=both
            inf_kind in 0usize..3,
            finite_endpoint in -10.0f32..10.0,
            alpha in -5.0f32..5.0,
        ) {
            let (l, u) = match inf_kind {
                0 => (f32::NEG_INFINITY, finite_endpoint.max(0.5)), // u finite > 0
                1 => (finite_endpoint.min(-0.5), f32::INFINITY),    // l finite < 0
                _ => (f32::NEG_INFINITY, f32::INFINITY),
            };

            let relax = leaky_relu_linear_relaxation(l, u, alpha);
            let ls = relax.lower_slope;
            let li = relax.lower_intercept;
            let us = relax.upper_slope;
            let ui = relax.upper_intercept;

            // At least one endpoint is infinite by construction.
            proptest::prop_assert!(l.is_infinite() || u.is_infinite());

            // The relaxation must never produce NaN coefficients.
            proptest::prop_assert!(
                !ls.is_nan() && !li.is_nan() && !us.is_nan() && !ui.is_nan(),
                "LeakyReLU infinite-domain relaxation produced NaN: ls={} li={} us={} ui={} for [{}, {}] alpha={}",
                ls, li, us, ui, l, u, alpha
            );

            let alpha64 = alpha as f64;

            // Build a finite probe grid spanning the bounded part of the domain plus a
            // large excursion into any unbounded direction. Soundness must hold at every
            // probe, including the far-out points exercising the unbounded direction.
            let lo_probe = if l.is_infinite() { -1.0e6_f64 } else { l as f64 };
            let hi_probe = if u.is_infinite() { 1.0e6_f64 } else { u as f64 };

            for k in 0..=400 {
                let t = k as f64 / 400.0;
                let x = lo_probe + t * (hi_probe - lo_probe);
                let fx = leaky_relu_f64_reference(x, alpha64);

                // Lower plane: if it is the conservative -inf intercept, it is trivially
                // sound; otherwise check ls*x + li <= f(x).
                if li.is_finite() {
                    let lower_val = ls as f64 * x + li as f64;
                    proptest::prop_assert!(
                        lower_val <= fx + 1e-3 * fx.abs().max(1.0),
                        "LeakyReLU INFINITE-domain lower UNSOUND at x={}: {} > f(x)={}, \
                         interval=[{}, {}], alpha={}", x, lower_val, fx, l, u, alpha
                    );
                } else {
                    proptest::prop_assert!(li == f32::NEG_INFINITY,
                        "non-finite lower intercept must be -inf, got {}", li);
                }

                // Upper plane: symmetric treatment.
                if ui.is_finite() {
                    let upper_val = us as f64 * x + ui as f64;
                    proptest::prop_assert!(
                        upper_val + 1e-3 * fx.abs().max(1.0) >= fx,
                        "LeakyReLU INFINITE-domain upper UNSOUND at x={}: {} < f(x)={}, \
                         interval=[{}, {}], alpha={}", x, upper_val, fx, l, u, alpha
                    );
                } else {
                    proptest::prop_assert!(ui == f32::INFINITY,
                        "non-finite upper intercept must be +inf, got {}", ui);
                }
            }
        }
    }
}
