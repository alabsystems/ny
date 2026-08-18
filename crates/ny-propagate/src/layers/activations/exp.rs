// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{f64_to_f32_down, f64_to_f32_up, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use super::LinearRelaxation;
use crate::bounds::{nan_propagating_max, nan_propagating_max_zero};
use crate::layers::common::{impl_elementwise_activation, BoundPropagation};

/// Maximum safe input for f32 exp before overflow to infinity.
/// exp(88.0) ≈ 1.65e38 (within f32::MAX ≈ 3.4e38).
/// exp(89.0) ≈ 4.49e38 (exceeds f32::MAX, returns +inf).
/// Threshold 88.0 is conservative but safe.
const EXP_OVERFLOW_THRESHOLD: f32 = 88.0;

/// Exp layer: y = exp(x)
///
/// Element-wise exponential function. Common in softmax decomposition
/// and various neural network architectures.
#[derive(Debug, Clone, Default)]
pub struct ExpLayer;

impl ExpLayer {
    /// Create a new Exp layer.
    pub fn new() -> Self {
        Self
    }
}

impl BoundPropagation for ExpLayer {
    /// IBP for Exp: y = exp(x)
    ///
    /// Exp is monotonically increasing, so bounds are straightforward.
    /// Returns `NumericalInstability` if any input bound is non-finite or
    /// if upper bounds exceed the f32 exp overflow threshold (~88).
    ///
    /// Category B per domain validation policy (designs/2026-02-07-domain-validation-policy.md):
    /// exp is defined for all real inputs, but f32 exp overflows for x > ~88.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Guard: reject non-finite inputs
        if input.lower().iter().any(|x| !x.is_finite())
            || input.upper().iter().any(|x| !x.is_finite())
        {
            return Err(NyError::NumericalInstability(
                "Exp IBP: non-finite input bounds".to_string(),
            ));
        }

        // Guard: reject overflow-risk upper bounds
        if input.upper().iter().any(|&x| x > EXP_OVERFLOW_THRESHOLD) {
            let max_upper = input
                .upper()
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, nan_propagating_max);
            return Err(NyError::NumericalInstability(format!(
                "Exp IBP: upper bound {:.1} exceeds overflow threshold {:.1}",
                max_upper, EXP_OVERFLOW_THRESHOLD
            )));
        }

        // Directed rounding: compute exp in f64 for precision, cast to f32 with
        // next_down/next_up to guarantee lower bounds round DOWN and upper bounds
        // round UP. Raw f32 exp can round either direction. (#1483)
        // Range clamp: exp(x) >= 0 for all real x. next_down_f32 can push past
        // zero for extreme underflow (e.g., exp(-1000) → 0 → -1e-45). (#3316)
        // NaN-propagating: .max() swallows NaN (IEEE 754-2008). (#3316)
        let lower = input
            .lower()
            .mapv(|x| nan_propagating_max_zero(next_down_f32(f64_to_f32_down((x as f64).exp()))));
        let upper = input
            .upper()
            .mapv(|x| next_up_f32(f64_to_f32_up((x as f64).exp())));
        BoundedTensor::new(lower, upper)
    }
    impl_elementwise_activation!(
        @trait_methods
        ExpLayer,
        NyError::UnsupportedOp(
            "Exp is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
                .to_string()
        )
    );
}

/// Sound intercept widening for one bound direction of the exp relaxation.
///
/// The envelope is consumed as `slope_f32 * x + intercept_f32` evaluated in
/// f32 and compared against a (faithfully rounded) f32 `exp`. Relative to the
/// exact f64 line, that evaluation can err by:
///   E1: slope f64→f32 truncation   <= |slope_f64 - slope_f32| * |x|
///   E2: f32 multiplication         <= |slope_f32| * |x| * eps
///   E3: f32 addition rounding      <= (|slope_f32 * x| + |intercept|) * eps
///   E4: f32::exp() faithful round  <= exp(x) * eps
/// The sum is accumulated in f64 and charged to the intercept BEFORE the
/// final directed-rounding cast — adding it after the f32 cast would let the
/// whole correction be absorbed by ULP granularity when the intercept is
/// orders of magnitude larger than the correction (e.g. x near 19, where the
/// intercept is ~-4e9 with a 512 ULP while E2 alone is worth hundreds).
fn exp_intercept_correction(
    slope_f64: f64,
    slope_f32: f32,
    intercept: f64,
    max_abs_x: f64,
    max_exp_val: f64,
) -> f64 {
    let eps = f32::EPSILON as f64;
    let slope_err = (slope_f64 - slope_f32 as f64).abs() * max_abs_x;
    let mul_err = slope_f32.abs() as f64 * max_abs_x * eps;
    let eval_add_err = (slope_f32.abs() as f64 * max_abs_x + intercept.abs()) * eps;
    let exp_faithful_err = max_exp_val * eps;
    slope_err + mul_err + eval_add_err + exp_faithful_err
}

/// Reject an affine envelope that overflowed while being assembled.
///
/// Non-zero infinite slopes/intercepts are not a usable conservative line:
/// ordinary evaluation can produce `inf - inf = NaN`.  The zero-slope wide
/// fallback remains well-defined for every finite input.
#[inline]
fn finite_exp_relaxation_or_fallback(relaxation: LinearRelaxation) -> LinearRelaxation {
    if relaxation.lower_slope.is_finite()
        && relaxation.lower_intercept.is_finite()
        && relaxation.upper_slope.is_finite()
        && relaxation.upper_intercept.is_finite()
    {
        relaxation
    } else {
        LinearRelaxation::nan_fallback()
    }
}

/// Linear relaxation for exp on interval [l, u].
///
/// exp(x) is convex and monotonically increasing.
/// For convex functions: chord is upper bound, tangent is lower bound.
///
/// **Upper bound (chord/secant through endpoints):**
///   slope_u = (exp(u) - exp(l)) / (u - l)
///   intercept_u = exp(l) - slope_u * l
///
/// **Lower bound (tangent at midpoint m):**
///   m = (l + u) / 2               [gap-area-optimal fixed tangent point]
///   slope_l = exp(m)                 [derivative of exp at m]
///   intercept_l = exp(m) * (1 - m)   [tangent: y = exp(m)*(x - m + 1)]
///
/// **Numerical stability (#1745):** Intermediate computation uses f64 to prevent
/// catastrophic cancellation when pre-activation values are large. For x near 8,
/// `exp(m) * (1 - m)` produces intercepts ~-20000 while slopes ~3000 yield
/// `slope * x` ~24000. The 8x cancellation in `slope * x + intercept` loses ~3
/// bits of f32 mantissa, which compounds through the CROWN backward composition.
/// f64 intermediates provide ~30 extra mantissa bits, eliminating this issue.
///
/// Reference: alpha-beta-CROWN `BoundExp.bound_relax` in
/// `auto_LiRPA/operators/convex_concave.py:298-313`
pub fn exp_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    // Guard invalid/unbounded intervals: return conservative intercepts so
    // direct callers cannot receive NaN coefficients. Normal CROWN entry
    // points reject these bounds in `exp_crown_domain_guard`, but this helper
    // is public and must also fail closed on its own.
    if !l.is_finite() || !u.is_finite() || l > u {
        return LinearRelaxation::nan_fallback();
    }

    // Use f64 intermediates to prevent catastrophic cancellation (#1745).
    let l64 = l as f64;
    let u64 = u as f64;

    // Handle degenerate case: point interval. Both envelopes are the tangent
    // at l; the curvature gap over a <1e-8-wide interval is below exp(x) *
    // 5e-17, absorbed by the E4 term of the correction. The full E1-E4
    // charge is still required: at l ≈ 19.25 the slope is ~2.3e8, so the f32
    // caller multiply alone (E2) can land the evaluated line hundreds below
    // exp(l).
    if (u64 - l64).abs() < 1e-8 {
        let exp_l = l64.exp();
        let slope = exp_l;
        let intercept = exp_l * (1.0 - l64);
        let slope_f32 = slope as f32;
        let total_err =
            exp_intercept_correction(slope, slope_f32, intercept, l.abs() as f64, exp_l);
        return finite_exp_relaxation_or_fallback(LinearRelaxation::new(
            slope_f32,
            next_down_f32(f64_to_f32_down(intercept - total_err)),
            slope_f32,
            next_up_f32(f64_to_f32_up(intercept + total_err)),
        ));
    }

    let exp_l = l64.exp();
    let exp_u = u64.exp();

    // Upper bound: chord (secant) through (l, exp(l)) and (u, exp(u)).
    // For convex exp, the chord lies above the function.
    let upper_slope = (exp_u - exp_l) / (u64 - l64);
    let upper_intercept = exp_l - upper_slope * l64;

    // Lower bound: tangent line at the arithmetic midpoint m = (l+u)/2.
    // The tangent to a convex function lies below the function for ANY m,
    // so this is always a sound lower bound. The arithmetic midpoint is the
    // gap-AREA-optimal fixed tangent point: with
    //   I(m) = ∫_l^u [exp(x) − exp(m)(x−m+1)] dx
    //        = (exp(u)−exp(l)) − exp(m)·(u−l)·(mid − m + 1),
    //   dI/dm = −(u−l)·exp(m)·(mid − m),  which is zero at m = mid,
    //   d²I/dm² |_{m=mid} = (u−l)·exp(m) > 0  ⇒ a minimum.
    // The tangent slope exp(mid) is bounded above by the chord slope
    // (exp(u)−exp(l))/(u−l) (convexity), i.e. it never exceeds the slope
    // already used for the upper-bound chord — so there is no "runaway
    // slope". Matches alpha-beta-CROWN's uncapped midpoint tangent.
    let m = f64::midpoint(l64, u64);
    let exp_m = m.exp();
    let lower_slope = exp_m;
    let lower_intercept = exp_m * (1.0 - m);

    // Widen intercepts to absorb every error the caller's f32 evaluation of
    // `slope_f32 * x + intercept_f32` can incur relative to true exp(x); see
    // `exp_intercept_correction` for the term-by-term E1-E4 derivation. The
    // correction is applied in f64 before the directed-rounding cast so it
    // cannot be absorbed by f32 ULP granularity at large intercept magnitudes.
    let max_abs_x = nan_propagating_max(l.abs(), u.abs()) as f64;
    let lower_slope_f32 = lower_slope as f32;
    let upper_slope_f32 = upper_slope as f32;
    let max_exp_val = exp_u.max(exp_l);

    let lower_correction = exp_intercept_correction(
        lower_slope,
        lower_slope_f32,
        lower_intercept,
        max_abs_x,
        max_exp_val,
    );
    let upper_correction = exp_intercept_correction(
        upper_slope,
        upper_slope_f32,
        upper_intercept,
        max_abs_x,
        max_exp_val,
    );

    finite_exp_relaxation_or_fallback(LinearRelaxation::new(
        lower_slope_f32,
        next_down_f32(f64_to_f32_down(lower_intercept - lower_correction)),
        upper_slope_f32,
        next_up_f32(f64_to_f32_up(upper_intercept + upper_correction)),
    ))
}

fn exp_crown_domain_guard(pre_activation: &BoundedTensor) -> Result<()> {
    // Guard: reject non-finite pre-activation bounds
    if pre_activation.lower().iter().any(|x| !x.is_finite())
        || pre_activation.upper().iter().any(|x| !x.is_finite())
    {
        return Err(NyError::NumericalInstability(
            "Exp CROWN: non-finite pre-activation bounds".to_string(),
        ));
    }

    // Guard: reject overflow-risk upper bounds (relaxation computes exp(u))
    if pre_activation
        .upper()
        .iter()
        .any(|&x| x > EXP_OVERFLOW_THRESHOLD)
    {
        let max_upper = pre_activation
            .upper()
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, nan_propagating_max);
        return Err(NyError::NumericalInstability(format!(
            "Exp CROWN: upper bound {:.1} exceeds overflow threshold {:.1}",
            max_upper, EXP_OVERFLOW_THRESHOLD
        )));
    }

    Ok(())
}

impl ExpLayer {
    impl_elementwise_activation!(
        @inherent_methods
        ExpLayer,
        exp_linear_relaxation,
        domain_guard: exp_crown_domain_guard
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LinearBounds;

    /// Verify that the linear relaxation forms a valid envelope around exp(x)
    /// for all x in [l, u]: lower_slope * x + lower_intercept <= exp(x) <= upper_slope * x + upper_intercept
    fn assert_exp_envelope(l: f32, u: f32) {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = exp_linear_relaxation(l, u);
        let samples = if (u - l).abs() < 1e-8 { 1 } else { 100 };
        for i in 0..=samples {
            let t = i as f32 / samples as f32;
            let x = l + (u - l) * t;
            let fx = x.exp();
            let lower = ls * x + li;
            let upper = us * x + ui;
            let tol = 1e-4 * fx.abs().max(1.0);
            assert!(
                lower <= fx + tol,
                "exp lower envelope violated for [{l}, {u}] at x={x}: lower={lower} > exp(x)={fx}"
            );
            assert!(
                upper + tol >= fx,
                "exp upper envelope violated for [{l}, {u}] at x={x}: upper={upper} < exp(x)={fx}"
            );
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn exp_relaxation_envelope_basic() {
        let intervals = [
            (0.0, 0.0),    // point
            (0.0, 1.0),    // small positive
            (-1.0, 1.0),   // crosses zero
            (-3.0, 0.0),   // negative
            (-5.0, 5.0),   // wide symmetric
            (1.0, 2.0),    // positive narrow
            (-10.0, -5.0), // deep negative
            (0.0, 10.0),   // wide positive
            (-2.0, 3.0),   // asymmetric crossing
        ];
        for (l, u) in intervals {
            assert_exp_envelope(l, u);
        }
    }

    #[ntest::timeout(5000)]
    #[test]
    fn exp_relaxation_degenerate_point() {
        // Point interval: both bounds should be tight
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = exp_linear_relaxation(1.0, 1.0);
        let exp_1 = 1.0_f32.exp();
        // At x=1: tangent line y = exp(1) * x + exp(1) * (1 - 1) = e*x
        assert!((ls * 1.0 + li - exp_1).abs() < 1e-5);
        assert!((us * 1.0 + ui - exp_1).abs() < 1e-5);
    }

    /// The intercept widening must cover the full f32 caller-evaluation
    /// error model — slope f64→f32 truncation (E1), the f32 multiply (E2),
    /// the f32 add (E3), and the caller's faithfully rounded f32 exp (E4) —
    /// on the point/near-point path exactly as on the chord path. Near
    /// x ≈ 19.25, exp(x) ≈ 2.3e8 and slope·x ≈ 4.4e9, so each eps-sized term
    /// is worth hundreds in absolute value; dropping any of them puts the
    /// evaluated envelope below exp(x). Zero tolerance: the envelope must
    /// dominate any faithful f32 exp, i.e. one ULP past this platform's
    /// correctly rounded value.
    #[ntest::timeout(5000)]
    #[test]
    fn exp_envelope_covers_f32_caller_evaluation_near_point() {
        let intervals: [(f32, f32); 3] = [
            (19.2458, 19.25), // measured near-point interval (chord path)
            (19.25, 19.25),   // exact point (tangent path)
            (19.25, 19.251),  // narrow (chord path)
        ];
        for (l, u) in intervals {
            let LinearRelaxation {
                lower_slope: ls,
                lower_intercept: li,
                upper_slope: us,
                upper_intercept: ui,
            } = exp_linear_relaxation(l, u);
            let samples = 4096;
            for i in 0..=samples {
                let t = i as f32 / samples as f32;
                let x = (l + (u - l) * t).clamp(l, u);
                // f32 caller model: f32 multiply then f32 add, compared
                // against a faithful f32 exp (up to one ULP either side of
                // the correctly rounded value).
                let fx_hi = next_up_f32(x.exp());
                let fx_lo = next_down_f32(x.exp());
                let upper = us * x + ui;
                let lower = ls * x + li;
                assert!(
                    upper >= fx_hi,
                    "exp upper envelope below f32 exp on [{l}, {u}] at x={x}: {upper} < {fx_hi}"
                );
                assert!(
                    lower <= fx_lo,
                    "exp lower envelope above f32 exp on [{l}, {u}] at x={x}: {lower} > {fx_lo}"
                );
            }
        }
    }

    // ── CROWN backward tests ───────────────────────────────────────────

    #[test]
    fn test_crown_backward_soundness() {
        use ndarray::arr1;
        let layer = ExpLayer::new();
        let l = -2.0_f32;
        let u = 3.0_f32;
        let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = x.exp();
            let tol = 1e-3 * y.max(1.0);
            assert!(
                la * x + lb <= y + tol,
                "Exp CROWN lb violated at x={x}: {} > {y}",
                la * x + lb
            );
            assert!(
                ua * x + ub >= y - tol,
                "Exp CROWN ub violated at x={x}: {} < {y}",
                ua * x + ub
            );
        }
    }

    #[test]
    fn test_crown_backward_negative_region() {
        use ndarray::arr1;
        let layer = ExpLayer::new();
        let l = -5.0_f32;
        let u = -1.0_f32;
        let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = x.exp();
            assert!(
                result.lower_a[[0, 0]] * x + result.lower_b[0] <= y + 1e-4,
                "Exp negative lb violated at x={x}"
            );
            assert!(
                result.upper_a[[0, 0]] * x + result.upper_b[0] >= y - 1e-4,
                "Exp negative ub violated at x={x}"
            );
        }
    }

    #[test]
    fn test_crown_backward_multi_neuron() {
        use ndarray::arr1;
        let layer = ExpLayer::new();
        let pre = BoundedTensor::new(
            arr1(&[-3.0_f32, 0.0]).into_dyn(),
            arr1(&[0.0_f32, 2.0]).into_dyn(),
        )
        .unwrap();
        let bounds = LinearBounds::identity(2);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre).unwrap();

        for neuron in 0..2 {
            let la = result.lower_a[[neuron, neuron]];
            let lb = result.lower_b[neuron];
            let ua = result.upper_a[[neuron, neuron]];
            let ub = result.upper_b[neuron];
            let lo = pre.lower()[neuron];
            let hi = pre.upper()[neuron];

            for k in 0..=20 {
                let x = lo + (hi - lo) * (k as f32 / 20.0);
                let y = x.exp();
                let tol = 1e-3 * y.max(1.0);
                assert!(
                    la * x + lb <= y + tol,
                    "neuron {neuron} lb violated at x={x}"
                );
                assert!(
                    ua * x + ub >= y - tol,
                    "neuron {neuron} ub violated at x={x}"
                );
            }
        }
    }

    #[test]
    fn test_crown_backward_overflow_guard() {
        use ndarray::arr1;
        let layer = ExpLayer::new();
        // Upper bound above overflow threshold should be rejected
        let pre =
            BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[100.0_f32]).into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        assert!(
            layer.propagate_linear_with_bounds(&bounds, &pre).is_err(),
            "Exp CROWN should reject upper bounds above overflow threshold"
        );
    }

    #[test]
    fn test_ibp_rejects_nan_input_bounds() {
        use ndarray::{ArrayD, IxDyn};
        let layer = ExpLayer::new();
        let lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
        let input = BoundedTensor::new_unchecked(lower, upper).unwrap();
        let err = layer
            .propagate_ibp(&input)
            .expect_err("Exp IBP should reject NaN input bounds");
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "expected NumericalInstability, got {err:?}"
        );
    }

    #[test]
    fn test_crown_backward_rejects_nan_preactivation() {
        use ndarray::{ArrayD, IxDyn};
        let layer = ExpLayer::new();
        let lower = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NAN]).unwrap();
        let pre = BoundedTensor::new_unchecked(lower, upper).unwrap();
        let bounds = LinearBounds::identity(1);
        let err = layer
            .propagate_linear_with_bounds(&bounds, &pre)
            .expect_err("Exp CROWN should reject NaN pre-activation bounds");
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "expected NumericalInstability, got {err:?}"
        );
    }

    #[test]
    fn test_propagate_linear_requires_preact() {
        let layer = ExpLayer::new();
        let bounds = LinearBounds::identity(1);
        assert!(
            layer.propagate_linear(&bounds).is_err(),
            "Exp CROWN without pre-activation bounds should fail"
        );
        assert!(layer.requires_pre_activation_bounds());
    }

    #[test]
    fn test_relaxation_nan_bounds_return_wide_bounds() {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = exp_linear_relaxation(f32::NAN, 1.0);
        assert_eq!(ls, 0.0);
        assert!(li.is_infinite() && li.is_sign_negative());
        assert_eq!(us, 0.0);
        assert!(ui.is_infinite() && ui.is_sign_positive());
    }

    #[test]
    fn test_relaxation_invalid_or_unbounded_bounds_return_wide_bounds_4369() {
        for (lower, upper) in [(1.0, -1.0), (f32::NEG_INFINITY, 0.0), (0.0, f32::INFINITY)] {
            let result = exp_linear_relaxation(lower, upper);
            assert_eq!(result.lower_slope, 0.0);
            assert!(
                result.lower_intercept.is_infinite() && result.lower_intercept.is_sign_negative()
            );
            assert_eq!(result.upper_slope, 0.0);
            assert!(
                result.upper_intercept.is_infinite() && result.upper_intercept.is_sign_positive()
            );
        }
    }

    #[test]
    fn test_relaxation_finite_overflow_risk_returns_wide_bounds_4369() {
        for (lower, upper) in [(89.0, 89.0), (1_000.0, 1_001.0)] {
            assert_eq!(
                exp_linear_relaxation(lower, upper),
                LinearRelaxation::nan_fallback(),
                "finite endpoints must still fail closed when affine coefficients overflow"
            );
        }
    }

    #[test]
    fn exp_underflow_bounds_and_lines_remain_sound_under_ftz() {
        use ndarray::arr1;

        let x = -100.0_f32;
        let reference = (x as f64).exp();
        let input = BoundedTensor::new(arr1(&[x]).into_dyn(), arr1(&[x]).into_dyn()).unwrap();
        let ibp = ExpLayer::new().propagate_ibp(&input).unwrap();
        assert_eq!(ibp.lower()[[0]], 0.0);
        assert!(
            ibp.upper()[[0]] >= f32::MIN_POSITIVE,
            "positive upper endpoints must not depend on binary32 subnormals"
        );
        assert!((ibp.upper()[[0]] as f64) >= reference);

        let relaxation = exp_linear_relaxation(x, x);
        assert!(
            relaxation.upper_intercept >= f32::MIN_POSITIVE,
            "the affine upper line must survive FTZ: {relaxation:?}"
        );
        let lower = relaxation.lower_slope * x + relaxation.lower_intercept;
        let upper = relaxation.upper_slope * x + relaxation.upper_intercept;
        assert!((lower as f64) <= reference, "{lower:e} > {reference:e}");
        assert!((upper as f64) >= reference, "{upper:e} < {reference:e}");
    }

    #[ntest::timeout(5000)]
    #[test]
    fn exp_relaxation_tangent_below_chord_above() {
        // For a non-degenerate interval, lower (tangent) should be tighter
        // than upper (chord) at the tangent point.
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = exp_linear_relaxation(0.0, 2.0);
        // At the midpoint (tangent touches):
        let m = f32::midpoint(0.0, 2.0);
        let fx = m.exp();
        let lower_at_m = ls * m + li;
        let upper_at_m = us * m + ui;
        // Lower should touch the function at tangent point
        assert!((lower_at_m - fx).abs() < 1e-4, "tangent should touch at m");
        // Upper (chord) should be above at midpoint
        assert!(upper_at_m >= fx - 1e-4, "chord should be above at midpoint");
    }
}
