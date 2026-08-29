// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core mathematical functions for SiLU (Swish) activation.
//!
//! Contains sigmoid, SiLU evaluation, derivatives, critical points,
//! inflection points, and basic interval min/max computation.

use std::sync::OnceLock;

use crate::bounds::{nan_propagating_max, nan_propagating_min};
use ny_core::{f32_affine_eval_error, f64_to_f32_down, f64_to_f32_up};
use ny_tensor::{next_down_f32, next_up_f32};

static SILU_CRITICAL_POINT: OnceLock<f32> = OnceLock::new();
static SILU_INFLECTION_POINTS: OnceLock<(f32, f32)> = OnceLock::new();

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Evaluate SiLU (Swish) activation at a point: x * sigmoid(x).
///
/// SiLU is non-monotonic with a global minimum near x ≈ -1.28.
/// As x → -∞, SiLU(x) → 0. As x → +∞, SiLU(x) → x.
#[inline]
pub fn silu_eval(x: f32) -> f32 {
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    x * sigmoid(x)
}

#[inline]
pub(crate) fn silu_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// Second derivative of SiLU: SiLU''(x) = σ(x)(1-σ(x))(2 + x(1 - 2σ(x)))
#[inline]
pub(crate) fn silu_second_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    let s1 = 1.0 - s;
    s * s1 * (2.0 + x * (1.0 - 2.0 * s))
}

#[inline]
pub(crate) fn silu_critical_point() -> f32 {
    *SILU_CRITICAL_POINT.get_or_init(|| {
        let mut x = -1.28_f32;
        let eps = 1.0e-5_f32;
        for _ in 0..20 {
            let d = silu_derivative(x);
            let dd = (silu_derivative(x + eps) - silu_derivative(x - eps)) / (2.0 * eps);
            if dd.abs() < 1.0e-10 {
                break;
            }
            x -= d / dd;
        }
        x
    })
}

/// Compute the two inflection points of SiLU where SiLU''(x) = 0.
///
/// SiLU''(x) = σ(x)(1-σ(x))(2 + x - 2xσ(x)) = σ(x)(1-σ(x))(2 + x(1 - 2σ(x)))
/// Since σ(x)(1-σ(x)) > 0 for finite x, the inflection points satisfy:
///   2 + x(1 - 2σ(x)) = 0
///
/// Left inflection p₁ ≈ -2.3994, right inflection p₂ ≈ +2.3994.
/// Reference: designs/2026-02-08-silu-crown-relaxation.md, "Key points" table.
pub(crate) fn silu_inflection_points() -> (f32, f32) {
    *SILU_INFLECTION_POINTS.get_or_init(|| {
        // f(x) = 2 + x(1 - 2σ(x))
        // f'(x) = (1 - 2σ(x)) + x · (-2σ'(x)) = (1 - 2σ(x)) - 2xσ(x)(1-σ(x))
        let f = |x: f32| -> f32 {
            let s = sigmoid(x);
            2.0 + x * (1.0 - 2.0 * s)
        };
        let f_deriv = |x: f32| -> f32 {
            let s = sigmoid(x);
            (1.0 - 2.0 * s) - 2.0 * x * s * (1.0 - s)
        };

        // Newton's method for left inflection (starting near -2.4)
        let mut x_left = -2.4_f32;
        for _ in 0..30 {
            let fd = f_deriv(x_left);
            if fd.abs() < 1.0e-12 {
                break;
            }
            let step = f(x_left) / fd;
            x_left -= step;
            if step.abs() < 1.0e-8 {
                break;
            }
        }

        // Newton's method for right inflection (starting near 2.4)
        let mut x_right = 2.4_f32;
        for _ in 0..30 {
            let fd = f_deriv(x_right);
            if fd.abs() < 1.0e-12 {
                break;
            }
            let step = f(x_right) / fd;
            x_right -= step;
            if step.abs() < 1.0e-8 {
                break;
            }
        }

        (x_left, x_right)
    })
}

/// Directed rounding: compute in f64, apply next_down/next_up to EACH
/// intermediate evaluation BEFORE min/max selection. Plain `as f32` rounds to
/// nearest — a candidate min could round UP, causing the final min to miss the
/// true minimum. next_down/next_up on the final result alone cannot recover if
/// the wrong candidate was selected. (#3245, #3336)
///
/// Ref: GELU (#3336) and f64→f32 cast (#3132) directed rounding fixes.
pub(crate) fn silu_min_max(l: f32, u: f32) -> (f32, f32) {
    let fl = silu_eval_f64(l as f64) as f32;
    let fu = silu_eval_f64(u as f64) as f32;

    // Apply directed rounding to each intermediate before min/max selection.
    let mut min_val = nan_propagating_min(next_down_f32(fl), next_down_f32(fu));
    let mut max_val = nan_propagating_max(next_up_f32(fl), next_up_f32(fu));

    let critical = silu_critical_point();
    if l < critical && critical < u {
        let f_min = silu_eval_f64(critical as f64) as f32;
        min_val = nan_propagating_min(min_val, next_down_f32(f_min));
        max_val = nan_propagating_max(max_val, next_up_f32(f_min));
    }

    (min_val, max_val)
}

/// Evaluate SiLU in f64 to avoid catastrophic cancellation in chord slope.
/// SiLU(x) = x · σ(x) where σ is the logistic sigmoid.
/// Used by relaxation.rs for f64-precision verification checks (#2434).
#[inline]
pub(crate) fn silu_eval_f64(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let sigmoid = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    };
    x * sigmoid
}

/// Compute chord line between (l, SiLU(l)) and (u, SiLU(u)).
/// Returns (slope, lower_intercept, upper_intercept) with directed rounding.
///
/// Precondition: `(u - l).abs() >= 1e-8`. Point intervals should be handled by
/// the caller before requesting a chord line.
///
/// Uses f64 intermediates to prevent catastrophic cancellation when u ≈ l.
/// Directed rounding absorbs f64→f32 slope truncation error into the intercept:
/// - `lower_intercept`: guaranteed `slope*x + lower_intercept <= true_chord(x)` ∀x∈[l,u]
/// - `upper_intercept`: guaranteed `slope*x + upper_intercept >= true_chord(x)` ∀x∈[l,u]
///
/// Ref: Exp (exp.rs:119-125), ELU (elu.rs:121-124) directed rounding pattern.
/// Fixes: #2434 — SiLU chord used raw `as f32` without directed rounding.
#[inline]
pub(crate) fn silu_chord(l: f32, u: f32) -> (f32, f32, f32) {
    debug_assert!(
        (u - l).abs() >= 1.0e-8,
        "silu_chord requires a non-point interval; caller must guard |u - l| < 1e-8"
    );
    let l64 = l as f64;
    let u64 = u as f64;
    let fl64 = silu_eval_f64(l64);
    let fu64 = silu_eval_f64(u64);
    // Keep slope in f64 for intercept computation to avoid secondary precision
    // loss from truncating slope to f32 before multiplying by l64.
    // Ref: GELU-sound pattern (sound_relax.rs:55-58), ELU (elu.rs:186-187).
    let slope64 = (fu64 - fl64) / (u64 - l64);
    let intercept64 = fl64 - slope64 * l64;

    // Directed rounding: absorb f64→f32 slope truncation error.
    // When slope is truncated, the line `slope_f32 * x + intercept` deviates
    // from the true f64 line by up to |slope_f32 - slope64| * max(|l|, |u|).
    // Ref: Exp (exp.rs:159-166), ELU (elu.rs:197-206).
    let slope_f32 = slope64 as f32;
    let max_abs_x = l.abs().max(u.abs());
    let eval_err = f32_affine_eval_error(slope64, slope_f32, intercept64, max_abs_x);
    (
        slope_f32,
        next_down_f32(f64_to_f32_down(intercept64 - eval_err)),
        next_up_f32(f64_to_f32_up(intercept64 + eval_err)),
    )
}

/// Compute tangent line at point d: y = SiLU'(d) * (x - d) + SiLU(d).
/// Returns (slope, lower_intercept, upper_intercept) with directed rounding.
///
/// `max_abs_x` is the maximum |x| over the interval where this tangent will
/// be evaluated — typically `max(|l|, |u|)` for the relaxation interval [l, u].
///
/// Uses f64 intermediates for the intercept computation to avoid catastrophic
/// cancellation when |d| is large: silu_eval(d) ≈ d and slope * d ≈ d, so
/// the subtraction loses most significant digits in f32.
///
/// Ref: Exp (exp.rs:119-125), ELU (elu.rs:121-124) directed rounding pattern.
/// Fixes: #2434 — SiLU tangent used raw `as f32` without directed rounding.
#[inline]
pub(crate) fn silu_tangent(d: f32, max_abs_x: f32) -> (f32, f32, f32) {
    let d64 = d as f64;
    let slope64 = silu_derivative_f64(d64);
    let intercept64 = silu_eval_f64(d64) - slope64 * d64;

    // Directed rounding: absorb f64→f32 slope truncation error.
    // Ref: Exp (exp.rs:159-166), ELU (elu.rs:197-206).
    let slope_f32 = slope64 as f32;
    let eval_err = f32_affine_eval_error(slope64, slope_f32, intercept64, max_abs_x);
    (
        slope_f32,
        next_down_f32(f64_to_f32_down(intercept64 - eval_err)),
        next_up_f32(f64_to_f32_up(intercept64 + eval_err)),
    )
}

/// Raw tangent line at point d without directed rounding.
/// Returns (slope, intercept) as plain f32 casts. Used for binary search
/// verification where we need the tight (non-rounded) value for candidate
/// selection. The final output should use `silu_tangent` with directed rounding.
#[inline]
pub(crate) fn silu_tangent_raw(d: f32) -> (f32, f32) {
    let d64 = d as f64;
    let slope64 = silu_derivative_f64(d64);
    let intercept64 = silu_eval_f64(d64) - slope64 * d64;
    (slope64 as f32, intercept64 as f32)
}

/// SiLU derivative in f64: σ(x) + x · σ(x) · (1 - σ(x)) = σ(x)(1 + x(1 - σ(x))).
#[inline]
fn silu_derivative_f64(x: f64) -> f64 {
    let s = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    };
    s * (1.0 + x * (1.0 - s))
}

/// The high-precision truth that `silu_critical_point` approximates.
///
/// `SiLU'(x) = σ(x)·(1 + x·(1 - σ(x)))` has exactly one root. Computed with
/// mpmath at dps = 60, re-derived 2026-08-28:
///
/// ```text
/// x*         = -1.27846454276107379510935873902
/// SiLU(x*)   = -0.278464542761073795109358739023
/// SiLU''(x*) =  0.217811705719800098779702924073   (> 0, so x* is the minimum)
/// ```
///
/// Those two values share every digit because `SiLU'(x*) = 0` forces
/// `σ(x*) = 1 + 1/x*`, hence `SiLU(x*) = x*·σ(x*) = x* + 1` exactly (residual
/// 3.9e-62 at dps 60). That identity makes the pair self-checking: the two
/// literals declared here must differ by exactly 1, so an edit to one without
/// the other is visible by inspection.
///
/// This literal is the nearest f64 to `SiLU(x*)` and lies 1.5524e-18 BELOW it,
/// so asserting `certified <= SILU_TRUE_MIN` is strictly HARDER than asserting
/// `certified <= SiLU(x*)`: the pins fail closed and cannot pass by rounding.
#[cfg(test)]
pub(crate) const SILU_TRUE_MIN: f64 = -0.2784645427610738;

/// Nearest f64 to `x*`, lying 1.0947e-16 above it. Derivation and the
/// `x* + 1` cross-check are on [`SILU_TRUE_MIN`].
#[cfg(test)]
pub(crate) const SILU_TRUE_ARGMIN: f64 = -1.2784645427610737;

/// Pins for the numeric premise underneath every SiLU lower bound.
///
/// `silu_critical_point` above finds SiLU's argmin by Newton's method in f32,
/// with a CENTRAL DIFFERENCE second derivative at `eps = 1.0e-5_f32`.
/// `silu_min_max` then evaluates SiLU there and applies `next_down_f32` — ONE
/// f32 ULP of backoff — to publish a certified lower bound. Whether that bound
/// is sound therefore depends on Newton having converged, and until this module
/// landed nothing asserted that — it was arithmetic in a transcript. The same
/// shape was live in SELU on 2026-08-27: `-(LAMBDA * ALPHA)` in f32
/// round-to-nearest sits 2.879e-08 ABOVE the true asymptote, so the certified
/// lower bound sat above a REACHABLE output (SELU(-18)), and the commit that
/// found it shipped a test asserting the value it had just disproved
/// (5d48d59e9, then f3cfeea88). An unasserted premise is how that happens.
///
/// MEASURED 2026-08-28, not argued: `sigmoid`, `silu_derivative` and this loop
/// were replayed bit-for-bit in python3 (struct pack/unpack on every rounding
/// step; f64 is wide enough that computing +-*/ there and rounding once to f32
/// is the correctly-rounded f32 result, and exp was rounded to nearest f32 from
/// mpmath).
///
/// ```text
/// returned x_c = -1.2784645557403564  (bits 0xbfa3a4ba) — the NEAREST f32 to
///                x*, so no f32 can do better. Newton converges in ONE step:
///                iterations 1..19 recompute d == 0.0 exactly, and the `dd`
///                guard never fires.
/// x_c - x*     = -1.297928e-08  =  -0.1089 f32 ULP of x
/// certified    = next_down_f32(silu_eval_f64(x_c) as f32) = -0.27846458554267883
/// margin       = SiLU(x*) - certified = +4.278161e-08 = +1.435512 f32 ULP
/// ```
///
/// Budget, in f32 ULP at |0.2784| (one ULP there is 2.9802e-08): Newton's own
/// excess `0.5·SiLU''(x*)·d²` is +1.83e-17 (+6.2e-10 ULP); the f64 evaluation
/// error is -1.99e-17 (-6.7e-10 ULP); round-to-nearest f32 contributes -0.436
/// ULP (the DOMINANT term); the `next_down_f32` backoff is exactly -1.000 ULP.
/// Round-to-nearest error always lies in [-0.5, +0.5] ULP and the backoff is
/// always exactly -1 ULP, so the margin lies in [+0.5, +1.5] ULP for ANY
/// rounding outcome: the bound is sound STRUCTURALLY, not by luck, and Newton
/// consumes 1.2e-07 % of the worst-case slack.
///
/// Do NOT "fix" this by moving the iteration to f64 or by widening the backoff.
/// Either moves the published bound, hence verdicts, for zero soundness gain —
/// and tighter is the dangerous direction (+10 correct, -150 incorrect).
#[cfg(test)]
mod newton_pin {
    use super::{
        next_down_f32, next_up_f32, silu_critical_point, silu_min_max, SILU_TRUE_ARGMIN,
        SILU_TRUE_MIN,
    };

    /// Radius within which ANY critical point still yields a sound bound.
    ///
    /// Analytic: the quadratic excess `0.5·SiLU''(x*)·d²` eats the worst-case
    /// 0.5-ULP rounding slack at
    /// `|d| = sqrt(2 · 0.5 · 2.9802e-08 / 0.21781170571980) = 3.699e-04`.
    /// Measured by sweeping every f32 within +/-8000 ULP of the returned critical
    /// point and recomputing the full certified bound: the first UNSOUND
    /// critical point is at +/-4528 ULP, i.e. |d| = 5.3978e-04 on both sides. The
    /// analytic figure is the conservative one, so it is the threshold here.
    /// Newton actually lands at |d| = 1.29793e-08 — 28499x inside the radius.
    const NEWTON_ROBUSTNESS_RADIUS: f64 = 3.6e-4;

    /// ZERO tolerance: the certified lower bound `silu_min_max` publishes for an
    /// interval that brackets the minimum must never rise above the true
    /// minimum. A LOOSER bound is always acceptable; a TIGHTER one is a false
    /// proof, and a false proof costs -150 against +10 for a correct verdict.
    #[test]
    fn certified_minimum_never_exceeds_true_silu_minimum() {
        // Every interval here satisfies the strict `l < critical && critical < u`
        // guard, so the critical-point candidate is the one that binds. The two
        // `NEG_INFINITY` cases are the branch `silu_sound_linear_relaxation`
        // actually takes for `l == -inf`, where `min_val` becomes the published
        // constant lower intercept verbatim.
        let cases: [(f32, f32); 5] = [
            (-2.0, 0.0),
            (-3.0, 1.0),
            (-1.5, -1.0),
            (f32::NEG_INFINITY, 0.0),
            (f32::NEG_INFINITY, 2.0),
        ];
        for (l, u) in cases {
            let (min_val, max_val) = silu_min_max(l, u);
            // Anti-vacuity: a NaN or infinite min would settle the comparison
            // below by IEEE ordering rather than by soundness.
            assert!(
                min_val.is_finite(),
                "[{l}, {u}]: certified min {min_val} is not finite — the \
                 comparison below would be decided by ordering, not soundness"
            );
            assert!(
                f64::from(min_val) <= SILU_TRUE_MIN,
                "[{}, {}]: certified lower bound {:.17e} EXCEEDS the true SiLU \
                 minimum {:.17e} — this certifies a bound SiLU goes BELOW. The \
                 shipped f32 Newton result clears it by +1.435512 f32 ULP \
                 (+4.278161e-08), so a failure here means that margin was \
                 CONSUMED; it is not a flaky epsilon.",
                l,
                u,
                f64::from(min_val),
                SILU_TRUE_MIN
            );
            assert!(
                min_val <= max_val,
                "[{l}, {u}]: certified min {min_val} exceeds max {max_val}"
            );
        }
    }

    /// Turns "Newton converged" from an argument into an assertion: the landing
    /// point must sit inside the radius where the published bound stays sound.
    #[test]
    fn newton_lands_inside_the_certified_robustness_radius() {
        let critical = f64::from(silu_critical_point());
        let offset = (critical - SILU_TRUE_ARGMIN).abs();
        assert!(
            offset < NEWTON_ROBUSTNESS_RADIUS,
            "Newton returned {:.17e}, which is {:.6e} from the true argmin \
             {:.17e}. Beyond {:.1e} the quadratic excess 0.5·f''·d² eats the \
             worst-case 0.5-ULP rounding slack and the certified bound stops \
             being sound. Measured landing on the shipped binary: 1.29793e-08.",
            critical,
            offset,
            SILU_TRUE_ARGMIN,
            NEWTON_ROBUSTNESS_RADIUS
        );
    }

    /// `silu_min_max` guards its critical-point candidate with a STRICT
    /// `l < critical && critical < u`. If any f32 fell strictly between the
    /// returned critical point and the true argmin, an interval with an endpoint
    /// there would CONTAIN the true minimum yet SKIP the candidate, and the
    /// endpoint-only min would sit above it. None does, because the returned
    /// point is the nearest f32 to x*: the next f32 up is already past it. That
    /// premise is load-bearing and, before this test, nothing protected it.
    ///
    /// Deliberately CONSERVATIVE, and measured to be so: a mutant that shifts
    /// the critical point ONE ULP toward -inf still publishes a sound bound
    /// (the certified value is unchanged, margin still +1.435512 ULP) and still
    /// clears the robustness radius, yet fires this assertion — because the
    /// unshifted point then lies strictly inside (critical, x*) and the general
    /// argument stops holding for every interval. Red here means "re-derive the
    /// argument", never "loosen the assertion".
    #[test]
    fn no_f32_lies_between_the_critical_point_and_the_true_argmin() {
        let critical = silu_critical_point();
        assert!(
            f64::from(critical) < SILU_TRUE_ARGMIN,
            "critical point {:.17e} is not below the true argmin {:.17e}; the \
             assertion that follows assumes it approaches x* from -inf",
            f64::from(critical),
            SILU_TRUE_ARGMIN
        );
        assert!(
            f64::from(next_up_f32(critical)) > SILU_TRUE_ARGMIN,
            "an f32 ({:.17e}) sits between the critical point {:.17e} and the \
             true argmin {:.17e}. An interval with that endpoint contains the \
             true minimum but fails the strict `l < critical` guard, so \
             silu_min_max would skip the minimum entirely.",
            f64::from(next_up_f32(critical)),
            f64::from(critical),
            SILU_TRUE_ARGMIN
        );
    }

    /// Anti-vacuity plus the property the guard premise rests on: the returned
    /// point is finite, strictly negative, and the NEAREST f32 to x* — both
    /// neighbours are strictly farther away. Measured: |d| = 1.297928e-08
    /// against 1.321886e-07 below and 1.062300e-07 above. Without this, a
    /// `OnceLock` that lost its initialiser and returned 0.0 would still have to
    /// be caught indirectly, through a radius.
    #[test]
    fn critical_point_is_the_nearest_f32_to_the_true_argmin() {
        let critical = silu_critical_point();
        assert!(
            critical.is_finite() && critical < 0.0,
            "critical point is {critical}; SiLU's minimiser is finite and \
             strictly negative, so this value cannot have come from the Newton \
             iteration"
        );
        let here = (f64::from(critical) - SILU_TRUE_ARGMIN).abs();
        let below = (f64::from(next_down_f32(critical)) - SILU_TRUE_ARGMIN).abs();
        let above = (f64::from(next_up_f32(critical)) - SILU_TRUE_ARGMIN).abs();
        assert!(
            here < below && here < above,
            "critical point {:.17e} is {:.6e} from x*, but a neighbour is \
             closer ({:.6e} below, {:.6e} above) — it is no longer the nearest \
             f32 to x*, which is what makes the strict interval guard safe",
            f64::from(critical),
            here,
            below,
            above
        );
    }
}
