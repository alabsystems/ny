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
