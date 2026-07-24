// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound GELU CROWN relaxation (no sampling).
//!
//! Uses precomputed tangent tables for both Erf and Tanh GELU approximations.
//! Reference: α,β-CROWN Team, auto_LiRPA @ 9d100ec070868440b48d34e2f1dd21b97aab9172

use super::eval::{
    gelu_bound_interval, gelu_derivative, gelu_derivative_erf_f64, gelu_derivative_tanh_f64,
    gelu_erf, gelu_erf_f64, gelu_infinite_bounds_relaxation, gelu_tanh, gelu_tanh_f64,
    gelu_tanh_inflection_point,
};
use super::tables::{
    gelu_posthoc_adjust, gelu_tangent_at, gelu_tanh_tangent_at, get_gelu_precompute,
    get_gelu_tanh_precompute, SQRT_2,
};
use crate::rounding::{next_down_f32, next_up_f32};
use crate::types::GeluApproximation;

/// Convert an optionally-assigned bound into a concrete `(slope, intercept)` pair.
///
/// `None` means no case mask assigned a line, so we fall back to a constant
/// bound with `slope = 0`. `Some((0, 0))` is a legitimate assigned line and
/// must be preserved (cannot be treated as an "unset" sentinel).
#[inline]
fn finalize_bound(bound: Option<(f32, f32)>, fallback_intercept: f32) -> (f32, f32) {
    bound.unwrap_or((0.0, fallback_intercept))
}

/// Compute sound linear relaxation for Erf GELU on interval [l, u].
///
/// Uses precomputed tangent tables instead of sampling.
/// Reference: α,β-CROWN Team, auto_LiRPA @ 9d100ec070868440b48d34e2f1dd21b97aab9172
pub fn gelu_sound_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle infinite/NaN bounds: identity relaxation is UNSOUND (see #1837).
    if let Some(result) = gelu_infinite_bounds_relaxation(l, u, GeluApproximation::Erf) {
        return result;
    }

    // Maximum absolute value in the evaluation interval — used to bound slope
    // truncation error for directed rounding (#3156).
    let max_abs_x = l.abs().max(u.abs());

    if (u - l).abs() < 1e-8 {
        // Point interval: compute tangent at l with directed rounding.
        let (slope, lower_i, upper_i) = gelu_tangent_at(l, max_abs_x);
        return (slope, lower_i, slope, upper_i);
    }

    let tables = get_gelu_precompute();

    // Compute chord slope and intercept in f64 to avoid catastrophic cancellation
    // when u ≈ l (both numerator and denominator approach 0 in f32).
    // Same pattern as exp/log relaxation f64 upgrade (#1745).
    // Directed rounding (#3156): absorb f64→f32 slope truncation error in intercepts.
    let gl_64 = gelu_erf_f64(l as f64);
    let gu_64 = gelu_erf_f64(u as f64);
    let k_direct_64 = (gu_64 - gl_64) / (u as f64 - l as f64);
    let b_direct_64 = gl_64 - k_direct_64 * l as f64;
    let k_direct = k_direct_64 as f32;
    let chord_slope_err =
        next_up_f32(((k_direct_64 - k_direct as f64).abs() * max_abs_x as f64) as f32);
    // Account for f32 multiplication rounding: `slope * x` has error up to
    // |slope| * |x| * f32::EPSILON. Same fix as sqrt.rs (#4368).
    let chord_mul_err = next_up_f32((k_direct.abs() * max_abs_x) * f32::EPSILON);
    let b_direct_f32 = b_direct_64 as f32;
    let b_direct_lower = next_down_f32(b_direct_f32 - chord_slope_err - chord_mul_err);
    let b_direct_upper = next_up_f32(b_direct_f32 + chord_slope_err + chord_mul_err);

    // auto_LiRPA "not optimized (vanilla CROWN)" mode uses a mid-point tangent in some cases.
    // Compute in f64 to avoid cancellation in b_mid = gelu(m) - k_mid * m.
    // Directed rounding (#3156): same slope truncation error pattern as chord.
    let m = 0.5 * (l + u);
    let m_64 = m as f64;
    let k_mid_64 = gelu_derivative_erf_f64(m_64);
    let b_mid_64 = gelu_erf_f64(m_64) - k_mid_64 * m_64;
    let k_mid = k_mid_64 as f32;
    let mid_slope_err = next_up_f32(((k_mid_64 - k_mid as f64).abs() * max_abs_x as f64) as f32);
    let mid_mul_err = next_up_f32((k_mid.abs() * max_abs_x) * f32::EPSILON);
    let b_mid_f32 = b_mid_64 as f32;
    let b_mid_lower = next_down_f32(b_mid_f32 - mid_slope_err - mid_mul_err);
    let b_mid_upper = next_up_f32(b_mid_f32 + mid_slope_err + mid_mul_err);

    // Case masks (scalar version of auto_LiRPA's BoundGelu._init_masks()).
    let mask_left_pos = l >= -SQRT_2 && u <= 0.0;
    let mask_left_neg = u <= -SQRT_2;
    let mask_left = (u <= 0.0) ^ (mask_left_pos || mask_left_neg);

    let mask_right_pos = l >= SQRT_2;
    let mask_right_neg = u <= SQRT_2 && l >= 0.0;
    let mask_right = (l >= 0.0) ^ (mask_right_pos || mask_right_neg);

    let mask_2 = u > 0.0 && u <= SQRT_2 && (-SQRT_2..0.0).contains(&l);
    let mask_left_3 = l < -SQRT_2 && u > 0.0 && u <= SQRT_2;
    let mask_right_3 = u > SQRT_2 && (-SQRT_2..0.0).contains(&l);
    let mask_4 = l < -SQRT_2 && u > SQRT_2;
    let mask_both = mask_2 || mask_4 || mask_left_3 || mask_right_3;

    // Track assigned bounds explicitly. `None` means no mask matched.
    let mut lower: Option<(f32, f32)> = None;
    let mut upper: Option<(f32, f32)> = None;

    // Upper bound is always the direct line for left_pos / right_neg / both.
    // Use b_direct_upper (rounded up) for upper bounds.
    if mask_left_pos || mask_right_neg || mask_both {
        upper = Some((k_direct, b_direct_upper));
    }
    // Lower bound is always the direct line for left_neg / right_pos.
    // Use b_direct_lower (rounded down) for lower bounds.
    if mask_left_neg || mask_right_pos {
        lower = Some((k_direct, b_direct_lower));
    }

    // Middle-point tangent bounds in the single-side regions.
    if mask_left_pos || mask_right_neg || mask_2 {
        lower = Some((k_mid, b_mid_lower));
    }
    if mask_right_pos || mask_left_neg {
        upper = Some((k_mid, b_mid_upper));
    }

    // Cross-(-sqrt2) negative intervals: choose between direct line and a left-side tangent.
    if mask_left {
        let use_direct_lower = k_direct > gelu_derivative(u, GeluApproximation::Erf);
        if use_direct_lower {
            lower = Some((k_direct, b_direct_lower));
        } else {
            let d = tables.retrieve(&tables.d_lower_left, -l - SQRT_2, u);
            let (s, lower_i, _upper_i) = gelu_tangent_at(d, max_abs_x);
            lower = Some((s, lower_i));
        }

        let use_direct_upper = k_direct > gelu_derivative(l, GeluApproximation::Erf);
        if use_direct_upper {
            upper = Some((k_direct, b_direct_upper));
        } else {
            let d = tables.retrieve(&tables.d_upper_left, -l - SQRT_2, u);
            let (s, _lower_i, upper_i) = gelu_tangent_at(d, max_abs_x);
            upper = Some((s, upper_i));
        }
    }

    // Cross-(+sqrt2) positive intervals: choose between direct line and a right-side tangent.
    if mask_right {
        let use_direct_lower = k_direct < gelu_derivative(l, GeluApproximation::Erf);
        if use_direct_lower {
            lower = Some((k_direct, b_direct_lower));
        } else {
            let d = tables.retrieve(&tables.d_lower_right, u - SQRT_2, l);
            let (s, lower_i, _upper_i) = gelu_tangent_at(d, max_abs_x);
            lower = Some((s, lower_i));
        }

        let use_direct_upper = k_direct < gelu_derivative(u, GeluApproximation::Erf);
        if use_direct_upper {
            upper = Some((k_direct, b_direct_upper));
        } else {
            let d = tables.retrieve(&tables.d_upper_right, -l + SQRT_2, u);
            let (s, _lower_i, upper_i) = gelu_tangent_at(d, max_abs_x);
            upper = Some((s, upper_i));
        }
    }

    // Cross-zero intervals: upper is always the direct line, lower depends on which subregion we cross.
    if mask_left_3 {
        let d = tables.retrieve(&tables.d_lower_left, -l - SQRT_2, u);
        let (s, lower_i, _upper_i) = gelu_tangent_at(d, max_abs_x);
        lower = Some((s, lower_i));
    }
    if mask_right_3 || mask_4 {
        let d = tables.retrieve(&tables.d_lower_right, u - SQRT_2, l);
        let (s, lower_i, _upper_i) = gelu_tangent_at(d, max_abs_x);
        lower = Some((s, lower_i));
    }

    let (min_v, max_v) = gelu_bound_interval(l, u, GeluApproximation::Erf);
    let (lower_slope, lower_intercept) = finalize_bound(lower, min_v);
    let (upper_slope, upper_intercept) = finalize_bound(upper, max_v);

    // Post-hoc soundness verification: table precomputation uses f64 bisection for
    // high-precision tangent points, but step=0.01 discretization and f32 tangent
    // evaluation at runtime can still cause small violations. This safety net
    // checks 5 sample points and adjusts intercepts by the observed violation.
    gelu_posthoc_adjust(
        l,
        u,
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
        gelu_erf,
    )
}

/// Compute sound linear relaxation for tanh-approx GELU on interval [l, u].
///
/// Uses precomputed tangent tables instead of sampling.
pub fn gelu_tanh_sound_linear_relaxation(l: f32, u: f32) -> (f32, f32, f32, f32) {
    // Handle infinite/NaN bounds: identity relaxation is UNSOUND (see #1837).
    if let Some(result) = gelu_infinite_bounds_relaxation(l, u, GeluApproximation::Tanh) {
        return result;
    }

    // Maximum absolute value in the evaluation interval — directed rounding (#3156).
    let max_abs_x = l.abs().max(u.abs());

    if (u - l).abs() < 1e-8 {
        // Point interval: compute tangent at l with directed rounding.
        let (slope, lower_i, upper_i) = gelu_tanh_tangent_at(l, max_abs_x);
        return (slope, lower_i, slope, upper_i);
    }

    let tables = get_gelu_tanh_precompute();
    let split = gelu_tanh_inflection_point();

    // Compute chord slope and intercept in f64 to avoid catastrophic cancellation
    // when u ≈ l (both numerator and denominator approach 0 in f32).
    // Same pattern as exp/log relaxation f64 upgrade (#1745).
    // Directed rounding (#3156): absorb f64→f32 slope truncation error in intercepts.
    let gl_64 = gelu_tanh_f64(l as f64);
    let gu_64 = gelu_tanh_f64(u as f64);
    let k_direct_64 = (gu_64 - gl_64) / (u as f64 - l as f64);
    let b_direct_64 = gl_64 - k_direct_64 * l as f64;
    let k_direct = k_direct_64 as f32;
    let chord_slope_err =
        next_up_f32(((k_direct_64 - k_direct as f64).abs() * max_abs_x as f64) as f32);
    let chord_mul_err = next_up_f32((k_direct.abs() * max_abs_x) * f32::EPSILON);
    let b_direct_f32 = b_direct_64 as f32;
    let b_direct_lower = next_down_f32(b_direct_f32 - chord_slope_err - chord_mul_err);
    let b_direct_upper = next_up_f32(b_direct_f32 + chord_slope_err + chord_mul_err);

    // Compute midpoint tangent in f64 to avoid cancellation in b_mid = gelu(m) - k_mid * m.
    // Directed rounding (#3156): same slope truncation error pattern as chord.
    let m = 0.5 * (l + u);
    let m_64 = m as f64;
    let k_mid_64 = gelu_derivative_tanh_f64(m_64);
    let b_mid_64 = gelu_tanh_f64(m_64) - k_mid_64 * m_64;
    let k_mid = k_mid_64 as f32;
    let mid_slope_err = next_up_f32(((k_mid_64 - k_mid as f64).abs() * max_abs_x as f64) as f32);
    let mid_mul_err = next_up_f32((k_mid.abs() * max_abs_x) * f32::EPSILON);
    let b_mid_f32 = b_mid_64 as f32;
    let b_mid_lower = next_down_f32(b_mid_f32 - mid_slope_err - mid_mul_err);
    let b_mid_upper = next_up_f32(b_mid_f32 + mid_slope_err + mid_mul_err);

    // Case masks (tanh-approx split at ±split).
    let mask_left_pos = l >= -split && u <= 0.0;
    let mask_left_neg = u <= -split;
    let mask_left = (u <= 0.0) ^ (mask_left_pos || mask_left_neg);

    let mask_right_pos = l >= split;
    let mask_right_neg = u <= split && l >= 0.0;
    let mask_right = (l >= 0.0) ^ (mask_right_pos || mask_right_neg);

    let mask_2 = u > 0.0 && u <= split && l < 0.0 && l >= -split;
    let mask_left_3 = l < -split && u > 0.0 && u <= split;
    let mask_right_3 = u > split && l < 0.0 && l >= -split;
    let mask_4 = l < -split && u > split;
    let mask_both = mask_2 || mask_4 || mask_left_3 || mask_right_3;

    let mut lower: Option<(f32, f32)> = None;
    let mut upper: Option<(f32, f32)> = None;

    if mask_left_pos || mask_right_neg || mask_both {
        upper = Some((k_direct, b_direct_upper));
    }
    if mask_left_neg || mask_right_pos {
        lower = Some((k_direct, b_direct_lower));
    }

    if mask_left_pos || mask_right_neg || mask_2 {
        lower = Some((k_mid, b_mid_lower));
    }
    if mask_right_pos || mask_left_neg {
        upper = Some((k_mid, b_mid_upper));
    }

    if mask_left {
        let use_direct_lower = k_direct > gelu_derivative(u, GeluApproximation::Tanh);
        if use_direct_lower {
            lower = Some((k_direct, b_direct_lower));
        } else {
            let d = tables.retrieve(&tables.d_lower_left, -l - split, u);
            let (s, lower_i, _upper_i) = gelu_tanh_tangent_at(d, max_abs_x);
            lower = Some((s, lower_i));
        }

        let use_direct_upper = k_direct > gelu_derivative(l, GeluApproximation::Tanh);
        if use_direct_upper {
            upper = Some((k_direct, b_direct_upper));
        } else {
            let d = tables.retrieve(&tables.d_upper_left, -l - split, u);
            let (s, _lower_i, upper_i) = gelu_tanh_tangent_at(d, max_abs_x);
            upper = Some((s, upper_i));
        }
    }

    if mask_right {
        let use_direct_lower = k_direct < gelu_derivative(l, GeluApproximation::Tanh);
        if use_direct_lower {
            lower = Some((k_direct, b_direct_lower));
        } else {
            let d = tables.retrieve(&tables.d_lower_right, u - split, l);
            let (s, lower_i, _upper_i) = gelu_tanh_tangent_at(d, max_abs_x);
            lower = Some((s, lower_i));
        }

        let use_direct_upper = k_direct < gelu_derivative(u, GeluApproximation::Tanh);
        if use_direct_upper {
            upper = Some((k_direct, b_direct_upper));
        } else {
            let d = tables.retrieve(&tables.d_upper_right, -l + split, u);
            let (s, _lower_i, upper_i) = gelu_tanh_tangent_at(d, max_abs_x);
            upper = Some((s, upper_i));
        }
    }

    if mask_left_3 {
        let d = tables.retrieve(&tables.d_lower_left, -l - split, u);
        let (s, lower_i, _upper_i) = gelu_tanh_tangent_at(d, max_abs_x);
        lower = Some((s, lower_i));
    }
    if mask_right_3 || mask_4 {
        let d = tables.retrieve(&tables.d_lower_right, u - split, l);
        let (s, lower_i, _upper_i) = gelu_tanh_tangent_at(d, max_abs_x);
        lower = Some((s, lower_i));
    }

    let (min_v, max_v) = gelu_bound_interval(l, u, GeluApproximation::Tanh);
    let (lower_slope, lower_intercept) = finalize_bound(lower, min_v);
    let (upper_slope, upper_intercept) = finalize_bound(upper, max_v);

    // Post-hoc soundness verification (same as Erf path).
    gelu_posthoc_adjust(
        l,
        u,
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
        gelu_tanh,
    )
}

/// Compute sound linear relaxation for Erf GELU with parameterized lower bound tangent point.
///
/// `alpha` in [0, 1] controls the tangent point for the lower bound in the convex
/// region (between GELU inflection points at ±√2). The tangent point is:
///   t = l + alpha * (u - l)
///
/// When alpha = 0.5 (default), this produces the midpoint tangent — equivalent to
/// the standard `gelu_sound_linear_relaxation`. Different alpha values can be
/// optimized per-neuron to tighten the CROWN objective (alpha-CROWN for GELU).
///
/// **Soundness:** In the convex region, any tangent is a valid lower bound because
/// a convex function lies above all its tangent lines. The upper bound (chord) is
/// unchanged. For non-convex regions (outside ±√2), falls back to the standard
/// sound relaxation (alpha is ignored).
///
/// # Reference
/// alpha-beta-CROWN BoundOptimizableActivation, auto_LiRPA/operators/activations.py
pub fn gelu_sound_linear_relaxation_with_alpha(l: f32, u: f32, alpha: f32) -> (f32, f32, f32, f32) {
    // Clamp alpha to [0, 1].
    let alpha = alpha.clamp(0.0, 1.0);

    // Handle infinite/NaN bounds: delegate to standard relaxation.
    if let Some(result) = gelu_infinite_bounds_relaxation(l, u, GeluApproximation::Erf) {
        return result;
    }

    let max_abs_x = l.abs().max(u.abs());

    if (u - l).abs() < 1e-8 {
        let (slope, lower_i, upper_i) = gelu_tangent_at(l, max_abs_x);
        return (slope, lower_i, slope, upper_i);
    }

    // Determine which case mask applies.
    let convex_region = l >= -SQRT_2 && u <= SQRT_2;

    if !convex_region {
        // Outside the convex region: alpha parameterization doesn't apply cleanly.
        // Fall back to the standard sound relaxation.
        return gelu_sound_linear_relaxation(l, u);
    }

    // Convex region: GELU is convex on [-√2, √2] because g''(x) = φ(x)(2 - x²) > 0.
    // Any tangent line at t ∈ [l, u] is a valid lower bound.
    // The chord from (l, GELU(l)) to (u, GELU(u)) is a valid upper bound.

    // Compute upper bound: always the chord (tight for convex functions).
    let gl_64 = gelu_erf_f64(l as f64);
    let gu_64 = gelu_erf_f64(u as f64);
    let k_direct_64 = (gu_64 - gl_64) / (u as f64 - l as f64);
    let b_direct_64 = gl_64 - k_direct_64 * l as f64;
    let k_direct = k_direct_64 as f32;
    let chord_slope_err =
        next_up_f32(((k_direct_64 - k_direct as f64).abs() * max_abs_x as f64) as f32);
    let chord_mul_err = next_up_f32((k_direct.abs() * max_abs_x) * f32::EPSILON);
    let b_direct_f32 = b_direct_64 as f32;
    let b_direct_upper = next_up_f32(b_direct_f32 + chord_slope_err + chord_mul_err);

    // Compute lower bound: tangent at alpha-parameterized point.
    let t = l + alpha * (u - l);
    let (tangent_slope, tangent_lower_intercept, _tangent_upper_intercept) =
        gelu_tangent_at(t, max_abs_x);

    let (min_v, max_v) = gelu_bound_interval(l, u, GeluApproximation::Erf);

    // Apply masks for the specific sub-cases within the convex region.
    let mask_left_pos = l >= -SQRT_2 && u <= 0.0; // concave-ish sub-region
    let mask_right_neg = u <= SQRT_2 && l >= 0.0; // convex sub-region
    let mask_2 = u > 0.0 && u <= SQRT_2 && (-SQRT_2..0.0).contains(&l); // crossing zero

    // For all three sub-cases in the convex region, the tangent is a valid lower bound.
    let (lower_slope, lower_intercept) = if mask_left_pos || mask_right_neg || mask_2 {
        (tangent_slope, tangent_lower_intercept)
    } else {
        // Shouldn't reach here if convex_region is true, but guard.
        (0.0, min_v)
    };

    // Upper bound depends on sub-case.
    let (upper_slope, upper_intercept) = if mask_left_pos || mask_right_neg || mask_2 {
        // Convex: chord is the upper bound.
        (k_direct, b_direct_upper)
    } else {
        (0.0, max_v)
    };

    // Post-hoc soundness verification.
    gelu_posthoc_adjust(
        l,
        u,
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
        gelu_erf,
    )
}
