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
use super::sound_tables::{
    gelu_posthoc_adjust, gelu_tangent_at, gelu_tanh_tangent_at, get_gelu_precompute,
    get_gelu_tanh_precompute, SQRT_2,
};
use super::GeluApproximation;
use ny_tensor::{next_down_f32, next_up_f32};

/// Convert an optionally-assigned bound into a concrete `(slope, intercept)` pair.
///
/// `None` means no case mask assigned a line, so we fall back to a constant
/// bound with `slope = 0`. `Some((0, 0))` is a legitimate assigned line and
/// must be preserved (cannot be treated as an "unset" sentinel).
#[inline]
fn finalize_bound(bound: Option<(f32, f32)>, fallback_intercept: f32) -> (f32, f32) {
    bound.unwrap_or((0.0, fallback_intercept))
}

/// Clamp a sloped lower-relaxation line up to the constant interval-minimum floor
/// when the floor is the tighter sound lower bound.
///
/// GELU is bounded below by its global minimum (≈ −0.170 at x ≈ −0.752), so over
/// any interval the true minimum `min_v` (computed by `gelu_bound_interval`,
/// already rounded down for soundness) gives a sound *constant* lower line
/// `y = min_v`. The single-tangent lower lines selected for the wide negative-tail
/// case masks (`mask_4`, `mask_left_3`, `mask_right_3`, and the `mask_left`/`mask_right`
/// tangent branches) plunge far below `min_v` on the left because their slope is
/// positive (≈1.08): e.g. on `[-6.5, 1.5]` the tangent concretizes to ≈ −7.29 while
/// the true GELU minimum is −0.17 — a ~7-wide spurious lower gap that dominates the
/// op's bound width.
///
/// Both lines are valid sound lower bounds (a convex/secant tangent below GELU, and
/// the constant `min_v ≤ GELU(x)` everywhere), so selecting whichever has the higher
/// concretized minimum over `[l, u]` is a pure tighten-or-equal choice. The line is
/// affine in `x`, so its concretized minimum is at an endpoint. When the floor wins
/// we drop to slope 0 (the constant floor); otherwise the sloped line is kept exactly.
///
/// Soundness: `(0, min_v)` satisfies `min_v ≤ GELU(x)` for all `x ∈ [l, u]` because
/// `min_v` is the directed-rounded interval minimum (endpoints plus the interior
/// critical point when present). Never loosens: only replaces the line when the
/// floor's concretized value is strictly higher.
#[inline]
fn clamp_lower_to_floor(
    lower_slope: f32,
    lower_intercept: f32,
    l: f32,
    u: f32,
    min_v: f32,
) -> (f32, f32) {
    // A sloped line (typically positive slope ≈1.08 in the wide-negative-tail masks)
    // dips well below the floor at one endpoint; the `min_v > concretized` test below
    // handles every slope sign uniformly, including the already-constant slope-0 line.
    if !lower_slope.is_finite() || !lower_intercept.is_finite() || !min_v.is_finite() {
        return (lower_slope, lower_intercept);
    }
    let at_l = lower_slope * l + lower_intercept;
    let at_u = lower_slope * u + lower_intercept;
    let concretized = nan_min(at_l, at_u);
    if min_v > concretized {
        // Constant floor is the tighter (higher) sound lower bound.
        (0.0, min_v)
    } else {
        (lower_slope, lower_intercept)
    }
}

#[inline]
fn nan_min(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else {
        a.min(b)
    }
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
    let b_direct_f32 = b_direct_64 as f32;
    let b_direct_lower = next_down_f32(b_direct_f32 - chord_slope_err);
    let b_direct_upper = next_up_f32(b_direct_f32 + chord_slope_err);

    // auto_LiRPA "not optimized (vanilla CROWN)" mode uses a mid-point tangent in some cases.
    // Compute in f64 to avoid cancellation in b_mid = gelu(m) - k_mid * m.
    // Directed rounding (#3156): same slope truncation error pattern as chord.
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let m = 0.5 * (l + u);
    let m_64 = m as f64;
    let k_mid_64 = gelu_derivative_erf_f64(m_64);
    let b_mid_64 = gelu_erf_f64(m_64) - k_mid_64 * m_64;
    let k_mid = k_mid_64 as f32;
    let mid_slope_err = next_up_f32(((k_mid_64 - k_mid as f64).abs() * max_abs_x as f64) as f32);
    let b_mid_f32 = b_mid_64 as f32;
    let b_mid_lower = next_down_f32(b_mid_f32 - mid_slope_err);
    let b_mid_upper = next_up_f32(b_mid_f32 + mid_slope_err);

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

    // Clamp the lower line up to the constant interval-minimum floor when that floor
    // is the tighter sound lower bound (wide negative-tail tangents plunge far below
    // GELU's global minimum ≈ −0.17). Tighten-or-equal; see `clamp_lower_to_floor`.
    let (lower_slope, lower_intercept) =
        clamp_lower_to_floor(lower_slope, lower_intercept, l, u, min_v);

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
    let b_direct_f32 = b_direct_64 as f32;
    let b_direct_lower = next_down_f32(b_direct_f32 - chord_slope_err);
    let b_direct_upper = next_up_f32(b_direct_f32 + chord_slope_err);

    // Compute midpoint tangent in f64 to avoid cancellation in b_mid = gelu(m) - k_mid * m.
    // Directed rounding (#3156): same slope truncation error pattern as chord.
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let m = 0.5 * (l + u);
    let m_64 = m as f64;
    let k_mid_64 = gelu_derivative_tanh_f64(m_64);
    let b_mid_64 = gelu_tanh_f64(m_64) - k_mid_64 * m_64;
    let k_mid = k_mid_64 as f32;
    let mid_slope_err = next_up_f32(((k_mid_64 - k_mid as f64).abs() * max_abs_x as f64) as f32);
    let b_mid_f32 = b_mid_64 as f32;
    let b_mid_lower = next_down_f32(b_mid_f32 - mid_slope_err);
    let b_mid_upper = next_up_f32(b_mid_f32 + mid_slope_err);

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

    // Clamp the lower line up to the constant interval-minimum floor (see Erf path
    // and `clamp_lower_to_floor`). Tighten-or-equal.
    let (lower_slope, lower_intercept) =
        clamp_lower_to_floor(lower_slope, lower_intercept, l, u, min_v);

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
/// # Arguments
/// * `l` - Lower bound of the pre-activation interval
/// * `u` - Upper bound of the pre-activation interval
/// * `alpha` - Tangent point parameter in [0, 1]. 0.5 = midpoint (default behavior)
///
/// # Returns
/// `(lower_slope, lower_intercept, upper_slope, upper_intercept)`
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
    let b_direct_f32 = b_direct_64 as f32;
    let b_direct_upper = next_up_f32(b_direct_f32 + chord_slope_err);

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

    // NOTE: the floor clamp (see `clamp_lower_to_floor`) is intentionally NOT applied on
    // the alpha-parameterized path. The alpha-CROWN optimizer differentiates the lower
    // line w.r.t. the tangent point; collapsing it to a constant floor would kill that
    // gradient. The alpha lower line is still a sound (convex-region tangent) bound; the
    // floor tightening lands on the standard (non-alpha) relaxation that IBP/vanilla-CROWN
    // use. (The alpha path remains sound, just not floor-tightened.)

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_relaxation_sound;

    #[test]
    fn test_finalize_bound_preserves_legitimate_zero_bound() {
        let bound = finalize_bound(Some((0.0, 0.0)), -0.25);
        assert_eq!(bound, (0.0, 0.0));
    }

    #[test]
    fn test_finalize_bound_falls_back_when_unset() {
        let bound = finalize_bound(None, -0.25);
        assert_eq!(bound, (0.0, -0.25));
    }

    // =========================================================================
    // Erf sound relaxation
    // =========================================================================

    /// Sound Erf relaxation on a typical cross-zero interval.
    #[test]
    fn test_sound_erf_cross_zero() {
        let r = gelu_sound_linear_relaxation(-1.5, 1.5);
        assert_relaxation_sound(-1.5, 1.5, r.into(), gelu_erf, 1e-4, "SoundErf[-1.5,1.5]");
    }

    /// Sound Erf relaxation on purely positive interval (convex region).
    #[test]
    fn test_sound_erf_positive() {
        let r = gelu_sound_linear_relaxation(0.5, 3.0);
        assert_relaxation_sound(0.5, 3.0, r.into(), gelu_erf, 1e-4, "SoundErf[0.5,3.0]");
    }

    /// Sound Erf relaxation on purely negative interval (concave region).
    #[test]
    fn test_sound_erf_negative() {
        let r = gelu_sound_linear_relaxation(-3.0, -0.5);
        assert_relaxation_sound(-3.0, -0.5, r.into(), gelu_erf, 1e-4, "SoundErf[-3,-0.5]");
    }

    /// Sound Erf relaxation: point interval returns derivative-based.
    #[test]
    fn test_sound_erf_point_interval() {
        let (ls, li, us, ui) = gelu_sound_linear_relaxation(1.0, 1.0);
        assert!((ls - us).abs() < 1e-6, "Point interval slopes should match");
        assert!(
            (li - ui).abs() < 1e-6,
            "Point interval intercepts should match"
        );
    }

    /// Sound Erf: infinite bounds should not produce NaN.
    #[test]
    fn test_sound_erf_infinite_bounds() {
        let (ls, li, us, ui) = gelu_sound_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY);
        assert!(!ls.is_nan(), "lower slope NaN");
        assert!(!li.is_nan(), "lower intercept NaN");
        assert!(!us.is_nan(), "upper slope NaN");
        assert!(!ui.is_nan(), "upper intercept NaN");
    }

    /// Sweep Erf sound relaxation across many intervals.
    #[test]
    fn test_sound_erf_sweep() {
        let intervals: Vec<(f32, f32)> = vec![
            (-5.0, -2.0),
            (-2.0, -1.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 5.0),
            (-0.5, 0.5),
            (-3.0, 0.0),
            (0.0, 3.0),
            (-1.0, 2.0),
            (-2.0, 1.0),
        ];
        for (l, u) in intervals {
            let r = gelu_sound_linear_relaxation(l, u);
            assert_relaxation_sound(
                l,
                u,
                r.into(),
                gelu_erf,
                1e-4,
                &format!("SoundErf[{l},{u}]"),
            );
        }
    }

    // =========================================================================
    // Tanh sound relaxation
    // =========================================================================

    /// Sound Tanh relaxation on a typical cross-zero interval.
    #[test]
    fn test_sound_tanh_cross_zero() {
        let r = gelu_tanh_sound_linear_relaxation(-1.5, 1.5);
        assert_relaxation_sound(-1.5, 1.5, r.into(), gelu_tanh, 1e-4, "SoundTanh[-1.5,1.5]");
    }

    /// Sound Tanh relaxation on purely positive interval.
    #[test]
    fn test_sound_tanh_positive() {
        let r = gelu_tanh_sound_linear_relaxation(0.5, 3.0);
        assert_relaxation_sound(0.5, 3.0, r.into(), gelu_tanh, 1e-4, "SoundTanh[0.5,3.0]");
    }

    /// Sound Tanh relaxation on purely negative interval.
    #[test]
    fn test_sound_tanh_negative() {
        let r = gelu_tanh_sound_linear_relaxation(-3.0, -0.5);
        assert_relaxation_sound(-3.0, -0.5, r.into(), gelu_tanh, 1e-4, "SoundTanh[-3,-0.5]");
    }

    /// Sound Tanh: point interval.
    #[test]
    fn test_sound_tanh_point_interval() {
        let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(1.0, 1.0);
        assert!((ls - us).abs() < 1e-6, "Point interval slopes should match");
        assert!(
            (li - ui).abs() < 1e-6,
            "Point interval intercepts should match"
        );
    }

    /// Sound Tanh: infinite bounds should not produce NaN.
    #[test]
    fn test_sound_tanh_infinite_bounds() {
        let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY);
        assert!(!ls.is_nan(), "lower slope NaN");
        assert!(!li.is_nan(), "lower intercept NaN");
        assert!(!us.is_nan(), "upper slope NaN");
        assert!(!ui.is_nan(), "upper intercept NaN");
    }

    /// Sweep Tanh sound relaxation across many intervals.
    #[test]
    fn test_sound_tanh_sweep() {
        let intervals: Vec<(f32, f32)> = vec![
            (-5.0, -2.0),
            (-2.0, -1.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (1.0, 2.0),
            (2.0, 5.0),
            (-0.5, 0.5),
            (-3.0, 0.0),
            (0.0, 3.0),
            (-1.0, 2.0),
            (-2.0, 1.0),
        ];
        for (l, u) in intervals {
            let r = gelu_tanh_sound_linear_relaxation(l, u);
            assert_relaxation_sound(
                l,
                u,
                r.into(),
                gelu_tanh,
                1e-4,
                &format!("SoundTanh[{l},{u}]"),
            );
        }
    }

    // =========================================================================
    // Comparison: sound should be at least as tight as maximally-loose
    // =========================================================================

    /// Sound bounds should have finite intercepts for finite intervals (not maximally loose).
    #[test]
    fn test_sound_bounds_not_maximally_loose() {
        let (ls, li, us, ui) = gelu_sound_linear_relaxation(-1.0, 1.0);
        // Maximally loose would be (0, -inf, 0, +inf).
        assert!(
            li.is_finite(),
            "Sound erf [-1,1] lower intercept should be finite, got {li}"
        );
        assert!(
            ui.is_finite(),
            "Sound erf [-1,1] upper intercept should be finite, got {ui}"
        );
        // Non-trivial slopes for non-point intervals
        let has_nonzero = ls.abs() > 1e-10 || us.abs() > 1e-10;
        assert!(
            has_nonzero || (li.is_finite() && ui.is_finite()),
            "Sound erf [-1,1] should produce non-trivial bounds"
        );

        let (ls_t, li, us_t, ui) = gelu_tanh_sound_linear_relaxation(-1.0, 1.0);
        assert!(
            li.is_finite(),
            "Sound tanh [-1,1] lower intercept should be finite, got {li}"
        );
        assert!(
            ui.is_finite(),
            "Sound tanh [-1,1] upper intercept should be finite, got {ui}"
        );
        let has_nonzero_t = ls_t.abs() > 1e-10 || us_t.abs() > 1e-10;
        assert!(
            has_nonzero_t || (li.is_finite() && ui.is_finite()),
            "Sound tanh [-1,1] should produce non-trivial bounds"
        );
    }

    // =========================================================================
    // Narrow-interval regression tests for f64 upgrade (#2488)
    // =========================================================================

    /// Narrow intervals just above the 1e-8 early-return threshold trigger
    /// the chord computation (gu - gl) / (u - l). In f32, both numerator and
    /// denominator approach 0, causing catastrophic cancellation.
    /// The f64 upgrade (#2488) prevents this.
    #[test]
    fn test_sound_erf_narrow_interval_no_cancellation() {
        // Interval width 1e-5 at various locations — well above 1e-8 threshold
        // but narrow enough that f32 chord would suffer cancellation.
        let test_centers: &[f32] = &[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
        let half_width: f32 = 5e-6;
        for &c in test_centers {
            let l = c - half_width;
            let u = c + half_width;
            let (ls, li, us, ui) = gelu_sound_linear_relaxation(l, u);
            assert!(
                ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite(),
                "Erf narrow [{l},{u}]: got NaN/Inf: ls={ls}, li={li}, us={us}, ui={ui}"
            );
            // Verify soundness: bounds must contain GELU at endpoints and midpoint.
            let gx_l = gelu_erf(l);
            let gx_u = gelu_erf(u);
            let gx_m = gelu_erf(c);
            assert!(
                ls * l + li <= gx_l + 1e-4,
                "Erf narrow [{l},{u}]: lower bound violates at l"
            );
            assert!(
                us * u + ui >= gx_u - 1e-4,
                "Erf narrow [{l},{u}]: upper bound violates at u"
            );
            assert!(
                ls * c + li <= gx_m + 1e-4,
                "Erf narrow [{l},{u}]: lower bound violates at midpoint"
            );
            assert!(
                us * c + ui >= gx_m - 1e-4,
                "Erf narrow [{l},{u}]: upper bound violates at midpoint"
            );
        }
    }

    /// Same narrow-interval cancellation test for the Tanh path.
    #[test]
    fn test_sound_tanh_narrow_interval_no_cancellation() {
        let test_centers: &[f32] = &[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
        let half_width: f32 = 5e-6;
        for &c in test_centers {
            let l = c - half_width;
            let u = c + half_width;
            let (ls, li, us, ui) = gelu_tanh_sound_linear_relaxation(l, u);
            assert!(
                ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite(),
                "Tanh narrow [{l},{u}]: got NaN/Inf: ls={ls}, li={li}, us={us}, ui={ui}"
            );
            let gx_l = gelu_tanh(l);
            let gx_u = gelu_tanh(u);
            let gx_m = gelu_tanh(c);
            assert!(
                ls * l + li <= gx_l + 1e-4,
                "Tanh narrow [{l},{u}]: lower bound violates at l"
            );
            assert!(
                us * u + ui >= gx_u - 1e-4,
                "Tanh narrow [{l},{u}]: upper bound violates at u"
            );
            assert!(
                ls * c + li <= gx_m + 1e-4,
                "Tanh narrow [{l},{u}]: lower bound violates at midpoint"
            );
            assert!(
                us * c + ui >= gx_m - 1e-4,
                "Tanh narrow [{l},{u}]: upper bound violates at midpoint"
            );
        }
    }

    // =========================================================================
    // Alpha-parameterized relaxation (Phase 4, #3221)
    // =========================================================================

    /// Alpha-parameterized relaxation is sound for all alpha in [0, 1].
    #[test]
    fn test_alpha_relaxation_sound_sweep() {
        let intervals: &[(f32, f32)] = &[
            (-1.0, 1.0),
            (-0.5, 0.5),
            (0.0, 1.0),
            (-1.0, 0.0),
            (-1.4, 1.4), // Near inflection points
            (0.1, 1.2),
            (-1.2, -0.1),
        ];
        let alphas: &[f32] = &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];

        for &(l, u) in intervals {
            for &alpha in alphas {
                let r = gelu_sound_linear_relaxation_with_alpha(l, u, alpha);
                assert_relaxation_sound(
                    l,
                    u,
                    r.into(),
                    gelu_erf,
                    1e-4,
                    &format!("Alpha({alpha:.2})[{l},{u}]"),
                );
            }
        }
    }

    /// Alpha=0.5 matches the standard relaxation in the convex region **except**
    /// where the standard path's floor clamp (`clamp_lower_to_floor`) raises the
    /// lower line above the midpoint tangent. The alpha path is intentionally left
    /// unclamped (to preserve the alpha-CROWN gradient), so:
    ///   - the upper bound (chord) is always identical, and
    ///   - the standard lower line is >= the alpha=0.5 lower line (tighter-or-equal).
    #[test]
    fn test_alpha_0_5_matches_standard_in_convex_region() {
        let intervals: &[(f32, f32)] = &[(-0.5, 0.5), (0.0, 1.0), (-1.0, 0.0), (-1.0, 1.0)];

        for &(l, u) in intervals {
            let standard = gelu_sound_linear_relaxation(l, u);
            let alpha_0_5 = gelu_sound_linear_relaxation_with_alpha(l, u, 0.5);

            // Upper bound (chord) is unaffected by the floor clamp.
            assert!(
                (standard.2 - alpha_0_5.2).abs() < 1e-5
                    && (standard.3 - alpha_0_5.3).abs() < 1e-5,
                "Upper bound should match for [{l},{u}]: standard=({:.6},{:.6}), alpha=({:.6},{:.6})",
                standard.2,
                standard.3,
                alpha_0_5.2,
                alpha_0_5.3,
            );

            // Standard lower line must be tighter-or-equal to the alpha=0.5 tangent:
            // its concretized minimum is >= the alpha path's concretized minimum.
            let std_lo = concretized_lower(standard.0, standard.1, l, u);
            let alpha_lo = concretized_lower(alpha_0_5.0, alpha_0_5.1, l, u);
            assert!(
                std_lo >= alpha_lo - 1e-4,
                "Standard lower should be tighter-or-equal to alpha=0.5 for [{l},{u}]: \
                 std_lo={std_lo}, alpha_lo={alpha_lo}",
            );
        }
    }

    /// Outside the convex region, alpha relaxation falls back to standard.
    #[test]
    fn test_alpha_falls_back_outside_convex_region() {
        let intervals: &[(f32, f32)] = &[
            (-3.0, -2.0), // Concave region
            (2.0, 3.0),   // Concave region
            (-3.0, 0.0),  // Crosses inflection point
        ];

        for &(l, u) in intervals {
            let standard = gelu_sound_linear_relaxation(l, u);
            let alpha_any = gelu_sound_linear_relaxation_with_alpha(l, u, 0.3);
            // Outside convex region, alpha is ignored → same as standard.
            assert_eq!(
                standard, alpha_any,
                "Outside convex region [{l},{u}]: alpha should be ignored"
            );
        }
    }

    /// Different alpha values should produce different lower bound slopes
    /// in the convex region.
    #[test]
    fn test_alpha_produces_different_slopes() {
        let (l, u) = (-1.0, 1.0);
        let r_0 = gelu_sound_linear_relaxation_with_alpha(l, u, 0.0);
        let r_half = gelu_sound_linear_relaxation_with_alpha(l, u, 0.5);
        let r_1 = gelu_sound_linear_relaxation_with_alpha(l, u, 1.0);

        // Different alphas should give different lower bound slopes.
        assert!(
            (r_0.0 - r_half.0).abs() > 1e-6 || (r_0.0 - r_1.0).abs() > 1e-6,
            "Different alphas should produce different lower slopes: \
             a=0 slope={:.6}, a=0.5 slope={:.6}, a=1.0 slope={:.6}",
            r_0.0,
            r_half.0,
            r_1.0,
        );

        // Upper bound (chord) should be the same for all alpha values.
        assert!(
            (r_0.2 - r_half.2).abs() < 1e-6 && (r_0.2 - r_1.2).abs() < 1e-6,
            "Upper slope should be same for all alphas (chord): \
             a=0={:.6}, a=0.5={:.6}, a=1.0={:.6}",
            r_0.2,
            r_half.2,
            r_1.2,
        );
    }

    // =========================================================================
    // Floor-clamp lower bound (wide negative-tail tightening)
    // =========================================================================

    use crate::layers::softmax::gelu::eval::gelu_bound_interval;
    use proptest::prelude::*;

    /// Concretized lower bound of an affine line over [l, u] (extreme at an endpoint).
    fn concretized_lower(slope: f32, intercept: f32, l: f32, u: f32) -> f32 {
        (slope * l + intercept).min(slope * u + intercept)
    }

    /// Tightening regression: on wide-negative-tail intervals the lower line must
    /// now track GELU's global minimum (≈ −0.17), not plunge tens of units below it.
    /// Before the floor clamp, e.g. `[-6.5, 1.5]` concretized to ≈ −7.29.
    #[test]
    fn floor_clamp_recovers_wide_tail_lower_bound() {
        for &(l, u) in &[(-6.5_f32, 1.5_f32), (-6.0, 2.0), (-4.0, 4.0), (-8.0, 0.5)] {
            for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
                let (ls, li, _us, _ui) = match approx {
                    GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                    GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
                };
                let (min_v, _max_v) = gelu_bound_interval(l, u, approx);
                let clo = concretized_lower(ls, li, l, u);
                // The lower bound must be within a small margin of the true minimum,
                // i.e. it must NOT plunge to the old ≈ −7 tangent value.
                assert!(
                    clo >= min_v - 0.05,
                    "{approx:?} [{l},{u}]: lower concretizes to {clo}, far below true min {min_v}"
                );
                // And it must stay sound (at or below the true minimum).
                assert!(
                    clo <= min_v + 1e-3,
                    "{approx:?} [{l},{u}]: lower {clo} exceeds true min {min_v} (unsound)"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(2000) })]

        /// Soundness of the floor-clamped lower line over ANY finite interval:
        /// `lower_slope * x + lower_intercept <= GELU(x)` for all x in [l, u],
        /// for both approximations. Random input → true op output within bound.
        #[test]
        fn proptest_floor_clamp_lower_envelope_sound(
            a in -12.0f32..12.0,
            b in -12.0f32..12.0,
        ) {
            let (l, u) = (a.min(b), a.max(b));
            for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
                let (ls, li, us, ui) = match approx {
                    GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                    GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
                };
                let f = match approx {
                    GeluApproximation::Erf => gelu_erf,
                    GeluApproximation::Tanh => gelu_tanh,
                };
                // Dense sweep including endpoints and the global-minimum critical point.
                let n = 400;
                for i in 0..=n {
                    let t = i as f32 / n as f32;
                    let x = l + (u - l) * t;
                    let y = f(x);
                    let lower = ls * x + li;
                    let upper = us * x + ui;
                    // tol absorbs f32 sampling/representation noise only.
                    let tol = 1e-3 * (1.0 + y.abs());
                    prop_assert!(
                        lower <= y + tol,
                        "{approx:?} [{l},{u}] @ x={x}: LOWER {lower} > GELU {y}"
                    );
                    prop_assert!(
                        upper >= y - tol,
                        "{approx:?} [{l},{u}] @ x={x}: UPPER {upper} < GELU {y}"
                    );
                }
            }
        }

        /// Tighten-or-equal: the clamped lower line's concretized minimum must be
        /// >= the constant-floor value `min_v` (i.e. the clamp never loosens, it only
        /// raises the lower bound up toward the true minimum).
        #[test]
        fn proptest_floor_clamp_tighten_or_equal(
            a in -12.0f32..12.0,
            b in -12.0f32..12.0,
        ) {
            let (l, u) = (a.min(b), a.max(b));
            prop_assume!((u - l) > 1e-6);
            for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
                let (ls, li, _us, _ui) = match approx {
                    GeluApproximation::Erf => gelu_sound_linear_relaxation(l, u),
                    GeluApproximation::Tanh => gelu_tanh_sound_linear_relaxation(l, u),
                };
                let (min_v, _max_v) = gelu_bound_interval(l, u, approx);
                let clo = concretized_lower(ls, li, l, u);
                // Concretized lower must be at least the floor minus the posthoc safety
                // margin (gelu_posthoc_adjust widens the intercept slightly downward).
                prop_assert!(
                    clo >= min_v - 0.05,
                    "{approx:?} [{l},{u}]: concretized lower {clo} below floor {min_v} \
                     by more than the posthoc margin"
                );
            }
        }
    }
}
