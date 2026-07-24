// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound linear relaxation for SiLU activation.
//!
//! Computes lower and upper linear bounds for SiLU on an interval [l, u]
//! using chord + tangent bounds based on convexity structure.
//!
//! Reference: alpha-beta-CROWN BoundGelu, gelu.py:320-383
//! Reference: designs/2026-02-08-silu-crown-relaxation.md

use super::math::{
    silu_chord, silu_critical_point, silu_derivative, silu_eval, silu_eval_f64,
    silu_inflection_points, silu_min_max, silu_second_derivative, silu_tangent, silu_tangent_raw,
};
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::activations::LinearRelaxation;
use tracing::warn;

/// Sound linear relaxation for SiLU on interval [l, u].
///
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept) where:
/// - SiLU(x) >= lower_slope * x + lower_intercept  for all x in [l, u]
/// - SiLU(x) <= upper_slope * x + upper_intercept  for all x in [l, u]
///
/// Uses chord + tangent bounds based on the convexity structure of SiLU,
/// following the GeLU pattern from alpha-beta-CROWN (BoundGelu in gelu.py).
///
/// SiLU convexity regions (SiLU''(x) sign):
/// - x < p₁ ≈ -2.40: concave → chord lower, tangent upper
/// - p₁ < x < p₂ ≈ +2.40: convex → tangent lower, chord upper
/// - x > p₂: concave → chord lower, tangent upper
///
/// Reference: designs/2026-02-08-silu-crown-relaxation.md
/// Source: alpha-beta-CROWN BoundGelu, gelu.py:320-383
pub fn silu_sound_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    // Handle NaN: no valid relaxation possible.
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::nan_fallback();
    }

    // Handle infinite bounds: SiLU(x) → 0 as x → -∞, SiLU(x) → x as x → +∞.
    // No finite linear bound can contain SiLU over an infinite range on the
    // positive side. Return maximally loose bounds.
    // Reference: designs/2026-02-08-silu-crown-relaxation.md, "Fix for Bug 1"
    if u == f32::INFINITY {
        return LinearRelaxation::nan_fallback();
    }
    if l == f32::NEG_INFINITY {
        // SiLU(-∞) = 0, and SiLU has a finite minimum near x ≈ -1.28.
        // Best we can do without a finite lower bound: constant bounds at
        // the min/max of SiLU over (-∞, u].
        let fu = silu_eval(u);
        let f_crit = silu_eval(silu_critical_point());
        let min_val = nan_propagating_min(nan_propagating_min(f_crit, fu), 0.0);
        let max_val = nan_propagating_max(fu, 0.0);
        return LinearRelaxation::new(0.0, min_val, 0.0, max_val);
    }

    // Near-point interval. SOUNDNESS (false-proof fix): a single silu_eval(l) is NOT a sound
    // constant relaxation — silu is non-monotone (interior minimum at silu_critical_point()),
    // so it misses silu(u) and, when the narrow interval straddles the critical point, the
    // interior minimum too → a certified bound past the true value. Cover the endpoint range
    // plus the interior minimum (mirroring the boundary case above).
    if (u - l).abs() < 1e-8 {
        let y_l = silu_eval(l);
        let y_u = silu_eval(u);
        let mut lo = nan_propagating_min(y_l, y_u);
        let hi = nan_propagating_max(y_l, y_u);
        let x_crit = silu_critical_point();
        if l <= x_crit && x_crit <= u {
            lo = nan_propagating_min(lo, silu_eval(x_crit));
        }
        return LinearRelaxation::new(0.0, lo, 0.0, hi);
    }

    let (p1, p2) = silu_inflection_points();

    // Classify the interval by convexity region overlap.
    if l >= p1 && u <= p2 {
        return silu_relaxation_convex(l, u);
    }
    if u <= p1 {
        return silu_relaxation_concave(l, u);
    }
    if l >= p2 {
        return silu_relaxation_concave(l, u);
    }

    // Interval crosses at least one inflection point.
    silu_relaxation_crossing(l, u, p1, p2)
}

/// Relaxation for an interval entirely in the convex region [p1, p2].
/// Tangent lower, chord upper. With directed rounding (#2434).
fn silu_relaxation_convex(l: f32, u: f32) -> LinearRelaxation {
    let max_abs_x = l.abs().max(u.abs());
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let (ls, ls_lower_i, _ls_upper_i) = silu_tangent((l + u) / 2.0, max_abs_x);
    let (us, _us_lower_i, us_upper_i) = silu_chord(l, u);
    // Verify soundness: tangent (with sound lower intercept) must be below SiLU.
    // Strict comparison in f64: directed rounding (#3146) absorbs f64→f32 slope
    // truncation error, but f32 evaluation of the check itself can mask sub-ULP
    // violations. Using f64 catches these. Ref: alpha-beta-CROWN uses exact <=.
    let lower_at_l = ls as f64 * l as f64 + ls_lower_i as f64;
    let lower_at_u = ls as f64 * u as f64 + ls_lower_i as f64;
    if lower_at_l <= silu_eval_f64(l as f64) && lower_at_u <= silu_eval_f64(u as f64) {
        // Tangent as lower bound (lower_intercept), chord as upper bound (upper_intercept).
        return LinearRelaxation::new(ls, ls_lower_i, us, us_upper_i);
    }
    // Fallback to constant bounds if numerical issues arise (#2981 Slice 4).
    // Sound (over-approximation) but looser than tangent bounds.
    warn!(
        "SiLU convex tangent verification failed on [{l}, {u}], \
         falling back to constant bounds"
    );
    let (min_val, max_val) = silu_min_max(l, u);
    LinearRelaxation::new(0.0, min_val, 0.0, max_val)
}

/// Relaxation for an interval entirely in a concave region (left or right).
/// Chord lower, tangent upper (Jensen's inequality). With directed rounding (#2434).
fn silu_relaxation_concave(l: f32, u: f32) -> LinearRelaxation {
    let max_abs_x = l.abs().max(u.abs());
    let (ls, ls_lower_i, _ls_upper_i) = silu_chord(l, u);
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let (us, _us_lower_i, us_upper_i) = silu_tangent((l + u) / 2.0, max_abs_x);
    // Verify soundness: tangent (with sound upper intercept) must be above SiLU.
    // Strict comparison in f64 (#2434): catches sub-ULP violations masked by f32.
    // Ref: alpha-beta-CROWN uses exact >= in check_upper().
    let upper_at_l = us as f64 * l as f64 + us_upper_i as f64;
    let upper_at_u = us as f64 * u as f64 + us_upper_i as f64;
    if upper_at_l >= silu_eval_f64(l as f64) && upper_at_u >= silu_eval_f64(u as f64) {
        // Chord as lower bound (lower_intercept), tangent as upper bound (upper_intercept).
        return LinearRelaxation::new(ls, ls_lower_i, us, us_upper_i);
    }
    // Fallback: sound but looser (#2981 Slice 4).
    warn!(
        "SiLU concave tangent verification failed on [{l}, {u}], \
         falling back to constant bounds"
    );
    let (min_val, max_val) = silu_min_max(l, u);
    LinearRelaxation::new(0.0, min_val, 0.0, max_val)
}

/// Handle intervals that cross one or both inflection points.
///
/// SiLU convexity: convex on (p1, p2), concave outside. Crossing intervals
/// are sub-classified by which tails they reach:
///
/// | Sub-case     | l         | u         | Lower bound          | Upper bound         |
/// |--------------|-----------|-----------|----------------------|---------------------|
/// | cross_left   | l < p1    | u ≤ p2    | tangent (convex BS)  | chord               |
/// | cross_right  | l ≥ p1    | u > p2    | chord / tangent (BS) | tangent (concave BS)|
/// | cross_both   | l < p1    | u > p2    | tangent (convex BS)  | chord / tangent     |
///
/// For cross_left: the chord connects a point in the left concave region to
/// one in the convex region. SiLU dips below the chord in the concave region,
/// so the chord is a valid upper bound. The lower bound uses a tangent from
/// the convex region found via binary search.
///
/// For cross_right: the chord connects a point in the convex region to one in
/// the right concave region. In the right concave tail, SiLU curves *above*
/// the chord (concave + approaching x), so the chord is NOT a valid upper
/// bound. Instead, use a tangent from the right concave region as upper bound
/// (concavity guarantees tangent ≥ SiLU locally, binary search ensures global
/// validity). The chord may work as a lower bound; otherwise use convex tangent.
///
/// For cross_both: chord validity depends on whether SiLU exceeds the chord in
/// the right tail. If the chord slope is less than SiLU'(u), the chord is
/// valid as upper bound (SiLU is pulling away below the chord on both tails).
/// Otherwise, find an upper tangent via binary search.
///
/// Reference: alpha-beta-CROWN BoundGelu gelu.py:320-383
fn silu_relaxation_crossing(l: f32, u: f32, p1: f32, p2: f32) -> LinearRelaxation {
    let (min_val, max_val) = silu_min_max(l, u);

    // Use f64-intermediate chord helper with directed rounding (#2434).
    let (chord_slope, _chord_lower_i, chord_upper_i) = silu_chord(l, u);

    let crosses_left = l < p1;
    let crosses_right = u > p2;

    // Chord is used as upper bound in crossing — use upper_intercept.
    let upper = crossing_upper_bound(l, u, chord_slope, chord_upper_i, crosses_right, p2, max_val);
    let lower = crossing_lower_bound(l, u, crosses_left, p1, p2, min_val);

    LinearRelaxation::new(lower.0, lower.1, upper.0, upper.1)
}

/// Upper bound for a crossing interval.
///
/// Try chord first (valid for cross_left/cross_both). If chord fails and
/// interval crosses right, find a tangent from the right concave region.
/// Ref: alpha-beta-CROWN gelu.py:220-224 (chord as upper bound for mask_both).
fn crossing_upper_bound(
    l: f32,
    u: f32,
    chord_slope: f32,
    chord_intercept: f32,
    crosses_right: bool,
    p2: f32,
    max_val: f32,
) -> (f32, f32) {
    if verify_upper_bound(l, u, chord_slope, chord_intercept) {
        (chord_slope, chord_intercept)
    } else if crosses_right {
        find_upper_tangent_binary(l, u, p2).unwrap_or_else(|| {
            warn!(
                "SiLU crossing upper tangent search failed on [{l}, {u}], \
                 falling back to constant upper bound"
            );
            (0.0, max_val)
        })
    } else {
        warn!(
            "SiLU crossing chord verification failed on [{l}, {u}] (non-right-cross), \
             falling back to constant upper bound"
        );
        (0.0, max_val)
    }
}

/// Lower bound for a crossing interval.
///
/// Use a tangent from the convex region [p1, p2] as lower bound. Binary search
/// ensures validity at endpoints in the concave tails.
fn crossing_lower_bound(
    l: f32,
    u: f32,
    crosses_left: bool,
    p1: f32,
    p2: f32,
    min_val: f32,
) -> (f32, f32) {
    let label = if crosses_left {
        "crosses_left"
    } else {
        "cross_right"
    };
    find_lower_tangent_binary(l, u, p1, p2).unwrap_or_else(|| {
        warn!(
            "SiLU crossing lower tangent search failed on [{l}, {u}] ({label}), \
             falling back to constant lower bound"
        );
        (0.0, min_val)
    })
}

/// Verify that a linear function is a valid upper bound for SiLU on [l, u]
/// by finding the point of maximum deviation using Newton's method on
/// SiLU'(x) = slope (where SiLU - line is maximized).
///
/// This is more reliable than sampling because it finds the exact extremum.
fn verify_upper_bound(l: f32, u: f32, slope: f32, intercept: f32) -> bool {
    // SiLU(x) - line(x) is maximized where SiLU'(x) = slope.
    // Use Newton's method on g(x) = SiLU'(x) - slope.
    // g'(x) = SiLU''(x).
    // There may be multiple roots; start from several points.
    // Bit-identical Newton starts: f32::midpoint rounds differently at overflow edges.
    #[allow(clippy::manual_midpoint)]
    let starts = [l, (l + u) / 2.0, u, l + (u - l) * 0.25, l + (u - l) * 0.75];
    for &x0 in &starts {
        let mut x = x0;
        for _ in 0..30 {
            let g = silu_derivative(x) - slope;
            let gp = silu_second_derivative(x);
            if gp.abs() < 1e-12 {
                break;
            }
            x -= g / gp;
            x = x.clamp(l, u);
        }
        // Strict f64 comparison (#2434): if SiLU exceeds line anywhere, line is
        // not a valid upper bound. f64 catches sub-ULP violations masked by f32.
        let deviation = silu_eval_f64(x as f64) - (slope as f64 * x as f64 + intercept as f64);
        if deviation > 0.0 {
            return false;
        }
    }
    // Also check the critical point and inflection points explicitly.
    // For crossing intervals, the deviation maximum can occur near an inflection
    // point where curvature changes sign. Consistent with find_upper_tangent_binary
    // which checks [l, u, critical, p1, p2]. (#2434 audit F1)
    let critical = silu_critical_point();
    let (p1, p2) = silu_inflection_points();
    for &x in &[critical, p1, p2] {
        if x > l && x < u {
            let deviation = silu_eval_f64(x as f64) - (slope as f64 * x as f64 + intercept as f64);
            if deviation > 0.0 {
                return false;
            }
        }
    }
    true
}

/// Find a tangent point in the convex region [p1, p2] whose tangent line is
/// a valid lower bound for SiLU on [l, u], using binary search.
///
/// In the convex region, tangent lines lie below SiLU (by convexity). For the
/// tangent to be valid over the full [l, u], it must also be below SiLU at
/// the endpoints in the concave tails.
///
/// Binary search finds the range of valid tangent points by checking the
/// constraint at each endpoint independently, then intersects the ranges.
///
/// Reference: alpha-beta-CROWN gelu.py:96-118 (d_lower_right precomputation)
fn find_lower_tangent_binary(l: f32, u: f32, p1: f32, p2: f32) -> Option<(f32, f32)> {
    let search_l = nan_propagating_max(l, p1);
    let search_u = nan_propagating_min(u, p2);
    if search_l >= search_u {
        return None;
    }

    // Find the rightmost d where tangent is valid at l.
    // Kani 0.66.0 does not support `?` on `Option` (E0277: `std::ops::Try` not implemented).
    // Use explicit match instead. See #2305.
    #[allow(clippy::question_mark)]
    let d_max_for_l = if l >= p1 {
        search_u // l in convex region — no constraint
    } else {
        match binary_search_tangent_below(search_l, search_u, l, true) {
            Some(d) => d,
            None => return None,
        }
    };

    // Find the leftmost d where tangent is valid at u.
    let d_min_for_u = if u <= p2 {
        search_l // u in convex region — no constraint
    } else {
        match binary_search_tangent_below(search_l, search_u, u, false) {
            Some(d) => d,
            None => {
                // Even the rightmost tangent violates at u. Try d_max_for_l directly.
                // Use raw tangent for verification check.
                let (ts, ti) = silu_tangent_raw(d_max_for_l);
                // Strict f64 check (#2434): raw tangent must not exceed SiLU at u.
                if ts as f64 * u as f64 + ti as f64 <= silu_eval_f64(u as f64) {
                    // Return directed-rounded lower intercept for final output.
                    let max_abs_x = l.abs().max(u.abs());
                    let (_ts, lower_i, _upper_i) = silu_tangent(d_max_for_l, max_abs_x);
                    return Some((ts, lower_i));
                } else {
                    return None;
                }
            }
        }
    };

    if d_min_for_u > d_max_for_l + 1e-7 {
        return None;
    }

    // Pick the midpoint of the valid range for tightest average bound.
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let d_opt = (d_min_for_u + d_max_for_l) / 2.0;
    // Use raw tangent for verification, directed-rounded for output.
    let (ts, ti) = silu_tangent_raw(d_opt);

    // Final safety: verify at endpoints and critical point in f64 (#2434).
    // The returned intercept will be directed-rounded via silu_tangent().
    let critical = silu_critical_point();
    for &x in &[l, u, critical] {
        if x >= l && x <= u && ts as f64 * x as f64 + ti as f64 > silu_eval_f64(x as f64) {
            return None;
        }
    }

    // Return directed-rounded lower intercept for sound lower bound.
    let max_abs_x = l.abs().max(u.abs());
    let (_ts, lower_i, _upper_i) = silu_tangent(d_opt, max_abs_x);
    Some((ts, lower_i))
}

/// Binary search for a tangent point d in [search_l, search_u] where the
/// tangent at d lies below SiLU at a given point x.
///
/// Uses `silu_tangent_raw` for candidate checking (tight values for search).
/// If `find_rightmost`, returns the rightmost valid d (lo after convergence).
/// Otherwise, returns the leftmost valid d (hi after convergence).
/// Returns None if no valid d exists.
fn binary_search_tangent_below(
    search_l: f32,
    search_u: f32,
    x: f32,
    find_rightmost: bool,
) -> Option<f32> {
    let tangent_below_at = |d: f32| -> bool {
        // Strict f64 check (#2434): tangent at d must not exceed SiLU at x.
        let (ts, ti) = silu_tangent_raw(d);
        ts as f64 * x as f64 + ti as f64 <= silu_eval_f64(x as f64)
    };

    let check_start = if find_rightmost { search_l } else { search_u };
    if !tangent_below_at(check_start) {
        return None;
    }

    let mut lo = search_l;
    let mut hi = search_u;
    for _ in 0..60 {
        // Bit-identical bisection anchor: f32::midpoint rounds differently at overflow edges.
        #[allow(clippy::manual_midpoint)]
        let mid = (lo + hi) / 2.0;
        if tangent_below_at(mid) {
            if find_rightmost {
                lo = mid;
            } else {
                hi = mid;
            }
        } else if find_rightmost {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo) < 1e-7 {
            break;
        }
    }
    Some(if find_rightmost { lo } else { hi })
}

/// Find a tangent point in the right concave region [p2, ...] whose tangent
/// line is a valid upper bound for SiLU on [l, u], using binary search.
///
/// Uses `silu_tangent_raw` for candidate checking, `silu_tangent` with
/// directed rounding for the final output (#2434).
///
/// Reference: alpha-beta-CROWN gelu.py:122-139 (d_upper_right precomputation)
fn find_upper_tangent_binary(l: f32, u: f32, p2: f32) -> Option<(f32, f32)> {
    let search_l = nan_propagating_max(p2, l);
    let search_u = u;
    if search_l >= search_u {
        return None;
    }

    // Check if tangent at d is above SiLU(x) — use raw tangent for search.
    // Strict f64 check (#2434): tangent must not fall below SiLU at x.
    let tangent_above_at = |d: f32, x: f32| -> bool {
        let (ts, ti) = silu_tangent_raw(d);
        ts as f64 * x as f64 + ti as f64 >= silu_eval_f64(x as f64)
    };

    // Find the leftmost d in [search_l, search_u] where tangent is above SiLU
    // at l (the most constraining point in the convex/left-concave region).
    if !tangent_above_at(search_u, l) {
        return None;
    }

    let mut lo = search_l;
    let mut hi = search_u;
    for _ in 0..60 {
        // Bit-identical bisection anchor: f32::midpoint rounds differently at overflow edges.
        #[allow(clippy::manual_midpoint)]
        let mid = (lo + hi) / 2.0;
        if tangent_above_at(mid, l) {
            hi = mid; // mid works, try further left for tighter bound
        } else {
            lo = mid; // mid doesn't work, go right
        }
        if (hi - lo) < 1e-7 {
            break;
        }
    }
    let d_opt = hi;

    // Use raw tangent for verification checks.
    let (ts, ti) = silu_tangent_raw(d_opt);

    // Verify at the critical point and inflection point.
    let critical = silu_critical_point();
    let (p1, _) = silu_inflection_points();
    // Strict f64 check (#2434): raw tangent must not fall below SiLU.
    // The returned intercept will be directed-rounded via silu_tangent().
    for &x in &[l, u, critical, p1, p2] {
        if x >= l && x <= u && (ts as f64 * x as f64 + ti as f64) < silu_eval_f64(x as f64) {
            return None;
        }
    }

    // Also verify at the point of maximum SiLU deviation from the tangent line.
    // Bit-identical Newton start: f32::midpoint rounds differently at overflow edges.
    #[allow(clippy::manual_midpoint)]
    let mut x = (l + p2) / 2.0;
    for _ in 0..30 {
        let g = silu_derivative(x) - ts;
        let gp = silu_second_derivative(x);
        if gp.abs() < 1e-12 {
            break;
        }
        x -= g / gp;
        x = x.clamp(l, u);
    }
    if (ts as f64 * x as f64 + ti as f64) < silu_eval_f64(x as f64) {
        return None;
    }

    // Return directed-rounded upper intercept for sound upper bound.
    let max_abs_x = l.abs().max(u.abs());
    let (_ts, _lower_i, upper_i) = silu_tangent(d_opt, max_abs_x);
    Some((ts, upper_i))
}
