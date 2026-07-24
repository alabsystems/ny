// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sound linear relaxation for SiLU activation.
//!
//! Reference: alpha-beta-CROWN BoundGelu, gelu.py:320-383
//! Reference: designs/2026-02-08-silu-crown-relaxation.md

use super::math::{
    silu_chord, silu_critical_point, silu_derivative, silu_eval, silu_eval_f64,
    silu_inflection_points, silu_min_max, silu_second_derivative, silu_tangent, silu_tangent_raw,
};
use crate::types::LinearRelaxation;
use ny_core::{nan_propagating_max, nan_propagating_min};

/// Sound linear relaxation for SiLU on interval [l, u].
pub fn silu_sound_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::nan_fallback();
    }
    if u == f32::INFINITY {
        return LinearRelaxation::nan_fallback();
    }
    if l == f32::NEG_INFINITY {
        let fu = silu_eval(u);
        let f_crit = silu_eval(silu_critical_point());
        let min_val = nan_propagating_min(nan_propagating_min(f_crit, fu), 0.0);
        let max_val = nan_propagating_max(fu, 0.0);
        return LinearRelaxation::new(0.0, min_val, 0.0, max_val);
    }

    if (u - l).abs() < 1e-8 {
        // Near-point interval: constant bounds from BOTH endpoints (plus the
        // interior minimum when the critical point falls inside). A constant
        // band at silu(l) alone is unsound by up to an ulp when u != l.
        // Matches production (`ny_propagate::layers::activations::silu`).
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

    if l >= p1 && u <= p2 {
        return silu_relaxation_convex(l, u);
    }
    if u <= p1 {
        return silu_relaxation_concave(l, u);
    }
    if l >= p2 {
        return silu_relaxation_concave(l, u);
    }

    silu_relaxation_crossing(l, u, p1, p2)
}

fn silu_relaxation_convex(l: f32, u: f32) -> LinearRelaxation {
    let max_abs_x = l.abs().max(u.abs());
    let (ls, ls_lower_i, _ls_upper_i) = silu_tangent((l + u) / 2.0, max_abs_x);
    let (us, _us_lower_i, us_upper_i) = silu_chord(l, u);
    let lower_at_l = ls as f64 * l as f64 + ls_lower_i as f64;
    let lower_at_u = ls as f64 * u as f64 + ls_lower_i as f64;
    if lower_at_l <= silu_eval_f64(l as f64) && lower_at_u <= silu_eval_f64(u as f64) {
        return LinearRelaxation::new(ls, ls_lower_i, us, us_upper_i);
    }
    let (min_val, max_val) = silu_min_max(l, u);
    LinearRelaxation::new(0.0, min_val, 0.0, max_val)
}

fn silu_relaxation_concave(l: f32, u: f32) -> LinearRelaxation {
    let max_abs_x = l.abs().max(u.abs());
    let (ls, ls_lower_i, _ls_upper_i) = silu_chord(l, u);
    let (us, _us_lower_i, us_upper_i) = silu_tangent((l + u) / 2.0, max_abs_x);
    let upper_at_l = us as f64 * l as f64 + us_upper_i as f64;
    let upper_at_u = us as f64 * u as f64 + us_upper_i as f64;
    if upper_at_l >= silu_eval_f64(l as f64) && upper_at_u >= silu_eval_f64(u as f64) {
        return LinearRelaxation::new(ls, ls_lower_i, us, us_upper_i);
    }
    let (min_val, max_val) = silu_min_max(l, u);
    LinearRelaxation::new(0.0, min_val, 0.0, max_val)
}

fn silu_relaxation_crossing(l: f32, u: f32, p1: f32, p2: f32) -> LinearRelaxation {
    let (min_val, max_val) = silu_min_max(l, u);
    let (chord_slope, _chord_lower_i, chord_upper_i) = silu_chord(l, u);

    let crosses_right = u > p2;

    let upper = crossing_upper_bound(l, u, chord_slope, chord_upper_i, crosses_right, p2, max_val);
    let lower = crossing_lower_bound(l, u, l < p1, p1, p2, min_val);

    LinearRelaxation::new(lower.0, lower.1, upper.0, upper.1)
}

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
        find_upper_tangent_binary(l, u, p2).unwrap_or((0.0, max_val))
    } else {
        (0.0, max_val)
    }
}

fn crossing_lower_bound(
    l: f32,
    u: f32,
    _crosses_left: bool,
    p1: f32,
    p2: f32,
    min_val: f32,
) -> (f32, f32) {
    find_lower_tangent_binary(l, u, p1, p2).unwrap_or((0.0, min_val))
}

fn verify_upper_bound(l: f32, u: f32, slope: f32, intercept: f32) -> bool {
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
        let deviation = silu_eval_f64(x as f64) - (slope as f64 * x as f64 + intercept as f64);
        if deviation > 0.0 {
            return false;
        }
    }
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

fn find_lower_tangent_binary(l: f32, u: f32, p1: f32, p2: f32) -> Option<(f32, f32)> {
    let search_l = nan_propagating_max(l, p1);
    let search_u = nan_propagating_min(u, p2);
    if search_l >= search_u {
        return None;
    }

    #[allow(clippy::question_mark)]
    let d_max_for_l = if l >= p1 {
        search_u
    } else {
        match binary_search_tangent_below(search_l, search_u, l, true) {
            Some(d) => d,
            None => return None,
        }
    };

    let d_min_for_u = if u <= p2 {
        search_l
    } else {
        match binary_search_tangent_below(search_l, search_u, u, false) {
            Some(d) => d,
            None => {
                let (ts, ti) = silu_tangent_raw(d_max_for_l);
                if ts as f64 * u as f64 + ti as f64 <= silu_eval_f64(u as f64) {
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

    let d_opt = (d_min_for_u + d_max_for_l) / 2.0;
    let (ts, ti) = silu_tangent_raw(d_opt);

    let critical = silu_critical_point();
    for &x in &[l, u, critical] {
        if x >= l && x <= u && ts as f64 * x as f64 + ti as f64 > silu_eval_f64(x as f64) {
            return None;
        }
    }

    let max_abs_x = l.abs().max(u.abs());
    let (_ts, lower_i, _upper_i) = silu_tangent(d_opt, max_abs_x);
    Some((ts, lower_i))
}

fn binary_search_tangent_below(
    search_l: f32,
    search_u: f32,
    x: f32,
    find_rightmost: bool,
) -> Option<f32> {
    let tangent_below_at = |d: f32| -> bool {
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

fn find_upper_tangent_binary(l: f32, u: f32, p2: f32) -> Option<(f32, f32)> {
    let search_l = nan_propagating_max(p2, l);
    let search_u = u;
    if search_l >= search_u {
        return None;
    }

    let tangent_above_at = |d: f32, x: f32| -> bool {
        let (ts, ti) = silu_tangent_raw(d);
        ts as f64 * x as f64 + ti as f64 >= silu_eval_f64(x as f64)
    };

    if !tangent_above_at(search_u, l) {
        return None;
    }

    let mut lo = search_l;
    let mut hi = search_u;
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        if tangent_above_at(mid, l) {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo) < 1e-7 {
            break;
        }
    }
    let d_opt = hi;

    let (ts, ti) = silu_tangent_raw(d_opt);

    let critical = silu_critical_point();
    let (p1, _) = silu_inflection_points();
    for &x in &[l, u, critical, p1, p2] {
        if x >= l && x <= u && (ts as f64 * x as f64 + ti as f64) < silu_eval_f64(x as f64) {
            return None;
        }
    }

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

    let max_abs_x = l.abs().max(u.abs());
    let (_ts, _lower_i, upper_i) = silu_tangent(d_opt, max_abs_x);
    Some((ts, upper_i))
}
