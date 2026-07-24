// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GELU evaluation, derivatives, and critical/inflection point helpers.
//!
//! Provides both f32 (runtime) and f64 (table precomputation) versions of
//! GELU and its derivative for both Erf and Tanh approximations.

use std::sync::OnceLock;

use crate::rounding::{next_down_f32, next_up_f32};
use crate::types::GeluApproximation;
use ny_core::{nan_propagating_max, nan_propagating_min};

// =============================================================================
// f32 evaluation and derivatives (runtime)
// =============================================================================

pub(crate) fn gelu_erf(x: f32) -> f32 {
    // Guard against 0 * inf = NaN when x = ±inf.
    // GELU(x) = 0.5 * x * (1 + erf(x/√2)). At x = -inf: 0.5*(-inf)*(1+(-1)) = 0*inf = NaN.
    // Correct limits: GELU(-inf) = 0, GELU(+inf) = +inf.
    // Ref: SiLU guard pattern (silu.rs:100-105), fix for #1836.
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
    0.5 * x * (1.0 + libm::erff(x * inv_sqrt2))
}

pub(crate) fn gelu_tanh(x: f32) -> f32 {
    // Guard against 0 * inf = NaN when x = ±inf.
    // GELU_tanh(x) = 0.5 * x * (1 + tanh(...)). At x = -inf: 0.5*(-inf)*(1+(-1)) = 0*inf = NaN.
    // Correct limits: GELU(-inf) = 0, GELU(+inf) = +inf.
    // Ref: SiLU guard pattern (silu.rs:100-105), fix for #1836.
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let sqrt_2_over_pi = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

/// Evaluate GELU at a single point using the specified approximation.
pub fn gelu_eval(x: f32, approximation: GeluApproximation) -> f32 {
    match approximation {
        GeluApproximation::Erf => gelu_erf(x),
        GeluApproximation::Tanh => gelu_tanh(x),
    }
}

pub(crate) fn gelu_derivative(x: f32, approximation: GeluApproximation) -> f32 {
    // Guard: GELU'(-inf) = 0, GELU'(+inf) = 1.
    // At x = ±inf, the finite formulas include x*pdf terms that evaluate as inf*0 = NaN.
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { 1.0 };
    }
    match approximation {
        GeluApproximation::Erf => {
            let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
            let inv_sqrt_2pi: f32 = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
            let phi: f32 = 0.5 * (1.0 + libm::erff(x * inv_sqrt2));
            let pdf: f32 = inv_sqrt_2pi * (-0.5 * x * x).exp();
            phi + x * pdf
        }
        GeluApproximation::Tanh => {
            let k: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
            let t: f32 = k * (x + 0.044715 * x * x * x);
            let tanh_t: f32 = t.tanh();
            let sech2_t: f32 = 1.0 - tanh_t * tanh_t;
            let dt_dx: f32 = k * (1.0 + 3.0 * 0.044715 * x * x);
            0.5 * (1.0 + tanh_t) + 0.5 * x * sech2_t * dt_dx
        }
    }
}

fn gelu_tanh_second_derivative(x: f32) -> f32 {
    let h: f32 = 1e-3 * (1.0 + x.abs());
    let d_plus = gelu_derivative(x + h, GeluApproximation::Tanh);
    let d_minus = gelu_derivative(x - h, GeluApproximation::Tanh);
    (d_plus - d_minus) / (2.0 * h)
}

// =============================================================================
// f64 helpers for high-precision table precomputation.
// These are used only at startup during OnceLock initialization.
// Reference: α,β-CROWN uses f64 binary search with 100 iterations (~2^{-100}).
// =============================================================================

pub(crate) fn gelu_erf_f64(x: f64) -> f64 {
    // Guard against 0 * inf = NaN when x = ±inf.
    // GELU(x) = 0.5 * x * (1 + erf(x/√2)). At x = -inf: 0.5*(-inf)*(1+(-1)) = 0*inf = NaN.
    // Correct limits: GELU(-inf) = 0, GELU(+inf) = +inf.
    // Ref: f32 guard at eval.rs:18-31, fix for #1836, fix for #2504.
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let inv_sqrt2: f64 = 1.0 / 2.0_f64.sqrt();
    0.5 * x * (1.0 + libm::erf(x * inv_sqrt2))
}

pub(crate) fn gelu_tanh_f64(x: f64) -> f64 {
    // Guard against 0 * inf = NaN when x = ±inf.
    // Ref: f32 guard at eval.rs:33-46, fix for #2504.
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
}

/// Evaluate GELU in f64 using the specified approximation. (#3245)
fn gelu_eval_f64(x: f64, approximation: GeluApproximation) -> f64 {
    match approximation {
        GeluApproximation::Erf => gelu_erf_f64(x),
        GeluApproximation::Tanh => gelu_tanh_f64(x),
    }
}

pub(crate) fn gelu_derivative_erf_f64(x: f64) -> f64 {
    // Guard: GELU'(-inf) = 0, GELU'(+inf) = 1.
    // At x = -inf: phi→0, x*pdf→0 (Gaussian decays faster than x grows).
    // At x = +inf: phi→1, x*pdf→0.
    // Ref: fix for #2504.
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { 1.0 };
    }
    let inv_sqrt2: f64 = 1.0 / 2.0_f64.sqrt();
    let inv_sqrt_2pi: f64 = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let phi: f64 = 0.5 * (1.0 + libm::erf(x * inv_sqrt2));
    let pdf: f64 = inv_sqrt_2pi * (-0.5 * x * x).exp();
    phi + x * pdf
}

pub(crate) fn gelu_derivative_tanh_f64(x: f64) -> f64 {
    // Guard: GELU'(-inf) = 0, GELU'(+inf) = 1.
    // Ref: fix for #2504.
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { 1.0 };
    }
    let k: f64 = (2.0_f64 / std::f64::consts::PI).sqrt();
    let t: f64 = k * (x + 0.044715 * x * x * x);
    let tanh_t: f64 = t.tanh();
    let sech2_t: f64 = 1.0 - tanh_t * tanh_t;
    let dt_dx: f64 = k * (1.0 + 3.0 * 0.044715 * x * x);
    0.5 * (1.0 + tanh_t) + 0.5 * x * sech2_t * dt_dx
}

/// Check if tangent at f64 point `d` provides a valid lower bound at `upper` for Erf GELU.
#[inline]
pub(crate) fn check_lower_gelu_f64(upper: f64, d: f64) -> bool {
    let k = gelu_derivative_erf_f64(d);
    let tangent_at_upper = k * (upper - d) + gelu_erf_f64(d);
    tangent_at_upper <= gelu_erf_f64(upper)
}

/// Check if tangent at f64 point `d` provides a valid upper bound at `lower` for Erf GELU.
#[inline]
pub(crate) fn check_upper_gelu_f64(lower: f64, d: f64) -> bool {
    let k = gelu_derivative_erf_f64(d);
    let tangent_at_lower = k * (lower - d) + gelu_erf_f64(d);
    tangent_at_lower >= gelu_erf_f64(lower)
}

/// Check if tangent at f64 point `d` provides a valid lower bound at `upper` for tanh GELU.
#[inline]
pub(crate) fn check_lower_gelu_tanh_f64(upper: f64, d: f64) -> bool {
    let k = gelu_derivative_tanh_f64(d);
    let tangent_at_upper = k * (upper - d) + gelu_tanh_f64(d);
    tangent_at_upper <= gelu_tanh_f64(upper)
}

/// Check if tangent at f64 point `d` provides a valid upper bound at `lower` for tanh GELU.
#[inline]
pub(crate) fn check_upper_gelu_tanh_f64(lower: f64, d: f64) -> bool {
    let k = gelu_derivative_tanh_f64(d);
    let tangent_at_lower = k * (lower - d) + gelu_tanh_f64(d);
    tangent_at_lower >= gelu_tanh_f64(lower)
}

// =============================================================================
// Cached critical and inflection points
// =============================================================================

/// Fallback inflection split point for tanh-approx GELU.
const GELU_TANH_INFLECTION_FALLBACK: f32 = 1.418504;

/// Returns the inflection point of the tanh-approximation GELU (cached via `OnceLock`).
///
/// The inflection point is where the second derivative crosses zero,
/// found by bisection. Used to select tight linear relaxation strategies.
pub fn gelu_tanh_inflection_point() -> f32 {
    static INFLECTION: OnceLock<f32> = OnceLock::new();
    *INFLECTION.get_or_init(|| {
        let mut lo: f32 = 0.5;
        let mut hi: f32 = 2.5;
        let mut flo = gelu_tanh_second_derivative(lo);
        let mut fhi = gelu_tanh_second_derivative(hi);

        if !flo.is_finite() || !fhi.is_finite() {
            return GELU_TANH_INFLECTION_FALLBACK;
        }

        for _ in 0..8 {
            if flo.signum() != fhi.signum() {
                break;
            }
            lo = (lo * 0.5).max(1e-3);
            hi = (hi * 1.5).min(20.0);
            flo = gelu_tanh_second_derivative(lo);
            fhi = gelu_tanh_second_derivative(hi);
            if !flo.is_finite() || !fhi.is_finite() {
                return GELU_TANH_INFLECTION_FALLBACK;
            }
        }

        if flo.signum() == fhi.signum() {
            return GELU_TANH_INFLECTION_FALLBACK;
        }

        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let fmid = gelu_tanh_second_derivative(mid);
            if !fmid.is_finite() {
                return GELU_TANH_INFLECTION_FALLBACK;
            }
            if fmid.signum() == flo.signum() {
                lo = mid;
                flo = fmid;
            } else {
                hi = mid;
            }
        }

        0.5 * (lo + hi)
    })
}

pub(crate) fn gelu_critical_point(approximation: GeluApproximation) -> f32 {
    static GELU_CRITICAL_ERF: OnceLock<f32> = OnceLock::new();
    static GELU_CRITICAL_TANH: OnceLock<f32> = OnceLock::new();

    let slot = match approximation {
        GeluApproximation::Erf => &GELU_CRITICAL_ERF,
        GeluApproximation::Tanh => &GELU_CRITICAL_TANH,
    };

    *slot.get_or_init(|| {
        // GELU has a single global minimum for x < 0, near x ≈ -0.75.
        // Use bisection on derivative in a bracket known to straddle the root.
        let mut lo: f32 = -2.0;
        let mut hi: f32 = 0.0;
        let dlo: f32 = gelu_derivative(lo, approximation);
        let dhi: f32 = gelu_derivative(hi, approximation);

        if !(dlo < 0.0 && dhi > 0.0) {
            // Fallback: widen bracket.
            lo = -10.0;
            hi = 1.0;
            let dlo2: f32 = gelu_derivative(lo, approximation);
            let dhi2: f32 = gelu_derivative(hi, approximation);
            if !(dlo2 < 0.0 && dhi2 > 0.0) {
                return -0.75;
            }
        }

        // If we still can't bracket, fall back to a reasonable constant; callers still
        // evaluate endpoints so this only affects min tightening.
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let dmid = gelu_derivative(mid, approximation);
            if dmid > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        0.5 * (lo + hi)
    })
}

/// Directed rounding: compute in f64, apply next_down/next_up to EACH
/// intermediate evaluation BEFORE min/max selection. Plain `as f32` rounds to
/// nearest — a candidate min could round UP, causing the final min to miss the
/// true minimum. next_down/next_up on the final result alone cannot recover if
/// the wrong candidate was selected. (#3245, #3336)
///
/// Ref: SiLU (#3146) and f64→f32 cast (#3132) directed rounding fixes.
pub(crate) fn gelu_bound_interval(l: f32, u: f32, approximation: GeluApproximation) -> (f32, f32) {
    let gl = gelu_eval_f64(l as f64, approximation) as f32;
    let gu = gelu_eval_f64(u as f64, approximation) as f32;

    // Apply directed rounding to each intermediate before min/max selection.
    let mut min_v = nan_propagating_min(next_down_f32(gl), next_down_f32(gu));
    let mut max_v = nan_propagating_max(next_up_f32(gl), next_up_f32(gu));

    let critical_point = gelu_critical_point(approximation);
    if l <= critical_point && critical_point <= u {
        let gc = gelu_eval_f64(critical_point as f64, approximation) as f32;
        min_v = nan_propagating_min(min_v, next_down_f32(gc));
        max_v = nan_propagating_max(max_v, next_up_f32(gc));
    }

    (min_v, max_v)
}

/// Handle infinite/NaN bounds for GELU linear relaxation.
///
/// Returns `Some((ls, li, us, ui))` if bounds are infinite/NaN (handled),
/// or `None` if bounds are finite (caller should proceed with normal computation).
pub(crate) fn gelu_infinite_bounds_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
) -> Option<(f32, f32, f32, f32)> {
    // Finite bounds: caller handles normally
    if l.is_finite() && u.is_finite() {
        return None;
    }

    // NaN: maximally loose bounds, always sound.
    if l.is_nan() || u.is_nan() {
        return Some((0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY));
    }

    // u = +∞: GELU(x) → x as x → +∞, so no finite linear bound can contain
    // GELU over [l, +∞). Return maximally loose.
    if u == f32::INFINITY {
        return Some((0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY));
    }

    // l = -∞, u finite (or u = -∞ degenerate): GELU(-∞) = 0, and GELU has a finite minimum.
    // Use constant (slope=0) bounds based on the function's range over (-∞, u].
    // Guard: if l is not -inf (e.g., l=+inf from upstream corruption), return maximally
    // loose bounds instead of panicking. Consistent with NaN fallback above. (#2739)
    if l != f32::NEG_INFINITY {
        return Some((0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY));
    }

    let fu = gelu_eval(u, approximation);
    let f_crit = gelu_eval(gelu_critical_point(approximation), approximation);

    // Lower bound: min of function value at critical point, endpoint, and 0 (GELU(-∞))
    let min_val = nan_propagating_min(nan_propagating_min(f_crit, fu), 0.0);
    // Upper bound: max of function value at endpoint and 0 (GELU(-∞))
    let max_val = nan_propagating_max(fu, 0.0);

    Some((0.0, min_val, 0.0, max_val))
}
