// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GELU evaluation, derivatives, and critical/inflection point helpers.
//!
//! Provides both f32 (runtime) and f64 (table precomputation) versions of
//! GELU and its derivative for both Erf and Tanh approximations.

use std::sync::OnceLock;

use ny_tensor::{next_down_f32, next_up_f32};

use super::GeluApproximation;
use crate::bounds::{nan_propagating_max, nan_propagating_min};

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
            // Φ(x) = (1 + erf)/2; midpoint is bit-identical here (erf ∈ [-1, 1], no
            // overflow/subnormal edge).
            let phi: f32 = f32::midpoint(1.0, libm::erff(x * inv_sqrt2));
            let pdf: f32 = inv_sqrt_2pi * (-0.5 * x * x).exp();
            phi + x * pdf
        }
        GeluApproximation::Tanh => {
            let k: f32 = (2.0_f32 / std::f32::consts::PI).sqrt();
            let t: f32 = k * (x + 0.044715 * x * x * x);
            let tanh_t: f32 = t.tanh();
            let sech2_t: f32 = 1.0 - tanh_t * tanh_t;
            let dt_dx: f32 = k * (1.0 + 3.0 * 0.044715 * x * x);
            // (1 + tanh)/2 as midpoint is bit-identical (tanh ∈ [-1, 1]).
            f32::midpoint(1.0, tanh_t) + 0.5 * x * sech2_t * dt_dx
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
    // Φ(x) = (1 + erf)/2; midpoint is bit-identical here (erf ∈ [-1, 1], no
    // overflow edge, and f64::midpoint's fast path is `(a + b) * 0.5`).
    let phi: f64 = f64::midpoint(1.0, libm::erf(x * inv_sqrt2));
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
    // (1 + tanh)/2 as midpoint is bit-identical (tanh ∈ [-1, 1]).
    f64::midpoint(1.0, tanh_t) + 0.5 * x * sech2_t * dt_dx
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
            // Bit-identical: the bracket stays inside [1e-3, 20], far from any
            // overflow/subnormal edge.
            let mid = f32::midpoint(lo, hi);
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

        f32::midpoint(lo, hi)
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
            // Bit-identical: the bracket stays inside [-10, 1], far from any
            // overflow/subnormal edge.
            let mid = f32::midpoint(lo, hi);
            let dmid = gelu_derivative(mid, approximation);
            if dmid > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        f32::midpoint(lo, hi)
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
///
/// # Mathematics
///
/// GELU(x) = x · Φ(x) where Φ is the standard normal CDF.
/// - GELU(-∞) = 0 (since Φ(-∞) = 0)
/// - GELU(+∞) = +∞ (since Φ(+∞) = 1)
/// - GELU has a global minimum ≈ -0.170 at x ≈ -0.752
/// - GELU(x) ≤ x for all x (since Φ(x) ≤ 1)
///
/// The previous identity relaxation (1, 0, 1, 0) was **unsound**: the lower
/// bound GELU(x) ≥ x fails for 0 < x where Φ(x) < 1, e.g. GELU(0.3) ≈ 0.185 < 0.3.
///
/// Fix follows SiLU pattern (silu.rs:241-263, fix for #1673):
/// - NaN or u = +∞: maximally loose (no finite linear bound contains GELU over [l, +∞))
/// - l = -∞, u finite: constant bounds using function range over (-∞, u]
///
/// Reference: SiLU infinite bounds fix, silu.rs:241-263
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // gelu_derivative
    // =========================================================================

    /// GELU'(0) should be exactly 0.5 (both Erf and Tanh).
    /// GELU(x) = 0.5*x*(1+erf(x/√2)), so GELU'(0) = Φ(0) + 0·pdf(0) = 0.5.
    #[test]
    fn test_gelu_derivative_at_zero() {
        let d_erf = gelu_derivative(0.0, GeluApproximation::Erf);
        assert!(
            (d_erf - 0.5).abs() < 1e-5,
            "GELU'_erf(0) should be ~0.5, got {d_erf}"
        );
        let d_tanh = gelu_derivative(0.0, GeluApproximation::Tanh);
        assert!(
            (d_tanh - 0.5).abs() < 1e-4,
            "GELU'_tanh(0) should be ~0.5, got {d_tanh}"
        );
    }

    /// GELU'(x) → 1 as x → +∞ (since Φ(x) → 1 and x·pdf(x) → 0).
    #[test]
    fn test_gelu_derivative_large_positive() {
        let d = gelu_derivative(10.0, GeluApproximation::Erf);
        assert!(
            (d - 1.0).abs() < 1e-4,
            "GELU'_erf(10) should be ~1.0, got {d}"
        );
        let d = gelu_derivative(10.0, GeluApproximation::Tanh);
        assert!(
            (d - 1.0).abs() < 1e-3,
            "GELU'_tanh(10) should be ~1.0, got {d}"
        );
    }

    /// GELU'(x) → 0 as x → -∞ (since Φ(x) → 0 and x·pdf(x) → 0).
    #[test]
    fn test_gelu_derivative_large_negative() {
        let d = gelu_derivative(-10.0, GeluApproximation::Erf);
        assert!(d.abs() < 1e-4, "GELU'_erf(-10) should be ~0.0, got {d}");
        let d = gelu_derivative(-10.0, GeluApproximation::Tanh);
        assert!(d.abs() < 1e-3, "GELU'_tanh(-10) should be ~0.0, got {d}");
    }

    /// f32 derivative guards: GELU'(-inf)=0, GELU'(+inf)=1, GELU'(NaN)=NaN.
    #[test]
    fn test_gelu_derivative_f32_infinity_guards() {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            assert_eq!(gelu_derivative(f32::NEG_INFINITY, approx), 0.0);
            assert_eq!(gelu_derivative(f32::INFINITY, approx), 1.0);
            assert!(gelu_derivative(f32::NAN, approx).is_nan());
        }
    }

    /// Verify derivative against finite differences: GELU'(x) ≈ (GELU(x+h) - GELU(x-h)) / 2h.
    #[test]
    fn test_gelu_derivative_finite_differences() {
        let h = 1e-4;
        for &x in &[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
                let analytical = gelu_derivative(x, approx);
                let numerical = (gelu_eval(x + h, approx) - gelu_eval(x - h, approx)) / (2.0 * h);
                assert!(
                    (analytical - numerical).abs() < 1e-2,
                    "GELU'({approx:?}, {x}): analytical={analytical}, numerical={numerical}"
                );
            }
        }
    }

    // =========================================================================
    // gelu_critical_point
    // =========================================================================

    /// The critical point (GELU minimum) should be near x ≈ -0.75 for both approximations.
    #[test]
    fn test_gelu_critical_point_location() {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let cp = gelu_critical_point(approx);
            assert!(
                cp < 0.0 && cp > -1.5,
                "Critical point for {approx:?} should be in (-1.5, 0), got {cp}"
            );
            // Derivative should be near zero at critical point
            let d = gelu_derivative(cp, approx);
            assert!(
                d.abs() < 1e-3,
                "GELU'({approx:?}, {cp}) should be ~0, got {d}"
            );
        }
    }

    /// The GELU value at the critical point should be the global minimum ≈ -0.17.
    #[test]
    fn test_gelu_critical_point_value() {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let cp = gelu_critical_point(approx);
            let val = gelu_eval(cp, approx);
            assert!(
                val < 0.0 && val > -0.5,
                "GELU({approx:?}) at critical point should be in (-0.5, 0), got {val}"
            );
        }
    }

    // =========================================================================
    // gelu_tanh_inflection_point
    // =========================================================================

    /// The inflection point should be positive, near ~1.4.
    #[test]
    fn test_gelu_tanh_inflection_point_value() {
        let ip = gelu_tanh_inflection_point();
        assert!(
            ip > 1.0 && ip < 2.0,
            "Tanh GELU inflection point should be in (1, 2), got {ip}"
        );
    }

    /// Calling twice should return the same value (OnceLock caching).
    #[test]
    fn test_gelu_tanh_inflection_point_cached() {
        let ip1 = gelu_tanh_inflection_point();
        let ip2 = gelu_tanh_inflection_point();
        assert_eq!(ip1, ip2, "Inflection point should be deterministic");
    }

    // =========================================================================
    // gelu_bound_interval
    // =========================================================================

    /// For a point interval [x, x], bound_interval should return (GELU(x), GELU(x)).
    #[test]
    fn test_gelu_bound_interval_point() {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let x = 1.0;
            let (min_v, max_v) = gelu_bound_interval(x, x, approx);
            let gx = gelu_eval(x, approx);
            assert!(
                (min_v - gx).abs() < 1e-6,
                "{approx:?} bound_interval({x},{x}) min={min_v}, expected {gx}"
            );
            assert!(
                (max_v - gx).abs() < 1e-6,
                "{approx:?} bound_interval({x},{x}) max={max_v}, expected {gx}"
            );
        }
    }

    /// Bound interval containing the critical point should include the GELU minimum.
    #[test]
    fn test_gelu_bound_interval_contains_minimum() {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let (min_v, _max_v) = gelu_bound_interval(-2.0, 2.0, approx);
            let cp = gelu_critical_point(approx);
            let gc = gelu_eval(cp, approx);
            assert!(
                min_v <= gc + 1e-6,
                "{approx:?}: bound_interval min={min_v} should be <= critical value {gc}"
            );
        }
    }

    /// Purely positive interval [2, 4]: GELU is monotonically increasing there.
    #[test]
    fn test_gelu_bound_interval_positive_monotone() {
        for approx in [GeluApproximation::Erf, GeluApproximation::Tanh] {
            let (min_v, max_v) = gelu_bound_interval(2.0, 4.0, approx);
            let g2 = gelu_eval(2.0, approx);
            let g4 = gelu_eval(4.0, approx);
            assert!(
                (min_v - g2).abs() < 1e-5,
                "{approx:?}: min should be GELU(2)={g2}, got {min_v}"
            );
            assert!(
                (max_v - g4).abs() < 1e-5,
                "{approx:?}: max should be GELU(4)={g4}, got {max_v}"
            );
        }
    }

    // =========================================================================
    // gelu_infinite_bounds_relaxation
    // =========================================================================

    /// Finite bounds should return None (caller handles normally).
    #[test]
    fn test_infinite_bounds_relaxation_finite_returns_none() {
        assert!(gelu_infinite_bounds_relaxation(-1.0, 1.0, GeluApproximation::Erf).is_none());
        assert!(gelu_infinite_bounds_relaxation(-1.0, 1.0, GeluApproximation::Tanh).is_none());
    }

    /// NaN bounds should return maximally loose.
    #[test]
    fn test_infinite_bounds_relaxation_nan() {
        let result = gelu_infinite_bounds_relaxation(f32::NAN, 1.0, GeluApproximation::Erf);
        assert!(result.is_some());
        let (ls, li, us, ui) = result.unwrap();
        assert_eq!(ls, 0.0);
        assert_eq!(li, f32::NEG_INFINITY);
        assert_eq!(us, 0.0);
        assert_eq!(ui, f32::INFINITY);
    }

    /// u = +inf should return maximally loose.
    #[test]
    fn test_infinite_bounds_relaxation_upper_inf() {
        let result = gelu_infinite_bounds_relaxation(-1.0, f32::INFINITY, GeluApproximation::Erf);
        assert!(result.is_some());
        let (ls, _li, us, _ui) = result.unwrap();
        assert_eq!(ls, 0.0);
        assert_eq!(us, 0.0);
    }

    /// l = -inf, u finite: should return constant (slope=0) bounds containing range.
    #[test]
    fn test_infinite_bounds_relaxation_lower_neg_inf() {
        let result =
            gelu_infinite_bounds_relaxation(f32::NEG_INFINITY, 1.0, GeluApproximation::Erf);
        assert!(result.is_some());
        let (ls, li, us, ui) = result.unwrap();
        assert_eq!(ls, 0.0, "lower slope should be 0");
        assert_eq!(us, 0.0, "upper slope should be 0");
        // li should be <= GELU(x) for all x in (-inf, 1], and ui should be >= GELU(x)
        assert!(
            li <= 0.0,
            "lower intercept {li} should be <= 0 (GELU minimum is negative)"
        );
        assert!(
            ui >= 0.0,
            "upper intercept {ui} should be >= 0 (GELU(1) > 0)"
        );
    }

    /// Malformed non-finite intervals (e.g., l=+inf u=-inf, l=0 u=-inf) must not
    /// panic — return maximally loose bounds instead. Regression test for #2739.
    #[test]
    fn test_infinite_bounds_relaxation_malformed_no_panic() {
        // l=+inf, u=-inf: degenerate empty interval
        let result = gelu_infinite_bounds_relaxation(
            f32::INFINITY,
            f32::NEG_INFINITY,
            GeluApproximation::Erf,
        );
        assert!(
            result.is_some(),
            "should handle l=+inf, u=-inf without panic"
        );
        let (ls, li, us, ui) = result.unwrap();
        assert_eq!(ls, 0.0);
        assert_eq!(li, f32::NEG_INFINITY);
        assert_eq!(us, 0.0);
        assert_eq!(ui, f32::INFINITY);

        // l=+inf, u=1.0: degenerate (lower > upper)
        let result = gelu_infinite_bounds_relaxation(f32::INFINITY, 1.0, GeluApproximation::Tanh);
        assert!(
            result.is_some(),
            "should handle l=+inf, u=finite without panic"
        );
        let (ls, li, us, ui) = result.unwrap();
        assert_eq!(ls, 0.0);
        assert_eq!(li, f32::NEG_INFINITY);
        assert_eq!(us, 0.0);
        assert_eq!(ui, f32::INFINITY);
    }

    // =========================================================================
    // f64 helpers
    // =========================================================================

    /// f64 GELU should agree with f32 GELU for normal inputs (within f32 precision).
    #[test]
    fn test_gelu_erf_f64_matches_f32() {
        for &x in &[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let f32_val = gelu_erf(x as f32) as f64;
            let f64_val = gelu_erf_f64(x);
            assert!(
                (f32_val - f64_val).abs() < 1e-5,
                "f64 vs f32 mismatch at x={x}: f32={f32_val}, f64={f64_val}"
            );
        }
    }

    /// f64 tanh GELU should agree with f32 tanh GELU.
    #[test]
    fn test_gelu_tanh_f64_matches_f32() {
        for &x in &[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let f32_val = gelu_tanh(x as f32) as f64;
            let f64_val = gelu_tanh_f64(x);
            assert!(
                (f32_val - f64_val).abs() < 1e-5,
                "f64 tanh GELU mismatch at x={x}: f32={f32_val}, f64={f64_val}"
            );
        }
    }

    /// f64 derivative should agree with f32 derivative.
    #[test]
    fn test_gelu_derivative_f64_matches_f32() {
        for &x in &[-1.0, 0.0, 1.0] {
            let f32_val = gelu_derivative(x as f32, GeluApproximation::Erf) as f64;
            let f64_val = gelu_derivative_erf_f64(x);
            assert!(
                (f32_val - f64_val).abs() < 1e-5,
                "Erf derivative f64 vs f32 mismatch at x={x}"
            );

            let f32_val = gelu_derivative(x as f32, GeluApproximation::Tanh) as f64;
            let f64_val = gelu_derivative_tanh_f64(x);
            assert!(
                (f32_val - f64_val).abs() < 1e-4,
                "Tanh derivative f64 vs f32 mismatch at x={x}"
            );
        }
    }

    // =========================================================================
    // f64 infinity guards (#2504)
    // =========================================================================

    /// gelu_erf_f64 must handle ±infinity and NaN correctly.
    #[test]
    fn test_gelu_erf_f64_infinity_guards() {
        assert_eq!(gelu_erf_f64(f64::NEG_INFINITY), 0.0, "GELU(-inf) = 0");
        assert_eq!(
            gelu_erf_f64(f64::INFINITY),
            f64::INFINITY,
            "GELU(+inf) = +inf"
        );
        assert!(gelu_erf_f64(f64::NAN).is_nan(), "GELU(NaN) = NaN");
    }

    /// gelu_tanh_f64 must handle ±infinity and NaN correctly.
    #[test]
    fn test_gelu_tanh_f64_infinity_guards() {
        assert_eq!(gelu_tanh_f64(f64::NEG_INFINITY), 0.0, "GELU_tanh(-inf) = 0");
        assert_eq!(
            gelu_tanh_f64(f64::INFINITY),
            f64::INFINITY,
            "GELU_tanh(+inf) = +inf"
        );
        assert!(gelu_tanh_f64(f64::NAN).is_nan(), "GELU_tanh(NaN) = NaN");
    }

    /// gelu_derivative_erf_f64 must handle ±infinity and NaN.
    /// GELU'(-inf) = 0 (Phi(-inf) = 0), GELU'(+inf) = 1 (Phi(+inf) = 1).
    #[test]
    fn test_gelu_derivative_erf_f64_infinity_guards() {
        assert_eq!(
            gelu_derivative_erf_f64(f64::NEG_INFINITY),
            0.0,
            "GELU'(-inf) = 0"
        );
        assert_eq!(
            gelu_derivative_erf_f64(f64::INFINITY),
            1.0,
            "GELU'(+inf) = 1"
        );
        assert!(
            gelu_derivative_erf_f64(f64::NAN).is_nan(),
            "GELU'(NaN) = NaN"
        );
    }

    /// gelu_derivative_tanh_f64 must handle ±infinity and NaN.
    #[test]
    fn test_gelu_derivative_tanh_f64_infinity_guards() {
        assert_eq!(
            gelu_derivative_tanh_f64(f64::NEG_INFINITY),
            0.0,
            "GELU'_tanh(-inf) = 0"
        );
        assert_eq!(
            gelu_derivative_tanh_f64(f64::INFINITY),
            1.0,
            "GELU'_tanh(+inf) = 1"
        );
        assert!(
            gelu_derivative_tanh_f64(f64::NAN).is_nan(),
            "GELU'_tanh(NaN) = NaN"
        );
    }

    // =========================================================================
    // check_lower/upper_gelu_f64
    // =========================================================================

    /// check_lower_gelu_f64: in the convex region (x in [-√2, 0]), tangent at d
    /// should be a valid lower bound at nearby upper within the same convex segment.
    /// GELU is convex on approximately [-√2, 0] (second derivative > 0).
    #[test]
    fn test_check_lower_gelu_f64_convex_region() {
        // x ∈ [-1.0, -0.2] is well within the convex part of GELU
        assert!(
            check_lower_gelu_f64(-0.2, -0.8),
            "Tangent at -0.8 should be valid lower bound at -0.2 in convex region"
        );
    }

    /// check_lower_gelu_f64 returns bool without panicking for various inputs.
    #[test]
    fn test_check_lower_gelu_f64_no_panic() {
        // Various inputs should not panic
        for &(upper, d) in &[(2.0, 5.0), (-1.0, -2.0), (0.0, 0.0), (10.0, 0.5)] {
            let _ = check_lower_gelu_f64(upper, d);
        }
    }

    /// check_upper_gelu_f64: in the concave region of GELU (x > √2, where GELU is
    /// concave), tangent at d should be above the curve at lower.
    #[test]
    fn test_check_upper_gelu_f64_concave_region() {
        // For x > √2 ≈ 1.414, GELU is concave (second derivative < 0).
        // Tangent at d=3.0 should be above GELU at lower=2.0 (tangent of concave function is above it).
        assert!(
            check_upper_gelu_f64(2.0, 3.0),
            "Tangent at 3.0 should be valid upper bound at 2.0 in concave region"
        );
    }

    // =========================================================================
    // Tanh check helpers
    // =========================================================================

    #[test]
    fn test_check_lower_gelu_tanh_f64_convex_region() {
        // Same convex region [-split, 0] for tanh GELU
        assert!(
            check_lower_gelu_tanh_f64(-0.2, -0.8),
            "Tangent at -0.8 should be valid lower bound at -0.2 for tanh GELU"
        );
    }

    #[test]
    fn test_check_upper_gelu_tanh_f64_concave_region() {
        // Concave region for tanh GELU at x > split (~1.42)
        assert!(
            check_upper_gelu_tanh_f64(2.0, 3.0),
            "Tangent at 3.0 should be valid upper bound at 2.0 for tanh GELU"
        );
    }
}
