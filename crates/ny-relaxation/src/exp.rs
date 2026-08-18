// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Exp linear relaxation.

use crate::rounding::{next_down_f32, next_up_f32};
use crate::types::LinearRelaxation;
use ny_core::{f64_to_f32_down, f64_to_f32_up, nan_propagating_max};

/// Compute sound f64→f32 intercept correction for one bound direction.
///
/// Error sources when caller evaluates `slope_f32 * x + intercept_f32`:
///   E1: slope f64→f32 truncation  <= |slope_f64 - slope_f32| * |x|
///   E2: f32 multiplication        <= |slope_f32| * |x| * eps
///   E3: f32 addition rounding     <= (|slope_f32*x| + |intercept|) * eps
///   E4: f32::exp() faithful round <= exp(x) * eps
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

/// Linear relaxation for exp(x) on interval [l, u].
pub fn exp_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l > u {
        return LinearRelaxation::nan_fallback();
    }

    let l64 = l as f64;
    let u64 = u as f64;

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

    let upper_slope = (exp_u - exp_l) / (u64 - l64);
    let upper_intercept = exp_l - upper_slope * l64;

    // Uncapped midpoint tangent, matching the production implementation in
    // `ny_propagate::layers::activations::exp` (tangents to convex exp are
    // sound lower bounds at ANY point; the tangent slope exp(m) never exceeds
    // the chord slope, so no cap is needed). Kept textually identical to
    // production so tests/drift.rs can assert bit-for-bit equality.
    let m = f64::midpoint(l64, u64);
    let exp_m = m.exp();
    let lower_slope = exp_m;
    let lower_intercept = exp_m * (1.0 - m);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_or_unbounded_interval_returns_conservative_relaxation() {
        for (lower, upper) in [
            (f32::NAN, 0.0),
            (0.0, f32::NAN),
            (1.0, -1.0),
            (f32::NEG_INFINITY, 0.0),
            (0.0, f32::INFINITY),
        ] {
            assert_eq!(
                exp_linear_relaxation(lower, upper),
                LinearRelaxation::nan_fallback()
            );
        }
    }

    #[test]
    fn finite_overflow_risk_returns_conservative_relaxation() {
        for (lower, upper) in [(89.0, 89.0), (1_000.0, 1_001.0)] {
            assert_eq!(
                exp_linear_relaxation(lower, upper),
                LinearRelaxation::nan_fallback(),
                "finite endpoints must still fail closed when affine coefficients overflow"
            );
        }
    }

    #[test]
    fn underflow_point_line_remains_sound_under_ftz() {
        let x = -100.0_f32;
        let reference = (x as f64).exp();
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
}
