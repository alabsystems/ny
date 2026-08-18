// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Precomputed tangent point tables for sound GELU relaxation.
//!
//! Port of auto_LiRPA's BoundGelu precomputed tangent tables.
//! Reference: α,β-CROWN Team, auto_LiRPA @ 9d100ec070868440b48d34e2f1dd21b97aab9172

use std::sync::OnceLock;

use super::eval::{
    check_lower_gelu_f64, check_lower_gelu_tanh_f64, check_upper_gelu_f64,
    check_upper_gelu_tanh_f64, gelu_critical_point, gelu_derivative_erf_f64,
    gelu_derivative_tanh_f64, gelu_erf_f64, gelu_tanh_f64, gelu_tanh_inflection_point,
};
use super::GeluApproximation;
use ny_core::{f32_affine_eval_error, f64_to_f32_down, f64_to_f32_up};
use ny_tensor::{next_down_f32, next_up_f32};

/// sqrt(2) constant used for case-splitting in GELU relaxation.
pub(crate) const SQRT_2: f32 = std::f32::consts::SQRT_2;

/// Erf GELU global minimizer used by auto_LiRPA for table bracketing.
pub(crate) const GELU_MINIMIZER_X: f32 = -0.7517916;

/// Precomputed tangent point tables for sound GELU relaxation.
pub(crate) struct GeluPrecomputeTables {
    pub(crate) step_pre: f32,
    pub(crate) d_lower_right: Vec<f32>,
    pub(crate) d_lower_left: Vec<f32>,
    pub(crate) d_upper_right: Vec<f32>,
    pub(crate) d_upper_left: Vec<f32>,
}

impl GeluPrecomputeTables {
    fn new() -> Self {
        let step_pre: f32 = 0.01;
        let step_f64: f64 = step_pre as f64;
        let x_limit: f64 = 1000.0;
        // SAFETY: x_limit=1000.0 / step_f64=0.01 = 100000.0 — compile-time constants,
        // always positive, finite, and well within usize range.
        let num_points = (x_limit / step_f64) as usize + 5;
        let max_iter = 100;
        let sqrt2: f64 = std::f64::consts::SQRT_2;
        let minimizer_f64: f64 = GELU_MINIMIZER_X as f64;

        // Match auto_LiRPA's BoundGelu.precompute_relaxation() construction.
        // Bisection uses f64 for ~2^{-100} precision, matching the reference.
        // Results are stored as f32 after rounding.

        // d_lower_right: keyed by upper - sqrt(2), upper ∈ [sqrt(2), +∞).
        let mut d_lower_right = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let upper = step_f64 * (i as f64) + sqrt2;
            let mut r = 1.0_f64;
            let mut l = -1.0_f64;

            for _ in 0..200 {
                if check_lower_gelu_f64(upper, l) {
                    break;
                }
                l *= 2.0;
            }

            for _ in 0..max_iter {
                // Bit-identical: bounded bracket keeps f64::midpoint on its `(a + b) * 0.5` path.
                let m = f64::midpoint(l, r);
                if check_lower_gelu_f64(upper, m) {
                    l = m;
                } else {
                    r = m;
                }
            }

            d_lower_right.push(l as f32);
        }

        // d_upper_right: keyed by -lower + sqrt(2), lower ∈ (0, sqrt(2)] (clamped to 0.01).
        let mut d_upper_right = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let lower = (sqrt2 - step_f64 * (i as f64)).max(step_f64);
            let mut l = sqrt2;
            let mut r = x_limit;

            for _ in 0..200 {
                if check_upper_gelu_f64(lower, r) {
                    break;
                }
                r *= 2.0;
            }

            for _ in 0..max_iter {
                // Bit-identical: bounded bracket keeps f64::midpoint on its `(a + b) * 0.5` path.
                let m = f64::midpoint(l, r);
                if check_upper_gelu_f64(lower, m) {
                    r = m;
                } else {
                    l = m;
                }
            }

            d_upper_right.push(r as f32);
        }

        // d_lower_left: keyed by -lower - sqrt(2), upper ∈ (-∞, -sqrt(2)].
        let mut d_lower_left = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let upper = -(step_f64 * (i as f64)) - sqrt2;
            let mut l = -sqrt2;
            let mut r = minimizer_f64;

            for _ in 0..200 {
                if check_lower_gelu_f64(upper, r) {
                    break;
                }
                r *= 2.0;
            }

            for _ in 0..max_iter {
                // Bit-identical: bounded bracket keeps f64::midpoint on its `(a + b) * 0.5` path.
                let m = f64::midpoint(l, r);
                if check_lower_gelu_f64(upper, m) {
                    r = m;
                } else {
                    l = m;
                }
            }

            d_lower_left.push(r as f32);
        }

        // d_upper_left: keyed by -lower - sqrt(2), lower ∈ [-sqrt(2), 0] (clamped to 0).
        let mut d_upper_left = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let lower = (step_f64 * (i as f64) - sqrt2).min(0.0);
            let mut l = -x_limit;
            let mut r = -sqrt2;

            for _ in 0..200 {
                if check_upper_gelu_f64(lower, l) {
                    break;
                }
                l *= 2.0;
            }

            for _ in 0..max_iter {
                // Bit-identical: bounded bracket keeps f64::midpoint on its `(a + b) * 0.5` path.
                let m = f64::midpoint(l, r);
                if check_upper_gelu_f64(lower, m) {
                    r = m;
                } else {
                    l = m;
                }
            }

            d_upper_left.push(r as f32);
        }

        Self {
            step_pre,
            d_lower_right,
            d_lower_left,
            d_upper_right,
            d_upper_left,
        }
    }

    #[inline]
    pub(crate) fn retrieve(&self, table: &[f32], bound: f32, default_d: f32) -> f32 {
        // Clamp the float index to table length before integer conversion to prevent
        // overflow panic when bound is very large (e.g., 1e15 from multi-block IBP).
        // For out-of-range bounds, we return default_d (the asymptotic tangent slope).
        let raw = (bound / self.step_pre).floor();
        if raw >= table.len() as f32 || raw.is_nan() {
            return default_d;
        }
        // SAFETY: raw is finite (NaN filtered above), < table.len() as f32.
        // raw as isize may be negative for small bounds, .max(0) clamps to 0.
        // Final usize cast is from a non-negative isize, guaranteed in-bounds.
        let idx = (raw as isize + 1).max(0) as usize;
        if idx >= table.len() {
            default_d
        } else {
            table[idx]
        }
    }
}

pub(crate) fn get_gelu_precompute() -> &'static GeluPrecomputeTables {
    static TABLES: OnceLock<GeluPrecomputeTables> = OnceLock::new();
    TABLES.get_or_init(GeluPrecomputeTables::new)
}

/// Precomputed tangent point tables for sound GELU(tanh) relaxation.
pub(crate) struct GeluTanhPrecomputeTables {
    pub(crate) step_pre: f32,
    pub(crate) d_lower_right: Vec<f32>,
    pub(crate) d_lower_left: Vec<f32>,
    pub(crate) d_upper_right: Vec<f32>,
    pub(crate) d_upper_left: Vec<f32>,
}

impl GeluTanhPrecomputeTables {
    fn new() -> Self {
        let step_pre: f32 = 0.01;
        let step_f64: f64 = step_pre as f64;
        let x_limit: f64 = 1000.0;
        // SAFETY: x_limit=1000.0 / step_f64=0.01 = 100000.0 — compile-time constants,
        // always positive, finite, and well within usize range.
        let num_points = (x_limit / step_f64) as usize + 5;
        let max_iter = 100;
        let split = gelu_tanh_inflection_point() as f64;
        let minimizer = gelu_critical_point(GeluApproximation::Tanh) as f64;

        // Bisection uses f64 for ~2^{-100} precision, matching the reference.
        // Results are stored as f32 after rounding.

        // d_lower_right: keyed by upper - split, upper ∈ [split, +∞).
        let mut d_lower_right = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let upper = step_f64 * (i as f64) + split;
            let mut r = 1.0_f64;
            let mut l = -1.0_f64;

            for _ in 0..200 {
                if check_lower_gelu_tanh_f64(upper, l) {
                    break;
                }
                l *= 2.0;
            }

            for _ in 0..max_iter {
                // Bit-identical: bounded bracket keeps f64::midpoint on its `(a + b) * 0.5` path.
                let m = f64::midpoint(l, r);
                if check_lower_gelu_tanh_f64(upper, m) {
                    l = m;
                } else {
                    r = m;
                }
            }

            d_lower_right.push(l as f32);
        }

        // d_upper_right: keyed by -lower + split, lower ∈ (0, split] (clamped to 0.01).
        let mut d_upper_right = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let lower = (split - step_f64 * (i as f64)).max(step_f64);
            let mut l = split;
            let mut r = x_limit;

            for _ in 0..200 {
                if check_upper_gelu_tanh_f64(lower, r) {
                    break;
                }
                r *= 2.0;
            }

            for _ in 0..max_iter {
                // Bit-identical: bounded bracket keeps f64::midpoint on its `(a + b) * 0.5` path.
                let m = f64::midpoint(l, r);
                if check_upper_gelu_tanh_f64(lower, m) {
                    r = m;
                } else {
                    l = m;
                }
            }

            d_upper_right.push(r as f32);
        }

        // d_lower_left: keyed by -lower - split, upper ∈ (-∞, -split].
        let mut d_lower_left = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let upper = -(step_f64 * (i as f64)) - split;
            let mut l = -split;
            let mut r = minimizer;

            for _ in 0..200 {
                if check_lower_gelu_tanh_f64(upper, r) {
                    break;
                }
                r *= 2.0;
            }

            for _ in 0..max_iter {
                // Bit-identical: bounded bracket keeps f64::midpoint on its `(a + b) * 0.5` path.
                let m = f64::midpoint(l, r);
                if check_lower_gelu_tanh_f64(upper, m) {
                    r = m;
                } else {
                    l = m;
                }
            }

            d_lower_left.push(r as f32);
        }

        // d_upper_left: keyed by -lower - split, lower ∈ [-split, 0] (clamped to 0).
        let mut d_upper_left = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let lower = (step_f64 * (i as f64) - split).min(0.0);
            let mut l = -x_limit;
            let mut r = -split;

            for _ in 0..200 {
                if check_upper_gelu_tanh_f64(lower, l) {
                    break;
                }
                l *= 2.0;
            }

            for _ in 0..max_iter {
                // Bit-identical: bounded bracket keeps f64::midpoint on its `(a + b) * 0.5` path.
                let m = f64::midpoint(l, r);
                if check_upper_gelu_tanh_f64(lower, m) {
                    r = m;
                } else {
                    l = m;
                }
            }

            d_upper_left.push(r as f32);
        }

        Self {
            step_pre,
            d_lower_right,
            d_lower_left,
            d_upper_right,
            d_upper_left,
        }
    }

    #[inline]
    pub(crate) fn retrieve(&self, table: &[f32], bound: f32, default_d: f32) -> f32 {
        let raw = (bound / self.step_pre).floor();
        if raw >= table.len() as f32 || raw.is_nan() {
            return default_d;
        }
        // SAFETY: raw is finite (NaN filtered above), < table.len() as f32.
        // raw as isize may be negative for small bounds, .max(0) clamps to 0.
        // Final usize cast is from a non-negative isize, guaranteed in-bounds.
        let idx = (raw as isize + 1).max(0) as usize;
        if idx >= table.len() {
            default_d
        } else {
            table[idx]
        }
    }
}

pub(crate) fn get_gelu_tanh_precompute() -> &'static GeluTanhPrecomputeTables {
    static TABLES: OnceLock<GeluTanhPrecomputeTables> = OnceLock::new();
    TABLES.get_or_init(GeluTanhPrecomputeTables::new)
}

/// Compute tangent line parameters at point `d` for Erf GELU with directed rounding.
///
/// Returns `(slope, lower_intercept, upper_intercept)` where the lower intercept
/// is rounded down and the upper intercept is rounded up to absorb f64→f32
/// truncation error. This follows the SiLU directed rounding pattern (#1822, #3146).
///
/// `max_abs_x` is the maximum absolute value of x over the evaluation interval,
/// used to bound the slope truncation error: |slope64 - slope_f32| * max_abs_x.
///
/// # Mathematics
///
/// The tangent line at d is: y = GELU'(d) * (x - d) + GELU(d) = slope * x + intercept
/// where intercept = GELU(d) - GELU'(d) * d. For large |d| where GELU(d) ≈ d,
/// computing intercept = gd - slope * d in f32 suffers catastrophic cancellation.
/// Computing in f64 and then applying directed rounding eliminates this.
///
/// Reference: SiLU tangent pattern, silu/math.rs:216-231
#[inline]
pub(crate) fn gelu_tangent_at(d: f32, max_abs_x: f32) -> (f32, f32, f32) {
    let d64 = d as f64;
    let slope64 = gelu_derivative_erf_f64(d64);
    let intercept64 = gelu_erf_f64(d64) - slope64 * d64;

    // Directed rounding: absorb f64→f32 slope truncation error.
    // The f32 slope differs from f64 by up to 1 ULP. Over the interval [-max_abs_x, max_abs_x],
    // this contributes at most |slope64 - slope_f32| * max_abs_x to the bound error.
    let slope_f32 = slope64 as f32;
    let eval_err = f32_affine_eval_error(slope64, slope_f32, intercept64, max_abs_x);
    (
        slope_f32,
        next_down_f32(f64_to_f32_down(intercept64 - eval_err)),
        next_up_f32(f64_to_f32_up(intercept64 + eval_err)),
    )
}

/// Compute tangent line parameters at point `d` for tanh-approx GELU with directed rounding.
///
/// Same pattern as `gelu_tangent_at` but uses the tanh-approximation GELU.
/// Reference: SiLU tangent pattern, silu/math.rs:216-231
#[inline]
pub(crate) fn gelu_tanh_tangent_at(d: f32, max_abs_x: f32) -> (f32, f32, f32) {
    let d64 = d as f64;
    let slope64 = gelu_derivative_tanh_f64(d64);
    let intercept64 = gelu_tanh_f64(d64) - slope64 * d64;

    let slope_f32 = slope64 as f32;
    let eval_err = f32_affine_eval_error(slope64, slope_f32, intercept64, max_abs_x);
    (
        slope_f32,
        next_down_f32(f64_to_f32_down(intercept64 - eval_err)),
        next_up_f32(f64_to_f32_up(intercept64 + eval_err)),
    )
}

/// Post-hoc soundness verification for GELU linear relaxation.
///
/// Checks the relaxation at N sample points, then adjusts intercepts so the
/// lower line is at-or-below and the upper line is at-or-above the true GELU
/// value everywhere. The safety margin accounts for the maximum undetected
/// error between adjacent sample points using the curvature bound.
///
/// # Mathematical justification (#2445)
///
/// Between adjacent sample points separated by `h = (u-l)/N_INTERVALS`, the
/// maximum undetected "bulge" of GELU past a linear bound is bounded by:
///   `bulge <= |GELU''|_max * h^2 / 8`
///
/// GELU''(x) = phi(x)(2 - x^2) where phi is the standard normal PDF.
/// The maximum |GELU''(x)| = sqrt(2/pi) ≈ 0.798 (at x = 0). With N_INTERVALS=50:
///   For width 3.0: bulge <= 0.798 * (3.0/50)^2 / 8 ≈ 3.6e-4
///
/// The total safety margin is `SAFETY_EPS + curvature_margin` where
/// `curvature_margin = GELU_MAX_CURVATURE * h^2 / 8`.
///
/// Reference: α,β-CROWN uses f64 binary search (100 iterations, ~2^{-100} precision).
#[inline]
pub(crate) fn gelu_posthoc_adjust(
    l: f32,
    u: f32,
    lower_slope: f32,
    lower_intercept: f32,
    upper_slope: f32,
    upper_intercept: f32,
    gelu_fn: fn(f32) -> f32,
) -> (f32, f32, f32, f32) {
    const SAFETY_EPS: f32 = 1e-5;
    // Number of intervals between sample points (51 points = 50 intervals).
    const N_INTERVALS: u32 = 50;
    // Sound upper bound on |GELU''(x)| over all x.
    // GELU''(x) = phi(x)(2 - x^2), max |GELU''(x)| = sqrt(2/pi) ≈ 0.798 at x = 0.
    const GELU_MAX_CURVATURE: f32 = 0.8;

    let mut lower_violation = 0.0_f32;
    let mut upper_violation = 0.0_f32;

    // Check at N_INTERVALS+1 evenly spaced points including both endpoints.
    for i in 0..=N_INTERVALS {
        let t = i as f32 / N_INTERVALS as f32;
        let x = l + (u - l) * t;
        let gx = gelu_fn(x);
        // Lower bound should be <= gelu(x): violation when lower_line(x) > gelu(x).
        let lower_line = lower_slope * x + lower_intercept;
        lower_violation = lower_violation.max(lower_line - gx);
        // Upper bound should be >= gelu(x): violation when upper_line(x) < gelu(x).
        let upper_line = upper_slope * x + upper_intercept;
        upper_violation = upper_violation.max(gx - upper_line);
    }

    // Curvature-based margin: maximum undetected error between sample points.
    // h = (u - l) / N_INTERVALS; margin = GELU_MAX_CURVATURE * h^2 / 8
    let h = (u - l) / N_INTERVALS as f32;
    let curvature_margin = GELU_MAX_CURVATURE * h * h / 8.0;
    let total_margin = SAFETY_EPS + curvature_margin;

    let adj_lower = if lower_violation > 0.0 {
        lower_intercept - lower_violation - total_margin
    } else {
        lower_intercept - total_margin
    };
    let adj_upper = if upper_violation > 0.0 {
        upper_intercept + upper_violation + total_margin
    } else {
        upper_intercept + total_margin
    };

    (lower_slope, adj_lower, upper_slope, adj_upper)
}

#[cfg(test)]
mod tests;
