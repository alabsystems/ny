// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sqrt linear relaxation.

use crate::rounding::{next_down_f32, next_up_f32};
use crate::types::LinearRelaxation;

const SQRT_ALPHA_MIN_MID: f32 = 1e-6;

/// Linear relaxation for sqrt(x) on interval [l, u].
///
/// The default tangent point is the chord-parallel (minimal-gap) point
/// t* = ((sqrt(l)+sqrt(u))/2)^2, matching the production default in
/// `ny_propagate::layers::arithmetic::sqrt` (a tangent to concave sqrt is a
/// sound upper bound at ANY point in the domain, and the with-alpha body
/// re-clamps `mid` into [l, u] regardless).
pub fn sqrt_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    sqrt_linear_relaxation_with_alpha(
        l,
        u,
        f32::midpoint((l.max(0.0)).sqrt(), u.max(0.0).sqrt()).powi(2),
    )
}

/// Linear relaxation for sqrt(x) with a configurable tangent point `mid`.
pub fn sqrt_linear_relaxation_with_alpha(l: f32, u: f32, mid: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() || l > u {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    let original_l = l;
    let l = l.max(0.0);
    let u = u.max(0.0);

    if u <= 0.0 {
        return LinearRelaxation::new(0.0, 0.0, 0.0, 0.0);
    }
    if (u - l).abs() < 1e-8 {
        let lower = l.sqrt();
        let upper = u.sqrt();
        return LinearRelaxation::new(0.0, next_down_f32(lower), 0.0, next_up_f32(upper));
    }
    if u < 1e-12 {
        let lower = l.sqrt();
        let upper = u.sqrt();
        return LinearRelaxation::new(0.0, next_down_f32(lower), 0.0, next_up_f32(upper));
    }

    let l64 = l as f64;
    let u64 = u as f64;

    let sqrt_l = l64.sqrt();
    let sqrt_u = u64.sqrt();

    let chord_slope = (sqrt_u - sqrt_l) / (u64 - l64);
    let chord_intercept = sqrt_l - chord_slope * l64;

    let max_abs_x = l.abs().max(u.abs()) as f64;

    let lower_slope = chord_slope as f32;
    let lower_slope_err =
        next_up_f32(((chord_slope - lower_slope as f64).abs() * max_abs_x) as f32);
    // Account for f32 multiplication rounding: `slope * x` has error up to
    // |slope| * |x| * f32::EPSILON. For the lower bound we round DOWN.
    let lower_mul_err = next_up_f32((lower_slope.abs() * max_abs_x as f32) * f32::EPSILON);
    let lower_intercept = next_down_f32((chord_intercept as f32) - lower_slope_err - lower_mul_err);

    let mid = if u > 0.0 {
        mid.clamp(l.max(SQRT_ALPHA_MIN_MID.min(u)), u)
    } else {
        0.0
    };
    let mid64 = mid as f64;
    let sqrt_mid = mid64.sqrt();

    let tangent_slope_f64 = 0.5 / sqrt_mid;
    let tangent_intercept_f64 = sqrt_mid - tangent_slope_f64 * mid64;
    let tangent_slope = tangent_slope_f64 as f32;
    let tangent_slope_err =
        next_up_f32(((tangent_slope_f64 - tangent_slope as f64).abs() * max_abs_x) as f32);
    // Also account for f32 multiplication rounding: evaluating `slope * x` in f32
    // has error up to |slope| * |x| * f32::EPSILON. Without this, the upper bound
    // is unsound when mid is near lower (large slope, large multiplication error).
    // Discovered by Kani proof `sqrt_alpha_upper_sound_mid_at_lower`.
    let mul_rounding_err = next_up_f32((tangent_slope.abs() * max_abs_x as f32) * f32::EPSILON);
    let tangent_intercept =
        next_up_f32((tangent_intercept_f64 as f32) + tangent_slope_err + mul_rounding_err);
    let upper_slope = tangent_slope;
    let min_intercept = if original_l < 0.0 {
        next_up_f32(-tangent_slope * original_l)
    } else {
        tangent_intercept
    };
    let upper_intercept = tangent_intercept.max(min_intercept);

    LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
}
