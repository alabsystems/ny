// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Power-of-2 linear relaxation.

use crate::rounding::{next_down_f32, next_up_f32};
use crate::types::LinearRelaxation;

/// Linear relaxation for x^2 on interval [l, u].
pub fn pow2_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }

    let l64 = l as f64;
    let u64 = u as f64;

    if l == u {
        let y = (l64 * l64) as f32;
        return LinearRelaxation::new(0.0, next_down_f32(y), 0.0, next_up_f32(y));
    }

    let upper_slope = (l64 + u64) as f32;
    let s64 = upper_slope as f64;
    let needed_at_l = l64 * l64 - s64 * l64;
    let needed_at_u = u64 * u64 - s64 * u64;
    let needed_upper = needed_at_l.max(needed_at_u);
    let eps_f64: f64 = f64::from(f32::EPSILON);
    let max_sq = (l64 * l64).max(u64 * u64);
    let upper_margin = 4.0 * eps_f64 * max_sq;
    let upper_intercept = next_up_f32((needed_upper + upper_margin) as f32);
    let upper_intercept = if upper_intercept.is_finite() {
        upper_intercept
    } else {
        next_up_f32(needed_upper as f32)
    };

    if l < 0.0 && u > 0.0 {
        return LinearRelaxation::new(0.0, 0.0, upper_slope, upper_intercept);
    }

    if max_sq < f64::from(f32::MIN_POSITIVE) {
        return LinearRelaxation::new(0.0, 0.0, upper_slope, upper_intercept);
    }

    let m64 = 0.5 * (l64 + u64);
    let lower_slope = (2.0 * m64) as f32;
    let ls64 = lower_slope as f64;
    let vertex_x = ls64 / 2.0;
    let allowed_at_l = l64 * l64 - ls64 * l64;
    let allowed_at_u = u64 * u64 - ls64 * u64;
    let allowed_at_v = vertex_x * vertex_x - ls64 * vertex_x;
    let allowed_lower = allowed_at_l.min(allowed_at_u).min(allowed_at_v);
    let lower_margin = 4.0 * eps_f64 * max_sq;
    let lower_intercept = next_down_f32((allowed_lower - lower_margin) as f32);
    let lower_intercept = if lower_intercept.is_finite() {
        lower_intercept
    } else {
        next_down_f32(allowed_lower as f32)
    };

    LinearRelaxation::new(lower_slope, lower_intercept, upper_slope, upper_intercept)
}
