// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Abs linear relaxation.

use crate::types::LinearRelaxation;

/// Linear relaxation for |x| on interval [l, u].
pub fn abs_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    if !l.is_finite() || !u.is_finite() {
        return LinearRelaxation::new(0.0, f32::NEG_INFINITY, 0.0, f32::INFINITY);
    }
    if l >= 0.0 {
        LinearRelaxation::new(1.0, 0.0, 1.0, 0.0)
    } else if u <= 0.0 {
        LinearRelaxation::new(-1.0, 0.0, -1.0, 0.0)
    } else {
        let l64 = l as f64;
        let u64 = u as f64;
        let slope = ((u64 + l64) / (u64 - l64)) as f32;
        let slope_f64 = slope as f64;
        let needed_at_l = -l64 - slope_f64 * l64;
        let needed_at_u = u64 - slope_f64 * u64;
        let needed = needed_at_l.max(needed_at_u);
        let eps_f64: f64 = f64::from(f32::EPSILON);
        let max_endpoint = (-l64).max(u64);
        let margin = 4.0 * eps_f64 * max_endpoint;
        let intercept = (needed + margin) as f32;
        let intercept = if intercept.is_finite() {
            intercept
        } else {
            needed as f32
        };
        let lower_slope = if u > -l { 1.0 } else { -1.0 };
        LinearRelaxation::new(lower_slope, 0.0, slope, intercept)
    }
}
