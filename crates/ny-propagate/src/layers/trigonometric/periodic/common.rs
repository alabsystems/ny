// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared infrastructure for periodic activation layers (Sin, Cos, Tan).
//!
//! Contains normalization, tangent/secant relaxation, and constant-bound
//! fallback utilities used by all three periodic types.

use ndarray::Array2;
use ny_core::{NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

use crate::layers::activations::LinearRelaxation;
use crate::LinearBounds;

pub(super) const TRIG_RELAX_EPS: f32 = 1e-5;

pub(super) fn trig_constant_relaxation() -> LinearRelaxation {
    LinearRelaxation::new(0.0, -1.0 - TRIG_RELAX_EPS, 0.0, 1.0 + TRIG_RELAX_EPS)
}

pub(super) fn normalize_trig_interval(l: f64, u: f64) -> Option<(f64, f64)> {
    let two_pi = 2.0 * std::f64::consts::PI;
    if !l.is_finite() || !u.is_finite() || u <= l {
        return None;
    }
    if u - l >= two_pi {
        return None;
    }
    let k = (l / two_pi).floor();
    let l_norm = l - k * two_pi;
    let u_norm = u - k * two_pi;
    if u_norm >= two_pi {
        return None;
    }
    Some((l_norm, u_norm))
}

pub(super) fn trig_tangent_secant_relaxation<F, D>(
    l: f64,
    u: f64,
    f: F,
    df: D,
    is_concave: bool,
    constant_relaxation: fn() -> LinearRelaxation,
) -> LinearRelaxation
where
    F: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    if !l.is_finite() || !u.is_finite() || !((u - l).is_finite()) {
        return constant_relaxation();
    }

    let interval = u - l;
    if interval.abs() < 1e-12 {
        let f_l = f(l);
        let slope = df(l);
        let intercept = f_l - slope * l;
        if !f_l.is_finite() || !slope.is_finite() || !intercept.is_finite() {
            return constant_relaxation();
        }
        let slope_f32 = slope as f32;
        let max_abs_x = l.abs().max(u.abs());
        // Compute error in f64, cast with next_up_f32 to avoid underestimate (#2636).
        let slope_err = next_up_f32(((slope - slope_f32 as f64).abs() * max_abs_x) as f32);
        let intercept_f32 = next_down_f32((intercept as f32) - TRIG_RELAX_EPS - slope_err);
        let upper_intercept = next_up_f32((intercept as f32) + TRIG_RELAX_EPS + slope_err);
        return LinearRelaxation::new(slope_f32, intercept_f32, slope_f32, upper_intercept);
    }

    let f_l = f(l);
    let f_u = f(u);
    let secant_slope = (f_u - f_l) / interval;
    let secant_intercept = f_l - secant_slope * l;

    let t_l_slope = df(l);
    let t_l_intercept = f_l - t_l_slope * l;
    let t_u_slope = df(u);
    let t_u_intercept = f_u - t_u_slope * u;

    let t_l_at_u = t_l_slope * u + t_l_intercept;
    let t_u_at_l = t_u_slope * l + t_u_intercept;

    let (tan_slope, tan_intercept) = if is_concave {
        if t_l_at_u <= t_u_at_l {
            (t_l_slope, t_l_intercept)
        } else {
            (t_u_slope, t_u_intercept)
        }
    } else if t_l_at_u >= t_u_at_l {
        (t_l_slope, t_l_intercept)
    } else {
        (t_u_slope, t_u_intercept)
    };

    let (lower_slope, lower_intercept, upper_slope, upper_intercept) = if is_concave {
        (secant_slope, secant_intercept, tan_slope, tan_intercept)
    } else {
        (tan_slope, tan_intercept, secant_slope, secant_intercept)
    };

    let max_abs_x = l.abs().max(u.abs());
    if !max_abs_x.is_finite() {
        return constant_relaxation();
    }

    let lower_slope_f32 = lower_slope as f32;
    let upper_slope_f32 = upper_slope as f32;
    // Compute error in f64 (max_abs_x is already f64), cast with next_up_f32
    // to avoid rounding underestimate on intercept widening (#2636).
    let lower_err = next_up_f32(((lower_slope - lower_slope_f32 as f64).abs() * max_abs_x) as f32);
    let upper_err = next_up_f32(((upper_slope - upper_slope_f32 as f64).abs() * max_abs_x) as f32);

    let lower_intercept = next_down_f32((lower_intercept as f32) - TRIG_RELAX_EPS - lower_err);
    let upper_intercept = next_up_f32((upper_intercept as f32) + TRIG_RELAX_EPS + upper_err);

    LinearRelaxation::new(
        lower_slope_f32,
        lower_intercept,
        upper_slope_f32,
        upper_intercept,
    )
}

pub(super) fn constant_bounds_from_output(
    bounds: &LinearBounds,
    output_bounds: &BoundedTensor,
) -> Result<LinearBounds> {
    let output_flat = output_bounds.flatten();
    let output_len = output_flat.len();
    if bounds.num_inputs() != output_len {
        return Err(NyError::ShapeMismatch {
            expected: vec![bounds.num_inputs()],
            got: vec![output_len],
        });
    }

    let concretized = bounds.concretize_sound(&output_flat); // #2236: directed rounding for soundness
    let lower_shape = concretized.lower().shape().to_vec();
    let lower = concretized
        .lower()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![bounds.num_outputs()],
            got: lower_shape,
        })?;
    let upper_shape = concretized.upper().shape().to_vec();
    let upper = concretized
        .upper()
        .clone()
        .into_dimensionality::<ndarray::Ix1>()
        .map_err(|_| NyError::ShapeMismatch {
            expected: vec![bounds.num_outputs()],
            got: upper_shape,
        })?;

    LinearBounds::new_or_conservative(
        Array2::zeros((bounds.num_outputs(), bounds.num_inputs())),
        lower,
        Array2::zeros((bounds.num_outputs(), bounds.num_inputs())),
        upper,
    )
}
