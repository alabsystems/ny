// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core mathematical functions for SiLU (Swish) activation.

use std::sync::OnceLock;

use crate::rounding::{next_down_f32, next_up_f32};
use ny_core::{
    f32_affine_eval_error, f64_to_f32_down, f64_to_f32_up, nan_propagating_max, nan_propagating_min,
};

static SILU_CRITICAL_POINT: OnceLock<f32> = OnceLock::new();
static SILU_INFLECTION_POINTS: OnceLock<(f32, f32)> = OnceLock::new();

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Evaluate SiLU (Swish) activation at a point: x * sigmoid(x).
#[inline]
pub fn silu_eval(x: f32) -> f32 {
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    x * sigmoid(x)
}

#[inline]
pub(crate) fn silu_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

#[inline]
pub(crate) fn silu_second_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    let s1 = 1.0 - s;
    s * s1 * (2.0 + x * (1.0 - 2.0 * s))
}

#[inline]
pub(crate) fn silu_critical_point() -> f32 {
    *SILU_CRITICAL_POINT.get_or_init(|| {
        let mut x = -1.28_f32;
        let eps = 1.0e-5_f32;
        for _ in 0..20 {
            let d = silu_derivative(x);
            let dd = (silu_derivative(x + eps) - silu_derivative(x - eps)) / (2.0 * eps);
            if dd.abs() < 1.0e-10 {
                break;
            }
            x -= d / dd;
        }
        x
    })
}

pub(crate) fn silu_inflection_points() -> (f32, f32) {
    *SILU_INFLECTION_POINTS.get_or_init(|| {
        let f = |x: f32| -> f32 {
            let s = sigmoid(x);
            2.0 + x * (1.0 - 2.0 * s)
        };
        let f_deriv = |x: f32| -> f32 {
            let s = sigmoid(x);
            (1.0 - 2.0 * s) - 2.0 * x * s * (1.0 - s)
        };

        let mut x_left = -2.4_f32;
        for _ in 0..30 {
            let fd = f_deriv(x_left);
            if fd.abs() < 1.0e-12 {
                break;
            }
            let step = f(x_left) / fd;
            x_left -= step;
            if step.abs() < 1.0e-8 {
                break;
            }
        }

        let mut x_right = 2.4_f32;
        for _ in 0..30 {
            let fd = f_deriv(x_right);
            if fd.abs() < 1.0e-12 {
                break;
            }
            let step = f(x_right) / fd;
            x_right -= step;
            if step.abs() < 1.0e-8 {
                break;
            }
        }

        (x_left, x_right)
    })
}

pub(crate) fn silu_min_max(l: f32, u: f32) -> (f32, f32) {
    let fl = silu_eval_f64(l as f64) as f32;
    let fu = silu_eval_f64(u as f64) as f32;

    let mut min_val = nan_propagating_min(next_down_f32(fl), next_down_f32(fu));
    let mut max_val = nan_propagating_max(next_up_f32(fl), next_up_f32(fu));

    let critical = silu_critical_point();
    if l < critical && critical < u {
        let f_min = silu_eval_f64(critical as f64) as f32;
        min_val = nan_propagating_min(min_val, next_down_f32(f_min));
        max_val = nan_propagating_max(max_val, next_up_f32(f_min));
    }

    (min_val, max_val)
}

#[inline]
pub(crate) fn silu_eval_f64(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let sigmoid = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    };
    x * sigmoid
}

#[inline]
pub(crate) fn silu_chord(l: f32, u: f32) -> (f32, f32, f32) {
    let l64 = l as f64;
    let u64 = u as f64;
    let fl64 = silu_eval_f64(l64);
    let fu64 = silu_eval_f64(u64);
    let slope64 = (fu64 - fl64) / (u64 - l64);
    let intercept64 = fl64 - slope64 * l64;

    let slope_f32 = slope64 as f32;
    let max_abs_x = l.abs().max(u.abs());
    let eval_err = f32_affine_eval_error(slope64, slope_f32, intercept64, max_abs_x);
    (
        slope_f32,
        next_down_f32(f64_to_f32_down(intercept64 - eval_err)),
        next_up_f32(f64_to_f32_up(intercept64 + eval_err)),
    )
}

#[inline]
pub(crate) fn silu_tangent(d: f32, max_abs_x: f32) -> (f32, f32, f32) {
    let d64 = d as f64;
    let slope64 = silu_derivative_f64(d64);
    let intercept64 = silu_eval_f64(d64) - slope64 * d64;

    let slope_f32 = slope64 as f32;
    let eval_err = f32_affine_eval_error(slope64, slope_f32, intercept64, max_abs_x);
    (
        slope_f32,
        next_down_f32(f64_to_f32_down(intercept64 - eval_err)),
        next_up_f32(f64_to_f32_up(intercept64 + eval_err)),
    )
}

#[inline]
pub(crate) fn silu_tangent_raw(d: f32) -> (f32, f32) {
    let d64 = d as f64;
    let slope64 = silu_derivative_f64(d64);
    let intercept64 = silu_eval_f64(d64) - slope64 * d64;
    (slope64 as f32, intercept64 as f32)
}

#[inline]
fn silu_derivative_f64(x: f64) -> f64 {
    let s = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    };
    s * (1.0 + x * (1.0 - s))
}
