// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Safe interval arithmetic for bound computation.

use crate::rounding::{next_down_f32, next_up_f32};

/// Safe multiplication for bound computation.
///
/// In interval arithmetic, a coefficient of 0 means no contribution,
/// so 0 * inf = 0 (not NaN). This prevents NaN propagation when
/// computing bounds with saturated (infinite) input bounds.
#[inline]
pub fn safe_mul_for_bounds(a: f32, x: f32) -> f32 {
    if a == 0.0 || x == 0.0 {
        0.0
    } else if a.is_nan() || x.is_nan() {
        f32::NAN
    } else {
        a * x
    }
}

/// Safe addition for bound computation that handles inf + (-inf) = conservative bound.
#[inline]
pub fn safe_add_for_bounds_with_polarity(sum: f32, term: f32, is_lower: bool) -> f32 {
    let result = sum + term;
    if result.is_nan()
        && sum.is_infinite()
        && term.is_infinite()
        && (sum.is_sign_positive() != term.is_sign_positive())
    {
        if is_lower {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else {
        result
    }
}

/// Safe addition when the polarity isn't known (defaults to upper bound = conservative).
#[cfg(any(test, feature = "kani-proofs"))]
#[inline]
pub fn safe_add_for_bounds(sum: f32, term: f32) -> f32 {
    safe_add_for_bounds_with_polarity(sum, term, false)
}

/// Safe multiplication preserving 0 * anything = 0 for interval arithmetic.
#[inline]
pub fn safe_mul_pair_for_bounds(a: f32, b: f32) -> f32 {
    if a == 0.0 || b == 0.0 {
        0.0
    } else {
        a * b
    }
}

/// Interval multiplication: compute [a_l, a_u] * [b_l, b_u].
#[inline]
pub fn interval_mul_for_bounds(a_l: f32, a_u: f32, b_l: f32, b_u: f32) -> (f32, f32) {
    if a_l.is_nan() || a_u.is_nan() || b_l.is_nan() || b_u.is_nan() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    let products = [
        safe_mul_pair_for_bounds(a_l, b_l),
        safe_mul_pair_for_bounds(a_l, b_u),
        safe_mul_pair_for_bounds(a_u, b_l),
        safe_mul_pair_for_bounds(a_u, b_u),
    ];

    if products.iter().all(|p| p.is_infinite()) {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    let mut lower = products[0];
    let mut upper = products[0];

    for &p in &products[1..] {
        if p < lower {
            lower = p;
        }
        if p > upper {
            upper = p;
        }
    }

    if lower.is_nan() || upper.is_nan() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    // Directed OUTWARD rounding on the final endpoints: each corner product is
    // rounded to nearest (up to 0.5 ULP inward), so min/max of the corners
    // alone is not a sound enclosure. One ULP outward covers it. Matches the
    // production hardening in `ny_propagate::bounds::safe_math`
    // (#concretize-soundness-hardening); next_*_f32 are no-ops on ±inf.
    (next_down_f32(lower), next_up_f32(upper))
}

/// Safe addition for lower bounds: NaN → -inf (sound lower).
#[inline]
pub fn safe_add_lower_for_bounds(a: f32, b: f32) -> f32 {
    let s = a + b;
    if s.is_nan() {
        f32::NEG_INFINITY
    } else {
        s
    }
}

/// Safe addition for upper bounds: NaN → +inf (sound upper).
#[inline]
pub fn safe_add_upper_for_bounds(a: f32, b: f32) -> f32 {
    let s = a + b;
    if s.is_nan() {
        f32::INFINITY
    } else {
        s
    }
}
