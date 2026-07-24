// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Softmax/logsoftmax scalar interval helpers for Kani proofs.
//!
//! These are standalone scalar (or &[f32]) functions used by the Kani proof
//! harnesses. They do not depend on ndarray or any heavy tensor library.

use ny_core::{nan_propagating_max, NyError, Result};

/// Evaluate exp(x) with bounds on [lower, upper].
///
/// For Kani verification: exp is monotonically increasing, so:
/// - exp(lower) is the lower bound of exp(x) for x in [lower, upper]
/// - exp(upper) is the upper bound of exp(x) for x in [lower, upper]
///
/// Returns (exp_lower, exp_upper) where exp_lower <= exp(x) <= exp_upper for all x in [lower, upper].
#[inline]
pub fn exp_interval_bounds(lower: f32, upper: f32) -> Result<(f32, f32)> {
    if lower > upper {
        return Err(NyError::NumericalInstability(format!(
            "exp_interval_bounds received inverted interval: lower ({lower}) > upper ({upper})"
        )));
    }
    // exp is monotonically increasing, so bounds are simply exp at endpoints
    Ok((lower.exp(), upper.exp()))
}

/// Compute softmax_i(x) for x in [lower, upper] using IBP bounds.
///
/// The Auto-LiRPA softmax interval formula (using pre-computed exp values):
/// - Lower bound: exp_lower_i / (sum_exp_upper - exp_upper_i + exp_lower_i)
/// - Upper bound: exp_upper_i / (sum_exp_lower - exp_lower_i + exp_upper_i)
#[inline]
pub fn softmax_ibp_element_bounds(
    exp_lower_i: f32,
    exp_upper_i: f32,
    sum_exp_lower: f32,
    sum_exp_upper: f32,
) -> (f32, f32) {
    // Per-coordinate monotone optimum (denominators exact, no additive epsilon).
    // SOUNDNESS (#4231): a fixed `+ SOFTMAX_EPSILON` UNDER-approximates p_hi for a
    // reachable key when the legitimate denominator terms are sub-1e-12 (the underflow
    // / large-score-gap regime) — eps swamps them and the ratio collapses to ~0, a
    // FALSE certificate. The dominant denominator term keeps it positive whenever the
    // numerator is positive; an exactly-zero denominator widens to the conservative
    // endpoint via the guards below.
    let denom_for_lower = sum_exp_upper - exp_upper_i + exp_lower_i;
    let denom_for_upper = sum_exp_lower - exp_lower_i + exp_upper_i;

    let raw_lower = if denom_for_lower > 0.0 && denom_for_lower.is_finite() {
        exp_lower_i / denom_for_lower
    } else {
        0.0
    };

    let raw_upper = if denom_for_upper > 0.0 && denom_for_upper.is_finite() {
        exp_upper_i / denom_for_upper
    } else {
        1.0
    };

    // f32::clamp absorbs NaN to the lower endpoint (0.0), which can create a
    // falsely-tight [0, 0] interval when both raw bounds are NaN. Keep NaN
    // handling explicit so non-finite numerators widen to [0, 1].
    let lower = if raw_lower.is_nan() {
        0.0
    } else {
        raw_lower.clamp(0.0, 1.0)
    };
    let upper = if raw_upper.is_nan() {
        1.0
    } else {
        raw_upper.clamp(0.0, 1.0)
    };

    if lower > upper {
        (0.0, 1.0)
    } else {
        (lower, upper)
    }
}

/// Compute logsumexp of a slice of values.
///
/// logsumexp(x) = max(x) + ln(sum(exp(x - max(x))))
///
/// This is the numerically stable formulation.
#[inline]
pub fn logsumexp_slice(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NEG_INFINITY;
    }

    // NaN-propagating fold: if any value is NaN, max is NaN — see #2577.
    let max_val = values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, nan_propagating_max);
    if !max_val.is_finite() {
        return max_val;
    }

    // f64 accumulation for sum of exponentials to prevent precision loss
    // for large slices (seq_len=512+). Part of #2423.
    let max_f64 = max_val as f64;
    let sum_exp: f64 = values.iter().map(|&v| (v as f64 - max_f64).exp()).sum();
    (max_f64 + sum_exp.ln()) as f32
}

/// Compute logsoftmax_i(x) = x_i - logsumexp(x).
///
/// For sound IBP bounds on logsoftmax:
/// - Lower bound: lower_i - logsumexp(upper)
/// - Upper bound: upper_i - logsumexp(lower)
#[inline]
pub fn logsoftmax_ibp_bounds(
    lower_i: f32,
    upper_i: f32,
    lse_lower: f32,
    lse_upper: f32,
) -> (f32, f32) {
    // logsoftmax_i = x_i - logsumexp(x)
    // Lower: x_i^L - logsumexp(x^U)
    // Upper: x_i^U - logsumexp(x^L)
    (lower_i - lse_upper, upper_i - lse_lower)
}
