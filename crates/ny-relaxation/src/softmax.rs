// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Softmax/logsoftmax scalar interval helpers for Kani proofs.
//!
//! These are standalone scalar (or &[f32]) functions used by the Kani proof
//! harnesses. They do not depend on ndarray or any heavy tensor library.

use crate::rounding::{next_down_f32, next_up_f32};
use ny_core::{f64_to_f32_down, f64_to_f32_up, nan_propagating_max, NyError, Result};

/// Round `exp(x)` down to an `f32` endpoint without turning finite overflow
/// into an invalid `+inf` lower bound.
#[inline]
fn exp_lower_endpoint(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        return 0.0;
    }
    if x == f32::INFINITY {
        return f32::INFINITY;
    }

    let reference = (x as f64).exp();
    if reference > f32::MAX as f64 {
        // `x` is finite, so mathematical exp(x) is finite even when it is
        // larger than binary32 can represent.  MAX is therefore a valid
        // representable lower endpoint; +inf would not be.
        f32::MAX
    } else {
        // Classify the binary32-subnormal range in f64 before conversion.
        // A direct `as f32` may flush to zero under FTZ/DAZ; stepping once
        // from that zero is still below exp(-100).
        next_down_f32(f64_to_f32_down(reference)).max(0.0)
    }
}

/// Round `exp(x)` up to an `f32` endpoint.
#[inline]
fn exp_upper_endpoint(x: f32) -> f32 {
    if x == f32::NEG_INFINITY {
        0.0
    } else {
        next_up_f32(f64_to_f32_up((x as f64).exp()))
    }
}

/// Evaluate exp(x) with bounds on [lower, upper].
///
/// For Kani verification: exp is monotonically increasing, so:
/// - exp(lower) is the lower bound of exp(x) for x in [lower, upper]
/// - exp(upper) is the upper bound of exp(x) for x in [lower, upper]
///
/// Returns (exp_lower, exp_upper) where exp_lower <= exp(x) <= exp_upper for all x in [lower, upper].
#[inline]
pub fn exp_interval_bounds(lower: f32, upper: f32) -> Result<(f32, f32)> {
    if lower.is_nan() || upper.is_nan() || lower > upper {
        return Err(NyError::NumericalInstability(format!(
            "exp_interval_bounds received invalid interval: lower ({lower}), upper ({upper})"
        )));
    }
    // `f32::exp` is rounded to nearest, so using it verbatim for both
    // directions is not an enclosure even for a point interval.  Evaluate in
    // f64 and take one binary32 ULP outward at both endpoints.
    Ok((exp_lower_endpoint(lower), exp_upper_endpoint(upper)))
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

#[cfg(test)]
mod tests {
    use super::exp_interval_bounds;

    #[test]
    fn exp_interval_bounds_enclose_real_endpoint_rounding() {
        // At -100 the nearest binary32 exponential can land on either side
        // of the f64 reference as x moves by one binary32 ULP.  A point
        // interval therefore requires distinct outward endpoints.
        for x in [-100.0_f32, -1.0, 0.0, 1.0, 80.0, 89.0] {
            let (lower, upper) = exp_interval_bounds(x, x).expect("valid point interval");
            let reference = (x as f64).exp();
            assert!(
                (lower as f64) <= reference,
                "lower endpoint {lower:e} exceeds exp({x:e})={reference:e}"
            );
            assert!(
                reference <= upper as f64,
                "upper endpoint {upper:e} is below exp({x:e})={reference:e}"
            );
            assert!(lower >= 0.0);
            assert!(lower <= upper);
        }
    }

    #[test]
    fn exp_interval_bounds_handle_special_values_fail_closed() {
        assert!(exp_interval_bounds(f32::NAN, 0.0).is_err());
        assert!(exp_interval_bounds(0.0, f32::NAN).is_err());
        assert!(exp_interval_bounds(1.0, -1.0).is_err());
        assert_eq!(
            exp_interval_bounds(f32::NEG_INFINITY, f32::NEG_INFINITY)
                .expect("exp(-inf) is defined"),
            (0.0, 0.0)
        );
        assert_eq!(
            exp_interval_bounds(f32::INFINITY, f32::INFINITY).expect("exp(+inf) is defined"),
            (f32::INFINITY, f32::INFINITY)
        );

        let (lower, upper) =
            exp_interval_bounds(89.0, 89.0).expect("finite overflow point is valid");
        assert_eq!(lower, f32::MAX);
        assert_eq!(upper, f32::INFINITY);

        let (_, underflow_upper) =
            exp_interval_bounds(-100.0, -100.0).expect("finite underflow point is valid");
        assert!(
            underflow_upper >= f32::MIN_POSITIVE,
            "an FTZ-safe upper endpoint must never publish a positive subnormal: {underflow_upper:e}"
        );
        assert!((underflow_upper as f64) >= (-100.0_f64).exp());
    }
}
