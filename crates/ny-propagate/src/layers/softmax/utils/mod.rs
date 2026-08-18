// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2};
use ny_core::{f64_to_f32_down, f64_to_f32_up, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32};

use crate::bounds::nan_propagating_max;
use crate::LinearBounds;

/// Small epsilon for numerical stability.
pub(super) const SOFTMAX_EPSILON: f32 = 1e-12;
pub(super) const SOFTMAX_SANITIZE_MARGIN: f32 = 1e-6;

// =============================================================================
// Standalone functions for Kani verification
// =============================================================================

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
    let exp_lower = if lower == f32::NEG_INFINITY {
        0.0
    } else if lower == f32::INFINITY {
        f32::INFINITY
    } else {
        let reference = (lower as f64).exp();
        if reference > f32::MAX as f64 {
            // Mathematical exp(finite) is finite.  +inf cannot be a lower
            // endpoint, while MAX is representable and remains below it.
            f32::MAX
        } else {
            // Classify the binary32-subnormal range in f64 before conversion.
            // A direct `as f32` may flush to zero under FTZ/DAZ; stepping once
            // from that zero is still below exp(-100).
            next_down_f32(f64_to_f32_down(reference)).max(0.0)
        }
    };
    let exp_upper = if upper == f32::NEG_INFINITY {
        0.0
    } else {
        next_up_f32(f64_to_f32_up((upper as f64).exp()))
    };
    Ok((exp_lower, exp_upper))
}

/// Compute softmax_i(x) for x in [lower, upper] using IBP bounds.
///
/// The Auto-LiRPA softmax interval formula (using pre-computed exp values):
/// - Lower bound: exp_lower_i / (sum_exp_upper - exp_upper_i + exp_lower_i)
/// - Upper bound: exp_upper_i / (sum_exp_lower - exp_lower_i + exp_upper_i)
///
/// # Arguments
/// * `exp_lower_i` - exp(lower_i - max_upper), the shifted exponential of lower bound
/// * `exp_upper_i` - exp(upper_i - max_upper), the shifted exponential of upper bound
/// * `sum_exp_lower` - sum of all exp(lower_j - max_upper) over j
/// * `sum_exp_upper` - sum of all exp(upper_j - max_upper) over j
///
/// # Returns
/// (softmax_lower_i, softmax_upper_i) for the i-th element, clamped to [0, 1].
///
/// # Note
/// The actual IBP implementation in ibp.rs uses `sanitize_softmax_unit_bounds` which
/// applies a small margin expansion for conservativeness. This function provides the
/// raw clamped bounds suitable for Kani verification of the core formula.
#[inline]
pub fn softmax_ibp_element_bounds(
    exp_lower_i: f32,
    exp_upper_i: f32,
    sum_exp_lower: f32,
    sum_exp_upper: f32,
) -> (f32, f32) {
    // Per-coordinate monotone optimum (denominators are exact, no additive epsilon).
    //
    // SOUNDNESS (#4231): a fixed `+ SOFTMAX_EPSILON` in the denominator UNDER-
    // approximates p_hi for a reachable key when the legitimate terms are sub-1e-12
    // (the underflow / large-score-gap regime): eps swamps them and the ratio
    // collapses to ~0, producing a FALSE certificate. The denominator's dominant
    // term already keeps it positive whenever the matching numerator is positive, so
    // no epsilon is needed; an exactly-zero denominator is widened outward to the
    // conservative endpoint by the guards below.
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

// =============================================================================
// Internal array-based helpers (used by layers)
// =============================================================================

pub(super) fn logsumexp_1d(values: &Array1<f32>) -> f32 {
    let max_val = values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, nan_propagating_max);
    if !max_val.is_finite() {
        return max_val;
    }
    // f64 accumulation for sum of exponentials. Part of #2423.
    let max_f64 = max_val as f64;
    let sum_exp: f64 = values.iter().map(|&v| (v as f64 - max_f64).exp()).sum();
    (max_f64 + sum_exp.ln()) as f32
}

pub(super) fn softmax_1d(values: &Array1<f32>) -> Array1<f32> {
    let max_val = values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, nan_propagating_max);
    // f64 accumulation for exp and sum to prevent precision loss. Part of #2423.
    let max_f64 = max_val as f64;
    let exp_vals_f64: Array1<f64> = values.mapv(|v| (v as f64 - max_f64).exp());
    let sum_exp: f64 = exp_vals_f64.sum();
    exp_vals_f64.mapv(|v| (v / sum_exp) as f32)
}

pub(super) fn sanitize_softmax_unit_bounds(mut lower: f32, mut upper: f32) -> (f32, f32) {
    // Softmax outputs are always in [0, 1]. When we detect numerical issues, prefer a
    // conservative widening rather than propagating NaN/inf.
    if !lower.is_finite() || !(0.0..=1.0).contains(&lower) {
        lower = 0.0;
    } else {
        lower = (lower - SOFTMAX_SANITIZE_MARGIN).max(0.0);
    }

    if !upper.is_finite() || upper < 0.0 {
        upper = 1.0;
    } else {
        upper = (upper + SOFTMAX_SANITIZE_MARGIN).min(1.0);
    }

    if lower > upper {
        (0.0, 1.0)
    } else {
        (lower, upper)
    }
}

// =============================================================================
// CROWN affine backward composition (shared by softmax + logsoftmax)
// =============================================================================

/// Compose upstream CROWN linear bounds with per-neuron affine relaxations.
///
/// Given upstream bounds `bounds` (output <- neuron) and per-neuron affine
/// relaxations `lower_a/b`, `upper_a/b` (neuron <- input), produces the
/// composed bounds (output <- input) using the standard CROWN backward rule:
///
///   new_lower[out, k] = sum_i { la_i * (la_i > 0 ? lower_a : upper_a)[i, k] }
///   new_upper[out, k] = sum_i { ua_i * (ua_i > 0 ? upper_a : lower_a)[i, k] }
///
/// Accumulation uses f64 to prevent catastrophic cancellation in
/// softmax/logsoftmax Jacobians, which have mixed-sign terms that sum near
/// zero (#1745, #2169). Bias conversion uses directed rounding: `next_down_f32`
/// for lower, `next_up_f32` for upper. A-matrix uses nearest rounding (#2208).
///
/// Previously duplicated in `logsoftmax/crown.rs` and `linear/sound.rs`.
/// Consolidated per #2528.
pub(super) fn apply_affine_bounds_f64(
    bounds: &LinearBounds,
    lower_a: &Array2<f32>,
    lower_b: &Array1<f32>,
    upper_a: &Array2<f32>,
    upper_b: &Array1<f32>,
) -> Result<LinearBounds> {
    let num_outputs = bounds.num_outputs();
    let num_neurons = lower_b.len();

    let mut new_lower_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut new_lower_b_f64 = bounds.lower_b().mapv(|x| x as f64);
    let mut new_upper_a_f64 = Array2::<f64>::zeros((num_outputs, num_neurons));
    let mut new_upper_b_f64 = bounds.upper_b().mapv(|x| x as f64);

    for out_idx in 0..num_outputs {
        for i in 0..num_neurons {
            let la = bounds.lower_a()[[out_idx, i]];
            let ua = bounds.upper_a()[[out_idx, i]];

            // Guard: skip zero coefficients to avoid 0*inf NaN (#1739).
            if la > 0.0 {
                let la_f64 = la as f64;
                for k in 0..num_neurons {
                    new_lower_a_f64[[out_idx, k]] += la_f64 * lower_a[[i, k]] as f64;
                }
                new_lower_b_f64[out_idx] += la_f64 * lower_b[i] as f64;
            } else if la < 0.0 {
                let la_f64 = la as f64;
                for k in 0..num_neurons {
                    new_lower_a_f64[[out_idx, k]] += la_f64 * upper_a[[i, k]] as f64;
                }
                new_lower_b_f64[out_idx] += la_f64 * upper_b[i] as f64;
            }

            if ua > 0.0 {
                let ua_f64 = ua as f64;
                for k in 0..num_neurons {
                    new_upper_a_f64[[out_idx, k]] += ua_f64 * upper_a[[i, k]] as f64;
                }
                new_upper_b_f64[out_idx] += ua_f64 * upper_b[i] as f64;
            } else if ua < 0.0 {
                let ua_f64 = ua as f64;
                for k in 0..num_neurons {
                    new_upper_a_f64[[out_idx, k]] += ua_f64 * lower_a[[i, k]] as f64;
                }
                new_upper_b_f64[out_idx] += ua_f64 * lower_b[i] as f64;
            }
        }
    }

    // A-matrix: standard f64->f32 rounding (round-to-nearest), matching
    // alpha-beta-CROWN. Directed rounding on A is not unconditionally sound (#2208).
    LinearBounds::new_or_conservative(
        new_lower_a_f64.mapv(|x| x as f32),
        new_lower_b_f64.mapv(|x| next_down_f32(x as f32)),
        new_upper_a_f64.mapv(|x| x as f32),
        new_upper_b_f64.mapv(|x| next_up_f32(x as f32)),
    )
}

#[cfg(test)]
mod tests;
