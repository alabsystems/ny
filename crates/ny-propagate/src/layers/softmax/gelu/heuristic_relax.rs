// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Heuristic GELU linear relaxation modes: Chord, Tangent, TwoSlope, Adaptive.

use super::eval::{
    gelu_critical_point, gelu_derivative_erf_f64, gelu_derivative_tanh_f64, gelu_erf_f64,
    gelu_eval, gelu_infinite_bounds_relaxation, gelu_tanh_f64,
};
use super::GeluApproximation;
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use ny_tensor::{next_down_f32, next_up_f32};

/// Evaluate GELU in f64 for the given approximation mode.
/// Used to avoid catastrophic cancellation in chord slope computations.
#[inline]
fn gelu_eval_f64(x: f64, approximation: GeluApproximation) -> f64 {
    match approximation {
        GeluApproximation::Erf => gelu_erf_f64(x),
        GeluApproximation::Tanh => gelu_tanh_f64(x),
    }
}

/// Evaluate GELU derivative in f64 for the given approximation mode.
#[inline]
fn gelu_derivative_f64(x: f64, approximation: GeluApproximation) -> f64 {
    match approximation {
        GeluApproximation::Erf => gelu_derivative_erf_f64(x),
        GeluApproximation::Tanh => gelu_derivative_tanh_f64(x),
    }
}

/// Relaxation mode for activation functions.
///
/// Different modes provide different trade-offs between tightness and computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RelaxationMode {
    /// Chord-based relaxation: connect endpoints, shift for soundness.
    /// Uses same slope for both lower and upper bounds.
    /// Fast and always sound, but may be loose for asymmetric regions.
    #[default]
    Chord,

    /// Tangent-based relaxation: use tangent line at center point.
    /// Optimal for small intervals where Taylor expansion is accurate.
    /// Better than chord when interval is small relative to curvature.
    Tangent,

    /// Two-slope relaxation: independent optimal slopes for lower/upper.
    /// Uses tangent lines at strategic points for each bound.
    /// Tighter than chord for most cases but requires more computation.
    TwoSlope,

    /// Adaptive selection: automatically choose the tightest relaxation.
    /// Evaluates multiple strategies and returns the one with smallest
    /// bound width (upper_intercept - lower_intercept) at the interval center.
    Adaptive,
}

/// Compute linear relaxation parameters for GELU on interval [l, u].
///
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept) where:
/// - GELU(x) >= lower_slope * x + lower_intercept
/// - GELU(x) <= upper_slope * x + upper_intercept
///
/// Uses a chord-based relaxation with sampling-based margins; this is heuristic,
/// not a proof of global soundness. GELU has complex convexity patterns, so we
/// empirically verify bounds.
pub fn gelu_linear_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
) -> (f32, f32, f32, f32) {
    // Handle infinite/NaN bounds: identity relaxation is UNSOUND (see #1837).
    // GELU(x) = x·Φ(x), so GELU(x) ≥ x fails for 0 < x < +∞ where Φ(x) < 1.
    if let Some(result) = gelu_infinite_bounds_relaxation(l, u, approximation) {
        return result;
    }

    // Maximum absolute value for directed rounding slope_err pattern (#3329).
    let max_abs_x = l.abs().max(u.abs());

    // Handle degenerate cases
    if (u - l).abs() < 1e-8 {
        // Point interval: f64 derivative + directed rounding (#3329).
        let slope_f64 = gelu_derivative_f64(l as f64, approximation);
        let slope = slope_f64 as f32;
        let slope_err = next_up_f32(((slope_f64 - slope as f64).abs() * max_abs_x as f64) as f32);
        let intercept_f64 = gelu_eval_f64(l as f64, approximation) - slope_f64 * l as f64;
        let intercept_f32 = intercept_f64 as f32;
        return (
            slope,
            next_down_f32(intercept_f32 - slope_err),
            slope,
            next_up_f32(intercept_f32 + slope_err),
        );
    }

    // Compute chord slope and intercept in f64 to avoid catastrophic cancellation
    // when u ≈ l (both numerator and denominator approach 0 in f32).
    // Same pattern as GELU-sound (#2488), exp/log (#1745), ELU/SELU (#1754).
    let gl64 = gelu_eval_f64(l as f64, approximation);
    let gu64 = gelu_eval_f64(u as f64, approximation);
    // Keep slope in f64 for intercept to avoid secondary precision loss.
    // Ref: GELU-sound (sound_relax.rs:55-58).
    let chord_slope64 = (gu64 - gl64) / (u as f64 - l as f64);
    let chord_intercept64 = gl64 - chord_slope64 * l as f64;
    let chord_slope = chord_slope64 as f32;
    // Directed rounding: absorb f64→f32 slope truncation error in intercepts (#3329).
    let chord_slope_err =
        next_up_f32(((chord_slope64 - chord_slope as f64).abs() * max_abs_x as f64) as f32);
    let chord_intercept = chord_intercept64 as f32;

    // Sample the interval to find max deviation from chord.
    // Scale sample count with interval width to maintain density (#3329).
    let width = u - l;
    let num_samples = 100usize.max(((width * 20.0) as usize).min(10_000));
    let mut max_above_chord = 0.0_f32; // max(GELU(x) - chord(x))
    let mut max_below_chord = 0.0_f32; // max(chord(x) - GELU(x))

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let gx = gelu_eval(x, approximation);
        let cx = chord_slope * x + chord_intercept;
        let diff = gx - cx;

        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Also check critical point (minimum of GELU) if it's in the interval
    let critical_point = gelu_critical_point(approximation);
    if l <= critical_point && critical_point <= u {
        let gc = gelu_eval(critical_point, approximation);
        let cc = chord_slope * critical_point + chord_intercept;
        let diff = gc - cc;
        if diff > max_above_chord {
            max_above_chord = diff;
        }
        if -diff > max_below_chord {
            max_below_chord = -diff;
        }
    }

    // Analytical inter-sample error bound: between sample points spaced h apart,
    // max undetected deviation is |g''|_max * h² / 8 where |g''|_max ≤ 1.0 for GELU.
    // Ref: Taylor remainder theorem, GELU g''(x) = φ(x)(2 - x²), max ≈ 0.798 at x=0.
    let h = width / num_samples as f32;
    let sample_gap = h * h / 8.0;
    let eps = 1e-5 + sample_gap;

    // Lower bound: chord shifted down by max_below_chord (ensures chord - shift <= GELU)
    // Directed rounding: next_down + slope_err for sound lower bound (#3329).
    let lower_slope = chord_slope;
    let lower_intercept = next_down_f32(chord_intercept - max_below_chord - eps - chord_slope_err);

    // Upper bound: chord shifted up by max_above_chord (ensures GELU <= chord + shift)
    // Directed rounding: next_up + slope_err for sound upper bound (#3329).
    let upper_slope = chord_slope;
    let upper_intercept = next_up_f32(chord_intercept + max_above_chord + eps + chord_slope_err);

    (lower_slope, lower_intercept, upper_slope, upper_intercept)
}

/// Compute tangent-based relaxation for GELU.
///
/// Uses the tangent line at the center point (l+u)/2.
/// This is optimal for small intervals but may not be sound for large intervals
/// without proper error bounding.
fn gelu_tangent_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
) -> (f32, f32, f32, f32) {
    // Handle infinite/NaN bounds: identity relaxation is UNSOUND (see #1837).
    if let Some(result) = gelu_infinite_bounds_relaxation(l, u, approximation) {
        return result;
    }

    let max_abs_x = l.abs().max(u.abs());

    if (u - l).abs() < 1e-8 {
        // Point interval: f64 derivative + directed rounding (#3329).
        let slope_f64 = gelu_derivative_f64(l as f64, approximation);
        let slope = slope_f64 as f32;
        let slope_err = next_up_f32(((slope_f64 - slope as f64).abs() * max_abs_x as f64) as f32);
        let intercept_f64 = gelu_eval_f64(l as f64, approximation) - slope_f64 * l as f64;
        let intercept_f32 = intercept_f64 as f32;
        return (
            slope,
            next_down_f32(intercept_f32 - slope_err),
            slope,
            next_up_f32(intercept_f32 + slope_err),
        );
    }

    // Tangent line at center — f64 intermediate for precision (#3329).
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let c = (l + u) / 2.0;
    let c_f64 = c as f64;
    let slope_f64 = gelu_derivative_f64(c_f64, approximation);
    let slope = slope_f64 as f32;
    let slope_err = next_up_f32(((slope_f64 - slope as f64).abs() * max_abs_x as f64) as f32);
    let intercept_f64 = gelu_eval_f64(c_f64, approximation) - slope_f64 * c_f64;
    let intercept = intercept_f64 as f32;

    // Find max deviation from tangent line (above and below).
    // Scale sample count with interval width to maintain density (#3329).
    let width = u - l;
    let num_samples = 50usize.max(((width * 20.0) as usize).min(10_000));
    let mut max_above = 0.0_f32;
    let mut max_below = 0.0_f32;

    for i in 0..=num_samples {
        let t = i as f32 / num_samples as f32;
        let x = l + (u - l) * t;
        let gx = gelu_eval(x, approximation);
        let tx = slope * x + intercept;
        let diff = gx - tx;

        if diff > max_above {
            max_above = diff;
        }
        if -diff > max_below {
            max_below = -diff;
        }
    }

    // Also check critical point
    let critical_point = gelu_critical_point(approximation);
    if l <= critical_point && critical_point <= u {
        let gc_crit = gelu_eval(critical_point, approximation);
        let tc_crit = slope * critical_point + intercept;
        let diff = gc_crit - tc_crit;
        if diff > max_above {
            max_above = diff;
        }
        if -diff > max_below {
            max_below = -diff;
        }
    }

    // Analytical inter-sample error bound (#3329): |g''|_max * h² / 8, |g''|_max ≤ 1.0.
    let h = width / num_samples as f32;
    let sample_gap = h * h / 8.0;
    let eps = 1e-5 + sample_gap;
    // Directed rounding + slope_err for sound intercepts (#3329).
    let lower_intercept = next_down_f32(intercept - max_below - eps - slope_err);
    let upper_intercept = next_up_f32(intercept + max_above + eps + slope_err);

    (slope, lower_intercept, slope, upper_intercept)
}

/// Compute two-slope relaxation for GELU.
///
/// Uses independent optimal slopes for lower and upper bounds.
/// For the lower bound: tangent at a point that minimizes underestimation.
/// For the upper bound: tangent at a point that minimizes overestimation.
fn gelu_two_slope_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
) -> (f32, f32, f32, f32) {
    // Handle infinite/NaN bounds: identity relaxation is UNSOUND (see #1837).
    if let Some(result) = gelu_infinite_bounds_relaxation(l, u, approximation) {
        return result;
    }

    let max_abs_x = l.abs().max(u.abs());

    if (u - l).abs() < 1e-8 {
        // Point interval: f64 derivative + directed rounding (#3329).
        let slope_f64 = gelu_derivative_f64(l as f64, approximation);
        let slope = slope_f64 as f32;
        let slope_err = next_up_f32(((slope_f64 - slope as f64).abs() * max_abs_x as f64) as f32);
        let intercept_f64 = gelu_eval_f64(l as f64, approximation) - slope_f64 * l as f64;
        let intercept_f32 = intercept_f64 as f32;
        return (
            slope,
            next_down_f32(intercept_f32 - slope_err),
            slope,
            next_up_f32(intercept_f32 + slope_err),
        );
    }

    // Compute f64 endpoint evaluations for chord slope precision.
    // Same pattern as GELU-sound (#2488), exp/log (#1745), ELU/SELU (#1754).
    let gl64 = gelu_eval_f64(l as f64, approximation);
    let gu64 = gelu_eval_f64(u as f64, approximation);

    // Strategy: Find the tightest lower bound line that stays below GELU
    // and tightest upper bound line that stays above GELU.

    // For lower bound: try chord and tangents at l, u, center
    // Pick the one with highest minimum value over [l, u]
    // Bit-identical tangent anchors: f32::midpoint rounds differently at overflow edges.
    #[allow(clippy::manual_midpoint)]
    let candidates = [l, u, (l + u) / 2.0];
    // Scale sample count with interval width to maintain density (#3329).
    let width = u - l;
    let num_samples = 30usize.max(((width * 20.0) as usize).min(10_000));
    // Analytical inter-sample error bound: |g''|_max * h² / 8, |g''|_max ≤ 1.0.
    let h = width / num_samples as f32;
    let sample_gap = h * h / 8.0;
    let eps = 1e-5 + sample_gap;

    // Lower bound: line must be <= GELU(x) for all x in [l, u]
    let mut best_lower_slope = 0.0_f32;
    let mut best_lower_intercept = f32::NEG_INFINITY;

    for &point in &candidates {
        // f64 tangent with directed rounding (#3329).
        let p_f64 = point as f64;
        let slope_f64 = gelu_derivative_f64(p_f64, approximation);
        let slope = slope_f64 as f32;
        let slope_err = next_up_f32(((slope_f64 - slope as f64).abs() * max_abs_x as f64) as f32);
        let intercept_f64 = gelu_eval_f64(p_f64, approximation) - slope_f64 * p_f64;
        let intercept = intercept_f64 as f32;

        // Find min margin (GELU(x) - line(x)) over interval
        let mut min_margin = f32::INFINITY;
        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let gx = gelu_eval(x, approximation);
            let lx = slope * x + intercept;
            min_margin = nan_propagating_min(min_margin, gx - lx);
        }

        // Shift intercept down by min_margin to ensure soundness.
        // Directed rounding: next_down + slope_err (#3329).
        let adjusted_intercept = next_down_f32(intercept + min_margin - eps - slope_err);

        // We want the highest lower bound (closest to function)
        // Evaluate at center to compare
        let eval_center = slope * (l + u) / 2.0 + adjusted_intercept;
        let current_best = best_lower_slope * (l + u) / 2.0 + best_lower_intercept;

        if eval_center > current_best {
            best_lower_slope = slope;
            best_lower_intercept = adjusted_intercept;
        }
    }

    // Also try chord for lower bound (f64 chord slope to avoid cancellation)
    {
        let slope64 = (gu64 - gl64) / (u as f64 - l as f64);
        let chord_slope = slope64 as f32;
        let chord_slope_err =
            next_up_f32(((slope64 - chord_slope as f64).abs() * max_abs_x as f64) as f32);
        let intercept = (gl64 - slope64 * l as f64) as f32;

        let mut min_margin = f32::INFINITY;
        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let gx = gelu_eval(x, approximation);
            let lx = chord_slope * x + intercept;
            min_margin = nan_propagating_min(min_margin, gx - lx);
        }

        let adjusted_intercept = next_down_f32(intercept + min_margin - eps - chord_slope_err);
        let eval_center = chord_slope * (l + u) / 2.0 + adjusted_intercept;
        let current_best = best_lower_slope * (l + u) / 2.0 + best_lower_intercept;

        if eval_center > current_best {
            best_lower_slope = chord_slope;
            best_lower_intercept = adjusted_intercept;
        }
    }

    // Upper bound: line must be >= GELU(x) for all x in [l, u]
    let mut best_upper_slope = 0.0_f32;
    let mut best_upper_intercept = f32::INFINITY;

    for &point in &candidates {
        // f64 tangent with directed rounding (#3329).
        let p_f64 = point as f64;
        let slope_f64 = gelu_derivative_f64(p_f64, approximation);
        let slope = slope_f64 as f32;
        let slope_err = next_up_f32(((slope_f64 - slope as f64).abs() * max_abs_x as f64) as f32);
        let intercept_f64 = gelu_eval_f64(p_f64, approximation) - slope_f64 * p_f64;
        let intercept = intercept_f64 as f32;

        // Find max(GELU(x) - line(x)) over interval: positive means line is below GELU.
        let mut max_margin = f32::NEG_INFINITY;
        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let gx = gelu_eval(x, approximation);
            let lx = slope * x + intercept;
            max_margin = nan_propagating_max(max_margin, gx - lx);
        }

        // Shift intercept up to ensure soundness.
        // Directed rounding: next_up + slope_err (#3329).
        let adjusted_intercept = next_up_f32(intercept + max_margin + eps + slope_err);

        // We want the lowest upper bound (closest to function)
        let eval_center = slope * (l + u) / 2.0 + adjusted_intercept;
        let current_best = best_upper_slope * (l + u) / 2.0 + best_upper_intercept;

        if eval_center < current_best {
            best_upper_slope = slope;
            best_upper_intercept = adjusted_intercept;
        }
    }

    // Also try chord for upper bound (f64 chord slope to avoid cancellation)
    {
        let slope64 = (gu64 - gl64) / (u as f64 - l as f64);
        let chord_slope = slope64 as f32;
        let chord_slope_err =
            next_up_f32(((slope64 - chord_slope as f64).abs() * max_abs_x as f64) as f32);
        let intercept = (gl64 - slope64 * l as f64) as f32;

        let mut max_margin = f32::NEG_INFINITY;
        for i in 0..=num_samples {
            let t = i as f32 / num_samples as f32;
            let x = l + (u - l) * t;
            let gx = gelu_eval(x, approximation);
            let lx = chord_slope * x + intercept;
            max_margin = nan_propagating_max(max_margin, gx - lx);
        }

        let adjusted_intercept = next_up_f32(intercept + max_margin + eps + chord_slope_err);
        let eval_center = chord_slope * (l + u) / 2.0 + adjusted_intercept;
        let current_best = best_upper_slope * (l + u) / 2.0 + best_upper_intercept;

        if eval_center < current_best {
            best_upper_slope = chord_slope;
            best_upper_intercept = adjusted_intercept;
        }
    }

    (
        best_lower_slope,
        best_lower_intercept,
        best_upper_slope,
        best_upper_intercept,
    )
}

/// Compute adaptive linear relaxation for GELU.
///
/// Tries multiple relaxation strategies and returns the tightest one
/// (smallest bound width at the interval center).
pub fn adaptive_gelu_linear_relaxation(
    l: f32,
    u: f32,
    approximation: GeluApproximation,
    mode: RelaxationMode,
) -> (f32, f32, f32, f32) {
    match mode {
        RelaxationMode::Chord => gelu_linear_relaxation(l, u, approximation),
        RelaxationMode::Tangent => gelu_tangent_relaxation(l, u, approximation),
        RelaxationMode::TwoSlope => gelu_two_slope_relaxation(l, u, approximation),
        RelaxationMode::Adaptive => {
            // Try all strategies and pick the tightest
            let chord = gelu_linear_relaxation(l, u, approximation);
            let tangent = gelu_tangent_relaxation(l, u, approximation);
            let two_slope = gelu_two_slope_relaxation(l, u, approximation);

            // Measure width at center point
            // Bit-identical tightness probe: f32::midpoint rounds differently at overflow edges.
            #[allow(clippy::manual_midpoint)]
            let c = (l + u) / 2.0;

            fn bound_width(relaxation: &(f32, f32, f32, f32), x: f32) -> f32 {
                let (ls, li, us, ui) = *relaxation;
                (us * x + ui) - (ls * x + li)
            }

            let chord_width = bound_width(&chord, c);
            let tangent_width = bound_width(&tangent, c);
            let two_slope_width = bound_width(&two_slope, c);

            // Return the tightest (smallest positive width)
            if chord_width <= tangent_width && chord_width <= two_slope_width {
                chord
            } else if tangent_width <= two_slope_width {
                tangent
            } else {
                two_slope
            }
        }
    }
}

#[cfg(test)]
#[path = "heuristic_relax_tests.rs"]
mod tests;
