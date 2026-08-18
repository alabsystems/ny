// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{
    f64_to_f32_down, f64_to_f32_up, nan_propagating_max_f64, nan_propagating_min_f64, NyError,
    Result,
};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use super::LinearRelaxation;
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};
use ny_tensor::{next_down_f32, next_up_f32};

/// HardSwish layer: y = x * HardSigmoid(x)
///
/// Used in MobileNetV3 as a more efficient alternative to Swish (SiLU).
/// ONNX authors `alpha` as the FLOAT32 value nearest 1/6 and `beta` as 0.5:
/// y = x * max(0, min(1, alpha * x + beta)).
#[derive(Debug, Clone, Default)]
pub struct HardSwishLayer;

impl HardSwishLayer {
    /// The exact FLOAT32 coefficient authored by the ONNX HardSwish function.
    ///
    /// This dyadic value is slightly larger than the real number 1/6.
    pub const ALPHA: f32 = f32::from_bits(0x3e2a_aaab);

    /// The exact FLOAT32 offset authored by the ONNX HardSwish function.
    pub const BETA: f32 = 0.5;

    /// Create a new HardSwish layer.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate the exact-real ONNX HardSwish function, rounded once to f32.
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        // Guard against 0 * inf = NaN when x = -inf.
        // A direct formula would compute (-inf)*0 at the negative limit.
        // The exact limiting values are 0 at -inf and +inf at +inf.
        // Ref: SiLU guard pattern (silu.rs:100-105), fix for #1836.
        if !x.is_finite() {
            if x.is_nan() {
                return f32::NAN;
            }
            return if x.is_sign_negative() { 0.0 } else { x };
        }
        hardswish_eval_f64(f64::from(x)) as f32
    }
}

impl BoundPropagation for HardSwishLayer {
    /// IBP for the exact-real ONNX HardSwish function.
    ///
    /// Three regions:
    /// - y = 0 when alpha*x + beta <= 0
    /// - y = x * (alpha*x + beta) when 0 < alpha*x + beta < 1
    /// - y = x when alpha*x + beta >= 1
    ///
    /// The quadratic region has derivative `2*alpha*x + beta`.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        let mut lower_vals = input.lower().clone();
        let mut upper_vals = input.upper().clone();

        ndarray::Zip::from(&mut lower_vals)
            .and(&mut upper_vals)
            .and(input.lower())
            .and(input.upper())
            .for_each(|out_l, out_u, &in_l, &in_u| {
                if !in_l.is_finite() || !in_u.is_finite() {
                    // Guard NaN/Inf from unchecked callers: min/max can skip NaN
                    // and silently narrow bounds. Fall back to loose sound bounds.
                    *out_l = f32::NEG_INFINITY;
                    *out_u = f32::INFINITY;
                } else {
                    // Evaluate the exact-real function in f64. Nearest-rounding
                    // an endpoint to one f32 would make a point interval exclude
                    // the authored dyadic function whenever it lies between two
                    // adjacent f32 values.
                    let in_l64 = f64::from(in_l);
                    let in_u64 = f64::from(in_u);
                    let y_l = hardswish_eval_f64(in_l64);
                    let y_u = hardswish_eval_f64(in_u64);

                    // The quadratic region's unique minimum is at
                    // -beta/(2*alpha). Endpoints cover every maximum, including
                    // the constant/quadratic boundary at value zero.
                    let critical = hardswish_critical_f64();
                    let min_at_critical = if in_l64 <= critical && critical <= in_u64 {
                        hardswish_eval_f64(critical)
                    } else {
                        f64::INFINITY
                    };

                    *out_l = f64_to_f32_down(y_l.min(y_u).min(min_at_critical));
                    *out_u = f64_to_f32_up(y_l.max(y_u));
                }
            });

        BoundedTensor::new_allow_infinite(lower_vals, upper_vals)
    }

    /// CROWN backward propagation requires pre-activation bounds.
    /// Use `HardSwishLayer::propagate_linear_with_bounds` instead.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::InvalidSpec(
            "HardSwish CROWN propagation requires pre-activation bounds. \
             Use propagate_linear_with_bounds() instead."
                .to_string(),
        ))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        // Delegate to the inherent method.
        HardSwishLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Evaluate the exact-real HardSwish function and round once to f32.
///
/// Note: caller `hardswish_linear_relaxation` guards for inf/NaN at entry,
/// so this is only called with finite inputs. The guard here is defense-in-depth.
#[inline]
#[cfg(test)]
fn hardswish_eval(x: f32) -> f32 {
    // Guard against 0 * inf = NaN at x = -inf (defense-in-depth, #1836).
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    hardswish_eval_f64(f64::from(x)) as f32
}

/// Evaluate HardSwish in f64 to avoid catastrophic cancellation in chord slope.
/// HardSwish(x) = x * clamp(alpha*x + beta, 0, 1), where both
/// coefficients are promoted exactly from their authored f32 values.
/// Same pattern as silu_eval_f64 (silu/math.rs), exp_eval_f64 (#1745).
#[inline]
fn hardswish_eval_f64(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let alpha = f64::from(HardSwishLayer::ALPHA);
    let beta = f64::from(HardSwishLayer::BETA);
    x * (alpha * x + beta).clamp(0.0, 1.0)
}

#[inline]
fn hardswish_lower_kink_f64() -> f64 {
    -f64::from(HardSwishLayer::BETA) / f64::from(HardSwishLayer::ALPHA)
}

#[inline]
fn hardswish_upper_kink_f64() -> f64 {
    (1.0 - f64::from(HardSwishLayer::BETA)) / f64::from(HardSwishLayer::ALPHA)
}

#[inline]
fn hardswish_critical_f64() -> f64 {
    -f64::from(HardSwishLayer::BETA) / (2.0 * f64::from(HardSwishLayer::ALPHA))
}

/// Compute HardSwish chord slope and intercept using f64 intermediates.
/// Returns (slope, lower_intercept, upper_intercept) with directed rounding.
///
/// Prevents catastrophic cancellation when u ≈ l.
/// Directed rounding absorbs f64→f32 slope truncation error into the intercept:
/// - `lower_intercept`: guaranteed `slope*x + lower_intercept <= true_chord(x)` ∀x∈[l,u]
/// - `upper_intercept`: guaranteed `slope*x + upper_intercept >= true_chord(x)` ∀x∈[l,u]
///
/// Ref: Exp (exp.rs:119-125), SiLU (silu/math.rs:177-201) directed rounding pattern.
/// Fixes: #3146 — HardSwish chord used raw `as f32` without directed rounding.
#[inline]
fn hardswish_chord(l: f32, u: f32) -> (f32, f32, f32) {
    let (l64, u64) = (l as f64, u as f64);
    let (fl64, fu64) = (hardswish_eval_f64(l64), hardswish_eval_f64(u64));
    let slope64 = (fu64 - fl64) / (u64 - l64);
    let intercept64 = fl64 - slope64 * l64;

    // Directed rounding: absorb f64→f32 slope truncation error.
    // Ref: Exp (exp.rs:159-166), SiLU (silu/math.rs:188-200).
    let slope_f32 = slope64 as f32;
    let max_abs_x = (l.abs().max(u.abs())) as f64;
    let slope_err = next_up_f32(((slope64 - slope_f32 as f64).abs() * max_abs_x) as f32);
    let intercept_f32 = intercept64 as f32;
    (
        slope_f32,
        next_down_f32(intercept_f32 - slope_err),
        next_up_f32(intercept_f32 + slope_err),
    )
}

/// Compute max deviation of HardSwish from chord on [l, u].
/// Returns (max_above, max_below) where max_above is the largest positive
/// deviation (function above chord) and max_below is the largest negative.
fn hardswish_chord_deviation(l: f32, u: f32, chord_slope: f32, chord_intercept: f32) -> (f32, f32) {
    let l64 = f64::from(l);
    let u64 = f64::from(u);
    let slope64 = f64::from(chord_slope);
    let intercept64 = f64::from(chord_intercept);
    let mut max_above = 0.0_f64;
    let mut max_below = 0.0_f64;
    let mut check = |x: f64| {
        let dev = hardswish_eval_f64(x) - (slope64 * x + intercept64);
        max_above = nan_propagating_max_f64(max_above, dev);
        max_below = nan_propagating_max_f64(max_below, -dev);
    };

    // The exact chord meets the function at both endpoints, but the f32
    // slope/nominal intercept can move either endpoint to either side. Check
    // them explicitly instead of relying on a fixed downstream epsilon to
    // absorb that scale-dependent rounding gap.
    check(l64);
    check(u64);

    // Region boundaries and the function's stationary point (when interior).
    let lower_kink = hardswish_lower_kink_f64();
    let upper_kink = hardswish_upper_kink_f64();
    let critical = hardswish_critical_f64();
    if l64 < lower_kink && lower_kink < u64 {
        check(lower_kink);
    }
    if l64 < upper_kink && upper_kink < u64 {
        check(upper_kink);
    }
    if l64 < critical && critical < u64 {
        check(critical);
    }

    // In the quadratic region, the deviation extremum solves
    // 2*alpha*x + beta - chord_slope = 0.
    let q_lo = nan_propagating_max_f64(l64, lower_kink);
    let q_hi = nan_propagating_min_f64(u64, upper_kink);
    if q_lo < q_hi {
        let x_ext =
            (slope64 - f64::from(HardSwishLayer::BETA)) / (2.0 * f64::from(HardSwishLayer::ALPHA));
        check(x_ext.clamp(q_lo, q_hi));
        if q_lo > l64 {
            check(q_lo);
        }
        if q_hi < u64 {
            check(q_hi);
        }
    }
    (f64_to_f32_up(max_above), f64_to_f32_up(max_below))
}

/// Analytical linear relaxation for HardSwish on interval [l, u].
///
/// Uses the chord with analytically-computed exact max deviation offset.
/// Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 4
fn hardswish_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    // Guard: NaN or Inf inputs → maximally loose bounds (always sound).
    if l.is_nan() || u.is_nan() || l.is_infinite() || u.is_infinite() {
        return LinearRelaxation::nan_fallback();
    }
    // A tangent evaluated only at `l` does not enclose a non-degenerate narrow
    // interval: in the convex half of the quadratic region it is a lower bound,
    // not an upper bound.  Cover the full endpoint/critical-value range with a
    // constant relaxation.  This also avoids cancellation in the chord when
    // adjacent f32 endpoints are less than 1e-8 apart.
    if (u - l).abs() < 1e-8 {
        let l64 = f64::from(l);
        let u64 = f64::from(u);
        let y_l = hardswish_eval_f64(l64);
        let y_u = hardswish_eval_f64(u64);
        let critical = hardswish_critical_f64();
        let y_critical = if l64 <= critical && critical <= u64 {
            hardswish_eval_f64(critical)
        } else {
            f64::INFINITY
        };
        let lower = f64_to_f32_down(y_l.min(y_u).min(y_critical));
        let upper = f64_to_f32_up(y_l.max(y_u));
        return LinearRelaxation::new(0.0, lower, 0.0, upper);
    }

    let (chord_slope, lower_intercept, upper_intercept) = hardswish_chord(l, u);
    // Deviation computed with nominal intercept (midpoint of lower/upper).
    // The directed rounding error is already absorbed in lower/upper_intercept.
    // Bit-identical band nominal: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let nominal_intercept = (lower_intercept + upper_intercept) / 2.0;
    let (max_above, max_below) = hardswish_chord_deviation(l, u, chord_slope, nominal_intercept);
    let margin64 = 4.0 * f64::from(f32::EPSILON);
    let lower_final = f64_to_f32_down(f64::from(lower_intercept) - f64::from(max_below) - margin64);
    let upper_final = f64_to_f32_up(f64::from(upper_intercept) + f64::from(max_above) + margin64);
    // Lower bound: lower_intercept (sound below true chord) shifted down by max_below.
    // Upper bound: upper_intercept (sound above true chord) shifted up by max_above.
    LinearRelaxation::new(chord_slope, lower_final, chord_slope, upper_final)
}

impl HardSwishLayer {
    /// CROWN backward propagation through HardSwish with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("HardSwish layer CROWN backward propagation with pre-activation bounds");
        non_finite_domain_guard("HardSwish", pre_activation)?;
        crown_elementwise_backward(bounds, pre_activation, hardswish_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("HardSwish layer batched CROWN backward propagation");
        non_finite_domain_guard("HardSwish", pre_activation)?;
        crown_elementwise_backward_batched(bounds, pre_activation, hardswish_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("HardSwish", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, hardswish_linear_relaxation)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) fn audit_hardswish_relax(l: f32, u: f32) -> LinearRelaxation {
    hardswish_linear_relaxation(l, u)
}
