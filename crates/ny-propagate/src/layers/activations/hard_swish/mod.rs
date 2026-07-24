// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use super::LinearRelaxation;
use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};
use ny_tensor::{next_down_f32, next_up_f32};

/// HardSwish layer: y = x * HardSigmoid(x)
///
/// Used in MobileNetV3 as a more efficient alternative to Swish (SiLU).
/// y = x * max(0, min(1, (x + 3) / 6))
#[derive(Debug, Clone, Default)]
pub struct HardSwishLayer;

impl HardSwishLayer {
    /// Create a new HardSwish layer.
    pub fn new() -> Self {
        Self
    }

    /// Evaluate HardSwish at a point: x * max(0, min(1, (x + 3) / 6))
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        // Guard against 0 * inf = NaN when x = -inf.
        // HardSwish(x) = x * clamp((x+3)/6, 0, 1). At x = -inf: (-inf)*0 = NaN.
        // Correct limits: HardSwish(x) = 0 for x <= -3, HardSwish(x) = x for x >= 3.
        // Ref: SiLU guard pattern (silu.rs:100-105), fix for #1836.
        if !x.is_finite() {
            if x.is_nan() {
                return f32::NAN;
            }
            return if x.is_sign_negative() { 0.0 } else { x };
        }
        x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)
    }
}

impl BoundPropagation for HardSwishLayer {
    /// IBP for HardSwish: y = x * max(0, min(1, (x + 3) / 6))
    ///
    /// Three regions:
    /// - y = 0 when x <= -3 (HardSigmoid = 0)
    /// - y = x * (x + 3) / 6 when -3 < x < 3 (quadratic)
    /// - y = x when x >= 3 (HardSigmoid = 1)
    ///
    /// The quadratic region has derivative (2x + 3) / 6, zero at x = -1.5
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
                    // Evaluate at bounds
                    let y_l = self.eval(in_l);
                    let y_u = self.eval(in_u);

                    // In the quadratic region (-3 < x < 3), the minimum is at x = -1.5
                    // where y = -1.5 * (-1.5 + 3) / 6 = -1.5 * 1.5 / 6 = -0.375
                    // Check if interval contains -1.5 (the critical point is in quadratic region
                    // which spans -3 to 3, so if interval contains -1.5, it overlaps the region)
                    let min_at_critical = if in_l < -1.5 && in_u > -1.5 {
                        self.eval(-1.5) // = -0.375
                    } else {
                        f32::INFINITY
                    };

                    *out_l = nan_propagating_min(nan_propagating_min(y_l, y_u), min_at_critical);
                    *out_u = nan_propagating_max(y_l, y_u);
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

/// Evaluate HardSwish: x * HardSigmoid(x) = x * max(0, min(1, (x + 3) / 6))
///
/// Note: caller `hardswish_linear_relaxation` guards for inf/NaN at entry,
/// so this is only called with finite inputs. The guard here is defense-in-depth.
#[inline]
fn hardswish_eval(x: f32) -> f32 {
    // Guard against 0 * inf = NaN at x = -inf (defense-in-depth, #1836).
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

/// Evaluate HardSwish in f64 to avoid catastrophic cancellation in chord slope.
/// HardSwish(x) = x * clamp((x + 3) / 6, 0, 1).
/// Same pattern as silu_eval_f64 (silu/math.rs), exp_eval_f64 (#1745).
#[inline]
fn hardswish_eval_f64(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    x * ((x + 3.0) / 6.0).clamp(0.0, 1.0)
}

/// HardSwish derivative in f64.
///
/// HardSwish'(x) = 0 for x ≤ -3, (2x + 3)/6 for -3 < x < 3, 1 for x ≥ 3.
/// Used by the point-interval directed rounding path to avoid f32 precision loss.
/// Fixes: #3190 — point-interval path used f32-only computation.
#[inline]
fn hardswish_derivative_f64(x: f64) -> f64 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (2.0 * x + 3.0) / 6.0
    }
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
    let mut max_above: f32 = 0.0;
    let mut max_below: f32 = 0.0;
    let mut check = |x: f32| {
        let dev = hardswish_eval(x) - (chord_slope * x + chord_intercept);
        max_above = nan_propagating_max(max_above, dev);
        max_below = nan_propagating_max(max_below, -dev);
    };

    // Region boundaries -3, 3 and critical point -1.5 (if interior to [l, u])
    if l < -3.0 && -3.0 < u {
        check(-3.0);
    }
    if l < 3.0 && 3.0 < u {
        check(3.0);
    }
    if l < -1.5 && -1.5 < u {
        check(-1.5);
    }

    // Quadratic region [-3, 3] ∩ [l, u]: analytical extremum of deviation.
    // dev'(x) = (2x + 3)/6 - chord_slope = 0 → x_ext = 3*chord_slope - 1.5
    let q_lo = nan_propagating_max(l, -3.0);
    let q_hi = nan_propagating_min(u, 3.0);
    if q_lo < q_hi {
        check(((3.0 * chord_slope) - 1.5).clamp(q_lo, q_hi));
        if q_lo > l {
            check(q_lo);
        }
        if q_hi < u {
            check(q_hi);
        }
    }
    (max_above, max_below)
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
    // Handle degenerate point interval with directed rounding.
    // Uses f64 intermediates to avoid catastrophic cancellation in f(l) - f'(l)*l.
    // Ref: Exp point-interval (exp.rs:114-124), SiLU tangent (silu/math.rs:216-231).
    // Fixes: #3190 — previously used pure f32 without directed rounding.
    if (u - l).abs() < 1e-8 {
        let l64 = l as f64;
        let y64 = hardswish_eval_f64(l64);
        let slope64 = hardswish_derivative_f64(l64);
        let intercept64 = y64 - slope64 * l64;
        let slope_f32 = slope64 as f32;
        let max_abs_x = (l.abs().max(u.abs())) as f64;
        let slope_err = next_up_f32(((slope64 - slope_f32 as f64).abs() * max_abs_x) as f32);
        let intercept_f32 = intercept64 as f32;
        return LinearRelaxation::new(
            slope_f32,
            next_down_f32(intercept_f32 - slope_err),
            slope_f32,
            next_up_f32(intercept_f32 + slope_err),
        );
    }

    let (chord_slope, lower_intercept, upper_intercept) = hardswish_chord(l, u);
    // Deviation computed with nominal intercept (midpoint of lower/upper).
    // The directed rounding error is already absorbed in lower/upper_intercept.
    // Bit-identical band nominal: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let nominal_intercept = (lower_intercept + upper_intercept) / 2.0;
    let (max_above, max_below) = hardswish_chord_deviation(l, u, chord_slope, nominal_intercept);
    let margin = 4.0 * f32::EPSILON;
    // Lower bound: lower_intercept (sound below true chord) shifted down by max_below.
    // Upper bound: upper_intercept (sound above true chord) shifted up by max_above.
    LinearRelaxation::new(
        chord_slope,
        lower_intercept - max_below - margin,
        chord_slope,
        upper_intercept + max_above + margin,
    )
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
