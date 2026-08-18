// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

use super::LinearRelaxation;
use ny_tensor::{next_down_f32, next_up_f32};

/// Softsign layer: y = x / (1 + |x|)
///
/// Output range is (-1, 1), similar to tanh but computationally cheaper.
/// The function is monotonically increasing and passes through origin.
#[derive(Debug, Clone)]
pub struct SoftsignLayer;

impl SoftsignLayer {
    /// Create a new Softsign layer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for SoftsignLayer {
    fn default() -> Self {
        Self
    }
}

/// Evaluate Softsign: x / (1 + |x|)
#[cfg(test)]
fn softsign_scalar(x: f32) -> f32 {
    // Guard against inf/inf = NaN when x = ±inf.
    // Softsign(x) = x / (1 + |x|). At x = ±inf: ±inf / inf = NaN.
    // Correct limits: Softsign(-inf) = -1, Softsign(+inf) = +1.
    // Ref: SiLU guard pattern (silu.rs:100-105), fix for #1836.
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { -1.0 } else { 1.0 };
    }
    x / (1.0 + x.abs())
}

/// Evaluate Softsign in f64 for directed rounding precision. (#3245)
fn softsign_f64(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { -1.0 } else { 1.0 };
    }
    x / (1.0 + x.abs())
}

impl BoundPropagation for SoftsignLayer {
    /// IBP for Softsign: y = x / (1 + |x|)
    ///
    /// Softsign is monotonically increasing, so bounds are straightforward:
    /// lower_out = softsign(lower_in)
    /// upper_out = softsign(upper_in)
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Softsign is monotonically increasing, so we just apply element-wise.
        // Directed rounding: compute in f64, apply next_down/next_up to guarantee
        // lower bounds round DOWN and upper bounds round UP. (#3245)
        // Range clamp: softsign(x) ∈ [-1, 1] for all x, but directed rounding
        // can push values one ULP outside the mathematical range. (#3316)
        let lower = input
            .lower()
            .mapv(|x| next_down_f32(softsign_f64(x as f64) as f32).clamp(-1.0, 1.0));
        let upper = input
            .upper()
            .mapv(|x| next_up_f32(softsign_f64(x as f64) as f32).clamp(-1.0, 1.0));

        BoundedTensor::new(lower, upper)
    }

    /// CROWN propagation requires pre-activation bounds for nonlinear Softsign.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp(
            "Softsign is nonlinear — use propagate_linear_with_bounds with pre-activation bounds"
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
        SoftsignLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Derivative of softsign: s'(x) = 1 / (1 + |x|)^2
#[inline]
fn softsign_derivative(x: f32) -> f32 {
    let d = 1.0 + x.abs();
    1.0 / (d * d)
}

/// Compute tangent line at point d: y = s'(d)*(x - d) + s(d)
/// Returns (slope, lower_intercept, upper_intercept) with directed rounding.
///
/// Uses f64 intermediates for the intercept to avoid catastrophic cancellation
/// when |d| is large (softsign(d) ≈ ±1 and slope*d ≈ 0, but the subtraction
/// loses precision in f32).
///
/// `max_abs_x` is the maximum |x| over the interval where this tangent will
/// be evaluated — typically `max(|l|, |u|)` for the relaxation interval [l, u].
///
/// Ref: SiLU (silu/math.rs:216-231) directed rounding pattern.
/// Fixes: #3146 — Softsign tangent used raw f32 without directed rounding.
#[inline]
fn softsign_tangent(d: f32, max_abs_x: f32) -> (f32, f32, f32) {
    let d64 = d as f64;
    let slope64 = {
        let denom = 1.0 + d64.abs();
        1.0 / (denom * denom)
    };
    let intercept64 = d64 / (1.0 + d64.abs()) - slope64 * d64;

    let slope_f32 = slope64 as f32;
    let slope_err = next_up_f32(((slope64 - slope_f32 as f64).abs() * max_abs_x as f64) as f32);
    let intercept_f32 = intercept64 as f32;
    (
        slope_f32,
        next_down_f32(intercept_f32 - slope_err),
        next_up_f32(intercept_f32 + slope_err),
    )
}

/// Compute chord slope and intercept for softsign in f64, cast back to f32.
/// Returns (slope, lower_intercept, upper_intercept) with directed rounding.
///
/// Prevents catastrophic cancellation when u ≈ l.
/// Directed rounding absorbs f64→f32 slope truncation error into the intercept.
///
/// Ref: Exp (exp.rs:119-125), SiLU (silu/math.rs:177-201) directed rounding pattern.
/// Fixes: #3146 — Softsign chord used raw `as f32` without directed rounding.
#[inline]
fn softsign_chord_f64(l: f32, u: f32) -> (f32, f32, f32) {
    let l64 = l as f64;
    let u64 = u as f64;
    let sl64 = l64 / (1.0 + l64.abs());
    let su64 = u64 / (1.0 + u64.abs());
    // Keep slope in f64 for intercept to avoid secondary precision loss.
    // Ref: GELU-sound (sound_relax.rs:55-58), ELU (elu.rs:186-187).
    let slope64 = (su64 - sl64) / (u64 - l64);
    let intercept64 = sl64 - slope64 * l64;

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

/// Find the tangent point d such that the tangent line at d upper-bounds
/// softsign at `target_x`. The tangent must satisfy:
///   s'(d)*(target_x - d) + s(d) >= s(target_x)
///
/// For concave region (d > 0): tangent is always above the function.
/// We want the tightest tangent that still passes above s(target_x).
/// Valid tangent points are those closer to target_x (steeper slopes near 0
/// overshoot more). Binary search: 50 iterations for f64 precision.
///
/// Uses f64 arithmetic throughout to avoid binary search rounding errors that
/// could place the tangent on the wrong side of the boundary. (#3285)
///
/// Search in [lo_init, hi_init]. Tangent at hi_init should be valid (above
/// target), tangent at lo_init may be invalid.
fn find_upper_tangent(lo_init: f32, hi_init: f32, target_x: f32) -> f32 {
    let target_x64 = target_x as f64;
    let target_val = softsign_f64(target_x64);
    let (mut lo, mut hi) = (lo_init as f64, hi_init as f64);
    for _ in 0..50 {
        let mid = f64::midpoint(lo, hi);
        let denom = 1.0 + mid.abs();
        let k = 1.0 / (denom * denom);
        let s_mid = mid / (1.0 + mid.abs());
        let tangent_at_target = k * (target_x64 - mid) + s_mid;
        if tangent_at_target >= target_val {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    // Return hi as f32 with conservative ULP bias. The f64 binary search
    // converges to the exact boundary, but f64→f32 rounding may place the
    // point on the wrong side. next_up_f32 biases toward hi_init (away from
    // boundary, toward the known-valid region for upper tangent). (#3285)
    next_up_f32(hi as f32)
}

/// Find the tangent point d such that the tangent line at d lower-bounds
/// softsign at `target_x`. The tangent must satisfy:
///   s'(d)*(target_x - d) + s(d) <= s(target_x)
///
/// For convex region (d < 0): tangent is always below the function.
/// We want the tightest tangent that still passes below s(target_x).
/// Valid tangent points are those closer to target_x (steeper slopes near 0
/// overshoot more). Binary search converges on the boundary.
///
/// Uses f64 arithmetic throughout to avoid binary search rounding errors that
/// could place the tangent on the wrong side of the boundary. (#3285)
///
/// Search in [lo_init, hi_init]. Tangent at lo_init should be valid (below
/// target), tangent at hi_init may be invalid.
fn find_lower_tangent(lo_init: f32, hi_init: f32, target_x: f32) -> f32 {
    let target_x64 = target_x as f64;
    let target_val = softsign_f64(target_x64);
    let (mut lo, mut hi) = (lo_init as f64, hi_init as f64);
    for _ in 0..50 {
        let mid = f64::midpoint(lo, hi);
        let denom = 1.0 + mid.abs();
        let k = 1.0 / (denom * denom);
        let s_mid = mid / (1.0 + mid.abs());
        let tangent_at_target = k * (target_x64 - mid) + s_mid;
        if tangent_at_target <= target_val {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Return lo as f32 with conservative ULP bias. The f64 binary search
    // converges to the exact boundary, but f64→f32 rounding may place the
    // point on the wrong side. next_down_f32 biases toward lo_init (away from
    // boundary, toward the known-valid region for lower tangent). (#3285)
    next_down_f32(lo as f32)
}

/// Analytical linear relaxation for softsign on interval [l, u].
///
/// Softsign s(x) = x / (1 + |x|) is S-shaped:
/// - **Convex** for x < 0: s''(x) = 2/(1+|x|)^3 > 0
/// - **Concave** for x > 0: s''(x) = -2/(1+|x|)^3 < 0
/// - Inflection at x = 0
///
/// Uses the BoundSShaped tangent-line approach from alpha-beta-CROWN
/// (auto_LiRPA/bound_ops/s_shaped.py:279-344) with per-element binary search.
fn softsign_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    // Guard: NaN or Inf inputs → maximally loose bounds (always sound).
    // softsign(±∞) = ±1 but NaN/Inf in interval bounds means no useful
    // linear relaxation exists.
    if l.is_nan() || u.is_nan() || l.is_infinite() || u.is_infinite() {
        return LinearRelaxation::nan_fallback();
    }

    // Handle degenerate point interval with directed rounding.
    // Delegates to softsign_tangent() which uses f64 intermediates and absorbs
    // f64→f32 slope truncation error into the intercept via next_down/next_up.
    // Ref: Exp point-interval (exp.rs:114-124).
    // Fixes: #3190 — previously used pure f32 without directed rounding.
    if (u - l).abs() < 1e-8 {
        let max_abs_x = l.abs().max(u.abs());
        let (slope, lower_i, upper_i) = softsign_tangent(l, max_abs_x);
        return LinearRelaxation::new(slope, lower_i, slope, upper_i);
    }

    // Chord through endpoints (f64 to avoid catastrophic cancellation).
    // Now returns directed-rounded intercepts (#3146).
    let (chord_slope, chord_lower_i, chord_upper_i) = softsign_chord_f64(l, u);
    let max_abs_x = l.abs().max(u.abs());

    if u <= 0.0 {
        // Case 1: Entirely convex (x < 0).
        // Upper: chord (lies above convex function) → use chord upper_intercept.
        // Lower: tangent at optimal point d → use tangent lower_intercept.
        let d = find_lower_tangent(l, u, u);
        let (tk, tk_lower_i, _tk_upper_i) = softsign_tangent(d, max_abs_x);
        return LinearRelaxation::new(tk, tk_lower_i, chord_slope, chord_upper_i);
    }

    if l >= 0.0 {
        // Case 2: Entirely concave (x > 0).
        // Lower: chord (lies below concave function) → use chord lower_intercept.
        // Upper: tangent at optimal point d → use tangent upper_intercept.
        let d = find_upper_tangent(l, u, l);
        let (tk, _tk_lower_i, tk_upper_i) = softsign_tangent(d, max_abs_x);
        return LinearRelaxation::new(chord_slope, chord_lower_i, tk, tk_upper_i);
    }

    // Case 3: Crosses inflection at x = 0 (l < 0 < u).
    // Following BoundSShaped.bound_relax_impl from alpha-beta-CROWN
    // (auto_LiRPA/operators/s_shaped.py:310-321).
    //
    // For S-shaped function (convex x<0, concave x>0):
    // - chord_slope < s'(l) → chord lies below the function → valid lower bound
    // - chord_slope < s'(u) → chord lies above the function → valid upper bound
    //
    // When the interval is wide (|l| and |u| large), both s'(l) and s'(u) are
    // very small while chord_slope is moderate, so neither condition holds and
    // we fall through to the general case with independent tangent lines.
    let dl = softsign_derivative(l);
    let du = softsign_derivative(u);

    let chord_lower_valid = chord_slope < dl;
    let chord_upper_valid = chord_slope < du;

    if chord_lower_valid && chord_upper_valid {
        // Chord is valid for both bounds (rare — interval is narrow near inflection).
        LinearRelaxation::new(chord_slope, chord_lower_i, chord_slope, chord_upper_i)
    } else if chord_lower_valid {
        // Chord is valid lower bound only → use chord lower_intercept.
        // Upper: tangent in the concave region → use tangent upper_intercept.
        let d = find_upper_tangent(0.0, u, l);
        let (tk, _tk_lower_i, tk_upper_i) = softsign_tangent(d, max_abs_x);
        LinearRelaxation::new(chord_slope, chord_lower_i, tk, tk_upper_i)
    } else if chord_upper_valid {
        // Chord is valid upper bound only → use chord upper_intercept.
        // Lower: tangent in the convex region → use tangent lower_intercept.
        let d = find_lower_tangent(l, 0.0, u);
        let (tk, tk_lower_i, _tk_upper_i) = softsign_tangent(d, max_abs_x);
        LinearRelaxation::new(tk, tk_lower_i, chord_slope, chord_upper_i)
    } else {
        // General crossing: chord is neither a valid upper nor lower bound.
        // Need independent tangent lines for both bounds.
        // Upper: tangent in concave region → use upper_intercept.
        let d_upper = find_upper_tangent(0.0, u, l);
        let (ku, _ku_lower_i, ku_upper_i) = softsign_tangent(d_upper, max_abs_x);
        // Lower: tangent in convex region → use lower_intercept.
        let d_lower = find_lower_tangent(l, 0.0, u);
        let (kl, kl_lower_i, _kl_upper_i) = softsign_tangent(d_lower, max_abs_x);
        LinearRelaxation::new(kl, kl_lower_i, ku, ku_upper_i)
    }
}

impl SoftsignLayer {
    /// CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Softsign layer CROWN backward propagation with pre-activation bounds");
        non_finite_domain_guard("Softsign", pre_activation)?;
        crown_elementwise_backward(bounds, pre_activation, softsign_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("Softsign layer batched CROWN backward propagation");
        non_finite_domain_guard("Softsign", pre_activation)?;
        crown_elementwise_backward_batched(bounds, pre_activation, softsign_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Softsign", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, softsign_linear_relaxation)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) fn audit_softsign_relax(l: f32, u: f32) -> LinearRelaxation {
    softsign_linear_relaxation(l, u)
}
