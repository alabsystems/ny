// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;
use tracing::debug;

use crate::bounds::{nan_propagating_max, nan_propagating_min};
use crate::layers::common::{
    crown_elementwise_backward, crown_elementwise_backward_batched,
    crown_elementwise_backward_patches, non_finite_domain_guard, BoundPropagation,
};
use crate::{BatchedLinearBounds, LinearBounds};

use super::LinearRelaxation;
use ny_tensor::{next_down_f32, next_up_f32};

/// Mish layer: y = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
///
/// Mish is a self-regularized non-monotonic activation function.
/// It has been shown to outperform ReLU/Swish in various benchmarks
/// (e.g., YOLOv4). Unlike ReLU, Mish is smooth and allows small
/// negative values, which can improve gradient flow.
#[derive(Debug, Clone, Default)]
pub struct MishLayer;

impl MishLayer {
    /// Create a new Mish layer.
    pub fn new() -> Self {
        Self
    }
}

/// Evaluate Mish: x * tanh(softplus(x))
#[inline]
fn mish_eval(x: f32) -> f32 {
    // Guard against 0 * inf = NaN when x = ±inf.
    // Mish(x) = x * tanh(softplus(x)). At x = -inf: (-inf)*tanh(0) = (-inf)*0 = NaN.
    // Correct limits: Mish(-inf) = 0, Mish(+inf) = +inf.
    // Ref: SiLU guard pattern (silu.rs:100-105), fix for #1836.
    if !x.is_finite() {
        if x.is_nan() {
            return f32::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    // softplus(x) = ln(1 + exp(x))
    // For numerical stability, use log1p(exp(x)) for small x
    // and x + log1p(exp(-x)) for large x
    let softplus = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    };
    x * softplus.tanh()
}

/// Evaluate Mish in f64 to avoid catastrophic cancellation in chord slope.
/// Mish(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + exp(x)))
#[inline]
fn mish_eval_f64(x: f64) -> f64 {
    if !x.is_finite() {
        if x.is_nan() {
            return f64::NAN;
        }
        return if x.is_sign_negative() { 0.0 } else { x };
    }
    let softplus = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f64 + x.exp()).ln()
    };
    x * softplus.tanh()
}

/// Compute chord slope and intercept for Mish in f64, cast back to f32.
/// Returns (slope, lower_intercept, upper_intercept) with directed rounding.
///
/// Prevents catastrophic cancellation when u ≈ l.
/// Directed rounding absorbs f64→f32 slope truncation error into the intercept:
/// - `lower_intercept`: guaranteed `slope*x + lower_intercept <= true_chord(x)` ∀x∈[l,u]
/// - `upper_intercept`: guaranteed `slope*x + upper_intercept >= true_chord(x)` ∀x∈[l,u]
///
/// Ref: Exp (exp.rs:119-125), SiLU (silu/math.rs:177-201) directed rounding pattern.
/// Fixes: #3146 — Mish chord used raw `as f32` without directed rounding.
#[inline]
fn mish_chord_f64(l: f32, u: f32) -> (f32, f32, f32) {
    let l64 = l as f64;
    let u64 = u as f64;
    let fl64 = mish_eval_f64(l64);
    let fu64 = mish_eval_f64(u64);
    // Keep slope in f64 for intercept to avoid secondary precision loss.
    // Ref: GELU-sound (sound_relax.rs:55-58), ELU (elu.rs:186-187).
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

/// Mish derivative in f64: tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
///
/// Used by the point-interval directed rounding path to avoid f32 precision loss.
/// Ref: mish_derivative (f32 version below), Exp point-interval (exp.rs:114-124).
/// Fixes: #3190 — point-interval path used f32-only computation.
#[inline]
fn mish_derivative_f64(x: f64) -> f64 {
    let softplus = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f64 + x.exp()).ln()
    };
    let tanh_sp = softplus.tanh();
    let sech2_sp = 1.0 - tanh_sp * tanh_sp;
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    tanh_sp + x * sech2_sp * sigmoid
}

/// Mish derivative: tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
#[inline]
fn mish_derivative(x: f32) -> f32 {
    // For numerical stability
    let softplus = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f32 + x.exp()).ln()
    };
    let tanh_sp = softplus.tanh();
    let sech2_sp = 1.0 - tanh_sp * tanh_sp;
    let sigmoid = 1.0 / (1.0 + (-x).exp());

    // d/dx[x * tanh(softplus(x))]
    // = tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
    tanh_sp + x * sech2_sp * sigmoid
}

/// Second derivative of Mish, computed in f64.
///
/// Let s = softplus(x), t = tanh(s), σ = sigmoid(x), and note s'(x) = σ(x).
/// Mish'(x)  = t + x · (1 - t²) · σ
/// Mish''(x) = d/dx[t + x(1-t²)σ]
///           = (1-t²)σ                          (from d/dx t)
///           + (1-t²)σ                          (from d/dx of x → 1, times (1-t²)σ)
///           + x · d/dx[(1-t²)σ]
/// where d/dx[(1-t²)σ] = (-2t)(1-t²)σ · σ + (1-t²) · σ(1-σ)
///   (since d/dx t = (1-t²)σ, d/dx(1-t²) = -2t·(1-t²)σ, and σ' = σ(1-σ)).
///
/// Used only for Newton iterations in inflection-point finding and for the
/// extremum search in upper-bound verification — NOT on the soundness-critical
/// path (the final bound is always verified directly against Mish in f64).
/// Ref: silu_second_derivative (silu/math.rs:49-55).
#[inline]
fn mish_second_derivative_f64(x: f64) -> f64 {
    let softplus = if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0_f64 + x.exp()).ln()
    };
    let t = softplus.tanh();
    let one_minus_t2 = 1.0 - t * t;
    let sigma = 1.0 / (1.0 + (-x).exp());
    // d/dx t = (1 - t²) σ
    let dt = one_minus_t2 * sigma;
    // d/dx[(1 - t²) σ] = (-2 t · dt) σ + (1 - t²) σ(1-σ)
    let d_g = (-2.0 * t * dt) * sigma + one_minus_t2 * (sigma * (1.0 - sigma));
    // Mish'' = 2 (1 - t²) σ + x · d/dx[(1 - t²) σ]
    2.0 * one_minus_t2 * sigma + x * d_g
}

/// The two inflection points of Mish, where Mish''(x) = 0.
///
/// Mish has concave-convex-concave structure (verified numerically):
/// - x < p₁ ≈ -2.2564: concave (Mish'' < 0)
/// - p₁ < x < p₂ ≈ +1.4906: convex (Mish'' > 0)
/// - x > p₂: concave (Mish'' < 0)
///
/// This is topologically identical to SiLU's curvature structure, which is why
/// SiLU's region-classified relaxation transfers soundly to Mish. The inflection
/// points are asymmetric (unlike SiLU's ±2.40), but the construction only relies
/// on the curvature SIGN per region, not symmetry.
///
/// Found via Newton's method on Mish''(x) = 0 in f64.
/// Ref: silu_inflection_points (silu/math.rs:82-125).
fn mish_inflection_points() -> (f32, f32) {
    static MISH_INFLECTION_POINTS: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    *MISH_INFLECTION_POINTS.get_or_init(|| {
        // Newton on f(x) = Mish''(x); f'(x) ≈ Mish'''(x) via central difference.
        // Mish'' is smooth, so a finite-difference Newton converges robustly.
        let newton = |mut x: f64, iters: usize| -> f64 {
            let eps = 1.0e-5_f64;
            for _ in 0..iters {
                let f = mish_second_derivative_f64(x);
                let fd = (mish_second_derivative_f64(x + eps)
                    - mish_second_derivative_f64(x - eps))
                    / (2.0 * eps);
                if fd.abs() < 1.0e-14 {
                    break;
                }
                let step = f / fd;
                x -= step;
                if step.abs() < 1.0e-12 {
                    break;
                }
            }
            x
        };
        // Left inflection near -2.2564, right inflection near +1.4906.
        let p1 = newton(-2.256, 50);
        let p2 = newton(1.491, 50);
        (p1 as f32, p2 as f32)
    })
}

/// Raw tangent line at point d: y = Mish'(d)·(x - d) + Mish(d), as plain f32
/// casts of f64 intermediates. Used for binary-search candidate checking where
/// the tight (non-rounded) value is needed. Final output uses `mish_tangent`
/// with directed rounding. Ref: silu_tangent_raw (silu/math.rs:251-256).
#[inline]
fn mish_tangent_raw(d: f32) -> (f32, f32) {
    let d64 = d as f64;
    let slope64 = mish_derivative_f64(d64);
    let intercept64 = mish_eval_f64(d64) - slope64 * d64;
    (slope64 as f32, intercept64 as f32)
}

/// Tangent line at point d with directed rounding.
/// Returns (slope, lower_intercept, upper_intercept) such that, for all x with
/// |x| <= max_abs_x:
/// - `slope*x + lower_intercept <= true_tangent(x)`
/// - `slope*x + upper_intercept >= true_tangent(x)`
///
/// `max_abs_x` is the maximum |x| over the interval where the tangent is used,
/// typically max(|l|, |u|). Uses f64 intermediates to avoid catastrophic
/// cancellation when |d| is large (Mish(d) ≈ d and slope·d ≈ d).
/// Ref: silu_tangent (silu/math.rs:229-244).
#[inline]
fn mish_tangent(d: f32, max_abs_x: f32) -> (f32, f32, f32) {
    let d64 = d as f64;
    let slope64 = mish_derivative_f64(d64);
    let intercept64 = mish_eval_f64(d64) - slope64 * d64;

    let slope_f32 = slope64 as f32;
    let slope_err = next_up_f32(((slope64 - slope_f32 as f64).abs() * max_abs_x as f64) as f32);
    let intercept_f32 = intercept64 as f32;
    (
        slope_f32,
        next_down_f32(intercept_f32 - slope_err),
        next_up_f32(intercept_f32 + slope_err),
    )
}

impl BoundPropagation for MishLayer {
    /// IBP for Mish: y = x * tanh(softplus(x))
    ///
    /// Mish is NOT monotonic - it has a minimum near x ≈ -0.31.
    /// We need to check for the critical point in each interval.
    #[inline]
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        // Mish has a critical point (minimum) near x ≈ -0.31 where derivative = 0
        // We use Newton's method to find it more precisely
        static MISH_CRITICAL: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let critical = *MISH_CRITICAL.get_or_init(|| {
            // Newton's method to find where derivative = 0
            let mut x = -0.31_f32;
            for _ in 0..20 {
                let d = mish_derivative(x);
                // Numerical derivative of derivative
                let eps = 1e-5_f32;
                let dd = (mish_derivative(x + eps) - mish_derivative(x - eps)) / (2.0 * eps);
                if dd.abs() < 1e-10 {
                    break;
                }
                x -= d / dd;
            }
            x
        });
        let critical_val = mish_eval_f64(critical as f64) as f32;

        // Directed rounding: apply next_down/next_up to EACH intermediate evaluation
        // BEFORE min/max selection. Plain `as f32` rounds to nearest — a candidate
        // min could round UP, causing the final min to miss the true minimum.
        // next_down/next_up on the final result alone cannot recover.
        // Ref: #3336, same pattern as GELU and SiLU directed rounding fixes.
        let bound_fn = |l: f32, u: f32| -> (f32, f32) {
            let fl = mish_eval_f64(l as f64) as f32;
            let fu = mish_eval_f64(u as f64) as f32;

            let (min_val, max_val) = if l <= critical && critical <= u {
                (
                    nan_propagating_min(
                        next_down_f32(critical_val),
                        nan_propagating_min(next_down_f32(fl), next_down_f32(fu)),
                    ),
                    nan_propagating_max(next_up_f32(fl), next_up_f32(fu)),
                )
            } else {
                (
                    nan_propagating_min(next_down_f32(fl), next_down_f32(fu)),
                    nan_propagating_max(next_up_f32(fl), next_up_f32(fu)),
                )
            };
            (min_val, max_val)
        };

        let lower_shape = input.lower().shape().to_vec();
        let mut lower_data = Vec::with_capacity(input.lower().len());
        let mut upper_data = Vec::with_capacity(input.upper().len());

        for (l, u) in input.lower().iter().zip(input.upper().iter()) {
            if !l.is_finite() || !u.is_finite() {
                // Guard NaN/Inf from unchecked callers: min/max chains can skip
                // NaN and silently narrow bounds. Fall back to loose sound bounds.
                lower_data.push(f32::NEG_INFINITY);
                upper_data.push(f32::INFINITY);
                continue;
            }
            let (lo, hi) = bound_fn(*l, *u);
            lower_data.push(lo);
            upper_data.push(hi);
        }

        let lower = ArrayD::from_shape_vec(IxDyn(&lower_shape), lower_data)
            .map_err(|e| NyError::InvalidSpec(format!("Mish lower reshape: {}", e)))?;
        let upper = ArrayD::from_shape_vec(IxDyn(&lower_shape), upper_data)
            .map_err(|e| NyError::InvalidSpec(format!("Mish upper reshape: {}", e)))?;

        BoundedTensor::new_allow_infinite(lower, upper)
    }

    /// CROWN backward propagation requires pre-activation bounds.
    /// Use `MishLayer::propagate_linear_with_bounds` instead.
    #[inline]
    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::InvalidSpec(
            "Mish CROWN propagation requires pre-activation bounds. \
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
        MishLayer::propagate_linear_with_bounds(self, bounds, pre_activation)
    }
}

/// Find a root of g(x) = Mish'(x) - target in the bracket [a, b] using bisection.
/// Assumes g(a) and g(b) have opposite signs. Returns the root.
// Bit-identical bisection anchors: f32::midpoint rounds differently at overflow/subnormal edges.
#[allow(clippy::manual_midpoint)]
fn bisect_mish_derivative_root(a: f32, b: f32, target: f32) -> f32 {
    let mut lo = a;
    let mut hi = b;

    // 50 bisection iterations give ~15 decimal digits of precision
    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        let g_mid = mish_derivative(mid) - target;
        let g_lo = mish_derivative(lo) - target;

        if g_mid == 0.0 {
            return mid;
        }
        if g_lo.is_sign_positive() == g_mid.is_sign_positive() {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Analytical linear relaxation for Mish on interval [l, u].
///
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept) such that:
///   lower_slope * x + lower_intercept <= Mish(x) <= upper_slope * x + upper_intercept
/// for all x in [l, u].
///
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
///
/// Mish is smooth but NOT monotonic — has a minimum near x ≈ -1.19.
/// Neither globally convex nor concave (has inflection points).
///
/// Strategy: chord ± analytical max deviation.
///   h(x) = Mish(x) - chord(x), with h(l) = h(u) = 0.
///   h'(x) = Mish'(x) - chord_slope = 0 at critical points.
///   Mish' has one local minimum, so h'(x) = 0 has at most 2 roots in [l, u].
///   We isolate roots by scanning for sign changes, then refine with bisection.
///
/// No sampling. No epsilon slack. Provably sound analytical bounds.
/// Mish global minimum value (approximately -0.30884 at x ≈ -1.1924).
/// Used as a globally sound constant lower bound for infinite ranges.
const MISH_GLOBAL_MIN: f32 = -0.309;

fn mish_linear_relaxation(l: f32, u: f32) -> LinearRelaxation {
    // Guard: NaN bounds → return (-inf, +inf) intercepts so CROWN drives bounds to ±inf.
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::nan_fallback();
    }

    // Guard: infinite bounds cause inf/inf = NaN in chord computation.
    // Mish(x) -> 0 as x -> -inf, Mish(x) -> x as x -> +inf.
    // Global range: [MISH_GLOBAL_MIN, +inf).
    if l.is_infinite() && u.is_infinite() {
        // Constant bounds: y in [MISH_GLOBAL_MIN, +inf). Sound but maximally loose.
        return LinearRelaxation::new(0.0, MISH_GLOBAL_MIN, 0.0, f32::INFINITY);
    }
    if l.is_infinite() {
        // l = -inf, u finite: Mish(-inf) -> 0, Mish(u) = fu.
        // Lower: constant at min(MISH_GLOBAL_MIN, Mish(u)).
        // Upper: constant at max(0, Mish(u)) (since Mish -> 0 from below as x -> -inf).
        let fu = mish_eval(u);
        // NaN-propagating: if fu is NaN, the NaN flows through instead of
        // being silently absorbed by IEEE .min()/.max(). (#2714)
        return LinearRelaxation::new(
            0.0,
            nan_propagating_min(MISH_GLOBAL_MIN, fu),
            0.0,
            nan_propagating_max(fu, 0.0),
        );
    }
    if u.is_infinite() {
        // l finite, u = +inf: no finite affine upper envelope exists.
        // Mish is non-convex, so tangent-at-l is NOT globally sound as a lower
        // envelope on [l, +inf). Use the global minimum as a constant lower bound.
        return LinearRelaxation::new(0.0, MISH_GLOBAL_MIN, 0.0, f32::INFINITY);
    }

    // Point interval: tangent at l with directed rounding.
    // Uses f64 intermediates to avoid catastrophic cancellation in f(l) - f'(l)*l.
    // Ref: Exp point-interval (exp.rs:114-124), SiLU tangent (silu/math.rs:216-231).
    // Fixes: #3190 — previously used pure f32 without directed rounding.
    if (u - l).abs() < 1e-8 {
        let l64 = l as f64;
        let y64 = mish_eval_f64(l64);
        let slope64 = mish_derivative_f64(l64);
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

    // Finite, non-point interval [l, u]. Route through a SiLU-style
    // convexity-region-classified relaxation (tangent + chord + binary search),
    // which gives tight, provably-sound bounds. Mish has the same
    // concave-convex-concave structure as SiLU (see `mish_inflection_points`),
    // so the SiLU construction transfers exactly.
    //
    // Each candidate line is VERIFIED to stay on the correct side of Mish across
    // the WHOLE [l, u] (in f64). If verification fails for either bound, that
    // bound falls back to the historically-sound chord±deviation band
    // (`mish_fallback_band`). This guarantees we never emit an unsound line.
    region_classified_relaxation(l, u)
}

// ===================================================================
// SiLU-style convexity-region-classified relaxation for Mish.
//
// Mish is concave on (-inf, p1), convex on (p1, p2), concave on (p2, +inf),
// with p1 ≈ -2.2564, p2 ≈ +1.4906 (see `mish_inflection_points`). This is the
// same concave-convex-concave structure as SiLU, so we mirror SiLU's
// construction (silu/relaxation.rs) EXACTLY:
//
//   - fully convex  [p1 <= l, u <= p2]: tangent is a global LOWER bound,
//                                        chord is a global UPPER bound.
//   - fully concave [u <= p1 or l >= p2]: chord LOWER, tangent UPPER (Jensen).
//   - crossing: lower = tangent from the convex region (binary-searched so it
//               stays below Mish at the concave-tail endpoints);
//               upper = chord if valid, else a tangent from the right concave
//               region (binary-searched).
//
// SOUNDNESS: every emitted line is verified in f64 to stay on the correct side
// of Mish across the whole [l, u] (endpoints + inflection points + the interior
// extremum found by Newton on Mish'(x) = slope). If a candidate cannot be
// verified, that bound falls back to `mish_fallback_band`, which is always
// sound. Thus no unsound line is ever emitted.
// ===================================================================

fn region_classified_relaxation(l: f32, u: f32) -> LinearRelaxation {
    let (p1, p2) = mish_inflection_points();

    // Fully convex sub-interval: tangent lower, chord upper.
    if l >= p1 && u <= p2 {
        return mish_relaxation_convex(l, u);
    }
    // Fully concave (left tail or right tail): chord lower, tangent upper.
    if u <= p1 || l >= p2 {
        return mish_relaxation_concave(l, u);
    }
    // Interval crosses at least one inflection point.
    mish_relaxation_crossing(l, u, p1, p2)
}

/// Build a result from a (lower, upper) pair of (slope, intercept).
#[inline]
fn combine(lower: (f32, f32), upper: (f32, f32)) -> LinearRelaxation {
    LinearRelaxation::new(lower.0, lower.1, upper.0, upper.1)
}

/// Mean value of `slope*x + intercept` over [l, u] (in f64). Used only to choose
/// the tighter of two ALREADY-SOUND candidate lines — never affects soundness.
#[inline]
fn line_mean(l: f32, u: f32, slope: f32, intercept: f32) -> f64 {
    // For a line, the mean over [l, u] is the value at the midpoint.
    let mid = f64::midpoint(l as f64, u as f64);
    slope as f64 * mid + intercept as f64
}

/// Choose the tighter LOWER bound between a candidate and the band's lower bound.
/// Both must be sound; the tighter (higher mean) one is returned. The candidate
/// is only considered if `verified` is true. Never returns an unsound line: the
/// band lower bound is always sound, and the candidate is only used when verified.
#[inline]
fn pick_lower(
    l: f32,
    u: f32,
    candidate: Option<(f32, f32)>,
    band: &LinearRelaxation,
) -> (f32, f32) {
    let band_lo = (band.lower_slope, band.lower_intercept);
    match candidate {
        Some((s, i)) if verify_lower_line(l, u, s, i) => {
            // Higher mean = tighter lower bound (closer to Mish from below).
            if line_mean(l, u, s, i) >= line_mean(l, u, band_lo.0, band_lo.1) {
                (s, i)
            } else {
                band_lo
            }
        }
        _ => band_lo,
    }
}

/// Choose the tighter UPPER bound between a candidate and the band's upper bound.
/// Both must be sound; the tighter (lower mean) one is returned. The candidate is
/// only considered if `verified` is true.
#[inline]
fn pick_upper(
    l: f32,
    u: f32,
    candidate: Option<(f32, f32)>,
    band: &LinearRelaxation,
) -> (f32, f32) {
    let band_hi = (band.upper_slope, band.upper_intercept);
    match candidate {
        Some((s, i)) if verify_upper_line(l, u, s, i) => {
            // Lower mean = tighter upper bound (closer to Mish from above).
            if line_mean(l, u, s, i) <= line_mean(l, u, band_hi.0, band_hi.1) {
                (s, i)
            } else {
                band_hi
            }
        }
        _ => band_hi,
    }
}

/// Does `slope*x + intercept` stay <= Mish(x) for all x in [l, u]?
/// Checked in f64 at the endpoints, the inflection points, and the interior
/// extremum of (Mish(x) - line) found via Newton on Mish'(x) = slope.
/// Ref: silu verify (via binary-search endpoint checks + Newton extremum).
// Bit-identical verification sample points: f32::midpoint rounds differently at overflow edges.
#[allow(clippy::manual_midpoint)]
fn verify_lower_line(l: f32, u: f32, slope: f32, intercept: f32) -> bool {
    let line = |x: f64| slope as f64 * x + intercept as f64;
    // Mish(x) - line(x) is minimized where Mish'(x) = slope. Newton on
    // g(x) = Mish'(x) - slope, g'(x) = Mish''(x). Start from several points.
    let mut samples: Vec<f32> = vec![l, u, (l + u) / 2.0, l + 0.25 * (u - l), l + 0.75 * (u - l)];
    let (p1, p2) = mish_inflection_points();
    for &p in &[p1, p2] {
        if p > l && p < u {
            samples.push(p);
        }
    }
    let starts = [l, (l + u) / 2.0, u];
    for &x0 in &starts {
        let mut x = x0 as f64;
        for _ in 0..30 {
            let g = mish_derivative_f64(x) - slope as f64;
            let gp = mish_second_derivative_f64(x);
            if gp.abs() < 1.0e-14 {
                break;
            }
            x -= g / gp;
            x = x.clamp(l as f64, u as f64);
        }
        samples.push(x as f32);
    }
    for &x in &samples {
        let x64 = (x as f64).clamp(l as f64, u as f64);
        // line above Mish anywhere → not a valid lower bound.
        if line(x64) > mish_eval_f64(x64) {
            return false;
        }
    }
    true
}

/// Does `slope*x + intercept` stay >= Mish(x) for all x in [l, u]?
/// Mirror of `verify_lower_line` for the upper side.
// Bit-identical verification sample points: f32::midpoint rounds differently at overflow edges.
#[allow(clippy::manual_midpoint)]
fn verify_upper_line(l: f32, u: f32, slope: f32, intercept: f32) -> bool {
    let line = |x: f64| slope as f64 * x + intercept as f64;
    let mut samples: Vec<f32> = vec![l, u, (l + u) / 2.0, l + 0.25 * (u - l), l + 0.75 * (u - l)];
    let (p1, p2) = mish_inflection_points();
    for &p in &[p1, p2] {
        if p > l && p < u {
            samples.push(p);
        }
    }
    let starts = [l, (l + u) / 2.0, u];
    for &x0 in &starts {
        let mut x = x0 as f64;
        for _ in 0..30 {
            let g = mish_derivative_f64(x) - slope as f64;
            let gp = mish_second_derivative_f64(x);
            if gp.abs() < 1.0e-14 {
                break;
            }
            x -= g / gp;
            x = x.clamp(l as f64, u as f64);
        }
        samples.push(x as f32);
    }
    for &x in &samples {
        let x64 = (x as f64).clamp(l as f64, u as f64);
        // Mish above line anywhere → not a valid upper bound.
        if mish_eval_f64(x64) > line(x64) {
            return false;
        }
    }
    true
}

/// Fully-convex sub-interval [p1 <= l, u <= p2].
///
/// On a convex interval: every tangent line is a global LOWER bound (function
/// lies above its tangents), and the endpoint chord is a global UPPER bound
/// (function lies below its chords). We use the midpoint tangent for the lower
/// bound. Both candidates are verified in f64; on failure, fall back to the band.
fn mish_relaxation_convex(l: f32, u: f32) -> LinearRelaxation {
    let max_abs_x = l.abs().max(u.abs());
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let (ls, ls_lo, _ls_hi) = mish_tangent((l + u) / 2.0, max_abs_x);
    let (us, _us_lo, us_hi) = mish_chord_f64(l, u);

    let band = mish_fallback_band(l, u);
    let lower = pick_lower(l, u, Some((ls, ls_lo)), &band);
    let upper = pick_upper(l, u, Some((us, us_hi)), &band);
    combine(lower, upper)
}

/// Fully-concave sub-interval (u <= p1, or l >= p2).
///
/// On a concave interval: the endpoint chord is a global LOWER bound (function
/// lies above its chords), and every tangent is a global UPPER bound (function
/// lies below its tangents). We use the midpoint tangent for the upper bound.
/// Both candidates are verified in f64; on failure, fall back to the band.
fn mish_relaxation_concave(l: f32, u: f32) -> LinearRelaxation {
    let max_abs_x = l.abs().max(u.abs());
    let (ls, ls_lo, _ls_hi) = mish_chord_f64(l, u);
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let (us, _us_lo, us_hi) = mish_tangent((l + u) / 2.0, max_abs_x);

    let band = mish_fallback_band(l, u);
    let lower = pick_lower(l, u, Some((ls, ls_lo)), &band);
    let upper = pick_upper(l, u, Some((us, us_hi)), &band);
    combine(lower, upper)
}

/// Crossing interval: [l, u] straddles p1 and/or p2.
///
/// Mirrors SiLU's crossing construction (silu/relaxation.rs):
/// - LOWER bound: a tangent from the convex region [p1, p2], binary-searched so
///   it stays below Mish at the concave-tail endpoints. On the convex sub-part
///   the tangent is below by convexity; verification confirms it stays below in
///   the concave tails too.
/// - UPPER bound: the endpoint chord if it is verified above Mish everywhere
///   (the cross_left / cross_both case where Mish dips below the chord); else,
///   if the interval reaches into the right concave region, a tangent from that
///   region, binary-searched to stay above Mish (cross_right case).
///
/// Anything unverifiable falls back to the corresponding band bound.
fn mish_relaxation_crossing(l: f32, u: f32, p1: f32, p2: f32) -> LinearRelaxation {
    let band = mish_fallback_band(l, u);

    // ---- Lower: convex-region tangent, binary-searched. ----
    // pick_lower keeps it only if verified AND tighter than the band.
    let lower = pick_lower(l, u, find_lower_tangent_crossing(l, u, p1, p2), &band);

    // ---- Upper: prefer chord; if it doesn't verify and the interval reaches the
    // right concave region, try a right-concave tangent. pick_upper enforces both
    // soundness (re-verifies) and tightness vs. the band, so a looser construction
    // (e.g. a steep concave tangent) can never make the upper bound worse. ----
    let (cs, _clo, chi) = mish_chord_f64(l, u);
    let upper_candidate = if verify_upper_line(l, u, cs, chi) {
        Some((cs, chi))
    } else if u > p2 {
        find_upper_tangent_crossing(l, u, p2)
    } else {
        None
    };
    let upper = pick_upper(l, u, upper_candidate, &band);

    combine(lower, upper)
}

/// Find a tangent point d in the convex region [p1, p2] ∩ [l, u] whose tangent
/// line is a valid LOWER bound for Mish on [l, u]. Mirrors SiLU's
/// `find_lower_tangent_binary`: the tangent is below Mish in the convex part by
/// convexity; binary search picks d so it also stays below at the concave-tail
/// endpoints. Returns the directed-rounded (slope, lower_intercept).
fn find_lower_tangent_crossing(l: f32, u: f32, p1: f32, p2: f32) -> Option<(f32, f32)> {
    let search_l = nan_propagating_max(l, p1);
    let search_u = nan_propagating_min(u, p2);
    if search_l >= search_u {
        return None;
    }

    // Tangent at d lies below Mish at point x?  (raw f32 tangent, f64 check)
    let tangent_below_at = |d: f32, x: f32| -> bool {
        let (ts, ti) = mish_tangent_raw(d);
        ts as f64 * x as f64 + ti as f64 <= mish_eval_f64(x as f64)
    };

    // Rightmost d in [search_l, search_u] valid at l (constraint from left tail).
    let d_max_for_l = if l >= p1 {
        search_u
    } else {
        binary_search_tangent(search_l, search_u, |d| tangent_below_at(d, l), true)?
    };
    // Leftmost d valid at u (constraint from right tail).
    let d_min_for_u = if u <= p2 {
        search_l
    } else {
        binary_search_tangent(search_l, search_u, |d| tangent_below_at(d, u), false)?
    };

    if d_min_for_u > d_max_for_l + 1.0e-6 {
        return None;
    }
    // Bit-identical tangent anchor: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let d_opt = (d_min_for_u + d_max_for_l) / 2.0;
    let max_abs_x = l.abs().max(u.abs());
    let (_s, lo_i, _hi_i) = mish_tangent(d_opt, max_abs_x);
    let (s_raw, _i_raw) = mish_tangent_raw(d_opt);
    Some((s_raw, lo_i))
}

/// Find a tangent point d in the right concave region [max(p2, l), u] whose
/// tangent line is a valid UPPER bound for Mish on [l, u]. Mirrors SiLU's
/// `find_upper_tangent_binary`. Returns directed-rounded (slope, upper_intercept).
fn find_upper_tangent_crossing(l: f32, u: f32, p2: f32) -> Option<(f32, f32)> {
    let search_l = nan_propagating_max(p2, l);
    let search_u = u;
    if search_l >= search_u {
        return None;
    }

    // Tangent at d lies above Mish at point x?
    let tangent_above_at = |d: f32, x: f32| -> bool {
        let (ts, ti) = mish_tangent_raw(d);
        ts as f64 * x as f64 + ti as f64 >= mish_eval_f64(x as f64)
    };

    // Leftmost d where tangent is above Mish at l (most constraining point on
    // the convex/left side). Larger d (further right) makes the concave tangent
    // steeper/higher on the left, so validity is monotone in d.
    if !tangent_above_at(search_u, l) {
        return None;
    }
    let d_opt = binary_search_tangent(search_l, search_u, |d| tangent_above_at(d, l), false)?;

    let max_abs_x = l.abs().max(u.abs());
    let (_s, _lo_i, hi_i) = mish_tangent(d_opt, max_abs_x);
    let (s_raw, _i_raw) = mish_tangent_raw(d_opt);
    Some((s_raw, hi_i))
}

/// Binary search for a tangent point in [lo0, hi0] satisfying `valid`.
/// `valid` must be monotone in d over the search range. If `find_rightmost`,
/// returns the largest valid d; else the smallest. Returns None if no endpoint
/// of the search range is valid in the required direction.
/// Ref: silu binary_search_tangent_below (silu/relaxation.rs:369-406).
fn binary_search_tangent(
    lo0: f32,
    hi0: f32,
    valid: impl Fn(f32) -> bool,
    find_rightmost: bool,
) -> Option<f32> {
    let check_start = if find_rightmost { lo0 } else { hi0 };
    if !valid(check_start) {
        return None;
    }
    let mut lo = lo0;
    let mut hi = hi0;
    for _ in 0..60 {
        // Bit-identical bisection anchor: f32::midpoint rounds differently at overflow edges.
        #[allow(clippy::manual_midpoint)]
        let mid = (lo + hi) / 2.0;
        if valid(mid) {
            if find_rightmost {
                lo = mid;
            } else {
                hi = mid;
            }
        } else if find_rightmost {
            hi = mid;
        } else {
            lo = mid;
        }
        if (hi - lo) < 1.0e-7 {
            break;
        }
    }
    Some(if find_rightmost { lo } else { hi })
}

/// Sound chord ± analytical-deviation band for Mish on [l, u].
///
/// This is the historical relaxation (#3146, #3285): a parallel band around the
/// endpoint chord, widened by the max/min deviation of Mish from the chord found
/// via root isolation + dense f64 grid sampling. It is always sound (the band
/// fully encloses Mish) but generally looser than the region-classified lines.
///
/// Used as the guaranteed-sound fallback whenever the region-classified
/// tangent/chord construction cannot be verified for a given bound.
///
/// Precondition: l, u finite, l < u, (u - l) >= 1e-8.
fn mish_fallback_band(l: f32, u: f32) -> LinearRelaxation {
    // Chord through endpoints (f64 to avoid catastrophic cancellation)
    // Now returns directed-rounded intercepts (#3146).
    let (chord_slope, lower_intercept, upper_intercept) = mish_chord_f64(l, u);
    // Nominal intercept for deviation measurement (midpoint of directed bounds).
    // Bit-identical band nominal: f32::midpoint rounds differently at overflow/subnormal edges.
    #[allow(clippy::manual_midpoint)]
    let nominal_intercept = (lower_intercept + upper_intercept) / 2.0;

    // h(x) = Mish(x) - chord(x); h(l) = h(u) ≈ 0 by construction.
    // Find extrema of h by locating roots of h'(x) = Mish'(x) - chord_slope = 0.
    //
    // Mish'(x) has one local minimum (near the Mish minimum at x ≈ -0.31),
    // so h'(x) = 0 has at most 2 roots. We scan sub-intervals for sign changes,
    // then refine each with bisection.

    // Track max/min of h in f64 to avoid catastrophic cancellation.
    // At large |x|, Mish(x) ≈ x and chord(x) ≈ x, so h(x) is small relative
    // to Mish(x). f32 subtraction loses significant digits here. (#3285)
    let chord_slope64 = chord_slope as f64;
    let nominal_intercept64 = nominal_intercept as f64;
    let mut max_h: f64 = 0.0;
    let mut min_h: f64 = 0.0;

    // Scan for sign changes in g(x) = Mish'(x) - chord_slope.
    // Use 200 sub-intervals — Mish' is smooth and slowly varying, so this
    // is far more than enough to isolate the at-most-2 roots.
    let n_scan = 200;
    let step = (u - l) / n_scan as f32;
    let mut prev_x = l;
    let mut prev_g = mish_derivative(l) - chord_slope;

    for i in 1..=n_scan {
        let curr_x = if i == n_scan { u } else { l + step * i as f32 };
        let curr_g = mish_derivative(curr_x) - chord_slope;

        // Sign change detected — there's a root in (prev_x, curr_x)
        if prev_g.is_sign_positive() != curr_g.is_sign_positive() {
            let root = bisect_mish_derivative_root(prev_x, curr_x, chord_slope);
            // Clamp to [l, u] for safety
            let root = root.clamp(l, u);
            // Use f64 to avoid cancellation in the deviation computation.
            let root64 = root as f64;
            let h_root = mish_eval_f64(root64) - chord_slope64 * root64 - nominal_intercept64;
            if h_root > max_h {
                max_h = h_root;
            }
            if h_root < min_h {
                min_h = h_root;
            }
        }

        prev_x = curr_x;
        prev_g = curr_g;
    }

    // Also check interior points where curvature changes could cause
    // the chord deviation to be large, even if they're not roots of h'.
    // The Mish minimum (~-0.31) and x=0 (softplus kink) are notable.
    for &x_check in &[-0.31_f32, 0.0] {
        if x_check > l && x_check < u {
            let x64 = x_check as f64;
            let h_check = mish_eval_f64(x64) - chord_slope64 * x64 - nominal_intercept64;
            if h_check > max_h {
                max_h = h_check;
            }
            if h_check < min_h {
                min_h = h_check;
            }
        }
    }

    // Dense f64 grid sampling: catches deviations when the root-finding scan
    // misses sign changes (e.g., for nearly-linear intervals at large |x| where
    // Mish'(x) - chord_slope is within f32 ULP of zero). (#3285)
    let l64 = l as f64;
    let u64 = u as f64;
    for k in 1..n_scan {
        let t = k as f64 / n_scan as f64;
        let x64 = l64 + t * (u64 - l64);
        let h_grid = mish_eval_f64(x64) - chord_slope64 * x64 - nominal_intercept64;
        if h_grid > max_h {
            max_h = h_grid;
        }
        if h_grid < min_h {
            min_h = h_grid;
        }
    }

    // Cast deviation to f32 with directed rounding: min_h rounds down, max_h rounds up.
    // This ensures the f32 deviation values are conservative. (#3285)
    let max_h_f32 = next_up_f32(max_h as f32);
    let min_h_f32 = next_down_f32(min_h as f32);

    // Lower bound: lower_intercept (sound below true chord) + min_h.
    // Upper bound: upper_intercept (sound above true chord) + max_h.
    // Apply next_down/next_up to the final f32 sum to absorb the addition rounding. (#3285)
    LinearRelaxation::new(
        chord_slope,
        next_down_f32(lower_intercept + min_h_f32),
        chord_slope,
        next_up_f32(upper_intercept + max_h_f32),
    )
}

impl MishLayer {
    /// CROWN backward propagation through Mish with pre-activation bounds.
    pub fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        debug!("Mish layer CROWN backward propagation with pre-activation bounds");
        non_finite_domain_guard("Mish", pre_activation)?;
        crown_elementwise_backward(bounds, pre_activation, mish_linear_relaxation)
    }

    /// Batched CROWN backward propagation with pre-activation bounds.
    pub fn propagate_linear_batched_with_bounds(
        &self,
        bounds: &BatchedLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<BatchedLinearBounds> {
        debug!("Mish layer batched CROWN backward propagation");
        non_finite_domain_guard("Mish", pre_activation)?;
        crown_elementwise_backward_batched(bounds, pre_activation, mish_linear_relaxation)
    }

    /// Patches CROWN backward propagation with pre-activation bounds.
    /// Part of #2613 Phase 2: generic activation Patches support.
    pub(crate) fn propagate_patches_with_bounds(
        &self,
        bounds: &crate::bounds::patches::PatchesLinearBounds,
        pre_activation: &BoundedTensor,
    ) -> Result<crate::bounds::patches::CrownBounds> {
        non_finite_domain_guard("Mish", pre_activation)?;
        crown_elementwise_backward_patches(bounds, pre_activation, mish_linear_relaxation)
    }
}

#[cfg(test)]
mod tests;
