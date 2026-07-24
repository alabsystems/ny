// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared linear relaxation for ELU-family activations (ELU, SELU, CELU).
//!
//! Extracts the common algorithmic skeleton parameterized by the mathematical
//! expressions that differ between the three activations. Each activation
//! creates its `EluFamilyParams` and delegates to `elu_family_linear_relaxation`.
//!
//! Part of #2834.
//! Design: designs/archive/2026-02-18-code-structure-wave22-elu-family-alpha-loop-dedup.md
//! Ref: alpha-beta-CROWN auto_LiRPA/operators/nonlinear.py (ELU/SELU/CELU relaxation)

use ny_core::{nan_propagating_max_f64, nan_propagating_min_f64};
use ny_tensor::{next_down_f32, next_up_f32};

use super::LinearRelaxation;
use crate::bounds::{nan_propagating_max, nan_propagating_min};

/// Parameters for an ELU-family activation's linear relaxation.
///
/// The family shares the structure: f(x) = positive_slope * x for x >= 0,
/// f(x) = eval_negative(x) for x < 0, with known derivative and saturation.
pub(crate) struct EluFamilyParams {
    /// Slope of the positive branch (1.0 for ELU/CELU, lambda for SELU).
    pub positive_slope: f64,
    /// f(x) for x < 0, computed in f64 for precision.
    pub eval_negative: fn(f64, &EluFamilyParams) -> f64,
    /// f'(x) for x < 0, computed in f64.
    pub deriv_negative: fn(f64, &EluFamilyParams) -> f64,
    /// Saturation value: lim(x -> -inf) f(x) (e.g., -alpha for ELU).
    pub saturation: f32,
    /// Primary scale for critical point formula: x_crit = ln(chord_slope / scale).
    /// (alpha for ELU, lambda*alpha for SELU, unused for CELU.)
    pub scale: f64,
    /// Secondary parameter (alpha for CELU's exp(x/alpha), alpha for ELU/SELU).
    pub alpha: f64,
    /// If true, use chord upper + tangent lower in crossing region (globally convex).
    /// If false, use chord + deviation (derivative drops at x=0).
    pub globally_convex: bool,
}

/// Shared linear relaxation for ELU-family activations on interval [l, u].
///
/// Returns bounds such that:
///   lower_slope * x + lower_intercept <= f(x) <= upper_slope * x + upper_intercept
/// for all x in [l, u].
///
/// Callers handle parameter validation (e.g., alpha > 0 guard) before calling.
/// This function handles: NaN bounds, infinite bounds, f64 promotion, point intervals,
/// purely-positive, purely-negative (chord upper + tangent lower), and crossing
/// (chord+deviation or chord+tangent depending on `globally_convex`).
///
/// Ref: elu.rs, selu.rs, celu.rs (pre-dedup implementations).
pub(crate) fn elu_family_linear_relaxation(
    l: f32,
    u: f32,
    params: &EluFamilyParams,
) -> LinearRelaxation {
    let positive_slope_f32 = params.positive_slope as f32;

    // Guard: NaN bounds → (-inf, +inf) intercepts so CROWN drives bounds to ±inf.
    if l.is_nan() || u.is_nan() {
        return LinearRelaxation::nan_fallback();
    }

    // Guard: infinite bounds cause inf/inf = NaN in chord computation.
    if l.is_infinite() && u.is_infinite() {
        return LinearRelaxation::new(0.0, params.saturation, 0.0, f32::INFINITY);
    }
    if l.is_infinite() {
        // l = -inf, u finite: f(-inf) -> saturation, f(u) = fu.
        let fu = if u >= 0.0 {
            positive_slope_f32 * u
        } else {
            (params.eval_negative)(u as f64, params) as f32
        };
        // NaN-propagating: if fu is NaN, propagate instead of absorbing via IEEE min/max. (#2714)
        return LinearRelaxation::new(
            0.0,
            nan_propagating_min(params.saturation, fu),
            0.0,
            nan_propagating_max(fu, 0.0),
        );
    }
    if u.is_infinite() {
        // l finite, u = +inf.
        if l >= 0.0 {
            return LinearRelaxation::new(positive_slope_f32, 0.0, 0.0, f32::INFINITY);
        }
        // Lower: tangent at l, if slope <= positive_slope (otherwise tangent overtakes f).
        let fl = (params.eval_negative)(l as f64, params);
        let dl = (params.deriv_negative)(l as f64, params);
        let intercept = fl - dl * (l as f64);

        if dl <= params.positive_slope {
            let dl_f32 = dl as f32;
            let dl_err = next_up_f32(((dl - dl_f32 as f64).abs() * l.abs() as f64) as f32);
            return LinearRelaxation::new(
                dl_f32,
                next_down_f32((intercept as f32) - dl_err),
                0.0,
                f32::INFINITY,
            );
        }
        // Tangent slope > positive_slope: global lower bound y = positive_slope*x + saturation.
        return LinearRelaxation::new(positive_slope_f32, params.saturation, 0.0, f32::INFINITY);
    }

    // f64 intermediates to prevent catastrophic cancellation (#1745).
    let l64 = l as f64;
    let u64 = u as f64;

    let eval64 = |x: f64| -> f64 {
        if x >= 0.0 {
            params.positive_slope * x
        } else {
            (params.eval_negative)(x, params)
        }
    };

    // Point interval: tangent at l.
    if (u64 - l64).abs() < 1e-8 {
        let y = eval64(l64);
        let slope = if l >= 0.0 {
            params.positive_slope
        } else {
            (params.deriv_negative)(l64, params)
        };
        let intercept = y - slope * l64;
        let slope_f32 = slope as f32;
        let slope_err =
            next_up_f32(((slope - slope_f32 as f64).abs() * l.abs().max(u.abs()) as f64) as f32);
        return LinearRelaxation::new(
            slope_f32,
            next_down_f32((intercept as f32) - slope_err),
            slope_f32,
            next_up_f32((intercept as f32) + slope_err),
        );
    }

    let fl = eval64(l64);
    let fu = eval64(u64);

    // Case 1: purely positive — linear.
    if l >= 0.0 {
        if (params.positive_slope - 1.0).abs() < 1e-12 {
            return LinearRelaxation::identity();
        }
        return LinearRelaxation::new(positive_slope_f32, 0.0, positive_slope_f32, 0.0);
    }

    // Case 2: purely negative — convex region.
    // Upper: chord (secant). Lower: tangent.
    if u <= 0.0 {
        let upper_slope = (fu - fl) / (u64 - l64);
        let upper_intercept = fl - upper_slope * l64;

        // Lower bound: tangent to the convex negative branch. Any tangent is a
        // global lower bound (convexity); the parallel-to-chord tangent (where
        // the negative-branch derivative equals the chord slope) is the tightest
        // and, by the MVT plus monotone derivative, lies in [l, u].
        //
        // For ELU/SELU the negative branch is scale*(exp(x)-1) with derivative
        // scale*exp(x), so deriv_negative(d) = chord_slope solves in closed form
        // as d = ln(chord_slope / scale). This closed form is specific to the
        // scale*exp(x) derivative; CELU (globally_convex = true) uses a different
        // negative branch (exp(x/alpha)) for which this d is NOT the tangent
        // point, so it is excluded and falls back to the midpoint tangent.
        //
        // Guard: scale > 0 && chord_slope > 0 && d in [l, u]; otherwise fall back
        // to the midpoint tangent (also a valid lower bound by convexity).
        let mut lower_point = f64::midpoint(l64, u64);
        if !params.globally_convex && params.scale > 0.0 && upper_slope > 0.0 {
            let d = (upper_slope / params.scale).ln();
            if d.is_finite() && d >= l64 && d <= u64 {
                lower_point = d;
            }
        }
        let m = lower_point;
        let lower_slope = (params.deriv_negative)(m, params);
        let fm = eval64(m);
        let lower_intercept = fm - lower_slope * m;

        let max_abs_x = l.abs().max(u.abs()) as f64;
        let ls_f32 = lower_slope as f32;
        let us_f32 = upper_slope as f32;
        let ls_err = next_up_f32(((lower_slope - ls_f32 as f64).abs() * max_abs_x) as f32);
        let us_err = next_up_f32(((upper_slope - us_f32 as f64).abs() * max_abs_x) as f32);
        return LinearRelaxation::new(
            ls_f32,
            next_down_f32((lower_intercept as f32) - ls_err),
            us_f32,
            next_up_f32((upper_intercept as f32) + us_err),
        );
    }

    // Case 3: crossing region (l < 0 <= u).
    let chord_slope = (fu - fl) / (u64 - l64);
    let chord_intercept = fl - chord_slope * l64;

    if params.globally_convex {
        // Globally convex (CELU): chord upper, tangent at midpoint of [l, 0] lower.
        let m = l64 / 2.0;
        let lower_slope = (params.deriv_negative)(m, params);
        let fm = eval64(m);
        let lower_intercept = fm - lower_slope * m;

        let max_abs_x = l.abs().max(u.abs()) as f64;
        let ls_f32 = lower_slope as f32;
        let us_f32 = chord_slope as f32;
        let ls_err = next_up_f32(((lower_slope - ls_f32 as f64).abs() * max_abs_x) as f32);
        let us_err = next_up_f32(((chord_slope - us_f32 as f64).abs() * max_abs_x) as f32);
        LinearRelaxation::new(
            ls_f32,
            next_down_f32((lower_intercept as f32) - ls_err),
            us_f32,
            next_up_f32((chord_intercept as f32) + us_err),
        )
    } else {
        // Not globally convex (ELU, SELU): chord + analytical deviation.
        let h_at_0 = -chord_intercept;
        let mut max_h = nan_propagating_max_f64(h_at_0, 0.0);
        let mut min_h = nan_propagating_min_f64(h_at_0, 0.0);

        // Critical point in negative piece: deriv(x_crit) == chord_slope
        // => x_crit = ln(chord_slope / scale)
        if chord_slope > 0.0 && params.scale > 0.0 {
            let ratio = chord_slope / params.scale;
            if ratio > 0.0 {
                let x_crit = ratio.ln();
                if x_crit > l64 && x_crit < 0.0 {
                    let h_crit = eval64(x_crit) - chord_slope * x_crit - chord_intercept;
                    max_h = nan_propagating_max_f64(max_h, h_crit);
                    min_h = nan_propagating_min_f64(min_h, h_crit);
                }
            }
        }

        let max_abs_x = l.abs().max(u.abs()) as f64;
        let cs_f32 = chord_slope as f32;
        let cs_err = next_up_f32(((chord_slope - cs_f32 as f64).abs() * max_abs_x) as f32);
        LinearRelaxation::new(
            cs_f32,
            next_down_f32(((chord_intercept + min_h) as f32) - cs_err),
            cs_f32,
            next_up_f32(((chord_intercept + max_h) as f32) + cs_err),
        )
    }
}
