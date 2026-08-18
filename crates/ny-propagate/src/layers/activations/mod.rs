// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Activations layers for bound propagation.

use ny_tensor::next_up_f32;

/// Per-neuron linear relaxation bounds for an elementwise activation.
///
/// Represents: `lower_slope * x + lower_intercept <= f(x) <= upper_slope * x + upper_intercept`
/// for `x` in `[l, u]`.
///
/// Replaces bare `(f32, f32, f32, f32)` tuples — named fields prevent silent field swaps
/// that would produce unsound bounds. Part of #2978.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LinearRelaxation {
    /// Slope of the lower bounding line.
    pub lower_slope: f32,
    /// Intercept of the lower bounding line.
    pub lower_intercept: f32,
    /// Slope of the upper bounding line.
    pub upper_slope: f32,
    /// Intercept of the upper bounding line.
    pub upper_intercept: f32,
}

impl LinearRelaxation {
    /// Construct a new `LinearRelaxation` with named fields.
    #[inline]
    pub fn new(
        lower_slope: f32,
        lower_intercept: f32,
        upper_slope: f32,
        upper_intercept: f32,
    ) -> Self {
        Self {
            lower_slope,
            lower_intercept,
            upper_slope,
            upper_intercept,
        }
    }

    /// Identity relaxation: `f(x) = x` (both bounds are the identity line).
    #[inline]
    pub fn identity() -> Self {
        Self {
            lower_slope: 1.0,
            lower_intercept: 0.0,
            upper_slope: 1.0,
            upper_intercept: 0.0,
        }
    }

    /// Zero relaxation: `f(x) = 0` (both bounds are the zero line).
    #[inline]
    pub fn zero() -> Self {
        Self {
            lower_slope: 0.0,
            lower_intercept: 0.0,
            upper_slope: 0.0,
            upper_intercept: 0.0,
        }
    }

    /// Constant relaxation: lower bound at `lower`, upper bound at `upper`.
    #[inline]
    pub fn constant(lower: f32, upper: f32) -> Self {
        Self {
            lower_slope: 0.0,
            lower_intercept: lower,
            upper_slope: 0.0,
            upper_intercept: upper,
        }
    }

    /// NaN-safe fallback: slopes are zero, intercepts are ±infinity.
    ///
    /// Used when input bounds contain NaN — drives CROWN output bounds to ±infinity (sound).
    #[inline]
    pub fn nan_fallback() -> Self {
        Self {
            lower_slope: 0.0,
            lower_intercept: f32::NEG_INFINITY,
            upper_slope: 0.0,
            upper_intercept: f32::INFINITY,
        }
    }
}

/// Compute the ReLU upper chord for a finite crossing interval `l < 0 < u`.
///
/// The width is formed in `f64` so `u - l` does not overflow to `+inf` for large
/// opposite-signed finite endpoints. When a floor is provided, it matches the
/// baseline's minimum-width relaxation for the scalar helper path.
#[inline]
pub(crate) fn relu_crossing_upper_chord(l: f32, u: f32, min_width: Option<f32>) -> (f32, f32) {
    debug_assert!(l.is_finite() && u.is_finite() && l < 0.0 && u > 0.0);

    let exact_width = (u as f64) - (l as f64);
    debug_assert!(exact_width.is_finite() && exact_width > 0.0);
    // SOUNDNESS (false-proof fix, audit 2026-06-27): the upper chord encloses ReLU on
    // [l,u] iff its slope λ ≥ u/(u−l) — the tight chord through (l,0) and (u,u). The former
    // `max(exact_width, floor)` WIDENS the denominator, LOWERING λ below u/(u−l), so for a
    // crossing interval narrower than the floor (e.g. u−l < 1e-8) the chord drops below
    // ReLU(u)=u → a certified upper bound under the true value (false `VERIFIED`). There is
    // no overflow the floor was guarding against: for l<0<u, u/(u−l) ∈ (0,1) and exact_width
    // is finite f64. (The α,β-CROWN `+1e-8` is a float regularizer, not sound for the 0-wrong
    // moat.) Always use the exact width; `min_width` is intentionally ignored for the chord.
    let _ = min_width;
    let width = exact_width;

    // Round the stored upper chord upward after the f64 computation so the cast
    // back to f32 stays conservative for both slope and intercept.
    let lambda = next_up_f32((u as f64 / width) as f32);
    let lambda_intercept = next_up_f32((-(lambda as f64) * (l as f64)) as f32);
    (lambda, lambda_intercept)
}

/// Bridge conversion: legacy `(lower_slope, lower_intercept, upper_slope, upper_intercept)` tuple
/// to the validated struct. Used during the #2978 migration for call sites not yet converted.
impl From<(f32, f32, f32, f32)> for LinearRelaxation {
    #[inline]
    fn from(
        (lower_slope, lower_intercept, upper_slope, upper_intercept): (f32, f32, f32, f32),
    ) -> Self {
        Self {
            lower_slope,
            lower_intercept,
            upper_slope,
            upper_intercept,
        }
    }
}

/// Reverse conversion: `LinearRelaxation` to legacy tuple for callers not yet migrated.
impl From<LinearRelaxation> for (f32, f32, f32, f32) {
    #[inline]
    fn from(r: LinearRelaxation) -> Self {
        (
            r.lower_slope,
            r.lower_intercept,
            r.upper_slope,
            r.upper_intercept,
        )
    }
}

pub(crate) mod celu;
mod clip;
pub(crate) mod elu;
pub(crate) mod elu_family;
#[cfg(test)]
mod envelope_audit;
#[cfg(test)]
mod envelope_audit_sigmoid;
pub(crate) mod exp;
mod hard_sigmoid;
mod hard_swish;
mod leaky_relu;
pub(crate) mod log;
mod mish;
mod prelu;
pub(crate) mod relu;
mod selu;
mod shrink;
mod silu;
pub(crate) mod snake;
mod softsign;
mod thresholded_relu;
pub(crate) mod validate;

pub use celu::CeluLayer;
pub use clip::ClipLayer;
pub use elu::EluLayer;
pub use exp::exp_linear_relaxation;
pub use exp::ExpLayer;
pub use hard_sigmoid::HardSigmoidLayer;
pub use hard_swish::HardSwishLayer;
pub use leaky_relu::LeakyReLULayer;
pub use log::log_linear_relaxation;
pub use log::LogLayer;
pub use mish::MishLayer;
pub use prelu::PReluLayer;
pub use relu::ReLULayer;
pub(crate) use relu::RELU_RELAX_MIN_WIDTH;
pub use selu::SeluLayer;
pub use shrink::ShrinkLayer;
pub use silu::SiLULayer;
pub use silu::{silu_eval, silu_sound_linear_relaxation};
pub use snake::SnakeLayer;
pub use softsign::SoftsignLayer;
pub use thresholded_relu::ThresholdedReluLayer;

#[cfg(test)]
pub(crate) fn silu_critical_point() -> f32 {
    silu::silu_critical_point()
}
