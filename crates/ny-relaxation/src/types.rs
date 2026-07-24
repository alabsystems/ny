// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Core types for linear relaxation bounds.

/// Per-neuron linear relaxation bounds for an elementwise activation.
///
/// Represents: `lower_slope * x + lower_intercept <= f(x) <= upper_slope * x + upper_intercept`
/// for `x` in `[l, u]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearRelaxation {
    pub lower_slope: f32,
    pub lower_intercept: f32,
    pub upper_slope: f32,
    pub upper_intercept: f32,
}

impl LinearRelaxation {
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

    /// Maximally loose relaxation for NaN/invalid inputs.
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

/// GELU approximation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluApproximation {
    /// Exact: `0.5 * x * (1 + erf(x / sqrt(2)))`.
    Erf,
    /// Approximate: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    Tanh,
}
