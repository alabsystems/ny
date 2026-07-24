// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! NaN-propagating arithmetic helpers for sound bound computation.
//!
//! IEEE 754-2008 specifies that `NaN.min(x) = x` and `NaN.max(x) = x`,
//! which silently absorbs NaN into non-NaN operands. For neural network
//! verification, NaN values must propagate so downstream guards can fire
//! and produce conservative (sound) bounds.
//!
//! These helpers are placed in `ny-core` so every crate in the workspace
//! can use them without depending on `ny-propagate`.
//!
//! Reference: Issue #2577 — NaN-absorbing fold pattern across 24+ production sites.

/// NaN-propagating `min(a, b)` for fold-based lower bound computation.
///
/// Returns `a.min(b)` when both operands are non-NaN, and `NaN` if either
/// operand is `NaN`. Use as: `.fold(f32::INFINITY, nan_propagating_min)`.
///
/// # Why
///
/// `f32::min` follows IEEE 754-2008: `NaN.min(x) = x`. When folding over
/// bound-computation results, a single NaN from upstream division-by-zero
/// or `Inf − Inf` would be silently absorbed, producing an incorrect
/// (too-tight) lower bound. NaN-propagation ensures the downstream NaN
/// guard fires and widens to a conservative fallback.
#[inline]
#[must_use]
pub fn nan_propagating_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

/// NaN-propagating `max(a, b)` for fold-based upper bound computation.
///
/// Returns `a.max(b)` when both operands are non-NaN, and `NaN` if either
/// operand is `NaN`. Use as: `.fold(f32::NEG_INFINITY, nan_propagating_max)`.
///
/// # Why
///
/// `f32::max` follows IEEE 754-2008: `NaN.max(x) = x`. When folding over
/// bound-computation results, a single NaN would be silently absorbed,
/// producing an incorrect (too-tight) upper bound.
#[inline]
#[must_use]
pub fn nan_propagating_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// NaN-propagating `min(a, b)` for `f64` fold-based computation.
///
/// Identical semantics to [`nan_propagating_min`] but operates on `f64` values.
/// Use as: `.fold(f64::INFINITY, nan_propagating_min_f64)`.
///
/// Reference: Issue #2616 — extend NaN-safe fold pattern to f64 sites.
#[inline]
#[must_use]
pub fn nan_propagating_min_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// NaN-propagating `max(a, b)` for `f64` fold-based computation.
///
/// Identical semantics to [`nan_propagating_max`] but operates on `f64` values.
/// Use as: `.fold(f64::NEG_INFINITY, nan_propagating_max_f64)`.
///
/// Reference: Issue #2616 — extend NaN-safe fold pattern to f64 sites.
#[inline]
#[must_use]
pub fn nan_propagating_max_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// NaN-propagating `max(a, 0.0)` for positive/negative coefficient split.
///
/// Returns `a` if `a >= 0.0`, `0.0` if `a < 0.0`, and `NaN` if `a` is NaN.
/// Unlike `f32::max(a, 0.0)` which returns `0.0` when `a` is NaN (IEEE 754-2008),
/// this preserves NaN so downstream NaN guards fire.
///
/// Used in CROWN A-matrix coefficient splitting: positive coefficients multiply
/// upper bounds, negative coefficients multiply lower bounds. A NaN coefficient
/// must poison the accumulator, not silently become zero.
///
/// Reference: Issue #2415 — NaN A-matrix coefficients silently become zero.
#[inline]
#[must_use]
pub fn nan_propagating_max_zero(a: f32) -> f32 {
    if a.is_nan() {
        f32::NAN
    } else if a > 0.0 {
        a
    } else {
        0.0
    }
}

/// NaN-propagating `min(a, 0.0)` for positive/negative coefficient split.
///
/// Mirror of [`nan_propagating_max_zero`] for the negative-coefficient branch.
/// Returns `a` if `a <= 0.0`, `0.0` if `a > 0.0`, and `NaN` if `a` is NaN.
///
/// Reference: Issue #2415.
#[inline]
#[must_use]
pub fn nan_propagating_min_zero(a: f32) -> f32 {
    if a.is_nan() {
        f32::NAN
    } else if a < 0.0 {
        a
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_propagating_min_normal() {
        assert_eq!(nan_propagating_min(1.0, 2.0), 1.0);
        assert_eq!(nan_propagating_min(2.0, 1.0), 1.0);
        assert_eq!(nan_propagating_min(-1.0, 1.0), -1.0);
    }

    #[test]
    fn test_nan_propagating_min_nan_propagates() {
        let r = nan_propagating_min(f32::NAN, 1.0);
        assert!(r.is_nan(), "NaN left operand must propagate, got {r}");
        let r = nan_propagating_min(1.0, f32::NAN);
        assert!(r.is_nan(), "NaN right operand must propagate, got {r}");
        let r = nan_propagating_min(f32::NAN, f32::NAN);
        assert!(r.is_nan(), "both NaN must propagate, got {r}");
    }

    #[test]
    fn test_nan_propagating_min_infinity() {
        assert_eq!(nan_propagating_min(f32::INFINITY, 1.0), 1.0);
        assert_eq!(
            nan_propagating_min(f32::NEG_INFINITY, 1.0),
            f32::NEG_INFINITY
        );
    }

    #[test]
    fn test_nan_propagating_max_normal() {
        assert_eq!(nan_propagating_max(1.0, 2.0), 2.0);
        assert_eq!(nan_propagating_max(2.0, 1.0), 2.0);
        assert_eq!(nan_propagating_max(-1.0, 1.0), 1.0);
    }

    #[test]
    fn test_nan_propagating_max_nan_propagates() {
        let r = nan_propagating_max(f32::NAN, 1.0);
        assert!(r.is_nan(), "NaN left operand must propagate, got {r}");
        let r = nan_propagating_max(1.0, f32::NAN);
        assert!(r.is_nan(), "NaN right operand must propagate, got {r}");
        let r = nan_propagating_max(f32::NAN, f32::NAN);
        assert!(r.is_nan(), "both NaN must propagate, got {r}");
    }

    #[test]
    fn test_nan_propagating_max_infinity() {
        assert_eq!(nan_propagating_max(f32::NEG_INFINITY, 1.0), 1.0);
        assert_eq!(nan_propagating_max(f32::INFINITY, 1.0), f32::INFINITY);
    }

    #[test]
    fn test_fold_pattern_min() {
        let vals = [3.0, 1.0, 2.0];
        let result = vals
            .iter()
            .copied()
            .fold(f32::INFINITY, nan_propagating_min);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_fold_pattern_min_with_nan() {
        let vals = [3.0, f32::NAN, 2.0];
        let result = vals
            .iter()
            .copied()
            .fold(f32::INFINITY, nan_propagating_min);
        assert!(
            result.is_nan(),
            "fold with NaN element must produce NaN, got {result}"
        );
    }

    #[test]
    fn test_fold_pattern_max() {
        let vals = [1.0, 3.0, 2.0];
        let result = vals
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, nan_propagating_max);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_fold_pattern_max_with_nan() {
        let vals = [1.0, f32::NAN, 2.0];
        let result = vals
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, nan_propagating_max);
        assert!(
            result.is_nan(),
            "fold with NaN element must produce NaN, got {result}"
        );
    }

    // --- nan_propagating_min_f64 / nan_propagating_max_f64 tests ---

    #[test]
    fn test_nan_propagating_min_f64_normal() {
        assert_eq!(nan_propagating_min_f64(1.0, 2.0), 1.0);
        assert_eq!(nan_propagating_min_f64(2.0, 1.0), 1.0);
        assert_eq!(nan_propagating_min_f64(-1.0, 1.0), -1.0);
    }

    #[test]
    fn test_nan_propagating_min_f64_nan_propagates() {
        let r = nan_propagating_min_f64(f64::NAN, 1.0);
        assert!(r.is_nan(), "NaN left operand must propagate, got {r}");
        let r = nan_propagating_min_f64(1.0, f64::NAN);
        assert!(r.is_nan(), "NaN right operand must propagate, got {r}");
        let r = nan_propagating_min_f64(f64::NAN, f64::NAN);
        assert!(r.is_nan(), "both NaN must propagate, got {r}");
    }

    #[test]
    fn test_nan_propagating_max_f64_normal() {
        assert_eq!(nan_propagating_max_f64(1.0, 2.0), 2.0);
        assert_eq!(nan_propagating_max_f64(2.0, 1.0), 2.0);
        assert_eq!(nan_propagating_max_f64(-1.0, 1.0), 1.0);
    }

    #[test]
    fn test_nan_propagating_max_f64_nan_propagates() {
        let r = nan_propagating_max_f64(f64::NAN, 1.0);
        assert!(r.is_nan(), "NaN left operand must propagate, got {r}");
        let r = nan_propagating_max_f64(1.0, f64::NAN);
        assert!(r.is_nan(), "NaN right operand must propagate, got {r}");
        let r = nan_propagating_max_f64(f64::NAN, f64::NAN);
        assert!(r.is_nan(), "both NaN must propagate, got {r}");
    }

    #[test]
    fn test_fold_pattern_min_f64() {
        let vals = [3.0_f64, 1.0, 2.0];
        let result = vals
            .iter()
            .copied()
            .fold(f64::INFINITY, nan_propagating_min_f64);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_fold_pattern_min_f64_with_nan() {
        let vals = [3.0_f64, f64::NAN, 2.0];
        let result = vals
            .iter()
            .copied()
            .fold(f64::INFINITY, nan_propagating_min_f64);
        assert!(
            result.is_nan(),
            "f64 fold with NaN element must produce NaN, got {result}"
        );
    }

    #[test]
    fn test_fold_pattern_max_f64() {
        let vals = [1.0_f64, 3.0, 2.0];
        let result = vals
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, nan_propagating_max_f64);
        assert_eq!(result, 3.0);
    }

    #[test]
    fn test_fold_pattern_max_f64_with_nan() {
        let vals = [1.0_f64, f64::NAN, 2.0];
        let result = vals
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, nan_propagating_max_f64);
        assert!(
            result.is_nan(),
            "f64 fold with NaN element must produce NaN, got {result}"
        );
    }

    // --- nan_propagating_max_zero tests ---

    #[test]
    fn test_nan_propagating_max_zero_positive() {
        assert_eq!(nan_propagating_max_zero(3.0), 3.0);
        assert_eq!(nan_propagating_max_zero(0.5), 0.5);
    }

    #[test]
    fn test_nan_propagating_max_zero_negative() {
        assert_eq!(nan_propagating_max_zero(-3.0), 0.0);
        assert_eq!(nan_propagating_max_zero(-0.5), 0.0);
    }

    #[test]
    fn test_nan_propagating_max_zero_zero() {
        assert_eq!(nan_propagating_max_zero(0.0), 0.0);
        assert_eq!(nan_propagating_max_zero(-0.0), 0.0);
    }

    #[test]
    fn test_nan_propagating_max_zero_nan_propagates() {
        let r = nan_propagating_max_zero(f32::NAN);
        assert!(r.is_nan(), "max_zero(NaN) must propagate NaN, got {r}");
    }

    #[test]
    fn test_nan_propagating_max_zero_inf() {
        assert_eq!(nan_propagating_max_zero(f32::INFINITY), f32::INFINITY);
        assert_eq!(nan_propagating_max_zero(f32::NEG_INFINITY), 0.0);
    }

    // --- nan_propagating_min_zero tests ---

    #[test]
    fn test_nan_propagating_min_zero_negative() {
        assert_eq!(nan_propagating_min_zero(-3.0), -3.0);
        assert_eq!(nan_propagating_min_zero(-0.5), -0.5);
    }

    #[test]
    fn test_nan_propagating_min_zero_positive() {
        assert_eq!(nan_propagating_min_zero(3.0), 0.0);
        assert_eq!(nan_propagating_min_zero(0.5), 0.0);
    }

    #[test]
    fn test_nan_propagating_min_zero_zero() {
        assert_eq!(nan_propagating_min_zero(0.0), 0.0);
        assert_eq!(nan_propagating_min_zero(-0.0), 0.0);
    }

    #[test]
    fn test_nan_propagating_min_zero_nan_propagates() {
        let r = nan_propagating_min_zero(f32::NAN);
        assert!(r.is_nan(), "min_zero(NaN) must propagate NaN, got {r}");
    }

    #[test]
    fn test_nan_propagating_min_zero_inf() {
        assert_eq!(
            nan_propagating_min_zero(f32::NEG_INFINITY),
            f32::NEG_INFINITY
        );
        assert_eq!(nan_propagating_min_zero(f32::INFINITY), 0.0);
    }
}
