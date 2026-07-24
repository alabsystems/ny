// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared CROWN backward composition: coefficient × relaxation slope/intercept
//! with directed rounding and non-finite tracking.
//!
//! This module deduplicates the inner loop that was previously copy-pasted
//! across `crown_elementwise_backward_indexed`, `crown_elementwise_backward_batched_indexed`,
//! and `crown_elementwise_backward_patches` in `common/mod.rs`.
//!
//! Reference: designs/2026-03-01-common-mod-crown-backward-dedup-and-split.md

use ny_tensor::{next_down_f32, next_up_f32};

use crate::layers::activations::LinearRelaxation;

/// Result of composing one coefficient with a relaxation for one bound direction.
pub(crate) struct ComposeResult {
    /// New A-matrix coefficient after relaxation composition.
    pub(crate) new_coeff: f32,
    /// Intercept contribution to accumulate in f64 bias.
    pub(crate) intercept_contrib: f64,
    /// Whether the coefficient × slope product overflowed to non-finite.
    pub(crate) nonfinite: bool,
}

impl ComposeResult {
    /// Zero coefficient — no contribution, no non-finite flag.
    pub(crate) const ZERO: Self = Self {
        new_coeff: 0.0,
        intercept_contrib: 0.0,
        nonfinite: false,
    };
}

/// Compose one coefficient with a relaxation for the **lower** bound direction.
///
/// For the lower bound:
/// - Positive coefficient `a > 0`: use *lower* relaxation (preserves direction)
/// - Negative coefficient `a < 0`: use *upper* relaxation (flips direction)
/// - Zero coefficient: skip to avoid IEEE 754 NaN from `0.0 * ±Inf` (#1736)
///
/// Directed rounding (#2786): `next_down_f32` moves the product away from the
/// true value toward -∞, making the lower bound sound (conservative).
///
/// Non-finite detection (#3009): when `coeff × slope` overflows to ±Inf,
/// the coefficient is zeroed and the nonfinite flag is set for row fallback.
#[inline(always)]
pub(crate) fn compose_lower(coeff: f32, relax: &LinearRelaxation) -> ComposeResult {
    if coeff > 0.0 {
        let product = coeff * relax.lower_slope;
        ComposeResult {
            new_coeff: if product.is_finite() {
                next_down_f32(product)
            } else {
                0.0
            },
            intercept_contrib: coeff as f64 * relax.lower_intercept as f64,
            nonfinite: !product.is_finite(),
        }
    } else if coeff < 0.0 {
        let product = coeff * relax.upper_slope;
        ComposeResult {
            new_coeff: if product.is_finite() {
                next_down_f32(product)
            } else {
                0.0
            },
            intercept_contrib: coeff as f64 * relax.upper_intercept as f64,
            nonfinite: !product.is_finite(),
        }
    } else {
        ComposeResult::ZERO
    }
}

/// Compose one coefficient with a relaxation for the **upper** bound direction.
///
/// For the upper bound:
/// - Positive coefficient `a > 0`: use *upper* relaxation (preserves direction)
/// - Negative coefficient `a < 0`: use *lower* relaxation (flips direction)
/// - Zero coefficient: skip to avoid IEEE 754 NaN from `0.0 * ±Inf` (#1736)
///
/// Directed rounding (#2786): `next_up_f32` moves the product away from the
/// true value toward +∞, making the upper bound sound (conservative).
///
/// Non-finite detection (#3009): same as [`compose_lower`].
#[inline(always)]
pub(crate) fn compose_upper(coeff: f32, relax: &LinearRelaxation) -> ComposeResult {
    if coeff > 0.0 {
        let product = coeff * relax.upper_slope;
        ComposeResult {
            new_coeff: if product.is_finite() {
                next_up_f32(product)
            } else {
                0.0
            },
            intercept_contrib: coeff as f64 * relax.upper_intercept as f64,
            nonfinite: !product.is_finite(),
        }
    } else if coeff < 0.0 {
        let product = coeff * relax.lower_slope;
        ComposeResult {
            new_coeff: if product.is_finite() {
                next_up_f32(product)
            } else {
                0.0
            },
            intercept_contrib: coeff as f64 * relax.lower_intercept as f64,
            nonfinite: !product.is_finite(),
        }
    } else {
        ComposeResult::ZERO
    }
}

/// Precompute per-neuron [`LinearRelaxation`] values from pre-activation bounds.
///
/// Calls `relaxation_fn(l, u, i)` for each neuron `i` where `l` and `u` are
/// the lower and upper pre-activation bounds. The `i` index supports per-neuron
/// parameters (e.g., PReLU per-channel slopes).
pub(crate) fn precompute_relaxations<F>(
    pre_lower: &[f32],
    pre_upper: &[f32],
    relaxation_fn: &F,
) -> Vec<LinearRelaxation>
where
    F: Fn(f32, f32, usize) -> LinearRelaxation,
{
    pre_lower
        .iter()
        .zip(pre_upper.iter())
        .enumerate()
        .map(|(i, (&l, &u))| relaxation_fn(l, u, i))
        .collect()
}

/// Log non-finite row fallback statistics at debug level.
///
/// Called after the main composition loop when non-finite coefficient × slope
/// products were detected. The affected rows will get maximally loose but
/// sound bounds: lower = -Inf, upper = +Inf.
///
/// #3009: Pattern shared across Dense, Batched, and Patches paths.
pub(crate) fn log_nonfinite_fallback(
    label: &str,
    lower_affected: usize,
    upper_affected: usize,
    total_rows: usize,
) {
    if lower_affected > 0 || upper_affected > 0 {
        tracing::debug!(
            "{label} CROWN backward: non-finite coeff×slope overflow in \
             {lower_affected}/{total_rows} lower rows, \
             {upper_affected}/{total_rows} upper rows — \
             falling back to ±Inf bias for affected rows"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_tensor::{next_down_f32, next_up_f32};

    fn relu_relaxation() -> LinearRelaxation {
        // ReLU relaxation for input in [-1, 2]:
        // lower: slope = 2/3, intercept = 2/3
        // upper: slope = 1, intercept = 0
        LinearRelaxation::new(2.0 / 3.0, 2.0 / 3.0, 1.0, 0.0)
    }

    // --- compose_lower: zero coefficient ---

    #[test]
    fn test_compose_lower_zero_coeff_returns_zero() {
        let r = compose_lower(0.0, &relu_relaxation());
        assert_eq!(r.new_coeff, 0.0);
        assert_eq!(r.intercept_contrib, 0.0);
        assert!(!r.nonfinite);
    }

    #[test]
    fn test_compose_lower_negative_zero_returns_zero() {
        // IEEE 754: -0.0 > 0.0 is false, -0.0 < 0.0 is false → falls through to ZERO
        let r = compose_lower(-0.0, &relu_relaxation());
        assert_eq!(r.new_coeff, 0.0);
        assert_eq!(r.intercept_contrib, 0.0);
        assert!(!r.nonfinite);
    }

    // --- compose_lower: positive coefficient ---

    #[test]
    fn test_compose_lower_positive_coeff_uses_lower_relaxation() {
        let relax = LinearRelaxation::new(0.5, 1.0, 0.8, 2.0);
        let r = compose_lower(3.0, &relax);
        // positive coeff → lower relaxation: 3.0 * 0.5 = 1.5
        assert_eq!(r.new_coeff, next_down_f32(3.0 * 0.5));
        // intercept: 3.0 * 1.0 = 3.0 (in f64)
        assert_eq!(r.intercept_contrib, 3.0f64 * 1.0f64);
        assert!(!r.nonfinite);
    }

    // --- compose_lower: negative coefficient ---

    #[test]
    fn test_compose_lower_negative_coeff_uses_upper_relaxation() {
        let relax = LinearRelaxation::new(0.5, 1.0, 0.8, 2.0);
        let r = compose_lower(-3.0, &relax);
        // negative coeff → upper relaxation: -3.0 * 0.8 = -2.4
        assert_eq!(r.new_coeff, next_down_f32(-3.0 * 0.8));
        // intercept: -3.0 * 2.0 = -6.0 (in f64)
        assert_eq!(r.intercept_contrib, -3.0f64 * 2.0f64);
        assert!(!r.nonfinite);
    }

    // --- compose_upper: zero coefficient ---

    #[test]
    fn test_compose_upper_zero_coeff_returns_zero() {
        let r = compose_upper(0.0, &relu_relaxation());
        assert_eq!(r.new_coeff, 0.0);
        assert_eq!(r.intercept_contrib, 0.0);
        assert!(!r.nonfinite);
    }

    #[test]
    fn test_compose_upper_negative_zero_returns_zero() {
        let r = compose_upper(-0.0, &relu_relaxation());
        assert_eq!(r.new_coeff, 0.0);
        assert_eq!(r.intercept_contrib, 0.0);
        assert!(!r.nonfinite);
    }

    // --- compose_upper: positive coefficient ---

    #[test]
    fn test_compose_upper_positive_coeff_uses_upper_relaxation() {
        let relax = LinearRelaxation::new(0.5, 1.0, 0.8, 2.0);
        let r = compose_upper(3.0, &relax);
        // positive coeff → upper relaxation: 3.0 * 0.8 = 2.4
        assert_eq!(r.new_coeff, next_up_f32(3.0 * 0.8));
        // intercept: 3.0 * 2.0 = 6.0 (in f64)
        assert_eq!(r.intercept_contrib, 3.0f64 * 2.0f64);
        assert!(!r.nonfinite);
    }

    // --- compose_upper: negative coefficient ---

    #[test]
    fn test_compose_upper_negative_coeff_uses_lower_relaxation() {
        let relax = LinearRelaxation::new(0.5, 1.0, 0.8, 2.0);
        let r = compose_upper(-3.0, &relax);
        // negative coeff → lower relaxation: -3.0 * 0.5 = -1.5
        assert_eq!(r.new_coeff, next_up_f32(-3.0 * 0.5));
        // intercept: -3.0 * 1.0 = -3.0 (in f64)
        assert_eq!(r.intercept_contrib, -3.0f64 * 1.0f64);
        assert!(!r.nonfinite);
    }

    // --- Directed rounding soundness ---

    #[test]
    fn test_compose_lower_rounds_toward_neg_infinity() {
        // For any finite product, compose_lower must return a value <= the true product.
        let relax = LinearRelaxation::new(0.7, 0.0, 0.9, 0.0);
        let r = compose_lower(2.5, &relax);
        let true_product = 2.5f32 * 0.7;
        assert!(
            r.new_coeff <= true_product,
            "compose_lower must round toward -inf: {} <= {}",
            r.new_coeff,
            true_product
        );
    }

    #[test]
    fn test_compose_upper_rounds_toward_pos_infinity() {
        // For any finite product, compose_upper must return a value >= the true product.
        let relax = LinearRelaxation::new(0.7, 0.0, 0.9, 0.0);
        let r = compose_upper(2.5, &relax);
        let true_product = 2.5f32 * 0.9;
        assert!(
            r.new_coeff >= true_product,
            "compose_upper must round toward +inf: {} >= {}",
            r.new_coeff,
            true_product
        );
    }

    // --- Non-finite overflow ---

    #[test]
    fn test_compose_lower_overflow_zeros_coeff_sets_nonfinite() {
        let relax = LinearRelaxation::new(f32::MAX, 0.0, f32::MAX, 0.0);
        let r = compose_lower(f32::MAX, &relax);
        // MAX * MAX overflows to Inf
        assert_eq!(r.new_coeff, 0.0, "overflowed coeff must be zeroed");
        assert!(r.nonfinite, "nonfinite flag must be set on overflow");
    }

    #[test]
    fn test_compose_upper_overflow_zeros_coeff_sets_nonfinite() {
        let relax = LinearRelaxation::new(f32::MAX, 0.0, f32::MAX, 0.0);
        let r = compose_upper(f32::MAX, &relax);
        assert_eq!(r.new_coeff, 0.0, "overflowed coeff must be zeroed");
        assert!(r.nonfinite, "nonfinite flag must be set on overflow");
    }

    #[test]
    fn test_compose_lower_negative_overflow_zeros_coeff() {
        let relax = LinearRelaxation::new(f32::MAX, 0.0, f32::MAX, 0.0);
        let r = compose_lower(-f32::MAX, &relax);
        // -MAX * MAX = -Inf
        assert_eq!(r.new_coeff, 0.0);
        assert!(r.nonfinite);
    }

    // --- Intercept precision in f64 ---

    #[test]
    fn test_compose_lower_intercept_uses_f64_precision() {
        // Two large f32 values whose product loses precision in f32 but not f64.
        let coeff: f32 = 1e7;
        let intercept: f32 = 1.5e-1;
        let relax = LinearRelaxation::new(1.0, intercept, 1.0, intercept);
        let r = compose_lower(coeff, &relax);
        // f64 product preserves full precision
        let expected = coeff as f64 * intercept as f64;
        assert_eq!(r.intercept_contrib, expected);
    }

    // --- Relaxation with Inf slope (zero-width domain) ---

    #[test]
    fn test_compose_lower_inf_slope_with_zero_coeff_safe() {
        // Zero coefficient must NOT be multiplied with Inf slope (#1736)
        let relax = LinearRelaxation::new(f32::INFINITY, 0.0, f32::INFINITY, 0.0);
        let r = compose_lower(0.0, &relax);
        assert_eq!(r.new_coeff, 0.0);
        assert!(!r.nonfinite);
    }

    #[test]
    fn test_compose_upper_inf_slope_with_zero_coeff_safe() {
        let relax = LinearRelaxation::new(f32::INFINITY, 0.0, f32::INFINITY, 0.0);
        let r = compose_upper(0.0, &relax);
        assert_eq!(r.new_coeff, 0.0);
        assert!(!r.nonfinite);
    }

    // --- NaN coefficient ---

    #[test]
    fn test_compose_lower_nan_coeff_returns_zero() {
        // NaN > 0.0 is false, NaN < 0.0 is false → falls through to ZERO
        let r = compose_lower(f32::NAN, &relu_relaxation());
        assert_eq!(r.new_coeff, 0.0);
        assert_eq!(r.intercept_contrib, 0.0);
        assert!(!r.nonfinite);
    }

    #[test]
    fn test_compose_upper_nan_coeff_returns_zero() {
        let r = compose_upper(f32::NAN, &relu_relaxation());
        assert_eq!(r.new_coeff, 0.0);
        assert_eq!(r.intercept_contrib, 0.0);
        assert!(!r.nonfinite);
    }

    // --- precompute_relaxations ---

    #[test]
    fn test_precompute_relaxations_identity() {
        let lower = vec![-1.0, 0.0, 1.0];
        let upper = vec![1.0, 2.0, 3.0];
        let relaxations =
            precompute_relaxations(&lower, &upper, &|_l, _u, _i| LinearRelaxation::identity());
        assert_eq!(relaxations.len(), 3);
        for r in &relaxations {
            assert_eq!(r.lower_slope, 1.0);
            assert_eq!(r.upper_slope, 1.0);
        }
    }

    #[test]
    fn test_precompute_relaxations_uses_index() {
        let lower = vec![0.0; 3];
        let upper = vec![1.0; 3];
        let relaxations = precompute_relaxations(&lower, &upper, &|_l, _u, i| {
            LinearRelaxation::new(i as f32, 0.0, i as f32, 0.0)
        });
        assert_eq!(relaxations[0].lower_slope, 0.0);
        assert_eq!(relaxations[1].lower_slope, 1.0);
        assert_eq!(relaxations[2].lower_slope, 2.0);
    }

    // --- Soundness property: compose_lower <= true <= compose_upper ---

    #[test]
    fn test_compose_soundness_positive_coeff() {
        // For any linear relaxation bounding f(x), and any positive coeff a:
        //   compose_lower(a, relax).new_coeff <= a * relax.lower_slope
        //   compose_upper(a, relax).new_coeff >= a * relax.upper_slope
        let relax = LinearRelaxation::new(0.3, 0.1, 0.9, 0.05);
        let a = 5.0f32;
        let lo = compose_lower(a, &relax);
        let hi = compose_upper(a, &relax);
        assert!(lo.new_coeff <= a * relax.lower_slope);
        assert!(hi.new_coeff >= a * relax.upper_slope);
    }

    #[test]
    fn test_compose_soundness_negative_coeff() {
        // For negative coeff a < 0:
        //   compose_lower uses upper relaxation, compose_upper uses lower relaxation
        //   The sign flip + directed rounding must still be sound.
        let relax = LinearRelaxation::new(0.3, 0.1, 0.9, 0.05);
        let a = -5.0f32;
        let lo = compose_lower(a, &relax);
        let hi = compose_upper(a, &relax);
        // compose_lower(neg): a * upper_slope, rounded down
        assert!(lo.new_coeff <= a * relax.upper_slope);
        // compose_upper(neg): a * lower_slope, rounded up
        assert!(hi.new_coeff >= a * relax.lower_slope);
    }
}
