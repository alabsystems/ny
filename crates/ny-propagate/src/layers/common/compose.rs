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

/// Pure gate predicate: exactly `"1"` enables (same idiom as
/// `iter0_parity_trace::gate_on`).
fn exact_zero_slope_gate(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// `NY_COMPOSE_EXACT_ZERO_SLOPE=1` stores an EXACT `0.0` coefficient when the
/// relaxation slope is exactly zero, instead of stepping the zero product one
/// subnormal outward. Default unset ⇒ byte-identical historical behavior.
///
/// WHY: a dead ReLU (`u <= 0`) relaxes to the exact linear function `y = 0`,
/// so `coeff × 0.0` is exactly `±0.0` for every finite coefficient — there is
/// no rounding to guard against, and `next_down_f32(0.0) = -2^-149` /
/// `next_up_f32(0.0) = +2^-149` only smear one-ULP subnormal garbage over the
/// whole A-tensor. On the cifar100 iter-0 parity trace this produced the
/// bit-identical `abs_max=1.401e-45` "sentinel" patches at Conv_49/43/37
/// (genuinely all-dead trunk blocks; see
/// docs/ADD28_COEFF_ERR_AND_PATCHES_SENTINEL_DIAGNOSIS_2026-07-30.md).
/// Sound and strictly tighter when on: the stored value becomes the true
/// coefficient exactly. Underflowed products of NONZERO slopes keep the
/// outward step (the true product is nonzero there), and non-finite products
/// keep the poison path.
pub(crate) fn exact_zero_slope_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE.get_or_init(|| {
        exact_zero_slope_gate(std::env::var("NY_COMPOSE_EXACT_ZERO_SLOPE").ok().as_deref())
    })
}

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
    compose_lower_with_policy(coeff, relax, exact_zero_slope_enabled())
}

/// Policy-explicit body of [`compose_lower`]; `exact_zero_slope` is the
/// resolved `NY_COMPOSE_EXACT_ZERO_SLOPE` gate (threaded so tests exercise
/// both policies without mutating process env).
#[inline(always)]
pub(crate) fn compose_lower_with_policy(
    coeff: f32,
    relax: &LinearRelaxation,
    exact_zero_slope: bool,
) -> ComposeResult {
    if coeff > 0.0 {
        let product = coeff * relax.lower_slope;
        ComposeResult {
            new_coeff: directed_coeff(product, relax.lower_slope, exact_zero_slope, next_down_f32),
            intercept_contrib: coeff as f64 * relax.lower_intercept as f64,
            nonfinite: !product.is_finite(),
        }
    } else if coeff < 0.0 {
        let product = coeff * relax.upper_slope;
        ComposeResult {
            new_coeff: directed_coeff(product, relax.upper_slope, exact_zero_slope, next_down_f32),
            intercept_contrib: coeff as f64 * relax.upper_intercept as f64,
            nonfinite: !product.is_finite(),
        }
    } else {
        ComposeResult::ZERO
    }
}

/// Apply the directed rounding to a composed coefficient product.
///
/// - Non-finite product: `0.0` (the caller's nonfinite flag poisons the row).
/// - Exactly-zero SLOPE under the gate: the true product of a finite nonzero
///   coefficient with a zero slope is exactly `±0.0`, so store `0.0` — no
///   outward step is needed for a value that is already exact. The slope (not
///   the product) is tested so an UNDERFLOWED product of a nonzero slope
///   (true value nonzero) keeps its sound outward step.
/// - Otherwise: the historical directed step (`next_down`/`next_up`).
#[inline(always)]
fn directed_coeff(
    product: f32,
    slope: f32,
    exact_zero_slope: bool,
    round: impl Fn(f32) -> f32,
) -> f32 {
    if !product.is_finite() {
        return 0.0;
    }
    if exact_zero_slope && slope == 0.0 {
        return 0.0;
    }
    round(product)
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
    compose_upper_with_policy(coeff, relax, exact_zero_slope_enabled())
}

/// Policy-explicit body of [`compose_upper`]; see [`compose_lower_with_policy`].
#[inline(always)]
pub(crate) fn compose_upper_with_policy(
    coeff: f32,
    relax: &LinearRelaxation,
    exact_zero_slope: bool,
) -> ComposeResult {
    if coeff > 0.0 {
        let product = coeff * relax.upper_slope;
        ComposeResult {
            new_coeff: directed_coeff(product, relax.upper_slope, exact_zero_slope, next_up_f32),
            intercept_contrib: coeff as f64 * relax.upper_intercept as f64,
            nonfinite: !product.is_finite(),
        }
    } else if coeff < 0.0 {
        let product = coeff * relax.lower_slope;
        ComposeResult {
            new_coeff: directed_coeff(product, relax.lower_slope, exact_zero_slope, next_up_f32),
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

    // --- Exact-zero-slope gate (NY_COMPOSE_EXACT_ZERO_SLOPE, #iter0-alpha-parity) ---

    /// Dead-ReLU relaxation: slopes 0, intercepts 0.
    fn dead_relaxation() -> LinearRelaxation {
        LinearRelaxation::new(0.0, 0.0, 0.0, 0.0)
    }

    #[test]
    fn test_gate_requires_exactly_one() {
        assert!(exact_zero_slope_gate(Some("1")));
        assert!(!exact_zero_slope_gate(Some("true")));
        assert!(!exact_zero_slope_gate(Some("0")));
        assert!(!exact_zero_slope_gate(None));
    }

    #[test]
    fn test_zero_slope_product_rounds_one_ulp_outward_by_default() {
        // This pins the "bit-identical subnormal sentinel" mechanism the
        // cifar100 iter-0 parity trace localized at Conv_49/43/37: a genuinely
        // dead layer (all slopes 0) turns every nonzero coefficient into
        // exactly ±2^-149 because the directed step is applied to an exact
        // zero product.
        let lo = compose_lower_with_policy(0.7, &dead_relaxation(), false);
        assert_eq!(lo.new_coeff, -f32::from_bits(1));
        let up = compose_upper_with_policy(0.7, &dead_relaxation(), false);
        assert_eq!(up.new_coeff, f32::from_bits(1));
    }

    #[test]
    fn test_zero_slope_product_stays_exactly_zero_when_gated() {
        for coeff in [0.7f32, -0.7, 1e-30, -1e30] {
            let lo = compose_lower_with_policy(coeff, &dead_relaxation(), true);
            assert_eq!(lo.new_coeff, 0.0, "lower coeff {coeff}");
            assert_eq!(lo.intercept_contrib, 0.0);
            assert!(!lo.nonfinite);
            let up = compose_upper_with_policy(coeff, &dead_relaxation(), true);
            assert_eq!(up.new_coeff, 0.0, "upper coeff {coeff}");
            assert_eq!(up.intercept_contrib, 0.0);
            assert!(!up.nonfinite);
        }
    }

    #[test]
    fn test_gated_exact_zero_preserves_nonfinite_coefficient_poisoning() {
        // inf * 0.0 = NaN: the row must still poison, never claim an exact 0.
        let lo = compose_lower_with_policy(f32::INFINITY, &dead_relaxation(), true);
        assert_eq!(lo.new_coeff, 0.0);
        assert!(lo.nonfinite, "inf x 0 slope must keep the poison path");
        let up = compose_upper_with_policy(f32::INFINITY, &dead_relaxation(), true);
        assert!(up.nonfinite);
    }

    #[test]
    fn test_gated_exact_zero_leaves_nonzero_slopes_untouched() {
        let relax = LinearRelaxation::new(0.5, 1.0, 0.8, 2.0);
        let gated = compose_lower_with_policy(3.0, &relax, true);
        let plain = compose_lower_with_policy(3.0, &relax, false);
        assert_eq!(gated.new_coeff, plain.new_coeff);
        assert_eq!(gated.new_coeff, next_down_f32(3.0 * 0.5));
    }

    #[test]
    fn test_gated_exact_zero_does_not_claim_exactness_for_underflowed_products() {
        // Product underflows to 0.0 but the SLOPE is nonzero, so the true
        // product is nonzero: the outward step must survive the gate.
        let tiny = f32::MIN_POSITIVE * 0.5; // subnormal, nonzero
        let relax = LinearRelaxation::new(tiny, 0.0, tiny, 0.0);
        let lo = compose_lower_with_policy(tiny, &relax, true);
        assert_eq!(
            lo.new_coeff,
            next_down_f32(tiny * tiny),
            "underflowed nonzero-slope product must keep its directed step"
        );
        assert_eq!(lo.new_coeff, -f32::from_bits(1));
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
