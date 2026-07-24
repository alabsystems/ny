// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ReLU CROWN relaxation — Kani-verified reference implementation.
//!
//! This module provides [`relu_crown_relaxation`], a formally verified ReLU
//! relaxation function used by Kani proof harnesses (`proofs/kani/`).
//! It is **not** called from production runtime paths — the production ReLU
//! CROWN backward pass lives in `layers/activations/relu/mod.rs`.
//!
//! IBP propagation (`relu_ibp`) has been moved to `layers/activations/relu/ibp.rs`.

/// CROWN linear relaxation for ReLU — Kani-verified reference implementation.
///
/// This function is **not called from production runtime paths**. The production
/// ReLU CROWN backward pass uses [`crate::layers::activations::relu::relu_linear_relaxation`]
/// instead, which handles NaN/Inf inputs gracefully.
///
/// This function exists as a formally verified reference used by:
/// - Kani proof harnesses in `proofs/kani/` (6 harnesses verify soundness, slope bounds,
///   region correctness, and endpoint constraints)
/// - Unit tests in this module and `tests/ibp.rs`
///
/// The `assert!` preconditions (lines 28-30) enforce the strict contract required for
/// formal verification. They are **not reachable from production code** and should not
/// appear in panic-cliff audits of runtime paths.
///
/// For x in [l, u]:
/// - If l >= 0: ReLU(x) = x (pass-through)
/// - If u <= 0: ReLU(x) = 0 (zero)
/// - If l < 0 < u: Use linear relaxation
///
/// # REQUIRES
/// - `lower` and `upper` are finite (not NaN or infinite)
/// - `lower <= upper` (well-formed interval)
///
/// # ENSURES
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept) where:
/// - For all x in [lower, upper]: `lower_slope * x + lower_intercept <= ReLU(x)`
/// - For all x in [lower, upper]: `upper_slope * x + upper_intercept >= ReLU(x)`
/// - All returned slopes are in [0, 1]
/// - In crossing region (lower < 0 < upper): upper relaxation is sound at both endpoints
pub fn relu_crown_relaxation(lower: f32, upper: f32) -> (f32, f32, f32, f32) {
    // REQUIRES contract — active in release builds (#2136).
    assert!(lower.is_finite(), "lower must be finite, got {lower}");
    assert!(upper.is_finite(), "upper must be finite, got {upper}");
    assert!(lower <= upper, "lower ({lower}) > upper ({upper})");

    // Returns (lower_slope, lower_intercept, upper_slope, upper_intercept)
    if lower >= 0.0 {
        // Positive region: identity
        (1.0, 0.0, 1.0, 0.0)
    } else if upper <= 0.0 {
        // Negative region: zero
        (0.0, 0.0, 0.0, 0.0)
    } else {
        fn next_up_nonneg(x: f32) -> f32 {
            if !x.is_finite() {
                return x;
            }
            if x <= 0.0 {
                // Smallest positive subnormal.
                return f32::from_bits(1);
            }
            f32::from_bits(x.to_bits() + 1)
        }

        /// Check if a value is denormalized (subnormal).
        /// Denormals are non-zero values with |x| < MIN_POSITIVE.
        /// Arithmetic on denormals has reduced precision and can cause soundness issues.
        #[inline]
        fn is_denormal(x: f32) -> bool {
            x != 0.0 && x.abs() < f32::MIN_POSITIVE
        }

        // Crossing region: linear relaxation
        //
        // The upper bound line passes through (lower, 0) and (upper, upper).
        // Due to floating point precision issues with extreme values (e.g., when
        // |upper| << |lower|), we must ensure soundness via explicit checks.
        //
        // See issue #335: Kani found counterexamples with:
        // 1. lower=-4.4e-36, upper=8.6e-39 (extreme ratio)
        // 2. lower=-1.4e-45, upper=1.4e-45 (denormalized floats)
        // The ratio-based fix handles (1) but not (2). Denormal inputs need
        // explicit handling because arithmetic precision is severely reduced.

        // Threshold for detecting degenerate ratios where floating point
        // precision cannot maintain both soundness constraints
        const RATIO_THRESHOLD: f32 = 1e-30;

        let neg_lower = -lower; // Positive value

        // Handle denormalized inputs first - arithmetic on denormals is imprecise
        // and can violate soundness even when ratios look normal.
        // Use conservative fallback: horizontal line at y = upper.
        let (upper_slope, upper_intercept) = if is_denormal(lower) || is_denormal(upper) {
            // Denormal detected: use conservative bound y = upper
            // For any x in [lower, upper]: ReLU(x) = max(0,x) <= upper
            (0.0, upper)
        } else {
            let ratio = upper / neg_lower;

            // Handle degenerate cases where ratio is extreme
            if ratio < RATIO_THRESHOLD {
                // upper << |lower|: Use horizontal line at y = upper (always sound)
                // For any x in [lower, upper]: ReLU(x) = max(0,x) <= upper
                (0.0, upper)
            } else if ratio > 1.0 / RATIO_THRESHOLD {
                // |lower| << upper: Nearly positive region
                // Standard formula would give slope ≈ 1, intercept ≈ 0
                // But we need intercept >= -slope * lower to satisfy bound at x=lower
                // Use slope = 1, intercept = |lower| to ensure soundness
                (1.0, neg_lower)
            } else {
                // Normal case: standard CROWN upper bound, but with numerical guards.
                //
                // Kani found that even if the line is exact at endpoints, the
                // non-fused evaluation `slope * x + intercept` can under-approximate
                // the true upper bound by ~1 ULP at interior points (see #335).
                //
                // Strategy:
                // 1) Compute slope in f64, then cast to f32.
                // 2) Choose intercept to satisfy both endpoint constraints.
                // 3) Add a tiny safety margin, then bump intercept until the upper
                //    endpoint has strict slack: slope*upper + intercept > upper.
                let width = upper - lower; // positive in crossing region
                let mut slope = ((upper as f64) / (width as f64)) as f32;
                slope = slope.clamp(0.0, 1.0);

                // Choose the minimum intercept that makes the upper relaxation sound
                // at both endpoints under non-fused evaluation.
                let min_for_lower = -slope * lower; // ensure slope*lower + intercept >= 0
                let min_for_upper = upper - slope * upper; // ensure slope*upper + intercept >= upper
                let mut intercept = min_for_lower.max(min_for_upper).max(0.0);

                // Always add a small relative safety margin in the crossing region.
                let magnitude = intercept.abs().max(upper.abs()).max(neg_lower.abs());
                intercept = (intercept + f32::EPSILON * magnitude * 8.0).max(intercept);

                // Bump intercept until endpoint constraints hold with a little slack
                // under non-fused evaluation. This fixes Kani counterexamples where
                // the interior point under-approximates by 1 ULP when the upper
                // endpoint is exact.
                if slope < 1.0 {
                    for _ in 0..32 {
                        let at_lower = slope * lower + intercept;
                        let at_upper = slope * upper + intercept;
                        if at_lower >= 0.0 && at_upper > upper {
                            break;
                        }
                        intercept = next_up_nonneg(intercept);
                    }
                } else {
                    for _ in 0..32 {
                        let at_lower = slope * lower + intercept;
                        if at_lower >= 0.0 {
                            break;
                        }
                        intercept = next_up_nonneg(intercept);
                    }
                }

                (slope, intercept)
            }
        };

        // For lower bound, α-CROWN optimizes this; default to zero or identity
        // depending on which gives tighter bound
        let lower_slope = if upper > neg_lower { 1.0 } else { 0.0 };
        let lower_intercept = 0.0;

        (lower_slope, lower_intercept, upper_slope, upper_intercept)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_positive_region() {
        let (ls, li, us, ui) = relu_crown_relaxation(0.5, 2.0);
        assert_eq!((ls, li, us, ui), (1.0, 0.0, 1.0, 0.0));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_negative_region() {
        let (ls, li, us, ui) = relu_crown_relaxation(-2.0, -0.5);
        assert_eq!((ls, li, us, ui), (0.0, 0.0, 0.0, 0.0));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_crossing_upper_dominant() {
        let (ls, li, us, ui) = relu_crown_relaxation(-1.0, 3.0);
        assert_eq!(ls, 1.0);
        assert_eq!(li, 0.0);
        // Slope should be close to 0.75, intercept >= 0.75 due to safety margin
        assert!((us - 0.75).abs() < 1e-5, "slope {} not close to 0.75", us);
        assert!(ui >= 0.75 - 1e-6, "intercept {} should be >= 0.75", ui);
        assert!(ui < 0.76, "intercept {} too large", ui); // sanity upper bound
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_crossing_lower_dominant() {
        let (ls, li, us, ui) = relu_crown_relaxation(-3.0, 1.0);
        assert_eq!(ls, 0.0);
        assert_eq!(li, 0.0);
        // Slope should be close to 0.25, intercept >= 0.75 due to safety margin
        assert!((us - 0.25).abs() < 1e-5, "slope {} not close to 0.25", us);
        assert!(ui >= 0.75 - 1e-6, "intercept {} should be >= 0.75", ui);
        assert!(ui < 0.76, "intercept {} too large", ui); // sanity upper bound
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_boundary_zero_lower() {
        let (ls, li, us, ui) = relu_crown_relaxation(0.0, 1.0);
        assert_eq!((ls, li, us, ui), (1.0, 0.0, 1.0, 0.0));
    }

    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_boundary_zero_upper() {
        let (ls, li, us, ui) = relu_crown_relaxation(-1.0, 0.0);
        assert_eq!((ls, li, us, ui), (0.0, 0.0, 0.0, 0.0));
    }

    /// Regression test for issue #335: extreme floating point values caused
    /// soundness violation where upper bound < ReLU(x) at x = upper.
    ///
    /// Counterexample found by Kani: lower=-4.411234e-36, upper=8.625892e-39
    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_extreme_values_soundness_335() {
        // Original counterexample from Kani
        let lower: f32 = -4.411234e-36;
        let upper: f32 = 8.625892e-39;
        let x = upper; // The point where soundness was violated

        let (ls, li, us, ui) = relu_crown_relaxation(lower, upper);

        // Verify upper bound soundness: us * x + ui >= ReLU(x)
        let relu_x = x.max(0.0);
        let upper_bound = us * x + ui;
        assert!(
            upper_bound >= relu_x,
            "Upper bound violated at x={}: bound={} < ReLU(x)={}",
            x,
            upper_bound,
            relu_x
        );

        // Verify lower bound soundness: ls * x + li <= ReLU(x)
        let lower_bound = ls * x + li;
        assert!(
            lower_bound <= relu_x,
            "Lower bound violated at x={}: bound={} > ReLU(x)={}",
            x,
            lower_bound,
            relu_x
        );

        // Check at both endpoints
        let relu_lower = lower.max(0.0);
        let upper_bound_at_lower = us * lower + ui;
        assert!(
            upper_bound_at_lower >= relu_lower,
            "Upper bound violated at lower={}: bound={} < ReLU={}",
            lower,
            upper_bound_at_lower,
            relu_lower
        );

        // Also check lower bound at x=lower
        let lower_bound_at_lower = ls * lower + li;
        assert!(
            lower_bound_at_lower <= relu_lower,
            "Lower bound violated at lower={}: bound={} > ReLU={}",
            lower,
            lower_bound_at_lower,
            relu_lower
        );

        // Check at x=0 (important interior point in crossing region)
        let x_zero: f32 = 0.0;
        let relu_zero = x_zero.max(0.0);
        let upper_bound_at_zero = us * x_zero + ui;
        let lower_bound_at_zero = ls * x_zero + li;
        assert!(
            upper_bound_at_zero >= relu_zero,
            "Upper bound violated at x=0: bound={} < ReLU={}",
            upper_bound_at_zero,
            relu_zero
        );
        assert!(
            lower_bound_at_zero <= relu_zero,
            "Lower bound violated at x=0: bound={} > ReLU={}",
            lower_bound_at_zero,
            relu_zero
        );
    }

    /// Regression test for issue #335: Kani concrete playback found an interior
    /// point where the non-fused evaluation `slope * x + intercept` was 1 ULP
    /// below `ReLU(x)` even though the relaxation was exact at endpoints.
    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_kani_concrete_playback_335() {
        let lower = f32::from_bits(0xa03bf0eb);
        let upper = f32::from_bits(0x20b21f9a);
        let x = f32::from_bits(0x20b21f99);

        let (_, _, us, ui) = relu_crown_relaxation(lower, upper);

        let relu_x = x.max(0.0);
        let upper_bound = us * x + ui;
        assert!(
            upper_bound >= relu_x,
            "Upper bound violated at x={}: bound={} < ReLU(x)={}",
            x,
            upper_bound,
            relu_x
        );

        // Ensure we have some slack at the upper endpoint to cover interior-rounding.
        let at_upper = us * upper + ui;
        assert!(at_upper > upper, "Expected strict slack at upper endpoint");
    }

    /// Test additional extreme ratio cases to ensure robustness.
    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_extreme_ratios() {
        // Various extreme ratios
        let test_cases: &[(f32, f32)] = &[
            (-1e38, 1e-38),  // upper << |lower|
            (-1e-38, 1e38),  // |lower| << upper
            (-1e-30, 1e-35), // Both very small, upper << |lower|
            (-1e-35, 1e-30), // Both very small, |lower| << upper
            (-1.0, 1e-38),   // Normal lower, tiny upper
            (-1e-38, 1.0),   // Tiny lower, normal upper
            // Near threshold boundary tests (RATIO_THRESHOLD = 1e-30)
            (-1.0, 1e-29), // ratio = 1e-29, just above threshold
            (-1.0, 1e-31), // ratio = 1e-31, just below threshold
            (-1e-29, 1.0), // ratio = 1e29, just below upper threshold
            (-1e-31, 1.0), // ratio = 1e31, just above upper threshold
        ];

        for &(lower, upper) in test_cases {
            let (ls, li, us, ui) = relu_crown_relaxation(lower, upper);

            // Test at multiple points in the interval
            let test_points = [lower, lower / 2.0, 0.0, upper / 2.0, upper];
            for x in test_points {
                if x < lower || x > upper {
                    continue;
                }
                let relu_x = x.max(0.0);
                let upper_bound = us * x + ui;
                let lower_bound = ls * x + li;

                assert!(
                    upper_bound >= relu_x || (upper_bound - relu_x).abs() < 1e-45,
                    "Upper bound violated for [{}, {}] at x={}: bound={} < ReLU={}",
                    lower,
                    upper,
                    x,
                    upper_bound,
                    relu_x
                );
                assert!(
                    lower_bound <= relu_x || (lower_bound - relu_x).abs() < 1e-45,
                    "Lower bound violated for [{}, {}] at x={}: bound={} > ReLU={}",
                    lower,
                    upper,
                    x,
                    lower_bound,
                    relu_x
                );
            }
        }
    }

    /// Regression test for issue #335: denormalized float inputs.
    /// Prover found counterexample with smallest positive/negative denormals.
    /// These require explicit handling because arithmetic on denormals loses precision.
    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_denormal_inputs_soundness_335() {
        // Smallest positive denormal = f32::from_bits(1) ≈ 1.4e-45
        let smallest_denormal = f32::from_bits(1);
        assert!(smallest_denormal > 0.0 && smallest_denormal < f32::MIN_POSITIVE);

        // Counterexample from Prover: symmetric denormal interval
        let lower = -smallest_denormal;
        let upper = smallest_denormal;

        let (ls, li, us, ui) = relu_crown_relaxation(lower, upper);

        // Test soundness at multiple points
        let test_points = [lower, 0.0, upper];
        for x in test_points {
            let relu_x = x.max(0.0);
            let upper_bound = us * x + ui;
            let lower_bound = ls * x + li;

            assert!(
                upper_bound >= relu_x,
                "Upper bound violated at x={}: bound={} < ReLU={}",
                x,
                upper_bound,
                relu_x
            );
            assert!(
                lower_bound <= relu_x,
                "Lower bound violated at x={}: bound={} > ReLU={}",
                x,
                lower_bound,
                relu_x
            );
        }
    }

    /// Test various denormal input combinations.
    #[ntest::timeout(10000)]
    #[test]
    fn test_relu_crown_denormal_combinations() {
        let denormal = f32::from_bits(1); // smallest positive denormal
        let normal_small = f32::MIN_POSITIVE * 2.0; // small but normal

        let test_cases: &[(f32, f32)] = &[
            (-denormal, denormal),         // both denormal
            (-denormal, normal_small),     // lower denormal, upper normal
            (-normal_small, denormal),     // lower normal, upper denormal
            (-denormal * 100.0, denormal), // larger denormal lower
            (-denormal, denormal * 100.0), // larger denormal upper
        ];

        for &(lower, upper) in test_cases {
            let (ls, li, us, ui) = relu_crown_relaxation(lower, upper);

            // Test at endpoints
            for x in [lower, 0.0, upper] {
                let relu_x = x.max(0.0);
                let upper_bound = us * x + ui;
                let lower_bound = ls * x + li;

                assert!(
                    upper_bound >= relu_x,
                    "Upper bound violated for [{:e}, {:e}] at x={:e}: bound={:e} < ReLU={:e}",
                    lower,
                    upper,
                    x,
                    upper_bound,
                    relu_x
                );
                assert!(
                    lower_bound <= relu_x,
                    "Lower bound violated for [{:e}, {:e}] at x={:e}: bound={:e} > ReLU={:e}",
                    lower,
                    upper,
                    x,
                    lower_bound,
                    relu_x
                );
            }
        }
    }
}
