// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! ReLU CROWN relaxation — Kani-verified reference implementation.
//!
//! This function exists as a formally verified reference used by Kani proof
//! harnesses. The production ReLU CROWN backward pass lives in
//! `ny_propagate::layers::activations::relu`.

/// CROWN linear relaxation for ReLU.
///
/// For x in [lower, upper]:
/// - If lower >= 0: ReLU(x) = x (pass-through)
/// - If upper <= 0: ReLU(x) = 0 (zero)
/// - If lower < 0 < upper: Use linear relaxation
///
/// Returns (lower_slope, lower_intercept, upper_slope, upper_intercept).
pub fn relu_crown_relaxation(lower: f32, upper: f32) -> (f32, f32, f32, f32) {
    assert!(lower.is_finite(), "lower must be finite, got {lower}");
    assert!(upper.is_finite(), "upper must be finite, got {upper}");
    assert!(lower <= upper, "lower ({lower}) > upper ({upper})");

    if lower >= 0.0 {
        (1.0, 0.0, 1.0, 0.0)
    } else if upper <= 0.0 {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        fn next_up_nonneg(x: f32) -> f32 {
            if !x.is_finite() {
                return x;
            }
            if x <= 0.0 {
                return f32::from_bits(1);
            }
            f32::from_bits(x.to_bits() + 1)
        }

        #[inline]
        fn is_denormal(x: f32) -> bool {
            x != 0.0 && x.abs() < f32::MIN_POSITIVE
        }

        const RATIO_THRESHOLD: f32 = 1e-30;
        let neg_lower = -lower;

        let (upper_slope, upper_intercept) = if is_denormal(lower) || is_denormal(upper) {
            (0.0, upper)
        } else {
            let ratio = upper / neg_lower;

            if ratio < RATIO_THRESHOLD {
                (0.0, upper)
            } else if ratio > 1.0 / RATIO_THRESHOLD {
                (1.0, neg_lower)
            } else {
                let width = upper - lower;
                let mut slope = ((upper as f64) / (width as f64)) as f32;
                slope = slope.clamp(0.0, 1.0);

                let min_for_lower = -slope * lower;
                let min_for_upper = upper - slope * upper;
                let mut intercept = min_for_lower.max(min_for_upper).max(0.0);

                let magnitude = intercept.abs().max(upper.abs()).max(neg_lower.abs());
                intercept = (intercept + f32::EPSILON * magnitude * 8.0).max(intercept);

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

        let lower_slope = if upper > neg_lower { 1.0 } else { 0.0 };
        let lower_intercept = 0.0;

        (lower_slope, lower_intercept, upper_slope, upper_intercept)
    }
}
