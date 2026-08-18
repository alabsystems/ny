// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! A rigorous, SELF-CERTIFYING enclosure of an arithmetic mean over f64 values.
//!
//! # What this replaces, and why
//!
//! `ReduceMeanLayer::propagate_ibp` used to accumulate in f64 with ndarray's
//! `mean_axis` and then narrow to f32 with the tree-wide idiom
//!
//! ```text
//! lower = next_down_f32(accumulator as f32)
//! upper = next_up_f32(accumulator as f32)
//! ```
//!
//! That is sound but it conflates two different jobs into one unconditional
//! ULP step:
//!
//! 1. covering the round-to-nearest f64 -> f32 narrowing, and
//! 2. covering the rounding committed by the f64 accumulation itself.
//!
//! Job 1 is now [`ny_tensor::cast_f64_to_f32_down`], which steps only when the
//! narrowing actually rounded the wrong way. Job 2 is this module, which
//! MEASURES the accumulation error instead of assuming an f32 ULP dominates it.
//!
//! Two things follow, and both matter:
//!
//! * **It is tighter.** When the reduction is exact — every reduction over an
//!   axis of length 1, every power-of-two-length reduction of like-scaled
//!   values, and in practice most small reductions — the charge is exactly
//!   zero and the bound is returned unwidened. The old code gave away a full
//!   f32 ULP on every bound of every reduction unconditionally. `ReduceMean`
//!   over a size-1 axis is a no-op in ONNX and appeared in the graph as an
//!   identity; it was silently costing 1 ULP per bound per invocation.
//! * **It is more rigorous.** The old code had no explicit charge for the f64
//!   accumulation error at all: it relied on the f32 ULP step (relative
//!   `2^-24`) swallowing the f64 accumulation error (relative `~m * 2^-53`).
//!   That holds for `m` well below `2^29` and was never stated, let alone
//!   checked. Here the charge is derived from an exact error-free transform, so
//!   it holds for every `m`.
//!
//! # The method
//!
//! Summation is Knuth's `TwoSum`, which is EXACT for all finite inputs: for any
//! finite `a, b` with no overflow, `two_sum(a, b) = (hi, lo)` satisfies
//! `a + b == hi + lo` in the reals. Accumulating `|lo|` therefore yields a
//! rigorous bound on everything the running sum discarded.
//!
//! The division is certified by its own residual. For IEEE-754 round-to-nearest
//! division `q = s / n`, the fused multiply-add `fma(-q, n, s)` computes
//! `s - q*n` EXACTLY, so `|s/n - q| = |r| / n` with no inequality slack at all.
//!
//! Both are combined outward, so the returned pair encloses the exact
//! arithmetic mean of the inputs viewed as exact reals.
//!
//! # Non-finite inputs
//!
//! `TwoSum`'s residual is meaningless once an infinity enters (`inf - inf`
//! is NaN), so a non-finite accumulator short-circuits and returns the raw
//! value in both slots. That reproduces the previous behaviour exactly and
//! leaves the decision to the centralized NaN/Inf repair
//! (`RepairStrategy::Conservative`) that runs downstream, which is where
//! infeasible-interval policy lives.

use ny_core::dd::{next_down_f64, next_up_f64};

/// Knuth's `TwoSum`: returns `(hi, lo)` with `hi == fl(a + b)` and
/// `a + b == hi + lo` exactly, for all finite `a`, `b` without overflow.
///
/// Unlike `FastTwoSum` this needs no ordering assumption on `|a|` vs `|b|`,
/// which matters because a reduction visits its terms in layout order.
#[inline]
fn two_sum_f64(a: f64, b: f64) -> (f64, f64) {
    let hi = a + b;
    let a_virtual = hi - b;
    let b_virtual = hi - a_virtual;
    let a_round = a - a_virtual;
    let b_round = b - b_virtual;
    (hi, a_round + b_round)
}

/// `a + b` rounded toward +inf, EXACT when the addition is exact.
///
/// The obvious spelling `next_up_f64(a + b)` is wrong for this module's purpose:
/// `next_up_f64(0.0)` is the smallest positive subnormal, so an error charge
/// built that way is never zero even when nothing was ever rounded — which is
/// precisely the case the certified reduction exists to detect. TwoSum tells us
/// whether the add lost anything, so the step is taken only when it did.
#[inline]
fn add_up(a: f64, b: f64) -> f64 {
    let (hi, lo) = two_sum_f64(a, b);
    if lo > 0.0 {
        next_up_f64(hi)
    } else {
        hi
    }
}

/// `a - b` rounded toward -inf, exact when the subtraction is exact.
#[inline]
fn sub_down(a: f64, b: f64) -> f64 {
    let (hi, lo) = two_sum_f64(a, -b);
    if lo < 0.0 {
        next_down_f64(hi)
    } else {
        hi
    }
}

/// `a / b` rounded toward +inf for `a >= 0`, `b > 0`, exact when the division
/// is exact. `fma(-q, b, a)` is the exact remainder, so the step is again taken
/// only when something was actually lost.
#[inline]
fn div_up(a: f64, b: f64) -> f64 {
    let quotient = a / b;
    if !quotient.is_finite() {
        return quotient;
    }
    if f64::mul_add(-quotient, b, a) > 0.0 {
        next_up_f64(quotient)
    } else {
        quotient
    }
}

/// A rigorous enclosure `[lo, hi]` of the exact arithmetic mean of `values`,
/// each element read as an exact real.
///
/// `lo == hi` exactly when the whole reduction was committed without a single
/// rounding — the common case that the previous unconditional-ULP code could
/// not express.
///
/// # Guarantees
///
/// * `lo <= mean(values) <= hi` for every finite input, for every length.
/// * `lo <= hi` always.
/// * Non-finite accumulation returns `(v, v)` for the raw f64 value `v`,
///   deferring to the downstream repair.
#[must_use]
pub(super) fn certified_mean_enclosure(values: impl ExactSizeIterator<Item = f64>) -> (f64, f64) {
    let count = values.len();
    if count == 0 {
        // A mean over nothing constrains nothing. The caller's shape checks
        // reject this earlier; returning the whole line keeps it sound if one
        // is ever relaxed.
        return (f64::NEG_INFINITY, f64::INFINITY);
    }

    // Exact summation: `sum + discarded` is the true total, and `discarded`
    // never exceeds `residual_charge`.
    let mut sum = 0.0_f64;
    let mut residual_charge = 0.0_f64;
    for value in values {
        let (hi, lo) = two_sum_f64(sum, value);
        sum = hi;
        // `lo` is the EXACT part `sum` could not hold. `add_up` keeps the charge
        // an upper bound on its own accumulation while staying exactly zero for
        // a reduction that never rounded.
        residual_charge = add_up(residual_charge, lo.abs());
    }

    if !sum.is_finite() || !residual_charge.is_finite() {
        return (sum, sum);
    }

    // Certified division: `fma(-q, n, sum)` is the exact remainder `sum - q*n`.
    let n = count as f64;
    let quotient = sum / n;
    if !quotient.is_finite() {
        return (quotient, quotient);
    }
    let remainder = f64::mul_add(-quotient, n, sum);

    // |true_mean - quotient| <= (|remainder| + residual_charge) / n, with the
    // add and the divide each rounded outward only if they actually rounded.
    let half_width = div_up(add_up(remainder.abs(), residual_charge), n);

    if half_width == 0.0 {
        // Nothing was rounded anywhere: the mean is EXACTLY `quotient`, and the
        // caller may narrow it to f32 without giving up a single ULP.
        return (quotient, quotient);
    }

    (sub_down(quotient, half_width), add_up(quotient, half_width))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enclosure(values: &[f64]) -> (f64, f64) {
        certified_mean_enclosure(values.iter().copied())
    }

    #[test]
    fn a_single_element_mean_is_exact_and_unwidened() {
        // The case the old unconditional ULP step gave away for nothing: ONNX
        // `ReduceMean` over an axis of length 1 is an identity.
        for probe in [0.1_f32, 0.3, -2.5, 1e-30, 3.4e38, 0.0] {
            let x = f64::from(probe);
            let (lo, hi) = enclosure(&[x]);
            assert_eq!(lo, x, "single-element mean must not move the lower bound");
            assert_eq!(hi, x, "single-element mean must not move the upper bound");
        }
    }

    #[test]
    fn exact_reductions_report_zero_width() {
        // Powers of two of like-scaled integers sum and divide exactly.
        let (lo, hi) = enclosure(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(lo, 2.5);
        assert_eq!(hi, 2.5);

        let (lo, hi) = enclosure(&[-1.0, 1.0]);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn inexact_reductions_enclose_the_true_mean() {
        // 1/3 is not representable; the enclosure must straddle it.
        let (lo, hi) = enclosure(&[1.0, 0.0, 0.0]);
        let truth = 1.0_f64 / 3.0;
        assert!(lo <= truth && truth <= hi, "[{lo}, {hi}] must contain 1/3");
        assert!(lo < hi, "an inexact division must report nonzero width");
    }

    #[test]
    fn catastrophic_absorption_is_still_enclosed() {
        // The failure the f64 accumulation exists to prevent, one exponent
        // below f64's own resolution: 2^53 absorbs 1 under round-to-nearest.
        let big = 9_007_199_254_740_992.0_f64; // 2^53
        let values = [big, 1.0];
        let (lo, hi) = enclosure(&values);
        // True mean is (2^53 + 1) / 2 = 2^52 + 0.5, exactly representable.
        let truth = 4_503_599_627_370_496.5_f64;
        assert!(
            lo <= truth && truth <= hi,
            "[{lo}, {hi}] must contain the true mean {truth} despite absorption"
        );
    }

    #[test]
    fn the_enclosure_holds_against_an_exact_rational_reference() {
        // Sweep lengths and magnitudes, comparing against a mean accumulated in
        // integer arithmetic so the reference has no rounding of its own.
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for len in [1_usize, 2, 3, 5, 8, 17, 64, 129, 1000] {
            for _ in 0..40 {
                // Integer-valued f64s: the true mean is exactly sum_int / len.
                let ints: Vec<i64> = (0..len)
                    .map(|_| i64::from((next() % 2_000_001) as i32 - 1_000_000))
                    .collect();
                let values: Vec<f64> = ints.iter().map(|&i| i as f64).collect();
                let (lo, hi) = enclosure(&values);
                let total: i128 = ints.iter().map(|&i| i128::from(i)).sum();
                // Compare as exact rationals: lo <= total/len <= hi becomes
                // lo*len <= total <= hi*len, evaluated in i128.
                let len_i = len as i128;
                assert!(
                    (lo * len as f64).floor() as i128 <= total,
                    "len {len}: lower bound {lo} exceeds the exact mean {total}/{len_i}"
                );
                assert!(
                    (hi * len as f64).ceil() as i128 >= total,
                    "len {len}: upper bound {hi} falls below the exact mean {total}/{len_i}"
                );
                assert!(lo <= hi, "len {len}: enclosure must be ordered");
            }
        }
    }

    #[test]
    fn non_finite_input_defers_to_the_downstream_repair() {
        let (lo, hi) = enclosure(&[f64::INFINITY, 1.0]);
        assert_eq!(lo, hi, "a non-finite accumulator returns a degenerate pair");
        assert!(lo.is_infinite());

        let (lo, hi) = enclosure(&[f64::NAN, 1.0]);
        assert!(
            lo.is_nan() && hi.is_nan(),
            "NaN must reach the repair intact"
        );
    }

    #[test]
    fn an_empty_reduction_constrains_nothing() {
        let (lo, hi) = enclosure(&[]);
        assert_eq!(lo, f64::NEG_INFINITY);
        assert_eq!(hi, f64::INFINITY);
    }

    #[test]
    fn two_sum_is_exact_where_the_plain_add_is_not() {
        let (hi, lo) = two_sum_f64(1.0, f64::EPSILON / 2.0);
        assert_eq!(hi, 1.0, "the add absorbs the tiny operand");
        assert_eq!(
            lo,
            f64::EPSILON / 2.0,
            "and TwoSum recovers exactly what was absorbed"
        );
    }

    #[test]
    fn directed_helpers_are_exact_when_the_operation_is_exact() {
        // The bug these exist to prevent: `next_up_f64(0.0)` is a subnormal, so
        // a charge built with a blind step is never zero and every reduction
        // looks inexact.
        assert_eq!(
            add_up(0.0, 0.0),
            0.0,
            "an exact add must not accrue a charge"
        );
        assert_eq!(add_up(1.0, 2.0), 3.0);
        assert_eq!(sub_down(2.5, 0.0), 2.5);
        assert_eq!(
            div_up(0.0, 4.0),
            0.0,
            "an exact divide must not accrue a charge"
        );
        assert_eq!(div_up(10.0, 4.0), 2.5);
    }

    #[test]
    fn directed_helpers_round_outward_when_the_operation_is_not_exact() {
        // 1 + eps/2 is not representable: the true sum sits above fl(1+eps/2)=1.
        let inexact = add_up(1.0, f64::EPSILON / 2.0);
        assert!(inexact > 1.0, "an inexact add must round up, got {inexact}");

        let down = sub_down(1.0, f64::EPSILON / 2.0);
        assert!(
            down < 1.0,
            "an inexact subtract must round down, got {down}"
        );

        let q = div_up(1.0, 3.0);
        assert!(
            q >= 1.0 / 3.0 && q * 3.0 >= 1.0,
            "an inexact divide must round up, got {q}"
        );
    }
}
