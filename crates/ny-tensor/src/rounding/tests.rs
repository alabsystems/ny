// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_next_up_down_basic() {
    assert!(
        next_up_f32(1.0) > 1.0,
        "next_up_f32(1.0) should advance above 1.0"
    );
    assert!(
        next_down_f32(1.0) < 1.0,
        "next_down_f32(1.0) should move below 1.0"
    );
    assert!(
        next_up_f32(-1.0) > -1.0,
        "next_up_f32(-1.0) should move toward zero"
    );
    assert!(
        next_down_f32(-1.0) < -1.0,
        "next_down_f32(-1.0) should move away from zero"
    );
}

/// Regression test for #3149: infinity sentinels must be preserved.
/// `mark_infeasible_all()` uses (+inf, -inf) to mark infeasible neurons.
/// Directed rounding must not convert these into finite bounds.
#[test]
fn test_next_up_down_preserves_infinity_sentinels_3149() {
    // next_down_f32(+inf) was returning f32::MAX — must preserve +inf
    assert_eq!(next_down_f32(f32::INFINITY), f32::INFINITY);
    // next_up_f32(-inf) was returning f32::MIN — must preserve -inf
    assert_eq!(next_up_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
    // Same-sign infinity: already worked, verify no regression
    assert_eq!(next_up_f32(f32::INFINITY), f32::INFINITY);
    assert_eq!(next_down_f32(f32::NEG_INFINITY), f32::NEG_INFINITY);
}

#[test]
fn test_shift_n_ulps_zero_is_noop() {
    assert_eq!(shift_up_n_ulps(1.0, 0), 1.0);
    assert_eq!(shift_down_n_ulps(1.0, 0), 1.0);
    assert_eq!(shift_up_n_ulps(-1.0, 0), -1.0);
    assert_eq!(shift_down_n_ulps(-1.0, 0), -1.0);
}

#[test]
fn test_shift_one_ulp_matches_next_up_down() {
    for x in [1.0f32, -1.0, 0.5, -0.5, 100.0, -100.0, 1e-10, -1e-10] {
        assert_eq!(shift_up_n_ulps(x, 1), next_up_f32(x), "shift_up_1 for {x}");
        assert_eq!(
            shift_down_n_ulps(x, 1),
            next_down_f32(x),
            "shift_down_1 for {x}"
        );
    }
}

#[test]
fn test_shift_n_ulps_positive_direction() {
    let x = 1.0f32;
    let shifted = shift_up_n_ulps(x, 10);
    assert!(
        shifted > x,
        "shift_up_n_ulps({x}, 10) should increase, got {shifted}"
    );
    // Verify it's exactly 10 ULPs above
    assert_eq!(shifted.to_bits() - x.to_bits(), 10);
}

#[test]
fn test_shift_n_ulps_negative_direction() {
    let x = 1.0f32;
    let shifted = shift_down_n_ulps(x, 10);
    assert!(
        shifted < x,
        "shift_down_n_ulps({x}, 10) should decrease, got {shifted}"
    );
    // Verify it's exactly 10 ULPs below
    assert_eq!(x.to_bits() - shifted.to_bits(), 10);
}

#[test]
fn test_shift_n_ulps_negative_values() {
    let x = -1.0f32;
    // Shifting up: moving toward zero
    let up = shift_up_n_ulps(x, 10);
    assert!(
        up > x,
        "shift_up_n_ulps({x}, 10) should move toward zero, got {up}"
    );
    assert!(
        up < 0.0,
        "shift_up_n_ulps({x}, 10) should remain negative before crossing zero, got {up}"
    );
    // Shifting down: moving away from zero
    let down = shift_down_n_ulps(x, 10);
    assert!(
        down < x,
        "shift_down_n_ulps({x}, 10) should move away from zero, got {down}"
    );
}

#[test]
fn test_shift_n_ulps_crosses_zero_up() {
    // Smallest negative subnormal
    let x = f32::from_bits(0x8000_0001); // -smallest subnormal
    let shifted = shift_up_n_ulps(x, 5);
    // Should cross zero into positive subnormals
    assert!(
        shifted > 0.0,
        "shift_up_n_ulps should cross zero into positive subnormals, got {shifted}"
    );
    assert_eq!(shifted.to_bits(), 4); // 5 - 1 = 4 ULPs into positive
}

#[test]
fn test_shift_n_ulps_crosses_zero_down() {
    // Smallest positive subnormal
    let x = f32::from_bits(1); // smallest positive subnormal
    let shifted = shift_down_n_ulps(x, 5);
    // Should cross zero into negative subnormals
    assert!(
        shifted < 0.0,
        "shift_down_n_ulps should cross zero into negative subnormals, got {shifted}"
    );
    assert_eq!(shifted.to_bits(), 0x8000_0004); // 5 - 1 = 4 ULPs into negative
}

#[test]
fn test_shift_n_ulps_from_zero() {
    assert_eq!(shift_up_n_ulps(0.0, 5).to_bits(), 5);
    assert_eq!(shift_down_n_ulps(0.0, 5).to_bits(), 0x8000_0005);
}

#[test]
fn test_shift_n_ulps_saturates_to_infinity() {
    let x = f32::MAX;
    let shifted = shift_up_n_ulps(x, 1000);
    assert!(
        shifted.is_infinite(),
        "shift_up_n_ulps({x}, 1000) should saturate to infinity, got {shifted}"
    );
    assert!(
        shifted > 0.0,
        "shift_up_n_ulps({x}, 1000) should saturate to +inf, got {shifted}"
    );

    let x = f32::MIN; // most negative finite
    let shifted = shift_down_n_ulps(x, 1000);
    assert!(
        shifted.is_infinite(),
        "shift_down_n_ulps({x}, 1000) should saturate to infinity, got {shifted}"
    );
    assert!(
        shifted < 0.0,
        "shift_down_n_ulps({x}, 1000) should saturate to -inf, got {shifted}"
    );
}

#[test]
fn test_shift_n_ulps_nan_passthrough() {
    assert!(
        shift_up_n_ulps(f32::NAN, 10).is_nan(),
        "shift_up_n_ulps(NaN, 10) should preserve NaN"
    );
    assert!(
        shift_down_n_ulps(f32::NAN, 10).is_nan(),
        "shift_down_n_ulps(NaN, 10) should preserve NaN"
    );
}

#[test]
fn test_shift_n_ulps_infinity_passthrough() {
    assert_eq!(shift_up_n_ulps(f32::INFINITY, 10), f32::INFINITY);
    assert_eq!(shift_down_n_ulps(f32::NEG_INFINITY, 10), f32::NEG_INFINITY);
}

/// Regression test for #3149: opposite-sign infinity must be preserved.
/// `mark_infeasible_all()` uses sentinel (+inf lower, -inf upper).
/// `round_for_soundness_n_ulps` calls `shift_down_n_ulps(+inf, n)` on lower
/// and `shift_up_n_ulps(-inf, n)` on upper. Both must preserve infinity.
/// Previously, only same-sign infinity was checked (== INFINITY / == NEG_INFINITY)
/// instead of `is_infinite()`, causing sentinels to become large finite values.
#[test]
fn test_shift_n_ulps_opposite_sign_infinity_3149() {
    // Infeasible sentinel: lower=+inf, upper=-inf
    // shift_down on lower (+inf) must preserve +inf
    assert_eq!(
        shift_down_n_ulps(f32::INFINITY, 10),
        f32::INFINITY,
        "shift_down must preserve +inf (infeasible lower sentinel)"
    );
    assert_eq!(
        shift_down_n_ulps(f32::INFINITY, 770),
        f32::INFINITY,
        "shift_down must preserve +inf with typical linear layer n"
    );
    // shift_up on upper (-inf) must preserve -inf
    assert_eq!(
        shift_up_n_ulps(f32::NEG_INFINITY, 10),
        f32::NEG_INFINITY,
        "shift_up must preserve -inf (infeasible upper sentinel)"
    );
    assert_eq!(
        shift_up_n_ulps(f32::NEG_INFINITY, 770),
        f32::NEG_INFINITY,
        "shift_up must preserve -inf with typical linear layer n"
    );
}

#[test]
fn test_shift_n_ulps_monotonicity() {
    // Shifting up by more ULPs should give a larger result
    let x = 1.0f32;
    let s5 = shift_up_n_ulps(x, 5);
    let s10 = shift_up_n_ulps(x, 10);
    let s100 = shift_up_n_ulps(x, 100);
    assert!(
        s5 < s10,
        "shift_up_n_ulps monotonicity failed: {s5} should be < {s10}"
    );
    assert!(
        s10 < s100,
        "shift_up_n_ulps monotonicity failed: {s10} should be < {s100}"
    );

    // Shifting down by more ULPs should give a smaller result
    let d5 = shift_down_n_ulps(x, 5);
    let d10 = shift_down_n_ulps(x, 10);
    let d100 = shift_down_n_ulps(x, 100);
    assert!(
        d5 > d10,
        "shift_down_n_ulps monotonicity failed: {d5} should be > {d10}"
    );
    assert!(
        d10 > d100,
        "shift_down_n_ulps monotonicity failed: {d10} should be > {d100}"
    );
}

#[test]
fn test_shift_n_ulps_typical_linear_layer() {
    // Simulate rounding for a linear layer with 768 input features
    // (typical transformer hidden size)
    let result = 42.0f32;
    let n = 768 + 2; // in_features + matmul_combine + bias
    let widened_up = shift_up_n_ulps(result, n);
    let widened_down = shift_down_n_ulps(result, n);
    // Should widen by ~770 ULPs in each direction
    assert!(
        widened_up > result,
        "shift_up_n_ulps should widen above {result}, got {widened_up}"
    );
    assert!(
        widened_down < result,
        "shift_down_n_ulps should widen below {result}, got {widened_down}"
    );
    // Width should be approximately 2 * 770 * ULP(42.0)
    // ULP(42.0) = 2^(ceil(log2(42))-23) ≈ 2^(6-23) = 2^-17 ≈ 7.6e-6
    let width = widened_up - widened_down;
    assert!(
        width > 0.0,
        "widened interval width should be positive, got {width}"
    );
    assert!(
        width < 0.02,
        "widened interval width should stay below 0.02, got {width}"
    ); // Should be ~0.012 for 770 ULPs at magnitude 42
}

// -- Regression tests for #2788: large-n overflow in zero/cross-zero branches --

/// shift_up from 0.0 with n >= inf_bits must saturate to +inf, not produce NaN.
#[test]
fn test_shift_up_from_zero_large_n_saturates_2788() {
    let inf_bits = f32::INFINITY.to_bits(); // 0x7F80_0000

    // Exactly at inf boundary
    let r = shift_up_n_ulps(0.0, inf_bits);
    assert_eq!(r, f32::INFINITY, "n=inf_bits from 0 must saturate to +inf");

    // Past inf boundary
    let r = shift_up_n_ulps(0.0, inf_bits + 1);
    assert_eq!(r, f32::INFINITY, "n>inf_bits from 0 must saturate to +inf");

    // Maximum possible n
    let r = shift_up_n_ulps(0.0, u32::MAX);
    assert_eq!(r, f32::INFINITY, "n=u32::MAX from 0 must saturate to +inf");
    assert!(!r.is_nan(), "must not produce NaN");
}

/// shift_down from 0.0 with n >= inf_bits must saturate to -inf, not produce NaN.
#[test]
fn test_shift_down_from_zero_large_n_saturates_2788() {
    let inf_bits = f32::INFINITY.to_bits();

    let r = shift_down_n_ulps(0.0, inf_bits);
    assert_eq!(
        r,
        f32::NEG_INFINITY,
        "n=inf_bits from 0 must saturate to -inf"
    );

    let r = shift_down_n_ulps(0.0, inf_bits + 1);
    assert_eq!(
        r,
        f32::NEG_INFINITY,
        "n>inf_bits from 0 must saturate to -inf"
    );

    let r = shift_down_n_ulps(0.0, u32::MAX);
    assert_eq!(
        r,
        f32::NEG_INFINITY,
        "n=u32::MAX from 0 must saturate to -inf"
    );
    assert!(!r.is_nan(), "must not produce NaN");
}

/// shift_up from negative crossing zero with large remainder must saturate to +inf.
/// Example: x = -1.0, n = u32::MAX → crosses zero, remainder is huge.
#[test]
fn test_shift_up_cross_zero_large_remainder_saturates_2788() {
    let x = -1.0f32;
    let r = shift_up_n_ulps(x, u32::MAX);
    assert_eq!(
        r,
        f32::INFINITY,
        "huge cross-zero remainder must saturate to +inf"
    );
    assert!(!r.is_nan(), "must not produce NaN");
}

/// shift_down from positive crossing zero with large remainder must saturate to -inf.
/// Example: x = 1.0, n = u32::MAX → crosses zero, remainder is huge.
#[test]
fn test_shift_down_cross_zero_large_remainder_saturates_2788() {
    let x = 1.0f32;
    let r = shift_down_n_ulps(x, u32::MAX);
    assert_eq!(
        r,
        f32::NEG_INFINITY,
        "huge cross-zero remainder must saturate to -inf"
    );
    assert!(!r.is_nan(), "must not produce NaN");
}

/// shift_up from smallest negative subnormal with n that barely crosses zero
/// and has large positive remainder must saturate.
#[test]
fn test_shift_up_cross_zero_boundary_2788() {
    // Smallest negative subnormal: magnitude_bits = 1
    let x = f32::from_bits(0x8000_0001);
    let inf_bits = f32::INFINITY.to_bits();

    // n = 1 + inf_bits → remainder = inf_bits, should saturate
    let r = shift_up_n_ulps(x, 1 + inf_bits);
    assert_eq!(r, f32::INFINITY, "remainder at inf_bits must saturate");

    // n = 2 + inf_bits → remainder = inf_bits + 1, should saturate
    let r = shift_up_n_ulps(x, 2 + inf_bits);
    assert_eq!(r, f32::INFINITY, "remainder past inf_bits must saturate");
}

/// shift_down from smallest positive subnormal with n that barely crosses zero
/// and has large negative remainder must saturate.
#[test]
fn test_shift_down_cross_zero_boundary_2788() {
    // Smallest positive subnormal: bits = 1
    let x = f32::from_bits(1);
    let inf_bits = f32::INFINITY.to_bits();

    // n = 1 + inf_bits → remainder = inf_bits, should saturate
    let r = shift_down_n_ulps(x, 1 + inf_bits);
    assert_eq!(r, f32::NEG_INFINITY, "remainder at inf_bits must saturate");

    // n = 2 + inf_bits → remainder past inf_bits, should saturate
    let r = shift_down_n_ulps(x, 2 + inf_bits);
    assert_eq!(
        r,
        f32::NEG_INFINITY,
        "remainder past inf_bits must saturate"
    );
}

/// Existing small-n tests must not regress after saturation fix.
#[test]
fn test_shift_small_n_unchanged_2788() {
    // These are small n values that should not be affected by the saturation fix.
    assert_eq!(shift_up_n_ulps(0.0, 1).to_bits(), 1);
    assert_eq!(shift_up_n_ulps(0.0, 100).to_bits(), 100);
    assert_eq!(shift_down_n_ulps(0.0, 1).to_bits(), 0x8000_0001);
    assert_eq!(shift_down_n_ulps(0.0, 100).to_bits(), 0x8000_0064);

    // Cross-zero with small remainder
    let x = f32::from_bits(0x8000_0001); // smallest negative subnormal
    let r = shift_up_n_ulps(x, 5);
    assert_eq!(r.to_bits(), 4, "small cross-zero remainder should be exact");
}
