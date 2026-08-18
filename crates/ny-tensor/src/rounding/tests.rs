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

#[test]
fn test_directed_steps_preserve_subnormal_bit_order() {
    let positive = f32::from_bits(7);
    let negative = f32::from_bits(0x8000_0007);
    assert_eq!(next_up_f32(positive).to_bits(), 8);
    assert_eq!(next_down_f32(positive).to_bits(), 6);
    assert_eq!(next_up_f32(negative).to_bits(), 0x8000_0006);
    assert_eq!(next_down_f32(negative).to_bits(), 0x8000_0008);
    assert_eq!(shift_up_n_ulps(positive, 3).to_bits(), 10);
    assert_eq!(shift_down_n_ulps(positive, 3).to_bits(), 4);
    assert_eq!(shift_up_n_ulps(negative, 3).to_bits(), 0x8000_0004);
    assert_eq!(shift_down_n_ulps(negative, 3).to_bits(), 0x8000_000a);
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

// ---------------------------------------------------------------------------
// Directed f64 -> f32 narrowing casts.
//
// The invariant that matters for soundness is the ENCLOSURE one:
//   cast_f64_to_f32_down(x) <= x <= cast_f64_to_f32_up(x)
// The invariant that matters for TIGHTNESS is that neither is ever worse than
// the `next_down_f32(x as f32)` idiom they replace, and that both are exact
// whenever `x` is representable.
// ---------------------------------------------------------------------------

/// A spread of f64 values that exercises exact-representable, mid-ULP,
/// subnormal, huge, and boundary inputs.
fn directed_cast_probe_values() -> Vec<f64> {
    let mut xs = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        // Exactly representable in f32 (they came FROM f32).
        f64::from(0.1_f32),
        f64::from(-0.1_f32),
        f64::from(0.3_f32),
        f64::from(1e-10_f32),
        f64::from(f32::MAX),
        f64::from(f32::MIN),
        f64::from(f32::MIN_POSITIVE),
        // NOT representable in f32: the f64 nearest to the decimal literal.
        0.1,
        -0.1,
        0.3,
        1.0 / 3.0,
        -1.0 / 3.0,
        // Beyond the f32 range in both directions.
        1e300,
        -1e300,
        f64::from(f32::MAX) * 1.5,
        f64::from(f32::MIN) * 1.5,
        // Below the smallest f32 subnormal but nonzero.
        1e-60,
        -1e-60,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    // A deterministic sweep just above and below each f32 grid point near 1.0.
    let first_bits = 1.0_f32.to_bits();
    for bits in first_bits..first_bits + 64 {
        let g = f64::from(f32::from_bits(bits));
        xs.push(g);
        xs.push(g * (1.0 + 1e-9));
        xs.push(g * (1.0 - 1e-9));
    }
    xs
}

#[test]
fn directed_casts_enclose_the_f64_value() {
    for x in directed_cast_probe_values() {
        let lo = cast_f64_to_f32_down(x);
        let hi = cast_f64_to_f32_up(x);
        assert!(
            f64::from(lo) <= x,
            "cast_f64_to_f32_down({x:e}) = {lo:e} must not exceed x"
        );
        assert!(
            f64::from(hi) >= x,
            "cast_f64_to_f32_up({x:e}) = {hi:e} must not fall below x"
        );
    }
}

#[test]
fn directed_casts_are_the_tightest_such_f32() {
    for x in directed_cast_probe_values() {
        let lo = cast_f64_to_f32_down(x);
        if lo.is_finite() {
            let tighter = next_up_f32(lo);
            assert!(
                f64::from(tighter) > x,
                "cast_f64_to_f32_down({x:e}) = {lo:e} is not maximal: {tighter:e} also fits"
            );
        }
        let hi = cast_f64_to_f32_up(x);
        if hi.is_finite() {
            let tighter = next_down_f32(hi);
            assert!(
                f64::from(tighter) < x,
                "cast_f64_to_f32_up({x:e}) = {hi:e} is not minimal: {tighter:e} also fits"
            );
        }
    }
}

#[test]
fn directed_casts_are_never_looser_than_the_idiom_they_replace() {
    for x in directed_cast_probe_values() {
        if x.is_nan() {
            continue;
        }
        let idiom_lo = next_down_f32(x as f32);
        // Only meaningful where the idiom is itself a valid lower bound; see
        // `the_idiom_is_unsound_on_overflow_which_is_why_this_primitive_exists`.
        if f64::from(idiom_lo) <= x {
            assert!(
                cast_f64_to_f32_down(x) >= idiom_lo,
                "down-cast of {x:e} must not be below the next_down_f32 idiom"
            );
        }
        let idiom_hi = next_up_f32(x as f32);
        if f64::from(idiom_hi) >= x {
            assert!(
                cast_f64_to_f32_up(x) <= idiom_hi,
                "up-cast of {x:e} must not be above the next_up_f32 idiom"
            );
        }
    }
}

/// The `next_down_f32(x as f32)` idiom is sound for every f64 that lands inside
/// the f32 range, and UNSOUND for every one that does not.
///
/// `next_down_f32` documents `+-inf -> +-inf` as deliberate: it preserves the
/// infeasible-interval sentinel. That is right for a value already at infinity
/// and wrong for a cast that *overflowed to* infinity, because the composed
/// idiom then returns `+inf` as a LOWER bound on a large finite number. A bound
/// pair of `[+inf, +inf]` reads as an infeasible interval downstream, which is
/// the false-proof direction.
///
/// This test pins the defect rather than the fix, so it stays honest if
/// `next_down_f32`'s own sentinel contract is ever revisited.
#[test]
fn the_idiom_is_unsound_on_overflow_which_is_why_this_primitive_exists() {
    // A finite f64 above the f32 range. Reachable wherever two near-max f32
    // bounds are widened to f64, added, and narrowed back.
    let over = f64::from(f32::MAX) + f64::from(f32::MAX);
    assert!(over.is_finite(), "the probe value must be a finite f64");

    let idiom_lo = next_down_f32(over as f32);
    assert_eq!(
        idiom_lo,
        f32::INFINITY,
        "the idiom returns +inf here, which is the defect"
    );
    assert!(
        f64::from(idiom_lo) > over,
        "and +inf is strictly ABOVE the value it is supposed to lower-bound"
    );

    // The directed cast returns the tightest f32 that is genuinely below it.
    let fixed = cast_f64_to_f32_down(over);
    assert_eq!(fixed, f32::MAX);
    assert!(f64::from(fixed) <= over);

    // Mirror for the upper bound.
    let under = -over;
    let idiom_hi = next_up_f32(under as f32);
    assert_eq!(idiom_hi, f32::NEG_INFINITY);
    assert!(f64::from(idiom_hi) < under);
    assert_eq!(cast_f64_to_f32_up(under), f32::MIN);
}

#[test]
fn directed_casts_are_exact_on_values_that_came_from_f32() {
    // This is the case the idiom gives away a full ULP on, and it is the common
    // one: any bound widened to f64 for accumulation and narrowed straight back.
    let mut bits = 0_u32;
    let mut checked = 0_usize;
    while bits < 0x7f80_0000 {
        let v = f32::from_bits(bits);
        let x = f64::from(v);
        assert_eq!(
            cast_f64_to_f32_down(x).to_bits(),
            v.to_bits(),
            "down-cast must be exact on the f32 grid point {v:e}"
        );
        assert_eq!(
            cast_f64_to_f32_up(x).to_bits(),
            v.to_bits(),
            "up-cast must be exact on the f32 grid point {v:e}"
        );
        // Same for the negative twin.
        let n = f32::from_bits(bits | 0x8000_0000);
        let nx = f64::from(n);
        assert_eq!(cast_f64_to_f32_down(nx).to_bits(), n.to_bits());
        assert_eq!(cast_f64_to_f32_up(nx).to_bits(), n.to_bits());
        checked += 1;
        // Stride the exponent/mantissa space rather than all 2^31 patterns.
        bits += 0x0001_0001;
    }
    assert!(
        checked > 30_000,
        "the sweep should cover the f32 grid widely"
    );
}

#[test]
fn directed_casts_saturate_correctly_outside_the_f32_range() {
    // A value above f32::MAX: the tightest f32 lower bound is f32::MAX, and the
    // only correct upper bound is +inf.
    let huge = f64::from(f32::MAX) * 2.0;
    assert_eq!(cast_f64_to_f32_down(huge), f32::MAX);
    assert_eq!(cast_f64_to_f32_up(huge), f32::INFINITY);

    let tiny = f64::from(f32::MIN) * 2.0;
    assert_eq!(cast_f64_to_f32_up(tiny), f32::MIN);
    assert_eq!(cast_f64_to_f32_down(tiny), f32::NEG_INFINITY);

    assert_eq!(cast_f64_to_f32_down(f64::INFINITY), f32::INFINITY);
    assert_eq!(cast_f64_to_f32_up(f64::NEG_INFINITY), f32::NEG_INFINITY);
}

#[test]
fn directed_casts_pass_nan_through() {
    assert!(cast_f64_to_f32_down(f64::NAN).is_nan());
    assert!(cast_f64_to_f32_up(f64::NAN).is_nan());
}

// ---------------------------------------------------------------------------
// Directed interval addition.
// ---------------------------------------------------------------------------

#[test]
fn directed_adds_bracket_the_exact_sum() {
    // Reference the exact sum in f64, which is exact for any two f32 operands
    // (f64 has more than twice f32's significand, so no f32+f32 can round).
    // The long probe below is a decimal transcription of a specific f32 bit
    // pattern; shortening it is a change to the value being probed, so the
    // digits stay and the lint is accepted here.
    #[allow(clippy::excessive_precision)]
    let probes = [
        0.0_f32,
        -0.0,
        1.0,
        -1.0,
        10.0,
        -0.5855390429496765,
        0.1,
        -0.1,
        1e-30,
        -1e-30,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        16_777_216.0, // 2^24: adding 1 is not representable
        1.0 / 3.0,
    ];
    for &a in &probes {
        for &b in &probes {
            let exact = f64::from(a) + f64::from(b);
            let lo = add_down_f32(a, b);
            let hi = add_up_f32(a, b);
            if exact.is_finite() {
                assert!(
                    f64::from(lo) <= exact,
                    "add_down_f32({a:e}, {b:e}) = {lo:e} exceeds the exact sum {exact:e}"
                );
                assert!(
                    f64::from(hi) >= exact,
                    "add_up_f32({a:e}, {b:e}) = {hi:e} is below the exact sum {exact:e}"
                );
            }
            assert!(
                lo <= hi,
                "directed adds must stay ordered for {a:e} + {b:e}"
            );
        }
    }
}

#[test]
fn directed_adds_are_exact_when_the_addition_is_exact() {
    // No slack is given away when nothing was rounded — this is what keeps the
    // soundness fix from costing tightness on the overwhelmingly common case.
    for (a, b, want) in [
        (1.0_f32, 2.0_f32, 3.0_f32),
        (0.5, 0.25, 0.75),
        (-10.0, 10.0, 0.0),
        (0.0, 0.0, 0.0),
        (1e10, 0.0, 1e10),
    ] {
        assert_eq!(add_down_f32(a, b), want, "{a} + {b} is exact");
        assert_eq!(add_up_f32(a, b), want, "{a} + {b} is exact");
    }
}

#[test]
fn directed_adds_separate_when_the_addition_rounds() {
    // 2^24 + 1 is not representable: round-to-nearest gives 2^24, which is
    // BELOW the true sum, so a plain add would publish an upper bound that
    // excludes the truth.
    let a = 16_777_216.0_f32; // 2^24
    let b = 1.0_f32;
    assert_eq!(a + b, a, "the plain add absorbs the operand");

    let hi = add_up_f32(a, b);
    assert!(
        f64::from(hi) >= f64::from(a) + f64::from(b),
        "the up-add must clear the absorbed operand, got {hi}"
    );
    assert!(hi > a, "and it must actually move");

    let lo = add_down_f32(a, b);
    assert_eq!(lo, a, "the down-add is already correct and must not move");
}

#[test]
fn directed_subs_bracket_the_exact_difference() {
    // The case that broke InstanceNorm: 10 - 0.5855390429496765 rounds up by
    // 0.19 ULP under round-to-nearest, lifting a LOWER bound above the truth.
    let a = 10.0_f32;
    let b = 0.585_539_04_f32;
    let exact = f64::from(a) - f64::from(b);
    let plain = a - b;
    assert!(
        f64::from(plain) > exact,
        "the plain subtract rounds UP here, which is why a lower bound needs sub_down_f32"
    );
    assert!(f64::from(sub_down_f32(a, b)) <= exact);
    assert!(f64::from(sub_up_f32(a, b)) >= exact);
}

#[test]
fn directed_adds_saturate_correctly_on_overflow() {
    // f32::MAX + f32::MAX is a FINITE real above the f32 range. The plain add
    // gives +inf, and `next_down_f32` would preserve it as a sentinel, so the
    // naive spelling returns +inf as a LOWER bound. Same defect class as
    // `the_idiom_is_unsound_on_overflow_which_is_why_this_primitive_exists`.
    let m = f32::MAX;
    assert_eq!(m + m, f32::INFINITY, "the plain add overflows");
    assert_eq!(
        add_down_f32(m, m),
        f32::MAX,
        "the lower bound must stay finite"
    );
    assert_eq!(
        add_up_f32(m, m),
        f32::INFINITY,
        "and +inf is the only upper bound"
    );

    assert_eq!(add_up_f32(-m, -m), f32::MIN);
    assert_eq!(add_down_f32(-m, -m), f32::NEG_INFINITY);
}

#[test]
fn directed_adds_pass_nan_and_saturate_at_infinity() {
    assert!(add_down_f32(f32::NAN, 1.0).is_nan());
    assert!(add_up_f32(1.0, f32::NAN).is_nan());
    assert_eq!(add_up_f32(f32::INFINITY, 1.0), f32::INFINITY);
    assert_eq!(add_down_f32(f32::NEG_INFINITY, -1.0), f32::NEG_INFINITY);
}

// ---------------------------------------------------------------------------
// Directed interval multiplication and division.
// ---------------------------------------------------------------------------

fn mul_div_probe_values() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        0.5,
        -0.5,
        3.0,
        -7.0,
        0.1,
        -0.1,
        1.0 / 3.0,
        1e-20,
        -1e-20,
        1e20,
        -1e20,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1), // smallest subnormal
        f32::MAX,
        f32::MIN,
        16_777_217.0, // not representable; becomes 2^24
        0.585_539_04,
        1.707_828,
    ]
}

#[test]
fn the_f64_product_of_two_f32s_is_exact() {
    // The premise `mul_down_f32` rests on: 24 + 24 <= 53 significand bits and
    // the exponent range stays far inside f64's, so no rounding occurs.
    for &a in &mul_div_probe_values() {
        for &b in &mul_div_probe_values() {
            let product = f64::from(a) * f64::from(b);
            if !product.is_finite() {
                continue;
            }
            // Re-derive the product with exact integer significand arithmetic:
            // if the f64 multiply had rounded, splitting it back out would not
            // reproduce it bit-for-bit.
            let back = product / f64::from(b);
            if b != 0.0 && f64::from(a) != 0.0 && back.is_finite() {
                assert_eq!(
                    back,
                    f64::from(a),
                    "f64 product of {a:e} and {b:e} was not exact"
                );
            }
        }
    }
}

#[test]
fn directed_muls_bracket_the_exact_product() {
    for &a in &mul_div_probe_values() {
        for &b in &mul_div_probe_values() {
            let exact = f64::from(a) * f64::from(b); // exact, per the test above
            let lo = mul_down_f32(a, b);
            let hi = mul_up_f32(a, b);
            if exact.is_finite() {
                assert!(
                    f64::from(lo) <= exact,
                    "mul_down_f32({a:e}, {b:e}) = {lo:e} exceeds the exact product {exact:e}"
                );
                assert!(
                    f64::from(hi) >= exact,
                    "mul_up_f32({a:e}, {b:e}) = {hi:e} is below the exact product {exact:e}"
                );
            }
            assert!(
                lo <= hi,
                "directed muls must stay ordered for {a:e} * {b:e}"
            );
        }
    }
}

#[test]
fn directed_muls_are_exact_when_the_product_is_representable() {
    // Scaling by a power of two, multiplying by 1 or 0, and small integer
    // products all round nowhere — the overwhelmingly common case in a network.
    for (a, b, want) in [
        (3.0_f32, 4.0_f32, 12.0_f32),
        (1.5, 2.0, 3.0),
        (0.1, 1.0, 0.1),
        (-2.5, 4.0, -10.0),
        (7.0, 0.0, 0.0),
        // NOT `(1e20, 1e-20, 1.0)`: neither f32 literal is the exact decimal,
        // so their product is genuinely just below 1 and the down-cast
        // correctly returns 0.99999994. Reciprocal-looking pairs are only exact
        // when both factors are powers of two.
        (1024.0, 1.0 / 1024.0, 1.0),
    ] {
        assert_eq!(
            mul_down_f32(a, b),
            want,
            "{a} * {b} is exactly representable"
        );
        assert_eq!(mul_up_f32(a, b), want, "{a} * {b} is exactly representable");
    }
}

#[test]
fn directed_muls_separate_when_the_product_is_not_representable() {
    // 0.1 * 0.1 is not representable in f32; the interval must straddle it.
    let lo = mul_down_f32(0.1, 0.1);
    let hi = mul_up_f32(0.1, 0.1);
    let exact = f64::from(0.1_f32) * f64::from(0.1_f32);
    assert!(lo < hi, "an inexact product must report nonzero width");
    assert!(f64::from(lo) <= exact && exact <= f64::from(hi));
    assert_eq!(
        next_up_f32(lo),
        hi,
        "and the straddle must be tight: exactly one ULP"
    );
}

#[test]
fn directed_muls_saturate_correctly_on_overflow() {
    let m = f32::MAX;
    assert_eq!(m * m, f32::INFINITY, "the plain multiply overflows");
    assert_eq!(
        mul_down_f32(m, m),
        f32::MAX,
        "the lower bound must stay finite"
    );
    assert_eq!(mul_up_f32(m, m), f32::INFINITY);
    assert_eq!(mul_up_f32(m, -m), f32::MIN);
    assert_eq!(mul_down_f32(m, -m), f32::NEG_INFINITY);
}

#[test]
fn directed_divs_bracket_the_exact_quotient() {
    for &a in &mul_div_probe_values() {
        for &b in &mul_div_probe_values() {
            if b == 0.0 {
                continue;
            }
            let lo = div_down_f32(a, b);
            let hi = div_up_f32(a, b);
            assert!(
                lo <= hi,
                "directed divs must stay ordered for {a:e} / {b:e}"
            );
            // Verify against the exact rational: lo <= a/b  <=>  lo*b <= a
            // (comparison done in f64 where lo*b is exact).
            if lo.is_finite() && b.is_finite() {
                let scaled = f64::from(lo) * f64::from(b);
                if b > 0.0 {
                    assert!(
                        scaled <= f64::from(a) || !scaled.is_finite(),
                        "div_down_f32({a:e}, {b:e}) = {lo:e} is above the true quotient"
                    );
                } else {
                    assert!(
                        scaled >= f64::from(a) || !scaled.is_finite(),
                        "div_down_f32({a:e}, {b:e}) = {lo:e} is above the true quotient"
                    );
                }
            }
            if hi.is_finite() && b.is_finite() {
                let scaled = f64::from(hi) * f64::from(b);
                if b > 0.0 {
                    assert!(
                        scaled >= f64::from(a) || !scaled.is_finite(),
                        "div_up_f32({a:e}, {b:e}) = {hi:e} is below the true quotient"
                    );
                } else {
                    assert!(
                        scaled <= f64::from(a) || !scaled.is_finite(),
                        "div_up_f32({a:e}, {b:e}) = {hi:e} is below the true quotient"
                    );
                }
            }
        }
    }
}

#[test]
fn directed_divs_are_exact_when_the_quotient_is_representable() {
    for (a, b, want) in [
        (12.0_f32, 4.0_f32, 3.0_f32),
        (1.0, 2.0, 0.5),
        (-10.0, 4.0, -2.5),
        (0.1, 1.0, 0.1),
        (0.0, 5.0, 0.0),
    ] {
        assert_eq!(
            div_down_f32(a, b),
            want,
            "{a} / {b} is exactly representable"
        );
        assert_eq!(div_up_f32(a, b), want, "{a} / {b} is exactly representable");
    }
}

#[test]
fn directed_divs_straddle_an_irrational_quotient_by_one_ulp() {
    let lo = div_down_f32(1.0, 3.0);
    let hi = div_up_f32(1.0, 3.0);
    assert!(lo < hi);
    assert_eq!(next_up_f32(lo), hi, "the straddle must be exactly one ULP");
    assert!(f64::from(lo) * 3.0 <= 1.0);
    assert!(f64::from(hi) * 3.0 >= 1.0);
}
