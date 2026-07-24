// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Edge case tests for `BoundedScalar` f64→f32 directed rounding.
//!
//! Covers: NaN, ±infinity, subnormals, zero, exact f32 values, and finite
//! overflow beyond the f32 range.

use ny_tensor::BoundedScalar;

/// NaN must propagate through both directions (no silent value substitution).
#[test]
fn test_f64_to_f32_directed_rounding_nan() {
    let nan = f64::NAN;
    assert!(
        <f64 as BoundedScalar>::to_f32_down(nan).is_nan(),
        "to_f32_down(NaN) must return NaN"
    );
    assert!(
        <f64 as BoundedScalar>::to_f32_up(nan).is_nan(),
        "to_f32_up(NaN) must return NaN"
    );

    // Negative NaN
    let neg_nan = f64::from_bits(0xFFF8_0000_0000_0000);
    assert!(neg_nan.is_nan());
    assert!(
        <f64 as BoundedScalar>::to_f32_down(neg_nan).is_nan(),
        "to_f32_down(negative NaN) must return NaN"
    );
    assert!(
        <f64 as BoundedScalar>::to_f32_up(neg_nan).is_nan(),
        "to_f32_up(negative NaN) must return NaN"
    );
}

/// ±infinity must pass through unchanged — valid bound sentinels.
#[test]
fn test_f64_to_f32_directed_rounding_infinity() {
    assert_eq!(
        <f64 as BoundedScalar>::to_f32_down(f64::INFINITY),
        f32::INFINITY,
        "to_f32_down(+inf) must be +inf"
    );
    assert_eq!(
        <f64 as BoundedScalar>::to_f32_up(f64::INFINITY),
        f32::INFINITY,
        "to_f32_up(+inf) must be +inf"
    );
    assert_eq!(
        <f64 as BoundedScalar>::to_f32_down(f64::NEG_INFINITY),
        f32::NEG_INFINITY,
        "to_f32_down(-inf) must be -inf"
    );
    assert_eq!(
        <f64 as BoundedScalar>::to_f32_up(f64::NEG_INFINITY),
        f32::NEG_INFINITY,
        "to_f32_up(-inf) must be -inf"
    );
}

/// Subnormal f64 values smaller than the smallest f32 subnormal (~1.4e-45).
/// These round to ±0.0 in f32; directed rounding must widen correctly.
#[test]
fn test_f64_to_f32_directed_rounding_subnormals() {
    // Positive subnormal smaller than f32 min subnormal
    let tiny_pos = 1.0e-46_f64;
    assert!(tiny_pos > 0.0);
    let lo = <f64 as BoundedScalar>::to_f32_down(tiny_pos);
    let hi = <f64 as BoundedScalar>::to_f32_up(tiny_pos);
    assert!(
        f64::from(lo) <= tiny_pos,
        "to_f32_down({tiny_pos:e})={lo:e} must be <= original"
    );
    assert!(
        f64::from(hi) >= tiny_pos,
        "to_f32_up({tiny_pos:e})={hi:e} must be >= original"
    );
    assert!(lo <= hi, "lo={lo:e} > hi={hi:e}");

    // Negative subnormal smaller than f32 min subnormal
    let tiny_neg = -1.0e-46_f64;
    let lo = <f64 as BoundedScalar>::to_f32_down(tiny_neg);
    let hi = <f64 as BoundedScalar>::to_f32_up(tiny_neg);
    assert!(
        f64::from(lo) <= tiny_neg,
        "to_f32_down({tiny_neg:e})={lo:e} must be <= original"
    );
    assert!(
        f64::from(hi) >= tiny_neg,
        "to_f32_up({tiny_neg:e})={hi:e} must be >= original"
    );
    assert!(lo <= hi, "lo={lo:e} > hi={hi:e}");
}

/// Both +0.0 and -0.0 must round to zero (exact representation).
#[test]
fn test_f64_to_f32_directed_rounding_zero() {
    let lo = <f64 as BoundedScalar>::to_f32_down(0.0_f64);
    let hi = <f64 as BoundedScalar>::to_f32_up(0.0_f64);
    assert_eq!(lo, 0.0f32, "to_f32_down(0) must be 0");
    assert_eq!(hi, 0.0f32, "to_f32_up(0) must be 0");

    let lo = <f64 as BoundedScalar>::to_f32_down(-0.0_f64);
    let hi = <f64 as BoundedScalar>::to_f32_up(-0.0_f64);
    assert_eq!(lo, 0.0f32, "to_f32_down(-0) must be 0 or -0");
    assert_eq!(hi, 0.0f32, "to_f32_up(-0) must be 0 or -0");
}

/// Values exactly representable in f32 must be returned unchanged.
#[test]
fn test_f64_to_f32_directed_rounding_exact_f32() {
    let cases = [
        1.0f64,
        -1.0,
        0.5,
        -0.5,
        42.0,
        -42.0,
        f64::from(f32::MAX),
        f64::from(f32::MIN),
    ];
    for val in cases {
        let lo = <f64 as BoundedScalar>::to_f32_down(val);
        let hi = <f64 as BoundedScalar>::to_f32_up(val);
        assert_eq!(
            f64::from(lo),
            val,
            "to_f32_down({val}) should be exact for f32-representable value"
        );
        assert_eq!(
            f64::from(hi),
            val,
            "to_f32_up({val}) should be exact for f32-representable value"
        );
    }
}

#[test]
fn test_f64_to_f32_directed_rounding_overflow() {
    let big_pos = f64::from(f32::MAX) * 1.5;
    assert!(big_pos.is_finite());
    let lo = <f64 as BoundedScalar>::to_f32_down(big_pos);
    assert_eq!(lo, f32::MAX, "to_f32_down must return f32::MAX, not +inf");
    assert!(f64::from(lo) <= big_pos, "soundness: result <= original");

    let big_neg = f64::from(f32::MIN) * 1.5;
    assert!(big_neg.is_finite());
    let hi = <f64 as BoundedScalar>::to_f32_up(big_neg);
    assert_eq!(hi, f32::MIN, "to_f32_up must return f32::MIN, not -inf");
    assert!(f64::from(hi) >= big_neg, "soundness: result >= original");
}
