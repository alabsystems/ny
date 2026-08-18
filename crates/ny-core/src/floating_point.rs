// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Bit-exact floating-point representation conversions used by proof paths.

use crate::error::{NyError, Result};

const F64_FRACTION_BITS: u32 = 52;
const F64_EXPONENT_BIAS: i32 = 1023;
const F64_MAGNITUDE_MASK: u64 = 0x7fff_ffff_ffff_ffff;
const F64_INFINITY_BITS: u64 = 0x7ff0_0000_0000_0000;
const F32_INFINITY_BITS: u32 = 0x7f80_0000;
const F64_INTERVAL_ENVIRONMENT_REQUIREMENT: &str = "f64 interval proof requires IEEE-754 \
    binary64 round-to-nearest with gradual underflow (FTZ/DAZ disabled)";

/// True when the observed results establish the required binary64 environment.
///
/// Kept separate from the live operations so rejection cases can be tested
/// without mutating a thread's floating-point control state.
#[inline]
fn f64_interval_environment_probe_passes(
    half_min_normal: f64,
    recovered_min_subnormal: f64,
    added_subnormals: f64,
    upper_halfway: f64,
    lower_halfway: f64,
) -> bool {
    half_min_normal.to_bits() == 0x0008_0000_0000_0000
        && recovered_min_subnormal.to_bits() == 1
        && added_subnormals.to_bits() == 2
        // At 1.0, 2^-53 is halfway to the next value above and 2^-54
        // is halfway to the next value below. Ties-to-even selects 1.0
        // in both directions; the directed modes fail at least one probe.
        && upper_halfway.to_bits() == 1.0_f64.to_bits()
        && lower_halfway.to_bits() == 1.0_f64.to_bits()
}

/// Whether the calling thread uses binary64 round-to-nearest/ties-to-even and
/// preserves subnormal operands and results.
///
/// `black_box` prevents constant folding: the operations execute in the
/// calling thread's active floating-point environment. The probes cover a
/// subnormal result (FTZ), a subnormal input (DAZ), subnormal addition, and
/// halfway additions on both sides of 1.0 (all directed rounding modes).
#[inline]
#[must_use]
pub fn has_f64_interval_proof_environment() -> bool {
    let half = std::hint::black_box(0.5_f64);
    let min_normal = std::hint::black_box(f64::MIN_POSITIVE);
    let min_subnormal = std::hint::black_box(f64::from_bits(1));
    let two_subnormals = std::hint::black_box(f64::from_bits(2));
    let one = std::hint::black_box(1.0_f64);
    let half_ulp_above_one = std::hint::black_box(f64::from_bits(u64::from(1023_u16 - 53) << 52));
    let half_ulp_below_one = std::hint::black_box(f64::from_bits(u64::from(1023_u16 - 54) << 52));
    let half_min_normal = std::hint::black_box(min_normal * half);
    let recovered_min_subnormal = std::hint::black_box(two_subnormals * half);
    let added_subnormals = std::hint::black_box(min_subnormal + min_subnormal);
    let upper_halfway = std::hint::black_box(one + half_ulp_above_one);
    let lower_halfway = std::hint::black_box(one - half_ulp_below_one);
    f64_interval_environment_probe_passes(
        half_min_normal,
        recovered_min_subnormal,
        added_subnormals,
        upper_halfway,
        lower_halfway,
    )
}

/// Refuse outward interval arithmetic unless binary64 round-to-nearest with
/// gradual underflow is active.
///
/// Adjacent-float and Higham widening do not provide their documented
/// enclosures under a directed rounding mode or when a subnormal operand or
/// result is silently replaced by zero.
pub fn require_f64_interval_proof_environment() -> Result<()> {
    if !has_f64_interval_proof_environment() {
        return Err(NyError::SoundnessRefusal(
            F64_INTERVAL_ENVIRONMENT_REQUIREMENT.to_string(),
        ));
    }
    Ok(())
}

/// Convert a binary32 bit pattern to its exact binary64 representation without
/// presenting a binary32 subnormal operand to a hardware conversion.
///
/// An ordinary `f32 as f64`/`f64::from(f32)` may be executed by an instruction
/// whose DAZ mode treats a nonzero binary32 subnormal as signed zero. Reading
/// the representation with [`f32::to_bits`] and constructing the binary64 bits
/// directly makes the conversion independent of that mode. Every finite
/// binary32 value is represented exactly; signed zero, infinities, and NaNs
/// preserve their class and sign (NaN payload preservation is not promised).
#[inline]
#[must_use]
pub fn f32_to_f64_exact(value: f32) -> f64 {
    let bits = value.to_bits();
    let sign = u64::from(bits >> 31) << 63;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x7f_ffff;

    match (exponent, fraction) {
        (0, 0) => f64::from_bits(sign),
        (0, _) => {
            let leading = fraction.ilog2();
            let unbiased_exponent = leading as i32 - 149;
            let exponent64 = (unbiased_exponent + F64_EXPONENT_BIAS) as u64;
            let leading_bit = 1_u32 << leading;
            let fraction64 = u64::from(fraction - leading_bit) << (F64_FRACTION_BITS - leading);
            f64::from_bits(sign | (exponent64 << F64_FRACTION_BITS) | fraction64)
        }
        (0xff, 0) => f64::from_bits(sign | (0x7ff_u64 << F64_FRACTION_BITS)),
        // Always construct a quiet NaN. Preserving a binary32 payload is not
        // part of this helper's contract, and emitting a signaling NaN would
        // make even diagnostic consumers depend on the active FP exception
        // configuration.
        (0xff, _) => f64::from_bits(sign | (0x7ff_u64 << F64_FRACTION_BITS) | (1_u64 << 51)),
        _ => {
            let unbiased_exponent = exponent as i32 - 127;
            let exponent64 = (unbiased_exponent + F64_EXPONENT_BIAS) as u64;
            let fraction64 = u64::from(fraction) << (F64_FRACTION_BITS - 23);
            f64::from_bits(sign | (exponent64 << F64_FRACTION_BITS) | fraction64)
        }
    }
}

/// Conservative absolute error when evaluating an f64-authored affine line
/// as binary32 `slope_f32 * x + intercept_f32`.
///
/// The returned radius covers f64→f32 slope conversion, multiplication
/// rounding, addition rounding, and two binary32 minimum-normal floors for
/// FTZ/DAZ environments. `max_abs_x` is accepted as binary32 and decoded by
/// representation so a DAZ conversion cannot erase a subnormal domain. It
/// assumes `|x| <= max_abs_x`; invalid or overflowing inputs return `+∞` so
/// callers can fail closed.
#[inline]
#[must_use]
pub fn f32_affine_eval_error(
    slope_f64: f64,
    slope_f32: f32,
    intercept_f64: f64,
    max_abs_x: f32,
) -> f64 {
    if !slope_f64.is_finite()
        || !slope_f32.is_finite()
        || !intercept_f64.is_finite()
        || !max_abs_x.is_finite()
        || max_abs_x < 0.0
    {
        return f64::INFINITY;
    }

    let eps = f32::EPSILON as f64;
    let slope_f32_exact = f32_to_f64_exact(slope_f32);
    let max_abs_x_exact = f32_to_f64_exact(max_abs_x);
    let product_mag = slope_f32_exact.abs() * max_abs_x_exact;
    if !product_mag.is_finite() || product_mag > f32::MAX as f64 {
        return f64::INFINITY;
    }
    let underflow_floor = f32::MIN_POSITIVE as f64;
    let slope_err = (slope_f64 - slope_f32_exact).abs() * max_abs_x_exact;
    // DAZ may replace a subnormal coefficient or input by zero. The input
    // term is charged unconditionally because a range described only by its
    // maximum magnitude may still contain subnormals.
    let daz_slope_err = if slope_f32 != 0.0 && !slope_f32.is_normal() {
        product_mag
    } else {
        0.0
    };
    let daz_input_err = slope_f32_exact.abs() * underflow_floor.min(max_abs_x_exact);
    let mul_err = product_mag * eps + daz_slope_err + daz_input_err + underflow_floor;
    let add_mag = product_mag + mul_err + intercept_f64.abs();
    if !add_mag.is_finite() || add_mag > f32::MAX as f64 {
        return f64::INFINITY;
    }
    let add_err = add_mag * eps + underflow_floor;
    slope_err + mul_err + add_err
}

#[inline]
fn next_down_f32_bits(value: f32) -> f32 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let negative = bits & 0x8000_0000 != 0;
    let next = match (negative, magnitude) {
        (_, m) if m > F32_INFINITY_BITS => return value,
        (true, F32_INFINITY_BITS) => return value,
        (_, 0) => 0x8000_0001,
        (false, _) => bits - 1,
        (true, _) => bits + 1,
    };
    f32::from_bits(next)
}

#[inline]
fn next_up_f32_bits(value: f32) -> f32 {
    let bits = value.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    let negative = bits & 0x8000_0000 != 0;
    let next = match (negative, magnitude) {
        (_, m) if m > F32_INFINITY_BITS => return value,
        (false, F32_INFINITY_BITS) => return value,
        (_, 0) => 1,
        (false, _) => bits + 1,
        (true, _) => bits - 1,
    };
    f32::from_bits(next)
}

/// Convert a binary64 value to a binary32 lower endpoint without ever
/// publishing a binary32 subnormal.
///
/// A hardware conversion may flush a binary32-subnormal result to signed
/// zero. This function classifies that range from the binary64 bits first:
/// positive values widen down to `+0`, while negative values widen down to
/// `-f32::MIN_POSITIVE`. Outside that range it checks the converted value via
/// [`f32_to_f64_exact`] and bit-steps down when necessary. NaN becomes
/// `-∞`; infinities and finite overflow are handled directionally.
#[inline]
#[must_use]
pub fn f64_to_f32_down(value: f64) -> f32 {
    let bits = value.to_bits();
    let magnitude = bits & F64_MAGNITUDE_MASK;
    let negative = bits >> 63 != 0;
    if magnitude > F64_INFINITY_BITS {
        return f32::NEG_INFINITY;
    }
    if magnitude == F64_INFINITY_BITS {
        return if negative {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    if magnitude == 0 {
        return f32::from_bits((negative as u32) << 31);
    }

    let binary32_min_normal_bits = u64::from((F64_EXPONENT_BIAS - 126) as u16) << F64_FRACTION_BITS;
    if magnitude < binary32_min_normal_bits {
        return if negative { -f32::MIN_POSITIVE } else { 0.0 };
    }

    let nearest = value as f32;
    match nearest.to_bits() {
        0x7f80_0000 => f32::MAX,
        0xff80_0000 => f32::NEG_INFINITY,
        _ if f32_to_f64_exact(nearest) <= value => nearest,
        _ => next_down_f32_bits(nearest),
    }
}

/// Convert a binary64 value to a binary32 upper endpoint without ever
/// publishing a binary32 subnormal.
///
/// Positive values in the binary32-subnormal range widen up to
/// `f32::MIN_POSITIVE`; negative values widen up to `-0`. Outside that range
/// the converted value is checked exactly and bit-stepped up when necessary.
/// NaN becomes `+∞`; infinities and finite overflow are handled directionally.
#[inline]
#[must_use]
pub fn f64_to_f32_up(value: f64) -> f32 {
    let bits = value.to_bits();
    let magnitude = bits & F64_MAGNITUDE_MASK;
    let negative = bits >> 63 != 0;
    if magnitude > F64_INFINITY_BITS {
        return f32::INFINITY;
    }
    if magnitude == F64_INFINITY_BITS {
        return if negative {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }
    if magnitude == 0 {
        return f32::from_bits((negative as u32) << 31);
    }

    let binary32_min_normal_bits = u64::from((F64_EXPONENT_BIAS - 126) as u16) << F64_FRACTION_BITS;
    if magnitude < binary32_min_normal_bits {
        return if negative {
            f32::from_bits(0x8000_0000)
        } else {
            f32::MIN_POSITIVE
        };
    }

    let nearest = value as f32;
    match nearest.to_bits() {
        0xff80_0000 => f32::MIN,
        0x7f80_0000 => f32::INFINITY,
        _ if f32_to_f64_exact(nearest) >= value => nearest,
        _ => next_up_f32_bits(nearest),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_eval_error_covers_conversion_and_two_f32_operations() {
        let slope64 = 0.1_f64;
        let slope32 = slope64 as f32;
        let intercept = -0.234_429_77_f64;
        let max_abs_x = 1.5_f32;
        let error = f32_affine_eval_error(slope64, slope32, intercept, max_abs_x);
        let eps = f32::EPSILON as f64;
        let product_mag = f32_to_f64_exact(slope32).abs() * f32_to_f64_exact(max_abs_x);
        assert!(error >= (slope64 - f32_to_f64_exact(slope32)).abs() * f32_to_f64_exact(max_abs_x));
        assert!(error >= product_mag * eps);
        assert!(error >= (product_mag + intercept.abs()) * eps);
        assert!(error.is_finite());
    }

    #[test]
    fn affine_eval_error_fails_closed_on_invalid_or_overflowing_inputs() {
        assert!(f32_affine_eval_error(f64::NAN, 1.0, 0.0, 1.0).is_infinite());
        assert!(f32_affine_eval_error(1.0, 1.0, 0.0, -1.0).is_infinite());
        assert!(f32_affine_eval_error(f64::MAX, f32::MAX, 0.0, f32::MAX).is_infinite());
    }

    #[test]
    fn exact_conversion_preserves_binary32_classes_and_subnormals() {
        let smallest = f32::from_bits(1);
        let largest_subnormal = f32::from_bits(0x007f_ffff);
        let cases = [
            0.0,
            -0.0,
            smallest,
            -smallest,
            largest_subnormal,
            f32::MIN_POSITIVE,
            1.0,
            -1.5,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for value in cases {
            let exact = f32_to_f64_exact(value);
            if value.is_normal() || value == 0.0 {
                assert_eq!(exact, value as f64);
            } else if value.is_infinite() {
                assert_eq!(exact.is_infinite(), value.is_infinite());
                assert_eq!(exact.is_sign_negative(), value.is_sign_negative());
            }
        }
        assert_eq!(
            f32_to_f64_exact(smallest).to_bits(),
            (u64::from(1023_u16 - 149)) << 52
        );
        assert!(f32_to_f64_exact(f32::NAN).is_nan());
    }

    #[test]
    fn interval_probe_accepts_live_environment_and_rejects_unsafe_modes() {
        require_f64_interval_proof_environment()
            .expect("test host must use round-to-nearest and preserve binary64 subnormals");
        assert!(f64_interval_environment_probe_passes(
            f64::from_bits(0x0008_0000_0000_0000),
            f64::from_bits(1),
            f64::from_bits(2),
            1.0,
            1.0,
        ));
        assert!(!f64_interval_environment_probe_passes(
            0.0,
            f64::from_bits(1),
            f64::from_bits(2),
            1.0,
            1.0,
        ));
        assert!(!f64_interval_environment_probe_passes(
            f64::from_bits(0x0008_0000_0000_0000),
            0.0,
            0.0,
            1.0,
            1.0,
        ));
        assert!(!f64_interval_environment_probe_passes(
            f64::from_bits(0x0008_0000_0000_0000),
            f64::from_bits(1),
            f64::from_bits(2),
            1.0_f64.next_up(),
            1.0,
        ));
        assert!(!f64_interval_environment_probe_passes(
            f64::from_bits(0x0008_0000_0000_0000),
            f64::from_bits(1),
            f64::from_bits(2),
            1.0,
            1.0_f64.next_down(),
        ));
    }

    #[test]
    fn outward_binary32_conversions_are_ftz_safe_and_directional() {
        let min_normal = f32_to_f64_exact(f32::MIN_POSITIVE);
        let below_min_normal = min_normal.next_down();
        assert_eq!(f64_to_f32_down(below_min_normal).to_bits(), 0);
        assert_eq!(
            f64_to_f32_up(below_min_normal).to_bits(),
            f32::MIN_POSITIVE.to_bits()
        );
        assert_eq!(
            f64_to_f32_down(-below_min_normal).to_bits(),
            (-f32::MIN_POSITIVE).to_bits()
        );
        assert_eq!(f64_to_f32_up(-below_min_normal).to_bits(), 0x8000_0000);

        let tiny = f64::from_bits(1);
        assert_eq!(f64_to_f32_down(tiny).to_bits(), 0);
        assert_eq!(f64_to_f32_up(tiny), f32::MIN_POSITIVE);
        assert_eq!(f64_to_f32_down(-tiny), -f32::MIN_POSITIVE);
        assert_eq!(f64_to_f32_up(-tiny).to_bits(), 0x8000_0000);

        let halfway_above_one = 1.0 + f64::from_bits(u64::from(1023_u16 - 24) << 52);
        assert_eq!(f64_to_f32_down(halfway_above_one), 1.0);
        assert_eq!(f64_to_f32_up(halfway_above_one), 1.0_f32.next_up());
        assert_eq!(f64_to_f32_down(f64::MAX), f32::MAX);
        assert_eq!(f64_to_f32_up(-f64::MAX), f32::MIN);
        assert_eq!(f64_to_f32_down(f64::NAN), f32::NEG_INFINITY);
        assert_eq!(f64_to_f32_up(f64::NAN), f32::INFINITY);
    }
}
