// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

/// Compute the next representable f32 above x (toward +inf).
///
/// Used for directed rounding in interval arithmetic to ensure soundness.
/// For any computed upper bound, apply `next_up_f32` to guarantee the
/// true value is not above the stored bound.
///
/// # Special cases
/// - NaN → NaN
/// - ±inf → ±inf (preserves infeasible sentinels for interval arithmetic)
/// - 0.0 → smallest positive subnormal
#[inline]
#[must_use]
pub fn next_up_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= f32::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        // Smallest positive subnormal.
        return f32::from_bits(1);
    }

    if bits & 0x8000_0000 == 0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// Compute the next representable f32 below x (toward -inf).
///
/// Used for directed rounding in interval arithmetic to ensure soundness.
/// For any computed lower bound, apply `next_down_f32` to guarantee the
/// true value is not below the stored bound.
///
/// # Special cases
/// - NaN → NaN
/// - ±inf → ±inf (preserves infeasible sentinels for interval arithmetic)
/// - 0.0 → smallest negative subnormal
#[inline]
#[must_use]
pub fn next_down_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= f32::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        // Smallest negative subnormal.
        return f32::from_bits(0x8000_0001);
    }

    if bits & 0x8000_0000 == 0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// Shift an f32 value upward by `n` ULPs (toward +inf).
///
/// For a dot product of `n` terms, floating-point rounding error is bounded
/// by `n` ULPs of the result. Widening the upper bound by `n` ULPs ensures
/// the true mathematical result is below the stored bound.
///
/// # Special cases
/// - NaN → NaN
/// - ±inf → ±inf (preserves infeasible sentinels for interval arithmetic)
/// - Saturates to +inf rather than overflowing
#[inline]
#[must_use]
pub fn shift_up_n_ulps(x: f32, n: u32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if n == 0 || magnitude >= f32::INFINITY.to_bits() {
        return x;
    }
    if n == 1 {
        return next_up_f32(x);
    }

    // Saturation threshold: bits for +inf (0x7F80_0000).
    // Any ULP count at or above this would produce inf or NaN bit patterns.
    let inf_bits = f32::INFINITY.to_bits();

    if magnitude == 0 {
        // 0.0 → n-th positive subnormal, saturating to +inf.
        return f32::from_bits(n.min(inf_bits));
    }

    if bits & 0x8000_0000 == 0 {
        // Moving away from zero: add n to bit pattern.
        // Saturate to +inf (0x7F80_0000) on overflow.
        f32::from_bits(bits.saturating_add(n).min(inf_bits))
    } else {
        // Negative moving toward zero: subtract n from bit pattern.
        if n >= magnitude {
            // Would cross zero or overshoot. Compute remainder past zero
            // and return that many ULPs into the positive subnormals.
            let remainder = n - magnitude; // n past -0.0
            if remainder == 0 {
                return 0.0;
            }
            // remainder ULPs above +0.0, saturating to +inf.
            f32::from_bits(remainder.min(inf_bits))
        } else {
            f32::from_bits(bits - n)
        }
    }
}

/// Shift an f32 value downward by `n` ULPs (toward -inf).
///
/// For a dot product of `n` terms, floating-point rounding error is bounded
/// by `n` ULPs of the result. Widening the lower bound by `n` ULPs ensures
/// the true mathematical result is above the stored bound.
///
/// # Special cases
/// - NaN → NaN
/// - ±inf → ±inf (preserves infeasible sentinels for interval arithmetic)
/// - Saturates to -inf rather than overflowing
#[inline]
#[must_use]
pub fn shift_down_n_ulps(x: f32, n: u32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if n == 0 || magnitude >= f32::INFINITY.to_bits() {
        return x;
    }
    if n == 1 {
        return next_down_f32(x);
    }

    // Saturation threshold: magnitude bits for inf (0x7F80_0000).
    let inf_bits = f32::INFINITY.to_bits();

    if magnitude == 0 {
        // 0.0 → n-th negative subnormal, saturating to -inf.
        return f32::from_bits(0x8000_0000 | n.min(inf_bits));
    }

    if bits & 0x8000_0000 != 0 {
        // Moving away from zero: add n to magnitude bits.
        // Saturate to -inf (0xFF80_0000) on overflow.
        let new_magnitude = magnitude.saturating_add(n).min(inf_bits);
        f32::from_bits(0x8000_0000 | new_magnitude)
    } else {
        // Positive moving toward zero: subtract n from bit pattern.
        if n >= bits {
            // Would cross zero or overshoot. Compute remainder past zero
            // and return that many ULPs into the negative subnormals.
            let remainder = n - bits; // n past +0.0
            if remainder == 0 {
                return -0.0;
            }
            // remainder ULPs below -0.0, saturating to -inf.
            f32::from_bits(0x8000_0000 | remainder.min(inf_bits))
        } else {
            f32::from_bits(bits - n)
        }
    }
}

/// Narrow an f64 to f32 rounding toward -inf: the LARGEST f32 that is `<= x`.
///
/// This is IEEE-754 `roundTowardNegative` for the f64 -> f32 conversion, done
/// without touching the FPU rounding mode.
///
/// # Why this is not `next_down_f32(x as f32)`
///
/// The idiom `next_down_f32(x as f32)` appears throughout the tree at f64 -> f32
/// bound boundaries. It is SOUND — the round-to-nearest cast moves by at most
/// half an ULP, so one full ULP down always clears it — but it is *round toward
/// -inf plus one unconditional ULP*. When the cast is already exact (`x` is
/// representable in f32, which is the case for every value that merely round-
/// trips through f64, and for every reduction that happened to be exact) it
/// gives away a full ULP for nothing.
///
/// This function steps ONLY when the round-to-nearest cast landed above `x`.
/// It is therefore never looser than the idiom and never wider than necessary:
/// the result is the tightest f32 lower bound on `x` that exists.
///
/// # Special cases
/// - NaN → NaN (comparisons are false, so no step is taken)
/// - `x` too large negative to represent → -inf (a correct lower bound)
/// - `x` too large positive to represent → +inf, then stepped to `f32::MAX`,
///   which is the tightest representable f32 that is still `<= x`
#[inline]
#[must_use]
pub fn cast_f64_to_f32_down(x: f64) -> f32 {
    let candidate = x as f32;
    if candidate.is_nan() {
        return candidate;
    }
    // OVERFLOW, handled before the generic step. `next_down_f32` deliberately
    // treats +-inf as an infeasible-interval SENTINEL and returns it unchanged,
    // so the generic path below would hand back `+inf` as a LOWER bound for a
    // large finite `x` — above the value it is supposed to sit under.
    if candidate == f32::INFINITY {
        // `x` is either genuinely +inf (then +inf is the answer) or a finite
        // value above `f32::MAX` (then `f32::MAX` is the tightest f32 under it).
        return if x.is_infinite() { candidate } else { f32::MAX };
    }
    // `candidate == -inf` needs no special case: -inf is a correct lower bound
    // for any `x` that underflowed to it, and it is the only one available.
    if f64::from(candidate) <= x {
        candidate
    } else {
        next_down_f32(candidate)
    }
}

/// Narrow an f64 to f32 rounding toward +inf: the SMALLEST f32 that is `>= x`.
///
/// The upper-bound mirror of [`cast_f64_to_f32_down`]; see its docs for why this
/// is preferable to `next_up_f32(x as f32)`.
///
/// # Special cases
/// - NaN → NaN (comparisons are false, so no step is taken)
/// - `x` too large positive to represent → +inf (a correct upper bound)
/// - `x` too large negative to represent → -inf, then stepped to `f32::MIN`,
///   which is the tightest representable f32 that is still `>= x`
#[inline]
#[must_use]
pub fn cast_f64_to_f32_up(x: f64) -> f32 {
    let candidate = x as f32;
    if candidate.is_nan() {
        return candidate;
    }
    // The mirror of the overflow case in [`cast_f64_to_f32_down`]: without it a
    // large negative finite `x` would get `-inf` back as an UPPER bound.
    if candidate == f32::NEG_INFINITY {
        return if x.is_infinite() { candidate } else { f32::MIN };
    }
    if f64::from(candidate) >= x {
        candidate
    } else {
        next_up_f32(candidate)
    }
}

/// Knuth's `TwoSum` in f32: `(hi, lo)` with `hi == fl(a + b)` and, for all
/// finite `a`, `b` without overflow, `a + b == hi + lo` in the reals.
#[inline]
fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let hi = a + b;
    let a_virtual = hi - b;
    let b_virtual = hi - a_virtual;
    (hi, (a - a_virtual) + (b - b_virtual))
}

/// `a + b` rounded toward -inf: the largest f32 that is `<= a + b` exactly.
///
/// # Why interval addition needs this
///
/// `a + b` in f32 is round-to-NEAREST, so a lower-bound addition can land
/// strictly ABOVE the true sum by up to half an ULP. Interval arithmetic is
/// only sound if each endpoint moves outward, which is why every interval op
/// that adds two bounds must round in the direction of its own endpoint.
///
/// `TwoSum` reports exactly what the addition lost, so this steps ONLY when the
/// add was inexact — an add of two exactly-representable values costs nothing.
///
/// # Special cases
/// - NaN operands → NaN (the residual is NaN, so no step is taken)
/// - overflow to `-inf` → `-inf`, a correct lower bound
#[inline]
#[must_use]
pub fn add_down_f32(a: f32, b: f32) -> f32 {
    let (hi, lo) = two_sum_f32(a, b);
    // OVERFLOW first: `TwoSum`'s residual is meaningless once `hi` is not
    // finite, and `next_down_f32` preserves infinities as sentinels, so the
    // generic path would hand back `+inf` as a LOWER bound on a finite sum.
    if !hi.is_finite() {
        return if hi == f32::INFINITY && a.is_finite() && b.is_finite() {
            f32::MAX
        } else {
            // Genuine +-inf operands, NaN, or a negative overflow: -inf is a
            // correct lower bound and +inf/NaN are correct passthroughs.
            hi
        };
    }
    if lo < 0.0 {
        next_down_f32(hi)
    } else {
        hi
    }
}

/// `a + b` rounded toward +inf: the smallest f32 that is `>= a + b` exactly.
///
/// The upper-bound mirror of [`add_down_f32`]; see its docs.
#[inline]
#[must_use]
pub fn add_up_f32(a: f32, b: f32) -> f32 {
    let (hi, lo) = two_sum_f32(a, b);
    // The mirror of the overflow case in [`add_down_f32`].
    if !hi.is_finite() {
        return if hi == f32::NEG_INFINITY && a.is_finite() && b.is_finite() {
            f32::MIN
        } else {
            hi
        };
    }
    if lo > 0.0 {
        next_up_f32(hi)
    } else {
        hi
    }
}

/// `a - b` rounded toward -inf: the largest f32 that is `<= a - b` exactly.
#[inline]
#[must_use]
pub fn sub_down_f32(a: f32, b: f32) -> f32 {
    add_down_f32(a, -b)
}

/// `a - b` rounded toward +inf: the smallest f32 that is `>= a - b` exactly.
#[inline]
#[must_use]
pub fn sub_up_f32(a: f32, b: f32) -> f32 {
    add_up_f32(a, -b)
}

/// `a * b` rounded toward -inf: the largest f32 that is `<= a * b` exactly.
///
/// # Why this is exact rather than approximate
///
/// The product of two f32s is ALWAYS exact in f64. The significand needs at
/// most `24 + 24 = 48` bits and f64 carries 53; the exponent reaches at most
/// `2^256` and down to `2^-298` (two minimal subnormals), both far inside f64's
/// normal range. So `f64::from(a) * f64::from(b)` commits no rounding at all,
/// and the only rounding in the whole operation is the narrowing cast — which
/// [`cast_f64_to_f32_down`] performs in the correct direction and only when it
/// is actually needed.
///
/// This makes the result the tightest f32 lower bound on the true product that
/// exists, with no error-term bookkeeping, no `TwoProduct` underflow floor, and
/// no special case for subnormals.
///
/// # Why interval multiplication needs it
///
/// `[a,b] * [c,d]` takes the min and max over four endpoint products. Each is a
/// plain f32 `*` today, i.e. round-to-NEAREST, so the min can round UP and the
/// max can round DOWN — both inward, both unsound.
#[inline]
#[must_use]
pub fn mul_down_f32(a: f32, b: f32) -> f32 {
    cast_f64_to_f32_down(f64::from(a) * f64::from(b))
}

/// `a * b` rounded toward +inf: the smallest f32 that is `>= a * b` exactly.
///
/// The upper-bound mirror of [`mul_down_f32`]; see its docs for why the f64
/// product is exact.
#[inline]
#[must_use]
pub fn mul_up_f32(a: f32, b: f32) -> f32 {
    cast_f64_to_f32_up(f64::from(a) * f64::from(b))
}

/// `a / b` rounded toward -inf: the largest f32 that is `<= a / b` exactly.
///
/// Unlike multiplication, division of two f32s is NOT exact in f64 — `1.0/3.0`
/// is irrational in binary. So the f64 quotient is certified by its own exact
/// remainder: for IEEE round-to-nearest division, `fma(-q, y, x)` computes
/// `x - q*y` EXACTLY, and the sign of `r / y` says which side of `q` the true
/// quotient lies on. One `next_down_f64` covers it, because `|r / y|` never
/// exceeds half an ULP of `q`.
#[inline]
#[must_use]
pub fn div_down_f32(a: f32, b: f32) -> f32 {
    let x = f64::from(a);
    let y = f64::from(b);
    let quotient = x / y;
    if !quotient.is_finite() || y == 0.0 {
        // Division by zero, or a non-finite operand: hand the raw value to the
        // narrowing cast, which passes NaN through and saturates infinities.
        return cast_f64_to_f32_down(quotient);
    }
    let remainder = f64::mul_add(-quotient, y, x);
    // `remainder / y < 0` exactly when the operands' signs differ, and that is
    // the case where the true quotient sits BELOW `quotient`.
    let true_is_below = (remainder < 0.0) != (y < 0.0) && remainder != 0.0;
    cast_f64_to_f32_down(if true_is_below {
        next_down_f64(quotient)
    } else {
        quotient
    })
}

/// `a / b` rounded toward +inf: the smallest f32 that is `>= a / b` exactly.
///
/// The upper-bound mirror of [`div_down_f32`].
#[inline]
#[must_use]
pub fn div_up_f32(a: f32, b: f32) -> f32 {
    let x = f64::from(a);
    let y = f64::from(b);
    let quotient = x / y;
    if !quotient.is_finite() || y == 0.0 {
        return cast_f64_to_f32_up(quotient);
    }
    let remainder = f64::mul_add(-quotient, y, x);
    let true_is_above = (remainder < 0.0) == (y < 0.0) && remainder != 0.0;
    cast_f64_to_f32_up(if true_is_above {
        next_up_f64(quotient)
    } else {
        quotient
    })
}

/// The next representable f64 below `x`. Local to keep `ny-tensor` free of a
/// dependency on `ny-core` for two bit-twiddles.
#[inline]
fn next_down_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f64::from_bits(0x8000_0000_0000_0001);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

/// The next representable f64 above `x`. See [`next_down_f64`].
#[inline]
fn next_up_f64(x: f64) -> f64 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff_ffff_ffff;
    if magnitude >= f64::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f64::from_bits(1);
    }
    if bits & 0x8000_0000_0000_0000 == 0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

#[cfg(test)]
mod tests;
