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
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        // Smallest positive subnormal.
        return f32::from_bits(1);
    }

    let bits = x.to_bits();
    if x.is_sign_positive() {
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
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        // Smallest negative subnormal.
        return f32::from_bits(0x8000_0001);
    }

    let bits = x.to_bits();
    if x.is_sign_positive() {
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
    if n == 0 || x.is_nan() || x.is_infinite() {
        return x;
    }
    if n == 1 {
        return next_up_f32(x);
    }

    let bits = x.to_bits();

    // Saturation threshold: bits for +inf (0x7F80_0000).
    // Any ULP count at or above this would produce inf or NaN bit patterns.
    let inf_bits = f32::INFINITY.to_bits();

    if x == 0.0 {
        // 0.0 → n-th positive subnormal, saturating to +inf.
        return f32::from_bits(n.min(inf_bits));
    }

    if x.is_sign_positive() {
        // Moving away from zero: add n to bit pattern.
        // Saturate to +inf (0x7F80_0000) on overflow.
        f32::from_bits(bits.saturating_add(n).min(inf_bits))
    } else {
        // Negative moving toward zero: subtract n from bit pattern.
        let magnitude_bits = bits & 0x7FFF_FFFF;
        if n >= magnitude_bits {
            // Would cross zero or overshoot. Compute remainder past zero
            // and return that many ULPs into the positive subnormals.
            let remainder = n - magnitude_bits; // n past -0.0
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
    if n == 0 || x.is_nan() || x.is_infinite() {
        return x;
    }
    if n == 1 {
        return next_down_f32(x);
    }

    let bits = x.to_bits();

    // Saturation threshold: magnitude bits for inf (0x7F80_0000).
    let inf_bits = f32::INFINITY.to_bits();

    if x == 0.0 {
        // 0.0 → n-th negative subnormal, saturating to -inf.
        return f32::from_bits(0x8000_0000 | n.min(inf_bits));
    }

    if x.is_sign_negative() {
        // Moving away from zero: add n to magnitude bits.
        // Saturate to -inf (0xFF80_0000) on overflow.
        let magnitude = bits & 0x7FFF_FFFF;
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

#[cfg(test)]
mod tests;
