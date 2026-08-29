// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! IEEE 754 directed rounding utilities for f32.
//!
//! These are the same implementations as `ny_tensor::rounding` but
//! provided here to avoid pulling in the full ny-tensor dependency
//! (ndarray, rayon, half) for proof compilation.

/// Return the next representable f32 above `x`.
#[inline]
pub fn next_up_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= f32::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f32::from_bits(1);
    }
    if bits & 0x8000_0000 == 0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// Return the next representable f32 below `x`.
#[inline]
pub fn next_down_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let magnitude = bits & 0x7fff_ffff;
    if magnitude >= f32::INFINITY.to_bits() {
        return x;
    }
    if magnitude == 0 {
        return f32::from_bits(0x8000_0001);
    }
    if bits & 0x8000_0000 == 0 {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}

/// Narrow an f64 to f32 rounding toward -inf: the LARGEST f32 that is `<= x`.
///
/// Byte-for-byte the same algorithm as `ny_tensor::cast_f64_to_f32_down`
/// (ny-tensor/src/rounding.rs:195). It is duplicated rather than imported for
/// the reason given in this module's header: ny-relaxation must not depend on
/// ny-tensor, so that `proofs/kani` can compile the scalar surface without
/// ndarray/rayon/half.
///
/// It must stay bit-identical to ny-tensor's, and `tests/drift.rs` asserts
/// exactly that by comparing mirror against production with `band_eq`. NOTE
/// that `ny_core::f64_to_f32_down` is NOT a substitute: it FLUSHES the f32
/// subnormal range (returning `0.0` for any positive `x` below
/// `f32::MIN_POSITIVE`) and maps NaN to -inf. Both are sound, but neither is
/// bit-equal, and `drift.rs`'s grid contains `f32::from_bits(1)` and a mid
/// subnormal — at the smallest positive subnormal the two disagree by
/// 1.1754944e-38 vs 1.4012985e-45.
///
/// # Special cases
/// - NaN → NaN (comparisons are false, so no step is taken)
/// - `x` too large negative to represent → -inf (a correct lower bound)
/// - `x` too large positive to represent → `f32::MAX`, the tightest f32 under
///   it. Handled explicitly because `next_down_f32` treats ±Inf as an
///   infeasible-interval SENTINEL and returns it unchanged, which would hand
///   back `+inf` as a LOWER bound for a large finite `x`.
#[inline]
pub fn cast_f64_to_f32_down(x: f64) -> f32 {
    let candidate = x as f32;
    if candidate.is_nan() {
        return candidate;
    }
    if candidate == f32::INFINITY {
        return if x.is_infinite() { candidate } else { f32::MAX };
    }
    if f64::from(candidate) <= x {
        candidate
    } else {
        next_down_f32(candidate)
    }
}

/// Narrow an f64 to f32 rounding toward +inf: the SMALLEST f32 that is `>= x`.
///
/// The upper-bound mirror of [`cast_f64_to_f32_down`]; see its docs for why
/// this is duplicated here and why `ny_core::f64_to_f32_up` cannot stand in.
///
/// # Special cases
/// - NaN → NaN (comparisons are false, so no step is taken)
/// - `x` too large positive to represent → +inf (a correct upper bound)
/// - `x` too large negative to represent → `f32::MIN`, the tightest f32 above
///   it — without this a large negative finite `x` would get `-inf` back as an
///   UPPER bound.
#[inline]
pub fn cast_f64_to_f32_up(x: f64) -> f32 {
    let candidate = x as f32;
    if candidate.is_nan() {
        return candidate;
    }
    if candidate == f32::NEG_INFINITY {
        return if x.is_infinite() { candidate } else { f32::MIN };
    }
    if f64::from(candidate) >= x {
        candidate
    } else {
        next_up_f32(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::{cast_f64_to_f32_down, cast_f64_to_f32_up, next_down_f32, next_up_f32};

    /// The overflow arms exist so a large FINITE value never comes back as an
    /// infinity in the direction that has to be finite. Without them the
    /// generic `next_*_f32` step returns ±Inf unchanged (sentinel semantics),
    /// which would publish `+inf` as a lower bound / `-inf` as an upper bound.
    #[test]
    fn directed_casts_saturate_outward_not_to_infinity() {
        // Just above f32::MAX, still finite in f64.
        let big = f64::from(f32::MAX) * 1.000_000_1;
        assert_eq!(cast_f64_to_f32_down(big), f32::MAX);
        assert_eq!(cast_f64_to_f32_up(big), f32::INFINITY);
        assert_eq!(cast_f64_to_f32_up(-big), f32::MIN);
        assert_eq!(cast_f64_to_f32_down(-big), f32::NEG_INFINITY);
        // Genuine infinities pass through unchanged.
        assert_eq!(cast_f64_to_f32_down(f64::INFINITY), f32::INFINITY);
        assert_eq!(cast_f64_to_f32_up(f64::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(cast_f64_to_f32_down(f64::NAN).is_nan());
        assert!(cast_f64_to_f32_up(f64::NAN).is_nan());
    }

    /// Subnormals are PRESERVED, not flushed. This is the property that
    /// separates these from `ny_core::f64_to_f32_{down,up}` and the reason
    /// ny-tensor's version had to be duplicated rather than swapped for the
    /// ny-core one already in this crate's dependency graph.
    #[test]
    fn directed_casts_preserve_the_f32_subnormal_range() {
        // GELU(f32::from_bits(1)) under the erf form: 0.5 * x * (1 + erf(~0)).
        let tiny = f64::from(f32::from_bits(1)) * 0.5;
        assert_eq!(cast_f64_to_f32_down(tiny), 0.0);
        assert_eq!(cast_f64_to_f32_up(tiny), f32::from_bits(1));
        assert!(cast_f64_to_f32_up(tiny) < f32::MIN_POSITIVE);
    }

    /// Bracketing, at zero tolerance: for any f64, the pair must straddle it.
    #[test]
    fn directed_casts_bracket_their_input() {
        for &x in &[
            0.0_f64,
            -0.0,
            1.0,
            -1.0,
            0.1,
            -0.1,
            1e-45,
            -1e-45,
            1e38,
            -1e38,
            std::f64::consts::PI,
            -0.169_971_207_479_903_66,
        ] {
            let lo = cast_f64_to_f32_down(x);
            let hi = cast_f64_to_f32_up(x);
            assert!(f64::from(lo) <= x, "down({x:e}) = {lo:e} is above x");
            assert!(f64::from(hi) >= x, "up({x:e}) = {hi:e} is below x");
        }
    }

    #[test]
    fn directed_steps_follow_subnormal_bit_order() {
        let positive = f32::from_bits(7);
        let negative = f32::from_bits(0x8000_0007);
        assert_eq!(next_up_f32(positive).to_bits(), 8);
        assert_eq!(next_down_f32(positive).to_bits(), 6);
        assert_eq!(next_up_f32(negative).to_bits(), 0x8000_0006);
        assert_eq!(next_down_f32(negative).to_bits(), 0x8000_0008);
    }
}
