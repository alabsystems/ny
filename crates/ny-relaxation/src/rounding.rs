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

#[cfg(test)]
mod tests {
    use super::{next_down_f32, next_up_f32};

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
