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
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let bits = x.to_bits();
    if x.is_sign_positive() {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// Return the next representable f32 below `x`.
#[inline]
pub fn next_down_f32(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(0x8000_0001);
    }
    let bits = x.to_bits();
    if x.is_sign_positive() {
        f32::from_bits(bits - 1)
    } else {
        f32::from_bits(bits + 1)
    }
}
