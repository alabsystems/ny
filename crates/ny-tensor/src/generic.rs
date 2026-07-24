// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Generic, consumer-facing wrappers around [`crate::BoundedTensor`].
//!
//! ny uses `f32` internally for performance and GPU compatibility.
//! External consumers (e.g. DSL verifiers) may want to work with other scalar
//! types (`f64`, `f16`, etc). This module provides a thin adapter that converts
//! bounds to `f32` for internal use while preserving interval soundness on
//! conversion for wider types.

use std::marker::PhantomData;

use half::{bf16, f16};
use ndarray::ArrayD;
use ny_core::Result;

use crate::{next_down_f32, next_up_f32, BoundedTensor};

/// A scalar type that can be used to represent interval bounds.
///
/// Implementations must provide conversion to and from `f32`. For soundness,
/// conversions of *bounds* should be directed: lower bounds round down and upper
/// bounds round up.
pub trait BoundedScalar: Copy + PartialOrd + Send + Sync + 'static {
    /// Convert a value to `f32` (default rounding).
    fn to_f32(self) -> f32;

    /// Convert a lower bound to `f32`, rounding toward -∞ as needed.
    #[inline]
    fn to_f32_down(self) -> f32 {
        self.to_f32()
    }

    /// Convert an upper bound to `f32`, rounding toward +∞ as needed.
    #[inline]
    fn to_f32_up(self) -> f32 {
        self.to_f32()
    }

    /// Convert an `f32` back to the scalar type.
    fn from_f32(v: f32) -> Self;

    /// Convert a lower bound from `f32`, rounding toward -∞ as needed.
    #[inline]
    fn from_f32_down(v: f32) -> Self {
        Self::from_f32(v)
    }

    /// Convert an upper bound from `f32`, rounding toward +∞ as needed.
    #[inline]
    fn from_f32_up(v: f32) -> Self {
        Self::from_f32(v)
    }
}

#[inline]
fn next_up_f16(x: f16) -> f16 {
    if x.is_nan() || x == f16::INFINITY {
        return x;
    }
    if x == f16::from_f32(0.0) {
        return f16::from_bits(1);
    }

    let bits = x.to_bits();
    let is_sign_positive = (bits & 0x8000) == 0;
    if is_sign_positive {
        f16::from_bits(bits + 1)
    } else {
        f16::from_bits(bits - 1)
    }
}

#[inline]
fn next_down_f16(x: f16) -> f16 {
    if x.is_nan() || x == f16::NEG_INFINITY {
        return x;
    }
    if x == f16::from_f32(0.0) {
        return f16::from_bits(0x8001);
    }

    let bits = x.to_bits();
    let is_sign_positive = (bits & 0x8000) == 0;
    if is_sign_positive {
        f16::from_bits(bits - 1)
    } else {
        f16::from_bits(bits + 1)
    }
}

#[inline]
fn next_up_bf16(x: bf16) -> bf16 {
    if x.is_nan() || x == bf16::INFINITY {
        return x;
    }
    if x == bf16::from_f32(0.0) {
        return bf16::from_bits(1);
    }

    let bits = x.to_bits();
    let is_sign_positive = (bits & 0x8000) == 0;
    if is_sign_positive {
        bf16::from_bits(bits + 1)
    } else {
        bf16::from_bits(bits - 1)
    }
}

#[inline]
fn next_down_bf16(x: bf16) -> bf16 {
    if x.is_nan() || x == bf16::NEG_INFINITY {
        return x;
    }
    if x == bf16::from_f32(0.0) {
        return bf16::from_bits(0x8001);
    }

    let bits = x.to_bits();
    let is_sign_positive = (bits & 0x8000) == 0;
    if is_sign_positive {
        bf16::from_bits(bits - 1)
    } else {
        bf16::from_bits(bits + 1)
    }
}

impl BoundedScalar for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
}

impl BoundedScalar for f16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }

    #[inline]
    fn from_f32_down(v: f32) -> Self {
        let rounded = f16::from_f32(v);
        if rounded.is_nan() || v.is_nan() {
            return rounded;
        }
        if rounded.to_f32() > v {
            next_down_f16(rounded)
        } else {
            rounded
        }
    }

    #[inline]
    fn from_f32_up(v: f32) -> Self {
        let rounded = f16::from_f32(v);
        if rounded.is_nan() || v.is_nan() {
            return rounded;
        }
        if rounded.to_f32() < v {
            next_up_f16(rounded)
        } else {
            rounded
        }
    }
}

impl BoundedScalar for bf16 {
    #[inline]
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }

    #[inline]
    fn from_f32_down(v: f32) -> Self {
        let rounded = bf16::from_f32(v);
        if rounded.is_nan() || v.is_nan() {
            return rounded;
        }
        if rounded.to_f32() > v {
            next_down_bf16(rounded)
        } else {
            rounded
        }
    }

    #[inline]
    fn from_f32_up(v: f32) -> Self {
        let rounded = bf16::from_f32(v);
        if rounded.is_nan() || v.is_nan() {
            return rounded;
        }
        if rounded.to_f32() < v {
            next_up_bf16(rounded)
        } else {
            rounded
        }
    }
}

impl BoundedScalar for f64 {
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }

    #[inline]
    fn to_f32_down(self) -> f32 {
        let rounded = self as f32;
        if rounded.is_nan() || self.is_nan() {
            return rounded;
        }
        if rounded == f32::INFINITY && self.is_finite() {
            return f32::MAX;
        }
        if (rounded as f64) > self {
            next_down_f32(rounded)
        } else {
            rounded
        }
    }

    #[inline]
    fn to_f32_up(self) -> f32 {
        let rounded = self as f32;
        if rounded.is_nan() || self.is_nan() {
            return rounded;
        }
        if rounded == f32::NEG_INFINITY && self.is_finite() {
            return f32::MIN;
        }
        if (rounded as f64) < self {
            next_up_f32(rounded)
        } else {
            rounded
        }
    }

    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }
}

/// Generic interval bounds wrapper.
///
/// Internally, bounds are stored as `f32` in a [`BoundedTensor`]. This type
/// provides a generic interface for consuming code without exposing internal
/// implementation details.
#[derive(Debug, Clone)]
pub struct GenericBounds<T: BoundedScalar> {
    inner: BoundedTensor,
    _marker: PhantomData<T>,
}

impl<T: BoundedScalar> GenericBounds<T> {
    /// Build from lower/upper bounds in the consumer scalar type.
    pub fn new(lower: ArrayD<T>, upper: ArrayD<T>) -> Result<Self> {
        let lower_f32 = lower.mapv(T::to_f32_down);
        let upper_f32 = upper.mapv(T::to_f32_up);
        Ok(Self {
            inner: BoundedTensor::new(lower_f32, upper_f32)?,
            _marker: PhantomData,
        })
    }

    /// Wrap an existing `f32` bounded tensor.
    #[inline]
    pub fn from_f32(inner: BoundedTensor) -> Self {
        Self {
            inner,
            _marker: PhantomData,
        }
    }

    /// Borrow the internal `f32` bounds (for use by propagation code).
    #[inline]
    pub fn as_f32(&self) -> &BoundedTensor {
        &self.inner
    }

    /// Consume this wrapper and return the internal `f32` bounds.
    #[inline]
    pub fn into_f32(self) -> BoundedTensor {
        self.inner
    }

    /// Materialize lower bounds in the consumer scalar type.
    #[inline]
    pub fn lower(&self) -> ArrayD<T> {
        self.inner.lower().mapv(T::from_f32_down)
    }

    /// Materialize upper bounds in the consumer scalar type.
    #[inline]
    pub fn upper(&self) -> ArrayD<T> {
        self.inner.upper().mapv(T::from_f32_up)
    }

    /// Forwarded tensor shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }
}

impl<T: BoundedScalar> From<BoundedTensor> for GenericBounds<T> {
    #[inline]
    fn from(value: BoundedTensor) -> Self {
        Self::from_f32(value)
    }
}

impl<T: BoundedScalar> From<GenericBounds<T>> for BoundedTensor {
    #[inline]
    fn from(value: GenericBounds<T>) -> Self {
        value.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_bounds_sound<T: BoundedScalar + std::fmt::Debug>(v: f32) {
        if v.is_nan() {
            return;
        }

        let lo = <T as BoundedScalar>::from_f32_down(v);
        let hi = <T as BoundedScalar>::from_f32_up(v);

        let lo_f32 = lo.to_f32();
        let hi_f32 = hi.to_f32();

        assert!(lo_f32 <= v, "lo_f32={lo_f32:?} > v={v:?} for {lo:?}");
        assert!(hi_f32 >= v, "hi_f32={hi_f32:?} < v={v:?} for {hi:?}");
        assert!(lo_f32 <= hi_f32, "lo_f32={lo_f32:?} > hi_f32={hi_f32:?}");
    }

    #[test]
    fn test_f16_from_f32_directed_rounding_midpoints() {
        for bits in (0_u32..=0x7BFF_u32).step_by(257) {
            let h = f16::from_bits(bits as u16);
            if h.is_nan() || h.is_infinite() {
                continue;
            }
            let next = next_up_f16(h);
            if next.is_nan() || next.is_infinite() {
                continue;
            }
            let mid = f32::midpoint(h.to_f32(), next.to_f32());
            assert_bounds_sound::<f16>(mid);
        }
    }

    #[test]
    fn test_bf16_from_f32_directed_rounding_midpoints() {
        for bits in (0_u32..=0x7F7F_u32).step_by(257) {
            let h = bf16::from_bits(bits as u16);
            if h.is_nan() || h.is_infinite() {
                continue;
            }
            let next = next_up_bf16(h);
            if next.is_nan() || next.is_infinite() {
                continue;
            }
            let mid = f32::midpoint(h.to_f32(), next.to_f32());
            assert_bounds_sound::<bf16>(mid);
        }
    }

    #[test]
    fn test_f64_to_f32_directed_rounding_midpoints() {
        let a = f32::from_bits(0x3F80_0000); // 1.0
        let b = next_up_f32(a);
        let mid = f64::midpoint(a as f64, b as f64);

        let lo = <f64 as BoundedScalar>::to_f32_down(mid);
        let hi = <f64 as BoundedScalar>::to_f32_up(mid);

        assert!((lo as f64) <= mid, "lo={lo:?} > mid={mid:?}");
        assert!((hi as f64) >= mid, "hi={hi:?} < mid={mid:?}");
        assert!(lo <= hi, "lo={lo:?} > hi={hi:?}");
    }
}
