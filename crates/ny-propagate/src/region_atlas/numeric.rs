// Copyright 2026 Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Outward binary64 intervals for exact-real stored coefficients. Each
//! nontrivial RN operation is widened in both directions, including underflow.
//! Overflow/NaN refuses replay. No hardware rounding mode or SIMD reduction.

use super::AtlasRefusal;
use crate::margin_row::rounding::{next_down, next_up};

#[derive(Debug, Clone, Copy)]
pub(super) struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub const ZERO: Self = Self { lo: 0.0, hi: 0.0 };

    pub fn point(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    fn checked(lo: f64, hi: f64) -> Result<Self, AtlasRefusal> {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(AtlasRefusal::NonFinite);
        }
        Ok(Self { lo, hi })
    }

    fn is_point(self, value: f64) -> bool {
        self.lo == value && self.hi == value
    }

    pub fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }

    pub fn add(self, rhs: Self) -> Result<Self, AtlasRefusal> {
        if self.is_point(0.0) {
            return Ok(rhs);
        }
        if rhs.is_point(0.0) {
            return Ok(self);
        }
        // Exact representable cancellation does not need a rounding charge.
        if self.lo == self.hi && rhs.lo == rhs.hi && self.lo == -rhs.lo {
            return Ok(Self::ZERO);
        }
        Self::checked(next_down(self.lo + rhs.lo), next_up(self.hi + rhs.hi))
    }

    pub fn sub(self, rhs: Self) -> Result<Self, AtlasRefusal> {
        self.add(rhs.neg())
    }

    pub fn mul(self, rhs: Self) -> Result<Self, AtlasRefusal> {
        if self.is_point(0.0) || rhs.is_point(0.0) {
            return Ok(Self::ZERO);
        }
        if self.is_point(1.0) {
            return Ok(rhs);
        }
        if rhs.is_point(1.0) {
            return Ok(self);
        }
        if self.is_point(-1.0) {
            return Ok(rhs.neg());
        }
        if rhs.is_point(-1.0) {
            return Ok(self.neg());
        }
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        if products.iter().any(|value| !value.is_finite()) {
            return Err(AtlasRefusal::NonFinite);
        }
        let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::checked(next_down(lo), next_up(hi))
    }
}
