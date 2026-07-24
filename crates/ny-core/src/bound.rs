// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use std::ops::RangeInclusive;

use crate::NyError;

/// A bound on a scalar value: [lower, upper].
///
/// # Invariants (enforced by constructors and deserialization)
/// - `lower` and `upper` are not NaN
/// - `lower <= upper`
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Bound {
    /// Lower endpoint of the interval.
    pub(crate) lower: f32,
    /// Upper endpoint of the interval.
    pub(crate) upper: f32,
}

/// Custom deserialization for `Bound` that validates invariants (#2367).
///
/// Serde's derived `Deserialize` bypasses the constructor, allowing NaN and
/// inverted bounds to silently enter the system. This custom impl rejects
/// invalid data at the deserialization boundary.
impl<'de> Deserialize<'de> for Bound {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct BoundRaw {
            lower: f32,
            upper: f32,
        }
        let raw = BoundRaw::deserialize(deserializer)?;
        if raw.lower.is_nan() || raw.upper.is_nan() {
            return Err(serde::de::Error::custom(format!(
                "Bound: NaN values not allowed, got [{}, {}]",
                raw.lower, raw.upper
            )));
        }
        if raw.lower > raw.upper {
            return Err(serde::de::Error::custom(format!(
                "Bound: inverted interval [{}, {}]",
                raw.lower, raw.upper
            )));
        }
        Ok(Bound {
            lower: raw.lower,
            upper: raw.upper,
        })
    }
}

impl Bound {
    /// Create a new bound.
    ///
    /// # REQUIRES
    /// - `lower <= upper` (bounds must be well-formed)
    /// - `lower` and `upper` should be finite for reliable propagation
    ///
    /// # ENSURES
    /// - Returns `Bound` where `result.lower == lower` and `result.upper == upper`
    #[inline]
    #[must_use]
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(
            lower.is_finite(),
            "Bound::new: lower bound is not finite: {lower}"
        );
        assert!(
            upper.is_finite(),
            "Bound::new: upper bound is not finite: {upper}"
        );
        assert!(
            lower <= upper,
            "Bound::new: invalid bound {lower} > {upper}"
        );
        Self { lower, upper }
    }

    /// Create a new bound that may include infinite endpoints.
    ///
    /// # REQUIRES
    /// - `lower` and `upper` must not be NaN
    /// - `lower <= upper` (bounds must be well-formed)
    ///
    /// # ENSURES
    /// - Returns `Bound` where `result.lower == lower` and `result.upper == upper`
    #[inline]
    #[must_use]
    pub fn new_allow_infinite(lower: f32, upper: f32) -> Self {
        assert!(
            !lower.is_nan(),
            "Bound::new_allow_infinite: lower bound is NaN: {lower}"
        );
        assert!(
            !upper.is_nan(),
            "Bound::new_allow_infinite: upper bound is NaN: {upper}"
        );
        assert!(
            lower <= upper,
            "Bound::new_allow_infinite: invalid bound {lower} > {upper}"
        );
        Self { lower, upper }
    }

    /// Create a concrete (point) bound.
    ///
    /// # ENSURES
    /// - Returns `Bound` where `result.lower == result.upper == value`
    /// - `result.width() == 0`
    #[inline]
    #[must_use]
    pub fn concrete(value: f32) -> Self {
        assert!(
            value.is_finite(),
            "Bound::concrete: value must be finite, got {value}"
        );
        Self {
            lower: value,
            upper: value,
        }
    }

    /// Fallible constructor: returns `Err` on non-finite or inverted bounds.
    ///
    /// Use this instead of `new` when the inputs may be invalid (e.g., from
    /// numerical computation that could produce NaN/Inf).
    ///
    /// # ENSURES
    /// - On `Ok`: `result.lower == lower`, `result.upper == upper`, both finite
    /// - On `Err`: `NumericalInstability` with diagnostic message
    #[inline]
    pub fn try_new(lower: f32, upper: f32) -> crate::Result<Self> {
        if !lower.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "Bound::try_new: lower is not finite: {lower}"
            )));
        }
        if !upper.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "Bound::try_new: upper is not finite: {upper}"
            )));
        }
        if lower > upper {
            return Err(NyError::NumericalInstability(format!(
                "Bound::try_new: inverted interval [{lower}, {upper}]"
            )));
        }
        Ok(Self { lower, upper })
    }

    /// Fallible constructor allowing infinite endpoints.
    ///
    /// Returns `Err` on NaN or inverted bounds, but allows `±Inf`.
    ///
    /// # ENSURES
    /// - On `Ok`: `result.lower == lower`, `result.upper == upper`, neither NaN
    /// - On `Err`: `NumericalInstability` with diagnostic message
    #[inline]
    pub fn try_new_allow_infinite(lower: f32, upper: f32) -> crate::Result<Self> {
        if lower.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "Bound::try_new_allow_infinite: lower is NaN: {lower}"
            )));
        }
        if upper.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "Bound::try_new_allow_infinite: upper is NaN: {upper}"
            )));
        }
        if lower > upper {
            return Err(NyError::NumericalInstability(format!(
                "Bound::try_new_allow_infinite: inverted interval [{lower}, {upper}]"
            )));
        }
        Ok(Self { lower, upper })
    }

    /// Fallible concrete (point) bound constructor.
    ///
    /// Returns `Err` if value is not finite.
    ///
    /// # ENSURES
    /// - On `Ok`: `result.lower == result.upper == value`, value is finite
    /// - On `Err`: `NumericalInstability` with diagnostic message
    #[inline]
    pub fn try_concrete(value: f32) -> crate::Result<Self> {
        if !value.is_finite() {
            return Err(NyError::NumericalInstability(format!(
                "Bound::try_concrete: value is not finite: {value}"
            )));
        }
        Ok(Self {
            lower: value,
            upper: value,
        })
    }

    /// Read-only access to the lower bound.
    #[inline]
    pub fn lower(&self) -> f32 {
        self.lower
    }

    /// Read-only access to the upper bound.
    #[inline]
    pub fn upper(&self) -> f32 {
        self.upper
    }

    /// Check if this bound contains a value.
    ///
    /// # ENSURES
    /// - Returns `true` iff `self.lower <= value <= self.upper`
    #[inline]
    #[must_use]
    pub fn contains(&self, value: f32) -> bool {
        self.lower <= value && value <= self.upper
    }

    /// Width of the bound interval.
    ///
    /// # ENSURES
    /// - Returns `self.upper - self.lower`
    /// - Result is non-negative if bound is well-formed (`self.lower <= self.upper`)
    #[inline]
    #[must_use]
    pub fn width(&self) -> f32 {
        self.upper - self.lower
    }

    /// Check if bounds are tight (width below threshold).
    ///
    /// # REQUIRES
    /// - `epsilon >= 0` for meaningful comparison
    ///
    /// # ENSURES
    /// - Returns `true` iff `self.width() <= epsilon`
    #[inline]
    #[must_use]
    pub fn is_tight(&self, epsilon: f32) -> bool {
        self.width() <= epsilon
    }

    /// Check if bounds have exploded to infinity.
    ///
    /// # ENSURES
    /// - Returns `true` iff `self.lower` or `self.upper` is infinite
    #[inline]
    #[must_use]
    pub fn is_unbounded(&self) -> bool {
        self.lower.is_infinite() || self.upper.is_infinite()
    }

    /// Intersect two bounds.
    ///
    /// Returns `None` if either bound has a NaN endpoint (soundness: NaN
    /// indicates upstream numerical corruption and must not be silently
    /// absorbed into a tighter interval). Also returns `None` for disjoint
    /// intervals (`lower > upper` after intersection).
    ///
    /// # Why NaN → None
    ///
    /// IEEE 754: `NaN.max(x) == x` and `NaN.min(x) == x`. Without this
    /// guard, a partially-NaN interval like `[NaN, 5.0]` intersected with
    /// `[1.0, 3.0]` would silently produce `[1.0, 3.0]` — absorbing the
    /// NaN and labeling the result as CROWN-tightened. Callers (CROWN-IBP
    /// merge in ibp.rs, crown.rs, alpha.rs) treat `None` as a trigger to
    /// fall back to conservative IBP bounds. Reference: alpha-beta-CROWN
    /// asserts `not dm_lb.isnan().any()` before bound intersection.
    ///
    /// # ENSURES
    /// - Returns `None` if any endpoint in `self` or `other` is NaN
    /// - Returns `Some(result)` iff the intersection is non-empty and NaN-free
    /// - If `Some(result)`, then `result.lower >= self.lower` and `result.lower >= other.lower`
    /// - If `Some(result)`, then `result.upper <= self.upper` and `result.upper <= other.upper`
    /// - For any value in result, `self.contains(value) && other.contains(value)`
    #[inline]
    #[must_use]
    pub fn intersect(&self, other: &Bound) -> Option<Bound> {
        // NaN in any endpoint → None (soundness: #2640)
        if self.lower.is_nan()
            || self.upper.is_nan()
            || other.lower.is_nan()
            || other.upper.is_nan()
        {
            return None;
        }
        let lower = self.lower.max(other.lower);
        let upper = self.upper.min(other.upper);
        if lower <= upper {
            Some(Bound { lower, upper })
        } else {
            None
        }
    }

    /// Union of two bounds (convex hull).
    ///
    /// # ENSURES
    /// - Result contains all values in `self`: `result.lower <= self.lower`, `result.upper >= self.upper`
    /// - Result contains all values in `other`: `result.lower <= other.lower`, `result.upper >= other.upper`
    /// - For any value where `self.contains(value)`, `result.contains(value)` (sound overapproximation)
    #[inline]
    #[must_use]
    pub fn union(&self, other: &Bound) -> Bound {
        assert!(
            !self.lower.is_nan() && !self.upper.is_nan(),
            "Bound::union: self bound contains NaN endpoint [{}, {}]",
            self.lower,
            self.upper
        );
        assert!(
            !other.lower.is_nan() && !other.upper.is_nan(),
            "Bound::union: other bound contains NaN endpoint [{}, {}]",
            other.lower,
            other.upper
        );
        Bound::new_allow_infinite(self.lower.min(other.lower), self.upper.max(other.upper))
    }
}

impl From<RangeInclusive<f32>> for Bound {
    fn from(range: RangeInclusive<f32>) -> Self {
        Self::new(*range.start(), *range.end())
    }
}
