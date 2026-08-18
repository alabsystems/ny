// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Numeric operations on BoundedTensor: width, rounding, scalar access, intersection, repair.

use ndarray::{ArrayD, IxDyn};
use ny_core::{nan_propagating_max, Bound, NyError, Result};

use crate::{next_down_f32, next_up_f32, shift_down_n_ulps, shift_up_n_ulps};

use super::BoundedTensor;

impl BoundedTensor {
    #[inline]
    fn scalar_endpoints(&self, index: &[usize]) -> Result<(f32, f32)> {
        if index.len() != self.ndim() {
            return Err(NyError::InvalidSpec(format!(
                "BoundedTensor::try_get: index rank {} does not match tensor ndim {}",
                index.len(),
                self.ndim()
            )));
        }

        for (axis, (&axis_index, &axis_len)) in index.iter().zip(self.shape()).enumerate() {
            if axis_index >= axis_len {
                return Err(NyError::InvalidSpec(format!(
                    "BoundedTensor::try_get: index {axis_index} out of bounds for axis {axis} with len {axis_len}"
                )));
            }
        }

        Ok((self.lower[index], self.upper[index]))
    }

    /// Check if this tensor contains any NaN or Inf values.
    #[inline]
    pub fn has_overflow(&self) -> bool {
        Self::has_nan_or_inf(self.lower.as_array()) || Self::has_nan_or_inf(self.upper.as_array())
    }

    /// Apply directed rounding for mathematically sound interval arithmetic.
    ///
    /// IEEE 754 floating-point operations round to nearest-even by default.
    /// For strict interval arithmetic soundness, lower bounds should round DOWN
    /// (toward -∞) and upper bounds should round UP (toward +∞).
    ///
    /// This method widens bounds by taking the next representable float toward
    /// -∞ for lower bounds and toward +∞ for upper bounds (at most 1 ULP per element).
    ///
    /// # Use Cases
    /// - Apply after critical propagation steps to ensure containment
    /// - Use for final verification bounds when strict soundness is required
    ///
    /// # Performance
    /// Adds ~1 ULP of looseness per application. For typical verification
    /// with 100+ layers, this accumulates to ~100 ULPs, which is negligible
    /// compared to relaxation approximation errors (typically 10^3 - 10^6 ULPs).
    #[inline]
    pub fn round_for_soundness(&self) -> Self {
        // Drop the L2 annotation on this widened copy (sound: only loses
        // tightening). The in-place variant keeps it, since widening the box can
        // only continue to contain the same sphere.
        Self::from_parts_with_l2(
            self.lower.mapv(next_down_f32),
            self.upper.mapv(next_up_f32),
            None,
        )
    }

    /// Apply directed rounding in place.
    ///
    /// Modifies this tensor to widen bounds by 1 ULP for soundness.
    /// See [`Self::round_for_soundness`] for details.
    #[inline]
    pub fn round_for_soundness_inplace(&mut self) {
        self.lower.mapv_inplace(next_down_f32);
        self.upper.mapv_inplace(next_up_f32);
    }

    /// Apply directed 1-ULP widening in place while cooperatively polling.
    ///
    /// This has the same element-wise result as
    /// [`Self::round_for_soundness_inplace`]. If `poll` returns an error, the
    /// tensor may be partially widened and must not be published by the caller.
    pub fn round_for_soundness_inplace_with_poll<F>(&mut self, mut poll: F) -> Result<()>
    where
        F: FnMut() -> Result<()>,
    {
        const POLL_ELEMENTS: usize = 4_096;

        poll()?;
        for (index, value) in self.lower.iter_mut().enumerate() {
            if index.is_multiple_of(POLL_ELEMENTS) {
                poll()?;
            }
            *value = next_down_f32(*value);
        }
        for (index, value) in self.upper.iter_mut().enumerate() {
            if index.is_multiple_of(POLL_ELEMENTS) {
                poll()?;
            }
            *value = next_up_f32(*value);
        }
        poll()
    }

    /// Widen bounds by `n` ULPs in each direction for soundness.
    ///
    /// For a dot product of `n` terms, IEEE 754 rounding can accumulate up to
    /// `n` ULPs of error. This method widens lower bounds by `n` ULPs toward
    /// -∞ and upper bounds by `n` ULPs toward +∞.
    ///
    /// Use after operations with known accumulation depth (e.g., linear layer
    /// with `in_features` multiply-accumulate operations plus bias addition).
    ///
    /// Reference: Higham, "Accuracy and Stability of Numerical Algorithms",
    /// Theorem 3.1: the error of a sum of n terms is bounded by (n-1) * eps * |sum|,
    /// which is at most (n-1) ULPs. We use `n` ULPs as a conservative bound
    /// that also covers the final addition (bias, etc.).
    #[inline]
    pub fn round_for_soundness_n_ulps(&self, n: u32) -> Self {
        // Drop the L2 annotation on this widened copy (sound: only loses
        // tightening). See `round_for_soundness`.
        Self::from_parts_with_l2(
            self.lower.mapv(|v| shift_down_n_ulps(v, n)),
            self.upper.mapv(|v| shift_up_n_ulps(v, n)),
            None,
        )
    }

    /// Widen bounds by `n` ULPs in each direction, in place.
    ///
    /// See [`Self::round_for_soundness_n_ulps`] for details.
    #[inline]
    pub fn round_for_soundness_n_ulps_inplace(&mut self, n: u32) {
        self.lower.mapv_inplace(|v| shift_down_n_ulps(v, n));
        self.upper.mapv_inplace(|v| shift_up_n_ulps(v, n));
    }

    /// Apply directed `n`-ULP widening in place while cooperatively polling.
    ///
    /// This has the same element-wise result as
    /// [`Self::round_for_soundness_n_ulps_inplace`]. If `poll` returns an
    /// error, the tensor may be partially widened and must not be published by
    /// the caller.
    pub fn round_for_soundness_n_ulps_inplace_with_poll<F>(
        &mut self,
        n: u32,
        mut poll: F,
    ) -> Result<()>
    where
        F: FnMut() -> Result<()>,
    {
        const POLL_ELEMENTS: usize = 4_096;

        poll()?;
        for (index, value) in self.lower.iter_mut().enumerate() {
            if index.is_multiple_of(POLL_ELEMENTS) {
                poll()?;
            }
            *value = shift_down_n_ulps(*value, n);
        }
        for (index, value) in self.upper.iter_mut().enumerate() {
            if index.is_multiple_of(POLL_ELEMENTS) {
                poll()?;
            }
            *value = shift_up_n_ulps(*value, n);
        }
        poll()
    }

    /// Get bounds for a specific feasible element.
    ///
    /// This legacy convenience accessor panics for malformed indices or
    /// infeasible sentinels. Callers that may observe infeasible elements or
    /// prefer non-panicking index handling should use [`Self::try_get`].
    #[inline]
    pub fn get(&self, index: &[usize]) -> Bound {
        self.try_get(index)
            .expect("BoundedTensor::get: invalid index; use try_get() for non-panicking access")
            .expect("BoundedTensor::get: element is infeasible; use try_get()")
    }

    /// Get bounds for a specific element without panicking.
    ///
    /// Returns `Ok(Some(bound))` for feasible elements, including infinite
    /// endpoints from conservative/public constructors. Returns `Ok(None)` for
    /// infeasible sentinel elements (`+inf, -inf`). Invalid indices return
    /// `NyError::InvalidSpec`.
    #[inline]
    pub fn try_get(&self, index: &[usize]) -> Result<Option<Bound>> {
        let (lower, upper) = self.scalar_endpoints(index)?;

        if lower.is_nan() || upper.is_nan() {
            return Err(NyError::NumericalInstability(format!(
                "BoundedTensor::try_get: element at {:?} contains NaN endpoints [{lower}, {upper}]",
                index
            )));
        }

        if lower <= upper {
            return Ok(Some(Bound::new_allow_infinite(lower, upper)));
        }

        Ok(None)
    }

    /// Compute the width (upper - lower) for each element.
    #[inline]
    pub fn width(&self) -> ArrayD<f32> {
        self.upper.as_array() - self.lower.as_array()
    }

    /// Maximum width across all elements.
    pub fn max_width(&self) -> f32 {
        self.width()
            .iter()
            .cloned()
            .fold(0.0_f32, nan_propagating_max)
    }

    /// Check if any bounds have exploded.
    pub fn has_unbounded(&self) -> bool {
        self.lower.iter().any(|v| v.is_infinite()) || self.upper.iter().any(|v| v.is_infinite())
    }

    /// Compute the center point of the bounded tensor: (lower + upper) / 2.
    ///
    /// Handles symmetric infinite bounds `[-inf, +inf]` by returning 0.0 instead
    /// of NaN. Such bounds arise from `new_conservative()` and `new_repaired(Widen)`.
    /// Reference: `new_repaired(Widen)` sets `[-inf, +inf]` for corrupt elements. (#3291 F3, #3423)
    // `f32::midpoint` would silently return a finite center for exploded bounds where
    // both |l|,|u| > f32::MAX/2; `(l + u) / 2.0` overflows to ±inf there, and downstream
    // consumers must keep seeing that ±inf rather than a fabricated finite center.
    #[allow(clippy::manual_midpoint)]
    pub fn center(&self) -> ArrayD<f32> {
        ndarray::Zip::from(self.lower.as_array())
            .and(self.upper.as_array())
            .map_collect(|&l, &u| {
                if l.is_infinite() && u.is_infinite() && l.signum() != u.signum() {
                    0.0
                } else {
                    (l + u) / 2.0
                }
            })
    }

    /// Compute [`Self::center`] while cooperatively polling bounded chunks.
    ///
    /// Allocator calls themselves are not preemptible, but every element scan
    /// is split by `poll` calls. The returned array has the same shape and
    /// element values as [`Self::center`].
    // Preserve `Self::center` bit-for-bit, including its intentional overflow
    // behavior for exploded finite bounds; `f32::midpoint` would change it.
    #[allow(clippy::manual_midpoint)]
    pub fn center_with_poll<F>(&self, mut poll: F) -> Result<ArrayD<f32>>
    where
        F: FnMut() -> Result<()>,
    {
        const POLL_ELEMENTS: usize = 4_096;

        poll()?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.lower.len())
            .map_err(|error| {
                NyError::InvalidSpec(format!(
                    "BoundedTensor::center_with_poll: allocation failed for {} elements: {error}",
                    self.lower.len()
                ))
            })?;
        poll()?;
        for (index, (&lower, &upper)) in self.lower.iter().zip(self.upper.iter()).enumerate() {
            if index.is_multiple_of(POLL_ELEMENTS) {
                poll()?;
            }
            values.push(
                if lower.is_infinite() && upper.is_infinite() && lower.signum() != upper.signum() {
                    0.0
                } else {
                    (lower + upper) / 2.0
                },
            );
        }
        poll()?;
        let center = ArrayD::from_shape_vec(IxDyn(self.shape()), values).map_err(|error| {
            NyError::InternalError(format!(
                "BoundedTensor::center_with_poll: output reshape failed: {error}"
            ))
        })?;
        poll()?;
        Ok(center)
    }

    /// Compute the intersection of two bounded tensors.
    ///
    /// Per-element intersection with union fallback for disjoint elements.
    ///
    /// For each element position:
    /// - If `max(lower_a, lower_b) <= min(upper_a, upper_b)`: intersection (tighter bounds)
    /// - Otherwise: union (conservative fallback, preserves soundness)
    ///
    /// Returns `None` only for shape mismatch or NaN in any endpoint.
    /// Returns `(result, disjoint_count)` where `disjoint_count` is the number of
    /// elements that fell back to union.
    ///
    /// Matches alpha-beta-CROWN reference behavior (`bound_general.py:1079-1084, 1452-1453`)
    /// and our own `tighten_crown_with_forward_bounds` in `crown_utils.rs`.
    ///
    /// # NaN behavior
    ///
    /// Same as [`intersection`]: returns `None` if any endpoint in either tensor
    /// is NaN. NaN indicates upstream numerical corruption and must not be silently
    /// absorbed. (#2640)
    pub fn intersection_per_element(&self, other: &Self) -> Option<(Self, usize)> {
        if self.shape() != other.shape() {
            return None;
        }

        // NaN in any endpoint → None (soundness: #2640, same as intersection())
        let has_nan = self.lower.iter().any(|v| v.is_nan())
            || self.upper.iter().any(|v| v.is_nan())
            || other.lower.iter().any(|v| v.is_nan())
            || other.upper.iter().any(|v| v.is_nan());
        if has_nan {
            return None;
        }

        let mut lower = self.lower.as_array().clone();
        let mut upper = self.upper.as_array().clone();
        let mut disjoint_count = 0usize;

        for ((sl, su), (ol, ou)) in lower
            .iter_mut()
            .zip(upper.iter_mut())
            .zip(other.lower.iter().zip(other.upper.iter()))
        {
            let tl = sl.max(*ol);
            let tu = su.min(*ou);
            if tl <= tu {
                *sl = tl;
                *su = tu;
            } else {
                // Union fallback — preserves soundness by widening to contain both
                // intervals. Matches tighten_crown_with_forward_bounds behavior.
                disjoint_count += 1;
                *sl = sl.min(*ol);
                *su = su.max(*ou);
            }
        }

        // Drop any L2 annotation: the per-element intersection produces a fresh
        // interval whose sphere is not tracked. Sound (only loses tightening).
        Some((Self::from_parts_with_l2(lower, upper, None), disjoint_count))
    }

    /// Pollable counterpart to [`Self::intersection_per_element`].
    ///
    /// The no-error result is element-for-element identical to the ordinary
    /// method. If `poll` returns an error, no partially constructed
    /// intersection is returned.
    pub fn intersection_per_element_with_poll<F>(
        &self,
        other: &Self,
        mut poll: F,
    ) -> Result<Option<(Self, usize)>>
    where
        F: FnMut() -> Result<()>,
    {
        const POLL_ELEMENTS: usize = 4_096;

        poll()?;
        if self.shape() != other.shape() {
            return Ok(None);
        }

        // Preserve the ordinary method's all-endpoint NaN rejection before
        // constructing any result.
        for endpoints in [
            self.lower.as_array(),
            self.upper.as_array(),
            other.lower.as_array(),
            other.upper.as_array(),
        ] {
            for (index, &value) in endpoints.iter().enumerate() {
                if index.is_multiple_of(POLL_ELEMENTS) {
                    poll()?;
                }
                if value.is_nan() {
                    return Ok(None);
                }
            }
        }

        let elements = self.lower.len();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        lower.try_reserve_exact(elements).map_err(|error| {
            NyError::InvalidSpec(format!(
                "BoundedTensor::intersection_per_element_with_poll: lower allocation failed for \
                 {elements} elements: {error}"
            ))
        })?;
        poll()?;
        upper.try_reserve_exact(elements).map_err(|error| {
            NyError::InvalidSpec(format!(
                "BoundedTensor::intersection_per_element_with_poll: upper allocation failed for \
                 {elements} elements: {error}"
            ))
        })?;
        poll()?;

        let mut disjoint_count = 0usize;
        for (index, (((&self_lower, &self_upper), &other_lower), &other_upper)) in self
            .lower
            .iter()
            .zip(self.upper.iter())
            .zip(other.lower.iter())
            .zip(other.upper.iter())
            .enumerate()
        {
            if index.is_multiple_of(POLL_ELEMENTS) {
                poll()?;
            }
            let tightened_lower = self_lower.max(other_lower);
            let tightened_upper = self_upper.min(other_upper);
            if tightened_lower <= tightened_upper {
                lower.push(tightened_lower);
                upper.push(tightened_upper);
            } else {
                disjoint_count += 1;
                lower.push(self_lower.min(other_lower));
                upper.push(self_upper.max(other_upper));
            }
        }
        poll()?;

        let lower = ArrayD::from_shape_vec(IxDyn(self.shape()), lower).map_err(|error| {
            NyError::InternalError(format!(
                "BoundedTensor::intersection_per_element_with_poll: lower reshape failed: {error}"
            ))
        })?;
        let upper = ArrayD::from_shape_vec(IxDyn(self.shape()), upper).map_err(|error| {
            NyError::InternalError(format!(
                "BoundedTensor::intersection_per_element_with_poll: upper reshape failed: {error}"
            ))
        })?;
        poll()?;
        Ok(Some((
            Self::from_parts_with_l2(lower, upper, None),
            disjoint_count,
        )))
    }
}
