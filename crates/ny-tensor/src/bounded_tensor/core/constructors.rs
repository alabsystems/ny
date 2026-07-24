// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! BoundedTensor constructors, setters, and sanitization methods.

use ndarray::{ArrayD, Axis, IxDyn};
use ny_core::{NyError, Result};
use tracing::debug;

use super::inversion_repair::{repair_inverted_bounds_nd, InversionRepair};
use super::BoundedTensor;

/// Strategy for repairing non-finite (NaN/Inf) values in bound arrays.
///
/// Used by [`BoundedTensor::new_repaired`] to centralize NaN/Inf handling
/// at the type boundary instead of at 164 ad-hoc repair sites. Part of #3423.
///
/// Unlike [`BoundedTensor::new_sanitized`] which clamps all values (including
/// finite ones), `new_repaired` only repairs NaN elements; ±Inf endpoints and
/// finite values are left unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairStrategy {
    /// Repair NaN to the conservative direction (`-inf` for lower, `+inf` for
    /// upper) and preserve ±Inf endpoints. Default for IBP/CROWN output: a
    /// non-finite endpoint proves nothing, so replacing it with any finite
    /// value would tighten a bound that was never established.
    Conservative,
    /// Replace NaN with ±inf (conservative direction). Leave ±Inf as-is.
    /// For intermediate bounds that will be refined later. Same repair as
    /// [`Self::Conservative`].
    Widen,
    /// Hard error on NaN/Inf. Same behavior as [`BoundedTensor::new`].
    /// For final verification results where non-finite indicates a bug.
    Strict,
}

/// Sanitize a single scalar bound: NaN → `nan_default`, ±Inf → ±`clamp_val`, else clamp.
#[inline]
fn sanitize_scalar(x: f32, nan_default: f32, clamp_val: f32) -> f32 {
    if x.is_nan() {
        nan_default
    } else if x.is_infinite() {
        if x > 0.0 {
            clamp_val
        } else {
            -clamp_val
        }
    } else {
        x.clamp(-clamp_val, clamp_val)
    }
}

impl BoundedTensor {
    /// Replace lower bounds after validating shape, finiteness, and ordering.
    #[inline]
    pub fn set_lower(&mut self, lower: ArrayD<f32>) -> Result<()> {
        if lower.shape() != self.upper.shape() {
            return Err(NyError::shape_mismatch(
                self.upper.shape().to_vec(),
                lower.shape().to_vec(),
            ));
        }
        if Self::has_nan_or_inf(&lower) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor::set_lower: lower bounds contain NaN or Inf".to_string(),
            ));
        }
        let bounds_valid = ndarray::Zip::from(&lower)
            .and(&self.upper)
            .all(|&l, &u| l <= u);
        if !bounds_valid {
            return Err(NyError::InvalidSpec(
                "BoundedTensor::set_lower: found lower > upper (inverted bounds)".to_string(),
            ));
        }
        self.lower = lower;
        Ok(())
    }

    /// Replace upper bounds after validating shape, finiteness, and ordering.
    #[inline]
    pub fn set_upper(&mut self, upper: ArrayD<f32>) -> Result<()> {
        if upper.shape() != self.lower.shape() {
            return Err(NyError::shape_mismatch(
                self.lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }
        if Self::has_nan_or_inf(&upper) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor::set_upper: upper bounds contain NaN or Inf".to_string(),
            ));
        }
        let bounds_valid = ndarray::Zip::from(&self.lower)
            .and(&upper)
            .all(|&l, &u| l <= u);
        if !bounds_valid {
            return Err(NyError::InvalidSpec(
                "BoundedTensor::set_upper: found lower > upper (inverted bounds)".to_string(),
            ));
        }
        self.upper = upper;
        Ok(())
    }

    /// Mark every element infeasible using the canonical `(+inf, -inf)` sentinel.
    #[inline]
    pub fn mark_infeasible_all(&mut self) {
        self.lower.fill(f32::INFINITY);
        self.upper.fill(f32::NEG_INFINITY);
    }

    /// Mark one axis slice infeasible using the canonical `(+inf, -inf)` sentinel.
    #[inline]
    pub fn mark_infeasible_at(&mut self, axis: usize, index: usize) -> Result<()> {
        if axis >= self.lower.ndim() {
            return Err(NyError::InvalidSpec(format!(
                "BoundedTensor::mark_infeasible_at: axis {axis} out of bounds for ndim {}",
                self.lower.ndim()
            )));
        }

        let axis_len = self.lower.shape()[axis];
        if index >= axis_len {
            return Err(NyError::InvalidSpec(format!(
                "BoundedTensor::mark_infeasible_at: index {index} out of bounds for axis {axis} with len {axis_len}"
            )));
        }

        self.lower
            .index_axis_mut(Axis(axis), index)
            .fill(f32::INFINITY);
        self.upper
            .index_axis_mut(Axis(axis), index)
            .fill(f32::NEG_INFINITY);
        Ok(())
    }

    /// Create a bounded tensor from lower and upper bound arrays.
    /// Rejects NaN, Inf, shape mismatch, and inverted bounds.
    /// All checks run in all builds (not just debug) for soundness.
    #[inline]
    pub fn new(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(NyError::shape_mismatch(
                lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }

        if Self::has_nan_or_inf(&lower) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor::new: lower bounds contain NaN or Inf".to_string(),
            ));
        }
        if Self::has_nan_or_inf(&upper) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor::new: upper bounds contain NaN or Inf".to_string(),
            ));
        }
        let bounds_valid = ndarray::Zip::from(&lower).and(&upper).all(|&l, &u| l <= u);
        if !bounds_valid {
            return Err(NyError::InvalidSpec(
                "BoundedTensor::new: found lower > upper (inverted bounds)".to_string(),
            ));
        }

        Ok(Self {
            lower,
            upper,
            l2: None,
        })
    }

    /// Like [`Self::new`] but allows infinite endpoints. Rejects NaN and inverted bounds.
    /// Use for conservative fallback bounds (e.g. `[-inf, +inf]`).
    #[inline]
    pub fn new_allow_infinite(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(NyError::shape_mismatch(
                lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }
        if lower.iter().any(|&v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor::new_allow_infinite: lower bounds contain NaN".to_string(),
            ));
        }
        if upper.iter().any(|&v| v.is_nan()) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor::new_allow_infinite: upper bounds contain NaN".to_string(),
            ));
        }
        let bounds_valid = ndarray::Zip::from(&lower).and(&upper).all(|&l, &u| l <= u);
        if !bounds_valid {
            return Err(NyError::InvalidSpec(
                "BoundedTensor::new_allow_infinite: found lower > upper (inverted bounds)"
                    .to_string(),
            ));
        }
        Ok(Self {
            lower,
            upper,
            l2: None,
        })
    }

    /// Create conservative `[-inf, +inf]` bounds for every element of `shape`.
    ///
    /// This constructor is infallible: both arrays are created from the same shape and
    /// contain non-NaN endpoints only.
    #[inline]
    pub fn new_conservative(shape: &[usize]) -> Self {
        let shape = IxDyn(shape);
        let lower = ArrayD::from_elem(shape.clone(), f32::NEG_INFINITY);
        let upper = ArrayD::from_elem(shape, f32::INFINITY);
        Self {
            lower,
            upper,
            l2: None,
        }
    }

    /// Create a concrete tensor (lower == upper). Returns `Err` if `values` contains NaN or Inf.
    pub fn concrete(values: ArrayD<f32>) -> Result<Self> {
        if Self::has_nan_or_inf(&values) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor::concrete: values contain NaN or Inf".to_string(),
            ));
        }
        Ok(Self {
            lower: values.clone(),
            upper: values,
            l2: None,
        })
    }

    /// Create bounds as `[value - epsilon, value + epsilon]`. Returns `Err` if values contain
    /// NaN/Inf or epsilon is negative/non-finite. Clamps overflow to `f32::MAX`/`f32::MIN`.
    pub fn from_epsilon(values: ArrayD<f32>, epsilon: f32) -> Result<Self> {
        if Self::has_nan_or_inf(&values) {
            return Err(NyError::NumericalInstability(
                "BoundedTensor::from_epsilon: values contain NaN or Inf".to_string(),
            ));
        }
        if !(epsilon >= 0.0 && epsilon.is_finite()) {
            return Err(NyError::InvalidSpec(format!(
                "BoundedTensor::from_epsilon: epsilon must be non-negative and finite, got {}",
                epsilon
            )));
        }
        Ok(Self {
            lower: values.mapv(|v| {
                let r = v - epsilon;
                if r.is_infinite() {
                    f32::MIN
                } else {
                    r
                }
            }),
            upper: values.mapv(|v| {
                let r = v + epsilon;
                if r.is_infinite() {
                    f32::MAX
                } else {
                    r
                }
            }),
            l2: None,
        })
    }

    /// Internal shape-only constructor shared by all `new_unchecked` variants.
    #[inline]
    #[cfg(any(test, feature = "test-utils"))]
    fn new_unchecked_impl(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(NyError::shape_mismatch(
                lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }
        Ok(Self {
            lower,
            upper,
            l2: None,
        })
    }

    /// Bypass NaN/Inf/ordering checks. Only validates shape.
    ///
    /// This API is public only when `ny-tensor/test-utils` is enabled for
    /// cross-crate tests that need to construct intentionally-invalid bounds.
    #[inline]
    #[cfg(feature = "test-utils")]
    pub fn new_unchecked(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        Self::new_unchecked_impl(lower, upper)
    }

    /// Bypass NaN/Inf/ordering checks. Only validates shape.
    ///
    /// In crate-local tests this constructor remains available without enabling
    /// the public `test-utils` feature.
    #[inline]
    #[cfg(all(test, not(feature = "test-utils")))]
    pub(crate) fn new_unchecked(lower: ArrayD<f32>, upper: ArrayD<f32>) -> Result<Self> {
        Self::new_unchecked_impl(lower, upper)
    }

    /// Sanitize lower/upper arrays and enforce lower <= upper ordering.
    ///
    /// Shared implementation for [`Self::new_sanitized`] and [`Self::sanitize`].
    /// NaN → conservative direction (lower gets -clamp_val, upper gets +clamp_val).
    /// ±Inf → ±clamp_val. Finite values clamped to `[-clamp_val, +clamp_val]`.
    fn sanitize_and_order(
        lower: ArrayD<f32>,
        upper: ArrayD<f32>,
        clamp_val: f32,
    ) -> (ArrayD<f32>, ArrayD<f32>) {
        let mut lower = lower.mapv(|x| sanitize_scalar(x, -clamp_val, clamp_val));
        let mut upper = upper.mapv(|x| sanitize_scalar(x, clamp_val, clamp_val));
        let _ = repair_inverted_bounds_nd(&mut lower, &mut upper, InversionRepair::Swap);

        (lower, upper)
    }

    /// Sanitize bounds by clamping NaN/Inf/out-of-range to `[-clamp_val, +clamp_val]`,
    /// swapping inverted bounds. For `continue_after_overflow` mode.
    #[inline]
    pub fn new_sanitized(lower: ArrayD<f32>, upper: ArrayD<f32>, clamp_val: f32) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(NyError::shape_mismatch(
                lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }

        let original_shape = lower.shape().to_vec();
        let (result_lower, result_upper) = Self::sanitize_and_order(lower, upper, clamp_val);

        debug_assert_eq!(result_lower.shape(), &original_shape[..]);
        Ok(Self {
            lower: result_lower,
            upper: result_upper,
            l2: None,
        })
    }

    /// Sanitize this tensor by clamping NaN/Inf values.
    ///
    /// Returns a new tensor with the same shape where all NaN/Inf values
    /// have been replaced with clamped finite values.
    ///
    /// See [`Self::new_sanitized`] for details on the clamping behavior.
    #[inline]
    pub fn sanitize(&self, clamp_val: f32) -> Self {
        let (lower, upper) =
            Self::sanitize_and_order(self.lower.clone(), self.upper.clone(), clamp_val);

        // KEEP unchecked: sanitize_and_order() preserves shape and repairs
        // non-finite / inverted entries before reconstruction.
        Self::from_parts_unchecked(lower, upper)
    }

    /// Construct BoundedTensor with automatic NaN repair per the given strategy.
    ///
    /// This is the preferred constructor for all propagation output. Centralizes
    /// NaN/Inf handling at the type boundary. Part of #3423.
    ///
    /// Unlike [`Self::new_sanitized`], only NaN values are modified — finite
    /// values are left unchanged regardless of magnitude, and ±Inf endpoints
    /// are preserved: an interval with a non-finite endpoint carries no proven
    /// bound in that direction, so any finite replacement would be an unsound
    /// tightening.
    ///
    /// Repair steps:
    /// 1. Shape validation
    /// 2. Repair NaN per strategy
    /// 3. Fix inverted bounds (swap where lower > upper after repair)
    /// 4. Log repair count for observability (#2717)
    pub fn new_repaired(
        lower: ArrayD<f32>,
        upper: ArrayD<f32>,
        strategy: RepairStrategy,
    ) -> Result<Self> {
        if lower.shape() != upper.shape() {
            return Err(NyError::shape_mismatch(
                lower.shape().to_vec(),
                upper.shape().to_vec(),
            ));
        }

        match strategy {
            RepairStrategy::Strict => Self::new(lower, upper),
            RepairStrategy::Conservative | RepairStrategy::Widen => {
                let mut repair_count = 0usize;
                let lower = lower.mapv(|x| {
                    if x.is_nan() {
                        repair_count += 1;
                        f32::NEG_INFINITY
                    } else {
                        x // finite and ±Inf values left as-is
                    }
                });
                let upper = upper.mapv(|x| {
                    if x.is_nan() {
                        repair_count += 1;
                        f32::INFINITY
                    } else {
                        x
                    }
                });
                if repair_count > 0 {
                    debug!(
                        repair_count,
                        ?strategy,
                        "BoundedTensor::new_repaired: repaired {} NaN elements",
                        repair_count
                    );
                }
                let (lower, upper) = Self::fix_inverted(lower, upper);
                Ok(Self {
                    lower,
                    upper,
                    l2: None,
                })
            }
        }
    }

    /// Fix inverted bounds by swapping elements where lower > upper.
    fn fix_inverted(mut lower: ArrayD<f32>, mut upper: ArrayD<f32>) -> (ArrayD<f32>, ArrayD<f32>) {
        let _ = repair_inverted_bounds_nd(&mut lower, &mut upper, InversionRepair::Swap);
        (lower, upper)
    }
}
