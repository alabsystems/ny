// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Interval arithmetic and transforms for bounded tensors.

use ndarray::{ArrayD, IxDyn};
use ny_core::{NyError, Result};

use super::{BoundedTensor, RepairStrategy};
use crate::rounding::{add_down_f32, add_up_f32, mul_down_f32, mul_up_f32};

/// NaN-safe extrema for interval endpoint products.
///
/// If **any** product is NaN (from undefined IEEE-754 operations like `0 * inf`),
/// returns conservative unbounded interval `[-inf, inf]` to preserve soundness.
///
/// Rationale: dropping NaN products and taking min/max of the rest can produce
/// bounds that are too tight. For example, `[-1, 0] * [inf, inf]` has products
/// `(-inf, -inf, NaN, NaN)` — dropping NaN gives `[-inf, -inf]`, but the true
/// interval should include 0 (by continuous extension at the indeterminate point).
/// The alpha-beta-CROWN reference (auto_LiRPA/interval_bound.py:73-93) handles
/// this by replacing any NaN bound with `-inf`/`+inf` after propagation.
#[inline]
fn nan_safe_product_extrema(p1: f32, p2: f32, p3: f32, p4: f32) -> (f32, f32) {
    // If any product is NaN, conservatively widen to unbounded.
    if p1.is_nan() || p2.is_nan() || p3.is_nan() || p4.is_nan() {
        return (f32::NEG_INFINITY, f32::INFINITY);
    }

    let min_val = p1.min(p2).min(p3.min(p4));
    let max_val = p1.max(p2).max(p3.max(p4));
    (min_val, max_val)
}

/// Interval arithmetic operations on bounded tensors.
impl BoundedTensor {
    /// Element-wise addition of two bounded tensors.
    #[inline]
    pub fn add(&self, other: &BoundedTensor) -> Result<BoundedTensor> {
        if self.shape() != other.shape() {
            return Err(NyError::shape_mismatch(
                self.shape().to_vec(),
                other.shape().to_vec(),
            ));
        }
        // DIRECTED: a plain f32 `+` is round-to-NEAREST, so `a_lo + b_lo` can
        // land up to half an ULP ABOVE the true sum — a lower bound that
        // excludes the value it bounds. Each endpoint must move outward.
        // `add_down_f32`/`add_up_f32` step only when the addition was inexact,
        // so exactly-representable sums (the common case) cost nothing.
        let lower = ndarray::Zip::from(self.lower())
            .and(other.lower())
            .map_collect(|&x, &y| add_down_f32(x, y));
        let upper = ndarray::Zip::from(self.upper())
            .and(other.upper())
            .map_collect(|&x, &y| add_up_f32(x, y));
        // Repair NaN from inf + (-inf) etc. at the type boundary (#3423).
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
    }

    /// Element-wise multiplication (interval multiplication).
    /// Interval multiplication: `[a,b] * [c,d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]`
    #[inline]
    pub fn mul(&self, other: &BoundedTensor) -> Result<BoundedTensor> {
        if self.shape() != other.shape() {
            return Err(NyError::shape_mismatch(
                self.shape().to_vec(),
                other.shape().to_vec(),
            ));
        }

        let a = self.lower();
        let b = self.upper();
        let c = other.lower();
        let d = other.upper();

        // Compute the four endpoint products element-wise inside a single Zip
        // over the four input views, writing directly into freshly-allocated
        // output arrays. This is bit-identical to materializing `a*c, a*d, b*c,
        // b*d` as whole arrays (f32 multiplication is element-wise deterministic),
        // but allocates only the two outputs instead of 4 product arrays plus
        // 2 clones (6 allocations → 2). (#perf)
        let mut lower = ArrayD::<f32>::zeros(a.raw_dim());
        let mut upper = ArrayD::<f32>::zeros(a.raw_dim());

        ndarray::Zip::from(&mut lower)
            .and(&mut upper)
            .and(a)
            .and(b)
            .and(c)
            .and(d)
            .for_each(|l, u, &a, &b, &c, &d| {
                // DIRECTED: the min over the four endpoint products must round
                // DOWN and the max must round UP. A plain f32 `*` rounds to
                // nearest, which moves both inward. Computing each corner twice
                // (once each way) is what keeps the enclosure valid; the two
                // agree whenever the product is representable, which is most of
                // the time, so this is not two intervals' worth of slack.
                let (min_val, _) = nan_safe_product_extrema(
                    mul_down_f32(a, c),
                    mul_down_f32(a, d),
                    mul_down_f32(b, c),
                    mul_down_f32(b, d),
                );
                let (_, max_val) = nan_safe_product_extrema(
                    mul_up_f32(a, c),
                    mul_up_f32(a, d),
                    mul_up_f32(b, c),
                    mul_up_f32(b, d),
                );
                *l = min_val;
                *u = max_val;
            });

        // KEEP unchecked: nan_safe_product_extrema() returns ordered finite/inf
        // endpoints and never leaves NaN behind; lower/upper shapes still match.
        Ok(BoundedTensor::from_parts_unchecked(lower, upper))
    }

    /// Scalar multiplication.
    ///
    /// If `scalar` is NaN or Inf, returns conservative `[-inf, +inf]` bounds.
    /// Without this guard, `NaN >= 0.0` → false (IEEE 754), causing the code
    /// to incorrectly treat NaN as negative and swap bounds. See #2895.
    pub fn scale(&self, scalar: f32) -> BoundedTensor {
        if !scalar.is_finite() {
            return BoundedTensor::new_conservative(self.shape());
        }
        // DIRECTED, and the direction follows the OUTPUT endpoint, not the
        // input one: after a negative scalar swaps them, the array that becomes
        // the lower bound must round down.
        let (lower, upper) = if scalar >= 0.0 {
            (
                self.lower().mapv(|v| mul_down_f32(v, scalar)),
                self.upper().mapv(|v| mul_up_f32(v, scalar)),
            )
        } else {
            // Negative scalar swaps bounds
            (
                self.upper().mapv(|v| mul_down_f32(v, scalar)),
                self.lower().mapv(|v| mul_up_f32(v, scalar)),
            )
        };
        // SAFETY: new_repaired() repairs NaN from edge cases like 0 * inf (#3423).
        // The .expect() can only fail on shape mismatch, which is impossible here
        // because both arrays derive from the same BoundedTensor via mapv().
        // WARNING: Do NOT replace new_repaired().expect() with from_parts_unchecked —
        // new_repaired() performs essential NaN repair (e.g., inf * 0.0 → NaN → [-inf, +inf])
        // that from_parts_unchecked would skip in release builds. (#4253)
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
            .expect("invariant: shapes from same BoundedTensor always match")
    }

    /// Scalar addition.
    pub fn shift(&self, scalar: f32) -> BoundedTensor {
        // DIRECTED, same reasoning as `add`.
        let lower = self.lower().mapv(|v| add_down_f32(v, scalar));
        let upper = self.upper().mapv(|v| add_up_f32(v, scalar));
        // SAFETY: new_repaired() repairs NaN from inf + (-inf) etc. (#3423).
        // The .expect() can only fail on shape mismatch, which is impossible here
        // because both arrays derive from the same BoundedTensor via mapv().
        // WARNING: Do NOT replace new_repaired().expect() with from_parts_unchecked —
        // new_repaired() performs essential NaN repair that from_parts_unchecked would
        // skip in release builds. (#4253)
        BoundedTensor::new_repaired(lower, upper, RepairStrategy::Conservative)
            .expect("invariant: shapes from same BoundedTensor always match")
    }

    /// General transpose with arbitrary permutation.
    ///
    /// # Arguments
    /// * `perm` - Permutation of dimensions. E.g., [0, 2, 1, 3] swaps dims 1 and 2.
    ///
    /// # Example
    /// ```text
    /// // Swap heads and sequence dimensions:
    /// let transposed = tensor.transpose(&[0, 2, 1, 3])?;  // [B,S,H,D] -> [B,H,S,D]
    /// ```
    pub fn transpose(&self, perm: &[usize]) -> Result<BoundedTensor> {
        let shape = self.shape();
        let ndim = shape.len();

        if perm.len() != ndim {
            return Err(NyError::InvalidSpec(format!(
                "Permutation length {} doesn't match tensor ndim {}",
                perm.len(),
                ndim
            )));
        }

        // Validate permutation
        let mut sorted_perm = perm.to_vec();
        sorted_perm.sort_unstable();
        let expected: Vec<usize> = (0..ndim).collect();
        if sorted_perm != expected {
            return Err(NyError::InvalidSpec(format!(
                "Invalid permutation {:?}, expected a permutation of 0..{}",
                perm, ndim
            )));
        }

        let lower = self.lower().clone().permuted_axes(IxDyn(perm));
        let upper = self.upper().clone().permuted_axes(IxDyn(perm));

        // Ensure contiguous memory layout
        let lower = lower.as_standard_layout().into_owned();
        let upper = upper.as_standard_layout().into_owned();

        BoundedTensor::new(lower, upper)
    }

    /// Transpose the last two dimensions.
    ///
    /// For a tensor of shape [..., M, N], returns shape [..., N, M].
    /// This is commonly used for attention: K^T = K.transpose_last_two()
    pub fn transpose_last_two(&self) -> Result<BoundedTensor> {
        let shape = self.shape();
        let ndim = shape.len();

        if ndim < 2 {
            return Err(NyError::InvalidSpec(
                "Cannot transpose tensor with fewer than 2 dimensions".to_string(),
            ));
        }

        // Build the permutation: [..., ndim-1, ndim-2]
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(ndim - 2, ndim - 1);

        let lower = self.lower().clone().permuted_axes(IxDyn(&perm));
        let upper = self.upper().clone().permuted_axes(IxDyn(&perm));

        // Ensure contiguous memory layout
        let lower = lower.as_standard_layout().into_owned();
        let upper = upper.as_standard_layout().into_owned();

        BoundedTensor::new(lower, upper)
    }
}
