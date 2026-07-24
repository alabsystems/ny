// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_tensor::BoundedTensor;
use tracing::warn;

use crate::bounds::{nan_propagating_max, nan_propagating_min};

/// Check if output bounds have any element where `lower > upper` (both finite).
///
/// Inverted bounds indicate the domain is infeasible — the constraint set has no
/// solution. This is distinct from NaN/Inf overflow (which may be recoverable).
///
/// Reference: alpha-beta-CROWN `optimized_bounds.py:626-644` detects infeasibility
/// and sets `lower=+inf, upper=-inf` to signal the BaB engine to prune the domain.
/// (#2950)
pub(super) fn has_inverted_output_bounds(bounds: &BoundedTensor) -> bool {
    bounds
        .lower()
        .iter()
        .zip(bounds.upper().iter())
        .any(|(&l, &u)| l.is_finite() && u.is_finite() && l > u)
}

/// Extension trait for `BoundedTensor` to get scalar values.
///
/// # NaN/Inf handling (soundness fix for #2561)
///
/// When all bounds are NaN/Inf (e.g., from CROWN numerical instability),
/// these methods return conservative fallbacks instead of panicking:
/// - `lower_scalar()` → `f32::NEG_INFINITY` (most conservative lower bound)
/// - `upper_scalar()` → `f32::INFINITY` (most conservative upper bound)
/// - `argmin_lower_flat_idx()` → `None` (no valid index)
///
/// This ensures the BaB engine treats numerically-unstable domains as
/// "Unknown" rather than crashing the verification process.
pub(super) trait BoundedTensorExt {
    fn lower_scalar(&self) -> f32;
    fn upper_scalar(&self) -> f32;
    /// Returns the flat index of the element with the smallest valid lower bound,
    /// or `None` if all lower bounds are NaN/+Inf.
    fn argmin_lower_flat_idx(&self) -> Option<usize>;
}

impl BoundedTensorExt for BoundedTensor {
    fn lower_scalar(&self) -> f32 {
        // Empty tensors produce inverted bounds (INF, -INF) which cause false
        // verification results. See #2096.
        debug_assert!(
            !self.lower().is_empty(),
            "BoundedTensorExt::lower_scalar called on empty tensor — \
             this would produce f32::INFINITY and corrupt verification decisions"
        );
        // Skip NaN and +Inf values.
        // NaN: f32::min(x, NaN) returns NaN, silently corrupting decisions.
        // +Inf lower bound causes false verification (Inf > threshold is always true).
        // Keep -Inf: it is a valid (conservative) lower bound and must not be discarded.
        let result = self
            .lower()
            .iter()
            .copied()
            .filter(|v| !v.is_nan() && *v != f32::INFINITY)
            .fold(f32::INFINITY, nan_propagating_min);
        // If fold returned INFINITY, all values were NaN/+Inf — return conservative
        // fallback instead of panicking. NEG_INFINITY is the most conservative lower
        // bound: it can never cause false verification (NEG_INFINITY > threshold is
        // always false). See #2561.
        if result == f32::INFINITY && !self.lower().is_empty() {
            warn!(
                "BoundedTensorExt::lower_scalar: all {} lower bounds are NaN/+Inf, \
                 returning NEG_INFINITY as conservative fallback",
                self.lower().len()
            );
            return f32::NEG_INFINITY;
        }
        result
    }

    fn upper_scalar(&self) -> f32 {
        // Empty tensors produce inverted bounds (INF, -INF) which cause false
        // verification results. See #2096.
        debug_assert!(
            !self.upper().is_empty(),
            "BoundedTensorExt::upper_scalar called on empty tensor — \
             this would produce f32::NEG_INFINITY and corrupt verification decisions"
        );
        // Skip NaN and -Inf values.
        // NaN: f32::max(x, NaN) returns NaN, silently corrupting decisions.
        // -Inf upper bound causes false verification in verify_upper_bound mode
        // (ub < threshold is always true when ub = -Inf). See #2438.
        // Keep +Inf: it is a valid conservative upper bound and must not be discarded.
        let result = self
            .upper()
            .iter()
            .copied()
            .filter(|v| !v.is_nan() && *v != f32::NEG_INFINITY)
            .fold(f32::NEG_INFINITY, nan_propagating_max);
        // If fold returned NEG_INFINITY, all values were NaN/-Inf — return conservative
        // fallback instead of panicking. INFINITY is the most conservative upper bound:
        // it can never cause false verification (INFINITY < threshold is always false).
        // See #2561.
        if result == f32::NEG_INFINITY && !self.upper().is_empty() {
            warn!(
                "BoundedTensorExt::upper_scalar: all {} upper bounds are NaN/-Inf, \
                 returning INFINITY as conservative fallback",
                self.upper().len()
            );
            return f32::INFINITY;
        }
        result
    }

    fn argmin_lower_flat_idx(&self) -> Option<usize> {
        // Empty tensors would return index 0 which is meaningless. See #2096.
        debug_assert!(
            !self.lower().is_empty(),
            "BoundedTensorExt::argmin_lower_flat_idx called on empty tensor — \
             no valid index exists"
        );
        let mut best_idx: Option<usize> = None;
        let mut best_val = f32::INFINITY;
        for (idx, v) in self.lower().iter().copied().enumerate() {
            if v.is_nan() || v == f32::INFINITY {
                continue;
            }
            if v < best_val {
                best_val = v;
                best_idx = Some(idx);
            }
        }
        if best_idx.is_none() && !self.lower().is_empty() {
            warn!(
                "BoundedTensorExt::argmin_lower_flat_idx: all {} lower bounds are NaN/+Inf, \
                 no valid index exists",
                self.lower().len()
            );
        }
        best_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn make_bounds(lower: &[f32], upper: &[f32]) -> BoundedTensor {
        let l = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap();
        BoundedTensor::new(l, u).unwrap()
    }

    #[test]
    fn lower_scalar_returns_minimum() {
        let bt = make_bounds(&[-1.0, 0.5, -3.0], &[1.0, 2.0, 0.0]);
        assert_eq!(bt.lower_scalar(), -3.0);
    }

    #[test]
    fn upper_scalar_returns_maximum() {
        let bt = make_bounds(&[-1.0, 0.5, -3.0], &[1.0, 2.0, 0.0]);
        assert_eq!(bt.upper_scalar(), 2.0);
    }

    #[test]
    fn argmin_lower_returns_index_of_minimum() {
        let bt = make_bounds(&[-1.0, 0.5, -3.0], &[1.0, 2.0, 0.0]);
        assert_eq!(bt.argmin_lower_flat_idx(), Some(2));
    }

    #[test]
    fn single_element_bounds() {
        let bt = make_bounds(&[0.5], &[1.5]);
        assert_eq!(bt.lower_scalar(), 0.5);
        assert_eq!(bt.upper_scalar(), 1.5);
        assert_eq!(bt.argmin_lower_flat_idx(), Some(0));
    }

    /// Test #2096: NaN values in bounds must be skipped, not propagated.
    /// Without NaN guards, f32::min(x, NaN) returns NaN, silently corrupting
    /// downstream verification decisions.
    #[test]
    fn nan_values_are_skipped() {
        // Create bounds with NaN mixed in — use new_unchecked since
        // BoundedTensor::new rejects NaN.
        let l = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, -2.0, 1.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, f32::NAN, 5.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();

        // lower_scalar should skip NaN and return -2.0
        assert_eq!(bt.lower_scalar(), -2.0);
        // upper_scalar should skip NaN and return 5.0
        assert_eq!(bt.upper_scalar(), 5.0);
        // argmin_lower should skip NaN and return index 1 (value -2.0)
        assert_eq!(bt.argmin_lower_flat_idx(), Some(1));
    }

    /// Regression test for #2096: empty tensors trigger debug_assert.
    /// In debug builds this panics; in release builds it returns conservative
    /// fallbacks (NEG_INFINITY / INFINITY / None).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "empty tensor")]
    fn lower_scalar_panics_on_empty_debug() {
        let bt =
            BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()).unwrap();
        let _ = bt.lower_scalar();
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "empty tensor")]
    fn upper_scalar_panics_on_empty_debug() {
        let bt =
            BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()).unwrap();
        let _ = bt.upper_scalar();
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "empty tensor")]
    fn argmin_lower_panics_on_empty_debug() {
        let bt =
            BoundedTensor::concrete(ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).unwrap()).unwrap();
        let _ = bt.argmin_lower_flat_idx();
    }

    /// #2561: All-NaN lower bounds return None from argmin (no valid index).
    #[test]
    fn argmin_lower_returns_none_on_all_nan() {
        let l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, f32::NAN]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.argmin_lower_flat_idx(), None);
    }

    /// #2561: All-NaN lower bounds return NEG_INFINITY (conservative fallback).
    /// NEG_INFINITY can never cause false verification since NEG_INFINITY > threshold
    /// is always false.
    #[test]
    fn all_nan_lower_returns_neg_infinity() {
        let l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, f32::NAN]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.lower_scalar(), f32::NEG_INFINITY);
    }

    /// #2561: All-NaN upper bounds return INFINITY (conservative fallback).
    /// INFINITY can never cause false verification since INFINITY < threshold
    /// is always false.
    #[test]
    fn all_nan_upper_returns_infinity() {
        let l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -2.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, f32::NAN]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.upper_scalar(), f32::INFINITY);
    }

    /// #2414: Inf lower bound must be filtered — Inf > threshold is always true,
    /// causing false verification.
    #[test]
    fn inf_lower_is_filtered() {
        let l = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, -1.0, 2.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::INFINITY, 3.0, 5.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.lower_scalar(), -1.0);
        // argmin should also skip Inf and return index 1
        assert_eq!(bt.argmin_lower_flat_idx(), Some(1));
    }

    /// #2414: -Inf does not affect max upper bound selection.
    #[test]
    fn neg_inf_upper_does_not_change_max() {
        let l = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-5.0, -1.0, -3.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 3.0, 5.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.upper_scalar(), 5.0);
    }

    /// #2561: All-Inf lower bounds return NEG_INFINITY (conservative fallback).
    #[test]
    fn all_inf_lower_returns_neg_infinity() {
        let l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, f32::INFINITY]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, f32::INFINITY]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.lower_scalar(), f32::NEG_INFINITY);
    }

    /// +Inf upper bounds must be preserved (conservative, never tightened away).
    #[test]
    fn pos_inf_upper_is_preserved() {
        let l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -2.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.upper_scalar(), f32::INFINITY);
    }

    /// #2561: All -Inf upper bounds return INFINITY (conservative fallback).
    #[test]
    fn all_neg_inf_upper_returns_infinity() {
        let l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -2.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, f32::NEG_INFINITY])
            .unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.upper_scalar(), f32::INFINITY);
    }

    /// #2561: Mixed NaN/-Inf upper bounds return INFINITY (conservative fallback).
    #[test]
    fn all_nan_or_neg_inf_upper_returns_infinity() {
        let l = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, -2.0, -3.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, f32::NEG_INFINITY, f32::NAN])
            .unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.upper_scalar(), f32::INFINITY);
    }

    /// #2438: -Inf upper bounds are filtered from max computation.
    #[test]
    fn neg_inf_upper_is_filtered() {
        let l = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-5.0, -1.0, -3.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, -2.0, -1.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.upper_scalar(), -1.0);
    }

    /// -Inf lower bounds are valid and must be preserved (most conservative).
    #[test]
    fn neg_inf_lower_is_preserved() {
        let l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, -1.0]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.lower_scalar(), f32::NEG_INFINITY);
        assert_eq!(bt.argmin_lower_flat_idx(), Some(0));
    }

    /// #2561: All-Inf lower bounds return None from argmin.
    #[test]
    fn argmin_lower_returns_none_on_all_inf() {
        let l = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, f32::INFINITY]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, f32::INFINITY]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.argmin_lower_flat_idx(), None);
    }

    /// #2561: Mixed NaN/+Inf lower bounds return NEG_INFINITY from lower_scalar.
    #[test]
    fn mixed_nan_inf_lower_returns_neg_infinity() {
        let l =
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, f32::INFINITY, f32::NAN]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.lower_scalar(), f32::NEG_INFINITY);
    }

    /// #2561: Mixed NaN/+Inf lower bounds return None from argmin.
    #[test]
    fn argmin_lower_returns_none_on_mixed_nan_inf() {
        let l =
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, f32::INFINITY, f32::NAN]).unwrap();
        let u = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let bt = BoundedTensor::new_unchecked(l, u).unwrap();
        assert_eq!(bt.argmin_lower_flat_idx(), None);
    }
}
