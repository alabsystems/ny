// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Shared math utilities for normalization layer IBP bound propagation.
//!
//! These free functions were duplicated across InstanceNorm, RmsNorm, and
//! AdaIN IBP implementations. Extracting them here eliminates duplication
//! and centralizes the interval arithmetic.
//!
//! Reference: designs/2026-02-27-normalization-trait-dedup.md (Slice 6)

use ndarray::ArrayD;
use ny_tensor::next_up_f32;

/// Split finite interval endpoints into a midpoint and outward-rounded radius.
///
/// The direct `f32` formulas `(lower + upper) / 2` and
/// `(upper - lower) / 2` can overflow and can lose an endpoint when the stored
/// midpoint is not equidistant from both endpoints. This routine computes the
/// midpoint in `f64`, then rounds the maximum distance from the stored
/// midpoint upward. Consequently, `center ± radius` contains the original
/// interval in real arithmetic.
///
/// Returns `None` for mismatched shapes, non-finite or inverted endpoints, or
/// when a finite outward radius cannot be represented.
pub(crate) fn outward_midpoint_radius(
    lower: &ArrayD<f32>,
    upper: &ArrayD<f32>,
) -> Option<(ArrayD<f32>, ArrayD<f32>)> {
    if lower.shape() != upper.shape() {
        return None;
    }

    let mut center = lower.clone();
    let mut radius = lower.clone();
    for (((center_slot, radius_slot), &lo), &hi) in center
        .iter_mut()
        .zip(radius.iter_mut())
        .zip(lower.iter())
        .zip(upper.iter())
    {
        if !lo.is_finite() || !hi.is_finite() || lo > hi {
            return None;
        }
        if lo == hi {
            *center_slot = lo;
            *radius_slot = 0.0;
            continue;
        }

        let stored_center = f64::midpoint(f64::from(lo), f64::from(hi)) as f32;
        let required_radius_f64 = (f64::from(stored_center) - f64::from(lo))
            .abs()
            .max((f64::from(hi) - f64::from(stored_center)).abs());
        // The unconditional outward step covers both the f64 subtraction's
        // rounding and the subsequent f64-to-f32 conversion.
        let stored_radius = next_up_f32(required_radius_f64 as f32);
        if !stored_center.is_finite() || !stored_radius.is_finite() {
            return None;
        }

        *center_slot = stored_center;
        *radius_slot = stored_radius;
    }

    Some((center, radius))
}

/// Compute interval bounds for squaring: `[xl, xu] → [xl², xu²]` with
/// correct handling of intervals that straddle zero.
///
/// - If `xl >= 0`: both bounds are non-negative, `[xl², xu²]`
/// - If `xu <= 0`: both bounds are non-positive, `[xu², xl²]` (reversed)
/// - If `xl < 0 < xu`: interval straddles zero, `[0, max(|xl|, |xu|)²]`
pub(crate) fn square_interval_bounds(xl: f32, xu: f32) -> (f32, f32) {
    if xl >= 0.0 {
        (xl * xl, xu * xu)
    } else if xu <= 0.0 {
        (xu * xu, xl * xl)
    } else {
        let max_sq = xl.abs().max(xu.abs());
        (0.0, max_sq * max_sq)
    }
}

/// Convert a flat batch index to multi-dimensional batch prefix indices.
///
/// Given a tensor shape `[..batch_dims, C, T]` (where `ndim` is the total
/// number of dimensions), converts a flat batch index into the corresponding
/// multi-dimensional indices for the batch dimensions.
///
/// For example, shape `[2, 3, C, T]` with `batch_idx=4` → `[1, 1]`.
pub(crate) fn compute_batch_prefix(shape: &[usize], ndim: usize, batch_idx: usize) -> Vec<usize> {
    let batch_dims = &shape[..ndim - 2];
    let mut prefix = vec![0usize; batch_dims.len()];
    let mut remaining = batch_idx;
    for d in (0..batch_dims.len()).rev() {
        prefix[d] = remaining % batch_dims[d];
        remaining /= batch_dims[d];
    }
    prefix
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_outward_midpoint_radius_contains_asymmetric_narrow_interval() {
        let lower = arr1(&[-0.20841855, 0.023820192]).into_dyn();
        let upper = arr1(&[-0.20841847, 0.023820221]).into_dyn();
        let (center, radius) =
            outward_midpoint_radius(&lower, &upper).expect("finite ordered interval");

        for i in 0..lower.len() {
            assert!(f64::from(center[[i]]) - f64::from(radius[[i]]) <= f64::from(lower[[i]]));
            assert!(f64::from(center[[i]]) + f64::from(radius[[i]]) >= f64::from(upper[[i]]));
        }
    }

    #[test]
    fn test_outward_midpoint_radius_preserves_point_interval() {
        let point = arr1(&[-1.0, -0.0, 0.0, 1.0]).into_dyn();
        let (center, radius) =
            outward_midpoint_radius(&point, &point).expect("finite point interval");
        assert_eq!(center, point);
        assert!(radius.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn test_outward_midpoint_radius_contains_adjacent_large_finite_endpoints() {
        let lower_value = ny_tensor::next_down_f32(f32::MAX);
        let lower = arr1(&[lower_value]).into_dyn();
        let upper = arr1(&[f32::MAX]).into_dyn();
        let (center, radius) =
            outward_midpoint_radius(&lower, &upper).expect("large finite interval");

        assert!(radius[[0]] > 0.0);
        assert!(radius[[0]].is_finite());
        assert!(f64::from(center[[0]]) - f64::from(radius[[0]]) <= f64::from(lower_value));
        assert!(f64::from(center[[0]]) + f64::from(radius[[0]]) >= f64::from(f32::MAX));
    }

    #[test]
    fn test_outward_midpoint_radius_rejects_invalid_or_unrepresentable_input() {
        for (lower, upper) in [
            (arr1(&[f32::NAN]).into_dyn(), arr1(&[0.0]).into_dyn()),
            (
                arr1(&[f32::NEG_INFINITY]).into_dyn(),
                arr1(&[0.0]).into_dyn(),
            ),
            (arr1(&[0.0]).into_dyn(), arr1(&[f32::INFINITY]).into_dyn()),
            (arr1(&[1.0]).into_dyn(), arr1(&[0.0]).into_dyn()),
            (arr1(&[f32::MIN]).into_dyn(), arr1(&[f32::MAX]).into_dyn()),
        ] {
            assert!(outward_midpoint_radius(&lower, &upper).is_none());
        }
    }

    #[test]
    fn test_square_interval_both_positive() {
        let (lo, hi) = square_interval_bounds(2.0, 5.0);
        assert!((lo - 4.0).abs() < 1e-6);
        assert!((hi - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_square_interval_both_negative() {
        let (lo, hi) = square_interval_bounds(-5.0, -2.0);
        // xu² = (-2)² = 4, xl² = (-5)² = 25
        assert!((lo - 4.0).abs() < 1e-6);
        assert!((hi - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_square_interval_straddles_zero() {
        let (lo, hi) = square_interval_bounds(-3.0, 4.0);
        assert_eq!(lo, 0.0);
        assert!((hi - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_square_interval_straddles_zero_neg_dominant() {
        let (lo, hi) = square_interval_bounds(-5.0, 2.0);
        assert_eq!(lo, 0.0);
        assert!((hi - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_square_interval_zero_boundary() {
        let (lo, hi) = square_interval_bounds(0.0, 3.0);
        assert_eq!(lo, 0.0);
        assert!((hi - 9.0).abs() < 1e-6);

        let (lo2, hi2) = square_interval_bounds(-3.0, 0.0);
        assert_eq!(lo2, 0.0);
        assert!((hi2 - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_square_interval_point() {
        let (lo, hi) = square_interval_bounds(3.0, 3.0);
        assert!((lo - 9.0).abs() < 1e-6);
        assert!((hi - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_batch_prefix_simple() {
        // shape [2, 3, C, T], ndim=4, batch_idx=4 → [1, 1]
        let prefix = compute_batch_prefix(&[2, 3, 5, 7], 4, 4);
        assert_eq!(prefix, vec![1, 1]);
    }

    #[test]
    fn test_compute_batch_prefix_zero() {
        let prefix = compute_batch_prefix(&[2, 3, 5, 7], 4, 0);
        assert_eq!(prefix, vec![0, 0]);
    }

    #[test]
    fn test_compute_batch_prefix_last() {
        // shape [2, 3, C, T], ndim=4, batch_idx=5 → [1, 2]
        let prefix = compute_batch_prefix(&[2, 3, 5, 7], 4, 5);
        assert_eq!(prefix, vec![1, 2]);
    }
}
