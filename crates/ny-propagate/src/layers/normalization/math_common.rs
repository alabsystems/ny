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
