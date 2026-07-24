// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for #4204: NaN guard coverage in `finalize_bias_bounds`.

use super::common::{finalize_bias_bounds, BackwardBatchContext};
use crate::BatchedLinearBounds;
use ndarray::{Array1, Array2, ArrayD, IxDyn};

/// Helper: build a 1x1 context for single-element bias tests.
fn ctx_1x1(conv_in_size: usize) -> BackwardBatchContext {
    BackwardBatchContext {
        total_batch: 1,
        total_rows: 1,
        out_dim: 1,
        mid_dim: 1,
        out_a_shape: vec![1, 1, conv_in_size],
        out_b_shape: vec![1, 1],
    }
}

/// Regression test for #4204: finalize_bias_bounds NaN guard on inf + (-inf).
///
/// When input lower_b is +inf (from a previous layer's ±inf fallback) and
/// the A-matrix contains inf values that produce -inf bias contribution,
/// the sum is NaN. The guard at common.rs lines 183-188 should replace NaN
/// with conservative -inf/+inf bounds instead of propagating NaN.
///
/// This is defense-in-depth: normally zero_nonfinite_rows catches inf
/// A-matrix entries first. We simulate a bypass by passing
/// nonfinite_rows=[false].
#[test]
fn test_finalize_bias_bounds_nan_guard_inf_plus_neg_inf_4204() {
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), f32::INFINITY),
        ArrayD::from_elem(IxDyn(&[1, 1]), f32::INFINITY),
        ArrayD::from_elem(IxDyn(&[1, 1, 1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1, 1]), f32::NEG_INFINITY),
        vec![1, 1],
        vec![1, 1],
    )
    .expect("valid bounds");

    let ctx = ctx_1x1(1);

    // A-matrix views: lower=+inf, upper=-inf.
    let la = Array2::from_elem((1, 1), f32::INFINITY);
    let ua = Array2::from_elem((1, 1), f32::NEG_INFINITY);
    let la3 = la.view().into_shape_with_order((1, 1, 1)).expect("reshape");
    let ua3 = ua.view().into_shape_with_order((1, 1, 1)).expect("reshape");

    // bias=[-1.0]: spatial_sum(+inf)*(-1.0) = -inf.
    // lower_b(+inf) + (-inf) = NaN → guard → -inf.
    let bias = Array1::from_vec(vec![-1.0_f32]);

    let (new_lb, new_ub) = finalize_bias_bounds(
        &bounds,
        &ctx,
        1,
        1,
        la3,
        ua3,
        Some(&bias),
        &[false],
        &[false],
    )
    .expect("should not fail");

    let lb = new_lb[[0, 0]];
    let ub = new_ub[[0, 0]];
    assert!(!lb.is_nan(), "lower_b NaN guard should fire, got {lb}");
    assert!(!ub.is_nan(), "upper_b NaN guard should fire, got {ub}");
    assert_eq!(lb, f32::NEG_INFINITY, "NaN lower_b → -inf");
    assert_eq!(ub, f32::INFINITY, "NaN upper_b → +inf");
}

/// Regression test for #4204: finalize_bias_bounds produces sound output
/// for normal finite inputs (no NaN guard triggered).
#[test]
fn test_finalize_bias_bounds_finite_no_nan_4204() {
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![1.0, -0.5]).expect("la"),
        ArrayD::zeros(IxDyn(&[1, 1])),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![0.5, 1.0]).expect("ua"),
        ArrayD::zeros(IxDyn(&[1, 1])),
        vec![1, 2],
        vec![1, 1],
    )
    .expect("valid bounds");

    let ctx = BackwardBatchContext {
        total_batch: 1,
        total_rows: 1,
        out_dim: 1,
        mid_dim: 2,
        out_a_shape: vec![1, 1, 2],
        out_b_shape: vec![1, 1],
    };

    let la = Array2::from_shape_vec((1, 2), vec![1.0_f32, -0.5]).expect("la");
    let ua = Array2::from_shape_vec((1, 2), vec![0.5_f32, 1.0]).expect("ua");
    let la3 = la.view().into_shape_with_order((1, 1, 2)).expect("reshape");
    let ua3 = ua.view().into_shape_with_order((1, 1, 2)).expect("reshape");

    let bias = Array1::from_vec(vec![0.5_f32, -0.3]);

    let (new_lb, new_ub) = finalize_bias_bounds(
        &bounds,
        &ctx,
        2,
        1,
        la3,
        ua3,
        Some(&bias),
        &[false],
        &[false],
    )
    .expect("should not fail");

    let lb = new_lb[[0, 0]];
    let ub = new_ub[[0, 0]];
    assert!(lb.is_finite(), "lower_b should be finite, got {lb}");
    assert!(ub.is_finite(), "upper_b should be finite, got {ub}");
    // lower_sum = 1.0*0.5 + (-0.5)*(-0.3) = 0.65
    // upper_sum = 0.5*0.5 + 1.0*(-0.3) = -0.05
    assert!((lb - 0.65).abs() < 0.01, "lower_b ≈ 0.65, got {lb}");
    assert!((ub - (-0.05)).abs() < 0.01, "upper_b ≈ -0.05, got {ub}");
}
