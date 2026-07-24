// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== ReduceSum tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_reduce_sum_last_axis() {
    // Test sum over last axis with keepdims=true
    // Input: 2x3 tensor
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let reduce = ReduceSumLayer::new(vec![-1], true);
    let output = reduce.propagate_ibp(&input).unwrap();

    // Output shape should be [2, 1]
    assert_eq!(output.shape(), &[2, 1]);

    // Row 0: sum([1,2,3]) = 6, sum([2,3,4]) = 9. The sound IBP forward directed-rounds the
    // f64 sum OUTWARD (lower down, upper up), so it ENCLOSES the true value within ~1 ULP.
    assert!(
        output.lower()[[0, 0]] <= 6.0 && (output.lower()[[0, 0]] - 6.0).abs() < 1e-5,
        "Sum lower of [1,2,3] should enclose 6, got {}",
        output.lower()[[0, 0]]
    );
    assert!(
        output.upper()[[0, 0]] >= 9.0 && (output.upper()[[0, 0]] - 9.0).abs() < 1e-5,
        "Sum upper of [2,3,4] should enclose 9, got {}",
        output.upper()[[0, 0]]
    );

    // Row 1: sum([4,5,6]) = 15, sum([5,6,7]) = 18
    assert!(
        output.lower()[[1, 0]] <= 15.0 && (output.lower()[[1, 0]] - 15.0).abs() < 1e-5,
        "Sum lower of [4,5,6] should enclose 15, got {}",
        output.lower()[[1, 0]]
    );
    assert!(
        output.upper()[[1, 0]] >= 18.0 && (output.upper()[[1, 0]] - 18.0).abs() < 1e-5,
        "Sum upper of [5,6,7] should enclose 18, got {}",
        output.upper()[[1, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_reduce_sum_no_keepdims() {
    // Test sum over last axis with keepdims=false
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let upper = lower.clone();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let reduce = ReduceSumLayer::new(vec![-1], false);
    let output = reduce.propagate_ibp(&input).unwrap();

    // Output shape should be [2] (dimension removed)
    assert_eq!(output.shape(), &[2]);

    // Row 0: sum([1,2,3,4]) = 10 (point input; lower directed-rounds DOWN, encloses 10).
    assert!(
        output.lower()[[0]] <= 10.0 && (output.lower()[[0]] - 10.0).abs() < 1e-5,
        "Sum of [1,2,3,4] should enclose 10, got {}",
        output.lower()[[0]]
    );

    // Row 1: sum([5,6,7,8]) = 26
    assert!(
        output.lower()[[1]] <= 26.0 && (output.lower()[[1]] - 26.0).abs() < 1e-5,
        "Sum of [5,6,7,8] should enclose 26, got {}",
        output.lower()[[1]]
    );
}

// ==================== ReduceSum CROWN tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_reduce_sum_crown_backward_keepdims() {
    // Test CROWN backward pass for ReduceSum with keepdims=true
    // Input: 2x3 tensor, output: 2x1 tensor
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds on the output (2 elements after reduction)
    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceSumLayer::new(vec![-1], true);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Output should have shape (2, 6) - 2 outputs, 6 inputs
    assert_eq!(result.lower_a.nrows(), 2);
    assert_eq!(result.lower_a.ncols(), 6);

    // For sum over axis -1, each coefficient should be 1 (no scaling unlike mean)
    // Row 0 should have 1 for columns 0,1,2 (input row 0) and 0 elsewhere
    for j in 0..3 {
        assert!(
            (result.lower_a[[0, j]] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }
    for j in 3..6 {
        assert!(
            result.lower_a[[0, j]].abs() < 1e-6,
            "Expected 0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }

    // Row 1 should have 1 for columns 3,4,5 (input row 1) and 0 elsewhere
    for j in 0..3 {
        assert!(
            result.lower_a[[1, j]].abs() < 1e-6,
            "Expected 0, got {} at [1,{}]",
            result.lower_a[[1, j]],
            j
        );
    }
    for j in 3..6 {
        assert!(
            (result.lower_a[[1, j]] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {} at [1,{}]",
            result.lower_a[[1, j]],
            j
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reduce_sum_crown_soundness() {
    // Test that CROWN bounds are sound for ReduceSum
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceSumLayer::new(vec![-1], true);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Concretize bounds
    let concrete = result.concretize(&pre_activation);

    // IBP result for comparison
    let ibp_result = reduce.propagate_ibp(&pre_activation).unwrap();

    // CROWN should give bounds that contain IBP bounds (or be equal for linear ops)
    // Row 0: sum of [1,2,3] = 6 (lower), sum of [2,3,4] = 9 (upper)
    assert!(
        (concrete.lower()[[0]] - ibp_result.lower()[[0, 0]]).abs() < 1e-5,
        "CROWN lower {} should equal IBP lower {} for linear op",
        concrete.lower()[[0]],
        ibp_result.lower()[[0, 0]]
    );
    assert!(
        (concrete.upper()[[0]] - ibp_result.upper()[[0, 0]]).abs() < 1e-5,
        "CROWN upper {} should equal IBP upper {} for linear op",
        concrete.upper()[[0]],
        ibp_result.upper()[[0, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_reduce_sum_crown_no_keepdims() {
    // Test CROWN backward pass for ReduceSum with keepdims=false
    // Input: 2x4 tensor, output: 2 elements (dimension removed)
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let pre_upper = pre_lower.clone();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds on the output (2 elements after reduction)
    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceSumLayer::new(vec![-1], false);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Output should have shape (2, 8) - 2 outputs, 8 inputs
    assert_eq!(result.lower_a.nrows(), 2);
    assert_eq!(result.lower_a.ncols(), 8);

    // Row 0 should have 1 for columns 0,1,2,3 and 0 elsewhere
    for j in 0..4 {
        assert!(
            (result.lower_a[[0, j]] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }
    for j in 4..8 {
        assert!(
            result.lower_a[[0, j]].abs() < 1e-6,
            "Expected 0, got {} at [0,{}]",
            result.lower_a[[0, j]],
            j
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reduce_sum_linear_without_shape_fails_fast() {
    let reduce = ReduceSumLayer::new(vec![-1], true);
    let linear_bounds = LinearBounds::identity(2);

    let err = reduce.propagate_linear(&linear_bounds).unwrap_err();
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

/// Regression test #2816: ReduceSum CROWN backward with zero-sized dimension
/// in pre-activation shape must return error, not panic on divide-by-zero stride.
#[test]
fn test_reduce_sum_crown_zero_dimension_returns_error_2816() {
    // Pre-activation with a zero-sized dimension: shape [3, 0]
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[3, 0]), vec![]).expect("invariant: valid zero-dim shape");
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[3, 0]), vec![]).expect("invariant: valid zero-dim shape");
    let pre_activation =
        BoundedTensor::new(pre_lower, pre_upper).expect("invariant: matching shapes");

    let linear_bounds = LinearBounds::identity(3);
    let reduce = ReduceSumLayer::new(vec![-1], true);

    // Must return error, not panic.
    let result = reduce.propagate_linear_with_bounds(&linear_bounds, &pre_activation);
    assert!(
        result.is_err(),
        "zero-dimension pre-activation must return error, not panic"
    );
}

/// Duplicate axes must be rejected — double-reduction produces wrong bounds
/// or panics when `output_shape.remove(axis)` shifts indices. (#2946)
#[ntest::timeout(10000)]
#[test]
fn test_reduce_sum_duplicate_axes_returns_error_2946() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vec![1.0; 24]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vec![2.0; 24]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // axes=[1, 1]: explicit duplicate
    let reduce = ReduceSumLayer::new(vec![1, 1], true);
    let result = reduce.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "duplicate axes [1, 1] must return error, got {:?}",
        result
    );
}

/// #vnncomp-aw-soundness self-audit regression: the sound ReduceSum/Mean IBP forward must
/// ENCLOSE the true sum under f32 absorption, where the old round-to-nearest sum_axis EXCLUDED
/// it. sum over [2^24, 1]: f32 gives 2^24 (16777217 not representable) but the true sum is
/// 2^24+1 — the old box ceiling 2^24 lay BELOW the true value (a false-proof). After the fix
/// the upper bound is directed-rounded UP and encloses 2^24+1.
#[test]
fn reduce_sum_ibp_encloses_under_f32_absorption() {
    let two24 = 16_777_216.0_f32; // 2^24
    let pt = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![two24, 1.0]).unwrap();
    let input = BoundedTensor::new(pt.clone(), pt).unwrap();
    let out = ReduceSumLayer::new(vec![-1], false)
        .propagate_ibp(&input)
        .unwrap();
    let true_sum = (two24 as f64) + 1.0; // 16777217, exact in f64
    let upper = out.upper()[[0]] as f64;
    assert!(
        upper >= true_sum,
        "upper {upper} must ENCLOSE the true sum {true_sum} (old f32 sum gave {two24} < true)"
    );
}
