// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== ReduceMean tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_reduce_mean_last_axis() {
    // Test mean over last axis with keepdims=true
    // Input: 2x3 tensor
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let reduce = ReduceMeanLayer::new(vec![-1], true);
    let output = reduce.propagate_ibp(&input).unwrap();

    // Output shape should be [2, 1]
    assert_eq!(output.shape(), &[2, 1]);

    // Row 0: mean([1,2,3]) = 2, mean([2,3,4]) = 3
    assert!(
        (output.lower()[[0, 0]] - 2.0).abs() < 1e-6,
        "Mean lower of [1,2,3] should be 2, got {}",
        output.lower()[[0, 0]]
    );
    assert!(
        (output.upper()[[0, 0]] - 3.0).abs() < 1e-6,
        "Mean upper of [2,3,4] should be 3, got {}",
        output.upper()[[0, 0]]
    );

    // Row 1: mean([4,5,6]) = 5, mean([5,6,7]) = 6
    assert!(
        (output.lower()[[1, 0]] - 5.0).abs() < 1e-6,
        "Mean lower of [4,5,6] should be 5, got {}",
        output.lower()[[1, 0]]
    );
    assert!(
        (output.upper()[[1, 0]] - 6.0).abs() < 1e-6,
        "Mean upper of [5,6,7] should be 6, got {}",
        output.upper()[[1, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_reduce_mean_no_keepdims() {
    // Test mean over last axis with keepdims=false
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let upper = lower.clone();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let reduce = ReduceMeanLayer::new(vec![-1], false);
    let output = reduce.propagate_ibp(&input).unwrap();

    // Output shape should be [2] (dimension removed)
    assert_eq!(output.shape(), &[2]);

    // Row 0: mean([1,2,3,4]) = 2.5
    assert!(
        (output.lower()[[0]] - 2.5).abs() < 1e-6,
        "Mean of [1,2,3,4] should be 2.5, got {}",
        output.lower()[[0]]
    );

    // Row 1: mean([5,6,7,8]) = 6.5
    assert!(
        (output.lower()[[1]] - 6.5).abs() < 1e-6,
        "Mean of [5,6,7,8] should be 6.5, got {}",
        output.lower()[[1]]
    );
}

// ==================== ReduceMean CROWN tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_reduce_mean_crown_backward_keepdims() {
    // Test CROWN backward pass for ReduceMean with keepdims=true
    // Input: 2x3 tensor, output: 2x1 tensor
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    // Identity linear bounds on the output (2 elements after reduction)
    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceMeanLayer::new(vec![-1], true);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Output should have shape (2, 6) - 2 outputs, 6 inputs
    assert_eq!(result.lower_a.nrows(), 2);
    assert_eq!(result.lower_a.ncols(), 6);

    // For mean over axis -1 with 3 elements, each coefficient should be 1/3
    let scale = 1.0 / 3.0;

    // Row 0 should have 1/3 for columns 0,1,2 (input row 0) and 0 elsewhere
    for j in 0..3 {
        assert!(
            (result.lower_a[[0, j]] - scale).abs() < 1e-6,
            "Expected {}, got {} at [0,{}]",
            scale,
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

    // Row 1 should have 1/3 for columns 3,4,5 (input row 1) and 0 elsewhere
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
            (result.lower_a[[1, j]] - scale).abs() < 1e-6,
            "Expected {}, got {} at [1,{}]",
            scale,
            result.lower_a[[1, j]],
            j
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reduce_mean_crown_soundness() {
    // Test that CROWN bounds are sound for ReduceMean
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceMeanLayer::new(vec![-1], true);

    let result = reduce
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Concretize bounds
    let concrete = result.concretize(&pre_activation);

    // IBP result for comparison
    let ibp_result = reduce.propagate_ibp(&pre_activation).unwrap();

    // CROWN should give bounds that contain IBP bounds (or be equal)
    // Row 0: mean of [1,2,3] = 2 (lower), mean of [2,3,4] = 3 (upper)
    assert!(
        concrete.lower()[[0]] <= ibp_result.lower()[[0, 0]] + 1e-5,
        "CROWN lower {} should be <= IBP lower {}",
        concrete.lower()[[0]],
        ibp_result.lower()[[0, 0]]
    );
    assert!(
        concrete.upper()[[0]] >= ibp_result.upper()[[0, 0]] - 1e-5,
        "CROWN upper {} should be >= IBP upper {}",
        concrete.upper()[[0]],
        ibp_result.upper()[[0, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_reduce_mean_linear_without_shape_fails_fast() {
    let reduce = ReduceMeanLayer::new(vec![-1], true);
    let linear_bounds = LinearBounds::identity(2);

    let err = reduce.propagate_linear(&linear_bounds).unwrap_err();
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

/// Regression test #2816: ReduceMean CROWN backward with zero-sized dimension
/// in pre-activation shape must return error, not panic on divide-by-zero stride.
#[test]
fn test_reduce_mean_crown_zero_dimension_returns_error_2816() {
    // Pre-activation with a zero-sized dimension: shape [2, 0]
    let pre_lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 0]), vec![]).expect("invariant: valid zero-dim shape");
    let pre_upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 0]), vec![]).expect("invariant: valid zero-dim shape");
    let pre_activation =
        BoundedTensor::new(pre_lower, pre_upper).expect("invariant: matching shapes");

    let linear_bounds = LinearBounds::identity(2);
    let reduce = ReduceMeanLayer::new(vec![-1], true);

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
fn test_reduce_mean_duplicate_axes_returns_error_2946() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vec![1.0; 24]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vec![2.0; 24]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // axes=[1, 1]: explicit duplicate
    let reduce = ReduceMeanLayer::new(vec![1, 1], true);
    let result = reduce.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "duplicate axes [1, 1] must return error, got {:?}",
        result
    );

    // axes=[1, -2]: both resolve to axis 1 on a 3D tensor
    let reduce2 = ReduceMeanLayer::new(vec![1, -2], true);
    let result2 = reduce2.propagate_ibp(&input);
    assert!(
        result2.is_err(),
        "duplicate axes [1, -2] (both resolve to 1) must return error, got {:?}",
        result2
    );
}

/// Regression (#vnncomp-aw-soundness self-audit): ReduceMean CROWN backward multiplies each
/// coefficient by scale = fl(1/n) in f32, which rounds — the stored coeff differs from the
/// true real coeff/n and MUST carry a certified error. mean over 3 elements: each coeff is
/// fl(1/3); the certified [stored-err, stored+err] must enclose the true real 1/3.
#[test]
fn reduce_mean_crown_backward_coeff_err_encloses_real_reciprocal() {
    let pre =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[1, 3])), ArrayD::zeros(IxDyn(&[1, 3]))).unwrap();
    let reduce = ReduceMeanLayer::new(vec![-1], true);
    let result = reduce
        .propagate_linear_with_bounds(&LinearBounds::identity(1), &pre)
        .unwrap();
    let err = result
        .lower_a_err()
        .expect("ReduceMean coeff must carry the *1/n multiply error");
    let stored = result.lower_a[[0, 0]] as f64;
    let e = err[[0, 0]] as f64;
    let true_third = 1.0_f64 / 3.0;
    assert!(e > 0.0, "ReduceMean coeff err must be nonzero");
    assert!(
        stored - e <= true_third && true_third <= stored + e,
        "certified [stored-err, stored+err] must enclose true 1/3: stored={stored} err={e}"
    );
}
