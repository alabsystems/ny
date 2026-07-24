// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32};

// ========== Constructors / Accessors ==========

#[test]
fn test_new_stores_axis() {
    let layer = ConcatLayer::new(1);
    assert_eq!(layer.axis, 1);
    assert!(
        layer.input_shapes.is_none(),
        "new layer should have no input shapes"
    );
    assert!(
        layer.constant_inputs.is_none(),
        "new layer should have no constant inputs"
    );
}

#[test]
fn test_with_input_shapes() {
    let layer = ConcatLayer::with_input_shapes(0, vec![vec![2, 3], vec![2, 5]]);
    assert_eq!(layer.axis, 0);
    assert_eq!(layer.input_shape(0), Some([2, 3].as_slice()));
    assert_eq!(layer.input_shape(1), Some([2, 5].as_slice()));
    assert_eq!(layer.input_shape(2), None);
}

#[test]
fn test_with_constants() {
    let t = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
    )
    .unwrap();
    let layer = ConcatLayer::with_constants(0, vec![vec![2], vec![3]], vec![Some(t), None]);
    assert!(
        layer.constant_input(0).is_some(),
        "input 0 should have a constant"
    );
    assert!(
        layer.constant_input(1).is_none(),
        "input 1 should have no constant"
    );
}

#[test]
fn test_normalize_axis_negative() {
    let layer = ConcatLayer::new(-1);
    assert_eq!(layer.normalize_axis(3).unwrap(), 2);
}

#[test]
fn test_normalize_axis_positive() {
    let layer = ConcatLayer::new(1);
    assert_eq!(layer.normalize_axis(3).unwrap(), 1);
}

#[test]
fn test_normalize_axis_out_of_range() {
    let layer = ConcatLayer::new(5);
    assert!(
        layer.normalize_axis(3).is_err(),
        "axis 5 out of range for ndim=3"
    );
}

#[test]
fn test_normalize_axis_negative_out_of_range() {
    let layer = ConcatLayer::new(-4);
    assert!(
        layer.normalize_axis(3).is_err(),
        "axis -4 out of range for ndim=3"
    );
}

// ========== IBP Binary ==========

#[test]
fn test_ibp_binary_same_shape() {
    // Concat two [3] tensors along axis 0 -> [6]
    let layer = ConcatLayer::new(0);

    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![40.0, 50.0, 60.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp_binary(&a, &b).unwrap();
    assert_eq!(out.shape(), &[6]);
    // First 3 from a, last 3 from b
    assert!(
        (out.lower()[[0]] - 1.0).abs() < 1e-5,
        "lower[0] expected 1.0, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.lower()[[3]] - 10.0).abs() < 1e-5,
        "lower[3] expected 10.0, got {}",
        out.lower()[[3]]
    );
    assert!(
        (out.upper()[[2]] - 6.0).abs() < 1e-5,
        "upper[2] expected 6.0, got {}",
        out.upper()[[2]]
    );
    assert!(
        (out.upper()[[5]] - 60.0).abs() < 1e-5,
        "upper[5] expected 60.0, got {}",
        out.upper()[[5]]
    );
}

#[test]
fn test_ibp_binary_2d_axis1() {
    // Concat [2,3] and [2,2] along axis 1 -> [2,5]
    let layer = ConcatLayer::new(1);

    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![10.0, 20.0, 30.0, 40.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp_binary(&a, &b).unwrap();
    assert_eq!(out.shape(), &[2, 5]);
}

#[test]
fn test_ibp_binary_broadcast_a_missing_batch() {
    // a: [3], b: [2, 3] -> broadcast a to [2,3], concat along axis 1 -> [2, 6]
    let layer = ConcatLayer::new(1);

    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp_binary(&a, &b).unwrap();
    assert_eq!(out.shape(), &[2, 6]);
}

#[test]
fn test_ibp_binary_broadcast_restores_squeezed_feature_axis() {
    // ONNX axis=1 becomes axis=0 after squeezing batch during loading. When IBP
    // temporarily restores batch, the effective concat axis must shift back.
    let layer = ConcatLayer::new(0).with_restored_batch_axis_shift(true);

    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp_binary(&a, &b).unwrap();
    assert_eq!(out.shape(), &[2, 6]);
}

#[test]
fn test_ibp_binary_broadcast_keeps_explicit_batch_axis_without_shift() {
    let layer = ConcatLayer::new(0);

    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp_binary(&a, &b).unwrap();
    assert_eq!(out.shape(), &[4, 3]);
    assert!(
        (out.lower()[[0, 0]] - 1.0).abs() < 1e-5,
        "lower[0,0] expected 1.0, got {}",
        out.lower()[[0, 0]]
    );
    assert!(
        (out.lower()[[2, 0]] - 10.0).abs() < 1e-5,
        "lower[2,0] expected 10.0, got {}",
        out.lower()[[2, 0]]
    );
}

#[test]
fn test_ibp_binary_axis_out_of_bounds() {
    let layer = ConcatLayer::new(5);
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let result = layer.propagate_ibp_binary(&a, &a);
    assert!(result.is_err(), "axis 5 out of bounds for 1D input");
}

#[test]
fn test_ibp_binary_shape_mismatch() {
    // [2,3] and [2,4] along axis 0 should fail (axis-1 dims don't match)
    let layer = ConcatLayer::new(0);
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0; 6]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![2.0; 8]).unwrap(),
    )
    .unwrap();
    let result = layer.propagate_ibp_binary(&a, &b);
    assert!(
        result.is_err(),
        "shape mismatch [2,3] vs [2,4] on axis 0 should error"
    );
}

#[test]
fn test_ibp_binary_soundness() {
    // Verify: for all corners, concat(corner_a, corner_b) lies within output bounds
    let layer = ConcatLayer::new(0);

    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 5.0]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![4.0, 1.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp_binary(&a, &b).unwrap();
    // Output [4], lower from concat of lowers, upper from concat of uppers
    assert!(
        (out.lower()[[0]] - (-1.0)).abs() < 1e-5,
        "lower[0] expected -1.0, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.lower()[[1]] - 2.0).abs() < 1e-5,
        "lower[1] expected 2.0, got {}",
        out.lower()[[1]]
    );
    assert!(
        (out.lower()[[2]] - 0.0).abs() < 1e-5,
        "lower[2] expected 0.0, got {}",
        out.lower()[[2]]
    );
    assert!(
        (out.lower()[[3]] - (-2.0)).abs() < 1e-5,
        "lower[3] expected -2.0, got {}",
        out.lower()[[3]]
    );
    assert!(
        (out.upper()[[0]] - 3.0).abs() < 1e-5,
        "upper[0] expected 3.0, got {}",
        out.upper()[[0]]
    );
    assert!(
        (out.upper()[[1]] - 5.0).abs() < 1e-5,
        "upper[1] expected 5.0, got {}",
        out.upper()[[1]]
    );
    assert!(
        (out.upper()[[2]] - 4.0).abs() < 1e-5,
        "upper[2] expected 4.0, got {}",
        out.upper()[[2]]
    );
    assert!(
        (out.upper()[[3]] - 1.0).abs() < 1e-5,
        "upper[3] expected 1.0, got {}",
        out.upper()[[3]]
    );
}

// ========== IBP N-ary ==========

#[test]
fn test_ibp_nary_three_inputs() {
    let layer = ConcatLayer::new(0);

    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![5.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![6.0]).unwrap(),
    )
    .unwrap();
    let c = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![7.0, 8.0, 9.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 11.0, 12.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp_nary(&[&a, &b, &c]).unwrap();
    assert_eq!(out.shape(), &[6]);
    assert!(
        (out.lower()[[0]] - 1.0).abs() < 1e-5,
        "lower[0] expected 1.0, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.lower()[[2]] - 5.0).abs() < 1e-5,
        "lower[2] expected 5.0, got {}",
        out.lower()[[2]]
    );
    assert!(
        (out.lower()[[3]] - 7.0).abs() < 1e-5,
        "lower[3] expected 7.0, got {}",
        out.lower()[[3]]
    );
    assert!(
        (out.upper()[[5]] - 12.0).abs() < 1e-5,
        "upper[5] expected 12.0, got {}",
        out.upper()[[5]]
    );
}

#[test]
fn test_ibp_nary_mixed_ndim_broadcasts_missing_batch() {
    let layer = ConcatLayer::new(0).with_restored_batch_axis_shift(true);

    let one_d = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
    )
    .unwrap();
    let two_d = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![10.0, 20.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![30.0, 40.0]).unwrap(),
    )
    .unwrap();

    let out = layer
        .propagate_ibp_nary(&[&one_d, &two_d, &one_d, &two_d, &one_d, &two_d])
        .unwrap();
    assert_eq!(out.shape(), &[1, 12]);
    assert!(
        (out.lower()[[0, 0]] - 1.0).abs() < 1e-5,
        "lower[0,0] expected 1.0, got {}",
        out.lower()[[0, 0]]
    );
    assert!(
        (out.lower()[[0, 2]] - 10.0).abs() < 1e-5,
        "lower[0,2] expected 10.0, got {}",
        out.lower()[[0, 2]]
    );
    assert!(
        (out.lower()[[0, 10]] - 10.0).abs() < 1e-5,
        "lower[0,10] expected 10.0, got {}",
        out.lower()[[0, 10]]
    );
    assert!(
        (out.upper()[[0, 1]] - 4.0).abs() < 1e-5,
        "upper[0,1] expected 4.0, got {}",
        out.upper()[[0, 1]]
    );
    assert!(
        (out.upper()[[0, 7]] - 40.0).abs() < 1e-5,
        "upper[0,7] expected 40.0, got {}",
        out.upper()[[0, 7]]
    );
    assert!(
        (out.upper()[[0, 11]] - 40.0).abs() < 1e-5,
        "upper[0,11] expected 40.0, got {}",
        out.upper()[[0, 11]]
    );
}

#[test]
fn test_ibp_nary_single_input_returns_clone() {
    let layer = ConcatLayer::new(0);
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp_nary(&[&a]).unwrap();
    assert_eq!(out.shape(), a.shape());
    assert!(
        (out.lower()[[0]] - 1.0).abs() < 1e-5,
        "lower[0] expected 1.0, got {}",
        out.lower()[[0]]
    );
}

#[test]
fn test_ibp_nary_empty_errors() {
    let layer = ConcatLayer::new(0);
    let result = layer.propagate_ibp_nary(&[]);
    assert!(result.is_err(), "empty input list should error");
}

// ========== CROWN Linear Binary ==========

#[test]
fn test_linear_binary_splits_coefficients() {
    // Input A: size 3, Input B: size 2, total: 5
    // Bounds: 2 outputs, 5 inputs
    let layer = ConcatLayer::new(0);

    let bounds = LinearBounds::new(
        Array2::from_shape_vec(
            (2, 5),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap(),
        Array1::from_vec(vec![0.1, 0.2]),
        Array2::from_shape_vec(
            (2, 5),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        )
        .unwrap(),
        Array1::from_vec(vec![0.3, 0.4]),
    )
    .unwrap();

    let (ba, bb) = layer.propagate_linear_binary(&bounds, &[3], &[2]).unwrap();

    // A gets first 3 columns
    assert_eq!(ba.lower_a.shape(), &[2, 3]);
    assert!(
        (ba.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "ba.lower_a[0,0] expected 1.0, got {}",
        ba.lower_a[[0, 0]]
    );
    assert!(
        (ba.lower_a[[0, 2]] - 3.0).abs() < 1e-5,
        "ba.lower_a[0,2] expected 3.0, got {}",
        ba.lower_a[[0, 2]]
    );
    assert!(
        (ba.lower_a[[1, 0]] - 6.0).abs() < 1e-5,
        "ba.lower_a[1,0] expected 6.0, got {}",
        ba.lower_a[[1, 0]]
    );

    // B gets last 2 columns
    assert_eq!(bb.lower_a.shape(), &[2, 2]);
    assert!(
        (bb.lower_a[[0, 0]] - 4.0).abs() < 1e-5,
        "bb.lower_a[0,0] expected 4.0, got {}",
        bb.lower_a[[0, 0]]
    );
    assert!(
        (bb.lower_a[[0, 1]] - 5.0).abs() < 1e-5,
        "bb.lower_a[0,1] expected 5.0, got {}",
        bb.lower_a[[0, 1]]
    );

    // Bias split evenly (divided by 2)
    assert!(
        (ba.lower_b[0] - 0.05).abs() < 1e-5,
        "ba.lower_b[0] expected 0.05, got {}",
        ba.lower_b[0]
    );
    assert!(
        (bb.lower_b[0] - 0.05).abs() < 1e-5,
        "bb.lower_b[0] expected 0.05, got {}",
        bb.lower_b[0]
    );
}

#[test]
fn test_linear_binary_size_mismatch_errors() {
    let layer = ConcatLayer::new(0);
    let bounds = LinearBounds::new(
        Array2::eye(3),
        Array1::zeros(3),
        Array2::eye(3),
        Array1::zeros(3),
    )
    .unwrap();
    // Total input = 3, but we claim 2+2=4
    let result = layer.propagate_linear_binary(&bounds, &[2], &[2]);
    assert!(
        result.is_err(),
        "size mismatch (total=3, claimed=4) should error"
    );
}

// ========== CROWN Batched Linear Binary ==========

#[test]
fn test_linear_batched_binary_splits_coefficients_and_bias() {
    // Input A: size 3, Input B: size 2, total: 5
    // Batch dim = 1, output dim = 2, input dim = 5 → shape [2, 5]
    let layer = ConcatLayer::new(0);

    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 5]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 5]),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        )
        .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.3, 0.4]).unwrap(),
        vec![5],
        vec![2],
    );

    let (ba, bb) = layer
        .propagate_linear_batched_binary(&bounds, &[3], &[2])
        .unwrap();

    // A gets first 3 columns
    assert_eq!(ba.lower_a.shape(), &[2, 3]);
    assert!(
        (ba.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "batched ba.lower_a[0,0] expected 1.0, got {}",
        ba.lower_a[[0, 0]]
    );
    assert!(
        (ba.lower_a[[0, 2]] - 3.0).abs() < 1e-5,
        "batched ba.lower_a[0,2] expected 3.0, got {}",
        ba.lower_a[[0, 2]]
    );
    assert!(
        (ba.lower_a[[1, 0]] - 6.0).abs() < 1e-5,
        "batched ba.lower_a[1,0] expected 6.0, got {}",
        ba.lower_a[[1, 0]]
    );

    // B gets last 2 columns
    assert_eq!(bb.lower_a.shape(), &[2, 2]);
    assert!(
        (bb.lower_a[[0, 0]] - 4.0).abs() < 1e-5,
        "batched bb.lower_a[0,0] expected 4.0, got {}",
        bb.lower_a[[0, 0]]
    );
    assert!(
        (bb.lower_a[[0, 1]] - 5.0).abs() < 1e-5,
        "batched bb.lower_a[0,1] expected 5.0, got {}",
        bb.lower_a[[0, 1]]
    );
}

#[test]
fn test_linear_batched_binary_bias_uses_directed_rounding() {
    let layer = ConcatLayer::new(0);

    // Choose bias values where plain f32 halving would be unsound after re-sum.
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0e-4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![3.0, 4.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.7e-3]).unwrap(),
        vec![2],
        vec![1],
    );

    let (ba, bb) = layer
        .propagate_linear_batched_binary(&bounds, &[1], &[1])
        .unwrap();

    let expected_lower = next_down_f32(bounds.lower_b[[0]] * 0.5);
    let expected_upper = next_up_f32(bounds.upper_b[[0]] * 0.5);

    assert_eq!(ba.lower_b[[0]], expected_lower);
    assert_eq!(bb.lower_b[[0]], expected_lower);
    assert_eq!(ba.upper_b[[0]], expected_upper);
    assert_eq!(bb.upper_b[[0]], expected_upper);

    // Conservativeness: re-summed halves must be conservative
    let sum_lower = ba.lower_b[[0]] + bb.lower_b[[0]];
    let sum_upper = ba.upper_b[[0]] + bb.upper_b[[0]];
    assert!(
        sum_lower <= bounds.lower_b[[0]],
        "batched split lower bias must remain conservative: sum {sum_lower} > original {}",
        bounds.lower_b[[0]]
    );
    assert!(
        sum_upper >= bounds.upper_b[[0]],
        "batched split upper bias must remain conservative: sum {sum_upper} < original {}",
        bounds.upper_b[[0]]
    );
}

#[test]
fn test_linear_batched_binary_size_mismatch_errors() {
    let layer = ConcatLayer::new(0);
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        vec![3],
        vec![1],
    );
    // Total input = 3, but we claim 2+2=4
    let result = layer.propagate_linear_batched_binary(&bounds, &[2], &[2]);
    assert!(
        result.is_err(),
        "batched size mismatch (total=3, claimed=4) should error"
    );
}

// ========== CROWN Binary Directed Rounding ==========

#[test]
fn test_linear_binary_bias_uses_directed_rounding() {
    let layer = ConcatLayer::new(0);

    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap(),
        Array1::from_vec(vec![1.0e-4]),
        Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap(),
        Array1::from_vec(vec![1.7e-3]),
    )
    .unwrap();

    let (ba, bb) = layer.propagate_linear_binary(&bounds, &[1], &[1]).unwrap();

    let expected_lower = next_down_f32(bounds.lower_b[0] * 0.5);
    let expected_upper = next_up_f32(bounds.upper_b[0] * 0.5);

    assert_eq!(ba.lower_b[0], expected_lower);
    assert_eq!(bb.lower_b[0], expected_lower);
    assert_eq!(ba.upper_b[0], expected_upper);
    assert_eq!(bb.upper_b[0], expected_upper);

    // Conservativeness: re-summed halves must be conservative
    let sum_lower = ba.lower_b[0] + bb.lower_b[0];
    let sum_upper = ba.upper_b[0] + bb.upper_b[0];
    assert!(
        sum_lower <= bounds.lower_b[0],
        "binary split lower bias must remain conservative: sum {sum_lower} > original {}",
        bounds.lower_b[0]
    );
    assert!(
        sum_upper >= bounds.upper_b[0],
        "binary split upper bias must remain conservative: sum {sum_upper} < original {}",
        bounds.upper_b[0]
    );
}

// ========== CROWN Linear N-ary ==========

#[test]
fn test_linear_nary_three_way_split() {
    let layer = ConcatLayer::new(0);

    // 1 output, 6 inputs (split as 2+1+3)
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 6), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        Array1::from_vec(vec![0.9]),
        Array2::from_shape_vec((1, 6), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap(),
        Array1::from_vec(vec![1.5]),
    )
    .unwrap();

    let parts = layer
        .propagate_linear_nary(&bounds, &[vec![2], vec![1], vec![3]])
        .unwrap();

    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0].lower_a.shape(), &[1, 2]);
    assert_eq!(parts[1].lower_a.shape(), &[1, 1]);
    assert_eq!(parts[2].lower_a.shape(), &[1, 3]);

    assert!(
        (parts[0].lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "parts[0].lower_a[0,0] expected 1.0, got {}",
        parts[0].lower_a[[0, 0]]
    );
    assert!(
        (parts[1].lower_a[[0, 0]] - 3.0).abs() < 1e-5,
        "parts[1].lower_a[0,0] expected 3.0, got {}",
        parts[1].lower_a[[0, 0]]
    );
    assert!(
        (parts[2].lower_a[[0, 0]] - 4.0).abs() < 1e-5,
        "parts[2].lower_a[0,0] expected 4.0, got {}",
        parts[2].lower_a[[0, 0]]
    );
    assert!(
        (parts[2].lower_a[[0, 2]] - 6.0).abs() < 1e-5,
        "parts[2].lower_a[0,2] expected 6.0, got {}",
        parts[2].lower_a[[0, 2]]
    );

    // Bias split by 3
    assert!(
        (parts[0].lower_b[0] - 0.3).abs() < 1e-5,
        "parts[0].lower_b[0] expected 0.3, got {}",
        parts[0].lower_b[0]
    );
    assert!(
        (parts[1].lower_b[0] - 0.3).abs() < 1e-5,
        "parts[1].lower_b[0] expected 0.3, got {}",
        parts[1].lower_b[0]
    );
    assert!(
        (parts[2].lower_b[0] - 0.3).abs() < 1e-5,
        "parts[2].lower_b[0] expected 0.3, got {}",
        parts[2].lower_b[0]
    );
}

#[test]
fn test_linear_nary_bias_split_uses_directed_rounding_and_is_conservative() {
    let layer = ConcatLayer::new(0);

    // Values selected so plain f32 division by 3 is unsound:
    // lower parts re-sum above the original lower bias, and upper parts re-sum
    // below the original upper bias.
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap(),
        Array1::from_vec(vec![1.0e-4]),
        Array2::from_shape_vec((1, 3), vec![4.0, 5.0, 6.0]).unwrap(),
        Array1::from_vec(vec![1.7e-3]),
    )
    .unwrap();

    let parts = layer
        .propagate_linear_nary(&bounds, &[vec![1], vec![1], vec![1]])
        .unwrap();
    assert_eq!(parts.len(), 3);

    let divisor = 3.0_f32;
    let expected_lower = next_down_f32(bounds.lower_b[0] / divisor);
    let expected_upper = next_up_f32(bounds.upper_b[0] / divisor);

    for part in &parts {
        assert_eq!(part.lower_b[0], expected_lower);
        assert_eq!(part.upper_b[0], expected_upper);
    }

    let accumulated_lower: f32 = parts.iter().map(|part| part.lower_b[0]).sum();
    let accumulated_upper: f32 = parts.iter().map(|part| part.upper_b[0]).sum();

    assert!(
        accumulated_lower <= bounds.lower_b[0],
        "split lower bias must remain conservative after re-accumulation"
    );
    assert!(
        accumulated_upper >= bounds.upper_b[0],
        "split upper bias must remain conservative after re-accumulation"
    );
}

#[test]
fn test_linear_nary_empty_errors() {
    let layer = ConcatLayer::new(0);
    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_linear_nary(&bounds, &[]);
    assert!(result.is_err(), "empty shape list should error");
}

#[test]
fn test_linear_nary_size_mismatch_errors() {
    let layer = ConcatLayer::new(0);
    let bounds = LinearBounds::new(
        Array2::eye(3),
        Array1::zeros(3),
        Array2::eye(3),
        Array1::zeros(3),
    )
    .unwrap();
    // Total = 3, but shapes sum to 5
    let result = layer.propagate_linear_nary(&bounds, &[vec![2], vec![3]]);
    assert!(
        result.is_err(),
        "nary size mismatch (total=3, claimed=5) should error"
    );
}
