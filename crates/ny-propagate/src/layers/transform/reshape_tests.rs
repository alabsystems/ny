// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for ReshapeLayer and FlattenLayer (extracted from reshape.rs for file-size limit).

use super::{FlattenLayer, ReshapeLayer};
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;
use ndarray::{ArrayD, IxDyn};
use ny_core::{reshape_copy_axis_sentinel, NyError};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

fn make_bounded(lower: ArrayD<f32>, upper: ArrayD<f32>) -> BoundedTensor {
    BoundedTensor::new(lower, upper).unwrap()
}

// =========================================================================
// ReshapeLayer::compute_output_shape
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_reshape_explicit_shape() {
    let layer = ReshapeLayer::new(vec![2, 3]);
    let result = layer.compute_output_shape(&[6]).unwrap();
    assert_eq!(result, vec![2, 3]);
}

#[ntest::timeout(5000)]
#[test]
fn test_reshape_infer_dimension() {
    let layer = ReshapeLayer::new(vec![2, -1]);
    let result = layer.compute_output_shape(&[2, 3]).unwrap();
    assert_eq!(result, vec![2, 3]);
}

#[ntest::timeout(5000)]
#[test]
fn test_reshape_infer_with_larger_tensor() {
    let layer = ReshapeLayer::new(vec![3, -1, 2]);
    let result = layer.compute_output_shape(&[12]).unwrap();
    assert_eq!(result, vec![3, 2, 2]);
}

#[ntest::timeout(5000)]
#[test]
fn test_reshape_dim_zero_copies_input() {
    // dim=0 means copy from input
    let layer = ReshapeLayer::new(vec![0, -1]);
    let result = layer.compute_output_shape(&[4, 3]).unwrap();
    assert_eq!(result, vec![4, 3]);
}

#[ntest::timeout(5000)]
#[test]
fn test_reshape_copy_axis_sentinel_copies_moved_input_axis() {
    let copy_axis_2 = reshape_copy_axis_sentinel(2).expect("axis in range");
    let layer = ReshapeLayer::new(vec![-1, copy_axis_2, 128]);
    let result = layer.compute_output_shape(&[8, 2, 16, 128]).unwrap();
    assert_eq!(result, vec![16, 16, 128]);
}

#[ntest::timeout(5000)]
#[test]
fn test_reshape_two_inferred_dims_error() {
    let layer = ReshapeLayer::new(vec![-1, -1]);
    assert!(layer.compute_output_shape(&[6]).is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_reshape_element_count_mismatch_error() {
    let layer = ReshapeLayer::new(vec![2, 5]);
    assert!(layer.compute_output_shape(&[6]).is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_reshape_dim_zero_out_of_range_error() {
    let layer = ReshapeLayer::new(vec![0, 0, -1]);
    // Input only has 1 dim, but target has dim=0 at index 1
    assert!(layer.compute_output_shape(&[6]).is_err());
}

// =========================================================================
// ReshapeLayer IBP
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_reshape_ibp_preserves_values() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).unwrap();
    let input = make_bounded(lower, upper);

    let layer = ReshapeLayer::new(vec![3, 2]);
    let result = layer.propagate_ibp(&input).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    // Values should be the same, just rearranged
    let lower_flat: Vec<f32> = result.lower().iter().copied().collect();
    assert_eq!(lower_flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[ntest::timeout(5000)]
#[test]
fn test_reshape_ibp_flatten_to_1d() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let input = make_bounded(lower, upper);

    let layer = ReshapeLayer::new(vec![-1]);
    let result = layer.propagate_ibp(&input).unwrap();

    assert_eq!(result.shape(), &[4]);
}

// =========================================================================
// ReshapeLayer CROWN backward
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_reshape_propagate_linear_borrows() {
    let layer = ReshapeLayer::new(vec![2, 3]);
    let bounds = LinearBounds::new(
        ndarray::Array2::eye(6),
        ndarray::Array1::zeros(6),
        ndarray::Array2::eye(6),
        ndarray::Array1::zeros(6),
    )
    .unwrap();
    // Reshape should return Borrowed (identity in coefficient space)
    let result = layer.propagate_linear(&bounds).unwrap();
    assert!(matches!(result, Cow::Borrowed(_)));
}

// =========================================================================
// FlattenLayer::compute_output_shape
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_flatten_axis1_default() {
    let layer = FlattenLayer::new(1);
    let result = layer.compute_output_shape(&[2, 3, 4]).unwrap();
    // axis=1: (2, 3*4) = (2, 12)
    assert_eq!(result, vec![2, 12]);
}

#[ntest::timeout(5000)]
#[test]
fn test_flatten_axis0_prepends_one() {
    let layer = FlattenLayer::flatten_all();
    let result = layer.compute_output_shape(&[2, 3, 4]).unwrap();
    // axis=0: (1, 2*3*4) = (1, 24)
    assert_eq!(result, vec![1, 24]);
}

#[ntest::timeout(5000)]
#[test]
fn test_flatten_axis_equals_ndim() {
    let layer = FlattenLayer::new(3);
    let result = layer.compute_output_shape(&[2, 3, 4]).unwrap();
    // axis=3 = ndim: (2*3*4, 1) = (24, 1)
    assert_eq!(result, vec![24, 1]);
}

#[ntest::timeout(5000)]
#[test]
fn test_flatten_negative_axis() {
    let layer = FlattenLayer::new(-1);
    let result = layer.compute_output_shape(&[2, 3, 4]).unwrap();
    // -1 resolves to axis=2: (2*3, 4) = (6, 4)
    assert_eq!(result, vec![6, 4]);
}

#[ntest::timeout(5000)]
#[test]
fn test_flatten_axis_out_of_range() {
    let layer = FlattenLayer::new(5);
    assert!(layer.compute_output_shape(&[2, 3]).is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_flatten_negative_axis_out_of_range() {
    let layer = FlattenLayer::new(-4);
    assert!(layer.compute_output_shape(&[2, 3]).is_err());
}

#[ntest::timeout(5000)]
#[test]
fn test_flatten_zero_dim_input_error() {
    let layer = FlattenLayer::new(0);
    assert!(layer.compute_output_shape(&[]).is_err());
}

// =========================================================================
// FlattenLayer IBP
// =========================================================================

#[ntest::timeout(5000)]
#[test]
fn test_flatten_ibp_preserves_values() {
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), (1..=12).map(|x| x as f32).collect()).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), (11..=22).map(|x| x as f32).collect()).unwrap();
    let input = make_bounded(lower.clone(), upper);

    let layer = FlattenLayer::new(1);
    let result = layer.propagate_ibp(&input).unwrap();

    assert_eq!(result.shape(), &[2, 6]);
    // All values should be preserved
    let lower_flat: Vec<f32> = result.lower().iter().copied().collect();
    let orig_flat: Vec<f32> = lower.iter().copied().collect();
    assert_eq!(lower_flat, orig_flat);
}

#[ntest::timeout(5000)]
#[test]
fn test_flatten_propagate_linear_borrows() {
    let layer = FlattenLayer::new(1);
    let bounds = LinearBounds::new(
        ndarray::Array2::eye(6),
        ndarray::Array1::zeros(6),
        ndarray::Array2::eye(6),
        ndarray::Array1::zeros(6),
    )
    .unwrap();
    let result = layer.propagate_linear(&bounds).unwrap();
    assert!(matches!(result, Cow::Borrowed(_)));
}

/// Regression test for #2911: negative dims (other than -1, 0) should error.
#[ntest::timeout(5000)]
#[test]
fn test_reshape_negative_dim_error_2911() {
    // -2 is not a valid ONNX reshape dimension
    let layer = ReshapeLayer::new(vec![-2, 3]);
    let err = layer
        .compute_output_shape(&[6])
        .expect_err("negative dim -2 should be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "negative dim -2 should be InvalidSpec, got: {err:?}"
    );

    // -100 should also be rejected
    let layer2 = ReshapeLayer::new(vec![2, -100]);
    let err = layer2
        .compute_output_shape(&[6])
        .expect_err("negative dim -100 should be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "negative dim -100 should be InvalidSpec, got: {err:?}"
    );
}
