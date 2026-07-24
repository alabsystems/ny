// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for shape transformation layers (Reshape, Transpose, Tile, Slice, Flatten, etc.).

use super::*;
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use std::borrow::Cow;

// ==================== GatherLayer Tests ====================

#[ntest::timeout(5000)]
#[test]
fn expand_like_last_axis_ibp_repeats_last_axis() {
    let layer = ExpandLikeLastAxisLayer::new();
    let source = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![2.0, 4.0]).unwrap(),
    )
    .unwrap();
    let reference =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[2, 3])), ArrayD::zeros(IxDyn(&[2, 3]))).unwrap();

    let output = layer.propagate_ibp_binary(&source, &reference).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
    );
    assert_eq!(
        output.upper().as_slice().unwrap(),
        &[2.0, 2.0, 2.0, 4.0, 4.0, 4.0]
    );
}

#[ntest::timeout(5000)]
#[test]
fn expand_like_last_axis_crown_sums_replica_coefficients() {
    let layer = ExpandLikeLastAxisLayer::new();
    let source =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[2, 1])), ArrayD::zeros(IxDyn(&[2, 1]))).unwrap();
    let reference =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[2, 3])), ArrayD::zeros(IxDyn(&[2, 3]))).unwrap();
    let bounds = LinearBounds::new_or_conservative(
        Array2::from_shape_vec((1, 6), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ndarray::arr1(&[0.25]),
        Array2::from_shape_vec((1, 6), vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap(),
        ndarray::arr1(&[0.75]),
    )
    .unwrap();

    let (source_bounds, reference_bounds) = layer
        .propagate_linear_binary(&bounds, &source, &reference)
        .unwrap();

    assert_eq!(source_bounds.lower_a().as_slice().unwrap(), &[6.0, 15.0]);
    assert_eq!(source_bounds.upper_a().as_slice().unwrap(), &[15.0, 6.0]);
    assert_eq!(source_bounds.lower_b().as_slice().unwrap(), &[0.25]);
    assert_eq!(source_bounds.upper_b().as_slice().unwrap(), &[0.75]);
    assert!(
        reference_bounds.lower_a().iter().all(|&v| v == 0.0)
            && reference_bounds.upper_a().iter().all(|&v| v == 0.0),
        "reference input should carry zero coefficients"
    );
}

#[ntest::timeout(5000)]
#[test]
fn gather_runtime_last_axis_len_returns_exact_scalar() {
    let layer = GatherLayer::runtime_last_axis_len(vec![]);
    let input =
        BoundedTensor::new(ArrayD::zeros(IxDyn(&[4, 7])), ArrayD::zeros(IxDyn(&[4, 7]))).unwrap();

    let output = layer
        .propagate_ibp(&input)
        .expect("runtime last-axis query should succeed");

    assert_eq!(output.shape(), &[] as &[usize]);
    assert_eq!(output.lower().as_slice_memory_order(), Some(&[7.0][..]));
    assert_eq!(output.upper().as_slice_memory_order(), Some(&[7.0][..]));
}

/// Gather with static indices, axis 0: select specific rows.
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_static_indices_axis0() {
    // Input shape: [3, 2], indices: [0, 2] -> output shape: [2, 2]
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 2]).unwrap();
    let gather = GatherLayer::new(0, Some(indices), vec![]);
    let output = gather.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // Row 0: [1,2] / [2,3]; Row 2: [5,6] / [6,7]
    assert_eq!(output.lower().as_slice().unwrap(), &[1.0, 2.0, 5.0, 6.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[2.0, 3.0, 6.0, 7.0]);
}

/// Gather axis 1, contiguous indices (formerly hit reshape bug #1724, now fixed).
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_axis1_contiguous() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 1]).unwrap();
    let gather = GatherLayer::new(1, Some(indices), vec![]);
    let output = gather.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // Row 0: cols [0,1] -> [1,2] / [2,3]; Row 1: cols [0,1] -> [4,5] / [5,6]
    assert_eq!(output.lower().as_slice().unwrap(), &[1.0, 2.0, 4.0, 5.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[2.0, 3.0, 5.0, 6.0]);
}

/// Gather axis 1 with reordered indices (formerly hit reshape bug #1724, now fixed).
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_axis1_reordered() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Reversed indices [2, 0] on axis 1
    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2_i64, 0]).unwrap();
    let gather = GatherLayer::new(1, Some(indices), vec![]);
    let output = gather.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // Row 0: cols [2,0] -> [3,1] / [4,2]; Row 1: cols [2,0] -> [6,4] / [7,5]
    assert_eq!(output.lower().as_slice().unwrap(), &[3.0, 1.0, 6.0, 4.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[4.0, 2.0, 7.0, 5.0]);
}

/// Gather with negative indices.
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_negative_indices() {
    // Input shape: [3, 2], indices: [-1] -> select last row
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1_i64]).unwrap();
    let gather = GatherLayer::new(0, Some(indices), vec![]);
    let output = gather.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2]);
    assert_eq!(output.lower().as_slice().unwrap(), &[5.0, 6.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[6.0, 7.0]);
}

/// Gather with scalar index removes the gather axis.
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_scalar_index() {
    // Input shape: [3, 2], scalar index 1 on axis 0 -> output shape: [2]
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // Scalar (0-D) index
    let indices = ArrayD::from_shape_vec(IxDyn(&[]), vec![1_i64]).unwrap();
    let gather = GatherLayer::new(0, Some(indices), vec![]);
    let output = gather.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2]);
    assert_eq!(output.lower().as_slice().unwrap(), &[3.0, 4.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[4.0, 5.0]);
}

/// Gather with dynamic indices uses conservative min/max bounds.
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_dynamic_indices_conservative() {
    // Input shape: [3, 2], dynamic indices shape [2] on axis 0
    // Should take min(lower) and max(upper) along axis 0
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    // No indices (dynamic), but we know the shape
    let gather = GatherLayer::new(0, None, vec![2]);
    let output = gather.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // min of lower along axis 0: [min(1,3,5), min(2,4,6)] = [1,2]
    // max of upper along axis 0: [max(2,4,6), max(3,5,7)] = [6,7]
    // broadcast to [2, 2]
    assert_eq!(output.lower().as_slice().unwrap(), &[1.0, 2.0, 1.0, 2.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[6.0, 7.0, 6.0, 7.0]);
}

/// Gather rejects out-of-range indices.
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_rejects_out_of_range_index() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![5_i64]).unwrap();
    let gather = GatherLayer::new(0, Some(indices), vec![]);
    let err = gather.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{err}").contains("out of bounds"),
        "Expected out-of-bounds error, got: {err}"
    );
}

/// Gather rejects out-of-bounds axis.
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_rejects_out_of_bounds_axis() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0_i64]).unwrap();
    let gather = GatherLayer::new(5, Some(indices), vec![]);
    let err = gather.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{err}").contains("out of range"),
        "Expected axis out-of-range error, got: {err}"
    );
}

/// Gather with negative axis (-2 on a 2D tensor = axis 0).
#[ntest::timeout(5000)]
#[test]
fn gather_ibp_negative_axis() {
    // Input [3, 2], axis -2 (=0), indices [0, 2]
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let indices = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0_i64, 2]).unwrap();
    let gather = GatherLayer::new(-2, Some(indices), vec![]);
    let output = gather.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // Rows 0 and 2: [1,2,5,6] / [2,3,6,7]
    assert_eq!(output.lower().as_slice().unwrap(), &[1.0, 2.0, 5.0, 6.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[2.0, 3.0, 6.0, 7.0]);
}

// ==================== SliceLayer Tests ====================

/// Slice basic: select a range along axis 0.
#[ntest::timeout(5000)]
#[test]
fn slice_ibp_axis0() {
    // Input shape: [4, 2], slice axis 0, start=1, end=3 -> [2, 2]
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slice = SliceLayer::new(0, 1, 3);
    let output = slice.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // Rows 1-2: [3,4,5,6] / [4,5,6,7]
    assert_eq!(output.lower().as_slice().unwrap(), &[3.0, 4.0, 5.0, 6.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[4.0, 5.0, 6.0, 7.0]);
}

/// Slice along axis 1.
#[ntest::timeout(5000)]
#[test]
fn slice_ibp_axis1() {
    // Input shape: [2, 4], slice axis 1, start=0, end=2 -> [2, 2]
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slice = SliceLayer::new(1, 0, 2);
    let output = slice.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // First 2 cols: [1,2,5,6] / [2,3,6,7]
    assert_eq!(output.lower().as_slice().unwrap(), &[1.0, 2.0, 5.0, 6.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[2.0, 3.0, 6.0, 7.0]);
}

/// Slice with negative axis.
#[ntest::timeout(5000)]
#[test]
fn slice_ibp_negative_axis() {
    // Input shape: [2, 4], axis -1 (=1), start=2, end=4 -> [2, 2]
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 4]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slice = SliceLayer::new(-1, 2, 4);
    let output = slice.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    // Last 2 cols: [3,4,7,8] / [4,5,8,9]
    assert_eq!(output.lower().as_slice().unwrap(), &[3.0, 4.0, 7.0, 8.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[4.0, 5.0, 8.0, 9.0]);
}

/// Slice clamps out-of-range end to axis size (ONNX INT64_MAX sentinel convention).
#[ntest::timeout(5000)]
#[test]
fn slice_ibp_clamps_out_of_range_end() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slice = SliceLayer::new(0, 0, 5); // end=5 > axis_size=3 → clamped to 3
    let result = slice.propagate_ibp(&input).unwrap();
    assert_eq!(result.shape(), &[3, 2]); // whole axis slice
}

/// Slice rejects empty range (start >= end).
#[ntest::timeout(5000)]
#[test]
fn slice_ibp_rejects_empty_range() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slice = SliceLayer::new(0, 2, 2); // start == end
    let err = slice.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{err}").contains("empty after clamping"),
        "Expected empty-slice error after clamping, got: {err}"
    );
}

/// Slice rejects out-of-bounds axis.
#[ntest::timeout(5000)]
#[test]
fn slice_ibp_rejects_out_of_bounds_axis() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0; 6]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slice = SliceLayer::new(5, 0, 1);
    let err = slice.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{err}").contains("out of range"),
        "Expected axis out-of-range error, got: {err}"
    );
}

/// Slice preserves bound relationship (lower <= upper element-wise).
#[ntest::timeout(5000)]
#[test]
fn slice_ibp_preserves_bound_ordering() {
    // 3D tensor: [2, 3, 4]
    let n = 2 * 3 * 4;
    let lower_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let upper_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), lower_data).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), upper_data).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let slice = SliceLayer::new(1, 1, 3);
    let output = slice.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 2, 4]);
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "Bound ordering violated: {} > {}", l, u);
    }
}

// ==================== SqueezeLayer Tests ====================

#[ntest::timeout(5000)]
#[test]
fn squeeze_ibp_removes_axis_of_size_one() {
    // Input shape: [2, 1, 3] -> Squeeze axis 1 -> Output shape: [2, 3]
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    let squeeze = SqueezeLayer::new(1);
    let output = squeeze.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    // Values should be unchanged, just reshape
    assert_eq!(
        output.lower().iter().collect::<Vec<_>>(),
        lower.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        output.upper().iter().collect::<Vec<_>>(),
        upper.iter().collect::<Vec<_>>()
    );
}

#[ntest::timeout(5000)]
#[test]
fn squeeze_ibp_negative_axis() {
    // Input shape: [2, 3, 1] -> Squeeze axis -1 -> Output shape: [2, 3]
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 1]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 1]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let squeeze = SqueezeLayer::new(-1);
    let output = squeeze.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
}

#[ntest::timeout(5000)]
#[test]
fn squeeze_ibp_rejects_axis_zero() {
    // Squeezing batch dimension (axis 0) is forbidden per alpha-beta-CROWN design
    let lower = ArrayD::from_elem(IxDyn(&[1, 2, 3]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2, 3]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let squeeze = SqueezeLayer::new(0);
    let result = squeeze.propagate_ibp(&input);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{:?}", err).contains("axis 0 forbidden"),
        "Expected 'axis 0 forbidden' error, got: {:?}",
        err
    );
}

#[ntest::timeout(5000)]
#[test]
fn squeeze_ibp_rejects_non_unit_dimension() {
    // Cannot squeeze axis with size != 1
    let lower = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3, 4]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let squeeze = SqueezeLayer::new(1);
    let result = squeeze.propagate_ibp(&input);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        format!("{:?}", err).contains("expected 1"),
        "Expected 'expected 1' error, got: {:?}",
        err
    );
}

#[ntest::timeout(5000)]
#[test]
fn squeeze_ibp_out_of_bounds_axis() {
    let lower = ArrayD::from_elem(IxDyn(&[2, 1, 3]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 1, 3]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let squeeze = SqueezeLayer::new(5); // Out of bounds
    let result = squeeze.propagate_ibp(&input);

    assert!(result.is_err());
}

#[ntest::timeout(5000)]
#[test]
fn squeeze_linear_passthrough() {
    // CROWN backward pass is pass-through (element count unchanged)
    let bounds = LinearBounds::identity(6);
    let squeeze = SqueezeLayer::new(1);
    let output = squeeze.propagate_linear(&bounds).unwrap();

    // Should be borrowed (unchanged)
    assert!(matches!(output, Cow::Borrowed(_)));
}

// ==================== UnsqueezeLayer Tests ====================

#[ntest::timeout(5000)]
#[test]
fn unsqueeze_ibp_inserts_axis_of_size_one() {
    // Input shape: [2, 3] -> Unsqueeze axis 1 -> Output shape: [2, 1, 3]
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    let unsqueeze = UnsqueezeLayer::new(1);
    let output = unsqueeze.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 1, 3]);
    // Values should be unchanged, just reshape
    assert_eq!(
        output.lower().iter().collect::<Vec<_>>(),
        lower.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        output.upper().iter().collect::<Vec<_>>(),
        upper.iter().collect::<Vec<_>>()
    );
}

#[ntest::timeout(5000)]
#[test]
fn unsqueeze_ibp_negative_axis() {
    // Input shape: [2, 3] -> Unsqueeze axis -1 -> Output shape: [2, 3, 1]
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let unsqueeze = UnsqueezeLayer::new(-1);
    let output = unsqueeze.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 3, 1]);
}

#[ntest::timeout(5000)]
#[test]
fn unsqueeze_ibp_axis_zero_inserts_leading_dim() {
    // Unsqueeze at axis 0 inserts a leading dimension (used by lsnc quadrotor2d_output)
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    let unsqueeze = UnsqueezeLayer::new(0);
    let output = unsqueeze.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[1, 2, 3]);
    // Values should be unchanged, just reshape
    assert_eq!(
        output.lower().iter().collect::<Vec<_>>(),
        lower.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        output.upper().iter().collect::<Vec<_>>(),
        upper.iter().collect::<Vec<_>>()
    );
}

#[ntest::timeout(5000)]
#[test]
fn unsqueeze_ibp_out_of_bounds_axis() {
    let lower = ArrayD::from_elem(IxDyn(&[2, 3]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let unsqueeze = UnsqueezeLayer::new(5); // Out of bounds for output_ndim=3
    let result = unsqueeze.propagate_ibp(&input);

    assert!(result.is_err());
}

#[ntest::timeout(5000)]
#[test]
fn unsqueeze_linear_passthrough() {
    // CROWN backward pass is pass-through (element count unchanged)
    let bounds = LinearBounds::identity(6);
    let unsqueeze = UnsqueezeLayer::new(1);
    let output = unsqueeze.propagate_linear(&bounds).unwrap();

    // Should be borrowed (unchanged)
    assert!(matches!(output, Cow::Borrowed(_)));
}

// ==================== Squeeze-Unsqueeze Roundtrip Tests ====================

#[ntest::timeout(5000)]
#[test]
fn squeeze_unsqueeze_roundtrip() {
    // Squeeze then Unsqueeze should give back original shape
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let original = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Squeeze axis 1: [2, 1, 3] -> [2, 3]
    let squeeze = SqueezeLayer::new(1);
    let squeezed = squeeze.propagate_ibp(&original).unwrap();
    assert_eq!(squeezed.shape(), &[2, 3]);

    // Unsqueeze axis 1: [2, 3] -> [2, 1, 3]
    let unsqueeze = UnsqueezeLayer::new(1);
    let restored = unsqueeze.propagate_ibp(&squeezed).unwrap();
    assert_eq!(restored.shape(), &[2, 1, 3]);

    // Values should be preserved
    assert_eq!(
        restored.lower().iter().collect::<Vec<_>>(),
        lower.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        restored.upper().iter().collect::<Vec<_>>(),
        upper.iter().collect::<Vec<_>>()
    );
}

#[ntest::timeout(5000)]
#[test]
fn unsqueeze_squeeze_roundtrip() {
    // Unsqueeze then Squeeze should give back original shape
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap();
    let original = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();

    // Unsqueeze axis 1: [2, 3] -> [2, 1, 3]
    let unsqueeze = UnsqueezeLayer::new(1);
    let unsqueezed = unsqueeze.propagate_ibp(&original).unwrap();
    assert_eq!(unsqueezed.shape(), &[2, 1, 3]);

    // Squeeze axis 1: [2, 1, 3] -> [2, 3]
    let squeeze = SqueezeLayer::new(1);
    let restored = squeeze.propagate_ibp(&unsqueezed).unwrap();
    assert_eq!(restored.shape(), &[2, 3]);

    // Values should be preserved
    assert_eq!(
        restored.lower().iter().collect::<Vec<_>>(),
        lower.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        restored.upper().iter().collect::<Vec<_>>(),
        upper.iter().collect::<Vec<_>>()
    );
}

// ==================== TransposeLayer CROWN Backward Tests ====================

/// TransposeLayer CROWN backward with identity bounds produces a permutation matrix.
///
/// For a 2x3 input transposed with axes [1,0] (swap), the CROWN backward pass
/// should permute the identity matrix columns. If x has shape [2,3] = 6 elements
/// and y = transpose(x) has shape [3,2] = 6 elements, then for identity bounds
/// I @ y, the result should be a permutation matrix P where P @ x = y.
#[ntest::timeout(5000)]
#[test]
fn transpose_crown_backward_permutes_identity() {
    // Input shape: [2, 3] -> Transpose axes [1, 0] -> Output shape: [3, 2]
    // Flat mapping: input[r,c] at index r*3+c maps to output[c,r] at index c*2+r
    let mut layer = TransposeLayer::new(vec![1, 0]);
    layer.set_input_shape(vec![2, 3]);

    // Identity bounds: 6x6 identity (one output per element)
    let bounds = LinearBounds::identity(6);
    let result = layer.propagate_linear(&bounds).unwrap();

    // The result should be a permutation matrix.
    // For transpose [1,0] on [2,3]:
    //   output flat 0 = [0,0] -> input [0,0] = flat 0
    //   output flat 1 = [0,1] -> input [1,0] = flat 3
    //   output flat 2 = [1,0] -> input [0,1] = flat 1
    //   output flat 3 = [1,1] -> input [1,1] = flat 4
    //   output flat 4 = [2,0] -> input [0,2] = flat 2
    //   output flat 5 = [2,1] -> input [1,2] = flat 5
    // So column i of the result should have a 1 at the row corresponding
    // to the inverse mapping.
    let result_ref = result.as_ref();
    let la = &result_ref.lower_a;

    // Each row should have exactly one 1.0 and rest 0.0 (permutation matrix)
    for row in 0..6 {
        let mut ones = 0;
        for col in 0..6 {
            let val = la[[row, col]];
            assert!(
                val == 0.0 || val == 1.0,
                "Expected 0 or 1 at [{row}, {col}], got {val}"
            );
            if val == 1.0 {
                ones += 1;
            }
        }
        assert_eq!(ones, 1, "Row {row} should have exactly one 1.0");
    }

    // Verify specific permutation: row 0 of output (= output flat 0) maps to
    // input flat 0 (identity row 0, column 0 should be 1).
    // After CROWN backward, result[row, col] means: output_row depends on input_col.
    // For identity bounds through transpose:
    //   result[0, 0] = 1 (output 0 = input 0)
    //   result[1, 3] = 1 (output 1 = input 3)
    //   result[2, 1] = 1 (output 2 = input 1)
    //   result[3, 4] = 1 (output 3 = input 4)
    //   result[4, 2] = 1 (output 4 = input 2)
    //   result[5, 5] = 1 (output 5 = input 5)
    assert_eq!(la[[0, 0]], 1.0, "output[0] = input[0]");
    assert_eq!(la[[1, 3]], 1.0, "output[1] = input[3]");
    assert_eq!(la[[2, 1]], 1.0, "output[2] = input[1]");
    assert_eq!(la[[3, 4]], 1.0, "output[3] = input[4]");
    assert_eq!(la[[4, 2]], 1.0, "output[4] = input[2]");
    assert_eq!(la[[5, 5]], 1.0, "output[5] = input[5]");
}

/// CROWN backward with non-identity bounds correctly permutes columns.
///
/// Uses a specific coefficient matrix to verify that column permutation
/// produces the correct output for non-trivial bounds.
#[ntest::timeout(5000)]
#[test]
fn transpose_crown_backward_permutes_columns() {
    use ndarray::Array1;

    // Input shape: [2, 2] -> Transpose axes [1, 0] -> Output shape: [2, 2]
    // Flat mapping for [2,2] with perm [1,0]:
    //   output[0,0] = input[0,0] -> flat 0 -> 0
    //   output[0,1] = input[1,0] -> flat 1 -> 2
    //   output[1,0] = input[0,1] -> flat 2 -> 1
    //   output[1,1] = input[1,1] -> flat 3 -> 3
    let mut layer = TransposeLayer::new(vec![1, 0]);
    layer.set_input_shape(vec![2, 2]);

    // Non-identity bounds: 1 output, 4 inputs
    // A = [[1, 2, 3, 4]] means: out = 1*y0 + 2*y1 + 3*y2 + 4*y3
    // After transpose backward: out = 1*x0 + 2*x2 + 3*x1 + 4*x3
    //   = 1*x0 + 3*x1 + 2*x2 + 4*x3
    //   = [[1, 3, 2, 4]]
    let lower_a = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper_a = Array2::from_shape_vec((1, 4), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let bounds = LinearBounds::new(lower_a, Array1::zeros(1), upper_a, Array1::zeros(1)).unwrap();

    let result = layer.propagate_linear(&bounds).unwrap();
    let result_ref = result.as_ref();

    assert_eq!(
        result_ref.lower_a.as_slice().unwrap(),
        &[1.0, 3.0, 2.0, 4.0],
        "Lower A columns should be permuted"
    );
    assert_eq!(
        result_ref.upper_a.as_slice().unwrap(),
        &[5.0, 7.0, 6.0, 8.0],
        "Upper A columns should be permuted"
    );
    // Bias unchanged
    assert_eq!(result_ref.lower_b.as_slice().unwrap(), &[0.0]);
    assert_eq!(result_ref.upper_b.as_slice().unwrap(), &[0.0]);
}

/// CROWN backward returns error when input_shape is not set.
#[ntest::timeout(5000)]
#[test]
fn transpose_crown_backward_error_no_input_shape() {
    let layer = TransposeLayer::new(vec![1, 0]);
    // Do NOT call set_input_shape

    let bounds = LinearBounds::identity(4);
    let result = layer.propagate_linear(&bounds);

    assert!(result.is_err(), "Should return Err without input_shape");
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("input_shape"),
        "Error should mention input_shape, got: {err}"
    );
}

/// CROWN backward returns error on shape mismatch.
#[ntest::timeout(5000)]
#[test]
fn transpose_crown_backward_error_shape_mismatch() {
    let mut layer = TransposeLayer::new(vec![1, 0]);
    layer.set_input_shape(vec![2, 3]); // 6 elements

    // Bounds with wrong number of inputs (4 != 6)
    let bounds = LinearBounds::identity(4);
    let result = layer.propagate_linear(&bounds);

    assert!(result.is_err(), "Should return Err on shape mismatch");
}

/// Batched CROWN backward correctly permutes columns (matches non-batched).
///
/// Verifies that propagate_linear_batched produces the same column permutation
/// as propagate_linear for a single-batch case.
#[ntest::timeout(5000)]
#[test]
fn transpose_batched_crown_backward_permutes_columns() {
    use ndarray::Array1;

    // Input shape: [2, 2] -> Transpose axes [1, 0]
    let mut layer = TransposeLayer::new(vec![1, 0]);
    layer.set_input_shape(vec![2, 2]);

    // Non-batched: A = [[1, 2, 3, 4]] (1 output, 4 inputs)
    let lower_a_2d = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper_a_2d = Array2::from_shape_vec((1, 4), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let non_batched =
        LinearBounds::new(lower_a_2d, Array1::zeros(1), upper_a_2d, Array1::zeros(1)).unwrap();
    let non_batched_result = layer.propagate_linear(&non_batched).unwrap();

    // Batched: same data but as ArrayD with shape [1, 4]
    let lower_a_nd = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper_a_nd = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let batched = BatchedLinearBounds::from_parts_unchecked(
        lower_a_nd,
        ArrayD::zeros(IxDyn(&[1])),
        upper_a_nd,
        ArrayD::zeros(IxDyn(&[1])),
        vec![2, 2],
        vec![2, 2],
    );
    let batched_result = layer.propagate_linear_batched(&batched).unwrap();

    // Both should produce the same column permutation
    assert_eq!(
        non_batched_result.lower_a.as_slice().unwrap(),
        batched_result.lower_a.as_slice().unwrap(),
        "Batched and non-batched lower_a should match"
    );
    assert_eq!(
        non_batched_result.upper_a.as_slice().unwrap(),
        batched_result.upper_a.as_slice().unwrap(),
        "Batched and non-batched upper_a should match"
    );
}

/// Batched CROWN backward returns error when input_shape is not set.
#[ntest::timeout(5000)]
#[test]
fn transpose_batched_crown_backward_error_no_input_shape() {
    let layer = TransposeLayer::new(vec![1, 0]);
    // Do NOT call set_input_shape

    let batched = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::zeros(IxDyn(&[1, 4])),
        ArrayD::zeros(IxDyn(&[1])),
        ArrayD::zeros(IxDyn(&[1, 4])),
        ArrayD::zeros(IxDyn(&[1])),
        vec![2, 2],
        vec![2, 2],
    );
    let result = layer.propagate_linear_batched(&batched);

    assert!(result.is_err(), "Should return Err without input_shape");
    let err = result.unwrap_err();
    assert!(
        format!("{err}").contains("input_shape"),
        "Error should mention input_shape, got: {err}"
    );
}

/// Batched CROWN backward with multiple batch dimensions.
///
/// Tests that column permutation is applied independently across batch positions.
#[ntest::timeout(5000)]
#[test]
fn transpose_batched_crown_backward_multi_batch() {
    // Input shape: [2, 2] -> Transpose axes [1, 0]
    // Batched bounds with shape [2, 1, 4] (2 batch positions, 1 output, 4 inputs)
    let mut layer = TransposeLayer::new(vec![1, 0]);
    layer.set_input_shape(vec![2, 2]);

    // Batch 0: A = [[1, 2, 3, 4]]
    // Batch 1: A = [[10, 20, 30, 40]]
    // After permutation: Batch 0: [[1, 3, 2, 4]], Batch 1: [[10, 30, 20, 40]]
    let lower_a = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 4]),
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    )
    .unwrap();
    let upper_a = lower_a.clone();

    let batched = BatchedLinearBounds::from_parts_unchecked(
        lower_a,
        ArrayD::zeros(IxDyn(&[2, 1])),
        upper_a,
        ArrayD::zeros(IxDyn(&[2, 1])),
        vec![2, 2],
        vec![2, 2],
    );
    let result = layer.propagate_linear_batched(&batched).unwrap();

    assert_eq!(
        result.lower_a.as_slice().unwrap(),
        &[1.0, 3.0, 2.0, 4.0, 10.0, 30.0, 20.0, 40.0],
        "Each batch position should have independently permuted columns"
    );
}

// ==================== SliceLayer CROWN panic cliff regression (#2759) ====================

/// CROWN backward with start > end must return Err, not panic from usize underflow.
#[ntest::timeout(5000)]
#[test]
fn slice_crown_start_gt_end_returns_err_2759() {
    let mut layer = SliceLayer::new(0, 3, 1);
    layer.set_input_shape(vec![4]);

    let bounds = LinearBounds::identity(1);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("start > end must return Err, not panic");
    assert!(
        err.to_string().contains("empty after clamping"),
        "expected range validation error, got: {err}"
    );
}

/// CROWN backward with start == end (zero-length slice) must return Err.
#[ntest::timeout(5000)]
#[test]
fn slice_crown_start_eq_end_returns_err_2759() {
    let mut layer = SliceLayer::new(0, 2, 2);
    layer.set_input_shape(vec![4]);

    let bounds = LinearBounds::identity(1);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("start == end must return Err");
    assert!(
        err.to_string().contains("empty after clamping"),
        "expected range validation error, got: {err}"
    );
}

/// CROWN backward with end > axis size must return Err.
#[ntest::timeout(5000)]
#[test]
fn slice_crown_end_exceeds_axis_clamps_2759() {
    // ONNX INT64_MAX sentinel: end > axis_size is clamped to axis_size.
    let mut layer = SliceLayer::new(0, 0, 10);
    layer.set_input_shape(vec![4]);

    // identity(4) since output after clamping is [0:4] = 4 elements
    let bounds = LinearBounds::identity(4);
    let result = layer
        .propagate_linear(&bounds)
        .expect("end > axis size should be clamped, not error");
    // Backward should map all 4 output positions to 4 input positions (identity)
    assert_eq!(result.into_owned().lower_a.shape(), &[4, 4]);
}

/// compute_output_shape with start > end must return Err (the original panic site).
#[ntest::timeout(5000)]
#[test]
fn slice_compute_output_shape_start_gt_end_returns_err_2759() {
    let layer = SliceLayer::new(0, 5, 2);
    let err = layer
        .compute_output_shape(&[10])
        .expect_err("start > end must return Err in compute_output_shape");
    assert!(
        err.to_string().contains("empty after clamping"),
        "expected range validation error, got: {err}"
    );
}

/// propagate_linear_with_bounds trait dispatch with start > end must return Err.
#[ntest::timeout(5000)]
#[test]
fn slice_crown_with_bounds_start_gt_end_returns_err_2759() {
    let layer = SliceLayer::new(0, 3, 1);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0; 4]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).expect("valid shape"),
    )
    .expect("valid bounds");

    let bounds = LinearBounds::identity(1);
    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect_err("trait dispatch must also return Err for start > end");
    assert!(
        err.to_string().contains("empty after clamping"),
        "expected range validation error, got: {err}"
    );
}

/// Regression test for #3206: Slice with start exceeding axis size must clamp start
/// (not just end) and produce the "empty after clamping" error.
/// This is the exact scenario from linearizenn: Slice(axis=0, start=2, end=large)
/// on an input of size 1 at axis 0.
#[ntest::timeout(5000)]
#[test]
fn slice_start_exceeds_axis_size_clamped_3206() {
    // linearizenn error: Slice range [2:1) on axis 0 size 1
    let layer = SliceLayer::new(0, 2, usize::MAX);
    let err = layer
        .compute_output_shape(&[1])
        .expect_err("start > axis_len must produce empty-slice error after clamping");
    let msg = err.to_string();
    assert!(
        msg.contains("empty after clamping"),
        "expected ONNX-compliant clamped error, got: {msg}"
    );
    assert!(
        msg.contains("3206"),
        "error should reference issue #3206, got: {msg}"
    );
}

/// Regression test for #3206 (lsnc variant): Slice(axis=0, start=1, end=large)
/// on an input of size 1 at axis 0. After clamping: start=1, end=1 → empty.
#[ntest::timeout(5000)]
#[test]
fn slice_start_eq_axis_size_clamped_3206() {
    // lsnc error: Slice range [1:1) on axis 0 size 1
    let layer = SliceLayer::new(0, 1, usize::MAX);
    let err = layer
        .compute_output_shape(&[1])
        .expect_err("start == axis_len must produce empty-slice error after clamping");
    let msg = err.to_string();
    assert!(
        msg.contains("empty after clamping"),
        "expected clamped error, got: {msg}"
    );
}
