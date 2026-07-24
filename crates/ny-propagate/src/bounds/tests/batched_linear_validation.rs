// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for BatchedLinearBounds NaN validation (#2979).
//!
//! Mirrors linear_validation.rs but for BatchedLinearBounds. Unlike LinearBounds,
//! BatchedLinearBounds allows ±Inf in coefficients because compose() legitimately
//! produces ±Inf as conservative NaN fallbacks.

use crate::bounds::BatchedLinearBounds;
use ndarray::{ArrayD, IxDyn};

/// Helper to create a simple [out_dim, in_dim] coefficient array.
fn coeff(data: &[f32], out_dim: usize, in_dim: usize) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[out_dim, in_dim]), data.to_vec()).expect("coeff shape mismatch")
}

/// Helper to create a simple [out_dim] bias array.
fn bias(data: &[f32]) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(&[data.len()]), data.to_vec()).expect("bias shape mismatch")
}

/// BatchedLinearBounds::new() rejects NaN in lower_a coefficients.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_rejects_nan_lower_a() {
    let err = BatchedLinearBounds::new(
        coeff(&[f32::NAN, 1.0], 1, 2),
        bias(&[0.0]),
        coeff(&[1.0, 1.0], 1, 2),
        bias(&[0.0]),
        vec![2],
        vec![1],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower_a") && msg.contains("NaN"),
        "expected lower_a NaN error, got: {msg}"
    );
}

/// BatchedLinearBounds::new() rejects NaN in upper_a coefficients.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_rejects_nan_upper_a() {
    let err = BatchedLinearBounds::new(
        coeff(&[1.0, 1.0], 1, 2),
        bias(&[0.0]),
        coeff(&[1.0, f32::NAN], 1, 2),
        bias(&[0.0]),
        vec![2],
        vec![1],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("upper_a") && msg.contains("NaN"),
        "expected upper_a NaN error, got: {msg}"
    );
}

/// BatchedLinearBounds::new() rejects NaN in lower_b bias.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_rejects_nan_lower_b() {
    let err = BatchedLinearBounds::new(
        coeff(&[1.0], 1, 1),
        bias(&[f32::NAN]),
        coeff(&[1.0], 1, 1),
        bias(&[0.0]),
        vec![1],
        vec![1],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower_b") && msg.contains("NaN"),
        "expected lower_b NaN error, got: {msg}"
    );
}

/// BatchedLinearBounds::new() rejects NaN in upper_b bias.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_rejects_nan_upper_b() {
    let err = BatchedLinearBounds::new(
        coeff(&[1.0], 1, 1),
        bias(&[0.0]),
        coeff(&[1.0], 1, 1),
        bias(&[f32::NAN]),
        vec![1],
        vec![1],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("upper_b") && msg.contains("NaN"),
        "expected upper_b NaN error, got: {msg}"
    );
}

/// BatchedLinearBounds::new() allows ±Inf in bias vectors (conservative bounds).
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_allows_inf_bias() {
    let result = BatchedLinearBounds::new(
        coeff(&[1.0], 1, 1),
        bias(&[f32::NEG_INFINITY]),
        coeff(&[1.0], 1, 1),
        bias(&[f32::INFINITY]),
        vec![1],
        vec![1],
    );
    assert!(
        result.is_ok(),
        "±Inf biases should be allowed for conservative bounds"
    );
}

/// BatchedLinearBounds::new() allows ±Inf in coefficients (compose() produces these).
/// This differs from LinearBounds which rejects Inf coefficients.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_allows_inf_coefficients() {
    let result = BatchedLinearBounds::new(
        coeff(&[f32::NEG_INFINITY], 1, 1),
        bias(&[0.0]),
        coeff(&[f32::INFINITY], 1, 1),
        bias(&[0.0]),
        vec![1],
        vec![1],
    );
    assert!(
        result.is_ok(),
        "±Inf coefficients should be allowed (compose() produces these)"
    );
}

/// BatchedLinearBounds::new() accepts clean finite values.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_accepts_clean_values() {
    let result = BatchedLinearBounds::new(
        coeff(&[1.0, -0.5, 0.3, 2.0], 2, 2),
        bias(&[0.1, -0.2]),
        coeff(&[1.2, -0.3, 0.5, 2.1], 2, 2),
        bias(&[0.3, 0.0]),
        vec![2],
        vec![2],
    );
    assert!(result.is_ok(), "clean finite values should be accepted");
}

/// BatchedLinearBounds::new() rejects 1D coefficient arrays.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_rejects_1d_coeff() {
    let err = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::zeros(IxDyn(&[])),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::zeros(IxDyn(&[])),
        vec![2],
        vec![2],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("ndim >= 2"), "expected ndim error, got: {msg}");
}

/// BatchedLinearBounds::new() rejects mismatched A shapes.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_rejects_mismatched_a_shapes() {
    let err = BatchedLinearBounds::new(
        coeff(&[1.0, 2.0], 1, 2),
        bias(&[0.0]),
        coeff(&[1.0, 2.0, 3.0, 4.0], 2, 2),
        bias(&[0.0, 0.0]),
        vec![2],
        vec![1],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower_a shape") && msg.contains("upper_a shape"),
        "expected A shape mismatch error, got: {msg}"
    );
}

/// BatchedLinearBounds::new() rejects mismatched bias shapes.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_rejects_mismatched_b_shapes() {
    let err = BatchedLinearBounds::new(
        coeff(&[1.0, 2.0], 1, 2),
        bias(&[0.0]),
        coeff(&[1.0, 2.0], 1, 2),
        bias(&[0.0, 0.0]),
        vec![2],
        vec![1],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower_b shape") && msg.contains("upper_b shape"),
        "expected bias shape mismatch error, got: {msg}"
    );
}

/// BatchedLinearBounds::new() rejects bias shape inconsistent with A shape.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_rejects_wrong_bias_dims() {
    let err = BatchedLinearBounds::new(
        coeff(&[1.0, 2.0, 3.0, 4.0], 2, 2),
        bias(&[0.0, 0.0, 0.0]), // 3 elements, but A has out_dim=2
        coeff(&[1.0, 2.0, 3.0, 4.0], 2, 2),
        bias(&[0.0, 0.0, 0.0]),
        vec![2],
        vec![2],
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lower_b shape") && msg.contains("expected"),
        "expected bias/A shape mismatch error, got: {msg}"
    );
}

/// Accessors return correct references.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_accessors() {
    let bounds = BatchedLinearBounds::new(
        coeff(&[1.0, 2.0, 3.0, 4.0], 2, 2),
        bias(&[0.5, -0.5]),
        coeff(&[5.0, 6.0, 7.0, 8.0], 2, 2),
        bias(&[1.0, 1.0]),
        vec![2],
        vec![2],
    )
    .unwrap();

    assert_eq!(bounds.lower_a().shape(), &[2, 2]);
    assert_eq!(bounds.upper_a().shape(), &[2, 2]);
    assert_eq!(bounds.lower_b().shape(), &[2]);
    assert_eq!(bounds.upper_b().shape(), &[2]);
    assert_eq!(bounds.input_shape(), &[2]);
    assert_eq!(bounds.output_shape(), &[2]);

    // Verify actual values
    assert_eq!(bounds.lower_a()[[0, 0]], 1.0);
    assert_eq!(bounds.upper_a()[[1, 1]], 8.0);
    assert_eq!(bounds.lower_b()[[0]], 0.5);
    assert_eq!(bounds.upper_b()[[1]], 1.0);
}

/// Direct field mutation within crate works (fields are pub(crate)).
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_field_mutation() {
    let mut bounds = BatchedLinearBounds::new(
        coeff(&[1.0], 1, 1),
        bias(&[0.0]),
        coeff(&[1.0], 1, 1),
        bias(&[0.0]),
        vec![1],
        vec![1],
    )
    .unwrap();

    bounds.lower_a[[0, 0]] = 42.0;
    bounds.lower_b[[0]] = -1.0;
    assert_eq!(bounds.lower_a()[[0, 0]], 42.0);
    assert_eq!(bounds.lower_b()[[0]], -1.0);
}

/// into_parts() returns all six components.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_into_parts() {
    let bounds = BatchedLinearBounds::new(
        coeff(&[1.0], 1, 1),
        bias(&[2.0]),
        coeff(&[3.0], 1, 1),
        bias(&[4.0]),
        vec![1],
        vec![1],
    )
    .unwrap();

    let (la, lb, ua, ub, is, os) = bounds.into_parts();
    assert_eq!(la[[0, 0]], 1.0);
    assert_eq!(lb[[0]], 2.0);
    assert_eq!(ua[[0, 0]], 3.0);
    assert_eq!(ub[[0]], 4.0);
    assert_eq!(is, vec![1]);
    assert_eq!(os, vec![1]);
}

/// Batched (3D) coefficient arrays work correctly.
#[ntest::timeout(10000)]
#[test]
fn test_batched_linear_bounds_new_batched_3d() {
    // Shape: [batch=2, out_dim=1, in_dim=3] for coefficients
    // Shape: [batch=2, out_dim=1] for biases
    let lower_a =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let upper_a =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5]).unwrap();
    let lower_b = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![0.0, 0.0]).unwrap();
    let upper_b = ArrayD::from_shape_vec(IxDyn(&[2, 1]), vec![1.0, 1.0]).unwrap();

    let result =
        BatchedLinearBounds::new(lower_a, lower_b, upper_a, upper_b, vec![2, 3], vec![2, 1]);
    assert!(result.is_ok(), "3D batched bounds should be accepted");
}
