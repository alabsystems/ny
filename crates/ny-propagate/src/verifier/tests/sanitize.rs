// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// Licensed under the Apache License, Version 2.0

use super::super::*;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

#[ntest::timeout(5000)]
#[test]
fn test_sanitize_output_bounds_normal_values() {
    let tensor = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
    )
    .unwrap();

    let result = Verifier::sanitize_output_bounds(tensor).unwrap();

    assert_eq!(result.lower()[[0]], -1.0);
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.lower()[[2]], 1.0);
    assert_eq!(result.upper()[[0]], 1.0);
    assert_eq!(result.upper()[[1]], 2.0);
    assert_eq!(result.upper()[[2]], 3.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_sanitize_output_bounds_nan_lower() {
    // Use new_unchecked to allow NaN values for testing sanitization
    let tensor = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
    )
    .unwrap();

    let result = Verifier::sanitize_output_bounds(tensor).unwrap();

    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.lower()[[2]], 1.0);
    assert_eq!(result.upper()[[0]], 1.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_sanitize_output_bounds_nan_upper() {
    let tensor = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 2.0, 3.0]).unwrap(),
    )
    .unwrap();

    let result = Verifier::sanitize_output_bounds(tensor).unwrap();

    assert_eq!(result.lower()[[0]], -1.0);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
    assert_eq!(result.upper()[[1]], 2.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_sanitize_output_bounds_both_nan() {
    let tensor = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap(),
    )
    .unwrap();

    let result = Verifier::sanitize_output_bounds(tensor).unwrap();

    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.upper()[[1]], 1.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_sanitize_output_bounds_inverted_bounds() {
    // lower > upper should be sanitized
    let tensor = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0, 0.0]).unwrap(), // lower > upper
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 1.0]).unwrap(),
    )
    .unwrap();

    let result = Verifier::sanitize_output_bounds(tensor).unwrap();

    // Inverted bounds should become (-inf, +inf)
    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
    // Normal bounds unchanged
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.upper()[[1]], 1.0);
}

#[ntest::timeout(5000)]
#[test]
fn test_sanitize_output_bounds_all_nan() {
    let tensor = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, f32::NAN, f32::NAN]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, f32::NAN, f32::NAN]).unwrap(),
    )
    .unwrap();

    let result = Verifier::sanitize_output_bounds(tensor).unwrap();

    for i in 0..3 {
        assert_eq!(result.lower()[[i]], f32::NEG_INFINITY);
        assert_eq!(result.upper()[[i]], f32::INFINITY);
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_sanitize_output_bounds_multidimensional() {
    let tensor = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![f32::NAN, -1.0, 0.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, f32::NAN, 1.0, 1.0]).unwrap(), // [1,1] inverted
    )
    .unwrap();

    let result = Verifier::sanitize_output_bounds(tensor).unwrap();

    // [0,0]: NaN lower
    assert_eq!(result.lower()[[0, 0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0, 0]], 1.0);
    // [0,1]: NaN upper
    assert_eq!(result.lower()[[0, 1]], -1.0);
    assert_eq!(result.upper()[[0, 1]], f32::INFINITY);
    // [1,0]: normal
    assert_eq!(result.lower()[[1, 0]], 0.0);
    assert_eq!(result.upper()[[1, 0]], 1.0);
    // [1,1]: inverted (2.0 > 1.0)
    assert_eq!(result.lower()[[1, 1]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[1, 1]], f32::INFINITY);
}

#[ntest::timeout(5000)]
#[test]
fn test_sanitize_preserves_infinity_bounds() {
    // Valid infinite bounds should be preserved, not sanitized
    // Use new_unchecked since BoundedTensor::new rejects infinities
    let tensor = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap(),
    )
    .unwrap();

    let result = Verifier::sanitize_output_bounds(tensor).unwrap();

    assert_eq!(result.lower()[[0]], f32::NEG_INFINITY);
    assert_eq!(result.upper()[[0]], f32::INFINITY);
    assert_eq!(result.lower()[[1]], 0.0);
    assert_eq!(result.upper()[[1]], 1.0);
}
