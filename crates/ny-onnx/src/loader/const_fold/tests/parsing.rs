// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::common::{normalize_axis, parse_scalar_i64, parse_shape_i64};
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_parse_scalar_i64_rounding() {
    assert_eq!(parse_scalar_i64(3.0), Some(3));
    assert_eq!(parse_scalar_i64(-2.0), Some(-2));
    assert_eq!(parse_scalar_i64(1.00001), Some(1));
    assert_eq!(parse_scalar_i64(1.25), None);
    assert_eq!(parse_scalar_i64(1.0e20), None);
    assert_eq!(parse_scalar_i64(f32::NAN), None);
}

#[test]
fn test_parse_shape_i64_rejects_non_integers() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.5]).unwrap();
    assert_eq!(parse_shape_i64(&arr), None);
}

#[test]
fn test_parse_shape_i64_accepts_integers() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 0.0, 4.0]).unwrap();
    assert_eq!(parse_shape_i64(&arr), Some(vec![1, 0, 4]));
}

#[test]
fn test_parse_shape_i64_accepts_negative_integers() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -3.0]).unwrap();
    assert_eq!(parse_shape_i64(&arr), Some(vec![-1, -3]));
}

#[test]
fn test_parse_shape_i64_rejects_non_finite() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, f32::INFINITY]).unwrap();
    assert_eq!(parse_shape_i64(&arr), None);
}

#[test]
fn test_parse_shape_i64_rounding_tolerance() {
    let arr = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.00009, 2.0002]).unwrap();
    assert_eq!(parse_shape_i64(&arr), None);
    let arr = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.00009]).unwrap();
    assert_eq!(parse_shape_i64(&arr), Some(vec![1]));
}

#[test]
fn test_normalize_axis_bounds() {
    assert_eq!(normalize_axis(0, 3), Some(0));
    assert_eq!(normalize_axis(-1, 3), Some(2));
    assert_eq!(normalize_axis(3, 3), None);
    assert_eq!(normalize_axis(-4, 3), None);
}

#[test]
fn test_normalize_axis_zero_ndim() {
    assert_eq!(normalize_axis(0, 0), None);
    assert_eq!(normalize_axis(-1, 0), None);
}
