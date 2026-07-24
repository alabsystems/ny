// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// Licensed under the Apache License, Version 2.0

//! Tests for BaB output bounds NaN handling (#2589).

use super::super::*;
use ndarray::{ArrayD, IxDyn};
use ny_core::Bound;
use ny_tensor::BoundedTensor;

/// Helper: create a fallback bound array of [-inf, +inf].
fn fallback_bounds(n: usize) -> Vec<Bound> {
    vec![Bound::new_allow_infinite(f32::NEG_INFINITY, f32::INFINITY); n]
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_bab_output_bounds_normal() {
    let mut output = fallback_bounds(3);
    let bab = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
    )
    .unwrap();

    Verifier::apply_bab_output_bounds(&mut output, &bab);

    assert_eq!(output[0].lower(), -1.0);
    assert_eq!(output[0].upper(), 1.0);
    assert_eq!(output[1].lower(), 0.0);
    assert_eq!(output[1].upper(), 2.0);
    assert_eq!(output[2].lower(), 1.0);
    assert_eq!(output[2].upper(), 3.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_bab_output_bounds_nan_lower_preserves_fallback() {
    let mut output = fallback_bounds(3);
    let bab = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![5.0, 2.0, 3.0]).unwrap(),
    )
    .unwrap();

    Verifier::apply_bab_output_bounds(&mut output, &bab);

    // NaN lower → fallback preserved
    assert_eq!(output[0].lower(), f32::NEG_INFINITY);
    assert_eq!(output[0].upper(), f32::INFINITY);
    // Normal bounds applied
    assert_eq!(output[1].lower(), 0.0);
    assert_eq!(output[1].upper(), 2.0);
    assert_eq!(output[2].lower(), 1.0);
    assert_eq!(output[2].upper(), 3.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_bab_output_bounds_nan_upper_preserves_fallback() {
    let mut output = fallback_bounds(2);
    let bab = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 2.0]).unwrap(),
    )
    .unwrap();

    Verifier::apply_bab_output_bounds(&mut output, &bab);

    // NaN upper → fallback preserved
    assert_eq!(output[0].lower(), f32::NEG_INFINITY);
    assert_eq!(output[0].upper(), f32::INFINITY);
    // Normal bounds applied
    assert_eq!(output[1].lower(), 0.0);
    assert_eq!(output[1].upper(), 2.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_bab_output_bounds_both_nan() {
    let mut output = fallback_bounds(2);
    let bab = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 3.0]).unwrap(),
    )
    .unwrap();

    Verifier::apply_bab_output_bounds(&mut output, &bab);

    // Both NaN → fallback preserved
    assert_eq!(output[0].lower(), f32::NEG_INFINITY);
    assert_eq!(output[0].upper(), f32::INFINITY);
    // Normal bounds applied
    assert_eq!(output[1].lower(), 1.0);
    assert_eq!(output[1].upper(), 3.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_bab_output_bounds_all_nan() {
    let mut output = fallback_bounds(3);
    let bab = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN; 3]).unwrap(),
    )
    .unwrap();

    Verifier::apply_bab_output_bounds(&mut output, &bab);

    // All fallback preserved
    for bound in &output {
        assert_eq!(bound.lower(), f32::NEG_INFINITY);
        assert_eq!(bound.upper(), f32::INFINITY);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_bab_output_bounds_inverted_skipped() {
    let mut output = fallback_bounds(2);
    let bab = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0, 0.0]).unwrap(), // inverted: lower > upper
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 1.0]).unwrap(),
    )
    .unwrap();

    Verifier::apply_bab_output_bounds(&mut output, &bab);

    // Inverted bounds → fallback preserved (not NaN, just inverted)
    assert_eq!(output[0].lower(), f32::NEG_INFINITY);
    assert_eq!(output[0].upper(), f32::INFINITY);
    // Normal bounds applied
    assert_eq!(output[1].lower(), 0.0);
    assert_eq!(output[1].upper(), 1.0);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_bab_output_bounds_fewer_bab_than_output() {
    let mut output = fallback_bounds(4);
    let bab = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, 4.0]).unwrap(),
    )
    .unwrap();

    Verifier::apply_bab_output_bounds(&mut output, &bab);

    // First 2 applied
    assert_eq!(output[0].lower(), 1.0);
    assert_eq!(output[0].upper(), 3.0);
    assert_eq!(output[1].lower(), 2.0);
    assert_eq!(output[1].upper(), 4.0);
    // Remaining fallback preserved
    assert_eq!(output[2].lower(), f32::NEG_INFINITY);
    assert_eq!(output[3].lower(), f32::NEG_INFINITY);
}

#[ntest::timeout(10000)]
#[test]
fn test_apply_bab_output_bounds_infinite_bounds_applied() {
    let mut output = fallback_bounds(2);
    // Infinite but non-NaN bounds should be applied
    let bab = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap(),
    )
    .unwrap();

    Verifier::apply_bab_output_bounds(&mut output, &bab);

    assert_eq!(output[0].lower(), f32::NEG_INFINITY);
    assert_eq!(output[0].upper(), f32::INFINITY);
    assert_eq!(output[1].lower(), 0.0);
    assert_eq!(output[1].upper(), 1.0);
}
