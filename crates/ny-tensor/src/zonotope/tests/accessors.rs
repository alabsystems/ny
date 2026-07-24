// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

#[test]
fn test_shape_returns_correct_dimensions() {
    // Test shape() returns the actual element_shape, not empty/[0]/[1]
    // Kills: replace shape -> &[usize] with Vec::leak(Vec::new())
    // Kills: replace shape -> &[usize] with Vec::leak(vec![0])
    // Kills: replace shape -> &[usize] with Vec::leak(vec![1])
    let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(); // shape [2, 3]
    let z = ZonotopeTensor::concrete(values);

    let shape = z.shape();
    assert_eq!(shape.len(), 2, "shape should have 2 dimensions");
    assert_eq!(shape[0], 2, "first dimension should be 2");
    assert_eq!(shape[1], 3, "second dimension should be 3");
}

#[test]
fn test_len_returns_correct_count() {
    // Test len() returns product of dimensions, not 0 or 1
    // Kills: replace len -> usize with 0
    // Kills: replace len -> usize with 1
    let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn(); // 6 elements
    let z = ZonotopeTensor::concrete(values);

    assert_eq!(z.len(), 6, "len should be 2*3=6");

    // Also test a larger tensor
    let values_3d = ArrayD::<f32>::zeros(IxDyn(&[2, 3, 4])); // 24 elements
    let z_3d = ZonotopeTensor::concrete(values_3d);
    assert_eq!(z_3d.len(), 24, "len should be 2*3*4=24");
}

#[test]
fn test_is_empty_true_for_zero_elements() {
    // Test is_empty() returns true when len() == 0
    // Kills: replace is_empty -> bool with false
    // Kills: replace == with != in is_empty
    let values = ArrayD::<f32>::zeros(IxDyn(&[0])); // empty tensor
    let z = ZonotopeTensor::concrete(values);

    assert!(z.is_empty(), "zonotope with 0 elements should be empty");
    assert_eq!(z.len(), 0, "len should be 0 for empty zonotope");
}

#[test]
fn test_is_empty_false_for_non_zero_elements() {
    // Test is_empty() returns false when len() > 0
    // Kills: replace is_empty -> bool with true
    let values = arr1(&[1.0, 2.0]).into_dyn();
    let z = ZonotopeTensor::concrete(values);

    assert!(!z.is_empty(), "zonotope with elements should not be empty");
    assert_eq!(z.len(), 2, "len should equal element count");
}

#[test]
fn test_has_unbounded_false_for_finite_coeffs() {
    // Test has_unbounded() returns false when all coeffs are finite
    // Kills: replace has_unbounded -> bool with true
    let values = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let z = ZonotopeTensor::from_input_shared(&values, 0.1);

    assert!(
        !z.has_unbounded(),
        "zonotope with finite coeffs should not be unbounded"
    );
}

#[test]
fn test_has_unbounded_true_for_infinite_coeffs() {
    // Test has_unbounded() returns true when coeffs contain infinity
    // Kills: replace has_unbounded -> bool with false
    let values = arr1(&[1.0, f32::INFINITY, 3.0]).into_dyn();
    let z = ZonotopeTensor::concrete(values);

    assert!(
        z.has_unbounded(),
        "zonotope with infinite center should be unbounded"
    );

    // Also test with infinite error term
    let mut z2 = ZonotopeTensor::from_input_shared(&arr1(&[1.0, 2.0]).into_dyn(), 0.1);
    z2.coeffs[[1, 0]] = f32::INFINITY;
    assert!(
        z2.has_unbounded(),
        "zonotope with infinite error coeff should be unbounded"
    );
}
