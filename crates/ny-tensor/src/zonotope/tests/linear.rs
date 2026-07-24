// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

#[test]
fn test_linear_rejects_empty_shape() {
    // Kills: replace || with && in line 387
    // The condition is: is_empty() || last() != Some(&in_features)
    // With &&, only fails when BOTH conditions are true (which can't happen for empty)

    // Create a scalar zonotope (element_shape = [])
    let coeffs = ArrayD::<f32>::zeros(IxDyn(&[1])); // 0 error terms, scalar
    let z = ZonotopeTensor::new(coeffs).unwrap();
    assert!(
        z.element_shape.is_empty(),
        "should have empty element_shape"
    );

    let weight = arr2(&[[1.0, 2.0]]); // 1x2 matrix
    let result = z.linear(&weight, None);
    assert!(result.is_err(), "linear should reject empty element_shape");
}

#[test]
fn test_linear_rejects_shape_mismatch() {
    // Complements empty test: non-empty but wrong last dimension
    let values = arr1(&[1.0, 2.0, 3.0]).into_dyn(); // shape [3]
    let z = ZonotopeTensor::concrete(values);

    let weight = arr2(&[[1.0, 2.0]]); // expects input dim 2, got 3
    let result = z.linear(&weight, None);
    assert!(result.is_err(), "linear should reject dimension mismatch");
}

#[test]
fn test_linear_bias_addition() {
    // Kills: replace += with -= in line 436 (lane += &b.view())
    // Kills: replace += with *= in line 436
    let values = arr1(&[1.0, 2.0]).into_dyn(); // [2]
    let z = ZonotopeTensor::concrete(values);

    let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // identity 2x2
    let bias = arr1(&[10.0, 20.0]); // bias

    let result = z.linear(&weight, Some(&bias)).unwrap();
    let center = result.center();

    // With identity weight, output center should be input + bias
    // [1, 2] * I + [10, 20] = [11, 22]
    assert!(
        (center[[0]] - 11.0).abs() < 1e-6,
        "bias should be added (got {})",
        center[[0]]
    );
    assert!(
        (center[[1]] - 22.0).abs() < 1e-6,
        "bias should be added (got {})",
        center[[1]]
    );
}

#[test]
fn test_linear_bias_with_batch() {
    // Test bias addition with multi-dimensional input
    let values = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(); // [2, 2]
    let z = ZonotopeTensor::concrete(values);

    let weight = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // identity
    let bias = arr1(&[100.0, 200.0]);

    let result = z.linear(&weight, Some(&bias)).unwrap();
    let center = result.center();

    // Both rows should have bias added
    assert!((center[[0, 0]] - 101.0).abs() < 1e-6, "row 0 bias");
    assert!((center[[0, 1]] - 202.0).abs() < 1e-6, "row 0 bias");
    assert!((center[[1, 0]] - 103.0).abs() < 1e-6, "row 1 bias");
    assert!((center[[1, 1]] - 204.0).abs() < 1e-6, "row 1 bias");
}
