// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use ndarray::{arr1, arr2};

#[test]
fn test_transpose_last_two() {
    // Create (2, 3) zonotope
    let values = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let z = ZonotopeTensor::from_input_2d(&values, 0.1);

    assert_eq!(z.element_shape, vec![2, 3]);

    let transposed = z.transpose_last_two().unwrap();

    assert_eq!(transposed.element_shape, vec![3, 2]);
    assert_eq!(transposed.n_error_terms, z.n_error_terms);

    let orig_center = z.center();
    let trans_center = transposed.center();

    // z[i,j] should equal transposed[j,i]
    assert!((orig_center[[0, 0]] - trans_center[[0, 0]]).abs() < 1e-6);
    assert!((orig_center[[0, 1]] - trans_center[[1, 0]]).abs() < 1e-6);
    assert!((orig_center[[0, 2]] - trans_center[[2, 0]]).abs() < 1e-6);
    assert!((orig_center[[1, 0]] - trans_center[[0, 1]]).abs() < 1e-6);
}

// ============== transpose_last_two Mutation-Killing Tests ==============

#[test]
fn test_transpose_last_two_comparison() {
    // Kills: replace < with > in line 1504

    // Test with 1D input - should fail (less than 2 dims)
    let z_1d = ZonotopeTensor::concrete(arr1(&[1.0_f32, 2.0]).into_dyn());
    let result = z_1d.transpose_last_two();
    assert!(
        result.is_err(),
        "1D zonotope should fail transpose_last_two"
    );

    // Test with 2D input - should work (exactly 2 dims)
    let z_2d = ZonotopeTensor::concrete(arr2(&[[1.0_f32, 2.0], [3.0, 4.0]]).into_dyn());
    let result = z_2d.transpose_last_two();
    assert!(
        result.is_ok(),
        "2D zonotope should work with transpose_last_two"
    );
}

#[test]
fn test_transpose_last_two_axis_swap() {
    // Kills: replace - with / in line 1514 (ndim - 2, ndim - 1)

    // Test 2D transpose: (2, 3) -> (3, 2)
    let values = arr2(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let z = ZonotopeTensor::concrete(values.into_dyn());

    let transposed = z.transpose_last_two().unwrap();

    assert_eq!(
        transposed.element_shape,
        vec![3, 2],
        "shape should be transposed from (2,3) to (3,2)"
    );

    // Verify values are correctly transposed
    let center = transposed.center();

    // Original [0,0]=1, [0,1]=2, [0,2]=3, [1,0]=4, [1,1]=5, [1,2]=6
    // Transposed [0,0]=1, [0,1]=4, [1,0]=2, [1,1]=5, [2,0]=3, [2,1]=6
    assert!((center[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((center[[0, 1]] - 4.0).abs() < 1e-6);
    assert!((center[[1, 0]] - 2.0).abs() < 1e-6);
    assert!((center[[1, 1]] - 5.0).abs() < 1e-6);
    assert!((center[[2, 0]] - 3.0).abs() < 1e-6);
    assert!((center[[2, 1]] - 6.0).abs() < 1e-6);
}

#[test]
fn test_transpose_last_two_3d() {
    // Test 3D transpose: (2, 3, 4) -> (2, 4, 3)
    // Only last two dimensions should be swapped

    let values =
        ndarray::Array3::<f32>::from_shape_fn((2, 3, 4), |(a, b, c)| (a * 100 + b * 10 + c) as f32)
            .into_dyn();

    let z = ZonotopeTensor::concrete(values);
    let transposed = z.transpose_last_two().unwrap();

    assert_eq!(
        transposed.element_shape,
        vec![2, 4, 3],
        "3D shape (2,3,4) should become (2,4,3)"
    );

    // Verify first dimension is unchanged, last two are swapped
    let center = transposed.center();

    // Original [0, 1, 2] = 12.0 should become transposed [0, 2, 1] = 12.0
    assert!(
        (center[[0, 2, 1]] - 12.0).abs() < 1e-6,
        "value at [0,1,2] should move to [0,2,1]"
    );

    // Original [1, 0, 3] = 103.0 should become transposed [1, 3, 0] = 103.0
    assert!(
        (center[[1, 3, 0]] - 103.0).abs() < 1e-6,
        "value at [1,0,3] should move to [1,3,0]"
    );
}
