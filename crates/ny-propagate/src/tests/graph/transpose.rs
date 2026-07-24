// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! GraphNetwork transpose layer tests.
use crate::*;
use ndarray::Array2;

#[ntest::timeout(10000)]
#[test]
fn test_transpose_layer_2d() {
    // Test 2D transpose
    let transpose = TransposeLayer::transpose_2d();

    let input = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = transpose.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    // Check transposed values
    assert!((output.lower()[[0, 0]] - 1.0).abs() < 1e-5);
    assert!((output.lower()[[0, 1]] - 4.0).abs() < 1e-5);
    assert!((output.lower()[[1, 0]] - 2.0).abs() < 1e-5);
    assert!((output.lower()[[1, 1]] - 5.0).abs() < 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_transpose_layer_batched() {
    // Test batched transpose (swap last two dims of 3D tensor)
    let transpose = TransposeLayer::batched_transpose();

    // Shape: (2, 3, 4) -> (2, 4, 3)
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let input = BoundedTensor::new(
        ndarray::Array3::from_shape_vec((2, 3, 4), data.clone())
            .unwrap()
            .into_dyn(),
        ndarray::Array3::from_shape_vec((2, 3, 4), data)
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = transpose.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 4, 3]);
}

#[ntest::timeout(10000)]
#[test]
fn test_transpose_layer_interval_soundness() {
    // Test that transpose preserves interval bounds correctly
    let transpose = TransposeLayer::transpose_2d();

    let input = BoundedTensor::new(
        Array2::from_shape_vec((2, 3), vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0])
            .unwrap()
            .into_dyn(),
        Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_dyn(),
    )
    .unwrap();

    let output = transpose.propagate_ibp(&input).unwrap();

    // Check that bounds are preserved for each element
    // Original [0,1] at (0,0) should be at (0,0) after transpose
    assert!((output.lower()[[0, 0]] - 0.0).abs() < 1e-5);
    assert!((output.upper()[[0, 0]] - 1.0).abs() < 1e-5);

    // Original [3,4] at (1,0) should be at (0,1) after transpose
    assert!((output.lower()[[0, 1]] - 3.0).abs() < 1e-5);
    assert!((output.upper()[[0, 1]] - 4.0).abs() < 1e-5);
}
