// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::common::BoundPropagation;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

/// Helper: build a BoundedTensor from flat lower/upper vecs and shape.
fn bounded(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap();
    BoundedTensor::new(l, u).unwrap()
}

// =========================================================================
// Constructor
// =========================================================================

/// TileLayer constructor and set_input_shape.
#[ntest::timeout(5000)]
#[test]
fn test_constructor() {
    let mut layer = TileLayer::new(0, 3);
    assert_eq!(layer.axis, 0);
    assert_eq!(layer.reps, 3);
    assert!(layer.input_shape.is_none());

    layer.set_input_shape(vec![2, 3]);
    assert_eq!(layer.input_shape, Some(vec![2, 3]));
}

// =========================================================================
// IBP propagation
// =========================================================================

/// Tile 1D tensor along axis 0 with reps=3: [a, b] -> [a, b, a, b, a, b].
#[ntest::timeout(5000)]
#[test]
fn test_ibp_1d_axis0_reps3() {
    let input = bounded(&[2], vec![1.0, 2.0], vec![3.0, 4.0]);
    let layer = TileLayer::new(0, 3);
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[6]);
    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
    assert_eq!(
        output.upper().as_slice().unwrap(),
        &[3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    );
}

/// Tile 2D tensor along axis 0: [[a,b],[c,d]] -> [[a,b],[c,d],[a,b],[c,d]].
#[ntest::timeout(5000)]
#[test]
fn test_ibp_2d_axis0_reps2() {
    let input = bounded(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);
    let layer = TileLayer::new(0, 2);
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[4, 2]);
    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
}

/// Tile 2D tensor along axis 1 (last dim): [[a,b]] -> [[a,b,a,b]].
#[ntest::timeout(5000)]
#[test]
fn test_ibp_2d_axis1_reps2() {
    let input = bounded(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);
    let layer = TileLayer::new(1, 2);
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 4]);
    // Row 0: [1,2,1,2], Row 1: [3,4,3,4]
    // Note: ndarray::concatenate along non-0 axis may produce non-contiguous arrays,
    // so we index element-wise instead of using as_slice().
    let lower = output.lower();
    let expected_lower = [[1.0, 2.0, 1.0, 2.0], [3.0, 4.0, 3.0, 4.0]];
    for (i, row) in expected_lower.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert_eq!(
                lower[[i, j]],
                val,
                "lower[{i},{j}] expected {val}, got {}",
                lower[[i, j]]
            );
        }
    }
}

/// Tile with negative axis (-1 == last dim).
#[ntest::timeout(5000)]
#[test]
fn test_ibp_negative_axis() {
    let input = bounded(&[2, 3], vec![0.0; 6], vec![1.0; 6]);
    let layer = TileLayer::new(-1, 2);
    let output = layer.propagate_ibp(&input).unwrap();

    // axis -1 on 2D = axis 1, so [2, 3] * 2 on axis 1 = [2, 6]
    assert_eq!(output.shape(), &[2, 6]);
}

/// Tile with reps=1 should be a no-op (identity).
#[ntest::timeout(5000)]
#[test]
fn test_ibp_reps1_noop() {
    let input = bounded(&[3, 2], vec![1.0; 6], vec![2.0; 6]);
    let layer = TileLayer::new(0, 1);
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(
        output.lower().as_slice().unwrap(),
        input.lower().as_slice().unwrap()
    );
    assert_eq!(
        output.upper().as_slice().unwrap(),
        input.upper().as_slice().unwrap()
    );
}

/// Tile with reps=0 should error.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_reps0_error() {
    let input = bounded(&[2], vec![1.0, 2.0], vec![3.0, 4.0]);
    let layer = TileLayer::new(0, 0);
    let err = layer.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{err}").contains("at least 1"),
        "Expected reps error, got: {err}"
    );
}

/// Tile with zero-valued dimension should error, not panic with division-by-zero. (#2806)
#[ntest::timeout(5000)]
#[test]
fn test_ibp_zero_dimension_returns_error() {
    // Shape [2, 0]: zero-valued dimension
    let l = ArrayD::from_shape_vec(IxDyn(&[2, 0]), vec![]).expect("valid shape");
    let u = ArrayD::from_shape_vec(IxDyn(&[2, 0]), vec![]).expect("valid shape");
    let input = BoundedTensor::new(l, u).expect("valid bounds");
    let layer = TileLayer::new(0, 2);
    let err = layer.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{err}").contains("zero-valued dimension"),
        "Expected zero-dimension error, got: {err}"
    );
}

/// CROWN backward with zero-valued dimension should error, not panic. (#2806)
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_zero_dimension_returns_error() {
    let mut layer = TileLayer::new(0, 2);
    layer.set_input_shape(vec![2, 0]);

    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .unwrap();

    let err = layer.propagate_linear(&bounds).unwrap_err();
    assert!(
        format!("{err}").contains("zero-valued dimension"),
        "Expected zero-dimension error, got: {err}"
    );
}

/// CROWN backward with bounds (propagate_linear_with_bounds) zero-dimension error. (#2806)
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_with_bounds_zero_dimension_returns_error() {
    let l = ArrayD::from_shape_vec(IxDyn(&[2, 0]), vec![]).expect("valid shape");
    let u = ArrayD::from_shape_vec(IxDyn(&[2, 0]), vec![]).expect("valid shape");
    let input = BoundedTensor::new(l, u).expect("valid bounds");

    let layer = TileLayer::new(0, 2);
    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .unwrap();

    let err = layer
        .propagate_linear_with_bounds(&bounds, &input)
        .unwrap_err();
    assert!(
        format!("{err}").contains("zero-valued dimension"),
        "Expected zero-dimension error, got: {err}"
    );
}

/// Tile IBP soundness: output bounds should contain the tiled values.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_soundness() {
    let input = bounded(&[3], vec![-1.0, 0.0, 2.0], vec![1.0, 3.0, 5.0]);
    let layer = TileLayer::new(0, 4);
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[12]);
    // Each triple of output should have the same bounds as the input
    for rep in 0..4 {
        for i in 0..3 {
            let idx = rep * 3 + i;
            assert_eq!(output.lower()[idx], input.lower()[i]);
            assert_eq!(output.upper()[idx], input.upper()[i]);
        }
    }
}

// =========================================================================
// CROWN backward propagation (propagate_linear)
// =========================================================================

/// CROWN backward with identity bounds through tile reps=2 on 1D.
/// Output identity is [6x6], input is [3], so backward should sum
/// pairs of columns: new_a[:, i] = a[:, 2*i] + a[:, 2*i+1] basically.
///
/// Actually for reps=2 on axis 0 of shape [3]:
/// output[0..3] = input[0..3], output[3..6] = input[0..3]
/// backward: new_a[:, i] = a[:, i] + a[:, i+3]
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_1d_reps2() {
    let mut layer = TileLayer::new(0, 2);
    layer.set_input_shape(vec![3]);

    // Identity bounds for the 6-element output
    let bounds = LinearBounds::new(
        Array2::eye(6),
        Array1::zeros(6),
        Array2::eye(6),
        Array1::zeros(6),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds).unwrap();
    let result = result.as_ref();

    // After backward: new bounds should be [6, 3]
    assert_eq!(result.lower_a.shape(), &[6, 3]);

    // For row 0 (output[0] = input[0]): new_lower_a[0, 0] = 1 (from both replicas)
    // Actually output[0] maps to input[0] (rep 0) and output[3] maps to input[0] (rep 1)
    // Identity bounds: row i has 1 at column i, 0 elsewhere
    // Row 0 of identity: [1,0,0,0,0,0] -> backward -> input[0] gets coeff from output[0] = 1
    assert_eq!(result.lower_a[[0, 0]], 1.0);
    assert_eq!(result.lower_a[[0, 1]], 0.0);
    assert_eq!(result.lower_a[[0, 2]], 0.0);

    // Row 3 of identity: [0,0,0,1,0,0] -> output[3] = input[0] (rep 1)
    // backward -> input[0] gets coeff 1
    assert_eq!(result.lower_a[[3, 0]], 1.0);
    assert_eq!(result.lower_a[[3, 1]], 0.0);
    assert_eq!(result.lower_a[[3, 2]], 0.0);
}

/// CROWN backward with reps=1 should be identity (no-op).
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_reps1_noop() {
    let mut layer = TileLayer::new(0, 1);
    layer.set_input_shape(vec![3]);

    let bounds = LinearBounds::new(
        Array2::eye(3),
        Array1::zeros(3),
        Array2::eye(3),
        Array1::zeros(3),
    )
    .unwrap();

    let result = layer.propagate_linear(&bounds).unwrap();
    // reps=1 returns Borrowed (no change)
    assert_eq!(result.lower_a, bounds.lower_a);
}

/// CROWN backward with reps=0 should error.
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_reps0_error() {
    let mut layer = TileLayer::new(0, 0);
    layer.set_input_shape(vec![3]);

    let bounds = LinearBounds::new(
        Array2::eye(3),
        Array1::zeros(3),
        Array2::eye(3),
        Array1::zeros(3),
    )
    .unwrap();

    let err = layer.propagate_linear(&bounds).unwrap_err();
    assert!(
        format!("{err}").contains("at least 1"),
        "Expected reps error, got: {err}"
    );
}

/// CROWN backward without input_shape should error.
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_no_input_shape_error() {
    let layer = TileLayer::new(0, 2);
    // Not calling set_input_shape

    let bounds = LinearBounds::new(
        Array2::eye(6),
        Array1::zeros(6),
        Array2::eye(6),
        Array1::zeros(6),
    )
    .unwrap();

    let err = layer.propagate_linear(&bounds).unwrap_err();
    assert!(
        format!("{err}").contains("input_shape"),
        "Expected input_shape error, got: {err}"
    );
}

/// CROWN backward soundness: IBP through tile then CROWN backward should
/// produce consistent results. For identity bounds through tile, the backward
/// should map each output position back to its source input position.
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_soundness_2d_axis0() {
    let input = bounded(&[2, 2], vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);

    let mut layer = TileLayer::new(0, 3);
    layer.set_input_shape(vec![2, 2]);

    // First verify IBP output shape
    let ibp_output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(ibp_output.shape(), &[6, 2]); // [2*3, 2]

    // CROWN backward with custom 1-row bounds that sums all outputs
    // This tests that backward correctly routes contributions
    let n_out = 12; // 6*2 flattened output
    let n_in = 4; // 2*2 flattened input

    // Use all-ones row to sum all outputs
    let ones = Array2::from_elem((1, n_out), 1.0_f32);
    let bounds = LinearBounds::new(ones.clone(), Array1::zeros(1), ones, Array1::zeros(1)).unwrap();

    let result_bounds = layer.propagate_linear_with_bounds(&bounds, &input).unwrap();

    // After backward: summing all outputs. Each input element appears 3 times (reps=3),
    // so each coefficient should be 3.
    assert_eq!(result_bounds.lower_a.shape(), &[1, n_in]);
    for &coeff in result_bounds.lower_a.iter() {
        assert_eq!(coeff, 3.0, "Each input appears 3 times, coeff should be 3");
    }
}

/// CROWN backward for tile along axis 1 on a 2D tensor.
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_2d_axis1() {
    let input = bounded(&[2, 3], vec![0.0; 6], vec![1.0; 6]);

    let mut layer = TileLayer::new(1, 2);
    layer.set_input_shape(vec![2, 3]);

    // Output shape: [2, 6], flattened = 12
    let n_out = 12;
    let n_in = 6;

    // Identity bounds for all flattened output positions
    let bounds = LinearBounds::new(
        Array2::eye(n_out),
        Array1::zeros(n_out),
        Array2::eye(n_out),
        Array1::zeros(n_out),
    )
    .unwrap();

    let result = layer.propagate_linear_with_bounds(&bounds, &input).unwrap();
    assert_eq!(result.lower_a.shape(), &[n_out, n_in]);

    // Row 0 (output[0,0]) maps to input[0,0]: new_a[0, 0] = 1
    assert_eq!(result.lower_a[[0, 0]], 1.0);
    // Row 3 (output[0,3]) maps to input[0,0] (first element of rep 1): new_a[3, 0] = 1
    assert_eq!(result.lower_a[[3, 0]], 1.0);
    // Row 1 (output[0,1]) maps to input[0,1]: new_a[1, 1] = 1
    assert_eq!(result.lower_a[[1, 1]], 1.0);
    // Row 4 (output[0,4]) maps to input[0,1] (rep 1): new_a[4, 1] = 1
    assert_eq!(result.lower_a[[4, 1]], 1.0);
}

/// Bounds mismatch: bounds num_inputs != output_size should error.
#[ntest::timeout(5000)]
#[test]
fn test_crown_backward_size_mismatch_error() {
    let input = bounded(&[3], vec![0.0; 3], vec![1.0; 3]);

    let mut layer = TileLayer::new(0, 2);
    layer.set_input_shape(vec![3]);

    // Output should be 6, but provide bounds with 4 inputs
    let bounds = LinearBounds::new(
        Array2::eye(4),
        Array1::zeros(4),
        Array2::eye(4),
        Array1::zeros(4),
    )
    .unwrap();

    let err = layer
        .propagate_linear_with_bounds(&bounds, &input)
        .unwrap_err();
    assert!(
        !format!("{err}").is_empty(),
        "Expected shape mismatch error"
    );
}

// =========================================================================
// Batched CROWN backward propagation (#287)
// =========================================================================

/// Batched CROWN backward: tile 1D [3] along axis 0 with reps=2.
/// Output [6], A shape [6, 6] (identity) -> backward -> [6, 3].
/// Each output position routes back to its source input position.
/// Input[0] receives coefficients from output[0] and output[3] (its replica).
#[ntest::timeout(5000)]
#[test]
fn test_batched_crown_backward_1d_reps2() {
    use crate::BatchedLinearBounds;

    let layer = TileLayer::new(0, 2);
    let pre_act = bounded(&[3], vec![0.0; 3], vec![1.0; 3]);

    // Identity bounds for 6-element tiled output
    let eye_6: Vec<f32> = (0..36)
        .map(|i| if i / 6 == i % 6 { 1.0 } else { 0.0 })
        .collect();
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[6, 6]), eye_6.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[6]), vec![0.0; 6]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[6, 6]), eye_6).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[6]), vec![0.0; 6]).unwrap(),
        vec![6],
        vec![6],
    )
    .unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();
    assert_eq!(result.lower_a.shape(), &[6, 3]);

    // Row 0 (output[0] = input[0], rep 0): coeff at input[0] = 1
    assert_eq!(result.lower_a[[0, 0]], 1.0);
    assert_eq!(result.lower_a[[0, 1]], 0.0);
    assert_eq!(result.lower_a[[0, 2]], 0.0);

    // Row 3 (output[3] = input[0], rep 1): coeff at input[0] = 1
    assert_eq!(result.lower_a[[3, 0]], 1.0);
    assert_eq!(result.lower_a[[3, 1]], 0.0);
    assert_eq!(result.lower_a[[3, 2]], 0.0);

    // Row 1 (output[1] = input[1]): coeff at input[1] = 1
    assert_eq!(result.lower_a[[1, 1]], 1.0);
    // Row 4 (output[4] = input[1], rep 1): coeff at input[1] = 1
    assert_eq!(result.lower_a[[4, 1]], 1.0);
}

/// Batched CROWN backward: tile 2D [2,3] along axis 0 with reps=3.
/// Output [6,3], flat=18. All-ones row should produce coefficients = 3
/// for each input position (each appears in 3 replicas).
#[ntest::timeout(5000)]
#[test]
fn test_batched_crown_backward_2d_axis0_reps3_allones() {
    use crate::BatchedLinearBounds;

    let layer = TileLayer::new(0, 3);
    let pre_act = bounded(&[2, 3], vec![0.0; 6], vec![1.0; 6]);

    // Single all-ones row: sums all 18 output positions
    let ones_18 = vec![1.0_f32; 18];
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 18]), ones_18.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 18]), ones_18).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        vec![18],
        vec![1],
    )
    .unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();
    assert_eq!(result.lower_a.shape(), &[1, 6]);

    // Each input position appears 3 times, so coefficient = 3.0
    for &coeff in result.lower_a.iter() {
        assert_eq!(coeff, 3.0, "Each input appears 3 times, coeff should be 3");
    }
}

/// Batched CROWN backward: tile along axis 1 on [2,3] with reps=2.
/// Output [2,6], flat=12. Identity bounds [12,12] -> backward -> [12,6].
/// Verifies axis-1 tiling maps output columns correctly.
#[ntest::timeout(5000)]
#[test]
fn test_batched_crown_backward_2d_axis1_reps2() {
    use crate::BatchedLinearBounds;

    let layer = TileLayer::new(1, 2);
    let pre_act = bounded(&[2, 3], vec![0.0; 6], vec![1.0; 6]);

    // Identity bounds for 12-element tiled output
    let eye_12: Vec<f32> = (0..144)
        .map(|i| if i / 12 == i % 12 { 1.0 } else { 0.0 })
        .collect();
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[12, 12]), eye_12.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[12]), vec![0.0; 12]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[12, 12]), eye_12).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[12]), vec![0.0; 12]).unwrap(),
        vec![12],
        vec![12],
    )
    .unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();
    assert_eq!(result.lower_a.shape(), &[12, 6]);

    // Output layout (flat): [2, 6] where axis 1 is tiled
    // output[0,0]=input[0,0], output[0,1]=input[0,1], output[0,2]=input[0,2],
    // output[0,3]=input[0,0], output[0,4]=input[0,1], output[0,5]=input[0,2]
    //
    // flat output 0 (=[0,0]) -> input flat 0 (=[0,0]): row 0, col 0 = 1
    assert_eq!(result.lower_a[[0, 0]], 1.0);
    // flat output 3 (=[0,3]) -> input flat 0 (=[0,0]): row 3, col 0 = 1
    assert_eq!(result.lower_a[[3, 0]], 1.0);
    // flat output 1 (=[0,1]) -> input flat 1 (=[0,1]): row 1, col 1 = 1
    assert_eq!(result.lower_a[[1, 1]], 1.0);
    // flat output 4 (=[0,4]) -> input flat 1 (=[0,1]): row 4, col 1 = 1
    assert_eq!(result.lower_a[[4, 1]], 1.0);
    // flat output 6 (=[1,0]) -> input flat 3 (=[1,0]): row 6, col 3 = 1
    assert_eq!(result.lower_a[[6, 3]], 1.0);
    // flat output 9 (=[1,3]) -> input flat 3 (=[1,0]): row 9, col 3 = 1
    assert_eq!(result.lower_a[[9, 3]], 1.0);
}

/// Batched CROWN backward: reps=1 should be no-op.
#[ntest::timeout(5000)]
#[test]
fn test_batched_crown_backward_reps1_noop() {
    use crate::BatchedLinearBounds;

    let layer = TileLayer::new(0, 1);
    let pre_act = bounded(&[3], vec![0.0; 3], vec![1.0; 3]);

    let eye_3: Vec<f32> = (0..9)
        .map(|i| if i / 3 == i % 3 { 1.0 } else { 0.0 })
        .collect();
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 3]), eye_3.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 3]), eye_3).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0; 3]).unwrap(),
        vec![3],
        vec![3],
    )
    .unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();
    assert_eq!(result.lower_a.shape(), &[3, 3]);
    // Identity preserved
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(result.lower_a[[i, j]], expected);
        }
    }
}

/// Batched CROWN backward: 3D tensor with batch dims in A.
/// Tile [2, 4] along axis 0 with reps=2 -> [4, 4], flat=16.
/// A shape [3, 2, 16] (3 batch dims, 2 out_dim, 16 in_dim) -> [3, 2, 8].
#[ntest::timeout(5000)]
#[test]
fn test_batched_crown_backward_with_batch_dims() {
    use crate::BatchedLinearBounds;

    let layer = TileLayer::new(0, 2);
    let pre_act = bounded(&[2, 4], vec![0.0; 8], vec![1.0; 8]);

    // A shape: [3, 2, 16]. Fill with ones to test summation.
    let a_vals = vec![1.0_f32; 3 * 2 * 16];
    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 16]), a_vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2, 16]), a_vals).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.0; 6]).unwrap(),
        vec![16],
        vec![3, 2],
    )
    .unwrap();

    let result = layer.propagate_linear_batched(&bounds, &pre_act).unwrap();
    assert_eq!(result.lower_a.shape(), &[3, 2, 8]);

    // All-ones input with reps=2: each input position receives 1+1 = 2
    for &coeff in result.lower_a.iter() {
        assert_eq!(coeff, 2.0, "Each input appears 2 times, coeff should be 2");
    }
}

/// Batched CROWN backward: reps=0 should error.
#[ntest::timeout(5000)]
#[test]
fn test_batched_crown_backward_reps0_error() {
    use crate::BatchedLinearBounds;

    let layer = TileLayer::new(0, 0);
    let pre_act = bounded(&[3], vec![0.0; 3], vec![1.0; 3]);

    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![1.0; 9]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![1.0; 9]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0; 3]).unwrap(),
        vec![3],
        vec![3],
    )
    .unwrap();

    let err = layer
        .propagate_linear_batched(&bounds, &pre_act)
        .unwrap_err();
    assert!(
        format!("{err}").contains("at least 1"),
        "Expected reps error, got: {err}"
    );
}

/// Batched CROWN backward: zero-valued dimension should error. (#2806)
#[ntest::timeout(5000)]
#[test]
fn test_batched_crown_backward_zero_dimension_error() {
    use crate::BatchedLinearBounds;

    let layer = TileLayer::new(0, 2);
    let l = ArrayD::from_shape_vec(IxDyn(&[2, 0]), vec![]).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(&[2, 0]), vec![]).unwrap();
    let pre_act = BoundedTensor::new(l, u).unwrap();

    let bounds = BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        vec![1],
        vec![1],
    )
    .unwrap();

    let err = layer
        .propagate_linear_batched(&bounds, &pre_act)
        .unwrap_err();
    assert!(
        format!("{err}").contains("zero-valued dimension"),
        "Expected zero-dimension error, got: {err}"
    );
}
