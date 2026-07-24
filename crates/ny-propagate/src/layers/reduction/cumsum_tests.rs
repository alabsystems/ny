// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::cumsum::CumsumLayer;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array1, Array2, ArrayD, Ix1, Ix2, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

fn bounded_from_1d(lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(&[lower.len()]), lower.to_vec()).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(&[upper.len()]), upper.to_vec()).unwrap();
    BoundedTensor::new(l, u).unwrap()
}

fn bounded_from_shape(shape: &[usize], lower: &[f32], upper: &[f32]) -> BoundedTensor {
    let l = ArrayD::from_shape_vec(IxDyn(shape), lower.to_vec()).unwrap();
    let u = ArrayD::from_shape_vec(IxDyn(shape), upper.to_vec()).unwrap();
    BoundedTensor::new(l, u).unwrap()
}

fn assert_batched_matrix_matches(result: &BatchedLinearBounds, expected: &[[f32; 3]; 3]) {
    for batch in 0..2 {
        for (row, expected_row) in expected.iter().enumerate().take(3) {
            for (col, expected_value) in expected_row.iter().enumerate().take(3) {
                let actual_lower = result.lower_a()[[batch, row, col]];
                let actual_upper = result.upper_a()[[batch, row, col]];
                assert!(
                    (actual_lower - *expected_value).abs() < 1e-6,
                    "lower_a[{batch},{row},{col}] = {actual_lower} expected {expected_value}"
                );
                assert!(
                    (actual_upper - *expected_value).abs() < 1e-6,
                    "upper_a[{batch},{row},{col}] = {actual_upper} expected {expected_value}"
                );
            }
        }
    }
}

fn linear_from_flat_batched(bounds: BatchedLinearBounds) -> LinearBounds {
    let (lower_a, lower_b, upper_a, upper_b, _, _) = bounds.into_parts();
    LinearBounds::new_or_conservative(
        lower_a.into_dimensionality::<Ix2>().unwrap(),
        lower_b.into_dimensionality::<Ix1>().unwrap(),
        upper_a.into_dimensionality::<Ix2>().unwrap(),
        upper_b.into_dimensionality::<Ix1>().unwrap(),
    )
    .unwrap()
}

#[test]
fn test_cumsum_crown_backward_forward_exclusive() {
    let layer = CumsumLayer::new(0, true, false);
    let input = bounded_from_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
    let bounds = LinearBounds::new_or_conservative(
        Array2::<f32>::eye(3),
        Array1::<f32>::zeros(3),
        Array2::<f32>::eye(3),
        Array1::<f32>::zeros(3),
    )
    .unwrap();

    let result = layer.propagate_linear_with_bounds(&bounds, &input).unwrap();
    let expected = ndarray::array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
    assert_eq!(result.lower_a(), &expected);
    assert_eq!(result.upper_a(), &expected);
}

#[test]
fn test_cumsum_crown_backward_reverse_exclusive() {
    // Reverse exclusive cumsum: y[i] = sum(x[i+1..T])
    // J[i, j] = 1 if j > i (strictly upper-triangular)
    // J^T[i, j] = 1 if i > j (strictly lower-triangular)
    // new_A = A @ J: new_A[row, col] = sum_{k < col} A[row, k] (shifted prefix sum)
    let layer = CumsumLayer::new(0, true, true);
    let input = bounded_from_1d(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
    let bounds = LinearBounds::new_or_conservative(
        Array2::<f32>::eye(3),
        Array1::<f32>::zeros(3),
        Array2::<f32>::eye(3),
        Array1::<f32>::zeros(3),
    )
    .unwrap();

    let result = layer.propagate_linear_with_bounds(&bounds, &input).unwrap();

    // J = [[0,1,1],[0,0,1],[0,0,0]] (strictly upper-triangular)
    // new_A = I @ J = J
    let expected = ndarray::array![[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]];
    assert_eq!(result.lower_a(), &expected);
    assert_eq!(result.upper_a(), &expected);
}

#[test]
fn test_cumsum_batched_crown_forward_inclusive_exact_math() {
    let layer = CumsumLayer::new(-1, false, false);
    let input = bounded_from_shape(
        &[2, 3],
        &[-1.0, 0.0, 1.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 1.0, 1.5, 2.0],
    );
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();

    let result = layer.propagate_linear_batched(&bounds, &input).unwrap();
    let expected = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]];
    assert_batched_matrix_matches(&result, &expected);
}

#[test]
fn test_cumsum_batched_crown_forward_exclusive_exact_math() {
    let layer = CumsumLayer::new(-1, true, false);
    let input = bounded_from_shape(
        &[2, 3],
        &[-1.0, 0.0, 1.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 1.0, 1.5, 2.0],
    );
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();

    let result = layer.propagate_linear_batched(&bounds, &input).unwrap();
    let expected = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
    assert_batched_matrix_matches(&result, &expected);
}

#[test]
fn test_cumsum_batched_crown_reverse_inclusive_exact_math() {
    let layer = CumsumLayer::new(-1, false, true);
    let input = bounded_from_shape(
        &[2, 3],
        &[-1.0, 0.0, 1.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 1.0, 1.5, 2.0],
    );
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();

    let result = layer.propagate_linear_batched(&bounds, &input).unwrap();
    let expected = [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]];
    assert_batched_matrix_matches(&result, &expected);
}

#[test]
fn test_cumsum_batched_crown_reverse_exclusive_exact_math() {
    let layer = CumsumLayer::new(-1, true, true);
    let input = bounded_from_shape(
        &[2, 3],
        &[-1.0, 0.0, 1.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 1.0, 1.5, 2.0],
    );
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();

    let result = layer.propagate_linear_batched(&bounds, &input).unwrap();
    let expected = [[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]];
    assert_batched_matrix_matches(&result, &expected);
}

#[test]
fn test_cumsum_batched_crown_matches_unbatched_forward_exclusive() {
    let layer = CumsumLayer::new(-1, true, false);
    let input = bounded_from_shape(
        &[2, 3],
        &[-1.0, 0.5, 1.0, -0.5, 0.0, 2.0],
        &[0.0, 1.5, 2.0, 0.5, 1.0, 3.0],
    );
    let bounds = BatchedLinearBounds::new_or_conservative(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 3]),
            vec![
                1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 0.0, 1.5, -2.0, -1.0, 0.0, 2.0, 3.0, 1.0, -0.5, 2.0,
                -3.0, 4.0,
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6]).unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3, 3]),
            vec![
                1.5, 2.5, 3.5, 4.5, -0.5, 1.0, 0.25, 1.75, -1.5, -0.5, 0.5, 2.5, 3.5, 1.5, 0.0,
                2.5, -2.5, 4.5,
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.2, -0.1, 0.4, -0.3, 0.6, -0.5]).unwrap(),
        vec![2, 3],
        vec![2, 3],
    )
    .unwrap();

    let batched_result = layer.propagate_linear_batched(&bounds, &input).unwrap();
    let flat_bounds = linear_from_flat_batched(bounds.flatten_to_block_diagonal().unwrap());
    let dense_result = layer
        .propagate_linear_with_bounds(&flat_bounds, &input)
        .unwrap();
    let batched_flat = batched_result.flatten_to_block_diagonal().unwrap();

    assert_eq!(
        batched_flat
            .lower_a()
            .view()
            .into_dimensionality::<Ix2>()
            .unwrap(),
        dense_result.lower_a().view()
    );
    assert_eq!(
        batched_flat
            .upper_a()
            .view()
            .into_dimensionality::<Ix2>()
            .unwrap(),
        dense_result.upper_a().view()
    );
    assert_eq!(
        batched_flat
            .lower_b()
            .view()
            .into_dimensionality::<Ix1>()
            .unwrap(),
        dense_result.lower_b().view()
    );
    assert_eq!(
        batched_flat
            .upper_b()
            .view()
            .into_dimensionality::<Ix1>()
            .unwrap(),
        dense_result.upper_b().view()
    );
}

#[test]
fn test_cumsum_batched_crown_non_last_axis_rejected() {
    let layer = CumsumLayer::new(0, false, false);
    let input = bounded_from_shape(
        &[2, 3],
        &[-1.0, 0.0, 1.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 1.0, 1.5, 2.0],
    );
    let bounds = BatchedLinearBounds::identity(&[2, 3]).unwrap();

    let err = layer.propagate_linear_batched(&bounds, &input).unwrap_err();
    assert!(matches!(err, NyError::UnsupportedOp(_)));
    assert!(
        err.to_string().contains("last-axis"),
        "expected last-axis rejection, got {err}"
    );
}

#[test]
fn test_cumsum_batched_crown_flat_block_diagonal_rejected() -> Result<()> {
    let layer = CumsumLayer::new(-1, false, false);
    let input = bounded_from_shape(
        &[2, 3],
        &[-1.0, 0.0, 1.0, 0.5, 1.0, 1.5],
        &[0.0, 1.0, 2.0, 1.0, 1.5, 2.0],
    );
    let bounds = BatchedLinearBounds::identity(&[2, 3])?.flatten_to_block_diagonal()?;

    let err = layer.propagate_linear_batched(&bounds, &input).unwrap_err();
    assert!(matches!(err, NyError::UnsupportedOp(_)));
    assert!(
        err.to_string().contains("shape-preserving grouped bounds"),
        "expected grouped-layout rejection, got {err}"
    );
    Ok(())
}

/// #vnncomp-aw-soundness self-audit regression: the batched cumsum CROWN backward must carry
/// a certified coefficient error covering the f32 partial-sum cancellation that the
/// already-fixed non-batched path carries. Inclusive forward cumsum over a fiber
/// [-2^24, 2^24, 1]: the suffix sum stores col0 = fl(2^24 + (-2^24)) = 0 while the TRUE coeff
/// is -2^24+2^24+1 = 1 — a dropped unit coefficient. Pre-fix new_or_conservative left
/// lower_a_err = None (trusted exact = false proof); the fix attaches an err >= the drop.
#[test]
fn cumsum_batched_crown_carries_cancellation_coeff_error() {
    let two24 = 16_777_216.0_f32; // 2^24
    let input = bounded_from_shape(&[1, 3], &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
    let fiber = vec![-two24, two24, 1.0, -two24, two24, 1.0, -two24, two24, 1.0];
    let a = ArrayD::from_shape_vec(IxDyn(&[1, 3, 3]), fiber).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![0.0, 0.0, 0.0]).unwrap();
    let bounds = BatchedLinearBounds::new_or_conservative(
        a.clone(),
        b.clone(),
        a,
        b,
        vec![1, 3],
        vec![1, 3],
    )
    .unwrap();

    let layer = CumsumLayer::new(-1, false, false); // inclusive forward
    let result = layer.propagate_linear_batched(&bounds, &input).unwrap();

    let lerr = result
        .lower_a_err
        .as_ref()
        .expect("batched cumsum must attach a certified coeff error under cancellation");
    // col0 of each output fiber is where the unit coefficient was dropped (stored 0 vs true 1).
    for out in 0..3 {
        let e = lerr[[0, out, 0]] as f64;
        assert!(
            e >= 1.0,
            "coeff err at the cancellation cell [0,{out},0] = {e} must cover the dropped coeff 1.0"
        );
    }
}
