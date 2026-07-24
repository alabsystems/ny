// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::common::{
    batched_bounds_to_scalar, batched_bounds_to_scalar_multi_dim, finalize_decomposed_norm_bounds,
    scalar_bounds_to_batched, scalar_bounds_to_batched_multi_dim, validate_norm_against_fused_ibp,
    DecomposedNormFinalizeMetadata,
};
use super::tests_support::constant_batched_bounds;
use crate::LinearBounds;
use ndarray::{arr1, arr2, Array1, Array2, Array3, Ix1, Ix2, Ix3};
use ny_core::Result;
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};

fn finalize_result_tensors(
    result: &crate::BatchedLinearBounds,
) -> (Array3<f32>, Array3<f32>, Array2<f32>, Array2<f32>) {
    (
        result
            .lower_a()
            .clone()
            .into_dimensionality::<Ix3>()
            .unwrap(),
        result
            .upper_a()
            .clone()
            .into_dimensionality::<Ix3>()
            .unwrap(),
        result
            .lower_b()
            .clone()
            .into_dimensionality::<Ix2>()
            .unwrap(),
        result
            .upper_b()
            .clone()
            .into_dimensionality::<Ix2>()
            .unwrap(),
    )
}

fn flattened_row_bounds(
    bounds: &crate::BatchedLinearBounds,
) -> (Array2<f32>, Array2<f32>, Array1<f32>, Array1<f32>) {
    (
        bounds
            .lower_a()
            .clone()
            .into_dimensionality::<Ix2>()
            .unwrap(),
        bounds
            .upper_a()
            .clone()
            .into_dimensionality::<Ix2>()
            .unwrap(),
        bounds
            .lower_b()
            .clone()
            .into_dimensionality::<Ix1>()
            .unwrap(),
        bounds
            .upper_b()
            .clone()
            .into_dimensionality::<Ix1>()
            .unwrap(),
    )
}

#[ntest::timeout(10000)]
#[test]
fn test_finalize_nonfinite_rows_zeroed() -> Result<()> {
    let result = finalize_decomposed_norm_bounds(
        Array2::from_shape_vec((4, 3), (0..12).map(|v| v as f32 + 1.0).collect()).unwrap(),
        Array2::from_shape_vec((4, 3), (0..12).map(|v| v as f32 + 21.0).collect()).unwrap(),
        Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
        Array2::from_shape_vec((2, 2), vec![1.1, 1.2, 1.3, 1.4]).unwrap(),
        DecomposedNormFinalizeMetadata {
            lower_nonfinite_rows: &[true, false, false, true],
            upper_nonfinite_rows: &[false, true, false, true],
            total_rows: 4,
            out_dim: 2,
            n: 3,
            batch_dims: &[2],
            input_shape: &[2, 3],
            output_shape: &[2, 2],
            label: "test",
        },
    )?;
    let (lower_a, upper_a, lower_b, upper_b) = finalize_result_tensors(&result);

    assert_eq!(
        lower_a.slice(ndarray::s![0, .., ..]).to_owned(),
        arr2(&[[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]])
    );
    assert_eq!(
        upper_a.slice(ndarray::s![0, .., ..]).to_owned(),
        arr2(&[[21.0, 22.0, 23.0], [0.0, 0.0, 0.0]])
    );
    assert_eq!(
        lower_a.slice(ndarray::s![1, .., ..]).to_owned(),
        arr2(&[[7.0, 8.0, 9.0], [0.0, 0.0, 0.0]])
    );
    assert_eq!(
        upper_a.slice(ndarray::s![1, .., ..]).to_owned(),
        arr2(&[[27.0, 28.0, 29.0], [0.0, 0.0, 0.0]])
    );
    assert_eq!(lower_b[[0, 0]], f32::NEG_INFINITY);
    assert_eq!(upper_b[[0, 1]], f32::INFINITY);
    assert_eq!(lower_b[[1, 1]], f32::NEG_INFINITY);
    assert_eq!(upper_b[[1, 1]], f32::INFINITY);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_finalize_directed_rounding_bias() -> Result<()> {
    let result = finalize_decomposed_norm_bounds(
        arr2(&[[1.0, -2.0]]),
        arr2(&[[3.0, 4.0]]),
        arr2(&[[0.1_f64]]),
        arr2(&[[0.1_f64]]),
        DecomposedNormFinalizeMetadata {
            lower_nonfinite_rows: &[false],
            upper_nonfinite_rows: &[false],
            total_rows: 1,
            out_dim: 1,
            n: 2,
            batch_dims: &[],
            input_shape: &[2],
            output_shape: &[1],
            label: "rounding",
        },
    )?;

    let lower_b = result
        .lower_b()
        .clone()
        .into_dimensionality::<Ix1>()
        .unwrap();
    let upper_b = result
        .upper_b()
        .clone()
        .into_dimensionality::<Ix1>()
        .unwrap();
    assert_eq!(lower_b[0], next_down_f32(0.1));
    assert_eq!(upper_b[0], next_up_f32(0.1));
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_finalize_reshape_round_trip() -> Result<()> {
    let new_a_l =
        Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0]).unwrap();
    let new_a_u =
        Array2::from_shape_vec((4, 2), vec![5.0, 6.0, 7.0, 8.0, -5.0, -6.0, -7.0, -8.0]).unwrap();
    let result = finalize_decomposed_norm_bounds(
        new_a_l.clone(),
        new_a_u.clone(),
        arr2(&[[0.25_f64, -0.5_f64], [1.5_f64, -2.25_f64]]),
        arr2(&[[0.75_f64, -1.5_f64], [2.5_f64, -3.25_f64]]),
        DecomposedNormFinalizeMetadata {
            lower_nonfinite_rows: &[false, false, false, false],
            upper_nonfinite_rows: &[false, false, false, false],
            total_rows: 4,
            out_dim: 2,
            n: 2,
            batch_dims: &[2],
            input_shape: &[2, 2],
            output_shape: &[2, 2],
            label: "reshape",
        },
    )?;
    let (lower_a, upper_a, _, _) = finalize_result_tensors(&result);
    assert_eq!(
        lower_a.slice(ndarray::s![0, .., ..]).to_owned(),
        new_a_l.slice(ndarray::s![0..2, ..]).to_owned()
    );
    assert_eq!(
        lower_a.slice(ndarray::s![1, .., ..]).to_owned(),
        new_a_l.slice(ndarray::s![2..4, ..]).to_owned()
    );
    assert_eq!(
        upper_a.slice(ndarray::s![0, .., ..]).to_owned(),
        new_a_u.slice(ndarray::s![0..2, ..]).to_owned()
    );
    assert_eq!(
        upper_a.slice(ndarray::s![1, .., ..]).to_owned(),
        new_a_u.slice(ndarray::s![2..4, ..]).to_owned()
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_all_rows_within_ibp() -> Result<()> {
    let fallback = constant_batched_bounds(
        Array2::zeros((4, 2)),
        arr1(&[-1.0, 0.0, -2.0, 1.0]),
        Array2::zeros((4, 2)),
        arr1(&[1.0, 0.5, 2.0, 1.5]),
        2,
    );
    let mut candidate = constant_batched_bounds(
        Array2::zeros((4, 2)),
        arr1(&[-0.5, 0.1, -1.5, 1.1]),
        Array2::zeros((4, 2)),
        arr1(&[0.8, 0.4, 1.9, 1.4]),
        2,
    );
    let x_ibp = BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[0.0, 0.0]).into_dyn())?;
    let fused_ibp = BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[0.0, 0.0]).into_dyn())?;
    let original = candidate.clone();

    let fallback_rows =
        validate_norm_against_fused_ibp(&mut candidate, &fallback, &fused_ibp, &x_ibp, 4, 2)?;

    assert_eq!(fallback_rows, 0);
    assert_eq!(candidate.lower_b(), original.lower_b());
    assert_eq!(candidate.upper_b(), original.upper_b());
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_some_rows_escape_ibp() -> Result<()> {
    let fallback = constant_batched_bounds(
        Array2::zeros((4, 2)),
        arr1(&[-1.0, 0.0, -2.0, 1.0]),
        Array2::zeros((4, 2)),
        arr1(&[1.0, 0.5, 2.0, 1.5]),
        2,
    );
    let mut candidate = constant_batched_bounds(
        arr2(&[[0.5, -0.5], [0.25, 0.75], [1.0, 1.0], [0.2, -0.1]]),
        arr1(&[-2.0, 0.1, -1.5, 1.1]),
        arr2(&[[0.5, -0.5], [0.25, 0.75], [1.0, 1.0], [0.2, -0.1]]),
        arr1(&[0.8, 0.4, 2.5, 1.4]),
        2,
    );
    let x_ibp = BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[0.0, 0.0]).into_dyn())?;
    let fused_ibp = BoundedTensor::new(arr1(&[0.0, 0.0]).into_dyn(), arr1(&[0.0, 0.0]).into_dyn())?;
    let fallback_interval = fallback.concretize_sound(&fused_ibp)?;
    let expected_lower = fallback_interval
        .lower()
        .clone()
        .into_dimensionality::<Ix1>()
        .unwrap();
    let expected_upper = fallback_interval
        .upper()
        .clone()
        .into_dimensionality::<Ix1>()
        .unwrap();

    let fallback_rows =
        validate_norm_against_fused_ibp(&mut candidate, &fallback, &fused_ibp, &x_ibp, 4, 2)?;
    let (lower_a, upper_a, lower_b, upper_b) = flattened_row_bounds(&candidate);

    assert_eq!(fallback_rows, 2);
    assert_eq!(lower_a.row(0).to_owned(), arr1(&[0.0, 0.0]));
    assert_eq!(upper_a.row(0).to_owned(), arr1(&[0.0, 0.0]));
    assert_eq!(lower_b[0], expected_lower[0]);
    assert_eq!(upper_b[0], expected_upper[0]);
    assert_eq!(lower_a.row(2).to_owned(), arr1(&[0.0, 0.0]));
    assert_eq!(upper_a.row(2).to_owned(), arr1(&[0.0, 0.0]));
    assert_eq!(lower_b[2], expected_lower[2]);
    assert_eq!(upper_b[2], expected_upper[2]);
    assert_eq!(lower_a.row(1).to_owned(), arr1(&[0.25, 0.75]));
    assert_eq!(upper_a.row(3).to_owned(), arr1(&[0.2, -0.1]));
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_validate_returns_correct_count() -> Result<()> {
    let fallback = constant_batched_bounds(
        Array2::zeros((4, 1)),
        arr1(&[-1.0, -2.0, -3.0, -4.0]),
        Array2::zeros((4, 1)),
        arr1(&[1.0, 2.0, 3.0, 4.0]),
        1,
    );
    let mut candidate = constant_batched_bounds(
        Array2::zeros((4, 1)),
        arr1(&[-1.5, -1.5, -3.5, -3.5]),
        Array2::zeros((4, 1)),
        arr1(&[0.5, 2.5, 2.5, 4.5]),
        1,
    );
    let x_ibp = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[0.0]).into_dyn())?;
    let fused_ibp = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[0.0]).into_dyn())?;

    let fallback_rows =
        validate_norm_against_fused_ibp(&mut candidate, &fallback, &fused_ibp, &x_ibp, 4, 1)?;

    assert_eq!(fallback_rows, 4);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_scalar_batched_round_trip() -> Result<()> {
    let bounds = LinearBounds::new(
        arr2(&[[1.0, -0.5, 0.25], [0.75, 0.0, -1.25]]),
        arr1(&[-0.1, 0.2]),
        arr2(&[[1.5, -0.25, 0.5], [0.5, 0.25, -1.0]]),
        arr1(&[0.4, -0.3]),
    )?;

    let recovered = batched_bounds_to_scalar(&scalar_bounds_to_batched(&bounds)?)?;
    assert_eq!(recovered.lower_a(), bounds.lower_a());
    assert_eq!(recovered.lower_b(), bounds.lower_b());
    assert_eq!(recovered.upper_a(), bounds.upper_a());
    assert_eq!(recovered.upper_b(), bounds.upper_b());
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_multi_dim_scalar_batched_round_trip() -> Result<()> {
    let bounds = LinearBounds::new(
        arr2(&[
            [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
            [0.5, -0.5, 0.75, 1.25, -1.5, 2.0],
        ]),
        arr1(&[-0.2, 0.3]),
        arr2(&[
            [1.5, 2.5, 3.5, -0.5, -1.5, -2.5],
            [0.75, -0.25, 1.0, 1.5, -1.25, 2.25],
        ]),
        arr1(&[0.4, 0.8]),
    )?;

    let batched = scalar_bounds_to_batched_multi_dim(&bounds, 2, 3)?;
    let recovered =
        batched_bounds_to_scalar_multi_dim(&batched, bounds.lower_b(), bounds.upper_b())?;

    assert_eq!(recovered.lower_a(), bounds.lower_a());
    assert_eq!(recovered.lower_b(), bounds.lower_b());
    assert_eq!(recovered.upper_a(), bounds.upper_a());
    assert_eq!(recovered.upper_b(), bounds.upper_b());
    Ok(())
}
