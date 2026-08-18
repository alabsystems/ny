// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::tests::assert_close;
use ndarray::{arr1, Array2, ArrayBase, ArrayD, Data, Dimension, IxDyn};
use ny_core::Result;

const TOL: f32 = 1e-5;

#[inline]
fn soundness_tol(z_true: f32) -> f32 {
    (4.0 * f32::EPSILON * z_true.abs().max(1.0)).max(TOL)
}

fn make_bounded(shape: &[usize], lower: Vec<f32>, upper: Vec<f32>) -> Result<BoundedTensor> {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(shape), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(shape), upper).unwrap(),
    )
}

fn make_contiguous_equivalent(arr: &ArrayD<f32>) -> ArrayD<f32> {
    ArrayD::from_shape_vec(IxDyn(arr.shape()), arr.iter().copied().collect()).unwrap()
}

fn make_non_contiguous_matrix_3x2(values: Vec<f32>) -> ArrayD<f32> {
    let base = ArrayD::from_shape_vec(IxDyn(&[2, 3]), values).unwrap();
    let noncontiguous = base.view().reversed_axes().to_owned();
    assert_eq!(noncontiguous.shape(), &[3, 2]);
    assert!(
        noncontiguous.as_slice().is_none(),
        "precondition: reversed_axes().to_owned() should preserve non-standard layout"
    );
    noncontiguous
}

fn make_non_contiguous_matrix_2x2(values: Vec<f32>) -> ArrayD<f32> {
    let base = ArrayD::from_shape_vec(IxDyn(&[2, 2]), values).unwrap();
    let noncontiguous = base.view().reversed_axes().to_owned();
    assert_eq!(noncontiguous.shape(), &[2, 2]);
    assert!(
        noncontiguous.as_slice().is_none(),
        "precondition: reversed_axes().to_owned() should preserve non-standard layout"
    );
    noncontiguous
}

fn make_non_contiguous_cube_2x2x2(values: Vec<f32>) -> ArrayD<f32> {
    let base = ArrayD::from_shape_vec(IxDyn(&[2, 2, 2]), values).unwrap();
    let noncontiguous = base.view().permuted_axes(IxDyn(&[1, 0, 2])).to_owned();
    assert_eq!(noncontiguous.shape(), &[2, 2, 2]);
    assert!(
        noncontiguous.as_slice().is_none(),
        "precondition: permuted_axes().to_owned() should preserve non-standard layout"
    );
    noncontiguous
}

fn make_non_contiguous_tensor_pair_3x2(
    lower: Vec<f32>,
    upper: Vec<f32>,
) -> Result<(BoundedTensor, BoundedTensor)> {
    let noncontiguous_lower = make_non_contiguous_matrix_3x2(lower);
    let noncontiguous_upper = make_non_contiguous_matrix_3x2(upper);
    let contiguous_lower = make_contiguous_equivalent(&noncontiguous_lower);
    let contiguous_upper = make_contiguous_equivalent(&noncontiguous_upper);

    Ok((
        BoundedTensor::new(noncontiguous_lower, noncontiguous_upper)?,
        BoundedTensor::new(contiguous_lower, contiguous_upper)?,
    ))
}

fn make_non_contiguous_tensor_pair_2x2(
    lower: Vec<f32>,
    upper: Vec<f32>,
) -> Result<(BoundedTensor, BoundedTensor)> {
    let noncontiguous_lower = make_non_contiguous_matrix_2x2(lower);
    let noncontiguous_upper = make_non_contiguous_matrix_2x2(upper);
    let contiguous_lower = make_contiguous_equivalent(&noncontiguous_lower);
    let contiguous_upper = make_contiguous_equivalent(&noncontiguous_upper);

    Ok((
        BoundedTensor::new(noncontiguous_lower, noncontiguous_upper)?,
        BoundedTensor::new(contiguous_lower, contiguous_upper)?,
    ))
}

fn make_batched_non_contiguous_inputs_4247() -> Result<(
    (BoundedTensor, BoundedTensor),
    (BoundedTensor, BoundedTensor),
)> {
    let a_pair = make_non_contiguous_tensor_pair_2x2(
        vec![1.0, 2.0, 10.0, 20.0],
        vec![3.0, 4.0, 30.0, 40.0],
    )?;
    let b_pair = make_non_contiguous_tensor_pair_2x2(
        vec![5.0, 6.0, 50.0, 60.0],
        vec![7.0, 8.0, 70.0, 80.0],
    )?;

    assert!(
        a_pair.0.lower().as_slice().is_none(),
        "precondition: lhs lower bounds must be non-contiguous"
    );
    assert!(
        b_pair.0.upper().as_slice().is_none(),
        "precondition: rhs upper bounds must be non-contiguous"
    );

    Ok((a_pair, b_pair))
}

fn make_batched_non_contiguous_bounds_pair_4247() -> (BatchedLinearBounds, BatchedLinearBounds) {
    let lower_a_noncontiguous =
        make_non_contiguous_cube_2x2x2(vec![1.0, 0.0, 0.25, -0.5, 0.0, 1.0, 0.75, -1.25]);
    let upper_a_noncontiguous =
        make_non_contiguous_cube_2x2x2(vec![1.2, -0.25, 0.4, 0.3, -0.1, 0.8, 0.6, -0.75]);
    let lower_b_noncontiguous = make_non_contiguous_matrix_2x2(vec![0.1, -0.2, 0.3, -0.4]);
    let upper_b_noncontiguous = make_non_contiguous_matrix_2x2(vec![0.5, 0.0, 0.8, 0.2]);

    assert!(
        lower_a_noncontiguous.as_slice().is_none(),
        "precondition: lower_a coefficients must be non-contiguous"
    );
    assert!(
        lower_b_noncontiguous.as_slice().is_none(),
        "precondition: lower_b bias must be non-contiguous"
    );

    let noncontiguous_bounds = BatchedLinearBounds::from_parts_unchecked(
        lower_a_noncontiguous.clone(),
        lower_b_noncontiguous.clone(),
        upper_a_noncontiguous.clone(),
        upper_b_noncontiguous.clone(),
        vec![2, 2],
        vec![2, 2],
    );
    let contiguous_bounds = BatchedLinearBounds::from_parts_unchecked(
        make_contiguous_equivalent(&lower_a_noncontiguous),
        make_contiguous_equivalent(&lower_b_noncontiguous),
        make_contiguous_equivalent(&upper_a_noncontiguous),
        make_contiguous_equivalent(&upper_b_noncontiguous),
        vec![2, 2],
        vec![2, 2],
    );

    (noncontiguous_bounds, contiguous_bounds)
}

fn assert_array_values_close<S1, S2, D>(
    actual: &ArrayBase<S1, D>,
    expected: &ArrayBase<S2, D>,
    context: &str,
) where
    S1: Data<Elem = f32>,
    S2: Data<Elem = f32>,
    D: Dimension,
{
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{context}: shape mismatch actual={:?} expected={:?}",
        actual.shape(),
        expected.shape()
    );

    for (idx, (actual_value, expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (*actual_value - *expected_value).abs();
        assert!(
            delta <= TOL,
            "{context}[{idx}] mismatch: actual={actual_value}, expected={expected_value}, delta={delta}"
        );
    }
}

fn assert_linear_bounds_close(actual: &LinearBounds, expected: &LinearBounds, context: &str) {
    assert_array_values_close(
        &actual.lower_a,
        &expected.lower_a,
        &format!("{context} lower_a"),
    );
    assert_array_values_close(
        &actual.upper_a,
        &expected.upper_a,
        &format!("{context} upper_a"),
    );
    assert_array_values_close(
        &actual.lower_b,
        &expected.lower_b,
        &format!("{context} lower_b"),
    );
    assert_array_values_close(
        &actual.upper_b,
        &expected.upper_b,
        &format!("{context} upper_b"),
    );
}

fn assert_batched_linear_bounds_close(
    actual: &BatchedLinearBounds,
    expected: &BatchedLinearBounds,
    context: &str,
) {
    assert_array_values_close(
        &actual.lower_a,
        &expected.lower_a,
        &format!("{context} lower_a"),
    );
    assert_array_values_close(
        &actual.upper_a,
        &expected.upper_a,
        &format!("{context} upper_a"),
    );
    assert_array_values_close(
        &actual.lower_b,
        &expected.lower_b,
        &format!("{context} lower_b"),
    );
    assert_array_values_close(
        &actual.upper_b,
        &expected.upper_b,
        &format!("{context} upper_b"),
    );
}

fn make_alpha_broadcast_case_3499() -> Result<(
    MulBinaryLayer,
    BoundedTensor,
    BoundedTensor,
    LinearBounds,
    Array2<f32>,
)> {
    let layer = MulBinaryLayer;
    let a = make_bounded(
        &[2, 3],
        vec![-1.0_f32, 0.25, -0.5, 0.5, -2.0, 0.1],
        vec![2.0_f32, 1.0, 0.75, 2.5, -0.25, 1.4],
    )?;
    let b = make_bounded(&[2, 1], vec![0.2_f32, -1.5], vec![1.1_f32, 0.4])?;

    let weights =
        Array2::from_shape_vec((1, 6), vec![1.0_f32, -0.5, 0.25, -1.0, 0.75, 2.0]).unwrap();
    let bounds = LinearBounds::new(weights.clone(), arr1(&[0.0]), weights, arr1(&[0.0]))?;
    let alphas = Array2::from_shape_vec(
        (2, 6),
        vec![
            0.1_f32, 0.7, 0.4, 0.9, 0.2, 0.6, 0.8, 0.3, 0.5, 0.1, 0.9, 0.4,
        ],
    )
    .unwrap();

    Ok((layer, a, b, bounds, alphas))
}

fn expected_reduced_rhs_coefficients_3499(
    bounds: &LinearBounds,
    a: &BoundedTensor,
    b: &BoundedTensor,
    alphas: &Array2<f32>,
) -> ([f32; 2], [f32; 2]) {
    let a_lower = a.lower().as_slice().unwrap();
    let a_upper = a.upper().as_slice().unwrap();
    let b_lower = b.lower().as_slice().unwrap();
    let b_upper = b.upper().as_slice().unwrap();

    // [2, 3] * [2, 1] broadcasts the RHS across time, so output positions
    // [0,1,2] reduce into RHS column 0 and [3,4,5] reduce into column 1.
    let rhs_column_for_output = [0usize, 0, 0, 1, 1, 1];
    let mut expected_lower = [0.0_f32; 2];
    let mut expected_upper = [0.0_f32; 2];

    for j in 0..6 {
        let rhs_idx = rhs_column_for_output[j];
        let lx = a_lower[j];
        let ux = a_upper[j];
        let ly = b_lower[rhs_idx];
        let uy = b_upper[rhs_idx];
        let r_l = alphas[[0, j]].clamp(0.0, 1.0);
        let r_u = alphas[[1, j]].clamp(0.0, 1.0);
        let (_alpha_l, beta_l, _ny_l, _alpha_u, beta_u, _ny_u) =
            MulBinaryLayer::compute_interpolated_coefficients(lx, ux, ly, uy, r_l, r_u);

        let w_lower = bounds.lower_a()[[0, j]];
        let w_upper = bounds.upper_a()[[0, j]];

        expected_lower[rhs_idx] += if w_lower >= 0.0 {
            w_lower * beta_l
        } else {
            w_lower * beta_u
        };
        expected_upper[rhs_idx] += if w_upper >= 0.0 {
            w_upper * beta_u
        } else {
            w_upper * beta_l
        };
    }

    (expected_lower, expected_upper)
}

fn assert_se_block_broadcast_corners_3499(
    bounds: &LinearBounds,
    a: &BoundedTensor,
    b: &BoundedTensor,
    lower: f32,
    upper: f32,
) {
    let a_lower = a.lower().as_slice().unwrap();
    let a_upper = a.upper().as_slice().unwrap();
    let b_lower = b.lower().as_slice().unwrap();
    let b_upper = b.upper().as_slice().unwrap();
    let weights = bounds.lower_a().row(0);

    for mask in 0..(1 << 8) {
        let mut x = [0.0_f32; 6];
        for idx in 0..6 {
            x[idx] = if (mask >> idx) & 1 == 0 {
                a_lower[idx]
            } else {
                a_upper[idx]
            };
        }
        let y0 = if (mask >> 6) & 1 == 0 {
            b_lower[0]
        } else {
            b_upper[0]
        };
        let y1 = if (mask >> 7) & 1 == 0 {
            b_lower[1]
        } else {
            b_upper[1]
        };
        let z_true = weights[0] * x[0] * y0
            + weights[1] * x[1] * y0
            + weights[2] * x[2] * y0
            + weights[3] * x[3] * y1
            + weights[4] * x[4] * y1
            + weights[5] * x[5] * y1;

        assert!(
            lower <= z_true + soundness_tol(z_true),
            "#3499 alpha broadcast lower unsound for mask {mask:#010b}: lower={lower}, true={z_true}"
        );
        assert!(
            upper >= z_true - soundness_tol(z_true),
            "#3499 alpha broadcast upper unsound for mask {mask:#010b}: upper={upper}, true={z_true}"
        );
    }
}

// ---- IBP same-shape ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_positive_times_positive() -> Result<()> {
    let layer = MulBinaryLayer;
    // [1, 2] * [3, 4] = [3, 8]  (both positive, no sign flipping)
    let a = BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0, 1.0]).into_dyn(), arr1(&[4.0, 3.0]).into_dyn())?;
    let out = layer.propagate_ibp_binary(&a, &b)?;
    // lower = min(1*2, 1*4, 3*2, 3*4) = 2, min(2*1, 2*3, 4*1, 4*3) = 2
    assert_close(out.lower()[[0]], 2.0, TOL);
    assert_close(out.lower()[[1]], 2.0, TOL);
    // upper = max(1*2, 1*4, 3*2, 3*4) = 12, max(2*1, 2*3, 4*1, 4*3) = 12
    assert_close(out.upper()[[0]], 12.0, TOL);
    assert_close(out.upper()[[1]], 12.0, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_mixed_sign() -> Result<()> {
    let layer = MulBinaryLayer;
    // x in [-1, 1], y in [-2, 3]
    // Products: (-1)*(-2)=2, (-1)*(3)=-3, (1)*(-2)=-2, (1)*(3)=3
    // lower = -3, upper = 3
    let a = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let out = layer.propagate_ibp_binary(&a, &b)?;
    assert_close(out.lower()[[0]], -3.0, TOL);
    assert_close(out.upper()[[0]], 3.0, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_point_times_point() -> Result<()> {
    let layer = MulBinaryLayer;
    // Point inputs: 3 * 4 = 12
    let a = BoundedTensor::new(arr1(&[3.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[4.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;
    let out = layer.propagate_ibp_binary(&a, &b)?;
    assert_close(out.lower()[[0]], 12.0, TOL);
    assert_close(out.upper()[[0]], 12.0, TOL);
    Ok(())
}

// ---- IBP broadcasting ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_broadcast_scalar_times_vector() -> Result<()> {
    let layer = MulBinaryLayer;
    // scalar [2,3] * vector [1,2; 3,4] with broadcasting
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    )?;
    let b = BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn())?;
    let out = layer.propagate_ibp_binary(&a, &b)?;
    assert_eq!(out.shape(), &[2]);
    // dim 0: [2,3]*[1,3] -> lower=2, upper=9
    assert_close(out.lower()[[0]], 2.0, TOL);
    assert_close(out.upper()[[0]], 9.0, TOL);
    // dim 1: [2,3]*[2,4] -> lower=4, upper=12
    assert_close(out.lower()[[1]], 4.0, TOL);
    assert_close(out.upper()[[1]], 12.0, TOL);
    Ok(())
}

// ---- McCormick plane selection ----

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_lower_positive_weight() {
    // z = x*y with x in [0, 2], y in [0, 3]
    // L1: y_l*x + x_l*y - x_l*y_l = 0*x + 0*y - 0 = (0, 0, 0), eval at (1,1.5) = 0
    // L2: y_u*x + x_u*y - x_u*y_u = 3*x + 2*y - 6 = (3, 2, -6), eval at (1,1.5) = 3+3-6 = 0
    // Both evaluate to 0 at midpoint, so either is valid.
    let (ax, ay, c) = select_mccormick_plane(0.0, 2.0, 0.0, 3.0, 1.0, 1.5, 1.0, BoundDir::Lower);
    // Verify it's a valid lower bound at corners
    for &x in &[0.0, 2.0] {
        for &y in &[0.0, 3.0] {
            let z_true = x * y;
            let z_plane = ax * x + ay * y + c;
            assert!(
                z_plane <= z_true + 1e-5,
                "McCormick lower plane should be <= true at ({x},{y}): plane={z_plane}, true={z_true}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_upper_positive_weight() {
    // z = x*y with x in [0, 2], y in [0, 3]
    // U1: y_u*x + x_l*y - x_l*y_u = 3*x + 0*y - 0 = (3, 0, 0)
    // U2: y_l*x + x_u*y - x_u*y_l = 0*x + 2*y - 0 = (0, 2, 0)
    let (ax, ay, c) = select_mccormick_plane(0.0, 2.0, 0.0, 3.0, 1.0, 1.5, 1.0, BoundDir::Upper);
    // Verify it's a valid upper bound at corners
    for &x in &[0.0, 2.0] {
        for &y in &[0.0, 3.0] {
            let z_true = x * y;
            let z_plane = ax * x + ay * y + c;
            assert!(
                z_plane >= z_true - 1e-5,
                "McCormick upper plane should be >= true at ({x},{y}): plane={z_plane}, true={z_true}"
            );
        }
    }
}

// ---- McCormick negative weight ----

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_lower_negative_weight() {
    // With negative weight, lower bound selects the UPPER envelope plane
    // z = x*y with x in [-1, 2], y in [-3, 1] (both zero-crossing)
    let (ax, ay, c) =
        select_mccormick_plane(-1.0, 2.0, -3.0, 1.0, 0.5, -1.0, -1.0, BoundDir::Lower);
    // With negative weight, this selects an upper plane.
    // Verify: the plane is a valid upper bound at corners.
    for &x in &[-1.0, 2.0] {
        for &y in &[-3.0, 1.0] {
            let z_true = x * y;
            let z_plane = ax * x + ay * y + c;
            assert!(
                z_plane >= z_true - 1e-5,
                "McCormick plane (neg weight, lower) should be upper bound at ({x},{y}): plane={z_plane}, true={z_true}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_upper_negative_weight() {
    // With negative weight, upper bound selects the LOWER envelope plane
    // z = x*y with x in [-1, 2], y in [-3, 1]
    let (ax, ay, c) =
        select_mccormick_plane(-1.0, 2.0, -3.0, 1.0, 0.5, -1.0, -1.0, BoundDir::Upper);
    // With negative weight, this selects a lower plane.
    // Verify: the plane is a valid lower bound at corners.
    for &x in &[-1.0, 2.0] {
        for &y in &[-3.0, 1.0] {
            let z_true = x * y;
            let z_plane = ax * x + ay * y + c;
            assert!(
                z_plane <= z_true + 1e-5,
                "McCormick plane (neg weight, upper) should be lower bound at ({x},{y}): plane={z_plane}, true={z_true}"
            );
        }
    }
}

// ---- McCormick zero-crossing soundness ----

#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_soundness_zero_crossing_intervals() -> Result<()> {
    let layer = MulBinaryLayer;
    // Both x and y cross zero — critical case for McCormick plane selection
    let bounds = LinearBounds::identity(1);
    let a = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    // Check soundness at all corners including interior zero
    for &x in &[-2.0, 0.0, 3.0] {
        for &y in &[-1.0, 0.0, 4.0] {
            let z_true = x * y;
            let z_lower =
                bounds_a.lower_a[[0, 0]] * x + bounds_b.lower_a[[0, 0]] * y + bounds_a.lower_b[0];
            let z_upper =
                bounds_a.upper_a[[0, 0]] * x + bounds_b.upper_a[[0, 0]] * y + bounds_a.upper_b[0];
            assert!(
                z_lower <= z_true + soundness_tol(z_true),
                "McCormick lower unsound at ({x},{y}): lower={z_lower}, true={z_true}"
            );
            assert!(
                z_upper >= z_true - soundness_tol(z_true),
                "McCormick upper unsound at ({x},{y}): upper={z_upper}, true={z_true}"
            );
        }
    }
    Ok(())
}

/// Regression for #4247: reshape/broadcast paths can preserve non-standard
/// strides, so MulBinary CROWN must flatten via iteration rather than rejecting
/// the bounds with InternalError.
#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_matches_contiguous_for_non_contiguous_inputs_4247() -> Result<()> {
    let layer = MulBinaryLayer;
    let (a_noncontiguous, a_contiguous) = make_non_contiguous_tensor_pair_3x2(
        vec![-1.0, 0.25, 2.0, -0.5, 1.5, -2.0],
        vec![0.5, 1.25, 3.0, 0.75, 2.5, -0.25],
    )?;
    let (b_noncontiguous, b_contiguous) = make_non_contiguous_tensor_pair_3x2(
        vec![0.2, -1.5, 1.0, -0.75, 0.4, 2.0],
        vec![1.1, -0.2, 1.75, 0.5, 1.6, 3.5],
    )?;
    let bounds = LinearBounds::identity(6);

    assert!(
        a_noncontiguous.lower().as_slice().is_none(),
        "precondition: lhs lower bounds must be non-contiguous"
    );
    assert!(
        b_noncontiguous.upper().as_slice().is_none(),
        "precondition: rhs upper bounds must be non-contiguous"
    );

    let (actual_a, actual_b) = layer.propagate_linear_binary(
        &bounds,
        &a_noncontiguous,
        &b_noncontiguous,
        MulBinaryRelaxationMode::McCormick,
    )?;
    let (expected_a, expected_b) = layer.propagate_linear_binary(
        &bounds,
        &a_contiguous,
        &b_contiguous,
        MulBinaryRelaxationMode::McCormick,
    )?;

    assert_linear_bounds_close(&actual_a, &expected_a, "#4247 mccormick lhs");
    assert_linear_bounds_close(&actual_b, &expected_b, "#4247 mccormick rhs");
    Ok(())
}

// ---- IBP negative times negative ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_negative_times_negative() -> Result<()> {
    let layer = MulBinaryLayer;
    // x in [-4, -1], y in [-3, -2]
    // Products: (-4)*(-3)=12, (-4)*(-2)=8, (-1)*(-3)=3, (-1)*(-2)=2
    // lower = 2, upper = 12
    let a = BoundedTensor::new(arr1(&[-4.0]).into_dyn(), arr1(&[-1.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[-2.0]).into_dyn())?;
    let out = layer.propagate_ibp_binary(&a, &b)?;
    assert_close(out.lower()[[0]], 2.0, TOL);
    assert_close(out.upper()[[0]], 12.0, TOL);
    Ok(())
}

// ---- Batched CROWN ----

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_mccormick_soundness_2d() -> Result<()> {
    // Batched CROWN with shape [out_dim, in_dim] (no batch dims, simplest case)
    let layer = MulBinaryLayer;
    let n = 2;

    // Identity bounds: shape [2, 2]
    let identity_bounds = BatchedLinearBounds::identity(&[n])?;

    // x in [1, 3], [0.5, 2.5]; y in [2, 4], [1, 3]
    let a = BoundedTensor::new(arr1(&[1.0, 0.5]).into_dyn(), arr1(&[3.0, 2.5]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0, 1.0]).into_dyn(), arr1(&[4.0, 3.0]).into_dyn())?;

    let (bounds_a, bounds_b) = layer.propagate_linear_batched_binary(
        &identity_bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::McCormick,
    )?;

    // Verify soundness at corners for each output dimension
    let x_bounds = [(1.0_f32, 3.0_f32), (0.5, 2.5)];
    let y_bounds = [(2.0_f32, 4.0_f32), (1.0, 3.0)];

    for dim in 0..n {
        let (xl, xu) = x_bounds[dim];
        let (yl, yu) = y_bounds[dim];
        for &x in &[xl, xu] {
            for &y in &[yl, yu] {
                let z_true = x * y;
                // For identity bounds: output dim == input dim, so check the
                // diagonal element
                let z_lower = bounds_a.lower_a[[dim, dim]] * x
                    + bounds_b.lower_a[[dim, dim]] * y
                    + bounds_a.lower_b[[dim]];
                let z_upper = bounds_a.upper_a[[dim, dim]] * x
                    + bounds_b.upper_a[[dim, dim]] * y
                    + bounds_a.upper_b[[dim]];
                assert!(
                    z_lower <= z_true + soundness_tol(z_true),
                    "Batched McCormick lower unsound at dim={dim}, x={x}, y={y}: lb={z_lower}, true={z_true}"
                );
                assert!(
                    z_upper >= z_true - soundness_tol(z_true),
                    "Batched McCormick upper unsound at dim={dim}, x={x}, y={y}: ub={z_upper}, true={z_true}"
                );
            }
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_middle_soundness_2d() -> Result<()> {
    let layer = MulBinaryLayer;
    let n = 2;
    let identity_bounds = BatchedLinearBounds::identity(&[n])?;

    let a = BoundedTensor::new(arr1(&[1.0, 0.5]).into_dyn(), arr1(&[3.0, 2.5]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0, 1.0]).into_dyn(), arr1(&[4.0, 3.0]).into_dyn())?;

    let (bounds_a, bounds_b) = layer.propagate_linear_batched_binary(
        &identity_bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::Middle,
    )?;

    let x_bounds = [(1.0_f32, 3.0_f32), (0.5, 2.5)];
    let y_bounds = [(2.0_f32, 4.0_f32), (1.0, 3.0)];

    for dim in 0..n {
        let (xl, xu) = x_bounds[dim];
        let (yl, yu) = y_bounds[dim];
        for &x in &[xl, xu] {
            for &y in &[yl, yu] {
                let z_true = x * y;
                let z_lower = bounds_a.lower_a[[dim, dim]] * x
                    + bounds_b.lower_a[[dim, dim]] * y
                    + bounds_a.lower_b[[dim]];
                let z_upper = bounds_a.upper_a[[dim, dim]] * x
                    + bounds_b.upper_a[[dim, dim]] * y
                    + bounds_a.upper_b[[dim]];
                assert!(
                    z_lower <= z_true + soundness_tol(z_true),
                    "Batched Middle lower unsound at dim={dim}, x={x}, y={y}: lb={z_lower}, true={z_true}"
                );
                assert!(
                    z_upper >= z_true - soundness_tol(z_true),
                    "Batched Middle upper unsound at dim={dim}, x={x}, y={y}: ub={z_upper}, true={z_true}"
                );
            }
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_rejects_1d_input() {
    let layer = MulBinaryLayer;
    // 1D bounds should be rejected (need at least 2D for batched)
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::zeros(IxDyn(&[3])),
        ArrayD::zeros(IxDyn(&[3])),
        ArrayD::zeros(IxDyn(&[3])),
        ArrayD::zeros(IxDyn(&[3])),
        vec![3],
        vec![3],
    );
    let a = BoundedTensor::new(
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
        arr1(&[2.0, 3.0, 4.0]).into_dyn(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
        arr1(&[2.0, 3.0, 4.0]).into_dyn(),
    )
    .unwrap();
    let err = layer
        .propagate_linear_batched_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)
        .expect_err("1D should be rejected");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

/// Verify that batched CROWN path matches non-batched for single element.
/// Both paths should produce identical coefficients and bias for the same inputs
/// with identity bounds and no batch dimensions.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_mccormick_matches_nonbatched_single_element() -> Result<()> {
    let layer = MulBinaryLayer;

    // x in [1,3], y in [2,4] — all positive, simplest McCormick case
    let a = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;

    // Non-batched
    let nb_bounds = LinearBounds::identity(1);
    let (nb_a, nb_b) =
        layer.propagate_linear_binary(&nb_bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    // Batched (no batch dims — shape [1,1])
    let bat_bounds = BatchedLinearBounds::identity(&[1])?;
    let (bat_a, bat_b) = layer.propagate_linear_batched_binary(
        &bat_bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::McCormick,
    )?;

    // Compare coefficient matrices (for input a)
    assert!(
        (bat_a.lower_a[[0, 0]] - nb_a.lower_a[[0, 0]]).abs() < 1e-4,
        "lower_a mismatch: batched={}, non-batched={}",
        bat_a.lower_a[[0, 0]],
        nb_a.lower_a[[0, 0]]
    );
    assert!(
        (bat_a.upper_a[[0, 0]] - nb_a.upper_a[[0, 0]]).abs() < 1e-4,
        "upper_a mismatch: batched={}, non-batched={}",
        bat_a.upper_a[[0, 0]],
        nb_a.upper_a[[0, 0]]
    );

    // Compare bias
    assert!(
        (bat_a.lower_b[[0]] - nb_a.lower_b[0]).abs() < 1e-4,
        "lower_b mismatch: batched={}, non-batched={}",
        bat_a.lower_b[[0]],
        nb_a.lower_b[0]
    );
    assert!(
        (bat_a.upper_b[[0]] - nb_a.upper_b[0]).abs() < 1e-4,
        "upper_b mismatch: batched={}, non-batched={}",
        bat_a.upper_b[[0]],
        nb_a.upper_b[0]
    );

    // Compare coefficient matrices (for input b)
    assert!(
        (bat_b.lower_a[[0, 0]] - nb_b.lower_a[[0, 0]]).abs() < 1e-4,
        "b lower_a mismatch: batched={}, non-batched={}",
        bat_b.lower_a[[0, 0]],
        nb_b.lower_a[[0, 0]]
    );
    assert!(
        (bat_b.upper_a[[0, 0]] - nb_b.upper_a[[0, 0]]).abs() < 1e-4,
        "b upper_a mismatch: batched={}, non-batched={}",
        bat_b.upper_a[[0, 0]],
        nb_b.upper_a[[0, 0]]
    );

    Ok(())
}

/// Verify soundness of batched McCormick with zero-crossing intervals.
/// The linear bounds (a*x + b*y + c) must enclose the true product at all corners.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_mccormick_soundness_zero_crossing() -> Result<()> {
    let layer = MulBinaryLayer;

    // x in [-2, 3], y in [-1, 4] — both cross zero
    let a = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;

    let bat_bounds = BatchedLinearBounds::identity(&[1])?;
    let (bat_a, bat_b) = layer.propagate_linear_batched_binary(
        &bat_bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::McCormick,
    )?;

    for &x in &[-2.0, 0.0, 3.0] {
        for &y in &[-1.0, 0.0, 4.0] {
            let z_true = x * y;
            let z_lower =
                bat_a.lower_a[[0, 0]] * x + bat_b.lower_a[[0, 0]] * y + bat_a.lower_b[[0]];
            let z_upper =
                bat_a.upper_a[[0, 0]] * x + bat_b.upper_a[[0, 0]] * y + bat_a.upper_b[[0]];
            assert!(
                z_lower <= z_true + soundness_tol(z_true),
                "Batched McCormick lower unsound at (x={x}, y={y}): lb={z_lower}, true={z_true}"
            );
            assert!(
                z_upper >= z_true - soundness_tol(z_true),
                "Batched McCormick upper unsound at (x={x}, y={y}): ub={z_upper}, true={z_true}"
            );
        }
    }
    Ok(())
}

// ---- CROWN shape validation ----

#[ntest::timeout(10000)]
#[test]
fn test_crown_rejects_mismatched_input_sizes() {
    let layer = MulBinaryLayer;
    let bounds = LinearBounds::identity(3);
    let a = BoundedTensor::new(
        arr1(&[0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
    )
    .unwrap();
    // b has different size than bounds.num_inputs()
    let b = BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn()).unwrap();
    let err = layer
        .propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)
        .expect_err("mismatched sizes");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

/// Regression for #3499: alpha-parameterized MulBinary CROWN must support
/// broadcasted inputs like the ECAPA-TDNN SE blocks `[C,T] * [C,1]`.
#[ntest::timeout(10000)]
#[test]
fn test_crown_alpha_broadcast_soundness_se_block_pattern_3499() -> Result<()> {
    let (layer, a, b, bounds, alphas) = make_alpha_broadcast_case_3499()?;

    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary_with_alpha(&bounds, &a, &b, Some(&alphas))?;

    assert_eq!(
        bounds_a.num_inputs(),
        6,
        "lhs coefficients should keep [2,3] inputs"
    );
    assert_eq!(
        bounds_b.num_inputs(),
        2,
        "rhs coefficients should reduce the broadcast [2,1] input"
    );

    let concrete_a = bounds_a.concretize(&a);
    let concrete_b = bounds_b.concretize(&b);
    let lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
    let upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];
    assert_se_block_broadcast_corners_3499(&bounds, &a, &b, lower, upper);

    Ok(())
}

/// Regression for #3499: broadcasted RHS coefficients must be reduced by
/// summation, matching alpha-beta-CROWN `reduce_broadcast_dims`.
#[ntest::timeout(10000)]
#[test]
fn test_crown_alpha_broadcast_rhs_columns_accumulate_3499() -> Result<()> {
    let (layer, a, b, bounds, alphas) = make_alpha_broadcast_case_3499()?;
    let (_bounds_a, bounds_b) =
        layer.propagate_linear_binary_with_alpha(&bounds, &a, &b, Some(&alphas))?;
    let (expected_lower, expected_upper) =
        expected_reduced_rhs_coefficients_3499(&bounds, &a, &b, &alphas);

    assert_eq!(
        bounds_b.num_inputs(),
        2,
        "broadcasted RHS should reduce to its true [2,1] input width"
    );

    for idx in 0..2 {
        assert_close(bounds_b.lower_a()[[0, idx]], expected_lower[idx], TOL);
        assert_close(bounds_b.upper_a()[[0, idx]], expected_upper[idx], TOL);
    }

    Ok(())
}

/// Regression for #4247: alpha-CROWN shares the same flattening surface as the
/// fixed McCormick path and must accept non-standard-layout bounds.
#[ntest::timeout(10000)]
#[test]
fn test_crown_alpha_matches_contiguous_for_non_contiguous_inputs_4247() -> Result<()> {
    let layer = MulBinaryLayer;
    let (a_noncontiguous, a_contiguous) = make_non_contiguous_tensor_pair_3x2(
        vec![-1.5, 0.0, 1.0, -0.25, 0.5, 2.0],
        vec![0.5, 1.0, 2.25, 0.75, 1.25, 3.5],
    )?;
    let (b_noncontiguous, b_contiguous) = make_non_contiguous_tensor_pair_3x2(
        vec![0.2, -1.0, 0.5, -0.4, 1.25, -0.75],
        vec![1.4, -0.1, 1.75, 0.6, 2.0, 0.5],
    )?;
    let lower_weights =
        Array2::from_shape_vec((1, 6), vec![1.0, -0.5, 0.25, -1.0, 0.75, 2.0]).unwrap();
    let upper_weights =
        Array2::from_shape_vec((1, 6), vec![-0.25, 1.25, -0.75, 0.3, -1.0, 1.5]).unwrap();
    let bounds = LinearBounds::new(lower_weights, arr1(&[0.0]), upper_weights, arr1(&[0.0]))?;
    let alphas = Array2::from_shape_vec(
        (2, 6),
        vec![0.1, 0.7, 0.4, 0.9, 0.2, 0.6, 0.8, 0.3, 0.5, 0.1, 0.9, 0.4],
    )
    .unwrap();

    assert!(
        a_noncontiguous.lower().as_slice().is_none(),
        "precondition: lhs lower bounds must be non-contiguous"
    );
    assert!(
        b_noncontiguous.upper().as_slice().is_none(),
        "precondition: rhs upper bounds must be non-contiguous"
    );

    let (actual_a, actual_b) = layer.propagate_linear_binary_with_alpha(
        &bounds,
        &a_noncontiguous,
        &b_noncontiguous,
        Some(&alphas),
    )?;
    let (expected_a, expected_b) = layer.propagate_linear_binary_with_alpha(
        &bounds,
        &a_contiguous,
        &b_contiguous,
        Some(&alphas),
    )?;

    assert_linear_bounds_close(&actual_a, &expected_a, "#4247 alpha lhs");
    assert_linear_bounds_close(&actual_b, &expected_b, "#4247 alpha rhs");
    Ok(())
}

// ---- CROWN McCormick soundness ----

#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_bounds_contain_true_product() -> Result<()> {
    let layer = MulBinaryLayer;
    // z = x * y, identity bounds, single element
    let bounds = LinearBounds::identity(1);
    let a = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    // Evaluate bounds at corners and verify they're sound
    for &x in &[1.0, 3.0] {
        for &y in &[2.0, 4.0] {
            let z_true = x * y;
            let z_lower =
                bounds_a.lower_a[[0, 0]] * x + bounds_b.lower_a[[0, 0]] * y + bounds_a.lower_b[0];
            let z_upper =
                bounds_a.upper_a[[0, 0]] * x + bounds_b.upper_a[[0, 0]] * y + bounds_a.upper_b[0];
            assert!(
                z_lower <= z_true + soundness_tol(z_true),
                "McCormick lower should be <= true product: lower={z_lower}, true={z_true} at x={x}, y={y}"
            );
            assert!(
                z_upper >= z_true - soundness_tol(z_true),
                "McCormick upper should be >= true product: upper={z_upper}, true={z_true} at x={x}, y={y}"
            );
        }
    }
    Ok(())
}

// ---- CROWN McCormick soundness: both-negative intervals ----

/// McCormick CROWN must produce sound bounds when both x and y are entirely negative.
/// x ∈ [lx=-4, ux=-1], y ∈ [ly=-3, uy=-2]. Products at corners:
///   (-4)·(-3)=12, (-4)·(-2)=8, (-1)·(-3)=3, (-1)·(-2)=2
/// True range: [2, 12]. McCormick lower planes (z >= ly·x + lx·y - lx·ly):
///   L1 = ly·x + lx·y - lx·ly = (-3)·x + (-4)·y - 12
///   L2 = uy·x + ux·y - ux·uy = (-2)·x + (-1)·y - 2
/// Both are valid lower bounds for z = x·y.
#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_soundness_both_negative_intervals() -> Result<()> {
    let layer = MulBinaryLayer;
    let bounds = LinearBounds::identity(1);
    let a = BoundedTensor::new(arr1(&[-4.0]).into_dyn(), arr1(&[-1.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[-2.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    // Soundness at all 4 corners and the interior midpoint
    for &x in &[-4.0, -2.5, -1.0] {
        for &y in &[-3.0, -2.5, -2.0] {
            let z_true = x * y;
            let z_lower =
                bounds_a.lower_a[[0, 0]] * x + bounds_b.lower_a[[0, 0]] * y + bounds_a.lower_b[0];
            let z_upper =
                bounds_a.upper_a[[0, 0]] * x + bounds_b.upper_a[[0, 0]] * y + bounds_a.upper_b[0];
            assert!(
                z_lower <= z_true + soundness_tol(z_true),
                "both-negative lower unsound at ({x},{y}): lower={z_lower}, true={z_true}"
            );
            assert!(
                z_upper >= z_true - soundness_tol(z_true),
                "both-negative upper unsound at ({x},{y}): upper={z_upper}, true={z_true}"
            );
        }
    }
    Ok(())
}

// ---- CROWN McCormick soundness: positive-times-negative ----

/// McCormick CROWN must produce sound bounds when x is positive and y is negative.
/// x ∈ [1, 3], y ∈ [-4, -2]. Products at corners:
///   1·(-4)=-4, 1·(-2)=-2, 3·(-4)=-12, 3·(-2)=-6
/// True range: [-12, -2].
#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_soundness_positive_times_negative() -> Result<()> {
    let layer = MulBinaryLayer;
    let bounds = LinearBounds::identity(1);
    let a = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-4.0]).into_dyn(), arr1(&[-2.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    for &x in &[1.0, 2.0, 3.0] {
        for &y in &[-4.0, -3.0, -2.0] {
            let z_true = x * y;
            let z_lower =
                bounds_a.lower_a[[0, 0]] * x + bounds_b.lower_a[[0, 0]] * y + bounds_a.lower_b[0];
            let z_upper =
                bounds_a.upper_a[[0, 0]] * x + bounds_b.upper_a[[0, 0]] * y + bounds_a.upper_b[0];
            assert!(
                z_lower <= z_true + soundness_tol(z_true),
                "pos×neg lower unsound at ({x},{y}): lower={z_lower}, true={z_true}"
            );
            assert!(
                z_upper >= z_true - soundness_tol(z_true),
                "pos×neg upper unsound at ({x},{y}): upper={z_upper}, true={z_true}"
            );
        }
    }
    Ok(())
}

// ---- CROWN McCormick soundness: negative-times-positive ----

/// McCormick CROWN with x negative, y positive: x ∈ [-5, -1], y ∈ [2, 6].
/// True range: [-30, -2].
#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_soundness_negative_times_positive() -> Result<()> {
    let layer = MulBinaryLayer;
    let bounds = LinearBounds::identity(1);
    let a = BoundedTensor::new(arr1(&[-5.0]).into_dyn(), arr1(&[-1.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[6.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    for &x in &[-5.0, -3.0, -1.0] {
        for &y in &[2.0, 4.0, 6.0] {
            let z_true = x * y;
            let z_lower =
                bounds_a.lower_a[[0, 0]] * x + bounds_b.lower_a[[0, 0]] * y + bounds_a.lower_b[0];
            let z_upper =
                bounds_a.upper_a[[0, 0]] * x + bounds_b.upper_a[[0, 0]] * y + bounds_a.upper_b[0];
            assert!(
                z_lower <= z_true + soundness_tol(z_true),
                "neg×pos lower unsound at ({x},{y}): lower={z_lower}, true={z_true}"
            );
            assert!(
                z_upper >= z_true - soundness_tol(z_true),
                "neg×pos upper unsound at ({x},{y}): upper={z_upper}, true={z_true}"
            );
        }
    }
    Ok(())
}

// ---- CROWN Middle relaxation ----

#[ntest::timeout(10000)]
#[test]
fn test_crown_middle_relaxation_produces_valid_bounds() -> Result<()> {
    let layer = MulBinaryLayer;
    let bounds = LinearBounds::identity(1);
    let a = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::Middle)?;

    // Middle relaxation should produce coefficients (not all zero)
    let a_coeff_sum = bounds_a.lower_a[[0, 0]].abs() + bounds_a.upper_a[[0, 0]].abs();
    let b_coeff_sum = bounds_b.lower_a[[0, 0]].abs() + bounds_b.upper_a[[0, 0]].abs();
    assert!(
        a_coeff_sum > 0.0 || b_coeff_sum > 0.0,
        "Middle relaxation should produce non-zero coefficients"
    );

    // Verify soundness at corners
    for &x in &[1.0, 3.0] {
        for &y in &[2.0, 4.0] {
            let z_true = x * y;
            let z_lower =
                bounds_a.lower_a[[0, 0]] * x + bounds_b.lower_a[[0, 0]] * y + bounds_a.lower_b[0];
            let z_upper =
                bounds_a.upper_a[[0, 0]] * x + bounds_b.upper_a[[0, 0]] * y + bounds_a.upper_b[0];
            assert!(
                z_lower <= z_true + soundness_tol(z_true),
                "Middle lower should be <= true: lower={z_lower}, true={z_true} at x={x}, y={y}"
            );
            assert!(
                z_upper >= z_true - soundness_tol(z_true),
                "Middle upper should be >= true: upper={z_upper}, true={z_true} at x={x}, y={y}"
            );
        }
    }
    Ok(())
}

// ---- Middle coefficient computation ----

#[ntest::timeout(10000)]
#[test]
fn test_middle_coefficients_hand_computed() {
    // x in [1, 3], y in [2, 4]
    // x_mid = 2, y_mid = 3
    // alpha_l = (y_l - y_u) * 0.5 + y_u = (2 - 4)*0.5 + 4 = 3.0
    // beta_l  = (x_l - x_u) * 0.5 + x_u = (1 - 3)*0.5 + 3 = 2.0
    // ny_l = (y_u*x_u - y_l*x_l)*0.5 - y_u*x_u = (12-2)*0.5 - 12 = -7.0
    let (alpha_l, beta_l, ny_l, alpha_u, beta_u, ny_u) =
        MulBinaryLayer::compute_middle_coefficients(1.0, 3.0, 2.0, 4.0);

    assert_close(alpha_l, 3.0, TOL);
    assert_close(beta_l, 2.0, TOL);
    assert_close(ny_l, -7.0, TOL);

    // alpha_u = (y_u - y_l) * 0.5 + y_l = (4-2)*0.5 + 2 = 3.0
    // beta_u  = (x_l - x_u) * 0.5 + x_u = (1-3)*0.5 + 3 = 2.0
    // ny_u = (y_l*x_u - y_u*x_l)*0.5 - y_l*x_u = (6-4)*0.5 - 6 = -5.0
    assert_close(alpha_u, 3.0, TOL);
    assert_close(beta_u, 2.0, TOL);
    assert_close(ny_u, -5.0, TOL);
}

// ---- McCormick NaN guard (#2741) ----

/// `select_mccormick_plane` must return conservative trivial bounds when any
/// input is NaN, rather than propagating NaN coefficients through the CROWN
/// backward pass. Lower → (0, 0, -inf), Upper → (0, 0, +inf).
#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_nan_lx_returns_conservative_lower() {
    let (ax, ay, c) =
        select_mccormick_plane(f32::NAN, 2.0, 0.0, 3.0, 1.0, 1.5, 1.0, BoundDir::Lower);
    assert_eq!(ax, 0.0);
    assert_eq!(ay, 0.0);
    assert_eq!(c, f32::NEG_INFINITY);
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_nan_uy_returns_conservative_upper() {
    let (ax, ay, c) =
        select_mccormick_plane(0.0, 2.0, 0.0, f32::NAN, 1.0, 1.5, 1.0, BoundDir::Upper);
    assert_eq!(ax, 0.0);
    assert_eq!(ay, 0.0);
    assert_eq!(c, f32::INFINITY);
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_nan_x0_returns_conservative() {
    // NaN in evaluation point should also trigger the guard.
    let (ax, ay, c) =
        select_mccormick_plane(0.0, 2.0, 0.0, 3.0, f32::NAN, 1.5, 1.0, BoundDir::Lower);
    assert_eq!(ax, 0.0);
    assert_eq!(ay, 0.0);
    assert_eq!(c, f32::NEG_INFINITY);
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_all_nan_returns_conservative() {
    let (ax_l, ay_l, c_l) = select_mccormick_plane(
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        1.0,
        BoundDir::Lower,
    );
    assert_eq!(ax_l, 0.0);
    assert_eq!(ay_l, 0.0);
    assert_eq!(c_l, f32::NEG_INFINITY);

    let (ax_u, ay_u, c_u) = select_mccormick_plane(
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        f32::NAN,
        1.0,
        BoundDir::Upper,
    );
    assert_eq!(ax_u, 0.0);
    assert_eq!(ay_u, 0.0);
    assert_eq!(c_u, f32::INFINITY);
}

/// Infinity inputs can produce NaN via `0 * inf` in McCormick plane products
/// (e.g., `-lx * ly` with `lx = 0, ly = -inf`). The guard must catch infinity
/// as well as NaN.
#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_inf_lx_returns_conservative() {
    let (ax, ay, c) = select_mccormick_plane(
        f32::NEG_INFINITY,
        2.0,
        0.0,
        3.0,
        1.0,
        1.5,
        1.0,
        BoundDir::Lower,
    );
    assert_eq!(ax, 0.0);
    assert_eq!(ay, 0.0);
    assert_eq!(c, f32::NEG_INFINITY);
}

#[ntest::timeout(10000)]
#[test]
fn test_mccormick_plane_inf_uy_returns_conservative_upper() {
    let (ax, ay, c) =
        select_mccormick_plane(0.0, 2.0, 0.0, f32::INFINITY, 1.0, 1.5, 1.0, BoundDir::Upper);
    assert_eq!(ax, 0.0);
    assert_eq!(ay, 0.0);
    assert_eq!(c, f32::INFINITY);
}

// ---- CROWN overflow protection ----

#[ntest::timeout(10000)]
#[test]
fn test_crown_rejects_infinite_input_bounds() {
    let layer = MulBinaryLayer;
    let bounds = LinearBounds::identity(1);
    // Use huge magnitude to trigger overflow guard
    let a = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[2e19]).into_dyn()).unwrap();
    let b = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    let err = layer
        .propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)
        .expect_err("large magnitude should trigger overflow guard");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

// ---- Bias placement regression (#2520) ----

/// MulBinary returns two affine forms (for x and y) but only one shared
/// McCormick constant. The bias must appear exactly once to avoid DAG-CROWN
/// double-counting when both branches are accumulated.
#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_bias_only_on_primary_output_2520() -> Result<()> {
    let layer = MulBinaryLayer;
    let bounds = LinearBounds::identity(1);
    let a = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    // Primary output carries the shared McCormick constant.
    assert!(
        bounds_a.lower_b[0].abs() > 1e-6,
        "McCormick lower bias should be nonzero for zero-crossing: {}",
        bounds_a.lower_b[0]
    );
    assert_eq!(
        bounds_b.lower_b[0], 0.0,
        "Secondary lower bias must be zero to avoid duplicate constant accumulation"
    );
    assert_eq!(
        bounds_b.upper_b[0], 0.0,
        "Secondary upper bias must be zero to avoid duplicate constant accumulation"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_mccormick_bias_only_on_primary_output_2520() -> Result<()> {
    let layer = MulBinaryLayer;
    let bounds = BatchedLinearBounds::identity(&[1])?;
    let a = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;
    let (bounds_a, bounds_b) = layer.propagate_linear_batched_binary(
        &bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::McCormick,
    )?;

    assert!(
        bounds_a.lower_b[[0]].abs() > 1e-6,
        "Primary batched lower bias should be nonzero for zero-crossing: {}",
        bounds_a.lower_b[[0]]
    );
    assert_eq!(
        bounds_b.lower_b[[0]],
        0.0,
        "Secondary batched lower bias must be zero to avoid duplicate constant accumulation"
    );
    assert_eq!(
        bounds_b.upper_b[[0]],
        0.0,
        "Secondary batched upper bias must be zero to avoid duplicate constant accumulation"
    );
    Ok(())
}

// ---- f64 bias accumulation and directed rounding (#2471) ----

/// Verify that CROWN bias uses directed rounding: lower_b <= upper_b for all outputs.
/// With f32 accumulation and no directed rounding, cancellation-prone inputs
/// could produce lower_b > upper_b (unsound inversion).
///
/// Uses multi-element inputs with alternating-sign bounds to stress cancellation.
/// McCormick constant terms (e.g., -lx*ly, -ux*uy) with mixed signs cancel badly in f32.
#[ntest::timeout(10000)]
#[test]
fn test_crown_f64_bias_directed_rounding_mccormick_2471() -> Result<()> {
    use ndarray::Array2;

    let layer = MulBinaryLayer;
    let n = 64; // enough elements to stress accumulation

    // Alternating-sign bounds near the McCormick overflow threshold sqrt(1.84e19) ≈ 4.29e9.
    // Use large magnitudes just under the guard to maximize cancellation.
    let scale = 1e9_f32;
    let mut a_lower_vec = vec![0.0_f32; n];
    let mut a_upper_vec = vec![0.0_f32; n];
    let mut b_lower_vec = vec![0.0_f32; n];
    let mut b_upper_vec = vec![0.0_f32; n];

    for i in 0..n {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        a_lower_vec[i] = sign * scale;
        a_upper_vec[i] = sign * scale + scale * 0.1;
        b_lower_vec[i] = -sign * scale;
        b_upper_vec[i] = -sign * scale + scale * 0.1;
    }

    let a = BoundedTensor::new(arr1(&a_lower_vec).into_dyn(), arr1(&a_upper_vec).into_dyn())?;
    let b = BoundedTensor::new(arr1(&b_lower_vec).into_dyn(), arr1(&b_upper_vec).into_dyn())?;

    // Use non-identity incoming bounds with mixed weights to force cancellation
    let mut lower_a_data = Array2::<f32>::zeros((1, n));
    let mut upper_a_data = Array2::<f32>::zeros((1, n));
    for j in 0..n {
        let w = if j % 3 == 0 { 1.5 } else { -0.7 };
        lower_a_data[[0, j]] = w;
        upper_a_data[[0, j]] = w + 0.2;
    }
    let bounds = LinearBounds::new(lower_a_data, arr1(&[0.0]), upper_a_data, arr1(&[0.0])).unwrap();

    let (bounds_a, _bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    // Key soundness check: lower_b <= upper_b (no inversion from rounding)
    for i in 0..bounds_a.lower_b.len() {
        assert!(
            bounds_a.lower_b[i] <= bounds_a.upper_b[i],
            "Bias inversion at output {i}: lower_b={} > upper_b={} — \
             directed rounding should prevent this (#2471)",
            bounds_a.lower_b[i],
            bounds_a.upper_b[i],
        );
    }

    Ok(())
}

/// Same as above but for the Middle relaxation mode.
#[ntest::timeout(10000)]
#[test]
fn test_crown_f64_bias_directed_rounding_middle_2471() -> Result<()> {
    use ndarray::Array2;

    let layer = MulBinaryLayer;
    let n = 64;

    let scale = 1e9_f32;
    let mut a_lower_vec = vec![0.0_f32; n];
    let mut a_upper_vec = vec![0.0_f32; n];
    let mut b_lower_vec = vec![0.0_f32; n];
    let mut b_upper_vec = vec![0.0_f32; n];

    for i in 0..n {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        a_lower_vec[i] = sign * scale;
        a_upper_vec[i] = sign * scale + scale * 0.1;
        b_lower_vec[i] = -sign * scale;
        b_upper_vec[i] = -sign * scale + scale * 0.1;
    }

    let a = BoundedTensor::new(arr1(&a_lower_vec).into_dyn(), arr1(&a_upper_vec).into_dyn())?;
    let b = BoundedTensor::new(arr1(&b_lower_vec).into_dyn(), arr1(&b_upper_vec).into_dyn())?;

    let mut lower_a_data = Array2::<f32>::zeros((1, n));
    let mut upper_a_data = Array2::<f32>::zeros((1, n));
    for j in 0..n {
        let w = if j % 3 == 0 { 1.5 } else { -0.7 };
        lower_a_data[[0, j]] = w;
        upper_a_data[[0, j]] = w + 0.2;
    }
    let bounds = LinearBounds::new(lower_a_data, arr1(&[0.0]), upper_a_data, arr1(&[0.0])).unwrap();

    let (bounds_a, _bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::Middle)?;

    for i in 0..bounds_a.lower_b.len() {
        assert!(
            bounds_a.lower_b[i] <= bounds_a.upper_b[i],
            "Middle bias inversion at output {i}: lower_b={} > upper_b={} (#2471)",
            bounds_a.lower_b[i],
            bounds_a.upper_b[i],
        );
    }

    Ok(())
}

// ---- Batched CROWN with batched input bounds (#2521) ----

/// Build a [2, 2, 2] batched identity coefficient matrix (batch=2, out_dim=2, n=2).
fn make_batched_identity_2x2x2() -> BatchedLinearBounds {
    let a_shape = IxDyn(&[2, 2, 2]);
    let b_shape = IxDyn(&[2, 2]);
    let mut lower_a = ArrayD::<f32>::zeros(a_shape.clone());
    let mut upper_a = ArrayD::<f32>::zeros(a_shape);
    for batch in 0..2 {
        for i in 0..2 {
            lower_a[[batch, i, i]] = 1.0;
            upper_a[[batch, i, i]] = 1.0;
        }
    }
    BatchedLinearBounds::from_parts_unchecked(
        lower_a,
        ArrayD::zeros(b_shape.clone()),
        upper_a,
        ArrayD::zeros(b_shape),
        vec![2, 2],
        vec![2, 2],
    )
}

/// Batched input bounds (input_len == batch_size * n) must index each batch
/// into its own slice rather than broadcasting. Before the fix (#2521), a dead
/// conditional and silent `.min(input_len - 1)` clamp masked the wrong index.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_with_batched_input_bounds_2521() -> Result<()> {
    let layer = MulBinaryLayer;
    let bat_bounds = make_batched_identity_2x2x2();

    // Batched input bounds: [4] = batch_size(2) * n(2).
    // Batch 0: x in [1,3],[2,4]; Batch 1: x in [10,30],[20,40]
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 10.0, 20.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![3.0, 4.0, 30.0, 40.0]).unwrap(),
    )?;
    // Batch 0: y in [5,7],[6,8]; Batch 1: y in [50,70],[60,80]
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![5.0, 6.0, 50.0, 60.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![7.0, 8.0, 70.0, 80.0]).unwrap(),
    )?;

    let (ba, bb) = layer.propagate_linear_batched_binary(
        &bat_bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::McCormick,
    )?;

    // If batch indexing is wrong, batch 1 would use batch 0's bounds → unsound.
    let xb = [
        [(1.0_f32, 3.0_f32), (2.0, 4.0)],
        [(10.0, 30.0), (20.0, 40.0)],
    ];
    let yb = [
        [(5.0_f32, 7.0_f32), (6.0, 8.0)],
        [(50.0, 70.0), (60.0, 80.0)],
    ];

    for bi in 0..2 {
        for d in 0..2 {
            let (xl, xu) = xb[bi][d];
            let (yl, yu) = yb[bi][d];
            for &x in &[xl, xu] {
                for &y in &[yl, yu] {
                    let z = x * y;
                    let lo = ba.lower_a[[bi, d, d]] * x
                        + bb.lower_a[[bi, d, d]] * y
                        + ba.lower_b[[bi, d]];
                    let hi = ba.upper_a[[bi, d, d]] * x
                        + bb.upper_a[[bi, d, d]] * y
                        + ba.upper_b[[bi, d]];
                    assert!(
                        lo <= z + soundness_tol(z),
                        "lower unsound: batch={bi}, dim={d}, ({x},{y}): {lo} > {z}"
                    );
                    assert!(
                        hi >= z - soundness_tol(z),
                        "upper unsound: batch={bi}, dim={d}, ({x},{y}): {hi} < {z}"
                    );
                }
            }
        }
    }
    Ok(())
}

/// Middle mode must support the same per-batch input-bound indexing semantics as
/// McCormick mode when input_len == batch_size * n.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_middle_with_batched_input_bounds_2532() -> Result<()> {
    let layer = MulBinaryLayer;
    let bat_bounds = make_batched_identity_2x2x2();

    // Batch 0: x in [1,3],[2,4], y in [5,7],[6,8]
    // Batch 1: x in [10,30],[20,40], y in [50,70],[60,80]
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 10.0, 20.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![3.0, 4.0, 30.0, 40.0]).unwrap(),
    )?;
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![5.0, 6.0, 50.0, 60.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![7.0, 8.0, 70.0, 80.0]).unwrap(),
    )?;

    let (ba, bb) = layer.propagate_linear_batched_binary(
        &bat_bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::Middle,
    )?;

    let xb = [
        [(1.0_f32, 3.0_f32), (2.0, 4.0)],
        [(10.0, 30.0), (20.0, 40.0)],
    ];
    let yb = [
        [(5.0_f32, 7.0_f32), (6.0, 8.0)],
        [(50.0, 70.0), (60.0, 80.0)],
    ];

    for bi in 0..2 {
        for d in 0..2 {
            let (xl, xu) = xb[bi][d];
            let (yl, yu) = yb[bi][d];
            for &x in &[xl, xu] {
                for &y in &[yl, yu] {
                    let z = x * y;
                    let lo = ba.lower_a[[bi, d, d]] * x
                        + bb.lower_a[[bi, d, d]] * y
                        + ba.lower_b[[bi, d]];
                    let hi = ba.upper_a[[bi, d, d]] * x
                        + bb.upper_a[[bi, d, d]] * y
                        + ba.upper_b[[bi, d]];
                    assert!(
                        lo <= z + soundness_tol(z),
                        "middle lower unsound: batch={bi}, dim={d}, ({x},{y}): {lo} > {z}"
                    );
                    assert!(
                        hi >= z - soundness_tol(z),
                        "middle upper unsound: batch={bi}, dim={d}, ({x},{y}): {hi} < {z}"
                    );
                }
            }
        }
    }

    Ok(())
}

/// Regression for #4247: batched MulBinary CROWN must tolerate non-standard
/// layouts in both the input bounds and the incoming batched coefficient/bias
/// tensors, matching the contiguous semantics exactly.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_matches_contiguous_for_non_contiguous_inputs_and_bounds_4247() -> Result<()> {
    let layer = MulBinaryLayer;
    let ((a_noncontiguous, a_contiguous), (b_noncontiguous, b_contiguous)) =
        make_batched_non_contiguous_inputs_4247()?;
    let (actual_bounds, expected_bounds) = make_batched_non_contiguous_bounds_pair_4247();

    let (actual_a, actual_b) = layer.propagate_linear_batched_binary(
        &actual_bounds,
        &a_noncontiguous,
        &b_noncontiguous,
        MulBinaryRelaxationMode::McCormick,
    )?;
    let (expected_a, expected_b) = layer.propagate_linear_batched_binary(
        &expected_bounds,
        &a_contiguous,
        &b_contiguous,
        MulBinaryRelaxationMode::McCormick,
    )?;

    assert_batched_linear_bounds_close(&actual_a, &expected_a, "#4247 batched lhs");
    assert_batched_linear_bounds_close(&actual_b, &expected_b, "#4247 batched rhs");
    Ok(())
}

/// Mismatched batched input bounds (not n and not batch_size*n) must error.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_rejects_mismatched_batched_input_len_2521() {
    let layer = MulBinaryLayer;
    let bat_bounds = make_batched_identity_2x2x2();

    // 3 elements: not n=2 and not batch_size*n=4
    let a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();

    let err = layer
        .propagate_linear_batched_binary(&bat_bounds, &a, &b, MulBinaryRelaxationMode::McCormick)
        .expect_err("mismatched input length should be rejected");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

// ---- McCormick plane selection proptest (Prover directive P1 1090) ----

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Proptest: select_mccormick_plane soundness for random bounds and weights.
    ///
    /// For any box [lx, ux] × [ly, uy], the selected McCormick plane must satisfy:
    /// - BoundDir::Lower, w >= 0: plane(x,y) <= x*y at all corners (lower bound)
    /// - BoundDir::Upper, w >= 0: plane(x,y) >= x*y at all corners (upper bound)
    /// - BoundDir::Lower, w < 0: plane(x,y) >= x*y at all corners (upper, flipped)
    /// - BoundDir::Upper, w < 0: plane(x,y) <= x*y at all corners (lower, flipped)
    ///
    /// Replaces 2 hardcoded positive/negative weight tests with random coverage.
    /// Reference: McCormick (1976), "Computability of global solutions".
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_mccormick_plane_soundness(
        lx in -5.0f32..5.0,
        dx in 0.01f32..5.0,
        ly in -5.0f32..5.0,
        dy in 0.01f32..5.0,
        // t_x/t_y in [0,1] to interpolate midpoint within bounds
        t_x in 0.0f32..1.0,
        t_y in 0.0f32..1.0,
        w in prop::sample::select(vec![-2.0f32, -1.0, -0.5, 0.5, 1.0, 2.0]),
    ) {
        let ux = lx + dx;
        let uy = ly + dy;
        let x0 = lx + t_x * dx;
        let y0 = ly + t_y * dy;

        let corners = [(lx, ly), (lx, uy), (ux, ly), (ux, uy)];
        let tol = 1e-4;

        for &bound_dir in &[BoundDir::Lower, BoundDir::Upper] {
            let (ax, ay, c) = select_mccormick_plane(lx, ux, ly, uy, x0, y0, w, bound_dir);

            for &(x, y) in &corners {
                let z_true = x * y;
                let z_plane = ax * x + ay * y + c;

                match (bound_dir, w >= 0.0) {
                    // Lower bound with positive weight: plane <= true
                    (BoundDir::Lower, true) => prop_assert!(
                        z_plane <= z_true + tol,
                        "Lower/pos: plane({x},{y})={z_plane} > true={z_true}, w={w}"
                    ),
                    // Upper bound with positive weight: plane >= true
                    (BoundDir::Upper, true) => prop_assert!(
                        z_plane >= z_true - tol,
                        "Upper/pos: plane({x},{y})={z_plane} < true={z_true}, w={w}"
                    ),
                    // Lower bound with negative weight: selects upper plane
                    (BoundDir::Lower, false) => prop_assert!(
                        z_plane >= z_true - tol,
                        "Lower/neg: plane({x},{y})={z_plane} < true={z_true}, w={w}"
                    ),
                    // Upper bound with negative weight: selects lower plane
                    (BoundDir::Upper, false) => prop_assert!(
                        z_plane <= z_true + tol,
                        "Upper/neg: plane({x},{y})={z_plane} > true={z_true}, w={w}"
                    ),
                }
            }
        }
    }

    /// Proptest: propagate_linear_binary CROWN backward soundness.
    ///
    /// For random input bounds [la, ua] × [lb, ub] and identity CROWN weights,
    /// the concretized CROWN bounds must contain the true product x_a * x_b
    /// at all sampled points. Tests both McCormick and Middle relaxation modes.
    ///
    /// This covers the full CROWN backward path (not just plane selection).
    /// Part of #3439 acceptance criteria.
    #[ntest::timeout(60000)]
    #[test]
    fn proptest_crown_backward_soundness(
        la in -5.0f32..5.0,
        da in 0.01f32..5.0,
        lb in -5.0f32..5.0,
        db in 0.01f32..5.0,
    ) {
        let ua = la + da;
        let ub = lb + db;

        let input_a = BoundedTensor::new(
            arr1(&[la]).into_dyn(),
            arr1(&[ua]).into_dyn(),
        ).unwrap();
        let input_b = BoundedTensor::new(
            arr1(&[lb]).into_dyn(),
            arr1(&[ub]).into_dyn(),
        ).unwrap();

        let layer = MulBinaryLayer;

        // Identity CROWN weights: output = 1*input, bias = 0
        let identity = LinearBounds::new(
            Array2::eye(1),
            Array1::zeros(1),
            Array2::eye(1),
            Array1::zeros(1),
        ).unwrap();

        for mode in &[MulBinaryRelaxationMode::McCormick, MulBinaryRelaxationMode::Middle] {
            let (bounds_a, bounds_b) = layer
                .propagate_linear_binary(&identity, &input_a, &input_b, *mode)
                .unwrap();

            // Concretize both linear bound sets
            let concrete_a = bounds_a.concretize(&input_a);
            let concrete_b = bounds_b.concretize(&input_b);

            // Combined CROWN lower/upper = sum of per-input concretizations
            // (bias is folded into the linear bounds by propagate_linear_binary)
            let crown_lower = concrete_a.lower()[[0]] + concrete_b.lower()[[0]];
            let crown_upper = concrete_a.upper()[[0]] + concrete_b.upper()[[0]];

            // Sample 21 points per dimension (441 total)
            let n_samples = 21;
            for ia in 0..n_samples {
                let ta = ia as f32 / (n_samples - 1) as f32;
                let xa = la + ta * da;
                for ib in 0..n_samples {
                    let tb = ib as f32 / (n_samples - 1) as f32;
                    let xb = lb + tb * db;
                    let y_true = xa * xb;
                    let tol = soundness_tol(y_true);

                    prop_assert!(
                        y_true >= crown_lower - tol,
                        "Lower bound violation: {mode:?} xa={xa}, xb={xb}, \
                         y={y_true}, lb={crown_lower}, bounds_a=[{la},{ua}], bounds_b=[{lb},{ub}]"
                    );
                    prop_assert!(
                        y_true <= crown_upper + tol,
                        "Upper bound violation: {mode:?} xa={xa}, xb={xb}, \
                         y={y_true}, ub={crown_upper}, bounds_a=[{la},{ua}], bounds_b=[{lb},{ub}]"
                    );
                }
            }
        }
    }
}

/// Regression (#vnncomp-aw-soundness self-audit): broadcast McCormick coefficient absorption
/// is a false-proof unless the per-cell f32-accumulation error is carried. input_a is broadcast
/// [1] -> [n], so EVERY one of the n inner-j iterations f32-accumulates into the SAME coefficient
/// cell [0,0]. A large downstream weight followed by thousands of tiny ones makes the naive f32
/// running sum absorb the tiny terms (stored coeff TIGHTER than the true real coeff). After the
/// fix the returned bounds MUST carry a non-zero coeff_err for the absorbed side (depth n,
/// gamma_n*S, mirroring the matmul crown_dense McCormick fix).
#[test]
fn test_mccormick_coeff_err_broadcast_absorption_aw_soundness() -> Result<()> {
    let layer = MulBinaryLayer;
    let n: usize = 4096; // large fan-in so tiny terms are absorbed in f32

    let mut w = Array2::<f32>::zeros((1, n));
    w[[0, 0]] = 8192.0;
    for j in 1..n {
        w[[0, j]] = 3.0e-4;
    }
    let bias = arr1(&[0.0f32]);
    let bounds = LinearBounds::new_or_conservative(w.clone(), bias.clone(), w.clone(), bias)?;

    // input_a broadcast [1] -> all n output positions map to a_idx == 0.
    let a = BoundedTensor::new(arr1(&[1.0f32]).into_dyn(), arr1(&[2.0f32]).into_dyn())?;
    let b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[n]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[n]), 3.0f32),
    )?;

    let (bounds_a, _bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    assert!(
        bounds_a.has_coeff_err(),
        "McCormick broadcast absorption must carry a coeff_err (else false-proof)"
    );
    Ok(())
}

/// FALSE-BOUND regression for the MulBinary McCormick planes
/// (#mulbinary-mccormick-f32, `docs/MULBINARY_MCCORMICK_F32_CANCELLATION_2026-07-28.md`).
///
/// The coefficients used to be built in f32:
/// ```text
/// let beta_l = (x_l - x_u) * r_l + x_u;
/// ```
/// Once `|x_l| < ulp(x_u)`, `fl(x_l - x_u)` returns `-x_u` and the `+ x_u`
/// round-trip yields `0`, so `beta_l` COLLAPSES from `x_l` to `0` and the
/// "lower" plane rises above the true product. Measured pre-fix on
/// `x=[-1, 2^24]`, `y=[1, 100]`, `r_l=1`: claimed lower bound `-1` at corner
/// `(-1, 100)` where the true product is `-100`.
///
/// `plane - product` is BILINEAR, so its extremum over the box is attained at a
/// corner; checking the four corners is therefore an EXACT enclosure test, not a
/// sampled one.
#[test]
fn mccormick_planes_enclose_under_catastrophic_cancellation() {
    // (x_l, x_u, y_l, y_u) cases spanning the cancellation regime and ordinary use.
    let cases = [
        (-1.0f32, 16_777_216.0f32, 1.0f32, 100.0f32), // total collapse of beta_l
        (-0.026_068_168, 1_153_335.9, 7_220.358, 148_963.05), // wide asymmetric
        (-0.5, 340_000.0, -2.1, 9_700.0),             // both operands wide
        (-1.0, 2.0, 1.0, 3.0),                        // benign control
        (1e-30, 1e30, -1e20, 1e20),                   // extreme exponent spread
    ];
    for &(x_l, x_u, y_l, y_u) in &cases {
        for &r in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let (a_l, b_l, n_l, a_u, b_u, n_u) =
                MulBinaryLayer::compute_interpolated_coefficients(x_l, x_u, y_l, y_u, r, r);
            for &(cx, cy) in &[(x_l, y_l), (x_l, y_u), (x_u, y_l), (x_u, y_u)] {
                let prod = f64::from(cx) * f64::from(cy);
                let lower = f64::from(a_l) * f64::from(cx)
                    + f64::from(b_l) * f64::from(cy)
                    + f64::from(n_l);
                let upper = f64::from(a_u) * f64::from(cx)
                    + f64::from(b_u) * f64::from(cy)
                    + f64::from(n_u);
                assert!(
                    lower <= prod,
                    "LOWER plane exceeds product: x=[{x_l:e},{x_u:e}] y=[{y_l:e},{y_u:e}] r={r} \
                     corner=({cx:e},{cy:e}) plane={lower:e} > prod={prod:e} (by {:e})",
                    lower - prod
                );
                assert!(
                    upper >= prod,
                    "UPPER plane below product: x=[{x_l:e},{x_u:e}] y=[{y_l:e},{y_u:e}] r={r} \
                     corner=({cx:e},{cy:e}) plane={upper:e} < prod={prod:e} (by {:e})",
                    prod - upper
                );
            }
        }
    }
}

/// TEETH for [`mccormick_planes_enclose_under_catastrophic_cancellation`].
///
/// A regression test that passes against the broken implementation pins nothing.
/// This reproduces the OLD f32 construction verbatim and asserts the enclosure
/// check FAILS on it — so we know the assertions above have force.
#[test]
fn mccormick_f32_construction_is_actually_caught() {
    // Verbatim pre-fix arithmetic (all f32).
    fn legacy_f32(x_l: f32, x_u: f32, y_l: f32, y_u: f32, r_l: f32) -> (f32, f32, f32) {
        let alpha_l = (y_l - y_u) * r_l + y_u;
        let beta_l = (x_l - x_u) * r_l + x_u;
        let ny_l = (y_u * x_u - y_l * x_l) * r_l - y_u * x_u;
        (alpha_l, beta_l, ny_l)
    }
    let (x_l, x_u, y_l, y_u, r) = (-1.0f32, 16_777_216.0f32, 1.0f32, 100.0f32, 1.0f32);

    let (a, b, n) = legacy_f32(x_l, x_u, y_l, y_u, r);
    assert_eq!(b, 0.0, "precondition: legacy f32 collapses beta_l to zero");
    let worst_legacy = [(x_l, y_l), (x_l, y_u), (x_u, y_l), (x_u, y_u)]
        .iter()
        .map(|&(cx, cy)| {
            f64::from(a) * f64::from(cx) + f64::from(b) * f64::from(cy) + f64::from(n)
                - f64::from(cx) * f64::from(cy)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        worst_legacy > 0.0,
        "TEETH FAILURE: the legacy f32 construction must violate enclosure, got {worst_legacy:e}"
    );

    // The shipped implementation must NOT violate on the same input.
    let (a2, b2, n2) = {
        let (a_l, b_l, n_l, ..) =
            MulBinaryLayer::compute_interpolated_coefficients(x_l, x_u, y_l, y_u, r, r);
        (a_l, b_l, n_l)
    };
    let worst_fixed = [(x_l, y_l), (x_l, y_u), (x_u, y_l), (x_u, y_u)]
        .iter()
        .map(|&(cx, cy)| {
            f64::from(a2) * f64::from(cx) + f64::from(b2) * f64::from(cy) + f64::from(n2)
                - f64::from(cx) * f64::from(cy)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        worst_fixed <= 0.0,
        "fixed construction still violates enclosure by {worst_fixed:e}"
    );
}
