// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proof coverage tests for MulBinary batched CROWN and McCormick edge cases.
//!
//! Added by Prover as part of #2519 proof_coverage audit.
//! Tests exercise:
//! - `propagate_linear_batched_binary()` (253 lines, was zero test coverage)
//! - McCormick CROWN with zero-crossing intervals (missing coverage)
//! - IBP with both-negative intervals (missing coverage)
//!
//! NOTE: `select_mccormick_plane` and `BoundDir` are `pub(super)` and cannot be
//! tested directly from `src/tests/`. The McCormick negative-weight paths are
//! tested indirectly through `propagate_linear_binary` with non-identity bounds
//! that produce negative weights on the diagonal.
//!
//! NOTE: `relu_intercept_score` is private to `branching/mod.rs`. Direct unit
//! tests for its edge cases (NaN, Inf, near-zero width) must be added inline
//! by a Worker. See #2519.

use ndarray::{arr1, Array2, ArrayD, IxDyn};
use ny_core::Result;

use crate::layers::binary_ops::MulBinaryLayer;
use crate::{BatchedLinearBounds, LinearBounds, MulBinaryRelaxationMode};
use ny_tensor::BoundedTensor;

use super::assert_close;

const TOL: f32 = 1e-3;

// ---- CROWN McCormick zero-crossing soundness ----

/// McCormick with both inputs crossing zero — the critical case for plane selection.
/// All 4 corners and interior zero must be enclosed by the linear bounds.
#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_soundness_zero_crossing_both_inputs() -> Result<()> {
    let layer = MulBinaryLayer;
    let bounds = LinearBounds::identity(1);
    let a = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    for &x in &[-2.0, 0.0, 3.0] {
        for &y in &[-1.0, 0.0, 4.0] {
            let z_true = x * y;
            let z_lower =
                bounds_a.lower_a[[0, 0]] * x + bounds_b.lower_a[[0, 0]] * y + bounds_a.lower_b[0];
            let z_upper =
                bounds_a.upper_a[[0, 0]] * x + bounds_b.upper_a[[0, 0]] * y + bounds_a.upper_b[0];
            assert!(
                z_lower <= z_true + TOL,
                "McCormick lower unsound at ({x},{y}): lb={z_lower}, true={z_true}"
            );
            assert!(
                z_upper >= z_true - TOL,
                "McCormick upper unsound at ({x},{y}): ub={z_upper}, true={z_true}"
            );
        }
    }
    Ok(())
}

/// McCormick with negative incoming weight — exercises the w < 0 branch
/// in select_mccormick_plane where lower and upper plane roles swap.
#[ntest::timeout(10000)]
#[test]
fn test_crown_mccormick_negative_incoming_weight_soundness() -> Result<()> {
    let layer = MulBinaryLayer;
    // Non-identity bounds: negative weight on the diagonal
    let bounds = LinearBounds::new(
        Array2::from_elem((1, 1), -1.0),
        arr1(&[0.0]),
        Array2::from_elem((1, 1), -1.0),
        arr1(&[0.0]),
    )
    .unwrap();
    let a = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;
    let (bounds_a, bounds_b) =
        layer.propagate_linear_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    // With -1 weight, the output is -z where z = x*y.
    // Soundness: lower(-z) <= -z_true and upper(-z) >= -z_true at corners.
    for &x in &[1.0, 3.0] {
        for &y in &[2.0, 4.0] {
            let neg_z_true = -(x * y);
            let z_lower =
                bounds_a.lower_a[[0, 0]] * x + bounds_b.lower_a[[0, 0]] * y + bounds_a.lower_b[0];
            let z_upper =
                bounds_a.upper_a[[0, 0]] * x + bounds_b.upper_a[[0, 0]] * y + bounds_a.upper_b[0];
            assert!(
                z_lower <= neg_z_true + TOL,
                "Neg weight lower unsound at (x={x}, y={y}): lb={z_lower}, true={neg_z_true}"
            );
            assert!(
                z_upper >= neg_z_true - TOL,
                "Neg weight upper unsound at (x={x}, y={y}): ub={z_upper}, true={neg_z_true}"
            );
        }
    }
    Ok(())
}

// ---- IBP negative times negative ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_negative_times_negative_intervals() -> Result<()> {
    let layer = MulBinaryLayer;
    // x in [-4, -1], y in [-3, -2]
    // lower = 2, upper = 12
    let a = BoundedTensor::new(arr1(&[-4.0]).into_dyn(), arr1(&[-1.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[-2.0]).into_dyn())?;
    let out = layer.propagate_ibp_binary(&a, &b)?;
    assert_close(out.lower()[[0]], 2.0, 1e-5);
    assert_close(out.upper()[[0]], 12.0, 1e-5);
    Ok(())
}

// ---- Batched CROWN ----

/// Batched McCormick soundness with 2D identity bounds (simplest batched case).
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_mccormick_soundness_2d() -> Result<()> {
    let layer = MulBinaryLayer;
    let n = 2;
    let identity_bounds = BatchedLinearBounds::identity(&[n])?;

    let a = BoundedTensor::new(arr1(&[1.0, 0.5]).into_dyn(), arr1(&[3.0, 2.5]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0, 1.0]).into_dyn(), arr1(&[4.0, 3.0]).into_dyn())?;

    let (bounds_a, bounds_b) = layer.propagate_linear_batched_binary(
        &identity_bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::McCormick,
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
                    z_lower <= z_true + TOL,
                    "Batched lower unsound at dim={dim}, x={x}, y={y}: \
                     lb={z_lower}, true={z_true}"
                );
                assert!(
                    z_upper >= z_true - TOL,
                    "Batched upper unsound at dim={dim}, x={x}, y={y}: \
                     ub={z_upper}, true={z_true}"
                );
            }
        }
    }
    Ok(())
}

/// Batched Middle relaxation soundness with 2D identity bounds.
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
                    z_lower <= z_true + TOL,
                    "Batched Middle lower unsound at dim={dim}, x={x}, y={y}: \
                     lb={z_lower}, true={z_true}"
                );
                assert!(
                    z_upper >= z_true - TOL,
                    "Batched Middle upper unsound at dim={dim}, x={x}, y={y}: \
                     ub={z_upper}, true={z_true}"
                );
            }
        }
    }
    Ok(())
}

/// Batched CROWN rejects 1D input (needs at least 2D for [out_dim, in_dim]).
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_rejects_1d_input() {
    let layer = MulBinaryLayer;
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
    .expect("valid bounds");
    let b = BoundedTensor::new(
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
        arr1(&[2.0, 3.0, 4.0]).into_dyn(),
    )
    .expect("valid bounds");
    let err = layer
        .propagate_linear_batched_binary(&bounds, &a, &b, MulBinaryRelaxationMode::McCormick)
        .expect_err("1D should be rejected");
    assert!(matches!(err, ny_core::NyError::ShapeMismatch { .. }));
}

/// Verify batched CROWN matches non-batched for single element.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_matches_nonbatched_single_element() -> Result<()> {
    let layer = MulBinaryLayer;

    let a = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let b = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[4.0]).into_dyn())?;

    let nb_bounds = LinearBounds::identity(1);
    let (nb_a, nb_b) =
        layer.propagate_linear_binary(&nb_bounds, &a, &b, MulBinaryRelaxationMode::McCormick)?;

    let bat_bounds = BatchedLinearBounds::identity(&[1])?;
    let (bat_a, bat_b) = layer.propagate_linear_batched_binary(
        &bat_bounds,
        &a,
        &b,
        MulBinaryRelaxationMode::McCormick,
    )?;

    assert!(
        (bat_a.lower_a[[0, 0]] - nb_a.lower_a[[0, 0]]).abs() < TOL,
        "lower_a: batched={}, non-batched={}",
        bat_a.lower_a[[0, 0]],
        nb_a.lower_a[[0, 0]]
    );
    assert!(
        (bat_a.upper_a[[0, 0]] - nb_a.upper_a[[0, 0]]).abs() < TOL,
        "upper_a: batched={}, non-batched={}",
        bat_a.upper_a[[0, 0]],
        nb_a.upper_a[[0, 0]]
    );
    assert!(
        (bat_a.lower_b[[0]] - nb_a.lower_b[0]).abs() < TOL,
        "lower_b: batched={}, non-batched={}",
        bat_a.lower_b[[0]],
        nb_a.lower_b[0]
    );
    assert!(
        (bat_a.upper_b[[0]] - nb_a.upper_b[0]).abs() < TOL,
        "upper_b: batched={}, non-batched={}",
        bat_a.upper_b[[0]],
        nb_a.upper_b[0]
    );
    assert!(
        (bat_b.lower_a[[0, 0]] - nb_b.lower_a[[0, 0]]).abs() < TOL,
        "b lower_a: batched={}, non-batched={}",
        bat_b.lower_a[[0, 0]],
        nb_b.lower_a[[0, 0]]
    );
    assert!(
        (bat_b.upper_a[[0, 0]] - nb_b.upper_a[[0, 0]]).abs() < TOL,
        "b upper_a: batched={}, non-batched={}",
        bat_b.upper_a[[0, 0]],
        nb_b.upper_a[[0, 0]]
    );

    Ok(())
}

/// Verify soundness of batched McCormick with zero-crossing intervals.
#[ntest::timeout(10000)]
#[test]
fn test_batched_crown_mccormick_soundness_zero_crossing() -> Result<()> {
    let layer = MulBinaryLayer;

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
                z_lower <= z_true + TOL,
                "Lower unsound at (x={x}, y={y}): lb={z_lower}, true={z_true}"
            );
            assert!(
                z_upper >= z_true - TOL,
                "Upper unsound at (x={x}, y={y}): ub={z_upper}, true={z_true}"
            );
        }
    }
    Ok(())
}
