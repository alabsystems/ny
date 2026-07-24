// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::*;
use crate::BoundedTensor;
use ndarray::{arr1, arr2};

#[test]
fn test_from_bounded_per_position_radius_calculation() {
    // Kills: replace - with / in line 973 (radius = (upper - lower) / 2.0)

    // Create bounds with lower=0.0, upper=2.0
    // Correct radius = (2.0 - 0.0) / 2.0 = 1.0
    // If - was /: radius = (2.0 / 0.0) / 2.0 = inf

    let lower = arr2(&[[0.0_f32, 1.0], [2.0, 3.0]]).into_dyn();
    let upper = arr2(&[[2.0_f32, 3.0], [4.0, 5.0]]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

    // Check that radius is correct (1.0 for all elements)
    let out_bounds = z.to_bounded_tensor().unwrap();

    // Lower should be original lower (0.0, 1.0, 2.0, 3.0)
    assert!(
        (out_bounds.lower()[[0, 0]] - 0.0).abs() < 1e-6,
        "lower bound should be 0.0, got {}",
        out_bounds.lower()[[0, 0]]
    );
    assert!((out_bounds.lower()[[0, 1]] - 1.0).abs() < 1e-6);
    assert!((out_bounds.lower()[[1, 0]] - 2.0).abs() < 1e-6);
    assert!((out_bounds.lower()[[1, 1]] - 3.0).abs() < 1e-6);

    // Upper should be original upper (2.0, 3.0, 4.0, 5.0)
    assert!(
        (out_bounds.upper()[[0, 0]] - 2.0).abs() < 1e-6,
        "upper bound should be 2.0, got {}",
        out_bounds.upper()[[0, 0]]
    );

    // Verify bounds are finite (would be inf if - was /)
    assert!(
        out_bounds.lower().iter().all(|&x| x.is_finite()),
        "lower should be finite"
    );
    assert!(
        out_bounds.upper().iter().all(|&x| x.is_finite()),
        "upper should be finite"
    );
}

#[test]
fn test_from_bounded_per_position_n_error_terms() {
    // Kills: replace * with + in line 986 (n_error_terms = batch_size * seq)
    // Kills: replace * with / in line 986

    // Create 2x3 bounds (seq=2, dim=3)
    // Correct n_error_terms = 1 * 2 = 2 (batch_size=1, seq=2)
    // If * was +: n_error_terms = 1 + 2 = 3
    // If * was /: n_error_terms = 1 / 2 = 0

    let lower = arr2(&[[0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]]).into_dyn();
    let upper = arr2(&[[1.0_f32, 1.0, 1.0], [1.0, 1.0, 1.0]]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

    // Should have 2 error terms (one per sequence position)
    assert_eq!(
        z.n_error_terms, 2,
        "should have batch*seq = 1*2 = 2 error terms"
    );

    // Coeffs should be (1 + 2) x 2 x 3 = 3 x 2 x 3
    assert_eq!(
        z.coeffs.shape(),
        &[3, 2, 3],
        "coeffs shape should be (1+n_err, seq, dim)"
    );
}

#[test]
fn test_from_bounded_per_position_coeffs_shape() {
    // Kills: replace + with - in line 988 (1 + n_error_terms)
    // Kills: replace + with * in line 988

    // Create 3x4 bounds (seq=3, dim=4)
    let lower = ndarray::Array2::<f32>::zeros((3, 4)).into_dyn();
    let upper = ndarray::Array2::<f32>::ones((3, 4)).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

    // n_error_terms = 3
    // Correct coeffs first dim = 1 + 3 = 4
    // If + was -: 1 - 3 = -2 (would fail on Array allocation)
    // If + was *: 1 * 3 = 3 (would be missing center row)

    assert_eq!(
        z.coeffs.shape()[0],
        4,
        "first dim of coeffs should be 1 + n_error_terms = 4"
    );

    // Verify center row exists and has correct values
    let center = z.center();
    assert_eq!(center.shape(), &[3, 4], "center should be 3x4");

    // Center should be (lower + upper) / 2 = 0.5
    for val in center.iter() {
        assert!((*val - 0.5).abs() < 1e-6, "center should be 0.5");
    }
}

#[test]
fn test_from_bounded_per_position_error_index() {
    // Kills: replace + with - in line 995 (1 + b * seq + s)
    // Kills: replace + with * in line 995
    // Kills: replace * with + in line 995 (b * seq)
    // Kills: replace * with / in line 995

    // Create 3D bounds: (2, 3, 4) = (batch=2, seq=3, dim=4)
    let lower = ndarray::Array3::<f32>::zeros((2, 3, 4)).into_dyn();
    let mut upper = ndarray::Array3::<f32>::zeros((2, 3, 4)).into_dyn();

    // Give each position a unique radius so we can verify error assignment
    // Position (b, s) gets radius = (b * 10 + s + 1) / 10
    for b in 0..2 {
        for s in 0..3 {
            for d in 0..4 {
                upper[[b, s, d]] = (b * 10 + s + 1) as f32 / 5.0;
            }
        }
    }

    let bounds = BoundedTensor::new(lower, upper).unwrap();
    let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

    // n_error_terms = 2 * 3 = 6
    assert_eq!(z.n_error_terms, 6);

    // Error term index should be 1 + b * seq + s
    // For (b=0, s=0): err = 1 + 0*3 + 0 = 1
    // For (b=0, s=1): err = 1 + 0*3 + 1 = 2
    // For (b=0, s=2): err = 1 + 0*3 + 2 = 3
    // For (b=1, s=0): err = 1 + 1*3 + 0 = 4
    // For (b=1, s=1): err = 1 + 1*3 + 1 = 5
    // For (b=1, s=2): err = 1 + 1*3 + 2 = 6

    // Verify each position has error in the correct slot
    // The radius for (b,s) is (b*10+s+1)/10, so error coeff = radius = (b*10+s+1)/10

    // Check (b=0, s=0) has error in slot 1
    let r_00 = 0.5 * 1.0 / 5.0; // radius = upper/2 = 1/10
    assert!(
        (z.coeffs[[1, 0, 0, 0]] - r_00).abs() < 1e-5,
        "err 1 should have radius for (0,0), got {}",
        z.coeffs[[1, 0, 0, 0]]
    );
    // Other error slots for this position should be 0
    assert!(
        (z.coeffs[[2, 0, 0, 0]] - 0.0).abs() < 1e-6,
        "err 2 should be 0 for (0,0)"
    );

    // Check (b=1, s=1) has error in slot 5
    let r_11 = 0.5 * 12.0 / 5.0; // b*10+s+1 = 12, radius = 12/10 = 1.2
    assert!(
        (z.coeffs[[5, 1, 1, 0]] - r_11).abs() < 1e-4,
        "err 5 should have radius for (1,1), got {}",
        z.coeffs[[5, 1, 1, 0]]
    );
}

#[test]
fn test_from_bounded_per_position_batch_shape() {
    // Regression: coeffs must be 4D for batched inputs.

    let lower = ndarray::Array3::<f32>::zeros((2, 3, 4)).into_dyn();
    let upper = ndarray::Array3::<f32>::ones((2, 3, 4)).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let z = ZonotopeTensor::from_bounded_tensor_per_position(&bounds).unwrap();

    assert_eq!(
        z.coeffs.shape(),
        &[1 + 2 * 3, 2, 3, 4],
        "coeffs shape should be (1+n_err, batch, seq, dim)"
    );
}

#[test]
fn test_from_bounded_per_position_2d_accepts_2d() {
    // Kills: replace != with == in line 1029

    // 2D input should be accepted
    let lower = arr2(&[[0.0_f32, 1.0]]).into_dyn();
    let upper = arr2(&[[2.0_f32, 3.0]]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let result = ZonotopeTensor::from_bounded_tensor_per_position_2d(&bounds);
    assert!(result.is_ok(), "2D bounds should be accepted");
}

#[test]
fn test_from_bounded_per_position_2d_rejects_1d() {
    // Kills: replace != with == in line 1029

    // 1D input should be rejected
    let lower = arr1(&[0.0_f32, 1.0]).into_dyn();
    let upper = arr1(&[2.0_f32, 3.0]).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let result = ZonotopeTensor::from_bounded_tensor_per_position_2d(&bounds);
    assert!(
        result.is_err(),
        "1D bounds should be rejected by 2D variant"
    );
}

#[test]
fn test_from_bounded_per_position_2d_rejects_3d() {
    // Kills: replace != with == in line 1029

    // 3D input should be rejected
    let lower = ndarray::Array3::<f32>::zeros((2, 3, 4)).into_dyn();
    let upper = ndarray::Array3::<f32>::ones((2, 3, 4)).into_dyn();
    let bounds = BoundedTensor::new(lower, upper).unwrap();

    let result = ZonotopeTensor::from_bounded_tensor_per_position_2d(&bounds);
    assert!(
        result.is_err(),
        "3D bounds should be rejected by 2D variant"
    );
}
