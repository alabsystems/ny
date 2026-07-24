// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{arr1, Array2, ArrayD, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;

const MAX_RESIDUAL_ABS: f32 = 1024.0;

/// Regression test for #2277: mean-only CROWN must compute 1/n in f64.
///
/// With `inv_n` in f32 for n=3, `1/3` rounds up, and for large coefficients this
/// creates a residual around 3e7 instead of ~0. The f64 path keeps this residual tiny.
#[ntest::timeout(10000)]
#[test]
fn test_crown_mean_only_f64_inv_n_scalar_regression() -> Result<()> {
    let layer = LayerNormLayer::new_default(3, 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);
    let huge = 1.0e15_f32;
    let lower_a = Array2::from_shape_vec((1, 3), vec![huge, huge, huge])
        .expect("invariant: shape (1,3) matches 3 coefficients");
    let upper_a = lower_a.clone();
    let bounds = LinearBounds::new(lower_a, arr1(&[0.0]), upper_a, arr1(&[0.0]))?;
    let pre_act = BoundedTensor::new(
        arr1(&[-1.0, 0.0, 1.0]).into_dyn(),
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
    )?;

    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    for i in 0..3 {
        assert!(
            result.lower_a[[0, i]].abs() < MAX_RESIDUAL_ABS,
            "dim {i}: lower_a residual {} too large (f32 inv_n regression)",
            result.lower_a[[0, i]]
        );
        assert!(
            result.upper_a[[0, i]].abs() < MAX_RESIDUAL_ABS,
            "dim {i}: upper_a residual {} too large (f32 inv_n regression)",
            result.upper_a[[0, i]]
        );
    }
    Ok(())
}

/// Batched counterpart to #2277 scalar regression.
#[ntest::timeout(10000)]
#[test]
fn test_crown_mean_only_f64_inv_n_batched_regression() -> Result<()> {
    let layer = LayerNormLayer::new_default(3, 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);
    let huge = 1.0e15_f32;

    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![huge; 6])
            .expect("invariant: shape [2,1,3] matches 6 coefficients"),
        ArrayD::zeros(IxDyn(&[2, 1])),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![huge; 6])
            .expect("invariant: shape [2,1,3] matches 6 coefficients"),
        ArrayD::zeros(IxDyn(&[2, 1])),
        vec![2, 3],
        vec![2, 1],
    );
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
            .expect("invariant: shape [2,3] matches 6 lower elements"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("invariant: shape [2,3] matches 6 upper elements"),
    )?;

    let result = layer.propagate_linear_batched_with_bounds(&bounds, &pre_act)?;

    for b in 0..2 {
        for i in 0..3 {
            assert!(
                result.lower_a[[b, 0, i]].abs() < MAX_RESIDUAL_ABS,
                "batch {b} dim {i}: lower_a residual {} too large (f32 inv_n regression)",
                result.lower_a[[b, 0, i]]
            );
            assert!(
                result.upper_a[[b, 0, i]].abs() < MAX_RESIDUAL_ABS,
                "batch {b} dim {i}: upper_a residual {} too large (f32 inv_n regression)",
                result.upper_a[[b, 0, i]]
            );
        }
    }
    Ok(())
}
