// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::layers::common::PatchesPropagation;
use crate::layers::ReLULayer;
use crate::tests::assert_linear_bounds_close;
use crate::{bounds::LinearBounds, BoundedTensor};
use ndarray::{arr1, arr2, ArrayD, IxDyn};

#[test]
fn test_dense_spatial_rows_roundtrip_to_patches_3813() {
    let dense = LinearBounds::new(
        arr2(&[[1.0_f32, -2.0, 3.0, -4.0], [0.5, 1.5, -0.5, 2.5]]),
        arr1(&[0.25_f32, -0.75]),
        arr2(&[[-1.0_f32, 2.0, -3.0, 4.0], [1.25, -0.25, 0.75, -1.5]]),
        arr1(&[1.5_f32, -2.0]),
    )
    .unwrap();

    let patches = PatchesLinearBounds::from_dense_spatial_rows(&dense, (1, 2, 2)).unwrap();
    assert_eq!(patches.row_count, 2);
    assert_eq!(
        patches
            .lower_a
            .patches
            .as_ref()
            .expect("row-aware lower patches")
            .shape(),
        &[2, 1, 2, 2, 1, 1, 1]
    );
    let roundtrip = patches.to_dense().unwrap();
    assert_linear_bounds_close(&dense, &roundtrip, 1e-6, "dense_rows_roundtrip");

    let identity = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
    let identity_dense = identity.to_dense().unwrap();
    assert_linear_bounds_close(
        &LinearBounds::identity(4),
        &identity_dense,
        1e-6,
        "identity_roundtrip",
    );
}

#[test]
fn test_relu_patches_backward_multi_row_dense_parity_3813() {
    let dense_bounds = LinearBounds::new(
        arr2(&[[1.5_f32, -0.25, 0.75, -1.0], [0.5, 1.0, -0.5, 2.0]]),
        arr1(&[0.0_f32, -0.25]),
        arr2(&[[-0.75_f32, 1.25, -1.5, 0.5], [1.0, -0.5, 0.25, 0.75]]),
        arr1(&[0.5_f32, 1.0]),
    )
    .unwrap();
    let patches = PatchesLinearBounds::from_dense_spatial_rows(&dense_bounds, (1, 2, 2)).unwrap();

    let pre_activation = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-1.0_f32, -0.5, 0.25, -2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![2.0_f32, 1.5, 1.0, 0.5]).unwrap(),
    )
    .unwrap();

    let relu = ReLULayer;
    let dense_result = relu
        .propagate_linear_with_bounds(&dense_bounds, &pre_activation)
        .unwrap();
    let patches_result = relu
        .propagate_patches_with_bounds(&patches, &pre_activation)
        .unwrap();
    let CrownBounds::Patches(patches_result) = patches_result else {
        panic!("expected row-aware ReLU backward to remain in Patches mode");
    };
    let patches_dense = patches_result.to_dense().unwrap();

    assert_linear_bounds_close(&dense_result, &patches_dense, 1e-6, "relu_multi_row");
}
