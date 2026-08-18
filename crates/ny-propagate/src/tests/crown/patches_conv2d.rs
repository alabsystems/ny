// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for `Conv2dLayer::conv2d_patches_backward` via `propagate_patches`.
//!
//! Covers:
//! - Identity patches: first Conv2d in backward chain creates initial patches
//!   from the convolution kernel.
//! - Dense-Patches parity: patches backward and dense backward produce
//!   equivalent bounds when both are converted to dense form.
//! - Bias contribution: patches bias accumulation matches dense bias.
//! - Stride/padding composition: composed stride and padding are correct.
//!
//! Part of #3463

use crate::bounds::patches::{CrownBounds, PatchGeometry, PatchesLinearBounds};
use crate::layers::common::PatchesPropagation;
use crate::layers::convolution::conv2d::Conv2dLayer;
use crate::layers::BoundPropagation;
use ndarray::{arr1, arr2, Array1, ArrayD, IxDyn};
use std::borrow::Cow;

fn into_owned_linear_bounds(
    bounds: Cow<'_, crate::bounds::LinearBounds>,
) -> crate::bounds::LinearBounds {
    match bounds {
        Cow::Owned(lb) => lb,
        Cow::Borrowed(lb) => lb.clone(),
    }
}

use super::super::assert_linear_bounds_close;

/// Identity patches through a 1-in, 2-out, 3x3 Conv2d.
///
/// Starting from identity patches (output_shape = (2, 2, 2) for a 4x4 input
/// with 3x3 kernel, stride 1, padding 0), the backward should produce patches
/// of shape (2, 2, 2, 1, 3, 3) where each patch contains the conv kernel
/// values for the corresponding output channel.
#[test]
fn test_conv2d_patches_backward_identity_creates_kernel_patches() {
    // 1 in_channel, 2 out_channels, 3x3 kernel
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 3, 3]),
        vec![
            // out_c=0: ascending values
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // out_c=1: descending values
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ],
    )
    .unwrap();

    let mut conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    // Input: 1 channel, 4x4 spatial → output: 2 channels, 2x2 spatial
    conv.set_input_shape(4, 4);

    // Identity patches at output shape (2, 2, 2), input shape (1, 4, 4)
    let identity_bounds = PatchesLinearBounds::identity((2, 2, 2), (1, 4, 4));

    let result = conv.propagate_patches(&identity_bounds).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected Patches output, got Dense");
    };

    // Output patches should be (2, 2, 2, 1, 3, 3)
    let lower_patches = result
        .lower_a
        .patches
        .as_ref()
        .expect("materialized patches");
    assert_eq!(lower_patches.shape(), &[2, 2, 2, 1, 3, 3]);

    // Each patch for out_c=0 should contain kernel[0,:,:,:]
    assert_eq!(lower_patches[[0, 0, 0, 0, 0, 0]], 1.0);
    assert_eq!(lower_patches[[0, 0, 0, 0, 1, 1]], 5.0);
    assert_eq!(lower_patches[[0, 0, 0, 0, 2, 2]], 9.0);

    // Each patch for out_c=1 should contain kernel[1,:,:,:]
    assert_eq!(lower_patches[[1, 0, 0, 0, 0, 0]], 9.0);
    assert_eq!(lower_patches[[1, 0, 0, 0, 1, 1]], 5.0);
    assert_eq!(lower_patches[[1, 0, 0, 0, 2, 2]], 1.0);

    // All spatial positions within the same output channel have identical patches
    assert_eq!(
        lower_patches[[0, 0, 0, 0, 0, 0]],
        lower_patches[[0, 1, 1, 0, 0, 0]]
    );
    assert_eq!(
        lower_patches[[1, 0, 0, 0, 0, 0]],
        lower_patches[[1, 0, 1, 0, 0, 0]]
    );

    // Geometry should match the conv.
    assert_eq!(
        result.lower_a.geometry,
        PatchGeometry::affine((1, 1), (0, 0, 0, 0))
    );

    // Identity is cleared
    assert!(!result.lower_a.identity);
    assert!(!result.upper_a.identity);

    // Zero bias (no bias on conv)
    assert!(result.lower_b.iter().all(|&v| v == 0.0));
    assert!(result.upper_b.iter().all(|&v| v == 0.0));
}

/// Dense-Patches parity: both backward paths produce equivalent A-matrices.
///
/// Creates a small Conv2d (1 in_c, 1 out_c, 2x2 kernel), runs backward in
/// both Dense and Patches mode, converts Patches to Dense, and verifies
/// element-wise equivalence.
#[test]
fn test_conv2d_patches_backward_dense_parity() {
    // 1→1, 2x2 kernel, stride 1, no padding
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    let mut conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    // Input: 1x3x3 → Output: 1x2x2
    conv.set_input_shape(3, 3);

    // Dense backward: identity LinearBounds of dimension out_dim = 1*2*2 = 4
    let out_dim = 4;
    let in_dim = 9; // 1*3*3
    let dense_identity = crate::bounds::LinearBounds::identity(out_dim);
    let dense_result = conv.propagate_linear(&dense_identity).unwrap();
    let dense_lb = match dense_result {
        Cow::Owned(ref lb) => lb,
        Cow::Borrowed(lb) => lb,
    };

    // Patches backward: identity PatchesLinearBounds
    let patches_identity = PatchesLinearBounds::identity((1, 2, 2), (1, 3, 3));
    let patches_result = conv.propagate_patches(&patches_identity).unwrap();

    // Convert patches to dense for comparison
    let patches_dense = match patches_result {
        CrownBounds::Patches(ref pb) => pb.to_dense().unwrap(),
        CrownBounds::Dense(ref lb) => lb.clone(),
    };

    // Compare A-matrices element by element
    assert_eq!(dense_lb.lower_a.shape(), patches_dense.lower_a.shape());
    assert_eq!(dense_lb.upper_a.shape(), patches_dense.upper_a.shape());
    assert_eq!(dense_lb.lower_a.shape(), &[out_dim, in_dim]);

    for i in 0..out_dim {
        for j in 0..in_dim {
            assert!(
                (dense_lb.lower_a[[i, j]] - patches_dense.lower_a[[i, j]]).abs() < 1e-6,
                "lower_a mismatch at [{}, {}]: dense={}, patches={}",
                i,
                j,
                dense_lb.lower_a[[i, j]],
                patches_dense.lower_a[[i, j]],
            );
            assert!(
                (dense_lb.upper_a[[i, j]] - patches_dense.upper_a[[i, j]]).abs() < 1e-6,
                "upper_a mismatch at [{}, {}]: dense={}, patches={}",
                i,
                j,
                dense_lb.upper_a[[i, j]],
                patches_dense.upper_a[[i, j]],
            );
        }
    }

    // Bias parity
    for i in 0..out_dim {
        assert!(
            (dense_lb.lower_b[i] - patches_dense.lower_b[i]).abs() < 1e-6,
            "lower_b mismatch at [{}]: dense={}, patches={}",
            i,
            dense_lb.lower_b[i],
            patches_dense.lower_b[i],
        );
        assert!(
            (dense_lb.upper_b[i] - patches_dense.upper_b[i]).abs() < 1e-6,
            "upper_b mismatch at [{}]: dense={}, patches={}",
            i,
            dense_lb.upper_b[i],
            patches_dense.upper_b[i],
        );
    }
}

/// Conv2d with bias: patches backward accumulates bias correctly.
///
/// Verifies that the bias term flows through to patches lower_b/upper_b.
/// For identity incoming bounds with zero bias, the output bias should equal
/// the conv bias replicated across spatial positions.
#[test]
fn test_conv2d_patches_backward_with_bias() {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 2, 2]),
        vec![
            1.0, 0.0, 0.0, 1.0, // out_c=0: diagonal
            0.5, 0.5, 0.5, 0.5, // out_c=1: uniform
        ],
    )
    .unwrap();
    let bias = Array1::from_vec(vec![0.25_f32, -0.5]);

    let mut conv = Conv2dLayer::new(kernel, Some(bias), (1, 1), (0, 0)).unwrap();
    // Input: 1x3x3 → Output: 2x2x2
    conv.set_input_shape(3, 3);

    let identity_bounds = PatchesLinearBounds::identity((2, 2, 2), (1, 3, 3));
    let result = conv.propagate_patches(&identity_bounds).unwrap();

    let (lower_b, upper_b) = match &result {
        CrownBounds::Patches(pb) => (&pb.lower_b, &pb.upper_b),
        CrownBounds::Dense(lb) => (&lb.lower_b, &lb.upper_b),
    };

    // For identity incoming bounds, output bias = conv bias replicated per spatial position.
    // out_dim = 2*2*2 = 8, with bias[0]=0.25 for positions 0..4, bias[1]=-0.5 for 4..8
    assert_eq!(lower_b.len(), 8);
    for i in 0..4 {
        assert!(
            (lower_b[i] - 0.25).abs() < 1e-6,
            "lower_b[{}] = {} expected 0.25",
            i,
            lower_b[i],
        );
    }
    for i in 4..8 {
        assert!(
            (lower_b[i] - (-0.5)).abs() < 1e-6,
            "lower_b[{}] = {} expected -0.5",
            i,
            lower_b[i],
        );
    }
    // Lower and upper bias should be identical for identity incoming bounds
    for i in 0..8 {
        assert!(
            (lower_b[i] - upper_b[i]).abs() < 1e-6,
            "lower_b[{}] != upper_b[{}]: {} vs {}",
            i,
            i,
            lower_b[i],
            upper_b[i],
        );
    }
}

/// Conv2d with stride=2: verifies composed stride is correct.
///
/// A 1→1 Conv2d with stride=2 on a 4x4 input produces 2x2 output (with 1x1 kernel).
/// The patches stride should be (2, 2) after backward through the identity start.
#[test]
fn test_conv2d_patches_backward_stride_composition() {
    // Minimal 1x1 kernel (identity-like), stride=2
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).unwrap();

    let mut conv = Conv2dLayer::new(kernel, None, (2, 2), (0, 0)).unwrap();
    // Input: 1x4x4 → Output: 1x2x2 (stride=2 on 4x4 with 1x1 kernel)
    conv.set_input_shape(4, 4);

    let identity_bounds = PatchesLinearBounds::identity((1, 2, 2), (1, 4, 4));
    let result = conv.propagate_patches(&identity_bounds).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected Patches output");
    };

    // Stride should be (2, 2) from the conv.
    let expected_geometry = PatchGeometry::affine((2, 2), (0, 0, 0, 0));
    assert_eq!(result.lower_a.geometry, expected_geometry);
    assert_eq!(result.upper_a.geometry, expected_geometry);

    // Patches shape: (1, 2, 2, 1, 1, 1) — kernel is 1x1
    let patches = result.lower_a.patches.as_ref().expect("patches");
    assert_eq!(patches.shape(), &[1, 2, 2, 1, 1, 1]);

    // All patches should be 1.0 (identity kernel value)
    assert!((patches[[0, 0, 0, 0, 0, 0]] - 1.0).abs() < 1e-6);
    assert!((patches[[0, 1, 1, 0, 0, 0]] - 1.0).abs() < 1e-6);
}

/// Conv2d with padding: verifies composed padding is correct.
#[test]
fn test_conv2d_patches_backward_padding_composition() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), vec![1.0; 9]).unwrap();

    let mut conv = Conv2dLayer::new(kernel, None, (1, 1), (1, 1)).unwrap();
    // Input: 1x3x3, padding=1 → Output: 1x3x3
    conv.set_input_shape(3, 3);

    let identity_bounds = PatchesLinearBounds::identity((1, 3, 3), (1, 3, 3));
    let result = conv.propagate_patches(&identity_bounds).unwrap();

    // May fall back to dense because kernel (3x3) covers entire input (3x3)
    // Either way, verify the result is valid
    match result {
        CrownBounds::Patches(ref pb) => {
            assert_eq!(
                pb.lower_a.geometry,
                PatchGeometry::affine((1, 1), (1, 1, 1, 1))
            );
        }
        CrownBounds::Dense(ref lb) => {
            // Dense fallback is expected when kernel covers entire input.
            // Verify dimensions are correct.
            assert_eq!(lb.lower_a.shape(), &[9, 9]);
        }
    }
}

#[test]
fn test_conv2d_patches_typed_refuses_anchored_geometry() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).unwrap();
    let mut conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
    conv.set_input_shape(2, 2);
    let mut bounds = PatchesLinearBounds::identity((1, 2, 2), (1, 2, 2));
    let anchored = PatchGeometry::anchored(vec![0, 1], vec![0, 1]).unwrap();
    bounds.lower_a.geometry = anchored.clone();
    bounds.upper_a.geometry = anchored;

    let error = conv
        .propagate_patches(&bounds)
        .expect_err("Conv2d is not yet implemented for anchored geometry");
    assert!(matches!(
        error,
        ny_core::NyError::UnsupportedConfiguration(_)
    ));
}

#[test]
fn test_conv2d_patches_backward_grouped_identity_zeroes_cross_group_channels() {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[4, 2, 1, 1]),
        vec![
            1.0_f32, 2.0, // out_c=0, group 0
            3.0, 4.0, // out_c=1, group 0
            5.0, 6.0, // out_c=2, group 1
            7.0, 8.0, // out_c=3, group 1
        ],
    )
    .unwrap();

    let conv = Conv2dLayer::with_input_shape_full(kernel, None, (1, 1), (0, 0), 2, 2, 2).unwrap();
    let identity_bounds = PatchesLinearBounds::identity((4, 2, 2), (4, 2, 2));

    let result = conv.propagate_patches(&identity_bounds).unwrap();
    let CrownBounds::Patches(result) = result else {
        panic!("expected grouped Conv2d identity path to stay in Patches mode");
    };

    let lower_patches = result
        .lower_a
        .patches
        .as_ref()
        .expect("grouped identity patches should materialize");
    assert_eq!(lower_patches.shape(), &[4, 2, 2, 4, 1, 1]);

    assert_eq!(lower_patches[[0, 0, 0, 0, 0, 0]], 1.0);
    assert_eq!(lower_patches[[0, 0, 0, 1, 0, 0]], 2.0);
    assert_eq!(lower_patches[[0, 0, 0, 2, 0, 0]], 0.0);
    assert_eq!(lower_patches[[0, 0, 0, 3, 0, 0]], 0.0);

    assert_eq!(lower_patches[[2, 1, 1, 0, 0, 0]], 0.0);
    assert_eq!(lower_patches[[2, 1, 1, 1, 0, 0]], 0.0);
    assert_eq!(lower_patches[[2, 1, 1, 2, 0, 0]], 5.0);
    assert_eq!(lower_patches[[2, 1, 1, 3, 0, 0]], 6.0);
}

#[test]
fn test_conv2d_patches_backward_grouped_non_identity_spatial_dense_parity() {
    let grouped_kernel = ArrayD::from_shape_vec(
        IxDyn(&[4, 2, 2, 2]),
        vec![
            1.0_f32, 0.5, -0.25, 2.0, //
            -1.0, 0.75, 1.5, -0.5, //
            0.25, -0.5, 1.25, 0.5, //
            2.0, -1.5, 0.0, 0.75, //
            1.0, -0.25, 0.5, 1.5, //
            -0.75, 1.0, -1.25, 0.25, //
            0.5, 0.5, -0.5, 1.0, //
            -1.0, 0.25, 0.75, -0.75, //
        ],
    )
    .unwrap();
    let grouped_conv =
        Conv2dLayer::with_input_shape_full(grouped_kernel, None, (2, 2), (0, 0), 2, 6, 6).unwrap();

    let downstream_kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 4, 2, 2]),
        (0..32).map(|v| (v as f32 - 12.0) / 8.0).collect(),
    )
    .unwrap();
    let downstream_conv =
        Conv2dLayer::with_input_shape(downstream_kernel, None, (1, 1), (0, 0), 3, 3).unwrap();

    let dense_identity = crate::bounds::LinearBounds::identity(2 * 2 * 2);
    let dense_after_downstream =
        into_owned_linear_bounds(downstream_conv.propagate_linear(&dense_identity).unwrap());
    let dense_after_both = into_owned_linear_bounds(
        grouped_conv
            .propagate_linear(&dense_after_downstream)
            .unwrap(),
    );

    let patches_identity = PatchesLinearBounds::identity((2, 2, 2), (2, 2, 2));
    let patches_after_downstream = downstream_conv
        .propagate_patches(&patches_identity)
        .unwrap();
    let CrownBounds::Patches(patches_after_downstream) = patches_after_downstream else {
        panic!("expected downstream 2x2 Conv2d to remain in Patches mode");
    };
    let patches_after_both = grouped_conv
        .propagate_patches(&patches_after_downstream)
        .unwrap();
    let CrownBounds::Patches(patches_after_both) = patches_after_both else {
        panic!("expected grouped spatial Conv2d composition to remain in Patches mode");
    };
    assert_eq!(
        patches_after_both
            .lower_a
            .patches
            .as_ref()
            .expect("grouped spatial composition should materialize patches")
            .shape(),
        &[2, 2, 2, 4, 4, 4]
    );
    let expected_geometry = PatchGeometry::affine((2, 2), (0, 0, 0, 0));
    assert_eq!(patches_after_both.lower_a.geometry, expected_geometry);
    assert_eq!(patches_after_both.upper_a.geometry, expected_geometry);
    let patches_dense = patches_after_both.to_dense().unwrap();

    assert_linear_bounds_close(&dense_after_both, &patches_dense, 1e-6, "grouped_spatial");
}

#[test]
fn test_conv2d_patches_backward_multi_row_dense_parity_3813() {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -0.5, 0.25, 0.75]).unwrap();
    let mut conv = Conv2dLayer::new(kernel, Some(arr1(&[0.2_f32])), (1, 1), (0, 0)).unwrap();
    conv.set_input_shape(3, 3);

    let dense_bounds = crate::bounds::LinearBounds::new(
        arr2(&[[1.0_f32, -2.0, 0.5, 0.25], [-0.75, 0.5, 1.5, -1.25]]),
        arr1(&[0.3_f32, -0.6]),
        arr2(&[[0.5_f32, 1.0, -1.5, 0.75], [1.25, -0.25, 0.5, -0.5]]),
        arr1(&[-0.2_f32, 0.4]),
    )
    .unwrap();

    let dense_result = into_owned_linear_bounds(conv.propagate_linear(&dense_bounds).unwrap());
    let patches_bounds = PatchesLinearBounds::from_dense_spatial_rows(&dense_bounds, (1, 2, 2))
        .expect("dense classifier rows should convert to row-aware patches");
    let patches_result = conv.propagate_patches(&patches_bounds).unwrap();
    let CrownBounds::Patches(patches_result) = patches_result else {
        panic!("expected row-aware Conv2d backward to remain in Patches mode");
    };
    let patches_dense = patches_result.to_dense().unwrap();

    assert_linear_bounds_close(&dense_result, &patches_dense, 1e-6, "multi_row_conv2d");
}

/// Crash guard: a Conv2d whose `groups` metadata is inconsistent with its channel
/// counts must return a clean error instead of panicking in the identity-patches
/// build loop. The constructor enforces `out_c % groups == 0` and `groups >= 1`,
/// so we build a valid layer and then corrupt `groups` to simulate inconsistent
/// metadata reaching the patches backward (e.g. from a malformed import).
///
/// `out_c % groups != 0` makes `out_c_per_group = out_c / groups` truncate so that
/// `group_idx = oc / out_c_per_group` exceeds `groups - 1`, pushing the derived
/// input-channel index `ic = group_idx * in_c_per_group + ic_local` past `in_c`
/// and panicking on `patches[[.., ic, ..]]`. With the guard it is an InvalidSpec.
#[test]
fn test_conv2d_patches_inconsistent_groups_out_c_not_divisible_returns_error() {
    // Valid 4-out, groups=2 layer: kernel (4, 2, 1, 1), in_c = 2*2 = 4.
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[4, 2, 1, 1]),
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let mut conv =
        Conv2dLayer::with_input_shape_full(kernel, None, (1, 1), (0, 0), 2, 2, 2).unwrap();
    // Corrupt groups so 4 % 3 != 0 -> out_c_per_group truncates to 1.
    conv.groups = 3;

    let identity_bounds = PatchesLinearBounds::identity((4, 2, 2), (4, 2, 2));
    let result = conv.propagate_patches(&identity_bounds);
    assert!(
        matches!(result, Err(ny_core::NyError::InvalidSpec(_))),
        "expected InvalidSpec for non-divisible groups, got {result:?}"
    );
}

/// Crash guard: `groups == 0` would divide by zero at `out_c / groups`. The guard
/// must reject it with a clean InvalidSpec instead of panicking.
#[test]
fn test_conv2d_patches_inconsistent_groups_zero_returns_error() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![1.0_f32, 2.0]).unwrap();
    let mut conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2).unwrap();
    // Corrupt groups to 0 (constructor forbids this; simulate malformed metadata).
    conv.groups = 0;

    let identity_bounds = PatchesLinearBounds::identity((2, 2, 2), (1, 2, 2));
    let result = conv.propagate_patches(&identity_bounds);
    assert!(
        matches!(result, Err(ny_core::NyError::InvalidSpec(_))),
        "expected InvalidSpec for groups == 0, got {result:?}"
    );
}
