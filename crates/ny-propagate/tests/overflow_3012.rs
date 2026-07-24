// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{array, ArrayD, IxDyn};
use ny_core::{checked_dim_product, checked_shape_product, NyError};
use ny_propagate::layers::{Conv2dLayer, ConvTranspose2dLayer};
use ny_propagate::BatchedLinearBounds;

fn make_conv2d(weight: f32, bias: Option<f32>, input_hw: (usize, usize)) -> Conv2dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![weight]).expect("kernel");
    let bias = bias.map(|b| array![b]);
    Conv2dLayer::with_input_shape(kernel, bias, (1, 1), (0, 0), input_hw.0, input_hw.1)
        .expect("valid conv2d")
}

fn make_convtranspose2d(
    weight: f32,
    bias: Option<f32>,
    input_hw: (usize, usize),
) -> ConvTranspose2dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![weight]).expect("kernel");
    let bias = bias.map(|b| array![b]);
    ConvTranspose2dLayer::with_input_shape(kernel, bias, (1, 1), (0, 0), input_hw.0, input_hw.1)
        .expect("valid convtranspose2d")
}

// ---------- Integration tests: Conv2d / ConvTranspose2d ----------
// These layers accept input dimensions as constructor parameters, so overflow
// is detected in the production guard BEFORE ndarray touches the shape.

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_batched_input_dims_overflow_returns_error_3012() {
    let layer = make_conv2d(1.0, None, (usize::MAX, 2));
    let bounds = BatchedLinearBounds::identity(&[1, 1]).expect("small incoming bounds");

    let err = layer
        .propagate_linear_batched(&bounds, None)
        .expect_err("overflowing Conv2d input dimensions should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("input dims product overflows")),
        "expected input-dims overflow error, got: {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_batched_input_dims_overflow_returns_error_3012() {
    let layer = make_convtranspose2d(1.0, None, (usize::MAX, 2));
    let bounds = BatchedLinearBounds::identity(&[1, 1]).expect("small incoming bounds");

    let err = layer
        .propagate_linear_batched(&bounds)
        .expect_err("overflowing ConvTranspose2d input dimensions should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("input dims product overflows")),
        "expected input-dims overflow error, got: {err:?}"
    );
}

// ---------- Unit tests: checked_shape_product for pool-typical shapes ----------
//
// Integration tests for AvgPool/MaxPool overflow guards are not possible because
// ndarray panics in debug mode when constructing arrays whose shape product
// overflows usize — the internal `Dimension::size()` method uses unchecked `*`
// (dimension_trait.rs:142). Since pool layers extract dimensions from
// BoundedTensor::shape(), which requires a valid ndarray, we cannot create a
// tensor with an overflowing shape to trigger the guard.
//
// Instead, we directly test `checked_shape_product` with the same shape patterns
// that the pool guards use: [batch, channels, in_h, in_w]. The Conv2d integration
// tests above prove the full error path (checked_shape_product → ok_or_else →
// NyError::InvalidSpec) works end-to-end; the pool guards use the identical
// code pattern (see average.rs:362 and max.rs:276).
//
// The remaining #3012 packets add guards to ndarray-backed shapes (MulBinary and
// decomposed normalization row counts). Those exact overflows are similarly
// impossible to realize end-to-end because ndarray must already accept the shape,
// so we regression-test the shared helper that now routes those guard failures.

#[test]
fn test_checked_shape_product_4d_pool_overflow_returns_none_3012() {
    // AvgPool/MaxPool guard: checked_shape_product(&[batch, ch, in_h, in_w])
    // Shape [2, 1, usize::MAX, 2]: product overflows
    assert!(
        checked_shape_product(&[2, 1, usize::MAX, 2]).is_none(),
        "4D pool shape with usize::MAX height should overflow"
    );
}

#[test]
fn test_checked_shape_product_3d_pool_overflow_returns_none_3012() {
    // AvgPool/MaxPool 3D input guard: checked_shape_product(&[ch, in_h, in_w])
    assert!(
        checked_shape_product(&[1, usize::MAX, 2]).is_none(),
        "3D pool shape with usize::MAX height should overflow"
    );
}

#[test]
fn test_checked_shape_product_no_overflow_returns_some_3012() {
    // Sanity: valid shapes return Some
    assert_eq!(checked_shape_product(&[2, 3, 4, 5]), Some(120));
    assert_eq!(checked_shape_product(&[1, 1, 1]), Some(1));
    assert_eq!(checked_shape_product(&[]), Some(1));
}

#[test]
fn test_checked_shape_product_single_max_returns_some_3012() {
    // A single usize::MAX dimension is fine (product = usize::MAX, no overflow)
    assert_eq!(checked_shape_product(&[usize::MAX]), Some(usize::MAX));
}

#[test]
fn test_checked_shape_product_two_large_dims_overflow_3012() {
    // Two dimensions whose product overflows: (usize::MAX/2 + 1) * 2
    let half_plus_one = (usize::MAX / 2) + 1;
    assert!(
        checked_shape_product(&[2, half_plus_one]).is_none(),
        "product of 2 * (usize::MAX/2 + 1) should overflow"
    );
}

#[test]
fn test_checked_dim_product_overflow_returns_invalid_spec_3012() {
    let err = checked_dim_product(&[2, (usize::MAX / 2) + 1], "overflow helper")
        .expect_err("overflowing dimension product should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg)
            if msg.contains("overflow helper")
                && msg.contains("dimension product overflows usize")),
        "expected checked_dim_product InvalidSpec, got: {err:?}"
    );
}

#[test]
fn test_checked_dim_product_no_overflow_returns_value_3012() {
    assert_eq!(
        checked_dim_product(&[2, 3, 4], "no overflow").expect("small product should succeed"),
        24
    );
}
