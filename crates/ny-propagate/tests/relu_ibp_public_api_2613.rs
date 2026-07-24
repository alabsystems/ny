// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::arr1;
use ny_propagate::layers::{BoundPropagation, ReLULayer};
use ny_tensor::BoundedTensor;

#[ntest::timeout(10000)]
#[test]
fn test_relu_ibp_public_api_preserves_infinite_endpoints() {
    let relu = ReLULayer::new();
    let input = BoundedTensor::new_allow_infinite(
        arr1(&[f32::NEG_INFINITY, -2.0, 1.0]).into_dyn(),
        arr1(&[-1.0, f32::INFINITY, f32::INFINITY]).into_dyn(),
    )
    .expect("finite+infinite interval should be constructible");

    let output = relu
        .propagate_ibp(&input)
        .expect("ReLU IBP should accept infinite endpoints");

    assert_eq!(output.lower()[[0]], 0.0);
    assert_eq!(output.upper()[[0]], 0.0);
    assert_eq!(output.lower()[[1]], 0.0);
    assert!(output.upper()[[1]].is_infinite() && output.upper()[[1]].is_sign_positive());
    assert_eq!(output.lower()[[2]], 1.0);
    assert!(output.upper()[[2]].is_infinite() && output.upper()[[2]].is_sign_positive());
}

#[ntest::timeout(10000)]
#[test]
fn test_relu_ibp_public_api_matches_point_relu_on_degenerate_intervals() {
    let relu = ReLULayer::new();
    let input = BoundedTensor::new(
        arr1(&[-3.0, -0.0, 0.0, 2.5]).into_dyn(),
        arr1(&[-3.0, -0.0, 0.0, 2.5]).into_dyn(),
    )
    .expect("degenerate intervals are valid bounded tensors");

    let output = relu
        .propagate_ibp(&input)
        .expect("ReLU IBP should succeed on degenerate intervals");

    assert_eq!(output.lower()[[0]], 0.0);
    assert_eq!(output.upper()[[0]], 0.0);
    assert_eq!(output.lower()[[1]], 0.0);
    assert_eq!(output.upper()[[1]], 0.0);
    assert_eq!(output.lower()[[2]], 0.0);
    assert_eq!(output.upper()[[2]], 0.0);
    assert_eq!(output.lower()[[3]], 2.5);
    assert_eq!(output.upper()[[3]], 2.5);
}
