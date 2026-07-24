// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::assert_all_close;
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

#[ntest::timeout(10000)]
#[test]
fn test_add_constant_batched_linear_applies_scalar_bias_shift() {
    let add = AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), 2.5));
    let bounds = BatchedLinearBounds::identity(&[3]).unwrap();

    let result = add.propagate_linear_batched(&bounds).unwrap();

    // For identity A, A @ c = c for each output coordinate.
    for i in 0..3 {
        assert!(
            (result.lower_b[[i]] - 2.5).abs() < 1e-6,
            "lower_b[{i}] should equal scalar shift 2.5, got {}",
            result.lower_b[[i]]
        );
        assert!(
            (result.upper_b[[i]] - 2.5).abs() < 1e-6,
            "upper_b[{i}] should equal scalar shift 2.5, got {}",
            result.upper_b[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_add_constant_batched_matches_non_batched_linear() {
    let add = AddConstantLayer::new(ArrayD::from_elem(IxDyn(&[]), -1.75));

    let linear_bounds = LinearBounds::new(
        arr2(&[[1.0, -2.0, 0.5], [0.2, 0.0, -1.0]]),
        arr1(&[0.3, -0.7]),
        arr2(&[[0.5, 1.0, -0.5], [-0.1, 2.0, 0.4]]),
        arr1(&[1.2, -0.2]),
    )
    .unwrap();

    let expected = add.propagate_linear(&linear_bounds).unwrap().into_owned();

    let batched_bounds = BatchedLinearBounds::from_parts_unchecked(
        linear_bounds.lower_a.clone().into_dyn(),
        linear_bounds.lower_b.clone().into_dyn(),
        linear_bounds.upper_a.clone().into_dyn(),
        linear_bounds.upper_b.clone().into_dyn(),
        vec![linear_bounds.num_inputs()],
        vec![linear_bounds.num_outputs()],
    );

    let actual = add.propagate_linear_batched(&batched_bounds).unwrap();

    let expected_lower_a = expected.lower_a.clone().into_dyn();
    let expected_lower_b = expected.lower_b.clone().into_dyn();
    let expected_upper_a = expected.upper_a.clone().into_dyn();
    let expected_upper_b = expected.upper_b.into_dyn();

    assert_all_close(&actual.lower_a, &expected_lower_a, 1e-6, "lower_a");
    assert_all_close(&actual.lower_b, &expected_lower_b, 1e-6, "lower_b");
    assert_all_close(&actual.upper_a, &expected_upper_a, 1e-6, "upper_a");
    assert_all_close(&actual.upper_b, &expected_upper_b, 1e-6, "upper_b");
}

/// Regression test for #2818: empty constant must return `Err`, not `% 0` panic.
#[test]
fn test_add_constant_crown_backward_empty_constant_returns_error_2818() {
    let layer = AddConstantLayer::new(
        ArrayD::from_shape_vec(IxDyn(&[0]), vec![]).expect("invariant: valid empty shape"),
    );
    let bounds = LinearBounds::identity(3);
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "empty constant must return error, not panic"
    );
}
