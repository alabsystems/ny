// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::div::{backward_div_to_numerator, DivBackwardResult};
use crate::{BoundedTensor, LinearBounds};
use ndarray::{arr1, arr2};

fn div_overflow_fixture_3438() -> (LinearBounds, BoundedTensor, BoundedTensor, BoundedTensor) {
    let node_lb = LinearBounds::new(
        arr2(&[[f32::MAX]]),
        arr1(&[0.0_f32]),
        arr2(&[[f32::MAX]]),
        arr1(&[0.0_f32]),
    )
    .expect("node linear bounds should be well-formed");
    let input_a_bounds =
        BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("numerator bounds should be valid");
    let input_b_bounds = BoundedTensor::new(
        arr1(&[1.0e-38_f32]).into_dyn(),
        arr1(&[2.0e-38_f32]).into_dyn(),
    )
    .expect("denominator bounds should be valid");
    let node_output_bounds =
        BoundedTensor::new(arr1(&[0.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
            .expect("output bounds should be valid");
    (node_lb, input_a_bounds, input_b_bounds, node_output_bounds)
}

fn assert_conservative_scalar_bounds_3438(bounds: &LinearBounds, label: &str) {
    assert_eq!(
        bounds.lower_a()[[0, 0]],
        0.0,
        "{label} lower A should fall back to conservative zero coefficients"
    );
    assert_eq!(
        bounds.upper_a()[[0, 0]],
        0.0,
        "{label} upper A should fall back to conservative zero coefficients"
    );
    assert!(
        bounds.lower_b()[0].is_infinite() && bounds.lower_b()[0].is_sign_negative(),
        "{label} lower bias should fall back to -Inf, got {}",
        bounds.lower_b()[0]
    );
    assert!(
        bounds.upper_b()[0].is_infinite() && bounds.upper_b()[0].is_sign_positive(),
        "{label} upper bias should fall back to +Inf, got {}",
        bounds.upper_b()[0]
    );
}

#[test]
fn test_graph_alpha_div_overflow_falls_back_3438() {
    let (node_lb, input_a_bounds, input_b_bounds, node_output_bounds) = div_overflow_fixture_3438();
    let result = backward_div_to_numerator(
        "div",
        &node_lb,
        &input_a_bounds,
        &input_b_bounds,
        &node_output_bounds,
    )
    .expect("Div backward should succeed");
    let DivBackwardResult::PropagateNumerator(bounds) = result else {
        panic!("positive denominators with matching shapes should propagate numerator bounds");
    };

    assert_conservative_scalar_bounds_3438(&bounds, "graph-alpha Div");
}
