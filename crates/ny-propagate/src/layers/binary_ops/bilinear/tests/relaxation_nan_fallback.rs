// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::super::*;

fn infinite_bounds(shape: &[usize]) -> BoundedTensor {
    BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(shape), f32::INFINITY),
        ArrayD::from_elem(IxDyn(shape), f32::INFINITY),
    )
    .expect("infinite bounds")
}

fn checkerboard_downstream() -> crate::BatchedLinearBounds {
    let checkerboard = vec![1.0_f32, -1.0, -1.0, 1.0];
    crate::BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), checkerboard.clone()).expect("lower_a"),
        ArrayD::zeros(IxDyn(&[1])),
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), checkerboard).expect("upper_a"),
        ArrayD::zeros(IxDyn(&[1])),
        vec![4],
        vec![1],
    )
    .expect("checkerboard downstream")
}

fn scalar_bias_downstream(lower_bias: f32, upper_bias: f32) -> crate::BatchedLinearBounds {
    crate::BatchedLinearBounds::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).expect("lower_a"),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![lower_bias]).expect("lower_b"),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).expect("upper_a"),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![upper_bias]).expect("upper_b"),
        vec![1],
        vec![1],
    )
    .expect("scalar downstream")
}

#[ntest::timeout(10000)]
#[test]
fn compose_backward_broadcast_nan_products_widen_to_infinity_4204() {
    let q_bounds = infinite_bounds(&[2, 1]);
    let k_bounds = infinite_bounds(&[2, 1]);
    let relaxation = BilinearRelaxation::from_bounds(&q_bounds, &k_bounds, true, None)
        .expect("relaxation should build from infinite bounds");

    let (q_result, k_result) = relaxation
        .compose_backward_broadcast(&checkerboard_downstream())
        .expect("broadcast compose should widen NaN contractions");

    for (label, bounds) in [("Q", &q_result), ("K", &k_result)] {
        assert!(
            !bounds.lower_a().iter().any(|v| v.is_nan()),
            "{label} lower_a must not contain NaN",
        );
        assert!(
            !bounds.upper_a().iter().any(|v| v.is_nan()),
            "{label} upper_a must not contain NaN",
        );
        assert!(
            !bounds.lower_b().iter().any(|v| v.is_nan()),
            "{label} lower_b must not contain NaN",
        );
        assert!(
            !bounds.upper_b().iter().any(|v| v.is_nan()),
            "{label} upper_b must not contain NaN",
        );
        assert!(
            bounds.lower_a().iter().all(|v| *v == f32::NEG_INFINITY),
            "{label} lower_a should widen every NaN contraction to -inf",
        );
        assert!(
            bounds.upper_a().iter().all(|v| *v == f32::INFINITY),
            "{label} upper_a should widen every NaN contraction to +inf",
        );
        assert!(
            bounds.lower_b().iter().all(|v| *v == f32::NEG_INFINITY),
            "{label} lower_b should widen NaN bias sums to -inf",
        );
        assert!(
            bounds.upper_b().iter().all(|v| *v == f32::INFINITY),
            "{label} upper_b should widen NaN bias sums to +inf",
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn compose_backward_interval_bias_nan_widens_to_infinity_4204() {
    let q_bounds = infinite_bounds(&[1, 1]);
    let k_bounds = infinite_bounds(&[1, 1]);
    let relaxation = BilinearRelaxation::from_bounds(&q_bounds, &k_bounds, true, None)
        .expect("relaxation should build from infinite bounds");

    let (q_result, k_result) = relaxation
        .compose_backward(&scalar_bias_downstream(f32::INFINITY, f32::NEG_INFINITY))
        .expect("interval compose should widen NaN bias sums");

    for (label, bounds) in [("Q", &q_result), ("K", &k_result)] {
        assert!(
            !bounds.lower_b().iter().any(|v| v.is_nan()),
            "{label} lower_b must not contain NaN",
        );
        assert!(
            !bounds.upper_b().iter().any(|v| v.is_nan()),
            "{label} upper_b must not contain NaN",
        );
        assert_eq!(
            bounds.lower_b()[[0]],
            f32::NEG_INFINITY,
            "{label} lower bias should widen to -inf after NaN accumulation",
        );
        assert_eq!(
            bounds.upper_b()[[0]],
            f32::INFINITY,
            "{label} upper bias should widen to +inf after NaN accumulation",
        );
    }
}
