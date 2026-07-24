// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::super::*;

fn nd_interval_downstream() -> crate::BatchedLinearBounds {
    crate::BatchedLinearBounds::new(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 1, 2]),
            vec![
                1.0, -1.5, // i=0: mixed-sign downstream to trigger inf + (-inf) accumulation.
                1.0, -1.5, // i=1
            ],
        )
        .expect("lower_a"),
        ArrayD::zeros(IxDyn(&[1, 2, 1])),
        ArrayD::from_shape_vec(
            IxDyn(&[1, 2, 1, 2]),
            vec![
                1.5,
                -1.0, // keep the same sign pattern while making the downstream interval-valued.
                1.5, -1.0,
            ],
        )
        .expect("upper_a"),
        ArrayD::zeros(IxDyn(&[1, 2, 1])),
        vec![1, 2, 2],
        vec![1, 2, 1],
    )
    .expect("N-D downstream")
}

fn finite_bounds(shape: &[usize], lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(shape), lower),
        ArrayD::from_elem(IxDyn(shape), upper),
    )
    .expect("finite bounds")
}

fn assert_no_nan(bounds: &crate::BatchedLinearBounds, label: &str) {
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
}

fn assert_k_path_zero(bounds: &crate::BatchedLinearBounds) {
    assert!(
        bounds.lower_a().iter().all(|v| *v == 0.0),
        "N-D one-sided compose should concretize K into bias (lower_a)"
    );
    assert!(
        bounds.upper_a().iter().all(|v| *v == 0.0),
        "N-D one-sided compose should concretize K into bias (upper_a)"
    );
    assert!(
        bounds.lower_b().iter().all(|v| *v == 0.0),
        "N-D one-sided compose should keep the K lower bias at zero"
    );
    assert!(
        bounds.upper_b().iter().all(|v| *v == 0.0),
        "N-D one-sided compose should keep the K upper bias at zero"
    );
}

#[ntest::timeout(10000)]
#[test]
fn propagate_nd_one_sided_q_nan_bias_widens_to_infinity_4204() {
    let mut q_upper = ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32);
    q_upper[[0, 0, 0]] = f32::NAN;
    let q_bounds =
        BoundedTensor::new_unchecked(ArrayD::from_elem(IxDyn(&[1, 2, 2]), -1.0_f32), q_upper)
            .expect("Q bounds with NaN");
    let k_bounds = finite_bounds(&[1, 2, 2], -0.5, 0.5);

    let (q_result, k_result) = BilinearCrownLayer::new(true, None)
        .propagate_nd_one_sided(
            &nd_interval_downstream(),
            &q_bounds,
            &k_bounds,
            2,
            2,
            2,
            1.0,
            &[1],
        )
        .expect("Q-side NaN should widen the N-D bias instead of leaking NaN");

    assert_no_nan(&q_result, "Q result");
    assert_eq!(
        q_result.lower_b()[[0, 0, 0]],
        f32::NEG_INFINITY,
        "Q-side NaN should widen the affected lower bias to -inf",
    );
    assert_eq!(
        q_result.upper_b()[[0, 0, 0]],
        f32::INFINITY,
        "Q-side NaN should widen the affected upper bias to +inf",
    );

    assert_no_nan(&k_result, "K result");
    assert_k_path_zero(&k_result);
}

#[ntest::timeout(10000)]
#[test]
fn propagate_nd_one_sided_k_nan_bias_widens_to_infinity_4204() {
    let q_bounds = finite_bounds(&[1, 2, 2], -1.0, 1.0);
    let mut k_upper = ArrayD::from_elem(IxDyn(&[1, 2, 2]), 0.5_f32);
    k_upper[[0, 0, 0]] = f32::NAN;
    let k_bounds =
        BoundedTensor::new_unchecked(ArrayD::from_elem(IxDyn(&[1, 2, 2]), -0.5_f32), k_upper)
            .expect("K bounds with NaN");

    let (q_result, k_result) = BilinearCrownLayer::new(true, None)
        .propagate_nd_one_sided(
            &nd_interval_downstream(),
            &q_bounds,
            &k_bounds,
            2,
            2,
            2,
            1.0,
            &[1],
        )
        .expect("K-side NaN should widen the N-D bias instead of leaking NaN");

    assert_no_nan(&q_result, "Q result");
    assert_eq!(
        q_result.lower_b()[[0, 0, 0]],
        f32::NEG_INFINITY,
        "K-side NaN should widen the affected lower bias to -inf",
    );
    assert_eq!(
        q_result.upper_b()[[0, 0, 0]],
        f32::INFINITY,
        "K-side NaN should widen the affected upper bias to +inf",
    );

    assert_no_nan(&k_result, "K result");
    assert_k_path_zero(&k_result);
}

#[ntest::timeout(10000)]
#[test]
fn propagate_nd_one_sided_infinite_k_coeff_nan_widens_to_infinity_4204() {
    let q_bounds = finite_bounds(&[1, 2, 2], -1.0, 1.0);
    let k_bounds = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), f32::INFINITY),
        ArrayD::from_elem(IxDyn(&[1, 2, 2]), f32::INFINITY),
    )
    .expect("infinite K bounds");

    let (q_result, k_result) = BilinearCrownLayer::new(true, None)
        .propagate_nd_one_sided(
            &nd_interval_downstream(),
            &q_bounds,
            &k_bounds,
            2,
            2,
            2,
            1.0,
            &[1],
        )
        .expect("mixed-sign infinite K contributions should widen coefficient NaNs");

    assert_no_nan(&q_result, "Q result");
    assert_eq!(
        q_result.lower_a()[[0, 0, 0, 0]],
        f32::NEG_INFINITY,
        "opposing infinite lower contributions should widen to -inf",
    );
    assert_eq!(
        q_result.upper_a()[[0, 0, 0, 0]],
        f32::INFINITY,
        "opposing infinite upper contributions should widen to +inf",
    );

    assert_no_nan(&k_result, "K result");
    assert_k_path_zero(&k_result);
}
