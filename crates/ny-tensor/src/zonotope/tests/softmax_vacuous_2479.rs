// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Non-vacuous zonotope softmax regressions for #2479.
//!
//! These tests use per-element perturbations instead of `from_input_shared`, so
//! the evaluated corners move off softmax's shift-invariant all-ones direction.

use super::super::*;
use ndarray::arr1;

use crate::BoundedTensor;

fn softmax_1d(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = values.iter().map(|value| (value - max_val).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    exp_values.into_iter().map(|value| value / sum).collect()
}

fn assert_elementwise_input_symbols(z: &ZonotopeTensor, epsilon: f32) {
    let coeffs = z.coeffs();
    let dim = z.shape()[0];
    for error_row in 0..dim {
        for output_dim in 0..dim {
            let actual = coeffs[[1 + error_row, output_dim]];
            let expected = if error_row == output_dim {
                epsilon
            } else {
                0.0
            };
            assert!(
                (actual - expected).abs() < 1e-6,
                "from_input_elementwise row {} col {} should be {}, got {}",
                1 + error_row,
                output_dim,
                expected,
                actual
            );
        }
    }
}

fn assert_softmax_bounds_contain_all_elementwise_corners(
    center: &[f32],
    epsilon: f32,
    bounds: &BoundedTensor,
    context: &str,
) {
    let center_softmax = softmax_1d(center);
    let total_corners = 1usize << center.len();
    let mut observed_non_shift_change = false;
    for corner_mask in 0..total_corners {
        let point: Vec<f32> = center
            .iter()
            .enumerate()
            .map(|(dim, &value)| {
                let sign = if (corner_mask >> dim) & 1 == 0 {
                    -1.0_f32
                } else {
                    1.0_f32
                };
                value + epsilon * sign
            })
            .collect();
        let true_softmax = softmax_1d(&point);
        if true_softmax
            .iter()
            .zip(center_softmax.iter())
            .any(|(&corner_value, &center_value)| (corner_value - center_value).abs() > 1e-6)
        {
            observed_non_shift_change = true;
        }

        for (dim, &actual) in true_softmax.iter().enumerate() {
            assert!(
                bounds.lower()[dim] <= actual,
                "{context}: lower[{dim}] = {} must contain softmax {:?}[{dim}] = {} at corner mask {corner_mask:0width$b}",
                bounds.lower()[dim],
                point,
                actual,
                width = center.len()
            );
            assert!(
                bounds.upper()[dim] >= actual,
                "{context}: upper[{dim}] = {} must contain softmax {:?}[{dim}] = {} at corner mask {corner_mask:0width$b}",
                bounds.upper()[dim],
                point,
                actual,
                width = center.len()
            );
        }
    }
    assert!(
        observed_non_shift_change,
        "{context}: elementwise corners must move softmax off the shift-invariant center output"
    );
}

#[test]
fn test_softmax_affine_with_error_1d() {
    let values = arr1(&[2.0_f32, 2.0, 2.0]);
    let epsilon = 0.1_f32;
    let z = ZonotopeTensor::from_input_elementwise(&values.clone().into_dyn(), epsilon);
    assert_eq!(
        z.n_error_terms, 3,
        "elementwise input must create 3 error terms"
    );
    assert_elementwise_input_symbols(&z, epsilon);

    let result = z.softmax_affine(-1).unwrap();

    let center = result.center();
    for i in 0..3 {
        assert!(
            (center[i] - 1.0 / 3.0).abs() < 1e-4,
            "softmax center[{}] should be 1/3, got {}",
            i,
            center[i]
        );
    }

    assert_eq!(
        result.n_error_terms,
        z.n_error_terms + values.len(),
        "softmax 1D should add one approximation error term per output element"
    );

    let bounds = result.to_bounded_tensor().unwrap();
    assert_softmax_bounds_contain_all_elementwise_corners(
        values.as_slice().expect("1D values should be contiguous"),
        epsilon,
        &bounds,
        "#2479 with_error_1d",
    );

    for i in 0..3 {
        assert!(
            bounds.lower()[i] >= 0.0,
            "softmax lower bound should be >= 0"
        );
        assert!(
            bounds.upper()[i] <= 1.0,
            "softmax upper bound should be <= 1"
        );
        assert!(
            bounds.lower()[i] < bounds.upper()[i],
            "softmax bounds should have nonzero width"
        );
    }
}

#[test]
fn test_softmax_affine_jacobian_correctness() {
    let center = arr1(&[1.0_f32, 2.0]);
    let epsilon = 0.1_f32;
    let z = ZonotopeTensor::from_input_elementwise(&center.clone().into_dyn(), epsilon);

    let result = z.softmax_affine(-1).unwrap();
    let coeffs = result.coeffs();

    let delta = 1e-4_f32;
    for input_dim in 0..2 {
        let mut plus = center.to_vec();
        plus[input_dim] += delta;
        let mut minus = center.to_vec();
        minus[input_dim] -= delta;
        let softmax_plus = softmax_1d(&plus);
        let softmax_minus = softmax_1d(&minus);

        for output_dim in 0..2 {
            let numerical_derivative =
                (softmax_plus[output_dim] - softmax_minus[output_dim]) / (2.0 * delta);
            let propagated_derivative = coeffs[[1 + input_dim, output_dim]] / epsilon;
            assert!(
                (propagated_derivative - numerical_derivative).abs() < 5e-3,
                "softmax Jacobian column {input_dim}, row {output_dim}: propagated {} vs numerical {}",
                propagated_derivative,
                numerical_derivative
            );
        }
    }

    let bounds = result.to_bounded_tensor().unwrap();
    assert_softmax_bounds_contain_all_elementwise_corners(
        center.as_slice().expect("1D center should be contiguous"),
        epsilon,
        &bounds,
        "#2479 jacobian_correctness",
    );
}

#[test]
fn test_softmax_affine_bounds_valid() {
    let values = arr1(&[1.0_f32, 2.0, 3.0, 2.5]);
    let epsilon = 0.1_f32;
    let z = ZonotopeTensor::from_input_elementwise(&values.clone().into_dyn(), epsilon);

    let result = z.softmax_affine(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();
    assert_softmax_bounds_contain_all_elementwise_corners(
        values.as_slice().expect("1D values should be contiguous"),
        epsilon,
        &bounds,
        "#2479 bounds_valid",
    );

    for i in 0..4 {
        assert!(
            bounds.lower()[i] >= -0.2,
            "softmax lower[{}] = {} should be near 0 for small epsilon",
            i,
            bounds.lower()[i]
        );
        assert!(
            bounds.upper()[i] <= 1.2,
            "softmax upper[{}] = {} should be near 1 for small epsilon",
            i,
            bounds.upper()[i]
        );
    }
}

#[test]
fn test_softmax_affine_large_epsilon_sound() {
    let center = [1.0_f32, 2.0, 3.0];
    let values = arr1(&center);
    let epsilon = 0.5_f32;
    let z = ZonotopeTensor::from_input_elementwise(&values.into_dyn(), epsilon);

    let result = z.softmax_affine(-1).unwrap();
    let bounds = result.to_bounded_tensor().unwrap();
    assert_softmax_bounds_contain_all_elementwise_corners(
        &center,
        epsilon,
        &bounds,
        "#2479 large_epsilon_sound",
    );
}
