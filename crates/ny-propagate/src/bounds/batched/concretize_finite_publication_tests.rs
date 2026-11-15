// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::BatchedLinearBounds;
use ndarray::{ArrayD, IxDyn};
use ny_core::CROWN_COEFF_MAX;
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

const SAFE_LARGE_COEFFICIENT: f32 = 1e9;
const LARGE_POINT: f32 = 1e30;

fn deadline() -> Instant {
    Instant::now() + Duration::from_secs(60)
}

fn one_by_one(lower_coefficient: f32, upper_coefficient: f32) -> BatchedLinearBounds {
    let lower_a = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![lower_coefficient]).unwrap();
    let upper_a = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![upper_coefficient]).unwrap();
    let bias = ArrayD::zeros(IxDyn(&[1]));
    BatchedLinearBounds::new(lower_a, bias.clone(), upper_a, bias, vec![1], vec![1]).unwrap()
}

fn point_box(value: f32) -> BoundedTensor {
    let point = ArrayD::from_shape_vec(IxDyn(&[1]), vec![value]).unwrap();
    BoundedTensor::new(point.clone(), point).unwrap()
}

fn endpoints(bounds: &BoundedTensor) -> (f32, f32) {
    (bounds.lower()[[0]], bounds.upper()[[0]])
}

#[test]
fn finite_deadline_publication_keeps_an_ordinary_row_finite_and_sound() {
    let output = one_by_one(2.0, 2.0)
        .concretize_sound_with_deadline(&point_box(3.0), Some(deadline()))
        .unwrap();
    let (lower, upper) = endpoints(&output);
    assert!(lower.is_finite() && upper.is_finite());
    assert!(lower <= 6.0 && 6.0 <= upper);
}

#[test]
fn finite_deadline_publication_encloses_both_finite_f32_overflow_directions() {
    for coefficient in [SAFE_LARGE_COEFFICIENT, -SAFE_LARGE_COEFFICIENT] {
        let output = one_by_one(coefficient, coefficient)
            .concretize_sound_with_deadline(&point_box(LARGE_POINT), Some(deadline()))
            .unwrap();
        let (lower, upper) = endpoints(&output);
        let truth = f64::from(coefficient) * f64::from(LARGE_POINT);
        assert!(truth.is_finite() && truth.abs() > f64::from(f32::MAX));
        assert!(
            f64::from(lower) <= truth && truth <= f64::from(upper),
            "[{lower:e}, {upper:e}] excludes finite exact row value {truth:e}"
        );
        if coefficient.is_sign_positive() {
            assert!(
                lower.is_finite(),
                "positive finite overflow cannot publish +inf below"
            );
        } else {
            assert!(
                upper.is_finite(),
                "negative finite overflow cannot publish -inf above"
            );
        }
    }
}

#[test]
fn finite_deadline_publication_repairs_wrong_direction_infinity_sentinels() {
    for bias_value in [f32::INFINITY, f32::NEG_INFINITY] {
        let coefficients = ArrayD::zeros(IxDyn(&[1, 1]));
        let bias = ArrayD::from_shape_vec(IxDyn(&[1]), vec![bias_value]).unwrap();
        let bounds = BatchedLinearBounds::new(
            coefficients.clone(),
            bias.clone(),
            coefficients,
            bias,
            vec![1],
            vec![1],
        )
        .unwrap();
        let output = bounds
            .concretize_sound_with_deadline(&point_box(0.0), Some(deadline()))
            .unwrap();
        assert_eq!(endpoints(&output), (f32::NEG_INFINITY, f32::INFINITY));
    }
}

#[test]
fn finite_deadline_publication_preserves_the_independent_finite_half() {
    // The upper coefficient reaches the fail-closed guard; the lower relation
    // remains independently valid and should not be discarded.
    let bounds = one_by_one(1.0, CROWN_COEFF_MAX);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    )
    .unwrap();
    let legacy = bounds.concretize_sound(&input).unwrap();
    let finite = bounds
        .concretize_sound_with_deadline(&input, Some(deadline()))
        .unwrap();
    for output in [&legacy, &finite] {
        let (lower, upper) = endpoints(output);
        assert!(lower.is_finite() && lower <= -2.0);
        assert_eq!(upper, f32::INFINITY);
    }
}
