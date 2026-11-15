// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::common::BoundPropagation;

fn exact_affine_layer(gamma: f32) -> BatchNormLayer {
    let gamma = ArrayD::from_shape_vec(IxDyn(&[1]), vec![gamma]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let mean = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0e6]).unwrap();
    let variance = ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap();
    BatchNormLayer::new(&gamma, &beta, &mean, &variance, 1.0).unwrap()
}

fn point_box(value: f32) -> BoundedTensor {
    let point = ArrayD::from_shape_vec(IxDyn(&[1]), vec![value]).unwrap();
    BoundedTensor::new(point.clone(), point).unwrap()
}

#[test]
fn ordinary_affine_stays_finite_and_encloses_both_interval_endpoints() {
    require_f64_interval_proof_environment()
        .expect("the normal test environment must admit binary64 proof arithmetic");
    let scale = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    assert!(output.lower()[[0]].is_finite() && output.upper()[[0]].is_finite());
    assert!(output.lower()[[0]] <= 5.0);
    assert!(output.upper()[[0]] >= 7.0);
}

#[test]
fn cancellation_at_the_product_magnitude_is_enclosed_for_both_scale_signs() {
    let point = 1_000_000.06_f32;
    for (gamma, truth) in [(3.0, 0.093_75_f64), (-3.0, -0.093_75_f64)] {
        let layer = exact_affine_layer(gamma);
        let output = layer.propagate_ibp(&point_box(point)).unwrap();
        let lower = f64::from(output.lower()[[0]]);
        let upper = f64::from(output.upper()[[0]]);
        assert!(
            lower <= truth && truth <= upper,
            "BatchNorm box [{lower}, {upper}] excludes exact cancellation value {truth}"
        );
    }
}

#[test]
fn representation_exact_evaluation_does_not_lose_a_subnormal_input_to_daz() {
    let scale_value = 1.0e30_f32;
    let input_value = f32::from_bits(1);
    let scale = ArrayD::from_shape_vec(IxDyn(&[1]), vec![scale_value]).unwrap();
    let bias = ArrayD::zeros(IxDyn(&[1]));
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    let output = layer.propagate_ibp(&point_box(input_value)).unwrap();
    let truth = f32_to_f64_exact(scale_value) * f32_to_f64_exact(input_value);
    assert!(
        f64::from(output.lower()[[0]]) <= truth && truth <= f64::from(output.upper()[[0]]),
        "BatchNorm box [{}, {}] excludes exact subnormal-input product {truth:e}",
        output.lower()[[0]],
        output.upper()[[0]]
    );
}

#[test]
fn subnormal_input_still_charges_public_model_error_metadata() {
    let zero = ArrayD::zeros(IxDyn(&[1]));
    let mut layer = BatchNormLayer::from_scale_bias(zero.clone(), zero).unwrap();
    layer.scale_err[[0]] = 1.0e30;
    let input_value = f32::from_bits(1);
    let output = layer.propagate_ibp(&point_box(input_value)).unwrap();
    let radius = f32_to_f64_exact(input_value) * f32_to_f64_exact(layer.scale_err[[0]]);
    assert!(f64::from(output.lower()[[0]]) <= -radius);
    assert!(f64::from(output.upper()[[0]]) >= radius);
}

#[test]
fn negative_subnormal_scale_selects_endpoints_by_its_sign_bit() {
    let scale_value = f32::from_bits(0x8000_0001);
    let scale = ArrayD::from_shape_vec(IxDyn(&[1]), vec![scale_value]).unwrap();
    let bias = ArrayD::zeros(IxDyn(&[1]));
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    let lower_input = 1.0e30_f32;
    let upper_input = 2.0e30_f32;
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![lower_input]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![upper_input]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    let scale_exact = f32_to_f64_exact(scale_value);
    let true_lower = scale_exact * f32_to_f64_exact(upper_input);
    let true_upper = scale_exact * f32_to_f64_exact(lower_input);
    assert!(f64::from(output.lower()[[0]]) <= true_lower);
    assert!(f64::from(output.upper()[[0]]) >= true_upper);
}

#[test]
fn one_sided_saturation_keeps_the_independent_endpoint_finite() {
    let scale = ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    let input = BoundedTensor::new_allow_infinite(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![f32::NEG_INFINITY]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![5.0]).unwrap(),
    )
    .unwrap();
    let output = layer.propagate_ibp(&input).unwrap();
    assert_eq!(output.lower()[[0]], f32::NEG_INFINITY);
    assert!(output.upper()[[0]].is_finite() && output.upper()[[0]] >= 11.0);
}
