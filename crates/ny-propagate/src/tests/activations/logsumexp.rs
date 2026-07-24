// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

fn logsumexp(values: &[f32]) -> f32 {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum_exp.ln()
}

// ==================== LogSumExp tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_ibp_basic() {
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 0.0, 1.0, 2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let logsumexp_layer = LogSumExpLayer::new(vec![-1], true);
    let output = logsumexp_layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2, 1]);

    let expected_lower_0 = logsumexp(&[1.0, 2.0, 3.0]);
    let expected_upper_0 = logsumexp(&[2.0, 3.0, 4.0]);
    assert!(
        (output.lower()[[0, 0]] - expected_lower_0).abs() < 1e-6,
        "Row 0 lower bound mismatch: {} vs {}",
        output.lower()[[0, 0]],
        expected_lower_0
    );
    assert!(
        (output.upper()[[0, 0]] - expected_upper_0).abs() < 1e-6,
        "Row 0 upper bound mismatch: {} vs {}",
        output.upper()[[0, 0]],
        expected_upper_0
    );

    let expected_lower_1 = logsumexp(&[-1.0, 0.0, 1.0]);
    let expected_upper_1 = logsumexp(&[0.0, 1.0, 2.0]);
    assert!(
        (output.lower()[[1, 0]] - expected_lower_1).abs() < 1e-6,
        "Row 1 lower bound mismatch: {} vs {}",
        output.lower()[[1, 0]],
        expected_lower_1
    );
    assert!(
        (output.upper()[[1, 0]] - expected_upper_1).abs() < 1e-6,
        "Row 1 upper bound mismatch: {} vs {}",
        output.upper()[[1, 0]],
        expected_upper_1
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_no_keepdims_shape() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let upper = lower.clone();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let logsumexp_layer = LogSumExpLayer::new(vec![-1], false);
    let output = logsumexp_layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[2]);
}

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_crown_soundness() {
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 0.0, 1.0, 2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let logsumexp_layer = LogSumExpLayer::new(vec![-1], true);
    let ibp_output = logsumexp_layer.propagate_ibp(&input).unwrap();

    let linear_bounds = LinearBounds::identity(2);
    let crown_bounds = logsumexp_layer
        .propagate_linear_with_bounds(&linear_bounds, &input)
        .unwrap();
    let input_len: usize = input.shape().iter().product();
    assert_eq!(crown_bounds.lower_a.ncols(), input_len);
    assert_eq!(crown_bounds.upper_a.ncols(), input_len);
    let concrete = crown_bounds.concretize(&input);

    assert!(
        (concrete.lower()[[0]] - ibp_output.lower()[[0, 0]]).abs() < 1e-6,
        "CROWN lower mismatch: {} vs {}",
        concrete.lower()[[0]],
        ibp_output.lower()[[0, 0]]
    );
    assert!(
        (concrete.upper()[[0]] - ibp_output.upper()[[0, 0]]).abs() < 1e-6,
        "CROWN upper mismatch: {} vs {}",
        concrete.upper()[[0]],
        ibp_output.upper()[[0, 0]]
    );
    assert!(
        (concrete.lower()[[1]] - ibp_output.lower()[[1, 0]]).abs() < 1e-6,
        "CROWN lower mismatch row 1: {} vs {}",
        concrete.lower()[[1]],
        ibp_output.lower()[[1, 0]]
    );
    assert!(
        (concrete.upper()[[1]] - ibp_output.upper()[[1, 0]]).abs() < 1e-6,
        "CROWN upper mismatch row 1: {} vs {}",
        concrete.upper()[[1]],
        ibp_output.upper()[[1, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_layer_enum_dispatch() {
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.5, 2.0, -1.0, 0.25, 1.5]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.5, 1.0, 2.5, -0.5, 0.75, 2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = Layer::LogSumExp(LogSumExpLayer::new(vec![-1], true));
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(layer.layer_type(), "LogSumExp");
    assert_eq!(output.shape(), &[2, 1]);
}

// ==================== Domain validation tests ====================
// Per domain validation policy (designs/2026-02-07-domain-validation-policy.md):
// LogSumExp is Category B — defined for all finite inputs, but non-finite
// inputs indicate upstream numerical issues and must return NumericalInstability.

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_ibp_rejects_infinity_in_lower() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![f32::NEG_INFINITY, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_elem(IxDyn(&[1, 3]), 5.0f32);
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let layer = LogSumExpLayer::new(vec![-1], true);
    let result = layer.propagate_ibp(&input);
    assert!(result.is_err(), "Should reject -inf in lower bounds");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("non-finite"),
        "Error should mention non-finite: {err_msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_ibp_rejects_infinity_in_upper() {
    let lower = ArrayD::from_elem(IxDyn(&[1, 3]), 0.0f32);
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, f32::INFINITY, 3.0]).unwrap();
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let layer = LogSumExpLayer::new(vec![-1], true);
    let result = layer.propagate_ibp(&input);
    assert!(result.is_err(), "Should reject +inf in upper bounds");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("non-finite"),
        "Error should mention non-finite: {err_msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_ibp_rejects_nan() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![f32::NAN, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_elem(IxDyn(&[1, 3]), 5.0f32);
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let layer = LogSumExpLayer::new(vec![-1], true);
    let result = layer.propagate_ibp(&input);
    assert!(result.is_err(), "Should reject NaN in lower bounds");
}

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_crown_rejects_nonfinite_preactivation() {
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![f32::NEG_INFINITY, 1.0, 2.0]).unwrap();
    let upper = ArrayD::from_elem(IxDyn(&[1, 3]), 5.0f32);
    let input = BoundedTensor::new_unchecked(lower, upper).unwrap();

    let layer = LogSumExpLayer::new(vec![-1], true);
    let linear_bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&linear_bounds, &input);
    assert!(
        result.is_err(),
        "CROWN should reject non-finite pre-activation bounds"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("non-finite"),
        "Error should mention non-finite: {err_msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_logsumexp_ibp_accepts_finite_bounds() {
    // Verify that valid finite inputs still work correctly after adding guards
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-10.0, -5.0, 0.0, 5.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), vec![-5.0, 0.0, 5.0, 10.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = LogSumExpLayer::new(vec![-1], true);
    let output = layer.propagate_ibp(&input).unwrap();

    assert_eq!(output.shape(), &[1, 1]);
    // LogSumExp output should be finite for finite inputs
    assert!(output.lower()[[0, 0]].is_finite());
    assert!(output.upper()[[0, 0]].is_finite());
    // Soundness: lower <= upper
    assert!(output.lower()[[0, 0]] <= output.upper()[[0, 0]]);
    // LogSumExp >= max(inputs), so lower bound should be >= max(lower_inputs)
    assert!(output.lower()[[0, 0]] >= logsumexp(&[-10.0, -5.0, 0.0, 5.0]) - 1e-5);
}
