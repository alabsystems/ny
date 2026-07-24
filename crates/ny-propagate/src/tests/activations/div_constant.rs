// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== DivConstant tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_ibp_positive_divisor() {
    // Test division by positive constant
    let lower = ArrayD::from_elem(IxDyn(&[3]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 6.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let div = DivConstantLayer::scalar(2.0);
    let output = div.propagate_ibp(&input).unwrap();

    // [2, 6] / 2 = [1, 3]
    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - 1.0).abs() < 1e-6,
            "2/2 should be 1, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 3.0).abs() < 1e-6,
            "6/2 should be 3, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_ibp_negative_divisor() {
    // Test division by negative constant (bounds swap)
    let lower = ArrayD::from_elem(IxDyn(&[2]), 2.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 6.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let div = DivConstantLayer::scalar(-2.0);
    let output = div.propagate_ibp(&input).unwrap();

    // [2, 6] / -2 = [-3, -1]
    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - (-3.0)).abs() < 1e-6,
            "6/-2 should be -3, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - (-1.0)).abs() < 1e-6,
            "2/-2 should be -1, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_ibp_mixed_sign_divisor() {
    // Test element-wise sign handling for per-element divisors
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0f32, 2.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![6.0f32, 12.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let constant = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0f32, -4.0]).unwrap();
    let div = DivConstantLayer::new(constant);
    let output = div.propagate_ibp(&input).unwrap();

    // [2, 6] / 2 = [1, 3]
    assert!(
        (output.lower()[[0]] - 1.0).abs() < 1e-6,
        "2/2 should be 1, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 3.0).abs() < 1e-6,
        "6/2 should be 3, got {}",
        output.upper()[[0]]
    );

    // [2, 12] / -4 = [-3, -0.5]
    assert!(
        (output.lower()[[1]] - (-3.0)).abs() < 1e-6,
        "12/-4 should be -3, got {}",
        output.lower()[[1]]
    );
    assert!(
        (output.upper()[[1]] - (-0.5)).abs() < 1e-6,
        "2/-4 should be -0.5, got {}",
        output.upper()[[1]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_ibp_near_zero_divisor() {
    let err =
        DivConstantLayer::try_scalar(1.0e-12).expect_err("near-zero divisor should be rejected");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_linear_near_zero_divisor() {
    let err =
        DivConstantLayer::try_scalar(1.0e-12).expect_err("near-zero divisor should be rejected");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_linear_batched_near_zero_divisor() {
    let err =
        DivConstantLayer::try_scalar(1.0e-12).expect_err("near-zero divisor should be rejected");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_ibp_zero_divisor() {
    let err = DivConstantLayer::try_scalar(0.0).expect_err("zero divisor should be rejected");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_ibp_negative_near_zero_divisor() {
    let err = DivConstantLayer::try_scalar(-1.0e-12)
        .expect_err("negative near-zero divisor should be rejected");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_div_constant_ibp_small_nonzero_divisor() {
    // Small but non-zero divisor should produce finite bounds
    let lower = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let div = DivConstantLayer::scalar(1.0e-6);
    let output = div.propagate_ibp(&input).unwrap();

    assert!(
        (output.lower()[[0]] - 1.0e6).abs() < 1.0,
        "1/1e-6 should be 1e6, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 2.0e6).abs() < 1.0,
        "2/1e-6 should be 2e6, got {}",
        output.upper()[[0]]
    );
}
