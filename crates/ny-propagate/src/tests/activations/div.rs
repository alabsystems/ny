// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for binary division layer (DivLayer).

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Binary Division (DivLayer) tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_positive_divisor() {
    // Test division A / B where B is strictly positive
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3]), 2.0f32),
        ArrayD::from_elem(IxDyn(&[3]), 6.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[3]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[3]), 2.0f32),
    )
    .unwrap();

    let div = DivLayer;
    let output = div.propagate_ibp_binary(&input_a, &input_b).unwrap();

    // A ∈ [2, 6], B ∈ [1, 2]
    // C_lower = A_l / B_u = 2/2 = 1
    // C_upper = A_u / B_l = 6/1 = 6
    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - 1.0).abs() < 1e-6,
            "C_lower should be 1, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 6.0).abs() < 1e-6,
            "C_upper should be 6, got {}",
            output.upper()[[i]]
        );
    }
}

/// Regression test for issue #1606: DivLayer should reject non-positive divisor.
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_rejects_zero_divisor_1606() {
    // Test that zero divisor is rejected
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 2.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 0.0f32),
    )
    .unwrap();

    let div = DivLayer;
    let err = div
        .propagate_ibp_binary(&input_a, &input_b)
        .expect_err("Zero divisor should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("0 < lower <= upper")
            || msg.contains("strictly positive")
            || msg.contains("positive divisor"),
        "error should mention positive requirement: {msg}"
    );
}

/// Regression test for issue #1606: DivLayer should reject negative divisor.
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_rejects_negative_divisor_1606() {
    // Test that negative divisor is rejected
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 2.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -2.0f32),
        ArrayD::from_elem(IxDyn(&[2]), -1.0f32),
    )
    .unwrap();

    let div = DivLayer;
    let err = div
        .propagate_ibp_binary(&input_a, &input_b)
        .expect_err("Negative divisor should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("0 < lower <= upper")
            || msg.contains("strictly positive")
            || msg.contains("positive divisor"),
        "error should mention positive requirement: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_broadcasting() {
    // Test with broadcasting: A has shape [2, 3], B has shape [3]
    let input_a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).unwrap(),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 4.0, 6.0]).unwrap(),
    )
    .unwrap();

    let div = DivLayer;
    let output = div.propagate_ibp_binary(&input_a, &input_b).unwrap();

    // Check shape is correct after broadcast
    assert_eq!(output.shape(), &[2, 3]);

    // Verify bounds are sound (lower <= actual <= upper)
    // For A[0,0]/B[0]: A ∈ [1, 2], B ∈ [1, 2]
    // C_lower = 1/2 = 0.5, C_upper = 2/1 = 2
    assert!(
        (output.lower()[[0, 0]] - 0.5).abs() < 1e-6,
        "C_lower[0,0] should be 0.5, got {}",
        output.lower()[[0, 0]]
    );
    assert!(
        (output.upper()[[0, 0]] - 2.0).abs() < 1e-6,
        "C_upper[0,0] should be 2, got {}",
        output.upper()[[0, 0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_small_positive_divisor() {
    // Test with small but strictly positive divisor
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1e-6f32),
        ArrayD::from_elem(IxDyn(&[1]), 1e-5f32),
    )
    .unwrap();

    let div = DivLayer;
    let output = div.propagate_ibp_binary(&input_a, &input_b).unwrap();

    // A ∈ [1, 2], B ∈ [1e-6, 1e-5]
    // C_lower = 1/1e-5 = 1e5
    // C_upper = 2/1e-6 = 2e6
    assert!(
        (output.lower()[[0]] - 1e5).abs() < 1e3,
        "C_lower should be 1e5, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 2e6).abs() < 1e4,
        "C_upper should be 2e6, got {}",
        output.upper()[[0]]
    );
}

/// Regression test for issue #1643: negative numerator with positive divisor.
/// f(b) = a/b is increasing in b when a < 0, so the old formula that assumed
/// decreasing monotonicity produced bounds that were too tight.
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_negative_numerator_1643() {
    // A ∈ [-6, -2], B ∈ [1, 2]
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -6.0f32),
        ArrayD::from_elem(IxDyn(&[1]), -2.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .unwrap();

    let div = DivLayer;
    let output = div.propagate_ibp_binary(&input_a, &input_b).unwrap();

    // A_u <= 0: C_lower = A_l/B_l = -6/1 = -6, C_upper = A_u/B_u = -2/2 = -1
    assert!(
        (output.lower()[[0]] - (-6.0)).abs() < 1e-6,
        "C_lower should be -6, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - (-1.0)).abs() < 1e-6,
        "C_upper should be -1, got {}",
        output.upper()[[0]]
    );
}

/// Regression test for issue #1643: mixed-sign numerator with positive divisor.
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_mixed_sign_numerator_1643() {
    // A ∈ [-3, 5], B ∈ [1, 2]
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -3.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 5.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .unwrap();

    let div = DivLayer;
    let output = div.propagate_ibp_binary(&input_a, &input_b).unwrap();

    // Mixed: C_lower = A_l/B_l = -3/1 = -3, C_upper = A_u/B_l = 5/1 = 5
    assert!(
        (output.lower()[[0]] - (-3.0)).abs() < 1e-6,
        "C_lower should be -3, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 5.0).abs() < 1e-6,
        "C_upper should be 5, got {}",
        output.upper()[[0]]
    );
}

/// Regression test for issue #1643: zero-crossing numerator [-1, 1].
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_zero_crossing_numerator_1643() {
    // A ∈ [-1, 1], B ∈ [2, 4]
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 4.0f32),
    )
    .unwrap();

    let div = DivLayer;
    let output = div.propagate_ibp_binary(&input_a, &input_b).unwrap();

    // Mixed: C_lower = A_l/B_l = -1/2 = -0.5, C_upper = A_u/B_l = 1/2 = 0.5
    assert!(
        (output.lower()[[0]] - (-0.5)).abs() < 1e-6,
        "C_lower should be -0.5, got {}",
        output.lower()[[0]]
    );
    assert!(
        (output.upper()[[0]] - 0.5).abs() < 1e-6,
        "C_upper should be 0.5, got {}",
        output.upper()[[0]]
    );
}

/// Regression test for #1689: divisor bounds spanning zero should be rejected.
/// B ∈ [-1, 2] — the interval contains zero, so division is undefined.
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_rejects_divisor_spanning_zero_1689() {
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 3.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[2]), 2.0f32),
    )
    .unwrap();

    let div = DivLayer;
    let err = div
        .propagate_ibp_binary(&input_a, &input_b)
        .expect_err("Divisor spanning zero should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("0 < lower <= upper")
            || msg.contains("strictly positive")
            || msg.contains("positive divisor"),
        "error should mention positive requirement: {msg}"
    );
}

/// Regression test for #1689: divisor with b_lower = 0 exactly should be rejected.
/// B ∈ [0, 5] — lower bound touches zero, not strictly positive.
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_rejects_divisor_touching_zero_1689() {
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 0.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 5.0f32),
    )
    .unwrap();

    let div = DivLayer;
    let err = div
        .propagate_ibp_binary(&input_a, &input_b)
        .expect_err("Divisor touching zero should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("0 < lower <= upper") || msg.contains("strictly positive"),
        "error should mention strictly positive: {msg}"
    );
}

/// Regression test for #1689: relying only on b_lower > 0 is unsound when
/// divisor intervals are malformed (lower > upper).
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_rejects_inverted_positive_divisor_bounds_1689() {
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 0.5f32),
    )
    .unwrap();

    let div = DivLayer;
    let err = div
        .propagate_ibp_binary(&input_a, &input_b)
        .expect_err("Malformed divisor bounds should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("0 < lower <= upper"),
        "error should mention divisor invariant: {msg}"
    );
}

/// Regression test for #1689: non-finite divisor bounds must be rejected
/// before division to avoid NaN propagation.
#[ntest::timeout(10000)]
#[test]
fn test_div_layer_ibp_rejects_non_finite_divisor_bounds_1689() {
    let input_a = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), 2.0f32),
    )
    .unwrap();
    let input_b = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), 1.0f32),
        ArrayD::from_elem(IxDyn(&[1]), f32::NAN),
    )
    .unwrap();

    let div = DivLayer;
    let err = div
        .propagate_ibp_binary(&input_a, &input_b)
        .expect_err("Non-finite divisor bounds should be rejected");
    let msg = match err {
        NyError::InvalidSpec(msg) => msg,
        other => panic!("unexpected error type: {other:?}"),
    };
    assert!(
        msg.contains("finite divisor bounds"),
        "error should mention finite divisor requirement: {msg}"
    );
}
