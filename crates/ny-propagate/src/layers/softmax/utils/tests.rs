// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::array;

// ========== sanitize_softmax_unit_bounds tests ==========

#[ntest::timeout(10000)]
#[test]
fn test_sanitize_softmax_unit_bounds_handles_nan_inf() {
    let (lower, upper) = sanitize_softmax_unit_bounds(f32::NAN, f32::INFINITY);
    assert_eq!(lower, 0.0);
    assert_eq!(upper, 1.0);

    let (lower, upper) = sanitize_softmax_unit_bounds(f32::NEG_INFINITY, 0.5);
    assert_eq!(lower, 0.0);
    let expected_upper = (0.5 + SOFTMAX_SANITIZE_MARGIN).min(1.0);
    assert!(
        (upper - expected_upper).abs() < 1e-6,
        "upper {upper} should be close to expected {expected_upper}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sanitize_softmax_unit_bounds_clamps_out_of_range() {
    let (lower, upper) = sanitize_softmax_unit_bounds(-0.2, 1.2);
    assert_eq!(lower, 0.0);
    assert_eq!(upper, 1.0);

    let (lower, upper) = sanitize_softmax_unit_bounds(0.2, 0.8);
    let expected_lower = (0.2 - SOFTMAX_SANITIZE_MARGIN).max(0.0);
    let expected_upper = (0.8 + SOFTMAX_SANITIZE_MARGIN).min(1.0);
    assert!(
        (lower - expected_lower).abs() < 1e-6,
        "lower {lower} should be close to expected {expected_lower}"
    );
    assert!(
        (upper - expected_upper).abs() < 1e-6,
        "upper {upper} should be close to expected {expected_upper}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_sanitize_softmax_unit_bounds_inverted_bounds_falls_back() {
    let (lower, upper) = sanitize_softmax_unit_bounds(0.9, 0.1);
    assert_eq!(lower, 0.0);
    assert_eq!(upper, 1.0);
}

// ========== logsumexp_1d tests ==========

#[test]
fn logsumexp_1d_single_element() {
    let result = logsumexp_1d(&array![3.0]);
    // logsumexp([x]) = x + ln(exp(0)) = x + 0 = x
    assert!(
        (result - 3.0).abs() < 1e-6,
        "logsumexp([3]) = {}, expected 3",
        result
    );
}

#[test]
fn logsumexp_1d_two_equal_elements() {
    let result = logsumexp_1d(&array![1.0, 1.0]);
    // logsumexp([1, 1]) = 1 + ln(2) ≈ 1.693
    let expected = 1.0 + 2.0_f32.ln();
    assert!(
        (result - expected).abs() < 1e-5,
        "logsumexp([1,1]) = {}, expected {}",
        result,
        expected
    );
}

#[test]
fn logsumexp_1d_different_elements() {
    let result = logsumexp_1d(&array![1.0, 2.0, 3.0]);
    // max=3, logsumexp = 3 + ln(exp(-2) + exp(-1) + 1)
    let expected = 3.0 + ((-2.0_f32).exp() + (-1.0_f32).exp() + 1.0).ln();
    assert!(
        (result - expected).abs() < 1e-5,
        "got {}, expected {}",
        result,
        expected
    );
}

#[test]
fn logsumexp_1d_empty() {
    let result = logsumexp_1d(&Array1::<f32>::zeros(0));
    assert_eq!(result, f32::NEG_INFINITY);
}

#[test]
fn logsumexp_1d_large_values() {
    // Numerical stability: should not overflow for large values
    let result = logsumexp_1d(&array![1000.0, 1001.0, 999.0]);
    assert!(
        result.is_finite(),
        "logsumexp should be finite for large inputs"
    );
    // Should be close to max(inputs) + small correction
    assert!(result > 1000.0 && result < 1002.0, "got {}", result);
}

#[test]
fn logsumexp_1d_large_negative() {
    let result = logsumexp_1d(&array![-1000.0, -999.0, -998.0]);
    assert!(
        result.is_finite(),
        "logsumexp should be finite for large negative inputs"
    );
    assert!(result > -999.0 && result < -997.0, "got {}", result);
}

#[test]
fn logsumexp_1d_non_finite_input() {
    let result = logsumexp_1d(&array![f32::NEG_INFINITY, 1.0, 2.0]);
    // NEG_INFINITY doesn't affect max if others exist; exp(NEG_INF) = 0
    // max=2, logsumexp = 2 + ln(exp(-inf) + exp(-1) + 1) = 2 + ln(0 + exp(-1) + 1)
    let expected = 2.0 + ((-1.0_f32).exp() + 1.0).ln();
    assert!(
        (result - expected).abs() < 1e-5,
        "got {}, expected {}",
        result,
        expected
    );

    // All NEG_INFINITY → max is NEG_INFINITY → returns NEG_INFINITY
    let result2 = logsumexp_1d(&array![f32::NEG_INFINITY, f32::NEG_INFINITY]);
    assert_eq!(result2, f32::NEG_INFINITY);
}

// ========== softmax_1d tests ==========

#[test]
fn softmax_1d_sums_to_one() {
    let vals = array![1.0, 2.0, 3.0];
    let s = softmax_1d(&vals);
    assert!((s.sum() - 1.0).abs() < 1e-6, "softmax sum = {}", s.sum());
}

#[test]
fn softmax_1d_all_positive() {
    let vals = array![-5.0, 0.0, 5.0];
    let s = softmax_1d(&vals);
    for (i, &si) in s.iter().enumerate() {
        assert!(si > 0.0, "softmax_1d[{}] = {} should be > 0", i, si);
        assert!(si <= 1.0, "softmax_1d[{}] = {} should be <= 1", i, si);
    }
}

#[test]
fn softmax_1d_uniform_input() {
    let vals = array![0.0, 0.0, 0.0, 0.0];
    let s = softmax_1d(&vals);
    for (i, &si) in s.iter().enumerate() {
        assert!(
            (si - 0.25).abs() < 1e-6,
            "softmax_1d[{}] = {}, expected 0.25",
            i,
            si
        );
    }
}

#[test]
fn softmax_1d_large_input_stability() {
    let vals = array![500.0, 501.0, 499.0];
    let s = softmax_1d(&vals);
    assert!((s.sum() - 1.0).abs() < 1e-5, "sum = {}", s.sum());
    for &si in s.iter() {
        assert!(!si.is_nan(), "softmax_1d should not produce NaN");
    }
}

#[test]
fn softmax_1d_single_element() {
    let vals = array![42.0];
    let s = softmax_1d(&vals);
    assert!(
        (s[0] - 1.0).abs() < 1e-6,
        "single element softmax should be 1.0"
    );
}
