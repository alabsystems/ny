// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::array;
use ny_core::NyError;

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

// ========== exp_interval_bounds tests ==========

#[test]
fn exp_interval_bounds_monotonicity() {
    // exp is monotonically increasing, so exp(lower) <= exp(x) <= exp(upper)
    let (el, eu) = exp_interval_bounds(0.0, 1.0).expect("valid interval should succeed");
    assert!((el - 1.0).abs() < 1e-6, "exp(0) should be 1.0");
    let e1 = 1.0_f32.exp();
    assert!((eu - e1).abs() < 1e-5, "exp(1) should be ~2.718");
    assert!(el < eu, "exp(lower) < exp(upper) when lower < upper");
}

#[test]
fn exp_interval_bounds_negative_inputs() {
    let (el, eu) = exp_interval_bounds(-5.0, -1.0).expect("valid interval should succeed");
    let expected_l = (-5.0_f32).exp();
    let expected_u = (-1.0_f32).exp();
    assert!(
        (el - expected_l).abs() < 1e-6,
        "exp lower {el} should be close to expected {expected_l}"
    );
    assert!(
        (eu - expected_u).abs() < 1e-5,
        "exp upper {eu} should be close to expected {expected_u}"
    );
    assert!(el < eu, "exp lower {el} should be < exp upper {eu}");
}

#[test]
fn exp_interval_bounds_point_interval() {
    // When lower == upper, both bounds should equal exp(point)
    let (el, eu) = exp_interval_bounds(2.0, 2.0).expect("valid interval should succeed");
    let expected = 2.0_f32.exp();
    assert!(
        (el - expected).abs() < 1e-5,
        "exp lower {el} should equal exp(2.0) = {expected}"
    );
    assert!(
        (eu - expected).abs() < 1e-5,
        "exp upper {eu} should equal exp(2.0) = {expected}"
    );
}

#[test]
fn exp_interval_bounds_large_positive() {
    // Large positive: should produce large but finite value
    let (el, eu) = exp_interval_bounds(80.0, 88.0).expect("valid interval should succeed");
    assert!(el.is_finite(), "exp(80) should be finite");
    assert!(eu.is_finite(), "exp(88) should be finite");
    assert!(el < eu, "exp lower {el} should be < exp upper {eu}");
}

#[test]
fn exp_interval_bounds_large_negative() {
    // Large negative: should produce very small but non-negative value
    let (el, eu) = exp_interval_bounds(-100.0, -50.0).expect("valid interval should succeed");
    assert!(el >= 0.0, "exp of anything should be non-negative");
    assert!(eu >= 0.0, "exp upper {eu} should be non-negative");
    assert!(el < eu, "exp lower {el} should be < exp upper {eu}");
}

#[test]
fn exp_interval_bounds_rejects_inverted_interval() {
    let err = exp_interval_bounds(1.0, -1.0).expect_err("inverted interval must return error");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got {err:?}"
    );
}

// ========== softmax_ibp_element_bounds tests ==========

#[test]
fn softmax_ibp_element_bounds_uniform_two_elements() {
    // Two identical elements: softmax = [0.5, 0.5]
    // exp(0)=1 for all shifted values when uniform
    let exp_l = 1.0_f32;
    let exp_u = 1.0_f32;
    let sum_l = 2.0_f32;
    let sum_u = 2.0_f32;
    let (lower, upper) = softmax_ibp_element_bounds(exp_l, exp_u, sum_l, sum_u);
    // For uniform input, bounds should contain 0.5
    assert!(lower <= 0.5 + 1e-5, "lower {} should be <= 0.5", lower);
    assert!(upper >= 0.5 - 1e-5, "upper {} should be >= 0.5", upper);
    // Bounds in [0, 1]
    assert!(lower >= 0.0, "softmax lower {lower} should be non-negative");
    assert!(upper <= 1.0, "softmax upper {upper} should be <= 1.0");
}

#[test]
fn softmax_ibp_element_bounds_dominant_element() {
    // One element much larger: its softmax should be near 1
    let exp_l_dominant = 100.0_f32;
    let exp_u_dominant = 200.0_f32;
    let exp_l_other = 0.01_f32;
    let exp_u_other = 0.02_f32;
    let sum_l = exp_l_dominant + exp_l_other;
    let sum_u = exp_u_dominant + exp_u_other;

    let (lower, upper) = softmax_ibp_element_bounds(exp_l_dominant, exp_u_dominant, sum_l, sum_u);
    assert!(
        lower > 0.9,
        "dominant element lower {} should be > 0.9",
        lower
    );
    assert!(upper <= 1.0, "upper {} should be <= 1.0", upper);
}

#[test]
fn softmax_ibp_element_bounds_clamps_to_unit() {
    // Even with extreme values, output should be in [0, 1]
    let (lower, upper) = softmax_ibp_element_bounds(0.0, 1e30, 1.0, 1e30);
    assert!(lower >= 0.0, "softmax lower {lower} should be non-negative");
    assert!(upper <= 1.0, "softmax upper {upper} should be <= 1.0");
}

#[test]
fn softmax_ibp_element_bounds_non_finite_denominator() {
    // Zero denominator should produce safe fallback
    let (lower, upper) = softmax_ibp_element_bounds(1.0, 1.0, 0.0, 0.0);
    // When denom_for_lower = 0 + 0 - 1 + 1 + eps = eps > 0 (actually ok here)
    // But when sum_exp_lower - exp_lower_i = 0 - 1 = -1, denom = -1 + 1 + eps = eps
    // This is still positive due to epsilon. Let's test truly bad input:
    assert!(lower >= 0.0, "softmax lower {lower} should be non-negative");
    assert!(upper <= 1.0, "softmax upper {upper} should be <= 1.0");
}

#[test]
fn softmax_ibp_element_bounds_inf_inputs() {
    let (lower, upper) =
        softmax_ibp_element_bounds(f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY);
    // Non-finite denominator: should fall back to [0, 1]
    assert!(lower >= 0.0, "softmax lower {lower} should be non-negative");
    assert!(upper <= 1.0, "softmax upper {upper} should be <= 1.0");
}

#[test]
fn softmax_ibp_element_bounds_nan_raw_bounds_widen_to_unit_interval() {
    let (lower, upper) = softmax_ibp_element_bounds(f32::NAN, f32::NAN, 1.0, 1.0);
    assert_eq!(lower, 0.0, "NaN lower must widen to 0");
    assert_eq!(upper, 1.0, "NaN upper must widen to 1");
}

// ========== logsumexp_slice tests ==========

#[test]
fn logsumexp_slice_single_element() {
    let result = logsumexp_slice(&[3.0]);
    // logsumexp([x]) = x + ln(exp(0)) = x + 0 = x
    assert!(
        (result - 3.0).abs() < 1e-6,
        "logsumexp([3]) = {}, expected 3",
        result
    );
}

#[test]
fn logsumexp_slice_two_equal_elements() {
    let result = logsumexp_slice(&[1.0, 1.0]);
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
fn logsumexp_slice_different_elements() {
    let result = logsumexp_slice(&[1.0, 2.0, 3.0]);
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
fn logsumexp_slice_empty() {
    let result = logsumexp_slice(&[]);
    assert_eq!(result, f32::NEG_INFINITY);
}

#[test]
fn logsumexp_slice_large_values() {
    // Numerical stability: should not overflow for large values
    let result = logsumexp_slice(&[1000.0, 1001.0, 999.0]);
    assert!(
        result.is_finite(),
        "logsumexp should be finite for large inputs"
    );
    // Should be close to max(inputs) + small correction
    assert!(result > 1000.0 && result < 1002.0, "got {}", result);
}

#[test]
fn logsumexp_slice_large_negative() {
    let result = logsumexp_slice(&[-1000.0, -999.0, -998.0]);
    assert!(
        result.is_finite(),
        "logsumexp should be finite for large negative inputs"
    );
    assert!(result > -999.0 && result < -997.0, "got {}", result);
}

#[test]
fn logsumexp_slice_non_finite_input() {
    let result = logsumexp_slice(&[f32::NEG_INFINITY, 1.0, 2.0]);
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
    let result2 = logsumexp_slice(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
    assert_eq!(result2, f32::NEG_INFINITY);
}

// ========== logsoftmax_ibp_bounds tests ==========

#[test]
fn logsoftmax_ibp_bounds_basic() {
    // logsoftmax_lower = lower_i - lse_upper
    // logsoftmax_upper = upper_i - lse_lower
    let (lb, ub) = logsoftmax_ibp_bounds(1.0, 2.0, 3.0, 5.0);
    assert!(
        (lb - (1.0 - 5.0)).abs() < 1e-6,
        "lower = {}, expected -4.0",
        lb
    );
    assert!(
        (ub - (2.0 - 3.0)).abs() < 1e-6,
        "upper = {}, expected -1.0",
        ub
    );
}

#[test]
fn logsoftmax_ibp_bounds_soundness_three_elements() {
    // For x in [lower, upper], verify logsoftmax(x)_i is within bounds
    let lower = [0.0, 1.0, 2.0];
    let upper = [1.0, 2.0, 3.0];
    let lse_lower = logsumexp_slice(&lower);
    let lse_upper = logsumexp_slice(&upper);

    for i in 0..3 {
        let (lb, ub) = logsoftmax_ibp_bounds(lower[i], upper[i], lse_lower, lse_upper);
        // Check that actual logsoftmax at corners is within bounds
        for mask in 0..8u32 {
            let x: Vec<f32> = (0..3)
                .map(|j| {
                    if (mask >> j) & 1 == 0 {
                        lower[j]
                    } else {
                        upper[j]
                    }
                })
                .collect();
            let lse = logsumexp_slice(&x);
            let actual = x[i] - lse;
            assert!(
                lb <= actual + 1e-5,
                "logsoftmax_ibp lower[{}] = {} > actual {} at x={:?}",
                i,
                lb,
                actual,
                x
            );
            assert!(
                ub >= actual - 1e-5,
                "logsoftmax_ibp upper[{}] = {} < actual {} at x={:?}",
                i,
                ub,
                actual,
                x
            );
        }
    }
}

#[test]
fn logsoftmax_ibp_bounds_lower_always_negative() {
    // logsoftmax lower bound = lower_i - lse_upper.
    // Since lse_upper >= max(upper) >= upper_i >= lower_i, the lower bound
    // is always <= 0. (The upper bound upper_i - lse_lower can be positive
    // because lse_lower < upper_i is possible when intervals are wide.)
    let lower = [-1.0, 0.0, 1.0];
    let upper = [0.0, 1.0, 2.0];
    let lse_lower = logsumexp_slice(&lower);
    let lse_upper = logsumexp_slice(&upper);

    for i in 0..3 {
        let (lb, _) = logsoftmax_ibp_bounds(lower[i], upper[i], lse_lower, lse_upper);
        assert!(
            lb <= 1e-5,
            "logsoftmax lower[{}] = {} should be <= 0",
            i,
            lb
        );
    }
}

// ========== logsumexp_1d tests ==========

#[test]
fn logsumexp_1d_matches_slice_version() {
    let vals = array![1.0, 2.0, 3.0, -1.0];
    let result_1d = logsumexp_1d(&vals);
    let result_slice = logsumexp_slice(&[1.0, 2.0, 3.0, -1.0]);
    assert!(
        (result_1d - result_slice).abs() < 1e-6,
        "1d={}, slice={}",
        result_1d,
        result_slice
    );
}

#[test]
fn logsumexp_1d_single() {
    let vals = array![5.0];
    let result = logsumexp_1d(&vals);
    assert!(
        (result - 5.0).abs() < 1e-6,
        "logsumexp_1d([5.0]) = {result}, expected 5.0"
    );
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
