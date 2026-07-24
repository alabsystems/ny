// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Inf/NaN/domain-boundary edge case tests for activation IBP propagation.
//!
//! #2435: Log missing Inf/NaN edge cases, Sigmoid/Tanh/Cos/Softplus missing
//! Inf/NaN edge case coverage.
//!
//! These are deterministic unit tests (not property tests) targeting specific
//! edge cases, following the `infinite_bounds.rs` pattern.

use crate::layers::common::BoundPropagation;
use crate::layers::{CosLayer, LogLayer, SigmoidLayer, SinLayer, SoftplusLayer, TanhLayer};
use ndarray::arr1;
use ntest::timeout;
use ny_tensor::BoundedTensor;

// =========================================================================
// LOG EDGE CASES (#2435)
// Log domain: (0, +inf). log(0) = -inf, log(-x) = NaN.
// =========================================================================

/// Log IBP rejects zero in lower bound (log(0) = -inf is outside valid domain).
#[timeout(10000)]
#[test]
fn log_ibp_rejects_zero_lower() {
    // Use new_unchecked to guarantee we exercise LogLayer, not BoundedTensor validation.
    let input =
        BoundedTensor::new_unchecked(arr1(&[0.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let result = LogLayer.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Log IBP must reject input with lower bound = 0 (log(0) = -inf)"
    );
}

/// Log IBP rejects negative lower bound (log(-x) is undefined).
#[timeout(10000)]
#[test]
fn log_ibp_rejects_negative_lower() {
    // Use new_unchecked to bypass BoundedTensor validation
    let input =
        BoundedTensor::new_unchecked(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let result = LogLayer.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Log IBP must reject input with negative lower bound"
    );
}

/// Log IBP rejects NaN in lower bound.
#[timeout(10000)]
#[test]
fn log_ibp_rejects_nan_lower() {
    let input = BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())
        .unwrap();

    let result = LogLayer.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Log IBP must reject input with NaN lower bound"
    );
}

/// Log IBP rejects NaN in upper bound.
#[timeout(10000)]
#[test]
fn log_ibp_rejects_nan_upper() {
    let input = BoundedTensor::new_unchecked(arr1(&[0.5]).into_dyn(), arr1(&[f32::NAN]).into_dyn())
        .unwrap();

    let result = LogLayer.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "Log IBP must reject input with NaN upper bound"
    );
}

/// Log IBP handles near-zero positive input correctly.
/// log(1e-10) ≈ -23.03, log(1) = 0.
#[timeout(10000)]
#[test]
fn log_ibp_near_zero_positive() {
    let input = BoundedTensor::new(arr1(&[1e-10]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let output = LogLayer.propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Log IBP lower is NaN for [1e-10, 1]");
    assert!(!upper.is_nan(), "Log IBP upper is NaN for [1e-10, 1]");
    assert!(lower <= upper, "Log IBP bounds inverted: {lower} > {upper}");
    // ln(1e-10) ≈ -23.03
    assert!(
        lower < -22.0,
        "Log IBP lower for [1e-10, 1] should be < -22, got {lower}"
    );
    // ln(1) = 0
    assert!(
        upper.abs() < 0.01,
        "Log IBP upper for [1e-10, 1] should be ≈ 0, got {upper}"
    );
}

/// Log IBP handles large positive input correctly.
/// log(1e30) ≈ 69.08.
#[timeout(10000)]
#[test]
fn log_ibp_large_positive() {
    let input = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[1e30]).into_dyn()).unwrap();

    let output = LogLayer.propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Log IBP lower is NaN for [1, 1e30]");
    assert!(!upper.is_nan(), "Log IBP upper is NaN for [1, 1e30]");
    assert!(lower <= upper, "Log IBP bounds inverted: {lower} > {upper}");
    // ln(1) = 0
    assert!(
        lower.abs() < 0.01,
        "Log IBP lower for [1, 1e30] should be ≈ 0, got {lower}"
    );
    // ln(1e30) ≈ 69.08
    assert!(
        upper > 69.0,
        "Log IBP upper for [1, 1e30] should be > 69, got {upper}"
    );
}

// =========================================================================
// SIGMOID EDGE CASES (#2435)
// Sigmoid range: (0, 1). sigmoid(-inf) → 0, sigmoid(+inf) → 1.
// =========================================================================

/// Sigmoid IBP with -inf lower bound: sigmoid(-inf) → 0.
#[timeout(10000)]
#[test]
fn sigmoid_ibp_neg_infinity_lower() {
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
    )
    .unwrap();

    let output = SigmoidLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Sigmoid IBP lower is NaN for [-inf, 0]");
    assert!(!upper.is_nan(), "Sigmoid IBP upper is NaN for [-inf, 0]");
    assert!(
        lower <= upper,
        "Sigmoid IBP bounds inverted: {lower} > {upper}"
    );
    // sigmoid(-inf) = 0, sigmoid(0) = 0.5
    assert!(
        (0.0..=0.01).contains(&lower),
        "Sigmoid IBP lower for [-inf, 0] should be ≈ 0, got {lower}"
    );
    assert!(
        (upper - 0.5).abs() < 0.01,
        "Sigmoid IBP upper for [-inf, 0] should be ≈ 0.5, got {upper}"
    );
}

/// Sigmoid IBP with +inf upper bound: sigmoid(+inf) → 1.
#[timeout(10000)]
#[test]
fn sigmoid_ibp_pos_infinity_upper() {
    let input =
        BoundedTensor::new_unchecked(arr1(&[0.0]).into_dyn(), arr1(&[f32::INFINITY]).into_dyn())
            .unwrap();

    let output = SigmoidLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Sigmoid IBP lower is NaN for [0, inf]");
    assert!(!upper.is_nan(), "Sigmoid IBP upper is NaN for [0, inf]");
    assert!(
        lower <= upper,
        "Sigmoid IBP bounds inverted: {lower} > {upper}"
    );
    // sigmoid(0) = 0.5, sigmoid(+inf) = 1
    assert!(
        (lower - 0.5).abs() < 0.01,
        "Sigmoid IBP lower for [0, inf] should be ≈ 0.5, got {lower}"
    );
    assert!(
        (0.99..=1.0).contains(&upper),
        "Sigmoid IBP upper for [0, inf] should be ≈ 1.0, got {upper}"
    );
}

/// Sigmoid IBP with full infinite range produces no NaN, bounds in [0, 1].
#[timeout(10000)]
#[test]
fn sigmoid_ibp_full_infinite_no_nan() {
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .unwrap();

    let output = SigmoidLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Sigmoid IBP lower is NaN for [-inf, inf]");
    assert!(!upper.is_nan(), "Sigmoid IBP upper is NaN for [-inf, inf]");
    assert!(
        lower <= upper,
        "Sigmoid IBP bounds inverted: {lower} > {upper}"
    );
    assert!(
        lower >= 0.0 && upper <= 1.0,
        "Sigmoid IBP bounds for [-inf, inf] should be in [0, 1], got [{lower}, {upper}]"
    );
}

// =========================================================================
// TANH EDGE CASES (#2435)
// Tanh range: (-1, 1). tanh(-inf) → -1, tanh(+inf) → 1.
// =========================================================================

/// Tanh IBP with -inf lower bound: tanh(-inf) → -1.
#[timeout(10000)]
#[test]
fn tanh_ibp_neg_infinity_lower() {
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
    )
    .unwrap();

    let output = TanhLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Tanh IBP lower is NaN for [-inf, 0]");
    assert!(!upper.is_nan(), "Tanh IBP upper is NaN for [-inf, 0]");
    assert!(
        lower <= upper,
        "Tanh IBP bounds inverted: {lower} > {upper}"
    );
    // tanh(-inf) = -1, tanh(0) = 0
    assert!(
        (-1.0..=-0.99).contains(&lower),
        "Tanh IBP lower for [-inf, 0] should be ≈ -1, got {lower}"
    );
    assert!(
        upper.abs() < 0.01,
        "Tanh IBP upper for [-inf, 0] should be ≈ 0, got {upper}"
    );
}

/// Tanh IBP with +inf upper bound: tanh(+inf) → 1.
#[timeout(10000)]
#[test]
fn tanh_ibp_pos_infinity_upper() {
    let input =
        BoundedTensor::new_unchecked(arr1(&[0.0]).into_dyn(), arr1(&[f32::INFINITY]).into_dyn())
            .unwrap();

    let output = TanhLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Tanh IBP lower is NaN for [0, inf]");
    assert!(!upper.is_nan(), "Tanh IBP upper is NaN for [0, inf]");
    assert!(
        lower <= upper,
        "Tanh IBP bounds inverted: {lower} > {upper}"
    );
    // tanh(0) = 0, tanh(+inf) = 1
    assert!(
        lower.abs() < 0.01,
        "Tanh IBP lower for [0, inf] should be ≈ 0, got {lower}"
    );
    assert!(
        (0.99..=1.0).contains(&upper),
        "Tanh IBP upper for [0, inf] should be ≈ 1, got {upper}"
    );
}

/// Tanh IBP with full infinite range produces no NaN, bounds in [-1, 1].
#[timeout(10000)]
#[test]
fn tanh_ibp_full_infinite_no_nan() {
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .unwrap();

    let output = TanhLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Tanh IBP lower is NaN for [-inf, inf]");
    assert!(!upper.is_nan(), "Tanh IBP upper is NaN for [-inf, inf]");
    assert!(
        lower <= upper,
        "Tanh IBP bounds inverted: {lower} > {upper}"
    );
    assert!(
        lower >= -1.0 && upper <= 1.0,
        "Tanh IBP bounds for [-inf, inf] should be in [-1, 1], got [{lower}, {upper}]"
    );
}

// =========================================================================
// COS EDGE CASES (#2435)
// Cos range: [-1, 1]. cos(inf) = NaN (IEEE 754). cos has non-finite guard.
// =========================================================================

/// Cos IBP with infinite input falls back to [-1, 1].
#[timeout(10000)]
#[test]
fn cos_ibp_infinite_falls_back_to_full_range() {
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .unwrap();

    let output = CosLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Cos IBP lower is NaN for [-inf, inf]");
    assert!(!upper.is_nan(), "Cos IBP upper is NaN for [-inf, inf]");
    assert_eq!(
        lower, -1.0,
        "Cos IBP lower for [-inf, inf] should be -1.0, got {lower}"
    );
    assert_eq!(
        upper, 1.0,
        "Cos IBP upper for [-inf, inf] should be 1.0, got {upper}"
    );
}

/// Cos IBP with NaN input (via unchecked) falls back to [-1, 1].
#[timeout(10000)]
#[test]
fn cos_ibp_nan_falls_back_to_full_range() {
    let input = BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())
        .unwrap();

    let output = CosLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    // Cos's non-finite guard returns (-1, 1) for non-finite inputs.
    // But BoundedTensor::new may fail if the output contains values that
    // don't satisfy lower <= upper ordering. Let's just verify no NaN.
    assert!(!lower.is_nan(), "Cos IBP lower is NaN for [NaN, 1]");
    assert!(!upper.is_nan(), "Cos IBP upper is NaN for [NaN, 1]");
    assert!(
        lower >= -1.0 && upper <= 1.0,
        "Cos IBP bounds for [NaN, 1] should be in [-1, 1], got [{lower}, {upper}]"
    );
}

/// Sin IBP with infinite input falls back to [-1, 1] (same guard as Cos).
#[timeout(10000)]
#[test]
fn sin_ibp_infinite_falls_back_to_full_range() {
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[f32::INFINITY]).into_dyn(),
    )
    .unwrap();

    let output = SinLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Sin IBP lower is NaN for [-inf, inf]");
    assert!(!upper.is_nan(), "Sin IBP upper is NaN for [-inf, inf]");
    assert_eq!(
        lower, -1.0,
        "Sin IBP lower for [-inf, inf] should be -1.0, got {lower}"
    );
    assert_eq!(
        upper, 1.0,
        "Sin IBP upper for [-inf, inf] should be 1.0, got {upper}"
    );
}

// =========================================================================
// SOFTPLUS EDGE CASES (#2435)
// Softplus range: (0, +inf). softplus(x) = log(1 + exp(x)).
// softplus(-inf) → 0, softplus(+inf) → +inf.
// =========================================================================

/// Softplus IBP with -inf lower: softplus(-inf) → 0.
#[timeout(10000)]
#[test]
fn softplus_ibp_neg_infinity_lower() {
    let input = BoundedTensor::new_unchecked(
        arr1(&[f32::NEG_INFINITY]).into_dyn(),
        arr1(&[0.0]).into_dyn(),
    )
    .unwrap();

    let output = SoftplusLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Softplus IBP lower is NaN for [-inf, 0]");
    assert!(!upper.is_nan(), "Softplus IBP upper is NaN for [-inf, 0]");
    assert!(
        lower <= upper,
        "Softplus IBP bounds inverted: {lower} > {upper}"
    );
    // softplus(-inf) = 0, softplus(0) = ln(2) ≈ 0.693
    assert!(
        (0.0..=0.01).contains(&lower),
        "Softplus IBP lower for [-inf, 0] should be ≈ 0, got {lower}"
    );
    assert!(
        (upper - std::f32::consts::LN_2).abs() < 0.01,
        "Softplus IBP upper for [-inf, 0] should be ≈ ln(2) ≈ 0.693, got {upper}"
    );
}

/// Softplus IBP with large positive input doesn't overflow to NaN.
/// softplus(80) ≈ 80, softplus(88) ≈ 88 (large x: softplus(x) ≈ x).
#[timeout(10000)]
#[test]
fn softplus_ibp_large_positive_no_overflow() {
    let input = BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[80.0]).into_dyn()).unwrap();

    let output = SoftplusLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(!lower.is_nan(), "Softplus IBP lower is NaN for [0, 80]");
    assert!(!upper.is_nan(), "Softplus IBP upper is NaN for [0, 80]");
    assert!(
        lower <= upper,
        "Softplus IBP bounds inverted: {lower} > {upper}"
    );
    // softplus(0) = ln(2) ≈ 0.693, softplus(80) ≈ 80
    assert!(
        (lower - std::f32::consts::LN_2).abs() < 0.01,
        "Softplus IBP lower for [0, 80] should be ≈ ln(2), got {lower}"
    );
    assert!(
        (upper - 80.0).abs() < 0.1,
        "Softplus IBP upper for [0, 80] should be ≈ 80, got {upper}"
    );
}

/// Softplus IBP lower bound is always >= 0 (softplus(x) > 0 for all x).
#[timeout(10000)]
#[test]
fn softplus_ibp_lower_bound_nonnegative() {
    // Test with very negative inputs where softplus → 0
    let input =
        BoundedTensor::new(arr1(&[-1000.0]).into_dyn(), arr1(&[-500.0]).into_dyn()).unwrap();

    let output = SoftplusLayer::new().propagate_ibp(&input).unwrap();
    let lower = output.lower()[[0]];
    let upper = output.upper()[[0]];

    assert!(
        !lower.is_nan(),
        "Softplus IBP lower is NaN for [-1000, -500]"
    );
    assert!(
        !upper.is_nan(),
        "Softplus IBP upper is NaN for [-1000, -500]"
    );
    assert!(
        lower >= 0.0,
        "Softplus IBP lower must be >= 0 (range clamp #3316), got {lower}"
    );
    assert!(upper >= 0.0, "Softplus IBP upper must be >= 0, got {upper}");
}
