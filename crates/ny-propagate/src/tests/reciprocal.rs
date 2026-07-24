// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::common::BoundPropagation;
use crate::layers::misc::reciprocal::reciprocal_linear_relaxation;
use ndarray::arr1;

fn eval_linear(bounds: &LinearBounds, x: f32) -> (f32, f32) {
    let lower = bounds.lower_a[[0, 0]] * x + bounds.lower_b[0];
    let upper = bounds.upper_a[[0, 0]] * x + bounds.upper_b[0];
    (lower, upper)
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_crown_bounds_positive_interval() {
    let bounds = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[4.0]).into_dyn()).unwrap();

    let recip = ReciprocalLayer::new();
    let out = recip
        .propagate_linear_with_bounds(&bounds, &pre_activation)
        .unwrap();

    for x in [2.0f32, 3.0, 4.0] {
        let (lower, upper) = eval_linear(&out, x);
        let true_val = 1.0 / x;
        assert!(
            lower <= true_val && true_val <= upper,
            "x={}, lower={}, true={}, upper={}",
            x,
            lower,
            true_val,
            upper
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_crown_bounds_negative_interval() {
    let bounds = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[-4.0]).into_dyn(), arr1(&[-2.0]).into_dyn()).unwrap();

    let recip = ReciprocalLayer::new();
    let out = recip
        .propagate_linear_with_bounds(&bounds, &pre_activation)
        .unwrap();

    for x in [-4.0f32, -3.0, -2.0] {
        let (lower, upper) = eval_linear(&out, x);
        let true_val = 1.0 / x;
        assert!(
            lower <= true_val && true_val <= upper,
            "x={}, lower={}, true={}, upper={}",
            x,
            lower,
            true_val,
            upper
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_crown_rejects_zero_crossing_interval() {
    let bounds = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();

    let recip = ReciprocalLayer::new();
    let result = recip.propagate_linear_with_bounds(&bounds, &pre_activation);
    assert!(result.is_err());
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_relaxation_nan_guard() {
    use crate::layers::misc::reciprocal::reciprocal_linear_relaxation;

    // NaN lower bound
    let r = reciprocal_linear_relaxation(f32::NAN, 2.0);
    assert_eq!(r.lower_slope, 0.0);
    assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
    assert_eq!(r.upper_slope, 0.0);
    assert_eq!(r.upper_intercept, f32::INFINITY);

    // NaN upper bound
    let r = reciprocal_linear_relaxation(1.0, f32::NAN);
    assert_eq!(r.lower_slope, 0.0);
    assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
    assert_eq!(r.upper_slope, 0.0);
    assert_eq!(r.upper_intercept, f32::INFINITY);

    // Both NaN
    let r = reciprocal_linear_relaxation(f32::NAN, f32::NAN);
    assert_eq!(r.lower_slope, 0.0);
    assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
    assert_eq!(r.upper_slope, 0.0);
    assert_eq!(r.upper_intercept, f32::INFINITY);
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_crown_rejects_zero_endpoint() {
    let bounds = LinearBounds::identity(1);
    let pre_activation =
        BoundedTensor::new(arr1(&[0.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();

    let recip = ReciprocalLayer::new();
    let result = recip.propagate_linear_with_bounds(&bounds, &pre_activation);
    assert!(result.is_err());
}

// =========================================================================
// IBP soundness tests — propagate_ibp had zero direct coverage.
// Part of proof_coverage audit.
// =========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_ibp_positive_interval() {
    let input = BoundedTensor::new(arr1(&[2.0]).into_dyn(), arr1(&[4.0]).into_dyn()).unwrap();
    let layer = ReciprocalLayer::new();
    let out = layer.propagate_ibp(&input).unwrap();

    // 1/x is decreasing on (0, inf): 1/4 <= y <= 1/2
    assert!(
        out.lower()[[0]] <= 0.25,
        "IBP lower {} should be <= 1/4",
        out.lower()[[0]]
    );
    assert!(
        out.upper()[[0]] >= 0.5,
        "IBP upper {} should be >= 1/2",
        out.upper()[[0]]
    );
    // Bounds should be tight (within 1 ULP of exact due to directed rounding).
    assert!(
        (out.lower()[[0]] - 0.25).abs() < 1e-6,
        "IBP lower {} should be close to 0.25",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[0]] - 0.5).abs() < 1e-6,
        "IBP upper {} should be close to 0.5",
        out.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_ibp_negative_interval() {
    let input = BoundedTensor::new(arr1(&[-4.0]).into_dyn(), arr1(&[-2.0]).into_dyn()).unwrap();
    let layer = ReciprocalLayer::new();
    let out = layer.propagate_ibp(&input).unwrap();

    // 1/x is decreasing on (-inf, 0): 1/(-4) = -0.25, 1/(-2) = -0.5
    // Decreasing: lb maps to upper, ub maps to lower → y in [-0.5, -0.25]
    assert!(
        out.lower()[[0]] <= -0.5,
        "IBP lower {} should be <= -0.5",
        out.lower()[[0]]
    );
    assert!(
        out.upper()[[0]] >= -0.25,
        "IBP upper {} should be >= -0.25",
        out.upper()[[0]]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_ibp_zero_crossing_conservative() {
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn()).unwrap();
    let layer = ReciprocalLayer::new();
    let out = layer.propagate_ibp(&input).unwrap();

    // Zero crossing: 1/x is unbounded on both sides, so [-inf, +inf] is the
    // only sound enclosure; new_repaired preserves the ±Inf endpoints rather
    // than substituting a finite clamp (#3423).
    assert_eq!(
        out.lower()[[0]],
        f32::NEG_INFINITY,
        "IBP zero-crossing lower should be -inf"
    );
    assert_eq!(
        out.upper()[[0]],
        f32::INFINITY,
        "IBP zero-crossing upper should be +inf"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_ibp_grid_soundness_positive() {
    let layer = ReciprocalLayer::new();
    for &l in &[0.01f32, 0.1, 0.5, 1.0, 2.0, 10.0] {
        for &u in &[0.1f32, 0.5, 1.0, 2.0, 10.0, 100.0] {
            if l >= u {
                continue;
            }
            let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
            let out = layer.propagate_ibp(&input).unwrap();

            for i in 0..=200 {
                let t = i as f32 / 200.0;
                let x = (l + (u - l) * t).clamp(l, u);
                let y = 1.0 / x;
                assert!(
                    out.lower()[[0]] <= y + 1e-5,
                    "IBP unsound: lower {} > 1/{x} = {y} for [{l}, {u}]",
                    out.lower()[[0]]
                );
                assert!(
                    out.upper()[[0]] >= y - 1e-5,
                    "IBP unsound: upper {} < 1/{x} = {y} for [{l}, {u}]",
                    out.upper()[[0]]
                );
            }
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_ibp_grid_soundness_negative() {
    let layer = ReciprocalLayer::new();
    for &l in &[-100.0f32, -10.0, -2.0, -1.0, -0.5, -0.1] {
        for &u in &[-10.0f32, -2.0, -1.0, -0.5, -0.1, -0.01] {
            if l >= u {
                continue;
            }
            let input = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn()).unwrap();
            let out = layer.propagate_ibp(&input).unwrap();

            for i in 0..=200 {
                let t = i as f32 / 200.0;
                let x = (l + (u - l) * t).clamp(l, u);
                let y = 1.0 / x;
                assert!(
                    out.lower()[[0]] <= y + 1e-5,
                    "IBP unsound: lower {} > 1/{x} = {y} for [{l}, {u}]",
                    out.lower()[[0]]
                );
                assert!(
                    out.upper()[[0]] >= y - 1e-5,
                    "IBP unsound: upper {} < 1/{x} = {y} for [{l}, {u}]",
                    out.upper()[[0]]
                );
            }
        }
    }
}

// =========================================================================
// Direct relaxation soundness tests.
// reciprocal_linear_relaxation() was only tested through CROWN backward.
// =========================================================================

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_relaxation_positive_interval_sound() {
    for &(l, u) in &[(0.5f32, 2.0), (0.01, 0.1), (1.0, 10.0), (0.1, 100.0)] {
        let r = reciprocal_linear_relaxation(l, u);
        assert_relaxation_sound(
            l,
            u,
            r,
            |x| 1.0 / x,
            1e-4,
            &format!("Reciprocal [{l}, {u}]"),
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_relaxation_negative_interval_sound() {
    for &(l, u) in &[
        (-2.0f32, -0.5),
        (-0.1, -0.01),
        (-10.0, -1.0),
        (-100.0, -0.1),
    ] {
        let r = reciprocal_linear_relaxation(l, u);
        assert_relaxation_sound(
            l,
            u,
            r,
            |x| 1.0 / x,
            1e-4,
            &format!("Reciprocal [{l}, {u}]"),
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_relaxation_degenerate_interval() {
    // l == u: relaxation should be tangent-at-point for both bounds.
    for &x in &[0.5f32, 1.0, 2.0, -1.0, -0.5] {
        let r = reciprocal_linear_relaxation(x, x);
        let y = 1.0 / x;
        let lower = r.lower_slope * x + r.lower_intercept;
        let upper = r.upper_slope * x + r.upper_intercept;
        assert!(
            lower <= y + 1e-5,
            "Degenerate relaxation lower {lower} > true {y} at x={x}"
        );
        assert!(
            upper >= y - 1e-5,
            "Degenerate relaxation upper {upper} < true {y} at x={x}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_reciprocal_relaxation_zero_crossing_returns_conservative() {
    let r = reciprocal_linear_relaxation(-1.0, 1.0);
    assert_eq!(r.lower_slope, 0.0);
    assert_eq!(r.lower_intercept, f32::NEG_INFINITY);
    assert_eq!(r.upper_slope, 0.0);
    assert_eq!(r.upper_intercept, f32::INFINITY);
}
