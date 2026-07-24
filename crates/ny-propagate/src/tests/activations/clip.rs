// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== Clip tests ====================

#[ntest::timeout(10000)]
#[test]
fn test_clip_ibp_entirely_within() {
    // Input entirely within [min, max] → identity
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = ClipLayer::new(0.0, 10.0);
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - input.lower()[[i]]).abs() < 1e-6,
            "Clip within: lower[{}] should be unchanged",
            i
        );
        assert!(
            (output.upper()[[i]] - input.upper()[[i]]).abs() < 1e-6,
            "Clip within: upper[{}] should be unchanged",
            i
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_ibp_entirely_below() {
    // Input entirely below min → output = min
    let lower = ArrayD::from_elem(IxDyn(&[2]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), -3.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = ClipLayer::new(0.0, 1.0);
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[i]].abs() < 1e-6,
            "Clip below: lower should be 0"
        );
        assert!(
            output.upper()[[i]].abs() < 1e-6,
            "Clip below: upper should be 0"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_ibp_entirely_above() {
    // Input entirely above max → output = max
    let lower = ArrayD::from_elem(IxDyn(&[2]), 5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 10.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = ClipLayer::new(0.0, 1.0);
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - 1.0).abs() < 1e-6,
            "Clip above: lower should be 1"
        );
        assert!(
            (output.upper()[[i]] - 1.0).abs() < 1e-6,
            "Clip above: upper should be 1"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_ibp_crossing_both_boundaries() {
    // Input [-5, 5] with clip [0, 1] → [0, 1]
    let lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = ClipLayer::new(0.0, 1.0);
    let output = layer.propagate_ibp(&input).unwrap();

    assert!(output.lower()[[0]].abs() < 1e-6, "lower should be 0");
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-6,
        "upper should be 1"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_ibp_crossing_lower_boundary() {
    // Input [-1, 0.5] with clip [0, 1] → [0, 0.5]
    let lower = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 0.5f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = ClipLayer::new(0.0, 1.0);
    let output = layer.propagate_ibp(&input).unwrap();

    assert!(output.lower()[[0]].abs() < 1e-6, "lower should be 0");
    assert!(
        (output.upper()[[0]] - 0.5).abs() < 1e-6,
        "upper should be 0.5"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_ibp_negative_range() {
    // Clip with negative range: min=-3, max=-1
    let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-5.0, -2.0, 0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-4.0, -0.5, 2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = ClipLayer::new(-3.0, -1.0);
    let output = layer.propagate_ibp(&input).unwrap();

    // Element 0: [-5, -4] entirely below -3 → [-3, -3]
    assert!((output.lower()[[0]] - (-3.0)).abs() < 1e-6);
    assert!((output.upper()[[0]] - (-3.0)).abs() < 1e-6);

    // Element 1: [-2, -0.5] straddles upper boundary → [-2, -1]
    assert!((output.lower()[[1]] - (-2.0)).abs() < 1e-6);
    assert!((output.upper()[[1]] - (-1.0)).abs() < 1e-6);

    // Element 2: [0, 2] entirely above -1 → [-1, -1]
    assert!((output.lower()[[2]] - (-1.0)).abs() < 1e-6);
    assert!((output.upper()[[2]] - (-1.0)).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_linear_requires_preactivation_bounds() {
    let bounds = LinearBounds::identity(4);
    let layer = ClipLayer::new(0.0, 1.0);
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "Clip::propagate_linear should error without pre-activation bounds"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_crown_soundness_within() {
    // Entirely within: should be identity relaxation
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), 0.3f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 0.7f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let layer = ClipLayer::new(0.0, 1.0);

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Should be identity: slope=1, intercept=0
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-6,
        "lower slope should be 1"
    );
    assert!(
        result.lower_b[0].abs() < 1e-6,
        "lower intercept should be 0"
    );
    assert!(
        (result.upper_a[[0, 0]] - 1.0).abs() < 1e-6,
        "upper slope should be 1"
    );
    assert!(
        result.upper_b[0].abs() < 1e-6,
        "upper intercept should be 0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_crown_soundness_below() {
    // Entirely below min: constant=min
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), -3.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let layer = ClipLayer::new(0.0, 1.0);

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Should be constant 0: slope=0, intercept=0
    assert!(
        result.lower_a[[0, 0]].abs() < 1e-6,
        "lower slope should be 0"
    );
    assert!(
        result.lower_b[0].abs() < 1e-6,
        "lower intercept should be 0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_crown_soundness_non_crossing() {
    // Test CROWN soundness for intervals entirely within one region.
    // Crossing-boundary relaxation was once unsound (same chord for both upper
    // and lower bounds); now fixed by the 6-case analytical relaxation and
    // covered by test_clip_crown_crossing_soundness below.
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-3.0, 0.3, 2.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-1.0, 0.7, 4.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let min_val = 0.0f32;
    let max_val = 1.0f32;
    let layer = ClipLayer::new(min_val, max_val);

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let intervals = [(-3.0f32, -1.0f32), (0.3, 0.7), (2.0, 4.0)];
    let tol = 1e-4;

    for (j, (l, u)) in intervals.iter().enumerate() {
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let x = l + (u - l) * t;
            let fx = x.clamp(min_val, max_val);

            let lb = result.lower_a[[j, j]] * x + result.lower_b[j];
            let ub = result.upper_a[[j, j]] * x + result.upper_b[j];

            assert!(
                lb <= fx + tol,
                "Clip CROWN lower violated at x={}: lb={} > f(x)={}",
                x,
                lb,
                fx
            );
            assert!(
                ub >= fx - tol,
                "Clip CROWN upper violated at x={}: ub={} < f(x)={}",
                x,
                ub,
                fx
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_crown_crossing_soundness() {
    // Previously UNSOUND: Clip CROWN used the same chord for both upper and
    // lower bounds when crossing boundaries. Fixed by 6-case analytical
    // relaxation. Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 1
    //
    // Interval [-2, 0.5] crosses the lower boundary of clip(0, 1).
    // Case 5 (lower boundary crossing): upper = chord, lower = adaptive.
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), -2.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 0.5f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let layer = ClipLayer::new(0.0, 1.0);

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Verify soundness across the interval: lb <= f(x) <= ub for all x in [-2, 0.5].
    let l = -2.0f32;
    let u = 0.5f32;
    let tol = 1e-5;
    for i in 0..=100 {
        let t = i as f32 / 100.0;
        let x = (l + t * (u - l)).clamp(l, u);
        let fx = x.clamp(0.0, 1.0);
        let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lb <= fx + tol,
            "Clip CROWN lower violated at x={}: lb={} > f(x)={}",
            x,
            lb,
            fx
        );
        assert!(
            ub >= fx - tol,
            "Clip CROWN upper violated at x={}: ub={} < f(x)={}",
            x,
            ub,
            fx
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_clip_crown_nan_inf_guard() {
    // Updated for #2977: CROWN backward domain_guard rejects non-finite pre-activation
    // with NumericalInstability. Previously this tested maximally-loose fallback bounds;
    // now it verifies the early rejection is correct.
    let layer = ClipLayer::new(0.0, 6.0);
    let linear_bounds = LinearBounds::identity(1);

    for (l, u, desc) in [
        (f32::NAN, 1.0f32, "NaN lower"),
        (1.0f32, f32::NAN, "NaN upper"),
        (f32::NEG_INFINITY, 1.0f32, "neg-inf lower"),
        (1.0f32, f32::INFINITY, "pos-inf upper"),
        (f32::NEG_INFINITY, f32::INFINITY, "both inf"),
    ] {
        let pre = BoundedTensor::new_unchecked(
            ArrayD::from_elem(IxDyn(&[1]), l),
            ArrayD::from_elem(IxDyn(&[1]), u),
        )
        .unwrap();

        let result = layer.propagate_linear_with_bounds(&linear_bounds, &pre);
        assert!(
            matches!(result, Err(NyError::NumericalInstability(_))),
            "Clip NaN/Inf ({desc}): non-finite pre-activation should trigger domain_guard: got {:?}",
            result
        );
    }
}
