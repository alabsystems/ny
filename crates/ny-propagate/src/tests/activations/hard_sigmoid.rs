// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::*;
use ndarray::{ArrayD, IxDyn};

// ==================== HardSigmoid tests ====================

/// Helper: evaluate HardSigmoid with default parameters (alpha=0.2, beta=0.5)
fn hard_sigmoid_eval(x: f32) -> f32 {
    (0.2 * x + 0.5).clamp(0.0, 1.0)
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_ibp_entirely_zero_region() {
    // Default params: alpha=0.2, beta=0.5
    // y=0 when 0.2*x + 0.5 <= 0 → x <= -2.5
    let lower = ArrayD::from_elem(IxDyn(&[3]), -5.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), -3.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = HardSigmoidLayer::default();
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!(
            output.lower()[[i]].abs() < 1e-6,
            "HardSigmoid zero region: lower[{}] should be 0, got {}",
            i,
            output.lower()[[i]]
        );
        assert!(
            output.upper()[[i]].abs() < 1e-6,
            "HardSigmoid zero region: upper[{}] should be 0, got {}",
            i,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_ibp_entirely_one_region() {
    // y=1 when 0.2*x + 0.5 >= 1 → x >= 2.5
    let lower = ArrayD::from_elem(IxDyn(&[3]), 3.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[3]), 5.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = HardSigmoidLayer::default();
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..3 {
        assert!(
            (output.lower()[[i]] - 1.0).abs() < 1e-6,
            "HardSigmoid one region: lower[{}] should be 1, got {}",
            i,
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 1.0).abs() < 1e-6,
            "HardSigmoid one region: upper[{}] should be 1, got {}",
            i,
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_ibp_linear_region() {
    // In linear region: x ∈ [-2.5, 2.5] → y = 0.2*x + 0.5
    // Input [0, 1]: y = [0.5, 0.7]
    let lower = ArrayD::from_elem(IxDyn(&[2]), 0.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = HardSigmoidLayer::default();
    let output = layer.propagate_ibp(&input).unwrap();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - 0.5).abs() < 1e-6,
            "HardSigmoid linear: lower should be 0.5, got {}",
            output.lower()[[i]]
        );
        assert!(
            (output.upper()[[i]] - 0.7).abs() < 1e-6,
            "HardSigmoid linear: upper should be 0.7, got {}",
            output.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_ibp_crossing_all_regions() {
    // Input [-4, 4] spans all three regions
    // hard_sigmoid(-4) = 0, hard_sigmoid(4) = 1
    let lower = ArrayD::from_elem(IxDyn(&[1]), -4.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 4.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = HardSigmoidLayer::default();
    let output = layer.propagate_ibp(&input).unwrap();

    assert!(output.lower()[[0]].abs() < 1e-6, "lower should be 0");
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-6,
        "upper should be 1"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_ibp_custom_params() {
    // Custom alpha=0.5, beta=0.25
    // y = clamp(0.5*x + 0.25, 0, 1)
    // x=-1 → 0.5*(-1)+0.25 = -0.25 → clamped to 0
    // x=2 → 0.5*2+0.25 = 1.25 → clamped to 1
    let lower = ArrayD::from_elem(IxDyn(&[1]), -1.0f32);
    let upper = ArrayD::from_elem(IxDyn(&[1]), 2.0f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let layer = HardSigmoidLayer::new(0.5, 0.25);
    let output = layer.propagate_ibp(&input).unwrap();

    assert!(output.lower()[[0]].abs() < 1e-6, "lower should be 0");
    assert!(
        (output.upper()[[0]] - 1.0).abs() < 1e-6,
        "upper should be 1"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_linear_requires_preactivation_bounds() {
    let bounds = LinearBounds::identity(4);
    let layer = HardSigmoidLayer::default();
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "HardSigmoid::propagate_linear should error without pre-activation bounds"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_crown_soundness_zero_region() {
    // Entirely in y=0 region: [-5, -3]
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), -5.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), -3.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let layer = HardSigmoidLayer::default();

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // In zero region: slope=0, intercept=0
    assert!(
        result.lower_a[[0, 0]].abs() < 1e-6,
        "lower slope should be 0"
    );
    assert!(
        result.lower_b[0].abs() < 1e-6,
        "lower intercept should be 0"
    );
    assert!(
        result.upper_a[[0, 0]].abs() < 1e-6,
        "upper slope should be 0"
    );
    assert!(
        result.upper_b[0].abs() < 1e-6,
        "upper intercept should be 0"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_crown_soundness_linear_region() {
    // Entirely in linear region: [0, 1]
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), 0.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 1.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let layer = HardSigmoidLayer::default();

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // In linear region: slope=alpha=0.2, intercept=beta=0.5
    assert!(
        (result.lower_a[[0, 0]] - 0.2).abs() < 1e-6,
        "lower slope should be 0.2, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_b[0] - 0.5).abs() < 1e-6,
        "lower intercept should be 0.5, got {}",
        result.lower_b[0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_crown_soundness_non_crossing() {
    // Test CROWN soundness for intervals that don't cross region boundaries.
    // Crossing-boundary relaxation was once unsound (same chord for both upper
    // and lower bounds); now fixed by the 6-case analytical relaxation and
    // covered by test_hard_sigmoid_crown_crossing_soundness below.
    let pre_lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-5.0, 0.0, 3.0]).unwrap();
    let pre_upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![-3.0, 1.0, 5.0]).unwrap();
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(3);
    let layer = HardSigmoidLayer::default();

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    let intervals = [(-5.0f32, -3.0f32), (0.0, 1.0), (3.0, 5.0)];
    let tol = 1e-4;

    for (j, (l, u)) in intervals.iter().enumerate() {
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let x = l + (u - l) * t;
            let fx = hard_sigmoid_eval(x);

            let lb = result.lower_a[[j, j]] * x + result.lower_b[j];
            let ub = result.upper_a[[j, j]] * x + result.upper_b[j];

            assert!(
                lb <= fx + tol,
                "HardSigmoid CROWN lower violated at x={}: lb={} > f(x)={}",
                x,
                lb,
                fx
            );
            assert!(
                ub >= fx - tol,
                "HardSigmoid CROWN upper violated at x={}: ub={} < f(x)={}",
                x,
                ub,
                fx
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_crown_crossing_soundness() {
    // Previously UNSOUND: HardSigmoid CROWN used the same chord for both
    // upper and lower bounds when crossing region boundaries. Fixed by 6-case
    // analytical relaxation. Reference: designs/2026-02-08-piecewise-crown-relaxation-fixes.md Part 2
    //
    // Interval [-4, 4] crosses all three regions of HardSigmoid (Case 4).
    let pre_lower = ArrayD::from_elem(IxDyn(&[1]), -4.0f32);
    let pre_upper = ArrayD::from_elem(IxDyn(&[1]), 4.0f32);
    let pre_activation = BoundedTensor::new(pre_lower, pre_upper).unwrap();

    let linear_bounds = LinearBounds::identity(1);
    let layer = HardSigmoidLayer::default();

    let result = layer
        .propagate_linear_with_bounds(&linear_bounds, &pre_activation)
        .unwrap();

    // Verify soundness across the interval: lb <= f(x) <= ub for all x in [-4, 4].
    let l = -4.0f32;
    let u = 4.0f32;
    let tol = 1e-5;
    for i in 0..=100 {
        let t = i as f32 / 100.0;
        let x = (l + t * (u - l)).clamp(l, u);
        let fx = hard_sigmoid_eval(x);
        let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lb <= fx + tol,
            "HardSigmoid CROWN lower violated at x={}: lb={} > f(x)={}",
            x,
            lb,
            fx
        );
        assert!(
            ub >= fx - tol,
            "HardSigmoid CROWN upper violated at x={}: ub={} < f(x)={}",
            x,
            ub,
            fx
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_hard_sigmoid_crown_nan_inf_guard() {
    // NaN/Inf pre-activation bounds are now rejected by domain_guard (#2836)
    // with NumericalInstability error, triggering CROWN→IBP fallback.
    let layer = HardSigmoidLayer::default();
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
            "HardSigmoid NaN/Inf ({desc}): expected NumericalInstability, got {:?}",
            result
        );
    }
}
