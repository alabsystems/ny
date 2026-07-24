// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::LinearBounds;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use proptest::prelude::ProptestConfig;

#[test]
fn test_new_stores_params() {
    let layer = ShrinkLayer::new(0.5, 1.0);
    assert!(
        (layer.bias() - 0.5).abs() < 1e-5,
        "bias expected 0.5, got {}",
        layer.bias()
    );
    assert!(
        (layer.lambd() - 1.0).abs() < 1e-5,
        "lambd expected 1.0, got {}",
        layer.lambd()
    );
}

#[test]
fn test_default_params() {
    let layer = ShrinkLayer::default();
    assert!(
        (layer.bias() - 0.0).abs() < 1e-5,
        "default bias expected 0.0, got {}",
        layer.bias()
    );
    assert!(
        (layer.lambd() - 0.5).abs() < 1e-5,
        "default lambd expected 0.5, got {}",
        layer.lambd()
    );
}

#[test]
fn test_try_new_rejects_invalid_params_2551() {
    for bias in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = ShrinkLayer::try_new(bias, 0.5).expect_err("invalid bias should be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }

    for lambd in [-0.1, f32::NAN, f32::INFINITY] {
        let err = ShrinkLayer::try_new(0.0, lambd).expect_err("invalid lambd should be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }
}

#[test]
fn test_shrink_scalar_regions() {
    // Region A (x < -lambd): y = x + bias
    assert!(
        (shrink_scalar(-2.0, 0.5, 1.0) - (-1.5)).abs() < 1e-5,
        "region A: shrink(-2) expected -1.5, got {}",
        shrink_scalar(-2.0, 0.5, 1.0)
    );
    // Dead zone: y = 0
    assert!(
        (shrink_scalar(0.0, 0.5, 1.0) - 0.0).abs() < 1e-5,
        "dead zone: shrink(0) expected 0, got {}",
        shrink_scalar(0.0, 0.5, 1.0)
    );
    assert!(
        (shrink_scalar(-0.5, 0.5, 1.0) - 0.0).abs() < 1e-5,
        "dead zone: shrink(-0.5) expected 0, got {}",
        shrink_scalar(-0.5, 0.5, 1.0)
    );
    // Region C (x > lambd): y = x - bias
    assert!(
        (shrink_scalar(2.0, 0.5, 1.0) - 1.5).abs() < 1e-5,
        "region C: shrink(2) expected 1.5, got {}",
        shrink_scalar(2.0, 0.5, 1.0)
    );
}

#[test]
fn test_relaxation_infinite_interval_uses_infinite_intercepts() {
    let layer = ShrinkLayer::new(0.5, 0.5);
    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = layer.relaxation(f32::NEG_INFINITY, f32::INFINITY);

    assert!(
        (lower_slope - 0.0).abs() < 1e-6,
        "lower_slope expected 0.0, got {lower_slope}"
    );
    assert!(
        lower_intercept.is_infinite() && lower_intercept.is_sign_negative(),
        "lower_intercept expected -inf, got {lower_intercept}"
    );
    assert!(
        (upper_slope - 0.0).abs() < 1e-6,
        "upper_slope expected 0.0, got {upper_slope}"
    );
    assert!(
        upper_intercept.is_infinite() && upper_intercept.is_sign_positive(),
        "upper_intercept expected +inf, got {upper_intercept}"
    );

    // Regression for #2337: non-finite interval must not collapse to (0, 0, 0, 0).
    for x in [-10.0_f32, -1.0, 0.0, 1.0, 10.0] {
        let y = shrink_scalar(x, layer.bias(), layer.lambd());
        let lower_bound = lower_slope * x + lower_intercept;
        let upper_bound = upper_slope * x + upper_intercept;
        assert!(
            lower_bound <= y,
            "lower bound {lower_bound} > shrink({x})={y}"
        );
        assert!(
            upper_bound >= y,
            "upper bound {upper_bound} < shrink({x})={y}"
        );
    }
}

#[test]
fn test_inf_guard_bias_gt_lambd_includes_breakpoint_extrema() {
    // Regression for #3322 audit finding: when bias > lambd, Shrink has
    // discontinuities at ±lambd. The function approaches -lambd + bias > 0
    // from the left at -lambd, and lambd - bias < 0 from the right at +lambd.
    // The Inf guard must include these breakpoint approach values in fmax/fmin.
    let layer = ShrinkLayer::new(5.0, 1.0); // bias=5 > lambd=1

    // Case: l = -inf, u = 0. The neg breakpoint at -lambd = -1 is in [-inf, 0].
    // Function approaches -1 + 5 = 4 from the left. Upper bound must be >= 4.
    let r = layer.relaxation(f32::NEG_INFINITY, 0.0);
    assert!(
        r.upper_intercept >= 4.0 - f32::EPSILON,
        "upper_intercept {} must cover breakpoint approach value 4.0",
        r.upper_intercept
    );
    // Verify soundness at the critical point near -lambd
    for &x in &[-1.5, -1.01, -1.001, -0.5, 0.0] {
        let y = shrink_scalar(x, 5.0, 1.0);
        assert!(
            r.upper_slope * x + r.upper_intercept >= y - f32::EPSILON,
            "upper bound violated at x={}: {} < {}",
            x,
            r.upper_slope * x + r.upper_intercept,
            y
        );
        assert!(
            r.lower_slope * x + r.lower_intercept <= y + f32::EPSILON,
            "lower bound violated at x={}: {} > {}",
            x,
            r.lower_slope * x + r.lower_intercept,
            y
        );
    }

    // Case: l = 0, u = +inf. The pos breakpoint at lambd = 1 is in [0, +inf].
    // Function approaches 1 - 5 = -4 from the right. Lower bound must be <= -4.
    let r = layer.relaxation(0.0, f32::INFINITY);
    assert!(
        r.lower_intercept <= -4.0 + f32::EPSILON,
        "lower_intercept {} must cover breakpoint approach value -4.0",
        r.lower_intercept
    );
    for &x in &[0.0, 0.5, 1.001, 1.5, 2.0] {
        let y = shrink_scalar(x, 5.0, 1.0);
        assert!(
            r.upper_slope * x + r.upper_intercept >= y - f32::EPSILON,
            "upper bound violated at x={}: {} < {}",
            x,
            r.upper_slope * x + r.upper_intercept,
            y
        );
        assert!(
            r.lower_slope * x + r.lower_intercept <= y + f32::EPSILON,
            "lower bound violated at x={}: {} > {}",
            x,
            r.lower_slope * x + r.lower_intercept,
            y
        );
    }
}

#[test]
fn test_case5_chord_breakpoint_validity_bias_gt_lambd() {
    // Regression for #3322: Case 5 chord from (l, 0) to (u, fu) was accepted
    // without checking at the positive breakpoint +lambd. When bias > lambd,
    // the function dips to lambd - bias < 0 at the breakpoint, but the chord
    // passes above this dip, yielding an unsound lower bound.
    let layer = ShrinkLayer::new(1.5, 0.1); // bias=1.5 > lambd=0.1
    let r = layer.relaxation(0.0, 0.5);

    // Verify soundness at dense points around the breakpoint
    for k in 0..=100 {
        let x = 0.0 + k as f32 * 0.005;
        let y = shrink_scalar(x, 1.5, 0.1);
        let lower = r.lower_slope * x + r.lower_intercept;
        let upper = r.upper_slope * x + r.upper_intercept;
        assert!(
            lower <= y + f32::EPSILON,
            "Case 5 lower bound UNSOUND at x={}: {} > shrink({})={}",
            x,
            lower,
            x,
            y
        );
        assert!(
            upper >= y - f32::EPSILON,
            "Case 5 upper bound UNSOUND at x={}: {} < shrink({})={}",
            x,
            upper,
            x,
            y
        );
    }
}

#[test]
fn test_relaxation_nan_interval_uses_infinite_intercepts() {
    let layer = ShrinkLayer::new(0.5, 0.5);
    let LinearRelaxation {
        lower_slope,
        lower_intercept,
        upper_slope,
        upper_intercept,
    } = layer.relaxation(f32::NAN, 1.0);

    assert!(
        (lower_slope - 0.0).abs() < 1e-6,
        "NaN lower_slope expected 0.0, got {lower_slope}"
    );
    assert!(
        lower_intercept.is_infinite() && lower_intercept.is_sign_negative(),
        "NaN lower_intercept expected -inf, got {lower_intercept}"
    );
    assert!(
        (upper_slope - 0.0).abs() < 1e-6,
        "NaN upper_slope expected 0.0, got {upper_slope}"
    );
    assert!(
        upper_intercept.is_infinite() && upper_intercept.is_sign_positive(),
        "NaN upper_intercept expected +inf, got {upper_intercept}"
    );
}

#[test]
fn test_ibp_entirely_dead_zone() {
    // Case 2: [-0.3, 0.4] with lambd=0.5
    let layer = ShrinkLayer::new(0.0, 0.5);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-0.3, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.4, 0.3]).unwrap(),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert!(
        (out.lower()[[0]] - 0.0).abs() < 1e-5,
        "dead zone lower[0] expected 0.0, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[0]] - 0.0).abs() < 1e-5,
        "dead zone upper[0] expected 0.0, got {}",
        out.upper()[[0]]
    );
    assert!(
        (out.lower()[[1]] - 0.0).abs() < 1e-5,
        "dead zone lower[1] expected 0.0, got {}",
        out.lower()[[1]]
    );
    assert!(
        (out.upper()[[1]] - 0.0).abs() < 1e-5,
        "dead zone upper[1] expected 0.0, got {}",
        out.upper()[[1]]
    );
}

#[test]
fn test_ibp_entirely_region_a() {
    // Case 1: [-3, -2] with lambd=1.0, bias=0.5
    // y = x + 0.5: lower = -3+0.5=-2.5, upper = -2+0.5=-1.5
    let layer = ShrinkLayer::new(0.5, 1.0);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert!(
        (out.lower()[[0]] - (-2.5)).abs() < 1e-5,
        "region A lower expected -2.5, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[0]] - (-1.5)).abs() < 1e-5,
        "region A upper expected -1.5, got {}",
        out.upper()[[0]]
    );
}

#[test]
fn test_ibp_entirely_region_c() {
    // Case 3: [2, 4] with lambd=1.0, bias=0.5
    // y = x - 0.5: lower = 2-0.5=1.5, upper = 4-0.5=3.5
    let layer = ShrinkLayer::new(0.5, 1.0);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![4.0]).unwrap(),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert!(
        (out.lower()[[0]] - 1.5).abs() < 1e-5,
        "region C lower expected 1.5, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[0]] - 3.5).abs() < 1e-5,
        "region C upper expected 3.5, got {}",
        out.upper()[[0]]
    );
}

#[test]
fn test_ibp_spans_neg_breakpoint() {
    // Case 4: [-2, 0.3] with lambd=0.5, bias=0.0
    // f(-2) = -2+0 = -2, f(-0.5)=0, f(0.3)=0
    // Lower: min(-2, 0) = -2, Upper: max(-2, 0) = 0
    let layer = ShrinkLayer::new(0.0, 0.5);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.3]).unwrap(),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert!(
        out.lower()[[0]] <= -2.0 + 1e-5,
        "neg breakpoint lower {} should be <= -2.0",
        out.lower()[[0]]
    );
    assert!(
        out.upper()[[0]] >= 0.0 - 1e-5,
        "neg breakpoint upper {} should be >= 0.0",
        out.upper()[[0]]
    );
}

#[test]
fn test_ibp_spans_pos_breakpoint() {
    // Case 5: [-0.3, 2] with lambd=0.5, bias=0.0
    // f(-0.3)=0, f(0.5)=0, f(2)=2-0=2
    // Lower=0, Upper=2
    let layer = ShrinkLayer::new(0.0, 0.5);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert!(
        out.lower()[[0]] <= 0.0 + 1e-5,
        "pos breakpoint lower {} should be <= 0.0",
        out.lower()[[0]]
    );
    assert!(
        out.upper()[[0]] >= 2.0 - 1e-5,
        "pos breakpoint upper {} should be >= 2.0",
        out.upper()[[0]]
    );
}

#[test]
fn test_ibp_spans_both_breakpoints() {
    // Case 6: [-3, 3] with lambd=1.0, bias=0.0
    // f(-3)=-3, f(-1)=0, f(0)=0, f(1)=0, f(3)=3
    // Lower=-3, Upper=3
    let layer = ShrinkLayer::new(0.0, 1.0);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();
    assert!(
        out.lower()[[0]] <= -3.0 + 1e-5,
        "both breakpoints lower {} should be <= -3.0",
        out.lower()[[0]]
    );
    assert!(
        out.upper()[[0]] >= 3.0 - 1e-5,
        "both breakpoints upper {} should be >= 3.0",
        out.upper()[[0]]
    );
}

#[test]
fn test_ibp_soundness_grid() {
    // Verify soundness across many evaluation points
    let layer = ShrinkLayer::new(0.3, 0.8);
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    )
    .unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    for i in 0..41 {
        let x = -2.0 + (i as f32) * 0.1;
        let y = shrink_scalar(x, 0.3, 0.8);
        assert!(
            out.lower()[[0]] <= y + 1e-5,
            "Lower {} > eval {} at x={}",
            out.lower()[[0]],
            y,
            x
        );
        assert!(
            out.upper()[[0]] >= y - 1e-5,
            "Upper {} < eval {} at x={}",
            out.upper()[[0]],
            y,
            x
        );
    }
}

#[test]
fn test_propagate_linear_returns_error() {
    let layer = ShrinkLayer::default();
    let bounds = LinearBounds::new(
        Array2::eye(2),
        Array1::zeros(2),
        Array2::eye(2),
        Array1::zeros(2),
    )
    .unwrap();
    assert!(
        layer.propagate_linear(&bounds).is_err(),
        "propagate_linear without pre-activation should error"
    );
}

#[test]
fn test_requires_pre_activation_bounds() {
    let layer = ShrinkLayer::default();
    assert!(
        layer.requires_pre_activation_bounds(),
        "Shrink should require pre-activation bounds"
    );
}

#[test]
fn test_crown_case1_entirely_negative() {
    // Case 1: entirely in region A (u < -lambd) => exact: y = x + bias
    // With identity CROWN bounds, result should be exact
    let layer = ShrinkLayer::new(0.5, 1.0);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.5]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    let result = BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act).unwrap();
    // Exact: y = x + 0.5, so slope=1, intercept=0.5
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "case1 lower_a expected 1.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_b[0] - 0.5).abs() < 1e-5,
        "case1 lower_b expected 0.5, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_a[[0, 0]] - 1.0).abs() < 1e-5,
        "case1 upper_a expected 1.0, got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        (result.upper_b[0] - 0.5).abs() < 1e-5,
        "case1 upper_b expected 0.5, got {}",
        result.upper_b[0]
    );
}

#[test]
fn test_crown_case2_dead_zone() {
    // Case 2: dead zone => exact: y = 0
    let layer = ShrinkLayer::new(0.0, 1.0);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-0.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.8]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    let result = BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act).unwrap();
    // Exact: y = 0, so all coefficients and biases should be 0
    assert!(
        (result.lower_a[[0, 0]]).abs() < 1e-5,
        "case2 lower_a expected 0.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_b[0]).abs() < 1e-5,
        "case2 lower_b expected 0.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.upper_a[[0, 0]]).abs() < 1e-5,
        "case2 upper_a expected 0.0, got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        (result.upper_b[0]).abs() < 1e-5,
        "case2 upper_b expected 0.0, got {}",
        result.upper_b[0]
    );
}

#[test]
fn test_crown_case3_entirely_positive() {
    // Case 3: entirely in region C (l > lambd) => exact: y = x - bias
    let layer = ShrinkLayer::new(0.5, 1.0);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    let result = BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act).unwrap();
    // Exact: y = x - 0.5
    assert!(
        (result.lower_a[[0, 0]] - 1.0).abs() < 1e-5,
        "case3 lower_a expected 1.0, got {}",
        result.lower_a[[0, 0]]
    );
    assert!(
        (result.lower_b[0] - (-0.5)).abs() < 1e-5,
        "case3 lower_b expected -0.5, got {}",
        result.lower_b[0]
    );
}

#[test]
fn test_crown_soundness_crossing_region() {
    // Test soundness for a crossing case (spans both breakpoints)
    let layer = ShrinkLayer::new(0.0, 1.0);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![2.0]).unwrap(),
    )
    .unwrap();
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    let result = BoundPropagation::propagate_linear_with_bounds(&layer, &bounds, &pre_act).unwrap();

    // Verify linear bounds contain f(x) for grid of x values
    for i in 0..41 {
        let x = -2.0 + (i as f32) * 0.1;
        let y = shrink_scalar(x, 0.0, 1.0);
        let lb = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let ub = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(lb <= y + 1e-5, "CROWN lower {} > eval {} at x={}", lb, y, x);
        assert!(ub >= y - 1e-5, "CROWN upper {} < eval {} at x={}", ub, y, x);
    }
}

#[test]
fn test_crown_no_pre_activation_errors() {
    let layer = ShrinkLayer::default();
    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .unwrap();
    assert!(
        layer.propagate_crown_backward(&bounds, None).is_err(),
        "CROWN backward without pre-activation bounds should error"
    );
}

// ── Regression: Inf guard soundness (#2337) ─────────────────────
//
// When l=-inf and u=+inf, the Inf guard must produce sound bounds.
// Without the fix, the is_finite() fallback returned 0.0 for both
// intercepts, claiming Shrink(x) = 0 for all x — unsound for x
// outside the dead zone.

/// Post-#2977: domain_guard now rejects non-finite pre-activation bounds with
/// NumericalInstability error instead of passing to the internal relaxation
/// Inf guard. This is the correct behavior — non-finite pre-activation bounds
/// indicate upstream corruption, not a valid input configuration.
#[test]
fn test_crown_both_infinite_bounds_regression_2337() -> Result<()> {
    let layer = ShrinkLayer::new(0.5, 0.5);
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1]), f32::INFINITY),
    )?;
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre);
    assert!(
        result.is_err(),
        "Non-finite pre-activation bounds should be rejected by domain_guard (#2977)"
    );
    Ok(())
}

/// Post-#2977: domain_guard rejects NaN pre-activation bounds.
#[test]
fn test_crown_nan_bounds_regression_2337() -> Result<()> {
    let layer = ShrinkLayer::new(0.5, 0.5);
    let pre = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), f32::NAN),
        ArrayD::from_elem(IxDyn(&[1]), 1.0),
    )?;
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre);
    assert!(
        result.is_err(),
        "NaN pre-activation bounds should be rejected by domain_guard (#2977)"
    );
    Ok(())
}

// ── IBP guard regression tests (#3278) ────────────────────────────

/// NaN in lower bounds silently fell into dead-zone branch, returning
/// unsound (0.0, 0.0). Guard now catches this.
#[test]
fn test_ibp_nan_input_lower_rejected_3278() {
    let layer = ShrinkLayer::default();
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), f32::NAN),
        ArrayD::from_elem(IxDyn(&[1]), 1.0),
    )
    .unwrap();
    let err = layer.propagate_ibp(&input).expect_err("NaN input lower");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

#[test]
fn test_ibp_nan_input_upper_rejected_3278() {
    let layer = ShrinkLayer::default();
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), -1.0),
        ArrayD::from_elem(IxDyn(&[1]), f32::NAN),
    )
    .unwrap();
    let err = layer.propagate_ibp(&input).expect_err("NaN input upper");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

#[test]
fn test_ibp_inf_input_rejected_3278() {
    let layer = ShrinkLayer::default();
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_elem(IxDyn(&[1]), f32::NEG_INFINITY),
        ArrayD::from_elem(IxDyn(&[1]), f32::INFINITY),
    )
    .unwrap();
    let err = layer.propagate_ibp(&input).expect_err("Inf input");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

// ── CROWN relaxation soundness proptest (#3321) ─────────────────────

/// Reference Shrink in f64, independent of the crate f32 implementation.
fn shrink_f64_reference(x: f64, bias: f64, lambd: f64) -> f64 {
    if x > lambd {
        x - bias
    } else if x < -lambd {
        x + bias
    } else {
        0.0
    }
}

proptest::proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #3321: Verify ShrinkLayer::relaxation produces strictly sound bounds.
    /// For random intervals, the lower bound must satisfy
    ///   lower_slope * x + lower_intercept <= Shrink(x)  for all x in [l, u]
    /// and the upper bound must satisfy
    ///   upper_slope * x + upper_intercept >= Shrink(x)  for all x in [l, u]
    /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
    ///
    /// Ref: ELU proptest_elu_relaxation_strict_soundness (elu.rs:841).
    #[test]
    fn proptest_shrink_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
        bias in -10.0f32..10.0,
        lambd in 0.01f32..10.0,
    ) {
        let u = l + width;
        let layer = ShrinkLayer::new(bias, lambd);
        let relax = layer.relaxation(l, u);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        // Skip NaN fallback (infinite bounds).
        proptest::prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        let bias64 = bias as f64;
        let lambd64 = lambd as f64;

        // Dense grid: 200 points, evaluated in f64 for mathematical precision.
        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = shrink_f64_reference(x, bias64, lambd64);

            let lower_val = ls as f64 * x + li as f64;
            proptest::prop_assert!(
                lower_val <= fx,
                "Shrink lower bound UNSOUND at x={}: {} > Shrink({})={}, \
                 interval=[{}, {}], bias={}, lambd={}, gap={}", x, lower_val, x, fx, l, u, bias, lambd, lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            proptest::prop_assert!(
                upper_val >= fx,
                "Shrink upper bound UNSOUND at x={}: {} < Shrink({})={}, \
                 interval=[{}, {}], bias={}, lambd={}, gap={}", x, upper_val, x, fx, l, u, bias, lambd, fx - upper_val
            );
        }
    }
}
