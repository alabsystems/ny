// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::tests::assert_close;
use crate::LinearBounds;
use ndarray::arr1;
use ny_core::Result;
use proptest::prelude::ProptestConfig;

const TOL: f32 = 1e-5;

/// Evaluate PReLU pointwise: y = x if x >= 0, else slope * x.
fn prelu_eval(x: f32, slope: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        slope * x
    }
}

// ---- Constructor tests ----

#[ntest::timeout(5000)]
#[test]
fn test_new_stores_slope() {
    let layer = PReluLayer::new(arr1(&[0.1, 0.2, 0.3])).expect("invariant: non-empty slope");
    assert_eq!(layer.slope.len(), 3);
    assert_close(layer.slope[0], 0.1, TOL);
    assert_close(layer.slope[2], 0.3, TOL);
}

#[ntest::timeout(5000)]
#[test]
fn test_from_scalar_creates_single_slope() {
    let layer = PReluLayer::from_scalar(0.25);
    assert_eq!(layer.slope.len(), 1);
    assert_close(layer.slope[0], 0.25, TOL);
}

#[ntest::timeout(5000)]
#[test]
fn test_get_slope_broadcasts_single() {
    let layer = PReluLayer::from_scalar(0.25);
    assert_close(layer.slope(0), 0.25, TOL);
    assert_close(layer.slope(5), 0.25, TOL);
    assert_close(layer.slope(100), 0.25, TOL);
}

#[ntest::timeout(5000)]
#[test]
fn test_get_slope_indexes_per_channel() {
    let layer = PReluLayer::new(arr1(&[0.1, 0.2, 0.3])).expect("invariant: non-empty slope");
    assert_close(layer.slope(0), 0.1, TOL);
    assert_close(layer.slope(1), 0.2, TOL);
    assert_close(layer.slope(2), 0.3, TOL);
    // Wraps around via modulo
    assert_close(layer.slope(3), 0.1, TOL);
}

// ---- IBP soundness (positive slope) ----

#[ntest::timeout(5000)]
#[test]
fn test_ibp_positive_slope_all_positive() -> Result<()> {
    // x in [1, 3], slope = 0.25 → all positive → identity
    let layer = PReluLayer::from_scalar(0.25);
    let input = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], 1.0, TOL);
    assert_close(out.upper()[[0]], 3.0, TOL);
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_positive_slope_all_negative() -> Result<()> {
    // x in [-3, -1], slope = 0.25 → all negative → y = 0.25*x
    let layer = PReluLayer::from_scalar(0.25);
    let input = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[-1.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], -0.75, TOL); // 0.25 * -3
    assert_close(out.upper()[[0]], -0.25, TOL); // 0.25 * -1
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_positive_slope_crossing() -> Result<()> {
    // x in [-2, 3], slope = 0.25
    // lower: min(0.25*(-2), 0) = -0.5 (at x=-2, y=0.25*(-2)=-0.5)
    // upper: max(3, 0.25*3) = 3 (at x=3, y=3 since x>0)
    let layer = PReluLayer::from_scalar(0.25);
    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], -0.5, TOL);
    assert_close(out.upper()[[0]], 3.0, TOL);
    Ok(())
}

// ---- IBP soundness (negative slope) ----

#[ntest::timeout(5000)]
#[test]
fn test_ibp_negative_slope_all_negative() -> Result<()> {
    // x in [-3, -1], slope = -0.5 → y = -0.5 * x
    // Since slope < 0 and both negative: lower = slope*u, upper = slope*l
    let layer = PReluLayer::from_scalar(-0.5);
    let input = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[-1.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    // slope * u = -0.5 * (-1) = 0.5
    // slope * l = -0.5 * (-3) = 1.5
    assert_close(out.lower()[[0]], 0.5, TOL);
    assert_close(out.upper()[[0]], 1.5, TOL);
    Ok(())
}

// ---- IBP per-channel slopes ----

#[ntest::timeout(5000)]
#[test]
fn test_ibp_per_channel_slopes() -> Result<()> {
    let layer = PReluLayer::new(arr1(&[0.1, 0.5])).expect("invariant: non-empty slope");
    // x0 in [-2, 1], x1 in [-1, 3]
    let input = BoundedTensor::new(arr1(&[-2.0, -1.0]).into_dyn(), arr1(&[1.0, 3.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;

    // Verify bounds contain concrete evals
    for &x0 in &[-2.0, -1.0, 0.0, 0.5, 1.0] {
        if !(-2.0..=1.0).contains(&x0) {
            continue;
        }
        let y0 = prelu_eval(x0, 0.1);
        assert!(
            out.lower()[[0]] <= y0 + 1e-5,
            "dim 0: lower {} > eval({x0}) = {y0}",
            out.lower()[[0]]
        );
        assert!(
            out.upper()[[0]] >= y0 - 1e-5,
            "dim 0: upper {} < eval({x0}) = {y0}",
            out.upper()[[0]]
        );
    }
    for &x1 in &[-1.0, 0.0, 1.5, 3.0] {
        let y1 = prelu_eval(x1, 0.5);
        assert!(
            out.lower()[[1]] <= y1 + 1e-5,
            "dim 1: lower {} > eval({x1}) = {y1}",
            out.lower()[[1]]
        );
        assert!(
            out.upper()[[1]] >= y1 - 1e-5,
            "dim 1: upper {} < eval({x1}) = {y1}",
            out.upper()[[1]]
        );
    }
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_non_finite_input_falls_back_to_infinite_bounds() -> Result<()> {
    let layer = PReluLayer::from_scalar(-0.5);
    let input =
        BoundedTensor::new_unchecked(arr1(&[f32::NAN]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    let lower = out.lower()[[0]];
    let upper = out.upper()[[0]];
    assert!(lower.is_infinite() && lower.is_sign_negative());
    assert!(upper.is_infinite() && upper.is_sign_positive());
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_non_finite_input_falls_back_to_infinite_bounds_when_slope_non_finite() -> Result<()> {
    let layer = PReluLayer::from_scalar(f32::NAN);
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[1.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    let lower = out.lower()[[0]];
    let upper = out.upper()[[0]];
    assert!(lower.is_infinite() && lower.is_sign_negative());
    assert!(upper.is_infinite() && upper.is_sign_positive());
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_non_finite_slope_returns_wide_bounds() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = prelu_linear_relaxation(-1.0, 1.0, f32::NAN);
    assert_eq!(ls, 0.0);
    assert!(li.is_infinite() && li.is_sign_negative());
    assert_eq!(us, 0.0);
    assert!(ui.is_infinite() && ui.is_sign_positive());
}

// ---- CROWN backward (crossing region) ----

#[ntest::timeout(5000)]
#[test]
fn test_crown_all_positive_is_identity() -> Result<()> {
    let layer = PReluLayer::from_scalar(0.25);
    let bounds = LinearBounds::identity(2);
    let pre_act = BoundedTensor::new(arr1(&[1.0, 2.0]).into_dyn(), arr1(&[3.0, 4.0]).into_dyn())?;
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;
    // All positive → identity: slope=1, intercept=0
    for i in 0..2 {
        assert_close(result.lower_a[[i, i]], 1.0, TOL);
        assert_close(result.upper_a[[i, i]], 1.0, TOL);
    }
    assert_close(result.lower_b[0], 0.0, TOL);
    assert_close(result.upper_b[0], 0.0, TOL);
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_all_negative_is_scaled() -> Result<()> {
    let layer = PReluLayer::from_scalar(0.25);
    let bounds = LinearBounds::identity(2);
    let pre_act = BoundedTensor::new(
        arr1(&[-4.0, -3.0]).into_dyn(),
        arr1(&[-1.0, -0.5]).into_dyn(),
    )?;
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;
    // All negative → scaled: slope=0.25, intercept=0
    for i in 0..2 {
        assert_close(result.lower_a[[i, i]], 0.25, TOL);
        assert_close(result.upper_a[[i, i]], 0.25, TOL);
    }
    assert_close(result.lower_b[0], 0.0, TOL);
    assert_close(result.upper_b[0], 0.0, TOL);
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_crossing_bounds_are_sound() -> Result<()> {
    // Crossing region: l < 0 < u, slope = 0.25
    let layer = PReluLayer::from_scalar(0.25);
    let bounds = LinearBounds::identity(1);
    let pre_act = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // CROWN gives linear bounds: lower_a*x + lower_b <= prelu(x) <= upper_a*x + upper_b
    // Verify at several sample points in [-2, 3]
    for &x in &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
        let y_true = prelu_eval(x, 0.25);
        let y_lower = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let y_upper = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            y_lower <= y_true + 1e-5,
            "CROWN lower({x}) = {y_lower} > true = {y_true}"
        );
        assert!(
            y_upper >= y_true - 1e-5,
            "CROWN upper({x}) = {y_upper} < true = {y_true}"
        );
    }
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_near_point_crossing_guard_is_sound() {
    let slope = -0.5_f32;
    let l = -1e-20_f32;
    let u = 1e-20_f32;
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = prelu_linear_relaxation(l, u, slope);
    assert_eq!(ls, 0.0, "near-point guard should return constant lower");
    assert_eq!(us, 0.0, "near-point guard should return constant upper");
    assert!(
        li.is_finite() && ui.is_finite(),
        "near-point guard must avoid Inf/NaN coefficients"
    );
    for &x in &[l, 0.0, u] {
        let y_true = prelu_eval(x, slope);
        assert!(
            li <= y_true + 1e-12,
            "lower {} > y {} at x={}",
            li,
            y_true,
            x
        );
        assert!(
            ui >= y_true - 1e-12,
            "upper {} < y {} at x={}",
            ui,
            y_true,
            x
        );
    }
}

// ---- Shape validation ----

#[ntest::timeout(5000)]
#[test]
fn test_crown_rejects_mismatched_bounds_size() {
    let layer = PReluLayer::from_scalar(0.25);
    let bounds = LinearBounds::identity(3);
    let pre_act = BoundedTensor::new(
        arr1(&[0.0, 1.0]).into_dyn(), // 2 != 3
        arr1(&[1.0, 2.0]).into_dyn(),
    )
    .unwrap();
    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect_err("size mismatch");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

// ---- propagate_linear returns error ----

#[ntest::timeout(5000)]
#[test]
fn test_propagate_linear_requires_pre_activation() {
    let layer = PReluLayer::from_scalar(0.25);
    let bounds = LinearBounds::identity(2);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("should require pre-activation");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

// ---- requires_pre_activation_bounds ----

#[ntest::timeout(5000)]
#[test]
fn test_requires_pre_activation_bounds_is_true() {
    let layer = PReluLayer::from_scalar(0.25);
    assert!(layer.requires_pre_activation_bounds());
}

// ---- CROWN backward with non-identity incoming bounds ----

/// Tests that CROWN backward correctly handles negative incoming coefficients.
/// With A = [[-1]], the sign-swap logic must use the upper relaxation for the
/// lower bound and vice versa. This exercises the `la < 0` / `ua < 0` branches
/// in the PReLU-specific backward implementation (mod.rs:298-322).
#[ntest::timeout(5000)]
#[test]
fn test_crown_negative_incoming_coefficients() -> Result<()> {
    use ndarray::Array1;

    let layer = PReluLayer::from_scalar(0.25);
    let pre = BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[3.0_f32]).into_dyn())?;

    let neg_bounds = LinearBounds::new(
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_linear_with_bounds(&neg_bounds, &pre)?;

    let l = -2.0_f32;
    let u = 3.0_f32;
    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    // Sample points and verify bounds contain -PReLU(x)
    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = -(prelu_eval(x, 0.25));
        let bound_lo = la.max(0.0) * l + la.min(0.0) * u + lb;
        let bound_hi = ua.max(0.0) * u + ua.min(0.0) * l + ub;
        assert!(
            bound_lo <= y + 1e-5,
            "negative coeff: lower {} > -prelu({}) = {} at x={}",
            bound_lo,
            x,
            y,
            x
        );
        assert!(
            bound_hi >= y - 1e-5,
            "negative coeff: upper {} < -prelu({}) = {} at x={}",
            bound_hi,
            x,
            y,
            x
        );
    }
    Ok(())
}

/// Tests CROWN backward with a 2-neuron, 2-output non-identity coefficient matrix
/// A = [[1, -1], [0.5, 0.5]], verifying that the composed output A @ PReLU(x) is
/// soundly bounded for all (x0, x1) in the pre-activation box. This exercises both
/// positive and negative coefficient branches simultaneously on crossing intervals.
#[ntest::timeout(5000)]
#[test]
fn test_crown_non_identity_bounds() -> Result<()> {
    use ndarray::Array1;

    let layer = PReluLayer::from_scalar(0.25);
    // Neuron 0: crossing [-1, 2], Neuron 1: crossing [-3, 1]
    let pre = BoundedTensor::new(
        arr1(&[-1.0_f32, -3.0]).into_dyn(),
        arr1(&[2.0_f32, 1.0]).into_dyn(),
    )?;

    let a = ndarray::Array2::from_shape_vec((2, 2), vec![1.0_f32, -1.0, 0.5, 0.5])
        .expect("invariant: static 2x2 shape matches 4 elements");
    let bounds = LinearBounds::new(a.clone(), Array1::zeros(2), a, Array1::zeros(2)).unwrap();
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    let l0 = -1.0_f32;
    let u0 = 2.0_f32;
    let l1 = -3.0_f32;
    let u1 = 1.0_f32;
    let lowers = [l0, l1];
    let uppers = [u0, u1];

    for k0 in 0..=10 {
        for k1 in 0..=10 {
            let x0 = l0 + (u0 - l0) * (k0 as f32 / 10.0);
            let x1 = l1 + (u1 - l1) * (k1 as f32 / 10.0);
            let r0 = prelu_eval(x0, 0.25);
            let r1 = prelu_eval(x1, 0.25);

            // True outputs: A @ PReLU(x)
            let y0 = r0 - r1;
            let y1 = 0.5 * r0 + 0.5 * r1;

            for i in 0..2 {
                let y = if i == 0 { y0 } else { y1 };
                let mut lo = result.lower_b[i];
                let mut hi = result.upper_b[i];
                for j in 0..2 {
                    let la = result.lower_a[[i, j]];
                    let ua = result.upper_a[[i, j]];
                    lo += la.max(0.0) * lowers[j] + la.min(0.0) * uppers[j];
                    hi += ua.max(0.0) * uppers[j] + ua.min(0.0) * lowers[j];
                }
                assert!(
                    lo <= y + 1e-3,
                    "output {} lower {} > true {} at ({}, {})",
                    i,
                    lo,
                    y,
                    x0,
                    x1
                );
                assert!(
                    hi >= y - 1e-3,
                    "output {} upper {} < true {} at ({}, {})",
                    i,
                    hi,
                    y,
                    x0,
                    x1
                );
            }
        }
    }
    Ok(())
}

// ---- CROWN backward with negative slope (V-shape) ----

/// Tests CROWN backward with negative slope in the crossing region.
/// PReLU with alpha < 0 creates a V-shape (both branches go up from x=0).
/// This is convex, so chord should be the upper bound.
#[ntest::timeout(5000)]
#[test]
fn test_crown_negative_slope_crossing_soundness() -> Result<()> {
    let layer = PReluLayer::from_scalar(-0.5);
    let l = -2.0_f32;
    let u = 3.0_f32;
    let pre = BoundedTensor::new(arr1(&[l]).into_dyn(), arr1(&[u]).into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    // Verify soundness at sample points
    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = prelu_eval(x, -0.5);
        let lower_bound = la * x + lb;
        let upper_bound = ua * x + ub;
        assert!(
            lower_bound <= y + 1e-5,
            "neg slope: lower {} > prelu({}, -0.5) = {} at x={}",
            lower_bound,
            x,
            y,
            x
        );
        assert!(
            upper_bound >= y - 1e-5,
            "neg slope: upper {} < prelu({}, -0.5) = {} at x={}",
            upper_bound,
            x,
            y,
            x
        );
    }
    Ok(())
}

/// Tests CROWN backward with negative slope and non-identity incoming coefficients.
/// Exercises the sign-swap logic when the underlying function is V-shaped.
#[ntest::timeout(5000)]
#[test]
fn test_crown_negative_slope_negative_coeff_soundness() -> Result<()> {
    use ndarray::Array1;

    let layer = PReluLayer::from_scalar(-0.5);
    let pre = BoundedTensor::new(arr1(&[-2.0_f32]).into_dyn(), arr1(&[3.0_f32]).into_dyn())?;

    let neg_bounds = LinearBounds::new(
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
        ndarray::Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_linear_with_bounds(&neg_bounds, &pre)?;

    let l = -2.0_f32;
    let u = 3.0_f32;
    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = -(prelu_eval(x, -0.5));
        let bound_lo = la.max(0.0) * l + la.min(0.0) * u + lb;
        let bound_hi = ua.max(0.0) * u + ua.min(0.0) * l + ub;
        assert!(
            bound_lo <= y + 1e-3,
            "neg slope neg coeff: lower {} > -prelu({}, -0.5) = {} at x={}",
            bound_lo,
            x,
            y,
            x
        );
        assert!(
            bound_hi >= y - 1e-3,
            "neg slope neg coeff: upper {} < -prelu({}, -0.5) = {} at x={}",
            bound_hi,
            x,
            y,
            x
        );
    }
    Ok(())
}

// ---- Alpha=0 boundary case (#1914) ----

/// PReLU with alpha=0 should behave identically to ReLU.
/// Part of #1914.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_alpha_zero_reduces_to_relu() -> Result<()> {
    let layer = PReluLayer::from_scalar(0.0);

    // Crossing case: [-2, 3]
    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], 0.0, TOL); // ReLU(-2) = 0
    assert_close(out.upper()[[0]], 3.0, TOL); // ReLU(3) = 3

    // All negative: [-3, -1]
    let input = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[-1.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], 0.0, TOL); // ReLU(-3) = 0
    assert_close(out.upper()[[0]], 0.0, TOL); // ReLU(-1) = 0

    // All positive: [1, 3]
    let input = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], 1.0, TOL);
    assert_close(out.upper()[[0]], 3.0, TOL);
    Ok(())
}

/// PReLU CROWN with alpha=0 should produce ReLU-equivalent relaxation.
/// Part of #1914.
#[ntest::timeout(5000)]
#[test]
fn test_crown_alpha_zero_reduces_to_relu() -> Result<()> {
    let layer = PReluLayer::from_scalar(0.0);
    let bounds = LinearBounds::identity(1);
    let pre_act = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // Verify soundness at sample points: bounds must contain ReLU(x)
    for k in 0..=50 {
        let x = -2.0 + 5.0 * (k as f32 / 50.0);
        let y = x.max(0.0); // ReLU
        let lower_bound = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let upper_bound = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lower_bound <= y + 1e-5,
            "alpha=0 CROWN lower {} > relu({}) = {}",
            lower_bound,
            x,
            y
        );
        assert!(
            upper_bound >= y - 1e-5,
            "alpha=0 CROWN upper {} < relu({}) = {}",
            upper_bound,
            x,
            y
        );
    }
    Ok(())
}

// ---- IBP negative slope crossing precision (#1914) ----

/// Regression test for #1914: IBP with negative slope in crossing region
/// must produce tight bounds (lower=0, upper=max(slope*l, u)), not the
/// overly conservative bounds from the old code.
#[ntest::timeout(5000)]
#[test]
fn test_ibp_negative_slope_crossing_tight_bounds() -> Result<()> {
    // slope=-0.5, l=-2, u=3: true range = [0, 3]
    let layer = PReluLayer::from_scalar(-0.5);
    let input = BoundedTensor::new(arr1(&[-2.0]).into_dyn(), arr1(&[3.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], 0.0, TOL);
    assert_close(out.upper()[[0]], 3.0, TOL);

    // slope=-2.0, l=-1, u=0.5: PReLU(-1)=2, PReLU(0)=0, PReLU(0.5)=0.5
    // true range = [0, 2], upper = max(slope*l, u) = max(2, 0.5) = 2
    let layer = PReluLayer::from_scalar(-2.0);
    let input = BoundedTensor::new(arr1(&[-1.0]).into_dyn(), arr1(&[0.5]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], 0.0, TOL);
    assert_close(out.upper()[[0]], 2.0, TOL);

    // slope=-1.0, l=-3, u=2: symmetric V-shape at 0.
    // PReLU(-3)=3, PReLU(0)=0, PReLU(2)=2
    // true range = [0, 3]
    let layer = PReluLayer::from_scalar(-1.0);
    let input = BoundedTensor::new(arr1(&[-3.0]).into_dyn(), arr1(&[2.0]).into_dyn())?;
    let out = layer.propagate_ibp(&input)?;
    assert_close(out.lower()[[0]], 0.0, TOL);
    assert_close(out.upper()[[0]], 3.0, TOL);
    Ok(())
}

// ---- Empty slope rejection (#2865) ----

/// Regression test for #2865: `PReluLayer::new` with empty slope must return
/// an error, not construct a layer that panics on `get_slope(idx % 0)`.
#[test]
fn test_new_empty_slope_is_rejected() {
    let result = PReluLayer::new(arr1(&[]));
    assert!(result.is_err(), "empty slope must be rejected");
    let err = result.unwrap_err();
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
}

// ── CROWN relaxation soundness proptest (#3321) ─────────────────────

/// Reference PReLU in f64, independent of the crate f32 implementation.
fn prelu_f64_reference(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * x
    }
}

proptest::proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(1000) })]

    /// #3321: Verify prelu_linear_relaxation produces strictly sound bounds.
    /// For random intervals, the lower bound must satisfy
    ///   lower_slope * x + lower_intercept <= PReLU(x)  for all x in [l, u]
    /// and the upper bound must satisfy
    ///   upper_slope * x + upper_intercept >= PReLU(x)  for all x in [l, u]
    /// with NO positive tolerance. Evaluated in f64 for mathematical precision.
    ///
    /// Ref: ELU proptest_elu_relaxation_strict_soundness (elu.rs:841).
    #[test]
    fn proptest_prelu_relaxation_strict_soundness(
        l in -10.0f32..10.0,
        width in 0.01f32..20.0,
        alpha in -5.0f32..5.0,
    ) {
        let u = l + width;
        let relax = prelu_linear_relaxation(l, u, alpha);
        let ls = relax.lower_slope;
        let li = relax.lower_intercept;
        let us = relax.upper_slope;
        let ui = relax.upper_intercept;

        // Skip NaN fallback (infinite bounds).
        proptest::prop_assume!(ls.is_finite() && li.is_finite() && us.is_finite() && ui.is_finite());

        let alpha64 = alpha as f64;

        // Dense grid: 200 points, evaluated in f64 for mathematical precision.
        for k in 0..=200 {
            let t = k as f64 / 200.0;
            let x = l as f64 + t * (u as f64 - l as f64);
            let x = x.clamp(l as f64, u as f64);
            let fx = prelu_f64_reference(x, alpha64);

            let lower_val = ls as f64 * x + li as f64;
            proptest::prop_assert!(
                lower_val <= fx,
                "PReLU lower bound UNSOUND at x={}: {} > PReLU({})={}, \
                 interval=[{}, {}], alpha={}, gap={}", x, lower_val, x, fx, l, u, alpha, lower_val - fx
            );

            let upper_val = us as f64 * x + ui as f64;
            proptest::prop_assert!(
                upper_val >= fx,
                "PReLU upper bound UNSOUND at x={}: {} < PReLU({})={}, \
                 interval=[{}, {}], alpha={}, gap={}", x, upper_val, x, fx, l, u, alpha, fx - upper_val
            );
        }
    }
}

// ── Per-channel PReLU with spatial dimensions (regression #4168) ─────

/// Regression test for #4168: IBP with per-channel slopes on [C=2, T=3] input.
/// The old modulo-based slope(idx % C) gave wrong channel assignments for spatial
/// dims; stride-based slope_for_flat(idx, stride=T) with idx/T is correct.
#[ntest::timeout(5000)]
#[test]
fn test_per_channel_ibp_with_spatial_dims_4168() -> Result<()> {
    use ndarray::Array2;

    // 2 channels, 3 time steps: slopes = [0.1, 0.5]
    let layer = PReluLayer::new(arr1(&[0.1, 0.5])).expect("invariant: non-empty slope");

    // Input shape [C=2, T=3] with crossing intervals for each element.
    // Row-major: [c0t0, c0t1, c0t2, c1t0, c1t1, c1t2]
    // Channel 0 (slope=0.1): elements 0,1,2
    // Channel 1 (slope=0.5): elements 3,4,5
    let lower = Array2::from_shape_vec((2, 3), vec![-2.0, -1.0, -3.0, -1.0, -2.0, -0.5])
        .expect("invariant: 2x3 shape matches 6 elements")
        .into_dyn();
    let upper = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 0.5, 3.0, 1.0, 2.0])
        .expect("invariant: 2x3 shape matches 6 elements")
        .into_dyn();
    let input = BoundedTensor::new(lower, upper)?;
    let out = layer.propagate_ibp(&input)?;

    // Verify each element uses the correct per-channel slope.
    let slopes = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5];
    let lowers_flat = [-2.0, -1.0, -3.0, -1.0, -2.0, -0.5];
    let uppers_flat = [1.0, 2.0, 0.5, 3.0, 1.0, 2.0];

    let out_lower = out.lower().as_slice().expect("contiguous");
    let out_upper = out.upper().as_slice().expect("contiguous");

    for i in 0..6 {
        let s = slopes[i];
        let l = lowers_flat[i];
        let u = uppers_flat[i];
        // Sample points within [l, u]
        for k in 0..=10 {
            let x = l + (u - l) * (k as f32 / 10.0);
            let y = prelu_eval(x, s);
            assert!(
                out_lower[i] <= y + 1e-5,
                "elem {i} (slope={s}): lower {} > eval({x}) = {y}",
                out_lower[i]
            );
            assert!(
                out_upper[i] >= y - 1e-5,
                "elem {i} (slope={s}): upper {} < eval({x}) = {y}",
                out_upper[i]
            );
        }
    }
    Ok(())
}

/// Regression test for #4168: CROWN backward with per-channel slopes on [C=2, T=2] input.
/// Verifies that the diagonal relaxation entries use the correct channel slope.
#[ntest::timeout(5000)]
#[test]
fn test_per_channel_crown_with_spatial_dims_4168() -> Result<()> {
    use ndarray::Array2;

    // 2 channels, 2 time steps: slopes = [0.1, 0.8]
    let layer = PReluLayer::new(arr1(&[0.1, 0.8])).expect("invariant: non-empty slope");

    // Shape [C=2, T=2], 4 elements total.
    // Row-major: [c0t0, c0t1, c1t0, c1t1]
    // Channel 0 (slope=0.1): elements 0,1
    // Channel 1 (slope=0.8): elements 2,3
    let lower = Array2::from_shape_vec((2, 2), vec![-2.0, -1.0, -3.0, -1.5])
        .expect("invariant: 2x2 shape matches 4 elements")
        .into_dyn();
    let upper = Array2::from_shape_vec((2, 2), vec![3.0, 2.0, 1.0, 0.5])
        .expect("invariant: 2x2 shape matches 4 elements")
        .into_dyn();
    let pre_act = BoundedTensor::new(lower, upper)?;

    let bounds = LinearBounds::identity(4);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // With identity bounds, diagonal entries encode per-neuron relaxation slopes.
    // Verify soundness at sample points for each neuron.
    let slopes = [0.1, 0.1, 0.8, 0.8];
    let l_vals = [-2.0_f32, -1.0, -3.0, -1.5];
    let u_vals = [3.0_f32, 2.0, 1.0, 0.5];

    for i in 0..4 {
        let la = result.lower_a[[i, i]];
        let lb = result.lower_b[i];
        let ua = result.upper_a[[i, i]];
        let ub = result.upper_b[i];
        let s = slopes[i];

        for k in 0..=20 {
            let x = l_vals[i] + (u_vals[i] - l_vals[i]) * (k as f32 / 20.0);
            let y = prelu_eval(x, s);
            let y_lower = la * x + lb;
            let y_upper = ua * x + ub;
            assert!(
                y_lower <= y + 1e-5,
                "CROWN neuron {i} (slope={s}): lower {y_lower} > prelu({x}) = {y}"
            );
            assert!(
                y_upper >= y - 1e-5,
                "CROWN neuron {i} (slope={s}): upper {y_upper} < prelu({x}) = {y}"
            );
        }
    }
    Ok(())
}

/// Regression test for #4168: stride mismatch when total elements not divisible by slope count.
#[ntest::timeout(5000)]
#[test]
fn test_per_channel_stride_mismatch_returns_error() {
    let layer = PReluLayer::new(arr1(&[0.1, 0.2, 0.3])).expect("invariant: non-empty slope");
    // 5 elements can't divide into 3 channels evenly.
    let input = BoundedTensor::new(
        arr1(&[-1.0, -1.0, -1.0, -1.0, -1.0]).into_dyn(),
        arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]).into_dyn(),
    )
    .unwrap();
    let err = layer
        .propagate_ibp(&input)
        .expect_err("5 elements / 3 channels should fail");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

proptest::proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    /// Property regression for #4168: per-channel slopes must stay constant
    /// across spatial positions within a channel for multi-element `[C, T]`
    /// inputs. The old modulo lookup (`idx % C`) violated this when `T > 1`.
    #[test]
    fn proptest_prelu_ibp_spatial_per_channel_sound_4168(
        channels in 2usize..=8,
        spatial in 2usize..=16,
        slope_base in 0.05f32..0.9f32,
        slope_step in 0.01f32..0.15f32,
    ) {
        let slopes: Vec<f32> = (0..channels)
            .map(|channel| slope_base + slope_step * channel as f32)
            .collect();
        let layer = PReluLayer::new(Array1::from_vec(slopes.clone()))
            .expect("invariant: generated slope vector is non-empty");

        let total = channels * spatial;
        let lower_flat: Vec<f32> = (0..total)
            .map(|flat_idx| {
                -(1.0 + (flat_idx % spatial) as f32 * 0.25 + (flat_idx / spatial) as f32 * 0.1)
            })
            .collect();
        let upper_flat: Vec<f32> = (0..total)
            .map(|flat_idx| 0.25 + (flat_idx % spatial) as f32 * 0.1)
            .collect();

        let input = BoundedTensor::new(
            ndarray::Array2::from_shape_vec((channels, spatial), lower_flat.clone())
                .expect("invariant: generated lower shape matches channels * spatial")
                .into_dyn(),
            ndarray::Array2::from_shape_vec((channels, spatial), upper_flat.clone())
                .expect("invariant: generated upper shape matches channels * spatial")
                .into_dyn(),
        )
        .expect("invariant: generated lower <= upper elementwise");

        let output = layer
            .propagate_ibp(&input)
            .expect("per-channel spatial inputs should propagate successfully");
        let lower = output.lower().as_slice().expect("invariant: dense output is contiguous");
        let upper = output.upper().as_slice().expect("invariant: dense output is contiguous");

        for flat_idx in 0..total {
            let channel = flat_idx / spatial;
            let slope = slopes[channel];
            let l = lower_flat[flat_idx];
            let u = upper_flat[flat_idx];
            let y_l = prelu_eval(l, slope);
            let y_u = prelu_eval(u, slope);

            proptest::prop_assert!(
                lower[flat_idx] <= y_l + TOL,
                "flat_idx={flat_idx}, channel={channel}, slope={slope}: \
                 lower {} > PReLU(l={l})={y_l}",
                lower[flat_idx]
            );
            proptest::prop_assert!(
                upper[flat_idx] >= y_l - TOL,
                "flat_idx={flat_idx}, channel={channel}, slope={slope}: \
                 upper {} < PReLU(l={l})={y_l}",
                upper[flat_idx]
            );
            proptest::prop_assert!(
                lower[flat_idx] <= y_u + TOL,
                "flat_idx={flat_idx}, channel={channel}, slope={slope}: \
                 lower {} > PReLU(u={u})={y_u}",
                lower[flat_idx]
            );
            proptest::prop_assert!(
                upper[flat_idx] >= y_u - TOL,
                "flat_idx={flat_idx}, channel={channel}, slope={slope}: \
                 upper {} < PReLU(u={u})={y_u}",
                upper[flat_idx]
            );
            proptest::prop_assert!(
                lower[flat_idx] <= 0.0f32 + TOL,
                "flat_idx={flat_idx}, channel={channel}, slope={slope}: \
                 lower {} > PReLU(0)=0",
                lower[flat_idx]
            );
            proptest::prop_assert!(
                upper[flat_idx] >= 0.0f32 - TOL,
                "flat_idx={flat_idx}, channel={channel}, slope={slope}: \
                 upper {} < PReLU(0)=0",
                upper[flat_idx]
            );
        }
    }
}
