// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for ReLU layer — relaxation, IBP, CROWN, α-CROWN.

use super::*;
use ndarray::{array, Array1, ArrayD, IxDyn};
use proptest::prelude::*;

/// Reference ReLU in f64, independent of the f32 implementation.
fn relu_f64_reference(x: f64) -> f64 {
    x.max(0.0)
}

fn assert_close(actual: f32, expected: f32, tol: f32, label: impl std::fmt::Display) {
    assert!(
        (actual - expected).abs() < tol,
        "{label}: expected {expected}, got {actual}"
    );
}

// ===== relu_linear_relaxation tests =====

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_positive_region() {
    // l >= 0: identity (slope=1, intercept=0)
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(1.0, 5.0);
    assert_close(ls, 1.0, 1e-6, "positive lower slope");
    assert_close(li, 0.0, 1e-6, "positive lower intercept");
    assert_close(us, 1.0, 1e-6, "positive upper slope");
    assert_close(ui, 0.0, 1e-6, "positive upper intercept");
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_negative_region() {
    // u <= 0: zero (slope=0, intercept=0)
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(-5.0, -1.0);
    assert_close(ls, 0.0, 1e-6, "negative lower slope");
    assert_close(li, 0.0, 1e-6, "negative lower intercept");
    assert_close(us, 0.0, 1e-6, "negative upper slope");
    assert_close(ui, 0.0, 1e-6, "negative upper intercept");
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_crossing_region() {
    // l < 0 < u: crossing case
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(-2.0, 3.0);
    // α = 1.0 since u=3 > -l=2
    assert!(
        (ls - 1.0).abs() < 1e-6,
        "lower slope should be 1.0 (α), got {}",
        ls
    );
    assert!(li.abs() < 1e-6, "lower intercept should be 0");
    // λ = u/(u-l) = 3/5 = 0.6
    assert!(
        (us - 0.6).abs() < 1e-5,
        "upper slope should be 0.6, got {}",
        us
    );
    // upper intercept = -λ*l = -0.6*(-2) = 1.2
    assert!(
        (ui - 1.2).abs() < 1e-5,
        "upper intercept should be 1.2, got {}",
        ui
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_crossing_alpha_zero() {
    // When u < -l, α = 0
    let LinearRelaxation {
        lower_slope: ls, ..
    } = relu_linear_relaxation(-5.0, 1.0);
    assert_close(ls, 0.0, 1e-6, "α should be 0 when u=1 < -l=5");
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_soundness_crossing() {
    // Verify the relaxation bounds contain ReLU for all x in [l, u]
    let l = -3.0_f32;
    let u = 2.0_f32;
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(l, u);

    for k in 0..100 {
        let t = k as f32 / 99.0;
        let x = (l + t * (u - l)).clamp(l, u);
        let relu_x = x.max(0.0);

        let lower_bound = ls * x + li;
        let upper_bound = us * x + ui;
        assert!(
            lower_bound <= relu_x + 1e-5,
            "lower {} > relu {} at x={}",
            lower_bound,
            relu_x,
            x
        );
        assert!(
            upper_bound >= relu_x - 1e-5,
            "upper {} < relu {} at x={}",
            upper_bound,
            relu_x,
            x
        );
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_nan_bounds() {
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(f32::NAN, 1.0);
    assert_close(ls, 0.0, 1e-6, "NaN lower slope");
    assert_eq!(li, f32::NEG_INFINITY, "NaN → -inf intercept");
    assert_close(us, 0.0, 1e-6, "NaN upper slope");
    assert_eq!(ui, f32::INFINITY, "NaN → +inf intercept");
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_both_infinite() {
    let LinearRelaxation {
        lower_slope: ls,
        upper_slope: us,
        upper_intercept: ui,
        ..
    } = relu_linear_relaxation(f32::NEG_INFINITY, f32::INFINITY);
    assert_close(ls, 0.0, 1e-6, "both infinite lower slope");
    assert_close(us, 0.0, 1e-6, "both infinite upper slope");
    assert_eq!(ui, f32::INFINITY, "both infinite → +inf upper intercept");
}

// ===== IBP tests =====

#[ntest::timeout(5000)]
#[test]
fn test_ibp_positive() -> Result<()> {
    let layer = ReLULayer::new();
    let lower = array![1.0_f32, 2.0, 3.0].into_dyn();
    let upper = array![4.0_f32, 5.0, 6.0].into_dyn();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    // All positive → ReLU is identity
    assert_close(output.lower()[0], 1.0, 1e-6, "positive lower[0]");
    assert_close(output.upper()[2], 6.0, 1e-6, "positive upper[2]");
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_negative() -> Result<()> {
    let layer = ReLULayer::new();
    let lower = array![-5.0_f32, -3.0].into_dyn();
    let upper = array![-1.0_f32, -0.5].into_dyn();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    // All negative → ReLU outputs 0
    assert_close(output.lower()[0], 0.0, 1e-6, "negative lower[0]");
    assert_close(output.upper()[0], 0.0, 1e-6, "negative upper[0]");
    assert_close(output.lower()[1], 0.0, 1e-6, "negative lower[1]");
    assert_close(output.upper()[1], 0.0, 1e-6, "negative upper[1]");
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_ibp_crossing() -> Result<()> {
    let layer = ReLULayer::new();
    let lower = array![-3.0_f32, -1.0].into_dyn();
    let upper = array![2.0_f32, 5.0].into_dyn();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    // Crossing: lower clamped to 0, upper unchanged
    assert_close(output.lower()[0], 0.0, 1e-6, "crossing lower[0]");
    assert_close(output.upper()[0], 2.0, 1e-6, "crossing upper[0]");
    assert_close(output.lower()[1], 0.0, 1e-6, "crossing lower[1]");
    assert_close(output.upper()[1], 5.0, 1e-6, "crossing upper[1]");
    Ok(())
}

// ===== CROWN backward tests =====

#[ntest::timeout(5000)]
#[test]
fn test_crown_positive_preact() -> Result<()> {
    // Pre-activation [2, 5] → ReLU is identity → coefficients unchanged
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(array![2.0_f32].into_dyn(), array![5.0_f32].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    // Identity pass-through: A should be unchanged
    assert_close(result.lower_a[[0, 0]], 1.0, 1e-5, "positive lower_a[0,0]");
    assert_close(result.upper_a[[0, 0]], 1.0, 1e-5, "positive upper_a[0,0]");
    assert_close(result.lower_b[0], 0.0, 1e-5, "positive lower_b[0]");
    assert_close(result.upper_b[0], 0.0, 1e-5, "positive upper_b[0]");
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_negative_preact() -> Result<()> {
    // Pre-activation [-5, -1] → ReLU is always 0 → zero coefficients
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(array![-5.0_f32].into_dyn(), array![-1.0_f32].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    assert_close(result.lower_a[[0, 0]], 0.0, 1e-5, "negative lower_a[0,0]");
    assert_close(result.upper_a[[0, 0]], 0.0, 1e-5, "negative upper_a[0,0]");
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_crown_crossing_preact_soundness() -> Result<()> {
    // Pre-activation [-2, 3] → crossing → verify concretized bounds are sound
    let layer = ReLULayer::new();
    let l = -2.0_f32;
    let u = 3.0_f32;
    let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    // Concretize: lower_bound = max(la, 0)*l + min(la, 0)*u + lb
    //             upper_bound = max(ua, 0)*u + min(ua, 0)*l + ub
    let la = result.lower_a[[0, 0]];
    let lb = result.lower_b[0];
    let ua = result.upper_a[[0, 0]];
    let ub = result.upper_b[0];

    let conc_lower = la.max(0.0) * l + la.min(0.0) * u + lb;
    let conc_upper = ua.max(0.0) * u + ua.min(0.0) * l + ub;

    // Bounds must contain all ReLU(x) for x in [l, u]
    // min ReLU = 0 (at x=l=-2), max ReLU = 3 (at x=u=3)
    assert!(
        conc_lower <= 0.0 + 1e-5,
        "lower {} should be <= 0 (min relu)",
        conc_lower
    );
    assert!(
        conc_upper >= 3.0 - 1e-5,
        "upper {} should be >= 3 (max relu)",
        conc_upper
    );
    Ok(())
}

// ===== α-CROWN tests =====

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_custom_alpha() -> Result<()> {
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(array![-2.0_f32].into_dyn(), array![3.0_f32].into_dyn())?;
    let bounds = LinearBounds::identity(1);

    // α=0.5 for the crossing neuron
    let alpha = array![0.5_f32];
    let (result, gradient, _gradient_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, None)?;

    // Lower slope should be α=0.5 (since identity bounds have la=1.0 > 0)
    assert_close(
        result.lower_a[[0, 0]],
        0.5,
        1e-5,
        "custom alpha lower_a[0,0]",
    );

    // Gradient ∂(lower_bound)/∂α: for crossing neuron with la=1.0 > 0 and l=-2.0,
    // gradient = la * l = 1.0 * (-2.0) = -2.0. The negative sign is correct:
    // increasing alpha makes the lower bound more negative (worse), so the optimizer
    // (which negates before Adam update) will decrease alpha. Fix: #3294.
    assert_close(gradient[0], -2.0, 1e-5, "custom alpha gradient[0]");
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_bound_only_is_byte_identical() -> Result<()> {
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(
        array![-2.0_f32, 1.0, -4.0].into_dyn(),
        array![3.0_f32, 5.0, -1.0].into_dyn(),
    )?;
    let bounds = LinearBounds::from_parts_unchecked(
        array![[1.25_f32, -0.5, 2.0], [-3.0, 0.25, -0.75]],
        array![0.125_f32, -0.25],
        array![[-0.75_f32, 2.0, 0.5], [1.5, -1.0, 0.25]],
        array![0.5_f32, -0.375],
    );
    let alpha = array![0.25_f32, 0.75, 0.5];
    let alpha_upper = array![0.6_f32, 0.4, 0.1];

    let (with_grad, _, _) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, Some(&alpha_upper))?;
    let bound_only =
        layer.propagate_linear_with_alpha_bound_only(&bounds, &pre, &alpha, Some(&alpha_upper))?;

    assert_eq!(bound_only.lower_a(), with_grad.lower_a());
    assert_eq!(bound_only.lower_b(), with_grad.lower_b());
    assert_eq!(bound_only.upper_a(), with_grad.upper_a());
    assert_eq!(bound_only.upper_b(), with_grad.upper_b());
    assert_eq!(bound_only.lower_a_err(), with_grad.lower_a_err());
    assert_eq!(bound_only.upper_a_err(), with_grad.upper_a_err());
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_stable_neuron_ignores_alpha() -> Result<()> {
    let layer = ReLULayer::new();
    // Positive pre-activation: always active, ignores α
    let pre = BoundedTensor::new(array![1.0_f32].into_dyn(), array![5.0_f32].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let alpha = array![0.0_f32]; // would make coefficient 0 if used

    let (result, gradient, _gradient_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, None)?;
    // Stable positive neuron: should use effective_alpha=1.0 regardless
    assert_close(
        result.lower_a[[0, 0]],
        1.0,
        1e-5,
        "stable positive lower_a[0,0]",
    );
    assert_close(gradient[0], 0.0, 1e-5, "stable positive gradient[0]");
    Ok(())
}

// ===== Error path tests =====

#[ntest::timeout(5000)]
#[test]
fn test_propagate_linear_requires_preact() {
    let layer = ReLULayer::new();
    let bounds = LinearBounds::identity(1);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("requires pre-activation");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec for missing pre-activation bounds, got {err:?}"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_shape_mismatch() {
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(ArrayD::zeros(IxDyn(&[3])), ArrayD::ones(IxDyn(&[3]))).unwrap();
    let bounds = LinearBounds::identity(3);
    let wrong_alpha = Array1::<f32>::zeros(2); // should be 3
    let err = layer
        .propagate_linear_with_alpha(&bounds, &pre, &wrong_alpha, None)
        .expect_err("alpha size mismatch");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch for wrong alpha size, got {err:?}"
    );
}

// ===== Multi-neuron CROWN backward =====

#[ntest::timeout(5000)]
#[test]
fn test_crown_multi_neuron_mixed_regions() -> Result<()> {
    // 3 neurons: [2,5] (positive), [-5,-1] (negative), [-2,3] (crossing)
    // Tests that each neuron gets the correct per-element relaxation.
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(
        array![2.0_f32, -5.0, -2.0].into_dyn(),
        array![5.0_f32, -1.0, 3.0].into_dyn(),
    )?;
    let bounds = LinearBounds::identity(3);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    // Neuron 0 (positive): identity → coefficient 1, bias 0
    assert_close(result.lower_a[[0, 0]], 1.0, 1e-5, "n0 lower slope");
    assert_close(result.upper_a[[0, 0]], 1.0, 1e-5, "n0 upper slope");
    assert_close(result.lower_b[0], 0.0, 1e-5, "n0 lower bias");
    assert_close(result.upper_b[0], 0.0, 1e-5, "n0 upper bias");

    // Neuron 1 (negative): zero → coefficient 0, bias 0
    assert_close(result.lower_a[[1, 1]], 0.0, 1e-5, "n1 lower slope");
    assert_close(result.upper_a[[1, 1]], 0.0, 1e-5, "n1 upper slope");
    assert_close(result.lower_b[1], 0.0, 1e-5, "n1 lower bias");
    assert_close(result.upper_b[1], 0.0, 1e-5, "n1 upper bias");

    // Neuron 2 (crossing [-2,3]): upper slope = 3/(3-(-2)) = 0.6
    // lower: u=3 > -l=2 → alpha=1 → slope=1, intercept=0
    let expected_upper_slope = 3.0 / 5.0;
    assert_close(
        result.upper_a[[2, 2]],
        expected_upper_slope,
        1e-5,
        "n2 upper slope",
    );
    assert_close(
        result.lower_a[[2, 2]],
        1.0,
        1e-5,
        "n2 lower slope (alpha=1)",
    );

    // Cross-terms should all be zero (diagonal structure for identity bounds)
    assert_close(result.lower_a[[0, 1]], 0.0, 1e-5, "cross-term lower_a[0,1]");
    assert_close(result.lower_a[[0, 2]], 0.0, 1e-5, "cross-term lower_a[0,2]");
    assert_close(result.lower_a[[1, 0]], 0.0, 1e-5, "cross-term lower_a[1,0]");
    assert_close(result.lower_a[[2, 0]], 0.0, 1e-5, "cross-term lower_a[2,0]");
    Ok(())
}

// ===== Negative coefficients in incoming bounds =====

#[ntest::timeout(5000)]
#[test]
fn test_crown_negative_incoming_coefficients() -> Result<()> {
    // Incoming bounds with A = [[-1]], testing the coefficient-sign-dependent swap.
    // For ReLU with crossing pre-activation, negative A should swap lower/upper relaxations.
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(array![-2.0_f32].into_dyn(), array![3.0_f32].into_dyn())?;

    // Negative coefficient: A = [[-1]]
    let neg_bounds = LinearBounds::new(
        Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
        Array2::from_elem((1, 1), -1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();
    let neg_result = layer.propagate_linear_with_bounds(&neg_bounds, &pre)?;

    // With A=-1 and crossing pre-activation:
    // For lower bound: A<0 so it uses upper relaxation's slope, negated direction
    // For upper bound: A<0 so it uses lower relaxation's slope, negated direction
    // The new coefficient is A * relu_slope, and bias accumulates accordingly.
    // Verify soundness: concretize and check
    let l = -2.0_f32;
    let u = 3.0_f32;
    let la = neg_result.lower_a[[0, 0]];
    let lb = neg_result.lower_b[0];
    let ua = neg_result.upper_a[[0, 0]];
    let ub = neg_result.upper_b[0];

    // Sample points and verify bounds contain -ReLU(x)
    for k in 0..=20 {
        let x = l + (u - l) * (k as f32 / 20.0);
        let y = -(x.max(0.0)); // -ReLU(x)
        let bound_lo = la.max(0.0) * l + la.min(0.0) * u + lb;
        let bound_hi = ua.max(0.0) * u + ua.min(0.0) * l + ub;
        assert!(
            bound_lo <= y + 1e-5,
            "negative coeff: lower {} > -relu({}) = {} at x={}",
            bound_lo,
            x,
            y,
            x
        );
        assert!(
            bound_hi >= y - 1e-5,
            "negative coeff: upper {} < -relu({}) = {} at x={}",
            bound_hi,
            x,
            y,
            x
        );
    }
    Ok(())
}

// ===== Edge cases for relaxation =====

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_lower_boundary_zero() {
    // l = 0: on the boundary between positive and crossing
    // Should treat as positive region (l >= 0)
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(0.0, 5.0);
    assert_close(ls, 1.0, 1e-6, "l=0 lower slope");
    assert_close(li, 0.0, 1e-6, "l=0 lower intercept");
    assert_close(us, 1.0, 1e-6, "l=0 upper slope");
    assert_close(ui, 0.0, 1e-6, "l=0 upper intercept");
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_upper_boundary_zero() {
    // u = 0: on the boundary between negative and crossing
    // Should treat as negative region (u <= 0)
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(-5.0, 0.0);
    assert_close(ls, 0.0, 1e-6, "u=0 lower slope");
    assert_close(li, 0.0, 1e-6, "u=0 lower intercept");
    assert_close(us, 0.0, 1e-6, "u=0 upper slope");
    assert_close(ui, 0.0, 1e-6, "u=0 upper intercept");
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_point_interval_positive() {
    // l = u = 3: point interval in positive region
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(3.0, 3.0);
    assert_close(ls, 1.0, 1e-6, "point positive lower slope");
    assert_close(li, 0.0, 1e-6, "point positive lower intercept");
    assert_close(us, 1.0, 1e-6, "point positive upper slope");
    assert_close(ui, 0.0, 1e-6, "point positive upper intercept");
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_point_interval_zero() {
    // l = u = 0: point interval at the kink
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(0.0, 0.0);
    // At x=0, ReLU(x)=0. With l=u=0, falls into l>=0 case → identity
    assert_close(ls, 1.0, 1e-6, "point zero lower slope");
    assert_close(li, 0.0, 1e-6, "point zero lower intercept");
    assert_close(us, 1.0, 1e-6, "point zero upper slope");
    assert_close(ui, 0.0, 1e-6, "point zero upper intercept");
}

// ===== RELU_RELAX_MIN_WIDTH guard tests (#2382) =====

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_near_zero_width_below_guard_2382() {
    // u - l = 2e-20, far below RELU_RELAX_MIN_WIDTH = 1e-8.
    // Without the guard, u / (u - l) could produce Inf in f32.
    let l = -1e-20_f32;
    let u = 1e-20_f32;
    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(l, u);

    // All outputs must be finite (the guard prevents division by near-zero)
    assert!(ls.is_finite(), "lower slope must be finite, got {ls}");
    assert!(li.is_finite(), "lower intercept must be finite, got {li}");
    assert!(us.is_finite(), "upper slope must be finite, got {us}");
    assert!(ui.is_finite(), "upper intercept must be finite, got {ui}");

    // Upper relaxation soundness: λ*x + intercept >= ReLU(x) for all x in [l, u].
    // At x = u: λ*u + intercept >= u (since u > 0, ReLU(u) = u). RELATIVE tolerance — the
    // old absolute `u - 1e-12` was trivially true for u=1e-20 (it equals -1e-12), masking
    // the false-proof chord the `max(u−l,1e-8)` floor produced. The exact-width chord encloses.
    let upper_at_u = us * u + ui;
    assert!(
        upper_at_u >= u * (1.0 - 1e-4),
        "upper bound at u={u}: {upper_at_u} < ReLU({u}) — upper chord does not enclose"
    );
    // At x = l: λ*l + intercept >= 0 (since l < 0, ReLU(l) = 0)
    let upper_at_l = us * l + ui;
    assert!(
        upper_at_l >= -1e-12,
        "upper bound at l={l}: {upper_at_l} < ReLU({l})=0"
    );
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_exact_guard_boundary_2382() {
    // u - l == RELU_RELAX_MIN_WIDTH exactly.
    // The guard should not fire (width equals threshold), but result must still be finite.
    let half = RELU_RELAX_MIN_WIDTH / 2.0;
    let l = -half;
    let u = half;
    // Verify we constructed the boundary correctly
    assert_close(
        u - l,
        RELU_RELAX_MIN_WIDTH,
        1e-15,
        "guard-boundary width should equal RELU_RELAX_MIN_WIDTH",
    );

    let LinearRelaxation {
        lower_slope: ls,
        lower_intercept: li,
        upper_slope: us,
        upper_intercept: ui,
    } = relu_linear_relaxation(l, u);

    assert!(ls.is_finite(), "lower slope must be finite, got {ls}");
    assert!(li.is_finite(), "lower intercept must be finite, got {li}");
    assert!(us.is_finite(), "upper slope must be finite, got {us}");
    assert!(ui.is_finite(), "upper intercept must be finite, got {ui}");

    // At the exact boundary u−l == RELU_RELAX_MIN_WIDTH the exact width equals the old floor,
    // so the (now exact-width) upper slope is unchanged: u/(u−l) = half/1e-8 = 0.5.
    let expected_lambda = u / RELU_RELAX_MIN_WIDTH;
    assert_close(us, expected_lambda, 1e-5, "guard-boundary upper slope");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    /// ENCLOSURE (the false-proof class the 2026-06-27 audit flagged + the suite missed):
    /// the ReLU upper chord must satisfy chord(x) >= ReLU(x) for EVERY x in [l,u], including
    /// crossing intervals narrower than the former 1e-8 floor. Runtime form of ny-cert
    /// `FloatAdequacy.lean::interval_outward_contains`. Default-reject.
    #[test]
    fn proptest_relu_upper_chord_encloses(
        l in -1.0e3_f32..=-1.0e-30_f32,
        u in 1.0e-30_f32..=1.0e3_f32,
    ) {
        let r = relu_linear_relaxation(l, u);
        prop_assert!(r.upper_slope.is_finite() && r.upper_intercept.is_finite());
        for k in 0..=32 {
            let x = l as f64 + (k as f64 / 32.0) * (u as f64 - l as f64);
            let relu = x.max(0.0);
            let chord = (r.upper_slope as f64) * x + (r.upper_intercept as f64);
            let slack = relu.abs() * 1e-4 + 1e-30;
            prop_assert!(
                chord >= relu - slack,
                "upper chord {chord} < ReLU({x})={relu} on [l={l}, u={u}] — false-proof"
            );
        }
    }
}

#[ntest::timeout(5000)]
#[test]
fn test_relaxation_large_crossing_width_stays_finite() {
    let l = -f32::MAX;
    let u = f32::MAX;
    let LinearRelaxation {
        upper_slope: us,
        upper_intercept: ui,
        ..
    } = relu_linear_relaxation(l, u);

    assert!(
        us.is_finite() && us > 0.0,
        "upper slope must stay positive, got {us}"
    );
    assert!(
        ui.is_finite() && ui > 0.0,
        "upper intercept must stay positive, got {ui}"
    );

    let upper_at_u = us * u + ui;
    assert!(
        upper_at_u.is_infinite() || upper_at_u >= u,
        "upper bound at u={u} should dominate ReLU(u), got {upper_at_u}"
    );
}

// ===== CROWN crossing soundness with interior sampling =====

#[ntest::timeout(5000)]
#[test]
fn test_crown_crossing_soundness_grid() -> Result<()> {
    // Test CROWN soundness for multiple crossing intervals with dense sampling.
    let layer = ReLULayer::new();
    let test_ranges: &[(f32, f32)] = &[
        (-1.0, 1.0),
        (-5.0, 1.0),    // alpha=0 (u < -l)
        (-1.0, 5.0),    // alpha=1 (u > -l)
        (-0.01, 100.0), // very asymmetric positive-heavy
        (-100.0, 0.01), // very asymmetric negative-heavy
    ];

    for &(l, u) in test_ranges {
        let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn())?;
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;
        let la = result.lower_a[[0, 0]];
        let lb = result.lower_b[0];
        let ua = result.upper_a[[0, 0]];
        let ub = result.upper_b[0];

        // Sample 51 points
        for k in 0..=50 {
            let x = l + (u - l) * (k as f32 / 50.0);
            let y = x.max(0.0); // ReLU(x)
            let lower_bound = la * x + lb;
            let upper_bound = ua * x + ub;
            assert!(
                lower_bound <= y + 1e-5,
                "lower {} > relu({}) = {} for [{}, {}]",
                lower_bound,
                x,
                y,
                l,
                u
            );
            assert!(
                upper_bound >= y - 1e-5,
                "upper {} < relu({}) = {} for [{}, {}]",
                upper_bound,
                x,
                y,
                l,
                u
            );
        }
    }
    Ok(())
}

// ===== Batched CROWN backward =====

#[ntest::timeout(5000)]
#[test]
fn test_batched_crown_mixed_regions() -> Result<()> {
    // 2 neurons: positive [1,3] and negative [-3,-1]
    // Batched bounds with identity
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(
        array![1.0_f32, -3.0].into_dyn(),
        array![3.0_f32, -1.0].into_dyn(),
    )?;
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        vec![2],
        vec![2],
    );
    let result = layer.propagate_linear_batched_with_bounds(&bounds, &pre)?;

    // Neuron 0 (positive): identity pass-through
    assert_close(
        result.lower_a[[0, 0]],
        1.0,
        1e-5,
        "batched positive lower_a[0,0]",
    );
    assert_close(
        result.upper_a[[0, 0]],
        1.0,
        1e-5,
        "batched positive upper_a[0,0]",
    );

    // Neuron 1 (negative): zero
    assert_close(
        result.lower_a[[1, 1]],
        0.0,
        1e-5,
        "batched negative lower_a[1,1]",
    );
    assert_close(
        result.upper_a[[1, 1]],
        0.0,
        1e-5,
        "batched negative upper_a[1,1]",
    );
    Ok(())
}

// ===== alpha-CROWN stable negative neuron =====

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_stable_negative_ignores_alpha() -> Result<()> {
    // Stable negative neuron: u <= 0. Alpha should be ignored; effective_alpha=0.
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(array![-5.0_f32].into_dyn(), array![-1.0_f32].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let alpha = array![0.8_f32]; // should be ignored
    let (result, grad, _grad_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, None)?;

    // Negative region: all zero regardless of alpha
    assert_close(
        result.lower_a[[0, 0]],
        0.0,
        1e-5,
        "stable negative lower_a[0,0]",
    );
    assert_close(
        result.upper_a[[0, 0]],
        0.0,
        1e-5,
        "stable negative upper_a[0,0]",
    );
    // Gradient should be 0 for stable neurons (no useful gradient info)
    assert_close(grad[0], 0.0, 1e-5, "stable negative gradient[0]");
    Ok(())
}

// ===== Non-identity incoming bounds with multiple outputs =====

#[ntest::timeout(5000)]
#[test]
fn test_crown_non_identity_bounds() -> Result<()> {
    // 2 neurons, incoming bounds: A = [[1, -1], [0.5, 0.5]]
    // Pre-activation: neuron 0 crossing [-1, 2], neuron 1 positive [1, 3]
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(
        array![-1.0_f32, 1.0].into_dyn(),
        array![2.0_f32, 3.0].into_dyn(),
    )?;

    let a = Array2::from_shape_vec((2, 2), vec![1.0_f32, -1.0, 0.5, 0.5]).unwrap();
    let bounds = LinearBounds::new(a.clone(), Array1::zeros(2), a, Array1::zeros(2)).unwrap();
    let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

    // Verify soundness by sampling
    let l0 = -1.0_f32;
    let u0 = 2.0_f32;
    let l1 = 1.0_f32;
    let u1 = 3.0_f32;

    for k0 in 0..=10 {
        for k1 in 0..=10 {
            let x0 = l0 + (u0 - l0) * (k0 as f32 / 10.0);
            let x1 = l1 + (u1 - l1) * (k1 as f32 / 10.0);
            let r0 = x0.max(0.0);
            let r1 = x1.max(0.0); // = x1 since x1 >= 1

            // True outputs of A @ relu(x)
            let y0 = r0 - r1;
            let y1 = 0.5 * r0 + 0.5 * r1;

            // Concretize CROWN bounds
            for i in 0..2 {
                let y = if i == 0 { y0 } else { y1 };
                let mut lo = result.lower_b[i];
                let mut hi = result.upper_b[i];
                for j in 0..2 {
                    let la = result.lower_a[[i, j]];
                    let ua = result.upper_a[[i, j]];
                    lo += la.max(0.0) * [l0, l1][j] + la.min(0.0) * [u0, u1][j];
                    hi += ua.max(0.0) * [u0, u1][j] + ua.min(0.0) * [l0, l1][j];
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

// ===== SDP-CROWN / alpha-CROWN near-zero-width crossing tests =====

/// Verify SDP-CROWN backward pass stays finite on the smallest subnormal crossing
/// interval. Re: #2410 — the reported `u - l -> 0` failure mode cannot happen on
/// this branch because `l < 0 < u` makes the denominator an opposite-signed
/// subtraction.
#[ntest::timeout(5000)]
#[test]
fn test_sdp_crown_near_zero_width_crossing_safe() -> Result<()> {
    let layer = ReLULayer::new();
    let min_subnormal = f32::from_bits(1);
    let l = -min_subnormal;
    let u = min_subnormal;
    let width = u - l;
    assert_eq!(
        width,
        f32::from_bits(2),
        "crossing subtraction should stay positive even at the subnormal floor"
    );
    let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let x_hat = array![0.0_f32];
    let rho = min_subnormal;

    let result = layer.propagate_linear_with_bounds_sdp(&bounds, &pre, &x_hat, rho)?;

    assert!(
        (0.5..=1.0).contains(&result.upper_a[[0, 0]]),
        "crossing slope should stay in [0.5, 1.0], got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        result.upper_b[0].is_finite(),
        "SDP-CROWN upper_b must be finite for crossing neuron: got {}",
        result.upper_b[0]
    );
    assert!(
        result.lower_b[0].is_finite(),
        "SDP-CROWN lower_b must be finite for crossing neuron: got {}",
        result.lower_b[0]
    );
    // Soundness: sample points in [l, u] and verify affine bounds contain ReLU(x).
    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = x.max(0.0);
        let lower_bound = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let upper_bound = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lower_bound <= y + 1e-24,
            "SDP-CROWN lower {} > relu({})={} for [{}, {}]",
            lower_bound,
            x,
            y,
            l,
            u
        );
        assert!(
            upper_bound >= y - 1e-24,
            "SDP-CROWN upper {} < relu({})={} for [{}, {}]",
            upper_bound,
            x,
            y,
            l,
            u
        );
    }
    Ok(())
}

/// Post-#2977: NaN in x_hat propagates to LinearBounds bias. The
/// new_or_conservative firewall catches the NaN and falls back to
/// conservative bounds (A=0, b=+/-Inf) instead of returning Err.
#[ntest::timeout(5000)]
#[test]
fn test_sdp_crown_rho_zero_nan_xhat_keeps_nan_offset() -> Result<()> {
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(
        array![-1.0_f32, 0.5].into_dyn(),
        array![2.0_f32, 3.0].into_dyn(),
    )?;
    let bounds = LinearBounds::identity(2);
    let x_hat = array![f32::NAN, 0.0_f32];

    let result = layer.propagate_linear_with_bounds_sdp(&bounds, &pre, &x_hat, 0.0)?;
    // new_or_conservative falls back to conservative bounds on NaN bias
    assert_eq!(
        result.lower_b()[0],
        f32::NEG_INFINITY,
        "NaN in x_hat should produce conservative lower bound"
    );
    assert_eq!(
        result.upper_b()[0],
        f32::INFINITY,
        "NaN in x_hat should produce conservative upper bound"
    );
    Ok(())
}

/// Same verification for the alpha-CROWN crossing path.
#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_near_zero_width_crossing_safe() -> Result<()> {
    let layer = ReLULayer::new();
    let min_subnormal = f32::from_bits(1);
    let l = -min_subnormal;
    let u = min_subnormal;
    let width = u - l;
    assert_eq!(
        width,
        f32::from_bits(2),
        "crossing subtraction should stay positive even at the subnormal floor"
    );
    let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let alpha = array![0.5_f32]; // crossing neuron, any alpha in [0,1]

    let (result, _gradient, _gradient_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, None)?;

    assert!(
        (0.5..=1.0).contains(&result.upper_a[[0, 0]]),
        "crossing slope should stay in [0.5, 1.0], got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        result.upper_b[0].is_finite(),
        "alpha-CROWN upper_b must be finite for crossing neuron: got {}",
        result.upper_b[0]
    );
    assert!(
        result.lower_b[0].is_finite(),
        "alpha-CROWN lower_b must be finite for crossing neuron: got {}",
        result.lower_b[0]
    );
    // Soundness: sample points in [l, u] and verify affine bounds contain ReLU(x).
    for k in 0..=50 {
        let x = l + (u - l) * (k as f32 / 50.0);
        let y = x.max(0.0);
        let lower_bound = result.lower_a[[0, 0]] * x + result.lower_b[0];
        let upper_bound = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            lower_bound <= y + 1e-24,
            "alpha-CROWN lower {} > relu({})={} for [{}, {}]",
            lower_bound,
            x,
            y,
            l,
            u
        );
        assert!(
            upper_bound >= y - 1e-24,
            "alpha-CROWN upper {} < relu({})={} for [{}, {}]",
            upper_bound,
            x,
            y,
            l,
            u
        );
    }
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_sdp_crown_large_crossing_width_safe() -> Result<()> {
    let layer = ReLULayer::new();
    let l = -f32::MAX;
    let u = f32::MAX;
    let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let x_hat = array![0.0_f32];

    let result = layer.propagate_linear_with_bounds_sdp(&bounds, &pre, &x_hat, 0.0)?;

    assert!(
        result.upper_a[[0, 0]] >= 0.5,
        "SDP-CROWN crossing slope should stay at or above the exact 0.5 chord, got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        result.upper_b[0].is_finite(),
        "SDP-CROWN upper_b must stay finite for large finite crossings, got {}",
        result.upper_b[0]
    );
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_large_crossing_width_safe() -> Result<()> {
    let layer = ReLULayer::new();
    let l = -f32::MAX;
    let u = f32::MAX;
    let pre = BoundedTensor::new(array![l].into_dyn(), array![u].into_dyn())?;
    let bounds = LinearBounds::identity(1);
    let alpha = array![0.5_f32];

    let (result, _gradient, _gradient_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, None)?;

    for &x in &[l, 0.0_f32, u] {
        let y = x.max(0.0);
        let upper_bound = result.upper_a[[0, 0]] * x + result.upper_b[0];
        assert!(
            upper_bound >= y,
            "alpha-CROWN upper {} < relu({})={} for [{}, {}]",
            upper_bound,
            x,
            y,
            l,
            u
        );
    }
    Ok(())
}

/// #3086: Verify that Inf coefficient in one row only degrades that row, not all rows.
///
/// Before this fix, a single Inf coefficient caused `new_or_conservative` to replace
/// the entire LinearBounds with conservative (A=0, b=±Inf). With per-row nonfinite
/// tracking, only the affected row is degraded.
#[ntest::timeout(5000)]
#[test]
fn test_alpha_crown_inf_coefficient_per_row_fallback() -> Result<()> {
    let layer = ReLULayer::new();
    // Crossing neuron: l=-1, u=2
    let pre = BoundedTensor::new(
        array![-1.0_f32, -1.0_f32].into_dyn(),
        array![2.0_f32, 2.0_f32].into_dyn(),
    )?;

    // Two output rows. Row 0 has Inf coefficient, row 1 has finite coefficient.
    // Both target neuron 0 (crossing).
    // Use direct struct construction (pub(crate) fields) to bypass debug_assert in
    // from_parts_unchecked — simulating accumulated bounds from safe_add that contain Inf.
    let bounds = LinearBounds {
        lower_a: ndarray::arr2(&[[f32::INFINITY, 0.0], [1.0, 0.0]]),
        lower_b: ndarray::arr1(&[0.0, 0.0]),
        upper_a: ndarray::arr2(&[[1.0, 0.0], [1.0, 0.0]]),
        upper_b: ndarray::arr1(&[0.0, 0.0]),
        lower_a_err: None,
        upper_a_err: None,
    };

    let alpha = array![0.5_f32, 0.5_f32];
    let (result, _gradient, _gradient_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, None)?;

    // Row 0 lower: Inf * 0.5 = Inf → row should be zeroed with -Inf bias
    assert!(
        result.lower_a[[0, 0]] == 0.0 && result.lower_a[[0, 1]] == 0.0,
        "Row 0 lower A should be zeroed, got [{}, {}]",
        result.lower_a[[0, 0]],
        result.lower_a[[0, 1]]
    );
    assert!(
        result.lower_b[0] == f32::NEG_INFINITY,
        "Row 0 lower b should be -Inf, got {}",
        result.lower_b[0]
    );

    // Row 1 lower: 1.0 * 0.5 = 0.5 → should be preserved (finite)
    assert!(
        (result.lower_a[[1, 0]] - 0.5).abs() < 1e-5,
        "Row 1 lower A should be preserved at 0.5, got {}",
        result.lower_a[[1, 0]]
    );
    assert!(
        result.lower_b[1].is_finite(),
        "Row 1 lower b should be finite, got {}",
        result.lower_b[1]
    );

    // Upper bounds: both rows have finite coefficients (1.0 * lambda)
    // Upper should be fully preserved (no Inf in upper_a input)
    assert!(
        result.upper_a[[0, 0]].is_finite(),
        "Row 0 upper A should be finite, got {}",
        result.upper_a[[0, 0]]
    );
    assert!(
        result.upper_a[[1, 0]].is_finite(),
        "Row 1 upper A should be finite, got {}",
        result.upper_a[[1, 0]]
    );

    Ok(())
}

// ===== Dual alpha behavioral tests (#3393) =====

/// Dual alpha: non-identity bounds (la>0, ua<0) produce different gradients and bounds
/// for lower vs upper paths. Identity bounds always have ua>=0, hiding dual alpha.
/// Reference: auto_LiRPA/operators/relu.py:647-652
#[ntest::timeout(5000)]
#[test]
fn test_dual_alpha_produces_different_gradients_3393() -> Result<()> {
    let layer = ReLULayer::new();
    let pre = BoundedTensor::new(array![-1.0_f32].into_dyn(), array![2.0_f32].into_dyn())?;

    // la>0 triggers alpha_lower path; ua<0 triggers alpha_upper path.
    let bounds = LinearBounds::new(
        ndarray::arr2(&[[2.0_f32]]),
        ndarray::arr1(&[0.0_f32]),
        ndarray::arr2(&[[-3.0_f32]]),
        ndarray::arr1(&[0.0_f32]),
    )?;
    let alpha_lower = array![0.3_f32];
    let alpha_upper = array![0.8_f32];

    let (result_dual, grad_lower, grad_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha_lower, Some(&alpha_upper))?;
    let (result_single, grad_lower_single, _) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha_lower, None)?;

    // gradient_lower = la * l = 2.0 * (-1.0) = -2.0 (same for dual and single)
    assert!(
        (grad_lower[0] - grad_lower_single[0]).abs() < 1e-6,
        "dual={}, single={}",
        grad_lower[0],
        grad_lower_single[0]
    );
    assert!(
        (grad_lower[0] - (-2.0)).abs() < 1e-5,
        "got {}",
        grad_lower[0]
    );

    // gradient_upper = ua * l = -3.0 * (-1.0) = 3.0
    assert!((grad_upper[0] - 3.0).abs() < 1e-5, "got {}", grad_upper[0]);

    // KEY: gradient_lower != gradient_upper (la*l vs ua*l)
    assert!(
        (grad_lower[0] - grad_upper[0]).abs() > 1.0,
        "lower={}, upper={}",
        grad_lower[0],
        grad_upper[0]
    );

    // Upper bounds differ: dual uses alpha_upper=0.8, single uses alpha_lower=0.3
    assert!(
        (result_dual.upper_a[[0, 0]] - result_single.upper_a[[0, 0]]).abs() > 0.1,
        "dual={}, single={}",
        result_dual.upper_a[[0, 0]],
        result_single.upper_a[[0, 0]]
    );
    // Lower_a identical (both use alpha_lower=0.3)
    assert!(
        (result_dual.lower_a[[0, 0]] - result_single.lower_a[[0, 0]]).abs() < 1e-6,
        "dual={}, single={}",
        result_dual.lower_a[[0, 0]],
        result_single.lower_a[[0, 0]]
    );

    Ok(())
}

/// Verify that dual alpha gradients are zero for stable neurons.
/// Stable positive (l >= 0) and stable negative (u <= 0) neurons have fixed
/// slopes (1.0 and 0.0 respectively), so alpha is unused and gradients must be zero.
#[ntest::timeout(5000)]
#[test]
fn test_dual_alpha_stable_neurons_zero_gradient_3393() -> Result<()> {
    let layer = ReLULayer::new();
    // Neuron 0: stable positive (l=1, u=3) — slope = 1, alpha ignored
    // Neuron 1: stable negative (l=-4, u=-1) — slope = 0, alpha ignored
    let pre = BoundedTensor::new(
        array![1.0_f32, -4.0].into_dyn(),
        array![3.0_f32, -1.0].into_dyn(),
    )?;
    let bounds = LinearBounds::new(
        ndarray::arr2(&[[2.0_f32, -1.0]]),
        ndarray::arr1(&[0.0_f32]),
        ndarray::arr2(&[[-1.0_f32, 2.0]]),
        ndarray::arr1(&[0.0_f32]),
    )?;

    let alpha = array![0.5_f32, 0.5];
    let alpha_upper = array![0.9_f32, 0.1];

    let (_result, grad_lower, grad_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, Some(&alpha_upper))?;

    // Both gradients must be zero for stable neurons
    for i in 0..2 {
        assert!(
            grad_lower[i].abs() < 1e-6,
            "gradient_lower[{}] should be 0 for stable neuron, got {}",
            i,
            grad_lower[i]
        );
        assert!(
            grad_upper[i].abs() < 1e-6,
            "gradient_upper[{}] should be 0 for stable neuron, got {}",
            i,
            grad_upper[i]
        );
    }

    Ok(())
}

/// Verify dual alpha soundness: bounds must contain ReLU(x) for all x in domain.
/// Uses non-identity accumulated bounds with different alpha_lower and alpha_upper.
#[ntest::timeout(5000)]
#[test]
fn test_dual_alpha_soundness_sampling_3393() -> Result<()> {
    let layer = ReLULayer::new();
    // 2 neurons: neuron 0 crossing [-2, 1], neuron 1 crossing [-1, 3]
    let pre = BoundedTensor::new(
        array![-2.0_f32, -1.0].into_dyn(),
        array![1.0_f32, 3.0].into_dyn(),
    )?;

    // Non-identity bounds with mixed signs to exercise both alpha paths
    let bounds = LinearBounds::new(
        ndarray::arr2(&[[1.5_f32, -0.5]]),
        ndarray::arr1(&[0.0_f32]),
        ndarray::arr2(&[[-0.8_f32, 2.0]]),
        ndarray::arr1(&[0.0_f32]),
    )?;

    let alpha = array![0.4_f32, 0.7];
    let alpha_upper = array![0.9_f32, 0.2];

    let (result, _grad_lower, _grad_upper) =
        layer.propagate_linear_with_alpha(&bounds, &pre, &alpha, Some(&alpha_upper))?;

    // Sample points and verify that linearized bounds contain the true CROWN output
    let l0 = -2.0_f32;
    let u0 = 1.0_f32;
    let l1 = -1.0_f32;
    let u1 = 3.0_f32;

    for k0 in 0..=20 {
        for k1 in 0..=20 {
            let x0 = l0 + (u0 - l0) * (k0 as f32 / 20.0);
            let x1 = l1 + (u1 - l1) * (k1 as f32 / 20.0);

            // Concretize the linear bounds at this point
            let lower_bound =
                result.lower_a[[0, 0]] * x0 + result.lower_a[[0, 1]] * x1 + result.lower_b[0];
            let upper_bound =
                result.upper_a[[0, 0]] * x0 + result.upper_a[[0, 1]] * x1 + result.upper_b[0];

            // True ReLU outputs
            let r0 = x0.max(0.0);
            let r1 = x1.max(0.0);

            // True linear combination of ReLU outputs (original bounds applied to ReLU output)
            // lower path: 1.5 * relu(x0) - 0.5 * relu(x1)
            let true_lower_y = 1.5 * r0 - 0.5 * r1;
            // upper path: -0.8 * relu(x0) + 2.0 * relu(x1)
            let true_upper_y = -0.8 * r0 + 2.0 * r1;

            assert!(
                lower_bound <= true_lower_y + 1e-3,
                "Dual alpha lower bound {} > true {} at ({}, {})",
                lower_bound,
                true_lower_y,
                x0,
                x1
            );
            assert!(
                upper_bound >= true_upper_y - 1e-3,
                "Dual alpha upper bound {} < true {} at ({}, {})",
                upper_bound,
                true_upper_y,
                x0,
                x1
            );
        }
    }

    Ok(())
}

// ===== Infinite-domain relaxation + CROWN soundness =====

/// Deterministic soundness check for the three infinite ReLU pre-activation cases.
/// These exercise the proven infinite-case arms of `relu_linear_relaxation` that the
/// NaN-only domain guard now allows to run on unbounded inputs.
#[ntest::timeout(5000)]
#[test]
fn test_relaxation_infinite_domain_soundness() {
    // (l, u, finite probe lo, finite probe hi)
    let cases: &[(f32, f32, f32, f32, &str)] = &[
        (f32::NEG_INFINITY, 3.0, -1.0e6, 3.0, "l=-inf,u=3"),
        (-2.0, f32::INFINITY, -2.0, 1.0e6, "l=-2,u=+inf"),
        (f32::NEG_INFINITY, f32::INFINITY, -1.0e6, 1.0e6, "both inf"),
    ];
    for &(l, u, lo, hi, label) in cases {
        let LinearRelaxation {
            lower_slope: ls,
            lower_intercept: li,
            upper_slope: us,
            upper_intercept: ui,
        } = relu_linear_relaxation(l, u);

        assert!(
            !ls.is_nan() && !li.is_nan() && !us.is_nan() && !ui.is_nan(),
            "{label}: NaN in relaxation ls={ls} li={li} us={us} ui={ui}"
        );

        for k in 0..=400 {
            let x = (lo as f64) + (hi as f64 - lo as f64) * (k as f64 / 400.0);
            let fx = relu_f64_reference(x);
            if li.is_finite() {
                let lower = ls as f64 * x + li as f64;
                assert!(
                    lower <= fx + 1e-3 * fx.abs().max(1.0),
                    "{label} lower UNSOUND at x={x}: {lower} > relu(x)={fx}"
                );
            }
            if ui.is_finite() {
                let upper = us as f64 * x + ui as f64;
                assert!(
                    upper + 1e-3 * fx.abs().max(1.0) >= fx,
                    "{label} upper UNSOUND at x={x}: {upper} < relu(x)={fx}"
                );
            }
        }
    }
}

/// CROWN backward through ReLU with infinite pre-activation bounds must succeed
/// (NaN-only guard) and yield a sound relaxation.
#[ntest::timeout(5000)]
#[test]
fn test_crown_infinite_domain_soundness() -> Result<()> {
    let layer = ReLULayer::new();
    let cases: &[(f32, f32, f32, f32, &str)] = &[
        (f32::NEG_INFINITY, 3.0, -1.0e6, 3.0, "l=-inf,u=3"),
        (-2.0, f32::INFINITY, -2.0, 1.0e6, "l=-2,u=+inf"),
        (f32::NEG_INFINITY, f32::INFINITY, -1.0e6, 1.0e6, "both inf"),
    ];
    for &(l, u, lo, hi, label) in cases {
        let pre =
            BoundedTensor::new_allow_infinite(array![l].into_dyn(), array![u].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let result = layer.propagate_linear_with_bounds(&bounds, &pre)?;

        let ls = result.lower_a[[0, 0]];
        let li = result.lower_b[0];
        let us = result.upper_a[[0, 0]];
        let ui = result.upper_b[0];
        assert!(
            !ls.is_nan() && !li.is_nan() && !us.is_nan() && !ui.is_nan(),
            "{label}: NaN in CROWN result"
        );

        for k in 0..=400 {
            let x = (lo as f64) + (hi as f64 - lo as f64) * (k as f64 / 400.0);
            let fx = relu_f64_reference(x);
            if li.is_finite() {
                let lower = ls as f64 * x + li as f64;
                assert!(
                    lower <= fx + 1e-3 * fx.abs().max(1.0),
                    "{label} CROWN lower UNSOUND at x={x}: {lower} > relu(x)={fx}"
                );
            }
            if ui.is_finite() {
                let upper = us as f64 * x + ui as f64;
                assert!(
                    upper + 1e-3 * fx.abs().max(1.0) >= fx,
                    "{label} CROWN upper UNSOUND at x={x}: {upper} < relu(x)={fx}"
                );
            }
        }
    }
    Ok(())
}

/// NaN pre-activation must still be rejected by the CROWN backward guard.
#[ntest::timeout(5000)]
#[test]
fn test_crown_nan_still_rejected() {
    let layer = ReLULayer::new();
    for (l, u) in [(f32::NAN, 1.0_f32), (-1.0_f32, f32::NAN)] {
        let pre = BoundedTensor::new_unchecked(array![l].into_dyn(), array![u].into_dyn()).unwrap();
        let bounds = LinearBounds::identity(1);
        let err = layer
            .propagate_linear_with_bounds(&bounds, &pre)
            .expect_err("NaN pre-activation must be rejected");
        assert!(
            matches!(err, NyError::NumericalInstability(_)),
            "expected NumericalInstability for NaN, got {err:?}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// Strict soundness of `relu_linear_relaxation` on INFINITE pre-activation
    /// bounds (l=-inf and/or u=+inf). Probes a finite grid covering the bounded part
    /// of the domain plus a large excursion into the unbounded direction; soundness
    /// (lower(x) <= ReLU(x) <= upper(x)) must hold at every probe, with any genuinely
    /// unbounded plane represented by a conservative ±Inf intercept.
    #[test]
    fn proptest_relu_relaxation_infinite_domain_soundness(
        inf_kind in 0usize..3,
        finite_endpoint in -10.0f32..10.0,
    ) {
        let (l, u) = match inf_kind {
            0 => (f32::NEG_INFINITY, finite_endpoint.max(0.5)),
            1 => (finite_endpoint.min(-0.5), f32::INFINITY),
            _ => (f32::NEG_INFINITY, f32::INFINITY),
        };

        let LinearRelaxation { lower_slope: ls, lower_intercept: li, upper_slope: us, upper_intercept: ui }
            = relu_linear_relaxation(l, u);

        prop_assert!(l.is_infinite() || u.is_infinite());
        prop_assert!(
            !ls.is_nan() && !li.is_nan() && !us.is_nan() && !ui.is_nan(),
            "NaN coeff for [{}, {}]: ls={} li={} us={} ui={}", l, u, ls, li, us, ui
        );

        let lo = if l.is_infinite() { -1.0e6_f64 } else { l as f64 };
        let hi = if u.is_infinite() { 1.0e6_f64 } else { u as f64 };

        for k in 0..=400 {
            let x = lo + (hi - lo) * (k as f64 / 400.0);
            let fx = relu_f64_reference(x);
            if li.is_finite() {
                let lower = ls as f64 * x + li as f64;
                prop_assert!(
                    lower <= fx + 1e-3 * fx.abs().max(1.0),
                    "ReLU INFINITE lower UNSOUND at x={}: {} > relu={} for [{}, {}]", x, lower, fx, l, u
                );
            } else {
                prop_assert!(li == f32::NEG_INFINITY);
            }
            if ui.is_finite() {
                let upper = us as f64 * x + ui as f64;
                prop_assert!(
                    upper + 1e-3 * fx.abs().max(1.0) >= fx,
                    "ReLU INFINITE upper UNSOUND at x={}: {} < relu={} for [{}, {}]", x, upper, fx, l, u
                );
            } else {
                prop_assert!(ui == f32::INFINITY);
            }
        }
    }
}
