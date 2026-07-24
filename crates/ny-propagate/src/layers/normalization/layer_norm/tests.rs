// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::decomposed::decomposed_norm_crown_backward;
use crate::tests::assert_close;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{arr1, arr2, Array1, ArrayD, Ix1, Ix2, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;

const TOL: f32 = 1e-5;

// ---- Constructor tests ----

#[ntest::timeout(10000)]
#[test]
fn test_new_stores_ny_beta_eps() {
    let ny = arr1(&[2.0, 3.0]);
    let beta = arr1(&[0.5, -0.5]);
    let layer = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5).unwrap();
    assert_eq!(layer.ny, ny);
    assert_eq!(layer.beta, beta);
    assert_close(layer.eps, 1e-5, TOL);
    assert!(!layer.forward_mode, "default forward_mode should be false");
    assert_eq!(layer.crown_mode, LayerNormCrownMode::IbpValidated);
    assert_eq!(layer.mode, LayerNormMode::Standard);
}

#[ntest::timeout(10000)]
#[test]
fn test_new_default_creates_identity_ny_zero_beta() {
    let layer = LayerNormLayer::new_default(4, 1e-5).unwrap();
    assert_eq!(layer.ny, Array1::<f32>::ones(4));
    assert_eq!(layer.beta, Array1::<f32>::zeros(4));
    assert_close(layer.eps, 1e-5, TOL);
}

#[ntest::timeout(10000)]
#[test]
fn test_builder_methods_chain_correctly() {
    let layer = LayerNormLayer::new_default(3, 1e-5)
        .unwrap()
        .with_forward_mode(true)
        .with_crown_mode(LayerNormCrownMode::Cut)
        .with_mode(LayerNormMode::MeanOnly);
    assert!(
        layer.forward_mode,
        "forward_mode should be true after with_forward_mode(true)"
    );
    assert_eq!(layer.crown_mode, LayerNormCrownMode::Cut);
    assert_eq!(layer.mode, LayerNormMode::MeanOnly);
}

// ---- eval correctness ----

#[ntest::timeout(10000)]
#[test]
fn test_eval_standard_hand_computed() {
    // x = [1, 3], ny = [1, 1], beta = [0, 0], eps = 0 (clamped to 1e-12)
    // mean = 2, var = ((1-2)^2 + (3-2)^2)/2 = 1, std ≈ 1
    // y ≈ [1*(1-2)/1 + 0, 1*(3-2)/1 + 0] = [-1, 1]
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0]), arr1(&[0.0, 0.0]), 0.0).unwrap();
    let y = layer.eval(&arr1(&[1.0, 3.0])).unwrap();
    assert_close(y[0], -1.0, TOL);
    assert_close(y[1], 1.0, TOL);
}

#[ntest::timeout(10000)]
#[test]
fn test_eval_standard_with_ny_beta() {
    // x = [1, 3], ny = [2, 0.5], beta = [10, -5], eps = 0 (clamped to 1e-12)
    // mean = 2, std ≈ 1 (from above)
    // y ≈ [2*(-1) + 10, 0.5*(1) + (-5)] = [8, -4.5]
    let layer = LayerNormLayer::new(arr1(&[2.0, 0.5]), arr1(&[10.0, -5.0]), 0.0).unwrap();
    let y = layer.eval(&arr1(&[1.0, 3.0])).unwrap();
    assert_close(y[0], 8.0, TOL);
    assert_close(y[1], -4.5, TOL);
}

#[ntest::timeout(10000)]
#[test]
fn test_eval_mean_only_mode() {
    // x = [1, 3], ny = [1, 1], beta = [0, 0]
    // mean = 2
    // y = [1*(1-2) + 0, 1*(3-2) + 0] = [-1, 1]
    let mut layer = LayerNormLayer::new(arr1(&[1.0, 1.0]), arr1(&[0.0, 0.0]), 0.0).unwrap();
    layer.mode = LayerNormMode::MeanOnly;
    let y = layer.eval(&arr1(&[1.0, 3.0])).unwrap();
    assert_close(y[0], -1.0, TOL);
    assert_close(y[1], 1.0, TOL);
}

// ---- Jacobian correctness ----

#[ntest::timeout(10000)]
#[test]
fn test_jacobian_matches_numerical_gradient() {
    let layer = LayerNormLayer::new(arr1(&[2.0, 0.5, 1.0]), arr1(&[0.1, -0.2, 0.3]), 1e-5).unwrap();
    let x = arr1(&[1.0, 2.0, 0.5]);
    let jac = layer.jacobian(&x).unwrap();

    // Numerical Jacobian via central differences for better accuracy
    let eps = 1e-3;
    for j in 0..3 {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[j] += eps;
        x_minus[j] -= eps;
        let y_plus = layer.eval(&x_plus).unwrap();
        let y_minus = layer.eval(&x_minus).unwrap();
        for i in 0..3 {
            let numerical = (y_plus[i] - y_minus[i]) / (2.0 * eps);
            assert!(
                (jac[[i, j]] - numerical).abs() < 5e-3,
                "Jacobian[{i},{j}] analytical={} numerical={numerical} diff={}",
                jac[[i, j]],
                (jac[[i, j]] - numerical).abs()
            );
        }
    }
}

// ---- IBP soundness (standard mode) ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_standard_contains_eval_at_corners() -> Result<()> {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5).unwrap();
    let input = BoundedTensor::new(
        arr1(&[0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[1.0, 3.0, 4.0]).into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    // Eval at lower and upper corners — both must be contained in output bounds
    let y_lower = layer.eval(&arr1(&[0.0, 1.0, 2.0])).unwrap();
    let y_upper = layer.eval(&arr1(&[1.0, 3.0, 4.0])).unwrap();
    let y_mid = layer.eval(&arr1(&[0.5, 2.0, 3.0])).unwrap();

    for i in 0..3 {
        assert!(
            output.lower()[[i]] <= y_lower[i] + 1e-5,
            "dim {i}: IBP lower {} > eval(lower_corner) {}",
            output.lower()[[i]],
            y_lower[i]
        );
        assert!(
            output.upper()[[i]] >= y_upper[i] - 1e-5,
            "dim {i}: IBP upper {} < eval(upper_corner) {}",
            output.upper()[[i]],
            y_upper[i]
        );
        assert!(
            output.lower()[[i]] <= y_mid[i] + 1e-5,
            "dim {i}: IBP lower {} > eval(midpoint) {}",
            output.lower()[[i]],
            y_mid[i]
        );
        assert!(
            output.upper()[[i]] >= y_mid[i] - 1e-5,
            "dim {i}: IBP upper {} < eval(midpoint) {}",
            output.upper()[[i]],
            y_mid[i]
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_ibp_standard_point_input_gives_tight_bounds() -> Result<()> {
    // When lower == upper (point input), IBP bounds should be tight
    let layer = LayerNormLayer::new(arr1(&[2.0, 0.5]), arr1(&[1.0, -1.0]), 1e-5).unwrap();
    let x = arr1(&[1.0, 3.0]);
    let input = BoundedTensor::new(x.clone().into_dyn(), x.clone().into_dyn())?;
    let output = layer.propagate_ibp(&input)?;
    let y_exact = layer.eval(&x).unwrap();

    for i in 0..2 {
        assert!(
            (output.lower()[[i]] - y_exact[i]).abs() < 1e-5,
            "dim {i}: point-input IBP lower {} far from exact {}",
            output.lower()[[i]],
            y_exact[i]
        );
        assert!(
            (output.upper()[[i]] - y_exact[i]).abs() < 1e-5,
            "dim {i}: point-input IBP upper {} far from exact {}",
            output.upper()[[i]],
            y_exact[i]
        );
    }
    Ok(())
}

// ---- IBP forward-mode ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_forward_mode_contains_concrete_evals() -> Result<()> {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5)
        .unwrap()
        .with_forward_mode(true);
    let input = BoundedTensor::new(
        arr1(&[0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[1.0, 3.0, 4.0]).into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    // Forward mode should still contain concrete eval at center
    let center = arr1(&[0.5, 2.0, 3.0]);
    let y_center = layer.eval(&center).unwrap();
    for i in 0..3 {
        assert!(
            output.lower()[[i]] <= y_center[i] + 1e-4,
            "dim {i}: forward-mode lower {} > eval(center) {}",
            output.lower()[[i]],
            y_center[i]
        );
        assert!(
            output.upper()[[i]] >= y_center[i] - 1e-4,
            "dim {i}: forward-mode upper {} < eval(center) {}",
            output.upper()[[i]],
            y_center[i]
        );
    }
    Ok(())
}

/// Regression test for #2074: forward-mode IBP with large ny must contain
/// concrete evaluations at corners. Before the fix, heuristic caps
/// (MAX_EFFECTIVE_NY=3.0, MIN_EFFECTIVE_STD=0.3) would underestimate
/// sensitivity for ny=10, producing bounds that miss actual outputs.
#[ntest::timeout(10000)]
#[test]
fn test_ibp_forward_mode_large_ny_sound() -> Result<()> {
    // ny=10 exceeds the old MAX_EFFECTIVE_NY=3.0 cap.
    // Low-variance inputs (all near 1.0) produce std ≈ 0.08 < old MIN_EFFECTIVE_STD=0.3.
    let layer = LayerNormLayer::new(arr1(&[10.0, 10.0, 10.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5)
        .unwrap()
        .with_forward_mode(true);
    let input = BoundedTensor::new(
        arr1(&[0.9, 0.95, 1.0]).into_dyn(),
        arr1(&[1.1, 1.05, 1.0]).into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    // Evaluate at several points including corners and verify bounds contain every one.
    for &(x0, x1, x2) in &[
        (0.9, 0.95, 1.0),
        (1.1, 1.05, 1.0),
        (0.9, 1.05, 1.0),
        (1.1, 0.95, 1.0),
    ] {
        let y = layer.eval(&arr1(&[x0, x1, x2]))?;
        for i in 0..3 {
            assert!(
                output.lower()[[i]] <= y[i] + 1e-3,
                "x=[{x0},{x1},{x2}] dim {i}: forward-mode lower {} > eval {} \
                 (ny=10, this was unsound before #2074 fix)",
                output.lower()[[i]],
                y[i]
            );
            assert!(
                output.upper()[[i]] >= y[i] - 1e-3,
                "x=[{x0},{x1},{x2}] dim {i}: forward-mode upper {} < eval {} \
                 (ny=10, this was unsound before #2074 fix)",
                output.upper()[[i]],
                y[i]
            );
        }
    }
    Ok(())
}

// ---- IBP MeanOnly mode ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_mean_only_contains_eval() -> Result<()> {
    let layer = LayerNormLayer::new(arr1(&[2.0, 0.5]), arr1(&[1.0, -1.0]), 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);
    let input = BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[2.0, 3.0]).into_dyn())?;
    let output = layer.propagate_ibp(&input)?;

    // Evaluate at several points — all must be within bounds
    for &(x0, x1) in &[(0.0, 1.0), (2.0, 3.0), (1.0, 2.0), (0.5, 2.5)] {
        let y = layer.eval(&arr1(&[x0, x1])).unwrap();
        for i in 0..2 {
            assert!(
                output.lower()[[i]] <= y[i] + 1e-4,
                "x=[{x0},{x1}] dim {i}: IBP lower {} > eval {}",
                output.lower()[[i]],
                y[i]
            );
            assert!(
                output.upper()[[i]] >= y[i] - 1e-4,
                "x=[{x0},{x1}] dim {i}: IBP upper {} < eval {}",
                output.upper()[[i]],
                y[i]
            );
        }
    }
    Ok(())
}

// ---- IBP shape validation ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_rejects_ny_size_mismatch() {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5).unwrap();
    let input =
        BoundedTensor::new(arr1(&[0.0, 1.0]).into_dyn(), arr1(&[1.0, 2.0]).into_dyn()).unwrap();
    let err = layer.propagate_ibp(&input).expect_err("ny size mismatch");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

// ---- Non-finite input rejection at BoundedTensor construction ----
// BoundedTensor::new rejects NaN/Inf at construction, so LayerNorm never sees
// non-finite inputs. Verify the boundary is enforced.

#[ntest::timeout(10000)]
#[test]
fn test_bounded_tensor_rejects_nan_at_construction() {
    let err = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 0.0]).unwrap(),
        arr1(&[1.0, 2.0]).into_dyn(),
    )
    .expect_err("NaN lower bounds should be rejected");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_bounded_tensor_rejects_inf_at_construction() {
    let err = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NEG_INFINITY, 0.0]).unwrap(),
        arr1(&[1.0, 2.0]).into_dyn(),
    )
    .expect_err("Inf lower bounds should be rejected");
    assert!(matches!(err, NyError::NumericalInstability(_)));
}

// ---- CROWN mode gating ----

#[ntest::timeout(10000)]
#[test]
fn test_crown_sound_mode_returns_soundness_refusal() {
    let layer = LayerNormLayer::new_default(3, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sound);
    let bounds = LinearBounds::identity(3);
    let pre_act = BoundedTensor::new(
        arr1(&[0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[1.0, 3.0, 4.0]).into_dyn(),
    )
    .unwrap();
    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect_err("Sound mode should refuse");
    assert!(matches!(err, NyError::SoundnessRefusal(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_cut_mode_returns_identity() -> Result<()> {
    let layer = LayerNormLayer::new_default(3, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Cut);
    let bounds = LinearBounds::identity(3);
    let pre_act = BoundedTensor::new(
        arr1(&[0.0, 1.0, 2.0]).into_dyn(),
        arr1(&[1.0, 3.0, 4.0]).into_dyn(),
    )?;
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;
    // Cut mode should return the input bounds unchanged
    assert_eq!(result.lower_a, bounds.lower_a);
    assert_eq!(result.upper_a, bounds.upper_a);
    assert_eq!(result.lower_b, bounds.lower_b);
    assert_eq!(result.upper_b, bounds.upper_b);
    Ok(())
}

fn scalar_bounds_to_batched_for_test(bounds: &LinearBounds) -> BatchedLinearBounds {
    BatchedLinearBounds::new(
        bounds.lower_a().clone().into_dyn(),
        bounds.lower_b().clone().into_dyn(),
        bounds.upper_a().clone().into_dyn(),
        bounds.upper_b().clone().into_dyn(),
        vec![bounds.num_inputs()],
        vec![bounds.num_outputs()],
    )
    .expect("scalar bounds should reshape into BatchedLinearBounds")
}

fn batched_bounds_to_scalar_for_test(bounds: &BatchedLinearBounds) -> LinearBounds {
    LinearBounds::new(
        bounds
            .lower_a()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("expected 2D lower_a"),
        bounds
            .lower_b()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("expected 1D lower_b"),
        bounds
            .upper_a()
            .clone()
            .into_dimensionality::<Ix2>()
            .expect("expected 2D upper_a"),
        bounds
            .upper_b()
            .clone()
            .into_dimensionality::<Ix1>()
            .expect("expected 1D upper_b"),
    )
    .expect("converted scalar bounds should be valid")
}

fn assert_interval_within_fused_ibp(actual: &BoundedTensor, fused_ibp: &BoundedTensor) {
    for (idx, (actual_lower, fused_lower)) in actual
        .lower()
        .iter()
        .zip(fused_ibp.lower().iter())
        .enumerate()
    {
        assert!(
            *actual_lower >= *fused_lower - 1e-5,
            "lower[{idx}] escaped fused LayerNorm IBP envelope: actual={} fused={}",
            actual_lower,
            fused_lower,
        );
    }
    for (idx, (actual_upper, fused_upper)) in actual
        .upper()
        .iter()
        .zip(fused_ibp.upper().iter())
        .enumerate()
    {
        assert!(
            *actual_upper <= *fused_upper + 1e-5,
            "upper[{idx}] escaped fused LayerNorm IBP envelope: actual={} fused={}",
            actual_upper,
            fused_upper,
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_layernorm_ibpvalidated_scalar_matches_decomposed_helper_2077() -> Result<()> {
    let layer = LayerNormLayer::new(arr1(&[1.5, -0.75, 0.25]), arr1(&[0.1, -0.2, 0.3]), 1e-5)?;
    let bounds = LinearBounds::new(
        arr2(&[[1.0, -0.5, 0.25], [0.0, 0.75, -1.25], [0.2, 0.1, 0.3]]),
        arr1(&[0.0, 0.25, -0.1]),
        arr2(&[[1.0, -0.5, 0.25], [0.0, 0.75, -1.25], [0.2, 0.1, 0.3]]),
        arr1(&[0.0, 0.25, -0.1]),
    )?;
    let pre_act = BoundedTensor::new(
        arr1(&[-1.0, 0.25, 0.5]).into_dyn(),
        arr1(&[0.5, 1.5, 2.0]).into_dyn(),
    )?;

    let actual = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;
    let expected = decomposed_norm_crown_backward(
        &scalar_bounds_to_batched_for_test(&bounds),
        &layer.ny,
        &layer.beta,
        layer.eps,
        &pre_act,
        layer.forward_mode,
    )?;
    let expected_scalar = batched_bounds_to_scalar_for_test(&expected.bounds);

    assert_eq!(actual.lower_a, expected_scalar.lower_a);
    assert_eq!(actual.lower_b, expected_scalar.lower_b);
    assert_eq!(actual.upper_a, expected_scalar.upper_a);
    assert_eq!(actual.upper_b, expected_scalar.upper_b);

    let concretized = actual.concretize_sound(&pre_act);
    let fused_ibp = layer.propagate_ibp(&pre_act)?;
    let fused_envelope = bounds.concretize_sound(&fused_ibp);
    assert_interval_within_fused_ibp(&concretized, &fused_envelope);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_layernorm_ibpvalidated_batched_matches_decomposed_helper_2077() -> Result<()> {
    let layer = LayerNormLayer::new(arr1(&[0.75, -1.25, 1.5]), arr1(&[0.0, 0.2, -0.1]), 1e-5)?;
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 0.5, -0.25, 1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, 1.0, 1.5, 0.75, 2.0, 3.0]).unwrap(),
    )?;

    let actual = layer.propagate_linear_batched_with_bounds(&bounds, &pre_act)?;
    let expected = decomposed_norm_crown_backward(
        &bounds,
        &layer.ny,
        &layer.beta,
        layer.eps,
        &pre_act,
        layer.forward_mode,
    )?;

    assert_eq!(actual.lower_a, expected.bounds.lower_a);
    assert_eq!(actual.lower_b, expected.bounds.lower_b);
    assert_eq!(actual.upper_a, expected.bounds.upper_a);
    assert_eq!(actual.upper_b, expected.bounds.upper_b);

    let concretized = actual.concretize_sound(&pre_act)?;
    let fused_ibp = layer.propagate_ibp(&pre_act)?;
    let fused_envelope = bounds.concretize_sound(&fused_ibp)?;
    assert_interval_within_fused_ibp(&concretized, &fused_envelope);
    Ok(())
}

// ---- CROWN backward requires pre-activation ----

#[ntest::timeout(10000)]
#[test]
fn test_crown_backward_requires_pre_activation() {
    let layer = LayerNormLayer::new_default(3, 1e-5).unwrap();
    let bounds = LinearBounds::identity(3);
    let err = layer
        .propagate_crown_backward(&bounds, None)
        .expect_err("should require pre-activation");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

// ---- propagate_linear returns UnsupportedOp ----

#[ntest::timeout(10000)]
#[test]
fn test_propagate_linear_returns_unsupported() {
    let layer = LayerNormLayer::new_default(3, 1e-5).unwrap();
    let bounds = LinearBounds::identity(3);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("should return UnsupportedOp");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

// ---- requires_pre_activation_bounds ----

#[ntest::timeout(10000)]
#[test]
fn test_requires_pre_activation_bounds_is_true() {
    let layer = LayerNormLayer::new_default(3, 1e-5).unwrap();
    assert!(
        layer.requires_pre_activation_bounds(),
        "layer norm requires pre-activation bounds"
    );
}

// ---- IBP 2D batched input ----

#[ntest::timeout(10000)]
#[test]
fn test_ibp_2d_batch_contains_eval() -> Result<()> {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0]), arr1(&[0.0, 0.0]), 1e-5).unwrap();
    // 2D input: [batch=2, norm_size=2]
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 3.0, 4.0, 5.0]).unwrap(),
    )?;
    let output = layer.propagate_ibp(&input)?;
    assert_eq!(output.shape(), &[2, 2]);

    // Eval at concrete points in each batch
    let y0 = layer.eval(&arr1(&[0.5, 2.0])).unwrap();
    let y1 = layer.eval(&arr1(&[3.0, 4.0])).unwrap();

    for i in 0..2 {
        assert!(
            output.lower()[[0, i]] <= y0[i] + 1e-4,
            "batch 0, dim {i}: lower {} > eval {}",
            output.lower()[[0, i]],
            y0[i]
        );
        assert!(
            output.upper()[[0, i]] >= y0[i] - 1e-4,
            "batch 0, dim {i}: upper {} < eval {}",
            output.upper()[[0, i]],
            y0[i]
        );
        assert!(
            output.lower()[[1, i]] <= y1[i] + 1e-4,
            "batch 1, dim {i}: lower {} > eval {}",
            output.lower()[[1, i]],
            y1[i]
        );
        assert!(
            output.upper()[[1, i]] >= y1[i] - 1e-4,
            "batch 1, dim {i}: upper {} < eval {}",
            output.upper()[[1, i]],
            y1[i]
        );
    }
    Ok(())
}

/// Regression test for #2806: LayerNorm IBP with zero-valued dimension must
/// return an error, not panic from integer division-by-zero. Tests both
/// standard and forward-mode code paths.
#[ntest::timeout(10000)]
#[test]
fn test_ibp_zero_dimension_returns_error_2806() {
    let ny = arr1(&[1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0]);
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 0, 2]), vec![]).expect("valid shape");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 0, 2]), vec![]).expect("valid shape");
    let input = BoundedTensor::new(lower, upper).expect("valid bounds");

    // Standard mode
    let layer = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5).unwrap();
    let err = layer.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{err}").contains("zero-valued dimension"),
        "got: {err}"
    );

    // Forward mode (same guard, different code path)
    let mut fwd = LayerNormLayer::new(ny, beta, 1e-5).unwrap();
    fwd.forward_mode = true;
    let err = fwd.propagate_ibp(&input).unwrap_err();
    assert!(
        format!("{err}").contains("zero-valued dimension"),
        "got: {err}"
    );
}

// ---- Invalid epsilon regression tests (#2729) ----

#[ntest::timeout(10000)]
#[test]
fn test_new_rejects_negative_eps() {
    let err = LayerNormLayer::new(arr1(&[1.0]), arr1(&[0.0]), -1.0).unwrap_err();
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "negative eps should return InvalidSpec, got: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_new_rejects_nan_eps() {
    let err = LayerNormLayer::new(arr1(&[1.0]), arr1(&[0.0]), f32::NAN).unwrap_err();
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "NaN eps should return InvalidSpec, got: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_new_rejects_inf_eps() {
    let err = LayerNormLayer::new(arr1(&[1.0]), arr1(&[0.0]), f32::INFINITY).unwrap_err();
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "Inf eps should return InvalidSpec, got: {err}"
    );
}

/// Regression test for #2901: LayerNorm CROWN Jacobian overflows to Inf when
/// ny is large and inputs are nearly constant (std ≈ sqrt(eps) is tiny).
/// CROWN backward must return NumericalInstability, not silently produce Inf bounds.
#[ntest::timeout(10000)]
#[test]
fn test_crown_sampling_jacobian_overflow_returns_error_2901() {
    // ny = 1e35, eps = minimum (1e-12) → std = sqrt(1e-12) = 1e-6
    // ny/std = 1e35/1e-6 = 1e41 → overflows f32 to Inf.
    let large_gamma = arr1(&[1e35, 1e35, 1e35]);
    let beta = arr1(&[0.0, 0.0, 0.0]);
    let layer = LayerNormLayer::new(large_gamma, beta, 0.0) // eps clamped to 1e-12
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let bounds = LinearBounds::identity(3);
    // Nearly constant inputs → var ≈ 0, std ≈ sqrt(eps)
    let pre_act = BoundedTensor::new(
        arr1(&[5.0, 5.0, 5.0]).into_dyn(),
        arr1(&[5.0, 5.0, 5.0]).into_dyn(),
    )
    .unwrap();

    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect_err("should return NumericalInstability for Inf Jacobian");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got: {err}"
    );
}

// ---- Inf coefficient NaN regression tests (#3027) ----

/// Helper: build batched mean-only LayerNorm CROWN result from Inf-coefficient input.
/// Before #3027 fix, this produced NumericalInstability from Inf-Inf=NaN.
fn batched_mean_only_inf_result_3027() -> BatchedLinearBounds {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);
    let (norm_size, out_dim) = (3, 2);
    // Row 0: one Inf coefficient (from compose NaN→Inf). Row 1: all finite.
    let la = ArrayD::from_shape_vec(
        IxDyn(&[out_dim, norm_size]),
        vec![f32::NEG_INFINITY, 1.0, 0.0, 0.5, 0.5, 0.5],
    )
    .unwrap();
    let ua = ArrayD::from_shape_vec(
        IxDyn(&[out_dim, norm_size]),
        vec![f32::INFINITY, 1.0, 0.0, 0.5, 0.5, 0.5],
    )
    .unwrap();
    let lb = ArrayD::from_shape_vec(IxDyn(&[out_dim]), vec![0.0, 0.0]).unwrap();
    let ub = ArrayD::from_shape_vec(IxDyn(&[out_dim]), vec![0.0, 0.0]).unwrap();
    let bounds = BatchedLinearBounds::new(la, lb, ua, ub, vec![norm_size], vec![out_dim]).unwrap();
    let pre_act = BoundedTensor::new(
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
        arr1(&[4.0, 5.0, 6.0]).into_dyn(),
    )
    .unwrap();
    layer
        .propagate_linear_batched_with_bounds(&bounds, &pre_act)
        .expect("should not fail after #3027 fix")
}

/// Regression test for #3027 (batched path): no NaN in output after Inf input.
#[ntest::timeout(10000)]
#[test]
fn test_crown_batched_mean_only_inf_no_nan_3027() {
    let result = batched_mean_only_inf_result_3027();
    assert!(!result.lower_a.iter().any(|v| v.is_nan()), "lower_a NaN");
    assert!(!result.upper_a.iter().any(|v| v.is_nan()), "upper_a NaN");
    assert!(!result.lower_b.iter().any(|v| v.is_nan()), "lower_b NaN");
    assert!(!result.upper_b.iter().any(|v| v.is_nan()), "upper_b NaN");
}

/// Regression test for #3027 (batched path): Inf row gets conservative bounds.
///
/// Per-row NaN guard (P1#759 fix): when any element in a row produces NaN from
/// Inf-Inf cancellation, the ENTIRE row is zeroed and bias becomes ±Inf.
/// This matches the scalar path strategy and prevents coefficient inversions
/// (lower_a > upper_a) at non-Inf positions in the same row.
#[ntest::timeout(10000)]
#[test]
fn test_crown_batched_mean_only_inf_row_conservative_3027() {
    let result = batched_mean_only_inf_result_3027();
    // Row 0 (had Inf at col 0): per-row guard zeroes ALL coefficients and sets ±Inf bias.
    // This matches the scalar path behavior (see test_crown_scalar_mean_only_inf_row_conservative_3027).
    for j in 0..3 {
        assert_eq!(
            result.lower_a[[0, j]],
            0.0,
            "lower_a[0,{j}] should be 0.0 from per-row NaN guard"
        );
        assert_eq!(
            result.upper_a[[0, j]],
            0.0,
            "upper_a[0,{j}] should be 0.0 from per-row NaN guard"
        );
    }
    assert_eq!(
        result.lower_b[[0]],
        f32::NEG_INFINITY,
        "lower_b[0] should be NEG_INFINITY from per-row NaN guard"
    );
    assert_eq!(
        result.upper_b[[0]],
        f32::INFINITY,
        "upper_b[0] should be INFINITY from per-row NaN guard"
    );
}

/// Regression test for #3027 (batched path): finite row preserved correctly.
#[ntest::timeout(10000)]
#[test]
fn test_crown_batched_mean_only_finite_row_preserved_3027() {
    let result = batched_mean_only_inf_result_3027();
    for j in 0..3 {
        assert!(result.lower_a[[1, j]].is_finite(), "lower_a[1,{j}]");
        assert!(result.upper_a[[1, j]].is_finite(), "upper_a[1,{j}]");
    }
}

/// Regression test for P1#759: batched Inf row must not have coefficient inversions.
///
/// The per-element NaN guard (pre-fix) produced lower_a[0,1] = +Inf, upper_a[0,1] = -Inf
/// (inverted) because non-Inf elements got `finite - Inf = -Inf` for lower_a and
/// `finite - (-Inf) = +Inf` for upper_a. The per-row guard prevents this by zeroing
/// the entire row when any element produces NaN.
#[ntest::timeout(10000)]
#[test]
fn test_crown_batched_mean_only_inf_no_coefficient_inversion_3027() {
    let result = batched_mean_only_inf_result_3027();
    // Verify no coefficient inversions in ANY row.
    let norm_size = 3;
    let out_dim = 2;
    for row in 0..out_dim {
        for col in 0..norm_size {
            assert!(
                result.lower_a[[row, col]] <= result.upper_a[[row, col]],
                "Coefficient inversion at [{row},{col}]: lower_a={} > upper_a={}",
                result.lower_a[[row, col]],
                result.upper_a[[row, col]],
            );
        }
    }
}

/// Helper: build scalar mean-only LayerNorm CROWN result from Inf-coefficient input.
fn scalar_mean_only_inf_result_3027() -> LinearBounds {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly);
    // Direct struct construction: both new() and from_parts_unchecked reject
    // Inf coefficients (new() validates, from_parts_unchecked has debug_assert).
    // This test intentionally needs Inf A-coefficients to simulate compose()'s
    // NaN→Inf fallback (#3027), so we construct the struct directly.
    let bounds = LinearBounds {
        lower_a: ndarray::Array2::from_shape_vec(
            (2, 3),
            vec![f32::NEG_INFINITY, 1.0, 0.0, 0.5, 0.5, 0.5],
        )
        .unwrap(),
        lower_b: arr1(&[0.0, 0.0]),
        upper_a: ndarray::Array2::from_shape_vec(
            (2, 3),
            vec![f32::INFINITY, 1.0, 0.0, 0.5, 0.5, 0.5],
        )
        .unwrap(),
        upper_b: arr1(&[0.0, 0.0]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let pre_act = BoundedTensor::new(
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
        arr1(&[4.0, 5.0, 6.0]).into_dyn(),
    )
    .unwrap();
    layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("should not fail after #3027 fix")
}

/// Regression test for #3027 (scalar path): Inf row gets conservative bounds.
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_mean_only_inf_row_conservative_3027() {
    let result = scalar_mean_only_inf_result_3027();
    // No NaN anywhere
    assert!(
        !result.lower_a.iter().any(|v| v.is_nan()),
        "lower_a has NaN"
    );
    assert!(
        !result.upper_a.iter().any(|v| v.is_nan()),
        "upper_a has NaN"
    );
    // Row 0 (had Inf): zeroed coefficients, ±Inf bias
    for j in 0..3 {
        assert_eq!(result.lower_a[[0, j]], 0.0, "lower_a[0,{j}]");
        assert_eq!(result.upper_a[[0, j]], 0.0, "upper_a[0,{j}]");
    }
    assert_eq!(result.lower_b[0], f32::NEG_INFINITY, "lower_b[0]");
    assert_eq!(result.upper_b[0], f32::INFINITY, "upper_b[0]");
}

/// Regression test for #3027 (scalar path): finite row preserved correctly.
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_mean_only_finite_row_preserved_3027() {
    let result = scalar_mean_only_inf_result_3027();
    // Row 1 (all-finite input) should have finite coefficients and bias
    for j in 0..3 {
        assert!(result.lower_a[[1, j]].is_finite(), "lower_a[1,{j}]");
        assert!(result.upper_a[[1, j]].is_finite(), "upper_a[1,{j}]");
    }
    assert!(result.lower_b[1].is_finite(), "lower_b[1]");
    assert!(result.upper_b[1].is_finite(), "upper_b[1]");
}

// ── Forward-mode IBP soundness: Jacobian-based (#3098) ──────────────────────

/// Regression test for #3098: forward-mode IBP must contain all concrete
/// evaluations within the input interval. The old `max_radius / n` coupling
/// correction underestimated the effect of input perturbation on the
/// normalization denominator.
#[ntest::timeout(10000)]
#[test]
fn test_ibp_forward_mode_soundness_issue_3098_wide_perturbation() -> Result<()> {
    // dim=8 with wide perturbation regime (large radii relative to center)
    let n = 8;
    let layer = LayerNormLayer::new(Array1::ones(n), Array1::zeros(n), 1e-5)
        .unwrap()
        .with_forward_mode(true);

    let configs: Vec<(Vec<f32>, Vec<f32>)> = vec![
        // (center, radius)
        (
            vec![-3.2, 2.1, 4.5, -1.0, 0.3, -2.8, 1.7, 3.9],
            vec![2.5, 1.8, 3.0, 4.0, 1.5, 2.2, 3.5, 0.8],
        ),
        (
            vec![4.0, -4.0, 3.0, -3.0, 2.0, -2.0, 1.0, -1.0],
            vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ),
        (
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
        ),
    ];

    for (center, rad) in &configs {
        let lower: Vec<f32> = center.iter().zip(rad.iter()).map(|(c, r)| c - r).collect();
        let upper: Vec<f32> = center.iter().zip(rad.iter()).map(|(c, r)| c + r).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.clone()).unwrap(),
        )?;

        let output = layer.propagate_ibp(&input)?;

        // Sample 500 random points and verify containment
        for s in 0..500 {
            let mut point = Vec::with_capacity(n);
            for i in 0..n {
                let t = ((s as u32).wrapping_mul(2654435761_u32) ^ (i as u32).wrapping_mul(7))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                point.push(lower[i] + (upper[i] - lower[i]) * t);
            }
            let x = Array1::from_vec(point.clone());
            let y = layer.eval(&x)?;
            for i in 0..n {
                assert!(
                    y[i] >= output.lower()[[i]] - 1e-3,
                    "LN forward #3098 sample={s}: y[{i}]={} < lower={}",
                    y[i],
                    output.lower()[[i]]
                );
                assert!(
                    y[i] <= output.upper()[[i]] + 1e-3,
                    "LN forward #3098 sample={s}: y[{i}]={} > upper={}",
                    y[i],
                    output.upper()[[i]]
                );
            }
        }
    }
    Ok(())
}

/// Forward-mode soundness with non-trivial ny/beta (negative, large, mixed).
#[ntest::timeout(10000)]
#[test]
fn test_ibp_forward_mode_soundness_custom_ny_3098() -> Result<()> {
    type GammaBetaConfig = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let configs: Vec<GammaBetaConfig> = vec![
        // (ny, beta, lower, upper)
        (
            vec![2.0, -1.0, 0.5],
            vec![0.1, -0.2, 0.3],
            vec![-3.0, -2.0, -1.0],
            vec![3.0, 2.0, 1.0],
        ),
        (
            vec![10.0, 10.0, 10.0],
            vec![0.0, 0.0, 0.0],
            vec![0.9, 0.95, 1.0],
            vec![1.1, 1.05, 1.0],
        ),
        (
            vec![-5.0, 3.0, -0.1, 7.0],
            vec![1.0, -1.0, 0.5, -0.5],
            vec![-2.0, -2.0, -2.0, -2.0],
            vec![2.0, 2.0, 2.0, 2.0],
        ),
    ];

    for (ny, beta, lower, upper) in &configs {
        let n = ny.len();
        let layer = LayerNormLayer::new(
            Array1::from_vec(ny.clone()),
            Array1::from_vec(beta.clone()),
            1e-5,
        )
        .unwrap()
        .with_forward_mode(true);
        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[n]), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[n]), upper.clone()).unwrap(),
        )?;

        let output = layer.propagate_ibp(&input)?;

        for s in 0..300 {
            let mut point = Vec::with_capacity(n);
            for i in 0..n {
                let t = ((s as u32).wrapping_mul(2654435761_u32) ^ (i as u32))
                    .wrapping_mul(2654435761_u32) as f32
                    / u32::MAX as f32;
                point.push(lower[i] + (upper[i] - lower[i]) * t);
            }
            let x = Array1::from_vec(point.clone());
            let y = layer.eval(&x)?;
            for i in 0..n {
                assert!(
                    y[i] >= output.lower()[[i]] - 1e-3,
                    "LN ny={ny:?} sample={s}: y[{i}]={} < lower={}",
                    y[i],
                    output.lower()[[i]]
                );
                assert!(
                    y[i] <= output.upper()[[i]] + 1e-3,
                    "LN ny={ny:?} sample={s}: y[{i}]={} > upper={}",
                    y[i],
                    output.upper()[[i]]
                );
            }
        }
    }
    Ok(())
}

// ── Forward-mode MeanOnly interval mean regression (#3142) ──────────────────
// The forward-mode MeanOnly path previously used center-point mean instead of
// interval mean bounds, producing unsound bounds. The path is currently
// unreachable through propagate_ibp (MeanOnly returns before forward_mode check),
// but we test the fixed logic through the private method to prevent regressions
// if the call graph is refactored.

/// Forward-mode MeanOnly IBP must contain all concrete evaluations. Regression
/// for #3142: center-point mean made lower bounds too tight (unsound).
#[ntest::timeout(10000)]
#[test]
fn test_ibp_forward_mode_mean_only_contains_evals_3142() -> Result<()> {
    let layer = LayerNormLayer::new(arr1(&[2.0, -1.0, 0.5]), arr1(&[0.1, -0.2, 0.3]), 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly)
        .with_forward_mode(true);

    let input = BoundedTensor::new(
        arr1(&[-1.0, 0.0, 1.0]).into_dyn(),
        arr1(&[2.0, 3.0, 4.0]).into_dyn(),
    )?;

    // Call propagate_ibp_forward_mode directly (private method) to exercise
    // the MeanOnly branch that is otherwise unreachable through propagate_ibp.
    let output = layer.propagate_ibp_forward_mode(&input)?;

    // Evaluate at corners and interior points — all must be within bounds.
    for &(x0, x1, x2) in &[
        (-1.0, 0.0, 1.0), // lower corner
        (2.0, 3.0, 4.0),  // upper corner
        (-1.0, 3.0, 1.0), // mixed corner
        (2.0, 0.0, 4.0),  // mixed corner
        (0.5, 1.5, 2.5),  // center
        (0.0, 1.0, 2.0),  // interior
    ] {
        let y = layer.eval(&arr1(&[x0, x1, x2]))?;
        for i in 0..3 {
            assert!(
                output.lower()[[i]] <= y[i] + 1e-4,
                "x=[{x0},{x1},{x2}] dim {i}: forward mean-only lower {} > eval {} (#3142)",
                output.lower()[[i]],
                y[i]
            );
            assert!(
                output.upper()[[i]] >= y[i] - 1e-4,
                "x=[{x0},{x1},{x2}] dim {i}: forward mean-only upper {} < eval {} (#3142)",
                output.upper()[[i]],
                y[i]
            );
        }
    }
    Ok(())
}

/// Forward-mode MeanOnly batched (2D input) must contain concrete evaluations.
#[ntest::timeout(10000)]
#[test]
fn test_ibp_forward_mode_mean_only_batched_contains_evals_3142() -> Result<()> {
    let layer = LayerNormLayer::new(arr1(&[1.5, -2.0]), arr1(&[0.0, 0.0]), 1e-5)
        .unwrap()
        .with_mode(LayerNormMode::MeanOnly)
        .with_forward_mode(true);

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, 0.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 4.0, 5.0]).unwrap(),
    )?;

    let output = layer.propagate_ibp_forward_mode(&input)?;

    // Test each batch element at corner and center points
    for &(x0, x1) in &[(-1.0, 0.0), (1.0, 2.0), (0.0, 1.0)] {
        let y = layer.eval(&arr1(&[x0, x1]))?;
        for i in 0..2 {
            assert!(
                output.lower()[[0, i]] <= y[i] + 1e-4,
                "batch 0 x=[{x0},{x1}] dim {i}: lower {} > eval {} (#3142)",
                output.lower()[[0, i]],
                y[i]
            );
            assert!(
                output.upper()[[0, i]] >= y[i] - 1e-4,
                "batch 0 x=[{x0},{x1}] dim {i}: upper {} < eval {} (#3142)",
                output.upper()[[0, i]],
                y[i]
            );
        }
    }
    for &(x0, x1) in &[(2.0, 3.0), (4.0, 5.0), (3.0, 4.0)] {
        let y = layer.eval(&arr1(&[x0, x1]))?;
        for i in 0..2 {
            assert!(
                output.lower()[[1, i]] <= y[i] + 1e-4,
                "batch 1 x=[{x0},{x1}] dim {i}: lower {} > eval {} (#3142)",
                output.lower()[[1, i]],
                y[i]
            );
            assert!(
                output.upper()[[1, i]] >= y[i] - 1e-4,
                "batch 1 x=[{x0},{x1}] dim {i}: upper {} < eval {} (#3142)",
                output.upper()[[1, i]],
                y[i]
            );
        }
    }
    Ok(())
}

// ── CROWN scalar NaN/Inf pre-activation guard tests ─────────────────────────
// These test the has_infinite guard at crown_scalar.rs:105-106 which returns
// identity relaxation when pre-activation bounds contain NaN or Inf.
// Previously untested — only the Jacobian overflow path (line 127) had coverage.

/// NaN in pre-activation lower bound triggers identity relaxation (CROWN scalar).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_lower_returns_constant_bounds() {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5)
        .expect("valid LayerNorm")
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let bounds = LinearBounds::identity(3);
    // NaN in lower bound → non-finite guard fires → constant bounds (#3259).
    // Use new_unchecked since BoundedTensor::new rejects NaN/Inf inputs.
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 1.0, 2.0]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("NaN pre-activation should return constant bounds, not error");

    // Constant bounds: A = 0, bias = [-inf, +inf] (trivially sound, #3259).
    assert!(
        result.lower_a().iter().all(|&v| v == 0.0),
        "lower_a should be all zeros"
    );
    assert!(
        result.upper_a().iter().all(|&v| v == 0.0),
        "upper_a should be all zeros"
    );
    assert!(
        result.lower_b().iter().all(|&v| v == f32::NEG_INFINITY),
        "lower_b should be -inf"
    );
    assert!(
        result.upper_b().iter().all(|&v| v == f32::INFINITY),
        "upper_b should be +inf"
    );
}

/// NaN in pre-activation upper bound triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_upper_returns_constant_bounds() {
    let layer = LayerNormLayer::new(arr1(&[2.0, 0.5]), arr1(&[0.1, -0.1]), 1e-5)
        .expect("valid LayerNorm")
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let bounds = LinearBounds::identity(2);
    // Use new_unchecked since BoundedTensor::new rejects NaN/Inf inputs.
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![3.0, f32::NAN]).expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("NaN pre-activation should return constant bounds, not error");

    assert!(
        result.lower_a().iter().all(|&v| v == 0.0),
        "lower_a should be all zeros"
    );
    assert!(
        result.upper_a().iter().all(|&v| v == 0.0),
        "upper_a should be all zeros"
    );
    assert!(
        result.lower_b().iter().all(|&v| v == f32::NEG_INFINITY),
        "lower_b should be -inf"
    );
    assert!(
        result.upper_b().iter().all(|&v| v == f32::INFINITY),
        "upper_b should be +inf"
    );
}

/// Inf in pre-activation bounds triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_inf_pre_activation_returns_constant_bounds() {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5)
        .expect("valid LayerNorm")
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let bounds = LinearBounds::identity(3);
    // Use new_unchecked since BoundedTensor::new rejects NaN/Inf inputs.
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NEG_INFINITY, 1.0, 2.0])
            .expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("Inf pre-activation should return constant bounds, not error");

    assert!(
        result.lower_a().iter().all(|&v| v == 0.0),
        "lower_a should be all zeros"
    );
    assert!(
        result.upper_a().iter().all(|&v| v == 0.0),
        "upper_a should be all zeros"
    );
    assert!(
        result.lower_b().iter().all(|&v| v == f32::NEG_INFINITY),
        "lower_b should be -inf"
    );
    assert!(
        result.upper_b().iter().all(|&v| v == f32::INFINITY),
        "upper_b should be +inf"
    );
}

// ── CROWN scalar Sampling path per-row NaN guard (#3128) ────────────────────
// The Standard Sampling path (crown_scalar.rs:213-249) accumulates Jacobian
// coefficients via la_f64 * jacobian[[i, k]]. When compose() sends Inf
// coefficients, Inf * 0.0 = NaN can poison the A-matrix. The per-row guard
// detects this and widens only the affected row to conservative bounds.

/// Helper: build scalar Standard Sampling CROWN result from Inf-coefficient input.
fn scalar_sampling_inf_result_3128() -> LinearBounds {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);
    // Direct struct construction to inject Inf coefficients (simulates compose()
    // NaN→Inf fallback). Row 0 has Inf at col 0; row 1 is all-finite.
    let bounds = LinearBounds {
        lower_a: ndarray::Array2::from_shape_vec(
            (2, 3),
            vec![f32::NEG_INFINITY, 1.0, 0.0, 0.5, 0.5, 0.5],
        )
        .unwrap(),
        lower_b: arr1(&[0.0, 0.0]),
        upper_a: ndarray::Array2::from_shape_vec(
            (2, 3),
            vec![f32::INFINITY, 1.0, 0.0, 0.5, 0.5, 0.5],
        )
        .unwrap(),
        upper_b: arr1(&[0.0, 0.0]),
        lower_a_err: None,
        upper_a_err: None,
    };
    let pre_act = BoundedTensor::new(
        arr1(&[1.0, 2.0, 3.0]).into_dyn(),
        arr1(&[4.0, 5.0, 6.0]).into_dyn(),
    )
    .unwrap();
    layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("should not fail after #3128 fix")
}

/// Regression test for #3128: Inf row gets conservative bounds in Sampling path.
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_sampling_inf_row_conservative_3128() {
    let result = scalar_sampling_inf_result_3128();
    // No NaN anywhere
    assert!(
        !result.lower_a.iter().any(|v| v.is_nan()),
        "lower_a has NaN"
    );
    assert!(
        !result.upper_a.iter().any(|v| v.is_nan()),
        "upper_a has NaN"
    );
    // Row 0 (had Inf at col 0): zeroed coefficients, ±Inf bias from per-row guard
    for j in 0..3 {
        assert_eq!(
            result.lower_a[[0, j]],
            0.0,
            "lower_a[0,{j}] should be 0.0 from per-row NaN guard"
        );
        assert_eq!(
            result.upper_a[[0, j]],
            0.0,
            "upper_a[0,{j}] should be 0.0 from per-row NaN guard"
        );
    }
    assert_eq!(
        result.lower_b[0],
        f32::NEG_INFINITY,
        "lower_b[0] should be NEG_INFINITY from per-row NaN guard"
    );
    assert_eq!(
        result.upper_b[0],
        f32::INFINITY,
        "upper_b[0] should be INFINITY from per-row NaN guard"
    );
}

/// Regression test for #3128: finite row preserved in Sampling path.
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_sampling_finite_row_preserved_3128() {
    let result = scalar_sampling_inf_result_3128();
    // Row 1 (all-finite input) should have finite coefficients and bias
    for j in 0..3 {
        assert!(result.lower_a[[1, j]].is_finite(), "lower_a[1,{j}]");
        assert!(result.upper_a[[1, j]].is_finite(), "upper_a[1,{j}]");
    }
    assert!(result.lower_b[1].is_finite(), "lower_b[1]");
    assert!(result.upper_b[1].is_finite(), "upper_b[1]");
}

/// Regression test for #3128: no coefficient inversions in Sampling path.
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_sampling_inf_no_coefficient_inversion_3128() {
    let result = scalar_sampling_inf_result_3128();
    for row in 0..2 {
        for col in 0..3 {
            assert!(
                result.lower_a[[row, col]] <= result.upper_a[[row, col]],
                "Coefficient inversion at [{row},{col}]: lower_a={} > upper_a={}",
                result.lower_a[[row, col]],
                result.upper_a[[row, col]],
            );
        }
    }
}

// ── LayerNorm IBP NaN integration tests (#2627) ─────────────────────────────
// LayerNorm IBP rejects non-finite input bounds with NumericalInstability error
// per Category B domain validation policy (ibp/common.rs:84-92). The input guard
// fires before the nan_propagating_min/max fold sites, preventing NaN from
// reaching the normalization arithmetic (0/0, inf/inf).

/// LayerNorm IBP with NaN in lower bound returns NumericalInstability error.
/// Exercises the non-finite input guard at ibp/common.rs:87-92.
#[ntest::timeout(10000)]
#[test]
fn test_ibp_nan_in_lower_returns_error_2627() {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5).unwrap();
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap();
    let err = layer
        .propagate_ibp(&input)
        .expect_err("NaN input should return NumericalInstability");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got: {err}"
    );
}

/// LayerNorm IBP with NaN in upper bound returns NumericalInstability error.
#[ntest::timeout(10000)]
#[test]
fn test_ibp_nan_in_upper_returns_error_2627() {
    let layer = LayerNormLayer::new(arr1(&[2.0, -1.0]), arr1(&[0.5, -0.5]), 1e-5).unwrap();
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, f32::NAN]).unwrap(),
    )
    .unwrap();
    let err = layer
        .propagate_ibp(&input)
        .expect_err("NaN in upper should return NumericalInstability");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got: {err}"
    );
}

/// LayerNorm IBP with all-NaN input returns NumericalInstability error.
#[ntest::timeout(10000)]
#[test]
fn test_ibp_all_nan_returns_error_2627() {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0]), arr1(&[0.0, 0.0, 0.0]), 1e-5).unwrap();
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN; 3]).unwrap(),
    )
    .unwrap();
    let err = layer
        .propagate_ibp(&input)
        .expect_err("all-NaN input should return NumericalInstability");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got: {err}"
    );
}

/// LayerNorm IBP 2D batched with NaN in one batch element returns
/// NumericalInstability (input guard is global, not per-batch).
#[ntest::timeout(10000)]
#[test]
fn test_ibp_2d_nan_in_batch_returns_error_2627() {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0]), arr1(&[0.0, 0.0]), 1e-5).unwrap();
    let input = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 1.0, f32::NAN, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 4.0, 5.0]).unwrap(),
    )
    .unwrap();
    let err = layer
        .propagate_ibp(&input)
        .expect_err("batched NaN should return NumericalInstability");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got: {err}"
    );
}

/// Both NaN and Inf in pre-activation bounds triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_mixed_nan_inf_pre_activation_returns_constant_bounds() {
    let layer = LayerNormLayer::new(arr1(&[1.0, 1.0, 1.0, 1.0]), arr1(&[0.0; 4]), 1e-5)
        .expect("valid LayerNorm")
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let bounds = LinearBounds::identity(4);
    // Use new_unchecked since BoundedTensor::new rejects NaN/Inf inputs.
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![f32::NAN, f32::NEG_INFINITY, 1.0, 2.0])
            .expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, f32::INFINITY, f32::NAN, 4.0])
            .expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("mixed NaN/Inf pre-activation should return constant bounds");

    assert!(
        result.lower_a().iter().all(|&v| v == 0.0),
        "lower_a should be all zeros"
    );
    assert!(
        result.upper_a().iter().all(|&v| v == 0.0),
        "upper_a should be all zeros"
    );
    assert!(
        result.lower_b().iter().all(|&v| v == f32::NEG_INFINITY),
        "lower_b should be -inf"
    );
    assert!(
        result.upper_b().iter().all(|&v| v == f32::INFINITY),
        "upper_b should be +inf"
    );
}

// ── Low-rank sampling parity tests (#1957) ──────────────────────────────────
//
// The low-rank path in `sampling_low_rank.rs` must produce results equivalent
// to the dense generic path in `crown_common::sampling_crown_scalar()`. Since
// both paths use the same deterministic sampling strategy (identical hash-based
// pseudo-random points), they should agree within floating-point tolerance.

/// Full-path parity test: low-rank Sampling result matches dense reference
/// on a small 3-neuron LayerNorm with non-trivial ny/beta.
#[ntest::timeout(10000)]
#[test]
fn test_layernorm_sampling_low_rank_matches_dense_scalar_path_1957() {
    use crate::layers::normalization::crown_common::sampling_crown_scalar;

    let ny = arr1(&[2.0, 0.5, 1.5]);
    let beta = arr1(&[0.1, -0.2, 0.3]);
    let layer = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let bounds = LinearBounds::identity(3);
    let pre_lower = arr1(&[1.0, 2.0, 3.0]);
    let pre_upper = arr1(&[4.0, 5.0, 6.0]);
    let pre_act =
        BoundedTensor::new(pre_lower.clone().into_dyn(), pre_upper.clone().into_dyn()).unwrap();

    // New low-rank path (production)
    let low_rank_result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("low-rank path should succeed");

    // Dense reference (test oracle): call the generic helper directly
    let dense_result = sampling_crown_scalar(&layer, &bounds, &pre_lower, &pre_upper)
        .expect("dense path should succeed");

    // Compare A-coefficients and biases within tolerance.
    // The low-rank path uses f64 Jacobian arithmetic while the dense path uses
    // f32 Jacobian, so small differences are expected.
    let tol = 1e-4;
    for i in 0..3 {
        for j in 0..3 {
            let lr_la = low_rank_result.lower_a()[[i, j]];
            let dn_la = dense_result.lower_a()[[i, j]];
            assert!(
                (lr_la - dn_la).abs() < tol,
                "lower_a[{i},{j}] mismatch: low_rank={lr_la}, dense={dn_la}"
            );
            let lr_ua = low_rank_result.upper_a()[[i, j]];
            let dn_ua = dense_result.upper_a()[[i, j]];
            assert!(
                (lr_ua - dn_ua).abs() < tol,
                "upper_a[{i},{j}] mismatch: low_rank={lr_ua}, dense={dn_ua}"
            );
        }
        let lr_lb = low_rank_result.lower_b()[i];
        let dn_lb = dense_result.lower_b()[i];
        assert!(
            (lr_lb - dn_lb).abs() < tol,
            "lower_b[{i}] mismatch: low_rank={lr_lb}, dense={dn_lb}"
        );
        let lr_ub = low_rank_result.upper_b()[i];
        let dn_ub = dense_result.upper_b()[i];
        assert!(
            (lr_ub - dn_ub).abs() < tol,
            "upper_b[{i}] mismatch: low_rank={lr_ub}, dense={dn_ub}"
        );
    }
}

/// Full-path parity with non-identity bounds (2 outputs, 3 inputs).
#[ntest::timeout(10000)]
#[test]
fn test_layernorm_sampling_low_rank_non_identity_bounds_1957() {
    use crate::layers::normalization::crown_common::sampling_crown_scalar;

    let ny = arr1(&[1.0, 1.0, 1.0]);
    let beta = arr1(&[0.0, 0.0, 0.0]);
    let layer = LayerNormLayer::new(ny, beta, 1e-5)
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::Sampling);

    // Non-identity bounds: 2 outputs from 3 inputs with mixed coefficients
    let la = arr2(&[[0.5, 0.3, -0.2], [-0.1, 0.8, 0.1]]);
    let ua = arr2(&[[0.6, 0.4, -0.1], [0.0, 0.9, 0.2]]);
    let lb = arr1(&[-0.5, 0.1]);
    let ub = arr1(&[0.5, 0.2]);
    let bounds = LinearBounds::new(la, lb, ua, ub).unwrap();

    let pre_lower = arr1(&[0.5, 1.0, 1.5]);
    let pre_upper = arr1(&[2.5, 3.0, 3.5]);
    let pre_act =
        BoundedTensor::new(pre_lower.clone().into_dyn(), pre_upper.clone().into_dyn()).unwrap();

    let low_rank_result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("low-rank path should succeed");
    let dense_result = sampling_crown_scalar(&layer, &bounds, &pre_lower, &pre_upper)
        .expect("dense path should succeed");

    let tol = 1e-4;
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (low_rank_result.lower_a()[[i, j]] - dense_result.lower_a()[[i, j]]).abs() < tol,
                "lower_a[{i},{j}] mismatch"
            );
            assert!(
                (low_rank_result.upper_a()[[i, j]] - dense_result.upper_a()[[i, j]]).abs() < tol,
                "upper_a[{i},{j}] mismatch"
            );
        }
        assert!(
            (low_rank_result.lower_b()[i] - dense_result.lower_b()[i]).abs() < tol,
            "lower_b[{i}] mismatch"
        );
        assert!(
            (low_rank_result.upper_b()[i] - dense_result.upper_b()[i]).abs() < tol,
            "upper_b[{i}] mismatch"
        );
    }
}
