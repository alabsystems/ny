// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::borrow::Cow;

use crate::layers::activations::LinearRelaxation;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::{next_down_f32, next_up_f32};

// ── compute_strides tests ─────────────────────────────────────────

#[test]
fn test_strides_3d() {
    let strides = compute_strides(&[2, 3, 4]).expect("3D strides should succeed");
    assert_eq!(strides, vec![12, 4, 1]);
}

#[test]
fn test_strides_1d() {
    let strides = compute_strides(&[5]).expect("1D strides should succeed");
    assert_eq!(strides, vec![1]);
}

#[test]
fn test_strides_empty() {
    let strides = compute_strides(&[]).expect("empty strides should succeed");
    assert!(
        strides.is_empty(),
        "empty shape should produce empty strides"
    );
}

#[test]
fn test_strides_2d() {
    let strides = compute_strides(&[4, 7]).expect("2D strides should succeed");
    assert_eq!(strides, vec![7, 1]);
}

#[test]
fn test_strides_overflow_returns_invalid_spec_3012() {
    let err =
        compute_strides(&[2, (usize::MAX / 2) + 1, 2]).expect_err("stride overflow must fail");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("compute_strides: stride overflow")),
        "expected stride overflow InvalidSpec, got: {err:?}"
    );
}

// ── crown_elementwise_backward tests ──────────────────────────────

// Use a simple linear activation f(x) = 2x + 1 as the "relaxation".
// For any interval: lower_slope=2, lower_intercept=1, upper_slope=2, upper_intercept=1.
fn linear_relaxation(_l: f32, _u: f32) -> LinearRelaxation {
    LinearRelaxation::new(2.0, 1.0, 2.0, 1.0)
}

/// Assert CROWN coefficient tolerance: `|actual - expected| < tol`.
fn assert_coeff(actual: f32, expected: f32, label: &str) {
    assert!(
        (actual - expected).abs() < 1e-5,
        "{label}: expected {expected}, got {actual}"
    );
}

#[test]
fn test_elementwise_identity_bounds_linear_activation() {
    // With identity incoming bounds and a linear activation f(x)=2x+1:
    // new_A = I * 2 = [[2,0],[0,2]], new_b = [0+1*1+1*0, 0+0*1+1*1] = ...
    // For identity bounds: lower_a = upper_a = [[1,0],[0,1]], b = [0,0]
    // Each output j: coefficient la[j,i] > 0 when j==i.
    // new_lower_a[j,i] = la[j,i] * 2 for la>0 = 2*I
    // new_lower_b[j] += la[j,i] * 1.0 for each i where la>0
    // j=0: la[0,0]=1 > 0 → b += 1.0, la[0,1]=0 → skip → b[0] = 1.0
    // j=1: la[1,0]=0, la[1,1]=1 → b[1] = 1.0
    let bounds = LinearBounds::identity(2);
    let pre = BoundedTensor::new(
        array![-1.0_f32, 0.0].into_dyn(),
        array![1.0_f32, 2.0].into_dyn(),
    )
    .unwrap();
    let result = crown_elementwise_backward(&bounds, &pre, linear_relaxation).unwrap();
    assert_coeff(result.lower_a[[0, 0]], 2.0, "lower_a[0,0]");
    assert_coeff(result.lower_a[[0, 1]], 0.0, "lower_a[0,1]");
    assert_coeff(result.lower_a[[1, 0]], 0.0, "lower_a[1,0]");
    assert_coeff(result.lower_a[[1, 1]], 2.0, "lower_a[1,1]");
    assert_coeff(result.lower_b[0], 1.0, "lower_b[0]");
    assert_coeff(result.lower_b[1], 1.0, "lower_b[1]");
}

#[test]
fn test_elementwise_negative_coefficients_swap() {
    // Negative incoming coefficient should swap lower/upper relaxation.
    // Use an asymmetric relaxation: lower=(1, 0), upper=(3, 2)
    fn asym_relax(_l: f32, _u: f32) -> LinearRelaxation {
        LinearRelaxation::new(1.0, 0.0, 3.0, 2.0) // lower: x, upper: 3x+2
    }

    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![-1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![-1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();
    let pre = BoundedTensor::new(array![0.0_f32].into_dyn(), array![1.0_f32].into_dyn()).unwrap();
    let result = crown_elementwise_backward(&bounds, &pre, asym_relax).unwrap();

    // la = -1 < 0: new_lower_a[0,0] = la * upper_slope = -1 * 3 = -3
    //              new_lower_b[0] += la * upper_intercept = -1 * 2 = -2
    assert_coeff(result.lower_a[[0, 0]], -3.0, "neg coeff lower_a");
    assert_coeff(result.lower_b[0], -2.0, "neg coeff lower_b");

    // ua = -1 < 0: new_upper_a[0,0] = ua * lower_slope = -1 * 1 = -1
    //              new_upper_b[0] += ua * lower_intercept = -1 * 0 = 0
    assert_coeff(result.upper_a[[0, 0]], -1.0, "neg coeff upper_a");
    assert_coeff(result.upper_b[0], 0.0, "neg coeff upper_b");
}

#[test]
fn test_elementwise_zero_coefficient_guard() {
    // Zero coefficient should remain zero (no NaN from 0 * inf)
    fn inf_relax(_l: f32, _u: f32) -> LinearRelaxation {
        LinearRelaxation::new(
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        )
    }

    let bounds = LinearBounds::new(
        Array2::zeros((1, 2)),
        Array1::zeros(1),
        Array2::zeros((1, 2)),
        Array1::zeros(1),
    )
    .unwrap();
    let pre = BoundedTensor::new(
        array![-1.0_f32, -1.0].into_dyn(),
        array![1.0_f32, 1.0].into_dyn(),
    )
    .unwrap();
    let result = crown_elementwise_backward(&bounds, &pre, inf_relax).unwrap();
    // Coefficients should be exactly zero (no NaN from 0 * inf)
    assert!(
        result.lower_a.iter().all(|&v| v == 0.0),
        "lower_a should be all zeros"
    );
    assert!(
        result.upper_a.iter().all(|&v| v == 0.0),
        "upper_a should be all zeros"
    );
    // Bias: directed rounding (#1992) shifts 0.0 by 1 ULP:
    // next_down_f32(0.0) = -1.4e-45, next_up_f32(0.0) = 1.4e-45.
    // This is sound (lower_b <= 0.0 <= upper_b) and no NaN.
    assert!(
        result.lower_b.iter().all(|&v| v.is_finite() && v <= 0.0),
        "lower_b should be finite and <= 0"
    );
    assert!(
        result.upper_b.iter().all(|&v| v.is_finite() && v >= 0.0),
        "upper_b should be finite and >= 0"
    );
}

/// Regression test: crown_elementwise_backward with large slopes exercises
/// f64 bias accumulation (#1745) and verifies no overflow in coefficient output.
///
/// The f64 accumulation fix (#1745) targets "slope~3000, intercept~-20000, 8x
/// cancellation" but had no direct unit test with those magnitudes.
///
/// Part of #1932 (CROWN coefficient overflow class).
#[test]
fn test_elementwise_large_slope_f64_bias_accumulation_1745() {
    // Relaxation with large slopes/intercepts. Combined with a large incoming
    // bias seed, this creates catastrophic cancellation: old f32 accumulation
    // drifts away from the small residual while f64 accumulation preserves it.
    fn large_slope_relax(_l: f32, _u: f32) -> LinearRelaxation {
        LinearRelaxation::new(
            3000.0,   // lower_slope
            -20000.0, // lower_intercept
            3000.0,   // upper_slope
            -19999.0, // upper_intercept (close to lower for cancellation test)
        )
    }

    // 8 neurons with large incoming coefficients so la * intercept ~= O(1e8)
    // contributions that nearly cancel a large initial bias term.
    let n = 8;
    let mut lower_a = Array2::<f32>::zeros((1, n));
    let mut upper_a = Array2::<f32>::zeros((1, n));
    for i in 0..n {
        let coeff = 3000.0 + 0.01 * i as f32;
        lower_a[[0, i]] = coeff;
        upper_a[[0, i]] = coeff;
    }
    let lower_intercept = -20000.0_f32;
    let upper_intercept = -19999.0_f32;
    let lower_cancel_seed = -(0..n)
        .map(|i| lower_a[[0, i]] as f64 * lower_intercept as f64)
        .sum::<f64>()
        + 1.0;
    let upper_cancel_seed = -(0..n)
        .map(|i| upper_a[[0, i]] as f64 * upper_intercept as f64)
        .sum::<f64>()
        + 1.0;

    let bounds = LinearBounds::new(
        lower_a,
        Array1::from_vec(vec![lower_cancel_seed as f32]),
        upper_a,
        Array1::from_vec(vec![upper_cancel_seed as f32]),
    )
    .unwrap();
    let pre = BoundedTensor::new(
        Array1::from_elem(n, -1.0_f32).into_dyn(),
        Array1::from_elem(n, 1.0_f32).into_dyn(),
    )
    .unwrap();

    let result = crown_elementwise_backward(&bounds, &pre, large_slope_relax).unwrap();

    // Coefficients: new_a[0,i] = old_a[0,i] * slope = (3000 + 0.01i) * 3000
    for i in 0..n {
        let expected = (3000.0 + 0.01 * i as f32) * 3000.0;
        assert!(
            (result.lower_a[[0, i]] - expected).abs() < 4.0,
            "lower_a[0,{}] = {} expected ~{}",
            i,
            result.lower_a[[0, i]],
            expected,
        );
    }

    // Bias: verify the precise cancellation residual (about +1.0 before directed
    // rounding), not only finiteness. This fails under old f32 accumulation.
    let expected_lower_raw = bounds.lower_b[0] as f64
        + (0..n)
            .map(|i| bounds.lower_a[[0, i]] as f64 * lower_intercept as f64)
            .sum::<f64>();
    let expected_upper_raw = bounds.upper_b[0] as f64
        + (0..n)
            .map(|i| bounds.upper_a[[0, i]] as f64 * upper_intercept as f64)
            .sum::<f64>();
    assert!(
        expected_lower_raw.abs() < 10.0,
        "test setup should produce strong cancellation in lower bias, got {}",
        expected_lower_raw
    );
    assert!(
        expected_upper_raw.abs() < 10.0,
        "test setup should produce strong cancellation in upper bias, got {}",
        expected_upper_raw
    );

    let expected_lower_b = next_down_f32(expected_lower_raw as f32);
    let expected_upper_b = next_up_f32(expected_upper_raw as f32);
    assert!(
        (result.lower_b[0] - expected_lower_b).abs() <= 1e-4,
        "lower_b should match directed-rounded f64 accumulation: expected {}, got {}",
        expected_lower_b,
        result.lower_b[0],
    );
    assert!(
        (result.upper_b[0] - expected_upper_b).abs() <= 1e-4,
        "upper_b should match directed-rounded f64 accumulation: expected {}, got {}",
        expected_upper_b,
        result.upper_b[0],
    );

    // Verify no NaN anywhere
    assert!(!result.lower_a.iter().any(|v| v.is_nan()), "NaN in lower_a");
    assert!(!result.upper_a.iter().any(|v| v.is_nan()), "NaN in upper_a");
    assert!(!result.lower_b.iter().any(|v| v.is_nan()), "NaN in lower_b");
    assert!(!result.upper_b.iter().any(|v| v.is_nan()), "NaN in upper_b");
}

#[test]
fn test_elementwise_shape_mismatch() {
    let bounds = LinearBounds::identity(3);
    let pre = BoundedTensor::new(
        array![-1.0_f32, 0.0].into_dyn(),
        array![1.0_f32, 2.0].into_dyn(),
    )
    .unwrap();
    let err =
        crown_elementwise_backward(&bounds, &pre, linear_relaxation).expect_err("shape mismatch");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn test_elementwise_multi_output() {
    // 2 outputs, 3 inputs
    let bounds = LinearBounds::new(
        Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.5]).unwrap(),
        Array1::from_vec(vec![0.0, 0.0]),
        Array2::from_shape_vec((2, 3), vec![1.0, 0.0, -1.0, 0.0, 1.0, 0.5]).unwrap(),
        Array1::from_vec(vec![0.0, 0.0]),
    )
    .unwrap();
    let pre = BoundedTensor::new(
        array![-1.0_f32, 0.0, -2.0].into_dyn(),
        array![1.0_f32, 2.0, 3.0].into_dyn(),
    )
    .unwrap();
    let result = crown_elementwise_backward(&bounds, &pre, linear_relaxation).unwrap();
    assert_eq!(result.lower_a.shape(), &[2, 3]);
    assert_eq!(result.lower_b.len(), 2);
    // j=0, i=0: la=1>0 → new_a = 1*2 = 2, b += 1*1 = 1
    assert_coeff(result.lower_a[[0, 0]], 2.0, "multi-out lower_a[0,0]");
    // j=0, i=2: la=-1<0 → new_a = -1*2 = -2 (uses upper slope), b += -1*1 = -1
    assert_coeff(result.lower_a[[0, 2]], -2.0, "multi-out lower_a[0,2]");
    // j=0 total bias: 0 + 1 + 0 + (-1) = 0
    assert_coeff(result.lower_b[0], 0.0, "multi-out lower_b[0]");
}

// ── crown_elementwise_backward_batched tests ──────────────────────

#[test]
fn test_batched_identity_linear_activation() {
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        vec![2],
        vec![2],
    );
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
    )
    .unwrap();
    let result = crown_elementwise_backward_batched(&bounds, &pre, linear_relaxation).unwrap();
    // Same as non-batched: slopes doubled, intercepts in bias
    assert_coeff(result.lower_a[[0, 0]], 2.0, "batched lower_a[0,0]");
    assert_coeff(result.lower_a[[1, 1]], 2.0, "batched lower_a[1,1]");
    assert_coeff(result.lower_b[[0]], 1.0, "batched lower_b[0]");
    assert_coeff(result.lower_b[[1]], 1.0, "batched lower_b[1]");
}

#[test]
fn test_batched_dim_mismatch() {
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap(),
        ArrayD::zeros(IxDyn(&[2])),
        vec![3],
        vec![2],
    );
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![0.0; 5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[5]), vec![1.0; 5]).unwrap(),
    )
    .unwrap();
    let err = crown_elementwise_backward_batched(&bounds, &pre, linear_relaxation)
        .expect_err("dim mismatch");
    assert!(
        matches!(err, NyError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
}

#[test]
fn test_batched_1d_error() {
    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0; 3]).unwrap(),
        ArrayD::zeros(IxDyn(&[1])),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0; 3]).unwrap(),
        ArrayD::zeros(IxDyn(&[1])),
        vec![3],
        vec![1],
    );
    let pre = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0; 3]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0; 3]).unwrap(),
    )
    .unwrap();
    let err =
        crown_elementwise_backward_batched(&bounds, &pre, linear_relaxation).expect_err("< 2 dims");
    assert!(
        matches!(err, NyError::InvalidSpec(_)),
        "expected InvalidSpec, got {err:?}"
    );
}

// ── BoundPropagation trait dispatch tests ─────────────────────────

// A simple test implementation of BoundPropagation for testing dispatch
struct TestLinearLayer;

impl BoundPropagation for TestLinearLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        Ok(input.clone())
    }

    fn propagate_linear<'a>(&self, bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Ok(Cow::Borrowed(bounds))
    }
}

struct TestNonlinearLayer;

impl BoundPropagation for TestNonlinearLayer {
    fn propagate_ibp(&self, input: &BoundedTensor) -> Result<BoundedTensor> {
        Ok(input.clone())
    }

    fn propagate_linear<'a>(&self, _bounds: &'a LinearBounds) -> Result<Cow<'a, LinearBounds>> {
        Err(NyError::UnsupportedOp("needs preact".to_string()))
    }

    fn requires_pre_activation_bounds(&self) -> bool {
        true
    }

    fn propagate_linear_with_bounds(
        &self,
        bounds: &LinearBounds,
        _pre_activation: &BoundedTensor,
    ) -> Result<LinearBounds> {
        Ok(bounds.clone())
    }
}

#[test]
fn test_trait_dispatch_linear() {
    let layer = TestLinearLayer;
    assert!(
        !layer.requires_pre_activation_bounds(),
        "linear layer should not require pre-activation bounds"
    );
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_crown_backward(&bounds, None).unwrap();
    assert_coeff(result.lower_a[[0, 0]], 1.0, "linear dispatch lower_a[0,0]");
}

#[test]
fn test_trait_dispatch_nonlinear_with_preact() {
    let layer = TestNonlinearLayer;
    assert!(
        layer.requires_pre_activation_bounds(),
        "nonlinear layer should require pre-activation bounds"
    );
    let bounds = LinearBounds::identity(2);
    let pre = BoundedTensor::new(
        array![-1.0_f32, 0.0].into_dyn(),
        array![1.0_f32, 2.0].into_dyn(),
    )
    .unwrap();
    let result = layer.propagate_crown_backward(&bounds, Some(&pre)).unwrap();
    assert_coeff(
        result.lower_a[[0, 0]],
        1.0,
        "nonlinear dispatch lower_a[0,0]",
    );
}

#[test]
fn test_trait_dispatch_nonlinear_missing_preact() {
    let layer = TestNonlinearLayer;
    let bounds = LinearBounds::identity(2);
    let err = layer
        .propagate_crown_backward(&bounds, None)
        .expect_err("missing preact");
    assert!(
        matches!(err, NyError::UnsupportedOp(_)),
        "expected UnsupportedOp, got {err:?}"
    );
}

/// Regression test for #1992: f64→f32 bias cast must use directed rounding
/// (next_down_f32 for lower_b, next_up_f32 for upper_b) to maintain soundness.
///
/// This test constructs a case where the f64 accumulated bias is NOT exactly
/// representable as f32, and verifies the rounding direction is sound:
/// - lower_b must be <= true f64 value (round toward -inf)
/// - upper_b must be >= true f64 value (round toward +inf)
#[test]
fn test_bias_f64_to_f32_rounding_direction_1992() {
    use ny_tensor::{next_down_f32, next_up_f32};

    // 100 neurons, coefficient=1.0, intercept=0.1_f32.
    //
    // f32(0.1) = 0.100000001490116119384765625 (slightly above true 0.1)
    // f64 sum of 100 * f32(0.1): 10.000000149011612 (not representable as f32)
    // f32 nearest: 10.0 (rounds DOWN via round-to-nearest-even)
    //
    // For upper_b: true value is 10.000000149, so upper_b must be >= 10.0.
    // For lower_b: true value is 10.000000149, so lower_b must be <= 10.000000149.
    let n = 100;

    fn symmetric_relax(_l: f32, _u: f32) -> LinearRelaxation {
        LinearRelaxation::new(1.0, 0.1_f32, 1.0, 0.1_f32)
    }

    let bounds = LinearBounds::new(
        Array2::from_elem((1, n), 1.0_f32),
        Array1::zeros(1),
        Array2::from_elem((1, n), 1.0_f32),
        Array1::zeros(1),
    )
    .unwrap();
    let pre = BoundedTensor::new(
        Array1::from_elem(n, -1.0_f32).into_dyn(),
        Array1::from_elem(n, 1.0_f32).into_dyn(),
    )
    .unwrap();

    let result = crown_elementwise_backward(&bounds, &pre, symmetric_relax).unwrap();

    // Reproduce the f64 accumulation that the production code does
    let intercept_f64 = 0.1_f32 as f64;
    let true_bias_f64: f64 = (0..n).map(|_| 1.0_f64 * intercept_f64).sum();
    let cast_f32 = true_bias_f64 as f32;

    // Verify our setup: true_bias_f64 is NOT exactly representable as f32
    let gap = true_bias_f64 - cast_f32 as f64;
    assert!(
        gap > 0.0,
        "Test setup error: expected f64 bias ({}) > f32 cast ({}), gap={}",
        true_bias_f64,
        cast_f32,
        gap,
    );

    // With directed rounding (#1992 fix):
    // lower_b uses next_down_f32: rounds toward -inf → sound for lower bound
    // upper_b uses next_up_f32: rounds toward +inf → sound for upper bound
    let expected_lower = next_down_f32(cast_f32);
    let expected_upper = next_up_f32(cast_f32);

    assert_eq!(
        result.lower_b[0], expected_lower,
        "lower_b must use next_down_f32 (round toward -inf)",
    );
    assert_eq!(
        result.upper_b[0], expected_upper,
        "upper_b must use next_up_f32 (round toward +inf)",
    );

    // Soundness: lower_b <= true_bias <= upper_b
    assert!(
        (result.lower_b[0] as f64) <= true_bias_f64,
        "SOUNDNESS: lower_b ({}) must be <= true bias ({})",
        result.lower_b[0],
        true_bias_f64,
    );
    assert!(
        (result.upper_b[0] as f64) >= true_bias_f64,
        "SOUNDNESS: upper_b ({}) must be >= true bias ({})",
        result.upper_b[0],
        true_bias_f64,
    );
}
