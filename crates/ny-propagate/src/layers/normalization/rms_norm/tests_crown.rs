// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward tests for RmsNorm layer.
//!
//! Tests the CROWN scalar path (crown_common.rs) including:
//! - NaN/Inf pre-activation guard (non-finite → constant bounds, #3259)
//! - Sound mode refusal
//! - Cut mode identity passthrough
//! - Jacobian overflow → NumericalInstability
//!
//! Part of #3103.

use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::NyError;
use ny_tensor::BoundedTensor;

use super::types::RmsNormLayer;
use crate::layers::common::BoundPropagation;
use crate::layers::normalization::layer_norm::types::LayerNormCrownMode;
use crate::LinearBounds;

// ── CROWN scalar NaN/Inf pre-activation guard tests ─────────────────────────
// These test the non-finite guard in crown_common.rs which returns constant
// bounds (A=0, bias=[-inf, +inf]) when pre-activation bounds contain NaN or
// Inf. Previously returned unsound identity passthrough (#3259).

/// NaN in pre-activation lower bound triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_lower_returns_constant_bounds() {
    let layer = RmsNormLayer::new(arr1(&[1.0, 1.0, 1.0]), 1e-5)
        .expect("valid RmsNorm")
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let bounds = LinearBounds::identity(3);
    // Use new_unchecked since BoundedTensor::new rejects NaN/Inf inputs.
    // We need NaN to reach the non-finite guard inside crown_common.rs.
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![f32::NAN, 1.0, 2.0]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).expect("valid shape"),
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

/// NaN in pre-activation upper bound triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_upper_returns_constant_bounds() {
    let layer = RmsNormLayer::new(arr1(&[2.0, 0.5]), 1e-5)
        .expect("valid RmsNorm")
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
    let layer = RmsNormLayer::new(arr1(&[1.0, 1.0, 1.0]), 1e-5)
        .expect("valid RmsNorm")
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

/// Sound mode returns SoundnessRefusal error.
#[ntest::timeout(10000)]
#[test]
fn test_crown_sound_mode_returns_soundness_refusal() {
    let layer = RmsNormLayer::new(arr1(&[1.0, 1.0, 1.0]), 1e-5)
        .expect("valid RmsNorm")
        .with_crown_mode(LayerNormCrownMode::Sound);

    let bounds = LinearBounds::identity(3);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect_err("Sound mode should refuse");
    assert!(
        matches!(err, NyError::SoundnessRefusal(_)),
        "expected SoundnessRefusal, got: {err}"
    );
}

/// Cut mode returns identity relaxation (passthrough).
#[ntest::timeout(10000)]
#[test]
fn test_crown_cut_mode_returns_identity() {
    let layer = RmsNormLayer::new(arr1(&[1.0, 1.0, 1.0]), 1e-5)
        .expect("valid RmsNorm")
        .with_crown_mode(LayerNormCrownMode::Cut);

    let bounds = LinearBounds::identity(3);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("Cut mode should return identity");

    assert_eq!(result.lower_a(), bounds.lower_a());
    assert_eq!(result.upper_a(), bounds.upper_a());
}

/// IBP-validated margins widen CROWN bounds for near-zero inputs (#3162).
///
/// The #3162 regression seed (c0=0, c1=-0.131, c2=0, hw0=0.05, hw1=0.22, hw2=0.05)
/// has near-zero inputs where the RMS denominator is small and the function
/// has extreme curvature. Without IBP validation, sampling-only margins are
/// insufficient and CROWN returns unsound bounds.
///
/// This test verifies:
/// 1. The IBP margin path is actually exercised (not silently skipped)
/// 2. The CROWN lower bound is below the actual function value at the worst corner
#[ntest::timeout(10000)]
#[test]
fn test_crown_ibp_margin_widens_near_zero_inputs_3162() {
    let ny = arr1(&[0.5, 0.5, 0.5]);
    let eps = 1e-5_f32;
    let layer = RmsNormLayer::new(ny.clone(), eps)
        .expect("valid RmsNorm")
        .with_crown_mode(LayerNormCrownMode::IbpValidated);

    // #3162 regression seed: near-zero centers with small half-widths
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![-0.05, -0.351, -0.05]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.05, 0.089, 0.05]).expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let identity = LinearBounds::identity(3);
    let result = layer
        .propagate_linear_with_bounds(&identity, &pre_act)
        .expect("should succeed");

    let concrete = result.concretize(&pre_act);

    // IBP must also succeed for this input
    let ibp = layer
        .propagate_ibp(&pre_act)
        .expect("IBP should succeed for finite inputs");

    // CROWN bounds must be at least as wide as IBP (the fix guarantees this)
    let tol = 1e-4;
    for i in 0..3 {
        assert!(
            concrete.lower()[[i]] <= ibp.lower()[[i]] + tol,
            "dim {i}: CROWN lower {} should be <= IBP lower {} + tol",
            concrete.lower()[[i]],
            ibp.lower()[[i]]
        );
        assert!(
            concrete.upper()[[i]] >= ibp.upper()[[i]] - tol,
            "dim {i}: CROWN upper {} should be >= IBP upper {} - tol",
            concrete.upper()[[i]],
            ibp.upper()[[i]]
        );
    }

    // The specific corner that was unsound before the fix
    let x_corner = arr1(&[-0.05_f32, -0.02111736, -0.025]);
    let n = x_corner.len() as f32;
    let rms = (x_corner.iter().map(|&xi| xi * xi).sum::<f32>() / n + eps).sqrt();
    let rmsnorm_val: Vec<f32> = x_corner
        .iter()
        .zip(ny.iter())
        .map(|(&xi, &g)| g * xi / rms)
        .collect();

    for (i, &val) in rmsnorm_val.iter().enumerate() {
        assert!(
            concrete.lower()[[i]] - tol <= val,
            "dim {i}: CROWN lower {} > actual rmsnorm {val} — unsound!",
            concrete.lower()[[i]],
        );
        assert!(
            concrete.upper()[[i]] + tol >= val,
            "dim {i}: CROWN upper {} < actual rmsnorm {val} — unsound!",
            concrete.upper()[[i]],
        );
    }
}

/// Jacobian overflow (huge ny, tiny std) returns NumericalInstability.
#[ntest::timeout(10000)]
#[test]
fn test_crown_sampling_jacobian_overflow_returns_numerical_instability() {
    // ny = 1e35, eps = minimum → ny/rms overflows f32 to Inf.
    let layer = RmsNormLayer::new(arr1(&[1e35, 1e35, 1e35]), 0.0) // eps clamped to 1e-12
        .expect("valid RmsNorm")
        .with_crown_mode(LayerNormCrownMode::Sampling);

    let bounds = LinearBounds::identity(3);
    // Nearly-zero inputs → rms ≈ sqrt(eps)
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1e-20, 1e-20, 1e-20]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1e-20, 1e-20, 1e-20]).expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect_err("should return NumericalInstability for Inf Jacobian");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got: {err}"
    );
}
