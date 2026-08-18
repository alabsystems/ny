// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward tests for AdaIN1d layer.
//!
//! Tests the CROWN scalar path (crown_common.rs) including:
//! - NaN/Inf pre-activation guard (non-finite → constant bounds, #3259)
//! - Sound mode refusal
//!
//! Part of #3103.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::types::AdaIN1dLayer;
use crate::layers::normalization::layer_norm::types::LayerNormCrownMode;
use crate::layers::normalization::InstanceNorm1dLayer;
use crate::{BatchedLinearBounds, LinearBounds};
use ny_core::NyError;

fn make_adain(num_channels: usize, style_gamma: &[f32], style_beta: &[f32]) -> AdaIN1dLayer {
    let inn = InstanceNorm1dLayer::new_default(num_channels, 1e-5).expect("valid InstanceNorm1d");
    AdaIN1dLayer::new(
        inn,
        Array1::from_vec(style_gamma.to_vec()),
        Array1::from_vec(style_beta.to_vec()),
    )
    .expect("valid AdaIN1d")
}

// ── CROWN scalar NaN/Inf pre-activation guard tests ─────────────────────────
// These test the non-finite guard in crown_common.rs which returns constant
// bounds (A=0, bias=[-inf, +inf]) when pre-activation bounds contain NaN or
// Inf. Previously returned unsound identity passthrough (#3259).

/// NaN in pre-activation lower bound triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_lower_returns_constant_bounds() {
    let layer =
        make_adain(2, &[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Sampling);

    let total = 2 * 3; // C=2, T=3
    let bounds = LinearBounds::identity(total);
    // Use new_unchecked since BoundedTensor::new rejects NaN/Inf inputs.
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![f32::NAN, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("valid shape"),
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
    let layer =
        make_adain(2, &[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Sampling);

    let total = 2 * 3; // C=2, T=3
    let bounds = LinearBounds::identity(total);
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, f32::NAN, 4.0, 5.0, 6.0, 7.0])
            .expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("NaN upper pre-activation should return constant bounds, not error");

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
    let layer =
        make_adain(2, &[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Sampling);

    let total = 2 * 3;
    let bounds = LinearBounds::identity(total);
    // Use new_unchecked since BoundedTensor::new rejects NaN/Inf inputs.
    let pre_act = BoundedTensor::new_unchecked(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![f32::NEG_INFINITY, 1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("valid shape"),
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
    let layer = make_adain(2, &[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Sound);

    let total = 2 * 3;
    let bounds = LinearBounds::identity(total);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("valid shape"),
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
    let layer = make_adain(2, &[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Cut);

    let total = 2 * 3;
    let bounds = LinearBounds::identity(total);
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("Cut mode should return identity");

    assert_eq!(result.lower_a(), bounds.lower_a());
    assert_eq!(result.upper_a(), bounds.upper_a());
}

/// Jacobian overflow (huge style_gamma, tiny std) returns NumericalInstability.
#[ntest::timeout(10000)]
#[test]
fn test_crown_sampling_jacobian_overflow_returns_numerical_instability() {
    let inn = InstanceNorm1dLayer::new(
        Array1::from_vec(vec![1e35, 1e35]),
        Array1::from_vec(vec![0.0, 0.0]),
        crate::layers::normalization::NORMALIZATION_MIN_EPS,
    )
    .expect("valid InstanceNorm1d")
    .with_crown_mode(LayerNormCrownMode::Sampling);
    let adain = AdaIN1dLayer::new(
        inn,
        Array1::from_vec(vec![1.0, 1.0]),
        Array1::from_vec(vec![0.0, 0.0]),
    )
    .expect("valid AdaIN1d");

    let total = 2 * 3;
    let bounds = LinearBounds::identity(total);
    // Nearly-constant inputs per channel → var ≈ 0, std ≈ sqrt(eps)
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![5.0; 6]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![5.0; 6]).expect("valid shape"),
    )
    .expect("valid BoundedTensor");

    let err = adain
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect_err("should return NumericalInstability for Inf Jacobian");
    assert!(
        matches!(err, NyError::NumericalInstability(_)),
        "expected NumericalInstability, got: {err}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibpvalidated_scalar_matches_effective_instance_norm_3912() -> ny_core::Result<()> {
    let ny = Array1::from_vec(vec![1.5, -0.75]);
    let beta = Array1::from_vec(vec![0.1, -0.25]);
    let style_gamma = Array1::from_vec(vec![0.5, -1.2]);
    let style_beta = Array1::from_vec(vec![0.2, 0.4]);
    let eps = 1e-5;

    let adain = AdaIN1dLayer::new(
        InstanceNorm1dLayer::new(ny, beta, eps)?,
        style_gamma,
        style_beta,
    )?
    .with_crown_mode(LayerNormCrownMode::IbpValidated);
    let effective = adain.effective_instance_norm()?;

    let bounds = LinearBounds::new(
        Array2::from_shape_vec(
            (2, 6),
            vec![
                1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.2, 0.1, 0.3, -0.4, 0.5, 0.6,
            ],
        )
        .expect("valid AdaIN scalar lower_a"),
        Array1::from_vec(vec![0.0, -0.1]),
        Array2::from_shape_vec(
            (2, 6),
            vec![
                1.0, -0.5, 0.25, 0.0, 0.75, -1.25, 0.2, 0.1, 0.3, -0.4, 0.5, 0.6,
            ],
        )
        .expect("valid AdaIN scalar upper_a"),
        Array1::from_vec(vec![0.0, -0.1]),
    )?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.25, 0.5, -0.75, 0.0, 1.0])
            .expect("valid AdaIN scalar lower"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, 1.5, 2.0, 0.25, 1.0, 2.5])
            .expect("valid AdaIN scalar upper"),
    )?;

    let adain_actual = adain.propagate_linear_with_bounds(&bounds, &pre_act)?;
    let effective_actual = effective.propagate_linear_with_bounds(&bounds, &pre_act)?;

    assert_eq!(adain_actual.lower_a(), effective_actual.lower_a());
    assert_eq!(adain_actual.lower_b(), effective_actual.lower_b());
    assert_eq!(adain_actual.upper_a(), effective_actual.upper_a());
    assert_eq!(adain_actual.upper_b(), effective_actual.upper_b());
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibpvalidated_batched_matches_effective_instance_norm_3912() -> ny_core::Result<()> {
    let ny = Array1::from_vec(vec![0.75, -1.25]);
    let beta = Array1::from_vec(vec![0.0, 0.2]);
    let style_gamma = Array1::from_vec(vec![1.4, -0.6]);
    let style_beta = Array1::from_vec(vec![0.15, -0.35]);
    let eps = 1e-5;

    let adain = AdaIN1dLayer::new(
        InstanceNorm1dLayer::new(ny, beta, eps)?,
        style_gamma,
        style_beta,
    )?
    .with_crown_mode(LayerNormCrownMode::IbpValidated);
    let effective = adain.effective_instance_norm()?;

    let bounds = BatchedLinearBounds::identity(&[2, 6])?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 6]),
            vec![
                -1.0, 0.0, 0.5, -0.25, 1.0, 2.0, -0.5, 0.25, 1.5, 0.0, 0.75, 2.25,
            ],
        )
        .expect("valid AdaIN batched lower"),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 6]),
            vec![
                0.5, 1.0, 1.5, 0.75, 2.0, 3.0, 0.25, 1.25, 2.0, 1.0, 1.5, 3.5,
            ],
        )
        .expect("valid AdaIN batched upper"),
    )?;

    let adain_actual = adain.propagate_linear_batched_with_bounds(&bounds, &pre_act)?;
    let effective_actual = effective.propagate_linear_batched_with_bounds(&bounds, &pre_act)?;

    assert_eq!(adain_actual.lower_a(), effective_actual.lower_a());
    assert_eq!(adain_actual.lower_b(), effective_actual.lower_b());
    assert_eq!(adain_actual.upper_a(), effective_actual.upper_a());
    assert_eq!(adain_actual.upper_b(), effective_actual.upper_b());
    Ok(())
}
