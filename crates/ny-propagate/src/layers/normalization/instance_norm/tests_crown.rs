// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward soundness tests for InstanceNorm1d.
//!
//! Tests both the sampling-based fallback path and the decomposed
//! `IbpValidated` path introduced in #3830.

use super::types::InstanceNorm1dLayer;
use crate::layers::normalization::decomposed::decomposed_instance_norm_crown_backward;
use crate::layers::normalization::trait_norm::NormLayer;
use crate::layers::normalization::LayerNormCrownMode;
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{arr1, arr2, Array1, ArrayD, Ix1, Ix2, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

fn custom_in1d(ny: &[f32], beta: &[f32]) -> InstanceNorm1dLayer {
    InstanceNorm1dLayer::new(
        Array1::from_vec(ny.to_vec()),
        Array1::from_vec(beta.to_vec()),
        1e-5,
    )
    .expect("valid custom InstanceNorm1d")
}

/// Helper: run CROWN backward with identity bounds, concretize, sample, verify.
#[allow(clippy::too_many_arguments)] // test helper with explicit per-parameter semantics
fn verify_crown_soundness(
    layer: &InstanceNorm1dLayer,
    num_channels: usize,
    time_len: usize,
    lower_vals: &[f32],
    upper_vals: &[f32],
    num_samples: u32,
    tolerance: f32,
    label: &str,
) {
    let total = num_channels * time_len;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), lower_vals.to_vec())
            .expect("valid lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), upper_vals.to_vec())
            .expect("valid upper shape"),
    )
    .expect("valid BoundedTensor");

    let bounds = LinearBounds::identity(total);
    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("Sampling CROWN should succeed");

    let input_flat = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower_vals.to_vec()).expect("valid lower flat"),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper_vals.to_vec()).expect("valid upper flat"),
    )
    .expect("valid flat BoundedTensor");
    let concrete = result.concretize(&input_flat);

    for s in 0..num_samples {
        let sample: Vec<f32> = (0..total)
            .map(|i| {
                let t = ((s.wrapping_mul(2654435761) ^ (i as u32)).wrapping_mul(2654435761)) as f32
                    / u32::MAX as f32;
                lower_vals[i] + (upper_vals[i] - lower_vals[i]) * t
            })
            .collect();

        let mut y_flat: Vec<f32> = Vec::with_capacity(total);
        for c in 0..num_channels {
            let start = c * time_len;
            let channel_input = arr1(&sample[start..start + time_len]);
            let y_channel = layer.eval_channel(&channel_input, c).expect("eval_channel");
            y_flat.extend(y_channel.iter());
        }

        for (i, &y_val) in y_flat.iter().enumerate().take(total) {
            assert!(
                concrete.lower()[[i]] <= y_val + tolerance,
                "{label}: lower violated dim {i} sample {s}: {} > {}",
                concrete.lower()[[i]],
                y_val
            );
            assert!(
                concrete.upper()[[i]] >= y_val - tolerance,
                "{label}: upper violated dim {i} sample {s}: {} < {}",
                concrete.upper()[[i]],
                y_val
            );
        }
    }
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

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibpvalidated_scalar_matches_decomposed_helper_3830() -> Result<()> {
    let layer =
        custom_in1d(&[1.5, -0.75], &[0.1, -0.25]).with_crown_mode(LayerNormCrownMode::IbpValidated);
    let bounds = LinearBounds::new(
        arr2(&[
            [1.0, -0.5, 0.25, 0.0, 0.75, -1.25],
            [0.2, 0.1, 0.3, -0.4, 0.5, 0.6],
        ]),
        arr1(&[0.0, -0.1]),
        arr2(&[
            [1.0, -0.5, 0.25, 0.0, 0.75, -1.25],
            [0.2, 0.1, 0.3, -0.4, 0.5, 0.6],
        ]),
        arr1(&[0.0, -0.1]),
    )?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.25, 0.5, -0.75, 0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.5, 1.5, 2.0, 0.25, 1.0, 2.5]).unwrap(),
    )?;

    let actual = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;
    let expected = decomposed_instance_norm_crown_backward(
        &scalar_bounds_to_batched_for_test(&bounds),
        &layer.ny,
        &layer.beta,
        layer.eps,
        &pre_act,
        layer.forward_mode,
        layer.num_channels(),
    )?;
    let expected_scalar = batched_bounds_to_scalar_for_test(&expected.bounds);

    assert_eq!(actual.lower_a(), expected_scalar.lower_a());
    assert_eq!(actual.lower_b(), expected_scalar.lower_b());
    assert_eq!(actual.upper_a(), expected_scalar.upper_a());
    assert_eq!(actual.upper_b(), expected_scalar.upper_b());
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibpvalidated_batched_matches_decomposed_helper_3830() -> Result<()> {
    let layer =
        custom_in1d(&[0.75, -1.25], &[0.0, 0.2]).with_crown_mode(LayerNormCrownMode::IbpValidated);
    let bounds = BatchedLinearBounds::identity(&[2, 6])?;
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[2, 6]),
            vec![
                -1.0, 0.0, 0.5, -0.25, 1.0, 2.0, -0.5, 0.25, 1.5, 0.0, 0.75, 2.25,
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[2, 6]),
            vec![
                0.5, 1.0, 1.5, 0.75, 2.0, 3.0, 0.25, 1.25, 2.0, 1.0, 1.5, 3.5,
            ],
        )
        .unwrap(),
    )?;

    let actual = layer.propagate_linear_batched_with_bounds(&bounds, &pre_act)?;
    let expected = decomposed_instance_norm_crown_backward(
        &bounds,
        &layer.ny,
        &layer.beta,
        layer.eps,
        &pre_act,
        layer.forward_mode,
        layer.num_channels(),
    )?;

    assert_eq!(actual.lower_a(), expected.bounds.lower_a());
    assert_eq!(actual.lower_b(), expected.bounds.lower_b());
    assert_eq!(actual.upper_a(), expected.bounds.upper_a());
    assert_eq!(actual.upper_b(), expected.bounds.upper_b());
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_crown_ibpvalidated_per_channel_independence_3830() -> Result<()> {
    let layer =
        custom_in1d(&[1.0, 0.5], &[0.0, 0.25]).with_crown_mode(LayerNormCrownMode::IbpValidated);
    let bounds = LinearBounds::identity(6);

    let pre_act_a = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-1.0, 0.0, 1.0, -0.5, 0.25, 1.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 0.5, 1.25, 2.5]).unwrap(),
    )?;
    let pre_act_b = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-3.0, -2.0, -1.0, -0.5, 0.25, 1.5]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![-2.0, -1.0, 0.0, 0.5, 1.25, 2.5]).unwrap(),
    )?;

    let result_a = layer.propagate_linear_with_bounds(&bounds, &pre_act_a)?;
    let result_b = layer.propagate_linear_with_bounds(&bounds, &pre_act_b)?;

    for row in 3..6 {
        for col in 0..3 {
            assert_eq!(
                result_a.lower_a()[[row, col]],
                0.0,
                "channel 1 row {row} should not depend on channel 0 col {col}"
            );
            assert_eq!(
                result_a.upper_a()[[row, col]],
                0.0,
                "channel 1 row {row} should not depend on channel 0 col {col}"
            );
        }
        for col in 3..6 {
            assert_eq!(
                result_a.lower_a()[[row, col]],
                result_b.lower_a()[[row, col]]
            );
            assert_eq!(
                result_a.upper_a()[[row, col]],
                result_b.upper_a()[[row, col]]
            );
        }
        assert_eq!(result_a.lower_b()[row], result_b.lower_b()[row]);
        assert_eq!(result_a.upper_b()[row], result_b.upper_b()[row]);
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(128) })]

    #[ntest::timeout(60000)]
    #[test]
    fn proptest_decomposed_instancenorm_crown_contains_forward_output_3830(
        c0 in -1.5f32..1.5,
        c1 in -1.5f32..1.5,
        c2 in -1.5f32..1.5,
        c3 in -1.5f32..1.5,
        c4 in -1.5f32..1.5,
        c5 in -1.5f32..1.5,
        hw in 0.05f32..0.25,
        g0 in -2.0f32..2.0,
        g1 in -2.0f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
    ) {
        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let pre_act = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_v.clone()).unwrap(),
        ).unwrap();
        let flat_input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[6]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[6]), upper_v.clone()).unwrap(),
        ).unwrap();
        let layer = InstanceNorm1dLayer::new(
            Array1::from_vec(vec![g0, g1]),
            Array1::from_vec(vec![b0, b1]),
            1e-5,
        )
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let helper = decomposed_instance_norm_crown_backward(
            &scalar_bounds_to_batched_for_test(&LinearBounds::identity(6)),
            &layer.ny,
            &layer.beta,
            layer.eps,
            &pre_act,
            layer.forward_mode,
            layer.num_channels(),
        )
        .map_err(|e| {
            TestCaseError::fail(format!(
                "decomposed InstanceNorm helper failed: {e}"
            ))
        })?;
        let concrete = helper
            .bounds
            .concretize_sound(&flat_input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "decomposed InstanceNorm concretize failed: {e}"
                ))
            })?;

        for s in 0..16_u32 {
            let sample: Vec<f32> = (0..6)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();
            let y = layer.eval(&arr1(&sample)).expect("eval should succeed");

            for i in 0..6 {
                prop_assert!(
                    concrete.lower()[[i]] <= y[i] + 1e-4,
                    "decomposed helper lower violation at dim {i}: {} > {}",
                    concrete.lower()[[i]],
                    y[i]
                );
                prop_assert!(
                    concrete.upper()[[i]] >= y[i] - 1e-4,
                    "decomposed helper upper violation at dim {i}: {} < {}",
                    concrete.upper()[[i]],
                    y[i]
                );
            }
        }
    }
}

/// CROWN backward soundness with identity incoming bounds (Sampling mode).
#[test]
fn test_crown_backward_soundness_identity_sampling() {
    let layer =
        custom_in1d(&[2.0, 0.5], &[1.0, -1.0]).with_crown_mode(LayerNormCrownMode::Sampling);

    verify_crown_soundness(
        &layer,
        2,
        3,
        &[-1.0, -2.0, 0.0, 1.0, -1.0, 0.5],
        &[1.0, 0.0, 2.0, 3.0, 1.0, 2.5],
        200,
        1e-3,
        "identity",
    );
}

/// CROWN backward soundness with negative coefficients (A = -I).
#[test]
fn test_crown_backward_soundness_negative_coeff() {
    use ndarray::Array2;

    let layer = custom_in1d(&[1.5], &[0.0]).with_crown_mode(LayerNormCrownMode::Sampling);

    let time_len = 4;
    let total = time_len;
    let lower_vals = vec![-2.0_f32, -1.0, 0.0, 1.0];
    let upper_vals = vec![0.0_f32, 1.0, 2.0, 3.0];

    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, time_len]), lower_vals.clone())
            .expect("valid lower shape"),
        ArrayD::from_shape_vec(IxDyn(&[1, time_len]), upper_vals.clone())
            .expect("valid upper shape"),
    )
    .expect("valid BoundedTensor");

    let neg_eye = Array2::from_elem((total, total), 0.0_f32) - Array2::<f32>::eye(total);
    let bounds = LinearBounds::new(
        neg_eye.clone(),
        Array1::zeros(total),
        neg_eye,
        Array1::zeros(total),
    )
    .expect("valid LinearBounds");

    let result = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("Sampling CROWN should succeed");

    let input_flat = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower_vals.clone()).expect("valid lower flat"),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper_vals.clone()).expect("valid upper flat"),
    )
    .expect("valid flat BoundedTensor");
    let concrete = result.concretize(&input_flat);

    for s in 0..100_u32 {
        let sample: Vec<f32> = (0..total)
            .map(|i| {
                let t = ((s.wrapping_mul(2654435761) ^ (i as u32)).wrapping_mul(2654435761)) as f32
                    / u32::MAX as f32;
                lower_vals[i] + (upper_vals[i] - lower_vals[i]) * t
            })
            .collect();

        let y = layer.eval_channel(&arr1(&sample), 0).expect("eval_channel");
        for i in 0..total {
            let negated = -y[i];
            assert!(
                concrete.lower()[[i]] <= negated + 1e-3,
                "neg coeff lower violated"
            );
            assert!(
                concrete.upper()[[i]] >= negated - 1e-3,
                "neg coeff upper violated"
            );
        }
    }
}

/// CROWN backward soundness with randomized configs.
///
/// Uses 1e-2 tolerance for sampling-based linearization (50 internal
/// samples, 1.1x safety margin). See crown_scalar.rs.
#[test]
fn test_crown_backward_soundness_multi_config() {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42_3053);

    let configs: Vec<(Vec<f32>, Vec<f32>, usize)> = vec![
        (vec![1.0, 1.0], vec![0.0, 0.0], 5),
        (vec![1.5, -1.0], vec![0.5, -0.5], 4),
        (vec![0.5, 0.5, 0.5], vec![0.0; 3], 3),
        (vec![2.0], vec![-1.0], 6),
    ];

    for (ny, beta, time_len) in &configs {
        let nc = ny.len();
        let tl = *time_len;
        let total = nc * tl;

        let layer = custom_in1d(ny, beta).with_crown_mode(LayerNormCrownMode::Sampling);

        let center: Vec<f32> = (0..total).map(|_| rng.random_range(-2.0..2.0)).collect();
        let half_w: Vec<f32> = (0..total).map(|_| rng.random_range(0.1..0.5)).collect();

        let lower_vals: Vec<f32> = center.iter().zip(&half_w).map(|(&c, &h)| c - h).collect();
        let upper_vals: Vec<f32> = center.iter().zip(&half_w).map(|(&c, &h)| c + h).collect();

        verify_crown_soundness(
            &layer,
            nc,
            tl,
            &lower_vals,
            &upper_vals,
            300,
            1e-2,
            &format!("C={nc} T={tl} ny={ny:?}"),
        );
    }
}

// ── CROWN scalar NaN/Inf pre-activation guard tests ─────────────────────────
// These test the non-finite guard in crown_common.rs which returns constant
// bounds (A=0, bias=[-inf, +inf]) when pre-activation bounds contain NaN or
// Inf. Previously returned unsound identity passthrough (#3259).

/// NaN in pre-activation upper bound triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_upper_returns_constant_bounds() {
    let layer = custom_in1d(&[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Sampling);

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

/// Cut mode returns identity relaxation (passthrough).
#[ntest::timeout(10000)]
#[test]
fn test_crown_cut_mode_returns_identity() {
    let layer = custom_in1d(&[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Cut);

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

/// NaN in pre-activation lower bound triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_nan_pre_activation_lower_returns_constant_bounds() {
    let layer = custom_in1d(&[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Sampling);

    let total = 2 * 3; // C=2, T=3
    let bounds = LinearBounds::identity(total);
    // NaN in lower bound of channel 0, element 0.
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

/// Inf in pre-activation bounds triggers constant bounds fallback (#3259).
#[ntest::timeout(10000)]
#[test]
fn test_crown_scalar_inf_pre_activation_returns_constant_bounds() {
    let layer = custom_in1d(&[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Sampling);

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
    let layer = custom_in1d(&[1.0, 1.0], &[0.0, 0.0]).with_crown_mode(LayerNormCrownMode::Sound);

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

/// Jacobian overflow returns NumericalInstability.
#[ntest::timeout(10000)]
#[test]
fn test_crown_sampling_jacobian_overflow_returns_numerical_instability() {
    let layer = InstanceNorm1dLayer::new(
        Array1::from_vec(vec![1e35, 1e35]),
        Array1::from_vec(vec![0.0, 0.0]),
        0.0, // eps clamped to 1e-12
    )
    .expect("valid InstanceNorm1d")
    .with_crown_mode(LayerNormCrownMode::Sampling);

    let total = 2 * 3;
    let bounds = LinearBounds::identity(total);
    // Nearly-constant inputs per channel → var ≈ 0, std ≈ sqrt(eps)
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![5.0; 6]).expect("valid shape"),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![5.0; 6]).expect("valid shape"),
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
