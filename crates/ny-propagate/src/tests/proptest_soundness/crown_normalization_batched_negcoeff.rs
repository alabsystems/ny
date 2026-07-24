// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward proptest soundness tests with non-trivial incoming
//! coefficients for normalization layers (RmsNorm, InstanceNorm1d, AdaIN1d).
//!
//! Split from crown_normalization_batched.rs to stay under the 1000-line limit.
//! These tests exercise the mixed-sign coefficient path where negative incoming
//! values flip upper/lower bound selection during CROWN backward composition —
//! a common source of soundness bugs. These containment tests use
//! `LayerNormCrownMode::IbpValidated`, the sound normalization CROWN mode.
//!
//! Part of #3175, #3820.

use crate::layers::normalization::{
    AdaIN1dLayer, InstanceNorm1dLayer, LayerNormCrownMode, RmsNormLayer,
};
use crate::BatchedLinearBounds;
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{adain_eval_channel, instance_norm_channel, rms_norm};

/// Tolerance for sampled-point containment checks.
/// Matches the other normalization CROWN proptests.
const SAMPLING_CROWN_TOLERANCE: f32 = 1e-2;

/// Concretize batched CROWN linear bounds against input interval bounds.
/// Returns (lower_bounds, upper_bounds) as Vecs of the flattened output.
fn concretize_batched_crown(
    result: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let concrete = result
        .concretize(pre_activation)
        .expect("concretize should not fail for valid bounds");
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

// =============================================================================
// RMSNORM BATCHED CROWN BACKWARD SOUNDNESS (IBP-VALIDATED, NON-TRIVIAL INCOMING)
// =============================================================================
//
// Tests the batched CROWN backward path with mixed-sign incoming coefficients.
// Negative coefficients flip upper/lower bound selection — a common source of
// soundness bugs in CROWN backward composition.
//
// Part of #3175.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// RmsNorm batched CROWN soundness with non-trivial incoming coefficients.
    /// Uses a single output row with mixed-sign coefficients per batch position,
    /// verifying that the composed CROWN bounds contain the true output.
    /// Part of #3175.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_rmsnorm_batched_crown_negcoeff(
        // Batch 0 centers
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        // Batch 1 centers
        c3 in -2.0f32..2.0,
        c4 in -2.0f32..2.0,
        c5 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Non-trivial incoming coefficients (shared across batch positions)
        ic0 in -2.0f32..2.0,
        ic1 in -2.0f32..2.0,
        ic2 in -2.0f32..2.0,
    ) {
        // Ensure at least one coefficient is non-trivial.
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01);

        let norm_size = 3;
        let batch = 2;
        let out_dim = 1; // single output row per batch
        let ny = Array1::ones(norm_size);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), upper_v.clone()).unwrap(),
        ).unwrap();

        let rn = RmsNormLayer::new(ny.clone(), eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        // Non-trivial batched bounds: [batch, out_dim=1, norm_size]
        let coeffs = [ic0, ic1, ic2];
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, norm_size]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, norm_size]));
        for b in 0..batch {
            for i in 0..norm_size {
                la[[b, 0, i]] = coeffs[i];
                ua[[b, 0, i]] = coeffs[i];
            }
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![batch, norm_size],
            vec![batch, out_dim],
        ).unwrap();

        let result = rn
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "RmsNorm batched CROWN negcoeff failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        // Sample 50 random points and check each batch independently
        for s in 0..50_u32 {
            let sample: Vec<f32> = (0..6)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            for b in 0..batch {
                let x_batch = arr1(&sample[b * norm_size..(b + 1) * norm_size]);
                let y_true = rms_norm(&x_batch, &ny, eps);
                // Apply incoming coefficients: combined = ic . y_true
                let combined = ic0 * y_true[0] + ic1 * y_true[1] + ic2 * y_true[2];
                let idx = b; // out_dim=1, so index is just batch

                prop_assert!(
                    combined >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched RmsNorm CROWN negcoeff lower at batch {b}: {combined} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    combined <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched RmsNorm CROWN negcoeff upper at batch {b}: {combined} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}

// =============================================================================
// INSTANCENORM1D BATCHED CROWN BACKWARD SOUNDNESS (IBP-VALIDATED, NON-TRIVIAL INCOMING)
// =============================================================================
//
// InstanceNorm1d operates on [C, T] per batch position. Uses 2 channels ×
// 2 timesteps = in_dim=4. Non-trivial incoming coefficients are applied across
// the full flattened dimension.
//
// Part of #3175.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// InstanceNorm1d batched CROWN soundness with non-trivial incoming coefficients.
    /// Uses 2 channels, 2 timesteps per channel (in_dim=4), batch=2, out_dim=1.
    /// Part of #3175.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_instancenorm_batched_crown_negcoeff(
        // Batch 0: 4 neurons (2 channels × 2 timesteps)
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0,
        // Batch 1: 4 neurons
        c4 in -2.0f32..2.0,
        c5 in -2.0f32..2.0,
        c6 in -2.0f32..2.0,
        c7 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Non-trivial incoming coefficients (4 for in_dim=4)
        ic0 in -2.0f32..2.0,
        ic1 in -2.0f32..2.0,
        ic2 in -2.0f32..2.0,
        ic3 in -2.0f32..2.0,
    ) {
        // Ensure at least one coefficient is non-trivial.
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01 || ic3.abs() > 0.01);

        let num_channels = 2;
        let time_len = 2;
        let in_dim = num_channels * time_len; // 4
        let batch = 2;
        let out_dim = 1;
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5, c6, c7];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, in_dim]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, in_dim]), upper_v.clone()).unwrap(),
        ).unwrap();

        let inn = InstanceNorm1dLayer::new(
            Array1::ones(num_channels),
            Array1::zeros(num_channels),
            eps,
        )
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::IbpValidated);

        // Non-trivial batched bounds: [batch, out_dim=1, in_dim=4]
        let coeffs = [ic0, ic1, ic2, ic3];
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        for b in 0..batch {
            for i in 0..in_dim {
                la[[b, 0, i]] = coeffs[i];
                ua[[b, 0, i]] = coeffs[i];
            }
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![batch, in_dim],
            vec![batch, out_dim],
        ).unwrap();

        let result = inn
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "InstanceNorm1d batched CROWN negcoeff failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        // Sample 50 random points and check each batch independently
        for s in 0..50_u32 {
            let sample: Vec<f32> = (0..8)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            for b in 0..batch {
                let batch_offset = b * in_dim;
                // Evaluate InstanceNorm per channel and compute dot product
                let mut y_true: Vec<f32> = Vec::with_capacity(in_dim);
                for c in 0..num_channels {
                    let start = batch_offset + c * time_len;
                    let x_ch = arr1(&sample[start..start + time_len]);
                    let y_ch = instance_norm_channel(&x_ch, 1.0, 0.0, eps);
                    y_true.extend(y_ch.iter());
                }

                let combined = ic0 * y_true[0] + ic1 * y_true[1] + ic2 * y_true[2] + ic3 * y_true[3];
                let idx = b; // out_dim=1, so index is just batch

                prop_assert!(
                    combined >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched InstanceNorm CROWN negcoeff lower at batch {b}: {combined} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    combined <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched InstanceNorm CROWN negcoeff upper at batch {b}: {combined} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}

// =============================================================================
// ADAIN1D BATCHED CROWN BACKWARD SOUNDNESS (IBP-VALIDATED, NON-TRIVIAL INCOMING)
// =============================================================================
//
// AdaIN1d wraps InstanceNorm1d and applies style_gamma * norm(x) + style_beta.
// Uses the same [C, T] layout: 2 channels, 2 timesteps, in_dim=4, batch=2.
// Non-trivial incoming coefficients exercise the composed backward path.
//
// Part of #3175.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// AdaIN1d batched CROWN soundness with non-trivial incoming coefficients.
    /// Uses 2 channels, 2 timesteps per channel (in_dim=4), batch=2, out_dim=1.
    /// Part of #3175.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_adain_batched_crown_negcoeff(
        // Batch 0: 4 neurons (2 channels × 2 timesteps)
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0,
        // Batch 1: 4 neurons
        c4 in -2.0f32..2.0,
        c5 in -2.0f32..2.0,
        c6 in -2.0f32..2.0,
        c7 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Style ny/beta (per channel)
        sg0 in 0.5f32..2.0,
        sg1 in 0.5f32..2.0,
        sb0 in -1.0f32..1.0,
        sb1 in -1.0f32..1.0,
        // Non-trivial incoming coefficients (4 for in_dim=4)
        ic0 in -2.0f32..2.0,
        ic1 in -2.0f32..2.0,
        ic2 in -2.0f32..2.0,
        ic3 in -2.0f32..2.0,
    ) {
        // Ensure at least one coefficient is non-trivial.
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01 || ic3.abs() > 0.01);

        let num_channels = 2;
        let time_len = 2;
        let in_dim = num_channels * time_len; // 4
        let batch = 2;
        let out_dim = 1;
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5, c6, c7];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, in_dim]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, in_dim]), upper_v.clone()).unwrap(),
        ).unwrap();

        let inn = InstanceNorm1dLayer::new(
            Array1::ones(num_channels),
            Array1::zeros(num_channels),
            eps,
        )
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let adain = AdaIN1dLayer::new(
            inn,
            Array1::from_vec(vec![sg0, sg1]),
            Array1::from_vec(vec![sb0, sb1]),
        )
        .unwrap();

        // Non-trivial batched bounds: [batch, out_dim=1, in_dim=4]
        let coeffs = [ic0, ic1, ic2, ic3];
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        for b in 0..batch {
            for i in 0..in_dim {
                la[[b, 0, i]] = coeffs[i];
                ua[[b, 0, i]] = coeffs[i];
            }
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![batch, in_dim],
            vec![batch, out_dim],
        ).unwrap();

        let result = adain
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "AdaIN1d batched CROWN negcoeff failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        let style_gammas = [sg0, sg1];
        let style_betas = [sb0, sb1];

        // Sample 50 random points and check each batch independently
        for s in 0..50_u32 {
            let sample: Vec<f32> = (0..8)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            for b in 0..batch {
                let batch_offset = b * in_dim;
                // Evaluate AdaIN per channel: style_gamma * InstanceNorm(x) + style_beta
                let mut y_true: Vec<f32> = Vec::with_capacity(in_dim);
                for c in 0..num_channels {
                    let start = batch_offset + c * time_len;
                    let x_ch = arr1(&sample[start..start + time_len]);
                    let y_ch = adain_eval_channel(
                        &x_ch,
                        1.0,
                        0.0,
                        style_gammas[c],
                        style_betas[c],
                        eps,
                    );
                    y_true.extend(y_ch.iter());
                }

                let combined = ic0 * y_true[0] + ic1 * y_true[1] + ic2 * y_true[2] + ic3 * y_true[3];
                let idx = b; // out_dim=1, so index is just batch

                prop_assert!(
                    combined >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched AdaIN CROWN negcoeff lower at batch {b}: {combined} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    combined <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched AdaIN CROWN negcoeff upper at batch {b}: {combined} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}
