// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Batched CROWN backward proptest soundness tests with asymmetric incoming
//! coefficients (lower_a != upper_a) for normalization layers.
//!
//! The asymmetric coefficient path exercises sign-switching logic where lower
//! and upper bounds choose different relaxation slopes. This is historically
//! where soundness bugs occur in CROWN backward composition.
//!
//! Covers: RmsNorm, InstanceNorm1d, AdaIN1d, GroupNorm.
//! LayerNorm already has batched asymmetric coverage in crown_normalization_layernorm.rs.
//! All batched containment tests in this file use
//! `LayerNormCrownMode::IbpValidated`, the sound normalization CROWN mode.
//!
//! Part of #3284.

use crate::layers::normalization::{
    AdaIN1dLayer, GroupNormLayer, InstanceNorm1dLayer, LayerNormCrownMode, RmsNormLayer,
};
use crate::BatchedLinearBounds;
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{adain_eval_channel, group_norm_group, instance_norm_channel, rms_norm};

/// Tolerance for sampled-point containment checks.
/// Matching crown_normalization.rs and crown_normalization_batched.rs.
const SAMPLING_CROWN_TOLERANCE: f32 = 1e-2;

/// Concretize batched CROWN linear bounds against input interval bounds.
/// Returns (lower_bounds, upper_bounds) as Vecs of the flattened output.
fn concretize_batched_crown(
    result: &BatchedLinearBounds,
    pre_activation: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let concrete = result
        .concretize(pre_activation)
        .expect("concretize should not fail for IbpValidated normalization bounds");
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

// =============================================================================
// RMSNORM BATCHED CROWN ASYMMETRIC (lower_a != upper_a)
// =============================================================================
//
// RmsNorm: norm_size=3, batch=2, out_dim=1.
// Asymmetric incoming: different coefficients for lower and upper bounds.
//
// Part of #3284.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// RmsNorm batched CROWN soundness with asymmetric incoming (la != ua).
    /// Part of #3284.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_rmsnorm_batched_crown_asymmetric(
        // Batch 0 centers
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        // Batch 1 centers
        c3 in -2.0f32..2.0,
        c4 in -2.0f32..2.0,
        c5 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Lower incoming coefficients
        cl0 in -2.0f32..2.0,
        cl1 in -2.0f32..2.0,
        cl2 in -2.0f32..2.0,
        // Upper incoming coefficients (different from lower)
        cu0 in -2.0f32..2.0,
        cu1 in -2.0f32..2.0,
        cu2 in -2.0f32..2.0,
    ) {
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01);
        prop_assume!(
            (cl0 - cu0).abs() > 0.01
                || (cl1 - cu1).abs() > 0.01
                || (cl2 - cu2).abs() > 0.01
        );

        let norm_size = 3;
        let batch = 2;
        let out_dim = 1;
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

        // Asymmetric batched bounds: la != ua, [batch, out_dim=1, norm_size]
        let cl = [cl0, cl1, cl2];
        let cu = [cu0, cu1, cu2];
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, norm_size]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, norm_size]));
        for b in 0..batch {
            for i in 0..norm_size {
                la[[b, 0, i]] = cl[i];
                ua[[b, 0, i]] = cu[i];
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
                    "RmsNorm batched CROWN asymmetric failed: {e}"
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
                // Lower bound uses cl coefficients, upper uses cu
                let lower_val = cl0 * y_true[0] + cl1 * y_true[1] + cl2 * y_true[2];
                let upper_val = cu0 * y_true[0] + cu1 * y_true[1] + cu2 * y_true[2];
                let idx = b;

                prop_assert!(
                    lower_val >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched RmsNorm CROWN asymmetric lower at batch {b}: {lower_val} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    upper_val <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched RmsNorm CROWN asymmetric upper at batch {b}: {upper_val} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}

// =============================================================================
// INSTANCENORM1D BATCHED CROWN ASYMMETRIC (lower_a != upper_a, IBP-VALIDATED)
// =============================================================================
//
// InstanceNorm1d operates on [C, T] per batch position. Uses 2 channels x
// 2 timesteps = in_dim=4. Asymmetric incoming coefficients are applied across
// the full flattened dimension.
//
// Part of #3284.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// InstanceNorm1d batched CROWN soundness with asymmetric incoming (la != ua).
    /// Uses 2 channels, 2 timesteps per channel (in_dim=4), batch=2, out_dim=1.
    /// Part of #3284.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_instancenorm_batched_crown_asymmetric(
        // Batch 0: 4 neurons (2 channels x 2 timesteps)
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
        // Lower incoming coefficients (4 for in_dim=4)
        cl0 in -2.0f32..2.0,
        cl1 in -2.0f32..2.0,
        cl2 in -2.0f32..2.0,
        cl3 in -2.0f32..2.0,
        // Upper incoming coefficients (different from lower)
        cu0 in -2.0f32..2.0,
        cu1 in -2.0f32..2.0,
        cu2 in -2.0f32..2.0,
        cu3 in -2.0f32..2.0,
    ) {
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01 || cl3.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01 || cu3.abs() > 0.01);
        prop_assume!(
            (cl0 - cu0).abs() > 0.01
                || (cl1 - cu1).abs() > 0.01
                || (cl2 - cu2).abs() > 0.01
                || (cl3 - cu3).abs() > 0.01
        );

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

        // Asymmetric batched bounds: la != ua, [batch, out_dim=1, in_dim=4]
        let cl = [cl0, cl1, cl2, cl3];
        let cu = [cu0, cu1, cu2, cu3];
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        for b in 0..batch {
            for i in 0..in_dim {
                la[[b, 0, i]] = cl[i];
                ua[[b, 0, i]] = cu[i];
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
                    "InstanceNorm1d batched CROWN asymmetric failed: {e}"
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

                // Lower bound uses cl coefficients, upper uses cu
                let lower_val = cl0 * y_true[0] + cl1 * y_true[1] + cl2 * y_true[2] + cl3 * y_true[3];
                let upper_val = cu0 * y_true[0] + cu1 * y_true[1] + cu2 * y_true[2] + cu3 * y_true[3];
                let idx = b;

                prop_assert!(
                    lower_val >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched InstanceNorm CROWN asymmetric lower at batch {b}: {lower_val} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    upper_val <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched InstanceNorm CROWN asymmetric upper at batch {b}: {upper_val} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}

// =============================================================================
// ADAIN1D BATCHED CROWN ASYMMETRIC (lower_a != upper_a, IBP-VALIDATED)
// =============================================================================
//
// AdaIN1d wraps InstanceNorm1d and applies style_gamma * norm(x) + style_beta.
// Uses the same [C, T] layout: 2 channels, 2 timesteps, in_dim=4, batch=2.
// Asymmetric incoming coefficients exercise the composed backward path.
//
// Part of #3284.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// AdaIN1d batched CROWN soundness with asymmetric incoming (la != ua).
    /// Uses 2 channels, 2 timesteps per channel (in_dim=4), batch=2, out_dim=1.
    /// Part of #3284.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_adain_batched_crown_asymmetric(
        // Batch 0: 4 neurons (2 channels x 2 timesteps)
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
        // Lower incoming coefficients (4 for in_dim=4)
        cl0 in -2.0f32..2.0,
        cl1 in -2.0f32..2.0,
        cl2 in -2.0f32..2.0,
        cl3 in -2.0f32..2.0,
        // Upper incoming coefficients (different from lower)
        cu0 in -2.0f32..2.0,
        cu1 in -2.0f32..2.0,
        cu2 in -2.0f32..2.0,
        cu3 in -2.0f32..2.0,
    ) {
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01 || cl3.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01 || cu3.abs() > 0.01);
        prop_assume!(
            (cl0 - cu0).abs() > 0.01
                || (cl1 - cu1).abs() > 0.01
                || (cl2 - cu2).abs() > 0.01
                || (cl3 - cu3).abs() > 0.01
        );

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

        // Asymmetric batched bounds: la != ua, [batch, out_dim=1, in_dim=4]
        let cl = [cl0, cl1, cl2, cl3];
        let cu = [cu0, cu1, cu2, cu3];
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        for b in 0..batch {
            for i in 0..in_dim {
                la[[b, 0, i]] = cl[i];
                ua[[b, 0, i]] = cu[i];
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
                    "AdaIN1d batched CROWN asymmetric failed: {e}"
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

                // Lower bound uses cl coefficients, upper uses cu
                let lower_val = cl0 * y_true[0] + cl1 * y_true[1] + cl2 * y_true[2] + cl3 * y_true[3];
                let upper_val = cu0 * y_true[0] + cu1 * y_true[1] + cu2 * y_true[2] + cu3 * y_true[3];
                let idx = b;

                prop_assert!(
                    lower_val >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched AdaIN CROWN asymmetric lower at batch {b}: {lower_val} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    upper_val <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched AdaIN CROWN asymmetric upper at batch {b}: {upper_val} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}

// =============================================================================
// GROUPNORM BATCHED CROWN ASYMMETRIC (lower_a != upper_a)
// =============================================================================
//
// GroupNorm: C=2, T=3, in_dim=6, batch=2, num_groups=1, out_dim=1.
// Asymmetric incoming coefficients exercise the group_norm flatten/reshape
// composition with different lower/upper relaxation slopes.
//
// Part of #3284.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// GroupNorm batched CROWN soundness with asymmetric incoming (la != ua).
    /// Uses C=2, T=3 (in_dim=6), batch=2, num_groups=1, out_dim=1.
    /// Part of #3284.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_groupnorm_batched_crown_asymmetric(
        // Batch 0: 6 neurons (C=2, T=3)
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0, c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        // Batch 1: 6 neurons
        c6 in -2.0f32..2.0, c7 in -2.0f32..2.0, c8 in -2.0f32..2.0,
        c9 in -2.0f32..2.0, c10 in -2.0f32..2.0, c11 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Lower incoming coefficients (6 for in_dim=6)
        cl0 in -2.0f32..2.0, cl1 in -2.0f32..2.0, cl2 in -2.0f32..2.0,
        cl3 in -2.0f32..2.0, cl4 in -2.0f32..2.0, cl5 in -2.0f32..2.0,
        // Upper incoming coefficients (different from lower)
        cu0 in -2.0f32..2.0, cu1 in -2.0f32..2.0, cu2 in -2.0f32..2.0,
        cu3 in -2.0f32..2.0, cu4 in -2.0f32..2.0, cu5 in -2.0f32..2.0,
    ) {
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01
            || cl3.abs() > 0.01 || cl4.abs() > 0.01 || cl5.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01
            || cu3.abs() > 0.01 || cu4.abs() > 0.01 || cu5.abs() > 0.01);
        prop_assume!(
            (cl0 - cu0).abs() > 0.01 || (cl1 - cu1).abs() > 0.01 || (cl2 - cu2).abs() > 0.01
                || (cl3 - cu3).abs() > 0.01 || (cl4 - cu4).abs() > 0.01 || (cl5 - cu5).abs() > 0.01
        );

        let in_dim = 6; // C=2, T=3
        let batch = 2;
        let out_dim = 1;
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, in_dim]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, in_dim]), upper_v.clone()).unwrap(),
        ).unwrap();

        let gn = GroupNormLayer::new_default(2, 1, eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        // Asymmetric batched bounds: la != ua, [batch, out_dim=1, in_dim=6]
        let cl = [cl0, cl1, cl2, cl3, cl4, cl5];
        let cu = [cu0, cu1, cu2, cu3, cu4, cu5];
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        for b in 0..batch {
            for i in 0..in_dim {
                la[[b, 0, i]] = cl[i];
                ua[[b, 0, i]] = cu[i];
            }
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![batch, in_dim],
            vec![batch, out_dim],
        ).unwrap();

        let result = gn
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "GroupNorm batched CROWN asymmetric failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        // Sample 50 random points and check each batch independently
        for s in 0..50_u32 {
            let sample: Vec<f32> = (0..12)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            for b in 0..batch {
                let batch_offset = b * in_dim;
                let batch_slice = &sample[batch_offset..batch_offset + in_dim];
                let y_true = group_norm_group(batch_slice, &[1.0, 1.0], &[0.0, 0.0], 2, 3, eps);

                // Lower bound uses cl coefficients, upper uses cu
                let lower_val: f32 = cl.iter().zip(y_true.iter()).map(|(a, b)| a * b).sum();
                let upper_val: f32 = cu.iter().zip(y_true.iter()).map(|(a, b)| a * b).sum();
                let idx = b;

                prop_assert!(
                    lower_val >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched GroupNorm CROWN asymmetric lower at batch {b}: {lower_val} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    upper_val <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched GroupNorm CROWN asymmetric upper at batch {b}: {upper_val} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}
