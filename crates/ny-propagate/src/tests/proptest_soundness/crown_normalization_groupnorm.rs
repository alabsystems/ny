// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward proptest soundness tests for GroupNorm.
//!
//! Structural GroupNorm coverage keeps one `Sampling`-mode validity test, while
//! the containment tests use `LayerNormCrownMode::IbpValidated`, the sound
//! normalization CROWN mode. This verifies that the GroupNorm-specific
//! flatten/reshape logic (groups, multi-channel) doesn't break soundness.
//!
//! Most tests use C=2, T=3, num_groups=1 (LayerNorm-like -- all channels in one
//! group). The `2groups` tests use C=4, T=2, num_groups=2 to exercise the
//! group-splitting CROWN backward path where the Jacobian is block-diagonal.
//!
//! Covers: scalar CROWN (identity, negcoeff), asymmetric incoming,
//! batched CROWN (identity, negcoeff).
//!
//! Part of #3258.

use crate::layers::normalization::{GroupNormLayer, LayerNormCrownMode};
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::group_norm_group;

/// Tolerance for sampled-point containment checks.
/// Matching crown_normalization.rs.
const SAMPLING_CROWN_TOLERANCE: f32 = 1e-2;

/// Concretize CROWN linear bounds against input interval bounds.
/// Returns (lower_bounds, upper_bounds) as Vecs.
fn concretize_crown(result: &LinearBounds, pre_activation: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

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
// SCALAR CROWN BACKWARD STRUCTURAL + SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// GroupNorm CROWN structural: identity params, identity incoming.
    /// Part of #3258.
    #[ntest::timeout(60000)]
    #[test]
    fn structural_groupnorm_crown_sampling_identity(
        l0 in -3.0f32..3.0, d0 in 0.1f32..2.0,
        l1 in -3.0f32..3.0, d1 in 0.1f32..2.0,
        l2 in -3.0f32..3.0, d2 in 0.1f32..2.0,
        l3 in -3.0f32..3.0, d3 in 0.1f32..2.0,
        l4 in -3.0f32..3.0, d4 in 0.1f32..2.0,
        l5 in -3.0f32..3.0, d5 in 0.1f32..2.0,
    ) {
        let us = [
            (l0 + d0).min(3.0), (l1 + d1).min(3.0), (l2 + d2).min(3.0),
            (l3 + d3).min(3.0), (l4 + d4).min(3.0), (l5 + d5).min(3.0),
        ];
        let ls = [l0, l1, l2, l3, l4, l5];

        // Pre-activation shape: [C=2, T=3] for IBP validation
        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), ls.to_vec()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), us.to_vec()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let gn = GroupNormLayer::new_default(2, 1, 1e-5)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::Sampling);

        let identity = LinearBounds::identity(6);

        let result = gn
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("GroupNorm CROWN failed: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[6]), ls.to_vec()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[6]), us.to_vec()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        prop_assert_eq!(cl.len(), 6);
        prop_assert_eq!(cu.len(), 6);

        for i in 0..6 {
            prop_assert!(
                cl[i].is_finite(),
                "GroupNorm lower[{i}] not finite: {}",
                cl[i]
            );
            prop_assert!(
                cu[i].is_finite(),
                "GroupNorm upper[{i}] not finite: {}",
                cu[i]
            );
            prop_assert!(
                cl[i] <= cu[i] + 1e-6,
                "GroupNorm lower[{i}]={} > upper[{i}]={}",
                cl[i],
                cu[i]
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// GroupNorm CROWN soundness with identity incoming, tight perturbation.
    /// Part of #3258.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_groupnorm_crown_identity_tight(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0, c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        g0 in 0.5f32..2.0, g1 in 0.5f32..2.0,
    ) {
        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![g0, g1]);
        let beta = Array1::from_vec(vec![0.0, 0.0]);
        let gn = GroupNormLayer::new(ny, beta, 1, eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let identity = LinearBounds::identity(6);
        let result = gn
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("GroupNorm CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[6]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[6]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        let gammas = [g0, g1];

        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..6)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            // Evaluate GroupNorm: 1 group, C=2, T=3
            let y_true = group_norm_group(
                &sample, &gammas, &[0.0, 0.0], 2, 3, eps,
            );

            for i in 0..6 {
                prop_assert!(
                    y_true[i] >= cl[i] - SAMPLING_CROWN_TOLERANCE,
                    "GroupNorm CROWN lower dim {i}: {} < {}",
                    y_true[i],
                    cl[i]
                );
                prop_assert!(
                    y_true[i] <= cu[i] + SAMPLING_CROWN_TOLERANCE,
                    "GroupNorm CROWN upper dim {i}: {} > {}",
                    y_true[i],
                    cu[i]
                );
            }
        }
    }

    /// GroupNorm CROWN soundness with negative-coefficient incoming bounds.
    /// Part of #3258.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_groupnorm_crown_negcoeff(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0, c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        ic0 in -2.0f32..2.0, ic1 in -2.0f32..2.0, ic2 in -2.0f32..2.0,
        ic3 in -2.0f32..2.0, ic4 in -2.0f32..2.0, ic5 in -2.0f32..2.0,
    ) {
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01
            || ic3.abs() > 0.01 || ic4.abs() > 0.01 || ic5.abs() > 0.01);

        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let gn = GroupNormLayer::new_default(2, 1, eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let ic = vec![ic0, ic1, ic2, ic3, ic4, ic5];
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 6), ic.clone()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 6), ic.clone()).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        let result = gn
            .propagate_linear_with_bounds(&incoming, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("GroupNorm CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[6]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[6]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..6)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            let y_true = group_norm_group(
                &sample, &[1.0, 1.0], &[0.0, 0.0], 2, 3, eps,
            );
            let combined: f32 = ic.iter().zip(y_true.iter()).map(|(a, b)| a * b).sum();

            prop_assert!(
                combined >= cl[0] - SAMPLING_CROWN_TOLERANCE,
                "GroupNorm negcoeff lower: {combined} < {}",
                cl[0]
            );
            prop_assert!(
                combined <= cu[0] + SAMPLING_CROWN_TOLERANCE,
                "GroupNorm negcoeff upper: {combined} > {}",
                cu[0]
            );
        }
    }
}

// =============================================================================
// ASYMMETRIC INCOMING (lower_a != upper_a)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// GroupNorm CROWN soundness with asymmetric incoming (lower_a != upper_a).
    /// Part of #3258.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_groupnorm_crown_asymmetric(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0, c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Lower incoming coefficients
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

        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let gn = GroupNormLayer::new_default(2, 1, eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let cl_coeffs = vec![cl0, cl1, cl2, cl3, cl4, cl5];
        let cu_coeffs = vec![cu0, cu1, cu2, cu3, cu4, cu5];

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 6), cl_coeffs.clone()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 6), cu_coeffs.clone()).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        let result = gn
            .propagate_linear_with_bounds(&incoming, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("GroupNorm CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[6]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[6]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (crown_l, crown_u) = concretize_crown(&result, &flat_input);

        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..6)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            let y_true = group_norm_group(
                &sample, &[1.0, 1.0], &[0.0, 0.0], 2, 3, eps,
            );
            let lower_val: f32 = cl_coeffs.iter().zip(y_true.iter()).map(|(a, b)| a * b).sum();
            let upper_val: f32 = cu_coeffs.iter().zip(y_true.iter()).map(|(a, b)| a * b).sum();

            prop_assert!(
                lower_val >= crown_l[0] - SAMPLING_CROWN_TOLERANCE,
                "GroupNorm asymmetric lower: {lower_val} < {}",
                crown_l[0]
            );
            prop_assert!(
                upper_val <= crown_u[0] + SAMPLING_CROWN_TOLERANCE,
                "GroupNorm asymmetric upper: {upper_val} > {}",
                crown_u[0]
            );
        }
    }
}

// =============================================================================
// BATCHED CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// GroupNorm batched CROWN soundness with identity incoming.
    /// Uses C=2, T=3 (in_dim=6), batch=2, num_groups=1.
    /// Part of #3258.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_groupnorm_batched_crown_identity_tight(
        // Batch 0: 6 neurons (C=2, T=3)
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0, c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        // Batch 1: 6 neurons
        c6 in -2.0f32..2.0, c7 in -2.0f32..2.0, c8 in -2.0f32..2.0,
        c9 in -2.0f32..2.0, c10 in -2.0f32..2.0, c11 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        g0 in 0.5f32..2.0, g1 in 0.5f32..2.0,
    ) {
        let in_dim = 6; // C=2, T=3
        let batch = 2;
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, in_dim]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, in_dim]), upper_v.clone()).unwrap(),
        ).unwrap();

        let ny = Array1::from_vec(vec![g0, g1]);
        let beta = Array1::from_vec(vec![0.0, 0.0]);
        let gn = GroupNormLayer::new(ny, beta, 1, eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        // Identity batched bounds: [batch, in_dim, in_dim]
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, in_dim, in_dim]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, in_dim, in_dim]));
        for b in 0..batch {
            for i in 0..in_dim {
                la[[b, i, i]] = 1.0;
                ua[[b, i, i]] = 1.0;
            }
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[batch, in_dim]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[batch, in_dim]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![batch, in_dim],
            vec![batch, in_dim],
        ).unwrap();

        let result = gn
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "GroupNorm batched CROWN failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        let gammas = [g0, g1];

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
                // 1 group, C=2, T=3
                let y_true = group_norm_group(batch_slice, &gammas, &[0.0, 0.0], 2, 3, eps);

                for (i, &y_val) in y_true.iter().enumerate().take(in_dim) {
                    let idx = batch_offset + i;
                    prop_assert!(
                        y_val >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                        "Batched GroupNorm CROWN lower at batch {b} dim {i}: {} < {}",
                        y_val,
                        crown_lower[idx],
                    );
                    prop_assert!(
                        y_val <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                        "Batched GroupNorm CROWN upper at batch {b} dim {i}: {} > {}",
                        y_val,
                        crown_upper[idx],
                    );
                }
            }
        }
    }

    /// GroupNorm batched CROWN soundness with non-trivial incoming coefficients.
    /// Uses C=2, T=3 (in_dim=6), batch=2, out_dim=1, num_groups=1.
    /// Part of #3258.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_groupnorm_batched_crown_negcoeff(
        // Batch 0: 6 neurons (C=2, T=3)
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0, c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        // Batch 1: 6 neurons
        c6 in -2.0f32..2.0, c7 in -2.0f32..2.0, c8 in -2.0f32..2.0,
        c9 in -2.0f32..2.0, c10 in -2.0f32..2.0, c11 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Incoming coefficients (mixed sign)
        ic0 in -2.0f32..2.0, ic1 in -2.0f32..2.0, ic2 in -2.0f32..2.0,
        ic3 in -2.0f32..2.0, ic4 in -2.0f32..2.0, ic5 in -2.0f32..2.0,
    ) {
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01
            || ic3.abs() > 0.01 || ic4.abs() > 0.01 || ic5.abs() > 0.01);

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

        // Non-trivial incoming: [batch, out_dim=1, in_dim=6], same coeffs per batch
        let ic = [ic0, ic1, ic2, ic3, ic4, ic5];
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, out_dim, in_dim]));
        for b in 0..batch {
            for i in 0..in_dim {
                la[[b, 0, i]] = ic[i];
                ua[[b, 0, i]] = ic[i];
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
                    "GroupNorm batched CROWN negcoeff failed: {e}"
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

                let combined: f32 = ic.iter().zip(y_true.iter()).map(|(a, b)| a * b).sum();
                let idx = b; // out_dim=1, so index is just batch

                prop_assert!(
                    combined >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched GroupNorm CROWN negcoeff lower at batch {b}: {combined} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    combined <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched GroupNorm CROWN negcoeff upper at batch {b}: {combined} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}

// =============================================================================
// MULTI-GROUP CROWN BACKWARD SOUNDNESS (num_groups > 1)
//
// All tests above use num_groups=1 (LayerNorm-like). These tests use
// num_groups=2 to exercise the block-diagonal Jacobian structure where each
// group's CROWN backward is independent.
//
// Re: P1 iter 42 finding — GroupNorm CROWN proptests need num_groups > 1.
// Part of #3258.
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// GroupNorm CROWN soundness with 2 groups, identity incoming.
    /// C=4, T=2, num_groups=2 → cpg=2, 4 elements per group, 8 total.
    /// Exercises group-splitting logic in CROWN backward path.
    /// Part of #3258.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_groupnorm_crown_2groups_identity(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0, c3 in -2.0f32..2.0,
        c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        c6 in -2.0f32..2.0, c7 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
    ) {
        let num_channels = 4;
        let time_len = 2;
        let in_dim = num_channels * time_len; // 8
        let num_groups = 2;
        let cpg = num_channels / num_groups; // 2
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5, c6, c7];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        // Pre-activation shape: [C=4, T=2]
        let lower = ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let gn = GroupNormLayer::new_default(num_channels, num_groups, eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let identity = LinearBounds::identity(in_dim);

        let result = gn
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "GroupNorm 2-group CROWN failed: {e}"
                ))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[in_dim]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[in_dim]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        // Sample 100 random points and verify soundness
        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..in_dim)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            // Evaluate each group independently
            // Group 0: channels 0-1, elements [0..4] in flattened [C, T] order
            // Group 1: channels 2-3, elements [4..8]
            let mut y_true = vec![0.0_f32; in_dim];
            for g in 0..num_groups {
                let group_start = g * cpg * time_len;
                let group_end = group_start + cpg * time_len;
                let group_slice = &sample[group_start..group_end];
                let group_gammas: Vec<f32> = (0..cpg).map(|_| 1.0).collect();
                let group_betas: Vec<f32> = (0..cpg).map(|_| 0.0).collect();
                let group_out = group_norm_group(
                    group_slice, &group_gammas, &group_betas, cpg, time_len, eps,
                );
                y_true[group_start..group_end].copy_from_slice(&group_out);
            }

            for i in 0..in_dim {
                prop_assert!(
                    y_true[i] >= cl[i] - SAMPLING_CROWN_TOLERANCE,
                    "GroupNorm 2-group CROWN lower dim {i}: {} < {}",
                    y_true[i], cl[i]
                );
                prop_assert!(
                    y_true[i] <= cu[i] + SAMPLING_CROWN_TOLERANCE,
                    "GroupNorm 2-group CROWN upper dim {i}: {} > {}",
                    y_true[i], cu[i]
                );
            }
        }
    }

    /// GroupNorm CROWN soundness with 2 groups and negative-coefficient incoming.
    /// C=4, T=2, num_groups=2, incoming has mixed-sign coefficients.
    /// Part of #3258.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_groupnorm_crown_2groups_negcoeff(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0, c3 in -2.0f32..2.0,
        c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        c6 in -2.0f32..2.0, c7 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        ic0 in -2.0f32..2.0, ic1 in -2.0f32..2.0,
        ic2 in -2.0f32..2.0, ic3 in -2.0f32..2.0,
        ic4 in -2.0f32..2.0, ic5 in -2.0f32..2.0,
        ic6 in -2.0f32..2.0, ic7 in -2.0f32..2.0,
    ) {
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01
            || ic3.abs() > 0.01 || ic4.abs() > 0.01 || ic5.abs() > 0.01
            || ic6.abs() > 0.01 || ic7.abs() > 0.01);

        let num_channels = 4;
        let time_len = 2;
        let in_dim = num_channels * time_len; // 8
        let num_groups = 2;
        let cpg = num_channels / num_groups; // 2
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5, c6, c7];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let lower = ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[num_channels, time_len]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let gn = GroupNormLayer::new_default(num_channels, num_groups, eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let ic = [ic0, ic1, ic2, ic3, ic4, ic5, ic6, ic7];
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, in_dim), ic.to_vec()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, in_dim), ic.to_vec()).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        let result = gn
            .propagate_linear_with_bounds(&incoming, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "GroupNorm 2-group negcoeff CROWN failed: {e}"
                ))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[in_dim]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[in_dim]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..in_dim)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            // Evaluate GroupNorm per-group, then apply incoming coefficients
            let mut y_true = vec![0.0_f32; in_dim];
            for g in 0..num_groups {
                let group_start = g * cpg * time_len;
                let group_end = group_start + cpg * time_len;
                let group_slice = &sample[group_start..group_end];
                let group_gammas: Vec<f32> = (0..cpg).map(|_| 1.0).collect();
                let group_betas: Vec<f32> = (0..cpg).map(|_| 0.0).collect();
                let group_out = group_norm_group(
                    group_slice, &group_gammas, &group_betas, cpg, time_len, eps,
                );
                y_true[group_start..group_end].copy_from_slice(&group_out);
            }

            let combined: f32 = ic.iter().zip(y_true.iter()).map(|(a, b)| a * b).sum();

            prop_assert!(
                combined >= cl[0] - SAMPLING_CROWN_TOLERANCE,
                "GroupNorm 2-group negcoeff lower: {combined} < {}",
                cl[0]
            );
            prop_assert!(
                combined <= cu[0] + SAMPLING_CROWN_TOLERANCE,
                "GroupNorm 2-group negcoeff upper: {combined} > {}",
                cu[0]
            );
        }
    }
}
