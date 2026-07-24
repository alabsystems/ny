// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward proptest soundness tests for normalization layers.
//!
//! Tests LayerNorm, RmsNorm, InstanceNorm1d, and AdaIN1d CROWN backward paths.
//!
//! Two test contracts coexist in this file:
//! - **Soundness** (`soundness_*`): use `IbpValidated` mode (provably sound).
//!   Assert sampled true outputs stay within concretized CROWN bounds.
//! - **Structural** (`structural_*`): use `Sampling` mode (heuristic, NOT
//!   provably sound). Assert only finite, non-inverted bounds (no containment).
//!
//! Calls `propagate_linear_with_bounds()` directly on layer types (not via
//! Layer enum dispatch) to isolate the CROWN implementation under test.
//!
//! Part of #3103, #3820.

use crate::layers::normalization::{
    AdaIN1dLayer, InstanceNorm1dLayer, LayerNormCrownMode, LayerNormLayer, RmsNormLayer,
};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{adain_eval_channel, instance_norm_channel, rms_norm, sample_points};

/// Concretize CROWN linear bounds against input interval bounds.
/// Returns (lower_bounds, upper_bounds) as Vecs.
fn concretize_crown(result: &LinearBounds, pre_activation: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

/// Tolerance for sampling-based CROWN soundness.
/// Sampling linearization uses 3^n grid + 50 random samples with 1.5x safety factor.
/// With tight perturbation (half-width <= 0.3) and moderate parameters,
/// the heuristic typically holds within 1e-2.
const SAMPLING_CROWN_TOLERANCE: f32 = 1e-2;

// Reference eval functions consolidated in super (mod.rs):
// rms_norm, instance_norm_channel, adain_eval_channel

// =============================================================================
// RMSNORM CROWN BACKWARD STRUCTURAL (SAMPLING MODE)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// RmsNorm CROWN structural: identity params, identity incoming.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn structural_rmsnorm_crown_sampling_identity(
        l0 in -3.0f32..3.0,
        d0 in 0.1f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.1f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.1f32..2.0,
    ) {
        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let rn = RmsNormLayer::new_default(3, 1e-5)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::Sampling);

        let identity = LinearBounds::identity(3);

        let result = rn
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("RmsNorm CROWN failed: {e}"))
            })?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        prop_assert_eq!(crown_lower.len(), 3);
        prop_assert_eq!(crown_upper.len(), 3);

        for i in 0..3 {
            prop_assert!(
                crown_lower[i].is_finite(),
                "RmsNorm lower[{i}] not finite: {}",
                crown_lower[i]
            );
            prop_assert!(
                crown_upper[i].is_finite(),
                "RmsNorm upper[{i}] not finite: {}",
                crown_upper[i]
            );
            prop_assert!(
                crown_lower[i] <= crown_upper[i] + 1e-6,
                "RmsNorm lower[{i}]={} > upper[{i}]={}",
                crown_lower[i],
                crown_upper[i]
            );
        }
    }
}

// =============================================================================
// RMSNORM CROWN BACKWARD SOUNDNESS (TIGHT PERTURBATION)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// RmsNorm CROWN soundness with identity incoming, tight perturbation.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_rmsnorm_crown_identity_tight(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw0 in 0.05f32..0.3,
        hw1 in 0.05f32..0.3,
        hw2 in 0.05f32..0.3,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
    ) {
        let l0 = c0 - hw0;
        let u0 = c0 + hw0;
        let l1 = c1 - hw1;
        let u1 = c1 + hw1;
        let l2 = c2 - hw2;
        let u2 = c2 + hw2;

        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let rn = RmsNormLayer::new(ny.clone(), eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let identity = LinearBounds::identity(3);

        let result = rn
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("RmsNorm CROWN failed: {e}"))
            })?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        let s0_pts = sample_points(l0, u0, 5);
        let s1_pts = sample_points(l1, u1, 5);
        let s2_pts = sample_points(l2, u2, 5);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let rn_val = rms_norm(&point, &ny, eps);

                    for i in 0..3 {
                        prop_assert!(
                            rn_val[i] >= crown_lower[i] - SAMPLING_CROWN_TOLERANCE,
                            "RmsNorm CROWN lower violation at dim {i}: \
                             rmsnorm({x0},{x1},{x2})[{i}]={} < lb={}",
                            rn_val[i],
                            crown_lower[i],
                        );
                        prop_assert!(
                            rn_val[i] <= crown_upper[i] + SAMPLING_CROWN_TOLERANCE,
                            "RmsNorm CROWN upper violation at dim {i}: \
                             rmsnorm({x0},{x1},{x2})[{i}]={} > ub={}",
                            rn_val[i],
                            crown_upper[i],
                        );
                    }
                }
            }
        }
    }

    /// RmsNorm CROWN soundness with negative-coefficient incoming bounds.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_rmsnorm_crown_negcoeff(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw0 in 0.05f32..0.3,
        hw1 in 0.05f32..0.3,
        hw2 in 0.05f32..0.3,
        ic0 in -2.0f32..2.0,
        ic1 in -2.0f32..2.0,
        ic2 in -2.0f32..2.0,
    ) {
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01);

        let l0 = c0 - hw0;
        let u0 = c0 + hw0;
        let l1 = c1 - hw1;
        let u1 = c1 + hw1;
        let l2 = c2 - hw2;
        let u2 = c2 + hw2;

        let eps = 1e-5_f32;
        let ny = Array1::ones(3);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let rn = RmsNormLayer::new(ny.clone(), eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![ic0, ic1, ic2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![ic0, ic1, ic2]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        let result = rn
            .propagate_linear_with_bounds(&incoming, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("RmsNorm CROWN failed: {e}"))
            })?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        let s0_pts = sample_points(l0, u0, 5);
        let s1_pts = sample_points(l1, u1, 5);
        let s2_pts = sample_points(l2, u2, 5);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let rv = rms_norm(&point, &ny, eps);
                    let combined = ic0 * rv[0] + ic1 * rv[1] + ic2 * rv[2];

                    prop_assert!(
                        combined >= crown_lower[0] - SAMPLING_CROWN_TOLERANCE,
                        "RmsNorm negcoeff lower: {combined} < {}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        combined <= crown_upper[0] + SAMPLING_CROWN_TOLERANCE,
                        "RmsNorm negcoeff upper: {combined} > {}",
                        crown_upper[0]
                    );
                }
            }
        }
    }
}

// =============================================================================
// LAYERNORM CROWN BACKWARD SOUNDNESS (TIGHT PERTURBATION)
// =============================================================================
//
// LayerNorm soundness tests use IbpValidated mode (provably sound). They verify
// that computed CROWN bounds actually contain sampled true outputs within
// SAMPLING_CROWN_TOLERANCE.
//
// Reference: the layernorm eval function in mod.rs computes
//   y = ny * (x - mean) / sqrt(var + eps) + beta

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// LayerNorm CROWN soundness with identity incoming, tight perturbation.
    /// Verifies that for sampled concrete inputs within bounds, the true
    /// LayerNorm output falls within the CROWN-computed bounds.
    /// Part of #2167.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_crown_identity_tight(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw0 in 0.05f32..0.3,
        hw1 in 0.05f32..0.3,
        hw2 in 0.05f32..0.3,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
    ) {
        let l0 = c0 - hw0;
        let u0 = c0 + hw0;
        let l1 = c1 - hw1;
        let u1 = c1 + hw1;
        let l2 = c2 - hw2;
        let u2 = c2 + hw2;

        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let identity = LinearBounds::identity(3);

        let result = ln
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("LayerNorm CROWN failed: {e}"))
            })?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // 5^3 = 125 grid points (sample_points returns 5 evenly spaced points per dim)
        let s0_pts = sample_points(l0, u0, 5);
        let s1_pts = sample_points(l1, u1, 5);
        let s2_pts = sample_points(l2, u2, 5);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let ln_val = super::layernorm(&point, &ny, &beta, eps);

                    for i in 0..3 {
                        prop_assert!(
                            ln_val[i] >= crown_lower[i] - SAMPLING_CROWN_TOLERANCE,
                            "LayerNorm CROWN lower violation at dim {i}: \
                             layernorm({x0},{x1},{x2})[{i}]={} < lb={}",
                            ln_val[i],
                            crown_lower[i],
                        );
                        prop_assert!(
                            ln_val[i] <= crown_upper[i] + SAMPLING_CROWN_TOLERANCE,
                            "LayerNorm CROWN upper violation at dim {i}: \
                             layernorm({x0},{x1},{x2})[{i}]={} > ub={}",
                            ln_val[i],
                            crown_upper[i],
                        );
                    }
                }
            }
        }
    }

    /// LayerNorm CROWN soundness with negative-coefficient incoming bounds.
    /// Part of #2167.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_crown_negcoeff(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw0 in 0.05f32..0.3,
        hw1 in 0.05f32..0.3,
        hw2 in 0.05f32..0.3,
        ic0 in -2.0f32..2.0,
        ic1 in -2.0f32..2.0,
        ic2 in -2.0f32..2.0,
    ) {
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01);

        let l0 = c0 - hw0;
        let u0 = c0 + hw0;
        let l1 = c1 - hw1;
        let u1 = c1 + hw1;
        let l2 = c2 - hw2;
        let u2 = c2 + hw2;

        let eps = 1e-5_f32;
        let ny = Array1::ones(3);
        let beta = Array1::zeros(3);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![ic0, ic1, ic2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![ic0, ic1, ic2]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        let result = ln
            .propagate_linear_with_bounds(&incoming, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("LayerNorm CROWN failed: {e}"))
            })?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        let s0_pts = sample_points(l0, u0, 5);
        let s1_pts = sample_points(l1, u1, 5);
        let s2_pts = sample_points(l2, u2, 5);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let lv = super::layernorm(&point, &ny, &beta, eps);
                    let combined = ic0 * lv[0] + ic1 * lv[1] + ic2 * lv[2];

                    prop_assert!(
                        combined >= crown_lower[0] - SAMPLING_CROWN_TOLERANCE,
                        "LayerNorm negcoeff lower: {combined} < {}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        combined <= crown_upper[0] + SAMPLING_CROWN_TOLERANCE,
                        "LayerNorm negcoeff upper: {combined} > {}",
                        crown_upper[0]
                    );
                }
            }
        }
    }
}

// =============================================================================
// INSTANCENORM1D CROWN BACKWARD
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// InstanceNorm1d CROWN structural: 2 channels, 3 timesteps.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn structural_instancenorm_crown_sampling(
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

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), ls.to_vec()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), us.to_vec()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new(Array1::ones(2), Array1::zeros(2), 1e-5)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::Sampling);

        let identity = LinearBounds::identity(6);

        let result = inn
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("InstanceNorm CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[6]), ls.to_vec()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[6]), us.to_vec()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        for i in 0..6 {
            prop_assert!(cl[i].is_finite() && cu[i].is_finite());
            prop_assert!(cl[i] <= cu[i] + 1e-6, "lower > upper at {i}");
        }
    }

    /// InstanceNorm1d CROWN soundness: 2 channels, 3 timesteps, tight perturbation.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_instancenorm_crown_identity_tight(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0, c4 in -2.0f32..2.0, c5 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        g0 in 0.5f32..2.0, g1 in 0.5f32..2.0,
        b0 in -1.0f32..1.0, b1 in -1.0f32..1.0,
    ) {
        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new(
            Array1::from_vec(vec![g0, g1]),
            Array1::from_vec(vec![b0, b1]),
            eps,
        )
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let identity = LinearBounds::identity(6);
        let result = inn
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("InstanceNorm CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[6]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[6]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        let gammas = [g0, g1];
        let betas = [b0, b1];

        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..6)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            let mut y_true: Vec<f32> = Vec::with_capacity(6);
            for c in 0..2 {
                let start = c * 3;
                let x_ch = arr1(&sample[start..start + 3]);
                let y_ch = instance_norm_channel(&x_ch, gammas[c], betas[c], eps);
                y_true.extend(y_ch.iter());
            }

            for i in 0..6 {
                prop_assert!(
                    y_true[i] >= cl[i] - SAMPLING_CROWN_TOLERANCE,
                    "InstanceNorm lower dim {i}: {} < {}",
                    y_true[i],
                    cl[i]
                );
                prop_assert!(
                    y_true[i] <= cu[i] + SAMPLING_CROWN_TOLERANCE,
                    "InstanceNorm upper dim {i}: {} > {}",
                    y_true[i],
                    cu[i]
                );
            }
        }
    }

    /// InstanceNorm1d CROWN soundness with negative incoming coefficients.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_instancenorm_crown_negcoeff(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0, c3 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        ic0 in -2.0f32..2.0, ic1 in -2.0f32..2.0,
        ic2 in -2.0f32..2.0, ic3 in -2.0f32..2.0,
    ) {
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01 || ic3.abs() > 0.01);

        let centers = [c0, c1, c2, c3];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 4]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 4]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new(
            Array1::from_vec(vec![1.5]),
            Array1::from_vec(vec![-0.5]),
            eps,
        )
        .unwrap()
        .with_crown_mode(LayerNormCrownMode::IbpValidated);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 4), vec![ic0, ic1, ic2, ic3]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 4), vec![ic0, ic1, ic2, ic3]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        let result = inn
            .propagate_linear_with_bounds(&incoming, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("InstanceNorm CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[4]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[4]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..4)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            let x_ch = arr1(&sample);
            let y_ch = instance_norm_channel(&x_ch, 1.5, -0.5, eps);
            let combined = ic0 * y_ch[0] + ic1 * y_ch[1] + ic2 * y_ch[2] + ic3 * y_ch[3];

            prop_assert!(combined >= cl[0] - SAMPLING_CROWN_TOLERANCE);
            prop_assert!(combined <= cu[0] + SAMPLING_CROWN_TOLERANCE);
        }
    }
}

// =============================================================================
// ADAIN1D CROWN BACKWARD
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// AdaIN1d CROWN structural: 2 channels, 2 timesteps.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn structural_adain_crown_sampling(
        l0 in -3.0f32..3.0, d0 in 0.1f32..2.0,
        l1 in -3.0f32..3.0, d1 in 0.1f32..2.0,
        l2 in -3.0f32..3.0, d2 in 0.1f32..2.0,
        l3 in -3.0f32..3.0, d3 in 0.1f32..2.0,
        sg0 in 0.5f32..2.0, sg1 in 0.5f32..2.0,
        sb0 in -1.0f32..1.0, sb1 in -1.0f32..1.0,
    ) {
        let us = [
            (l0 + d0).min(3.0), (l1 + d1).min(3.0),
            (l2 + d2).min(3.0), (l3 + d3).min(3.0),
        ];
        let ls = [l0, l1, l2, l3];

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), ls.to_vec()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), us.to_vec()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new(Array1::ones(2), Array1::zeros(2), 1e-5)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::Sampling);
        let adain = AdaIN1dLayer::new(
            inn,
            Array1::from_vec(vec![sg0, sg1]),
            Array1::from_vec(vec![sb0, sb1]),
        )
        .unwrap();

        let identity = LinearBounds::identity(4);
        let result = adain
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("AdaIN CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[4]), ls.to_vec()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[4]), us.to_vec()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        for i in 0..4 {
            prop_assert!(cl[i].is_finite() && cu[i].is_finite());
            prop_assert!(cl[i] <= cu[i] + 1e-6, "lower > upper at {i}");
        }
    }

    /// AdaIN1d CROWN soundness: 2 channels, 2 timesteps, tight perturbation.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_adain_crown_identity_tight(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0, c3 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        g0 in 0.5f32..2.0, g1 in 0.5f32..2.0,
        b0 in -0.5f32..0.5, b1 in -0.5f32..0.5,
        sg0 in 0.5f32..2.0, sg1 in 0.5f32..2.0,
        sb0 in -0.5f32..0.5, sb1 in -0.5f32..0.5,
    ) {
        let centers = [c0, c1, c2, c3];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new(
            Array1::from_vec(vec![g0, g1]),
            Array1::from_vec(vec![b0, b1]),
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

        let identity = LinearBounds::identity(4);
        let result = adain
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("AdaIN CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[4]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[4]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        let gammas = [g0, g1];
        let betas = [b0, b1];
        let sgs = [sg0, sg1];
        let sbs = [sb0, sb1];

        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..4)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            let mut y_true: Vec<f32> = Vec::with_capacity(4);
            for c in 0..2 {
                let start = c * 2;
                let x_ch = arr1(&sample[start..start + 2]);
                let y_ch = adain_eval_channel(&x_ch, gammas[c], betas[c], sgs[c], sbs[c], eps);
                y_true.extend(y_ch.iter());
            }

            for i in 0..4 {
                prop_assert!(
                    y_true[i] >= cl[i] - SAMPLING_CROWN_TOLERANCE,
                    "AdaIN lower dim {i}: {} < {}",
                    y_true[i],
                    cl[i]
                );
                prop_assert!(
                    y_true[i] <= cu[i] + SAMPLING_CROWN_TOLERANCE,
                    "AdaIN upper dim {i}: {} > {}",
                    y_true[i],
                    cu[i]
                );
            }
        }
    }

    /// AdaIN1d CROWN soundness with negative incoming coefficients.
    /// 1 channel, 3 timesteps.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_adain_crown_negcoeff(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        ic0 in -2.0f32..2.0, ic1 in -2.0f32..2.0, ic2 in -2.0f32..2.0,
        sg in 0.5f32..2.0, sb in -1.0f32..1.0,
    ) {
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01);

        let centers = [c0, c1, c2];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();
        let eps = 1e-5_f32;

        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), lower_v.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3]), upper_v.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let inn = InstanceNorm1dLayer::new(Array1::ones(1), Array1::zeros(1), eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);
        let adain = AdaIN1dLayer::new(
            inn,
            Array1::from_vec(vec![sg]),
            Array1::from_vec(vec![sb]),
        )
        .unwrap();

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![ic0, ic1, ic2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![ic0, ic1, ic2]).unwrap(),
            Array1::zeros(1),
        )
        .unwrap();

        let result = adain
            .propagate_linear_with_bounds(&incoming, &input)
            .map_err(|e| {
                TestCaseError::fail(format!("AdaIN CROWN: {e}"))
            })?;

        let flat_l = ArrayD::from_shape_vec(IxDyn(&[3]), lower_v.clone()).unwrap();
        let flat_u = ArrayD::from_shape_vec(IxDyn(&[3]), upper_v.clone()).unwrap();
        let flat_input = BoundedTensor::new(flat_l, flat_u).unwrap();
        let (cl, cu) = concretize_crown(&result, &flat_input);

        for s in 0..100_u32 {
            let sample: Vec<f32> = (0..3)
                .map(|i| {
                    let t = ((s.wrapping_mul(2654435761) ^ (i as u32))
                        .wrapping_mul(2654435761)) as f32
                        / u32::MAX as f32;
                    lower_v[i] + (upper_v[i] - lower_v[i]) * t
                })
                .collect();

            let x_ch = arr1(&sample);
            let y_ch = adain_eval_channel(&x_ch, 1.0, 0.0, sg, sb, eps);
            let combined = ic0 * y_ch[0] + ic1 * y_ch[1] + ic2 * y_ch[2];

            prop_assert!(combined >= cl[0] - SAMPLING_CROWN_TOLERANCE);
            prop_assert!(combined <= cu[0] + SAMPLING_CROWN_TOLERANCE);
        }
    }
}
