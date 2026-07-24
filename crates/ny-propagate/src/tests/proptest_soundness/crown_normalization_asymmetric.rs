// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward proptest soundness tests for normalization layers with
//! asymmetric incoming coefficients (lower_a != upper_a).
//!
//! Split from `crown_normalization.rs` which retains identity, structural,
//! and symmetric non-identity (negcoeff) tests.
//!
//! In real CROWN backward propagation, after composing through several layers,
//! lower_a and upper_a diverge. These tests verify soundness with lower_a !=
//! upper_a for RmsNorm, LayerNorm, InstanceNorm1d, and AdaIN1d.
//!
//! These containment tests use `LayerNormCrownMode::IbpValidated`, the sound
//! normalization CROWN mode. Structural `Sampling` coverage remains in
//! `crown_normalization.rs`.
//!
//! Part of #3103.

use crate::layers::normalization::{
    AdaIN1dLayer, InstanceNorm1dLayer, LayerNormCrownMode, LayerNormLayer, RmsNormLayer,
};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

/// Tolerance for sampled-point containment checks.
/// Matches `crown_normalization.rs`.
const SAMPLING_CROWN_TOLERANCE: f32 = 1e-2;

/// Concretize CROWN linear bounds against input interval bounds.
fn concretize_crown(result: &LinearBounds, pre_activation: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

// Reference eval functions — reused from crown_normalization.rs.

/// Reference RMSNorm evaluation: y_i = ny_i * x_i / sqrt(mean(x^2) + eps)
fn rmsnorm_eval(x: &Array1<f32>, ny: &Array1<f32>, eps: f32) -> Array1<f32> {
    let n = x.len() as f32;
    let mean_sq = x.iter().map(|&xi| xi * xi).sum::<f32>() / n;
    let rms = (mean_sq + eps).sqrt();
    x.iter()
        .zip(ny.iter())
        .map(|(&xi, &g)| g * xi / rms)
        .collect()
}

/// Reference InstanceNorm1d evaluation for a single channel.
fn instance_norm_eval_channel(x: &Array1<f32>, ny: f32, beta: f32, eps: f32) -> Array1<f32> {
    let mean = x.mean().unwrap_or(0.0);
    let var = x.mapv(|xi| (xi - mean).powi(2)).mean().unwrap_or(0.0);
    let std = (var + eps).sqrt();
    x.mapv(|xi| ny * (xi - mean) / std + beta)
}

/// Reference AdaIN1d evaluation for a single channel.
fn adain_eval_channel(
    x: &Array1<f32>,
    ny: f32,
    beta: f32,
    style_gamma: f32,
    style_beta: f32,
    eps: f32,
) -> Array1<f32> {
    let normed = instance_norm_eval_channel(x, ny, beta, eps);
    normed.mapv(|y| style_gamma * y + style_beta)
}

// =============================================================================
// ASYMMETRIC INCOMING COEFFICIENT TESTS
// =============================================================================
//
// The soundness property: for any x in [lower, upper],
//   crown_lower <= cl . f(x)   (lower bound holds for lower-bound linear fn)
//   cu . f(x) <= crown_upper   (upper bound holds for upper-bound linear fn)

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// RmsNorm CROWN backward with asymmetric incoming coefficients.
    ///
    /// lower_a != upper_a exercises the coefficient split path where positive
    /// and negative coefficient handling differs for lower vs upper bounds.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_rmsnorm_crown_asymmetric_incoming(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw0 in 0.05f32..0.3,
        hw1 in 0.05f32..0.3,
        hw2 in 0.05f32..0.3,
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
            Array2::from_shape_vec((1, 3), vec![cl0, cl1, cl2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![cu0, cu1, cu2]).unwrap(),
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
                    let rv = rmsnorm_eval(&point, &ny, eps);
                    let lower_val = cl0 * rv[0] + cl1 * rv[1] + cl2 * rv[2];
                    let upper_val = cu0 * rv[0] + cu1 * rv[1] + cu2 * rv[2];

                    prop_assert!(
                        lower_val >= crown_lower[0] - SAMPLING_CROWN_TOLERANCE,
                        "RmsNorm asymmetric lower: cl.rn({x0},{x1},{x2})={lower_val} < lb={}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        upper_val <= crown_upper[0] + SAMPLING_CROWN_TOLERANCE,
                        "RmsNorm asymmetric upper: cu.rn({x0},{x1},{x2})={upper_val} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }

    /// LayerNorm CROWN backward with asymmetric incoming coefficients.
    ///
    /// lower_a != upper_a exercises the coefficient split path where positive
    /// and negative coefficient handling differs for lower vs upper bounds.
    /// LayerNorm is the mean+variance normalization (vs RmsNorm which is RMS-only).
    /// Part of #2167.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_crown_asymmetric_incoming(
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        hw0 in 0.05f32..0.3,
        hw1 in 0.05f32..0.3,
        hw2 in 0.05f32..0.3,
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
            Array2::from_shape_vec((1, 3), vec![cl0, cl1, cl2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![cu0, cu1, cu2]).unwrap(),
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
                    let lower_val = cl0 * lv[0] + cl1 * lv[1] + cl2 * lv[2];
                    let upper_val = cu0 * lv[0] + cu1 * lv[1] + cu2 * lv[2];

                    prop_assert!(
                        lower_val >= crown_lower[0] - SAMPLING_CROWN_TOLERANCE,
                        "LayerNorm asymmetric lower: cl.ln({x0},{x1},{x2})={lower_val} < lb={}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        upper_val <= crown_upper[0] + SAMPLING_CROWN_TOLERANCE,
                        "LayerNorm asymmetric upper: cu.ln({x0},{x1},{x2})={upper_val} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }

    /// InstanceNorm1d CROWN backward with asymmetric incoming coefficients.
    /// 1 channel, 4 timesteps.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_instancenorm_crown_asymmetric_incoming(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0, c3 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Lower incoming coefficients
        cl0 in -2.0f32..2.0, cl1 in -2.0f32..2.0,
        cl2 in -2.0f32..2.0, cl3 in -2.0f32..2.0,
        // Upper incoming coefficients (different from lower)
        cu0 in -2.0f32..2.0, cu1 in -2.0f32..2.0,
        cu2 in -2.0f32..2.0, cu3 in -2.0f32..2.0,
    ) {
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01 || cl3.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01 || cu3.abs() > 0.01);
        prop_assume!(
            (cl0 - cu0).abs() > 0.01 || (cl1 - cu1).abs() > 0.01
                || (cl2 - cu2).abs() > 0.01 || (cl3 - cu3).abs() > 0.01
        );

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
            Array2::from_shape_vec((1, 4), vec![cl0, cl1, cl2, cl3]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 4), vec![cu0, cu1, cu2, cu3]).unwrap(),
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
            let y_ch = instance_norm_eval_channel(&x_ch, 1.5, -0.5, eps);
            let lower_val = cl0 * y_ch[0] + cl1 * y_ch[1] + cl2 * y_ch[2] + cl3 * y_ch[3];
            let upper_val = cu0 * y_ch[0] + cu1 * y_ch[1] + cu2 * y_ch[2] + cu3 * y_ch[3];

            prop_assert!(
                lower_val >= cl[0] - SAMPLING_CROWN_TOLERANCE,
                "InstanceNorm asymmetric lower: {lower_val} < {}",
                cl[0]
            );
            prop_assert!(
                upper_val <= cu[0] + SAMPLING_CROWN_TOLERANCE,
                "InstanceNorm asymmetric upper: {upper_val} > {}",
                cu[0]
            );
        }
    }

    /// AdaIN1d CROWN backward with asymmetric incoming coefficients.
    /// 1 channel, 3 timesteps.
    /// Part of #3103.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_adain_crown_asymmetric_incoming(
        c0 in -2.0f32..2.0, c1 in -2.0f32..2.0, c2 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Lower incoming coefficients
        cl0 in -2.0f32..2.0, cl1 in -2.0f32..2.0, cl2 in -2.0f32..2.0,
        // Upper incoming coefficients (different from lower)
        cu0 in -2.0f32..2.0, cu1 in -2.0f32..2.0, cu2 in -2.0f32..2.0,
        sg in 0.5f32..2.0, sb in -1.0f32..1.0,
    ) {
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01);
        prop_assume!(
            (cl0 - cu0).abs() > 0.01
                || (cl1 - cu1).abs() > 0.01
                || (cl2 - cu2).abs() > 0.01
        );

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
            Array2::from_shape_vec((1, 3), vec![cl0, cl1, cl2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![cu0, cu1, cu2]).unwrap(),
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
        let (crown_l, crown_u) = concretize_crown(&result, &flat_input);

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
            let lower_val = cl0 * y_ch[0] + cl1 * y_ch[1] + cl2 * y_ch[2];
            let upper_val = cu0 * y_ch[0] + cu1 * y_ch[1] + cu2 * y_ch[2];

            prop_assert!(
                lower_val >= crown_l[0] - SAMPLING_CROWN_TOLERANCE,
                "AdaIN asymmetric lower: {lower_val} < {}",
                crown_l[0]
            );
            prop_assert!(
                upper_val <= crown_u[0] + SAMPLING_CROWN_TOLERANCE,
                "AdaIN asymmetric upper: {upper_val} > {}",
                crown_u[0]
            );
        }
    }
}
