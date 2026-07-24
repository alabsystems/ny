// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! LayerNorm CROWN backward proptest soundness tests: batched IbpValidated + MeanOnly.
//!
//! Split from crown_normalization.rs to keep file size under 1000 lines.
//! The scalar structural `Sampling` coverage remains in crown_normalization.rs.
//! These containment tests use `LayerNormCrownMode::IbpValidated`, the sound
//! normalization CROWN mode.
//!
//! Part of #2426.

use crate::layers::normalization::{LayerNormCrownMode, LayerNormLayer, LayerNormMode};
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{layernorm_mean_only, sample_points};

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
// LAYERNORM BATCHED CROWN BACKWARD SOUNDNESS (IBP-VALIDATED MODE)
// =============================================================================
//
// The batched path (propagate_linear_batched_with_bounds) delegates to the
// scalar path per batch position. These tests verify the batched wrapper
// itself doesn't break soundness during reshape/reassembly.
//
// Part of #2426.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LayerNorm batched CROWN soundness with identity incoming, tight perturbation.
    /// Verifies that for sampled concrete inputs within bounds, the true LayerNorm
    /// output falls within the CROWN-computed bounds for a 2-batch input.
    /// Part of #2426.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_batched_crown_identity_tight(
        // Batch 0 centers
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        // Batch 1 centers
        c3 in -2.0f32..2.0,
        c4 in -2.0f32..2.0,
        c5 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        g2 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
    ) {
        let norm_size = 3;
        let batch = 2;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), upper_v.clone()).unwrap(),
        ).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        // Identity batched bounds: [batch, norm_size, norm_size]
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, norm_size, norm_size]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, norm_size, norm_size]));
        for b in 0..batch {
            for i in 0..norm_size {
                la[[b, i, i]] = 1.0;
                ua[[b, i, i]] = 1.0;
            }
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[batch, norm_size]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[batch, norm_size]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![batch, norm_size],
            vec![batch, norm_size],
        ).unwrap();

        let result = ln
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "LayerNorm batched CROWN failed: {e}"
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
                let y_true = super::layernorm(&x_batch, &ny, &beta, eps);
                for i in 0..norm_size {
                    let idx = b * norm_size + i;
                    prop_assert!(
                        y_true[i] >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                        "Batched LN CROWN lower at batch {b} dim {i}: {} < {}",
                        y_true[i],
                        crown_lower[idx],
                    );
                    prop_assert!(
                        y_true[i] <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                        "Batched LN CROWN upper at batch {b} dim {i}: {} > {}",
                        y_true[i],
                        crown_upper[idx],
                    );
                }
            }
        }
    }
}

// =============================================================================
// LAYERNORM BATCHED CROWN BACKWARD SOUNDNESS (IBP-VALIDATED, NON-TRIVIAL INCOMING)
// =============================================================================
//
// Prover P1#914 finding: batched soundness coverage only used identity incoming.
// This test uses non-trivial (mixed-sign) incoming coefficients to exercise
// the batched CROWN backward path with coefficient composition.
//
// Part of #2426.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LayerNorm batched CROWN soundness with non-trivial incoming coefficients.
    /// Uses a single output row with mixed-sign coefficients per batch position,
    /// verifying that the composed CROWN bounds contain the true output.
    /// Part of #2426.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_batched_crown_negcoeff(
        // Batch 0 centers
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        // Batch 1 centers
        c3 in -2.0f32..2.0,
        c4 in -2.0f32..2.0,
        c5 in -2.0f32..2.0,
        hw in 0.05f32..0.3,
        // Non-trivial incoming coefficients (one set per batch position)
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
        let beta = Array1::zeros(norm_size);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), upper_v.clone()).unwrap(),
        ).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), eps)
            .unwrap()
            .with_crown_mode(LayerNormCrownMode::IbpValidated);

        // Non-trivial batched bounds: [batch, out_dim=1, norm_size]
        // Same coefficients for both batch positions.
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

        let result = ln
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "LayerNorm batched CROWN negcoeff failed: {e}"
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
                let y_true = super::layernorm(&x_batch, &ny, &beta, eps);
                // Apply incoming coefficients: combined = ic . y_true
                let combined = ic0 * y_true[0] + ic1 * y_true[1] + ic2 * y_true[2];
                let idx = b; // out_dim=1, so index is just batch

                prop_assert!(
                    combined >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched LN CROWN negcoeff lower at batch {b}: {combined} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    combined <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched LN CROWN negcoeff upper at batch {b}: {combined} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}

// =============================================================================
// LAYERNORM MEAN-ONLY CROWN BACKWARD SOUNDNESS
// =============================================================================
//
// MeanOnly LayerNorm: y = ny * (x - mean(x)) + beta
// This is an affine function of x (linear in each x_i), so the CROWN
// linearization should be exact (no sampling needed). Test both scalar
// and batched paths.
//
// Part of #2426.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// LayerNorm MeanOnly CROWN scalar soundness with identity incoming.
    /// Since mean-only is affine, the CROWN bounds should tightly contain
    /// all evaluations.
    /// Part of #2426.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_mean_only_crown_identity(
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
        c2 in -3.0f32..3.0,
        hw0 in 0.05f32..1.0,
        hw1 in 0.05f32..1.0,
        hw2 in 0.05f32..1.0,
        g0 in 0.5f32..3.0,
        g1 in 0.5f32..3.0,
        g2 in 0.5f32..3.0,
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

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5)
            .unwrap()
            .with_mode(LayerNormMode::MeanOnly);

        let identity = LinearBounds::identity(3);

        let result = ln
            .propagate_linear_with_bounds(&identity, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "LayerNorm MeanOnly CROWN failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // MeanOnly is affine, so use tighter tolerance than sampling-based
        let mean_only_tol = 1e-5;

        let s0_pts = sample_points(l0, u0, 5);
        let s1_pts = sample_points(l1, u1, 5);
        let s2_pts = sample_points(l2, u2, 5);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let y = layernorm_mean_only(&point, &ny, &beta);

                    for i in 0..3 {
                        prop_assert!(
                            y[i] >= crown_lower[i] - mean_only_tol,
                            "MeanOnly CROWN lower at dim {i}: eval({x0},{x1},{x2})[{i}]={} < lb={}",
                            y[i],
                            crown_lower[i],
                        );
                        prop_assert!(
                            y[i] <= crown_upper[i] + mean_only_tol,
                            "MeanOnly CROWN upper at dim {i}: eval({x0},{x1},{x2})[{i}]={} > ub={}",
                            y[i],
                            crown_upper[i],
                        );
                    }
                }
            }
        }
    }

    /// LayerNorm MeanOnly CROWN soundness with negative-coefficient incoming.
    /// Part of #2426.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_mean_only_crown_negcoeff(
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
        c2 in -3.0f32..3.0,
        hw0 in 0.05f32..1.0,
        hw1 in 0.05f32..1.0,
        hw2 in 0.05f32..1.0,
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

        let ny = Array1::ones(3);
        let beta = Array1::zeros(3);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5)
            .unwrap()
            .with_mode(LayerNormMode::MeanOnly);

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
                TestCaseError::fail(format!(
                    "LayerNorm MeanOnly CROWN failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        let mean_only_tol = 1e-5;

        let s0_pts = sample_points(l0, u0, 5);
        let s1_pts = sample_points(l1, u1, 5);
        let s2_pts = sample_points(l2, u2, 5);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let lv = layernorm_mean_only(&point, &ny, &beta);
                    let combined = ic0 * lv[0] + ic1 * lv[1] + ic2 * lv[2];

                    prop_assert!(
                        combined >= crown_lower[0] - mean_only_tol,
                        "MeanOnly negcoeff lower: {combined} < {}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        combined <= crown_upper[0] + mean_only_tol,
                        "MeanOnly negcoeff upper: {combined} > {}",
                        crown_upper[0]
                    );
                }
            }
        }
    }
}

// =============================================================================
// LAYERNORM MEAN-ONLY BATCHED CROWN BACKWARD SOUNDNESS
// =============================================================================
//
// Part of #2426.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LayerNorm MeanOnly batched CROWN soundness with identity incoming.
    /// Part of #2426.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_mean_only_batched_crown_identity(
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
        c2 in -3.0f32..3.0,
        c3 in -3.0f32..3.0,
        c4 in -3.0f32..3.0,
        c5 in -3.0f32..3.0,
        hw in 0.05f32..1.0,
        g0 in 0.5f32..3.0,
        g1 in 0.5f32..3.0,
        g2 in 0.5f32..3.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        b2 in -1.0f32..1.0,
    ) {
        let norm_size = 3;
        let batch = 2;
        let ny = Array1::from_vec(vec![g0, g1, g2]);
        let beta = Array1::from_vec(vec![b0, b1, b2]);

        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), upper_v.clone()).unwrap(),
        ).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5)
            .unwrap()
            .with_mode(LayerNormMode::MeanOnly);

        // Identity batched bounds: [batch, norm_size, norm_size]
        let mut la = ArrayD::<f32>::zeros(IxDyn(&[batch, norm_size, norm_size]));
        let mut ua = ArrayD::<f32>::zeros(IxDyn(&[batch, norm_size, norm_size]));
        for b in 0..batch {
            for i in 0..norm_size {
                la[[b, i, i]] = 1.0;
                ua[[b, i, i]] = 1.0;
            }
        }
        let lb = ArrayD::<f32>::zeros(IxDyn(&[batch, norm_size]));
        let ub = ArrayD::<f32>::zeros(IxDyn(&[batch, norm_size]));
        let bounds = BatchedLinearBounds::new(
            la, lb, ua, ub,
            vec![batch, norm_size],
            vec![batch, norm_size],
        ).unwrap();

        let result = ln
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "LayerNorm MeanOnly batched CROWN failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        let mean_only_tol = 1e-5;

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
                let y_true = layernorm_mean_only(&x_batch, &ny, &beta);
                for i in 0..norm_size {
                    let idx = b * norm_size + i;
                    prop_assert!(
                        y_true[i] >= crown_lower[idx] - mean_only_tol,
                        "MeanOnly batched CROWN lower at batch {b} dim {i}: {} < {}",
                        y_true[i],
                        crown_lower[idx],
                    );
                    prop_assert!(
                        y_true[i] <= crown_upper[idx] + mean_only_tol,
                        "MeanOnly batched CROWN upper at batch {b} dim {i}: {} > {}",
                        y_true[i],
                        crown_upper[idx],
                    );
                }
            }
        }
    }
}

// =============================================================================
// LAYERNORM MEAN-ONLY CROWN: ASYMMETRIC AND BATCHED NEGCOEFF
// =============================================================================
//
// MeanOnly CROWN has identity and negcoeff (symmetric la==ua) scalar tests,
// plus a batched identity test. These two tests fill the remaining gaps:
// asymmetric incoming (la != ua) and batched with non-trivial coefficients.
//
// Part of #3333.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LayerNorm MeanOnly CROWN scalar soundness with asymmetric incoming (la != ua).
    ///
    /// In multi-layer CROWN backward propagation, lower and upper incoming
    /// coefficients diverge. MeanOnly is affine so the bounds should be tight,
    /// but the coefficient split path must be exercised.
    /// Part of #3333.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_mean_only_crown_asymmetric(
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
        c2 in -3.0f32..3.0,
        hw0 in 0.05f32..1.0,
        hw1 in 0.05f32..1.0,
        hw2 in 0.05f32..1.0,
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

        let ny = Array1::ones(3);
        let beta = Array1::zeros(3);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5)
            .unwrap()
            .with_mode(LayerNormMode::MeanOnly);

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
                TestCaseError::fail(format!(
                    "LayerNorm MeanOnly CROWN asymmetric failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        let mean_only_tol = 1e-5;

        let s0_pts = sample_points(l0, u0, 5);
        let s1_pts = sample_points(l1, u1, 5);
        let s2_pts = sample_points(l2, u2, 5);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let lv = layernorm_mean_only(&point, &ny, &beta);
                    // Lower bound uses cl coefficients, upper uses cu
                    let lower_val = cl0 * lv[0] + cl1 * lv[1] + cl2 * lv[2];
                    let upper_val = cu0 * lv[0] + cu1 * lv[1] + cu2 * lv[2];

                    prop_assert!(
                        lower_val >= crown_lower[0] - mean_only_tol,
                        "MeanOnly asymmetric lower: {lower_val} < {}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        upper_val <= crown_upper[0] + mean_only_tol,
                        "MeanOnly asymmetric upper: {upper_val} > {}",
                        crown_upper[0]
                    );
                }
            }
        }
    }

    /// LayerNorm MeanOnly batched CROWN soundness with non-trivial incoming.
    ///
    /// The existing batched MeanOnly test only uses identity incoming.
    /// This tests mixed-sign coefficients to exercise the batched composition path.
    /// Part of #3333.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_mean_only_batched_crown_negcoeff(
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
        c2 in -3.0f32..3.0,
        c3 in -3.0f32..3.0,
        c4 in -3.0f32..3.0,
        c5 in -3.0f32..3.0,
        hw in 0.05f32..1.0,
        ic0 in -2.0f32..2.0,
        ic1 in -2.0f32..2.0,
        ic2 in -2.0f32..2.0,
    ) {
        prop_assume!(ic0.abs() > 0.01 || ic1.abs() > 0.01 || ic2.abs() > 0.01);

        let norm_size = 3;
        let batch = 2;
        let out_dim = 1;
        let ny = Array1::ones(norm_size);
        let beta = Array1::zeros(norm_size);

        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), upper_v.clone()).unwrap(),
        ).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), 1e-5)
            .unwrap()
            .with_mode(LayerNormMode::MeanOnly);

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

        let result = ln
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "LayerNorm MeanOnly batched CROWN negcoeff failed: {e}"
                ))
            })?;

        let (crown_lower, crown_upper) = concretize_batched_crown(&result, &input);

        let mean_only_tol = 1e-5;

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
                let y_true = layernorm_mean_only(&x_batch, &ny, &beta);
                let combined = ic0 * y_true[0] + ic1 * y_true[1] + ic2 * y_true[2];
                let idx = b;

                prop_assert!(
                    combined >= crown_lower[idx] - mean_only_tol,
                    "MeanOnly batched negcoeff lower at batch {b}: {combined} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    combined <= crown_upper[idx] + mean_only_tol,
                    "MeanOnly batched negcoeff upper at batch {b}: {combined} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}

// =============================================================================
// LAYERNORM BATCHED CROWN BACKWARD SOUNDNESS (IBP-VALIDATED, ASYMMETRIC INCOMING)
// =============================================================================
//
// Tests the batched CROWN backward path with asymmetric incoming coefficients
// (lower_a != upper_a). In real CROWN backward propagation, after composing
// through several layers, lower_a and upper_a diverge. This exercises the
// coefficient split path in the batched wrapper.
//
// Part of #3174.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// LayerNorm batched CROWN soundness with asymmetric incoming (la != ua).
    /// Part of #3174.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_layernorm_batched_crown_asymmetric(
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
        let beta = Array1::zeros(norm_size);
        let eps = 1e-5_f32;

        let centers = [c0, c1, c2, c3, c4, c5];
        let lower_v: Vec<f32> = centers.iter().map(|&c| c - hw).collect();
        let upper_v: Vec<f32> = centers.iter().map(|&c| c + hw).collect();

        let input = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), lower_v.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&[batch, norm_size]), upper_v.clone()).unwrap(),
        ).unwrap();

        let ln = LayerNormLayer::new(ny.clone(), beta.clone(), eps)
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

        let result = ln
            .propagate_linear_batched_with_bounds(&bounds, &input)
            .map_err(|e| {
                TestCaseError::fail(format!(
                    "LayerNorm batched CROWN asymmetric failed: {e}"
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
                let y_true = super::layernorm(&x_batch, &ny, &beta, eps);
                // Lower bound uses cl coefficients, upper uses cu
                let lower_val = cl0 * y_true[0] + cl1 * y_true[1] + cl2 * y_true[2];
                let upper_val = cu0 * y_true[0] + cu1 * y_true[1] + cu2 * y_true[2];
                let idx = b;

                prop_assert!(
                    lower_val >= crown_lower[idx] - SAMPLING_CROWN_TOLERANCE,
                    "Batched LN CROWN asymmetric lower at batch {b}: {lower_val} < {}",
                    crown_lower[idx],
                );
                prop_assert!(
                    upper_val <= crown_upper[idx] + SAMPLING_CROWN_TOLERANCE,
                    "Batched LN CROWN asymmetric upper at batch {b}: {upper_val} > {}",
                    crown_upper[idx],
                );
            }
        }
    }
}
