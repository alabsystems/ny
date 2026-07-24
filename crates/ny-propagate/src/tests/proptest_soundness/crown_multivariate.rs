// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward soundness proptests for multivariate layers (Softmax, LayerNorm, BatchNorm).
//!
//! Unlike element-wise activations (tested in crown_elementwise.rs and crown_piecewise.rs),
//! these layers have full n×n Jacobians — softmax, layer normalization, and other
//! multi-dimensional operations where each output depends on all inputs.
//!
//! The soundness property tested: for identity incoming bounds, the concretized CROWN
//! bounds must contain f(x) for all x in the input interval [lower, upper].
//!
//! LogSoftmax, CausalSoftmax, and asymmetric incoming coefficient tests are in
//! [`crown_multivariate_asymmetric`](super::crown_multivariate_asymmetric).
//! Part of #40.

use crate::layers::common::BoundPropagation;
use crate::layers::normalization::{LayerNormCrownMode, LayerNormLayer};
use crate::layers::softmax::SoftmaxLayer;
use crate::layers::BatchNormLayer;
use crate::layers::Layer;
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{batchnorm, sample_points};

/// Tolerance for CROWN backward soundness checks.
/// Nonlinear relaxations introduce approximation error, so we allow a wider margin
/// than the element-wise tolerance. Softmax LSE-based bounds are provably sound
/// but concretization involves floating-point summation of n terms.
const CROWN_MULTI_TOLERANCE: f32 = 1e-4;

/// Concretize CROWN linear bounds against input interval bounds.
/// Returns (lower_bounds, upper_bounds) as Vecs.
fn concretize_crown(result: &LinearBounds, pre_activation: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

// =============================================================================
// SOFTMAX CROWN BACKWARD SOUNDNESS (SOUND MODE: LSE-BASED)
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Softmax CROWN backward soundness (1D, sound mode).
    ///
    /// Verifies that for any x in [lower, upper], softmax(x) is within the
    /// concretized CROWN bounds. Uses sound LSE-based affine relaxation.
    ///
    /// Reference: "Fast and Complete Verification of Neural Networks" (Shi et al., 2020)
    /// for the LSE-based convex/concave envelope of softmax components.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_softmax_crown_1d_sound(
        l0 in -5.0f32..5.0,
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.01f32..3.0,
    ) {
        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);

        prop_assume!(u0 > l0 + 0.001);
        prop_assume!(u1 > l1 + 0.001);
        prop_assume!(u2 > l2 + 0.001);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);
        let layer = Layer::Softmax(softmax.clone());

        // Identity incoming bounds (3 outputs -> 3 inputs for CROWN)
        let identity = LinearBounds::identity(3);

        // Call via trait dispatch (the actual path used during CROWN propagation)
        let result = layer
            .propagate_crown_backward(&identity, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        // Concretize bounds
        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // Sample points and verify soundness
        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let softmax_val = softmax.eval(&point);

                    for i in 0..3 {
                        prop_assert!(
                            softmax_val[i] >= crown_lower[i] - CROWN_MULTI_TOLERANCE,
                            "Softmax CROWN lower bound violation at dim {i}: \
                             softmax([{x0},{x1},{x2}])[{i}]={} < lb={}",
                            softmax_val[i], crown_lower[i]
                        );
                        prop_assert!(
                            softmax_val[i] <= crown_upper[i] + CROWN_MULTI_TOLERANCE,
                            "Softmax CROWN upper bound violation at dim {i}: \
                             softmax([{x0},{x1},{x2}])[{i}]={} > ub={}",
                            softmax_val[i], crown_upper[i]
                        );
                    }
                }
            }
        }
    }

    /// Softmax CROWN backward with non-identity incoming coefficients.
    ///
    /// Tests that CROWN composition with arbitrary incoming linear bounds preserves
    /// soundness. This exercises the coefficient composition path in apply_affine_bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_softmax_crown_nonidentity_incoming(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        // Incoming coefficients: 1 output that combines all 3 softmax outputs
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        prop_assume!(u0 > l0 + 0.001);
        prop_assume!(u1 > l1 + 0.001);
        prop_assume!(u2 > l2 + 0.001);
        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01 || c2.abs() > 0.01);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);
        let layer = Layer::Softmax(softmax.clone());

        // Non-identity incoming: 1 output = c0*softmax_0 + c1*softmax_1 + c2*softmax_2
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![c0, c1, c2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![c0, c1, c2]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // Sample and verify: c . softmax(x) should be in [crown_lower, crown_upper]
        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let sv = softmax.eval(&point);
                    let combined = c0 * sv[0] + c1 * sv[1] + c2 * sv[2];

                    prop_assert!(
                        combined >= crown_lower[0] - CROWN_MULTI_TOLERANCE,
                        "Softmax CROWN (non-identity) lower violation: \
                         c.softmax([{x0},{x1},{x2}])={combined} < lb={}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        combined <= crown_upper[0] + CROWN_MULTI_TOLERANCE,
                        "Softmax CROWN (non-identity) upper violation: \
                         c.softmax([{x0},{x1},{x2}])={combined} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }
}

// =============================================================================
// LAYERNORM CROWN BACKWARD STRUCTURAL TESTS (SAMPLING MODE)
// =============================================================================
//
// LayerNorm sampling-based CROWN is a HEURISTIC linearization that is explicitly
// not provably sound. These tests verify structural correctness: no panics,
// finite bounds, correct dimensions, and lower <= upper.
//
// Soundness proptests (verifying bounds contain sampled true outputs) are in
// crown_normalization.rs and crown_normalization_asymmetric.rs, alongside the
// RmsNorm/InstanceNorm/AdaIN soundness tests. Those use SAMPLING_CROWN_TOLERANCE
// = 1e-2 and pass with the 3^n grid + 1.5x safety factor sampling strategy.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    /// LayerNorm CROWN backward structural test (sampling mode).
    ///
    /// Verifies that the sampling-based CROWN path:
    /// 1. Does not panic or return errors
    /// 2. Produces finite bounds
    /// 3. Maintains lower <= upper after concretization
    ///
    /// Does NOT assert soundness — sampling mode has known violations (tracked in #40).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn structural_layernorm_crown_sampling_identity_params(
        l0 in -5.0f32..5.0,
        d0 in 0.1f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.1f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.1f32..3.0,
        l3 in -5.0f32..5.0,
        d3 in 0.1f32..3.0,
    ) {
        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);
        let u3 = (l3 + d3).min(5.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer_norm = LayerNormLayer::new(
            Array1::ones(4),
            Array1::zeros(4),
            1e-5,
        ).unwrap().with_crown_mode(LayerNormCrownMode::Sampling);
        let layer = Layer::LayerNorm(layer_norm);

        let identity = LinearBounds::identity(4);

        let result = layer
            .propagate_crown_backward(&identity, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // Structural checks
        prop_assert_eq!(crown_lower.len(), 4, "Expected 4 lower bounds");
        prop_assert_eq!(crown_upper.len(), 4, "Expected 4 upper bounds");

        for i in 0..4 {
            prop_assert!(
                crown_lower[i].is_finite(),
                "Lower bound at dim {i} is not finite: {}",
                crown_lower[i]
            );
            prop_assert!(
                crown_upper[i].is_finite(),
                "Upper bound at dim {i} is not finite: {}",
                crown_upper[i]
            );
            prop_assert!(
                crown_lower[i] <= crown_upper[i] + 1e-6,
                "Lower > upper at dim {i}: {} > {}",
                crown_lower[i], crown_upper[i]
            );
        }
    }

    /// LayerNorm CROWN backward structural test with non-trivial ny/beta.
    ///
    /// Verifies structural correctness of the affine transform composition path.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn structural_layernorm_crown_sampling_with_params(
        l0 in -3.0f32..3.0,
        d0 in 0.1f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.1f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.1f32..2.0,
        l3 in -3.0f32..3.0,
        d3 in 0.1f32..2.0,
        g0 in 0.5f32..2.0,
        g1 in 0.5f32..2.0,
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
    ) {
        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);
        let u3 = (l3 + d3).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![g0, g1, g0, g1]);
        let beta = Array1::from_vec(vec![b0, b1, b0, b1]);
        let layer_norm = LayerNormLayer::new(ny, beta, 1e-5).unwrap()
            .with_crown_mode(LayerNormCrownMode::Sampling);
        let layer = Layer::LayerNorm(layer_norm);

        let identity = LinearBounds::identity(4);

        let result = layer
            .propagate_crown_backward(&identity, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // Structural checks
        prop_assert_eq!(crown_lower.len(), 4, "Expected 4 lower bounds");
        prop_assert_eq!(crown_upper.len(), 4, "Expected 4 upper bounds");

        for i in 0..4 {
            prop_assert!(
                crown_lower[i].is_finite(),
                "Lower bound at dim {i} is not finite: {}",
                crown_lower[i]
            );
            prop_assert!(
                crown_upper[i].is_finite(),
                "Upper bound at dim {i} is not finite: {}",
                crown_upper[i]
            );
            prop_assert!(
                crown_lower[i] <= crown_upper[i] + 1e-6,
                "Lower > upper at dim {i}: {} > {}",
                crown_lower[i], crown_upper[i]
            );
        }
    }
}

// =============================================================================
// BATCHNORM CROWN BACKWARD SOUNDNESS (EXACT AFFINE TRANSFORM)
// =============================================================================
//
// BatchNorm at inference time is y = scale * x + bias (affine). CROWN backward
// should compose exactly: A_new = A @ diag(scale), b_new = b + A @ bias.
// Since this is exact (no relaxation), we use a tight FP tolerance.

/// Tight tolerance for BatchNorm CROWN soundness.
/// BatchNorm is an exact affine transform at inference, so CROWN backward
/// is also exact. The only error source is FP arithmetic during concretization.
const BATCHNORM_CROWN_TOLERANCE: f32 = 1e-5;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// BatchNorm CROWN backward soundness with identity incoming bounds.
    ///
    /// Verifies that for any x in [lower, upper], batchnorm(x) is within the
    /// concretized CROWN bounds. Since BatchNorm is affine, this should hold
    /// with very tight tolerance.
    ///
    /// Includes negative ny values to test the negative-scale CROWN path.
    /// Reference: designs/2026-01-29-crown-affine-negative-scale.md
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_batchnorm_crown_identity(
        l0 in -5.0f32..5.0,
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.01f32..3.0,
        // Running statistics
        mean0 in -2.0f32..2.0,
        mean1 in -2.0f32..2.0,
        mean2 in -2.0f32..2.0,
        var0 in 0.1f32..5.0,
        var1 in 0.1f32..5.0,
        var2 in 0.1f32..5.0,
        // Scale (ny) — allow negative for negative-scale path
        gamma0 in -2.0f32..2.0,
        gamma1 in -2.0f32..2.0,
        gamma2 in -2.0f32..2.0,
        // Shift (beta)
        beta0 in -1.0f32..1.0,
        beta1 in -1.0f32..1.0,
        beta2 in -1.0f32..1.0,
    ) {
        prop_assume!(gamma0.abs() > 0.01);
        prop_assume!(gamma1.abs() > 0.01);
        prop_assume!(gamma2.abs() > 0.01);

        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![gamma0, gamma1, gamma2]);
        let beta = Array1::from_vec(vec![beta0, beta1, beta2]);
        let running_mean = Array1::from_vec(vec![mean0, mean1, mean2]);
        let running_var = Array1::from_vec(vec![var0, var1, var2]);

        let bn = BatchNormLayer::new(
            &ny.clone().into_dyn(),
            &beta.clone().into_dyn(),
            &running_mean.clone().into_dyn(),
            &running_var.clone().into_dyn(),
            1e-5,
        ).unwrap();
        let layer = Layer::BatchNorm(bn);

        let identity = LinearBounds::identity(3);

        let result = layer
            .propagate_crown_backward(&identity, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // Sample points and verify soundness
        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let bn_val = batchnorm(
                        &point, &ny, &beta, &running_mean, &running_var, 1e-5,
                    );

                    for i in 0..3 {
                        prop_assert!(
                            bn_val[i] >= crown_lower[i] - BATCHNORM_CROWN_TOLERANCE,
                            "BatchNorm CROWN lower violation at dim {i}: \
                             bn([{x0},{x1},{x2}])[{i}]={} < lb={}\n\
                             ny={:?}, beta={:?}, mean={:?}, var={:?}",
                            bn_val[i], crown_lower[i], ny, beta, running_mean, running_var
                        );
                        prop_assert!(
                            bn_val[i] <= crown_upper[i] + BATCHNORM_CROWN_TOLERANCE,
                            "BatchNorm CROWN upper violation at dim {i}: \
                             bn([{x0},{x1},{x2}])[{i}]={} > ub={}\n\
                             ny={:?}, beta={:?}, mean={:?}, var={:?}",
                            bn_val[i], crown_upper[i], ny, beta, running_mean, running_var
                        );
                    }
                }
            }
        }
    }

    /// BatchNorm CROWN backward with non-identity incoming coefficients.
    ///
    /// Tests that CROWN composition with arbitrary incoming linear bounds preserves
    /// soundness through the affine BatchNorm transform. Since BatchNorm is linear,
    /// this should be exact.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_batchnorm_crown_nonidentity_incoming(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        // Incoming coefficients: 1 output = c0*bn_0 + c1*bn_1 + c2*bn_2
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        // Parameters
        gamma0 in -2.0f32..2.0,
        gamma1 in -2.0f32..2.0,
        var0 in 0.1f32..3.0,
        var1 in 0.1f32..3.0,
    ) {
        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01 || c2.abs() > 0.01);
        prop_assume!(gamma0.abs() > 0.01);
        prop_assume!(gamma1.abs() > 0.01);

        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![gamma0, gamma1, 1.0]);
        let beta = Array1::zeros(3);
        let running_mean = Array1::zeros(3);
        let running_var = Array1::from_vec(vec![var0, var1, 1.0]);

        let bn = BatchNormLayer::new(
            &ny.clone().into_dyn(),
            &beta.clone().into_dyn(),
            &running_mean.clone().into_dyn(),
            &running_var.clone().into_dyn(),
            1e-5,
        ).unwrap();
        let layer = Layer::BatchNorm(bn);

        // Non-identity incoming: 1 output combining all 3 BatchNorm outputs
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![c0, c1, c2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![c0, c1, c2]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // Sample and verify: c . batchnorm(x) in [crown_lower, crown_upper]
        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let bn_val = batchnorm(
                        &point, &ny, &beta, &running_mean, &running_var, 1e-5,
                    );
                    let combined = c0 * bn_val[0] + c1 * bn_val[1] + c2 * bn_val[2];

                    prop_assert!(
                        combined >= crown_lower[0] - BATCHNORM_CROWN_TOLERANCE,
                        "BatchNorm CROWN (non-identity) lower violation: \
                         c.bn([{x0},{x1},{x2}])={combined} < lb={}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        combined <= crown_upper[0] + BATCHNORM_CROWN_TOLERANCE,
                        "BatchNorm CROWN (non-identity) upper violation: \
                         c.bn([{x0},{x1},{x2}])={combined} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }
}
