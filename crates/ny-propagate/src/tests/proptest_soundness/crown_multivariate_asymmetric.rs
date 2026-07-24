// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! CROWN backward soundness proptests for LogSoftmax, CausalSoftmax, and
//! asymmetric incoming coefficients on multivariate layers.
//!
//! Split from `crown_multivariate.rs` which retains Softmax, LayerNorm, and
//! BatchNorm CROWN tests with identity and symmetric non-identity incoming bounds.
//!
//! The asymmetric tests are critical: in real CROWN propagation, after composing
//! through several layers, lower_a and upper_a diverge. These tests verify
//! soundness with lower_a != upper_a for Softmax, LogSoftmax, BatchNorm, and
//! CausalSoftmax.
//!
//! Part of #40, #1793.

use crate::layers::common::BoundPropagation;
use crate::layers::softmax::{CausalSoftmaxLayer, LogSoftmaxLayer, SoftmaxLayer};
use crate::layers::BatchNormLayer;
use crate::layers::Layer;
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{batchnorm, causal_softmax, logsoftmax, sample_points};

/// Tolerance for CROWN backward soundness checks on multivariate layers.
/// Nonlinear relaxations introduce approximation error, so we allow a wider margin
/// than the element-wise tolerance. Softmax LSE-based bounds are provably sound
/// but concretization involves floating-point summation of n terms.
const CROWN_MULTI_TOLERANCE: f32 = 1e-4;

/// Tight tolerance for BatchNorm CROWN soundness.
/// BatchNorm is an exact affine transform at inference, so CROWN backward
/// is also exact. The only error source is FP arithmetic during concretization.
const BATCHNORM_CROWN_TOLERANCE: f32 = 1e-5;

/// Concretize CROWN linear bounds against input interval bounds.
/// Returns (lower_bounds, upper_bounds) as Vecs.
fn concretize_crown(result: &LinearBounds, pre_activation: &BoundedTensor) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

// =============================================================================
// LOGSOFTMAX CROWN BACKWARD SOUNDNESS (SOUND MODE: LSE-BASED)
// =============================================================================
//
// LogSoftmax: y_i = x_i - log(sum(exp(x_j)))
// Sound mode uses LSE-based affine bounds similar to softmax.
// Reference: alpha-beta-CROWN logsoftmax.py

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LogSoftmax CROWN backward soundness (sound mode, 3D).
    ///
    /// Verifies that for any x in [lower, upper], logsoftmax(x) is within the
    /// concretized CROWN bounds. Uses sound LSE-based affine relaxation.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsoftmax_crown_1d_sound(
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

        let logsoftmax_layer = LogSoftmaxLayer::new(-1).with_sound_mode(true);
        let layer = Layer::LogSoftmax(logsoftmax_layer);

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
                    let lsm_val = logsoftmax(&point);

                    for i in 0..3 {
                        prop_assert!(
                            lsm_val[i] >= crown_lower[i] - CROWN_MULTI_TOLERANCE,
                            "LogSoftmax CROWN lower violation at dim {i}: \
                             logsoftmax([{x0},{x1},{x2}])[{i}]={} < lb={}",
                            lsm_val[i], crown_lower[i]
                        );
                        prop_assert!(
                            lsm_val[i] <= crown_upper[i] + CROWN_MULTI_TOLERANCE,
                            "LogSoftmax CROWN upper violation at dim {i}: \
                             logsoftmax([{x0},{x1},{x2}])[{i}]={} > ub={}",
                            lsm_val[i], crown_upper[i]
                        );
                    }
                }
            }
        }
    }

    /// LogSoftmax CROWN backward with non-identity incoming coefficients.
    ///
    /// Tests CROWN composition through LogSoftmax with arbitrary incoming bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsoftmax_crown_nonidentity_incoming(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
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

        let logsoftmax_layer = LogSoftmaxLayer::new(-1).with_sound_mode(true);
        let layer = Layer::LogSoftmax(logsoftmax_layer);

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

        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let lsm_val = logsoftmax(&point);
                    let combined = c0 * lsm_val[0] + c1 * lsm_val[1] + c2 * lsm_val[2];

                    prop_assert!(
                        combined >= crown_lower[0] - CROWN_MULTI_TOLERANCE,
                        "LogSoftmax CROWN (non-identity) lower violation: \
                         c.logsoftmax([{x0},{x1},{x2}])={combined} < lb={}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        combined <= crown_upper[0] + CROWN_MULTI_TOLERANCE,
                        "LogSoftmax CROWN (non-identity) upper violation: \
                         c.logsoftmax([{x0},{x1},{x2}])={combined} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }
}

// =============================================================================
// CAUSAL SOFTMAX CROWN BACKWARD SOUNDNESS (SOUND MODE: IBP-DERIVED CONSTANTS)
// =============================================================================
//
// In sound mode, CausalSoftmax CROWN backward returns constant bounds derived
// from output IBP bounds (A=0, b=concretized IBP bounds). This proptest verifies
// that the IBP-derived constant CROWN bounds contain the true causal_softmax(x)
// for all sampled x in the input interval.
//
// CausalSoftmax operates on 2D input [seq_q, seq_k] where row i computes
// softmax over positions 0..=i (causal mask). The CROWN backward flattens
// the 2D output to 1D for the linear bounds.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// CausalSoftmax CROWN backward soundness (sound mode, 3x3 matrix).
    ///
    /// Verifies that for any x in [lower, upper], causal_softmax(x) is within
    /// the concretized CROWN bounds. Sound mode uses IBP-derived constant bounds.
    ///
    /// Uses 3x3 to exercise deeper causal structure:
    /// - Row 0: softmax over 1 element (trivially [1, 0, 0])
    /// - Row 1: softmax over 2 elements (non-trivial)
    /// - Row 2: softmax over 3 elements (non-trivial, full row)
    ///
    /// Active positions: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2) = 6 active.
    /// Masked positions: (0,1), (0,2), (1,2) = 3 masked (output always 0).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_causal_softmax_crown_3x3_sound(
        // 6 active positions: bounds for each
        l0 in -5.0f32..5.0, // (0,0)
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0, // (1,0)
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0, // (1,1)
        d2 in 0.01f32..3.0,
        l3 in -5.0f32..5.0, // (2,0)
        d3 in 0.01f32..3.0,
        l4 in -5.0f32..5.0, // (2,1)
        d4 in 0.01f32..3.0,
        l5 in -5.0f32..5.0, // (2,2)
        d5 in 0.01f32..3.0,
    ) {
        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);
        let u3 = (l3 + d3).min(5.0);
        let u4 = (l4 + d4).min(5.0);
        let u5 = (l5 + d5).min(5.0);

        // 3x3 input: row-major [row0: (0,0) (0,1) (0,2), row1: ..., row2: ...]
        // Masked positions get fixed bounds (don't affect output).
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![l0, 0.0, 0.0, l1, l2, 0.0, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[3, 3]),
            vec![u0, 0.0, 0.0, u1, u2, 0.0, u3, u4, u5],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let cs = CausalSoftmaxLayer::new(-1); // sound=true by default
        let layer = Layer::CausalSoftmax(cs);

        // Identity incoming bounds over flattened output (9 elements)
        let identity = LinearBounds::identity(9);

        let result = layer
            .propagate_crown_backward(&identity, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // Sample the 6 active dimensions, holding others at midpoint
        let active_lowers = [l0, l1, l2, l3, l4, l5];
        let active_uppers = [u0, u1, u2, u3, u4, u5];
        // Map from active index to flat 3x3 index
        let active_to_flat: [usize; 6] = [0, 3, 4, 6, 7, 8];
        let spts = 4;

        let mids_flat = [
            f32::midpoint(l0, u0), 0.0, 0.0,
            f32::midpoint(l1, u1), f32::midpoint(l2, u2), 0.0,
            f32::midpoint(l3, u3), f32::midpoint(l4, u4), f32::midpoint(l5, u5),
        ];

        // Per-dimension sampling: vary each active dim while others at midpoint
        for active_dim in 0..6 {
            let pts = sample_points(active_lowers[active_dim], active_uppers[active_dim], spts);
            for &val in &pts {
                let mut flat = mids_flat;
                flat[active_to_flat[active_dim]] = val;
                let point = Array2::from_shape_vec((3, 3), flat.to_vec()).unwrap();
                let cs_val = causal_softmax(&point);
                let out_flat: Vec<f32> = cs_val.iter().copied().collect();

                for i in 0..9 {
                    prop_assert!(
                        out_flat[i] >= crown_lower[i] - CROWN_MULTI_TOLERANCE,
                        "CausalSoftmax CROWN lower violation at dim {i}: \
                         cs(x)[{i}]={} < lb={}, active_dim={active_dim}, val={val}",
                        out_flat[i], crown_lower[i]
                    );
                    prop_assert!(
                        out_flat[i] <= crown_upper[i] + CROWN_MULTI_TOLERANCE,
                        "CausalSoftmax CROWN upper violation at dim {i}: \
                         cs(x)[{i}]={} > ub={}, active_dim={active_dim}, val={val}",
                        out_flat[i], crown_upper[i]
                    );
                }
            }
        }

        // Corner sampling: endpoints of all active dimensions
        // Row 2 has 3 active elements — test all 8 corners of (x20, x21, x22)
        for &x20 in &[l3, u3] {
            for &x21 in &[l4, u4] {
                for &x22 in &[l5, u5] {
                    let flat = [
                        f32::midpoint(l0, u0), 0.0, 0.0,
                        f32::midpoint(l1, u1), f32::midpoint(l2, u2), 0.0,
                        x20, x21, x22,
                    ];
                    let point = Array2::from_shape_vec((3, 3), flat.to_vec()).unwrap();
                    let cs_val = causal_softmax(&point);
                    let out_flat: Vec<f32> = cs_val.iter().copied().collect();

                    for i in 0..9 {
                        prop_assert!(
                            out_flat[i] >= crown_lower[i] - CROWN_MULTI_TOLERANCE,
                            "CausalSoftmax CROWN corner lower violation at dim {i}: \
                             cs(x)[{i}]={} < lb={}",
                            out_flat[i], crown_lower[i]
                        );
                        prop_assert!(
                            out_flat[i] <= crown_upper[i] + CROWN_MULTI_TOLERANCE,
                            "CausalSoftmax CROWN corner upper violation at dim {i}: \
                             cs(x)[{i}]={} > ub={}",
                            out_flat[i], crown_upper[i]
                        );
                    }
                }
            }
        }
    }

    /// CausalSoftmax CROWN backward with non-identity incoming bounds (sound mode).
    ///
    /// Tests that CROWN composition with arbitrary incoming linear bounds preserves
    /// soundness through the constant (IBP-derived) CausalSoftmax bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_causal_softmax_crown_nonidentity_incoming(
        // 2x2 input (minimal size for causal softmax)
        l00 in -3.0f32..3.0,
        d00 in 0.01f32..2.0,
        l01 in -3.0f32..3.0,
        d01 in 0.01f32..2.0,
        l10 in -3.0f32..3.0,
        d10 in 0.01f32..2.0,
        l11 in -3.0f32..3.0,
        d11 in 0.01f32..2.0,
        // Incoming coefficients: 1 output = sum(c_i * cs_i) over flattened 4 elements
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
        c2 in -2.0f32..2.0,
        c3 in -2.0f32..2.0,
    ) {
        let u00 = (l00 + d00).min(3.0);
        let u01 = (l01 + d01).min(3.0);
        let u10 = (l10 + d10).min(3.0);
        let u11 = (l11 + d11).min(3.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01 || c2.abs() > 0.01 || c3.abs() > 0.01);

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![l00, l01, l10, l11],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![u00, u01, u10, u11],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let cs = CausalSoftmaxLayer::new(-1); // sound mode
        let layer = Layer::CausalSoftmax(cs);

        // Non-identity incoming: 1 output combining all 4 elements
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 4), vec![c0, c1, c2, c3]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 4), vec![c0, c1, c2, c3]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // Sample and verify: c . causal_softmax(x) in [crown_lower, crown_upper]
        let spts = 5;
        let s00 = sample_points(l00, u00, spts);
        let s01 = sample_points(l01, u01, spts);
        let s10 = sample_points(l10, u10, spts);
        let s11 = sample_points(l11, u11, spts);

        for &x00 in &s00 {
            for &x01 in &s01 {
                for &x10 in &s10 {
                    for &x11 in &s11 {
                        let point = Array2::from_shape_vec(
                            (2, 2), vec![x00, x01, x10, x11],
                        ).unwrap();
                        let cs_val = causal_softmax(&point);
                        let flat: Vec<f32> = cs_val.iter().copied().collect();
                        let combined = c0 * flat[0] + c1 * flat[1]
                            + c2 * flat[2] + c3 * flat[3];

                        prop_assert!(
                            combined >= crown_lower[0] - CROWN_MULTI_TOLERANCE,
                            "CausalSoftmax CROWN (non-identity) lower violation: \
                             c.cs([{x00},{x01};{x10},{x11}])={combined} < lb={}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            combined <= crown_upper[0] + CROWN_MULTI_TOLERANCE,
                            "CausalSoftmax CROWN (non-identity) upper violation: \
                             c.cs([{x00},{x01};{x10},{x11}])={combined} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }
}

// =============================================================================
// ASYMMETRIC INCOMING COEFFICIENT TESTS
// =============================================================================
//
// All previous non-identity tests use symmetric coefficients (lower_a == upper_a).
// In real CROWN propagation, after composing through several layers, lower_a and
// upper_a diverge. These tests verify soundness with lower_a != upper_a.
//
// The soundness property: for any x in [lower, upper], the true output must satisfy
//   crown_lower <= f(x) <= crown_upper
// where crown_lower/upper come from concretizing the CROWN linear bounds.
// Part of #40.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// Softmax CROWN backward with asymmetric incoming coefficients.
    ///
    /// lower_a != upper_a -- different linear combinations bound from below and above.
    /// This exercises the coefficient split path in apply_affine_bounds where positive
    /// and negative coefficient handling differs for lower vs upper bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_softmax_crown_asymmetric_incoming(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        // Lower incoming coefficients
        cl0 in -2.0f32..2.0,
        cl1 in -2.0f32..2.0,
        cl2 in -2.0f32..2.0,
        // Upper incoming coefficients (different from lower)
        cu0 in -2.0f32..2.0,
        cu1 in -2.0f32..2.0,
        cu2 in -2.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        prop_assume!(u0 > l0 + 0.001);
        prop_assume!(u1 > l1 + 0.001);
        prop_assume!(u2 > l2 + 0.001);
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01);
        // Ensure asymmetry: at least one pair of lower/upper coefficients differs
        prop_assume!((cl0 - cu0).abs() > 0.01 || (cl1 - cu1).abs() > 0.01 || (cl2 - cu2).abs() > 0.01);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let softmax = SoftmaxLayer::new(-1).with_sound_mode(true);
        let layer = Layer::Softmax(softmax.clone());

        // Asymmetric incoming: lower_a != upper_a
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![cl0, cl1, cl2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![cu0, cu1, cu2]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        // For asymmetric incoming, the true value at any point x is:
        //   value = cl . softmax(x)   (for the lower-bound linear function)
        //   value = cu . softmax(x)   (for the upper-bound linear function)
        // The CROWN bounds must satisfy:
        //   crown_lower <= cl . softmax(x)   for all x
        //   cu . softmax(x) <= crown_upper   for all x
        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let sv = softmax.eval(&point);
                    let lower_val = cl0 * sv[0] + cl1 * sv[1] + cl2 * sv[2];
                    let upper_val = cu0 * sv[0] + cu1 * sv[1] + cu2 * sv[2];

                    prop_assert!(
                        lower_val >= crown_lower[0] - CROWN_MULTI_TOLERANCE,
                        "Softmax CROWN (asymmetric) lower violation: \
                         cl.softmax([{x0},{x1},{x2}])={lower_val} < lb={}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        upper_val <= crown_upper[0] + CROWN_MULTI_TOLERANCE,
                        "Softmax CROWN (asymmetric) upper violation: \
                         cu.softmax([{x0},{x1},{x2}])={upper_val} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }

    /// LogSoftmax CROWN backward with asymmetric incoming coefficients.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsoftmax_crown_asymmetric_incoming(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        cl0 in -2.0f32..2.0,
        cl1 in -2.0f32..2.0,
        cl2 in -2.0f32..2.0,
        cu0 in -2.0f32..2.0,
        cu1 in -2.0f32..2.0,
        cu2 in -2.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        prop_assume!(u0 > l0 + 0.001);
        prop_assume!(u1 > l1 + 0.001);
        prop_assume!(u2 > l2 + 0.001);
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01);
        prop_assume!((cl0 - cu0).abs() > 0.01 || (cl1 - cu1).abs() > 0.01 || (cl2 - cu2).abs() > 0.01);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let logsoftmax_layer = LogSoftmaxLayer::new(-1).with_sound_mode(true);
        let layer = Layer::LogSoftmax(logsoftmax_layer);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![cl0, cl1, cl2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![cu0, cu1, cu2]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let lsm_val = logsoftmax(&point);
                    let lower_val = cl0 * lsm_val[0] + cl1 * lsm_val[1] + cl2 * lsm_val[2];
                    let upper_val = cu0 * lsm_val[0] + cu1 * lsm_val[1] + cu2 * lsm_val[2];

                    prop_assert!(
                        lower_val >= crown_lower[0] - CROWN_MULTI_TOLERANCE,
                        "LogSoftmax CROWN (asymmetric) lower violation: \
                         cl.lsm([{x0},{x1},{x2}])={lower_val} < lb={}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        upper_val <= crown_upper[0] + CROWN_MULTI_TOLERANCE,
                        "LogSoftmax CROWN (asymmetric) upper violation: \
                         cu.lsm([{x0},{x1},{x2}])={upper_val} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }

    /// BatchNorm CROWN backward with asymmetric incoming coefficients.
    ///
    /// BatchNorm is an exact affine transform, so asymmetric coefficients should
    /// still produce exact bounds. This verifies the coefficient composition path
    /// handles lower_a != upper_a correctly for affine layers.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_batchnorm_crown_asymmetric_incoming(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        cl0 in -2.0f32..2.0,
        cl1 in -2.0f32..2.0,
        cl2 in -2.0f32..2.0,
        cu0 in -2.0f32..2.0,
        cu1 in -2.0f32..2.0,
        cu2 in -2.0f32..2.0,
        gamma0 in -2.0f32..2.0,
        var0 in 0.1f32..3.0,
    ) {
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01);
        prop_assume!((cl0 - cu0).abs() > 0.01 || (cl1 - cu1).abs() > 0.01 || (cl2 - cu2).abs() > 0.01);
        prop_assume!(gamma0.abs() > 0.01);

        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        let lower = ArrayD::from_shape_vec(IxDyn(&[3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let ny = Array1::from_vec(vec![gamma0, 1.0, -gamma0]);
        let beta = Array1::zeros(3);
        let running_mean = Array1::zeros(3);
        let running_var = Array1::from_vec(vec![var0, 1.0, var0]);

        let bn = BatchNormLayer::new(
            &ny.clone().into_dyn(),
            &beta.clone().into_dyn(),
            &running_mean.clone().into_dyn(),
            &running_var.clone().into_dyn(),
            1e-5,
        ).unwrap();
        let layer = Layer::BatchNorm(bn);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 3), vec![cl0, cl1, cl2]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 3), vec![cu0, cu1, cu2]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

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
                    let lower_val = cl0 * bn_val[0] + cl1 * bn_val[1] + cl2 * bn_val[2];
                    let upper_val = cu0 * bn_val[0] + cu1 * bn_val[1] + cu2 * bn_val[2];

                    prop_assert!(
                        lower_val >= crown_lower[0] - BATCHNORM_CROWN_TOLERANCE,
                        "BatchNorm CROWN (asymmetric) lower violation: \
                         cl.bn([{x0},{x1},{x2}])={lower_val} < lb={}",
                        crown_lower[0]
                    );
                    prop_assert!(
                        upper_val <= crown_upper[0] + BATCHNORM_CROWN_TOLERANCE,
                        "BatchNorm CROWN (asymmetric) upper violation: \
                         cu.bn([{x0},{x1},{x2}])={upper_val} > ub={}",
                        crown_upper[0]
                    );
                }
            }
        }
    }

    /// CausalSoftmax CROWN backward with asymmetric incoming coefficients.
    ///
    /// Sound mode returns constant bounds (IBP-derived), so asymmetric coefficients
    /// should still produce valid bounds. Tests the composition path for the
    /// constant-bounds case with lower_a != upper_a.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_causal_softmax_crown_asymmetric_incoming(
        l00 in -3.0f32..3.0,
        d00 in 0.01f32..2.0,
        l01 in -3.0f32..3.0,
        d01 in 0.01f32..2.0,
        l10 in -3.0f32..3.0,
        d10 in 0.01f32..2.0,
        l11 in -3.0f32..3.0,
        d11 in 0.01f32..2.0,
        // Lower incoming coefficients
        cl0 in -2.0f32..2.0,
        cl1 in -2.0f32..2.0,
        cl2 in -2.0f32..2.0,
        cl3 in -2.0f32..2.0,
        // Upper incoming coefficients (different)
        cu0 in -2.0f32..2.0,
        cu1 in -2.0f32..2.0,
        cu2 in -2.0f32..2.0,
        cu3 in -2.0f32..2.0,
    ) {
        let u00 = (l00 + d00).min(3.0);
        let u01 = (l01 + d01).min(3.0);
        let u10 = (l10 + d10).min(3.0);
        let u11 = (l11 + d11).min(3.0);

        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01 || cl2.abs() > 0.01 || cl3.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01 || cu2.abs() > 0.01 || cu3.abs() > 0.01);
        prop_assume!(
            (cl0 - cu0).abs() > 0.01 || (cl1 - cu1).abs() > 0.01
            || (cl2 - cu2).abs() > 0.01 || (cl3 - cu3).abs() > 0.01
        );

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![l00, l01, l10, l11],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 2]),
            vec![u00, u01, u10, u11],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let cs = CausalSoftmaxLayer::new(-1);
        let layer = Layer::CausalSoftmax(cs);

        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 4), vec![cl0, cl1, cl2, cl3]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 4), vec![cu0, cu1, cu2, cu3]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown(&result, &input);

        let spts = 5;
        let s00 = sample_points(l00, u00, spts);
        let s01 = sample_points(l01, u01, spts);
        let s10 = sample_points(l10, u10, spts);
        let s11 = sample_points(l11, u11, spts);

        for &x00 in &s00 {
            for &x01 in &s01 {
                for &x10 in &s10 {
                    for &x11 in &s11 {
                        let point = Array2::from_shape_vec(
                            (2, 2), vec![x00, x01, x10, x11],
                        ).unwrap();
                        let cs_val = causal_softmax(&point);
                        let flat: Vec<f32> = cs_val.iter().copied().collect();
                        let lower_val = cl0 * flat[0] + cl1 * flat[1]
                            + cl2 * flat[2] + cl3 * flat[3];
                        let upper_val = cu0 * flat[0] + cu1 * flat[1]
                            + cu2 * flat[2] + cu3 * flat[3];

                        prop_assert!(
                            lower_val >= crown_lower[0] - CROWN_MULTI_TOLERANCE,
                            "CausalSoftmax CROWN (asymmetric) lower violation: \
                             cl.cs([{x00},{x01};{x10},{x11}])={lower_val} < lb={}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            upper_val <= crown_upper[0] + CROWN_MULTI_TOLERANCE,
                            "CausalSoftmax CROWN (asymmetric) upper violation: \
                             cu.cs([{x00},{x01};{x10},{x11}])={upper_val} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }
}
