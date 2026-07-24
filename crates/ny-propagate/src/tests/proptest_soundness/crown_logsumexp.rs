// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness tests for LogSumExp IBP and CROWN backward.
//!
//! LogSumExp: y = log(sum(exp(x))) over specified axes.
//! - IBP: monotonically increasing, so IBP applies independently to lower/upper.
//! - CROWN backward: uses IBP-derived constant bounds (A=0, b=concretized IBP).
//!   This is the same strategy as CausalSoftmax — sound but loose.
//!
//! Part of #40.

use crate::layers::common::BoundPropagation;
use crate::layers::softmax::LogSumExpLayer;
use crate::layers::Layer;
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::sample_points;

/// Tolerance for LogSumExp IBP soundness.
/// LogSumExp involves exp/log which amplify FP error, but monotonicity means
/// IBP bounds should be tight — only FP rounding separates them from truth.
const LOGSUMEXP_IBP_TOLERANCE: f32 = 1e-5;

/// Tolerance for LogSumExp CROWN backward soundness.
/// CROWN path concretizes IBP output through incoming linear bounds,
/// adding one more level of FP accumulation.
const LOGSUMEXP_CROWN_TOLERANCE: f32 = 1e-4;

/// Reference LogSumExp computation for soundness checking.
/// Uses the log-sum-exp trick for numerical stability.
fn logsumexp_eval(x: &Array1<f32>) -> f32 {
    let max_x = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp: f32 = x.mapv(|xi| (xi - max_x).exp()).sum();
    max_x + sum_exp.ln()
}

// =============================================================================
// LOGSUMEXP IBP SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// LogSumExp IBP soundness: for all x in [lower, upper], logsumexp(x) must
    /// be within the computed IBP bounds.
    ///
    /// LogSumExp is monotonically increasing in each argument (∂LSE/∂x_i = softmax_i > 0),
    /// so IBP lower = LSE(lower), IBP upper = LSE(upper). This test verifies
    /// that sampled interior points also satisfy the bounds.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsumexp_ibp_3d(
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

        // 2D input [1, 3], reduce over last axis with keepdims -> output [1, 1]
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = LogSumExpLayer::new(vec![-1], true);
        let output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        let ibp_lower = output.lower()[[0, 0]];
        let ibp_upper = output.upper()[[0, 0]];

        // Verify bounds are ordered
        prop_assert!(
            ibp_lower <= ibp_upper + LOGSUMEXP_IBP_TOLERANCE,
            "IBP lower > upper: {} > {}",
            ibp_lower, ibp_upper
        );

        // Sample and verify soundness
        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let lse_val = logsumexp_eval(&point);

                    prop_assert!(
                        lse_val >= ibp_lower - LOGSUMEXP_IBP_TOLERANCE,
                        "LogSumExp IBP lower violation: \
                         lse([{x0},{x1},{x2}])={lse_val} < lb={ibp_lower}"
                    );
                    prop_assert!(
                        lse_val <= ibp_upper + LOGSUMEXP_IBP_TOLERANCE,
                        "LogSumExp IBP upper violation: \
                         lse([{x0},{x1},{x2}])={lse_val} > ub={ibp_upper}"
                    );
                }
            }
        }
    }

    /// LogSumExp IBP soundness with keepdims=true and 2D input [2, 3].
    ///
    /// Reduces over the last axis, producing [2, 1] output.
    /// Verifies per-row soundness with interior sampling.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsumexp_ibp_2d_keepdims(
        l00 in -4.0f32..4.0,
        d00 in 0.01f32..2.0,
        l01 in -4.0f32..4.0,
        d01 in 0.01f32..2.0,
        l02 in -4.0f32..4.0,
        d02 in 0.01f32..2.0,
        l10 in -4.0f32..4.0,
        d10 in 0.01f32..2.0,
        l11 in -4.0f32..4.0,
        d11 in 0.01f32..2.0,
        l12 in -4.0f32..4.0,
        d12 in 0.01f32..2.0,
    ) {
        let u00 = (l00 + d00).min(4.0);
        let u01 = (l01 + d01).min(4.0);
        let u02 = (l02 + d02).min(4.0);
        let u10 = (l10 + d10).min(4.0);
        let u11 = (l11 + d11).min(4.0);
        let u12 = (l12 + d12).min(4.0);

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l00, l01, l02, l10, l11, l12],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u00, u01, u02, u10, u11, u12],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = LogSumExpLayer::new(vec![-1], true);
        let output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        prop_assert_eq!(output.shape(), &[2, 1]);

        // Verify row 0 soundness
        let spts = 4;
        for &x0 in &sample_points(l00, u00, spts) {
            for &x1 in &sample_points(l01, u01, spts) {
                for &x2 in &sample_points(l02, u02, spts) {
                    let lse = logsumexp_eval(&arr1(&[x0, x1, x2]));
                    prop_assert!(
                        lse >= output.lower()[[0, 0]] - LOGSUMEXP_IBP_TOLERANCE,
                        "Row 0 lower violation: lse({x0},{x1},{x2})={lse} < {}",
                        output.lower()[[0, 0]]
                    );
                    prop_assert!(
                        lse <= output.upper()[[0, 0]] + LOGSUMEXP_IBP_TOLERANCE,
                        "Row 0 upper violation: lse({x0},{x1},{x2})={lse} > {}",
                        output.upper()[[0, 0]]
                    );
                }
            }
        }

        // Verify row 1 soundness
        for &x0 in &sample_points(l10, u10, spts) {
            for &x1 in &sample_points(l11, u11, spts) {
                for &x2 in &sample_points(l12, u12, spts) {
                    let lse = logsumexp_eval(&arr1(&[x0, x1, x2]));
                    prop_assert!(
                        lse >= output.lower()[[1, 0]] - LOGSUMEXP_IBP_TOLERANCE,
                        "Row 1 lower violation: lse({x0},{x1},{x2})={lse} < {}",
                        output.lower()[[1, 0]]
                    );
                    prop_assert!(
                        lse <= output.upper()[[1, 0]] + LOGSUMEXP_IBP_TOLERANCE,
                        "Row 1 upper violation: lse({x0},{x1},{x2})={lse} > {}",
                        output.upper()[[1, 0]]
                    );
                }
            }
        }
    }
}

// =============================================================================
// LOGSUMEXP CROWN BACKWARD SOUNDNESS (IBP-DERIVED CONSTANT BOUNDS)
// =============================================================================
//
// LogSumExp CROWN backward returns constant bounds (A=0, b=concretized IBP output).
// This is the same strategy as CausalSoftmax — sound but loose, because there is
// no closed-form linear relaxation for LogSumExp.
//
// The soundness property: CROWN bounds concretized against input interval must
// contain logsumexp(x) for all x in the input interval.

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(300) })]

    /// LogSumExp CROWN backward soundness with identity incoming bounds (1D, 3 elements).
    ///
    /// Since CROWN returns constant bounds for LogSumExp, the concretized CROWN
    /// bounds should exactly match IBP bounds (modulo FP rounding).
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsumexp_crown_identity_1d(
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

        // Use 2D input [1, 3] with keepdims=true so output is [1, 1] -> flattened to 1
        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let logsumexp = LogSumExpLayer::new(vec![-1], true);
        let layer = Layer::LogSumExp(logsumexp.clone());

        // Identity incoming bounds: 1 output
        let identity = LinearBounds::identity(1);

        let result = layer
            .propagate_crown_backward(&identity, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        // Concretize against input interval
        let concrete = result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        // Verify CROWN-IBP equivalence: since CROWN uses IBP-derived constants,
        // concretized bounds should match IBP output
        let ibp_output = logsumexp.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;
        let ibp_lower = ibp_output.lower()[[0, 0]];
        let ibp_upper = ibp_output.upper()[[0, 0]];

        prop_assert!(
            (crown_lower - ibp_lower).abs() <= LOGSUMEXP_IBP_TOLERANCE,
            "CROWN-IBP lower mismatch: crown={crown_lower}, ibp={ibp_lower}"
        );
        prop_assert!(
            (crown_upper - ibp_upper).abs() <= LOGSUMEXP_IBP_TOLERANCE,
            "CROWN-IBP upper mismatch: crown={crown_upper}, ibp={ibp_upper}"
        );

        // Sample and verify soundness
        let samples_per_dim = 5;
        let s0_pts = sample_points(l0, u0, samples_per_dim);
        let s1_pts = sample_points(l1, u1, samples_per_dim);
        let s2_pts = sample_points(l2, u2, samples_per_dim);

        for &x0 in &s0_pts {
            for &x1 in &s1_pts {
                for &x2 in &s2_pts {
                    let point = arr1(&[x0, x1, x2]);
                    let lse_val = logsumexp_eval(&point);

                    prop_assert!(
                        lse_val >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
                        "LogSumExp CROWN lower violation: \
                         lse([{x0},{x1},{x2}])={lse_val} < lb={crown_lower}"
                    );
                    prop_assert!(
                        lse_val <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
                        "LogSumExp CROWN upper violation: \
                         lse([{x0},{x1},{x2}])={lse_val} > ub={crown_upper}"
                    );
                }
            }
        }
    }

    /// LogSumExp CROWN backward with non-identity incoming coefficients.
    ///
    /// Tests CROWN composition: incoming = c0*lse(row0) + c1*lse(row1), where
    /// the input is [2, 3] and reduction is over axis -1. This exercises the
    /// constant-bounds concretization path with non-trivial coefficients.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsumexp_crown_nonidentity_incoming(
        l00 in -3.0f32..3.0,
        d00 in 0.01f32..2.0,
        l01 in -3.0f32..3.0,
        d01 in 0.01f32..2.0,
        l02 in -3.0f32..3.0,
        d02 in 0.01f32..2.0,
        l10 in -3.0f32..3.0,
        d10 in 0.01f32..2.0,
        l11 in -3.0f32..3.0,
        d11 in 0.01f32..2.0,
        l12 in -3.0f32..3.0,
        d12 in 0.01f32..2.0,
        // Incoming coefficients: 1 output combining 2 LogSumExp outputs
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
    ) {
        let u00 = (l00 + d00).min(3.0);
        let u01 = (l01 + d01).min(3.0);
        let u02 = (l02 + d02).min(3.0);
        let u10 = (l10 + d10).min(3.0);
        let u11 = (l11 + d11).min(3.0);
        let u12 = (l12 + d12).min(3.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l00, l01, l02, l10, l11, l12],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u00, u01, u02, u10, u11, u12],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let logsumexp = LogSumExpLayer::new(vec![-1], true);
        let layer = Layer::LogSumExp(logsumexp);

        // Non-identity incoming: 1 output = c0*lse_row0 + c1*lse_row1
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let concrete = result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        prop_assert!(
            crown_lower <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
            "LogSumExp CROWN (non-identity) produced inverted bounds: \
             lb={crown_lower} > ub={crown_upper}"
        );

        // Analytical extremum check for c0*lse(row0) + c1*lse(row1).
        // Each term c*lse(row) is monotone over the box (increasing if c >= 0,
        // decreasing if c < 0), so global extrema are attained at row endpoints.
        let row0_min = logsumexp_eval(&arr1(&[l00, l01, l02]));
        let row0_max = logsumexp_eval(&arr1(&[u00, u01, u02]));
        let row1_min = logsumexp_eval(&arr1(&[l10, l11, l12]));
        let row1_max = logsumexp_eval(&arr1(&[u10, u11, u12]));

        let term0_min = if c0 >= 0.0 { c0 * row0_min } else { c0 * row0_max };
        let term0_max = if c0 >= 0.0 { c0 * row0_max } else { c0 * row0_min };
        let term1_min = if c1 >= 0.0 { c1 * row1_min } else { c1 * row1_max };
        let term1_max = if c1 >= 0.0 { c1 * row1_max } else { c1 * row1_min };
        let analytic_min = term0_min + term1_min;
        let analytic_max = term0_max + term1_max;

        prop_assert!(
            analytic_min >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
            "LogSumExp CROWN (non-identity) analytical lower violation: \
             min(c.lse(row0,row1))={analytic_min} < lb={crown_lower}"
        );
        prop_assert!(
            analytic_max <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
            "LogSumExp CROWN (non-identity) analytical upper violation: \
             max(c.lse(row0,row1))={analytic_max} > ub={crown_upper}"
        );

        // Sample and verify: c0*lse(x_row0) + c1*lse(x_row1) must be in bounds
        let spts = 4;
        let s00 = sample_points(l00, u00, spts);
        let s01 = sample_points(l01, u01, spts);
        let s02 = sample_points(l02, u02, spts);
        let s10 = sample_points(l10, u10, spts);
        let s11 = sample_points(l11, u11, spts);
        let s12 = sample_points(l12, u12, spts);

        // Sample row 0 and row 1 independently (per-dimension to avoid 6D explosion)
        for &x00 in &s00 {
            for &x01 in &s01 {
                for &x02 in &s02 {
                    let lse_row0 = logsumexp_eval(&arr1(&[x00, x01, x02]));
                    // Use midpoint for row 1
                    let mid10 = f32::midpoint(l10, u10);
                    let mid11 = f32::midpoint(l11, u11);
                    let mid12 = f32::midpoint(l12, u12);
                    let lse_row1 = logsumexp_eval(&arr1(&[mid10, mid11, mid12]));
                    let combined = c0 * lse_row0 + c1 * lse_row1;

                    prop_assert!(
                        combined >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
                        "LogSumExp CROWN (non-identity) lower violation: \
                         c.lse([{x00},{x01},{x02}; mid])={combined} < lb={crown_lower}"
                    );
                    prop_assert!(
                        combined <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
                        "LogSumExp CROWN (non-identity) upper violation: \
                         c.lse([{x00},{x01},{x02}; mid])={combined} > ub={crown_upper}"
                    );
                }
            }
        }

        // Also sample row 1 with midpoint for row 0
        for &x10 in &s10 {
            for &x11 in &s11 {
                for &x12 in &s12 {
                    let mid00 = f32::midpoint(l00, u00);
                    let mid01 = f32::midpoint(l01, u01);
                    let mid02 = f32::midpoint(l02, u02);
                    let lse_row0 = logsumexp_eval(&arr1(&[mid00, mid01, mid02]));
                    let lse_row1 = logsumexp_eval(&arr1(&[x10, x11, x12]));
                    let combined = c0 * lse_row0 + c1 * lse_row1;

                    prop_assert!(
                        combined >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
                        "LogSumExp CROWN (non-identity) lower violation: \
                         c.lse([mid; {x10},{x11},{x12}])={combined} < lb={crown_lower}"
                    );
                    prop_assert!(
                        combined <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
                        "LogSumExp CROWN (non-identity) upper violation: \
                         c.lse([mid; {x10},{x11},{x12}])={combined} > ub={crown_upper}"
                    );
                }
            }
        }

        // Full joint endpoint sampling: all 2^6 = 64 corners of both rows.
        // Each term c*lse(row) is monotone over the interval box, with
        // direction determined by sign(c). Therefore, extrema of
        // c0*lse(row0) + c1*lse(row1) occur at joint endpoint corners.
        // The previous sampling
        // (row0 varied with row1 at midpoint and vice versa) missed the
        // joint extreme corners where both rows contribute worst-case values.
        for &x00 in &[l00, u00] {
            for &x01 in &[l01, u01] {
                for &x02 in &[l02, u02] {
                    for &x10 in &[l10, u10] {
                        for &x11 in &[l11, u11] {
                            for &x12 in &[l12, u12] {
                                let lse_row0 = logsumexp_eval(&arr1(&[x00, x01, x02]));
                                let lse_row1 = logsumexp_eval(&arr1(&[x10, x11, x12]));
                                let combined = c0 * lse_row0 + c1 * lse_row1;

                                prop_assert!(
                                    combined >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
                                    "LogSumExp CROWN corner lower violation: \
                                     c.lse([{x00},{x01},{x02}; {x10},{x11},{x12}])={combined} < lb={crown_lower}"
                                );
                                prop_assert!(
                                    combined <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
                                    "LogSumExp CROWN corner upper violation: \
                                     c.lse([{x00},{x01},{x02}; {x10},{x11},{x12}])={combined} > ub={crown_upper}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// LogSumExp CROWN backward with asymmetric incoming coefficients.
    ///
    /// lower_a != upper_a — different linear combinations bound from below and above.
    /// Since LogSumExp CROWN returns constant bounds, asymmetric coefficients exercise
    /// the concretization path where lower_b and upper_b are split against different
    /// coefficient combinations.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsumexp_crown_asymmetric_incoming(
        l0 in -3.0f32..3.0,
        d0 in 0.01f32..2.0,
        l1 in -3.0f32..3.0,
        d1 in 0.01f32..2.0,
        l2 in -3.0f32..3.0,
        d2 in 0.01f32..2.0,
        // Lower incoming coefficients
        cl0 in -2.0f32..2.0,
        cl1 in -2.0f32..2.0,
        // Upper incoming coefficients (different from lower)
        cu0 in -2.0f32..2.0,
        cu1 in -2.0f32..2.0,
    ) {
        let u0 = (l0 + d0).min(3.0);
        let u1 = (l1 + d1).min(3.0);
        let u2 = (l2 + d2).min(3.0);

        prop_assume!(u0 > l0 + 0.001);
        prop_assume!(u1 > l1 + 0.001);
        prop_assume!(u2 > l2 + 0.001);
        prop_assume!(cl0.abs() > 0.01 || cl1.abs() > 0.01);
        prop_assume!(cu0.abs() > 0.01 || cu1.abs() > 0.01);
        prop_assume!((cl0 - cu0).abs() > 0.01 || (cl1 - cu1).abs() > 0.01);

        // Input [2, 3], reduce over axis -1 with keepdims -> output [2, 1] -> flattened to 2
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l1, l2, l0],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u1, u2, u0],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let logsumexp = LogSumExpLayer::new(vec![-1], true);
        let layer = Layer::LogSumExp(logsumexp);

        // Asymmetric incoming: lower_a != upper_a
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![cl0, cl1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![cu0, cu1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let concrete = result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        // For asymmetric incoming, verify:
        //   crown_lower <= cl . lse(x) for all x
        //   cu . lse(x) <= crown_upper for all x
        let spts = 5;
        let s0 = sample_points(l0, u0, spts);
        let s1 = sample_points(l1, u1, spts);
        let s2 = sample_points(l2, u2, spts);

        // Sample row 0 (row 1 uses permuted lower/upper from same variables)
        for &x0 in &s0 {
            for &x1 in &s1 {
                for &x2 in &s2 {
                    let lse_row0 = logsumexp_eval(&arr1(&[x0, x1, x2]));
                    let lse_row1 = logsumexp_eval(&arr1(&[x1, x2, x0]));
                    let lower_val = cl0 * lse_row0 + cl1 * lse_row1;
                    let upper_val = cu0 * lse_row0 + cu1 * lse_row1;

                    prop_assert!(
                        lower_val >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
                        "LogSumExp CROWN (asymmetric) lower violation: \
                         cl.lse([{x0},{x1},{x2}])={lower_val} < lb={crown_lower}"
                    );
                    prop_assert!(
                        upper_val <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
                        "LogSumExp CROWN (asymmetric) upper violation: \
                         cu.lse([{x0},{x1},{x2}])={upper_val} > ub={crown_upper}"
                    );
                }
            }
        }
    }

    /// LogSumExp CROWN backward stress test on high-dimensional inputs [4, 8]
    /// with wider numeric ranges.
    ///
    /// This verifies asymmetric incoming coefficients under larger reduction
    /// dimensions and broader intervals than the baseline 3-element tests.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsumexp_crown_asymmetric_incoming_4x8_wide_ranges(
        lower_vals in prop::collection::vec(-10.0f32..10.0, 32),
        widths in prop::collection::vec(0.01f32..6.0, 32),
        lower_coeffs in prop::collection::vec(-3.0f32..3.0, 4),
        upper_coeffs in prop::collection::vec(-3.0f32..3.0, 4),
        lower_bias in -1.0f32..1.0,
        upper_bias in -1.0f32..1.0,
    ) {
        let upper_vals: Vec<f32> = lower_vals
            .iter()
            .zip(widths.iter())
            .map(|(l, w)| (l + w).min(12.0))
            .collect();

        let has_lower_signal = lower_coeffs.iter().any(|c| c.abs() > 0.01);
        let has_upper_signal = upper_coeffs.iter().any(|c| c.abs() > 0.01);
        let coeffs_differ = lower_coeffs
            .iter()
            .zip(upper_coeffs.iter())
            .any(|(cl, cu)| (cl - cu).abs() > 0.01);
        prop_assume!(has_lower_signal);
        prop_assume!(has_upper_signal);
        prop_assume!(coeffs_differ);

        let lower = ArrayD::from_shape_vec(IxDyn(&[4, 8]), lower_vals.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4, 8]), upper_vals.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = Layer::LogSumExp(LogSumExpLayer::new(vec![-1], true));
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 4), lower_coeffs.clone()).unwrap(),
            Array1::from_vec(vec![lower_bias]),
            Array2::from_shape_vec((1, 4), upper_coeffs.clone()).unwrap(),
            Array1::from_vec(vec![upper_bias]),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let concrete = result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        let mut row_min = [0.0f32; 4];
        let mut row_max = [0.0f32; 4];
        for row in 0..4 {
            let start = row * 8;
            let end = start + 8;
            row_min[row] = logsumexp_eval(&Array1::from_vec(lower_vals[start..end].to_vec()));
            row_max[row] = logsumexp_eval(&Array1::from_vec(upper_vals[start..end].to_vec()));
        }

        let analytic_lower_min: f32 = lower_bias
            + (0..4)
                .map(|row| {
                    let c = lower_coeffs[row];
                    let lse = if c >= 0.0 { row_min[row] } else { row_max[row] };
                    c * lse
                })
                .sum::<f32>();
        let analytic_upper_max: f32 = upper_bias
            + (0..4)
                .map(|row| {
                    let c = upper_coeffs[row];
                    let lse = if c >= 0.0 { row_max[row] } else { row_min[row] };
                    c * lse
                })
                .sum::<f32>();

        prop_assert!(
            analytic_lower_min >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
            "LogSumExp [4,8] analytical lower violation: \
             min(cl.lse)={analytic_lower_min} < lb={crown_lower}"
        );
        prop_assert!(
            analytic_upper_max <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
            "LogSumExp [4,8] analytical upper violation: \
             max(cu.lse)={analytic_upper_max} > ub={crown_upper}"
        );

        // Cross-check corners over row endpoints (2^4 combinations).
        for mask in 0usize..(1usize << 4) {
            let mut lse_rows = [0.0f32; 4];
            for row in 0..4 {
                lse_rows[row] = if (mask & (1 << row)) == 0 {
                    row_min[row]
                } else {
                    row_max[row]
                };
            }
            let lower_val = lower_bias
                + lower_coeffs
                    .iter()
                    .zip(lse_rows.iter())
                    .map(|(c, lse)| c * lse)
                    .sum::<f32>();
            let upper_val = upper_bias
                + upper_coeffs
                    .iter()
                    .zip(lse_rows.iter())
                    .map(|(c, lse)| c * lse)
                    .sum::<f32>();

            prop_assert!(
                lower_val >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
                "LogSumExp [4,8] corner lower violation (mask={mask}): \
                 cl.lse={lower_val} < lb={crown_lower}"
            );
            prop_assert!(
                upper_val <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
                "LogSumExp [4,8] corner upper violation (mask={mask}): \
                 cu.lse={upper_val} > ub={crown_upper}"
            );
        }
    }
}

// =============================================================================
// LOGSUMEXP GAP COVERAGE: keepdims=false, axis=0, 3D input, wide IBP ranges
// =============================================================================
//
// Gaps identified by P271 proof coverage audit:
// 1. Only axis=-1 tested — no axis=0 or multi-axis coverage
// 2. keepdims=false path never tested
// 3. No true 1D input (tests use [1,3] not [3])
// 4. No 3D+ inputs (transformers use [batch, seq, hidden])
// 5. No IBP test at wide ranges ([-10,10])

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// LogSumExp IBP soundness with keepdims=false: output rank decreases.
    ///
    /// Input [2, 3], axis=-1, keepdims=false → output [2] (not [2, 1]).
    /// Verifies shape correctness and soundness for the no-keepdims path.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsumexp_ibp_no_keepdims(
        l0 in -5.0f32..5.0,
        d0 in 0.01f32..3.0,
        l1 in -5.0f32..5.0,
        d1 in 0.01f32..3.0,
        l2 in -5.0f32..5.0,
        d2 in 0.01f32..3.0,
        l3 in -5.0f32..5.0,
        d3 in 0.01f32..3.0,
        l4 in -5.0f32..5.0,
        d4 in 0.01f32..3.0,
        l5 in -5.0f32..5.0,
        d5 in 0.01f32..3.0,
    ) {
        let u0 = (l0 + d0).min(5.0);
        let u1 = (l1 + d1).min(5.0);
        let u2 = (l2 + d2).min(5.0);
        let u3 = (l3 + d3).min(5.0);
        let u4 = (l4 + d4).min(5.0);
        let u5 = (l5 + d5).min(5.0);

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = LogSumExpLayer::new(vec![-1], false);
        let output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        // keepdims=false: output shape should be [2], not [2, 1]
        prop_assert_eq!(output.shape(), &[2]);

        // Row 0 soundness
        let spts = 5;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                for &x2 in &sample_points(l2, u2, spts) {
                    let lse = logsumexp_eval(&arr1(&[x0, x1, x2]));
                    prop_assert!(
                        lse >= output.lower()[[0]] - LOGSUMEXP_IBP_TOLERANCE,
                        "Row 0 lower violation (no keepdims): lse({x0},{x1},{x2})={lse} < {}",
                        output.lower()[[0]]
                    );
                    prop_assert!(
                        lse <= output.upper()[[0]] + LOGSUMEXP_IBP_TOLERANCE,
                        "Row 0 upper violation (no keepdims): lse({x0},{x1},{x2})={lse} > {}",
                        output.upper()[[0]]
                    );
                }
            }
        }

        // Row 1 soundness
        for &x3 in &sample_points(l3, u3, spts) {
            for &x4 in &sample_points(l4, u4, spts) {
                for &x5 in &sample_points(l5, u5, spts) {
                    let lse = logsumexp_eval(&arr1(&[x3, x4, x5]));
                    prop_assert!(
                        lse >= output.lower()[[1]] - LOGSUMEXP_IBP_TOLERANCE,
                        "Row 1 lower violation (no keepdims): lse({x3},{x4},{x5})={lse} < {}",
                        output.lower()[[1]]
                    );
                    prop_assert!(
                        lse <= output.upper()[[1]] + LOGSUMEXP_IBP_TOLERANCE,
                        "Row 1 upper violation (no keepdims): lse({x3},{x4},{x5})={lse} > {}",
                        output.upper()[[1]]
                    );
                }
            }
        }
    }

    /// LogSumExp IBP soundness with axis=0 reduction.
    ///
    /// Input [3, 2], axis=0, keepdims=true → output [1, 2].
    /// LogSumExp reduces over the row axis (dim 0), applying logsumexp across
    /// rows for each column. This exercises the non-last-axis reduction path.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsumexp_ibp_axis0(
        l00 in -5.0f32..5.0,
        d00 in 0.01f32..3.0,
        l01 in -5.0f32..5.0,
        d01 in 0.01f32..3.0,
        l10 in -5.0f32..5.0,
        d10 in 0.01f32..3.0,
        l11 in -5.0f32..5.0,
        d11 in 0.01f32..3.0,
        l20 in -5.0f32..5.0,
        d20 in 0.01f32..3.0,
        l21 in -5.0f32..5.0,
        d21 in 0.01f32..3.0,
    ) {
        let u00 = (l00 + d00).min(5.0);
        let u01 = (l01 + d01).min(5.0);
        let u10 = (l10 + d10).min(5.0);
        let u11 = (l11 + d11).min(5.0);
        let u20 = (l20 + d20).min(5.0);
        let u21 = (l21 + d21).min(5.0);

        // Input [3, 2] — 3 rows, 2 columns
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[3, 2]),
            vec![l00, l01, l10, l11, l20, l21],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[3, 2]),
            vec![u00, u01, u10, u11, u20, u21],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = LogSumExpLayer::new(vec![0], true);
        let output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        prop_assert_eq!(output.shape(), &[1, 2]);

        // Column 0: lse over [x00, x10, x20]
        let spts = 5;
        for &x00 in &sample_points(l00, u00, spts) {
            for &x10 in &sample_points(l10, u10, spts) {
                for &x20 in &sample_points(l20, u20, spts) {
                    let lse = logsumexp_eval(&arr1(&[x00, x10, x20]));
                    prop_assert!(
                        lse >= output.lower()[[0, 0]] - LOGSUMEXP_IBP_TOLERANCE,
                        "Col 0 lower violation (axis=0): lse({x00},{x10},{x20})={lse} < {}",
                        output.lower()[[0, 0]]
                    );
                    prop_assert!(
                        lse <= output.upper()[[0, 0]] + LOGSUMEXP_IBP_TOLERANCE,
                        "Col 0 upper violation (axis=0): lse({x00},{x10},{x20})={lse} > {}",
                        output.upper()[[0, 0]]
                    );
                }
            }
        }

        // Column 1: lse over [x01, x11, x21]
        for &x01 in &sample_points(l01, u01, spts) {
            for &x11 in &sample_points(l11, u11, spts) {
                for &x21 in &sample_points(l21, u21, spts) {
                    let lse = logsumexp_eval(&arr1(&[x01, x11, x21]));
                    prop_assert!(
                        lse >= output.lower()[[0, 1]] - LOGSUMEXP_IBP_TOLERANCE,
                        "Col 1 lower violation (axis=0): lse({x01},{x11},{x21})={lse} < {}",
                        output.lower()[[0, 1]]
                    );
                    prop_assert!(
                        lse <= output.upper()[[0, 1]] + LOGSUMEXP_IBP_TOLERANCE,
                        "Col 1 upper violation (axis=0): lse({x01},{x11},{x21})={lse} > {}",
                        output.upper()[[0, 1]]
                    );
                }
            }
        }
    }

    /// LogSumExp IBP soundness at wide ranges [-10, 10].
    ///
    /// Tests numerical stability of the log-sum-exp trick at wider intervals
    /// where exp(10) ≈ 22026 and the dynamic range within a single interval
    /// can span many orders of magnitude.
    /// Part of #40.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_logsumexp_ibp_wide_range(
        l0 in -10.0f32..10.0,
        d0 in 0.01f32..8.0,
        l1 in -10.0f32..10.0,
        d1 in 0.01f32..8.0,
        l2 in -10.0f32..10.0,
        d2 in 0.01f32..8.0,
    ) {
        let u0 = (l0 + d0).min(10.0);
        let u1 = (l1 + d1).min(10.0);
        let u2 = (l2 + d2).min(10.0);

        prop_assume!(u0 > l0 + 0.001);
        prop_assume!(u1 > l1 + 0.001);
        prop_assume!(u2 > l2 + 0.001);

        let lower = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![l0, l1, l2]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![u0, u1, u2]).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = LogSumExpLayer::new(vec![-1], true);
        let output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        let ibp_lower = output.lower()[[0, 0]];
        let ibp_upper = output.upper()[[0, 0]];

        prop_assert!(
            ibp_lower <= ibp_upper + LOGSUMEXP_IBP_TOLERANCE,
            "IBP bounds inverted at wide range: {} > {}",
            ibp_lower, ibp_upper
        );

        // Wider tolerance at extreme ranges due to exp amplification
        let wide_tol = 1e-4;

        let spts = 5;
        for &x0 in &sample_points(l0, u0, spts) {
            for &x1 in &sample_points(l1, u1, spts) {
                for &x2 in &sample_points(l2, u2, spts) {
                    let lse = logsumexp_eval(&arr1(&[x0, x1, x2]));
                    prop_assert!(
                        lse >= ibp_lower - wide_tol,
                        "Wide-range IBP lower violation: \
                         lse([{x0},{x1},{x2}])={lse} < lb={ibp_lower}"
                    );
                    prop_assert!(
                        lse <= ibp_upper + wide_tol,
                        "Wide-range IBP upper violation: \
                         lse([{x0},{x1},{x2}])={lse} > ub={ibp_upper}"
                    );
                }
            }
        }
    }

    /// LogSumExp IBP soundness on 3D input [2, 3, 4] reducing over last axis.
    ///
    /// Simulates a transformer-style input (batch=2, seq=3, hidden=4) with
    /// LogSumExp over the hidden dimension. Verifies both keepdims=true output
    /// shape [2, 3, 1] and soundness of each row's bounds.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsumexp_ibp_3d_input(
        lower_vals in prop::collection::vec(-5.0f32..5.0, 24),
        widths in prop::collection::vec(0.01f32..3.0, 24),
    ) {
        let upper_vals: Vec<f32> = lower_vals.iter().zip(widths.iter())
            .map(|(&l, &w)| (l + w).min(6.0))
            .collect();

        let lower = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), lower_vals.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), upper_vals.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = LogSumExpLayer::new(vec![-1], true);
        let output = layer.propagate_ibp(&input)
            .map_err(|e| TestCaseError::fail(
                format!("propagate_ibp failed: {e}")
            ))?;

        prop_assert_eq!(output.shape(), &[2, 3, 1]);

        // Verify each of the 6 rows (batch * seq = 2 * 3)
        for batch in 0..2 {
            for seq in 0..3 {
                let offset = (batch * 3 + seq) * 4;
                let row_lower: Vec<f32> = (0..4).map(|i| lower_vals[offset + i]).collect();
                let row_upper: Vec<f32> = (0..4).map(|i| upper_vals[offset + i]).collect();

                // Sample 3 interior points per dim to keep total under 81
                let spts = 3;
                let s: Vec<Vec<f32>> = (0..4)
                    .map(|i| sample_points(row_lower[i], row_upper[i], spts))
                    .collect();

                // Use a subset of joint samples to avoid 3^4=81 explosion per row
                for &x0 in &s[0] {
                    for &x1 in &s[1] {
                        // Use midpoints for x2, x3 to keep iteration count manageable
                        let x2 = f32::midpoint(row_lower[2], row_upper[2]);
                        let x3 = f32::midpoint(row_lower[3], row_upper[3]);
                        let lse = logsumexp_eval(&arr1(&[x0, x1, x2, x3]));
                        prop_assert!(
                            lse >= output.lower()[[batch, seq, 0]] - LOGSUMEXP_IBP_TOLERANCE,
                            "3D [{batch},{seq}] lower violation: lse={lse} < {}",
                            output.lower()[[batch, seq, 0]]
                        );
                        prop_assert!(
                            lse <= output.upper()[[batch, seq, 0]] + LOGSUMEXP_IBP_TOLERANCE,
                            "3D [{batch},{seq}] upper violation: lse={lse} > {}",
                            output.upper()[[batch, seq, 0]]
                        );
                    }
                }

                // Also check endpoints of all dims (2^4 = 16 corners)
                for mask in 0usize..(1usize << 4) {
                    let point: Vec<f32> = (0..4).map(|i| {
                        if (mask & (1 << i)) == 0 { row_lower[i] } else { row_upper[i] }
                    }).collect();
                    let lse = logsumexp_eval(&Array1::from_vec(point));
                    prop_assert!(
                        lse >= output.lower()[[batch, seq, 0]] - LOGSUMEXP_IBP_TOLERANCE,
                        "3D [{batch},{seq}] corner (mask={mask}) lower violation"
                    );
                    prop_assert!(
                        lse <= output.upper()[[batch, seq, 0]] + LOGSUMEXP_IBP_TOLERANCE,
                        "3D [{batch},{seq}] corner (mask={mask}) upper violation"
                    );
                }
            }
        }
    }

    /// LogSumExp CROWN backward with keepdims=false.
    ///
    /// Input [2, 3], axis=-1, keepdims=false → IBP output [2].
    /// Tests that CROWN correctly composes with the no-keepdims flattened output.
    /// Part of #40.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_logsumexp_crown_no_keepdims(
        l00 in -3.0f32..3.0,
        d00 in 0.01f32..2.0,
        l01 in -3.0f32..3.0,
        d01 in 0.01f32..2.0,
        l02 in -3.0f32..3.0,
        d02 in 0.01f32..2.0,
        l10 in -3.0f32..3.0,
        d10 in 0.01f32..2.0,
        l11 in -3.0f32..3.0,
        d11 in 0.01f32..2.0,
        l12 in -3.0f32..3.0,
        d12 in 0.01f32..2.0,
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
    ) {
        let u00 = (l00 + d00).min(3.0);
        let u01 = (l01 + d01).min(3.0);
        let u02 = (l02 + d02).min(3.0);
        let u10 = (l10 + d10).min(3.0);
        let u11 = (l11 + d11).min(3.0);
        let u12 = (l12 + d12).min(3.0);

        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01);

        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![l00, l01, l02, l10, l11, l12],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![u00, u01, u02, u10, u11, u12],
        ).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let logsumexp = LogSumExpLayer::new(vec![-1], false);
        let layer = Layer::LogSumExp(logsumexp);

        // Non-identity incoming: 1 output = c0*lse(row0) + c1*lse(row1)
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 2), vec![c0, c1]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_crown_backward(&incoming, Some(&input))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let concrete = result.concretize(&input);
        let crown_lower = concrete.lower()[[0]];
        let crown_upper = concrete.upper()[[0]];

        prop_assert!(
            crown_lower <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
            "LogSumExp CROWN (no keepdims) inverted: lb={crown_lower} > ub={crown_upper}"
        );

        // Analytical extremum check
        let row0_min = logsumexp_eval(&arr1(&[l00, l01, l02]));
        let row0_max = logsumexp_eval(&arr1(&[u00, u01, u02]));
        let row1_min = logsumexp_eval(&arr1(&[l10, l11, l12]));
        let row1_max = logsumexp_eval(&arr1(&[u10, u11, u12]));

        let term0_min = if c0 >= 0.0 { c0 * row0_min } else { c0 * row0_max };
        let term0_max = if c0 >= 0.0 { c0 * row0_max } else { c0 * row0_min };
        let term1_min = if c1 >= 0.0 { c1 * row1_min } else { c1 * row1_max };
        let term1_max = if c1 >= 0.0 { c1 * row1_max } else { c1 * row1_min };
        let analytic_min = term0_min + term1_min;
        let analytic_max = term0_max + term1_max;

        prop_assert!(
            analytic_min >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
            "LogSumExp CROWN (no keepdims) analytical lower violation: \
             min={analytic_min} < lb={crown_lower}"
        );
        prop_assert!(
            analytic_max <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
            "LogSumExp CROWN (no keepdims) analytical upper violation: \
             max={analytic_max} > ub={crown_upper}"
        );

        // Joint endpoint corners: 2^6 = 64
        for &x00 in &[l00, u00] {
            for &x01 in &[l01, u01] {
                for &x02 in &[l02, u02] {
                    for &x10 in &[l10, u10] {
                        for &x11 in &[l11, u11] {
                            for &x12 in &[l12, u12] {
                                let lse_row0 = logsumexp_eval(&arr1(&[x00, x01, x02]));
                                let lse_row1 = logsumexp_eval(&arr1(&[x10, x11, x12]));
                                let combined = c0 * lse_row0 + c1 * lse_row1;

                                prop_assert!(
                                    combined >= crown_lower - LOGSUMEXP_CROWN_TOLERANCE,
                                    "LogSumExp CROWN (no keepdims) corner lower violation"
                                );
                                prop_assert!(
                                    combined <= crown_upper + LOGSUMEXP_CROWN_TOLERANCE,
                                    "LogSumExp CROWN (no keepdims) corner upper violation"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
