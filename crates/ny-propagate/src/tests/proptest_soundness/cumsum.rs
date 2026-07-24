// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Property-based soundness tests for CumSum (cumulative sum) layer.
//!
//! CumSum is a linear operator with a lower-triangular (forward) or
//! upper-triangular (reverse) Jacobian of ones. Because it's linear:
//! - IBP is exact (monotone in each input)
//! - CROWN backward with identity incoming should match IBP exactly
//!
//! Tests cover all 4 variants: {forward, reverse} × {inclusive, exclusive}.
//! Part of #3919.

use crate::layers::common::BoundPropagation;
use crate::layers::{CumsumLayer, Layer};
use crate::LinearBounds;
use ndarray::{arr1, Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

/// Compute forward inclusive cumsum on a slice.
fn cumsum_forward_inclusive(x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let mut acc = 0.0f32;
    for &v in x {
        acc += v;
        out.push(acc);
    }
    out
}

/// Compute forward exclusive cumsum on a slice: y[i] = sum(x[0..i]).
fn cumsum_forward_exclusive(x: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let mut acc = 0.0f32;
    for &v in x {
        out.push(acc);
        acc += v;
    }
    out
}

/// Compute reverse inclusive cumsum (suffix sum) on a slice: y[i] = sum(x[i..T]).
fn cumsum_reverse_inclusive(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    let mut acc = 0.0f32;
    for i in (0..x.len()).rev() {
        acc += x[i];
        out[i] = acc;
    }
    out
}

/// Compute reverse exclusive cumsum on a slice: y[i] = sum(x[i+1..T]).
fn cumsum_reverse_exclusive(x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    let mut acc = 0.0f32;
    for i in (0..x.len()).rev() {
        out[i] = acc;
        acc += x[i];
    }
    out
}

/// Helper: concretize CROWN linear bounds against a pre-activation tensor.
fn concretize_crown_1d(
    result: &LinearBounds,
    pre_activation: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    // =========================================================================
    // IBP SOUNDNESS TESTS
    // =========================================================================

    /// Forward inclusive CumSum IBP: for any x in [l, u], cumsum(x) in bounds.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cumsum_ibp_forward_inclusive(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2, l3, l4]).into_dyn(),
            arr1(&[u0, u1, u2, u3, u4]).into_dyn(),
        ).unwrap();

        let layer = CumsumLayer::new(0, false, false);
        let output = layer.propagate_ibp(&input).unwrap();

        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3), (l4, u4)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        // Check sampled combinations (3^5 = 243)
        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        for &s4 in &samples[4] {
                            let x = [s0, s1, s2, s3, s4];
                            let y = cumsum_forward_inclusive(&x);
                            for (i, &yi) in y.iter().enumerate() {
                                let ol = output.lower().as_slice().unwrap()[i];
                                let ou = output.upper().as_slice().unwrap()[i];
                                prop_assert!(
                                    ol - FP_TOLERANCE <= yi && yi <= ou + FP_TOLERANCE,
                                    "Forward inclusive IBP violation at {i}: \
                                     cumsum({x:?})[{i}]={yi} not in [{ol}, {ou}]"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Forward exclusive CumSum IBP soundness.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cumsum_ibp_forward_exclusive(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2, l3]).into_dyn(),
            arr1(&[u0, u1, u2, u3]).into_dyn(),
        ).unwrap();

        let layer = CumsumLayer::new(0, true, false);
        let output = layer.propagate_ibp(&input).unwrap();

        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let x = [s0, s1, s2, s3];
                        let y = cumsum_forward_exclusive(&x);
                        for (i, &yi) in y.iter().enumerate() {
                            let ol = output.lower().as_slice().unwrap()[i];
                            let ou = output.upper().as_slice().unwrap()[i];
                            prop_assert!(
                                ol - FP_TOLERANCE <= yi && yi <= ou + FP_TOLERANCE,
                                "Forward exclusive IBP violation at {i}: \
                                 cumsum({x:?})[{i}]={yi} not in [{ol}, {ou}]"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Reverse inclusive CumSum IBP soundness (suffix sum).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cumsum_ibp_reverse_inclusive(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2, l3]).into_dyn(),
            arr1(&[u0, u1, u2, u3]).into_dyn(),
        ).unwrap();

        let layer = CumsumLayer::new(0, false, true);
        let output = layer.propagate_ibp(&input).unwrap();

        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let x = [s0, s1, s2, s3];
                        let y = cumsum_reverse_inclusive(&x);
                        for (i, &yi) in y.iter().enumerate() {
                            let ol = output.lower().as_slice().unwrap()[i];
                            let ou = output.upper().as_slice().unwrap()[i];
                            prop_assert!(
                                ol - FP_TOLERANCE <= yi && yi <= ou + FP_TOLERANCE,
                                "Reverse inclusive IBP violation at {i}: \
                                 suffix_sum({x:?})[{i}]={yi} not in [{ol}, {ou}]"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Reverse exclusive CumSum IBP soundness.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cumsum_ibp_reverse_exclusive(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let input = BoundedTensor::new(
            arr1(&[l0, l1, l2, l3]).into_dyn(),
            arr1(&[u0, u1, u2, u3]).into_dyn(),
        ).unwrap();

        let layer = CumsumLayer::new(0, true, true);
        let output = layer.propagate_ibp(&input).unwrap();

        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let x = [s0, s1, s2, s3];
                        let y = cumsum_reverse_exclusive(&x);
                        for (i, &yi) in y.iter().enumerate() {
                            let ol = output.lower().as_slice().unwrap()[i];
                            let ou = output.upper().as_slice().unwrap()[i];
                            prop_assert!(
                                ol - FP_TOLERANCE <= yi && yi <= ou + FP_TOLERANCE,
                                "Reverse exclusive IBP violation at {i}: \
                                 excl_suffix_sum({x:?})[{i}]={yi} not in [{ol}, {ou}]"
                            );
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // CROWN BACKWARD SOUNDNESS TESTS
    // =========================================================================

    /// CumSum CROWN backward via trait dispatch: forward inclusive.
    ///
    /// Since CumSum is linear, CROWN with identity incoming bounds should
    /// produce the same concretized bounds as IBP (within FP tolerance).
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cumsum_crown_forward_inclusive(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let cumsum = CumsumLayer::new(0, false, false);
        let layer = Layer::CumSum(cumsum.clone());

        let identity = LinearBounds::identity(4);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        // Shape: (4 outputs, 4 inputs)
        prop_assert_eq!(result.lower_a.nrows(), 4);
        prop_assert_eq!(result.lower_a.ncols(), 4);

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = cumsum.propagate_ibp(&pre_activation).unwrap();

        for i in 0..4 {
            let ibp_l = ibp_result.lower().as_slice().unwrap()[i];
            let ibp_u = ibp_result.upper().as_slice().unwrap()[i];

            prop_assert!(
                (crown_lower[i] - ibp_l).abs() < FP_TOLERANCE,
                "CumSum forward inclusive CROWN-IBP lower mismatch at {i}: \
                 crown={} ibp={ibp_l}",
                crown_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_u).abs() < FP_TOLERANCE,
                "CumSum forward inclusive CROWN-IBP upper mismatch at {i}: \
                 crown={} ibp={ibp_u}",
                crown_upper[i]
            );
        }
    }

    /// CumSum CROWN backward via trait dispatch: reverse inclusive.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cumsum_crown_reverse_inclusive(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let cumsum = CumsumLayer::new(0, false, true);
        let layer = Layer::CumSum(cumsum.clone());

        let identity = LinearBounds::identity(4);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = cumsum.propagate_ibp(&pre_activation).unwrap();

        for i in 0..4 {
            let ibp_l = ibp_result.lower().as_slice().unwrap()[i];
            let ibp_u = ibp_result.upper().as_slice().unwrap()[i];

            prop_assert!(
                (crown_lower[i] - ibp_l).abs() < FP_TOLERANCE,
                "CumSum reverse inclusive CROWN-IBP lower mismatch at {i}: \
                 crown={} ibp={ibp_l}",
                crown_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_u).abs() < FP_TOLERANCE,
                "CumSum reverse inclusive CROWN-IBP upper mismatch at {i}: \
                 crown={} ibp={ibp_u}",
                crown_upper[i]
            );
        }
    }

    /// CumSum CROWN backward via trait dispatch: forward exclusive.
    ///
    /// Forward exclusive cumsum has a strictly lower-triangular Jacobian.
    /// With identity incoming bounds, concretized CROWN should match IBP.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cumsum_crown_forward_exclusive(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let cumsum = CumsumLayer::new(0, true, false);
        let layer = Layer::CumSum(cumsum.clone());

        let identity = LinearBounds::identity(4);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = cumsum.propagate_ibp(&pre_activation).unwrap();

        for i in 0..4 {
            let ibp_l = ibp_result.lower().as_slice().unwrap()[i];
            let ibp_u = ibp_result.upper().as_slice().unwrap()[i];

            prop_assert!(
                (crown_lower[i] - ibp_l).abs() < FP_TOLERANCE,
                "CumSum forward exclusive CROWN-IBP lower mismatch at {i}: \
                 crown={} ibp={ibp_l}",
                crown_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_u).abs() < FP_TOLERANCE,
                "CumSum forward exclusive CROWN-IBP upper mismatch at {i}: \
                 crown={} ibp={ibp_u}",
                crown_upper[i]
            );
        }
    }

    /// CumSum CROWN backward via trait dispatch: reverse exclusive.
    ///
    /// Reverse exclusive cumsum has a strictly upper-triangular Jacobian.
    /// With identity incoming bounds, concretized CROWN should match IBP.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_cumsum_crown_reverse_exclusive(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let cumsum = CumsumLayer::new(0, true, true);
        let layer = Layer::CumSum(cumsum.clone());

        let identity = LinearBounds::identity(4);

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = cumsum.propagate_ibp(&pre_activation).unwrap();

        for i in 0..4 {
            let ibp_l = ibp_result.lower().as_slice().unwrap()[i];
            let ibp_u = ibp_result.upper().as_slice().unwrap()[i];

            prop_assert!(
                (crown_lower[i] - ibp_l).abs() < FP_TOLERANCE,
                "CumSum reverse exclusive CROWN-IBP lower mismatch at {i}: \
                 crown={} ibp={ibp_l}",
                crown_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_u).abs() < FP_TOLERANCE,
                "CumSum reverse exclusive CROWN-IBP upper mismatch at {i}: \
                 crown={} ibp={ibp_u}",
                crown_upper[i]
            );
        }
    }

    /// CumSum CROWN backward with non-identity incoming bounds.
    ///
    /// Tests where the incoming linear bounds have non-trivial coefficients,
    /// exercising the full suffix-sum backward computation. Uses the Layer
    /// enum for trait dispatch (the actual path during CROWN propagation).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_cumsum_crown_nonidentity_incoming(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
        c2 in -3.0f32..3.0,
        c3 in -3.0f32..3.0,
    ) {
        // Incoming: one output row with coefficients [c0, c1, c2, c3]
        // applied to the 4-element cumsum output.
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 4), vec![c0, c1, c2, c3]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 4), vec![c0, c1, c2, c3]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let cumsum = CumsumLayer::new(0, false, false);
        let layer = Layer::CumSum(cumsum);

        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // f(x) = c0*cumsum(x)[0] + c1*cumsum(x)[1] + c2*cumsum(x)[2] + c3*cumsum(x)[3]
        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let y = cumsum_forward_inclusive(&[s0, s1, s2, s3]);
                        let true_output = c0 * y[0] + c1 * y[1] + c2 * y[2] + c3 * y[3];

                        let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);
                        prop_assert!(
                            crown_lower[0] - scale_tol <= true_output,
                            "CumSum CROWN lower violated: lb={} > f={true_output} \
                             (c=[{c0},{c1},{c2},{c3}], x=[{s0},{s1},{s2},{s3}])",
                            crown_lower[0]
                        );
                        prop_assert!(
                            true_output <= crown_upper[0] + scale_tol,
                            "CumSum CROWN upper violated: f={true_output} > ub={} \
                             (c=[{c0},{c1},{c2},{c3}], x=[{s0},{s1},{s2},{s3}])",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// CumSum CROWN backward with asymmetric lower_a/upper_a.
    ///
    /// After passing through nonlinear layers, CROWN backward produces
    /// different lower_a and upper_a coefficient matrices. This test verifies
    /// soundness with such asymmetric incoming bounds.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_cumsum_crown_asymmetric_bounds(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        lc0 in -3.0f32..3.0,
        lc1 in -3.0f32..3.0,
        lc2 in -3.0f32..3.0,
        lc3 in -3.0f32..3.0,
        uc0 in -3.0f32..3.0,
        uc1 in -3.0f32..3.0,
        uc2 in -3.0f32..3.0,
        uc3 in -3.0f32..3.0,
    ) {
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, 4), vec![lc0, lc1, lc2, lc3]).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 4), vec![uc0, uc1, uc2, uc3]).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let lower = ArrayD::from_shape_vec(IxDyn(&[4]), vec![l0, l1, l2, l3]).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&[4]), vec![u0, u1, u2, u3]).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let cumsum = CumsumLayer::new(0, false, false);
        let layer = Layer::CumSum(cumsum);

        let result = layer
            .propagate_crown_backward(&incoming, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward (asymmetric) failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let y = cumsum_forward_inclusive(&[s0, s1, s2, s3]);

                        let lower_output = lc0 * y[0] + lc1 * y[1] + lc2 * y[2] + lc3 * y[3];
                        let upper_output = uc0 * y[0] + uc1 * y[1] + uc2 * y[2] + uc3 * y[3];

                        let lower_tol = FP_TOLERANCE * lower_output.abs().max(1.0);
                        let upper_tol = FP_TOLERANCE * upper_output.abs().max(1.0);

                        prop_assert!(
                            crown_lower[0] - lower_tol <= lower_output,
                            "CumSum asymmetric lower violated: lb={} > f={lower_output}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            upper_output <= crown_upper[0] + upper_tol,
                            "CumSum asymmetric upper violated: f={upper_output} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// CumSum 2D CROWN backward: cumsum along last axis of [2, 3] input.
    ///
    /// Exercises the multi-fiber backward path where the cumsum axis is not
    /// the only axis. This is the practical case for Kokoro TTS (T=24000).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_cumsum_crown_2d_last_axis(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        (l4, u4) in valid_interval(5.0),
        (l5, u5) in valid_interval(5.0),
    ) {
        let lower = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![l0, l1, l2, l3, l4, l5],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), vec![u0, u1, u2, u3, u4, u5],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        // axis=-1 (last axis, length 3)
        let cumsum = CumsumLayer::new(-1, false, false);
        let layer = Layer::CumSum(cumsum.clone());

        let identity = LinearBounds::identity(6); // flattened 2*3 = 6

        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("propagate_crown_backward 2D failed: {e}")
            ))?;

        prop_assert_eq!(result.lower_a.nrows(), 6);
        prop_assert_eq!(result.lower_a.ncols(), 6);

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);
        let ibp_result = cumsum.propagate_ibp(&pre_activation).unwrap();

        // CROWN should match IBP for linear ops
        let ibp_lower: Vec<f32> = ibp_result.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_result.upper().iter().copied().collect();

        for i in 0..6 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "CumSum 2D CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "CumSum 2D CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }

        // Also verify pointwise: sample inputs, compute cumsum along axis 1
        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3), (l4, u4), (l5, u5)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lj, uj)| sample_points(lj, uj, 3))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    // Row 0: cumsum([s0, s1, s2]) = [s0, s0+s1, s0+s1+s2]
                    let y0 = cumsum_forward_inclusive(&[s0, s1, s2]);
                    for (j, &yj) in y0.iter().enumerate() {
                        prop_assert!(
                            crown_lower[j] - FP_TOLERANCE <= yj
                            && yj <= crown_upper[j] + FP_TOLERANCE,
                            "CumSum 2D row 0 violation at {j}: y={yj}, \
                             bounds=[{}, {}]",
                            crown_lower[j], crown_upper[j]
                        );
                    }
                }
            }
        }
        for &s3 in &samples[3] {
            for &s4 in &samples[4] {
                for &s5 in &samples[5] {
                    let y1 = cumsum_forward_inclusive(&[s3, s4, s5]);
                    for (j, &yj) in y1.iter().enumerate() {
                        prop_assert!(
                            crown_lower[3 + j] - FP_TOLERANCE <= yj
                            && yj <= crown_upper[3 + j] + FP_TOLERANCE,
                            "CumSum 2D row 1 violation at {j}: y={yj}, \
                             bounds=[{}, {}]",
                            crown_lower[3 + j], crown_upper[3 + j]
                        );
                    }
                }
            }
        }
    }
}
