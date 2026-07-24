// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest soundness tests for the Resize (nearest-neighbor upsample) layer.
//!
//! Resize upsamples the last two spatial dimensions by integer scale factors.
//! The operation is an exact linear replication: each input cell is duplicated
//! into a `scale_h × scale_w` block. IBP is exact (monotone replication).
//! CROWN backward sums each output block back to its source input cell, matching
//! the alpha-beta-CROWN reference `avg_pool2d(..., divisor_override=1)`.
//!
//! Reference:
//! `alpha-beta-CROWN (github.com/Verified-Intelligence/alpha-beta-CROWN) auto_LiRPA/auto_LiRPA/operators/resize.py:27-82`
//!
//! Part of #3919.

use crate::layers::common::BoundPropagation;
use crate::layers::transform::ResizeLayer;
use crate::layers::Layer;
use crate::LinearBounds;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::{sample_points, valid_interval, FP_TOLERANCE};

/// Reference nearest-neighbor upsample: replicate each input cell into a
/// `scale_h × scale_w` block over the last two dimensions.
fn reference_resize(
    input: &[f32],
    input_shape: &[usize],
    scale_h: usize,
    scale_w: usize,
) -> Vec<f32> {
    let ndim = input_shape.len();
    assert!(ndim >= 2);
    let in_h = input_shape[ndim - 2];
    let in_w = input_shape[ndim - 1];
    let out_h = in_h * scale_h;
    let out_w = in_w * scale_w;

    // Leading dimensions product (everything before H, W).
    let leading: usize = input_shape[..ndim - 2].iter().product();
    let in_spatial = in_h * in_w;
    let out_spatial = out_h * out_w;

    let mut output = Vec::with_capacity(leading * out_spatial);
    for batch in 0..leading {
        let base = batch * in_spatial;
        for oh in 0..out_h {
            for ow in 0..out_w {
                let ih = oh / scale_h;
                let iw = ow / scale_w;
                output.push(input[base + ih * in_w + iw]);
            }
        }
    }
    output
}

/// Concretize LinearBounds against pre-activation interval bounds.
fn concretize_crown_1d(
    result: &LinearBounds,
    pre_activation: &BoundedTensor,
) -> (Vec<f32>, Vec<f32>) {
    let concrete = result.concretize(pre_activation);
    let lower: Vec<f32> = concrete.lower().iter().copied().collect();
    let upper: Vec<f32> = concrete.upper().iter().copied().collect();
    (lower, upper)
}

// =============================================================================
// IBP SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// IBP soundness: 2×2 upsample on [1, 2, 3] (C=1, H=2, W=3).
    ///
    /// For any concrete point within input bounds, the resized output must lie
    /// within IBP output bounds. Resize is exact (monotone replication), so
    /// bounds should match the reference exactly.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_resize_ibp_2x2(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
        (l4, u4) in valid_interval(10.0),
        (l5, u5) in valid_interval(10.0),
    ) {
        let shape = [1usize, 2, 3];
        let lower_vals = vec![l0, l1, l2, l3, l4, l5];
        let upper_vals = vec![u0, u1, u2, u3, u4, u5];

        let lower = ArrayD::from_shape_vec(IxDyn(&shape), lower_vals.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&shape), upper_vals.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(2, 2);
        let output = layer.propagate_ibp(&input).map_err(|e| {
            TestCaseError::fail(format!("Resize IBP failed: {e}"))
        })?;

        prop_assert_eq!(output.shape(), &[1, 4, 6]);

        // Verify reference output matches bounds at endpoints.
        let ref_lower = reference_resize(&lower_vals, &shape, 2, 2);
        let ref_upper = reference_resize(&upper_vals, &shape, 2, 2);

        let out_lower = output.lower().as_slice().unwrap().to_vec();
        let out_upper = output.upper().as_slice().unwrap().to_vec();

        for i in 0..ref_lower.len() {
            let diff_l = (out_lower[i] - ref_lower[i]).abs();
            let diff_u = (out_upper[i] - ref_upper[i]).abs();
            prop_assert!(
                diff_l < FP_TOLERANCE,
                "Resize IBP lower mismatch at {i}: bound={} ref={}",
                out_lower[i], ref_lower[i]
            );
            prop_assert!(
                diff_u < FP_TOLERANCE,
                "Resize IBP upper mismatch at {i}: bound={} ref={}",
                out_upper[i], ref_upper[i]
            );
        }

        // Sample concrete points and verify containment.
        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3), (l4, u4), (l5, u5)];
        for idx in 0..6 {
            let (lo, hi) = intervals[idx];
            for &x in &sample_points(lo, hi, 5) {
                let mut point = lower_vals.clone();
                point[idx] = x;
                let ref_out = reference_resize(&point, &shape, 2, 2);
                for j in 0..ref_out.len() {
                    let y = ref_out[j];
                    prop_assert!(
                        out_lower[j] - FP_TOLERANCE <= y && y <= out_upper[j] + FP_TOLERANCE,
                        "Resize IBP soundness violation at output {j}: y={y} not in [{}, {}]",
                        out_lower[j], out_upper[j]
                    );
                }
            }
        }
    }

    /// IBP soundness: asymmetric scale factors (scale_h=1, scale_w=3).
    ///
    /// Tests that non-square scale factors work correctly. Only width is upsampled.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_resize_ibp_asymmetric_scale(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let shape = [2usize, 2];
        let lower_vals = vec![l0, l1, l2, l3];
        let upper_vals = vec![u0, u1, u2, u3];

        let lower = ArrayD::from_shape_vec(IxDyn(&shape), lower_vals.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&shape), upper_vals.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(1, 3);
        let output = layer.propagate_ibp(&input).map_err(|e| {
            TestCaseError::fail(format!("Resize IBP asym failed: {e}"))
        })?;

        prop_assert_eq!(output.shape(), &[2, 6]);

        let ref_lower = reference_resize(&lower_vals, &shape, 1, 3);
        let ref_upper = reference_resize(&upper_vals, &shape, 1, 3);

        let out_lower = output.lower().as_slice().unwrap().to_vec();
        let out_upper = output.upper().as_slice().unwrap().to_vec();

        for i in 0..ref_lower.len() {
            let diff_l = (out_lower[i] - ref_lower[i]).abs();
            let diff_u = (out_upper[i] - ref_upper[i]).abs();
            prop_assert!(
                diff_l < FP_TOLERANCE,
                "Resize IBP asym lower mismatch at {i}: bound={} ref={}",
                out_lower[i], ref_lower[i]
            );
            prop_assert!(
                diff_u < FP_TOLERANCE,
                "Resize IBP asym upper mismatch at {i}: bound={} ref={}",
                out_upper[i], ref_upper[i]
            );
        }
    }

    /// IBP soundness: 4D input [N, C, H, W] with batch and channel dims.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_resize_ibp_4d(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        // Shape [1, 1, 2, 2] -> [1, 1, 4, 4] with 2x2 scale
        let shape = [1usize, 1, 2, 2];
        let lower_vals = vec![l0, l1, l2, l3];
        let upper_vals = vec![u0, u1, u2, u3];

        let lower = ArrayD::from_shape_vec(IxDyn(&shape), lower_vals.clone()).unwrap();
        let upper = ArrayD::from_shape_vec(IxDyn(&shape), upper_vals.clone()).unwrap();
        let input = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(2, 2);
        let output = layer.propagate_ibp(&input).map_err(|e| {
            TestCaseError::fail(format!("Resize IBP 4D failed: {e}"))
        })?;

        prop_assert_eq!(output.shape(), &[1, 1, 4, 4]);

        let ref_lower = reference_resize(&lower_vals, &shape, 2, 2);
        let ref_upper = reference_resize(&upper_vals, &shape, 2, 2);

        let out_lower = output.lower().as_slice().unwrap().to_vec();
        let out_upper = output.upper().as_slice().unwrap().to_vec();

        for i in 0..ref_lower.len() {
            let diff_l = (out_lower[i] - ref_lower[i]).abs();
            let diff_u = (out_upper[i] - ref_upper[i]).abs();
            prop_assert!(
                diff_l < FP_TOLERANCE,
                "Resize IBP 4D lower mismatch at {i}: bound={} ref={}",
                out_lower[i], ref_lower[i]
            );
            prop_assert!(
                diff_u < FP_TOLERANCE,
                "Resize IBP 4D upper mismatch at {i}: bound={} ref={}",
                out_upper[i], ref_upper[i]
            );
        }
    }
}

// =============================================================================
// CROWN BACKWARD SOUNDNESS
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(500) })]

    /// CROWN backward with identity incoming bounds.
    ///
    /// For a linear operator like Resize, identity CROWN backward should produce
    /// bounds that match IBP exactly: the resized lower/upper bounds.
    /// Input [1, 2, 2] -> output [1, 4, 4] (16 elements) with 2×2 scale.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_resize_crown_identity(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let shape = [1usize, 2, 2];
        let lower = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(2, 2);

        // Output size = 1*4*4 = 16
        let identity = LinearBounds::identity(16);

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .map_err(|e| TestCaseError::fail(
                format!("Resize CROWN identity failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // Compare against IBP
        let ibp_result = layer.propagate_ibp(&pre_activation).unwrap();
        let ibp_lower: Vec<f32> = ibp_result.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_result.upper().iter().copied().collect();

        for i in 0..16 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Resize CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Resize CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }
    }

    /// CROWN backward with non-identity incoming bounds.
    ///
    /// Random coefficients over the 16-element output. For every sampled input
    /// point, the true composed output must lie within concretized CROWN bounds.
    /// Input [1, 2, 2] -> output [1, 4, 4] with 2×2 scale.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_resize_crown_nonidentity(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        let shape = [1usize, 2, 2];
        let lower = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(2, 2);

        // 1-row incoming: alternating coefficients over 16-element output.
        // Pattern: c0 and c1 alternate to exercise mixed-sign accumulation.
        let coeffs: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { c0 } else { c1 }).collect();
        let incoming = LinearBounds::new_or_conservative(
            Array2::from_shape_vec((1, 16), coeffs.clone()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 16), coeffs.clone()).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&incoming, &pre_activation)
            .map_err(|e| TestCaseError::fail(
                format!("Resize CROWN nonidentity failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        // Sample concrete inputs and verify containment.
        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lo, hi)| sample_points(lo, hi, 5))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let input_point = vec![s0, s1, s2, s3];
                        let resized = reference_resize(&input_point, &shape, 2, 2);
                        let true_output: f32 = resized.iter()
                            .zip(coeffs.iter())
                            .map(|(&r, &c)| c * r)
                            .sum();

                        let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);
                        prop_assert!(
                            crown_lower[0] - scale_tol <= true_output,
                            "Resize CROWN lower violated: lb={} > true={true_output}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            true_output <= crown_upper[0] + scale_tol,
                            "Resize CROWN upper violated: true={true_output} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// CROWN backward with asymmetric lower/upper coefficient matrices.
    ///
    /// Tests the case where lower_a != upper_a, which occurs after passing
    /// through nonlinear layers in multi-layer CROWN backward.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_resize_crown_asymmetric(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        lc0 in -3.0f32..3.0,
        lc1 in -3.0f32..3.0,
        uc0 in -3.0f32..3.0,
        uc1 in -3.0f32..3.0,
    ) {
        let shape = [1usize, 2, 2];
        let lower = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(2, 2);

        let lower_coeffs: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { lc0 } else { lc1 }).collect();
        let upper_coeffs: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { uc0 } else { uc1 }).collect();

        let incoming = LinearBounds::new_or_conservative(
            Array2::from_shape_vec((1, 16), lower_coeffs.clone()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 16), upper_coeffs.clone()).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&incoming, &pre_activation)
            .map_err(|e| TestCaseError::fail(
                format!("Resize CROWN asymmetric failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lo, hi)| sample_points(lo, hi, 5))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let input_point = vec![s0, s1, s2, s3];
                        let resized = reference_resize(&input_point, &shape, 2, 2);

                        // Lower bound uses lower_coeffs, upper uses upper_coeffs.
                        let lower_val: f32 = resized.iter()
                            .zip(lower_coeffs.iter())
                            .map(|(&r, &c)| c * r)
                            .sum();
                        let upper_val: f32 = resized.iter()
                            .zip(upper_coeffs.iter())
                            .map(|(&r, &c)| c * r)
                            .sum();

                        let lower_tol = FP_TOLERANCE * lower_val.abs().max(1.0);
                        let upper_tol = FP_TOLERANCE * upper_val.abs().max(1.0);

                        prop_assert!(
                            crown_lower[0] - lower_tol <= lower_val,
                            "Resize CROWN asym lower violated: lb={} > val={lower_val}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            upper_val <= crown_upper[0] + upper_tol,
                            "Resize CROWN asym upper violated: val={upper_val} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// CROWN backward via Layer enum dispatch (tests the full propagate_crown_backward path).
    ///
    /// Ensures the BoundPropagation trait dispatch for Resize works correctly
    /// when invoked through the Layer enum, including pre_activation routing.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_resize_crown_via_layer_dispatch(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let shape = [1usize, 2, 2];
        let lower = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let layer = Layer::Resize(ResizeLayer::new(2, 2));

        let identity = LinearBounds::identity(16);

        // Must pass Some(pre_activation) since Resize requires it.
        let result = layer
            .propagate_crown_backward(&identity, Some(&pre_activation))
            .map_err(|e| TestCaseError::fail(
                format!("Resize CROWN dispatch failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let ibp_result = ResizeLayer::new(2, 2).propagate_ibp(&pre_activation).unwrap();
        let ibp_lower: Vec<f32> = ibp_result.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_result.upper().iter().copied().collect();

        for i in 0..16 {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() < FP_TOLERANCE,
                "Resize Layer dispatch CROWN-IBP lower mismatch at {i}: crown={} ibp={}",
                crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() < FP_TOLERANCE,
                "Resize Layer dispatch CROWN-IBP upper mismatch at {i}: crown={} ibp={}",
                crown_upper[i], ibp_upper[i]
            );
        }
    }

    /// CROWN backward with asymmetric scale (scale_h=2, scale_w=3).
    ///
    /// Verifies coefficient accumulation is correct when scale_h != scale_w.
    /// Input [2, 2] -> output [4, 6] (24 elements).
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_resize_crown_asymmetric_scale(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        c0 in -2.0f32..2.0,
        c1 in -2.0f32..2.0,
    ) {
        let shape = [2usize, 2];
        let lower = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(2, 3);

        // 24-element output, 1-row coefficients
        let coeffs: Vec<f32> = (0..24).map(|i| if i % 2 == 0 { c0 } else { c1 }).collect();
        let incoming = LinearBounds::new_or_conservative(
            Array2::from_shape_vec((1, 24), coeffs.clone()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 24), coeffs.clone()).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&incoming, &pre_activation)
            .map_err(|e| TestCaseError::fail(
                format!("Resize CROWN asym scale failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lo, hi)| sample_points(lo, hi, 5))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let input_point = vec![s0, s1, s2, s3];
                        let resized = reference_resize(&input_point, &shape, 2, 3);
                        let true_output: f32 = resized.iter()
                            .zip(coeffs.iter())
                            .map(|(&r, &c)| c * r)
                            .sum();

                        let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);
                        prop_assert!(
                            crown_lower[0] - scale_tol <= true_output,
                            "Resize CROWN asym scale lower violated: lb={} > true={true_output}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            true_output <= crown_upper[0] + scale_tol,
                            "Resize CROWN asym scale upper violated: true={true_output} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }

    /// CROWN backward with identity scale (1×1) is a passthrough.
    ///
    /// Scale factor 1 should produce identical bounds to the input.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_resize_crown_identity_scale(
        (l0, u0) in valid_interval(10.0),
        (l1, u1) in valid_interval(10.0),
        (l2, u2) in valid_interval(10.0),
        (l3, u3) in valid_interval(10.0),
    ) {
        let shape = [2usize, 2];
        let lower = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(1, 1);

        // Output size = input size = 4
        let identity = LinearBounds::identity(4);

        let result = layer
            .propagate_linear_with_bounds(&identity, &pre_activation)
            .map_err(|e| TestCaseError::fail(
                format!("Resize CROWN identity scale failed: {e}")
            ))?;

        // Scale 1×1 CROWN backward should return identity coefficients.
        prop_assert_eq!(result.lower_a().shape(), &[4, 4]);
        prop_assert_eq!(result.upper_a().shape(), &[4, 4]);

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!(
                    (result.lower_a()[[i, j]] - expected).abs() < FP_TOLERANCE,
                    "Resize 1x1 lower_a[{i},{j}] = {} != {expected}",
                    result.lower_a()[[i, j]]
                );
                prop_assert!(
                    (result.upper_a()[[i, j]] - expected).abs() < FP_TOLERANCE,
                    "Resize 1x1 upper_a[{i},{j}] = {} != {expected}",
                    result.upper_a()[[i, j]]
                );
            }
        }
    }

    /// CROWN backward with 4D input [N, C, H, W].
    ///
    /// Verifies that leading batch/channel dimensions are handled correctly
    /// in the output-to-input mapping and coefficient accumulation.
    #[ntest::timeout(60000)]
    #[test]
    fn soundness_resize_crown_4d(
        (l0, u0) in valid_interval(5.0),
        (l1, u1) in valid_interval(5.0),
        (l2, u2) in valid_interval(5.0),
        (l3, u3) in valid_interval(5.0),
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
    ) {
        // [1, 1, 2, 2] -> [1, 1, 4, 4], 16 output elements
        let shape = [1usize, 1, 2, 2];
        let lower = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![l0, l1, l2, l3],
        ).unwrap();
        let upper = ArrayD::from_shape_vec(
            IxDyn(&shape), vec![u0, u1, u2, u3],
        ).unwrap();
        let pre_activation = BoundedTensor::new(lower, upper).unwrap();

        let layer = ResizeLayer::new(2, 2);

        let coeffs: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { c0 } else { c1 }).collect();
        let incoming = LinearBounds::new_or_conservative(
            Array2::from_shape_vec((1, 16), coeffs.clone()).unwrap(),
            Array1::zeros(1),
            Array2::from_shape_vec((1, 16), coeffs.clone()).unwrap(),
            Array1::zeros(1),
        ).unwrap();

        let result = layer
            .propagate_linear_with_bounds(&incoming, &pre_activation)
            .map_err(|e| TestCaseError::fail(
                format!("Resize CROWN 4D failed: {e}")
            ))?;

        let (crown_lower, crown_upper) = concretize_crown_1d(&result, &pre_activation);

        let intervals = [(l0, u0), (l1, u1), (l2, u2), (l3, u3)];
        let samples: Vec<Vec<f32>> = intervals.iter()
            .map(|&(lo, hi)| sample_points(lo, hi, 5))
            .collect();

        for &s0 in &samples[0] {
            for &s1 in &samples[1] {
                for &s2 in &samples[2] {
                    for &s3 in &samples[3] {
                        let input_point = vec![s0, s1, s2, s3];
                        let resized = reference_resize(&input_point, &shape, 2, 2);
                        let true_output: f32 = resized.iter()
                            .zip(coeffs.iter())
                            .map(|(&r, &c)| c * r)
                            .sum();

                        let scale_tol = FP_TOLERANCE * true_output.abs().max(1.0);
                        prop_assert!(
                            crown_lower[0] - scale_tol <= true_output,
                            "Resize CROWN 4D lower violated: lb={} > true={true_output}",
                            crown_lower[0]
                        );
                        prop_assert!(
                            true_output <= crown_upper[0] + scale_tol,
                            "Resize CROWN 4D upper violated: true={true_output} > ub={}",
                            crown_upper[0]
                        );
                    }
                }
            }
        }
    }
}
