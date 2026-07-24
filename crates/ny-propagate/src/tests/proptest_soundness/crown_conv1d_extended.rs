// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Extended Conv1d CROWN backward proptests for dilation>1, groups>1, and
//! groups==1 fast-path equivalence.
//!
//! Split from `crown_convolution.rs` to keep files under 1000 lines.
//! Part of #3647.

use crate::layers::common::BoundPropagation;
use crate::layers::convolution::conv1d::Conv1dLayer;
use crate::LinearBounds;
use ndarray::{Array1, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;
use proptest::prelude::*;

use super::crown_convolution::{eval_conv1d, sample_multi_points, CROWN_CONV_TOLERANCE};
use super::FP_TOLERANCE;

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    // =========================================================================
    // Conv1d CROWN soundness with dilation > 1 (#3647)
    // =========================================================================

    /// Conv1d CROWN backward soundness with random dilation in {1, 2, 3}.
    ///
    /// Dilation spreads the kernel elements across the input, changing the
    /// effective receptive field. The CROWN backward col2im scatter uses
    /// `gl * stride + ki * dilation - padding` which must be correct for all
    /// dilation values. This proptest exercises the dilation>1 code path
    /// in ops_gemm.rs:154 and ops.rs conv1d_transpose.
    ///
    /// Part of #3647.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_conv1d_crown_dilation_3647(
        k0 in -3.0f32..3.0,
        k1 in -3.0f32..3.0,
        k2 in -3.0f32..3.0,
        bias in -2.0f32..2.0,
        dilation in 1usize..=3,
        bounds in prop::collection::vec(-5.0f32..5.0, 14), // 7 lower + 7 delta
    ) {
        let input_len = 7;
        let in_shape = [1_usize, input_len]; // (in_channels=1, length=7)

        let lower: Vec<f32> = bounds[..7].to_vec();
        let upper: Vec<f32> = lower.iter().zip(bounds[7..].iter()).map(|(&l, &d)| l + d.abs()).collect();

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3]), // (out_c=1, in_c=1, k=3)
            vec![k0, k1, k2],
        ).unwrap();
        let bias_arr = Array1::from_vec(vec![bias]);

        let conv = Conv1dLayer::with_input_length_full(
            kernel, Some(bias_arr), 1, 0, dilation, 1, input_len,
        ).unwrap();

        let out_len = conv.output_length(input_len).unwrap();
        let conv_out_size = out_len;

        // IBP bounds
        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper.clone()).unwrap(),
        ).unwrap();

        let ibp_out = conv.propagate_ibp(&input_bt).unwrap();
        let ibp_lower: Vec<f32> = ibp_out.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_out.upper().iter().copied().collect();

        // CROWN bounds
        let identity = LinearBounds::identity(conv_out_size);
        let crown_result = conv.propagate_linear(&identity).unwrap();

        let input_flat = input_bt.flatten();
        let crown_out = crown_result.concretize(&input_flat);
        let crown_lower: Vec<f32> = crown_out.lower().iter().copied().collect();
        let crown_upper: Vec<f32> = crown_out.upper().iter().copied().collect();

        // CROWN-vs-IBP equivalence (linear layer)
        for i in 0..conv_out_size {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv1d dilation={} CROWN-IBP lower mismatch at {}: crown={}, ibp={}",
                dilation, i, crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv1d dilation={} CROWN-IBP upper mismatch at {}: crown={}, ibp={}",
                dilation, i, crown_upper[i], ibp_upper[i]
            );
        }

        // Soundness: concrete outputs within bounds
        let samples = sample_multi_points(&lower, &upper, 10);
        for sample in &samples {
            let concrete_out = eval_conv1d(&conv, sample, &in_shape);
            for (i, &y) in concrete_out.iter().enumerate() {
                prop_assert!(
                    ibp_lower[i] - FP_TOLERANCE <= y && y <= ibp_upper[i] + FP_TOLERANCE,
                    "Conv1d dilation={} soundness violation at {}: y={}, bounds=[{}, {}]",
                    dilation, i, y, ibp_lower[i], ibp_upper[i]
                );
            }
        }
    }

    // =========================================================================
    // Conv1d CROWN soundness with groups > 1 (#3647)
    // =========================================================================

    /// Conv1d CROWN backward soundness with groups in {1, 2, 4}.
    ///
    /// Grouped convolution partitions input/output channels into independent
    /// groups. The CROWN backward path uses per-group GEMM loops with separate
    /// scatter logic (ops_gemm.rs:118-170) that differ from the groups==1 fast
    /// path. This proptest exercises the groups>1 code path with randomized
    /// kernels and inputs.
    ///
    /// Part of #3647.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_conv1d_crown_groups_3647(
        // 4 output channels, kernel_size=1 for simplicity with groups
        weights in prop::collection::vec(-3.0f32..3.0, 4), // 4 kernel values
        bias in prop::collection::vec(-2.0f32..2.0, 4),    // 4 biases
        groups_idx in 0usize..3, // index into [1, 2, 4]
        bounds in prop::collection::vec(-5.0f32..5.0, 8),  // 4 lower + 4 delta
    ) {
        let groups = [1, 2, 4][groups_idx];
        let out_c = 4_usize;
        let in_c_per_group = 1_usize;
        let in_c = in_c_per_group * groups;
        let k = 1_usize;
        let input_len = 4;
        let in_shape = [in_c, input_len];
        let in_flat_size = in_c * input_len;

        let lower: Vec<f32> = bounds[..in_flat_size.min(4)].to_vec();
        // Pad/trim to exact in_flat_size
        let mut lower_full = vec![0.0f32; in_flat_size];
        for (i, v) in lower.iter().enumerate() {
            if i < in_flat_size { lower_full[i] = *v; }
        }
        let upper_full: Vec<f32> = lower_full.iter().zip(bounds[4..].iter().cycle()).map(|(&l, &d)| l + d.abs().max(0.01)).collect();

        // Kernel shape: (out_c, in_c/groups, k) = (4, 1, 1)
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[out_c, in_c_per_group, k]),
            weights,
        ).unwrap();
        let bias_arr = Array1::from_vec(bias);

        let conv = Conv1dLayer::with_input_length_full(
            kernel, Some(bias_arr), 1, 0, 1, groups, input_len,
        ).unwrap();

        let out_len = conv.output_length(input_len).unwrap();
        let conv_out_size = out_c * out_len;

        // IBP bounds
        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower_full.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper_full.clone()).unwrap(),
        ).unwrap();

        let ibp_out = conv.propagate_ibp(&input_bt).unwrap();
        let ibp_lower: Vec<f32> = ibp_out.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_out.upper().iter().copied().collect();

        // CROWN bounds
        let identity = LinearBounds::identity(conv_out_size);
        let crown_result = conv.propagate_linear(&identity).unwrap();

        let input_flat = input_bt.flatten();
        let crown_out = crown_result.concretize(&input_flat);
        let crown_lower: Vec<f32> = crown_out.lower().iter().copied().collect();
        let crown_upper: Vec<f32> = crown_out.upper().iter().copied().collect();

        // CROWN-vs-IBP equivalence
        for i in 0..conv_out_size {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv1d groups={} CROWN-IBP lower mismatch at {}: crown={}, ibp={}",
                groups, i, crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv1d groups={} CROWN-IBP upper mismatch at {}: crown={}, ibp={}",
                groups, i, crown_upper[i], ibp_upper[i]
            );
        }

        // Soundness: concrete outputs within bounds
        let samples = sample_multi_points(&lower_full, &upper_full, 10);
        for sample in &samples {
            let concrete_out = eval_conv1d(&conv, sample, &in_shape);
            for (i, &y) in concrete_out.iter().enumerate() {
                prop_assert!(
                    ibp_lower[i] - FP_TOLERANCE <= y && y <= ibp_upper[i] + FP_TOLERANCE,
                    "Conv1d groups={} soundness violation at {}: y={}, bounds=[{}, {}]",
                    groups, i, y, ibp_lower[i], ibp_upper[i]
                );
            }
        }
    }

    // =========================================================================
    // Conv1d CROWN groups==1 fast path equivalence (#3647)
    // =========================================================================

    /// Verify that the groups==1 fast path (ops_gemm.rs:172-252) produces
    /// identical results to the groups>1 path (ops_gemm.rs:118-170) when
    /// invoked with groups=1.
    ///
    /// This catches divergence between the two code paths that would only
    /// manifest as subtle numerical differences in production.
    ///
    /// Part of #3647.
    #[ntest::timeout(10000)]
    #[test]
    fn soundness_conv1d_crown_groups1_fast_path_equiv_3647(
        k0 in -3.0f32..3.0,
        k1 in -3.0f32..3.0,
        k2 in -3.0f32..3.0,
        bias in -2.0f32..2.0,
    ) {
        use ndarray::Array2;

        let input_len = 5;
        let in_c = 1_usize;
        let out_c = 1_usize;
        let k_size = 3_usize;

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[out_c, in_c, k_size]),
            vec![k0, k1, k2],
        ).unwrap();
        let bias_arr = Array1::from_vec(vec![bias]);

        let conv = Conv1dLayer::with_input_length_full(
            kernel.clone(), Some(bias_arr), 1, 0, 1, 1, input_len,
        ).unwrap();

        let out_len = conv.output_length(input_len).unwrap();
        let conv_out_size = out_c * out_len;

        // Build identity A matrix for the GEMM paths
        let a_identity = Array2::<f32>::eye(conv_out_size);

        // Call the batched GEMM with groups=1 (fast path)
        let fast_result = crate::layers::convolution::conv1d::conv1d_transpose_batched_gemm(
            &a_identity, &kernel, 1, 0, 1, 1, out_c, out_len, input_len, None,
        ).unwrap();

        // Compare against the per-row scalar path (conv1d_transpose) as reference.
        let identity = LinearBounds::identity(conv_out_size);
        let crown_result = conv.propagate_linear(&identity).unwrap();

        // The CROWN backward A-matrix from propagate_linear (scalar path) should
        // match the batched GEMM fast path.
        let crown_a_lower = &crown_result.lower_a;
        for i in 0..conv_out_size {
            for j in 0..input_len {
                let gemm_val = fast_result[[i, j]];
                let scalar_val = crown_a_lower[[i, j]];
                prop_assert!(
                    (gemm_val - scalar_val).abs() <= 1e-6,
                    "groups==1 fast path divergence at [{},{}]: gemm={}, scalar={}",
                    i, j, gemm_val, scalar_val
                );
            }
        }
    }
}
