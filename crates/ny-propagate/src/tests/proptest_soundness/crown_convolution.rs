// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Proptest CROWN soundness tests for convolution layers.
//!
//! Convolutions are linear layers, so CROWN backward is equivalent to
//! `propagate_linear` (transposed convolution). These tests verify:
//! 1. IBP bounds are sound (concrete conv output within [lower, upper])
//! 2. CROWN bounds concretize to the same result as IBP (linear layer property)
//! 3. CROWN backward produces correct coefficient matrices for random kernels
//!
//! Part of #40.
//!
//! WALL-CLOCK POLICY FOR THIS FILE: every `#[ntest::timeout(..)]` below is a
//! HANG SENTINEL, not a performance assertion, and the walls are deliberately
//! far above these tests' isolated cost (well under a second each).
//!
//! They have to be. These tests participate in the `ny-test-utils` env lock --
//! either holding the shared half so a concurrent writer cannot leak
//! `NY_DENSE_BUDGET_MB` into them mid-run, or holding the exclusive half
//! themselves. Waiting on that lock is CORRECT behaviour, not a hang, and the
//! wait can be long: `margin_row`'s `root_build_bit_identical_across_conv_grain`
//! holds the exclusive half across a loop that runs over 60 seconds. A 10s wall
//! turns that legitimate wait into a spurious failure -- measured, 17 of 20
//! full-suite failures at --test-threads=8 were exactly this, with zero
//! remaining `crown=-inf` leaks.
//!
//! MEASURE BEFORE LOWERING THEM.

use crate::layers::common::BoundPropagation;
use crate::layers::convolution::conv1d::{Conv1dLayer, ConvTranspose1dLayer};
use crate::layers::convolution::conv2d::{Conv2dLayer, ConvTranspose2dLayer};
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use proptest::prelude::*;

use super::FP_TOLERANCE;

/// Tolerance for CROWN-vs-IBP equivalence on linear layers.
/// Slightly looser than FP_TOLERANCE due to different computation paths
/// (transposed conv vs W+/W- splitting).
pub(super) const CROWN_CONV_TOLERANCE: f32 = 1e-4;

/// Evaluate Conv1d on a concrete input by wrapping it in a point BoundedTensor.
/// Returns a flat vector of output values.
pub(super) fn eval_conv1d(layer: &Conv1dLayer, input_flat: &[f32], shape: &[usize]) -> Vec<f32> {
    let input_nd =
        ArrayD::from_shape_vec(IxDyn(shape), input_flat.to_vec()).expect("shape mismatch");
    let point = BoundedTensor::new(input_nd.clone(), input_nd).expect("point tensor");
    let out = layer.propagate_ibp(&point).expect("conv1d eval");
    // Lower == upper for point input
    out.lower().iter().copied().collect()
}

/// Regression for #2183: Conv1d batched CROWN bias path must use f64
/// accumulation with directed rounding on the final f32 cast.
/// Converted from proptest with `_case in 0u8..1` (zero randomization).
#[ntest::timeout(300000)]
#[test]
fn directed_rounding_conv1d_batched_bias_2183() {
    let input_len = 100usize;
    let conv = Conv1dLayer::with_input_length(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0_f32]).unwrap(),
        Some(Array1::from_vec(vec![0.1_f32])),
        1,
        0,
        input_len,
    )
    .unwrap();

    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[1, input_len]), vec![1.0_f32; input_len]).unwrap(),
        ArrayD::zeros(IxDyn(&[1])),
        ArrayD::from_shape_vec(IxDyn(&[1, input_len]), vec![1.0_f32; input_len]).unwrap(),
        ArrayD::zeros(IxDyn(&[1])),
        vec![input_len],
        vec![1],
    );

    let result = conv
        .propagate_linear_batched(&bounds)
        .expect("Conv1d batched CROWN failed");

    let true_f64: f64 = (0..input_len).map(|_| 0.1_f32 as f64).sum();
    let nearest = true_f64 as f32;
    let expected_lower = if nearest as f64 <= true_f64 {
        nearest
    } else {
        next_down_f32(nearest)
    };
    let expected_upper = if nearest as f64 >= true_f64 {
        nearest
    } else {
        next_up_f32(nearest)
    };

    let mut f32_sum = 0.0_f32;
    for _ in 0..input_len {
        f32_sum += 0.1_f32;
    }
    assert_ne!(
        f32_sum.to_bits(),
        (true_f64 as f32).to_bits(),
        "test setup must exercise f64 vs f32 accumulation divergence",
    );

    assert_eq!(
        result.lower_b[[0]].to_bits(),
        expected_lower.to_bits(),
        "Conv1d lower_b must be the tight directed f32 rounding of the f64 accumulation",
    );
    assert_eq!(
        result.upper_b[[0]].to_bits(),
        expected_upper.to_bits(),
        "Conv1d upper_b must be the tight directed f32 rounding of the f64 accumulation",
    );
    assert!(
        (result.lower_b[[0]] as f64) <= true_f64,
        "Conv1d lower_b must stay <= true f64 bias",
    );
    assert!(
        (result.upper_b[[0]] as f64) >= true_f64,
        "Conv1d upper_b must stay >= true f64 bias",
    );
}

/// Regression for #2183: Conv2d batched CROWN bias path must use f64
/// accumulation with directed rounding on the final f32 cast.
/// Converted from proptest with `_case in 0u8..1` (zero randomization).
#[ntest::timeout(300000)]
#[test]
fn directed_rounding_conv2d_batched_bias_2183() {
    let _env_lock = ny_test_utils::env::lock_env();
    let _budget = ny_test_utils::env::ScopedEnvVar::set("NY_DENSE_BUDGET_MB", "2048");
    let in_h = 10usize;
    let in_w = 10usize;
    let out_elems = in_h * in_w; // kernel=1,stride=1,pad=0
    let conv = Conv2dLayer::with_input_shape(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).unwrap(),
        Some(Array1::from_vec(vec![0.1_f32])),
        (1, 1),
        (0, 0),
        in_h,
        in_w,
    )
    .unwrap();

    let bounds = BatchedLinearBounds::from_parts_unchecked(
        ArrayD::from_shape_vec(IxDyn(&[1, out_elems]), vec![1.0_f32; out_elems]).unwrap(),
        ArrayD::zeros(IxDyn(&[1])),
        ArrayD::from_shape_vec(IxDyn(&[1, out_elems]), vec![1.0_f32; out_elems]).unwrap(),
        ArrayD::zeros(IxDyn(&[1])),
        vec![out_elems],
        vec![1],
    );

    let result = conv
        .propagate_linear_batched(&bounds, None)
        .expect("Conv2d batched CROWN failed");

    let true_f64: f64 = (0..out_elems).map(|_| 0.1_f32 as f64).sum();
    let nearest = true_f64 as f32;
    let expected_lower = if nearest as f64 <= true_f64 {
        nearest
    } else {
        next_down_f32(nearest)
    };
    let expected_upper = if nearest as f64 >= true_f64 {
        nearest
    } else {
        next_up_f32(nearest)
    };

    let mut f32_sum = 0.0_f32;
    for _ in 0..out_elems {
        f32_sum += 0.1_f32;
    }
    assert_ne!(
        f32_sum.to_bits(),
        (true_f64 as f32).to_bits(),
        "test setup must exercise f64 vs f32 accumulation divergence",
    );

    assert_eq!(
        result.lower_b[[0]].to_bits(),
        expected_lower.to_bits(),
        "Conv2d lower_b must be the tight directed f32 rounding of the f64 accumulation",
    );
    assert_eq!(
        result.upper_b[[0]].to_bits(),
        expected_upper.to_bits(),
        "Conv2d upper_b must be the tight directed f32 rounding of the f64 accumulation",
    );
    assert!(
        (result.lower_b[[0]] as f64) <= true_f64,
        "Conv2d lower_b must stay <= true f64 bias",
    );
    assert!(
        (result.upper_b[[0]] as f64) >= true_f64,
        "Conv2d upper_b must stay >= true f64 bias",
    );
}

/// Evaluate Conv2d on a concrete input by wrapping it in a point BoundedTensor.
/// Returns a flat vector of output values.
fn eval_conv2d(layer: &Conv2dLayer, input_flat: &[f32], shape: &[usize]) -> Vec<f32> {
    let input_nd =
        ArrayD::from_shape_vec(IxDyn(shape), input_flat.to_vec()).expect("shape mismatch");
    let point = BoundedTensor::new(input_nd.clone(), input_nd).expect("point tensor");
    let out = layer.propagate_ibp(&point).expect("conv2d eval");
    out.lower().iter().copied().collect()
}

/// Evaluate ConvTranspose1d on a concrete input by wrapping it in a point BoundedTensor.
/// Returns a flat vector of output values.
fn eval_conv_transpose1d(
    layer: &ConvTranspose1dLayer,
    input_flat: &[f32],
    shape: &[usize],
) -> Vec<f32> {
    let input_nd =
        ArrayD::from_shape_vec(IxDyn(shape), input_flat.to_vec()).expect("shape mismatch");
    let point = BoundedTensor::new(input_nd.clone(), input_nd).expect("point tensor");
    let out = layer.propagate_ibp(&point).expect("conv_transpose1d eval");
    out.lower().iter().copied().collect()
}

/// Evaluate ConvTranspose2d on a concrete input by wrapping it in a point BoundedTensor.
/// Returns a flat vector of output values.
fn eval_conv_transpose2d(
    layer: &ConvTranspose2dLayer,
    input_flat: &[f32],
    shape: &[usize],
) -> Vec<f32> {
    let input_nd =
        ArrayD::from_shape_vec(IxDyn(shape), input_flat.to_vec()).expect("shape mismatch");
    let point = BoundedTensor::new(input_nd.clone(), input_nd).expect("point tensor");
    let out = layer.propagate_ibp(&point).expect("conv_transpose2d eval");
    out.lower().iter().copied().collect()
}

/// Sample random points within interval bounds.
/// Returns `num_samples` vectors, each with values within the corresponding
/// [lower, upper] intervals.
pub(super) fn sample_multi_points(
    lower: &[f32],
    upper: &[f32],
    num_samples: usize,
) -> Vec<Vec<f32>> {
    // Deterministic sampling: use evenly-spaced interpolation factors
    let mut points = Vec::with_capacity(num_samples);
    for s in 0..num_samples {
        let t = s as f32 / (num_samples - 1).max(1) as f32;
        let point: Vec<f32> = lower
            .iter()
            .zip(upper.iter())
            .enumerate()
            .map(|(i, (&l, &u))| {
                // Use different phases per dimension to avoid always sampling corners
                let phase = (i as f32 * 0.618_034 + t) % 1.0; // golden ratio spacing
                l + (u - l) * phase
            })
            .collect();
        points.push(point);
    }
    // Always include corners: all-lower and all-upper
    points.push(lower.to_vec());
    points.push(upper.to_vec());
    points
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(200) })]

    // =========================================================================
    // Conv1d CROWN soundness (1 channel in, 1 channel out, kernel_size=3)
    // =========================================================================

    /// Conv1d IBP + CROWN soundness with random kernel, no padding.
    ///
    /// Tests a minimal Conv1d (1 in_channel, 1 out_channel, kernel_size=3, stride=1)
    /// with random weights on a length-5 input.
    ///
    /// Verifies:
    /// 1. IBP bounds contain all sampled concrete outputs
    /// 2. CROWN concretized bounds match IBP (linear layer equivalence)
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_conv1d_crown_1c_k3(
        k0 in -3.0f32..3.0,
        k1 in -3.0f32..3.0,
        k2 in -3.0f32..3.0,
        bias in -2.0f32..2.0,
        bounds in prop::collection::vec(-5.0f32..5.0, 10), // 5 pairs -> 5 elements
    ) {
        // Excluded from overlapping an env WRITER. The leak is specific and
        // known: `NY_DENSE_BUDGET_MB`, read process-globally by
        // `crown_memory::explicit_cpu_crown_dense_budget_bytes`. A concurrent
        // test setting it to 0 starves this one's CROWN into an IBP fallback,
        // which surfaces here as `crown=-inf` -- an enclosure violation that
        // is really a race. Observed failing at --test-threads=4 and =8.
        let _env = crate::tests::lock_env_shared();
        // Reconstruct bounds as pairs from flat vector
        let input_len = 5;
        let in_shape = [1_usize, input_len]; // (in_channels=1, length=5)

        // Create lower/upper from the flat vector (first 5 = lower centers, next 5 = deltas)
        let lower: Vec<f32> = bounds[..5].to_vec();
        let upper: Vec<f32> = lower.iter().zip(bounds[5..].iter()).map(|(&l, &d)| l + d.abs()).collect();

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3]), // (out_channels=1, in_channels=1, kernel_size=3)
            vec![k0, k1, k2],
        ).unwrap();
        let bias_arr = Array1::from_vec(vec![bias]);

        let conv = Conv1dLayer::with_input_length(
            kernel, Some(bias_arr), 1, 0, input_len,
        ).unwrap();

        // IBP bounds
        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper.clone()).unwrap(),
        ).unwrap();

        let ibp_out = conv.propagate_ibp(&input_bt).unwrap();
        let ibp_lower: Vec<f32> = ibp_out.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_out.upper().iter().copied().collect();

        // CROWN bounds: identity incoming on the flattened output
        let out_len = input_len - 3 + 1; // = 3 (no padding, stride 1)
        let conv_out_size = out_len; // 3 (1 output channel)
        let identity = LinearBounds::identity(conv_out_size);
        let crown_result = conv.propagate_linear(&identity).unwrap();

        // Concretize CROWN bounds with flattened input
        let input_flat = input_bt.flatten();
        let crown_out = crown_result.concretize(&input_flat);
        let crown_lower: Vec<f32> = crown_out.lower().iter().copied().collect();
        let crown_upper: Vec<f32> = crown_out.upper().iter().copied().collect();

        // 1. CROWN-vs-IBP equivalence for linear layers
        for i in 0..conv_out_size {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv1d CROWN-IBP lower mismatch at {}: crown={}, ibp={}",
                i, crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv1d CROWN-IBP upper mismatch at {}: crown={}, ibp={}",
                i, crown_upper[i], ibp_upper[i]
            );
        }

        // 2. Soundness: concrete outputs within bounds
        let samples = sample_multi_points(&lower, &upper, 10);
        for sample in &samples {
            let concrete_out = eval_conv1d(&conv, sample, &in_shape);
            for (i, &y) in concrete_out.iter().enumerate() {
                prop_assert!(
                    ibp_lower[i] - FP_TOLERANCE <= y && y <= ibp_upper[i] + FP_TOLERANCE,
                    "Conv1d soundness violation at output {}: y={}, bounds=[{}, {}]",
                    i, y, ibp_lower[i], ibp_upper[i]
                );
            }
        }
    }

    // =========================================================================
    // Conv1d CROWN with non-identity incoming bounds
    // =========================================================================

    /// Conv1d CROWN with non-identity incoming linear bounds (mixed-sign coefficients).
    ///
    /// Tests that the transposed convolution correctly composes with arbitrary
    /// incoming coefficient matrices, including negative coefficients that exercise
    /// the sign-switching paths.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_conv1d_crown_non_identity(
        k0 in -3.0f32..3.0,
        k1 in -3.0f32..3.0,
        k2 in -3.0f32..3.0,
        bias in -2.0f32..2.0,
        c0 in -3.0f32..3.0,
        c1 in -3.0f32..3.0,
        c2 in -3.0f32..3.0,
        offset in -1.0f32..1.0,
        bounds in prop::collection::vec(-3.0f32..3.0, 10),
    ) {
        // At least one non-trivial coefficient
        prop_assume!(c0.abs() > 0.01 || c1.abs() > 0.01 || c2.abs() > 0.01);

        let input_len = 5;
        let in_shape = [1_usize, input_len];
        let lower: Vec<f32> = bounds[..5].to_vec();
        let upper: Vec<f32> = lower.iter().zip(bounds[5..].iter()).map(|(&l, &d)| l + d.abs()).collect();

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3]),
            vec![k0, k1, k2],
        ).unwrap();
        let bias_arr = Array1::from_vec(vec![bias]);

        let conv = Conv1dLayer::with_input_length(
            kernel, Some(bias_arr), 1, 0, input_len,
        ).unwrap();

        let out_len = input_len - 3 + 1; // 3
        let conv_out_size = out_len;

        // Non-identity incoming bounds: 1 output = c0*y0 + c1*y1 + c2*y2 + offset
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, conv_out_size), vec![c0, c1, c2]).unwrap(),
            Array1::from_vec(vec![offset]),
            Array2::from_shape_vec((1, conv_out_size), vec![c0, c1, c2]).unwrap(),
            Array1::from_vec(vec![offset]),
        ).unwrap();

        let crown_result = conv.propagate_linear(&incoming).unwrap();

        // Concretize
        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper.clone()).unwrap(),
        ).unwrap();
        let input_flat = input_bt.flatten();
        let crown_out = crown_result.concretize(&input_flat);
        let crown_lower = crown_out.lower()[[0]];
        let crown_upper = crown_out.upper()[[0]];

        // Soundness: for each sample, compute conv then apply linear combination
        let samples = sample_multi_points(&lower, &upper, 15);
        for sample in &samples {
            let concrete_conv = eval_conv1d(&conv, sample, &in_shape);
            // Apply c0*y0 + c1*y1 + c2*y2 + offset
            let concrete_combined = c0 * concrete_conv[0]
                + c1 * concrete_conv[1]
                + c2 * concrete_conv[2]
                + offset;

            let tol = CROWN_CONV_TOLERANCE * concrete_combined.abs().max(1.0);
            prop_assert!(
                crown_lower - tol <= concrete_combined && concrete_combined <= crown_upper + tol,
                "Conv1d non-identity CROWN soundness violation: y={}, bounds=[{}, {}], coeffs=[{},{},{}]",
                concrete_combined, crown_lower, crown_upper, c0, c1, c2
            );
        }
    }

    // =========================================================================
    // Conv2d CROWN soundness (1 channel in, 2 channels out, kernel_size=3x3)
    // =========================================================================

    /// Conv2d IBP + CROWN soundness with random kernel, no padding.
    ///
    /// Tests Conv2d (1 in_channel, 2 out_channels, 3x3 kernel, stride=1)
    /// with random weights on a 4x4 input.
    ///
    /// Verifies:
    /// 1. IBP bounds contain all sampled concrete outputs
    /// 2. CROWN concretized bounds match IBP (linear layer equivalence)
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_conv2d_crown_1c_k3(
        weights in prop::collection::vec(-2.0f32..2.0, 18), // 2 * 1 * 3 * 3 = 18
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        bounds in prop::collection::vec(-3.0f32..3.0, 32), // 16 lower + 16 delta -> 4x4 input
    ) {
        // This test requires the dense Conv2d CROWN path. Serialize with tests
        // that deliberately set the process-wide budget to zero, otherwise a
        // concurrent budget-guard test can turn this into an unrelated
        // CpuMemoryExceeded failure (or universal CROWN fallback).
        let _env_lock = ny_test_utils::env::lock_env();
        let _budget = ny_test_utils::env::ScopedEnvVar::set("NY_DENSE_BUDGET_MB", "2048");

        let in_h = 4_usize;
        let in_w = 4_usize;
        let in_c = 1_usize;
        let out_c = 2_usize;
        let kh = 3_usize;
        let kw = 3_usize;
        let in_shape = [in_c, in_h, in_w]; // (1, 4, 4)
        let in_flat_size = in_c * in_h * in_w; // 16

        let lower: Vec<f32> = bounds[..in_flat_size].to_vec();
        let upper: Vec<f32> = lower.iter()
            .zip(bounds[in_flat_size..].iter())
            .map(|(&l, &d)| l + d.abs())
            .collect();

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[out_c, in_c, kh, kw]),
            weights,
        ).unwrap();
        let bias = Array1::from_vec(vec![b0, b1]);

        let conv = Conv2dLayer::with_input_shape(
            kernel, Some(bias), (1, 1), (0, 0), in_h, in_w,
        ).unwrap();

        let out_h = in_h - kh + 1; // 2
        let out_w = in_w - kw + 1; // 2
        let conv_out_size = out_c * out_h * out_w; // 2 * 2 * 2 = 8

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

        // CROWN-vs-IBP equivalence
        for i in 0..conv_out_size {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv2d CROWN-IBP lower mismatch at {}: crown={}, ibp={}",
                i, crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv2d CROWN-IBP upper mismatch at {}: crown={}, ibp={}",
                i, crown_upper[i], ibp_upper[i]
            );
        }

        // Soundness: concrete outputs within bounds
        let samples = sample_multi_points(&lower, &upper, 10);
        for sample in &samples {
            let concrete_out = eval_conv2d(&conv, sample, &in_shape);
            for (i, &y) in concrete_out.iter().enumerate() {
                prop_assert!(
                    ibp_lower[i] - FP_TOLERANCE <= y && y <= ibp_upper[i] + FP_TOLERANCE,
                    "Conv2d soundness violation at output {}: y={}, bounds=[{}, {}]",
                    i, y, ibp_lower[i], ibp_upper[i]
                );
            }
        }
    }

    // =========================================================================
    // Conv2d CROWN with non-identity incoming bounds
    // =========================================================================

    /// Conv2d CROWN with non-identity incoming linear bounds.
    ///
    /// Tests CROWN backward composition with arbitrary coefficients through a
    /// 2-output-channel Conv2d. Verifies that transposed convolution correctly
    /// handles mixed-sign coefficients across spatial and channel dimensions.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_conv2d_crown_non_identity(
        weights in prop::collection::vec(-2.0f32..2.0, 18),
        b0 in -1.0f32..1.0,
        b1 in -1.0f32..1.0,
        coeffs in prop::collection::vec(-3.0f32..3.0, 8), // 8 = conv_out_size
        offset in -1.0f32..1.0,
        bounds in prop::collection::vec(-3.0f32..3.0, 32),
    ) {
        // At least one non-trivial coefficient
        prop_assume!(coeffs.iter().any(|c| c.abs() > 0.01));

        // Pin the shared dense budget while exercising the exact CROWN path;
        // other parallel tests intentionally set it to zero to test fallback.
        let _env_lock = ny_test_utils::env::lock_env();
        let _budget = ny_test_utils::env::ScopedEnvVar::set("NY_DENSE_BUDGET_MB", "2048");

        let in_h = 4_usize;
        let in_w = 4_usize;
        let in_c = 1_usize;
        let out_c = 2_usize;
        let in_shape = [in_c, in_h, in_w];
        let in_flat_size = in_c * in_h * in_w;

        let lower: Vec<f32> = bounds[..in_flat_size].to_vec();
        let upper: Vec<f32> = lower.iter()
            .zip(bounds[in_flat_size..].iter())
            .map(|(&l, &d)| l + d.abs())
            .collect();

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[out_c, in_c, 3, 3]),
            weights,
        ).unwrap();
        let bias = Array1::from_vec(vec![b0, b1]);

        let conv = Conv2dLayer::with_input_shape(
            kernel, Some(bias), (1, 1), (0, 0), in_h, in_w,
        ).unwrap();

        let out_h = in_h - 3 + 1;
        let out_w = in_w - 3 + 1;
        let conv_out_size = out_c * out_h * out_w; // 8

        // Non-identity incoming: 1 output = sum(c_i * y_i) + offset
        let incoming = LinearBounds::new(
            Array2::from_shape_vec((1, conv_out_size), coeffs.clone()).unwrap(),
            Array1::from_vec(vec![offset]),
            Array2::from_shape_vec((1, conv_out_size), coeffs.clone()).unwrap(),
            Array1::from_vec(vec![offset]),
        ).unwrap();

        let crown_result = conv.propagate_linear(&incoming).unwrap();

        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper.clone()).unwrap(),
        ).unwrap();
        let input_flat = input_bt.flatten();
        let crown_out = crown_result.concretize(&input_flat);
        let crown_lower = crown_out.lower()[[0]];
        let crown_upper = crown_out.upper()[[0]];

        // Soundness check
        let samples = sample_multi_points(&lower, &upper, 15);
        for sample in &samples {
            let concrete_conv = eval_conv2d(&conv, sample, &in_shape);
            let concrete_combined: f32 = coeffs.iter()
                .zip(concrete_conv.iter())
                .map(|(c, y)| c * y)
                .sum::<f32>() + offset;

            let tol = CROWN_CONV_TOLERANCE * concrete_combined.abs().max(1.0);
            prop_assert!(
                crown_lower - tol <= concrete_combined && concrete_combined <= crown_upper + tol,
                "Conv2d non-identity CROWN soundness violation: y={}, bounds=[{}, {}]",
                concrete_combined, crown_lower, crown_upper
            );
        }
    }

    // =========================================================================
    // Conv1d CROWN with padding
    // =========================================================================

    /// Conv1d CROWN soundness with padding=1.
    ///
    /// Padding introduces implicit zero-valued neighbors, which the transposed
    /// convolution must correctly handle. This test catches bugs where the
    /// padding offset is wrong in the CROWN backward.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_conv1d_crown_with_padding(
        k0 in -3.0f32..3.0,
        k1 in -3.0f32..3.0,
        k2 in -3.0f32..3.0,
        bias in -2.0f32..2.0,
        bounds in prop::collection::vec(-5.0f32..5.0, 8), // 4 lower + 4 delta
    ) {
        // Excluded from overlapping an env WRITER. The leak is specific and
        // known: `NY_DENSE_BUDGET_MB`, read process-globally by
        // `crown_memory::explicit_cpu_crown_dense_budget_bytes`. A concurrent
        // test setting it to 0 starves this one's CROWN into an IBP fallback,
        // which surfaces here as `crown=-inf` -- an enclosure violation that
        // is really a race. Observed failing at --test-threads=4 and =8.
        let _env = crate::tests::lock_env_shared();
        let input_len = 4;
        let in_shape = [1_usize, input_len];
        let lower: Vec<f32> = bounds[..4].to_vec();
        let upper: Vec<f32> = lower.iter().zip(bounds[4..].iter()).map(|(&l, &d)| l + d.abs()).collect();

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3]),
            vec![k0, k1, k2],
        ).unwrap();
        let bias_arr = Array1::from_vec(vec![bias]);

        let conv = Conv1dLayer::with_input_length(
            kernel, Some(bias_arr), 1, 1, input_len, // padding=1
        ).unwrap();

        // With padding=1, output_length = input_len + 2*1 - 3 + 1 = input_len = 4
        let out_len = input_len;
        let conv_out_size = out_len;

        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper.clone()).unwrap(),
        ).unwrap();

        let ibp_out = conv.propagate_ibp(&input_bt).unwrap();
        let ibp_lower: Vec<f32> = ibp_out.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_out.upper().iter().copied().collect();

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
                "Conv1d padded CROWN-IBP lower mismatch at {}: crown={}, ibp={}",
                i, crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv1d padded CROWN-IBP upper mismatch at {}: crown={}, ibp={}",
                i, crown_upper[i], ibp_upper[i]
            );
        }

        // Soundness
        let samples = sample_multi_points(&lower, &upper, 10);
        for sample in &samples {
            let concrete_out = eval_conv1d(&conv, sample, &in_shape);
            for (i, &y) in concrete_out.iter().enumerate() {
                prop_assert!(
                    ibp_lower[i] - FP_TOLERANCE <= y && y <= ibp_upper[i] + FP_TOLERANCE,
                    "Conv1d padded soundness violation at {}: y={}, bounds=[{}, {}]",
                    i, y, ibp_lower[i], ibp_upper[i]
                );
            }
        }
    }

    // =========================================================================
    // Conv2d CROWN with stride
    // =========================================================================

    /// Conv2d CROWN soundness with stride=2.
    ///
    /// Stride changes the spatial mapping between output and input positions.
    /// The transposed convolution must correctly space out the contributions.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_conv2d_crown_stride2(
        weights in prop::collection::vec(-2.0f32..2.0, 9), // 1 * 1 * 3 * 3 = 9
        bias in -1.0f32..1.0,
        bounds in prop::collection::vec(-3.0f32..3.0, 72), // 36 lower + 36 delta -> 6x6 input
    ) {
        // Keep deliberate zero-budget tests from changing this process-wide
        // prerequisite while the stride-specific CROWN path is under test.
        let _env_lock = ny_test_utils::env::lock_env();
        let _budget = ny_test_utils::env::ScopedEnvVar::set("NY_DENSE_BUDGET_MB", "2048");

        let in_h = 6_usize;
        let in_w = 6_usize;
        let in_c = 1_usize;
        let out_c = 1_usize;
        let in_shape = [in_c, in_h, in_w];
        let in_flat_size = in_c * in_h * in_w; // 36

        let lower: Vec<f32> = bounds[..in_flat_size].to_vec();
        let upper: Vec<f32> = lower.iter()
            .zip(bounds[in_flat_size..].iter())
            .map(|(&l, &d)| l + d.abs())
            .collect();

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[out_c, in_c, 3, 3]),
            weights,
        ).unwrap();
        let bias_arr = Array1::from_vec(vec![bias]);

        let conv = Conv2dLayer::with_input_shape(
            kernel, Some(bias_arr), (2, 2), (0, 0), in_h, in_w, // stride=2
        ).unwrap();

        // output_h = (6 - 3)/2 + 1 = 2, output_w = (6 - 3)/2 + 1 = 2
        let out_h = (in_h - 3) / 2 + 1; // 2
        let out_w = (in_w - 3) / 2 + 1; // 2
        let conv_out_size = out_c * out_h * out_w; // 4

        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper.clone()).unwrap(),
        ).unwrap();
        let ibp_out = conv.propagate_ibp(&input_bt).unwrap();
        let ibp_lower: Vec<f32> = ibp_out.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_out.upper().iter().copied().collect();

        let identity = LinearBounds::identity(conv_out_size);
        let crown_result = conv.propagate_linear(&identity).unwrap();

        let input_flat = input_bt.flatten();
        let crown_out = crown_result.concretize(&input_flat);
        let crown_lower: Vec<f32> = crown_out.lower().iter().copied().collect();
        let crown_upper: Vec<f32> = crown_out.upper().iter().copied().collect();

        for i in 0..conv_out_size {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv2d stride=2 CROWN-IBP lower mismatch at {}: crown={}, ibp={}",
                i, crown_lower[i], ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() <= CROWN_CONV_TOLERANCE,
                "Conv2d stride=2 CROWN-IBP upper mismatch at {}: crown={}, ibp={}",
                i, crown_upper[i], ibp_upper[i]
            );
        }

        let samples = sample_multi_points(&lower, &upper, 10);
        for sample in &samples {
            let concrete_out = eval_conv2d(&conv, sample, &in_shape);
            for (i, &y) in concrete_out.iter().enumerate() {
                prop_assert!(
                    ibp_lower[i] - FP_TOLERANCE <= y && y <= ibp_upper[i] + FP_TOLERANCE,
                    "Conv2d stride=2 soundness violation at {}: y={}, bounds=[{}, {}]",
                    i, y, ibp_lower[i], ibp_upper[i]
                );
            }
        }
    }

    // =========================================================================
    // ConvTranspose CROWN soundness
    // =========================================================================

    /// ConvTranspose1d IBP + CROWN soundness with random kernel.
    ///
    /// This verifies the dedicated ConvTranspose1d CROWN backward path
    /// (`propagate_linear` uses regular convolution of incoming A with the
    /// transpose kernel and adds broadcast bias contribution).
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_conv_transpose1d_crown_1c_k3(
        k0 in -3.0f32..3.0,
        k1 in -3.0f32..3.0,
        k2 in -3.0f32..3.0,
        bias in -2.0f32..2.0,
        bounds in prop::collection::vec(-4.0f32..4.0, 8), // 4 lower + 4 delta
    ) {
        // Excluded from overlapping an env WRITER. The leak is specific and
        // known: `NY_DENSE_BUDGET_MB`, read process-globally by
        // `crown_memory::explicit_cpu_crown_dense_budget_bytes`. A concurrent
        // test setting it to 0 starves this one's CROWN into an IBP fallback,
        // which surfaces here as `crown=-inf` -- an enclosure violation that
        // is really a race. Observed failing at --test-threads=4 and =8.
        let _env = crate::tests::lock_env_shared();
        let input_len = 4_usize;
        let in_shape = [1_usize, input_len];

        let lower: Vec<f32> = bounds[..4].to_vec();
        let upper: Vec<f32> = lower
            .iter()
            .zip(bounds[4..].iter())
            .map(|(&l, &d)| l + d.abs())
            .collect();

        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3]), // (in_channels=1, out_channels=1, kernel=3)
            vec![k0, k1, k2],
        )
        .unwrap();
        let bias_arr = Array1::from_vec(vec![bias]);

        let conv_t =
            ConvTranspose1dLayer::with_input_length(kernel, Some(bias_arr), 1, 0, input_len)
                .unwrap();

        let out_len = (input_len - 1) + 3; // stride=1, padding=0
        let conv_out_size = out_len;

        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper.clone()).unwrap(),
        )
        .unwrap();
        let ibp_out = conv_t.propagate_ibp(&input_bt).unwrap();
        let ibp_lower: Vec<f32> = ibp_out.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_out.upper().iter().copied().collect();

        let identity = LinearBounds::identity(conv_out_size);
        let crown_result = conv_t.propagate_linear(&identity).unwrap();
        let input_flat = input_bt.flatten();
        let crown_out = crown_result.concretize(&input_flat);
        let crown_lower: Vec<f32> = crown_out.lower().iter().copied().collect();
        let crown_upper: Vec<f32> = crown_out.upper().iter().copied().collect();

        for i in 0..conv_out_size {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() <= CROWN_CONV_TOLERANCE,
                "ConvTranspose1d CROWN-IBP lower mismatch at {}: crown={}, ibp={}",
                i,
                crown_lower[i],
                ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() <= CROWN_CONV_TOLERANCE,
                "ConvTranspose1d CROWN-IBP upper mismatch at {}: crown={}, ibp={}",
                i,
                crown_upper[i],
                ibp_upper[i]
            );
        }

        let samples = sample_multi_points(&lower, &upper, 8);
        for sample in &samples {
            let concrete_out = eval_conv_transpose1d(&conv_t, sample, &in_shape);
            for (i, &y) in concrete_out.iter().enumerate() {
                prop_assert!(
                    ibp_lower[i] - FP_TOLERANCE <= y && y <= ibp_upper[i] + FP_TOLERANCE,
                    "ConvTranspose1d soundness violation at output {}: y={}, bounds=[{}, {}]",
                    i,
                    y,
                    ibp_lower[i],
                    ibp_upper[i]
                );
            }
        }
    }

    /// ConvTranspose2d IBP + CROWN soundness with random kernel.
    ///
    /// This verifies non-identity coefficient mapping in ConvTranspose2d CROWN
    /// backward instead of treating ConvTranspose as a structural identity op.
    #[ntest::timeout(300000)]
    #[test]
    fn soundness_conv_transpose2d_crown_1c_k3(
        weights in prop::collection::vec(-2.0f32..2.0, 9), // 1 * 1 * 3 * 3
        bias in -1.0f32..1.0,
        bounds in prop::collection::vec(-3.0f32..3.0, 8), // 4 lower + 4 delta -> 2x2 input
    ) {
        // Excluded from overlapping an env WRITER. The leak is specific and
        // known: `NY_DENSE_BUDGET_MB`, read process-globally by
        // `crown_memory::explicit_cpu_crown_dense_budget_bytes`. A concurrent
        // test setting it to 0 starves this one's CROWN into an IBP fallback,
        // which surfaces here as `crown=-inf` -- an enclosure violation that
        // is really a race. Observed failing at --test-threads=4 and =8.
        let _env = crate::tests::lock_env_shared();
        let in_h = 2_usize;
        let in_w = 2_usize;
        let in_shape = [1_usize, in_h, in_w];

        let lower: Vec<f32> = bounds[..4].to_vec();
        let upper: Vec<f32> = lower
            .iter()
            .zip(bounds[4..].iter())
            .map(|(&l, &d)| l + d.abs())
            .collect();

        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), weights).unwrap();
        let bias_arr = Array1::from_vec(vec![bias]);

        let conv_t = ConvTranspose2dLayer::with_input_shape(
            kernel,
            Some(bias_arr),
            (1, 1),
            (0, 0),
            in_h,
            in_w,
        )
        .unwrap();

        let out_h = (in_h - 1) + 3; // stride=1, padding=0
        let out_w = (in_w - 1) + 3;
        let conv_out_size = out_h * out_w;

        let input_bt = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&in_shape), lower.clone()).unwrap(),
            ArrayD::from_shape_vec(IxDyn(&in_shape), upper.clone()).unwrap(),
        )
        .unwrap();
        let ibp_out = conv_t.propagate_ibp(&input_bt).unwrap();
        let ibp_lower: Vec<f32> = ibp_out.lower().iter().copied().collect();
        let ibp_upper: Vec<f32> = ibp_out.upper().iter().copied().collect();

        let identity = LinearBounds::identity(conv_out_size);
        let crown_result = conv_t.propagate_linear(&identity).unwrap();
        let input_flat = input_bt.flatten();
        let crown_out = crown_result.concretize(&input_flat);
        let crown_lower: Vec<f32> = crown_out.lower().iter().copied().collect();
        let crown_upper: Vec<f32> = crown_out.upper().iter().copied().collect();

        for i in 0..conv_out_size {
            prop_assert!(
                (crown_lower[i] - ibp_lower[i]).abs() <= CROWN_CONV_TOLERANCE,
                "ConvTranspose2d CROWN-IBP lower mismatch at {}: crown={}, ibp={}",
                i,
                crown_lower[i],
                ibp_lower[i]
            );
            prop_assert!(
                (crown_upper[i] - ibp_upper[i]).abs() <= CROWN_CONV_TOLERANCE,
                "ConvTranspose2d CROWN-IBP upper mismatch at {}: crown={}, ibp={}",
                i,
                crown_upper[i],
                ibp_upper[i]
            );
        }

        let samples = sample_multi_points(&lower, &upper, 8);
        for sample in &samples {
            let concrete_out = eval_conv_transpose2d(&conv_t, sample, &in_shape);
            for (i, &y) in concrete_out.iter().enumerate() {
                prop_assert!(
                    ibp_lower[i] - FP_TOLERANCE <= y && y <= ibp_upper[i] + FP_TOLERANCE,
                    "ConvTranspose2d soundness violation at output {}: y={}, bounds=[{}, {}]",
                    i,
                    y,
                    ibp_lower[i],
                    ibp_upper[i]
                );
            }
        }
    }
}
