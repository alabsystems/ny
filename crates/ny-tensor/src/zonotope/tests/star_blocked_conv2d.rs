// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Differential and resource gates for the unwired block-generator convolution.

use super::super::*;
use ndarray::{Array1, Array2, Array4, ArrayD, Axis, IxDyn};
use ny_core::NyError;
use proptest::prelude::*;

fn unrestricted(block_rows: usize) -> StarConv2dBlockLimits {
    StarConv2dBlockLimits {
        block_rows,
        max_workspace_bytes: usize::MAX,
        max_return_bytes: usize::MAX,
        max_peak_owned_bytes: usize::MAX,
        max_multiply_accumulates: usize::MAX,
    }
}

fn patterned_values(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let centered = ((index.wrapping_mul(37).wrapping_add(salt * 19)) % 127) as i32 - 63;
            centered as f32 / 41.0
        })
        .collect()
}

fn patterned_star(rows: usize, shape: [usize; 3], predicate_rows: usize) -> Star {
    assert!(rows >= 1);
    let element_count = shape.iter().product::<usize>();
    let coeffs = ArrayD::from_shape_vec(
        IxDyn(&[rows, shape[0], shape[1], shape[2]]),
        patterned_values(rows * element_count, 3),
    )
    .unwrap();
    let zono = ZonotopeTensor::new(coeffs).unwrap();
    let alpha_dim = rows - 1;
    let a = Array2::from_shape_vec(
        (predicate_rows, alpha_dim),
        patterned_values(predicate_rows * alpha_dim, 5),
    )
    .unwrap();
    let b = Array1::from_vec(patterned_values(predicate_rows, 7));
    Star::new(zono, a, b).unwrap()
}

fn patterned_weight(shape: [usize; 4]) -> Array4<f32> {
    let len = shape.iter().product();
    Array4::from_shape_vec(
        (shape[0], shape[1], shape[2], shape[3]),
        patterned_values(len, 11),
    )
    .unwrap()
}

fn metaroom_stages() -> [([usize; 3], [usize; 4], (usize, usize), (usize, usize)); 4] {
    [
        ([3, 32, 56], [32, 3, 3, 3], (1, 1), (1, 1)),
        ([32, 32, 56], [32, 32, 3, 3], (1, 1), (1, 1)),
        ([32, 32, 56], [64, 32, 3, 3], (2, 2), (1, 1)),
        ([64, 16, 28], [64, 64, 3, 3], (1, 1), (1, 1)),
    ]
}

fn assert_scalar_close(scalar: &Star, blocked: &Star, context: &str) {
    assert_eq!(scalar.shape(), blocked.shape(), "{context}: value shape");
    assert_eq!(
        scalar.alpha_dim(),
        blocked.alpha_dim(),
        "{context}: alpha dim"
    );
    assert_eq!(
        scalar.constraints(),
        blocked.constraints(),
        "{context}: predicate must be copied exactly"
    );
    let scalar_coeffs = scalar.zonotope().coeffs();
    let blocked_coeffs = blocked.zonotope().coeffs();
    assert_eq!(scalar_coeffs.shape(), blocked_coeffs.shape());
    for (index, (&expected, &actual)) in scalar_coeffs.iter().zip(blocked_coeffs.iter()).enumerate()
    {
        // This is a differential regression tolerance, not a soundness budget.
        // The public API explicitly refuses to treat it as an enclosure.
        let tolerance = 2.0e-4_f32 * (1.0 + expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: coeff[{index}] scalar={expected} blocked={actual} delta={} tolerance={tolerance}",
            (actual - expected).abs()
        );
    }
}

/// Independent scalar f32 contraction. This is deliberately not ndarray GEMM:
/// `Star::conv2d` can reject valid tiny matrices when GEMM returns a non-standard
/// layout that its direct reshape cannot consume. The explicit loop remains a
/// stable mathematical oracle for the block kernel across every valid geometry.
fn scalar_loop_oracle(
    star: &Star,
    weight: &Array4<f32>,
    bias: Option<&Array1<f32>>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Star {
    let input_shape = star.shape();
    let (input_channels, input_height, input_width) =
        (input_shape[0], input_shape[1], input_shape[2]);
    let (output_channels, kernel_height, kernel_width) =
        (weight.shape()[0], weight.shape()[2], weight.shape()[3]);
    let output_height = (input_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let output_width = (input_width + 2 * padding.1 - kernel_width) / stride.1 + 1;
    let rows = star.alpha_dim() + 1;
    let input = star
        .zonotope()
        .coeffs()
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .unwrap();
    let mut output = Array4::<f32>::zeros((rows, output_channels, output_height, output_width));
    for row in 0..rows {
        for output_channel in 0..output_channels {
            for output_row in 0..output_height {
                for output_column in 0..output_width {
                    let mut accumulator = 0.0_f32;
                    for input_channel in 0..input_channels {
                        for kernel_row in 0..kernel_height {
                            for kernel_column in 0..kernel_width {
                                let padded_row = output_row * stride.0 + kernel_row;
                                let padded_column = output_column * stride.1 + kernel_column;
                                if padded_row >= padding.0 && padded_column >= padding.1 {
                                    let input_row = padded_row - padding.0;
                                    let input_column = padded_column - padding.1;
                                    if input_row < input_height && input_column < input_width {
                                        accumulator += weight[[
                                            output_channel,
                                            input_channel,
                                            kernel_row,
                                            kernel_column,
                                        ]] * input
                                            [[row, input_channel, input_row, input_column]];
                                    }
                                }
                            }
                        }
                    }
                    if row == 0 {
                        if let Some(bias) = bias {
                            accumulator += bias[output_channel];
                        }
                    }
                    output[[row, output_channel, output_row, output_column]] = accumulator;
                }
            }
        }
    }
    let (a, b) = star.constraints();
    Star::new(
        ZonotopeTensor::new(output.into_dyn()).unwrap(),
        a.clone(),
        b.clone(),
    )
    .unwrap()
}

#[test]
fn blocked_matches_scalar_multichannel_padding_stride_and_bias() {
    let star = patterned_star(7, [2, 5, 6], 3);
    let weight = patterned_weight([3, 2, 3, 2]);
    let bias = Array1::from_vec(patterned_values(3, 13));
    let scalar = star.conv2d(&weight, Some(&bias), (1, 2), (1, 0)).unwrap();
    let blocked = star
        .conv2d_blocked_unwired(&weight, Some(&bias), (1, 2), (1, 0), unrestricted(3))
        .unwrap();
    assert_scalar_close(&scalar, &blocked, "stride(1,2)/pad(1,0)/bias");
}

#[test]
fn blocked_matches_scalar_stride_two_padding_no_bias() {
    let star = patterned_star(6, [1, 6, 5], 0);
    let weight = patterned_weight([2, 1, 2, 3]);
    let scalar = star.conv2d(&weight, None, (2, 1), (0, 1)).unwrap();
    let blocked = star
        .conv2d_blocked_unwired(&weight, None, (2, 1), (0, 1), unrestricted(4))
        .unwrap();
    assert_scalar_close(&scalar, &blocked, "stride(2,1)/pad(0,1)/no-bias");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    #[test]
    fn blocked_matches_scalar_across_small_geometries(
        rows in 1_usize..8,
        input_channels in 1_usize..4,
        input_height in 2_usize..8,
        input_width in 2_usize..8,
        output_channels in 1_usize..5,
        kernel_height in 1_usize..4,
        kernel_width in 1_usize..4,
        stride_height in 1_usize..4,
        stride_width in 1_usize..4,
        pad_height in 0_usize..3,
        pad_width in 0_usize..3,
        requested_block_rows in 1_usize..7,
        use_bias in any::<bool>(),
    ) {
        prop_assume!(input_height + 2 * pad_height >= kernel_height);
        prop_assume!(input_width + 2 * pad_width >= kernel_width);
        let predicate_rows = usize::from(rows > 1);
        let star = patterned_star(
            rows,
            [input_channels, input_height, input_width],
            predicate_rows,
        );
        let weight = patterned_weight([
            output_channels,
            input_channels,
            kernel_height,
            kernel_width,
        ]);
        let bias = use_bias.then(|| Array1::from_vec(patterned_values(output_channels, 23)));
        let scalar = scalar_loop_oracle(
            &star,
            &weight,
            bias.as_ref(),
            (stride_height, stride_width),
            (pad_height, pad_width),
        );
        let blocked = star
            .conv2d_blocked_unwired(
                &weight,
                bias.as_ref(),
                (stride_height, stride_width),
                (pad_height, pad_width),
                unrestricted(requested_block_rows),
            )
            .unwrap();
        assert_scalar_close(&scalar, &blocked, "small-geometry proptest");
    }
}

#[test]
fn blocked_zero_symbol_star_and_bias_center_only() {
    let center = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), patterned_values(24, 17)).unwrap();
    let star = Star::from_zonotope(ZonotopeTensor::concrete(center));
    let weight = patterned_weight([2, 2, 1, 1]);
    let bias = Array1::from_vec(vec![0.25, -0.5]);
    let plan = star
        .plan_conv2d_blocked_unwired(&weight, (2, 2), (0, 0), unrestricted(16))
        .unwrap();
    assert_eq!(plan.rows, 1, "one center and zero generators is valid");
    assert_eq!(plan.gemm_calls, 1);
    let scalar = star.conv2d(&weight, Some(&bias), (2, 2), (0, 0)).unwrap();
    let blocked = star
        .conv2d_blocked_unwired(&weight, Some(&bias), (2, 2), (0, 0), unrestricted(16))
        .unwrap();
    assert_eq!(blocked.alpha_dim(), 0);
    assert_scalar_close(&scalar, &blocked, "zero-symbol concrete star");

    let without_bias = star
        .conv2d_blocked_unwired(&weight, None, (2, 2), (0, 0), unrestricted(16))
        .unwrap();
    let with_coeffs = blocked.zonotope().coeffs().index_axis(Axis(0), 0);
    let without_coeffs = without_bias.zonotope().coeffs().index_axis(Axis(0), 0);
    for channel in 0..2 {
        for row in 0..with_coeffs.shape()[1] {
            for column in 0..with_coeffs.shape()[2] {
                assert_eq!(
                    with_coeffs[[channel, row, column]],
                    without_coeffs[[channel, row, column]] + bias[channel]
                );
            }
        }
    }
}

#[test]
fn malformed_zero_coefficient_rows_fail_closed_without_panicking() {
    // This malformed value is constructible through the historical
    // `ZonotopeTensor::new` contract: the leading zero saturates alpha_dim to zero,
    // but there is no center row to execute. The blocked planner must inspect the
    // backing shape instead of inventing `alpha_dim + 1 == 1` row.
    let zono = ZonotopeTensor::new(ArrayD::zeros(IxDyn(&[0, 1, 2, 2]))).unwrap();
    let star = Star::new(zono, Array2::zeros((0, 0)), Array1::zeros(0)).unwrap();
    let weight = Array4::ones((1, 1, 1, 1));

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        star.conv2d_blocked_unwired(&weight, None, (1, 1), (0, 0), unrestricted(1))
    }));
    assert!(outcome.is_ok(), "malformed backing reached a panic");
    assert!(matches!(
        outcome.unwrap(),
        Err(NyError::InvalidSpec(message)) if message.contains("at least one center row")
    ));
}

#[test]
fn bias_does_not_touch_generator_rows() {
    let star = patterned_star(5, [2, 4, 4], 1);
    let weight = patterned_weight([3, 2, 3, 3]);
    let bias = Array1::from_vec(vec![0.75, -0.25, 0.125]);
    let with_bias = star
        .conv2d_blocked_unwired(&weight, Some(&bias), (1, 1), (1, 1), unrestricted(2))
        .unwrap();
    let without_bias = star
        .conv2d_blocked_unwired(&weight, None, (1, 1), (1, 1), unrestricted(2))
        .unwrap();
    for row in 1..5 {
        assert_eq!(
            with_bias.zonotope().coeffs().index_axis(Axis(0), row),
            without_bias.zonotope().coeffs().index_axis(Axis(0), row),
            "generator row {row} changed under center bias"
        );
    }
}

#[test]
fn block_plan_reduces_gemm_calls_without_changing_operations() {
    let scalar = StarConv2dBlockPlan::estimate(
        5_118,
        0,
        [32, 32, 56],
        [32, 32, 3, 3],
        (1, 1),
        (1, 1),
        unrestricted(1),
    )
    .unwrap();
    let blocked = StarConv2dBlockPlan::estimate(
        5_118,
        0,
        [32, 32, 56],
        [32, 32, 3, 3],
        (1, 1),
        (1, 1),
        unrestricted(16),
    )
    .unwrap();
    assert_eq!(scalar.gemm_calls, 5_118);
    assert_eq!(blocked.gemm_calls, 320);
    assert_eq!(scalar.multiply_accumulates, blocked.multiply_accumulates);
    assert_eq!(
        scalar.coefficient_output_bytes,
        blocked.coefficient_output_bytes
    );
    assert!(blocked.workspace_bytes > scalar.workspace_bytes);
}

#[test]
fn metaroom_uniform_5118_row_projection_is_conditional_on_sparse_input() {
    // Metaroom 6cnn_ry_39_6: input 3x32x56; Conv widths 32,32,64,64;
    // kernels 3x3; the third Conv has stride two. The 5_117-symbol ceiling is a
    // deliberately conservative projection that is conditional on replacing the
    // current one-symbol-per-input-element constructor with the future 161-symbol
    // sparse constructor. It is not the current from_input_box execution model.
    let plans: Vec<_> = metaroom_stages()
        .into_iter()
        .map(|(input, weight, stride, padding)| {
            StarConv2dBlockPlan::estimate(
                5_118,
                0,
                input,
                weight,
                stride,
                padding,
                unrestricted(16),
            )
            .unwrap()
        })
        .collect();

    assert_eq!(
        plans.iter().map(|plan| plan.gemm_calls).collect::<Vec<_>>(),
        vec![320; 4]
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.coefficient_output_bytes)
            .collect::<Vec<_>>(),
        vec![1_173_946_368, 1_173_946_368, 586_973_184, 586_973_184]
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.workspace_bytes)
            .collect::<Vec<_>>(),
        vec![6_770_048, 36_737_024, 10_166_272, 18_497_536]
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.multiply_accumulates)
            .collect::<Vec<_>>(),
        vec![
            7_924_137_984,
            84_524_138_496,
            42_262_069_248,
            84_524_138_496
        ]
    );
}

#[test]
fn metaroom_future_sparse_staged_resource_model_is_pinned() {
    // Future hypothesis only: 161 initial sparse symbols followed by the observed
    // unstable-ReLU additions. Two convex-hull predicate rows are added per
    // unstable ReLU, hence cumulative predicate row counts 0/954/3248/6354.
    let staged_rows = [162, 639, 1_786, 3_339];
    let predicate_rows = [0, 954, 3_248, 6_354];
    let plans: Vec<_> = metaroom_stages()
        .into_iter()
        .zip(staged_rows)
        .zip(predicate_rows)
        .map(|(((input, weight, stride, padding), rows), constraints)| {
            StarConv2dBlockPlan::estimate(
                rows,
                constraints,
                input,
                weight,
                stride,
                padding,
                unrestricted(16),
            )
            .unwrap()
        })
        .collect();

    assert_eq!(plans.iter().map(|plan| plan.gemm_calls).sum::<usize>(), 372);
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.multiply_accumulates)
            .sum::<usize>(),
        80_695_738_368
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.predicate_clone_bytes)
            .collect::<Vec<_>>(),
        vec![0, 2_438_424, 23_203_712, 84_864_024]
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.return_bytes)
            .collect::<Vec<_>>(),
        vec![37_158_912, 149_009_688, 228_036_480, 467_807_256]
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.peak_owned_bytes)
            .collect::<Vec<_>>(),
        vec![43_928_960, 183_308_288, 228_036_480, 467_807_256]
    );
}

#[test]
fn metaroom_current_dense_input_box_resource_model_is_pinned() {
    // Today's Star::from_input_box creates one symbol for every one of the 5_376
    // inputs. With the same observed unstable-ReLU additions, the actual staged
    // row counts are much larger than the future sparse hypothesis.
    let staged_rows = [5_377, 5_854, 7_001, 8_554];
    let predicate_rows = [0, 954, 3_248, 6_354];
    let plans: Vec<_> = metaroom_stages()
        .into_iter()
        .zip(staged_rows)
        .zip(predicate_rows)
        .map(|(((input, weight, stride, padding), rows), constraints)| {
            StarConv2dBlockPlan::estimate(
                rows,
                constraints,
                input,
                weight,
                stride,
                padding,
                unrestricted(16),
            )
            .unwrap()
        })
        .collect();

    assert_eq!(
        plans.iter().map(|plan| plan.gemm_calls).sum::<usize>(),
        1_676
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.multiply_accumulates)
            .sum::<usize>(),
        304_085_311_488
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.predicate_clone_bytes)
            .collect::<Vec<_>>(),
        vec![0, 22_338_864, 90_956_992, 217_408_464]
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.return_bytes)
            .collect::<Vec<_>>(),
        vec![1_233_354_752, 1_365_105_968, 893_887_680, 1_198_449_616]
    );
    assert_eq!(
        plans
            .iter()
            .map(|plan| plan.peak_owned_bytes)
            .collect::<Vec<_>>(),
        vec![1_240_124_800, 1_379_504_128, 893_887_680, 1_198_449_616]
    );
}

#[test]
fn every_resource_limit_fails_closed_before_execution() {
    let base = StarConv2dBlockPlan::estimate(
        9,
        2,
        [2, 5, 5],
        [3, 2, 3, 3],
        (1, 1),
        (1, 1),
        unrestricted(4),
    )
    .unwrap();

    let mut limits = unrestricted(4);
    limits.max_return_bytes = base.return_bytes - 1;
    assert!(matches!(
        StarConv2dBlockPlan::estimate(9, 2, [2, 5, 5], [3, 2, 3, 3], (1, 1), (1, 1), limits),
        Err(NyError::CpuMemoryExceeded { .. })
    ));

    limits = unrestricted(4);
    limits.max_workspace_bytes = base.workspace_bytes - 1;
    assert!(matches!(
        StarConv2dBlockPlan::estimate(9, 2, [2, 5, 5], [3, 2, 3, 3], (1, 1), (1, 1), limits),
        Err(NyError::CpuMemoryExceeded { .. })
    ));

    limits = unrestricted(4);
    limits.max_peak_owned_bytes = base.peak_owned_bytes - 1;
    assert!(matches!(
        StarConv2dBlockPlan::estimate(9, 2, [2, 5, 5], [3, 2, 3, 3], (1, 1), (1, 1), limits),
        Err(NyError::CpuMemoryExceeded { .. })
    ));

    limits = unrestricted(4);
    limits.max_multiply_accumulates = base.multiply_accumulates - 1;
    assert!(matches!(
        StarConv2dBlockPlan::estimate(9, 2, [2, 5, 5], [3, 2, 3, 3], (1, 1), (1, 1), limits),
        Err(NyError::InvalidConfig(_))
    ));
}

#[test]
fn invalid_geometry_overflow_and_nonfinite_values_fail_closed() {
    assert!(StarConv2dBlockPlan::estimate(
        2,
        0,
        [1, 3, 3],
        [1, 1, 1, 1],
        (1, 1),
        (0, 0),
        unrestricted(0)
    )
    .is_err());
    assert!(StarConv2dBlockPlan::estimate(
        2,
        0,
        [1, usize::MAX, 2],
        [1, 1, 1, 1],
        (1, 1),
        (1, 0),
        unrestricted(1)
    )
    .is_err());
    assert!(StarConv2dBlockPlan::estimate(
        usize::MAX,
        usize::MAX,
        [usize::MAX, 1, 1],
        [1, usize::MAX, 1, 1],
        (1, 1),
        (0, 0),
        unrestricted(1)
    )
    .is_err());

    let star = patterned_star(2, [1, 2, 2], 0);
    let mut weight = patterned_weight([1, 1, 1, 1]);
    weight[[0, 0, 0, 0]] = f32::NAN;
    assert!(matches!(
        star.conv2d_blocked_unwired(&weight, None, (1, 1), (0, 0), unrestricted(1)),
        Err(NyError::NumericalInstability(_))
    ));

    let star = patterned_star(2, [1, 2, 2], 0);
    let weight = Array4::from_elem((1, 1, 1, 1), f32::MAX);
    assert!(matches!(
        star.conv2d_blocked_unwired(&weight, None, (1, 1), (0, 0), unrestricted(1)),
        Err(NyError::NumericalInstability(_))
    ));
}
