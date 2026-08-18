// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NaiveCpuGemmEngine, NyError, Result};
use ny_tensor::{next_down_f32, next_up_f32, BoundedTensor};
use ny_test_utils::{assert_bounded_tensor_close, CountingGemmEngine};

use super::*;
use crate::layers::common::BoundPropagation;
use crate::tests::{assert_batched_bounds_close, assert_close};
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{array, Array1, Array2, ArrayD, IxDyn};
use proptest::prelude::*;

const TOL: f32 = 1e-6;
const JACOBIAN_TOL: f32 = 1e-5;

fn make_conv1d(weight: f32, bias: Option<f32>, input_length: usize) -> Conv1dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![weight]).expect("kernel");
    let bias = bias.map(|b| array![b]);
    Conv1dLayer::with_input_length(kernel, bias, 1, 0, input_length).expect("valid conv1d")
}

fn make_convtranspose1d(
    weight: f32,
    bias: Option<f32>,
    input_length: usize,
) -> ConvTranspose1dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![weight]).expect("kernel");
    let bias = bias.map(|b| array![b]);
    ConvTranspose1dLayer::with_input_length(kernel, bias, 1, 0, input_length)
        .expect("valid convtranspose1d")
}

fn make_convtranspose1d_full(
    kernel: ArrayD<f32>,
    bias: Option<Array1<f32>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    input_length: usize,
) -> ConvTranspose1dLayer {
    ConvTranspose1dLayer::with_input_length_full(
        kernel,
        bias,
        stride,
        padding,
        dilation,
        groups,
        input_length,
    )
    .expect("valid convtranspose1d")
}

#[derive(Debug, Clone)]
struct ConvTranspose1dJacobianCase {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    input_length: usize,
    num_objectives: usize,
    output_length: usize,
    kernel: Vec<f32>,
    bias: Vec<f32>,
    incoming_a: Vec<f32>,
    incoming_b: Vec<f32>,
}

impl ConvTranspose1dJacobianCase {
    fn output_dim(&self) -> usize {
        self.out_channels * self.output_length
    }
}

fn convtranspose1d_output_len(
    input_length: usize,
    stride: usize,
    kernel_size: usize,
    padding: usize,
) -> Option<usize> {
    let expanded = input_length
        .checked_sub(1)?
        .checked_mul(stride)?
        .checked_add(kernel_size)?;
    let double_padding = padding.checked_mul(2)?;
    if expanded > double_padding {
        Some(expanded - double_padding)
    } else {
        None
    }
}

fn convtranspose1d_jacobian_case() -> impl Strategy<Value = ConvTranspose1dJacobianCase> {
    (
        1usize..=2,
        1usize..=2,
        1usize..=3,
        1usize..=3,
        0usize..=2,
        1usize..=4,
        1usize..=3,
    )
        .prop_filter(
            "ConvTranspose1d output length must stay positive",
            |(_, _, kernel_size, stride, padding, input_length, _)| {
                convtranspose1d_output_len(*input_length, *stride, *kernel_size, *padding).is_some()
            },
        )
        .prop_flat_map(
            |(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                input_length,
                num_objectives,
            )| {
                let output_length =
                    convtranspose1d_output_len(input_length, stride, kernel_size, padding)
                        .expect("filtered above");
                let output_dim = out_channels * output_length;
                (
                    Just((
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        input_length,
                        num_objectives,
                        output_length,
                    )),
                    prop::collection::vec(-2.0f32..2.0, in_channels * out_channels * kernel_size),
                    prop::collection::vec(-1.0f32..1.0, out_channels),
                    prop::collection::vec(-1.5f32..1.5, num_objectives * output_dim),
                    prop::collection::vec(-0.5f32..0.5, num_objectives),
                )
                    .prop_map(
                        move |(
                            (
                                in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                input_length,
                                num_objectives,
                                output_length,
                            ),
                            kernel,
                            bias,
                            incoming_a,
                            incoming_b,
                        )| ConvTranspose1dJacobianCase {
                            in_channels,
                            out_channels,
                            kernel_size,
                            stride,
                            padding,
                            input_length,
                            num_objectives,
                            output_length,
                            kernel,
                            bias,
                            incoming_a,
                            incoming_b,
                        },
                    )
            },
        )
}

fn explicit_convtranspose1d_jacobian(layer: &ConvTranspose1dLayer) -> Result<Array2<f32>> {
    let input_length = layer.input_length.ok_or_else(|| {
        NyError::UnsupportedConfiguration(
            "explicit ConvTranspose1d Jacobian requires input_length".to_string(),
        )
    })?;
    let input_channels = layer.in_channels();
    let output_channels = layer.out_channels();
    let output_length = layer.output_length(input_length)?;
    let input_dim = input_channels * input_length;
    let output_dim = output_channels * output_length;
    let mut jacobian = Array2::zeros((output_dim, input_dim));
    let groups = layer.groups;
    let in_c_per_group = input_channels / groups;
    let out_c_per_group = output_channels / groups;

    for group in 0..groups {
        let ic_start = group * in_c_per_group;
        let oc_start = group * out_c_per_group;
        for input_local in 0..in_c_per_group {
            let input_channel = ic_start + input_local;
            for input_pos in 0..input_length {
                let input_index = input_channel * input_length + input_pos;
                for kernel_pos in 0..layer.kernel_size() {
                    let output_pos = (input_pos * layer.stride + kernel_pos * layer.dilation)
                        as isize
                        - layer.padding as isize;
                    if !(0..output_length as isize).contains(&output_pos) {
                        continue;
                    }
                    for output_local in 0..out_c_per_group {
                        let output_channel = oc_start + output_local;
                        let output_index = output_channel * output_length + output_pos as usize;
                        jacobian[[output_index, input_index]] +=
                            layer.kernel[[input_channel, output_local, kernel_pos]];
                    }
                }
            }
        }
    }

    Ok(jacobian)
}

fn explicit_convtranspose1d_bias(layer: &ConvTranspose1dLayer) -> Result<Array1<f32>> {
    let input_length = layer.input_length.ok_or_else(|| {
        NyError::UnsupportedConfiguration(
            "explicit ConvTranspose1d bias requires input_length".to_string(),
        )
    })?;
    let output_length = layer.output_length(input_length)?;
    let output_channels = layer.out_channels();
    let bias = layer
        .bias
        .clone()
        .unwrap_or_else(|| Array1::zeros(output_channels));
    let mut flattened = Array1::zeros(output_channels * output_length);
    for output_channel in 0..output_channels {
        for output_pos in 0..output_length {
            flattened[output_channel * output_length + output_pos] = bias[output_channel];
        }
    }
    Ok(flattened)
}

fn explicit_convtranspose1d_bias_bounds(
    incoming_a: &Array2<f32>,
    incoming_b: &Array1<f32>,
    flattened_bias: &Array1<f32>,
) -> (Array1<f32>, Array1<f32>) {
    let mut lower = Array1::zeros(incoming_b.len());
    let mut upper = Array1::zeros(incoming_b.len());

    for row in 0..incoming_a.nrows() {
        let mut total = incoming_b[row] as f64;
        for col in 0..incoming_a.ncols() {
            total += (incoming_a[[row, col]] as f64) * (flattened_bias[col] as f64);
        }
        let total = total as f32;
        lower[row] = next_down_f32(total);
        upper[row] = next_up_f32(total);
    }

    (lower, upper)
}

fn assert_convtranspose1d_backward_matches_explicit(
    layer: &ConvTranspose1dLayer,
    bounds: &LinearBounds,
) -> Result<()> {
    let actual = layer
        .propagate_linear_with_engine(bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();
    let explicit_jacobian = explicit_convtranspose1d_jacobian(layer)?;
    let explicit_bias = explicit_convtranspose1d_bias(layer)?;
    let expected_lower_a = bounds.lower_a().dot(&explicit_jacobian);
    let expected_upper_a = bounds.upper_a().dot(&explicit_jacobian);
    let (expected_lower_b, _) =
        explicit_convtranspose1d_bias_bounds(bounds.lower_a(), bounds.lower_b(), &explicit_bias);
    let (_, expected_upper_b) =
        explicit_convtranspose1d_bias_bounds(bounds.upper_a(), bounds.upper_b(), &explicit_bias);

    for (idx, (&actual_value, &expected_value)) in actual
        .lower_a()
        .iter()
        .zip(expected_lower_a.iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= JACOBIAN_TOL,
            "lower_a mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .upper_a()
        .iter()
        .zip(expected_upper_a.iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= JACOBIAN_TOL,
            "upper_a mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .lower_b()
        .iter()
        .zip(expected_lower_b.iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= TOL,
            "lower_b mismatch at index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }
    for (idx, (&actual_value, &expected_value)) in actual
        .upper_b()
        .iter()
        .zip(expected_upper_b.iter())
        .enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= TOL,
            "upper_b mismatch at index {idx}: actual={actual_value}, expected={expected_value}"
        );
    }

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_new_rejects_non_3d_kernel() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0_f32]).expect("kernel");
    let err = Conv1dLayer::new(kernel, None, 1, 0).expect_err("kernel must be 3D");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_ibp_single_channel_exact_positive_kernel() -> Result<()> {
    // y = 2x + 0.5 for each position.
    let layer = make_conv1d(2.0, Some(0.5), 3);
    let input = BoundedTensor::new(
        array![[1.0_f32, 2.0, 3.0]].into_dyn(),
        array![[4.0_f32, 5.0, 6.0]].into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 3]);
    assert_close(output.lower()[[0, 0]], 2.5, TOL);
    assert_close(output.lower()[[0, 1]], 4.5, TOL);
    assert_close(output.lower()[[0, 2]], 6.5, TOL);
    assert_close(output.upper()[[0, 0]], 8.5, TOL);
    assert_close(output.upper()[[0, 1]], 10.5, TOL);
    assert_close(output.upper()[[0, 2]], 12.5, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_requires_input_length() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0_f32]).expect("kernel");
    let layer = Conv1dLayer::new(kernel, None, 1, 0).expect("valid conv1d");
    let bounds = LinearBounds::identity(1);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("missing input length should fail");
    assert!(matches!(err, NyError::UnsupportedConfiguration(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_identity_bounds_maps_to_scaled_identity() -> Result<()> {
    // With 1x1 kernel=2 and identity incoming A, backward pass should produce 2*I.
    let layer = make_conv1d(2.0, Some(0.5), 3);
    let bounds = LinearBounds::identity(3);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[3, 3]);
    assert_eq!(result.upper_a.shape(), &[3, 3]);
    for row in 0..3 {
        for col in 0..3 {
            let expected = if row == col { 2.0 } else { 0.0 };
            assert_close(result.lower_a[[row, col]], expected, TOL);
            assert_close(result.upper_a[[row, col]], expected, TOL);
        }
        // Bias contributes once per row because each identity row selects one conv output.
        assert_close(result.lower_b[row], 0.5, TOL);
        assert_close(result.upper_b[row], 0.5, TOL);
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_shape_mismatch_on_wrong_mid_dim() {
    let layer = make_conv1d(2.0, None, 3);
    let wrong_bounds = LinearBounds::identity(2);
    let err = layer
        .propagate_linear(&wrong_bounds)
        .expect_err("mid-dim mismatch should fail");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_batched_identity_bounds_maps_to_scaled_identity() -> Result<()> {
    let layer = make_conv1d(2.0, Some(0.5), 3);
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;
    let result = layer.propagate_linear_batched(&bounds)?;

    assert_eq!(result.lower_a.shape(), &[2, 3, 3]);
    assert_eq!(result.upper_a.shape(), &[2, 3, 3]);
    assert_eq!(result.lower_b.shape(), &[2, 3]);
    assert_eq!(result.upper_b.shape(), &[2, 3]);
    assert_eq!(result.input_shape, vec![2, 3]);

    for batch in 0..2 {
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 2.0 } else { 0.0 };
                assert_close(result.lower_a[[batch, row, col]], expected, TOL);
                assert_close(result.upper_a[[batch, row, col]], expected, TOL);
            }
            assert_close(result.lower_b[[batch, row]], 0.5, TOL);
            assert_close(result.upper_b[[batch, row]], 0.5, TOL);
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_batched_crown_engine_matches_cpu_3622() -> Result<()> {
    let layer = make_conv1d(2.0, Some(0.5), 3);
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;
    let expected = layer.propagate_linear_batched(&bounds)?;
    let engine = CountingGemmEngine::new();
    let actual = layer.propagate_linear_batched_maybe_engine(&bounds, Some(&engine))?;

    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "#3622 regression: Conv1d batched CROWN should invoke GemmEngine, got {calls} calls"
    );
    assert_batched_bounds_close(&actual, &expected, TOL, "conv1d_gemm");
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_ibp_single_channel_exact_positive_kernel() -> Result<()> {
    // y = 3x + 0.25 for each position with 1x1 kernel.
    let layer = make_convtranspose1d(3.0, Some(0.25), 3);
    let input = BoundedTensor::new(
        array![[1.0_f32, 2.0, 3.0]].into_dyn(),
        array![[4.0_f32, 5.0, 6.0]].into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 3]);
    assert_close(output.lower()[[0, 0]], 3.25, TOL);
    assert_close(output.lower()[[0, 1]], 6.25, TOL);
    assert_close(output.lower()[[0, 2]], 9.25, TOL);
    assert_close(output.upper()[[0, 0]], 12.25, TOL);
    assert_close(output.upper()[[0, 1]], 15.25, TOL);
    assert_close(output.upper()[[0, 2]], 18.25, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_identity_bounds_maps_to_scaled_identity() -> Result<()> {
    let layer = make_convtranspose1d(3.0, Some(0.25), 3);
    let bounds = LinearBounds::identity(3);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[3, 3]);
    assert_eq!(result.upper_a.shape(), &[3, 3]);
    for row in 0..3 {
        for col in 0..3 {
            let expected = if row == col { 3.0 } else { 0.0 };
            assert_close(result.lower_a[[row, col]], expected, TOL);
            assert_close(result.upper_a[[row, col]], expected, TOL);
        }
        assert_close(result.lower_b[row], 0.25, TOL);
        assert_close(result.upper_b[row], 0.25, TOL);
    }
    Ok(())
}

// ===== Non-trivial kernel tests =====

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_ibp_negative_kernel_swaps_bounds() -> Result<()> {
    // Negative weight: IBP must use W+/W- splitting correctly for transposed conv.
    let layer = make_convtranspose1d(-2.0, Some(1.0), 3);
    let input = BoundedTensor::new(
        array![[1.0_f32, 2.0, 3.0]].into_dyn(),
        array![[4.0_f32, 5.0, 6.0]].into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 3]);
    // lower = -2 * upper + 1 = [-7, -9, -11]
    assert_close(output.lower()[[0, 0]], -7.0, TOL);
    assert_close(output.lower()[[0, 1]], -9.0, TOL);
    assert_close(output.lower()[[0, 2]], -11.0, TOL);
    // upper = -2 * lower + 1 = [-1, -3, -5]
    assert_close(output.upper()[[0, 0]], -1.0, TOL);
    assert_close(output.upper()[[0, 1]], -3.0, TOL);
    assert_close(output.upper()[[0, 2]], -5.0, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_kernel3_backward() -> Result<()> {
    // ConvTranspose1d with kernel size 3, stride 1, no padding.
    // Input length 3 → output length = (3-1)*1 + 3 - 0 = 5.
    // Kernel K = [1, -1, 2], shape (in_c=1, out_c=1, k=3).
    //
    // CROWN backward of transposed conv = regular conv(A, K).
    // With identity A (5x5), each row of the result is the gradient of y[j] w.r.t. x.
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -1.0, 2.0]).expect("valid shape");
    let layer = ConvTranspose1dLayer::with_input_length(kernel, None, 1, 0, 3)?;

    let bounds = LinearBounds::identity(5);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[5, 3]);
    assert_eq!(result.upper_a.shape(), &[5, 3]);

    // Row 0: y[0] = 1*x[0]
    assert_close(result.lower_a[[0, 0]], 1.0, TOL);
    assert_close(result.lower_a[[0, 1]], 0.0, TOL);
    assert_close(result.lower_a[[0, 2]], 0.0, TOL);
    // Row 1: y[1] = -1*x[0] + 1*x[1]
    assert_close(result.lower_a[[1, 0]], -1.0, TOL);
    assert_close(result.lower_a[[1, 1]], 1.0, TOL);
    assert_close(result.lower_a[[1, 2]], 0.0, TOL);
    // Row 2: y[2] = 2*x[0] - x[1] + x[2]
    assert_close(result.lower_a[[2, 0]], 2.0, TOL);
    assert_close(result.lower_a[[2, 1]], -1.0, TOL);
    assert_close(result.lower_a[[2, 2]], 1.0, TOL);
    // Row 3: y[3] = 2*x[1] - x[2]
    assert_close(result.lower_a[[3, 0]], 0.0, TOL);
    assert_close(result.lower_a[[3, 1]], 2.0, TOL);
    assert_close(result.lower_a[[3, 2]], -1.0, TOL);
    // Row 4: y[4] = 2*x[2]
    assert_close(result.lower_a[[4, 0]], 0.0, TOL);
    assert_close(result.lower_a[[4, 1]], 0.0, TOL);
    assert_close(result.lower_a[[4, 2]], 2.0, TOL);

    // ConvTranspose is linear: upper_a must equal lower_a
    for row in 0..5 {
        for col in 0..3 {
            assert_close(result.upper_a[[row, col]], result.lower_a[[row, col]], TOL);
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_kernel3_soundness() -> Result<()> {
    // Verify CROWN bounds match IBP for ConvTranspose1d (linear layer).
    // Kernel size 3, stride 1, no padding.
    // Input length 4 → output length = (4-1)*1 + 3 = 6.
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.5, 0.3]).expect("valid shape");
    let bias = array![0.1_f32];
    let layer = ConvTranspose1dLayer::with_input_length(kernel, Some(bias), 1, 0, 4)?;

    let out_dim = 6;
    let in_dim = 4;
    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    let lower_vals: Vec<f32> = vec![-1.0, 0.5, -0.3, 2.0];
    let upper_vals: Vec<f32> = vec![1.0, 2.5, 0.7, 4.0];
    let input_lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), lower_vals.clone()).expect("valid shape");
    let input_upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), upper_vals.clone()).expect("valid shape");
    let input_bt = BoundedTensor::new(input_lower, input_upper)?;
    let ibp = layer.propagate_ibp(&input_bt)?;

    for d in 0..out_dim {
        let mut lo = result.lower_b[d];
        let mut hi = result.upper_b[d];
        for j in 0..in_dim {
            let al = result.lower_a[[d, j]];
            let au = result.upper_a[[d, j]];
            lo += if al >= 0.0 {
                al * lower_vals[j]
            } else {
                al * upper_vals[j]
            };
            hi += if au >= 0.0 {
                au * upper_vals[j]
            } else {
                au * lower_vals[j]
            };
        }
        let ibp_lo = *ibp.lower().iter().nth(d).expect("ibp has output d");
        let ibp_hi = *ibp.upper().iter().nth(d).expect("ibp has output d");
        assert!(
            (lo - ibp_lo).abs() < 1e-4,
            "CROWN lower {} != IBP lower {} for output {}",
            lo,
            ibp_lo,
            d
        );
        assert!(
            (hi - ibp_hi).abs() < 1e-4,
            "CROWN upper {} != IBP upper {} for output {}",
            hi,
            ibp_hi,
            d
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_groups1_dilation1_regression_3771() -> Result<()> {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.75, 0.25]).expect("valid shape");
    let bias = array![0.2_f32];
    let legacy =
        ConvTranspose1dLayer::with_input_length(kernel.clone(), Some(bias.clone()), 2, 1, 4)?;
    let widened = ConvTranspose1dLayer::with_input_length_full(kernel, Some(bias), 2, 1, 1, 1, 4)?;

    let out_dim = legacy.out_channels() * legacy.output_length(4)?;
    let bounds = LinearBounds::identity(out_dim);

    let legacy_result = legacy.propagate_linear(&bounds)?.into_owned();
    let widened_result = widened.propagate_linear(&bounds)?.into_owned();
    assert_linear_bounds_parity(
        &legacy_result,
        &widened_result,
        TOL,
        "ConvTranspose1d groups=1 dilation=1 regression",
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_backward_groups2_3771() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 2]), vec![1.0_f32, -0.5, 0.25, 2.0])
        .expect("valid shape");
    let bias = array![0.1_f32, -0.2];
    let layer = make_convtranspose1d_full(kernel, Some(bias), 1, 0, 1, 2, 3);

    let out_dim = layer.out_channels() * layer.output_length(3)?;
    let bounds = LinearBounds::identity(out_dim);
    assert_convtranspose1d_backward_matches_explicit(&layer, &bounds)?;

    let input = BoundedTensor::new(
        array![[0.0_f32, -1.0, 0.5], [1.0, -0.5, 2.0]].into_dyn(),
        array![[1.0_f32, 0.5, 2.0], [2.0, 1.5, 3.0]].into_dyn(),
    )?;
    let ibp = layer.propagate_ibp(&input)?;
    let crown = layer
        .propagate_linear(&LinearBounds::identity(out_dim))?
        .into_owned()
        .concretize(&input)
        .reshape(&[layer.out_channels(), layer.output_length(3)?])?;

    for (idx, (&crown_value, &ibp_value)) in
        crown.lower().iter().zip(ibp.lower().iter()).enumerate()
    {
        assert!(
            (crown_value - ibp_value).abs() <= 1e-4,
            "grouped ConvTranspose1d lower mismatch at {idx}: {crown_value} vs {ibp_value}"
        );
    }
    for (idx, (&crown_value, &ibp_value)) in
        crown.upper().iter().zip(ibp.upper().iter()).enumerate()
    {
        assert!(
            (crown_value - ibp_value).abs() <= 1e-4,
            "grouped ConvTranspose1d upper mismatch at {idx}: {crown_value} vs {ibp_value}"
        );
    }

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_backward_dilation2_3771() -> Result<()> {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -1.0, 0.5]).expect("valid shape");
    let layer = make_convtranspose1d_full(kernel, None, 1, 0, 2, 1, 4);

    let out_dim = layer.out_channels() * layer.output_length(4)?;
    let bounds = LinearBounds::identity(out_dim);
    assert_convtranspose1d_backward_matches_explicit(&layer, &bounds)?;

    let input = BoundedTensor::new(
        array![[0.0_f32, -1.0, 0.5, 1.0]].into_dyn(),
        array![[1.0_f32, 0.0, 2.0, 3.0]].into_dyn(),
    )?;
    let ibp = layer.propagate_ibp(&input)?;
    let crown = layer
        .propagate_linear(&LinearBounds::identity(out_dim))?
        .into_owned()
        .concretize(&input)
        .reshape(&[layer.out_channels(), layer.output_length(4)?])?;

    for (idx, (&crown_value, &ibp_value)) in
        crown.lower().iter().zip(ibp.lower().iter()).enumerate()
    {
        assert!(
            (crown_value - ibp_value).abs() <= 1e-4,
            "dilated ConvTranspose1d lower mismatch at {idx}: {crown_value} vs {ibp_value}"
        );
    }
    for (idx, (&crown_value, &ibp_value)) in
        crown.upper().iter().zip(ibp.upper().iter()).enumerate()
    {
        assert!(
            (crown_value - ibp_value).abs() <= 1e-4,
            "dilated ConvTranspose1d upper mismatch at {idx}: {crown_value} vs {ibp_value}"
        );
    }

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_ibp_negative_kernel_swaps_bounds() -> Result<()> {
    // Negative weight: IBP must use W- splitting.
    // y = -2x + 1, kernel = [[-2]], bias = [1]
    let layer = make_conv1d(-2.0, Some(1.0), 3);
    let input = BoundedTensor::new(
        array![[1.0_f32, 2.0, 3.0]].into_dyn(),
        array![[4.0_f32, 5.0, 6.0]].into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 3]);
    // lower = -2 * upper + 1 = [-7, -9, -11]
    assert_close(output.lower()[[0, 0]], -7.0, TOL);
    assert_close(output.lower()[[0, 1]], -9.0, TOL);
    assert_close(output.lower()[[0, 2]], -11.0, TOL);
    // upper = -2 * lower + 1 = [-1, -3, -5]
    assert_close(output.upper()[[0, 0]], -1.0, TOL);
    assert_close(output.upper()[[0, 1]], -3.0, TOL);
    assert_close(output.upper()[[0, 2]], -5.0, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_ibp_3d_batched() -> Result<()> {
    // 3D batched IBP: (batch=2, in_c=1, length=3)
    let layer = make_conv1d(2.0, Some(0.5), 3);
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![4.0, 5.0, 6.0, 40.0, 50.0, 60.0]).unwrap();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[2, 1, 3]);
    // Batch 0: lower = 2*[1,2,3]+0.5 = [2.5,4.5,6.5]
    assert_close(output.lower()[[0, 0, 0]], 2.5, TOL);
    assert_close(output.lower()[[0, 0, 2]], 6.5, TOL);
    // Batch 1: upper = 2*[40,50,60]+0.5 = [80.5,100.5,120.5]
    assert_close(output.upper()[[1, 0, 0]], 80.5, TOL);
    assert_close(output.upper()[[1, 0, 2]], 120.5, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_kernel3_backward() -> Result<()> {
    // Kernel size 3 on input length 4 → output length 2 (stride 1, no padding).
    // Kernel K = [1, -1, 2], 1 in-channel, 1 out-channel.
    // y[0] = 1*x[0] + (-1)*x[1] + 2*x[2]
    // y[1] = 1*x[1] + (-1)*x[2] + 2*x[3]
    //
    // CROWN backward with identity A (2x2): transposed_conv produces
    // the Toeplitz-like gradient matrix from the kernel.
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -1.0, 2.0]).unwrap();
    let layer = Conv1dLayer::with_input_length(kernel, None, 1, 0, 4)?;

    // Output dim = 1*2 = 2, input dim = 1*4 = 4
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    // Result should be (2x4):
    // Row 0 (y[0] gradient): [1, -1, 2, 0]
    // Row 1 (y[1] gradient): [0, 1, -1, 2]
    assert_eq!(result.lower_a.shape(), &[2, 4]);
    assert_eq!(result.upper_a.shape(), &[2, 4]);
    // y[0] depends on x[0..3]
    assert_close(result.lower_a[[0, 0]], 1.0, TOL);
    assert_close(result.lower_a[[0, 1]], -1.0, TOL);
    assert_close(result.lower_a[[0, 2]], 2.0, TOL);
    assert_close(result.lower_a[[0, 3]], 0.0, TOL);
    // y[1] depends on x[1..4]
    assert_close(result.lower_a[[1, 0]], 0.0, TOL);
    assert_close(result.lower_a[[1, 1]], 1.0, TOL);
    assert_close(result.lower_a[[1, 2]], -1.0, TOL);
    assert_close(result.lower_a[[1, 3]], 2.0, TOL);
    // Conv is linear: upper_a must equal lower_a
    for row in 0..2 {
        for col in 0..4 {
            assert_close(result.upper_a[[row, col]], result.lower_a[[row, col]], TOL);
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_kernel3_soundness() -> Result<()> {
    // Verify CROWN bounds match IBP for linear layer.
    // Kernel size 3 on input length 5 → output length 3.
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.5, 0.3]).unwrap();
    let bias = array![0.1_f32];
    let layer = Conv1dLayer::with_input_length(kernel, Some(bias), 1, 0, 5)?;

    let out_dim = 3;
    let in_dim = 5;
    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    // Evaluate CROWN bounds via interval arithmetic and compare with IBP
    let lower_vals: Vec<f32> = vec![-1.0, 0.5, -0.3, 2.0, -1.5];
    let upper_vals: Vec<f32> = vec![1.0, 2.5, 0.7, 4.0, 0.5];
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[1, 5]), lower_vals.clone()).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[1, 5]), upper_vals.clone()).unwrap();
    let input_bt = BoundedTensor::new(input_lower, input_upper)?;
    let ibp = layer.propagate_ibp(&input_bt)?;

    for d in 0..out_dim {
        let mut lo = result.lower_b[d];
        let mut hi = result.upper_b[d];
        for j in 0..in_dim {
            let al = result.lower_a[[d, j]];
            let au = result.upper_a[[d, j]];
            lo += if al >= 0.0 {
                al * lower_vals[j]
            } else {
                al * upper_vals[j]
            };
            hi += if au >= 0.0 {
                au * upper_vals[j]
            } else {
                au * lower_vals[j]
            };
        }
        let ibp_lo = *ibp.lower().iter().nth(d).unwrap();
        let ibp_hi = *ibp.upper().iter().nth(d).unwrap();
        assert!(
            (lo - ibp_lo).abs() < 1e-4,
            "CROWN lower {} != IBP lower {} for output {}",
            lo,
            ibp_lo,
            d
        );
        assert!(
            (hi - ibp_hi).abs() < 1e-4,
            "CROWN upper {} != IBP upper {} for output {}",
            hi,
            ibp_hi,
            d
        );
    }
    Ok(())
}

// --- Regression tests for #2828: stride=0 and invalid spatial configs ---

/// Regression test for #2828: Conv1d stride=0 must be rejected by constructor.
#[ntest::timeout(10000)]
#[test]
fn conv1d_zero_stride_rejected() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0; 3]).expect("kernel");
    let err = Conv1dLayer::new(kernel, None, 0, 0).expect_err("stride=0 must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("stride")),
        "expected InvalidSpec about stride, got: {err:?}"
    );
}

/// Regression test for #2828: ConvTranspose1d stride=0 must be rejected.
#[ntest::timeout(10000)]
#[test]
fn conv_transpose1d_zero_stride_rejected() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0; 3]).expect("kernel");
    let err = ConvTranspose1dLayer::new(kernel, None, 0, 0).expect_err("stride=0 must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("stride")),
        "expected InvalidSpec about stride, got: {err:?}"
    );
}

/// Regression test for #2828: Conv1d output_length underflow (kernel > padded input).
#[ntest::timeout(10000)]
#[test]
fn conv1d_output_length_underflow_returns_error() -> Result<()> {
    // kernel_size=5, input_len=2, padding=0 → padded=2 < 5 → underflow
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![1.0; 5]).expect("kernel");
    let conv = Conv1dLayer::new(kernel, None, 1, 0)?;
    let err = conv
        .output_length(2)
        .expect_err("kernel > padded input must error");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("kernel")),
        "expected InvalidSpec about kernel size, got: {err:?}"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn conv1d_output_length_checked_arithmetic_rejects_overflow() -> Result<()> {
    let kernel = ArrayD::from_elem(IxDyn(&[1, 1, 2]), 1.0_f32);
    let conv = Conv1dLayer::new_full(kernel.clone(), None, 1, 0, usize::MAX, 1)?;
    assert!(
        matches!(conv.output_length(2), Err(NyError::InvalidSpec(_))),
        "Conv1d effective-kernel overflow must return an error"
    );

    let transpose = ConvTranspose1dLayer::new_full(kernel, None, usize::MAX, 0, usize::MAX, 1)?;
    assert!(
        matches!(
            transpose.output_length(usize::MAX),
            Err(NyError::InvalidSpec(_))
        ),
        "ConvTranspose1d expanded-length overflow must return an error"
    );
    Ok(())
}

/// Regression test for #2828: Conv1d valid stride=1 is accepted.
#[ntest::timeout(10000)]
#[test]
fn conv1d_stride_one_accepted() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0; 3]).expect("kernel");
    Conv1dLayer::new(kernel, None, 1, 0).expect("stride=1 must be accepted");
}

/// Regression test for #2877: IBP forward with kernel > padded input returns error, not panic.
#[ntest::timeout(10000)]
#[test]
fn conv1d_ibp_kernel_oversized_returns_error() -> Result<()> {
    // kernel_size=5, input_len=2, padding=0 → padded=2 < 5 → conv1d_single underflow.
    // Before #2877 fix, this caused usize underflow panic in conv1d_single.
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 5]), vec![1.0; 5]).expect("kernel");
    let conv = Conv1dLayer::with_input_length(kernel, None, 1, 0, 2)?;
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0_f32, 1.0]).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![2.0_f32, 3.0]).expect("upper"),
    )?;
    let err = conv.propagate_ibp(&input);
    assert!(
        err.is_err(),
        "Expected error for oversized kernel, got {:?}",
        err
    );
    Ok(())
}

/// Regression test for #2747: Conv1d CROWN backward with NaN kernel returns
/// NumericalInstability error instead of silently producing NaN coefficients.
#[ntest::timeout(10000)]
#[test]
fn conv1d_crown_backward_nan_kernel_returns_error() {
    let layer = make_conv1d(f32::NAN, None, 4);
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "NaN kernel should return NumericalInstability, got {:?}",
        result
    );
}

/// Regression test for #2747: ConvTranspose1d CROWN backward with NaN kernel
/// returns NumericalInstability error.
#[ntest::timeout(10000)]
#[test]
fn convtranspose1d_crown_backward_nan_kernel_returns_error() {
    let layer = make_convtranspose1d(f32::NAN, None, 4);
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "NaN kernel should return NumericalInstability, got {:?}",
        result
    );
}

/// Regression test for #4204 / #2747: Conv1d CROWN backward with NaN bias
/// returns NumericalInstability error (crown_helpers.rs line 25 guard).
///
/// Conv2d already has conv2d_crown_backward_nan_bias_returns_error; this
/// exercises the same shared guard_nan_weights path through the Conv1d caller.
#[ntest::timeout(10000)]
#[test]
fn conv1d_crown_backward_nan_bias_returns_error_4204() {
    let layer = make_conv1d(1.0, Some(f32::NAN), 4);
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "NaN bias should return NumericalInstability, got {:?}",
        result
    );
}

/// Regression test for #4204 / #2747: ConvTranspose1d CROWN backward with NaN
/// bias returns NumericalInstability error.
#[ntest::timeout(10000)]
#[test]
fn convtranspose1d_crown_backward_nan_bias_returns_error_4204() {
    let layer = make_convtranspose1d(1.0, Some(f32::NAN), 4);
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "NaN bias should return NumericalInstability, got {:?}",
        result
    );
}

/// Assert that a row has been replaced with the non-finite fallback pattern:
/// all-zero A-matrix row and ±inf bias.
fn assert_nonfinite_row_fallback(lb: &LinearBounds, row: usize) {
    for j in 0..lb.lower_a().ncols() {
        assert_eq!(lb.lower_a()[[row, j]], 0.0, "row {row} lower_a[{j}]");
        assert_eq!(lb.upper_a()[[row, j]], 0.0, "row {row} upper_a[{j}]");
    }
    assert_eq!(lb.lower_b()[row], f32::NEG_INFINITY);
    assert_eq!(lb.upper_b()[row], f32::INFINITY);
}

/// Assert row has all-finite coefficients and bias.
fn assert_finite_row(lb: &LinearBounds, row: usize) {
    for j in 0..lb.lower_a().ncols() {
        assert!(lb.lower_a()[[row, j]].is_finite(), "row {row} lower_a[{j}]");
        assert!(lb.upper_a()[[row, j]].is_finite(), "row {row} upper_a[{j}]");
    }
    assert!(lb.lower_b()[row].is_finite(), "row {row} lower_b");
    assert!(lb.upper_b()[row].is_finite(), "row {row} upper_b");
}

/// Build LinearBounds: row 0 has 1e19 coefficients (triggers overflow), row 1 has 1.0 (safe).
fn make_overflow_bounds(num_cols: usize) -> LinearBounds {
    use ndarray::{Array1, Array2};
    let mut la = Array2::<f32>::zeros((2, num_cols));
    let mut ua = Array2::<f32>::zeros((2, num_cols));
    for j in 0..num_cols {
        la[[0, j]] = 1e19;
        ua[[0, j]] = 1e19;
        la[[1, j]] = 1.0;
        ua[[1, j]] = 1.0;
    }
    LinearBounds::new(la, Array1::zeros(2), ua, Array1::zeros(2)).expect("valid bounds")
}

/// Regression (#2812, #3228): Conv1d CROWN backward per-row coefficient magnitude fallback.
///
/// When large A-matrix coefficients (1e19) multiply weights (1e5), the output
/// exceeds CROWN_COEFF_MAX (1e10). The per-row fallback zeros the affected row
/// and sets bias to ±inf, producing sound [-inf, +inf] bounds for that output.
/// Row 1 (A=1.0) produces 1e5 output which is below CROWN_COEFF_MAX and stays safe.
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_backward_nonfinite_row_fallback() -> Result<()> {
    // 1 in_channel, 1 out_channel, kernel_size=1; input_length=2.
    // Large weight 1e5 * 1e19 coefficient → exceeds CROWN_COEFF_MAX.
    let layer = make_conv1d(1e5, Some(1.0), 2);
    let out_c = 1;
    let out_len = 2;
    let bounds = make_overflow_bounds(out_c * out_len);
    let lb = layer
        .propagate_linear(&bounds)
        .expect("should handle overflow via row fallback")
        .into_owned();
    assert_nonfinite_row_fallback(&lb, 0);
    assert_finite_row(&lb, 1);
    Ok(())
}

// ===== Batched CROWN tests for ConvTranspose1d =====

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_batched_kernel3_soundness() -> Result<()> {
    // Verify batched CROWN bounds match IBP for ConvTranspose1d (linear layer).
    // Kernel size 3, stride 1, no padding.
    // Input length 4 → output length = (4-1)*1 + 3 = 6.
    //
    // For linear layers, batched CROWN with identity A must concretize
    // to the same bounds as IBP. This tests the batched path in
    // ConvTranspose1dLayer::propagate_linear_batched (types.rs lines 456+).
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.5, 0.3]).expect("valid shape");
    let bias = array![0.1_f32];
    let layer = ConvTranspose1dLayer::with_input_length(kernel, Some(bias), 1, 0, 4)?;

    let out_dim = 6;
    let in_dim = 4;
    let batch_size = 2;

    // Batched identity bounds: shape [batch, out_dim, out_dim]
    let bounds = BatchedLinearBounds::identity(&[batch_size, out_dim])?;
    let result = layer.propagate_linear_batched(&bounds)?;

    assert_eq!(result.lower_a.shape(), &[batch_size, out_dim, in_dim]);
    assert_eq!(result.upper_a.shape(), &[batch_size, out_dim, in_dim]);
    assert_eq!(result.lower_b.shape(), &[batch_size, out_dim]);
    assert_eq!(result.upper_b.shape(), &[batch_size, out_dim]);

    // Concretize CROWN bounds and compare against IBP.
    let lower_vals: Vec<f32> = vec![-1.0, 0.5, -0.3, 2.0];
    let upper_vals: Vec<f32> = vec![1.0, 2.5, 0.7, 4.0];
    let input_lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), lower_vals.clone()).expect("valid shape");
    let input_upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 4]), upper_vals.clone()).expect("valid shape");
    let input_bt = BoundedTensor::new(input_lower, input_upper)?;
    let ibp = layer.propagate_ibp(&input_bt)?;

    // Each batch should give identical results (identity A is the same per batch).
    for b in 0..batch_size {
        for d in 0..out_dim {
            let mut lo = result.lower_b[[b, d]] as f64;
            let mut hi = result.upper_b[[b, d]] as f64;
            for j in 0..in_dim {
                let al = result.lower_a[[b, d, j]] as f64;
                let au = result.upper_a[[b, d, j]] as f64;
                lo += if al >= 0.0 {
                    al * lower_vals[j] as f64
                } else {
                    al * upper_vals[j] as f64
                };
                hi += if au >= 0.0 {
                    au * upper_vals[j] as f64
                } else {
                    au * lower_vals[j] as f64
                };
            }
            let ibp_lo = *ibp.lower().iter().nth(d).expect("ibp has output d") as f64;
            let ibp_hi = *ibp.upper().iter().nth(d).expect("ibp has output d") as f64;
            assert!(
                (lo - ibp_lo).abs() < 1e-3,
                "batch {} CROWN lower {} != IBP lower {} for output {}",
                b,
                lo,
                ibp_lo,
                d
            );
            assert!(
                (hi - ibp_hi).abs() < 1e-3,
                "batch {} CROWN upper {} != IBP upper {} for output {}",
                b,
                hi,
                ibp_hi,
                d
            );
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_batched_crown_engine_matches_cpu_3622() -> Result<()> {
    let layer = make_convtranspose1d(3.0, Some(0.25), 3);
    let bounds = BatchedLinearBounds::identity(&[2, 3])?;
    let expected = layer.propagate_linear_batched(&bounds)?;
    let engine = CountingGemmEngine::new();
    let actual = layer.propagate_linear_batched_maybe_engine(&bounds, Some(&engine))?;

    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "#3622 regression: ConvTranspose1d batched CROWN should invoke GemmEngine, got {calls} calls"
    );
    assert_batched_bounds_close(&actual, &expected, TOL, "conv_transpose1d_gemm");
    Ok(())
}

/// Regression (#2812): ConvTranspose1d CROWN backward per-row non-finite fallback.
/// Weight 1e5 chosen so row 0 (coeff 1e19) overflows: 1e19 * 1e5 = 1e24 > CROWN_COEFF_MAX (1e10),
/// but row 1 (coeff 1.0) stays safe: 1.0 * 1e5 = 1e5 < 1e10.
/// (Prior test used 1e20 which made BOTH rows exceed CROWN_COEFF_MAX after #3228 tightened
/// the threshold from is_finite() to is_crown_coeff_safe().)
#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_backward_nonfinite_row_fallback() -> Result<()> {
    let layer = make_convtranspose1d(1e5, Some(1.0), 2);
    let out_c = 1;
    let out_len = 2;
    let bounds = make_overflow_bounds(out_c * out_len);
    let lb = layer
        .propagate_linear(&bounds)
        .expect("should handle overflow via row fallback")
        .into_owned();
    assert_nonfinite_row_fallback(&lb, 0);
    assert_finite_row(&lb, 1);
    Ok(())
}

// ===== Dilation tests =====

/// Validate dilation=0 is rejected.
#[ntest::timeout(10000)]
#[test]
fn conv1d_zero_dilation_rejected() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0; 3]).expect("kernel");
    let err =
        Conv1dLayer::new_full(kernel, None, 1, 0, 0, 1).expect_err("dilation=0 must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("dilation")),
        "expected InvalidSpec about dilation, got: {err:?}"
    );
}

/// Validate groups=0 is rejected.
#[ntest::timeout(10000)]
#[test]
fn conv1d_zero_groups_rejected() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0; 3]).expect("kernel");
    let err =
        Conv1dLayer::new_full(kernel, None, 1, 0, 1, 0).expect_err("groups=0 must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("groups")),
        "expected InvalidSpec about groups, got: {err:?}"
    );
}

/// Validate out_channels not divisible by groups is rejected.
#[ntest::timeout(10000)]
#[test]
fn conv1d_out_channels_not_divisible_by_groups_rejected() {
    // out_c=3, groups=2 → 3 % 2 != 0
    let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1]), vec![1.0; 3]).expect("kernel");
    let err = Conv1dLayer::new_full(kernel, None, 1, 0, 1, 2).expect_err("out_c % groups != 0");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("divisible")),
        "expected InvalidSpec about divisibility, got: {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn conv1d_zero_kernel_dimensions_rejected() {
    for shape in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
        let kernel = ArrayD::zeros(IxDyn(&shape));
        let error = Conv1dLayer::new_full(kernel, None, 1, 0, 1, 1)
            .expect_err("zero Conv1d kernel dimension must be rejected");
        assert!(
            matches!(error, NyError::InvalidSpec(ref message) if message.contains("nonzero")),
            "expected nonzero-dimension error for {shape:?}, got {error:?}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn conv1d_total_input_channels_overflow_rejected() {
    let kernel = ArrayD::zeros(IxDyn(&[1, 2, 1]));
    let error = Conv1dLayer::new_full(kernel, None, 1, 0, 1, usize::MAX)
        .expect_err("total input-channel overflow must be rejected");
    assert!(
        matches!(error, NyError::InvalidSpec(ref message) if message.contains("overflow")),
        "expected input-channel overflow error, got {error:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn convtranspose1d_zero_kernel_dimensions_rejected() {
    for shape in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
        let kernel = ArrayD::zeros(IxDyn(&shape));
        let error = ConvTranspose1dLayer::new_full(kernel, None, 1, 0, 1, 1)
            .expect_err("zero ConvTranspose1d kernel dimension must be rejected");
        assert!(
            matches!(error, NyError::InvalidSpec(ref message) if message.contains("nonzero")),
            "expected nonzero-dimension error for {shape:?}, got {error:?}"
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn finite_deadline_conv1d_revalidates_publicly_mutable_geometry() {
    let input =
        BoundedTensor::concrete(ArrayD::zeros(IxDyn(&[1, 1]))).expect("concrete input bounds");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);

    let mut conv =
        Conv1dLayer::new_full(ArrayD::zeros(IxDyn(&[1, 2, 1])), None, 1, 0, 1, 1).expect("conv");
    conv.groups = usize::MAX;
    let conv_error = conv
        .propagate_ibp_with_engine_and_deadline(&input, None, Some(deadline))
        .expect_err("mutated Conv1d geometry must return a structured error");
    assert!(
        matches!(conv_error, NyError::InvalidSpec(ref message) if message.contains("overflow")),
        "expected Conv1d geometry overflow, got {conv_error:?}"
    );
    let conv_sound_error = conv
        .propagate_ibp_sound_with_engine(&input, None)
        .expect_err("sound Conv1d must revalidate mutated geometry without a deadline");
    assert!(
        matches!(conv_sound_error, NyError::InvalidSpec(ref message) if message.contains("overflow")),
        "expected sound Conv1d geometry overflow, got {conv_sound_error:?}"
    );

    let mut transpose =
        ConvTranspose1dLayer::new_full(ArrayD::zeros(IxDyn(&[1, 2, 1])), None, 1, 0, 1, 1)
            .expect("transpose");
    transpose.groups = usize::MAX;
    let transpose_error = transpose
        .propagate_ibp_with_engine_and_deadline(&input, None, Some(deadline))
        .expect_err("mutated ConvTranspose1d geometry must return a structured error");
    assert!(
        matches!(transpose_error, NyError::InvalidSpec(ref message) if message.contains("divisible")),
        "expected ConvTranspose1d geometry error, got {transpose_error:?}"
    );
    let transpose_sound_error = transpose
        .propagate_ibp_sound_with_engine(&input, None)
        .expect_err("sound ConvTranspose1d must revalidate mutated geometry without a deadline");
    assert!(
        matches!(transpose_sound_error, NyError::InvalidSpec(ref message) if message.contains("divisible")),
        "expected sound ConvTranspose1d geometry error, got {transpose_sound_error:?}"
    );
}

/// Conv1d IBP with dilation=2: verify output shape and bounds.
///
/// Kernel K = [1, -1, 2], in_c=1, out_c=1, k=3, dilation=2.
/// Input length=7, padding=0.
/// Effective kernel footprint = 2*(3-1) + 1 = 5.
/// Output length = (7 + 0 - 5) / 1 + 1 = 3.
///
/// y[0] = 1*x[0] + (-1)*x[2] + 2*x[4]
/// y[1] = 1*x[1] + (-1)*x[3] + 2*x[5]
/// y[2] = 1*x[2] + (-1)*x[4] + 2*x[6]
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_ibp_dilation2() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -1.0, 2.0]).unwrap();
    let layer = Conv1dLayer::with_input_length_full(kernel, None, 1, 0, 2, 1, 7)?;

    // Point input: lower == upper, so IBP should give exact result.
    let vals = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 7]), vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 7]), vals).unwrap(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 3]);
    // y[0] = 1*1 + (-1)*3 + 2*5 = 1 - 3 + 10 = 8
    assert_close(output.lower()[[0, 0]], 8.0, TOL);
    // y[1] = 1*2 + (-1)*4 + 2*6 = 2 - 4 + 12 = 10
    assert_close(output.lower()[[0, 1]], 10.0, TOL);
    // y[2] = 1*3 + (-1)*5 + 2*7 = 3 - 5 + 14 = 12
    assert_close(output.lower()[[0, 2]], 12.0, TOL);
    Ok(())
}

/// Conv1d CROWN backward with dilation=2: verify coefficient matrix.
///
/// Same config as IBP test: K = [1, -1, 2], dilation=2, input_len=7, out_len=3.
/// With identity A (3x3), CROWN backward should give gradient matrix (3x7).
///
/// Row 0: y[0] = 1*x[0] + (-1)*x[2] + 2*x[4]  → [1, 0, -1, 0, 2, 0, 0]
/// Row 1: y[1] = 1*x[1] + (-1)*x[3] + 2*x[5]  → [0, 1, 0, -1, 0, 2, 0]
/// Row 2: y[2] = 1*x[2] + (-1)*x[4] + 2*x[6]  → [0, 0, 1, 0, -1, 0, 2]
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_backward_dilation2() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -1.0, 2.0]).unwrap();
    let layer = Conv1dLayer::with_input_length_full(kernel, None, 1, 0, 2, 1, 7)?;

    let bounds = LinearBounds::identity(3);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[3, 7]);

    // Row 0: [1, 0, -1, 0, 2, 0, 0]
    let expected_row0 = [1.0, 0.0, -1.0, 0.0, 2.0, 0.0, 0.0];
    for (j, &e) in expected_row0.iter().enumerate() {
        assert_close(result.lower_a[[0, j]], e, TOL);
    }
    // Row 1: [0, 1, 0, -1, 0, 2, 0]
    let expected_row1 = [0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0];
    for (j, &e) in expected_row1.iter().enumerate() {
        assert_close(result.lower_a[[1, j]], e, TOL);
    }
    // Row 2: [0, 0, 1, 0, -1, 0, 2]
    let expected_row2 = [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 2.0];
    for (j, &e) in expected_row2.iter().enumerate() {
        assert_close(result.lower_a[[2, j]], e, TOL);
    }
    // Linear layer: upper_a == lower_a
    for row in 0..3 {
        for col in 0..7 {
            assert_close(result.upper_a[[row, col]], result.lower_a[[row, col]], TOL);
        }
    }
    Ok(())
}

/// Conv1d CROWN backward dilation=2 soundness: CROWN bounds match IBP.
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_dilation2_soundness() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.5, 0.3]).unwrap();
    let bias = array![0.1_f32];
    let layer =
        Conv1dLayer::with_input_length_full(kernel.clone(), Some(bias.clone()), 1, 0, 2, 1, 7)?;

    let out_dim = 3;
    let in_dim = 7;
    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();
    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    let lower_vals: Vec<f32> = vec![-1.0, 0.5, -0.3, 2.0, -1.5, 1.0, 0.2];
    let upper_vals: Vec<f32> = vec![1.0, 2.5, 0.7, 4.0, 0.5, 3.0, 1.8];
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[1, 7]), lower_vals.clone()).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[1, 7]), upper_vals.clone()).unwrap();

    let ibp_layer = Conv1dLayer::new_full(kernel, Some(bias), 1, 0, 2, 1)?;
    let input_bt = BoundedTensor::new(input_lower, input_upper)?;
    let ibp = ibp_layer.propagate_ibp(&input_bt)?;

    for d in 0..out_dim {
        let mut lo = result.lower_b[d] as f64;
        let mut hi = result.upper_b[d] as f64;
        for j in 0..in_dim {
            let al = result.lower_a[[d, j]] as f64;
            let au = result.upper_a[[d, j]] as f64;
            lo += if al >= 0.0 {
                al * lower_vals[j] as f64
            } else {
                al * upper_vals[j] as f64
            };
            hi += if au >= 0.0 {
                au * upper_vals[j] as f64
            } else {
                au * lower_vals[j] as f64
            };
        }
        let ibp_lo = *ibp.lower().iter().nth(d).unwrap() as f64;
        let ibp_hi = *ibp.upper().iter().nth(d).unwrap() as f64;
        assert!(
            (lo - ibp_lo).abs() < 1e-3,
            "CROWN lower {} != IBP lower {} for output {}",
            lo,
            ibp_lo,
            d
        );
        assert!(
            (hi - ibp_hi).abs() < 1e-3,
            "CROWN upper {} != IBP upper {} for output {}",
            hi,
            ibp_hi,
            d
        );
    }
    Ok(())
}

// ===== Groups tests =====

/// Conv1d IBP with groups=2: verify output shape and bounds.
///
/// Kernel shape: (out_c=4, in_c/groups=1, k=1), groups=2.
/// Input: (in_c=2, length=3). Group 0: ic=0, oc=0,1. Group 1: ic=1, oc=2,3.
/// Kernels: oc0=[2.0], oc1=[-1.0], oc2=[3.0], oc3=[0.5]
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_ibp_groups2() -> Result<()> {
    // out_c=4, in_c_per_group=1, k=1; total in_c = 1 * 2 = 2
    let kernel = ArrayD::from_shape_vec(IxDyn(&[4, 1, 1]), vec![2.0, -1.0, 3.0, 0.5]).unwrap();
    let layer = Conv1dLayer::with_input_length_full(kernel, None, 1, 0, 1, 2, 3)?;

    assert_eq!(layer.in_channels(), 2);
    assert_eq!(layer.out_channels(), 4);

    // Point input (lower == upper)
    let vals = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2, 3)
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3]), vals).unwrap(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[4, 3]);
    // Group 0 (ic=0): oc0 = 2*x0, oc1 = -1*x0
    assert_close(output.lower()[[0, 0]], 2.0, TOL); // 2*1
    assert_close(output.lower()[[0, 1]], 4.0, TOL); // 2*2
    assert_close(output.lower()[[0, 2]], 6.0, TOL); // 2*3
    assert_close(output.lower()[[1, 0]], -1.0, TOL); // -1*1
    assert_close(output.lower()[[1, 1]], -2.0, TOL); // -1*2
                                                     // Group 1 (ic=1): oc2 = 3*x1, oc3 = 0.5*x1
    assert_close(output.lower()[[2, 0]], 12.0, TOL); // 3*4
    assert_close(output.lower()[[2, 1]], 15.0, TOL); // 3*5
    assert_close(output.lower()[[3, 0]], 2.0, TOL); // 0.5*4
    assert_close(output.lower()[[3, 1]], 2.5, TOL); // 0.5*5
    Ok(())
}

/// Conv1d CROWN backward with groups=2: verify gradient matrix.
///
/// Same config: out_c=4, in_c_per_group=1, k=1, groups=2.
/// Output is (4, 3) → flat dim 12.
/// Input is (2, 3) → flat dim 6.
/// With identity A (12x12), CROWN backward gives (12x6).
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_backward_groups2() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[4, 1, 1]), vec![2.0_f32, -1.0, 3.0, 0.5]).unwrap();
    let layer = Conv1dLayer::with_input_length_full(kernel, None, 1, 0, 1, 2, 3)?;

    let out_dim = 4 * 3; // out_c * out_len
    let in_dim = 2 * 3; // in_c * in_len
    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    // Check that group 0 outputs only depend on group 0 inputs (ic=0)
    // and group 1 outputs only depend on group 1 inputs (ic=1).
    // oc0,1 → ic0; oc2,3 → ic1.
    // Flat layout: output[oc*3+ol], input[ic*3+il]

    // oc=0 (group 0), ol=0: y = 2 * x[0,0]
    // → row 0, col 0 should be 2.0, col 3..5 should be 0
    assert_close(result.lower_a[[0, 0]], 2.0, TOL);
    assert_close(result.lower_a[[0, 3]], 0.0, TOL); // no cross-group

    // oc=2 (group 1), ol=0: y = 3 * x[1,0]
    // → row 6, col 3 should be 3.0, col 0..2 should be 0
    assert_close(result.lower_a[[6, 3]], 3.0, TOL);
    assert_close(result.lower_a[[6, 0]], 0.0, TOL); // no cross-group

    // Linear: upper_a == lower_a
    for row in 0..out_dim {
        for col in 0..in_dim {
            assert_close(result.upper_a[[row, col]], result.lower_a[[row, col]], TOL);
        }
    }
    Ok(())
}

/// Conv1d CROWN groups=2 soundness: CROWN bounds match IBP.
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_groups2_soundness() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[4, 1, 1]), vec![2.0_f32, -1.0, 3.0, 0.5]).unwrap();
    let bias = array![0.1_f32, -0.2, 0.3, 0.0];
    let layer =
        Conv1dLayer::with_input_length_full(kernel.clone(), Some(bias.clone()), 1, 0, 1, 2, 3)?;

    let out_c = 4;
    let out_len = 3;
    let out_dim = out_c * out_len;
    let in_c = 2;
    let in_len = 3;
    let in_dim = in_c * in_len;

    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();
    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    let lower_vals: Vec<f32> = vec![-1.0, 0.5, -0.3, 2.0, -1.5, 1.0]; // (2, 3)
    let upper_vals: Vec<f32> = vec![1.0, 2.5, 0.7, 4.0, 0.5, 3.0];
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[2, 3]), lower_vals.clone()).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[2, 3]), upper_vals.clone()).unwrap();

    let ibp_layer = Conv1dLayer::new_full(kernel, Some(bias), 1, 0, 1, 2)?;
    let input_bt = BoundedTensor::new(input_lower, input_upper)?;
    let ibp = ibp_layer.propagate_ibp(&input_bt)?;

    // Flatten IBP output for comparison
    let ibp_lower_flat: Vec<f32> = ibp.lower().iter().cloned().collect();
    let ibp_upper_flat: Vec<f32> = ibp.upper().iter().cloned().collect();

    for d in 0..out_dim {
        let mut lo = result.lower_b[d] as f64;
        let mut hi = result.upper_b[d] as f64;
        for j in 0..in_dim {
            let al = result.lower_a[[d, j]] as f64;
            let au = result.upper_a[[d, j]] as f64;
            lo += if al >= 0.0 {
                al * lower_vals[j] as f64
            } else {
                al * upper_vals[j] as f64
            };
            hi += if au >= 0.0 {
                au * upper_vals[j] as f64
            } else {
                au * lower_vals[j] as f64
            };
        }
        assert!(
            (lo - ibp_lower_flat[d] as f64).abs() < 1e-3,
            "CROWN lower {} != IBP lower {} for output {}",
            lo,
            ibp_lower_flat[d],
            d
        );
        assert!(
            (hi - ibp_upper_flat[d] as f64).abs() < 1e-3,
            "CROWN upper {} != IBP upper {} for output {}",
            hi,
            ibp_upper_flat[d],
            d
        );
    }
    Ok(())
}

/// Conv1d with depthwise convolution (groups == in_channels == out_channels).
///
/// Each channel has its own independent 1D convolution.
/// out_c=3, in_c_per_group=1, k=2, groups=3, in_c=3.
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_ibp_depthwise() -> Result<()> {
    // 3 independent channels, each with kernel size 2
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[3, 1, 2]), vec![1.0_f32, 2.0, -1.0, 0.5, 3.0, -3.0])
            .unwrap();
    let layer = Conv1dLayer::with_input_length_full(kernel, None, 1, 0, 1, 3, 4)?;

    assert_eq!(layer.in_channels(), 3);
    assert_eq!(layer.out_channels(), 3);
    assert_eq!(layer.output_length(4)?, 3);

    // Point input (3, 4)
    let vals = vec![
        1.0, 2.0, 3.0, 4.0, // channel 0
        5.0, 6.0, 7.0, 8.0, // channel 1
        9.0, 10.0, 11.0, 12.0, // channel 2
    ];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 4]), vals.clone()).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 4]), vals).unwrap(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[3, 3]);
    // Channel 0: k=[1,2], y[0] = 1*1 + 2*2 = 5
    assert_close(output.lower()[[0, 0]], 5.0, TOL);
    // Channel 0: y[1] = 1*2 + 2*3 = 8
    assert_close(output.lower()[[0, 1]], 8.0, TOL);
    // Channel 1: k=[-1,0.5], y[0] = -1*5 + 0.5*6 = -2
    assert_close(output.lower()[[1, 0]], -2.0, TOL);
    // Channel 2: k=[3,-3], y[0] = 3*9 + (-3)*10 = -3
    assert_close(output.lower()[[2, 0]], -3.0, TOL);
    Ok(())
}

/// Conv1d output_length with dilation: verify formula correctness.
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_output_length_with_dilation() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0; 3]).unwrap();

    // dilation=1: effective_k=3, out = (10 + 0 - 3) / 1 + 1 = 8
    let layer1 = Conv1dLayer::new_full(kernel.clone(), None, 1, 0, 1, 1)?;
    assert_eq!(layer1.output_length(10)?, 8);

    // dilation=2: effective_k=5, out = (10 + 0 - 5) / 1 + 1 = 6
    let layer2 = Conv1dLayer::new_full(kernel.clone(), None, 1, 0, 2, 1)?;
    assert_eq!(layer2.output_length(10)?, 6);

    // dilation=3: effective_k=7, out = (10 + 0 - 7) / 1 + 1 = 4
    let layer3 = Conv1dLayer::new_full(kernel.clone(), None, 1, 0, 3, 1)?;
    assert_eq!(layer3.output_length(10)?, 4);

    // dilation=4: effective_k=9, out = (10 + 0 - 9) / 1 + 1 = 2
    let layer4 = Conv1dLayer::new_full(kernel.clone(), None, 1, 0, 4, 1)?;
    assert_eq!(layer4.output_length(10)?, 2);

    // dilation=5: effective_k=11 > 10, should error
    let layer5 = Conv1dLayer::new_full(kernel, None, 1, 0, 5, 1)?;
    assert!(
        layer5.output_length(10).is_err(),
        "dilation=5 effective_k=11 > input=10 should error"
    );

    Ok(())
}

// ─── GemmEngine baseline parity tests (#3598) ───────────────────────────────
//
// Verify that propagate_linear_with_engine(bounds, Some(&NaiveCpuGemmEngine))
// produces identical results to propagate_linear_with_engine(bounds, None).
// These pin that threading the engine through Conv1d/ConvTranspose1d CROWN
// backward does not change bound values.

/// Helper: assert two LinearBounds are element-wise equal within tolerance.
fn assert_linear_bounds_parity(
    baseline: &LinearBounds,
    with_engine: &LinearBounds,
    tol: f32,
    label: &str,
) {
    assert_eq!(
        baseline.lower_a.shape(),
        with_engine.lower_a.shape(),
        "{label}: lower_a shape mismatch"
    );
    assert_eq!(
        baseline.upper_a.shape(),
        with_engine.upper_a.shape(),
        "{label}: upper_a shape mismatch"
    );
    let rows = baseline.num_outputs();
    let cols = baseline.num_inputs();
    for r in 0..rows {
        for c in 0..cols {
            assert!(
                (baseline.lower_a[[r, c]] - with_engine.lower_a[[r, c]]).abs() <= tol,
                "{label}: lower_a[{r},{c}] mismatch: baseline={}, engine={}",
                baseline.lower_a[[r, c]],
                with_engine.lower_a[[r, c]],
            );
            assert!(
                (baseline.upper_a[[r, c]] - with_engine.upper_a[[r, c]]).abs() <= tol,
                "{label}: upper_a[{r},{c}] mismatch: baseline={}, engine={}",
                baseline.upper_a[[r, c]],
                with_engine.upper_a[[r, c]],
            );
        }
        assert!(
            (baseline.lower_b[r] - with_engine.lower_b[r]).abs() <= tol,
            "{label}: lower_b[{r}] mismatch: baseline={}, engine={}",
            baseline.lower_b[r],
            with_engine.lower_b[r],
        );
        assert!(
            (baseline.upper_b[r] - with_engine.upper_b[r]).abs() <= tol,
            "{label}: upper_b[{r}] mismatch: baseline={}, engine={}",
            baseline.upper_b[r],
            with_engine.upper_b[r],
        );
    }
}

#[test]
fn finite_deadline_conv1d_paths_refuse_generic_engine_and_preserve_parity() -> Result<()> {
    let deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(30));
    for (label, baseline, finite, calls) in {
        let conv = make_conv1d(2.0, Some(0.25), 3);
        let conv_bounds = LinearBounds::identity(3);
        let conv_baseline = conv
            .propagate_linear_with_engine(&conv_bounds, None)?
            .into_owned();
        let conv_engine = CountingGemmEngine::new();
        let conv_finite = conv
            .propagate_linear_with_engine_and_deadline(&conv_bounds, Some(&conv_engine), deadline)?
            .into_owned();

        let transpose = make_convtranspose1d(-1.5, Some(0.5), 3);
        let transpose_bounds = LinearBounds::identity(3);
        let transpose_baseline = transpose
            .propagate_linear_with_engine(&transpose_bounds, None)?
            .into_owned();
        let transpose_engine = CountingGemmEngine::new();
        let transpose_finite = transpose
            .propagate_linear_with_engine_and_deadline(
                &transpose_bounds,
                Some(&transpose_engine),
                deadline,
            )?
            .into_owned();
        [
            (
                "Conv1d finite deadline",
                conv_baseline,
                conv_finite,
                conv_engine.gemm_calls(),
            ),
            (
                "ConvTranspose1d finite deadline",
                transpose_baseline,
                transpose_finite,
                transpose_engine.gemm_calls(),
            ),
        ]
    } {
        assert_eq!(
            calls, 0,
            "{label}: finite deadline must not enter caller GemmEngine"
        );
        assert_linear_bounds_parity(&baseline, &finite, TOL, label);
    }

    let expired = Some(
        std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(1))
            .expect("Instant epoch is at least one millisecond old"),
    );
    let engine = CountingGemmEngine::new();
    let conv_error = make_conv1d(1.0, None, 3)
        .propagate_linear_with_engine_and_deadline(
            &LinearBounds::identity(3),
            Some(&engine),
            expired,
        )
        .expect_err("expired Conv1d authority must refuse before dispatch");
    assert!(matches!(conv_error, NyError::DeadlineExceeded(_)));
    let transpose_error = make_convtranspose1d(1.0, None, 3)
        .propagate_linear_with_engine_and_deadline(
            &LinearBounds::identity(3),
            Some(&engine),
            expired,
        )
        .expect_err("expired ConvTranspose1d authority must refuse before dispatch");
    assert!(matches!(transpose_error, NyError::DeadlineExceeded(_)));
    assert_eq!(engine.gemm_calls(), 0);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn finite_deadline_conv1d_ibp_refuses_generic_engine_and_preserves_parity() -> Result<()> {
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 6]),
        (0..12).map(|i| -1.0 + i as f32 * 0.05).collect(),
    )
    .expect("lower");
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 6]),
        (0..12).map(|i| 0.5 + i as f32 * 0.1).collect(),
    )
    .expect("upper");
    let input = BoundedTensor::new(lower, upper)?;
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 3]),
        vec![
            0.5, -0.25, 0.75, -0.1, 0.2, 0.3, -0.4, 0.6, 0.1, 0.8, -0.5, 0.25,
        ],
    )
    .expect("kernel");
    let conv = Conv1dLayer::new_full(kernel.clone(), Some(array![0.2_f32, -0.3]), 1, 1, 1, 1)?;
    let transpose =
        ConvTranspose1dLayer::new_full(kernel, Some(array![0.1_f32, -0.2]), 1, 1, 1, 1)?;
    let deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(30));

    let conv_expected = conv.propagate_ibp_sound_with_engine(&input, None)?;
    let conv_engine = CountingGemmEngine::new();
    let conv_actual =
        conv.propagate_ibp_sound_with_engine_and_deadline(&input, Some(&conv_engine), deadline)?;
    assert_eq!(
        conv_engine.gemm_calls(),
        0,
        "finite Conv1d IBP must refuse the generic engine"
    );
    assert_bounded_tensor_close(
        &conv_actual,
        &conv_expected,
        1e-4,
        "finite Conv1d certified IBP",
    );

    let transpose_expected = transpose.propagate_ibp_sound_with_engine(&input, None)?;
    let transpose_engine = CountingGemmEngine::new();
    let transpose_actual = transpose.propagate_ibp_sound_with_engine_and_deadline(
        &input,
        Some(&transpose_engine),
        deadline,
    )?;
    assert_eq!(
        transpose_engine.gemm_calls(),
        0,
        "finite ConvTranspose1d IBP must refuse the generic engine"
    );
    assert_bounded_tensor_close(
        &transpose_actual,
        &transpose_expected,
        1e-4,
        "finite ConvTranspose1d certified IBP",
    );

    let historical_engine = CountingGemmEngine::new();
    conv.propagate_ibp_sound_with_engine_and_deadline(&input, Some(&historical_engine), None)?;
    assert!(
        historical_engine.gemm_calls() > 0,
        "deadline=None must preserve historical Conv1d engine dispatch"
    );

    let expired = Some(
        std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(1))
            .expect("Instant supports a 1ms subtraction"),
    );
    let expired_engine = CountingGemmEngine::new();
    let conv_error = conv
        .propagate_ibp_sound_with_engine_and_deadline(&input, Some(&expired_engine), expired)
        .expect_err("expired Conv1d IBP must abort");
    assert!(matches!(conv_error, NyError::DeadlineExceeded(_)));
    let transpose_error = transpose
        .propagate_ibp_sound_with_engine_and_deadline(&input, Some(&expired_engine), expired)
        .expect_err("expired ConvTranspose1d IBP must abort");
    assert!(matches!(transpose_error, NyError::DeadlineExceeded(_)));
    assert_eq!(expired_engine.gemm_calls(), 0);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn finite_deadline_batched_grouped_dilated_conv1d_ibp_matches_historical() -> Result<()> {
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 7]),
        (0..28).map(|i| -0.8 + i as f32 * 0.025).collect(),
    )
    .expect("lower");
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 7]),
        (0..28).map(|i| 0.4 + i as f32 * 0.04).collect(),
    )
    .expect("upper");
    let input = BoundedTensor::new(lower, upper)?;
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.5, -0.25, 0.75, -0.4, 0.6, 0.1])
        .expect("kernel");
    let conv = Conv1dLayer::new_full(kernel.clone(), Some(array![0.2_f32, -0.3]), 1, 2, 2, 2)?;
    let transpose =
        ConvTranspose1dLayer::new_full(kernel, Some(array![0.1_f32, -0.2]), 1, 2, 2, 2)?;
    let deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(30));
    let engine = CountingGemmEngine::new();

    let conv_expected = conv.propagate_ibp_sound_with_engine(&input, None)?;
    let conv_actual =
        conv.propagate_ibp_sound_with_engine_and_deadline(&input, Some(&engine), deadline)?;
    assert_bounded_tensor_close(
        &conv_actual,
        &conv_expected,
        1e-4,
        "finite batched grouped dilated Conv1d IBP",
    );

    let transpose_expected = transpose.propagate_ibp_sound_with_engine(&input, None)?;
    let transpose_actual =
        transpose.propagate_ibp_sound_with_engine_and_deadline(&input, Some(&engine), deadline)?;
    assert_bounded_tensor_close(
        &transpose_actual,
        &transpose_expected,
        1e-4,
        "finite batched grouped dilated ConvTranspose1d IBP",
    );
    assert_eq!(
        engine.gemm_calls(),
        0,
        "finite batched grouped 1D convolutions must refuse the generic engine"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn finite_deadline_conv1d_ibp_keeps_certified_cancellation_enclosure() -> Result<()> {
    let input = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0_f32; 3]).expect("Conv1d input"),
    )?;
    let conv = Conv1dLayer::new_full(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 1, 3]),
            vec![16_777_216.0_f32, 1.0, -16_777_216.0],
        )
        .expect("Conv1d kernel"),
        None,
        1,
        0,
        1,
        1,
    )?;
    let engine = CountingGemmEngine::new();
    let finite = conv.propagate_ibp_sound_with_engine_and_deadline(
        &input,
        Some(&engine),
        Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
    )?;
    assert_eq!(engine.gemm_calls(), 0);
    assert!(
        finite.lower()[[0, 0]] <= 1.0 && finite.upper()[[0, 0]] >= 1.0,
        "finite Conv1d certified bounds must enclose the exact sum 1.0, got [{}, {}]",
        finite.lower()[[0, 0]],
        finite.upper()[[0, 0]]
    );

    let transpose_input = BoundedTensor::concrete(
        ArrayD::from_shape_vec(IxDyn(&[3, 1]), vec![1.0_f32; 3]).expect("ConvTranspose1d input"),
    )?;
    let transpose = ConvTranspose1dLayer::new_full(
        ArrayD::from_shape_vec(
            IxDyn(&[3, 1, 1]),
            vec![16_777_216.0_f32, 1.0, -16_777_216.0],
        )
        .expect("ConvTranspose1d kernel"),
        None,
        1,
        0,
        1,
        1,
    )?;
    let transpose_engine = CountingGemmEngine::new();
    let finite = transpose.propagate_ibp_sound_with_engine_and_deadline(
        &transpose_input,
        Some(&transpose_engine),
        Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
    )?;
    assert_eq!(transpose_engine.gemm_calls(), 0);
    assert!(
        finite.lower()[[0, 0]] <= 1.0 && finite.upper()[[0, 0]] >= 1.0,
        "finite ConvTranspose1d certified bounds must enclose the exact sum 1.0, got [{}, {}]",
        finite.lower()[[0, 0]],
        finite.upper()[[0, 0]]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn certified_conv1d_ibp_encloses_product_underflow_with_or_without_deadline() -> Result<()> {
    // Each exact product is 2^-150, below the binary32 subnormal midpoint.
    // Five terms sum to 5*2^-150 = 2.5*2^-149, which the old relative-only
    // Higham certificate excluded after both y and S rounded/flushed to zero.
    let x = f32::MIN_POSITIVE; // 2^-126, normal: DAZ cannot erase the source.
    let w = f32::from_bits(103_u32 << 23); // 2^-24, normal.
    let exact_product = f64::from_bits((1_023_u64 - 150) << 52);
    let exact_sum = 5.0_f64 * exact_product;

    let conv_input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[1, 5]), x))?;
    let conv = Conv1dLayer::new_full(ArrayD::from_elem(IxDyn(&[1, 1, 5]), w), None, 1, 0, 1, 1)?;
    let plain = conv.propagate_ibp(&conv_input)?;
    assert_eq!(plain.lower()[[0, 0]], 0.0);
    assert_eq!(plain.upper()[[0, 0]], 0.0);
    for deadline in [
        None,
        Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
    ] {
        let certified =
            conv.propagate_ibp_sound_with_engine_and_deadline(&conv_input, None, deadline)?;
        assert!(
            f64::from(certified.lower()[[0, 0]]) <= exact_sum
                && f64::from(certified.upper()[[0, 0]]) >= exact_sum,
            "Conv1d [1,1,5] certificate excluded exact 5*2^-150: [{}, {}]",
            certified.lower()[[0, 0]],
            certified.upper()[[0, 0]]
        );
    }

    let transpose_input = BoundedTensor::concrete(ArrayD::from_elem(IxDyn(&[5, 1]), x))?;
    let transpose =
        ConvTranspose1dLayer::new_full(ArrayD::from_elem(IxDyn(&[5, 1, 1]), w), None, 1, 0, 1, 1)?;
    let plain = transpose.propagate_ibp(&transpose_input)?;
    assert_eq!(plain.lower()[[0, 0]], 0.0);
    assert_eq!(plain.upper()[[0, 0]], 0.0);
    for deadline in [
        None,
        Some(std::time::Instant::now() + std::time::Duration::from_secs(30)),
    ] {
        let certified = transpose.propagate_ibp_sound_with_engine_and_deadline(
            &transpose_input,
            None,
            deadline,
        )?;
        assert!(
            f64::from(certified.lower()[[0, 0]]) <= exact_sum
                && f64::from(certified.upper()[[0, 0]]) >= exact_sum,
            "ConvTranspose1d [5,1,1] certificate excluded exact 5*2^-150: [{}, {}]",
            certified.lower()[[0, 0]],
            certified.upper()[[0, 0]]
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn finite_deadline_non_depthwise_grouped_stride2_conv1d_ibp_matches_historical() -> Result<()> {
    const BATCH: usize = 2;
    const IN_CHANNELS: usize = 4;
    const OUT_CHANNELS: usize = 6;
    const GROUPS: usize = 2;

    let conv_input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[BATCH, IN_CHANNELS, 11]),
            (0..BATCH * IN_CHANNELS * 11)
                .map(|index| -1.0 + index as f32 * 0.005)
                .collect(),
        )
        .expect("Conv1d grouped lower"),
        ArrayD::from_shape_vec(
            IxDyn(&[BATCH, IN_CHANNELS, 11]),
            (0..BATCH * IN_CHANNELS * 11)
                .map(|index| 0.25 + index as f32 * 0.01)
                .collect(),
        )
        .expect("Conv1d grouped upper"),
    )?;
    let conv_kernel = ArrayD::from_shape_vec(
        IxDyn(&[OUT_CHANNELS, IN_CHANNELS / GROUPS, 3]),
        (0..OUT_CHANNELS * (IN_CHANNELS / GROUPS) * 3)
            .map(|index| (index as f32 - 17.0) * 0.04)
            .collect(),
    )
    .expect("Conv1d grouped kernel");
    let conv = Conv1dLayer::new_full(
        conv_kernel,
        Some(array![0.1_f32, -0.2, 0.3, -0.4, 0.5, -0.6]),
        2,
        1,
        1,
        GROUPS,
    )?;

    let transpose_input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[BATCH, IN_CHANNELS, 6]),
            (0..BATCH * IN_CHANNELS * 6)
                .map(|index| -0.75 + index as f32 * 0.01)
                .collect(),
        )
        .expect("ConvTranspose1d grouped lower"),
        ArrayD::from_shape_vec(
            IxDyn(&[BATCH, IN_CHANNELS, 6]),
            (0..BATCH * IN_CHANNELS * 6)
                .map(|index| 0.5 + index as f32 * 0.015)
                .collect(),
        )
        .expect("ConvTranspose1d grouped upper"),
    )?;
    let transpose_kernel = ArrayD::from_shape_vec(
        IxDyn(&[IN_CHANNELS, OUT_CHANNELS / GROUPS, 3]),
        (0..IN_CHANNELS * (OUT_CHANNELS / GROUPS) * 3)
            .map(|index| (13.0 - index as f32) * 0.035)
            .collect(),
    )
    .expect("ConvTranspose1d grouped kernel");
    let transpose = ConvTranspose1dLayer::new_full(
        transpose_kernel,
        Some(array![0.2_f32, -0.1, 0.4, -0.3, 0.6, -0.5]),
        2,
        1,
        1,
        GROUPS,
    )?;

    let deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(30));
    let engine = CountingGemmEngine::new();
    let conv_expected = conv.propagate_ibp_sound_with_engine(&conv_input, None)?;
    let conv_actual =
        conv.propagate_ibp_sound_with_engine_and_deadline(&conv_input, Some(&engine), deadline)?;
    assert_bounded_tensor_close(
        &conv_actual,
        &conv_expected,
        1e-4,
        "non-depthwise groups=2 stride=2 Conv1d certified IBP",
    );

    let transpose_expected = transpose.propagate_ibp_sound_with_engine(&transpose_input, None)?;
    let transpose_actual = transpose.propagate_ibp_sound_with_engine_and_deadline(
        &transpose_input,
        Some(&engine),
        deadline,
    )?;
    assert_bounded_tensor_close(
        &transpose_actual,
        &transpose_expected,
        1e-4,
        "non-depthwise groups=2 stride=2 ConvTranspose1d certified IBP",
    );
    assert_eq!(
        engine.gemm_calls(),
        0,
        "finite non-depthwise grouped stride-2 paths must refuse the generic engine"
    );
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn finite_deadline_large_conv1d_ibp_work_aborts_cooperatively() -> Result<()> {
    const CHANNELS: usize = 16;
    const INPUT_LEN: usize = 8_192;
    const KERNEL_LEN: usize = 15;
    let input = BoundedTensor::new(
        ArrayD::zeros(IxDyn(&[CHANNELS, INPUT_LEN])),
        ArrayD::ones(IxDyn(&[CHANNELS, INPUT_LEN])),
    )?;
    let kernel = ArrayD::ones(IxDyn(&[CHANNELS, CHANNELS, KERNEL_LEN]));
    let conv = Conv1dLayer::new_full(kernel.clone(), None, 1, 0, 1, 1)?;
    let transpose = ConvTranspose1dLayer::new_full(kernel, None, 1, 0, 1, 1)?;
    let engine = CountingGemmEngine::new();

    let conv_error = conv
        .propagate_ibp_sound_with_engine_and_deadline(
            &input,
            Some(&engine),
            Some(std::time::Instant::now() + std::time::Duration::from_millis(2)),
        )
        .expect_err("large Conv1d work must observe its finite deadline");
    assert!(matches!(conv_error, NyError::DeadlineExceeded(_)));

    let transpose_error = transpose
        .propagate_ibp_sound_with_engine_and_deadline(
            &input,
            Some(&engine),
            Some(std::time::Instant::now() + std::time::Duration::from_millis(2)),
        )
        .expect_err("large ConvTranspose1d work must observe its finite deadline");
    assert!(matches!(transpose_error, NyError::DeadlineExceeded(_)));
    assert_eq!(
        engine.gemm_calls(),
        0,
        "finite large Conv1d work must never enter the generic engine"
    );
    Ok(())
}

#[ntest::timeout(5000)]
#[test]
fn finite_deadline_zero_output_channel_convtranspose1d_polls_input_traversal() {
    const INPUT_LEN: usize = 1_000_000;
    let lower = ArrayD::zeros(IxDyn(&[1, INPUT_LEN]));
    let upper = ArrayD::ones(IxDyn(&[1, INPUT_LEN]));
    let kernel = ArrayD::zeros(IxDyn(&[1, 0, 1]));

    let started = std::time::Instant::now();
    let ordinary_error = conv1d_transpose_ibp_forward_with_deadline(
        lower.view(),
        upper.view(),
        &kernel,
        1,
        0,
        1,
        1,
        std::time::Instant::now() + std::time::Duration::from_millis(2),
    )
    .expect_err("zero-output-channel ordinary scatter must poll input traversal");
    assert!(matches!(ordinary_error, NyError::DeadlineExceeded(_)));
    assert!(
        started.elapsed() < std::time::Duration::from_secs(1),
        "ordinary zero-output-channel traversal did not abort promptly"
    );

    let started = std::time::Instant::now();
    let certified_error = conv1d_transpose_ibp_certified_forward(
        lower.view(),
        upper.view(),
        &kernel,
        None,
        1,
        0,
        1,
        1,
        Some(std::time::Instant::now() + std::time::Duration::from_millis(2)),
    )
    .expect_err("zero-output-channel certified scatter must poll input traversal");
    assert!(matches!(certified_error, NyError::DeadlineExceeded(_)));
    assert!(
        started.elapsed() < std::time::Duration::from_secs(1),
        "certified zero-output-channel traversal did not abort promptly"
    );
}

/// Conv1d CROWN backward: engine=None matches engine=Some(NaiveCpuGemmEngine).
/// Kernel size 3, no bias, single channel. (#3598)
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_engine_parity_kernel3_no_bias_3598() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -1.0, 2.0]).unwrap();
    let layer = Conv1dLayer::with_input_length(kernel, None, 1, 0, 4)?;

    let bounds = LinearBounds::identity(2); // out_c * out_len = 1 * 2
    let baseline = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let with_engine = layer
        .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();

    assert_linear_bounds_parity(&baseline, &with_engine, TOL, "Conv1d k=3 no-bias");
    Ok(())
}

/// Conv1d CROWN backward: engine parity with bias. (#3598)
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_engine_parity_with_bias_3598() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.5, 0.3]).unwrap();
    let bias = array![0.1_f32];
    let layer = Conv1dLayer::with_input_length(kernel, Some(bias), 1, 0, 5)?;

    let bounds = LinearBounds::identity(3); // out_c * out_len = 1 * 3
    let baseline = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let with_engine = layer
        .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();

    assert_linear_bounds_parity(&baseline, &with_engine, TOL, "Conv1d k=3 with-bias");
    Ok(())
}

/// Conv1d CROWN backward: engine parity with multi-channel (2 in, 3 out). (#3598)
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_engine_parity_multichannel_3598() -> Result<()> {
    // out_channels=3, in_channels=2, kernel_size=2
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[3, 2, 2]),
        vec![
            1.0_f32, -1.0, 0.5, 0.5, // out=0: in_c=0 [1,-1], in_c=1 [0.5,0.5]
            2.0, 0.0, -1.0, 1.0, // out=1: in_c=0 [2,0], in_c=1 [-1,1]
            0.3, -0.3, 0.7, 0.7, // out=2: in_c=0 [0.3,-0.3], in_c=1 [0.7,0.7]
        ],
    )
    .unwrap();
    let bias = array![0.1_f32, -0.2, 0.0];
    let input_length = 4;
    let layer = Conv1dLayer::with_input_length(kernel, Some(bias), 1, 0, input_length)?;

    let out_len = layer.output_length(input_length)?;
    let out_c = layer.out_channels();
    let bounds = LinearBounds::identity(out_c * out_len);

    let baseline = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let with_engine = layer
        .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();

    assert_linear_bounds_parity(&baseline, &with_engine, TOL, "Conv1d multichannel");
    Ok(())
}

/// Conv1d CROWN backward: engine parity with dilation=2. (#3598)
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_engine_parity_dilation2_3598() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -1.0, 2.0]).unwrap();
    // dilation=2: effective_k=5, out = (8 + 0 - 5) / 1 + 1 = 4
    let layer = Conv1dLayer::with_input_length_full(kernel, None, 1, 0, 2, 1, 8)?;

    let bounds = LinearBounds::identity(4);
    let baseline = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let with_engine = layer
        .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();

    assert_linear_bounds_parity(&baseline, &with_engine, TOL, "Conv1d dilation=2");
    Ok(())
}

/// Conv1d CROWN backward: engine parity with non-identity incoming bounds. (#3598)
///
/// Using non-identity A matrices exercises the full GEMM path, not just
/// the trivial identity-times-Toeplitz case.
#[ntest::timeout(10000)]
#[test]
fn test_conv1d_crown_engine_parity_non_identity_bounds_3598() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.5, 0.3]).unwrap();
    let layer = Conv1dLayer::with_input_length(kernel, Some(array![0.1_f32]), 1, 0, 5)?;

    // Non-identity bounds: 2 output objectives over 3 conv outputs
    let lower_a = Array2::from_shape_vec((2, 3), vec![0.5_f32, -0.3, 1.0, -1.0, 0.7, 0.2]).unwrap();
    let upper_a = lower_a.clone();
    let lower_b = Array1::from_vec(vec![0.0_f32, 0.0]);
    let upper_b = lower_b.clone();
    let bounds = LinearBounds::new(lower_a, lower_b, upper_a, upper_b)?;

    let baseline = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let with_engine = layer
        .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();

    assert_linear_bounds_parity(&baseline, &with_engine, TOL, "Conv1d non-identity");
    Ok(())
}

/// ConvTranspose1d CROWN backward: engine=None matches engine=Some. (#3598)
#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_engine_parity_3598() -> Result<()> {
    // ONNX ConvTranspose layout: (in_channels, out_channels, kernel_size)
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.5, 0.3]).unwrap();
    let layer = ConvTranspose1dLayer::with_input_length(kernel, None, 1, 0, 4)?;

    let out_len = layer.output_length(4)?;
    let out_c = layer.out_channels();
    let bounds = LinearBounds::identity(out_c * out_len);

    let baseline = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let with_engine = layer
        .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();

    assert_linear_bounds_parity(&baseline, &with_engine, TOL, "ConvTranspose1d no-bias");
    Ok(())
}

/// ConvTranspose1d CROWN backward: engine parity with bias. (#3598)
#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_engine_parity_with_bias_3598() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![2.0_f32, -1.0, 0.5]).unwrap();
    let bias = array![0.3_f32];
    let layer = ConvTranspose1dLayer::with_input_length(kernel, Some(bias), 1, 0, 4)?;

    let out_len = layer.output_length(4)?;
    let out_c = layer.out_channels();
    let bounds = LinearBounds::identity(out_c * out_len);

    let baseline = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let with_engine = layer
        .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();

    assert_linear_bounds_parity(&baseline, &with_engine, TOL, "ConvTranspose1d with-bias");
    Ok(())
}

/// ConvTranspose1d CROWN backward: engine parity with non-identity bounds. (#3598)
#[ntest::timeout(10000)]
#[test]
fn test_convtranspose1d_crown_engine_parity_non_identity_3598() -> Result<()> {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3]), vec![1.0_f32, -0.5, 0.3]).unwrap();
    let bias = array![0.1_f32];
    let layer = ConvTranspose1dLayer::with_input_length(kernel, Some(bias), 1, 0, 4)?;

    let out_len = layer.output_length(4)?;
    let out_c = layer.out_channels();

    // Non-identity bounds: 2 objectives over out_c * out_len conv outputs
    let conv_out_dim = out_c * out_len;
    let lower_a = Array2::from_shape_fn((2, conv_out_dim), |(r, c)| {
        ((r as f32 + 1.0) * 0.3 - c as f32 * 0.1).sin()
    });
    let upper_a = lower_a.clone();
    let lower_b = Array1::from_vec(vec![0.0_f32; 2]);
    let upper_b = lower_b.clone();
    let bounds = LinearBounds::new(lower_a, lower_b, upper_a, upper_b)?;

    let baseline = layer
        .propagate_linear_with_engine(&bounds, None)?
        .into_owned();
    let with_engine = layer
        .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))?
        .into_owned();

    assert_linear_bounds_parity(&baseline, &with_engine, TOL, "ConvTranspose1d non-identity");
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { max_shrink_time: 5000, ..ProptestConfig::with_cases(64) })]

    /// #3500 / #3622: ConvTranspose1d has no auto_LiRPA 1D reference, so prove
    /// the GEMM-backed backward pass against the explicit flattened forward
    /// Jacobian under ny's row-vector A convention.
    #[ntest::timeout(10000)]
    #[test]
    fn proptest_convtranspose1d_crown_backward_matches_explicit_jacobian_3500(
        case in convtranspose1d_jacobian_case()
    ) {
        let kernel = ArrayD::from_shape_vec(
            IxDyn(&[case.in_channels, case.out_channels, case.kernel_size]),
            case.kernel.clone(),
        )
        .expect("kernel shape");
        let bias = Array1::from_vec(case.bias.clone());
        let layer = ConvTranspose1dLayer::with_input_length(
            kernel,
            Some(bias),
            case.stride,
            case.padding,
            case.input_length,
        )
        .expect("valid ConvTranspose1d");

        let incoming_a = Array2::from_shape_vec(
            (case.num_objectives, case.output_dim()),
            case.incoming_a.clone(),
        )
        .expect("incoming A shape");
        let incoming_b = Array1::from_vec(case.incoming_b);
        let bounds = LinearBounds::new(
            incoming_a.clone(),
            incoming_b.clone(),
            incoming_a.clone(),
            incoming_b.clone(),
        )
        .expect("linear bounds");

        let actual = layer
            .propagate_linear_with_engine(&bounds, Some(&NaiveCpuGemmEngine))
            .expect("ConvTranspose1d backward should succeed")
            .into_owned();
        let explicit_jacobian =
            explicit_convtranspose1d_jacobian(&layer).expect("explicit Jacobian");
        let explicit_bias = explicit_convtranspose1d_bias(&layer).expect("explicit bias");
        let expected_a = incoming_a.dot(&explicit_jacobian);
        let (expected_lower_b, expected_upper_b) =
            explicit_convtranspose1d_bias_bounds(&incoming_a, &incoming_b, &explicit_bias);

        prop_assert_eq!(actual.lower_a.shape(), expected_a.shape());
        prop_assert_eq!(actual.upper_a.shape(), expected_a.shape());
        for (idx, (&actual_value, &expected_value)) in actual
            .lower_a
            .iter()
            .zip(expected_a.iter())
            .enumerate()
        {
            prop_assert!(
                (actual_value - expected_value).abs() <= JACOBIAN_TOL,
                "lower_a mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
            );
        }
        for (idx, (&actual_value, &expected_value)) in actual
            .upper_a
            .iter()
            .zip(expected_a.iter())
            .enumerate()
        {
            prop_assert!(
                (actual_value - expected_value).abs() <= JACOBIAN_TOL,
                "upper_a mismatch at flat index {idx}: actual={actual_value}, expected={expected_value}"
            );
        }
        for (idx, (&actual_value, &expected_value)) in actual
            .lower_b
            .iter()
            .zip(expected_lower_b.iter())
            .enumerate()
        {
            prop_assert!(
                (actual_value - expected_value).abs() <= TOL,
                "lower_b mismatch at index {idx}: actual={actual_value}, expected={expected_value}"
            );
        }
        for (idx, (&actual_value, &expected_value)) in actual
            .upper_b
            .iter()
            .zip(expected_upper_b.iter())
            .enumerate()
        {
            prop_assert!(
                (actual_value - expected_value).abs() <= TOL,
                "upper_b mismatch at index {idx}: actual={actual_value}, expected={expected_value}"
            );
        }
    }
}
