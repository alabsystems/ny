// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

use super::*;
use crate::layers::common::BoundPropagation;
use crate::tests::{assert_batched_bounds_close, assert_close};
use crate::{BatchedLinearBounds, LinearBounds};
use ndarray::{array, Array2, ArrayD, IxDyn};

const TOL: f32 = 1e-6;

fn make_conv2d(weight: f32, bias: Option<f32>, input_hw: (usize, usize)) -> Conv2dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![weight]).expect("kernel");
    let bias = bias.map(|b| array![b]);
    Conv2dLayer::with_input_shape(kernel, bias, (1, 1), (0, 0), input_hw.0, input_hw.1)
        .expect("valid conv2d")
}

fn make_convtranspose2d(
    weight: f32,
    bias: Option<f32>,
    input_hw: (usize, usize),
) -> ConvTranspose2dLayer {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![weight]).expect("kernel");
    let bias = bias.map(|b| array![b]);
    ConvTranspose2dLayer::with_input_shape(kernel, bias, (1, 1), (0, 0), input_hw.0, input_hw.1)
        .expect("valid convtranspose2d")
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_new_rejects_non_4d_kernel() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0_f32]).expect("kernel");
    let err = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).expect_err("kernel must be 4D");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_single_channel_exact_positive_kernel() -> Result<()> {
    // y = 2x + 0.5 for each spatial position with a 1x1 kernel.
    let layer = make_conv2d(2.0, Some(0.5), (2, 2));
    let input = BoundedTensor::new(
        array![[[1.0_f32, 2.0], [3.0, 4.0]]].into_dyn(),
        array![[[5.0_f32, 6.0], [7.0, 8.0]]].into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_close(output.lower()[[0, 0, 0]], 2.5, TOL);
    assert_close(output.lower()[[0, 0, 1]], 4.5, TOL);
    assert_close(output.lower()[[0, 1, 0]], 6.5, TOL);
    assert_close(output.lower()[[0, 1, 1]], 8.5, TOL);
    assert_close(output.upper()[[0, 0, 0]], 10.5, TOL);
    assert_close(output.upper()[[0, 0, 1]], 12.5, TOL);
    assert_close(output.upper()[[0, 1, 0]], 14.5, TOL);
    assert_close(output.upper()[[0, 1, 1]], 16.5, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_requires_input_shape() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).expect("kernel");
    let layer = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).expect("valid conv2d");
    let bounds = LinearBounds::identity(4);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("missing input shape should fail");
    assert!(matches!(err, NyError::UnsupportedConfiguration(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_identity_bounds_maps_to_scaled_identity() -> Result<()> {
    // With 1x1 kernel=2 and identity incoming A, backward pass should produce 2*I.
    let layer = make_conv2d(2.0, Some(0.5), (2, 2));
    let bounds = LinearBounds::identity(4);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[4, 4]);
    assert_eq!(result.upper_a.shape(), &[4, 4]);
    for row in 0..4 {
        for col in 0..4 {
            let expected = if row == col { 2.0 } else { 0.0 };
            assert_close(result.lower_a[[row, col]], expected, TOL);
            assert_close(result.upper_a[[row, col]], expected, TOL);
        }
        assert_close(result.lower_b[row], 0.5, TOL);
        assert_close(result.upper_b[row], 0.5, TOL);
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_shape_mismatch_on_wrong_mid_dim() {
    let layer = make_conv2d(2.0, None, (2, 2));
    let wrong_bounds = LinearBounds::identity(3);
    let err = layer
        .propagate_linear(&wrong_bounds)
        .expect_err("mid-dim mismatch should fail");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_batched_identity_bounds_maps_to_scaled_identity() -> Result<()> {
    let layer = make_conv2d(2.0, Some(0.5), (2, 2));
    let bounds = BatchedLinearBounds::identity(&[2, 4])?;
    let result = layer.propagate_linear_batched(&bounds, None)?;

    assert_eq!(result.lower_a.shape(), &[2, 4, 4]);
    assert_eq!(result.upper_a.shape(), &[2, 4, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 4]);
    assert_eq!(result.upper_b.shape(), &[2, 4]);
    assert_eq!(result.input_shape, vec![2, 4]);

    for batch in 0..2 {
        for row in 0..4 {
            for col in 0..4 {
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
fn test_conv2d_crown_batched_input_dims_overflow_returns_error_3012() {
    let layer = make_conv2d(1.0, None, (usize::MAX, 2));
    let bounds = BatchedLinearBounds::identity(&[1, 1]).expect("small incoming bounds");

    let err = layer
        .propagate_linear_batched(&bounds, None)
        .expect_err("overflowing Conv2d input dimensions should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("input dims product overflows")),
        "expected input-dims overflow error, got: {err:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_new_rejects_non_4d_kernel() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0_f32]).expect("kernel");
    let err =
        ConvTranspose2dLayer::new(kernel, None, (1, 1), (0, 0)).expect_err("kernel must be 4D");
    assert!(matches!(err, NyError::ShapeMismatch { .. }));
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_ibp_single_channel_exact_positive_kernel() -> Result<()> {
    // y = 3x + 0.25 for each spatial position with a 1x1 kernel.
    let layer = make_convtranspose2d(3.0, Some(0.25), (2, 2));
    let input = BoundedTensor::new(
        array![[[1.0_f32, 2.0], [3.0, 4.0]]].into_dyn(),
        array![[[5.0_f32, 6.0], [7.0, 8.0]]].into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_close(output.lower()[[0, 0, 0]], 3.25, TOL);
    assert_close(output.lower()[[0, 0, 1]], 6.25, TOL);
    assert_close(output.lower()[[0, 1, 0]], 9.25, TOL);
    assert_close(output.lower()[[0, 1, 1]], 12.25, TOL);
    assert_close(output.upper()[[0, 0, 0]], 15.25, TOL);
    assert_close(output.upper()[[0, 0, 1]], 18.25, TOL);
    assert_close(output.upper()[[0, 1, 0]], 21.25, TOL);
    assert_close(output.upper()[[0, 1, 1]], 24.25, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_requires_input_shape() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).expect("kernel");
    let layer =
        ConvTranspose2dLayer::new(kernel, None, (1, 1), (0, 0)).expect("valid convtranspose2d");
    let bounds = LinearBounds::identity(4);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("missing input shape should fail");
    assert!(matches!(err, NyError::UnsupportedConfiguration(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_identity_bounds_maps_to_scaled_identity() -> Result<()> {
    let layer = make_convtranspose2d(3.0, Some(0.25), (2, 2));
    let bounds = LinearBounds::identity(4);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[4, 4]);
    assert_eq!(result.upper_a.shape(), &[4, 4]);
    for row in 0..4 {
        for col in 0..4 {
            let expected = if row == col { 3.0 } else { 0.0 };
            assert_close(result.lower_a[[row, col]], expected, TOL);
            assert_close(result.upper_a[[row, col]], expected, TOL);
        }
        assert_close(result.lower_b[row], 0.25, TOL);
        assert_close(result.upper_b[row], 0.25, TOL);
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_batched_identity_bounds_maps_to_scaled_identity() -> Result<()> {
    let layer = make_convtranspose2d(3.0, Some(0.25), (2, 2));
    let bounds = BatchedLinearBounds::identity(&[2, 4])?;
    let result = layer.propagate_linear_batched(&bounds)?;

    assert_eq!(result.lower_a.shape(), &[2, 4, 4]);
    assert_eq!(result.upper_a.shape(), &[2, 4, 4]);
    assert_eq!(result.lower_b.shape(), &[2, 4]);
    assert_eq!(result.upper_b.shape(), &[2, 4]);
    assert_eq!(result.input_shape, vec![2, 4]);

    for batch in 0..2 {
        for row in 0..4 {
            for col in 0..4 {
                let expected = if row == col { 3.0 } else { 0.0 };
                assert_close(result.lower_a[[batch, row, col]], expected, TOL);
                assert_close(result.upper_a[[batch, row, col]], expected, TOL);
            }
            assert_close(result.lower_b[[batch, row]], 0.25, TOL);
            assert_close(result.upper_b[[batch, row]], 0.25, TOL);
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_batched_input_dims_overflow_returns_error_3012() {
    let layer = make_convtranspose2d(1.0, None, (usize::MAX, 2));
    let bounds = BatchedLinearBounds::identity(&[1, 1]).expect("small incoming bounds");

    let err = layer
        .propagate_linear_batched(&bounds)
        .expect_err("overflowing ConvTranspose2d input dimensions should fail");

    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("input dims product overflows")),
        "expected input-dims overflow error, got: {err:?}"
    );
}

// ===== Non-trivial kernel tests =====

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_negative_kernel_swaps_bounds() -> Result<()> {
    // Negative weight: IBP must use W- splitting (lower from upper, upper from lower).
    // y = -2x + 1, kernel = [[-2]], bias = [1]
    let layer = make_conv2d(-2.0, Some(1.0), (2, 2));
    let input = BoundedTensor::new(
        array![[[1.0_f32, 2.0], [3.0, 4.0]]].into_dyn(),
        array![[[5.0_f32, 6.0], [7.0, 8.0]]].into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 2, 2]);
    // lower = -2 * upper + 1 = [-9, -11, -13, -15]
    assert_close(output.lower()[[0, 0, 0]], -9.0, TOL);
    assert_close(output.lower()[[0, 0, 1]], -11.0, TOL);
    assert_close(output.lower()[[0, 1, 0]], -13.0, TOL);
    assert_close(output.lower()[[0, 1, 1]], -15.0, TOL);
    // upper = -2 * lower + 1 = [-1, -3, -5, -7]
    assert_close(output.upper()[[0, 0, 0]], -1.0, TOL);
    assert_close(output.upper()[[0, 0, 1]], -3.0, TOL);
    assert_close(output.upper()[[0, 1, 0]], -5.0, TOL);
    assert_close(output.upper()[[0, 1, 1]], -7.0, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_4d_batched() -> Result<()> {
    // 4D batched IBP: (batch=2, in_c=1, h=2, w=2) with positive kernel
    let layer = make_conv2d(2.0, Some(0.5), (2, 2));
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 2, 2]),
        vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 2, 2]),
        vec![5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0],
    )
    .unwrap();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[2, 1, 2, 2]);
    // Batch 0: lower = 2*[1,2,3,4]+0.5 = [2.5,4.5,6.5,8.5]
    assert_close(output.lower()[[0, 0, 0, 0]], 2.5, TOL);
    assert_close(output.lower()[[0, 0, 0, 1]], 4.5, TOL);
    // Batch 0: upper = 2*[5,6,7,8]+0.5 = [10.5,12.5,14.5,16.5]
    assert_close(output.upper()[[0, 0, 0, 0]], 10.5, TOL);
    // Batch 1: lower = 2*[10,20,30,40]+0.5 = [20.5,40.5,60.5,80.5]
    assert_close(output.lower()[[1, 0, 0, 0]], 20.5, TOL);
    assert_close(output.lower()[[1, 0, 1, 0]], 60.5, TOL);
    // Batch 1: upper = 2*[50,60,70,80]+0.5 = [100.5,120.5,140.5,160.5]
    assert_close(output.upper()[[1, 0, 0, 0]], 100.5, TOL);
    assert_close(output.upper()[[1, 0, 1, 1]], 160.5, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_2x2_kernel_backward() -> Result<()> {
    // 2x2 kernel on 2x2 input → 1x1 output (stride 1, no padding).
    // Kernel K = [[1,2],[3,4]], 1 in-channel, 1 out-channel.
    // Forward: y[0,0] = K[0,0]*x[0,0] + K[0,1]*x[0,1] + K[1,0]*x[1,0] + K[1,1]*x[1,1]
    //        = 1*x[0,0] + 2*x[0,1] + 3*x[1,0] + 4*x[1,1]
    //
    // CROWN backward with identity A (1x1): backward = transposed_conv(I, K)
    // Since output is a single scalar = sum(K * input), the gradient w.r.t. input
    // is just the kernel weights: [1, 2, 3, 4] (flattened).
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap();
    let layer = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2)?;

    // Output dim = 1*1*1 = 1, input dim = 1*2*2 = 4
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    // Result should be (1x4) with the kernel weights as coefficients
    assert_eq!(result.lower_a.shape(), &[1, 4]);
    assert_eq!(result.upper_a.shape(), &[1, 4]);
    assert_close(result.lower_a[[0, 0]], 1.0, TOL); // K[0,0]
    assert_close(result.lower_a[[0, 1]], 2.0, TOL); // K[0,1]
    assert_close(result.lower_a[[0, 2]], 3.0, TOL); // K[1,0]
    assert_close(result.lower_a[[0, 3]], 4.0, TOL); // K[1,1]
                                                    // Conv is linear: upper_a must equal lower_a
    assert_close(result.upper_a[[0, 0]], 1.0, TOL);
    assert_close(result.upper_a[[0, 1]], 2.0, TOL);
    assert_close(result.upper_a[[0, 2]], 3.0, TOL);
    assert_close(result.upper_a[[0, 3]], 4.0, TOL);
    // No bias
    assert_close(result.lower_b[0], 0.0, TOL);
    assert_close(result.upper_b[0], 0.0, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_2x2_kernel_soundness() -> Result<()> {
    // Verify CROWN bounds contain all true outputs.
    // 2x2 kernel on 3x3 input → 2x2 output (stride 1, no padding).
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -1.0, 0.5, 2.0]).unwrap();
    let bias = array![0.25_f32];
    let layer = Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 3, 3)?;

    // in_dim = 1*3*3 = 9, out_dim = 1*2*2 = 4
    let out_dim = 4;
    let in_dim = 9;
    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    // Verify by sampling: for each corner of [lower, upper] cube,
    // compute conv output and check it's within CROWN bounds.
    let lower_vals: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();
    let upper_vals: Vec<f32> = (0..9).map(|i| i as f32 * 0.1 + 1.0).collect();
    let input_lower = ArrayD::from_shape_vec(IxDyn(&[1, 3, 3]), lower_vals.clone()).unwrap();
    let input_upper = ArrayD::from_shape_vec(IxDyn(&[1, 3, 3]), upper_vals.clone()).unwrap();
    let input_bt = BoundedTensor::new(input_lower, input_upper)?;

    // Evaluate IBP (concrete bounds)
    let ibp = layer.propagate_ibp(&input_bt)?;

    // Compute concrete CROWN bounds using interval arithmetic:
    // lower_bound[d] = min_{x in [l,u]} (A_lower[d,:] @ x + b_lower[d])
    //   = sum_j (a_j >= 0 ? a_j * l_j : a_j * u_j) + b_lower[d]
    // upper_bound[d] = max_{x in [l,u]} (A_upper[d,:] @ x + b_upper[d])
    //   = sum_j (a_j >= 0 ? a_j * u_j : a_j * l_j) + b_upper[d]
    let mut crown_lo = vec![0.0f32; out_dim];
    let mut crown_hi = vec![0.0f32; out_dim];
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
        crown_lo[d] = lo;
        crown_hi[d] = hi;
    }

    // For a linear layer (conv is linear), CROWN with identity incoming bounds
    // should give exactly the same bounds as IBP.
    for d in 0..out_dim {
        let ibp_lo = *ibp.lower().iter().nth(d).unwrap();
        let ibp_hi = *ibp.upper().iter().nth(d).unwrap();
        assert!(
            (crown_lo[d] - ibp_lo).abs() < 1e-4,
            "CROWN lower {} != IBP lower {} for output {}",
            crown_lo[d],
            ibp_lo,
            d
        );
        assert!(
            (crown_hi[d] - ibp_hi).abs() < 1e-4,
            "CROWN upper {} != IBP upper {} for output {}",
            crown_hi[d],
            ibp_hi,
            d
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_ibp_negative_kernel_swaps_bounds() -> Result<()> {
    // Negative weight: IBP must use W+/W- splitting correctly for transposed conv.
    let layer = make_convtranspose2d(-2.0, Some(1.0), (2, 2));
    let input = BoundedTensor::new(
        array![[[1.0_f32, 2.0], [3.0, 4.0]]].into_dyn(),
        array![[[5.0_f32, 6.0], [7.0, 8.0]]].into_dyn(),
    )?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 2, 2]);
    // lower = -2 * upper + 1 = [-9, -11, -13, -15]
    assert_close(output.lower()[[0, 0, 0]], -9.0, TOL);
    assert_close(output.lower()[[0, 0, 1]], -11.0, TOL);
    assert_close(output.lower()[[0, 1, 0]], -13.0, TOL);
    assert_close(output.lower()[[0, 1, 1]], -15.0, TOL);
    // upper = -2 * lower + 1 = [-1, -3, -5, -7]
    assert_close(output.upper()[[0, 0, 0]], -1.0, TOL);
    assert_close(output.upper()[[0, 0, 1]], -3.0, TOL);
    assert_close(output.upper()[[0, 1, 0]], -5.0, TOL);
    assert_close(output.upper()[[0, 1, 1]], -7.0, TOL);
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_2x2_kernel_backward() -> Result<()> {
    // ConvTranspose2d with 2x2 kernel on 2x2 input → 3x3 output (stride 1, no padding).
    // Kernel K = [[1,2],[3,4]], shape (in_c=1, out_c=1, 2, 2).
    //
    // CROWN backward of transposed conv = regular conv(A, K).
    // Output 3x3 from input 2x2 with 2x2 kernel means backward maps 9→4.
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, 2.0, 3.0, 4.0])
        .expect("valid shape");
    let layer = ConvTranspose2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2)?;

    let bounds = LinearBounds::identity(9);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[9, 4]);
    assert_eq!(result.upper_a.shape(), &[9, 4]);

    // ConvTranspose is linear: upper_a must equal lower_a
    for row in 0..9 {
        for col in 0..4 {
            assert_close(result.upper_a[[row, col]], result.lower_a[[row, col]], TOL);
        }
    }

    // y[0,0] = K[0,0]*x[0,0] = 1*x[0,0]
    assert_close(result.lower_a[[0, 0]], 1.0, TOL);
    assert_close(result.lower_a[[0, 1]], 0.0, TOL);
    assert_close(result.lower_a[[0, 2]], 0.0, TOL);
    assert_close(result.lower_a[[0, 3]], 0.0, TOL);

    // y[2,2] = K[1,1]*x[1,1] = 4*x[1,1]
    assert_close(result.lower_a[[8, 0]], 0.0, TOL);
    assert_close(result.lower_a[[8, 1]], 0.0, TOL);
    assert_close(result.lower_a[[8, 2]], 0.0, TOL);
    assert_close(result.lower_a[[8, 3]], 4.0, TOL);

    // y[1,1] = K[0,0]*x[1,1] + K[0,1]*x[1,0] + K[1,0]*x[0,1] + K[1,1]*x[0,0]
    //        = 1*x[1,1] + 2*x[1,0] + 3*x[0,1] + 4*x[0,0]
    assert_close(result.lower_a[[4, 0]], 4.0, TOL); // x[0,0]
    assert_close(result.lower_a[[4, 1]], 3.0, TOL); // x[0,1]
    assert_close(result.lower_a[[4, 2]], 2.0, TOL); // x[1,0]
    assert_close(result.lower_a[[4, 3]], 1.0, TOL); // x[1,1]
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_2x2_kernel_soundness() -> Result<()> {
    // Verify CROWN bounds match IBP for ConvTranspose2d (linear layer).
    // 2x2 kernel on 2x2 input → 3x3 output (stride 1, no padding).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -1.0, 0.5, 2.0])
        .expect("valid shape");
    let bias = array![0.25_f32];
    let layer = ConvTranspose2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 2, 2)?;

    let out_dim = 9;
    let in_dim = 4;
    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    let lower_vals: Vec<f32> = vec![-1.0, 0.5, -0.3, 2.0];
    let upper_vals: Vec<f32> = vec![1.0, 2.5, 0.7, 4.0];
    let input_lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), lower_vals.clone()).expect("valid shape");
    let input_upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), upper_vals.clone()).expect("valid shape");
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

// --- Regression tests for #2828: stride=0 and invalid spatial configs ---

/// Regression test for #2828: Conv2d stride=0 must be rejected by constructor.
#[ntest::timeout(10000)]
#[test]
fn conv2d_zero_stride_h_rejected() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), vec![1.0; 9]).expect("kernel");
    let err =
        Conv2dLayer::new(kernel, None, (0, 1), (0, 0)).expect_err("stride=(0,1) must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("stride")),
        "expected InvalidSpec about stride, got: {err:?}"
    );
}

/// Regression test for #2828: Conv2d stride=0 in width must be rejected.
#[ntest::timeout(10000)]
#[test]
fn conv2d_zero_stride_w_rejected() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), vec![1.0; 9]).expect("kernel");
    let err =
        Conv2dLayer::new(kernel, None, (1, 0), (0, 0)).expect_err("stride=(1,0) must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("stride")),
        "expected InvalidSpec about stride, got: {err:?}"
    );
}

/// Regression test for #2828: ConvTranspose2d stride=0 must be rejected.
#[ntest::timeout(10000)]
#[test]
fn conv_transpose2d_zero_stride_rejected() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), vec![1.0; 9]).expect("kernel");
    let err = ConvTranspose2dLayer::new(kernel, None, (0, 0), (0, 0))
        .expect_err("stride=(0,0) must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("stride")),
        "expected InvalidSpec about stride, got: {err:?}"
    );
}

/// Regression test for #2828: Conv2d output_size underflow (kernel > padded input).
#[ntest::timeout(10000)]
#[test]
fn conv2d_output_size_underflow_returns_error() -> Result<()> {
    // kernel 5x5, input 2x2, no padding → padded = 2 < 5 → underflow
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 5, 5]), vec![1.0; 25]).expect("kernel");
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0))?;
    let err = conv
        .output_size(2, 2)
        .expect_err("kernel > padded input must error");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("kernel")),
        "expected InvalidSpec about kernel size, got: {err:?}"
    );
    Ok(())
}

/// Regression test for #2828: Conv2d valid stride=(1,1) is accepted.
#[ntest::timeout(10000)]
#[test]
fn conv2d_stride_one_accepted() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 3, 3]), vec![1.0; 9]).expect("kernel");
    Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).expect("stride=(1,1) must be accepted");
}

/// Regression test for #2877: IBP forward with kernel > padded input returns error, not panic.
#[ntest::timeout(10000)]
#[test]
fn conv2d_ibp_kernel_oversized_returns_error() -> Result<()> {
    // kernel=(5,5), input=(2,2), padding=(0,0) → padded=(2,2) < (5,5) → conv2d_single underflow.
    // Before #2877 fix, this caused usize underflow panic in conv2d_single.
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 5, 5]), vec![1.0; 25]).expect("kernel");
    let conv = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 2, 2)?;
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.0_f32; 4]).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0_f32; 4]).expect("upper"),
    )?;
    let err = conv.propagate_ibp(&input);
    assert!(
        err.is_err(),
        "Expected error for oversized kernel, got {:?}",
        err
    );
    Ok(())
}

/// Regression test for #2747: Conv2d CROWN backward with NaN kernel returns
/// NumericalInstability error instead of silently producing NaN coefficients.
#[ntest::timeout(10000)]
#[test]
fn conv2d_crown_backward_nan_kernel_returns_error() {
    let layer = make_conv2d(f32::NAN, None, (2, 2));
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "NaN kernel should return NumericalInstability, got {:?}",
        result
    );
}

/// Regression test for #2747: Conv2d CROWN backward with NaN bias returns
/// NumericalInstability error.
#[ntest::timeout(10000)]
#[test]
fn conv2d_crown_backward_nan_bias_returns_error() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0_f32]).expect("kernel");
    let bias = Some(array![f32::NAN]);
    let layer =
        Conv2dLayer::with_input_shape(kernel, bias, (1, 1), (0, 0), 2, 2).expect("valid conv2d");
    let (out_c, out_h, out_w) = (1_usize, 2, 2);
    let bounds = LinearBounds::identity(out_c * out_h * out_w);
    let result = layer.propagate_linear(&bounds);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "NaN bias should return NumericalInstability, got {:?}",
        result
    );
}

/// Regression test for #2747: ConvTranspose2d CROWN backward with NaN kernel
/// returns NumericalInstability error.
#[ntest::timeout(10000)]
#[test]
fn convtranspose2d_crown_backward_nan_kernel_returns_error() {
    let layer = make_convtranspose2d(f32::NAN, None, (2, 2));
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear(&bounds);
    assert!(
        matches!(result, Err(NyError::NumericalInstability(_))),
        "NaN kernel should return NumericalInstability, got {:?}",
        result
    );
}

/// Regression test for #3030: Conv2d IBP with overflow-producing inputs returns Ok
/// with the overflowed ±Inf endpoints preserved, not NumericalInstability error.
///
/// Before #3030, conv2d lacked NaN repair and would hard-fail via
/// `BoundedTensor::new()` rejecting Inf output. Now it matches linear layer behavior.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_overflow_preserved_as_inf() {
    // Large weight × extreme bounds → Inf in convolution output
    let conv = make_conv2d(1e20, None, (1, 1));
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1e20_f32]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![2e20_f32]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let result = conv.propagate_ibp(&input);
    assert!(
        result.is_ok(),
        "Conv2d IBP should widen overflow, not hard-fail: {:?}",
        result
    );

    let output = result.unwrap();
    // Overflowed endpoints pass through as +Inf: a non-finite endpoint carries
    // no proven bound in that direction, so any finite substitute
    // (FALLBACK_BOUND included) would be an unsound tightening.
    for &v in output.lower().iter() {
        assert!(!v.is_nan(), "lower bound must not be NaN, got {}", v);
    }
    for &v in output.upper().iter() {
        assert_eq!(
            v,
            f32::INFINITY,
            "overflowed upper should stay +Inf, got {}",
            v
        );
    }
}

/// Assert row has non-finite fallback: zeroed A, ±Inf bias (#2812).
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

/// Build LinearBounds: row 0 has 1e19 coefficients (overflow), row 1 has 1.0 (safe).
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

/// Regression (#2812, #3228): Conv2d CROWN backward per-row coefficient magnitude fallback.
/// Weight 1e5 * row 0 coeff 1e19 → 1e24 exceeds CROWN_COEFF_MAX, triggers fallback.
/// Weight 1e5 * row 1 coeff 1.0 → 1e5 stays below CROWN_COEFF_MAX, remains safe.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_backward_nonfinite_row_fallback() -> Result<()> {
    let layer = make_conv2d(1e5, Some(1.0), (2, 2));
    let (out_c, out_h, out_w) = (1, 2, 2);
    let bounds = make_overflow_bounds(out_c * out_h * out_w);
    let lb = layer
        .propagate_linear(&bounds)
        .expect("should handle overflow via row fallback")
        .into_owned();
    assert_nonfinite_row_fallback(&lb, 0);
    assert_finite_row(&lb, 1);
    Ok(())
}

// ===== Batched CROWN tests for ConvTranspose2d (non-trivial kernel) =====

#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_batched_2x2_kernel_soundness() -> Result<()> {
    // Verify batched CROWN bounds match IBP for ConvTranspose2d (linear layer).
    // 2x2 kernel on 2x2 input → 3x3 output (stride 1, no padding).
    //
    // For linear layers, batched CROWN with identity A must concretize
    // to the same bounds as IBP. This tests the batched path in
    // ConvTranspose2dLayer::propagate_linear_batched (types.rs lines 492+).
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0_f32, -1.0, 0.5, 2.0])
        .expect("valid shape");
    let bias = array![0.25_f32];
    let layer = ConvTranspose2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (0, 0), 2, 2)?;

    let out_dim = 9; // 3x3 output flattened
    let in_dim = 4; // 2x2 input flattened
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
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), lower_vals.clone()).expect("valid shape");
    let input_upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), upper_vals.clone()).expect("valid shape");
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
fn test_convtranspose2d_batched_crown_engine_matches_cpu_3622() -> Result<()> {
    let layer = make_convtranspose2d(3.0, Some(0.25), (2, 2));
    let bounds = BatchedLinearBounds::identity(&[2, 4])?;
    let expected = layer.propagate_linear_batched(&bounds)?;
    let engine = CountingGemmEngine::new();
    let actual = layer.propagate_linear_batched_maybe_engine(&bounds, Some(&engine))?;

    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "#3622 regression: ConvTranspose2d batched CROWN should invoke GemmEngine, got {calls} calls"
    );
    assert_batched_bounds_close(&actual, &expected, TOL, "conv_transpose2d_gemm");
    Ok(())
}

/// Regression (#2812, #3228): ConvTranspose2d CROWN backward per-row coefficient magnitude fallback.
/// Weight 1e5 * row 0 coeff 1e19 → 1e24 exceeds CROWN_COEFF_MAX, triggers fallback.
/// Weight 1e5 * row 1 coeff 1.0 → 1e5 stays below CROWN_COEFF_MAX, remains safe.
#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_crown_backward_nonfinite_row_fallback() -> Result<()> {
    let layer = make_convtranspose2d(1e5, Some(1.0), (2, 2));
    let (out_c, out_h, out_w) = (1, 2, 2);
    let bounds = make_overflow_bounds(out_c * out_h * out_w);
    let lb = layer
        .propagate_linear(&bounds)
        .expect("should handle overflow via row fallback")
        .into_owned();
    assert_nonfinite_row_fallback(&lb, 0);
    assert_finite_row(&lb, 1);
    Ok(())
}

// ===== GEMM-based conv2d_transpose tests (#3382) =====

/// Verify conv2d_transpose_batched_gemm matches per-row conv2d_transpose for a 3x3 kernel.
///
/// This tests the mathematical equivalence of the GEMM+col2im path (single faer matmul)
/// against the reference scalar conv2d_transpose (6-deep nested loop) called per-row.
/// Design doc: designs/2026-03-06-conv-crown-backward-gemm.md (#3382).
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_transpose_batched_gemm_matches_scalar_3x3_kernel() -> Result<()> {
    use ndarray::Array2;

    // Conv layer: 2 out_channels, 1 in_channel, 3x3 kernel, stride=1, padding=1.
    // Input: 1 in_channel, 4x4 spatial → output: 2 out_channels, 4x4 spatial.
    let out_c = 2;
    let in_c = 1;
    let kh = 3;
    let kw = 3;
    let in_h = 4;
    let in_w = 4;
    let stride = (1, 1);
    let padding = (1, 1);
    let out_h = (in_h + 2 * padding.0 - kh) / stride.0 + 1; // 4
    let out_w = (in_w + 2 * padding.1 - kw) / stride.1 + 1; // 4

    // Kernel with non-trivial values.
    let kernel_data: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|i| i as f32 * 0.1 - 0.8)
        .collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), kernel_data).unwrap();

    // A-coefficient matrix: 3 objectives × (out_c * out_h * out_w) = 3 × 32.
    let num_objectives = 3;
    let a_cols = out_c * out_h * out_w;
    let a_data: Vec<f32> = (0..num_objectives * a_cols)
        .map(|i| (i as f32 * 0.37).sin() * 2.0)
        .collect();
    let a_coefficients = Array2::from_shape_vec((num_objectives, a_cols), a_data).unwrap();

    // Reference: per-row scalar conv2d_transpose.
    let conv_in_size = in_c * in_h * in_w;
    let mut reference = Array2::<f32>::zeros((num_objectives, conv_in_size));
    for row_idx in 0..num_objectives {
        let row = a_coefficients.row(row_idx);
        let row_3d = ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), row.to_vec()).unwrap();
        let trans = conv2d_transpose(&row_3d, &kernel, stride, padding, (1, 1), (in_h, in_w))?;
        for (i, &val) in trans.iter().enumerate() {
            reference[[row_idx, i]] = val;
        }
    }

    // GEMM path (CPU — no engine).
    let gemm_result = conv2d_transpose_batched_gemm(
        &a_coefficients,
        &kernel,
        stride,
        padding,
        (1, 1),
        (in_h, in_w),
        (out_h, out_w),
        out_c,
        None,
    )?;

    // Compare: f32 accumulation order differs, so allow ~1e-5 relative error.
    assert_eq!(gemm_result.shape(), reference.shape());
    for obj in 0..num_objectives {
        for j in 0..conv_in_size {
            let expected = reference[[obj, j]];
            let actual = gemm_result[[obj, j]];
            let abs_diff = (expected - actual).abs();
            let rel_diff = if expected.abs() > 1e-8 {
                abs_diff / expected.abs()
            } else {
                abs_diff
            };
            assert!(
                rel_diff < 1e-4,
                "Mismatch at obj={obj}, j={j}: expected={expected}, got={actual}, \
                 abs_diff={abs_diff}, rel_diff={rel_diff}"
            );
        }
    }
    Ok(())
}

/// Verify GEMM path produces correct CROWN backward for multi-channel Conv2d.
///
/// Tests the full `propagate_linear` path (which now uses GEMM internally)
/// against IBP for a multi-channel conv with 3x3 kernel and padding.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_backward_multichannel_3x3_matches_ibp() -> Result<()> {
    // 2 out_channels, 1 in_channel, 3x3 kernel, stride=1, padding=1, 4x4 input.
    let out_c = 2;
    let in_c = 1;
    let (kh, kw) = (3, 3);
    let (in_h, in_w) = (4, 4);

    let kernel_data: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|i| i as f32 * 0.15 - 0.6)
        .collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), kernel_data).unwrap();
    let bias = array![0.1_f32, -0.2];
    let layer = Conv2dLayer::with_input_shape(kernel, Some(bias), (1, 1), (1, 1), in_h, in_w)?;

    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    let out_dim = out_c * out_h * out_w;
    let in_dim = in_c * in_h * in_w;

    // CROWN backward with identity A.
    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();
    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    // Concretize and compare against IBP.
    let lower_vals: Vec<f32> = (0..in_dim).map(|i| i as f32 * 0.1 - 0.5).collect();
    let upper_vals: Vec<f32> = (0..in_dim).map(|i| i as f32 * 0.1 + 0.5).collect();
    let input_lower =
        ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), lower_vals.clone()).unwrap();
    let input_upper =
        ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), upper_vals.clone()).unwrap();
    let input_bt = BoundedTensor::new(input_lower, input_upper)?;
    let ibp = layer.propagate_ibp(&input_bt)?;

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
            "CROWN lower {lo} != IBP lower {ibp_lo} for output {d}"
        );
        assert!(
            (hi - ibp_hi).abs() < 1e-3,
            "CROWN upper {hi} != IBP upper {ibp_hi} for output {d}"
        );
    }
    Ok(())
}

// ===== Grouped convolution tests (#3770) =====

/// Helper to verify CROWN bounds match IBP for a Conv2dLayer.
/// For a linear layer, CROWN with identity A concretized on [lower, upper]
/// must equal IBP bounds.
fn assert_crown_matches_ibp(
    layer: &Conv2dLayer,
    lower_vals: &[f32],
    upper_vals: &[f32],
    in_c: usize,
    in_h: usize,
    in_w: usize,
    tol: f64,
) -> Result<()> {
    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    let out_c = layer.out_channels();
    let out_dim = out_c * out_h * out_w;
    let in_dim = in_c * in_h * in_w;

    let bounds = LinearBounds::identity(out_dim);
    let result = layer.propagate_linear(&bounds)?.into_owned();
    assert_eq!(result.lower_a.shape(), &[out_dim, in_dim]);

    let input_lower =
        ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), lower_vals.to_vec()).unwrap();
    let input_upper =
        ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), upper_vals.to_vec()).unwrap();
    let input_bt = BoundedTensor::new(input_lower, input_upper)?;
    let ibp = layer.propagate_ibp(&input_bt)?;

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
            (lo - ibp_lo).abs() < tol,
            "CROWN lower {lo} != IBP lower {ibp_lo} for output {d}"
        );
        assert!(
            (hi - ibp_hi).abs() < tol,
            "CROWN upper {hi} != IBP upper {ibp_hi} for output {d}"
        );
    }
    Ok(())
}

/// #3770: IBP forward with groups=2, 1x1 kernel.
///
/// Kernel shape: (4 out_c, 3 in_c_per_group, 1, 1), groups=2.
/// Total in_c = 3*2 = 6. Group 0 maps in_c[0..3] → out_c[0..2],
/// group 1 maps in_c[3..6] → out_c[2..4].
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_grouped_two_groups_1x1() -> Result<()> {
    // groups=2, out_c=4 (2 per group), in_c_per_group=3, total in_c=6
    let groups = 2;
    let out_c = 4;
    let in_c_per_group = 3;
    let (kh, kw) = (1, 1);

    // Kernel: each weight = 1.0 for simplicity.
    let kernel = ArrayD::from_elem(IxDyn(&[out_c, in_c_per_group, kh, kw]), 1.0_f32);
    let bias = array![0.0_f32, 0.0, 0.0, 0.0];
    let in_c = in_c_per_group * groups; // 6
    let (in_h, in_w) = (2, 2);

    let layer =
        Conv2dLayer::with_input_shape_full(kernel, Some(bias), (1, 1), (0, 0), groups, in_h, in_w)?;
    assert_eq!(layer.in_channels(), in_c);
    assert_eq!(layer.out_channels(), out_c);
    assert_eq!(layer.groups, groups);

    // Input: lower all 1.0, upper all 2.0. Shape: (6, 2, 2).
    let lower = ArrayD::from_elem(IxDyn(&[in_c, in_h, in_w]), 1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[in_c, in_h, in_w]), 2.0_f32);
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[out_c, in_h, in_w]);
    // Each output channel sums in_c_per_group=3 input channels with weight 1.0.
    // lower_y = 3 * 1.0 = 3.0, upper_y = 3 * 2.0 = 6.0
    for oc in 0..out_c {
        for h in 0..in_h {
            for w in 0..in_w {
                assert_close(output.lower()[[oc, h, w]], 3.0, TOL);
                assert_close(output.upper()[[oc, h, w]], 6.0, TOL);
            }
        }
    }
    Ok(())
}

/// #3770: IBP forward verifies group isolation — no cross-group leakage.
///
/// Group 0 has weight 1.0, group 1 has weight -1.0.
/// If groups are incorrect, cross-group contamination would produce wrong bounds.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_grouped_isolation() -> Result<()> {
    // groups=2, out_c=2 (1 per group), in_c_per_group=2, total in_c=4
    let groups = 2;
    let out_c = 2;
    let in_c_per_group = 2;
    let (kh, kw) = (1, 1);

    // Group 0 kernel: all 1.0. Group 1 kernel: all -1.0.
    // Kernel shape: (2, 2, 1, 1). [oc=0] = 1.0, [oc=1] = -1.0
    let mut kernel = ArrayD::from_elem(IxDyn(&[out_c, in_c_per_group, kh, kw]), 1.0_f32);
    kernel[[1, 0, 0, 0]] = -1.0;
    kernel[[1, 1, 0, 0]] = -1.0;

    let in_c = in_c_per_group * groups; // 4
    let (in_h, in_w) = (1, 1);

    let layer =
        Conv2dLayer::with_input_shape_full(kernel, None, (1, 1), (0, 0), groups, in_h, in_w)?;

    // Input: lower=[1,2,3,4], upper=[5,6,7,8] across channels, spatial 1x1.
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[out_c, in_h, in_w]);
    // Group 0: out_c=0, weight=+1.0, in_c=[0,1]. lower = 1+2 = 3, upper = 5+6 = 11.
    assert_close(output.lower()[[0, 0, 0]], 3.0, TOL);
    assert_close(output.upper()[[0, 0, 0]], 11.0, TOL);
    // Group 1: out_c=1, weight=-1.0, in_c=[2,3]. lower = -1*(7+8) = -15, upper = -1*(3+4) = -7.
    assert_close(output.lower()[[1, 0, 0]], -15.0, TOL);
    assert_close(output.upper()[[1, 0, 0]], -7.0, TOL);
    Ok(())
}

/// #3770: CROWN backward soundness for groups=2 with 3x3 kernel.
///
/// Verifies CROWN bounds concretized on [lower, upper] match IBP.
/// This exercises the grouped GEMM path in conv2d_transpose_batched_gemm_grouped.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_grouped_two_groups_soundness() -> Result<()> {
    let groups = 2;
    let out_c = 4; // 2 per group
    let in_c_per_group = 2;
    let in_c = in_c_per_group * groups; // 4
    let (kh, kw) = (3, 3);
    let (in_h, in_w) = (4, 4);

    let kernel_data: Vec<f32> = (0..out_c * in_c_per_group * kh * kw)
        .map(|i| (i as f32 * 0.23).sin() * 0.5)
        .collect();
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[out_c, in_c_per_group, kh, kw]), kernel_data).unwrap();
    let bias = array![0.1_f32, -0.2, 0.05, -0.15];
    let layer =
        Conv2dLayer::with_input_shape_full(kernel, Some(bias), (1, 1), (1, 1), groups, in_h, in_w)?;

    let in_dim = in_c * in_h * in_w;
    let lower_vals: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.17).cos() * 0.3).collect();
    let upper_vals: Vec<f32> = lower_vals.iter().map(|&v| v + 1.0).collect();

    assert_crown_matches_ibp(&layer, &lower_vals, &upper_vals, in_c, in_h, in_w, 1e-3)
}

/// #3770: CROWN backward correctness for groups=2, verifying A-coefficients.
///
/// With groups=2 and 1x1 kernel, the CROWN backward should produce a block-diagonal
/// A matrix: group 0 channels only depend on group 0 input channels, and vice versa.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_grouped_block_diagonal() -> Result<()> {
    let groups = 2;
    let out_c = 2; // 1 per group
    let in_c_per_group = 2;
    let (kh, kw) = (1, 1);
    let (in_h, in_w) = (1, 1);

    // Group 0 kernel: [[2.0], [3.0]], Group 1 kernel: [[5.0], [7.0]]
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[out_c, in_c_per_group, kh, kw]),
        vec![2.0_f32, 3.0, 5.0, 7.0],
    )
    .unwrap();
    let layer =
        Conv2dLayer::with_input_shape_full(kernel, None, (1, 1), (0, 0), groups, in_h, in_w)?;

    // out_dim = 2*1*1 = 2, in_dim = 4*1*1 = 4
    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear(&bounds)?.into_owned();

    assert_eq!(result.lower_a.shape(), &[2, 4]);
    // Row 0 (out_c=0, group 0): depends on in_c[0,1] with weights [2,3], zero for in_c[2,3]
    assert_close(result.lower_a[[0, 0]], 2.0, TOL);
    assert_close(result.lower_a[[0, 1]], 3.0, TOL);
    assert_close(result.lower_a[[0, 2]], 0.0, TOL);
    assert_close(result.lower_a[[0, 3]], 0.0, TOL);
    // Row 1 (out_c=1, group 1): depends on in_c[2,3] with weights [5,7], zero for in_c[0,1]
    assert_close(result.lower_a[[1, 0]], 0.0, TOL);
    assert_close(result.lower_a[[1, 1]], 0.0, TOL);
    assert_close(result.lower_a[[1, 2]], 5.0, TOL);
    assert_close(result.lower_a[[1, 3]], 7.0, TOL);
    // Conv is linear: upper_a == lower_a
    for row in 0..2 {
        for col in 0..4 {
            assert_close(result.upper_a[[row, col]], result.lower_a[[row, col]], TOL);
        }
    }
    Ok(())
}

/// #3770: Depthwise convolution (groups=in_channels) IBP forward.
///
/// Depthwise separable convolution: each input channel has its own kernel.
/// Kernel shape: (4, 1, 3, 3) with groups=4 means each channel is independent.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_depthwise() -> Result<()> {
    let groups = 4;
    let out_c = 4; // 1 per group, same as groups for depthwise
    let in_c_per_group = 1;
    let in_c = in_c_per_group * groups; // 4
    let (kh, kw) = (3, 3);
    let (in_h, in_w) = (3, 3);

    // Each channel has distinct kernel weights.
    // Channel 0: all 1.0, Channel 1: all 0.5, Channel 2: all -1.0, Channel 3: all 2.0
    let mut kernel_data = vec![0.0_f32; out_c * in_c_per_group * kh * kw];
    for (channel, value) in [1.0_f32, 0.5, -1.0, 2.0].into_iter().enumerate() {
        for i in 0..9 {
            kernel_data[channel * 9 + i] = value;
        }
    }
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[out_c, in_c_per_group, kh, kw]), kernel_data).unwrap();

    let layer =
        Conv2dLayer::with_input_shape_full(kernel, None, (1, 1), (1, 1), groups, in_h, in_w)?;
    assert_eq!(layer.in_channels(), 4);

    // Input: all channels = 1.0 lower, 2.0 upper. Shape: (4, 3, 3).
    let lower = ArrayD::from_elem(IxDyn(&[in_c, in_h, in_w]), 1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[in_c, in_h, in_w]), 2.0_f32);
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    assert_eq!(output.shape(), &[out_c, out_h, out_w]);

    // Center pixel at (1,1): all 9 kernel elements contribute.
    // Ch 0: weight=1.0, lower = 9*1.0 = 9.0, upper = 9*2.0 = 18.0
    assert_close(output.lower()[[0, 1, 1]], 9.0, TOL);
    assert_close(output.upper()[[0, 1, 1]], 18.0, TOL);
    // Ch 1: weight=0.5, lower = 9*0.5*1.0 = 4.5, upper = 9*0.5*2.0 = 9.0
    assert_close(output.lower()[[1, 1, 1]], 4.5, TOL);
    assert_close(output.upper()[[1, 1, 1]], 9.0, TOL);
    // Ch 2: weight=-1.0, lower = 9*(-1.0)*2.0 = -18.0, upper = 9*(-1.0)*1.0 = -9.0
    assert_close(output.lower()[[2, 1, 1]], -18.0, TOL);
    assert_close(output.upper()[[2, 1, 1]], -9.0, TOL);
    // Ch 3: weight=2.0, lower = 9*2.0*1.0 = 18.0, upper = 9*2.0*2.0 = 36.0
    assert_close(output.lower()[[3, 1, 1]], 18.0, TOL);
    assert_close(output.upper()[[3, 1, 1]], 36.0, TOL);
    Ok(())
}

/// #3770: Depthwise convolution (groups=in_channels) CROWN backward soundness.
///
/// Verifies CROWN bounds match IBP for depthwise separable convolution.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_crown_depthwise_soundness() -> Result<()> {
    let groups = 4;
    let out_c = 4;
    let in_c_per_group = 1;
    let in_c = in_c_per_group * groups;
    let (kh, kw) = (3, 3);
    let (in_h, in_w) = (4, 4);

    let kernel_data: Vec<f32> = (0..out_c * in_c_per_group * kh * kw)
        .map(|i| (i as f32 * 0.31).sin() * 0.7)
        .collect();
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[out_c, in_c_per_group, kh, kw]), kernel_data).unwrap();
    let bias = array![0.1_f32, -0.1, 0.2, -0.2];
    let layer =
        Conv2dLayer::with_input_shape_full(kernel, Some(bias), (1, 1), (1, 1), groups, in_h, in_w)?;

    let in_dim = in_c * in_h * in_w;
    let lower_vals: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.13).cos() * 0.5).collect();
    let upper_vals: Vec<f32> = lower_vals.iter().map(|&v| v + 1.0).collect();

    assert_crown_matches_ibp(&layer, &lower_vals, &upper_vals, in_c, in_h, in_w, 1e-3)
}

/// #3770: Batched CROWN backward with groups=2 matches CPU path.
///
/// Tests the `propagate_linear_batched` GEMM path with grouped convolution.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_batched_crown_grouped_engine_matches_cpu() -> Result<()> {
    let groups = 2;
    let out_c = 4;
    let in_c_per_group = 2;
    let (kh, kw) = (3, 3);
    let (in_h, in_w) = (4, 4);

    let kernel_data: Vec<f32> = (0..out_c * in_c_per_group * kh * kw)
        .map(|i| (i as f32 * 0.19).sin() * 0.4)
        .collect();
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[out_c, in_c_per_group, kh, kw]), kernel_data).unwrap();
    let bias = array![0.1_f32, -0.1, 0.05, -0.05];
    let layer =
        Conv2dLayer::with_input_shape_full(kernel, Some(bias), (1, 1), (1, 1), groups, in_h, in_w)?;

    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    let out_dim = out_c * out_h * out_w;
    let batch_size = 2;

    let bounds = BatchedLinearBounds::identity(&[batch_size, out_dim])?;
    // CPU path (no engine)
    let expected = layer.propagate_linear_batched(&bounds, None)?;
    // Engine path
    let engine = CountingGemmEngine::new();
    let actual = layer.propagate_linear_batched(&bounds, Some(&engine))?;

    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "#3770 regression: grouped Conv2d batched CROWN should invoke GemmEngine, got {calls} calls"
    );
    assert_batched_bounds_close(&actual, &expected, TOL, "grouped_conv2d_gemm");
    Ok(())
}

/// #3770: GEMM grouped transpose matches scalar grouped transpose.
///
/// Verifies conv2d_transpose_batched_gemm_grouped produces the same results
/// as per-row conv2d_transpose_grouped for groups=2.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_transpose_batched_gemm_grouped_matches_scalar() -> Result<()> {
    use super::conv2d_transpose_grouped;
    use ndarray::Array2;

    let groups = 2;
    let out_c = 4; // 2 per group
    let in_c_per_group = 2;
    let (kh, kw) = (3, 3);
    let (in_h, in_w) = (4, 4);
    let stride = (1, 1);
    let padding = (1, 1);
    let out_h = (in_h + 2 * padding.0 - kh) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kw) / stride.1 + 1;

    let kernel_data: Vec<f32> = (0..out_c * in_c_per_group * kh * kw)
        .map(|i| (i as f32 * 0.29).sin() * 0.6)
        .collect();
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[out_c, in_c_per_group, kh, kw]), kernel_data).unwrap();

    // A-coefficient matrix: 3 objectives × (out_c * out_h * out_w)
    let num_objectives = 3;
    let a_cols = out_c * out_h * out_w;
    let a_data: Vec<f32> = (0..num_objectives * a_cols)
        .map(|i| (i as f32 * 0.41).sin() * 1.5)
        .collect();
    let a_coefficients = Array2::from_shape_vec((num_objectives, a_cols), a_data).unwrap();

    // Reference: per-row scalar conv2d_transpose_grouped.
    let total_in_c = in_c_per_group * groups;
    let conv_in_size = total_in_c * in_h * in_w;
    let mut reference = Array2::<f32>::zeros((num_objectives, conv_in_size));
    for row_idx in 0..num_objectives {
        let row = a_coefficients.row(row_idx);
        let row_3d = ArrayD::from_shape_vec(IxDyn(&[out_c, out_h, out_w]), row.to_vec()).unwrap();
        let trans = conv2d_transpose_grouped(
            &row_3d,
            &kernel,
            stride,
            padding,
            (1, 1),
            (in_h, in_w),
            groups,
        )?;
        for (i, &val) in trans.iter().enumerate() {
            reference[[row_idx, i]] = val;
        }
    }

    // GEMM path with groups.
    let gemm_result = conv2d_transpose_batched_gemm_grouped(
        &a_coefficients,
        &kernel,
        stride,
        padding,
        (1, 1),
        (in_h, in_w),
        (out_h, out_w),
        out_c,
        groups,
        1,
        None,
    )?;

    assert_eq!(gemm_result.shape(), reference.shape());
    for obj in 0..num_objectives {
        for j in 0..conv_in_size {
            let expected = reference[[obj, j]];
            let actual = gemm_result[[obj, j]];
            let abs_diff = (expected - actual).abs();
            let rel_diff = if expected.abs() > 1e-8 {
                abs_diff / expected.abs()
            } else {
                abs_diff
            };
            // Scalar and GEMM paths accumulate in different order, allow ~5e-4 relative error.
            assert!(
                rel_diff < 5e-4,
                "Mismatch at obj={obj}, j={j}: expected={expected}, got={actual}, \
                 abs_diff={abs_diff}, rel_diff={rel_diff}"
            );
        }
    }
    Ok(())
}

/// Verify deadline-aware path with NaiveCpuGemmEngine (fused conv_transpose_2d)
/// produces the same result as the non-deadline path without engine.
///
/// This triggers the `use_chunked_deadline` branch with total_spatial > 256,
/// exercising the fused GPU code path added in Part of #3813. The CPU engine
/// serves as a fused-path reference since its conv_transpose_2d does GEMM + col2im.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_transpose_deadline_fused_matches_nonfused() -> Result<()> {
    use ndarray::Array2;
    use ny_core::NaiveCpuGemmEngine;
    use std::time::{Duration, Instant};

    // Conv layer: 2 out_channels, 1 in_channel, 3x3 kernel, stride=1, padding=1.
    // 4x4 spatial → 4x4 output.
    let out_c = 2;
    let in_c = 1;
    let (kh, kw) = (3, 3);
    let (in_h, in_w) = (4, 4);
    let stride = (1, 1);
    let padding = (1, 1);
    let out_h = (in_h + 2 * padding.0 - kh) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kw) / stride.1 + 1;

    let kernel_data: Vec<f32> = (0..out_c * in_c * kh * kw)
        .map(|i| i as f32 * 0.1 - 0.8)
        .collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[out_c, in_c, kh, kw]), kernel_data).unwrap();

    // Use enough objectives so total_spatial > DEADLINE_GEMM_ROW_CHUNK (256).
    // spatial_per_obj = 4 * 4 = 16, so 17 objectives → 272 total_spatial > 256.
    let num_objectives = 17;
    let a_cols = out_c * out_h * out_w;
    let a_data: Vec<f32> = (0..num_objectives * a_cols)
        .map(|i| (i as f32 * 0.37).sin() * 2.0)
        .collect();
    let a_coefficients = Array2::from_shape_vec((num_objectives, a_cols), a_data).unwrap();

    // Reference: non-deadline path without engine.
    let reference = conv2d_transpose_batched_gemm(
        &a_coefficients,
        &kernel,
        stride,
        padding,
        (1, 1),
        (in_h, in_w),
        (out_h, out_w),
        out_c,
        None,
    )?;

    // Deadline-aware path with NaiveCpuGemmEngine (exercises fused conv_transpose_2d).
    let engine = NaiveCpuGemmEngine;
    let far_deadline = Instant::now() + Duration::from_mins(1);
    let fused_result = conv2d_transpose_batched_gemm_grouped_with_deadline(
        &a_coefficients,
        &kernel,
        stride,
        padding,
        (1, 1),
        (in_h, in_w),
        (out_h, out_w),
        out_c,
        1,
        1,
        Some(&engine as &dyn GemmEngine),
        Some(far_deadline),
    )?;

    assert_eq!(fused_result.shape(), reference.shape());
    let conv_in_size = in_c * in_h * in_w;
    for obj in 0..num_objectives {
        for j in 0..conv_in_size {
            let expected = reference[[obj, j]];
            let actual = fused_result[[obj, j]];
            let abs_diff = (expected - actual).abs();
            let rel_diff = if expected.abs() > 1e-8 {
                abs_diff / expected.abs()
            } else {
                abs_diff
            };
            assert!(
                rel_diff < 1e-4,
                "Deadline fused mismatch at obj={obj}, j={j}: expected={expected}, got={actual}, \
                 abs_diff={abs_diff}, rel_diff={rel_diff}"
            );
        }
    }
    Ok(())
}

/// #3770: Constructor rejects groups=0.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_new_full_rejects_groups_zero() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![1.0_f32; 2]).unwrap();
    let err = Conv2dLayer::new_full(kernel, None, (1, 1), (0, 0), 0)
        .expect_err("groups=0 must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("groups")),
        "expected InvalidSpec about groups, got: {err:?}"
    );
}

/// #3770: Constructor rejects out_channels not divisible by groups.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_new_full_rejects_out_c_not_divisible_by_groups() {
    // out_c=3, groups=2: 3 % 2 != 0
    let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1, 1]), vec![1.0_f32; 3]).unwrap();
    let err = Conv2dLayer::new_full(kernel, None, (1, 1), (0, 0), 2)
        .expect_err("out_channels not divisible by groups must be rejected");
    assert!(
        matches!(err, NyError::InvalidSpec(ref msg) if msg.contains("divisible")),
        "expected InvalidSpec about divisibility, got: {err:?}"
    );
}

// ── #4238: IBP GEMM parity tests ──────────────────────────────────────────────
//
// Verify propagate_ibp_with_engine (GEMM path via ops_ibp_gemm.rs) produces
// identical bounds to propagate_ibp (CPU path) across varying kernel sizes,
// strides, padding, bias, and input dimensionality.

/// Helper: assert parity between CPU IBP and GEMM IBP paths.
fn assert_ibp_gemm_parity(layer: &Conv2dLayer, input: &BoundedTensor, label: &str) {
    use ny_core::NaiveCpuGemmEngine;
    let cpu = layer.propagate_ibp(input).expect("CPU IBP should succeed");
    let engine = NaiveCpuGemmEngine;
    let gemm = layer
        .propagate_ibp_with_engine(input, Some(&engine))
        .expect("GEMM IBP should succeed");
    assert_eq!(
        cpu.lower().shape(),
        gemm.lower().shape(),
        "{label}: shape mismatch"
    );
    for (i, (c, g)) in cpu.lower().iter().zip(gemm.lower().iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-5,
            "{label}: lower[{i}] mismatch: cpu={c}, gemm={g}"
        );
    }
    for (i, (c, g)) in cpu.upper().iter().zip(gemm.upper().iter()).enumerate() {
        assert!(
            (c - g).abs() < 1e-5,
            "{label}: upper[{i}] mismatch: cpu={c}, gemm={g}"
        );
    }
}

/// #4238: 1x1 kernel, 3D input (no batch), with bias.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_gemm_parity_1x1_3d_with_bias() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![0.5_f32, -0.3]).unwrap();
    let layer =
        Conv2dLayer::with_input_shape(kernel, Some(array![0.1_f32, -0.2]), (1, 1), (0, 0), 3, 3)
            .unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 3, 3]), vec![-1.0_f32; 9]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 3, 3]), vec![1.0_f32; 9]).unwrap(),
    )
    .unwrap();
    assert_ibp_gemm_parity(&layer, &input, "1x1_3d_bias");
}

/// #4238: 1x1 kernel, 4D input (with batch), with bias.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_gemm_parity_1x1_4d_with_bias() {
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![0.5_f32, -0.3]).unwrap();
    let layer =
        Conv2dLayer::with_input_shape(kernel, Some(array![0.1_f32, -0.2]), (1, 1), (0, 0), 4, 4)
            .unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 4, 4]), vec![-0.5_f32; 32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 4, 4]), vec![0.5_f32; 32]).unwrap(),
    )
    .unwrap();
    assert_ibp_gemm_parity(&layer, &input, "1x1_4d_bias");
}

/// #4238: 2x2 kernel, mixed positive/negative weights, no bias.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_gemm_parity_2x2_no_bias() {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.5_f32, -0.1, 0.25, -0.75]).unwrap();
    let layer = Conv2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 4, 4).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 4, 4]),
            vec![
                -0.5, -0.3, 0.1, -0.2, -0.4, 0.0, -0.1, 0.2, -0.6, 0.3, -0.15, 0.05, -0.25, 0.1,
                -0.35, 0.15,
            ],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[1, 4, 4]),
            vec![
                0.5, 0.7, 0.9, 0.4, 0.6, 0.8, 0.3, 0.55, 0.65, 0.75, 0.45, 0.85, 0.35, 0.95, 0.5,
                0.6,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    assert_ibp_gemm_parity(&layer, &input, "2x2_no_bias");
}

/// #4238: 3x3 kernel with stride=2 and padding=1, 4D input, with bias.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_gemm_parity_3x3_stride2_pad1_4d() {
    let k: Vec<f32> = (0..18).map(|i| (i as f32 * 0.17).sin()).collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 3, 3]), k).unwrap();
    let layer =
        Conv2dLayer::with_input_shape(kernel, Some(array![0.05_f32, -0.1]), (2, 2), (1, 1), 6, 6)
            .unwrap();
    let lower_data: Vec<f32> = (0..36).map(|i| -0.5 + (i as f32 * 0.03)).collect();
    let upper_data: Vec<f32> = (0..36).map(|i| 0.5 + (i as f32 * 0.02)).collect();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 6, 6]), lower_data).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 6, 6]), upper_data).unwrap(),
    )
    .unwrap();
    assert_ibp_gemm_parity(&layer, &input, "3x3_stride2_pad1_4d");
}

/// #4238: Multi-channel input (3 channels), 2x2 kernel, with bias.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_gemm_parity_multichannel_2x2() {
    let k: Vec<f32> = (0..24).map(|i| (i as f32 * 0.23).cos() * 0.5).collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2, 2]), k).unwrap();
    let layer =
        Conv2dLayer::with_input_shape(kernel, Some(array![0.1_f32, -0.05]), (1, 1), (0, 0), 4, 4)
            .unwrap();
    let lower_data: Vec<f32> = (0..48).map(|i| -1.0 + (i as f32 * 0.04)).collect();
    let upper_data: Vec<f32> = (0..48).map(|i| 0.0 + (i as f32 * 0.04)).collect();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3, 4, 4]), lower_data).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3, 4, 4]), upper_data).unwrap(),
    )
    .unwrap();
    assert_ibp_gemm_parity(&layer, &input, "multichannel_2x2");
}

/// #4238: groups>1 falls back to CPU (no crash, same result).
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_ibp_gemm_grouped_falls_back_to_cpu() {
    use ny_core::NaiveCpuGemmEngine;
    // groups=2: each group has 1 in_channel, 1 out_channel
    let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 1, 1, 1]), vec![0.5_f32, -0.3]).unwrap();
    let layer = Conv2dLayer::with_input_shape_full(
        kernel,
        Some(array![0.1_f32, -0.2]),
        (1, 1),
        (0, 0),
        2,
        3,
        3,
    )
    .unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 3]), vec![-1.0_f32; 18]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2, 3, 3]), vec![1.0_f32; 18]).unwrap(),
    )
    .unwrap();
    let cpu = layer.propagate_ibp(&input).unwrap();
    let engine = NaiveCpuGemmEngine;
    let gemm = layer
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();
    assert_eq!(
        cpu.lower(),
        gemm.lower(),
        "grouped IBP should fall back to CPU exactly"
    );
    assert_eq!(
        cpu.upper(),
        gemm.upper(),
        "grouped IBP should fall back to CPU exactly"
    );
}

// ===== Dilated Conv2d + ConvTranspose2d output_padding (dilated-conv support) =====

/// Dilated Conv2d forward (IBP with degenerate bounds) must equal a directly
/// hand-computed dilated convolution. 1 channel, kernel 3x3, dilation 2,
/// stride 1, pad 0, input 5x5 → effective span 5 → output 1x1.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_dilation2_forward_matches_direct() -> Result<()> {
    let kh = 3;
    let kw = 3;
    let (in_h, in_w) = (5usize, 5usize);
    let dilation = (2usize, 2usize);

    // Kernel and input with distinct values for a meaningful check.
    let kernel_vals: Vec<f32> = (0..(kh * kw)).map(|i| (i as f32) + 1.0).collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, kh, kw]), kernel_vals.clone()).unwrap();
    let input_vals: Vec<f32> = (0..(in_h * in_w)).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let input = ArrayD::from_shape_vec(IxDyn(&[1, in_h, in_w]), input_vals.clone()).unwrap();

    let layer = Conv2dLayer::new_dilated(kernel, None, (1, 1), (0, 0), dilation, 1)?;

    // Degenerate bounds: lower == upper == input, so IBP output == concrete conv.
    let bt = BoundedTensor::new(input.clone(), input)?;
    let out = layer.propagate_ibp(&bt)?;
    assert_eq!(out.lower().shape(), &[1, 1, 1]);

    // Direct hand computation: single output cell at (0,0).
    let mut expected = 0.0f32;
    for ki in 0..kh {
        for kj in 0..kw {
            let ih = ki * dilation.0;
            let iw = kj * dilation.1;
            expected += input_vals[ih * in_w + iw] * kernel_vals[ki * kw + kj];
        }
    }
    assert_close(out.lower()[[0, 0, 0]], expected, 1e-4);
    assert_close(out.upper()[[0, 0, 0]], expected, 1e-4);
    Ok(())
}

/// Dilated Conv2d CROWN backward must round-trip: applying CROWN backward to an
/// identity objective and re-evaluating at a point reproduces the forward conv
/// output (Conv is affine, so CROWN is exact). 1 channel, kernel 3x3,
/// dilation 2, stride 1, pad 0, input 5x5.
#[ntest::timeout(10000)]
#[test]
fn test_conv2d_dilation2_crown_matches_forward() -> Result<()> {
    let kh = 3;
    let kw = 3;
    let (in_h, in_w) = (5usize, 5usize);
    let dilation = (2usize, 2usize);

    let kernel_vals: Vec<f32> = (0..(kh * kw)).map(|i| (i as f32) * 0.3 + 0.1).collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, kh, kw]), kernel_vals.clone()).unwrap();
    let input_vals: Vec<f32> = (0..(in_h * in_w))
        .map(|i| (i as f32) * 0.25 - 1.0)
        .collect();

    let mut layer =
        Conv2dLayer::new_dilated(kernel, Some(array![0.5]), (1, 1), (0, 0), dilation, 1)?;
    layer.set_input_shape(in_h, in_w);

    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    let out_dim = out_h * out_w; // out_c = 1
                                 // Identity objective over the conv output.
    let lin = LinearBounds::identity(out_dim);
    let back = layer.propagate_linear(&lin)?;

    // Evaluate the back-propagated affine bound at the concrete input.
    let in_dim = in_h * in_w;
    let lower_a = back.lower_a();
    let lower_b = back.lower_b();
    for o in 0..out_dim {
        let mut acc = lower_b[o] as f64;
        for j in 0..in_dim {
            acc += lower_a[[o, j]] as f64 * input_vals[j] as f64;
        }
        // Direct forward value for this output cell.
        let oh = o / out_w;
        let ow = o % out_w;
        let mut expected = 0.5f64; // bias
        for ki in 0..kh {
            for kj in 0..kw {
                let ih = oh + ki * dilation.0;
                let iw = ow + kj * dilation.1;
                expected += input_vals[ih * in_w + iw] as f64 * kernel_vals[ki * kw + kj] as f64;
            }
        }
        assert!(
            (acc - expected).abs() < 1e-3,
            "CROWN backward mismatch at out {o}: got {acc}, expected {expected}"
        );
    }
    Ok(())
}

/// ConvTranspose2d forward with output_padding must produce a larger output
/// whose valid region equals the output_padding=0 result, with the extra
/// high-end row/column receiving only the bias. 1 channel, kernel 3x3,
/// stride 2, pad 0, output_padding 1, input 2x2.
#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_output_padding_forward() -> Result<()> {
    let kh = 3;
    let kw = 3;
    let (in_h, in_w) = (2usize, 2usize);
    let stride = (2usize, 2usize);

    let kernel_vals: Vec<f32> = (0..(kh * kw)).map(|i| (i as f32) + 1.0).collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, kh, kw]), kernel_vals).unwrap();
    let input_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input = ArrayD::from_shape_vec(IxDyn(&[1, in_h, in_w]), input_vals).unwrap();
    let bias = 0.25f32;

    // output_padding = 0 → out = 2*(2-1)+3 = 5.
    let base = ConvTranspose2dLayer::new(kernel.clone(), Some(array![bias]), stride, (0, 0))?;
    let base_bt = BoundedTensor::new(input.clone(), input)?;
    let base_out = base.propagate_ibp(&base_bt)?;
    assert_eq!(base_out.lower().shape(), &[1, 5, 5]);

    // output_padding = 1 → out = 5 + 1 = 6.
    let padded =
        ConvTranspose2dLayer::new_full(kernel, Some(array![bias]), stride, (0, 0), (1, 1), (1, 1))?;
    let out = padded.propagate_ibp(&base_bt)?;
    assert_eq!(out.lower().shape(), &[1, 6, 6]);

    // The 5x5 valid region must match base exactly; the extra last row/col are
    // bias-only (no scatter reaches them).
    for oh in 0..6 {
        for ow in 0..6 {
            let v = out.lower()[[0, oh, ow]];
            if oh < 5 && ow < 5 {
                assert_close(v, base_out.lower()[[0, oh, ow]], 1e-4);
            } else {
                assert_close(v, bias, 1e-4);
            }
        }
    }
    Ok(())
}

/// ConvTranspose2d CROWN backward with output_padding must be exact: an
/// identity objective re-evaluated at a point reproduces the forward
/// transposed-conv output (affine op). stride 2, output_padding 1.
#[ntest::timeout(10000)]
#[test]
fn test_convtranspose2d_output_padding_crown_matches_forward() -> Result<()> {
    let kh = 3;
    let kw = 3;
    let (in_h, in_w) = (2usize, 2usize);
    let stride = (2usize, 2usize);

    let kernel_vals: Vec<f32> = (0..(kh * kw)).map(|i| (i as f32) * 0.2 + 0.1).collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 1, kh, kw]), kernel_vals).unwrap();
    let input_vals: Vec<f32> = vec![0.7, -1.3, 2.1, 0.4];
    let input = ArrayD::from_shape_vec(IxDyn(&[1, in_h, in_w]), input_vals.clone()).unwrap();
    let bias = -0.6f32;

    let mut layer =
        ConvTranspose2dLayer::new_full(kernel, Some(array![bias]), stride, (0, 0), (1, 1), (1, 1))?;
    layer.set_input_shape(in_h, in_w);

    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    assert_eq!((out_h, out_w), (6, 6));
    let out_dim = out_h * out_w;

    // Forward reference (degenerate bounds).
    let bt = BoundedTensor::new(input.clone(), input)?;
    let fwd = layer.propagate_ibp(&bt)?;

    let lin = LinearBounds::identity(out_dim);
    let back = layer.propagate_linear(&lin)?;
    let in_dim = in_h * in_w;
    let lower_a = back.lower_a();
    let lower_b = back.lower_b();
    for o in 0..out_dim {
        let mut acc = lower_b[o] as f64;
        for j in 0..in_dim {
            acc += lower_a[[o, j]] as f64 * input_vals[j] as f64;
        }
        let oh = o / out_w;
        let ow = o % out_w;
        let expected = fwd.lower()[[0, oh, ow]] as f64;
        assert!(
            (acc - expected).abs() < 1e-3,
            "ConvTranspose CROWN mismatch at out ({oh},{ow}): got {acc}, expected {expected}"
        );
    }
    Ok(())
}

// ===== Patches-mode CROWN backward equivalence vs. dense (#hotpath) =====
//
// SOUNDNESS: Patches-mode CROWN backward represents the same linear operator as
// the dense `conv2d_transpose_batched_gemm` path, just structured as convolution
// patches instead of a materialized [out_dim x in_dim] A-matrix. These tests
// assert the two paths produce IDENTICAL coefficient matrices (within f32
// tolerance) across stride/padding/groups, and that the bounds remain sound
// (contain the forward eval at sampled points).

use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
use crate::layers::common::PatchesPropagation;

/// Build a Conv2d kernel with deterministic, non-trivial signed values.
fn patches_test_kernel(out_c: usize, in_c_per_group: usize, kh: usize, kw: usize) -> ArrayD<f32> {
    let n = out_c * in_c_per_group * kh * kw;
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as f32) * 0.37).sin() * 0.8 - 0.1)
        .collect();
    ArrayD::from_shape_vec(IxDyn(&[out_c, in_c_per_group, kh, kw]), data).unwrap()
}

/// Compare patches-mode CROWN backward against the dense GEMM transpose for a
/// single Conv2d layer starting from identity incoming bounds.
///
/// `identity` starting bounds is the common case at the CNN trunk boundary; the
/// resulting A-matrix is exactly the conv backward linear operator. Both paths
/// must agree element-wise.
#[allow(clippy::too_many_arguments)]
fn assert_patches_matches_dense_backward(
    out_c: usize,
    in_c: usize,
    kh: usize,
    kw: usize,
    in_h: usize,
    in_w: usize,
    stride: (usize, usize),
    padding: (usize, usize),
    groups: usize,
) -> Result<()> {
    let kernel = patches_test_kernel(out_c, in_c / groups, kh, kw);
    // No bias: isolates the linear A-operator so both paths produce zero bias and
    // identical coefficient matrices (bias rounding differs between paths).
    let layer =
        Conv2dLayer::with_input_shape_full(kernel, None, stride, padding, groups, in_h, in_w)?;
    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    let out_dim = out_c * out_h * out_w;
    let in_dim = in_c * in_h * in_w;

    // --- Dense path: identity A -> conv2d_transpose_batched_gemm via propagate_linear ---
    let dense_bounds = LinearBounds::identity(out_dim);
    let dense = layer.propagate_linear(&dense_bounds)?.into_owned();
    assert_eq!(dense.lower_a.shape(), &[out_dim, in_dim]);

    // --- Patches path: identity patches -> propagate_patches -> materialize dense ---
    let patches_in = PatchesLinearBounds::identity((out_c, out_h, out_w), (out_c, out_h, out_w));
    let crown = layer.propagate_patches(&patches_in)?;
    let patches_dense = match crown {
        CrownBounds::Patches(pb) => pb.to_dense()?,
        CrownBounds::Dense(lb) => lb,
    };
    assert_eq!(
        patches_dense.lower_a.shape(),
        &[out_dim, in_dim],
        "patches->dense A shape mismatch (s={stride:?} p={padding:?} g={groups})"
    );

    // The two A-matrices must be element-wise identical (same linear operator).
    for d in 0..out_dim {
        for j in 0..in_dim {
            let lo_dense = dense.lower_a[[d, j]];
            let lo_patch = patches_dense.lower_a[[d, j]];
            let hi_dense = dense.upper_a[[d, j]];
            let hi_patch = patches_dense.upper_a[[d, j]];
            assert!(
                (lo_dense - lo_patch).abs() < 1e-4,
                "lower_a mismatch at [{d},{j}] (s={stride:?} p={padding:?} g={groups}): \
                 dense={lo_dense} patches={lo_patch}"
            );
            assert!(
                (hi_dense - hi_patch).abs() < 1e-4,
                "upper_a mismatch at [{d},{j}] (s={stride:?} p={padding:?} g={groups}): \
                 dense={hi_dense} patches={hi_patch}"
            );
        }
    }

    // --- Soundness: concretized bounds must contain forward eval at sampled points ---
    let lower_vals: Vec<f32> = (0..in_dim)
        .map(|i| ((i as f32) * 0.21).cos() - 0.3)
        .collect();
    let upper_vals: Vec<f32> = lower_vals.iter().map(|&v| v + 0.5).collect();
    let input_lower =
        ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), lower_vals.clone()).unwrap();
    let input_upper =
        ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), upper_vals.clone()).unwrap();

    // Sample a few interior points and verify patches bounds contain forward eval.
    for t in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let sample: Vec<f32> = (0..in_dim)
            .map(|j| lower_vals[j] + t * (upper_vals[j] - lower_vals[j]))
            .collect();
        let sample_arr =
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), sample.clone()).unwrap();
        let bt = BoundedTensor::new(sample_arr.clone(), sample_arr)?;
        let fwd = layer.propagate_ibp(&bt)?; // degenerate bounds == forward eval
        for d in 0..out_dim {
            let mut lo = patches_dense.lower_b[d] as f64;
            let mut hi = patches_dense.upper_b[d] as f64;
            for j in 0..in_dim {
                let al = patches_dense.lower_a[[d, j]] as f64;
                let au = patches_dense.upper_a[[d, j]] as f64;
                // Concretize the bilinear bound at the sampled point.
                lo += al * sample[j] as f64;
                hi += au * sample[j] as f64;
            }
            let f = *fwd.lower().iter().nth(d).unwrap() as f64;
            assert!(
                lo <= f + 1e-3 && f <= hi + 1e-3,
                "patches bounds unsound at out {d}, t={t}: lo={lo} f={f} hi={hi}"
            );
        }
    }

    // Keep input bounds referenced (documents the concretization domain).
    let _ = (input_lower, input_upper);
    Ok(())
}

#[ntest::timeout(20000)]
#[test]
fn test_patches_crown_matches_dense_stride1_pad1() -> Result<()> {
    assert_patches_matches_dense_backward(2, 2, 3, 3, 5, 5, (1, 1), (1, 1), 1)
}

#[ntest::timeout(20000)]
#[test]
fn test_patches_crown_matches_dense_stride1_pad0() -> Result<()> {
    assert_patches_matches_dense_backward(3, 2, 3, 3, 6, 6, (1, 1), (0, 0), 1)
}

#[ntest::timeout(20000)]
#[test]
fn test_patches_crown_matches_dense_stride2_pad1() -> Result<()> {
    assert_patches_matches_dense_backward(2, 1, 3, 3, 7, 7, (2, 2), (1, 1), 1)
}

#[ntest::timeout(20000)]
#[test]
fn test_patches_crown_matches_dense_stride2_pad0_2x2() -> Result<()> {
    assert_patches_matches_dense_backward(2, 2, 2, 2, 6, 6, (2, 2), (0, 0), 1)
}

#[ntest::timeout(20000)]
#[test]
fn test_patches_crown_matches_dense_grouped_depthwise() -> Result<()> {
    // Depthwise: groups == in_c == out_c, kernel (out_c, 1, kh, kw).
    assert_patches_matches_dense_backward(4, 4, 3, 3, 6, 6, (1, 1), (1, 1), 4)
}

#[ntest::timeout(20000)]
#[test]
fn test_patches_crown_matches_dense_grouped_2() -> Result<()> {
    // 2 groups: in_c=4, out_c=4, in_c_per_group=2.
    assert_patches_matches_dense_backward(4, 4, 3, 3, 5, 5, (1, 1), (1, 1), 2)
}

#[ntest::timeout(20000)]
#[test]
fn test_patches_crown_matches_dense_asymmetric_stride_pad() -> Result<()> {
    assert_patches_matches_dense_backward(2, 2, 3, 3, 8, 6, (2, 1), (1, 0), 1)
}

/// Dilated convolutions are NOT yet supported in patches mode; `propagate_patches`
/// must reject them with `UnsupportedConfiguration` so the caller falls back to the
/// dilation-aware dense CROWN path (never silently produce wrong bounds).
#[ntest::timeout(20000)]
#[test]
fn test_patches_crown_rejects_dilation_falls_back_to_dense() -> Result<()> {
    let (out_c, in_c, kh, kw, in_h, in_w) = (2, 1, 3, 3, 7, 7);
    let kernel = patches_test_kernel(out_c, in_c, kh, kw);
    let mut layer = Conv2dLayer::new_dilated(kernel, None, (1, 1), (1, 1), (2, 2), 1)?;
    layer.set_input_shape(in_h, in_w);
    let (out_h, out_w) = layer.output_size(in_h, in_w)?;
    let patches_in = PatchesLinearBounds::identity((out_c, out_h, out_w), (out_c, out_h, out_w));
    let err = layer.propagate_patches(&patches_in).unwrap_err();
    assert!(
        matches!(err, NyError::UnsupportedConfiguration(_)),
        "expected UnsupportedConfiguration for dilated patches CROWN, got {err:?}"
    );
    // And the dense path must still succeed for the dilated conv (the fallback).
    let out_dim = out_c * out_h * out_w;
    let dense_in = LinearBounds::identity(out_dim);
    let dense = layer.propagate_linear(&dense_in)?;
    assert_eq!(dense.lower_a().shape()[0], out_dim);
    Ok(())
}

// ===== Deep conv-stack patches-vs-dense equivalence (#hotpath) =====
//
// SOUNDNESS + threshold change: raising the patches->dense crossover from the
// fixed 75%-per-dimension rule to the true MEMORY-area crossover keeps patches
// mode active through more stacked conv layers. These tests chain 3-4 convs and
// drive CROWN backward through the whole stack in BOTH patches and dense modes,
// asserting the final A-matrices are element-wise identical (same linear
// operator). The receptive field grows past the OLD 75% threshold partway
// through the stack, so they directly exercise the deeper-patches path.

/// One conv layer in a stack: its kernel + hyperparameters, with input spatial
/// dims resolved during the forward shape walk.
struct StackConv {
    layer: Conv2dLayer,
    out_c: usize,
    out_h: usize,
    out_w: usize,
    in_h: usize,
    in_w: usize,
}

/// Build a stack of stride-1/stride-2 convs over the given input, returning each
/// layer with resolved shapes (forward order) plus the final output dims.
fn build_conv_stack(
    in_c0: usize,
    in_h0: usize,
    in_w0: usize,
    specs: &[(usize, usize, (usize, usize), (usize, usize))], // (out_c, kernel, stride, padding)
) -> Result<Vec<StackConv>> {
    let mut convs = Vec::new();
    let (mut cur_c, mut cur_h, mut cur_w) = (in_c0, in_h0, in_w0);
    for &(out_c, k, stride, padding) in specs {
        let kernel = patches_test_kernel(out_c, cur_c, k, k);
        let layer =
            Conv2dLayer::with_input_shape_full(kernel, None, stride, padding, 1, cur_h, cur_w)?;
        let (oh, ow) = layer.output_size(cur_h, cur_w)?;
        convs.push(StackConv {
            layer,
            out_c,
            out_h: oh,
            out_w: ow,
            in_h: cur_h,
            in_w: cur_w,
        });
        cur_c = out_c;
        cur_h = oh;
        cur_w = ow;
    }
    Ok(convs)
}

/// Drive CROWN backward through a conv stack in dense and patches modes and
/// assert the final A-matrices match element-wise. Returns the number of conv
/// layers (counted from the output side) that stay in patches mode (i.e. neither
/// the area crossover nor the padding-composition soundness guard bailed).
fn assert_stack_patches_matches_dense(
    in_c: usize,
    in_h: usize,
    in_w: usize,
    specs: &[(usize, usize, (usize, usize), (usize, usize))],
) -> Result<usize> {
    let convs = build_conv_stack(in_c, in_h, in_w, specs)?;
    let last = convs.last().unwrap();
    let out_dim = last.out_c * last.out_h * last.out_w;
    let in_dim = in_c * in_h * in_w;

    // --- Dense path: identity -> propagate_linear through each conv, output->input ---
    let mut dense = LinearBounds::identity(out_dim);
    for sc in convs.iter().rev() {
        dense = sc.layer.propagate_linear(&dense)?.into_owned();
    }
    assert_eq!(dense.lower_a.shape(), &[out_dim, in_dim]);

    // --- Patches path: identity patches -> propagate_patches through each conv ---
    let mut crown = CrownBounds::Patches(Box::new(PatchesLinearBounds::identity(
        (last.out_c, last.out_h, last.out_w),
        (last.out_c, last.out_h, last.out_w),
    )));
    // Count layers (from output side) that stay in patches.
    let mut patches_depth = 0usize;
    let mut still_patches = true;
    for sc in convs.iter().rev() {
        // Mirror the dispatcher's pre-check (would_conv_compose_cover_input).
        if let CrownBounds::Patches(pb) = &crown {
            let (kh, kw) = sc.layer.kernel_size();
            let bail = pb.lower_a.would_conv_compose_cover_input(
                sc.layer.stride,
                (kh, kw),
                sc.in_h,
                sc.in_w,
            ) || pb.upper_a.would_conv_compose_cover_input(
                sc.layer.stride,
                (kh, kw),
                sc.in_h,
                sc.in_w,
            );
            if bail {
                still_patches = false;
                let lb = crown.ensure_dense()?;
                crown = CrownBounds::Dense(sc.layer.propagate_linear(lb)?.into_owned());
                continue;
            }
            // Mirror the dispatcher: on the soundness guard (UnsupportedConfiguration)
            // fall back to the exact dense path for this conv.
            match sc.layer.propagate_patches(pb) {
                Ok(result) => {
                    if still_patches {
                        patches_depth += 1;
                    }
                    crown = result;
                }
                Err(NyError::UnsupportedConfiguration(_)) => {
                    still_patches = false;
                    let lb = crown.ensure_dense()?;
                    crown = CrownBounds::Dense(sc.layer.propagate_linear(lb)?.into_owned());
                }
                Err(e) => return Err(e),
            }
        } else if let CrownBounds::Dense(lb) = &crown {
            crown = CrownBounds::Dense(sc.layer.propagate_linear(lb)?.into_owned());
        }
    }

    let patches_dense = match crown {
        CrownBounds::Patches(pb) => pb.to_dense()?,
        CrownBounds::Dense(lb) => lb,
    };
    assert_eq!(patches_dense.lower_a.shape(), &[out_dim, in_dim]);

    for d in 0..out_dim {
        for j in 0..in_dim {
            let lo_dense = dense.lower_a[[d, j]];
            let lo_patch = patches_dense.lower_a[[d, j]];
            let hi_dense = dense.upper_a[[d, j]];
            let hi_patch = patches_dense.upper_a[[d, j]];
            assert!(
                (lo_dense - lo_patch).abs() < 1e-4,
                "stack lower_a mismatch at [{d},{j}]: dense={lo_dense} patches={lo_patch}"
            );
            assert!(
                (hi_dense - hi_patch).abs() < 1e-4,
                "stack upper_a mismatch at [{d},{j}]: dense={hi_dense} patches={hi_patch}"
            );
        }
    }
    Ok(patches_depth)
}

/// 3 stacked stride-1 3x3 PAD-0 ("valid") convs over a 12x12 input. The composed
/// receptive field grows 3 -> 5 -> 7; over 12x12 the area stays 49 < 144, so the
/// new memory-area crossover keeps ALL THREE convs in patches mode — well past
/// the OLD fixed-75%-per-dimension rule (which bailed once a dim reached
/// ceil(12*3/4)=9). Patches must match dense element-wise across the whole stack.
///
/// Pad-0 chains are the case where patches composition is provably exact at every
/// position, so the deeper-patches path is both sound AND active here.
#[ntest::timeout(60000)]
#[test]
fn test_stack_stride1_3x3_x3_patches_matches_dense() -> Result<()> {
    let depth = assert_stack_patches_matches_dense(
        2,
        12,
        12,
        &[
            (3, 3, (1, 1), (0, 0)),
            (3, 3, (1, 1), (0, 0)),
            (2, 3, (1, 1), (0, 0)),
        ],
    )?;
    // All 3 pad-0 stride-1 convs stay in patches under the area crossover.
    assert_eq!(
        depth, 3,
        "all 3 stride-1 pad-0 convs should stay in patches mode"
    );
    Ok(())
}

/// 4 stacked stride-1 3x3 PAD-0 convs over 14x14: receptive field 3->5->7->9
/// (area 81 < 196). All four stay in patches under the area crossover — a depth
/// the OLD 75% rule would not reach. Deep chained-composition equivalence test.
#[ntest::timeout(120000)]
#[test]
fn test_stack_stride1_3x3_x4_patches_matches_dense() -> Result<()> {
    let depth = assert_stack_patches_matches_dense(
        1,
        14,
        14,
        &[
            (2, 3, (1, 1), (0, 0)),
            (2, 3, (1, 1), (0, 0)),
            (2, 3, (1, 1), (0, 0)),
            (1, 3, (1, 1), (0, 0)),
        ],
    )?;
    assert_eq!(
        depth, 4,
        "all 4 stride-1 pad-0 convs should stay in patches mode"
    );
    Ok(())
}

/// Mixed stride-2 / stride-1 PAD-0 stack over 17x17. Stride-2 convs compound the
/// composed stride and kernel; verifies the area crossover and composition math
/// stay sound (and exact) when strides compound. Equivalence is asserted
/// regardless of which layers stay in patches vs bail to dense on the area
/// crossover.
#[ntest::timeout(120000)]
#[test]
fn test_stack_stride2_mixed_patches_matches_dense() -> Result<()> {
    let depth = assert_stack_patches_matches_dense(
        2,
        17,
        17,
        &[
            (3, 3, (2, 2), (0, 0)), // 17->8
            (3, 3, (1, 1), (0, 0)), // 8->6
            (2, 3, (2, 2), (0, 0)), // 6->2
        ],
    )?;
    // At least the output-side conv stays in patches; deeper layers may bail once
    // the composed receptive field reaches the dense area.
    assert!(depth >= 1, "output-side conv should start in patches mode");
    Ok(())
}

/// Direct demonstration that the NEW threshold keeps patches active where the OLD
/// 75%-per-dimension rule would have bailed. Two stride-1 3x3 PAD-0 convs over an
/// 8x8 input: the composed kernel reaches 5x5 (area 25 < 64, patches ~2.5x cheaper
/// than dense) — but the OLD threshold of ceil(8*3/4)=6 would have bailed at the
/// next dim step. Both convs stay in patches AND match dense element-wise.
#[ntest::timeout(60000)]
#[test]
fn test_stack_keeps_patches_past_old_threshold() -> Result<()> {
    let depth = assert_stack_patches_matches_dense(
        2,
        8,
        8,
        &[(2, 3, (1, 1), (0, 0)), (2, 3, (1, 1), (0, 0))],
    )?;
    assert_eq!(
        depth, 2,
        "both stride-1 3x3 pad-0 convs over 8x8 (max area 25<64) stay in patches"
    );
    Ok(())
}

/// SOUNDNESS guard for padded chains (#hotpath): patches composition through a
/// conv whose INCOMING patches carry nonzero padding is NOT element-wise
/// equivalent to dense (boundary truncation of the intermediate conv is lost), so
/// `propagate_patches` rejects it and the dispatcher falls back to the exact dense
/// path. This test stacks three pad-1 stride-1 convs: the output-side conv runs in
/// patches (identity incoming, zero padding), but the next conv sees nonzero
/// incoming padding and falls back to dense. The end-to-end result must STILL
/// match the all-dense operator element-wise (the fallback preserves exactness).
#[ntest::timeout(60000)]
#[test]
fn test_stack_padded_chain_falls_back_but_stays_exact() -> Result<()> {
    let depth = assert_stack_patches_matches_dense(
        2,
        10,
        10,
        &[
            (3, 3, (1, 1), (1, 1)),
            (3, 3, (1, 1), (1, 1)),
            (2, 3, (1, 1), (1, 1)),
        ],
    )?;
    // Only the output-side conv (identity incoming, zero padding) stays in patches;
    // the padding-composition guard converts the rest to dense.
    assert_eq!(
        depth, 1,
        "padded chain keeps only the first (output-side) conv in patches; \
         the guard falls the rest back to dense"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// im2col+GEMM forward-IBP optimization equivalence & soundness (#hot-conv-ibp)
// ---------------------------------------------------------------------------

/// OLD element-wise reference: the exact W+/W- splitting via the naive
/// nested-loop `conv2d_single_grouped`, as the CPU IBP path did before the
/// im2col+GEMM rewrite. Used as ground truth for the equivalence test.
fn conv2d_ibp_forward_reference(
    input_lower: &ArrayD<f32>,
    input_upper: &ArrayD<f32>,
    kernel: &ArrayD<f32>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> (ArrayD<f32>, ArrayD<f32>) {
    let kernel_pos = kernel.mapv(crate::bounds::nan_propagating_max_zero);
    let kernel_neg = kernel.mapv(crate::bounds::nan_propagating_min_zero);
    let lp =
        conv2d_single_grouped(input_lower, &kernel_pos, stride, padding, dilation, groups).unwrap();
    let ln =
        conv2d_single_grouped(input_upper, &kernel_neg, stride, padding, dilation, groups).unwrap();
    let up =
        conv2d_single_grouped(input_upper, &kernel_pos, stride, padding, dilation, groups).unwrap();
    let un =
        conv2d_single_grouped(input_lower, &kernel_neg, stride, padding, dilation, groups).unwrap();
    (lp + ln, up + un)
}

/// Tiny deterministic LCG -> f32 in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32; // 24 bits
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }
}

#[allow(clippy::too_many_arguments)]
fn rand_arr(rng: &mut Lcg, shape: &[usize], scale: f32) -> ArrayD<f32> {
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n).map(|_| rng.next_f32() * scale).collect();
    ArrayD::from_shape_vec(IxDyn(shape), v).unwrap()
}

#[ntest::timeout(60000)]
#[test]
fn test_conv2d_ibp_gemm_matches_elementwise_reference() {
    use super::ops_ibp_fwd::conv2d_ibp_forward_grouped;
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    // (in_c, out_c, kh, kw, stride, pad, dil, groups, in_h, in_w)
    let cases = [
        (
            3usize,
            4usize,
            3usize,
            3usize,
            (1, 1),
            (0, 0),
            (1, 1),
            1usize,
            7usize,
            7usize,
        ),
        (3, 6, 3, 3, (1, 1), (1, 1), (1, 1), 3, 8, 8), // depthwise-ish groups=3
        (4, 8, 3, 3, (2, 2), (1, 1), (1, 1), 2, 9, 9), // stride 2, groups 2
        (6, 6, 1, 1, (1, 1), (0, 0), (1, 1), 6, 5, 5), // 1x1 depthwise
        (2, 4, 3, 3, (1, 1), (2, 2), (2, 2), 1, 7, 7), // dilation 2
        (8, 8, 2, 2, (2, 2), (0, 0), (1, 1), 4, 10, 10), // even kernel
    ];
    for (i, &(in_c, out_c, kh, kw, stride, pad, dil, groups, in_h, in_w)) in
        cases.iter().enumerate()
    {
        let kernel = rand_arr(&mut rng, &[out_c, in_c / groups, kh, kw], 1.0);
        // Build a valid interval: center +/- nonneg radius.
        let center = rand_arr(&mut rng, &[in_c, in_h, in_w], 2.0);
        let radius = rand_arr(&mut rng, &[in_c, in_h, in_w], 0.5).mapv(f32::abs);
        let lower = &center - &radius;
        let upper = &center + &radius;

        let (ref_l, ref_u) =
            conv2d_ibp_forward_reference(&lower, &upper, &kernel, stride, pad, dil, groups);
        let fwd =
            conv2d_ibp_forward_grouped(&lower, &upper, &kernel, stride, pad, dil, groups, None)
                .unwrap();

        assert_eq!(fwd.lower.shape(), ref_l.shape(), "case {i} lower shape");
        for (a, b) in fwd.lower.iter().zip(ref_l.iter()) {
            let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
            assert!((a - b).abs() <= tol, "case {i} lower mismatch: {a} vs {b}");
        }
        for (a, b) in fwd.upper.iter().zip(ref_u.iter()) {
            let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
            assert!((a - b).abs() <= tol, "case {i} upper mismatch: {a} vs {b}");
        }

        // SOUNDNESS: bounds must contain the concrete forward eval at sampled
        // input points within [lower, upper]. Sample center and the corners.
        for sample in [&center, &lower, &upper] {
            let y = conv2d_single_grouped(sample, &kernel, stride, pad, dil, groups).unwrap();
            for (idx, &yv) in y.indexed_iter() {
                let lo = fwd.lower[&idx];
                let hi = fwd.upper[&idx];
                let slack = 1e-3 * yv.abs().max(1.0);
                assert!(
                    yv >= lo - slack && yv <= hi + slack,
                    "case {i} soundness: {yv} not in [{lo}, {hi}]"
                );
            }
        }
    }
}

/// Engine-routed Conv2d IBP forward (`engine = Some`) must match the CPU faer
/// path (`engine = None`) within ~1e-4 across grouped and non-grouped convs,
/// and the engine must actually be invoked (4 matmuls per group). This covers
/// the #hot-conv-ibp engine routing: the GemmEngine produces the same matmul
/// result, the W+/W- interval decomposition is unchanged, so bounds are equal.
///
/// GPU/Metal parity (ny-gpu engine) is covered by ny-gpu's own GEMM parity
/// tests; here we use the in-crate `CountingGemmEngine` (delegates to the naive
/// CPU GEMM) so we can both assert numerical equivalence and confirm the engine
/// is dispatched rather than bypassed.
#[ntest::timeout(60000)]
#[test]
fn test_conv2d_ibp_forward_engine_matches_cpu_faer() {
    use super::ops_ibp_fwd::conv2d_ibp_forward_grouped;
    let mut rng = Lcg(0x0bad_f00d_1234_5678);
    // (in_c, out_c, kh, kw, stride, pad, dil, groups, in_h, in_w)
    let cases = [
        (
            3usize,
            4usize,
            3usize,
            3usize,
            (1, 1),
            (0, 0),
            (1, 1),
            1usize,
            7usize,
            7usize,
        ),
        (3, 6, 3, 3, (1, 1), (1, 1), (1, 1), 3, 8, 8), // depthwise-ish groups=3
        (4, 8, 3, 3, (2, 2), (1, 1), (1, 1), 2, 9, 9), // stride 2, groups 2
        (6, 6, 1, 1, (1, 1), (0, 0), (1, 1), 6, 5, 5), // 1x1 depthwise
        (2, 4, 3, 3, (1, 1), (2, 2), (2, 2), 1, 7, 7), // dilation 2
        (8, 8, 2, 2, (2, 2), (0, 0), (1, 1), 4, 10, 10), // even kernel, groups 4
    ];
    for (i, &(in_c, out_c, kh, kw, stride, pad, dil, groups, in_h, in_w)) in
        cases.iter().enumerate()
    {
        let kernel = rand_arr(&mut rng, &[out_c, in_c / groups, kh, kw], 1.0);
        let center = rand_arr(&mut rng, &[in_c, in_h, in_w], 2.0);
        let radius = rand_arr(&mut rng, &[in_c, in_h, in_w], 0.5).mapv(f32::abs);
        let lower = &center - &radius;
        let upper = &center + &radius;

        // CPU faer path (engine = None).
        let cpu =
            conv2d_ibp_forward_grouped(&lower, &upper, &kernel, stride, pad, dil, groups, None)
                .unwrap();

        // Engine-routed path (engine = Some). CountingGemmEngine delegates to the
        // naive CPU GEMM and records call count.
        let engine = CountingGemmEngine::new();
        let eng = conv2d_ibp_forward_grouped(
            &lower,
            &upper,
            &kernel,
            stride,
            pad,
            dil,
            groups,
            Some(&engine),
        )
        .unwrap();

        // The engine must be dispatched: 4 matmuls (l_pos, l_neg, u_pos, u_neg)
        // per group. If the engine were bypassed this would be 0.
        assert_eq!(
            engine.gemm_calls(),
            4 * groups,
            "case {i}: expected 4 GEMM calls per group ({groups} groups)"
        );

        assert_eq!(eng.lower.shape(), cpu.lower.shape(), "case {i} lower shape");
        assert_eq!(eng.upper.shape(), cpu.upper.shape(), "case {i} upper shape");
        for (a, b) in eng.lower.iter().zip(cpu.lower.iter()) {
            let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "case {i} lower engine/cpu mismatch: {a} vs {b}"
            );
        }
        for (a, b) in eng.upper.iter().zip(cpu.upper.iter()) {
            let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "case {i} upper engine/cpu mismatch: {a} vs {b}"
            );
        }

        // SOUNDNESS: engine bounds must still contain the concrete forward eval
        // at sampled points within [lower, upper].
        for sample in [&center, &lower, &upper] {
            let y = conv2d_single_grouped(sample, &kernel, stride, pad, dil, groups).unwrap();
            for (idx, &yv) in y.indexed_iter() {
                let lo = eng.lower[&idx];
                let hi = eng.upper[&idx];
                let slack = 1e-3 * yv.abs().max(1.0);
                assert!(
                    yv >= lo - slack && yv <= hi + slack,
                    "case {i} engine soundness: {yv} not in [{lo}, {hi}]"
                );
            }
        }
    }
}

/// `Conv2dLayer::propagate_ibp_with_engine` on a grouped (groups > 1) conv must
/// route through the engine (previously it dropped to CPU and ignored the
/// engine) and produce bounds equal to the CPU `propagate_ibp`. Confirms the
/// caller-chain threading from bound.rs reaches the grouped GEMM path.
#[ntest::timeout(60000)]
#[test]
fn test_conv2d_layer_ibp_with_engine_grouped_routes_and_matches_cpu() -> Result<()> {
    let mut rng = Lcg(0xfeed_face_0102_0304);
    let (in_c, out_c, kh, kw, groups, in_h, in_w) = (4, 8, 3, 3, 2, 9, 9);
    let kernel = rand_arr(&mut rng, &[out_c, in_c / groups, kh, kw], 1.0);
    let bias: Vec<f32> = (0..out_c).map(|_| rng.next_f32()).collect();
    let bias_arr = ndarray::Array1::from(bias);
    let layer = Conv2dLayer::new_dilated(kernel, Some(bias_arr), (1, 1), (1, 1), (1, 1), groups)?;

    let center = rand_arr(&mut rng, &[in_c, in_h, in_w], 2.0);
    let radius = rand_arr(&mut rng, &[in_c, in_h, in_w], 0.5).mapv(f32::abs);
    let lower = &center - &radius;
    let upper = &center + &radius;
    let bt = BoundedTensor::new(lower, upper)?;

    let cpu = layer.propagate_ibp(&bt)?;

    let engine = CountingGemmEngine::new();
    let eng = layer.propagate_ibp_with_engine(&bt, Some(&engine))?;

    // groups > 1 must reach the engine via the grouped im2col+GEMM forward.
    assert_eq!(
        engine.gemm_calls(),
        4 * groups,
        "grouped propagate_ibp_with_engine must dispatch 4 GEMMs per group"
    );

    for (a, b) in eng.lower().iter().zip(cpu.lower().iter()) {
        let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "layer lower engine/cpu mismatch: {a} vs {b}"
        );
    }
    for (a, b) in eng.upper().iter().zip(cpu.upper().iter()) {
        let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= tol,
            "layer upper engine/cpu mismatch: {a} vs {b}"
        );
    }
    Ok(())
}

#[ntest::timeout(60000)]
#[test]
fn test_conv2d_layer_ibp_matches_elementwise_reference_with_bias() -> Result<()> {
    let mut rng = Lcg(0xdead_beef_cafe_1234);
    let (in_c, out_c, kh, kw, groups, in_h, in_w) = (4, 8, 3, 3, 2, 9, 9);
    let kernel = rand_arr(&mut rng, &[out_c, in_c / groups, kh, kw], 1.0);
    let bias: Vec<f32> = (0..out_c).map(|_| rng.next_f32()).collect();
    let bias_arr = ndarray::Array1::from(bias.clone());
    let layer = Conv2dLayer::new_dilated(
        kernel.clone(),
        Some(bias_arr),
        (1, 1),
        (1, 1),
        (1, 1),
        groups,
    )?;

    let center = rand_arr(&mut rng, &[in_c, in_h, in_w], 2.0);
    let radius = rand_arr(&mut rng, &[in_c, in_h, in_w], 0.5).mapv(f32::abs);
    let lower = &center - &radius;
    let upper = &center + &radius;
    let bt = BoundedTensor::new(lower.clone(), upper.clone())?;
    let out = layer.propagate_ibp(&bt)?;

    let (mut ref_l, mut ref_u) =
        conv2d_ibp_forward_reference(&lower, &upper, &kernel, (1, 1), (1, 1), (1, 1), groups);
    let (oc, oh, ow) = (ref_l.shape()[0], ref_l.shape()[1], ref_l.shape()[2]);
    for c in 0..oc {
        for h in 0..oh {
            for w in 0..ow {
                ref_l[[c, h, w]] += bias[c];
                ref_u[[c, h, w]] += bias[c];
            }
        }
    }
    for (a, b) in out.lower().iter().zip(ref_l.iter()) {
        let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
        assert!((a - b).abs() <= tol, "layer lower mismatch: {a} vs {b}");
    }
    for (a, b) in out.upper().iter().zip(ref_u.iter()) {
        let tol = 1e-4 * a.abs().max(b.abs()).max(1.0);
        assert!((a - b).abs() <= tol, "layer upper mismatch: {a} vs {b}");
    }
    Ok(())
}

// Manual timing (run with: cargo test -p ny-propagate --lib --release
//   convolution::tests::bench_conv2d_ibp_forward -- --ignored --nocapture).
// Representative conv: 64 in/out channels, 27x27, stride 1, 3x3, groups 1.
// Measured on dev hardware (release, 20 iters):
//   old(elementwise) = 312.7 ms/iter   new(im2col+gemm) = 3.57 ms/iter
//   => ~87x speedup, since the inner products become a cache-friendly faer
//   GEMM instead of per-output-element dynamic ArrayD indexing.
#[test]
#[ignore = "manual timing benchmark; run explicitly with --ignored --nocapture"]
fn bench_conv2d_ibp_forward() {
    use super::ops_ibp_fwd::conv2d_ibp_forward_grouped;
    use std::time::Instant;
    let mut rng = Lcg(7);
    let (in_c, out_c, kh, kw, h, w) = (64usize, 64usize, 3usize, 3usize, 27usize, 27usize);
    let kernel = rand_arr(&mut rng, &[out_c, in_c, kh, kw], 0.1);
    let center = rand_arr(&mut rng, &[in_c, h, w], 1.0);
    let radius = rand_arr(&mut rng, &[in_c, h, w], 0.1).mapv(f32::abs);
    let lower = &center - &radius;
    let upper = &center + &radius;
    let iters = 20;

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = conv2d_ibp_forward_reference(&lower, &upper, &kernel, (1, 1), (0, 0), (1, 1), 1);
    }
    let old = t0.elapsed();

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ =
            conv2d_ibp_forward_grouped(&lower, &upper, &kernel, (1, 1), (0, 0), (1, 1), 1, None)
                .unwrap();
    }
    let new = t1.elapsed();

    println!(
        "conv2d IBP forward 64ch 27x27 3x3: old(elementwise)={:?}/iter  new(im2col+gemm)={:?}/iter  speedup={:.2}x",
        old / iters,
        new / iters,
        old.as_secs_f64() / new.as_secs_f64()
    );
}

/// Fail-before / pass-after repro for the conv IBP-forward under-widening
/// (#vnncomp-aw-soundness). The f32 window-sum accumulation can deviate from the
/// true value by far more than the generic 1-ULP `round_for_soundness` widening
/// under cancellation; `propagate_ibp_sound_with_engine` adds the certified Higham
/// error term so the box soundly encloses the true conv output.
#[test]
fn conv2d_sound_ibp_forward_encloses_under_f32_cancellation() {
    use crate::layers::common::BoundPropagation;

    // 1x1 conv, in_c=4, out_c=1; weights chosen for catastrophic f32 cancellation:
    // 2^24 + 1 - 2^24 + 4 = 5 exactly, but f32 accumulation drops the +1 -> 4.
    let p = (1u32 << 24) as f32;
    let kernel = ArrayD::from_shape_vec(IxDyn(&[1, 4, 1, 1]), vec![p, 1.0, -p, 4.0]).unwrap();
    let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();

    // Point input x = [1,1,1,1] (degenerate box). True output = 5 (exact in f64).
    let x = ArrayD::from_shape_vec(IxDyn(&[4, 1, 1]), vec![1.0_f32; 4]).unwrap();
    let input = BoundedTensor::concrete(x).unwrap();
    let true_val = 5.0_f64;

    // SOUND path must enclose the true value.
    let sound = conv.propagate_ibp_sound_with_engine(&input, None).unwrap();
    let (slo, shi) = (
        sound.lower()[[0, 0, 0]] as f64,
        sound.upper()[[0, 0, 0]] as f64,
    );
    assert!(
        slo <= true_val && shi >= true_val,
        "sound conv IBP [{slo},{shi}] must enclose true value {true_val}"
    );

    // Fail-before demonstration: the pre-fix path (f32 forward + 1-ULP widening) MISSES it.
    let mut old = conv.propagate_ibp(&input).unwrap();
    old.round_for_soundness_inplace();
    let (olo, ohi) = (old.lower()[[0, 0, 0]] as f64, old.upper()[[0, 0, 0]] as f64);
    assert!(
        olo > true_val || ohi < true_val,
        "expected the pre-fix 1-ULP conv IBP path to MISS true {true_val}, got [{olo},{ohi}] \
         (if this fires, the cancellation demo no longer triggers — revisit the kernel)"
    );

    // Realistic (non-cancellation) intervals: sound box must enclose the exact f64 range.
    let cases: [(Vec<f32>, (f32, f32)); 2] = [
        (vec![0.8, -0.3, 0.4, 0.9], (-1.0, 1.0)),
        (vec![1.5, -2.0, 0.7, -0.4], (-0.5, 0.5)),
    ];
    for (w, (xl, xu)) in cases {
        let n = w.len();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[1, n, 1, 1]), w.clone()).unwrap();
        let conv = Conv2dLayer::new(kernel, None, (1, 1), (0, 0)).unwrap();
        let lo = ArrayD::from_shape_vec(IxDyn(&[n, 1, 1]), vec![xl; n]).unwrap();
        let hi = ArrayD::from_shape_vec(IxDyn(&[n, 1, 1]), vec![xu; n]).unwrap();
        let input = BoundedTensor::new(lo, hi).unwrap();
        let (mut tmin, mut tmax) = (0.0_f64, 0.0_f64);
        for &wi in &w {
            let wi = wi as f64;
            if wi >= 0.0 {
                tmin += wi * xl as f64;
                tmax += wi * xu as f64;
            } else {
                tmin += wi * xu as f64;
                tmax += wi * xl as f64;
            }
        }
        let sb = conv.propagate_ibp_sound_with_engine(&input, None).unwrap();
        let (lo2, hi2) = (sb.lower()[[0, 0, 0]] as f64, sb.upper()[[0, 0, 0]] as f64);
        assert!(
            lo2 <= tmin + 1e-5 && hi2 >= tmax - 1e-5,
            "sound conv [{lo2},{hi2}] must enclose true range [{tmin},{tmax}] for w={w:?}"
        );
    }
}

// ---- #wall-deadwork oracles (NY_CONV_SKIP_DEAD_F32) ----
// These tests use the crate-wide environment lock shared with other CROWN tests
// that use the helper, including the existing NY_CROWN_MEM_CAP_MB oracles.

/// 2→3 channels, 2x2 kernel over 3x3 input: out 3x2x2 = 12, in 2x3x3 = 18.
/// Dense enough that the transpose gather, grouped GEMM, and error channel all
/// do real work (unlike the 1x1 identity fixtures above).
fn deadwork_conv() -> Conv2dLayer {
    let n = 3 * 2 * 2 * 2;
    let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7311).sin() * 0.5).collect();
    let kernel = ArrayD::from_shape_vec(IxDyn(&[3, 2, 2, 2]), w).expect("kernel");
    Conv2dLayer::with_input_shape(kernel, Some(array![0.1, -0.2, 0.3]), (1, 1), (0, 0), 3, 3)
        .expect("valid conv2d")
}

fn deadwork_bounds(with_err: bool) -> LinearBounds {
    let n = 12;
    let mut b = LinearBounds::identity(n);
    for (i, v) in b.lower_a.iter_mut().enumerate() {
        *v = ((i as f32) * 0.317).sin();
    }
    for (i, v) in b.upper_a.iter_mut().enumerate() {
        *v = ((i as f32) * 0.317).sin().abs() + ((i as f32) * 0.111).cos() * 0.25;
    }
    for (i, v) in b.lower_b.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 - 0.2;
    }
    for (i, v) in b.upper_b.iter_mut().enumerate() {
        *v = (i as f32) * 0.05 + 0.2;
    }
    if with_err {
        b.lower_a_err = Some(Array2::from_shape_fn((n, n), |(i, j)| {
            (((i * 13 + j) % 7) as f32) * 1e-6
        }));
        b.upper_a_err = Some(Array2::from_shape_fn((n, n), |(i, j)| {
            (((i * 7 + j) % 5) as f32) * 1e-6
        }));
    }
    b
}

fn assert_linear_bounds_bitwise_eq(a: &LinearBounds, b: &LinearBounds, ctx: &str) {
    assert_eq!(a.lower_a.shape(), b.lower_a.shape(), "{ctx}: lower_a shape");
    for (x, y) in a.lower_a.iter().zip(b.lower_a.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: lower_a coeff mismatch");
    }
    for (x, y) in a.upper_a.iter().zip(b.upper_a.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: upper_a coeff mismatch");
    }
    for (x, y) in a.lower_b.iter().zip(b.lower_b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: lower_b mismatch");
    }
    for (x, y) in a.upper_b.iter().zip(b.upper_b.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: upper_b mismatch");
    }
    match (&a.lower_a_err, &b.lower_a_err) {
        (Some(ea), Some(eb)) => {
            for (x, y) in ea.iter().zip(eb.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: lower_a_err mismatch");
            }
        }
        (None, None) => {}
        _ => panic!("{ctx}: lower_a_err presence mismatch"),
    }
    match (&a.upper_a_err, &b.upper_a_err) {
        (Some(ea), Some(eb)) => {
            for (x, y) in ea.iter().zip(eb.iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "{ctx}: upper_a_err mismatch");
            }
        }
        (None, None) => {}
        _ => panic!("{ctx}: upper_a_err presence mismatch"),
    }
}

/// The skip must be BITWISE identical to the shipped path on recompute success —
/// the f32 pair's values are dead (overwritten by the rounded f64 recompute), so
/// removing them may change nothing at all. With and without incoming error
/// matrices, with and without a (future) deadline.
#[ntest::timeout(30000)]
#[test]
fn wall_deadwork_skip_is_bitwise_identical() {
    crate::tests::with_env_edits(|env| {
        env.remove("NY_CONV_SKIP_DEAD_F32");
        let layer = deadwork_conv();
        for with_err in [false, true] {
            for dl in [
                None,
                Some(std::time::Instant::now() + std::time::Duration::from_mins(10)),
            ] {
                let bounds = deadwork_bounds(with_err);
                env.remove("NY_CONV_SKIP_DEAD_F32");
                let off = layer
                    .propagate_linear_with_engine_and_deadline(&bounds, None, dl)
                    .expect("off path")
                    .into_owned();
                env.set("NY_CONV_SKIP_DEAD_F32", "1");
                let on = layer
                    .propagate_linear_with_engine_and_deadline(&bounds, None, dl)
                    .expect("on path")
                    .into_owned();
                env.remove("NY_CONV_SKIP_DEAD_F32");
                assert_linear_bounds_bitwise_eq(
                    &off,
                    &on,
                    &format!("with_err={with_err} deadline={:?}", dl.is_some()),
                );
            }
        }
    });
}

/// With the gate on, an already-expired per-node deadline must abort with the
/// same DeadlineExceeded the pair path uses (the collector's sound
/// reference-bounds fallback), never propagate garbage.
#[ntest::timeout(30000)]
#[test]
fn wall_deadwork_skip_expired_deadline_aborts() {
    crate::tests::with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32", "1")], || {
        let layer = deadwork_conv();
        let bounds = deadwork_bounds(false);
        let expired = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(10))
            .expect("now() is at least 10ms past the Instant epoch");
        let err = layer
            .propagate_linear_with_engine_and_deadline(&bounds, None, Some(expired))
            .expect_err("expired deadline must abort under the skip");
        assert!(
            matches!(err, NyError::DeadlineExceeded(_)),
            "expected DeadlineExceeded, got {err:?}"
        );
    });
}

/// With the gate on, the memory-cap refusal must fire exactly as on the pair
/// path (CpuMemoryExceeded → the collector's sound IBP fallback), not attempt
/// the allocation. The cap parses integer MB (min enforceable 1MB), so the
/// fixture needs enough objective rows that the output buffer exceeds 1MB:
/// 15000 rows x 18 input cols x 4B ≈ 1.03MB.
#[ntest::timeout(30000)]
#[test]
fn wall_deadwork_skip_respects_mem_cap() {
    crate::tests::with_serialized_env_vars(
        &[("NY_CONV_SKIP_DEAD_F32", "1"), ("NY_CROWN_MEM_CAP_MB", "1")],
        || {
            let layer = deadwork_conv();
            let rows = 15000;
            let bounds = LinearBounds::new(
                Array2::from_shape_fn((rows, 12), |(i, j)| (((i * 12 + j) as f32) * 0.013).sin()),
                ndarray::Array1::zeros(rows),
                Array2::from_shape_fn((rows, 12), |(i, j)| {
                    (((i * 12 + j) as f32) * 0.013).sin().abs()
                }),
                ndarray::Array1::zeros(rows),
            )
            .expect("big bounds");
            let result = layer.propagate_linear_with_engine_and_deadline(&bounds, None, None);
            match result {
                Err(NyError::CpuMemoryExceeded { .. }) => {}
                other => panic!(
                    "expected CpuMemoryExceeded under 1MB cap, got {:?}",
                    other.map(|_| "Ok(bounds)")
                ),
            }
        },
    );
}

/// The kill-switch (`NY_CONV_SKIP_DEAD_F32=0`) restores the pair path. The
/// discriminating observable: an already-expired deadline on a SMALL workload
/// — the skip aborts (strict check), while the unchunked pair path finishes.
#[ntest::timeout(30000)]
#[test]
fn wall_deadwork_kill_switch_restores_pair_path() {
    crate::tests::with_env_edits(|env| {
        env.set("NY_CONV_SKIP_DEAD_F32", "0");
        let layer = deadwork_conv();
        let bounds = deadwork_bounds(false);
        let expired = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(10))
            .expect("now() is at least 10ms past the Instant epoch");
        let off = layer.propagate_linear_with_engine_and_deadline(&bounds, None, Some(expired));
        assert!(
            off.is_ok(),
            "kill-switch must restore the pair path (small workload finishes despite expired deadline)"
        );
        // Default (unset): the skip is ON and aborts on the expired deadline.
        env.remove("NY_CONV_SKIP_DEAD_F32");
        let on = layer.propagate_linear_with_engine_and_deadline(&bounds, None, Some(expired));
        assert!(
            matches!(on, Err(NyError::DeadlineExceeded(_))),
            "default-on skip must abort on expired deadline, got Ok"
        );
    });
}

/// #wall-deadwork ConvTranspose port: 2-ch → 3-ch, 2x2 kernel over a 3x3
/// input (output 3x4x4 = 48, input 2x3x3 = 18) — dense enough that the
/// forward-conv gather, the f64 recompute, and the exact err composition all
/// do real work. The skip must be BITWISE identical to the pair path, with
/// and without incoming error matrices.
#[ntest::timeout(30000)]
#[test]
fn wall_deadwork_convtranspose_skip_is_bitwise_identical() {
    crate::tests::with_env_edits(|env| {
        let n = 2 * 3 * 2 * 2;
        let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7311).sin() * 0.5).collect();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2, 2]), w).expect("kernel");
        let layer = ConvTranspose2dLayer::with_input_shape(
            kernel,
            Some(array![0.1, -0.2, 0.3]),
            (1, 1),
            (0, 0),
            3,
            3,
        )
        .expect("valid convtranspose2d");
        let out_dim = 3 * 4 * 4;
        for with_err in [false, true] {
            let mut b = LinearBounds::identity(out_dim);
            for (i, v) in b.lower_a.iter_mut().enumerate() {
                *v = ((i as f32) * 0.317).sin();
            }
            for (i, v) in b.upper_a.iter_mut().enumerate() {
                *v = ((i as f32) * 0.317).sin().abs() + ((i as f32) * 0.111).cos() * 0.25;
            }
            for (i, v) in b.lower_b.iter_mut().enumerate() {
                *v = (i as f32) * 0.05 - 0.2;
            }
            for (i, v) in b.upper_b.iter_mut().enumerate() {
                *v = (i as f32) * 0.05 + 0.2;
            }
            if with_err {
                b.lower_a_err = Some(Array2::from_shape_fn((out_dim, out_dim), |(i, j)| {
                    (((i * 13 + j) % 7) as f32) * 1e-6
                }));
                b.upper_a_err = Some(Array2::from_shape_fn((out_dim, out_dim), |(i, j)| {
                    (((i * 7 + j) % 5) as f32) * 1e-6
                }));
            }
            for dl in [
                None,
                Some(std::time::Instant::now() + std::time::Duration::from_mins(10)),
            ] {
                env.set("NY_CONV_SKIP_DEAD_F32", "0");
                let off = layer
                    .propagate_linear_with_engine_and_deadline(&b, None, dl)
                    .expect("off path")
                    .into_owned();
                env.set("NY_CONV_SKIP_DEAD_F32", "1");
                let on = layer
                    .propagate_linear_with_engine_and_deadline(&b, None, dl)
                    .expect("on path")
                    .into_owned();
                env.remove("NY_CONV_SKIP_DEAD_F32");
                assert_linear_bounds_bitwise_eq(
                    &off,
                    &on,
                    &format!(
                        "convtranspose with_err={with_err} deadline={:?}",
                        dl.is_some()
                    ),
                );
            }
        }
    });
}

/// With the ConvTranspose skip on, an already-expired per-node deadline must
/// abort with DeadlineExceeded (the collector's sound reference-bounds
/// fallback), never propagate garbage.
#[ntest::timeout(30000)]
#[test]
fn wall_deadwork_convtranspose_skip_expired_deadline_aborts() {
    crate::tests::with_serialized_env_vars(&[("NY_CONV_SKIP_DEAD_F32", "1")], || {
        let n = 2 * 3 * 2 * 2;
        let w: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.7311).sin() * 0.5).collect();
        let kernel = ArrayD::from_shape_vec(IxDyn(&[2, 3, 2, 2]), w).expect("kernel");
        let layer = ConvTranspose2dLayer::with_input_shape(kernel, None, (1, 1), (0, 0), 3, 3)
            .expect("valid convtranspose2d");
        let bounds = LinearBounds::identity(3 * 4 * 4);
        let expired = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_millis(10))
            .expect("now() is at least 10ms past the Instant epoch");
        let err = layer
            .propagate_linear_with_engine_and_deadline(&bounds, None, Some(expired))
            .expect_err("expired deadline must abort under the skip");
        assert!(
            matches!(err, NyError::DeadlineExceeded(_)),
            "expected DeadlineExceeded, got {err:?}"
        );
    });
}
