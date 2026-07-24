// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

use super::BatchNormLayer;
use crate::layers::common::BoundPropagation;
use crate::{BatchedLinearBounds, LinearBounds};

fn make_bn_layer() -> BatchNormLayer {
    // ny=2, beta=1, mean=0.5, var=0.25, eps=0
    // scale = 2 / sqrt(0.25) = 2/0.5 = 4
    // bias = 1 - 0.5*4 = -1
    let ny = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 2.0]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let mean = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 0.5]).unwrap();
    let var = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.25, 0.25]).unwrap();
    BatchNormLayer::new(&ny, &beta, &mean, &var, 0.0).unwrap()
}

#[test]
fn test_new_computes_scale_and_bias() {
    let layer = make_bn_layer();
    assert_eq!(layer.num_channels, 2);
    // scale = ny / sqrt(var + eps) = 2 / sqrt(0.25) = 4
    assert!(
        (layer.scale[[0]] - 4.0).abs() < 1e-5,
        "scale[0] expected 4.0, got {}",
        layer.scale[[0]]
    );
    // bias = beta - mean * scale = 1 - 0.5*4 = -1
    assert!(
        (layer.bias[[0]] - (-1.0)).abs() < 1e-5,
        "bias[0] expected -1.0, got {}",
        layer.bias[[0]]
    );
}

#[test]
fn test_from_scale_bias() {
    let scale = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.1, 0.2, 0.3]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();
    assert_eq!(layer.num_channels, 3);
    assert!(
        (layer.scale[[1]] - 2.0).abs() < 1e-5,
        "scale[1] expected 2.0, got {}",
        layer.scale[[1]]
    );
    assert!(
        (layer.bias[[2]] - 0.3).abs() < 1e-5,
        "bias[2] expected 0.3, got {}",
        layer.bias[[2]]
    );
}

#[test]
fn test_ibp_1d_positive_scale() {
    // scale=[4,4], bias=[-1,-1], input: lower=[0,1], upper=[2,3]
    // y = 4*x - 1, positive scale => no swap
    // lower: 4*[0,1] - 1 = [-1, 3], upper: 4*[2,3] - 1 = [7, 11]
    let layer = make_bn_layer();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 3.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert!(
        (out.lower()[[0]] - (-1.0)).abs() < 1e-5,
        "lower[0] expected -1.0, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.lower()[[1]] - 3.0).abs() < 1e-5,
        "lower[1] expected 3.0, got {}",
        out.lower()[[1]]
    );
    assert!(
        (out.upper()[[0]] - 7.0).abs() < 1e-5,
        "upper[0] expected 7.0, got {}",
        out.upper()[[0]]
    );
    assert!(
        (out.upper()[[1]] - 11.0).abs() < 1e-5,
        "upper[1] expected 11.0, got {}",
        out.upper()[[1]]
    );
}

#[test]
fn test_ibp_1d_negative_scale_swaps() {
    // scale=-2, bias=3 => y = -2*x + 3
    // For lower=1, upper=4: y_l = -2*4+3 = -5, y_u = -2*1+3 = 1
    let scale = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-2.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[1]), vec![3.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![4.0]).unwrap(),
    )
    .unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert!(
        (out.lower()[[0]] - (-5.0)).abs() < 1e-5,
        "lower[0] expected -5.0 (neg scale swap), got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[0]] - 1.0).abs() < 1e-5,
        "upper[0] expected 1.0 (neg scale swap), got {}",
        out.upper()[[0]]
    );
}

#[test]
fn test_ibp_2d_nchw() {
    // 2 channels, scale=[2, -1], bias=[0, 0]
    // Shape [2, 2]: channel heuristic for 2D prefers index 1 when shape[1]==num_channels
    // So layout is (N=2, C=2): batch dim=0, channel dim=1
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, -1.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    // lower: [[1,2],[3,4]], upper: [[5,6],[7,8]]
    // [i,j]: batch i, channel j
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();

    let out = layer.propagate_ibp(&input).unwrap();
    assert_eq!(out.shape(), &[2, 2]);

    // Channel 0 (positive scale=2): [0,0] l=2, u=10; [1,0] l=6, u=14
    assert!((out.lower()[[0, 0]] - 2.0).abs() < 1e-5);
    assert!((out.upper()[[0, 0]] - 10.0).abs() < 1e-5);
    assert!((out.lower()[[1, 0]] - 6.0).abs() < 1e-5);
    assert!((out.upper()[[1, 0]] - 14.0).abs() < 1e-5);

    // Channel 1 (negative scale=-1): swap
    // [0,1] l=2,u=6 -> lower=-6, upper=-2; [1,1] l=4,u=8 -> lower=-8, upper=-4
    assert!((out.lower()[[0, 1]] - (-6.0)).abs() < 1e-5);
    assert!((out.upper()[[0, 1]] - (-2.0)).abs() < 1e-5);
    assert!((out.lower()[[1, 1]] - (-8.0)).abs() < 1e-5);
    assert!((out.upper()[[1, 1]] - (-4.0)).abs() < 1e-5);
}

#[test]
fn test_ibp_rank4_batch_count_equal_to_channels_scales_per_channel() {
    // Rank-4 input is batched [N, C, H, W], so the channel axis is 1 even when
    // the batch count N collides with the channel count C. A value-resolved
    // axis would pick axis 0 here and scale every element of batch row n by
    // channel n's affine instead of its own channel's.
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 10.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let ones = ArrayD::from_elem(IxDyn(&[2, 2, 1, 1]), 1.0);
    let input = BoundedTensor::new(ones.clone(), ones).unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    // Per-channel scaling: [n, c] -> scale[c], for BOTH batch rows.
    for n in 0..2 {
        assert!(
            (out.lower()[[n, 0, 0, 0]] - 1.0).abs() < 1e-4,
            "batch {} channel 0 expected 1.0, got {}",
            n,
            out.lower()[[n, 0, 0, 0]]
        );
        assert!(
            (out.lower()[[n, 1, 0, 0]] - 10.0).abs() < 1e-4,
            "batch {} channel 1 expected 10.0, got {}",
            n,
            out.lower()[[n, 1, 0, 0]]
        );
    }
}

#[test]
fn test_ibp_rank4_channel_axis_mismatch_errors() {
    // Rank-4 with channels only on axis 0 must be rejected, not guessed:
    // batched [N, C, H, W] layout puts channels on axis 1.
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 10.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let ones = ArrayD::from_elem(IxDyn(&[2, 5, 1, 1]), 1.0);
    let input = BoundedTensor::new(ones.clone(), ones).unwrap();
    assert!(
        layer.propagate_ibp(&input).is_err(),
        "rank-4 input with channels on axis 0 only should be rejected"
    );
}

#[test]
fn test_ibp_rank3_chw_spatial_dim_equal_to_channels_stays_channel_first() {
    // Rank 3 is the squeezed [C, H, W] convention: channel axis 0, even when a
    // spatial extent collides with the channel count.
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 10.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let ones = ArrayD::from_elem(IxDyn(&[2, 2, 3]), 1.0);
    let input = BoundedTensor::new(ones.clone(), ones).unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    for h in 0..2 {
        for w in 0..3 {
            assert!(
                (out.lower()[[0, h, w]] - 1.0).abs() < 1e-4,
                "channel 0 expected 1.0, got {}",
                out.lower()[[0, h, w]]
            );
            assert!(
                (out.lower()[[1, h, w]] - 10.0).abs() < 1e-4,
                "channel 1 expected 10.0, got {}",
                out.lower()[[1, h, w]]
            );
        }
    }
}

#[test]
fn test_ibp_soundness_corners() {
    // Verify all corner evaluations fall within bounds
    let layer = make_bn_layer();
    // Scale=4, bias=-1 for both channels
    let lower = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, 0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, 3.0]).unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone()).unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    // Test all 4 corners
    for mask in 0..4u32 {
        for c in 0..2 {
            let x = if mask & (1 << c) != 0 {
                upper[[c]]
            } else {
                lower[[c]]
            };
            let y = x * 4.0 - 1.0;
            assert!(
                out.lower()[[c]] <= y + 1e-5,
                "Channel {} lower {} > eval {} at mask {}",
                c,
                out.lower()[[c]],
                y,
                mask
            );
            assert!(
                out.upper()[[c]] >= y - 1e-5,
                "Channel {} upper {} < eval {} at mask {}",
                c,
                out.upper()[[c]],
                y,
                mask
            );
        }
    }
}

#[test]
fn test_ibp_point_input() {
    let layer = make_bn_layer();
    let vals = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
    let input = BoundedTensor::new(vals.clone(), vals).unwrap();
    let out = layer.propagate_ibp(&input).unwrap();

    // y = 4*x - 1: [4*1-1, 4*2-1] = [3, 7]
    assert!(
        (out.lower()[[0]] - 3.0).abs() < 1e-5,
        "lower[0] expected 3.0, got {}",
        out.lower()[[0]]
    );
    assert!(
        (out.upper()[[1]] - 7.0).abs() < 1e-5,
        "upper[1] expected 7.0, got {}",
        out.upper()[[1]]
    );
    // Point input => lower == upper
    assert!(
        (out.lower()[[0]] - out.upper()[[0]]).abs() < 1e-5,
        "point input: lower[0] should equal upper[0]"
    );
}

#[test]
fn test_ibp_channel_mismatch_errors() {
    let layer = make_bn_layer(); // 2 channels
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap(),
    )
    .unwrap();
    let result = layer.propagate_ibp(&input);
    assert!(
        result.is_err(),
        "IBP should reject channel mismatch (3 vs 2)"
    );
}

#[test]
fn test_propagate_linear_returns_unsupported() {
    let layer = make_bn_layer();
    let bounds = LinearBounds::new(
        Array2::eye(2),
        Array1::zeros(2),
        Array2::eye(2),
        Array1::zeros(2),
    )
    .unwrap();
    let result = layer.propagate_linear(&bounds);
    assert!(
        result.is_err(),
        "propagate_linear should be unsupported for batch norm"
    );
}

#[test]
fn test_requires_pre_activation_bounds() {
    let layer = make_bn_layer();
    assert!(
        layer.requires_pre_activation_bounds(),
        "batch norm requires pre-activation bounds"
    );
}

#[test]
fn test_crown_backward_1d() {
    // scale=[4, 4], bias=[-1, -1]
    // Identity CROWN bounds: lower_a=I, upper_a=I, b=0
    // After BN backward: y = 4*x - 1
    // new_A = A * diag(scale) = I * 4 = 4*I
    // new_b = b + A @ bias = 0 + I @ [-1,-1] = [-1,-1]
    let layer = make_bn_layer();
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    let bounds = LinearBounds::new(
        Array2::eye(2),
        Array1::zeros(2),
        Array2::eye(2),
        Array1::zeros(2),
    )
    .unwrap();

    let result = layer
        .propagate_crown_backward(&bounds, Some(&pre_act))
        .unwrap();

    // lower_a = I * 4 (diagonal)
    assert!((result.lower_a[[0, 0]] - 4.0).abs() < 1e-5);
    assert!((result.lower_a[[0, 1]] - 0.0).abs() < 1e-5);
    assert!((result.lower_a[[1, 1]] - 4.0).abs() < 1e-5);

    // lower_b = 0 + I @ [-1, -1] = [-1, -1]
    assert!(
        (result.lower_b[0] - (-1.0)).abs() < 1e-5,
        "lower_b[0] expected -1.0, got {}",
        result.lower_b[0]
    );
    assert!(
        (result.lower_b[1] - (-1.0)).abs() < 1e-5,
        "lower_b[1] expected -1.0, got {}",
        result.lower_b[1]
    );
}

#[test]
fn test_crown_backward_no_pre_activation_errors() {
    let layer = make_bn_layer();
    let bounds = LinearBounds::new(
        Array2::eye(2),
        Array1::zeros(2),
        Array2::eye(2),
        Array1::zeros(2),
    )
    .unwrap();
    let result = layer.propagate_crown_backward(&bounds, None);
    assert!(
        result.is_err(),
        "CROWN backward should error without pre-activation bounds"
    );
}

#[test]
fn test_crown_backward_negative_scale_no_swap() {
    // CROWN backward should NOT swap for negative scale
    // scale=[-1], bias=[0] => y = -x
    // With identity CROWN: new_A = I * (-1), new_b = 0
    let scale = ArrayD::from_shape_vec(IxDyn(&[1]), vec![-1.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap(),
    )
    .unwrap();

    let bounds = LinearBounds::new(
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
        Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(),
        Array1::zeros(1),
    )
    .unwrap();

    let result = layer
        .propagate_crown_backward(&bounds, Some(&pre_act))
        .unwrap();

    // Both lower and upper should be scaled by -1 (no swap in CROWN)
    assert!((result.lower_a[[0, 0]] - (-1.0)).abs() < 1e-5);
    assert!((result.upper_a[[0, 0]] - (-1.0)).abs() < 1e-5);
}

#[test]
fn test_crown_batched_1d_matches_ibp() {
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0, -1.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.5, 1.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![5.0, 6.0]).unwrap(),
    )
    .unwrap();

    let incoming = BatchedLinearBounds::identity(&[2]).unwrap();
    let result = layer
        .propagate_linear_batched_with_bounds(&incoming, &input)
        .unwrap();
    let concrete = result.concretize(&input).unwrap();
    let ibp = layer.propagate_ibp(&input).unwrap();

    // The batched concretization now accumulates the dot product in f64 (BLAS
    // DGEMV on f32-cast operands) and rounds only the single final f64->f32 cast,
    // so CROWN matches IBP up to ordinary FP slack. On top of that, BatchNorm's
    // backward now certifies the f32 column-scaling rounding error
    // (#vnncomp-aw-soundness): `|A·scale - exact| ≤ |A·scale|·2^-24`, which concretize
    // applies OUTWARD against the input box. For this case (|A·scale| ≈ 10.5, input
    // box magnitude ≈ 6) that adds ~|A·scale|·2^-24·|x| ≈ 4e-6 of sound widening, so
    // the bound is strictly looser than IBP by the certified margin. Loosen the
    // tolerance from 1e-6 to 1e-5 to admit this sound penalty while still catching
    // real (much larger) regressions. The direction is checked separately below
    // (CROWN must remain a valid OUTER bound of IBP).
    let tol = 1e-5;
    assert_eq!(concrete.shape(), ibp.shape());
    // CROWN must remain a sound OUTER bound of IBP (lower no higher, upper no lower),
    // close up to `tol` (ordinary FP slack + the certified coeff-error penalty).
    for (&actual, &expected) in concrete.lower().iter().zip(ibp.lower().iter()) {
        assert!(
            actual <= expected + tol && actual > expected - tol,
            "lower mismatch: actual={actual}, expected={expected}, tol={tol}",
        );
    }
    for (&actual, &expected) in concrete.upper().iter().zip(ibp.upper().iter()) {
        assert!(
            actual >= expected - tol && actual < expected + tol,
            "upper mismatch: actual={actual}, expected={expected}, tol={tol}",
        );
    }
}

#[test]
fn test_new_with_epsilon() {
    // With epsilon, sqrt(var+eps) changes
    let ny = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let mean = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let var = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let layer = BatchNormLayer::new(&ny, &beta, &mean, &var, 1.0).unwrap();
    // scale = 1 / sqrt(0 + 1) = 1.0
    assert!(
        (layer.scale[[0]] - 1.0).abs() < 1e-5,
        "scale[0] expected 1.0, got {}",
        layer.scale[[0]]
    );
    // bias = 0 - 0*1 = 0
    assert!(
        (layer.bias[[0]] - 0.0).abs() < 1e-5,
        "bias[0] expected 0.0, got {}",
        layer.bias[[0]]
    );
}

/// CROWN backward with zero-valued spatial dimension should error, not panic
/// with division-by-zero. (#2806)
#[test]
fn test_crown_backward_zero_spatial_dimension_returns_error() {
    // Shape [1, 2, 0]: batch=1, channels=2, spatial=0
    // elements_per_channel = 0, causing `i % chw / 0` panic without guard
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let layer = BatchNormLayer::from_scale_bias(scale, bias).unwrap();

    let l = ArrayD::from_shape_vec(IxDyn(&[1, 2, 0]), vec![]).expect("valid shape");
    let u = ArrayD::from_shape_vec(IxDyn(&[1, 2, 0]), vec![]).expect("valid shape");
    let pre_act = BoundedTensor::new(l, u).expect("valid bounds");

    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .unwrap();

    let err = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .unwrap_err();
    assert!(
        format!("{err}").contains("zero-valued"),
        "Expected zero-dimension error, got: {err}"
    );
}

/// Regression test: negative variance produces NaN scale, poisoning the
/// entire CROWN backward chain. `new()` must reject this. (#2814)
#[test]
fn test_new_rejects_negative_variance() {
    let ny = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let mean = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    // Channel 1 has negative variance: var=-1.0 + eps=1e-5 < 0
    let var = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.25, -1.0]).unwrap();
    let result = BatchNormLayer::new(&ny, &beta, &mean, &var, 1e-5);
    assert!(result.is_err(), "Expected error for negative variance");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("negative"),
        "Expected negative-variance error, got: {err_msg}"
    );
}

/// Valid variance with sufficient epsilon should still succeed.
#[test]
fn test_new_accepts_zero_variance_with_epsilon() {
    let ny = ArrayD::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let mean = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    let var = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0.0]).unwrap();
    // var=0, eps=1e-5: var+eps = 1e-5 >= 0, should succeed
    let result = BatchNormLayer::new(&ny, &beta, &mean, &var, 1e-5);
    assert!(
        result.is_ok(),
        "Zero variance with positive epsilon should succeed"
    );
}

// --- Rank-0 regression tests (#2868) ---

/// Rank-0 BoundedTensor must return Err, not panic, from BatchNorm IBP.
#[test]
fn test_ibp_rank0_returns_error_not_panic() {
    let layer = make_bn_layer();
    let lower = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[]), vec![2.0]).unwrap();
    let input = BoundedTensor::new(lower, upper).unwrap();
    let result = layer.propagate_ibp(&input);
    assert!(result.is_err(), "rank-0 input should return Err, not panic");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("rank-0"),
        "Error should mention rank-0: {err_msg}"
    );
}

/// Rank-0 BoundedTensor must return Err, not panic, from BatchNorm CROWN backward.
#[test]
fn test_crown_rank0_returns_error_not_panic() {
    let layer = make_bn_layer();
    // Pre-activation is rank-0
    let lower = ArrayD::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[]), vec![2.0]).unwrap();
    let pre_act = BoundedTensor::new(lower, upper).unwrap();
    // LinearBounds: 1x1 identity (minimal valid bounds)
    let bounds = LinearBounds::new(
        Array2::eye(1),
        Array1::zeros(1),
        Array2::eye(1),
        Array1::zeros(1),
    )
    .unwrap();
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act);
    assert!(
        result.is_err(),
        "rank-0 pre-activation should return Err, not panic"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("rank-0"),
        "Error should mention rank-0: {err_msg}"
    );
}

// --- from_scale_bias validation tests (#3339) ---

#[test]
fn test_from_scale_bias_rejects_shape_mismatch() {
    let scale = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).unwrap();
    let result = BatchNormLayer::from_scale_bias(scale, bias);
    assert!(result.is_err(), "Mismatched shapes should return Err");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("shape") || err_msg.contains("Shape"),
        "Error should mention shape mismatch: {err_msg}"
    );
}

#[test]
fn test_from_scale_bias_rejects_nan_scale() {
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, f32::NAN]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let result = BatchNormLayer::from_scale_bias(scale, bias);
    assert!(result.is_err(), "NaN in scale should return Err");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("non-finite") && err_msg.contains("scale"),
        "Error should mention non-finite scale: {err_msg}"
    );
}

#[test]
fn test_from_scale_bias_rejects_nan_bias() {
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::NAN, 0.0]).unwrap();
    let result = BatchNormLayer::from_scale_bias(scale, bias);
    assert!(result.is_err(), "NaN in bias should return Err");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("non-finite") && err_msg.contains("bias"),
        "Error should mention non-finite bias: {err_msg}"
    );
}

#[test]
fn test_from_scale_bias_rejects_inf_scale() {
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![f32::INFINITY, 1.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let result = BatchNormLayer::from_scale_bias(scale, bias);
    assert!(result.is_err(), "Inf in scale should return Err");
}

#[test]
fn test_from_scale_bias_rejects_neg_inf_bias() {
    let scale = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
    let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, f32::NEG_INFINITY]).unwrap();
    let result = BatchNormLayer::from_scale_bias(scale, bias);
    assert!(result.is_err(), "Neg Inf in bias should return Err");
}

// --- 7D explicit-rows coeff_err closure: 6D byte-identity pin ---
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §8.4 T2; validation gate §13 item 1)

/// BYTE-IDENTITY PIN: the 6D dense (non-identity, `unstable_idx: None`)
/// BatchNorm patches backward with NONZERO incoming `coeff_err` on both sides
/// and NONZERO layer `scale_err`/`bias_err` must stay bit-for-bit unchanged by
/// the 7D explicit-rows coeff_err closure (which adds a NEW 7D arm beside the
/// 6D arm without touching it — spec §8.3 step 2 keeps the 6D block textually
/// byte-identical).
///
/// Committed and verified green against the UNMODIFIED (pre-closure) tree; the
/// bit literals below were captured from a run of that tree
/// (`RUSTFLAGS="-C target-cpu=native" cargo test -p ny-propagate --release`).
/// Must pass unmodified after the closure lands.
///
/// Fixture notes (spec §8.4): 6D layout [oc=2, oh=1, ow=2, ic=2, kh=2, kw=1],
/// non-dyadic values (products must round), pad_top=1 so half the taps are
/// padding (exercises the valid-tap predicate in both the bias fold and the
/// HOLE1 widen loop), asymmetric per-side errs including exact-0.0 rows, and a
/// BatchNorm layer built as a struct literal with nonzero scale_err/bias_err
/// (`from_scale_bias` zeroes them — deliberately not used).
#[test]
fn test_bn_patches_6d_coeff_err_byte_identical_regression() {
    use crate::bounds::patches::{CrownBounds, PatchesData, PatchesLinearBounds};
    use crate::layers::common::PatchesPropagation;

    let layer = BatchNormLayer {
        scale: ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.1_f32, -0.7]).unwrap(),
        bias: ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.3_f32, -0.9]).unwrap(),
        scale_err: ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.5e-4_f32, 3.0e-5]).unwrap(),
        bias_err: ArrayD::from_shape_vec(IxDyn(&[2]), vec![2.0e-4_f32, 7.0e-6]).unwrap(),
        num_channels: 2,
    };

    // 6D patches [oc=2, oh=1, ow=2, ic=2, kh=2, kw=1]; out-neuron count 4.
    let lower_vals = vec![
        0.7_f32, -1.3, 0.55, 2.4, -0.35, 0.85, 1.15, -0.65, 0.05, -0.95, 1.7, -2.2, 0.45, 0.9,
        -1.05, 0.6,
    ];
    let upper_vals = vec![
        -0.15_f32, 1.25, -0.85, 0.95, 2.05, -0.55, 0.35, 1.45, -1.35, 0.75, -0.25, 1.05, -0.5,
        2.15, 0.65, -1.75,
    ];
    let make_side = |vals: Vec<f32>, err: Vec<f32>| PatchesData {
        coeff_err: Some(Array1::from_vec(err)),
        patches: Some(ArrayD::from_shape_vec(IxDyn(&[2, 1, 2, 2, 2, 1]), vals).unwrap()),
        stride: (1, 1),
        // pad_top = 1: ki=0 taps are padding (invalid), ki=1 taps valid.
        padding: (0, 0, 1, 0),
        identity: false,
        output_shape: (2, 1, 2),
        input_shape: (2, 1, 2),
        unstable_idx: None,
    };
    let bounds = PatchesLinearBounds {
        row_count: 4,
        lower_a: make_side(lower_vals, vec![1.0e-3, 5.0e-4, 0.0, 2.5e-4]),
        lower_b: Array1::from_vec(vec![0.1_f32, -0.2, 0.3, -0.4]),
        upper_a: make_side(upper_vals, vec![2.0e-3, 0.0, 7.5e-4, 1.0e-4]),
        upper_b: Array1::from_vec(vec![0.5_f32, 0.6, -0.7, 0.8]),
    };

    let result = layer
        .propagate_patches(&bounds)
        .expect("bn patches backward");
    let pb = match result {
        CrownBounds::Patches(pb) => pb,
        CrownBounds::Dense(_) => panic!("expected Patches output from BatchNorm backward"),
    };

    assert_eq!(pb.row_count, 4);
    assert_eq!(pb.lower_a.stride, (1, 1));
    assert_eq!(pb.lower_a.padding, (0, 0, 1, 0));
    assert_eq!(pb.lower_a.output_shape, (2, 1, 2));
    assert_eq!(pb.lower_a.input_shape, (2, 1, 2));
    assert!(!pb.lower_a.identity && !pb.upper_a.identity);
    assert!(pb.lower_a.unstable_idx.is_none() && pb.upper_a.unstable_idx.is_none());

    // Bit literals captured from pre-change HEAD (see doc comment).
    const EXP_LOWER_PATCHES: [u32; 16] = [
        0x3F451EB8, 0xBFB70A3D, 0xBEC51EB8, 0xBFD70A3E, 0xBEC51EB8, 0x3F6F5C2A, 0xBF4E147A,
        0x3EE8F5C2, 0x3D6147AF, 0xBF85C28F, 0xBF9851EC, 0x3FC51EB8, 0x3EFD70A4, 0x3F7D70A4,
        0x3F3C28F5, 0xBED70A3E,
    ];
    const EXP_UPPER_PATCHES: [u32; 16] = [
        0xBE28F5C3, 0x3FB00000, 0x3F1851EC, 0xBF2A3D70, 0x401051EC, 0xBF1AE148, 0xBE7AE147,
        0xBF81EB85, 0xBFBE147B, 0x3F533334, 0x3E333333, 0xBF3C28F5, 0xBF0CCCCD, 0x40175C2A,
        0xBEE8F5C2, 0x3F9CCCCD,
    ];
    const EXP_LOWER_B: [u32; 4] = [0xC01CE500, 0x3F23A446, 0x3FFF556D, 0xBF2BA4DB];
    const EXP_UPPER_B: [u32; 4] = [0x3CB99A8B, 0xBF5EB071, 0xBFB59FE8, 0x404150E5];
    const EXP_LOWER_ERR: [u32; 4] = [0x3AA9C4E4, 0x3A31A309, 0x39157CC7, 0x39D7023A];
    const EXP_UPPER_ERR: [u32; 4] = [0x3B1C7E05, 0x39A149FF, 0x3A86B3EE, 0x39E2D61C];

    let lower_patches: Vec<f32> = pb
        .lower_a
        .patches
        .as_ref()
        .unwrap()
        .iter()
        .copied()
        .collect();
    let upper_patches: Vec<f32> = pb
        .upper_a
        .patches
        .as_ref()
        .unwrap()
        .iter()
        .copied()
        .collect();
    let lower_err: Vec<f32> = pb
        .lower_a
        .coeff_err
        .as_ref()
        .expect("6D BN backward must emit lower coeff_err")
        .to_vec();
    let upper_err: Vec<f32> = pb
        .upper_a
        .coeff_err
        .as_ref()
        .expect("6D BN backward must emit upper coeff_err")
        .to_vec();

    check_bit_pins(&[
        ("EXP_LOWER_PATCHES", &lower_patches, &EXP_LOWER_PATCHES),
        ("EXP_UPPER_PATCHES", &upper_patches, &EXP_UPPER_PATCHES),
        ("EXP_LOWER_B", pb.lower_b.as_slice().unwrap(), &EXP_LOWER_B),
        ("EXP_UPPER_B", pb.upper_b.as_slice().unwrap(), &EXP_UPPER_B),
        ("EXP_LOWER_ERR", &lower_err, &EXP_LOWER_ERR),
        ("EXP_UPPER_ERR", &upper_err, &EXP_UPPER_ERR),
    ]);
}

/// Compare f32 slices against pinned bit literals. On ANY mismatch, dump ALL
/// actual arrays in copy-pastable `const` form (one-shot capture from the
/// pre-change tree), then panic. NaN bit patterns must also match exactly.
fn check_bit_pins(pins: &[(&str, &[f32], &[u32])]) {
    let mut mismatch = false;
    for (label, actual, expected) in pins {
        let bits: Vec<u32> = actual.iter().map(|v| v.to_bits()).collect();
        if bits.as_slice() != *expected {
            mismatch = true;
            eprintln!("PIN MISMATCH: {label}");
        }
    }
    if mismatch {
        for (label, actual, _) in pins {
            let dump: Vec<String> = actual
                .iter()
                .map(|v| format!("{:#010X}", v.to_bits()))
                .collect();
            eprintln!(
                "const {label}: [u32; {}] = [{}];",
                actual.len(),
                dump.join(", ")
            );
        }
        panic!("byte-identity pin mismatch — actual bit arrays dumped above");
    }
}

/// Root-cause regression for the cifar100_2024 graph beta-CROWN abort.
///
/// `BatchNormLayer::new`'s guard only rejects `var + eps < 0`; a channel with
/// `var + eps == 0` (e.g. `var = -eps`) yields `scale = ny / sqrt(0) = +inf`.
/// In CROWN backward the column coefficients are scaled by `scale`. A zero
/// incoming coefficient times the Inf scale is `0 * inf`, which with a plain
/// multiply is `NaN`. That NaN poisoned the linear bounds and ultimately
/// aborted intermediate-bound construction at `BoundedTensor::new` (NaN/Inf
/// rejection) in the graph beta-CROWN BaB clip path.
///
/// After the fix the BatchNorm CROWN backward uses `safe_mul_for_bounds`
/// (`0 * inf = 0`): a zero coefficient composes to exactly 0, so no NaN is
/// produced. A nonzero coefficient times the Inf scale yields a ±Inf
/// coefficient, which `new_or_conservative` / concretize handle soundly.
/// This test asserts the backward output contains NO NaN.
#[test]
fn test_crown_backward_inf_scale_no_nan_from_zero_coeff_4xxx() {
    // var = -eps -> var + eps == 0 -> scale[0] = inf; channel 1 well-behaved.
    let eps = 1e-5_f32;
    let ny = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap();
    let beta = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let mean = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).unwrap();
    let var = ArrayD::from_shape_vec(IxDyn(&[2]), vec![-eps, 1.0]).unwrap();
    let layer = BatchNormLayer::new(&ny, &beta, &mean, &var, eps).unwrap();
    assert!(
        !layer.scale[[0]].is_finite(),
        "test precondition: channel-0 scale must be inf, got {}",
        layer.scale[[0]]
    );

    // pre-activation shape [2] -> channel axis 0, one element per channel.
    let pre_act = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![-1.0, -1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 1.0]).unwrap(),
    )
    .unwrap();

    // Incoming bounds whose first column coefficient is 0 (so column 0 hits the
    // 0 * inf case) and second is 1.
    let lower_a = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
    let upper_a = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
    let bounds = LinearBounds::new(lower_a, Array1::zeros(1), upper_a, Array1::zeros(1)).unwrap();

    // Scalar backward must not produce NaN coefficients or NaN bias.
    let out = layer
        .propagate_linear_with_bounds(&bounds, &pre_act)
        .expect("scalar CROWN backward must not error on inf-scale BatchNorm");
    assert!(
        out.lower_a().iter().all(|v| !v.is_nan()) && out.upper_a().iter().all(|v| !v.is_nan()),
        "scalar backward produced NaN coefficient (0 * inf): lower_a={:?} upper_a={:?}",
        out.lower_a(),
        out.upper_a()
    );
    assert!(
        out.lower_b().iter().all(|v| !v.is_nan()) && out.upper_b().iter().all(|v| !v.is_nan()),
        "scalar backward produced NaN bias"
    );
    // Precision: without the 0*inf=0 fix the whole matrix would have NaN'd and
    // collapsed to the conservative firewall (A=0, b=±inf). With the fix the
    // degenerate column composes to exactly 0 while the well-behaved channel-1
    // column keeps its finite coefficient (~scale[1] = 1/sqrt(1+eps) ~= 1) and
    // the bias stays finite. Assert that to prove no whole-matrix degradation.
    assert!(
        out.lower_a()[[0, 0]] == 0.0 && out.upper_a()[[0, 0]] == 0.0,
        "degenerate column should compose to exactly 0, got [{}, {}]",
        out.lower_a()[[0, 0]],
        out.upper_a()[[0, 0]]
    );
    assert!(
        (out.lower_a()[[0, 1]] - layer.scale[[1]]).abs() < 1e-4,
        "well-behaved column coefficient should be preserved (~{}), got {} \
         (whole-matrix conservative fallback would zero it)",
        layer.scale[[1]],
        out.lower_a()[[0, 1]]
    );
    assert!(
        out.lower_b().iter().all(|v| v.is_finite()) && out.upper_b().iter().all(|v| v.is_finite()),
        "bias should stay finite (conservative fallback would set ±inf): lower_b={:?} upper_b={:?}",
        out.lower_b(),
        out.upper_b()
    );

    // Batched backward must likewise stay NaN-free and avoid whole-matrix
    // degradation. Identity incoming has off-diagonal zeros in the channel-0
    // (Inf-scale) column, which is exactly the 0*inf case.
    let incoming = BatchedLinearBounds::identity(&[2]).unwrap();
    let out_b = layer
        .propagate_linear_batched_with_bounds(&incoming, &pre_act)
        .expect("batched CROWN backward must not error on inf-scale BatchNorm");
    assert!(
        out_b.lower_a.iter().all(|v| !v.is_nan()) && out_b.upper_a.iter().all(|v| !v.is_nan()),
        "batched backward produced NaN coefficient (0 * inf)"
    );
    assert!(
        out_b.lower_b.iter().all(|v| !v.is_nan()) && out_b.upper_b.iter().all(|v| !v.is_nan()),
        "batched backward produced NaN bias"
    );
    // Channel-1 diagonal coefficient preserved (no conservative collapse).
    assert!(
        (out_b.lower_a[[1, 1]] - layer.scale[[1]]).abs() < 1e-4,
        "batched well-behaved column coefficient should be preserved (~{}), got {}",
        layer.scale[[1]],
        out_b.lower_a[[1, 1]]
    );
    assert!(
        out_b.lower_b.iter().all(|v| v.is_finite()) && out_b.upper_b.iter().all(|v| v.is_finite()),
        "batched bias should stay finite (conservative fallback would set ±inf)"
    );
}

// --- 7D explicit-rows coeff_err closure: Site 5 (BatchNorm) tests ---
// (docs/PATCHES_7D_COEFF_ERR_CLOSURE.md §8.4; the T2 byte-identity pin lives
// above as `test_bn_patches_6d_coeff_err_byte_identical_regression`; T6 is the
// integration roundtrip and lands with the full Site 1 + Site I chain.)

/// Deterministic non-dyadic fill with exact zeros sprinkled in (spec §8.4:
/// products must round; zeros exercise the structural-zero / ABS-skip cases).
fn lcg_vals(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if i % 17 == 0 {
                0.0
            } else {
                let u = ((s >> 33) as f64) / f64::from(1u32 << 31); // [0, 1)
                ((u * 2.0 - 1.0) * 1.25) as f32
            }
        })
        .collect()
}

/// Shared §8.4 fixture: 7D [row=2, oc=2, oh=2, ow=2, ic=3, kh=2, kw=2]
/// (192 coeffs/side), stride (1,1), padding (1,0,1,0) mixing valid and padding
/// taps (geometry-consistent: out 2x2 from in 2x2 with k 2x2), and a BN layer
/// built as a struct literal with NONZERO scale_err/bias_err
/// (`from_scale_bias` zeroes them — deliberately not used).
fn make_bn_7d_fixture(
    lower_err: Option<Vec<f32>>,
    upper_err: Option<Vec<f32>>,
) -> (BatchNormLayer, crate::bounds::patches::PatchesLinearBounds) {
    use crate::bounds::patches::{PatchesData, PatchesLinearBounds};
    let layer = BatchNormLayer {
        scale: ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.1_f32, -0.7, 0.35]).unwrap(),
        bias: ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.3_f32, -0.9, 0.55]).unwrap(),
        scale_err: ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.5e-4_f32, 3.0e-5, 8.0e-5]).unwrap(),
        bias_err: ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0e-4_f32, 7.0e-6, 4.5e-5]).unwrap(),
        num_channels: 3,
    };
    let shape = [2usize, 2, 2, 2, 3, 2, 2];
    let n: usize = shape.iter().product();
    let make_side = |seed: u64, err: Option<Vec<f32>>| PatchesData {
        coeff_err: err.map(Array1::from_vec),
        patches: Some(ArrayD::from_shape_vec(IxDyn(&shape), lcg_vals(seed, n)).unwrap()),
        stride: (1, 1),
        // pad_left = 1, pad_top = 1: the (oh=0, ki=0) rows / (ow=0, kj=0) cols
        // map outside the 2x2 input (padding, excluded from the bias fold);
        // every other tap is valid — both predicate arms exercised.
        padding: (1, 0, 1, 0),
        identity: false,
        output_shape: (2, 2, 2),
        input_shape: (3, 2, 2),
        unstable_idx: None,
    };
    let bounds = PatchesLinearBounds {
        row_count: 2,
        lower_a: make_side(0x5EED_0001, lower_err),
        lower_b: Array1::from_vec(vec![0.13_f32, -0.41]),
        upper_a: make_side(0x5EED_0002, upper_err),
        upper_b: Array1::from_vec(vec![0.57_f32, 0.29]),
    };
    (layer, bounds)
}

/// Fixture valid-tap predicate replica (sh = sw = 1, pad_top = pad_left = 1,
/// in_h = in_w = 2) — matches the HOLE1 predicate in the implementation.
fn bn_7d_tap_valid(oh: usize, ow: usize, ki: usize, kj: usize) -> bool {
    let ih = (oh + ki) as isize - 1;
    let iw = (ow + kj) as isize - 1;
    ih >= 0 && (ih as usize) < 2 && iw >= 0 && (iw as usize) < 2
}

/// §8.4 T1: f64 oracle coverage on the 7D explicit-rows arm.
///
/// Coefficients: per tap, 4 adversarial corners `a_true = a ± oe`,
/// `s_real = scale ∓ se` — the emitted spec-row err must cover
/// `|stored − a_true·s_real|` at every tap of the row (max-lift).
///
/// Bias: `F_min`/`F_max` over per-tap-independent corners
/// `(a ± oe)·(bias ∓ be)` summed over VALID taps only (sum-lift across all
/// positions into the one spec-row slot). The corners DELIBERATELY include the
/// `oe·be` cross-mass (~4.5e-6 here, ≳4x the directed-cast slack at this bias
/// magnitude), so a verbatim-6D widen (`|a|·be + oe·bb`, missing `oe·be`)
/// FAILS this oracle — pins the R5 cross-term fix.
#[test]
fn test_bn_patches_7d_coeff_err_covers_f64_oracle() {
    use crate::bounds::patches::{CrownBounds, PatchesData};
    use crate::layers::common::PatchesPropagation;

    let (layer, bounds) = make_bn_7d_fixture(Some(vec![1.0e-3, 5.0e-4]), Some(vec![2.0e-3, 0.0]));
    let result = layer.propagate_patches(&bounds).expect("bn 7d backward");
    let pb = match result {
        CrownBounds::Patches(pb) => pb,
        CrownBounds::Dense(_) => panic!("expected Patches output"),
    };
    assert_eq!(pb.row_count, 2);

    // Returns (F_min, F_max): outward per-tap-independent corner folds of the
    // true bias for this side, per spec row.
    let check_side = |pin: &PatchesData,
                      pout: &PatchesData,
                      bias_in: &Array1<f32>,
                      side: &str|
     -> (Vec<f64>, Vec<f64>) {
        let a_in = pin.patches.as_ref().unwrap();
        let a_out = pout.patches.as_ref().unwrap();
        let err_in = pin.coeff_err.as_ref().unwrap();
        let err_out = pout
            .coeff_err
            .as_ref()
            .unwrap_or_else(|| panic!("{side}: 7D BN backward must emit coeff_err"));
        assert_eq!(err_out.len(), 2, "{side}: err len must be row_count");
        let mut f_min = vec![0.0f64; 2];
        let mut f_max = vec![0.0f64; 2];
        for row in 0..2 {
            let oe = f64::from(err_in[row]);
            let ne = f64::from(err_out[row]);
            assert!(
                ne.is_finite() && ne >= 0.0,
                "{side} row {row}: err must be finite and >= 0, got {ne}"
            );
            f_min[row] = f64::from(bias_in[row]);
            f_max[row] = f64::from(bias_in[row]);
            for oc in 0..2 {
                for oh in 0..2 {
                    for ow in 0..2 {
                        for ic in 0..3 {
                            let s = f64::from(layer.scale[[ic]]);
                            let se = f64::from(layer.scale_err[[ic]]);
                            let b = f64::from(layer.bias[[ic]]);
                            let be = f64::from(layer.bias_err[[ic]]);
                            for ki in 0..2 {
                                for kj in 0..2 {
                                    let a = f64::from(a_in[[row, oc, oh, ow, ic, ki, kj]]);
                                    let stored = f64::from(a_out[[row, oc, oh, ow, ic, ki, kj]]);
                                    for da in [-oe, oe] {
                                        for ds in [-se, se] {
                                            let truth = (a + da) * (s + ds);
                                            assert!(
                                                (stored - truth).abs() <= ne,
                                                "{side} row {row} tap ({oc},{oh},{ow},{ic},{ki},{kj}): \
                                                 |{stored} - {truth}| > err {ne}"
                                            );
                                        }
                                    }
                                    if bn_7d_tap_valid(oh, ow, ki, kj) {
                                        let mut lo = f64::INFINITY;
                                        let mut hi = f64::NEG_INFINITY;
                                        for da in [-oe, oe] {
                                            for db in [-be, be] {
                                                let v = (a + da) * (b + db);
                                                lo = lo.min(v);
                                                hi = hi.max(v);
                                            }
                                        }
                                        f_min[row] += lo;
                                        f_max[row] += hi;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        (f_min, f_max)
    };

    let (l_min, _) = check_side(&bounds.lower_a, &pb.lower_a, &bounds.lower_b, "lower");
    let (_, u_max) = check_side(&bounds.upper_a, &pb.upper_a, &bounds.upper_b, "upper");
    for row in 0..2 {
        let lb = f64::from(pb.lower_b[row]);
        let ub = f64::from(pb.upper_b[row]);
        assert!(
            lb <= l_min[row],
            "row {row}: lower bias {lb} must be <= adversarial-corner minimum {} \
             (a verbatim-6D widen without the oe·be cross term under-covers here)",
            l_min[row]
        );
        assert!(
            ub >= u_max[row],
            "row {row}: upper bias {ub} must be >= adversarial-corner maximum {}",
            u_max[row]
        );
    }
}

/// §8.4 T3: a carried `Some` err whose length != row_count is a hard
/// `Err(ShapeMismatch)` (=> the caller's sound dense-BN fallback), never a
/// silent `.get().unwrap_or(0.0)` under-count (spec I6/R6). Both sides.
#[test]
fn test_bn_patches_7d_err_length_mismatch_errors() {
    use crate::layers::common::PatchesPropagation;
    use ny_core::NyError;

    let (layer, bounds) = make_bn_7d_fixture(Some(vec![1.0e-3, 5.0e-4, 1.0e-5]), None);
    match layer.propagate_patches(&bounds) {
        Err(NyError::ShapeMismatch { expected, got }) => {
            assert_eq!(expected, vec![2]);
            assert_eq!(got, vec![3]);
        }
        other => panic!("lower err len 3 vs row_count 2 must be ShapeMismatch, got {other:?}"),
    }

    let (layer, bounds) = make_bn_7d_fixture(None, Some(vec![1.0e-3]));
    assert!(
        matches!(
            layer.propagate_patches(&bounds),
            Err(NyError::ShapeMismatch { .. })
        ),
        "upper err len 1 vs row_count 2 must be ShapeMismatch"
    );
}

/// §8.4 T4: `oe = +INF` with `bias = bias_err = 0` — the would-be
/// `INF·(bb+be) = INF·0 = NaN` corner. The poisoned row must degrade outward
/// (err `+INF`, lower bias `-INF`) with NO NaN anywhere; the clean row stays
/// finite. NaN / negative carried errs poison identically (I5).
#[test]
fn test_bn_patches_7d_nonfinite_err_poisons_row_no_nan() {
    use crate::bounds::patches::{CrownBounds, PatchesLinearBounds};
    use crate::layers::common::PatchesPropagation;

    // Zero bias AND zero bias_err: the poison arm must not touch them (skip
    // accumulation entirely — no INF·0).
    let layer = BatchNormLayer {
        scale: ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.1_f32, -0.7, 0.35]).unwrap(),
        bias: ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 0.0, 0.0]).unwrap(),
        scale_err: ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.5e-4_f32, 3.0e-5, 8.0e-5]).unwrap(),
        bias_err: ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 0.0, 0.0]).unwrap(),
        num_channels: 3,
    };

    let run = |bounds: &PatchesLinearBounds| {
        let result = layer.propagate_patches(bounds).expect("bn 7d backward");
        let pb = match result {
            CrownBounds::Patches(pb) => pb,
            CrownBounds::Dense(_) => panic!("expected Patches output"),
        };
        // NO NaN anywhere: coefficients, biases, errs.
        for (name, arr) in [
            ("lower_a", pb.lower_a.patches.as_ref().unwrap()),
            ("upper_a", pb.upper_a.patches.as_ref().unwrap()),
        ] {
            assert!(
                arr.iter().all(|v| !v.is_nan()),
                "{name} contains NaN coefficients"
            );
        }
        assert!(pb.lower_b.iter().all(|v| !v.is_nan()), "lower_b has NaN");
        assert!(pb.upper_b.iter().all(|v| !v.is_nan()), "upper_b has NaN");
        for side in [&pb.lower_a, &pb.upper_a] {
            let e = side.coeff_err.as_ref().expect("7D arm must emit Some err");
            assert!(e.iter().all(|v| !v.is_nan()), "coeff_err has NaN");
        }
        pb
    };

    let (_, bounds) = make_bn_7d_fixture(Some(vec![f32::INFINITY, 1.0e-4]), Some(vec![0.0, 0.0]));
    let pb = run(&bounds);
    let le = pb.lower_a.coeff_err.as_ref().unwrap();
    assert_eq!(le[0], f32::INFINITY, "poisoned row 0 err must be +INF");
    assert_eq!(
        pb.lower_b[0],
        f32::NEG_INFINITY,
        "poisoned row 0 lower bias must discharge to -INF (vacuous)"
    );
    assert!(le[1].is_finite(), "clean row 1 err must stay finite");
    assert!(
        pb.lower_b[1].is_finite(),
        "clean row 1 lower bias must stay finite"
    );
    // The upper side (its own err channel) is unaffected by the lower poison.
    let ue = pb.upper_a.coeff_err.as_ref().unwrap();
    assert!(
        ue.iter().all(|v| v.is_finite()),
        "upper errs must be finite"
    );
    assert!(
        pb.upper_b.iter().all(|v| v.is_finite()),
        "upper biases must stay finite"
    );

    // NaN and negative carried errs poison the same way (I5: NEVER NaN -> 0).
    let (_, bounds) = make_bn_7d_fixture(Some(vec![f32::NAN, -1.0]), Some(vec![0.0, 0.0]));
    let pb = run(&bounds);
    let le = pb.lower_a.coeff_err.as_ref().unwrap();
    assert_eq!(le[0], f32::INFINITY, "NaN carried err must poison to +INF");
    assert_eq!(
        le[1],
        f32::INFINITY,
        "negative carried err must poison to +INF"
    );
    assert_eq!(pb.lower_b[0], f32::NEG_INFINITY);
    assert_eq!(pb.lower_b[1], f32::NEG_INFINITY);
}

/// §8.4 T5: the err channel is read-only over values (I3). An err-carrying run
/// (nonzero incoming errs, nonzero layer errs) must produce BIT-IDENTICAL
/// coefficient tensors to an err-free run with the zero-err layer, and its
/// biases must enclose the err-free run's. Also pins that the 7D arm emits
/// `Some` even for `None` incoming err (the fold discharge is intrinsic).
#[test]
fn test_bn_patches_7d_err_channel_does_not_perturb_values() {
    use crate::bounds::patches::CrownBounds;
    use crate::layers::common::PatchesPropagation;

    let (layer_a, bounds_a) =
        make_bn_7d_fixture(Some(vec![1.0e-3, 5.0e-4]), Some(vec![2.0e-3, 0.0]));
    let (_, bounds_b) = make_bn_7d_fixture(None, None);
    let layer_b = BatchNormLayer {
        scale: layer_a.scale.clone(),
        bias: layer_a.bias.clone(),
        scale_err: ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 0.0, 0.0]).unwrap(),
        bias_err: ArrayD::from_shape_vec(IxDyn(&[3]), vec![0.0_f32, 0.0, 0.0]).unwrap(),
        num_channels: 3,
    };

    let run = |layer: &BatchNormLayer, bounds| match layer.propagate_patches(bounds) {
        Ok(CrownBounds::Patches(pb)) => pb,
        other => panic!("expected Patches output, got {other:?}"),
    };
    let pa = run(&layer_a, &bounds_a);
    let pb = run(&layer_b, &bounds_b);

    for (name, a, b) in [
        ("lower_a", &pa.lower_a, &pb.lower_a),
        ("upper_a", &pa.upper_a, &pb.upper_a),
    ] {
        let av = a.patches.as_ref().unwrap();
        let bv = b.patches.as_ref().unwrap();
        assert_eq!(av.shape(), bv.shape());
        assert!(
            av.iter()
                .zip(bv.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "{name}: err channel perturbed coefficient VALUES (must be bit-identical)"
        );
    }
    // None-in still emits Some out on the 7D arm.
    assert!(pb.lower_a.coeff_err.is_some() && pb.upper_a.coeff_err.is_some());
    for row in 0..2 {
        assert!(
            pa.lower_b[row] <= pb.lower_b[row],
            "row {row}: err-carrying lower bias must enclose (<=) the err-free one"
        );
        assert!(
            pa.upper_b[row] >= pb.upper_b[row],
            "row {row}: err-carrying upper bias must enclose (>=) the err-free one"
        );
    }
}

/// DIFFERENTIAL ORACLE for the vectorized `propagate_linear_with_bounds`.
///
/// The production dense CROWN-backward path was rewritten from a scalar
/// double-loop into a vectorized (ndarray broadcast + contiguous-row) form. The
/// original body is retained verbatim as
/// `propagate_linear_with_bounds_scalar_reference`. This test drives BOTH over
/// random shapes/layouts — including nonzero layer `scale_err`/`bias_err`,
/// incoming certified coeff-err on both sides, exact zeros in the coefficient
/// matrices, and a degenerate Inf-scale/Inf-bias channel — and asserts:
///   (1) BIT-IDENTITY of the coefficient matrices AND the emitted coeff-err
///       (the vectorization must not perturb any value), and
///   (2) the task-specified numeric criteria: main (coefficient) terms match to
///       rel-err <= 1e-5, and the error-widened biases satisfy the SOUNDNESS
///       direction (vec.lower_b <= scalar.lower_b + 1e-6 and
///       vec.upper_b >= scalar.upper_b - 1e-6) on finite rows.
#[test]
fn test_propagate_linear_vectorized_matches_scalar_reference() {
    // Tiny deterministic LCG — no `rand` dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next_u32() as f32) / (u32::MAX as f32)
        }
        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + self.unit() * (hi - lo)
        }
    }

    // (input_shape, num_channels, num_outputs)
    let configs: &[(&[usize], usize, usize)] = &[
        (&[4], 4, 5),
        (&[2, 3, 3], 2, 6),
        (&[3, 2, 2], 3, 4),
        (&[5], 5, 3),
        (&[1, 4, 4], 1, 7),
    ];

    let mut rng = Lcg(0x1234_5678_9abc_def0);

    for &(shape, num_channels, num_outputs) in configs {
        let num_inputs: usize = shape.iter().product();

        for &with_coeff_err in &[false, true] {
            for &inf_channel in &[false, true] {
                // Per-channel scale/bias and their certified precompute errors.
                let mut scale_v = vec![0.0f32; num_channels];
                let mut bias_v = vec![0.0f32; num_channels];
                let mut scale_err_v = vec![0.0f32; num_channels];
                let mut bias_err_v = vec![0.0f32; num_channels];
                for c in 0..num_channels {
                    scale_v[c] = rng.range(-2.0, 2.0);
                    bias_v[c] = rng.range(-1.0, 1.0);
                    // Nonzero precompute error (soundness margin) on every channel.
                    scale_err_v[c] = rng.range(0.0, 2.0e-4);
                    bias_err_v[c] = rng.range(0.0, 2.0e-4);
                }
                if inf_channel {
                    // Degenerate var+eps -> 0 channel: Inf scale/bias/errs.
                    scale_v[0] = f32::INFINITY;
                    bias_v[0] = f32::INFINITY;
                    scale_err_v[0] = f32::INFINITY;
                    bias_err_v[0] = f32::INFINITY;
                }

                let layer = BatchNormLayer {
                    scale: ArrayD::from_shape_vec(IxDyn(&[num_channels]), scale_v).unwrap(),
                    bias: ArrayD::from_shape_vec(IxDyn(&[num_channels]), bias_v).unwrap(),
                    scale_err: ArrayD::from_shape_vec(IxDyn(&[num_channels]), scale_err_v).unwrap(),
                    bias_err: ArrayD::from_shape_vec(IxDyn(&[num_channels]), bias_err_v).unwrap(),
                    num_channels,
                };

                // Incoming linear bounds: finite A (with ~20% exact zeros to
                // exercise the 0*inf=0 short-circuit), finite b.
                let gen_a = |rng: &mut Lcg| {
                    Array2::from_shape_fn((num_outputs, num_inputs), |_| {
                        if rng.unit() < 0.2 {
                            0.0
                        } else {
                            rng.range(-2.0, 2.0)
                        }
                    })
                };
                let lower_a = gen_a(&mut rng);
                let upper_a = gen_a(&mut rng);
                let lower_b = Array1::from_shape_fn(num_outputs, |_| rng.range(-1.0, 1.0));
                let upper_b = Array1::from_shape_fn(num_outputs, |_| rng.range(-1.0, 1.0));
                let mut bounds = LinearBounds::new(lower_a, lower_b, upper_a, upper_b).unwrap();

                if with_coeff_err {
                    // Non-negative certified coeff-err, ~30% exact zeros.
                    let gen_err = |rng: &mut Lcg| {
                        Array2::from_shape_fn((num_outputs, num_inputs), |_| {
                            if rng.unit() < 0.3 {
                                0.0
                            } else {
                                rng.range(0.0, 1.0e-3)
                            }
                        })
                    };
                    let le = gen_err(&mut rng);
                    let ue = gen_err(&mut rng);
                    bounds.set_coeff_err(le, ue);
                }

                // pre-activation box: lower <= upper elementwise, finite.
                let pre_lo = ArrayD::from_shape_fn(IxDyn(shape), |_| rng.range(-1.0, 0.2));
                let pre_hi = {
                    let mut hi = pre_lo.clone();
                    hi.mapv_inplace(|lo| lo + rng.range(0.0, 1.5));
                    hi
                };
                let pre_act = BoundedTensor::new(pre_lo, pre_hi).unwrap();

                let vec_res = layer
                    .propagate_linear_with_bounds(&bounds, &pre_act)
                    .expect("vectorized path");
                let ref_res = layer
                    .propagate_linear_with_bounds_scalar_reference(&bounds, &pre_act)
                    .expect("scalar reference path");

                let tag = format!("shape={shape:?} coeff_err={with_coeff_err} inf={inf_channel}");

                // (1) BIT-IDENTITY on the coefficient matrices.
                let bits_eq = |a: &Array2<f32>, b: &Array2<f32>| {
                    a.shape() == b.shape()
                        && a.iter()
                            .zip(b.iter())
                            .all(|(x, y)| x.to_bits() == y.to_bits())
                };
                assert!(
                    bits_eq(vec_res.lower_a(), ref_res.lower_a()),
                    "{tag}: lower_a must be bit-identical to the scalar reference"
                );
                assert!(
                    bits_eq(vec_res.upper_a(), ref_res.upper_a()),
                    "{tag}: upper_a must be bit-identical to the scalar reference"
                );
                // (1b) BIT-IDENTITY on the emitted coeff-err (the soundness margin).
                assert_eq!(
                    vec_res.lower_a_err().is_some(),
                    ref_res.lower_a_err().is_some(),
                    "{tag}: lower coeff-err presence must match"
                );
                if let (Some(v), Some(r)) = (vec_res.lower_a_err(), ref_res.lower_a_err()) {
                    assert!(
                        bits_eq(v, r),
                        "{tag}: lower coeff-err must be bit-identical"
                    );
                }
                if let (Some(v), Some(r)) = (vec_res.upper_a_err(), ref_res.upper_a_err()) {
                    assert!(
                        bits_eq(v, r),
                        "{tag}: upper coeff-err must be bit-identical"
                    );
                }
                // Bias bit-identity (strongest guarantee; complements (2)).
                assert!(
                    vec_res
                        .lower_b()
                        .iter()
                        .zip(ref_res.lower_b().iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{tag}: lower_b must be bit-identical to the scalar reference"
                );
                assert!(
                    vec_res
                        .upper_b()
                        .iter()
                        .zip(ref_res.upper_b().iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{tag}: upper_b must be bit-identical to the scalar reference"
                );

                // (2) Task-specified numeric criteria on finite elements.
                //   main coefficient terms: rel-err <= 1e-5.
                for (v, r) in vec_res.lower_a().iter().zip(ref_res.lower_a().iter()) {
                    if v.is_finite() && r.is_finite() {
                        let rel = (v - r).abs() / (r.abs().max(1.0));
                        assert!(rel <= 1e-5, "{tag}: lower_a rel-err {rel} > 1e-5");
                    }
                }
                for (v, r) in vec_res.upper_a().iter().zip(ref_res.upper_a().iter()) {
                    if v.is_finite() && r.is_finite() {
                        let rel = (v - r).abs() / (r.abs().max(1.0));
                        assert!(rel <= 1e-5, "{tag}: upper_a rel-err {rel} > 1e-5");
                    }
                }
                //   error-widened biases: SOUNDNESS direction.
                for (v, r) in vec_res.lower_b().iter().zip(ref_res.lower_b().iter()) {
                    if v.is_finite() && r.is_finite() {
                        assert!(
                            *v <= *r + 1e-6,
                            "{tag}: vec lower_b {v} must be <= scalar {r} + 1e-6 (sound)"
                        );
                    }
                }
                for (v, r) in vec_res.upper_b().iter().zip(ref_res.upper_b().iter()) {
                    if v.is_finite() && r.is_finite() {
                        assert!(
                            *v >= *r - 1e-6,
                            "{tag}: vec upper_b {v} must be >= scalar {r} - 1e-6 (sound)"
                        );
                    }
                }
            }
        }
    }
}
