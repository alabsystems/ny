// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for pooling layers (AveragePool, MaxPool2d).

use super::*;
use crate::layers::common::BoundPropagation;
use crate::LinearBounds;
use ndarray::{ArrayD, IxDyn};
use ny_tensor::BoundedTensor;

// ===== AveragePoolLayer tests =====

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_output_size_basic() -> Result<()> {
    // 4x4 input, kernel 2x2, stride 2, no padding → 2x2
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    let (oh, ow) = layer.output_size(4, 4)?;
    assert_eq!((oh, ow), (2, 2));
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_output_size_with_padding() -> Result<()> {
    // 3x3 input, kernel 3x3, stride 1, padding 1 → 3x3
    let layer = AveragePoolLayer::new((3, 3), (1, 1), (1, 1), false);
    let (oh, ow) = layer.output_size(3, 3)?;
    assert_eq!((oh, ow), (3, 3));
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_output_size_zero_stride() {
    let layer = AveragePoolLayer::new((2, 2), (0, 0), (0, 0), false);
    let err = layer.output_size(4, 4).expect_err("zero stride");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_global() -> Result<()> {
    let layer = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);
    assert!(layer.is_global());
    let (oh, ow) = layer.output_size(10, 10)?;
    assert_eq!((oh, ow), (1, 1));
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_rejects_padding_ge_kernel() -> Result<()> {
    // padding >= kernel creates pooling windows made entirely of padding;
    // such a window averages no inputs (0/0 under count_include_pad=false),
    // so the geometry must be rejected rather than given a fabricated value.
    for layer in [
        AveragePoolLayer::new((2, 2), (2, 2), (3, 3), false),
        AveragePoolLayer::new((2, 2), (2, 2), (2, 2), false),
        AveragePoolLayer::new((2, 3), (1, 1), (2, 1), false), // height dimension only
        AveragePoolLayer::new((3, 2), (1, 1), (1, 2), false), // width dimension only
        AveragePoolLayer::new((2, 2), (2, 2), (2, 2), true),  // count_include_pad too
    ] {
        let err = layer
            .output_size(4, 4)
            .expect_err("padding >= kernel must be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }
    // Boundary: padding == kernel - 1 is still accepted.
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (1, 1), false);
    assert_eq!(layer.output_size(4, 4)?, (3, 3));
    // The global-pool kernel (0, 0) sentinel stays accepted regardless of padding.
    let layer = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);
    assert_eq!(layer.output_size(10, 10)?, (1, 1));
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_padding_ge_kernel_fails_closed() {
    // Every propagation path must refuse the all-padding-window geometry:
    // IBP would emit a fabricated 0 for such windows (count.max(1) guards
    // the 0/0 divisor) and the CROWN backward would silently drop their
    // coefficients.
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (3, 3), false);
    let lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 5.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 6.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let err = layer
        .propagate_ibp(&input)
        .expect_err("IBP must reject all-padding windows");
    assert!(matches!(err, NyError::InvalidSpec(_)));

    // Would-be 5x5 output under the degenerate geometry.
    let bounds = LinearBounds::identity(25);
    let err = layer
        .propagate_linear_with_bounds(&bounds, &input)
        .expect_err("CROWN must reject all-padding windows");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_ibp_3d_point_interval() -> Result<()> {
    // 1 channel, 4x4 input with non-uniform values, kernel 2x2, stride 2
    // Point interval: lower == upper, so output must equal exact average
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    // Non-uniform values so an identity function would NOT pass
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let arr = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), data).unwrap();
    let input = BoundedTensor::new(arr.clone(), arr)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 2, 2]);
    // Top-left 2x2: avg(1,2,5,6) = 3.5
    assert!(
        (output.lower()[[0, 0, 0]] - 3.5).abs() < 1e-5,
        "avgpool top-left should be 3.5, got {}",
        output.lower()[[0, 0, 0]]
    );
    // Top-right 2x2: avg(3,4,7,8) = 5.5
    assert!(
        (output.lower()[[0, 0, 1]] - 5.5).abs() < 1e-5,
        "avgpool top-right should be 5.5, got {}",
        output.lower()[[0, 0, 1]]
    );
    // Bottom-left 2x2: avg(9,10,13,14) = 11.5
    assert!(
        (output.lower()[[0, 1, 0]] - 11.5).abs() < 1e-5,
        "avgpool bottom-left should be 11.5, got {}",
        output.lower()[[0, 1, 0]]
    );
    // Bottom-right 2x2: avg(11,12,15,16) = 13.5
    assert!(
        (output.lower()[[0, 1, 1]] - 13.5).abs() < 1e-5,
        "avgpool bottom-right should be 13.5, got {}",
        output.lower()[[0, 1, 1]]
    );
    // Point interval: upper must equal lower
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (output.upper()[[0, i, j]] - output.lower()[[0, i, j]]).abs() < 1e-5,
                "point interval: upper[{},{}] {} != lower[{},{}] {}",
                i,
                j,
                output.upper()[[0, i, j]],
                i,
                j,
                output.lower()[[0, i, j]]
            );
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_ibp_global_3d() -> Result<()> {
    // Global average pool: 1 channel, 2x2 input → 1x1 output
    let layer = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);
    // lower = [1, 2; 3, 4] → avg = 2.5
    // upper = [5, 6; 7, 8] → avg = 6.5
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 1, 1]);
    assert!(
        (output.lower()[[0, 0, 0]] - 2.5).abs() < 1e-5,
        "global avg lower should be 2.5, got {}",
        output.lower()[[0, 0, 0]]
    );
    assert!(
        (output.upper()[[0, 0, 0]] - 6.5).abs() < 1e-5,
        "global avg upper should be 6.5, got {}",
        output.upper()[[0, 0, 0]]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_ibp_4d_matches_per_batch_3d() -> Result<()> {
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);

    let batch0: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let batch1: Vec<f32> = (101..=116).map(|x| x as f32).collect();
    let batched_data: Vec<f32> = batch0.iter().chain(batch1.iter()).copied().collect();
    let batched_arr = ArrayD::from_shape_vec(IxDyn(&[2, 1, 4, 4]), batched_data)
        .expect("invariant: valid shape for test data");
    let batched_input = BoundedTensor::new(batched_arr.clone(), batched_arr)?;
    let batched_output = layer.propagate_ibp(&batched_input)?;

    assert_eq!(batched_output.shape(), &[2, 1, 2, 2]);

    for (batch_idx, batch_data) in [&batch0, &batch1].iter().enumerate() {
        let single_arr = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), (*batch_data).clone())
            .expect("invariant: valid shape for test data");
        let single_input = BoundedTensor::new(single_arr.clone(), single_arr)?;
        let single_output = layer.propagate_ibp(&single_input)?;

        for oh in 0..2 {
            for ow in 0..2 {
                let batched_lower = batched_output.lower()[[batch_idx, 0, oh, ow]];
                let batched_upper = batched_output.upper()[[batch_idx, 0, oh, ow]];
                let single_lower = single_output.lower()[[0, oh, ow]];
                let single_upper = single_output.upper()[[0, oh, ow]];

                assert!(
                    (batched_lower - single_lower).abs() < 1e-5,
                    "batch {} lower[0,{},{}] mismatch: batched={}, single={}",
                    batch_idx,
                    oh,
                    ow,
                    batched_lower,
                    single_lower
                );
                assert!(
                    (batched_upper - single_upper).abs() < 1e-5,
                    "batch {} upper[0,{},{}] mismatch: batched={}, single={}",
                    batch_idx,
                    oh,
                    ow,
                    batched_upper,
                    single_upper
                );
            }
        }
    }

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_ibp_soundness() -> Result<()> {
    // Verify avg pool bounds are sound: for any x in [l, u], avg_pool(x) in [lower, upper]
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    // 1 channel, 4x4, varying bounds
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), (0..16).map(|i| i as f32).collect()).unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), (0..16).map(|i| i as f32 + 2.0).collect())
            .unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone())?;
    let output = layer.propagate_ibp(&input)?;

    // Structural: lower <= upper for all outputs
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "lower {} > upper {}", l, u);
    }

    // Soundness: evaluate at corners (all-lower and all-upper) and verify containment
    let lower_pt = BoundedTensor::new(lower.clone(), lower)?;
    let upper_pt = BoundedTensor::new(upper.clone(), upper)?;
    let eval_lower = layer.propagate_ibp(&lower_pt)?;
    let eval_upper = layer.propagate_ibp(&upper_pt)?;

    for (idx, ((ol, ou), (el, eu))) in output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .zip(eval_lower.lower().iter().zip(eval_upper.upper().iter()))
        .enumerate()
    {
        assert!(
            *ol <= *el + 1e-5,
            "avgpool soundness: output lower {} > eval-at-lower {} at idx {}",
            ol,
            el,
            idx
        );
        assert!(
            *ou + 1e-5 >= *eu,
            "avgpool soundness: output upper {} < eval-at-upper {} at idx {}",
            ou,
            eu,
            idx
        );
    }

    // Verify concrete values: top-left 2x2 kernel averages lower=[0,1,4,5]→avg=2.5
    assert!(
        (output.lower()[[0, 0, 0]] - 2.5).abs() < 1e-5,
        "avgpool lower[0,0,0] should be 2.5, got {}",
        output.lower()[[0, 0, 0]]
    );
    // upper=[2,3,6,7]→avg=4.5
    assert!(
        (output.upper()[[0, 0, 0]] - 4.5).abs() < 1e-5,
        "avgpool upper[0,0,0] should be 4.5, got {}",
        output.upper()[[0, 0, 0]]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_ibp_4d_soundness() -> Result<()> {
    // Verify batched avg pool bounds are sound: for any x in [l, u], avg_pool(x) in [lower, upper]
    // This is acceptance criteria 3 for #2497: lb <= true_output <= ub for batched inputs.
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    // 2 batches, 1 channel, 4x4
    let lower_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 4.0).collect();
    let upper_data: Vec<f32> = lower_data.iter().map(|x| x + 3.0).collect();
    let lower = ArrayD::from_shape_vec(IxDyn(&[2, 1, 4, 4]), lower_data)
        .expect("invariant: valid shape for test data");
    let upper = ArrayD::from_shape_vec(IxDyn(&[2, 1, 4, 4]), upper_data)
        .expect("invariant: valid shape for test data");
    let input = BoundedTensor::new(lower.clone(), upper.clone())?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[2, 1, 2, 2]);

    // Structural: lower <= upper for all outputs
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "lower {} > upper {}", l, u);
    }

    // Soundness: evaluate at corners (all-lower and all-upper) and verify containment
    let lower_pt = BoundedTensor::new(lower.clone(), lower)?;
    let upper_pt = BoundedTensor::new(upper.clone(), upper)?;
    let eval_lower = layer.propagate_ibp(&lower_pt)?;
    let eval_upper = layer.propagate_ibp(&upper_pt)?;

    for (idx, ((ol, ou), (el, eu))) in output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .zip(eval_lower.lower().iter().zip(eval_upper.upper().iter()))
        .enumerate()
    {
        assert!(
            *ol <= *el + 1e-5,
            "batched avgpool soundness: output lower {} > eval-at-lower {} at idx {}",
            ol,
            el,
            idx
        );
        assert!(
            *ou + 1e-5 >= *eu,
            "batched avgpool soundness: output upper {} < eval-at-upper {} at idx {}",
            ou,
            eu,
            idx
        );
    }

    // Non-zero check: verify outputs are NOT all-zero (the original bug).
    let any_nonzero = output
        .lower()
        .iter()
        .chain(output.upper().iter())
        .any(|v| v.abs() > 1e-10);
    assert!(any_nonzero, "batched avgpool output must not be all-zero");

    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_ibp_rejects_2d() {
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    let lower = ArrayD::zeros(IxDyn(&[4, 4])); // 2D, not valid
    let upper = ArrayD::zeros(IxDyn(&[4, 4]));
    let input = BoundedTensor::new(lower, upper).unwrap();
    let err = layer.propagate_ibp(&input).expect_err("2D not supported");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

// ===== MaxPool2dLayer tests =====

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_output_size() -> Result<()> {
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    assert_eq!(layer.output_size(4, 4)?, (2, 2));
    assert_eq!(layer.output_size(6, 6)?, (3, 3));
    Ok(())
}
#[ntest::timeout(10000)]
#[test]
fn test_maxpool_output_size_with_padding() -> Result<()> {
    let layer = MaxPool2dLayer::new((3, 3), (1, 1), (1, 1));
    assert_eq!(layer.output_size(3, 3)?, (3, 3));
    let err = MaxPool2dLayer::new((2, 2), (0, 0), (0, 0))
        .output_size(4, 4)
        .expect_err("zero stride");
    assert!(matches!(err, NyError::InvalidSpec(_)));
    let err = MaxPool2dLayer::new((5, 5), (1, 1), (0, 0))
        .output_size(4, 4)
        .expect_err("kernel larger than input should error");
    assert!(matches!(err, NyError::InvalidSpec(_)));
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_rejects_padding_ge_kernel() -> Result<()> {
    // padding >= kernel creates pooling windows made entirely of padding;
    // max over such a window is -inf (undefined), so the geometry must be
    // rejected rather than bounded.
    for layer in [
        MaxPool2dLayer::new((2, 2), (2, 2), (3, 3)),
        MaxPool2dLayer::new((2, 2), (2, 2), (2, 2)),
        MaxPool2dLayer::new((2, 3), (1, 1), (2, 1)), // height dimension only
        MaxPool2dLayer::new((3, 2), (1, 1), (1, 2)), // width dimension only
        MaxPool2dLayer::with_padding_mode((2, 2), (2, 2), (2, 2), false),
    ] {
        let err = layer
            .output_size(4, 4)
            .expect_err("padding >= kernel must be rejected");
        assert!(matches!(err, NyError::InvalidSpec(_)));
    }
    // Boundary: padding == kernel - 1 is still accepted.
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (1, 1));
    assert_eq!(layer.output_size(4, 4)?, (3, 3));
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_padding_ge_kernel_fails_closed() {
    // Every propagation path must refuse the all-padding-window geometry:
    // IBP would emit -inf outputs and the CROWN backward an unexamined
    // ~[0,0] row for them.
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (3, 3));
    let lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 5.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 6.0_f32);
    let input = BoundedTensor::new(lower, upper).unwrap();

    let err = layer
        .propagate_ibp(&input)
        .expect_err("IBP must reject all-padding windows");
    assert!(matches!(err, NyError::InvalidSpec(_)));

    // Would-be 5x5 output under the degenerate geometry.
    let bounds = LinearBounds::identity(25);
    let err = layer
        .propagate_linear_with_bounds(&bounds, &input)
        .expect_err("CROWN must reject all-padding windows");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_ibp_3d_point_interval() -> Result<()> {
    // Max pool of non-uniform values — point interval so output == exact max per window
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let arr = ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), data).unwrap();
    let input = BoundedTensor::new(arr.clone(), arr)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 2, 2]);
    assert!(
        (output.lower()[[0, 0, 0]] - 6.0).abs() < 1e-5,
        "maxpool top-left should be 6.0, got {}",
        output.lower()[[0, 0, 0]]
    );
    assert!(
        (output.lower()[[0, 0, 1]] - 8.0).abs() < 1e-5,
        "maxpool top-right should be 8.0, got {}",
        output.lower()[[0, 0, 1]]
    );
    assert!(
        (output.lower()[[0, 1, 0]] - 14.0).abs() < 1e-5,
        "maxpool bottom-left should be 14.0, got {}",
        output.lower()[[0, 1, 0]]
    );
    assert!(
        (output.lower()[[0, 1, 1]] - 16.0).abs() < 1e-5,
        "maxpool bottom-right should be 16.0, got {}",
        output.lower()[[0, 1, 1]]
    );
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (output.upper()[[0, i, j]] - output.lower()[[0, i, j]]).abs() < 1e-5,
                "point interval: upper[{},{}] {} != lower[{},{}] {}",
                i,
                j,
                output.upper()[[0, i, j]],
                i,
                j,
                output.lower()[[0, i, j]]
            );
        }
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_ibp_3d_picks_max() -> Result<()> {
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let input = BoundedTensor::new(lower, upper)?;
    let output = layer.propagate_ibp(&input)?;

    assert_eq!(output.shape(), &[1, 1, 1]);
    assert!(
        (output.lower()[[0, 0, 0]] - 4.0).abs() < 1e-5,
        "max lower should be 4.0, got {}",
        output.lower()[[0, 0, 0]]
    );
    assert!(
        (output.upper()[[0, 0, 0]] - 8.0).abs() < 1e-5,
        "max upper should be 8.0, got {}",
        output.upper()[[0, 0, 0]]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_ibp_soundness() -> Result<()> {
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), (0..16).map(|i| i as f32 - 8.0).collect())
            .unwrap();
    let upper =
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), (0..16).map(|i| i as f32 - 5.0).collect())
            .unwrap();
    let input = BoundedTensor::new(lower.clone(), upper.clone())?;
    let output = layer.propagate_ibp(&input)?;

    // Structural: lower <= upper for all outputs
    for (l, u) in output.lower().iter().zip(output.upper().iter()) {
        assert!(l <= u, "lower {} > upper {}", l, u);
    }

    // Soundness: evaluate at corners and verify containment
    let lower_pt = BoundedTensor::new(lower.clone(), lower)?;
    let upper_pt = BoundedTensor::new(upper.clone(), upper)?;
    let eval_lower = layer.propagate_ibp(&lower_pt)?;
    let eval_upper = layer.propagate_ibp(&upper_pt)?;

    for (idx, ((ol, ou), (el, eu))) in output
        .lower()
        .iter()
        .zip(output.upper().iter())
        .zip(eval_lower.lower().iter().zip(eval_upper.upper().iter()))
        .enumerate()
    {
        assert!(
            *ol <= *el + 1e-5,
            "maxpool soundness: output lower {} > eval-at-lower {} at idx {}",
            ol,
            el,
            idx
        );
        assert!(
            *ou + 1e-5 >= *eu,
            "maxpool soundness: output upper {} < eval-at-upper {} at idx {}",
            ou,
            eu,
            idx
        );
    }

    // Verify concrete values: top-left 2x2 kernel: lower=[-8,-7,-4,-3]→max=-3
    assert!(
        (output.lower()[[0, 0, 0]] - (-3.0)).abs() < 1e-5,
        "maxpool lower[0,0,0] should be -3.0, got {}",
        output.lower()[[0, 0, 0]]
    );
    // upper=[-5,-4,-1,0]→max=0
    assert!(
        (output.upper()[[0, 0, 0]] - 0.0).abs() < 1e-5,
        "maxpool upper[0,0,0] should be 0.0, got {}",
        output.upper()[[0, 0, 0]]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_ibp_rejects_2d() {
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let lower = ArrayD::zeros(IxDyn(&[4, 4]));
    let upper = ArrayD::zeros(IxDyn(&[4, 4]));
    let input = BoundedTensor::new(lower, upper).unwrap();
    let err = layer.propagate_ibp(&input).expect_err("2D not supported");
    assert!(matches!(err, NyError::InvalidSpec(_)));
}

// ===== CROWN backward error path tests =====

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_propagate_linear_requires_bounds() {
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    let bounds = LinearBounds::identity(4);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("requires bounds");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_propagate_linear_requires_bounds() {
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let bounds = LinearBounds::identity(4);
    let err = layer
        .propagate_linear(&bounds)
        .expect_err("requires bounds");
    assert!(matches!(err, NyError::UnsupportedOp(_)));
}

// ===== AveragePool CROWN backward with pre-activation bounds =====

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_crown_backward_uniform_weight() -> Result<()> {
    // 1 channel, 2x2 input → global avg pool → 1 output
    // The Jacobian is J = [1/4, 1/4, 1/4, 1/4]
    // With identity bounds (A=I), new_A = I @ J^T = J^T
    let layer = AveragePoolLayer::new((0, 0), (1, 1), (0, 0), false);
    let pre_act_lower = ArrayD::from_elem(IxDyn(&[1, 2, 2]), 0.0_f32);
    let pre_act_upper = ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32);
    let pre_act = BoundedTensor::new(pre_act_lower, pre_act_upper)?;

    // Output is 1x1x1 = 1 element, so identity bounds is 1x1
    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // new_A should have shape (1, 4) with all entries = 1/4
    assert_eq!(result.lower_a.nrows(), 1);
    assert_eq!(result.lower_a.ncols(), 4);
    for j in 0..4 {
        assert!(
            (result.lower_a[[0, j]] - 0.25).abs() < 1e-5,
            "expected 0.25, got {} at col {}",
            result.lower_a[[0, j]],
            j
        );
    }
    Ok(())
}

// ===== MaxPool CROWN backward with definite winner =====

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_crown_backward_definite_winner() -> Result<()> {
    // 1 channel, 2x2 input, kernel 2x2, stride 2 → 1 output
    // Pre-activation: element 3 (bottom-right) clearly dominates:
    //   lower = [0, 0, 0, 10], upper = [1, 1, 1, 20]
    //   Element 3's lower (10) > all others' uppers (1)
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.0, 0.0, 0.0, 10.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 1.0, 1.0, 20.0]).unwrap();
    let pre_act = BoundedTensor::new(lower, upper)?;

    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // Only element 3 should have nonzero coefficient (definite winner)
    assert!(
        result.lower_a[[0, 3]].abs() > 0.5,
        "definite winner (idx 3) should have coefficient ~1.0"
    );
    assert!(
        result.lower_a[[0, 0]].abs() < 1e-5,
        "non-winner (idx 0) should have zero coefficient"
    );
    assert!(
        result.lower_a[[0, 1]].abs() < 1e-5,
        "non-winner (idx 1) should have zero coefficient"
    );
    assert!(
        result.lower_a[[0, 2]].abs() < 1e-5,
        "non-winner (idx 2) should have zero coefficient"
    );
    Ok(())
}

// ===== AveragePool CROWN backward: non-global kernel =====

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_crown_backward_2x2_stride2() -> Result<()> {
    // 1 channel, 4x4 input, kernel 2x2, stride 2 → 2x2 output (4 elements)
    // Each output averages a 2x2 region: coefficient 1/4 per input in that window.
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    let pre_act_lower = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.0_f32);
    let pre_act_upper = ArrayD::from_elem(IxDyn(&[1, 4, 4]), 1.0_f32);
    let pre_act = BoundedTensor::new(pre_act_lower, pre_act_upper)?;

    // Output is 1*2*2 = 4 elements. Identity bounds: 4x4
    let bounds = LinearBounds::identity(4);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // new_A should have shape (4, 16)
    assert_eq!(result.lower_a.nrows(), 4);
    assert_eq!(result.lower_a.ncols(), 16);

    // Output element 0 = avg of input [0:2, 0:2] = indices 0,1,4,5 in flat order
    // (flat index for 1-channel 4x4: c*16 + h*4 + w)
    // Element (0,0,0) → h=0,w=0 → flat=0
    // Element (0,0,1) → h=0,w=1 → flat=1
    // Element (0,1,0) → h=1,w=0 → flat=4
    // Element (0,1,1) → h=1,w=1 → flat=5
    for &idx in &[0, 1, 4, 5] {
        assert!(
            (result.lower_a[[0, idx]] - 0.25).abs() < 1e-5,
            "output 0 coefficient at idx {} should be 0.25, got {}",
            idx,
            result.lower_a[[0, idx]]
        );
    }
    // Other indices should be zero for output 0
    for idx in 0..16 {
        if ![0, 1, 4, 5].contains(&idx) {
            assert!(
                result.lower_a[[0, idx]].abs() < 1e-5,
                "output 0 coefficient at idx {} should be 0, got {}",
                idx,
                result.lower_a[[0, idx]]
            );
        }
    }

    // Output element 1 = avg of input [0:2, 2:4] = indices 2,3,6,7
    for &idx in &[2, 3, 6, 7] {
        assert!(
            (result.lower_a[[1, idx]] - 0.25).abs() < 1e-5,
            "output 1 coefficient at idx {} should be 0.25, got {}",
            idx,
            result.lower_a[[1, idx]]
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_avgpool_crown_backward_soundness() -> Result<()> {
    // Verify CROWN bounds for avgpool contain all true outputs.
    // For average pooling (linear op), CROWN should give exact bounds.
    let layer = AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false);
    let lower =
        ArrayD::from_shape_vec(IxDyn(&[1, 4, 4]), (0..16).map(|i| i as f32 * 0.1).collect())
            .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[1, 4, 4]),
        (0..16).map(|i| i as f32 * 0.1 + 1.0).collect(),
    )
    .unwrap();
    let pre_act = BoundedTensor::new(lower.clone(), upper.clone())?;

    // IBP bounds for reference
    let ibp = layer.propagate_ibp(&pre_act)?;

    // CROWN backward: output has 4 elements
    let bounds = LinearBounds::identity(4);
    let crown = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // Concretize CROWN: for each output i,
    // lower_i = sum(max(A_i,j,0)*l_j + min(A_i,j,0)*u_j) + b_i
    let lower_flat: Vec<f32> = lower.iter().cloned().collect();
    let upper_flat: Vec<f32> = upper.iter().cloned().collect();
    for i in 0..4 {
        let mut crown_lo = crown.lower_b[i];
        let mut crown_hi = crown.upper_b[i];
        for j in 0..16 {
            let la = crown.lower_a[[i, j]];
            let ua = crown.upper_a[[i, j]];
            crown_lo += la.max(0.0) * lower_flat[j] + la.min(0.0) * upper_flat[j];
            crown_hi += ua.max(0.0) * upper_flat[j] + ua.min(0.0) * lower_flat[j];
        }
        // CROWN concretized should match IBP (avgpool is linear)
        assert!(
            (crown_lo - ibp.lower().iter().nth(i).unwrap()).abs() < 1e-3,
            "CROWN lower {} != IBP lower {} for output {}",
            crown_lo,
            ibp.lower().iter().nth(i).unwrap(),
            i
        );
        assert!(
            (crown_hi - ibp.upper().iter().nth(i).unwrap()).abs() < 1e-3,
            "CROWN upper {} != IBP upper {} for output {}",
            crown_hi,
            ibp.upper().iter().nth(i).unwrap(),
            i
        );
    }
    Ok(())
}

// ===== MaxPool CROWN backward: no definite winner (IBP fallback) =====

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_crown_backward_no_definite_winner() -> Result<()> {
    // All elements overlap: lower=[0,0,0,0], upper=[1,1,1,1]
    // No element's lower > all others' uppers → no definite winner.
    //
    // SOUND lower relaxation (dense path): the lower row routes linearly
    // through i* = argmax_i l_i, since y = max(x) >= x_{i*} pointwise.
    // The upper row (ua>0) stays constant at max_upper (x_{i*} is not an
    // upper bound on the max).
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let lower = ArrayD::from_elem(IxDyn(&[1, 2, 2]), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(&[1, 2, 2]), 1.0_f32);
    let pre_act = BoundedTensor::new(lower, upper)?;

    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // Lower row (la=1>0): routed through exactly one input i* with coeff 1.0,
    // the rest zero. lower_b absorbs no constant.
    let lower_a_nonzero: Vec<usize> = (0..4)
        .filter(|&j| result.lower_a[[0, j]].abs() > 1e-5)
        .collect();
    assert_eq!(
        lower_a_nonzero.len(),
        1,
        "lower row should route through exactly one i*, got nonzero at {:?}",
        lower_a_nonzero
    );
    let istar = lower_a_nonzero[0];
    assert!(
        (result.lower_a[[0, istar]] - 1.0).abs() < 1e-5,
        "lower_a[0,{}] should be 1.0 (la), got {}",
        istar,
        result.lower_a[[0, istar]]
    );
    assert!(
        result.lower_b[0].abs() < 1e-5,
        "lower_b should be 0 (no constant; routed linearly), got {}",
        result.lower_b[0]
    );

    // Upper row (ua=1>0): UNCHANGED, stays constant at max_upper=1, no gradient.
    for j in 0..4 {
        assert!(
            result.upper_a[[0, j]].abs() < 1e-5,
            "upper row unchanged: upper_a[0,{}] should be 0, got {}",
            j,
            result.upper_a[[0, j]]
        );
    }
    assert!(
        (result.upper_b[0] - 1.0).abs() < 1e-5,
        "upper_b should be max_upper=1, got {}",
        result.upper_b[0]
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_crown_backward_no_winner_asymmetric() -> Result<()> {
    // Overlapping but asymmetric: lower=[1,0,2,0], upper=[3,4,3,5]
    // max_lower = max(1,0,2,0) = 2 (at index 2), max_upper = max(3,4,3,5) = 5
    // No definite winner (no element's lower > all others' upper).
    //
    // SOUND lower relaxation (dense path): i* = argmax_i l_i = index 2.
    // The lower row (la=1>0) routes linearly through x_2 (y >= x_2), the
    // upper row (ua=1>0) stays constant at max_upper.
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 0.0, 2.0, 0.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![3.0, 4.0, 3.0, 5.0]).unwrap();
    let pre_act = BoundedTensor::new(lower, upper)?;

    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // Lower row routed through i* = index 2 (argmax l_i, l_2 = 2) with coeff 1.0.
    assert!(
        (result.lower_a[[0, 2]] - 1.0).abs() < 1e-5,
        "lower_a[0,2] should be 1.0 (routed through i*), got {}",
        result.lower_a[[0, 2]]
    );
    for j in [0, 1, 3] {
        assert!(
            result.lower_a[[0, j]].abs() < 1e-5,
            "lower_a[0,{}] should be 0, got {}",
            j,
            result.lower_a[[0, j]]
        );
    }
    // lower_b absorbs no constant now (routed linearly through x_2).
    assert!(
        result.lower_b[0].abs() < 1e-5,
        "lower_b should be 0 (routed linearly), got {}",
        result.lower_b[0]
    );

    // Upper row (ua=1>0): UNCHANGED, constant at max_upper = 5, no gradient.
    for j in 0..4 {
        assert!(
            result.upper_a[[0, j]].abs() < 1e-5,
            "upper row unchanged: upper_a[0,{}] should be 0, got {}",
            j,
            result.upper_a[[0, j]]
        );
    }
    assert!(
        (result.upper_b[0] - 5.0).abs() < 1e-5,
        "upper_b should be max_upper=5, got {}",
        result.upper_b[0]
    );

    // Soundness + tightness sanity check via concretization:
    // The new lower row, concretized over the box, must not exceed the true
    // max-pool lower bound (max_lower = 2) and must be >= the OLD constant (also 2
    // here, since i* concretizes to l_2 = 2 at the worst corner). Tightness over
    // the interior corners is exercised in the dedicated test below.
    let lower_flat = [1.0_f32, 0.0, 2.0, 0.0];
    let upper_flat = [3.0_f32, 4.0, 3.0, 5.0];
    let mut crown_lo = result.lower_b[0];
    for j in 0..4 {
        let la = result.lower_a[[0, j]];
        crown_lo += la.max(0.0) * lower_flat[j] + la.min(0.0) * upper_flat[j];
    }
    assert!(
        crown_lo <= 2.0 + 1e-4,
        "concretized lower {} must not exceed true max_lower 2",
        crown_lo
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_crown_backward_soundness() -> Result<()> {
    // Verify CROWN bounds contain IBP bounds (CROWN should be at least as tight).
    // For maxpool, CROWN with definite winner is tighter; without, it's equivalent.
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![2.0, 3.0, 4.0, 5.0]).unwrap();
    let pre_act = BoundedTensor::new(lower.clone(), upper.clone())?;

    // IBP
    let ibp = layer.propagate_ibp(&pre_act)?;

    // CROWN
    let bounds = LinearBounds::identity(1);
    let crown = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // Concretize CROWN
    let lower_flat: Vec<f32> = lower.iter().cloned().collect();
    let upper_flat: Vec<f32> = upper.iter().cloned().collect();
    let mut crown_lo = crown.lower_b[0];
    let mut crown_hi = crown.upper_b[0];
    for j in 0..4 {
        let la = crown.lower_a[[0, j]];
        let ua = crown.upper_a[[0, j]];
        crown_lo += la.max(0.0) * lower_flat[j] + la.min(0.0) * upper_flat[j];
        crown_hi += ua.max(0.0) * upper_flat[j] + ua.min(0.0) * lower_flat[j];
    }

    // CROWN bounds should be sound: contain true maxpool range
    // True lower bound of maxpool = max(lower) = max(0,1,2,3) = 3
    // True upper bound of maxpool = max(upper) = max(2,3,4,5) = 5
    let ibp_lo = *ibp.lower().iter().next().unwrap();
    let ibp_hi = *ibp.upper().iter().next().unwrap();
    assert!(
        crown_lo <= ibp_lo + 1e-4,
        "CROWN lower {} > IBP lower {}",
        crown_lo,
        ibp_lo
    );
    assert!(
        crown_hi >= ibp_hi - 1e-4,
        "CROWN upper {} < IBP upper {}",
        crown_hi,
        ibp_hi
    );
    Ok(())
}

// ===== MaxPool CROWN backward: multi-channel =====

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_crown_backward_multichannel() -> Result<()> {
    // 2 channels, 2x2 input each, kernel 2x2, stride 2 → 2x1x1 = 2 output elements
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    // Channel 0: definite winner at idx 3 (lower 10 > others' upper 1)
    // Channel 1: no definite winner (all [0,1])
    let lower = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let upper = ArrayD::from_shape_vec(
        IxDyn(&[2, 2, 2]),
        vec![1.0, 1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let pre_act = BoundedTensor::new(lower, upper)?;

    let bounds = LinearBounds::identity(2);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // Output 0 (channel 0): definite winner at flat idx 3
    assert!(
        (result.lower_a[[0, 3]] - 1.0).abs() < 1e-5,
        "channel 0 winner coeff should be 1"
    );
    assert!(result.lower_a[[0, 0]].abs() < 1e-5, "channel 0 non-winner");

    // Output 1 (channel 1): no definite winner. SOUND dense lower relaxation
    // routes the lower row through i* = argmax_i l_i (one of indices 4..8),
    // with coeff 1.0; the rest stay zero.
    let ch1_nonzero: Vec<usize> = (4..8)
        .filter(|&j| result.lower_a[[1, j]].abs() > 1e-5)
        .collect();
    assert_eq!(
        ch1_nonzero.len(),
        1,
        "channel 1 lower row should route through exactly one i*, got {:?}",
        ch1_nonzero
    );
    assert!(
        (result.lower_a[[1, ch1_nonzero[0]]] - 1.0).abs() < 1e-5,
        "channel 1 i* coeff should be 1.0, got {}",
        result.lower_a[[1, ch1_nonzero[0]]]
    );
    // Upper row for channel 1 (ua>0) stays constant → no gradient.
    for j in 4..8 {
        assert!(
            result.upper_a[[1, j]].abs() < 1e-5,
            "channel 1 upper_a at {} should be 0 (upper row unchanged)",
            j
        );
    }
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn test_maxpool_crown_backward_no_winner_lower_linear_tighter_and_sound() -> Result<()> {
    // Dedicated validation for the SOUND tighter no-winner LOWER relaxation
    // (dense path). lower=[1,0,2,0], upper=[3,4,3,5], no definite winner.
    // i* = argmax_i l_i = index 2 (l_2 = 2). y = max(x) >= x_2 pointwise.
    //
    // We assert:
    //  (1) The new lower row ENCLOSES max(x) at every corner of the box
    //      (soundness): concretized lower row value <= true max(x).
    //  (2) The new lower row is TIGHTER than the OLD constant max_lower at
    //      interior corners where x_{i*} > l_{i*}.
    //  (3) The upper-row arms (la<0 / ua>0) are UNCHANGED (constant max_upper).
    let layer = MaxPool2dLayer::new((2, 2), (2, 2), (0, 0));
    let lower_v = [1.0_f32, 0.0, 2.0, 0.0];
    let upper_v = [3.0_f32, 4.0, 3.0, 5.0];
    let lower = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), lower_v.to_vec()).unwrap();
    let upper = ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), upper_v.to_vec()).unwrap();
    let pre_act = BoundedTensor::new(lower, upper)?;

    let bounds = LinearBounds::identity(1);
    let result = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;

    // i* = index 2 with coeff 1.0; lower_b absorbs no constant.
    assert!(
        (result.lower_a[[0, 2]] - 1.0).abs() < 1e-5,
        "lower row should route through i*=2 with coeff 1.0, got {}",
        result.lower_a[[0, 2]]
    );

    let old_constant_max_lower = 2.0_f32; // max(l_i) = l_2

    // Evaluate the new lower row as an affine function of x: f(x) = lower_b + sum la_j x_j.
    let eval_lower = |x: &[f32; 4]| -> f32 {
        let mut v = result.lower_b[0];
        for j in 0..4 {
            v += result.lower_a[[0, j]] * x[j];
        }
        v
    };

    // (1) Soundness + (2) tightness over all 2^4 box corners.
    let mut saw_strictly_tighter = false;
    for mask in 0..16u32 {
        let mut x = [0.0f32; 4];
        for j in 0..4 {
            x[j] = if mask & (1 << j) != 0 {
                upper_v[j]
            } else {
                lower_v[j]
            };
        }
        let true_max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lo = eval_lower(&x);
        // Soundness: linear lower bound must not exceed the true max at any corner.
        assert!(
            lo <= true_max + 1e-4,
            "UNSOUND: new lower {} > true max {} at corner {:?}",
            lo,
            true_max,
            x
        );
        // Tightness: new lower bound is never below the old constant max_lower,
        // and strictly above it at corners where x_2 > l_2.
        assert!(
            lo >= old_constant_max_lower - 1e-4,
            "regression: new lower {} < old constant {} at corner {:?}",
            lo,
            old_constant_max_lower,
            x
        );
        if lo > old_constant_max_lower + 1e-4 {
            saw_strictly_tighter = true;
        }
    }
    assert!(
        saw_strictly_tighter,
        "new lower row should be strictly tighter than the old constant at some corner"
    );

    // (3) Upper-row arm (ua>0) UNCHANGED: constant max_upper=5, no gradient.
    for j in 0..4 {
        assert!(
            result.upper_a[[0, j]].abs() < 1e-5,
            "upper row (ua>0) must stay constant: upper_a[0,{}]={}",
            j,
            result.upper_a[[0, j]]
        );
    }
    assert!(
        (result.upper_b[0] - 5.0).abs() < 1e-5,
        "upper_b should be unchanged max_upper=5, got {}",
        result.upper_b[0]
    );

    // (3b) la<0 / ua>0 arms unchanged: feed a spec with NEGATIVE lower coeff and
    // POSITIVE upper coeff; both must stay constant (no gradient routed).
    let mut neg_lower = LinearBounds::identity(1);
    neg_lower.lower_a[[0, 0]] = -1.0; // la < 0 arm
    neg_lower.upper_a[[0, 0]] = 1.0; // ua > 0 arm
    let res2 = layer.propagate_linear_with_bounds(&neg_lower, &pre_act)?;
    for j in 0..4 {
        assert!(
            res2.lower_a[[0, j]].abs() < 1e-5,
            "la<0 arm must stay constant: lower_a[0,{}]={}",
            j,
            res2.lower_a[[0, j]]
        );
        assert!(
            res2.upper_a[[0, j]].abs() < 1e-5,
            "ua>0 arm must stay constant: upper_a[0,{}]={}",
            j,
            res2.upper_a[[0, j]]
        );
    }
    // la=-1 < 0  → lower_b += la * max_upper = -1 * 5 = -5
    assert!(
        (res2.lower_b[0] - (-5.0)).abs() < 1e-3,
        "la<0 arm: lower_b should be la*max_upper=-5, got {}",
        res2.lower_b[0]
    );
    // ua=+1 > 0  → upper_b += ua * max_upper = 5
    assert!(
        (res2.upper_b[0] - 5.0).abs() < 1e-3,
        "ua>0 arm: upper_b should be ua*max_upper=5, got {}",
        res2.upper_b[0]
    );
    Ok(())
}

// ===== AvgPool carried-coeff-err: enclosure (soundness) + discharge-vs-propagate A/B =====

/// Deterministic LCG in [0,1).
fn lcg01(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f64 / (1u64 << 31) as f64) as f32
}

/// AvgPool backward with a KNOWN incoming certified coeff-err must ENCLOSE every
/// sampled true value `Ã·avgpool(x)` for `Ã ∈ [A−E, A+E]`, `x ∈ box` — the soundness
/// gate for the EXACT per-column err composition in `average.rs` (#cgan-conv-err-compose
/// generalization).
///
/// It ALSO measures the discharge-vs-propagate concretized width (the A/B) and LOCKS
/// IN the negative finding that motivates keeping AvgPool a discharge op: `disc` folds
/// E into the bias over AvgPool's OWN output box (the dispatcher's non-carrier policy),
/// `prop` carries E per-column to the input box. Because AvgPool backward is a SCATTER
/// (many output errors sum into one input column) `max|outbox_c| ≤ Σ_{j∈win}w·max|inbox_j|`,
/// so propagate is provably ≥ discharge width — discharge wins. The assertion below
/// guards against a future change naively flipping AvgPool to `propagates_coeff_err`
/// on a false "tighter" claim.
#[ntest::timeout(30000)]
#[test]
fn test_avgpool_carried_coeff_err_encloses_and_ab_width() -> Result<()> {
    // Two configs: non-overlapping 2x2/s2 and overlapping 3x3/s1/pad1.
    let configs: [(AveragePoolLayer, &str); 2] = [
        (
            AveragePoolLayer::new((2, 2), (2, 2), (0, 0), false),
            "2x2s2",
        ),
        (
            AveragePoolLayer::new((3, 3), (1, 1), (1, 1), false),
            "3x3s1p1",
        ),
    ];
    // Input box: mixed-sign (strong averaging cancellation) so max|outbox| ≪
    // Σweight·max|inbox| — the regime that most favors discharge.
    let (ch, ih, iw) = (2usize, 4usize, 4usize);
    let mut st = 0xC0FFEEu64;
    let mut lo = ArrayD::<f32>::zeros(IxDyn(&[ch, ih, iw]));
    let mut up = ArrayD::<f32>::zeros(IxDyn(&[ch, ih, iw]));
    for (l, u) in lo.iter_mut().zip(up.iter_mut()) {
        let c = (lcg01(&mut st) - 0.5) * 4.0; // center in [-2, 2]
        let r = lcg01(&mut st) * 0.6 + 0.1; // radius in [0.1, 0.7]
        *l = c - r;
        *u = c + r;
    }
    let pre_act = BoundedTensor::new(lo.clone(), up.clone())?;

    for (layer, tag) in &configs {
        let out_box = layer.propagate_ibp(&pre_act)?;
        let n_out = out_box.len();
        // Downstream exact-affine map A (3 spec rows × n_out) with certified err E.
        let n_rows = 3usize;
        let mut a = ndarray::Array2::<f32>::zeros((n_rows, n_out));
        let mut e = ndarray::Array2::<f32>::zeros((n_rows, n_out));
        for r in 0..n_rows {
            for c in 0..n_out {
                a[[r, c]] = (lcg01(&mut st) - 0.5) * 3.0;
                e[[r, c]] = lcg01(&mut st) * 0.15; // err up to 0.15
            }
        }
        let b = ndarray::Array1::<f32>::zeros(n_rows);
        let bounds = LinearBounds::new_or_conservative_with_err(
            a.clone(),
            b.clone(),
            a.clone(),
            b.clone(),
            e.clone(),
            e.clone(),
        )?;

        // PROPAGATE (the changed path): backward on err-carrying bounds.
        let prop = layer.propagate_linear_with_bounds(&bounds, &pre_act)?;
        let prop_conc = prop.concretize_sound(&pre_act);

        // DISCHARGE (dispatcher non-carrier baseline): fold E into bias over the
        // AvgPool OUTPUT box, then backward on err-free bounds.
        let mut disc_in = bounds.clone();
        let obf = out_box.flatten();
        let obl: Vec<f32> = obf.lower().iter().copied().collect();
        let obu: Vec<f32> = obf.upper().iter().copied().collect();
        disc_in.fold_coeff_err_into_bias(&obl, &obu);
        let disc = layer.propagate_linear_with_bounds(&disc_in, &pre_act)?;
        let disc_conc = disc.concretize_sound(&pre_act);

        // --- ENCLOSURE (soundness): 4000 samples of (x, Ã). ---
        let inf = pre_act.flatten();
        let inl: Vec<f32> = inf.lower().iter().copied().collect();
        let inu: Vec<f32> = inf.upper().iter().copied().collect();
        let mut violations = 0usize;
        for _ in 0..4000 {
            // Sample x in the box.
            let mut xflat = vec![0.0f32; inl.len()];
            for k in 0..xflat.len() {
                xflat[k] = inl[k] + lcg01(&mut st) * (inu[k] - inl[k]);
            }
            let xarr = ArrayD::from_shape_vec(IxDyn(&[ch, ih, iw]), xflat.clone()).unwrap();
            let point = BoundedTensor::new(xarr.clone(), xarr)?;
            let y = layer.propagate_ibp(&point)?; // exact avgpool(x)
            let yflat: Vec<f64> = y.flatten().lower().iter().map(|v| *v as f64).collect();
            // Sample Ã ∈ [A−E, A+E].
            for r in 0..n_rows {
                let mut tv = 0.0f64;
                for c in 0..n_out {
                    let t = (lcg01(&mut st) * 2.0 - 1.0) as f64; // [-1,1]
                    let atil = a[[r, c]] as f64 + t * e[[r, c]] as f64;
                    tv += atil * yflat[c];
                }
                let plo = prop_conc.lower()[r] as f64;
                let pup = prop_conc.upper()[r] as f64;
                let tol = 1e-4 * (1.0 + tv.abs());
                if tv < plo - tol || tv > pup + tol {
                    violations += 1;
                }
            }
        }
        assert_eq!(
            violations, 0,
            "{tag}: PROPAGATE path under-encloses the true value ({violations} violations) — UNSOUND"
        );

        // --- A/B WIDTH (discharge vs propagate), reported per row. ---
        let mut prop_w = 0.0f64;
        let mut disc_w = 0.0f64;
        for r in 0..n_rows {
            prop_w += (prop_conc.upper()[r] - prop_conc.lower()[r]) as f64;
            disc_w += (disc_conc.upper()[r] - disc_conc.lower()[r]) as f64;
        }
        println!(
            "AVGPOOL-ERR-AB[{tag}] sum-width: discharge={disc_w:.6}  propagate={prop_w:.6}  ratio(prop/disc)={:.4}",
            prop_w / disc_w
        );
        // Guard the negative finding: propagate must NOT be meaningfully tighter than
        // discharge for AvgPool (the triangle-inequality argument makes it provably
        // ≥). If this ever fails, re-examine before flipping AvgPool to a propagator.
        assert!(
            prop_w >= disc_w * 0.999,
            "{tag}: propagate width ({prop_w:.6}) is unexpectedly tighter than discharge \
             ({disc_w:.6}) — AvgPool's discharge-vs-propagate calculus changed; re-examine \
             query.rs::propagates_coeff_err before trusting a 'propagate is tighter' claim"
        );
    }
    Ok(())
}
