// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Conv3d bound-propagation tests.
//!
//! The enclosure tests compute their reference convolution in `f64` from the
//! same weights and a CONCRETE input sampled across the box, then assert
//! `lower <= reference <= upper`. That is the property that matters: a bound
//! that fails it is a false proof waiting to happen.

use ndarray::{array, Array1, ArrayD, IxDyn};
use ny_core::{NyError, Result};
use ny_tensor::BoundedTensor;
use std::time::{Duration, Instant};

use super::*;
use crate::layers::common::BoundPropagation;
use crate::layers::Conv2dLayer;
use crate::LinearBounds;

/// Deterministic pseudo-random values in `[-1, 1)`, so a failure reproduces.
fn pseudo_random(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 33) as f64) / ((1_u64 << 31) as f64);
            (unit - 1.0) as f32
        })
        .collect()
}

struct Conv3dCase {
    layer: Conv3dLayer,
    input: BoundedTensor,
    in_shape: [usize; 3],
}

/// Build a Conv3d layer plus an input box, with all shapes derived from the
/// requested geometry so the case is always well formed.
#[allow(clippy::too_many_arguments)]
fn case(
    seed: u64,
    in_c: usize,
    out_c: usize,
    in_shape: [usize; 3],
    kernel_shape: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    with_bias: bool,
) -> Conv3dCase {
    let in_c_per_group = in_c / groups;
    let kernel_len = out_c * in_c_per_group * kernel_shape[0] * kernel_shape[1] * kernel_shape[2];
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[
            out_c,
            in_c_per_group,
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
        ]),
        pseudo_random(seed, kernel_len),
    )
    .expect("kernel");
    let bias = with_bias.then(|| Array1::from(pseudo_random(seed ^ 0x5eed, out_c)));
    let layer =
        Conv3dLayer::new_dilated(kernel, bias, stride, padding, dilation, groups).expect("layer");

    let input_len = in_c * in_shape[0] * in_shape[1] * in_shape[2];
    let centers = pseudo_random(seed ^ 0xc0ffee, input_len);
    let radii = pseudo_random(seed ^ 0xbeef, input_len);
    let lower: Vec<f32> = centers
        .iter()
        .zip(radii.iter())
        .map(|(c, r)| c - 0.25 * r.abs())
        .collect();
    let upper: Vec<f32> = centers
        .iter()
        .zip(radii.iter())
        .map(|(c, r)| c + 0.25 * r.abs())
        .collect();
    let shape = [in_c, in_shape[0], in_shape[1], in_shape[2]];
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&shape), lower).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&shape), upper).expect("upper"),
    )
    .expect("input box");
    Conv3dCase {
        layer,
        input,
        in_shape,
    }
}

/// Direct 3-D convolution of a CONCRETE input, accumulated in `f64`.
///
/// Deliberately independent of the layer implementation: plain nested loops
/// over the ONNX/PyTorch definition, no interval arithmetic, no shared helper.
fn reference_conv3d_f64(layer: &Conv3dLayer, input: &ArrayD<f32>) -> ArrayD<f64> {
    let shape = input.shape().to_vec();
    let (in_c, in_d, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
    let kernel_shape = layer.kernel.shape().to_vec();
    let out_c = kernel_shape[0];
    let in_c_per_group = kernel_shape[1];
    let (kd, kh, kw) = (kernel_shape[2], kernel_shape[3], kernel_shape[4]);
    let out = layer
        .output_size([in_d, in_h, in_w])
        .expect("reference output size");
    let out_c_per_group = out_c / layer.groups;

    let mut result = ArrayD::<f64>::zeros(IxDyn(&[out_c, out[0], out[1], out[2]]));
    for oc in 0..out_c {
        let group = oc / out_c_per_group;
        for od in 0..out[0] {
            for oh in 0..out[1] {
                for ow in 0..out[2] {
                    let mut acc = layer
                        .bias
                        .as_ref()
                        .map_or(0.0_f64, |bias| f64::from(bias[oc]));
                    for icl in 0..in_c_per_group {
                        let ic = group * in_c_per_group + icl;
                        for ki in 0..kd {
                            for kj in 0..kh {
                                for kk in 0..kw {
                                    let id = od * layer.stride[0] + ki * layer.dilation[0];
                                    let ih = oh * layer.stride[1] + kj * layer.dilation[1];
                                    let iw = ow * layer.stride[2] + kk * layer.dilation[2];
                                    if id < layer.padding[0] || ih < layer.padding[1] {
                                        continue;
                                    }
                                    if iw < layer.padding[2] {
                                        continue;
                                    }
                                    let (id, ih, iw) = (
                                        id - layer.padding[0],
                                        ih - layer.padding[1],
                                        iw - layer.padding[2],
                                    );
                                    if id >= in_d || ih >= in_h || iw >= in_w {
                                        continue;
                                    }
                                    debug_assert!(ic < in_c);
                                    acc += f64::from(layer.kernel[[oc, icl, ki, kj, kk]])
                                        * f64::from(input[[ic, id, ih, iw]]);
                                }
                            }
                        }
                    }
                    result[[oc, od, oh, ow]] = acc;
                }
            }
        }
    }
    result
}

/// Concrete inputs sampled across the box: lower corner, upper corner,
/// midpoint, and two deterministic interior mixes.
fn box_samples(input: &BoundedTensor) -> Vec<ArrayD<f32>> {
    let lower = input.lower();
    let upper = input.upper();
    let len = lower.len();
    let mixes = pseudo_random(0x5a17, len * 2);
    let mut samples = vec![lower.clone(), upper.clone()];
    for corner in 0..3 {
        let mut sample = lower.clone();
        for (index, value) in sample.iter_mut().enumerate() {
            let l = f64::from(lower.iter().nth(index).copied().unwrap_or(0.0));
            let u = f64::from(upper.iter().nth(index).copied().unwrap_or(0.0));
            let t = match corner {
                0 => 0.5,
                1 => 0.5 * (f64::from(mixes[index]) + 1.0),
                _ => 0.5 * (f64::from(mixes[len + index]) + 1.0),
            };
            *value = (l + t * (u - l)) as f32;
        }
        samples.push(sample);
    }
    samples
}

/// Assert the IBP box encloses the f64 reference at every sampled point.
fn assert_encloses_samples(case: &Conv3dCase, label: &str) -> Result<()> {
    let bounds = case.layer.propagate_ibp(&case.input)?;
    let out = case.layer.output_size(case.in_shape)?;
    let out_c = case.layer.out_channels();
    assert_eq!(bounds.shape(), &[out_c, out[0], out[1], out[2]], "{label}");

    for (sample_index, sample) in box_samples(&case.input).into_iter().enumerate() {
        let reference = reference_conv3d_f64(&case.layer, &sample);
        for (index, &truth) in reference.iter().enumerate() {
            let lower = f64::from(bounds.lower().iter().nth(index).copied().expect("lower"));
            let upper = f64::from(bounds.upper().iter().nth(index).copied().expect("upper"));
            assert!(
                lower <= truth && truth <= upper,
                "{label}: sample {sample_index} element {index} escaped the IBP box: \
                 {truth} not in [{lower}, {upper}]"
            );
        }
    }
    Ok(())
}

#[ntest::timeout(30000)]
#[test]
fn conv3d_ibp_encloses_direct_evaluation_across_the_box() -> Result<()> {
    assert_encloses_samples(
        &case(
            1,
            2,
            3,
            [4, 5, 6],
            [2, 3, 2],
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
            1,
            true,
        ),
        "dense",
    )
}

#[ntest::timeout(30000)]
#[test]
fn conv3d_ibp_encloses_under_stride_padding_dilation_and_groups() -> Result<()> {
    // Every supported knob, alone and combined. Non-cubic on purpose: a
    // depth/height/width mix-up survives cubic geometry.
    let cases: Vec<(&str, Conv3dCase)> = vec![
        (
            "stride",
            case(
                2,
                2,
                2,
                [6, 7, 5],
                [2, 2, 2],
                [2, 3, 1],
                [0, 0, 0],
                [1, 1, 1],
                1,
                true,
            ),
        ),
        (
            "padding",
            case(
                3,
                1,
                2,
                [3, 4, 5],
                [3, 3, 3],
                [1, 1, 1],
                [1, 2, 0],
                [1, 1, 1],
                1,
                true,
            ),
        ),
        (
            "dilation",
            case(
                4,
                2,
                2,
                [7, 6, 8],
                [2, 2, 3],
                [1, 1, 1],
                [0, 0, 0],
                [3, 2, 2],
                1,
                false,
            ),
        ),
        (
            "groups",
            case(
                5,
                4,
                6,
                [4, 4, 4],
                [2, 2, 2],
                [1, 1, 1],
                [0, 0, 0],
                [1, 1, 1],
                2,
                true,
            ),
        ),
        (
            "depthwise",
            case(
                6,
                3,
                3,
                [5, 5, 5],
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                3,
                true,
            ),
        ),
        (
            "all-knobs",
            case(
                7,
                4,
                4,
                [8, 7, 9],
                [2, 3, 2],
                [2, 1, 3],
                [1, 2, 1],
                [2, 1, 2],
                2,
                true,
            ),
        ),
    ];
    for (label, conv_case) in cases {
        assert_encloses_samples(&conv_case, label)?;
    }
    Ok(())
}

#[ntest::timeout(30000)]
#[test]
fn conv3d_ibp_encloses_batched_input() -> Result<()> {
    let conv_case = case(
        8,
        2,
        3,
        [3, 4, 4],
        [2, 2, 2],
        [1, 2, 1],
        [1, 0, 1],
        [1, 1, 1],
        1,
        true,
    );
    // Stack the same volume twice; a (N, C, D, H, W) input must reproduce the
    // unbatched result in every batch slot.
    let unbatched = conv_case.layer.propagate_ibp(&conv_case.input)?;
    let shape = conv_case.input.lower().shape().to_vec();
    let batched_shape = [2, shape[0], shape[1], shape[2], shape[3]];
    let stack = |values: &ArrayD<f32>| {
        let mut doubled: Vec<f32> = values.iter().copied().collect();
        doubled.extend(values.iter().copied());
        ArrayD::from_shape_vec(IxDyn(&batched_shape), doubled).expect("batched")
    };
    let batched_input = BoundedTensor::new(
        stack(conv_case.input.lower()),
        stack(conv_case.input.upper()),
    )?;
    let batched = conv_case.layer.propagate_ibp(&batched_input)?;

    let per_item = unbatched.lower().len();
    assert_eq!(batched.lower().len(), 2 * per_item);
    for item in 0..2 {
        for index in 0..per_item {
            assert_eq!(
                batched.lower().iter().nth(item * per_item + index),
                unbatched.lower().iter().nth(index),
                "batched lower differs from unbatched at item {item} index {index}"
            );
            assert_eq!(
                batched.upper().iter().nth(item * per_item + index),
                unbatched.upper().iter().nth(index),
                "batched upper differs from unbatched at item {item} index {index}"
            );
        }
    }
    Ok(())
}

/// Depth-1 parity: a Conv3d whose depth extent, kernel depth, stride, padding
/// and dilation are all trivial IS a Conv2d. Its box must enclose the same
/// truth, sit INSIDE Conv2d's certified sound box, and agree with Conv2d's
/// (uncertified, round-to-nearest) plain IBP to a few ULP.
#[ntest::timeout(30000)]
#[test]
fn conv3d_matches_a_conv2d_embedded_at_depth_one() -> Result<()> {
    for (groups, stride2, padding2, dilation2) in [
        (1usize, (1usize, 1usize), (0usize, 0usize), (1usize, 1usize)),
        (1, (2, 1), (1, 2), (1, 1)),
        (2, (1, 2), (0, 1), (2, 1)),
    ] {
        let (in_c, out_c) = (4usize, 4usize);
        let in_c_per_group = in_c / groups;
        let (kh, kw) = (3usize, 2usize);
        let kernel_values = pseudo_random(11, out_c * in_c_per_group * kh * kw);
        let bias = Array1::from(pseudo_random(12, out_c));

        let kernel_2d = ArrayD::from_shape_vec(
            IxDyn(&[out_c, in_c_per_group, kh, kw]),
            kernel_values.clone(),
        )
        .expect("2d kernel");
        let kernel_3d =
            ArrayD::from_shape_vec(IxDyn(&[out_c, in_c_per_group, 1, kh, kw]), kernel_values)
                .expect("3d kernel");

        let conv2d = Conv2dLayer::new_dilated(
            kernel_2d,
            Some(bias.clone()),
            stride2,
            padding2,
            dilation2,
            groups,
        )?;
        let conv3d = Conv3dLayer::new_dilated(
            kernel_3d,
            Some(bias),
            [1, stride2.0, stride2.1],
            [0, padding2.0, padding2.1],
            [1, dilation2.0, dilation2.1],
            groups,
        )?;

        let (in_h, in_w) = (6usize, 7usize);
        let len = in_c * in_h * in_w;
        let centers = pseudo_random(13, len);
        let radii = pseudo_random(14, len);
        let lower: Vec<f32> = centers
            .iter()
            .zip(radii.iter())
            .map(|(c, r)| c - 0.2 * r.abs())
            .collect();
        let upper: Vec<f32> = centers
            .iter()
            .zip(radii.iter())
            .map(|(c, r)| c + 0.2 * r.abs())
            .collect();
        let input_2d = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), lower.clone()).expect("2d lower"),
            ArrayD::from_shape_vec(IxDyn(&[in_c, in_h, in_w]), upper.clone()).expect("2d upper"),
        )?;
        let input_3d = BoundedTensor::new(
            ArrayD::from_shape_vec(IxDyn(&[in_c, 1, in_h, in_w]), lower).expect("3d lower"),
            ArrayD::from_shape_vec(IxDyn(&[in_c, 1, in_h, in_w]), upper).expect("3d upper"),
        )?;

        let out_2d = conv2d.propagate_ibp(&input_2d)?;
        let sound_2d = conv2d.propagate_ibp_sound_with_engine(&input_2d, None)?;
        let out_3d = conv3d.propagate_ibp(&input_3d)?;

        let label =
            format!("groups={groups} stride={stride2:?} pad={padding2:?} dil={dilation2:?}");
        assert_eq!(
            out_3d.shape()[0],
            out_2d.shape()[0],
            "{label}: channel count"
        );
        assert_eq!(out_3d.shape()[1], 1, "{label}: depth-1 output");
        assert_eq!(
            &out_3d.shape()[2..],
            &out_2d.shape()[1..],
            "{label}: spatial"
        );

        let count = out_2d.lower().len();
        for index in 0..count {
            let l3 = out_3d.lower().iter().nth(index).copied().expect("l3");
            let u3 = out_3d.upper().iter().nth(index).copied().expect("u3");
            let l2 = out_2d.lower().iter().nth(index).copied().expect("l2");
            let u2 = out_2d.upper().iter().nth(index).copied().expect("u2");
            let ls = sound_2d.lower().iter().nth(index).copied().expect("ls");
            let us = sound_2d.upper().iter().nth(index).copied().expect("us");

            // No looser than Conv2d's own certified sound forward.
            assert!(
                ls <= l3 && u3 <= us,
                "{label}: element {index}: Conv3d [{l3}, {u3}] is not inside Conv2d's certified \
                 sound box [{ls}, {us}]"
            );
            // ...and within a few ULP of Conv2d's plain (uncertified) IBP.
            let scale = l2.abs().max(u2.abs()).max(1.0);
            assert!(
                (l3 - l2).abs() <= 8.0 * f32::EPSILON * scale
                    && (u3 - u2).abs() <= 8.0 * f32::EPSILON * scale,
                "{label}: element {index}: Conv3d [{l3}, {u3}] drifted from Conv2d [{l2}, {u2}]"
            );
        }
    }
    Ok(())
}

/// The cancellation window from `conv_family_sound_ibp_dispatch.rs`, in 3-D:
/// a round-to-nearest f32 accumulation returns exactly `0` while the true
/// value is `-1`. The certified forward must still enclose `-1`.
#[ntest::timeout(10000)]
#[test]
fn conv3d_ibp_encloses_a_catastrophic_cancellation_window() -> Result<()> {
    const TWO24: f32 = 16_777_216.0;
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 3, 1, 1, 1]), vec![TWO24, -1.0, -TWO24]).expect("kernel");
    let layer = Conv3dLayer::new(kernel, Some(array![0.0_f32]), [1, 1, 1], [0, 0, 0])?;
    let point = ArrayD::from_shape_vec(IxDyn(&[3, 1, 1, 1]), vec![1.0_f32; 3]).expect("point");
    let input = BoundedTensor::new(point.clone(), point)?;

    let bounds = layer.propagate_ibp(&input)?;
    let lower = f64::from(bounds.lower()[[0, 0, 0, 0]]);
    let upper = f64::from(bounds.upper()[[0, 0, 0, 0]]);
    assert!(
        lower <= -1.0 && -1.0 <= upper,
        "certified Conv3d box [{lower}, {upper}] excludes the true value -1"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn conv3d_ibp_degrades_non_finite_weights_to_a_vacuous_cell() -> Result<()> {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 1, 2]), vec![f32::NAN, 1.0]).expect("kernel");
    let layer = Conv3dLayer::new(kernel, None, [1, 1, 1], [0, 0, 0])?;
    let point = ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2]), vec![1.0_f32, 2.0]).expect("point");
    let input = BoundedTensor::new(point.clone(), point)?;

    let bounds = layer.propagate_ibp(&input)?;
    assert_eq!(bounds.lower()[[0, 0, 0, 0]], f32::NEG_INFINITY);
    assert_eq!(bounds.upper()[[0, 0, 0, 0]], f32::INFINITY);
    Ok(())
}

// ---------------------------------------------------------------------------
// Refusal pins: every unsupported parameter combination must fail BY NAME.
// ---------------------------------------------------------------------------

fn unit_kernel(shape: &[usize]) -> ArrayD<f32> {
    ArrayD::from_elem(IxDyn(shape), 1.0_f32)
}

#[ntest::timeout(10000)]
#[test]
fn conv3d_refuses_non_rank5_kernels() {
    for shape in [vec![1, 1, 1], vec![1, 1, 1, 1], vec![1, 1, 1, 1, 1, 1]] {
        let result = Conv3dLayer::new(unit_kernel(&shape), None, [1, 1, 1], [0, 0, 0]);
        assert!(
            matches!(result, Err(NyError::ShapeMismatch { .. })),
            "rank-{} kernel must be refused, got {result:?}",
            shape.len()
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn conv3d_refuses_zero_stride_dilation_and_groups_by_name() {
    let kernel = || unit_kernel(&[1, 1, 1, 1, 1]);
    for axis in 0..3 {
        let mut stride = [1, 1, 1];
        stride[axis] = 0;
        let error = Conv3dLayer::new(kernel(), None, stride, [0, 0, 0])
            .expect_err("zero stride must be refused");
        assert!(
            matches!(&error, NyError::InvalidSpec(message) if message.contains("stride")),
            "stride refusal must name stride: {error:?}"
        );

        let mut dilation = [1, 1, 1];
        dilation[axis] = 0;
        let error = Conv3dLayer::new_dilated(kernel(), None, [1, 1, 1], [0, 0, 0], dilation, 1)
            .expect_err("zero dilation must be refused");
        assert!(
            matches!(&error, NyError::InvalidSpec(message) if message.contains("dilation")),
            "dilation refusal must name dilation: {error:?}"
        );
    }

    let error = Conv3dLayer::new_full(kernel(), None, [1, 1, 1], [0, 0, 0], 0)
        .expect_err("zero groups must be refused");
    assert!(
        matches!(&error, NyError::InvalidSpec(message) if message.contains("groups")),
        "groups refusal must name groups: {error:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn conv3d_refuses_channels_not_divisible_by_groups_and_mismatched_bias() {
    let error = Conv3dLayer::new_full(unit_kernel(&[3, 1, 1, 1, 1]), None, [1, 1, 1], [0, 0, 0], 2)
        .expect_err("out_channels % groups != 0 must be refused");
    assert!(
        matches!(&error, NyError::InvalidSpec(message)
            if message.contains("out_channels") && message.contains("groups")),
        "refusal must name out_channels and groups: {error:?}"
    );

    let error = Conv3dLayer::new(
        unit_kernel(&[2, 1, 1, 1, 1]),
        Some(array![0.0_f32]),
        [1, 1, 1],
        [0, 0, 0],
    )
    .expect_err("bias length mismatch must be refused");
    assert!(
        matches!(&error, NyError::ShapeMismatch { expected, got } if expected == &[2] && got == &[1]),
        "bias refusal must report the expected length: {error:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn conv3d_refuses_zero_kernel_extents() {
    let error = Conv3dLayer::new(unit_kernel(&[1, 1, 0, 1, 1]), None, [1, 1, 1], [0, 0, 0])
        .expect_err("zero kernel extent must be refused");
    assert!(
        matches!(&error, NyError::InvalidSpec(message) if message.contains("kernel dimensions")),
        "refusal must name the kernel dimensions: {error:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn conv3d_refuses_an_input_smaller_than_the_effective_kernel() -> Result<()> {
    let layer = Conv3dLayer::new_dilated(
        unit_kernel(&[1, 1, 3, 1, 1]),
        None,
        [1, 1, 1],
        [0, 0, 0],
        [4, 1, 1],
        1,
    )?;
    let error = layer
        .output_size([5, 1, 1])
        .expect_err("effective kernel 9 > padded depth 5 must be refused");
    assert!(
        matches!(&error, NyError::InvalidSpec(message)
            if message.contains("depth") && message.contains("effective kernel")),
        "refusal must name the offending axis: {error:?}"
    );

    let point = ArrayD::from_elem(IxDyn(&[1, 5, 1, 1]), 0.0_f32);
    let input = BoundedTensor::new(point.clone(), point)?;
    assert!(
        layer.propagate_ibp(&input).is_err(),
        "the forward must refuse the same geometry the output-size query refuses"
    );
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn conv3d_refuses_wrong_input_rank_and_channel_count() -> Result<()> {
    let layer = Conv3dLayer::new(unit_kernel(&[1, 2, 1, 1, 1]), None, [1, 1, 1], [0, 0, 0])?;

    let rank3 = ArrayD::from_elem(IxDyn(&[2, 1, 1]), 0.0_f32);
    let input = BoundedTensor::new(rank3.clone(), rank3)?;
    assert!(
        matches!(
            layer.propagate_ibp(&input),
            Err(NyError::ShapeMismatch { .. })
        ),
        "a rank-3 input must be refused"
    );

    let wrong_channels = ArrayD::from_elem(IxDyn(&[3, 1, 1, 1]), 0.0_f32);
    let input = BoundedTensor::new(wrong_channels.clone(), wrong_channels)?;
    assert!(
        matches!(
            layer.propagate_ibp(&input),
            Err(NyError::ShapeMismatch { .. })
        ),
        "a channel-count mismatch must be refused"
    );
    Ok(())
}

/// Mutating a public geometry field after construction must be caught by the
/// entry-point revalidation, not indexed blindly.
#[ntest::timeout(10000)]
#[test]
fn conv3d_revalidates_publicly_mutated_geometry() -> Result<()> {
    let mut layer = Conv3dLayer::new(unit_kernel(&[1, 1, 1, 1, 1]), None, [1, 1, 1], [0, 0, 0])?;
    layer.stride = [1, 0, 1];
    let point = ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 1.0_f32);
    let input = BoundedTensor::new(point.clone(), point)?;
    assert!(
        matches!(layer.propagate_ibp(&input), Err(NyError::InvalidSpec(_))),
        "a zero stride written after construction must still be refused"
    );
    assert!(layer.try_out_channels().is_err());
    assert!(layer.try_in_channels().is_err());
    assert!(layer.try_kernel_size().is_err());
    Ok(())
}

// ---------------------------------------------------------------------------
// CROWN must fail closed, never vacuously succeed.
// ---------------------------------------------------------------------------

#[ntest::timeout(10000)]
#[test]
fn conv3d_crown_backward_fails_closed() -> Result<()> {
    let layer = Conv3dLayer::new(unit_kernel(&[1, 1, 1, 1, 1]), None, [1, 1, 1], [0, 0, 0])?;
    let bounds = LinearBounds::identity(1);

    let error = layer
        .propagate_linear(&bounds)
        .expect_err("Conv3d has no CROWN backward");
    assert!(
        matches!(&error, NyError::UnsupportedOp(message)
            if message.contains("Conv3d") && message.contains("CROWN")),
        "the refusal must name the layer and the path: {error:?}"
    );

    // The trait-level entry point must refuse too — no silent vacuous plane.
    let error = layer
        .propagate_crown_backward(&bounds, None)
        .expect_err("Conv3d has no CROWN backward");
    assert!(matches!(error, NyError::UnsupportedOp(_)), "{error:?}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Deadline behaviour.
// ---------------------------------------------------------------------------

#[ntest::timeout(30000)]
#[test]
fn conv3d_certified_forward_honours_a_finite_deadline() -> Result<()> {
    let conv_case = case(
        9,
        4,
        8,
        [10, 12, 12],
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        1,
        true,
    );
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap_or_else(Instant::now);
    let error = conv_case
        .layer
        .propagate_ibp_with_deadline(&conv_case.input, Some(expired))
        .expect_err("an expired deadline must be reported");
    assert!(
        matches!(&error, NyError::DeadlineExceeded(message) if message.contains("Conv3d")),
        "expiry must be typed and name the layer: {error:?}"
    );

    // A generous deadline must produce the same box as no deadline at all.
    let generous = Instant::now() + Duration::from_secs(30);
    let with_deadline = conv_case
        .layer
        .propagate_ibp_with_deadline(&conv_case.input, Some(generous))?;
    let without = conv_case.layer.propagate_ibp(&conv_case.input)?;
    assert_eq!(with_deadline.lower(), without.lower());
    assert_eq!(with_deadline.upper(), without.upper());
    Ok(())
}

#[ntest::timeout(10000)]
#[test]
fn conv3d_sound_entry_point_refuses_a_bounded_executor_without_a_deadline() -> Result<()> {
    use ny_core::{GemmEngine, NyError as CoreError};

    /// Minimal bounded-host stand-in: forbids unbounded CPU fallback and
    /// refuses every GEMM, so any use of it as an arithmetic engine would show.
    #[derive(Debug)]
    struct BoundedHost;
    impl GemmEngine for BoundedHost {
        fn gemm_f32(
            &self,
            _m: usize,
            _k: usize,
            _n: usize,
            _a: &[f32],
            _b: &[f32],
        ) -> Result<Vec<f32>> {
            Err(CoreError::UnsupportedOp("bounded host gemm".into()))
        }

        fn forbids_unbounded_cpu_fallback(&self) -> bool {
            true
        }
    }

    let conv_case = case(
        10,
        1,
        1,
        [2, 2, 2],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        1,
        false,
    );
    let engine = BoundedHost;
    let error = conv_case
        .layer
        .propagate_ibp_sound_with_engine(&conv_case.input, Some(&engine))
        .expect_err("a bounded executor without a deadline must be refused");
    assert!(
        matches!(&error, NyError::UnsupportedOp(message) if message.contains("deadline")),
        "the refusal must point at the deadline-aware entry: {error:?}"
    );

    // With a deadline the same executor is served.
    let generous = Instant::now() + Duration::from_secs(30);
    let bounded = conv_case
        .layer
        .propagate_ibp_sound_with_engine_and_deadline(
            &conv_case.input,
            Some(&engine),
            Some(generous),
        )?;
    let plain = conv_case.layer.propagate_ibp(&conv_case.input)?;
    assert_eq!(bounded.lower(), plain.lower());
    assert_eq!(bounded.upper(), plain.upper());
    Ok(())
}

// ---------------------------------------------------------------------------
// Enum-level registration: the wiring must agree that Conv3d is IBP-only.
// ---------------------------------------------------------------------------

/// A Conv3d that ended up in `propagates_coeff_err` or `supports_batched_crown`
/// would be a soundness bug, not a missing feature: the first tells the scalar
/// dispatcher to hand this layer a certified coefficient error its backward
/// would silently drop, the second admits a whole network to a batched path
/// this node cannot take. Both must stay false while the CROWN backward is
/// absent, and the batched dispatch must refuse rather than return a plane.
#[ntest::timeout(10000)]
#[test]
fn conv3d_is_registered_as_an_ibp_only_layer() -> Result<()> {
    use crate::{BatchedLinearBounds, Layer};

    let layer = Layer::Conv3d(Conv3dLayer::new(
        unit_kernel(&[1, 1, 1, 1, 1]),
        None,
        [1, 1, 1],
        [0, 0, 0],
    )?);

    assert_eq!(layer.layer_type(), "Conv3d");
    assert!(
        !layer.propagates_coeff_err(),
        "Conv3d has no backward to propagate a certified coefficient error through"
    );
    assert!(
        !layer.is_exact_linear_coeff_err_carrier(),
        "the carrier path re-runs the backward Conv3d does not have"
    );
    assert!(
        !layer.supports_batched_crown(),
        "admitting Conv3d to the batched path would promise an arm that refuses"
    );

    let bounds = BatchedLinearBounds::identity(&[1])?;
    let point = ArrayD::from_elem(IxDyn(&[1, 1, 1, 1]), 0.5_f32);
    let pre = BoundedTensor::new(point.clone(), point)?;
    let error = layer
        .propagate_crown_backward_batched(&bounds, Some(&pre), None)
        .expect_err("the batched dispatch must refuse Conv3d");
    assert!(
        matches!(&error, NyError::UnsupportedOp(message) if message.contains("Conv3d")),
        "the batched refusal must name the layer: {error:?}"
    );

    // The IBP forward, by contrast, must reach the certified kernel through
    // the enum — this is the path a Conv3d node actually takes.
    let out = layer.propagate_ibp(&pre)?;
    assert_eq!(out.shape(), &[1, 1, 1, 1]);
    assert!(out.lower()[[0, 0, 0, 0]] <= 0.5 && 0.5 <= out.upper()[[0, 0, 0, 0]]);
    Ok(())
}
