// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::layers::{Conv2dLayer, FlattenLayer, LinearLayer, ReLULayer};
use crate::tests::crown::helpers::assert_bounds_finite;
use crate::*;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

fn scalar_bounds_3813(bounds: &BoundedTensor) -> (f32, f32) {
    let lower = bounds
        .lower()
        .iter()
        .next()
        .copied()
        .expect("toy truncated CROWN lower bound should contain one element");
    let upper = bounds
        .upper()
        .iter()
        .next()
        .copied()
        .expect("toy truncated CROWN upper bound should contain one element");
    (lower, upper)
}

fn build_truncated_conv_network_3813() -> (Network, BoundedTensor) {
    let kernel1 =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.6_f32, -0.3, 0.2, 0.5]).unwrap();
    let kernel2 =
        ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![0.4_f32, 0.1, -0.2, 0.7]).unwrap();
    let conv1 =
        Conv2dLayer::with_input_shape(kernel1, Some(arr1(&[0.05_f32])), (1, 1), (0, 0), 4, 4)
            .unwrap();
    let conv2 =
        Conv2dLayer::with_input_shape(kernel2, Some(arr1(&[-0.1_f32])), (1, 1), (0, 0), 3, 3)
            .unwrap();
    let linear =
        LinearLayer::new(arr2(&[[0.5_f32, -0.4, 0.3, 0.2]]), Some(arr1(&[0.05_f32]))).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Conv2d(conv2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Flatten(FlattenLayer::new(0)));
    network.add_layer(Layer::Linear(linear));

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), -0.4_f32),
        ArrayD::from_elem(IxDyn(&[1, 4, 4]), 0.6_f32),
    )
    .unwrap();

    (network, input)
}

fn truncated_conv_forward_3813(input: &[f32; 16]) -> f32 {
    let mut conv1 = [0.0_f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            let top_left = input[row * 4 + col];
            let top_right = input[row * 4 + col + 1];
            let bottom_left = input[(row + 1) * 4 + col];
            let bottom_right = input[(row + 1) * 4 + col + 1];
            let value =
                0.6 * top_left - 0.3 * top_right + 0.2 * bottom_left + 0.5 * bottom_right + 0.05;
            conv1[row * 3 + col] = value.max(0.0);
        }
    }

    let mut conv2 = [0.0_f32; 4];
    for row in 0..2 {
        for col in 0..2 {
            let top_left = conv1[row * 3 + col];
            let top_right = conv1[row * 3 + col + 1];
            let bottom_left = conv1[(row + 1) * 3 + col];
            let bottom_right = conv1[(row + 1) * 3 + col + 1];
            let value =
                0.4 * top_left + 0.1 * top_right - 0.2 * bottom_left + 0.7 * bottom_right - 0.1;
            conv2[row * 2 + col] = value.max(0.0);
        }
    }

    0.5 * conv2[0] - 0.4 * conv2[1] + 0.3 * conv2[2] + 0.2 * conv2[3] + 0.05
}

#[ntest::timeout(10000)]
#[test]
fn test_truncated_crown_sequential_contains_sampled_outputs_3813() {
    let (network, input) = build_truncated_conv_network_3813();
    let truncated = network
        .propagate_crown_with_engine_and_deadline_and_limits(&input, None, None, Some(2))
        .expect("truncated CROWN should succeed on the toy conv network");
    assert_bounds_finite(&truncated, "truncated CROWN sequential sampled output");
    let (truncated_lower, truncated_upper) = scalar_bounds_3813(&truncated);
    let midpoint = [0.1_f32; 16];
    let all_lower = [-0.4_f32; 16];
    let all_upper = [0.6_f32; 16];
    let checkerboard = core::array::from_fn(|idx| {
        let row = idx / 4;
        let col = idx % 4;
        if (row + col) % 2 == 0 {
            -0.4
        } else {
            0.6
        }
    });

    for (label, point) in [
        ("midpoint", midpoint),
        ("all-lower", all_lower),
        ("all-upper", all_upper),
        ("checkerboard", checkerboard),
    ] {
        let output = truncated_conv_forward_3813(&point);
        assert!(
            truncated_lower - 1e-5 <= output && output <= truncated_upper + 1e-5,
            "#3813 sampled sequential output must stay inside truncated CROWN: label={}, output={}, bounds=[{}, {}]",
            label,
            output,
            truncated_lower,
            truncated_upper,
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_truncated_crown_sequential_stays_tighter_than_ibp_3813() {
    let (network, input) = build_truncated_conv_network_3813();
    let ibp = network
        .propagate_ibp(&input)
        .expect("IBP should succeed on the toy conv network");
    let truncated = network
        .propagate_crown_with_engine_and_deadline_and_limits(&input, None, None, Some(2))
        .expect("truncated CROWN should succeed on the toy conv network");
    assert_bounds_finite(
        &truncated,
        "truncated CROWN sequential IBP-tightness output",
    );
    let (ibp_lower, ibp_upper) = scalar_bounds_3813(&ibp);
    let (truncated_lower, truncated_upper) = scalar_bounds_3813(&truncated);

    assert!(
        ibp_lower <= truncated_lower + 1e-5,
        "#3813 truncated CROWN lower should stay at least as tight as IBP: ibp={}, truncated={}",
        ibp_lower,
        truncated_lower,
    );
    assert!(
        truncated_upper <= ibp_upper + 1e-5,
        "#3813 truncated CROWN upper should stay at least as tight as IBP: truncated={}, ibp={}",
        truncated_upper,
        ibp_upper,
    );
}
