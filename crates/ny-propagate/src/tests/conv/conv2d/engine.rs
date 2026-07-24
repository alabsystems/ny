// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::tests::{assert_all_close, crown::helpers::CountingGemmEngine};
use ndarray::{arr1, ArrayD, IxDyn};

fn build_conv2d_network_case() -> (Network, BoundedTensor) {
    let kernel = ArrayD::from_shape_vec(
        IxDyn(&[2, 1, 2, 2]),
        vec![0.5, -0.1, 0.25, 0.75, -0.2, 0.3, 0.4, -0.15],
    )
    .unwrap();
    let conv =
        Conv2dLayer::with_input_shape(kernel, Some(arr1(&[0.1_f32, -0.2])), (1, 1), (0, 0), 3, 3)
            .unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv2d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![-0.5_f32, -0.25, 0.0, -0.1, -0.2, -0.3, 0.2, -0.4, 0.1],
        )
        .unwrap(),
        ArrayD::from_shape_vec(
            IxDyn(&[1, 3, 3]),
            vec![0.75_f32, 0.5, 0.4, 0.6, 0.8, 0.7, 0.9, 0.3, 0.5],
        )
        .unwrap(),
    )
    .unwrap();

    (network, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_ibp_with_engine_threads_conv2d_layers_4081() {
    let (network, input) = build_conv2d_network_case();
    let baseline = network.propagate_ibp(&input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = network
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        "network conv2d ibp with engine lower",
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        "network conv2d ibp with engine upper",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#4081 regression: Network::propagate_ibp_with_engine should thread GemmEngine to Conv2d layers"
    );
}
