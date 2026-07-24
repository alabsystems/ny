// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::tests::{assert_all_close, crown::helpers::CountingGemmEngine};
use ndarray::{arr1, ArrayD, IxDyn};

#[ntest::timeout(10000)]
#[test]
fn test_network_propagate_ibp_with_engine_threads_conv1d_layers_4081() {
    let kernel =
        ArrayD::from_shape_vec(IxDyn(&[2, 1, 3]), vec![0.5, -0.25, 0.75, -0.2, 0.4, 0.1]).unwrap();
    let conv =
        Conv1dLayer::with_input_length(kernel, Some(arr1(&[0.15_f32, -0.05])), 1, 1, 6).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Conv1d(conv));
    network.add_layer(Layer::ReLU(ReLULayer));

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![-0.5_f32, -0.25, 0.0, -0.1, -0.2, -0.3])
            .unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 6]), vec![0.75_f32, 0.5, 0.4, 0.6, 0.8, 0.7]).unwrap(),
    )
    .unwrap();

    let baseline = network.propagate_ibp(&input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = network
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        "network conv1d ibp with engine lower",
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        "network conv1d ibp with engine upper",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#4081 regression: Network::propagate_ibp_with_engine should thread GemmEngine to Conv1d layers"
    );
}
