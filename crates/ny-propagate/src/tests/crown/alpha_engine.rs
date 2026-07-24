// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Sequential alpha-CROWN engine parity regressions.

use crate::tests::crown::helpers::{assert_bounds_finite, CountingGemmEngine};
use crate::*;
use ndarray::{arr1, arr2};
use ny_test_utils::assert_bounded_tensor_close;

fn build_relu_network() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    let w1 = arr2(&[[1.0_f32, 2.0], [-1.0, 1.0], [0.5, -0.5]]);
    let b1 = arr1(&[0.1_f32, -0.2, 0.3]);
    let w2 = arr2(&[[1.0_f32, -1.0, 0.5], [0.5, 1.0, -0.5]]);
    let b2 = arr1(&[0.0_f32, 0.1]);

    network.add_layer(Layer::Linear(
        LinearLayer::new(w1, Some(b1)).expect("valid first Linear layer"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2, Some(b2)).expect("valid second Linear layer"),
    ));

    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -0.5]).into_dyn(),
        arr1(&[0.5_f32, 0.5]).into_dyn(),
    )
    .expect("valid input bounds");

    (network, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_sound_with_engine_matches_baseline_3772() {
    let (network, input) = build_relu_network();

    let baseline = network
        .propagate_alpha_crown_sound(&input)
        .expect("baseline sequential alpha-CROWN sound path should succeed");

    let engine = CountingGemmEngine::new();
    let with_engine = network
        .propagate_alpha_crown_sound_with_engine(&input, Some(&engine))
        .expect("engine-aware sequential alpha-CROWN sound path should succeed");

    assert_bounds_finite(&with_engine, "alpha-CROWN sound with engine output");
    assert_bounded_tensor_close(
        &with_engine,
        &baseline,
        1e-6,
        "#3772 sequential alpha-CROWN sound wrapper parity",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#3772 regression: sequential propagate_alpha_crown_sound_with_engine should hit GemmEngine"
    );
}
