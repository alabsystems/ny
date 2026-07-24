// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{Layer, Network};
use ny_tensor::BoundedTensor;
use ny_test_utils::assert_bounded_tensor_close;

const TOL: f32 = 1e-6;

fn build_compare_test_network() -> Network {
    let hidden_weight = arr2(&[[1.0_f32, 0.5], [-0.5, 1.0], [0.3, -0.7], [-0.2, 0.8]]);
    let output_weight = arr2(&[[1.0_f32, -0.5, 0.3, 0.2]]);
    let mut network = Network::new();
    network.add_layer(Layer::Linear(
        LinearLayer::new(hidden_weight, None).expect("hidden linear"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(
        LinearLayer::new(output_weight, None).expect("output linear"),
    ));
    network
}

#[test]
fn test_compare_cpu_default_crown_matches_engine_none_3622() {
    // compare.rs maps backend="cpu" to engine=None. This regression test proves
    // that the engine-aware API preserves the legacy CPU-default CROWN bounds.
    let network = build_compare_test_network();
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -0.75]).into_dyn(),
        arr1(&[1.0_f32, 0.5]).into_dyn(),
    )
    .expect("bounded input");

    let baseline = network
        .propagate_crown(&input)
        .expect("baseline CROWN should succeed");
    let with_engine_none = network
        .propagate_crown_with_engine(&input, None)
        .expect("engine=None CROWN should succeed");

    assert_bounded_tensor_close(
        &baseline,
        &with_engine_none,
        TOL,
        "compare cpu-default crown parity",
    );
}

#[test]
fn test_compare_cpu_default_alpha_matches_engine_none_3622() {
    // compare.rs maps backend="cpu" to engine=None. This regression test proves
    // that the engine-aware API preserves the legacy CPU-default alpha-CROWN bounds.
    let network = build_compare_test_network();
    let input = BoundedTensor::new(
        arr1(&[-0.5_f32, -1.0]).into_dyn(),
        arr1(&[0.75_f32, 1.0]).into_dyn(),
    )
    .expect("bounded input");

    let baseline = network
        .propagate_alpha_crown(&input)
        .expect("baseline alpha-CROWN should succeed");
    let with_engine_none = network
        .propagate_alpha_crown_with_engine(&input, None)
        .expect("engine=None alpha-CROWN should succeed");

    assert_bounded_tensor_close(
        &baseline,
        &with_engine_none,
        TOL,
        "compare cpu-default alpha parity",
    );
}
