// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, ArrayD, IxDyn};
use ny_propagate::{
    layers::{Conv1dLayer, Conv2dLayer, ReLULayer},
    GraphNetwork, Layer, Network, PgdAttacker, PgdConfig,
};
use ny_tensor::BoundedTensor;
use ny_test_utils::CountingGemmEngine;

fn assert_all_close(actual: &ArrayD<f32>, expected: &ArrayD<f32>, tol: f32, label: &str) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape mismatch {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (actual_value - expected_value).abs() <= tol,
            "{label}: element {index} mismatch: actual={actual_value} expected={expected_value} tol={tol}"
        );
    }
}

fn build_conv1d_network_case() -> (Network, BoundedTensor) {
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

    (network, input)
}

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

fn build_positive_conv1d_network() -> Network {
    let mut network = Network::new();
    network.add_layer(Layer::Conv1d(
        Conv1dLayer::with_input_length(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0_f32]).unwrap(),
            Some(arr1(&[2.0_f32])),
            1,
            0,
            1,
        )
        .unwrap(),
    ));
    network
}

fn assert_engine_parity(network: &Network, input: &BoundedTensor, label: &str) {
    let baseline = network.propagate_ibp(input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = network
        .propagate_ibp_with_engine(input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        &format!("{label} lower"),
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        &format!("{label} upper"),
    );
    assert!(
        engine.gemm_calls() > 0,
        "{label}: expected GemmEngine-backed IBP path to perform at least one GEMM"
    );
}

#[test]
fn network_propagate_ibp_with_engine_threads_conv1d_layers_4081() {
    let (network, input) = build_conv1d_network_case();
    assert_engine_parity(&network, &input, "network conv1d ibp with engine");
}

#[test]
fn network_propagate_ibp_with_engine_threads_conv2d_layers_4081() {
    let (network, input) = build_conv2d_network_case();
    assert_engine_parity(&network, &input, "network conv2d ibp with engine");
}

#[test]
fn graph_network_propagate_ibp_with_engine_threads_conv1d_nodes_4081() {
    let (network, input) = build_conv1d_network_case();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let baseline = graph.propagate_ibp(&input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        "graph conv1d ibp with engine lower",
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        "graph conv1d ibp with engine upper",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#4081 regression: GraphNetwork::propagate_ibp_with_engine should thread GemmEngine to Conv1d nodes"
    );
}

#[test]
fn graph_network_propagate_ibp_with_engine_threads_conv2d_nodes_4081() {
    let (network, input) = build_conv2d_network_case();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let baseline = graph.propagate_ibp(&input).unwrap();
    let engine = CountingGemmEngine::new();
    let with_engine = graph
        .propagate_ibp_with_engine(&input, Some(&engine))
        .unwrap();

    assert_all_close(
        with_engine.lower(),
        baseline.lower(),
        1e-5,
        "graph conv2d ibp with engine lower",
    );
    assert_all_close(
        with_engine.upper(),
        baseline.upper(),
        1e-5,
        "graph conv2d ibp with engine upper",
    );
    assert!(
        engine.gemm_calls() > 0,
        "#4081 regression: GraphNetwork::propagate_ibp_with_engine should thread GemmEngine to Conv2d nodes"
    );
}

#[test]
fn pgd_attack_threads_gemm_engine_for_conv1d_batches_4081() {
    let network = build_positive_conv1d_network();
    let engine = CountingGemmEngine::new();
    let attacker = PgdAttacker::new(PgdConfig {
        num_restarts: 4,
        num_steps: 3,
        step_size: 0.01,
        spsa_delta: 0.001,
        seed: 42,
        parallel: false,
        deadline: None,
        restart_when_stuck: false,
        ..Default::default()
    })
    .with_engine(&engine);

    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0_f32]).unwrap(),
    )
    .unwrap();

    let result = attacker
        .attack(&network, &input_bounds, 0, 0.0, false)
        .unwrap();

    assert!(
        !result.found_counterexample,
        "x + 2 stays above zero across [0, 1], so Conv1d PGD should not find a violation"
    );
    assert!(
        engine.gemm_calls() > 0,
        "#4081 regression: PgdAttacker should thread GemmEngine through Conv1d IBP batches"
    );
}
