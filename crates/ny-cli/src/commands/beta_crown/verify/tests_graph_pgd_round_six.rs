// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! #4096 whitelist rejection tests: axis-sensitive blocked graph families.
//!
//! Split from `tests_graph_pgd.rs` for file size compliance.

use ny_propagate::{
    layers::{
        GatherLayer, LayerNormLayer, LinearLayer, ReduceMeanLayer, SqueezeLayer, UnsqueezeLayer,
    },
    GraphNetwork, Layer, Network,
};
use ny_test_utils::CountingGemmEngine;
use ndarray::{arr1, arr2, ArrayD, IxDyn};

use super::graph_pgd::try_graph_pgd_upfront;
use super::{make_interval_input, make_upper_bound_spec};
use super::{make_multi_input_upper_bound_spec, make_tensor_interval_input};

fn make_layernorm_linear_graph() -> GraphNetwork {
    let ln = LayerNormLayer::new(arr1(&[1.0, 1.0]), arr1(&[0.0, 0.0]), 1e-5)
        .expect("layernorm params should be valid");
    let mut network = Network::new();
    network.add_layer(Layer::LayerNorm(ln));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("layernorm+linear network should convert to graph")
}

fn make_squeeze_linear_graph() -> GraphNetwork {
    // Squeeze axis 1 on a [N, 1] input → [N], then linear maps N→1.
    let mut network = Network::new();
    network.add_layer(Layer::Squeeze(SqueezeLayer::new(1)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0, 1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("squeeze+linear network should convert to graph")
}

fn make_unsqueeze_linear_graph() -> GraphNetwork {
    let mut network = Network::new();
    network.add_layer(Layer::Unsqueeze(UnsqueezeLayer::new(0)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("unsqueeze+linear network should convert to graph")
}

fn make_reducemean_linear_graph() -> GraphNetwork {
    // ReduceMean with keepdims=true so the output retains rank for the linear layer.
    let mut network = Network::new();
    network.add_layer(Layer::ReduceMean(ReduceMeanLayer::new(vec![-1], true)));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network)
        .expect("reducemean+linear network should convert to graph")
}

fn make_gather_linear_graph() -> GraphNetwork {
    let indices = ArrayD::from_shape_vec(IxDyn(&[1]), vec![0i64]).unwrap();
    let mut network = Network::new();
    network.add_layer(Layer::Gather(GatherLayer::new(0, Some(indices), vec![1])));
    network.add_layer(Layer::Linear(
        LinearLayer::new(arr2(&[[1.0]]), Some(arr1(&[0.0]))).unwrap(),
    ));
    GraphNetwork::from_sequential(&network).expect("gather+linear network should convert to graph")
}

#[test]
fn graph_pgd_whitelist_rejects_layernorm_graphs_4096() {
    let graph = make_layernorm_linear_graph();
    let input = make_tensor_interval_input(&[2], 0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(2, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("layernorm graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed layernorm graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: layernorm graphs should miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_whitelist_rejects_squeeze_graphs_4096() {
    let graph = make_squeeze_linear_graph();
    let input = make_tensor_interval_input(&[2, 1], 0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(2, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("squeeze graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed squeeze graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: squeeze graphs should miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_whitelist_rejects_unsqueeze_graphs_4096() {
    let graph = make_unsqueeze_linear_graph();
    let input = make_interval_input(0.0, 1.0);
    let spec = make_upper_bound_spec(0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("unsqueeze graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed unsqueeze graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: unsqueeze graphs should miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_whitelist_rejects_reducemean_graphs_4096() {
    let graph = make_reducemean_linear_graph();
    let input = make_tensor_interval_input(&[2], 0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(2, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("reducemean graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed reducemean graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: reducemean graphs should miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}

#[test]
fn graph_pgd_whitelist_rejects_gather_graphs_4096() {
    let graph = make_gather_linear_graph();
    let input = make_tensor_interval_input(&[2], 0.0, 1.0);
    let spec = make_multi_input_upper_bound_spec(2, 0.0, 1.0, -1.0);
    let engine = CountingGemmEngine::new();

    let result = try_graph_pgd_upfront(
        &graph,
        &input,
        &spec,
        10,
        5,
        Default::default(),
        20,
        None,
        Some(&engine),
        true,
        false,
    )
    .expect("gather graphs should stay on the sequential fallback path");

    assert!(
        result.is_none(),
        "the fixed gather graph should stay safely above the impossible threshold"
    );
    assert!(
        engine.gemm_calls() > 10,
        "#4096 regression: gather graphs should miss the batched restart path, got {} GEMM calls",
        engine.gemm_calls()
    );
}
