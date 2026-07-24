// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! External public API smoke test for engine-aware parallel helpers (#3772).
//!
//! This test proves that `verify_parallel_with_engine` and
//! `verify_parallel_with_method_and_engine` are reachable from both the crate
//! root (`ny_propagate::{...}`) and the prelude (`ny_propagate::prelude::*`).
//!
//! A unit test inside `src/parallel/tests.rs` compiles inside the crate and
//! cannot prove public surface discoverability; this integration test compiles
//! as an external consumer.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_core::NaiveCpuGemmEngine;
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::types::PropagationMethod;
use ny_propagate::{
    verify_parallel_with_engine, verify_parallel_with_method_and_engine, GraphNetwork, GraphNode,
    Layer, Network, ParallelConfig, ParallelVerifier,
};
use ny_tensor::BoundedTensor;
use ny_test_utils::{assert_bounded_tensor_close, CountingGemmEngine};

// Verify the same symbols are reachable via the prelude.
#[allow(unused_imports)]
use ny_propagate::prelude::{
    verify_parallel_with_engine as _prelude_engine,
    verify_parallel_with_method_and_engine as _prelude_method_engine,
};

/// Build a minimal Linear → ReLU → Linear graph for parallel position testing.
fn build_small_mlp(hidden_dim: usize) -> GraphNetwork {
    let linear1 = LinearLayer::new(
        Array2::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
            if i == j {
                0.5_f32
            } else {
                0.01
            }
        }),
        Some(Array1::zeros(hidden_dim)),
    )
    .expect("linear1 construction should succeed");

    let relu = ReLULayer::new();

    let linear2 = LinearLayer::new(
        Array2::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
            if i == j {
                0.3_f32
            } else {
                -0.01
            }
        }),
        Some(Array1::zeros(hidden_dim)),
    )
    .expect("linear2 construction should succeed");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(relu),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// Make a 2D bounded input: [seq_len, hidden_dim].
fn make_2d_input(seq_len: usize, hidden_dim: usize) -> BoundedTensor {
    let values = ArrayD::from_elem(IxDyn(&[seq_len, hidden_dim]), 0.5_f32);
    BoundedTensor::from_epsilon(values, 0.1).expect("bounded input should be valid")
}

fn build_small_sequential_network() -> (Network, BoundedTensor) {
    let mut network = Network::new();
    let w1 = Array2::from_shape_vec((3, 2), vec![1.0_f32, 2.0, -1.0, 1.0, 0.5, -0.5])
        .expect("first Linear weights should be valid");
    let b1 = Array1::from_vec(vec![0.1_f32, -0.2, 0.3]);
    let w2 = Array2::from_shape_vec((2, 3), vec![1.0_f32, -1.0, 0.5, 0.5, 1.0, -0.5])
        .expect("second Linear weights should be valid");
    let b2 = Array1::from_vec(vec![0.0_f32, 0.1]);

    network.add_layer(Layer::Linear(
        LinearLayer::new(w1, Some(b1)).expect("first Linear layer should be valid"),
    ));
    network.add_layer(Layer::ReLU(ReLULayer::new()));
    network.add_layer(Layer::Linear(
        LinearLayer::new(w2, Some(b2)).expect("second Linear layer should be valid"),
    ));

    let input = BoundedTensor::new(
        Array1::from_vec(vec![-0.5_f32, -0.5]).into_dyn(),
        Array1::from_vec(vec![0.5_f32, 0.5]).into_dyn(),
    )
    .expect("sequential input bounds should be valid");

    (network, input)
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_parallel_with_engine_compiles_and_runs_from_crate_root() {
    let graph = build_small_mlp(4);
    let input = make_2d_input(2, 4);
    let engine = Arc::new(NaiveCpuGemmEngine);

    // Call the crate-root re-export. This proves the symbol is publicly reachable.
    let result = verify_parallel_with_engine(&graph, &input, 0, engine);
    assert!(
        result.is_ok(),
        "verify_parallel_with_engine should succeed on a simple MLP: {:?}",
        result.err()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_verify_parallel_with_method_and_engine_compiles_and_runs_from_crate_root() {
    let graph = build_small_mlp(4);
    let input = make_2d_input(2, 4);
    let engine = Arc::new(NaiveCpuGemmEngine);

    let result =
        verify_parallel_with_method_and_engine(&graph, &input, 0, PropagationMethod::Crown, engine);
    assert!(
        result.is_ok(),
        "verify_parallel_with_method_and_engine should succeed on a simple MLP: {:?}",
        result.err()
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_engine_aware_parallel_matches_cpu_default() {
    let graph = build_small_mlp(4);
    let input = make_2d_input(2, 4);
    let engine = Arc::new(NaiveCpuGemmEngine);

    // Call without engine (CPU default).
    let cpu_result = ny_propagate::verify_parallel(&graph, &input, 0)
        .expect("CPU-default verify_parallel should succeed");

    // Call with NaiveCpuGemmEngine — should produce identical bounds.
    let engine_result = verify_parallel_with_engine(&graph, &input, 0, engine)
        .expect("engine-aware verify_parallel should succeed");

    assert_eq!(
        cpu_result.lower(),
        engine_result.lower(),
        "lower bounds should match between CPU-default and NaiveCpuGemmEngine"
    );
    assert_eq!(
        cpu_result.upper(),
        engine_result.upper(),
        "upper bounds should match between CPU-default and NaiveCpuGemmEngine"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_parallel_verifier_new_with_engine_accessible_from_crate_root() {
    // Verify that ParallelVerifier::new_with_engine is also accessible.
    let config = ParallelConfig::default();
    let engine = Arc::new(NaiveCpuGemmEngine);
    let _verifier = ParallelVerifier::new_with_engine(config, engine);
}

#[ntest::timeout(10000)]
#[test]
fn test_alpha_crown_sound_with_engine_matches_baseline_3772() {
    let (network, input) = build_small_sequential_network();

    let baseline = network
        .propagate_alpha_crown_sound(&input)
        .expect("baseline sequential alpha-CROWN sound path should succeed");

    let engine = CountingGemmEngine::new();
    let with_engine = network
        .propagate_alpha_crown_sound_with_engine(&input, Some(&engine))
        .expect("engine-aware sequential alpha-CROWN sound path should succeed");

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
