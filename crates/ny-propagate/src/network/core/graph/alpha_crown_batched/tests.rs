// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Tests for batched alpha-CROWN optimization (BilinearCrown attention Q@K^T).

use ndarray::{arr1, array};
use ny_tensor::BoundedTensor;

use crate::bounds::AlphaCrownConfig;
use crate::layers::binary_ops::BilinearCrownLayer;
use crate::layers::linear::LinearLayer;
use crate::layers::Layer;
use crate::network::alpha_crown_loop::finite_lower_sum;
use crate::network::core::graph::{GraphNetwork, GraphNode};

/// Build a minimal graph with a BilinearCrown node for testing.
fn build_bilinear_graph() -> GraphNetwork {
    use crate::network::core::graph::NETWORK_INPUT;
    let linear_q = LinearLayer::new(
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .unwrap();
    let linear_k = LinearLayer::new(
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .unwrap();
    let bilinear = BilinearCrownLayer::new(false, Some(1.0));
    let readout = LinearLayer::new(
        array![[1.0_f32, 0.0], [0.0, 1.0]],
        Some(arr1(&[0.0_f32, 0.0])),
    )
    .unwrap();

    let mut graph = GraphNetwork {
        output_node: "readout".to_string(),
        ..GraphNetwork::new()
    };
    graph.nodes.insert(
        "linear_q".to_string(),
        GraphNode {
            name: "linear_q".to_string(),
            layer: Layer::Linear(linear_q),
            inputs: vec![NETWORK_INPUT.to_string()],
        },
    );
    graph.nodes.insert(
        "linear_k".to_string(),
        GraphNode {
            name: "linear_k".to_string(),
            layer: Layer::Linear(linear_k),
            inputs: vec![NETWORK_INPUT.to_string()],
        },
    );
    graph.nodes.insert(
        "bilinear".to_string(),
        GraphNode {
            name: "bilinear".to_string(),
            layer: Layer::BilinearCrown(bilinear),
            inputs: vec!["linear_q".to_string(), "linear_k".to_string()],
        },
    );
    graph.nodes.insert(
        "readout".to_string(),
        GraphNode {
            name: "readout".to_string(),
            layer: Layer::Linear(readout),
            inputs: vec!["bilinear".to_string()],
        },
    );
    graph.node_order = vec![
        "linear_q".to_string(),
        "linear_k".to_string(),
        "bilinear".to_string(),
        "readout".to_string(),
    ];
    graph
}

fn make_small_2d_input() -> BoundedTensor {
    BoundedTensor::new(
        ndarray::array![[-0.5_f32, -0.5], [-0.5, -0.5]].into_dyn(),
        ndarray::array![[0.5_f32, 0.5], [0.5, 0.5]].into_dyn(),
    )
    .unwrap()
}

#[test]
fn test_collect_bilinear_nodes_finds_bilinear() {
    let graph = build_bilinear_graph();
    let input = BoundedTensor::new(
        ndarray::array![[-1.0_f32, -1.0], [-1.0, -1.0]].into_dyn(),
        ndarray::array![[1.0_f32, 1.0], [1.0, 1.0]].into_dyn(),
    )
    .unwrap();

    let nodes = graph.collect_bilinear_nodes(&input, None).unwrap();
    assert_eq!(nodes.len(), 1);
    assert!(nodes.contains_key("bilinear"));
    let (m, n, k) = nodes["bilinear"];
    assert_eq!((m, n, k), (2, 2, 2));
}

#[test]
fn test_alpha_crown_batched_runs_without_panic() {
    let graph = build_bilinear_graph();
    let input = make_small_2d_input();
    let config = AlphaCrownConfig {
        iterations: 3,
        spsa_samples: 1,
        ..AlphaCrownConfig::default()
    };

    let result = graph
        .alpha_crown_batched_optimize(&input, &config, None)
        .unwrap();
    assert!(!result.bounds.lower().iter().any(|v| v.is_nan()));
    assert!(!result.bounds.upper().iter().any(|v| v.is_nan()));
}

#[test]
fn test_alpha_crown_batched_not_worse_than_crown() {
    let graph = build_bilinear_graph();
    let input = make_small_2d_input();

    let crown_bounds = graph.propagate_crown_batched(&input).unwrap();

    let config = AlphaCrownConfig {
        iterations: 5,
        spsa_samples: 2,
        ..AlphaCrownConfig::default()
    };
    let alpha_result = graph
        .alpha_crown_batched_optimize(&input, &config, None)
        .unwrap();

    let crown_lower_sum = finite_lower_sum(crown_bounds.lower());
    let alpha_lower_sum = finite_lower_sum(alpha_result.bounds.lower());

    assert!(
        alpha_lower_sum >= crown_lower_sum - 1e-6,
        "alpha-CROWN lower_sum ({}) should be >= CROWN ({}) (element-wise best tracking)",
        alpha_lower_sum,
        crown_lower_sum
    );
}

/// Regression test for #3588: the no-ReLU DAG alpha path must thread the
/// GemmEngine into the batched bilinear alpha optimizer, not drop it.
///
/// Uses a CountingGemmEngine to verify the no-ReLU alpha-CROWN path performs
/// real GEMM work through the provided engine.
#[test]
fn test_alpha_crown_batched_threads_engine_3588() {
    use ny_test_utils::CountingGemmEngine;

    let graph = build_bilinear_graph();
    let input = make_small_2d_input();

    // Verify the batched alpha-CROWN optimizer accepts a non-None engine.
    // Before #3588, propagate_alpha_crown_batched had no engine parameter.
    let engine = CountingGemmEngine::new();
    let config = AlphaCrownConfig {
        iterations: 2,
        spsa_samples: 1,
        ..AlphaCrownConfig::default()
    };

    let result = graph
        .propagate_alpha_crown_batched(&input, &config, Some(&engine))
        .unwrap();

    // Bounds must be valid (no NaN).
    assert!(
        !result.lower().iter().any(|v| v.is_nan()),
        "#3588 regression: engine-threaded batched alpha-CROWN produced NaN lower bounds"
    );
    assert!(
        !result.upper().iter().any(|v| v.is_nan()),
        "#3588 regression: engine-threaded batched alpha-CROWN produced NaN upper bounds"
    );

    // Bounds must be at least as tight as the None-engine baseline.
    let baseline = graph
        .propagate_alpha_crown_batched(&input, &config, None)
        .unwrap();
    let engine_lower_sum = finite_lower_sum(result.lower());
    let baseline_lower_sum = finite_lower_sum(baseline.lower());
    assert!(
        engine_lower_sum >= baseline_lower_sum - 1e-6,
        "#3588 regression: engine path lower_sum ({}) should be >= baseline ({})",
        engine_lower_sum,
        baseline_lower_sum
    );

    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "#3588 regression: no-ReLU batched alpha-CROWN should hit GemmEngine, got {calls} calls"
    );
}

/// Regression test for #3588: the full DAG alpha-CROWN entrypoint must
/// thread the engine through the no-ReLU bilinear branch.
///
/// This graph has no ReLU nodes, so the DAG dispatcher enters the
/// `propagate_alpha_crown_batched` path. Uses the public
/// `propagate_alpha_crown_with_config_and_engine` which routes to the
/// DAG implementation for non-sequential graphs.
#[test]
fn test_dag_alpha_crown_no_relu_threads_engine_3588() {
    use ny_test_utils::CountingGemmEngine;

    let graph = build_bilinear_graph();
    let input = make_small_2d_input();
    let engine = CountingGemmEngine::new();

    let config = AlphaCrownConfig {
        iterations: 2,
        spsa_samples: 1,
        ..AlphaCrownConfig::default()
    };

    // Before #3588, the no-ReLU branch called
    // propagate_alpha_crown_batched(input, config) — dropping the engine.
    // After #3588, engine is threaded all the way through.
    // This routes to DAG alpha-CROWN because the bilinear graph is
    // non-sequential.
    let result = graph
        .propagate_alpha_crown_with_config_and_engine(&input, &config, Some(&engine))
        .unwrap();

    assert!(
        !result.lower().iter().any(|v: &f32| v.is_nan()),
        "#3588 regression: DAG alpha-CROWN no-ReLU path produced NaN"
    );
    let calls = engine.gemm_calls();
    assert!(
        calls > 0,
        "#3588 regression: DAG no-ReLU alpha-CROWN should hit GemmEngine, got {calls} calls"
    );
}

#[test]
fn test_dag_alpha_crown_no_relu_engine_matches_none_baseline_3588() {
    let graph = build_bilinear_graph();
    let input = make_small_2d_input();
    let config = AlphaCrownConfig {
        iterations: 2,
        spsa_samples: 1,
        ..AlphaCrownConfig::default()
    };

    let baseline = graph
        .propagate_alpha_crown_with_config_and_engine(&input, &config, None)
        .unwrap();
    let with_engine = graph
        .propagate_alpha_crown_with_config_and_engine(
            &input,
            &config,
            Some(&ny_core::NaiveCpuGemmEngine),
        )
        .unwrap();

    assert_eq!(
        with_engine.shape(),
        baseline.shape(),
        "#3588 regression: engine/no-engine DAG alpha-CROWN shapes diverged"
    );

    for (idx, (&actual, &expected)) in with_engine
        .lower()
        .iter()
        .zip(baseline.lower().iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "#3588 regression: lower bound mismatch at flat index {idx}: actual={actual}, expected={expected}"
        );
    }

    for (idx, (&actual, &expected)) in with_engine
        .upper()
        .iter()
        .zip(baseline.upper().iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "#3588 regression: upper bound mismatch at flat index {idx}: actual={actual}, expected={expected}"
        );
    }
}
