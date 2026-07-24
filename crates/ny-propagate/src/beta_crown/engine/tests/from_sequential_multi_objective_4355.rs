// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Regression test: Sequential→Graph multi-objective BaB with clip_in_alpha_crown.
//!
//! Reproduces the NaN child domain CROWN backward observed on cora_2024 when
//! `optimize_disjuncts_separately` routes through multi-objective Graph BaB
//! on models loaded as Sequential (MLP) and converted via `from_sequential`.
//!
//! Part of #4355.

use super::prelude::*;

fn build_cora_like_sequential() -> Network {
    build_mlp_sequential(8, 6, 4, 0.5)
}

fn build_cora_scale_sequential() -> Network {
    build_mlp_sequential(100, 50, 10, 0.5)
}

fn build_mlp_sequential(
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    scale: f32,
) -> Network {
    let w1 = Array2::from_shape_fn((hidden_dim, input_dim), |(i, j)| {
        ((i * input_dim + j) as f32 * 0.37).sin() * scale
    });
    let b1 = Array1::from_shape_fn(hidden_dim, |i| (i as f32 * 0.23).sin() * 0.1);
    let linear1 = LinearLayer::new(w1, Some(b1)).unwrap();

    let w2 = Array2::from_shape_fn((hidden_dim, hidden_dim), |(i, j)| {
        ((i * hidden_dim + j + 10000) as f32 * 0.43).sin() * scale
    });
    let b2 = Array1::from_shape_fn(hidden_dim, |i| (i as f32 * 0.31).sin() * 0.1);
    let linear2 = LinearLayer::new(w2, Some(b2)).unwrap();

    let w3 = Array2::from_shape_fn((output_dim, hidden_dim), |(i, j)| {
        ((i * hidden_dim + j + 20000) as f32 * 0.31).sin() * scale * 0.8
    });
    let b3 = Array1::from_shape_fn(output_dim, |i| (i as f32 * 0.41).sin() * 0.05);
    let linear3 = LinearLayer::new(w3, Some(b3)).unwrap();

    let mut network = Network::new();
    network.add_layer(Layer::Linear(linear1));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear2));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(linear3));
    network
}

fn cora_like_disjunctive_objectives() -> (Vec<Vec<f32>>, Vec<f32>) {
    let objectives = vec![
        vec![1.0, -1.0, 0.0, 0.0],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![1.0, 0.0, 0.0, -1.0],
    ];
    let thresholds = vec![0.0, 0.0, 0.0];
    (objectives, thresholds)
}

#[ntest::timeout(30000)]
#[test]
fn test_from_sequential_multi_objective_no_clip_4355() {
    let network = build_cora_like_sequential();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[8]), vec![-1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[8]), vec![1.0; 8]).unwrap(),
    )
    .unwrap();
    let (objectives, thresholds) = cora_like_disjunctive_objectives();

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(10),
        max_domains: 200,
        max_depth: 10,
        batch_size: 1,
        enable_cuts: false,
        clip_in_alpha_crown: false,
        beta_iterations: 5,
        ..Default::default()
    };
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("multi-objective Graph BaB without clip should not error");

    assert!(
        !matches!(result.result, BabVerificationStatus::Unknown { ref reason } if reason.contains("NaN")),
        "no-clip: should not fail due to NaN: {:?}",
        result.result
    );
}

#[ntest::timeout(30000)]
#[test]
fn test_from_sequential_multi_objective_with_clip_4355() {
    let network = build_cora_like_sequential();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[8]), vec![-1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[8]), vec![1.0; 8]).unwrap(),
    )
    .unwrap();
    let (objectives, thresholds) = cora_like_disjunctive_objectives();

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(10),
        max_domains: 200,
        max_depth: 10,
        batch_size: 1,
        enable_cuts: false,
        clip_in_alpha_crown: true,
        beta_iterations: 5,
        ..Default::default()
    };
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("multi-objective Graph BaB with clip should not error");

    assert!(
        !matches!(result.result, BabVerificationStatus::Unknown { ref reason } if reason.contains("NaN")),
        "with-clip: should not fail due to NaN: {:?}",
        result.result
    );
}

#[ntest::timeout(30000)]
#[test]
fn test_from_sequential_multi_objective_per_disjunct_alpha_4355() {
    let network = build_cora_like_sequential();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[8]), vec![-1.0; 8]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[8]), vec![1.0; 8]).unwrap(),
    )
    .unwrap();
    let (objectives, thresholds) = cora_like_disjunctive_objectives();

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(10),
        max_domains: 200,
        max_depth: 10,
        batch_size: 1,
        enable_cuts: false,
        clip_in_alpha_crown: true,
        beta_iterations: 5,
        optimize_disjuncts_separately: true,
        ..Default::default()
    };
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("per-disjunct alpha should not error");

    assert!(
        !matches!(result.result, BabVerificationStatus::Unknown { ref reason } if reason.contains("NaN")),
        "per-disjunct: should not fail due to NaN: {:?}",
        result.result
    );
}

fn build_direct_graph_4out() -> (GraphNetwork, BoundedTensor) {
    let w1 = Array2::from_shape_fn((6, 4), |(i, j)| ((i * 4 + j) as f32 * 0.37).sin() * 0.5);
    let b1 = Array1::from_shape_fn(6, |i| (i as f32 * 0.23).sin() * 0.1);
    let w2 = Array2::from_shape_fn((6, 6), |(i, j)| {
        ((i * 6 + j + 100) as f32 * 0.43).sin() * 0.5
    });
    let b2 = Array1::from_shape_fn(6, |i| (i as f32 * 0.31).sin() * 0.1);
    let w3 = Array2::from_shape_fn((4, 6), |(i, j)| {
        ((i * 6 + j + 200) as f32 * 0.29).sin() * 0.4
    });

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, Some(b1)).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, Some(b2)).unwrap()),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu2",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear3",
        Layer::Linear(LinearLayer::new(w3, None).unwrap()),
        vec!["relu2".to_string()],
    ));
    graph.set_output("linear3");

    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![-1.0; 4]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0; 4]).unwrap(),
    )
    .unwrap();
    (graph, input)
}

/// Direct GraphNetwork (not from_sequential) with per-disjunct alpha (#4355).
/// Regression test for "Child propagation failed" on benchmark models.
#[ntest::timeout(30000)]
#[test]
fn test_direct_graph_multi_objective_per_disjunct_alpha_4355() {
    let (graph, input) = build_direct_graph_4out();
    let objectives = vec![
        vec![1.0, -1.0, 0.0, 0.0],
        vec![1.0, 0.0, -1.0, 0.0],
        vec![1.0, 0.0, 0.0, -1.0],
    ];
    let thresholds = vec![0.0, 0.0, 0.0];

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(10),
        max_domains: 200,
        max_depth: 10,
        batch_size: 1,
        enable_cuts: false,
        clip_in_alpha_crown: true,
        beta_iterations: 5,
        optimize_disjuncts_separately: true,
        ..Default::default()
    };
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("direct graph per-disjunct alpha should not error");

    assert!(
        !matches!(result.result, BabVerificationStatus::Unknown { ref reason }
            if reason.contains("Child propagation failed")),
        "direct graph: should not fail on child propagation: {:?}",
        result.result
    );
    assert!(
        result.domains_explored > 1,
        "direct graph: per-disjunct should explore >1 domains (got {})",
        result.domains_explored
    );
}

/// Larger MLP (100→50→50→10) with clip_in_alpha_crown + per-disjunct alpha.
/// Closer to cora's scale to stress-test forward linear bounds accumulation.
#[ntest::timeout(60000)]
#[test]
fn test_from_sequential_scale_multi_objective_clip_4355() {
    let network = build_cora_scale_sequential();
    let graph = GraphNetwork::from_sequential(&network).unwrap();
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[100]), vec![-0.5; 100]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[100]), vec![0.5; 100]).unwrap(),
    )
    .unwrap();
    let objectives: Vec<Vec<f32>> = (1..10)
        .map(|j| {
            let mut obj = vec![0.0_f32; 10];
            obj[0] = 1.0;
            obj[j] = -1.0;
            obj
        })
        .collect();
    let thresholds = vec![0.0_f32; 9];

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(20),
        max_domains: 100,
        max_depth: 5,
        batch_size: 1,
        enable_cuts: false,
        clip_in_alpha_crown: true,
        beta_iterations: 5,
        optimize_disjuncts_separately: true,
        ..Default::default()
    };
    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective_with_engine(
            &graph,
            &input,
            &objectives,
            &thresholds,
            None,
            None,
        )
        .expect("scale multi-objective Graph BaB should not error");

    assert!(
        !matches!(result.result, BabVerificationStatus::Unknown { ref reason } if reason.contains("NaN")),
        "scale: should not fail due to NaN: {:?}",
        result.result
    );
}
