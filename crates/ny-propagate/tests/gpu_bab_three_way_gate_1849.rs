// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use ndarray::{arr1, Array, Array2};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownResult, BetaCrownVerifier, BranchingHeuristic,
    GraphNetwork, GraphNode, Layer, Network,
};
use ny_tensor::BoundedTensor;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn build_three_way_networks_1849() -> (Network, GraphNetwork) {
    let mut rng = StdRng::seed_from_u64(42);

    let w1: Array2<f32> = Array::from_shape_fn((8, 2), |_| {
        use rand::RngExt;
        rng.random_range(-1.0..1.0)
    });
    let w2: Array2<f32> = Array::from_shape_fn((8, 8), |_| {
        use rand::RngExt;
        rng.random_range(-0.5..0.5)
    });
    let w3: Array2<f32> = Array::from_shape_fn((8, 8), |_| {
        use rand::RngExt;
        rng.random_range(-0.5..0.5)
    });
    let w4: Array2<f32> = Array::from_shape_fn((8, 8), |_| {
        use rand::RngExt;
        rng.random_range(-0.5..0.5)
    });
    let w5: Array2<f32> = Array::from_shape_fn((1, 8), |_| {
        use rand::RngExt;
        rng.random_range(-1.0..1.0)
    });

    let mut network = Network::new();
    network.add_layer(Layer::Linear(LinearLayer::new(w1.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w2.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w3.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w4.clone(), None).unwrap()));
    network.add_layer(Layer::ReLU(ReLULayer));
    network.add_layer(Layer::Linear(LinearLayer::new(w5.clone(), None).unwrap()));

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "linear1",
        Layer::Linear(LinearLayer::new(w1, None).unwrap()),
    ));
    graph.add_node(GraphNode::new(
        "relu1",
        Layer::ReLU(ReLULayer),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(LinearLayer::new(w2, None).unwrap()),
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
    graph.add_node(GraphNode::new(
        "relu3",
        Layer::ReLU(ReLULayer),
        vec!["linear3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear4",
        Layer::Linear(LinearLayer::new(w4, None).unwrap()),
        vec!["relu3".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu4",
        Layer::ReLU(ReLULayer),
        vec!["linear4".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear5",
        Layer::Linear(LinearLayer::new(w5, None).unwrap()),
        vec!["relu4".to_string()],
    ));
    graph.set_output("linear5");

    (network, graph)
}

fn sample_max_output_1849(network: &Network) -> f32 {
    const GRID: usize = 101;
    let mut max_out = f32::NEG_INFINITY;

    for i in 0..GRID {
        let x0 = -1.0 + 2.0 * (i as f32 / (GRID - 1) as f32);
        for j in 0..GRID {
            let x1 = -1.0 + 2.0 * (j as f32 / (GRID - 1) as f32);
            let point =
                BoundedTensor::new(arr1(&[x0, x1]).into_dyn(), arr1(&[x0, x1]).into_dyn()).unwrap();
            let output = network.propagate_ibp(&point).unwrap();
            max_out = max_out.max(output.lower()[[0]]);
        }
    }

    max_out
}

fn result_kind(result: &BabVerificationStatus) -> &'static str {
    match result {
        BabVerificationStatus::Verified => "Verified",
        BabVerificationStatus::Violated { .. } => "Violated",
        BabVerificationStatus::PotentialViolation => "PotentialViolation",
        BabVerificationStatus::Unknown { .. } => "Unknown",
        BabVerificationStatus::Timeout => "Timeout",
    }
}

fn emit_gate_metric(path: &str, result: &BetaCrownResult) {
    let verify_rate = if result.domains_explored > 0 {
        result.domains_verified as f64 / result.domains_explored as f64
    } else {
        0.0
    };
    println!(
        "GATE_METRIC path={} result={} explored={} verified={} elapsed_sec={:.6} verify_rate={:.6}",
        path,
        result_kind(&result.result),
        result.domains_explored,
        result.domains_verified,
        result.time_elapsed.as_secs_f64(),
        verify_rate,
    );
}

#[ntest::timeout(60000)]
#[test]
fn test_three_way_bab_gate_1849() {
    let (network, graph) = build_three_way_networks_1849();
    let input =
        BoundedTensor::new(arr1(&[-1.0, -1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();
    let objective = vec![1.0_f32];

    let sampled_max = sample_max_output_1849(&network);
    let threshold = sampled_max + 0.5;

    let config = BetaCrownConfig {
        verify_upper_bound: true,
        timeout: Duration::from_secs(15),
        max_domains: 50000,
        max_depth: 50,
        batch_size: 64,
        branching_heuristic: BranchingHeuristic::BoundImpact,
        ..Default::default()
    };

    println!(
        "GATE_CONFIG timeout_sec=15 max_domains=50000 max_depth=50 batch_size=64 heuristic=BoundImpact"
    );
    println!(
        "GATE_INPUT sampled_max={:.6} threshold={:.6}",
        sampled_max, threshold
    );

    let sequential = BetaCrownVerifier::new(config.clone())
        .verify(&network, &input, threshold)
        .unwrap();
    let graph_split = BetaCrownVerifier::new(config.clone())
        .verify_graph_relu_split(&graph, &input, &objective, threshold)
        .unwrap();
    let gpu = BetaCrownVerifier::new(config)
        .verify_graph_gpu_domain_list(&graph, &input, &objective, threshold, None, None)
        .unwrap();

    emit_gate_metric("sequential", &sequential);
    emit_gate_metric("graph_split", &graph_split);
    emit_gate_metric("gpu", &gpu);

    assert!(
        sequential.domains_verified > 0,
        "sequential path should verify at least one domain: {:?}",
        sequential.result
    );
    assert!(
        graph_split.domains_verified > 0,
        "graph split path should verify at least one domain: {:?}",
        graph_split.result
    );
    assert!(
        gpu.domains_verified > 0,
        "gpu path should verify at least one domain: {:?}",
        gpu.result
    );
}
