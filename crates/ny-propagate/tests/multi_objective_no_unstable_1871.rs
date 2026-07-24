// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use ndarray::arr1;
use ny_propagate::layers::ReLULayer;
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, GraphNetwork, GraphNode, Layer,
};
use ny_tensor::BoundedTensor;

#[ntest::timeout(10000)]
#[test]
fn test_multi_objective_no_unstable_unverified_returns_unknown_1871() {
    // One unstable ReLU at root. After branching:
    // - inactive child becomes verified for objective -y > -0.5,
    // - active child is fully-constrained but still not verified.
    // The verifier must return Unknown, not Verified.
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::new(
        "relu0",
        Layer::ReLU(ReLULayer),
        vec!["_input".to_string()],
    ));
    graph.set_output("relu0");

    let input = BoundedTensor::new(arr1(&[-1.0_f32]).into_dyn(), arr1(&[1.0_f32]).into_dyn())
        .expect("valid input interval");

    let objectives = vec![vec![-1.0_f32]];
    let thresholds = vec![-0.5_f32];

    let config = BetaCrownConfig {
        timeout: Duration::from_secs(5),
        max_domains: 64,
        max_depth: 8,
        batch_size: 1,
        ..Default::default()
    };

    let result = BetaCrownVerifier::new(config)
        .verify_graph_relu_split_multi_objective(&graph, &input, &objectives, &thresholds)
        .expect("verification should complete");

    assert!(
        matches!(result.result, BabVerificationStatus::Unknown { .. }),
        "expected Unknown when a fully-constrained domain remains unresolved, got {:?}",
        result.result
    );
    assert_eq!(
        result.domains_verified, 1,
        "inactive branch should verify while active no-unstable branch remains unresolved"
    );
}
