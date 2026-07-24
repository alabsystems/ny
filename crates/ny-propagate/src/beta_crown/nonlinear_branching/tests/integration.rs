// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2};
use ny_tensor::BoundedTensor;

use super::super::{NonlinearBranching, NonlinearBranchingConfig};
use crate::layers::{GELULayer, Layer, LinearLayer};
use crate::network::{GraphNetwork, GraphNode};

#[ntest::timeout(5000)]
#[test]
fn test_get_decisions_gelu_network() {
    let mut graph = GraphNetwork::new();

    let weights = arr2(&[[1.0_f32, -0.5], [0.5, 1.0]]);
    let linear1 = LinearLayer::new(weights, Some(arr1(&[0.0, 0.1]))).unwrap();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "gelu",
        Layer::GELU(GELULayer::default()),
        vec!["linear1".to_string()],
    ));
    graph.set_output("gelu");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, -1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();
    let _output_bounds = graph.propagate_ibp(&input).unwrap();

    let mut node_bounds = std::collections::HashMap::new();
    node_bounds.insert(
        "gelu".to_string(),
        BoundedTensor::new(
            arr1(&[-1.5_f32, -0.4]).into_dyn(),
            arr1(&[1.5_f32, 1.2]).into_dyn(),
        )
        .unwrap(),
    );

    let branching = NonlinearBranching::new(NonlinearBranchingConfig {
        num_candidates: 2,
        ..Default::default()
    });
    let decisions = branching
        .decisions(&graph, &node_bounds, &["gelu".to_string()])
        .unwrap();

    assert_eq!(decisions.len(), 2);
    for decision in &decisions {
        assert_eq!(decision.points.len(), 1);
        let (lower, upper) = decision.original_bounds;
        let point = decision.points[0];
        assert!(point > lower && point < upper);
    }

    let splits = decisions[0].to_splits().expect("valid decision");
    assert_eq!(splits.len(), 2);
}
