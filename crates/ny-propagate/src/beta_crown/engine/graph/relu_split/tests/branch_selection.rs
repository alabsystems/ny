// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_tensor::BoundedTensor;

use crate::beta_crown::domain::GraphBabDomain;
use crate::beta_crown::engine::domain_results::GraphDomainResult;
use crate::beta_crown::BetaCrownVerifier;
use crate::GraphNetwork;

/// Regression test for #2038: branch-selection failures in the ReLU-split
/// loop must become PropagationFailure (domain unresolved), not Err aborts.
#[ntest::timeout(5000)]
#[test]
fn test_select_graph_branch_failure_maps_to_propagation_failure_2038() {
    let verifier = BetaCrownVerifier::new(crate::beta_crown::BetaCrownConfig::default());

    let mut graph = GraphNetwork::new();
    graph.add_node(crate::GraphNode::from_input(
        "relu",
        crate::Layer::ReLU(crate::ReLULayer),
    ));
    graph.add_node(crate::GraphNode::new(
        "linear1",
        crate::Layer::Linear(
            crate::LinearLayer::new(ndarray::arr2(&[[1.0, 1.0]]), None)
                .expect("invariant: 2x1 weight matrix is valid"),
        ),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear1");

    let input = BoundedTensor::new(
        ndarray::arr1(&[-1.0f32, -1.0]).into_dyn(),
        ndarray::arr1(&[1.0f32, 1.0]).into_dyn(),
    )
    .expect("invariant: symmetric bounds are valid");
    let domain = GraphBabDomain::root(std::collections::HashMap::new(), -1.0, 1.0, &input, false)
        .expect("invariant: finite test bounds are valid");

    let empty_unstable: Vec<(String, usize)> = vec![];
    let result = verifier.select_graph_branch_or_propagation_failure_in_relu_split(
        &graph,
        &domain,
        &empty_unstable,
    );

    assert!(
        matches!(result, Err(GraphDomainResult::PropagationFailure)),
        "select_graph_branch failure must map to PropagationFailure, got {result:?}"
    );
}
