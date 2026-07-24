// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! External public API smoke test for block-wise CROWN result exports (#3961).
//!
//! This test compiles as a downstream consumer, proving the public
//! `propagate_crown_block_wise()` return type is nameable from both the crate
//! root and the `network` module surface.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use ny_propagate::layers::{LinearLayer, ReLULayer};
use ny_propagate::network::BlockSpec as NetworkBlockSpec;
use ny_propagate::network::BlockSpecEntry as NetworkBlockSpecEntry;
use ny_propagate::{
    network::BlockWiseCrownResult as NetworkBlockWiseCrownResult, BlockSpec, BlockSpecEntry,
    BlockWiseCrownResult, GraphNetwork, GraphNode, Layer,
};
use ny_tensor::BoundedTensor;

fn build_single_block_graph(hidden: usize) -> GraphNetwork {
    let linear1 = LinearLayer::new(
        Array2::from_shape_fn(
            (hidden, hidden),
            |(i, j)| if i == j { 0.75_f32 } else { 0.05 },
        ),
        Some(Array1::zeros(hidden)),
    )
    .expect("first Linear layer should be valid");
    let linear2 = LinearLayer::new(
        Array2::from_shape_fn(
            (hidden, hidden),
            |(i, j)| if i == j { 0.5_f32 } else { -0.02 },
        ),
        Some(Array1::zeros(hidden)),
    )
    .expect("second Linear layer should be valid");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input(
        "layer0_linear1",
        Layer::Linear(linear1),
    ));
    graph.add_node(GraphNode::new(
        "layer0_relu",
        Layer::ReLU(ReLULayer::new()),
        vec!["layer0_linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "layer0_linear2",
        Layer::Linear(linear2),
        vec!["layer0_relu".to_string()],
    ));
    graph.set_output("layer0_linear2");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_crown_result_is_publicly_nameable_3961() {
    let graph = build_single_block_graph(4);
    let epsilon = 0.05_f32;
    let input =
        BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), epsilon).expect("input valid");

    let result: BlockWiseCrownResult = graph
        .propagate_crown_block_wise(&input, epsilon)
        .expect("public block-wise CROWN API should run on a simple graph");
    let network_result: NetworkBlockWiseCrownResult = result;

    assert_eq!(
        network_result.total_blocks, 1,
        "#3961 expected one detected block from layer0_* node names"
    );
    assert_eq!(
        network_result.blocks.len(),
        1,
        "#3961 public result should expose per-block comparisons"
    );
    assert_eq!(
        network_result.blocks[0].block_name, "layer0",
        "#3961 block-wise public API should preserve block metadata"
    );
}

/// #4024: `BlockSpec` and `BlockSpecEntry` are nameable from both the crate
/// root (`ny_propagate::BlockSpec`) and the network module
/// (`ny_propagate::network::BlockSpec`), and the explicit-block API works
/// from a downstream consumer.
#[ntest::timeout(10000)]
#[test]
fn test_block_spec_is_publicly_nameable_and_callable_4024() {
    let graph = build_single_block_graph(4);
    let epsilon = 0.05_f32;
    let input =
        BoundedTensor::from_epsilon(ArrayD::zeros(IxDyn(&[4])), epsilon).expect("input valid");

    // Prove both import paths compile.
    let _root_spec: BlockSpec;
    let _root_entry: BlockSpecEntry;
    let _net_spec: NetworkBlockSpec;
    let _net_entry: NetworkBlockSpecEntry;

    // Build an explicit BlockSpec matching the single-block graph.
    let spec = BlockSpec {
        blocks: vec![BlockSpecEntry {
            block_index: 0,
            block_name: "layer0".to_string(),
            node_names: vec![
                "layer0_linear1".to_string(),
                "layer0_relu".to_string(),
                "layer0_linear2".to_string(),
            ],
        }],
    };

    let result: BlockWiseCrownResult = graph
        .propagate_crown_with_blocks(&input, epsilon, &spec)
        .expect("#4024 explicit-block API should run on a simple graph");

    assert_eq!(result.total_blocks, 1, "#4024 expected one explicit block");
    assert_eq!(
        result.blocks[0].block_name, "layer0",
        "#4024 explicit block metadata should survive into result"
    );
    assert_eq!(result.blocks[0].block_index, 0);
}
