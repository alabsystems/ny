// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Direct proof-coverage tests for batched/block-wise CROWN guard surfaces (#4280).

use crate::*;
use ndarray::{arr1, ArrayD, IxDyn};
use ny_core::NyError;

fn scalar_interval(lower: f32, upper: f32) -> BoundedTensor {
    BoundedTensor::new(arr1(&[lower]).into_dyn(), arr1(&[upper]).into_dyn())
        .expect("scalar interval should construct")
}

fn resize_shape_mismatch_input_4146() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![-0.5_f32, -0.25, 0.0, 0.25]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![0.75_f32, 0.5, 0.4, 0.9]).unwrap(),
    )
    .unwrap()
}

fn make_block_ibp_graph_4280() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu1", Layer::ReLU(ReLULayer)));
    graph.add_node(GraphNode::new(
        "bias1",
        Layer::AddConstant(AddConstantLayer::new(ArrayD::from_elem(
            IxDyn(&[]),
            1.0_f32,
        ))),
        vec!["relu1".to_string()],
    ));
    graph.add_node(GraphNode::binary(
        "sum",
        Layer::Add(AddLayer),
        "bias1",
        NETWORK_INPUT,
    ));
    graph.set_output("sum");
    graph
}

fn build_resize_layernorm_pending_path_graph_4280(resize: ResizeLayer) -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    // Names are ordered so reverse topological traversal processes the failing
    // resize -> layernorm branch while the tile branch is still pending.
    graph.add_node(GraphNode::from_input(
        "a_tile_rows",
        Layer::Tile(TileLayer::new(1, 2)),
    ));
    graph.add_node(GraphNode::new(
        "b_tile_cols",
        Layer::Tile(TileLayer::new(2, 2)),
        vec!["a_tile_rows".into()],
    ));
    graph.add_node(GraphNode::from_input("y_resize", Layer::Resize(resize)));
    graph.add_node(GraphNode::new(
        "z_resize_ln",
        Layer::LayerNorm(LayerNormLayer::new_default(4, 1e-5).unwrap()),
        vec!["y_resize".into()],
    ));
    graph.add_node(GraphNode::binary(
        "zz_sum",
        Layer::Add(AddLayer),
        "z_resize_ln",
        "b_tile_cols",
    ));
    graph.set_output("zz_sum");
    graph
}

fn build_block_wise_mul_graph_4280() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::binary(
        "mul",
        Layer::MulBinary(MulBinaryLayer),
        NETWORK_INPUT,
        NETWORK_INPUT,
    ));
    graph.set_output("mul");
    graph
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_block_ibp_bounds_handles_add_and_network_input_4280() {
    let graph = make_block_ibp_graph_4280();
    let block_nodes = vec!["relu1".to_string(), "bias1".to_string(), "sum".to_string()];
    let block_input = scalar_interval(-1.0, 1.0);

    let bounds = graph
        .collect_block_ibp_bounds(&block_nodes, &block_input)
        .expect("block IBP should succeed");

    assert_eq!(bounds.len(), 3, "all block nodes should have cached bounds");
    assert_eq!(bounds["relu1"].lower()[[0]], 0.0);
    assert_eq!(bounds["relu1"].upper()[[0]], 1.0);
    assert_eq!(bounds["bias1"].lower()[[0]], 1.0);
    assert_eq!(bounds["bias1"].upper()[[0]], 2.0);
    assert_eq!(
        bounds["sum"].lower()[[0]],
        0.0,
        "binary Add should combine in-block and NETWORK_INPUT bounds"
    );
    assert_eq!(
        bounds["sum"].upper()[[0]],
        3.0,
        "outside-block inputs should resolve to block_input for binary Add"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_collect_block_ibp_bounds_missing_node_returns_error_4280() {
    let graph = make_block_ibp_graph_4280();
    let block_input = scalar_interval(-1.0, 1.0);
    let error = graph
        .collect_block_ibp_bounds(&["ghost".to_string()], &block_input)
        .expect_err("missing block node should fail");

    assert!(
        matches!(&error, NyError::InvalidSpec(message) if message.contains("Node not found: ghost")),
        "expected InvalidSpec for missing block node, got {error:?}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_batched_crown_resize_pending_paths_propagate_error_4280() {
    tests::with_crown_dense_budget_mb("2048", || {
        let graph = build_resize_layernorm_pending_path_graph_4280(ResizeLayer::new(2, 2));
        let input = resize_shape_mismatch_input_4146();

        let error = graph
            .propagate_crown_batched(&input)
            .expect_err("pending-path ShapeMismatch must propagate instead of partial fallback");

        assert!(
            matches!(error, NyError::ShapeMismatch { .. }),
            "expected ShapeMismatch from pending-path guard, got {error:?}"
        );
    });
}

#[ntest::timeout(10000)]
#[test]
fn test_block_wise_crown_mulbinary_fallback_matches_ibp_bounds_4280() {
    let graph = build_block_wise_mul_graph_4280();
    let block_nodes = vec!["mul".to_string()];
    let block_input = scalar_interval(-1.0, 1.0);

    let block_node_bounds = graph
        .collect_block_ibp_bounds(&block_nodes, &block_input)
        .expect("block IBP should succeed");
    let (bounds, _stats, provenance) = graph
        .crown_backward_within_block(&block_nodes, &block_node_bounds, &block_input)
        .expect("block-wise CROWN should degrade through partial fallback");

    assert_eq!(
        provenance,
        BoundsProvenance::Crown,
        "bias-only fallback should still surface Crown provenance after exact concretization"
    );
    assert_eq!(
        bounds.lower(),
        block_node_bounds["mul"].lower(),
        "block-wise fallback bounds should match the cached IBP lower bound"
    );
    assert_eq!(
        bounds.upper(),
        block_node_bounds["mul"].upper(),
        "block-wise fallback bounds should match the cached IBP upper bound"
    );
}
