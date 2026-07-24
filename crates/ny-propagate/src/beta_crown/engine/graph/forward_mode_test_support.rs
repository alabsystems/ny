// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ndarray::{arr1, arr2, ArrayD, IxDyn};
use ny_core::Result;
use ny_tensor::BoundedTensor;

use crate::layers::{ConcatLayer, GatherLayer, LinearLayer, ReLULayer, SliceLayer};
use crate::network::{GraphNode, SpecCrownRequest};
use crate::{GraphNetwork, Layer};

fn add_forward_mode_relu_prefix_4354(graph: &mut GraphNetwork) {
    let linear = LinearLayer::new(
        arr2(&[[1.0_f32, -0.75], [-0.5_f32, 1.5], [0.8_f32, 0.6]]),
        Some(arr1(&[0.1_f32, -0.2, 0.05])),
    )
    .expect("fixture linear should be valid");
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    // Second affine layer: forward-linear preserves input correlations through
    // the composition, while IBP loses them at the linear→linear2 boundary.
    // This ensures forward-linear intermediate bounds are strictly tighter than
    // IBP at the relu input, making forward+crown CROWN output distinguishable.
    let linear2 = LinearLayer::new(
        arr2(&[
            [0.6_f32, -0.3, 0.5],
            [0.4_f32, 0.7, -0.2],
            [-0.5_f32, 0.1, 0.9],
        ]),
        Some(arr1(&[-0.1_f32, 0.15, 0.0])),
    )
    .expect("fixture linear2 should be valid");
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["linear".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["linear2".to_string()],
    ));
}

fn add_forward_mode_concat_path_4354(graph: &mut GraphNetwork) {
    let gather = GatherLayer::new(
        0,
        Some(
            ArrayD::from_shape_vec(IxDyn(&[3]), vec![2_i64, 0_i64, 1_i64])
                .expect("fixture gather indices should shape"),
        ),
        vec![3],
    );
    graph.add_node(GraphNode::new(
        "gather",
        Layer::Gather(gather),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "slice_head",
        Layer::Slice(SliceLayer::new(0, 0, 2)),
        vec!["gather".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "slice_tail",
        Layer::Slice(SliceLayer::new(0, 1, 2)),
        vec!["relu".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "concat",
        Layer::Concat(ConcatLayer::new(0)),
        vec!["slice_head".to_string(), "slice_tail".to_string()],
    ));
}

fn add_forward_mode_output_4354(graph: &mut GraphNetwork) {
    let output = LinearLayer::new(
        arr2(&[[0.5_f32, -1.25, 1.0], [-0.3_f32, 0.8, 0.4]]),
        Some(arr1(&[0.2_f32, -0.1])),
    )
    .expect("fixture output linear should be valid");
    graph.add_node(GraphNode::new(
        "out",
        Layer::Linear(output),
        vec!["concat".to_string()],
    ));
    graph.set_output("out");
}

fn forward_mode_input_bounds_4354() -> BoundedTensor {
    BoundedTensor::new(
        arr1(&[-1.0_f32, -0.25]).into_dyn(),
        arr1(&[1.5_f32, 0.75]).into_dyn(),
    )
    .expect("fixture input bounds should be valid")
}

pub(crate) fn build_forward_mode_graph_fixture_4354() -> (GraphNetwork, BoundedTensor) {
    let mut graph = GraphNetwork::new();
    add_forward_mode_relu_prefix_4354(&mut graph);
    add_forward_mode_concat_path_4354(&mut graph);
    add_forward_mode_output_4354(&mut graph);
    (graph, forward_mode_input_bounds_4354())
}

pub(crate) fn expected_forward_root_output_4354(
    graph: &GraphNetwork,
    input: &BoundedTensor,
) -> Result<BoundedTensor> {
    let forward_node_bounds = graph.collect_forward_linear_bounds_dag_with_engine(input, None)?;
    let output_bounds = forward_node_bounds
        .get("out")
        .expect("forward bootstrap should include the output node");
    let identity_spec = ndarray::Array2::<f32>::eye(output_bounds.len());
    SpecCrownRequest::new(graph, input, &identity_spec, None)
        .node_bounds(&forward_node_bounds)
        .run()?
        .reshape(output_bounds.shape())
}

/// CROWN backward with plain IBP intermediates — matches behavior of large
/// nn4sys-style graphs where CROWN-IBP per-node intermediates are too expensive.
/// Forward+crown should produce tighter output than this baseline.
pub(crate) fn plain_graph_crown_output_4354(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    _crown_backward_layers: Option<usize>,
) -> Result<BoundedTensor> {
    let ibp_node_bounds = graph.collect_node_bounds_with_engine(input, None)?;
    let output_bounds = ibp_node_bounds
        .get(graph.output_name())
        .expect("IBP should include output node");
    let identity_spec = ndarray::Array2::<f32>::eye(output_bounds.len());
    SpecCrownRequest::new(graph, input, &identity_spec, None)
        .node_bounds(&ibp_node_bounds)
        .run()?
        .reshape(output_bounds.shape())
}

pub(crate) fn assert_bounds_close_4354(
    actual: &BoundedTensor,
    expected: &BoundedTensor,
    label: &str,
) {
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{label}: shape mismatch {:?} vs {:?}",
        actual.shape(),
        expected.shape()
    );
    for (idx, (actual, expected)) in actual
        .lower()
        .iter()
        .zip(expected.lower().iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "{label}: lower mismatch at idx {idx}: actual={actual}, expected={expected}"
        );
    }
    for (idx, (actual, expected)) in actual
        .upper()
        .iter()
        .zip(expected.upper().iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "{label}: upper mismatch at idx {idx}: actual={actual}, expected={expected}"
        );
    }
}
