// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_onnx::{
    AttributeValue, DataType, GraphNetworkOptions, LayerSpec, Network, OnnxModel, TensorSpec,
    WeightStore,
};
use ny_propagate::Layer;
use std::collections::HashMap;

#[test]
fn split_missing_split_negative_axis_infers_sizes() {
    let split = LayerSpec {
        name: "split".to_string(),
        layer_type: ny_core::LayerType::Slice,
        inputs: vec!["input".to_string()],
        outputs: vec![
            "split_out_0".to_string(),
            "split_out_1".to_string(),
            "split_out_2".to_string(),
        ],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(-1))]),
    };

    let network = Network {
        name: "split_negative_axis".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![-1, -1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "split_out_2".to_string(),
            shape: vec![2, 2],
            dtype: DataType::Float32,
        }],
        layers: vec![split],
        param_count: 0,
    };

    let model = OnnxModel::empty_with_network(network, WeightStore::new())
        .with_tensor_shapes(HashMap::from([("input".to_string(), vec![2, 6])]));

    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph conversion succeeds");

    let slice_2 = graph.node("split_slice_2").expect("slice 2 node exists");
    match slice_2.layer() {
        Layer::Slice(layer) => {
            assert_eq!(layer.axis, -1);
            assert_eq!(layer.start, 4);
            assert_eq!(layer.end, 6);
        }
        _ => panic!("expected Slice layer for split_slice_2"),
    }
}
