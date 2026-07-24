// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{DataType, LayerSpec, Network, OnnxModel, TensorSpec, WeightStore};
use ndarray::arr1;
use ny_core::LayerType;
use ny_tensor::BoundedTensor;
use std::collections::HashMap;

fn build_branching_add_model() -> OnnxModel {
    let relu_a = LayerSpec {
        name: "relu_a".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_a_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let relu_b = LayerSpec {
        name: "relu_b".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_b_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["relu_a_out".to_string(), "relu_b_out".to_string()],
        outputs: vec!["add_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "quantize_branching_add".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "add_out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu_a, relu_b, add],
        param_count: 0,
    };

    OnnxModel::empty_with_network(network, WeightStore::new())
}

fn branching_add_config() -> QuantizeConfig {
    QuantizeConfig {
        epsilon: 0.01,
        continue_after_overflow: true,
        input: Some(
            BoundedTensor::new(arr1(&[1.0_f32]).into_dyn(), arr1(&[2.0_f32]).into_dyn())
                .expect("bounded input"),
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_analyze_quantization_graph_tracks_binary_add_bounds() {
    let model = build_branching_add_model();
    let graph = model.to_graph_network().expect("graph conversion");
    let config = branching_add_config();

    let result =
        analyze_quantization_graph(&graph, &config, &[1]).expect("graph quantization succeeds");

    assert_eq!(
        result
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect::<Vec<_>>(),
        vec!["relu_a", "relu_b", "add"]
    );

    let add = result
        .layers
        .iter()
        .find(|layer| layer.name == "add")
        .expect("add layer present");
    assert!(!add.propagation_failed);
    assert!((add.min_bound - 2.0).abs() < 1e-6);
    assert!((add.max_bound - 4.0).abs() < 1e-6);
}

#[ntest::timeout(10000)]
#[test]
fn test_analyze_quantization_model_routes_binary_graph_to_graph_path() {
    let model = build_branching_add_model();
    let graph = model.to_graph_network().expect("graph conversion");
    let config = branching_add_config();

    assert!(
        graph
            .node_names()
            .iter()
            .filter_map(|name| graph.node(name))
            .any(|node| node.layer().is_binary()),
        "expected a binary node to trigger graph routing"
    );

    let routed = analyze_quantization_model(&model, &config).expect("model quantization succeeds");
    let direct_graph =
        analyze_quantization_graph(&graph, &config, &[1]).expect("graph quantization succeeds");

    assert_eq!(
        routed
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect::<Vec<_>>(),
        direct_graph
            .layers
            .iter()
            .map(|layer| layer.name.as_str())
            .collect::<Vec<_>>()
    );

    let routed_add = routed
        .layers
        .iter()
        .find(|layer| layer.name == "add")
        .expect("routed add layer present");
    let direct_add = direct_graph
        .layers
        .iter()
        .find(|layer| layer.name == "add")
        .expect("direct add layer present");

    assert!(!routed_add.propagation_failed);
    assert!((routed_add.min_bound - direct_add.min_bound).abs() < 1e-6);
    assert!((routed_add.max_bound - direct_add.max_bound).abs() < 1e-6);
    assert!((routed_add.max_bound - 4.0).abs() < 1e-6);
}
