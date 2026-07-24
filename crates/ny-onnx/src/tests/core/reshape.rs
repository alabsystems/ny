// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::arr1;
use ny_core::{LayerType, NyError, Result};
use std::collections::HashMap;

#[ntest::timeout(10000)]
#[test]
fn test_to_propagate_network_dynamic_reshape_errors() {
    let model = OnnxModel {
        network: Network {
            name: "dynamic-reshape".to_string(),
            inputs: vec![
                TensorSpec {
                    name: "input".to_string(),
                    shape: vec![4],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "shape".to_string(),
                    shape: vec![1],
                    dtype: DataType::Int64,
                },
            ],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "reshape".to_string(),
                layer_type: LayerType::Reshape,
                inputs: vec!["input".to_string(), "shape".to_string()],
                outputs: vec!["output".to_string()],
                weights: None,
                attributes: HashMap::new(),
            }],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    };

    assert_dynamic_reshape_error(model.to_propagate_network(), "reshape");
}

#[ntest::timeout(10000)]
#[test]
fn test_to_propagate_network_dynamic_reshape_permissive_skips() {
    let model = OnnxModel {
        network: Network {
            name: "dynamic-reshape-permissive".to_string(),
            inputs: vec![
                TensorSpec {
                    name: "input".to_string(),
                    shape: vec![4],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "shape".to_string(),
                    shape: vec![1],
                    dtype: DataType::Int64,
                },
            ],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "reshape".to_string(),
                layer_type: LayerType::Reshape,
                inputs: vec!["input".to_string(), "shape".to_string()],
                outputs: vec!["output".to_string()],
                weights: None,
                attributes: HashMap::new(),
            }],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    };

    let network = model
        .to_propagate_network_with_options(PropagateNetworkOptions::permissive())
        .expect("permissive conversion should skip dynamic reshape");
    assert!(network.layers().is_empty());
}

#[ntest::timeout(10000)]
#[test]
fn test_to_propagate_network_dynamic_reshape_skips_only_reshape() {
    let model = OnnxModel {
        network: Network {
            name: "dynamic-reshape-middle".to_string(),
            inputs: vec![
                TensorSpec {
                    name: "input".to_string(),
                    shape: vec![4],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "shape".to_string(),
                    shape: vec![1],
                    dtype: DataType::Int64,
                },
            ],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
            }],
            layers: vec![
                LayerSpec {
                    name: "relu1".to_string(),
                    layer_type: LayerType::ReLU,
                    inputs: vec!["input".to_string()],
                    outputs: vec!["relu1_out".to_string()],
                    weights: None,
                    attributes: HashMap::new(),
                },
                LayerSpec {
                    name: "reshape".to_string(),
                    layer_type: LayerType::Reshape,
                    inputs: vec!["relu1_out".to_string(), "shape".to_string()],
                    outputs: vec!["reshaped".to_string()],
                    weights: None,
                    attributes: HashMap::new(),
                },
                LayerSpec {
                    name: "relu2".to_string(),
                    layer_type: LayerType::ReLU,
                    inputs: vec!["reshaped".to_string()],
                    outputs: vec!["output".to_string()],
                    weights: None,
                    attributes: HashMap::new(),
                },
            ],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    };

    let network = model
        .to_propagate_network_with_options(PropagateNetworkOptions::permissive())
        .expect("permissive conversion should skip dynamic reshape");
    assert_eq!(network.layers().len(), 2);
    assert_eq!(network.layers()[0].layer_type(), "ReLU");
    assert_eq!(network.layers()[1].layer_type(), "ReLU");
}

#[ntest::timeout(10000)]
#[test]
fn test_to_propagate_network_dynamic_reshape_errors_with_neighbors() {
    let tensor_producer = dynamic_reshape_producer_map();

    let model = OnnxModel {
        network: Network {
            name: "dynamic-reshape-middle-errors".to_string(),
            inputs: vec![
                TensorSpec {
                    name: "input".to_string(),
                    shape: vec![4],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "shape".to_string(),
                    shape: vec![1],
                    dtype: DataType::Int64,
                },
            ],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
            }],
            layers: vec![
                LayerSpec {
                    name: "relu1".to_string(),
                    layer_type: LayerType::ReLU,
                    inputs: vec!["input".to_string()],
                    outputs: vec!["relu1_out".to_string()],
                    weights: None,
                    attributes: HashMap::new(),
                },
                LayerSpec {
                    name: "reshape".to_string(),
                    layer_type: LayerType::Reshape,
                    inputs: vec!["relu1_out".to_string(), "shape".to_string()],
                    outputs: vec!["reshaped".to_string()],
                    weights: None,
                    attributes: HashMap::new(),
                },
                LayerSpec {
                    name: "relu2".to_string(),
                    layer_type: LayerType::ReLU,
                    inputs: vec!["reshaped".to_string()],
                    outputs: vec!["output".to_string()],
                    weights: None,
                    attributes: HashMap::new(),
                },
            ],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer,
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    };

    assert_dynamic_reshape_error(model.to_propagate_network(), "reshape");
}

fn dynamic_reshape_producer_map() -> HashMap<String, String> {
    // Precompute producer map so the test model can be immutable.
    let mut tensor_producer = HashMap::new();
    tensor_producer.insert("relu1_out".to_string(), "relu1".to_string());
    tensor_producer.insert("reshaped".to_string(), "reshape".to_string());
    tensor_producer.insert("output".to_string(), "relu2".to_string());
    tensor_producer
}

fn constant_reshape_model(name: &str, shape_name: &str, shape: &[f32]) -> OnnxModel {
    let mut weights = WeightStore::new();
    weights.insert(shape_name.to_string(), arr1(shape).into_dyn());

    OnnxModel {
        network: Network {
            name: name.to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "reshape".to_string(),
                layer_type: LayerType::Reshape,
                inputs: vec!["input".to_string(), shape_name.to_string()],
                outputs: vec!["output".to_string()],
                weights: None,
                attributes: HashMap::new(),
            }],
            param_count: 0,
        },
        weights,
        tensor_producer: HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    }
}

fn assert_invalid_reshape_error<T: std::fmt::Debug>(
    result: Result<T>,
    layer_name: &str,
    detail: &str,
    expected_value: &str,
) {
    match result {
        Err(NyError::InvalidSpec(msg)) => {
            assert!(msg.contains(detail), "unexpected error message: {msg}");
            assert!(
                msg.contains(expected_value),
                "missing expected detail {expected_value}: {msg}"
            );
            assert!(
                msg.contains(layer_name),
                "missing layer name {layer_name}: {msg}"
            );
        }
        other => panic!("expected invalid reshape InvalidSpec, got {:?}", other),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_dynamic_reshape_errors() {
    let model = OnnxModel {
        network: Network {
            name: "dynamic-reshape-graph".to_string(),
            inputs: vec![
                TensorSpec {
                    name: "input".to_string(),
                    shape: vec![4],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "shape".to_string(),
                    shape: vec![1],
                    dtype: DataType::Int64,
                },
            ],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "reshape".to_string(),
                layer_type: LayerType::Reshape,
                inputs: vec!["input".to_string(), "shape".to_string()],
                outputs: vec!["output".to_string()],
                weights: None,
                attributes: HashMap::new(),
            }],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    };

    match model.to_graph_network() {
        Err(NyError::UnsupportedOp(msg)) => {
            assert!(
                msg.contains("dynamic shape"),
                "unexpected error message: {}",
                msg
            );
            assert!(
                msg.contains("GraphNetworkOptions::permissive"),
                "missing permissive hint: {}",
                msg
            );
        }
        other => panic!("expected dynamic reshape UnsupportedOp, got {:?}", other),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_to_propagate_network_invalid_negative_reshape_errors_2587() {
    let model = constant_reshape_model("invalid-negative-reshape", "shape", &[-2.0, 2.0]);
    assert_invalid_reshape_error(
        model.to_propagate_network(),
        "reshape",
        "invalid negative dimension",
        "-2",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_to_propagate_network_invalid_negative_reshape_permissive_still_errors_2587() {
    let model = constant_reshape_model("invalid-negative-reshape", "shape", &[-2.0, 2.0]);
    assert_invalid_reshape_error(
        model.to_propagate_network_with_options(PropagateNetworkOptions::permissive()),
        "reshape",
        "invalid negative dimension",
        "-2",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_invalid_negative_reshape_errors_2587() {
    let model = constant_reshape_model("invalid-negative-reshape-graph", "shape", &[-2.0, 2.0]);
    assert_invalid_reshape_error(
        model.to_graph_network(),
        "reshape",
        "invalid negative dimension",
        "-2",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_invalid_negative_reshape_permissive_still_errors_2587() {
    let model = constant_reshape_model("invalid-negative-reshape-graph", "shape", &[-2.0, 2.0]);
    assert_invalid_reshape_error(
        model.to_graph_network_with_options(GraphNetworkOptions::permissive()),
        "reshape",
        "invalid negative dimension",
        "-2",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_to_propagate_network_multiple_infer_reshape_errors_2587() {
    let model = constant_reshape_model("duplicate-infer-reshape", "shape", &[-1.0, -1.0, 2.0]);
    assert_invalid_reshape_error(
        model.to_propagate_network(),
        "reshape",
        "multiple inferred dimensions",
        "-1",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_multiple_infer_reshape_errors_2587() {
    let model =
        constant_reshape_model("duplicate-infer-reshape-graph", "shape", &[-1.0, -1.0, 2.0]);
    assert_invalid_reshape_error(
        model.to_graph_network(),
        "reshape",
        "multiple inferred dimensions",
        "-1",
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_dynamic_reshape_permissive_skips() {
    let model = OnnxModel {
        network: Network {
            name: "dynamic-reshape-graph-permissive".to_string(),
            inputs: vec![
                TensorSpec {
                    name: "input".to_string(),
                    shape: vec![4],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "shape".to_string(),
                    shape: vec![1],
                    dtype: DataType::Int64,
                },
            ],
            outputs: vec![TensorSpec {
                name: "output".to_string(),
                shape: vec![4],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "reshape".to_string(),
                layer_type: LayerType::Reshape,
                inputs: vec!["input".to_string(), "shape".to_string()],
                outputs: vec!["output".to_string()],
                weights: None,
                attributes: HashMap::new(),
            }],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    };

    let graph = model
        .to_graph_network_with_options(GraphNetworkOptions::permissive())
        .expect("permissive conversion should skip dynamic reshape");
    assert_eq!(graph.num_nodes(), 1);
    let skip_name = "reshape__skip";
    let node = graph
        .node(skip_name)
        .expect("expected OpaqueSkip node for dynamic reshape with multiple activation inputs");
    assert!(matches!(node.layer(), ny_propagate::Layer::OpaqueSkip(_)));
}
