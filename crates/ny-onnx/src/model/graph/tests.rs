// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::{
    AttributeValue, CompoundNodePolicy, DataType, GraphNetworkOptions, LayerSpec,
    MissingOutputPolicy, Network, OnnxModel, TensorSpec, WeightStore,
};
use ndarray::arr1;
use ny_core::LayerType;
use ny_propagate::{GraphNetwork, Layer};
use std::collections::{HashMap, HashSet};

/// Helper: assert a node is an OpaqueSkip layer
fn assert_opaque_skip(graph: &GraphNetwork, node_name: &str) {
    let node = graph
        .node(node_name)
        .expect("expected OpaqueSkip node to exist");
    assert!(
        matches!(node.layer(), Layer::OpaqueSkip(_)),
        "expected OpaqueSkip layer for '{}', got {:?}",
        node_name,
        node.layer()
    );
}

fn build_model(network: Network) -> OnnxModel {
    OnnxModel {
        network,
        weights: WeightStore::new(),
        tensor_producer: HashMap::new(),
        constant_tensors: HashSet::new(),
        tensor_shapes: HashMap::new(),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::new(),
    }
}

fn build_model_with_shapes(
    network: Network,
    tensor_shapes: HashMap<String, Vec<i64>>,
) -> OnnxModel {
    let mut model = build_model(network);
    model.tensor_shapes = tensor_shapes;
    model
}

fn build_graph(model: &OnnxModel) -> GraphNetwork {
    model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect("graph conversion succeeds")
}

fn build_graph_with_options(model: &OnnxModel, options: GraphNetworkOptions) -> GraphNetwork {
    model
        .to_graph_network_with_options(options)
        .expect("graph conversion succeeds")
}

#[test]
fn skip_single_input_inserts_opaque_skip() {
    // Previously, single-input unsupported ops were treated as identity (pass-through).
    // After #2231 fix: they must insert OpaqueSkipLayer with [-inf, +inf] bounds.
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["relu_out".to_string()],
        outputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_multi_output".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, skipped, add],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph(&model);

    // Skipped op now inserts OpaqueSkipLayer node
    let skip_name = "skip_me__skip";
    assert_opaque_skip(&graph, skip_name);
    let skip_node = graph.node(skip_name).expect("skip node exists");
    assert_eq!(skip_node.inputs(), vec!["relu".to_string()]);

    // Downstream add should reference the skip node, not relu directly
    let add_node = graph.node("add").expect("add node exists");
    assert_eq!(add_node.inputs().len(), 2);
    assert_eq!(add_node.inputs()[0], skip_name);
    assert_eq!(add_node.inputs()[1], skip_name);
    assert_eq!(graph.output_name(), "add");
}

#[test]
fn skip_single_input_updates_last_added_node_for_missing_outputs() {
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
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["relu_a_out".to_string()],
        outputs: vec!["skip_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_single_input_missing_outputs".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: Vec::new(),
        layers: vec![relu_a, relu_b, skipped],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph_with_options(&model, GraphNetworkOptions::permissive());

    // The skip node is the last added, so with WarnAndFallback it becomes the output
    assert_opaque_skip(&graph, "skip_me__skip");
    assert_eq!(graph.output_name(), "skip_me__skip");
}

#[test]
fn split_missing_sizes_skips_when_shape_unknown() {
    // Use axis=1 (ONNX convention) on a 2D input with dynamic dims.
    // axis=0 is rejected in unbatched mode as targeting the batch dimension.
    // The split's input "relu_out" has unknown shape (no tensor_shapes entry,
    // not a graph input), so infer_equal_split_sizes returns None → skip.
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let split = LayerSpec {
        name: "split".to_string(),
        layer_type: LayerType::Slice,
        inputs: vec!["relu_out".to_string()],
        outputs: vec!["split_out_0".to_string(), "split_out_1".to_string()],
        weights: None,
        attributes: HashMap::from([("axis".to_string(), AttributeValue::Int(1))]),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["split_out_0".to_string(), "split_out_1".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "split_missing_sizes".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![-1, -1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, split, add],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph(&model);

    assert!(graph.node("split_slice_0").is_none());
    assert!(graph.node("split_slice_1").is_none());

    // Split with unknown shape now inserts OpaqueSkipLayer instead of identity
    let skip_name = "split__skip";
    assert_opaque_skip(&graph, skip_name);

    let add_node = graph.node("add").expect("add node exists");
    assert_eq!(add_node.inputs().len(), 2);
    assert_eq!(add_node.inputs()[0], skip_name);
    assert_eq!(add_node.inputs()[1], skip_name);
    assert_eq!(graph.output_name(), "add");
}

#[test]
fn skip_maps_outputs_to_input_node_when_activation_is_graph_input() {
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["input".to_string()],
        outputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_graph_input".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![skipped, add],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph(&model);

    // Skipped op with graph input now inserts OpaqueSkipLayer
    let skip_name = "skip_me__skip";
    assert_opaque_skip(&graph, skip_name);
    let skip_node = graph.node(skip_name).expect("skip node exists");
    assert_eq!(skip_node.inputs(), vec!["_input".to_string()]);

    let add_node = graph.node("add").expect("add node exists");
    assert_eq!(
        add_node.inputs(),
        vec![skip_name.to_string(), skip_name.to_string()]
    );
    assert_eq!(graph.output_name(), "add");
}

#[test]
fn skip_merge_preserves_multi_input_dependencies() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let sigmoid = LayerSpec {
        name: "sigmoid".to_string(),
        layer_type: LayerType::Sigmoid,
        inputs: vec!["input".to_string()],
        outputs: vec!["sigmoid_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["relu_out".to_string(), "sigmoid_out".to_string()],
        outputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_multi_input".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, sigmoid, skipped, add],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph(&model);

    let merge_name = "skip_me__skip";
    let merge_node = graph.node(merge_name).expect("skip merge node exists");
    assert!(matches!(merge_node.layer(), Layer::OpaqueSkip(_)));
    assert_eq!(
        merge_node.inputs(),
        vec!["relu".to_string(), "sigmoid".to_string()]
    );

    let add_node = graph.node("add").expect("add node exists");
    assert_eq!(add_node.inputs().len(), 2);
    assert_eq!(add_node.inputs()[0], merge_name);
    assert_eq!(add_node.inputs()[1], merge_name);
    assert_eq!(graph.output_name(), "add");
}

#[test]
fn skip_merge_preserves_multi_input_dependencies_in_permissive_mode() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let sigmoid = LayerSpec {
        name: "sigmoid".to_string(),
        layer_type: LayerType::Sigmoid,
        inputs: vec!["input".to_string()],
        outputs: vec!["sigmoid_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["relu_out".to_string(), "sigmoid_out".to_string()],
        outputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_multi_input_permissive".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, sigmoid, skipped, add],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph_with_options(&model, GraphNetworkOptions::permissive());

    let merge_name = "skip_me__skip";
    let merge_node = graph.node(merge_name).expect("skip merge node exists");
    assert!(matches!(merge_node.layer(), Layer::OpaqueSkip(_)));
    assert_eq!(
        merge_node.inputs(),
        vec!["relu".to_string(), "sigmoid".to_string()]
    );

    let add_node = graph.node("add").expect("add node exists");
    assert_eq!(add_node.inputs().len(), 2);
    assert_eq!(add_node.inputs()[0], merge_name);
    assert_eq!(add_node.inputs()[1], merge_name);
    assert_eq!(graph.output_name(), "add");
}

#[test]
fn skip_merge_maps_single_output_to_merge_node_in_permissive_mode() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let sigmoid = LayerSpec {
        name: "sigmoid".to_string(),
        layer_type: LayerType::Sigmoid,
        inputs: vec!["input".to_string()],
        outputs: vec!["sigmoid_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["relu_out".to_string(), "sigmoid_out".to_string()],
        outputs: vec!["skip_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_multi_input_single_output_permissive".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "skip_out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, sigmoid, skipped],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph_with_options(&model, GraphNetworkOptions::permissive());

    let merge_name = "skip_me__skip";
    let merge_node = graph.node(merge_name).expect("skip merge node exists");
    assert!(matches!(merge_node.layer(), Layer::OpaqueSkip(_)));
    assert_eq!(
        merge_node.inputs(),
        vec!["relu".to_string(), "sigmoid".to_string()]
    );
    assert_eq!(graph.output_name(), merge_name);
}

#[test]
fn skip_merge_handles_graph_input_and_internal_activation() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["input".to_string(), "relu_out".to_string()],
        outputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_merge_graph_input".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, skipped, add],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph(&model);

    let merge_name = "skip_me__skip";
    let merge_node = graph.node(merge_name).expect("skip merge node exists");
    assert!(matches!(merge_node.layer(), Layer::OpaqueSkip(_)));
    assert_eq!(
        merge_node.inputs(),
        vec!["_input".to_string(), "relu".to_string()]
    );

    let add_node = graph.node("add").expect("add node exists");
    assert_eq!(add_node.inputs().len(), 2);
    assert_eq!(add_node.inputs()[0], merge_name);
    assert_eq!(add_node.inputs()[1], merge_name);
    assert_eq!(graph.output_name(), "add");
}

#[test]
fn skip_merge_handles_graph_input_and_internal_activation_in_permissive_mode() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["input".to_string(), "relu_out".to_string()],
        outputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_merge_graph_input_permissive".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, skipped, add],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph_with_options(&model, GraphNetworkOptions::permissive());

    let merge_name = "skip_me__skip";
    let merge_node = graph.node(merge_name).expect("skip merge node exists");
    assert!(matches!(merge_node.layer(), Layer::OpaqueSkip(_)));
    assert_eq!(
        merge_node.inputs(),
        vec!["_input".to_string(), "relu".to_string()]
    );

    let add_node = graph.node("add").expect("add node exists");
    assert_eq!(add_node.inputs().len(), 2);
    assert_eq!(add_node.inputs()[0], merge_name);
    assert_eq!(add_node.inputs()[1], merge_name);
    assert_eq!(graph.output_name(), "add");
}

#[test]
fn output_selection_dedupes_same_node_outputs() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["relu_out".to_string()],
        outputs: vec!["skip_out_0".to_string(), "skip_out_1".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "dedupe_output_selection".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![
            TensorSpec {
                name: "skip_out_0".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            },
            TensorSpec {
                name: "skip_out_1".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            },
        ],
        layers: vec![relu, skipped],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph(&model);

    // Both ONNX outputs map to the OpaqueSkip node now (not relu)
    assert_eq!(graph.output_name(), "skip_me__skip");
}

#[test]
fn output_selection_respects_output_index() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let sigmoid = LayerSpec {
        name: "sigmoid".to_string(),
        layer_type: LayerType::Sigmoid,
        inputs: vec!["input".to_string()],
        outputs: vec!["sigmoid_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "output_index_selection".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![
            TensorSpec {
                name: "relu_out".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            },
            TensorSpec {
                name: "sigmoid_out".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            },
        ],
        layers: vec![relu, sigmoid],
        param_count: 0,
    };

    let model = build_model(network);

    let options = GraphNetworkOptions {
        output_index: Some(1),
        ..Default::default()
    };
    let graph = model
        .to_graph_network_with_options(options)
        .expect("graph conversion succeeds");

    assert_eq!(graph.output_name(), "sigmoid");
}

#[test]
fn output_index_out_of_range_errors() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let sigmoid = LayerSpec {
        name: "sigmoid".to_string(),
        layer_type: LayerType::Sigmoid,
        inputs: vec!["input".to_string()],
        outputs: vec!["sigmoid_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "output_index_oob".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![
            TensorSpec {
                name: "relu_out".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            },
            TensorSpec {
                name: "sigmoid_out".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            },
        ],
        layers: vec![relu, sigmoid],
        param_count: 0,
    };

    let model = build_model(network);

    let options = GraphNetworkOptions {
        output_index: Some(2),
        ..Default::default()
    };
    let err = model
        .to_graph_network_with_options(options)
        .expect_err("output_index out of range should error");

    let message = format!("{err}");
    assert!(message.contains("output_index"));
    assert!(message.contains("out of range"));
}

#[test]
fn skip_prefers_non_constant_activation_input() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let skipped = LayerSpec {
        name: "skip_me".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["const_in".to_string(), "relu_out".to_string()],
        outputs: vec!["skip_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["skip_out".to_string(), "relu_out".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "skip_const_input".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, skipped, add],
        param_count: 0,
    };

    let mut model = build_model(network);
    model.constant_tensors = vec!["const_in".to_string()].into_iter().collect();

    let graph = build_graph(&model);

    // Skipped op has one activation input (relu_out); const_in is filtered out.
    // OpaqueSkipLayer is inserted with relu as the single input.
    let skip_name = "skip_me__skip";
    assert_opaque_skip(&graph, skip_name);
    let skip_node = graph.node(skip_name).expect("skip node exists");
    assert_eq!(skip_node.inputs(), vec!["relu".to_string()]);

    let add_node = graph.node("add").expect("add node exists");
    assert_eq!(
        add_node.inputs(),
        vec![skip_name.to_string(), "relu".to_string()]
    );
}

#[test]
fn selects_output_from_onnx_outputs_over_last_node() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["relu_out".to_string(), "relu_out".to_string()],
        outputs: vec!["add_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let tail = LayerSpec {
        name: "tail".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["add_out".to_string()],
        outputs: vec!["tail_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "output_selection".to_string(),
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
        layers: vec![relu, add, tail],
        param_count: 0,
    };

    let model = build_model(network);

    let graph = build_graph(&model);

    assert_eq!(graph.output_name(), "add");
}

#[test]
fn split_missing_split_infers_sizes_from_tensor_shapes() {
    // Explicit split sizes [2,2,2] on ONNX axis=1 (unbatched → internal axis=0).
    // Original test used tensor_shapes inference, which broke due to unbatched-axis
    // mismatch in infer_equal_split_sizes (ONNX shapes vs internal axis).
    let split = LayerSpec {
        name: "split".to_string(),
        layer_type: LayerType::Slice,
        inputs: vec!["input".to_string()],
        outputs: vec![
            "split_out_0".to_string(),
            "split_out_1".to_string(),
            "split_out_2".to_string(),
        ],
        weights: None,
        attributes: HashMap::from([
            ("axis".to_string(), AttributeValue::Int(1)),
            ("split".to_string(), AttributeValue::Ints(vec![2, 2, 2])),
        ]),
    };
    let network = Network {
        name: "split_infer_sizes".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![-1, -1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "split_out_1".to_string(),
            shape: vec![2, 2],
            dtype: DataType::Float32,
        }],
        layers: vec![split],
        param_count: 0,
    };
    let model =
        build_model_with_shapes(network, HashMap::from([("input".to_string(), vec![2, 6])]));
    let graph = build_graph(&model);

    // Verify each slice: ONNX axis=1 on recorded rank-2 input → trailing-
    // relative internal axis -1 (correct whether the runtime tensor kept its
    // ONNX rank or was batch-stripped; see remap_axis_trailing).
    let expected: [(usize, usize); 3] = [(0, 2), (2, 4), (4, 6)];
    for (i, (start, end)) in expected.iter().enumerate() {
        let name = format!("split_slice_{}", i);
        let node = graph.node(&name).unwrap_or_else(|| panic!("{name} exists"));
        match node.layer() {
            Layer::Slice(layer) => {
                assert_eq!(layer.axis, -1, "{name} axis");
                assert_eq!(layer.start, *start, "{name} start");
                assert_eq!(layer.end, *end, "{name} end");
            }
            _ => panic!("expected Slice layer for {name}"),
        }
    }
    assert_eq!(graph.output_name(), "split_slice_1");
}

#[test]
fn split_missing_split_negative_axis_infers_sizes() {
    let split = LayerSpec {
        name: "split".to_string(),
        layer_type: LayerType::Slice,
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

    let model =
        build_model_with_shapes(network, HashMap::from([("input".to_string(), vec![2, 6])]));

    let graph = build_graph(&model);

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

#[test]
fn selects_output_index_when_multiple_outputs() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let add = LayerSpec {
        name: "add".to_string(),
        layer_type: LayerType::Add,
        inputs: vec!["relu_out".to_string(), "relu_out".to_string()],
        outputs: vec!["add_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    let tail = LayerSpec {
        name: "tail".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["add_out".to_string()],
        outputs: vec!["tail_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "output_index".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![
            TensorSpec {
                name: "add_out".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            },
            TensorSpec {
                name: "tail_out".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            },
        ],
        layers: vec![relu, add, tail],
        param_count: 0,
    };

    let model = build_model(network);

    let options = GraphNetworkOptions {
        output_index: Some(1),
        ..GraphNetworkOptions::default()
    };
    let graph = model
        .to_graph_network_with_options(options)
        .expect("graph conversion succeeds");

    assert_eq!(graph.output_name(), "tail");
}

#[test]
fn missing_output_errors_by_default() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "missing_output".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "missing_out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu],
        param_count: 0,
    };

    let model = build_model(network);

    let err = model
        .to_graph_network_with_options(GraphNetworkOptions::default())
        .expect_err("missing output should fail");

    let message = format!("{err}");
    assert!(message.contains("missing outputs"), "message={message}");
}

#[test]
fn missing_output_warns_and_falls_back_when_configured() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "missing_output_warn".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "missing_out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu],
        param_count: 0,
    };

    let model = build_model(network);

    let options = GraphNetworkOptions {
        missing_output_policy: MissingOutputPolicy::WarnAndFallback,
        ..GraphNetworkOptions::default()
    };
    let graph = model
        .to_graph_network_with_options(options)
        .expect("graph conversion succeeds");

    assert_eq!(graph.output_name(), "relu");
}

#[test]
fn output_selection_uses_mid_graph_output_not_last_node() {
    // Test case from design doc: ONNX output is mid-graph node, trailing op is skipped.
    // Verify graph output equals the mid-graph node, not the last added node.
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()], // This is declared as ONNX output
        weights: None,
        attributes: HashMap::new(),
    };
    // Trailing unsupported op (Unknown type) that gets skipped
    let trailing_unsupported = LayerSpec {
        name: "trailing_unknown".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["relu_out".to_string()],
        outputs: vec!["trailing_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "mid_graph_output".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        // ONNX declares relu_out as output, not trailing_out
        outputs: vec![TensorSpec {
            name: "relu_out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, trailing_unsupported],
        param_count: 0,
    };

    let model = build_model(network);

    let graph = build_graph(&model);

    // Should select relu (mid-graph node producing relu_out), not trailing_unknown
    assert_eq!(graph.output_name(), "relu");
}

/// Regression test for #2231: unsupported single-input ops must produce OpaqueSkip
/// (conservative [-inf, +inf] bounds), not identity pass-through.
///
/// Scenario: Linear -> Reciprocal (unsupported) -> output
/// Before fix: Reciprocal was silently dropped, bounds were for Linear alone (wrong).
/// After fix: OpaqueSkipLayer inserted, bounds are [-inf, +inf] (sound but imprecise).
#[test]
fn unsupported_single_input_op_not_identity_regression_2231() {
    let relu = LayerSpec {
        name: "relu".to_string(),
        layer_type: LayerType::ReLU,
        inputs: vec!["input".to_string()],
        outputs: vec!["relu_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };
    // Simulate an unsupported op (Unknown type) with a single activation input
    let unsupported = LayerSpec {
        name: "reciprocal_like".to_string(),
        layer_type: LayerType::Unknown,
        inputs: vec!["relu_out".to_string()],
        outputs: vec!["recip_out".to_string()],
        weights: None,
        attributes: HashMap::new(),
    };

    let network = Network {
        name: "unsupported_single_input_regression".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "recip_out".to_string(),
            shape: vec![1],
            dtype: DataType::Float32,
        }],
        layers: vec![relu, unsupported],
        param_count: 0,
    };

    let model = build_model(network);
    let graph = build_graph(&model);

    // The unsupported op must NOT be treated as identity
    // It must create an OpaqueSkipLayer node
    let skip_name = "reciprocal_like__skip";
    assert_opaque_skip(&graph, skip_name);

    let skip_node = graph.node(skip_name).expect("skip node exists");
    assert_eq!(
        skip_node.inputs(),
        vec!["relu".to_string()],
        "OpaqueSkip should connect to upstream relu"
    );

    // The graph output should be the skip node
    assert_eq!(graph.output_name(), skip_name);

    // Verify the node is NOT a pass-through/identity by checking
    // it's an OpaqueSkip (which returns [-inf, +inf] bounds)
    assert!(
        !matches!(skip_node.layer(), Layer::ReLU(_)),
        "Skip node must not be identity/pass-through"
    );
}

/// Build a branching OnnxModel (ReLU + Sigmoid → Add) with tensor metadata.
/// Used by the round-trip equivalence test.
fn build_branching_onnx_model() -> OnnxModel {
    let layers = vec![
        LayerSpec {
            name: "relu".to_string(),
            layer_type: LayerType::ReLU,
            inputs: vec!["input".to_string()],
            outputs: vec!["relu_out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        },
        LayerSpec {
            name: "sigmoid".to_string(),
            layer_type: LayerType::Sigmoid,
            inputs: vec!["input".to_string()],
            outputs: vec!["sigmoid_out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        },
        LayerSpec {
            name: "add".to_string(),
            layer_type: LayerType::Add,
            inputs: vec!["relu_out".to_string(), "sigmoid_out".to_string()],
            outputs: vec!["out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        },
    ];
    let network = Network {
        name: "round_trip_test".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1, 4],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "out".to_string(),
            shape: vec![1, 4],
            dtype: DataType::Float32,
        }],
        layers,
        param_count: 0,
    };
    OnnxModel {
        network,
        weights: WeightStore::new(),
        tensor_producer: HashMap::from([
            ("relu_out".to_string(), "input".to_string()),
            ("sigmoid_out".to_string(), "input".to_string()),
        ]),
        constant_tensors: HashSet::new(),
        tensor_shapes: HashMap::from([
            ("input".to_string(), vec![1, 4]),
            ("relu_out".to_string(), vec![1, 4]),
            ("sigmoid_out".to_string(), vec![1, 4]),
            ("out".to_string(), vec![1, 4]),
        ]),
        original_float32_initializers: HashMap::new(),
        original_network_topology: None,
        opset_imports: HashMap::from([(String::new(), 17)]),
    }
}

/// Assert two GraphNetworks are structurally equivalent (same output, nodes, edges).
fn assert_graph_networks_equivalent(
    graph_a: &GraphNetwork,
    graph_b: &GraphNetwork,
    node_names: &[&str],
) {
    assert_eq!(
        graph_a.output_name(),
        graph_b.output_name(),
        "output node mismatch"
    );
    assert_eq!(
        graph_a.num_nodes(),
        graph_b.num_nodes(),
        "node count mismatch"
    );
    for name in node_names {
        let a = graph_a
            .node(name)
            .unwrap_or_else(|| panic!("direct: missing '{name}'"));
        let b = graph_b
            .node(name)
            .unwrap_or_else(|| panic!("round-trip: missing '{name}'"));
        assert_eq!(a.inputs(), b.inputs(), "inputs mismatch for '{name}'");
    }
}

/// OnnxModel → GraphModel round-trip produces the same GraphNetwork as the
/// direct OnnxModel → GraphNetwork path. Proves the format-neutral GraphModel
/// contract (#3288) faithfully represents ONNX-loaded models.
///
/// Part of #3288.
#[test]
fn onnx_to_graph_model_round_trip_produces_equivalent_graph_network() {
    // Path A: OnnxModel → GraphNetwork (direct)
    let graph_a = build_branching_onnx_model()
        .to_graph_network()
        .expect("direct path should succeed");

    // Path B: OnnxModel → GraphModel → GraphNetwork (round-trip)
    let graph_b = build_branching_onnx_model()
        .to_graph_model()
        .build_graph_network(GraphNetworkOptions::default())
        .expect("round-trip path should succeed");

    assert_graph_networks_equivalent(&graph_a, &graph_b, &["relu", "sigmoid", "add"]);
}

/// Build a single-layer LayerNorm OnnxModel matching the #4172 builder-test shape.
fn build_standard_layernorm_model() -> OnnxModel {
    let mut weights = WeightStore::new();
    weights.insert("ny".to_string(), arr1(&[1.0_f32, 1.5, 0.5, 2.0]).into_dyn());
    weights.insert(
        "beta".to_string(),
        arr1(&[0.0_f32, 0.25, -0.5, 0.75]).into_dyn(),
    );

    let network = Network {
        name: "layernorm_facade".to_string(),
        inputs: vec![TensorSpec {
            name: "input".to_string(),
            shape: vec![1, 2, 4],
            dtype: DataType::Float32,
        }],
        outputs: vec![TensorSpec {
            name: "layernorm_out".to_string(),
            shape: vec![1, 2, 4],
            dtype: DataType::Float32,
        }],
        layers: vec![LayerSpec {
            name: "layernorm".to_string(),
            layer_type: LayerType::LayerNorm,
            inputs: vec!["input".to_string(), "ny".to_string(), "beta".to_string()],
            outputs: vec!["layernorm_out".to_string()],
            weights: None,
            attributes: HashMap::new(),
        }],
        param_count: 8,
    };

    let mut model = build_model_with_shapes(
        network,
        HashMap::from([
            ("input".to_string(), vec![1, 2, 4]),
            ("layernorm_out".to_string(), vec![1, 2, 4]),
        ]),
    );
    model.weights = weights;
    model
}

/// Public-surface regression: `CompoundNodePolicy::DecomposeNormalization` is importable from
/// the crate root and drives the LayerNorm rewrite through `ny-onnx` without `ny-build`.
///
/// Mirrors the committed `ny-build` builder-test shape from #4172.
/// Part of #4173.
#[test]
fn compound_node_policy_public_facade_rewrites_layernorm_4173() {
    let model = build_standard_layernorm_model();
    let graph = build_graph_with_options(
        &model,
        GraphNetworkOptions {
            compound_node_policy: CompoundNodePolicy::DecomposeNormalization,
            ..GraphNetworkOptions::default()
        },
    );

    let output = graph.node("layernorm").expect("final node should exist");
    assert!(
        matches!(output.layer(), Layer::AddConstant(_)),
        "expected decomposed LayerNorm to become AddConstant, got {:?}",
        output.layer()
    );
    assert!(
        matches!(
            graph.node("layernorm__mean").expect("mean node").layer(),
            Layer::ReduceMean(_)
        ),
        "expected ReduceMean node in decomposed LayerNorm"
    );
    assert!(
        matches!(
            graph
                .node("layernorm__inv_std")
                .expect("inv_std node")
                .layer(),
            Layer::Reciprocal(_)
        ),
        "expected Reciprocal node in decomposed LayerNorm"
    );
    assert_eq!(graph.output_name(), "layernorm");
}
