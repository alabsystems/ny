// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use ndarray::arr1;
use ny_tensor::BoundedTensor;
use tempfile::tempdir;

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_basic() {
    // Test that to_graph_network creates a valid DAG
    let path = require_test_model("linear_relu.onnx");

    let model = load_onnx(&path).expect("Failed to load model");

    // Convert to graph network
    let graph = model
        .to_graph_network()
        .expect("Failed to convert to graph network");

    // Should have nodes for each layer
    assert!(graph.num_nodes() > 0, "Graph should have nodes");

    // Test IBP propagation through the graph
    let input = BoundedTensor::new(
        arr1(&[0.0_f32, 0.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .unwrap();

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through graph should succeed");

    // Verify soundness: test corner points of input interval
    let seq_network = model
        .to_propagate_network()
        .expect("Sequential conversion failed");

    let test_points = [
        [0.0_f32, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ];

    for point in test_points {
        let concrete_input =
            BoundedTensor::new(arr1(&point).into_dyn(), arr1(&point).into_dyn()).unwrap();

        let concrete_output = seq_network.propagate_ibp(&concrete_input).unwrap();

        for i in 0..concrete_output.lower().len() {
            assert!(
                concrete_output.lower()[[i]] >= output.lower()[[i]] - 1e-5
                    && concrete_output.upper()[[i]] <= output.upper()[[i]] + 1e-5,
                "Graph IBP bounds should contain concrete outputs"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_onnx_loader_round_trip_sequential_graph() {
    let path = require_test_model("linear_relu.onnx");
    assert_round_trip_sequential(&path);
}

#[ntest::timeout(10000)]
#[test]
fn test_onnx_loader_round_trip_smoke_single_linear() {
    let path = require_test_model("single_linear.onnx");
    assert_round_trip_sequential(&path);
}

#[ntest::timeout(10000)]
#[test]
fn test_onnx_loader_round_trip_simple_mlp() {
    let path = require_test_model("simple_mlp.onnx");
    assert_round_trip_sequential(&path);
}

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_skipped_multi_output_maps_all_outputs() {
    use crate::onnx_proto;
    use prost::Message;
    use std::path::Path;

    fn tensor_value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
        let dims = shape
            .iter()
            .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
                value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                    *dim,
                )),
            })
            .collect();
        onnx_proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx_proto::TypeProto {
                tensor_type: Some(onnx_proto::TensorTypeProto {
                    elem_type: 1,
                    shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
                }),
            }),
        }
    }

    fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: name.to_string(),
            op_type: op_type.to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }
    }

    fn write_onnx_model(path: &Path, graph: onnx_proto::GraphProto) {
        let model = onnx_proto::ModelProto {
            ir_version: 9,
            opset_import: vec![onnx_proto::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "ny-onnx-fixture".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(graph),
        };
        let mut buf = Vec::new();
        model.encode(&mut buf).expect("Failed to encode ONNX");
        std::fs::write(path, buf).expect("Failed to write ONNX");
    }

    let temp = tempdir().expect("Failed to create temp dir");
    let path = temp.path().join("skip_multi_output.onnx");

    let graph = onnx_proto::GraphProto {
        node: vec![
            node("relu1", "Relu", &["input"], &["relu_out"]),
            // Use "Identity" (a known skip-op returning LayerType::Unknown) instead
            // of "UnsupportedOp" which returns Err after #2931 soundness fix.
            node(
                "skip1",
                "Identity",
                &["relu_out"],
                &["skip_out0", "skip_out1"],
            ),
            node("add1", "Add", &["skip_out0", "skip_out1"], &["add_out"]),
        ],
        name: "skip_multi_output".to_string(),
        initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("add_out", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };

    write_onnx_model(&path, graph);

    let model = load_onnx(path.to_str().expect("Path should be UTF-8"))
        .expect("Failed to load skip_multi_output model");
    let graph = model
        .to_graph_network()
        .expect("GraphNetwork conversion should wire through Identity ops");

    // Identity ops are wire-through: excluded from the layer list at load time
    // and traced via tensor_producer. Both outputs (skip_out0, skip_out1) map
    // back to the upstream producer (relu1). No OpaqueSkipLayer is created.
    assert!(
        graph.node("skip1").is_none(),
        "Identity op should not appear as a graph node"
    );
    assert!(
        graph.node("skip1__skip").is_none(),
        "Wire-through Identity should not create OpaqueSkip node"
    );

    assert_eq!(
        graph.output_name(),
        "add1",
        "Graph output should map to the downstream node"
    );

    let add_node = graph.node("add1").expect("add1 node should exist in graph");
    assert_eq!(
        add_node.inputs(),
        vec!["relu1".to_string(), "relu1".to_string()],
        "Identity wire-through should map both outputs to the upstream producer"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_skipped_outputs_map_to_input() {
    use crate::onnx_proto;
    use prost::Message;
    use std::path::Path;

    fn tensor_value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
        let dims = shape
            .iter()
            .map(|dim| onnx_proto::tensor_shape_proto::Dimension {
                value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                    *dim,
                )),
            })
            .collect();
        onnx_proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(onnx_proto::TypeProto {
                tensor_type: Some(onnx_proto::TensorTypeProto {
                    elem_type: 1,
                    shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
                }),
            }),
        }
    }

    fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: name.to_string(),
            op_type: op_type.to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }
    }

    fn write_onnx_model(path: &Path, graph: onnx_proto::GraphProto) {
        let model = onnx_proto::ModelProto {
            ir_version: 9,
            opset_import: vec![onnx_proto::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: "ny-onnx-fixture".to_string(),
            producer_version: String::new(),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(graph),
        };
        let mut buf = Vec::new();
        model.encode(&mut buf).expect("Failed to encode ONNX");
        std::fs::write(path, buf).expect("Failed to write ONNX");
    }

    let temp = tempdir().expect("Failed to create temp dir");
    let path = temp.path().join("skip_outputs_map_to_input.onnx");

    let graph = onnx_proto::GraphProto {
        node: vec![
            // Use "Identity" (a known skip-op returning LayerType::Unknown) instead
            // of "UnsupportedOp" which returns Err after #2931 soundness fix.
            node("skip1", "Identity", &["input"], &["skip_out0", "skip_out1"]),
            node("add1", "Add", &["skip_out0", "skip_out1"], &["add_out"]),
        ],
        name: "skip_outputs_map_to_input".to_string(),
        initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("add_out", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };

    write_onnx_model(&path, graph);

    let model = load_onnx(path.to_str().expect("Path should be UTF-8"))
        .expect("Failed to load skip_outputs_map_to_input model");
    let graph = model
        .to_graph_network()
        .expect("GraphNetwork conversion should wire through Identity ops");

    // Identity ops are wire-through: both outputs map back to the graph input
    // via tensor_producer. No OpaqueSkipLayer is created.
    assert!(
        graph.node("skip1__skip").is_none(),
        "Wire-through Identity should not create OpaqueSkip node"
    );

    let add_node = graph.node("add1").expect("add1 node should exist in graph");
    assert_eq!(
        add_node.inputs(),
        vec!["_input".to_string(), "_input".to_string()],
        "Identity wire-through should map both outputs to the graph input"
    );
}

/// Regression test for #3186: binary op with one constant-tensor input
/// (produced by a chain the pre-evaluator can't handle) and one activation
/// input should load and produce a graph network, not error.
#[ntest::timeout(10000)]
#[test]
fn test_to_graph_network_const_activation_binary_op_3186() {
    let path = require_test_model("const_activation_binary_op.onnx");

    let model = load_onnx(&path).expect("Model with const+activation Add should load (#3186)");

    // The graph network builder must handle the Add where one input is a
    // constant tensor without a pre-evaluated value (Neg output).
    let graph = model
        .to_graph_network()
        .expect("GraphNetwork conversion should handle const+activation binary op (#3186)");

    assert!(
        graph.num_nodes() > 0,
        "Graph should have nodes after loading const+activation model"
    );
}
