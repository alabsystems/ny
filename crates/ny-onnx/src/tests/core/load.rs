// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::load_onnx_bytes;
use approx::assert_relative_eq;
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::arr1;
use ny_core::LayerType;
use ny_propagate::Layer as PropLayer;
use ny_tensor::BoundedTensor;
use std::io::Write;
use tempfile::tempdir;

fn tensor_value_info(name: &str, shape: &[i64], elem_type: i32) -> onnx_proto::ValueInfoProto {
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
                elem_type,
                shape: Some(onnx_proto::TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_single_linear() {
    let path = require_test_model("single_linear.onnx");

    let model = load_onnx(&path).expect("Failed to load model");

    // Check structure
    assert_eq!(model.network.inputs.len(), 1);
    assert_eq!(model.network.outputs.len(), 1);
    assert_eq!(model.network.layers.len(), 1);
    assert_eq!(model.network.layers[0].layer_type, LayerType::Linear);

    // Check weights were loaded
    assert!(model.weights.get("weight").is_some());
    assert!(model.weights.get("bias").is_some());

    // Verify weight values
    let weight = model.weights.get("weight").unwrap();
    assert_eq!(weight.shape(), &[3, 2]);

    // Expected weights: [[1.0, 2.0], [3.0, -1.0], [-2.0, 1.0]]
    assert_relative_eq!(weight[[0, 0]], 1.0, epsilon = 1e-6);
    assert_relative_eq!(weight[[0, 1]], 2.0, epsilon = 1e-6);
    assert_relative_eq!(weight[[1, 0]], 3.0, epsilon = 1e-6);
    assert_relative_eq!(weight[[1, 1]], -1.0, epsilon = 1e-6);

    let bias = model.weights.get("bias").unwrap();
    assert_eq!(bias.shape(), &[3]);
    assert_relative_eq!(bias[[0]], 0.5, epsilon = 1e-6);
    assert_relative_eq!(bias[[1]], -0.5, epsilon = 1e-6);
    assert_relative_eq!(bias[[2]], 1.0, epsilon = 1e-6);

    assert!(
        model.opset_imports.contains_key("ai.onnx"),
        "expected opset imports to include ai.onnx"
    );
    assert!(
        model.opset_imports.contains_key(""),
        "expected opset imports to include the default domain"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_onnx_gzip() {
    let src_path = require_test_model("simple_mlp.onnx");

    let src_bytes = std::fs::read(&src_path).expect("Failed to read test ONNX model");
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&src_bytes)
        .expect("Failed to gzip test ONNX model");
    let gz_bytes = enc.finish().expect("Failed to finalize gzip stream");

    let dir = tempdir().expect("Failed to create temp dir");
    let gz_path = dir.path().join("simple_mlp.onnx.gz");
    std::fs::write(&gz_path, gz_bytes).expect("Failed to write gzipped ONNX model");

    let plain = load_onnx(&src_path).expect("Failed to load plain ONNX");
    let gz = load_onnx(&gz_path).expect("Failed to load gzipped ONNX");

    assert_eq!(gz.network.inputs.len(), plain.network.inputs.len());
    assert_eq!(gz.network.outputs.len(), plain.network.outputs.len());
    assert_eq!(gz.network.layers.len(), plain.network.layers.len());
    assert_eq!(gz.network.param_count, plain.network.param_count);
}

#[ntest::timeout(10000)]
#[test]
fn test_value_info_dtype_mapping() {
    use prost::Message;

    let graph = onnx_proto::GraphProto {
        node: Vec::new(),
        name: "dtype_test".to_string(),
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 4], 1)],
        output: vec![tensor_value_info("output", &[1, 2], 7)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
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

    let bytes = model.encode_to_vec();
    // This metadata-only fixture has no producer for its output. Bypass native
    // shape inference so the test covers ny's dtype parser in isolation.
    let config = crate::OnnxLoadConfig::default()
        .with_shape_inference_policy(crate::ShapeInferencePolicy::Skip);
    let loaded = crate::load_onnx_bytes_with_config("dtype_test.onnx", &bytes, &config)
        .expect("Failed to load ONNX bytes");

    assert_eq!(loaded.network.inputs.len(), 1);
    assert_eq!(loaded.network.outputs.len(), 1);
    assert_eq!(loaded.network.inputs[0].dtype, DataType::Float32);
    assert_eq!(loaded.network.outputs[0].dtype, DataType::Int64);
}

#[cfg(not(feature = "onnx-value-info"))]
#[ntest::timeout(10000)]
#[test]
fn test_load_bytes_without_value_info_feature() {
    use prost::Message;

    // Regression for #1216: parsing should succeed when GraphProto omits value_info.
    let graph = onnx_proto::GraphProto {
        node: Vec::new(),
        name: "no_value_info_feature".to_string(),
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 2], 1)],
        output: vec![tensor_value_info("output", &[1, 2], 1)],
    };
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

    let bytes = model.encode_to_vec();
    let loaded = load_onnx_bytes("no_value_info_feature.onnx", &bytes)
        .expect("Failed to load no-value-info model");

    assert_eq!(loaded.network.inputs.len(), 1);
    assert_eq!(loaded.network.outputs.len(), 1);
    assert!(loaded.network.layers.is_empty());
    assert_eq!(loaded.tensor_shapes["input"], vec![1, 2]);
    assert_eq!(loaded.tensor_shapes["output"], vec![1, 2]);
}

#[cfg(feature = "onnx-value-info")]
#[ntest::timeout(10000)]
#[test]
fn test_load_ignores_value_info_without_type() {
    use prost::Message;

    let graph = onnx_proto::GraphProto {
        node: Vec::new(),
        name: "value_info_missing_type".to_string(),
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 4], 1)],
        output: vec![tensor_value_info("output", &[1, 4], 1)],
        value_info: vec![onnx_proto::ValueInfoProto {
            name: "intermediate".to_string(),
            r#type: None,
        }],
    };
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

    let bytes = model.encode_to_vec();
    // An untyped ValueInfo plus a graph with no output producer is an
    // intentionally adverse parser fixture, not valid input to ORT.
    let config = crate::OnnxLoadConfig::default()
        .with_shape_inference_policy(crate::ShapeInferencePolicy::Skip);
    let loaded =
        crate::load_onnx_bytes_with_config("value_info_missing_type.onnx", &bytes, &config)
            .expect("Failed to load ONNX");

    assert_eq!(loaded.network.inputs.len(), 1);
    assert_eq!(loaded.network.outputs.len(), 1);
    assert!(
        !loaded.tensor_shapes.contains_key("intermediate"),
        "value_info without type should not populate tensor_shapes"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_onnx_bytes_empty_fails() {
    let err = match load_onnx_bytes("empty.onnx", &[]) {
        Ok(_) => panic!("Expected empty ONNX to fail"),
        Err(err) => err,
    };
    let msg = format!("{err}");
    assert_parse_or_missing_graph_error(&msg, "empty ONNX bytes");
}

#[ntest::timeout(10000)]
#[test]
fn test_load_onnx_bytes_garbage_fails() {
    let bytes = [0x01u8, 0x02, 0x03, 0x04];
    assert_parse_error("garbage.onnx", &bytes);
}

fn assert_parse_error(name: &str, data: &[u8]) {
    let err = match load_onnx_bytes(name, data) {
        Ok(_) => panic!("Expected ONNX parse to fail"),
        Err(err) => err,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("Failed to parse ONNX"),
        "Unexpected error for {name}: {msg}"
    );
}

fn assert_missing_graph_error(msg: &str, context: &str) {
    let has_missing_graph = msg.contains("has no graph") || msg.contains("missing graph");
    assert!(
        has_missing_graph,
        "Expected missing graph error for {context}: {msg}"
    );
}

fn assert_parse_or_missing_graph_error(msg: &str, context: &str) {
    let is_parse_error = msg.contains("Failed to parse ONNX");
    if !is_parse_error {
        assert_missing_graph_error(msg, context);
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_onnx_bytes_missing_graph_fails() {
    use prost::Message;

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
        graph: None,
    };
    let bytes = model.encode_to_vec();
    let err = match load_onnx_bytes("missing_graph.onnx", &bytes) {
        Ok(_) => panic!("Expected missing graph ONNX to fail"),
        Err(err) => err,
    };
    let msg = format!("{err}");
    assert_missing_graph_error(&msg, "missing graph ONNX bytes");
}

#[ntest::timeout(10000)]
#[test]
fn test_load_linear_relu() {
    let path = require_test_model("linear_relu.onnx");

    let model = load_onnx(&path).expect("Failed to load model");

    // Should have 2 layers: Linear + ReLU
    assert_eq!(model.network.layers.len(), 2);
    assert_eq!(model.network.layers[0].layer_type, LayerType::Linear);
    assert_eq!(model.network.layers[1].layer_type, LayerType::ReLU);

    // Convert and test propagation
    let network = model.to_propagate_network().expect("Failed to convert");

    let input =
        BoundedTensor::new(arr1(&[1.0, 1.0]).into_dyn(), arr1(&[1.0, 1.0]).into_dyn()).unwrap();

    let output = network.propagate_ibp(&input).unwrap();

    // After linear: [3.5, 1.5, 0.0]
    // After ReLU: [3.5, 1.5, 0.0] (all >= 0)
    assert_relative_eq!(output.lower()[[0]], 3.5, epsilon = 1e-5);
    assert_relative_eq!(output.lower()[[1]], 1.5, epsilon = 1e-5);
    assert_relative_eq!(output.lower()[[2]], 0.0, epsilon = 1e-5);
}

#[ntest::timeout(10000)]
#[test]
fn test_load_simple_mlp() {
    let path = require_test_model("simple_mlp.onnx");

    let model = load_onnx(&path).expect("Failed to load model");

    // Should have 3 layers: Linear + ReLU + Linear
    assert_eq!(model.network.layers.len(), 3);
    assert_eq!(model.network.layers[0].layer_type, LayerType::Linear);
    assert_eq!(model.network.layers[1].layer_type, LayerType::ReLU);
    assert_eq!(model.network.layers[2].layer_type, LayerType::Linear);

    // Verify weight shapes
    let w1 = model.weights.get("w1").unwrap();
    assert_eq!(w1.shape(), &[4, 2]); // 4 outputs, 2 inputs

    let w2 = model.weights.get("w2").unwrap();
    assert_eq!(w2.shape(), &[2, 4]); // 2 outputs, 4 inputs
}

#[ntest::timeout(10000)]
#[test]
fn test_load_mul_binary() {
    let path = require_test_model("mul_binary.onnx");

    let model = load_onnx(&path).expect("Failed to load MulBinary model");
    assert_eq!(model.network.layers.len(), 1);
    assert_eq!(model.network.layers[0].layer_type, LayerType::Mul);

    let network = model
        .to_propagate_network()
        .expect("Failed to convert MulBinary");
    assert!(
        network
            .layers()
            .iter()
            .any(|l| matches!(l, PropLayer::MulBinary(_))),
        "Expected propagate network to contain MulBinary layer"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_mul_binary_graph_has_two_inputs() {
    let path = require_test_model("mul_binary.onnx");

    let model = load_onnx(&path).expect("Failed to load MulBinary model");
    let graph = model
        .to_graph_network()
        .expect("Failed to convert MulBinary to graph network");
    let mut mul_nodes = Vec::new();

    for name in graph.node_names() {
        let node = graph
            .node(name)
            .unwrap_or_else(|| panic!("Missing graph node '{}'", name));
        if matches!(node.layer(), PropLayer::MulBinary(_)) {
            mul_nodes.push(node);
        }
    }

    assert_eq!(
        mul_nodes.len(),
        1,
        "MulBinary graph should have exactly one MulBinary node"
    );
    let output_name = graph.output_name();
    let output_node = graph
        .node(output_name)
        .unwrap_or_else(|| panic!("Missing output node '{}'", output_name));
    match output_node.layer() {
        PropLayer::MulBinary(_) => {}
        other => panic!(
            "Expected output node '{}' to be MulBinary, got {:?}",
            output_name,
            other.layer_type()
        ),
    }

    assert_eq!(
        output_node.inputs().len(),
        2,
        "MulBinary graph node should have two activation inputs"
    );
    for input in output_node.inputs() {
        assert!(
            !input.is_empty(),
            "MulBinary graph input name should not be empty"
        );
        if input != "_input" {
            assert!(
                graph.contains_node(input),
                "MulBinary graph input '{}' should be a graph node or _input",
                input
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_mul_binary_activation_inputs_graph_links_inputs() {
    let path = require_test_model("mul_binary_activation_inputs.onnx");

    let model = load_onnx(&path).expect("Failed to load MulBinary activation inputs model");
    let network = model
        .to_propagate_network()
        .expect("Failed to convert MulBinary activation inputs to sequential network");
    assert!(
        network
            .layers()
            .iter()
            .any(|l| matches!(l, PropLayer::MulBinary(_))),
        "Expected propagate network to contain MulBinary layer"
    );
    assert_eq!(
        network.layers().len(),
        3,
        "Expected Relu, Sigmoid, MulBinary layers in sequential network"
    );
    assert!(
        matches!(network.layers().last(), Some(PropLayer::MulBinary(_))),
        "Expected MulBinary to be the final sequential layer"
    );

    let graph = model
        .to_graph_network()
        .expect("Failed to convert MulBinary activation inputs to graph network");

    let output_name = graph.output_name();
    let output_node = graph
        .node(output_name)
        .unwrap_or_else(|| panic!("Missing output node '{}'", output_name));
    match output_node.layer() {
        PropLayer::MulBinary(_) => {}
        other => panic!(
            "Expected output node '{}' to be MulBinary, got {:?}",
            output_name,
            other.layer_type()
        ),
    }

    assert_eq!(
        output_node.inputs().len(),
        2,
        "MulBinary activation graph node should have two inputs"
    );
    assert!(
        output_node.inputs().contains(&"relu".to_string()),
        "MulBinary activation input should include relu node"
    );
    assert!(
        output_node.inputs().contains(&"sigmoid".to_string()),
        "MulBinary activation input should include sigmoid node"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_mul_binary_activation_broadcastable_inputs() {
    let path = require_test_model("mul_binary_activation_broadcast.onnx");

    let model = load_onnx(&path).expect("Failed to load MulBinary activation broadcast model");
    let network = model
        .to_propagate_network()
        .expect("Failed to convert MulBinary activation broadcast to sequential network");
    assert!(
        network
            .layers()
            .iter()
            .any(|l| matches!(l, PropLayer::MulBinary(_))),
        "Expected propagate network to contain MulBinary layer"
    );
    assert_eq!(
        network.layers().len(),
        3,
        "Expected Relu, Sigmoid, MulBinary layers in sequential network"
    );
    assert!(
        matches!(network.layers()[0], PropLayer::ReLU(_)),
        "Expected first layer to be ReLU"
    );
    assert!(
        matches!(network.layers()[1], PropLayer::Sigmoid(_)),
        "Expected second layer to be Sigmoid"
    );
    assert!(
        matches!(network.layers().last(), Some(PropLayer::MulBinary(_))),
        "Expected MulBinary to be the final sequential layer"
    );

    let graph = model
        .to_graph_network()
        .expect("Failed to convert MulBinary activation broadcast to graph network");
    let output_name = graph.output_name();
    let output_node = graph
        .node(output_name)
        .unwrap_or_else(|| panic!("Missing output node '{}'", output_name));
    match output_node.layer() {
        PropLayer::MulBinary(_) => {}
        other => panic!(
            "Expected output node '{}' to be MulBinary, got {:?}",
            output_name,
            other.layer_type()
        ),
    }
    assert_eq!(
        output_node.inputs().len(),
        2,
        "MulBinary activation broadcast node should have two inputs"
    );
    assert!(
        output_node.inputs().contains(&"relu".to_string()),
        "MulBinary activation broadcast input should include relu node"
    );
    assert!(
        output_node.inputs().contains(&"sigmoid".to_string()),
        "MulBinary activation broadcast input should include sigmoid node"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_mul_const_broadcast_fails_conversion() {
    use prost::Message;

    fn tensor_f32(name: &str, shape: &[i64], data: &[f32]) -> onnx_proto::TensorProto {
        assert_eq!(shape.iter().product::<i64>() as usize, data.len());
        onnx_proto::TensorProto {
            dims: shape.to_vec(),
            data_type: 1,
            name: name.to_string(),
            raw_data: Vec::new(),
            float_data: data.to_vec(),
            ..Default::default()
        }
    }

    let graph = onnx_proto::GraphProto {
        node: vec![onnx_proto::NodeProto {
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["out".to_string()],
            name: "mul".to_string(),
            op_type: "Mul".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }],
        name: "mul_const_broadcast".to_string(),
        initializer: vec![
            tensor_f32("a", &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            tensor_f32("b", &[2], &[5.0, 6.0]),
        ],
        sparse_initializer: Vec::new(),
        input: Vec::new(),
        output: vec![tensor_value_info("out", &[2, 2], 1)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
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
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");

    let model = load_onnx_bytes("mul_const_broadcast.onnx", &bytes)
        .expect("Failed to load Mul const broadcast model");
    // Both Mul inputs are initializers, so const-folding evaluates the node
    // and no layer remains. The sequential lane must fail closed rather than
    // return an empty (identity) network for a constant-output model.
    let err = model
        .to_propagate_network()
        .expect_err("Expected Mul const broadcast conversion to fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("empty network"),
        "Expected constant-only model rejection, got: {msg}"
    );
    assert!(
        msg.contains("identity"),
        "Expected the identity-hazard rationale in the error, got: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_mul_const_incompatible_shapes_fails() {
    // Test fixture with two constant inputs that have incompatible shapes ([2,3] vs [3,2]).
    // This exercises the try_convert_mul → broadcast-incompatible → error path.
    let path = require_test_model("mul_const_incompatible_shapes.onnx");

    let model = load_onnx(&path).expect("Failed to load mul_const_incompatible_shapes.onnx");
    // The Mul node has two constant inputs, so the sequential builder skips
    // it as a constant computation (the broadcast-incompatible shapes mean it
    // can never actually fold to a value). Conversion must fail closed on the
    // resulting empty network instead of silently returning an identity net.
    let err = model
        .to_propagate_network()
        .expect_err("Expected Mul const incompatible shapes conversion to fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("empty network"),
        "Expected constant-only model rejection, got: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_load_softmax() {
    let path = require_test_model_with_hint("softmax.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load softmax model");

    // Check we have a softmax layer
    let softmax = model
        .network
        .layers
        .iter()
        .find(|l| l.layer_type == LayerType::Softmax)
        .expect("Expected Softmax layer in model");

    // Verify axis attribute was captured from ONNX node
    assert_eq!(
        softmax.attributes.get("axis"),
        Some(&AttributeValue::Int(-1))
    );

    // Convert and test IBP
    let network = model.to_propagate_network().expect("Failed to convert");

    // Test with bounded input
    let input = BoundedTensor::new(
        arr1(&[0.0, 1.0, 2.0, 3.0]).into_dyn(),
        arr1(&[0.5, 1.5, 2.5, 3.5]).into_dyn(),
    )
    .unwrap();

    let output = network.propagate_ibp(&input).expect("IBP failed");

    // Softmax outputs should be in [0, 1]
    for &l in output.lower().iter() {
        assert!(l >= 0.0, "Softmax lower bound {} < 0", l);
    }
    for &u in output.upper().iter() {
        assert!(u <= 1.0, "Softmax upper bound {} > 1", u);
    }

    // Outputs should sum close to 1 (for a point in the interval)
    let lower_sum: f32 = output.lower().iter().sum();
    let upper_sum: f32 = output.upper().iter().sum();
    // Bounds on the sum
    assert!(lower_sum <= 1.0 + 0.01, "Lower sum {} > 1", lower_sum);
    assert!(upper_sum >= 1.0 - 0.01, "Upper sum {} < 1", upper_sum);
}

#[ntest::timeout(10000)]
#[test]
fn test_load_gelu_decomposed() {
    let path = require_test_model_with_hint("gelu.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load gelu model");
    assert!(
        model
            .network
            .layers
            .iter()
            .any(|l| l.layer_type == LayerType::Erf),
        "decomposed GELU must preserve its exact Erf primitive"
    );
    assert!(
        !model
            .network
            .layers
            .iter()
            .any(|l| l.layer_type == LayerType::GELU),
        "decomposed GELU must not use the disabled canonical GELU fusion"
    );

    // The decomposed formula reuses x in its final multiplication, so it is a
    // DAG rather than a sequential chain.  Verify it through the graph path.
    let network = model.to_graph_network().expect("Failed to convert");
    assert!(
        network
            .node_names()
            .iter()
            .filter_map(|name| network.node(name))
            .any(|node| matches!(node.layer(), PropLayer::Erf(_))),
        "propagate network must preserve the exact Erf primitive"
    );

    // Soundness: sample points in input interval, verify outputs are within bounds.
    let input = BoundedTensor::new(
        arr1(&[-2.0, -1.0, 0.0, 2.0]).into_dyn(),
        arr1(&[-1.5, -0.5, 0.5, 3.0]).into_dyn(),
    )
    .unwrap();
    let bounds = network.propagate_ibp(&input).expect("IBP failed");

    let test_points = vec![
        arr1(&[-2.0, -1.0, 0.0, 2.0]),
        arr1(&[-1.5, -0.5, 0.5, 3.0]),
        arr1(&[-1.75, -0.75, 0.25, 2.5]),
    ];

    let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
    for point in test_points {
        for (i, &x) in point.iter().enumerate() {
            let y = 0.5 * x * (1.0 + libm::erff(x * inv_sqrt2));
            assert!(
                y >= bounds.lower()[[i]] - 1e-5,
                "GELU output {} = {} below lower bound {}",
                i,
                y,
                bounds.lower()[[i]]
            );
            assert!(
                y <= bounds.upper()[[i]] + 1e-5,
                "GELU output {} = {} above upper bound {}",
                i,
                y,
                bounds.upper()[[i]]
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_layer_norm_decomposed() {
    let path = require_test_model_with_hint("layer_norm.onnx", TRANSFORMER_TEST_MODEL_HINT);

    let model = load_onnx(&path).expect("Failed to load layer_norm model");
    assert!(
        model
            .network
            .layers
            .iter()
            .any(|l| l.layer_type == LayerType::LayerNorm),
        "Expected LayerNorm layer after pattern fusion"
    );

    let network = model.to_propagate_network().expect("Failed to convert");
    let layer_norm = network
        .layers()
        .iter()
        .find_map(|l| match l {
            PropLayer::LayerNorm(ln) => Some(ln),
            _ => None,
        })
        .expect("Expected propagate network to contain LayerNorm layer");

    // Soundness (single sample): evaluate LayerNorm at a point and ensure it lies in bounds.
    let input = BoundedTensor::new(
        arr1(&[-1.0, -0.5, 0.0, 0.5]).into_dyn(),
        arr1(&[0.0, 0.5, 1.0, 1.5]).into_dyn(),
    )
    .unwrap();
    let bounds = network.propagate_ibp(&input).expect("IBP failed");

    let x = arr1(&[-0.5, 0.0, 0.5, 1.0]);
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std = (var + layer_norm.eps).sqrt();

    for i in 0..x.len() {
        let y = (x[i] - mean) / std;
        let out = layer_norm.ny[i] * y + layer_norm.beta[i];
        assert!(
            out >= bounds.lower()[[i]] - 1e-4,
            "LayerNorm output {} = {} below lower bound {}",
            i,
            out,
            bounds.lower()[[i]]
        );
        assert!(
            out <= bounds.upper()[[i]] + 1e-4,
            "LayerNorm output {} = {} above upper bound {}",
            i,
            out,
            bounds.upper()[[i]]
        );
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_layer_norm_reciprocal_fusion() {
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

    fn tensor_f32(name: &str, shape: &[i64], data: &[f32]) -> onnx_proto::TensorProto {
        let elements = shape.iter().product::<i64>() as usize;
        assert_eq!(elements, data.len());
        onnx_proto::TensorProto {
            dims: shape.to_vec(),
            data_type: 1,
            name: name.to_string(),
            raw_data: Vec::new(),
            float_data: data.to_vec(),
            ..Default::default()
        }
    }

    fn attr_ints(name: &str, values: &[i64]) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            r#type: 7,
            ints: values.to_vec(),
            ..Default::default()
        }
    }

    fn attr_tensor(name: &str, tensor: onnx_proto::TensorProto) -> onnx_proto::AttributeProto {
        onnx_proto::AttributeProto {
            name: name.to_string(),
            t: Some(tensor),
            r#type: 4,
            ..Default::default()
        }
    }

    fn node(
        name: &str,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attrs: Vec<onnx_proto::AttributeProto>,
    ) -> onnx_proto::NodeProto {
        onnx_proto::NodeProto {
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: name.to_string(),
            op_type: op_type.to_string(),
            domain: String::new(),
            attribute: attrs,
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
    let path = temp.path().join("layer_norm_reciprocal.onnx");

    let graph = onnx_proto::GraphProto {
        node: vec![
            node(
                "mean1",
                "ReduceMean",
                &["input"],
                &["mean1_out"],
                vec![attr_ints("axes", &[-1])],
            ),
            node(
                "sub",
                "Sub",
                &["input", "mean1_out"],
                &["sub_out"],
                Vec::new(),
            ),
            node(
                "square",
                "Mul",
                &["sub_out", "sub_out"],
                &["square_out"],
                Vec::new(),
            ),
            node(
                "mean2",
                "ReduceMean",
                &["square_out"],
                &["mean2_out"],
                vec![attr_ints("axes", &[-1])],
            ),
            node(
                "eps",
                "Constant",
                &[],
                &["eps_out"],
                vec![attr_tensor("value", tensor_f32("eps", &[], &[1.0e-5]))],
            ),
            node(
                "add_eps",
                "Add",
                &["mean2_out", "eps_out"],
                &["var_eps_out"],
                Vec::new(),
            ),
            node("sqrt", "Sqrt", &["var_eps_out"], &["std_out"], Vec::new()),
            node("inv", "Reciprocal", &["std_out"], &["inv_out"], Vec::new()),
            node(
                "norm",
                "Mul",
                &["sub_out", "inv_out"],
                &["norm_out"],
                Vec::new(),
            ),
            node(
                "scale",
                "Mul",
                &["norm_out", "ny"],
                &["scaled_out"],
                Vec::new(),
            ),
            node(
                "shift",
                "Add",
                &["scaled_out", "beta"],
                &["output"],
                Vec::new(),
            ),
        ],
        name: "layer_norm_reciprocal".to_string(),
        initializer: vec![
            tensor_f32("ny", &[4], &[1.0, 1.0, 1.0, 1.0]),
            tensor_f32("beta", &[4], &[0.0, 0.0, 0.0, 0.0]),
        ],
        sparse_initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 4])],
        output: vec![tensor_value_info("output", &[1, 4])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };

    write_onnx_model(&path, graph);

    let model = load_onnx(&path).expect("Failed to load reciprocal LayerNorm model");
    assert!(
        model
            .network
            .layers
            .iter()
            .any(|l| l.layer_type == LayerType::LayerNorm),
        "Expected LayerNorm layer after reciprocal pattern fusion"
    );
}
