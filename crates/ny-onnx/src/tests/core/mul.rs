// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::load_onnx_bytes;
use ndarray::arr1;
use ny_core::LayerType;
use ny_propagate::Layer as PropLayer;
use ny_tensor::BoundedTensor;
use prost::Message;

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_two_bounded_inputs_to_mul_binary() {
    let model = OnnxModel {
        network: Network {
            name: "test".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    };

    let layer = model.convert_layer(&spec).unwrap();
    match layer {
        PropLayer::MulBinary(_) => {}
        other => panic!(
            "Expected MulBinary layer for bounded Mul, got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_mul_binary_propagates_bounded_inputs() {
    let model = OnnxModel {
        network: Network {
            name: "mul-ibp".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    };

    let layer = model.convert_layer(&spec).unwrap();
    let input_a = BoundedTensor::new(arr1(&[1.0]).into_dyn(), arr1(&[2.0]).into_dyn()).unwrap();
    let input_b = BoundedTensor::new(arr1(&[3.0]).into_dyn(), arr1(&[5.0]).into_dyn()).unwrap();

    let output = match layer {
        PropLayer::MulBinary(mul) => mul.propagate_ibp_binary(&input_a, &input_b).unwrap(),
        other => panic!("Expected MulBinary layer, got {:?}", other.layer_type()),
    };

    assert!(
        (output.lower()[0] - 3.0).abs() < 1e-5,
        "Unexpected lower bound: {}",
        output.lower()[0]
    );
    assert!(
        (output.upper()[0] - 10.0).abs() < 1e-5,
        "Unexpected upper bound: {}",
        output.upper()[0]
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_missing_inputs_errors() {
    let model = OnnxModel {
        network: Network {
            name: "mul-missing-input".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["a".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    };

    let err = model
        .convert_layer(&spec)
        .expect_err("Expected Mul with missing inputs to error");
    let msg = format!("{err}");
    assert!(
        msg.contains("Model loading failed:"),
        "Missing ModelLoad prefix in error message: {msg}"
    );
    assert!(
        msg.contains("Mul mul requires exactly 2 inputs"),
        "Missing Mul missing-input error detail: {msg}"
    );
    assert!(
        msg.contains("got 1"),
        "Missing input count detail in error message: {msg}"
    );
    assert!(
        msg.contains("inputs=[a]"),
        "Missing input detail in error message: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_scale_attribute_to_mul_constant() {
    let model = OnnxModel {
        network: Network {
            name: "mul-scale".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["a".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::from([(
            "scale".to_string(),
            AttributeValue::Float(0.25),
        )]),
    };

    let layer = model.convert_layer(&spec).unwrap();
    match layer {
        PropLayer::MulConstant(layer) => {
            assert!(layer.constant().shape().is_empty());
            let value = *layer
                .constant()
                .iter()
                .next()
                .expect("MulConstant scalar should have a value");
            assert!((value - 0.25).abs() < 1e-6, "Unexpected scale: {value}");
        }
        other => panic!(
            "Expected MulConstant layer for scale attribute, got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_scale_attribute_missing_inputs_errors() {
    let model = OnnxModel {
        network: Network {
            name: "mul-scale-missing-input".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec![],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::from([(
            "scale".to_string(),
            AttributeValue::Float(0.25),
        )]),
    };

    let err = model
        .convert_layer(&spec)
        .expect_err("Expected Mul with scale attribute and no inputs to error");
    let msg = format!("{err}");
    assert!(
        msg.contains("Mul mul has no inputs"),
        "Unexpected error message: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_scale_attribute_with_extra_inputs_errors() {
    let model = OnnxModel {
        network: Network {
            name: "mul-scale-extra-inputs".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::from([(
            "scale".to_string(),
            AttributeValue::Float(0.5),
        )]),
    };

    let err = model
        .convert_layer(&spec)
        .expect_err("Expected Mul with scale attribute and extra inputs to error");
    let msg = format!("{err}");
    assert!(
        msg.contains("Mul mul with scale attribute requires 1 input"),
        "Unexpected error message: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_extra_inputs_errors() {
    let model = OnnxModel {
        network: Network {
            name: "mul-extra-inputs".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["a".to_string(), "b".to_string(), "c".to_string()],
        outputs: vec!["out".to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    };

    let err = model
        .convert_layer(&spec)
        .expect_err("Expected Mul with extra inputs to error");
    let msg = format!("{err}");
    assert!(
        msg.contains("Mul mul requires exactly 2 inputs"),
        "Unexpected error message: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_constant_input_to_mul_constant() {
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), arr1(&[2.0, 3.0]).into_dyn());

    let model = OnnxModel {
        network: Network {
            name: "mul-const".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights,
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["a".to_string(), "w".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    };

    let layer = model.convert_layer(&spec).unwrap();
    match layer {
        PropLayer::MulConstant(layer) => {
            assert_eq!(layer.constant().shape(), &[2]);
            let values: Vec<f32> = layer.constant().iter().copied().collect();
            assert!(
                (values[0] - 2.0).abs() < 1e-6,
                "Unexpected value: {}",
                values[0]
            );
            assert!(
                (values[1] - 3.0).abs() < 1e-6,
                "Unexpected value: {}",
                values[1]
            );
        }
        other => panic!(
            "Expected MulConstant layer for constant input, got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_constant_input_first_to_mul_constant() {
    let mut weights = WeightStore::new();
    weights.insert("w".to_string(), arr1(&[4.0, 5.0]).into_dyn());

    let model = OnnxModel {
        network: Network {
            name: "mul-const-first".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights,
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["w".to_string(), "a".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    };

    let layer = model.convert_layer(&spec).unwrap();
    match layer {
        PropLayer::MulConstant(layer) => {
            assert_eq!(layer.constant().shape(), &[2]);
            let values: Vec<f32> = layer.constant().iter().copied().collect();
            assert!(
                (values[0] - 4.0).abs() < 1e-6,
                "Unexpected value: {}",
                values[0]
            );
            assert!(
                (values[1] - 5.0).abs() < 1e-6,
                "Unexpected value: {}",
                values[1]
            );
        }
        other => panic!(
            "Expected MulConstant layer for constant input, got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_convert_mul_both_constants_returns_scalar_mul_constant() {
    let mut weights = WeightStore::new();
    weights.insert("w1".to_string(), arr1(&[2.0]).into_dyn());
    weights.insert("w2".to_string(), arr1(&[3.0]).into_dyn());

    let model = OnnxModel {
        network: Network {
            name: "mul-const-const".to_string(),
            inputs: vec![],
            outputs: vec![],
            layers: vec![],
            param_count: 0,
        },
        weights,
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let spec = LayerSpec {
        name: "mul".to_string(),
        layer_type: LayerType::Mul,
        inputs: vec!["w1".to_string(), "w2".to_string()],
        outputs: vec!["c".to_string()],
        weights: None,
        attributes: std::collections::HashMap::new(),
    };

    let layer = model.convert_layer(&spec).unwrap();
    match layer {
        PropLayer::MulConstant(layer) => {
            assert!(layer.constant().shape().is_empty());
            let value = *layer
                .constant()
                .iter()
                .next()
                .expect("MulConstant scalar should have a value");
            assert!((value - 6.0).abs() < 1e-6, "Unexpected scalar: {value}");
        }
        other => panic!(
            "Expected MulConstant layer for constant inputs, got {:?}",
            other.layer_type()
        ),
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_mul_binary_uses_two_inputs() {
    let model = OnnxModel {
        network: Network {
            name: "mul-graph".to_string(),
            inputs: vec![
                TensorSpec {
                    name: "a".to_string(),
                    shape: vec![1],
                    dtype: DataType::Float32,
                },
                TensorSpec {
                    name: "b".to_string(),
                    shape: vec![1],
                    dtype: DataType::Float32,
                },
            ],
            outputs: vec![TensorSpec {
                name: "c".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "mul".to_string(),
                layer_type: LayerType::Mul,
                inputs: vec!["a".to_string(), "b".to_string()],
                outputs: vec!["c".to_string()],
                weights: None,
                attributes: std::collections::HashMap::new(),
            }],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let graph = model
        .to_graph_network()
        .expect("Failed to convert MulBinary graph");
    let node = graph.node("mul").expect("Mul node missing from graph");

    match node.layer() {
        PropLayer::MulBinary(_) => {}
        other => panic!("Expected MulBinary layer, got {:?}", other.layer_type()),
    }
    assert_eq!(
        node.inputs().len(),
        2,
        "MulBinary graph node should have 2 inputs"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_mul_missing_inputs_errors() {
    let model = OnnxModel {
        network: Network {
            name: "mul-missing-input".to_string(),
            inputs: vec![TensorSpec {
                name: "a".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            }],
            outputs: vec![TensorSpec {
                name: "c".to_string(),
                shape: vec![1],
                dtype: DataType::Float32,
            }],
            layers: vec![LayerSpec {
                name: "mul".to_string(),
                layer_type: LayerType::Mul,
                inputs: vec!["a".to_string()],
                outputs: vec!["c".to_string()],
                weights: None,
                attributes: std::collections::HashMap::new(),
            }],
            param_count: 0,
        },
        weights: WeightStore::new(),
        tensor_producer: std::collections::HashMap::new(),
        constant_tensors: std::collections::HashSet::new(),
        tensor_shapes: std::collections::HashMap::new(),
        original_float32_initializers: std::collections::HashMap::new(),
        original_network_topology: None,
        opset_imports: std::collections::HashMap::new(),
    };

    let err = model
        .to_graph_network()
        .expect_err("Expected Mul with missing inputs to error");
    let msg = format!("{err}");
    assert!(
        msg.contains("Mul mul requires exactly 2 inputs"),
        "Unexpected error message: {msg}"
    );
    assert!(
        msg.contains("inputs=[a]"),
        "Missing input detail in error message: {msg}"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mul_binary_broadcastable_inputs_convert() {
    let data = mul_model_bytes("mul_broadcast", &[1, 11, 128], &[1, 11, 1], &[1, 11, 128]);
    let model = load_onnx_bytes("mul_broadcast", &data).expect("Failed to load model bytes");
    let network = model
        .to_propagate_network()
        .expect("Mul broadcast conversion should succeed");
    assert!(
        network
            .layers()
            .iter()
            .any(|layer| matches!(layer, PropLayer::MulBinary(_))),
        "Expected MulBinary layer for broadcastable inputs"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_graph_network_mul_binary_broadcastable_inputs() {
    let data = mul_model_bytes(
        "mul_broadcast_graph",
        &[1, 11, 128],
        &[1, 11, 1],
        &[1, 11, 128],
    );
    let model = load_onnx_bytes("mul_broadcast_graph", &data).expect("Failed to load model bytes");
    let graph = model
        .to_graph_network()
        .expect("Mul broadcast graph conversion should succeed");
    let node = graph.node("mul").expect("Mul node missing from graph");

    match node.layer() {
        PropLayer::MulBinary(_) => {}
        other => panic!(
            "Expected MulBinary layer for broadcastable inputs in graph, got {:?}",
            other.layer_type()
        ),
    }
    assert_eq!(
        node.inputs().len(),
        2,
        "MulBinary graph node should have 2 inputs"
    );
}

#[ntest::timeout(10000)]
#[test]
fn test_mul_binary_incompatible_inputs_error() {
    let data = mul_model_bytes("mul_incompatible", &[1, 2], &[3, 4], &[1, 2]);
    let model = load_onnx_bytes("mul_incompatible", &data).expect("Failed to load model bytes");
    let err = model
        .to_propagate_network()
        .expect_err("Expected incompatible broadcast to error");
    let msg = format!("{err}");
    assert!(
        msg.contains("broadcast-compatible"),
        "Unexpected error message: {msg}"
    );
}

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

fn mul_model_bytes(name: &str, shape_a: &[i64], shape_b: &[i64], shape_out: &[i64]) -> Vec<u8> {
    let graph = onnx_proto::GraphProto {
        node: vec![node("mul", "Mul", &["a", "b"], &["out"])],
        name: name.to_string(),
        initializer: Vec::new(),
        input: vec![
            tensor_value_info("a", shape_a),
            tensor_value_info("b", shape_b),
        ],
        output: vec![tensor_value_info("out", shape_out)],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };

    let model = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        producer_name: "ny-onnx-test".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    };
    model.encode_to_vec()
}
