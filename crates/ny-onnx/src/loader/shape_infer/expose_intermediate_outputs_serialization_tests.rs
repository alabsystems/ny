// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::expose_intermediate_outputs;
use crate::onnx_proto::{
    tensor_shape_proto, tensor_shape_proto::Dimension, AttributeProto, GraphProto, ModelProto,
    NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TensorTypeProto, TypeProto,
    ValueInfoProto,
};
use prost::Message;

fn encode_model(graph: GraphProto) -> Vec<u8> {
    let model = ModelProto {
        ir_version: super::DEFAULT_IR_VERSION,
        opset_import: vec![OperatorSetIdProto {
            version: super::DEFAULT_OPSET_VERSION,
            domain: String::new(),
        }],
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        graph: Some(graph),
    };
    let mut buf = Vec::new();
    model.encode(&mut buf).expect("encode model");
    buf
}

fn value_info(name: &str) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: None,
    }
}

fn typed_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|&value| Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(value)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

#[test]
fn expose_outputs_skips_serialization_without_runtime_inputs() {
    let graph = GraphProto {
        node: vec![NodeProto {
            input: vec!["weight".to_string()],
            output: vec!["out".to_string()],
            name: String::new(),
            op_type: "Identity".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }],
        name: String::new(),
        initializer: vec![TensorProto {
            dims: vec![1],
            data_type: 1,
            name: "weight".to_string(),
            raw_data: Vec::new(),
            float_data: vec![1.0],
            ..Default::default()
        }],
        sparse_initializer: Vec::new(),
        input: vec![value_info("weight")],
        output: Vec::new(),
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };

    let bytes = encode_model(graph);
    let exposed = expose_intermediate_outputs(&bytes).expect("expose outputs");

    assert!(!exposed.has_runtime_inputs);
    assert!(exposed.bytes.is_empty());
}

#[test]
fn expose_outputs_serializes_with_runtime_inputs() {
    let graph = GraphProto {
        node: vec![NodeProto {
            input: vec!["input".to_string()],
            output: vec!["out".to_string()],
            name: String::new(),
            op_type: "Identity".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }],
        name: String::new(),
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: vec![value_info("input")],
        output: Vec::new(),
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };

    let bytes = encode_model(graph);
    let exposed = expose_intermediate_outputs(&bytes).expect("expose outputs");

    assert!(exposed.has_runtime_inputs);
    assert!(!exposed.bytes.is_empty());
}

#[test]
fn expose_outputs_sets_default_ir_and_opset() {
    let graph = GraphProto {
        node: vec![NodeProto {
            input: vec!["input".to_string()],
            output: vec!["out".to_string()],
            name: String::new(),
            op_type: "Identity".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }],
        name: String::new(),
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: vec![value_info("input")],
        output: Vec::new(),
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = ModelProto {
        ir_version: 0,
        opset_import: Vec::new(),
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        graph: Some(graph),
    };
    let bytes = model.encode_to_vec();
    let exposed = expose_intermediate_outputs(&bytes).expect("expose outputs");
    let decoded = ModelProto::decode(exposed.bytes.as_slice()).expect("decode exposed model");

    assert_eq!(decoded.ir_version, super::DEFAULT_IR_VERSION);
    assert!(
        decoded
            .opset_import
            .iter()
            .any(|opset| opset.version == super::DEFAULT_OPSET_VERSION),
        "expected default opset version"
    );
}

#[test]
fn expose_outputs_adds_default_domain_opset_when_missing() {
    let graph = GraphProto {
        node: vec![NodeProto {
            input: vec!["input".to_string()],
            output: vec!["out".to_string()],
            name: String::new(),
            op_type: "Identity".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }],
        name: String::new(),
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: vec![value_info("input")],
        output: Vec::new(),
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = ModelProto {
        ir_version: super::DEFAULT_IR_VERSION,
        opset_import: vec![OperatorSetIdProto {
            version: 1,
            domain: "custom".to_string(),
        }],
        producer_name: String::new(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 0,
        doc_string: String::new(),
        graph: Some(graph),
    };
    let bytes = model.encode_to_vec();
    let exposed = expose_intermediate_outputs(&bytes).expect("expose outputs");
    let decoded = ModelProto::decode(exposed.bytes.as_slice()).expect("decode exposed model");

    assert!(decoded
        .opset_import
        .iter()
        .any(|opset| opset.domain.is_empty()));
    assert!(decoded
        .opset_import
        .iter()
        .any(|opset| opset.domain == "custom" && opset.version == 1));
}

#[test]
fn expose_outputs_does_not_apply_standard_type_rules_to_custom_lookalikes() {
    let graph = GraphProto {
        node: vec![
            NodeProto {
                input: vec!["input".to_string()],
                output: vec!["standard_out".to_string()],
                name: "standard_identity".to_string(),
                op_type: "Identity".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
            NodeProto {
                input: vec!["input".to_string()],
                output: vec!["custom_out".to_string()],
                name: "custom_identity".to_string(),
                op_type: "Identity".to_string(),
                domain: "vendor.example".to_string(),
                attribute: Vec::new(),
            },
        ],
        input: vec![typed_value_info("input", &[2, 3])],
        ..Default::default()
    };

    let exposed = expose_intermediate_outputs(&encode_model(graph)).expect("expose outputs");
    let decoded = ModelProto::decode(exposed.bytes.as_slice()).expect("decode exposed model");
    let outputs = decoded.graph.expect("graph").output;
    let standard = outputs
        .iter()
        .find(|output| output.name == "standard_out")
        .expect("standard output");
    let custom = outputs
        .iter()
        .find(|output| output.name == "custom_out")
        .expect("custom output");

    assert!(standard.r#type.is_some());
    assert!(
        custom.r#type.is_none(),
        "a custom-domain Identity may not be shape preserving"
    );
}

#[test]
fn ort_transpose_normalization_ignores_custom_domain_lookalikes() {
    let perm = || AttributeProto {
        name: "perm".to_string(),
        ints: vec![0, 2, 1],
        ..Default::default()
    };
    let mut graph = GraphProto {
        input: vec![typed_value_info("input", &[48])],
        node: vec![
            NodeProto {
                input: vec!["input".to_string()],
                output: vec!["standard_out".to_string()],
                name: "standard_transpose".to_string(),
                op_type: "Transpose".to_string(),
                domain: String::new(),
                attribute: vec![perm()],
            },
            NodeProto {
                input: vec!["input".to_string()],
                output: vec!["custom_out".to_string()],
                name: "custom_transpose".to_string(),
                op_type: "Transpose".to_string(),
                domain: "vendor.example".to_string(),
                attribute: vec![perm()],
            },
        ],
        ..Default::default()
    };

    super::normalize_transpose_perms_for_ort(&mut graph);

    assert_eq!(graph.node[0].attribute[0].ints, vec![0]);
    assert_eq!(
        graph.node[1].attribute[0].ints,
        vec![0, 2, 1],
        "custom-domain attributes must remain owned by the custom operator"
    );
}
