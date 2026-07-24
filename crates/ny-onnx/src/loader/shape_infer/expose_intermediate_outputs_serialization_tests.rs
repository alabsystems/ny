// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::expose_intermediate_outputs;
use crate::onnx_proto::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, ValueInfoProto,
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
