// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! In-memory ONNX protobuf builders shared by the front-end parity tests.
//!
//! These build the SAME bytes that are handed to `load_onnx_bytes` and to the
//! trusted forward (`diff::run_inference_bytes`), so a parity assertion is
//! about one model, not two hand-kept-in-sync fixtures.

use crate::onnx_proto;
use prost::Message;

pub(super) const FLOAT: i32 = 1;
pub(super) const INT64: i32 = 7;

pub(super) fn value_info(name: &str, shape: &[i64]) -> onnx_proto::ValueInfoProto {
    let dim = shape
        .iter()
        .map(|dimension| onnx_proto::tensor_shape_proto::Dimension {
            value: Some(onnx_proto::tensor_shape_proto::dimension::Value::DimValue(
                *dimension,
            )),
        })
        .collect();
    onnx_proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx_proto::TypeProto {
            tensor_type: Some(onnx_proto::TensorTypeProto {
                elem_type: FLOAT,
                shape: Some(onnx_proto::TensorShapeProto { dim }),
            }),
        }),
    }
}

pub(super) fn f32_initializer(name: &str, shape: &[i64], data: &[f32]) -> onnx_proto::TensorProto {
    assert_eq!(shape.iter().product::<i64>() as usize, data.len());
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: FLOAT,
        name: name.to_string(),
        raw_data: Vec::new(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

pub(super) fn i64_initializer(name: &str, shape: &[i64], data: &[i64]) -> onnx_proto::TensorProto {
    assert_eq!(shape.iter().product::<i64>() as usize, data.len());
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: INT64,
        name: name.to_string(),
        raw_data: data.iter().flat_map(|value| value.to_le_bytes()).collect(),
        float_data: Vec::new(),
        ..Default::default()
    }
}

pub(super) fn int_attr(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        i: Some(value),
        r#type: onnx_proto::attribute_type::INT,
        ..Default::default()
    }
}

pub(super) fn string_attr(name: &str, value: &str) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        s: Some(value.as_bytes().to_vec()),
        r#type: onnx_proto::attribute_type::STRING,
        ..Default::default()
    }
}

pub(super) fn node(
    name: &str,
    op_type: &str,
    inputs: &[&str],
    outputs: &[&str],
    attribute: Vec<onnx_proto::AttributeProto>,
) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        name: name.to_string(),
        op_type: op_type.to_string(),
        input: inputs.iter().map(|value| (*value).to_string()).collect(),
        output: outputs.iter().map(|value| (*value).to_string()).collect(),
        domain: String::new(),
        attribute,
    }
}

/// Serialize one single-graph ONNX model.
pub(super) fn model_bytes(
    name: &str,
    opset: i64,
    nodes: Vec<onnx_proto::NodeProto>,
    initializer: Vec<onnx_proto::TensorProto>,
    inputs: Vec<onnx_proto::ValueInfoProto>,
    outputs: Vec<onnx_proto::ValueInfoProto>,
) -> Vec<u8> {
    let graph = onnx_proto::GraphProto {
        node: nodes,
        name: name.to_string(),
        initializer,
        sparse_initializer: Vec::new(),
        input: inputs,
        output: outputs,
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: opset,
        }],
        producer_name: "ny-onnx-fixture".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    }
    .encode_to_vec()
}

/// Run the trusted forward over the SAME bytes ny loaded, returning the flat
/// output values.
///
/// This is the concrete-evaluation oracle every parity test in this directory
/// checks its bounds against: an independent implementation of the ONNX
/// operator, not a re-statement of ny's own arithmetic.
#[cfg(feature = "ort")]
pub(super) fn trusted_forward(model_bytes: &[u8], input: &ndarray::ArrayD<f32>) -> Vec<f32> {
    let outputs = crate::diff::run_inference_bytes(model_bytes, input)
        .expect("the trusted runtime must evaluate the fixture");
    assert_eq!(outputs.len(), 1, "fixtures declare exactly one output");
    outputs[0].iter().copied().collect()
}
