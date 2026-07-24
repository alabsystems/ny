// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::parse_onnx_bytes;
use crate::loader::{CustomOpRegistry, ShapeInferBackend, ShapeInferencePolicy};
use crate::onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto,
    TensorTypeProto, TypeProto, ValueInfoProto,
};
use prost::Message;

fn tensor_value_info(name: &str, shape: &[i64]) -> ValueInfoProto {
    let dims = shape
        .iter()
        .map(|dim| crate::onnx_proto::tensor_shape_proto::Dimension {
            value: Some(crate::onnx_proto::tensor_shape_proto::dimension::Value::DimValue(*dim)),
        })
        .collect();
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: 1,
                shape: Some(TensorShapeProto { dim: dims }),
            }),
        }),
    }
}

fn node(op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: outputs.iter().map(|value| value.to_string()).collect(),
        name: String::new(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: crate::onnx_proto::attribute_type::INT,
        ..Default::default()
    }
}

fn attr_ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints: values.to_vec(),
        r#type: crate::onnx_proto::attribute_type::INTS,
        ..Default::default()
    }
}

fn build_shape_infer_model_bytes() -> Vec<u8> {
    let graph = GraphProto {
        name: "shape_infer_graph".to_string(),
        input: vec![
            tensor_value_info("a", &[1, 3]),
            tensor_value_info("b", &[1, 3]),
        ],
        output: vec![tensor_value_info("out", &[1, 3])],
        node: vec![
            node("Add", &["a", "b"], &["sum"]),
            node("Relu", &["sum"], &["out"]),
        ],
        ..Default::default()
    };
    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-parse-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    };
    model.encode_to_vec()
}

#[test]
fn test_parse_onnx_bytes_includes_ort_inferred_shapes() {
    let bytes = build_shape_infer_model_bytes();
    let registry = CustomOpRegistry::default();
    let (_, _, _, _, _, _, tensor_shapes, _, _) = parse_onnx_bytes(
        &bytes,
        &registry,
        ShapeInferencePolicy::Ort,
        &ShapeInferBackend::InProcess,
        false,
        false,
    )
    .expect("parse onnx bytes");

    assert_eq!(
        tensor_shapes.get("sum").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
}

#[test]
fn test_parse_onnx_bytes_skip_shape_inference_omits_intermediate_shapes() {
    let bytes = build_shape_infer_model_bytes();
    let registry = CustomOpRegistry::default();
    let (_, _, _, _, _, _, tensor_shapes, _, _) = parse_onnx_bytes(
        &bytes,
        &registry,
        ShapeInferencePolicy::Skip,
        &ShapeInferBackend::InProcess,
        false,
        false,
    )
    .expect("parse onnx bytes");

    assert!(
        !tensor_shapes.contains_key("sum"),
        "skip mode should avoid ORT-only intermediate shape discovery"
    );
    assert_eq!(
        tensor_shapes.get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
}

#[test]
fn test_parse_onnx_bytes_retains_scalar_output_shape() {
    let mut reduce_sum = node("ReduceSum", &["input"], &["output"]);
    reduce_sum.attribute = vec![attr_ints("axes", &[0, 1]), attr_int("keepdims", 0)];
    let graph = GraphProto {
        name: "scalar_output_shape_graph".to_string(),
        input: vec![tensor_value_info("input", &[1, 1])],
        output: vec![tensor_value_info("output", &[])],
        node: vec![reduce_sum],
        ..Default::default()
    };
    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-scalar-shape-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    let registry = CustomOpRegistry::default();
    let (_, _, _, _, _, _, tensor_shapes, _, _) = parse_onnx_bytes(
        &model.encode_to_vec(),
        &registry,
        ShapeInferencePolicy::Ort,
        &ShapeInferBackend::InProcess,
        false,
        false,
    )
    .expect("parse onnx bytes");

    assert_eq!(
        tensor_shapes.get("output").map(Vec::as_slice),
        Some([].as_slice())
    );
}

#[test]
fn test_parse_onnx_bytes_corrects_gemm_shape_conflict() {
    use crate::onnx_proto::{AttributeProto, TensorProto};

    let weight = TensorProto {
        dims: vec![98, 30],
        data_type: 1,
        name: "w".to_string(),
        float_data: vec![0.0; 98 * 30],
        ..Default::default()
    };
    let bias = TensorProto {
        dims: vec![98],
        data_type: 1,
        name: "b".to_string(),
        float_data: vec![0.0; 98],
        ..Default::default()
    };

    let mut gemm_node = NodeProto {
        op_type: "Gemm".to_string(),
        input: vec!["x".to_string(), "w".to_string(), "b".to_string()],
        output: vec!["y".to_string()],
        name: "gemm0".to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    };
    gemm_node.attribute.push(AttributeProto {
        name: "transB".to_string(),
        i: 1,
        ..Default::default()
    });

    let graph = GraphProto {
        name: "shape_conflict_graph".to_string(),
        input: vec![tensor_value_info("x", &[1, 30])],
        output: vec![tensor_value_info("y", &[1, 30])],
        node: vec![gemm_node],
        initializer: vec![weight, bias],
        ..Default::default()
    };

    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-shape-conflict-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    let bytes = model.encode_to_vec();
    let registry = CustomOpRegistry::default();
    let (_, _, _, _, _, _, tensor_shapes, _, _) = parse_onnx_bytes(
        &bytes,
        &registry,
        ShapeInferencePolicy::Ort,
        &ShapeInferBackend::InProcess,
        false,
        false,
    )
    .expect("parse onnx bytes");

    assert_eq!(
        tensor_shapes.get("y").map(Vec::as_slice),
        Some([1, 98].as_slice()),
        "Gemm output shape should be corrected from proto's [1,30] to weight-derived [1,98]"
    );
}
