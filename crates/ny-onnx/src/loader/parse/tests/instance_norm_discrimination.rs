// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::super::parse_onnx_bytes;
use crate::loader::{CustomOpRegistry, ShapeInferBackend, ShapeInferencePolicy};
use crate::onnx_proto::{
    AttributeProto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto,
    TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use ny_core::LayerType;
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

fn tensor_f32(name: &str, shape: &[i64], values: &[f32]) -> TensorProto {
    TensorProto {
        dims: shape.to_vec(),
        data_type: 1,
        name: name.to_string(),
        float_data: values.to_vec(),
        ..Default::default()
    }
}

fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: outputs.iter().map(|value| value.to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
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

fn build_instance_norm_model_bytes() -> Vec<u8> {
    let mut mean1 = node("mean1", "ReduceMean", &["x"], &["mean1_out"]);
    mean1.attribute.push(attr_ints("axes", &[-1]));

    let sub = node("sub", "Sub", &["x", "mean1_out"], &["centered"]);
    let square = node("square", "Mul", &["centered", "centered"], &["squared"]);

    let mut mean2 = node("mean2", "ReduceMean", &["squared"], &["mean2_out"]);
    mean2.attribute.push(attr_ints("axes", &[-1]));

    let add_eps = node("add_eps", "Add", &["mean2_out", "eps"], &["var_eps"]);
    let sqrt = node("sqrt", "Sqrt", &["var_eps"], &["std"]);
    let reciprocal = node("reciprocal", "Reciprocal", &["std"], &["inv_std"]);
    let mul_norm = node("mul_norm", "Mul", &["centered", "inv_std"], &["norm"]);
    let mul_gamma = node("mul_gamma", "Mul", &["norm", "ny"], &["scaled"]);
    let add_beta = node("add_beta", "Add", &["scaled", "beta"], &["out"]);

    let graph = GraphProto {
        name: "instance_norm_discrimination_graph".to_string(),
        input: vec![tensor_value_info("x", &[1, 4, 3])],
        output: vec![tensor_value_info("out", &[1, 4, 3])],
        node: vec![
            mean1, sub, square, mean2, add_eps, sqrt, reciprocal, mul_norm, mul_gamma, add_beta,
        ],
        initializer: vec![
            tensor_f32("eps", &[], &[1e-5]),
            tensor_f32("ny", &[4], &[1.0, 1.0, 1.0, 1.0]),
            tensor_f32("beta", &[4], &[0.0, 0.0, 0.0, 0.0]),
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
fn test_parse_onnx_bytes_discriminates_instance_norm_before_conversion_3591() {
    let bytes = build_instance_norm_model_bytes();
    let registry = CustomOpRegistry::default();
    let (layers, _, _, _, _, _, tensor_shapes, _, _) = parse_onnx_bytes(
        &bytes,
        &registry,
        ShapeInferencePolicy::Ort,
        &ShapeInferBackend::InProcess,
        false,
        false,
    )
    .expect("parse onnx bytes");

    assert_eq!(
        tensor_shapes.get("x").map(Vec::as_slice),
        Some([1, 4, 3].as_slice()),
        "parser should retain the input tensor shape used for normalization fusion"
    );
    assert!(
        layers
            .iter()
            .any(|layer| layer.layer_type == LayerType::InstanceNorm),
        "decomposed normalization on [B, C, T] with ny [C] should fuse to InstanceNorm"
    );
}
