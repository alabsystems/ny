// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use ny_onnx::{load_onnx_bytes, onnx_proto};
use prost::Message;

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
        attribute: attrs,
        ..Default::default()
    }
}

fn model_from_graph(graph: onnx_proto::GraphProto) -> onnx_proto::ModelProto {
    onnx_proto::ModelProto {
        graph: Some(graph),
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-test".to_string(),
        ..Default::default()
    }
}

#[test]
fn test_constant_of_shape_constant_fold() {
    let shape_tensor = tensor_f32("shape", &[2], &[2.0, 1.0]);
    let graph = onnx_proto::GraphProto {
        name: "const_of_shape_graph".to_string(),
        initializer: vec![shape_tensor],
        output: vec![tensor_value_info("out", &[2, 1])],
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let model = model_from_graph(graph);
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");

    let model = load_onnx_bytes("const_of_shape", &bytes).expect("load onnx bytes");
    let out = model
        .weights
        .get("out")
        .expect("ConstantOfShape output should be folded");
    assert_eq!(out.shape(), &[2, 1]);
    assert!(out.iter().all(|v| *v == 0.0));
}

#[test]
fn test_constant_of_shape_rejects_non_integer_shape() {
    let shape_tensor = tensor_f32("shape", &[2], &[1.0, 2.25]);
    let graph = onnx_proto::GraphProto {
        name: "const_of_shape_non_integer_shape".to_string(),
        initializer: vec![shape_tensor],
        output: vec![tensor_value_info("out", &[1, 2])],
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let model = model_from_graph(graph);
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");

    let model =
        load_onnx_bytes("const_of_shape_non_integer_shape", &bytes).expect("load onnx bytes");
    assert!(model.weights.get("out").is_none());
}

#[test]
fn test_constant_fold_skips_multi_output_nodes() {
    let shape_tensor = tensor_f32("shape", &[2], &[2.0, 1.0]);
    let graph = onnx_proto::GraphProto {
        name: "const_of_shape_multi_output".to_string(),
        initializer: vec![shape_tensor],
        output: vec![tensor_value_info("out", &[2, 1])],
        node: vec![node(
            "cos",
            "ConstantOfShape",
            &["shape"],
            &["out", "extra"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let model = model_from_graph(graph);
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");

    let model = load_onnx_bytes("const_of_shape_multi_output", &bytes).expect("load onnx bytes");
    assert!(model.weights.get("out").is_none());
    assert!(model.weights.get("extra").is_none());
}

/// Mul of two constant initializers with broadcast-compatible shapes folds to
/// the correctly-broadcast product. The const-folder routes Mul/Add/Sub/Div
/// through `broadcast_binop`, which implements NumPy broadcasting soundly (see
/// the `loader::const_fold::tests::elementwise::test_mul_constant_fold_broadcasts_shapes`
/// unit test). `[2,2] * [2]` broadcasts the row vector across both rows.
#[test]
fn test_mul_constant_fold_broadcasts_shapes() {
    let a_tensor = tensor_f32("a", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b_tensor = tensor_f32("b", &[2], &[5.0, 6.0]);
    let graph = onnx_proto::GraphProto {
        name: "mul_const_broadcast_fold".to_string(),
        initializer: vec![a_tensor, b_tensor],
        output: vec![tensor_value_info("out", &[2, 2])],
        node: vec![node("mul", "Mul", &["a", "b"], &["out"], Vec::new())],
        ..Default::default()
    };
    let model = model_from_graph(graph);
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");

    let model = load_onnx_bytes("mul_const_broadcast_fold", &bytes).expect("load onnx bytes");
    let out = model
        .weights
        .get("out")
        .expect("broadcast-compatible Mul of two constants should fold");
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(
        out.iter().copied().collect::<Vec<_>>(),
        vec![5.0, 12.0, 15.0, 24.0],
        "broadcast Mul must compute the NumPy-broadcast product"
    );
}
