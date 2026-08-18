// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end regression for the narrow opset-7..=11 `Dropout` admission that
//! restores `vggnet16_2022`.
//!
//! `vgg16-7.onnx` is an opset-8 graph whose two `Dropout` nodes carry
//! `ratio=0.5`, one input and no mask output. At 5382ddb8 the loader refused
//! the whole model with "cannot be erased at opset 8", zeroing all 18
//! instances, so the first test here fails at that SHA. The paired
//! `expect_err` cases pin the arms that must stay closed: an opset that CAN
//! express training mode, and an observable mask output.

use super::super::parse_onnx_bytes;
use crate::loader::{
    BatchNormFoldingPolicy, CustomOpRegistry, ShapeInferBackend, ShapeInferencePolicy,
};
use crate::onnx_proto::{
    attribute_type, tensor_shape_proto, AttributeProto, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorProto, TensorShapeProto, TensorTypeProto, TypeProto, ValueInfoProto,
};
use prost::Message;

const FLOAT: i32 = 1;
const BOOL: i32 = 9;

fn float_info(name: &str, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type: FLOAT,
                shape: Some(TensorShapeProto {
                    dim: shape
                        .iter()
                        .map(|dim| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
                        })
                        .collect(),
                }),
            }),
        }),
    }
}

fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
    NodeProto {
        name: name.to_string(),
        op_type: op_type.to_string(),
        input: inputs.iter().map(|value| value.to_string()).collect(),
        output: outputs.iter().map(|value| value.to_string()).collect(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn float_attr(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: Some(value),
        r#type: attribute_type::FLOAT,
        ..Default::default()
    }
}

fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: Some(value),
        r#type: attribute_type::INT,
        ..Default::default()
    }
}

/// `Relu -> Dropout -> Relu`, the shape `vgg16-7.onnx` uses around each of its
/// two dropouts (`Gemm -> Relu -> Dropout`).
fn dropout_graph(dropout: NodeProto) -> GraphProto {
    GraphProto {
        name: "dropout_graph".to_string(),
        input: vec![float_info("x", &[1, 4])],
        output: vec![float_info("y", &[1, 4])],
        node: vec![
            node("relu_in", "Relu", &["x"], &["h"]),
            dropout,
            node("relu_out", "Relu", &["d"], &["y"]),
        ],
        ..Default::default()
    }
}

fn model_bytes(graph: GraphProto, opset: i64) -> Vec<u8> {
    ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            version: opset,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-dropout-admission-test".to_string(),
        graph: Some(graph),
        ..Default::default()
    }
    .encode_to_vec()
}

fn load(bytes: &[u8]) -> ny_core::Result<Vec<crate::LayerSpec>> {
    let registry = CustomOpRegistry::default();
    let (layers, ..) = parse_onnx_bytes(
        bytes,
        &registry,
        // Skip ORT so the test pins the loader's own admission and stays
        // deterministic without an ORT build.
        ShapeInferencePolicy::Skip,
        &ShapeInferBackend::InProcess,
        false,
        BatchNormFoldingPolicy::LegacyEnvironment,
        false,
    )?;
    Ok(layers)
}

#[test]
fn opset8_dropout_with_nonzero_ratio_loads_as_the_inference_identity() {
    let mut dropout = node("dropout", "Dropout", &["h"], &["d"]);
    dropout.attribute = vec![float_attr("ratio", 0.5)];
    let layers = load(&model_bytes(dropout_graph(dropout), 8))
        .expect("opset-8 Dropout has no authored training-mode control and is an identity");
    assert!(
        !layers.iter().any(|layer| layer.name == "dropout"),
        "the erased Dropout must not survive as a layer: {:?}",
        layers
            .iter()
            .map(|layer| (layer.name.clone(), layer.layer_type.clone()))
            .collect::<Vec<_>>()
    );

    // Opset 11 is the last opset before `training_mode` exists, and an absent
    // `ratio` attribute defaults to 0.5: same identity.
    let dropout = node("dropout", "Dropout", &["h"], &["d"]);
    load(&model_bytes(dropout_graph(dropout), 11)).expect("opset-11 Dropout is an identity too");
}

#[test]
fn dropout_admission_stops_at_the_opsets_that_can_express_training() {
    // Opset 6 DOES carry an authored mode control (`is_test`, default 0 =
    // training), so a non-zero ratio still fails closed.
    let mut dropout = node("dropout", "Dropout", &["h"], &["d"]);
    dropout.attribute = vec![float_attr("ratio", 0.5)];
    let error = load(&model_bytes(dropout_graph(dropout), 6))
        .expect_err("is_test defaults to training mode at opset 6");
    assert!(
        error.to_string().contains("is_test defaults to training"),
        "{error}"
    );

    // Opset 6 with is_test=1 is an authored inference node and still loads.
    let mut dropout = node("dropout", "Dropout", &["h"], &["d"]);
    dropout.attribute = vec![float_attr("ratio", 0.5), int_attr("is_test", 1)];
    load(&model_bytes(dropout_graph(dropout), 6)).expect("is_test=1 is authored inference mode");

    // Opset 12 CAN express training mode, so an explicit `training_mode`
    // operand is refused rather than assumed false.
    let dropout = node("dropout", "Dropout", &["h", "ratio", "training"], &["d"]);
    let mut graph = dropout_graph(dropout);
    graph.initializer = vec![
        TensorProto {
            name: "ratio".to_string(),
            dims: vec![],
            data_type: FLOAT,
            float_data: vec![0.5],
            ..Default::default()
        },
        TensorProto {
            name: "training".to_string(),
            dims: vec![],
            data_type: BOOL,
            int32_data: vec![0],
            ..Default::default()
        },
    ];
    let error = load(&model_bytes(graph, 12))
        .expect_err("an explicit training_mode operand is never assumed false");
    assert!(error.to_string().contains("training_mode"), "{error}");

    // A requested mask output is observable, so it is refused at every opset.
    let mut dropout = node("dropout", "Dropout", &["h"], &["d", "mask"]);
    dropout.attribute = vec![float_attr("ratio", 0.5)];
    let error =
        load(&model_bytes(dropout_graph(dropout), 8)).expect_err("the mask output is observable");
    assert!(error.to_string().contains("mask output"), "{error}");
}
