// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{load_onnx_bytes, load_onnx_bytes_with_config, OnnxLoadConfig, OnnxOptimizationFlag};
use approx::assert_relative_eq;
use ndarray::arr2;
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
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 1,
        name: name.to_string(),
        raw_data: Vec::new(),
        float_data: data.to_vec(),
        ..Default::default()
    }
}

fn node(name: &str, op_type: &str, inputs: &[&str], outputs: &[&str]) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        domain: String::new(),
        attribute: Vec::new(),
    }
}

fn build_merge_linear_model_bytes() -> Vec<u8> {
    let graph = onnx_proto::GraphProto {
        node: vec![
            node("mm1", "MatMul", &["input", "w1"], &["h1"]),
            node("mm2", "MatMul", &["h1", "w2"], &["h2"]),
            node("relu", "Relu", &["h2"], &["output"]),
        ],
        name: "merge_linear_fixture".to_string(),
        initializer: vec![
            tensor_f32("w1", &[2, 3], &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
            tensor_f32("w2", &[3, 1], &[2.0, -1.0, 0.5]),
        ],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 1])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let model = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "ny-onnx-fixture".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    };

    model.encode_to_vec()
}

#[test]
fn merge_linear_flag_reduces_loaded_layer_count() {
    let bytes = build_merge_linear_model_bytes();
    let plain = load_onnx_bytes("plain_merge_linear.onnx", &bytes).expect("plain load should work");
    let config =
        OnnxLoadConfig::default().with_optimization_flag(OnnxOptimizationFlag::MergeLinear);
    let merged = load_onnx_bytes_with_config("merged_merge_linear.onnx", &bytes, &config)
        .expect("merge_linear load should work");

    assert_eq!(plain.network.layers.len(), 3);
    assert_eq!(merged.network.layers.len(), 2);
    assert_eq!(
        plain.network.layers[0].layer_type,
        ny_core::LayerType::MatMul
    );
    assert_eq!(
        merged.network.layers[0].layer_type,
        ny_core::LayerType::Linear
    );
}

#[test]
fn merge_linear_flag_preserves_concrete_outputs() {
    let bytes = build_merge_linear_model_bytes();
    let plain = load_onnx_bytes("plain_merge_linear.onnx", &bytes).expect("plain load should work");
    let config =
        OnnxLoadConfig::default().with_optimization_flag(OnnxOptimizationFlag::MergeLinear);
    let merged = load_onnx_bytes_with_config("merged_merge_linear.onnx", &bytes, &config)
        .expect("merge_linear load should work");

    let plain_seq = plain
        .to_propagate_network()
        .expect("plain fixture should convert");
    let merged_seq = merged
        .to_propagate_network()
        .expect("merged fixture should convert");
    let input = ny_tensor::BoundedTensor::new(
        arr2(&[[1.0_f32, -1.0_f32]]).into_dyn(),
        arr2(&[[1.0_f32, -1.0_f32]]).into_dyn(),
    )
    .expect("concrete input should be valid");

    let plain_output = plain_seq
        .propagate_ibp(&input)
        .expect("plain propagation should succeed");
    let merged_output = merged_seq
        .propagate_ibp(&input)
        .expect("merged propagation should succeed");

    assert_relative_eq!(
        plain_output.lower()[[0, 0]],
        merged_output.lower()[[0, 0]],
        epsilon = 1e-6
    );
    assert_relative_eq!(
        plain_output.upper()[[0, 0]],
        merged_output.upper()[[0, 0]],
        epsilon = 1e-6
    );
}
