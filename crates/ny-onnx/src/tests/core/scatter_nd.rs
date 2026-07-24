// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::load_onnx_bytes;
use ndarray::{arr1, arr2};
use ny_core::LayerType;
use ny_tensor::BoundedTensor;
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

fn tensor_i64(name: &str, shape: &[i64], data: &[i64]) -> onnx_proto::TensorProto {
    assert_eq!(shape.iter().product::<i64>() as usize, data.len());
    onnx_proto::TensorProto {
        dims: shape.to_vec(),
        data_type: 7,
        name: name.to_string(),
        raw_data: data.iter().flat_map(|value| value.to_le_bytes()).collect(),
        float_data: Vec::new(),
        ..Default::default()
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_scatter_nd_end_to_end() {
    let graph = onnx_proto::GraphProto {
        node: vec![onnx_proto::NodeProto {
            input: vec![
                "data".to_string(),
                "indices".to_string(),
                "updates".to_string(),
            ],
            output: vec!["out".to_string()],
            name: "scatter".to_string(),
            op_type: "ScatterND".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }],
        name: "scatter_nd".to_string(),
        initializer: vec![
            tensor_f32("data", &[4], &[0.0, 0.0, 0.0, 0.0]),
            tensor_i64("indices", &[2, 1], &[1, 3]),
        ],
        input: vec![tensor_value_info("updates", &[2])],
        output: vec![tensor_value_info("out", &[4])],
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

    let model = load_onnx_bytes("scatter_nd.onnx", &bytes).expect("Failed to load ScatterND ONNX");
    assert_eq!(model.network.layers.len(), 1);
    assert_eq!(model.network.layers[0].layer_type, LayerType::ScatterND);

    let graph = model
        .to_graph_network()
        .expect("Failed to convert ScatterND model to graph network");
    let input = BoundedTensor::new(
        arr1(&[1.5_f32, -2.0]).into_dyn(),
        arr1(&[1.5_f32, -2.0]).into_dyn(),
    )
    .unwrap();
    let output = graph
        .propagate_ibp(&input)
        .expect("ScatterND graph IBP should succeed");

    assert_eq!(output.lower().as_slice().unwrap(), &[0.0, 1.5, 0.0, -2.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[0.0, 1.5, 0.0, -2.0]);
}

#[ntest::timeout(10000)]
#[test]
fn test_load_scatter_nd_dynamic_indices_end_to_end() {
    let graph = onnx_proto::GraphProto {
        node: vec![
            onnx_proto::NodeProto {
                input: vec!["input".to_string(), "updates_shape".to_string()],
                output: vec!["updates".to_string()],
                name: "reshape".to_string(),
                op_type: "Reshape".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
            onnx_proto::NodeProto {
                input: vec![
                    "data".to_string(),
                    "input".to_string(),
                    "updates".to_string(),
                ],
                output: vec!["out".to_string()],
                name: "scatter".to_string(),
                op_type: "ScatterND".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
        ],
        name: "scatter_nd_dynamic".to_string(),
        initializer: vec![
            tensor_f32("data", &[4], &[0.0, 0.0, 0.0, 0.0]),
            tensor_i64("updates_shape", &[1], &[2]),
        ],
        input: vec![tensor_value_info("input", &[2, 1])],
        output: vec![tensor_value_info("out", &[4])],
        #[cfg(feature = "onnx-value-info")]
        value_info: vec![tensor_value_info("updates", &[2])],
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

    let model = load_onnx_bytes("scatter_nd_dynamic.onnx", &bytes)
        .expect("Failed to load dynamic ScatterND");
    let graph = model
        .to_graph_network()
        .expect("Failed to convert dynamic ScatterND model to graph network");
    let input = BoundedTensor::new(
        arr2(&[[0.0_f32], [1.0]]).into_dyn(),
        arr2(&[[3.0_f32], [3.0]]).into_dyn(),
    )
    .unwrap();
    let output = graph
        .propagate_ibp(&input)
        .expect("Dynamic ScatterND graph IBP should succeed");

    assert_eq!(output.lower().as_slice().unwrap(), &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[3.0, 3.0, 3.0, 3.0]);
}
