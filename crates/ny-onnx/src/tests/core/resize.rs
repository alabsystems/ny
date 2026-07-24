// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::load_onnx_bytes;
use ndarray::ArrayD;
use ny_core::LayerType;
use ny_tensor::BoundedTensor;
use prost::Message;
use std::path::Path;

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

fn resize_attribute(name: &str, value: &str) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: onnx_proto::attribute_type::STRING,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_load_resize_end_to_end() {
    let graph = onnx_proto::GraphProto {
        node: vec![onnx_proto::NodeProto {
            input: vec!["input".to_string(), "roi".to_string(), "scales".to_string()],
            output: vec!["out".to_string()],
            name: "resize".to_string(),
            op_type: "Resize".to_string(),
            domain: String::new(),
            attribute: vec![
                resize_attribute("mode", "nearest"),
                resize_attribute("coordinate_transformation_mode", "asymmetric"),
                resize_attribute("nearest_mode", "floor"),
            ],
        }],
        name: "resize".to_string(),
        initializer: vec![
            tensor_f32("roi", &[0], &[]),
            tensor_f32("scales", &[4], &[1.0, 1.0, 2.0, 2.0]),
        ],
        input: vec![tensor_value_info("input", &[1, 1, 2, 2])],
        output: vec![tensor_value_info("out", &[1, 1, 4, 4])],
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

    let model = load_onnx_bytes("resize.onnx", &bytes).expect("Failed to load Resize ONNX");
    assert_eq!(model.network.layers.len(), 1);
    assert_eq!(model.network.layers[0].layer_type, LayerType::Resize);

    let graph = model
        .to_graph_network()
        .expect("Failed to convert Resize model to graph network");
    let input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap();
    let output = graph
        .propagate_ibp(&input)
        .expect("Resize graph IBP should succeed");

    assert_eq!(
        output.lower().as_slice().unwrap(),
        &[1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]
    );
    assert_eq!(output.lower(), output.upper());
}

#[ntest::timeout(10000)]
#[test]
fn test_cctsdb_yolo_no_longer_fails_at_resize() {
    let path = Path::new("benchmarks/vnncomp2025/benchmarks/cctsdb_yolo_2023/onnx/patch-1.onnx");
    let result = load_onnx(path);

    match result {
        Ok(model) => {
            assert!(
                model
                    .network
                    .layers
                    .iter()
                    .any(|layer| layer.layer_type == LayerType::Resize),
                "cctsdb_yolo should contain a Resize layer"
            );
        }
        Err(err) => {
            let msg = err.to_string();
            assert!(
                !msg.contains("Resize"),
                "cctsdb_yolo should not fail at Resize anymore: {msg}"
            );
        }
    }
}

#[ntest::timeout(10000)]
#[test]
fn test_collins_yolo_no_longer_fails_at_resize() {
    let path = Path::new(
        "benchmarks/vnncomp2025/benchmarks/collins_aerospace_benchmark/onnx/yolov5nano_LRelu_640.onnx",
    );
    let result = load_onnx(path);

    match result {
        Ok(model) => {
            let resize_count = model
                .network
                .layers
                .iter()
                .filter(|layer| layer.layer_type == LayerType::Resize)
                .count();
            assert!(
                resize_count >= 1,
                "collins_yolo should contain Resize layers"
            );
        }
        Err(err) => {
            let msg = err.to_string();
            assert!(
                !msg.contains("Resize"),
                "collins_yolo should not fail at Resize anymore: {msg}"
            );
        }
    }
}
