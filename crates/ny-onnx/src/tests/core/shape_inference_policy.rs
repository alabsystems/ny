// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{load_onnx_bytes_with_config, OnnxLoadConfig, ShapeInferencePolicy};
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

fn build_shape_infer_model_bytes() -> Vec<u8> {
    let graph = onnx_proto::GraphProto {
        node: vec![
            onnx_proto::NodeProto {
                input: vec!["a".to_string(), "b".to_string()],
                output: vec!["sum".to_string()],
                name: "sum_node".to_string(),
                op_type: "Add".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
            onnx_proto::NodeProto {
                input: vec!["sum".to_string()],
                output: vec!["out".to_string()],
                name: "relu_node".to_string(),
                op_type: "Relu".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
        ],
        name: "shape_infer_graph".to_string(),
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: vec![
            tensor_value_info("a", &[1, 3]),
            tensor_value_info("b", &[1, 3]),
        ],
        output: vec![tensor_value_info("out", &[1, 3])],
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

#[ntest::timeout(10000)]
#[test]
fn test_load_onnx_bytes_skip_shape_inference_omits_intermediate_shapes() {
    let bytes = build_shape_infer_model_bytes();
    let config = OnnxLoadConfig::default().with_shape_inference_policy(ShapeInferencePolicy::Skip);

    let loaded = load_onnx_bytes_with_config("skip_shape.onnx", &bytes, &config)
        .expect("load onnx bytes with skip shape inference");

    assert!(
        !loaded.tensor_shapes().contains_key("sum"),
        "skip mode should avoid ORT-only intermediate shape discovery"
    );
    assert_eq!(
        loaded.tensor_shapes().get("out").map(Vec::as_slice),
        Some([1, 3].as_slice())
    );
}
