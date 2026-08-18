// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

use crate::onnx_proto;
use crate::{
    load_onnx_bytes, load_onnx_bytes_with_config, CustomOpHandler, CustomOpRegistry, OnnxLoadConfig,
};
use ndarray::arr1;
use ny_core::{LayerType, NyError};
use ny_tensor::BoundedTensor;
use prost::Message;
use std::sync::{Arc, Mutex};

const CUSTOM_DOMAIN: &str = "com.acme";
const CUSTOM_OP_TYPE: &str = "AcmeRelu";

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

fn node(name: &str, inputs: &[&str], outputs: &[&str]) -> onnx_proto::NodeProto {
    onnx_proto::NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.to_string(),
        op_type: CUSTOM_OP_TYPE.to_string(),
        domain: CUSTOM_DOMAIN.to_string(),
        attribute: Vec::new(),
    }
}

fn build_custom_op_model_bytes() -> Vec<u8> {
    let graph = onnx_proto::GraphProto {
        node: vec![node("acme_relu", &["input"], &["output"])],
        name: "acme_custom_op".to_string(),
        initializer: Vec::new(),
        sparse_initializer: Vec::new(),
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("output", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };

    let model = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![
            onnx_proto::OperatorSetIdProto {
                domain: "ai.onnx".to_string(),
                version: 17,
            },
            onnx_proto::OperatorSetIdProto {
                domain: CUSTOM_DOMAIN.to_string(),
                version: 1,
            },
        ],
        producer_name: "ny-onnx-fixture".to_string(),
        producer_version: String::new(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
    };

    let mut buf = Vec::new();
    model.encode(&mut buf).expect("Failed to encode ONNX");
    buf
}

struct RecordingCustomOp {
    seen: Arc<Mutex<bool>>,
}

impl CustomOpHandler for RecordingCustomOp {
    fn try_convert(&self, node: &onnx_proto::NodeProto) -> Option<crate::LayerSpec> {
        if node.op_type != CUSTOM_OP_TYPE || node.domain != CUSTOM_DOMAIN {
            return None;
        }

        // Schema check: 1 input, 1 output.
        if node.input.len() != 1 || node.output.len() != 1 {
            return None;
        }

        *self.seen.lock().expect("lock seen flag") = true;

        Some(crate::LayerSpec {
            name: "acme_relu".to_string(),
            layer_type: LayerType::ReLU,
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            weights: None,
            attributes: Default::default(),
        })
    }

    fn try_convert_with_context(
        &self,
        node: &onnx_proto::NodeProto,
        opset_version: Option<i64>,
    ) -> Option<crate::LayerSpec> {
        // Verify opset_version is passed correctly when called with context.
        assert_eq!(opset_version, Some(1));
        self.try_convert(node)
    }

    fn supports(&self, op_type: &str) -> bool {
        op_type == CUSTOM_OP_TYPE
    }

    fn supports_with_context(
        &self,
        op_type: &str,
        domain: &str,
        _opset_version: Option<i64>,
    ) -> bool {
        op_type == CUSTOM_OP_TYPE && domain == CUSTOM_DOMAIN
    }
}

#[ntest::timeout(10000)]
#[test]
fn custom_op_is_resolved_via_registry_and_propagates() {
    let bytes = build_custom_op_model_bytes();
    let seen = Arc::new(Mutex::new(false));
    let handler = RecordingCustomOp {
        seen: Arc::clone(&seen),
    };
    let registry = CustomOpRegistry::from_handlers(vec![Arc::new(handler)]);
    let config = OnnxLoadConfig::new(registry);

    let model = load_onnx_bytes_with_config("custom_op_test", &bytes, &config)
        .expect("custom op load should succeed");

    assert!(
        *seen.lock().expect("lock seen flag"),
        "custom handler should be invoked"
    );
    assert_eq!(model.network.layers.len(), 1);
    assert_eq!(
        model.network.layers[0].layer_type,
        LayerType::ReLU,
        "CustomOpHandler lowers the ONNX node to a built-in LayerSpec; ny does not keep a runtime custom layer node"
    );

    let seq = model
        .to_propagate_network()
        .expect("convert to sequential network");

    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 1.0]).into_dyn(),
        arr1(&[1.0_f32, 1.0]).into_dyn(),
    )
    .expect("bounds should be valid");

    let output = seq
        .propagate_ibp(&input)
        .expect("IBP propagation should succeed");

    for (lower, upper) in output.lower().iter().zip(output.upper().iter()) {
        let (lower, upper): (&f32, &f32) = (lower, upper);
        assert!(lower.is_finite() && upper.is_finite());
        assert!(*lower >= 0.0);
        assert!(*upper <= 1.0 + 1e-6);
    }
}

#[ntest::timeout(10000)]
#[test]
fn custom_op_missing_registry_is_structured_error() {
    let bytes = build_custom_op_model_bytes();

    let err = load_onnx_bytes("custom_op_missing", &bytes)
        .expect_err("custom op should be rejected without registry");
    match err {
        NyError::UnsupportedConfiguration(message) => {
            assert!(message.contains("domain=\"com.acme\""));
            assert!(message.contains("op_type=\"AcmeRelu\""));
            assert!(message.contains("opset_version=1"));
            assert!(message.contains("CustomOpHandler"));
        }
        other => panic!("expected UnsupportedConfiguration, got {:?}", other),
    }
}
