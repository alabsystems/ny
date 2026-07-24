// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end regression tests for the batch-squeeze reduction-axis
//! miscompile (#pensieve ReduceSum no-op).
//!
//! ny's internal runtime tensors are NOT rank-uniform: Flatten and rank-2
//! Gemm outputs RETAIN a leading size-1 axis (`[1, n]`), while Split/Slice
//! outputs are batch-stripped. The legacy blanket `axis >= 1 → axis - 1`
//! conversion turned the pensieve `ReduceSum(axes=[1])` into a size-1-axis
//! NO-OP on the runtime `[1, n]` tensor: the graph forward computed
//! `w = p / ReduceSum(p) = p / p = 1` — bounding a DIFFERENT function than
//! the ONNX semantics (ORT). The fix stores reduction axes TRAILING-RELATIVE
//! (negative), which selects the same semantic dim under both layouts.

use super::*;
use crate::load_onnx_bytes;
use ndarray::arr1;
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

fn build_model(graph: onnx_proto::GraphProto) -> Vec<u8> {
    let model = onnx_proto::ModelProto {
        ir_version: 9,
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            domain: String::new(),
            version: 12,
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

fn attr_ints(name: &str, values: &[i64]) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: onnx_proto::attribute_proto::AttributeType::Ints as i32,
        ints: values.to_vec(),
        ..Default::default()
    }
}

fn attr_int(name: &str, value: i64) -> onnx_proto::AttributeProto {
    onnx_proto::AttributeProto {
        name: name.to_string(),
        r#type: onnx_proto::attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

/// The pensieve head shape, minimized: `x → Flatten(axis=1) → ReduceSum(axes=[1])
/// → Div(flat, sum)`. The Flatten output is a RETAINED-rank runtime `[1, 6]`
/// tensor, so the legacy internal `axes=[0]` reduced the size-1 leading axis (a
/// no-op) and the graph forward returned `x/x = 1` everywhere. The forward must
/// equal the true ONNX reduction: `y_i = x_i / Σ_j x_j`.
#[ntest::timeout(10000)]
#[test]
fn reduce_sum_axis1_on_retained_rank_runtime_tensor_matches_onnx_semantics() {
    let graph = onnx_proto::GraphProto {
        node: vec![
            onnx_proto::NodeProto {
                input: vec!["input".to_string()],
                output: vec!["flat".to_string()],
                name: "Flatten_0".to_string(),
                op_type: "Flatten".to_string(),
                domain: String::new(),
                attribute: vec![attr_int("axis", 1)],
            },
            onnx_proto::NodeProto {
                input: vec!["flat".to_string()],
                output: vec!["sum".to_string()],
                name: "ReduceSum_0".to_string(),
                op_type: "ReduceSum".to_string(),
                domain: String::new(),
                attribute: vec![attr_ints("axes", &[1]), attr_int("keepdims", 1)],
            },
            onnx_proto::NodeProto {
                input: vec!["flat".to_string(), "sum".to_string()],
                output: vec!["out".to_string()],
                name: "Div_0".to_string(),
                op_type: "Div".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
        ],
        name: "reduce_batch_axis_frac".to_string(),
        initializer: vec![],
        input: vec![tensor_value_info("input", &[1, 6])],
        output: vec![tensor_value_info("out", &[1, 6])],
        #[cfg(feature = "onnx-value-info")]
        value_info: vec![
            tensor_value_info("flat", &[1, 6]),
            tensor_value_info("sum", &[1, 1]),
        ],
    };
    let bytes = build_model(graph);

    let model = load_onnx_bytes("reduce_batch_axis_frac.onnx", &bytes)
        .expect("ReduceSum(axes=[1]) model should load");
    let graph = model
        .to_graph_network()
        .expect("model should convert to graph network");

    // Concrete forward at x = [1..6]: ONNX semantics y_i = x_i / 21.
    let x = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = BoundedTensor::concrete(arr1(&x).into_dyn()).expect("concrete input");
    let out = graph
        .propagate_concrete_point(&input, None, None)
        .expect("concrete forward should succeed");
    let y = out.center();
    let y: Vec<f32> = y.iter().copied().collect();
    assert_eq!(y.len(), 6, "output length");

    let total: f32 = x.iter().sum();
    for (i, (&yi, &xi)) in y.iter().zip(x.iter()).enumerate() {
        let expected = xi / total;
        assert!(
            (yi - expected).abs() <= 1e-6,
            "index {i}: NY forward {yi} != ONNX semantics {expected} \
             (the defective batch-squeeze conversion returned the p/p = 1 no-op)"
        );
    }
}

/// Same head via the opset-13 axes-as-input encoding (`ReduceSum` axes come
/// from a constant input tensor instead of an attribute).
#[ntest::timeout(10000)]
#[test]
fn reduce_sum_axes_input_tensor_on_retained_rank_runtime_tensor() {
    let axes_init = onnx_proto::TensorProto {
        dims: vec![1],
        data_type: 7, // INT64
        name: "axes_const".to_string(),
        raw_data: 1_i64.to_le_bytes().to_vec(),
        float_data: Vec::new(),
        ..Default::default()
    };
    let graph = onnx_proto::GraphProto {
        node: vec![
            onnx_proto::NodeProto {
                input: vec!["input".to_string()],
                output: vec!["flat".to_string()],
                name: "Flatten_0".to_string(),
                op_type: "Flatten".to_string(),
                domain: String::new(),
                attribute: vec![attr_int("axis", 1)],
            },
            onnx_proto::NodeProto {
                input: vec!["flat".to_string(), "axes_const".to_string()],
                output: vec!["sum".to_string()],
                name: "ReduceSum_0".to_string(),
                op_type: "ReduceSum".to_string(),
                domain: String::new(),
                attribute: vec![attr_int("keepdims", 1)],
            },
            onnx_proto::NodeProto {
                input: vec!["flat".to_string(), "sum".to_string()],
                output: vec!["out".to_string()],
                name: "Div_0".to_string(),
                op_type: "Div".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
        ],
        name: "reduce_batch_axis_frac_opset13".to_string(),
        initializer: vec![axes_init],
        input: vec![tensor_value_info("input", &[1, 4])],
        output: vec![tensor_value_info("out", &[1, 4])],
        #[cfg(feature = "onnx-value-info")]
        value_info: vec![
            tensor_value_info("flat", &[1, 4]),
            tensor_value_info("sum", &[1, 1]),
        ],
    };
    let bytes = build_model(graph);

    let model = load_onnx_bytes("reduce_batch_axis_frac_opset13.onnx", &bytes)
        .expect("opset-13 ReduceSum model should load");
    let graph = model
        .to_graph_network()
        .expect("model should convert to graph network");

    let x = [2.0_f32, 4.0, 6.0, 8.0];
    let input = BoundedTensor::concrete(arr1(&x).into_dyn()).expect("concrete input");
    let out = graph
        .propagate_concrete_point(&input, None, None)
        .expect("concrete forward should succeed");
    let y: Vec<f32> = out.center().iter().copied().collect();
    let total: f32 = x.iter().sum();
    for (i, (&yi, &xi)) in y.iter().zip(x.iter()).enumerate() {
        let expected = xi / total;
        assert!(
            (yi - expected).abs() <= 1e-6,
            "index {i}: NY forward {yi} != ONNX semantics {expected}"
        );
    }
}
