// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for `Slice` on the ONNX batch axis (axis=0) in unbatched mode.
//!
//! `ny` strips the ONNX batch dimension before propagation, so an ONNX `Slice`
//! with `axis=0` (which targets the batch dim) needs special handling:
//!   * a no-op slice covering the whole batch extent lowers to an identity, and
//!   * a genuine slice of a constant (unbatched) data axis lowers to a real Slice.
//!
//! Both are sound; a genuine partial slice of a non-constant batch dim is rejected.

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

fn build_model(graph: onnx_proto::GraphProto) -> Vec<u8> {
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
    model.encode_to_vec()
}

/// A no-op `Slice` on the batch axis (axis=0, start=0, end=INT64_MAX) must lower
/// to an identity so the model loads and propagates unchanged. This is the
/// cctsdb_yolo_2023 detection-head pattern.
#[ntest::timeout(10000)]
#[test]
fn slice_batch_axis_noop_loads_and_propagates_identity() {
    // Graph: input [1, 3] --Slice(axis=0, [0:INT_MAX])--> out [1, 3]
    // Add a trailing Relu so the Slice is a genuine activation-path op, not a
    // bare passthrough that the loader might fold differently.
    let graph = onnx_proto::GraphProto {
        node: vec![
            onnx_proto::NodeProto {
                input: vec![
                    "input".to_string(),
                    "starts".to_string(),
                    "ends".to_string(),
                    "axes".to_string(),
                ],
                output: vec!["sliced".to_string()],
                name: "Slice_0".to_string(),
                op_type: "Slice".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
            onnx_proto::NodeProto {
                input: vec!["sliced".to_string()],
                output: vec!["out".to_string()],
                name: "Relu_0".to_string(),
                op_type: "Relu".to_string(),
                domain: String::new(),
                attribute: Vec::new(),
            },
        ],
        name: "slice_batch_noop".to_string(),
        initializer: vec![
            tensor_i64("starts", &[1], &[0]),
            tensor_i64("ends", &[1], &[i64::MAX]),
            tensor_i64("axes", &[1], &[0]),
        ],
        input: vec![tensor_value_info("input", &[1, 3])],
        output: vec![tensor_value_info("out", &[1, 3])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let bytes = build_model(graph);

    let model = load_onnx_bytes("slice_batch_noop.onnx", &bytes)
        .expect("no-op batch-axis Slice should load");

    let graph = model
        .to_graph_network()
        .expect("model with no-op batch Slice should convert to graph network");

    // Unbatched internal shape is [3]. Identity Slice must preserve shape + values.
    let input = BoundedTensor::new(
        arr1(&[-1.0_f32, 0.5, 2.0]).into_dyn(),
        arr1(&[1.0_f32, 0.5, 3.0]).into_dyn(),
    )
    .unwrap();
    let output = graph
        .propagate_ibp(&input)
        .expect("no-op batch Slice IBP should succeed");

    assert_eq!(
        output.lower().shape(),
        &[3],
        "shape preserved through identity+relu"
    );
    // Relu([-1,1]) = [0,1]; Relu([0.5,0.5]) = [0.5,0.5]; Relu([2,3]) = [2,3].
    assert_eq!(output.lower().as_slice().unwrap(), &[0.0, 0.5, 2.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[1.0, 0.5, 3.0]);

    // IBP soundness: concrete corner points must lie within the bounds.
    let seq = model
        .to_propagate_network()
        .expect("sequential conversion should succeed");
    for point in [[-1.0_f32, 0.5, 2.0], [1.0, 0.5, 3.0], [0.0, 0.5, 2.5]] {
        let concrete =
            BoundedTensor::new(arr1(&point).into_dyn(), arr1(&point).into_dyn()).unwrap();
        let cout = seq.propagate_ibp(&concrete).unwrap();
        for i in 0..cout.lower().len() {
            assert!(
                cout.lower()[[i]] >= output.lower()[[i]] - 1e-5
                    && cout.upper()[[i]] <= output.upper()[[i]] + 1e-5,
                "IBP bounds must contain concrete output at index {i}"
            );
        }
    }
}

/// A no-op batch-axis Slice expressed with an explicit finite end that still
/// covers the whole batch extent (end >= batch_size) also lowers to identity.
#[ntest::timeout(10000)]
#[test]
fn slice_batch_axis_full_finite_end_loads() {
    let graph = onnx_proto::GraphProto {
        node: vec![onnx_proto::NodeProto {
            input: vec![
                "input".to_string(),
                "starts".to_string(),
                "ends".to_string(),
                "axes".to_string(),
            ],
            output: vec!["out".to_string()],
            name: "Slice_0".to_string(),
            op_type: "Slice".to_string(),
            domain: String::new(),
            attribute: Vec::new(),
        }],
        name: "slice_batch_full_finite".to_string(),
        initializer: vec![
            tensor_i64("starts", &[1], &[0]),
            // batch size is 1; end=1 covers the full batch extent → no-op.
            tensor_i64("ends", &[1], &[1]),
            tensor_i64("axes", &[1], &[0]),
        ],
        input: vec![tensor_value_info("input", &[1, 2])],
        output: vec![tensor_value_info("out", &[1, 2])],
        #[cfg(feature = "onnx-value-info")]
        value_info: Vec::new(),
    };
    let bytes = build_model(graph);

    let model = load_onnx_bytes("slice_batch_full_finite.onnx", &bytes)
        .expect("full-coverage finite-end batch Slice should load");
    let graph = model
        .to_graph_network()
        .expect("should convert to graph network");

    let input = BoundedTensor::new(
        arr1(&[1.0_f32, -2.0]).into_dyn(),
        arr1(&[4.0_f32, 5.0]).into_dyn(),
    )
    .unwrap();
    let output = graph.propagate_ibp(&input).expect("IBP should succeed");
    assert_eq!(output.lower().as_slice().unwrap(), &[1.0, -2.0]);
    assert_eq!(output.upper().as_slice().unwrap(), &[4.0, 5.0]);
}
