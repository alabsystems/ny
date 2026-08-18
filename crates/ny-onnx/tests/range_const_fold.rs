// Copyright 2026 Andrew Yates
// Author: Andrew Yates <andrewyates.name@gmail.com>
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the ONNX `Range` op (and `Range` + `ConstantOfShape`
//! combined) flowing through the loader's constant-folding pass.
//!
//! `Range(start, limit, delta)` and `ConstantOfShape(shape, value)` are
//! shape-derived ops used by the cctsdb_yolo_2023 detection head. In a
//! verification graph with fixed input shapes their three scalar inputs are
//! constants, so the loader resolves each to an exact constant tensor at load
//! time. The fold is *exact* (no bounds approximation): the materialized value
//! `v` corresponds to the degenerate interval `[v, v]`, which is the soundest
//! possible IBP enclosure of a constant.

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

/// An INT64 (`data_type=7`) initializer encoded via little-endian `raw_data`.
/// This is how shape/index scalars (`start`/`limit`/`delta`, shape vectors)
/// appear in real detection graphs, and it populates the WeightStore integer
/// payload so the exact-integer fold path is exercised.
fn int64_initializer(name: &str, dims: &[i64], values: &[i64]) -> onnx_proto::TensorProto {
    let mut raw_data = Vec::new();
    for value in values {
        raw_data.extend_from_slice(&value.to_le_bytes());
    }
    onnx_proto::TensorProto {
        dims: dims.to_vec(),
        data_type: 7,
        name: name.to_string(),
        raw_data,
        float_data: Vec::new(),
        ..Default::default()
    }
}

fn float_initializer(name: &str, dims: &[i64], values: &[f32]) -> onnx_proto::TensorProto {
    onnx_proto::TensorProto {
        dims: dims.to_vec(),
        data_type: 1,
        name: name.to_string(),
        raw_data: Vec::new(),
        float_data: values.to_vec(),
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

fn model_from_graph(graph: onnx_proto::GraphProto) -> Vec<u8> {
    let model = onnx_proto::ModelProto {
        graph: Some(graph),
        opset_import: vec![onnx_proto::OperatorSetIdProto {
            version: 13,
            domain: String::new(),
        }],
        producer_name: "ny-onnx-test".to_string(),
        ..Default::default()
    };
    let mut bytes = Vec::new();
    model.encode(&mut bytes).expect("encode model");
    bytes
}

#[test]
fn test_range_integer_constant_fold() {
    // Range(0, 5, 1) -> [0, 1, 2, 3, 4], a typical index sequence.
    let graph = onnx_proto::GraphProto {
        name: "range_int_graph".to_string(),
        initializer: vec![
            int64_initializer("start", &[], &[0]),
            int64_initializer("limit", &[], &[5]),
            int64_initializer("delta", &[], &[1]),
        ],
        output: vec![tensor_value_info("out", &[5])],
        node: vec![
            node(
                "range",
                "Range",
                &["start", "limit", "delta"],
                &["seq"],
                Vec::new(),
            ),
            node("seq_shape", "Shape", &["seq"], &["shape"], Vec::new()),
            node(
                "materialize",
                "ConstantOfShape",
                &["shape"],
                &["out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let bytes = model_from_graph(graph);

    let model = load_onnx_bytes("range_int", &bytes).expect("load onnx bytes");
    let seq = model
        .weights
        .get("seq")
        .expect("Range output should be constant-folded");
    // Exact value == exact (degenerate) IBP interval bound: no approximation.
    assert_eq!(seq.shape(), &[5]);
    assert_eq!(seq.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_range_step_two_constant_fold() {
    // Range(1, 10, 2) -> [1, 3, 5, 7, 9]; ceil((10-1)/2) = 5 elements.
    let graph = onnx_proto::GraphProto {
        name: "range_step2_graph".to_string(),
        initializer: vec![
            int64_initializer("start", &[], &[1]),
            int64_initializer("limit", &[], &[10]),
            int64_initializer("delta", &[], &[2]),
        ],
        output: vec![tensor_value_info("out", &[5])],
        node: vec![
            node(
                "range",
                "Range",
                &["start", "limit", "delta"],
                &["seq"],
                Vec::new(),
            ),
            node("seq_shape", "Shape", &["seq"], &["shape"], Vec::new()),
            node(
                "materialize",
                "ConstantOfShape",
                &["shape"],
                &["out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let bytes = model_from_graph(graph);

    let model = load_onnx_bytes("range_step2", &bytes).expect("load onnx bytes");
    let seq = model
        .weights
        .get("seq")
        .expect("Range output should be constant-folded");
    assert_eq!(seq.as_slice().unwrap(), &[1.0, 3.0, 5.0, 7.0, 9.0]);
}

#[test]
fn test_range_float_constant_fold() {
    // Non-integral start/limit/delta exercise the f32 fold path.
    // Range(0.5, 2.0, 0.5) -> [0.5, 1.0, 1.5].
    let graph = onnx_proto::GraphProto {
        name: "range_float_graph".to_string(),
        initializer: vec![
            float_initializer("start", &[], &[0.5]),
            float_initializer("limit", &[], &[2.0]),
            float_initializer("delta", &[], &[0.5]),
        ],
        output: vec![tensor_value_info("seq", &[3])],
        node: vec![node(
            "range",
            "Range",
            &["start", "limit", "delta"],
            &["seq"],
            Vec::new(),
        )],
        ..Default::default()
    };
    let bytes = model_from_graph(graph);

    let model = load_onnx_bytes("range_float", &bytes).expect("load onnx bytes");
    let seq = model
        .weights
        .get("seq")
        .expect("float Range output should be constant-folded");
    assert_eq!(seq.as_slice().unwrap(), &[0.5, 1.0, 1.5]);
}

#[test]
fn test_constant_of_shape_then_range_combined() {
    // Mirrors the cctsdb_yolo_2023 detection-head pattern: ConstantOfShape
    // builds a fixed-shape constant while Range builds an index sequence; both
    // must fold out of the graph so verification sees only data-path ops.
    let graph = onnx_proto::GraphProto {
        name: "cos_plus_range_graph".to_string(),
        initializer: vec![
            // ConstantOfShape: shape [2, 3] filled with the default (0.0).
            int64_initializer("cos_shape", &[2], &[2, 3]),
            // Range(0, 6, 1) -> [0, 1, 2, 3, 4, 5].
            int64_initializer("start", &[], &[0]),
            int64_initializer("limit", &[], &[6]),
            int64_initializer("delta", &[], &[1]),
        ],
        output: vec![
            tensor_value_info("cos_out", &[2, 3]),
            tensor_value_info("seq_out", &[6]),
        ],
        node: vec![
            node(
                "cos",
                "ConstantOfShape",
                &["cos_shape"],
                &["cos_out"],
                Vec::new(),
            ),
            node(
                "range",
                "Range",
                &["start", "limit", "delta"],
                &["seq"],
                Vec::new(),
            ),
            node("seq_shape", "Shape", &["seq"], &["shape"], Vec::new()),
            node(
                "seq_materialize",
                "ConstantOfShape",
                &["shape"],
                &["seq_out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let bytes = model_from_graph(graph);

    let model = load_onnx_bytes("cos_plus_range", &bytes).expect("load onnx bytes");

    let cos_out = model
        .weights
        .get("cos_out")
        .expect("ConstantOfShape output should be constant-folded");
    assert_eq!(cos_out.shape(), &[2, 3]);
    assert!(cos_out.iter().all(|v| *v == 0.0));

    let seq = model
        .weights
        .get("seq")
        .expect("Range output should be constant-folded");
    assert_eq!(seq.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_range_feeds_reshape_target_shape() {
    // Detection-head style: Range produces an index/shape vector that is
    // consumed by a downstream constant op. Here a Range output is `Mul`-ed to
    // form a Reshape target shape, then a Reshape of a constant data tensor
    // folds end-to-end. This exercises "dynamic Reshape where operands are
    // shape-derived -> const-eval".
    //
    //   shape_seq = Range(2, 4, 1) = [2, 3]
    //   data (constant) is [6] -> Reshape(data, [2, 3]) -> [2, 3]
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let graph = onnx_proto::GraphProto {
        name: "range_reshape_graph".to_string(),
        initializer: vec![
            int64_initializer("rstart", &[], &[2]),
            int64_initializer("rlimit", &[], &[4]),
            int64_initializer("rdelta", &[], &[1]),
            float_initializer("data", &[6], &data),
        ],
        output: vec![tensor_value_info("reshaped", &[2, 3])],
        node: vec![
            node(
                "range",
                "Range",
                &["rstart", "rlimit", "rdelta"],
                &["target_shape"],
                Vec::new(),
            ),
            node(
                "reshape",
                "Reshape",
                &["data", "target_shape"],
                &["reshaped"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let bytes = model_from_graph(graph);

    let model = load_onnx_bytes("range_reshape", &bytes).expect("load onnx bytes");

    // The Range target shape [2, 3] must drive the Reshape so the data tensor
    // folds into the verification graph as a [2, 3] constant.
    let reshaped = model
        .weights
        .get("reshaped")
        .expect("Reshape driven by a folded Range shape should be constant-folded");
    assert_eq!(reshaped.shape(), &[2, 3]);
    assert_eq!(
        reshaped.as_slice().unwrap(),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn test_range_empty_sequence_constant_fold() {
    // limit not reached in the delta direction -> empty 1-D sequence. The fold
    // still resolves (it does not fall back to a data-dependent error) so a
    // degenerate detection branch loads cleanly.
    let graph = onnx_proto::GraphProto {
        name: "range_empty_graph".to_string(),
        initializer: vec![
            int64_initializer("start", &[], &[5]),
            int64_initializer("limit", &[], &[5]),
            int64_initializer("delta", &[], &[1]),
        ],
        output: vec![tensor_value_info("out", &[0])],
        node: vec![
            node(
                "range",
                "Range",
                &["start", "limit", "delta"],
                &["seq"],
                Vec::new(),
            ),
            node("seq_shape", "Shape", &["seq"], &["shape"], Vec::new()),
            node(
                "materialize",
                "ConstantOfShape",
                &["shape"],
                &["out"],
                Vec::new(),
            ),
        ],
        ..Default::default()
    };
    let bytes = model_from_graph(graph);

    let model = load_onnx_bytes("range_empty", &bytes).expect("load onnx bytes");
    let seq = model
        .weights
        .get("seq")
        .expect("empty Range output should still be constant-folded");
    assert_eq!(seq.len(), 0);
}
